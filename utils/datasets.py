import glob
import math
import os
import random
import shutil
import time
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.utils import xyxy2xywh, xywh2xyxy

help_url = 'https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data'
img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']
vid_formats = ['.mov', '.avi', '.mp4']

class LoadImages:  # for inference
    def __init__(self, path, img_size=416, half=False):
        path = str(Path(path))  # os-agnostic
        files = []
        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, '*.*')))
        elif os.path.isfile(path):
            files = [path]

        images = [x for x in files if os.path.splitext(x)[-1].lower() in img_formats]
        nI= len(images)

        self.img_size = img_size

        self.files = images
        self.nF = nI  # number of files
        self.mode = 'images'
        self.half = half  # half precision fp16 images
        assert self.nF > 0, 'No images or videos found in ' + path

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nF:
            raise StopIteration
        path = self.files[self.count]

        # Read image
        self.count += 1
        img0 = cv2.imread(path)  # BGR
        assert img0 is not None, 'Image Not Found ' + path
        print('image %g/%g %s: ' % (self.count, self.nF, path), end='')

        # Padded resize
        img = letterbox(img0, new_shape=self.img_size)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img, dtype=np.float16 if self.half else np.float32)  # uint8 to fp16/fp32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        # cv2.imwrite(path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return path, img, img0

    def __len__(self):
        return self.nF  # number of files

class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, path, img_size=416, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_labels=False, cache_images=False, single_cls=False):
        path = str(Path(path))  # os-agnostic
        assert os.path.isfile(path), 'File not found %s. See %s' % (path, help_url)
        with open(path, 'r') as f:
            self.img_files = [x for x in f.read().splitlines() if os.path.splitext(x)[-1].lower() in img_formats]

        n = len(self.img_files)
        assert n > 0, 'No images found in %s. See %s' % (path, help_url)
        bi = np.floor(n / batch_size).astype(np.int)  # batch index

        self.n = n
        self.batch = bi  # batch index of image
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect

        # Define labels
        self.label_files = [x.replace('images', 'labels').replace(os.path.splitext(x)[-1], '.txt')
                            for x in self.img_files]

        # Preload labels (required for weighted CE training)
        self.imgs = [None] * n
        self.labels = [None] * n

        if cache_labels or image_weights:  # cache labels for faster training
            self.labels = [np.zeros((0, 5))] * n

            pbar = tqdm(self.label_files, desc='Caching labels')
            nm, nf, ne, ns, nd = 0, 0, 0, 0, 0  # number missing, found, empty, datasubset, duplicate

            for i, file in enumerate(pbar):
                try:
                    with open(file, 'r') as f:
                        l = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
                except:
                    nm += 1  # print('missing labels for image %s' % self.img_files[i])  # file missing
                    continue

                if l.shape[0]:
                    assert l.shape[1] == 5, '> 5 label columns: %s' % file
                    assert (l >= 0).all(), 'negative labels: %s' % file
                    assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels: %s' % file
                    if np.unique(l, axis=0).shape[0] < l.shape[0]:  # duplicate rows
                        nd += 1  # print('WARNING: duplicate rows in %s' % self.label_files[i])  # duplicate rows
                    self.labels[i] = l
                    nf += 1  # file found
                else:
                    ne += 1  # print('empty labels for image %s' % self.img_files[i])  # file empty
                pbar.desc = 'Caching labels (%g found, %g missing, %g empty, %g duplicate, for %g images)' % (nf, nm, ne, nd, n)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):

        img_path = self.img_files[index]
        label_path = self.label_files[index]

        hyp = self.hyp
        mosaic = False  # load 4 images at a time into a mosaic (only during training)

        # Load image
        img = self.imgs[index]
        if img is None:  # not cached
            img_path = self.img_files[index]
            img = cv2.imread(img_path)  # BGR
            assert img is not None, 'Image Not Found ' + img_path

            h0, w0 = img.shape[:2]  # orig hw
            r = self.img_size / max(h0, w0)  # resize image to img_size
            if r < 1 or (self.augment and (r != 1)):  # always resize down, only resize up if training with augmentation
                interp = cv2.INTER_LINEAR if self.augment else cv2.INTER_AREA  # LINEAR for training, AREA for testing
                img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
                (h,w) = img.shape[:2]  # img, hw_original, hw_resized

        # Letterbox
        shape = self.img_size  # final letterboxed shape

        img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

        # Load labels
        labels = []
        if os.path.isfile(label_path):
            x = self.labels[index]
            if x.size > 0:
                # Normalized xywh to pixel xyxy format
                labels = x.copy()
                labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]  # pad width
                labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]  # pad height
                labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
                labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]

        nL = len(labels)  # number of labels
        print (f'nL {nL}')
        if nL:
            # convert xyxy to xywh
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

            # Normalize coordinates 0 - 1
            labels[:, [2, 4]] /= img.shape[0]  # height
            labels[:, [1, 3]] /= img.shape[1]  # width

        if self.augment:
            # random left-right flip
            lr_flip = True
            if lr_flip and random.random() < 0.5:
                img = np.fliplr(img)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]

            # random up-down flip
            ud_flip = False
            if ud_flip and random.random() < 0.5:
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]

        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        # print (f'before loader {torch.from_numpy(img).shape}_{labels_out}_{img_path}')
        return torch.from_numpy(img), labels_out, img_path, shapes

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

def letterbox(img, new_shape=(416, 416), color=(128, 128, 128),auto=False, scaleFill=False, scaleup=True, interp=cv2.INTER_AREA):

    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # Scale ratio (new / old)
    r = max(new_shape) / max(shape)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=interp)  # INTER_AREA is better, INTER_LINEAR is faster
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)