import glob
import math
import os
import random
import shutil
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm

from . import torch_utils  # , google_utils

matplotlib.rc('font', **{'size': 11})

# Set printoptions
torch.set_printoptions(linewidth=320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5

# Prevent OpenCV from multithreading (to use PyTorch DataLoader)
cv2.setNumThreads(0)


def floatn(x, n=3):  # format floats to n decimals
    return float(format(x, '.%gf' % n))


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch_utils.init_seeds(seed=seed)


def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)


def labels_to_class_weights(labels, nc=80):
    # Get class weights (inverse frequency) from training labels
    if labels[0] is None:  # no labels loaded
        return torch.Tensor()

    labels = np.concatenate(labels, 0)  # labels.shape = (866643, 5) for COCO
    classes = labels[:, 0].astype(np.int)  # labels = [class xywh]
    weights = np.bincount(classes, minlength=nc)  # occurences per class

    # Prepend gridpoint count (for uCE trianing)
    # gpi = ((320 / 32 * np.array([1, 2, 4])) ** 2 * 3).sum()  # gridpoints per image
    # weights = np.hstack([gpi * len(labels)  - weights.sum() * 9, weights * 9]) ** 0.5  # prepend gridpoints to start

    weights[weights == 0] = 1  # replace empty bins with 1
    weights = 1 / weights  # number of targets per class
    weights /= weights.sum()  # normalize
    return torch.from_numpy(weights)

def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x

def xyxy2xywh(x):
    # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def xywh2xyxy(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = max(img1_shape) / max(img0_shape)  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=img_shape[1])  # clip x
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=img_shape[0])  # clip y


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls: Predicted object classes (nparray).
        target_cls: True object classes (nparray).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    s = [len(unique_classes), tp.shape[1]]  # number class, number iou thresholds (i.e. 10 for mAP0.5...0.95)
    ap, p, r = np.zeros(s), np.zeros(s), np.zeros(s)
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 or n_gt == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_gt + 1e-16)  # recall curve
            r[ci] = recall[-1]

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = precision[-1]

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j] = compute_ap(recall[:, j], precision[:, j])

            # Plot
            # fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            # ax.plot(recall, precision)
            # ax.set_xlabel('Recall')
            # ax.set_ylabel('Precision')
            # ax.set_xlim(0, 1.01)
            # ax.set_ylim(0, 1.01)
            # fig.tight_layout()
            # fig.savefig('PR_curve.png', dpi=300)

    # Compute F1 score (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype('int32')


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [min(recall[-1] + 1E-3, 1.)]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.t()

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # x, y, w, h = box1
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    iou = inter / union  # iou
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            return iou - (c_area - union) / c_area  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU

    return iou


def box_iou(boxes1, boxes2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(boxes1.t())
    area2 = box_area(boxes2.t())

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    inter = (rb - lt).clamp(min=0).prod(2)  # [N,M]
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

def wh_iou(wh1, wh2):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)

def compute_loss(p, targets, model, giou_flag=True):  # predictions, targets, model
    ft = torch.cuda.FloatTensor if p[0].is_cuda else torch.Tensor

    lcls, lbox, lobj = ft([0]), ft([0]), ft([0])

    tcls, tbox, indices, anchor_vec = build_targets(model, targets)
    # print(tcls.shape, tbox.shape, indices.shape, anchor_vec.shape)
    h = model.hyp  # hyperparameters
    arc = model.arc  # # (default, uCE, uBCE) detection architectures
    red = 'sum'  # Loss reduction (sum or mean)
    print (f'arc {arc}')
    # Define criteria
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=ft([h['cls_pw']]), reduction=red)
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=ft([h['obj_pw']]), reduction=red)
    BCE = nn.BCEWithLogitsLoss(reduction=red)
    CE = nn.CrossEntropyLoss(reduction=red)  # weight=model.class_weights

    if 'F' in arc:  # add focal loss
        g = h['fl_gamma']
        BCEcls, BCEobj, BCE, CE = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g), FocalLoss(BCE, g), FocalLoss(CE, g)

    # Compute losses
    np, ng = 0, 0  # number grid points, targets
    for i, pi in enumerate(p):  # layer index, layer predictions
        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
        tobj = torch.zeros_like(pi[..., 0])  # target obj
        np += tobj.numel()

        # Compute losses
        nb = len(b)
        if nb:  # number of targets
            ng += nb
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
            # ps[:, 2:4] = torch.sigmoid(ps[:, 2:4])  # wh power loss (uncomment)

            # GIoU
            pxy = torch.sigmoid(ps[:, 0:2])  # pxy = pxy * s - (s - 1) / 2,  s = 1.5  (scale_xy)
            pwh = torch.exp(ps[:, 2:4]).clamp(max=1E3) * anchor_vec[i]
            pbox = torch.cat((pxy, pwh), 1)  # predicted box
            giou = bbox_iou(pbox.t(), tbox[i], x1y1x2y2=False, GIoU=True)  # giou computation
            lbox += (1.0 - giou).sum() if red == 'sum' else (1.0 - giou).mean()  # giou loss
            tobj[b, a, gj, gi] = giou.detach().type(tobj.dtype) if giou_flag else 1.0

            if 'default' in arc and model.nc > 1:  # cls loss (only if multiple classes)
                t = torch.zeros_like(ps[:, 5:])  # targets
                t[range(nb), tcls[i]] = 1.0
                lcls += BCEcls(ps[:, 5:], t)  # BCE
                # lcls += CE(ps[:, 5:], tcls[i])  # CE

                # Instance-class weighting (use with reduction='none')
                # nt = t.sum(0) + 1  # number of targets per class
                # lcls += (BCEcls(ps[:, 5:], t) / nt).mean() * nt.mean()  # v1
                # lcls += (BCEcls(ps[:, 5:], t) / nt[tcls[i]].view(-1,1)).mean() * nt.mean()  # v2

            # Append targets to text file
            # with open('targets.txt', 'a') as file:
            #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

        if 'default' in arc:  # separate obj and cls
            lobj += BCEobj(pi[..., 4], tobj)  # obj loss

        elif 'BCE' in arc:  # unified BCE (80 classes)
            t = torch.zeros_like(pi[..., 5:])  # targets
            if nb:
                t[b, a, gj, gi, tcls[i]] = 1.0
            lobj += BCE(pi[..., 5:], t)

        elif 'CE' in arc:  # unified CE (1 background + 80 classes)
            t = torch.zeros_like(pi[..., 0], dtype=torch.long)  # targets
            if nb:
                t[b, a, gj, gi] = tcls[i] + 1
            lcls += CE(pi[..., 4:].view(-1, model.nc + 1), t.view(-1))

    lbox *= h['giou']
    lobj *= h['obj']
    lcls *= h['cls']
    if red == 'sum':
        bs = tobj.shape[0]  # batch size
        lobj *= 3 / (6300 * bs) * 2  # 3 / np * 2
        if ng:
            lcls *= 3 / ng / model.nc
            lbox *= 3 / ng

    loss = lbox + lobj + lcls
    return loss, torch.cat((lbox, lobj, lcls, loss)).detach()

# used is LOSS CALC
def build_targets(model, targets):
    # targets = [image, class, x, y, w, h]

    nt = len(targets)
    tcls, tbox, indices, av = [], [], [], []
    multi_gpu = type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
    reject, use_all_anchors = True, True
    for i in model.yolo_layers:
        # get number of grid points and anchor vec for this yolo layer
        if multi_gpu:
            ng, anchor_vec = model.module.module_list[i].ng, model.module.module_list[i].anchor_vec
        else:
            ng, anchor_vec = model.module_list[i].ng, model.module_list[i].anchor_vec

        # iou of targets-anchors
        t, a = targets, []
        gwh = t[:, 4:6] * ng
        if nt:
            iou = wh_iou(anchor_vec, gwh)

            if use_all_anchors:
                na = len(anchor_vec)  # number of anchors
                a = torch.arange(na).view((-1, 1)).repeat([1, nt]).view(-1)
                t = targets.repeat([na, 1])
                gwh = gwh.repeat([na, 1])
            else:  # use best anchor only
                iou, a = iou.max(0)  # best iou and anchor

            # reject anchors below iou_thres (OPTIONAL, increases P, lowers R)
            if reject:
                j = iou.view(-1) > model.hyp['iou_t']  # iou threshold hyperparameter
                t, a, gwh = t[j], a[j], gwh[j]

        # Indices
        b, c = t[:, :2].long().t()  # target image, class
        gxy = t[:, 2:4] * ng  # grid x, y
        gi, gj = gxy.long().t()  # grid x, y indices
        indices.append((b, a, gj, gi))

        # Box
        gxy -= gxy.floor()  # xy
        tbox.append(torch.cat((gxy, gwh), 1))  # xywh (grids)
        av.append(anchor_vec[a])  # anchor vec

        # Class
        tcls.append(c)
        if c.shape[0]:  # if any targets
            assert c.max() < model.nc, 'Model accepts %g classes labeled from 0-%g, however you labelled a class %g. ' \
                                       'See https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data' % (
                                           model.nc, model.nc - 1, c.max())

    return tcls, tbox, indices, av


def non_max_suppression(prediction, conf_thres=0.5, iou_thres=0.5, multi_cls=True, classes=None, agnostic=False):
    """
    Removes detections with lower object confidence score than 'conf_thres'
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, conf, class)
    """
    # NMS methods https://github.com/ultralytics/yolov3/issues/679 'or', 'and', 'merge', 'vision', 'vision_batch'

    # Box constraints
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height

    method = 'vision_batch'
    nc = prediction[0].shape[1] - 5  # number of classes
    multi_cls = multi_cls and (nc > 1)  # allow multiple classes per anchor
    output = [None] * len(prediction)
    for image_i, pred in enumerate(prediction):
        # Apply conf constraint
        pred = pred[pred[:, 4] > conf_thres]

        # Apply width-height constraint
        pred = pred[(pred[:, 2:4] > min_wh).all(1) & (pred[:, 2:4] < max_wh).all(1)]

        # If none remain process next image
        if len(pred) == 0:
            continue

        # Compute conf
        pred[..., 5:] *= pred[..., 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(pred[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_cls:
            i, j = (pred[:, 5:] > conf_thres).nonzero().t()
            pred = torch.cat((box[i], pred[i, j + 5].unsqueeze(1), j.float().unsqueeze(1)), 1)
        else:  # best class only
            conf, j = pred[:, 5:].max(1)
            pred = torch.cat((box, conf.unsqueeze(1), j.float().unsqueeze(1)), 1)

        # Filter by class
        if classes:
            pred = pred[(j.view(-1, 1) == torch.tensor(classes, device=j.device)).any(1)]

        # Apply finite constraint
        if not torch.isfinite(pred).all():
            pred = pred[torch.isfinite(pred).all(1)]

        # Batched NMS
        if method == 'vision_batch':
            c = pred[:, 5] * 0 if agnostic else pred[:, 5]  # class-agnostic NMS
            output[image_i] = pred[torchvision.ops.boxes.batched_nms(pred[:, :4], pred[:, 4], c, iou_thres)]
            continue

        # Sort by confidence
        if not method.startswith('vision'):
            pred = pred[pred[:, 4].argsort(descending=True)]

        # All other NMS methods
        det_max = []
        cls = pred[:, -1]
        for c in cls.unique():
            dc = pred[cls == c]  # select class c
            n = len(dc)
            if n == 1:
                det_max.append(dc)  # No NMS required if only 1 prediction
                continue
            elif n > 500:
                dc = dc[:500]  # limit to first 500 boxes: https://github.com/ultralytics/yolov3/issues/117

            if method == 'vision':
                det_max.append(dc[torchvision.ops.boxes.nms(dc[:, :4], dc[:, 4], iou_thres)])

            elif method == 'or':  # default
                # METHOD1
                # ind = list(range(len(dc)))
                # while len(ind):
                # j = ind[0]
                # det_max.append(dc[j:j + 1])  # save highest conf detection
                # reject = (bbox_iou(dc[j], dc[ind]) > iou_thres).nonzero()
                # [ind.pop(i) for i in reversed(reject)]

                # METHOD2
                while dc.shape[0]:
                    det_max.append(dc[:1])  # save highest conf detection
                    if len(dc) == 1:  # Stop if we're at the last detection
                        break
                    iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                    dc = dc[1:][iou < iou_thres]  # remove ious > threshold

            elif method == 'and':  # requires overlap, single boxes erased
                while len(dc) > 1:
                    iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                    if iou.max() > 0.5:
                        det_max.append(dc[:1])
                    dc = dc[1:][iou < iou_thres]  # remove ious > threshold

            elif method == 'merge':  # weighted mixture box
                while len(dc):
                    if len(dc) == 1:
                        det_max.append(dc)
                        break
                    i = bbox_iou(dc[0], dc) > iou_thres  # iou with other boxes
                    weights = dc[i, 4:5]
                    dc[0, :4] = (weights * dc[i, :4]).sum(0) / weights.sum()
                    det_max.append(dc[:1])
                    dc = dc[i == 0]

            elif method == 'soft':  # soft-NMS https://arxiv.org/abs/1704.04503
                sigma = 0.5  # soft-nms sigma parameter
                while len(dc):
                    if len(dc) == 1:
                        det_max.append(dc)
                        break
                    det_max.append(dc[:1])
                    iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                    dc = dc[1:]
                    dc[:, 4] *= torch.exp(-iou ** 2 / sigma)  # decay confidences
                    dc = dc[dc[:, 4] > conf_thres]  # https://github.com/ultralytics/yolov3/issues/362

        if len(det_max):
            det_max = torch.cat(det_max)  # concatenate
            output[image_i] = det_max[(-det_max[:, 4]).argsort()]  # sort

    return output


def get_yolo_layers(model):
    bool_vec = [x['type'] == 'yolo' for x in model.module_defs]
    return [i for i, x in enumerate(bool_vec) if x]  # [82, 94, 106] for yolov3


def print_model_biases(model):
    # prints the bias neurons preceding each yolo layer
    print('\nModel Bias Summary: %8s%18s%18s%18s' % ('layer', 'regression', 'objectness', 'classification'))
    multi_gpu = type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
    for l in model.yolo_layers:  # print pretrained biases
        if multi_gpu:
            na = model.module.module_list[l].na  # number of anchors
            b = model.module.module_list[l - 1][0].bias.view(na, -1)  # bias 3x85
        else:
            na = model.module_list[l].na
            b = model.module_list[l - 1][0].bias.view(na, -1)  # bias 3x85
        print(' ' * 20 + '%8g %18s%18s%18s' % (l, '%5.2f+/-%-5.2f' % (b[:, :4].mean(), b[:, :4].std()),
                                               '%5.2f+/-%-5.2f' % (b[:, 4].mean(), b[:, 4].std()),
                                               '%5.2f+/-%-5.2f' % (b[:, 5:].mean(), b[:, 5:].std())))

def fitness(x):
    # Returns fitness (for use with results.txt or evolve.txt)
    w = [0.0, 0.01, 0.99, 0.00]  # weights for [P, R, mAP, F1]@0.5 or [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)


# Plotting functions --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def plot_images(imgs, targets, paths=None, fname='images.png'):
    # Plots training images overlaid with targets
    imgs = imgs.cpu().numpy()
    targets = targets.cpu().numpy()
    # targets = targets[targets[:, 1] == 21]  # plot only one class

    fig = plt.figure(figsize=(10, 10))
    bs, _, h, w = imgs.shape  # batch size, _, height, width
    bs = min(bs, 16)  # limit plot to 16 images
    ns = np.ceil(bs ** 0.5)  # number of subplots

    for i in range(bs):
        boxes = xywh2xyxy(targets[targets[:, 0] == i, 2:6]).T
        boxes[[0, 2]] *= w
        boxes[[1, 3]] *= h
        plt.subplot(ns, ns, i + 1).imshow(imgs[i].transpose(1, 2, 0))
        plt.plot(boxes[[0, 2, 2, 0, 0]], boxes[[1, 1, 3, 3, 1]], '.-')
        plt.axis('off')
        if paths is not None:
            s = Path(paths[i]).name
            plt.title(s[:min(len(s), 40)], fontdict={'size': 8})  # limit to 40 characters
    fig.tight_layout()
    fig.savefig(fname, dpi=200)
    plt.close()

def plot_results(start=0, stop=0, bucket='', id=()):  # from utils.utils import *; plot_results()
    # Plot training results files 'results*.txt'
    fig, ax = plt.subplots(2, 5, figsize=(14, 7))
    ax = ax.ravel()
    s = ['GIoU', 'Objectness', 'Classification', 'Precision', 'Recall',
         'val GIoU', 'val Objectness', 'val Classification', 'mAP@0.5', 'F1']
    if bucket:
        os.system('rm -rf storage.googleapis.com')
        files = ['https://storage.googleapis.com/%s/results%g.txt' % (bucket, x) for x in id]
    else:
        files = glob.glob('results*.txt') + glob.glob('../../Downloads/results*.txt')
    for f in sorted(files):
        results = np.loadtxt(f, usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T
        n = results.shape[1]  # number of rows
        x = range(start, min(stop, n) if stop else n)
        for i in range(10):
            y = results[i, x]
            if i in [0, 1, 2, 5, 6, 7]:
                y[y == 0] = np.nan  # dont show zero loss values
                # y /= y[0]  # normalize
            ax[i].plot(x, y, marker='.', label=Path(f).stem)
            ax[i].set_title(s[i])
            if i in [5, 6, 7]:  # share train and val loss y axes
                ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])

    fig.tight_layout()
    ax[1].legend()
    fig.savefig('results.png', dpi=200)