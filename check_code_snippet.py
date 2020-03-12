import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch

# lab='/media/persistance/storage/deep _learning/yolov3/data/coco/labels/val2014/COCO_val2014_000000000074.txt'
# img=lab.replace('labels','images').split('.')[0]+'.jpg'
#
# # print (img)
# img=cv2.imread(img)
# hei, wid, _=img.shape
#
# with open(lab) as file:
#     vals=file.read().split('\n')
# vals=np.array([i.split(' ') for i in vals if i])
#
# def convert(vals):
#     x, width = round(vals[1]*wid), round(vals[3]*wid)
#     y, height = round(vals[2]*hei), round(vals[4]*hei)
#     st= round(x-(width/2)) , round(y-(height/2))
#     end= round(x+(width/2)), round(y+(height/2))
#     return st, end
# #
# color = (255, 0, 0)
# thickness = 2
# # #
# for j in vals:
#     st, end=convert(list(map(float, j)))
#     image = cv2.rectangle(img, st, end, color, thickness)
# cv2.imshow('show', image)
# cv2.waitKey(0)




bn = 1
filters = 32
size = 3
stride = 1
pad = 1

module_list=nn.ModuleList()

for i in range (2):

    if i==0:
        modules = nn.Sequential()
        modules.add_module('Conv2d', nn.Conv2d(in_channels=3, out_channels=filters, kernel_size=size,
                                               stride=stride, padding=pad, bias=not bn))
        modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.1))
        modules.add_module('Activation', nn.LeakyReLU(0.1, inplace=True))

    elif i>0:
        modules = nn.Upsample(scale_factor=2, mode='nearest')

    module_list.append(modules)

print (module_list)