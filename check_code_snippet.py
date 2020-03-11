import cv2
import numpy as np
import matplotlib.pyplot as plt

lab='/media/persistance/storage/deep _learning/yolov3/data/coco/labels/val2014/COCO_val2014_000000000074.txt'
img=lab.replace('labels','images').split('.')[0]+'.jpg'

# print (img)
img=cv2.imread(img)
hei, wid, _=img.shape

with open(lab) as file:
    vals=file.read().split('\n')
vals=np.array([i.split(' ') for i in vals if i])

def convert(vals):
    x, width = round(vals[1]*wid), round(vals[3]*wid)
    y, height = round(vals[2]*hei), round(vals[4]*hei)
    st= round(x-(width/2)) , round(y-(height/2))
    end= round(x+(width/2)), round(y+(height/2))
    return st, end
#
color = (255, 0, 0)
thickness = 2
# #
for j in vals:
    st, end=convert(list(map(float, j)))
    image = cv2.rectangle(img, st, end, color, thickness)
cv2.imshow('show', image)
cv2.waitKey(0)