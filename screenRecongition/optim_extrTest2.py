# -*- coding: utf-8 -*-#
# Author:       weiz
# Date:         2019/9/4 17:38
# Name:         optim_extrTest2
# Description:

import cv2
import numpy as np
from screenRecongition import black_extr
import datetime

starttime = datetime.datetime.now()

# Step1. 转换为HSV
img = cv2.imread('../data/ui.jpg')
hue_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
back = np.zeros(img.shape, np.uint8)
white_backg = np.ones(img.shape, np.uint8) * 255



# Step2. 用颜色分割图像
green_low_range = np.array([35, 43, 46])
green_high_range = np.array([77, 255, 255])
mask = cv2.inRange(hue_image, green_low_range, green_high_range)
#cv2.imshow('mask', mask)


# Step3. 形态学运算，开运算
kernel_ell = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
mask_opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_ell)
#cv2.imshow('mask_opening', mask_opening)

l_cil = []
l_rect = []
contours, hierarch = cv2.findContours(mask_opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
for i in range(len(contours)):  # 提取圆周围的文本
    area = cv2.contourArea(contours[i])
    if 300 < area < 380:
        (x, y), radius = cv2.minEnclosingCircle(contours[i])
        center = (int(x), int(y))
        radius = int(radius)
        l_cil.append([int(x), int(y), int(radius)])
        cv2.circle(back, center, radius + 46, (255, 255, 255), -1)
        cv2.circle(back, center, radius, (0, 0, 0), -1)
        cv2.circle(white_backg, center, radius + 46, (0, 0, 0), -1)
        ROI = cv2.bitwise_and(img, back)


for i in range(len(contours)):  # 提取矩形周围的文本
    area = cv2.contourArea(contours[i])
    if 420 < area < 460:
        x, y, w, h = cv2.boundingRect(contours[i])
        l_rect.append([x, y, w, h])
        cv2.rectangle(back, (x - 20, y - 20), (x + w + 20, y + h + 20), (255, 255, 255), -1)
        cv2.rectangle(white_backg, (x - 20, y - 20), (x + w + 20, y + h + 20), (0, 0, 0), -1)
        cv2.rectangle(back, (x, y), (x + w, y + h), (0, 0, 0), -1)
        ROI = cv2.bitwise_and(img, back)

#cv2.imshow('img', img)
#cv2.imshow('back', back)
#cv2.imshow('ROI', ROI)

im_line = black_extr.getLineImage()
im_line = cv2.cvtColor(im_line, cv2.COLOR_BGR2GRAY)
im_opening = black_extr.getOpening()
gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
white_backg = cv2.cvtColor(white_backg, cv2.COLOR_BGR2GRAY)
#cv2.imshow('gray', gray)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
#cv2.imshow('thresh', thresh)

thresh = im_line + thresh
#thresh = im_opening + thresh
#cv2.imshow('thresh_1', thresh)

thresh = white_backg + thresh
cv2.imshow('white_backg', thresh)

for i in range(len(l_cil)):
    cv2.circle(thresh, (l_cil[i][0], l_cil[i][1]), l_cil[i][2] + 3, (255, 255, 255), -1)

for i in range(len(l_rect)):
    cv2.rectangle(thresh, (l_rect[i][0] - 2, l_rect[i][1] - 2),
                  (l_rect[i][0] + l_rect[i][2] + 2, l_rect[i][1] + l_rect[i][3] + 2), (255, 255, 255), -1)

thresh = 255 - thresh


if __name__ == '__main__':
    endtime = datetime.datetime.now()
    print(((endtime - starttime).microseconds) / 10 ** 6)
    cv2.imshow('thresh_2', thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()