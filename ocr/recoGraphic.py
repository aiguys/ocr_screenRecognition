# -*- coding: utf-8 -*-#
# Author:       weiz
# Date:         2019/9/4 14:22
# Name:         recoGraphic
# Description:  利用颜色特征提取特定区域

import cv2
import numpy as np

# Step1. 转换为HSV颜色空间
img = cv2.imread('./image/ui.jpg')
hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = np.zeros(img.shape, np.uint8)


# Step2. 用颜色分割图像，提取绿色区域
green_low_range = np.array([35, 43, 46])
green_high_range = np.array([77, 255, 255])
green_reg = cv2.inRange(hsv_image, green_low_range, green_high_range)
#cv2.imshow('green_reg', green_reg)


# Step3. 使用形态学运算去噪：开运算
kernel_ell = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
green_reg_opening = cv2.morphologyEx(green_reg, cv2.MORPH_OPEN, kernel_ell)
#cv2.imshow('green_reg_opening', green_reg_opening)

# Step4. 使用区域的面积筛选
contours, _ = cv2.findContours(green_reg_opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
for i in range(len(contours)):  # 提取圆
    area = cv2.contourArea(contours[i])
    if 300 < area < 380:
        (x, y), radius = cv2.minEnclosingCircle(contours[i])
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(mask, center, radius, (255, 255, 255), -1)
        ROI = cv2.bitwise_and(img, mask)

for i in range(len(contours)):  # 提取矩形
    area = cv2.contourArea(contours[i])
    if 420 < area < 440:
        x, y, w, h = cv2.boundingRect(contours[i])
        cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), -1)
        ROI = cv2.bitwise_and(img, mask)

cv2.imshow('img', img)
cv2.imshow('mask', mask)
cv2.imshow('ROI', ROI)

cv2.waitKey(0)
cv2.destroyAllWindows()