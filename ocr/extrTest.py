# -*- coding: utf-8 -*-#
# Author:       weiz
# Date:         2019/9/4 14:44
# Name:         extrTest
# Description:  先利用颜色特征定位到指定区域，在该区域周围提取包含文本的候选区域。在候选区域中利用形态学提取文本

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
kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
green_reg_opening = cv2.morphologyEx(green_reg, cv2.MORPH_OPEN, kernel_ell)
#cv2.imshow('green_reg_opening', green_reg_opening)

# Step4. 使用区域的面积筛选
contours, _ = cv2.findContours(green_reg_opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
for i in range(len(contours)):  # 提取圆周围包含文本的候选区域
    area = cv2.contourArea(contours[i])
    if 300 < area < 380:
        (x, y), radius = cv2.minEnclosingCircle(contours[i])
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(mask, center, radius + 25, (255, 255, 255), -1)
        cv2.circle(mask, center, radius, (0, 0, 0), -1)
        candid = cv2.bitwise_and(img, mask)


for i in range(len(contours)):  # 提取矩形周围包含文本的候选区域
    area = cv2.contourArea(contours[i])
    if 420 < area < 460:
        x, y, w, h = cv2.boundingRect(contours[i])
        cv2.rectangle(mask, (x - 20, y - 20), (x + w + 20, y + h + 20), (255, 255, 255), -1)
        cv2.rectangle(mask, (x, y), (x + w, y + h), (0, 0, 0), -1)
        candid = cv2.bitwise_and(img, mask)

#cv2.imshow('img', img)
#cv2.imshow('mask', mask)
#cv2.imshow('Candidate region', candid)

# Step5. 对候选区域进行形态学处理
gray = cv2.cvtColor(candid, cv2.COLOR_BGR2GRAY)
#cv2.imshow('gray', gray)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
#cv2.imshow('thresh', thresh)
thresh_inve = 255 - thresh
#cv2.imshow('thresh_inve', thresh_inve)
thresh_inve_close = cv2.morphologyEx(thresh_inve, cv2.MORPH_CLOSE, kernel_rect)
#cv2.imshow('thresh_inve_close', thresh_inve_close)


dilation = cv2.dilate(thresh, kernel_rect, iterations=1)
#cv2.imshow('dilation', dilation)
erosion = cv2.erode(dilation, kernel_rect, iterations=1)
#cv2.imshow('erosion', erosion)

ret = cv2.bitwise_and(thresh_inve_close, erosion)
#cv2.imshow('ret', ret)
ret_opening = cv2.morphologyEx(ret, cv2.MORPH_OPEN, kernel_ell)
cv2.imshow('ret_opening', ret_opening)
contours, hierarchy = cv2.findContours(ret_opening, 1, 2)
for i in range(len(contours)):
    x, y, w, h = cv2.boundingRect(contours[i])
    cv2.rectangle(ret_opening, (x, y), (x+w, y+h), (255, 255, 255), -1)
#cv2.imshow('ret_opening1', ret_opening)

color_ret = cv2.cvtColor(ret_opening, cv2.COLOR_GRAY2BGR)
fin_ret = cv2.bitwise_and(img, color_ret)
cv2.imshow('Final Results', fin_ret)

cv2.waitKey(0)
cv2.destroyAllWindows()