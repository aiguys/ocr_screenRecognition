# -*- coding: utf-8 -*-#
# Author:       weiz
# Date:         2019/9/9 14:10
# Name:         part_reconPipline
# Description:  1.利用颜色特征，定位特定区域；2.在该特定区域提取需要的文本区域；3.处理包含文本区域并进行识别；4.输入某个
#                 字符片段，如果存在则标记该区域，并且标记离该区域最近的指示灯。

import cv2
import numpy as np

def filterColorReg(im, low, high):
    """
    根据颜色不同选取特定区域
    :param im: 输入的HSV格式图片
    :param low: 颜色阈值的下限
    :param high: 颜色阈值的上限
    :return: 返回根据颜色所选取的区域
    """
    reg = cv2.inRange(im, low, high)
    kernel_ell = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    reg = cv2.morphologyEx(reg, cv2.MORPH_OPEN, kernel_ell)
    return reg

def filterAearReg(im, minA, maxA):
    """

    :param im:
    :param minA:
    :param maxA:
    :return:
    """
    l_cil = []
    l_rect = []
    back = np.zeros(im.shape, np.uint8)
    # Step4. 使用面积筛选
    contours, hierarch = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):  # 提取圆周围的文本
        area = cv2.contourArea(contours[i])
        if 300 < area < 380:
            (x, y), radius = cv2.minEnclosingCircle(contours[i])
            center = (int(x), int(y))
            radius = int(radius)
            l_cil.append([int(x), int(y), int(radius) + 5])
            cv2.circle(back, center, radius + 25, (255, 255, 255), -1)
            cv2.circle(back, center, radius, (0, 0, 0), -1)
            ROI = cv2.bitwise_and(img, back)

    for i in range(len(contours)):  # 提取矩形周围的文本
        area = cv2.contourArea(contours[i])
        if 420 < area < 460:
            x, y, w, h = cv2.boundingRect(contours[i])
            l_rect.append([x - 2, y - 2, w + 4, h + 4])
            cv2.rectangle(back, (x - 20, y - 20), (x + w + 20, y + h + 20), (255, 255, 255), -1)
            cv2.rectangle(back, (x, y), (x + w, y + h), (0, 0, 0), -1)
            ROI = cv2.bitwise_and(img, back)

    return ROI, l_cil, l_rect

if __name__ == '__main__':
    # 转换为HSV
    img = cv2.imread('../data/ui.jpg')
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #cv2.imshow("hsv", hsv_image)

    # 绿色的HSV区间
    green_low_range = np.array([35, 43, 46])
    green_high_range = np.array([77, 255, 255])
    mask_green = filterColorReg(hsv_image, green_low_range, green_high_range)
    #cv2.imshow("green", mask_green)

    # 红色的HSV区间
    red_low_range1 = np.array([0, 43, 46]) #red_low_range2 = np.array([156, 43, 46])
    red_high_range1 = np.array([10, 255, 255]) #red_high_range2 = np.array([180, 255, 255])
    mask_red = filterColorReg(hsv_image, red_low_range1, red_high_range1)
    cv2.imshow('red1', mask_red)

    filterAearReg(mask_green)

    cv2.waitKey(0)
    cv2.destroyAllWindows()