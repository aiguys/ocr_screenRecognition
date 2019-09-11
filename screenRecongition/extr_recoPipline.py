# -*- coding: utf-8 -*-#
# Author:       weiz
# Date:         2019/9/9 10:22
# Name:         extr_recoPipline
# Description:  1.对整张图进行处理，过滤其他噪音，提取图片中的字符区域；2.对整张图进行字符识别；3.输入某个字符片段，如果存在
#                 则标记该区域，并且标记离该区域最近的指示灯。

import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import part_reconPipline

def threshTrackbar(img):
    """
    使二值化图片过程可视化
    :param img: 待二值化的灰度图
    :return thr_v,Shading:阈值，大于阈值时替代的值
    """
    img_copy = img
    def nothing(x):
        pass
    # Create a window
    cv2.namedWindow('image')
    # create trackbars for color change
    cv2.createTrackbar('thr', 'image', 121, 255, nothing)
    cv2.createTrackbar('Shading', 'image', 255, 255, nothing)
    while (True):
        cv2.imshow('image', img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        # get current positions of four trackbars
        thr_v = cv2.getTrackbarPos('thr', 'image')
        shading = cv2.getTrackbarPos('Shading', 'image')
        _, img = cv2.threshold(img_copy, thr_v, shading, cv2.THRESH_BINARY)

    cv2.destroyAllWindows()
    return thr_v, shading

def recoText(im):
    """
    识别字符并返回所识别的字符及它们的坐标
    :param im: 需要识别的图片
    :return data: 字符及它们在图片的位置
    """
    data = {}
    # 你需要有hwsoft2字符识别库
    d = pytesseract.image_to_data(im, output_type=Output.DICT, lang='hwsoft2')
    for i in range(len(d['text'])):
        if 3 < len(d['text'][i]) < 8:
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            data[d['text'][i]] = ([d['left'][i], d['top'][i], d['width'][i], d['height'][i]])
            #cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
            #cv2.putText(im, d['text'][i], (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

    #cv2.imshow("recoText", im)
    return data

def drawMark(im, cil, rect, tdata, text):
    """
    在界面上标记出字符和最近的指示灯
    :param im: 待标记的图片
    :param cil: 圆指示灯位置信息
    :param rect: 矩形指示灯位置信息
    :param data: 图片中所有被识别的字符信息
    :param text: 需要标记的字符
    :return:
    """
    # 字符的中心坐标
    tx = tdata[text][0] + int(tdata[text][2] / 2)
    ty = tdata[text][1] + int(tdata[text][3] / 2)

    minc = 0 # 记录字符中心离圆形指示灯最近的那个灯
    minr = 0 # 记录字符中心里矩形指示灯最近的那个灯
    dc = 10000 # 距离
    dr = 10000
    # 找到字符中心离圆形指示灯最近的那个灯
    for i in range(len(cil)):
        if dc > getDis(tx, ty, cil[i][0], cil[i][1]):
            dc = getDis(tx, ty, cil[i][0], cil[i][1])
            minc = i
    # 找到字符中心离矩形指示灯最近的那个灯
    for i in range(len(rect)):
        if dr > getDis(tx, ty, rect[i][0], rect[i][1]):
            dr = getDis(tx, ty, rect[i][0], rect[i][1])
            minr = i

    if dc < dr:
        cv2.circle(im, (cil[minc][0], l_cil[minc][1]), l_cil[minc][2]-2, (255, 0, 0), 3)
    else:
        cv2.rectangle(im, (rect[minr][0], rect[minr][1]),
                      (rect[minr][0] + rect[minr][2], rect[minr][1] + rect[minr][3]), (255, 0, 0), 3)
    cv2.rectangle(im, (tdata[text][0], tdata[text][1]),
                  (tdata[text][0] + tdata[text][2], tdata[text][1] + tdata[text][3]), (255, 0, 0), 2)

    return im

def getDis(x1, y1, x2, y2):
    """
    获得两个点的距离
    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :return:
    """
    return ((x1-x2)**2 + (y1-y2)**2)**0.5


if __name__ == '__main__':
    e1 = cv2.getTickCount()
    img = cv2.imread('../data/ui.jpg')
    #cv2.imshow("src", img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("gray", gray)

    #thr, shading = threshTrackbar(gray)
    #print(thr, shading)
    _, thr_img = cv2.threshold(gray, 121, 255, cv2.THRESH_BINARY)
    thr_img = 255 - thr_img
    #cv2.imshow("thr", thr_img)

    data = recoText(thr_img)

    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 绿色的HSV区间
    green_low_range = np.array([35, 43, 46])
    green_high_range = np.array([77, 255, 255])
    mask_green = part_reconPipline.filterColorReg(hsv_image, green_low_range, green_high_range)

    # 红色的HSV区间
    red_low_range1 = np.array([0, 43, 46])  # red_low_range2 = np.array([156, 43, 46])
    red_high_range1 = np.array([10, 255, 255])  # red_high_range2 = np.array([180, 255, 255])
    mask_red = part_reconPipline.filterColorReg(hsv_image, red_low_range1, red_high_range1)

    # 提取绿色区域
    mask_green_cil, l_green_cil = part_reconPipline.filterAearCil(mask_green)
    mask_green_rect, l_green_rect = part_reconPipline.filterAearRect(mask_green)

    # 提取红色区域
    mask_red_cil, l_red_cil = part_reconPipline.filterAearCil(mask_red)
    mask_red_rect, l_red_rect = part_reconPipline.filterAearRect(mask_red)

    # 合并
    l_cil = l_green_cil + l_red_cil
    l_rect = l_green_rect + l_red_rect

    ret = drawMark(img, l_cil, l_rect, data, "151217") #102017

    e2 = cv2.getTickCount()
    time = (e2 - e1) / cv2.getTickFrequency()
    print(time)

    cv2.imshow("ret", ret)

    cv2.waitKey(0)
    cv2.destroyAllWindows()