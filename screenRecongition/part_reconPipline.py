# -*- coding: utf-8 -*-#
# Author:       weiz
# Date:         2019/9/9 14:10
# Name:         part_reconPipline
# Description:  1.利用颜色特征，定位特定区域；2.在该特定区域提取需要的文本区域；3.处理包含文本区域并进行识别；4.输入某个
#                 字符片段，如果存在则标记该区域，并且标记离该区域最近的指示灯。

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

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

def filterAearCil(im, minA=None, maxA=None):
    """
    根据连通区域的面积筛选圆的区域
    :param im:待处理的二值图像
    :param minA:筛选面积的下限
    :param maxA:筛选面积的上限
    :return ROI,l_cil:筛选后的区域，圆的位置信息
    """
    if minA == None:
        minA = 300
    if maxA == None:
        maxA = 420
    l_cil = []
    back = np.zeros(im.shape, np.uint8)
    # 使用面积筛选
    contours, hierarch = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):  # 提取圆周围的文本
        area = cv2.contourArea(contours[i])
        if minA < area < maxA:
            (x, y), radius = cv2.minEnclosingCircle(contours[i])
            center = (int(x), int(y))
            radius = int(radius)
            l_cil.append([int(x), int(y), int(radius) + 5])
            #cv2.circle(back, center, radius + 25, (255, 255, 255), -1)
            cv2.circle(back, center, radius, (255, 255, 255), -1)
            #ROI = cv2.bitwise_and(im, back)

    return back, l_cil

def filterAearRect(im, minA=None, maxA=None):
    """
    根据连通区域的面积筛选矩形的区域
    :param im:待处理的二值图像
    :param minA:筛选面积的下限
    :param maxA:筛选面积的上限
    :return ROI,l_rect:筛选后的区域，矩形的位置信息
    """
    if minA == None:
        minA = 420
    if maxA == None:
        maxA = 530
    l_rect = []
    back = np.zeros(im.shape, np.uint8)
    # 使用面积筛选
    contours, hierarch = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):  # 提取矩形周围的文本
        area = cv2.contourArea(contours[i])
        if minA < area < maxA:
            x, y, w, h = cv2.boundingRect(contours[i])
            l_rect.append([x - 2, y - 2, w + 4, h + 4])
            #cv2.rectangle(back, (x - 20, y - 20), (x + w + 20, y + h + 20), (255, 255, 255), -1)
            cv2.rectangle(back, (x, y), (x + w, y + h), (255, 255, 255), -1)
            #ROI = cv2.bitwise_and(im, back)

    return back, l_rect

def getStrReg(im, l_cil=None, l_rect=None):
    """
    提取包含字符的区域
    :param im: 输入图片
    :param l_cil: 以圆为图形的数据列表
    :param l_rect: 以矩形为图形的数据列表
    :return: 返回提取了字符的区域
    """
    if l_cil == None:
        l_cil = []
    if l_rect == None:
        l_rect = []
    back = np.zeros(im.shape, np.uint8)
    for i in range(len(l_cil)):
        cv2.circle(back, (l_cil[i][0], l_cil[i][1]), l_cil[i][2]+42, (255, 255, 255), -1)
    for i in range(len(l_rect)):
        cv2.rectangle(back, (l_rect[i][0]-20, l_rect[i][1]-20),
                      (l_rect[i][0] + l_rect[i][2]+20, l_rect[i][1] + l_rect[i][3]+20), (255, 255, 255), -1)

    str_reg = cv2.bitwise_and(im, back)
    return str_reg, back

def getTrainSet(im, l_cil, l_rect):
    """
    获得训练数据
    :param im1: 输入图片
    :param l_cil:
    :param l_rect:
    :return:
    """
    mask = np.ones(img.shape, np.uint8)*255
    for i in range(len(l_cil)):
        cv2.circle(im, (l_cil[i][0], l_cil[i][1]), l_cil[i][2]-2, (255, 255, 255), -1)
    for i in range(len(l_rect)):
        cv2.rectangle(im, (l_rect[i][0]-2, l_rect[i][1]-2),
                      (l_rect[i][0] + l_rect[i][2]+2, l_rect[i][1] + l_rect[i][3]+2), (255, 255, 255), -1)

    for i in range(len(l_cil)):
        cv2.circle(mask, (l_cil[i][0], l_cil[i][1]), l_cil[i][2]+42, (0, 0, 0), -1)
    for i in range(len(l_rect)):
        cv2.rectangle(mask, (l_rect[i][0]-20, l_rect[i][1]-20),
                      (l_rect[i][0] + l_rect[i][2]+20, l_rect[i][1] + l_rect[i][3]+20), (0, 0, 0), -1)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    im = im + mask
    l_line = getLine()
    for i in range(len(l_line)):
        cv2.drawContours(im, l_line[i], -1, (255, 255, 255), thickness=4)

    im = 255 - im
    cv2.imshow("trainset", im)

def recoText(im, img):
    """
    识别字符并返回所识别的字符及它们的坐标
    :param im: 需要识别的图片
    :return data: 字符及它们在图片的位置
    """
    data = {}
    d = pytesseract.image_to_data(im, output_type=Output.DICT, lang='hwsoft')
    for i in range(len(d['text'])):
        if 3 < len(d['text'][i]) < 8:
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            data[d['text'][i]] = ([d['left'][i], d['top'][i], d['width'][i], d['height'][i]])
            #cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            #cv2.putText(img, d['text'][i], (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
    #cv2.imshow("recoText", img)
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
    #cv2.imshow('red', mask_red)

    # 提取绿色区域
    mask_green_cil, l_green_cil = filterAearCil(mask_green)
    mask_green_rect, l_green_rect = filterAearRect(mask_green)
    #cv2.imshow("mask_green_cil", mask_green_cil)
    #cv2.imshow("mask_green_rect", mask_green_rect)
    ROI_green = cv2.add(mask_green_cil, mask_green_rect)
    #cv2.imshow("mask_green", ROI_green)

    # 提取红色区域
    mask_red_cil, l_red_cil = filterAearCil(mask_red)
    mask_red_rect, l_red_rect = filterAearRect(mask_red)
    #cv2.imshow("mask_red_cil", mask_red_cil)
    #cv2.imshow("mask_red_rect", mask_red_rect)
    ROI_red = cv2.add(mask_red_cil, mask_red_rect)
    #cv2.imshow("mask_red", ROI_red)

    # 合并
    ROI = cv2.add(ROI_green, ROI_red)
    ROI = cv2.cvtColor(ROI, cv2.COLOR_GRAY2BGR)
    #ROI_img = cv2.bitwise_and(img, ROI)
    #cv2.imshow("ROI_img", ROI_img)
    l_cil = l_green_cil + l_red_cil
    l_rect = l_green_rect + l_red_rect

    # 提取原图中包含字符的区域
    img_str, back = getStrReg(img, l_cil, l_rect)
    #cv2.imshow("img_str", img_str)

    # 二值化
    img_str_gray = cv2.cvtColor(img_str, cv2.COLOR_BGR2GRAY)
    _, thr_img = cv2.threshold(img_str_gray, 121, 255, cv2.THRESH_BINARY)
    back = 255 - back
    back = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY)
    thr_img = cv2.add(thr_img, back)
    thr_img = 255 - thr_img
    #cv2.imshow("thr", thr_img)

    #getTrainSet(thr_img, l_cil, l_rect)

    #cv2.imshow("hah", thr_img)
    #cv2.imshow("hah1", back)
    data = recoText(thr_img, img)

    ret = drawMark(img, l_cil, l_rect, data, "05111") #05026

    e2 = cv2.getTickCount()
    time = (e2 - e1) / cv2.getTickFrequency()
    print(time)

    cv2.imshow("ret", ret)

    cv2.waitKey(0)
    cv2.destroyAllWindows()