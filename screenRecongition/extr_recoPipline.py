# -*- coding: utf-8 -*-#
# Author:       weiz
# Date:         2019/9/9 10:22
# Name:         extr_recoPipline
# Description:  1.对整张图进行处理，过滤其他噪音，提取图片中的字符区域；2.对整张图进行字符识别；3.输入某个字符片段，如果存在
#                 则标记该区域，并且标记离该区域最近的指示灯。

import cv2
import numpy as np

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
    cv2.createTrackbar('thr', 'image', 146, 255, nothing)
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


if __name__ == '__main__':
    img = cv2.imread('../data/ui.jpg')
    #cv2.imshow("src", img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("gray", gray)

    thr, shading = threshTrackbar(gray)
    print(thr,shading)
    _, thr_img = cv2.threshold(gray, 146, 255, cv2.THRESH_BINARY)
    cv2.imshow("thr", thr_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()