# -*- coding: utf-8 -*-#
# Author:       weiz
# Date:         2019/9/6 14:56
# Name:         statisCharNum
# Description:  统计文件名字字符的个数，将图片的其他格式转化为.tif格式

import os
import collections
import cv2

root = "C:\\Users\weiz\Desktop\\trainBinaryMap"

def findjpg(path):
    """Finding the *.txt file in specify path
    input:文件路径
    output：返回指定后缀名的文件路径
    """
    ret = []
    filelist = os.listdir(path)
    for filename in filelist:
        de_path = os.path.join(path, filename)
        if os.path.isfile(de_path):
            if de_path.endswith(".jpg"): #Specify to find the txt file.
                ret.append(de_path)
        else:
            findjpg(de_path)
    return ret

def statisticCharNum(fileList):
    """
    统计某个路径下文件名各个字符的个数
    :param fileList:
    :return:
    """
    str = ''
    for path in fileList:
        (filepath, tempfilename) = os.path.split(path)
        (filename, extension) = os.path.splitext(tempfilename)
        str = str + filename

    count_dict = dict(collections.Counter(str))
    count_dict = sorted(count_dict.items(), key=lambda d:d[0])
    return len(str), count_dict

def other2tif(pathList):
    """
    将图片的其他格式转化为.tif格式形式
    :param pathList: 文件路径
    :return:
    """
    for path in pathList:
        img = cv2.imread(path)
        (filepath, tempfilename) = os.path.split(path)
        (filename, extension) = os.path.splitext(tempfilename)
        filename = filename + '.tif'
        filepath = filepath + '\\' + filename
        cv2.imwrite(filepath, img)

if os.path.isdir(root):
    ret = []
    ret = findjpg(root)
    print(statisticCharNum(ret))
    other2tif(ret)
else:
    print("This path does not exist!")

