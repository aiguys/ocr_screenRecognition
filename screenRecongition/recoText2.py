# -*- coding: utf-8 -*-#
# Author:       weiz
# Date:         2019/9/9 16:16
# Name:         recoText2
# Description:  框出所识别的字符

import cv2
import pytesseract
from pytesseract import Output

filename = '../data/00102A.jpg'
img = cv2.imread(filename)
d = pytesseract.image_to_data(img, output_type=Output.DICT, lang='hwsoft')
print((d['text'][-1]))

for i in range(len(d['top'])):
    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    print(x, y, w, h)
    if i == 4:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

#print(pytesseract.image_to_data(img, lang='hwsoft'))
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()