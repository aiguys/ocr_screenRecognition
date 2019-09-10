# -*- coding: utf-8 -*-#
# Author:       weiz
# Date:         2019/9/5 18:47
# Name:         recoText
# Description: 框出所识别的字符

import cv2
import pytesseract
from PIL import Image

filename = '../data/00102A.jpg'

# read the image and get the dimensions
img = cv2.imread(filename)
h, w, _ = img.shape # assumes color image


# run tesseract, returning the bounding boxes
boxes = pytesseract.image_to_boxes(img) # also include any config options you use

# draw the bounding boxes on the image
for b in boxes.splitlines():
    b = b.split(' ')
    img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)
    cv2.putText(img, b[0], (int(b[1]), h - int(b[2])), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 1)
    print(b[0])

# show annotated image and wait for keypress
cv2.imshow(filename, img)

cv2.waitKey(0)
cv2.destroyAllWindows()