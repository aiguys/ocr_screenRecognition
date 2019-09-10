import cv2
import numpy as np
from extr_recoPipline import threshTrackbar

# Step1.
img = cv2.imread('../data/ui.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
line = np.zeros(img.shape, np.uint8)
#cv2.imshow('gray', gray)

#thr, shading = threshTrackbar(gray)
#print(thr, shading)

ret, white = cv2.threshold(gray, 68, 255, cv2.THRESH_BINARY)
black_line = 255 - white
#cv2.imshow('black_line', black_line)

kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
opening = cv2.morphologyEx(black_line, cv2.MORPH_OPEN, kernel_rect)
#cv2.imshow('opening', opening)
def getOpening():
    return opening

l_cnt = []
contours, _ = cv2.findContours(opening, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
for i in range(len(contours)):  # 提取圆周围的文本
    #area = cv2.contourArea(contours[i])
    #if 50 < area < 60000:
        l_cnt.append(contours[i])
        cv2.drawContours(line, contours[i], -1, (255, 255, 255), thickness=4)

#cv2.imshow('line', line)

ROI = cv2.bitwise_and(img, line)
#cv2.imshow("123", line)

def getLine():
    return l_cnt

def getLineImage():
    return line


if __name__ == '__main__':
    #cv2.imshow('ROI', ROI)
    cv2.waitKey(0)
    cv2.destroyAllWindows()