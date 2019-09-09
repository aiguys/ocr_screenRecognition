import cv2
import numpy as np
from screenRecongition import black_extr

# Step1. 转换为HSV
img = cv2.imread('../data/ui.jpg')
hue_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
back = np.zeros(img.shape, np.uint8)


# Step2. 用颜色分割图像
green_low_range = np.array([35, 43, 46])
green_high_range = np.array([77, 255, 255])
mask = cv2.inRange(hue_image, green_low_range, green_high_range)
#cv2.imshow('mask', mask)


# Step3. 形态学运算，开运算
kernel_ell = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
mask_opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_ell)
#cv2.imshow('mask_opening', mask_opening)

l_cil = []
l_rect = []

# Step4. 使用面积筛选
contours, hierarch = cv2.findContours(mask_opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
for i in range(len(contours)):  # 提取圆周围的文本
    area = cv2.contourArea(contours[i])
    if 300 < area < 380:
        (x, y), radius = cv2.minEnclosingCircle(contours[i])
        center = (int(x), int(y))
        radius = int(radius)
        l_cil.append([int(x), int(y), int(radius)+5])
        cv2.circle(back, center, radius + 25, (255, 255, 255), -1)
        cv2.circle(back, center, radius, (0, 0, 0), -1)
        ROI = cv2.bitwise_and(img, back)


for i in range(len(contours)):  # 提取矩形周围的文本
    area = cv2.contourArea(contours[i])
    if 420 < area < 460:
        x, y, w, h = cv2.boundingRect(contours[i])
        l_rect.append([x-2, y-2, w+4, h+4])
        cv2.rectangle(back, (x - 20, y - 20), (x + w + 20, y + h + 20), (255, 255, 255), -1)
        cv2.rectangle(back, (x, y), (x + w, y + h), (0, 0, 0), -1)
        ROI = cv2.bitwise_and(img, back)

#cv2.imshow('img', img)
#cv2.imshow('back', back)
#cv2.imshow('ROI', ROI)

gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
#cv2.imshow('gray', gray)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
#cv2.imshow('thresh', thresh)
thresh_1 = 255 - thresh
#cv2.imshow('thresh_1', thresh_1)
thresh_2 = cv2.morphologyEx(thresh_1, cv2.MORPH_CLOSE, kernel_rect)
#cv2.imshow('thresh_2', thresh_2)


dilation = cv2.dilate(thresh, kernel_rect, iterations=1)
#cv2.imshow('dilation', dilation)
erosion = cv2.erode(dilation, kernel_rect, iterations=1)
#cv2.imshow('erosion', erosion)

ret = cv2.bitwise_and(thresh_2, erosion)
#cv2.imshow('ret', ret)

l_line = black_extr.getLine()
cv2.drawContours(ret, l_line, -1, (0, 0, 0), thickness=3)
cv2.imshow('ret2', ret)

ret_opening = cv2.morphologyEx(ret, cv2.MORPH_OPEN, kernel_ell)
#cv2.imshow('ret_opening0', ret_opening)

for i in range(len(l_cil)):
    cv2.circle(ret_opening, (l_cil[i][0], l_cil[i][1]), l_cil[i][2], (0, 0, 0), -1)

for i in range(len(l_rect)):
    cv2.rectangle(ret_opening, (l_rect[i][0], l_rect[i][1]),
                  (l_rect[i][0] + l_rect[i][2], l_rect[i][1] + l_rect[i][3]), (0, 0, 0), -1)
#cv2.imshow('ret_opening1', ret_opening)

contours, hierarchy = cv2.findContours(ret_opening, 1, 2)
for i in range(len(contours)):
    x, y, w, h = cv2.boundingRect(contours[i])
    cv2.rectangle(ret_opening, (x, y), (x+w, y+h), (255, 255, 255), -1)
#cv2.imshow('ret_opening1', ret_opening)

color_ret = cv2.cvtColor(ret_opening, cv2.COLOR_GRAY2BGR)
fin_ret = cv2.bitwise_and(img, color_ret)
cv2.imshow('fin_ret', fin_ret)

cv2.waitKey(0)
cv2.destroyAllWindows()