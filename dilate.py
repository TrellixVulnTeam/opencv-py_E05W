import cv2  # 读取是BGR格式
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('pic/dige.png')
kernel = np.ones((3, 3), np.uint8)
print(kernel)
erosion = cv2.erode(img, kernel, iterations=1)
# cv2.imshow('erosion',erosion)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

kernel = np.ones((3, 3), np.uint8)
dige_dilate = cv2.dilate(erosion, kernel, iterations=3)
# cv2.imshow('dilate',dige_dilate)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 开运算
# 先腐蚀，再膨胀
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
# cv2.imshow('opening', opening)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 闭运算
# 先膨胀，后腐蚀
kernel = np.ones((3, 3), np.uint8)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
# cv2.imshow('closing', closing)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 梯度=膨胀-腐蚀
pie = cv2.imread('pic/pie.png')
kernel = np.ones((3, 3), np.uint8)
gradient = cv2.morphologyEx(pie, cv2.MORPH_GRADIENT, kernel)
# cv2.imshow('gradient', gradient)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 礼帽=原始输入-开运算结果
kernel = np.ones((3, 3), np.uint8)
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
# cv2.imshow('tophat', tophat)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 黑帽=闭运算-原始输入
kernel = np.ones((5, 5), np.uint8)
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
cv2.imshow('blackhat', blackhat)
cv2.waitKey(0)
cv2.destroyAllWindows()
