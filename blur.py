import cv2  # 读取是BGR格式
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('pic/lenaNoise.png')
# cv2.imshow('img', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 均值滤波
# 平均卷积操作
blur = cv2.blur(img, (3, 3))
# cv2.imshow('blur', blur)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 方框滤波
# 基本与均值一样，可以归一化
# normalize=True时进行归一化，=False时 若值>255将值设为255
box = cv2.boxFilter(img, -1, (3, 3), normalize=True)
# cv2.imshow('box', box)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 高斯滤波
# 高斯模糊的卷积核里的数值是满足高斯分布的，相当于更重视接近中间的
aussian = cv2.GaussianBlur(img, (5, 5), 1)
# cv2.imshow('aussian', aussian)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 中值滤波
# 用中值代替
median = cv2.medianBlur(img, 5)
# cv2.imshow('median', median)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 展示所有
res = np.hstack((blur, aussian, median))
cv2.imshow('res', res)
cv2.waitKey(0)
cv2.destroyAllWindows()
