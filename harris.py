import cv2  # 读取是BGR格式
import matplotlib.pyplot as plt
import numpy as np


def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


'''
cv2.cornerHarris()
img:数据类型为float32的输入图像
blockSize：角点检测中指定的区域大小
ksize：Sobel求导中使用的窗口大小
k：取值范围为[0.04,0.06]
'''
img = cv2.imread('pic/test_1.jpg')
print(f'img.shape:{img.shape}')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)
print(f'dst.shape:{dst.shape}')

img[dst > 0.1 * dst.max()] = [0, 0, 255]

cv_show('dst',img)
