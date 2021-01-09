import cv2  # 读取是BGR格式
import matplotlib.pyplot as plt
import numpy as np


def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


'''
高斯金字塔：向下采样方法（缩小）
    |1  4  6  4  1| 
    |4 16 24 16  4|     1）将gi与高斯内核卷积
1/16|6 24 36 24  6|     2）将所有偶数行和列去除
    |4 16 24 16  4|
    |1  4  6  4  1| 
    
高斯金字塔：向上采样方法（放大）
| 10 30 |    |10 0 30  0|
| 56 96 | -> |0  0  0  0|   1)将图像在每个方向扩大为原来的两倍，新增的行和列用0填充
             |56 0  96 0|   2)使用先前同样的内核（乘以4）与放大后的图像卷积，获得近似值
             |0  0  0  0|
'''

img=cv2.imread('pic/AM.png')
cv_show('img',img)
print(img.shape)

up=cv2.pyrUp(img)
cv_show('up',up)
print(up.shape)
