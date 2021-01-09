import cv2  # 读取是BGR格式
import matplotlib.pyplot as plt
import numpy as np


def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


'''
Canny边缘检测
1）使用高斯滤波器，平滑图像，消除噪音
2）计算图像中每个像素点的梯度强度和方向（用Sobel算子）
3）应用非极大值抑制，消除边缘检测带来的杂散相应
    采用线性插值法
4）应用双阈值检测来确定真实的和潜在的边缘
    梯度值>manVal:保留为边界
    minVal<梯度值<maxVal:连有边界则保留，否则舍弃
    梯度值<minVal:舍弃
5）通过抑制孤立的弱边缘最终完成边缘检测
'''
# img = cv2.imread('pic/lena.jpg', cv2.IMREAD_GRAYSCALE)
# v1 = cv2.Canny(img, 80, 150)
# v2 = cv2.Canny(img, 50, 100)
# res = np.hstack((v1, v2))
# cv_show('res', res)

img = cv2.imread('pic/car.png', cv2.IMREAD_GRAYSCALE)
v1 = cv2.Canny(img, 120, 250)
v2 = cv2.Canny(img, 50, 100)
res = np.hstack((v1, v2))
cv_show('res', res)