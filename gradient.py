import cv2  # 读取是BGR格式
import matplotlib.pyplot as plt
import numpy as np


def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


pie = cv2.imread('pic/pie.png')
# cv_show('pie', pie)

'''
Sobel算子
    |-1  0   1|       |-1  -2  -1|
gx= |-2  0   2| , gy= | 0   0   0|
    |-1  0   1|       | 1   2   1|
cv2.Sobel(src,ddepth,dx,dy,ksize)
ddepth:图像深度
dx和dy分别表示水平和竖直方向
ksize是Sobel算子的大小
sobelx = cv2.Sobel(pie, cv2.CV_64F, 1, 0, ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)
cv_show('sobelx', sobelx)
'''

# sobely = cv2.Sobel(pie, cv2.CV_64F, 0, 1, ksize=3)
# sobely = cv2.convertScaleAbs(sobely)
# cv_show('sobely', sobely)

# sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
# cv_show('sobelxy', sobelxy)

img = cv2.imread('pic/lena.jpg', cv2.IMREAD_GRAYSCALE)
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobely = cv2.convertScaleAbs(sobely)
sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
# cv_show('sobelxy', sobelxy)
# 不建议直接计算，应该分别计算x
# sobelxy1 = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=3)
# sobelxy1 = cv2.convertScaleAbs(sobelxy1)
# cv_show('sobelxy1', sobelxy1)

'''
Scharr算子
扩大Sobel算子的倍数
    |-3   0    3|       |-3  -10  -3|
gx= |-10  0   10| , gy= | 0   0    0|
    |-3   0    3|       | 3   10   3|
'''
scharrx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
scharrx = cv2.convertScaleAbs(scharrx)
scharry = cv2.Scharr(img, cv2.CV_64F, 0, 1)
scharry = cv2.convertScaleAbs(scharry)
scharrxy = cv2.addWeighted(scharrx, 0.5, scharry, 0.5, 0)
# cv_show('scharrxy', scharrxy)

'''
laplacian算子
    |0  1  0|
G = |1 -4  1|
    |0  1  0|
'''
laplacian = cv2.Laplacian(img, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)

res = np.hstack((sobelxy, scharrxy, laplacian))
cv_show('res', res)
