import cv2  # 读取是BGR格式
import matplotlib.pyplot as plt
import numpy as np

'''
ret,dst = cv2.threshold(sec,thresh,maxval,type)
src:输入图
dst:输出图
thresh：阈值
maxval：当像素超过阈值所赋予的值
type:二值化类型的操作有五种
cv2.THRESH_BINARY       超过阈值部分取maxval（最大值）
cv2.THRESH_BINARY_INV   上一个的反转
cv2.THRESH_TRUNC        大于阈值部分设为阈值，否则不变
cv2.THRESH_TOZERO       大于阈值的部分不变，否则设为0
cv2.THRESH_TOZERO_INV   上一个的反转
'''
img_cat_gray = cv2.imread('pic/cat.jpg', cv2.IMREAD_GRAYSCALE)
ret, thresh1 = cv2.threshold(img_cat_gray, 127, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img_cat_gray, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img_cat_gray, 127, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img_cat_gray, 127, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img_cat_gray, 127, 255, cv2.THRESH_TOZERO_INV)

titles = ['original', 'binary', 'binv', 'trunc', 'tozero', 'zinv']
images = [img_cat_gray, thresh1, thresh2, thresh3, thresh4, thresh5]
for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
