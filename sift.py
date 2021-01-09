import cv2  # 读取是BGR格式
import matplotlib.pyplot as plt
import numpy as np


def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread('pic/test_1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 找到特征点
sift = cv2.SIFT_create()
kp = sift.detect(gray, None)

img = cv2.drawKeypoints(gray, kp, img)
cv_show('img', img)

# 计算特征
kp, des = sift.compute(gray, kp)
print(np.array(kp).shape)
print(des.shape)
