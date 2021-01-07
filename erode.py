import cv2  # 读取是BGR格式
import matplotlib.pyplot as plt
import numpy as np

img=cv2.imread('pic/dige.png')
kernel=np.ones((3,3),np.uint8)
erosion = cv2.erode(img,kernel,iterations=1)
cv2.imshow('erosion',erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()

pie=cv2.imread('pic/pie.png')
kernel=np.ones((30,30),np.uint8)
erosion1 = cv2.erode(pie,kernel,iterations=1)
erosion2 = cv2.erode(pie,kernel,iterations=2)
erosion3 = cv2.erode(pie,kernel,iterations=3)
res=np.hstack((erosion1,erosion2,erosion3))
cv2.imshow('res',res)
cv2.waitKey(0)
cv2.destroyAllWindows()

