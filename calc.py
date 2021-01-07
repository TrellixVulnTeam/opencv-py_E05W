import cv2  # 读取是BGR格式
import matplotlib.pyplot as plt
import numpy as np

img_cat = cv2.imread('pic/cat.jpg')
img_dog = cv2.imread('pic/dog.jpg')

img_dog = cv2.resize(img_dog, (500, 414))
print(img_dog.shape)

res = cv2.addWeighted(img_dog, 0.4, img_cat, 0.6, 0)
plt.imshow(res)
plt.show()
