import cv2  # 读取是BGR格式
import matplotlib.pyplot as plt
import numpy as np


def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# vc = cv2.VideoCapture('pic/test.mp4')
# # 检查打开是否正确
# if vc.isOpened():
#     open, frame = vc.read()
# else:
#     open = False
#
# while open:
#     ret, frame = vc.read()
#     if frame is None:
#         break
#     if ret == True:
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         cv2.imshow('result', gray)
#         if cv2.waitKey(10) & 0xFF == 27:
#             break
# vc.release()
# cv2.destroyAllWindows()

# img = cv2.imread('pic/cat.jpg', cv2.IMREAD_GRAYSCALE) #灰度图
img = cv2.imread('pic/cat.jpg')
print(img)
print(img.shape)
print(type(img))
# cv_show('cat', img[0:50, 0:200])
b, g, r = cv2.split(img)
print(b)

# cur_img=img.copy()
# cur_img[:,:,0]=0
# cur_img[:,:,2]=0
# cv_show('r',cur_img)

top_size, bottom_size, left_size, right_size = (50, 50, 50, 50)
# 复制法，复制最边缘像素
replicate = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_REPLICATE)
# 反射法，将距离边缘近的图像反射
reflect = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_REFLECT)
# 反射法，去掉最边缘的像素
reflect101 = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_REFLECT_101)
# 外包装法，cdrfgh|abcdefgh|abcdrfg
wrap = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_WRAP)
# 常量法，需要指定value值
constant = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_CONSTANT, value=0)

plt.subplot(231), plt.imshow(img, 'gray'), plt.title('original')
plt.subplot(232), plt.imshow(replicate, 'gray'), plt.title('replicate')
plt.subplot(233), plt.imshow(reflect, 'gray'), plt.title('reflect')
plt.subplot(234), plt.imshow(reflect101, 'gray'), plt.title('reflect101')
plt.subplot(235), plt.imshow(wrap, 'gray'), plt.title('constant')
plt.subplot(236), plt.imshow(constant, 'gray'), plt.title('constant')

plt.show()
