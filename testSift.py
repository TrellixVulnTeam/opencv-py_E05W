import cv2  # 读取是BGR格式
import matplotlib.pyplot as plt
import numpy as np


def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img1 = cv2.imread('pic/left_01.png')
img2 = cv2.imread('pic/right_01.png')

# brute-force蛮力匹配
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
print(np.float32(kp1[1].pt))

# crossChech表示两个特征点要互相匹配
# NORM_L2：归一化数组的欧几里得距离，如果其他特征计算方法需要考虑不同的匹配计算方式
bf = cv2.BFMatcher(crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)
# cv_show('img3', img3)

# k对最佳匹配
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# print(matches)
# print(matches[0][0].distance,matches[1][0].trainIdx, matches[1][0].queryIdx)
# print(matches[0][1].distance,matches[1][1].trainIdx, matches[1][1].queryIdx)

good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)
# print(good)
img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags=2)
# cv_show('img3', img3)
