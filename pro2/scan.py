import cv2  # 读取是BGR格式
import matplotlib.pyplot as plt
import numpy as np
from imutils import contours


def cv_show(img, name):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def order_points(pts):
    # print(pts)
    rect = np.zeros((4, 2), dtype='float32')

    s = pts.sum(axis=1)
    # print(s)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    d = np.diff(pts, axis=1)
    # print(d)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]

    # print(rect)
    return rect


def four_point_transform(img, pts):
    # 得到坐标点(左上，右上，右下，左下)
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # 计算输入的w和h值
    widthTop = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    widthBtm = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthMax = max(int(widthTop), int(widthBtm))

    heightLeft = np.sqrt(((bl[0] - tl[0]) ** 2) + ((bl[1] - tl[1]) ** 2))
    heightRight = np.sqrt(((br[0] - tr[0]) ** 2) + ((br[1] - tr[1]) ** 2))
    heightMax = max(int(heightLeft), int(heightRight))

    dst = np.array([
        [0, 0],
        [widthMax, 0],
        [widthMax, heightMax],
        [0, heightMax]],
        dtype='float32')

    # 计算变换矩阵并透视变换
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (widthMax, heightMax))

    return warped


# 读取图像
img = cv2.imread('../pic/receipt.jpg')
# img = cv2.imread('../pic/page.jpg')
imgCopy = img.copy()
img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
cv_show(img, 'img')

# 预处理
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgGray = cv2.GaussianBlur(imgGray, (5, 5), 0)
cv_show(imgGray, 'imgGray')
imgEdge = cv2.Canny(imgGray, 75, 200)
cv_show(imgEdge, 'imgEdge')

# 轮廓检测
imgBin, imgCnts, hierarchy = cv2.findContours(imgEdge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
imgCnts = sorted(imgCnts, key=cv2.contourArea, reverse=True)[:5]

# 遍历轮廓
for c in imgCnts:
    # 计算轮廓近似
    # 轮廓周长
    cntPeri = cv2.arcLength(c, True)
    # c表示输入的轮廓点集
    # epsilon表示从原始轮廓到近似轮廓的最大距离，是一个准确度参数
    # True表示轮廓是封闭的
    cntApprox = cv2.approxPolyDP(c, 0.02 * cntPeri, True)

    if len(cntApprox) == 4:
        cntScreen = cntApprox
        break

imgDraw = cv2.drawContours(img.copy(), [cntScreen], -1, (0, 0, 255), 3)
cv_show(imgDraw, 'imgDraw')

# 透视变换
imgWarp = four_point_transform(imgCopy, cntScreen.reshape(4, 2) * 5)
cv_show(imgWarp, 'imgWarp')

# 二值化
imgWarp = cv2.cvtColor(imgWarp, cv2.COLOR_BGR2GRAY)
ref = cv2.threshold(imgWarp, 127, 255, cv2.THRESH_BINARY)[1]
cv_show(ref, 'ref')
cv2.imwrite('../pic/scan.jpg',ref)
