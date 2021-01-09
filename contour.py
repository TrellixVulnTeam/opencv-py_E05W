import cv2  # 读取是BGR格式
import matplotlib.pyplot as plt
import numpy as np


def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


'''
findContours(img,mode,method)
    mode:轮廓检测模式
        RETR_EXTERNAL:只检索最外面的轮廓
        RETR_LIST:检索所有的轮廓，并将其保存到一条链表中
        RETR_CCOMP:检索所有的轮廓，并将它们组织为两层：顶层是各部分的外部边界，第二次是空洞的边界
        RETR_TREE:检索所有轮廓，并重构嵌套轮廓的整个层次
    method:轮廓逼近方法
        CHAIN_APPROX_NONE:以Freeman链码的方式输出轮廓，所有其他方法输出多边形（顶点的序列）
        CHAIN_APPROX_SIMPLE:压缩水平的、处置的和斜的部分，函数只保留他们的终点部分
'''
img = cv2.imread('pic/contours2.png')
# cv_show('img', img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)  # 转换成二值图
cv_show('thresh', thresh)

'''
binary:二值化的结果
contours:轮廓信息
hierarchy:层级
'''
binary, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# 绘制轮廓:传入图像，轮廓，轮廓索引，颜色模式，线条厚度
draw_img = img.copy()
# res = cv2.drawContours(draw_img, contours, -1, (0, 0, 255), 2)
# cv_show('res', res)

cnt = contours[0]
# print(cv2.contourArea(cnt))  # 面积
# print(cv2.arcLength(cnt, True))  # 周长

# 轮廓近似
epsilon = 0.1 * cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt, epsilon, True)
print([approx])
print(approx)
res = cv2.drawContours(draw_img, [approx], -1, (0, 0, 255), 2)
cv_show('res', res)
