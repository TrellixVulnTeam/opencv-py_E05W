import cv2  # 读取是BGR格式
import matplotlib.pyplot as plt
import numpy as np
from imutils import contours
import myutils


def cv_show(img, name):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 读取模板
img = cv2.imread('../pic/ocr_a_reference.png')
cv_show(img, 'img')
# print(img.shape)

# 灰度
ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv_show(ref, 'ref')

# 二值化
ref = cv2.threshold(ref, 127, 255, cv2.THRESH_BINARY_INV)[1]
cv_show(ref, 'ref')

# 计算轮廓
ref_, refCnts, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, refCnts, -1, (0, 0, 255), 3)
cv_show(img, 'img')
# print(np.array(refCnts).shape)
refCnts = myutils.sort_contours(refCnts)[0]
digits = {}

# 遍历每一个轮廓
for (i, c) in enumerate(refCnts):
    # 计算外接矩形且resize成合适大小
    (x, y, w, h) = cv2.boundingRect(c)
    roi = ref[y:y + h, x:x + w]
    roi = cv2.resize(roi, (57, 88))

    # 每个数字对应一个模板
    digits[i] = roi

# 初始化卷积核
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# 读取输入图像
image = cv2.imread('../pic/credit_card_05.png')
cv_show(image, 'image')
image = myutils.resize(image, width=300)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv_show(gray, 'gray')

# 礼帽操作（原始输入-开运算结果）
# opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, rectKernel)
# cv_show(opening, 'opening')
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
cv_show(tophat, 'tophat')

#x方向梯度检测
gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradX = np.absolute(gradX)
# print(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
gradX = gradX.astype('uint8')

print(np.array(gradX).shape)
cv_show(gradX, 'gradX')

# 闭操作（先膨胀，后腐蚀）
closing = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
cv_show(closing, 'closing')

# THRESH_OTSU会自动寻找合适的阈值，适合双峰，需把阈值设为0
thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_OTSU)[1]
cv_show(thresh, 'thresh')

# 闭操作
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rectKernel)
cv_show(closing, 'closing')

# 计算轮廓
closing_, closingCnts, hierarchy = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cnts = closingCnts
cur_img = image.copy()
cv2.drawContours(cur_img, cnts, -1, (0, 0, 255), 2)
cv_show(cur_img, 'img')
locs = []

# 遍历轮廓
for (i, c) in enumerate(cnts):
    (x, y, w, h) = cv2.boundingRect(c)
    # cur_img = image.copy()
    # cur_img = cv2.rectangle(cur_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # cv_show(cur_img, 'img')
    ar = w / float(h)

    # 筛选符合条件的轮廓
    if ar > 2.5 and ar < 4:
        if (w > 40 and w < 55) and (h > 10 and h < 20):
            locs.append((x, y, w, h))

# 将符合的轮廓从左到右排序
locs = sorted(locs, key=lambda x: x[0])
output = []

# 遍历每个轮廓中的数字
for (i, (gX, gY, gW, gH)) in enumerate(locs):
    groupOutput = []

    # 根据坐标提取每一个组
    group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
    cv_show(group, 'group')

    # 预处理（二值化）
    group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv_show(group, 'group')

    # 计算每一组的轮廓
    group_, digitCnts, hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digitCnts = contours.sort_contours(digitCnts, method='left-to-right')[0]

    for c in digitCnts:
        # 找到当前数值的轮廓，resize成合适的大小
        (x, y, w, h) = cv2.boundingRect(c)
        roi = group[y:y + h, x:x + w]
        roi = cv2.resize(roi, (57, 88))
        cv_show(roi, 'roi')

        # 计算匹配得分
        scores = []

        # 在模板中计算每一个得分
        for (digit, digitROI) in digits.items():
            # 模板匹配
            result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)

        # 找到最合适的数字
        groupOutput.append(str(np.argmax(scores)))

    cv2.rectangle(image, (gX - 5, gY - 5), (gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
    cv2.putText(image, ''.join(groupOutput), (gX, gY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    output.extend(groupOutput)

cv_show(image, 'image')
