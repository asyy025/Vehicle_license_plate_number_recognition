import cv2
from matplotlib import pyplot as plt

# 显示图片
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()

# plt显示彩色图片 进行通道转换
def plt_show0(img):
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    plt.imshow(img)
    plt.show()

# 显示灰度图片
def plt_show(img):
    plt.imshow(img, cmap='gray')
    plt.show()

# 加载图片
rawimage = cv2.imread(r"duhdbc.jpeg")
plt_show0(rawimage)

# 高斯去噪
image = cv2.GaussianBlur(rawimage, (3, 3), 0)
# 预览效果
plt_show(image)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# sobel算子边缘检测(做一个x方向上的检测)
Sobel_x = cv2.Sobel(gray_image, cv2.CV_16S, 1, 0)# 参数x和y不能同时等于1，必须分开检测
# Sobel_y = cv2.Sobel(gray_image, cv2.CV_16S, 0,1)

#在经过处理后，需要用convertScaleAbs()函数将其转回原来的uint8形式，否则将无法显示图像，而只是一副灰色的窗口。
absX = cv2.convertScaleAbs(Sobel_x)  # 对一个随机数组取绝对值
# absX = cv2.convertScaleAbs(Sobel_y)
# dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
image = absX
plt_show(image)

# 自适应阈值处理
ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
plt_show(image)

# 闭运算，是白色部分练成整体  #闭运算：先膨胀 后腐蚀  作用：去除前景图像的噪点，也可以对前景图像进行连接
kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
print(kernelX)  # 函数生成内核（参数1：内核形状  参数2：内核尺寸  参数3：锚点位置），也可以自定义
image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernelX, iterations=3) #通用函数进行闭运算
plt_show(image)

# 除去一些小白点
kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 19))

# 膨胀， 腐蚀
image = cv2.dilate(image, kernelX)
image = cv2.erode(image, kernelX)

# 腐蚀 膨胀
image = cv2.erode(image, kernelY)
image = cv2.dilate(image, kernelY)

plt_show(image)

# 中值滤波去除噪点  用邻域内的所有像素点的中间值代替当前像素点的中间值
image = cv2.medianBlur(image, 15)
plt_show(image)

# 轮廓jiance
# cv2.RETR_EXTERNAL表示只检测外轮廓
# cv2.CHAIN_APPROX_SIMPLE压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓仅需要4个点来保存轮廓信息
contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 轮廓绘制
imagel = rawimage.copy()
cv2.drawContours(imagel, contours, -1, (0, 255, 0), 5)
plt_show(imagel)

# 筛选出车牌位置轮廓，
# 以车牌长宽比在3:1-4:1之间的判断
for item in contours:
    # cv2.boundingRect用一个最小的矩阵把找到的形状包起来
    rect = cv2.boundingRect(item)
    x = rect[0]
    y = rect[1]
    weight = rect[2]
    height = rect[3]
    # 440mm*140mm
    if (weight > (height*3)) and (weight < (height*4)):
        image = rawimage[y:y + height, x:x + weight]  # roi
        # cv_show('image', image)
        # 图像保存
        img = cv2.rectangle(rawimage, (x, y), (x+weight, y+height), (0, 0, 255), 2)
        plt_show0(img)
        plt_show(image)
        cv2.imwrite(r'D://image_1.png', image)










