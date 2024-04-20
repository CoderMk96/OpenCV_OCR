import numpy as np
import argparse  # 用于解析命令行参数的模块
import cv2

from PIL import Image #  Python 中用于处理图像的强大库
import cv2
import pytesseract # ocr字符识别库
import os # 提供了许多与操作系统交互的函数

def order_points(pts):
    # 创建一个名为 rect 的形状为 (4, 2) 的零矩阵，用于存储排序后的四个坐标点
    rect = np.zeros((4,2),dtype="float32")

    # 按顺序找到对应坐标0123 分别是左上，右上，右下，左下
    # 计算左上，右下
    # axis=1 表示沿着列的方向进行操作
    s = pts.sum(axis = 1) # 包含每个点 x 和 y 坐标之和的数组 s
    rect[0] = pts[np.argmin(s)] # s 数组中的最小值对应的索引
    rect[2] = pts[np.argmax(s)] # s 数组中的最大值对应的索引

    # 计算右上和左下
    diff = np.diff(pts,axis=1) # 包含每个点 x 和 y 坐标之差的数组 diff
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def order_points(pts):
	# 一共4个坐标点
	rect = np.zeros((4, 2), dtype = "float32")

	# 按顺序找到对应坐标0123分别是 左上，右上，右下，左下
	# 计算左上，右下
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	# 计算右上和左下
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	return rect

# 透视变换
def four_point_transform(image,pts):
    # 获取输入坐标点
    rect = order_points(pts)
    (tl,tr,br,bl) = rect

    # 计算输入的w和h值
    # 两点之间距离公式 = 根号 x平方 + y平方
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA),int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA),int(heightB))

    # 变换后的对应坐标位置
    dst = np.array([
        [0,0],
        [maxWidth-1,0],
        [maxWidth-1, maxHeight-1],
        [0,maxHeight-1]], dtype = "float32")

    # 计算变换矩阵
    M = cv2.getPerspectiveTransform(rect,dst)
    # 将原始图像 image 根据变换矩阵 M 进行透视变换
    warped = cv2.warpPerspective(image, M, (maxWidth,maxHeight))

    # 返回变换后结果
    return warped

def resize(image,width=None,height=None,inter=cv2.INTER_AREA):
    dim = None
    (h,w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r),height)
    else:
        r = width / float(w)
        dim = (width,int(h * r))
    resized = cv2.resize(image,dim,interpolation=inter)
    return resized

# 绘图展示
def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# receipt.jpg
image = cv2.imread("D:\code_study\opencv_study\ocr_text2\images\page.jpg")
# 坐标也会相同变化
ratio = image.shape[0] / 500.0  # image.shape[0] 获取图像的高度
orig = image.copy()

image = resize(orig,height=500)

# 预处理
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv_show('gray',gray)
# 高斯模糊可以降低图像的噪声，并使图像中的边缘更加平滑
# 对图像中的每个像素进行加权平均，从而降低图像中的高频噪声。
# 通过给予距离中心像素较近的像素更高的权重，距离较远的像素较低的权重，来进行平滑处理。
# 使得图像在平滑的同时，边缘信息得以保留
gray = cv2.GaussianBlur(gray,(5,5),0)
cv_show('GaussianBlur',gray)
#  Canny 边缘检测算法检测图像中的边缘
# 检测图像中的边缘，并输出二值化的图像，其中边缘像素值为白色，非边缘像素值为黑色
edged = cv2.Canny(gray,75,200)
cv_show('Canny',edged)

print("STEP 1: 边缘检测")

# 轮廓检测
# cv2.RETR_LIST 表示检测所有的轮廓
# [1]只需要第二个返回值，即轮廓列表
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[1]
# key=cv2.contourArea() 表示排序的关键字是轮廓的面积，即按照轮廓面积进行排序。
# reverse=True 表示按照降序排列，即面积最大的轮廓排在前面。
# [:5] 表示只取排序后的前五个轮廓
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

# 遍历轮廓
for c in cnts:
    # 函数来计算轮廓的周长，第二个参数 True 表示轮廓是闭合的
    peri = cv2.arcLength(c,True)
    # 对当前轮廓进行多边形逼近，以减少顶点数
    # c 表示输入的轮廓
    # epsilon 表示从原始轮廓到近似轮廓的最大距离，它是一个准确度参数
    # True 表示轮廓是闭合的
    approx = cv2.approxPolyDP(c, 0.02*peri, True)

    # 4个点的时候就拿出来，逼近后的轮廓是否是一个四边形
    if len(approx) == 4:
        screenCnt = approx
        break

# 展示结果
print("STEP 2: 获取轮廓")
cv2.drawContours(image,[screenCnt],-1,(0,255,0),2)
cv_show("Outline",image)

# 透视变换
warped = four_point_transform(orig,screenCnt.reshape(4,2)*ratio)

# 二值化处理
warped = cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)
ref = cv2.threshold(warped,100,255,cv2.THRESH_BINARY)[1] # 将大于阈值的像素值设为最大像素值，小于等于阈值的像素值设为0
cv2.imwrite('scan.jpg',ref)

# 展示结果
print("STEP 3: 变换")
cv2.imshow("Origianl",resize(orig,height=650))
cv2.imshow("Scanned",resize(ref,height=650))
cv2.waitKey(0)
cv2.destroyAllWindows()


print("STEP 4: 调用第三方ocr库")
preprocess = 'blur' # thresh

image = cv2.imread('scan.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

if preprocess == "thresh":
    gray = cv2.threshold(gray,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

if preprocess == "blur":
    # 中值滤波,用于去除图像中的噪声,去除椒盐噪声和斑点噪声
    # 3 表示滤波器的大小，这里指的是滤波器的卷积核尺寸，通常是一个奇数，例如 3、5、7 等
    gray = cv2.medianBlur(gray,3)

filename = "{}.png".format(os.getpid()) # 临时文件
cv2.imwrite(filename,gray)

text = pytesseract.image_to_string(Image.open(filename))
print(text)
os.remove(filename)

cv2.imshow("Image",image)
cv2.imshow("medianBlur",gray)
cv2.waitKey(0)




