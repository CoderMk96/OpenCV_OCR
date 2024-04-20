from PIL import Image #  Python 中用于处理图像的强大库
import cv2
import pytesseract # ocr字符识别库
import os # 提供了许多与操作系统交互的函数


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
cv2.imshow("Output",gray)
cv2.waitKey(0)


