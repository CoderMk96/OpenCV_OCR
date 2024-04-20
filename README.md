### 项目简介

#### 软件功能

将照片中的页面进行投影变换成正面，然后通过第三方开源OCR字符识别库pytesseract对照片中文本进行识别。

### 效果图

![](C:\Users\24111\AppData\Roaming\marktext\images\2024-04-20-16-56-59-image.png)

![](C:\Users\24111\AppData\Roaming\marktext\images\2024-04-20-16-57-32-image.png)

### 安装说明

python版本 3.6.4

opencv版本 3.4.1

pytesseract版本0.3.8：通过[Release v0.3.8 · madmaze/pytesseract (github.com)](https://github.com/madmaze/pytesseract/releases/tag/v0.3.8)下载源码进行安装 cd pytesseract && pip install -U .

安装pytesseract软件windows版（根目录上有安装包）

配置环境变量：

1. pytesseract软件目录

2. TESSDATA_PREFIX的环境变量，设置为安装目录下的tessdata目录   
   如:`D:\Program Files (x86)\Tesseract-OCR\tessdata`

### 更新日志

#### V1.0.0 版本

第一步 边缘检测

- 读取原始图像，根据比例放大

- 预处理：灰度

- 高斯模糊（降低图像噪音，使图像的边缘更平滑）

- Canny边缘检测算法

第二步 轮廓检测

- 获取边缘检测后图像中的轮廓

- 根据面积排序，取出前五的轮廓

- 遍历轮廓，对当前轮廓进行多边形逼近（当前轮廓的周长*0.02为原始轮廓到近似轮廓的最大距离）。当得出的轮廓是一个四边形则跳出循环，并将该轮廓保存下来。

第三步 投影变化

- 将坐标根据 左上、右上、右下、左下 进行排序

- 根据两点间距离公式（根号 x平方 + y平方），计算出w和h值。

- 以左上为(0,0)根据h和w得出变换后的四个坐标位置。

- 计算变换矩阵

- 将原始图像根据变换矩阵进行透视变换

第四步 调用第三方OCR库

- 对图像进行灰度

- 中值滤波（去除图像中的噪点，椒盐噪点和斑点噪点）

- 将图像保存到本地，生成临时图像文件（以进程号为文件名）

- 调用Image库打开图像，传入pytesseract接口

- 得出识别后的文字信息




