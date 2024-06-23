import cv2
import numpy as np

# 读取图像
image = cv2.imread('items.JPG')

# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用高斯模糊来减少噪声
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 使用阈值分割
_, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)

# 查找轮廓
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 创建一个文件名前缀
prefix = "item_"

# 分割和保存每个轮廓对应的物体
for i, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)
    # 过滤掉面积过小的轮廓
    if w * h > 5000:  # 调整这个值以适应你物体的实际大小
        cropped = image[y:y+h, x:x+w]
        cv2.imwrite(f"/Users/jermainezhao/hand_hand/{prefix}{i+1}.png", cropped)

print("分割完成，每个物体已单独保存为图片。")