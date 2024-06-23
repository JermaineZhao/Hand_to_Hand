import numpy as np
import cv2
from tensorflow.keras.models import load_model
import tensorflow as tf

# 读取图像
image = cv2.imread('items.JPG')

# 预处理图像
input_image = cv2.resize(image, (256,256))
input_image = np.expand_dims(input_image, axis=0) / 255.0

# 假设你有一个预训练的 U-Net 模型，模型应已保存为 'unet_model.h5'
# 如果没有预训练模型，可以在公开数据集上进行训练，例如 COCO 数据集
model = load_model('unet_model.h5')

# 进行预测
predicted_mask = model.predict(input_image)
predicted_mask = (predicted_mask > 0.5).astype(np.uint8)

# 转换为二值图像
predicted_mask = predicted_mask[0, :, :, 0] * 255
predicted_mask = cv2.resize(predicted_mask, (image.shape[1], image.shape[0]))

# 查找轮廓
contours, _ = cv2.findContours(predicted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 创建一个文件名前缀
prefix = "item_"

# 分割和保存每个轮廓对应的物体
for i, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)
    # 过滤掉面积过小的轮廓
    if w * h > 500:  # 调整这个值以适应你物体的实际大小
        cropped = image[y:y+h, x:x+w]
        cv2.imwrite(f"{prefix}{i+1}.png", cropped)

print("分割完成，每个物体已单独保存为图片。")