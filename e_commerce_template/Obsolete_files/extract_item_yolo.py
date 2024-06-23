# GGGGGGGGGOOOOAT
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import torch

# 加载预训练的 YOLO v8 模型
model = YOLO('yolov8m-seg.pt')

# 读取图像
image = cv2.imread('items_2.JPG')

# 设置置信度阈值
confidence_threshold = 0.09  # 根据需要调整阈值

# 进行实例分割
results = model.predict(image, conf=confidence_threshold)

# 检查分割结果
if len(results) == 0 or results[0].masks is None:
    print("No detections or masks found.")
    masks = np.array([])  # 设置 masks 为空数组，避免后续处理时报错
else:
    # 提取分割结果
    masks = results[0].masks.data.cpu().numpy()  # 使用 .data 提取 Tensor 数据并转换为 numpy 数组

# 确保 masks 是正确的 numpy 数组
print(f"Converted masks shape: {masks.shape if masks.size > 0 else 'Empty'}")
print(f"Converted masks type: {type(masks)}")

# 创建输出文件夹
import os
output_dir = 'segmented_objects'
os.makedirs(output_dir, exist_ok=True)

# 如果没有检测到物体，则不进行后续处理
if masks.size == 0:
    print("No masks to process.")
else:
    # 可视化分割结果
    for i, mask in enumerate(masks):
        plt.imshow(mask, cmap='gray', alpha=0.5)
        plt.axis('off')
        plt.show()

    # 后处理操作：形态学操作和连通域分析
    kernel = np.ones((3, 3), np.uint8)  # 定义形态学操作的核

    for i, mask in enumerate(masks):
        # 确保 mask 是 2D 数组
        if mask.ndim > 2:
            mask = mask.squeeze()

        # 形态学操作：先膨胀再腐蚀（闭运算），可以去除小孔洞
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 转换为 uint8 类型
        mask = (mask * 255).astype(np.uint8)

        # 连通域分析，去除小面积的连通域
        num_labels, labels_im = cv2.connectedComponents(mask)

        # 设置一个面积阈值，小于该阈值的连通域将被移除
        min_area = 5  # 根据需要调整阈值
        for label in range(1, num_labels):
            if np.sum(labels_im == label) < min_area:
                mask[labels_im == label] = 0

        # 保存分割对象
        cv2.imwrite(os.path.join(output_dir, f'object_{i}.png'), mask)