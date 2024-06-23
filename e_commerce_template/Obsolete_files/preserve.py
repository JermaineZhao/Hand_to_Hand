# This file operate the folder of masks, and preserve all different masks
import os
import cv2
import numpy as np

# 计算两个图像之间的IoU（交并比）
def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

# 加载所有mask图像
def load_masks(folder_path):
    masks = []
    filenames = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            filepath = os.path.join(folder_path, filename)
            mask = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            mask = mask / 255  # 将图像转换为二进制
            masks.append(mask)
            filenames.append(filename)
    return masks, filenames

# 筛选相似度过高的mask图像
def filter_similar_masks(masks, filenames, iou_threshold=0.5):
    filtered_masks = []
    filtered_filenames = []
    for i in range(len(masks)):
        is_similar = False
        for j in range(i):
            if calculate_iou(masks[i], masks[j]) > iou_threshold:
                is_similar = True
                break
        if not is_similar:
            filtered_masks.append(masks[i])
            filtered_filenames.append(filenames[i])
    return filtered_masks, filtered_filenames

# 保存筛选后的mask图像
def save_filtered_masks(folder_path, filtered_masks, filtered_filenames):
    output_folder = os.path.join(folder_path, "filtered")
    os.makedirs(output_folder, exist_ok=True)
    for mask, filename in zip(filtered_masks, filtered_filenames):
        filepath = os.path.join(output_folder, filename)
        cv2.imwrite(filepath, mask * 255)  # 将二进制图像转换回灰度图保存

# 主函数
def main(folder_path):
    masks, filenames = load_masks(folder_path)
    filtered_masks, filtered_filenames = filter_similar_masks(masks, filenames)
    save_filtered_masks(folder_path, filtered_masks, filtered_filenames)

# 使用示例
folder_path = '/Users/jermainezhao/hand_hand/segmented_objects'  # 替换为你的mask文件夹路径
main(folder_path)