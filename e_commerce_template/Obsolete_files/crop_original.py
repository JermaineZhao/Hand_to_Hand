# # # This file 还原 the masks to the original picture to get a restangular image of the items
# # import os
# # import cv2
# # import numpy as np

# # # 加载原图像和所有mask图像
# # def load_image_and_masks(image_path, mask_folder):
# #     image = cv2.imread(image_path)
# #     masks = []
# #     filenames = []
# #     for filename in os.listdir(mask_folder):
# #         if filename.endswith(".png"):
# #             filepath = os.path.join(mask_folder, filename)
# #             mask = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
# #             mask = mask / 255  # 将mask转换为二进制
# #             masks.append(mask)
# #             filenames.append(filename)
# #     return image, masks, filenames

# # # 计算mask的最小外接矩形
# # def get_bounding_rect(mask):
# #     coords = np.column_stack(np.where(mask > 0))
# #     x, y, w, h = cv2.boundingRect(coords)
# #     return x, y, w, h

# # # 在原图像中裁剪mask的最小外接矩形区域
# # def crop_image_with_mask(image, mask):
# #     x, y, w, h = get_bounding_rect(mask)
# #     cropped_image = image[y:y+h, x:x+w]
# #     return cropped_image

# # # 保存裁剪后的图像
# # def save_cropped_images(output_folder, cropped_images, filenames):
# #     os.makedirs(output_folder, exist_ok=True)
# #     for cropped_image, filename in zip(cropped_images, filenames):
# #         filepath = os.path.join(output_folder, filename)
# #         cv2.imwrite(filepath, cropped_image)

# # # 主函数
# # def main(image_path, mask_folder, output_folder):
# #     image, masks, filenames = load_image_and_masks(image_path, mask_folder)
# #     cropped_images = []
    
# #     for mask in masks:
# #         cropped_image = crop_image_with_mask(image, mask)
# #         cropped_images.append(cropped_image)
    
# #     save_cropped_images(output_folder, cropped_images, filenames)

# # # 使用示例
# # image_path = '/Users/jermainezhao/hand_hand/items_2.JPG'  # 替换为你的原图像路径
# # mask_folder = '/Users/jermainezhao/hand_hand/segmented_objects/filtered'  # 替换为你的mask文件夹路径
# # output_folder = '/Users/jermainezhao/hand_hand/segmented_objects/items'  # 替换为你希望保存结果的文件夹路径
# # main(image_path, mask_folder, output_folder)
# import cv2
# import numpy as np

# # 计算mask的最小外接矩形
# def get_bounding_rect(mask):
#     coords = np.column_stack(np.where(mask > 0))
#     x, y, w, h = cv2.boundingRect(coords)
#     return x, y, w, h

# # 在原图像中裁剪mask的最小外接矩形区域
# def crop_image_with_mask(image, mask):
#     x, y, w, h = get_bounding_rect(mask)
#     cropped_image = image[y:y+h, x:x+w]
#     return cropped_image

# # 主函数
# def main(original_image_path, mask_image_path, output_image_path):
#     # 读取原图像和mask图像
#     original_image = cv2.imread(original_image_path)
#     mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
    
#     # 调整mask图像大小以匹配原图像大小
#     mask_image = cv2.resize(mask_image, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
#     # 将mask图像二值化
#     _, mask_binary = cv2.threshold(mask_image, 127, 255, cv2.THRESH_BINARY)
    
#     # 裁剪图像
#     cropped_image = crop_image_with_mask(original_image, mask_binary)

#     # 保存裁剪后的图像
#     cv2.imwrite(output_image_path, cropped_image)

# # 使用示例
# original_image_path = '/Users/jermainezhao/hand_hand/items_2.JPG' # 原图像路径
# mask_image_path = '/Users/jermainezhao/hand_hand/segmented_objects/filtered/object_1.png'  # mask图像路径
# output_image_path = 'cropped_image.png'  # 保存裁剪后的图像路径

# main(original_image_path, mask_image_path, output_image_path)

import cv2
import numpy as np
from matplotlib import pyplot as plt

mask_path = '/Users/jermainezhao/hand_hand/segmented_objects/filtered/object_1.png'
mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# Load the image to be masked
image_to_mask_path = 'items_2.JPG'  # replace with your image path
image_to_mask = cv2.imread(image_to_mask_path)

# Resize the mask to match the image size
mask_resized = cv2.resize(mask_image, (image_to_mask.shape[1], image_to_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

# Threshold the mask image to binary
_, binary_mask = cv2.threshold(mask_resized, 127, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get the bounding box of the largest contour
x, y, w, h = cv2.boundingRect(contours[0])

# Create a new mask with the bounding rectangle
new_mask = np.zeros_like(mask_resized)
pad = 200
# cv2.rectangle(new_mask, (x - pad, y - pad), (x + w + pad, y + h + pad), 255, -1)  # Filled rectangle

if w < h:
    if w/h < 3/4:
        x_1 = x + w/2 - 3*h/8
        y_1 = y
        w_1 = 3*h/4
        h_1 = h
    else:
        x_1 = x
        y_1 = y + h/2 - 2*w/4
        w_1 = w
        h_1 = 4*w/3
else:
    if w/h > 4/3:
        x_1 = x
        y_1 = y + h/2 - 2*w/4
        w_1 = w
        h_1 = 4*w/3
    else:
        x_1 = x + w/2 - 3*h/8
        y_1 = y
        w_1 = 3*h/4
        h_1 = h

x_1 = int(x_1)
y_1 = int(y_1)
w_1 = int(w_1)
h_1 = int(h_1)
cv2.rectangle(new_mask, (x_1 - pad, y_1 - pad), (x_1 + w_1 + pad, y_1 + h_1 + pad), 255, -1)  # Filled rectangle


# Apply the new mask to the image
masked_image = cv2.bitwise_and(image_to_mask, image_to_mask, mask=new_mask)

# Display the original image, the mask, and the masked image
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image_to_mask, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('New Mask')
plt.imshow(new_mask, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Masked Image')
plt.imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()