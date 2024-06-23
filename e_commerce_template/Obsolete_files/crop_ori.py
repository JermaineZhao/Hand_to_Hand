# 批量处理图片，获得mask
import os
import cv2
import numpy as np

def load_image_and_masks(image_path, mask_folder):
    image = cv2.imread(image_path)
    masks = []
    filenames = []
    for filename in os.listdir(mask_folder):
        if filename.endswith(".png"):
            filepath = os.path.join(mask_folder, filename)
            mask = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            masks.append(mask)
            filenames.append(filename)
    return image, masks, filenames

def get_bounding_rect(mask):
    coords = np.column_stack(np.where(mask > 0))
    x, y, w, h = cv2.boundingRect(coords)
    return x, y, w, h

def process_mask(image, mask, pad=200):
    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    _, binary_mask = cv2.threshold(mask_resized, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None

    x, y, w, h = cv2.boundingRect(contours[0])
    new_mask = np.zeros_like(mask_resized)

    if w < h:
        if w / h < 3 / 4:
            x_1 = x + w / 2 - 3 * h / 8
            y_1 = y
            w_1 = 3 * h / 4
            h_1 = h
        else:
            x_1 = x
            y_1 = y + h / 2 - 2 * w / 4
            w_1 = w
            h_1 = 4 * w / 3
    else:
        if w / h > 4 / 3:
            x_1 = x
            y_1 = y + h / 2 - 2 * w / 4
            w_1 = w
            h_1 = 4 * w / 3
        else:
            x_1 = x + w / 2 - 3 * h / 8
            y_1 = y
            w_1 = 3 * h / 4
            h_1 = h

    x_1 = int(x_1)
    y_1 = int(y_1)
    w_1 = int(w_1)
    h_1 = int(h_1)
    cv2.rectangle(new_mask, (x_1 - pad, y_1 - pad), (x_1 + w_1 + pad, y_1 + h_1 + pad), 255, -1)
    
    return new_mask

def crop_non_black(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero(gray)
    x, y, w, h = cv2.boundingRect(coords)
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image

def save_images(output_folder, images, filenames):
    os.makedirs(output_folder, exist_ok=True)
    for image, filename in zip(images, filenames):
        filepath = os.path.join(output_folder, filename)
        cv2.imwrite(filepath, image)

def main(original_image_path, mask_folder, output_folder):
    image, masks, filenames = load_image_and_masks(original_image_path, mask_folder)
    processed_images = []

    for mask in masks:
        new_mask = process_mask(image, mask)
        if new_mask is not None:
            masked_image = cv2.bitwise_and(image, image, mask=new_mask)
            cropped_image = crop_non_black(masked_image)
            processed_images.append(cropped_image)

    save_images(output_folder, processed_images, filenames)

# 使用示例
original_image_path = '/Users/jermainezhao/hand_hand/items_2.JPG'  # 替换为你的原图像路径
mask_folder = '/Users/jermainezhao/hand_hand/segmented_objects/filtered'  # 替换为你的mask文件夹路径
output_folder = '/Users/jermainezhao/hand_hand/segmented_objects/items'  # 替换为你希望保存结果的文件夹路径
main(original_image_path, mask_folder, output_folder)