# We will use this
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image

def load_model(model_path='yolov8m-seg.pt'):
    return YOLO(model_path)

def read_image(image_path):
    return cv2.imread(image_path)

def instance_segmentation(model, image, conf_threshold=0.04):
    return model.predict(image, conf=conf_threshold)

def process_segmentation_results(results):
    if len(results) == 0 or results[0].masks is None:
        print("No detections or masks found.")
        return np.array([])  # Empty array to avoid errors
    else:
        masks = results[0].masks.data.cpu().numpy()
        return masks

def visualize_masks(masks):
    for i, mask in enumerate(masks):
        plt.imshow(mask, cmap='gray', alpha=0.5)
        plt.axis('off')
        # plt.show()

def post_process_masks(masks, output_dir='segmented_objects', min_area=5):
    kernel = np.ones((3, 3), np.uint8)
    os.makedirs(output_dir, exist_ok=True)
    
    for i, mask in enumerate(masks):
        if mask.ndim > 2:
            mask = mask.squeeze()
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = (mask * 255).astype(np.uint8)
        num_labels, labels_im = cv2.connectedComponents(mask)
        for label in range(1, num_labels):
            if np.sum(labels_im == label) < min_area:
                mask[labels_im == label] = 0
        cv2.imwrite(os.path.join(output_dir, f'object_{i}.png'), mask)

# def calculate_iou(mask1, mask2):
#     intersection = np.logical_and(mask1, mask2)
#     union = np.logical_or(mask1, mask2)
#     iou_score = np.sum(intersection) / np.sum(union)
#     return iou_score

def resize_mask(mask, shape):
    return cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)

def calculate_iou(mask1, mask2):
    common_shape = (max(mask1.shape[0], mask2.shape[0]), max(mask1.shape[1], mask2.shape[1]))
    resized_mask1 = resize_mask(mask1, common_shape)
    resized_mask2 = resize_mask(mask2, common_shape)
    
    intersection = np.logical_and(resized_mask1, resized_mask2)
    union = np.logical_or(resized_mask1, resized_mask2)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def load_masks(folder_path):
    masks = []
    filenames = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            filepath = os.path.join(folder_path, filename)
            mask = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            mask = mask / 255
            masks.append(mask)
            filenames.append(filename)
    return masks, filenames

# def filter_similar_masks(masks, filenames, iou_threshold=0.5):
#     filtered_masks = []
#     filtered_filenames = []
#     for i in range(len(masks)):
#         is_similar = False
#         for j in range(i):
#             if calculate_iou(masks[i], masks[j]) > iou_threshold:
#                 is_similar = True
#                 break
#         if not is_similar:
#             filtered_masks.append(masks[i])
#             filtered_filenames.append(filenames[i])
#     return filtered_masks, filtered_filenames

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

def save_filtered_masks(folder_path, filtered_masks, filtered_filenames):
    output_folder = os.path.join(folder_path, "filtered")
    os.makedirs(output_folder, exist_ok=True)
    for mask, filename in zip(filtered_masks, filtered_filenames):
        filepath = os.path.join(output_folder, filename)
        cv2.imwrite(filepath, mask * 255)

def main_segmentation(original_image_path):
    model = load_model()
    image = read_image(original_image_path)
    results = instance_segmentation(model, image)
    masks = process_segmentation_results(results)
    
    if masks.size > 0:
        print(f"Converted masks shape: {masks.shape}")
        post_process_masks(masks)
    else:
        print("No masks to process.")

def main_filter(folder_path):
    masks, filenames = load_masks(folder_path)
    filtered_masks, filtered_filenames = filter_similar_masks(masks, filenames)
    save_filtered_masks(folder_path, filtered_masks, filtered_filenames)

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

    x_min, y_min, x_max, y_max = float('inf'), float('inf'), 0, 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)

    x, y, w, h = x_min, y_min, x_max - x_min, y_max - y_min

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

def crop(original_image_path, mask_folder, output_folder):
    image, masks, filenames = load_image_and_masks(original_image_path, mask_folder)
    processed_images = []

    for mask in masks:
        new_mask = process_mask(image, mask)
        if new_mask is not None:
            masked_image = cv2.bitwise_and(image, image, mask=new_mask)
            cropped_image = crop_non_black(masked_image)
            processed_images.append(cropped_image)

    save_images(output_folder, processed_images, filenames)

def resize_image(file_path, max_size_bytes=4 * 1024 * 1024):
    with Image.open(file_path) as img:
        width, height = img.size
        scale_factor = 0.7
        while os.path.getsize(file_path) > max_size_bytes:
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            img = img.resize((new_width, new_height), Image.LANCZOS)
            img.save(file_path)
            width, height = new_width, new_height

def resize_images_in_folder(folder, max_size_mb=4):
    max_size_bytes = max_size_mb * 1024 * 1024
    for root, dirs, files in os.walk(folder):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.isfile(file_path):
                file_size = os.path.getsize(file_path)
                if file_size > max_size_bytes:
                    resize_image(file_path, max_size_bytes)

def main1(original_image_path):
    
    main_segmentation(original_image_path)
    folder_path = '/Users/jermainezhao/CalHack_Hand_to_Hand/Hand_to_Hand/e_commerce_template/segmented_objects'
    
    main_filter(folder_path)

    mask_folder = '/Users/jermainezhao/CalHack_Hand_to_Hand/Hand_to_Hand/e_commerce_template/segmented_objects/filtered'
    output_folder = '/Users/jermainezhao/CalHack_Hand_to_Hand/Hand_to_Hand/e_commerce_template/segmented_objects/items'
    crop(original_image_path, mask_folder, output_folder)

    resize_images_in_folder(output_folder)

