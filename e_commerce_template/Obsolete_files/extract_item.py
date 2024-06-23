import cv2
import numpy as np
import matplotlib.pyplot as plt
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
from mrcnn.model import MaskRCNN

# 配置 Mask R-CNN
class InferenceConfig(Config):
    NAME = "coco_inference"
    NUM_CLASSES = 1 + 80  # COCO数据集有80个类
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
model = MaskRCNN(mode="inference", model_dir="./", config=config)
model.load_weights("mask_rcnn_coco.h5", by_name=True)

# 加载图片
image = cv2.imread("/Users/jermainezhao/hand_hand/items.JPG")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 进行物体检测
results = model.detect([image], verbose=1)
r = results[0]

# 可视化并保存分割结果
for i in range(len(r['rois'])):
    mask = r['masks'][:, :, i]
    y1, x1, y2, x2 = r['rois'][i]
    extracted_object = image[y1:y2, x1:x2] * mask[y1:y2, x1:x2, np.newaxis]
    plt.imsave(f'object_{i}.png', extracted_object)

    # 显示提取的物体
    plt.imshow(extracted_object)
    plt.axis('off')
    plt.show()