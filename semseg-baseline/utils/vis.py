import os 
import cv2 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import json 


PALETTE = [
    (0, 0, 142), (220, 20, 60), (174, 57, 255), (0, 82, 0), (120, 166, 157),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
    (255, 77, 255), (0, 226, 252), (182, 182, 255), (110, 76, 0), (199, 100, 0), 
    (72, 0, 118), (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176),
]

target_class_colors = {
    'consolidation': PALETTE[0],  # 푸른색 
    'pneumothorax': PALETTE[1],   # 붉은색 
    'fibrosis': PALETTE[2],       # 보라색 
    'effusion': PALETTE[3],       # 초록색 
    'nodule': PALETTE[4],         # 연두색 
}

def make_mask(image, points):
    zero_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask = cv2.fillPoly(zero_mask, [points], 1)
    return mask 

def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = np.where(mask == 1, image[:, :, c] * (1 - alpha) + alpha * color[c], image[:, :, c])   
    return image 

def get_masked_image_from_json(image_path, label_path, alpha=0.4): 
    image = cv2.imread(image_path) 

    with open(label_path, 'r') as f: 
        label = json.load(f) 
    annotations = label['annotations'] 
    target_classes = label['target_classes']

    if len(annotations) == 0: 
        return image 
    else: 
        for ann in annotations: 
            cat_id = ann['category_id'] 
            points = ann['segmentation'][0] 
            polygon_x = [x for index, x in enumerate(points) if index % 2 == 0] 
            polygon_y = [y for index, y in enumerate(points) if index % 2 == 1] 
            polygon_xy = [(x, y) for x, y in zip(polygon_x, polygon_y)] 
            polygon_xy = np.array(polygon_xy, np.int32) 

            mask = make_mask(image, polygon_xy) 
            masked_image = apply_mask(image, mask, target_class_colors[target_classes[cat_id].lower()], alpha=alpha) 

    return masked_image 


def plot_masked_image_3x3(image_dir, label_dir, image_fnames, alpha=0.4): 
    fig, axes = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(8, 8)) 
    for i, image_fname in enumerate(image_fnames): 
        institution = image_fname.split('_')[0] 
        image_path = os.path.join(image_dir, institution, image_fname)
        label_path = os.path.join(label_dir, image_fname.replace('.jpg', '.json')) 
        if not os.path.exists(label_path): 
            print(f'Not exists: {label_path}')  
            continue 

        masked_image = get_masked_image_from_json(os.path.join(image_dir, image_path), os.path.join(label_dir, label_path), alpha=alpha) 
        
        axes[i//3, i%3].imshow(masked_image) 
        axes[i//3, i%3].set_title(image_fname, fontsize=6) 

    plt.show()