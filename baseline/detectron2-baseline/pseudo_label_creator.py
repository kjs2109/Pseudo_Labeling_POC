import os 
import cv2 
import json 
import datetime 
import numpy as np 
import pandas as pd 
import torch 

import albumentations as A 
from tqdm.auto import tqdm 

from inference.inference_model import InferenceModel 

BoxMode = 0 


def get_longest_polygon(polygons):
    max_polygon = polygons[0].squeeze()
    for polygon in polygons[1:]:
        polygon = polygon.squeeze()
        if len(max_polygon) < len(polygon): 
            max_polygon = polygon 

    return max_polygon  


def make_annotations(scores, cat_ids, bboxes, maskes): 
    prediction_string = '' 
    annotations = [] 
    for score, cat_id, bbox, mask in zip(scores, cat_ids, bboxes, maskes): 
        
        bbox = list(map(round, bbox))
        polygons, _ = cv2.findContours(mask.astype(np.uint8)*255, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
        
        if isinstance(polygons, tuple): 
            polygon = get_longest_polygon(polygons) 
        else: 
            print(np.array(polygons).shape)
            print(polygons) 
            continue 
        
        segmentation = [coord[i] for coord in polygon for i in range(2)]  
        cat_id = int(cat_id)

        annotation = {
            "bbox": bbox, 
            "bbox_mode": BoxMode, 
            "segmentation": [segmentation], 
            "category_id": cat_id, 
        }
        annotations.append(annotation) 

        prediction_string += (str(cat_id) + ' ' + str(score) + ' ' + str(bbox[0]) + ' ' + str(bbox[1]) + ' ' + str(bbox[2]) + ' ' + str(bbox[3]) + ' ')

    return annotations, prediction_string 


def make_psuedo_label(institutions, source_dir, label_dir, output_dir, target_classes, CONFIG_DICT): 

    image_ids = []
    prediction_strings = [] 

    teacher_model = InferenceModel(CONFIG_DICT) 

    for institution in institutions: 
        institution_source_dir = os.path.join(source_dir, institution) 
        source_fnames = os.listdir(institution_source_dir)
        print(f'Institution: {institution} ({len(source_fnames)})')

        for source_fname in tqdm(source_fnames): 

            prediction_string = None 
            source_fpath = os.path.join(institution_source_dir, source_fname) 
            img = cv2.imread(source_fpath, cv2.IMREAD_COLOR) 

            # inference
            scores, cat_ids, bboxes, maskes = teacher_model.inference(img) 

            if len(cat_ids) != 0:
                image_id = source_fname.split('.')[0] 

                annotations, prediction_string  = make_annotations(scores, cat_ids, bboxes, maskes) 

                pseudo_label = {
                    "image_id": image_id, 
                    "file_name": source_fpath, 
                    "institution": institution, 
                    "target_classes": target_classes,
                    "height": 1024, 
                    "width": 1024, 
                    "is_normal": False, 
                    "annotations": annotations 
                }

                with open(os.path.join(label_dir, image_id+'.json'), 'w') as f: 
                    json.dump(pseudo_label, f, indent='\t') 
            
            image_ids.append(str(os.path.join(institution, source_fname))) 
            prediction_strings.append(prediction_string) 

    outputs = pd.DataFrame({'image_id': image_ids, 'PredictionString': prediction_strings})
    outputs.to_csv(os.path.join(output_dir, f'output-{datetime.datetime.now().strftime("%m-%d-%H-%M-%S")}.csv'), index=False)


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    
    institutions = ['SIIM', 'MIMIC', 'NIH', 'VBD', 'CXD'] 
    source_dir = '/ai-data/chest/kjs2109/pseudo_label_dataset/chestALL/images'
    label_dir = '/ai-data/chest/kjs2109/pseudo_label_dataset/chestALL/annotations/1findings/v1'
    output_dir = '/ai-data/chest/kjs2109/pseudo_label_dataset/chestALL/outputs/1findings/v1'
    target_classes = ['pneumothorax']
    input_size = 1024 

    CONFIG_DICT = {
        'weight_path': '/ai-data/chest/kjs2109/baseline/detectron2-baseline/weight_dir/MRCNN/MRCNN_BORA_EVAL_1Findings_1gpus_512bs_8workers_MRCNN_1024_0.0001LR_WCLR_mask_rcnn_R_50_FPN_3x_gn/model_best_0.0000.pth', 
        'model_zoo_config': 'Misc/mask_rcnn_R_50_FPN_3x_gn.yaml',
        'input_min_size_test': input_size, 
        'input_max_size_test': input_size, 
        'model_roi_heads_batch': 512, 
        'model_roi_heads_threshold': 0.5,
        'CLASSES': target_classes, 
        'device': 'cuda' if torch.cuda.is_available() else 'cpu' 
    } 

    make_psuedo_label(institutions, source_dir, label_dir, output_dir, target_classes, CONFIG_DICT)