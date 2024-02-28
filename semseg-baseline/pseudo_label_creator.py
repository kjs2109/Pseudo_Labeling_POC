import os 
import json 
import time  
import datetime 

import cv2 
import numpy as np 
import pandas as pd 
from tqdm.auto import tqdm
from importlib import import_module 
import albumentations as A 

import torch 
from torch.utils.data import DataLoader

from dataset import PseudoLabelDataset 
from utils.helper import check_and_mkdir 
from inference.inference_model import load_ensemble_model, load_model  

BoxMode = 0 


def analyze_output_csv(model_info, institutions, output_path, target_classes, pos_thr, neg_thr): 

    output_df = pd.read_csv(output_path)
    
    analyze_per_institution = {
                            institution: 
                                {'count [pos, neg, labled, total]': [0, 0, 0, 0], 'count labeled disease per class': [0]*len(target_classes)} 
                                for institution in institutions
                        }

    for i, row in enumerate(output_df.itertuples()): 
        image_id = row.image_id 
        institution = image_id.split('/')[0] 
        prediction_string = row.PredictionString  

        analyze_per_institution[institution]['count [pos, neg, labled, total]'][3] += 1 
        if not isinstance(prediction_string, float): 
            analyze_per_institution[institution]['count [pos, neg, labled, total]'][2] += 1 

            if prediction_string == '-1': 
                analyze_per_institution[institution]['count [pos, neg, labled, total]'][1] += 1 
            else: 
                analyze_per_institution[institution]['count [pos, neg, labled, total]'][0] += 1 

                string_list = prediction_string.strip().split(' ') 
                cat_ids = string_list[::6] 

                for cat_id in cat_ids: 
                    analyze_per_institution[institution]['count labeled disease per class'][int(cat_id)] += 1

    count_positive_case, count_negative_case, labeled_data = 0, 0, 0 
    number_of_disease_per_class = [0]*len(target_classes)  
    for institution in institutions: 
        count_positive_case += analyze_per_institution[institution]['count [pos, neg, labled, total]'][0] 
        count_negative_case += analyze_per_institution[institution]['count [pos, neg, labled, total]'][1] 
        labeled_data += analyze_per_institution[institution]['count [pos, neg, labled, total]'][2] 
        for i in range(len(target_classes)): 
            number_of_disease_per_class[i] += analyze_per_institution[institution]['count labeled disease per class'][i]

    analyze_result = {
        'target_classes': target_classes, 
        'institution': institutions,
        'model_info': model_info, 
        'pos_thr': pos_thr,
        'neg_thr': neg_thr, 
        'total_data':len(output_df),
        'labeled_data':labeled_data,  
        'positive_case':count_positive_case, 
        'negative_case':count_negative_case, 
        'detail': analyze_per_institution, 
    } 

    return analyze_result

def post_process(pred_mask): 

    transform = A.Resize(1024, 1024, interpolation=cv2.INTER_CUBIC)

    pred_mask = pred_mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8) 
    step1 = cv2.erode(pred_mask, kernel, iterations=3)
    step2 = cv2.dilate(step1, kernel, iterations=6) 
    step3 = cv2.erode(step2, kernel, iterations=2) 
    step4 = transform(image=step3)['image'] 
    return step4

def make_annotations(scores, cat_ids, maskes):  # CHW -> 이미지 한 장에 대한 output 
    prediction_string = '' 
    annotations = [] 
    for score, cat_id, mask in zip(scores, cat_ids, maskes): 
        
        polygons, _ = cv2.findContours(mask.astype(np.uint8)*255, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
        
        if not isinstance(polygons, tuple): 
            continue 
        
        for polygon in polygons: 
            polygon = np.array(polygon.squeeze()) 
            polygon_x = polygon[:, 0] 
            polygon_y = polygon[:, 1]
            bbox = [polygon_x.min().item(), polygon_y.min().item(), polygon_x.max().item(), polygon_y.max().item()]
            segmentation = [coord[i] for coord in polygon.tolist() for i in range(2)]  
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

def make_pseudo_label(institutions, source_dir, label_dir, output_dir, target_classes, model_info, saved_model_dir, pos_thr, neg_thr): 
    
    if os.path.exists(label_dir):
        # 만들어진 label 덮어쓰지 않도록 예외처리
        raise ValueError('Already exists')   
    else: 
        check_and_mkdir(label_dir)

    print("######### Start making pseudo labels!! #########")

    transform = A.Resize(512, 512, interpolation=cv2.INTER_AREA) 

    dataset = PseudoLabelDataset(IMAGE_ROOT=source_dir, institutions=institutions, transforms=transform) 
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0) 

    use_ensemble = False 
    if len(model_info) == 1: 
        model = model_info[0]['model'] 
        model_class = model_info[0]['model_class'] 
        use_aux = model_info[0]['use_aux'] 
        weight_path = os.path.join(saved_model_dir, model_info[0]['exp_name'], model_info[0]['weight_fname'])
        model = load_model(model, model_class, target_classes, use_aux, weight_path)  
        model.cuda() 
        model.eval() 
    elif len(model_info) > 1: 
        use_ensemble = True 
        model = load_ensemble_model(model_info, target_classes, saved_model_dir, ensemble_mode='weighted_soft') 
    else: 
        raise ValueError('Invalid model_info') 

    image_ids = []
    prediction_strings = [] 

    with torch.no_grad(): 
        for batch in tqdm(dataloader):  
            input_images, image_fnames, h, w = batch[0], batch[1], batch[2], batch[3] 
            input_images = input_images.to('cuda') 
            outputs = model(input_images)  # BCHW 

            if use_ensemble: 
                prob_masks = outputs['mask_output'] 
                pred_masks = (prob_masks > 0.5)
                prob_cats = outputs['cls_output'] 
            else: 
                logit_masks = outputs['mask_output'] 
                prob_masks = torch.sigmoid(logit_masks) 
                pred_masks = (prob_masks > 0.5) 
                prob_cats = torch.sigmoid(outputs['cls_output'])  # auxiliary classifier가 있는 경우 
                if prob_cats == None:
                    prob_cats = prob_masks.view(len(image_fnames), -1, h*w).max(dim=-1)  # 마스크에서 뽑은 확률 
                

            for prob_mask, pred_mask, prob_cat, image_fname in zip(prob_masks, pred_masks, prob_cats, image_fnames):  # BCHW -> CHW 
                pos_scores, neg_scores, cat_ids, maskes = [], [], [], []
                for cat_idx, (target_prob_mask, target_pred_mask, target_prob) in enumerate(zip(prob_mask, pred_mask, prob_cat)): 

                    probability = (target_prob + target_prob_mask.max().item()) / 2 

                    if probability > pos_thr:
                        pos_scores.append(probability.cpu().item()) 
                        cat_ids.append(cat_idx)  
                        maskes.append(post_process(target_pred_mask.cpu().numpy())) 
                    elif probability < neg_thr:
                        neg_scores.append(probability.cpu().item())  
                            
                annotations, prediction_string = make_annotations(pos_scores, cat_ids, maskes) 
                
                # abnormal case 
                if len(annotations) != 0:
                    institution, image_id = image_fname.split('/') 
                    file_path = os.path.join(source_dir, institution, image_id) 
                    label_path = os.path.join(label_dir, image_id.replace('.jpg', '.json'))
                    pseudo_label = {
                        "image_id": str(image_id.split('.')[0]), 
                        "file_name": str(file_path), 
                        "institution": str(institution), 
                        "target_classes": target_classes, 
                        "height": 1024, 
                        "width": 1024, 
                        "is_normal": False, 
                        "annotations": annotations 
                    } 

                    with open(label_path, 'w') as f: 
                        json.dump(pseudo_label, f, indent='\t') 

                # normal case 
                elif len(neg_scores) == len(target_classes): 
                    institution, image_id = image_fname.split('/') 
                    file_path = os.path.join(source_dir, institution, image_id) 
                    label_path = os.path.join(label_dir, image_id.replace('.jpg', '.json'))
                    pseudo_label = {
                        "image_id": str(image_id.split('.')[0]), 
                        "file_name": str(file_path), 
                        "institution": str(institution), 
                        "target_classes": target_classes, 
                        "height": 1024, 
                        "width": 1024, 
                        "is_normal": True, 
                        "annotations": [] 
                    } 

                    with open(label_path, 'w') as f: 
                        json.dump(pseudo_label, f, indent='\t') 

                    prediction_string = '-1'

                image_ids.append(str(image_fname)) 
                prediction_strings.append(prediction_string)
    
    check_and_mkdir(output_dir)
    outputs = pd.DataFrame({'image_id': image_ids, 'PredictionString': prediction_strings}) 
    output_path = os.path.join(output_dir, f'output-{datetime.datetime.now().strftime("%m-%d-%H-%M-%S")}.csv') 
    outputs.to_csv(output_path, index=False) 
    
    analyze_result = analyze_output_csv(model_info, institutions, output_path, target_classes, pos_thr, neg_thr) 

    with open(output_path.replace('csv', 'json'), 'w') as f: 
        json.dump(analyze_result, f, indent=4) 


if __name__ == '__main__': 
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    annotation_version = 'v3'  # 만들고자 하는 pseudo label의 version

    institutions = ['SIIM', 'MIMIC', 'NIH', 'VBD', 'CXD', 'ETC']
    target_classes = ['consolidation', 'pneumothorax', 'fibrosis', 'effusion', 'nodule']  # ['pneumothorax']
    saved_model_dir = '/ai-data/chest/kjs2109/baseline/semseg-baseline/weight_dir'
    source_dir = '/ai-data/chest/kjs2109/pseudo_label_dataset/chestALL/images'
    label_dir = f'/ai-data/chest/kjs2109/pseudo_label_dataset/chestALL/annotations/{len(target_classes)}findings/{annotation_version}'
    output_dir = f'/ai-data/chest/kjs2109/pseudo_label_dataset/chestALL/outputs/{len(target_classes)}findings/{annotation_version}' 
    pos_thr = 0.97  
    neg_thr = 0.005   

    model_info = [
            {
                "model": "unet",
                "model_class": "EfficientUNet",
                "use_aux": True, 
                "exp_name": "exp5_5f_effiU-b5_aux-noise1-copypaste75",
                "weight_fname": "dice-8131-163999.pth"
            },
            {
                "model": "unet",
                "model_class": "EfficientUNetB3",
                "use_aux": True,
                "exp_name": "exp9_5f_effiU-b3_aux-noise1-loss55",
                "weight_fname": "dice-8115-171999.pth"
            },
            {
                "model": "unet",
                "model_class": "EfficientUNetB3",
                "use_aux": True, 
                "exp_name": "exp14_5f_effiU-b3_aux-noise2-loss19-copypaste25-2",
                "weight_fname": "dice-8197-163999.pth"
            }
        ]

    make_pseudo_label(institutions, source_dir, label_dir, output_dir, target_classes, model_info, saved_model_dir, pos_thr, neg_thr) 