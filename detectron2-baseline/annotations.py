import os 
import gc 
import cv2 
import random 
import numpy as np 
from tqdm.auto import tqdm 

import time 
import json


def split_train_test(lst, tv_rate):
    random.seed(42)
    shuffled_lst = random.sample(lst, len(lst))   
    n = round(len(shuffled_lst)*tv_rate[0])
    train_sample, valid_sample = shuffled_lst[:n], shuffled_lst[n:]
    print(f'total data: {len(shuffled_lst)}, train data: {len(train_sample)}, valid data: {len(valid_sample)}')
    return train_sample, valid_sample 


def get_dicts_from_json(ann_dirs, mode, findings, ann_version='1', train_institution='', eval_institution='', tv_rate=[0.9, 0.1]): 
    
    real_ann_dir = f'/ai-data/chest/kjs2109/pseudo_label_dataset/chestALL/annotations/{findings}findings/v0'
    ann_dir = os.path.join('/ai-data/chest/kjs2109/pseudo_label_dataset/chestALL/annotations', f'{findings}findings', f'v{ann_version}') 

    data_dicts = []

    unsorted_ann_fnames = os.listdir(real_ann_dir)  

    # 학습(train, valid)에 사용되는 데이터셋 준비 
    if train_institution != '' and mode in ['train', 'valid']:
        unsorted_filtered_ann_fnames = [] 
        for institution in train_institution: 
            unsorted_filtered_ann_fnames += ['real-' + fn for fn in unsorted_ann_fnames if institution in fn] 

        ann_fnames = sorted(unsorted_filtered_ann_fnames)

        # train과 val 나누기
        if tv_rate[0] + tv_rate[1] == 1:   
            train_ann_fnames, valid_ann_fnames = split_train_test(ann_fnames, tv_rate)
        else:
            print('Dataset divide issue')
            print("Advice  |  Check train & validation dataset rate. Their sum must be 1.")
            return
        
        if ann_version != '0': 
            target_ann_fnames = train_ann_fnames + os.listdir(ann_dir) if mode == "train" else valid_ann_fnames 
        else: 
            target_ann_fnames = train_ann_fnames if mode == "train" else valid_ann_fnames

    # 평가(test)에 사용되는 데이터셋 준비 
    if eval_institution != '' and mode == 'test': 
        unsorted_filtered_ann_fnames = [] 
        for institnution in eval_institution:
            unsorted_filtered_ann_fnames += [fn for fn in unsorted_ann_fnames if institnution in fn]

        target_ann_fnames = sorted(unsorted_filtered_ann_fnames) 

    print('#'*40, f'Getting the {mode} dataset dictionary progressed....', '#'*40) 
    print(f'amount of data: {len(target_ann_fnames)}')

    for ann_fname in tqdm(target_ann_fnames):

        if ann_fname.startswith('real-'): 
            ann_path = os.path.join(real_ann_dir, ann_fname.replace('real-', '')) 
        else:  
            ann_path = os.path.join(ann_dir, ann_fname)  

        if os.path.exists(ann_path):
            with open(ann_path, 'r') as f:
                label_dict = json.load(f) 

            detectron_data_dict = {
                "file_name": label_dict["file_name"], 
                "image_id": label_dict["image_id"], 
                "height": label_dict["height"], 
                "width": label_dict["width"],
                "annotations": label_dict['annotations']  # detectron_annotations 
            }

        data_dicts.append(detectron_data_dict) 
        gc.collect() 

    return data_dicts 
