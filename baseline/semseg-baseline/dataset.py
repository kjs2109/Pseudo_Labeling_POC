import os 
import copy 
import random 
import xml.etree.ElementTree as ET 

import torch 
from torch.utils.data import Dataset 

import cv2
import json 
import numpy as np 
from tqdm.auto import tqdm

import albumentations as A 

def split_train_test(lst, tv_rate):
    random.seed(42)
    shuffled_lst = random.sample(lst, len(lst))   
    n = round(len(shuffled_lst)*tv_rate[0])
    train_sample, valid_sample = shuffled_lst[:n], shuffled_lst[n:] 

    print(f'total data: {len(shuffled_lst)}, train data: {len(train_sample)}, valid data: {len(valid_sample)}')
    return train_sample, valid_sample 

class MyXRayDataset(Dataset): 
    def __init__(self, CLASSES, IMAGE_ROOT, LABEL_ROOT, is_train, transforms=None): 
        
        train_institution = ['KU', 'CB', 'PAIK']

        self.CLASSES = CLASSES 
        self.IMAGE_ROOT = IMAGE_ROOT 
        self.LABEL_ROOT = LABEL_ROOT 
        self.is_train = is_train 
        self.transforms = transforms if is_train else A.Compose([A.Resize(512, 512)]) 

        unsorted_data_fnames = [] 
        for institution in train_institution: 
            unsorted_data_fnames += [fn for fn in os.listdir(LABEL_ROOT) if institution in fn] 

        total_data_fnames = sorted(unsorted_data_fnames) 
        train_fnames, valid_fnames = split_train_test(total_data_fnames, [0.9, 0.1]) 
        if is_train: 
            self.ann_fnames = train_fnames
        else: 
            self.ann_fnames = valid_fnames  

    def __len__(self): 
        return len(self.ann_fnames) 


    def __getitem__(self, idx): 
        
        ann_path = os.path.join(self.LABEL_ROOT, self.ann_fnames[idx]) 
        image_path = os.path.join(self.IMAGE_ROOT, self.ann_fnames[idx].replace('json', 'jpg')) 

        image = cv2.imread(image_path, cv2.IMREAD_COLOR) 

        # annotation mask 
        with open(ann_path, 'r') as f: 
            ann_dict = json.load(f) 

        w, h = ann_dict['width'], ann_dict['height'] 
        anns = ann_dict['annotations'] 

        temp_mask = np.zeros((len(self.CLASSES), h, w), dtype=np.uint8) 
        for ann in anns: 
            cat_id = ann['category_id'] 
            points = ann['segmentation'][0]
            polygon_x = [x for index, x in enumerate(points) if index % 2 == 0]
            polygon_y = [x for index, x in enumerate(points) if index % 2 == 1]
            polygon_xy = [[x, y] for x, y in zip(polygon_x, polygon_y)]
            polygon_xy = np.array(polygon_xy, np.int32)
            cv2.fillPoly(temp_mask[cat_id], [polygon_xy], 1) 

        label = np.zeros((h, w, len(self.CLASSES)), dtype=np.uint8) 
        for i in range(len(self.CLASSES)): 
            label[..., i] = temp_mask[i] 

        # transform 
        if self.transforms is not None: 
            inputs = {"image": image, "mask": label} 
            result = self.transforms(**inputs) 
            image = result["image"] 
            label = result["mask"] 

        # 전처리 마지막 단계에서 해주기 
        image = image.transpose(2, 0, 1) # CHW 
        label = label.transpose(2, 0, 1) 
        image = torch.from_numpy(image / 255.).float()  # image.dtype: uint8 -> float64 -> float32
        label = torch.from_numpy(label).float() 

        return image, label


class MyXRayDatasetV1(Dataset):
    def __init__(self, CLASSES, IMAGE_ROOT, LABEL_ROOT, findings, ann_version, is_train, copypaste=0.0, num_copypaste=1, transforms=None): 
        self.real_label_institution = ['KU', 'CB', 'PAIK']

        self.CLASSES = CLASSES 
        self.IMAGE_ROOT = IMAGE_ROOT 
        self.LABEL_ROOT = os.path.join(LABEL_ROOT, f'{findings}findings')  
        self.ann_version = f'v{ann_version}'
        self.is_train = is_train 
        self.transforms = transforms if is_train else A.Compose([A.Resize(512, 512)]) 

        self.only_normal = False  # normal case에 대해서만 copy-paste를 적용하고 싶은 경우 True로 직접 수정 필요 (하드코딩 된 상태)  
        self.num_copypaste = num_copypaste  # copy-paste를 몇개의 abnormal case에 대해서 적용할지 결정 (command line --num_copypaste 로 설정)
        self.copypaste = copypaste  # copy-paste를 적용할 확률 (command line --copypaste 로 설정) 

        unsorted_fnames = [] 
        for institution in self.real_label_institution: 
            unsorted_fnames += [fn for fn in os.listdir(os.path.join(self.LABEL_ROOT, 'v0')) if institution in fn] 

        sorted_fnames = sorted(unsorted_fnames) 
        train_fnames, valid_fnames, abnormal_fnames = self._split_train_test(sorted_fnames, [0.9, 0.1]) 
        if is_train: 
            self.ann_fnames = train_fnames 
            if self.ann_version != 'v0': 
                pseudo_label_fnames = os.listdir(os.path.join(self.LABEL_ROOT, self.ann_version))
                self.ann_fnames += pseudo_label_fnames  
                print(f'add pseudo label {self.ann_version} data: +{len(pseudo_label_fnames)}, train data: {len(self.ann_fnames)}')
        else: 
            self.ann_fnames = valid_fnames  

        self.abnormal_fnames = abnormal_fnames 
    
    def _split_train_test(self, lst, tv_rate):
        random.seed(42)
        shuffled_lst = random.sample(lst, len(lst))   
        n = round(len(shuffled_lst)*tv_rate[1])

        valid_sample = []  # abnormal data의 10% (tv_rate[1])
        train_sample = [] 
        abnormal_sample = [] 
        print('Preparing train/valid data...')
        for i, fname in enumerate(shuffled_lst): 
            with open(os.path.join(self.LABEL_ROOT, 'v0', fname), 'r') as f: 
                ann_dict = json.load(f) 
            
            if not ann_dict['is_normal'] and i < n: 
                valid_sample.append(fname) 
            else: 
                if not ann_dict['is_normal']: 
                    abnormal_sample.append(fname) 
                train_sample.append(fname) 

        print(f'total data: {len(shuffled_lst)}, train data: {len(train_sample)}, valid data: {len(valid_sample)} (abnormal data: {len(abnormal_sample)})')
        return train_sample, valid_sample, abnormal_sample 

    def __len__(self): 
        return len(self.ann_fnames) 
    
    def apply_copypaste(self, image, temp_mask, cls_label, h, w): 
        sample_fname = random.choices(self.abnormal_fnames, k=1) 
        sample_path = os.path.join(self.LABEL_ROOT, 'v0', sample_fname[0]) 

        with open(sample_path, 'r') as f: 
            sample_dict = json.load(f) 
        sample_img_path = sample_dict['file_name']
        sample_anns = sample_dict['annotations'] 
        
        image_mask = np.zeros((h, w), dtype=np.uint8)
        sample_image = cv2.imread(sample_img_path, cv2.IMREAD_COLOR) 
        ori_image = copy.deepcopy(image)  

        for sample_ann in sample_anns: 
            cat_id = sample_ann['category_id'] 
            cls_label[cat_id] = 1 
            points = sample_ann['segmentation'][0]
            polygon_x = [x for index, x in enumerate(points) if index % 2 == 0]
            polygon_y = [x for index, x in enumerate(points) if index % 2 == 1]
            polygon_xy = [[x, y] for x, y in zip(polygon_x, polygon_y)]
            polygon_xy = np.array(polygon_xy, np.int32)
            cv2.fillPoly(temp_mask[cat_id], [polygon_xy], 1) 
            cv2.fillPoly(image_mask, [polygon_xy], 1) 

        cvt_image_mask = np.logical_not(image_mask).astype(np.uint8) 

        sample_image = sample_image * np.expand_dims(image_mask, axis=-1) 
        ori_image = ori_image * np.expand_dims(cvt_image_mask, axis=-1) 

        image = sample_image + ori_image

        return image, temp_mask, cls_label  

    def __getitem__(self, idx): 

        # load label  
        ann_fname = self.ann_fnames[idx] 
        institution = ann_fname.split('_')[0] 
        ann_path = os.path.join(self.LABEL_ROOT, 'v0' if institution in self.real_label_institution else self.ann_version, ann_fname)

        with open(ann_path, 'r') as f: 
            ann_dict = json.load(f)

        w, h = ann_dict['width'], ann_dict['height'] 
        target_classes = ann_dict['target_classes'] 
        is_normal = ann_dict['is_normal']
        anns = ann_dict['annotations'] 
        cls_label = np.zeros(len(target_classes), dtype=np.uint8)

        # load image 
        image_path = ann_dict['file_name']  
        image = cv2.imread(image_path, cv2.IMREAD_COLOR) 

        # polygon to mask 
        temp_mask = np.zeros((len(target_classes), h, w), dtype=np.uint8) 

        if self.only_normal: 
            if not is_normal: 
                for ann in anns: 
                    cat_id = ann['category_id'] 
                    cls_label[cat_id] = 1 
                    points = ann['segmentation'][0]
                    polygon_x = [x for index, x in enumerate(points) if index % 2 == 0]
                    polygon_y = [x for index, x in enumerate(points) if index % 2 == 1]
                    polygon_xy = [[x, y] for x, y in zip(polygon_x, polygon_y)]
                    polygon_xy = np.array(polygon_xy, np.int32)
                    cv2.fillPoly(temp_mask[cat_id], [polygon_xy], 1) 
            else: 
                for _ in range(self.num_copypaste): 
                    if random.random() < self.copypaste: 
                        image, temp_mask, cls_label = self.apply_copypaste(image, temp_mask, cls_label, h, w)
        else: 
            for ann in anns: 
                cat_id = ann['category_id'] 
                cls_label[cat_id] = 1 
                points = ann['segmentation'][0]
                polygon_x = [x for index, x in enumerate(points) if index % 2 == 0]
                polygon_y = [x for index, x in enumerate(points) if index % 2 == 1]
                polygon_xy = [[x, y] for x, y in zip(polygon_x, polygon_y)]
                polygon_xy = np.array(polygon_xy, np.int32)
                cv2.fillPoly(temp_mask[cat_id], [polygon_xy], 1) 

            for _ in range(self.num_copypaste): 
                if random.random() < self.copypaste:   
                    image, temp_mask, cls_label = self.apply_copypaste(image, temp_mask, cls_label, h, w) 

        mask_label = np.zeros((h, w, len(target_classes)), dtype=np.uint8) 
        for i in range(len(target_classes)): 
            mask_label[..., i] = temp_mask[i] 

        # transform 
        if self.transforms is not None: 
            inputs = {"image": image, "mask": mask_label} 
            if self.transforms.__class__.__name__ == 'MyRandAugment': 
                result = self.transforms(image)(**inputs)
            else:
                result = self.transforms(**inputs) 
            image = result["image"] 
            mask_label = result["mask"] 

        image = image.transpose(2, 0, 1) 
        mask_label = mask_label.transpose(2, 0, 1) 
        image = torch.from_numpy(image / 255.).float() 
        mask_label = torch.from_numpy(mask_label).float() 
        cls_label = torch.from_numpy(cls_label).float()

        return image, mask_label, cls_label 



class EvaluationDataset(Dataset):
    def __init__(self, CLASSES, IMAGE_ROOT, LABEL_ROOT, transforms=None): 
        
        self.eval_institution = ['BORA']

        self.CLASSES = CLASSES 
        self.IMAGE_ROOT = IMAGE_ROOT 
        self.LABEL_ROOT = LABEL_ROOT 
        self.transforms = transforms 

        unsorted_data_fnames = [] 
        for institution in self.eval_institution: 
            unsorted_data_fnames += [fn for fn in os.listdir(LABEL_ROOT) if institution in fn] 

        eval_data_fnames = sorted(unsorted_data_fnames) 

        self.ann_fnames = eval_data_fnames 
    
    def __len__(self):
        return len(self.ann_fnames)
    
    def __getitem__(self, idx): 
        ann_path = os.path.join(self.LABEL_ROOT, self.ann_fnames[idx]) 
        image_path = os.path.join(self.IMAGE_ROOT, 'BORA', self.ann_fnames[idx].replace('json', 'jpg')) 
        # image_path = os.path.join(self.IMAGE_ROOT, self.ann_fnames[idx].replace('json', 'jpg')) 

        image = cv2.imread(image_path, cv2.IMREAD_COLOR) 

        with open(ann_path, 'r') as f: 
            ann_dict = json.load(f) 

        w, h = ann_dict['width'], ann_dict['height'] 
        anns = ann_dict['annotations'] 

        temp_mask = np.zeros((len(self.CLASSES), h, w), dtype=np.uint8) 
        for ann in anns: 
            cat_id = ann['category_id'] 
            points = ann['segmentation'][0]
            polygon_x = [x for index, x in enumerate(points) if index % 2 == 0]
            polygon_y = [x for index, x in enumerate(points) if index % 2 == 1]
            polygon_xy = [[x, y] for x, y in zip(polygon_x, polygon_y)]
            polygon_xy = np.array(polygon_xy, np.int32)
            cv2.fillPoly(temp_mask[cat_id], [polygon_xy], 1) 

        label = np.zeros((h, w, len(self.CLASSES)), dtype=np.uint8) 
        for i in range(len(self.CLASSES)): 
            label[..., i] = temp_mask[i] 

        if self.transforms is not None: 
            inputs = {"image": image, "mask": label} 
            result = self.transforms(**inputs) 
            image = result["image"] 
            label = result["mask"] 

        image = image.transpose(2, 0, 1) 
        label = label.transpose(2, 0, 1) 
        image = torch.from_numpy(image / 255.).float() 
        label = torch.from_numpy(label).float() 

        return image, label, self.ann_fnames[idx] 
    

class PseudoLabelDataset(Dataset): 
    def __init__(self, IMAGE_ROOT, institutions, transforms=None): 
        self.IMAGE_ROOT = IMAGE_ROOT 
        self.transforms = transforms 

        image_fnames = [] 
        for institution in institutions: 
            image_dir = os.path.join(IMAGE_ROOT, institution)  
            image_fnames += [f'{institution}/{fname}' for fname in os.listdir(image_dir)]  

        self.image_fnames = image_fnames 

    def __len__(self): 
        return len(self.image_fnames) 

    def __getitem__(self, idx): 
        image_fname = self.image_fnames[idx]  # institution/fname.jpg 
        image_path = os.path.join(self.IMAGE_ROOT, image_fname) 
        image = cv2.imread(image_path, cv2.IMREAD_COLOR) 

        if self.transforms is not None: 
            image = self.transforms(image=image)['image'] 

        h, w = image.shape[:2]
        image = image.transpose(2, 0, 1) 
        image = torch.from_numpy(image / 255.).float() 

        return image, image_fname, h, w  