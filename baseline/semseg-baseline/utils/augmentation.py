import cv2
import random 
import numpy as np 
import albumentations as A  

class MyRandAugment: 

    def __init__(self, n, m): 
        self.n = n 
        self.shift_x = np.linspace(0,0.1,10)[m]
        self.shift_y = np.linspace(0,0.1,10)[m]
        self.scale = np.linspace(0, 0.5, 10)[m]
        self.rot = np.linspace(0,40,10)[m]
        self.post = [4,4,5,5,6,6,7,7,8,8][m]
        self.cont = [np.linspace(-0.8,-0.1,10),np.linspace(0.1,2,10)][m]
        self.bright = np.linspace(0.1,0.7,10)[m]
        self.shar = np.linspace(0.1,0.9,10)[m]
        self.cut = np.linspace(0,60,10)[m]

    def __call__(self, img):
        min_size = min(img.shape[:2])
        crop_size = random.randint(round(min_size*0.8), min_size)
        fill_value = img.mean() if np.random.rand() > 0.5 else 0.0

        # default
        default_Aug = [
            A.CropNonEmptyMaskIfExists(crop_size, crop_size, p=0.5),
            A.Resize(512, 512, interpolation=cv2.INTER_AREA),
            A.HorizontalFlip(p=0.5)
        ]
        # geo
        geo_Aug =[
                A.ElasticTransform(border_mode=cv2.BORDER_CONSTANT, approximate=True, interpolation=cv2.INTER_AREA, alpha_affine=20.0, p=0.5),
                A.Affine(rotate=(-20, 20), p=0.5),
        ]
        # noise
        noise_Aug = [
                A.GridDropout(unit_size_min=25, unit_size_max=50, fill_value=fill_value, random_offset=True, p=0.3),
                A.PixelDropout(dropout_prob=0.2, drop_value=fill_value, p=0.3),
                A.GaussianBlur(p=0.5),
                A.Sharpen(p=0.5),
        ]
        # color
        color_Aug = [
                A.Equalize(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Sharpen(p=0.5)
        ]

        ops = default_Aug + random.sample(geo_Aug + noise_Aug + color_Aug, self.n)
        transforms = A.Compose(ops)

        return transforms


def create_transforms(augmentation): 
    if augmentation == 'BaseAugmentation': 
        transforms = A.Compose([
            A.HorizontalFlip(p=0.5), 
            A.Resize(512, 512), # 896 
        ])
    elif augmentation == 'CustomAugmentation': 
        transforms = A.Compose([
            A.HorizontalFlip(p=0.5), 
            A.RandomCrop(1900, 1900, p=0.5),
            A.Resize(1024, 1024), 
            A.ElasticTransform(alpha=1, alpha_affine=100, border_mode=0, p=0.5),
        ])
    elif augmentation == 'MyRandAugment': 
        transforms = MyRandAugment(2, 0)
    else: 
        raise NotImplementedError 
    
    return transforms 