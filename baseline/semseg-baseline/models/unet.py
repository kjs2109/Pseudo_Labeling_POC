import torch 
from torch import nn 
import torchsummary 
import segmentation_models_pytorch as smp


class EfficientUNetB3(nn.Module):
    def __init__(self, classes, use_aux=False):  
        super().__init__()

        self.use_aux = use_aux 
        if use_aux:
            aux_params = {
                'pooling': 'avg', 
                'dropout': 0.25, 
                'classes': len(classes)
            }
        else: 
            aux_params = None

        self.model = smp.Unet(
            encoder_name="efficientnet-b3", 
            encoder_weights="imagenet",  
            in_channels=3,                
            classes=len(classes),    
            aux_params=aux_params                 
        )
        
    def forward(self, x): 
        if self.use_aux: 
            mask_output, cls_output = self.model(x) 
        else: 
            mask_output = self.model(x) 
            cls_output = None 

        output = {'mask_output': mask_output, 'cls_output': cls_output} 

        return output 
    

class EfficientUNet(nn.Module):
    def __init__(self, classes, use_aux=False):  
        super().__init__()

        self.use_aux = use_aux 
        if use_aux:
            aux_params = {
                'pooling': 'avg', 
                'dropout': 0.25, 
                'classes': len(classes)
            }
        else: 
            aux_params = None

        self.model = smp.Unet(
            encoder_name="efficientnet-b5", 
            encoder_weights="imagenet", # "imagenet",  
            in_channels=3,                
            classes=len(classes),    
            aux_params=aux_params                 
        )
        
    def forward(self, x): 
        if self.use_aux: 
            mask_output, cls_output = self.model(x) 
        else: 
            mask_output = self.model(x) 
            cls_output = None 

        output = {'mask_output': mask_output, 'cls_output': cls_output} 

        return output 


class EfficientUNetB6(nn.Module):
    def __init__(self, classes, use_aux=False):  
        super().__init__()

        self.use_aux = use_aux 
        if use_aux:
            aux_params = {
                'pooling': 'avg', 
                'dropout': 0.25, 
                'classes': len(classes)
            }
        else: 
            aux_params = None

        self.model = smp.Unet(
            encoder_name="efficientnet-b6", 
            encoder_weights="imagenet", # "imagenet",  
            in_channels=3,                
            classes=len(classes),    
            aux_params=aux_params                 
        )
        
    def forward(self, x): 
        if self.use_aux: 
            mask_output, cls_output = self.model(x) 
        else: 
            mask_output = self.model(x) 
            cls_output = None 

        output = {'mask_output': mask_output, 'cls_output': cls_output} 

        return output 
    
    
class EfficientUNetB7(nn.Module):
    def __init__(self, classes, use_aux=False):  
        super().__init__()

        self.use_aux = use_aux 
        if use_aux:
            aux_params = {
                'pooling': 'avg', 
                'dropout': 0.25, 
                'classes': len(classes)
            }
        else: 
            aux_params = None

        self.model = smp.Unet(
            encoder_name="efficientnet-b7", 
            encoder_weights="imagenet", # "imagenet",  
            in_channels=3,                
            classes=len(classes),    
            aux_params=aux_params                 
        )
        
    def forward(self, x): 
        if self.use_aux: 
            mask_output, cls_output = self.model(x) 
        else: 
            mask_output = self.model(x) 
            cls_output = None 

        output = {'mask_output': mask_output, 'cls_output': cls_output} 

        return output 


if __name__ == '__main__': 
    
    target_classes = ['Pneumothorax']  # ['Consolidation', 'Pneumothorax', 'Fibrosis', 'Effusion', 'Nodule']
    model = EfficientUNetB7(classes=target_classes, use_aux=False) 

    torchsummary.summary(model, (3, 512, 512), batch_size=1, device='cpu')