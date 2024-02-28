import torch 
from torch import nn 
import torchsummary 
import segmentation_models_pytorch as smp

class EfficientUNetPP(nn.Module): 
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

        self.model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b5", 
            encoder_weights="imagenet",  # noisy-student  
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
    
class EfficientUNetPPB4(nn.Module): 
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

        self.model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b4", 
            encoder_weights="imagenet",  # noisy-student  
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