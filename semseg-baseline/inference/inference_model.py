import os 
import torch 
import numpy as np 
from importlib import import_module


class EnsembleModel: 
    def __init__(self, model_list, ensemble_mode): 
        self.model_list = model_list 
        self.ensemble_mode = ensemble_mode 
        self.num_model = len(model_list) 


    def get_weight(self, prob, alpha=1.0): # prob_cls: EBC 
        if (alpha > 0) and (alpha < 2):
            conf = abs(prob - 0.5)  # EBC 
            m = conf - conf.mean(dim=0) + 1  # EBC, mean = 0 -> 1 
            weight = torch.where(m >= 1.0, alpha * torch.sqrt(m - 1) + 1, -alpha * torch.sqrt(1 - m) + 1) # EBC 
        else: 
            raise ValueError(f'Invalid alpha: {alpha} alpha must be in range (0, 2)') 

        return weight 

    def soft_ensemble_output(self, x): 
        b, c, h, w = x.shape  
        mask_output_list = [] # EBCHW 
        cls_output_list = []  # EBC 
        for model in self.model_list: 
            output = model(x) 
            prob_mask = torch.sigmoid(output['mask_output'])  # BCHW 

            if output['cls_output'] is not None: 
                prob_cls = torch.sigmoid(output['cls_output']) 
            else: 
                prob_cls, _ = prob_mask.view(b, -1, w*h).max(dim=-1)  
        
            mask_output_list.append(prob_mask) 
            cls_output_list.append(prob_cls) 

        mask_output = torch.stack(mask_output_list, dim=0).sum(dim=0) / self.num_model  # EBCHW -> BCHW
        cls_output = torch.stack(cls_output_list, dim=0).sum(dim=0) / self.num_model  # EBC -> BC 
        return {'mask_output': mask_output, 'cls_output': cls_output} 
    
    def weighted_soft_ensemble_output(self, x): 
        b, c, h, w = x.shape  
        mask_output_logit_list = []
        cls_output_list = [] 
        for model in self.model_list: 
            output = model(x) 
            prob_logit = output['mask_output']  # BCHW
            prob_mask = torch.sigmoid(prob_logit)  # BCHW 

            if output['cls_output'] is not None: 
                # prob_cls = torch.sigmoid(output['cls_output']) 
                prob_cls = torch.sigmoid(0.5*output['cls_output'] + 0.5*prob_mask.view(b, -1, w*h).max(dim=-1)[0]) 
            else: 
                prob_cls, _ = prob_mask.view(b, -1, w*h).max(dim=-1) 

            mask_output_logit_list.append(prob_logit) 
            cls_output_list.append(prob_cls) 

        weight = self.get_weight(torch.stack(cls_output_list, dim=0)).unsqueeze(-1).unsqueeze(-1) 
        weighted_prob_mask = torch.sigmoid(torch.stack(mask_output_logit_list, dim=0) * weight) 

        mask_output = weighted_prob_mask.sum(dim=0) / self.num_model   
        cls_output = torch.stack(cls_output_list, dim=0).sum(dim=0) / self.num_model  

        return {'mask_output': mask_output, 'cls_output': cls_output}   

    def __call__(self, x): 
        if self.ensemble_mode == 'soft': 
            output = self.soft_ensemble_output(x) 
        elif self.ensemble_mode == 'weighted_soft': 
            output = self.weighted_soft_ensemble_output(x) 
        else: 
            raise ValueError(f'Invalid ensemble_mode: {self.ensemble_mode}') 
        
        return output


def load_model(model, model_class, target_classes, use_aux, weight_path): 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    model_module = getattr(import_module(f"models.{model}"), model_class)
    model = model_module(target_classes, use_aux) 

    # load weight 
    model.load_state_dict(torch.load(weight_path)) 
    model = model.to(device)
    return model

def load_ensemble_model(ensemble_model_list, target_classes, saved_model_dir, ensemble_mode): 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    model_list = [] 
    for model_info in ensemble_model_list: 
        model_module = getattr(import_module(f"models.{model_info['model']}"), model_info['model_class'])
        model = model_module(target_classes, model_info['use_aux']) 

        weight_path = os.path.join(saved_model_dir, model_info['exp_name'], model_info['weight_fname']) 
        model.load_state_dict(torch.load(weight_path)) 
        model = model.to(device).eval()
        model_list.append(model) 
        print('Successfully loaded model: ', model_info['exp_name']) 

    model = EnsembleModel(model_list, ensemble_mode)
    return model 


# class InferenceModel: 

#     PALETTE = [(220, 20, 60), (0, 82, 0), (0, 0, 142), (220, 220, 0), (106, 0, 228)]

#     def __init__(self, model, target_classes, weight_path): 
#         self.target_classes = target_classes 
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

#         model = model.load_state_dict(torch.load(weight_path))
#         self.model.to(self.device).eval()

#     def _get_mask(self, logit_mask, thr=0.5): 
#         prob_mask = torch.sigmoid(logit_mask).cpu().numpy()  
#         pred_mask = (prob_mask > thr)
#         return pred_mask 

#     # def _get_max_score(self, logit_mask): 
#     #     prob_mask = torch.sigmoid(logit_mask).cpu().numpy()  
#     #     max_score = np.max(prob_mask.reshape(-1, w*h), axis=1)
#     #     return max_score

#     def inference(self, image): 
#         output = self.model(image) 



#         pred_max_scores = output.max()
#         pred_classes = output['pred_classes'] 
