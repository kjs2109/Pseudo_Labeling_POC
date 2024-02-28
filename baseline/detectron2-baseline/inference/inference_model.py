import os 
import cv2 
import torch 
import json 
import matplotlib.pyplot as plt 

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
import detectron2.data.transforms as T
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer 

from tqdm.auto import tqdm 

# SEG_CONFIG_DICT = {
#     'weight_path' : '/ai-data/chest/kjs2109/baseline/detectron2-baseline/weight_dir/COCO-InstanceSegmentation/COCO-InstanceSegmentation_BORA_EVAL_1Findings_1gpus_512bs_8workers_MRCNN_1024_0.0001LR_WCLR_mask_rcnn_R_101_FPN_3x/model_best_0.0000.pth'
#     'model_zoo_config': 'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml',
#     'input_min_size_test': 1024,
#     'input_max_size_test': 1024,
#     'model_roi_heads_batch': 512,
#     'model_roi_heads_threshold': 0.5,
#     'CLASSES': ['Pnueumothorax'],
#     'device': 'cuda' if torch.cuda.is_available() else 'cpu',
# }

class CustomPredictor: 
    def __init__(self, cfg): 
        self.cfg = cfg.clone() 
        self.aug = T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)
        self.input_format = cfg.INPUT.FORMAT 
        self.model = build_model(self.cfg) 
        self.model.eval() 
        
        checkpointer = DetectionCheckpointer(self.model) 
        checkpointer.load(cfg.MODEL.WEIGHTS) 

        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, image):  
        with torch.inference_mode(): 
            if self.input_format == "RGB": 
                image = image[:, :, ::-1] 
            height, width = image.shape[:2] 
            image = self.aug.get_transform(image).apply_image(image) 
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1)) 

            inputs = {"image": image, "height": height, "width": width} 
            prediction = self.model([inputs])[0] 

        return prediction

class InferenceModel:

    PALETTE = [(220, 20, 60), (0, 82, 0), (0, 0, 142), (220, 220, 0), (106, 0, 228)]  

    def __init__(self, config_dict): 
        self.target_classes = config_dict['CLASSES'] 
        self.cfg = self.setup_cfg(config_dict) 
        self.model = CustomPredictor(self.cfg)  
            

    def inference(self, image): 
        output = self.model(image)['instances']
        
        pred_scores = output.scores.cpu().numpy() 
        pred_classes = output.pred_classes.cpu().numpy() 
        pred_boxes = output.pred_boxes.tensor.cpu().numpy() 
        pred_masks = output.pred_masks.cpu().numpy() 

        return pred_scores, pred_classes, pred_boxes, pred_masks 
    

    def get_pred_show(self, image, mode=['mask'], threshold=0.5):  
        text_size = 2 
        text_thickness = 2 
        font = cv2.FONT_HERSHEY_SIMPLEX 

        scores, cat_ids, boxes, masks = self.inference(image) 
        draw_image = image.copy()
        for score, cat_id, bbox, mask in zip(scores, cat_ids, boxes, masks): 
            
            class_name = self.target_classes[cat_id]  
            text = f"{class_name}|{score:.2f}"
            if score > threshold: 
                color_idx = cat_id + 1 if self.target_classes == ['Pneumothorax'] else cat_id
                color = self.PALETTE[color_idx]
                x1, y1, x2, y2 = map(int, bbox)

                if 'mask' in mode:  
                    masked_image = draw_image[mask]  
                    draw_image[mask] = ([0.3*color[0], 0.3*color[1], 0.3*color[2]] + 0.7*masked_image).astype("uint8")

                if 'box' in mode: 
                    draw_image = cv2.rectangle(draw_image, (x1, y1), (x2, y2), color, 2)

                text_w, text_h = cv2.getTextSize(text, font, text_size, text_thickness)[0]
                cv2.rectangle(draw_image, (x1, y1), (x1+text_w+10, y1+text_h+10), color, -1) 
                cv2.putText(draw_image, text, (x1+5, y1+45), font, text_size, (255, 255, 255), text_thickness)

        return draw_image 

    def submission(self, image): 
        pass 

    def setup_cfg(self, config_dict): 
        cfg = get_cfg()

        cfg.merge_from_file(model_zoo.get_config_file(config_dict['model_zoo_config']))
        cfg.INPUT.MIN_SIZE_TEST = config_dict['input_min_size_test']
        cfg.INPUT.MAX_SIZE_TEST = config_dict['input_max_size_test'] 
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = config_dict['model_roi_heads_batch']
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(config_dict['CLASSES'])
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config_dict['model_roi_heads_threshold']
        cfg.MODEL.WEIGHTS = config_dict['weight_path']
        cfg.MODEL.DEVICE = config_dict['device']
        cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA = 0.5
        cfg.MODEL.ROI_BOX_HEAD.POSTPROCESS_MASKS = False
        cfg.MODEL.ROI_BOX_HEAD.RESOLUTION = 28
        cfg.MODEL.RPN.SMOOTH_L1_BETA = 0.5
        
        cfg.freeze()
        return cfg


def plot_inference_result(model, image_fnames, gt=True, mode=['mask'], threshold=0.5):
    n_cols = 3 
    n_rows = len(image_fnames) // n_cols 
    colors = model.PALETTE 
        
    image_dir = '/ai-data/chest/kjs2109/private_data/chestALL/images'
    ann_dir = '/ai-data/chest/kjs2109/private_data/chestALL/anns/5findings_v1'
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*6, n_rows*6)) 
   
    for i, image_fname in enumerate(tqdm(image_fnames)):
        image = cv2.imread(os.path.join(image_dir, image_fname))
        
         # pred 시각화 
        draw_image = model.get_pred_show(image, mode=mode, threshold=threshold) 

        # gt 시각화 
        if gt: 
            ann_path = os.path.join(ann_dir, image_fname.replace('jpg', 'json'))
            if os.path.exists(ann_path): 
                with open(ann_path, 'r') as f: 
                    anns = json.load(f)['annotations'] 

                for ann in anns: 
                    x1, y1, x2, y2 = ann['bbox']
                    cat_id = ann['category_id'] 
                    draw_image = cv2.rectangle(draw_image, (x1, y1), (x2, y2), colors[cat_id], 7) 

        axes[i//(n_rows)][i%n_cols].imshow(draw_image) 
        axes[i//(n_rows)][i%n_cols].set_title(f'{image_fname} {i}') 
        axes[i//(n_rows)][i%n_cols].axis('off') 

    plt.tight_layout() 
    plt.show() 
                

