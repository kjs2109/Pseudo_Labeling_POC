import sys, os, gc
from os.path import join
from pathlib import Path
sys.path.append('/ai-data/chest/kjs2109/baseline/detectron2-baseline')
import numpy as np
import cv2
import math
from torch.nn import functional as F
import pandas as pd
from tqdm.auto import tqdm 

from inference.MaskRCNN_prob_map import *     

def check_and_mkdir(target_path):
    print("Target_path: " + str(target_path))
    path_to_targets = os.path.split(target_path)
    print("path_to_targets: " + str(path_to_targets))

    path_history = '/'
    for path in path_to_targets:
        path_history = os.path.join(path_history, path)
        if not os.path.exists(path_history):
            os.mkdir(path_history)

def dice(im1, im2, k=1):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        
    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    if k == 255:
        im1 = im1/255
        im2 = im2/255

    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())

if __name__=='__main__':

    """
    # Setup for Class information
    """
    TARGET_CLASS = ['Pneumothorax']  
    target_class_sub_dir_name = ['Pneumothorax']
    # TARGET_CLASS = ['Consolidation', 'Pneumothorax', 'Fibrosis', 'Effusion', 'Nodule'] 
    # target_class_sub_dir_name = ['Consolidation', 'Pneumothorax', 'Fibrosis', 'Pleural_effusion', 'Nodule_Mass']

    """
    # Set up for Mask-RCNN
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'  ##################################################### 
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    config_fname = "Misc/mask_rcnn_R_50_FPN_3x_gn.yaml" 
    train_tag = 'Pseudo_v6_BORA_EVAL_1Findings_2gpus_512bs_8workers_MRCNN_1024_0.0001LR_WCLR_mask_rcnn_R_50_FPN_3x_gn'  ################################################
    eval_institution = 'BORA'
    weight_file_name = 'model_best_0.0000-161783.pth'  #####################################################
    dataset_version = 2

    SEG_CONFIG_DICT = {
        'model_zoo_config': config_fname,
        'input_min_size_test': 1024,
        'input_max_size_test': 1024,
        'model_roi_heads_batch': 512,
        'model_roi_heads_threshold': 0.5,
        'CLASSES': TARGET_CLASS,
        'device': device,
    }
    # {config_fname.split("/")[0]}
    seg_weight_path = f'/ai-data/chest/kjs2109/baseline/detectron2-baseline/weight_dir/{config_fname.split("/")[0]}/{train_tag}/{weight_file_name}'
    print(f"seg_weight_path: {seg_weight_path}")

    seg_model = MaskRCNN_NDL(SEG_CONFIG_DICT, seg_weight_path, device)

    """
    For Dataset setup
    """

    test_image_base_dir = '/ai-data/chest/DATA/PrivateDataset/chestALL/images/'
    test_image_file_names = [tifn for tifn in os.listdir(test_image_base_dir) if eval_institution in tifn]
    test_image_file_names.sort()

    """
    # For result recording
    """
    # {config_fname.split("/")[0]}
    csv_save_dir_path = f'/ai-data/chest/kjs2109/baseline/detectron2-baseline/weight_dir/{config_fname.split("/")[0]}/{train_tag}/eval_csvs/_eval_data_v{dataset_version}-{weight_file_name.replace(".pth", "")}'
    print(f"csv_save_dir_path: {csv_save_dir_path}")
    check_and_mkdir(csv_save_dir_path)

    # target class의 결과 저장 딕셔너리 
    all_class_result_dict = [({'File_name': [], 'GT_Class': [], 'Dice_score': [], 'Probability': []}, tc) for tc in TARGET_CLASS]

    ################################################### 모든 test image 별로 inference 평가  ####################################################
    for i, image_name in enumerate(test_image_file_names):
        print('='*40)
        print(f"{i+1}/{len(test_image_file_names)}: {image_name}")
        tic = time.time()

        image = cv2.imread(join(test_image_base_dir, image_name))

        for cls_idx, cls_name in enumerate(TARGET_CLASS):
            test_mask_dir_path = f'/ai-data/chest/hyoon/dataset/chest_findings/mask_by_classes_v{dataset_version}/{target_class_sub_dir_name[cls_idx]}/'
            seg_mask, class_max_score = seg_model.get_seg_mask_and_scores(image, cls_name)

            if not os.path.exists(join(test_mask_dir_path, image_name[:-4]+'.png')):
                mask = np.zeros((image.shape[0], image.shape[1]))
                n_or_abn = 'Normal'
                print("Normal")
                dsc = 0.0
            else:
                mask = cv2.imread(join(test_mask_dir_path, image_name[:-4]+'.png'), 0)
                n_or_abn = cls_name
                print(cls_name)
                dsc = dice(seg_mask, mask, k=255)
                dsc = 0.0 if math.isnan(dsc) else dsc
                        
            #                      class 저장_리스트  col 
            all_class_result_dict[cls_idx][0]['File_name'].append(image_name)
            all_class_result_dict[cls_idx][0]['GT_Class'].append(n_or_abn)
            all_class_result_dict[cls_idx][0]['Dice_score'].append(dsc)
            all_class_result_dict[cls_idx][0]['Probability'].append(class_max_score)            
            gc.collect()

        print("time took: " + str(time.time()-tic)) 

    for cls_idx, cls_name in enumerate(target_class_sub_dir_name):
        cls_df = pd.DataFrame(all_class_result_dict[cls_idx][0])
        cls_df.to_csv(f'{csv_save_dir_path}/{cls_name}_dsc_based_probs_v{dataset_version}.csv')
