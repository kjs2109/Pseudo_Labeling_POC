import os, gc, math, time, json
import argparse
from tqdm.auto import tqdm
from importlib import import_module

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader 
import albumentations as A 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

from dataset import EvaluationDataset 
from inference.inference_model import load_model, load_ensemble_model 
from utils.helper import check_and_mkdir, calc_average_dice, calc_binary_cls_metrics, dice_filter_probability, auroc_analysis, plot_bin_roc_curve


def dice_coef(im1, im2):
    im1 = np.asarray(im1).astype(np.bool_)
    im2 = np.asarray(im2).astype(np.bool_)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    intersection = np.logical_and(im1, im2)
    dice_score = 2. * intersection.sum() / (im1.sum() + im2.sum())

    return dice_score if not math.isnan(dice_score) else 0.0


def make_output_result(model, data_loader, target_classes, saved_dir, thr=0.5): 
    '''
        auxiliary classification을 사용하지 않는 모델의 병변별 예측 결과를 csv 파일로 저장(file_name, 양성/음성, dice, probability)하는 함수   
    '''

    print('Start Evaluation...') 
    model = model.cuda() 
    model.eval() 

    rles = [] 
    filename_and_class = [] 
    all_class_result_dict = [({'File_name': [], 'GT_Class': [], 'Dice_score': [], 'Probability': []}, tc) for tc in target_classes]
    with torch.no_grad(): 
        for i, (images, gt_masks, filenames) in enumerate(data_loader): 
            images, gt_masks = images.cuda(), gt_masks.cuda()  

            output = model(images)
            logit_masks = output['mask_output']
            prob_masks = torch.sigmoid(logit_masks)
            pred_masks = (prob_masks > thr) 
            
            for gt_mask, prob_mask, pred_mask, filename in zip(gt_masks, prob_masks, pred_masks, filenames):  # BCHW 
                for cat_idx, (target_mask, target_prob, target_pred) in enumerate(zip(gt_mask, prob_mask, pred_mask)):  # target_class 별로 검사   CHW 
                    if target_mask.sum() == 0: #  normal 
                        gt_class = 'Normal' 
                        probability = target_prob.max().item() 
                        probability = probability if probability > thr else 0.0  
                        dice_score = 0.0 
                    else:  # abnormal 
                        gt_class = target_classes[cat_idx] 
                        probability = target_prob.max().item()  
                        probability = probability if probability > thr else 0.0
                        dice_score = dice_coef(target_mask.cpu().numpy(), target_pred.cpu().numpy()) 

                    all_class_result_dict[cat_idx][0]['File_name'].append(filename) 
                    all_class_result_dict[cat_idx][0]['GT_Class'].append(gt_class) 
                    all_class_result_dict[cat_idx][0]['Dice_score'].append(dice_score) 
                    all_class_result_dict[cat_idx][0]['Probability'].append(probability) 
                    
                    print(f"[{i+1:06d}/{len(data_loader):06d}] {filename:<15} {gt_class:<10} {dice_score:.4f} {probability:.4f}")
                    gc.collect() 

    check_and_mkdir(saved_dir) 
    for cls_idx, cls_name in enumerate(target_classes):
        cls_df = pd.DataFrame(all_class_result_dict[cls_idx][0])
        cls_df.to_csv(os.path.join(f'{saved_dir}/{cls_name}_dsc_based_probs.csv')) 

def make_output_result_with_auxiliary(model, data_loader, target_classes, saved_dir, use_ensemble, mode, thr=0.5): 
    print('Start Evaluation...') 
    if not use_ensemble:
        model = model.cuda() 
        model.eval() 

    all_class_result_dict = [({'File_name': [], 'GT_Class': [], 'Dice_score': [], 'Probability': []}, tc) for tc in target_classes] 
    with torch.no_grad(): 
        for i, (images, gt_masks, filenames) in enumerate(data_loader): 

            images, gt_masks = images.cuda(), gt_masks.cuda()  

            output = model(images)
            if use_ensemble:
                prob_masks = output['mask_output'] 
                pred_masks = (prob_masks > thr) 
                prob_cats = output['cls_output'] 
            else: 
                logit_masks = output['mask_output']
                prob_masks = torch.sigmoid(logit_masks)
                pred_masks = (prob_masks > thr) 
                prob_cats = torch.sigmoid(output['cls_output']) 

            for gt_mask, prob_mask, pred_mask, prob_cat, filename in zip(gt_masks, prob_masks, pred_masks, prob_cats, filenames): 
                for cat_idx, (target_gt_mask, target_prob_mask, target_pred_mask, target_prob) in enumerate(zip(gt_mask, prob_mask, pred_mask, prob_cat)): 
                    
                    mask_probability = target_prob_mask.max().item() 
                    cls_probability = target_prob.item() 

                    if mode == 'cls': 
                        probability = cls_probability 
                    elif mode == 'mean': 
                        probability = (cls_probability + mask_probability) / 2 
                    elif mode == 'mask': 
                        probability = mask_probability
                    else: 
                        raise ValueError(f'Invalid mode: {mode}')

                    probability = probability if probability > thr else 0.0  
                    
                    if target_gt_mask.sum() == 0: 
                        gt_class = 'Normal' 
                        dice_score = 0.0 
                    else: 
                        gt_class = target_classes[cat_idx] 
                        dice_score = dice_coef(target_gt_mask.cpu().numpy(), target_pred_mask.cpu().numpy()) 


                    all_class_result_dict[cat_idx][0]['File_name'].append(filename) 
                    all_class_result_dict[cat_idx][0]['GT_Class'].append(gt_class) 
                    all_class_result_dict[cat_idx][0]['Dice_score'].append(dice_score) 
                    all_class_result_dict[cat_idx][0]['Probability'].append(probability) 
                    
                    print(f"[{i+1:06d}/{len(data_loader):06d}] {filename:<15} {gt_class:<14} {dice_score:.4f} {probability:.4f}")
                    gc.collect() 

                if len(gt_mask) > 1:
                    print()
        
    check_and_mkdir(saved_dir) 
    for cls_idx, cls_name in enumerate(target_classes):
        cls_df = pd.DataFrame(all_class_result_dict[cls_idx][0])
        cls_df.to_csv(os.path.join(f'{saved_dir}/{cls_name}_dsc_based_probs.csv')) 
            

def evaluate(saved_dir, dice_thr=0.2): 

    train_tag_result_dict = {'Chest_finding': [],
                             f'DSC>{dice_thr}_AUROC': [], 
                             f'DSC>{dice_thr}_AUROC_Optimal_threshold': [],
                             f'DSC>{dice_thr}_Sensitivity': [],
                             f'DSC>{dice_thr}_Specificity': [],
                             f'DSC>{dice_thr}_Accuracy': [],
                             f'DSC>{dice_thr}_Precision': [],
                             f'DSC>{dice_thr}_F1-score': [],
                             'Dice_score': []} 

    for csv_fname in os.listdir(saved_dir): 
        if not csv_fname.endswith('.csv'):
            continue

        target_class = csv_fname.split('_')[0] 
        print('='*20, target_class, '='*20)  

        result_df = pd.read_csv(os.path.join(saved_dir, csv_fname)) 

        file_names = list(result_df['File_name'])
        gt_classes = [0 if gt_class == 'Normal' else 1 for gt_class in list(result_df['GT_Class'])] 
        dice_scores = list(result_df['Dice_score']) 
        probabilities = list(result_df['Probability']) 

        dice_probability = dice_filter_probability(gt_classes, dice_scores, probabilities, dsc_threshold=dice_thr)
 
        opt_thr, auc, fpr, tpr, thrs = auroc_analysis(gt_classes, dice_probability) 

        binary_cls_result = calc_binary_cls_metrics(gt_classes, dice_probability, threshold=opt_thr) 

        train_tag_result_dict['Chest_finding'].append(target_class)
        train_tag_result_dict['Dice_score'].append(calc_average_dice(gt_classes, dice_scores))
        train_tag_result_dict[f'DSC>{dice_thr}_AUROC'].append(auc)
        train_tag_result_dict[f'DSC>{dice_thr}_Sensitivity'].append(binary_cls_result['sensitivity'])
        train_tag_result_dict[f'DSC>{dice_thr}_Specificity'].append(binary_cls_result['specificity'])
        train_tag_result_dict[f'DSC>{dice_thr}_Accuracy'].append(binary_cls_result['accuracy'])
        train_tag_result_dict[f'DSC>{dice_thr}_Precision'].append(binary_cls_result['precision'])
        train_tag_result_dict[f'DSC>{dice_thr}_F1-score'].append(binary_cls_result['f1_score'])
        train_tag_result_dict[f'DSC>{dice_thr}_AUROC_Optimal_threshold'].append(opt_thr)
        
        # save result 
        check_and_mkdir(os.path.join(saved_dir, 'result'))

        ax = plot_bin_roc_curve(target_class, auc, fpr, tpr, title=f"{target_class}_dice-based_ROC_curve")
        plt.savefig(os.path.join(saved_dir, 'result', f'{target_class}_dice_ROC_curve.png')) 
        plt.close() 

        save_dataframe = pd.DataFrame(train_tag_result_dict)
        column_order = [
            'Chest_finding', 
            'Dice_score', 
            f'DSC>{dice_thr}_AUROC', 
            f'DSC>{dice_thr}_Sensitivity', 
            f'DSC>{dice_thr}_Specificity', 
            f'DSC>{dice_thr}_Accuracy', 
            f'DSC>{dice_thr}_Precision', 
            f'DSC>{dice_thr}_F1-score',
            f'DSC>{dice_thr}_AUROC_Optimal_threshold', 
            ]
        save_dataframe.to_csv(os.path.join(saved_dir, 'result', f'metric_analysis_same_thresh_{dice_thr}.csv'), header=True, columns=column_order, index=False)

        # print result 
        print(f"{target_class} dice-AUROC: {auc}") 
        print(f"{target_class} average_dice: {calc_average_dice(gt_classes, dice_scores)}")
        print(f"{target_class} dice-AUROC_Optimal_threshold: {opt_thr}")
        print(f"{target_class} dice-confusion_matrix: {binary_cls_result['confusion_matrix']}")
        print(f"{target_class} dice-sensitivity: {binary_cls_result['sensitivity']}")
        print(f"{target_class} dice-specificity: {binary_cls_result['specificity']}")
        print(f"{target_class} dice-accuracy: {binary_cls_result['accuracy']}")
        print(f"{target_class} dice-precision: {binary_cls_result['precision']}")
        print(f"{target_class} dice-f1_score: {binary_cls_result['f1_score']}")

        gc.collect() 

    # print average result 
    print('='*20, "Average", '='*20) 
    print(f"Average dice-AUROC: {np.mean(train_tag_result_dict[f'DSC>{dice_thr}_AUROC'])}") 
    print(f"Average DICE: {np.mean(train_tag_result_dict['Dice_score'])}") 
    print(f"Average Optimal_threshold: {np.mean(train_tag_result_dict[f'DSC>{dice_thr}_AUROC_Optimal_threshold'])}") 
    print(f"Average Sensitivity: {np.mean(train_tag_result_dict[f'DSC>{dice_thr}_Sensitivity'])}") 
    print(f"Average Specificity: {np.mean(train_tag_result_dict[f'DSC>{dice_thr}_Specificity'])}") 
    print(f"Average Accuracy: {np.mean(train_tag_result_dict[f'DSC>{dice_thr}_Accuracy'])}") 
    print(f"Average Precision: {np.mean(train_tag_result_dict[f'DSC>{dice_thr}_Precision'])}") 
    print(f"Average F1-score: {np.mean(train_tag_result_dict[f'DSC>{dice_thr}_F1-score'])}") 


def main(args, ensemble_model_list): 
    datetime = time.strftime('%m-%d-%H-%M', time.localtime(time.time()))
    if args.use_ensemble: 
        models = [] 
        exp_names = [] 
        weight_fnames = [] 
        for model_info in ensemble_model_list: 
            models.append(model_info['model_class']) 
            exp_names.append(model_info['exp_name'].split('_')[0]) 
            weight_fnames.append(model_info['weight_fname'])
        args.exp_name = '_'.join(exp_names)
        findings = str(len(args.classes)) + 'findings'
        saved_dir = os.path.join(args.saved_model_dir, 'ensemble_result', findings, args.exp_name, datetime) 
        args.model = 'EnsembleModel'
        args.model_class = ' '.join(models) 
        args.weight_fname = ' '.join(weight_fnames)

        check_and_mkdir(saved_dir)
        with open(os.path.join(saved_dir, 'ensemble_model_list.json'), 'w') as f: 
            json.dump({f'model{i}': model_info for i, model_info in enumerate(ensemble_model_list)}, f, indent=4)  
    else:
        saved_dir = os.path.join(args.saved_model_dir, args.exp_name, args.weight_fname.split('.')[0] + datetime)

    if args.no_inference:
        result_list = [dir_name for dir_name in os.listdir(os.path.join(args.saved_model_dir, args.exp_name)) if os.path.isdir(os.path.join(args.saved_model_dir, args.exp_name, dir_name))] 
        if len(result_list) == 0: 
            print('There is no result directory. please make inference result first') 
            return
        else: 
            for i, result_dir in enumerate(result_list): 
                print(f'[{i}] {result_dir}')  
            result_dir_name = result_list[int(input('Select result directory you want to evaluate(input type: int): '))]
            saved_dir = os.path.join(args.saved_model_dir, args.exp_name, result_dir_name) 
    else:  
        print('#'*20, 'Inference start!!', '#'*20)
        print(f'Model: {args.model}.{args.model_class}') 
        print(f'Exp name: {args.exp_name}')
        print(f'Model weight: {args.weight_fname}') 

        eval_dataset = EvaluationDataset(args.classes, args.image_root, args.label_root, transforms=A.Resize(args.resize, args.resize)) 
        eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0) 

        if args.use_ensemble:
            model = load_ensemble_model(ensemble_model_list, args.classes, args.saved_model_dir, args.ensemble_mode) 
        else: 
            weight_path = os.path.join(args.saved_model_dir, args.exp_name, args.weight_fname)
            model = load_model(args.model, args.model_class, args.classes, args.use_aux, weight_path)  

        tick = time.time() 
        if args.use_aux or args.use_ensemble: 
            make_output_result_with_auxiliary(model, eval_dataloader, args.classes, saved_dir, args.use_ensemble, mode=args.eval_mode) 
        else: 
            if args.eval_mode != 'cls': 
                print("You can use only 'cls' mode when you use model without auxiliary") 
                return 
            make_output_result(model, eval_dataloader, args.classes, saved_dir) 
        tock = time.time() 
        print(f'Inference time: {round(tock-tick) // 60} min', '\n') 


    print('#'*20, 'Evaluation start!!', '#'*20) 
    print(f'Model: {args.model}.{args.model_class}') 
    print(f'Exp name: {args.exp_name}')
    print(f'Model weight: {args.weight_fname}') 
    print(f'Evaluation mode: {args.eval_mode}')

    evaluate(saved_dir, dice_thr=0.2) 



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='unet', help='model type (default: fcn)') 
    parser.add_argument('--model_class', type=str, default='EfficientUNet', help='model class type (default: EfficientUNet)')
    parser.add_argument('--use_aux', type=bool, default=False, help='use auxilary loss (default: False)') 
    parser.add_argument('--eval_mode', type=str, default='mean', help='if use auxiliary, select {"cls" | "mean" | "mask"} for final classification probability (default: mean)')
    parser.add_argument('--use_ensemble', type=bool, default=False, help='use ensemble model (default: False)')
    parser.add_argument('--ensemble_mode', type=str, default='weighted_soft', help='if use ensemble, select {"soft" | "weighted_soft"}')
    parser.add_argument('--classes', type=str, nargs='+', default=['Pneumothorax'], help='class list (default: Pneumothorax) 5finding - Consolidation Pneumothorax Fibrosis Effusion Nodule') 

    # parser.add_argument('--image_root', type=str, default="/ai-data/chest/kjs2109/private_data/chestALL/images")  # MRCNN과 비교를 위해 실험 초기 평가에 사용한 데이터셋 (318번 줄 주석 처리  & dataset 수정 필요)
    # parser.add_argument('--label_root', type=str, default="/ai-data/chest/kjs2109/private_data/chestALL/anns/1findings_v1")
    parser.add_argument('--image_root', type=str, default="/ai-data/chest/kjs2109/pseudo_label_dataset/chestALL/images")
    parser.add_argument('--label_root', type=str, default="/ai-data/chest/kjs2109/pseudo_label_dataset/chestALL/annotations/")
    parser.add_argument('--saved_model_dir', type=str, default="/ai-data/chest/kjs2109/baseline/semseg-baseline/weight_dir") 
    parser.add_argument('--gpu_id', type=str, default="1", help='gpu id number (default: 1)') 
    parser.add_argument('--exp_name', type=str, default='exp3_effi_b5', help='experiment name') 
    parser.add_argument('--weight_fname', type=str, default="dice-8136-21999.pth", help='weight file name')
    parser.add_argument('--eval_institution', type=list, nargs='+', default=["BORA"], help='evaluation institution (default: ["BORA"])')
    parser.add_argument('--resize', type=int, default=512, help='resize size for image when you trained (default: 512)')
    parser.add_argument('--no_inference', action='store_true', help='if you want to inference, set True (default: True)') 

    args = parser.parse_args() 

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id 
    args.label_root = os.path.join(args.label_root, f'{len(args.classes)}findings/v0')

    # --use_ensemble True 설정하고 ensemble 할 모델 정보 입력 
    if args.use_ensemble:
        ensemble_model_list = [
            {
                'model': 'unet', 
                'model_class': 'EfficientUNet', 
                'use_aux': True, 
                'exp_name': 'exp7_1f_effiU-b5_aux-copypaste', 
                'weight_fname': 'dice-7822-55999.pth'
            },
            {
                'model': 'unet', 
                'model_class': 'EfficientUNet', 
                'use_aux': False, 
                'exp_name': 'exp10_1f_effiU-b5_base-copypaste', 
                'weight_fname': 'dice-7987-43999.pth'
            },
            {
                'model': 'unet', 
                'model_class': 'EfficientUNet', 
                'use_aux': False, 
                'exp_name': 'exp9_1f_effiU-b5_base-v1', 
                'weight_fname': 'dice-7876-159999.pth'
            },
        ]
    else: 
        ensemble_model_list = None

    main(args, ensemble_model_list) 