import gc
import os
from os.path import join
import scikitplot as skplt
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import numpy as np
from matplotlib import pyplot


def convert_classes(gt_class_list):
    
    def conversion_rule(single_gt_cls):
        return 0 if single_gt_cls in ['Normal', 'normal'] else 1
    
    converted_classes = [conversion_rule(gt_cls) for gt_cls in gt_class_list]
    return converted_classes

def calc_confusion_matrix(y_trues, y_probs, threshold=0.5):
    confusion_matrix = [0, 0, 0, 0]     # TP, FN, FP, TN
    for t, p in zip(y_trues, y_probs):
        if t == 0:
            if p < threshold:
                confusion_matrix[3]+=1
            else:
                confusion_matrix[2]+=1
        else:
            if p < threshold:
                confusion_matrix[1]+=1
            else:
                confusion_matrix[0]+=1
    return confusion_matrix
   
def calc_sensitivity(conf_matrix):
    # sensitivity = TP / (TP + FN)
    if (conf_matrix[0] + conf_matrix[1]) == 0:
        return "N/A"
    else:
        return conf_matrix[0]/ (conf_matrix[0] + conf_matrix[1])

def calc_specificity(conf_matrix):
    # sensitivity = TN / (TN + FP)
    if (conf_matrix[3] + conf_matrix[2]) == 0:
        return "N/A"
    else:
        return conf_matrix[3]/ (conf_matrix[3] + conf_matrix[2])

def calc_accuracy(conf_matrix):
    if (conf_matrix[0]+conf_matrix[1]+conf_matrix[2]+conf_matrix[3]) == 0:
        return "N/A"
    else:
        return (conf_matrix[0]+conf_matrix[3]) / (conf_matrix[0]+conf_matrix[1]+conf_matrix[2]+conf_matrix[3])

def calc_precision(conf_matrix):
    if (conf_matrix[0]+conf_matrix[2]) == 0:
        return 0
        # return "N/A"
    else:
        return conf_matrix[0] / (conf_matrix[0]+conf_matrix[2])

def calc_f1_score(recall, precision):
    if (recall == 'N/A') or (precision == 'N/A') or (recall == 0) or (precision == 0):
        # return "N/A"
        return 0

    else:
        return (2*recall*precision) / (recall+precision)

def calc_average_dice(y_trues, dice_scores):
    dice_score_sum, pos_count = 0, 0
    for y_true, dsc in zip(y_trues, dice_scores):
        if y_true != 0:
            dice_score_sum += dsc
            pos_count += 1

    if pos_count != 0:
        return round(dice_score_sum / pos_count, 4) *100
    else:
        return 0

def calc_binary_cls_metrics(y_trues, y_probs, threshold=0.5):
    disease_confusion_matrix = calc_confusion_matrix(conv_gt_class,
                                                     probability, 
                                                     threshold=threshold)
    print(disease_confusion_matrix)
    sensitivity = calc_sensitivity(disease_confusion_matrix)
    precision = calc_precision(disease_confusion_matrix)
    metric_result_dict = {'confusion_matrix': disease_confusion_matrix,
                          'sensitivity': round(sensitivity, 4)*100,
                          'specificity': round(calc_specificity(disease_confusion_matrix), 4)*100,
                          'accuracy': round(calc_accuracy(disease_confusion_matrix), 4)*100,
                          'precision': round(precision, 4)*100,
                          'f1_score': round(calc_f1_score(sensitivity, precision), 4)*100
                          }
    return metric_result_dict


def dice_filter_probability(y_trues, dice_scores, probabilities, dsc_threshold=0.2):
    for i, (y_true, dsc, prob) in enumerate(zip(y_trues, dice_scores, probabilities)):
        if y_true != 0:
            if dsc < dsc_threshold:
                probabilities[i] = 0.0
    
    return probabilities

def auroc_analysis(y_trues, probabilities):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_trues, probabilities, pos_label=1)
    idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[idx]

    auc = round(sklearn.metrics.roc_auc_score(y_trues, probabilities), 4) *100
    print("roc_auc_score: " + str(auc))

    return round(optimal_threshold, 4), auc, fpr, tpr, thresholds

def plot_bin_roc_curve(disease, auc, fpr, tpr, title=''):
    text_fontsize="large"
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    if title == '':
        ax.set_title(f"{disease}_ROC_curve", fontsize=text_fontsize)
    else:
        ax.set_title(title, fontsize=text_fontsize)

    ax.plot(fpr, tpr, marker='.', label=f"{disease}: AUC={round(auc, 4)}")

    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=text_fontsize)
    ax.set_ylabel('True Positive Rate', fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    ax.legend(loc='lower right', fontsize=text_fontsize)
    return ax


if __name__=='__main__':
    eval_version='2'  
    weight_name = 'Pseudo_v6_BORA_EVAL_1Findings_2gpus_512bs_8workers_MRCNN_1024_0.0001LR_WCLR_mask_rcnn_R_50_FPN_3x_gn'
    prefix = 'Misc'
    weight_base_dir = f'/ai-data/chest/kjs2109/baseline/detectron2-baseline/weight_dir/{prefix}/{weight_name}/eval_csvs'
    dir_names = os.listdir(weight_base_dir)

    dice_threshold = 0.2

    train_tag_result_dict = {'Chest_finding': [],
                             'AUROC': [], f'DSC>{dice_threshold}_AUROC': [], 
                             'AUROC_Optimal_threshold': [], f'DSC>{dice_threshold}_AUROC_Optimal_threshold': [],
                             'Sensitivity': [], f'DSC>{dice_threshold}_Sensitivity': [],
                             'Specificity': [], f'DSC>{dice_threshold}_Specificity': [],
                             'Accuracy': [], f'DSC>{dice_threshold}_Accuracy': [],
                             'Precision': [], f'DSC>{dice_threshold}_Precision': [],
                             'F1-score': [], f'DSC>{dice_threshold}_F1-score': [],
                             'Dice_score': []}

    # dir_names = ['consolidation', 'fibrosis', 'nodule_mass', 'pleural_effusion', 'pneumothorax']
    # dir_names = ['model_best_0_4885_eval_data_v4']

    # version_name = 'v4'
    # csv_keyword = '_dsc_based_probs_{}.csv'.format(version_name)
    csv_keyword = '_dsc_'

    for i, dn in enumerate(dir_names):
        print('='*20)
        print(f"{i+1}/{len(dir_names)}: {dn}")
        csv_dir_path = join(weight_base_dir, dn)

        # if (not os.path.exists(csv_dir_path)) or os.path.exists(join(csv_dir_path, 'metric_analysis.csv')):
        #     continue
        csv_file_names = os.listdir(csv_dir_path)
        print(weight_name)
        print(csv_file_names)

        # csv_file_names = [csv_fn for csv_fn in csv_file_names if csv_keyword in csv_fn]

        writing_material_exists = False
        for j, cfn in enumerate(csv_file_names):
            if ('png' in cfn) or ('metric_analysis' in cfn):
                continue
            print('-'*20)
            print(cfn)
            print(cfn.split(csv_keyword)[0])
            disease = cfn.split(csv_keyword)[0]
            print(disease)
            csv_object = pd.read_csv(join(csv_dir_path, cfn))
            file_names = list(csv_object['File_name'])
            try:
                gt_class = list(csv_object['GT_Class'])
            except:
                gt_class = list(csv_object['Class'])
            conv_gt_class = convert_classes(gt_class)
            
            if len(np.unique(np.array(conv_gt_class))) in [0, 1]:
                continue
            dice_score = list(csv_object['Dice_score'])
            probability = list(csv_object['Probability'])

            gen_optimal_threshold, auc, fpr, tpr, thresholds = auroc_analysis(conv_gt_class, probability)

            ax = plot_bin_roc_curve(disease, auc, fpr, tpr)
            # plt.savefig(f'{csv_dir_path}/{disease}_ROC_curve.png')
            plt.close()

            binary_cls_result = calc_binary_cls_metrics(conv_gt_class, 
                                                        probability, 
                                                        threshold=gen_optimal_threshold)
            print(f"{disease} AUROC: {auc}")
            print(f"{disease} confusion_matrix: {binary_cls_result['confusion_matrix']}")
            print(f"{disease} sensitivity: {binary_cls_result['sensitivity']}")
            print(f"{disease} specificity: {binary_cls_result['specificity']}")
            print(f"{disease} accuracy: {binary_cls_result['accuracy']}")
            print(f"{disease} precision: {binary_cls_result['precision']}")
            print(f"{disease} f1_score: {binary_cls_result['f1_score']}")
            print(f"{disease} average_dice: {calc_average_dice(conv_gt_class, dice_score)}")

            train_tag_result_dict['Chest_finding'].append(disease)
            train_tag_result_dict['AUROC'].append(auc)
            train_tag_result_dict['AUROC_Optimal_threshold'].append(gen_optimal_threshold)
            train_tag_result_dict['Sensitivity'].append(binary_cls_result['sensitivity'])
            train_tag_result_dict['Specificity'].append(binary_cls_result['specificity'])
            train_tag_result_dict['Accuracy'].append(binary_cls_result['accuracy'])
            train_tag_result_dict['Precision'].append(binary_cls_result['precision'])
            train_tag_result_dict['F1-score'].append(binary_cls_result['f1_score'])
            train_tag_result_dict['Dice_score'].append(calc_average_dice(conv_gt_class, dice_score))
            

            dice_probability = dice_filter_probability(conv_gt_class, dice_score, probability, dsc_threshold=dice_threshold)

            # debug_csv = {'File_name': file_names, 'GT_Class': gt_class, 'Dice_score': dice_score, 'Dice-probability': dice_probability}
            # debug_dataframe = pd.DataFrame(debug_csv)
            # debug_dataframe.to_csv(join(csv_dir_path, f'{cfn[:-4]}_debug.csv'), header=True, index=False)

            optimal_threshold, auc, fpr, tpr, thresholds = auroc_analysis(conv_gt_class, dice_probability)

            ax = plot_bin_roc_curve(disease, auc, fpr, tpr, title=f"{disease}_dice-based_ROC_curve")
            plt.savefig(f'{csv_dir_path}/{disease}_dice_ROC_curve.png')
            plt.close()

            binary_cls_result = calc_binary_cls_metrics(conv_gt_class, 
                                                        dice_probability, 
                                                        # threshold=optimal_threshold)
                                                        threshold=gen_optimal_threshold)
            print(f"{disease} dice-AUROC: {auc}")
            print(f"{disease} dice-confusion_matrix: {binary_cls_result['confusion_matrix']}")
            print(f"{disease} dice-sensitivity: {binary_cls_result['sensitivity']}")
            print(f"{disease} dice-specificity: {binary_cls_result['specificity']}")
            print(f"{disease} dice-accuracy: {binary_cls_result['accuracy']}")
            print(f"{disease} dice-precision: {binary_cls_result['precision']}")
            print(f"{disease} dice-f1_score: {binary_cls_result['f1_score']}")

            train_tag_result_dict[f'DSC>{dice_threshold}_AUROC'].append(auc)
            # train_tag_result_dict[f'DSC>{dice_threshold}_AUROC_Optimal_threshold'].append(optimal_threshold)
            train_tag_result_dict[f'DSC>{dice_threshold}_AUROC_Optimal_threshold'].append(gen_optimal_threshold)
            train_tag_result_dict[f'DSC>{dice_threshold}_Sensitivity'].append(binary_cls_result['sensitivity'])
            train_tag_result_dict[f'DSC>{dice_threshold}_Specificity'].append(binary_cls_result['specificity'])
            train_tag_result_dict[f'DSC>{dice_threshold}_Accuracy'].append(binary_cls_result['accuracy'])
            train_tag_result_dict[f'DSC>{dice_threshold}_Precision'].append(binary_cls_result['precision'])
            train_tag_result_dict[f'DSC>{dice_threshold}_F1-score'].append(binary_cls_result['f1_score'])
            
            writing_material_exists = True

            gc.collect()
        
        if writing_material_exists:
            save_dataframe = pd.DataFrame(train_tag_result_dict)
            column_order = [
                'Chest_finding', 'Dice_score', f'DSC>{dice_threshold}_AUROC', f'DSC>{dice_threshold}_Sensitivity', 
                f'DSC>{dice_threshold}_Specificity', f'DSC>{dice_threshold}_Accuracy', f'DSC>{dice_threshold}_F1-score',
                f'DSC>{dice_threshold}_Precision', f'DSC>{dice_threshold}_AUROC_Optimal_threshold', 
                'AUROC','Sensitivity', 'Specificity','Accuracy', 'F1-score',  'Precision',  'AUROC_Optimal_threshold']
            save_dataframe.to_csv(join(csv_dir_path, f'metric_analysis_same_thresh_{dice_threshold}.csv'), header=True, columns=column_order, index=False)

