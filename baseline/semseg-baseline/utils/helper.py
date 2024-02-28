import os
import matplotlib.pyplot as plt
import sklearn
import numpy as np


def check_and_mkdir(target_path):
    print("Target_path: " + str(target_path))
    path_to_targets = os.path.split(target_path)

    path_history = '/'
    for path in path_to_targets:
        path_history = os.path.join(path_history, path)
        if not os.path.exists(path_history):
            os.mkdir(path_history) 

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

def calc_binary_cls_metrics(y_trues, y_probs,  threshold=0.5):
    disease_confusion_matrix = calc_confusion_matrix(y_trues,
                                                     y_probs, 
                                                     threshold=threshold)
    
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