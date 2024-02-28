import os 
import gc 
import json
import random
import datetime
import argparse 
import multiprocessing
from importlib import import_module 

import numpy as np 
from tqdm.auto import tqdm

import torch 
import torch.nn as nn 
from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F
import torch.optim as optim 
from torch.utils.data import DataLoader 
from torch.utils.data.distributed import DistributedSampler 

import wandb 
from dataset import MyXRayDataset, MyXRayDatasetV1  
from utils.loss import create_criterion 
from utils.augmentation import create_transforms 
from utils.scheduler import CosineAnnealingWarmUpRestarts

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)  


def save_config(args, exp_memo): 
    args_dict = vars(args)
    args_dict.update({'exp_memo': exp_memo}) 
    args_dict.update({'date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
    exp_dir = os.path.join(args.saved_model_dir, args.name)
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir) 
    with open(os.path.join(exp_dir, 'exp_config.json'), 'w') as f:
        json.dump(args_dict, f, indent=4)


def save_model(model, dice, loss, iteration, saved_model_dir, name, mode):

    if mode == 'dice': 
        dice_score = str(dice).split('.')[1][:4]
        file_name = f'dice-{dice_score}-{iteration}.pth'

    elif mode == 'loss': 
        loss_score = str(loss).split('.')[1][:4] 
        file_name = f'loss-{loss_score}-{iteration}.pth' 
    else: 
        raise ValueError(f'Unknown mode: {mode}') 
    
    output_folder_path = os.path.join(saved_model_dir, name)
    output_file_path = os.path.join(output_folder_path, file_name) 
    
    if not os.path.exists(output_folder_path):
        os.mkdir(output_folder_path)
    
    dice_file_list = sorted([fname for fname in os.listdir(output_folder_path) if (fname.startswith('dice') and fname.endswith('.pth'))])
    loss_file_list = sorted([fname for fname in os.listdir(output_folder_path) if (fname.startswith('loss') and fname.endswith('.pth'))])

    if len(dice_file_list) >= 5: 
        os.remove(os.path.join(output_folder_path, dice_file_list[0])) 
    if len(loss_file_list) >= 3: 
        os.remove(os.path.join(output_folder_path, loss_file_list[-1])) 
        
    torch.save(model.module.state_dict(), output_file_path) 
    

def get_lr(optimizer): 
    for param_group in optimizer.param_groups: 
        return param_group['lr'] 


def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten(2)  # (N, C, H * W) 
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)
    
    eps = 0.0001
    return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps) 


def accuracy_fn(y_true, y_pred, mode='average'):
    if mode == 'average': 
        correct = torch.eq(y_true, y_pred).sum().item() 
        acc = correct / (len(y_pred)*len(y_true[0]))
    elif mode == 'per_class':
        correct = torch.eq(y_true, y_pred).sum(dim=0).cpu().numpy()  # NC -> C 
        acc = correct / len(y_pred)
    else: 
        raise ValueError(f'Unknown mode: {mode}')
    return acc


def make_display_image(mask_dict): 
    results = [] 
    for class_name, mask in mask_dict.items(): 
        h, w = mask[0].shape  
        result = np.array(mask).transpose(1, 0, 2).reshape(h, -1) 
        results.append(wandb.Image(result, caption=f'class: {class_name}')) 

    return results 


def validation(iteration, epoch, model, data_loader, criterion, args, local_gpu_id, thr=0.5):
    print(f'Start validation... (epoch-iter: {epoch+1:2d}-{iteration+1})')
    model.eval()

    total_loss, total_data_cnt, cnt = (0, 0, 0) 

    dices = []  
    accuracies = [] 

    gt_mask_dict = {class_name:[] for class_name in args.classes}
    pred_mask_dict = {class_name:[] for class_name in args.classes}
    
    with torch.no_grad():
        for batch in tqdm(data_loader):

            images, masks, cls_labels = batch[0], batch[1], batch[2]
            images, masks, cls_labels = images.to(local_gpu_id), masks.to(local_gpu_id), cls_labels.to(local_gpu_id)   

            outputs = model(images) 
            logit_masks = outputs['mask_output']   
            prob_masks = torch.sigmoid(logit_masks) 
            output_masks = (prob_masks > thr)  # (B, C, H, W) 

            if args.use_aux: 
                loss = 0.3*criterion['cls_criterion'](outputs['cls_output'], cls_labels) + 0.7*criterion['mask_criterion'](logit_masks, masks) 
                cls_preds = torch.round(torch.sigmoid(outputs['cls_output']))  # NC 
            else: 
                h, w = masks.shape[-2:]
                loss = criterion['mask_criterion'](logit_masks, masks) 
                cls_preds = torch.round(prob_masks.view(len(images), -1, h*w).max(dim=-1)[0]) 

            acc = accuracy_fn(cls_labels, cls_preds, mode='per_class')  # C 
            accuracies.append(acc)

            total_loss += loss 
            dice = dice_coef(output_masks.detach().cpu(), masks.detach().cpu()) # (N, C) 
            dices.append(dice) 

            for mask, output_mask in zip(masks, output_masks): 
                if args.log_image and local_gpu_id == 0 and cnt < 9: 
                    cnt += 1 
                    for i, class_name in enumerate(args.classes): 
                        gt_mask_dict[class_name].append(mask[i].detach().cpu().numpy())  # HW append 
                        pred_mask_dict[class_name].append(output_mask[i].detach().cpu().numpy()) 
                        gc.collect() 
                gc.collect() 

            total_data_cnt += len(images) 
            gc.collect() 

        if args.log_image and local_gpu_id == 0:
            wandb.log({
                # 'Valid Image': [wandb.Image(image.squeeze().detach().cpu().numpy().transpose(1, 2, 0), caption='Valid Image')], 
                'GT masks': make_display_image(gt_mask_dict) if epoch == 0 else None,  
                'Pred masks': make_display_image(pred_mask_dict)
            })

        dices = torch.cat(dices, 0)  # (val_size, C) val_size = B*step (모든 validation 데이터)
        dices_per_class = torch.mean(dices, 0) 
        avg_dice = torch.mean(dices_per_class).item()
        dice_per_class_dict = {f'Val/dice-{c}': d.item() for c, d in zip(args.classes, dices_per_class)} 

        acc_per_class = np.mean(accuracies, 0)
        avg_acc = np.mean(accuracies)  
        acc_per_class_dict = {f'Val/acc-{c}': a.item() for c, a in zip(args.classes, acc_per_class)}

        val_loss = total_loss / total_data_cnt

        print(f'Number of evaluation data: {total_data_cnt} | Valid Loss: {val_loss}, Avg acc: {avg_acc:.4f}, Avg dice: {avg_dice:.4f}', '\n') 
        acc_str = [f"{c:<12}: {a:.4f}" for c, a in acc_per_class_dict.items()] 
        acc_str = "\n".join(acc_str) 
        print(acc_str, '\n')  
        dice_str = [f"{c:<12}: {d:.4f}" for c, d in dice_per_class_dict.items()]
        dice_str = "\n".join(dice_str) 
        print(dice_str, '\n') 

        dice_per_class_dict['Val/loss'] = val_loss 
        dice_per_class_dict['Val/avg_dice'] = avg_dice 
        dice_per_class_dict['Val/avg_acc'] = avg_acc 
        dice_per_class_dict.update(acc_per_class_dict) 

    gc.collect()
    torch.cuda.empty_cache()

    return val_loss, avg_dice, dice_per_class_dict


def setup_for_distributed(is_master):
    """
        master proccess에서만 print 사용 
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_training(rank, args): 
    args.gpu = args.gpu_ids[rank]  
    print(f'Use GPU: {args.gpu} for training') 

    torch.distributed.init_process_group(backend='nccl', 
                                        init_method=f'tcp://127.0.0.1:{args.port}', 
                                        world_size=len(args.gpu_ids),
                                        rank=rank)
    torch.cuda.set_device(rank) 
    torch.distributed.barrier(device_ids=[rank]) # 다른 프로세스의 작업이 끝날때까지 대기 
    setup_for_distributed(rank == 0) 
    if rank == 0: 
        if args.findings == 1: 
            wandb.init(project="{{'프로젝트1'}}", entity='{{"wandb username"}}', reinit=True) 
        elif args.findings == 5: 
            wandb.init(project="{{'프로젝트2'}}", entity='{{"wandb username"}}', reinit=True) 

        wandb.run.name = args.name

def train(rank, args): 
    # Set device 
    local_gpu_id = rank  
    set_seed(args.seed)  
    init_distributed_training(local_gpu_id, args) 
    
    # Get agmentation 
    tf = create_transforms(args.augmentation)
    
    # Make dataset 
    train_dataset = MyXRayDatasetV1(args.classes, args.image_root, args.label_root, args.findings, args.ann_version, 
                                    is_train=True, 
                                    copypaste=args.copypaste,  
                                    num_copypaste=args.num_copypaste,
                                    transforms=tf)
    valid_dataset = MyXRayDatasetV1(args.classes, args.image_root, args.label_root, args.findings, args.ann_version, 
                                    is_train=False, ) 

    # Load Data 
    train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True)
    batch_sampler_train = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_sampler=batch_sampler_train, 
        num_workers= 4, 
    )

    valid_loader = DataLoader(
        dataset=valid_dataset, 
        batch_size=args.batch_size,  
        shuffle=False,
        num_workers=4, 
        drop_last=False, 
    )
    
    # Define model  
    model_module = getattr(import_module(f"models.{args.model}"), args.model_class)  
    model = model_module(args.classes, args.use_aux)

    if args.transfer_learning == True:
        model.load_state_dict(torch.load('/ai-data/chest/kjs2109/baseline/semseg-baseline/weight_dir/exp26_1f_effiU-b7_aux-copypaste50-noise1-v6/dice-7825-240999.pth'))

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda(local_gpu_id)
    model = DistributedDataParallel(module=model, device_ids=[local_gpu_id], find_unused_parameters=True) 
    
    # Define Loss & optimizer 
    criterion = {
        'mask_criterion': create_criterion(args.criterion),  
        'cls_criterion': nn.BCEWithLogitsLoss() if args.use_aux else None  
    }
    
    opt_module = getattr(import_module("torch.optim"), args.optimizer)
    optimizer = opt_module(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-6
    )
    if args.use_scheduler:
        scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=16000, T_mult=1, eta_max=0.0003, T_up=1600, gamma=0.5)
    
    # Train model
    print(f'Start training..')
    
    iteration = 0 
    best_dice = 0. 
    min_loss = 1000.
    top5_dice = [] 
    
    for epoch in range(args.epochs):
        model.train()
        train_sampler.set_epoch(epoch) 

        loss_value = 0
        accuracy = 0 
        for step, batch in enumerate(train_loader):            
            # gpu 연산을 위해 device 할당
            images, masks, cls_labels = batch[0], batch[1], batch[2]
            images, masks, cls_labels = images.to(local_gpu_id), masks.to(local_gpu_id), cls_labels.to(local_gpu_id) 

            # inference
            outputs = model(images)
            
            # loss & accuracy 계산 
            if args.use_aux: 
                mask_loss = criterion['mask_criterion'](outputs['mask_output'], masks)
                cls_loss = criterion['cls_criterion'](outputs['cls_output'], cls_labels)  
                loss = 0.3*cls_loss + 0.7*mask_loss  
                pred_labels = torch.round(torch.sigmoid(outputs['cls_output']))  # NC 
            else:
                h, w = masks.shape[-2:]
                loss = criterion['mask_criterion'](outputs['mask_output'], masks)  
                prob_masks = torch.sigmoid(outputs['mask_output'])
                pred_labels = torch.round(prob_masks.view(len(images), -1, h*w).max(dim=-1)[0])  # (B, C, H, W) -> (B, C, H*W) -> (B, C)
            
            accuracy += accuracy_fn(cls_labels, pred_labels) # y_true, y_pred

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iteration += 1 
            
            loss_value += loss.item()
            
            # step 주기에 따른 loss 출력
            if (step + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval 
                train_acc = accuracy / args.log_interval  
                current_lr = get_lr(optimizer) 
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f'Epoch [{epoch+1}/{args.epochs}], '
                    f'Step: [{step+1}/{len(train_loader)}], ' 
                    f'Total iter: {iteration} | '
                    f'Loss: {round(train_loss,4)}, ' 
                    f'Accuracy: {round(train_acc, 4)}',
                )
                if local_gpu_id == 0: 
                    wandb.log({
                        "Train/loss": train_loss, 
                        "Train/accuracy": train_acc,
                        "Lr": current_lr
                    })
                loss_value = 0 
                accuracy = 0 
             
            if args.use_scheduler: 
                scheduler.step() 

            # validation 주기에 따른 loss 출력 및 best model 저장
            if (iteration + 1) % args.eval_period == 0: 
                val_loss, dice, eval_log = validation(iteration, epoch, model, valid_loader, criterion, args, local_gpu_id=local_gpu_id, thr=0.5)

                if local_gpu_id == 0:
                    wandb.log(eval_log)
                    if args.log_image:
                        wandb.log({'Train Images': [wandb.Image(images[0].squeeze().detach().cpu().numpy().transpose(1, 2, 0), caption='image')]})
                
                    top5_dice.append(dice)
                    top5_dice.sort(reverse=True)
                    if len(top5_dice) > 5: 
                        top5_dice.pop() 

                    if best_dice <= dice:
                        print(f"Best performance at epoch(iteration): {epoch + 1}({iteration}), {best_dice:.4f} -> {dice:.4f}")
                        print(f"Save model in {args.saved_model_dir}/{args.name}")
                        best_dice = dice
                        save_model(model, dice, val_loss, iteration, args.saved_model_dir, args.name, mode='dice') 
                    else: 
                        if dice in top5_dice:  
                            save_model(model, dice, val_loss, iteration, args.saved_model_dir, args.name, mode='dice') 

                    if min_loss >= val_loss: 
                        print(f"Best loss at epoch(iteration): {epoch + 1}({iteration}), {min_loss:.4f} -> {val_loss:.4f}")
                        print(f"Save model in {args.saved_model_dir}/{args.name}")
                        min_loss = val_loss 
                        save_model(model, dice, val_loss, iteration, args.saved_model_dir, args.name, mode='loss') 

            torch.distributed.barrier()

            gc.collect()
        gc.collect() 

if __name__ == '__main__': 
    parser = argparse.ArgumentParser() 
    
    # parser.add_argument('--image_root', type=str, default="/ai-data/chest/DATA/PrivateDataset/chestALL/images/")
    # parser.add_argument('--label_root', type=str, default="/ai-data/chest/kjs2109/private_data/chestALL/anns/1findings_v1")  # select 
    parser.add_argument('--image_root', type=str, default="/ai-data/chest/kjs2109/pseudo_label_dataset/chestALL/images") 
    parser.add_argument('--label_root', type=str, default="/ai-data/chest/kjs2109/pseudo_label_dataset/chestALL/annotations") 
    parser.add_argument('--saved_model_dir', type=str, default="/ai-data/chest/kjs2109/baseline/semseg-baseline/weight_dir") 

    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)') 
    parser.add_argument('--gpu_ids', type=str, nargs='+', default=['6', '7', '8', '9'], help='gpu index list (default: 6 7 8 9)')
    parser.add_argument('--port', type=str, default='51156', help='port number (default: 51156)')
    parser.add_argument('--findings', type=int, default=1, help='findings (default: 1)') 
    parser.add_argument('--ann_version', type=int, default=0, help='annotation version (default: v0)')
    parser.add_argument('--classes', type=str, nargs='+', default=['Pneumothorax'], help='class list (default: Pneumothorax) 5finding - Consolidation Pneumothorax Fibrosis Effusion Nodule')  
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train (default: 50)') 
    parser.add_argument('--copypaste', type=float, default=0.0, help='copy and paste (default: False)')
    parser.add_argument('--num_copypaste', type=int, default=1, help='number of copy-paste (default: 1)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument('--batch_size', type=int, default=4, help='input batch size per number of gpus for training (default: 2)') 
    parser.add_argument('--log_interval', type=int, default=100, help='how many batches to wait before logging training status')
    parser.add_argument('--eval_period', type=int, default=1000, help='how many batches to wait before logging validation status')
    parser.add_argument('--model', type=str, default='unet', help='model type (default: fcn)') 
    parser.add_argument('--model_class', type=str, default='EfficientUNet', help='model class type (default: EfficientUNet)')
    parser.add_argument('--transfer_learning', type=bool, default=False, help='if you want to use transfer learning, set --transfer_learning True & write weight_file path at line 273 (default: False)') 
    parser.add_argument('--use_aux', type=bool, default=False, help='use aux (default: False)') 
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: Adam)')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 1e-4)')
    parser.add_argument('--use_scheduler', type=bool, default=False, help='use scheduler (default: False)')
    parser.add_argument('--log_image', type=bool, default=False, help='log image (default: False)') 
    parser.add_argument('--criterion', type=str, default='BCEWithLogitsLoss', help='criterion type (default: cross_entropy)')
    parser.add_argument('--name', default='exp', help='saving model at ./weight_dir/{name}')


    args = parser.parse_args()
    exp_memo = ''  # 실험 내용 메모 
    save_config(args, exp_memo) 

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu_ids)

    torch.multiprocessing.spawn(train, args=(args,), nprocs=len(args.gpu_ids))
    
    