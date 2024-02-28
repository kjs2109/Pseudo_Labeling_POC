import os 
import torch

from detectron2.engine import DefaultTrainer, hooks, launch
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader, build_detection_test_loader, DatasetMapper  

from annotations import get_dicts_from_json   

import detectron2.data.transforms as T
from detectron2.evaluation import (CityscapesInstanceEvaluator, CityscapesSemSegEvaluator, COCOEvaluator,
                                   COCOPanopticEvaluator,
                                   DatasetEvaluators, LVISEvaluator, PascalVOCDetectionEvaluator, SemSegEvaluator,) 
import detectron2.utils.comm as comm 
from fvcore.nn.precise_bn import get_bn_modules 

from helper import BestCheckpointer, WandB_Printer, check_and_mkdir
from config import setup 
from config import XML_DIRs, CLASSES, TRAIN_INSTITUTION, WEIGHT_DIR, gpu_indexes, INPUT_LENGTH, NUM_WORKERS, NUM_GPU


class MyTrainer(DefaultTrainer):
    """Below is for ADAM optimizer. Default is SGD."""
    '''
    @classmethod
    def build_optimizer(cls, cfg: CfgNode, model: torch.nn.Module) -> torch.optim.Optimizer:
        """
        Build an optimizer from config.           
        """
        params = get_default_optimizer_params(
            model,
            base_lr=cfg.SOLVER.BASE_LR,
            weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
            bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
            weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
        )
        return maybe_add_gradient_clipping(cfg, torch.optim.Adam)(
            params,
            lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    '''
    @classmethod
    def build_train_loader(cls, cfg):  # train data에 대한 augmentation 
        augmentations = [
            T.ResizeShortestEdge([INPUT_LENGTH], max_size=INPUT_LENGTH, sample_style='choice'),
            T.Resize((INPUT_LENGTH, INPUT_LENGTH)),
            T.RandomFlip(prob=.5, horizontal=True, vertical=False),
            T.RandomRotation([-10, 10], expand=False),
            T.RandomBrightness(.5, 2.),
            T.RandomContrast(.9, 1.1),
        ]
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, is_train=True, augmentations=augmentations))

    @classmethod  
    def build_test_loader(cls, cfg, dataset_name): 
        augmentations = [
            T.ResizeShortestEdge([INPUT_LENGTH], max_size=INPUT_LENGTH, sample_style='choice'),
            T.RandomFlip(prob=.5, horizontal=True, vertical=False),
            # T.RandomRotation([-10, 10], expand=False),
            # T.RandomBrightness(.5, 2.),
            # T.RandomContrast(.9, 1.1),
        ]
        return build_detection_test_loader(cfg, dataset_name, mapper=DatasetMapper(cfg, is_train=True, augmentations=augmentations))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        # case 1 
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))

        # case 2 
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))

        # case 3 
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        
        if len(evaluator_list) == 0:
            raise NotImplementedError(f"no Evaluator for the dataset {dataset_name} with the type {evaluator_type}")
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        
        return DatasetEvaluators(evaluator_list) 


    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        # 최종적으로 return 되는 hook을 담은 list 
        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(),
            BestCheckpointer(),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            ) if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model) else None,
        ]

        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # 현재까지 [IterationTimer, LRScheduler, PeriodicCheckpointer, test_and_save_results, EvalHook]
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        
        if comm.is_main_process():
            # wandb 추가 
            ret.append(hooks.PeriodicWriter([WandB_Printer(name = self.cfg.OUTPUT_DIR.split("/")[8], project="{{'프로젝트명'}}",entity="{{'wandb username'}}")],period=1))
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=100))

        return ret 
        

def main(): 
    # 1. 데이터셋 등록 
    for dataset_mode in ['train', 'valid', 'test']: 
        try: 
            if dataset_mode == 'train': 
                DatasetCatalog.register(dataset_mode, lambda d=dataset_mode: get_dicts_from_json(XML_DIRs, d, findings='1', ann_version='6', train_institution=TRAIN_INSTITUTION))
                MetadataCatalog.get(dataset_mode).set(thing_classes=CLASSES) 
            elif dataset_mode == 'valid':
                DatasetCatalog.register(dataset_mode, lambda d=dataset_mode: get_dicts_from_json(XML_DIRs, d, findings='1', ann_version='6', train_institution=TRAIN_INSTITUTION))
                MetadataCatalog.get(dataset_mode).set(thing_classes=CLASSES, evaluator_type="coco")
        except AssertionError: 
            print(f'{dataset_mode} is already registered') 

    # 2. config 설정 
    cfg = setup()
    check_and_mkdir(WEIGHT_DIR)
    with open(os.path.join(WEIGHT_DIR, 'config.yaml'), 'w') as f:
        f.write(cfg.dump()) 
    
    # 3. trainer 생성 (augmentation은 trainer 내부에서 세팅)
    trainer = MyTrainer(cfg) 
    trainer.resume_or_load(resume=False)  # train 새로 시작 
    torch.cuda.empty_cache() 

    return trainer.train() 


if __name__ == "__main__":  

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_indexes

    # 4. 학습 시작 
    launch(main, num_gpus_per_machine=NUM_GPU, dist_url='auto')
