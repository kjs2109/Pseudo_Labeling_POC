from detectron2.config import get_cfg  
from detectron2 import model_zoo 

############################### train_tag 만들기 ############################################

EVAL_INSTITUTION = ['BORA'] 
TRAIN_INSTITUTION = ['KU', 'CB', 'PAIK']
DCM_DIRs = ['/ai-data/chest/DATA/PrivateDataset/chestALL/images/']
XML_DIRs = ['/ai-data/chest/DATA/PrivateDataset/chestALL/xmls/']

# CLASSES = ['Consolidation', 'Pneumothorax', 'Fibrosis', 'Effusion', 'Nodule'] 
CLASSES = ['Pneumothorax']
CONFIG_NAME = "Misc/mask_rcnn_R_50_FPN_3x_gn.yaml"
# CONFIG_NAME = "Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml"
prefix = 'Pseudo_v6'  # 실험 내용 간단히 표현 
INPUT_LENGTH = 1024 
ROI_HEADS_BATCH_SIZE = 512 # default: 512
LEARNING_RATE = 1e-4
SCHEDULER_NAME = 'WarmupCosineLR'

NUM_WORKERS = 8 

gpu_indexes = '0,1'
list_of_gpu = [g for g in gpu_indexes.split(',') if g != '']
NUM_GPU = len(list_of_gpu)

scheduler_short_name = 'WCLR' if SCHEDULER_NAME == 'WarmupCosineLR' else 'WMSLR'
config_file = (prefix, CONFIG_NAME)

base_weight_dir = f'/ai-data/chest/kjs2109/baseline/detectron2-baseline/weight_dir/{CONFIG_NAME.split("/")[0]}'  # 모델 
train_tag =f'{config_file[0]}_{"_".join(EVAL_INSTITUTION)}_EVAL_{len(CLASSES)}Findings_{NUM_GPU}gpus_{ROI_HEADS_BATCH_SIZE}bs_{NUM_WORKERS}workers_MRCNN_{INPUT_LENGTH}_{LEARNING_RATE}LR_{scheduler_short_name}_{CONFIG_NAME.split("/")[-1].replace(".yaml","")}'

WEIGHT_DIR = f'{base_weight_dir}/{train_tag}/' # 이번 실험의 OUTPUT_DIR 경로 


def setup():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(CONFIG_NAME))
    
    # dataset 
    cfg.DATASETS.TRAIN = ("train",)  #  List of the dataset names for training. Must be registered in DatasetCatalog
    cfg.DATASETS.TEST = ("valid",)

    # dataloader 
    cfg.DATALOADER.NUM_WORKERS = NUM_WORKERS  
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False

    cfg.INPUT.MASK_FORMAT = 'polygon'
    cfg.INPUT.MIN_SIZE_TEST = INPUT_LENGTH 
    cfg.INPUT.MAX_SIZE_TEST = INPUT_LENGTH 

    # model 
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = ROI_HEADS_BATCH_SIZE  
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASSES)  
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(CONFIG_NAME)  # '/ai-data/chest/kjs2109/baseline/detectron2-baseline/weight_dir/Misc/Pseudo_test1_BORA_EVAL_1Findings_2gpus_512bs_8workers_MRCNN_1024_0.0001LR_WCLR_mask_rcnn_R_50_FPN_3x_gn/model_best_0.0000-84082.pth'

    cfg.SOLVER.IMS_PER_BATCH = 4 * NUM_GPU  # 1step에 학습하는 이미지 개수 (3377 / 4 = 844 -> GPU 1개 사용할 때 844 iter == 1 epoch)
    cfg.SOLVER.IMS_PER_BATCH_TEST = 4 * NUM_GPU 
    cfg.SOLVER.LR_SCHEDULER_NAME = SCHEDULER_NAME
    cfg.SOLVER.BASE_LR = LEARNING_RATE  # 0.0001   
    cfg.SOLVER.CHECKPOINT_PERIOD = 10000  # Save a checkpoint after every this number of iterations
    cfg.TEST.EVAL_PERIOD = 1000           # valid 주기 
    cfg.SOLVER.MAX_ITER = 300000          # 학습할 iteration 횟수 100000
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.WARMUP_METHOD = "linear"

    cfg.OUTPUT_DIR = WEIGHT_DIR # It should be modified when re-training

    cfg.freeze()
    return cfg 