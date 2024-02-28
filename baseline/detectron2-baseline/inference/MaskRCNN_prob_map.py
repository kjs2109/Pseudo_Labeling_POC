import os, sys, cv2, time, logging, torch, gc
from pathlib import Path
sys.path.append(str(Path.cwd()))
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from os.path import join

from detectron2.utils.logger import setup_logger
setup_logger()
logger = logging.getLogger("detectron2")
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer

from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T

# from models.GradCAM.gradcam import GradCAM, GradCamPlusPlus
# from models.GradCAM.detectron2_gradcam import Detectron2GradCAM

# class MaskRCNNProbPredictor(DefaultPredictor):
#     def __init__(self, cfg):
#         super().__init__(cfg)

#     def __call__(self, original_image):
#         with torch.no_grad():
#             # Run object detection and get the predictions
#             outputs = self.model(original_image)
#             predictions = outputs["instances"].to(self.cpu_device)

#             # Convert the mask logits to probabilities
#             mask_probs = predictions.pred_masks.sigmoid()

#             # Resize the mask probabilities to the original image size
#             mask_probs = mask_probs[:, None].float()
#             mask_probs = self.model.roi_heads.mask_head.mask_fcn2conv(mask_probs)
#             mask_probs = self.model.roi_heads.mask_head.conv5_mask(mask_probs)
#             mask_probs = self.model.roi_heads.mask_head.deconv_mask(mask_probs)
#             mask_probs = mask_probs[:, :, :original_image.shape[0], :original_image.shape[1]]
#             mask_probs = mask_probs.squeeze(1).sigmoid()

#             # Update the predictions with the mask probabilities
#             predictions.pred_masks = mask_probs

#             return predictions


class CustomPredictor:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.
    Compared to using the model directly, this class does the following additions:
    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take one input image and produce a single output, instead of a batch.
    This is meant for simple demo purposes, so it does the above steps automatically.
    This is not meant for benchmarks or running complicated inference logic.
    If you'd like to do anything more fancy, please refer to its source code as examples
    to build and use the model manually.
    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.
    Examples:
    ::
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).
        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]


            
            return predictions

class MaskRCNN_NDL(object):
    def __init__(self, CONFIG_DICT, weight_path, device):
        
        # self.image_path = image_path
        self.weight_path = weight_path
        
        # detectron2 config
        self.model_zoo_config = CONFIG_DICT['model_zoo_config']
        # self.input_mask_format = CONFIG_DICT['input_mask_format']
        self.input_min_size_test = CONFIG_DICT['input_min_size_test']
        self.input_max_size_test = CONFIG_DICT['input_max_size_test']
        self.model_roi_heads_batch = CONFIG_DICT['model_roi_heads_batch']
        self.model_roi_heads_num_classes = len(CONFIG_DICT['CLASSES'])
        self.model_roi_heads_threshold = CONFIG_DICT['model_roi_heads_threshold']
        self.CLASSES = CONFIG_DICT['CLASSES']
        self.device = device
        self.cfg = self.setup()
        print(self.weight_path)
        self.MaskRCNN_model = CustomPredictor(self.cfg)
        # self.MaskRCNN_model = MaskRCNNProbPredictor(self.cfg)
        print(self.MaskRCNN_model)

    def setup(self):
        """ Function for setting Detectron2 config

            Args:
                None

            Returns:
                cfg : Detectron2 config for the prediction tasks
        """
        cfg = get_cfg()

        cfg.merge_from_file(model_zoo.get_config_file(self.model_zoo_config))
        # cfg.INPUT.MASK_FORMAT = self.input_mask_format
        cfg.INPUT.MIN_SIZE_TEST = self.input_min_size_test
        cfg.INPUT.MAX_SIZE_TEST = self.input_max_size_test
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = self.model_roi_heads_batch
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.CLASSES)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.model_roi_heads_threshold
        cfg.MODEL.WEIGHTS = self.weight_path
        cfg.MODEL.DEVICE = self.device
        cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA = 0.5
        cfg.MODEL.ROI_BOX_HEAD.POSTPROCESS_MASKS = False
        cfg.MODEL.ROI_BOX_HEAD.RESOLUTION = 28
        cfg.MODEL.RPN.SMOOTH_L1_BETA = 0.5
        
        cfg.freeze()
        return cfg

    def get_dict(self, image_path, img):
        """ Function for getting data dictionary included file name, image id, height and width

            Args:
                None

            Returns:
                dataset_dicts : image data dictionary
        """
        dataset_dict = dict()
        height, width = img.shape[:2]
        
        dataset_dict["file_name"] = image_path
        dataset_dict["image_id"] = str(image_path.split('.')[0].split('/')[-1])
        dataset_dict["height"] = height
        dataset_dict["width"] = width
        return dataset_dict


    def get_seg_mask_all(self, img):
        """ Function for getting binary segmentation mask array based on model's prediction

            Args:
                cfg : Detectron2 config
                dataset_dict : image data dictionary
                
            Returns:
                seg_mask(numpy.array) : binary segmentation mask image
        """


        outputs = self.MaskRCNN_model(img)
        self.seg_mask = outputs["instances"].pred_masks.cpu().numpy()
        print("seg_mask dtype: " + str(self.seg_mask.dtype))
        print('seg_mask_buffer',self.seg_mask.shape)
        self.seg_mask = self.seg_mask.sum(axis=0)*255
        return self.seg_mask

    def get_seg_mask(self, img, cls_name):
        """ Function for getting binary segmentation mask array based on model's prediction

            Args:
                cfg : Detectron2 config
                dataset_dict : image data dictionary
                
            Returns:
                seg_mask(numpy.array) : binary segmentation mask image
        """
        outputs = self.MaskRCNN_model(img)
        instances = outputs["instances"]
        pred_all_seg_mask = instances.pred_masks.cpu().numpy()
        pred_classes = list(instances.pred_classes.cpu().numpy())

        if self.CLASSES.index(cls_name) in pred_classes:
            corres_idx = pred_classes.index(self.CLASSES.index(cls_name))
            corres_idx = []
            for i, cls in enumerate(pred_classes):
                if cls == self.CLASSES.index(cls_name):
                    corres_idx.append(i)

            seg_mask = np.zeros((len(corres_idx), img.shape[0], img.shape[1]))
            for i, idx in enumerate(corres_idx):
                seg_mask[i] = pred_all_seg_mask[idx]

            seg_mask = seg_mask.sum(axis=0)*255
        else:
            seg_mask = np.zeros((img.shape[0], img.shape[1]))

        return seg_mask

    def get_seg_mask_and_scores(self, img, cls_name):
        """ Function for getting binary segmentation mask array based on model's prediction

            Args:
                cfg : Detectron2 config
                dataset_dict : image data dictionary
                
            Returns:
                seg_mask(numpy.array) : binary segmentation mask image
        """
        outputs = self.MaskRCNN_model(img)
        instances = outputs["instances"]
        # print(f"instances: {instances}")
        pred_all_seg_mask = instances.pred_masks.cpu().numpy()
        # print("pred_all_seg_mask type: " + str(type(pred_all_seg_mask)))
        # print("pred_all_seg_mask shape: " + str(pred_all_seg_mask.shape))
        # print("pred_all_seg_mask[0] shape: " + str(pred_all_seg_mask[0].shape))
        # print("pred_all_seg_mask[0] dtype: " + str(pred_all_seg_mask[0].dtype))
        # print("pred_all_seg_mask[0] unique: " + str(np.unique(pred_all_seg_mask[0])))
        pred_scores = list(instances.scores.cpu().numpy())

        pred_classes = list(instances.pred_classes.cpu().numpy())
        print(f"pred_classes: {pred_classes}")

        max_pred_class_score = 0
        if self.CLASSES.index(cls_name) in pred_classes:
            corres_idx = pred_classes.index(self.CLASSES.index(cls_name))
            corres_idx = []
            for i, cls in enumerate(pred_classes):
                if cls == self.CLASSES.index(cls_name):
                    corres_idx.append(i)

            seg_mask = np.zeros((len(corres_idx), img.shape[0], img.shape[1]))
            for i, idx in enumerate(corres_idx):
                seg_mask[i] = pred_all_seg_mask[idx]
                if pred_scores[idx] > max_pred_class_score:
                    max_pred_class_score = pred_scores[idx]

            seg_mask = seg_mask.sum(axis=0)
            print(f"np unique in maskrcnn prob map: {np.unique(seg_mask)}")
            seg_mask[seg_mask>0] = 255
        else:
            seg_mask = np.zeros((img.shape[0], img.shape[1]))

        return seg_mask, max_pred_class_score

    def get_seg_mask_and_scores_filter_thresh(self, img, cls_name, threshold):
        """ Function for getting binary segmentation mask array based on model's prediction

            Args:
                cfg : Detectron2 config
                dataset_dict : image data dictionary
                
            Returns:
                seg_mask(numpy.array) : binary segmentation mask image
        """
        outputs = self.MaskRCNN_model(img)
        instances = outputs["instances"]
        pred_all_seg_mask = instances.pred_masks.cpu().numpy()
        # print("pred_all_seg_mask type: " + str(type(pred_all_seg_mask)))
        # print("pred_all_seg_mask shape: " + str(pred_all_seg_mask.shape))
        # print("pred_all_seg_mask[0] shape: " + str(pred_all_seg_mask[0].shape))
        # print("pred_all_seg_mask[0] dtype: " + str(pred_all_seg_mask[0].dtype))
        # print("pred_all_seg_mask[0] unique: " + str(np.unique(pred_all_seg_mask[0])))
        pred_scores = list(instances.scores.cpu().numpy())
        pred_classes = list(instances.pred_classes.cpu().numpy())

        max_pred_class_score = 0
        if self.CLASSES.index(cls_name) in pred_classes:
            corres_idx = pred_classes.index(self.CLASSES.index(cls_name))
            corres_idx = []
            for i, cls in enumerate(pred_classes):
                if cls == self.CLASSES.index(cls_name):
                    corres_idx.append(i)

            seg_mask = np.zeros((len(corres_idx), img.shape[0], img.shape[1]))
            for i, idx in enumerate(corres_idx):
                if pred_scores[idx] > threshold:
                    seg_mask[i] = pred_all_seg_mask[idx]
                    if pred_scores[idx] > max_pred_class_score:
                        max_pred_class_score = pred_scores[idx]

            seg_mask = seg_mask.sum(axis=0)*255
        else:
            seg_mask = np.zeros((img.shape[0], img.shape[1]))

        return seg_mask, max_pred_class_score

    def get_raw_outputs(self, img):
        return self.MaskRCNN_model(img)

    def get_pred_classes(self, img):
        """Function for getting predicted classes

            Args:
                cfg : Detectron2 config
                dataset_dict : image data dictionary

            Returns:

                pred_classes(list) : strings of the predicted classes
        """

        outputs = self.MaskRCNN_model(img)
        pred_classes_buffer = outputs["instances"].pred_classes.cpu().numpy()
        pred_classes = []
        for NUM_class in pred_classes_buffer:
            pred_classes.append(self.CLASSES[NUM_class])
        return pred_classes

    def get_pred_scores(self, img):
        """Function for getting scores of the predicted classes

            Args:
                cfg : Detectron2 config
                dataset_dict : image data dictionary

            Returns:
                pred_classes(numpy.array) : scores of the predicted classes
        """
        outputs = self.MaskRCNN_model(img)
        pred_scores = outputs["instances"].scores.cpu().numpy()
        
        return pred_scores

    def visualize_pred(self, image_path, img):
        """ Visualizing the predicted results

            Args:
                cfg : Detectron2 config
                dataset_dict : image data dictionary

            Returns:
                vis_arr(numpy.array) : the array that prediction result overlaid to the original image
        """
        outputs = self.MaskRCNN_model(img)

        TIME = time.localtime()
        tmp_dataset_name = "%04d/%02d/%02d_%02d:%02d:%02d" %(TIME.tm_year, TIME.tm_mon, TIME.tm_mday,
                                                             TIME.tm_hour, TIME.tm_min, TIME.tm_sec)

        image_dict = self.get_dict(image_path, img)

        for d in [tmp_dataset_name]:
            DatasetCatalog.register(d, lambda d:image_dict)
        vis_buffer = Visualizer(img[:, :, ::-1],
                        MetadataCatalog.get(tmp_dataset_name).set(thing_classes=self.CLASSES),
                        scale=0.5)
        vis = vis_buffer.draw_instance_predictions(outputs["instances"].to("cpu"))
        vis_img = vis.get_image()[:, :, ::-1]
        vis_arr = np.array(vis_img)
        return vis_arr
    
    # def get_heatmap(self, image_path, image):
    #     cam_extractor = Detectron2Grad
        
    #     # outputs = self.MaskRCNN_model(image)
    #     # masks = outputs["instances"].pred_masks.cpu().numpy()
    #     # heatmap = np.sum(masks, axis=0)


    #     # # Normalize the heatmap
    #     # heatmap = heatmap / np.max(heatmap)

    #     # print("-"*30)
    #     # print(heatmap)
    #     # print(type(heatmap))
    #     # print(heatmap.shape)
    #     # print(heatmap.dtype)
    #     # print(np.unique(heatmap))
    #     # print("-"*30)

    #     # v = Visualizer(image[:, :, ::-1], metadata=MetadataCatalog.get(image_path).set(thing_classes=self.CLASSES), scale=1)
    #     # v = v.draw_heatmap(heatmap)
    #     # heatmap_image = v.get_image()[:, :, ::-1]



    #     # v = Visualizer(image[:, :, ::-1], metadata=MetadataCatalog.get(image_path).set(thing_classes=self.CLASSES), scale=0.5)
    #     # # v = Visualizer(image[:, :, ::-1], scale=0.5)
    #     # heatmap = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    #     # print("-"*30)
    #     # print(heatmap)
    #     # print(type(heatmap))
    #     # print(heatmap)
    #     # print("-"*30)
    #     # heatmap_image = heatmap.get_image()[:, :, ::-1]
    #     # print("heatmap_image shape: "  + str(heatmap_image.shape))
    #     # print("heatmap_image dtype: "  + str(heatmap_image.dtype))
    #     # print("heatmap_image np unique: "  + str(np.unique(heatmap_image)))



    #     # print("-"*30)
    #     # print(outputs)
    #     # print(type(outputs))
    #     # print(outputs.keys())
    #     # print(outputs['instances'])
    #     # print("-"*30)

    #     # # pred_scores = outputs["instances"].scores.cpu().numpy()
        
    #     return heatmap_image
    

def check_and_mkdir(target_path):
    print("Target_path: " + str(target_path))
    path_to_targets = os.path.split(target_path)
    print("path_to_targets: " + str(path_to_targets))

    path_history = '/'
    for path in path_to_targets:
        path_history = os.path.join(path_history, path)
        if not os.path.exists(path_history):
            os.mkdir(path_history)
    
    
if __name__=='__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    TARGET_CLASS = ['Consolidation', 'Pneumothorax', 'Fibrosis', 'Pleural effusion', 'Nodule/Mass', 'Fibrothorax']

    SEG_CONFIG_DICT = {
        'model_zoo_config': 'Misc/mask_rcnn_R_50_FPN_3x_gn.yaml',
        'input_min_size_test': 800,
        'input_max_size_test': 1333,
        'model_roi_heads_batch': 512,
        'model_roi_heads_threshold': 0.7,
        'CLASSES': TARGET_CLASS,
        'device': device,
    }
    train_tag = 'MRCNN_BORA_EVAL_5Findings_8gpus_128bs_8workers_MRCNN_2048_0.0001LR_WMSLR_polygon_BORA_XML_ADDED_v4'
    weight_file_name = 'model_best_0.4885.pth'


    # seg_weight_path = f'/aidata/chest/hyoon/weight_dir/nodule_mass_only_mask_rcnn/{train_tag}/model_final.pth'
    seg_weight_path = f'/aidata/chest/hyoon/weight_dir/chest_findings/{train_tag}/{weight_file_name}'
    seg_model = MaskRCNN_NDL(SEG_CONFIG_DICT, seg_weight_path, device)
    print(seg_model.MaskRCNN_model.model)
    print(seg_model.MaskRCNN_model.cfg)
    print(dir(seg_model.MaskRCNN_model))

    test_image_base_dir = '/aidata/chest/DATA/PrivateDataset/chest{}/image/'
    institution_list = ['CB']

    intference_base_dir_path = f'/aidata/chest/hyoon/result_inference_test/6findings/{train_tag}/'
    pred_mask_dir_path = join(intference_base_dir_path, 'pred_masks')
    pred_overlay_dir_path = join(intference_base_dir_path, 'pred_overlays')
    check_and_mkdir(intference_base_dir_path)
    check_and_mkdir(pred_mask_dir_path)
    check_and_mkdir(pred_overlay_dir_path)

    issue_files = []
    # test_image_dir = '/aidata/chest/DATA/PrivateDataset/chestPAIK/image/'
    for institution in institution_list:
        test_image_dir = test_image_base_dir.format(institution)
        test_image_names = os.listdir(test_image_dir)
        test_image_names.sort()

        for i, tin in enumerate(test_image_names):
            try:
                if os.path.exists(f'{pred_mask_dir_path}/{tin}.png'):
                    continue
                print(f"{i+1}/{len(test_image_names)}: {tin}")
                test_image_pth = join(test_image_dir, tin)
                image = cv2.imread(test_image_pth)

                seg_mask = seg_model.get_seg_mask(image)
                print("seg_mask shape: " +str(seg_mask.shape))
                # print("seg_mask np.unique min and mask: <MIN: " +str())
                # cv2.imwrite('', seg_mask)
                # cv2.imwrite(f'{pred_mask_dir_path}/{tin}.png', seg_mask)

                # overlay_image = seg_model.visualize_pred(test_image_pth, image)
                # cv2.imwrite(f'{pred_overlay_dir_path}/{tin}.png', overlay_image)

                gc.collect()
            except Exception as e:
                print(e)
                issue_files.append((tin, str(e)))
            gc.collect()



    for j, (fn, err) in enumerate(issue_files):
        print(f"{j}: {fn} : {err}")
        gc.collect()
