import os 
import logging 
import wandb 
from detectron2.engine.train_loop import HookBase
from detectron2.utils.events import EventWriter, get_event_storage 

import detectron2.utils.comm as comm 
from fvcore.nn.precise_bn import get_bn_modules 

from config import CLASSES, WEIGHT_DIR


def check_and_mkdir(target_path):
    print("Target_path: " + str(target_path))
    path_to_targets = os.path.split(target_path)
    print("path_to_targets: " + str(path_to_targets))

    path_history = '/'
    for path in path_to_targets:
        path_history = os.path.join(path_history, path)
        if not os.path.exists(path_history):
            os.makedirs(path_history)  


class BestCheckpointer(HookBase):
    def before_train(self):
        self.best_metric = 10.0
        self.logger = logging.getLogger("detectron2.trainer")
        self.logger.info("####################### Running best check pointer ############################")

    def after_step(self):
        if "total_loss" in self.trainer.storage._history:
            eval_metric, batches = self.trainer.storage.history("total_loss")._data[-1] 
            iteration = self.trainer.iter 
            if self.best_metric > eval_metric:
                for weight_fname in os.listdir(WEIGHT_DIR): 
                    if weight_fname.startswith("model_best"):
                        os.remove(os.path.join(WEIGHT_DIR, weight_fname))
                self.best_metric = eval_metric
                self.logger.info(f"####################### New best metric: {self.best_metric} ############################")
                self.trainer.checkpointer.save(f"model_best_{eval_metric:.4f}-{iteration}") 


class WandB_Printer(EventWriter):
    def __init__(self, name, project, entity) -> None:
        self._window_size=20
        self._matchDictList = self._makeMatchDictList()

        wandb.login(key="{{'wandb key'}}")

        self.wandb = wandb.init(project=project,entity=entity,name=name)

    def write(self):
        storage = get_event_storage()

        sendDict = self.makeSendDict(storage)
        self.wandb.log(sendDict)


    def makeSendDict(self, storage):
        sendDict = {}

        storageDict = self._makeStorageDict(storage)

        for item in self._matchDictList:
            sendDict = self._findValue(storageDict,item["key"],sendDict,item["customKey"],item["subKey"])

        return sendDict 

    def _makeStorageDict(self,storage):
        storageDict = {}
        for k,v in [(k, f"{v.median(self._window_size):.4g}") for k, v in storage.histories().items()]:
            if "AP" in k:
            # AP to mAP
                storageDict[k] = float(v) * 0.01
            else:
                storageDict[k] = float(v)

        return storageDict

    def _findValue(self,storageDict,key, retDict, customKey, subKey=None):
        if key in storageDict:
            if subKey is None:
                retDict[customKey] = storageDict[key]
            else:
                retDict["/".join([subKey,customKey])] = storageDict[key]

        return retDict

    def _makeMatchDictList(self):
        matchDictList = []

        matchDictList.append(self._makeMatchDict(key="lr",customKey="detectron_learning_rate",subKey='lr'))

        matchDictList.append(self._makeMatchDict(key="total_loss",customKey="loss",subKey="train"))
        matchDictList.append(self._makeMatchDict(key="loss_box_reg",customKey="loss_box_reg",subKey="train")) 
        matchDictList.append(self._makeMatchDict(key="loss_cls",customKey="loss_cls",subKey="train"))
        matchDictList.append(self._makeMatchDict(key="loss_mask",customKey="loss_mask",subKey="train"))
        matchDictList.append(self._makeMatchDict(key="mask_rcnn/accuracy",customKey="mask_rcnn-accuracy",subKey="train"))

        matchDictList.append(self._makeMatchDict(key="bbox/AP",customKey="bbox_mAP",subKey="val"))
        matchDictList.append(self._makeMatchDict(key="bbox/AP50",customKey="bbox_mAP_50",subKey="val"))
        matchDictList.append(self._makeMatchDict(key="bbox/AP75",customKey="bbox_mAP_75",subKey="val"))
        matchDictList.append(self._makeMatchDict(key="segm/AP",customKey="segm_mAP",subKey="val"))
        matchDictList.append(self._makeMatchDict(key="segm/AP50",customKey="segm_mAP50",subKey="val"))
        matchDictList.append(self._makeMatchDict(key="segm/AP75",customKey="segm_mAP75",subKey="val"))
        # matchDictList.append(self._makeMatchDict(key="bbox/APl",customKey="bbox_mAP_l",subKey="val"))
        # matchDictList.append(self._makeMatchDict(key="bbox/APm",customKey="bbox_mAP_m",subKey="val"))
        # matchDictList.append(self._makeMatchDict(key="bbox/APs",customKey="bbox_mAP_s",subKey="val"))

        matchDictList.append(self._makeMatchDict(key="eta_seconds",customKey="eta_seconds",subKey="temp"))
        matchDictList.append(self._makeMatchDict(key="time",customKey="time",subKey="temp"))
        matchDictList.append(self._makeMatchDict(key="data_time",customKey="data_time",subKey="temp"))

        for i in range(len(CLASSES)):
            matchDictList.append(self._makeMatchDict(key=f"bbox/AP-{CLASSES[i]}",customKey=f"mAP_{CLASSES[i]}",subKey="val/class/bbox"))
            matchDictList.append(self._makeMatchDict(key=f"segm/AP-{CLASSES[i]}",customKey=f"mAP_{CLASSES[i]}",subKey="val/class/segm"))

        return matchDictList

    def _makeMatchDict(self,key,customKey,subKey):
        return {"key":key, "customKey":customKey, "subKey":subKey}

