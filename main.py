import os
import numpy as np
import time
import json
import scipy.io as io
import torch
from utils import *
import argparse
import torch.nn as nn
import tqdm
from pytorch_lightning.trainer.trainer import Trainer
from data import RFData
from model import set_module
import random

import mlflow.pytorch
from mlflow.tracking import MlflowClient
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only
class MLFlowModelCheckpoint(ModelCheckpoint):
    def __init__(self, mlflow_logger, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mlflow_logger = mlflow_logger

    @rank_zero_only
    def on_test_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)
        run_id = self.mlflow_logger.run_id
        print(run_id)
        self.mlflow_logger.experiment.log_dict(run_id, {'checkpoint_path':self.best_model_path},artifact_file="checkpoint.txt")

def get_logger(C_exp):
    C_M = C_exp['mlflow']
    mlf_logger = MLFlowLogger(
            experiment_name=C_M['exp_name'], 
            run_name =C_M['run_name'] ,
            artifact_location=os.getenv("MLFLOW_ARTIFACTS_URI"),
            tracking_uri=os.getenv('MLFLOW_TRACKING_URI'))
            #tags=C_M['tag'])
    return  mlf_logger

if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='RF System')
    parser.add_argument("--config",default='exps/config/oracle.yaml',type=str) 
    parser.add_argument("--yaml_change",default=[],type=str,nargs='*') 
    parser.add_argument("--gpu",default=-1,type=int)
    args = parser.parse_args()
    C = Config(args.config,args.yaml_change)

    mlf_logger = get_logger(C) #ten_logger,
    mlf_logger.log_hyperparams(C)

    seed = C["seed"]#024
    random.seed(seed)     # python的随机性
    np.random.seed(seed)  # np的随机性
    os.environ['PYTHONHASHSEED'] = str(seed)

    model = set_module(C['model']['name'],C['model']['params'])
    data = RFData(C['data']['name'],C['data']['params'])
    # if you want to Auto log all MLflow entities
    # mlflow.pytorch.autolog('exps/mlruns/')
    # Train the model,if you want to use multi-logger,change the logger params as logger=[ten_logger,mlf_logger]
    checkpoint_callback1 = ModelCheckpoint(monitor="val/acc",mode="max", filename=str(mlf_logger.run_id)+"-{epoch:02d}-{val_acc:.2f}", save_top_k=3)
    checkpoint_callback2 = MLFlowModelCheckpoint(mlf_logger)
    trainer = Trainer(gpus=args.gpu,accelerator='ddp',logger= mlf_logger,**C['trainer'],callbacks=[checkpoint_callback1,checkpoint_callback2]) #
    trainer.fit(model,data)
    trainer.test(model,datamodule=data)

    # fetch the auto logged parameters and metrics
    # print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
