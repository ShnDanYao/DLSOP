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


import mlflow.pytorch
from mlflow.tracking import MlflowClient
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

#from torch.utils.data import Dataset,DataLoader
#from line_profiler import LineProfiler
#from tensorboardX import SummaryWriter
def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print("run_id: {}".format(r.info.run_id))
    print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    print("tags: {}".format(tags))

def get_logger(C_exp):
    C_M = C_exp['mlflow']
    C_T = C_exp['tensor']
    mlf_logger = MLFlowLogger(experiment_name=C_M['name'], tracking_uri="file:./exps/mlruns",tags=C_M['tag'])
    ten_logger = TensorBoardLogger("exps/tensorboard", name=C_T['name'],prefix=C_T['prefix'],version=C_T['version'])
    return ten_logger, mlf_logger

if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='RF System')
    parser.add_argument("--config",default='exps/test.yaml',type=str) 
    parser.add_argument("--gpu",default=0,type=int)
    args = parser.parse_args()
    C = Config(args.config)

    trainer = Trainer(resume_from_checkpoint="some/path/to/my_checkpoint.ckpt")

    # automatically restores model, epoch, step, LR schedulers, apex, etc...
    trainer.fit(model)
    model = MyLightingModule.load_from_checkpoint(PATH)

    checkpoint_callback = ModelCheckpoint(monitor="val/acc",mode="max",filename="sample-mnist-{epoch:02d}-{val_acc:.2f}", save_top_k=3,)
    ten_logger, mlf_logger = get_logger(C['experiment'])
    mlf_logger.log_hyperparams(C)
    #ten_logger.log_hyperparams(C,C['trainer'])
    model = set_module(C['model']['name'],C['model']['params'])
    data = RFData(C['data']['name'],C['data']['params'])
    # Auto log all MLflow entities
    # mlflow.pytorch.autolog('exps/mlruns/')

    # Train the model
    trainer = Trainer(gpus=1,logger=[ten_logger, mlf_logger],**C['trainer'],callbacks=[checkpoint_callback]) #
    #with mlflow.start_run(run_name='run_test') as run:#  use when mlflow autolog
    trainer.fit(model,data)
    trainer.test(model,datamodule=data)

    # fetch the auto logged parameters and metrics
    # print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
    
