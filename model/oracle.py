import torch
import numpy as np
import torch.nn as nn
import time
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning.metrics import functional as FM


class ORACLE(pl.LightningModule):
    def __init__(self,num_classes,lr):
        super(ORACLE,self).__init__()
        self.num_classes = num_classes
        self.lr = lr
        # TODO
        torch.set_default_tensor_type('torch.DoubleTensor')
        # 这里将IQ两路当做是两个通道进行输入
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=2,out_channels=128,kernel_size=7,padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=128,out_channels=128,kernel_size=5,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=128,out_channels=128,kernel_size=7,padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=128,out_channels=128,kernel_size=5,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=128,out_channels=128,kernel_size=7,padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=128,out_channels=128,kernel_size=5,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=128,out_channels=128,kernel_size=7,padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=128,out_channels=128,kernel_size=5,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2)
        )
        # self.avgpool = nn.AdapiveAvgPool1d((128*4))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128*4, out_features=256),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(inplace=True),

            nn.Linear(128, num_classes)
        )

    def forward(self,x):
        x = self.features(x)
        # x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x

    def training_step(self,batch,batch_idx):
        # logger.no_extra()

        inputs,label,dis=batch
        # print(i,label[0],dis[0])
        out = self(inputs)
        loss = F.cross_entropy(out,label)
        # self.logger.experiment.whatever_ml_flow_supports()
        #self.logger.log_metrics({'train/loss':loss},self.global_step)
        self.logger.experiment.log_metric(self.logger.run_id, 'train/loss', loss.item())
        return loss

    def validation_step(self,batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = [acc, loss]
        return metrics

    def validation_epoch_end(self, validation_step_outputs):
        val_acc,val_loss = np.mean(validation_step_outputs,axis=0)
        metrics = {'val/acc':val_acc,'val/loss':val_loss}
        #print('valacc: {:.2%},valloss: {:.2}'.format(val_acc,val_loss))
        #self.logger[0].log_metrics({'acc/val':val_acc,'loss/val':val_loss},self.current_epoch)
        for key,value in metrics.items():
            self.log(key,value,logger=True,prog_bar=True)#,on_epoch=True)
        self.log('val_acc',val_acc*100,logger=False,prog_bar=False)
        return 
        
    def test_step(self,batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = [acc, loss]
        return metrics

    # https://github.com/PyTorchLightning/pytorch-lightning/issues/1088
    def test_epoch_end(self, outputs):
        test_acc,test_loss = np.mean(outputs,axis=0)
        self.log('test/acc',test_acc,prog_bar=True,logger=False)
        self.log('test/loss',test_loss,prog_bar=True,logger=False)
        self.logger.experiment.log_metric(self.logger.run_id, 'test/acc', test_acc)
        self.logger.experiment.log_metric(self.logger.run_id, 'test/loss', test_loss)
        return {'test_acc': test_acc, 'avg_acc': test_loss}

    def _shared_eval_step(self, batch, batch_idx):
        inputs,label,dis=batch
        out = self(inputs)
        loss = F.cross_entropy(out, label)
        pred = torch.max(out, 1)[1]
        acc_num = (pred == label).sum()
        acc = float(acc_num)/len(label)
        return loss.item(), acc

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Oracle")
        parser.add_argument('--encoder_layers', type=int, default=12)
        parser.add_argument('--data_path', type=str, default='/some/path')
        return parent_parser

if __name__ == '__main__':
    # ----------------
    # trainer_main.py
    # ----------------
    from argparse import ArgumentParser
    parser = ArgumentParser()
    
    # add PROGRAM level args
    parser.add_argument('--conda_env', type=str, default='some_name')
    parser.add_argument('--notification_email', type=str, default='will@email.com')
    
    # add model specific args
    parser = ORACLE.add_model_specific_args(parser)
    
    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    #parser = Trainer.add_argparse_args(parser)
    
    args = parser.parse_args()
    net = ORACLE(10,1) 
    import pdb
    pdb.set_trace()
