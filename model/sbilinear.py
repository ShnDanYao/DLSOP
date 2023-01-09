import torch
import torch.nn as nn
import numpy as np
from torchvision.models import resnet18
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning.metrics import functional as FM

## 这里的resnet18作为Bilinerar CNN Model的骨架
class sBCNN(pl.LightningModule):
    def __init__(self,num_classes,lr):
        super(sBCNN,self).__init__()
        torch.set_default_tensor_type('torch.DoubleTensor')
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
        #self.features = nn.Sequential(resnet18().conv1,
        #                             resnet18().bn1,
        #                             resnet18().relu,
        #                             resnet18().maxpool,
        #                             resnet18().layer1,
        #                             resnet18().layer2,
        #                             resnet18().layer3,
        #                             resnet18().layer4)
        self.classifiers = nn.Sequential(nn.Linear(128**2,num_classes))
        self.lr = lr

    def forward(self,x):
        x=self.features(x)
        batch_size = x.size(0)
        feature_size = x.size(2)
        #x = x.view(batch_size , 512, feature_size)
        x = (torch.bmm(x, torch.transpose(x, 1, 2)) / feature_size).view(batch_size, -1)
        x = torch.nn.functional.normalize(torch.sign(x)*torch.sqrt(torch.abs(x)+1e-10))
        x = self.classifiers(x)
        return x

    def training_step(self,batch,batch_idx):
        # logger.no_extra()

        inputs,label,dis=batch
        # print(i,label[0],dis[0])
        out = self(inputs)
        loss = F.cross_entropy(out,label)
        # self.logger.experiment.whatever_ml_flow_supports()
        # self.logger[0].log_metrics({'train/loss':loss},self.global_step)
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
