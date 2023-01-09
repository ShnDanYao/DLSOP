import numpy as np
import time
import scipy.io as io
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
import tqdm
from . import NEUdataset
import pytorch_lightning as pl
from data.Augument import *

class RFData(pl.LightningDataModule):
    def __init__(self, data_name, params):
        super().__init__()
        self.args = params['dataloader']
        if self.args['type']=='train_test':
            params['train_test']['transform']['total_tran']=Transform(params['train']['transform']['total_tran'])
            params['train_test']['transform']['part_tran']=Transform(params['train']['transform']['part_tran'])
            traintest = eval(data_name)(params['dataset'],params['train_test'])
            val_size = int(len(traintest)*self.args['ratio'][1])
            test_size = int(len(traintest)*self.args['ratio'][2])
            train_size = len(traintest)-val_size-test_size
            self.trainset, self.valset,self.testset = torch.utils.data.random_split(traintest, [train_size,val_size,test_size])
        else:
            params['train']['transform']['total_tran']=Transform(params['train']['transform']['total_tran'])
            params['train']['transform']['part_tran']=Transform(params['train']['transform']['part_tran'])
            params['test']['transform']['total_tran']=Transform(params['test']['transform']['total_tran'])
            params['test']['transform']['part_tran']=Transform(params['test']['transform']['part_tran'])
            trainval = eval(data_name)(params['dataset'],params['train'])
            self.testset = eval(data_name)(params['dataset'], params['test'])
            val_size = int(len(trainval)*self.args['ratio'][1])
            train_size = len(trainval)-val_size
            self.trainset, self.valset = torch.utils.data.random_split(trainval, [train_size,val_size])
        self.batch_size = self.args['batch_size']

    def train_dataloader(self):
        train_loader = DataLoader(self.trainset,batch_size = self.batch_size, num_workers=6,shuffle=True)#, num_workers=4, drop_last=True)
        return train_loader #DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        val_loader = DataLoader(self.valset,batch_size = self.batch_size,num_workers=6)
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(self.testset,batch_size = self.batch_size,num_workers=6)
        return test_loader

    #def teardown(self, stage: Optional[str] = None):
    #    # Used to clean-up when the run is finished
    #    return

class Transform():
    def __init__(self, transforms_list):
        self.transforms_list = transforms_list
        return

    def __call__(self, x):
        for fp in self.transforms_list:
            f = fp[0]
            p = fp[1]
            x = eval(f)(x,**p)
        return x

if __name__=='__main__':


    dataset='ap'
    
    ## 训练设置
    #criterion = nn.CrossEntropyLoss()

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #num_class = 16
    #net = ORCLE(num_class)
    #net = net.double().to(device)

    #optim = torch.optim.Adam(net.parameters(),lr = args.lr)
    #scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size = args.lr_step, gamma = 0.1)

    #decrease = 0;
    #Epoch = args.epoch
    #last_acc = 0
    #for epoch in range(Epoch):
    #    net.train()
    #    loss = train(net,train_loader,device,logger)
    #    scheduler.step()
    #    val_acc = evaluate(net, val_loader,device)
    #    val_acc = float(val_acc)/len(val_set)*100
    #    writer.add_scalar('ans/loss',loss.item(),global_step=epoch)
    #    writer.add_scalar('ans/acc', val_acc,global_step=epoch)
    #    logger.now.info('Epoch: {:d}, Acc: {:.2f}%'.format(epoch, val_acc))
    #    # 结束判定条件
    #    if val_acc <= last_acc + 0.1:
    #        decrease+=1
    #    else:
    #        decrease = 0
    #    last_acc = val_acc
    #    if decrease == 10:
    #        break 
    #
    #test_acc = evaluate(net, test_loader,device)
    #logger.now.info("test_acc {:.2f}%".format(float(test_acc)/len(test_set)*100))

    #torch.save(net.state_dict(),'log/model/best.pth')
