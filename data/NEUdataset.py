import torch
import os
import json
import random
import numpy as np
import scipy.io as io
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
from data.Augument import *
import itertools

''' Device: a large pool of bit-similar(same hardware protocol physical address,MAC ID) devices(16?)
Signal: IQ sample at physical layer.
All transmitters are bit-similar USRP X310 radios that emit IEEE standards compliant frames generated via a MATLAB WLAN System
data frames generated contain random payload but have the same address fields, and are then streamed to the selected SDR for over-the air wireless transmission.
The receiver SDR samples the incoming singnals at 5 MS/s samplintg rate at a center frequency of 2.45GHz for WiFi.
'''
class NEUdataset():
    def __init__(self,data_dir,device,distance):
        '''this dataset is made by NEU
        author:NEU
        time:2018 
        '''
        self.F = data_dir
        self.D = device
        self.J = distance
        return 

    def dataIntorduce(self):
        ''' this dataset has len(Device_Id) device 
        total 11 distance(2ft~6ft~62ft)
        per sample per device per distance
        per sample introduce by follow
        '''
        # show the 3123D79 device signal in 2ft 
        signal = np.memmap(self.F.format(2,'3123D79',2),mode='r',dtype=np.complex128)
        print('shape of single sample:',signal.shape) #(20006400,)
        print('point data',signal[0]) # 0.02539
        signal = np.column_stack((signal.real, signal.imag)).T
        return

    def dataPlot(self):
        '''plot some pics about the datset
        '''
        device = Device_Id[0:self.D]
        device_n = self.D
        fts =  args.dis
        dis_n = len(fts)
        coms = itertools.product(device,self.J)
        nums = 1000
        for i,(device,ft) in enumerate(coms):
            plt.subplot(device_n,dis_n,i+1)
            signal = np.memmap(self.F.format(ft,device,ft),mode='r',dtype = np.complex128)
            plt.scatter(signal.real[0:nums],signal.imag[0:nums])
        plt.savefig('data/pic/try.png')

class IQRawset(Dataset):
    def __init__(self,data_params,split_params,transform='all_no',en_plus=1):
        root_dir = data_params['root_dir']
        Device_Id = data_params['Device_Id']
        data_format = data_params['data_format']
        runfile = split_params['run']
        dis_list = split_params['dis']
        margin = split_params['margin']
        signal_slices = split_params['range']
        self.slicelen = split_params['slicelen']
        self.Datas = []
        self.labels = []
        self.transform = split_params['transform']
        for dis in dis_list :
            for device in Device_Id:
                for run in runfile:
                    data_dir = data_format.format(device,dis,run)
                    data_dir = os.path.join(root_dir.format(dis),data_dir)
                    signal = np.memmap(data_dir, mode = 'r', dtype = np.complex128)
                    signal = split_params['transform']['total_tran'](signal)
                    len_signal = len(signal)
                    slices = [int(signal_slices[0]*len_signal) ,int(signal_slices[1]*len_signal)]
                    self.Datas.append(signal[slices[0]:slices[1]])
                    #TODO: 解决signal截取内存释放的问题del signal
                    label = {}
                    label['distance'] = dis 
                    label['device_id'] = device 
                    label['device_num'] = Device_Id.index(device)
                    self.labels.append(label)
    
        data_index = {}
        all_idx = []
        
        for i in tqdm(range(len(self.Datas))):
            length = len(self.Datas[i])
            all_idx.extend([(i,x) for x in range(length-margin+1) if x % margin == 0]) 

        self.index = all_idx
        self.transform = split_params['transform']['part_tran']
        
    def __len__(self):
        return len(self.index)

    def __getitem__(self,th):
        index = self.index[th]
        # self.Datas代表所有加载进来要截取的data，第一个索引是不同的整段信号，后一个索引是这段信号的切分.
        signal = self.Datas[index[0]][index[1]:index[1]+self.slicelen]
        # 此时的signal应该是slicelen长度的复数信号,经过下面的IQ拆分和stack就会变成slicelen×2的数组
        signal = np.column_stack((signal.real, signal.imag)).T
        signal=self.transform(signal)
        label = self.labels[index[0]]

        return signal,label['device_num'],label['distance']

class IQ3Dset(Dataset):

    def __init__(self,data_params,split_params,transform='all_no',en_plus=1):
        root_dir = data_params['root_dir']
        Device_Id = data_params['Device_Id']
        data_format = data_params['data_format']
        runfile = split_params['run']
        dis_list = split_params['dis']
        margin = split_params['margin']
        signal_slices = split_params['range']
        self.Datas = []
        self.labels = []
        self.transform = split_params['transform']
        for dis in dis_list :
            for device in Device_Id:
                for run in runfile:
                    data_dir = data_format.format(device,dis,run)
                    data_dir = os.path.join(root_dir.format(dis),data_dir)
                    signal = np.memmap(data_dir, mode = 'r', dtype = np.complex128)
                    signal = split_params['transform']['total_tran'](signal)
                    len_signal = len(signal)
                    slices = [int(signal_slices[0]*len_signal) ,int(signal_slices[1]*len_signal)]
                    self.Datas.append(signal[slices[0]:slices[1]])
                    #TODO: 解决signal截取内存释放的问题del signal
                    label = {}
                    label['distance'] = dis 
                    label['device_id'] = device 
                    label['device_num'] = Device_Id.index(device)
                    self.labels.append(label)
    
        data_index = {}
        all_idx = []
        
        for i in tqdm(range(len(self.Datas))):
            length = len(self.Datas[i])
            all_idx.extend([(i,x) for x in range(length-margin+1) if x % margin == 0]) 

        self.index = all_idx
        self.transform = split_params['transform']['part_tran']
        
        return 
    
    def __len__(self):

        return 

    def __getitem__(self,i):

        return

if __name__=='__main__':
    import argparse
    
    args = argparse.ArgumentParser()   
    
    args.add_argument("--device",
                type=int, 
                #nargs='+',
                default=2
                )
    args.add_argument("--dis",
                type=int, 
                nargs='+',
                default=[2]
                )
    args = args.parse_args()

    root_dir = "/home/syy/Data/RFFdataset/neu_m044q5210/KRI-16Devices-RawData/{}ft/"
    data_format = "WiFi_air_X310_{}_{}ft_run1.sigmf-data"
    file_dir = root_dir + data_format
    datashow = NEUdataset(file_dir, args.device, args.dis)
    datashow.dataIntorduce()
    datashow.dataPlot()

    margin = 128
    rf_list = ["2ft"]

    
    All_data = IQRawset(root_dir,data_format,rf_list,margin)

    #for data,label in dataloader:
    #    print(data,label)
    All_size = len(All_data)

    #train_size = int(0.8 * All_size)
    #test_size = int(0.1 * All_size)
    #val_size = All_size - (train_size + test_size)
    #print("total:%d | train:%d | val:%d | test:%d"%(All_size,train_size, val_size, test_size))
    #trainset,val_set,test_set = torch.utils.data.random_split(All_data, [train_size,val_size,test_size])


    trainloader = DataLoader(All_data,batch_size = 1)

    import pdb
    for data, label,dis in trainloader:
        pdb.set_trace()
    
