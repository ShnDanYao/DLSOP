import numpy as np

def normalization(x):
    """"
    归一化到区间{0,1]
    返回副本
    """
    _range = np.max(x) - np.min(x)
    return (x - np.min(x)) / _range

def standardization(x):
    """"
     将输入x 正态标准化  (x - mu) / sigma   ~   N(0,1)
     返回副本
    """
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    return (x - mu) / sigma


class enlarger():
    def __init__(self,plus):
        self.plus = plus

    def enlarge(self,x):
    
        return self.plus*x

def martain(x):
    return x
