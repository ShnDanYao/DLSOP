#from torchstat import stat
#from model import sBCNN
#
## 导入模型，输入一张输入图片的尺寸
#model = sBCNN(num_classes=16,lr=0.001)
#stat(model, (2, 1024))


from thop import clever_format
from thop import profile
from model import sBCNN
import torch

#class YourModule(nn.Module):
#    # your definition
#def count_your_model(model, x, y):
#    # your rule here


for lens in [128,256,512,1280]:
    input = torch.DoubleTensor(1, 2, lens)
    model = sBCNN(num_classes=16,lr=0.001)
    flops, params = profile(model, inputs=(input, ),)
                            #custom_ops={YourModule: count_your_model})
    flops, params = clever_format([flops, params], "%.3f")
    print(lens,":", flops, params)
