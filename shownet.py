
import torchvision.models as models

import torch

net=models.densenet161(pretrained=True)
print(net.features)
for sqq in list(net.children()):

    # print(sqq)
    print('******************************************')