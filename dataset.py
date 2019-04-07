import os
import torch
import numpy as np
import cv2
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms as tfs
from datetime import datetime
from numpy import *
from config import *




#read images and labels t list
def read_images(root=voc_root,train=True):
    txt_fname=root+ '/ImageSets/Segmentation/' + ('train.txt' if train else 'val.txt')
    with open(txt_fname,'r') as f:
        images=f.read().split()
    data=[os.path.join(root,'JPEGImages',i+'.jpg')for i in images]
    label=[os.path.join(root,'SegmentationClass',i+'.png')for i in images]
    return data ,label


'''
this is meaning that the size of many images are not the same ,
so we should crop the image to get the same size of it
'''
def random_crop(data, label, crop_size):


    height, width = crop_size
    data, rect = tfs.RandomCrop((height, width))(data)
    label = tfs.FixedCrop(*rect)(label)
    return data, label


# len(classes), len(colormap)

cm2lbl = np.zeros(256**3) # 每个像素点有 0 ~ 255 的选择，RGB 三个通道
for i,cm in enumerate(colormap):
    cm2lbl[(cm[0]*256+cm[1])*256+cm[2]] = i # 建立索引

def image2label(img):
    data = np.array(img, dtype='int32')
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    #print(idx)
    return np.array(cm2lbl[idx], dtype='int64') # 根据索引得到 label 矩阵



#pre  deal images
def img_transforms(img, label, crop_size):

    img_tfs = tfs.Compose([tfs.CenterCrop(crop_size),
        tfs.ToTensor(),
        tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])#the tensor is belong 0-1
    ])

    img = img_tfs(img)
    label=tfs.CenterCrop(crop_size)(label)
    #label = img_tfs(label)
    label = image2label(label)
    label = torch.from_numpy(label)
    return img, label


#redifine our datasets
class VOCSegDataset(Dataset):
    '''
    voc dataset
    '''

    def __init__(self, train, crop_size, transforms):
        self.crop_size = crop_size
        self.transforms = transforms
        data_list, label_list = read_images(train=train)
        self.data_list = self._filter(data_list)
        self.label_list = self._filter(label_list)
        print('Read ' + str(len(self.data_list)) + ' images')

    def _filter(self, images):  # 过滤掉图片大小小于 crop 大小的图片
        return [im for im in images if (Image.open(im).size[1] >= self.crop_size[0] and
                                        Image.open(im).size[0] >= self.crop_size[1])]

    def __getitem__(self, idx):
        img = self.data_list[idx]
        label = self.label_list[idx]
        img = Image.open(img)
        label = Image.open(label).convert('RGB')
        img, label = self.transforms(img, label, self.crop_size)
        #img=torch.Tensor(img)
        #label=torch.Tensor(label)
        return img, label

    def __len__(self):
        return len(self.data_list)