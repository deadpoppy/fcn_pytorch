import torchvision.models as models

# net=models.resnet34(pretrained=False)
# for m in  list(net.children()):
#     print(m,'\n')
#     print('*********************************')
from dataset import *
from config import *
import torch
import numpy as np
import cv2
import matplotlib as plt
#from fcn import fcn
import torchvision.models as models
pretrained_net= models.resnet34(pretrained=False)
class fcn(nn.Module):
    def __init__(self, num_classes):
        super(fcn, self).__init__()

        self.stage1 = nn.Sequential(*list(pretrained_net.children())[:-4])  # 第一段
        self.stage2 = list(pretrained_net.children())[-4]  # 第二段
        self.stage3 = list(pretrained_net.children())[-3]  # 第三段

        self.scores1 = nn.Conv2d(512, num_classes, 1)
        self.scores2 = nn.Conv2d(256, num_classes, 1)
        self.scores3 = nn.Conv2d(128, num_classes, 1)

        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, 4, bias=False)
        #self.upsample_8x.weight.data = bilinear_kernel(num_classes, num_classes, 16)  # 使用双线性 kernel

        self.upsample_4x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
        #self.upsample_4x.weight.data = bilinear_kernel(num_classes, num_classes, 4)  # 使用双线性 kernel

        self.upsample_2x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
        #self.upsample_2x.weight.data = bilinear_kernel(num_classes, num_classes, 4)  # 使用双线性 kernel

    def forward(self, x):
        x = self.stage1(x)
        s1 = x  # 1/8

        x = self.stage2(x)
        s2 = x  # 1/16

        x = self.stage3(x)
        s3 = x  # 1/32

        s3 = self.scores1(s3)
        s3 = self.upsample_2x(s3)
        s2 = self.scores2(s2)
        s2 = s2 + s3

        s1 = self.scores3(s1)
        s2 = self.upsample_4x(s2)
        s = s1 + s2

        s = self.upsample_8x(s)
        return s

 # 定义预测函数

def predict(im, label): # 预测结果
    print(im.shape)
    im = im.cuda()
    out = net(im)
    pred = out.max(1)[1].squeeze().cpu().data.numpy()
    pred = cm[pred]
    return pred, cm[label.numpy()]




cm = np.array(colormap).astype('uint8')
print(resume)
net=torch.load(resume)
#print(net)
voc_test = VOCSegDataset(False, input_shape, img_transforms)


val_loader = DataLoader(voc_test, batch_size=1, num_workers=4)

net= net.cuda()
for i in range(len(val_loader)):

    for img,label in val_loader:

        #img=img.cuda()
        img,label=predict(img,label)
        label=np.squeeze(label)
        cv2.imshow('src',img)
        cv2.imshow('dst',label)
        cv2.waitKey(10000)



# _, figs = plt.subplots(6, 3, figsize=(12, 10))
# for i in range(6):
#     test_data, test_label = voc_test[i]
#     pred, label = predict(test_data, test_label)
#     figs[i, 0].imshow(Image.open(voc_test.data_list[i]))
#     figs[i, 0].axes.get_xaxis().set_visible(False)
#     figs[i, 0].axes.get_yaxis().set_visible(False)
#     figs[i, 1].imshow(label)
#     figs[i, 1].axes.get_xaxis().set_visible(False)
#     figs[i, 1].axes.get_yaxis().set_visible(False)
#     figs[i, 2].imshow(pred)
#     figs[i, 2].axes.get_xaxis().set_visible(False)
#     figs[i, 2].axes.get_yaxis().set_visible(False)