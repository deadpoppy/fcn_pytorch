from dataset import *
# 导入需要的包
import os
import torch
import numpy as np
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms as tfs
from datetime import datetime
import torchvision.models as model_zoo
from config import *
import unet
num_classes =len(classes)


# 定义 bilinear kernel
def bilinear_kernel(in_channels, out_channels, kernel_size):
    '''
    return a bilinear filter tensor
    '''
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)







class fcn(nn.Module):
    def __init__(self, num_classes):
        super(fcn, self).__init__()
        pretrained_net = model_zoo.resnet34(pretrained=True)
        self.stage1 = nn.Sequential(*list(pretrained_net.children())[:-4])  # 第一段
        self.stage2 = list(pretrained_net.children())[-4]  # 第二段
        self.stage3 = list(pretrained_net.children())[-3]  # 第三段

        self.scores1 = nn.Conv2d(512, num_classes, 1)
        self.scores2 = nn.Conv2d(256, num_classes, 1)
        self.scores3 = nn.Conv2d(128, num_classes, 1)

        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, 4, bias=False)
        self.upsample_8x.weight.data = bilinear_kernel(num_classes, num_classes, 16)  # 使用双线性 kernel

        self.upsample_4x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
        self.upsample_4x.weight.data = bilinear_kernel(num_classes, num_classes, 4)  # 使用双线性 kernel

        self.upsample_2x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
        self.upsample_2x.weight.data = bilinear_kernel(num_classes, num_classes, 4)  # 使用双线性 kernel

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


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc



def train(epoch, train_loader, optimizer, criterion, net):
    n_batch = len(train_loader)
    net.train()
    for i, (data, lbl) in enumerate(train_loader):
        inputs = data.cuda()
        lbl = lbl.cuda()
        pred = net(inputs)
        loss = criterion(pred, lbl)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('epoch %d, step %d, loss %.4f'%(epoch, i, loss.data.item()))
        # print('best epoch %d, miou %.4f'%(logs['best_ep'], logs['best']))
        # visualize
        # writer.add_scalar('M_global', loss.data.item(), epoch*n_batch+i)



if __name__ == '__main__':

    print('start choose base net!')
    if netflag=='unet':
        net =unet.UNet(n_classes=num_classes, padding=True, up_mode='upsample')
    else:
        net = fcn(num_classes)
    net = net.cuda()
    print('set up net correctelly')

    # optimizer = optim.SGD(net.parameters(), lr=1e-2, weight_decay=1e-4)

    # 实例化数据集

    voc_train = VOCSegDataset(True, input_shape, img_transforms)
    voc_test = VOCSegDataset(False, input_shape, img_transforms)

    train_loader = DataLoader(voc_train, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(voc_test, batch_size=8, num_workers=4)


    # models
    criterion = nn.CrossEntropyLoss(ignore_index=-1).cuda()

    optimizer = torch.optim.Adam(net.parameters(),  1e-4,weight_decay=5e-4)

    if resume is not None:
        net.load_state_dict(torch.load(resume).state_dict())
    if dataparallel:
        net=torch.nn.DataParallel(net)
    for epoch in range(10000):
        net=net.train()

        train(epoch, train_loader, optimizer, criterion, net)

        if val_flag==True and epoch%5==0:
            prev_time = datetime.now()
            net = net.eval()
            eval_loss = 0
            eval_acc = 0
            eval_acc_cls = 0
            eval_mean_iu = 0
            eval_fwavacc = 0
            for data in val_loader:
                im = data[0].cuda()
                label = data[1].cuda()
                # forward
                out = net(im)
                # out = F.log_softmax(out, dim=1)
                loss = criterion(out, label)
                eval_loss += loss.data.item()

                label_pred = out.max(dim=1)[1].data.cpu().numpy()
                label_true = label.data.cpu().numpy()
                for lbt, lbp in zip(label_true, label_pred):
                    acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(lbt, lbp, num_classes)
                    eval_acc += acc
                    eval_acc_cls += acc_cls
                    eval_mean_iu += mean_iu
                    eval_fwavacc += fwavacc

            cur_time = datetime.now()
            h, remainder = divmod((cur_time - prev_time).seconds, 3600)
            m, s = divmod(remainder, 60)
            epoch_str = ('Epoch: {}, \
            Valid Loss: {:.5f}, Valid Acc: {:.5f}, Valid Mean IU: {:.5f} '.format(
                epoch,
                   eval_loss / len(val_loader), eval_acc / len(val_loader), eval_mean_iu / len(val_loader)))
            time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
            print(epoch_str + time_str + ' lr: {}'.format('0.0001'))
        print(epoch%10)
        # if epoch%40==0:
        #     print('start saving')
        #     torch.save(net,'./weight/fcn{}.pth'.format(epoch))
        #     print('save correctly')
            # net=torch.load('./weight/fcn{}.pth'.format(epoch)).cuda()
            # print('load correctly')
            # print(list(net.children()))



