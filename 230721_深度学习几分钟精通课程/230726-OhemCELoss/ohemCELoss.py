
import numpy as np
import math

import torch
import torch.nn as nn
import torchvision.transforms as transforms

class OhemCELoss(nn.Module):
    """
    算法本质：
    Ohem本质：核心思路是取所有损失大于阈值的像素点参与计算，但是最少也要保证取n_min个

    """
    def __init__(self, thresh, lb_ignore=255):
        super(OhemCELoss, self).__init__()
        # self.thresh = 0.3567     -ln(0.7) = 0.3567
        # CELoss = -log(Pt)
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float))
        # self.lb_ignore = 255
        self.lb_ignore = lb_ignore
        self.criteria = nn.CrossEntropyLoss(ignore_index=lb_ignore, reduction='none')

    def forward(self, logits, labels):
        # logits: [2,11,1088,896]  batch,classNum,height,width
        # labels: [2,1088,896]  batch,height,width

        # 1、计算n_min（最少算多少个像素点）
        # n_min的大小：一个batch的n张h*w的label图的所有的像素点的十六分之一
        n_min = labels[labels != self.lb_ignore].numel() // 16

        # 2、交叉熵预测得到loss之后，打平成一维的
        loss = self.criteria(logits, labels).view(-1)

        # 3、所有loss中大于阈值的，这边叫做loss hard，这些点才参与损失计算
        # 注意，这里是优化了pytorch中 Ohem 排序的，不然排序太耗时间了
        loss_hard = loss[loss > self.thresh]

        # 4、如果总数小于了n_min，那么肯定要保证有n_min个
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)

        # 5、如果参与的像素点的个数大于了n_min个，那么这些点都参与计算
        loss_hard_mean = torch.mean(loss_hard)

        # 6、返回损失的均值
        return loss_hard_mean


