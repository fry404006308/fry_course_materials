# -*- coding: utf-8 -*-
# @Author  : LG
from torch import nn
import torch
from torch.nn import functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, num_classes = 3, size_average=True):
        """
        FocalLoss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 FocalLoss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retina net中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retina net中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(FocalLoss,self).__init__()
        self.size_average = size_average
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        elif isinstance(alpha,list):
            assert len(alpha)==num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1   #如果α为一个常数,则降低第一类的影响,在目标检测中第一类为背景类
            self.alpha = torch.zeros(num_classes)

            # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
            self.alpha[0].fill_(alpha)
            self.alpha[1:].fill_(1-alpha)

        self.gamma = gamma
        
        print('Focal Loss:')
        print('    Alpha = {}'.format(self.alpha))
        print('    Gamma = {}'.format(self.gamma))
        
    def forward(self, preds, labels):
        """
        FocalLoss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1

        ###############################
        # 一、初始操作
        ###############################

        # 按照最后一个维度重新调整矩阵形状，因为最后一个维度是分类数
        preds = preds.view(-1,preds.size(-1))

        alpha = self.alpha.to(preds.device)

        ###############################
        # 二、计算预测概率Pt
        # focalLoss(pt) = -(1-pt)^γ * log(pt)
        ###############################

        # 将 preds 张量在第 1 个维度上执行 softmax 操作，过softmax之后的，就是pt
        pt = preds_softmax = F.softmax(preds, dim=1)
        # 交叉熵损失函数 CELoss(pt) = -log(pt)，这个pt，就是预估值，多分类是softmax后的概率值，二分类是sigmoid后的值
        # 在softmax后面接着log，这样算交叉熵只用前面加个负号
        log_pt = preds_logSoftmax = torch.log(pt)

        ###############################
        # 三、选真实标签对应的数据
        ###############################

        # labels.view(-1,1) 是将 labels 张量的形状调整为 (N, 1)
        # Ensure the labels are long, not float
        labelsView = labels.view(-1, 1).long()
        # 下面操作的目的就是选出真实标签对应的pt
        pt = pt.gather(1,labelsView)
        # 下面操作的目的就是选出真实标签对应的log_pt
        log_pt = log_pt.gather(1,labelsView)

        ###############################
        # 四、不带α的focal-loss
        ###############################

        # focalLoss(pt) = -(1-pt)^γ * log(pt)
        loss = -torch.mul(torch.pow((1-pt), self.gamma), log_pt)


        ###############################
        # 五、带上α的focal-loss
        ###############################
        # labels.view(-1) 的作用是将 labels 张量的形状调整为一个一维张量
        label_flatten=labelsView.view(-1)
        # 因为算softmax的时候，会把所有的值的softmax都算出来，然而我们需要的只要真实标签的那些而已
        # 所以要进行取值操作
        # 整句话的作用就是alpha根据label值，取到每个分类对应的数值α
        alpha = alpha.gather(0,label_flatten)
        # 损失乘上α向量操作
        loss = torch.mul(alpha, loss.t())


        # 根据需求，看损失是求平均还是求和
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss