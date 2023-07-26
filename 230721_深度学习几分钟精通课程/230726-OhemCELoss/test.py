import torch
import torch.nn as nn
import numpy as np
import random
from torch.autograd import Variable

from ohemCELoss import OhemCELoss


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)

# 一、生成预测值
# 假设我们有一个具有 2 个样本，每个样本有 3 个分类，高度和宽度均为 4 的预测输出
logits = Variable(torch.randn(2, 3, 4, 4))

# 二、生成真实标签
# 对应的真实标签，每个样本的标签是一个 4x4 的图像
labels = Variable(torch.randint(0, 3, (2, 4, 4)))


# 三、初始化
# 创建 OhemCELoss 的实例，阈值设为 0.7
ohem_criterion = OhemCELoss(thresh=0.7)

# 四、计算OHEM损失
# 计算损失
loss = ohem_criterion(logits, labels)

print('Loss:', loss.item())




