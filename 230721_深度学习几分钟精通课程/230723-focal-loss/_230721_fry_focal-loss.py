import torch
from focal_loss import FocalLoss
import numpy as np
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)

pred = torch.randn((3,5))
print("pred:",pred)

label = torch.tensor([2,3,4])
print("label:",label)


# GPU运行
pred = pred.to('cuda')
print("pred:",pred)
label = label.to('cuda')
print("label:",label)



loss_fn = FocalLoss(alpha=[1,2,3,1,2], gamma=2, num_classes=5)
# loss_fn = FocalLoss(alpha=0.25, gamma=2, num_classes=5)
loss = loss_fn(pred, label)
print(loss)
