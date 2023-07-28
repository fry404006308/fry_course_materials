import torch
import torch.nn as nn
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



# 创建一些简单的输入数据和目标数据
input_data = torch.randn(2, 3)
print(input_data)
target = torch.randn(2, 3)
print(target)


# 方式一：使用pytorch的 SmoothL1Loss方法
# 初始化损失函数
loss_fn = nn.SmoothL1Loss()
loss = loss_fn(input_data,target)

print(loss)

# 方式二：使用基础函数
def smooth_l1_loss(x, y,beta=1) :
    diff = torch.abs(x - y)
    loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    return loss.mean()

loss2 = smooth_l1_loss(input_data,target)
print(loss2)

"""
运行结果

tensor([[-1.2061,  0.0617,  1.1632],
        [-1.5008, -1.5944, -0.0187]])
tensor([[-2.1325, -0.5270, -0.1021],
        [ 0.0099, -0.4454, -1.4976]])
tensor(0.6677)
tensor(0.6677)

"""
