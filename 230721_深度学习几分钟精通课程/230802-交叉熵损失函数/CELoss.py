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
# 对于交叉熵损失，目标应该是类别的索引，这里我们随机生成0，1，2这三种类别
target = torch.tensor([random.randint(0, 2) for _ in range(2)], dtype=torch.long)
print(target)


# 使用Pytorch的 CrossEntropyLoss方法
# 初始化损失函数
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(input_data, target)

print(loss)

# 方式二：使用softmax和基础函数
def cross_entropy_loss(input, label) :
    prob = nn.functional.softmax(input, dim=1)
    log_prob = torch.log(prob)
    label_view = label.view(-1, 1)
    loss = -log_prob.gather(1, label_view)
    loss_mean = loss.mean()
    return loss_mean

loss2 = cross_entropy_loss(input_data, target)
print(loss2)


"""
结果：

tensor([[-1.2061,  0.0617,  1.1632],
        [-1.5008, -1.5944, -0.0187]])
tensor([2, 2])
tensor(0.3577)
tensor(0.3577)
"""


