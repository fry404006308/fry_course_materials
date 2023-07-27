import torch
import torch.nn as nn
import numpy as np
import random
from torch.autograd import Variable


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


# 方式一：使用pytorch的MSELoss方法
# 定义损失函数
loss_fn = nn.MSELoss()
loss = loss_fn(input_data,target)

print(loss)

# 方式二：使用基础函数
# 使用 PyTorch 的基础运算符来计算 MSE
diff = target - input_data
squared_diff = diff ** 2
mean_squared_error = torch.mean(squared_diff)

print(mean_squared_error)


"""
运行结果：

tensor([[-1.2061,  0.0617,  1.1632],
        [-1.5008, -1.5944, -0.0187]])
tensor([[-2.1325, -0.5270, -0.1021],
        [ 0.0099, -0.4454, -1.4976]])
tensor(1.4325)
tensor(1.4325)
"""


