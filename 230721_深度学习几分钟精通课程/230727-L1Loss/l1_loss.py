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


# 方式一：使用pytorch的L1Loss方法
# 初始化损失函数
loss_fn = nn.L1Loss()
loss = loss_fn(input_data,target)

print(loss)

# 方式二：使用基础函数
diff = torch.abs(target - input_data)
mean_squared_error = torch.mean(diff)

print(mean_squared_error)


"""
运行结果

tensor([[-1.2061,  0.0617,  1.1632],
        [-1.5008, -1.5944, -0.0187]])
tensor([[-2.1325, -0.5270, -0.1021],
        [ 0.0099, -0.4454, -1.4976]])
tensor(1.1532)
tensor(1.1532)

"""
