

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

# 准备数据集
from torch import nn
from torch.utils.data import DataLoader

import time

# 定义训练的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 训练数据集，所以train参数为True
# 应用了torchvision的transforms，将图片数据转换为了tensor格式
train_data = torchvision.datasets.CIFAR10(root=r"G:\_230220_dataSet", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
# 测试数据集，所以train参数为False
test_data = torchvision.datasets.CIFAR10(root=r"G:\_230220_dataSet", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)
# 如果train_data_size=10, 训练数据集的长度为：10
print("训练数据集长度为：{}".format(train_data_size))
print("测试数据集长度为：{}".format(test_data_size))


# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
# 搭建神经网络
class NetModel(nn.Module):
    def __init__(self):
        super(NetModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x
netModel = NetModel()
netModel = netModel.to(device)



# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 优化器
learning_rate = 1e-3
optimizer = torch.optim.Adam(netModel.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("logs/loss_and_accuracy")

begin_time = time.time()

for i in range(epoch):
    print("-------第 {} 轮训练开始-------".format(i+1))

    # 训练步骤开始
    # Sets the module in training mode.
    # This has any effect only on certain modules.
    # See documentations of particular modules for details of their behaviors in training/evaluation mode,
    # if they are affected, e.g. Dropout, BatchNorm, etc.
    netModel.train()
    # 遍历的方式从dataloader中获取数据
    for data in train_dataloader:
        # 获取图片数据和图片对应的标签
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        # 通过模型对图片得到输出
        outputs = netModel(imgs)
        # 通过损失函数计算损失
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        # 梯度置为0
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 进行下一步
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time-begin_time)
            # loss是tensor数据，loss.item()是普通数据，这里两种都可
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    # 主要是对 Dropout, BatchNorm, etc 有效
    # Sets the module in evaluation mode.
    # This has any effect only on certain modules.
    # See documentations of particular modules for details of their behaviors in training/evaluation mode,
    # if they are affected, e.g. Dropout, BatchNorm, etc.
    netModel.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = netModel(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            # 因为输出的结果是(batch_size，分类数)，所以在这里对应的是(64,10)
            # 因为这里的对每个输出显示的是10个分类里面的 对每个分类的概率的估计
            # argmax为1的话就是取里面数值最大的，就是预测的那个分类
            # 显示结果为true或者false，而sum求和就可以算出总共对了多少个
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step = total_test_step + 1

    # 保存模型
    torch.save(netModel, "netModel_{}.pth".format(i))
    print("模型已保存")

writer.close()













