import torchvision

# 准备的测试数据集
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10(root=r"G:\_230220_dataSet", train=False, transform=torchvision.transforms.ToTensor())

# batch_size 是每次从数据集取多少图片
# shuffle 是打乱
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

# 测试数据集中第一张图片及target
img, target = test_data[0]
print(img.shape)
print(target)


writer = SummaryWriter("logs/loadImg")
# epoch为2是训练两轮
for epoch in range(2):
    step = 0
    for data in test_loader:
        # 一个batch_size 的图片显示在一张图里面
        imgs, targets = data
        # print(imgs.shape)
        # print(targets)
        # 每一轮的图片显示 在一起
        writer.add_images("Epoch: {}".format(epoch), imgs, step)
        step = step + 1

writer.close()