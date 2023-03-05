# 引入SummaryWriter
from torch.utils.tensorboard import SummaryWriter

# 将信息写入logs文件夹，可以供TensorBoard消费，来可视化
writer = SummaryWriter("logs")

# 绘制 y = 2x 实例
x = range(100)
for i in x:
    writer.add_scalar('y=2x', i * 2, i)

# 关闭
writer.close()