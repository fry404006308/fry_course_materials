# 引入SummaryWriter
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')




##### 1、add_scalar实例
def add_scalar_demo():

    # 将信息写入logs文件夹，可以供TensorBoard消费，来可视化
    writer = SummaryWriter("logs/add_scalar")

    # 绘制 y = 2x 实例
    x = range(100)
    for i in x:
        writer.add_scalar('add_scalar实例：y=2x', i * 2, i)

    # 关闭
    writer.close()
add_scalar_demo()

##### 2、add_scalars 实例
def add_scalars_demo():
    # 将信息写入logs文件夹，可以供TensorBoard消费，来可视化
    writer = SummaryWriter("logs/add_scalars")
    r = 5
    for i in range(100):
        writer.add_scalars('add_scalars实例', {'xsinx':i*np.sin(i/r),
                                        'xcosx':i*np.cos(i/r),
                                        'tanx': np.tan(i/r)}, i)
    # 关闭
    writer.close()

add_scalars_demo()


##### 3、add_text 实例
def add_text_demo():
    # 将信息写入logs文件夹，可以供TensorBoard消费，来可视化
    writer = SummaryWriter("logs/add_text")

    writer.add_text('lstm', 'This is an lstm', 0)
    writer.add_text('rnn', 'This is an rnn', 10)
    # 关闭
    writer.close()

add_text_demo()





##### 4、add_graph 实例
def add_graph_demo():
    # 将信息写入logs文件夹，可以供TensorBoard消费，来可视化
    writer = SummaryWriter("logs/add_graph")

    img = torch.rand([1, 3, 64, 64], dtype=torch.float32)
    model = torchvision.models.AlexNet(num_classes=10)
    # print(model)
    writer.add_graph(model, input_to_model=img)
    # 关闭
    writer.close()

add_graph_demo()




##### 5、add_image 实例
def add_image_demo():

    # 将信息写入logs文件夹，可以供TensorBoard消费，来可视化
    writer = SummaryWriter("logs/add_image")


    img1 = np.random.randn(1, 100, 100)
    writer.add_image('add_image 实例：/imag1', img1)
    img2 = np.random.randn(100, 100, 3)
    writer.add_image('add_image 实例：/imag2', img2, dataformats='HWC')
    img = Image.open('../imgs/1.png')
    img_array = np.array(img)
    writer.add_image('add_image 实例：/cartoon', img_array, dataformats='HWC')
    # 关闭
    writer.close()

add_image_demo()



##### 6、add_images 实例
def add_images_demo():

    # 将信息写入logs文件夹，可以供TensorBoard消费，来可视化
    writer = SummaryWriter("logs/add_images")

    imgs1 = np.random.randn(8, 100, 100, 1)
    writer.add_images('add_images 实例/imgs1', imgs1, dataformats='NHWC')
    imgs2 = np.zeros((16, 3, 100, 100))
    for i in range(16):
        imgs2[i, 0] = np.arange(0, 10000).reshape(100, 100) / 10000 / 16 * i
        imgs2[i, 1] = (1 - np.arange(0, 10000).reshape(100, 100) / 10000) / 16 * i
    writer.add_images('add_images 实例/imgs2', imgs2)  # Default is :math:`(N, 3, H, W)`
    img = Image.open('../imgs/1.jpg')
    img3 = np.array(img)
    imgs4= np.zeros((5, img3.shape[0], img3.shape[1], img3.shape[2]))
    for i in range(5):
        imgs4[i] = img3//(i+1)
    writer.add_images('add_images 实例/imgs4', imgs4, dataformats='NHWC')  # Default is :math:`(N, 3, H, W)`

    # 关闭
    writer.close()
add_images_demo()


##### 7、add_figure 实例
def add_figure_demo():
    # 将信息写入logs文件夹，可以供TensorBoard消费，来可视化
    writer = SummaryWriter("logs/add_figure")

    # First create some toy data:
    x = np.linspace(0, 2 * np.pi, 400)
    y = np.sin(x ** 2)

    # Create just a figure and only one subplot
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title('Simple plot')

    writer.add_figure("add_figure 实例：figure", fig)
    # 关闭
    writer.close()

add_figure_demo()


##### 8、add_pr_curve 实例
def add_pr_curve_demo():
    # 将信息写入logs文件夹，可以供TensorBoard消费，来可视化
    writer = SummaryWriter("logs/add_pr_curve")

    labels = np.random.randint(2, size=100)  # binary label
    predictions = np.random.rand(100)
    writer.add_pr_curve('add_pr_curve 实例：pr_curve', labels, predictions, 0)
    # 关闭
    writer.close()

add_pr_curve_demo()


##### 9、add_embedding 实例
def add_embedding_demo():
    # 将信息写入logs文件夹，可以供TensorBoard消费，来可视化
    writer = SummaryWriter("logs/add_embedding")

    import tensorflow as tf
    import tensorboard as tb
    tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

    import keyword
    import torch
    meta = []
    while len(meta) < 100:
        meta = meta + keyword.kwlist  # get some strings
    meta = meta[:100]

    for i, v in enumerate(meta):
        meta[i] = v + str(i)

    label_img = torch.rand(100, 3, 10, 32)
    for i in range(100):
        label_img[i] *= i / 100.0

    writer.add_embedding(torch.randn(100, 5), metadata=meta, label_img=label_img)

    # 关闭
    writer.close()

add_embedding_demo()


