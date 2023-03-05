# tensorboard 几分钟精通课程

日期: 周日- 2023-02-26 20:58:01

作者: 范仁义

---

🍓

TODO:

---

1😍4💜10😈 复习记忆 🚩 重点 ⭐

---

🍎

🍓

# 课程

🍊

1、tensorboard 最最最最最简单实例（vscode）

2、pycharm 中运行 tensorboard

3、tensorboard 常见功能

4、tensorboard 常见使用实例

🍒

🍌

🍑

🍧

# 一、tensorboard 作用

🍓

📒

TensorBoard 是 tensorflow 官方推出的可视化工具，它可以将模型训练过程中的各种汇总数据展示出来，包括标量(Scalars)、图片(Images)、音频(Audio)、计算图(Graphs)、数据分布(Distributions)、直方图(Histograms)和 Embeddigngs 向量等。

🔧

![img](image/tensorboard使用/hier_tags.png)

🍍

🍉

# 二、tensorboard 安装

🍇

🍋

```
pip install tensorboard
```

🍅

或

```
pip install tensorflow
```

🍐

📖

🌱

🌺

🔥

# 三、tensorboard 最简单实例

✨

🍹

## 1、tensorboard 生成日志

```python
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
```

🧊

🍄

## 2、tensorboard 消费日志

```
tensorboard --logdir=logs

logs 是我们生成日志指定的目录


指定端口

tensorboard --logdir=logs --port=6007
```

🌷

💮

🌸

# 四、tensorboard 常见功能

🍁

🌳

![1677417489361](image/tensorboard使用/1677417489361.png)

🌲

这里列一下，详细的参数说明和例子请参考：
[https://pytorch.org/docs/stable/tensorboard.html#torch-utils-tensorboard](https://links.jianshu.com/go?to=https%3A%2F%2Fpytorch.org%2Fdocs%2Fstable%2Ftensorboard.html%23torch-utils-tensorboard)

- `add_scalar(tag, scalar_value, global_step=None, walltime=None)`：添加标量数据
- `add_scalars(main_tag, tag_scalar_dict, global_step=None, walltime=None)`：添加多个标量数据
- `add_histogram(tag, values, global_step=None, bins='tensorflow', walltime=None, max_bins=None)`：添加一个柱状图
- `add_image(tag, img_tensor, global_step=None, walltime=None, dataformats='CHW')`：添加一张图片
- `add_images(tag, img_tensor, global_step=None, walltime=None, dataformats='NCHW')`：添加多个图片
- `add_figure(tag, figure, global_step=None, close=True, walltime=None)`：渲染一个 `matplotlib`的图片然后添加到 TensorBoard
- `add_video(tag, vid_tensor, global_step=None, fps=4, walltime=None)`：添加视频
- `add_audio(tag, snd_tensor, global_step=None, sample_rate=44100, walltime=None)`：添加音频
- `add_text(tag, text_string, global_step=None, walltime=None)`：添加文本
- `add_graph(model, input_to_model=None, verbose=False)`：添加图像
- `add_embedding(mat, metadata=None, label_img=None, global_step=None, tag='default', metadata_header=None)`：添加嵌入式投影，一个很好的例子就是我们可以将高维数据映射到三维空间中进行直观地展示和可视化
- `add_pr_curve(tag, labels, predictions, global_step=None, num_thresholds=127, weights=None, walltime=None)`：添加 PR 曲线
- `add_custom_scalars(layout)`：添加用户定义的标量
- `add_mesh(tag, vertices, colors=None, faces=None, config_dict=None, global_step=None, walltime=None)`：添加 3D 模型
- `add_hparams(hparam_dict, metric_dict, hparam_domain_discrete=None, run_name=None)`：添加一些可以调节的超参数

🌴

🍎

```
参考


torch.utils.tensorboard — PyTorch 1.13 documentation
https://pytorch.org/docs/stable/tensorboard.html#torch-utils-tensorboard

详解在PyTorch中使用tensorboard – 月来客栈
https://www.ylkz.life/deeplearning/p10491220/

```

🍓

✨

🍹

🍄

🌷

💮

# 五、常用 tensorboard 使用实例

🍊

🍒

🍌

🍑

🍍

## 1、可视化训练情况

🍉

🍇

详见对应代码

🍋

🍅

🍐

📖

## 2、可视化训练的图片

🍧

🍓

详见对应代码

📒

🔧

🌱

🌺

🔥

## 3、可视化训练的模型

✨

🍹

详见对应代码

🧊

🍄

🌷

💮

🌸

## 4、可视化 Precision-Recall 曲线

🍁

🌳

`add_pr_curve`这个方法是用来在训练过程中可视化 Precision-Recall 曲线，即观察在不同阈值下精确率与召回率的平衡情况。更多关于 Precision-Recall 曲线内容的介绍可以参考文章**详解机器学习中的 Precision-Recall 曲线**。用法示例如下所示：

🌲

🌴

```python
def add_pr_curve_demo(writer):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import label_binarize
    def get_dataset():
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        x, y = load_iris(return_X_y=True)
        random_state = np.random.RandomState(2020)
        n_samples, n_features = x.shape
        # 为数据增加噪音维度以便更好观察pr曲线
        x = np.concatenate([x, random_state.randn(n_samples, 100 * n_features)], axis=1)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=random_state)
        return x_train, x_test, y_train, y_test

    x_train, x_test, y_train, y_test = get_dataset()
    model = LogisticRegression(multi_class="ovr")
    model.fit(x_train, y_train)
    y_scores = model.predict_proba(x_test)  # shape: (n,3)

    b_y = label_binarize(y_test, classes=[0, 1, 2])  # shape: (n,3)
    for i in range(3):
        writer.add_pr_curve(f"add_pr_curve 实例：/label_{i}", b_y[:, i], y_scores[:, i], global_step=1)
```

🍧

🍓

在上述代码中，第 2-19 行代码用来根据逻辑回归生成预测结果，其中第 10 行用来给原始数据加入噪音，目的是为了可视化得到更加真实的 PR 曲线；第 21 行用来将原始标签转化为 one-hot 编码形式的标签；第 22-23 行则是分别根据每个类别的预测结果画出对应的 PR 曲线。

📒

🔧

🌱

🌺

🔥

✨

🍹

🧊

🍄

🌷

💮

🌸
