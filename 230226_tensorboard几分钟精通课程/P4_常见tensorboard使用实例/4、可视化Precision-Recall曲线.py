import numpy as np
# 引入SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize


# 将信息写入logs文件夹，可以供TensorBoard消费，来可视化
writer = SummaryWriter("logs\Precision-Recall")

def add_pr_curve_demo(writer):

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

add_pr_curve_demo(writer)


# 关闭
writer.close()

