import pandas as pd # 开源数据分析和操作工具
import matplotlib.pyplot as plt  # 可视化工具
import paddle
import paddle.nn as nn
import os

mse_loss = nn.MSELoss()#计算预测值和目标值的均方差误差
paddle.seed(10)

# 利用pandas加载波士顿房价的数据集
data=pd.read_csv("nndl/多项式回归/基于线性回归的波士顿房价预测.py/boston_house_prices.csv")
# 预览前5行数据
data.head()
# 查看各字段缺失值统计情况
data.isna().sum()


# 箱线图查看异常值分布
def boxplot(data, fig_name):
    # 绘制每个属性的箱线图
    data_col = list(data.columns)

    # 连续画几个图片
    plt.figure(figsize=(5, 5), dpi=300)
    # 子图调整
    plt.subplots_adjust(wspace=0.6)
    # 每个特征画一个箱线图
    for i, col_name in enumerate(data_col):
        plt.subplot(3, 5, i + 1)
        # 画箱线图
        plt.boxplot(data[col_name],
                    showmeans=True,
                    meanprops={"markersize": 1, "marker": "D", "markeredgecolor": '#f19ec2'},  # 均值的属性
                    medianprops={"color": '#e4007f'},  # 中位数线的属性
                    whiskerprops={"color": '#e4007f', "linewidth": 0.4, 'linestyle': "--"},
                    flierprops={"markersize": 0.4},
                    )
        # 图名
        plt.title(col_name, fontdict={"size": 5}, pad=2)
        # y方向刻度
        plt.yticks(fontsize=4, rotation=90)
        plt.tick_params(pad=0.5)
        # x方向刻度
        plt.xticks([])
    plt.savefig(fig_name)
    plt.show()


boxplot(data, 'ml-vis5.pdf')
"""使用四分位值筛选出箱线图中分布的异常值，并将这些数据视为噪声，其将被临界值取代"""
# 四分位处理异常值
num_features = data.select_dtypes(exclude=['object', 'bool']).columns.tolist()

for feature in num_features:
    if feature == 'CHAS':
        continue

    Q1 = data[feature].quantile(q=0.25)  # 下四分位
    Q3 = data[feature].quantile(q=0.75)  # 上四分位

    IQR = Q3 - Q1
    top = Q3 + 1.5 * IQR  # 最大估计值
    bot = Q1 - 1.5 * IQR  # 最小估计值
    values = data[feature].values
    values[values > top] = top  # 临界值取代噪声
    values[values < bot] = bot  # 临界值取代噪声
    data[feature] = values.astype(data[feature].dtypes)

# 再次查看箱线图，异常值已被临界值替换（数据量较多或本身异常值较少时，箱线图展示会不容易体现出来）
boxplot(data, 'ml-vis6.pdf')



# 划分训练集和测试集
def train_test_split(X, y, train_percent=0.8):
    n = len(X)
    shuffled_indices = paddle.randperm(n)  # 返回一个数值在0到n-1、随机排列的1-D Tensor
    train_set_size = int(n * train_percent)
    train_indices = shuffled_indices[:train_set_size]
    test_indices = shuffled_indices[train_set_size:]

    X = X.values
    y = y.values

    X_train = X[train_indices]
    y_train = y[train_indices]

    X_test = X[test_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test


X = data.drop(['MEDV'], axis=1)
y = data['MEDV']

X_train, X_test, y_train, y_test = train_test_split(X, y)  # X_train每一行是个样本，shape[N,D]

"""为了消除纲量对数据特征之间影响，在模型训练前，需要对特征数据进行归一化处理，将数据缩放到[0, 1]区间内，使得不同特征之间具有可比性。"""
X_train = paddle.to_tensor(X_train,dtype='float32')
X_test = paddle.to_tensor(X_test,dtype='float32')
y_train = paddle.to_tensor(y_train,dtype='float32')
y_test = paddle.to_tensor(y_test,dtype='float32')

X_min = paddle.min(X_train,axis=0)
X_max = paddle.max(X_train,axis=0)

X_train = (X_train-X_min)/(X_max-X_min)

X_test  = (X_test-X_min)/(X_max-X_min)

# 训练集构造
train_dataset=(X_train,y_train)
# 测试集构造
test_dataset=(X_test)

from nndl.Linear import Linear

# 模型实例化
input_size = 12
model=Linear(input_size)

from nndl.opitimizer import optimizer_lsm


class Runner(object):
    def __init__(self, model, optimizer, loss_fn, metric):
        # 优化器和损失函数为None,不再关注

        # 模型
        self.model = model
        # 评估指标
        self.metric = metric
        # 优化器
        self.optimizer = optimizer

    def train(self, dataset, reg_lambda, model_dir):
        X, y = dataset
        self.optimizer(self.model, X, y, reg_lambda)

        # 保存模型
        self.save_model(model_dir)

    def evaluate(self, dataset, **kwargs):
        X, y = dataset

        y_pred = self.model(X)
        result = self.metric(y_pred, y)

        return result

    def predict(self, X, **kwargs):
        return self.model(X)

    def save_model(self, model_dir):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        params_saved_path = os.path.join(model_dir, 'params.pdtensor')
        paddle.save(model.params, params_saved_path)

    def load_model(self, model_dir):
        params_saved_path = os.path.join(model_dir, 'params.pdtensor')
        self.model.params = paddle.load(params_saved_path)

optimizer = optimizer_lsm

# 实例化Runner
runner = Runner(model, optimizer=optimizer,loss_fn=None, metric=mse_loss)

# 模型保存文件夹
saved_dir = 'D:\Workspace\pythonProject2\mdoels'

# 启动训练
runner.train(train_dataset,reg_lambda=0,model_dir=saved_dir)
columns_list = data.columns.to_list()
weights = runner.model.params['w'].tolist()
b = runner.model.params['b'].item()

for i in range(len(weights)):
    print(columns_list[i],"weight:",weights[i])

print("b:",b)
# 加载模型权重
runner.load_model(saved_dir)

mse = runner.evaluate(test_dataset)
print('MSE:', mse.item())
runner.load_model(saved_dir)
pred = runner.predict(X_test[:1])
print("真实房价：",y_test[:1].item())
print("预测的房价：",pred.item())