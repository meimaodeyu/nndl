import paddle
from matplotlib import pyplot as plt

def linear_func(x, w=1.2, b=0.5):
    y = w * x + b
    return y

def creat_toy_data(func, interval, sample_num, noise=0.0, add_outlier=False, outlier_ratio=0.001):
    """根据给定的函数，生成样本
    输入：
        func：函数
        interval： x的取值范围
        sample_num： 样本数目
        noise：噪声均方差
        add_outlier： 是否生成异常值
        outlier_ratio： 异常值占比
    输出：
        x：特征数据， shape=[n_samples, 1]
        y: 标签数据，shape=[n_samples, 1]
    """
    X = paddle.rand(shape=[sample_num]) * (interval[1] - interval[0]) + interval[0]
    y = func(X)
    epsilon = paddle.normal(0.0, std=noise, shape=(y.shape[0],))
    y = y + epsilon
    if add_outlier:
        outlier_num = int(len(y) * outlier_ratio)
        if outlier_num != 0:
            outlier_idx = paddle.randint(0, len(y), shape=[outlier_num])
            y[outlier_idx] = y[outlier_idx] * 5
    return X, y

# 生成训练和测试数据
func = linear_func
interval = (-10, 10)
train_num = 100
test_num = 50
noise = 2
X_train, y_train = creat_toy_data(func=func, interval=interval, sample_num=train_num, noise=noise, add_outlier=False)
X_test, y_test = creat_toy_data(func=func, interval=interval, sample_num=test_num, noise=noise, add_outlier=False)

X_train_large, y_train_large = creat_toy_data(func=func, interval=interval, sample_num=5000, noise=noise, add_outlier=False)
X_underlying = paddle.linspace(interval[0], interval[1], train_num)
y_underlying = linear_func(X_underlying)

plt.scatter(X_train.numpy(), y_train.numpy(), marker='*', facecolors='none', edgecolors='#e4008f', s=50, label='train data')
plt.scatter(X_test.numpy(), y_test.numpy(), marker='none', facecolors='#f19ec2', s=50, label='test data')
plt.plot(X_underlying.numpy(), y_underlying.numpy(), c='#000000', label='underlying distribution')
plt.legend(fontsize='x-large')
plt.show()
plt.savefig('ml-vis.pdf')

# 线性模型
class Linear:
    def __init__(self, input_size):
        self.input_size = input_size
        self.params = {
            'w': paddle.randn(shape=[self.input_size, 1], dtype='float32'),
            'b': paddle.zeros(shape=[1], dtype='float32')
        }

    def forward(self, X):
        y_pred = paddle.matmul(X, self.params['w']) + self.params['b']
        return y_pred

    def __call__(self, X):
        return self.forward(X)

def mean_squared_error(y_true, y_pred):
    error = paddle.mean(paddle.square(y_true - y_pred))
    return error

def optimizer_lsm(model, X, y, reg_lambda=0):
    N, D = X.shape
    x_bar_train = paddle.mean(X, axis=0).T
    y_bar = paddle.mean(y)
    x_sub = X - x_bar_train
    if paddle.all(x_sub == 0):
        model.params['b'] = y_bar
        model.params['w'] = paddle.zeros(shape=[D, 1])
        return model
    tmp = paddle.inverse(paddle.matmul(x_sub.T, x_sub) + reg_lambda * paddle.eye(D))
    w = paddle.matmul(paddle.matmul(tmp, x_sub.T), y - y_bar)
    b = y_bar - paddle.matmul(x_bar_train, w)
    model.params['b'] = b
    model.params['w'] = w
    return model

# 模型训练和评估
input_size = 1
model = Linear(input_size)
model = optimizer_lsm(model, X_train.reshape([-1, 1]), y_train.reshape([-1, 1]))
print("w_pred:", model.params['w'].item(), "b_pred:", model.params['b'].item())

y_train_pred = model(X_train.reshape([-1, 1])).squeeze()
train_error = mean_squared_error(y_true=y_train, y_pred=y_train_pred).item()
print("train_error:", train_error)

model_large = Linear(input_size)
model_large = optimizer_lsm(model_large, X_train_large.reshape([-1, 1]), y_train_large.reshape([-1, 1]))
print("w_pred_large:", model_large.params['w'].item(), "b_pred_large:", model_large.params['b'].item())

y_train_pred_large = model_large(X_train_large.reshape([-1, 1])).squeeze()
train_error_large = mean_squared_error(y_true=y_train_large, y_pred=y_train_pred_large).item()
print("train_error_large:", train_error_large)
