# 真实函数的参数缺省值为 w=1.2，b=0.5
import paddle
from matplotlib import pyplot as plt

#真实函数的参数缺省值 w=1.2，b=0.5
def linear_func(x, w=1.2,b=0.5):
    y = w * x + b
    return y

def create_toy_data(func, interval, sample_num, noise=0.0, add_outlier=False, outlier_ratio=0.001):

    """根据给定的函数，生成样本
     输入：
        func：函数
        interval： x的取值范围
        sample——num： 样本数目
        noise：噪声均方差
        add——outlier： 是否生成异常值
        outlier——ratio： 异常值占比
     输出：
        x：特征数据， shape=【n_samples，1】
        y: 标签数据，shape=【n_samples，1】
        :param sample_num:
     """

    # 均匀采样
    # 使用paddle。rand生成sample——num个随机数
    X = paddle.rand(shape=[sample_num]) * (interval[1] - interval[0]) + interval[0]
    y = func(X)

    # 生成高斯分布的标签噪声
    # 使用paddle。normal生成0均值，noise标准差的数据
    epsilon = paddle.normal(0, noise, paddle.to_tensor(y.shape[0]))
    y = y + epsilon

    if add_outlier:  # 生成额外的异常点
        outlier_num = int(len(y) * outlier_ratio)
        if outlier_num != 0:
            outlier_idx = paddle.randint(len(y), shape=[outlier_num]) #使用paddle.randint生成服从均匀分布的范围在【0，len（y）】的随机tensor
            y[outlier_idx] = y[outlier_idx] * 5

    return X, y



func = linear_func
interval = (-10, 10)
train_num = 100  # 训练样本数目
test_num = 50  # 测试样本数目
noise = 2

X_train, y_train = create_toy_data(func=func, interval=interval, sample_num=train_num, noise=noise, add_outlier=False)
X_test, y_test = create_toy_data(func=func, interval=interval, sample_num=test_num, noise=noise, add_outlier=False)
X_train_large, y_train_large = create_toy_data(func=func, interval=interval, sample_num=5000, noise=noise, add_outlier=False)

#paddle.linspace返回一个tensor，tensor的值为在区间start和stop上均匀间隔的num值，输出tensor的长度为num
X_underlying = paddle.linspace(interval[0], interval[1], train_num)
y_underlying = linear_func(X_underlying)

plt.scatter(X_train, y_train, marker='*', facecolor="none", edgecolor='#e4007f', s=50, label="train data")
plt.scatter(X_test, y_test, facecolor="none", edgecolor='#f19ec2', s=50, label="test data")
plt.plot(X_underlying, y_underlying, c='#000000', label="underlying distribution")
plt.legend(fontsize='x-large')  # 给图像加图例
plt.savefig('ml-vis.pdf')  # 保存图像到PDF文件中
plt.show()
