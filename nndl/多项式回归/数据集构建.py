import math
import paddle

#sin函数：sin（2*pi*x)
def sin(x):
    y = paddle.sin(2*math.pi*x)
    return y
def create_toy_data(func, interval, sample_num, noise=0.0, add_outlier=False, outlier_ratio = 0.001, ):


    """根据给定的函数，生成样本
     输入：
        func：函数
        interval： x的取值范围
        sample——num： 样本数目
        noise：噪声均方差
        add——outlier： 是否生成异常值
        outlier——ratio： 异常值占比
     输出：
        x：特征数据， shape=【n——samples，1】
        y: 标签数据，shape=【n——samples，1】
        :param sample_num:
     """

    # 均匀采样
    # 使用paddle。rand生成sample——num个随机数
    X = paddle.rand(shape=[sample_num]) * (interval[1] - interval[0]) + interval[0]
    y = func(X)

    # 生成高斯分布的标签噪声
    # 使用paddle。normal生成0均值，noise标准差的数据
    epsilon = paddle.normal(0.0, std=noise, shape=(y.shape[0]) )
    y = y + epsilon
    if add_outlier:
       outlier_num = int(len(y)*outlier_ratio)
       if outlier_num != 0:
         #使用paddle。randint生成服从均匀分布的，范围在【0，len（y）】的随机tensor
          outlier_idx = paddle.randint(len(y),shape=[outlier_num])
          y[outlier_idx] = y[outlier_idx] * 5
    return X,y
from  matplotlib import pyplot as plt
# 生成数据
func = sin
interval = (0,1)
train_num = 15
test_num = 10
noise = 0.5 #0.1 
X_train, y_train = create_toy_data(func=func, interval=interval, sample_num=train_num, noise = noise)
X_test, y_test = create_toy_data(func=func, interval=interval, sample_num=test_num, noise = noise)

X_underlying = paddle.linspace(interval[0],interval[1],num=100)
y_underlying = sin(X_underlying)

# 绘制图像
plt.rcParams['figure.figsize'] = (8.0, 6.0)
plt.scatter(X_train, y_train, facecolor="none", edgecolor='#e4007f', s=50, label="train data")
#plt.scatter(X_test, y_test, facecolor="none", edgecolor="r", s=50, label="test data")
plt.plot(X_underlying, y_underlying, c='#000000', label=r"$\sin(2\pi x)$")
plt.legend(fontsize='x-large')
plt.savefig('ml-vis2.pdf')
plt.show()