#真实函数的参数缺省值 w=1.2，b=0.5
def linear_func(x, w=1.2,b=0.5):
    y = w*x + b
    return y

import paddle

def creat_toy_data(func, interval, sample_num, noise=0.0, add_outlier=False, outlier_ratio = 0.001, ):


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

from matplotlib import pyplot as plt

func = linear_func
interval = (-10,10)
train_num = 100 #训练样本数量
test_num = 50 #测试样本数目
noise = 2
X_train,y_train = creat_toy_data(func=func, interval=interval, sample_num=train_num, noise=noise, add_outlier=False,)
X_test,y_test = creat_toy_data(func=func, interval=interval, sample_num=test_num, noise=noise, add_outlier=False,)

X_train_large, y_train_large = creat_toy_data(func=func, interval=interval, sample_num=5000, noise=noise, add_outlier=False)
#paddle.linspace返回一个tensor，tensor的值为在区间start和stop上均匀间隔的num值，输出tensor的长度为num
X_underlying = paddle.linspace(interval[0], interval[1],train_num)
y_underlying = linear_func(X_underlying)

plt.scatter(X_train,y_train,marker='*',facecolors='none',edgecolors='#e4008f',s=50, label='train date')
plt.scatter(X_test,y_test,marker='none',facecolors='#f19ec2', s=50, label='test date')
plt.plot(X_underlying,y_underlying,c='#000000', label='underlying distribution')
plt.legend(fontsize='x-large')#给图像加图例
plt.savefig('ml-vis.pdf')
plt.show()


import paddle
from nndl.op import (Op)

paddle.seed(10)#设计随机种子

#线性算子
class Linear(Op):
    def __init__(self, input_size):
        """
        输入：
        input_size:模型要处理的数据特征向量长度
        """

        self.input_size = input_size

        #模型参数
        self.params = {}
        self.params['w'] = paddle.randn(shape=[self.input_size, 1],dtype = 'float32')
        self.params['b'] = paddle.zeros(shape=[1],dtype = 'float32')

    def __call__(self, X):
        return self.forward(X)

    #前向函数
    def forward(self, X):
        """
        输入：
           ——X： tensor， shape=【N,D]
           注意这里的x矩阵是由N个xi昂两的转置拼接而成的， 与原教材行向量表示方式不一致
        输出：
           ——y_pred： tensor， shape【N】
        """
        N,D = X.shape

        if self.input_size == 0:
            return paddle.full(shape=[N, 1], fill_value =self.params['b'])

        assert D == self.input_size  #输入数据维度合法性验证

        # 使用paddle。matmul计算两个tensor的成绩
        y_pred = paddle.matmul(X,self.params['w'])+self.params['b']

        return y_pred
#这里为了和后面章节统一，这里的x矩阵是由n个x向量的专职拼接成的，与原教材行向量表示放式不一致。
input_size = 3
N = 2
X = paddle.randn(shape=[N, input_size], dtype = 'float32') #生成2个维度为3的数据
model = Linear(input_size)
y_pred = model(X)
print("y_pred: ", y_pred)


import paddle
def mean_squared_error(y_true, y_pred):
    """
    输入：
    y_true: tensor，样本真实标签
    y_pred: tensor，样本预测标签
    输出：
    ——error: float， 误差值
    """

    assert y_true.shape[0] == y_pred.shape[0]

    #paddle.square计算输入的平方值
    #paddle.mean沿 axis 计算 x 的平均值，默认axis是None，则对输入的全部元素计算平均值
    error = paddle.mean(paddle.square(y_true - y_pred))

    return error
# 构造一个简单的样例进行测试：【N,1],N=2
y_true = paddle.to_tensor([[-0,2],[4,9]], dtype='float32')
y_pred = paddle.to_tensor([[1,3],[2,5]],dtype='float32')
error = mean_squared_error(y_true=y_true, y_pred=y_pred)
print('error: ', error)

def optimizer_lsm(model, X, y, reg_lambda=0):
    """
    输入
    model: 模型
    X: tensor， 特征数据，shape=【N,D]
    y: tensor。 标签数据，shape=【N】
    reg_lambda: float，正则化系数，默认为0
    输出：
    return:优化好的模型
    """
    N, D = X.shape

#对输入特征数据所有特征向量求平均
    x_bar_train = paddle.mean(X,axis=0).T

#求标签的均值，shape=【1】
    y_bar = paddle.mean(y)

#paddle.subtract通过广播的方式实现矩阵减向量
    x_sub = paddle.subtract(X,x_bar_train)

#使用paddle。all判断输入tensor是否全为0
    if paddle.all(x_sub==0):
       model.params['b'] = y_bar
       model.params['w'] = paddle.zeros(shape=[D])
       return model

    tmp = paddle.inverse(paddle.matmul(x_sub.T,x_sub)+ reg_lambda*paddle.eye(num_rows=(D)))

    w = paddle.matmul(paddle.matmul(tmp,x_sub.T),(y - y_bar))

    b= y_bar-paddle.matmul(x_bar_train, w)

    model.params['b'] = b
    model.params['w'] = paddle.squeeze(w,axis=-1)
    return model

#通过以上实现的线性回归类来拟合训练数据，并输出模型在训练集上的损失
input_size = 1
model = Linear(input_size)
model = optimizer_lsm(model,X_train.reshape([-1,1]),y_train.reshape([-1,1]))
print("w_pred:",model.params['w'].item(),"b_per:",model.params['b'].item())

y_train_pred = model(X_train.reshape([-1,1])).squeeze()
train_error = mean_squared_error(y_true=y_train, y_pred=y_train_pred).item()
print("train_error:",train_error)

model_large = Linear(input_size)
model_large = optimizer_lsm(model_large,X_train_large.reshape([-1,1]),y_train_large.reshape([-1,1]))
print("w_per large:",model_large.params['w'].item(),"b_per large:",model_large.params['b'].item())

y_train_pred_large = model_large(X_train_large.reshape([-1,1])).squeeze()
train_error_large = mean_squared_error(y_true=y_train_large, y_pred=y_train_pred_large).item()
print("train_error_large:",train_error_large)
