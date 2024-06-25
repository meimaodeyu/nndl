import math
import paddle
from nndl.op import (Op)


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

def polynomial_basis_function(x, degree = 2):
    """
    输入：
       - x: tensor, 输入的数据，shape=[N,1]
       - degree: int, 多项式的阶数
       example Input: [[2], [3], [4]], degree=2
       example Output: [[2^1, 2^2], [3^1, 3^2], [4^1, 4^2]]
       注意：本案例中,在degree>=1时不生成全为1的一列数据；degree为0时生成形状与输入相同，全1的Tensor
    输出：
       - x_result： tensor
    """
    if degree == 0:
        return paddle.ones(shape=x.shape, dtype='float32')

    x_tmp = x
    x_result = x_tmp

    for i in range(2, degree + 1):
        x_tmp = paddle.multiply(x_tmp, x)  # 逐元素相乘
        x_result = paddle.concat((x_result, x_tmp), axis=-1)

    return x_result


# 简单测试
data = [[2], [3], [4]]
X = paddle.to_tensor(data=data, dtype='float32')
degree = 3
transformed_X = polynomial_basis_function(X, degree=degree)
print("转换前：", X)
print("阶数为", degree, "转换后：", transformed_X)

plt.rcParams['figure.figsize'] = (12.0, 8.0)

paddle.seed(10)#设计随机种子

#线性算子
class Linear(Op):
    def __init__(self, input_size):
        """
        输入：
        input_size:模型要处理的数据特征向量长度,权重向量的维度。
        """

        self.input_size = input_size

        #模型参数
        self.params = {}
        self.params['w'] = paddle.randn(shape=[self.input_size, 1],dtype = 'float32')#随机生成符合标准正态分布的Tensor。
        self.params['b'] = paddle.zeros(shape=[1],dtype = 'float32')#生成指定形状的全0Tensor

    def __call__(self, X):
        return self.forward(X)

    #前向函数
    def forward(self, X):
        """
        输入：
           ——X： tensor， shape=【N,D] 有n个样本，每个样本有D维的特征
           注意这里的x矩阵是由N个xi昂两的转置拼接而成的， 与原教材行向量表示方式不一致
        输出：
           ——y_pred： tensor， shape【N】
        """
        N,D = X.shape

        if self.input_size == 0:#如果权重向量为0
            return paddle.full(shape=[N, 1], fill_value =self.params['b'])

        assert D == self.input_size  #输入数据维度合法性验证

        # 使用paddle。matmul计算两个tensor的成绩 计算两个Tensor乘积遵循广播规则。
        y_pred = paddle.matmul(X,self.params['w'])+self.params['b']

        return y_pred
#这里为了和后面章节统一，这里的x矩阵是由n个x向量的专职拼接成的，与原教材行向量表示放式不一致。
input_size = 3
N = 2
X = paddle.randn(shape=[N, input_size], dtype = 'float32') #生成2个维度为3的数据
model = Linear(input_size)
y_pred = model(X)
print("y_pred: ", y_pred)
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

#paddle.subtract通过广播的方式实现矩阵减向量，逐元素相减
    x_sub = paddle.subtract(X,x_bar_train)

#使用paddle。all判断输入tensor是否全为0
    if paddle.all(x_sub==0):
       model.params['b'] = y_bar
       model.params['w'] = paddle.zeros(shape=[D])
       return model

    tmp = paddle.inverse(paddle.matmul(x_sub.T,x_sub)+ reg_lambda*paddle.eye(num_rows=(D)))#paddle,inverse计算方阵的逆,paddle.eye构建二维Tensor（主对角线元素为1，其他元素为0）

    w = paddle.matmul(paddle.matmul(tmp,x_sub.T),(y - y_bar))

    b= y_bar-paddle.matmul(x_bar_train, w)

    model.params['b'] = b
    model.params['w'] = paddle.squeeze(w,axis=-1)
    return model

for i, degree in enumerate([0, 1, 3, 8]):  # []中为多项式的阶数
    model = Linear(degree)
    X_train_transformed = polynomial_basis_function(X_train.reshape([-1, 1]), degree)
    X_underlying_transformed = polynomial_basis_function(X_underlying.reshape([-1, 1]), degree)

    model = optimizer_lsm(model, X_train_transformed, y_train.reshape([-1, 1]))  # 拟合得到参数

    y_underlying_pred = model(X_underlying_transformed).squeeze()

    print(model.params)

    # 绘制图像
    plt.subplot(2, 2, i + 1)
    plt.scatter(X_train, y_train, facecolor="none", edgecolor='#e4007f', s=50, label="train data")
    plt.plot(X_underlying, y_underlying, c='#000000', label=r"$\sin(2\pi x)$")
    plt.plot(X_underlying, y_underlying_pred, c='#f19ec2', label="predicted function")
    plt.ylim(-2, 1.5)
    plt.annotate("M={}".format(degree), xy=(0.95, -1.4))

# plt.legend(bbox_to_anchor=(1.05, 0.64), loc=2, borderaxespad=0.)
plt.legend(loc='lower left', fontsize='x-large')
plt.savefig('ml-vis3.pdf')
plt.show()

# 训练误差和测试误差
training_errors = []
test_errors = []
distribution_errors = []


# 遍历多项式阶数
for i in range(9):
    model = Linear(i)

    X_train_transformed = polynomial_basis_function(X_train.reshape([-1, 1]), i)
    X_test_transformed = polynomial_basis_function(X_test.reshape([-1, 1]), i)
    X_underlying_transformed = polynomial_basis_function(X_underlying.reshape([-1, 1]), i)

    optimizer_lsm(model, X_train_transformed, y_train.reshape([-1, 1]))

    y_train_pred = model(X_train_transformed).squeeze()
    y_test_pred = model(X_test_transformed).squeeze()
    y_underlying_pred = model(X_underlying_transformed).squeeze()

    train_mse = mean_squared_error(y_true=y_train, y_pred=y_train_pred).item()
    training_errors.append(train_mse)

    test_mse = mean_squared_error(y_true=y_test, y_pred=y_test_pred).item()
    test_errors.append(test_mse)

    # distribution_mse = mean_squared_error(y_true=y_underlying, y_pred=y_underlying_pred).item()
    # distribution_errors.append(distribution_mse)

print("train errors: \n", training_errors)
print("test errors: \n", test_errors)
# print ("distribution errors: \n", distribution_errors)

# 绘制图片
plt.rcParams['figure.figsize'] = (8.0, 6.0)
plt.plot(training_errors, '-.', mfc="none", mec='#e4007f', ms=10, c='#e4007f', label="Training")
plt.plot(test_errors, '--', mfc="none", mec='#f19ec2', ms=10, c='#f19ec2', label="Test")
# plt.plot(distribution_errors, '-', mfc="none", mec="#3D3D3F", ms=10, c="#3D3D3F", label="Distribution")
plt.legend(fontsize='x-large')
plt.xlabel("degree")
plt.ylabel("MSE")
plt.savefig('ml-mse-error.pdf')
plt.show()