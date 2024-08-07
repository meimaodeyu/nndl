import paddle
from nndl import op
import matplotlib.pyplot as plt
class Op(object):
    def __init__(self):
        pass
    def __call__(self, inputs):
        return self.forward(inputs)
    #前向函数
    #输入：张量inputs
    #输出:张量outputs
    def forward(self, inputs):
        # return output
       raise NotImplementedError

    # 反向函数
    #输入：最终输出对outputs的梯度outputs———grads
    #输出：最终输出对inputs的梯度inputs——grads
    def backward(self,outputs_grads):
        # return inputs——grads
        raise NotImplementedError

'加法算子'
class add(Op):
    def __init__(self):
        super(add, self).__init__()

    def __call__(self,x, y):
        return self.forward(x, y)
    def forward(self, x, y):
        self.x=x
        self.y=y
        outputs = x + y
        return outputs

    def backward(self, grads):
        grads_x=grads * 1
        grads_y=grads * 1
        return grads_x , grads_y

'定义x=1，y=4，根据反向计算，得到x，y的梯度'
x=1
y=4
add_op = add()
z=add_op(x,y)
grads_x, grads_y = add_op.backward(grads=1)
print("x's grad is:", grads_x)
print("y's grad is:", grads_y)

'乘法算子'
class multiply(Op):
    def __init__(self):
        super(multiply, self).__init__()
    def __call__(self,x,y):
        return self.forward(x,y)
    def forward(self, x, y):
        self.x=x
        self.y=y
        outputs = x * y
        return outputs
    def backward(self, grads):
        grads_x=grads * self.y
        grads_y=grads * self.x
        return grads_x,grads_y

'指数算子'
import math
class exponential(Op):
    def __init__(self):
        super(exponential, self).__init__()
    def forward(self,x):
        self.x = x
        outputs = math.exp(x)
        return outputs
    def backward(self, grads):
        grads_x=grads* math.exp(self.x)
        return grads_x
'分别指定a,b,c,d的值，通过实例化算子，调用加法，乘法和指数运算算子，计算得到y'
a,b,c,d = 2, 3, 2, 2
#实例化算子
multiply_op = multiply()
add_op = add()
exponential_op = exponential()
y = exponential_op(add_op(multiply_op(a, b), multiply_op(c, d)))
print('y:', y)

import paddle
from nndl.activation import softmax

paddle.seed(10)  # 设置随机种子


class Op(object):
    def __init__(self):
        pass

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, inputs):
        raise NotImplementedError


# 线性算子
class Linear(Op):
    def __init__(self, dimension):
        """
        输入：
           - dimension:模型要处理的数据特征向量长度
        """

        self.dim = dimension

        # 模型参数
        self.params = {}
        self.params['w'] = paddle.randn(shape=[self.dim, 1], dtype='float32')
        self.params['b'] = paddle.zeros(shape=[1], dtype='float32')

    def __call__(self, X):
        return self.forward(X)

    # 前向函数
    def forward(self, X):
        """
        输入：
           - X: tensor, shape=[N,D]
           注意这里的X矩阵是由N个x向量的转置拼接成的，与原教材行向量表示方式不一致
        输出：
           - y_pred： tensor, shape=[N]
        """

        N, D = X.shape

        if self.dim == 0:
            return paddle.full(shape=[N, 1], fill_value=self.params['b'])

        assert D == self.dim  # 输入数据维度合法性验证

        # 使用paddle.matmul计算两个tensor的乘积
        y_pred = paddle.matmul(X, self.params['w']) + self.params['b']

        return y_pred


# 新增Softmax算子
class model_SR(Op):
    def __init__(self, input_dim, output_dim):
        super(model_SR, self).__init__()
        self.params = {}
        # 将线性层的权重参数全部初始化为0
        self.params['W'] = paddle.zeros(shape=[input_dim, output_dim])
        # self.params['W'] = paddle.normal(mean=0, std=0.01, shape=[input_dim, output_dim])
        # 将线性层的偏置参数初始化为0
        self.params['b'] = paddle.zeros(shape=[output_dim])
        # 存放参数的梯度
        self.grads = {}
        self.X = None
        self.outputs = None
        self.output_dim = output_dim

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        self.X = inputs
        # 线性计算
        score = paddle.matmul(self.X, self.params['W']) + self.params['b']
        # Softmax 函数
        self.outputs = softmax(score)
        return self.outputs

    def backward(self, labels):
        """
        输入：
            - labels：真实标签，shape=[N, 1]，其中N为样本数量
        """
        # 计算偏导数
        N = labels.shape[0]
        labels = paddle.nn.functional.one_hot(labels, self.output_dim)
        self.grads['W'] = -1 / N * paddle.matmul(self.X.t(), (labels - self.outputs))
        self.grads['b'] = -1 / N * paddle.matmul(paddle.ones(shape=[N]), (labels - self.outputs))


# 新增多类别交叉熵损失
class MultiCrossEntropyLoss(Op):
    def __init__(self):
        self.predicts = None
        self.labels = None
        self.num = None

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)

    def forward(self, predicts, labels):
        """
        输入：
            - predicts：预测值，shape=[N, 1]，N为样本数量
            - labels：真实标签，shape=[N, 1]
        输出：
            - 损失值：shape=[1]
        """
        self.predicts = predicts
        self.labels = labels
        self.num = self.predicts.shape[0]
        loss = 0
        for i in range(0, self.num):
            index = self.labels[i]
            loss -= paddle.log(self.predicts[i][index])
        return loss / self.num