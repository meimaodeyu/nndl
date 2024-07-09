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

def softmax(X):
    """
    输入：
        - X：shape=[N, C]，N为向量数量，C为向量维度
    """
    x_max = paddle.max(X, axis=1, keepdim=True)#N,1
    x_exp = paddle.exp(X - x_max)
    partition = paddle.sum(x_exp, axis=1, keepdim=True)#N,1
    return x_exp / partition

# 观察softmax的计算方式
X = paddle.to_tensor([[0.1, 0.2, 0.3, 0.4],[1,2,3,4]])
predict = softmax(X)
print(predict)
class model_SR(op.Op):
    def __init__(self, input_dim, output_dim):
        super(model_SR, self).__init__()
        self.params = {}
        # 将线性层的权重参数全部初始化为0
        self.params['W'] = paddle.zeros(shape=[input_dim, output_dim])
        # self.params['W'] = paddle.normal(mean=0, std=0.01, shape=[input_dim, output_dim])
        # 将线性层的偏置参数初始化为0
        self.params['b'] = paddle.zeros(shape=[output_dim])
        self.outputs = None

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        """
        输入：
            - inputs: shape=[N,D], N是样本数量，D是特征维度
        输出：
            - outputs：预测值，shape=[N,C]，C是类别数
        """
        # 线性计算
        score = paddle.matmul(inputs, self.params['W']) + self.params['b']
        # Softmax 函数
        self.outputs = softmax(score)
        return self.outputs

def logistic(x):
    return 1 / (1 + paddle.exp(-x))#对输入Tensor逐元素进行一自然数e为底的指数运算

# 在[-10,10]的范围内生成一系列的输入值，用于绘制函数曲线
x = paddle.linspace(-10, 10, 10000)
plt.figure()
plt.plot(x.tolist(), logistic(x).tolist(), color="#e4007f", label="Logistic Function")
# 设置坐标轴
ax = plt.gca()
# 取消右侧和上侧坐标轴
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
# 设置默认的x轴和y轴方向
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
# 设置坐标原点为(0,0)
ax.spines['left'].set_position(('data',0))
ax.spines['bottom'].set_position(('data',0))
# 添加图例
plt.legend()
plt.savefig('linear-logistic.pdf')
plt.show()



class model_LR(op.Op):
    def __init__(self, input_dim):
        super(model_LR, self).__init__()
        self.params = {}
        # 将线性层的权重参数全部初始化为0
        self.params['w'] = paddle.zeros(shape=[input_dim, 1])
        # self.params['w'] = paddle.normal(mean=0, std=0.01, shape=[input_dim, 1])
        # 将线性层的偏置参数初始化为0
        self.params['b'] = paddle.zeros(shape=[1])

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        """
        输入：
            - inputs: shape=[N,D], N是样本数量，D为特征维度
        输出：
            - outputs：预测标签为1的概率，shape=[N,1]
        """
        # 线性计算
        score = paddle.matmul(inputs, self.params['w']) + self.params['b']
        # Logistic 函数
        outputs = logistic(score)
        return outputs

class BinaryCrossEntropyLoss(op.Op):
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
            loss = -1. / self.num * (#paddle.log 对输入Tensor逐元素的计算自然对数。 paddle.one 生成指定形状的全1Tensor.
                        paddle.matmul(self.labels.t(), paddle.log(self.predicts)) + paddle.matmul((1 - self.labels.t()),
                                                                                                  paddle.log( 1 - self.predicts)))
            loss = paddle.squeeze(loss, axis=1)
            return loss

class MultiCrossEntropyLoss(op.Op):
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

