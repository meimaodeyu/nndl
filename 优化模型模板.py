import paddle

class OptimizerLSM:
    def __init__(self, reg_lambda=0):#初始化优化器对象，设置正则化系数，防止过拟合
        self.reg_lambda = reg_lambda

    def fit(self, model, X, y):
        """
        输入：N为样本数，D为特征数
           - model: 模型对象，必须具有 get_params 和 set_params 方法
           - X: tensor, 特征数据，shape=[N, D]
           - y: tensor, 标签数据，shape=[N]
        输出：
           - model: 优化好的模型
        """
        N, D = X.shape

        # 获取模型参数
        params = model.get_params()
        w, b = params['w'], params['b']

        # 计算特征均值和标签均值
        x_bar = paddle.mean(X, axis=0)
        y_bar = paddle.mean(y)

        # 计算去均值后的特征数据，使数据去中心化
        x_sub = X - x_bar

        # 如果去均值后的特征数据全为0，直接设置权重为0，偏置为y的均值
        if paddle.all(x_sub == 0):
            b = y_bar
            w = paddle.zeros(shape=[D], dtype='float32')
            model.set_params({'w': w, 'b': b})
            return model

        # 计算正则化矩阵求逆
        tmp = paddle.inverse(paddle.matmul(x_sub.T, x_sub) + self.reg_lambda * paddle.eye(num_rows=D))

        # 计算权重w
        w = paddle.matmul(paddle.matmul(tmp, x_sub.T), (y - y_bar))

        # 计算偏置b
        b = y_bar - paddle.matmul(x_bar, w)

        # 更新模型参数，将权重改为一维向量
        model.set_params({'w': w.reshape([-1]), 'b': b})

        return model

# 定义线性回归模型类
class LinearModel:
    def __init__(self, input_size):
        self.input_size = input_size
        self.params = {
            'w': paddle.randn(shape=[self.input_size, 1], dtype='float32'),
            'b': paddle.zeros(shape=[1], dtype='float32')
        }

    def get_params(self):
        return self.params

    def set_params(self, params):
        self.params.update(params)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        y_pred = paddle.matmul(X, self.params['w']) + self.params['b']
        return y_pred

# 测试优化器模板
input_size = 3
model = LinearModel(input_size)

# 生成一些随机数据
X = paddle.randn(shape=[5, input_size], dtype='float32')
y = paddle.randn(shape=[5], dtype='float32')

# 创建优化器并优化模型参数
optimizer = OptimizerLSM(reg_lambda=0.1)
model = optimizer.fit(model, X, y)

# 输出优化后的参数
print("Updated weights: ", model.get_params()['w'])
print("Updated bias: ", model.get_params()['b'])