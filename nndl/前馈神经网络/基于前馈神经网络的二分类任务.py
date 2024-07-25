import paddle
import matplotlib.pyplot as plt
from nndl.dataset import make_moons

# 采样1000个样本
n_samples = 1000
X, y = make_moons(n_samples=n_samples, shuffle=True, noise=0.5)

num_train = 640
num_dev = 160
num_test = 200

X_train, y_train = X[:num_train], y[:num_train]
X_dev, y_dev = X[num_train:num_train + num_dev], y[num_train:num_train + num_dev]
X_test, y_test = X[num_train + num_dev:], y[num_train + num_dev:]

y_train = y_train.reshape([-1,1])
y_dev = y_dev.reshape([-1,1])
y_test = y_test.reshape([-1,1])


from nndl.op import Op
# 实现线性层算子
class Linear(Op):
    def __init__(self, input_size, output_size, name, weight_init=paddle.standard_normal, bias_init=paddle.zeros):
        """
        输入：
            - input_size：输入数据维度
            - output_size：输出数据维度
            - name：算子名称
            - weight_init：权重初始化方式，默认使用'paddle.standard_normal'进行标准正态分布初始化
            - bias_init：偏置初始化方式，默认使用全0初始化
        """

        super().__init__()
        self.params = {}
        # 初始化权重
        self.params['W'] = weight_init(shape=[input_size, output_size])
        # 初始化偏置
        self.params['b'] = bias_init(shape=[1, output_size])
        self.inputs = None

        self.name = name

    def forward(self, inputs):
        """
        输入：
            - inputs：shape=[N,input_size], N是样本数量
        输出：
            - outputs：预测值，shape=[N,output_size]
        """
        self.inputs = inputs

        outputs = paddle.matmul(self.inputs, self.params['W']) + self.params['b']
        return outputs

class Logistic(Op):
    def __init__(self):
        self.inputs = None
        self.outputs = None

    def forward(self, inputs):
        """
        输入：
            - inputs: shape=[N,D]
        输出：
            - outputs：shape=[N,D]
        """
        outputs = 1.0 / (1.0 + paddle.exp(-inputs))
        self.outputs = outputs
        return outputs

# 实现一个两层前馈神经网络
class Model_MLP_L2(Op):
    def __init__(self, input_size, hidden_size, output_size):
        """
        输入：
            - input_size：输入维度
            - hidden_size：隐藏层神经元数量
            - output_size：输出维度
        """
        self.fc1 = Linear(input_size, hidden_size, name="fc1")
        self.act_fn1 = Logistic()
        self.fc2 = Linear(hidden_size, output_size, name="fc2")
        self.act_fn2 = Logistic()

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        """
        输入：
            - X：shape=[N,input_size], N是样本数量
        输出：
            - a2：预测值，shape=[N,output_size]
        """
        z1 = self.fc1(X)
        a1 = self.act_fn1(z1)
        z2 = self.fc2(a1)
        a2 = self.act_fn2(z2)
        return a2


# 实现交叉熵损失函数
class BinaryCrossEntropyLoss(Op):
    def __init__(self, model):
        self.predicts = None
        self.labels = None
        self.num = None

        self.model = model

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
        loss = -1. / self.num * (paddle.matmul(self.labels.t(), paddle.log(self.predicts))
                                 + paddle.matmul((1 - self.labels.t()), paddle.log(1 - self.predicts)))

        loss = paddle.squeeze(loss, axis=1)
        return loss

    def backward(self):
        # 计算损失函数对模型预测的导数
        loss_grad_predicts = -1.0 * (self.labels / self.predicts -
                                     (1 - self.labels) / (1 - self.predicts)) / self.num

        # 梯度反向传播
        self.model.backward(loss_grad_predicts)

class Logistic(Op):
    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.params = None

    def forward(self, inputs):
        outputs = 1.0 / (1.0 + paddle.exp(-inputs))
        self.outputs = outputs
        return outputs

    def backward(self, grads):
        # 计算Logistic激活函数对输入的导数
        outputs_grad_inputs = paddle.multiply(self.outputs, (1.0 - self.outputs))
        return paddle.multiply(grads,outputs_grad_inputs)

class Linear(Op):
    def __init__(self, input_size, output_size, name, weight_init=paddle.standard_normal, bias_init=paddle.zeros):
        self.params = {}
        self.params['W'] = weight_init(shape=[input_size, output_size])
        self.params['b'] = bias_init(shape=[1, output_size])

        self.inputs = None
        self.grads = {}

        self.name = name

    def forward(self, inputs):
        self.inputs = inputs
        outputs = paddle.matmul(self.inputs, self.params['W']) + self.params['b']
        return outputs

    def backward(self, grads):
        """
        输入：
            - grads：损失函数对当前层输出的导数
        输出：
            - 损失函数对当前层输入的导数
        """
        self.grads['W'] = paddle.matmul(self.inputs.T, grads)
        self.grads['b'] = paddle.sum(grads, axis=0)

        # 线性层输入的梯度
        return paddle.matmul(grads, self.params['W'].T)


class Model_MLP_L2(Op):
    def __init__(self, input_size, hidden_size, output_size):
        # 线性层
        self.fc1 = Linear(input_size, hidden_size, name="fc1")
        # Logistic激活函数层
        self.act_fn1 = Logistic()
        self.fc2 = Linear(hidden_size, output_size, name="fc2")
        self.act_fn2 = Logistic()

        self.layers = [self.fc1, self.act_fn1, self.fc2, self.act_fn2]

    def __call__(self, X):
        return self.forward(X)

    # 前向计算
    def forward(self, X):
        z1 = self.fc1(X)
        a1 = self.act_fn1(z1)
        z2 = self.fc2(a1)
        a2 = self.act_fn2(z2)
        return a2

    # 反向计算
    def backward(self, loss_grad_a2):
        loss_grad_z2 = self.act_fn2.backward(loss_grad_a2)
        loss_grad_a1 = self.fc2.backward(loss_grad_z2)
        loss_grad_z1 = self.act_fn1.backward(loss_grad_a1)
        loss_grad_inputs = self.fc1.backward(loss_grad_z1)

from nndl.Optimizer  import Optimizer

class BatchGD(Optimizer):
    def __init__(self, init_lr, model):
        super(BatchGD, self).__init__(init_lr=init_lr, model=model)

    def step(self):
        # 参数更新
        for layer in self.model.layers: # 遍历所有层
            if isinstance(layer.params, dict):
                for key in layer.params.keys():
                    layer.params[key] = layer.params[key] - self.init_lr * layer.grads[key]


import os


class RunnerV2_1(object):
    def __init__(self, model, optimizer, metric, loss_fn, **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric

        # 记录训练过程中的评估指标变化情况
        self.train_scores = []
        self.dev_scores = []

        # 记录训练过程中的评价指标变化情况
        self.train_loss = []
        self.dev_loss = []

    def train(self, train_set, dev_set, **kwargs):
        # 传入训练轮数，如果没有传入值则默认为0
        num_epochs = kwargs.get("num_epochs", 0)
        # 传入log打印频率，如果没有传入值则默认为100
        log_epochs = kwargs.get("log_epochs", 100)

        # 传入模型保存路径
        save_dir = kwargs.get("save_dir", None)

        # 记录全局最优指标
        best_score = 0
        # 进行num_epochs轮训练
        for epoch in range(num_epochs):
            X, y = train_set
            # 获取模型预测
            logits = self.model(X)
            # 计算交叉熵损失
            trn_loss = self.loss_fn(logits, y)  # return a tensor

            self.train_loss.append(trn_loss.item())
            # 计算评估指标
            trn_score = self.metric(logits, y).item()
            self.train_scores.append(trn_score)

            self.loss_fn.backward()

            # 参数更新
            self.optimizer.step()

            dev_score, dev_loss = self.evaluate(dev_set)
            # 如果当前指标为最优指标，保存该模型
            if dev_score > best_score:
                print(f"[Evaluate] best accuracy performence has been updated: {best_score:.5f} --> {dev_score:.5f}")
                best_score = dev_score
                if save_dir:
                    self.save_model(save_dir)

            if log_epochs and epoch % log_epochs == 0:
                print(f"[Train] epoch: {epoch}/{num_epochs}, loss: {trn_loss.item()}")

    def evaluate(self, data_set):
        X, y = data_set
        # 计算模型输出
        logits = self.model(X)
        # 计算损失函数
        loss = self.loss_fn(logits, y).item()
        self.dev_loss.append(loss)
        # 计算评估指标
        score = self.metric(logits, y).item()
        self.dev_scores.append(score)
        return score, loss

    def predict(self, X):
        return self.model(X)

    def save_model(self, save_dir):
        # 对模型每层参数分别进行保存，保存文件名称与该层名称相同
        for layer in self.model.layers:  # 遍历所有层
            if isinstance(layer.params, dict):
                paddle.save(layer.params, os.path.join(save_dir, layer.name + ".pdparams"))

    def load_model(self, model_dir):
        # 获取所有层参数名称和保存路径之间的对应关系
        model_file_names = os.listdir(model_dir)
        name_file_dict = {}
        for file_name in model_file_names:
            name = file_name.replace(".pdparams", "")
            name_file_dict[name] = os.path.join(model_dir, file_name)

        # 加载每层参数
        for layer in self.model.layers:  # 遍历所有层
            if isinstance(layer.params, dict):
                name = layer.name
                file_path = name_file_dict[name]
                layer.params = paddle.load(file_path)

from nndl.metric import accuracy
paddle.seed(123)
epoch_num = 1000

model_saved_dir = "model"

# 输入层维度为2
input_size = 2
# 隐藏层维度为5
hidden_size = 5
# 输出层维度为1
output_size = 1

# 定义网络
model = Model_MLP_L2(input_size=input_size, hidden_size=hidden_size, output_size=output_size)

# 损失函数
loss_fn = BinaryCrossEntropyLoss(model)

# 优化器
learning_rate = 0.2
optimizer = BatchGD(learning_rate, model)

# 评价方法
metric = accuracy

# 实例化RunnerV2_1类，并传入训练配置
runner = RunnerV2_1(model, optimizer, metric, loss_fn)

runner.train([X_train, y_train], [X_dev, y_dev], num_epochs=epoch_num, log_epochs=50, save_dir=model_saved_dir)

# 打印训练集和验证集的损失
plt.figure()
plt.plot(range(epoch_num), runner.train_loss, color="#8E004D", label="Train loss")
plt.plot(range(epoch_num), runner.dev_loss, color="#E20079", linestyle='--', label="Dev loss")
plt.xlabel("epoch", fontsize='x-large')
plt.ylabel("loss", fontsize='x-large')
plt.legend(fontsize='large')
plt.savefig('fw-loss2.pdf')
plt.show()

# 加载训练好的模型
runner.load_model(model_saved_dir)
# 在测试集上对模型进行评价
score, loss = runner.evaluate([X_test, y_test])

print("[Test] score/loss: {:.4f}/{:.4f}".format(score, loss))

import math

# 均匀生成40000个数据点
x1, x2 = paddle.meshgrid(paddle.linspace(-math.pi, math.pi, 200), paddle.linspace(-math.pi, math.pi, 200))
x = paddle.stack([paddle.flatten(x1), paddle.flatten(x2)], axis=1)

# 预测对应类别
y = runner.predict(x)
y = paddle.squeeze(paddle.cast((y>=0.5),dtype='float32'),axis=-1)

# 绘制类别区域
plt.ylabel('x2')
plt.xlabel('x1')
plt.scatter(x[:,0].tolist(), x[:,1].tolist(), c=y.tolist(), cmap=plt.cm.Spectral)

plt.scatter(X_train[:, 0].tolist(), X_train[:, 1].tolist(), marker='*', c=paddle.squeeze(y_train,axis=-1).tolist())
plt.scatter(X_dev[:, 0].tolist(), X_dev[:, 1].tolist(), marker='*', c=paddle.squeeze(y_dev,axis=-1).tolist())
plt.scatter(X_test[:, 0].tolist(), X_test[:, 1].tolist(), marker='*', c=paddle.squeeze(y_test,axis=-1).tolist())