class Runner(boject):
    self.model = modle #模型
    self.optimizer = optimizer #优化器
    self.loss_fn =loss_fn #损失函数
    self.metrics =metrics #评价指标

    #模型训练
    def train(self, train_dataset, dev_dataset = None, **kwargs):
       pass

    #模型评价
    def evaluate(self, test_dataset, **kwargs):
        pass

    #模型预测
    def predict(self, x, **kwargs):
        pass

    #模型保存
    def save_model(self, save_path):
        pass

    #模型加载
    def load_model(self, load_path):
        pass