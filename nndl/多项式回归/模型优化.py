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
