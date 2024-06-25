import paddle

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