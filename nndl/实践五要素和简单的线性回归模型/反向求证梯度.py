import math


# 定义加法算子类
class Add:
    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        self.x = x
        self.y = y
        return x + y

    def backward(self, grad):
        return grad, grad


# 定义乘法算子类
class Multiply:
    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        self.x = x
        self.y = y
        return x * y

    def backward(self, grad):
        return grad * self.y, grad * self.x


# 定义指数运算算子类
class Exponential:
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.x = x
        return math.exp(x)

    def backward(self, grad):
        return grad * math.exp(self.x)


# 示例使用这些算子进行反向传播验证
if __name__ == "__main__":
    # 假设给定的变量值
    a = 2.0
    b = 3.0
    c = 4.0
    d = 5.0

    # 实例化算子
    add_op = Add()
    multiply_op = Multiply()
    exponential_op = Exponential()

    # 前向传播，计算得到 y
    step1 = exponential_op(a)
    step2 = multiply_op(step1, b)
    step3 = multiply_op(c, d)
    y = add_op(step2, step3)

    print("计算得到的 y 值:", y)

    # 反向传播，计算梯度
    # 假设损失函数对 y 的梯度为 1
    grad_y = 1.0

    # 反向传播计算梯度
    grad_step2, grad_step3 = add_op.backward(grad_y)
    grad_c, grad_d = multiply_op.backward(grad_step3)
    grad_step1, grad_b = multiply_op.backward(grad_step2)
    grad_a = exponential_op.backward(grad_step1)

    # 打印计算得到的梯度
    print("计算得到的梯度 (a):", grad_a)
    print("计算得到的梯度 (b):", grad_b)
    print("计算得到的梯度 (c):", grad_c)
    print("计算得到的梯度 (d):", grad_d)

    # 数值梯度计算方法（使用中心差分法）
    eps = 1e-6
    numerical_grad_a = (exponential_op(a + eps) - exponential_op(a - eps)) / (2 * eps)
    numerical_grad_b = (add_op(multiply_op(exponential_op(a), b + eps), multiply_op(c, d)) -
                        add_op(multiply_op(exponential_op(a), b - eps), multiply_op(c, d))) / (2 * eps)
    numerical_grad_c = (add_op(multiply_op(exponential_op(a), b), multiply_op(c + eps, d)) -
                        add_op(multiply_op(exponential_op(a), b), multiply_op(c - eps, d))) / (2 * eps)
    numerical_grad_d = (add_op(multiply_op(exponential_op(a), b), multiply_op(c, d + eps)) -
                        add_op(multiply_op(exponential_op(a), b), multiply_op(c, d - eps))) / (2 * eps)

    # 打印数值计算的梯度
    print("数值计算的梯度 (a):", numerical_grad_a)
    print("数值计算的梯度 (b):", numerical_grad_b)
    print("数值计算的梯度 (c):", numerical_grad_c)
    print("数值计算的梯度 (d):", numerical_grad_d)

    # 比较梯度
    tol = 1e-4
    assert abs(grad_a - numerical_grad_a) < tol, "梯度计算错误 (a)"
    assert abs(grad_b - numerical_grad_b) < tol, "梯度计算错误 (b)"
    assert abs(grad_c - numerical_grad_c) < tol, "梯度计算错误 (c)"
    assert abs(grad_d - numerical_grad_d) < tol, "梯度计算错误 (d)"

    print("梯度计算验证通过！")

"""
在数值计算中，tol 通常是指容忍误差或容忍度（tolerance）的缩写。在比较数值结果时，我们通常会允许一定的误差范围，而 tol 就是用来设定这个允许的误差范围的阈值。

具体来说，在我们验证梯度计算的过程中，我们会计算解析梯度（通过反向传播得到的梯度）和数值梯度（通过中心差分法计算得到的梯度）之间的差异。然后，我们将这个差异与 tol 进行比较：

如果差异小于 tol，我们可以接受这个计算结果，认为解析梯度和数值梯度是一致的。
如果差异大于 tol，则说明解析梯度和数值梯度之间的差异超出了我们设定的误差容忍度，可能需要进一步调查和修正。
在实际应用中，tol 的选择取决于问题的性质和数值计算的精度要求。通常情况下，tol 的值会根据具体的问题和计算环境来进行调整，以确保误差控制在合理的范围内。

例如，在验证神经网络的反向传播时，通常会将 tol 设置为一个较小的值，例如 1e-4 或 1e-6，以确保梯度计算的准确性和稳定性。

"""