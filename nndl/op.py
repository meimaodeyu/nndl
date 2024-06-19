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
