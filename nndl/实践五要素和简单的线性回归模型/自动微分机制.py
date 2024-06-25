'定义两个tensor a和b， 并用stop——gradient属性用来设置是否传递梯度，将a的stop——gradient属性设为false，会自动为a创建一个反向张量，将b的stop——gradient属性设为true，即不会为b创建反向tensor'
#定义张量a，stop——grdient=False代表进行梯度传导
import paddle
a= paddle.to_tensor(2.0, stop_gradient=False)
b= paddle.to_tensor(5.0, stop_gradient=True)
c= a * b
c.backward()
print("Tensor a's grad is:{}".format(a.grad))
print("Tensor b's grad is:{}".format(b.grad))
print("Tensor c's grad is:{}".format(c.grad))
