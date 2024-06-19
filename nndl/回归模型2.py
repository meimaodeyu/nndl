from matplotlib import pyplot as plt

     func = linear_func
interval = (-10,10)
train_num = 100 #训练样本数量
test_num = 50 #测试样本数目
noise = 2
X_train,y_train = creat_toy_data(func=func, interval=interval, sample_num=train_num, noise=noise, add_outlier=False,)
X_test,y_test = creat_toy_data(func=func, interval=interval, sample_num=test_num, noise=noise, add_outlier=False,)

X_train_large, y_train_large = creat_toy_data(func=func, interval=interval, sample_num=5000, noise=noise, add_outlier=False)
#paddle.linspace返回一个tensor，tensor的值为在区间start和stop上均匀间隔的num值，输出tensor的长度为num
X_underlying = paddle.linspace(interval[0], interval[1],train_num)
y_underlying = linear_func(X_underlying)
X_underlying_np = X_underlying.numpy()
y_underlying_np = y_underlying.numpy()

plt.scatter(X_train,y_train,marker='*',facecolors='none',edgecolors='#e4008f',s=50, label='train date')
plt.scatter(X_test,y_test,marker='none',facecolors='#f19ec2', s=50, label='test date')
plt.plot(X_underlying_np,y_underlying_np,c='#000000', label='underlying distribution')
plt.legend(fontsize='x-large')#给图像加图例
plt.savefig('ml-vis.pdf')
plt.show()