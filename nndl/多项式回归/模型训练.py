from  matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = (12.0, 8.0)

for i, degree in enumerate([0, 1, 3, 8]):  # []中为多项式的阶数
    model = Linear(degree)
    X_train_transformed = polynomial_basis_function(X_train.reshape([-1, 1]), degree)
    X_underlying_transformed = polynomial_basis_function(X_underlying.reshape([-1, 1]), degree)

    model = optimizer_lsm(model, X_train_transformed, y_train.reshape([-1, 1]))  # 拟合得到参数

    y_underlying_pred = model(X_underlying_transformed).squeeze()

    print(model.params)

    # 绘制图像
    plt.subplot(2, 2, i + 1)
    plt.scatter(X_train, y_train, facecolor="none", edgecolor='#e4007f', s=50, label="train data")
    plt.plot(X_underlying, y_underlying, c='#000000', label=r"$\sin(2\pi x)$")
    plt.plot(X_underlying, y_underlying_pred, c='#f19ec2', label="predicted function")
    plt.ylim(-2, 1.5)
    plt.annotate("M={}".format(degree), xy=(0.95, -1.4))

# plt.legend(bbox_to_anchor=(1.05, 0.64), loc=2, borderaxespad=0.)
plt.legend(loc='lower left', fontsize='x-large')
plt.savefig('ml-vis3.pdf')
plt.show()