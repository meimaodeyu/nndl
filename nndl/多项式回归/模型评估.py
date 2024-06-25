# 训练误差和测试误差
training_errors = []
test_errors = []
distribution_errors = []

# 遍历多项式阶数
for i in range(9):
    model = Linear(i)

    X_train_transformed = polynomial_basis_function(X_train.reshape([-1, 1]), i)
    X_test_transformed = polynomial_basis_function(X_test.reshape([-1, 1]), i)
    X_underlying_transformed = polynomial_basis_function(X_underlying.reshape([-1, 1]), i)

    optimizer_lsm(model, X_train_transformed, y_train.reshape([-1, 1]))

    y_train_pred = model(X_train_transformed).squeeze()
    y_test_pred = model(X_test_transformed).squeeze()
    y_underlying_pred = model(X_underlying_transformed).squeeze()

    train_mse = mean_squared_error(y_true=y_train, y_pred=y_train_pred).item()
    training_errors.append(train_mse)

    test_mse = mean_squared_error(y_true=y_test, y_pred=y_test_pred).item()
    test_errors.append(test_mse)

    # distribution_mse = mean_squared_error(y_true=y_underlying, y_pred=y_underlying_pred).item()
    # distribution_errors.append(distribution_mse)

print("train errors: \n", training_errors)
print("test errors: \n", test_errors)
# print ("distribution errors: \n", distribution_errors)

# 绘制图片
plt.rcParams['figure.figsize'] = (8.0, 6.0)
plt.plot(training_errors, '-.', mfc="none", mec='#e4007f', ms=10, c='#e4007f', label="Training")
plt.plot(test_errors, '--', mfc="none", mec='#f19ec2', ms=10, c='#f19ec2', label="Test")
# plt.plot(distribution_errors, '-', mfc="none", mec="#3D3D3F", ms=10, c="#3D3D3F", label="Distribution")
plt.legend(fontsize='x-large')
plt.xlabel("degree")
plt.ylabel("MSE")
plt.savefig('ml-mse-error.pdf')
plt.show()