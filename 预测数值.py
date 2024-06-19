import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from tensorflow.keras.models import Sequential

#生成随机数据
#设置随机数种子以确保结果可复现
np.random.seed(0)
#生成随即输入数据X和目标数据y
X = np.random.randn(1000,100,1)#1000个样本，每个样本100个时间步长，1个特征
y = np.random.randn(1000)#1000个对应的目标值
#划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#构建CNN模型
model = Sequential()
model.add(Conv1D(filters=64,kernel_size=3,activation='relu',input_shape=(X_train.shape[1],1)))
model.add(Conv1D(filters=64),kernel_size=3,activation='relu')
model.add(Flatten())
model.add(Dense(units=50,activation='relu'))
model.add(Dense(units=1))#输出层，一个神经元用于回归任务
#编辑模型
model.compile(optimizer='adam',loss='mse')
#训练模型
model.fit(X_train,y_train,batch_size=32,epochs=50,validation_split=0.1)
#进行预测并评估模型
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Test MSE: {mse}')
#绘制实际值和预测值的比较图
plt.figure(figsize=10,5)
plt.plot(y_test,'Actual')
plt.plot(y_pred.flatten(),'Predicted')
plt.title('Comparison of Actual and Predicted Values')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.legend()
plt.show()

error = y_test - y_pred.flatten()
plt.figure(figsize=(10,5))
plt.plot(error,'Errors')
plt.title('Prediction Error')
plt.xlabel('Sample Index')
plt.ylabel('Error')
plt.legend()
plt.show()