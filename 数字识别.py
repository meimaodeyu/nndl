mport tensorflow as tf
from keras.datasets import mnist
#引入数据集
import matplotlib.pyplot as plt
#引入绘制曲线库
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
# 引入数据集
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import optimizers
from keras.datasets import mnist
from keras.layers import Dense
# 引入绘制曲线库
from keras.models import Sequential

#引入ANN必要的类
(x_train, y_train), (x_test, y_test) = mnist.load_data()
"""
x_train 是mnist训练集图片，大小的28*28，y_train对应的标签是数字
x_test 是mnist测试集图片，大小的28*28， y_test对应的标签是数字
"""

print('mnist_date:', x_train.shape,y_train.shape,x_test.shape,y_test.shape)
def preprocess(x,y):
    x = tf.cast(x, tf.float32)/255
    x = tf.reshape(x,[28*28])
    y = tf.cast(y,tf.int32)
    y = tf.one_hot(y,10)
    return x,y
batch_size = 128 #每次输入给神经网络的图片数
db = tf.data.Dataset.from_tensor_slices(x_train,y_train)#构建训练集对象
db = db.map(preprocess).shuffle(60000).batch(batch_size)#将数据进行预处理，随即打散和批量处理
ds_val = tf.data.Dataset.from_tensor_slices(x_test,y_test)#构建测试集对象
ds_val = ds_val.map(preprocess).batch(batch_size)#将数据进行预处理，随即打散和批量处理
model = Sequential([Dense(256,activation='relu'),
                    Dense(128,activation='relu'),
                    Dense(64,activation='relu'),
                    Dense(32,activation='relu'),
                    Dense(10,activation='softmax')])
"""构建了五层ANN网络，蒙层的神经元个数分别是256，128，64，32，10，
隐藏层的激活函数是relu，输出层的激活函数是softmax
"""
model.compile(optimizer=optimizers.Adam(lr=0.01),
              loss=tf.losses.categorical_crossentropy(from_logits=False),
              metrics=['accuracy'])
"""
模型的优化器是Adam，学习率是0.01
损失函数是losses.categorical_crossentropy
性能指标是正确率accuracy
"""
history = model.fit(db,epochs=10,validation_data=ds_val,validation_freq=1)
"""
训练次数是5，每1次进行循环测试
"""
model.save('ann_mnist.h5')#以.h5文件格式保存模型
model.evaluate(ds_val)#得到测试集的正确率
simple = next(ds_val)#取一个测试集的数据
x = simple(0)#测试集数据
y = simple(1)#测试集标签
pred = model.predict(x)#将一个测试集数据输入到神经网络的结果
pred = tf.argmax(pred,axis=1)#每个预测结果的概率最大值的下标，预测的数字
y = tf.argmax(y,axis=1)#每个标签的最大值对应的下标，标签对应的数字
print(pred)#打印预测结果
print(y)#打印标签数字
acc = history.history['accuracy']#获取模型训练中的accuracy
val_acc = history.history['val_accuracy']#获取模型训练中的val_accuracy
loss = history.history['loss']
val_loss =history.history['val_loss']
#绘制accuracy曲线
plt.figure(1)
plt.plot(acc, label='Training acc')
plt.plot(val_acc, label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
#绘制loss曲线
plt.figure(2)
plt.plot(loss, label='Training loss')
plt.plot(val_loss, label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()