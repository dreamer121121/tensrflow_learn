import numpy as np
from keras.datasets import mnist #与tf类似主动从网站向拉去数据集
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt



data = input_data.read_data_sets("./data",one_hot=True)
x_train = data.train.images
y_train_ohe = data.train.labels
x_test = data.test.images
y_test_ohe = data.test.labels
print(y_train_ohe.shape)
# plt.imshow(x_train[0].reshape(28,28),cmap='gray')
# plt.show()

#查看数据的格式
# print(x_train[0].shape)
# print(y_train[0])

x_train = x_train.reshape(x_train.shape[0],28,28,1).astype('float32')
x_test = x_test.reshape(x_test.shape[0],28,28,1).astype("float32")

# #像素值归一化
# x_train /= 255
# x_test /= 255

# #One-hot编码
# def tran_y(y):
#     y_ohe = np.zeros(10)
#     y_ohe[y] = 1
#     return y_ohe

# y_train_ohe = np.array([tran_y(y_train[i]) for i in range(len(y_train))])
# y_test_ohe = np.array([tran_y(y_test[i]) for i in range(len(y_test))])

#开始构建模型
model = Sequential()
model.add(Conv2D(filters=34,kernel_size=(3,3),strides = (1,1),padding='same',input_shape=(28,28,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.5))

model.add(Conv2D(filters=128,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(256,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(10,activation='softmax'))
#构建模型结束

#设置训练参数
model.compile(loss='categorical_crossentropy',optimizer='adagrad',metrics=['accuracy'])
model.fit(x_train,y_train_ohe,validation_data=(x_test,y_test_ohe),epochs=20,batch_size=128)

scores = model.evaluate(x_test,y_test_ohe,verbose=0)




