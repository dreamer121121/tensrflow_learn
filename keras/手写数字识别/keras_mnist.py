import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint

data = input_data.read_data_sets("./data",one_hot=True)
x_train = data.train.images
y_train_ohe = data.train.labels
x_test = data.test.images
y_test_ohe = data.test.labels

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

model.add(Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(10,activation='softmax'))
#构建模型结束

#模型可视化
# plot_model(model,to_file='./networkstructure.jpg',show_shapes=True)


#设置训练参数
model.compile(loss='categorical_crossentropy',optimizer='adagrad',metrics=['accuracy'])
filepath = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath,monitor='val_acc',verbose=1,save_best_only=True,mode='max')
callback_list = [checkpoint] #利用回调函数来保存最好的模型
model.fit(x_train,y_train_ohe,validation_data=(x_test,y_test_ohe),epochs=20,batch_size=128,callbacks=callback_list)

scores = model.evaluate(x_test,y_test_ohe,verbose=0)
# model.save("m1.h5") #直接进行保存不好若训练过程中途中断，则需要重新进行训练





