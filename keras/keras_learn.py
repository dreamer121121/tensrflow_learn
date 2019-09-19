from keras import Sequential
from keras.layers import Dense,Activation
import numpy as np
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt


trX = np.linspace(-1,1,101)
trY = 3*trX + np.random.randn(*trX.shape)*0.33


plt.scatter(trX,trY)

model = Sequential()
model.add(Dense(input_dim=1,output_dim=1,init='uniform',activation='linear')) #添加一个层,一元线性回归只有一个神经元就足够了
weights = model.layers[0].get_weights()
w_final = weights[0][0][0]
b_final = weights[1][0]
print('Linear regression model is trained to have weight w: %.2f, b: %.2f' % (w_final, b_final))

#设置训练参数
model.compile(optimizer='sgd', loss='mse')

#训练模型
model.fit(trX, trY, epochs=200, verbose=1)


weights = model.layers[0].get_weights()
w_final = weights[0][0][0]
b_final = weights[1][0]
print('Linear regression model is trained to have weight w: %.2f, b: %.2f' % (w_final, b_final))
model.save_weights('./trained_weights_final.h5')
plot_model(model,to_file='./lineregression_model.jpg',show_shapes=True) #模型可视化


y = w_final*trX+b_final
plt.plot(trX,y)
plt.savefig("result.jpg")
plt.show()
