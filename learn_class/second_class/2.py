#线性回归
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (14,8)
n_observation = 100
xs = np.linspace(-3,3,n_observation)
ys = np.sin(xs)+np.random.uniform(-0.5,0.5,n_observation)
# plt.scatter(xs,ys,marker='s')
# plt.show()

#准备好placeholder
X = tf.placeholder(tf.float32,name="X")
Y = tf.placeholder(tf.float32,name="Y")


#初始化权重参数
W = tf.Variable(tf.random_normal([1]),name='weight') #默认均值为0标准差为1的正态分布
W_1 = tf.Variable(tf.random_normal([1]),name='weight1')
W_2 = tf.Variable(tf.random_normal([1]),name="weight2")
b = tf.Variable(tf.random_normal([1]),name='bias')

#计算预测结果：前项传递的过程
Y_pred = tf.add(tf.multiply(X,W),b)
Y_pred = tf.add(tf.multiply(tf.pow(X,2),W_1),Y_pred)
Y_pred = tf.add(tf.multiply(tf.pow(X,3),W_2),Y_pred)

#指定损失函数
# loss = tf.square(Y-Y_pred,name='loss')
samples_num = xs.shape[0]
loss = tf.reduce_sum(tf.pow(Y_pred-Y,2))/samples_num #为何要用这个损失函数？用原来的行不行？


#初始化optimizer(优化器)：自动进行误差反向传播的过程，调整权值和阈值
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

#指定迭代的次数，并在session里执行graph
n_samples = xs.shape[0]
init = tf.global_variables_initializer() #初始化变量且是全部变量初始化。


#准备工作结束，下面开始创建会话训练模型。
with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('./graph/linear_reg',sess.graph)
    iteras = 1000
    #训练模型
    for i in range(iteras):
        total_loss = 0
        for x,y in zip(xs,ys):
            _,l = sess.run([optimizer,loss],feed_dict={X:x,Y:y}) #两个op一起跑
            total_loss += l
        if i%20 == 0:
            print("Epoch {0}: {1}".format(i,total_loss/n_samples))
    writer.close()

    #取出W和b
    W,W_1,W_2,b = sess.run([W,W_1,W_2,b])

# 绘制出拟合的直线
w = W[0]
w1 = W_1[0]
w2 = W_2[0]
b = b[0]

plt.plot(xs,ys,'bo',label='Real_data')
plt.plot(xs,w*xs+w1*xs**2+w2*xs**3+b,'r',label='Predict_data')
plt.show()


