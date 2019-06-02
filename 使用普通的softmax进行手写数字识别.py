from tensorflow.examples.tutorials.mnist import  input_data
import tensorflow as tf
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
# print(mnist.train.images.shape,mnist.train.labels.shape)
# print(mnist.test.images.shape,mnist.test.labels.shape)
sess = tf.InteractiveSession() #启动一个会话
x = tf.placeholder(tf.float32,[None,784]) #创建一个数据输入的地方

#定义两个变量OP存储模型参数W和bias且初始化为0
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#实现softmax
y = tf.nn.softmax(tf.matmul(x,W)+b)

#定义cross-entropy

#存放真实的label值
y_ = tf.placeholder(tf.float32,[None,10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))


#tf全局参数初始化
tf.global_variables_initializer().run()
#定义优化器使用SGD算法进行优化
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#每次从训练样本中选出100个组成mini_batch做为训练样本，喂给train_step
for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    train_step.run({x:batch_xs,y_:batch_ys})

#correct_prediction判断预测分类和实际类别是否相符，输出为bool值
correct_predition = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))

#定义测评流程
accuracy = tf.reduce_mean(tf.cast(correct_predition,tf.float32))

print(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels}))
