from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import logging
import  logging.config
import os

logging_config = os.path.join(os.path.dirname(__file__), 'logging.conf')#logging.conf路径
logging.config.fileConfig(logging_config)
logger = logging.getLogger('root')

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
sess = tf.InteractiveSession()


def weight_variable(shape):
    """
    定义权值变量（卷积网络中的权值变量就是卷积核）
    :param shape:
    :return:
    """
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """
    定义偏置参数
    :param shape:
    :return:
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    """
    定义卷积函数以便复用
    :param x:
    :param W:
    :return:
    """
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2X2(x):
    """
    定义池化层以便复用
    :param x:
    :return:
    """
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

x = tf.placeholder(tf.float32,[None,784])
y_ = tf.placeholder(tf.float32,[None,10])
x_image = tf.reshape(x,[-1,28,28,1])


#--------定义第一个卷积层-------

#定义第一个卷积层的卷积核（5X5 1通道 32个卷积核）
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
h_pool1 = max_pool_2X2(h_conv1)

#定义第二个卷积层
W_conv2 = weight_variable([5,5,32,64]) #注意第二层卷积层的权值矩阵的第三个参数为32，即为32通道的（因其上一层卷积核的数量为32个）
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2 = max_pool_2X2(h_conv2)


#定义一个全连接层
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

#加入dropput层
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

#加入softmax层
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)


#定义损失函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv),reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


#定义评测准确率操作
correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


#开始训练过程
tf.global_variables_initializer().run()

for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict = {x:batch[0],y_:batch[1],keep_prob:1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
        logger.info("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})

print("test accuracy%g"%accuracy.eval(feed_dict={x:mnist.text.images,y_:mnist.test.labels,keep_prob:1.0}))
logger.info("test accuracy%g"%accuracy.eval(feed_dict={x:mnist.text.images,y_:mnist.test.labels,keep_prob:1.0}))
