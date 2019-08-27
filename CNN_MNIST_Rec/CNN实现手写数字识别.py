
# coding: utf-8

# In[10]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# In[11]:


#载入数据集
mnist = input_data.read_data_sets("./MNIST_data",one_hot=True)

#每个批次的大小
batch_size = 100
#计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

#定义权值初始化函数,权值和偏置值都是变量类型。
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
    
    
#定义偏置值初始化函数
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
    
#定义卷积操作
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

#定义池化操作
def max_pool_2X2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


#定义两个placeholder
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

#将x的格式转化为4D向量
x_image = tf.reshape(x,[-1,28,28,1])
#[]的第一个值-1代表batch_size我们已经设置为了100
#此处若不被设置为-1将会报如下错误
#Input to reshape is a tensor with 78400 values, but the requested shape has 784

#真正初始化第一个卷积层的权值和偏置值
w_conv1 = weight_variable([5,5,1,32])#卷积核大小为5X5，通道数为1，卷积核的数量为32
b_conv1 = bias_variable([32]) #每一个卷积核对应一个偏置值，32个卷积核对应32个偏置值

#第一个卷积层
#进行第一个卷积运算
h_conv1 = tf.nn.relu(conv2d(x_image,w_conv1)+b_conv1)#卷积运算并加上偏置值后用Relu函数进行激活
#马上进行池化运算
h_pool1 = max_pool_2X2(h_conv1)

#真正初始化第二个卷积层的权值和偏置值
w_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

#第二个卷积层
#进行第二个卷积运算
h_conv2 = tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2)
#马上进行池化运算
h_pool2 = max_pool_2X2(h_conv2)#至此得到64个7*7的图片

#初始化一个全连接层的权值和偏置值
w_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])



#把池化层2的输出平滑成1D数组
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
#求第一个全连接层的输出
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)

#keep_prob用来表示神经元的输出概率
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

#初始化第二个全连接层的权值和偏置
w_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])

#计算输出
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2)+b_fc2) #输出softmax即每一类的概率值

#交叉熵代价函数
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
#使用AdamOptimizer进行优化
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#结果存放在一个布尔列表中
correct_prediction = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))#返回一维张量中最大的值所在的位置

#计算准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#启动会话
with tf.Session()as sess:
    sess.run(tf.global_variables_initializer())#全部变量初始化
    # writer = tf.summary.FileWriter('./graph', sess.graph)
    for epoch in range(21):#总共迭代21个周期
        for batch in range(n_batch):# 每一次迭代会使用全部的训练数据集，每一次训练使用batch_size个数据
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.7})
        #完成一次训练，下面使用测试数据集对当前的模型额准确率进行测试
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0}) #
        print("Iter:"+str(epoch)+','"Testing,Accuracy="+str(acc))
    # writer.close()
        
            
    










    

