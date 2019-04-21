import tensorflow as tf
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)

#创建一个add节点
add = tf.add(input2,input3)
mul = tf.multiply(input1,add)

with tf.Session() as tf:
    
