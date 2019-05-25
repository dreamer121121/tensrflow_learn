import tensorflow as tf

#定义第一个op
# a = tf.add(3,5)#定义了一个加法op(画图）
# with tf.Session() as sess:
#     print(sess.run(a))

#定义稍微复杂点的模型：
#--------------------画图
x = 2
y = 3
op1 = tf.add(x,y)
op2 = tf.multiply(x,y)
op3 = tf.pow(op1,op2)
#-------------------画图
#启动会话
with  tf.Session() as sess:
    writer = tf.summary.FileWriter('./logfile',sess.graph)
    print(sess.run(op3))
writer.close()