import tensorflow as tf
# x = tf.Variable([2, 3])#b变量与常量的区别
# y = tf.constant([3, 3])
# #定义一个减法op
# sub = tf.subtract(x,y)
# #定义一个加法op
# add = tf.add(sub,x)
#
# #变量初始化
# init = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init)
#     print(sess.run(sub))
#     print(sess.run(add))

#创建一个变量初始化为0,变量可以赋值
state = tf.Variable(0,name="counter")

#创建一个加法op
new_value = tf.add(state,1)

#进行赋值操作tensorflow中赋值操作不能用=，定义一个赋值op
update = tf.assign(state,new_value)

#变量初始化
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(state))
    for i in range(5):
        sess.run(update)
        print(sess.run(state))
