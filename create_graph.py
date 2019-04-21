import tensorflow as tf

# 创建一个常量OP一行两列
m1 = tf.constant([[3, 3]])
# 创建一个常量OP两行一列
m2 = tf.constant([[2], [3]])
# 创建一个矩阵乘法的OP将m1和m2传入
product = tf.matmul(m1, m2)
# 此处的product尚未进行计算 返回的只是一个tensor.必须将其加入到graph中才能进行计算。

# 创建一个会话，启动默认一个图
Sess = tf.Session()
result = Sess.run(product)#真正进行运算。
Sess.close()
print("---result---", result)





