import torch as t
a = t.randn(4,4)
print(a)
#gather操作和scatter_操作
#注意：
#1.gather操作会生成与index一样的tensor,至于index中属于列索引还是行索引要看gather中指定的dim,dim=0(行) dim = 1(列)
#2.scatter_是gather的逆运算。
