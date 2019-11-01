import torch as t
from torch.autograd import Variable as V
from torch import nn

rnn = nn.LSTM(10,20,2) #输入信息为10维的特征向量，隐藏元维20维的向量，两层LSTM

#每句话有5个词
#总共3句话 （batch_size）
#每个词由10维的词向量表示
input = V(t.randn(5,3,10))

#隐藏元(cell state and hidden state)的初始值
#形状(num_layers,batch_size,hidden_size)

h0 = V(t.zeros(2,3,20))
c0 = V(t.zeros(2,3,20))

#output是最后一层所有隐藏元的值
#hn和cn是所有层(这里有两层)的最后一个隐藏元的值

output,(hn,cn) = rnn(input,(h0,c0))
print(output.size())
print(hn.size())
print(cn.size())
