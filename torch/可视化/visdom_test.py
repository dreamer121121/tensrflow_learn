import visdom
import torch as t
#创建客户端
vis = visdom.Visdom(env='test2')
# x = t.arange(1,30,0.01)
# y = t.sin(x)
# vis.line(X=x,Y=y,win='sinx',opts={'title':'y=sin(x)'})

for ii in range(0,10):
    x = t.Tensor([ii])
    y = x
    vis.line(X=x,Y=y,win='polynomial',update='append' if ii > 0 else None)

#updateTrace
x = t.arange(0,9,0.1)
y = (x ** 2)/9
vis.line(X=x,Y=y,win='polynomial',name = 'This is a new trace',update='appendd')
