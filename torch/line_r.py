import torch as t
import matplotlib.pyplot as plt
from IPython import display

def get_fake_data(batch_size=8):
    x = t.randn(batch_size,1)*20
    y = x*2+(1+t.randn(batch_size,1))*3
    return x,y

#随机初始化参数
w = t.rand(1,1)
b = t.zeros(1,1)
lr = 0.001

for li in range(20000):
    #mini_batch GD
    x,y = get_fake_data()

    #forward:计算LOss
    y_pred = x.mm(w)+b.expand_as(y)
    loss = 0.5*(y_pred-y)**2
    loss = loss.sum()

    #backward:手动计算梯度
    dloss = 1
    dy_pred = dloss*(y_pred-y)

    dw = x.t().mm(dy_pred)
    db = dy_pred.sum()

    #更新参数
    w.sub_(lr*dw)
    b.sub_(lr*db)

    if li%1000 == 0:

        #画图
        display.clear_output(wait=True)
        x = t.arange(0,20).view(-1,1)
        y = x.mm(w) + b.expand_as(x)
        plt.plot(x.numpy(),y.numpy())

        x2,y2 = get_fake_data(batch_size=20)
        plt.scatter(x2.numpy(),y2.numpy()) #true data

        plt.xlim(0,20)
        plt.ylim(0,41)
        plt.show()
        plt.pause(0.5)

print(w.squeeze()[0],b.squeeze()[0])



