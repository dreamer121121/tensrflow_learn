import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch as t

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__() #执行父类的构造函数
        self.conv1 = nn.Conv2d(1,6,5) #输入通道数为1，输出通道数为6，卷积核的大小为5
        self.conv2 = nn.Conv2d(6,16,5)
        #FC层
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,184)
        self.fc3 = nn.Linear(84,10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)

        #reshape
        x = x.view(x.size()[0],-1) #相当于flatten操作,x.size()[0]是样本的个数。
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

net = Net()
#forward函数必须是Variable，只有Variable才能具有自动求导功能，Tensor是没有的，所以在输入时，需要把TensorFlow封装成Variable
input = Variable(t.randn([1,1,32,32]))

out = net(input)
out.size()
