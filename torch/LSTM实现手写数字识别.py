# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 08:53:25 2018
@author: www
"""

import sys

sys.path.append('..')

import torch
import datetime
from torch.autograd import Variable
from torch import nn
from torch.utils.data import DataLoader

from torchvision import transforms as tfs
from torchvision.datasets import MNIST

# 定义数据
data_tf = tfs.Compose([
    tfs.ToTensor(),
    tfs.Normalize([0.5], [0.5])
])
train_set = MNIST(root='./data', train=True, transform=data_tf, download=False)
test_set = MNIST(root='./data', train=False, transform=data_tf, download=False)

train_data = DataLoader(train_set, 64, True, num_workers=4)
test_data = DataLoader(test_set, 128, False, num_workers=4)


# 定义模型
class rnn_classify(nn.Module):
    def __init__(self, in_feature=28, hidden_feature=100, num_class=10, num_layers=2):
        super(rnn_classify, self).__init__()
        self.rnn = nn.LSTM(in_feature, hidden_feature, num_layers)  # 使用两层lstm
        self.classifier = nn.Linear(hidden_feature, num_class)  # 将最后一个的rnn使用全连接的到最后的输出结果

    def forward(self, x):
        # x的大小为（batch，1，28,28），所以我们需要将其转化为rnn的输入格式（28，batch，28）
        x = x.squeeze()  # 去掉（batch，1,28,28）中的1，变成（batch， 28,28）
        x = x.permute(2, 0, 1)  # 将最后一维放到第一维，变成（batch，28,28）
        out, _ = self.rnn(x)  # 使用默认的隐藏状态，得到的out是（28， batch， hidden_feature）
        out = out[-1, :, :]  # 取序列中的最后一个，大小是（batch， hidden_feature)
        out = self.classifier(out)  # 得到分类结果
        return out


net = rnn_classify()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adadelta(net.parameters(), 1e-1)
# print(net)

# # 定义训练过程
def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / total


def train(net, train_data, valid_data, num_epochs, optimizer, criterion):
    if torch.cuda.is_available(): #检查是否有可用的GPU
        net = net.cuda()    #net是nn.Module的子类.cuda的方法是nn.Module的方法
    prev_time = datetime.datetime.now()
    for epoch in range(num_epochs):#10个epoch
        train_loss = 0 #每一个epoch开始设置train_loss 和 train_acc 为0
        train_acc = 0
        net = net.train()
        for im, label in train_data:#此处的im,lable由DataLoader产生并且已经转换为Tensor
            if torch.cuda.is_available():
                im = Variable(im.cuda())  # (bs, 3, h, w)
                label = Variable(label.cuda())  # (bs, h, w)
            else:
                im = Variable(im)
                label = Variable(label)
            # forward,为何没有调用forward方法直接就能进行前向传播呢？
            output = net(im)
            loss = criterion(output, label)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += get_acc(output, label)

        cur_time = datetime.datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            net = net.eval()
            for im, label in valid_data:
                if torch.cuda.is_available():
                    im = Variable(im.cuda())
                    label = Variable(label.cuda())
                else:
                    im = Variable(im)
                    label = Variable(label)
                output = net(im)
                loss = criterion(output, label)
                valid_loss += loss.item()
                valid_acc += get_acc(output, label)
            epoch_str = (
                    "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, "
                    % (epoch, train_loss / len(train_data),
                       train_acc / len(train_data), valid_loss / len(valid_data),
                       valid_acc / len(valid_data)))
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch, train_loss / len(train_data),
                          train_acc / len(train_data)))
        prev_time = cur_time
        print(epoch_str + time_str)


train(net, train_data, test_data, 10, optimizer, criterion)
