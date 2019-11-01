import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import torch as t
show=ToPILImage() #将tensor转换为PIL对象，方便实现可视化

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

#训练集
trainset = tv.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
trainloader = t.utils.data.DataLoader(trainset,batch_size = 4,shuffle=True,num_workers = 2)

#测试集
testset = tv.datasets.CIFAR10()
