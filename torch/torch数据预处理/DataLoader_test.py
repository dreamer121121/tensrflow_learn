from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder #类似于Keras中的Imgdatagenerator.flow_from_directory()
#ImageFolder也是继承自Dataset类,可以快速创建Dataset类型，但是ImageFolder只返回（img_path,label）
#并没有真正读取图片
catdog = ImageFolder(root='./train')
from collections import Iterable
# print(catdog.imgs)
# print(catdog.classes)
# print(catdog.class_to_idx)
# print(catdog[0][0].size())
#ImageFolder可以自动给图片打上标签，一般情况下通过自写.txt文件给图片打标签
#img_path label
train_loader = DataLoader(catdog) #dataloader是一个可迭代对象
# print(isinstance(train_loader,Iterable))
for i,(img,target) in enumerate(train_loader):
    index = i
    print(index)