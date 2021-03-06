import os
from torch.utils import data
import torch as t
import cv2
import numpy as np
from torchvision import transforms as T #专用的troch计算机视觉库
from PIL import Image
from torch.utils.data import DataLoader

transforms = T.Compose([ #将所有的转换操作进行拼接
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),#转换为tensor 并规范化到0-1之间。
    # T.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])
])#T是以对象的方式返回的，调用是在后台__call__进行的。

class cat_dog(data.Dataset):
    def __init__(self,root,transforms=None):
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root,img) for img in imgs]
        self.transforms = transforms

    def __getitem__(self, index):
        #获取一个样本
        img_path = self.imgs[index]

        label = 1 if 'dog' in img_path.split('/')[-1] else 0

        array = cv2.imread(img_path)#opencv读入图片为nd.array()

        if self.transforms:
            data = self.transforms(Image.fromarray(array))
            return data,label
        else:
            return t.from_numpy(array),label

    def __len__(self):
        #返回数据集的大小
        return len(self.imgs)


dataset = cat_dog(root='./cat_dog',transforms=transforms)
train_loader = DataLoader(dataset,batch_size=10) #迭代器 返回为4维矩阵[batch_size,c,h,w]
print(len(dataset))

# img,label = dataset[1]
# cv2.imshow('img',np.array(img))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# for img,label in dataset:
#     print(img.size(),label)
#     print(img)
# img,label = dataset[0]
# print(type(img))
# print(np.array(img))
# cv2.imshow('img',np.array(img))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
for i,(img,target) in enumerate(train_loader):
    print('index',i)                #i>batch_index
    print(img.size())
    # print(target)
    # break







