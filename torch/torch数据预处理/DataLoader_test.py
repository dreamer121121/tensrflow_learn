from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder #类似于Keras中的Imgdatagenerator.flow_from_folder()
dataset1 = ImageFolder(root='./train')
print('111')
print(dataset1.imgs)

