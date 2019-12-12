from PIL import Image
from torchvision.transforms import ToTensor,ToPILImage
import torch as t
from torch import nn

to_tensor = ToTensor()
to_pil = ToPILImage()
lena = Image.open('./learn.jpg')
input = to_tensor(lena).unsqueeze(0) #unsqueeze使得batch_size = 1
kernel = t.ones(3,3)/-9
kernel[1][1]=1
conv = nn.Conv2d(1,1,(3,3),1,bias=False)
