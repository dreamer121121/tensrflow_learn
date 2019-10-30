import torch as t
from torch.autograd import Variable as V
#y = x^2e^x

def f(x):
    y = t.exp(x)*x**2
    return y
def gradf(x):
    dx = 2 * x * t.exp(x) + x ** 2 * t.exp(x)
    return dx

# x = V(t.randn(3,4),requires_grad=True)
# y = f(x)
# dy = gradf(x)
# print("x:",x)
# print("y:",y)
# print("dy:",dy)
x = t.Tensor([2])
print(x)
print(f(x))

