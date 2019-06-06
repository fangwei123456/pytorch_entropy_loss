import torch
import torch.nn as nn
import torch.nn.functional as F
# 导入信息熵损失
from torch.utils.cpp_extension import load
entropy_loss_cuda = load(
    'entropy_loss_cuda', ['./pytorch_entropy_loss/entropy_loss_cuda.cpp', './pytorch_entropy_loss/entropy_loss_cuda_kernel.cu'], verbose=True)
from pytorch_entropy_loss.entropy_loss import EL

class Quantize(torch.autograd.Function): # 量化函数
    @staticmethod
    def forward(ctx, input):
        output = torch.round(input) # 量化
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output # 把量化器的导数当做1

def quantize(input):
    return Quantize.apply(input)

def EntropyLoss(input, minV, maxV):
    return EL.apply(input, minV, maxV)

class testNet(nn.Module):
    def __init__(self):
        super(testNet, self).__init__()
        self.conv0 = nn.Conv2d(4, 4, 1)
        self.conv1 = nn.Conv2d(4, 4, 1)

    def forward(self, x):
        return F.leaky_relu(self.conv1(F.leaky_relu(self.conv0(xf))))

xf = torch.randint(0,256,[2,4,8,8]).cuda().float()
xf.requires_grad_(True)
print(xf)
tn = testNet().cuda()
optimizer = torch.optim.Adam(tn.parameters(), lr=1e-5)
yMin = torch.tensor(100).float().cuda()
while(1):
    optimizer.zero_grad()
    encData = tn(xf)
    xq = quantize(encData)
    y = EntropyLoss(xq, xq.min().int(), xq.max().int())
    y.backward()
    optimizer.step()
    if(y<yMin):
        yMin = y
        print(yMin.item())




