import math
from torch import nn
from torch.autograd import Function
import torch

import entropy_loss_cuda

# from pytorch_entropy_loss.entropy_loss import EL
class EL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, minV, maxV):
        # 将区间分为[minV, maxV] 步长为1 统计落入各个值的数量
        # input应该为范围在[minV, maxV]之间的非负整数
        # input: b*m*n*c b个m*n*c的矩阵
        # p: b*v
        # g0: b*v
        # grad_input: b*m*n*c

        p = torch.zeros([input.shape[0], maxV - minV + 1]).cuda()  # 每行代表一个样本的概率分布
        for i in range(input.shape[0]):
            p[i] = torch.histc(input[i].view(-1).int(), bins = maxV - minV + 1, min = minV, max = maxV).float() / input[i].numel()

        # 由于0*torch.log2(0)=nan 而在信息熵的定义中 0*log2(0) = 0
        # 因此将p中为0的值换成1 这样在计算log2()的时候可以直接变成0
        p[p == 0] = 1

        ctx.save_for_backward(input, p)

        entropy = -p.mul(p.log2()).sum()
        return entropy

    @staticmethod
    def backward(ctx, grad_output):
        _input, p = ctx.saved_variables
        _c = -(torch.ones_like(p).cuda() / torch.log(torch.ones_like(p)*2).cuda() + p.log2())
        d_input = torch.zeros_like(_input).cuda()

        output = entropy_loss_cuda.backward(_input, _c, d_input)
        d_input = output[0]
        return d_input, None, None


'''
test.py:

import torch

from torch.utils.cpp_extension import load
entropy_loss_cuda = load(
    'entropy_loss_cuda', ['./pytorch_entropy_loss/entropy_loss_cuda.cpp', './pytorch_entropy_loss/entropy_loss_cuda_kernel.cu'], verbose=True)

from pytorch_entropy_loss.entropy_loss import EL

x = torch.randint(0,8,[2,4,4,4]).cuda() # 范围0-7
print(x)
print(x[0].view(-1).histc(bins=8,min=0,max=7))
print(x[1].view(-1).histc(bins=8,min=0,max=7))

xf = x.cuda().float().requires_grad_(True)


el = EL.apply

y = el(xf,0,7)
print(y)

y.backward()

print(xf.grad)
'''
