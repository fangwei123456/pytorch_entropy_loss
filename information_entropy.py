import torch
import torch.nn as nn
import torch.nn.functional as F

def information_entropy(input_data, C, sigma=1, mean=True):
    '''
    :param input_data: 量化后的数据，batch_size * ?
    :param C: 量化取值集合，L行1列
    :param sigma: 越大，算出的频率越接近实际频率，但导数会越趋近0，容易出现梯度消失
    近似脉冲函数为exp(-sigma*|x|)
    :param mean: 为True则返回batch_size个数据的平均，否则返回一个size=[batch_size]的Tensor
    :return: 信息熵
    '''
    IE = torch.zeros(size=[input_data.shape[0]], device=input_data.device)
    for i in range(input_data.shape[0]):
        approximate_p = torch.zeros_like(C).float()
        for j in range(C.shape[0]):
            approximate_p[j] = torch.exp(-sigma * torch.abs(input_data[i] - C[j])).sum() + 1e-6
            # + 1e-6防止出现log2(0)
        approximate_p = approximate_p / approximate_p.sum()
        IE[i] = -torch.sum(approximate_p * torch.log2(approximate_p))
    if mean:
        return IE.mean()  # 返回batchSize个数据的平均信息熵
    else:
        return IE

if __name__ == "__main__":
    x = torch.randint(low=0, high=16, size=[8, 3, 16, 16]).float()
    C = torch.arange(start=0, end=16).unsqueeze_(1).float()
    sigma = 1
    el = information_entropy(x, C, sigma)
    print(el)
