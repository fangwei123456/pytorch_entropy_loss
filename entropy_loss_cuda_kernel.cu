#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

namespace {
template <typename scalar_t>
__global__ void EL_cuda_backward_kernel(
        torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> _input,
        torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> _c,
        torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> d_input)
{
          d_input[blockIdx.x][blockIdx.y][threadIdx.x][threadIdx.y]
                  = _c[blockIdx.x][ (int)(_input[blockIdx.x][blockIdx.y][threadIdx.x][threadIdx.y]) ];
}

} // namespace

std::vector<torch::Tensor> EL_cuda_backward(torch::Tensor _input, torch::Tensor _c, torch::Tensor d_input)
{
    //_input: b*c*m*n 用来计算信息熵的数据，b个c通道m*n矩阵
    //如果运行出错，可以通过pytorch的reshape修改b c m n使得尽量均匀
    //_c: b*v, v = maxV - minV + 1 概率分布向量

    const auto size_b = _input.size(0);
    const auto size_c = _input.size(1);
    const auto size_m = _input.size(2);
    const auto size_n = _input.size(3);

    dim3 threadsPerBlock(size_m, size_n);
    dim3 numBlocks(size_b, size_c);

    AT_DISPATCH_FLOATING_TYPES(_c.type(), "EL_forward_cuda", ([&] {
        EL_cuda_backward_kernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
        _input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        _c.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        d_input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>());
  }));
  return {d_input};
}
