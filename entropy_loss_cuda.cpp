//https://pytorch.org/tutorials/advanced/cpp_extension.html#writing-a-c-extension
//python setup.py install
#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> EL_cuda_backward(torch::Tensor _input, torch::Tensor _c, torch::Tensor d_input);
// C++ interface
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> EL_backward(torch::Tensor _input, torch::Tensor _c, torch::Tensor d_input)
{
    CHECK_INPUT(_input);
    CHECK_INPUT(_c);
    return EL_cuda_backward(_input, _c, d_input);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("backward", &EL_backward, "EL backward (CUDA)");
}
