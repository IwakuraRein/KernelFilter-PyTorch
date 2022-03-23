#include <vector>

#include "kernel_filter.h"
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT_TENSOR(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor launch_kernel_filter(const torch::Tensor &grid,
                         const torch::Tensor &kernel,
                         const int dilation) {
                         // torch::Tensor &output) {
    CHECK_INPUT_TENSOR(grid)
    CHECK_INPUT_TENSOR(kernel)
    torch::Tensor output = torch::zeros_like(grid);          
    const int grid_size[] = {grid.size(0), grid.size(1), grid.size(2), grid.size(3)};
    const int kernel_size[] = {kernel.size(0), kernel.size(1), kernel.size(2), kernel.size(3)};
    KernelFilterKernelLauncher(
        (const float*)grid.data_ptr(),
        (const float*)kernel.data_ptr(),
        dilation,
        grid_size,
        kernel_size,
        (float*)output.data_ptr()
    );
    return output;
}

std::vector<torch::Tensor> launch_kernel_filter_grad(const torch::Tensor &grid,
                         const torch::Tensor &kernel,
                         const torch::Tensor &backprop,
                         const int dilation) {
                         //torch::Tensor &grid_grad_output,
                         //torch::Tensor &kernel_grad_output) {
    CHECK_INPUT_TENSOR(grid)
    CHECK_INPUT_TENSOR(kernel)
    CHECK_INPUT_TENSOR(backprop)
    torch::Tensor grid_grad_output = torch::zeros_like(grid);
    torch::Tensor kernel_grad_output = torch::zeros_like(kernel);
    const int grid_size[] = {grid.size(0), grid.size(1), grid.size(2), grid.size(3)};
    const int kernel_size[] = {kernel.size(0), kernel.size(1), kernel.size(2), kernel.size(3)};
    KernelFilterGradKernelLauncher(
        (const float*)grid.data_ptr(),
        (const float*)kernel.data_ptr(),
        dilation,
        (const float*)backprop.data_ptr(), 
        grid_size,
        kernel_size,
        (float*)grid_grad_output.data_ptr(),
        (float*)kernel_grad_output.data_ptr()
    );
    return {grid_grad_output, kernel_grad_output};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",
          &launch_kernel_filter,
          "Kernel Filter Forward");

    m.def("backward",
          &launch_kernel_filter_grad,
          "Kernel Filter Backward");
}