import torch
import KernelFilter

class kernel_filter_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grid, kernel, dilation):
        ctx.save_for_backward(grid, kernel)
        ctx.dilation = dilation
        output = KernelFilter.forward(grid, kernel, dilation)
        return output

    @staticmethod
    def backward(ctx, backprop):
        grid_grad_output, kernel_grad_output = KernelFilter.backward(*ctx.saved_tensors, backprop, ctx.dilation)
        return grid_grad_output, kernel_grad_output, None

class KernelFilterClass(torch.nn.Module):
    def __init__(self):
        super(KernelFilterClass, self).__init__()

    def forward(self, grid:torch.Tensor, kernel:torch.Tensor, dilation:int=1):
        return kernel_filter_function.apply(grid, kernel, dilation)

