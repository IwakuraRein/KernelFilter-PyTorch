#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

void KernelFilterKernelLauncher(
	const float* grid,
	const float* kernel,
	const int dilation,
	const int* grid_size,
	const int* kernel_size,
	float* output
);

void KernelFilterGradKernelLauncher(
	const float* grid,
	const float* kernel,
	const int dilation,
	const float* backprop,
	const int* grid_size,
	const int* kernel_size,
	float* output_grad,
	float* weight_grad
);