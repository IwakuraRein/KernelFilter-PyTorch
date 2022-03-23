# KernelFilter-PyTorch
A customized operation of PyTorch for implementing Kernel Prediction denoising networks.

See：

* [Kernel-Predicting Convolutional Networks for Denoising Monte Carlo Renderings](http://disneyresearch.s3.amazonaws.com/wp-content/uploads/20170630135237/Kernel-Predicting-Convolutional-Networks-for-Denoising-Monte-Carlo-Renderings-Paper33.pdf)
* [Neural Temporal Adaptive Sampling and Denoising](https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.13919)

## Kernel Filter
> KernelFilterClass.forward(*input, kernels, dilation=1*) → Tensor

Applies a 2D Kernel Filter over an input image.

For an input size *(N, C, H, W)*, the size of *kernels* must be *(N, K_Size x K_Size, H, W)*. The output size is the same as input size.

Letting <img src="https://render.githubusercontent.com/render/math?math=I_p^c"> denote the value at position *p*, channel *c* of the input image and <img src="https://render.githubusercontent.com/render/math?math=W_p^n"> denote the value at position *p*, channel *n* of the kernels, the Kernel Filter can be defined as:

<img src="https://render.githubusercontent.com/render/math?math=\tilde{I_p^c} = \frac{\sum_{n=1}^{K\_Size^2} W_p^n \cdot I_p^c}{\sum_{n=1}^{K\_Size^2} W_p^n }">

## Compile
```bash
bash ./install_kernel_filter.sh
```

