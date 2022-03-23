__global__ void KernelFilterKernel(
	int grid_count,
	const float* grid, 
	const float* kernel,
	const int batch,
	const int height,
	const int width,
	const int channel,
	const int k0, // filter size
	const int half_k, // k0 / 2
	const int dilation,
	float* output
) {
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < grid_count; i += blockDim.x * gridDim.x)
	{
		const int w = i % width;
		const int h = (i / width) % height;
		const int c = (i / (width * height)) % channel;
		const int b = (i / (channel * width * height)) % batch;
		const int k_sq = k0 * k0;
		// const int kernel_center = (k_sq - 1) / 2

		// const int sx = 1;
		const int sy = width;
		const int sc = width * height;
		const int sb = channel * width * height;

		float out_value = 0.0f;
		float out_weight = 0.0f;

		
		const int kernel_base = w + h*sy + b * k_sq * width * height; // find filter's weight
		for (int ii_o = -half_k; ii_o <= half_k; ii_o++)
		{
			int xx_o = w + ii_o*(dilation);
			if (xx_o < 0 || xx_o > width - 1)
				continue;
			for (int jj_o = -half_k; jj_o <= half_k; jj_o++)
			{
				int yy_o = h + jj_o*(dilation);
				if (yy_o < 0 || yy_o > height - 1)
					continue;
				int kernel_idx = ((ii_o + half_k) + (jj_o + half_k) * k0)*sc + kernel_base;
				int grid_idx = c*sc + xx_o + yy_o * sy + b * sb;
				if (grid[grid_idx] > 0.0f)
				{
					out_value += grid[grid_idx] * kernel[kernel_idx];
					out_weight += kernel[kernel_idx];
				}
			}	
		}
		output[i] = out_value / (out_weight + 1e-8f);
		// output[i] = h*1000+w;
	}
}

__global__ void KernelFilterGridGradKernel(
	int grid_count,
	const float* grid, 
	const float* kernel,
	const float* backprop,
	const int batch,
	const int height,
	const int width,
	const int channel,
	const int k0,
	const int half_k,
	const int dilation,
	float* grid_grad
) {
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < grid_count; i += blockDim.x * gridDim.x)
	{	
		// compute the location (b, h, w, c) of this pixel
		const int w = i % width;
		const int h = (i / width) % height;
		const int c = (i / (width * height)) % channel;
		const int b = (i / (channel * width * height)) % batch;
		const int k_sq = k0 * k0;
		// const int kernel_center = (k_sq - 1) / 2

		// scales
		// const int sx = 1;
		const int sy = width;
		const int sc = width*height;
		const int sb = channel * width * height;

		float out_value = 0.0f;
		const int kernel_base = k_sq * width * height * b; // batch offset
		
		for (int ii_o = -half_k; ii_o <= half_k; ii_o++)
		{
			int xx_o = w + ii_o*(dilation);
			if (xx_o < 0 || xx_o > width - 1)
				continue;
			for (int jj_o = -half_k; jj_o <= half_k; jj_o++)
			{
				int yy_o = h + jj_o*(dilation); // (xx_o, yy_o) is a filter's center position
				if (yy_o < 0 || yy_o > height - 1)
					continue;
				// the value of weight (-ii_o, -jj_o) of the filter (xx_o, yy_o)
				const int kernel_base_2 = xx_o + sy * yy_o + kernel_base;
				int kernel_idx = ((-ii_o + half_k) + (-jj_o + half_k) * k0)*sc + kernel_base_2;        
				
				float part1 = grid[i] > 0 ? kernel[kernel_idx] : 0;
				float part2 = 0;

				for (int ii_i = -half_k; ii_i <= half_k; ii_i++) // compute ∑weight of this filter
				{
					// (xx_i, yy_i) is the location of pixel match the weight (ii_i, jj_i)
					int xx_i = xx_o + ii_i*(dilation);
					if (xx_i < 0 || xx_i > width - 1)
						continue;
					for(int jj_i = -half_k; jj_i <= half_k; jj_i++)
					{
						int yy_i = yy_o + jj_i*(dilation);
						if (yy_i < 0 || yy_i > height - 1)
							continue;
						int grid_idx_i = xx_i + yy_i * sy + c * sc + b * sb;
						int kernel_idx_i = (ii_i + half_k + (jj_i + half_k) * k0)*sc + kernel_base_2;        
						if (grid[grid_idx_i] > 0)
						{
							part2 += kernel[kernel_idx_i];
						}
					}
				}
				int grid_idx_bp = xx_o + yy_o * sy + c * sc + b * sb;
				out_value += backprop[grid_idx_bp] * part1 / (part2 + 1e-8f);
			}
		}
		grid_grad[i] = out_value;
	}
}

__global__ void KernelFilterKernelGradKernel(
	int weight_count,
	const float* grid, 
	const float* kernel,
	const float* backprop,
	const int batch,
	const int height,
	const int width,
	const int channel,
	const int k0,
	const int half_k,
	const int dilation,
	float* kernel_grad
) {
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < weight_count; i += blockDim.x * gridDim.x)
	{	
		int k_sq = k0 * k0;

		// compute the center location (b, h, w, k) of this filter
		const int w = i % width;
		const int h = (i / width) % height;
		const int k = (i / (width * height)) % k_sq;
		const int b = (i / (width * height * k_sq)) % batch;
		// const int kernel_center = (k_sq - 1) / 2

		// scale
		//const int sx = 1;
		const int sy = width;
		const int sc = width*height;
		const int sb = channel * width * height;
		
		float out_value = 0.0f;
		const int kernel_base = i - k*sc; // the index of the first weight of this filter
		
		// (xx_o, yy_o) is the grid pixel that multiplied with this kernel_weight
		int xx_o = w + (k % k0 - half_k)*(dilation);
		int yy_o = h + (k / k0 - half_k)*(dilation);
		if (xx_o < 0 || xx_o > width - 1 || yy_o < 0 || yy_o > height - 1)
			out_value = 0.0f;
		else
		{
			for (int c = 0; c < channel; c++)
			{
				int grid_idx = xx_o + sy * yy_o + sc*c + sb * b; // the pixel that multiplied with this kernel_weight
				float part1 = grid[grid_idx];
				float part2 = 0.0f; // store the ∑weight of this filter
				float part3 = 0.0f; // store the filtered pixel
				if (part1 > 0)
				{
					for (int ii_i = -half_k; ii_i <= half_k; ii_i++)
					{
						int xx_i = w + ii_i*(dilation);
						if (xx_i < 0 || xx_i > width - 1)
							continue;
						for (int jj_i = -half_k; jj_i <= half_k; jj_i++)
						{
							int yy_i = h + jj_i*(dilation);
							if (yy_i < 0 || yy_i > height - 1)
								continue;
							int grid_idx_i = xx_i + yy_i * sy + c * sc + b * sb;
							int kernel_idx_i = (ii_i + half_k + (jj_i + half_k) * k0)*sc + kernel_base; 
							if (grid[grid_idx_i] > 0)
							{
								part3 += grid[grid_idx_i] * kernel[kernel_idx_i];
								part2 += kernel[kernel_idx_i];
							}
						}
					}
				int grid_idx_bp = w + sy * h + sc * c + sb * b;
				out_value += (part1 * part2 - part3) / (part2 * part2 + 1e-8f) * backprop[grid_idx_bp];
				}
			}
		}
		kernel_grad[i] = out_value;
	}
}

void KernelFilterKernelLauncher(
	const float* grid,
	const float* kernel,
	const int dilation,
	const int* grid_size,
	const int* kernel_size,
	float* output
) {
	int batch = grid_size[0];
	int channel = grid_size[1];
	int height = grid_size[2];
	int width = grid_size[3];

	int k0 = sqrt(kernel_size[1]);
	int half_k = k0 / 2;

	int grid_count = batch * height * width * channel;
	if (grid_count > 0) {
		dim3 GRID((grid_count + 1023) / 1024);
		dim3 BLOCK(1024);
		KernelFilterKernel<<<GRID, BLOCK>>>(
			grid_count,
			grid,
			kernel,
			batch,
			height,
			width,
			channel,
			k0,
			half_k,
			dilation,
			output);
	}
}

void KernelFilterGradKernelLauncher(
	const float* grid,
	const float* kernel,
	const int dilation,
	const float* backprop,
	const int* grid_size,
	const int* kernel_size,
	float* grid_grad,
	float* kernel_grad
) {
	int batch = grid_size[0];
	int channel = grid_size[1];
	int height = grid_size[2];
	int width = grid_size[3];

	int k0 = sqrt(kernel_size[1]);
	int half_k = k0 / 2;


	int grid_count = batch * height * width * channel;
	if (grid_count > 0) {
		dim3 GRID((grid_count + 1023) / 1024);
		dim3 BLOCK(1024);
		KernelFilterGridGradKernel<<<GRID, BLOCK>>>(
			grid_count,
			grid,
			kernel,
			backprop,
			batch,
			height,
			width,
			channel,
			k0,
			half_k,
			dilation,
			grid_grad);
	}

	int weight_count = batch * height * width * kernel_size[1];
	if(weight_count > 0) {
		dim3 GRID((grid_count + 1023) / 1024);
		dim3 BLOCK(1024);
		KernelFilterKernelGradKernel<<<GRID, BLOCK>>>(
			weight_count,
			grid,
			kernel,
			backprop,
			batch,
			height,
			width,
			channel,
			k0,
			half_k,
			dilation,
			kernel_grad);
	}
}