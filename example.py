import torch
import PIL.Image
import numpy as np

from kernel_filter import KernelFilterClass

device = torch.device('cuda:0')
Filter = KernelFilterClass()
ImgIn  = np.expand_dims(np.asarray(PIL.Image.open('./Lenna.png')) / 255, axis=0)
ImgIn  = torch.Tensor(ImgIn).permute([0, 3, 1, 2]).to(device).contiguous()
Kernel = torch.ones((1,49,512,512)).to(device) # 7x7 mean filter
ImgOut = Filter(ImgIn, Kernel).permute([0, 2, 3, 1]).cpu().numpy()
ImgOut = (ImgOut[0] * 255).astype(np.uint8)
PIL.Image.fromarray(ImgOut, mode='RGB').save('./Processed.png')
