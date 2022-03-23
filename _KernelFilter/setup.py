from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="pytorch-kernel-filter",
    version = "0.0.1", 
    include_dirs=["include"],
    ext_modules=[
        CUDAExtension(
            "KernelFilter",
            ["KernelFilter.cpp", "kernel_filter.cu"],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
    install_requires=[         
        'torch'
    ]
)