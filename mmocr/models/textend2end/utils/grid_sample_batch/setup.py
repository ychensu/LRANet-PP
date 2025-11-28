from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name="custom",
    include_dirs=["./src"],
    ext_modules=[
        CUDAExtension(
            "batch_grid_sample",
            [ "./src/GridSamplerBatch_cuda.cu","./src/grid_sampler.cpp","./src/GridSamplerBatch.cpp"],
            # extra_compile_args={'cxx': ['-g'],
            #                     'nvcc': ['-02']}
            include_dirs=['src'],
            extra_compile_args = {"cxx": [],
                                  'nvcc': []}

        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)