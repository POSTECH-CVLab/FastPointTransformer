from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cuda_sparse_ops',
    author='Chunghyun Park and Yoonwoo Jeong',
    version="0.1.0",
    ext_modules=[
        CUDAExtension('cuda_sparse_ops', [
            'src/cuda_ops_api.cpp',
            'src/dot_product/dot_product.cpp',
            'src/dot_product/dot_product_kernel.cu',
            'src/scalar_attention/scalar_attention.cpp',
            'src/scalar_attention/scalar_attention_kernel.cu',
        ],
                      extra_compile_args={
                          'cxx': ['-g'],
                          'nvcc': ['-O2']
                      })
    ],
    cmdclass={'build_ext': BuildExtension})
