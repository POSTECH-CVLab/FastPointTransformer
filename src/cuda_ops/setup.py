from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cuda_sparse_ops',
    version="0.11.0",
    ext_modules=[
        CUDAExtension('cuda_sparse_ops', [
            'src/cuda_ops_api.cpp',
            'src/add_sum_squares/add_sum_squares.cpp',
            'src/add_sum_squares/add_sum_squares_kernel.cu',
            'src/direct_dot_product/direct_dot_product.cpp',
            'src/direct_dot_product/direct_dot_product_kernel.cu',
            'src/direct_dot_product_shared/direct_dot_product_shared.cpp',
            'src/direct_dot_product_shared/direct_dot_product_shared_kernel.cu',
            'src/direct_dot_product_shared_with_key/direct_dot_product_shared_with_key.cpp',
            'src/direct_dot_product_shared_with_key/direct_dot_product_shared_with_key_kernel.cu',
            'src/direct_dot_product_with_key/direct_dot_product_with_key.cpp',
            'src/direct_dot_product_with_key/direct_dot_product_with_key_kernel.cu',
            'src/dot_product/dot_product.cpp',
            'src/dot_product/dot_product_kernel.cu',
            'src/dot_product_sample/dot_product_sample.cpp',
            'src/dot_product_sample/dot_product_sample_kernel.cu',
            'src/dot_product_sample_shared/dot_product_sample_shared.cpp',
            'src/dot_product_sample_shared/dot_product_sample_shared_kernel.cu',
            'src/dot_product_sample_with_key/dot_product_sample_with_key.cpp',
            'src/dot_product_sample_with_key/dot_product_sample_with_key_kernel.cu',
            'src/dot_product_intra/dot_product_intra.cpp',
            'src/dot_product_intra/dot_product_intra_kernel.cu',
            'src/dot_product_intra_inter/dot_product_intra_inter.cpp',
            'src/dot_product_intra_inter/dot_product_intra_inter_kernel.cu',
            'src/dot_product_key/dot_product_key.cpp',
            'src/dot_product_key/dot_product_key_kernel.cu',
            'src/dot_product_with_key/dot_product_with_key.cpp',
            'src/dot_product_with_key/dot_product_with_key_kernel.cu',
            'src/decomposed_dot_product_with_key/decomposed_dot_product_with_key.cpp',
            'src/decomposed_dot_product_with_key/decomposed_dot_product_with_key_kernel.cu',
            'src/scalar_attention/scalar_attention.cpp',
            'src/scalar_attention/scalar_attention_kernel.cu',
            'src/vector_attention/vector_attention.cpp',
            'src/vector_attention/vector_attention_kernel.cu',
        ],
                      extra_compile_args={
                          'cxx': ['-g'],
                          'nvcc': ['-O2']
                      })
    ],
    cmdclass={'build_ext': BuildExtension})
