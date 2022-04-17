#include <torch/serialize/tensor.h>
#include <torch/extension.h>

#include "dot_product/dot_product_kernel.h"
#include "scalar_attention/scalar_attention_kernel.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dot_product_forward", &dot_product_forward, "dot_product_forward");
    m.def("dot_product_backward", &dot_product_backward, "dot_product_backward");
    m.def("scalar_attention_forward", &scalar_attention_forward, "scalar_attention_forward");
    m.def("scalar_attention_backward", &scalar_attention_backward, "scalar_attention_backward");
}