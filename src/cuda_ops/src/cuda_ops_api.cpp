#include <torch/serialize/tensor.h>
#include <torch/extension.h>

#include "add_sum_squares/add_sum_squares_kernel.h"
#include "direct_dot_product/direct_dot_product_kernel.h"
#include "direct_dot_product_shared/direct_dot_product_shared_kernel.h"
#include "direct_dot_product_shared_with_key/direct_dot_product_shared_with_key_kernel.h"
#include "direct_dot_product_with_key/direct_dot_product_with_key_kernel.h"
#include "dot_product/dot_product_kernel.h"
#include "dot_product_sample/dot_product_sample_kernel.h"
#include "dot_product_sample_shared/dot_product_sample_shared_kernel.h"
#include "dot_product_sample_with_key/dot_product_sample_with_key_kernel.h"
#include "dot_product_intra/dot_product_intra_kernel.h"
#include "dot_product_intra_inter/dot_product_intra_inter_kernel.h"
#include "dot_product_key/dot_product_key_kernel.h"
#include "dot_product_with_key/dot_product_with_key_kernel.h"
#include "decomposed_dot_product_with_key/decomposed_dot_product_with_key_kernel.h"
#include "scalar_attention/scalar_attention_kernel.h"
#include "vector_attention/vector_attention_kernel.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_sum_squares_forward", &add_sum_squares_forward, "add_sum_squares_forward");
    m.def("add_sum_squares_backward", &add_sum_squares_backward, "add_sum_squares_backward");
    m.def("direct_dot_product_forward", &direct_dot_product_forward, "direct_dot_product_forward");
    m.def("direct_dot_product_backward", &direct_dot_product_backward, "direct_dot_product_backward");
    m.def("direct_dot_product_shared_forward", &direct_dot_product_shared_forward, "direct_dot_product_shared_forward");
    m.def("direct_dot_product_shared_backward", &direct_dot_product_shared_backward, "direct_dot_product_shared_backward");
    m.def("direct_dot_product_shared_with_key_forward", &direct_dot_product_shared_with_key_forward, "direct_dot_product_shared_with_key_forward");
    m.def("direct_dot_product_shared_with_key_backward", &direct_dot_product_shared_with_key_backward, "direct_dot_product_shared_with_key_backward");
    m.def("direct_dot_product_with_key_forward", &direct_dot_product_with_key_forward, "direct_dot_product_with_key_forward");
    m.def("direct_dot_product_with_key_backward", &direct_dot_product_with_key_backward, "direct_dot_product_with_key_backward");
    m.def("dot_product_forward", &dot_product_forward, "dot_product_forward");
    m.def("dot_product_backward", &dot_product_backward, "dot_product_backward");
    m.def("dot_product_sample_forward", &dot_product_sample_forward, "dot_product_sample_forward");
    m.def("dot_product_sample_backward", &dot_product_sample_backward, "dot_product_sample_backward");
    m.def("dot_product_sample_shared_forward", &dot_product_sample_shared_forward, "dot_product_sample_shared_forward");
    m.def("dot_product_sample_shared_backward", &dot_product_sample_shared_backward, "dot_product_sample_shared_backward");
    m.def("dot_product_sample_with_key_forward", &dot_product_sample_with_key_forward, "dot_product_sample_with_key_forward");
    m.def("dot_product_sample_with_key_backward", &dot_product_sample_with_key_backward, "dot_product_sample_with_key_backward");
    m.def("dot_product_intra_forward", &dot_product_intra_forward, "dot_product_intra_forward");
    m.def("dot_product_intra_backward", &dot_product_intra_backward, "dot_product_intra_backward");
    m.def("dot_product_intra_inter_forward", &dot_product_intra_inter_forward, "dot_product_intra_inter_forward");
    m.def("dot_product_intra_inter_backward", &dot_product_intra_inter_backward, "dot_product_intra_inter_backward");
    m.def("dot_product_key_forward", &dot_product_key_forward, "dot_product_key_forward");
    m.def("dot_product_key_backward", &dot_product_key_backward, "dot_product_key_backward");
    m.def("dot_product_with_key_forward", &dot_product_with_key_forward, "dot_product_with_key_forward");
    m.def("dot_product_with_key_backward", &dot_product_with_key_backward, "dot_product_with_key_backward");
    m.def("decomposed_dot_product_with_key_forward", &decomposed_dot_product_with_key_forward, "decomposed_dot_product_with_key_forward");
    m.def("decomposed_dot_product_with_key_backward", &decomposed_dot_product_with_key_backward, "decomposed_dot_product_with_key_backward");
    m.def("scalar_attention_forward", &scalar_attention_forward, "scalar_attention_forward");
    m.def("scalar_attention_backward", &scalar_attention_backward, "scalar_attention_backward");
    m.def("vector_attention_forward", &vector_attention_forward, "vector_attention_forward");
    m.def("vector_attention_backward", &vector_attention_backward, "vector_attention_backward");
}