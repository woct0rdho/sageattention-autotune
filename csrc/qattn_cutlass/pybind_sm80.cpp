#include "attn_cutlass_sm80.h"

#include <Python.h>
#include <torch/csrc/stable/library.h>

PyMODINIT_FUNC PyInit__qattn_cutlass_sm80(void)
{
    static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "_qattn_cutlass_sm80",
        NULL,
        -1,
        NULL,
    };
    return PyModule_Create(&module_def);
}

STABLE_TORCH_LIBRARY(sageattention_qattn_cutlass_sm80, m) {
    m.def("qk_int8_sv_f16_accum_f32_attn_cutlass("
            "Tensor query, "
            "Tensor key, "
            "Tensor value, "
            "Tensor(a!) output, "
            "Tensor query_scale, "
            "Tensor key_scale, "
            "int tensor_layout, "
            "float sm_scale, "
            "int blk_q, "
            "int blk_k, "
            "int warp_q, "
            "int warp_k, "
            "bool return_lse"
          ") -> Tensor");
    m.def("qk_int8_sv_f16_accum_f32_attn_bwd_cutlass("
            "Tensor query, "
            "Tensor key, "
            "Tensor query_scale, "
            "Tensor key_scale, "
            "Tensor value, "
            "Tensor output, "
            "Tensor grad_output, "
            "Tensor lse, "
            "Tensor(a!) grad_query, "
            "Tensor(b!) grad_key, "
            "Tensor(c!) grad_value, "
            "int tensor_layout, "
            "float sm_scale, "
            "int blk_q, "
            "int blk_k, "
            "int bwd_block_m, "
            "int bwd_block_n"
          ") -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(sageattention_qattn_cutlass_sm80, CUDA, m) {
    m.impl("qk_int8_sv_f16_accum_f32_attn_cutlass", TORCH_BOX(qk_int8_sv_f16_accum_f32_attn_cutlass));
    m.impl("qk_int8_sv_f16_accum_f32_attn_bwd_cutlass", TORCH_BOX(qk_int8_sv_f16_accum_f32_attn_bwd_cutlass));
}
