#include "attn_cuda_sm89.h"

#include <Python.h>
#include <torch/csrc/stable/library.h>

/* Creates a dummy empty _qattn_sm89 module that can be imported from Python.
   The import from Python will load the .so consisting of this file in this
   extension, so that the TORCH_LIBRARY static initializers below are run. */
PyMODINIT_FUNC PyInit__qattn_sm89(void)
{
    static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "_qattn_sm89",  /* name of module */
        NULL,           /* module documentation, may be NULL */
        -1,             /* size of per-interpreter state of the module,
                           or -1 if the module keeps state in global variables. */
        NULL,           /* methods */
    };
    return PyModule_Create(&module_def);
}

// Defines the operators
STABLE_TORCH_LIBRARY(sageattention_qattn_sm89, m) {
    m.def("qk_int8_sv_f8_accum_f32_fuse_v_scale_attn("
            "Tensor query, "
            "Tensor key, "
            "Tensor value, "
            "Tensor(a!) output, "
            "Tensor query_scale, "
            "Tensor key_scale, "
            "Tensor value_scale, "
            "int tensor_layout, "
            "bool is_causal, "
            "float sm_scale, "
            "int blk_q, "
            "int blk_k, "
            "int warp_q, "
            "int warp_k, "
            "bool return_lse"
          ") -> Tensor");
    m.def("qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf("
            "Tensor query, "
            "Tensor key, "
            "Tensor value, "
            "Tensor(a!) output, "
            "Tensor query_scale, "
            "Tensor key_scale, "
            "Tensor value_scale, "
            "int tensor_layout, "
            "bool is_causal, "
            "float sm_scale, "
            "int blk_q, "
            "int blk_k, "
            "int warp_q, "
            "int warp_k, "
            "bool return_lse"
          ") -> Tensor");
}

// Registers CUDA implementations
STABLE_TORCH_LIBRARY_IMPL(sageattention_qattn_sm89, CUDA, m) {
    m.impl("qk_int8_sv_f8_accum_f32_fuse_v_scale_attn", TORCH_BOX(qk_int8_sv_f8_accum_f32_fuse_v_scale_attn));
    m.impl("qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf", TORCH_BOX(qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf));
}
