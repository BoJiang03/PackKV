//
// Created by tut44803 on 2/28/25.
//
#include <torch/extension.h>
#include <kernels.h>
#include <cpu_compress.h>
#include "fused_kernels.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kq_mat_vec_mul", &kq_mat_vec_mul, "kq matrix-vector multiplication using CUDA");
    m.def("wv_mat_vec_mul", &wv_mat_vec_mul, "wv matrix-vector multiplication using CUDA");
    m.def("k_encode_cpu", &k_encode_cpu_pyi, "k encode using CPU");
    m.def("k_decode_cpu", &k_decode_cpu_pyi, "k decode using CPU");
    m.def("v_encode_cpu", &v_encode_cpu_pyi, "v encode using CPU");
    m.def("v_decode_cpu", &v_decode_cpu_pyi, "v decode using CPU");
    m.def("fused_kq", &fused_kq_launcher, "A fused kernel for KQ computation");
    m.def("fused_wv", &fused_wv_launcher, "A fused kernel for WV computation");
}