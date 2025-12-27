#ifndef PACKKV_FUSED_KQ_H
#define PACKKV_FUSED_KQ_H

#include <torch/extension.h>

float fused_kq_launcher(
    torch::Tensor& our_kq,
    const torch::Tensor& q,
    const torch::Tensor& k_quant_zero,
    const torch::Tensor& k_quant_scale,
    const torch::Tensor& t
);

float fused_wv_launcher(
    torch::Tensor& our_wv,
    const torch::Tensor& w,
    const torch::Tensor& v_quant_zero,
    const torch::Tensor& v_quant_scale,
    const torch::Tensor& term1
);

#endif //PACKKV_FUSED_KQ_H 