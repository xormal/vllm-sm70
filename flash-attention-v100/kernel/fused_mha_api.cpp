#include <torch/extension.h>
#include <ATen/ATen.h>
#include <stdexcept>
#include "fused_mha.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashAttention-2 implementation optimized for Volta";
    m.def("fwd", &flash_attention_forward, "FlashAttention-2 Forward Pass (Volta)");
    m.def("bwd", &flash_attention_backward, "FlashAttention-2 Backward Pass (Volta)");
    m.def(
        "decode_paged_fwd",
        &flash_attention_decode_paged,
        "FlashAttention decode over paged KV cache (Volta)");
    m.def(
        "prefill_paged_fwd",
        &flash_attention_prefill_paged,
        "FlashAttention prefill over paged KV cache (Volta)");
    m.def(
        "reshape_and_cache_fp8",
        &reshape_and_cache_fp8_v100,
        "Quantize+write K/V into paged fp8 KV cache (e4m3/e5m2, Volta)");
}
