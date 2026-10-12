// Custom fp8 KV-cache write for SM70 / Volta.
//
// vLLM's default KV write goes through a Triton kernel
// (reshape_and_cache_kernel_flash), but Triton on SM70 cannot compile e4m3fn
// (fp8e4nv) stores ("type fp8e4nv not supported in this architecture"), so the
// fork is stuck on e5m2 (2-bit mantissa). This CUDA kernel quantizes K/V to fp8
// (e4m3fn or e5m2) using CUDA's arch-agnostic __nv_cvt_float_to_fp8 (software
// bit-conversion, valid on SM70 — the limitation was Triton's codegen, not
// CUDA), unblocking e4m3 (3-bit mantissa) on Volta. Matches the paged layout
// the decode kernel reads, and the vLLM scale convention (store = value/scale;
// decode multiplies back by scale).

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <string>

namespace {

template <int E>  // 0 = e4m3fn, 1 = e5m2
__global__ void reshape_and_cache_fp8_kernel(
    const __half* __restrict__ key,    // [num_tokens, num_heads, head_size] (contig)
    const __half* __restrict__ value,
    uint8_t* __restrict__ key_cache,   // [num_blocks, block_size, num_heads, head_size]
    uint8_t* __restrict__ value_cache,
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int num_tokens,
    const int hd,                      // num_heads * head_size
    const int head_size,
    const int block_size,
    const int64_t key_token_stride,
    const int64_t kc_block_stride,
    const int64_t kc_token_stride,
    const int64_t kc_head_stride,
    const float inv_k_scale,
    const float inv_v_scale) {
  const int64_t token = blockIdx.x;
  if (token >= num_tokens) return;
  const int64_t slot = slot_mapping[token];
  if (slot < 0) return;  // padding slot — skip

  const int64_t phys_block = slot / block_size;
  const int64_t block_off = slot % block_size;
  const int64_t key_base = token * key_token_stride;
  const int64_t cache_base = phys_block * kc_block_stride + block_off * kc_token_stride;

  for (int i = threadIdx.x; i < hd; i += blockDim.x) {
    const int head = i / head_size;
    const int d = i % head_size;
    const int64_t cidx = cache_base + static_cast<int64_t>(head) * kc_head_stride + d;

    const float kf = __half2float(key[key_base + i]) * inv_k_scale;
    const float vf = __half2float(value[key_base + i]) * inv_v_scale;

    if constexpr (E == 0) {
      key_cache[cidx]   = __nv_cvt_float_to_fp8(kf, __NV_SATFINITE, __NV_E4M3);
      value_cache[cidx] = __nv_cvt_float_to_fp8(vf, __NV_SATFINITE, __NV_E4M3);
    } else {
      key_cache[cidx]   = __nv_cvt_float_to_fp8(kf, __NV_SATFINITE, __NV_E5M2);
      value_cache[cidx] = __nv_cvt_float_to_fp8(vf, __NV_SATFINITE, __NV_E5M2);
    }
  }
}

}  // namespace

void reshape_and_cache_fp8_v100(
    const at::Tensor& key,         // [num_tokens, num_heads, head_size] fp16
    const at::Tensor& value,
    at::Tensor& key_cache,         // [num_blocks, block_size, num_heads, head_size] uint8
    at::Tensor& value_cache,
    const at::Tensor& slot_mapping,// [num_tokens] int64
    const std::string& kv_cache_dtype,
    const double k_scale,
    const double v_scale) {
  TORCH_CHECK(key.is_cuda() && value.is_cuda(), "key/value must be CUDA");
  TORCH_CHECK(key.dtype() == torch::kFloat16 && value.dtype() == torch::kFloat16,
              "key/value must be fp16");
  TORCH_CHECK(key_cache.dtype() == torch::kUInt8 && value_cache.dtype() == torch::kUInt8,
              "fp8 caches must be uint8");
  TORCH_CHECK(key.dim() == 3 && value.dim() == 3, "key/value must be [T, H, D]");
  TORCH_CHECK(key_cache.dim() == 4 && value_cache.dim() == 4,
              "caches must be [num_blocks, block_size, H, D]");
  TORCH_CHECK(key_cache.stride(-1) == 1 && value_cache.stride(-1) == 1,
              "cache last dim must be contiguous");

  int E;
  if (kv_cache_dtype == "fp8_e5m2") E = 1;
  else if (kv_cache_dtype == "fp8" || kv_cache_dtype == "fp8_e4m3") E = 0;
  else { TORCH_CHECK(false, "reshape_and_cache_fp8_v100: bad dtype ", kv_cache_dtype); return; }

  const int num_tokens = key.size(0);
  const int num_heads = key.size(1);
  const int head_size = key.size(2);
  const int hd = num_heads * head_size;
  const int block_size = key_cache.size(1);
  TORCH_CHECK(key_cache.size(2) == num_heads && key_cache.size(3) == head_size,
              "cache head dims mismatch");

  if (num_tokens == 0) return;

  auto key_c = key.contiguous();
  auto val_c = value.contiguous();
  auto slot_i64 = slot_mapping.to(torch::kInt64).contiguous();

  const float inv_k = static_cast<float>(k_scale > 0.0 ? 1.0 / k_scale : 1.0);
  const float inv_v = static_cast<float>(v_scale > 0.0 ? 1.0 / v_scale : 1.0);

  const dim3 grid(num_tokens);
  const int block = hd < 256 ? hd : 256;

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  c10::cuda::CUDAGuard device_guard(key.device());

  auto launch = [&](auto e_tag) {
    constexpr int EE = decltype(e_tag)::value;
    reshape_and_cache_fp8_kernel<EE><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(key_c.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(val_c.data_ptr<at::Half>()),
        key_cache.data_ptr<uint8_t>(),
        value_cache.data_ptr<uint8_t>(),
        slot_i64.data_ptr<int64_t>(),
        num_tokens, hd, head_size, block_size,
        key_c.stride(0),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        inv_k, inv_v);
  };
  if (E == 0) launch(std::integral_constant<int, 0>{});
  else        launch(std::integral_constant<int, 1>{});

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
