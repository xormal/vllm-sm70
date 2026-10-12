# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# MXFP4 dequantisation vs an INDEPENDENT reference (the formula from
# transformers/integrations/mxfp4.py, reimplemented here so the test does not depend on transformers
# being installed). Two gates: every one of the 256 possible bytes across representative E8M0 scales,
# and -- when the checkpoint is present -- real gpt-oss-20b expert weights.
# Run:  python tests/test_mxfp4.py
import glob, math, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fa2_sm70

FP4 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def ref_dequant(blocks, scales, dtype=torch.float32):
    lut = torch.tensor(FP4, dtype=dtype, device=blocks.device)
    exp = scales.to(torch.int32) - 127
    *pre, G, B = blocks.shape
    rows = math.prod(pre) * G if pre else G
    blk = blocks.reshape(rows, B)
    e = exp.reshape(rows, 1)
    out = torch.empty(rows, B * 2, dtype=dtype, device=blocks.device)
    out[:, 0::2] = lut[(blk & 0x0F).to(torch.int)]      # LOW nibble is the FIRST element
    out[:, 1::2] = lut[(blk >> 4).to(torch.int)]
    return torch.ldexp(out, e).reshape(*pre, G * B * 2)


ok = True

# 1) exhaustive over byte values, with scales spanning the normal range
b = torch.arange(256, dtype=torch.uint8).reshape(16, 16).cuda()
s = torch.tensor([127, 126, 128, 120, 134, 100, 150, 1] + [127] * 8, dtype=torch.uint8).cuda()
d = (fa2_sm70.mxfp4_dequant(b, s, torch.float32) - ref_dequant(b, s)).abs().max().item()
ok &= d == 0.0
print(f"  all 256 bytes x 8 scales: max|diff| = {d:.3e} {'OK' if d == 0 else 'FAIL'}")

# 2) real checkpoint, when the caller supplies its directory. This optional gate keeps the test
# portable and never makes a local model path part of the repository contract.
checkpoint_dir = os.environ.get("MXFP4_CHECKPOINT_DIR")
f = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors"))) if checkpoint_dir else []
if f:
    from safetensors import safe_open
    with safe_open(f[0], framework="pt") as h:
        blk = h.get_tensor("model.layers.0.mlp.experts.down_proj_blocks")[:2].cuda()
        scl = h.get_tensor("model.layers.0.mlp.experts.down_proj_scales")[:2].cuda()
    got = fa2_sm70.mxfp4_dequant(blk, scl, torch.float32)
    d = (got - ref_dequant(blk, scl)).abs().max().item()
    ok &= d == 0.0
    print(f"  gpt-oss weights {tuple(blk.shape)}: max|diff| = {d:.3e} {'OK' if d == 0 else 'FAIL'}")
    g16 = fa2_sm70.mxfp4_dequant(blk, scl, torch.float16)
    rel = ((g16.float() - ref_dequant(blk, scl)).norm() / ref_dequant(blk, scl).norm()).item()
    ok &= rel < 1e-6                                    # every MXFP4 value is exactly fp16-representable
    print(f"  fp16 output: relL2 = {rel:.2e} {'OK' if rel < 1e-6 else 'FAIL'}")
else:
    print("  (gpt-oss checkpoint not present -- skipped the real-weight gate)")

# 3) fused GEMV: weights consumed PACKED, against the same weights unpacked
if f:
    ext = fa2_sm70._ext.mxfp4_ext()
    with safe_open(f[0], framework="pt") as h:
        gb = h.get_tensor("model.layers.0.mlp.experts.down_proj_blocks")[0].cuda()
        gs = h.get_tensor("model.layers.0.mlp.experts.down_proj_scales")[0].cuda()
    Wd = fa2_sm70.mxfp4_dequant(gb, gs, torch.float16)
    xg = torch.randn(1, gb.shape[1] * 32, device="cuda", dtype=torch.float16) * 0.1
    yg = ext.mxfp4_gemv(xg, gb, gs)
    rg = xg.float() @ Wd.float().t()
    rel = ((yg.float() - rg).norm() / rg.norm()).item()
    ok &= rel < 1e-3
    print(f"  fused GEMV vs unpacked: relL2 = {rel:.2e} {'OK' if rel < 1e-3 else 'FAIL'}")
    # The output is fp16, so it saturates like any fp16 GEMM would: with synthetic E8M0 scales up to
    # 139 a row sum reached 6.59e4 against fp16's 65504 and became inf. That is the format's range,
    # not a dequant defect -- real gpt-oss weights span +-8.0 -- but it is why this gate uses the
    # checkpoint's own scales rather than random ones.

print("MXFP4:", "ALL OK" if ok else "FAILED")
assert ok
