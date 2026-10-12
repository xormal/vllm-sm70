import torch, sys
torch.manual_seed(0)
dev="cuda"
BITS=8; GROUP=128
PF=32//BITS                                   # 4 int8 per int32

def quantize_rtn(W, bits=BITS, group=GROUP):
    """W [in, out] fp16 -> asymmetric per-group RTN, AutoGPTQ layout."""
    inf, outf = W.shape
    assert inf % group == 0
    Wg = W.float().reshape(inf//group, group, outf)
    mx = Wg.amax(1); mn = Wg.amin(1)                       # [ng, out]
    maxq = 2**bits - 1
    scale = ((mx-mn).clamp(min=1e-8)/maxq)
    zero = torch.round(-mn/scale).clamp(0, maxq)
    q = torch.clamp(torch.round(Wg/scale[:,None,:]) + zero[:,None,:], 0, maxq).to(torch.int32)
    return q.reshape(inf, outf), scale.half(), zero.to(torch.int32)

def pack(q, scale, zero, bits=BITS, group=GROUP, zminus1=True):
    inf, outf = q.shape
    qweight = torch.zeros((inf//PF, outf), dtype=torch.int32)
    qc = q.cpu()
    for j in range(PF):
        qweight |= (qc[j::PF] & 0xFF) << (bits*j)
    ng = scale.shape[0]
    z = (zero.cpu() - 1).clamp(min=0) if zminus1 else zero.cpu()
    qzeros = torch.zeros((ng, outf//PF), dtype=torch.int32)
    for j in range(PF):
        qzeros |= (z[:, j::PF] & 0xFF) << (bits*j)
    g_idx = (torch.arange(inf)//group).to(torch.int32)
    return qweight, qzeros, scale.cpu(), g_idx

IN, OUT = 512, 256
W = (torch.randn(IN, OUT)*0.05).half()
q, scale, zero = quantize_rtn(W)
qw, qz, sc, gi = pack(q, scale, zero)
# reference dequant from our own packing convention
deq = ((q.float() - zero.repeat_interleave(GROUP,0).float()) * scale.float().repeat_interleave(GROUP,0))
print("RTN int8 fidelity: relL2 =", ((deq-W.float()).norm()/W.float().norm()).item())

import vllm._C  # noqa
ops = torch.ops._C
x = (torch.randn(8, IN, device=dev)*0.5).half()
y_ref = x.float() @ deq.to(dev)
print(f"{'zero-1':>7} {'shuffle':>8} {'exllama':>8} {'v2':>4} | {'relL2':>10}")
best=None
for zm1 in (True, False):
    for shuf in (False, True):
        for exl in (False, True):
            for v2 in (False, True):
                qw2, qz2, sc2, gi2 = pack(q, scale, zero, zminus1=zm1)
                a,b,c,d2 = qw2.to(dev), qz2.to(dev), sc2.to(dev).half(), gi2.to(dev)
                try:
                    if shuf: ops.gptq_shuffle(a, d2, BITS)
                    y = ops.gptq_gemm(x, a, b, c, d2, exl, v2, BITS)
                    e = ((y.float()-y_ref).norm()/y_ref.norm()).item()
                except Exception as ex:
                    e = float("nan")
                print(f"{str(zm1):>7} {str(shuf):>8} {str(exl):>8} {str(v2):>4} | {e:10.3e}")
                if e == e and (best is None or e < best[0]): best=(e,zm1,shuf,exl,v2)
print("BEST:", best)
print("PACK_DONE")
