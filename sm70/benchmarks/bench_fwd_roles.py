# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# Данные для графика "ядро ролей против прочих путей". Одна обвязка на все пути, частоты
# зафиксированы, усечённое среднее. Вместе со временем ЗАМЕРЯЕТСЯ ТОЧНОСТЬ -- иначе сравнение
# fp16-пути с int8-путём некорректно по построению.
import json, os, sys, time, math, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fa2_sm70
from torch.nn.attention import sdpa_kernel, SDPBackend
D = 256
SHAPES = [(1,4,4,2048,2048),(1,4,4,4096,4096),(1,16,4,4096,4096),(1,4,4,8192,8192),(1,2,2,16384,16384)]
ext = fa2_sm70._ext.prefill_ext()
def bench(fn, it=12):
    for _ in range(5): fn()
    torch.cuda.synchronize(); ts=[]
    for _ in range(it):
        t0=time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter()-t0)
    ts.sort(); return sum(ts[3:-3])/len(ts[3:-3])*1e3
def qz(x):
    a=x.abs().amax(-1,keepdim=True).float()/127.
    q=torch.clamp(torch.round(x.float()/a),-127,127).to(torch.int8)
    return q.contiguous(), a.squeeze(-1).contiguous(), (q.float()*a).half().contiguous()
def ref64(q,k,v,H,Hkv,sc):
    Q=q.transpose(1,2).double(); K=k.repeat_interleave(H//Hkv,1).double(); V=v.repeat_interleave(H//Hkv,1).double()
    s=(Q@K.transpose(-1,-2))*sc; S=s.shape[-2]
    i=torch.arange(S,device=s.device).view(-1,1); j=torch.arange(s.shape[-1],device=s.device).view(1,-1)
    s=s.masked_fill(j>i+(s.shape[-1]-S), float('-inf'))
    return (torch.softmax(s,-1)@V).transpose(1,2)
out=[]
for B,H,Hkv,Sq,Sk in SHAPES:
    torch.manual_seed(Sq); sc=1.0/math.sqrt(D)
    q=(torch.randn(B,Sq,H,D,device='cuda',dtype=torch.float16)*0.3).contiguous()
    k=(torch.randn(B,Hkv,Sk,D,device='cuda',dtype=torch.float16)*0.3).contiguous()
    v=(torch.randn(B,Hkv,Sk,D,device='cuda',dtype=torch.float16)*0.3).contiguous()
    kq,ks,kd=qz(k); vq,vs,vd=qz(v)
    r={"shape":[B,H,Sq,Sk]}
    os.environ["FA2SM70_WS"]="0"; os.environ["FA2SM70_I8_OCC"]="0"
    r["i8_old"]=bench(lambda: ext.attn_fwd_volta_i8(q,kq,ks,vq,vs,sc,True))
    os.environ["FA2SM70_WS"]="1"
    r["i8_roles"]=bench(lambda: ext.attn_fwd_volta_i8(q,kq,ks,vq,vs,sc,True))
    r["fp16"]=bench(lambda: ext.attn_fwd_volta(q,k,v,sc,True))
    try: r["cutlass"]=bench(lambda: ext.attn_fwd_qbshd(q,k,v,sc,True))
    except Exception: r["cutlass"]=None
    if Sq==Sk:
        qb=q.transpose(1,2).contiguous(); kb=k.repeat_interleave(H//Hkv,1); vb=v.repeat_interleave(H//Hkv,1)
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            r["sdpa"]=bench(lambda: torch.nn.functional.scaled_dot_product_attention(qb,kb,vb,is_causal=True))
    else: r["sdpa"]=None
    # ТОЧНОСТЬ против эталона в fp64. Для int8 эталон считается по ДЕКВАНТОВАННЫМ K/V.
    if Sq<=4096:
        g=ref64(q,k,v,H,Hkv,sc); g8=ref64(q,kd,vd,H,Hkv,sc)
        o16,_=ext.attn_fwd_volta(q,k,v,sc,True)
        os.environ["FA2SM70_WS"]="1"; o8,_=ext.attn_fwd_volta_i8(q,kq,ks,vq,vs,sc,True)
        n=lambda a,b: ((a.double()-b).abs().max()/(b.abs().max()+1e-30)).item()
        r["err_fp16_vs_fp64"]=n(o16,g); r["err_i8_vs_deq_fp64"]=n(o8,g8); r["err_i8_vs_fp64"]=n(o8,g)
    out.append(r); print(r, flush=True)
result_path = os.environ.get("RESULTS_PATH")
if result_path:
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"saved: {result_path}")
else:
    print("results kept in stdout; set RESULTS_PATH to save JSON outside the repository")
