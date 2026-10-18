"""Гейт родной головы Clef (fa2sm70_clef: tap кусками + head_run) против эталона в fp32.

Эталон -- модули torch той же структуры, что JointSchemaHead из релиза Clef (код повторён, загруженный
файл не исполняется), веса те же, счёт в fp32 по fp16-входу. Родной путь получает скрытые кусками по
2048 строк, как в шаге префилла, и считает голову в конце.
Печатает расхождение логитов и вероятностей по вопросам, время отвода и головы, память процесса.
"""
import argparse, math, os, sys, time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
sys.path.insert(0, HERE)
from fa2_sm70._ext import clef_ext  # noqa: E402
import clef_golova  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--golova", default=os.path.join(os.environ.get("FA2SM70_CLEF_DIR", "clef"), "joint_head_fp16.safetensors"))
ap.add_argument("--N", type=int, default=3000)
ap.add_argument("--voprosy", type=int, default=8)
ap.add_argument("--kusok", type=int, default=2048)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--h-scale", type=float, default=3.0)
args = ap.parse_args()
dev = torch.device("cuda")
torch.manual_seed(args.seed)

import pynvml  # noqa: E402
pynvml.nvmlInit()
_nv = pynvml.nvmlDeviceGetHandleByUUID("GPU-" + str(torch.cuda.get_device_properties(0).uuid))


def used():
    torch.cuda.synchronize()
    return next(p.usedGpuMemory for p in pynvml.nvmlDeviceGetComputeRunningProcesses(_nv) if p.pid == os.getpid())


ext = clef_ext()
torch.zeros(1, device=dev)
ext.init_delta(1, 2048, 264)
ext.finalize_delta()
u0 = used()
W = clef_golova.zagruzit(ext, args.golova, Nmax=max(16384, args.N))   # иначе N > 16384 не помещается
u1 = used()

# ---------------------------------------------------------------- запрос
N, Qn = args.N, args.voprosy
counts = [int(x) for x in torch.randint(2, 10, (Qn,))]
Qo = sum(counts)
types = torch.randint(0, 3, (Qn,), dtype=torch.int32)
spans_q, spans_o = [], []
pos = N - 30 * (Qn + Qo) - 5
assert pos > 0
for i in range(Qn):
    L = int(torch.randint(5, 25, (1,)))
    spans_q.append((pos, pos + L)); pos += L + 3
    for j in range(counts[i]):
        L = int(torch.randint(1, 20, (1,)))
        spans_o.append((pos, pos + L)); pos += L + 2
assert pos < N
spans = spans_q + spans_o
spanid = torch.full((N,), -1, dtype=torch.int32)
for s, (a, b) in enumerate(spans):
    spanid[a:b] = s
inv_len = torch.tensor([1.0 / (b - a) for a, b in spans], dtype=torch.float32)
qidx = torch.tensor(sum([[i] * c for i, c in enumerate(counts)], []), dtype=torch.int32)
qoff = torch.tensor([0] + list(torch.tensor(counts).cumsum(0)), dtype=torch.int32)
h = (torch.randn(N, 5120, device=dev) * args.h_scale).half()
lex = (torch.randn(Qo, 5120) * 0.03).half()

# ---------------------------------------------------------------- родной путь
ext.req_begin(0, N, spanid, inv_len, qidx, qoff, types, lex)
torch.cuda.synchronize()
t0 = time.time()
for p0 in range(0, N, args.kusok):
    rows = min(args.kusok, N - p0)
    fin = int(p0 + rows == N)
    ext.set_step(torch.tensor([[0, rows, 0, p0, fin]], dtype=torch.int32))
    hk = h[p0:p0 + rows].contiguous()
    if fin:
        torch.cuda.synchronize(); t1 = time.time()
    ext.tap(hk)
torch.cuda.synchronize()
t2 = time.time()
nat = ext.result(0).clone()
u2 = used()
assert ext.ready(0), ext.last_error()
# повтор для времени головы без первого раза
ext.req_begin(1, N, spanid, inv_len, qidx, qoff, types, lex)
for p0 in range(0, N, args.kusok):
    rows = min(args.kusok, N - p0)
    fin = int(p0 + rows == N)
    ext.set_step(torch.tensor([[0, rows, 1, p0, fin]], dtype=torch.int32))
    hk = h[p0:p0 + rows].contiguous()
    if fin:
        torch.cuda.synchronize(); t3 = time.time()
    ext.tap(hk)
torch.cuda.synchronize()
t4 = time.time()
nat2 = ext.result(1).clone()
u3 = used()

# ---------------------------------------------------------------- эталон fp32 (структура JointSchemaHead)
class Evidence(torch.nn.Module):
    def __init__(s):
        super().__init__()
        s.query_norm = torch.nn.LayerNorm(1024); s.memory_norm = torch.nn.LayerNorm(1024)
        s.attention = torch.nn.MultiheadAttention(1024, 16, batch_first=True)
        s.feedforward_norm = torch.nn.LayerNorm(1024)
        s.feedforward = torch.nn.Sequential(torch.nn.Linear(1024, 4096), torch.nn.GELU(), torch.nn.Dropout(0.0),
                                            torch.nn.Linear(4096, 1024), torch.nn.Dropout(0.0))

    def forward(s, q, mem):
        r, _ = s.attention(s.query_norm(q), s.memory_norm(mem), s.memory_norm(mem), need_weights=False)
        q = q + r
        return q + s.feedforward(s.feedforward_norm(q))


class Head(torch.nn.Module):
    def __init__(s):
        super().__init__()
        s.hidden_norm = torch.nn.LayerNorm(5120)
        for n in ("memory", "question", "option_question", "global", "option_context", "option_lexical"):
            setattr(s, f"{n}_projection", torch.nn.Linear(5120, 1024, bias=False))
        s.type_embedding = torch.nn.Embedding(3, 1024)
        s.evidence_layers = torch.nn.ModuleList([Evidence() for _ in range(2)])
        s.option_summary_norm = torch.nn.LayerNorm(1024)
        s.layers = torch.nn.ModuleList([torch.nn.TransformerDecoderLayer(1024, 16, 4096, 0.0, "gelu", batch_first=True,
                                                                         norm_first=True) for _ in range(4)])
        s.field_norm = torch.nn.LayerNorm(1024); s.option_norm = torch.nn.LayerNorm(1024)
        s.residual_scorer = torch.nn.Sequential(torch.nn.Linear(4096, 1024), torch.nn.GELU(), torch.nn.Dropout(0.0),
                                                torch.nn.Linear(1024, 1))
        s.prior_logit_scale = torch.nn.Parameter(torch.zeros(()))
        s.joint_logit_scale = torch.nn.Parameter(torch.zeros(()))
        s.residual_gate = torch.nn.Parameter(torch.zeros(()))


ref = Head().to(dev).float().eval()
miss = ref.load_state_dict({k: v.float() for k, v in W.items()}, strict=True)
with torch.no_grad():
    nh = ref.hidden_norm(h.float())
    mem = ref.memory_projection(nh).unsqueeze(0)
    glob = nh[-1]
    qv = torch.stack([nh[a:b].mean(0) for a, b in spans_q])
    ctx = torch.stack([nh[a:b].mean(0) for a, b in spans_o])
    lx = lex.to(dev).float()
    oq = ref.option_context_projection(ctx) + ref.option_lexical_projection(lx) + \
        ref.option_question_projection(qv)[qidx.long().to(dev)]
    x = oq.unsqueeze(0)
    for layer in ref.evidence_layers:
        x = layer(x, mem)
    routed = x[0]
    split = list(torch.split(routed, counts))
    bf = ref.question_projection(qv)
    summ = []
    for fld, opts in zip(bf, split):
        w = torch.softmax(opts @ fld / math.sqrt(1024), 0)
        summ.append((w.unsqueeze(-1) * opts).sum(0))
    fields = bf + ref.option_summary_norm(torch.stack(summ)) + ref.global_projection(glob).unsqueeze(0) + \
        ref.type_embedding(types.long().to(dev))
    fields = fields.unsqueeze(0)
    for layer in ref.layers:
        fields = layer(fields, mem)
    fields = ref.field_norm(fields[0])
    out, lo = [], 0
    for i, (fld, opts) in enumerate(zip(fields, split)):
        l = lx[lo:lo + counts[i]]; lo += counts[i]
        anchor = F.normalize(qv[i] + glob, dim=-1)
        prior = ref.prior_logit_scale.clamp(max=math.log(100.0)).exp() * (F.normalize(l, dim=-1) @ anchor)
        o = ref.option_norm(opts)
        rf = fld.unsqueeze(0).expand_as(o)
        cos = F.cosine_similarity(rf, o, dim=-1)
        feat = torch.cat([rf, o, rf * o, (rf - o).abs()], -1)
        res = ref.residual_scorer(feat).squeeze(-1)
        joint = ref.joint_logit_scale.clamp(max=math.log(100.0)).exp() * cos + res
        out.append(prior + torch.sigmoid(ref.residual_gate) * joint)
    etal = torch.cat(out)

d = (nat - etal).abs()
pn = torch.cat([torch.softmax(t, 0) for t in torch.split(nat, counts)])
pe = torch.cat([torch.softmax(t, 0) for t in torch.split(etal, counts)])
arg_ok = sum(int(torch.argmax(a) == torch.argmax(b)) for a, b in zip(torch.split(nat, counts), torch.split(etal, counts)))
print(f"N={N} Qn={Qn} Qo={Qo}: |logit| эталона до {etal.abs().max():.3f}; max|Δlogit| {d.max():.2e}, "
      f"средн {d.mean():.2e}; max|Δp| {(pn - pe).abs().max():.2e}; argmax совпал {arg_ok}/{Qn}; "
      f"повтор побайтово {torch.equal(nat, nat2)}")
print(f"время: отвод {(t1 - t0) * 1e3:.1f} мс за {N // args.kusok} полных кусков, последний кусок+голова "
      f"{(t2 - t1) * 1e3:.1f} мс (повтор {(t4 - t3) * 1e3:.1f} мс)")
print(f"память процесса: голова при загрузке {(u1 - u0) / 1e6:.1f} МБ, после первого запроса {(u2 - u0) / 1e6:.1f} МБ, "
      f"после второго {(u3 - u0) / 1e6:.1f} МБ")
print("stats:", ext.stats(), "ошибка:", ext.last_error() or "-")
ok = d.max() < 2e-2 and (pn - pe).abs().max() < 5e-3 and arg_ok == Qn and ext.stats()["err"] == 0
print("ИТОГ:", "ПРОЙДЕН" if ok else "НЕ ПРОЙДЕН")
