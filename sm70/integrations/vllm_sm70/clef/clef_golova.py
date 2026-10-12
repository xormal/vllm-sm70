"""Загрузка головы Clef в расширение fa2sm70_clef: ступени весов в закреплённой памяти, постоянная часть
на карте. Только время загрузки."""
import torch
from safetensors import safe_open

STUPENI = [
    ["option_context_projection.", "option_lexical_projection.", "option_question_projection."],
    ["evidence_layers.0."], ["evidence_layers.1."],
    ["question_projection.", "global_projection.", "option_summary_norm.", "type_embedding."],
    ["layers.0."], ["layers.1."], ["layers.2."], ["layers.3."],
    ["field_norm.", "option_norm.", "residual_scorer."],
]
POSTOYANNO = ("hidden_norm.weight", "hidden_norm.bias", "memory_projection.weight")
SKALYARY = ("prior_logit_scale", "joint_logit_scale", "residual_gate")


def zagruzit(ext, path, R=2, Nmax=16384, Smax=512, Qomax=448, Qnmax=64, chunk=1024, device="cuda"):
    f = safe_open(path, "pt")
    W = {k: f.get_tensor(k) for k in f.keys()}
    widx, host = {}, []
    for st, pref in enumerate(STUPENI):
        names = [k for k in sorted(W) if any(k.startswith(p) for p in pref)]
        off, chunks = 0, []
        for k in names:
            t = W[k].to(torch.float16).reshape(-1)
            widx[k] = (st, off, t.numel())
            chunks.append((off, t))
            off += (t.numel() + 127) // 128 * 128
        buf = torch.zeros(off, dtype=torch.float16).pin_memory()
        for o, t in chunks:
            buf[o:o + t.numel()].copy_(t)
        host.append(buf)
    vse = set(widx) | set(POSTOYANNO) | set(SKALYARY)
    lishnie = set(W) - vse
    assert not lishnie, f"веса головы вне ступеней: {sorted(lishnie)[:5]}"
    hn_w, hn_b, mp_w = (W[k].to(device, torch.float16).contiguous() for k in POSTOYANNO)
    ps, js, g = (float(W[k].float().item()) for k in SKALYARY)
    ext.init_head(host, widx, hn_w, hn_b, mp_w, ps, js, g, R, Nmax, Smax, Qomax, Qnmax, chunk)
    return W
