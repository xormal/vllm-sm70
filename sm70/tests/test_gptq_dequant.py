# Сверка деквантования весов GPTQ int8 с НЕЗАВИСИМЫМ эталоном на torch.
# Ошибка здесь портит каждый линейный слой и не вызывает ни одного сбоя — только качество ответов.
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fa2_sm70
ext = fa2_sm70._ext.decode_ext(); dev = "cuda"; torch.manual_seed(0)

for (K, N, group) in [(256, 512, 128), (512, 256, 64), (1024, 512, 128)]:
    qw = torch.randint(0, 1 << 30, (K // 4, N), device=dev, dtype=torch.int32)
    G = K // group
    qz = torch.randint(0, 1 << 30, (G, N // 4), device=dev, dtype=torch.int32)
    sc = (torch.rand(G, N, device=dev, dtype=torch.float16) * 0.02 + 0.001)
    out = torch.empty(K, N, device=dev, dtype=torch.float16)
    ext.gptq_dequant_w8(qw, qz, sc, out)

    k = torch.arange(K, device=dev)
    n = torch.arange(N, device=dev)
    g = (k // group).view(-1, 1).expand(K, N)
    sh = (8 * (k & 3)).view(-1, 1)
    q = ((qw[(k >> 2)] >> sh) & 0xFF).long()                       # вес: 4 по 8 бит в int32
    zw = qz[g, (n >> 2).view(1, -1).expand(K, N)]
    z = ((zw >> (8 * (n & 3)).view(1, -1)) & 0xFF).long() + 1      # GPTQ v1: при упаковке хранится zero-1
    s = sc[g, n.view(1, -1).expand(K, N)].float()
    ref = ((q - z).float() * s).half()

    d = (out.float() - ref.float())
    nz = int((d != 0).sum())
    rel = (d.norm() / max(ref.float().norm().item(), 1e-9)).item()
    print(f"K={K:>5} N={N:>4} group={group:>4}: relL2 {rel:.3e}, отличается {nz}/{d.numel()} ({100.0*nz/d.numel():.4f}%)")
    if nz:
        i = (d != 0).nonzero()[0].tolist()
        print(f"   пример [{i[0]},{i[1]}]: наше {out[i[0],i[1]].item():.8g}, эталон {ref[i[0],i[1]].item():.8g}")
