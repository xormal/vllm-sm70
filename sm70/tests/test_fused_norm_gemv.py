"""ПРИЁМКА СЛИТОЙ ПАРЫ: RMSNorm + W8A16 GEMV и RMSNorm + gate_up + silu*mul.

ЗАЧЕМ ЭТОТ ФАЙЛ СУЩЕСТВУЕТ. Тело `fused_norm_gemv.cu` полгода числилось «написано и готово к
привязке» и НИ РАЗУ НЕ ИСПОЛНЯЛОСЬ. Первый же запуск (16.08.2026) дал relL2 3.0 при K=512 и 39
при K=5120 -- ответ был умножен на ЧИСЛО РАСЩЕПЛЕНИЙ ПО k: пускач заводил сетку (nblk, ksplit) и
передавал `groups_per_split`, а ядро не читало ни `blockIdx.y`, ни этот параметр, и каждый блок
считал свёртку по всему K, после чего все складывались атомарно.

ФОРМА ОШИБКИ И БЫЛА ДИАГНОЗОМ: она не «шумела», а оказалась ЦЕЛЫМ МНОЖИТЕЛЕМ, растущим ровно как
ksplit (4 при K=512, 40 при K=5120).

ПРАВИЛО: «написано» -- это НЕ «работает». Привязка непроверенного тела вносит дефект, который потом
ищут в другом месте.

Эталон строится ТРЕТЬИМ путём: разворот веса в плотный fp32 + норма в torch. Сверять слитое с
нашим же нес слитым нельзя -- две сверки, читающие один буфер, соглашаются, будучи обе неверны.
"""
import os, sys, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import fa2_sm70
from fa2_sm70.kvant_gptq8 import kvantovat_gptq8, razvernut_gptq8

e = fa2_sm70._ext.fused_norm_gemv_ext()
G, EPS = 128, 1e-6
плохо = 0

def норма(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)) * w.float()

print("=== rmsnorm_w8a16_gemv (норма + наше умножение) ===")
for (K, N) in [(512, 128), (1152, 256), (5120, 1024), (5120, 6144)]:
    torch.manual_seed(K + N)
    w = (torch.randn(N, K, device="cuda") * 0.05).half()
    qw, qz, sc = kvantovat_gptq8(w, G)
    wd = razvernut_gptq8(qw, qz, sc, G)                 # [N,K] fp32, независимый развал
    x = (torch.randn(1, K, device="cuda") * 0.35).half()
    rw = (torch.rand(K, device="cuda") * 0.5 + 0.75).half()
    у_э = норма(x, rw, EPS) @ wd.t()
    у = e.rmsnorm_w8a16_gemv(x, rw, EPS, qw, qz, sc, G, True).float()
    r = ((у - у_э).norm() / у_э.norm()).item()
    ок = r < 3e-3; плохо += 0 if ок else 1
    print(f"  K{K:>5} N{N:>5}  relL2 {r:.2e}" + ("" if ок else "   ПЛОХО"))

print("=== rmsnorm_gateup_silu (норма + gate/up + silu*mul) ===")
for (K, I) in [(512, 256), (5120, 1024), (5120, 8704)]:
    torch.manual_seed(K + I + 7)
    wg = (torch.randn(I, K, device="cuda") * 0.05).half()
    wu = (torch.randn(I, K, device="cuda") * 0.05).half()
    qg, zg, sg = kvantovat_gptq8(wg, G); qu, zu, su = kvantovat_gptq8(wu, G)
    dg, du = razvernut_gptq8(qg, zg, sg, G), razvernut_gptq8(qu, zu, su, G)
    x = (torch.randn(1, K, device="cuda") * 0.35).half()
    rw = (torch.rand(K, device="cuda") * 0.5 + 0.75).half()
    n = норма(x, rw, EPS)
    g, u = n @ dg.t(), n @ du.t()
    у_э = torch.nn.functional.silu(g) * u
    у = e.rmsnorm_gateup_silu(x, rw, EPS, False, qg, zg, sg, qu, zu, su, G, True).float()
    r = ((у - у_э).norm() / у_э.norm()).item()
    ок = r < 5e-3; плохо += 0 if ок else 1
    print(f"  K{K:>5} I{I:>5}  relL2 {r:.2e}" + ("" if ок else "   ПЛОХО"))
print("\nИТОГ:", "СЛИТАЯ ПАРА ВЕРНА" if плохо == 0 else f"ОТКАЗОВ {плохо}")

# ---------------------------------------------------------------------------------------------
# СЛУЧАЙ БОЕВОЙ СЕТКИ: вход fp32 и семантика Gemma
# Норма в Qwen3.5/3.8 -- это `GemmaRMSNorm`, а не обычная RMSNorm, и отличий ДВА:
#   1. множитель `(1 + weight)`, причём вес инициализирован НУЛЯМИ -- передать `weight` напрямую
#      значит занулить норму;
#   2. при fp16 невязка держится В fp32 (`x.float() + residual.float()`, `residual = x`), что
#      подтверждено снятым машинным кодом Triton: три входа, два fp16 и один fp32.
# Оба расхождения молчаливы: ядро не упадёт, а вывод поедет. Поэтому они -- ОТДЕЛЬНЫЙ гейт.
print("=== боевой случай: вход fp32, множитель (1+w) ===")
плохо32 = 0
for (K, I) in [(5120, 1024), (5120, 8704)]:
    torch.manual_seed(K + I + 99)
    wg = (torch.randn(I, K, device="cuda") * 0.05).half()
    wu = (torch.randn(I, K, device="cuda") * 0.05).half()
    qg, zg, sg = kvantovat_gptq8(wg, G); qu, zu, su = kvantovat_gptq8(wu, G)
    dg, du = razvernut_gptq8(qg, zg, sg, G), razvernut_gptq8(qu, zu, su, G)
    x32 = (torch.randn(1, K, device="cuda", dtype=torch.float32) * 0.35)   # невязка Gemma
    wn = (torch.randn(K, device="cuda") * 0.1).half()                      # вес нормы (около нуля)
    # эталон -- ровно `GemmaRMSNorm._forward_static_*`: норма в fp32, множитель (1+w)
    xf = x32.float()
    n = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)) * (1.0 + wn.float())
    у_э = torch.nn.functional.silu(n @ dg.t()) * (n @ du.t())
    # (1+w) теперь СВОРАЧИВАЕТСЯ В ЯДРЕ: снаружи передаётся СЫРОЙ вес нормы.
    у = e.rmsnorm_gateup_silu(x32, wn, EPS, True, qg, zg, sg, qu, zu, su, G, True).float()
    r = ((у - у_э).norm() / у_э.norm()).item()
    ок = r < 5e-3; плохо32 += 0 if ок else 1
    print(f"  K{K:>5} I{I:>5}  relL2 {r:.2e}" + ("" if ок else "   ПЛОХО"))
print("ИТОГ fp32:", "ВЕРНО" if плохо32 == 0 else f"ОТКАЗОВ {плохо32}")
