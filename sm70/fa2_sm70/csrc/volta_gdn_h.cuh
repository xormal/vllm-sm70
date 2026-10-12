// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
//
// МЕЖЧАНКОВОЕ СОСТОЯНИЕ ЧАНКОВОГО GATED-DELTA-NET НА ТЕНЗОРНЫХ ЯДРАХ VOLTA.
// Замена chunk_gated_delta_rule_fwd_kernel_h_blockdim64 из fla/Triton (chunk_delta_h.py).
//
// ЗАЧЕМ. Замерено (§4qq библиотеки примитивов): Triton при цели sm_70 НЕ ЭМИТИТ тензорных инструкций
// вообще -- в PTX mma.sync 0, wmma 0, hmma 0 и 1024 fma.rn.f32, то есть tl.dot компилируется в цикл
// FP32-FMA на обычных ядрах. Потолок 3.5 TFLOP/s против 83 у cuBLAS. Этот шаг идёт на 3.46 TFLOP/s --
// ровно на том потолке -- и составляет 33.5% скана. Настраивать там нечего, помогает переписывание.
//
// ЧТО СЧИТАЕТСЯ (ровно как в трионовском ядре; чанк BT=64, состояние H_c лежит как [v][k]):
//     (a) h[c]      = fp16(H_c)                                    -- состояние на НАЧАЛО чанка
//     (b) y         = u - w x fp16(H_c)^T                           -- GEMM1, [BT,DK]x[DK,BV]
//     (c) v_new     = fp16(y)                                       -- БЕЗ гейта
//     (d) vt        = fp16(y * exp(gL - g_t))                       -- гейт, ВТОРОЕ округление
//     (e) H_{c+1}   = exp(gL) * H_c + k^T x vt                      -- GEMM2, [BV,BT]x[BT,DK]
// Ловушка порядка, из-за которой нельзя сэкономить одно округление: v_new пишется ДО умножения на
// гейт, значит fp16(y) и fp16(y*gamma) -- ДВА РАЗНЫХ округления одного fp32-значения, а не одно.
// Вторая: округление H_c -> fp16 в (a) и в (b) -- ОДНО И ТО ЖЕ, поэтому состояние конвертируется
// один раз в разделяемую память и оттуда и пишется в h, и подаётся операндом B в GEMM1.
//
// РАСКЛАДКА РАБОТЫ И ПОЧЕМУ ТАК. По чанкам рекуррентность СТРОГО последовательна (245 итераций
// зависимой цепочки), параллелизм есть только по (последовательность, голова значения, срез DV).
// На боевой форме голов значения 24 -- без дробления по DV это 24 блока на 80 SM, то есть 30%
// машины. Отсюда параметр BV: блок владеет срезом строк [i_v*BV, i_v*BV+BV) измерения V и ВСЕМ K.
// Дробление НЕ добавляет ни одного MAC (каждый срез считает BT*DK*BV, сумма даёт BT*DK*V), но
// линейно множит чтение w и k. Расчётный оптимум BV=64 (48 блоков, пол по памяти 0.92 мс против
// пола по счёту 0.66-0.94) -- порог берётся ЗАМЕРОМ, шаблон допускает 128/64/32.
//
// =================================================================================================
// ГЛУБИНА ПРОВЕРКИ: NT=245 (БОЕВАЯ), И ПОЧЕМУ РАНЬШЕ СТЕНД "ЛОМАЛСЯ" НА NT=48
// =================================================================================================
// Ядро долго не проверялось дальше NT=40: на NT>=48 стенд выдавал inf. Это был дефект СТЕНДА, а не
// ядра, и разбор стоит запомнить целиком, потому что ошибка была в ПОСТАНОВКЕ.
// Стенд брал w и u НЕЗАВИСИМЫМИ случайными числами. Но в этой рекуррентности
//     H <- H*e^{gL} + k^T (gamma * (u - w H^T))
// скобка есть НЕВЯЗКА дельта-правила, и она мала ТОЛЬКО когда w и u согласованы (w = A(beta e^g k),
// u = A(beta v) с общей A). При независимых w,u член -w H^T становится обратной связью со случайным
// знаком, модуль состояния растёт геометрически (замерено x1.35 за чанк) и около 38-го чанка
// переваливает верхнюю границу fp16 (65504). То есть стенд подавал вход, КОТОРОГО ЗАДАЧА НЕ
// ПОРОЖДАЕТ, и мерил переполнение.
// Проверить это можно было счётом, не запуская ядро: у настоящего конвейера fla при g=0 и T=7936
// максимум модуля состояния 0.69 -- на пять порядков ниже границы формата.
// Стенд починен (tools/volta_gdn_h_test.cu::make_wu считает w,u ТЕМ ЖЕ выражением, что и бой,
// обращая (I+A) точной прямой подстановкой в fp64). На боевой форме T=15680, H=24, HG=8, NT=245:
//     h relL2 2.27e-04, v_new 1.04e-04, ht 1.54e-04   (класс fp16)
//     чанк 0 -- РОВНО 0 (h[0] = fp16(h0) по построению у обеих сторон)
//     макс. прирост relL2 за чанк 1.76e-05 при пределе одного округления fp16 2.82e-04
//     max|h| ПО ВСЕМ 245 чанкам = 0.24  -- состояние ОГРАНИЧЕНО, запаса до формата ~5 порядков
// Третий инвариант (диапазон состояния) добавлен в стенд именно затем, чтобы этот класс ошибки
// больше не читался как отказ ядра.
//
// ОРИЕНТАЦИЯ ОПЕРАНДОВ -- сверена по enum в volta_warp_mma.h, где имена ПРОТИВОПОЛОЖНЫ интуиции:
//   kKMajor: B это [N][K], непрерывен K -> инструкция .row.col (случай Q*K^T)
//   kNMajor: B это [K][N], непрерывен N -> инструкция .row.row (случай P*V)
//   kMMajor: A это [K][M], непрерывен M -> инструкция .col.*
// GEMM1 y[t][v] = SUM_k w[t][k]*Hhat[v][k]: A=w это [M=t][K=k] -> kKMajor; B=Hhat это [N=v][K=k],
//       k непрерывен -> kKMajor. Обе стороны k-мажорны.
// GEMM2 H[v][k] += SUM_t vt[t][v]*k[t][k]: аккумулятор держим как [v][k] -- ТА ЖЕ раскладка, что
//       нужна и для записи h, и для операнда GEMM1, то есть перекладок состояния нет вовсе. Тогда
//       A=vt лежит как [K=t][M=v] с непрерывным v -> kMMajor; B=k лежит как [K=t][N=k] с непрерывным
//       k -> kNMajor. Инструкция .col.row.
//
// ДОПОЛНЕНИЕ СТРОК СЧИТАНО ПО БАНКАМ, А НЕ ВЗЯТО НАУГАД (§4c примитивов, §3 плейбука). У каждого
// буфера СВОЙ шаг, потому что раскладка операнда задаёт РАЗНЫЙ закон конфликтов:
//   * k-мажорный операнд (sW, sHh): полоса берёт СТРОКУ, 16 разных строк на инструкцию. Нужен шаг
//     LD полуслов с LD/2 = 2u, u нечётное -> LD = DK+4 = 132 (66 слов, 66 = 2*33). Бесконфликтно.
//     Ценой: 132*2 = 264 Б не кратно 16, широкая ЗАПИСЬ в такой буфер незаконна -- кладём двумя
//     uint2. Это дешёвый размен: чтений фрагментов на порядок больше, чем записей перекладки.
//   * m/n-мажорный операнд (sVt, sK): полоса берёт 4 СМЕЖНЫХ элемента строки k0+(lane&3), то есть
//     адрес складывается как (lane&3)*LD/2 + base/2, и ОБА слагаемых пробегают по 4 значения. При
//     шаге, кратном 8 словам, суммы накладываются (замерено в §4c: 16 адресов на ~7 банков). Нужно
//     (LD/2) mod 32 == 8, то есть LD = 16 (mod 64): LDK = DK+16 = 144, LDT = 80 при BV<=64.
//     Заодно 144*2 = 288 кратно 16 -- в sK широкая запись ЗАКОННА и применяется.
//   * sU читается в эпилоге по карте АККУМУЛЯТОРА (строки из бит {0,3,4} полосы, столбцы из {1,2}),
//     там свой закон: LDU = BV+4 разводит все 32 полосы по разным банкам.
//
// ПОЧЕМУ v_new ПИШЕТСЯ ЧЕРЕЗ sU. Значение y живёт в РЕГИСТРАХ аккумулятора, и прямая запись в
// глобальную дала бы STG.U16 по рассеянным адресам -- ровно тот дефект, что найден разбором SASS в
// §4kk-девятое. Но каждую ячейку (t,v) держит РОВНО ОДНА полоса (свойство покрытия карты), и она же
// читает из sU значение u для этой ячейки. Значит fp16(y) можно записать ПОВЕРХ прочитанного u --
// тот же адрес, тот же поток, гонки нет по построению, -- а потом слить sU в глобальную сплошным
// коалесцированным проходом. Лишнего буфера это не стоит.
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "volta_warp_mma.h"

namespace fa2_sm70 {
namespace gdn {

// Шаг строки буфера vt: должен быть >= BV и давать (LD/2) mod 32 == 8 (см. шапку).
__host__ __device__ constexpr int gdn_h_ldt(int bv) { return ((bv - 16 + 63) / 64) * 64 + 16; }

// Размер разделяемой памяти -- считается и хостом (для cudaFuncSetAttribute), и ядром.
template <int BT, int DK, int BV>
constexpr size_t gdn_h_smem() {
  return (size_t)(BT * (DK + 4) + BV * (DK + 4) + BT * (DK + 16) + BT * gdn_h_ldt(BV) +
                  BT * (BV + 4)) * sizeof(__half);
}

// BT -- размер чанка (64); DK/DV -- размерности головы; BV -- срез по V, которым владеет блок;
// WM1 x WN1 -- сетка варпов на выходе GEMM1 [BT][BV]; WM2 x WN2 -- на состоянии [BV][DK];
// PIPE -- разнести выдачу LDG и запись STS окна чанка (см. ниже).
template <int BT, int DK, int DV, int BV, int WM1, int WN1, int WM2, int WN2, bool PIPE = true>
__global__ __launch_bounds__(WM1 * WN1 * 32, 1) void volta_gdn_chunk_h(
    const __half* __restrict__ kg,     // k   [T, HG, DK]      -- голова КЛЮЧА
    const __half* __restrict__ wg,     // w   [T, H,  DK]
    const __half* __restrict__ ug,     // u   [T, H,  DV]
    const float*  __restrict__ gg,     // g   [T, H]           -- накопленный логарифм гейта, fp32
    const float*  __restrict__ h0g,    // h0  [N, H, DV, DK]   -- fp32, может быть nullptr
    __half* __restrict__ hg,           // h   [NT_all, H, DV, DK]
    __half* __restrict__ vng,          // v_new [T, H, DV]
    float*  __restrict__ htg,          // ht  [N, H, DV, DK]   -- fp32, может быть nullptr
    const int* __restrict__ cu,        // cu_seqlens    [N+1]
    const int* __restrict__ coff,      // chunk_offsets [N+1]
    int H, int HG) {
  constexpr int LDW = DK + 4;             // k-мажорный A: бесконфликтен, широкая запись незаконна
  constexpr int LDH = DK + 4;             // k-мажорный B: то же
  constexpr int LDK = DK + 16;            // n-мажорный B: шаг 16 (mod 64), широкая запись законна
  constexpr int LDT = gdn_h_ldt(BV);      // m-мажорный A: тот же закон
  constexpr int LDU = BV + 4;             // читается по карте аккумулятора
  constexpr int NW = WM1 * WN1;
  constexpr int NTHR = NW * 32;
  constexpr int SM1 = BT / WM1, SN1 = BV / WN1;
  constexpr int SM2 = BV / WM2, SN2 = DK / WN2;
  static_assert(WM1 * WN1 == WM2 * WN2, "обе фазы работают одними и теми же варпами");
  static_assert(SM1 % 16 == 0 && SN1 % 16 == 0, "плитка варпа GEMM1 кратна шагу фрагмента");
  static_assert(SM2 % 16 == 0 && SN2 % 16 == 0, "плитка варпа GEMM2 кратна шагу фрагмента");
  static_assert(DK % 8 == 0 && BV % 8 == 0, "перекладка идёт векторами");

  extern __shared__ __half smem[];
  __half* sW  = smem;                       // [BT][LDW]  окно w  (операнд A GEMM1)
  __half* sHh = sW  + BT * LDW;             // [BV][LDH]  fp16-состояние (операнд B GEMM1 И выход h)
  __half* sK  = sHh + BV * LDH;             // [BT][LDK]  окно k  (операнд B GEMM2)
  __half* sVt = sK  + BT * LDK;             // [BT][LDT]  гейтованное значение (операнд A GEMM2)
  __half* sU  = sVt + BT * LDT;             // [BT][LDU]  сперва u, ПОТОМ v_new (см. шапку)
  __shared__ float sGam[BT];

  const int i_v = blockIdx.x, i_nh = blockIdx.y;
  const int i_n = i_nh / H, hv = i_nh % H;
  const int hk = hv / (H / HG);            // отображение голова значения -> голова ключа, БЛОЧНОЕ
  const int bos = cu[i_n], T = cu[i_n + 1] - bos;
  const int NT = (T + BT - 1) / BT;
  const int boh = coff[i_n];
  const int v0 = i_v * BV;

  const int sw = H * DK, su = H * DV, sk = HG * DK, sg = H;
  const long sh = (long)H * DV * DK;
  const __half* pW = wg + ((long)bos * H + hv) * DK;
  const __half* pK = kg + ((long)bos * HG + hk) * DK;
  const __half* pU = ug + ((long)bos * H + hv) * DV;
  __half*       pV = vng + ((long)bos * H + hv) * DV;
  const float*  pG = gg + (long)bos * H + hv;
  __half*       pH = hg + ((long)boh * H + hv) * (long)DV * DK;

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm1 = warp / WN1, wn1 = warp % WN1;
  const int wm2 = warp / WN2, wn2 = warp % WN2;
  const int r10 = wm1 * SM1, c10 = wn1 * SN1;     // угол плитки этого варпа в GEMM1
  const int r20 = wm2 * SM2, c20 = wn2 * SN2;     // ... и в состоянии

  // Состояние живёт в РЕГИСТРАХ весь проход по чанкам: 8192 float на блок при BV=64 -- это
  // 32 регистра на нить при 256 нитях. Тип аккумулятора задан ориентацией GEMM2 (см. шапку).
  WarpMma<SM2 / 16, SN2 / 16, BLayout::kNMajor, ALayout::kMMajor> st;
  st.clear();
  if (h0g != nullptr) {
    const float* p0 = h0g + (long)i_nh * DV * DK;
    st.visit(lane, [&](float& x, int r, int c) { x = p0[(long)(v0 + r20 + r) * DK + c20 + c]; });
  }

  // --- ОКНО ЧАНКА ЕДЕТ ЧЕРЕЗ РЕГИСТРЫ. Замерено (§13 каталога задержек, tools/volta_pipe_test.cu):
  // выигрыш +20..46% даёт именно РАЗНЕСЕНИЕ выдачи LDG и записи STS, а равномерное размазывание
  // выдачи по слотам не даёт ничего. Здесь разнесение максимальное из возможного: LDG следующего
  // чанка выдаётся сразу после барьера готовности, а его STS -- в начале СЛЕДУЮЩЕЙ итерации, то
  // есть между ними стоят оба умножения, эпилог и обе записи в глобальную.
  // Цена -- (2*BT*DK/8 + BT*BV/4)/NTHR векторов на нить, и её надо проверять по ptxas: на 16 варпах
  // потолок 128 регистров, и там конвейер может не окупиться.
  constexpr int NLW = BT * (DK / 8) / NTHR;
  constexpr int NLU = BT * (BV / 4) / NTHR;
  static_assert(NLW * NTHR == BT * (DK / 8), "перекладка w/k делится на нити нацело");
  static_assert(NLU * NTHR == BT * (BV / 4), "перекладка u делится на нити нацело");
  uint4 rw[NLW], rk[NLW];
  uint2 ru[NLU];
  float rgam = 0.f, rgexp = 1.f;

  // Гейт считается ЗДЕСЬ, на стадии подвоза: exp идёт по SFU, который перекрывается с FP32 почти
  // даром (§4j), и уезжает с критического пути счётной фазы. Строки за пределами T зануляются ДО
  // экспоненты -- g соседней последовательности может дать inf, и множитель ноль его не спасёт.
  auto ldg = [&](int tn) {
    const int rn = min(BT, T - tn), ln = tn + rn - 1;
    const float gl = pG[(long)ln * sg];
    rgexp = __expf(gl);
    if (tid < BT) rgam = (tid < rn) ? __expf(gl - pG[(long)(tn + tid) * sg]) : 0.f;
#pragma unroll
    for (int q = 0; q < NLW; ++q) {
      const int i = tid + q * NTHR, r = i / (DK / 8), c = (i % (DK / 8)) * 8;
      const bool ok = (tn + r) < T;
      rw[q] = ok ? *reinterpret_cast<const uint4*>(&pW[(long)(tn + r) * sw + c]) : uint4{0, 0, 0, 0};
      rk[q] = ok ? *reinterpret_cast<const uint4*>(&pK[(long)(tn + r) * sk + c]) : uint4{0, 0, 0, 0};
    }
#pragma unroll
    for (int q = 0; q < NLU; ++q) {
      const int i = tid + q * NTHR, r = i / (BV / 4), c = (i % (BV / 4)) * 4;
      ru[q] = ((tn + r) < T)
                  ? *reinterpret_cast<const uint2*>(&pU[(long)(tn + r) * su + v0 + c])
                  : uint2{0, 0};
    }
  };
  // Ширина записи в разделяемую -- какую позволяет ДОПОЛНЕНИЕ буфера: у sK шаг 288 Б кратен 16 и
  // идёт STS.128, у sW шаг 264 Б -- нет, и идут две STS.64. Размен осознанный: бесконфликтность
  // чтений фрагментов дороже, чем лишняя инструкция записи (их на порядок меньше).
  auto sts = [&]() {
    if (tid < BT) sGam[tid] = rgam;
#pragma unroll
    for (int q = 0; q < NLW; ++q) {
      const int i = tid + q * NTHR, r = i / (DK / 8), c = (i % (DK / 8)) * 8;
      *reinterpret_cast<uint2*>(&sW[r * LDW + c])     = make_uint2(rw[q].x, rw[q].y);
      *reinterpret_cast<uint2*>(&sW[r * LDW + c + 4]) = make_uint2(rw[q].z, rw[q].w);
      *reinterpret_cast<uint4*>(&sK[r * LDK + c])     = rk[q];
    }
#pragma unroll
    for (int q = 0; q < NLU; ++q) {
      const int i = tid + q * NTHR, r = i / (BV / 4), c = (i % (BV / 4)) * 4;
      *reinterpret_cast<uint2*>(&sU[r * LDU + c]) = ru[q];
    }
  };

  if (PIPE && NT > 0) ldg(0);

  for (int it = 0; it < NT; ++it) {
    const int t0 = it * BT;   // длина чанка и gL нужны только подвозу -- они посчитаны внутри ldg()

    __syncthreads();                                // GEMM2 прошлого чанка дочитал sW/sK/sU/sVt

    if (!PIPE) ldg(t0);
    sts();
    const float gexp = rgexp;                       // снимок: rgexp перезапишет подвоз следующего

    // --- (a) состояние -> fp16 ОДИН раз: отсюда и запись h, и операнд B первого умножения.
    // ЗАТУХАНИЕ (e) сделано ТУТ ЖЕ, одним проходом по накопителю: оно не зависит от выгруженного
    // значения, а перенос его сюда убирает второй обход и уводит цепочку FMUL с критического пути
    // перед GEMM2. Пары регистров (rg, rg+1) -- СОСЕДНИЕ столбцы (карта аккумулятора), поэтому
    // запись идёт по 4 Б: вдвое меньше инструкций MIO, чем поэлементная (§4kk-девятое: скалярные
    // обращения находятся только разбором SASS и стоят заметно).
#pragma unroll
    for (int i = 0; i < SM2 / 16; ++i)
#pragma unroll
      for (int j = 0; j < SN2 / 16; ++j)
#pragma unroll
        for (int rg = 0; rg < 8; rg += 2) {
          const int r = r20 + i * 16 + wmma_acc_row(lane, rg);
          const int c = c20 + j * 16 + wmma_acc_col(lane, rg);
          *reinterpret_cast<__half2*>(&sHh[r * LDH + c]) =
              __floats2half2_rn(st.acc[i][j][rg], st.acc[i][j][rg + 1]);
          st.acc[i][j][rg]     *= gexp;
          st.acc[i][j][rg + 1] *= gexp;
        }
    __syncthreads();

    // ВЫДАЧА РАНО: окно следующего чанка запрашивается ЗДЕСЬ, а кладётся в разделяемую только в
    // начале следующей итерации -- между ними оба умножения (§13).
    if (PIPE && it + 1 < NT) ldg(t0 + BT);

    // --- запись h. Срез [v0, v0+BV) x весь DK лежит в глобальной СПЛОШНЫМ куском (шаг строки
    // ровно DK), поэтому проход по нему полностью коалесцирован; из sHh читаем по 4 значения --
    // шире не даёт дополнение +4 (264 Б не кратно 16).
    {
      __half* dst = pH + (long)it * sh + (long)v0 * DK;
      for (int i = tid; i < BV * (DK / 4); i += NTHR) {
        const int r = i / (DK / 4), c = (i % (DK / 4)) * 4;
        *reinterpret_cast<uint2*>(&dst[(long)r * DK + c]) =
            *reinterpret_cast<const uint2*>(&sHh[r * LDH + c]);
      }
    }

    // --- (b) GEMM1: acc = w x Hhat^T. Оба операнда k-мажорны.
    WarpMma<SM1 / 16, SN1 / 16, BLayout::kKMajor> m1;
    m1.clear();
    m1.template accumulate<DK>(sW, LDW, r10, sHh, LDH, c10, lane);

    // --- (c)+(d) эпилог. Обход развёрнут ЯВНО, а не через visit: регистры аккумулятора должны
    // адресоваться константами времени компиляции (§4d), и заодно пара регистров (rg, rg+1) --
    // это СОСЕДНИЕ столбцы одной строки, значит и чтение u, и обе записи идут парами по 4 Б.
#pragma unroll
    for (int i = 0; i < SM1 / 16; ++i)
#pragma unroll
      for (int j = 0; j < SN1 / 16; ++j)
#pragma unroll
        for (int rg = 0; rg < 8; rg += 2) {
          const int t = r10 + i * 16 + wmma_acc_row(lane, rg);
          const int c = c10 + j * 16 + wmma_acc_col(lane, rg);
          const float y0 = __half2float(sU[t * LDU + c])     - m1.acc[i][j][rg];
          const float y1 = __half2float(sU[t * LDU + c + 1]) - m1.acc[i][j][rg + 1];
          // v_new -- БЕЗ гейта, поверх уже прочитанного u (тот же адрес, тот же поток)
          *reinterpret_cast<__half2*>(&sU[t * LDU + c]) = __floats2half2_rn(y0, y1);
          const float gm = sGam[t];
          // vt -- ВТОРОЕ, независимое округление того же fp32-значения
          *reinterpret_cast<__half2*>(&sVt[t * LDT + c]) = __floats2half2_rn(y0 * gm, y1 * gm);
        }
    __syncthreads();

    // --- запись v_new сплошным проходом (см. шапку: почему не прямо из регистров)
    for (int i = tid; i < BT * (BV / 4); i += NTHR) {
      const int r = i / (BV / 4), c = (i % (BV / 4)) * 4;
      if ((t0 + r) < T)
        *reinterpret_cast<uint2*>(&pV[(long)(t0 + r) * su + v0 + c]) =
            *reinterpret_cast<const uint2*>(&sU[r * LDU + c]);
    }

    // --- (f) внешнее произведение. Затухание (e) уже применено выше, одним проходом с выгрузкой.
    st.template accumulate<BT>(sVt, LDT, r20, sK, LDK, c20, lane);
  }

  if (htg != nullptr) {
    float* pt = htg + (long)i_nh * DV * DK;
    st.visit(lane, [&](float& x, int r, int c) { pt[(long)(v0 + r20 + r) * DK + c20 + c] = x; });
  }
}

}  // namespace gdn
}  // namespace fa2_sm70
