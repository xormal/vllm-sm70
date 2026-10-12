// СЛИТОЕ ТЕЛО: РЕКУРРЕНТНОСТЬ + ВЫХОД В ОДНОМ ПУСКЕ (h + o).
//
// ЗАЧЕМ. Разделённые `chunk_h` и `chunk_o` обмениваются через HBM плитой h [NT,H,DV,DK] и v_new:
// 75 МБ на слой при куске 4096 и 300 МБ при 16384, на 48 слоях GDN -- гигабайты туда и обратно.
// Плита существует НЕ потому, что данные не помещаются: состояние головы -- 32 КиБ, и оно уже
// лежит в регистрах блока рекуррентности. Она существует потому, что потребитель -- ДРУГОЙ ПУСК.
//
// ЧТО СДЕЛАНО. Блок, ведущий рекуррентность по паре (голова, срез BV), в точке, где он держит
// состояние чанка в sHh и собирался писать его наружу, ВМЕСТО ЗАПИСИ считает выход своего чанка:
//     o = scale * ( e^g (q h^T) + tril(e^{g_t-g_s} q k^T) v_new )
// Оба слагаемых берутся из того, что у блока уже есть: h -- в sHh, k -- в окне чанка, v_new --
// продукт GEMM1 этого же тела. Наружу не идёт ни h, ни v_new.
//
// БЮДЖЕТ РАЗДЕЛЯЕМОЙ (BT=64, DK=DV=128, BV=64): 86.5 КиБ динамической + 1.0 статической при
// потолке 96. sA лежит ПОВЕРХ sW, sAi -- ПОВЕРХ sVt, поэтому оба среза стоили памяти НОЛЬ.
//
// ВТОРОЙ СРЕЗ (WU=true): исчезают и w, и u. Это не «ещё одно слияние», а СОКРАЩЕНИЕ:
//     w = Ai (beta e^g K),  u = Ai (beta V)
//     v_new = u - w h^T = Ai diag(beta) [ V - diag(e^g) K h^T ]
// beta_j и e^{g_j} зависят только от индекса свёртки j, значит выносятся из скобки; скобка имеет
// ТУ ЖЕ форму [BT][BV], что уже считал GEMM1, и ложится в ТО ЖЕ окно sU. Поэтому прибавилось
// ровно одно умножение (Ai x скобка) той же формы, что и mv, а w не материализуется НИГДЕ -- ни в
// HBM, ни в разделяемой. Пуск gdn_wu исчезает целиком: пять ядер на слой становятся тремя.
//
// ЗАМЕРЕНО (боевой вид T=3136, 49 чанков, GQA=3, живое начальное состояние):
//   * цепочка скана x1.133 (0.842 -> 0.736 мс), устойчиво и на T=1024 (x1.128);
//   * снято с HBM: w+u = 38.5 МБ на слой при T=3136 (50.3 при куске 4096);
//   * ТОЧНОСТЬ ВЫРОСЛА. Арбитр -- вся цепочка в fp64 с точным обращением (I+A), tools/check_ho.py
//     --арбитр: o к истине 2.684e-04 против 2.732e-04 у пятёрки ядер, ht 1.267e-04 против 1.262e-04.
//     Сверять новый путь со СТАРЫМ на этот вопрос нельзя в принципе: подстановка меняет ПОРЯДОК
//     округлений, и требование «бит в бит» означало бы требование повторять чужие ошибки.
//   * регистры 254 при потолке 255, разлив 0, кадр стека 0 (проверено cuobjdump).
//
// ПРОВЕРКА. Эталон -- нынешняя пятёрка ядер на том же входе, посрезно (A, v_new, o, ht), плюс
// независимый fp64-арбитр. См. tools/check_ho.py. Косинус на выходе приёмкой НЕ является.

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

// Размер разделяемой памяти -- считается и хостом (для cudaFuncSetAttribute), и ядром.
template <int BT, int DK, int BV>
constexpr size_t gdn_ho_smem() {
  constexpr int PADO = 8;
  return (size_t)(BT * (DK + PADO) +                      // sQ -- вся доплата слияния (sA поверх sW)
                  BT * (DK + 4) + BV * (DK + 4) + BT * (DK + 16) + BT * gdn_h_ldt(BV) +
                  BT * (BV + 4)) * sizeof(__half);
}

// BT -- размер чанка (64); DK/DV -- размерности головы; BV -- срез по V, которым владеет блок;
// WM1 x WN1 -- сетка варпов на выходе GEMM1 [BT][BV]; WM2 x WN2 -- на состоянии [BV][DK];
// PIPE -- разнести выдачу LDG и запись STS окна чанка (см. ниже).
// WU=true -- ВТОРОЙ СРЕЗ СЛИЯНИЯ: w и u не читаются, потому что их НЕТ. Подстановка
//     w = Ai (beta e^g K),  u = Ai (beta V)   =>   v_new = u - w h^T = Ai diag(beta) [ V - diag(e^g) K h^T ]
// -- beta_j и e^{g_j} зависят ТОЛЬКО от индекса свёртки j, то есть от СТРОКИ скобки, и потому
// выносятся наружу целиком. Значит вместо двух плит [BT][DK] и [BT][BV] в HBM хватает k, v, Ai и
// beta, а лишнего буфера не нужно вовсе: скобка считается ТЕМ ЖЕ умножением (m1) в ТО ЖЕ окно sU,
// а Ai ложится поверх мёртвого sVt. Пуск gdn_wu исчезает целиком.
// ТИП СОСТОЯНИЯ -- ПАРАМЕТР (StateT: float или __half). В режиме префикс-кэша состояние
// лежит на КАЖДОЙ границе блока, и его формат прямо задаёт ёмкость контекста; конверсия
// снаружи была бы лишним полным проходом по HBM на каждый чанк.
template <int BT, int DK, int DV, int BV, int WM1, int WN1, int WM2, int WN2, bool PIPE = true,
          bool WU = false, typename StateT = float>
__global__ __launch_bounds__(WM1 * WN1 * 32, 1) void volta_gdn_chunk_ho(
    const __half* __restrict__ kg,     // k   [T, HG, DK]      -- голова КЛЮЧА
    const __half* __restrict__ wg,     // w   [T, H,  DK]      -- при WU НЕ ЧИТАЕТСЯ (может быть nullptr)
    const __half* __restrict__ ug,     // u   [T, H,  DV]      -- при WU здесь СЫРОЕ v
    const float*  __restrict__ gg,     // g   [T, H]           -- накопленный логарифм гейта, fp32
    const StateT* __restrict__ h0g,    // h0  [N, H, DV, DK]   -- StateT, может быть nullptr
    __half* __restrict__ hg,           // h   [NT_all, H, DV, DK]
    __half* __restrict__ vng,          // v_new [T, H, DV]
    StateT* __restrict__ htg,          // ht  [N, H, DV, DK]   -- StateT, может быть nullptr
    const int* __restrict__ cu,        // cu_seqlens    [N+1]
    const int* __restrict__ coff,      // chunk_offsets [N+1]
    const __half* __restrict__ qg,     // q   [T, HG, DK]   -- ВХОД СЛИЯНИЯ
    __half* __restrict__ og,           // o   [T, H,  DV]   -- ВЫХОД СЛИЯНИЯ
    __half* __restrict__ adbg,         // A   [NT_all, H, BT, BT] -- ТОЛЬКО для сверки, может быть nullptr
    const __half* __restrict__ aig,    // Ai  [T, H, BT]   -- ВХОД ВТОРОГО СРЕЗА (WU)
    const __half* __restrict__ bbg,    // beta[T, H]       -- ВХОД ВТОРОГО СРЕЗА (WU)
    float scale,
    int H, int HG) {
  constexpr int LDW = DK + 4;             // k-мажорный A: бесконфликтен, широкая запись незаконна
  constexpr int LDH = DK + 4;             // k-мажорный B: то же
  constexpr int LDK = DK + 16;            // n-мажорный B: шаг 16 (mod 64), широкая запись законна
  constexpr int LDT = gdn_h_ldt(BV);      // m-мажорный A: тот же закон
  constexpr int LDU = BV + 4;             // читается по карте аккумулятора
  constexpr int LDQ = DK + 8;             // окно q: шаг кратен 8 полусловам -- широкая подача uint4
  constexpr int LDA = BT + 8;             // [BT][BT] внимания внутри чанка
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
  __half* sQ  = sU  + BT * LDU;             // [BT][LDQ]  окно q -- ДОПЛАТА СЛИЯНИЯ (17 КиБ)
  // sA ЛОЖИТСЯ ПОВЕРХ sW: окно w дочитано первым умножением (GEMM1) ещё до эпилога, а переписано
  // будет только в начале следующего чанка -- за барьером в голове цикла. Доплата памятью НОЛЬ.
  // Без этого бюджет был 95.5 динамической + 0.75 статической = 96.25 при потолке 96, и запуск
  // отвергался драйвером ("invalid argument"), а не замедлялся.
  __half* sA  = sW;                         // [BT][LDA]  внутричанковое внимание, поверх мёртвого sW
  static_assert(BT * (BT + 8) <= BT * (DK + 4), "sA обязан помещаться в место sW");
  // sAi ЛОЖИТСЯ ПОВЕРХ sVt, и это ЗАКОННО по времени жизни, а не по удаче:
  //   * sVt прошлого чанка дочитан GEMM2 в конце прошлой итерации, а перед подвозом стоит барьер
  //     головы цикла -- значит с этого момента окно свободно;
  //   * sAi живёт от подвоза до умножения mn (скобка -> v_new), после которого стоит барьер;
  //   * sVt этого чанка пишется СРАЗУ ПОСЛЕ того барьера, когда Ai уже никому не нужен.
  // Доплата разделяемой -- НОЛЬ, поэтому бюджет остаётся 86.5 КиБ и запуск не упирается в потолок.
  constexpr int LDAI = BT + 4;              // k-мажорный A: шаг 34 слова, все 16 строк по разным банкам
  __half* sAi = sVt;                        // [BT][LDAI] обратная унитреугольная, поверх мёртвого sVt
  static_assert(!WU || BT * LDAI <= BT * LDT, "sAi обязан помещаться в место sVt");
  __shared__ float sGam[BT];
  __shared__ float sBt[BT];   // beta_t: множитель СТРОКИ скобки (при WU)
  // e^{g_t} ОТДЕЛЬНЫМ ЧТЕНИЕМ, а не как gexp/sGam[t]. Тождество exp(gL)/exp(gL-g_t) = exp(g_t)
  // верно алгебраически и НЕВЕРНО численно: gL накапливается по чанку далеко в минус, обе
  // экспоненты выпадают в ноль, и получается 0/0 = NaN. Замерено на боевом виде: срез 1 дал nan
  // при точном ht. Экономия одного чтения не стоит потери значения.
  __shared__ float sEg[BT];
  __shared__ float sGt[BT];   // сырой g_t: нужен маске e^{g_t-g_s} внутри чанка

  const int i_v = blockIdx.x, i_nh = blockIdx.y;
  const int i_n = i_nh / H, hv = i_nh % H;
  const int hk = hv / (H / HG);            // отображение голова значения -> голова ключа, БЛОЧНОЕ
  const int bos = cu[i_n], T = cu[i_n + 1] - bos;
  const int NT = (T + BT - 1) / BT;
  const int boh = coff[i_n];
  const int v0 = i_v * BV;

  const int sw = H * DK, su = H * DV, sk = HG * DK, sg = H;
  const long sh = (long)H * DV * DK;
  const __half* pQ = qg + ((long)bos * HG + hk) * DK;   // ВХОД СЛИЯНИЯ: q берётся головой КЛЮЧА
  __half*       pO = og + ((long)bos * H + hv) * DV;   // ВЫХОД СЛИЯНИЯ
  const __half* pW = (wg != nullptr) ? wg + ((long)bos * H + hv) * DK : nullptr;
  const __half* pK = kg + ((long)bos * HG + hk) * DK;
  const __half* pU = ug + ((long)bos * H + hv) * DV;
  __half*       pV = (vng != nullptr) ? vng + ((long)bos * H + hv) * DV : nullptr;
  const float*  pG = gg + (long)bos * H + hv;
  // Ai лежит как [T, H, BT]: шаг по токену -- H*BT, столбец s -- внутри чанка. beta -- [T, H].
  const __half* pAi = (aig != nullptr) ? aig + ((long)bos * H + hv) * BT : nullptr;
  const __half* pB  = (bbg != nullptr) ? bbg + (long)bos * H + hv : nullptr;
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
    const StateT* p0 = h0g + (long)i_nh * DV * DK;
    st.visit(lane, [&](float& x, int r, int c) {
      const StateT з = p0[(long)(v0 + r20 + r) * DK + c20 + c];
      if constexpr (sizeof(StateT) == 4) x = *reinterpret_cast<const float*>(&з);
      else x = __half2float(*reinterpret_cast<const __half*>(&з));
    });
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
  constexpr int NLA = WU ? BT * (BT / 8) / NTHR : 1;   // Ai [BT][BT] едет теми же векторами
  static_assert(!WU || NLA * NTHR == BT * (BT / 8), "перекладка Ai делится на нити нацело");
  uint4 rw[NLW], rk[NLW], rA[NLA];
  uint2 ru[NLU];
  float rgam = 0.f, rgexp = 1.f, rgt = 0.f, rgraw = 0.f, rbt = 0.f;

  // Гейт считается ЗДЕСЬ, на стадии подвоза: exp идёт по SFU, который перекрывается с FP32 почти
  // даром (§4j), и уезжает с критического пути счётной фазы. Строки за пределами T зануляются ДО
  // экспоненты -- g соседней последовательности может дать inf, и множитель ноль его не спасёт.
  auto ldg = [&](int tn) {
    const int rn = min(BT, T - tn), ln = tn + rn - 1;
    const float gl = pG[(long)ln * sg];
    rgexp = __expf(gl);
    if (tid < BT) {
      const float gt_ = (tid < rn) ? pG[(long)(tn + tid) * sg] : 0.f;
      rgam = (tid < rn) ? __expf(gl - gt_) : 0.f;
      rgt  = (tid < rn) ? __expf(gt_) : 0.f;
      rgraw = (tid < rn) ? gt_ : 0.f;
      // beta ЗА ПРЕДЕЛАМИ чанка -- ноль: строка скобки обнуляется целиком, и мусор из соседней
      // последовательности не доезжает до умножения.
      if constexpr (WU) rbt = (tid < rn) ? __half2float(pB[(long)(tn + tid) * sg]) : 0.f;
    }
#pragma unroll
    for (int q = 0; q < NLW; ++q) {
      const int i = tid + q * NTHR, r = i / (DK / 8), c = (i % (DK / 8)) * 8;
      const bool ok = (tn + r) < T;
      // при WU окна w нет: sW получит ТОТ ЖЕ k (одна выдача LDG вместо двух -- чтение k, бывшее
      // двойным (w и k), становится одинарным).
      if constexpr (!WU)
        rw[q] = ok ? *reinterpret_cast<const uint4*>(&pW[(long)(tn + r) * sw + c]) : uint4{0, 0, 0, 0};
      rk[q] = ok ? *reinterpret_cast<const uint4*>(&pK[(long)(tn + r) * sk + c]) : uint4{0, 0, 0, 0};
    }
    if constexpr (WU) {
#pragma unroll
      for (int q = 0; q < NLA; ++q) {
        const int i = tid + q * NTHR, r = i / (BT / 8), c = (i % (BT / 8)) * 8;
        rA[q] = ((tn + r) < T)
                    ? *reinterpret_cast<const uint4*>(&pAi[(long)(tn + r) * H * BT + c])
                    : uint4{0, 0, 0, 0};
      }
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
    if (tid < BT) {
      sGam[tid] = rgam; sEg[tid] = rgt; sGt[tid] = rgraw;
      if constexpr (WU) sBt[tid] = rbt;
    }
#pragma unroll
    for (int q = 0; q < NLW; ++q) {
      const int i = tid + q * NTHR, r = i / (DK / 8), c = (i % (DK / 8)) * 8;
      const uint4 a = WU ? rk[q] : rw[q];       // при WU левый операнд m1 -- сам k, без масштаба
      *reinterpret_cast<uint2*>(&sW[r * LDW + c])     = make_uint2(a.x, a.y);
      *reinterpret_cast<uint2*>(&sW[r * LDW + c + 4]) = make_uint2(a.z, a.w);
      *reinterpret_cast<uint4*>(&sK[r * LDK + c])     = rk[q];
    }
    if constexpr (WU) {
#pragma unroll
      for (int q = 0; q < NLA; ++q) {
        const int i = tid + q * NTHR, r = i / (BT / 8), c = (i % (BT / 8)) * 8;
        // шаг строки 68 half = 136 Б: НЕ кратен 16, поэтому широкая запись НЕЗАКОННА (нечётные
        // строки дали бы misaligned address). Две STS.64 -- та же цена, что у sW по той же причине.
        *reinterpret_cast<uint2*>(&sAi[r * LDAI + c])     = make_uint2(rA[q].x, rA[q].y);
        *reinterpret_cast<uint2*>(&sAi[r * LDAI + c + 4]) = make_uint2(rA[q].z, rA[q].w);
      }
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

    // --- ЗАПИСИ h БОЛЬШЕ НЕТ. Вместо неё выход своего чанка считается ЗДЕСЬ ЖЕ, из sHh.
    // СРЕЗ 1 СЛИЯНИЯ: первое слагаемое o = scale * e^{g_t} * (q h^T).
    //   * форма умножения ТА ЖЕ, что у GEMM1 ниже ([BT]x[BV], оба операнда k-мажорны), поэтому
    //     берётся существующая раскладка варпов и та же карта аккумулятора -- это снимает главный
    //     риск слияния (ошибка в индексе фрагмента не падает, а даёт правдоподобно неверные числа);
    //   * e^{g_t} НЕ ЧИТАЕТСЯ: gexp = e^{gL}, sGam[t] = e^{gL - g_t}, значит e^{g_t} = gexp/sGam[t];
    //   * подвоз q БЕЗ конвейера -- сперва верность, укладка потом (§13 каталога задержек).
    {
      const int rn2 = min(BT, T - t0);
      for (int i = tid; i < BT * (DK / 8); i += NTHR) {
        const int r = i / (DK / 8), c = (i % (DK / 8)) * 8;
        const int rr = r < rn2 ? r : rn2 - 1;      // хвост -- зажимом строки, не предикатом
        *reinterpret_cast<uint4*>(&sQ[r * LDQ + c]) =
            *reinterpret_cast<const uint4*>(&pQ[(long)(t0 + rr) * sk + c]);
      }
    }
    __syncthreads();
    WarpMma<SM1 / 16, SN1 / 16, BLayout::kKMajor> mo;
    mo.clear();
    mo.template accumulate<DK>(sQ, LDQ, r10, sHh, LDH, c10, lane);
    // накопитель mo доживает до слияния слагаемых -- см. СРЕЗ 2+3 после эпилога

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
          // При WU в sU лежит СЫРОЕ v, а m1 посчитал k h^T (без e^g): оба множителя строки,
          // beta_t и e^{g_t}, применяются ЗДЕСЬ -- получается скобка [ V - diag(e^g) K h^T ] * beta,
          // и она ЕЩЁ НЕ v_new: слева не хватает Ai (умножение mn ниже). Поэтому sVt здесь не
          // пишется -- гейтовать нечего, а его окно сейчас занято под Ai.
          const float m0 = WU ? sEg[t] : 1.f, bt = WU ? sBt[t] : 1.f;
          const float y0 = bt * (__half2float(sU[t * LDU + c])     - m0 * m1.acc[i][j][rg]);
          const float y1 = bt * (__half2float(sU[t * LDU + c + 1]) - m0 * m1.acc[i][j][rg + 1]);
          // v_new -- БЕЗ гейта, поверх уже прочитанного u (тот же адрес, тот же поток)
          *reinterpret_cast<__half2*>(&sU[t * LDU + c]) = __floats2half2_rn(y0, y1);
          if constexpr (!WU) {
            const float gm = sGam[t];
            // vt -- ВТОРОЕ, независимое округление того же fp32-значения
            *reinterpret_cast<__half2*>(&sVt[t * LDT + c]) = __floats2half2_rn(y0 * gm, y1 * gm);
          }
        }
    __syncthreads();

    // --- СРЕЗ 4 СЛИЯНИЯ: v_new = Ai x скобка. Форма ТА ЖЕ, что у mv ниже ([BT][BT] x [BT][BV],
    // операнд B n-мажорен), поэтому переиспользуется отлаженная раскладка варпов.
    if constexpr (WU) {
      WarpMma<SM1 / 16, SN1 / 16, BLayout::kNMajor> mn;
      mn.clear();
      mn.template accumulate<BT>(sAi, LDAI, r10, sU, LDU, c10, lane);
      // БАРЬЕР ОБЯЗАТЕЛЕН И ПО ДВУМ ПРИЧИНАМ: (1) sU перезаписывается результатом, а читали его
      // ВСЕ варпы, не только владелец ячейки; (2) sVt лежит поверх sAi, и запись ниже затирает Ai.
      __syncthreads();
#pragma unroll
      for (int i = 0; i < SM1 / 16; ++i)
#pragma unroll
        for (int j = 0; j < SN1 / 16; ++j)
#pragma unroll
          for (int rg = 0; rg < 8; rg += 2) {
            const int t = r10 + i * 16 + wmma_acc_row(lane, rg);
            const int c = c10 + j * 16 + wmma_acc_col(lane, rg);
            const float y0 = mn.acc[i][j][rg], y1 = mn.acc[i][j][rg + 1];
            *reinterpret_cast<__half2*>(&sU[t * LDU + c]) = __floats2half2_rn(y0, y1);
            const float gm = sGam[t];
            *reinterpret_cast<__half2*>(&sVt[t * LDT + c]) = __floats2half2_rn(y0 * gm, y1 * gm);
          }
      __syncthreads();
    }

    // --- запись v_new сплошным проходом (см. шапку: почему не прямо из регистров).
    // ГЕЙТ ПО УКАЗАТЕЛЮ, а не безусловно: в слитом теле v_new -- ВНУТРЕННЕЕ значение, его читает
    // только умножение mv двумя строками ниже. Запись без потребителя стоила 19.3 МБ на слой и была
    // тем обиднее, что я уже записал в память, будто v_new наружу не идёт. Читатель есть только у
    // прежней пары h+o (там v_new -- вход отдельного пуска gdn_o), поэтому решает указатель.
    if (pV != nullptr) {
      for (int i = tid; i < BT * (BV / 4); i += NTHR) {
        const int r = i / (BV / 4), c = (i % (BV / 4)) * 4;
        if ((t0 + r) < T)
          *reinterpret_cast<uint2*>(&pV[(long)(t0 + r) * su + v0 + c]) =
              *reinterpret_cast<const uint2*>(&sU[r * LDU + c]);
      }
    }

    // --- СРЕЗ 2+3 СЛИЯНИЯ: внутричанковая часть выхода.
    //   A[t][s] = e^{g_t-g_s} (q_t . k_s) при s<=t, иначе 0;   o += A x v_new
    // v_new лежит в sU (эпилог только что его туда положил), k -- в sK, q -- в sQ. Наружу за ними
    // ходить не надо: всё это уже в блоке. GEMM2 ниже sK только ЧИТАЕТ, поэтому порядок безопасен.
    {
      // СЕТКА ВАРПОВ ТА ЖЕ, ЧТО У ПЕРВОГО УМНОЖЕНИЯ. Выход A -- [BT][BT], а не [BT][BV], но
      // столбцов НА ВАРП столько же: 4 варпа по N x SN1=16 кроют ровно BT=64. Поставленное
      // сгоряча BT/16 давало каждому варпу вчетверо больше столбцов -- варпы писали внахлёст и
      // за конец строки (LDA=72), отчего |A| выходила 0.45 эталона: не мусор, а НЕДОЗАПОЛНЕНИЕ.
      static_assert(SN1 * WN1 == BT, "столбцы A должны ровно покрываться сеткой варпов");
      WarpMma<SM1 / 16, SN1 / 16, BLayout::kKMajor> ma;
      ma.clear();
      ma.template accumulate<DK>(sQ, LDQ, r10, sK, LDK, c10, lane);
      const int rn2 = min(BT, T - t0);
#pragma unroll
      for (int i = 0; i < SM1 / 16; ++i)
#pragma unroll
        for (int j = 0; j < SN1 / 16; ++j)
#pragma unroll
          for (int rg = 0; rg < 8; ++rg) {
            const int tt = r10 + i * 16 + wmma_acc_row(lane, rg);
            const int ss = c10 + j * 16 + wmma_acc_col(lane, rg);
            const bool ok = (tt < rn2) && (ss <= tt) && (ss < rn2);
            sA[(long)tt * LDA + ss] =
                __float2half(ok ? ma.acc[i][j][rg] * __expf(sGt[tt] - sGt[ss]) : 0.f);
          }
    }
    __syncthreads();
    // ВЫГРУЗКА A ДЛЯ СВЕРКИ. Обе половины по BV считают ОДИНАКОВУЮ A (она без измерения V),
    // поэтому пишет только срез 0 -- иначе две записи в одно место.
    if (adbg != nullptr && i_v == 0) {
      __half* dA = adbg + ((long)(boh + it) * H + hv) * BT * BT;
      for (int i = tid; i < BT * BT; i += NTHR) {
        const int r = i / BT, c = i % BT;
        dA[(long)r * BT + c] = sA[(long)r * LDA + c];
      }
    }
    {
      WarpMma<SM1 / 16, SN1 / 16, BLayout::kNMajor> mv;
      mv.clear();
      mv.template accumulate<BT>(sA, LDA, r10, sU, LDU, c10, lane);
      if (og != nullptr) {
#pragma unroll
        for (int i = 0; i < SM1 / 16; ++i)
#pragma unroll
          for (int j = 0; j < SN1 / 16; ++j)
#pragma unroll
            for (int rg = 0; rg < 8; ++rg) {
              const int tt = r10 + i * 16 + wmma_acc_row(lane, rg);
              const int cc = c10 + j * 16 + wmma_acc_col(lane, rg);
              if (t0 + tt < T)
                pO[(long)(t0 + tt) * su + v0 + cc] = __float2half(
#ifdef FA2SM70_HO_ТОЛЬКО1
                    scale * sEg[tt] * mo.acc[i][j][rg]);   // фальсификатор: только первое слагаемое
#else
                    scale * (sEg[tt] * mo.acc[i][j][rg] + mv.acc[i][j][rg]));
#endif
            }
      }
    }
    __syncthreads();

    // --- (f) внешнее произведение. Затухание (e) уже применено выше, одним проходом с выгрузкой.
    st.template accumulate<BT>(sVt, LDT, r20, sK, LDK, c20, lane);
  }

  if (htg != nullptr) {
    StateT* pt = htg + (long)i_nh * DV * DK;
    st.visit(lane, [&](float& x, int r, int c) {
      StateT& ячейка = pt[(long)(v0 + r20 + r) * DK + c20 + c];
      if constexpr (sizeof(StateT) == 4) *reinterpret_cast<float*>(&ячейка) = x;
      else *reinterpret_cast<__half*>(&ячейка) = __float2half_rn(x);
    });
  }
}

}  // namespace gdn
}  // namespace fa2_sm70
