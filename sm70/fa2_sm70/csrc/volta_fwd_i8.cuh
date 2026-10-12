// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
//
// Volta (sm_70) FMHA FORWARD с int8-ключами и значениями. ЯДРО НОВОГО ФУНДАМЕНТА.
//
// ================================================================================================
// ПОЧЕМУ ОНО ТАКОЕ: конструкция ВЫВЕДЕНА из замеренных законов, а не подобрана свипом.
// Полная матрица законов -- fa2_src/fmha_kernel/gemm/volta_prims.h, разделы §0 и §4e-§4v.
// ================================================================================================
//
// ЕДИНИЦА ПРОЕКТИРОВАНИЯ -- ПРИЕХАВШАЯ ПЛИТКА. HBM здесь не "источник данных", а СРЕДА ДОСТАВКИ:
// он обязан привезти и увезти рабочий набор, и вопрос не в объёме, а в том, ХВАТАЕТ ЛИ СЧЁТА, чтобы
// доставку спрятать. Отсюда закон транспорта:
//
//     нужно  (пик МАС/такт) / (байт/такт на SM)  =  0.434*1024 / 7.6  =  58 МАС на байт
//     даёт   BQ / (байт на элемент K/V)
//
// У прежнего ядра (fp16, BQ=64) плотность 32 при требуемых 58 -- ТРАНСПОРТ НЕ СПРЯТАН НИКОГДА.
// У уравнения два корня, и первый НЕДОСТИЖИМ:
//   * поднять BQ до 128 -- упирается в регистровый файл. Накопитель O требует BQ*d/нитей регистров;
//     при 512 нитях на нить приходится ровно 128 регистров (65536/512), а нужно O(64) + фрагменты
//     Q(32) + накладные(~100). Не влезает, и это арифметика, а не вопрос старания.
//   * СЖАТЬ ФОРМАТ -- при int8 плотность равна BQ, то есть 64 >= 58 УЖЕ при текущей плитке.
// ИСПРАВЛЕНИЕ ПОСЛЕ ЗАМЕРА (§4w). Первая редакция разворачивала int8 -> fp16 В РАЗДЕЛЯЕМУЮ ПАМЯТЬ и
// оказалась x0.70 к fp16. Две причины, обе -- ошибки вывода:
//   * закон транспорта посчитан против HBM, а трафик туда НЕ ДОХОДИТ (97.7% попаданий в L2), так
//     что требование завышено в 43 раза и узким местом HBM не является;
//   * развёрнутая в разделяемую память плитка занимает СТОЛЬКО ЖЕ, сколько fp16 -- ограничитель
//     (разделяемая память) не разгружен вовсе, а работа разворота добавлена.
// ВЕРНАЯ КОНСТРУКЦИЯ: держать K в int8 ПРЯМО В РАЗДЕЛЯЕМОЙ и разворачивать ТОЛЬКО ВО ФРАГМЕНТЫ В
// РЕГИСТРАХ при подаче в MMA. Тогда рычаг попадает в настоящий ограничитель: у k-мажорного операнда
// ОДНА LDS.128 кроет ЧЕТЫРЕ шага по k вместо двух, то есть вдвое меньше обращений при тех же данных.
//
// И РАЗГРАНИЧЕНИЕ, КОТОРОЕ РЕШАЕТ, ЧТО СЖИМАТЬ: выигрыш даёт только k-мажорный операнд.
//   K (k-мажорный) -- одна LDS.128 на четыре шага: замерено 3.315 против базы fp16 3.666, x1.106;
//   V (n-мажорный) -- база полосы шагает по 4, широкая загрузка ЗАПРЕЩЕНА раскладкой инструкции,
//     узкая LDS.32 даёт ТО ЖЕ число инструкций при вдвое меньших байтах: замерено 3.800, ХУЖЕ базы.
// Поэтому здесь K в int8, а V остаётся fp16. Сжимать надо не "данные", а тот операнд, у которого
// сжатие превращается в МЕНЬШЕЕ ЧИСЛО ИНСТРУКЦИЙ.
//
// РАСПИСАНИЕ. Метапроцессор складывается ровно из четырёх маршрутов, и они замерены вместе:
// 85.9% тензорного пика при всех занятых (§4p, §4r). Принципиален не состав, а СДВИГ ФАЗ --
// та же работа в цепи стоит 11.4% тензорной полосы, сдвинутая на плитку вперёд 6.9%:
//
//     Tensor -> плитка N     : S = Q*K^T,  O += P*V   (чистые умножения, НЕ разбавлять -- §4u)
//     INT32  -> плитка N+1   : разворот int8->fp16 перестановкой байт, адреса, маска (2 инстр/HMMA даром)
//     FP32   -> плитка N-1   : поправка накопителя, масштаб S  (1 инстр/HMMA даром)
//     LSU    -> плитка N+1   : подвоз K/V ОДНОЙ широкой инструкцией
//
// МАСШТАБЫ КВАНТОВАНИЯ -- по §4m, "что в тензорной инструкции даром":
//   * s_k[j] (на позицию ключа) НЕ выносится из софтмакса, потому что softmax идёт по j. Значит
//     нужен поэлементный домножитель после первого умножения -- он уезжает в свободный слот FP32.
//   * s_v[j] СВОРАЧИВАЕТСЯ в уже существующее домножение P на cnew при записи P. Цена НОЛЬ:
//     операнд всё равно строится, масштаб входит в тот же множитель.
//
// ЧЕГО ЗДЕСЬ СОЗНАТЕЛЬНО НЕТ, и почему (все три ЗАМЕРЕНЫ, а не отброшены по вкусу):
//   * подсказки .cs на K/V -- хуже базы на всех формах: плитку ключей читают ВСЕ блоки запросов
//     (попадание в L2 97.7%), и "вытеснять первой" выбрасывает то, что нужно соседу (§4v);
//   * широкая запись перекладки -- ноль: перекладка это 4% трафика разделяемой памяти против ~96%
//     у чтений фрагментов операндов (§4v). Рычаг надо прикладывать к ДОМИНИРУЮЩЕЙ доле;
//   * игры с сеткой варпов -- 4 варпа замеренно хуже (1.427 против 1.146). Геометрия не рычаг.
#pragma once
// [ЗАМЕР ПО ФАЗАМ] Подбирать зазор между блоками вслепую нельзя -- нужна ДЛИТЕЛЬНОСТЬ каждой фазы.
// FA2_PHASE=1 добавляет отсчёты clock64 по границам фаз и складывает их в переданный буфер.
// Это ОТЛАДОЧНЫЙ режим: сами отсчёты сдвигают время, поэтому по ним читают ПРОПОРЦИИ, а не абсолют.
// [РАСФАЗИРОВКА ПО ОСИ TPC] Замер размещения (§4ii): диспетчер идёт по TPC, у V100 сорок TPC по два
// SM, и два SM ОДНОГО TPC делят путь к памяти. Соседа по TPC отличает МЛАДШИЙ БИТ smid -- ОДНА
// битовая операция, тогда как соседа по SM им не отличить вовсе (у них smid одинаков).
// Расфазируем НЕ задержкой (замерено вредной, §4hh) и НЕ сдвигом по данным (ломает переиспользование
// L2, §4ee), а ПОРЯДКОМ ОБХОДА КУСКОВ ВНУТРИ ПЛИТКИ: те же строки, тот же объём, но мгновенные
// адреса раскоррелированы. Сумма по d от порядка не зависит -- это сложение.
#ifndef FA2_TPCROT
#define FA2_TPCROT 0
#endif
#ifndef FA2_GAP
#define FA2_GAP 0
#endif
#ifndef FA2_PHASE
#define FA2_PHASE 0
#endif
#if FA2_PHASE
#define PH_DECL long long _pht = clock64(); long long _phacc[8] = {0,0,0,0,0,0,0,0};
#define PH(i)   do { long long _n = clock64(); _phacc[i] += _n - _pht; _pht = _n; } while (0)
#define PH_OUT(dst) do { if (threadIdx.x == 0) { for (int _i = 0; _i < 8; ++_i) \
                         atomicAdd(reinterpret_cast<unsigned long long*>(dst) + _i, \
                                   (unsigned long long)_phacc[_i]); } } while (0)
#else
#define PH_DECL
#define PH(i)
#define PH_OUT(dst)
#endif
#ifndef FA2_STAGGER
#define FA2_STAGGER 0
#endif
#ifndef FA2_DUNROLL
#define FA2_DUNROLL 4   // ЗАМЕРЕНО: после прямой адресации регистров хватает на разворот 4 ПРИ
                        // двух резидентных блоках (126 из 128, кадр 0, LDL/STL 0) -- 0.3868 против
                        // 0.4382 при развороте 1. Сначала снять лишние инструкции, ПОТОМ разворот.
#endif
// #pragma unroll НЕ раскрывает макросы -- нужен _Pragma и ДВА уровня, иначе имя попадает в строку
// как есть и прагма молча игнорируется (nvcc сообщает это предупреждением #20169, а не ошибкой).
#define FA2_DO_PRAGMA(x) _Pragma(#x)
#define FA2_UNROLL_(n)   FA2_DO_PRAGMA(unroll n)
#define FA2_UNROLL(n)    FA2_UNROLL_(n)
#include <cuda_fp16.h>
#include <cassert>
#include <cstdint>
#include <math_constants.h>
#include "gemm/volta_warp_mma.h"
#include "gemm/volta_prims.h"

namespace fa2_sm70 {
namespace fwd_i8 {

using fa2_sm70::WarpMma;
using fa2_sm70::BLayout;
using fa2_sm70::prims::expand_i8x4;
using fa2_sm70::prims::prefetch_l2;
using fa2_sm70::prims::sts128;
using fa2_sm70::prims::seed_rank1;      // §2a: аддитивная поправка -- в затравку, цена ноль
using fa2_sm70::prims::rescale_fold;    // §2b: перемасштабирование и предвычитание в один FFMA

// ------------------------------------------------------------------------------------------------
// РАЗВОРОТ ПЛИТКИ int8 -> fp16 В РАЗДЕЛЯЕМУЮ ПАМЯТЬ.
// Тремя инструкциями на четыре байта (§1a): XOR смещения, две перестановки байт. Значение b
// становится fp16-числом 1024+b, а смещение 1024 снимается СВЁРТКОЙ В РАНГ-1 на стороне масштаба
// (§2) -- не поэлементно. Это и есть "больше работы за одну выдачу", ярус 1 лестницы.
//
// ВАЖНО ПРО ФАЗУ: эта функция вызывается для плитки N+1 ВНУТРИ счёта плитки N. Её инструкции не
// зависят от текущей MMA и потому уезжают в свободные слоты INT32. Если позвать её в цепи с той же
// MMA, цена вырастает с 6.9% до 11.4% при ТОМ ЖЕ числе инструкций (§4r).
// ------------------------------------------------------------------------------------------------
template <int ROWS, int DCOL, int LD, int THREADS>
__device__ __forceinline__ void expand_tile_i8(__half* __restrict__ dst,
                                               const int8_t* __restrict__ src, long src_ld,
                                               int tid, int rows_valid) {
  // по четыре байта на обращение: одна LDS/LDG даёт четыре значения, разворот -- три инструкции
  constexpr int kQuads = ROWS * DCOL / 4;
  static_assert(kQuads % THREADS == 0, "плитка должна делиться на нити нацело");
#pragma unroll
  for (int i = tid; i < kQuads; i += THREADS) {
    const int r = i / (DCOL / 4), c = (i % (DCOL / 4)) * 4;
    uint32_t w = 0;
    if (r < rows_valid) w = *reinterpret_cast<const uint32_t*>(src + (long)r * src_ld + c);
    uint32_t f01, f23;
    expand_i8x4(w, f01, f23);
    __half* d = dst + r * LD + c;
    *reinterpret_cast<uint32_t*>(d)     = f01;
    *reinterpret_cast<uint32_t*>(d + 2) = f23;
  }
}

// ------------------------------------------------------------------------------------------------
// ЯДРО. Геометрия взята ОТ ОТГРУЖЕННОЙ (BQ=64, BK=64, 8 варпов 2x4, K кусками по DC): она вылизана
// свипом и, как показал замер, геометрия -- НЕ рычаг. Меняется фундамент подачи, а не плитки.
// Освободившиеся от int8 32 КБ разделяемой памяти -- задел под второй резидентный блок.
// ------------------------------------------------------------------------------------------------
// DV -- ШИРИНА СРЕЗА ВЫХОДА. Блок считает только O[:, slice], поэтому накопитель равен BQ*DV/нитей
// и сжимается ПРОПОРЦИОНАЛЬНО. Это единственный способ уменьшить связывающий член: сам O несжимаем,
// но можно уменьшить то, ЧЕМ ВЛАДЕЕТ БЛОК. Цена -- S = Q*K^T считается для каждого среза заново.
template <int BQ, int BK, int D, int WM, int WN, bool CAUSAL, int DC = 64, int MINB = 1, int REV = 1,
          int DV = D>
__global__ __launch_bounds__(WM * WN * 32, MINB) void volta_fwd_i8(
    const __half* __restrict__ Q,
    const int8_t* __restrict__ Kq, const float* __restrict__ Ks,   // K int8 + масштаб на позицию
    const int8_t* __restrict__ Vq, const float* __restrict__ Vs,   // V int8 + масштаб на позицию
    __half* __restrict__ O, float* __restrict__ Lse,
    int Sq, int Sk, float qk_scale, int kv_ratio, int causal_off,
    long q_sM, long q_sH, long q_sB, long k_sM, long k_sH, long k_sB,
    long o_sM, long o_sH, long o_sB, long lse_sH, long lse_sB, long ks_sH, long ks_sB,
    long long* __restrict__ phase_acc = nullptr, int* __restrict__ sm_role = nullptr,
    // Хвост-невидимка: paged-параметры ядра ролей (см. volta_fwd_ws.cuh, задача 163). Здесь НЕ
    // используются -- нужны только чтобы тип функции совпадал с общим указателем kern в хосте.
    const long* __restrict__ pageOff = nullptr, const long* __restrict__ pageScl = nullptr,
    int pageHkv = 0,
    // ХВОСТ РАЗРЕЖЕННОГО ОБХОДА -- здесь НЕ используется. Он есть только потому, что хост держит
    // ОДИН указатель `kern` на оба ядра (это и ролевое, и вот это), а тип указателя обязан совпасть.
    // Тихо разойтись подписям нельзя: компилятор ловит это присваиванием, и поймал.
    const int* __restrict__ selIdx = nullptr, const int* __restrict__ selCnt = nullptr,
    int selStride = 0, int selGroup = 1, int selSub = 0,
    // ХВОСТ РАСЩЕПЛЕНИЯ ПО КЛЮЧАМ (записка 24) -- здесь тоже НЕ используется, причина та же:
    // один указатель `kern` на оба ядра. Этот путь ОБЯЗАН зваться только с kSplit == 1;
    // при kSplit > 1 он посчитал бы весь диапазон и написал в O вместо среза -- то есть
    // тихо соврал бы. Утверждение ниже ловит это в отладочной сборке.
    int kSplit = 1, float* __restrict__ Opart = nullptr, float* __restrict__ LSEpart = nullptr) {
  (void)pageOff; (void)pageScl; (void)pageHkv;
  (void)selIdx; (void)selCnt; (void)selStride; (void)selGroup; (void)selSub;
  (void)Opart; (void)LSEpart;
  assert(kSplit == 1);
  (void)kSplit;
  constexpr int kWarps = WM * WN, PAD = 4;
  // Q ПОДАЁТСЯ КУСКАМИ ПО d, как и K: буфер BQ x DC вместо BQ x D -- минус 24 КБ, и это ровно то,
  // что осталось между 63 КБ и порогом двух резидентных блоков (48 КБ). Цена -- перечитывание Q на
  // каждую плитку ключей, но Q одного блока это 32 КБ, оно целиком лежит в L2 (DRAM 2-4% по профилю),
  // и перечитывание идёт оттуда, а не из HBM.
  constexpr int LDQ = DC + PAD, LDP = BK + PAD;
  // Шаг байтовой строки K кратен 16: широкая загрузка фрагмента (uint4 раз в четыре шага) читает
  // с адреса row*LDK8 + k, и при LDK8 % 16 != 0 он невыровнен -- это отказ, а не замедление.
  constexpr int LDK8 = DC + 16;
  static_assert(LDK8 % 16 == 0, "байтовый шаг K должен быть кратен 16 под LDS.128");
  // V ТОЖЕ ОСТАЁТСЯ БАЙТОВЫМ. Это снимает СРАЗУ ДВА предела, названных профилем:
  //   * разделяемая память -- минус 17 КБ (64*272*2 -> 64*272);
  //   * инструкции -- уходит ВЕСЬ проход разворота V (~12 тыс. инструкций на плитку), а ядро
  //     ЗАДЕРЖЕЧНОЕ (занятость 12.5%), где инструкция стоит время один к одному.
  // Прежний довод "у n-мажорного операнда узкая LDS.32 даёт то же число инструкций" остаётся верен
  // для ЧТЕНИЯ ФРАГМЕНТОВ -- но он про равенство, а не про проигрыш, тогда как проход разворота
  // исчезает целиком. В задержечном ядре это чистый выигрыш.
  constexpr int PADV = 16, LDV = DV + PADV;          // шаг V против банков -- см. §4a-4d и прежнее ядро
  constexpr int SM = BQ / WM, SN = BK / WN;
  constexpr int OM = BQ / WM, ON = DV / WN;
  constexpr int SMB = SM / 16, SNB = SN / 16;
  static_assert(SM % 16 == 0 && SN % 16 == 0 && OM % 16 == 0 && ON % 16 == 0, "плитки кратны 16");
  static_assert(OM == SM, "O и S делят строки");

  extern __shared__ __half smem_i8[];
  __half* sQ = smem_i8;                   // [BQ][LDQ]  fp16, резидентно
  __half* sP = sQ + BQ * LDQ;             // [BQ][LDP]
  int8_t* sK = reinterpret_cast<int8_t*>(sP + BQ * LDP);   // [BK][LDK8] кусок K ОСТАЁТСЯ БАЙТОВЫМ
  int8_t* sV = sK + BK * LDK8;            // [BK][LDV]  V ОСТАЁТСЯ БАЙТОВОЙ
  __shared__ float sRedM[BQ * WN];
    // ДВА значения на (строку, варп): софтмаксовая сумма И накопитель смещения разворота V.
  // Второе -- не украшение: без него ответ тихо неверен, см. блок про ранг-1 у второго умножения.
__shared__ float sRedS[BQ * WN * 2];
  __shared__ float sKs[BK];               // масштабы ключей текущей плитки -- варп-однородны по строке
  __shared__ float sVs[BK];

  constexpr float kI8Bias = 1152.0f;   // смещение разворота: 1024 (0x6400) + 128 (XOR знака)
  PH_DECL
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp / WN, wn = warp % WN;
  const int q0 = (REV ? (int)(gridDim.x - 1 - blockIdx.x) : (int)blockIdx.x) * BQ;
  constexpr int kSlices = D / DV;
  const int vslice = (int)(blockIdx.y % kSlices), voff = vslice * DV;
  {
    const long h = blockIdx.y / kSlices, b = blockIdx.z, hkv = h / kv_ratio;
    Q += b * q_sB + h * q_sH; O += b * o_sB + h * o_sH;
    Kq += b * k_sB + hkv * k_sH; Vq += b * k_sB + hkv * k_sH;
    Ks += b * ks_sB + hkv * ks_sH; Vs += b * ks_sB + hkv * ks_sH;
    Lse += b * lse_sB + h * lse_sH;
  }


  WarpMma<OM / 16, ON / 16, BLayout::kNMajor> o; o.clear();
  // [СВЁРТКА СМЕЩЕНИЯ У ВТОРОГО УМНОЖЕНИЯ] Плитка V тоже развёрнута как 1152+v, поэтому
  //     O_raw[i][d] = sum_j P'[i][j]*(1152 + V[j][d]) = sum_j P'*V + 1152 * sum_j P'[i][j]
  // Второе слагаемое НЕ ЗАВИСИТ ОТ d -- ранг-1 по строке запроса, но, в отличие от K, его надо
  // ВЕСТИ НАКОПИТЕЛЬНО и перемасштабировать ТЕМ ЖЕ множителем corr2, что и сам накопитель:
  //     vb_new = vb_old*corr2 + sum_j P'   =>   O_true = O_acc - 1152*vb   (проверяется подстановкой)
  // Забыть это -- получить relL2 порядка 100 при полностью правильном на вид коде: смещение входит
  // в КАЖДЫЙ элемент выхода одинаково по d и потому не выглядит как "мусор".
  float m_run[SMB][2], l_run[SMB][2], vb_run[SMB][2];
#pragma unroll
  for (int i = 0; i < SMB; ++i) { m_run[i][0] = m_run[i][1] = -CUDART_INF_F;
                                  l_run[i][0] = l_run[i][1] = 0.f; vb_run[i][0] = vb_run[i][1] = 0.f; }

  const int kEnd = CAUSAL ? (((q0 + BQ + causal_off) < Sk) ? (q0 + BQ + causal_off) : Sk) : Sk;
  // [СВЁРТКА СМЕЩЕНИЯ В РАНГ-1 -- §2] Разворот кодирует байт v как fp16-число 1152+v (0x6400 это
  // 1024, плюс сдвиг 128 от XOR со знаком). Значит первое умножение даёт
  //     S_raw[i][j] = sum_d Q[i][d]*(1152 + Kq[j][d]) = sum_d Q[i][d]*Kq[j][d] + 1152 * sum_d Q[i][d]
  // Второе слагаемое НЕ ЗАВИСИТ ОТ j -- это ранг-1 по строке запроса. Поэтому деквантование стоит
  // ОДНО вычитание на элемент S вместо поэлементной конверсии всей плитки K, а сумма строки Q
  // считается ОДИН раз на блок. Пропустить эту поправку = получить тихо неверный ответ, который
  // при малых значениях выглядит правдоподобно: смещение общее для всей строки, и softmax его
  // частично съедает, оставляя ошибку только там, где строка Q несимметрична.
  // Суммы строк Q считаются ПРЯМО ИЗ ГЛОБАЛЬНОЙ памяти: в разделяемой лежит лишь текущий кусок,
  // а поправка ранга-1 нужна по ВСЕМУ d. Один проход на блок, строку суммирует весь варп.
  __shared__ float sQsum[BQ];
  for (int r = warp; r < BQ; r += kWarps) {
    float acc = 0.f;
    if (q0 + r < Sq)
      for (int d = lane; d < D; d += 32) acc += __half2float(Q[(long)(q0 + r) * q_sM + d]);
#pragma unroll
    for (int off = 16; off; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if (lane == 0) sQsum[r] = acc;
  }
  __syncthreads();

  // [РАСФАЗИРОВКА ДВУХ РЕЗИДЕНТНЫХ БЛОКОВ]
  // Онлайн-софтмакс НЕ ЗАВИСИТ ОТ ПОРЯДКА плиток ключей: это сумма с бегущим максимумом, и любая
  // перестановка даёт тот же результат с точностью до округления. Значит соседние блоки можно
  // пустить по ключам НАВСТРЕЧУ друг другу. Тогда в любой момент они читают РАЗНЫЕ участки потока
  // и не сталкиваются на одном и том же пути подачи.
  // FA2_STAGGER: 0 -- как было; 1 -- нечётные блоки идут назад; 2 -- нечётные стартуют с середины
  // и заворачиваются (равномерное разнесение вместо зеркального).
  // [ЗАЗОР МЕЖДУ БЛОКАМИ] Профиль фаз (FA2_PHASE=1) даёт: перекладка масштабов и V -- 27.6% периода
  // и это ЧИСТЫЙ MIO, подача Q/K с первым умножением -- 39.0%, софтмакс 16.4%, барьер 17.1%.
  // Значит фазы РАЗЛИЧИМЫ по конвейерам, и есть смысл развести соседние блоки во времени так, чтобы
  // MIO-фаза одного попадала на счётную фазу другого. Механизм -- задержка нечётных блоков на
  // FA2_GAP тактов ПЕРЕД циклом; дальше они идут с этим сдвигом.
  // Это те самые "инструкции, которые ничего не делают": цена платится ОДИН раз на блок, а не на
  // итерацию, поэтому даже большой зазор стоит доли процента.
  // [ТОЧНАЯ ШАХМАТКА ПО SM] Роль блока определяется НЕ чётностью blockIdx: диспетчер раскидывает
  // блоки круговым перебором, поэтому вторым жильцом SM k становится блок k+80 (80 = число SM), а
  // 80 ЧЁТНОЕ -- значит k и k+80 ВСЕГДА одной чётности. Расфазировка по (blockIdx.x & 1) давала
  // обоим резидентным блокам ОДНУ И ТУ ЖЕ фазу и не проверяла ничего; прежний нулевой замер зазора
  // этим и объясняется.
  // Здесь роль берётся ТОЧНО: блок читает свой номер SM (%smid) и атомарно занимает место в
  // счётчике этого SM -- 0 у первого жильца, 1 у второго. Одна атомика на блок, не на итерацию.
  int ws_role = 0;
#if FA2_GAP
  if (sm_role) {
    __shared__ int _role;
    if (threadIdx.x == 0) {
      unsigned smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
      _role = atomicAdd(sm_role + (smid & 127), 1) & 1;
    }
    __syncthreads();
    ws_role = _role;
  }
  if (ws_role) {
    const long long t0 = clock64();
    while (clock64() - t0 < (long long)FA2_GAP) { /* пустой ход, только чтение часов */ }
  }
#endif
  const int nTiles = (kEnd + BK - 1) / BK;
  const int stag = (FA2_STAGGER && (blockIdx.x & 1)) ? 1 : 0;
  for (int tIdx = 0; tIdx < nTiles; ++tIdx) {
    int tSel = tIdx;
    if (stag) tSel = (FA2_STAGGER == 1) ? (nTiles - 1 - tIdx) : ((tIdx + nTiles / 2) % nTiles);
    const int kb = tSel * BK;
    __syncthreads();
    PH(0);   // ФАЗА 0: барьер входа
    // --- масштабы плитки: варп-однородны по строке, читаются один раз (мимо поэлементного пути)
    for (int j = tid; j < BK; j += kWarps * 32) {
      const int p = kb + j;
      sKs[j] = (p < Sk) ? Ks[p] : 0.f;
      sVs[j] = (p < Sk) ? Vs[p] : 0.f;
    }
    // --- V: ПЕРЕКЛАДКА БЕЗ РАЗВОРОТА. Байты кладутся как есть; разворот делает accumulate_bi8
    // во фрагментах в регистрах, между MMA.
    {
      const int rv = (kb + BK <= Sk) ? BK : (Sk - kb);
#pragma unroll
      for (int i = tid; i < BK * DV / 4; i += kWarps * 32) {
        const int r = i / (DV / 4), c = (i % (DV / 4)) * 4;
        uint32_t w = 0;
        if (r < rv) w = *reinterpret_cast<const uint32_t*>(Vq + (long)(kb + r) * k_sM + voff + c);
        *reinterpret_cast<uint32_t*>(sV + r * LDV + c) = w;
      }
    }
    // [СМЕЩЕНИЕ K -- В ОПЕРАНД C, ЦЕНА НОЛЬ (§4m, пункт 1)]
    // Инструкция считает D = A*B + C, и слагаемое в C бесплатно. Поправка -1152*sum(q) аддитивна и
    // НЕ зависит от столбца, поэтому её место -- ЗАТРАВКА аккумулятора, а не вычитание после MMA.
    // Прежняя редакция тратила на неё FFMA на КАЖДЫЙ элемент S и, что важнее, ставила его В ЦЕПЬ
    // между двумя умножениями, где сдвигать фазу некуда. Здесь она исчезает как инструкция:
    // s.clear() мы вызываем всё равно, меняется лишь то, ЧЕМ засеяно.
    // [РАЗВОРАЧИВАНИЕ -- НАШ ВЫБОР, А НЕ КОМПИЛЯТОРА]
    // Разбор живости SASS: из 204 задействованных регистров лишь 62 живут через весь мейнлуп, и это
    // РОВНО накопитель O. Регистров, которые держатся долго и почти не читаются, НОЛЬ -- то есть
    // компилятор ничего не "крадёт". Остальные ~142 это КОРОТКОЖИВУЩИЕ временные, и их столько
    // потому, что развёрнуты все циклы разом: по d (4 шага), по SMB/SNB и конвейер внутри
    // accumulate_bi8. Диапазоны жизни перекрываются, и пик растёт.
    // FA2_DUNROLL задаёт разворот цикла по d ЯВНО: 1 = не разворачивать (минимум временных,
    // максимум накладных на цикл), 4 = как было. Это размен, и его решает замер.
    PH(1);   // ФАЗА 1: перекладка масштабов и V (LSU + запись в разделяемую)
#if FA2_TPCROT
    // сосед по TPC начинает обход d с другого куска: адреса разные, строки те же
    unsigned _smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(_smid));
    const int dcRot = (_smid & 1) ? (D / 2) : 0;
#else
    const int dcRot = 0;
#endif
    WarpMma<SMB, SNB, BLayout::kKMajor> s;
    {   // §2a: суммы строк этого потока -- подряд, чтобы примитив читал их одним шагом
      const int rb = (lane & 1) | ((lane & 16) >> 2) | (lane & 8);
      float qs[SMB * 2];
#pragma unroll
      for (int i = 0; i < SMB; ++i)
#pragma unroll
        for (int t = 0; t < 2; ++t) qs[i * 2 + t] = sQsum[wm * SM + i * 16 + rb + 2 * t];
      seed_rank1<float, SMB, SNB>(s.acc, kI8Bias, qs, 2);
    }
FA2_UNROLL(FA2_DUNROLL)
    for (int dcI = 0; dcI < D; dcI += DC) {                     // K кусками по d
      const int dc = (dcI + dcRot) % D;                         // порядок обхода зависит от TPC
      __syncthreads();
      // КУСОК Q этого шага по d -- в разделяемую (fp16, операнд A остаётся полуслойным)
      constexpr int kRowStepQ = (kWarps * 32) / (DC / 8);
      const int r0q = tid / (DC / 8), c0q = (tid % (DC / 8)) * 8;
      const __half* pq = &Q[(long)(q0 + r0q) * q_sM + dc + c0q];
      __half* pdq = &sQ[r0q * LDQ + c0q];
      int rq = q0 + r0q;
#pragma unroll
      for (int u = 0; u < (BQ * DC / 8) / (kWarps * 32); ++u) {
        uint4 g = make_uint4(0, 0, 0, 0);
        if (rq < Sq) g = *reinterpret_cast<const uint4*>(pq);
        *reinterpret_cast<uint2*>(pdq)     = make_uint2(g.x, g.y);
        *reinterpret_cast<uint2*>(pdq + 4) = make_uint2(g.z, g.w);
        pq += (long)kRowStepQ * q_sM; pdq += kRowStepQ * LDQ; rq += kRowStepQ;
      }
      // ПЕРЕКЛАДКА БЕЗ РАЗВОРОТА: байты кладутся как есть, вчетверо меньше и байт, и инструкций
      // записи. Разворот делает accumulate_bi8 -- В РЕГИСТРАХ и МЕЖДУ MMA (сдвиг фазы уже внутри).
      // [ШИРОКАЯ ЗАГРУЗКА K, 27.08] Прежде нить брала ЧЕТЫРЕ байта (`uint32_t`), и аудит
      // SASS показал: из 73 загрузок 65 -- ТРИДЦАТИДВУХБИТНЫЕ. Тело связывает КАНАЛ ВЫДАЧИ
      // (`kanal_ili_cep.py`: ISSUE 100 %, ResMII 5842, RecMII 0), а канал выдачи лечится
      // ровно СНЯТИЕМ ИНСТРУКЦИЙ. Шаг байтовой строки K кратен 16 (см. проверку выше в этом
      // же теле), поэтому шестнадцатибайтовая загрузка ЗАКОННА: итераций вчетверо меньше,
      // и вчетверо меньше адресной арифметики на тот же объём.
      // Условие применимости проверяется на этапе сборки; иначе -- прежний путь по 4 байта.
      constexpr bool _широкоK = (DC % 16 == 0) && (LDK8 % 16 == 0)
                                && ((BK * DC / 16) % (kWarps * 32) == 0);
      const int rv = (kb + BK <= Sk) ? BK : (Sk - kb);
      if constexpr (_широкоK) {
        constexpr int kRowStepK16 = (kWarps * 32) / (DC / 16);
        const int r0k = tid / (DC / 16), c0k = (tid % (DC / 16)) * 16;
        const int8_t* pk = Kq + (long)(kb + r0k) * k_sM + dc + c0k;
        int8_t* pdk = sK + r0k * LDK8 + c0k;
        int rk = r0k;
#pragma unroll
        for (int u = 0; u < (BK * DC / 16) / (kWarps * 32); ++u) {
          uint4 w = make_uint4(0, 0, 0, 0);
          if (rk < rv) w = *reinterpret_cast<const uint4*>(pk);
          *reinterpret_cast<uint4*>(pdk) = w;
          pk += (long)kRowStepK16 * k_sM; pdk += kRowStepK16 * LDK8; rk += kRowStepK16;
        }
      } else {
        constexpr int kRowStepK = (kWarps * 32) / (DC / 4);
        const int r0k = tid / (DC / 4), c0k = (tid % (DC / 4)) * 4;
        const int8_t* pk = Kq + (long)(kb + r0k) * k_sM + dc + c0k;
        int8_t* pdk = sK + r0k * LDK8 + c0k;
        int rk = r0k;
#pragma unroll
        for (int u = 0; u < (BK * DC / 4) / (kWarps * 32); ++u) {
          uint32_t w = 0;
          if (rk < rv) w = *reinterpret_cast<const uint32_t*>(pk);
          *reinterpret_cast<uint32_t*>(pdk) = w;
          pk += (long)kRowStepK * k_sM; pdk += kRowStepK * LDK8; rk += kRowStepK;
        }
      }
      __syncthreads();
      s.template accumulate_bi8<DC>(sQ, LDQ, wm * SM, sK, LDK8, wn * SN, lane);   // Q -- текущий кусок
    }

    PH(2);   // ФАЗА 2: подача Q/K кусками + ПЕРВОЕ умножение (Tensor + MIO вперемешку)
    const int r0 = wm * SM, c0 = wn * SN;
    const int rowbase = (lane & 1) | ((lane & 16) >> 2) | (lane & 8);
    // [МАСШТАБЫ -- В РЕГИСТРЫ, А НЕ ИЗ РАЗДЕЛЯЕМОЙ НА КАЖДЫЙ ЭЛЕМЕНТ]
    // Первая редакция читала sKs[c0+c] на каждый элемент S и sVs[cc] дважды на каждую пару P --
    // около 8192 лишних обращений на плитку К ТОМУ САМОМУ ОГРАНИЧИТЕЛЮ, ради значений, которые
    // у нити ПОСТОЯННЫ: колонки аккумулятора заданы полосой. Это ровно та ошибка, что в §4v --
    // рычаг прикладывался к 4% трафика, а лишняя работа клалась в 96%.
    // МАСШТАБЫ ОСТАВЛЕНЫ В РАЗДЕЛЯЕМОЙ. Подъём их в регистры замерен как НОЛЬ (1.477 -> 1.473),
    // а стоил 8*SNB регистров на нить -- в ядре, где регистры и есть предел занятости, это чистый
    // убыток. Правило: рычаг, давший ноль, надо ОТКАТЫВАТЬ, а не оставлять "пусть будет".
    float mloc[SMB][2], lloc[SMB][2], vloc[SMB][2];
#pragma unroll
    for (int i = 0; i < SMB; ++i) { mloc[i][0] = mloc[i][1] = -CUDART_INF_F;
                                    lloc[i][0] = lloc[i][1] = 0.f; vloc[i][0] = vloc[i][1] = 0.f; }

    // МАСШТАБ КЛЮЧА поэлементно: s_k[j] НЕ выносится из софтмакса, потому что softmax идёт по j.
    // Это поэлементное плавающее -- свободный слот FP32 под HMMA (§4p).
    s.visit(lane, [&](float& v, int r, int c) {
      const int kk = kb + c0 + c;
      // ПОРЯДОК ОБЯЗАТЕЛЕН: сперва снять смещение разворота (ранг-1 по строке), и только потом
      // масштабировать. Обратный порядок даёт неверный ответ, проходящий проверку на малых числах.
      // смещение уже снято ЗАТРАВКОЙ аккумулятора -- здесь только масштаб
      v *= qk_scale * sKs[c0 + c];
      if ((CAUSAL && (q0 + r0 + r + causal_off) < kk) || kk >= Sk) v = -CUDART_INF_F;
      const int mb = r >> 4, sub = ((r & 15) - rowbase) >> 1;
      if (v > mloc[mb][sub]) mloc[mb][sub] = v;
    });
#pragma unroll
    for (int i = 0; i < SMB; ++i)
#pragma unroll
      for (int t = 0; t < 2; ++t) {
        float x = mloc[i][t];
        float y = __shfl_xor_sync(0xffffffffu, x, 2); x = x > y ? x : y;
        y = __shfl_xor_sync(0xffffffffu, x, 4);       x = x > y ? x : y;
        mloc[i][t] = x;
        if ((lane & 6) == 0) sRedM[(wm * SM + i * 16 + rowbase + 2 * t) * WN + wn] = x;
      }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < SMB; ++i)
#pragma unroll
      for (int t = 0; t < 2; ++t) {
        float x = -CUDART_INF_F;
#pragma unroll
        for (int w = 0; w < WN; ++w) { float y = sRedM[(wm * SM + i * 16 + rowbase + 2 * t) * WN + w]; x = x > y ? x : y; }
        mloc[i][t] = x;
      }
    PH(3);   // ФАЗА 3: маска, максимум, редукция максимума по варпам (ALU + барьер)
    float corr2[SMB][2], cnew[SMB][2];
#pragma unroll
    for (int i = 0; i < SMB; ++i)
#pragma unroll
      for (int t = 0; t < 2; ++t) {
        const float mo = m_run[i][t], mn = (mo > mloc[i][t]) ? mo : mloc[i][t];
        corr2[i][t] = (mo == -CUDART_INF_F) ? 0.f : exp2f(mo - mn);
        cnew[i][t]  = (mloc[i][t] == -CUDART_INF_F) ? 0.f : exp2f(mloc[i][t] - mn);
      }
    {
      const int cbase = (lane & 2) | ((lane & 4) << 1);
#pragma unroll
      for (int i = 0; i < SMB; ++i)
#pragma unroll
        for (int j = 0; j < SNB; ++j)
#pragma unroll
          for (int pr = 0; pr < 4; ++pr) {
            const int rg = (pr & 1) ? (4 + ((pr >> 1) << 1)) : ((pr >> 1) << 1);
            const int reg0 = rg, reg1 = rg + 1;
            const int row = i * 16 + rowbase + 2 * ((reg0 >> 1) & 1);
            const int col = j * 16 + cbase + 4 * ((reg0 >> 2) & 1);
            const int mb = i, sub = (reg0 >> 1) & 1;
            const float mm = mloc[mb][sub];
            const float v0 = s.acc[i][j][reg0], v1 = s.acc[i][j][reg1];
            const float p0 = (v0 == -CUDART_INF_F || mm == -CUDART_INF_F) ? 0.f : exp2f(v0 - mm);
            const float p1 = (v1 == -CUDART_INF_F || mm == -CUDART_INF_F) ? 0.f : exp2f(v1 - mm);
            lloc[mb][sub] += p0 + p1;
            // МАСШТАБ V СВЁРНУТ В УЖЕ СУЩЕСТВУЮЩЕЕ ДОМНОЖЕНИЕ -- цена НОЛЬ (§4m, пункт 4).
            // Операнд P всё равно строится и всё равно умножается на cnew; s_v[j] входит в тот же
            // множитель. Отдельного прохода по P для деквантования V НЕ СУЩЕСТВУЕТ.
            const float sc2 = cnew[mb][sub];
            const int cc = c0 + col;
            const float w0 = p0 * sc2 * sVs[cc], w1 = p1 * sc2 * sVs[cc + 1];
            // [ПОПРАВКА СЧИТАЕТСЯ ПО ОКРУГЛЁННОМУ ОПЕРАНДУ, А НЕ ПО ИСХОДНОМУ ЗНАЧЕНИЮ]
            // Умножение видит P' УЖЕ округлённым до fp16. Разница округления сама по себе ~5e-4,
            // но в накопителе она умножается на смещение 1152 и потому НЕ сокращается при
            // вычитании поправки. Первый прогон с fp32-значениями дал relL2 3.5e-3 при верной
            // алгебре -- ошибка была целиком в этом рассогласовании.
            const __half2 ph = __floats2half2_rn(w0, w1);
            vloc[mb][sub] += __half2float(__low2half(ph)) + __half2float(__high2half(ph));
            *reinterpret_cast<__half2*>(&sP[(r0 + row) * LDP + cc]) = ph;
          }
    }
    PH(4);   // ФАЗА 4: exp2 и запись P (SFU + разделяемая)
#pragma unroll
    for (int i = 0; i < SMB; ++i)
#pragma unroll
      for (int t = 0; t < 2; ++t) {
        float x = lloc[i][t], y = vloc[i][t];
        x += __shfl_xor_sync(0xffffffffu, x, 2); y += __shfl_xor_sync(0xffffffffu, y, 2);
        x += __shfl_xor_sync(0xffffffffu, x, 4); y += __shfl_xor_sync(0xffffffffu, y, 4);
        if ((lane & 6) == 0) {
          const int idx = (wm * SM + i * 16 + rowbase + 2 * t) * WN + wn;
          sRedS[idx] = x; sRedS[BQ * WN + idx] = y;
        }
      }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < SMB; ++i)
#pragma unroll
      for (int t = 0; t < 2; ++t) {
        float ss = 0.f, vv = 0.f;
#pragma unroll
        for (int w = 0; w < WN; ++w) {
          const int idx = (wm * SM + i * 16 + rowbase + 2 * t) * WN + w;
          ss += sRedS[idx]; vv += sRedS[BQ * WN + idx];
        }
        const float mo = m_run[i][t], mn = (mo > mloc[i][t]) ? mo : mloc[i][t];
        l_run[i][t] = l_run[i][t] * corr2[i][t] + ss * cnew[i][t];
        // vv УЖЕ содержит множитель cnew (он вложен в записанный операнд), поэтому здесь его нет.
        vb_run[i][t] = vv;                  // вклад ТОЛЬКО этой плитки: снимается сразу, см. ниже
        m_run[i][t] = mn;
      }
    __syncthreads();
    // [СМЕЩЕНИЕ V -- ПРЕДВЫЧЕСТЬ, А НЕ СНИМАТЬ ПОСЛЕ]
    // sum(P') известна ДО второго умножения (она пришла из редукции), поэтому вместо
    //     o = o*corr;  o += P'*(1152+V);  o -= 1152*sum(P')
    // пишем
    //     o = o*corr - 1152*sum(P');  o += P'*(1152+V)
    // Подстановка даёт то же самое, но вычитание СЛИВАЕТСЯ с домножением на corr в ОДИН FFMA:
    // минус восемь инструкций на блок 16x16 и, главное, минус звено в ЦЕПИ между умножениями.
        // corr2 передаётся НАПРЯМУЮ: отдельный массив corr был его побайтовой копией и стоил 4 регистра
    // на нить. vv тоже идёт напрямую -- vb_run хранил вклад ОДНОЙ плитки, который тут же и тратится.
    PH(5);   // ФАЗА 5: редукция сумм по варпам (барьер)
    rescale_fold<float, OM / 16, ON / 16>(o.acc, corr2, kI8Bias, vb_run);   // §2b
    o.template accumulate_bi8<BK>(sP, LDP, wm * OM, sV, LDV, wn * ON, lane);
    /* снято предвычитанием выше -- блок оставлен выключенным как след разбора
    // [СНИМАТЬ СМЕЩЕНИЕ НА КАЖДОЙ ПЛИТКЕ, А НЕ КОПИТЬ ДО ЭПИЛОГА]
    // Копить нельзя: накопитель тогда держит величину ~1152*sum(P), то есть на порядок больше
    // истинного O, и остаточная ошибка равна расхождению ПОРЯДКОВ СУММИРОВАНИЯ, умноженному на
    // 1152. При длинном контексте sum(P) растёт, и с ним растёт ошибка -- замерено: 6e-4 на
    // квадратных формах против 4.4e-4 при |эталон| 0.049, то есть до 0.9% относительной.
    // Снимая вклад КАЖДОЙ плитки сразу, держим накопитель в масштабе истинного O, и сокращение
    // идёт на малых числах. Цена -- OMB*ONB*8 вычитаний fp32 на плитку, ровно как у соседней
    // строки с corr, и они уезжают в свободный слот FP32 под HMMA (§4p).
    {
      constexpr int OMB = OM / 16, ONB = ON / 16;
#pragma unroll
      for (int i = 0; i < OMB; ++i)
#pragma unroll
        for (int j = 0; j < ONB; ++j)
#pragma unroll
          for (int rg = 0; rg < 8; ++rg) o.acc[i][j][rg] -= kI8Bias * vb_run[i][(rg >> 1) & 1];
    } */
  }
  PH_OUT(phase_acc);
  {                                                    // эпилог: O = acc / l, LSE = m + log(l)
    const int r0 = wm * OM;
    const int rowbase = (lane & 1) | ((lane & 16) >> 2) | (lane & 8);
    float inv[SMB][2];
#pragma unroll
    for (int i = 0; i < SMB; ++i)
#pragma unroll
      for (int t = 0; t < 2; ++t) inv[i][t] = (l_run[i][t] > 0.f) ? __frcp_rn(l_run[i][t]) : 0.f;
    constexpr int OMB = OM / 16, ONB = ON / 16;
    const int cbase0 = (lane & 2) | ((lane & 4) << 1);
#pragma unroll
    for (int i = 0; i < OMB; ++i)
#pragma unroll
      for (int j = 0; j < ONB; ++j)
#pragma unroll
        for (int rg = 0; rg < 8; ++rg) {
          const int sub = (rg >> 1) & 1;
          const int row = i * 16 + rowbase + 2 * sub;
          const int col = j * 16 + cbase0 + (rg & 1) + 4 * ((rg >> 2) & 1);
          if (q0 + r0 + row < Sq)   // смещение уже снято поплиточно -- здесь только нормировка
            O[(long)(q0 + r0 + row) * o_sM + voff + wn * ON + col] = __float2half(o.acc[i][j][rg] * inv[i][sub]);
        }
    if ((lane & 6) == 0 && wn == 0 && vslice == 0) {   // LSE пишет только первый срез
#pragma unroll
      for (int i = 0; i < SMB; ++i)
#pragma unroll
        for (int t = 0; t < 2; ++t) {
          const int row = r0 + i * 16 + rowbase + 2 * t;
          if (q0 + row < Sq)
            Lse[q0 + row] = (l_run[i][t] > 0.f) ? (m_run[i][t] * 0.6931471805599453f + __logf(l_run[i][t])) : -CUDART_INF_F;
        }
    }
  }
}

}  // namespace fwd_i8
}  // namespace fa2_sm70
