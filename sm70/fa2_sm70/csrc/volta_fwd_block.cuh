// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
//
// Volta (sm_70) FMHA FORWARD, block-level hand-written mainloop -- head_dim 256.
//
// Извлечён из tools/volta_fwd_mainloop.cu (прототип доводился на D=128) и перенастроен на геометрию
// рабочей модели: head_dim=256, причинное внимание, чанковый префилл. Замер против отгруженного пути
// cutlass на тех же формах (V100, частоты зафиксированы, H=4, causal):
//     S=4096   1.607 мс -> 1.076   x1.49        S=8192   5.266 мс -> 3.663   x1.44
//     S=16384 (H=2)      8.617 -> 7.140  x1.21 против прежней геометрии прототипа
//
// ПОЧЕМУ ЭТА ГЕОМЕТРИЯ. ncu на пути cutlass: конвейер L1/разделяемой памяти 61.6% при тензорном 38%,
// занятость 12.5% и НЕ лимитирует (её изменение не двигало время). Значит узкое место -- число
// загрузок операндов на одну MMA. Свип по этой оси дал НЕОЧЕВИДНЫЙ ответ:
//     плитка варпа 32x32, 1.00 загр/mma, 254 регистра, разлив 16 Б   4.752 мс
//     плитка варпа 32x16, 1.50 загр/mma, 178 регистров, разлив 0     3.663 мс   <- отгружено
//     плитка варпа 16x16, 2.00 загр/mma, 182 регистра,  разлив 0     4.400 мс
//     плитка варпа 16x32, 1.50 загр/mma, 204 регистра,  разлив 0     5.964 мс
// Самая плотная подача ПРОИГРЫВАЕТ: она стоит регистров, а разлив съедает выигрыш. Загрузки-на-mma
// нельзя оптимизировать в отрыве от регистрового бюджета -- это трёхсторонний компромисс Volta
// (docs/VOLTA_SM70.md), а не одна ось.
//
// Отгруженная конфигурация: BQ=64, BK=64, 8 варпов (2x4), плитка варпа 32x16, K подаётся кусками по
// DC=64, программный DMA (PIPE=3), разворот порядка блоков. 178 регистров, разлив 0, smem 84 КБ.
// Замерено в окрестности: DC=32 -> 3.760, DC=128 -> 3.723, широкая загрузка 128 бит -> 3.944,
// дополнение вместо свизла -> 4.828, без разворота порядка -> 3.988.
//
// ЧТО ДОБАВЛЕНО ПРИ ПЕРЕНОСЕ (прототип этого не умел, и каждый пункт -- отдельный источник тихой
// порчи, если его пропустить):
//   * GQA: голова запроса h читает KV-голову h/kv_ratio;
//   * батч и ЯВНЫЕ шаги: BSHD (то, что даёт и ждёт vLLM) и BHSD без единого транспонирования;
//   * причинная маска выровнена ПО ПРАВОМУ НИЖНЕМУ углу (causal_off = Sk - Sq) -- прототип
//     предполагал Sq == Sk;
//   * рваные Sq и Sk: короткая строка читается НУЛЁМ (иначе мусор попадает в бегущий максимум),
//     ключи за Sk маскируются в -inf, запись выхода и LSE ограничена по Sq.
#pragma once
// FA2_QCHUNK=1 -- подавать Q КУСКАМИ по d (буфер BQ x DC вместо BQ x D). Снимает ~12 КБ, что вместе
// с DC=32 опускает разделяемую под порог двух резидентных блоков (48 КБ). Перечитывание Q идёт из
// L2 (DRAM по профилю занят на 2-4%). FA2_FUNROLL -- разворот цикла по d, задаётся ЯВНО: обычный
// #pragma unroll не раскрывает макросы (предупреждение #20169) и молча игнорируется.
#ifndef FA2_FUNROLL
#define FA2_FUNROLL 4
#endif
#define FA2B_DO_PRAGMA(x) _Pragma(#x)
#define FA2B_UNROLL_(n)   FA2B_DO_PRAGMA(unroll n)
#define FA2B_UNROLL(n)    FA2B_UNROLL_(n)
// FA2_FWD_TAILGUARD -- ограничение загрузок K/V ХВОСТОВОЙ плитки ключей по Sk (см. `rv` в мейнлупе).
// Ноль воспроизводит прежнюю сборку БЕЗ защиты БУКВАЛЬНО (макрос сводится к тождеству) и оставлен
// ТОЛЬКО как след A/B: без защиты ядро читает за конец тензора V и даёт плавающий NaN на рваном Sk.
// Отгружать можно лишь с 1.
// [ПОЧЕМУ ЗАЖИМ, А НЕ ПРЕДИКАТ] Первая редакция подавала строку за Sk НУЛЁМ: `uint4 g = 0; if (r<rv)
// g = ldg(...)`. Численно верно, но ЗАМЕРЕНО дорого -- 0.929x (7.7% на всём спектре, от 0.910 до
// 0.950): предикат превращает LDG в условную и добавляет выбор ЧЕТЫРЁХ регистров uint4 на каждую из
// 16 загрузок плитки. Зажим номера строки стоит ОДНУ целочисленную операцию на адрес и не трогает
// саму загрузку. Он законен, потому что вклад этих строк РОВНО ноль (их столбцы маскируются в -inf
// -> P = 0), а значит подать туда можно ЛЮБОЕ КОНЕЧНОЕ число -- нужен лишь адрес внутри тензора.
#ifndef FA2_FWD_TAILGUARD
#define FA2_FWD_TAILGUARD 1
#endif
#if FA2_FWD_TAILGUARD
#define FA2_KROW(r) (((r) < rv) ? (r) : 0)   // строка за Sk -> ЛЮБАЯ валидная (её P == 0)
#else
#define FA2_KROW(r) (r)
#endif
#include <cuda_fp16.h>
#include <cstdint>
#include <math_constants.h>
#include "gemm/volta_warp_mma.h"
#include "gemm/volta_prims.h"   // §1f: маршруты загрузки (подсказка шины, предвыборка, широкая запись)

namespace fa2_sm70 {
namespace block_fwd {

using fa2_sm70::WarpMma;
using fa2_sm70::BLayout;
using fa2_sm70::ALayout;
using fa2_sm70::prims::ldg_stream;   // §1f: маршруты загрузки
using fa2_sm70::prims::ldg_keep;
using fa2_sm70::prims::prefetch_l2;
using fa2_sm70::prims::sts128;

// [МАРШРУТЫ] ROUTE -- битовая маска: НЕ геометрия, а выбор КАНАЛА для той же самой работы.
// Геометрию (BQ/BK/сетку варпов) трогать бессмысленно -- она уже вылизана свипом; рычаг в том,
// каким маршрутом идут те же байты и на каком конвейере считаются те же значения (кукбук §0).
//   бит 0 (1) -- широкая запись V в разделяемую: одна STS.128 вместо двух STS.64 (§1f, §4n)
//   бит 1 (2) -- подсказка шины .cs на потоковых загрузках K/V (§4k) -- ОСТОРОЖНО с L2-переиспользованием
//   бит 2 (4) -- предвыборка prefetch.global.L2 следующего куска (§4n), безопасна всегда
// [АДАПТИВНАЯ РАЗРЕЖЕННОСТЬ] SPARSE -- КАК подан список оставленных блоков ключей. Плотное внимание
// (SPARSE=0) остаётся ПОБИТОВО тем же: цикл по ключам как был счётным, так и остался, только индекс
// плитки берётся из подачи, а не из счётчика. Форматы:
//   0 -- нет механизма (плотное/причинное, как отгружено)
//   1 -- СПИСОК: kmask[row][t] = номер оставленной плитки ключей, kcnt[row] -- сколько их
//   2 -- БИТОВАЯ КАРТА: kmask[row] -- слова по 32 бита, бит j = плитка j оставлена; kcnt[row] -- popcount
// row = (b*nheads + h)*nqb + номер блока запросов. Адрес чтения ОДНОРОДЕН по варпу (все нити блока
// читают один и тот же элемент), поэтому это широковещательная загрузка, а не расходящаяся -- ровно
// тот случай, который в §4k стоит 1 транзакцию вместо 16.
// JOBQ -- ОЧЕРЕДЬ ЗАДАНИЙ вместо трёхмерной сетки: блок берёт своё (блок запросов, голова) из
// jobq[blockIdx.x]. Нужна против НЕРАВНОМЕРНОЙ НАГРУЗКИ: при поголовной плотности задания в одной
// сетке весят по-разному, и порядок выдачи (самые тяжёлые -- первыми) определяет длину хвоста.
template <int BQ, int BK, int D, int WM, int WN, bool CAUSAL, int DC = D, bool SWZ = false, int PIPE = 0, bool QREG = false, int MINB = 0, int QPRE = 0, int REV = 0, int PADH = 0, int WIDE = 0, int PS = 0, int ROUTE = 0, int QCH = 0, int SPARSE = 0, int JOBQ = 0>
// [КОМПИЛЯТОР] Без второго аргумента __launch_bounds__ nvcc САМ выбирает цель по блокам на SM. Освободив
// 26 регистров (гипотеза 1), он ужался до 128 регистров ради ЧЕТЫРЁХ блоков и добавил 24 Б разлива —
// время 0.189 -> 0.202. Цель задаём мы: MINB=3 при 128 потоках даёт 65536/(128*3) = 170 регистров.
__global__ __launch_bounds__(WM * WN * 32, MINB ? MINB : 1) void volta_fwd_block(
    const __half* __restrict__ Q, const __half* __restrict__ Kk, const __half* __restrict__ Vv,
    __half* __restrict__ O, float* __restrict__ Lse,
    int Sq, int Sk, float qk_scale, int kv_ratio, int causal_off,
    long q_sM, long q_sH, long q_sB, long k_sM, long k_sH, long k_sB,
    long o_sM, long o_sH, long o_sB, long lse_sH, long lse_sB,
    const int32_t* __restrict__ kmask, const int32_t* __restrict__ kcnt,
    int mask_ld, int nqb, int nheads, const int32_t* __restrict__ jobq) {
  constexpr int kWarps = WM * WN, PAD = PADH ? PADH : 4;
  // [SMEM] K подаётся кусками по d: буфер BK x DC вместо BK x D. Профиль показал упор в 1 блок/SM
  // (smem 59.4 КБ), а инструкций после рукописного мейнлупа стало в 7 раз меньше — прятать задержку
  // нечем, и занятость СТАЛА рычагом (в отличие от cutlass, где лишние варпы вредят). Порог двух
  // блоков — 48 КБ. Цена: D/DC дополнительных барьеров, у нас их было 5 против ~20 у cutlass.
  // [ПЛОТНОСТЬ, шаг 2] При SWZ строка Q без дополнения (степень двойки) и с XOR-перемешиванием:
  // адрес кратен 16 Б, и ОДНА LDS.128 покрывает ДВА шага k. В backward это проиграло регистрам
  // (255 при разливе), здесь бюджет есть: 123 регистра, разлив ноль (docs §16, §27).
  // QCH -- ШАБЛОННЫЙ параметр, а не макрос: макрос один на всю единицу трансляции, а конфигураций
  // в ней две, и при выключенном макросе раскладка расходилась с выделенной памятью -> запись за
  // буфер и illegal memory access. Ровно этот дефект и был пойман.
  constexpr int LDQ = QCH ? (DC + PAD) : (SWZ ? D : (D + PAD)), LDP = BK + PAD;
  // [ШАГ СТРОКИ K КРАТЕН 16 БАЙТАМ, 27.08] Прежде LDK = DC + PAD = 68 полуслов, и в самом
  // ядре стояло: «у K такой возможности нет» -- запись шла ДВУМЯ STS.64 вместо одной
  // STS.128. Инструмент `kanal_ili_cep.py` показал, что тело связывает КАНАЛ MIO (ResMII
  // 8928 против тензорных 8192, RecMII = 0 -- цепи нет вовсе), то есть лечится ТОЛЬКО
  // снятием работы с разделяемой памяти, а половина её записей -- это лишняя половина K.
  // Дополнение 8 даёт LDK = 72, шаг строки 144 байта -- кратен 16, и STS становится ОДНОЙ.
  // Выключатель на случай худших банков: FA2SM70_LDK_PAD (умолчание 8, прежнее -- 4).
#ifndef FA2SM70_LDK_PAD
#define FA2SM70_LDK_PAD 8
#endif
  constexpr int PADK = FA2SM70_LDK_PAD;
  constexpr int LDK = DC + PADK;
  // [ШАГ СТРОКИ V ПРОТИВ БАНКОВ] Дополнение 16, а не 4, и это не вкусовщина, а арифметика банков.
  // У n-мажорного операнда адрес полосы равен (lane&3)*(LDV/2) + nbase/2, где nbase из {0,4,8,12}.
  // Нужно, чтобы вклад СТРОКИ был крупнее вклада полосы: при (LDV/2)%32 == 8 получаем
  // {0,8,16,24} + {0,2,4,6} = ШЕСТНАДЦАТЬ различных адресов, и 8-байтовый доступ кроет ровно 32
  // банка. При прежнем LDV = D+4 = 260 было (LDV/2)%32 = 2, оба слагаемых пробегали {0,2,4,6},
  // суммы накладывались вчетверо.
  // Замерено ncu на прежней версии: 158.7 млн конфликтов на 708.8 млн волновых фронтов = 22.4%,
  // mio_throttle 0.94 -- ПЕРВЫЙ простой. Свип дополнения (S=8192 H=4 / S=16384 H=2, мс):
  //     4 -> 3.666 / 7.146      8 -> 3.618      32 -> 3.619      16 -> 3.523 / 6.865  (1.041x)
  // Ранжирование предсказано моделью до замера: 32 даёт (lane&3)*16 = {0,16,0,16}, всего ДВА
  // различных значения, поэтому не лучше восьмёрки. Цена: +4.6% разделяемой памяти под V.
  // PADV=8 вместо 16 при подаче Q кусками: ровно тот килобайт, которого не хватало, чтобы взять
  // DC=64 и не платить дроблением куска. Цена по прежнему свипу -- 3.618 против 3.523 (2.7%),
  // тогда как DC=32 стоит 3.760 против 3.663. Меняем БОЛЬШУЮ плату на МЕНЬШУЮ.
  constexpr int PADV = QCH ? 8 : 16;
  constexpr int LDV = D + PADV;
  constexpr int SM = BQ / WM, SN = BK / WN;     // плитка варпа для S
  constexpr int OM = BQ / WM, ON = D / WN;      // плитка варпа для O
  constexpr int SMB = SM / 16, SNB = SN / 16;   // блоков 16x16 в плитке S
  static_assert(SM % 16 == 0 && SN % 16 == 0 && OM % 16 == 0 && ON % 16 == 0, "плитки кратны 16");
  static_assert(OM == SM, "O и S делят строки: одна и та же сетка варпов по m");

  extern __shared__ __half smem[];
  __half* sQ = smem;                    // [BQ][LDQ] резидентно
  __half* sP = sQ + BQ * LDQ;           // [BQ][LDP] — P между двумя gemm
  // [ГЛУБИНА 2] при PIPE==2 K/V двойной буфер: пока считается плитка i, в полёте загрузки i+2,
  // а плитка i+1 уже лежит в smem. Цена — вторая пара буферов; занятость измеренно не рычаг (§25,§27).
  // ВНИМАНИЕ: PIPE==2 (глубина 2) НЕ РАБОТАЕТ (relL2 nan) И измеренно МЕДЛЕННЕЕ глубины 1
  // (0.219 против 0.189; 0.239 против 0.205): удвоение smem стоит дороже дополнительного перекрытия.
  // Оставлено только как след эксперимента — НЕ включать без переделки.
  constexpr int kBuf = (PIPE == 2) ? 2 : 1;
  __half* sK = sP + BQ * LDP;                        // [kBuf][BK][LDK]
  __half* sV = sK + kBuf * BK * LDK;                 // [kBuf][BK][LDV]
  // [ГОНКА] Один буфер использовался ДВАЖДЫ за итерацию: сначала под максимумы, затем под суммы —
  // а барьера между ЧТЕНИЕМ максимумов и ЗАПИСЬЮ сумм не было. Ушедший вперёд варп затирал максимумы,
  // которые отставший ещё не прочитал. Проявлялось ТОЛЬКО при 2 блоках на SM (S=2048 И H=4): там
  // планировщик разводит варпы сильнее. Два раздельных буфера дешевле лишнего барьера (512 Б).
  __shared__ float sRedM[BQ * WN];       // максимумы по строке
  __shared__ float sRedS[BQ * WN];       // суммы по строке

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp / WN, wn = warp % WN;
  // [ВОЛНЫ] При причинной маске блок q0=0 считает ОДНУ плитку ключей, блок q0=Sq-BQ -- все Sq/BK.
  // Диспетчер раздаёт блоки по возрастанию blockIdx, поэтому в ОГРЫЗОК последней волны попадают самые
  // ТЯЖЁЛЫЕ блоки: 256 блоков на 240 слотов -> хвост из 16 блоков с максимальной работой на пустой
  // машине (эфф волн 53%). REV=1 разворачивает порядок: тяжёлые уходят в первую полную волну,
  // в огрызке остаются самые лёгкие.
  // [ОЧЕРЕДЬ ЗАДАНИЙ] При JOBQ задание берётся из подачи: упаковка (номер блока запросов << 12) | bh,
  // где bh = b*nheads + h. 12 бит на bh хватает на 4096 (b*h у нас <= 96), 19 -- на блоки запросов.
  // Порядок в очереди -- рычаг: она сортируется по УБЫВАНИЮ веса задания (LPT), и хвост волны
  // набирается из самых лёгких. Без неё диспетчер раздаёт по возрастанию blockIdx.
  const int qblk_ = JOBQ ? (jobq[blockIdx.x] >> 12)
                         : (REV ? (int)(gridDim.x - 1 - blockIdx.x) : (int)blockIdx.x);
  const int q0 = qblk_ * BQ;
  const int bh_ = (SPARSE || JOBQ) ? (JOBQ ? (jobq[blockIdx.x] & 0xfff)
                                           : (int)(blockIdx.z * gridDim.y + blockIdx.y))
                                   : 0;
  {   // GQA: query head h reads KV head h/kv_ratio. Strides are explicit so BSHD (vLLM hands us
      // [B,Sq,H,D] and wants it back) and BHSD both work without a transpose anywhere.
    const long h = JOBQ ? (bh_ % nheads) : (long)blockIdx.y;
    const long b = JOBQ ? (bh_ / nheads) : (long)blockIdx.z;
    const long hkv = h / kv_ratio;
    Q += b * q_sB + h * q_sH; O += b * o_sB + h * o_sH;
    Kk += b * k_sB + hkv * k_sH; Vv += b * k_sB + hkv * k_sH;
    Lse += b * lse_sB + h * lse_sH;
  }
#pragma unroll
  for (int i = tid; QCH ? false : (i < BQ * D / 8); i += kWarps * 32) {   // Q целиком (когда не кусками)
    const int r = i / (D / 8), c = (i % (D / 8)) * 8;
    // RAGGED Sq: the last query tile is short in a serving chunk. A short row must read as ZERO,
    // not as whatever follows the tensor -- its softmax row is discarded at the epilogue, but a NaN
    // or an inf here would poison the whole tile through the running max.
    uint4 g = make_uint4(0, 0, 0, 0);
    if (q0 + r < Sq) g = *reinterpret_cast<const uint4*>(&Q[(long)(q0 + r) * q_sM + c]);
    const int cs = SWZ ? ((((c >> 3) ^ (r & 15)) << 3) | (c & 7)) : c;   // перемешивание вместо дополнения
    __half* d = &sQ[r * LDQ + cs];
    *reinterpret_cast<uint2*>(d) = make_uint2(g.x, g.y);
    *reinterpret_cast<uint2*>(d + 4) = make_uint2(g.z, g.w);
  }

  WarpMma<OM / 16, ON / 16, BLayout::kNMajor> o; o.clear();      // O += P*V, V это [k][n] -> n-мажорный
  // Онлайн-состояние: у потока 2 строки на каждый m-блок плитки S.
  float m_run[SMB][2], l_run[SMB][2];
#pragma unroll
  for (int i = 0; i < SMB; ++i) { m_run[i][0] = m_run[i][1] = -CUDART_INF_F; l_run[i][0] = l_run[i][1] = 0.f; }

  // Конвейер грузит K/V ЦЕЛИКОМ по d (иначе нечего держать в регистрах между итерациями), поэтому
  // он несовместим с подачей K кусками. Раньше сочетание молча портило память: буфер на DC+PAD,
  // а запись на D. Ловим на этапе компиляции.
  static_assert(!(PIPE == 1 || PIPE == 2) || DC == D, "конвейер по kb требует DC == D");
  static_assert(PIPE != 3 || (BK * DC / 8) % (WM * WN * 32) == 0, "кусок должен делиться на нити нацело");
  // XOR-свизл меняет РАСКЛАДКУ sQ, а пути предзагрузки (QREG/QPRE) и конвейера читают её обычной
  // адресацией. Сочетание молча даёт неверный результат (relL2 1e-01). Третий случай подряд, когда
  // несовместимость шаблонных параметров проявляется как «геометрия не работает» — запрещаем статически.
  static_assert(!SWZ || (!QREG && !QPRE && !PIPE), "свизл несовместим с предзагрузкой Q и конвейером");
  constexpr int kStage = PIPE ? (BK * D / 8) / (WM * WN * 32) : 1;
  // [ДОЗИРОВАННЫЙ КОНВЕЙЕР] Конвейер был «всё или ничего»: либо ВСЕ kStage векторов плитки идут через
  // регистры, либо ни один. Замер налога: BQ=32/4 варпа 124 -> 168 (+44), BQ=16/2 варпа 124 -> 234
  // (+110) -- налог обратно пропорционален числу нитей, потому что на нить приходится (BK*D/8)/нитей
  // векторов. Но перекрытие не обязано быть полным: PS векторов ведём через регистры (LDG рано),
  // остальные грузим и кладём на месте в поздней записи. Освободившиеся регистры идут в QPRE --
  // единственный измеренно работающий рычаг (+16%), который упирался ровно в регистры (обрыв 28->32).
  constexpr int kPS = (PS && PS < kStage) ? PS : kStage;
  uint4 stK[kPS], stV[kPS];
  // [ПРЯМАЯ АДРЕСАЦИЯ] SASS показал IMAD 185 из 1384 инструкций (13.4%, больше всего после HMMA) —
  // это 64-битное умножение `(long)(kb + r) * D` на КАЖДОЙ итерации загрузки. Считаем базу ОДИН раз
  // и двигаем её на константу BK*D за итерацию: ни одного умножения в цикле.
  const __half* pKs[kStage]; const __half* pVs[kStage];
  if constexpr (PIPE == 1 || PIPE == 2) {
#pragma unroll
    for (int u = 0; u < kStage; ++u) {
      const int i = tid + u * kWarps * 32;
      const int r = i / (D / 8), c = (i % (D / 8)) * 8;
      pKs[u] = Kk + (long)r * k_sM + c;   // база: считается ОДИН раз на поток
      pVs[u] = Vv + (long)r * k_sM + c;
    }
  }
  // CAUSAL IS BOTTOM-RIGHT ALIGNED (causal_off = Sk - Sq), which is what chunked prefill means: the
  // queries of this chunk sit at the END of the key range, not at its start. The prototype assumed
  // Sq == Sk and masked q < k; with a 8192-query chunk against 120K keys that would mask away
  // everything but the first 8192 keys -- exactly the mistake torch SDPA's is_causal makes here.
  const int kEnd = CAUSAL ? (((q0 + BQ + causal_off) < Sk) ? (q0 + BQ + causal_off) : Sk) : Sk;
  if constexpr (PIPE == 1 || PIPE == 2) {      // пролог: первая плитка ключей в smem (буфер 0)
#pragma unroll
    for (int i = tid; i < BK * D / 8; i += kWarps * 32) {
      const int r = i / (D / 8), c = (i % (D / 8)) * 8;
      const uint4 gk = (r < Sk) ? *reinterpret_cast<const uint4*>(&Kk[(long)r * k_sM + c]) : make_uint4(0,0,0,0);
      const uint4 gv = (r < Sk) ? *reinterpret_cast<const uint4*>(&Vv[(long)r * k_sM + c]) : make_uint4(0,0,0,0);
      __half* pk = &sK[r * LDK + c]; __half* pv = &sV[r * LDV + c];
      *reinterpret_cast<uint2*>(pk) = make_uint2(gk.x, gk.y);
      *reinterpret_cast<uint2*>(pk + 4) = make_uint2(gk.z, gk.w);
      *reinterpret_cast<uint2*>(pv) = make_uint2(gv.x, gv.y);
      *reinterpret_cast<uint2*>(pv + 4) = make_uint2(gv.z, gv.w);
    }
  }
  // [Q В РЕГИСТРАХ] фрагменты A грузятся ОДИН раз на весь блок запросов
  WarpMma<SMB, SNB, BLayout::kKMajor> sTpl;
  uint32_t qfrag[QREG ? (D / 4) : (QPRE ? QPRE : 1)][SMB][2];
  if constexpr (QREG) { __syncthreads(); sTpl.template preload_a<D>(qfrag, sQ, LDQ, wm * SM, lane); }
  else if constexpr (QPRE) { __syncthreads(); sTpl.template preload_n<QPRE>(qfrag, sQ, LDQ, wm * SM, lane); }
  int buf = 0;
  // [ПОДАЧА МАСКИ] Цикл по ключам остаётся СЧЁТНЫМ и при разреженности: число итераций известно до
  // входа (kcnt), меняется только откуда берётся номер плитки. Это важно не из вкусовщины: счётный
  // цикл ptxas раскручивает и снижает индекс до сложения, а `while (найди следующий бит)` -- нет.
  // Битовая карта поэтому тоже несёт счётчик: карта даёт НОМЕР, счётчик -- ЧИСЛО ИТЕРАЦИЙ.
  const int nkbC = (kEnd + BK - 1) / BK;            // сколько плиток дало бы плотное внимание
  const int32_t* __restrict__ mrow = nullptr;
  int nIter = nkbC;
  if constexpr (SPARSE) {
    const long row = (long)bh_ * nqb + qblk_;
    mrow = kmask + row * (long)mask_ld;
    nIter = kcnt[row];                              // ОДНОРОДНЫЙ адрес: одна транзакция на блок
  }
  int bcur = 0;                                     // курсор битовой карты (SPARSE==2)
  // [ПОЧЕМУ СПИСОК ДОРОГ, И КАК ЭТО ЛЕЧИТСЯ] Замер: список стоит +7.25% при плотности 100%, битовая
  // карта +0.40%. По ИНСТРУКЦИЯМ разница ничтожна (SASS: +1..4 LDG на всё ядро), значит платится не
  // счёт, а ЗАВИСИМОСТЬ: номер плитки нужен НЕМЕДЛЕННО -- из него считаются адреса загрузок K и V,
  // -- поэтому задержка глобальной памяти встаёт в критический путь КАЖДОЙ итерации по ключам.
  // SPARSE==3 -- тот же список, но номер следующей плитки выдаётся В РЕГИСТР за итерацию до
  // использования: на её сокрытие есть вся итерация мейнлупа (~11.8 тыс. тактов против ~600 задержки).
  int nxt = 0;
  if constexpr (SPARSE == 3) { if (nIter > 0) nxt = mrow[0]; }
  for (int it = 0; it < nIter; ++it) {
    int kb;
    if constexpr (SPARSE == 3) {
      kb = nxt * BK;
      if (it + 1 < nIter) nxt = mrow[it + 1];       // LDG РАНО: результат нужен только в следующем круге
    } else if constexpr (SPARSE == 1) {
      kb = mrow[it] * BK;                           // ОДНОРОДНЫЙ адрес -> широковещательная LDG
    } else if constexpr (SPARSE == 2) {
      // Поиск следующего установленного бита. Слово читают все нити блока по ОДНОМУ адресу.
      const uint32_t* bm = reinterpret_cast<const uint32_t*>(mrow);
      for (;;) {
        const uint32_t w = bm[bcur >> 5] >> (bcur & 31);
        if (w) { bcur += __ffs((int)w) - 1; break; }
        bcur = (bcur + 32) & ~31;
      }
      kb = bcur * BK; ++bcur;
    } else {
      kb = it * BK;
    }
    // [РВАНЫЙ Sk -- КОРЕНЬ ПЛАВАЮЩЕГО NaN] Хвостовая плитка ключей заходит ЗА Sk (777 = 12*64 + 9).
    // Читать там глобальную память нельзя по двум РАЗНЫМ причинам:
    //   1) это обращение за конец тензора -- неопределённое поведение и риск illegal access;
    //   2) численно: столбцы за Sk маскируются в -inf (см. visit ниже), их P РОВНО ноль, а второй
    //      gemm всё равно считает 0 * V[хвост]. Если в хвосте лежит inf или NaN, то 0*NaN = NaN и он
    //      поражает ВСЮ строку выхода. А лежит там то, что кэширующий аллокатор оставил от прошлого
    //      тензора: например fp32 -inf эталонной маски, чьи СТАРШИЕ полуслова 0xFF80 -- это NaN fp16.
    // Отсюда «в наборе воспроизводится, в одиночку нет»: в одиночном прогоне за концом свежая
    // обнулённая страница, 0*0 = 0, и дефект не виден. Мусор попадает только в последнюю kv-голову
    // последнего элемента батча -- у остальных за концом лежат данные СЛЕДУЮЩЕЙ головы, они конечны.
    // rv -- сколько строк плитки лежит ВНУТРИ Sk; остальные подаются НУЛЁМ. Та же защита уже стоит
    // в байтовых ядрах (volta_fwd_i8.cuh, volta_fwd_ws.cuh) -- потому e5m2 на этих формах и конечен.
#if FA2_FWD_TAILGUARD
    const int rv = (kb + BK <= Sk) ? BK : (Sk - kb);
#endif
    __syncthreads();
    __half* sKc = sK + (PIPE == 2 ? buf : 0) * BK * LDK;          // текущий буфер
    __half* sVc = sV + (PIPE == 2 ? buf : 0) * BK * LDV;
    __half* sKn = sK + (PIPE == 2 ? (buf ^ 1) : 0) * BK * LDK;    // куда писать следующую
    __half* sVn = sV + (PIPE == 2 ? (buf ^ 1) : 0) * BK * LDV;
    if constexpr (PIPE == 0 || PIPE == 3) {    // PIPE==3 тоже грузит V здесь: конвейер у него по КУСКАМ K
      // [МАРШРУТ V] Шаг строки LDV = D + 16 = 272 полуслова = 544 БАЙТА, кратно 16 -- значит запись
      // ЗАКОННО идёт одной STS.128 вместо двух STS.64. Полоса та же, инструкций вдвое меньше, а MIO
      // считает ИНСТРУКЦИИ (§4n). У K такой возможности нет: LDK = 68 полуслов = 136 Б, не кратно.
      static_assert(!(ROUTE & 1) || ((D + PADV) * 2) % 16 == 0, "широкая запись V требует шаг, кратный 16 Б");
#pragma unroll
      // [АДРЕСА В ПРОЛОГ, 27.08] Разложение `i -> (r, c)` считалось КАЖДУЮ итерацию: деление
      // и остаток плюс 64-битный адрес. Аудит SASS показал 29.6 команд адресной арифметики
      // на ОДНУ загрузку и 12.4 % слотов выдачи на адреса при HMMA 39.8 % -- ровно та подпись,
      // которую уже лечили на плотном fp16-теле (задача 197). Когда шаг цикла кратен строке
      // (kWarps*32 % (D/8) == 0 -- у нас 128 % 32), столбец НЕ МЕНЯЕТСЯ, а строка растёт на
      // постоянную величину. Тогда разложение делается ОДИН раз, а в цикле остаётся сложение.
      // Преобразование ТОЖДЕСТВЕННО: то же множество (r, c), тот же порядок.
      constexpr int _шагV = (kWarps * 32) / (D / 8);
      constexpr bool _ровноV = ((kWarps * 32) % (D / 8)) == 0;
      const int _r0V = tid / (D / 8), _c0V = (tid % (D / 8)) * 8;
      int _rV = _r0V;
#pragma unroll
      for (int i = tid; i < BK * D / 8; i += kWarps * 32, _rV += _шагV) {      // V целиком, векторно
        const int r = _ровноV ? _rV : (i / (D / 8));
        const int c = _ровноV ? _c0V : ((i % (D / 8)) * 8);
        if constexpr (ROUTE & 4) {   // предвыборка СЛЕДУЮЩЕЙ плитки: безопасна, ничего не вытесняет
          if (kb + BK + r < Sk) prefetch_l2(&Vv[(long)(kb + BK + r) * D + c]);   // за Sk не трогаем
        }
        const int rq = FA2_KROW(r);                 // не выходим за конец V
        const uint4 gv = (ROUTE & 2) ? ldg_stream(&Vv[(long)(kb + rq) * D + c])
                                     : ldg_keep(&Vv[(long)(kb + rq) * D + c]);
        __half* pv = &sV[r * LDV + c];
        if constexpr (ROUTE & 1) {
          sts128(pv, gv);                                        // ОДНА инструкция вместо двух
        } else {
          *reinterpret_cast<uint2*>(pv) = make_uint2(gv.x, gv.y);
          *reinterpret_cast<uint2*>(pv + 4) = make_uint2(gv.z, gv.w);
        }
      }
    }
    WarpMma<SMB, SNB, BLayout::kKMajor> s; s.clear();            // S = Q*K^T, K это [n][k] -> k-мажорный
    if constexpr (PIPE == 1 || PIPE == 2) {
      // [ДВЕ ЦЕПОЧКИ] Загрузки СЛЕДУЮЩЕЙ плитки ключей выдаются ЗДЕСЬ, в регистры, до всякого счёта
      // по текущей. Запись в smem — в конце итерации. Это разнесение LDG и STS: в backward именно
      // оно дало +20-46%, тогда как размазывание выдачи по слотам — ноль (docs §13).
      if constexpr (PIPE == 2) {
        // Порядок обязателен: СНАЧАЛА выгрузить плитку i+1 из регистров в ДРУГОЙ буфер (со счётом по
        // текущему она не конфликтует, барьер не нужен), и только ПОТОМ выдавать загрузки плитки i+2 —
        // иначе i+2 затирает ещё не записанную i+1. Первая сборка падала именно на этом (relL2 nan).
        if (kb + BK < kEnd) {
#pragma unroll
          for (int u = 0; u < kStage; ++u) {
            const int i = tid + u * kWarps * 32;
            const int r = i / (D / 8), c = (i % (D / 8)) * 8;
            __half* pk = &sKn[r * LDK + c]; __half* pv = &sVn[r * LDV + c];
            *reinterpret_cast<uint2*>(pk) = make_uint2(stK[u].x, stK[u].y);
            *reinterpret_cast<uint2*>(pk + 4) = make_uint2(stK[u].z, stK[u].w);
            *reinterpret_cast<uint2*>(pv) = make_uint2(stV[u].x, stV[u].y);
            *reinterpret_cast<uint2*>(pv + 4) = make_uint2(stV[u].z, stV[u].w);
          }
        }
      }
      const int nb = kb + (PIPE == 2 ? 2 * BK : BK);
      if (nb < kEnd) {
#pragma unroll
        for (int u = 0; u < kPS; ++u) {             // адрес = база + СДВИГ, без умножения
          const int ru = (tid + u * kWarps * 32) / (D / 8);   // строка внутри плитки nb
          const bool okv = FA2_FWD_TAILGUARD ? (nb + ru < Sk) : true;   // хвостовая плитка -> нулём
          stK[u] = okv ? *reinterpret_cast<const uint4*>(pKs[u] + (long)nb * D) : make_uint4(0,0,0,0);
          stV[u] = okv ? *reinterpret_cast<const uint4*>(pVs[u] + (long)nb * D) : make_uint4(0,0,0,0);
        }
      }
      if constexpr (QREG)      s.template accumulate_pre<D>(qfrag, sKc, LDK, wn * SN, lane);
      else if constexpr (QPRE) s.template accumulate_part<D, QPRE>(qfrag, sQ, LDQ, wm * SM, sKc, LDK, wn * SN, lane);
      else if constexpr (WIDE)  s.template accumulate_w<D>(sQ, LDQ, wm * SM, sKc, LDK, wn * SN, lane);
      else                     s.template accumulate<D>(sQ, LDQ, wm * SM, sKc, LDK, wn * SN, lane);
    } else if constexpr (PIPE == 3) {
      // [softDMA НА ОСИ КУСКОВ] Прежний цикл кусков чередовал строго: sync -> LDG+STS -> sync -> mma.
      // Перекрытия нет вообще, поэтому 2 куска и 4 куска давали ОДНО время (0.265/0.266): всё теряется
      // на первом же дроблении. Здесь тот же приём, что дал +20-46% в backward (docs §13): LDG следующего
      // куска выдаётся ДО mma текущего и живёт в регистрах, STS -- поздно. Цена: kCh регистров uint4.
      constexpr int kCh = (BK * DC / 8) / (WM * WN * 32);
      uint4 stC[kCh];
#pragma unroll
      for (int u = 0; u < kCh; ++u) {                              // пролог: кусок 0 -> регистры
        const int i = tid + u * kWarps * 32, r = i / (DC / 8), c = (i % (DC / 8)) * 8;
        stC[u] = *reinterpret_cast<const uint4*>(&Kk[(long)(kb + FA2_KROW(r)) * D + c]);
      }
FA2B_UNROLL(FA2_FUNROLL)
      for (int dc = 0; dc < D; dc += DC) {
        __syncthreads();                                           // sK прочитана всеми -- можно писать
        if constexpr (QCH) {                                       // кусок Q этого шага по d
#pragma unroll
          for (int i = tid; i < BQ * DC / 8; i += kWarps * 32) {
            const int r = i / (DC / 8), c = (i % (DC / 8)) * 8;
            uint4 g = make_uint4(0, 0, 0, 0);
            if (q0 + r < Sq) g = *reinterpret_cast<const uint4*>(&Q[(long)(q0 + r) * q_sM + dc + c]);
            __half* dq = &sQ[r * LDQ + c];
            *reinterpret_cast<uint2*>(dq)     = make_uint2(g.x, g.y);
            *reinterpret_cast<uint2*>(dq + 4) = make_uint2(g.z, g.w);
          }
        }
#pragma unroll
        for (int u = 0; u < kCh; ++u) {                            // STS поздно: регистры -> smem
          const int i = tid + u * kWarps * 32, r = i / (DC / 8), c = (i % (DC / 8)) * 8;
          __half* pk = &sK[r * LDK + c];
          *reinterpret_cast<uint2*>(pk) = make_uint2(stC[u].x, stC[u].y);
          *reinterpret_cast<uint2*>(pk + 4) = make_uint2(stC[u].z, stC[u].w);
        }
        __syncthreads();
        if (dc + DC < D) {                                         // LDG рано: следующий кусок в полёте
#pragma unroll
          for (int u = 0; u < kCh; ++u) {
            const int i = tid + u * kWarps * 32, r = i / (DC / 8), c = (i % (DC / 8)) * 8;
            const __half* src = &Kk[(long)(kb + FA2_KROW(r)) * D + dc + DC + c];
            // предвыборка идёт НА ДВА куска вперёд: тот, что через один, начинает движение сейчас
            if constexpr (ROUTE & 4) { if (dc + 2 * DC < D) prefetch_l2(src + DC); }
            stC[u] = (ROUTE & 2) ? ldg_stream(src) : ldg_keep(src);
          }
        }
        // QPRE здесь НЕ применим: qfrag держит k-шаги 0..QPRE-1 ПОЛНОГО D, а не текущего куска dc.
        s.template accumulate<DC>(QCH ? sQ : (sQ + dc), LDQ, wm * SM, sK, LDK, wn * SN, lane);
      }
    } else {
#pragma unroll
      for (int dc = 0; dc < D; dc += DC) {                        // K кусками по d
        __syncthreads();
        // [АДРЕСА В ПРОЛОГ] то же лечение, что и у V выше.
        constexpr int _шагK = (kWarps * 32) / (DC / 8);
        constexpr bool _ровноK = ((kWarps * 32) % (DC / 8)) == 0;
        const int _r0K = tid / (DC / 8), _c0K = (tid % (DC / 8)) * 8;
        int _rK = _r0K;
#pragma unroll
        for (int i = tid; i < BK * DC / 8; i += kWarps * 32, _rK += _шагK) {
          const int r = _ровноK ? _rK : (i / (DC / 8));
          const int c = _ровноK ? _c0K : ((i % (DC / 8)) * 8);
          const uint4 gk = *reinterpret_cast<const uint4*>(&Kk[(long)(kb + FA2_KROW(r)) * D + dc + c]);
          __half* pk = &sK[r * LDK + c];
          if constexpr ((LDK * 2) % 16 == 0) {
            sts128(pk, gk);                    // ОДНА инструкция: шаг строки кратен 16 Б
          } else {
            *reinterpret_cast<uint2*>(pk) = make_uint2(gk.x, gk.y);
            *reinterpret_cast<uint2*>(pk + 4) = make_uint2(gk.z, gk.w);
          }
        }
        __syncthreads();
        if constexpr (SWZ && DC == D)
          s.template accumulate_swzA<DC>(sQ, LDQ, wm * SM, sK, LDK, wn * SN, lane);
        else
          s.template accumulate<DC>(sQ + dc, LDQ, wm * SM, sK, LDK, wn * SN, lane);
      }
    }

    const int r0 = wm * SM, c0 = wn * SN;
    const int rowbase = (lane & 1) | ((lane & 16) >> 2) | (lane & 8);   // строка внутри 16-блока
    float mloc[SMB][2], lloc[SMB][2];
#pragma unroll
    for (int i = 0; i < SMB; ++i) { mloc[i][0] = mloc[i][1] = -CUDART_INF_F; lloc[i][0] = lloc[i][1] = 0.f; }

    s.visit(lane, [&](float& v, int r, int c) {                   // масштаб и причинная маска
      v *= qk_scale;   // qk_scale уже = scale*log2(e), см. запуск: экспонента считается exp2
      const int kk = kb + c0 + c;
      if ((CAUSAL && (q0 + r0 + r + causal_off) < kk) || kk >= Sk) v = -CUDART_INF_F;   // ragged Sk too
      const int mb = r >> 4, sub = ((r & 15) - rowbase) >> 1;      // 0 или 1 (смещение 0 / 2)
      if (v > mloc[mb][sub]) mloc[mb][sub] = v;
    });
#pragma unroll
    for (int i = 0; i < SMB; ++i)
#pragma unroll
      for (int t = 0; t < 2; ++t) {                               // строку держат 4 полосы: биты 1 и 2
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
      for (int t = 0; t < 2; ++t) {                               // и по варпам вдоль n
        float x = -CUDART_INF_F;
#pragma unroll
        for (int w = 0; w < WN; ++w) { float y = sRedM[(wm * SM + i * 16 + rowbase + 2 * t) * WN + w]; x = x > y ? x : y; }
        mloc[i][t] = x;
      }

    // [БЕЗ ЛИШНЕГО НАКОПИТЕЛЯ] m_new и масштабы известны УЖЕ ЗДЕСЬ: они зависят от максимумов, а не
    // от сумм. Значит cn можно вложить в саму P при записи и копить P*V ПРЯМО в o — отдельный
    // накопитель add (32 регистра) не нужен. Освобождённые регистры уходят на рычаги, которые до сих
    // пор упирались в бюджет (см. §33: пять из шести попыток похоронили именно регистры).
    float corr2[SMB][2], cnew[SMB][2];
#pragma unroll
    for (int i = 0; i < SMB; ++i)
#pragma unroll
      for (int t = 0; t < 2; ++t) {
        const float mo = m_run[i][t], mn = (mo > mloc[i][t]) ? mo : mloc[i][t];
        corr2[i][t] = (mo == -CUDART_INF_F) ? 0.f : exp2f(mo - mn);
        cnew[i][t]  = (mloc[i][t] == -CUDART_INF_F) ? 0.f : exp2f(mloc[i][t] - mn);
      }
    // [ПЛОТНОСТЬ] P пишется в smem ПАРАМИ по 4 Б, а не по 2. Карта аккумулятора даёт это даром:
    //   acc_col(lane,reg) = base + (reg&1) + 4*((reg>>2)&1),  acc_row = rowbase + 2*((reg>>1)&1)
    // значит регистры (0,1), (2,3), (4,5), (6,7) — это ПАРЫ СОСЕДНИХ СТОЛБЦОВ одной строки.
    // Вдвое меньше инструкций записи; тот же приём, что в backward дал x1.265 на перезаливке.
    {
      const int cbase = (lane & 2) | ((lane & 4) << 1);
#pragma unroll
      for (int i = 0; i < SMB; ++i)
#pragma unroll
        for (int j = 0; j < SNB; ++j)
#pragma unroll
          for (int pr = 0; pr < 4; ++pr) {                       // пары регистров
            const int rg = (pr & 1) ? (4 + ((pr >> 1) << 1)) : ((pr >> 1) << 1);
            const int reg0 = rg, reg1 = rg + 1;
            const int row = i * 16 + rowbase + 2 * ((reg0 >> 1) & 1);
            const int col = j * 16 + cbase + 4 * ((reg0 >> 2) & 1);
            const int mb = i, sub = (reg0 >> 1) & 1;   // константы компиляции
            const float mm = mloc[mb][sub];
            const float v0 = s.acc[i][j][reg0], v1 = s.acc[i][j][reg1];
            const float p0 = (v0 == -CUDART_INF_F || mm == -CUDART_INF_F) ? 0.f : exp2f(v0 - mm);
            const float p1 = (v1 == -CUDART_INF_F || mm == -CUDART_INF_F) ? 0.f : exp2f(v1 - mm);
            lloc[mb][sub] += p0 + p1;                       // сумма — по НЕмасштабированной p
            const float sc2 = cnew[mb][sub];                  // масштаб нового вклада вложен в P
            *reinterpret_cast<__half2*>(&sP[(r0 + row) * LDP + c0 + col]) = __floats2half2_rn(p0 * sc2, p1 * sc2);
          }
    }
#pragma unroll
    for (int i = 0; i < SMB; ++i)
#pragma unroll
      for (int t = 0; t < 2; ++t) {
        float x = lloc[i][t];
        x += __shfl_xor_sync(0xffffffffu, x, 2);
        x += __shfl_xor_sync(0xffffffffu, x, 4);
        if ((lane & 6) == 0) sRedS[(wm * SM + i * 16 + rowbase + 2 * t) * WN + wn] = x;
      }
    __syncthreads();
    float corr[SMB][2];
#pragma unroll
    for (int i = 0; i < SMB; ++i)
#pragma unroll
      for (int t = 0; t < 2; ++t) {
        float ss = 0.f;
#pragma unroll
        for (int w = 0; w < WN; ++w) ss += sRedS[(wm * SM + i * 16 + rowbase + 2 * t) * WN + w];
        const float mo = m_run[i][t], mn = (mo > mloc[i][t]) ? mo : mloc[i][t];
        l_run[i][t] = l_run[i][t] * corr2[i][t] + ss * cnew[i][t];
        m_run[i][t] = mn; corr[i][t] = corr2[i][t];
      }
    __syncthreads();                                              // sP готова для всех варпов
    {                                    // индексы поправки — константы, а не runtime
      constexpr int OMB = OM / 16, ONB = ON / 16;
#pragma unroll
      for (int i = 0; i < OMB; ++i)
#pragma unroll
        for (int j = 0; j < ONB; ++j)
#pragma unroll
          for (int rg = 0; rg < 8; ++rg) o.acc[i][j][rg] *= corr[i][(rg >> 1) & 1];
    }
    o.template accumulate<BK>(sP, LDP, wm * OM, sVc, LDV, wn * ON, lane);   // ПРЯМО в o

    if constexpr (PIPE == 1) {                  // ПОЗДНЯЯ запись (глубина 1): счёт уже позади
      const int nb = kb + BK;
      if (nb < kEnd) {
        __syncthreads();
#pragma unroll
        for (int u = 0; u < kStage; ++u) {
          const int i = tid + u * kWarps * 32;
          const int r = i / (D / 8), c = (i % (D / 8)) * 8;
          __half* pk = &sKn[r * LDK + c]; __half* pv = &sVn[r * LDV + c];
          // u < kPS -- вектор ждал в регистрах (LDG выдан до счёта); иначе грузим прямо здесь
          const bool okv = FA2_FWD_TAILGUARD ? (nb + r < Sk) : true;    // хвостовая плитка -> нулём
          const uint4 zz = make_uint4(0, 0, 0, 0);
          const uint4 gk = (u < kPS) ? stK[u] : (okv ? *reinterpret_cast<const uint4*>(pKs[u] + (long)nb * D) : zz);
          const uint4 gv = (u < kPS) ? stV[u] : (okv ? *reinterpret_cast<const uint4*>(pVs[u] + (long)nb * D) : zz);
          *reinterpret_cast<uint2*>(pk) = make_uint2(gk.x, gk.y);
          *reinterpret_cast<uint2*>(pk + 4) = make_uint2(gk.z, gk.w);
          *reinterpret_cast<uint2*>(pv) = make_uint2(gv.x, gv.y);
          *reinterpret_cast<uint2*>(pv + 4) = make_uint2(gv.z, gv.w);
        }
      }
    }
    if constexpr (PIPE == 2) buf ^= 1;
  }
  // ---- эпилог: O = acc / l, LSE = m + log(l). Строки принадлежат этому варпу, обмена не нужно.
  {
    const int r0 = wm * OM;
    const int rowbase = (lane & 1) | ((lane & 16) >> 2) | (lane & 8);
    float inv[SMB][2];
#pragma unroll
    for (int i = 0; i < SMB; ++i)
#pragma unroll
      for (int t = 0; t < 2; ++t) inv[i][t] = (l_run[i][t] > 0.f) ? __frcp_rn(l_run[i][t]) : 0.f;
    {
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
            if (q0 + r0 + row < Sq)
              O[(long)(q0 + r0 + row) * o_sM + wn * ON + col] = __float2half(o.acc[i][j][rg] * inv[i][sub]);
          }
    }
    if ((lane & 6) == 0 && wn == 0) {                       // LSE пишет один варп на строку
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



}  // namespace block_fwd
#undef FA2_KROW
}  // namespace fa2_sm70
