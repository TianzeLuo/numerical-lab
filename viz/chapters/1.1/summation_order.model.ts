/**
 * `summation_order`（1.1 章）的 model：1e16 + 1 + 1 + … 的五种求和顺序。
 *
 * 面板与图层（顺序即绘制顺序，测试按此断言）
 * ------------------------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | line | `terms` | `terms`（前 64 项）的折线：big 一根独峰，其余全压在 0 上 |
 * | 2 | scatter | `terms` | 每一项一个点（这就是"线性纵轴下小项消失"的样子） |
 * | 3 | line | `error` | 当前 k 的游标（虚线竖线） |
 * | 4 | line | `error` | 地板（`meta.error_floor`，点线）：贴住它的点不是"误差 = 地板值" |
 * | 5 | line | `error` | 理论上界 `n·u·Σ|x|`（灰虚线） |
 * | 6 | line | `error` | 理论上界 `(2u + n·u²)·Σ|x|`（灰点线） |
 * | 7 | line | `error` | `error_naive`：朴素左折叠 |
 * | 8 | line | `error` | `error_kahan`：**选中的**补偿方式（`params.compensated`，标签带方案名） |
 * | 9 | line | `error` | `error_kahan_serial`：串行 Kahan 参考（仅在它与选中方案不同时才画） |
 * | 10 | line | `error` | `error_pairwise`：成对求和 |
 * | 11 | line | `error` | `error_sorted`：排序后朴素求和 |
 * | 12 | scatter | `error` | 三条曲线当前值的标记（朴素/补偿/成对） |
 *
 * 角标槽位（固定，见 `SLOT`）：0 来源 / 1 主轴自适应 / 2 副轴自适应 / 3 临界（`big` 的
 * 2^53）/ 4 状态（方案 + note + 比值）/ 5 前端预览 / 6 溢出 / 7 降级说明。
 *
 * 几条刻意的决定
 * --------------
 * 1. **主轴是线性轴，小项就该看不见**：manifest 给 `terms` 轴声明的是 `autoscaled: true`
 *    且没有 log。`1e16` 与 `1` 相差 16 个数量级，63 个小项在 `[0, 2e16]` 上不足 1 像素
 *    ——**这正是朴素求和把它们丢掉的样子**，不是画错。模型不偷偷换 log 轴，
 *    而是把这一点写进面板文字与 pitfalls：想看清小项得看数值行，不能看这张图。
 * 2. **"贴地板"与"一直准"是两件事**：后端把误差恰好为 0 的点抬到 `meta.error_floor`
 *    （`np.maximum(err, floor)`），所以补偿曲线与朴素曲线在头几步可能画出同一个高度
 *    （朴素求和在 k 很小的时候也是精确的）。判据是曲线相对 `n·u·Σ|x|` 那条界的**斜率**，
 *    模型把两条界都画出来，并在文本里给出当前 k 的两个误差值，让"斜率"可读。
 * 3. **`error_kahan` 是"选中的方案"，不叫"Kahan"**：`params.compensated` 是 enum
 *    （serial / shuffled / parallel8 / parallel64），`error_kahan` 装的是**选中方案**的误差，
 *    `error_kahan_serial` 才是串行 Kahan 参考。两者相等时（scheme = serial）不重复画线，
 *    只在文本里说明"本次归约 / 串行 = 1"——画两条完全重合的线等于伪造一个对比。
 * 4. **k 被夹住时说清楚**：时间轴 `tMax = 4096` 而项数 `n_terms` 可以更小
 *    （默认 512）。后端把 k 夹到 `[1, n]` 并写进 `meta.k`，模型据此画游标，
 *    并在 `|t − k| > 0.5` 时明说"请求的 t 已被夹到 k"——不假装 t 就是 k。
 */

import { formatNumber, niceLimits } from "../../shared/src/axes.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import type { Frame } from "../../shared/src/frame.js";
import {
  applyOverflow,
  canvasSize,
  clamp,
  criticalOf,
  dataSourceOf,
  degradedBadge,
  finiteMinMax,
  finitePositiveMinMax,
  metaNumber,
  palette,
  paramSummary,
  placeBadge,
  requireArray,
  requireDrawableLayers,
  requirePitfalls,
  requireSameLength,
  sceneShell,
  segmentedSeries,
  stackedPanels,
  stringOf,
  type Rect,
  type SceneModel,
} from "../../shared/src/model.js";
import {
  axisModeBadge,
  badge,
  criticalBadge,
  lineLayer,
  makeAxis,
  overflowBadge,
  previewBadge,
  scatterLayer,
  series,
  sourceBadge,
} from "../../shared/src/scene.js";
import {
  axisTitleOf,
  axisYlabelOf,
  constantArray,
  floorIndices,
  pick,
  textBlock,
  textBottom,
  textTop,
  unitBudget,
} from "./_panel.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "1.1";
const VIZ = "summation_order";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "求和顺序：朴素 vs Kahan vs 成对 vs 排序";

/** 固定角标槽位（见 `model.ts::BADGE_LAYOUT`）。 */
const SLOT = {
  source: 0,
  axisMode: 1,
  auxAxis: 2,
  critical: 3,
  status: 4,
  /** 仅当帧声明 `backend+preview` 时出现（本章的帧都是 backend）。 */
  preview: 5,
  overflow: 6,
  degraded: 7,
} as const;

/** `frame.critical.big` 缺失时的回落值：2^53 = 1/u（超过它，相邻整数不再可表示）。 */
const BIG_CRITICAL = 9007199254740992;

/** `params.compensated` 的四种取值 → 画面上的方案名（与后端 `scheme_note` 同口径）。 */
const SCHEME_LABELS: Record<string, string> = {
  serial: "串行 Kahan",
  shuffled: "置换后 Kahan",
  parallel8: "8 段并行（段内 Kahan、段间朴素）",
  parallel64: "64 段并行（段内 Kahan、段间朴素）",
};

export const summationOrderModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需数组 ——
  const termsK = requireArray(frame, "terms_k");
  const terms = requireArray(frame, "terms");
  const kGrid = requireArray(frame, "k_grid");
  const errorNaive = requireArray(frame, "error_naive");
  const errorKahan = requireArray(frame, "error_kahan");
  const errorKahanSerial = requireArray(frame, "error_kahan_serial");
  const errorPairwise = requireArray(frame, "error_pairwise");
  const errorSorted = requireArray(frame, "error_sorted");
  const boundNaive = requireArray(frame, "error_bound_naive");
  const boundKahan = requireArray(frame, "error_bound_kahan");
  requireSameLength(termsK, terms, "帧数组 terms_k 与 terms");
  requireSameLength(kGrid, errorNaive, "帧数组 k_grid 与 error_naive");
  requireSameLength(kGrid, errorKahan, "帧数组 k_grid 与 error_kahan");
  requireSameLength(kGrid, errorKahanSerial, "帧数组 k_grid 与 error_kahan_serial");
  requireSameLength(kGrid, errorPairwise, "帧数组 k_grid 与 error_pairwise");
  requireSameLength(kGrid, errorSorted, "帧数组 k_grid 与 error_sorted");
  requireSameLength(kGrid, boundNaive, "帧数组 k_grid 与 error_bound_naive");
  requireSameLength(kGrid, boundKahan, "帧数组 k_grid 与 error_bound_kahan");
  const pitfalls = requirePitfalls(frame);

  // —— 可选数组（当前值的标记）——
  const degraded: string[] = [];
  const naiveMark = optionalFirst(frame, "error_naive_mark");
  const kahanMark = optionalFirst(frame, "error_kahan_mark");
  const pairwiseMark = optionalFirst(frame, "error_pairwise_mark");
  const kMark = optionalFirst(frame, "k_mark");
  if (kMark === undefined) degraded.push("帧缺 k_mark：游标不画（不拿请求 t 冒充部分和长度 k）");

  // —— meta ——
  const tMeta = metaNumber(frame, "t");
  if (tMeta === undefined) degraded.push("帧 meta 缺 t：scene.t 改用请求 t（两者可能不同）");
  const big = metaNumber(frame, "big");
  const small = metaNumber(frame, "small");
  const nTerms = metaNumber(frame, "n_terms");
  const kMeta = metaNumber(frame, "k");
  if (kMeta === undefined) degraded.push("帧 meta 缺 k：当前部分和长度按 k_mark 画");
  const exactSum = metaNumber(frame, "exact_sum");
  const sumNaive = metaNumber(frame, "sum_naive");
  const errNaiveNow = metaNumber(frame, "error_naive");
  const errKahanNow = metaNumber(frame, "error_kahan");
  const errKahanSerialNow = metaNumber(frame, "error_kahan_serial");
  const errPairwiseNow = metaNumber(frame, "error_pairwise");
  const errSortedNow = metaNumber(frame, "error_sorted");
  const boundNaiveNow = metaNumber(frame, "bound_naive");
  const boundKahanNow = metaNumber(frame, "bound_kahan");
  const kahanOverNaive = metaNumber(frame, "kahan_over_naive");
  const parallelOverSerial = metaNumber(frame, "parallel_over_serial");
  const termsAbsSum = metaNumber(frame, "terms_abs_sum");
  const errorFloor = metaNumber(frame, "error_floor");
  const nExactZero = metaNumber(frame, "error_exact_zero_points");
  const unitRoundoff = metaNumber(frame, "unit_roundoff");
  const frameOverflow = (metaNumber(frame, "overflow") ?? 0) > 0.5;
  const scheme = stringOf(frame, "scheme") ?? String(opts.params.compensated ?? "");
  const schemeLabel = SCHEME_LABELS[scheme] ?? (scheme === "" ? "补偿求和（方案未给出）" : `${scheme}（未收录的方案名）`);
  if (scheme === "" || SCHEME_LABELS[scheme] === undefined) degraded.push("帧 strings 缺 scheme（或值不在已知表里）：方案名按原样显示，不翻译");
  const note = stringOf(frame, "note");
  const floorNote = stringOf(frame, "floor_note");
  if (note === undefined) degraded.push("帧 strings 缺 note：状态角标不自行编造结论");
  if (floorNote === undefined) degraded.push("帧 strings 缺 floor_note：地板的读法不自行编造（点数仍按 meta.error_floor 统计）");
  if (frame.critical["big"] === undefined) {
    degraded.push(`帧 critical 缺 big：临界值回落到 manifest 声明的 ${BIG_CRITICAL}（不是帧给的）`);
  }
  const schemeNote = stringOf(frame, "scheme_note");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: tMeta ?? opts.t,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  const base = stackedPanels(width, height);
  const mainRect: Rect = base.main;
  const auxRect: Rect = base.error;

  // —— 主轴：前 64 项的线性纵轴（autoscaled: true，manifest 没给 log）——
  const tRange = finiteMinMax(terms) ?? [0, 1];
  const mainAxis = makeAxis("terms", mainRect, {
    xlim: niceLimits(finiteMinMax(termsK)?.[0] ?? 0, finiteMinMax(termsK)?.[1] ?? 1, 8),
    ylim: niceLimits(Math.min(0, tRange[0]), Math.max(tRange[1], tRange[0] === tRange[1] ? tRange[0] + 1 : tRange[0]), 5),
    // 主轴 xlabel 置空（下方面板顶部要放说明文字，规则 8）；横轴含义写进面板文字。
    xlabel: "",
    ylabel: axisYlabelOf(SPEC?.axes, "terms", "x_i"),
    title: axisTitleOf(SPEC?.axes, "terms", "项序列（前 64 项）"),
    autoscaled: true,
    tickTarget: 6,
  });

  // —— 副轴：五条误差曲线 + 两条理论界共用的 log 纵轴 ——
  const pools = [errorNaive, errorKahan, errorKahanSerial, errorPairwise, errorSorted, boundNaive, boundKahan];
  let lo = Number.POSITIVE_INFINITY;
  let hi = 0;
  for (const p of pools) {
    const b = finitePositiveMinMax(p);
    if (b === undefined) continue;
    lo = Math.min(lo, b[0]);
    hi = Math.max(hi, b[1]);
  }
  if (errorFloor !== undefined && errorFloor > 0) {
    lo = Math.min(lo, errorFloor);
    hi = Math.max(hi, errorFloor);
  }
  if (!(Number.isFinite(lo) && Number.isFinite(hi))) {
    degraded.push("七条误差/界数组里没有 > 0 的有限值：副轴用占位范围 [1e-18, 1e18]，不画曲线");
  }
  const auxYlim: [number, number] =
    Number.isFinite(lo) && Number.isFinite(hi) ? niceLimits(lo, hi, 7, { log: true }) : [1e-18, 1e18];
  const kRange = finiteMinMax(kGrid) ?? [1, 1];
  const auxAxis = makeAxis("error", auxRect, {
    xlim: niceLimits(kRange[0], Math.max(kRange[1], kRange[0] + 1), 8),
    ylim: auxYlim,
    yscale: "log",
    xlabel: "部分和长度 k",
    ylabel: axisYlabelOf(SPEC?.axes, "error", "绝对误差 |fl(Σx) − Σx|"),
    title: axisTitleOf(SPEC?.axes, "error", "五种顺序的误差与两条理论界（对数）"),
    autoscaled: true,
    tickTarget: 6,
  });
  scene.axes.push(mainAxis, auxAxis);

  // —— 图层 1~2：项序列（线性纵轴下小项被压成 0 像素——这正是要看到的）——
  const termSeries = [];
  if (terms.length > 0) {
    termSeries.push(
      series(termsK, terms, pal.curves[0] ?? "#2b6cb0", { width: 1.2, label: "项序列 x_i（前 64 项）" }),
    );
  }
  if (termSeries.length > 0) scene.layers.push(lineLayer("terms", termSeries));
  else degraded.push("terms 为空：项序列画不出来");
  scene.layers.push(
    scatterLayer("terms", termsK, terms, { r: 3, color: pal.marker, alpha: 0.85, label: "每一项：big 一根独峰，63 个小项全部压在 0 上（线性纵轴）" }),
  );

  // —— 图层 3~4：游标与地板 ——
  const kNow = kMeta ?? kMark ?? clamp(Math.round(opts.t), kRange[0], kRange[1]);
  scene.layers.push(
    lineLayer("error", series([kNow, kNow], auxAxis.ylim, pal.cursor, { width: 1, dash: [3, 3], label: `当前 k = ${formatNumber(kNow, 5)}` })),
  );
  if (errorFloor !== undefined && errorFloor > 0) {
    scene.layers.push(
      lineLayer(
        "error",
        series([kRange[0], auxAxis.xlim[1]], [errorFloor, errorFloor], pal.reference, {
          width: 1.1,
          dash: [2, 3],
          label: `显示地板 u·max|Σx|/2 = ${formatNumber(errorFloor, 3)}：贴住它的点不是「误差 = 地板值」`,
        }),
      ),
    );
  } else {
    degraded.push("帧 meta 缺 error_floor：不画地板线（贴地板的点会看不出是地板）");
  }

  // —— 图层 5~6：两条理论界（规则 2：理论斜率必须与实测同图）——
  const bounds = segmentedSeries(kGrid, boundNaive, pal.reference, {
    width: 1.3,
    dash: [7, 4],
    label: "理论上界 n·u·Σ|x|（在对数纵轴上斜率 +1）",
    accept: (v) => v > 0,
  });
  if (bounds.length > 0) scene.layers.push(lineLayer("error", bounds));
  else degraded.push("error_bound_naive 没有 > 0 的有限值：朴素求和的界画不出来");
  const boundsK = segmentedSeries(kGrid, boundKahan, pal.reference, {
    width: 1.1,
    dash: [2, 4],
    label: "理论上界 (2u + n·u²)·Σ|x|（与 n 无关）",
    accept: (v) => v > 0,
  });
  if (boundsK.length > 0) scene.layers.push(lineLayer("error", boundsK));
  else degraded.push("error_bound_kahan 没有 > 0 的有限值：补偿求和的界画不出来");

  // —— 图层 7~11：五条误差曲线 ——
  const naiveSeries = segmentedSeries(kGrid, errorNaive, pal.curves[0] ?? "#2b6cb0", {
    width: 1.9,
    label: "朴素左折叠（误差随 k 线性增长）",
    accept: (v) => v > 0,
  });
  if (naiveSeries.length > 0) scene.layers.push(lineLayer("error", naiveSeries));
  else degraded.push("error_naive 的样本没有一个 > 0：log 轴上画不出朴素曲线");

  const schemeSeries = segmentedSeries(kGrid, errorKahan, pal.curves[1] ?? "#2f855a", {
    width: 1.9,
    label: `选中的补偿方式：${schemeLabel}`,
    accept: (v) => v > 0,
  });
  if (schemeSeries.length > 0) scene.layers.push(lineLayer("error", schemeSeries));
  else degraded.push("error_kahan 的样本没有一个 > 0：log 轴上画不出补偿曲线");

  // 串行 Kahan 参考：只在它与选中方案**真的不同**时画（重合的两条线是伪造的对比）。
  const serialDiffers =
    errorKahanSerial.length === errorKahan.length &&
    Array.from({ length: errorKahan.length }, (_, i) => Math.abs((errorKahanSerial[i] as number) - (errorKahan[i] as number)) > 0).some(Boolean);
  if (serialDiffers) {
    const serialSeries = segmentedSeries(kGrid, errorKahanSerial, pal.exact, {
      width: 1.5,
      dash: [6, 3],
      label: "串行 Kahan（参考：置换不该改变它，分段并行才该）",
      accept: (v) => v > 0,
    });
    if (serialSeries.length > 0) scene.layers.push(lineLayer("error", serialSeries));
  }

  const pairSeries = segmentedSeries(kGrid, errorPairwise, pal.curves[2] ?? "#b7791f", {
    width: 1.5,
    dash: [4, 3],
    label: "成对求和（误差 ~ u·⌈log2 k⌉·Σ|x|）",
    accept: (v) => v > 0,
  });
  if (pairSeries.length > 0) scene.layers.push(lineLayer("error", pairSeries));
  else degraded.push("error_pairwise 的样本没有一个 > 0：log 轴上画不出成对曲线");
  const sortedSeries = segmentedSeries(kGrid, errorSorted, pal.residual, {
    width: 1.4,
    dash: [2, 3],
    label: "排序后朴素求和",
    accept: (v) => v > 0,
  });
  if (sortedSeries.length > 0) scene.layers.push(lineLayer("error", sortedSeries));
  else degraded.push("error_sorted 的样本没有一个 > 0：log 轴上画不出排序曲线");

  // 贴地板的点（用帧给的标量地板铺成向量逐点比）。
  if (errorFloor !== undefined && errorFloor > 0) {
    const floorVec = constantArray(errorNaive.length, errorFloor);
    const onFloorNaive = floorIndices(errorNaive, floorVec);
    if (onFloorNaive.length > 0) {
      const p = pick(kGrid, errorNaive, onFloorNaive);
      scene.layers.push(
        scatterLayer("error", p.x, p.y, { r: 3, color: pal.warn, alpha: 0.9, label: `朴素曲线贴地板的 ${onFloorNaive.length} 个点（误差精确为 0）` }),
      );
    }
  }

  // —— 图层 12：三条曲线当前值的标记 ——
  if (kMark !== undefined) {
    const markX: number[] = [];
    const markY: number[] = [];
    const markColor: string[] = [];
    if (naiveMark !== undefined) {
      markX.push(kMark);
      markY.push(naiveMark);
      markColor.push(pal.curves[0] ?? "#2b6cb0");
    }
    if (kahanMark !== undefined) {
      markX.push(kMark);
      markY.push(kahanMark);
      markColor.push(pal.curves[1] ?? "#2f855a");
    }
    if (pairwiseMark !== undefined) {
      markX.push(kMark);
      markY.push(pairwiseMark);
      markColor.push(pal.curves[2] ?? "#b7791f");
    }
    markX.forEach((x, i) => {
      scene.layers.push(scatterLayer("error", [x], [markY[i] as number], { r: 4.5, color: markColor[i] as string, label: i === 0 ? "当前 k 的误差（朴素/补偿/成对）" : undefined }));
    });
  }

  // —— 文本 ——
  const summary = paramSummary(frame, opts, [
    { name: "big", label: "大数 big" },
    { name: "small", label: "小项 small" },
    { name: "n_terms", label: "项数 n" },
  ]);
  degraded.push(...summary.degraded);

  const head: string[] = [summary.text];
  if (small !== undefined) head.push(`小项=${formatNumber(small, 6)}`);
  head.push(`补偿方式=${schemeLabel}`);
  if (termsAbsSum !== undefined) head.push(`Σ|x|（前 k 项）=${formatNumber(termsAbsSum, 5)}`);
  if (unitRoundoff !== undefined) head.push(`u=${formatNumber(unitRoundoff, 3)}`);
  const mainBudget = unitBudget(mainRect[2]);
  scene.texts.push(...textBlock(mainRect[0] + 6, textTop(mainRect), textBottom(mainRect), [head.join("　")], pal.muted, mainBudget, "top"));

  // 主轴中间是一整块空白（除最左边那根 big 的尖峰）：把"这张图为什么看不出小项"写在这里，
  // 副轴才留得下位置画曲线。
  const midTop = mainRect[1] + Math.round(mainRect[3] * 0.3);
  const mid: string[] = [
    `横轴=项序号 i（前 64 项）　纵轴=x_i（线性 [0, ${formatNumber(mainAxis.ylim[1], 3)}]）`,
    "big 与 small 相差 16 个数量级：小项的高度不足 1 像素",
    "**这不是画错**——朴素求和丢掉的就是它们；它们的真实量级看副轴与数值行",
  ];
  if (schemeNote !== undefined) mid.push(schemeNote);
  if (floorNote !== undefined) mid.push(floorNote);
  scene.texts.push(...textBlock(mainRect[0] + 6, midTop, textBottom(mainRect), mid, pal.warn, mainBudget, "top"));

  // 副轴只留 4 行：帧 note、当前数值、误差与界、判据。
  const auxLines: string[] = [];
  if (note !== undefined) auxLines.push(`帧 note：${note}`);
  const nums: string[] = [`k=${formatNumber(kNow, 5)}`];
  if (exactSum !== undefined) nums.push(`精确和=${formatNumber(exactSum, 8)}`);
  if (sumNaive !== undefined) nums.push(`朴素和=${formatNumber(sumNaive, 8)}`);
  auxLines.push(nums.join("　"));
  const errs: string[] = [];
  if (errNaiveNow !== undefined) errs.push(`朴素=${formatNumber(errNaiveNow, 3)}`);
  if (errKahanNow !== undefined) errs.push(`补偿=${formatNumber(errKahanNow, 3)}`);
  if (errKahanSerialNow !== undefined) errs.push(`串行 Kahan=${formatNumber(errKahanSerialNow, 3)}`);
  if (errPairwiseNow !== undefined) errs.push(`成对=${formatNumber(errPairwiseNow, 3)}`);
  if (errSortedNow !== undefined) errs.push(`排序=${formatNumber(errSortedNow, 3)}`);
  if (errs.length > 0) auxLines.push(errs.join("　"));
  const boundsNow: string[] = [];
  if (boundNaiveNow !== undefined) boundsNow.push(`界 n·u·Σ|x|=${formatNumber(boundNaiveNow, 3)}`);
  if (boundKahanNow !== undefined) boundsNow.push(`界 (2u+n·u²)·Σ|x|=${formatNumber(boundKahanNow, 3)}`);
  if (kahanOverNaive !== undefined) boundsNow.push(`补偿/朴素=${formatNumber(kahanOverNaive, 3)}`);
  if (parallelOverSerial !== undefined) boundsNow.push(`本次归约/串行=${formatNumber(parallelOverSerial, 3)}`);
  if (boundsNow.length > 0) auxLines.push(boundsNow.join("　"));
  auxLines.push(
    "贴地板（误差恰为 0）≠ 一直很准：判据是曲线相对 n·u·Σ|x| 的斜率" +
      (nExactZero === undefined ? "" : `（本帧三法合计 ${formatNumber(nExactZero, 4)} 个点误差为 0）`),
  );
  if (!serialDiffers) {
    auxLines.push("本次方案=串行 Kahan：它与参考曲线逐点相同（比值 1.000），只画一条（重合的两条线是伪造的对比）");
  }
  // 时间轴口径：t 是帧参数（manifest 的 time.kind=index，tMax=4096），图上的采样点是 k（1..n_terms）。
  // 两者不是同一个量，**不**在这里重算后端的映射（TS 只读帧），只把口径讲清楚。
  auxLines.push(
    `时间轴口径：t=${formatNumber(opts.t, 5)} 是帧请求参数，游标与数值都取帧 meta.k=${formatNumber(kNow, 5)}` +
      (nTerms === undefined ? "" : `（本帧 n_terms=${formatNumber(nTerms, 5)}；t 与项数 k 不是一个量）`),
  );
  const auxBudget = unitBudget(auxRect[2]);
  scene.texts.push(...textBlock(auxRect[0] + 6, textTop(auxRect), textBottom(auxRect), auxLines, pal.muted, auxBudget, "top"));

  // 游标标签**不画**：副轴那一行被说明块占满（规则 8 的"文本压文本"抓到 21×14px 重叠），
  // 而 k 的值在说明块与角标里都写了——竖虚线仍在，不为了多一个数字挤掉一句话。

  // —— 角标（固定槽位）——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  if (scene.dataSource.startsWith("backend+preview")) {
    scene.badges.push(placeBadge(previewBadge("拖动时按显示空间插值"), SLOT.preview, width));
  }
  scene.badges.push(placeBadge(axisModeBadge(true), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(
      badge("autoscale", "副轴：自适应（log 误差轴按本帧七条曲线取整；自适应会掩盖量级变化）", 8, 8),
      SLOT.auxAxis,
      width,
    ),
  );
  if (big !== undefined) {
    scene.badges.push(placeBadge(criticalBadge("big", big, criticalOf(frame, "big", BIG_CRITICAL), { beyond: "above" }), SLOT.critical, width));
  }
  const status: string[] = [`方案=${schemeLabel}`];
  if (kahanOverNaive !== undefined) status.push(`补偿/朴素=${formatNumber(kahanOverNaive, 3)}`);
  if (parallelOverSerial !== undefined) status.push(`本次归约/串行=${formatNumber(parallelOverSerial, 3)}`);
  if (note !== undefined) status.push(note);
  scene.badges.push(placeBadge(badge("status", status.join("；"), 8, 8), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  const nBad =
    countNonFinite(termsK) +
    countNonFinite(terms) +
    countNonFinite(kGrid) +
    countNonFinite(errorNaive) +
    countNonFinite(errorKahan) +
    countNonFinite(errorKahanSerial) +
    countNonFinite(errorPairwise) +
    countNonFinite(errorSorted) +
    countNonFinite(boundNaive) +
    countNonFinite(boundKahan);
  applyOverflow(scene, nBad, { slot: SLOT.overflow, width });
  if (frameOverflow) {
    const note2 = [scene.overflow.note, "帧 meta.overflow=true：误差轨迹里出现了非有限值（求和本身已经溢出/未定义）"]
      .filter((s) => s.length > 0)
      .join("；");
    scene.overflow = { flag: true, note: note2 };
    if (!scene.badges.some((b) => b.kind === "overflow")) {
      scene.badges.push(placeBadge(overflowBadge(note2), SLOT.overflow, width));
    }
  }
  return scene;
};

// --------------------------------------------------------------------------- //
// 局部工具
// --------------------------------------------------------------------------- //

/** 取"长度 1 的数组"的第 0 个元素；缺数组、空数组、非有限值一律 undefined（降级，不猜）。 */
function optionalFirst(frame: Frame, name: string): number | undefined {
  if (!frame.has(name)) return undefined;
  const a = requireArray(frame, name);
  if (a.length === 0) return undefined;
  const v = a[0] as number;
  return Number.isFinite(v) ? v : undefined;
}
