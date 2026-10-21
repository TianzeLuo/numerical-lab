/**
 * `conv_theorem`（2.5 章）的 model：卷积定理与**循环卷绕**。
 *
 * 面板与图层（顺序即绘制顺序，测试按此断言）
 * ------------------------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | line | `curves` | 信号 `signal`（细虚线，灰色参照） |
 * | 2 | line | `curves` | 线性卷积 `linear`（前 N 项，对比色实线） |
 * | 3 | line | `curves` | 循环卷积 `circular`（频域相乘，主色实线） |
 * | 4 | line | `error` | `error` = \|循环 − 线性\|（柱+折线；requiredAux） |
 * | 5 | line | `error` | 卷绕落点区（`x ∈ [−0.5, M−1.5]` 的竖线边界）与 `first_wrap_index` 游标 |
 * | 6 | line | `safer` | `safe_difference`（补零到 N+M−1 后的差） |
 * | 7 | line | `safer` | `y = 0` 与 `±meta.safe_error` 参考线（"贴到哪条线"就是结论） |
 *
 * 角标槽位（0 来源 / 1 主轴自适应 / 2 差异轴固定 / 3 `n` 临界 / 4 状态 / 5 卷绕条数 / 6 溢出 / 7 降级）。
 *
 * 三条刻意的决定
 * --------------
 * 1. **"N 大一点就行"是错的，必须由差曲线与数说话**：`curves` 面板上循环卷积与线性卷积
 *    几乎重合（差异只在头部几项），所以本页把 `error`（循环−线性）单独画一块面板，
 *    并把「差异位置数 = M−1」从帧数组里**数出来**报出（不是抄常数）。pitfalls §13 的
 *    N 扫描（16→0.2508、24→6.19e-2、32→2.07e-2、64→2.63e-3，**不会自己变成 0**；
 *    只有补零到 N+M−1 才贴到 1.11e-16）以引用形式写在正文里，并标明来源与条件。
 * 2. **补零不是"更精确的卷积"**：`safer` 面板画的是同一个乘加结果与线性卷积的差，
 *    纵轴范围由帧的 `meta.safe_error` 定（约 1e-16），与上面的 0.25 差**四个数量级**；
 *    文本里写明"补零只让循环卷积**等于**线性卷积，一个乘加都没变"。
 * 3. **帧里的 `kernel` 数组不画**：它是后端 `np.resize(h, N)`——M 之后把 h **重复**了一遍
 *    （不是补零），画出来会被读成「长度 N 的滤波器」。这一点已写进交付报告；正文里也点一句，
 *    免得有人问「为什么滤波器没画」。
 */

import { niceLimits, niceTicks } from "../../shared/src/axes.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import {
  ModelInputError,
  resolveTextOverlaps,
  applyOverflow,
  canvasSize,
  clipBadgeText,
  criticalOf,
  dataSourceOf,
  degradedBadge,
  metaNumber,
  palette,
  paramSummary,
  placeBadge,
  requireArray,
  requireDrawableLayers,
  requirePitfalls,
  sceneShell,
  segmentedSeries,
  stringOf,
  type SceneModel,
} from "../../shared/src/model.js";
import {
  axisModeBadge,
  badge,
  barsLayer,
  criticalBadge,
  lineLayer,
  makeAxis,
  series,
  sourceBadge,
  type LineSeries,
} from "../../shared/src/scene.js";
import {
  CHAPTER,
  SLOT,
  absOf,
  axisAutoscaled,
  bottomStrip,
  axisTitle,
  axisYlabel,
  fmt,
  fmtOpt,
  minMaxOfAll,
  sci,
  specTitle,
  textBlock,
  textBlockRight,
  topWithTwoBelow,
  unitBudget,
  withZero,
} from "./common.js";

const VIZ = "conv_theorem";
const TITLE = specTitle(VIZ, "卷积定理与循环卷绕");

/** 差异被当作"真的卷绕了"的判据：超过它才算非零位置（帧里的尾巴是精确 0）。 */
const WRAP_EPS = 1e-15;

export const convTheoremModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需数组 ——
  const signal = requireArray(frame, "signal");
  const kernel = requireArray(frame, "kernel");
  const circular = requireArray(frame, "circular");
  const linear = requireArray(frame, "linear");
  const difference = requireArray(frame, "difference");
  const safeDifference = requireArray(frame, "safe_difference");
  const error = requireArray(frame, "error");
  const len = signal.length;
  for (const [name, arr] of [
    ["kernel", kernel],
    ["circular", circular],
    ["linear", linear],
    ["difference", difference],
    ["safe_difference", safeDifference],
    ["error", error],
  ] as const) {
    if (arr.length !== len) {
      throw new ModelInputError(`帧 ${CHAPTER}/${VIZ} 的 '${name}' 长 ${arr.length}，与 signal 的 ${len} 不符（拒绝画错位的对照曲线）`);
    }
  }
  const pitfalls = requirePitfalls(frame);

  // —— 可选字段 ——
  const degraded: string[] = [];
  const tMeta = metaNumber(frame, "t");
  if (tMeta === undefined) degraded.push("帧 meta 缺 t：scene.t 改用请求 t（两者可能不同）");
  const n = metaNumber(frame, "n");
  if (n === undefined) degraded.push(`帧 meta 缺 n：长度按信号数组给（${len}）`);
  const m = metaNumber(frame, "m");
  if (m === undefined) degraded.push("帧 meta 缺 m：不报「差异位置数 = M−1」（不按请求参数猜）");
  const need = metaNumber(frame, "need");
  if (need === undefined) degraded.push("帧 meta 缺 need（N+M−1）：不报「补零到多少才安全」");
  const maxWrap = metaNumber(frame, "max_wrap");
  if (maxWrap === undefined) degraded.push("帧 meta 缺 max_wrap：不报最大卷绕误差（不重算）");
  const firstWrap = metaNumber(frame, "first_wrap_index");
  const wrapRel = metaNumber(frame, "wrap_rel");
  const safeError = metaNumber(frame, "safe_error");
  if (safeError === undefined) degraded.push("帧 meta 缺 safe_error：补零后的差曲线纵轴退回帧数组极值");
  const theoremError = metaNumber(frame, "theorem_error");
  const directVsFft = metaNumber(frame, "direct_vs_fft");
  const kernelName = stringOf(frame, "kernel");
  const signalName = stringOf(frame, "signal");
  const note = stringOf(frame, "note");
  if (note === undefined) degraded.push("帧 strings 缺 note：状态角标不自行编造结论");
  const nonfinite = stringOf(frame, "nonfinite_policy");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: tMeta ?? opts.t,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  // —— 排版：上「两条卷积曲线」整宽，下「循环−线性」与「补零后」并排；底部留关键数字横条 ——
  const strip = bottomStrip(scene, width, height, 4);
  const plot = topWithTwoBelow(width, height, { footer: strip.footer });
  const curvesRect = plot.top;
  const errRect = plot.leftBottom;
  const saferRect = plot.rightBottom;

  const idx = Array.from({ length: len }, (_, i) => i);

  // 曲线面板：信号与两条卷积共用一条纵轴（都是同一量纲）。
  const curveRange = minMaxOfAll([signal, circular, linear]) ?? [0, 1];
  const curvesAxis = makeAxis("curves", curvesRect, {
    xlim: [-0.5, Math.max(0.5, len - 0.5)],
    ylim: niceLimits(curveRange[0], curveRange[1], 5, { pad: 0.08 }),
    xlabel: "样本序号（0…$N$−1）",
    ylabel: axisYlabel(VIZ, "curves", "值"),
    title: axisTitle(VIZ, "curves", "两条卷积曲线"),
    autoscaled: axisAutoscaled(VIZ, "curves"),
    xticks: niceTicks(-0.5, len - 0.5, 6),
  });

  // 差异面板（规则 3）：循环 − 线性，线性纵轴（有正有负，围绕 0 对称）。
  const diffRange = minMaxOfAll([difference, error]);
  const diffYlim: [number, number] = niceLimits(...withZero(diffRange ?? [-1, 1]), 5, { pad: 0.08 });
  const errAxis = makeAxis("error", errRect, {
    xlim: [-0.5, Math.max(0.5, len - 0.5)],
    ylim: diffYlim,
    xlabel: "样本序号",
    ylabel: axisYlabel(VIZ, "error", "delta"),
    title: axisTitle(VIZ, "error", "循环 − 线性（差异全在卷绕处）"),
    autoscaled: axisAutoscaled(VIZ, "error"),
    xticks: niceTicks(-0.5, len - 0.5, 5),
  });

  // 「补零后」面板：同一个差的量级，纵轴按 meta.safe_error 定（贴地板就是结论）。
  const safeScale = Math.max(safeError ?? 0, minMaxOfAll([absOf(safeDifference)])?.[1] ?? 0, 1e-17);
  const saferYlim: [number, number] = [-safeScale * 1.5, safeScale * 1.5];
  const saferAxis = makeAxis("safer", saferRect, {
    xlim: [-0.5, Math.max(0.5, len - 0.5)],
    ylim: saferYlim,
    xlabel: "样本序号",
    ylabel: axisYlabel(VIZ, "safer", "delta"),
    title: axisTitle(VIZ, "safer", "补零到 N+M-1 后的差异"),
    autoscaled: axisAutoscaled(VIZ, "safer"),
    xticks: niceTicks(-0.5, len - 0.5, 5),
  });
  scene.axes.push(curvesAxis, errAxis, saferAxis);

  // —— 差异位置数（从帧数组数出来，不是抄常数）——
  const wrapPositions: number[] = [];
  for (let i = 0; i < Math.min(difference.length, len); i += 1) {
    if (Math.abs(difference[i] as number) > WRAP_EPS) wrapPositions.push(i);
  }
  const expectPositions = m === undefined ? undefined : Math.max(0, Math.round(m) - 1);
  if (expectPositions !== undefined && wrapPositions.length !== expectPositions && (n ?? len) < (need ?? Number.POSITIVE_INFINITY)) {
    degraded.push(
      `帧 difference 的非零位置有 ${wrapPositions.length} 个，而 M−1=${expectPositions}：两处不一致，画面按**数组**报数`,
    );
  }

  // —— 图层 1~3：两条卷积曲线 + 信号 ——
  const curveSeries: LineSeries[] = [];
  const sigSeries = segmentedSeries(idx, signal, pal.reference, { width: 1.2, dash: [4, 4], label: "输入信号" });
  curveSeries.push(...sigSeries);
  const linSeries = segmentedSeries(idx, linear, pal.exact, { width: 1.8, dash: [7, 4], label: "线性卷积" });
  curveSeries.push(...linSeries);
  const cirSeries = segmentedSeries(idx, circular, pal.curves[0], { width: 1.8, label: "循环卷积" });
  curveSeries.push(...cirSeries);
  if (curveSeries.length > 0) scene.layers.push(lineLayer("curves", curveSeries));
  else degraded.push("帧 signal/circular/linear 没有可绘的有限值：上方面板不画曲线");

  // —— 图层 4~5：卷绕误差（requiredAux）+ 落点区 ——
  const errSeries = segmentedSeries(idx, absOf(error), pal.marker, {
    width: 1.6,
    accept: (v) => v > 0,
    label: "循环 − 线性",
  });
  scene.layers.push(
    barsLayer("error", idx, difference, { width: 0.8, color: pal.residual, label: "逐点差异" }),
  );
  if (errSeries.length > 0) scene.layers.push(lineLayer("error", errSeries));
  else degraded.push("帧 error 没有 >0 的有限值：对数/线性面板都画不出卷绕幅度");
  const wrapRefs: LineSeries[] = [];
  if (expectPositions !== undefined && expectPositions > 0) {
    const edge = expectPositions - 0.5;
    wrapRefs.push(series([edge, edge], [errAxis.ylim[0], errAxis.ylim[1]], pal.reference, { dash: [4, 3], width: 1, label: "卷绕起点" }));
    wrapRefs.push(series([-0.5, edge], [errAxis.ylim[1] * 0.92, errAxis.ylim[1] * 0.92], pal.reference, { width: 1, dash: [2, 4] }));
  }
  if (firstWrap !== undefined && Math.abs(firstWrap - Math.round(firstWrap)) < 1e-9) {
    wrapRefs.push(series([firstWrap, firstWrap], [errAxis.ylim[0], errAxis.ylim[1]], pal.marker, { width: 1, dash: [2, 3], label: "首个卷绕位置" }));
  }
  if (wrapRefs.length > 0) scene.layers.push(lineLayer("error", wrapRefs));

  // —— 图层 6~7：补零后的差 + 参考线 ——
  const saferSeries = segmentedSeries(idx, safeDifference, pal.curves[1], { width: 1.6, label: "补零后的差异" });
  const saferRefs: LineSeries[] = [series([-0.5, len - 0.5], [0, 0], pal.reference, { dash: [5, 4], width: 1, label: "零线" })];
  if (safeError !== undefined) {
    saferRefs.push(series([-0.5, len - 0.5], [safeError, safeError], pal.marker, { dash: [3, 3], width: 1, label: "补零后的误差界" }));
    saferRefs.push(series([-0.5, len - 0.5], [-safeError, -safeError], pal.marker, { dash: [3, 3], width: 1 }));
  } else {
    saferRefs.push(series([-0.5, len - 0.5], [safeScale, safeScale], pal.marker, { dash: [3, 3], width: 1, label: "补零后的误差界" }));
  }
  scene.layers.push(lineLayer("safer", saferRefs));
  if (saferSeries.length > 0) scene.layers.push(lineLayer("safer", saferSeries));
  else degraded.push("帧 safe_difference 没有可绘的有限值：补零面板不画曲线");

  // —— 文本：面板内只放"身份行"，长句子全部进底部横条 ——
  const summary = paramSummary(frame, opts, [
    { name: "n", label: "$N$" },
    { name: "m", label: "$M$" },
  ]);
  degraded.push(...summary.degraded);
  const curveLines: string[] = [summary.text + (signalName === undefined ? "" : `　信号 ${signalName}`) + (kernelName === undefined ? "" : `　核 ${kernelName}`)];
  if (need !== undefined) {
    curveLines.push(`线性卷积有 N+M−1 = ${fmt(need, 4)} 项，这里只画前 N = ${fmt(n ?? len, 4)} 项：多出来的尾部就是会卷绕回来的那一段`);
  }
  scene.texts.push(...textBlock(curvesRect[0] + 6, curvesRect[1] + 4, curvesRect[1] + curvesRect[3] - 6, curveLines, pal.muted, unitBudget(curvesRect[2]), "top"));

  const errLines: string[] = [
    (maxWrap === undefined ? "帧 meta 缺 max_wrap：只画差曲线，不报数值（不重算）" : `最大卷绕误差 = ${fmt(maxWrap, 6)}（帧 max_wrap）`) +
      (wrapRel === undefined ? "" : `，相对 ${fmt(wrapRel * 100, 3)}%`),
    `差异位置数 = ${wrapPositions.length}` + (expectPositions === undefined ? "（帧缺 M：不判「= M−1」）" : ` = M−1（M = ${fmt(m as number, 3)}）：全在头部，正好是你最可能去看的那几项`),
  ];
  scene.texts.push(
    ...textBlockRight(errRect[0] + errRect[2] - 6, errRect[1] + 4, errRect[1] + errRect[3] - 6, errLines, pal.muted, unitBudget(errRect[2]), "top"),
  );

  const saferLines: string[] = [
    (safeError === undefined ? "帧 meta 缺 safe_error：补零后的差按帧数组极值定纵轴（不重算）" : `补零到 N+M−1 后的最大偏差 = ${sci(safeError)}（帧 safe_error）：贴舍入地板，不是「更精确」`),
  ];
  if (theoremError !== undefined) saferLines.push(`定理残差 |DFT(a⊛b) − DFT(a)·DFT(b)| = ${sci(theoremError)}（帧 theorem_error）`);
  if (directVsFft !== undefined) saferLines.push(`显式循环卷积 vs 频域相乘 = ${sci(directVsFft)}（帧 direct_vs_fft）`);
  if (nonfinite !== undefined) saferLines.push(`地板口径（帧 strings）：${nonfinite}`);
  scene.texts.push(...textBlock(saferRect[0] + 6, saferRect[1] + 4, saferRect[1] + saferRect[3] - 6, saferLines, pal.muted, unitBudget(saferRect[2]), "top"));

  // 底部横条：「N 大一点就行」为什么错、硬条件是什么、kernel 数组为什么没画。
  const stripLines: string[] = [
    "N 大一点就行？pitfalls §13 实测（M=8、box 核）：N=16→0.2508、24→6.19e-2、32→2.07e-2、64→2.63e-3——它不会自己变成 0",
    `硬条件 N ≥ N_a+M−1 = ${fmtOpt(need, 4)}` + (n === undefined ? "" : `（本帧 N = ${fmt(n, 4)}）`) +
      (safeError === undefined ? "" : `：补零到 ${fmtOpt(need, 4)} 才贴到 ${sci(safeError)}，一个乘加都没变`),
    "帧 kernel 数组是 np.resize(h,N)（M 之后重复 h，不是补零）：本页不画它，免得读成「长度 N 的滤波器」",
  ];
  if (note !== undefined) stripLines.push(`帧 note：${note}`);
  scene.texts.push(...textBlock(curvesRect[0], strip.stripTop, strip.limit, stripLines, pal.muted, unitBudget(width - curvesRect[0] - 24), "top"));

  // —— 角标 ——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(true), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(badge("autoscale", "曲线轴：自适应（按本帧信号/卷积取整；自适应会掩盖量级变化）", 8, 8), SLOT.auxAxis, width),
  );
  if (n !== undefined) {
    scene.badges.push(placeBadge(criticalBadge("n", n, criticalOf(frame, "n", 24), { beyond: "below" }), SLOT.critical, width));
  } else {
    degraded.push("帧 meta 缺 n：不挂 n 的临界角标（不猜临界值）");
  }
  scene.badges.push(
    placeBadge(badge("critical", clipBadgeText(`差异位置数 = ${wrapPositions.length} = M−1（M=${fmtOpt(m, 3)}）：卷绕全落在头部`), 8, 8), SLOT.critical2, width),
  );
  const status: string[] = [];
  if (maxWrap !== undefined) status.push(`max|循环−线性|=${fmt(maxWrap, 4)}`);
  if (safeError !== undefined) status.push(`补零后=${sci(safeError)}`);
  if (note !== undefined) status.push(note);
  scene.badges.push(placeBadge(badge("status", clipBadgeText(status.join("；")), 8, 8), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  const nBad =
    countNonFinite(signal) +
    countNonFinite(kernel) +
    countNonFinite(circular) +
    countNonFinite(linear) +
    countNonFinite(difference) +
    countNonFinite(safeDifference) +
    countNonFinite(error);
  applyOverflow(scene, nBad, { slot: SLOT.overflow, width });

  scene.texts = resolveTextOverlaps(scene.texts, [8, 24, width - 16, strip.limit - 24]);
  return scene;
};

// --------------------------------------------------------------------------- //
// 局部工具
// --------------------------------------------------------------------------- //

/** 供测试引用：这份 model 依赖的帧数组名。 */
export const CONV_THEOREM_ARRAYS = ["signal", "kernel", "circular", "linear", "difference", "safe_difference", "error"] as const;
