/**
 * `transient_growth`（1.4 章）的 model：瞬态增长 —— ‖A^k‖ 先涨后落 vs 谱半径的单调衰减。
 *
 * 这一页要当场挡住本章第一个大坑（pitfalls 第一条）：
 * **ρ(A) < 1（甚至 = 1）不等于"范数会衰减"**。shear-10 的 ρ(A) = 1.000000，
 * 而 sup_k‖A^k‖ ≈ 1.09e10（pitfalls 的算例；本帧在 k=40 已经到 1.74e9 且还没见顶）。
 * 所以这一页的**主轴就是"增长倍数 ‖A^k‖/ρ(A)^k"**：谱半径的预测在这条轴上恒等于 1，
 * 曲线离开 1 有多远，"谱半径说了不算"就有多严重。
 *
 * 面板与图层（顺序即绘制顺序）
 * ----------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | line | `main` | `growth_ratio` = ‖A^k‖/ρ(A)^k（实测） |
 * | 2 | line | `main` | 参考线 1.0 = 谱半径的预测（`spectral_prediction` 在比值轴上恒为 1） |
 * | 3 | scatter | `main` | 峰值标记（只有 `peak_index ≤ step` 时才画在本帧范围内） |
 * | 4 | line | `main` | 当前 k 游标 |
 * | 5 | scatter | `main` | 当前点 |
 * | 6 | line | `aux` | `norms` = ‖A^k‖₂（实测） |
 * | 7 | line | `aux` | `spectral_prediction` = ρ(A)^k（理论衰减，规则 2 的第二条斜率） |
 * | 8 | line | `aux` | 当前 k 游标 |
 * | 9 | line | `err` | `error` = \|norms − spectral_prediction\|（= requiredAux，对数轴） |
 * | 10 | scatter | `err` | `error_mark`：当前步之差 |
 *
 * 角标槽位：0 来源 / 1 主轴自适应（**必须**戴这顶帽子）/ 2 副轴自适应 / 4 状态 / 6 溢出 / 7 降级。
 *
 * 几条刻意的决定
 * --------------
 * 1. **主轴也自适应、但要戴警告角标**：manifest 把主轴声明为 `autoscaled: true`，
 *    54 个数量级（1 → 1e9）也只能自适应；规则 4 的代价就是"自适应会掩盖发散"——
 *    这一页恰恰是"看起来在涨，但涨的是有限 k 的瞬态"，所以角标必须提醒读者别把
 *    自适应纵轴上的形状当成渐近行为。
 * 2. **参考线必须画在比值轴上**：谱半径的"预测"就是 ‖A^k‖ = ρ(A)^k，换成比值就是恒 1 的水平线。
 *    它不是模型编的理论，是帧里 `spectral_prediction` 那一条数组在比值轴上的样子。
 * 3. **`error` 里可以有精确的 0**（`norms == pred`，例如 k=0 两者都是 1）：对数轴上画不出来，
 *    必须显式丢弃并报数（不是"0 很小所以看不见"）。
 * 4. **`meta.non_normal` 不能当主判据**：它的定义是 `peak > 1 且 ρ < 1`，而 shear 族的 ρ 恰好 = 1，
 *    于是这个布尔量是 `False` —— 但图上的曲线明明涨了 9 个量级。文本里以**曲线与数字**为准，
 *    并把这件事写出来（这正是"别用一个布尔量代替看数据"）。
 * 5. **峰值可能还没到**：帧只算到 `k_max`，`peak_hit_max=True` 表示"峰值贴着右边界"。
 *    此时峰值标记只能画在 k_max 处并注明"可能还在后面"（不许把右端点说成峰值）。
 */

import { countNonFinite } from "../../shared/src/geometry.js";
import {
  ModelInputError,
  canvasSize,
  clamp,
  clipBadgeText,
  dataSourceOf,
  degradedBadge,
  metaNumber,
  palette,
  paramSummary,
  placeBadge,
  requireArray,
  requireDrawableLayers,
  requirePitfalls,
  requireSameLength,
  sampleLinear,
  sceneShell,
  segmentedSeries,
  stringOf,
  type SceneModel,
} from "../../shared/src/model.js";
import {
  axisModeBadge,
  badge,
  lineLayer,
  makeAxis,
  previewBadge,
  scatterLayer,
  series,
  sourceBadge,
} from "../../shared/src/scene.js";
import { manifest } from "./manifest.gen.js";
import {
  applyOverflowNote,
  argmaxIndex,
  axisTitleOf,
  axisYlabelOf,
  channelRange,
  chapterLayout,
  countNonPositive,
  fmtSci,
  integerTicks,
  log10Slope,
  logRange,
  panelBudget,
  pitfallsGuard,
  replaceFloors,
  textBlock,
  textBottom,
  textTop,
} from "./_shared.js";

const CHAPTER = "1.4";
const VIZ = "transient_growth";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "瞬态增长：‖A^k‖ 先涨后落 vs 谱半径的单调衰减";
const AXES = SPEC?.axes;

const SLOT = { source: 0, axisMode: 1, auxAxis: 2, critical: 3, status: 4, preview: 5, overflow: 6, degraded: 7 } as const;

export const transientGrowthModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需字段 ——
  const steps = requireArray(frame, "step_grid");
  const norms = requireArray(frame, "norms");
  const pred = requireArray(frame, "spectral_prediction");
  const ratio = requireArray(frame, "growth_ratio");
  const err = requireArray(frame, "error");
  const errMark = requireArray(frame, "error_mark");
  requireSameLength(steps, norms, "step_grid/norms");
  requireSameLength(steps, pred, "step_grid/spectral_prediction");
  requireSameLength(steps, ratio, "step_grid/growth_ratio");
  requireSameLength(steps, err, "step_grid/error");
  if (steps.length < 1) {
    throw new ModelInputError(`帧 ${CHAPTER}/${VIZ} 的 step_grid 是空数组：没有任何 k 可画（拒绝画空图）`);
  }
  const pitfalls = requirePitfalls(frame);

  // —— 可选字段 ——
  const degraded: string[] = [];
  const tMeta = metaNumber(frame, "t");
  if (tMeta === undefined) degraded.push("帧 meta 缺 t：scene.t 改用请求 t（两者可能不同）");
  const k = metaNumber(frame, "step");
  if (k === undefined) degraded.push("帧 meta 缺 step：不画当前 k 游标（不猜）");
  const kMax = metaNumber(frame, "k_max");
  const rho = metaNumber(frame, "spectral_radius");
  const peak = metaNumber(frame, "peak");
  const peakIndex = metaNumber(frame, "peak_index");
  const peakHitMax = (metaNumber(frame, "peak_hit_max") ?? 0) > 0.5;
  const growthRatio = metaNumber(frame, "growth_ratio");
  const kreiss = metaNumber(frame, "kreiss");
  const asymFactor = metaNumber(frame, "asymptotic_factor");
  const normCurrent = metaNumber(frame, "norm_current");
  const predCurrent = metaNumber(frame, "pred_current");
  const nonNormalFlag = metaNumber(frame, "non_normal");
  const status = stringOf(frame, "status");
  if (status === undefined) degraded.push("帧 strings 缺 status：状态角标不自行编造结论");
  const note = stringOf(frame, "note");
  const family = stringOf(frame, "family");
  if (family === undefined) degraded.push("帧 strings 缺 family：不显示矩阵族名（不按参数猜）");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: tMeta ?? opts.t,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  // —— 排版：左主图（增长倍数）+ 右列两块（范数 / 误差场）——
  const guard = pitfallsGuard(width, height, pitfalls);
  const layout = chapterLayout(width, height, {
    footer: guard.footer,
    rightShares: [1.2, 1],
    gutter: 62, // 右列自己的 y 刻度 + 轴名过道（本页没有热力图，不需要 162 的色标过道）
    mainWidth: 470,
    minRight: 240,
  });
  const mainRect = layout.main;
  const normRect = layout.right[0] as [number, number, number, number];
  const errRect = layout.right[1] as [number, number, number, number];
  const mainBudget = panelBudget(mainRect, layout.columns ? [normRect, errRect] : []);
  const normBudget = panelBudget(normRect);
  const errBudget = panelBudget(errRect);

  const kk = k === undefined ? steps.length - 1 : clamp(Math.round(k), 0, steps.length - 1);
  // k=0 的帧只有 1 个点：画不出"线"，改画点并说明（时间轴拖到右边才是完整曲线）。
  const pointLike = steps.length < 3;
  if (pointLike) {
    degraded.push(`本帧只有 ${steps.length} 个 k 点（k=0）：曲线画不出「线」，改画点；"先涨后落"要等 k≥2`);
  }
  const xMax = Math.max(1, steps[steps.length - 1] as number);

  // —— 主轴：增长倍数（log）——
  const ratioChan = channelRange(ratio, [1]);
  const mainYlim = logRange(Math.min(ratioChan.lo, 1), Math.max(ratioChan.hi, 10));
  scene.axes.push(
    makeAxis("main", mainRect, {
      xlim: [0, xMax],
      ylim: mainYlim,
      yscale: "log",
      xlabel: "幂次 $k$",
      ylabel: axisYlabelOf(AXES, "main", "倍数"),
      title: axisTitleOf(AXES, "main", "增长倍数 ‖A^k‖ / ρ(A)^k"),
      autoscaled: true,
      xticks: integerTicks(mainRect[2], steps.length),
    }),
  );

  const ratioSeries = segmentedSeries(steps, ratio, pal.curves[0], { width: 2, marker: "o", accept: (v) => v > 0 });
  if (ratioSeries.length > 0) scene.layers.push(lineLayer("main", ratioSeries));
  else degraded.push("growth_ratio 里没有一个 > 0 的值：主图画不出曲线（只在文本里报数）");
  // 参考线：谱半径的预测（比值 = 1）。它不是编的：`spectral_prediction` 换成比值就是这条。
  scene.layers.push(
    lineLayer("main", series([0, xMax], [1, 1], pal.reference, { dash: [7, 4], width: 1.3, label: "谱半径的预测 = 1" })),
  );
  // 峰值标记：只有落在**本帧**范围内才画（帧的数组截断到 step）
  let peakIdx: number | undefined = peakIndex === undefined ? undefined : Math.round(peakIndex);
  if (peakIdx === undefined) {
    // 帧没给峰值位置：用帧自己的数组取最大值位置（纯显示聚合，不重算物理量），并如实标注。
    peakIdx = argmaxIndex(ratio);
    degraded.push("帧 meta 缺 peak_index：峰值标记改用帧内 growth_ratio 的最大值位置（显示用聚合）");
  }
  if (peakIdx !== undefined && peakIdx >= 0 && peakIdx < steps.length) {
    const peakY = sampleLinear(steps, ratio, steps[peakIdx] as number);
    if (peakY !== undefined && peakY > 0) {
      scene.layers.push(
        scatterLayer("main", [steps[peakIdx] as number], [peakY], { r: 5, color: pal.marker, label: `峰值 k=${peakIdx}` }),
      );
    }
  } else if (peakIdx !== undefined) {
    degraded.push(
      `峰值在 k=${peakIdx}，而本帧只画到 k=${Math.round(steps[steps.length - 1] as number)}：不在图上标峰值（时间轴拖到 k_max 才看得到）`,
    );
  }
  scene.layers.push(lineLayer("main", series([steps[kk] as number, steps[kk] as number], mainYlim, pal.cursor, { dash: [3, 3], width: 1 })));
  const ratioNow = ratio[kk] as number;
  if (Number.isFinite(ratioNow) && ratioNow > 0) {
    scene.layers.push(scatterLayer("main", [steps[kk] as number], [ratioNow], { r: 4, color: pal.marker }));
  }

  // —— 副轴：‖A^k‖₂ 与 ρ(A)^k（manifest 的 aux 轴口径）——
  const normChan = channelRange(norms, [1]);
  const predChan = channelRange(pred, [1]);
  const normYlim = logRange(Math.min(normChan.lo, predChan.lo, 1e-16), Math.max(normChan.hi, predChan.hi, 1));
  scene.axes.push(
    makeAxis("aux", normRect, {
      xlim: [0, xMax],
      ylim: normYlim,
      yscale: "log",
      xlabel: "幂次 $k$",
      ylabel: axisYlabelOf(AXES, "aux", "2-范数"),
      title: axisTitleOf(AXES, "aux", "‖A^k‖₂ 与 ρ(A)^k"),
      autoscaled: true,
      xticks: integerTicks(normRect[2], steps.length),
    }),
  );
  const normMapped = replaceFloors(norms, normYlim[0]);
  const normSeries = segmentedSeries(steps, normMapped.values, pal.curves[0], { width: 1.8, marker: pointLike ? "o" : "none", accept: (v) => v > 0 });
  if (normSeries.length > 0) scene.layers.push(lineLayer("aux", normSeries));
  const predMapped = replaceFloors(pred, normYlim[0]);
  const predSeries = segmentedSeries(steps, predMapped.values, pal.reference, { width: 1.3, dash: [7, 4], marker: pointLike ? "o" : "none", accept: (v) => v > 0 });
  if (predSeries.length > 0) scene.layers.push(lineLayer("aux", predSeries));
  scene.layers.push(lineLayer("aux", series([steps[kk] as number, steps[kk] as number], normYlim, pal.cursor, { dash: [3, 3], width: 1 })));

  // —— 误差场轴（requiredAux: error）——
  // 对数轴上 0 不存在：**精确 0 显式丢弃并报数**（`norms == pred` 时真的会出现，例如 k=0）。
  const errChan = channelRange(err, [1]);
  const errYlim = logRange(errChan.lo, errChan.hi);
  scene.axes.push(
    makeAxis("err", errRect, {
      xlim: [0, xMax],
      ylim: errYlim,
      yscale: "log",
      xlabel: "幂次 $k$",
      ylabel: "$|\\|A^k\\|-\\rho^k|$",
      // 不设 title：右列两块是竖直切分的，标题带会压到上面那块的轴名带上（见 pseudospectrum 的说明）
      autoscaled: true,
      xticks: integerTicks(errRect[2], steps.length),
    }),
  );
  const errSeries = segmentedSeries(steps, err, pal.residual, { width: 1.6, marker: "o", accept: (v) => v > 0 });
  if (errSeries.length > 0) scene.layers.push(lineLayer("err", errSeries));
  const nErrZero = countNonPositive(err);
  const markVal = errMark.length > 0 ? (errMark[0] as number) : Number.NaN;
  if (Number.isFinite(markVal) && markVal > 0) {
    scene.layers.push(
      scatterLayer("err", [steps[kk] as number], [Math.max(markVal, errYlim[0])], { r: 4, color: pal.marker }),
    );
  }
  if (errSeries.length === 0) degraded.push("error 里没有一个 > 0 的值：误差场整条画不出来（只在文本里报数）");

  // —— 文本 ——
  const summary = paramSummary(frame, opts, [
    { name: "n", label: "$n$" },
    { name: "nonnormal", label: "非正规度 $s$" },
    { name: "k_max", label: "`k_max`" },
  ]);
  degraded.push(...summary.degraded);

  const mainLines: string[] = [
    `${summary.text}　族=${family ?? "?"}　ρ(A)=${rho === undefined ? "?" : fmtSci(rho)}`,
    `第 k=${Math.round(steps[kk] as number)} 步：‖A^k‖=${normCurrent === undefined ? "?" : fmtSci(normCurrent)}、ρ^k=${predCurrent === undefined ? "?" : fmtSci(predCurrent)}` +
      `、倍数=${Number.isFinite(ratioNow) ? fmtSci(ratioNow) : "?"}`,
    `本帧峰值 ${peak === undefined ? "?" : fmtSci(peak)}（k=${peakIdx ?? "?"}）` +
      (peakHitMax ? "，贴着 k_max：峰值可能还在后面（把 k_max 拖大再看）" : ""),
    `虚线 = 谱半径的预测（比值恒 1）：曲线离它多远，「谱半径说了不算」就有多严重${kMax === undefined ? "" : `（本帧只算到 k_max=${Math.round(kMax)}，峰值/衰减都可能还没出现）`}`,
    `⚠ ρ(A)=${rho === undefined ? "?" : fmtSci(rho)} ≤ 1 却涨到 ${growthRatio === undefined ? "?" : fmtSci(growthRatio)} 倍：谱半径只管 k→∞，对有限 k 什么都不保证`,
    nonNormalFlag !== undefined && nonNormalFlag < 0.5
      ? "帧的 meta.non_normal=False：它的定义是「ρ<1 且峰值>1」，shear 族 ρ=1 所以是 False——别拿这个布尔量当结论，看曲线"
      : "帧的 meta.non_normal=True：ρ<1 却有峰值，正是这个布尔量要表达的情形",

  ];
  scene.texts.push(...textBlock(mainRect[0] + 6, textTop(mainRect), textBottom(mainRect), mainLines, pal.ink, mainBudget, "top"));

  const lastRatio =
    kk >= 1 && Number.isFinite(norms[kk] as number) && Number.isFinite(norms[kk - 1] as number) && (norms[kk - 1] as number) > 0
      ? (norms[kk] as number) / (norms[kk - 1] as number)
      : undefined;
  const normLines: string[] = [
    "实线 = 实测 ‖A^k‖₂；长虚线 = 谱半径预测 ρ(A)^k（k→∞ 的渐近线，前几十步完全说不到一起）",
    `末步每步倍数 ‖A^k‖/‖A^{k−1}‖ = ${lastRatio === undefined ? "?" : lastRatio.toFixed(3)}` +
      (asymFactor === undefined ? "" : `（帧给的 asymptotic_factor=${fmtSci(asymFactor)}）`),
  ];
  const slopeNorm = log10Slope(
    Array.from({ length: steps.length }, (_, i) => steps[i] as number),
    Array.from(normMapped.values),
  );
  if (slopeNorm !== undefined) {
    normLines.push(`本帧实测 log10‖A^k‖ 的平均斜率 ${slopeNorm.toFixed(3)}/步（覆盖整段「先涨后落」，所以它是平均值，不是渐近率）`);
  }
  if (kreiss !== undefined) {
    normLines.push(`帧 note 把 ${fmtSci(kreiss)} 叫作「Kreiss 常数（实测峰值）」：它是实测的 sup_k‖A^k‖，不是理论保证`);
  }
  if (normMapped.nFloor > 0 || predMapped.nFloor > 0) {
    normLines.push(`${normMapped.nFloor + predMapped.nFloor} 个样本在地板 1e-300 上（后端抬高以便上 log 轴）：画在轴底`);
  }
  if (note !== undefined) normLines.push(`帧 note：${note}`);
  scene.texts.push(...textBlock(normRect[0] + 6, textTop(normRect), textBottom(normRect), normLines, pal.muted, normBudget, "top"));

  const errLines: string[] = [
    "误差场 |‖A^k‖ − ρ(A)^k|（requiredAux = error）：它是两条曲线的差值，不是相对误差",
    nErrZero > 0
      ? `${nErrZero}/${err.length} 个 k 上两者完全相等（差恰好是 0，例如 k=0 时都是 1）：对数轴上画不出来，已显式丢弃并计数`
      : "本帧每个 k 上两者都不相等（没有精确 0）",
    `当前差 ${Number.isFinite(markVal) ? fmtSci(markVal) : "?"}；整段最大 ${errChan.hi === Number.POSITIVE_INFINITY ? "?" : fmtSci(errChan.hi)}`,
    "它一直 > 0 就是「谱预测在这段 k 上完全失效」的直接证据",
  ];
  scene.texts.push(...textBlock(errRect[0] + 6, textTop(errRect), textBottom(errRect), errLines, pal.muted, errBudget, "top"));

  // —— 角标 ——
  scene.badges.push(placeBadge(sourceBadge(scene.dataSource), SLOT.source, width));
  if (scene.dataSource.startsWith("backend+preview")) {
    scene.badges.push(placeBadge(previewBadge("拖动时按显示空间插值"), SLOT.preview, width));
  }
  scene.badges.push(placeBadge(axisModeBadge(true), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(badge("autoscale", "副轴：自适应（log 纵轴按本帧范数取整；自适应会掩盖量级变化）", 8, 8), SLOT.auxAxis, width),
  );
  const statusBits: string[] = [];
  if (status !== undefined) statusBits.push(status);
  if (rho !== undefined) statusBits.push(`ρ(A)=${fmtSci(rho)}`);
  statusBits.push(
    `sup_k‖A^k‖（本帧）=${kreiss === undefined ? "?" : fmtSci(kreiss)}` +
      (nonNormalFlag !== undefined && nonNormalFlag < 0.5 ? "（meta.non_normal=False，定义见主面板文本）" : ""),
  );
  scene.badges.push(placeBadge(badge("status", clipBadgeText(statusBits.join("；")), 8, 8), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);

  // 规则 1：非有限值 + 被丢掉的 0 + 地板，全部记账。
  const extra: string[] = [];
  if (nErrZero > 0) extra.push(`error 有 ${nErrZero}/${err.length} 个样本恰好是 0（对数轴上不存在），已显式丢弃并计数`);
  if (normMapped.nFloor + predMapped.nFloor > 0) {
    extra.push(`norms/spectral_prediction 共 ${normMapped.nFloor + predMapped.nFloor} 个样本在地板 1e-300 上，已画在轴底 ${fmtSci(normYlim[0])}`);
  }
  const nBad = countNonFinite(norms) + countNonFinite(pred) + countNonFinite(ratio) + countNonFinite(err) + countNonFinite(steps);
  applyOverflowNote(scene, extra.join("；"), nBad, { slot: SLOT.overflow, width });
  return scene;
};
