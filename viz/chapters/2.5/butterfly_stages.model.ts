/**
 * `butterfly_stages`（2.5 章）的 model：radix-2 逐级蝶形（**本章唯一带时间轴的 viz**）。
 *
 * 时间轴语义（manifest `time = {mode:"scrub", tMax:1.0, dt:0.05}`）
 * --------------------------------------------------------------------------
 * `t ∈ [0, 1]` 映射到**级号** `stage = round(t·log2 N)`（后端
 * `server/chapters/2.5.py::_butterfly_stages`），一个 `t` 区间对应同一级。
 * 画面一律以帧 `meta.stage` 为准；请求 `t` 与实到档位差超过半级时，
 * `common.ts::currentStage` 会在正文里**明说**，不假装 `t` 就是级号。
 *
 * 面板与图层（顺序即绘制顺序，测试按此断言）
 * ------------------------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | line | `trace` | 已走过的级（`stage_values_real` 的第 `0…stage−1` 行，细虚线=轨迹） |
 * | 2 | line | `trace` | 当前级实部 `current_real`（粗线） |
 * | 3 | line | `trace` | 当前级虚部 `current_imag`（粗线） |
 * | 4 | line | `residual` | 逐级残差 `residual`（x = 级号 1…levels，圆点） |
 * | 5 | line | `residual` | `error`（|FFT−DFT|，整段常数）作为**数学对拍**参照线 |
 * | 6 | scatter | `residual` | 当前级在残差曲线上的位置 |
 * | 7 | bars | `spectrum` | 位反序重排后的 `spectrum_mag` |
 * | 8 | line | `spectrum` | 输入频率格 `k` 的参考竖线 + 峰值水平线 |
 *
 * 角标槽位（0 来源 / 1 主轴自适应 / 2 残差轴自适应 / 3 状态 / 6 溢出 / 7 降级）。
 *
 * 三条刻意的决定
 * --------------
 * 1. **"残差贴地板"必须在图上就能读成"这是实现一致性、不是数学正确性"**：残差面板上同时
 *    画 `residual`（向量化 vs 朴素 DIF，本帧 ≤ 2.06e-16）与 `error`（|FFT − DFT|，
 *    本帧 2.40e-15，来自帧数组）两条线。两条都在舍入量级、但**量的不是一件事**
 *    ——见 `frame.pitfalls` 与 `chapters/2.5/pitfalls.md` §5。
 * 2. **位反序是这张图唯一能证伪的东西**：`spectrum_mag` 是**重排后**的 |X_k|，所以谱峰必须
 *    落在输入余弦的频率格 `meta.k` 上。不重排时峰值会跑到 `bit_reverse(k)` 去（pitfalls §4：
 *    拿冲激测则"怎么都对"，因为冲激的谱对任何置换都不敏感）。
 * 3. **轨迹画"已走过的级"，不画未来**：`stage_values_real` 的 `0…stage` 行反映当前进度，
 *    后面的行不画（画出来会让人以为时间轴没动）。
 */

import { niceLimits, niceTicks } from "../../shared/src/axes.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import {
  ModelInputError,
  resolveTextOverlaps,
  applyOverflow,
  canvasSize,
  clipBadgeText,
  columnsLayout,
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
  sceneShell,
  segmentedSeries,
  stringOf,
  toNumeric,
  type SceneModel,
} from "../../shared/src/model.js";
import {
  axisModeBadge,
  badge,
  barsLayer,
  lineLayer,
  makeAxis,
  scatterLayer,
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
  currentStage,
  fmt,
  fmtOpt,
  minMaxOfAll,
  sci,
  specTitle,
  textBlock,
  textBlockRight,
  unitBudget,
} from "./common.js";

const VIZ = "butterfly_stages";
const TITLE = specTitle(VIZ, "radix-2 蝶形分级");

export const butterflyStagesModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需数组 ——
  const valuesReal = frame.matrix("stage_values_real");
  const valuesImag = frame.matrix("stage_values_imag");
  const currentReal = requireArray(frame, "current_real");
  const currentImag = requireArray(frame, "current_imag");
  const residual = requireArray(frame, "residual");
  const spectrumMag = requireArray(frame, "spectrum_mag");
  const error = requireArray(frame, "error");
  requireSameLength(currentReal, currentImag, "帧数组 current_real 与 current_imag");
  if (valuesReal.rows !== valuesImag.rows || valuesReal.cols !== valuesImag.cols) {
    throw new ModelInputError(`帧 ${CHAPTER}/${VIZ} 的实部/虚部快照形状不一致：${valuesReal.rows}×${valuesReal.cols} vs ${valuesImag.rows}×${valuesImag.cols}`);
  }
  if (currentReal.length !== valuesReal.cols) {
    throw new ModelInputError(`帧 ${CHAPTER}/${VIZ} 的 current_real 长 ${currentReal.length}，与快照列数 ${valuesReal.cols} 不符（拒绝画错位的轨迹）`);
  }
  const pitfalls = requirePitfalls(frame);

  // —— 可选字段 ——
  const degraded: string[] = [];
  const tMeta = metaNumber(frame, "t");
  if (tMeta === undefined) degraded.push("帧 meta 缺 t：scene.t 改用请求 t（两者可能不同）");
  const nMeta = metaNumber(frame, "n");
  const n = nMeta === undefined ? valuesReal.cols : Math.round(nMeta);
  if (nMeta === undefined) degraded.push(`帧 meta 缺 n：点数按快照列数给（${valuesReal.cols}）`);
  const levelsMeta = metaNumber(frame, "levels");
  if (levelsMeta === undefined) degraded.push("帧 meta 缺 levels：级数按快照行数−1 给（形状是数据不是猜的）");
  const levels = levelsMeta === undefined ? valuesReal.rows - 1 : Math.round(levelsMeta);
  if (levels !== valuesReal.rows - 1) {
    degraded.push(`帧 meta.levels=${fmt(levels, 3)} 与快照行数−1=${valuesReal.rows - 1} 不一致：以行数为准画轨迹`);
  }
  const nButterflies = metaNumber(frame, "n_butterflies");
  const maxResidual = metaNumber(frame, "max_level_residual");
  const spectrumError = metaNumber(frame, "spectrum_error");
  const spectrumPeak = metaNumber(frame, "spectrum_peak");
  const kBin = metaNumber(frame, "k");
  const signalName = stringOf(frame, "signal");
  if (signalName === undefined) degraded.push("帧 strings 缺 signal：不标输入信号名（不按请求参数猜）");
  const algorithm = stringOf(frame, "algorithm");
  const note = stringOf(frame, "note");
  if (note === undefined) degraded.push("帧 strings 缺 note：状态角标不自行编造结论");
  const nonfinite = stringOf(frame, "nonfinite_policy");

  const stageInfo = currentStage(frame, opts, levels);
  degraded.push(...stageInfo.notes);
  const stage = stageInfo.stage;

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: tMeta ?? opts.t,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  // —— 排版：左列「轨迹（上）+ 逐级残差（下）」，右列整高给位反序谱；底部留关键数字横条 ——
  const strip = bottomStrip(scene, width, height, 4);
  const panels = columnsLayout(width, height, { share: 0.56, gap: 54, footer: strip.footer });
  const traceRect = panels.main;
  const resRect = panels.error;
  const specRect = panels.side;

  const idx = Array.from({ length: n }, (_, i) => i);
  const stageIdx = Array.from({ length: levels }, (_, s) => s + 1);

  // 轨迹面板纵轴：当前级 + 已走过的级（两者都要放得下；自适应轴按帧数据取整）。
  const snapshotReal = toNumeric(valuesReal.values);
  const trailRange = minMaxOfAll([currentReal, currentImag, sliceRows(snapshotReal, valuesReal.cols, stage)]);
  const traceRange = trailRange ?? [-1, 1];
  const traceAxis = makeAxis("trace", traceRect, {
    xlim: [0, Math.max(1, n - 1)],
    ylim: niceLimits(traceRange[0], traceRange[1], 5, { pad: 0.1 }),
    xlabel: "样本序号（DIF 的级内位置）",
    ylabel: axisYlabel(VIZ, "trace", "值"),
    title: axisTitle(VIZ, "trace", "蝶形轨迹（当前级的值）"),
    autoscaled: axisAutoscaled(VIZ, "trace"),
    xticks: niceTicks(0, Math.max(1, n - 1), 6),
  });

  // 残差副轴（规则 3）：residual 与 error 共一条 log 纵轴（都是"差"，但量的东西不同）。
  const resPos = minMaxOfAll([absOf(residual), absOf(error)]);
  const resYlim: [number, number] = resPos === undefined ? [1e-18, 1e-12] : niceLimits(resPos[0], resPos[1], 5, { log: true, pad: 0.2 });
  if (resPos === undefined) degraded.push("帧 residual/error 没有 >0 的有限值：对数副轴用占位范围 [1e-18, 1e-12]，不画曲线");
  const resAxis = makeAxis("residual", resRect, {
    xlim: [0.5, Math.max(1.5, levels + 0.5)],
    ylim: resYlim,
    yscale: "log",
    xlabel: "级号 s（1…log2N）",
    ylabel: axisYlabel(VIZ, "residual", "max|delta|"),
    title: axisTitle(VIZ, "residual", "逐级残差（向量化 vs 朴素 DIF）"),
    autoscaled: axisAutoscaled(VIZ, "residual"),
    xticks: stageIdx.filter((_, i) => i % Math.max(1, Math.ceil(levels / 6)) === 0),
  });

  // 位反序谱面板：柱高 = |X_k|，纵轴到峰值上方 15%（固定轴，manifest autoscaled=false）。
  const peak = spectrumPeak ?? (minMaxOfAll([spectrumMag])?.[1] ?? 1);
  const specAxis = makeAxis("spectrum", specRect, {
    xlim: [-0.5, Math.max(0.5, n - 0.5)],
    ylim: [0, Math.max(1e-12, peak * 1.15)],
    xlabel: "频率格 k（位反序重排后的顺序）",
    ylabel: axisYlabel(VIZ, "spectrum", "|X|"),
    title: axisTitle(VIZ, "spectrum", "位反序重排后的 |X_k|"),
    autoscaled: axisAutoscaled(VIZ, "spectrum"),
  });
  scene.axes.push(traceAxis, resAxis, specAxis);

  // —— 图层 1~3：轨迹 ——
  const trails: LineSeries[] = [];
  for (let s = 0; s < stage; s += 1) {
    const row = new Float64Array(n);
    for (let c = 0; c < n; c += 1) row[c] = snapshotReal[s * valuesReal.cols + c] as number;
    trails.push(series(idx, row, pal.cursor, { width: 1, dash: [3, 3]}));
  }
  if (trails.length > 0) scene.layers.push(lineLayer("trace", trails));
  const curSeries: LineSeries[] = [];
  const realSeries = segmentedSeries(idx, currentReal, pal.curves[0], { width: 2.2});
  curSeries.push(...realSeries);
  const imagSeries = segmentedSeries(idx, currentImag, pal.curves[1], { width: 2.2, dash: [6, 3]});
  curSeries.push(...imagSeries);
  if (curSeries.length > 0) scene.layers.push(lineLayer("trace", curSeries));
  else degraded.push("当前级的实部/虚部没有可绘的有限值：轨迹面板不画曲线");

  // —— 图层 4~6：逐级残差 + 数学对拍参照线 ——
  const resSeries = segmentedSeries(stageIdx, residual, pal.residual, { width: 1.6, marker: "o", accept: (v) => v > 0});
  const errLine = absOf(error);
  const errConst = errLine.length > 0 ? (errLine[0] as number) : undefined;
  const refs: LineSeries[] = [];
  if (resSeries.length > 0) refs.push(...resSeries);
  else degraded.push("帧 residual 没有 >0 的有限值：对数副轴上画不出残差曲线");
  if (errConst !== undefined && Number.isFinite(errConst) && errConst > 0) {
    refs.push(
      series([0.5, Math.max(1.5, levels + 0.5)], [errConst, errConst], pal.exact, {
        width: 1.4,
        dash: [7, 4]
      }),
    );
  }
  scene.layers.push(lineLayer("residual", refs));
  const resNow = sampleAt(stageIdx, residual, stage);
  if (resNow !== undefined && resNow > 0) {
    scene.layers.push(scatterLayer("residual", [stage], [resNow], { r: 4, color: pal.marker}));
  }

  // —— 图层 7~8：位反序谱 ——
  scene.layers.push(barsLayer("spectrum", idx, spectrumMag, { width: 0.7, color: pal.curves[0]}));
  const specRefs: LineSeries[] = [];
  if (kBin !== undefined) {
    specRefs.push(series([kBin, kBin], [0, specAxis.ylim[1]], pal.reference, { dash: [3, 4], width: 1}));
  }
  specRefs.push(series([-0.5, n - 0.5], [peak, peak], pal.marker, { dash: [6, 3], width: 1}));
  scene.layers.push(lineLayer("spectrum", specRefs));

  // —— 文本：面板内只放"身份行"，长句子全部进底部横条 ——
  const summary = paramSummary(frame, opts, [
    { name: "n", label: "N" },
    { name: "k", label: "k" },
  ]);
  degraded.push(...summary.degraded);
  const traceLines: string[] = [summary.text + (signalName === undefined ? "" : `　信号 ${signalName}（帧 strings.signal）`)];
  traceLines.push(
    stageInfo.requested === stage
      ? `时间轴 t=${fmt(opts.t, 3)} → 第 ${stage}/${levels} 级（t → round(t·log2N)；0 = 输入，${levels} = 末行）`
      : `请求 t=${fmt(opts.t, 3)}（→ 第 ${stageInfo.requested} 级），本帧实到第 ${stage}/${levels} 级：` +
        `画面以帧 meta.stage 为准（0 = 输入，${levels} = 末行）`,
  );
  scene.texts.push(...textBlock(traceRect[0] + 6, traceRect[1] + 4, traceRect[1] + traceRect[3] - 6, traceLines, pal.muted, unitBudget(traceRect[2]), "top"));

  const resLines: string[] = [
    maxResidual === undefined
      ? `帧 meta 缺 max_level_residual：只画 ${residual.length} 级残差，不报数值（不重算）`
      : `逐级残差 ≤ ${sci(maxResidual)}（帧 max_level_residual）；横线 |FFT − DFT| = ${sci(spectrumError ?? Number.NaN)} 才是数学对拍`,
  ];
  if (nonfinite !== undefined) resLines.push(`地板口径（帧 strings）：${nonfinite}`);
  scene.texts.push(...textBlock(resRect[0] + 6, resRect[1] + 4, resRect[1] + resRect[3] - 6, resLines, pal.muted, unitBudget(resRect[2]), "top"));

  const specTop: string[] = [
    `峰值 = ${fmt(peak, 4)}（帧 spectrum_peak = N/2）` + (kBin === undefined ? "" : `；峰位 k = ${fmt(kBin, 3)} 与输入 cos 的频率格一致（帧 meta.k）`),
  ];
  scene.texts.push(...textBlockRight(specRect[0] + specRect[2] - 6, specRect[1] + 4, specRect[1] + specRect[3] - 6, specTop, pal.muted, unitBudget(specRect[2]), "top"));
  const specBot: string[] = ["末行是位反序的：经 bit_reverse_indices(N) 置换后才逐点等于 DFT(x)"];
  scene.texts.push(...textBlock(specRect[0] + 6, specRect[1] + 4, specRect[1] + specRect[3] - 6, specBot, pal.muted, unitBudget(specRect[2]), "bottom"));

  // 底部横条：算法规模、"残差 ≠ 正确性"、位反序。
  const stripLines: string[] = [
    `级数 log2N = ${levels}，蝶形数 (N/2)·log2N = ${fmtOpt(nButterflies, 6)}` + (algorithm === undefined ? "" : `（${algorithm}）`) + `；N = ${fmt(n, 6)}`,
    "帧 note：末行位反序，经 bit_reverse_indices 置换后必须逐点等于 DFT(x)",
    "逐级残差量的是「向量化 vs 朴素 DIF」的一致性，不是数学正确性（pitfalls §5）",
    "数学正确性由 |FFT − DFT| 横线负责；不做重排时相对误差 O(1)（pitfalls §4：n=8 时 1.071）",
  ];
  scene.texts.push(...textBlock(traceRect[0], strip.stripTop, strip.limit, stripLines, pal.muted, unitBudget(width - traceRect[0] - 24), "top"));

  // —— 角标 ——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(true), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(badge("autoscale", "残差副轴：自适应（log 纵轴按本帧 residual/error 取整；自适应会掩盖量级变化）", 8, 8), SLOT.auxAxis, width),
  );
  const status: string[] = [`第 ${stage}/${levels} 级`];
  if (nButterflies !== undefined) status.push(`蝶形数 ${fmt(nButterflies, 6)}`);
  if (maxResidual !== undefined) status.push(`残差≤${sci(maxResidual)}`);
  if (spectrumError !== undefined) status.push(`|FFT−DFT|=${sci(spectrumError)}`);
  scene.badges.push(placeBadge(badge("status", clipBadgeText(status.join("；")), 8, 8), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  const nBad =
    countNonFinite(toNumeric(valuesReal.values)) +
    countNonFinite(toNumeric(valuesImag.values)) +
    countNonFinite(currentReal) +
    countNonFinite(currentImag) +
    countNonFinite(residual) +
    countNonFinite(spectrumMag) +
    countNonFinite(error);
  applyOverflow(scene, nBad, { slot: SLOT.overflow, width });

  scene.texts = resolveTextOverlaps(scene.texts, [8, 24, width - 16, strip.limit - 24]);
  return scene;
};

// --------------------------------------------------------------------------- //
// 局部工具
// --------------------------------------------------------------------------- //

/** 快照矩阵的前 `rows` 行（行主序）拼成一个扁平数组，供"当前进度"的纵轴范围用。 */
function sliceRows(values: ArrayLike<number>, cols: number, rows: number): Float64Array {
  const n = Math.max(0, Math.min(rows, Math.floor(values.length / Math.max(1, cols)))) * cols;
  const out = new Float64Array(n);
  for (let i = 0; i < n; i += 1) out[i] = values[i] as number;
  return out;
}

/** 在离散网格上取 `x = xq` 处的值（找不到返回 undefined；不插值——级号是整数）。 */
function sampleAt(xs: readonly number[], ys: ArrayLike<number>, xq: number): number | undefined {
  for (let i = 0; i < Math.min(xs.length, ys.length); i += 1) {
    if (Math.abs((xs[i] as number) - xq) < 1e-12) return ys[i] as number;
  }
  return undefined;
}

/** 供测试引用：这份 model 依赖的帧数组名。 */
export const BUTTERFLY_ARRAYS = ["stage_values_real", "stage_values_imag", "current_real", "current_imag", "residual", "spectrum_mag", "error"] as const;
