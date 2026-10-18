/**
 * `resolution`（2.5 章）的 model：分辨率与不确定性——两个近邻音能不能分开。
 *
 * 面板与图层（顺序即绘制顺序，测试按此断言）
 * ------------------------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | line | `spec` | 两音 dB 谱 `spectrum_db`（只画 `f ≤ xHi` 这一段） |
 * | 2 | line | `spec` | 「分辨率尺」三条横段：当前两音间隔 `df`、实测阈值 `sep_measured`、Rayleigh `1/T` |
 * | 3 | line | `spec` | −3 dB 线与峰值/谷位参考竖线（峰位从帧谱线**读出**，不是抄参数） |
 * | 4 | scatter | `spec` | 两个谱峰 + 两峰之间的谷（判据的三要素：两个局部极大、谷低于矮峰） |
 * | 5 | line | `threshold` | `span_ratio`（可分辨 vs 观测点数，阶梯+圆点）+ `y=0/1` 参考线 |
 * | 6 | line | `threshold` | 当前观测点数 `meta.n` 的游标竖线 |
 * | 7 | line | `error` | `error`（`\|FFT − DFT\|`，整段常数；requiredAux） |
 *
 * 角标槽位（0 来源 / 1 主轴固定 / 2 误差轴自适应 / 3 `pad` 临界 / 4 状态 / 5 `df` 临界 / 6 溢出 / 7 降级）。
 *
 * 三条刻意的决定
 * --------------
 * 1. **判据是"谷低于矮峰"，不是"峰够不够尖"**：谱面板上把两个峰与它们之间的谷一起标出来，
 *    谷的深度写进正文。用"峰有多尖"判会得出相反结论（`frame.pitfalls` 与 pitfalls §12）。
 * 2. **`pad` 与 `T` 分开报**：`grid_df = 1/(N·pad)` 随 `pad` 变，`true_df = 1/N`（本页固定
 *    `T = 1`）不随 `pad` 变；实测阈值 `sep_measured` 与 Rayleigh `1/T` 并排。这一页的
 *    "补零提高了分辨率"是同一个坑的第二个入口（pitfalls §1/§11）。
 * 3. **窗的代价只在半格位置现形**：整数格扫描下四个窗都给 `1.000/T`，正文必须补上
 *    pitfalls §11 的半格实测（2.25/T 与 3.75/T：矩形窗仍可分、Hann 并包，主瓣 1.44 格已经
 *    贴住 1.5 格间隔）——否则读者会从这一页读出"分辨率与窗无关"。
 */

import { niceLimits, niceTicks } from "../../shared/src/axes.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import {
  resolveTextOverlaps,
  applyOverflow,
  canvasSize,
  clipBadgeText,
  criticalOf,
  dataSourceOf,
  degradedBadge,
  finiteMinMax,
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
  type SceneModel,
} from "../../shared/src/model.js";
import {
  axisModeBadge,
  badge,
  criticalBadge,
  lineLayer,
  makeAxis,
  makeErrorAxis,
  scatterLayer,
  series,
  sourceBadge,
  text,
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
  sliceRange,
  specTitle,
  textBlock,
  textBlockRight,
  topWithTwoBelow,
  unitBudget,
} from "./common.js";

const VIZ = "resolution";
const TITLE = specTitle(VIZ, "分辨率与不确定性");

/** 后端 `_resolution` 里固定 `T = 1`（`t_max = 1.0`）；帧 meta 没有它，写在正文里。 */
const T_RECORD = 1;

export const resolutionModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需数组 ——
  const specDb = requireArray(frame, "spectrum_db");
  const freq = requireArray(frame, "freq");
  const spanRatio = requireArray(frame, "span_ratio");
  const spanN = requireArray(frame, "span_n");
  const error = requireArray(frame, "error");
  requireSameLength(freq, specDb, "帧数组 freq 与 spectrum_db");
  requireSameLength(spanN, spanRatio, "帧数组 span_n 与 span_ratio");
  const pitfalls = requirePitfalls(frame);

  // —— 可选字段 ——
  const degraded: string[] = [];
  const tMeta = metaNumber(frame, "t");
  if (tMeta === undefined) degraded.push("帧 meta 缺 t：scene.t 改用请求 t（两者可能不同）");
  const n = metaNumber(frame, "n");
  if (n === undefined) degraded.push(`帧 meta 缺 n：观测点数按 error 数组长度给（${error.length}）`);
  const pad = metaNumber(frame, "pad");
  if (pad === undefined) degraded.push("帧 meta 缺 pad：不并排报 grid_df / true_df（不猜）");
  const dfWant = metaNumber(frame, "df");
  if (dfWant === undefined) degraded.push("帧 meta 缺 df（两音间隔）：不画间隔尺、不挂临界角标（不猜）");
  const amp2 = metaNumber(frame, "amp2");
  const separableMeta = metaNumber(frame, "separable");
  const sepMeasured = metaNumber(frame, "sep_measured");
  if (sepMeasured === undefined) degraded.push("帧 meta 缺 sep_measured：不报实测最小可分辨间隔（不重算）");
  const rayleigh = metaNumber(frame, "rayleigh");
  if (rayleigh === undefined) degraded.push("帧 meta 缺 rayleigh：不画 Rayleigh 参考尺（无法对照）");
  const mainlobe = metaNumber(frame, "mainlobe_bins");
  const sepOverMainlobe = metaNumber(frame, "sep_over_mainlobe");
  const gridDf = metaNumber(frame, "grid_df");
  const trueDf = metaNumber(frame, "true_df");
  if (gridDf === undefined || trueDf === undefined) degraded.push("帧 meta 缺 grid_df / true_df：不并排报「栅格随 pad 变、1/T 不变」");
  const maxError = metaNumber(frame, "max_error");
  const windowName = stringOf(frame, "window");
  if (windowName === undefined) degraded.push("帧 strings 缺 window：不标窗名（不按请求参数猜）");
  const note = stringOf(frame, "note");
  if (note === undefined) degraded.push("帧 strings 缺 note：状态角标不自行编造结论");
  const resolutionNote = stringOf(frame, "resolution_note");
  const nonfinite = stringOf(frame, "nonfinite_policy");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: tMeta ?? opts.t,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  // —— 排版：上「两音谱」整宽，下「能否分辨 vs N」与「对拍误差」并排；底部留关键数字横条 ——
  const strip = bottomStrip(scene, width, height, 4);
  const plot = topWithTwoBelow(width, height, { topShare: 0.55, footer: strip.footer });
  const specRect = plot.top;
  const thrRect = plot.leftBottom;
  const errRect = plot.rightBottom;

  // 谱面板：横轴只画到「第二音 + 一点余量」（df 最大 8 时也不会把第一音压扁）。
  const f2 = dfWant === undefined ? 4 : 2 + dfWant;
  const specXlim = niceLimits(0, Math.max(3, f2 * 1.4), 6);
  const specVisible = sliceRange(freq, specDb, specXlim[0], specXlim[1]);
  const specFloor = finiteMinMax(specVisible.y)?.[0] ?? -80;
  const specAxis = makeAxis("spec", specRect, {
    xlim: specXlim,
    ylim: [Math.min(-60, specFloor), 2],
    xlabel: "f（单位 1/T，本页固定 T = 1）",
    ylabel: axisYlabel(VIZ, "spec", "dB"),
    title: axisTitle(VIZ, "spec", "两音谱（dB，相对峰值）"),
    autoscaled: axisAutoscaled(VIZ, "spec"),
    xticks: niceTicks(specXlim[0], specXlim[1], 6),
  });

  // 「能否分辨 vs 观测点数」：y 只有 0/1，横轴线性（manifest 声明 linear），
  // 刻度直接取帧里的 span_n 采样点（它们本身不是整值，niceTicks 会对不上号）。
  const spanRange = minMaxOfAll([spanN]) ?? [32, 4096];
  const thrAxis = makeAxis("threshold", thrRect, {
    xlim: [spanRange[0], spanRange[1]],
    ylim: [-0.25, 1.25],
    xlabel: "观测点数 N（T 固定 = 1）",
    ylabel: axisYlabel(VIZ, "threshold", "可分辨（1/0）"),
    title: axisTitle(VIZ, "threshold", "能否分辨 vs 观测长度"),
    autoscaled: axisAutoscaled(VIZ, "threshold"),
    xticks: thinTicks(spanN, 4),
  });

  // 对拍副轴（规则 3）：`error` 是整段 max|FFT−DFT| 铺满 N 点，对数纵轴。
  const errPos = minMaxOfAll([absOf(error)]);
  const errYlim: [number, number] = errPos === undefined ? [1e-18, 1e-12] : niceLimits(errPos[0], errPos[1], 5, { log: true, pad: 0.25 });
  if (errPos === undefined) degraded.push("帧 error 没有 >0 的有限值：对数副轴用占位范围 [1e-18, 1e-12]，不画曲线");
  const errAxis = makeErrorAxis("error", errRect, {
    xlim: [spanRange[0], spanRange[1]],
    ylim: errYlim,
    yscale: "log",
    xlabel: "观测点数 N（与左侧同一段）",
    ylabel: axisYlabel(VIZ, "error", "max|delta|"),
    title: axisTitle(VIZ, "error", "|FFT − DFT|"),
    autoscaled: true,
    xticks: thinTicks(spanN, 4),
  });
  scene.axes.push(specAxis, thrAxis, errAxis);

  // —— 从帧谱线里读峰与谷（判据三要素：两个局部极大 + 谷低于矮峰）——
  const peaks = findPeaks(specVisible.x, specVisible.y, 3);
  const valley = peaks.length >= 2 ? findValley(specVisible.x, specVisible.y, peaks[0] as number, peaks[1] as number) : undefined;

  // —— 图层 1~4：谱 + 分辨率尺 + 峰/谷 ——
  const refs: LineSeries[] = [];
  const rulerY = (i: number): number => specAxis.ylim[1] - 4 - i * 6;
  if (dfWant !== undefined) {
    refs.push(series([0, Math.min(dfWant, specXlim[1])], [rulerY(0), rulerY(0)], pal.curves[2], { width: 2.5}));
  }
  if (sepMeasured !== undefined) {
    refs.push(series([0, Math.min(sepMeasured, specXlim[1])], [rulerY(1), rulerY(1)], pal.marker, { width: 2.5}));
  }
  if (rayleigh !== undefined) {
    refs.push(series([0, Math.min(rayleigh, specXlim[1])], [rulerY(2), rulerY(2)], pal.reference, { width: 2.5}));
  }
  refs.push(series([specXlim[0], specXlim[1]], [-3, -3], pal.reference, { dash: [5, 4], width: 1}));
  // 三条尺是**同色横条**，光看长度分不出哪条是 df、哪条是实测/Rayleigh：在尺下面放一块
  // **同色图例**（`text()` 的坐标是像素，不是数据坐标——尺的 y 是数据坐标，必须换算）。
  const legend: Array<[string, string]> = [];
  if (dfWant !== undefined) legend.push([pal.curves[2], `df 尺 = ${fmtOpt(dfWant, 4)}（两音间隔）`]);
  if (sepMeasured !== undefined) legend.push([pal.marker, `实测最小可分辨 = ${fmtOpt(sepMeasured, 4)}/T`]);
  if (rayleigh !== undefined) legend.push([pal.reference, `Rayleigh 判据 = ${fmtOpt(rayleigh, 4)}/T`]);
  legend.forEach(([color, body], i) => {
    scene.texts.push(text(specRect[0] + 6, specRect[1] + 42 + i * 15, body, { role: "note", size: 11, color }));
  });
  if (peaks.length >= 2) {
    for (const p of peaks) {
      refs.push(series([specVisible.x[p] as number, specVisible.x[p] as number], [specAxis.ylim[0], specVisible.y[p] as number], pal.cursor, { dash: [2, 3], width: 1 }));
    }
  }
  if (valley !== undefined) {
    refs.push(series([valley.x, valley.x], [specAxis.ylim[0], valley.y], pal.warn, { dash: [3, 3], width: 1}));
  }
  scene.layers.push(lineLayer("spec", refs));
  const specSeries = segmentedSeries(specVisible.x, specVisible.y, pal.curves[0], { width: 1.8});
  if (specSeries.length > 0) scene.layers.push(lineLayer("spec", specSeries));
  else degraded.push("帧 spectrum_db 没有可绘的有限值：谱面板不画曲线（也就不报峰/谷）");
  if (peaks.length >= 2) {
    scene.layers.push(
      scatterLayer("spec", [specVisible.x[peaks[0] as number] as number, specVisible.x[peaks[1] as number] as number], [specVisible.y[peaks[0] as number] as number, specVisible.y[peaks[1] as number] as number], {
        r: 4,
        color: pal.marker
      }),
    );
  } else {
    degraded.push(`从帧谱线里只找到 ${peaks.length} 个局部极大（可分辨判据要两个）：画面不标峰`);
  }
  if (valley !== undefined) {
    scene.layers.push(scatterLayer("spec", [valley.x], [valley.y], { r: 4, color: pal.warn}));
  }

  // —— 图层 5~6：能否分辨 vs N ——
  const thrRefs: LineSeries[] = [
    series([spanRange[0], spanRange[1]], [1, 1], pal.reference, { dash: [6, 4], width: 1}),
    series([spanRange[0], spanRange[1]], [0, 0], pal.reference, { dash: [2, 4], width: 1}),
  ];
  const stepSeries = segmentedSeries(spanN, spanRatio, pal.curves[0], { width: 1.8, marker: "o"});
  if (stepSeries.length > 0) thrRefs.push(...stepSeries);
  else degraded.push("帧 span_ratio 没有可绘的有限值：能否分辨面板只画参考线");
  if (n !== undefined) {
    thrRefs.push(series([n, n], [thrAxis.ylim[0], thrAxis.ylim[1]], pal.marker, { dash: [3, 3], width: 1}));
  }
  scene.layers.push(lineLayer("threshold", thrRefs));
  if (n !== undefined) {
    const ratioAtN = valueAt(spanN, spanRatio, n);
    if (ratioAtN !== undefined) {
      scene.layers.push(scatterLayer("threshold", [n], [ratioAtN], { r: 4, color: pal.marker}));
    } else {
      degraded.push(`帧 span_n 里没有 N = ${fmt(n, 6)} 这一档：不标当前 N 的判据点（不插值）`);
    }
  }

  // —— 图层 7：对拍误差（requiredAux）——
  const errSeries = segmentedSeries(indexOf(error.length), absOf(error), pal.residual, {
    width: 1.4,
    accept: (v) => v > 0
  });
  if (errSeries.length > 0) scene.layers.push(lineLayer("error", errSeries));
  else degraded.push("帧 error 没有 >0 的有限值：对数副轴画不出任何点");

  // —— 文本：面板内只放"身份行"，长句子全部进底部横条 ——
  const summary = paramSummary(frame, opts, [
    { name: "n", label: "N" },
    { name: "pad", label: "pad" },
    { name: "df", label: "两音间隔" },
    { name: "amp2", label: "幅度比" },
  ]);
  degraded.push(...summary.degraded);
  const specTop: string[] = [
    summary.text + (windowName === undefined ? "" : `　窗 ${windowName}`),
    `T = ${fmt(T_RECORD, 3)}（后端固定：N 只改采样密度，不改观测时长）`,
  ];
  if (separableMeta !== undefined) {
    specTop.push(`可分辨 = ${separableMeta > 0.5 ? "True" : "False"}（帧 meta.separable）：两个局部极大 + 谷低于矮峰`);
    if (resolutionNote !== undefined) specTop.push(`帧 resolution_note：${resolutionNote}`);
  }
  scene.texts.push(...textBlockRight(specRect[0] + specRect[2] - 6, specRect[1] + 4, specRect[1] + specRect[3] - 6, specTop, pal.muted, unitBudget(specRect[2]), "top"));
  const specBot: string[] = [];
  if (valley !== undefined && peaks.length >= 2) {
    const shortPeak = Math.min(specVisible.y[peaks[0] as number] as number, specVisible.y[peaks[1] as number] as number);
    specBot.push(`谷 ${fmt(valley.y, 4)} dB 比矮峰低 ${fmt(shortPeak - valley.y, 4)} dB：这才叫「分得开」（用「峰够不够尖」判会得出相反结论）`);
  } else if (peaks.length >= 2) {
    specBot.push("两峰之间没有低于矮峰的谷：判据判它「分不开」——这就是分辨率不够的样子");
  } else {
    specBot.push(`帧谱线上只找到 ${peaks.length} 个局部极大：判据要两个才算「分得开」，这里判它分不开（不硬凑两个峰）`);
  }
  if (note !== undefined) specBot.push(`帧 note：${note.split("；").slice(0, 2).join("；")}`);
  if (specBot.length > 0) {
    scene.texts.push(...textBlock(specRect[0] + 6, specRect[1] + 4, specRect[1] + specRect[3] - 6, specBot, pal.marker, unitBudget(specRect[2]), "bottom"));
  }

  const thrTop: string[] = [
    `${spanN.length} 档 N ∈ [${fmt(spanRange[0], 6)}, ${fmt(spanRange[1], 6)}] 上的判据：${Array.from(spanRatio).map((v) => (v > 0.5 ? "1" : "0")).join("")}`,
  ];
  scene.texts.push(...textBlock(thrRect[0] + 6, thrRect[1] + 4, thrRect[1] + thrRect[3] - 6, thrTop, pal.muted, unitBudget(thrRect[2]), "top"));

  const errLines: string[] = [
    maxError === undefined
      ? `帧 meta 缺 max_error：只画曲线 ${error.length} 点，不报数值（不重算）`
      : `|FFT − DFT| = ${sci(maxError)}（帧 max_error）：它证明这条谱算得对，与能不能分辨无关`,
  ];
  if (nonfinite !== undefined) errLines.push(`地板口径（帧 strings）：${nonfinite}`);
  scene.texts.push(...textBlock(errRect[0] + 6, errRect[1] + 4, errRect[1] + errRect[3] - 6, errLines, pal.muted, unitBudget(errRect[2]), "top"));

  // 底部横条：分辨率的两条账（T 决定阈值；pad 只加密栅格）。
  const stripLines: string[] = [
    `实测最小可分辨 ${fmtOpt(sepMeasured, 4)}/T vs Rayleigh 1/T = ${fmtOpt(rayleigh, 4)}/T` +
      (sepOverMainlobe === undefined ? "" : `；sep/主瓣宽度 = ${fmt(sepOverMainlobe, 4)}`) +
      (mainlobe === undefined ? "" : `（主瓣 ${fmt(mainlobe, 4)} 格）`),
    `grid_df（随 pad 变）= 1/(N·pad) = ${fmtOpt(gridDf, 6)}；true_df（不随 pad 变）= 1/N = ${fmtOpt(trueDf, 6)}` +
      (pad === undefined ? "" : `（pad = ${fmt(pad, 3)}）`) +
      "；横轴 f 的单位是 1/T",
    "补零只加密栅格，不让两个近邻音分开；要分开得采更久（T 更大）",
    "整数格下四个窗都给 1.000/T；半格位置（2.25/T、3.75/T）Hann 并包（pitfalls §11）；定位精度 0.0422 格 ≠ 分辨率（§16）",
  ];
  scene.texts.push(...textBlock(specRect[0], strip.stripTop, strip.limit, stripLines, pal.muted, unitBudget(width - specRect[0] - 24), "top"));

  // —— 角标 ——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(badge("autoscale", "对拍副轴：自适应（log 纵轴按本帧误差取整；自适应会掩盖量级变化）", 8, 8), SLOT.auxAxis, width),
  );
  scene.badges.push(
    placeBadge(
      badge(
        "critical",
        clipBadgeText(`pad = ${fmtOpt(pad, 3)}　临界 1（补零只加密栅格 grid_df→grid_df/pad，true_df 不动）` + ((pad ?? 1) > 1 ? "　⚠ 已越过" : "")),
        8,
        8,
      ),
      SLOT.critical,
      width,
    ),
  );
  if (dfWant !== undefined) {
    scene.badges.push(placeBadge(criticalBadge("df", dfWant, criticalOf(frame, "df", 1.44), { beyond: "below" }), SLOT.critical2, width));
  }
  const status: string[] = [];
  if (separableMeta !== undefined) status.push(separableMeta > 0.5 ? "可分辨" : "并成一个包");
  if (sepMeasured !== undefined) status.push(`阈值 ${fmt(sepMeasured, 3)}/T`);
  if (amp2 !== undefined) status.push(`幅度比 ${fmt(amp2, 3)}`);
  if (note !== undefined) status.push(note);
  scene.badges.push(placeBadge(badge("status", clipBadgeText(status.join("；")), 8, 8), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  const nBad = countNonFinite(specDb) + countNonFinite(freq) + countNonFinite(spanRatio) + countNonFinite(spanN) + countNonFinite(error);
  applyOverflow(scene, nBad, { slot: SLOT.overflow, width });

  scene.texts = resolveTextOverlaps(scene.texts, [8, 24, width - 16, strip.limit - 24]);
  return scene;
};

// --------------------------------------------------------------------------- //
// 局部工具
// --------------------------------------------------------------------------- //

/** `0…n−1` 的整数序列（误差曲线是"逐点"的，横轴就是样本序号）。 */
function indexOf(n: number): number[] {
  return Array.from({ length: n }, (_, i) => i);
}

/** 从 `values` 里取前 `count` 个采样点当刻度（它们本身不是整值，niceTicks 会对不上）。 */
function thinTicks(values: ArrayLike<number>, count: number): number[] {
  const total = values.length;
  if (total === 0) return [];
  const step = Math.max(1, Math.ceil(total / Math.max(1, count)));
  const out: number[] = [];
  for (let i = 0; i < total; i += step) out.push(values[i] as number);
  return out;
}

/** 在离散网格上取 `x = xq` 处的值（精确匹配才返回；不插值——N 是离散档位）。 */
function valueAt(xs: ArrayLike<number>, ys: ArrayLike<number>, xq: number): number | undefined {
  for (let i = 0; i < Math.min(xs.length, ys.length); i += 1) {
    if (Math.abs((xs[i] as number) - xq) <= 1e-9 * Math.max(1, Math.abs(xq))) return ys[i] as number;
  }
  return undefined;
}

/**
 * 找**局部极大**（比左右都高，端点不比），按高度排序取前 `count` 个。
 * 边界情况"两个峰等高/相邻"都返回实际找到的东西——找不到两个峰本身就是「分不开」的信号。
 */
function findPeaks(xs: ArrayLike<number>, ys: ArrayLike<number>, count: number): number[] {
  const found: number[] = [];
  const n = Math.min(xs.length, ys.length);
  for (let i = 1; i + 1 < n; i += 1) {
    const a = ys[i - 1] as number;
    const b = ys[i] as number;
    const c = ys[i + 1] as number;
    if (Number.isFinite(a) && Number.isFinite(b) && Number.isFinite(c) && b > a && b >= c) found.push(i);
  }
  return found.sort((i, j) => (ys[j] as number) - (ys[i] as number)).slice(0, count);
}

/** 两个峰之间最低的那个样本（"谷"）：判据要求的"低于矮峰的谷"就是它。 */
function findValley(xs: ArrayLike<number>, ys: ArrayLike<number>, i0: number, i1: number): { x: number; y: number } | undefined {
  const lo = Math.min(i0, i1);
  const hi = Math.min(Math.max(i0, i1), Math.min(xs.length, ys.length) - 1);
  let at = -1;
  for (let i = lo; i <= hi; i += 1) {
    if (at < 0 || (ys[i] as number) < (ys[at] as number)) at = i;
  }
  return at < 0 ? undefined : { x: xs[at] as number, y: ys[at] as number };
}

/** 供测试引用：这份 model 依赖的帧数组名。 */
export const RESOLUTION_ARRAYS = ["spectrum_db", "freq", "span_ratio", "span_n", "error"] as const;
