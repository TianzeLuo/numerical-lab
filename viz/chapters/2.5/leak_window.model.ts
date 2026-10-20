/**
 * `leak_window`（2.5 章）的 model：四种窗的窗形 + 加窗后的 dB 谱 + 解析窗谱对拍副轴。
 *
 * 面板与图层（顺序即绘制顺序，测试按此断言）
 * ------------------------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | line | `shape` | 四条窗 `window_shapes`（行序 = `rect/hann/hamming/blackman`），**当前窗加粗** |
 * | 2 | line | `shape` | `x_windowed`（加窗后的信号，虚线）：非整周期截断的样子 |
 * | 3 | line | `spec` | 渐近包络理论线 `envelope`（幅度→dB；**只在旁瓣区 `f·T ≥ 锚点` 画**） |
 * | 4 | line | `spec` | 加窗后的 dB 谱 `spectrum_db` |
 * | 5 | line | `spec` | 峰值旁瓣水平线（`meta.peak_sidelobe_db`）与 Harris 表值水平线（`meta.harris_sidelobe_db`） |
 * | 6 | scatter | `spec` | 理论线锚点 `(envelope_anchor, envelope_anchor_db)`：窗自己的**最高旁瓣** |
 * | 7 | line | `error` | `error`（`|FFT(窗)|` vs 解析连续谱的最大偏差，逐点） |
 *
 * 角标槽位（0 来源 / 1 主轴固定 / 2 副轴自适应 / 3 `offset` 临界 / 4 状态 / 6 溢出 / 7 降级）。
 *
 * 三条刻意的决定
 * --------------
 * 1. **旁瓣低 ≠ 衰减快（本页的第二条防误导）**：面板文本里峰值旁瓣（dB）与包络幂次 `p`
 *    **必须同时出现**，并给出 Hamming（−42.7 dB 但 p=1）与 Hann（−31.5 dB 但 p=3）的对照
 *    ——只报旁瓣会得出"Hamming 更好"的反结论（`chapters/2.5/pitfalls.md` §9）。
 * 2. **理论线锚在窗自己的最高旁瓣上，且主瓣区不画**：帧的 `envelope` 是 `a_anchor·(f/f_anchor)^-p`
 *    （幅度），锚点 `meta.envelope_anchor / envelope_anchor_db` 来自 `peak_sidelobe_level`。
 *    `f < 锚点` 那段是主瓣，`f^-p` 在那里没有意义——切片不画，并写明"主瓣区不适用"。
 *    横轴用 `f·T`（每记录周期数）而不是 `f·N` 的点数口径（pitfalls §22 的 36 dB 事故）。
 * 3. **误差曲线必须是同类对同类**：`error` 是 `|FFT(窗)|` 与解析窗谱 `window_spectrum(key, f·T)`
 *    的最大偏差（`meta.rel_error`）。面板文本写明被减数/减数；用加窗信号的谱去减窗的谱会得到
 *    `offset=0` 时恒等于 1.0 的假误差（pitfalls §21）。
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
  toNumeric,
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
  toDb,
  unitBudget,
} from "./common.js";

const VIZ = "leak_window";
const TITLE = specTitle(VIZ, "泄漏与窗函数旁瓣");

/** 帧 `window_shapes` 的**行序**（后端 `WINDOW_NAMES`，也写在数组 label 里）。 */
const WINDOW_ORDER = ["rect", "hann", "hamming", "blackman"] as const;

/** 四种窗的旁瓣包络幂次（manifest `theory` 与 `numcore.transform.fft.WINDOW_ENVELOPE_POWER`）。 */
const ENVELOPE_POWER: Record<string, number> = { rect: 1, hann: 3, hamming: 1, blackman: 3 };

export const leakWindowModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需数组 ——
  const shapes = frame.matrix("window_shapes");
  const shape = requireArray(frame, "window_shape");
  const xWindowed = requireArray(frame, "x_windowed");
  const specDb = requireArray(frame, "spectrum_db");
  const freq = requireArray(frame, "freq");
  const envelope = requireArray(frame, "envelope");
  const error = requireArray(frame, "error");
  requireSameLength(freq, specDb, "帧数组 freq 与 spectrum_db");
  requireSameLength(freq, envelope, "帧数组 freq 与 envelope");
  requireSameLength(shape, xWindowed, "帧数组 window_shape 与 x_windowed");
  if (shapes.rows !== WINDOW_ORDER.length) {
    throw new ModelInputError(`帧 ${CHAPTER}/${VIZ} 的 window_shapes 有 ${shapes.rows} 行，期望 ${WINDOW_ORDER.length}（rect/hann/hamming/blackman）`);
  }
  const pitfalls = requirePitfalls(frame);

  // —— 可选字段 ——
  const degraded: string[] = [];
  const tMeta = metaNumber(frame, "t");
  if (tMeta === undefined) degraded.push("帧 meta 缺 t：scene.t 改用请求 t（两者可能不同）");
  const n = metaNumber(frame, "n");
  if (n === undefined) degraded.push(`帧 meta 缺 n：窗长按 shape 数组长度给（${shape.length}）`);
  const key = stringOf(frame, "window");
  if (key === undefined) degraded.push("帧 strings 缺 window：不标当前窗（不按请求参数猜）");
  const offset = metaNumber(frame, "offset");
  if (offset === undefined) degraded.push("帧 meta 缺 offset：不标频率偏移与整周期判据（不猜）");
  const peakDb = metaNumber(frame, "peak_sidelobe_db");
  if (peakDb === undefined) degraded.push("帧 meta 缺 peak_sidelobe_db：不画实测旁瓣水平线");
  const harrisDb = metaNumber(frame, "harris_sidelobe_db");
  if (harrisDb === undefined) degraded.push("帧 meta 缺 harris_sidelobe_db：不画 Harris 表值线（无法对照）");
  const mainlobe = metaNumber(frame, "mainlobe_bins");
  const harrisMainlobe = metaNumber(frame, "harris_mainlobe_bins");
  const gain = metaNumber(frame, "coherent_gain");
  const pEnv = metaNumber(frame, "envelope_power");
  const anchor = metaNumber(frame, "envelope_anchor");
  const anchorDb = metaNumber(frame, "envelope_anchor_db");
  if (anchor === undefined || anchorDb === undefined) degraded.push("帧 meta 缺 envelope_anchor / envelope_anchor_db：不标理论线锚点（锚点必须来自窗自己）");
  const relError = metaNumber(frame, "rel_error");
  const note = stringOf(frame, "note");
  if (note === undefined) degraded.push("帧 strings 缺 note：状态角标不自行编造结论");
  const cola = stringOf(frame, "cola_note");
  const nonfinite = stringOf(frame, "nonfinite_policy");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: tMeta ?? opts.t,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  // —— 排版：左列「窗形（上）+ 对拍（下）」，右列整高给 dB 谱；底部留关键数字横条 ——
  const strip = bottomStrip(scene, width, height, 4);
  const panels = columnsLayout(width, height, { share: 0.52, gap: 54, footer: strip.footer });
  const shapeRect = panels.main;
  const errRect = panels.error;
  const specRect = panels.side;

  const idx = key === undefined ? -1 : WINDOW_ORDER.indexOf(key as (typeof WINDOW_ORDER)[number]);
  if (key !== undefined && idx < 0) degraded.push(`帧 strings.window=${key} 不在 rect/hann/hamming/blackman 里：无法标出当前窗`);

  // 窗形面板：横轴是窗内样本 m（0…N−1），纵轴是 w[m] 与 x·w（同一量纲，可同轴）。
  const mAxis = Array.from({ length: shape.length }, (_, i) => i);
  const shapeRange = minMaxOfAll([shape, xWindowed]) ?? [0, 1];
  const shapeAxis = makeAxis("shape", shapeRect, {
    xlim: [0, Math.max(1, shape.length - 1)],
    ylim: niceLimits(shapeRange[0], shapeRange[1], 5, { pad: 0.1 }),
    xlabel: "m（窗内样本序号）",
    ylabel: axisYlabel(VIZ, "shape", "w[m]"),
    title: axisTitle(VIZ, "shape", "四种窗的时域形状"),
    autoscaled: axisAutoscaled(VIZ, "shape"),
    xticks: niceTicks(0, Math.max(1, shape.length - 1), 6),
  });

  // 对拍副轴（规则 3）：`error` 是 |FFT(窗)| 与解析窗谱的最大偏差（逐点相同），对数纵轴。
  const errPos = minMaxOfAll([absOf(error)]);
  const errYlim: [number, number] = errPos === undefined ? [1e-18, 1e-2] : niceLimits(errPos[0], errPos[1], 5, { log: true, pad: 0.25 });
  if (errPos === undefined) degraded.push("帧 error 没有 >0 的有限值：对数副轴用占位范围 [1e-18, 1e-2]，不画曲线");
  const errAxis = makeErrorAxis("error", errRect, {
    xlim: [0, Math.max(1, shape.length - 1)],
    ylim: errYlim,
    yscale: "log",
    xlabel: "m（与上面同一段窗内样本）",
    ylabel: axisYlabel(VIZ, "error", "rel"),
    title: axisTitle(VIZ, "error", "解析窗谱 vs |FFT| 的相对差"),
    autoscaled: true,
    xticks: niceTicks(0, Math.max(1, shape.length - 1), 6),
  });

  // dB 谱面板：横轴取到 16（窗谱在 f·T > 16 之后已经贴地板，画出来只是压扁有用的一段）。
  const specXlim = niceLimits(0, Math.min(16, (freq[freq.length - 1] as number) ?? 16), 6);
  const envDb = toDb(envelope, -160);
  // 理论包络只在**旁瓣区**（f·T ≥ 锚点）画：主瓣区 f^-p 没有意义（见文件头第 2 条）。
  const anchorX = anchor ?? 0;
  const envVisible = sliceRange(freq, envDb, Math.max(specXlim[0], anchorX), specXlim[1]);
  const specVisible = sliceRange(freq, specDb, specXlim[0], specXlim[1]);
  const specFloor = finiteMinMax(specVisible.y)?.[0] ?? -120;
  const specAxis = makeAxis("spec", specRect, {
    xlim: specXlim,
    ylim: [Math.min(-100, specFloor), Math.max(5, anchorDb === undefined ? 0 : anchorDb + 5)],
    xlabel: "f·T（每记录周期数；等于 T = N·dt 时的 f·N）",
    ylabel: axisYlabel(VIZ, "spec", "dB"),
    title: axisTitle(VIZ, "spec", "加窗后的 dB 谱（相对峰值）"),
    autoscaled: axisAutoscaled(VIZ, "spec"),
    xticks: niceTicks(specXlim[0], specXlim[1], 6),
  });
  scene.axes.push(shapeAxis, errAxis, specAxis);

  // —— 图层 1~2：窗形 ——
  const shapeSeries: LineSeries[] = [];
  for (let r = 0; r < shapes.rows; r += 1) {
    const row = new Float64Array(shape.length);
    for (let c = 0; c < shape.length; c += 1) row[c] = shapes.values[r * shapes.cols + c] as number;
    shapeSeries.push(
      series(mAxis, row, r === idx ? pal.curves[0] : pal.reference, {
        width: r === idx ? 2.2 : 1,
        dash: r === idx ? undefined : [4, 3],
        label: r === idx ? "当前窗" : "其它窗",
      }),
    );
  }
  scene.layers.push(lineLayer("shape", shapeSeries));
  const xwSeries = segmentedSeries(mAxis, xWindowed, pal.exact, {
    width: 1.2,
    dash: [5, 4],
    label: "加窗后的信号",
  });
  if (xwSeries.length > 0) scene.layers.push(lineLayer("shape", xwSeries));

  // —— 图层 3~6：dB 谱 + 旁瓣水平线 + 锚点 ——
  const specRefs: LineSeries[] = [];
  if (peakDb !== undefined) {
    specRefs.push(series([0, specXlim[1]], [peakDb, peakDb], pal.marker, { width: 1.4, dash: [6, 3], label: "峰值高度" }));
  }
  if (harrisDb !== undefined) {
    specRefs.push(series([0, specXlim[1]], [harrisDb, harrisDb], pal.exact, { width: 1.2, dash: [2, 3], label: "Harris 旁瓣参考" }));
  }
  if (specRefs.length > 0) scene.layers.push(lineLayer("spec", specRefs));
  const envSeries = segmentedSeries(envVisible.x, envVisible.y, pal.residual, {
    width: 1.6,
    dash: [7, 4],
    label: "旁瓣包络",
  });
  if (envSeries.length > 0) scene.layers.push(lineLayer("spec", envSeries));
  const specSeries = segmentedSeries(specVisible.x, specVisible.y, pal.curves[0], { width: 1.8, label: "加窗谱" });
  if (specSeries.length > 0) scene.layers.push(lineLayer("spec", specSeries));
  else degraded.push("帧 spectrum_db 没有可绘的有限值：谱面板不画曲线");
  if (anchor !== undefined && anchorDb !== undefined && anchor <= specXlim[1]) {
    scene.layers.push(scatterLayer("spec", [anchor], [anchorDb], { r: 4, color: pal.marker, label: "当前旁瓣锚点" }));
  }

  // —— 图层 7：对拍误差（requiredAux）——
  const errSeries = segmentedSeries(mAxis, absOf(error), pal.residual, {
    width: 1.4,
    accept: (v) => v > 0,
    label: "解析窗谱 vs |FFT| 的差",
  });
  if (errSeries.length > 0) scene.layers.push(lineLayer("error", errSeries));
  else degraded.push("帧 error 没有 >0 的有限值：对数副轴画不出任何点");

  // —— 文本：面板内只放"身份行"，长句子全部进底部横条 ——
  const summary = paramSummary(frame, opts, [
    { name: "n", label: "N" },
    { name: "offset", label: "offset" },
    { name: "pad", label: "pad" },
  ]);
  degraded.push(...summary.degraded);
  const shapeLines: string[] = [summary.text];
  if (key !== undefined) {
    shapeLines.push(
      `窗 = ${key}（帧 strings.window）；行序 ${WINDOW_ORDER.join("/")}（帧数组 label）` +
        (offset === undefined ? "" : Math.abs(offset) < 1e-12 ? "；offset=0：整周期采样，矩形窗不泄漏" : `；offset=${fmt(offset, 3)} 格：非整周期截断才泄漏`),
    );
  }
  shapeLines.push(
    `四种窗（帧数组行序）：${WINDOW_ORDER.map((w) => `${w} p=${fmtOpt(ENVELOPE_POWER[w], 2)}`).join("　")}；当前窗加粗，其余细虚线`,
  );
  if (cola !== undefined) shapeLines.push(`帧 cola_note：${cola}`);
  scene.texts.push(
    ...textBlock(shapeRect[0] + 6, shapeRect[1] + 4, shapeRect[1] + shapeRect[3] - 6, shapeLines, Math.abs(offset ?? 0) < 1e-12 ? pal.muted : pal.warn, unitBudget(shapeRect[2]), "top"),
  );

  const errLines: string[] = [
    relError === undefined
      ? `帧 meta 缺 rel_error：只画曲线 ${error.length} 点，不报数值（不重算）`
      : `最大偏差 = ${fmt(relError, 4)}（帧 rel_error）：被减数 |FFT(窗)|，减数解析窗谱 window_spectrum(窗, f·T)`,
  ];
  errLines.push("两条曲线必须同类对同类：拿加窗信号的谱去减窗的谱，offset=0 时会得到恒等于 1.0 的假误差（pitfalls §21）");
  if (nonfinite !== undefined) errLines.push(`地板口径（帧 strings）：${nonfinite}`);
  scene.texts.push(...textBlock(errRect[0] + 6, errRect[1] + 4, errRect[1] + errRect[3] - 6, errLines, pal.muted, unitBudget(errRect[2]), "top"));

  // 谱面板顶部：**旁瓣（dB）与包络幂次 p 同时出现**（本页防误导的核心）。
  const specTop: string[] = [
    `峰值旁瓣 实测 ${fmtOpt(peakDb, 4)} dB vs Harris 表值 ${fmtOpt(harrisDb, 4)} dB`,
    `−3 dB 主瓣 ${fmtOpt(mainlobe, 4)} 格（Harris ${fmtOpt(harrisMainlobe, 4)} 格）；相干增益 ${fmtOpt(gain, 5)}`,
    `包络幂次 p = ${fmtOpt(pEnv, 3)}（幅度读数要除以相干增益）` +
      (pEnv === undefined ? "" : `：10→1000 倍频程上包络比 ${sci(Math.pow(100, -pEnv), 1)}`),
  ];
  scene.texts.push(
    ...textBlockRight(specRect[0] + specRect[2] - 6, specRect[1] + 4, specRect[1] + specRect[3] - 6, specTop, pal.muted, unitBudget(specRect[2]), "top"),
  );
  const specBot: string[] = [
    anchor === undefined || anchorDb === undefined
      ? "帧未给锚点：理论线不画（锚点必须来自窗自己的最高旁瓣）"
      : `理论线锚点 f·T = ${fmt(anchor, 4)}、${fmt(anchorDb, 5)} dB（窗自己的最高旁瓣）；主瓣区不画 f^−p`,
  ];
  scene.texts.push(
    ...textBlock(specRect[0] + 6, specRect[1] + 4, specRect[1] + specRect[3] - 6, specBot, pal.muted, unitBudget(specRect[2]), "bottom"),
  );

  // 底部横条：旁瓣/主瓣/增益的账 + "同一类"对拍口径。
  const stripLines: string[] = [
    "旁瓣低 ≠ 衰减快：hamming 旁瓣 −42.7 dB 但 p=1；hann −31.5 dB 但 p=3（10→1000 倍频程包络比 1e-2 vs 1e-6）",
    "选窗要看三个数：峰值旁瓣、−3 dB 主瓣宽度、相干增益（pitfalls §8）",
    "Blackman 旁瓣 −58.1 dB 的代价：主瓣 1.641 格、增益 0.4184；矩形窗旁瓣只 −13.3 dB",
    note === undefined
      ? "锚点是窗自己的最高旁瓣（rect 1.43、hamming 4.50，不是第一个零点，pitfalls §22）"
      : `帧 note：${note}`,
  ];
  scene.texts.push(...textBlock(shapeRect[0], strip.stripTop, strip.limit, stripLines, pal.muted, unitBudget(width - shapeRect[0] - 24), "top"));

  // —— 角标 ——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(badge("autoscale", "对拍副轴：自适应（log 纵轴按本帧偏差取整；自适应会掩盖量级变化）", 8, 8), SLOT.auxAxis, width),
  );
  if (offset !== undefined) {
    scene.badges.push(placeBadge(criticalBadge("offset", offset, criticalOf(frame, "offset", 0.5), { beyond: "above" }), SLOT.critical, width));
  }
  const status: string[] = [];
  if (key !== undefined) status.push(`窗=${key}`);
  if (mainlobe !== undefined) status.push(`主瓣=${fmt(mainlobe, 3)} 格`);
  if (pEnv !== undefined) status.push(`p=${fmt(pEnv, 3)}`);
  if (peakDb !== undefined) status.push(`旁瓣=${fmt(peakDb, 3)} dB`);
  if (note !== undefined) status.push(note);
  scene.badges.push(placeBadge(badge("status", clipBadgeText(status.join("；")), 8, 8), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  const nBad =
    countNonFinite(toNumeric(shapes.values)) +
    countNonFinite(shape) +
    countNonFinite(xWindowed) +
    countNonFinite(specDb) +
    countNonFinite(freq) +
    countNonFinite(envelope) +
    countNonFinite(error);
  applyOverflow(scene, nBad, { slot: SLOT.overflow, width });

  scene.texts = resolveTextOverlaps(scene.texts, [8, 24, width - 16, strip.limit - 24]);
  return scene;
};

/** 供测试引用：这份 model 依赖的帧数组名。 */
export const LEAK_WINDOW_ARRAYS = ["window_shapes", "window_shape", "x_windowed", "spectrum_db", "freq", "envelope", "error"] as const;
