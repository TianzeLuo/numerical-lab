/**
 * `signal_spectrum`（2.5 章）的 model：时域波形 ↔ 单边幅度谱双联动 + `|FFT − DFT|` 副轴。
 *
 * 面板与图层（顺序即绘制顺序，测试按此断言）
 * ------------------------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | line | `time` | `x = 0` 参考线（虚线） |
 * | 2 | line | `time` | 时域信号 `x`（帧数组，N 点） |
 * | 3 | scatter | `time` | 采样点（仅 N ≤ 64 时画；密采样时点会糊成一片） |
 * | 4 | line | `spec` | 谐波位置参考线 `f0 / 2f0 / 3f0`（虚线竖线，来自帧 `meta.f0`）+ −3 dB 半高全宽横段 + 谱线地板 |
 * | 5 | line | `spec` | 单边幅度谱 `spectrum_db`（**只画 `f ≤ xHi` 这一段**，其余到 Nyquist 全在地板上） |
 * | 6 | scatter | `spec` | `pad > 1` 时叠出**原栅格（pad=1）**的读数（stride = pad 抽取本帧谱线） |
 * | 7 | line | `error` | `error`（逐点 `|FFT − DFT|`，帧里是整段最大值铺满 N 点） |
 *
 * 角标槽位（固定，见 `common.ts::SLOT`）：0 来源 / 1 主轴固定 / 2 误差轴自适应 /
 * 3 `n` 临界 / 4 状态 / 5 `pad` 临界（补零只加密栅格）/ 6 溢出 / 7 降级。
 *
 * 三条刻意的决定
 * --------------
 * 1. **`df` 与 `df_true` 并排报，且永不合并成一个"频率分辨率"**：帧 `meta.df = 1/N_pad`
 *    随 `pad` 变，`meta.df_true = 1/N` 不随 `pad` 变。这一页最容易骗人的地方就是
 *    "谱线变光滑了 ⇒ 分辨率提高了"，所以两个数各占一行、各自标注是否随 `pad` 变，
 *    `pad > 1` 时用警示色（见 `frame.pitfalls` 第一条）。
 * 2. **主瓣半高全宽是"量"出来的，不是抄来的**：帧没有给 −3 dB 宽度（`meta.mainlobe_width`
 *    实际写的是 `1/N`，与 `df_true` 同值），所以本模型在**帧自己的谱线**上从峰顶向两侧
 *    走到 −3 dB 处、按线性插值取穿越点。`pad > 1` 时还按 `stride = 1,2,4,…,pad` 抽出
 *    **嵌套的粗栅格**（`fft(zero_pad(x,N·pad))[::stride]` 逐位就是 `fft(zero_pad(x,N·stride))`，
 *    这是补零恒等式，不是近似）逐档量一遍：读数随栅格变、曲线不动——这就是
 *    「原栅格逐位不变、变的只是量它的尺子」的可执行版本。−3 dB 之内还有第二个峰时
 *    读数**判为无意义**（返回 undefined 并写明），不硬报一个数。
 * 3. **混叠是等价类**：`meta.n_aliased > 0` 时用警示色写出 `strings.alias_note`
 *    （后端点名了哪条谐波超 Nyquist）；不为"谱上看到的峰"背书。
 */

import { niceLimits, niceTicks } from "../../shared/src/axes.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import {
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
  crossAt,
  fmt,
  fmtOpt,
  minMaxOfAll,
  sci,
  sliceRange,
  specTitle,
  textBlock,
  textBlockRight,
  unitBudget,
} from "./common.js";

const VIZ = "signal_spectrum";
const TITLE = specTitle(VIZ, "信号 ↔ 频谱联动");

/** 谱线地板（后端 `strings.nonfinite_policy` 写明的 −60 dB）；帧里取到更低值时用帧的。 */
const DB_FLOOR = -60;

export const signalSpectrumModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需数组：缺一个就抛 ModelInputError（宁可当场炸，也不画半张图）——
  const x = requireArray(frame, "x");
  const tAxis = requireArray(frame, "t_axis");
  const specDb = requireArray(frame, "spectrum_db");
  const freq = requireArray(frame, "freq");
  const error = requireArray(frame, "error");
  requireSameLength(tAxis, x, "帧数组 t_axis 与 x");
  requireSameLength(freq, specDb, "帧数组 freq 与 spectrum_db");
  const pitfalls = requirePitfalls(frame);

  // —— 可选字段：缺了就降级（记进 degraded，最终挂成一个 status 角标）——
  const degraded: string[] = [];
  const tMeta = metaNumber(frame, "t");
  if (tMeta === undefined) degraded.push("帧 meta 缺 t：scene.t 改用请求 t（两者可能不同）");
  const n = metaNumber(frame, "n");
  if (n === undefined) degraded.push(`帧 meta 缺 n：点数按 x 数组长度给（${x.length}，形状是数据不是猜的）`);
  const pad = metaNumber(frame, "pad");
  if (pad === undefined) degraded.push("帧 meta 缺 pad：不判断原栅格与补零栅格的差别（不猜）");
  const nPad = metaNumber(frame, "n_pad");
  const df = metaNumber(frame, "df");
  if (df === undefined) degraded.push("帧 meta 缺 df（补零后的栅格）：不并排报 df / df_true 就是误导（见 pitfalls 第一条）");
  const dfTrue = metaNumber(frame, "df_true");
  if (dfTrue === undefined) degraded.push("帧 meta 缺 df_true：不并排报 df / df_true 就是误导");
  const mainlobeWidth = metaNumber(frame, "mainlobe_width");
  const f0 = metaNumber(frame, "f0");
  if (f0 === undefined) degraded.push("帧 meta 缺 f0：不画谐波位置参考线（不按请求参数猜）");
  const nyquist = metaNumber(frame, "nyquist");
  const maxError = metaNumber(frame, "max_error");
  const nAliased = metaNumber(frame, "n_aliased");
  const note = stringOf(frame, "note");
  if (note === undefined) degraded.push("帧 strings 缺 note：状态角标不自行编造结论");
  const algorithm = stringOf(frame, "algorithm");
  const aliasNote = stringOf(frame, "alias_note");
  const nonfinite = stringOf(frame, "nonfinite_policy");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: tMeta ?? opts.t,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  // —— 排版：左列「时域（上）+ 自研对拍（下）」，右列整高给谱；底部留一条关键数字横条 ——
  const strip = bottomStrip(scene, width, height, 4);
  const panels = columnsLayout(width, height, { share: 0.52, gap: 54, footer: strip.footer });
  const timeRect = panels.main;
  const errRect = panels.error;
  const specRect = panels.side;

  // 时域：一个记录周期 T = N·dt（帧 t_axis 的单位就是 T）。x 上界不随 pad 变——
  // 补零添的是 0，落在 FFT 的尾部，这段记录一个样本都没多（见文件头第 1 条）。
  const xRange = minMaxOfAll([x]) ?? [-1, 1];
  const timeAxis = makeAxis("time", timeRect, {
    xlim: [0, 1],
    ylim: niceLimits(xRange[0], xRange[1], 5, { pad: 0.12 }),
    xlabel: "t（单位 T = N·dt；本帧记录就是一个周期）",
    ylabel: axisYlabel(VIZ, "time", "x"),
    title: axisTitle(VIZ, "time", "时域 x(t)"),
    autoscaled: axisAutoscaled(VIZ, "time"),
    xticks: niceTicks(0, 1, 6),
  });

  // 对拍副轴（规则 3）：`error` 是整段 max|FFT−DFT| 铺满 N 点，对数纵轴。
  const errPos = minMaxOfAll([absOf(error)]);
  const errYlim: [number, number] = errPos === undefined ? [1e-18, 1e-12] : niceLimits(errPos[0], errPos[1], 5, { log: true, pad: 0.25 });
  if (errPos === undefined) degraded.push("帧 error 没有 >0 的有限值：对数副轴用占位范围 [1e-18, 1e-12]，不画曲线");
  const errAxis = makeErrorAxis("error", errRect, {
    xlim: [0, 1],
    ylim: errYlim,
    yscale: "log",
    xlabel: "t（与上面同一段记录）",
    ylabel: axisYlabel(VIZ, "error", "max|delta|"),
    title: axisTitle(VIZ, "error", "|FFT − DFT|（自研对拍）"),
    autoscaled: true,
    xticks: niceTicks(0, 1, 6),
  });

  // 谱面板横轴：只画到三倍基频附近（其余到 Nyquist 全在地板上，画出来只会把有用的
  // 那一段压扁）。上界由帧 meta 的 f0 / nyquist 决定，不看请求参数。
  const want = f0 === undefined ? 16 : Math.max(3.5, 3 * f0 * 1.6);
  const specXlim = niceLimits(0, nyquist === undefined ? want : Math.min(want, nyquist), 6);
  const specVisible = sliceRange(freq, specDb, specXlim[0], specXlim[1]);
  // 非有限值不许污染纵轴（`Math.min(...NaN)` = NaN → `niceTicks` 直接抛）；NaN/∞ 的个数进溢出角标。
  const floorDb = finiteMinMax(specVisible.y)?.[0] ?? DB_FLOOR;
  const specAxis = makeAxis("spec", specRect, {
    xlim: specXlim,
    ylim: [Math.min(DB_FLOOR, floorDb), 2],
    xlabel: "f（单位 1/T = 每记录周期数）",
    ylabel: axisYlabel(VIZ, "spec", "dB"),
    title: axisTitle(VIZ, "spec", "单边幅度谱（相对峰值 dB）"),
    autoscaled: axisAutoscaled(VIZ, "spec"),
    xticks: niceTicks(specXlim[0], specXlim[1], 6),
  });
  scene.axes.push(timeAxis, errAxis, specAxis);

  // —— 图层 1~3：时域 ——
  scene.layers.push(lineLayer("time", [series([0, 1], [0, 0], pal.reference, { dash: [4, 4], width: 1 })]));
  const xSeries = segmentedSeries(tAxis, x, pal.curves[0], { width: 1.6});
  if (xSeries.length > 0) scene.layers.push(lineLayer("time", xSeries));
  else degraded.push("帧 x 没有可绘的有限值：时域面板不画曲线");
  if (x.length <= 64) {
    scene.layers.push(scatterLayer("time", tAxis, x, { r: 2, color: pal.marker, alpha: 0.85}));
  }

  // —— 半高全宽：在帧自己的谱线上量（见文件头第 2 条）——
  const halfWidth = measureHalfWidth(specVisible.x, specVisible.y);
  const widths = nestedWidths(freq, specDb, specXlim[1], pad);

  // —— 图层 4~6：谱 ——
  const refs: LineSeries[] = [];
  if (f0 !== undefined) {
    for (const mult of [1, 2, 3]) {
      const f = mult * f0;
      if (f <= 0 || f > specXlim[1]) continue;
      refs.push(series([f, f], [specAxis.ylim[0], specAxis.ylim[1]], pal.reference, { dash: [3, 4], width: 1}));
    }
  }
  if (halfWidth !== undefined) {
    refs.push(
      series([halfWidth.lo, halfWidth.hi], [halfWidth.level, halfWidth.level], pal.marker, {
        width: 2
      }),
    );
    refs.push(series([halfWidth.lo, halfWidth.lo], [specAxis.ylim[0], halfWidth.level], pal.marker, { dash: [2, 3], width: 1 }));
    refs.push(series([halfWidth.hi, halfWidth.hi], [specAxis.ylim[0], halfWidth.level], pal.marker, { dash: [2, 3], width: 1 }));
  }
  refs.push(series([0, specXlim[1]], [floorDb, floorDb], pal.reference, { dash: [6, 4], width: 1}));
  scene.layers.push(lineLayer("spec", refs));

  const specSeries = segmentedSeries(specVisible.x, specVisible.y, pal.curves[0], { width: 1.6});
  if (specSeries.length > 0) scene.layers.push(lineLayer("spec", specSeries));
  else degraded.push("帧 spectrum_db 没有可绘的有限值：谱面板不画曲线（也就不报半高全宽）");

  // 原栅格（pad=1）的读数：stride 抽取**本帧同一批数**，不是另一条曲线。
  if (pad !== undefined && pad > 1) {
    const coarse = decodeStride(specVisible.x, specVisible.y, Math.round(pad));
    if (coarse.x.length >= 2) {
      scene.layers.push(
        scatterLayer("spec", coarse.x, coarse.y, {
          r: 3,
          color: pal.exact,
          alpha: 0.9
        }),
      );
    }
  }

  // —— 图层 7：对拍误差（requiredAux）——
  const errSeries = segmentedSeries(tAxis, absOf(error), pal.residual, { width: 1.4, accept: (v) => v > 0});
  if (errSeries.length > 0) scene.layers.push(lineLayer("error", errSeries));
  else degraded.push("帧 error 没有 >0 的有限值：对数副轴画不出任何点");

  // —— 文本：面板内只放"身份行"（1~2 行），长句子全部进底部横条（见 common.ts::bottomStrip）——
  const summary = paramSummary(frame, opts, [
    { name: "n", label: "N" },
    { name: "pad", label: "pad" },
    { name: "f0", label: "f0" },
  ]);
  degraded.push(...summary.degraded);
  const timeLines: string[] = [summary.text];
  if (nPad !== undefined) {
    timeLines.push(`N_pad = N·pad = ${fmt(nPad, 6)}；记录 T = N·dt = 1（帧 t_axis ∈ [0, ${fmt(tAxis[tAxis.length - 1] ?? 1, 4)}]，与 pad 无关）`);
  }
  scene.texts.push(
    ...textBlock(timeRect[0] + 6, timeRect[1] + 4, timeRect[1] + timeRect[3] - 6, timeLines, pal.muted, unitBudget(timeRect[2]), "top"),
  );

  const errLines: string[] = [
    maxError === undefined
      ? `帧 meta 缺 max_error：只画曲线 ${error.length} 点，不报数值（不重算）`
      : `|FFT − DFT| = ${sci(maxError)}（帧 max_error，铺满 ${error.length} 点）：证据是「这条谱算得对」，与「分辨率」无关`,
  ];
  if (nonfinite !== undefined) errLines.push(`地板口径（帧 strings）：${nonfinite}`);
  scene.texts.push(...textBlock(errRect[0] + 6, errRect[1] + 4, errRect[1] + errRect[3] - 6, errLines, pal.muted, unitBudget(errRect[2]), "top"));

  // 谱面板顶部：**df / df_true 并排**（本页第一条防误导；两个数各自标注是否随 pad 变）。
  const specTop: string[] = [
    `df（DFT 栅格，随 pad 变）= 1/(N·pad) = ${fmtOpt(df, 6)}` + (pad === undefined ? "" : ` = df_true / ${fmt(pad, 3)}`),
    `df_true（不随 pad 变）= 1/N = ${fmtOpt(dfTrue, 6)}`,
  ];
  if ((pad ?? 1) > 1) specTop.push(`⚠ pad = ${fmt(pad as number, 3)}：栅格密了 ${fmt(pad as number, 3)} 倍，df_true 一位没变（读数才变）`);
  else if (mainlobeWidth !== undefined) specTop.push(`帧 meta.mainlobe_width = ${fmt(mainlobeWidth, 6)}（就是 1/N，不是量出来的半高全宽）`);
  scene.texts.push(
    ...textBlockRight(specRect[0] + specRect[2] - 6, specRect[1] + 4, specRect[1] + specRect[3] - 6, specTop, (pad ?? 1) > 1 ? pal.warn : pal.muted, unitBudget(specRect[2]), "top"),
  );
  const specBot: string[] = [
    halfWidth === undefined
      ? "−3 dB 之内还有第二个峰（并包）：半高全宽读数无意义——这本身就是「分辨不出来」"
      : `−3 dB 半高全宽读数 = ${fmt(halfWidth.hi - halfWidth.lo, 4)}/T（峰值 ${fmt(halfWidth.peak, 3)} dB 下 3 dB，占 ${halfWidth.bins} 个栅格点，两侧线性插值）`,
  ];
  scene.texts.push(
    ...textBlock(specRect[0] + 6, specRect[1] + 4, specRect[1] + specRect[3] - 6, specBot, pal.marker, unitBudget(specRect[2]), "bottom"),
  );

  // 底部横条：三条"必须在同屏被挡住"的事实（补零不提高分辨率）。
  const stripLines: string[] = [];
  if ((nAliased ?? 0) > 0.5 && aliasNote !== undefined) {
    stripLines.push(`⚠ 混叠：${aliasNote}——谱上看到的峰是等价类，不能凭峰位反推真实频率`);
  } else {
    stripLines.push("原栅格逐位不变：fft(zero_pad(x, N·pad))[::pad] ≡ fft(x)，差精确 0（tests/2.5 断言）");
    stripLines.push(`补零只是把同一条连续谱采得更密` + (algorithm === undefined ? "" : `；算法 ${algorithm}（帧 strings）`));
  }
  stripLines.push(
    widths.length > 1
      ? `同一帧按嵌套栅格量（−3 dB 半高全宽）：${widths
          .map((row) => `pad=${fmt(row.pad, 3)}→${row.fwhm === undefined ? "并包" : `${fmt(row.fwhm, 4)}/T`}`)
          .join("　")}`
      : "本帧 pad=1，栅格就是原栅格；把 pad 拖大（HUD）才看得到「读数变、曲线不变」",
  );
  stripLines.push(
    "pitfalls §1 实测（N=64、dt=0.01、k=5）：读数随 pad 单调下降但趋不到 0；df_true 全程不变",
  );
  scene.texts.push(...textBlock(timeRect[0], strip.stripTop, strip.limit, stripLines, pal.muted, unitBudget(width - timeRect[0] - 24), "top"));

  // —— 角标（固定槽位）——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(badge("autoscale", "误差副轴：自适应（log 纵轴按本帧误差取整；自适应会掩盖量级变化）", 8, 8), SLOT.auxAxis, width),
  );
  if (n !== undefined) {
    scene.badges.push(placeBadge(criticalBadge("n", n, criticalOf(frame, "n", 64), { beyond: "below" }), SLOT.critical, width));
  } else {
    degraded.push("帧 meta 缺 n：不挂 n 的临界角标（不猜临界值）");
  }
  scene.badges.push(
    placeBadge(
      badge(
        "critical",
        clipBadgeText(`pad = ${fmtOpt(pad, 3)}　临界 1（补零只加密栅格 df→df/pad，df_true 不动）` + ((pad ?? 1) > 1 ? "　⚠ 已越过" : "")),
        8,
        8,
      ),
      SLOT.critical2,
      width,
    ),
  );
  const status: string[] = [];
  if (nPad !== undefined) status.push(`N_pad=${fmt(nPad, 6)}`);
  if (maxError !== undefined) status.push(`max|FFT−DFT|=${sci(maxError)}`);
  if ((nAliased ?? 0) > 0.5) status.push(`⚠ ${fmt(nAliased as number, 3)} 条谐波超 Nyquist（混叠）`);
  if (note !== undefined) status.push(note);
  scene.badges.push(placeBadge(badge("status", clipBadgeText(status.join("；")), 8, 8), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  // 规则 1：图层越界 + 帧内非有限值，两者都要进 overflow 与角标。
  const nBad = countNonFinite(x) + countNonFinite(tAxis) + countNonFinite(specDb) + countNonFinite(freq) + countNonFinite(error);
  applyOverflow(scene, nBad, { slot: SLOT.overflow, width });

  // 规则 8：文本层之间、以及与底部 pitfalls 面板之间都不许重叠。
  // 第四个参数是**高度**（`textBoxOf` 口径）不是底边：底线取面板顶边往上 4px。
  scene.texts = resolveTextOverlaps(scene.texts, [8, 24, width - 16, strip.limit - 24]);
  return scene;
};

// --------------------------------------------------------------------------- //
// 局部工具
// --------------------------------------------------------------------------- //

interface Slice {
  x: Float64Array;
  y: Float64Array;
}

/** `stride` 抽取（嵌套栅格：从原始索引出发，`stride = 1,2,4,…`）。 */
function decodeStride(xs: ArrayLike<number>, ys: ArrayLike<number>, stride: number): Slice {
  const xsOut: number[] = [];
  const ysOut: number[] = [];
  for (let i = 0; i < Math.min(xs.length, ys.length); i += stride) {
    xsOut.push(xs[i] as number);
    ysOut.push(ys[i] as number);
  }
  return { x: Float64Array.from(xsOut), y: Float64Array.from(ysOut) };
}

interface HalfWidth {
  lo: number;
  hi: number;
  level: number;
  peak: number;
  bins: number;
}

/**
 * 在**帧自己的谱线**上量 −3 dB 半高全宽（`stride` 用来在嵌套栅格上重量）。
 *
 * 从全局峰向两侧走：只允许"还在下降"的方向（`db[i±1] ≤ db[i]`）。如果先撞上回升，
 * 说明 −3 dB 之内还有第二个峰，返回 `undefined`——那不是"宽"，是"并包"。
 * 穿越点用线性插值（在 dB 上，与渲染器画出来的折线一致）。
 */
function measureHalfWidth(
  freq: ArrayLike<number>,
  db: ArrayLike<number>,
  opts: { offset?: number; stride?: number; dbLo?: number; dbHi?: number } = {},
): HalfWidth | undefined {
  const offset = opts.offset ?? 0;
  const stride = opts.stride ?? 1;
  const xs: number[] = [];
  const ys: number[] = [];
  for (let i = offset; i < Math.min(freq.length, db.length); i += stride) {
    const xv = freq[i] as number;
    if (opts.dbLo !== undefined && xv < opts.dbLo) continue;
    if (opts.dbHi !== undefined && xv > opts.dbHi) continue;
    xs.push(xv);
    ys.push(db[i] as number);
  }
  const n = xs.length;
  if (n < 3) return undefined;
  let peakAt = 0;
  for (let i = 1; i < n; i += 1) if ((ys[i] as number) > (ys[peakAt] as number)) peakAt = i;
  const peak = ys[peakAt] as number;
  const level = peak - 3;
  let count = 1;
  let lo: number | undefined;
  let hi: number | undefined;
  for (const dir of [-1, 1] as const) {
    // 先沿"仍在 −3 dB 以上"走到该方向的最后一个样本，中途回升即判并包。
    let i = peakAt;
    while (i + dir >= 0 && i + dir < n && (ys[i + dir] as number) > level) {
      if ((ys[i + dir] as number) > (ys[i] as number)) return undefined; // 回升 = 第二个峰在 −3 dB 之内
      i += dir;
      count += 1;
    }
    const cross = crossAt(xs, ys, peakAt, dir, level);
    if (cross === undefined) return undefined; // 走到头都没穿过去：读数无意义
    if (dir < 0) lo = cross;
    else hi = cross;
  }
  if (lo === undefined || hi === undefined || !(hi > lo)) return undefined;
  return { lo, hi, level, peak, bins: count };
}

interface WidthRow {
  pad: number;
  fwhm: number | undefined;
  bins: number;
}

/**
 * 嵌套栅格上逐档量半高全宽：`stride = 1, 2, 4, …, pad`。
 * `pad = 1` 时只有一行（本帧栅格就是原栅格）。
 */
function nestedWidths(freq: ArrayLike<number>, db: ArrayLike<number>, xHi: number, pad: number | undefined): WidthRow[] {
  const rows: WidthRow[] = [];
  const hi = pad === undefined || pad < 1 ? 1 : Math.round(pad);
  for (let stride = 1; stride <= hi; stride *= 2) {
    const w = measureHalfWidth(freq, db, { stride, dbLo: 0, dbHi: xHi });
    rows.push({ pad: stride, fwhm: w === undefined ? undefined : w.hi - w.lo, bins: w === undefined ? 0 : w.bins });
  }
  return rows;
}

/** 供测试引用：这份 model 依赖的帧数组名（缺一个就该抛 ModelInputError）。 */
export const SIGNAL_SPECTRUM_ARRAYS = ["x", "t_axis", "spectrum_db", "freq", "error"] as const;
