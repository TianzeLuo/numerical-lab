/**
 * `time_scale_map`（2.6 章）的 model：SWT 尺度图 vs STFT 频谱图的并排对照。
 *
 * 核心阅读纪律：
 * - 两张热力图的**横轴都是样本下标**，纵轴分别是"尺度层"与"归一化频率"；
 *   把尺度层直接读成频率会低估低频的时间定位能力（第 j 层带宽 1/2^j、
 *   时间支撑 2^j，乘积近似守恒）。右下角的分辨率元面板把这件事画成数字。
 * - 热力图色标不是跨帧自适应的：vmin/vmax 由帧数据定，并在右上角写明。
 *   自适应色标会把"能量变小"化装成"分布没变"。
 * - SWT 是**不抽取**的：每层与信号等长。平移一格时 SWT 系数逐位平移，
 *   而 DWT 会变——帧的 `strings.shift_note` 直接写了两者实测值。
 *
 * `t` 对这个 viz 无意义（manifest: mode=none），scene.t 恒为 0。
 */

import { formatNumber } from "../../shared/src/axes.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import {
  applyOverflow,
  canvasSize,
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
  stringOf,
  type SceneModel,
} from "../../shared/src/model.js";
import {
  badge,
  criticalBadge,
  heatLayer,
  lineLayer,
  makeAxis,
  series,
  sourceBadge,
} from "../../shared/src/scene.js";
import {
  axisTitleOf,
  axisYlabelOf,
  columnsLayout,
  finiteMinMax,
  firstFinite,
  legendBlock,
  logRange,
  positiveMinMax,
  textBlock,
  unitBudget,
} from "../_common/panel.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "2.6";
const VIZ = "time_scale_map";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "时间-尺度图 vs STFT";
const LEVEL_CRITICAL = SPEC?.params?.find((p) => p.name === "level")?.critical ?? [5];
const FLOOR = 1e-18;

const SLOT = {
  source: 0,
  axisMode: 1,
  aux: 2,
  critical: 3,
  status: 4,
  preview: 5,
  overflow: 6,
  degraded: 7,
} as const;

export const timeScaleMapModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  const signal = requireArray(frame, "signal");
  const scalogram = requireArray(frame, "scalogram");
  const stftMap = requireArray(frame, "stft_map");
  const error = requireArray(frame, "error");
  const pitfalls = requirePitfalls(frame);

  const level = Math.max(1, Math.round(metaNumber(frame, "level") ?? 4));
  const n = Math.max(2, Math.round(metaNumber(frame, "n") ?? signal.length));
  const nBands = Math.max(1, Math.round(metaNumber(frame, "n_bands") ?? level + 1));
  const nFreq = Math.max(2, Math.round(metaNumber(frame, "n_stft_freq") ?? 0));
  const nTime = Math.max(2, Math.round(metaNumber(frame, "n_stft_time") ?? 0));
  if (scalogram.length !== nBands * n) {
    throw new Error(
      `帧 ${CHAPTER}/${VIZ} 的 scalogram 长度 ${scalogram.length} ≠ n_bands×n=${nBands * n}（拒绝画错位的热力图）`,
    );
  }
  if (stftMap.length !== nFreq * nTime) {
    throw new Error(
      `帧 ${CHAPTER}/${VIZ} 的 stft_map 长度 ${stftMap.length} ≠ n_freq×n_time=${nFreq * nTime}`,
    );
  }

  // 可选数组：STFT 的坐标轴数组与分辨率元。
  const degraded: string[] = [];
  const stftTime = frame.has("stft_time") ? requireArray(frame, "stft_time") : undefined;
  const stftFreq = frame.has("stft_freq") ? requireArray(frame, "stft_freq") : undefined;
  if (stftTime === undefined || stftTime.length !== nTime || stftFreq === undefined || stftFreq.length !== nFreq) {
    const missing = [
      stftTime === undefined || stftTime.length !== nTime ? "stft_time" : "",
      stftFreq === undefined || stftFreq.length !== nFreq ? "stft_freq" : "",
    ].filter((s) => s !== "");
    // 合并成一条短文案：degradedBadge 会裁到列宽（47 单位），两条长句会把第二条整个裁掉。
    degraded.push(`缺 ${missing.join("/")}：STFT 轴用下标（不是物理时刻/频率）`);
  }
  const wavWidth = frame.has("wav_width") ? requireArray(frame, "wav_width") : undefined;
  const wavBand = frame.has("wav_band") ? requireArray(frame, "wav_band") : undefined;
  const stftWidth = frame.has("stft_width") ? requireArray(frame, "stft_width") : undefined;
  const stftBand = frame.has("stft_band") ? requireArray(frame, "stft_band") : undefined;
  const cellArea = frame.has("cell_area") ? requireArray(frame, "cell_area") : undefined;
  if (wavWidth === undefined || wavBand === undefined) {
    degraded.push("帧缺 wav_width/wav_band：不画右下角的分辨率元柱（不按 2^j 自己推算）");
  }

  // —— meta ——
  const prError = metaNumber(frame, "pr_error");
  const energyRatio = metaNumber(frame, "swt_energy_ratio");
  const symbolMean = metaNumber(frame, "swt_symbol_mean");
  const redundancy = metaNumber(frame, "swt_redundancy");
  const stftWindow = metaNumber(frame, "stft_window");
  const stftHop = metaNumber(frame, "stft_hop");
  const cellMin = metaNumber(frame, "cell_area_min");
  const cellMax = metaNumber(frame, "cell_area_max");
  const signalName = stringOf(frame, "signal");
  const wavelet = stringOf(frame, "wavelet");
  const note = stringOf(frame, "note");
  const shiftNote = stringOf(frame, "shift_note");
  const nonfinitePolicy = stringOf(frame, "nonfinite_policy");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: 0,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  const L = columnsLayout(scene, width, height, { topShare: 0.6, auxH: 116, gap: 44 });

  // —— SWT 尺度图（主面板）——
  const scale = makeAxis("scale", [L.left, L.topY, L.leftW, L.topH], {
    xlim: [0, Math.max(1, n - 1)],
    ylim: [0, Math.max(1, nBands - 1)],
    title: axisTitleOf(SPEC?.axes, "scale", "SWT 系数（尺度 × 时间）"),
    ylabel: axisYlabelOf(SPEC?.axes, "scale", "尺度层 0=a_J, 1=d_J … J=d_1"),
    xlabel: "样本下标",
    autoscaled: false,
  });
  const swtRange = finiteMinMax(scalogram);
  const swtLo = swtRange?.[0] ?? -1;
  const swtHi = swtRange?.[1] ?? 1;
  scene.layers.push(
    heatLayer("scale", scalogram, n, nBands, [0, Math.max(1, n - 1), 0, Math.max(1, nBands - 1)], {
      cmap: "rdbu",
      vmin: Math.min(swtLo, -1e-12),
      vmax: Math.max(swtHi, 1e-12),
      label: "SWT 系数",
    }),
  );

  // —— STFT 幅度（下面板）——
  const t0 = firstFinite(stftTime) ?? 0;
  const t1 = stftTime !== undefined && stftTime.length > 0 ? (stftTime[stftTime.length - 1] as number) : Math.max(1, nTime - 1);
  const f0 = firstFinite(stftFreq) ?? 0;
  const f1 = stftFreq !== undefined && stftFreq.length > 0 ? (stftFreq[stftFreq.length - 1] as number) : Math.max(1, nFreq - 1);
  const stftAxis = makeAxis("stft", [L.left, L.botY, L.leftW, L.botH], {
    xlim: [t0, Math.max(t0 + 1, t1)],
    ylim: [f0, Math.max(f0 + 1e-9, f1)],
    title: axisTitleOf(SPEC?.axes, "stft", "STFT 幅度（频率 × 时间，固定窗）"),
    ylabel: axisYlabelOf(SPEC?.axes, "stft", "归一化频率（周期/样本）"),
    xlabel: "窗中心下标",
    autoscaled: false,
  });
  const stftPos = positiveMinMax(stftMap);
  const stftMin = stftPos?.[0] ?? 1e-6;
  const stftMax = stftPos?.[1] ?? 1;
  scene.layers.push(
    heatLayer(
      "stft",
      stftMap,
      nTime,
      nFreq,
      [t0, Math.max(t0 + 1, t1), f0, Math.max(f0 + 1e-9, f1)],
      {
        cmap: "magma",
        vmin: Math.max(stftMin, stftMax * 1e-5),
        vmax: stftMax,
        logScale: true,
        clipBelow: Math.max(stftMin, stftMax * 1e-5),
        label: "STFT 幅度",
      },
    ),
  );

  // —— 误差副轴（规则 3）——
  const errLo = positiveMinMax(error)?.[0] ?? FLOOR;
  const errHi = positiveMinMax(error)?.[1] ?? 1;
  const errAxis = makeAxis("error", [L.rightX, L.topY, L.rightW, L.auxH], {
    xlim: [0, Math.max(1, n - 1)],
    ylim: logRange(errLo * 0.5, Math.max(errHi, FLOOR) * 2),
    yscale: "log",
    title: axisTitleOf(SPEC?.axes, "error", "SWT 往返误差"),
    ylabel: axisYlabelOf(SPEC?.axes, "error", "max|delta|"),
    xlabel: "样本下标",
    autoscaled: true,
  });
  scene.axes.push(scale, stftAxis, errAxis);
  const xIdx = new Float64Array(n);
  for (let i = 0; i < n; i += 1) xIdx[i] = i;
  scene.layers.push(lineLayer("error", series(xIdx, error, pal.residual, { width: 2 })));
  if (prError !== undefined && prError > 0) {
    const y = Math.max(prError, FLOOR);
    scene.layers.push(
      lineLayer("error", series([errAxis.xlim[0], errAxis.xlim[1]], [y, y], pal.warn, { width: 1, dash: [4, 3] })),
    );
  }

  // —— 面板图例 ——
  scene.texts.push(
    ...legendBlock(
      L.left + 6,
      L.topY + 6,
      L.leftW,
      `SWT：${nBands} 层（0=a_J, 1=d_J … ${nBands - 1}=d_1），色标 ${formatNumber(swtLo, 3)} … ${formatNumber(swtHi, 3)}`,
      pal.muted,
    ),
    ...legendBlock(
      L.left + 6,
      L.botY + 6,
      L.leftW,
      `STFT：固定窗 ${stftWindow !== undefined ? formatNumber(stftWindow, 3) : "?"}，跳步 ${stftHop !== undefined ? formatNumber(stftHop, 3) : "?"}；对数色标（<${formatNumber(Math.max(stftMin, stftMax * 1e-5), 2)} 已裁掉）`,
      pal.muted,
    ),
  );

  // —— 右列文字 ——
  if (L.columns) {
    const head = paramSummary(frame, opts, [
      { name: "n", label: "$n$" },
      { name: "level", label: "$J$" },
    ]);
    degraded.push(...head.degraded);
    const lines: string[] = [];
    lines.push(`${head.text}　${signalName ?? "?"} / ${wavelet ?? "?"}`);
    if (redundancy !== undefined) lines.push(`SWT 冗余：每层全长，总系数 ${formatNumber((level + 1) * n, 4)} 个（信号 ${n}）`);
    if (prError !== undefined) lines.push(`SWT 往返误差 ${formatNumber(prError, 3)}（相对，贴 u 量级）`);
    if (energyRatio !== undefined && symbolMean !== undefined) {
      lines.push(`能量比实测 ${formatNumber(energyRatio, 4)} vs 框架符号平均 ${formatNumber(symbolMean, 4)}`);
      lines.push("能量比不是常数：拿 |系数|² 之和当能量会随信号频率漂移");
    }
    if (cellMin !== undefined && cellMax !== undefined) {
      lines.push(`分辨率元面积 ${formatNumber(cellMin, 4)} … ${formatNumber(cellMax, 4)}（近似守恒）`);
    }
    if (wavWidth !== undefined && wavBand !== undefined) {
      const nCells = Math.min(wavWidth.length, wavBand.length, 4);
      const widths = Array.from({ length: nCells }, (_, i) => formatNumber(wavWidth[i] as number, 3)).join("/");
      const bands = Array.from({ length: nCells }, (_, i) => formatNumber(wavBand[i] as number, 3)).join("/");
      lines.push(`小波宽度 ${widths}；带宽 ${bands}`);
    }
    if (stftWidth !== undefined && stftBand !== undefined) {
      lines.push(
        `STFT 恒为 宽 ${formatNumber(firstFinite(stftWidth) ?? 0, 3)} / 带 ${formatNumber(firstFinite(stftBand) ?? 0, 3)}`,
      );
    }
    if (note !== undefined) lines.push(note);
    if (shiftNote !== undefined) lines.push(shiftNote);
    if (nonfinitePolicy !== undefined) lines.push(nonfinitePolicy);
    scene.texts.push(...textBlock(L.rightX, L.textTop, L.floor, lines, pal.muted, unitBudget(L.rightW), "top"));
  } else {
    degraded.push("窄画布（单列）：右列说明未排下，关键数字见角标与 pitfalls 面板");
  }

  // —— 角标 ——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  scene.badges.push(
    placeBadge(badge("autoscale", "坐标轴：两张热力图固定窗口；误差轴自适应", 8, 26), SLOT.axisMode, width),
  );
  scene.badges.push(placeBadge(criticalBadge("J", level, LEVEL_CRITICAL), SLOT.critical, width));
  const status = [
    `J=${level}`,
    `SWT 往返 ${prError !== undefined ? formatNumber(prError, 3) : "?"}`,
    energyRatio !== undefined ? `能量比 ${formatNumber(energyRatio, 3)}` : "",
  ].filter((s) => s !== "");
  scene.badges.push(placeBadge(badge("status", status.join("；"), 8, 8), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  const nBad =
    countNonFinite(signal) +
    countNonFinite(scalogram) +
    countNonFinite(stftMap) +
    countNonFinite(error) +
    (cellArea !== undefined ? countNonFinite(cellArea) : 0);
  applyOverflow(scene, nBad, { slot: SLOT.overflow, width });
  return scene;
};
