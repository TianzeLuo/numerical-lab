/**
 * `denoise_threshold`（2.6 章）的 model：去噪结果 + 偏差²/方差/MSE 三条曲线。
 *
 * 这张图要挡住两个误读：
 * 1. **"看起来更平滑"不等于更好**：曲线面板上把 MSE、偏差²、方差画在同一个窗口里，
 *    最优点在中间；拉到通用阈值之后 MSE 可以超过"完全不去噪"（帧给的水平线）。
 * 2. **MSE 有两个口径**：`meta.mse` 是这一帧实际去噪结果（单次实现）的 MSE，
 *    `meta.mse_curve_here` 是蒙特卡洛曲线在最近栅格点上的值。两者都显示、不混用。
 *
 * 三条曲线在帧里已经取过 `1e-18` 地板（对数轴的显示约定），所以这里把 `curves`
 * 画在线性轴上、把 `error` 画在对数轴上；两者都不再自行裁剪。
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
  lineLayer,
  makeAxis,
  scatterLayer,
  series,
  sourceBadge,
} from "../../shared/src/scene.js";
import {
  axisTitleOf,
  axisYlabelOf,
  columnsLayout,
  finiteMinMax,
  firstFinite,
  indexAxis,
  legendBlock,
  linearRange,
  logRange,
  positiveMinMax,
  textBlock,
  unitBudget,
} from "../_common/panel.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "2.6";
const VIZ = "denoise_threshold";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "阈值去噪：偏差-方差";
const LAM_CRITICAL = SPEC?.params?.find((p) => p.name === "lam")?.critical ?? [0.28, 1.0];
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

export const denoiseThresholdModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  const clean = requireArray(frame, "clean");
  const noisy = requireArray(frame, "noisy");
  const denoised = requireArray(frame, "denoised");
  const error = requireArray(frame, "error");
  const lambdaGrid = requireArray(frame, "lambda_grid");
  const mseCurve = requireArray(frame, "mse_curve");
  const bias2Curve = requireArray(frame, "bias2_curve");
  const varCurve = requireArray(frame, "var_curve");
  const pitfalls = requirePitfalls(frame);
  for (const [name, arr] of [
    ["mse_curve", mseCurve],
    ["bias2_curve", bias2Curve],
    ["var_curve", varCurve],
  ] as const) {
    if (arr.length !== lambdaGrid.length) {
      throw new Error(`帧 ${CHAPTER}/${VIZ} 的 ${name} 长度 ${arr.length} ≠ lambda_grid 长度 ${lambdaGrid.length}`);
    }
  }
  const n = Math.min(clean.length, noisy.length, denoised.length, error.length);

  // —— 可选参考线：缺了只是少两条线，明确降级 ——
  const degraded: string[] = [];
  const mseNoisyLine = frame.has("mse_noisy_line") ? requireArray(frame, "mse_noisy_line") : undefined;
  const universalLine = frame.has("universal_line") ? requireArray(frame, "universal_line") : undefined;
  if (mseNoisyLine === undefined) degraded.push("帧缺 mse_noisy_line：不画「完全不去噪」的水平线");
  if (universalLine === undefined) degraded.push("帧缺 universal_line：不画通用阈值的竖直参考线");

  // —— meta ——
  const lam = metaNumber(frame, "lam");
  const lamOpt = metaNumber(frame, "lam_opt");
  const lamGrid = metaNumber(frame, "lam_grid");
  const universal = metaNumber(frame, "universal");
  const universalOverOpt = metaNumber(frame, "universal_over_opt");
  const mse = metaNumber(frame, "mse");
  const mseCurveHere = metaNumber(frame, "mse_curve_here");
  const mseNoisy = metaNumber(frame, "mse_noisy");
  const mseOpt = metaNumber(frame, "mse_opt");
  const mseAtUniversal = metaNumber(frame, "mse_at_universal");
  const sigma = metaNumber(frame, "sigma");
  const sigmaHat = metaNumber(frame, "sigma_hat");
  const sigmaHatBias = metaNumber(frame, "sigma_hat_bias");
  const snrIn = metaNumber(frame, "snr_in");
  const snrOut = metaNumber(frame, "snr_out");
  const relError = metaNumber(frame, "rel_error");
  const nTrials = metaNumber(frame, "n_trials");
  const signalName = stringOf(frame, "signal");
  const wavelet = stringOf(frame, "wavelet");
  const rule = stringOf(frame, "rule");
  const ruleNote = stringOf(frame, "rule_note");
  const sigmaNote = stringOf(frame, "sigma_note");
  const nonfinitePolicy = stringOf(frame, "nonfinite_policy");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: 0,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  const L = columnsLayout(scene, width, height, { topShare: 0.56, auxH: 120, gap: 46 });

  // —— 信号面板：干净 / 含噪 / 去噪 ——
  const sigRange = finiteMinMax(clean);
  let lo = sigRange?.[0] ?? -1;
  let hi = sigRange?.[1] ?? 1;
  for (const arr of [noisy, denoised]) {
    const r = finiteMinMax(arr);
    if (r !== undefined) {
      lo = Math.min(lo, r[0]);
      hi = Math.max(hi, r[1]);
    }
  }
  const sigAxis = makeAxis("signals", [L.left, L.topY, L.leftW, L.topH], {
    xlim: [0, Math.max(1, n - 1)],
    ylim: linearRange(lo, hi, 0.08),
    title: axisTitleOf(SPEC?.axes, "signals", "干净 / 含噪 / 去噪"),
    ylabel: axisYlabelOf(SPEC?.axes, "signals", "x"),
    xlabel: "样本下标",
    autoscaled: false,
  });

  // —— 曲线面板：MSE / 偏差² / 方差 vs λ ——
  const curveHi = Math.max(
    finiteMinMax(mseCurve)?.[1] ?? 1,
    finiteMinMax(bias2Curve)?.[1] ?? 1,
    finiteMinMax(varCurve)?.[1] ?? 1,
    finiteMinMax(mseNoisyLine)?.[1] ?? 0,
    mseNoisy ?? 0,
  );
  const curveAxis = makeAxis("curves", [L.left, L.botY, L.leftW, L.botH], {
    xlim: [firstFinite(lambdaGrid) ?? 0, lastFinite(lambdaGrid) ?? 1],
    ylim: [0, curveHi * 1.15],
    title: axisTitleOf(SPEC?.axes, "curves", "偏差²、方差、MSE vs lambda"),
    ylabel: axisYlabelOf(SPEC?.axes, "curves", "MSE"),
    xlabel: "阈值 lambda",
    autoscaled: true,
  });

  // —— 误差副轴（规则 3）——
  const errLo = positiveMinMax(error)?.[0] ?? FLOOR;
  const errHi = positiveMinMax(error)?.[1] ?? 1;
  const errAxis = makeAxis("error", [L.rightX, L.topY, L.rightW, L.auxH], {
    xlim: [0, Math.max(1, n - 1)],
    ylim: logRange(errLo * 0.5, Math.max(errHi, FLOOR) * 2),
    yscale: "log",
    title: axisTitleOf(SPEC?.axes, "error", "去噪残差 |x_hat - x|"),
    ylabel: axisYlabelOf(SPEC?.axes, "error", "max|delta|"),
    xlabel: "样本下标",
    autoscaled: true,
  });
  scene.axes.push(sigAxis, curveAxis, errAxis);

  const xIdx = indexAxis(n);
  scene.layers.push(
    lineLayer("signals", series(xIdx, clean, pal.reference, { width: 1.6, label: "干净信号" })),
    lineLayer("signals", series(xIdx, noisy, pal.muted, { width: 0.9, alpha: 0.7, label: "含噪信号" })),
    lineLayer("signals", series(xIdx, denoised, pal.curves[0], { width: 2.2, label: "去噪结果" })),
  );

  // 竖直参考线：当前 λ（实线）、实测最优（点线）、通用阈值（点划线）。
  const lamHere = lam ?? firstFinite(lambdaGrid) ?? 0;
  const vLines: { x: number; color: string; dash: number[]; label: string }[] = [
    { x: lamHere, color: pal.marker, dash: [], label: `当前 λ=${formatNumber(lamHere, 3)}` },
  ];
  if (lamOpt !== undefined) vLines.push({ x: lamOpt, color: pal.curves[1], dash: [3, 3], label: `实测最优 λ=${formatNumber(lamOpt, 3)}` });
  if (universal !== undefined) {
    vLines.push({ x: universal, color: pal.warn, dash: [2, 3], label: `通用阈值 λ=${formatNumber(universal, 3)}` });
  }
  for (const v of vLines) {
    scene.layers.push(
      lineLayer(
        "curves",
        series([v.x, v.x], [curveAxis.ylim[0], curveAxis.ylim[1]], v.color, { width: 1.2, dash: v.dash }),
      ),
    );
  }
  // 水平参考线：完全不去噪的 MSE。
  if (mseNoisyLine !== undefined) {
    scene.layers.push(
      lineLayer("curves", series(lambdaGrid, mseNoisyLine, pal.reference, { width: 1.1, dash: [5, 4], label: "不去噪 MSE" })),
    );
  }
  scene.layers.push(
    lineLayer("curves", series(lambdaGrid, bias2Curve, pal.exact, { width: 1.5, label: "偏差²" })),
    lineLayer("curves", series(lambdaGrid, varCurve, pal.curves[1], { width: 1.5, label: "方差" })),
    lineLayer("curves", series(lambdaGrid, mseCurve, pal.curves[0], { width: 2.3, label: "MSE" })),
  );
  // 当前 λ 在三条曲线上的读数（帧 meta 已给，不自己插值算）。
  const hereIdx = nearestIndex(lambdaGrid, lamHere);
  if (hereIdx >= 0) {
    const lamX = lambdaGrid[hereIdx] as number;
    const pts = [
      { y: bias2Curve[hereIdx] as number, color: pal.exact, r: 2.6 },
      { y: varCurve[hereIdx] as number, color: pal.curves[1], r: 2.6 },
      { y: mseCurve[hereIdx] as number, color: pal.curves[0], r: 3.2 },
    ];
    for (const p of pts) {
      if (Number.isFinite(p.y)) scene.layers.push(scatterLayer("curves", [lamX], [p.y], { r: p.r, color: p.color }));
    }
  }

  // 误差副轴。
  scene.layers.push(lineLayer("error", series(xIdx, error, pal.residual, { width: 2 })));
  if (relError !== undefined && relError > 0) {
    const y = Math.max(relError, FLOOR);
    scene.layers.push(
      lineLayer("error", series([errAxis.xlim[0], errAxis.xlim[1]], [y, y], pal.warn, { width: 1, dash: [4, 3] })),
    );
  }

  // —— 图例 ——
  scene.texts.push(
    ...legendBlock(L.left + 6, L.topY + 6, L.leftW, "灰实=干净　灰细=含噪　蓝=去噪结果（同一噪声实现，拖 λ 只换阈值）", pal.muted),
    ...legendBlock(
      L.left + 6,
      L.botY + 6,
      L.leftW,
      `橙=偏差²　绿=方差　蓝=MSE　灰虚=不去噪 MSE；竖线：红=当前 λ，绿点=最优，红点=通用阈值`,
      pal.muted,
    ),
  );

  // —— 右列文字 ——
  if (L.columns) {
    const head = paramSummary(frame, opts, [
      { name: "sigma", label: "σ" },
      { name: "lam", label: "λ" },
    ]);
    degraded.push(...head.degraded);
    const lines: string[] = [];
    lines.push(`${head.text}　${signalName ?? "?"} / ${wavelet ?? "?"} / ${rule ?? "?"}`);
    if (mse !== undefined) lines.push(`本帧去噪 MSE ${formatNumber(mse, 4)}（单次实现）`);
    if (mseCurveHere !== undefined && lamGrid !== undefined) {
      lines.push(`蒙特卡洛曲线 @λ=${formatNumber(lamGrid, 3)}：${formatNumber(mseCurveHere, 4)}`);
    }
    if (mseOpt !== undefined && lamOpt !== undefined) lines.push(`曲线最优 ${formatNumber(mseOpt, 4)} @ λ=${formatNumber(lamOpt, 3)}`);
    if (mseNoisy !== undefined) lines.push(`完全不去噪 MSE ${formatNumber(mseNoisy, 4)}`);
    if (universal !== undefined) {
      lines.push(
        `通用阈值 ${formatNumber(universal, 3)}` +
          (universalOverOpt !== undefined ? ` = 最优的 ${formatNumber(universalOverOpt, 3)} 倍` : ""),
      );
    }
    if (mseAtUniversal !== undefined) {
      lines.push(
        `通用阈值处 MSE ${formatNumber(mseAtUniversal, 4)}` +
          (mseNoisy !== undefined && mseAtUniversal > mseNoisy ? "（比不去噪还差）" : ""),
      );
    }
    if (sigma !== undefined && sigmaHat !== undefined) {
      lines.push(
        `MAD σ_hat=${formatNumber(sigmaHat, 4)}` +
          (sigmaHatBias !== undefined ? `（偏差 ${formatNumber(100 * sigmaHatBias, 2)}%）` : ""),
      );
    }
    if (snrIn !== undefined && snrOut !== undefined) lines.push(`SNR ${formatNumber(snrIn, 2)} → ${formatNumber(snrOut, 2)} dB`);
    if (relError !== undefined) lines.push(`去噪残差相对量 ${formatNumber(relError, 3)}`);
    if (nTrials !== undefined) lines.push(`曲线由 ${Math.round(nTrials)} 次蒙特卡洛试验估计（种子固定）`);
    if (ruleNote !== undefined) lines.push(ruleNote);
    if (sigmaNote !== undefined) lines.push(sigmaNote);
    if (nonfinitePolicy !== undefined) lines.push(nonfinitePolicy);
    scene.texts.push(...textBlock(L.rightX, L.textTop, L.floor, lines, pal.muted, unitBudget(L.rightW), "top"));
  } else {
    degraded.push("窄画布（单列）：右列说明未排下，关键数字见角标与 pitfalls 面板");
  }

  // —— 角标 ——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  scene.badges.push(
    placeBadge(badge("autoscale", "坐标轴：信号固定窗口；曲线/误差轴自适应", 8, 26), SLOT.axisMode, width),
  );
  if (lam !== undefined) scene.badges.push(placeBadge(criticalBadge("λ", lam, LAM_CRITICAL), SLOT.critical, width));
  const status = [
    rule ?? "?",
    `λ=${formatNumber(lamHere, 3)}`,
    `MSE ${mse !== undefined ? formatNumber(mse, 3) : "?"}`,
    mseOpt !== undefined ? `最优 ${formatNumber(mseOpt, 3)}` : "",
  ].filter((s) => s !== "");
  scene.badges.push(placeBadge(badge("status", status.join("；"), 8, 8), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  const nBad = [clean, noisy, denoised, error, mseCurve, bias2Curve, varCurve].reduce(
    (acc, arr) => acc + countNonFinite(arr),
    0,
  );
  applyOverflow(scene, nBad, { slot: SLOT.overflow, width });
  return scene;
};

function lastFinite(a: ArrayLike<number>): number | undefined {
  for (let i = a.length - 1; i >= 0; i -= 1) if (Number.isFinite(a[i])) return a[i];
  return undefined;
}

/** 栅格上离 `x` 最近的点的下标（显示用；不在前端插值物理量）。 */
function nearestIndex(grid: ArrayLike<number>, x: number): number {
  let best = -1;
  let bestD = Number.POSITIVE_INFINITY;
  for (let i = 0; i < grid.length; i += 1) {
    const d = Math.abs((grid[i] as number) - x);
    if (Number.isFinite(d) && d < bestD) {
      best = i;
      bestD = d;
    }
  }
  return best;
}
