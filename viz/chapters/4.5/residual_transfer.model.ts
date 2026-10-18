/**
 * `residual_transfer`（4.5 章）的 model：同一残差在细/粗网格上的形态 + 逐频率的可见性。
 *
 * 面板与图层（顺序即绘制顺序）
 * ----------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | line | `profiles` | `residual_fine`（细网格残差 r） |
 * | 2 | line | `profiles` | `residual_coarse`（限制到粗网格的 R r，画在粗层自己的 `coarse_x` 上） |
 * | 3 | line | `profiles` | `residual_prolonged`（粗层可见部分 P R r，搬回细网格比） |
 * | 4 | line | `profiles` | `residual_lost`（粗层看不见的部分 r − P R r） |
 * | 5 | line | `visibility` | `loss_vs_theta`（纯模态的不可见比例） |
 * | 6 | line | `visibility` | `coarse_visible_ratio`（粗层能看到多少） |
 * | 7 | line | `visibility` | `smoothing_vs_theta` × `coarse_correction_vs_theta`（两条细虚线） |
 * | 8 | line | `visibility` | `product_vs_theta`（互补性：两条细线的乘积） |
 * | 9 | line | `visibility` | 竖直临界线 θ/π=0.5（规则 6：粗网格能忠实表示的频率上界） |
 * | 10 | scatter | `visibility` | 当前模态（`meta.mode_index`）在两条曲线上的位置 |
 * | 11 | line | `loss` | `loss_history`（混合残差的不可见比例随 ν 的走向） |
 * | 12 | line | `loss` | `residual_norm_history`（残差自身在变小） |
 * | 13 | line | `loss` | 纯模态不可见比例 `meta.loss_at_current_theta` 的水平参考线 |
 *
 * 四条刻意的决定
 * --------------
 * 1. **`residual_lost` 必须与 `residual_fine` 同屏**：只说"粗层看到了 P R r"而不画"丢掉了
 *    r − P R r"，读者会把 P R r 当成残差本身。三条线一起画，"粗层只拿到一小块低频"
 *    才是看得见的事实。
 * 2. **`loss` 副轴专门用来挡第 4 个陷阱**：纯高频模态的不可见比例**恒为 0.999985**
 *    （与 ν 无关，形状不变、只是幅度每步乘 0.326），而**混合**残差的比例从 0.707 掉到 0.0018。
 *    把"纯模态的水平线"与"混合残差的下降曲线"画在同一条对数轴上，两者的差别一眼可见。
 * 3. **`alias_vs_theta` 不进图层、只进文本**：它是别名后的**频率**（有负值 = 方向反转），
 *    与"比值"不同量纲，塞进对数比值轴就是画错图；但它是"高频被认成低频"的关键证据，
 *    所以按帧 `meta.alias_theta_pi` 原样写进说明。
 * 4. **临界线 θ/π=0.5 画在 `visibility` 轴上**（规则 6）：manifest 的 `critical` 就是它，
 *    越过它粗网格不再忠实表示该频率，而是把它折叠成 2θ。
 *
 * 时间轴：`scrub`，`t = 预平滑步数 ν`（限制之前先做 ν 次平滑；ν=0 即直接限制未平滑的残差）。
 */

import { formatNumber } from "../../shared/src/axes.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import {
  applyOverflow,
  canvasSize,
  clamp,
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
  requireSameLength,
  sceneShell,
  type SceneModel,
} from "../../shared/src/model.js";
import {
  axisModeBadge,
  badge,
  criticalBadge,
  lineLayer,
  makeAxis,
  previewBadge,
  scatterLayer,
  series,
  sourceBadge,
} from "../../shared/src/scene.js";
import {
  axisTitleOf,
  axisYlabelOf,
  countMask,
  legendBlock,
  logRange,
  panel45Layout,
  textBlock,
  unitBudget,
} from "./_panel.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "4.5";
const VIZ = "residual_transfer";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "同一残差在细/粗网格上的形态对比";
const T_MAX = SPEC?.time.mode === "none" ? 0 : SPEC?.time.tMax ?? 6;

const SLOT = { source: 0, axisMode: 1, auxAxis: 2, critical: 3, status: 4, preview: 5, overflow: 6, degraded: 7 } as const;

/** 限制算子 / 平滑子的显示名（与 `server/chapters/4.5.py` 同口径）。 */
const RESTRICTS = ["full_weighting", "injection"] as const;
const SMOOTHERS = ["jacobi", "gs", "sor"] as const;

export const residualTransferModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需数组 ——
  const theta = requireArray(frame, "theta_over_pi");
  const lossTheta = requireArray(frame, "loss_vs_theta");
  const visRatio = requireArray(frame, "coarse_visible_ratio");
  const smoothTheta = requireArray(frame, "smoothing_vs_theta");
  const coarseCorr = requireArray(frame, "coarse_correction_vs_theta");
  const product = requireArray(frame, "product_vs_theta");
  const fineX = requireArray(frame, "fine_x");
  const rFine = requireArray(frame, "residual_fine");
  const rProlonged = requireArray(frame, "residual_prolonged");
  const rLost = requireArray(frame, "residual_lost");
  const coarseX = requireArray(frame, "coarse_x");
  const rCoarse = requireArray(frame, "residual_coarse");
  const lossHist = requireArray(frame, "loss_history");
  const lossAxis = requireArray(frame, "loss_axis");
  const residHist = requireArray(frame, "residual_norm_history");
  for (const [name, arr] of [
    ["loss_vs_theta", lossTheta],
    ["coarse_visible_ratio", visRatio],
    ["smoothing_vs_theta", smoothTheta],
    ["coarse_correction_vs_theta", coarseCorr],
    ["product_vs_theta", product],
  ] as const) {
    requireSameLength(theta, arr, `帧数组 theta_over_pi 与 ${name}`);
  }
  for (const [name, arr] of [
    ["residual_fine", rFine],
    ["residual_prolonged", rProlonged],
    ["residual_lost", rLost],
  ] as const) {
    requireSameLength(fineX, arr, `帧数组 fine_x 与 ${name}`);
  }
  requireSameLength(coarseX, rCoarse, "帧数组 coarse_x 与 residual_coarse");
  requireSameLength(lossAxis, lossHist, "帧数组 loss_axis 与 loss_history");
  const pitfalls = requirePitfalls(frame);

  // —— 可选数组 ——
  const degraded: string[] = [];
  const rFineFloorMask = frame.has("residual_at_floor") ? requireArray(frame, "residual_at_floor") : undefined;
  if (rFineFloorMask === undefined) degraded.push("帧缺 residual_at_floor：不报「多少点已到地板」");

  // —— meta ——
  const nu = metaNumber(frame, "nu");
  const tRequested = metaNumber(frame, "t_requested");
  if (tRequested !== undefined && nu !== undefined && Math.abs(tRequested - nu) > 1e-9) {
    degraded.push(`请求 t=${formatNumber(tRequested, 4)} 被夹到 ν=${nu}（帧 meta.nu 是实到步数）`);
  }
  const nuNow = nu ?? clamp(Math.round(opts.t), 0, T_MAX);
  const nCoarse = metaNumber(frame, "n_coarse");
  const thetaFrac = metaNumber(frame, "theta_frac");
  const modeIndex = metaNumber(frame, "mode_index");
  const mix = metaNumber(frame, "mix");
  const restrictIdx = metaNumber(frame, "restrict_index");
  const smootherIdx = metaNumber(frame, "smoother_index");
  const omega = metaNumber(frame, "omega");
  const floor = metaNumber(frame, "floor");
  const lossNow = metaNumber(frame, "loss_now");
  const lossUnsmoothed = metaNumber(frame, "loss_unsmoothed");
  const lossRatio = metaNumber(frame, "loss_ratio");
  const residRatio = metaNumber(frame, "residual_norm_ratio");
  const hfFine = metaNumber(frame, "hf_ratio_fine");
  const hfCoarse = metaNumber(frame, "hf_ratio_coarse_visible");
  const visAmplitude = metaNumber(frame, "coarse_visible_amplitude");
  const aliasTheta = metaNumber(frame, "alias_theta_pi");
  const lossAtTheta = metaNumber(frame, "loss_at_current_theta");
  const maxProduct = metaNumber(frame, "max_product");
  const stepFactorHigh = metaNumber(frame, "step_factor_high");
  if (thetaFrac === undefined) degraded.push("帧 meta 缺 theta_frac：当前模态位置不标（不按请求参数猜）");
  if (visAmplitude === undefined) degraded.push("帧 meta 缺 coarse_visible_amplitude：不报「粗层可见幅度只剩多少」（不自己算）");
  if (lossAtTheta === undefined) degraded.push("帧 meta 缺 loss_at_current_theta：不画纯模态的常数线（那是「形状不变」的直接证据，不自己拟合）");
  if (aliasTheta === undefined) degraded.push("帧 meta 缺 alias_theta_pi：不显示别名后的频率（那是本节的核心证据，不自己算）");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: nuNow,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  const L = panel45Layout(scene, width, height, { topShare: 0.58, auxH: 112, gap: 58 });

  // —— 主图：残差形态（细网格 / 粗层可见 / 看不见的部分）——
  let amp = 1e-12;
  for (const a of [rFine, rProlonged, rLost, rCoarse]) {
    const m = maxAbs(a);
    if (m !== undefined && m > amp) amp = m;
  }
  const profiles = makeAxis("profiles", [L.left, L.topY, L.leftW, L.topH], {
    xlim: [0, 1],
    ylim: [-amp * 1.15, amp * 1.15],
    xticks: [0, 0.25, 0.5, 0.75, 1],
    xlabel: "x（细网格内点位置；粗层点用粗层自己的坐标）",
    title: axisTitleOf(SPEC?.axes, "profiles", "细网格残差与粗层可见部分"),
    ylabel: axisYlabelOf(SPEC?.axes, "profiles", "r(x)"),
    autoscaled: true,
  });
  scene.layers.push(
    lineLayer("profiles", series(fineX, rFine, pal.curves[0], { width: 2.2})),
    lineLayer("profiles", series(coarseX, rCoarse, pal.residual, { width: 1.6, marker: "o"})),
    lineLayer("profiles", series(fineX, rProlonged, pal.curves[1], { width: 1.8, dash: [6, 4]})),
    lineLayer("profiles", series(fineX, rLost, pal.warn, { width: 1.6, dash: [2, 3]})),
  );

  // —— 副图：逐频率的可见性 / 互补性（对数）——
  const visLo = Math.min(posMin(visRatio) ?? 1, posMin(lossTheta) ?? 1, posMin(product) ?? 1, posMin(coarseCorr) ?? 1);
  const visHi = Math.max(maxFinite(lossTheta) ?? 1, maxFinite(coarseCorr) ?? 1, 1);
  const visibility = makeAxis("visibility", [L.left, L.botY, L.leftW, L.botH], {
    xlim: [0, 1],
    ylim: logRange(visLo * 0.6, visHi * 1.2),
    yscale: "log",
    xticks: [0, 0.25, 0.5, 0.75, 1],
    xlabel: "θ/π（细网格频率；0.5 以上只能被粗网格折叠表示）",
    title: axisTitleOf(SPEC?.axes, "visibility", "逐频率的不可见比例与互补性（对数轴）"),
    ylabel: axisYlabelOf(SPEC?.axes, "visibility", "比值"),
    autoscaled: true,
  });
  // 规则 6：临界值参考线（θ/π=0.5 是粗网格能忠实表示的频率上界）。
  const critTheta = criticalOf(frame, "theta_frac", numOr(SPEC?.params.find((p) => p.name === "theta_frac")?.critical, 0.5));
  const critValue = Array.isArray(critTheta) ? critTheta[0] ?? 0.5 : critTheta;
  scene.layers.push(
    lineLayer(
      "visibility",
      series([critValue, critValue], [visibility.ylim[0], visibility.ylim[1]], pal.reference, {
        dash: [6, 4],
        width: 1.2,
      }),
    ),
  );
  scene.layers.push(
    lineLayer("visibility", series(theta, smoothTheta, pal.reference, { width: 1.1, dash: [2, 3]})),
    lineLayer("visibility", series(theta, coarseCorr, pal.residual, { width: 1.1, dash: [2, 3]})),
    lineLayer("visibility", series(theta, lossTheta, pal.curves[0], { width: 2.2})),
    lineLayer("visibility", series(theta, visRatio, pal.curves[1], { width: 1.8, dash: [6, 4]})),
    lineLayer("visibility", series(theta, product, pal.exact, { width: 2})),
  );
  // 当前模态的落点（`mode_index` 是 1 基下标，帧给的真实选择）。
  if (modeIndex !== undefined && Number.isFinite(modeIndex)) {
    const idx = Math.round(modeIndex) - 1;
    if (idx >= 0 && idx < theta.length) {
      const xs = [theta[idx]];
      const lossPts = [lossTheta[idx]];
      const visPts = [visRatio[idx]];
      scene.layers.push(
        scatterLayer("visibility", xs, lossPts, { r: 4, color: pal.marker, label: `当前模态 θ/π=${formatNumber(theta[idx], 4)} 的不可见比例` }),
        scatterLayer("visibility", xs, visPts, { r: 4, color: pal.curves[1], alpha: 0.9, label: "同一模态粗层的可见幅度比" }),
      );
    } else {
      degraded.push(`帧 meta.mode_index=${formatNumber(modeIndex, 3)} 超出 θ 数组范围：当前模态不标（不猜下标）`);
    }
  }

  // —— 第三条轴：混合残差 vs 纯模态（第 4 个陷阱的落点）——
  const lossAxisSpec = makeAxis("loss", [L.rightX, L.topY, L.rightW, L.auxH], {
    xlim: [0, Math.max(1, lossAxis.length - 1)],
    ylim: logRange(
      Math.min(posMin(lossHist) ?? 1e-3, posMin(residHist) ?? 1e-3, lossAtTheta !== undefined && lossAtTheta > 0 ? lossAtTheta : 1),
      Math.max(maxFinite(lossHist) ?? 1, maxFinite(residHist) ?? 1, lossAtTheta ?? 1) * 1.2,
    ),
    yscale: "log",
    xticks: Array.from(new Set([0, Math.max(0, Math.round((lossAxis.length - 1) / 2)), Math.max(0, lossAxis.length - 1)])),
    xlabel: "预平滑步数 ν",
    title: "不可见比例随 ν（混合残差 vs 纯模态）",
    ylabel: "比值",
    autoscaled: true,
  });
  scene.axes.push(profiles, visibility, lossAxisSpec);
  if (lossAtTheta !== undefined && lossAtTheta > 0) {
    scene.layers.push(
      lineLayer(
        "loss",
        series([0, Math.max(1, lossAxis.length - 1)], [lossAtTheta, lossAtTheta], pal.warn, {
          dash: [5, 4],
          width: 1.3,
        }),
      ),
    );
  }
  scene.layers.push(
    lineLayer("loss", series(lossAxis, residHist, pal.curves[1], { width: 1.4, dash: [6, 4], marker: "s"})),
    lineLayer("loss", series(lossAxis, lossHist, pal.curves[0], { width: 2.4, marker: "o"})),
  );
  if (lossHist.length > 0) {
    const last = lossHist.length - 1;
    scene.layers.push(scatterLayer("loss", [lossAxis[last]], [lossHist[last]], { r: 4, color: pal.marker, label: `当前 ν=${formatNumber(lossAxis[last], 3)}` }));
  }

  // —— 文本 ——
  const restrictName = restrictIdx !== undefined ? RESTRICTS[Math.round(restrictIdx)] ?? "?" : "?";
  const smootherName = smootherIdx !== undefined ? SMOOTHERS[Math.round(smootherIdx)] ?? "?" : "?";
  if (L.columns) {
    const budget = unitBudget(L.rightW);
    const head = paramSummary(frame, opts, [
      { name: "n", label: "n" },
      { name: "mix", label: "mix" },
    ]);
    degraded.push(...head.degraded);
    const lines: string[] = [];
    lines.push(`ν=${nuNow}/${T_MAX}　${head.text}` + (nCoarse !== undefined ? `　粗层 n=${formatNumber(nCoarse, 4)}` : ""));
    lines.push(`${restrictName}　${smootherName}` + (thetaFrac !== undefined ? `　θ/π=${formatNumber(thetaFrac, 3)}` : ""));
    if (aliasTheta !== undefined) {
      lines.push(`限制后 → 粗层 ${formatNumber(aliasTheta, 4)}π（负号=反向）`);
      lines.push(`把高频认成低频，不是看不见高频`);
    }
    if (hfFine !== undefined && hfCoarse !== undefined) {
      lines.push(`高频比 细 ${formatNumber(hfFine, 3)}　粗层可见 ${formatNumber(hfCoarse, 3)}`);
      lines.push(`→ 粗层拿到的永远是低频形状`);
    }
    if (visAmplitude !== undefined) lines.push(`粗层可见幅度 ${formatNumber(visAmplitude, 4)}×（占原残差）`);
    if (lossAtTheta !== undefined) lines.push(`纯高频模态不可见比例 ${formatNumber(lossAtTheta, 6)}`);
    if (stepFactorHigh !== undefined) lines.push(`与 ν 无关：形状不变、幅度每步 ×${formatNumber(stepFactorHigh, 4)}`);
    if (lossUnsmoothed !== undefined && lossNow !== undefined) {
      lines.push(`混合残差 ${formatNumber(lossUnsmoothed, 4)}→${formatNumber(lossNow, 4)}（ν=0→${nuNow}）`);
    }
    if (lossRatio !== undefined) lines.push(`降低 ${formatNumber(1 / Math.max(lossRatio, 1e-300), 3)}×——这个才随 ν 变`);
    if (residRatio !== undefined) lines.push(`残差自身范数比 ${formatNumber(residRatio, 4)}：变的是幅度`);
    if (maxProduct !== undefined) lines.push(`互补性 max(平滑×粗校正)=${formatNumber(maxProduct, 4)} < 1`);
    if (rFineFloorMask !== undefined && countMask(rFineFloorMask) > 0) {
      lines.push(`细残差 ${formatNumber(countMask(rFineFloorMask), 4)} 个点已到地板 ${formatNumber(floor ?? 1e-16, 3)}`);
    }
    scene.texts.push(...textBlock(L.rightX, L.textTop, L.floor, lines, pal.muted, budget, "top"));
  } else {
    degraded.push("窄画布（单列）：右列说明未排下，关键数字见角标与 pitfalls 面板");
  }

  scene.texts.push(
    ...legendBlock(L.left + 6, L.topY + 6, L.leftW, "蓝=细残差 r　紫=限制后 R r　绿虚=粗层可见 P R r　红虚=看不见的部分 r−P R r", pal.muted),
    ...legendBlock(L.left + 6, L.botY + 6, L.leftW, "蓝=纯模态不可见比例　绿虚=粗层可见幅度比　橙=平滑×粗校正　灰/紫细虚=两个因子", pal.muted),
    ...legendBlock(L.rightX + 6, L.topY + 6, L.rightW, "红虚=纯模态（常数）　蓝=混合残差　绿虚=残差自身范数比", pal.muted),
  );

  // —— 角标 ——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  if (scene.dataSource.startsWith("backend+preview")) {
    scene.badges.push(placeBadge(previewBadge("拖动时只重取帧，不在前端做限制/延拓"), SLOT.preview, width));
  }
  scene.badges.push(placeBadge(axisModeBadge(true), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(badge("autoscale", `形态轴与对数副轴都自适应　ω=${omega !== undefined ? formatNumber(omega, 4) : "?"}`, 8, 26), SLOT.auxAxis, width),
  );
  scene.badges.push(
    placeBadge(
      criticalBadge("θ/π", thetaFrac ?? Number.NaN, critValue, { beyond: "above" }),
      SLOT.critical,
      width,
    ),
  );
  const status: string[] = [];
  if (lossAtTheta !== undefined) status.push(`纯模态 ${formatNumber(lossAtTheta, 6)}`);
  if (hfCoarse !== undefined) status.push(`粗层高频 ${formatNumber(hfCoarse, 3)}`);
  if (mix !== undefined) status.push(`mix=${formatNumber(mix, 3)}`);
  scene.badges.push(placeBadge(badge("status", clipBadgeText(status.join("；")), 8, 8), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  const nBad = countNonFinite(rFine) + countNonFinite(lossTheta) + countNonFinite(visRatio) + countNonFinite(lossHist);
  applyOverflow(scene, nBad, { slot: SLOT.overflow, width });
  return scene;
};

// --------------------------------------------------------------------------- //
// 局部工具（只做显示几何，不算物理量）
// --------------------------------------------------------------------------- //

/** 有限正值的最小值；没有就 undefined（对数轴只接受 > 0 的有限值）。 */
function posMin(a: ArrayLike<number>): number | undefined {
  let lo = Number.POSITIVE_INFINITY;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i];
    if (Number.isFinite(v) && v > 0 && v < lo) lo = v;
  }
  return Number.isFinite(lo) ? lo : undefined;
}

/** 有限值的最大值；没有就 undefined。 */
function maxFinite(a: ArrayLike<number>): number | undefined {
  let hi = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i];
    if (Number.isFinite(v) && v > hi) hi = v;
  }
  return Number.isFinite(hi) ? hi : undefined;
}

/** 绝对值上界；没有有限值就 undefined。 */
function maxAbs(a: ArrayLike<number>): number | undefined {
  let hi = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < a.length; i += 1) {
    const v = Math.abs(a[i]);
    if (Number.isFinite(v) && v > hi) hi = v;
  }
  return Number.isFinite(hi) ? hi : undefined;
}

/** manifest 里某个参数的 `critical` 列表的第一个值（缺省回落到给定值）。 */
function numOr(list: readonly number[] | undefined, fallback: number): number {
  return list !== undefined && list.length > 0 && Number.isFinite(list[0]) ? (list[0] as number) : fallback;
}
