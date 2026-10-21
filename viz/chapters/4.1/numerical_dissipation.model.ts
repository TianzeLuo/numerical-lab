/**
 * `numerical_dissipation`（4.1 章 viz 3）的 model：三条衰减曲线 + 掩码 + 耗散误差。
 *
 * 图层（顺序即绘制顺序）
 * ----------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | line | `decay` | `decay_exact`（真解 e^(−rφ²n)）、`decay_semidiscrete`、`decay_formula`（\|G\|ⁿ） |
 * | 2 | line | `decay` | `decay_measured`：**从真的解里分解出来的**存活率（只连 `decay_alive=1` 的点） |
 * | 3 | scatter | `decay` | 被掩码的模态钉在地板（×）——"量不出来"不是"等于 0" |
 * | 4 | line | `decay` | 地板线与 t=0 参考线（\|a_k/a_k(0)\| 的上界 1） |
 * | 5 | line | `dissipation` | 耗散误差（实测 − 半离散；负 = 数值衰减更快）+ 零基线 |
 * | 6 | line+scatter | `residual` | `amplitude_residual`（实测 − \|G\|ⁿ）：对拍精度，线性小窗口 |
 *
 * 时间轴（manifest `time.mode="scrub"`，tMax=0.01，真实时间）
 * ---------------------------------------------------------
 * 向右拖，整条曲线往下沉；`t=0` 时全部为 1。`meta.t_actual = n_steps·dt` 是实际到达
 * 的时刻，与请求的 `meta.t` 一般不等——面板文字同时给出两个数。
 *
 * 防误导七条里本章的重点
 * ----------------------
 * * **规则 1**：`decay_alive` 掩码必须真的被用上。掩码有**两种原因**，分别来自
 *   `meta.n_absent`（初值里没有这个模态 → 分母是舍入级，实测过纯 sin(πx) 初值下
 *   k≥2 的比值会给出 22.1 这种假放大率）与 `meta.n_masked_by_roundoff`（分子被别的
 *   模态的舍入淹没）。两个数**分别报**，掩码点画成 × 而不是连进曲线。
 * * **规则 4**：`decay` 轴窗口写死 1e-17–2（对数）；地板 1e-17 是双精度地板，
 *   贴上去是"量不出来"，不是"存活率为 0"。
 * * **规则 7**：`amplitude_residual` 是本 viz 的对拍证据（实测存活率与 \|G\|ⁿ 应当逐位
 *   重合，实测 ≤ 1e-14）；它同时守住"实测线不是画出来的解析线"这句话。
 */

import { formatNumber, niceLimits, niceTicks } from "../../shared/src/axes.js";
import type { Frame } from "../../shared/src/frame.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import {
  applyOverflow,
  canvasSize,
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
  resolveTextOverlaps,
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
  pitfallsBand,
  previewBadge,
  scatterLayer,
  series,
  sourceBadge,
} from "../../shared/src/scene.js";
import { manifest } from "./manifest.gen.js";
import {
  absolute,
  arrayLabelOf,
  axisAutoscaledOf,
  axisTitleFitted,
  axisYlabelOf,
  bandLayout,
  clip,
  endLabel,
  endLabelOf,
  contentBox,
  hLine,
  splitColumns,
  textBlock,
} from "./_panel.js";

const CHAPTER = "4.1";
const VIZ = "numerical_dissipation";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "数值耗散：解析衰减、半离散衰减与实测存活率";
/** manifest 的轴标题写着"窗口固定 1e-17–2"——对数纵轴写死，贴地板就看得见。 */
const DECAY_YLIM: [number, number] = [1e-17, 2];
const FLOOR_FALLBACK = 1e-17;

const SLOT = {
  source: 0,
  axisMode: 1,
  auxAxis: 2,
  critical: 3,
  overflow: 4,
  status: 5,
  preview: 6,
  degraded: 7,
} as const;

export const numericalDissipationModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需数组 ——
  const phi = requireArray(frame, "phi");
  const measured = requireArray(frame, "decay_measured");
  const alive = requireArray(frame, "decay_alive");
  const formula = requireArray(frame, "decay_formula");
  const semidiscrete = requireArray(frame, "decay_semidiscrete");
  const exact = requireArray(frame, "decay_exact");
  const dissError = requireArray(frame, "dissipation_error");
  requireSameLength(phi, measured, "帧数组 phi 与 decay_measured");
  requireSameLength(phi, alive, "帧数组 phi 与 decay_alive");
  requireSameLength(phi, formula, "帧数组 phi 与 decay_formula");
  requireSameLength(phi, semidiscrete, "帧数组 phi 与 decay_semidiscrete");
  requireSameLength(phi, exact, "帧数组 phi 与 decay_exact");
  requireSameLength(phi, dissError, "帧数组 phi 与 dissipation_error");
  const pitfalls = requirePitfalls(frame);

  // —— 可选数组 ——
  const degraded: string[] = [];
  const residualArr = optional(frame, "amplitude_residual");
  if (residualArr === undefined) degraded.push("帧缺数组 amplitude_residual：不画对拍残差（无法证明实测线不是解析线）");

  // —— 标量 meta ——
  const tMeta = metaNumber(frame, "t");
  if (tMeta === undefined) degraded.push("帧 meta 缺 t：改用请求 t（两者可能不同）");
  const tNow = tMeta ?? opts.t;
  const dt = metaNumber(frame, "dt");
  const tActual = metaNumber(frame, "t_actual");
  const nSteps = metaNumber(frame, "n_steps");
  if (dt === undefined) degraded.push("帧 meta 缺 dt：不判断请求时刻是否到达");
  if (tActual === undefined) degraded.push("帧 meta 缺 t_actual：不比对时刻（不假设相等）");
  const r = metaNumber(frame, "r");
  const theta = metaNumber(frame, "theta");
  const nMods = metaNumber(frame, "n_modes");
  const nAlive = metaNumber(frame, "n_alive");
  const nAbsent = metaNumber(frame, "n_absent");
  const nMasked = metaNumber(frame, "n_masked_by_roundoff");
  if (nAbsent === undefined) degraded.push("帧 meta 缺 n_absent：不报「初值里没有这个模态」的个数");
  if (nMasked === undefined) degraded.push("帧 meta 缺 n_masked_by_roundoff：不报「被舍入淹没」的个数");
  const maxResidual = metaNumber(frame, "max_amplitude_residual");
  const worstPhi = metaNumber(frame, "worst_phi");
  const worstDiss = metaNumber(frame, "worst_dissipation");
  const meanDiss = metaNumber(frame, "mean_dissipation");

  const dissCoef = metaNumber(frame, "dissipation_coefficient");
  const c4Num = metaNumber(frame, "c4_numeric");
  const c4Exact = metaNumber(frame, "c4_exact");
  const decayMin = metaNumber(frame, "decay_min");
  const decayMax = metaNumber(frame, "decay_max");
  const maskRel = metaNumber(frame, "mask_floor_rel");
  const hasFlip = frame.meta["has_flip"] === true;
  const phiFlip = metaNumber(frame, "phi_flip");

  const scheme = stringOf(frame, "scheme");
  const initial = stringOf(frame, "initial");
  const floor = numberFromString(stringOf(frame, "floor")) ?? FLOOR_FALLBACK;
  // 帧的 reading_note / mask_note 是完整句子（177/250 个单位）：面板通栏 6 行放不下，
  // 两条掩码原因与对拍精度在下面按"数字 + 判据"重述（信息不丢，只是换了说法）。
  const grad = paramSummary(frame, opts, [
    { name: "r", label: "$r$" },
    { name: "nx", label: "$nx$" },
  ]);
  degraded.push(...grad.degraded);

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: tNow,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  // —— 排版：左列「存活率（高）+ 耗散误差（矮）」，右列「对拍残差（通高）」；底部通栏数值块 ——
  const box = contentBox(width, height, pitfallsBand(scene)?.[1] ?? null);
  const cols = splitColumns(box, 0.62);
  // rows=7：本 viz 的数字最多（两种掩码计数必须**分别**报，且要给出判据与对拍精度），
  // 6 行会把图例挤掉；面板矮 15px 换一行数字是划算的。
  const band = bandLayout(box, { botH: Math.max(70, Math.round(box.height * 0.20)), rows: 7 });
  const decayRect: [number, number, number, number] = [cols.left.left, band.topY, cols.left.width, band.topH];
  const dissRect: [number, number, number, number] = [cols.left.left, band.botY, cols.left.width, band.botH];
  const residRect: [number, number, number, number] = [
    cols.right.left,
    band.topY,
    cols.right.width,
    band.topH + (band.botY - band.topY - band.topH) + band.botH,
  ];

  // —— 轴 ——
  const phiRange = finiteMinMax(phi) ?? [0, Math.PI];
  const phiXlim = niceLimits(phiRange[0], phiRange[1], 5);
  const decayAxis = makeAxis("decay", decayRect, {
    xlim: phiXlim,
    ylim: DECAY_YLIM,
    yscale: "log",
    ylabel: axisYlabelOf(SPEC, "decay", "|a_k(t)/a_k(0)|"),
    title: axisTitleFitted(SPEC, "decay", "逐模态存活率（对数纵轴，窗口 1e-17–2）", decayRect[2]),
    autoscaled: axisAutoscaledOf(SPEC, "decay", false),
    // 上排清空 xticks：下面那块的横轴是同一根 φ（省一条装饰带、也不与下排轴标题带对撞）。
    xticks: [],
  });

  const dissRange = finiteMinMax(dissError) ?? [-1, 0];
  const dissYlim = niceLimits(Math.min(dissRange[0], 0), Math.max(dissRange[1], 0), 5, { pad: 0.08 });
  const dissAxis = makeAxis("dissipation", dissRect, {
    xlim: phiXlim,
    ylim: dissYlim,
    ylabel: axisYlabelOf(SPEC, "dissipation", "Δ 存活率"),
    title: axisTitleFitted(SPEC, "dissipation", "耗散误差（负 = 数值衰减更快）", dissRect[2]),
    autoscaled: axisAutoscaledOf(SPEC, "dissipation", true),
    xticks: niceTicks(phiXlim[0], phiXlim[1], 5),
  });

  const residRange = residualArr === undefined ? undefined : finiteMinMax(residualArr);
  const residYlim = niceLimits(residRange?.[0] ?? -1e-14, residRange?.[1] ?? 1e-14, 5, { pad: 0.15 });
  const residAxis = makeAxis("residual", residRect, {
    xlim: phiXlim,
    ylim: residYlim,
    ylabel: "实测 − $|G|$ⁿ",
    title: "对拍残差（线性小窗口）",
    autoscaled: true,
    xticks: niceTicks(phiXlim[0], phiXlim[1], 5),
  });
  scene.axes.push(decayAxis, dissAxis, residAxis);

  // —— 图层 1：三条解析/半离散曲线 ——
  scene.layers.push(
    lineLayer("decay", [
      ...segmentedSeries(phi, exact, pal.exact, {
        width: 1.3,
        dash: [6, 3],
        accept: (v) => v > 0,
        label: endLabelOf(phi, phiXlim, decayRect[2], arrayLabelOf(frame, "decay_exact", "真解 e^(−rφ²n)")),
      }),
      ...segmentedSeries(phi, semidiscrete, pal.curves[1], {
        width: 1.3,
        dash: [2, 3],
        accept: (v) => v > 0,
        label: endLabelOf(phi, phiXlim, decayRect[2], arrayLabelOf(frame, "decay_semidiscrete", "半离散精确解")),
      }),
      ...segmentedSeries(phi, formula, pal.reference, {
        width: 1,
        accept: (v) => v > 0,
        label: endLabelOf(phi, phiXlim, decayRect[2], arrayLabelOf(frame, "decay_formula", "|G(φ)|ⁿ（解析）")),
      }),
    ]),
  );

  // —— 图层 2~3：实测存活率（只连存活点）+ 被掩码的点钉在地板 ——
  const aliveSeries = segmentedSeries(phi, measured, pal.curves[0], {
    width: 2.2,
    marker: "o",
    accept: (v, i) => (alive[i] as number) > 0.5 && v > 0,
    label: endLabelOf(phi, phiXlim, decayRect[2], arrayLabelOf(frame, "decay_measured", "实测存活率（从解里分解）")),
  });
  if (aliveSeries.length > 0) scene.layers.push(lineLayer("decay", aliveSeries));
  else degraded.push("帧 decay_alive 全为 0：没有任何可测的模态存活率");

  const maskedX: number[] = [];
  for (let i = 0; i < Math.min(phi.length, alive.length); i += 1) {
    if ((alive[i] as number) <= 0.5) maskedX.push(phi[i] as number);
  }
  if (maskedX.length > 0) {
    scene.layers.push(
      scatterLayer("decay", maskedX, maskedX.map(() => floor), {
        r: 3.4,
        color: pal.warn,
        label: endLabelOf(maskedX, phiXlim, decayRect[2], `被掩码的模态 ${maskedX.length} 个（显示在地板，不是 0）`),
      }),
    );
  }

  // —— 图层 4：地板线 + t=0 上界线 ——
  const floorLine = hLine(phiXlim[0], phiXlim[1], floor);
  const oneLine = hLine(phiXlim[0], phiXlim[1], 1);
  scene.layers.push(
    lineLayer("decay", [
      series(floorLine.x, floorLine.y, pal.reference, {
        width: 1,
        dash: [2, 3],
        label: endLabel(floorLine.x[1], phiXlim, decayRect[2], `地板 ${formatNumber(floor, 2)}`),
      }),
      series(oneLine.x, oneLine.y, pal.muted, {
        width: 1,
        dash: [4, 4],
        label: endLabel(oneLine.x[1], phiXlim, decayRect[2], "t=0 时全部为 1"),
      }),
    ]),
  );

  // —— 图层 5：耗散误差（requiredAux: error）+ 零基线 ——
  const dissSeries = segmentedSeries(phi, dissError, pal.residual, {
    width: 1.8,
    marker: "o",
    label: endLabelOf(phi, phiXlim, dissRect[2], arrayLabelOf(frame, "dissipation_error", "耗散误差（实测 − 半离散）")),
  });
  if (dissSeries.length > 0) scene.layers.push(lineLayer("dissipation", dissSeries));
  const zeroLine = hLine(phiXlim[0], phiXlim[1], 0);
  scene.layers.push(
    lineLayer("dissipation", [
      series(zeroLine.x, zeroLine.y, pal.reference, {
        width: 1,
        dash: [3, 3],
        label: endLabel(zeroLine.x[1], phiXlim, dissRect[2], "零基线（负 = 数值衰减更快）"),
      }),
    ]),
  );

  // —— 图层 6：对拍残差（requiredAux: residual）——
  if (residualArr !== undefined && residualArr.length === phi.length) {
    const absResid = absolute(residualArr);
    scene.layers.push(
      scatterLayer("residual", phi, residualArr, {
        r: 3,
        color: pal.note,
        label: endLabelOf(phi, phiXlim, residRect[2], arrayLabelOf(frame, "amplitude_residual", "实测存活率 − |G|ⁿ（应逐位为 0）")),
      }),
    );
    const maxAbs = maxOf(absResid);
    const bound = hLine(phiXlim[0], phiXlim[1], Math.max(maxAbs, 0));
    scene.layers.push(
      lineLayer("residual", [
        series(bound.x, bound.y, pal.warn, {
          width: 1,
          dash: [4, 3],
          label: endLabel(bound.x[1], phiXlim, residRect[2], `本帧 |残差| 上界 ${formatNumber(maxAbs, 3)}`),
        }),
      ]),
    );
  }

  // —— 文本：底部通栏（**两种掩码原因分别报数**；图例放最后，装不下时先砍它）——
  // 帧的 reading_note / mask_note 是完整句子（177/250 个单位）：通栏 6 行放不下三段散文，
  // 下面把两条掩码原因、判据、对拍精度都按"数字 + 最短判读"重述（信息不丢，只是换了说法）。
  const head: string[] = [
    `参数：格式=${scheme ?? "?"}　${grad.text}　θ=${formatNumber(theta ?? NaN, 4)}` +
      `　初值=${clip(initial ?? "帧未说明", 22)}　dt=${formatNumber(dt ?? NaN, 4)}　n_steps=${formatNumber(nSteps ?? 0, 6)}` +
      `　请求 t=${formatNumber(tNow, 4)} / 到达 ${formatNumber(tActual ?? NaN, 4)}`,
    `可测模态 ${formatNumber(nAlive ?? 0, 6)}/${formatNumber(nMods ?? 0, 6)}　掩码两种原因分别计数：` +
      `① 初值里没激发 n_absent=${formatNumber(nAbsent ?? 0, 6)}；② 被别的模态的舍入淹没 n_masked_by_roundoff=${formatNumber(nMasked ?? 0, 6)}`,
    `① 的分母是舍入级：纯 sin(πx) 初值下 k≥2 会给出 22.1 这种假放大率；被掩码的点画成 × 钉在地板（不是 0，是「量不出来」）`,
  ];
  const numbers: string[] = [];
  if (maxResidual !== undefined) {
    numbers.push(
      `对拍精度 max|实测−|G|ⁿ| = ${formatNumber(maxResidual, 3)}（分解出来的存活率 vs 解析 |G|ⁿ，逐位重合）` +
        (worstPhi !== undefined && worstDiss !== undefined
          ? `　耗散最重 φ=${formatNumber(worstPhi, 4)}：Δ=${formatNumber(worstDiss, 4)}（平均 ${formatNumber(meanDiss ?? NaN, 3)}）`
          : ""),
    );
  }
  if (dissCoef !== undefined || c4Num !== undefined) {
    numbers.push(
      `耗散方向系数 r/12+r²(θ−1/2)=${formatNumber(dissCoef ?? NaN, 4)}` +
        `　四阶系数 数值 ${formatNumber(c4Num ?? NaN, 4)} vs 真解 ${formatNumber(c4Exact ?? NaN, 4)}` +
        `　存活率 ${formatNumber(decayMin ?? 0, 3)}–${formatNumber(decayMax ?? 0, 3)}` +
        (hasFlip && phiFlip !== undefined ? `　φ*=${formatNumber(phiFlip, 4)}` : "　无符号翻转波数"),
    );
  }
  numbers.push(
    "图例：绿虚线=真解、蓝点线=半离散、灰线=|G|ⁿ、粗线=实测存活率、× =被掩码；系数 >0 = 衰减比真解慢；短波掉得最快 ⟹ 看起来平滑本身就是耗散证据。",
  );
  scene.texts.push(...textBlock(box.left, band.textTop, box.bottom, [...head, ...numbers], pal.muted, band.budget, "top"));

  // —— 角标 ——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  if (scene.dataSource.startsWith("backend+preview")) {
    scene.badges.push(placeBadge(previewBadge("拖动时按显示空间插值"), SLOT.preview, width));
  }
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(
      badge("autoscale", "存活率轴：固定 1e-17–2（对数）；耗散/对拍轴：自适应（会掩盖量级变化）", 8, 8),
      SLOT.auxAxis,
      width,
    ),
  );
  if (r !== undefined) {
    scene.badges.push(placeBadge(criticalBadge("r", r, criticalOfR(frame), { beyond: "above" }), SLOT.critical, width));
  }
  scene.badges.push(
    placeBadge(badge("status", maskStatus(nAbsent, nMasked, maskRel), 8, 8), SLOT.status, width),
  );
  const status = degradedBadge(degraded, SLOT.degraded, width);
  if (status !== undefined) scene.badges.push(status);

  requireDrawableLayers(scene);
  applyOverflow(scene, countNonFinite(measured) + countNonFinite(dissError) + countNonFinite(formula), {
    slot: SLOT.overflow,
    width,
  });

  const panelTop = pitfallsBand(scene)?.[1];
  scene.texts = resolveTextOverlaps(scene.texts, [8, 24, width - 16, (panelTop ?? height) - 28]);

  return scene;
};

// --------------------------------------------------------------------------- //
// 局部工具
// --------------------------------------------------------------------------- //

function optional(frame: Frame, name: string): ArrayLike<number> | undefined {
  return frame.has(name) ? (frame.array(name) as ArrayLike<number>) : undefined;
}

function numberFromString(s: string | undefined): number | undefined {
  if (s === undefined) return undefined;
  const v = Number(s);
  return Number.isFinite(v) ? v : undefined;
}

function maxOf(a: ArrayLike<number>): number {
  let m = 0;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i] as number;
    if (Number.isFinite(v) && v > m) m = v;
  }
  return m;
}

/** `frame.critical.r` 缺失时的回落值（manifest 里 `r.critical = [0.5]`）。 */
function criticalOfR(frame: Frame): number | number[] {
  const raw = frame.critical["r"];
  return raw === undefined ? 0.5 : raw;
}

/**
 * 状态角标：把**两种掩码原因分别报数**（pitfall 第 10 条）。
 *
 * 只写"有 N 个模态不可测"是不够的——"初值里没有"与"被舍入淹没"是两件事，
 * 前者是初值的问题，后者是**这个格式在这一步把量级做没了**的证据。
 */
function maskStatus(nAbsent: number | undefined, nMasked: number | undefined, maskRel: number | undefined): string {
  if (nAbsent === undefined && nMasked === undefined) return "帧 meta 缺掩码计数：不报「哪些模态不可测」";
  return (
    `掩码：初值没激发 ${nAbsent ?? "?"} / 被舍入淹没 ${nMasked ?? "?"}（判据 ${formatNumber(maskRel ?? NaN, 1)}）`
  );
}
