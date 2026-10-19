/**
 * `phase_error`（4.2 章）的 model：相速度必须**逐波数**读，而且必须知道自己在读哪条线。
 *
 * 图层（顺序即绘制顺序）
 * ----------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | band | `speed` | 预测失效区（从 `mark_breakdown` 到 φ=π，浅红） |
 * | 2 | line | `speed` | 精确相速度 1（灰虚线） |
 * | 3 | line | `speed` | 修正方程到 φ⁵ 的预测（橙虚线） |
 * | 4 | line | `speed` | 解析符号给的 c_num/c（蓝实线） |
 * | 5 | line | `speed` | φ=0.1 的读数列（灰点线）+ 该波数上的散点 |
 * | 6 | line | `amplitude` | \|G\|=1 参考线 |
 * | 7 | line | `amplitude` | \|G(φ)\|（蓝实线，固定窗口 0.15–1.15） |
 * | 8 | line | `phase` | 零相位线 |
 * | 9 | line | `phase` | ΔΦ = 2π(c_num/c−1)（紫实线，带符号） |
 *
 * 三条防误导约定
 * --------------
 * 1. **速度窗固定 −0.2–2.6、振幅窗固定 0.15–1.15**（manifest 的 `autoscaled=false`）：
 *    窗口必须包住全部实测点（c_num 最大 2.392、|G| 最小 0.283）；真越窗时如实计入
 *    溢出角标与正文，**绝不**用自适应把数据压回窗口。
 * 2. **预测失效区画成一块带**，而不是一句「预测只在收敛半径内有效」：带从
 *    `mark_breakdown` 开始，右边界就是 φ=π。`mark_breakdown = −1` 时不画带（全区间都在 1% 内），
 *    并在正文点名。
 * 3. **相位误差轴带符号**：ΔΦ>0 是超前、<0 是滞后。用对数轴画 |ΔΦ| 会把方向抹掉，
 *    正好抹掉「迎风滞后、Beam–Warming 超前」这件事。
 *
 * 时间轴：manifest `mode=none`，`scene.t` 恒为 0；ν 只从帧 `meta.nu` 读。
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
  placeBadge,
  requireArray,
  requireDrawableLayers,
  requirePitfalls,
  requireSameLength,
  sceneShell,
  stringOf,
  type SceneModel,
} from "../../shared/src/model.js";
import {
  badge,
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
  firstFinite,
  legendBlock,
  linearRange,
  textBlock,
  unitBudget,
} from "../_common/panel.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "4.2";
const VIZ = "phase_error";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "数值相速度误差曲线：逐波数谈";
const PHI_MAX = Math.PI;
const Y_SPEED: [number, number] = [-0.2, 2.6];
const Y_AMP: [number, number] = [0.15, 1.15];

const SLOT = {
  source: 0,
  axisMode: 1,
  auxAxis: 2,
  critical: 3,
  status: 4,
  preview: 5,
  overflow: 6,
  degraded: 7,
} as const;

export const phaseErrorModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需数组 ——
  const phi = requireArray(frame, "phi");
  const cNum = requireArray(frame, "c_num");
  const cPred = requireArray(frame, "c_pred");
  const cOne = requireArray(frame, "c_one");
  const phase = requireArray(frame, "phase_error");
  const gAbs = requireArray(frame, "g_abs");
  for (const [name, arr] of [
    ["c_num", cNum],
    ["c_pred", cPred],
    ["c_one", cOne],
    ["phase_error", phase],
    ["g_abs", gAbs],
  ] as const) {
    requireSameLength(phi, arr, `帧数组 phi 与 ${name}`);
  }
  const pitfalls = requirePitfalls(frame);

  // —— 可选数组 ——
  const degraded: string[] = [];
  const okMask = frame.has("prediction_ok") ? requireArray(frame, "prediction_ok") : undefined;
  if (okMask === undefined) degraded.push("帧缺 prediction_ok：不统计预测合格的波数比例");
  else requireSameLength(phi, okMask, "帧数组 phi 与 prediction_ok");
  const breakdownArr = frame.has("mark_breakdown") ? requireArray(frame, "mark_breakdown") : undefined;
  if (breakdownArr === undefined) degraded.push("帧缺 mark_breakdown：分界取 meta.prediction_breakdown_phi（缺失则不画）");

  // —— meta ——
  const nu = metaNumber(frame, "nu");
  if (nu === undefined) degraded.push("帧 meta 缺 nu：不显示 CFL 数（不按请求参数猜）");
  const d2 = metaNumber(frame, "d2");
  const a3 = metaNumber(frame, "a3");
  if (d2 === undefined || a3 === undefined) degraded.push("帧 meta 缺 d2/a3：耗散与色散系数不写（不自己展开）");
  const viscRatio = metaNumber(frame, "viscosity_ratio");
  const dispRatio = metaNumber(frame, "dispersion_ratio");
  const cAtMax = metaNumber(frame, "c_at_phi_max");
  const cPredAtMax = metaNumber(frame, "c_pred_at_phi_max");
  const breakdownMeta = metaNumber(frame, "prediction_breakdown_phi");
  const radius = metaNumber(frame, "series_radius");
  if (radius === undefined) degraded.push("帧 meta 缺 series_radius：不报级数收敛半径（不按 π 猜）");
  const cAt01 = metaNumber(frame, "speed_ratio_at_phi_0_1");
  const leads = metaNumber(frame, "leads_or_lags");
  const scheme = stringOf(frame, "scheme");

  const phiMin = firstFinite(phi) ?? 0;
  const phiMax = maxFinite(phi) ?? PHI_MAX;
  const breakdown = breakdownArr !== undefined ? firstFinite(breakdownArr) : breakdownMeta;
  const hasBand = breakdown !== undefined && breakdown > phiMin && breakdown < phiMax;

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    // mode=none：请求 t 一律不进 scene.t。
    t: 0,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  const L = columnsLayout(scene, width, height, { topShare: 0.58, auxH: 136, gap: 46 });
  const narrow = !L.columns;
  const panels = narrow ? narrowStack(L.floor, L.topY, width, 3) : undefined;

  const speed = makeAxis(
    "speed",
    panels ? panels[0] : [L.left, L.topY, L.leftW, L.topH],
    {
      xlim: [0, PHI_MAX],
      ylim: Y_SPEED,
      xticks: [],
      xlabel: "",
      title: axisTitleOf(SPEC?.axes, "speed", "相速度比与修正方程预测（固定窗口）"),
      ylabel: axisYlabelOf(SPEC?.axes, "speed", "c_num / c"),
      autoscaled: false,
    },
  );
  const amplitude = makeAxis(
    "amplitude",
    panels ? panels[1] : [L.left, L.botY, L.leftW, L.botH],
    {
      xlim: [0, PHI_MAX],
      ylim: Y_AMP,
      xticks: narrow ? [] : [0, 0.5, 1, 1.5, 2, 2.5, 3],
      xlabel: narrow ? "" : "φ = kΔx（0 … π）",
      title: axisTitleOf(SPEC?.axes, "amplitude", "同一波数上的 |G(φ)|（固定窗口）"),
      ylabel: axisYlabelOf(SPEC?.axes, "amplitude", "|G|"),
      autoscaled: false,
    },
  );
  const phaseAxis = makeAxis(
    "phase",
    panels ? panels[2] : [L.rightX, L.topY, L.rightW, L.auxH],
    {
      xlim: [0, PHI_MAX],
      ylim: phaseRange(phase),
      xticks: narrow ? [] : [0, 1, 2, 3],
      xlabel: narrow ? "" : "φ = kΔx",
      title: "相位误差 ΔΦ（带符号，自适应）",
      ylabel: "ΔΦ [rad]",
      autoscaled: true,
    },
  );
  if (narrow) {
    amplitude.title = "";
    phaseAxis.title = "";
  }
  scene.axes.push(speed, amplitude, phaseAxis);

  // —— 预测失效区：先画带，再画线（带不能压住读数）——
  if (hasBand) {
    scene.layers.push({
      kind: "band",
      axes: "speed",
      band: {
        x: [breakdown as number, PHI_MAX],
        ylo: [Y_SPEED[0], Y_SPEED[0]],
        yhi: [Y_SPEED[1], Y_SPEED[1]],
        color: pal.warn,
        alpha: 0.06,
        label: "预测失效区",
      },
    });
    scene.layers.push(
      lineLayer(
        "speed",
        series([breakdown as number, breakdown as number], [Y_SPEED[0], Y_SPEED[1]], pal.warn, {
          width: 1.2,
          dash: [6, 4],
          label: "预测相对差 >1%",
        }),
      ),
    );
  }

  // —— 主面板：1 / 预测 / 精确符号 ——
  scene.layers.push(
    lineLayer("speed", series(phi, cOne, pal.reference, { width: 1.2, dash: [5, 4] })),
    lineLayer("speed", series(phi, cPred, pal.exact, { width: 1.5, dash: [5, 4] })),
    lineLayer("speed", series(phi, cNum, pal.curves[0], { width: 2.3 })),
  );

  // φ=0.1 的读数列：把「ν 与波数同时谈」落到一个具体位置上。
  const idx01 = nearestIndex(phi, 0.1);
  if (idx01 >= 0) {
    const x01 = phi[idx01] as number;
    const y01 = cNum[idx01] as number;
    scene.layers.push(
      lineLayer("speed", series([x01, x01], [Y_SPEED[0], Y_SPEED[1]], pal.reference, { width: 0.8, dash: [2, 3] })),
      scatterLayer("speed", [x01], [y01], { r: 3, color: pal.marker }),
    );
  }
  if (hasBand) {
    const ib = nearestIndex(phi, breakdown as number);
    if (ib >= 0) {
      scene.layers.push(
        scatterLayer("speed", [phi[ib] as number], [cNum[ib] as number], {
          r: 3.2,
          color: pal.warn,
        }),
      );
    }
  }

  // —— 振幅面板：|G|=1 参考 + 直曲线 ——
  scene.layers.push(
    lineLayer("amplitude", series([0, PHI_MAX], [1, 1], pal.reference, { width: 1.2, dash: [5, 4] })),
    lineLayer("amplitude", series(phi, gAbs, pal.curves[0], { width: 2.2 })),
  );

  // —— 相位误差副轴（带符号）——
  scene.layers.push(
    lineLayer("phase", series([0, PHI_MAX], [0, 0], pal.reference, { width: 1, dash: [4, 4] })),
    lineLayer("phase", series(phi, phase, pal.residual, { width: 2 })),
  );

  // —— 面板图例 ——
  if (!narrow) {
    scene.texts.push(
      ...legendBlock(L.left + 6, L.topY + 6, L.leftW, "橙虚=修正方程预测　蓝=解析符号　灰虚=c=1　红带=预测失效区", pal.muted),
      ...legendBlock(
        L.left + 6,
        L.botY + 6,
        L.leftW,
        "蓝=|G(φ)|；窗口 0.15–1.15。低于下界的部分越窗（不是 0）",
        pal.muted,
      ),
      ...legendBlock(L.rightX + 6, L.topY + 6, L.rightW, "紫=ΔΦ；0 线以上超前、以下滞后", pal.muted),
    );
  }

  // —— 右列文字 ——
  const nPredBad = okMask !== undefined ? countMask(okMask, false) : undefined;
  const nAmpLow = countBelow(gAbs, Y_AMP[0]);
  const nSpeedOut = countOutside(cNum, Y_SPEED) + countOutside(cPred, Y_SPEED);
  if (!narrow) {
    const lines: string[] = [];
    lines.push(`${scheme ?? "?"}　ν=${nu !== undefined ? formatNumber(nu, 3) : "?"}　n_φ=${phi.length}`);
    if (leads !== undefined) {
      lines.push(
        leads > 0 ? "中频相位超前（曲线在 1 以上）" : leads < 0 ? "中频相位滞后（曲线在 1 以下）" : "中频相位与精确解重合",
      );
    }
    if (cAt01 !== undefined) lines.push(`φ=0.1：c=${formatNumber(cAt01, 5)}（ν→0 时趋近 1）`);
    if (cAtMax !== undefined) {
      const pred = cPredAtMax !== undefined ? `预测 ${formatNumber(cPredAtMax, 4)}` : "预测未给";
      lines.push(`φ_max：c=${formatNumber(cAtMax, 4)}，${pred}`);
    }
    if (breakdown !== undefined) {
      lines.push(
        breakdown > 0
          ? `预测失效 φ≥${formatNumber(breakdown, 3)}（相对差 >1%）`
          : "全区间预测相对差 ≤1%",
      );
    }
    if (radius !== undefined) lines.push(`级数收敛半径 |φ|≈${formatNumber(radius, 3)}`);
    if (nPredBad !== undefined) {
      lines.push(`预测合格 ${phi.length - nPredBad}/${phi.length}；速度越窗 ${nSpeedOut}`);
    } else if (nSpeedOut > 0) {
      lines.push(`相速度越窗 ${nSpeedOut} 点（不是缺失）`);
    }
    if (nAmpLow > 0) lines.push(`|G|<0.15 越窗 ${nAmpLow} 点（不是 0）`);
    if (d2 !== undefined) lines.push(`d2=${formatNumber(d2, 5)}　a3=${a3 !== undefined ? formatNumber(a3, 5) : "?"}`);
    if (viscRatio !== undefined || dispRatio !== undefined) {
      lines.push(`d2/ν=${viscRatio !== undefined ? formatNumber(viscRatio, 4) : "?"}　−a3/ν=${dispRatio !== undefined ? formatNumber(dispRatio, 4) : "?"}`);
    }
    lines.push("c_num 是符号值，无时间离散误差");
    scene.texts.push(...textBlock(L.rightX, L.textTop, L.floor, lines, pal.muted, unitBudget(L.rightW), "top"));
  } else {
    degraded.push("窄画布（单列）：右列说明与图例未排下，三条固定窗口仍然生效");
  }

  // —— 角标 ——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  scene.badges.push(
    placeBadge(badge("autoscale", "速度/振幅固定；相位误差副轴自适应（带符号）", 8, 26), SLOT.axisMode, width),
  );
  scene.badges.push(
    placeBadge(badge("autoscale", `速度窗 ${Y_SPEED[0]}–${Y_SPEED[1]}；振幅窗 0.15–1.15（不随帧变）`, 8, 44), SLOT.auxAxis, width),
  );
  if (nu !== undefined) {
    scene.badges.push(
      placeBadge(
        badge(
          "critical",
          `ν=${formatNumber(nu, 3)}　1/2：迎风 a3 变号（非失稳）`,
          8,
          62,
        ),
        SLOT.critical,
        width,
      ),
    );
  }
  const status: string[] = [];
  if (nu !== undefined) status.push(`ν ${formatNumber(nu, 3)}`);
  if (scheme !== undefined) status.push(scheme);
  if (cAt01 !== undefined) status.push(`c(0.1) ${formatNumber(cAt01, 4)}`);
  if (breakdown !== undefined && breakdown > 0) status.push(`预测失效 φ ${formatNumber(breakdown, 3)}`);
  scene.badges.push(placeBadge(badge("status", status.join("；"), 8, 8), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  const nBad =
    countNonFinite(phi) +
    countNonFinite(cNum) +
    countNonFinite(cPred) +
    countNonFinite(gAbs) +
    countNonFinite(phase) +
    (okMask !== undefined ? countNonFinite(okMask) : 0) +
    (breakdownArr !== undefined ? countNonFinite(breakdownArr) : 0);
  applyOverflow(scene, nBad, { slot: SLOT.overflow, width });
  return scene;
};

// --------------------------------------------------------------------------- //
// 局部工具（显示几何与计数，不算物理量）
// --------------------------------------------------------------------------- //

function narrowStack(
  floor: number,
  top: number,
  width: number,
  n: number,
): [number, number, number, number][] {
  const gap = 10;
  const avail = Math.max(40, floor - top);
  const h = Math.max(20, Math.floor((avail - gap * (n - 1)) / n));
  const w = Math.max(120, width - 88);
  const out: [number, number, number, number][] = [];
  for (let i = 0; i < n; i += 1) out.push([64, top + i * (h + gap), w, h]);
  return out;
}

function maxFinite(a: ArrayLike<number>): number | undefined {
  let hi = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i];
    if (Number.isFinite(v) && v > hi) hi = v;
  }
  return Number.isFinite(hi) ? hi : undefined;
}

function countMask(a: ArrayLike<number>, want = true): number {
  let n = 0;
  for (let i = 0; i < a.length; i += 1) {
    const hit = (a[i] ?? 0) > 0.5;
    if (hit === want) n += 1;
  }
  return n;
}

function countBelow(a: ArrayLike<number>, limit: number): number {
  let n = 0;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i];
    if (Number.isFinite(v) && v < limit) n += 1;
  }
  return n;
}

function countOutside(a: ArrayLike<number>, lim: readonly [number, number]): number {
  let n = 0;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i];
    if (Number.isFinite(v) && (v < lim[0] || v > lim[1])) n += 1;
  }
  return n;
}

/** 最近的 phi 下标（数组按升序给；不插值，只做「读数列」定位）。 */
function nearestIndex(a: ArrayLike<number>, target: number): number {
  let best = -1;
  let bestD = Number.POSITIVE_INFINITY;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i];
    if (!Number.isFinite(v)) continue;
    const d = Math.abs(v - target);
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  }
  return best;
}

/**
 * 相位误差轴的带符号范围：**强制包含 0**（全负的格式如 LW 也要看得到零线），
 * 全零给 ±0.5，避免退化轴。
 */
function phaseRange(a: ArrayLike<number>): [number, number] {
  let lo = Number.POSITIVE_INFINITY;
  let hi = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i];
    if (!Number.isFinite(v)) continue;
    lo = Math.min(lo, v);
    hi = Math.max(hi, v);
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [-0.5, 0.5];
  lo = Math.min(lo, 0);
  hi = Math.max(hi, 0);
  if (hi - lo < 1e-12) return [-0.5, 0.5];
  return linearRange(lo, hi, 0.08);
}
