/**
 * `modified_hamiltonian`（3.3 章）的 model：真实能量在带里振荡，修正能量 H_h 平在舍入级。
 *
 * 这张图的两个“不能读反”：
 *
 * 1. **绿线平 ≠ 方法准**：`modified_error` 是该方法真正守恒的二次型的相对变化；它与真实能量
 *    之间的差距正是灰带（带宽有闭式：Verlet h²/4、辛 Euler h/(2−h)）。灰带与绿线必须同屏。
 * 2. **RK4 的绿线在漂 ≠ 修正能量“算错了”**：det M = 1 − h⁶/72 ≠ 1，二次不变量**不存在**，
 *    同一条公式给出的曲线当然会漂。文本把 `det M` 与 `|1−det|` 一起报出来。
 *
 * 掩码：`mask_energy = 0` 的位置是积分停止后的占位值，按 `runsBy/sliceRun` 切段，不连线。
 * 副轴是对数轴：`|modified_error|` 与 `|1−det|` 都夹到 1e-16 地板并画出地板线。
 */

import { formatNumber } from "../../shared/src/axes.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import {
  applyOverflow,
  canvasSize,
  criticalOf,
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
  axisModeBadge,
  badge,
  bandLayer,
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
  legendBlock,
  logRange,
  runsBy,
  sliceRun,
  textBlock,
  unitBudget,
  type Run,
} from "../_common/panel.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "3.3";
const VIZ = "modified_hamiltonian";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "修正哈密顿量：真实能量在带里振荡";
const H_CRITICAL = SPEC?.params?.find((p) => p.name === "h")?.critical?.[0] ?? 2.0;
const FLOOR = 1e-16;

const SLOT = {
  source: 0,
  axisMode: 1,
  aux: 2,
  critical: 3,
  status: 4,
  overflow: 6,
  degraded: 7,
} as const;

function trustedRuns(mask: ArrayLike<number>): Run[] {
  return runsBy(mask).filter((r) => (mask[r.start] ?? 0) > 0.5);
}

function logValue(v: number): number {
  return Number.isFinite(v) && v > 0 ? Math.max(v, FLOOR) : FLOOR;
}

/** 方法标签的短名：去掉帧文案里的括注。 */
function shortLabel(v: string | undefined): string {
  return v === undefined ? "?" : v.split("（")[0]!.trim();
}

export const modifiedHamiltonianModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  const tArr = requireArray(frame, "t");
  const err = requireArray(frame, "energy_error");
  const modErr = requireArray(frame, "modified_error");
  const mask = requireArray(frame, "mask_energy");
  requireSameLength(err, mask, "帧数组 energy_error 与 mask_energy");
  requireSameLength(modErr, tArr, "帧数组 modified_error 与 t");
  const pitfalls = requirePitfalls(frame);

  const degraded: string[] = [];
  const bandHi = frame.has("energy_band_hi") ? requireArray(frame, "energy_band_hi") : undefined;
  const bandLo = frame.has("energy_band_lo") ? requireArray(frame, "energy_band_lo") : undefined;
  if (bandHi === undefined || bandLo === undefined) {
    degraded.push("帧缺 energy_band_hi/lo：不画真实能量的误差带（不按曲线自己算带宽）");
  } else if (bandHi.length !== err.length || bandLo.length !== err.length) {
    degraded.push("energy_band_hi/lo 与 energy_error 长度不一致：误差带不画（拒绝错位）");
  }
  const drift = frame.has("drift_line") ? requireArray(frame, "drift_line") : undefined;
  if (drift === undefined) degraded.push("帧缺 drift_line：不画真实能量误差的最小二乘趋势线");
  else if (drift.length !== tArr.length) degraded.push("drift_line 与 t 长度不一致：趋势线不画");

  const tAxis = metaNumber(frame, "t_axis") ?? metaNumber(frame, "t") ?? opts.t;
  const h = metaNumber(frame, "h");
  const nSteps = metaNumber(frame, "n_steps");
  const periods = metaNumber(frame, "n_periods_observed");
  const bandMeasured = metaNumber(frame, "band_measured");
  const bandTheory = metaNumber(frame, "band_theory");
  const bandRatio = metaNumber(frame, "band_ratio_theory");
  const trendBand = metaNumber(frame, "trend_over_band");
  const bandGrowth = metaNumber(frame, "band_growth");
  const modifiedSup = metaNumber(frame, "modified_sup");
  const modRoundoff = frame.meta["modified_at_roundoff"] === true;
  const det = metaNumber(frame, "det_map");
  const oneMinusDet = metaNumber(frame, "one_minus_det");
  const detTheoryRk4 = metaNumber(frame, "det_theory_rk4");
  const symplectic = frame.meta["symplectic"] === true;
  const blowup = frame.meta["blowup"] === true;
  const nMasks = metaNumber(frame, "n_masked");
  const method = stringOf(frame, "method");
  const status = stringOf(frame, "status");
  if (tAxis === undefined) degraded.push("帧 meta 缺 t_axis/t：时间轴按请求 t 显示");
  if (h === undefined) degraded.push("帧 meta 缺 h：不显示步长（不按请求参数猜）");
  if (bandMeasured === undefined) degraded.push("帧 meta 缺 band_measured：不显示实测带宽");
  if (bandTheory === undefined) degraded.push("帧 meta 缺 band_theory：不画闭式预测带（不按 h 自己推）");
  if (modifiedSup === undefined) degraded.push("帧 meta 缺 modified_sup：不显示修正量的上确界");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: tAxis ?? opts.t,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  const L = columnsLayout(scene, width, height, { topShare: 0.62, auxH: 146, gap: 44 });
  const mainH = Math.max(70, L.floor - 28 - L.topY);
  const auxH = Math.max(60, Math.min(120, Math.round(mainH * 0.46)));

  // 主轴窗口：真实能量误差与两条带一起框住（对称，零线永远在中间）。
  let maxAbs = FLOOR;
  for (const run of trustedRuns(mask)) {
    for (let i = run.start; i < Math.min(run.end, err.length); i += 1) {
      const v = err[i] as number;
      if (Number.isFinite(v)) maxAbs = Math.max(maxAbs, Math.abs(v));
    }
  }
  if (bandHi !== undefined && bandHi.length === err.length) {
    for (let i = 0; i < bandHi.length; i += 1) {
      const v = bandHi[i] as number;
      if (Number.isFinite(v)) maxAbs = Math.max(maxAbs, Math.abs(v));
    }
  }
  if (bandTheory !== undefined && Number.isFinite(bandTheory)) maxAbs = Math.max(maxAbs, Math.abs(bandTheory));

  let x1 = 0;
  for (let i = 0; i < tArr.length; i += 1) {
    const v = tArr[i] as number;
    if (Number.isFinite(v) && v > x1) x1 = v;
  }
  if (!(x1 > 0)) x1 = 1;
  const main = makeAxis("main", [L.left, L.topY, L.leftW, mainH], {
    xlim: [0, x1],
    ylim: [-1.35 * maxAbs, 1.35 * maxAbs],
    xlabel: "观测时间 $t$（数组只覆盖已观测步数）",
    title: axisTitleOf(SPEC?.axes, "main", "真实能量的相对误差与它的带"),
    ylabel: axisYlabelOf(SPEC?.axes, "main", "ΔE/E₀"),
    autoscaled: true,
    tickTarget: Math.max(3, Math.round(L.leftW / 100)),
  });

  if (bandHi !== undefined && bandLo !== undefined && bandHi.length === err.length && bandLo.length === err.length) {
    scene.layers.push(
      bandLayer("main", { x: tArr, ylo: bandLo, yhi: bandHi, color: pal.reference, alpha: 0.18, label: "实测误差带" }),
    );
  }
  scene.layers.push(lineLayer("main", series([0, x1], [0, 0], pal.reference, { width: 1, dash: [4, 4] })));
  if (bandHi !== undefined && bandLo !== undefined && bandHi.length > 0 && bandLo.length > 0) {
    const hi = bandHi[0] as number;
    const lo = bandLo[0] as number;
    if (Number.isFinite(hi)) {
      scene.layers.push(lineLayer("main", series([0, x1], [hi, hi], pal.reference, { width: 1, dash: [2, 3] })));
    }
    if (Number.isFinite(lo)) {
      scene.layers.push(lineLayer("main", series([0, x1], [lo, lo], pal.reference, { width: 1, dash: [2, 3] })));
    }
  }
  // 闭式预测带：±band_theory（h²/4 或 h/(2−h)）。它与实测带的差就是“理论没算错”的证据。
  if (bandTheory !== undefined && Number.isFinite(bandTheory) && bandTheory > 0) {
    scene.layers.push(
      lineLayer("main", series([0, x1], [bandTheory, bandTheory], pal.exact, { width: 1.2, dash: [7, 4] })),
      lineLayer("main", series([0, x1], [-bandTheory, -bandTheory], pal.exact, { width: 1.2, dash: [7, 4] })),
    );
  }
  if (drift !== undefined && drift.length === tArr.length) {
    scene.layers.push(lineLayer("main", series(tArr, drift, pal.muted, { width: 1.2, dash: [7, 4] })));
  }
  // 真实能量误差：按掩码分段（占位值不进图层）。
  let nSeg = 0;
  for (const run of trustedRuns(mask)) {
    const end = Math.min(run.end, err.length, tArr.length);
    if (end - run.start < 2) {
      if (run.start < err.length) {
        scene.layers.push(scatterLayer("main", [tArr[run.start]], [err[run.start]], { r: 2.5, color: pal.curves[0] as string }));
      }
      continue;
    }
    const seg = sliceRun(tArr, err, { start: run.start, end });
    scene.layers.push(lineLayer("main", series(seg.x, seg.y, pal.curves[0] as string, { width: 2.2 })));
    nSeg += 1;
  }
  if (nSeg === 0) {
    // 全是占位值（早期就爆掉）时也要有“真实数据”可看：画一个空图不如明确画地板。
    scene.layers.push(lineLayer("main", series([0, x1], [0, 0], pal.warn, { width: 1, dash: [3, 3] })));
  }

  // —— 副轴：|H_h 相对变化|（绿）与 |1−det M|（橙虚线），对数 + 地板 ——
  const modAbs = new Float64Array(modErr.length);
  let modHi = FLOOR;
  const trusted = trustedRuns(mask);
  for (let i = 0; i < modErr.length; i += 1) {
    modAbs[i] = logValue(Math.abs(modErr[i] as number));
  }
  for (const run of trusted) {
    for (let i = run.start; i < Math.min(run.end, modAbs.length); i += 1) {
      modHi = Math.max(modHi, modAbs[i] as number);
    }
  }
  const detDev = oneMinusDet !== undefined ? Math.abs(oneMinusDet) : 0;
  const detY = logValue(detDev);
  const aux = makeAxis("aux", [L.rightX, L.topY, L.rightW, auxH], {
    xlim: [0, x1],
    ylim: logRange(FLOOR / 2, Math.max(modHi, detY) * 2),
    yscale: "log",
    xlabel: "观测时间 $t$",
    title: axisTitleOf(SPEC?.axes, "aux", "修正能量 H_h 的相对变化（对数）"),
    ylabel: axisYlabelOf(SPEC?.axes, "aux", "修正量的相对变化"),
    autoscaled: true,
  });
  scene.axes.push(main, aux);
  scene.layers.push(
    lineLayer("aux", series([0, x1], [FLOOR, FLOOR], pal.warn, { width: 1, dash: [2, 3] })),
  );
  if (oneMinusDet !== undefined && Number.isFinite(oneMinusDet)) {
    scene.layers.push(
      lineLayer("aux", series([0, x1], [detY, detY], pal.exact, { width: 1.2, dash: [5, 4] })),
    );
  }
  const modSegs: ReturnType<typeof series>[] = [];
  for (const run of trusted) {
    const end = Math.min(run.end, modAbs.length, tArr.length);
    if (end - run.start < 2) continue;
    const seg = sliceRun(tArr, modAbs, { start: run.start, end });
    modSegs.push(series(seg.x, seg.y, pal.curves[1] as string, { width: 2 }));
  }
  if (modSegs.length > 0) scene.layers.push(lineLayer("aux", modSegs));
  // —— 面板图例 ——
  const mainBudget = Math.max(8, Math.floor((L.leftW - 12) / 7));
  scene.texts.push(
    ...legendBlock(
      L.left + 6,
      L.topY + 6,
      L.leftW,
      "蓝=真实能量误差　灰带=实测 ±sup　橙虚线=闭式带宽（±h²/4 或 ±h/(2−h)）",
      pal.muted,
    ),
    ...textBlock(
      L.left + 6,
      L.topY + 23,
      L.topY + 3 * 15,
      ["灰点线=窗口内最小二乘趋势（读方向，不读大小）"],
      pal.muted,
      mainBudget,
      "top",
    ),
    ...legendBlock(
      L.rightX + 6,
      L.topY + 6,
      L.rightW,
      "绿=修正能量　橙虚=|1−det M|",
      pal.muted,
    ),
    ...textBlock(
      L.rightX + 6,
      L.topY + 23,
      L.topY + Math.max(24, auxH - 3),
      ["取 |值|，地板 1e-16"],
      pal.muted,
      Math.max(8, Math.floor((L.rightW - 12) / 7)),
      "top",
    ),
  );

  // —— 右列文字 ——
  const textTop = L.topY + auxH + 28 + 8;
  if (L.columns) {
    const lines: string[] = [];
    lines.push(`${shortLabel(method)}　h=${h !== undefined ? formatNumber(h, 3) : "?"}`);
    if (nSteps !== undefined) {
      lines.push(
        `观测 ${Math.round(nSteps)} 步` +
          (periods !== undefined ? ` ≈ ${formatNumber(periods, 4)} 个精确周期` : "") +
          `（t=${formatNumber(scene.t, 3)}）`,
      );
    }
    if (bandMeasured !== undefined) lines.push(`实测带宽 ${formatNumber(bandMeasured, 4)}${modRoundoff ? "；H_h 变化在舍入级" : ""}`);
    if (bandTheory !== undefined) {
      if (bandTheory > 0) {
        lines.push(
          `闭式预测带宽 ${formatNumber(bandTheory, 4)}` +
            (bandRatio !== undefined ? `（实测/理论 ${formatNumber(bandRatio, 5)}）` : ""),
        );
      } else if (symplectic) {
        lines.push("闭式带宽为 0：该方法在谐振子上精确守恒能量（零线就是理论带）");
      } else {
        lines.push("该方法没有闭式带宽：det M ≠ 1，二次不变量不存在");
      }
    }
    if (trendBand !== undefined) lines.push(`趋势/带宽 ${formatNumber(trendBand, 4)}（≈1 才是线性漂移）`);
    if (bandGrowth !== undefined) lines.push(`后半/前半 sup 比 ${formatNumber(bandGrowth, 4)}（1=有界）`);
    if (modifiedSup !== undefined) {
      lines.push(`修正量 |ΔH_h/H_h| sup=${formatNumber(modifiedSup, 3)}${modRoundoff ? "（舍入级：它才是守恒量）" : "（在漂：二次不变量不存在）"}`);
    }
    if (det !== undefined) {
      lines.push(
        `det M=${formatNumber(det, 8)}　1−det=${oneMinusDet !== undefined ? formatNumber(oneMinusDet, 3) : "?"}` +
          (symplectic ? "（辛）" : "（非辛）"),
      );
    }
    if (detTheoryRk4 !== undefined && detTheoryRk4 > 0) {
      lines.push(`RK4 的 h⁶/72−h⁸/576 = ${formatNumber(detTheoryRk4, 3)}：det≠1 是“没有二次不变量”的指纹`);
    }
    lines.push("绿线平只说明它是这个方法守恒的量；它与真实 H 的距离正是灰带");
    if (blowup) lines.push(`积分越界停止：掩码 ${nMasks !== undefined ? Math.round(nMasks) : "?"} 点（占位值不画）`);
    if (status !== undefined) lines.push(`状态：${status}`);
    scene.texts.push(...textBlock(L.rightX, textTop, L.floor, lines, pal.muted, unitBudget(L.rightW), "top"));
  } else {
    degraded.push("窄画布（单列）：右列说明未排下，关键数字见角标与 pitfalls 面板");
  }

  // —— 角标 ——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(true), SLOT.axisMode, width));
  scene.badges.push(placeBadge(badge("autoscale", "副轴：对数自适应（地板 1e-16）", 8, 26), SLOT.aux, width));
  if (h !== undefined) {
    scene.badges.push(
      placeBadge(criticalBadge("h", h, criticalOf(frame, "h", H_CRITICAL), { beyond: "above" }), SLOT.critical, width),
    );
  }
  const bits: string[] = [];
  if (bandMeasured !== undefined) bits.push(`带宽 ${formatNumber(bandMeasured, 3)}`);
  if (bandRatio !== undefined && bandTheory !== undefined && bandTheory > 0) bits.push(`实测/理论 ${formatNumber(bandRatio, 4)}`);
  if (symplectic) bits.push("辛");
  if (modifiedSup !== undefined) bits.push(`H_h ${formatNumber(modifiedSup, 3)}`);
  if (status !== undefined) bits.push(status);
  scene.badges.push(placeBadge(badge("status", bits.join("；") || "帧未给出可判读的状态", 8, 8), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  let nBad = 0;
  for (const arr of [tArr, err, modErr, mask, bandHi, bandLo, drift]) {
    if (arr !== undefined) nBad += countNonFinite(arr);
  }
  applyOverflow(scene, nBad, { slot: SLOT.overflow, width });
  return scene;
};
