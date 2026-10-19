/**
 * `long_term_orbit`（3.3 章）的 model：Kepler 轨道的长期演化（e=0 / 0.6 / 0.9）。
 *
 * 三条防误导规则：
 *
 * 1. **“像不像”不是判据**：主图把数值轨道与精确椭圆画在一起，但结论必须看副轴上的
 *    能量误差、角动量漂移、近星点距离失真（全部取绝对值上对数轴，带 1e-16 地板）。
 * 2. **轨道按圈切段**：数值轨迹由 2000 个采样点组成，可能跨许多圈。用 `runsBy` 按
 *    `floor(t / 2π)` 切成一圈一段、再 `sliceRun` 逐段连线，绝不把“第 k 圈末”与
 *    “第 k+1 圈初”之间本不存在的直线画出来。每圈点数太少（如 200 圈 × 10 点）时
 *    改画散点——弦比点更会骗人。
 * 3. **主图固定窗口**：`autoscaled:false`。窗口按轨道族取 `±(1+e+0.12)`（只由 e 决定，
 *    不随帧/时间变），e=0.9 时远星点 1.9 也在窗内；e=0.6 用 ±1.72，而不是 manifest 文案里
 *    的 ±1.25——后者会把远星点裁掉。轴标题里的窗口数字同步改写成本帧真实值。
 *
 * 误差数组与轨道数组长度不同（前者 4001、后者 2000）是帧的真实形态：副轴横轴用**采样下标**，
 * 不按比例伪造一条时间轴。
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
  axisModeBadge,
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
  indexAxis,
  legendBlock,
  logRange,
  runsBy,
  sliceRun,
  textBlock,
  unitBudget,
} from "../_common/panel.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "3.3";
const VIZ = "long_term_orbit";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "行星轨道的长期演化（e=0 / 0.6 / 0.9）";
const H_CRITICAL = SPEC?.params?.find((p) => p.name === "h")?.critical?.[0] ?? 0.0023;
const FLOOR = 1e-16;
const TWO_PI = 2 * Math.PI;
const MIN_POINTS_PER_ORBIT = 24;
/** manifest 的近星点角步长判据：Δθ ≈ 0.1 rad。 */
const PERIHELION_ANGLE_STEP = 0.1;

const SLOT = {
  source: 0,
  axisMode: 1,
  aux: 2,
  critical: 3,
  status: 4,
  overflow: 6,
  degraded: 7,
} as const;

function logValue(v: number): number {
  return Number.isFinite(v) && v > 0 ? Math.max(v, FLOOR) : FLOOR;
}

/** 方法标签的短名：去掉帧文案里的括注。 */
function shortLabel(v: string | undefined): string {
  return v === undefined ? "?" : v.split("（")[0]!.trim();
}

export const longTermOrbitModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  const tx = requireArray(frame, "traj_x");
  const ty = requireArray(frame, "traj_y");
  const rx = requireArray(frame, "ref_x");
  const ry = requireArray(frame, "ref_y");
  const tSeries = requireArray(frame, "t_series");
  const energy = requireArray(frame, "energy_error");
  const angmom = requireArray(frame, "angmom_error");
  const closure = requireArray(frame, "closure_error");
  const mask = requireArray(frame, "mask_energy");
  requireSameLength(tx, ty, "帧数组 traj_x 与 traj_y");
  requireSameLength(energy, mask, "帧数组 energy_error 与 mask_energy");
  requireSameLength(angmom, mask, "帧数组 angmom_error 与 mask_energy");
  requireSameLength(closure, mask, "帧数组 closure_error 与 mask_energy");
  const pitfalls = requirePitfalls(frame);

  const degraded: string[] = [];
  if (tSeries.length !== tx.length) {
    degraded.push("t_series 与轨迹长度不一致：轨迹不按圈切段，改画散点（不按索引猜时间）");
  }
  if (rx.length !== ry.length) degraded.push("ref_x/ref_y 长度不一致：不画精确椭圆（拒绝错位）");
  else if (rx.length === 0) degraded.push("精确椭圆为空：只画数值轨道");
  const varpi = frame.has("varpi_drift") ? requireArray(frame, "varpi_drift") : undefined;
  const eccErr = frame.has("ecc_error") ? requireArray(frame, "ecc_error") : undefined;
  if (varpi === undefined || eccErr === undefined) {
    degraded.push("帧缺 varpi_drift/ecc_error：近星点方位与偏心率偏差只在文字里报 meta 值");
  }

  const tAxis = metaNumber(frame, "t_axis") ?? metaNumber(frame, "t") ?? opts.t;
  const h = metaNumber(frame, "h");
  const ecc = metaNumber(frame, "eccentricity");
  const nOrbits = metaNumber(frame, "n_orbits");
  const horizon = metaNumber(frame, "horizon_time");
  const bandEnergy = metaNumber(frame, "band_energy");
  const endEnergy = metaNumber(frame, "end_energy_error");
  const trendBand = metaNumber(frame, "trend_over_band");
  const angmomDrift = metaNumber(frame, "angmom_drift");
  const closureSup = metaNumber(frame, "closure_sup");
  const closureGrowth = metaNumber(frame, "closure_growth");
  const closureFirst = metaNumber(frame, "closure_first_half");
  const closureSecond = metaNumber(frame, "closure_second_half");
  const atRoundoff = frame.meta["closure_at_roundoff"] === true;
  const eccSup = metaNumber(frame, "ecc_error_sup");
  const eccEnd = metaNumber(frame, "ecc_error_end");
  const eccAbs = frame.meta["ecc_error_is_absolute"] === true;
  const varpiDefined = frame.meta["varpi_defined"] !== false;
  const varpiTotal = metaNumber(frame, "varpi_drift_total");
  const varpiPerOrbit = metaNumber(frame, "varpi_drift_per_orbit");
  const angleStep = metaNumber(frame, "angle_step_perihelion");
  const rPeri = metaNumber(frame, "r_peri_measured");
  const rApo = metaNumber(frame, "r_apo_measured");
  const blowup = frame.meta["blowup"] === true;
  const nMasks = metaNumber(frame, "n_masked");
  const method = stringOf(frame, "method");
  const caseName = stringOf(frame, "case");
  const status = stringOf(frame, "status");
  if (tAxis === undefined) degraded.push("帧 meta 缺 t_axis/t：时间轴按请求 t 显示");
  if (h === undefined) degraded.push("帧 meta 缺 h：不显示步长（不按请求参数猜）");
  if (ecc === undefined) degraded.push("帧 meta 缺 eccentricity：主图窗口按 ±1.6 的保守值");
  if (nOrbits === undefined) degraded.push("帧 meta 缺 n_orbits：每圈采样点数按 1 圈处理");
  if (angleStep === undefined) degraded.push("帧 meta 缺 angle_step_perihelion：h 的临界值用 manifest 值");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: tAxis ?? opts.t,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  const L = columnsLayout(scene, width, height, { topShare: 0.62, auxH: 146, gap: 44 });
  const availH = Math.max(80, L.floor - 28 - L.topY);
  const side = Math.max(80, Math.min(L.leftW, availH));
  const mainX = L.left + Math.max(0, Math.round((L.leftW - side) / 2));

  // 主图窗口：只由偏心率（轨道族）决定，不随帧/时间变。e=0.6 → ±1.72；e=0.9 → ±2.02。
  const e0 = ecc !== undefined && Number.isFinite(ecc) ? Math.max(0, ecc) : 0.6;
  const win = Math.max(1.1, 1 + e0 + 0.12);
  const baseTitle = axisTitleOf(SPEC?.axes, "main", "轨道平面 (x,y)：数值轨道 vs 精确椭圆");
  const mainTitle = /±[\d.]+/.test(baseTitle)
    ? baseTitle.replace(/±[\d.]+/g, `±${formatNumber(win, 3)}`)
    : baseTitle;
  const main = makeAxis("main", [mainX, L.topY, side, side], {
    xlim: [-win, win],
    ylim: [-win, win],
    xlabel: "x",
    title: mainTitle,
    ylabel: axisYlabelOf(SPEC?.axes, "main", "y"),
    autoscaled: false,
  });

  if (rx.length === ry.length && rx.length >= 2) {
    scene.layers.push(lineLayer("main", series(rx, ry, pal.exact, { width: 1.6 })));
  }

  // 轨道：每圈点数够 → 按圈切段连线；否则散点。绝不跨圈连一条不存在的弦。
  const nTraj = Math.min(tx.length, ty.length);
  const orbitsForDensity = Math.max(1, nOrbits ?? 0);
  const ptsPerOrbit = nTraj / orbitsForDensity;
  const canSegment = tSeries.length === tx.length && tSeries.length === ty.length;
  const drawLine = canSegment && ptsPerOrbit >= MIN_POINTS_PER_ORBIT;
  if (drawLine) {
    const lapKey = new Float64Array(tSeries.length);
    for (let i = 0; i < tSeries.length; i += 1) {
      const t = tSeries[i] as number;
      lapKey[i] = Number.isFinite(t) ? Math.floor((t + 1e-9) / TWO_PI) : Number.NaN;
    }
    const runs = runsBy(lapKey);
    const segs: ReturnType<typeof series>[] = [];
    for (const run of runs) {
      if (run.end - run.start < 2) continue;
      const seg = sliceRun(tx, ty, run);
      segs.push(series(seg.x, seg.y, pal.curves[0] as string, { width: 1.6 }));
    }
    if (segs.length > 0) {
      scene.layers.push(lineLayer("main", segs));
    } else {
      scene.layers.push(scatterLayer("main", tx, ty, { r: 1.4, color: pal.curves[0] as string, alpha: 0.8 }));
    }
  } else {
    scene.layers.push(scatterLayer("main", tx, ty, { r: 1.4, color: pal.curves[0] as string, alpha: 0.8 }));
  }

  // —— 副轴：|能量误差|、|角动量漂移|、|近星点距离失真|（对数 + 地板；横轴是采样下标）——
  const nAux = Math.max(energy.length, angmom.length, closure.length);
  const xIdx = indexAxis(nAux);
  const energyAbs = new Float64Array(energy.length);
  const angAbs = new Float64Array(angmom.length);
  const cloAbs = new Float64Array(closure.length);
  let hi = FLOOR;
  // 与能量误差共用 mask_energy：占位值段不参与画图，也不参与纵轴范围。
  const trusted = runsBy(mask).filter((r) => (mask[r.start] ?? 0) > 0.5);
  const trustedIdx = new Uint8Array(mask.length);
  for (const run of trusted) for (let i = run.start; i < Math.min(run.end, mask.length); i += 1) trustedIdx[i] = 1;
  for (let i = 0; i < energy.length; i += 1) {
    energyAbs[i] = logValue(Math.abs(energy[i] as number));
    if (trustedIdx[i] === 1) hi = Math.max(hi, energyAbs[i] as number);
  }
  for (let i = 0; i < angmom.length; i += 1) {
    angAbs[i] = logValue(Math.abs(angmom[i] as number));
    if (trustedIdx[i] === 1) hi = Math.max(hi, angAbs[i] as number);
  }
  for (let i = 0; i < closure.length; i += 1) {
    cloAbs[i] = logValue(Math.abs(closure[i] as number));
    if (trustedIdx[i] === 1) hi = Math.max(hi, cloAbs[i] as number);
  }
  if (bandEnergy !== undefined && bandEnergy > 0) hi = Math.max(hi, bandEnergy);
  const auxH = Math.max(60, Math.min(120, Math.round(side * 0.36)));
  const aux = makeAxis("aux", [L.rightX, L.topY, L.rightW, auxH], {
    xlim: [0, Math.max(1, nAux - 1)],
    ylim: logRange(FLOOR / 2, hi * 2),
    yscale: "log",
    xlabel: "采样下标（误差数组与轨迹数组长度不同）",
    title: axisTitleOf(SPEC?.axes, "aux", "能量误差 / 角动量漂移 / 近星点失真（对数）"),
    ylabel: axisYlabelOf(SPEC?.axes, "aux", "误差 / 漂移"),
    autoscaled: true,
  });
  scene.axes.push(main, aux);
  scene.layers.push(
    lineLayer("aux", series([0, Math.max(1, nAux - 1)], [FLOOR, FLOOR], pal.warn, { width: 1, dash: [2, 3] })),
  );
  if (bandEnergy !== undefined && bandEnergy > 0) {
    const y = logValue(bandEnergy);
    scene.layers.push(
      lineLayer("aux", series([0, Math.max(1, nAux - 1)], [y, y], pal.reference, { width: 1, dash: [5, 4] })),
    );
  }
  const segOf = (y: Float64Array, color: string, dash?: number[]): ReturnType<typeof series>[] => {
    const out: ReturnType<typeof series>[] = [];
    for (const run of trusted) {
      const end = Math.min(run.end, y.length);
      if (end - run.start < 2) continue;
      const seg = sliceRun(xIdx, y, { start: run.start, end });
      out.push(series(seg.x, seg.y, color, { width: 1.8, dash }));
    }
    return out;
  };
  const energySegs = segOf(energyAbs, pal.curves[0] as string);
  const angSegs = segOf(angAbs, pal.curves[1] as string, [5, 4]);
  const closSegs = segOf(cloAbs, pal.exact);
  if (energySegs.length > 0) scene.layers.push(lineLayer("aux", energySegs));
  if (angSegs.length > 0) scene.layers.push(lineLayer("aux", angSegs));
  if (closSegs.length > 0) scene.layers.push(lineLayer("aux", closSegs));

  // —— 面板图例 ——
  const mainBudget = Math.max(8, Math.floor((side - 12) / 7));
  scene.texts.push(
    ...legendBlock(
      mainX + 6,
      L.topY + 6,
      side,
      `橙=精确椭圆　蓝=${drawLine ? "数值轨道（按圈分段）" : "轨道采样点（太稀，不连线）"}`,
      pal.muted,
    ),
    ...textBlock(
      mainX + 6,
      L.topY + 23,
      L.topY + 3 * 15,
      [`窗口 ±${formatNumber(win, 3)} = 远星点 + 0.12，只由 e 决定`],
      pal.muted,
      mainBudget,
      "top",
    ),
    ...legendBlock(
      L.rightX + 6,
      L.topY + 6,
      L.rightW,
      "蓝=能量误差　绿虚=L 漂移　橙=近星点失真",
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
    lines.push(`${caseName ?? "?"}　${shortLabel(method)}　h=${h !== undefined ? formatNumber(h, 3) : "?"}`);
    if (nOrbits !== undefined) lines.push(`观测 ${Math.round(nOrbits)} 圈${horizon !== undefined ? `（t_end=${formatNumber(horizon, 4)}）` : ""}`);
    if (bandEnergy !== undefined) {
      lines.push(
        `能量误差带宽 ${formatNumber(bandEnergy, 3)}` +
          (trendBand !== undefined ? `，趋势/带宽 ${formatNumber(trendBand, 4)}` : "") +
          (endEnergy !== undefined ? `，末端 ${formatNumber(endEnergy, 3)}` : ""),
      );
    }
    if (angmomDrift !== undefined) {
      lines.push(
        `角动量相对漂移 ${formatNumber(angmomDrift, 3)}` +
          (angmomDrift <= 1e-12 ? "（在舍入级）" : "（已离开舍入级：近星点采样不足）"),
      );
    }
    if (closureSup !== undefined) {
      lines.push(
        `近星点距离失真 sup=${formatNumber(closureSup, 3)}` +
          (closureFirst !== undefined && closureSecond !== undefined
            ? `（前半 ${formatNumber(closureFirst, 3)} → 后半 ${formatNumber(closureSecond, 3)}` +
              (closureGrowth !== undefined ? `，比 ${formatNumber(closureGrowth, 4)}` : "") +
              "）"
            : ""),
      );
      if (atRoundoff) lines.push("形状在舍入级闭合（辛方法：能量有界 + 角动量逐位守恒）");
    }
    if (eccSup !== undefined) {
      lines.push(
        `偏心率${eccAbs ? "绝对" : "相对"}偏差 sup=${formatNumber(eccSup, 3)}` +
          (eccEnd !== undefined ? `，末端 ${formatNumber(eccEnd, 3)}` : ""),
      );
    }
    if (varpiDefined && varpiTotal !== undefined) {
      lines.push(
        `近星点进动 ${formatNumber(varpiTotal, 3)} rad` +
          (varpiPerOrbit !== undefined ? `（${formatNumber(varpiPerOrbit, 3)} rad/圈）` : "") +
          "：相位型误差，辛性管不了",
      );
    } else if (!varpiDefined) {
      lines.push("e=0 时没有近星点：varpi_drift 恒为 0 是占位值，不是“进动为零”");
    }
    if (angleStep !== undefined) {
      lines.push(`近星点角步长 h·L/r_p² = ${formatNumber(angleStep, 3)} rad（判据 0.1 rad）`);
    }
    if (rPeri !== undefined && rApo !== undefined) {
      lines.push(`实测近星点 ${formatNumber(rPeri, 4)} / 远星点 ${formatNumber(rApo, 4)}（初值 e=${formatNumber(e0, 3)}）`);
    }
    lines.push(
      drawLine
        ? `轨道按圈切段：每圈约 ${formatNumber(ptsPerOrbit, 3)} 点`
        : `每圈约 ${Number.isFinite(ptsPerOrbit) ? formatNumber(ptsPerOrbit, 3) : "?"} 点：太稀，只画散点`,
    );
    if (blowup) lines.push(`积分越界停止：掩码 ${nMasks !== undefined ? Math.round(nMasks) : "?"} 点（占位值不画）`);
    if (status !== undefined) lines.push(`状态：${status}`);
    scene.texts.push(...textBlock(L.rightX, textTop, L.floor, lines, pal.muted, unitBudget(L.rightW), "top"));
  } else {
    degraded.push("窄画布（单列）：右列说明未排下，关键数字见角标与 pitfalls 面板");
  }

  // —— 角标 ——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(placeBadge(badge("autoscale", "副轴：对数自适应（地板 1e-16）", 8, 26), SLOT.aux, width));
  // h 的临界值按本轨道的近星点角步长折算（manifest 的 0.0023 是 e=0.9 的值）。
  if (h !== undefined) {
    let hCrit = H_CRITICAL;
    let fromFrame = false;
    if (e0 > 1e-9 && angleStep !== undefined && angleStep > 0) {
      hCrit = h * (PERIHELION_ANGLE_STEP / angleStep);
      fromFrame = true;
    }
    if (e0 > 1e-9) {
      scene.badges.push(
        placeBadge(
          criticalBadge("h", h, fromFrame ? hCrit : H_CRITICAL, { beyond: "above" }),
          SLOT.critical,
          width,
        ),
      );
    } else {
      degraded.push("e=0 的圆轨道角速度均匀：近星点角步长判据失去意义，不挂 h 临界角标");
    }
  }
  const bits: string[] = [];
  if (nOrbits !== undefined) bits.push(`${Math.round(nOrbits)} 圈`);
  if (closureSup !== undefined) bits.push(`形状失真 ${formatNumber(closureSup, 3)}`);
  if (angmomDrift !== undefined) bits.push(`L 漂移 ${formatNumber(angmomDrift, 3)}`);
  if (status !== undefined) bits.push(status);
  scene.badges.push(placeBadge(badge("status", bits.join("；") || "帧未给出可判读的状态", 8, 8), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  let nBad = 0;
  for (const arr of [tx, ty, rx, ry, tSeries, energy, angmom, closure, mask, varpi, eccErr]) {
    if (arr !== undefined) nBad += countNonFinite(arr);
  }
  applyOverflow(scene, nBad, { slot: SLOT.overflow, width });
  return scene;
};
