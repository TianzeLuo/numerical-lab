/**
 * `phase_orbit`（3.3 章）的 model：相空间轨道是闭合曲线还是螺旋。
 *
 * 三条阅读纪律（对应本章 pitfalls）：
 *
 * 1. **“闭合”有两种读法，必须分开画**：橙线是**该方法自己的二次不变量**等值线，灰虚线是
 *    精确圆。到不变曲线的径向偏差（`closure_error`）说明“是否落在同一条曲线上”；到精确圆的
 *    偏差（`exact_error`）说明“离真值多远”。副轴把两者同屏，符号取绝对值（对数轴只收正值），
 *    并在文本里写明这一点。
 * 2. **采样点太稀时不连线**：1500 个采样点摊在 600 圈上只有 2.5 点/圈，按时间顺序连起来会
 *    画出横穿相平面的“星形弦”——那不是轨道。每圈点数低于 {@link MIN_POINTS_PER_LAP} 时
 *    轨道改画**散点**，并在面板里注明；点数够时才画折线。
 * 3. **主图窗口固定 ±1.6**（manifest 声明 `autoscaled:false`）：不随帧自适应，否则螺旋会被
 *    “重新框住”而看不出它在穿过参考曲线。越过窗口的轨道由 `applyOverflow` 挂溢出角标。
 *
 * 不变量曲线为空（h 越过稳定边界后没有实二次型）时**降级**而不是抛错：轨道与精确圆仍可读。
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
  textBlock,
  unitBudget,
} from "../_common/panel.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "3.3";
const VIZ = "phase_orbit";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "相空间轨道：闭合曲线还是螺旋";
const H_CRITICAL = SPEC?.params?.find((p) => p.name === "h")?.critical?.[0] ?? 2.0;
const FLOOR = 1e-16;
/** 每圈采样点数低于它就不连线（连线会画出不存在的弦）。 */
const MIN_POINTS_PER_LAP = 24;
const WINDOW = 1.6;

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

export const phaseOrbitModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需数组：轨道、参考曲线、两条偏差、采样时刻 ——
  const oq = requireArray(frame, "orbit_q");
  const op = requireArray(frame, "orbit_p");
  const iq = requireArray(frame, "invariant_q");
  const ip = requireArray(frame, "invariant_p");
  const eq = requireArray(frame, "exact_q");
  const ep = requireArray(frame, "exact_p");
  const closure = requireArray(frame, "closure_error");
  const exact = requireArray(frame, "exact_error");
  const tSeries = requireArray(frame, "t_series");
  requireSameLength(oq, op, "帧数组 orbit_q 与 orbit_p");
  requireSameLength(closure, tSeries, "帧数组 closure_error 与 t_series");
  requireSameLength(exact, tSeries, "帧数组 exact_error 与 t_series");
  const pitfalls = requirePitfalls(frame);

  const degraded: string[] = [];
  if (iq.length !== ip.length) {
    degraded.push("invariant_q/p 长度不一致：不画不变量等值线（拒绝错位曲线）");
  } else if (iq.length === 0) {
    degraded.push("不变量等值线为空（该 h 下没有实二次不变量）：只画数值轨道与精确圆");
  }
  if (eq.length !== ep.length) {
    degraded.push("exact_q/p 长度不一致：不画精确轨道（拒绝错位曲线）");
  } else if (eq.length === 0) {
    degraded.push("精确轨道为空：只画数值轨道与不变量曲线");
  }

  // —— meta ——
  const tAxis = metaNumber(frame, "t_axis") ?? metaNumber(frame, "t") ?? opts.t;
  const h = metaNumber(frame, "h");
  const periods = metaNumber(frame, "n_periods_observed");
  const closureSup = metaNumber(frame, "closure_error_sup");
  const exactSup = metaNumber(frame, "exact_error_sup");
  const closureGrowth = metaNumber(frame, "closure_error_growth");
  const exactGrowth = metaNumber(frame, "exact_error_growth");
  const phaseTurns = metaNumber(frame, "phase_error_turns");
  const phaseDefined = frame.meta["phase_defined"] !== false;
  const atRoundoff = frame.meta["closure_at_roundoff"] === true;
  const det = metaNumber(frame, "det_map");
  const oneMinusDet = metaNumber(frame, "one_minus_det");
  const symplectic = frame.meta["symplectic"] === true;
  const blowup = frame.meta["blowup"] === true;
  const nMasks = metaNumber(frame, "n_masked");
  const method = stringOf(frame, "method");
  const status = stringOf(frame, "status");
  if (tAxis === undefined) degraded.push("帧 meta 缺 t_axis/t：时间轴按请求 t 显示");
  if (h === undefined) degraded.push("帧 meta 缺 h：不显示步长（不按请求参数猜）");
  if (periods === undefined) degraded.push("帧 meta 缺 n_periods_observed：每圈采样点数按“未知”处理");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: tAxis ?? opts.t,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  const L = columnsLayout(scene, width, height, { topShare: 0.62, auxH: 146, gap: 44 });
  // 正方主图：相平面用同一数据尺度，圆的视觉半径才与数据一致（宽矩形会把它画成椭圆）。
  const availH = Math.max(80, L.floor - 28 - L.topY);
  const side = Math.max(80, Math.min(L.leftW, availH));
  const mainX = L.left + Math.max(0, Math.round((L.leftW - side) / 2));
  const main = makeAxis("main", [mainX, L.topY, side, side], {
    xlim: [-WINDOW, WINDOW],
    ylim: [-WINDOW, WINDOW],
    xlabel: "q",
    title: axisTitleOf(SPEC?.axes, "main", "相平面 (q,p)：数值轨道 vs 不变量曲线 vs 精确轨道"),
    ylabel: axisYlabelOf(SPEC?.axes, "main", "p"),
    autoscaled: false,
  });

  // —— 主图：参考曲线（线）+ 数值轨道（点数够才连线）——
  if (eq.length === ep.length && eq.length >= 2) {
    scene.layers.push(
      lineLayer("main", series(eq, ep, pal.reference, { width: 1.4, dash: [6, 4] })),
    );
  }
  if (iq.length === ip.length && iq.length >= 2) {
    scene.layers.push(
      lineLayer("main", series(iq, ip, pal.exact, { width: 1.6 })),
    );
  }
  const nPts = Math.min(oq.length, op.length);
  const ptsPerLap = periods !== undefined && periods > 0 ? nPts / periods : Number.POSITIVE_INFINITY;
  const drawLine = ptsPerLap >= MIN_POINTS_PER_LAP;
  if (drawLine) {
    scene.layers.push(lineLayer("main", series(oq, op, pal.curves[0] as string, { width: 2 })));
  } else {
    scene.layers.push(
      scatterLayer("main", oq, op, { r: 1.6, color: pal.curves[0] as string, alpha: 0.85 }),
    );
  }

  // —— 副轴：到不变量曲线 / 到精确圆的 |径向相对偏差|（对数）——
  const cx = tSeries;
  const cy = new Float64Array(Math.min(closure.length, tSeries.length));
  const ey = new Float64Array(Math.min(exact.length, tSeries.length));
  for (let i = 0; i < cy.length; i += 1) cy[i] = logValue(Math.abs(closure[i] as number));
  for (let i = 0; i < ey.length; i += 1) ey[i] = logValue(Math.abs(exact[i] as number));
  let hi = FLOOR;
  for (let i = 0; i < cy.length; i += 1) hi = Math.max(hi, cy[i] as number);
  for (let i = 0; i < ey.length; i += 1) hi = Math.max(hi, ey[i] as number);
  let tx0 = Number.POSITIVE_INFINITY;
  let tx1 = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < tSeries.length; i += 1) {
    const v = tSeries[i] as number;
    if (!Number.isFinite(v)) continue;
    if (v < tx0) tx0 = v;
    if (v > tx1) tx1 = v;
  }
  if (!Number.isFinite(tx0) || !Number.isFinite(tx1) || !(tx1 > tx0)) {
    tx0 = 0;
    tx1 = 1;
  }
  const auxH = Math.max(60, Math.min(120, Math.round(side * 0.36)));
  const aux = makeAxis("aux", [L.rightX, L.topY, L.rightW, auxH], {
    xlim: [tx0, tx1],
    ylim: logRange(FLOOR / 2, hi * 2),
    yscale: "log",
    xlabel: "采样时刻 t",
    title: axisTitleOf(SPEC?.axes, "aux", "到不变曲线 / 精确轨道的 |偏差|（对数）"),
    ylabel: axisYlabelOf(SPEC?.axes, "aux", "径向相对偏差"),
    autoscaled: true,
  });
  scene.axes.push(main, aux);
  scene.layers.push(
    lineLayer("aux", series([tx0, tx1], [FLOOR, FLOOR], pal.warn, { width: 1, dash: [2, 3] })),
    lineLayer("aux", series(cx, cy, pal.exact, { width: 1.8 })),
    lineLayer("aux", series(cx, ey, pal.curves[0] as string, { width: 1.8, dash: [5, 4] })),
  );

  // —— 面板内图例 ——
  scene.texts.push(
    ...legendBlock(
      mainX + 6,
      L.topY + 6,
      side,
      "灰虚=精确圆　橙=不变量曲线　蓝=" + (drawLine ? "数值轨道" : "轨道采样点"),
      pal.muted,
    ),
    ...legendBlock(
      L.rightX + 6,
      L.topY + 6,
      L.rightW,
      "橙=不变量偏差　蓝虚=精确圆偏差",
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
    if (periods !== undefined) lines.push(`观测 ${formatNumber(periods, 4)} 个精确周期`);
    if (closureSup !== undefined) {
      lines.push(`闭合偏差 sup=${formatNumber(closureSup, 3)}${atRoundoff ? "（舍入级：点集闭合）" : "（在长大）"}`);
    }
    if (exactSup !== undefined) lines.push(`到精确圆 sup=${formatNumber(exactSup, 3)}（另一种读法）`);
    if (closureGrowth !== undefined) lines.push(`前后半段偏差比 ${formatNumber(closureGrowth, 4)}（1=有界，2=线性累积）`);
    if (exactGrowth !== undefined && closureGrowth === undefined) {
      lines.push(`前后半段偏差比 ${formatNumber(exactGrowth, 4)}`);
    }
    lines.push(
      phaseDefined && phaseTurns !== undefined
        ? `相位误差 ${formatNumber(phaseTurns, 4)} 圈（累计）`
        : "相位误差本帧无法唯一确定（采样步长内转角 ≥ π）",
    );
    if (det !== undefined) {
      lines.push(
        `det M=${formatNumber(det, 8)}，1−det=${oneMinusDet !== undefined ? formatNumber(oneMinusDet, 3) : "?"}` +
          (symplectic ? "：辛" : "：非辛"),
      );
    }
    lines.push(
      `每圈采样 ${Number.isFinite(ptsPerLap) ? formatNumber(ptsPerLap, 3) : "?"} 点` +
        (drawLine ? "：点数足够，轨道连线" : "：点数太稀，轨道只画散点"),
    );
    lines.push("闭合看「到不变曲线的径向偏差」，不是「回到出发点」");
    if (blowup) lines.push(`积分越界停止：掩码 ${nMasks !== undefined ? Math.round(nMasks) : "?"} 点（占位值不画）`);
    if (status !== undefined) lines.push(`状态：${status}`);
    scene.texts.push(...textBlock(L.rightX, textTop, L.floor, lines, pal.muted, unitBudget(L.rightW), "top"));
  } else {
    degraded.push("窄画布（单列）：右列说明未排下，关键数字见角标与 pitfalls 面板");
  }

  // —— 角标 ——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(badge("autoscale", "副轴：对数自适应（地板 1e-16）", 8, 26), SLOT.aux, width),
  );
  if (h !== undefined) {
    scene.badges.push(
      placeBadge(criticalBadge("h", h, criticalOf(frame, "h", H_CRITICAL), { beyond: "above" }), SLOT.critical, width),
    );
  }
  const stateBits: string[] = [];
  if (periods !== undefined) stateBits.push(`${formatNumber(periods, 3)} 圈`);
  if (closureSup !== undefined) stateBits.push(`闭合偏差 ${formatNumber(closureSup, 3)}`);
  if (symplectic) stateBits.push("辛");
  if (blowup) stateBits.push("爆点");
  if (status !== undefined) stateBits.push(status);
  scene.badges.push(placeBadge(badge("status", stateBits.join("；") || "帧未给出可判读的状态", 8, 8), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  let nBad = 0;
  for (const arr of [oq, op, iq, ip, eq, ep, closure, exact, tSeries]) nBad += countNonFinite(arr);
  applyOverflow(scene, nBad, { slot: SLOT.overflow, width });
  return scene;
};
