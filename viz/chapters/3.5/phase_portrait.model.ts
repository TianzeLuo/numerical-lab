/**
 * `phase_portrait`（3.5 章）的 model：相平面（等比例）+ 零倾线 + 平衡点相型 + 极限环半径。
 *
 * 面板与图层（顺序即绘制顺序，测试按此断言）
 * ------------------------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | vectorfield | `phase` | 方向场（按 `grid_n` 的格点，箭头长度按格距归一化） |
 * | 2 | line | `phase` | 数值追踪的 x 零倾线（按分支号分段，不跨分支连线） |
 * | 3 | line | `phase` | 数值追踪的 y 零倾线 |
 * | 4 | line | `phase` | 闭式零倾线（虚线，`analytic_*`；条数与数值分支数一起写进文本） |
 * | 5 | line | `phase` | 极限环（只有闭式极限环的算例才有 `cycle_x/cycle_y`） |
 * | 6 | line | `phase` | 轨迹（`traj_x/traj_y` 的每一列一条；不同初值不同色） |
 * | 7 | scatter | `phase` | 初值点与当前端点 |
 * | 8 | scatter | `phase` | 平衡点（按 `fixed_kind` 分类着色：稳定/鞍点/不稳定/中心） |
 * | 9 | scatter | `residual` | 零倾线上 |f| 的残差（对数纵轴） |
 * | 10 | scatter | `residual` | 平衡点上 |f| 的残差 |
 *
 * 三条刻意的决定
 * --------------
 * 1. **相平面必须等比例**：manifest 的轴名写着"相平面（等比例）"，所以面板的像素
 *    宽高比按世界窗口 `(x_hi−x_lo):(y_hi−y_lo)` 收（`fitAspect`）。不成比例时
 *    "极限环是圆的/被压扁的"这种结论会被轴骗出来，而本图恰恰要让人看"μ 变小时
 *    环怎么缩"。
 * 2. **零倾线按分支号断开画**：零倾线在鞍点附近是两条分支（同一条曲线被切成两段），
 *    直接连起来会画出一条并不存在的"跨支直线"。`*_seg` 就是后端给的分支号，
 *    本模型照它断线，并把"数值分支数 vs 闭式分支数"写进文本——数不出来就是没对上。
 * 3. **残差单独一轴**：零倾线是**数值追踪**出来的（bisection/Newton），它的 |f|
 *    有多接近 0 是"这条线可不可信"的唯一硬指标；把 |f| 和坐标混在一根纵轴上会
 *    让 1e-16 和 1 看起来一样大。

 */

import { formatNumber, niceLimits } from "../../shared/src/axes.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import type { Frame } from "../../shared/src/frame.js";
import {
  applyOverflow,
  canvasSize,
  clipBadgeText,
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
  resolveTextOverlaps,
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
  pitfallsBand,
  scatterLayer,
  sourceBadge,
  text,
  vectorFieldLayer,
  type LineSeries,
} from "../../shared/src/scene.js";
import { badgeUnitCap, fitAspect, layout35, segmentRuns, stackTexts, textHeight } from "./common.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "3.5";
const VIZ = "phase_portrait";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "相图：零倾线、平衡点与极限环";
/** 后端给残差数组的地板（`_floor(..., 1e-18)`）——对数轴上的"0"，要画出来并说明。 */
const RESIDUAL_FLOOR = 1e-18;
/** `manifest.gen.ts` 里 `mu.critical` 的回落值（帧 `critical.mu` 缺失时才用）。 */
const MU_CRITICAL = 0;
/** 平衡点分类码（`strings.kind_codes` 给出全表；这里只定颜色，不改判据）。 */

const SLOT = {
  source: 0,
  phaseAxis: 1,
  residualAxis: 2,
  critical: 3,
  overflow: 4,
  status: 5,
  degraded: 6,
} as const;

export const phasePortraitModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需数组 ——
  const vfX = requireArray(frame, "vf_x");
  const vfY = requireArray(frame, "vf_y");
  const vfU = requireArray(frame, "vf_u");
  const vfV = requireArray(frame, "vf_v");
  const nullXX = requireArray(frame, "null_x_x");
  const nullXY = requireArray(frame, "null_x_y");
  const nullYX = requireArray(frame, "null_y_x");
  const nullYY = requireArray(frame, "null_y_y");
  const resNull = requireArray(frame, "residual_null");
  const pitfalls = requirePitfalls(frame);
  requireSameLength(vfX, vfY, "vf_x 与 vf_y");
  requireSameLength(vfX, vfU, "vf_x 与 vf_u");
  requireSameLength(vfX, vfV, "vf_x 与 vf_v");
  requireSameLength(nullXX, nullXY, "null_x_x 与 null_x_y");
  requireSameLength(nullYX, nullYY, "null_y_x 与 null_y_y");

  // —— 可选字段 ——
  const degraded: string[] = [];
  const summary = paramSummary(frame, opts, [
    { name: "mu", label: "μ" },
    { name: "c", label: "c" },
  ]);
  degraded.push(...summary.degraded);
  const nullXSeg = frame.has("null_x_seg") ? requireArray(frame, "null_x_seg") : undefined;
  const nullYSeg = frame.has("null_y_seg") ? requireArray(frame, "null_y_seg") : undefined;
  const anaXX = frame.has("analytic_x_x") ? requireArray(frame, "analytic_x_x") : undefined;
  const anaXY = frame.has("analytic_x_y") ? requireArray(frame, "analytic_x_y") : undefined;
  const anaYX = frame.has("analytic_y_x") ? requireArray(frame, "analytic_y_x") : undefined;
  const anaYY = frame.has("analytic_y_y") ? requireArray(frame, "analytic_y_y") : undefined;
  const anaXSeg = frame.has("analytic_x_seg") ? requireArray(frame, "analytic_x_seg") : undefined;
  const anaYSeg = frame.has("analytic_y_seg") ? requireArray(frame, "analytic_y_seg") : undefined;
  const fixedX = frame.has("fixed_x") ? requireArray(frame, "fixed_x") : undefined;
  const fixedY = frame.has("fixed_y") ? requireArray(frame, "fixed_y") : undefined;
  const fixedKind = frame.has("fixed_kind") ? requireArray(frame, "fixed_kind") : undefined;
  const fixedMaxReal = frame.has("fixed_max_real") ? requireArray(frame, "fixed_max_real") : undefined;
  const resFixed = frame.has("residual_fixed") ? requireArray(frame, "residual_fixed") : undefined;
  const cycleX = frame.has("cycle_x") ? requireArray(frame, "cycle_x") : undefined;
  const cycleY = frame.has("cycle_y") ? requireArray(frame, "cycle_y") : undefined;
  if (nullXSeg === undefined || nullYSeg === undefined) {
    degraded.push("帧缺 null_x_seg/null_y_seg：零倾线按连续有限段断开画（分支号缺失，可能把两条分支连起来）");
  }
  if (anaXX === undefined || anaXY === undefined || anaYX === undefined || anaYY === undefined) {
    degraded.push("帧缺 analytic_* 闭式零倾线：只画数值追踪的分支（不猜闭式解）");
  }
  if (fixedX === undefined || fixedY === undefined || fixedKind === undefined) {
    degraded.push("帧缺 fixed_x/fixed_y/fixed_kind：不画平衡点（不猜位置与相型）");
  }

  const xLo = metaNumber(frame, "x_lo");
  const xHi = metaNumber(frame, "x_hi");
  const yLo = metaNumber(frame, "y_lo");
  const yHi = metaNumber(frame, "y_hi");
  let window: [number, number, number, number];
  if (xLo !== undefined && xHi !== undefined && yLo !== undefined && yHi !== undefined && xHi > xLo && yHi > yLo) {
    window = [xLo, xHi, yLo, yHi];
  } else {
    const dx = finiteMinMax(vfX) ?? [-1, 1];
    const dy = finiteMinMax(vfY) ?? [-1, 1];
    window = [dx[0], dx[1] > dx[0] ? dx[1] : dx[0] + 1, dy[0], dy[1] > dy[0] ? dy[1] : dy[0] + 1];
    degraded.push("帧 meta 缺 x_lo/x_hi/y_lo/y_hi：相平面窗口按方向场格点数据取整（**这一次是自适应的**，不同参数下环的大小不可比）");
  }

  const systemCode = metaNumber(frame, "system_code");
  const mu = metaNumber(frame, "mu");
  const muCritical = metaNumber(frame, "mu_critical");
  const nTraj = metaNumber(frame, "n_traj");
  const gridN = metaNumber(frame, "grid_n");
  const tNow = metaNumber(frame, "t_now");
  const tEnd = metaNumber(frame, "t_end");
  const nStable = metaNumber(frame, "n_stable");
  const nUnstable = metaNumber(frame, "n_unstable");
  const nSaddle = metaNumber(frame, "n_saddle");
  const maxResNull = metaNumber(frame, "max_residual_null");
  const maxResFixed = metaNumber(frame, "max_residual_fixed");
  const limitRadius = metaNumber(frame, "limit_radius");
  const limitTheory = metaNumber(frame, "limit_radius_theory");
  const theoryNull = frame.meta["limit_radius_theory_null"] === true;
  const nNullBranches = metaNumber(frame, "n_null_branches");
  const nAnalyticBranches = metaNumber(frame, "n_analytic_branches");
  const nFEvals = metaNumber(frame, "n_f_evals");
  const blowup = frame.meta["blowup"] === true;
  const kindCodes = stringOf(frame, "kind_codes");
  const limitNote = stringOf(frame, "limit_cycle_note");
  const nullNote = stringOf(frame, "nullcline_protocol");
  if (mu === undefined) degraded.push("帧 meta 缺 mu：不标 μ、不标临界（不猜）");
  if (limitRadius === undefined) degraded.push("帧 meta 缺 limit_radius：不标极限环半径（不猜）");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: tNow ?? opts.t,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  // ------------------------------------------------------------------------- //
  // 排版：左「等比例相平面」+ 右「文本列」，底部残差轴
  // ------------------------------------------------------------------------- //
  const panels = layout35(width, height, { mainShare: 0.72, gap: 50 });
  const phaseRect = fitAspect(panels.main, window[1] - window[0], window[3] - window[2]);
  scene.axes.push(
    makeAxis("phase", phaseRect, {
      xlim: [window[0], window[1]],
      ylim: [window[2], window[3]],
      xlabel: "x",
      ylabel: "y",
      title: "相平面（等比例）",
      autoscaled: false,
    }),
  );
  const textX = phaseRect[0] + phaseRect[2] + 22;
  const textW = panels.main[0] + panels.main[2] - textX;

  // —— 图层 1：方向场 ——
  const grid = gridOf(vfX.length, gridN);
  if (grid === undefined) {
    degraded.push(
      `方向场有 ${vfX.length} 个点，但帧 meta 的 grid_n=${fmt(gridN, 4)} 与它不成平方数关系：不画方向场（不猜网格形状）`,
    );
  } else {
    const [nx, ny] = grid;
    const spanX = window[1] - window[0];
    const spanY = window[3] - window[2];
    const stepX = nx > 1 ? spanX / (nx - 1) : spanX;
    const stepY = ny > 1 ? spanY / (ny - 1) : spanY;
    let magMax = 0;
    for (let i = 0; i < vfU.length; i += 1) {
      const m = Math.hypot(vfU[i] as number, vfV[i] as number);
      if (Number.isFinite(m) && m > magMax) magMax = m;
    }
    // 箭头长度按格距归一化（显示几何）：最大的箭头占格距的 45%，这样箭头之间不会叠。
    const scale = magMax > 0 ? (0.45 * Math.min(stepX, stepY)) / magMax : 1;
    scene.layers.push(vectorFieldLayer("phase", vfU, vfV, nx, ny, window, { stride: 1, color: pal.muted, scale }));
  }

  // —— 图层 2~3：数值零倾线（按分支号断开）——
  const numX = branchSeries(nullXX, nullXY, nullXSeg, pal.curves[0], { width: 1.6, label: "数值零倾线：dx/dt = 0" });
  const numY = branchSeries(nullYX, nullYY, nullYSeg, pal.curves[1], { width: 1.6, label: "数值零倾线：dy/dt = 0" });
  if (numX.length > 0) scene.layers.push(lineLayer("phase", numX));
  if (numY.length > 0) scene.layers.push(lineLayer("phase", numY));
  // —— 图层 4：闭式零倾线（虚线）——
  if (anaXX !== undefined && anaXY !== undefined) {
    const s = branchSeries(anaXX, anaXY, anaXSeg, pal.exact, { width: 1.1, dash: [5, 4], label: "闭式零倾线" });
    if (s.length > 0) scene.layers.push(lineLayer("phase", s));
  }
  if (anaYX !== undefined && anaYY !== undefined) {
    const s = branchSeries(anaYX, anaYY, anaYSeg, pal.exact, { width: 1.1, dash: [5, 4], label: "闭式零倾线" });
    if (s.length > 0) scene.layers.push(lineLayer("phase", s));
  }
  // —— 图层 5：闭式极限环 ——
  if (cycleX !== undefined && cycleY !== undefined && cycleX.length === cycleY.length && cycleX.length > 0) {
    scene.layers.push(lineLayer("phase", segmentedSeries(cycleX, cycleY, pal.marker, { width: 1.8, dash: [7, 4], label: "闭式极限环" })));
  }
  // —— 图层 6：轨迹（每列一条）——
  const traj = trajectorySeries(frame, nTraj, degraded, pal);
  if (traj !== undefined) scene.layers.push(lineLayer("phase", traj.layers));
  // —— 图层 7：初值与当前端点 ——
  const startX = frame.has("start_x") ? requireArray(frame, "start_x") : undefined;
  const startY = frame.has("start_y") ? requireArray(frame, "start_y") : undefined;
  const markX = frame.has("mark_x") ? requireArray(frame, "mark_x") : undefined;
  const markY = frame.has("mark_y") ? requireArray(frame, "mark_y") : undefined;
  if (startX !== undefined && startY !== undefined && startX.length === startY.length && startX.length > 0) {
    scene.layers.push(scatterLayer("phase", startX, startY, { r: 1.6, color: pal.reference, label: "初值" }));
    if (markX !== undefined && markY !== undefined && markX.length === markY.length) {
      scene.layers.push(scatterLayer("phase", markX, markY, { r: 3.0, color: pal.ink, label: "当前端点" }));
    }
  } else {
    degraded.push("帧缺 start_x/start_y：不标初值点（不猜）");
  }
  // —— 图层 8：平衡点（按相型分组着色）——
  if (fixedX !== undefined && fixedY !== undefined && fixedKind !== undefined && fixedX.length === fixedKind.length) {
    const byKind = new Map<number, number[]>();
    for (let i = 0; i < fixedX.length; i += 1) {
      const k = Math.round(fixedKind[i] as number);
      const xs = byKind.get(k);
      if (xs === undefined) byKind.set(k, [i]);
      else xs.push(i);
    }
    for (const [k, idx] of [...byKind.entries()].sort((a, b) => a[0] - b[0])) {
      const xs = idx.map((i) => fixedX[i] as number);
      const ys = idx.map((i) => fixedY[i] as number);
      scene.layers.push(
        scatterLayer("phase", Float64Array.from(xs), Float64Array.from(ys), {
          r: 4.4,
          color: kindColor(k, pal),
        }),
      );
    }
  }

  // ------------------------------------------------------------------------- //
  // 残差轴：零倾线的 |f| 与平衡点的 |f|
  // ------------------------------------------------------------------------- //
  const resX = residualAxisX(nullXX, nullYX, resNull);
  const resYLim = residualLimits(resNull, resFixed);
  scene.axes.push(
    makeAxis("residual", panels.error, {
      xlim: resX.lim,
      ylim: resYLim,
      yscale: "log",
      xlabel: resX.label,
      ylabel: "|f|",
      title: "零倾线上的 |f| 残差",
      autoscaled: true,
    }),
  );
  scene.layers.push(
    lineLayer("residual", [
      { x: resX.lim, y: [RESIDUAL_FLOOR, RESIDUAL_FLOOR], color: pal.muted, width: 1, dash: [1, 3], label: "残差地板" },
    ]),
  );
  scene.layers.push(
    scatterLayer("residual", resX.x, resNull, { r: 1.1, color: pal.curves[0], alpha: 0.85, label: "零倾线上的残差" }),
  );
  let nFixedZero = 0;
  if (fixedX !== undefined && resFixed !== undefined && resFixed.length === fixedX.length) {
    const xs: number[] = [];
    const ys: number[] = [];
    for (let i = 0; i < resFixed.length; i += 1) {
      const v = resFixed[i] as number;
      if (Number.isFinite(v) && v > 0) {
        xs.push(fixedX[i] as number);
        ys.push(v);
      } else if (v === 0) {
        nFixedZero += 1;
      }
    }
    if (xs.length > 0) {
      scene.layers.push(
        scatterLayer("residual", Float64Array.from(xs), Float64Array.from(ys), { r: 4, color: pal.marker, label: "不动点残差" }),
      );
    }
  }

  // ------------------------------------------------------------------------- //
  // 文本
  // ------------------------------------------------------------------------- //
  const headLines: string[] = [];
  headLines.push(
    `${summary.text}　系统码 ${fmt(systemCode, 4)}　${nTraj === undefined ? "" : `${fmt(nTraj, 4)} 条轨迹`}${
      tEnd === undefined ? "" : `（积分到 t=${fmt(tEnd, 4)}）`
    }`,
  );
  headLines.push(
    `零倾线是数值追踪出来的：本帧 ${fmt(nNullBranches, 4)} 条分支（闭式 ${fmt(nAnalyticBranches, 4)} 条）——` +
      `两条数对不上就说明追踪漏了分支或多出假分支。${nullNote ?? ""}`,
  );
  if (maxResNull !== undefined) {
    const resText = `零倾线上 |f| 最大 ${formatNumber(maxResNull, 3)}`;
    headLines.push(
      nFixedZero > 0
        ? `${resText}；平衡点上 |f| 最大 ${fmt(maxResFixed, 3)}，其中 ${nFixedZero} 个「恰好是 0」（对数轴画不出 0）。`
        : `${resText}；平衡点上 |f| 最大 ${fmt(maxResFixed, 3)}。`,
    );
  }
  if (limitRadius !== undefined) {
    const theoryText = theoryNull
      ? "闭式半径：本系统没有（帧写明 limit_radius_theory=null）——这个读数是耗散/收敛尺度，不是极限环半径"
      : `闭式半径 √μ=${fmt(limitTheory, 6)}（差 ${fmt(limitTheory !== undefined ? Math.abs(limitRadius - limitTheory) : undefined, 3)}）`;
    headLines.push(`极限环半径（数值）${formatNumber(limitRadius, 6)}；${theoryText}。`);
  }
  if (limitNote !== undefined) headLines.push(limitNote);
  if (mu !== undefined && muCritical !== undefined) {
    headLines.push(
      `μ_c=${formatNumber(muCritical, 4)}（Hopf，manifest 的临界值）：μ ${mu > muCritical ? ">" : "≤"} μ_c —— ${
        mu > muCritical ? "平衡点失稳、闭式极限环出现" : "平衡点还稳定，环要么不存在要么由非线性决定"
      }`,
    );
  }
  if (fixedMaxReal !== undefined && fixedMaxReal.length > 0) {
    headLines.push(
      `平衡点特征值最大实部 ${Array.from({ length: fixedMaxReal.length }, (_, i) => formatNumber(fixedMaxReal[i] as number, 3)).join(" / ")}` +
        `（>0 = 不稳定；等号 0 = 临界，此时线性化判不了相型）`,
    );
  }
  if (nStable !== undefined || nUnstable !== undefined || nSaddle !== undefined) {
    headLines.push(
      `平衡点计数：稳定 ${fmt(nStable, 4)}、不稳定 ${fmt(nUnstable, 4)}、鞍点 ${fmt(nSaddle, 4)}（相型由特征值的实部定，见下方分类码）`,
    );
  }
  if (kindCodes !== undefined) headLines.push(kindCodes);
  if (blowup) headLines.push("⚠ 帧 meta.blowup=true：本帧有轨迹发散——发散时「看起来是平的」正是自适应轴最会骗人的地方。");
  if (textW > 110) {
    const blocks = headLines.map((body, i) => ({
      body,
      color: i === 0 ? pal.ink : pal.muted,
      size: 10,
      role: (i === 0 ? "label" : "note") as "label" | "note",
      gap: 4,
    }));
    const top = phaseRect[1] + 2;
    const bottom = phaseRect[1] + phaseRect[3] - 2;
    let st = stackTexts(textX, top, textW, bottom, blocks);
    // 一栏放不下就分两栏（说明列比相图宽得多时很常见）：宁可分栏，不可丢内容
    if (st.dropped.length > 0 && textW > 260) {
      // 按"高度"对半分，不按条数：块高差得很远时按条数分会把长块全塞进第一栏
      const colW = (textW - 14) / 2;
      const heights = blocks.map((b) => textHeight(b.body, b.size ?? 10, colW) + (b.gap ?? 4));
      const total = heights.reduce((x, y) => x + y, 0);
      let acc = 0;
      let k = 0;
      while (k < blocks.length - 1 && acc + (heights[k] as number) <= total / 2 + 24) {
        acc += heights[k] as number;
        k += 1;
      }
      const a = stackTexts(textX, top, colW, bottom, blocks.slice(0, k));
      const b = stackTexts(textX + colW + 14, top, colW, bottom, blocks.slice(k));
      st = { texts: [...a.texts, ...b.texts], dropped: [...a.dropped, ...b.dropped] };
    }
    scene.texts.push(...st.texts);
    if (st.dropped.length > 0) degraded.push(`相图旁说明列放不下 ${st.dropped.length} 段（列宽 ${Math.round(textW)}px）：未显示，口径仍见角标`);
  } else {
    degraded.push("画布偏窄：相图旁的说明列放不下，口径只保留角标与面板内文字");
  }
  scene.texts.push(
    text(
      phaseRect[0] + 6,
      phaseRect[1] + 4,
      `等比例相图：像素尺度两轴相同（窗口 ${fmt(window[1] - window[0], 3)}×${fmt(window[3] - window[2], 3)}）——不成比例时"环变扁/变圆"会被轴造出来。`,
      { role: "note", size: 10, color: pal.muted, maxWidth: phaseRect[2] * 0.96 },
    ),
  );

  // 残差轴上的口径说明（面板只有 50px 高，压到两行）
  let nResZero = 0;
  for (let i = 0; i < resNull.length; i += 1) {
    const v = resNull[i] as number;
    if (Number.isFinite(v) && v <= 0) nResZero += 1;
  }
  scene.texts.push(
    text(
      panels.error[0] + 6,
      panels.error[1] + 3,
      `|f|→0 更可信；f 求值 ${fmt(nFEvals, 6)} 次${nResZero > 0 ? `；${nResZero} 点 |f|=0（对数轴画不出→溢出角标）` : ""}`,
      { role: "note", size: 10, color: pal.muted, maxWidth: panels.error[2] * 0.98 },
    ),
  );

  // ------------------------------------------------------------------------- //
  // 角标（规则 4/5/6）
  // ------------------------------------------------------------------------- //
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.phaseAxis, width));
  scene.badges.push(placeBadge(badge("autoscale", "残差轴：自适应对数（按数据取整）", 8, 8), SLOT.residualAxis, width));
  if (mu !== undefined) {
    scene.badges.push(
      placeBadge(
        criticalBadge("μ", mu, criticalOf(frame, "mu", MU_CRITICAL), { beyond: "above" }),
        SLOT.critical,
        width,
      ),
    );
  }
  scene.badges.push(
    placeBadge(
      badge(
        "status",
        clipBadgeText(
          `t=${fmt(tNow, 4)}｜轨迹 ${fmt(nTraj, 4)} 条、方向场 ${grid === undefined ? "?" : `${grid[0]}×${grid[1]}`}｜` +
            `零倾线 |f|≤${fmt(maxResNull, 3)}｜${blowup ? "有发散" : "无发散"}`,
          badgeUnitCap(width),
        ),
        8,
        8,
      ),
      SLOT.status,
      width,
    ),
  );
  const status = degradedBadge(degraded, SLOT.degraded, width);
  if (status !== undefined) scene.badges.push(status);

  requireDrawableLayers(scene);
  applyOverflow(scene, countNonFinite(vfU) + countNonFinite(resNull) + countNonFinite(nullXX), { slot: SLOT.overflow, width });

  const panelTop = pitfallsBand(scene)?.[1];
  scene.texts = resolveTextOverlaps(scene.texts, [8, 24, width - 16, (panelTop ?? height) - 28]);
  return scene;
};

// --------------------------------------------------------------------------- //
// 局部工具
// --------------------------------------------------------------------------- //

function fmt(v: number | undefined, sig: number): string {
  return v === undefined ? "?" : formatNumber(v, sig);
}

function kindColor(kind: number, pal: ReturnType<typeof palette>): string {
  if (kind === 0 || kind === 1) return pal.curves[1];
  if (kind === 2) return pal.residual;
  if (kind === 3) return pal.exact;
  if (kind === 4 || kind === 5) return pal.marker;
  return pal.muted;
}

/** 方向场网格形状：`grid_n²` 与点数一致才认（认不出就别画，别猜）。 */
function gridOf(count: number, gridN: number | undefined): [number, number] | undefined {
  if (gridN !== undefined && gridN >= 2 && gridN * gridN === count) return [gridN, gridN];
  const n = Math.round(Math.sqrt(count));
  if (n >= 2 && n * n === count) return [n, n];
  return undefined;
}

/** 按分支号（`*_seg`）把一条曲线切成若干条 series；没有分支号时退化为"按非有限值断开"。 */
function branchSeries(
  x: ArrayLike<number>,
  y: ArrayLike<number>,
  seg: ArrayLike<number> | undefined,
  color: string,
  style: { width?: number; dash?: number[]; label?: string },
): LineSeries[] {
  if (seg === undefined || seg.length !== x.length) {
    return segmentedSeries(x, y, color, { ...style, width: style.width ?? 1.4, accept: (v, i) => Number.isFinite(v) && y[i] === y[i] });
  }
  const out: LineSeries[] = [];
  let first = true;
  for (const run of segmentRuns(seg)) {
    if (run.end - run.start < 2) continue;
    const xs: number[] = [];
    const ys: number[] = [];
    for (let i = run.start; i < run.end; i += 1) {
      xs.push(x[i] as number);
      ys.push(y[i] as number);
    }
    out.push({
      x: Float64Array.from(xs),
      y: Float64Array.from(ys),
      color,
      width: style.width ?? 1.4,
      ...(style.dash === undefined ? {} : { dash: style.dash }),
      label: first ? style.label : undefined,
    });
    first = false;
  }
  return out;
}

/** 残差轴的横坐标：优先用"零倾线点的 x"（dx/dt=0 在前、dy/dt=0 在后，与 case 顺序一致），对不上才退化为序号。 */
function residualAxisX(
  nullXX: ArrayLike<number>,
  nullYX: ArrayLike<number>,
  resNull: ArrayLike<number>,
): { x: Float64Array; lim: [number, number]; label: string } {
  if (nullXX.length + nullYX.length === resNull.length) {
    const out = new Float64Array(resNull.length);
    for (let i = 0; i < nullXX.length; i += 1) out[i] = nullXX[i] as number;
    for (let i = 0; i < nullYX.length; i += 1) out[nullXX.length + i] = nullYX[i] as number;
    const mm = finiteMinMax(out) ?? [0, 1];
    return { x: out, lim: mm[1] > mm[0] ? mm : [mm[0], mm[0] + 1], label: "零倾线点的 x" };
  }
  const out = new Float64Array(resNull.length);
  for (let i = 0; i < resNull.length; i += 1) out[i] = i;
  return { x: out, lim: [0, Math.max(1, resNull.length - 1)], label: "零倾线点序号（帧的对齐顺序未知）" };
}

function residualLimits(resNull: ArrayLike<number>, resFixed: ArrayLike<number> | undefined): [number, number] {
  const parts: ArrayLike<number>[] = [resNull, Float64Array.from([RESIDUAL_FLOOR])];
  if (resFixed !== undefined) parts.push(resFixed);
  let lo = Number.POSITIVE_INFINITY;
  let hi = Number.NEGATIVE_INFINITY;
  for (const a of parts) {
    for (let i = 0; i < a.length; i += 1) {
      const v = a[i] as number;
      if (!Number.isFinite(v) || v <= 0) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [1e-18, 1];
  return niceLimits(lo, hi, 4, { log: true, pad: 0.06 });
}

interface TrajSeries {
  layers: LineSeries[];
}

/** 轨迹矩阵的每一列一条线（列 = 初值，行 = 时间步）。 */

/** 轨迹矩阵的每一列一条线（列 = 初值，行 = 时间步）。 */
function trajectorySeries(
  frame: Frame,
  nTraj: number | undefined,
  degraded: string[],
  pal: ReturnType<typeof palette>,
): TrajSeries | undefined {
  if (!frame.has("traj_x") || !frame.has("traj_y")) {
    degraded.push("帧缺 traj_x/traj_y：不画轨迹（不猜）");
    return undefined;
  }
  let mx: { rows: number; cols: number };
  let my: { rows: number; cols: number };
  let mxc: ArrayLike<number>;
  let myc: ArrayLike<number>;
  try {
    const rawX = frame.matrix("traj_x");
    const rawY = frame.matrix("traj_y");
    mx = { rows: rawX.rows, cols: rawX.cols };
    my = { rows: rawY.rows, cols: rawY.cols };
    mxc = toNumeric(rawX.values);
    myc = toNumeric(rawY.values);
  } catch {
    degraded.push("traj_x/traj_y 不是 2 维（行=时间步、列=初值）：不画轨迹（不猜形状）");
    return undefined;
  }
  if (mx.rows !== my.rows || mx.cols !== my.cols) {
    degraded.push(`traj_x 形状 ${mx.rows}×${mx.cols} 与 traj_y 形状 ${my.rows}×${my.cols} 不一致：不画轨迹`);
    return undefined;
  }
  if (nTraj !== undefined && nTraj !== mx.cols) {
    degraded.push(`帧 meta.n_traj=${formatNumber(nTraj, 4)} 与轨迹矩阵列数 ${mx.cols} 不一致：按矩阵画并标出`);
  }
  const layers: LineSeries[] = [];
  for (let c = 0; c < mx.cols; c += 1) {
    const xs = new Float64Array(mx.rows);
    const ys = new Float64Array(my.rows);
    for (let r = 0; r < mx.rows; r += 1) {
      xs[r] = mxc[r * mx.cols + c] as number;
      ys[r] = myc[r * my.cols + c] as number;
    }
    layers.push(
      ...segmentedSeries(xs, ys, pal.curves[c % pal.curves.length] as string, {
        width: 1.1,
        alpha: 0.9,
        label: "数值轨迹",
      }),
    );
  }
  return { layers };
}
