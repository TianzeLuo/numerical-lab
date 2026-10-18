/**
 * `grid_independence`（4.5 章）的 model：跨 h 的收敛因子扫描 + 迭代数 + 代价账。
 *
 * 面板与图层（顺序即绘制顺序）
 * ----------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | line | `factors` | ρ = 1 参考线（固定窗口 0–1.05 的"不收缩"边界） |
 * | 2 | band | `factors` | MG 因子的 均值 ± 标准差 带（"网格无关 ≠ 常数"的直接证据） |
 * | 3 | line | `factors` | `residual_factor_two_grid`（按 `*_defined` 掩码，两网格因子恒 1/9） |
 * | 4 | line | `factors` | `error_factor_smoother`（单层平滑子，随 n 趋近 1） |
 * | 5 | line | `factors` | `residual_factor_mg`（当前 V/W 循环实测因子） |
 * | 6 | scatter | `factors` | 每个尺寸上的 MG 因子 |
 * | 7 | line | `iterations` | `iters_smoother`（**换算值**，虚线）/ `iters_mg_measured`（实测周期数） |
 * | 8 | line | `iterations` | `cg_iters` / `mgcg_iters`（同一右端的两种方法） |
 * | 9 | scatter | `iterations` | 每个尺寸上的实测点；触顶（`iters_at_cap`）另标 |
 * | 10 | line | `cost` | `work_mg_per_cycle` / `work_to_tol_mg` / `work_to_tol_smoother` |
 *
 * 四条刻意的决定
 * --------------
 * 1. **因子轴固定 0–1.05**（manifest `autoscaled: false`）：这一页最容易骗人的地方是"因子看着
 *    一直贴着 0 = 收敛很快"。固定窗口让 1.0 这条线始终在画面上，也让 0.16→0.19 的**缓慢变大**
 *    看得见；自适应会把它拉平成一条水平线。
 * 2. **均值 ± 标准差画成 `band`**：第 1 个陷阱（"网格无关 ≠ 常数"）不能只靠一行字。
 *    带子有多宽、是否随时间（这里随 n）整体上移，是一眼能看出来的东西。
 * 3. **`work_*` 单独一条副轴（log）**：第 2 个陷阱（"W-cycle 因子更小所以更好"）只有在
 *    "因子"与"代价"同屏时才挡得住。切到 `cycle=w` 会看到因子更接近 1/9、而代价整条抬上去。
 * 4. **换算值必须当场标出来**：`iters_smoother` 由实测因子换算（降到 1e-8 要 1e4~1e5 次扫描，
 *    帧里没真跑），画成虚线并在文本里点名；`iters_mg_measured` / `cg_iters` / `mgcg_iters`
 *    是真的数出来的。把两者画成一样的线是最容易犯的偷懒。
 *
 * 时间轴：`mode: "none"` —— **不画时间轴、不引用 tMax**（`scene.t` 恒为 0）。
 */

import { formatNumber } from "../../shared/src/axes.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import {
  applyOverflow,
  canvasSize,
  clipBadgeText,
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
  stringOf,
  type SceneModel,
} from "../../shared/src/model.js";
import {
  axisModeBadge,
  badge,
  bandLayer,
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
  legendBlock,
  logRange,
  maskIndicesWhere,
  panel45Layout,
  pick,
  textBlock,
  unitBudget,
} from "./_panel.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "4.5";
const VIZ = "grid_independence";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "收敛因子 vs 网格尺寸：跨 h 的实测与单层迭代的对比";

const SLOT = { source: 0, axisMode: 1, auxAxis: 2, status: 3, preview: 4, overflow: 5, degraded: 7 } as const;

/** 循环类型 / 平滑子的显示名（与 `server/chapters/4.5.py` 同口径）。 */
const CYCLES = ["v", "w"] as const;
const SMOOTHERS = ["jacobi", "gs", "sor"] as const;

export const gridIndependenceModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需数组 ——
  const nSeq = requireArray(frame, "n_seq");
  const levelSeq = requireArray(frame, "level_seq");
  const fMg = requireArray(frame, "residual_factor_mg");
  const fTg = requireArray(frame, "residual_factor_two_grid");
  const fSm = requireArray(frame, "error_factor_smoother");
  const itMg = requireArray(frame, "iters_mg_measured");
  const itSm = requireArray(frame, "iters_smoother");
  const cgIt = requireArray(frame, "cg_iters");
  const mgcgIt = requireArray(frame, "mgcg_iters");
  const workCycle = requireArray(frame, "work_mg_per_cycle");
  const workTolMg = requireArray(frame, "work_to_tol_mg");
  const workTolSm = requireArray(frame, "work_to_tol_smoother");
  for (const [name, arr] of [
    ["residual_factor_mg", fMg],
    ["residual_factor_two_grid", fTg],
    ["error_factor_smoother", fSm],
    ["iters_mg_measured", itMg],
    ["iters_smoother", itSm],
    ["cg_iters", cgIt],
    ["mgcg_iters", mgcgIt],
    ["work_mg_per_cycle", workCycle],
    ["work_to_tol_mg", workTolMg],
    ["work_to_tol_smoother", workTolSm],
  ] as const) {
    requireSameLength(nSeq, arr, `帧数组 n_seq 与 ${name}`);
  }
  const pitfalls = requirePitfalls(frame);

  // —— 可选数组 ——
  const degraded: string[] = [];
  const tgDefined = frame.has("residual_factor_two_grid_defined") ? requireArray(frame, "residual_factor_two_grid_defined") : undefined;
  if (tgDefined === undefined) {
    degraded.push("帧缺 residual_factor_two_grid_defined：两网格曲线不画（未测的位置填的是 0.0，画出来会被读成「完美收敛」）");
  }
  const atCap = frame.has("iters_at_cap") ? requireArray(frame, "iters_at_cap") : undefined;
  if (atCap === undefined) degraded.push("帧缺 iters_at_cap：不标「迭代数被上限截断」的尺寸（不猜哪些触了顶）");

  // —— meta ——
  const levelsMax = metaNumber(frame, "levels_max");
  const nu = metaNumber(frame, "nu");
  const cycleIdx = metaNumber(frame, "cycle_index");
  const smootherIdx = metaNumber(frame, "smoother_index");
  const omega = metaNumber(frame, "omega");
  const tol = metaNumber(frame, "tol");
  const mean = metaNumber(frame, "mg_factor_mean");
  const std = metaNumber(frame, "mg_factor_std");
  const drift = metaNumber(frame, "mg_factor_drift");
  const spread = metaNumber(frame, "mg_factor_relative_spread");
  const fMax = metaNumber(frame, "mg_factor_max");
  const fMin = metaNumber(frame, "mg_factor_min");
  const tgSpread = metaNumber(frame, "two_grid_factor_spread");
  const slope = metaNumber(frame, "smoother_slope_h2");
  if (slope === undefined) degraded.push("帧 meta 缺 smoother_slope_h2：不显示斜率（不自己拟合一遍）");
  const itSmMax = metaNumber(frame, "iters_smoother_max");
  const itMgMax = metaNumber(frame, "iters_mg_max");
  const cgMax = metaNumber(frame, "cg_iters_max");
  const mgcgMax = metaNumber(frame, "mgcg_iters_max");
  const speedupMax = metaNumber(frame, "speedup_max");
  const workMgMax = metaNumber(frame, "work_to_tol_mg_max");
  const workSmMax = metaNumber(frame, "work_to_tol_smoother_max");
  const nCapped = metaNumber(frame, "n_iters_capped");
  const itersCap = metaNumber(frame, "iters_cap");
  const converging = metaNumber(frame, "converging");
  const statusKey = stringOf(frame, "status");
  // 帧自报的几条 note：右列正文空间有限（每行 41 个显示宽度单位），长 note 不复述，
  // 只在这里检查"该有的字符串在不在"，缺了就如实降级。
  const curveNote = stringOf(frame, "curve_note");
  if (curveNote === undefined) degraded.push("帧缺 strings.curve_note：换算值与实测值的区别只在文本里说明");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: 0, // `time.mode = "none"`：没有时间轴，恒为 0（不引用 tMax）
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  const L = panel45Layout(scene, width, height, { topShare: 0.62, auxH: 88, gap: 58 });

  // —— 主轴：收敛因子（固定窗口 0–1.05）——
  const factors = makeAxis("factors", [L.left, L.topY, L.leftW, L.topH], {
    xlim: [nMin(nSeq), nMax(nSeq)],
    ylim: [0, 1.05],
    xscale: "log",
    xticks: Array.from(nSeq),
    xlabel: "n = 2^L − 1（对数横轴）",
    title: axisTitleOf(SPEC?.axes, "factors", "收敛因子（固定窗口 0-1.05，对数横轴）"),
    ylabel: axisYlabelOf(SPEC?.axes, "factors", "rho"),
    autoscaled: false,
  });
  scene.layers.push(
    lineLayer("factors", series(factors.xlim, [1, 1], pal.warn, { dash: [5, 4], width: 1.1})),
  );
  if (mean !== undefined && std !== undefined) {
    const ylo = new Float64Array(nSeq.length);
    const yhi = new Float64Array(nSeq.length);
    ylo.fill(mean - std);
    yhi.fill(mean + std);
    scene.layers.push(
      bandLayer("factors", {
        x: nSeq,
        ylo,
        yhi,
        color: pal.curves[0],
        alpha: 0.18,
        label: `均值 ${formatNumber(mean, 4)} ± 标准差 ${formatNumber(std, 3)}`,
      }),
    );
  }
  // 两网格曲线：只在"测过"的位置画（未测写的是 0.0，画出来等于宣称"一步收敛到 0"）。
  let tgDrawn = 0;
  if (tgDefined !== undefined) {
    const idx = maskIndicesWhere(tgDefined, fTg, (v) => Number.isFinite(v));
    tgDrawn = idx.length;
    if (idx.length > 0) {
      scene.layers.push(
        lineLayer("factors", series(pick(nSeq, fTg, idx).x, pick(nSeq, fTg, idx).y, pal.curves[1], {
          dash: [6, 4],
          width: 1.6,
          marker: "s",
        })),
      );
    }
  }
  scene.layers.push(
    lineLayer("factors", series(nSeq, fSm, pal.reference, { dash: [2, 3], width: 1.3, marker: "x"})),
    lineLayer("factors", series(nSeq, fMg, pal.curves[0], { width: 2.4, marker: "o"})),
  );

  // —— 副图：降到目标精度所需次数（对数纵轴）——
  const iterations = makeAxis("iterations", [L.left, L.botY, L.leftW, L.botH], {
    xlim: [nMin(nSeq), nMax(nSeq)],
    // 对数纵轴**必须显式给范围**：`makeAxis` 的缺省 [1e-16, 1] 会把 7…10061 全挤在轴外
    // （那是"曲线消失、账上也没有"的经典翻车方式）。
    ylim: logRange(
      Math.min(posMin(itSm) ?? 1, posMin(itMg) ?? 1, posMin(cgIt) ?? 1, posMin(mgcgIt) ?? 1),
      Math.max(maxFinite(itSm) ?? 1, maxFinite(cgIt) ?? 1) * 1.2,
    ),
    xscale: "log",
    yscale: "log",
    xticks: Array.from(nSeq),
    xlabel: "n = 2^L − 1（对数横轴）",
    title: axisTitleOf(SPEC?.axes, "iterations", "降到目标精度所需次数（对数轴）"),
    ylabel: axisYlabelOf(SPEC?.axes, "iterations", "次数"),
    autoscaled: true,
  });
  scene.layers.push(
    lineLayer("iterations", series(nSeq, itSm, pal.reference, { dash: [2, 3], width: 1.3, marker: "x"})),
    lineLayer("iterations", series(nSeq, cgIt, pal.curves[1], { width: 1.6, marker: "s"})),
    lineLayer("iterations", series(nSeq, mgcgIt, pal.exact, { width: 1.8, marker: "+"})),
    lineLayer("iterations", series(nSeq, itMg, pal.curves[0], { width: 2.4, marker: "o"})),
  );
  if (atCap !== undefined) {
    const idx = maskIndicesWhere(atCap, itSm, (v) => v > 0);
    if (idx.length > 0) {
      const pts = pick(nSeq, itSm, idx);
      scene.layers.push(
        scatterLayer("iterations", pts.x, pts.y, { r: 4, color: pal.warn, alpha: 0.95, label: "该尺寸的迭代数被上限截断（不是真的只要这么多次）" }),
      );
    }
  }

  // —— 第三条轴：代价账（log；因子更小 ≠ 更省）——
  const cost = makeAxis("cost", [L.rightX, L.topY, L.rightW, L.auxH], {
    xlim: [nMin(nSeq), nMax(nSeq)],
    ylim: logRange(
      Math.min(posMin(workCycle) ?? 1, posMin(workTolMg) ?? 1),
      Math.max(maxFinite(workTolSm) ?? 1, maxFinite(workTolMg) ?? 1) * 1.2,
    ),
    xscale: "log",
    yscale: "log",
    xticks: Array.from(nSeq),
    xlabel: "n（对数横轴）",
    title: "代价（最细层等价单位）",
    ylabel: "work",
    autoscaled: true,
  });
  scene.axes.push(factors, iterations, cost);
  scene.layers.push(
    lineLayer("cost", series(nSeq, workTolSm, pal.reference, { dash: [2, 3], width: 1.3, marker: "x"})),
    lineLayer("cost", series(nSeq, workCycle, pal.curves[1], { width: 1.6, marker: "s"})),
    lineLayer("cost", series(nSeq, workTolMg, pal.curves[0], { width: 2.2, marker: "o"})),
  );
  scene.layers.push(
    scatterLayer("factors", nSeq, fMg, { r: 2.6, color: pal.marker, alpha: 0.9, label: "每个尺寸上的 MG 因子" }),
    scatterLayer("iterations", nSeq, itMg, { r: 2.6, color: pal.marker, alpha: 0.9 }),
  );

  // —— 文本 ——
  const cycleName = cycleIdx !== undefined ? CYCLES[Math.round(cycleIdx)] ?? "?" : "?";
  const smootherName = smootherIdx !== undefined ? SMOOTHERS[Math.round(smootherIdx)] ?? "?" : "?";
  if (L.columns) {
    const budget = unitBudget(L.rightW);
    const head = paramSummary(frame, opts, [
      { name: "levels_max", label: "L_max" },
      { name: "nu", label: "ν" },
    ]);
    degraded.push(...head.degraded);
    const fmtList = (a: ArrayLike<number>, digits = 4): string =>
      Array.from({ length: a.length }, (_, i) => formatNumber(a[i] ?? Number.NaN, digits)).join(" / ");
    // 每行都**实测**过宽度（`textWidthUnits` ≤ 41 = 右列预算）：渲染层不会再折行，
    // 因此 1 个事实 = 1 行，行数正好等于可用行数（放不下的会带"另有 N 行未画"标注）。
    const lines: string[] = [];
    lines.push(
      `n=${formatNumber(nMin(nSeq), 3)}…${formatNumber(nMax(nSeq), 3)}` +
        `　L_max=${levelsMax !== undefined ? formatNumber(levelsMax, 3) : "?"}` +
        (levelSeq.length > 0
          ? `（L=${formatNumber(levelSeq[0] ?? Number.NaN, 3)}…${formatNumber(levelSeq[levelSeq.length - 1] ?? Number.NaN, 3)}）`
          : "") +
        `　ν=${nu !== undefined ? formatNumber(nu, 3) : "?"}`,
    );
    lines.push(
      `${cycleName}-cycle　${smootherName}` +
        (tol !== undefined ? `　tol=${formatNumber(tol, 2)}` : "") +
        (omega !== undefined ? `　ω=${formatNumber(omega, 4)}` : ""),
    );
    lines.push(`因子 ${fmtList(fMg, 4).replace(/ \/ /g, "/")}`);
    if (mean !== undefined && std !== undefined && drift !== undefined) {
      lines.push(`均值 ${formatNumber(mean, 4)}　std ${formatNumber(std, 4)}　漂移 ${drift >= 0 ? "+" : ""}${formatNumber(drift, 4)}`);
    }
    if (fMin !== undefined && fMax !== undefined) {
      lines.push(`范围 ${formatNumber(fMin, 4)}…${formatNumber(fMax, 4)}，随层数变大`);
    }
    if (spread !== undefined) lines.push(`相对散布 ${formatNumber(spread, 4)}：无关 ≠ 常数`);
    if (tgDefined !== undefined && tgSpread !== undefined) {
      lines.push(`两网格恒 ${formatNumber(fTg[0] ?? Number.NaN, 4)}（${tgDrawn}/${fTg.length}，散布 ${formatNumber(tgSpread, 3)}）`);
      lines.push(`粗层精确解时才真的与 n 无关`);
    }
    if (slope !== undefined) {
      lines.push(`平滑子 ${formatNumber(fSm[0] ?? Number.NaN, 4)}→${formatNumber(fSm[fSm.length - 1] ?? Number.NaN, 4)}，斜率 ${formatNumber(slope, 4)}`);
    }
    if (itMgMax !== undefined && itSmMax !== undefined) {
      lines.push(`迭代数 MG ${formatNumber(itMgMax, 3)} 周期；平滑子换算 ${formatNumber(itSm[0] ?? Number.NaN, 3)}→${formatNumber(itSmMax, 3)}`);
    }
    if (cgMax !== undefined && mgcgMax !== undefined) {
      lines.push(
        `CG ${formatNumber(cgIt[0] ?? Number.NaN, 3)}→${formatNumber(cgMax, 3)}` +
          `　MGCG ${formatNumber(mgcgIt[0] ?? Number.NaN, 3)}→${formatNumber(mgcgMax, 3)}` +
          (speedupMax !== undefined ? `　加速 ${formatNumber(speedupMax, 4)}×` : ""),
      );
    }
    if (workMgMax !== undefined) {
      lines.push(
        `代价/周期 ${formatNumber(workCycle[0] ?? Number.NaN, 4)}→${formatNumber(workCycle[workCycle.length - 1] ?? Number.NaN, 4)}` +
          `；总代价 ${formatNumber(workTolMg[0] ?? Number.NaN, 4)}→${formatNumber(workMgMax, 4)}`,
      );
    }
    if (workSmMax !== undefined) {
      lines.push(`平滑子总代价 ${formatNumber(workTolSm[0] ?? Number.NaN, 3)}→${formatNumber(workSmMax, 3)}（换算值）`);
    }
    lines.push(`W 因子更小 ≠ 更省：每周期贵得多`);
    if (nCapped !== undefined && nCapped > 0) {
      lines.push(`⚠ ${formatNumber(nCapped, 3)} 个尺寸触到迭代上限 ${formatNumber(itersCap ?? 0, 3)}`);
    }
    scene.texts.push(...textBlock(L.rightX, L.textTop, L.floor, lines, pal.muted, budget, "top"));
  } else {
    degraded.push("窄画布（单列）：右列说明未排下，关键数字见角标与 pitfalls 面板");
  }

  scene.texts.push(
    ...legendBlock(L.left + 6, L.topY + 6, L.leftW, "蓝实=MG 实测因子　蓝带=均值±标准差　绿虚=两网格 1/9　灰点=单层平滑子　红虚=ρ=1", pal.muted),
    ...legendBlock(L.left + 6, L.botY + 6, L.leftW, "灰虚=平滑子换算　绿=CG　橙=MGCG　蓝=MG 实测周期数", pal.muted),
    ...legendBlock(L.rightX + 6, L.topY + 6, L.rightW, "绿=每周期　蓝=总代价　灰虚=单层平滑子总代价", pal.muted),
  );

  // —— 角标 ——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  if (scene.dataSource.startsWith("backend+preview")) {
    scene.badges.push(placeBadge(previewBadge("不适用：本页无时间轴"), SLOT.preview, width));
  }
  scene.badges.push(placeBadge(axisModeBadge(true), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(badge("autoscale", `因子轴固定 0–1.05；converging=${converging !== undefined && converging > 0.5 ? "true" : "false"}`, 8, 26), SLOT.auxAxis, width),
  );
  const status: string[] = [];
  if (mean !== undefined && std !== undefined) status.push(`均值 ${formatNumber(mean, 4)}　std ${formatNumber(std, 3)}`);
  if (spread !== undefined) status.push(`散布 ${formatNumber(spread, 3)}`);
  if (statusKey !== undefined) status.push(statusKey.slice(0, 4));
  scene.badges.push(placeBadge(badge("status", clipBadgeText(status.join("；")), 8, 8), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  const nBad = countNonFinite(fMg) + countNonFinite(fTg) + countNonFinite(itSm) + countNonFinite(workCycle) + countNonFinite(nSeq);
  applyOverflow(scene, nBad, { slot: SLOT.overflow, width });
  return scene;
};

// --------------------------------------------------------------------------- //
// 局部工具（只做显示几何，不算物理量）
// --------------------------------------------------------------------------- //

/** 有限正值的最小值（对数横轴的定义域）；没有就退回 1。 */
function nMin(a: ArrayLike<number>): number {
  let lo = Number.POSITIVE_INFINITY;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i];
    if (Number.isFinite(v) && v > 0 && v < lo) lo = v;
  }
  return Number.isFinite(lo) ? lo : 1;
}

/** 有限正值的最大值；没有就退回 10。 */
function nMax(a: ArrayLike<number>): number {
  let hi = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i];
    if (Number.isFinite(v) && v > 0 && v > hi) hi = v;
  }
  return Number.isFinite(hi) ? hi : 10;
}

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

