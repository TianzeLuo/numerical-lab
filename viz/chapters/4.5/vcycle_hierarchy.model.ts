/**
 * `vcycle_hierarchy`（4.5 章）的 model：层级图 + `mg_apply(trace=...)` 的**真实**事件序列。
 *
 * 面板与图层（顺序即绘制顺序）
 * ----------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | line | `profile` | 细网格误差 `error_profile`（实线） |
 * | 2 | line | `profile` | 各粗层的限制结果（`profile_flat` 按 `profile_offset/size` 切片，一层一条） |
 * | 3 | scatter | `profile` | 各层自己的极值点（同一层内的读数锚点） |
 * | 4 | line | `events` | 事件序列，**按 `event_level` 分段**：每层一条自己的折线 |
 * | 5 | scatter | `events` | 每个事件（离散点） |
 * | 6 | scatter | `events` | 已到地板的事件（`event_at_floor`） |
 * | 7 | line | `convergence` | `error_norm_history`（逐周期的**同层**误差范数，规则 3 的 error） |
 * | 8 | line | `convergence` | `residual_norm_history`（逐周期的**同层**残差范数，规则 3 的 residual） |
 * | 9 | scatter | `convergence` | 当前周期 |
 *
 * 四条刻意的决定
 * --------------
 * 1. **事件折线按层切段，绝不跨层连线**（`runsBy(event_level)`）。真实事件序列里相邻两层的
 *    范数可以差几个数量级（本帧细层 0.93 / 第 1 层 30.5），连起来画出的是一段**不存在的搬运**，
 *    而那正好是第 3 个陷阱（跨层比范数）想让人相信的东西。分段之后，"同一层的锯齿"才是口径。
 * 2. **`convergence` 副轴是"同层口径"的落点**：`error_norm_history` / `residual_norm_history`
 *    是**同一个最细层**逐周期的范数，跨周期可比；而主图里各层的向量范数不可比（点数、h、
 *    残差/误差的定义都不同）。想看"粗层误差更小"的错觉怎么来的，看主图；想知道收敛快不快，
 *    看这条副轴——这正是不让第 3 个陷阱得逞的关键。
 * 3. **`event_norm` 的 1e-16 是占位**：进入粗层时该层向量初值为 0，帧按地板写成 1e-16 并置
 *    `event_at_floor`。它不是"残差已经很小"，图上单独用一种颜色标出来，文本里点明。
 * 4. **粗层横坐标按该层自己的 h 还原**（帧给 `level_h`/`level_n`，`x_i = i·h_l`）：
 *    这是显示坐标，不是插值——把 7 个点铺到 [0,1] 上是为了与细层对齐着看形状。
 *
 * 时间轴：`scrub`，`t = 周期 c`（做 c 个完整循环；c=0 即初值误差）。
 */

import { formatNumber } from "../../shared/src/axes.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import {
  applyOverflow,
  canvasSize,
  clamp,
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
  maskIndicesWhere,
  panel45Layout,
  runsBy,
  selectValues,
  sliceRun,
  textBlock,
  unitBudget,
} from "./_panel.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "4.5";
const VIZ = "vcycle_hierarchy";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "V-cycle 层级图：误差在粗细网格间的转移与校正";
const T_MAX = SPEC?.time.mode === "none" ? 0 : SPEC?.time.tMax ?? 5;

const SLOT = { source: 0, axisMode: 1, auxAxis: 2, status: 3, preview: 4, overflow: 5, degraded: 7 } as const;

/** 平滑子 / 循环类型 / 初值的显示名（与 `server/chapters/4.5.py` 同口径）。 */
const SMOOTHERS = ["jacobi", "gs", "sor"] as const;
const CYCLES = ["v", "w"] as const;
const CONTENTS = ["low", "high", "mixed"] as const;

export const vcycleHierarchyModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需数组 ——
  const fineX = requireArray(frame, "fine_x");
  const errProfile = requireArray(frame, "error_profile");
  const flat = requireArray(frame, "profile_flat");
  const offset = requireArray(frame, "profile_offset");
  const size = requireArray(frame, "profile_size");
  const levelN = requireArray(frame, "level_n");
  const levelH = requireArray(frame, "level_h");
  const evOrder = requireArray(frame, "event_order");
  const evLevel = requireArray(frame, "event_level");
  const evNorm = requireArray(frame, "event_norm");
  const errHist = requireArray(frame, "error_norm_history");
  const resHist = requireArray(frame, "residual_norm_history");
  requireSameLength(fineX, errProfile, "帧数组 fine_x 与 error_profile");
  requireSameLength(evOrder, evLevel, "帧数组 event_order 与 event_level");
  requireSameLength(evOrder, evNorm, "帧数组 event_order 与 event_norm");
  if (frame.has("event_phase")) requireArray(frame, "event_phase"); // 相位编码：数值在 strings.phase_legend 里解释
  const pitfalls = requirePitfalls(frame);

  // —— 可选数组 ——
  const degraded: string[] = [];
  const evSize = frame.has("event_size") ? requireArray(frame, "event_size") : undefined;
  const evFloor = frame.has("event_at_floor") ? requireArray(frame, "event_at_floor") : undefined;
  if (evFloor === undefined) degraded.push("帧缺 event_at_floor：不标「进入粗层时的 0 → 地板」事件（不猜）");
  const errFloor = frame.has("error_at_floor") ? requireArray(frame, "error_at_floor") : undefined;
  if (errFloor === undefined) degraded.push("帧缺 error_at_floor：不报「多少点已到地板」");

  // —— meta ——
  const cycles = metaNumber(frame, "cycles");
  const tRequested = metaNumber(frame, "t_requested");
  if (tRequested !== undefined && cycles !== undefined && Math.abs(tRequested - cycles) > 1e-9) {
    degraded.push(`请求 t=${formatNumber(tRequested, 4)} 被夹到周期 ${cycles}（帧 meta.cycles 是实到周期）`);
  }
  const t = cycles ?? clamp(Math.round(opts.t), 0, T_MAX);
  const levels = metaNumber(frame, "levels");
  const nu = metaNumber(frame, "nu");
  const cycleIdx = metaNumber(frame, "cycle_index");
  const smootherIdx = metaNumber(frame, "smoother_index");
  const contentIdx = metaNumber(frame, "content_index");
  const nEvents = metaNumber(frame, "n_events");
  const factor = metaNumber(frame, "factor_measured");
  if (factor === undefined) degraded.push("帧 meta 缺 factor_measured：不显示收敛因子（不自己估）");
  const converging = metaNumber(frame, "factor_converging");
  const cyclesToTol = metaNumber(frame, "cycles_to_1e8");
  const workPerCycle = metaNumber(frame, "work_per_cycle");
  const workFinest = metaNumber(frame, "work_finest_level");
  const errNow = metaNumber(frame, "error_norm_now");
  const errRatioLast = metaNumber(frame, "error_norm_ratio_last");
  const floor = metaNumber(frame, "floor");
  const phaseLegend = stringOf(frame, "phase_legend");
  // 帧自报的 `trace_note` / `factor_note` 都是长句：右列正文放不下，不复述；
  // 相位图例（`strings.phase_legend`）已经在面板里逐行解释了 trace 的语义。
  const phaseLegendKey = stringOf(frame, "phase_legend");
  if (phaseLegendKey === undefined) degraded.push("帧缺 strings.phase_legend：事件相位只能用数字读");
  const statusKey = stringOf(frame, "status");
  const timeSemantics = stringOf(frame, "time_semantics");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  const L = panel45Layout(scene, width, height, { topShare: 0.6, auxH: 88, gap: 58 });

  // —— 主图：误差形态（细层 + 各粗层，各按自己的 h 还原横坐标）——
  const profile = makeAxis("profile", [L.left, L.topY, L.leftW, L.topH], {
    xlim: [0, 1],
    xticks: [0, 0.25, 0.5, 0.75, 1],
    xlabel: "x（内点位置；粗层按其 h 还原）",
    title: axisTitleOf(SPEC?.axes, "profile", "误差形态（细网格与各层的限制结果）"),
    ylabel: axisYlabelOf(SPEC?.axes, "profile", "e(x)"),
    autoscaled: true,
  });
  scene.layers.push(
    lineLayer("profile", series(fineX, errProfile, pal.curves[0], { width: 2.4})),
  );
  const levelColors = [pal.curves[0], pal.exact, pal.curves[1], pal.residual, pal.curves[2]];
  const nLevels = Math.max(levelN.length, levelH.length, offset.length, size.length);
  const perLevel: string[] = [];
  let yLo = minFinite(errProfile) ?? -1;
  let yHi = maxFinite(errProfile) ?? 1;
  for (let l = 1; l < nLevels; l += 1) {
    const start = Math.round(offset[l] ?? Number.NaN);
    const len = Math.round(size[l] ?? Number.NaN);
    const nl = Math.round(levelN[l] ?? Number.NaN);
    const hl = levelH[l];
    if (!Number.isFinite(start) || !Number.isFinite(len) || !Number.isFinite(nl) || !Number.isFinite(hl)) {
      degraded.push(`帧的 profile_offset/size 第 ${l} 层不可用：该层不画（不按下标猜切片）`);
      continue;
    }
    const xs = new Float64Array(len);
    for (let i = 0; i < len; i += 1) xs[i] = (i + 1) * hl;
    const ys = new Float64Array(len);
    for (let i = 0; i < len; i += 1) ys[i] = flat[start + i];
    const lo = minFinite(ys);
    const hi = maxFinite(ys);
    if (lo !== undefined) yLo = Math.min(yLo, lo);
    if (hi !== undefined) yHi = Math.max(yHi, hi);
    perLevel.push(`L${l}: n=${nl}`);
    scene.layers.push(
      lineLayer("profile", series(xs, ys, levelColors[l % levelColors.length] as string, {
        width: 1.6,
        dash: [5, 4],
        marker: "o",
      })),
    );
  }
  const pad = Math.max((yHi - yLo) * 0.12, 1e-12);
  profile.ylim = [yLo - pad, yHi + pad];

  // —— 副图：真实事件序列（按层分段）——
  const events = makeAxis("events", [L.left, L.botY, L.leftW, L.botH], {
    xlim: [-0.5, Math.max(0.5, evOrder.length - 0.5)],
    ylim: logRange(minPositive(evNorm) ?? 1e-16, Math.max(maxFinite(evNorm) ?? 1, 1) * 1.2),
    yscale: "log",
    xticks: [0, Math.round((evOrder.length - 1) / 2), Math.max(0, evOrder.length - 1)].filter((v, i, a) => a.indexOf(v) === i),
    xlabel: "事件序号（下一个周期的真实搬运顺序）",
    title: axisTitleOf(SPEC?.axes, "events", "下一个周期的事件序列（该层向量范数）"),
    ylabel: axisYlabelOf(SPEC?.axes, "events", "‖v_l‖"),
    autoscaled: true,
  });
  const runs = runsBy(evLevel);
  for (const run of runs) {
    const seg = sliceRun(evOrder, evNorm, run);
    const lv = Math.round(evLevel[run.start] ?? 0);
    scene.layers.push(
      lineLayer("events", series(seg.x, seg.y, levelColors[lv % levelColors.length] as string, {
        width: 1.8,
      })),
    );
  };
  scene.layers.push(
    scatterLayer("events", evOrder, evNorm, { r: 2.2, color: pal.muted, alpha: 0.85, label: "事件点" }),
  );
  if (evFloor !== undefined) {
    const idx = maskIndicesWhere(evFloor, evNorm, (v) => v > 0);
    if (idx.length > 0) {
      const xs = selectValues(evOrder, idx);
      const ys = selectValues(evNorm, idx);
      scene.layers.push(
        scatterLayer("events", xs, ys, { r: 4, color: pal.warn, alpha: 0.95, label: "进入粗层：向量初值为 0（地板占位，不是残差小）" }),
      );
    }
  }

  // —— 第三条轴（规则 3 的落点）：同层口径的逐周期范数 ——
  const conv = makeAxis("convergence", [L.rightX, L.topY, L.rightW, L.auxH], {
    xlim: [0, Math.max(1, errHist.length - 1)],
    ylim: logRange(
      Math.min(minPositive(errHist) ?? 1, minPositive(resHist) ?? 1),
      Math.max(maxFinite(errHist) ?? 1, maxFinite(resHist) ?? 1) * 1.2,
    ),
    yscale: "log",
    xticks: Array.from(new Set([0, Math.max(0, Math.round((errHist.length - 1) / 2)), Math.max(0, errHist.length - 1)])),
    xlabel: "周期 c",
    title: "同一层的逐周期范数（跨周期可比）",
    ylabel: "比值 / 初值",
    autoscaled: true,
  });
  scene.axes.push(profile, events, conv);
  const histX = new Float64Array(errHist.length);
  for (let i = 0; i < errHist.length; i += 1) histX[i] = i;
  if (errHist.length > 1) {
    scene.layers.push(lineLayer("convergence", series(histX, errHist, pal.curves[0], { width: 2, marker: "o"})));
  }
  if (resHist.length > 1) {
    scene.layers.push(
      lineLayer("convergence", series(histX, resHist, pal.residual, { width: 1.6, dash: [5, 4]})),
    );
  }
  if (errHist.length > 0) {
    const last = errHist.length - 1;
    scene.layers.push(scatterLayer("convergence", [histX[last]], [errHist[last]], { r: 4, color: pal.marker, label: `当前 c=${last}` }));
  }

  // —— 事件账（按 `event_level` 逐条数出来，不是估的）——
  const counts = new Array<number>(nLevels).fill(0);
  const points = new Array<number>(nLevels).fill(0);
  for (let i = 0; i < evLevel.length; i += 1) {
    const lv = Math.round(evLevel[i]);
    if (lv < 0 || lv >= nLevels) continue;
    counts[lv] = (counts[lv] ?? 0) + 1;
    points[lv] = (points[lv] ?? 0) + (evSize !== undefined && Number.isFinite(evSize[i]) ? evSize[i] : 0);
  }
  // 一行放不下 4 层的事件账：拆成两行（每行都实测 ≤ 41 个显示宽度单位）。
  const accItem = (l: number): string =>
    `L${l} ${counts[l] ?? 0}/${formatNumber(points[l] ?? 0, 4)}点`;
  const half = Math.ceil(counts.length / 2);
  const accountA = counts.slice(0, half).map((_, i) => accItem(i)).join("　");
  const accountB = counts.slice(half).map((_, i) => accItem(i + half)).join("　");

  // —— 文本 ——
  const smootherName = smootherIdx !== undefined ? SMOOTHERS[Math.round(smootherIdx)] ?? "?" : "?";
  const cycleName = cycleIdx !== undefined ? CYCLES[Math.round(cycleIdx)] ?? "?" : "?";
  const contentName = contentIdx !== undefined ? CONTENTS[Math.round(contentIdx)] ?? "?" : "?";
  if (L.columns) {
    const budget = unitBudget(L.rightW);
    const head = paramSummary(frame, opts, [{ name: "n_fine", label: "n" }]);
    degraded.push(...head.degraded);
    // 每行都**实测**过宽度（`textWidthUnits` ≤ 41 = 右列预算）：1 个事实 = 1 行，
    // 渲染层不会再折行；超出的行会被 `textBlock` 截断并标注"另有 N 行未画"。
    const lines: string[] = [];
    lines.push(
      `c=${cycles ?? t}/${T_MAX}　${head.text}` +
        (levels !== undefined ? `　L=${formatNumber(levels, 3)}` : "") +
        (nu !== undefined ? `　ν=${formatNumber(nu, 3)}` : "") +
        `　${cycleName}-cycle`,
    );
    lines.push(`${smootherName}　初值=${contentName}`);
    if (phaseLegend !== undefined) lines.push(`相位：0 进入/1 前平滑/2 限制/3 粗层解`);
    if (phaseLegend !== undefined) lines.push(`4 校正/5 后平滑/6 最粗层解完`);
    if (nEvents !== undefined) lines.push(`下一周期 ${formatNumber(nEvents, 3)} 个事件（真实 trace 顺序）`);
    if (factor !== undefined) {
      lines.push(`因子 ${formatNumber(factor, 4)}（收敛=${converging !== undefined && converging > 0.5 ? "是" : "否"}）`);
      if (cyclesToTol !== undefined) lines.push(`降到 1e-8 还要 ${formatNumber(cyclesToTol, 3)} 个周期：它不是 0`);
      lines.push(`固定递归的循环只收缩，不是精确求解器`);
    }
    if (errRatioLast !== undefined) lines.push(`上周期误差比 ${formatNumber(errRatioLast, 3)}`);
    if (errNow !== undefined) lines.push(`当前误差范数 ${formatNumber(errNow, 3)}（帧 meta）`);
    if (workPerCycle !== undefined) {
      lines.push(
        `代价账 ${formatNumber(workPerCycle, 4)} 单位/周期` +
          (workFinest !== undefined ? `（最细层自身 ${formatNumber(workFinest, 3)}）` : ""),
      );
      lines.push(`粗层不是免费的：账要按整周期算`);
    }
    if (evSize !== undefined) lines.push(`事件账 ${accountA}`);
    if (evSize !== undefined) lines.push(`${accountB}（事件数/点数）`);
    if (timeSemantics !== undefined) lines.push(`时间轴：c 是完整循环数，不是物理时间`);
    if (errFloor !== undefined && countMask(errFloor) > 0) {
      lines.push(`主图 ${formatNumber(countMask(errFloor), 4)} 个点已到地板 ${formatNumber(floor ?? 1e-16, 3)}`);
    }
    scene.texts.push(...textBlock(L.rightX, L.textTop, L.floor, lines, pal.muted, budget, "top"));
  } else {
    degraded.push("窄画布（单列）：右列说明未排下，关键数字见角标与 pitfalls 面板");
  }

  // —— 面板内标注 ——
  scene.texts.push(
    ...legendBlock(L.left + 6, L.topY + 6, L.leftW, `实线=细层 e(x)　虚线=各粗层限制结果（${perLevel.join("　")}）`, pal.muted),
    ...legendBlock(L.left + 6, L.botY + 6, L.leftW, "每个事件一个点；同色为同一层，层与层之间不连线（跨层比范数是错觉）", pal.muted),
  );

  // —— 角标 ——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  if (scene.dataSource.startsWith("backend+preview")) {
    scene.badges.push(placeBadge(previewBadge("拖动时只重取帧"), SLOT.preview, width));
  }
  scene.badges.push(placeBadge(axisModeBadge(true), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(
      badge("autoscale", "事件轴（log）与形态轴都自适应", 8, 26),
      SLOT.auxAxis,
      width,
    ),
  );
  const status: string[] = [];
  if (factor !== undefined) status.push(`因子 ${formatNumber(factor, 4)}≠0`);
  if (cyclesToTol !== undefined) status.push(`；降到 1e-8 要 ${formatNumber(cyclesToTol, 3)} 周期`);
  if (statusKey !== undefined) status.push(`；${statusKey.slice(0, 4)}`);
  if (status.length === 0) status.push("帧未给出可判读的状态（预期有 meta.factor_measured 与 strings.status）");
  scene.badges.push(placeBadge(badge("status", clipBadgeText(status.join("；")), 8, 8), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  const nBad = countNonFinite(errProfile) + countNonFinite(evNorm) + countNonFinite(flat) + countNonFinite(errHist);
  applyOverflow(scene, nBad, { slot: SLOT.overflow, width });
  return scene;
};

// --------------------------------------------------------------------------- //
// 局部工具（只做显示几何，不算物理量）
// --------------------------------------------------------------------------- //

/** 有限正值的最小值；没有就 undefined。 */
function minPositive(a: ArrayLike<number>): number | undefined {
  let lo = Number.POSITIVE_INFINITY;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i];
    if (Number.isFinite(v) && v > 0 && v < lo) lo = v;
  }
  return Number.isFinite(lo) ? lo : undefined;
}

/** 有限最小值；没有就 undefined。 */
function minFinite(a: ArrayLike<number>): number | undefined {
  let lo = Number.POSITIVE_INFINITY;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i];
    if (Number.isFinite(v) && v < lo) lo = v;
  }
  return Number.isFinite(lo) ? lo : undefined;
}

/** 有限最大值；没有就 undefined。 */
function maxFinite(a: ArrayLike<number>): number | undefined {
  let hi = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i];
    if (Number.isFinite(v) && v > hi) hi = v;
  }
  return Number.isFinite(hi) ? hi : undefined;
}
