/**
 * `float_zoom`（1.1 章）的 model：两条浮点梯子 + 0.1 累加 k 次的残差。
 *
 * 面板与图层（顺序即绘制顺序，测试按此断言）
 * ------------------------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | line | `ladder` | 间距比 = 1 的参照线（"网格均匀"那条假想线） |
 * | 2 | line | `ladder` | **以最近的 2 的幂为中心**的梯子（虚线：跨过中心掉一半） |
 * | 3 | line | `ladder` | **以参数中心为中心**的梯子（实线：整个窗口在同一 binade，恒为 1） |
 * | 4 | line | `ladder` | 两条竖线：数学上的 0.3 与 0.1+0.2 的落点（各自中心的 ulp 偏移） |
 * | 5 | scatter | `ladder` | 两个落点（红 = 0.1+0.2，灰参照线标出数学上的 0.3） |
 * | 6 | line | `residual` | 当前 k 的游标（虚线竖线） |
 * | 7 | line | `residual` | `residual`：与**数学值 k/10** 的残差（实线） |
 * | 8 | line | `residual` | `residual_sum_only`：与 **k·fl(0.1)** 的残差（虚线） |
 * | 9 | line | `residual` | `residual_floor`：显示地板 u/2·max(|k/10|,0.1)（点线） |
 * | 10 | scatter | `residual` | 贴在地板上的采样点（圈）+ 当前 k 的标记（实心） |
 *
 * 角标槽位（固定，见 `SLOT`）：0 来源 / 1 主轴固定 / 2 副轴自适应 / 3 状态（note + 落点 +
 * `1e16+1`）/ 4 前端预览（仅 `backend+preview`）/ 5 溢出 / 7 降级说明。
 *
 * 几条刻意的决定
 * --------------
 * 1. **横轴必须是"距各自中心的 ulp 偏移"，不是真实坐标**：窗口里 21 个 double 用
 *    `%.17g` 打印出来都是 `0.3`（`strings.exact_center` 给了中心自己的 25 位展开），
 *    真实坐标会把整条梯子压成一个像素——`chapters/1.1/notes.tex` 的 Remark 说的就是这件事。
 *    两条梯子的横轴**各自相对自己的中心**，所以它们能共用一条轴：`x = 0` 只是"各自的中心"。
 * 2. **纵轴定死在 `[0.4, 1.15]`（`LADDER_YLIM`），不随帧自适应**：这条轴画的是
 *    "间距 / 该窗口中心的 ulp"，理论取值只有 `1`（同一 binade）与 `1/2`（跨过 2 的幂）
 *    两种。定死之后"掉一半"在任何参数下都是同一把尺子；若跟着数据走，
 *    中心恰是 2 的幂那一帧（两条梯子重合）会把 0.5 也拉到顶，"网格不均匀"这件事
 *    就被轴自己抹掉了——这正是规则 1/4 要防的。数据真超出这个范围时
 *    `applyOverflow` 会标出来，不会静默裁掉。
 * 3. **"贴地板"的点要画出来**：后端把"残差恰好为 0"与"低于地板"统一抬到
 *    `residual_floor`（`np.maximum(raw, floor)`）。到了像素上，"精确 0"与"= 地板值"
 *    完全同高，所以模型画两样东西：地板曲线本身 + 贴在地板上的那些点的空心圈。
 *    计数分两类如实报出：贴地板的个数（模型比两个数组比出来）与**恰好为 0** 的个数
 *    （帧 `meta.residual_exact_zero_steps`，后端用有理数算的，前端不重算）。
 * 4. **两个落点画在同一层**：`landing_y` / `ideal_y` 都是 1（帧给的），
 *    即"落在间距比为 1 的那条梯子上"；横坐标分别是 `landing_offset = 1` 与
 *    `ideal_offset = 0`。数学上的 0.3 正好是窗口中心、`fl(0.3)` 也是它，0.1+0.2 恰好在
 *    它上面 **1 个 ulp**——"差 1 个 ulp"在图上就是这两个点的间距（默认帧约 28px）。
 *    落点不在窗口内时帧给 `meta.landing_in_window = false`，模型不画点、只挂溢出说明。
 * 5. **说明文字一律不进绘图区**：宽画布（≥900px）在两块图右侧各开一条说明栏；
 *    窄画布只保留角标与轴标题（宁可少说，也不把字压在曲线上）。当前 k 写进状态角标，
 *    不再在图内漂浮 `k=` 标签。主图与副轴之间留 44px，让主图刻度带和副轴标题带互不重叠。
 */

import { countNonFinite } from "../../shared/src/geometry.js";
import { formatNumber, niceLimits, niceTicks } from "../../shared/src/axes.js";
import type { Frame } from "../../shared/src/frame.js";
import {
  applyOverflow,
  canvasSize,
  clamp,
  dataSourceOf,
  degradedBadge,
  finitePositiveMinMax,
  metaNumber,
  palette,
  paramSummary,
  placeBadge,
  requireArray,
  requireDrawableLayers,
  requirePitfalls,
  requireSameLength,
  sceneShell,
  segmentedSeries,
  stackedPanels,
  stringOf,
  type Rect,
  type SceneModel,
} from "../../shared/src/model.js";
import {
  axisModeBadge,
  badge,
  lineLayer,
  makeAxis,
  overflowBadge,
  pitfallsBand,
  previewBadge,
  scatterLayer,
  series,
  sourceBadge,
  type TextSpec,
} from "../../shared/src/scene.js";
import {
  TEXT_ROW_H,
  absMax,
  avoidTextOverlap,
  axisTitleOf,
  axisYlabelOf,
  floorIndices,
  pick,
  textBlock,
  unitBudget,
} from "./_panel.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "1.1";
const VIZ = "float_zoom";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "数轴放大镜：浮点网格与 0.1+0.2 的落点";

/** 固定角标槽位（见 `model.ts::BADGE_LAYOUT`）。 */
const SLOT = {
  source: 0,
  axisMode: 1,
  auxAxis: 2,
  status: 3,
  /** 仅当帧声明 `backend+preview` 时出现（本章的帧都是 backend）。 */
  preview: 4,
  overflow: 5,
  degraded: 7,
} as const;

/**
 * 梯子纵轴的**固定**范围（见文件头第 2 条）：理论取值只有 1 与 1/2，
 * 上下各留一点余量，让"掉一半"在任何一帧上都占同样的像素高度。
 */
const LADDER_YLIM: [number, number] = [0.4, 1.15];

/** 间距比的两种理论取值（定理给的两个数，不是前端算出来的物理量）。 */
const RATIO_UNIFORM = 1;
const RATIO_HALVED = 0.5;

/** 宽画布才开右侧说明栏；窄画布宁可少说，也不把文字压进曲线。 */
const TEXT_COLUMN_MIN_WIDTH = 900;
const TEXT_COLUMN_GAP = 26;
const TEXT_COLUMN_WIDTH = 300;
/** 两块面板之间要同时放下主图刻度带（约 18px）与副轴标题带（约 18px）。 */
const PANEL_GAP = 44;

interface FloatZoomLayout {
  main: Rect;
  aux: Rect;
  mainText?: Rect;
  auxText?: Rect;
}

/**
 * 上（梯子）下（残差）两块；宽画布各切一条右栏放说明。
 *
 * 为什么不用 `stackedPanels` 的默认 22px 间距：主图刻度从 `bottom+3` 画到 `+18`，
 * 副轴标题从 `top-18` 画到 `top-6`，22px 间距会让两段文字直接叠在一起（本图曾如此）。
 * 这里固定 44px，并把说明文字整体移出绘图区；窄画布退回满宽、不画说明栏。
 */
function floatZoomLayout(width: number, height: number, pitfalls: string): FloatZoomLayout {
  // 底部 pitfalls 面板由渲染器现画，窄画布会折成很多行；默认 102px 的 footer 不够，
  // 会把副轴图整个盖住。用共享的 `pitfallsBand` 量出真实高度，再留 36px 给副轴 xlabel。
  const band = pitfallsBand({ canvas: { w: width, h: height }, pitfalls });
  const footer = band === null ? 102 : height - band[1] + 36;
  const base = stackedPanels(width, height, { footer });
  const avail = base.bottom - base.top;
  const mainH = Math.round((avail - PANEL_GAP) * 0.59);
  const auxH = avail - PANEL_GAP - mainH;
  const twoCol = width >= TEXT_COLUMN_MIN_WIDTH;
  const sideW = twoCol ? TEXT_COLUMN_WIDTH : 0;
  const plotW = twoCol ? base.contentWidth - sideW - TEXT_COLUMN_GAP : base.contentWidth;
  const main: Rect = [base.left, base.top, plotW, mainH];
  const aux: Rect = [base.left, base.top + mainH + PANEL_GAP, plotW, auxH];
  if (!twoCol) return { main, aux };
  const textX = base.left + plotW + TEXT_COLUMN_GAP;
  const textW = base.left + base.contentWidth - textX;
  return {
    main,
    aux,
    mainText: [textX, main[1], textW, main[3]],
    auxText: [textX, aux[1], textW, aux[3]],
  };
}

export const floatZoomModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需数组：缺一个就抛 ModelInputError（宁可当场炸，也不画半张图）——
  const offset = requireArray(frame, "offset");
  const spacingRatio = requireArray(frame, "spacing_ratio");
  const boundaryOffset = requireArray(frame, "boundary_offset");
  const boundarySpacingRatio = requireArray(frame, "boundary_spacing_ratio");
  const kGrid = requireArray(frame, "k_grid");
  const residual = requireArray(frame, "residual");
  const residualSumOnly = requireArray(frame, "residual_sum_only");
  const residualFloor = requireArray(frame, "residual_floor");
  requireSameLength(offset, spacingRatio, "帧数组 offset 与 spacing_ratio");
  requireSameLength(boundaryOffset, boundarySpacingRatio, "帧数组 boundary_offset 与 boundary_spacing_ratio");
  requireSameLength(kGrid, residual, "帧数组 k_grid 与 residual");
  requireSameLength(kGrid, residualSumOnly, "帧数组 k_grid 与 residual_sum_only");
  requireSameLength(kGrid, residualFloor, "帧数组 k_grid 与 residual_floor");
  const pitfalls = requirePitfalls(frame);

  // —— 可选数组：缺了就降级（记进 degraded，最终挂成一个 status 角标）——
  const degraded: string[] = [];
  const landingOffset = optionalFirst(frame, "landing_offset");
  const landingY = optionalFirst(frame, "landing_y");
  const idealOffset = optionalFirst(frame, "ideal_offset");
  const idealY = optionalFirst(frame, "ideal_y");
  const residualMark = optionalFirst(frame, "residual_mark");
  const kMark = optionalFirst(frame, "k_mark");
  const sumValue = frame.has("sum_value") ? requireArray(frame, "sum_value") : undefined;
  if (!frame.has("landing_offset") || !frame.has("landing_y")) degraded.push("帧缺 landing_offset/landing_y：不画 0.1+0.2 的落点（不猜它在哪）");
  if (!frame.has("ideal_offset") || !frame.has("ideal_y")) degraded.push("帧缺 ideal_offset/ideal_y：不画数学上的 0.3（不猜它在哪）");
  if (!frame.has("residual_mark") || !frame.has("k_mark")) degraded.push("帧缺 residual_mark/k_mark：副轴不画当前点（不拿请求 t 冒充实测 k）");

  // —— meta ——
  const tMeta = metaNumber(frame, "t");
  if (tMeta === undefined) degraded.push("帧 meta 缺 t：scene.t 改用请求 t（两者可能不同）");
  const center = metaNumber(frame, "center");
  const spacingCenter = metaNumber(frame, "spacing_center");
  const nPoints = metaNumber(frame, "n_points");
  const boundary = metaNumber(frame, "boundary");
  const boundaryUlps = metaNumber(frame, "boundary_ulps");
  const windowUlps = metaNumber(frame, "window_ulps");
  const boundarySpacingMin = metaNumber(frame, "boundary_spacing_min");
  const boundarySpacingMax = metaNumber(frame, "boundary_spacing_max");
  const landingInWindow = (metaNumber(frame, "landing_in_window") ?? 1) > 0.5;
  const nExactZero = metaNumber(frame, "residual_exact_zero_steps");
  const nExactZeroSum = metaNumber(frame, "residual_sum_only_exact_zero_steps");
  const kMeta = metaNumber(frame, "k");
  if (kMeta === undefined) degraded.push("帧 meta 缺 k：游标按 k_mark 画（若 k_mark 也缺则按请求 t 夹取）");
  const bigPlusOneUnchanged = (metaNumber(frame, "big_plus_one_unchanged") ?? 0) > 0.5;
  const frameOverflow = (metaNumber(frame, "overflow") ?? 0) > 0.5;
  const note = stringOf(frame, "note");
  if (note === undefined) degraded.push("帧 strings 缺 note：状态角标不自行编造结论");
  const exactCenter = stringOf(frame, "exact_center");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: tMeta ?? opts.t,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  // —— 排版：上（梯子）下（残差）；宽画布右侧各留一条说明栏 ——
  const layout = floatZoomLayout(width, height, pitfalls);
  const mainRect: Rect = layout.main;
  const auxRect: Rect = layout.aux;

  // 横轴范围：两条梯子共用"±half 个 ulp"这个窗口（帧 meta 给的是同一个 half）。
  const half =
    Math.max(windowUlps ?? 0, boundaryUlps ?? 0) ||
    Math.max(absMax(offset) ?? 1, absMax(boundaryOffset) ?? 1);
  const ladderAxis = makeAxis("ladder", mainRect, {
    xlim: [-half, half],
    ylim: LADDER_YLIM,
    // 主轴 xlabel 置空：下方面板顶部要放说明文字，留着这条带子就会撞上（规则 8）。
    // 横轴的含义写在图的标题与图例里（manifest 的 axes 也只给了 title/ylabel）。
    xlabel: "",
    ylabel: axisYlabelOf(SPEC?.axes, "ladder", "间距 / 该窗口中心的 ulp"),
    title: axisTitleOf(SPEC?.axes, "ladder", "浮点梯子：中心窗口与最近的 2 的幂窗口"),
    autoscaled: false,
    xticks: niceTicks(-half, half, 9),
    yticks: [RATIO_HALVED, 0.75, RATIO_UNIFORM],
  });

  // 副轴（manifest 声明 autoscaled: true、scale: "log"）：三条曲线共用一条 log 纵轴。
  const bounds = finitePositiveMinMax(residual);
  const boundsSum = finitePositiveMinMax(residualSumOnly);
  const boundsFloor = finitePositiveMinMax(residualFloor);
  const lo = Math.min(
    bounds?.[0] ?? Number.POSITIVE_INFINITY,
    boundsSum?.[0] ?? Number.POSITIVE_INFINITY,
    boundsFloor?.[0] ?? Number.POSITIVE_INFINITY,
  );
  const hi = Math.max(bounds?.[1] ?? 0, boundsSum?.[1] ?? 0, boundsFloor?.[1] ?? 0);
  if (!(Number.isFinite(lo) && Number.isFinite(hi))) {
    degraded.push("residual / residual_sum_only / residual_floor 里没有 > 0 的有限值：副轴用占位范围 [1e-18, 1e-14]，不画曲线");
  }
  const auxYlim: [number, number] =
    Number.isFinite(lo) && Number.isFinite(hi) ? niceLimits(lo, hi, 6, { log: true, pad: 0.12 }) : [1e-18, 1e-14];
  const kMax = kGrid.length > 0 ? (kGrid[kGrid.length - 1] as number) : 1;
  const auxAxis = makeAxis("residual", auxRect, {
    xlim: [0, Math.max(1, kMax)],
    ylim: auxYlim,
    yscale: "log",
    xlabel: "0.1 的累加次数 k",
    ylabel: axisYlabelOf(SPEC?.axes, "residual", "|fl(Σ0.1) − k/10|"),
    title: axisTitleOf(SPEC?.axes, "residual", "0.1 累加 k 次与 k/10 的残差（0 以地板显示）"),
    autoscaled: true,
    tickTarget: 5,
  });
  scene.axes.push(ladderAxis, auxAxis);

  // —— 图层 1：间距比 = 1 的参照线（"网格均匀"的那条假想线）——
  scene.layers.push(
    lineLayer(
      "ladder",
      series([-half, half], [RATIO_UNIFORM, RATIO_UNIFORM], pal.reference, {
        width: 1,
        dash: [2, 4],
        label: "间距比 = 1（均匀网格的假想线）",
      }),
    ),
  );

  // —— 图层 2：以最近的 2 的幂为中心的梯子（虚线；跨过中心掉一半）——
  const boundarySeries = segmentedSeries(boundaryOffset, boundarySpacingRatio, pal.curves[1] ?? "#2f855a", {
    width: 1.8,
    dash: [7, 4],
  }).map((s) => ({ ...s, label: "跨过 2 的幂的网格" }));
  if (boundarySeries.length > 0) scene.layers.push(lineLayer("ladder", boundarySeries));
  else degraded.push("boundary_spacing_ratio 没有可绘的有限值：2 的幂梯子画不出来（网格均匀与否就没有证据了）");

  // —— 图层 3：以参数中心为中心的梯子（实线）——
  const centerSeries = segmentedSeries(offset, spacingRatio, pal.curves[0] ?? "#2b6cb0", {
    width: 2,
  }).map((s) => ({ ...s, label: "中心浮点网格" }));
  if (centerSeries.length > 0) scene.layers.push(lineLayer("ladder", centerSeries));
  else degraded.push("spacing_ratio 没有可绘的有限值：中心梯子画不出来");

  // —— 图层 4：两条竖线（数学上的 0.3 / 0.1+0.2 的落点）——
  const vlines = [];
  if (idealOffset !== undefined) {
    vlines.push(series([idealOffset, idealOffset], LADDER_YLIM, pal.reference, { width: 1.2, dash: [3, 3], label: "数学上的 0.3" }));
  }
  if (landingOffset !== undefined) {
    vlines.push(series([landingOffset, landingOffset], LADDER_YLIM, pal.marker, { width: 1.2, dash: [3, 3], label: "0.1 + 0.2 的落点" }));
  }
  if (vlines.length > 0) scene.layers.push(lineLayer("ladder", vlines));

  // —— 图层 5：落点（y 取帧给的 landing_y/ideal_y，不自己编位置）——
  const markX: number[] = [];
  const markY: number[] = [];
  if (idealOffset !== undefined && idealY !== undefined && inLadderY(idealY)) {
    markX.push(idealOffset);
    markY.push(idealY);
  }
  if (landingOffset !== undefined && landingY !== undefined && inLadderY(landingY)) {
    markX.push(landingOffset);
    markY.push(landingY);
  }
  if (markX.length > 0) {
    scene.layers.push(
      scatterLayer("ladder", markX, markY, { r: 5, color: pal.marker }),
    );
  }
  if (!landingInWindow) {
    // 落点被挪出窗口：帧 meta 说得很清楚，这里不画点，但必须让读者知道"看不见 ≠ 不在那里"。
    degraded.push("帧 meta.landing_in_window=false：0.1+0.2 的落点不在当前窗口内");
  }

  // —— 图层 6~10：副轴（规则 3 的残差场）——
  const kNow = kMeta ?? kMark ?? clamp(Math.round(opts.t), 0, kMax);
  scene.layers.push(
    lineLayer("residual", series([kNow, kNow], auxAxis.ylim, pal.cursor, { width: 1, dash: [3, 3] })),
  );
  const resSeries = segmentedSeries(kGrid, residual, pal.residual, {
    width: 1.9,
    accept: (v) => v > 0,
  }).map((s) => ({ ...s, label: "残差（对照 k/10）" }));
  if (resSeries.length > 0) scene.layers.push(lineLayer("residual", resSeries));
  else degraded.push("residual 的样本没有一个 > 0：log 轴上画不出这条曲线");
  const resSumSeries = segmentedSeries(kGrid, residualSumOnly, pal.curves[1] ?? "#2f855a", {
    width: 1.5,
    dash: [6, 3],
    accept: (v) => v > 0,
  }).map((s) => ({ ...s, label: "残差（对照 k·fl(0.1)）" }));
  if (resSumSeries.length > 0) scene.layers.push(lineLayer("residual", resSumSeries));
  else degraded.push("residual_sum_only 的样本没有一个 > 0：log 轴上画不出这条曲线");
  const floorSeries = segmentedSeries(kGrid, residualFloor, pal.reference, {
    width: 1.1,
    dash: [2, 3],
    accept: (v) => v > 0,
  }).map((s) => ({ ...s, label: "舍入地板" }));
  if (floorSeries.length > 0) scene.layers.push(lineLayer("residual", floorSeries));

  const onFloor = floorIndices(residual, residualFloor);
  const onFloorSum = floorIndices(residualSumOnly, residualFloor);
  if (onFloor.length > 0) {
    const p = pick(kGrid, residual, onFloor);
    scene.layers.push(
      scatterLayer("residual", p.x, p.y, { r: 3, color: pal.warn, alpha: 0.9 }),
    );
  }
  if (residualMark !== undefined && kMark !== undefined && residualMark > 0) {
    scene.layers.push(scatterLayer("residual", [kMark], [residualMark], { r: 4.5, color: pal.marker }));
  }

  // —— 文本：宽画布放在右栏；窄画布不铺说明，避免压住曲线 ——
  const summary = paramSummary(frame, opts, [
    { name: "center_log10", label: "中心(10 的幂)" },
    { name: "mantissa", label: "尾数" },
  ]);
  degraded.push(...summary.degraded);

  if (layout.mainText !== undefined) {
    const headData: string[] = [];
    if (center !== undefined) headData.push(`中心 ${formatNumber(center, 6)}`);
    if (spacingCenter !== undefined) headData.push(`ulp ${formatNumber(spacingCenter, 3)}`);
    if (nPoints !== undefined) headData.push(`窗口 ${formatNumber(nPoints, 4)} 个浮点数`);
    const head = [summary.text];
    if (headData.length > 0) head.push(headData.join("　"));

    const boundaryLabel = boundary === undefined ? "中心" : formatNumber(boundary, 6);
    const ratioJump =
      boundarySpacingMin !== undefined && boundarySpacingMax !== undefined
        ? `跨过它 ${formatNumber(boundarySpacingMin, 2)}→${formatNumber(boundarySpacingMax, 2)}`
        : "跨过它间距减半";
    const landingLabel =
      landingOffset === undefined ? "红点=0.1+0.2" : `红点=0.1+0.2（中心上 ${formatNumber(landingOffset, 3)} ulp）`;
    const legend = [
      "实线=中心窗口（同一 binade，间距比恒 1）",
      `虚线=2 的幂窗口 ${boundaryLabel}，${ratioJump}`,
      landingLabel,
      "灰虚线=数学上的 0.3",
    ];

    const why: string[] = [];
    if (center !== undefined && nPoints !== undefined) {
      why.push(`横轴是 ulp 偏移：窗口里 ${formatNumber(nPoints, 4)} 个 double 印成 16 位都是 ${formatNumber(center, 6)}`);
    } else {
      why.push("横轴是距各自中心的 ulp 偏移，不是真实坐标");
    }
    if (exactCenter !== undefined) why.push(`精确中心 ${exactCenter}`);
    if (bigPlusOneUnchanged) why.push("1e16+1=1e16 是正确结果：ulp=2，加 1 落在中点被 ties-to-even 舍回");

    pushTextColumn(scene.texts, layout.mainText, [
      { lines: head, color: pal.muted, gapAfter: 5 },
      { lines: legend, color: pal.muted, gapAfter: 5 },
      { lines: why, color: pal.note },
    ]);
  }

  if (layout.auxText !== undefined) {
    const kIdx = clamp(Math.round(kNow), 0, Math.max(0, kGrid.length - 1));
    const live: string[] = [`游标 k=${formatNumber(kNow, 4)}`];
    if (sumValue !== undefined && kIdx < sumValue.length) live.push(`fl(Σ0.1)=${formatNumber(sumValue[kIdx] as number, 8)}`);
    const residualLines: string[] = [];
    if (kIdx < residual.length) residualLines.push(`残差 ${formatNumber(residual[kIdx] as number, 3)}（紫，对 k/10）`);
    if (kIdx < residualSumOnly.length) residualLines.push(`残差 ${formatNumber(residualSumOnly[kIdx] as number, 3)}（绿虚，对 k·fl(0.1)）`);
    const floorLine = kIdx < residualFloor.length ? `地板 ${formatNumber(residualFloor[kIdx] as number, 3)}（灰点）` : "";
    const onFloorLine = `贴地板 residual ${onFloor.length}/${residual.length}（0 的 ${nExactZero === undefined ? "?" : formatNumber(nExactZero, 4)} 个）`;
    const onFloorSumLine = `sum_only ${onFloorSum.length}/${residualSumOnly.length}（0 的 ${nExactZeroSum === undefined ? "?" : formatNumber(nExactZeroSum, 4)} 个）`;

    pushTextColumn(scene.texts, layout.auxText, [
      { lines: [live.join("　")], color: pal.muted, gapAfter: 5 },
      { lines: residualLines, color: pal.muted, gapAfter: 3 },
      { lines: [floorLine], color: pal.muted, gapAfter: 5 },
      { lines: [onFloorLine, onFloorSumLine, "贴地板 ≠ 误差为 0：判据是它随 k 长不长"], color: pal.muted },
    ]);
  }

  // —— 角标（固定槽位）——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  if (scene.dataSource.startsWith("backend+preview")) {
    scene.badges.push(placeBadge(previewBadge("拖动时按显示空间插值"), SLOT.preview, width));
  }
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(
      badge("autoscale", "副轴：自适应 log 纵轴（按本帧三条残差取整）", 8, 8),
      SLOT.auxAxis,
      width,
    ),
  );
  const status: string[] = [`k=${formatNumber(kNow, 4)}`];
  if (landingInWindow && landingOffset !== undefined) {
    status.push(`0.1+0.2 在中心上 ${formatNumber(landingOffset, 3)} ulp`);
  } else if (!landingInWindow) {
    status.push("0.1+0.2 不在当前窗口");
  }
  if (note === undefined) status.push("帧未给出 note");
  if (status.length === 1) status.push("帧未给出可判读的落点状态");
  scene.badges.push(placeBadge(badge("status", status.join("；"), 8, 8), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  // 规则 1：图层越界 + 帧内非有限值，两者都要进 overflow 与角标。
  const nBad =
    countNonFinite(offset) +
    countNonFinite(spacingRatio) +
    countNonFinite(boundaryOffset) +
    countNonFinite(boundarySpacingRatio) +
    countNonFinite(kGrid) +
    countNonFinite(residual) +
    countNonFinite(residualSumOnly) +
    countNonFinite(residualFloor);
  applyOverflow(scene, nBad, { slot: SLOT.overflow, width });
  const extra: string[] = [];
  if (frameOverflow) extra.push("帧 meta.overflow=true：0.1+0.2 的落点不在当前窗口内（看不见不等于不存在）");
  if (extra.length > 0) {
    const note2 = [scene.overflow.note, ...extra].filter((s) => s.length > 0).join("；");
    scene.overflow = { flag: true, note: note2 };
    if (!scene.badges.some((b) => b.kind === "overflow")) {
      scene.badges.push(placeBadge(overflowBadge(note2), SLOT.overflow, width));
    }
  }
  // 窄画布（480px）下每块说明都会多折一两行，模型里按自己的行数排的下一块就会压上去。
  // 这里用共享盒子量法统一消解（放不下的如实计数并写明）；960×600 下是恒等变换。
  scene.texts = avoidTextOverlap(scene.texts, [8, 24, width - 16, layout.aux[1] + layout.aux[3] - 24]);

  return scene;
};

// --------------------------------------------------------------------------- //
// 局部工具
// --------------------------------------------------------------------------- //

interface TextBlockSpec {
  lines: readonly string[];
  color: string;
  gapAfter?: number;
}

/** 把几组说明按序排进右栏；放不下的行由 `textBlock` 截断并如实计数。 */
function pushTextColumn(out: TextSpec[], rect: Rect, blocks: readonly TextBlockSpec[]): void {
  const budget = unitBudget(rect[2]);
  const bottom = rect[1] + rect[3] - 4;
  let y = rect[1] + 4;
  for (const block of blocks) {
    const nodes = textBlock(rect[0], y, bottom, block.lines, block.color, budget, "top");
    out.push(...nodes);
    y += nodes.length * TEXT_ROW_H + (block.gapAfter ?? 0);
  }
}

/** 取"长度 1 的数组"的第 0 个元素；缺数组、空数组、非有限值一律 undefined（降级，不猜）。 */
function optionalFirst(frame: Frame, name: string): number | undefined {
  if (!frame.has(name)) return undefined;
  const a = requireArray(frame, name);
  if (a.length === 0) return undefined;
  const v = a[0] as number;
  return Number.isFinite(v) ? v : undefined;
}

/** 落点的 y 只有落在固定梯子范围里才画（否则会被 `checkOverflow` 记成越界，那是误报）。 */
function inLadderY(y: number): boolean {
  return y >= LADDER_YLIM[0] && y <= LADDER_YLIM[1];
}
