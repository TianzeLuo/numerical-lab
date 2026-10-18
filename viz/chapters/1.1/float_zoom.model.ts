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
 */

import { countNonFinite, dataToPixel } from "../../shared/src/geometry.js";
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
  previewBadge,
  scatterLayer,
  series,
  sourceBadge,
  text,
} from "../../shared/src/scene.js";
import {
  SAFE_GAP,
  absMax,
  avoidTextOverlap,
  axisTitleOf,
  axisYlabelOf,
  floorIndices,
  pick,
  textBlock,
  textBottom,
  textTop,
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
  const spacingMin = metaNumber(frame, "spacing_min");
  const spacingMax = metaNumber(frame, "spacing_max");
  const boundarySpacingMin = metaNumber(frame, "boundary_spacing_min");
  const boundarySpacingMax = metaNumber(frame, "boundary_spacing_max");
  const landingInWindow = (metaNumber(frame, "landing_in_window") ?? 1) > 0.5;
  const residual0102 = metaNumber(frame, "residual_0102_03");
  const residual0102Ulps = metaNumber(frame, "residual_0102_03_ulps");
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

  // —— 排版：上（梯子）下（残差）两块；两块都满宽 ——
  const base = stackedPanels(width, height);
  const mainRect: Rect = base.main;
  const auxRect: Rect = base.error;

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
        label: "间距比 = 1（同一 binade 内的均匀网格）",
      }),
    ),
  );

  // —— 图层 2：以最近的 2 的幂为中心的梯子（虚线；跨过中心掉一半）——
  const boundarySeries = segmentedSeries(boundaryOffset, boundarySpacingRatio, pal.curves[1] ?? "#2f855a", {
    width: 1.8,
    dash: [7, 4],
    label: "最近的 2 的幂窗口：跨过它间距减半（网格不均匀的直接证据）",
  });
  if (boundarySeries.length > 0) scene.layers.push(lineLayer("ladder", boundarySeries));
  else degraded.push("boundary_spacing_ratio 没有可绘的有限值：2 的幂梯子画不出来（网格均匀与否就没有证据了）");

  // —— 图层 3：以参数中心为中心的梯子（实线）——
  const centerSeries = segmentedSeries(offset, spacingRatio, pal.curves[0] ?? "#2b6cb0", {
    width: 2,
    label: "参数中心窗口：整个窗口在同一 binade 内，间距比恒为 1",
  });
  if (centerSeries.length > 0) scene.layers.push(lineLayer("ladder", centerSeries));
  else degraded.push("spacing_ratio 没有可绘的有限值：中心梯子画不出来");

  // —— 图层 4：两条竖线（数学上的 0.3 / 0.1+0.2 的落点）——
  const vlines = [];
  if (idealOffset !== undefined) {
    vlines.push(series([idealOffset, idealOffset], LADDER_YLIM, pal.reference, { width: 1.2, dash: [3, 3], label: "数学上的 0.3（= 窗口中心）" }));
  }
  if (landingOffset !== undefined) {
    vlines.push(series([landingOffset, landingOffset], LADDER_YLIM, pal.marker, { width: 1.2, dash: [3, 3], label: "0.1+0.2 的落点" }));
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
      scatterLayer("ladder", markX, markY, { r: 5, color: pal.marker, label: "0.1+0.2 的落点（与数学上的 0.3 相差 1 个 ulp）" }),
    );
  }
  if (!landingInWindow) {
    // 落点被挪出窗口：帧 meta 说得很清楚，这里不画点，但必须让读者知道"看不见 ≠ 不在那里"。
    degraded.push("帧 meta.landing_in_window=false：0.1+0.2 的落点不在当前窗口内");
  }

  // —— 图层 6~10：副轴（规则 3 的残差场）——
  const kNow = kMeta ?? kMark ?? clamp(Math.round(opts.t), 0, kMax);
  scene.layers.push(
    lineLayer("residual", series([kNow, kNow], auxAxis.ylim, pal.cursor, { width: 1, dash: [3, 3], label: `当前 k = ${formatNumber(kNow, 4)}` })),
  );
  const resSeries = segmentedSeries(kGrid, residual, pal.residual, {
    width: 1.9,
    label: "残差（相对数学值 k/10；精确值用有理数算）",
    accept: (v) => v > 0,
  });
  if (resSeries.length > 0) scene.layers.push(lineLayer("residual", resSeries));
  else degraded.push("residual 的样本没有一个 > 0：log 轴上画不出这条曲线");
  const resSumSeries = segmentedSeries(kGrid, residualSumOnly, pal.curves[1] ?? "#2f855a", {
    width: 1.5,
    dash: [6, 3],
    label: "残差（相对 k·fl(0.1)：只含求和误差，不含 0.1 的表示误差）",
    accept: (v) => v > 0,
  });
  if (resSumSeries.length > 0) scene.layers.push(lineLayer("residual", resSumSeries));
  else degraded.push("residual_sum_only 的样本没有一个 > 0：log 轴上画不出这条曲线");
  const floorSeries = segmentedSeries(kGrid, residualFloor, pal.reference, {
    width: 1.1,
    dash: [2, 3],
    label: "显示地板 u/2·max(|k/10|,0.1)：贴住它的点不是「误差 = 地板值」",
    accept: (v) => v > 0,
  });
  if (floorSeries.length > 0) scene.layers.push(lineLayer("residual", floorSeries));

  const onFloor = floorIndices(residual, residualFloor);
  const onFloorSum = floorIndices(residualSumOnly, residualFloor);
  if (onFloor.length > 0) {
    const p = pick(kGrid, residual, onFloor);
    scene.layers.push(
      scatterLayer("residual", p.x, p.y, { r: 3, color: pal.warn, alpha: 0.9, label: `贴地板的点（${onFloor.length} 个：精确 0 与低于地板在这里同高）` }),
    );
  }
  if (residualMark !== undefined && kMark !== undefined && residualMark > 0) {
    scene.layers.push(scatterLayer("residual", [kMark], [residualMark], { r: 4.5, color: pal.marker, label: "当前 k 的残差" }));
  }

  // —— 文本（一行一个 text 层：render2d 不拆 \n）——
  const summary = paramSummary(frame, opts, [
    { name: "center_log10", label: "中心(10 的幂)" },
    { name: "mantissa", label: "尾数" },
  ]);
  degraded.push(...summary.degraded);

  const head: string[] = [summary.text];
  if (center !== undefined) head.push(`中心=${formatNumber(center, 6)}`);
  if (spacingCenter !== undefined) head.push(`ulp(中心)=${formatNumber(spacingCenter, 3)}`);
  if (nPoints !== undefined) head.push(`窗口=${formatNumber(nPoints, 4)} 个可表示数`);
  else if (Number.isFinite(half)) head.push(`窗口=±${formatNumber(half, 4)} ulp`);
  if (spacingMin !== undefined && spacingMax !== undefined && spacingMin === spacingMax) {
    head.push(`中心梯子间距比恒为 ${formatNumber(spacingMin, 2)}`);
  }
  const mainBudget = unitBudget(mainRect[2]);
  scene.texts.push(...textBlock(mainRect[0] + 6, textTop(mainRect), textBottom(mainRect), [head.join("　")], pal.muted, mainBudget, "top"));

  // 两块文字都放在"没有曲线"的横带里：梯子只占 y≈0.5 与 y≈1.0 两条水平带，
  // 中间那条空带（0.55..0.95）放"怎么读"，下沿放图例。这样文字不会压住任何一条梯子。
  const midTop = mainRect[1] + Math.round(mainRect[3] * 0.26);
  const mid: string[] = [];
  mid.push("实线=参数中心窗口（整个窗口在同一 binade：间距比恒为 1）");
  if (boundarySpacingMin !== undefined && boundarySpacingMax !== undefined) {
    mid.push(
      `虚线=最近的 2 的幂窗口（${boundary === undefined ? "中心" : formatNumber(boundary, 6)}）：` +
        `跨过它间距比从 ${formatNumber(boundarySpacingMin, 2)} 跳到 ${formatNumber(boundarySpacingMax, 2)}——网格不均匀的直接证据`,
    );
  }
  if (bigPlusOneUnchanged) {
    mid.push("1e16 + 1 = 1e16（ulp(1e16) = 2，加 1 落在中点被 ties-to-even 舍回）：图上「看起来没变」是正确结果，不是渲染 bug");
  }
  scene.texts.push(...textBlock(mainRect[0] + 6, midTop, textBottom(mainRect), mid, pal.muted, mainBudget, "top"));

  // "为什么横轴不是真实坐标"必须写在图上，否则 ulp 偏移轴会被当成真实数轴。
  if (exactCenter !== undefined && nPoints !== undefined) {
    scene.texts.push(
      ...textBlock(
        mainRect[0] + 6,
        midTop + mid.length * 15 + 4,
        textBottom(mainRect),
        [
          `窗口里 ${formatNumber(nPoints, 4)} 个 double 在 16 位有效数字下印出来都是 ` +
            `${center === undefined ? "同一个数" : formatNumber(center, 6)}（中心的精确值 ${exactCenter}）：` +
            "真实坐标画不出这个差别，所以横轴必须是 ulp 偏移",
        ],
        pal.note,
        mainBudget,
        "top",
      ),
    );
  }

  const legend: string[] = [];
  legend.push("红=0.1+0.2 的落点（中心之上 1 个 ulp）　灰虚竖线=数学上的 0.3　横轴=距各自中心的 ulp 偏移");
  const crossCheck: string[] = [];
  if (residual0102 !== undefined) {
    crossCheck.push(
      `(0.1+0.2) − fl(0.3) = ${formatNumber(residual0102, 3)}` +
        (residual0102Ulps === undefined ? "" : `（恰好 ${formatNumber(residual0102Ulps, 2)} 个 ulp）`),
    );
  }
  crossCheck.push("而 (0.1+0.2) − 3/10 = 4.44e-17（副轴画的是这一条）：残差必须说清「相对谁」");
  legend.push(crossCheck.join("；"));
  scene.texts.push(...textBlock(mainRect[0] + 6, textTop(mainRect), textBottom(mainRect), legend, pal.muted, mainBudget, "bottom"));

  // 副轴：只留 3 行（帧 note + 当前数值 + 地板判读），曲线才看得见。
  const auxLines: string[] = [];
  if (note !== undefined) auxLines.push(`帧 note：${note}`);
  const kIdx = clamp(Math.round(kNow), 0, Math.max(0, kGrid.length - 1));
  const nums: string[] = [`k=${formatNumber(kNow, 4)}`];
  if (sumValue !== undefined && kIdx < sumValue.length) nums.push(`fl(Σ0.1)=${formatNumber(sumValue[kIdx] as number, 8)}`);
  if (kIdx < residual.length) nums.push(`残差(对 k/10)=${formatNumber(residual[kIdx] as number, 3)}`);
  if (kIdx < residualSumOnly.length) nums.push(`残差(对 k·fl(0.1))=${formatNumber(residualSumOnly[kIdx] as number, 3)}`);
  if (kIdx < residualFloor.length) nums.push(`地板=${formatNumber(residualFloor[kIdx] as number, 3)}`);
  auxLines.push(nums.join("　"));
  auxLines.push(
    `贴地板：residual ${onFloor.length}/${residual.length}（0 的 ` +
      `${nExactZero === undefined ? "?" : formatNumber(nExactZero, 4)} 个；` +
      `sum_only ${onFloorSum.length} 个，0 的 ${nExactZeroSum === undefined ? "?" : formatNumber(nExactZeroSum, 4)} 个）`,
  );
  auxLines.push("贴地板 ≠ 误差不为零：判据是它随 k 长不长");
  const auxBudget = unitBudget(auxRect[2]);
  scene.texts.push(...textBlock(auxRect[0] + 6, textTop(auxRect), textBottom(auxRect), auxLines, pal.muted, auxBudget, "top"));

  // 游标标签：贴着副轴底部（`SAFE_GAP` 之上），x 由**共享的** dataToPixel 给出。
  const anchorY = auxAxis.ylim[0];
  const [mx] = dataToPixel(auxAxis, kNow, anchorY);
  if (Number.isFinite(mx)) {
    scene.texts.push(
      text(
        clamp(mx + 4, auxRect[0] + 6, auxRect[0] + auxRect[2] - 60),
        auxRect[1] + auxRect[3] - SAFE_GAP - 15,
        `k=${formatNumber(kNow, 4)}`,
        { role: "note", size: 11, color: pal.marker, align: "left" },
      ),
    );
  }

  // —— 角标（固定槽位）——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  if (scene.dataSource.startsWith("backend+preview")) {
    scene.badges.push(placeBadge(previewBadge("拖动时按显示空间插值"), SLOT.preview, width));
  }
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(
      badge("autoscale", "副轴：自适应（log 纵轴按本帧三条残差曲线取整；自适应会掩盖量级变化）", 8, 8),
      SLOT.auxAxis,
      width,
    ),
  );
  const status: string[] = [];
  if (landingInWindow && landingOffset !== undefined) {
    status.push(`0.1+0.2 落在中心之上 ${formatNumber(landingOffset, 3)} 个 ulp`);
  } else if (!landingInWindow) {
    status.push("0.1+0.2 不在当前窗口内：把中心调到 0.3 再看");
  }
  if (note !== undefined) status.push(note);
  if (bigPlusOneUnchanged) status.push("1e16+1 = 1e16（ties-to-even），不是渲染 bug");
  if (status.length === 0) status.push("帧未给出可判读的状态（预期有 strings.note 与 meta.landing_ulps）");
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
  scene.texts = avoidTextOverlap(scene.texts, [8, 24, width - 16, height - 108]);

  return scene;
};

// --------------------------------------------------------------------------- //
// 局部工具
// --------------------------------------------------------------------------- //

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
