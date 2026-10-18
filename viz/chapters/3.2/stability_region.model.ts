/**
 * `stability_region`（3.2 章）的 model：复平面上的 λh 落点 + 精确稳定域边界 + **真算出来的解**。
 *
 * 面板与图层（顺序即绘制顺序，测试按此断言）
 * ------------------------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | heat | `main` | 网格上的 \|R(z)\|（固定窗口 Re∈[−8,4]、Im∈[−6,6]） |
 * | 2 | contour | `main` | 网格等值线（`levels`；**这就是"从网格上爬出来的边界"**，故意留在图上做对照） |
 * | 3 | line | `main` | 精确参数化的 \|R\|=1 边界 `z(θ)=ρ/σ`（窗口外的点不画，计数报出） |
 * | 4 | scatter | `main` | 扫描射线上 \|R\|≤1 的点（蓝） |
 * | 5 | scatter | `main` | 扫描射线上 \|R\|>1 的点（红）——域外 |
 * | 6 | scatter | `main` | 射线上的**真极点**（紫；画的是显示上限，不是数值） |
 * | 7 | scatter | `main` | 当前 λh 落点（大橙点） |
 * | 8 | scatter | `main` | 网格上的**真极点格子**（紫方块标记） |
 * | 9 | line | `sol` | 数值解 y₁ 与精确解 y₁（固定纵轴 ±2；越界样本截断并计数） |
 * | 10 | line | `aux` | 数值解误差（对数；地板 u/2 出现在文本里） |
 * | 11 | line | `aux` | 当前 \|R(λh)\| 的水平参考线 + \|R\|=1 临界线 |
 * | 12 | scatter | `aux` | 当前时刻误差标记 |
 *
 * 时间轴语义：`t = log10 h`（步长扫描，manifest `time.mode="scrub"`，`tMax=4`）。
 * 帧 meta 里的 `log10_h`/`h` 是**实到**值（后端夹到 [−3,1]），显示一律用它，不用请求值。
 *
 * 三条刻意的决定（每一条都对着一个具体的坑）
 * ----------------------------------------
 * 1. **极点不是数值**。BDF-k 在 z=α_k（1、1.5、11/6、25/12、137/60、49/20）有真极点 |R|=+∞，
 *    而本章的窗口网格**正好压在这些点上**（Re 步长 0.1：z=1、1.5、2 都是格点）。后端把格子写成
 *    `pole_fill = 1e6` 并用 `abs_R_pole` / `abs_R_ray_pole` 标出来——前端三件事必须做到：
 *    ① 色标给这些格子饱和色、再叠一枚**单独颜色**的极点标记；② 绝不把它当 1e6 的真值去说
 *    "最不稳定"（`meta.abs_R_max_finite` 才是"不含极点的真最大值"）；③ 文本里写明"这里没有
 *    有限值，画的是显示上限"。第一版把极点写成 0.0，图上读成"非常稳定"——正好读反。
 * 2. **落点在域外 ≠ 算错了**：`lam_re>0` 时真解本身在增长。所以状态角标与文本把两件事分开写：
 *    \|R\|>1（方法放大了这个模态）与误差变大（解不准），并给出 `lam_re` 的临界线 0。
 * 3. **窗口固定、越界要报账**：射线上有 241−`ray_inside` 个点在窗口外（h 最大到 10、|z| 到 60+），
 *    梯形的边界有 148/721 个点超出窗口（隐式方法的稳定域无界）。这些点**不画**，
 *    但点数、方向与理由都进 `overflow.note`——"图上看不见"必须是账上写得清的事。
 */

import { formatNumber } from "../../shared/src/axes.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import {
  ModelInputError,
  canvasSize,
  clipBadgeText,
  criticalOf,
  dataSourceOf,
  degradedBadge,
  finitePositiveMinMax,
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
  contourLayer,
  criticalBadge,
  heatLayer,
  lineLayer,
  makeAxis,
  scatterLayer,
  series,
  sourceBadge,
} from "../../shared/src/scene.js";
import {
  absMax,
  applyOverflowNote,
  argmaxIndex,
  axisTitleOf,
  axisYlabelOf,
  chapterLayout,
  clipRuns,
  countAtCeiling,
  countMask,
  logRange,
  pick,
  textBlock,
  textBottom,
  textTop,
  unitBudget,
} from "./_panel.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "3.2";
const VIZ = "stability_region";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "稳定域交互图：λh 落点与解的真实发散";

/** 固定角标槽位（见 `model.ts::BADGE_LAYOUT`；8 个槽位 = 两列 4 行，塞满会压到面板上）。 */
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

/** `sol` 面板的固定纵轴（与 viz 2/4 的"解曲线固定纵轴 ±2"同一口径）。 */
const SOL_LIM: [number, number] = [-2, 2];

/** 色标过道（`tools/raster.py` 把色标画在轴右 46px、宽 14px + 数值标签 + 居中说明）。 */
const COLORBAR_GUTTER = 46 + 14 + 52 + 42 + 8;

export const stabilityRegionModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需数组：缺一个就抛 ModelInputError（宁可当场炸，也不画半张图）——
  const absR = requireArray(frame, "abs_R");
  const absRPole = requireArray(frame, "abs_R_pole");
  const extent = requireArray(frame, "extent");
  const levels = requireArray(frame, "levels");
  const boundaryX = requireArray(frame, "boundary_x");
  const boundaryY = requireArray(frame, "boundary_y");
  const rayX = requireArray(frame, "ray_x");
  const rayY = requireArray(frame, "ray_y");
  const rayInside = requireArray(frame, "ray_inside");
  const absRRay = requireArray(frame, "abs_R_ray");
  const absRRayPole = requireArray(frame, "abs_R_ray_pole");
  const zPointX = requireArray(frame, "z_point_x");
  const zPointY = requireArray(frame, "z_point_y");
  const tSol = requireArray(frame, "t_solution");
  const y1Num = requireArray(frame, "y1_num");
  const y1Exact = requireArray(frame, "y1_exact");
  const error = requireArray(frame, "error");
  const errorMark = requireArray(frame, "error_mark");
  // 可选（只进文本）：边界自洽残差、第二分量、用于计数"触顶样本"的截断数组。
  const residual = frame.has("residual") ? requireArray(frame, "residual") : undefined;
  const yMagnitude = frame.has("y_magnitude") ? requireArray(frame, "y_magnitude") : undefined;

  const shape = frame.shapeOf("abs_R");
  if (shape.length !== 2) {
    throw new ModelInputError(`帧 ${CHAPTER}/${VIZ} 的 'abs_R' 形状 [${shape.join(", ")}] 不是二维：热力图需要 (ny, nx)`);
  }
  const ny = shape[0];
  const nx = shape[1];
  if (absR.length !== nx * ny) {
    throw new ModelInputError(`帧 ${CHAPTER}/${VIZ} 的 'abs_R' 有 ${absR.length} 个元素，与 ${ny}×${nx} 不符`);
  }
  if (absRPole.length !== absR.length) {
    throw new ModelInputError(`帧 ${CHAPTER}/${VIZ} 的 'abs_R_pole' 长度 ${absRPole.length} ≠ 'abs_R' 的 ${absR.length}`);
  }
  if (extent.length !== 4) {
    throw new ModelInputError(`帧 ${CHAPTER}/${VIZ} 的 'extent' 长度 ${extent.length} ≠ 4（[x0,x1,y0,y1]）`);
  }
  const [x0, x1, y0, y1] = [extent[0], extent[1], extent[2], extent[3]];
  if (!(x1 > x0) || !(y1 > y0)) {
    throw new ModelInputError(`帧 ${CHAPTER}/${VIZ} 的 extent 必须严格递增：[${x0}, ${x1}, ${y0}, ${y1}]`);
  }
  requireSameLength(boundaryX, boundaryY, "帧数组 boundary_x 与 boundary_y");
  requireSameLength(rayX, rayY, "帧数组 ray_x 与 ray_y");
  requireSameLength(rayX, absRRay, "帧数组 ray_x 与 abs_R_ray");
  requireSameLength(rayX, rayInside, "帧数组 ray_x 与 ray_inside");
  requireSameLength(tSol, y1Num, "帧数组 t_solution 与 y1_num");
  requireSameLength(tSol, y1Exact, "帧数组 t_solution 与 y1_exact");
  requireSameLength(tSol, error, "帧数组 t_solution 与 error");
  const pitfalls = requirePitfalls(frame);

  // —— 可选 meta：缺了就降级（记进 degraded，最后挂成一个 status 角标）——
  const degraded: string[] = [];
  const tMeta = metaNumber(frame, "t");
  if (tMeta === undefined) degraded.push("帧 meta 缺 t：scene.t 改用请求 t（两者可能不同）");
  const tNow = tMeta ?? opts.t;
  const log10H = metaNumber(frame, "log10_h");
  if (log10H === undefined) degraded.push("帧 meta 缺 log10_h：不显示步长扫描位置（不按 t 猜）");
  const h = metaNumber(frame, "h");
  const hExit = metaNumber(frame, "h_exit");
  const hExitUnbounded = (metaNumber(frame, "h_exit_unbounded") ?? 0) > 0.5;
  const absRCurrent = metaNumber(frame, "abs_R_current");
  const stable = (metaNumber(frame, "stable") ?? 0) > 0.5;
  const lamRe = metaNumber(frame, "lam_re");
  const lamIm = metaNumber(frame, "lam_im");
  const zRe = metaNumber(frame, "z_re");
  const zIm = metaNumber(frame, "z_im");
  const tEnd = metaNumber(frame, "t_end");
  const nSteps = metaNumber(frame, "n_steps");
  const nStepsPlanned = metaNumber(frame, "n_steps_planned");
  const errorFinal = metaNumber(frame, "error_final");
  const errorMax = metaNumber(frame, "error_max");
  const maxAbsY = metaNumber(frame, "max_abs_y");
  const displayCeiling = metaNumber(frame, "display_ceiling");
  const blowup = (metaNumber(frame, "blowup") ?? 0) > 0.5;
  const tBlowup = metaNumber(frame, "t_blowup");
  const pointVisible = (metaNumber(frame, "point_visible") ?? 0) > 0.5;
  const absRMaxFinite = metaNumber(frame, "abs_R_max_finite");
  const nPoleCellsMeta = metaNumber(frame, "n_pole_cells");
  const nPoleRayMeta = metaNumber(frame, "n_pole_ray");
  const poleCurrent = (metaNumber(frame, "pole_current") ?? 0) > 0.5;
  const poleFill = metaNumber(frame, "pole_fill");
  const roundoffFloor = metaNumber(frame, "roundoff_floor");
  const aStable = (metaNumber(frame, "a_stable") ?? 0) > 0.5;
  const methodName = stringOf(frame, "method_name");
  if (methodName === undefined) degraded.push("帧 strings 缺 method_name：状态行只用 method 键名");
  const note = stringOf(frame, "note");
  if (note === undefined) degraded.push("帧 strings 缺 note：状态角标不自行编造结论");
  const poleNote = stringOf(frame, "pole_note");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: tNow,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  // —— 排版：左方形复平面（+ 色标过道），右列上「解曲线」下「\|R\| 与误差」 ——
  const layout = chapterLayout(width, height, {
    rightShares: [0.46, 0.54],
    gutter: COLORBAR_GUTTER,
    mainReserve: 84, // 面板下方留给三条图例（各 15px + 装饰带 28px）
  });
  const mainRect = layout.main;
  const solRect = layout.right[0];
  const auxRect = layout.right[1];
  scene.axes.push(
    makeAxis("main", mainRect, {
      xlim: [x0, x1],
      ylim: [y0, y1],
      xlabel: "Re z",
      ylabel: axisYlabelOf(SPEC?.axes, "main", "Im z"),
      title: axisTitleOf(SPEC?.axes, "main", "复平面 z = λh（固定窗口）"),
      autoscaled: false,
    }),
  );
  // `sol` 是模型自加的面板（manifest 只声明 main/aux）：它画的是帧里已有的解数组，
  // 不是新物理量。固定纵轴 ±2，与本章 viz 2/4 的"解曲线固定纵轴 ±2"同一口径。
  const scene0 = tSol.length > 0 ? tSol[0] : 0;
  const scene1 = tSol.length > 0 ? tSol[tSol.length - 1] : 0;
  const solXlim: [number, number] = [Math.min(0, scene0), Math.max(tEnd ?? scene1, scene1)];
  scene.axes.push(
    makeAxis("sol", solRect, {
      xlim: solXlim[0] === solXlim[1] ? [0, 1] : solXlim,
      ylim: SOL_LIM,
      xlabel: "t（数值解与精确解共用）",
      ylabel: "y₁",
      title: "真算出来的解：固定纵轴 ±2",
      autoscaled: false,
    }),
  );
  // 副轴（manifest 声明 autoscaled: true、scale: log）：|R(λh)| 与误差共用一条对数纵轴。
  const errPos = finitePositiveMinMax(error);
  const markPos = finitePositiveMinMax(errorMark);
  const currentForRange = absRCurrent !== undefined && !poleCurrent && absRCurrent > 0 ? absRCurrent : undefined;
  const auxLo = Math.min(errPos?.[0] ?? Number.POSITIVE_INFINITY, markPos?.[0] ?? Number.POSITIVE_INFINITY, currentForRange ?? Number.POSITIVE_INFINITY, 1);
  const auxHi = Math.max(errPos?.[1] ?? Number.NEGATIVE_INFINITY, markPos?.[1] ?? Number.NEGATIVE_INFINITY, currentForRange ?? Number.NEGATIVE_INFINITY, 1);
  const auxYlim: [number, number] =
    Number.isFinite(auxLo) && Number.isFinite(auxHi) && auxLo > 0 && auxHi > 0 ? logRange(auxLo, auxHi) : [1e-16, 1];
  if (!(Number.isFinite(auxLo) && Number.isFinite(auxHi))) {
    degraded.push("error/abs_R_current 里没有 >0 的有限值：副轴用占位范围 [1e-16, 1]，不画曲线");
  }
  scene.axes.push(
    makeAxis("aux", auxRect, {
      xlim: solXlim[0] === solXlim[1] ? [0, 1] : solXlim,
      ylim: auxYlim,
      yscale: "log",
      xlabel: "t",
      ylabel: axisYlabelOf(SPEC?.axes, "aux", "|R| 与误差"),
      title: axisTitleOf(SPEC?.axes, "aux", "|R(λh)| 与数值解误差"),
      autoscaled: true,
    }),
  );
  const auxAxis = scene.axes[2];

  // —— 图层 1~2：\|R(z)\| 热力图 + 网格等值线 ——
  // 色标用**对数**，并且以 |R|=1 为几何中心：`[1/max, max]`（max = 帧 meta.abs_R_max_finite）。
  // 两个理由：
  //  1. "白/中性色 = |R|=1"正好落在稳定边界上——蓝=衰减、红=放大，颜色本身就是判据；
  //  2. **一个格子都不截断**：上端就是本帧的最大有限值（本章窗口里 |R| 能到 306，
  //     线性色标要么让 86% 的格子饱和、要么让边界附近糊成一片）。
  // 极点格子在色标里取上端（饱和色），再叠一枚紫色的极点标记：**没有有限值**这件事
  // 由标记与文本说，不由颜色说。
  const levelOne = nearestLevel(levels, 1.0) ?? 1.0;
  const maxFinite = absRMaxFinite ?? absMax(absR) ?? 1;
  const vmax = Math.max(maxFinite, levelOne * 1.000001);
  const vmin = 1 / vmax; // 几何对称：vmin·vmax = 1 ⇒ |R|=1 在色标正中
  const heatValues = new Float64Array(absR.length);
  let nPoleCells = 0;
  let nSaturated = 0;
  for (let i = 0; i < absR.length; i += 1) {
    const isPole = absRPole[i] > 0.5;
    const v = absR[i];
    if (isPole) nPoleCells += 1;
    if (isPole || !Number.isFinite(v)) {
      heatValues[i] = vmax; // 上端饱和：色标之上，绝不冒充有限值
      continue;
    }
    if (v > vmax * (1 + 1e-12) || v < vmin * (1 - 1e-12)) nSaturated += 1;
    heatValues[i] = v;
  }
  scene.layers.push(
    heatLayer("main", heatValues, nx, ny, [x0, x1, y0, y1], {
      cmap: "coolwarm",
      vmin,
      vmax,
      logScale: true,
      label: "|R(z)| 对数色标，中点=1",
    }),
  );
  const contourLevels = Array.from(levels).filter((v) => Number.isFinite(v) && v > 0);
  scene.layers.push(
    contourLayer("main", heatValues, nx, ny, [x0, x1, y0, y1], {
      levels: contourLevels,
      color: "#4a5568",
      width: 1,
      label: "网格等值线（分辨率有限，只给颜色参考）",
    }),
  );

  // —— 图层 3：精确参数化的 \|R\|=1 边界（窗口外的点不画，但计数报账）——
  // `clipRuns` 只按 y 裁；x 再过滤一次（隐式方法的边界会在 y 方向冲出窗口，
  // 而梯形的边界是**整条虚轴**：721 个点里有 148 个在 Im∈[−6,6] 之外）。
  let nBoundaryOutside = 0;
  for (let i = 0; i < boundaryX.length; i += 1) {
    const bx = boundaryX[i];
    const by = boundaryY[i];
    if (!Number.isFinite(bx) || !Number.isFinite(by) || bx < x0 || bx > x1 || by < y0 || by > y1) nBoundaryOutside += 1;
  }
  const boundaryRuns = clipRuns(boundaryX, boundaryY, y0, y1).runs.filter((r) => r.x.every((v) => v >= x0 && v <= x1));
  for (const run of boundaryRuns) {
    scene.layers.push(
      lineLayer(
        "main",
        series(run.x, run.y, pal.ink, { width: 2.4, label: "|R|=1" }),
      ),
    );
  }

  // —— 图层 4~6：扫描射线（按 ray_inside 与 |R| 是否 >1 分三组上色）——
  const rayInsideIdx: number[] = [];
  const rayOutIdx: number[] = [];
  const rayOutIdx2: number[] = [];
  const rayPoleIdx: number[] = [];
  for (let i = 0; i < rayX.length; i += 1) {
    if (absRRayPole[i] > 0.5) {
      if (rayInside[i] > 0.5) rayPoleIdx.push(i);
      continue;
    }
    if (rayInside[i] <= 0.5) {
      rayOutIdx2.push(i);
      continue;
    }
    if (absRRay[i] > levelOne) rayOutIdx.push(i);
    else rayInsideIdx.push(i);
  }
  const rayGroups: { idx: number[]; color: string; r: number; label: string }[] = [
    { idx: rayInsideIdx, color: pal.curves[0], r: 2.2, label: "λh 扫描射线（|R|≤1，域内）" },
    { idx: rayOutIdx, color: pal.marker, r: 2.6, label: "λh 扫描射线（|R|>1，域外）" },
    { idx: rayPoleIdx, color: "#805ad5", r: 3.4, label: "射线上的真极点 |R|=+∞（画的是显示上限）" },
  ];
  for (const g of rayGroups) {
    if (g.idx.length === 0) continue;
    const pts = pick(rayX, rayY, g.idx);
    scene.layers.push(scatterLayer("main", pts.x, pts.y, { r: g.r, color: g.color, label: g.label }));
  }
  // 射线第一次出域的位置（对帧数组做阈值比较，不是重算物理量）。
  const exitIdx = rayInsideIdx.length + rayOutIdx.length > 0 ? firstAbove(absRRay, levelOne, rayInside) : -1;
  if (exitIdx >= 0 && rayInside[exitIdx] > 0.5) {
    scene.layers.push(
      scatterLayer("main", [rayX[exitIdx]], [rayY[exitIdx]], {
        r: 4.5,
        color: pal.warn,
        alpha: 0.9,
        label: "射线穿过 |R|=1 的位置",
      }),
    );
  }

  // —— 图层 7~8：当前 λh 落点与网格上的真极点格子 ——
  const zx = zPointX.length > 0 ? zPointX[0] : zRe;
  const zy = zPointY.length > 0 ? zPointY[0] : zIm;
  const pointInWindow =
    zx !== undefined && zy !== undefined && zx >= x0 && zx <= x1 && zy >= y0 && zy <= y1;
  if (pointInWindow) {
    scene.layers.push(
      scatterLayer("main", [zx as number], [zy as number], { r: 6, color: pal.exact, label: "当前 λh（帧数组 z_point_x/z_point_y）" }),
    );
  }
  const poleCells: number[] = [];
  if (nPoleCells > 0) {
    for (let i = 0; i < absRPole.length; i += 1) {
      if (absRPole[i] > 0.5) poleCells.push(i);
    }
    const px: number[] = [];
    const py: number[] = [];
    for (const i of poleCells) {
      const j = Math.floor(i / nx);
      const ii = i % nx;
      px.push(x0 + ((x1 - x0) * ii) / Math.max(1, nx - 1));
      py.push(y0 + ((y1 - y0) * j) / Math.max(1, ny - 1));
    }
    scene.layers.push(
      scatterLayer("main", px, py, {
        r: 5,
        color: "#805ad5",
        alpha: 1,
        label: "真极点格子 |R|=+∞（帧掩码 abs_R_pole；不是 1e6 这样的数值）",
      }),
    );
  }

  // —— 图层 9：数值解 vs 精确解（固定 ±2；越界截断 + 计数）——
  const numClip = clipRuns(tSol, y1Num, SOL_LIM[0], SOL_LIM[1]);
  const exactClip = clipRuns(tSol, y1Exact, SOL_LIM[0], SOL_LIM[1]);
  for (const run of exactClip.runs) {
    scene.layers.push(lineLayer("sol", series(run.x, run.y, pal.exact, { width: 1.4, dash: [6, 3], label: "精确解" })));
  }
  for (const run of numClip.runs) {
    scene.layers.push(lineLayer("sol", series(run.x, run.y, pal.curves[0], { width: 1.8, label: "数值解" })));
  }
  if (numClip.firstClip !== undefined) {
    scene.layers.push(
      scatterLayer("sol", [numClip.firstClip[0]], [numClip.firstClip[1]], {
        r: 5,
        color: pal.warn,
        label: "数值解第一次超出 ±2 的位置（之后不画）",
      }),
    );
  }

  // —— 图层 10~12：副轴（规则 3 的误差场：同屏给出误差）——
  const errSeries = clipRuns(tSol, error, auxAxis.ylim[0], auxAxis.ylim[1], (v) => v > 0);
  for (const run of errSeries.runs) {
    scene.layers.push(lineLayer("aux", series(run.x, run.y, pal.residual, { width: 1.6, label: "误差" })));
  }
  if (errSeries.runs.length === 0) degraded.push(`帧 error 的 ${error.length} 个样本没有一个 > 0：对数轴上画不出误差曲线`);
  // 当前 |R(λh)| 的水平参考线：它**不随 t 变化**，是"这一刻方法的放大因子"。
  if (currentForRange !== undefined) {
    scene.layers.push(
      lineLayer(
        "aux",
        series([auxAxis.xlim[0], auxAxis.xlim[1]], [currentForRange, currentForRange], pal.marker, {
          width: 1.4,
          dash: [5, 3],
          label: "|R(λh)|",
        }),
      ),
    );
  }
  scene.layers.push(
    lineLayer(
      "aux",
      series([auxAxis.xlim[0], auxAxis.xlim[1]], [levelOne, levelOne], pal.reference, {
        width: 1.1,
        dash: [2, 3],
        label: "=1",
      }),
    ),
  );
  const markX = errorMark.length > 0 ? tSol[tSol.length - 1] : undefined;
  const markY = errorMark.length > 0 ? errorMark[0] : undefined;
  if (markX !== undefined && markY !== undefined && Number.isFinite(markY) && markY > 0) {
    scene.layers.push(scatterLayer("aux", [markX], [markY], { r: 4.5, color: pal.marker, label: "积分终点处的误差（帧 error_mark）" }));
  }

  // —— 文本（一行一个 text 层；`render2d.ts::drawText` 不拆 \n）——
  const head: string[] = [];
  const methodLabel = methodName ?? stringOf(frame, "method") ?? "?";
  if (log10H !== undefined) head.push(`log10 h=${formatNumber(log10H, 3)}${h === undefined ? "" : `（h=${formatNumber(h, 3)}）`}`);
  if (poleCurrent) {
    head.push(`|R(λh)|=+∞（真极点）→ 帧里占位 ${poleFill === undefined ? "?" : formatNumber(poleFill, 6)}，**不是真值**`);
  } else if (absRCurrent !== undefined) {
    head.push(`|R(λh)|=${formatNumber(absRCurrent, 4)}　${stable ? "域内（不放大）" : "⚠ 域外（放大）"}`);
  }
  if (lamRe !== undefined && lamIm !== undefined) {
    head.push(`λ=${formatNumber(lamRe, 3)}${lamIm >= 0 ? "+" : "−"}${formatNumber(Math.abs(lamIm), 3)}i`);
  }
  if (zx !== undefined && zy !== undefined) head.push(`z=(${formatNumber(zx, 3)}, ${formatNumber(zy, 3)})`);
  const mainBudget = unitBudget(mainRect[2]);
  const rayOutside = rayX.length - countMask(rayInside);
  // 图例三行：宽画布放在方形面板**下方**（那里没有别的面板），窄画布（单列）则收进面板里，
  // 否则文字会落在下方那些面板身上（面板不是校验器看得见的盒子，但它真的会压住曲线）。
  const legend: string[] = [
    "细灰=网格等值线；粗黑=参数化边界 |R|=1",
    "射线：蓝=域内、红=域外、紫=真极点；橙点=当前 λh",
    `色标中点=|R|=1（越界 ${nSaturated} 格）；窗外未画：边界 ${nBoundaryOutside} 点、射线 ${rayOutside} 点`,
  ];
  const headLines = [`${methodLabel}：${head.join("　")}`];
  // 这两条必须**紧跟第一行**：角标与溢出说明都会被 `clipBadgeText` 截断，写在尾部等于没写。
  if (nPoleCells > 0) {
    headLines.push(
      `极点 ${nPoleCells} 格：|R|=+∞，帧里的 ${poleFill === undefined ? "pole_fill" : formatNumber(poleFill, 6)} 是**显示上限不是数值**` +
        `（紫色标记；不要把它当"最不稳定"）`,
    );
  }
  if (pointVisible === false) headLines.push("当前 λh 不在固定窗口内（帧 meta.point_visible=False）：主图上**没有**落点标记");
  if (!layout.columns) headLines.push(...legend); // 单列时面板下方没有地方放图例
  scene.texts.push(...textBlock(mainRect[0] + 6, textTop(mainRect), textBottom(mainRect), headLines, pal.ink, mainBudget, "top"));
  if (layout.columns) {
    scene.texts.push(
      ...textBlock(
        mainRect[0] + 6,
        mainRect[1] + mainRect[3] + 32,
        layout.base.bottom + 20,
        legend,
        nPoleCells > 0 ? "#805ad5" : pal.muted,
        mainBudget,
        "top",
      ),
    );
  }

  const solLines: string[] = [];
  if (maxAbsY !== undefined) solLines.push(`真实 max|y_num|=${formatNumber(maxAbsY, 4)}（未截断，帧 meta）`);
  if (numClip.nClipped > 0) {
    solLines.push(
      `数值解有 ${numClip.nClipped}/${tSol.length} 个样本超出 ±2：已截断不画（不画成一条贴边的水平线）` +
        (numClip.clippedSpan === undefined ? "" : `，未画区间 t∈[${formatNumber(numClip.clippedSpan[0], 3)}, ${formatNumber(numClip.clippedSpan[1], 3)}]`),
    );
  } else {
    solLines.push("数值解全部落在 ±2 内（这一帧还没发散到画面外）");
  }
  if (blowup) solLines.push(`帧 meta.blowup=True：积分在 t=${formatNumber(tBlowup ?? 0, 3)} 处发散`);
  scene.texts.push(...textBlock(solRect[0] + 6, textTop(solRect), textBottom(solRect), solLines, numClip.nClipped > 0 ? pal.warn : pal.muted, unitBudget(solRect[2]), "top"));

  const auxLines: string[] = [];
  if (note !== undefined) auxLines.push(`帧 note：${note}`);
  if (hExit !== undefined) {
    auxLines.push(
      hExitUnbounded
        ? "这个方向上没有出域步长（整条射线都在稳定域内）"
        : `出域步长 h_exit=${formatNumber(hExit, 4)}（沿当前射线二分实测；h>h_exit 即出域）`,
    );
  }
  if (errorFinal !== undefined) auxLines.push(`终点误差=${formatNumber(errorFinal, 3)}　最大误差=${errorMax === undefined ? "?" : formatNumber(errorMax, 3)}`);
  if (residual !== undefined) {
    const residualMax = metaNumber(frame, "residual_max");
    auxLines.push(
      `边界自洽残差 max||R(z_θ)|−1|=${formatNumber(residualMax ?? absMax(residual) ?? 0, 3)}（帧数组 residual，共 ${residual.length} 点）：` +
        "边界是**参数化直接算的**，不是从网格上爬出来的等高线。",
    );
  }
  if (nSteps !== undefined) {
    auxLines.push(
      `步数 ${formatNumber(nSteps, 4)}${nStepsPlanned === undefined ? "" : `（计划 ${formatNumber(nStepsPlanned, 4)}）`}` +
        (roundoffFloor === undefined ? "" : `；误差地板 ${formatNumber(roundoffFloor, 3)}=u/2（贴地板的样本可能是精确 0）`),
    );
  }
  auxLines.push("同一条对数轴：|R(λh)| 是水平虚线（不随 t 变），误差是实线。|R|>1=方法放大模态，误差大=解不准，两件事分开读。");
  scene.texts.push(...textBlock(auxRect[0] + 6, textTop(auxRect), textBottom(auxRect), auxLines, pal.muted, unitBudget(auxRect[2]), "top"));

  // —— 角标（固定槽位）——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  scene.badges.push(
    placeBadge(badge("autoscale", `主轴：固定窗口 [${formatNumber(x0, 3)}, ${formatNumber(x1, 3)}]×[${formatNumber(y0, 3)}, ${formatNumber(y1, 3)}]`, 8, 26), SLOT.axisMode, width),
  );
  scene.badges.push(
    placeBadge(
      badge("autoscale", "副轴：自适应 log（按本帧取整；会掩盖量级变化）", 8, 44),
      SLOT.auxAxis,
      width,
    ),
  );
  const lamCritical = criticalOf(frame, "lam_re", 0);
  if (lamRe !== undefined) {
    scene.badges.push(
      placeBadge(criticalBadge("Re λ", lamRe, lamCritical, { beyond: "above" }), SLOT.critical, width),
    );
  }
  const status: string[] = [];
  if (poleCurrent) {
    status.push(`⚠ 当前 λh 正好落在极点：|R|=+∞，帧里的 ${poleFill === undefined ? "占位值" : formatNumber(poleFill, 6)} 是**显示上限不是数值**`);
  } else if (absRCurrent !== undefined) {
    status.push(`|R(λh)|=${formatNumber(absRCurrent, 4)} ${stable ? "≤1（域内）" : ">1（域外，模态被放大）"}`);
  }
  if (stable && blowup) status.push("⚠ |R|≤1 却发散了：看下面的分解");
  if (absRMaxFinite !== undefined) status.push(`网格上不含极点的最大 |R|=${formatNumber(absRMaxFinite, 4)}`);
  if (nPoleCellsMeta !== undefined && nPoleCells === 0 && nPoleCellsMeta > 0) {
    degraded.push(`帧 meta.n_pole_cells=${formatNumber(nPoleCellsMeta, 4)} 但 abs_R_pole 全是 0：按掩码画（掩码是数据）`);
  }
  if (nPoleRayMeta !== undefined && nPoleRayMeta > 0 && rayPoleIdx.length === 0) {
    degraded.push(`帧 meta.n_pole_ray=${formatNumber(nPoleRayMeta, 4)} 但 abs_R_ray_pole 全是 0：按掩码画`);
  }
  if (poleNote !== undefined) status.push(poleNote);
  if (note !== undefined) status.push(note);
  if (!aStable) status.push("该方法不是 A-稳定（左半平面并非全部落在稳定域内）");
  if (status.length === 0) status.push("帧未给出可判读的状态（预期有 strings.note 与 meta.stable）");
  scene.badges.push(placeBadge(badge("status", clipBadgeText(status.join("；")), 8, 8), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  // —— 规则 1：越界 + 非有限值 + **模型这一侧**的截断（色标饱和、极点占位、解曲线越界）——
  const nBad =
    countNonFinite(absR) +
    countNonFinite(boundaryX) +
    countNonFinite(boundaryY) +
    countNonFinite(rayX) +
    countNonFinite(rayY) +
    countNonFinite(tSol) +
    countNonFinite(y1Num) +
    countNonFinite(error);
  const extra: string[] = [];
  if (nPoleCells > 0) {
    extra.push(
      `${nPoleCells} 个网格格子是真极点（|R|=+∞，帧掩码 abs_R_pole）：色标里画成饱和色 + 紫色标记，` +
        `占位值 ${poleFill === undefined ? "pole_fill" : formatNumber(poleFill, 6)} 是**显示上限不是测量值**；` +
        `不含极点的真最大值见 meta.abs_R_max_finite`,
    );
  }
  if (nSaturated > 0) {
    extra.push(
      `热力图对数色标 [${formatNumber(vmin, 3)}, ${formatNumber(vmax, 3)}] 之外还有 ${nSaturated} 个格子（色标上端取的是 meta.abs_R_max_finite=${formatNumber(maxFinite, 4)}，` +
        `出现这种格子说明帧 meta 与数组不自洽——已按数组的实际上色）`,
    );
  }
  if (nBoundaryOutside > 0) extra.push(`|R|=1 边界有 ${nBoundaryOutside}/${boundaryX.length} 个点在固定窗口外（隐式方法的稳定域无界），未绘制`);
  if (countMask(rayInside) < rayX.length) extra.push(`λh 扫描射线有 ${rayX.length - countMask(rayInside)}/${rayX.length} 个点在固定窗口外，未绘制`);
  if (numClip.nClipped > 0) extra.push(`数值解有 ${numClip.nClipped} 个样本超出 ±2 显示上限，已截断（真实 max|y_num| 见文本）`);
  if (pointVisible === false) extra.push("当前 λh 不在固定窗口内（帧 meta.point_visible=False）：主图上没有落点标记");
  if (displayCeiling !== undefined && yMagnitude !== undefined) {
    const touched = countAtCeiling(yMagnitude, displayCeiling);
    if (touched > 0) extra.push(`|y| 有 ${touched}/${yMagnitude.length} 个样本触到显示上限 ${formatNumber(displayCeiling, 3)}（帧已截断，真值见 meta.max_abs_y）`);
  }
  if (Math.abs(absRMaxFinite === undefined ? 0 : absRMaxFinite) > 0) {
    // 只是把"真最大值"写进设计里：饱和色的读法必须靠它，不能靠 1e6 的占位值。
    const idxMax = argmaxIndex(absR);
    if (idxMax >= 0 && absRPole[idxMax] > 0.5) {
      degraded.push("网格最大值出现在极点格子上：色标上限与「最不稳定」的判读一律以 meta.abs_R_max_finite 为准");
    }
  }
  applyOverflowNote(scene, extra.join("；"), nBad, { slot: SLOT.overflow, width });
  return scene;
};

// --------------------------------------------------------------------------- //
// 局部工具
// --------------------------------------------------------------------------- //

/** `levels` 里最接近 `target` 的层级（色标上限与 |R|=1 判定都按它，避免前端自己发明阈值）。 */
function nearestLevel(levels: ArrayLike<number>, target: number): number | undefined {
  let best: number | undefined;
  let bestD = Number.POSITIVE_INFINITY;
  for (let i = 0; i < levels.length; i += 1) {
    const v = levels[i];
    if (!Number.isFinite(v) || v <= 0) continue;
    const d = Math.abs(v - target);
    if (d < bestD) {
      bestD = d;
      best = v;
    }
  }
  return best;
}

/** 第一个"值 > level 且落在窗口内"的下标（阈值比较，不是重算放大因子）。 */
function firstAbove(values: ArrayLike<number>, level: number, inside: ArrayLike<number>): number {
  for (let i = 0; i < values.length; i += 1) {
    if (inside[i] > 0.5 && Number.isFinite(values[i]) && values[i] > level) return i;
  }
  return -1;
}
