/**
 * `stability_surface`（3.2 章）的 model：|R(z)| 模曲面 + 精确参数化的 |R|=1 边界。
 *
 * 面板与图层（顺序即绘制顺序，测试按此断言）
 * ------------------------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | heat | `main` | 二维数组 `abs_R` 直接上色（固定窗口 ±half_width） |
 * | 2 | contour | `main` | 网格等值线（帧给的 `levels`） |
 * | 3 | line | `main` | 精确参数化边界 z(θ)=ρ/σ（窗口外的点不画，计数报出） |
 * | 4 | scatter | `main` | **真极点格子**（`abs_R_pole` 掩码；单独颜色，不是数值） |
 * | 5 | line | `aux` | 边界自洽残差 \|\|R(z_θ)\|−1\|（实线） |
 * | 6 | line | `aux` | 两条独立路径之差逐格值（虚线） |
 * | 7 | scatter | `aux` | 两条曲线的最大值标记（位置由扫描数组得到，数值取自 meta） |
 *
 * 时间轴语义：**没有时间轴**（manifest `time.mode="none"`）：模型不读 `opts.t`、
 * 不引用 `tMax`，scene.t 只作为"这一帧是谁"的标签（帧 meta.t 恒为 0）。
 *
 * 三条刻意的决定
 * --------------
 * 1. **两条辅助量的横轴是"归一化序号"**，因为帧里它们本来就没有共同的物理横轴：`residual`
 *    沿边界 θ 扫了 2049 个点，`error` 是 101×101 个格子（行主序）。本模型把两者各自
 *    映射到 `i/(N−1)`（**纯显示几何**），纵轴共用一个对数轴，并在文本里写明这条横轴
 *    "不是 θ、不是 r"——把它画成一条物理轴才是撒谎。
 * 2. **极点格子上色取上端，不当数值**：`abs_R` 里极点位置的占位值是 `pole_fill=1e6`，
 *    直接上色会让"没有有限值"读成"最不稳定"。本模型把掩码格子画成色标上端 + 一枚单独颜色的
 *    标记，并把 `n_pole_cells`、`n_route_mismatch`（应当相等）与 `pole_note` 摆到文本里。
 * 3. **色标以 \|R\|=1 为几何中心**（`[1/max, max]`，对数）：白色=边界，蓝=衰减、红=放大；
 *    同时**一个格子都不截断**——帧声明的 `vmin/vmax`（本帧 0/4）会让窗口角上的 |R|≈1e4 全部饱和，
 *    那正是 manifest pitfalls 点名的坑。若数组真的越出本模型的色标范围，会计数并写进溢出说明。
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
  countMask,
  logRange,
  textBlock,
  textBottom,
  textTop,
  unitBudget,
} from "./_panel.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "3.2";
const VIZ = "stability_surface";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "|R(z)| 模曲面与 |R|=1 等高线";

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

/** 色标过道（`tools/raster.py::_draw_colorbar`：轴右 46px、宽 14px + 数值标签 + 居中说明）。 */
const COLORBAR_GUTTER = 46 + 14 + 52 + 42 + 8;

export const stabilitySurfaceModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  const absR = requireArray(frame, "abs_R");
  const absRPole = requireArray(frame, "abs_R_pole");
  const extent = requireArray(frame, "extent");
  const levels = requireArray(frame, "levels");
  const boundaryX = requireArray(frame, "boundary_x");
  const boundaryY = requireArray(frame, "boundary_y");
  const residual = requireArray(frame, "residual");
  const errorGrid = requireArray(frame, "error");
  const errorMark = requireArray(frame, "error_mark");

  const shape = frame.shapeOf("abs_R");
  if (shape.length !== 2) {
    throw new ModelInputError(`帧 ${CHAPTER}/${VIZ} 的 'abs_R' 形状 [${shape.join(", ")}] 不是二维：热力图需要 (ny, nx)`);
  }
  const ny = shape[0];
  const nx = shape[1];
  if (absR.length !== nx * ny) throw new ModelInputError(`帧 ${CHAPTER}/${VIZ} 的 'abs_R' 有 ${absR.length} 个元素，与 ${ny}×${nx} 不符`);
  if (absRPole.length !== absR.length) throw new ModelInputError(`帧 ${CHAPTER}/${VIZ} 的 'abs_R_pole' 长度 ≠ 'abs_R'`);
  if (errorGrid.length !== nx * ny) throw new ModelInputError(`帧 ${CHAPTER}/${VIZ} 的 'error' 长度 ${errorGrid.length} ≠ ${ny}×${nx}`);
  if (extent.length !== 4) throw new ModelInputError(`帧 ${CHAPTER}/${VIZ} 的 'extent' 长度 ≠ 4`);
  requireSameLength(boundaryX, boundaryY, "帧数组 boundary_x 与 boundary_y");
  requireSameLength(boundaryX, residual, "帧数组 boundary_x 与 residual");
  const [x0, x1, y0, y1] = [extent[0], extent[1], extent[2], extent[3]];
  if (!(x1 > x0) || !(y1 > y0)) throw new ModelInputError(`帧 ${CHAPTER}/${VIZ} 的 extent 必须严格递增`);
  const pitfalls = requirePitfalls(frame);

  const degraded: string[] = [];
  const half = metaNumber(frame, "half_width");
  const gridN = metaNumber(frame, "grid_n");
  const orderEff = metaNumber(frame, "order_effective");
  if (orderEff === undefined) degraded.push("帧 meta 缺 order_effective：标题只写方法族名");
  const absRMax = metaNumber(frame, "abs_R_max");
  const absRMin = metaNumber(frame, "abs_R_min");
  const vminMeta = metaNumber(frame, "vmin");
  const vmaxMeta = metaNumber(frame, "vmax");
  const nInside = metaNumber(frame, "n_inside");
  const nGrid = metaNumber(frame, "n_grid");
  const insideFraction = metaNumber(frame, "inside_fraction");
  const residualMax = metaNumber(frame, "residual_max");
  const routeDiffMax = metaNumber(frame, "route_diff_max");
  const edgeMax = metaNumber(frame, "edge_max");
  const nPoleCellsMeta = metaNumber(frame, "n_pole_cells");
  const nPoleEdge = metaNumber(frame, "n_pole_edge");
  const nRouteMismatch = metaNumber(frame, "n_route_mismatch");
  const poleFill = metaNumber(frame, "pole_fill");
  const aStable = (metaNumber(frame, "a_stable") ?? 0) > 0.5;
  const lStable = (metaNumber(frame, "l_stable") ?? 0) > 0.5;
  const angleDeg = metaNumber(frame, "angle_deg");
  const stiffLimit = metaNumber(frame, "stiff_limit");
  const endpointFinite = (metaNumber(frame, "endpoint_finite") ?? 0) > 0.5;
  const negRealEndpoint = metaNumber(frame, "negative_real_endpoint");
  const maxOnImag = metaNumber(frame, "max_on_imaginary_axis");
  const roundoffFloor = metaNumber(frame, "roundoff_floor");
  const methodName = stringOf(frame, "method_name");
  const route = stringOf(frame, "route");
  if (route === undefined) degraded.push("帧 strings 缺 route：不说明边界是用哪条路线算的");
  const note = stringOf(frame, "note");
  if (note === undefined) degraded.push("帧 strings 缺 note：状态角标不自行编造结论");
  const poleNote = stringOf(frame, "pole_note");
  const colourNote = stringOf(frame, "colour_note");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: metaNumber(frame, "t") ?? opts.t,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  const layout = chapterLayout(width, height, { rightShares: [1], gutter: COLORBAR_GUTTER, minRight: 160 });
  const mainRect = layout.main;
  const auxRect = layout.right[0];
  scene.axes.push(
    makeAxis("main", mainRect, {
      xlim: [x0, x1],
      ylim: [y0, y1],
      xlabel: "$\\operatorname{Re} z$",
      ylabel: axisYlabelOf(SPEC?.axes, "main", "$\\mathrm{Im}\\,z$"),
      title: axisTitleOf(SPEC?.axes, "main", "|R(z)|（固定窗口）与 |R|=1 等高线"),
      autoscaled: false,
    }),
  );
  // 副轴：两条 1e-16 量级的曲线 + 它们的最大值标记。横轴是**归一化序号**（见文件头第 1 条）。
  const auxLo = Math.min(
    positiveMin(residual) ?? Number.POSITIVE_INFINITY,
    positiveMin(errorGrid) ?? Number.POSITIVE_INFINITY,
    // 地板若是 0（降级帧）不能进对数范围：0 会让 niceLimits 当场抛错。
    roundoffFloor !== undefined && roundoffFloor > 0 ? roundoffFloor : Number.POSITIVE_INFINITY,
  );
  const auxHi = Math.max(finiteMax(residual) ?? 0, finiteMax(errorGrid) ?? 0, roundoffFloor ?? 0);
  const auxYlim: [number, number] =
    Number.isFinite(auxLo) && auxHi > 0 ? logRange(auxLo, auxHi) : [1e-18, 1];
  if (!(Number.isFinite(auxLo) && auxHi > 0)) degraded.push("residual/error 里没有 >0 的有限值：副轴用占位范围，曲线画不出来");
  scene.axes.push(
    makeAxis("aux", auxRect, {
      xlim: [0, 1],
      ylim: auxYlim,
      yscale: "log",
      xlabel: "归一化序号 $i/(N-1)$",
      ylabel: axisYlabelOf(SPEC?.axes, "aux", "残差"),
      title: axisTitleOf(SPEC?.axes, "aux", "边界残差与双路径之差"),
      autoscaled: true,
    }),
  );

  // —— 图层 1~2：模曲面 + 等值线 ——
  const maxFinite = absRMax ?? absMax(absR) ?? 1;
  const vmax = Math.max(maxFinite, (levels.length > 0 ? levels[0] : 1) * 1.000001, 1.000001);
  const vmin = 1 / vmax;
  const heatValues = new Float64Array(absR.length);
  let nPoleCells = 0;
  let nSaturated = 0;
  const poleIdx: number[] = [];
  for (let i = 0; i < absR.length; i += 1) {
    const isPole = absRPole[i] > 0.5;
    const v = absR[i];
    if (isPole) {
      nPoleCells += 1;
      poleIdx.push(i);
      heatValues[i] = vmax;
      continue;
    }
    if (!Number.isFinite(v)) {
      heatValues[i] = vmax;
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
      label: "$|R(z)|$ 对数色标，中点 $=1$",
    }),
  );
  const contourLevels = Array.from(levels).filter((v) => Number.isFinite(v) && v > 0);
  scene.layers.push(
    contourLayer("main", heatValues, nx, ny, [x0, x1, y0, y1], {
      levels: contourLevels,
      color: "#4a5568",
      width: 1,
      label: "网格等值线（分辨率有限）",
    }),
  );

  // —— 图层 3：精确参数化边界（窗口外不画，计数报账）——
  let nBoundaryOutside = 0;
  for (let i = 0; i < boundaryX.length; i += 1) {
    const bx = boundaryX[i];
    const by = boundaryY[i];
    if (!Number.isFinite(bx) || !Number.isFinite(by) || bx < x0 || bx > x1 || by < y0 || by > y1) nBoundaryOutside += 1;
  }
  const boundaryRuns = clipRuns(boundaryX, boundaryY, y0, y1).runs.filter((r) => r.x.every((v) => v >= x0 && v <= x1));
  for (const run of boundaryRuns) {
    scene.layers.push(lineLayer("main", series(run.x, run.y, pal.ink, { width: 2.4 })));
  }

  // —— 图层 4：真极点格子（单独颜色 + 单独说明；不当数值）——
  if (poleIdx.length > 0) {
    const px: number[] = [];
    const py: number[] = [];
    for (const i of poleIdx) {
      const j = Math.floor(i / nx);
      const ii = i % nx;
      px.push(x0 + ((x1 - x0) * ii) / Math.max(1, nx - 1));
      py.push(y0 + ((y1 - y0) * j) / Math.max(1, ny - 1));
    }
    scene.layers.push(
      scatterLayer("main", px, py, {
        r: 5,
        color: "#805ad5",
        label: "真极点格子 $|R|=+∞$（掩码 $abs_R_pole$；不是 $1e6$ 这样的数值）",
      }),
    );
  }

  // —— 图层 5~7：副轴（边界自洽残差 + 双路径之差）——
  const residualX = normalizedIndex(residual.length);
  const residualClip = clipRuns(residualX, residual, auxYlim[0], auxYlim[1], (v) => v > 0);
  for (const run of residualClip.runs) {
    scene.layers.push(lineLayer("aux", series(run.x, run.y, pal.residual, { width: 1.5 })));
  }
  const errorX = normalizedIndex(errorGrid.length);
  const gridClip = clipRuns(errorX, errorGrid, auxYlim[0], auxYlim[1], (v) => v > 0);
  for (const run of gridClip.runs) {
    scene.layers.push(lineLayer("aux", series(run.x, run.y, pal.curves[0], { width: 1, dash: [4, 3] })));
  }
  if (residualClip.runs.length === 0 && gridClip.runs.length === 0) degraded.push("两条辅助曲线在对数范围内都没有可画的点");
  const residualAt = argmaxIndex(residual);
  if (residualAt >= 0) {
    scene.layers.push(
      scatterLayer("aux", [residualX[residualAt]], [residual[residualAt]], {
        r: 4,
        color: pal.residual,
        label: "边界残差最大值（位置由数组扫描得到，数值见 $meta.residual_max$）",
      }),
    );
  }
  const gridAt = argmaxIndex(errorGrid);
  if (gridAt >= 0) {
    scene.layers.push(
      scatterLayer("aux", [errorX[gridAt]], [errorGrid[gridAt]], {
        r: 4,
        color: pal.curves[0],
        label: "双路径之差最大值（数值见 $meta.route_diff_max$）",
      }),
    );
  }
  // 帧的 error_mark 声明的就是这张网格的最大值：与数组扫描对一次账，不一致就降级说明。
  const gridMaxDeclared = errorMark.length > 0 ? errorMark[0] : undefined;
  if (gridMaxDeclared !== undefined && gridAt >= 0) {
    const scanned = errorGrid[gridAt];
    if (Math.abs(scanned - gridMaxDeclared) > Math.max(1e-300, Math.abs(gridMaxDeclared) * 1e-9)) {
      degraded.push(
        `帧 error_mark=${formatNumber(gridMaxDeclared, 4)} 与 error 数组扫描出的最大值 ${formatNumber(scanned, 4)} 不一致：标记画在数组最大值处，文本报 meta 值`,
      );
    }
  }

  // —— 文本 ——
  const head: string[] = [];
  const family = methodName ?? stringOf(frame, "method") ?? "?";
  head.push(`${family}${orderEff === undefined ? "" : `（k=${formatNumber(orderEff, 3)}）`}${half === undefined ? "" : `　窗口 ±${formatNumber(half, 3)}`}${gridN === undefined ? "" : `　网格 ${formatNumber(gridN, 3)}²`}`);
  head.push(
    `A-稳定=${aStable ? "是" : "否"}　L-稳定=${lStable ? "是" : "否"}` +
      (angleDeg === undefined ? "" : `　A(α) 角 ${formatNumber(angleDeg, 4)}°`) +
      (stiffLimit === undefined ? "" : `　刚性限制 ${formatNumber(stiffLimit, 4)}`),
  );
  head.push(
    `对数色标 [${formatNumber(vmin, 3)}, ${formatNumber(vmax, 3)}]，中点=|R|=1（蓝=衰减、红=放大）：` +
      clipText(colourNote ?? "暖色只表示 |R|>1，右半平面的真解本来就在涨", 44),
  );
  head.push(`帧声明的 vmin/vmax=${formatNumber(vminMeta ?? 0, 3)}/${formatNumber(vmaxMeta ?? 0, 3)}（线性）会让角上的 |R| 全饱和：本模型改用对数色标。`);
  const legend: string[] = [
    `粗黑=参数化边界 |R|=1（${boundaryX.length} 点，窗口外 ${nBoundaryOutside} 点未画）；细灰=网格等值线`,
    `真实 |R|∈[${formatNumber(absRMin ?? 0, 4)}, ${formatNumber(absRMax ?? 0, 4)}]　盒内 ${formatNumber(nInside ?? 0, 4)}/${formatNumber(nGrid ?? 0, 4)} 格在域内（${formatNumber((insideFraction ?? 0) * 100, 4)}%）` +
      (edgeMax === undefined ? "" : `　盒边最大 ${formatNumber(edgeMax, 4)}`),
    nPoleCells > 0
      ? `紫色=真极点格子（${nPoleCells} 个）：|R|=+∞，帧里 ${poleFill === undefined ? "占位值" : formatNumber(poleFill, 6)} 是显示上限不是数值`
      : `本帧网格无极点（abs_R_pole 全 0）`,
  ];
  // 图例一律塞进方形面板顶部：面板下方就是底轴轴名带与 pitfalls 面板，放不下第二块文本。
  scene.texts.push(
    ...textBlock(mainRect[0] + 6, textTop(mainRect), textBottom(mainRect), [...head, ...legend], pal.ink, unitBudget(mainRect[2]), "top"),
  );

  const auxLines: string[] = [];
  auxLines.push(
    `边界自洽残差 max=${formatNumber(residualMax ?? (residualAt >= 0 ? residual[residualAt] : 0), 4)}（${residual.length} 个 θ 点）` +
      `；双路径之差 max=${formatNumber(routeDiffMax ?? (gridAt >= 0 ? errorGrid[gridAt] : 0), 4)}（${errorGrid.length} 格）`,
  );
  if (nRouteMismatch !== undefined && nPoleCellsMeta !== undefined) {
    auxLines.push(
      `两路不同时有限的格子数 ${formatNumber(nRouteMismatch, 4)}，极点格数 ${formatNumber(nPoleCellsMeta, 4)}：` +
        (nRouteMismatch === nPoleCellsMeta
          ? "相等（只有极点上两路才都是 +∞、差无定义）——这两条曲线的意义就在这把账对上。"
          : "⚠ 不相等：两条独立路径本身有问题，别当成数值噪声。"),
    );
  }
  if (roundoffFloor !== undefined) auxLines.push(`地板 ${formatNumber(roundoffFloor, 3)}=u/2：两条曲线都在 1e-15 量级，那是舍入不是物理差异。`);
  if (nPoleCells > 0) {
    auxLines.push(
      `紫色=极点格（${nPoleCells} 个，z=α_k）：|R|=+∞，帧里 ${poleFill === undefined ? "pole_fill" : formatNumber(poleFill, 6)} 是**显示上限不是数值**，两条路径在那里都记地板值。`,
    );
    if (poleNote !== undefined) auxLines.push(`极点说明（帧 pole_note 摘要）：${clipText(poleNote, 58)}`);
  }
  auxLines.push(`横轴=归一化序号 i/(N−1)：实线沿 θ 扫 ${residual.length} 点、虚线把 ${nx}×${ny} 格按行主序展平（不是 θ、不是 r）。`);
  if (route !== undefined) auxLines.push(`边界路线：${route}`);
  scene.texts.push(...textBlock(auxRect[0] + 6, textTop(auxRect), textBottom(auxRect), auxLines, pal.muted, unitBudget(auxRect[2]), "top"));

  // —— 角标 ——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  scene.badges.push(
    placeBadge(badge("autoscale", `主轴：固定窗口 [${formatNumber(x0, 3)}, ${formatNumber(x1, 3)}]²（自适应会把"窄带"画成"一大片"）`, 8, 26), SLOT.axisMode, width),
  );
  scene.badges.push(placeBadge(badge("autoscale", "副轴：自适应 log（按本帧残差取整；会掩盖量级变化）", 8, 26), SLOT.auxAxis, width));
  const orderParam = metaNumber(frame, "order_param");
  const ordCrit = criticalOf(frame, "order", 3);
  if (orderParam !== undefined) {
    scene.badges.push(placeBadge(criticalBadge("阶数 k", orderParam, ordCrit, { beyond: "above" }), SLOT.critical, width));
  }
  const status: string[] = [];
  status.push(`A-稳定=${aStable ? "是" : "否"}${angleDeg === undefined ? "" : `（A(α) 角 ${formatNumber(angleDeg, 4)}°）`}`);
  if (endpointFinite) {
    status.push(`z→−∞ 端有限：|R|→${formatNumber(negRealEndpoint ?? 0, 4)}`);
  } else {
    status.push("z→−∞ 端 |R|→∞（该方法在这一端不 L-稳定）");
  }
  if (maxOnImag !== undefined) status.push(`虚轴上 |R| 最大 ${formatNumber(maxOnImag, 4)}`);
  if (note !== undefined) status.push(note);
  scene.badges.push(placeBadge(badge("status", clipBadgeText(status.join("；")), 8, 8), SLOT.status, width));
  if (nPoleCellsMeta !== undefined && nPoleCellsMeta !== nPoleCells) {
    degraded.push(`帧 meta.n_pole_cells=${formatNumber(nPoleCellsMeta, 4)} 与掩码点数 ${nPoleCells} 不一致：按掩码画（掩码是数据）`);
  }
  if (nPoleEdge !== undefined && nPoleEdge > 0) degraded.push(`帧 meta.n_pole_edge=${formatNumber(nPoleEdge, 4)}：盒边采样点里也有极点（本模型只画网格极点）`);
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  const nBad =
    countNonFinite(absR) + countNonFinite(boundaryX) + countNonFinite(boundaryY) + countNonFinite(residual) + countNonFinite(errorGrid);
  const extra: string[] = [];
  if (nPoleCells > 0) {
    extra.push(
      `${nPoleCells} 个格子是真极点（帧掩码 abs_R_pole）：热力图取色标上端 + 紫色标记，` +
        `占位值 ${poleFill === undefined ? "pole_fill" : formatNumber(poleFill, 6)} 是**显示上限不是测量值**；双路径之差在这些格子上记地板值`,
    );
  }
  if (nSaturated > 0) extra.push(`本模型色标 [${formatNumber(vmin, 3)}, ${formatNumber(vmax, 3)}] 之外还有 ${nSaturated} 个格子（按数组实际上色，说明帧 meta.abs_R_max 与数组不自洽）`);
  if (nBoundaryOutside > 0) extra.push(`|R|=1 边界有 ${nBoundaryOutside}/${boundaryX.length} 个点在固定窗口外，未绘制`);
  if ((metaNumber(frame, "overflow") ?? 0) > 0.5) extra.push("帧 meta.overflow=True：后端已标记本帧存在非有限值（由 *_pole 掩码显式替换）");
  if (countMask(absRPole) !== nPoleCells) extra.push("掩码 abs_R_pole 的取值不是 0/1：已按 >0.5 判为极点");
  applyOverflowNote(scene, extra.join("；"), nBad, { slot: SLOT.overflow, width });
  return scene;
};

// --------------------------------------------------------------------------- //
// 局部工具
// --------------------------------------------------------------------------- //

/** `i/(N−1)`：把一条扫描序列映射到 [0,1]（纯显示几何，不改变任何数值）。 */
function normalizedIndex(n: number): Float64Array {
  const out = new Float64Array(Math.max(0, n));
  for (let i = 0; i < n; i += 1) out[i] = n <= 1 ? 0 : i / (n - 1);
  return out;
}

/**
 * 对数轴上能用的小值：**只收 > 0 的有限值**。
 *
 * 帧里恰为 0 的样本（后端把地板值抬到 u/2，但降级帧或别的章节可能真的给 0）如果进了
 * `niceLimits(lo, hi, {log:true})`，会当场抛 `RangeError`——那是"画不出来"，不是"帧坏了"。
 */
function positiveMin(a: ArrayLike<number>): number | undefined {
  let best: number | undefined;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i];
    if (!Number.isFinite(v) || v <= 0) continue;
    if (best === undefined || v < best) best = v;
  }
  return best;
}

/** 把帧里的长字符串裁到 `maxUnits` 个显示宽度单位（CJK=2、ASCII=1）——纯显示处理，不改数值。 */
function clipText(body: string, maxUnits: number): string {
  const out: string[] = [];
  let w = 0;
  for (const ch of body) {
    const cw = (ch.codePointAt(0) ?? 0) > 0x2e80 ? 2 : 1;
    if (w + cw > maxUnits) return `${out.join("").trimEnd()}…`;
    out.push(ch);
    w += cw;
  }
  return body;
}

function finiteMax(a: ArrayLike<number>): number | undefined {
  let best: number | undefined;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i];
    if (!Number.isFinite(v)) continue;
    if (best === undefined || v > best) best = v;
  }
  return best;
}
