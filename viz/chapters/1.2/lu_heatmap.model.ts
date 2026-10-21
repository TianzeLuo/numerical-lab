/**
 * `lu_heatmap`（1.2 章）的 model：消元过程热力图 + 增长因子/前导块残差历史。
 *
 * 面板与图层（顺序即绘制顺序，测试按此断言）
 * ------------------------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | heat | `main` | `matrix` 的 log10\|·\|（**行序翻转**：矩阵行 0 画在面板上方） |
 * | 2 | line | `main` | 已完成前导块的边界（两条虚线；`k = n−1` 时不画） |
 * | 3 | scatter | `main` | `indicator == 1` 的格子：已写回乘子的下三角 |
 * | 4 | scatter | `main` | `indicator == 2` 的格子：当前主元行/列（半透明红带） |
 * | 5 | scatter | `main` | 主元位置 `(k,k)` 的实心点 |
 * | 6 | line | `aux` | 当前历史样本的游标（虚线竖线） |
 * | 7 | line | `aux` | `rho_history`（log 纵轴） |
 * | 8 | line | `aux` | `residual_history`（log 纵轴，`u/2` 地板） |
 * | 9 | scatter | `aux` | 两枚当前值标记（ρ 与残差各一） |
 *
 * 角标槽位（固定，见 `SLOT`）：0 来源 / 1 主轴固定 / 2 副轴自适应 / 3 `dominance` 临界 /
 * 4 状态（奇异、ρ>10、ρ 越过 2^(n−1)、帧 note）/ 5 前端预览（仅 `backend+preview`）/
 * 6 溢出 / 7 降级说明。
 *
 * 几条刻意的决定
 * --------------
 * 1. **行序在模型里翻转，不动轴**：`heat` 图层的约定是"第 0 行 = `extent[2]`（y 最小）"，
 *    而矩阵的习惯是行 0 在上。`validateScene` 又要求 `ylim[0] < ylim[1]`（不许反向轴），
 *    所以只能把行序翻过来写进 `values`：`values[(n−1−r)·n + c] = matrix[r·n + c]`。
 *    这样下三角（乘子）在画面里仍然是下三角——不翻转的话它就跑到右上角去了。
 * 2. **配色前的 log10 与地板都在模型里做**：`heatLayer` 拿到的已经是 log10 值，
 *    因此 `logScale: false`（再取一次对数就成了 loglog，是另一张图）。0 与 |x|<1e-16
 *    一律压到 `LOG_FLOOR = −16`，并在色标标题与文本里写明"地板"——否则"一片深蓝"
 *    会被读成"接近零"，而 −16 与 −2 在这张图里只是深浅不同（见 `frame.pitfalls`）。
 * 3. **extent 直接取整条轴范围**：热力图的 `nx = n` 个像素铺满 `n` 个数据单位时，
 *    像素中心正好落在整数格点 `x = c`、`y = n−1−r` 上（浏览器快路径与 `tools/raster.py`
 *    都是整幅拉伸），因此 `indicator` 的散点直接用整数行列坐标就能与格子对齐。
 * 4. **主元高亮用 `scatter` 而不是分隔线**：`indicator` 是逐格掩码（0/1/2），
 *    散点画的正是后端点名的那些格子；分隔线只能画出前导块的几何边界，
 *    画不出"哪一列的乘子已经写回"。两者都画：线标边界、点标格子。
 * 5. **`meta.step` 与历史下标差一**：帧里的历史是 `[初始矩阵, 第0步后, …, 第n−2步后]`
 *    （`len = n`），第 `k` 步之后的快照是 `history[k+1]`，`meta.rho`/`meta.residual`
 *    也是取那一个（后端 `idx = min(k+1, len−1)`）。所以游标画在 `j = min(k+1, len−1)`，
 *    轴标签写清"j = k+1 是第 k 步之后"，不假装 `step_grid[meta.step]` 就是当前样本。
 * 6. **文本一行一个 `text` 层**：`render2d.ts::drawText` 不拆 `\n`（`raster.py` 拆），
 *    所以模型不写多行字符串，避免浏览器与光栅器的排版分叉。
 */

import { formatNumber, niceLimits } from "../../shared/src/axes.js";
import { countNonFinite, dataToPixel } from "../../shared/src/geometry.js";
import type { Frame } from "../../shared/src/frame.js";
import {
  ModelInputError,
  applyOverflow,
  canvasSize,
  clamp,
  clipBadgeText,
  criticalOf,
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
  sampleLinear,
  sceneShell,
  segmentedSeries,
  stackedPanels,
  stringOf,
  textWidthUnits,
  type Rect,
  type SceneModel,
} from "../../shared/src/model.js";
import {
  axisModeBadge,
  badge,
  criticalBadge,
  heatLayer,
  lineLayer,
  makeAxis,
  pitfallsBand,
  previewBadge,
  scatterLayer,
  series,
  sourceBadge,
  text,
  type TextSpec,
} from "../../shared/src/scene.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "1.2";
const VIZ = "lu_heatmap";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "消元过程热力图：增长因子与残差";
/** manifest 里 `dominance.critical = [1.0]`、`beyond = "below"`；`frame.critical.dominance` 缺失时的回落值。 */
const DOMINANCE_CRITICAL = 1.0;
/** manifest pitfalls 点名的"ρ 大到该说话"的门槛。 */
const RHO_WARN = 10;

/**
 * 固定角标槽位（见 `model.ts::BADGE_LAYOUT`）。
 * 长文本角标（溢出、降级）各占一行的左列，右侧留空，避免与邻居叠字。
 */
const SLOT = {
  source: 0,
  axisMode: 1,
  auxAxis: 2,
  critical: 3,
  status: 4,
  /** 仅当帧声明 `backend+preview` 时出现（本章的帧都是 backend）。 */
  preview: 5,
  overflow: 6,
  degraded: 7,
} as const;

/** 色标下端：log10|元素| 的地板（0 与 |x| < 1e-16 都画在这里）。 */
const LOG_FLOOR = -16;

export const luHeatmapModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需字段：缺一个就抛 ModelInputError（宁可当场炸，也不画半张图）——
  const matrix = requireSquareMatrix(frame, "matrix");
  const indicator = requireSquareMatrix(frame, "indicator");
  const rhoHistory = requireArray(frame, "rho_history");
  const residualHistory = requireArray(frame, "residual_history");
  const stepGrid = requireArray(frame, "step_grid");
  if (indicator.rows !== matrix.rows || indicator.cols !== matrix.cols) {
    throw new ModelInputError(
      `帧 ${CHAPTER}/${VIZ} 的 indicator 形状 ${indicator.rows}×${indicator.cols} 与 matrix ${matrix.rows}×${matrix.cols} 不一致：` +
        "两者必须逐格对应（拒绝画错位的高亮）",
    );
  }
  if (rhoHistory.length !== stepGrid.length || residualHistory.length !== stepGrid.length) {
    throw new ModelInputError(
      `帧 ${CHAPTER}/${VIZ} 的历史长度不一致：rho_history=${rhoHistory.length}、residual_history=${residualHistory.length}、` +
        `step_grid=${stepGrid.length}（三者必须同步长，才能共用同一条横轴）`,
    );
  }
  const pitfalls = requirePitfalls(frame);

  // —— 可选字段：缺了就降级（记进 degraded，最终挂成一个 status 角标）——
  const degraded: string[] = [];
  const tMeta = metaNumber(frame, "t");
  if (tMeta === undefined) degraded.push("帧 meta 缺 t：scene.t 改用请求 t（两者可能不同）");
  const tNow = tMeta ?? opts.t;
  const n = matrix.rows;
  const nMeta = metaNumber(frame, "n");
  if (nMeta === undefined) degraded.push("帧 meta 缺 n：矩阵规模按 matrix 的形状给出（形状是数据，不是猜的）");
  else if (Math.round(nMeta) !== n) degraded.push(`帧 meta.n=${formatNumber(nMeta, 6)} 与 matrix 形状 n=${n} 不一致：按矩阵形状画`);
  const k = metaNumber(frame, "step");
  if (k === undefined) degraded.push("帧 meta 缺 step：不画当前步标记（不猜是第几步）");
  const nSteps = metaNumber(frame, "n_steps");
  if (nSteps === undefined) degraded.push("帧 meta 缺 n_steps：不显示总步数");
  const dominance = metaNumber(frame, "dominance");
  if (dominance === undefined) degraded.push("帧 meta 缺 dominance：不标对角占优临界值（不猜）");
  const rho = metaNumber(frame, "rho");
  if (rho === undefined) degraded.push("帧 meta 缺 rho：数值行不显示 ρ（不重算）");
  const rhoBound = metaNumber(frame, "rho_bound");
  if (rhoBound === undefined) degraded.push("帧 meta 缺 rho_bound：不比较 ρ 与 2^(n−1)");
  const residual = metaNumber(frame, "residual");
  if (residual === undefined) degraded.push("帧 meta 缺 residual：数值行不显示当前残差（不重算）");
  const residualFinal = metaNumber(frame, "residual_final");
  const residualFloor = metaNumber(frame, "residual_floor");
  const nExactZero = metaNumber(frame, "residual_exact_zero_steps");
  const nPivots = metaNumber(frame, "n_pivots");
  const rowSwaps = metaNumber(frame, "row_swaps");
  const ops = metaNumber(frame, "ops");
  const pivotMaxAbs = metaNumber(frame, "pivot_max_abs");
  const pivotMinAbs = metaNumber(frame, "pivot_min_abs");
  const singular = (metaNumber(frame, "singular") ?? 0) > 0.5;
  const pivotName = stringOf(frame, "pivot");
  if (pivotName === undefined) degraded.push("帧 strings 缺 pivot：不显示主元策略名（不按参数猜）");
  const pivotKey = stringOf(frame, "pivot_key");
  const note = stringOf(frame, "note");
  if (note === undefined) degraded.push("帧 strings 缺 note：状态角标不自行编造结论");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: tNow,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  // —— 排版：左侧方形矩阵热力图，右侧历史曲线，中间留出**色标过道** ——
  // 为什么不用现成的 `columnsLayout`：
  //  1. 热力图是矩阵，"上下三角"只有在方格子里才读得出来（宽扁面板会把格子拉成矩形）；
  //  2. `tools/raster.py` 会给每个 heat 图层在**轴右侧 46px 起**自动配一条宽 14px 的色标
  //     （Scene 契约里没有色标节点，这是光栅器的加法），数值标签再往右 ~52px；
  //     而 `render2d.ts` 目前**不画**色标（`drawColorbar` 导出了但 `renderScene` 没调用，
  //     见交付说明）。两种渲染器的画面因此天生不同——留出这条 162px 的过道，
  //     色标就不会压到右列的曲线上；浏览器里这条过道就是空白。
  // 底部 pitfalls 的文案长度会变：把内容底边抬到它的上方，再给轴下说明留 42px。
  // 不这么做时，改一段长文案就会让最后一行说明压进 pitfalls 面板（规则 8 会当场拒绝）。
  const band = pitfallsBand({ canvas: { w: width, h: height }, pitfalls });
  const footer = band === null ? 102 : Math.max(102, height - band[1] + 42);
  const base = stackedPanels(width, height, { footer });
  // 方形边长：内容高度还要分一条带留给轴下的图例（render2d 的 xlabel 默认画在轴下 29px，
  // 这里把主轴 xlabel 置空、改成图例，所以先扣掉 46px）。
  const side = Math.max(110, Math.floor(Math.min(340, base.bottom - base.top - 46)));
  const mainRect: Rect = [base.left, base.top, side, side];
  const COLORBAR_GUTTER = 46 + 14 + 52 + 42 + 8; // 偏移 + 色标宽 + 数值标签 + 右列 ylabel + 余量
  const auxX = mainRect[0] + side + COLORBAR_GUTTER;
  // 副轴高度扣掉 62px = 轴自己的两条装饰带（刻度标签 bottom+4..15、xlabel bottom+16..28）
  // + 模型手写的"整段统计/末步残差"块（两行 30px）。少扣一点就会把数值块压进 xlabel 带：
  // 规则 8 会当场报违规（这不是理论风险，前两版就是这么挂的）。
  const auxRect: Rect = [auxX, base.top, Math.max(160, base.left + base.contentWidth - auxX), base.bottom - base.top - 62];

  // 矩阵格点坐标：数据坐标 x = 列 c、y = n−1−行 r（见文件头第 1、3 条）。
  const lim: [number, number] = [-0.5, n - 0.5];
  const mainAxis = makeAxis("main", mainRect, {
    xlim: lim,
    ylim: lim,
    // 轴下的位置留给图例（含色标说明）；"列 j" 的含义写进 head 行。
    xlabel: "",
    ylabel: "行 $i$（0 在上）",
    title: axisTitle("main", "$\\log_{10}|M|$（当前工作矩阵）"),
    autoscaled: false,
    xticks: integerTicks(n),
    yticks: integerTicks(n),
  });

  // 副轴（manifest 声明 `autoscaled: true`、`scale: "log"`）：ρ 与残差共用一条 log 纵轴。
  // 范围只由两者的**正有限值**决定（对数轴画不出 ≤0）。
  const rhoPos = finitePositiveMinMax(rhoHistory);
  const resPos = finitePositiveMinMax(residualHistory);
  const auxLo = Math.min(rhoPos?.[0] ?? Number.POSITIVE_INFINITY, resPos?.[0] ?? Number.POSITIVE_INFINITY);
  const auxHi = Math.max(rhoPos?.[1] ?? Number.NEGATIVE_INFINITY, resPos?.[1] ?? Number.NEGATIVE_INFINITY);
  const auxYlim: [number, number] =
    Number.isFinite(auxLo) && Number.isFinite(auxHi) ? niceLimits(auxLo, auxHi, 7, { log: true, pad: 0.12 }) : [1e-16, 1];
  if (!(Number.isFinite(auxLo) && Number.isFinite(auxHi))) {
    degraded.push("rho_history/residual_history 里没有 >0 的有限值：副轴用占位范围 [1e-16, 1]，不画曲线");
  }
  const auxLen = stepGrid.length;
  const auxAxis = makeAxis("aux", auxRect, {
    xlim: [-0.5, Math.max(0.5, auxLen - 0.5)],
    ylim: auxYlim,
    yscale: "log",
    xlabel: "历史样本 $j$（$j=0 为初始矩阵，$j=k+1 为第 $k$ 步之后）",
    ylabel: axisYlabel("aux", "$\\rho$ 与 $\\|PA-LU\\|/\\|A\\|$"),
    title: axisTitle("aux", "增长因子 ρ(k) 与残差"),
    autoscaled: true,
    xticks: integerTicks(auxLen),
  });
  scene.axes.push(mainAxis, auxAxis);

  // —— 图层 1：log10|matrix| 的 heat（行序翻转；0 与 <1e-16 压到地板）——
  const cmap = buildLogMagnitude(matrix.values, n);
  scene.layers.push(
    heatLayer("main", cmap.values, n, n, [lim[0], lim[1], lim[0], lim[1]], {
      cmap: "viridis",
      vmin: LOG_FLOOR,
      vmax: cmap.vmax,
      logScale: false, // 值已经是 log10；再开 logScale 就是 loglog 了（见文件头第 2 条）
      // 这个 label 只有**光栅器**用得上：tools/raster.py 会把 heat 图层的 label
      // 居中画在自动色标条下方，太长会糊到隔壁面板上；完整说明放在下面的图例里。
      label: "矩阵元素（$\\log_{10}$）",
    }),
  );

  // —— 图层 2：前导块边界（两条虚线）——
  // 只标"几何边界"；哪一格被写回乘子由下面的散点回答（见文件头第 4 条）。
  if (k !== undefined && k < n - 1) {
    const edge = k + 0.5;
    const edgeY = n - 1 - k - 0.5;
    scene.layers.push(
      lineLayer("main", [
        series([edge, edge], [lim[0], lim[1]], pal.reference, { dash: [4, 3], width: 1, label: "已完成的前导块边界" }),
        series([lim[0], lim[1]], [edgeY, edgeY], pal.reference, { dash: [4, 3], width: 1 }),
      ]),
    );
  }

  // —— 图层 3~5：indicator 的三类格子 ——
  const cellPx = Math.min(mainRect[2], mainRect[3]) / n;
  const doneX: number[] = [];
  const doneY: number[] = [];
  const activeX: number[] = [];
  const activeY: number[] = [];
  for (let r = 0; r < n; r += 1) {
    for (let c = 0; c < n; c += 1) {
      const flag = indicator.values[r * n + c];
      if (flag >= 2) {
        activeX.push(c);
        activeY.push(n - 1 - r);
      } else if (flag >= 1) {
        doneX.push(c);
        doneY.push(n - 1 - r);
      }
    }
  }
  if (doneX.length > 0) {
    scene.layers.push(
      scatterLayer("main", doneX, doneY, {
        r: Math.max(1.2, cellPx * 0.2),
        color: pal.curves[0],
        alpha: 0.55,
        label: "已写回乘子的下三角（$indicator=1$）",
      }),
    );
  }
  if (activeX.length > 0) {
    scene.layers.push(
      scatterLayer("main", activeX, activeY, {
        r: Math.max(1.5, cellPx * 0.4),
        color: pal.marker,
        alpha: 0.18,
        label: "当前主元行/列（$indicator=2$）",
      }),
    );
  }
  if (k !== undefined && k < n) {
    scene.layers.push(
      scatterLayer("main", [k], [n - 1 - k], {
        r: Math.max(2, cellPx * 0.1),
        color: pal.marker,
        label: `主元 (${k},${k}) 的位置`,
      }),
    );
  }

  // —— 图层 6~9：副轴（规则 3 的误差场：同屏给出残差）——
  // 历史下标与 meta.step 差一：第 k 步之后的快照是 history[k+1]（见文件头第 5 条）。
  const jNow = k === undefined ? undefined : clamp(Math.round(k) + 1, 0, Math.max(0, auxLen - 1));
  if (jNow !== undefined) {
    scene.layers.push(
      lineLayer("aux", series([jNow, jNow], [auxAxis.ylim[0], auxAxis.ylim[1]], pal.cursor, { dash: [3, 3], width: 1 })),
    );
  }
  const rhoSeries = segmentedSeries(stepGrid, rhoHistory, pal.curves[0], {
    width: 1.8,
    label: "$ρ(j)$ 增长因子（逐步实测）",
    accept: (v) => v > 0,
  });
  if (rhoSeries.length > 0) scene.layers.push(lineLayer("aux", rhoSeries));
  else degraded.push(`帧 rho_history 的 ${rhoHistory.length} 个样本没有一个 > 0：log 轴上画不出 ρ 曲线`);
  const resSeries = segmentedSeries(stepGrid, residualHistory, pal.residual, {
    width: 1.4,
    dash: [6, 3],
    label: "前导块残差（逐步实测，0 以 $u/2$ 显示）",
    accept: (v) => v > 0,
  });
  if (resSeries.length > 0) scene.layers.push(lineLayer("aux", resSeries));
  else degraded.push(`帧 residual_history 的 ${residualHistory.length} 个样本没有一个 > 0：log 轴上画不出残差曲线`);

  const rhoNow = jNow === undefined ? undefined : sampleLinear(stepGrid, rhoHistory, jNow);
  const resNow = jNow === undefined ? undefined : sampleLinear(stepGrid, residualHistory, jNow);
  const markX: number[] = [];
  const markY: number[] = [];
  if (jNow !== undefined && rhoNow !== undefined && rhoNow > 0) {
    markX.push(jNow);
    markY.push(rhoNow);
  }
  if (jNow !== undefined && resNow !== undefined && resNow > 0) {
    markX.push(jNow);
    markY.push(resNow);
  }
  if (markX.length > 0) {
    scene.layers.push(scatterLayer("aux", markX, markY, { r: 4, color: pal.marker, label: "当前步的 $ρ$ 与残差" }));
    const last = markX.length - 1;
    const [mx, my] = dataToPixel(auxAxis, markX[last] as number, markY[last] as number);
    if (Number.isFinite(mx) && Number.isFinite(my)) {
      const onRight = mx > auxRect[0] + auxRect[2] * 0.6;
      scene.texts.push(
        text(
          mx + (onRight ? -8 : 8),
          clamp(my + 8, auxRect[1] + 4, auxRect[1] + auxRect[3] - 20),
          `j = ${formatNumber(jNow as number, 4)}`,
          { role: "note", size: 11, color: pal.marker, align: onRight ? "right" : "left" },
        ),
      );
    }
  }

  // —— 文本（一行一个 text 层：见文件头第 6 条）——
  const summary = paramSummary(frame, opts, [
    { name: "n", label: "$n$" },
    { name: "dominance", label: "对角占优比" },
  ]);
  degraded.push(...summary.degraded);
  const head: string[] = [summary.text];
  if (pivotName !== undefined) head.push(`主元=${pivotName}`);
  head.push("列已按当前顺序重排");
  if (k !== undefined) {
    head.push(nSteps === undefined ? `step=${formatNumber(k, 4)}` : `step=${formatNumber(k, 4)}/${formatNumber(nSteps, 4)}`);
  }
  if (rho !== undefined) head.push(`ρ=${formatNumber(rho, 4)}`);
  if (residual !== undefined) head.push(`残差=${formatNumber(residual, 3)}`);
  // 主场：矩阵面板 340px 宽 → 约 46 个显示单位（size 11 时 1 单位 ≈ 7px）
  const mainBudget = unitBudget(mainRect[2]);
  scene.texts.push(...textBlock(mainRect[0] + 6, mainRect[1] + 4, mainRect[1], [head.join("　")], pal.muted, mainBudget, "top"));

  const legend: string[] = [
    `蓝=已写回乘子（${doneX.length} 格）　红=当前主元行/列（${activeX.length} 格）`,
    `虚线=已完成前导块的边界　色标：log10|元素|`,
    `地板 −16：本帧 ${cmap.nZero} 个 0、${cmap.nFloor} 个非零但低于地板` +
      (cmap.nNonFinite > 0 ? `、${cmap.nNonFinite} 个非有限（灰色同时意味着"没有数据"）` : ""),
  ];
  scene.texts.push(
    ...textBlock(
      mainRect[0] + 6,
      mainRect[1] + mainRect[3] + 6,
      base.bottom + 20,
      legend,
      cmap.nNonFinite > 0 ? pal.warn : pal.muted,
      mainBudget,
      "bottom",
    ),
  );

  const auxLines: string[] = [];
  if (note !== undefined) auxLines.push(`帧 note：${note}`);
  if (residualFloor !== undefined) {
    auxLines.push(
      `残差地板 ${formatNumber(residualFloor, 3)}（u/2）` +
        (nExactZero === undefined ? "" : `；本帧 ${formatNumber(nExactZero, 4)}/${residualHistory.length} 个样本恰好为 0，已按地板显示`),
    );
  }
  auxLines.push("两条曲线共用这一条 log 纵轴：ρ 在 1~1e2、残差贴在 u 附近，相差十几个数量级，ρ 的起伏因此被压平——");
  auxLines.push("量级差本身就是结论（规则 3 要求误差/残差与主图同屏出现），要看 ρ 的变化请读上面的数值行。");
  // 这条曾经写着"完全主元下残差按 P·A 取块、数值接近 O(1)"——那是**后端的 bug**
  // （最终的置换被当成了当步的置换），已经修好：现在逐步残差取的是当步的参照块
  // （`numcore.linalg.lu.leading_block_residuals`），三种策略都是 u 量级。
  // 模型侧不留过时说法：曲线本身已经能说明问题，不需要再替后端解释。
  if (pivotKey === "complete") {
    auxLines.push("完全主元也走同一把尺子：残差取**当步**前导块的参照矩阵（不是最终置换），三种策略都应是 u 量级。");
  }
  const auxBudget = unitBudget(auxRect[2]);
  scene.texts.push(
    ...textBlock(auxRect[0] + 6, auxRect[1] + 4, auxRect[1] + auxRect[3] - 6, auxLines, pivotKey === "complete" ? pal.warn : pal.muted, auxBudget, "top"),
  );

  const auxNum: string[] = [];
  if (nPivots !== undefined) auxNum.push(`n_pivots=${formatNumber(nPivots, 4)}`);
  if (rowSwaps !== undefined) auxNum.push(`row_swaps=${formatNumber(rowSwaps, 4)}`);
  if (ops !== undefined) auxNum.push(`ops=${formatNumber(ops, 6)}`);
  if (residualFinal !== undefined) auxNum.push(`残差(末步)=${formatNumber(residualFinal, 3)}`);
  if (pivotMinAbs !== undefined && pivotMaxAbs !== undefined) {
    auxNum.push(`|主元|∈[${formatNumber(pivotMinAbs, 3)}, ${formatNumber(pivotMaxAbs, 3)}]（整段 U 的对角，帧 meta）`);
  }
  if (auxNum.length > 0) {
    // 放在副轴**下方**而不是轴内底部：游标标签（`j = k`）被夹在轴内（上限 轴底−20），
    // 两者若都在轴内就会在最后几步叠在一起（实测 t=0 与 t=3 真重叠）。
    scene.texts.push(
      ...textBlock(auxRect[0] + 6, auxRect[1] + auxRect[3] + 32, base.bottom + 20, [auxNum.join("　")], pal.muted, auxBudget, "top"),
    );
  }

  // —— 角标（固定槽位）——
  // 规则 5：恒挂 source；预览型数据源另挂 preview（kind 各表一意，见 scene.ts）。
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  if (scene.dataSource.startsWith("backend+preview")) {
    scene.badges.push(placeBadge(previewBadge("拖动时按显示空间插值"), SLOT.preview, width));
  }
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(
      badge("autoscale", "副轴：自适应（log 纵轴按本帧 ρ 与残差取整；自适应会掩盖量级变化）", 8, 8),
      SLOT.auxAxis,
      width,
    ),
  );
  if (dominance !== undefined) {
    scene.badges.push(
      placeBadge(
        criticalBadge("dominance", dominance, criticalOf(frame, "dominance", DOMINANCE_CRITICAL), { beyond: "below" }),
        SLOT.critical,
        width,
      ),
    );
  }
  const status: string[] = [];
  if (singular) status.push("⚠ 遇到零主元：矩阵在给定策略下奇异（帧 meta.singular）");
  if (rho !== undefined && rhoBound !== undefined && rho > rhoBound) {
    status.push(`⚠ ρ=${formatNumber(rho, 4)} 已超过理论上界 2^(n−1)=${formatNumber(rhoBound, 4)}`);
  } else if (rho !== undefined && rho > RHO_WARN) {
    status.push(
      `⚠ ρ=${formatNumber(rho, 4)} > ${RHO_WARN}：增长已经明显` +
        (rhoBound === undefined ? "" : `（界 2^(n−1)=${formatNumber(rhoBound, 4)}）`),
    );
  }
  if (note !== undefined) status.push(note);
  if (status.length === 0) status.push("帧未给出可判读的状态（预期有 meta.singular 与 strings.note）");
  scene.badges.push(placeBadge(badge("status", clipBadgeText(status.join("；")), 8, 8), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  // 规则 1：图层越界 + 帧内非有限值，两者都要进 overflow 与角标。
  const nBad =
    countNonFinite(matrix.values) +
    countNonFinite(indicator.values) +
    countNonFinite(rhoHistory) +
    countNonFinite(residualHistory) +
    countNonFinite(stepGrid);
  applyOverflow(scene, nBad, { slot: SLOT.overflow, width });
  return scene;
};

// --------------------------------------------------------------------------- //
// 局部工具
// --------------------------------------------------------------------------- //

/** 方阵：`requireArray`（缺失时抛 `ModelInputError` 并列出可用数组）+ 形状必须是 `n×n`。 */
function requireSquareMatrix(frame: Frame, name: string): { rows: number; cols: number; values: ArrayLike<number> } {
  const values = requireArray(frame, name);
  const shape = frame.shapeOf(name);
  if (shape.length !== 2) {
    throw new ModelInputError(`帧 ${CHAPTER}/${VIZ} 的 '${name}' 形状 [${shape.join(", ")}] 不是二维：矩阵热力图需要 n×n`);
  }
  const rows = shape[0] as number;
  const cols = shape[1] as number;
  if (rows !== cols) throw new ModelInputError(`帧 ${CHAPTER}/${VIZ} 的 '${name}' 是 ${rows}×${cols}，不是方阵`);
  if (values.length !== rows * cols) {
    throw new ModelInputError(`帧 ${CHAPTER}/${VIZ} 的 '${name}' 有 ${values.length} 个元素，与 ${rows}×${cols} 不符`);
  }
  return { rows, cols, values };
}

/** `log10|元素|` + 地板；同时返回色标上端与三类计数。行序**翻转**（矩阵行 0 → 面板上方）。 */
function buildLogMagnitude(
  matrix: ArrayLike<number>,
  n: number,
): { values: Float64Array; vmax: number; nZero: number; nFloor: number; nNonFinite: number } {
  const values = new Float64Array(n * n);
  let maxLog = Number.NEGATIVE_INFINITY;
  let nZero = 0;
  let nFloor = 0;
  let nNonFinite = 0;
  for (let r = 0; r < n; r += 1) {
    for (let c = 0; c < n; c += 1) {
      const v = matrix[r * n + c];
      const at = (n - 1 - r) * n + c; // 翻转行序：见文件头第 1 条
      if (!Number.isFinite(v)) {
        nNonFinite += 1;
        values[at] = LOG_FLOOR;
        continue;
      }
      const a = Math.abs(v);
      if (a === 0) {
        nZero += 1;
        values[at] = LOG_FLOOR;
        continue;
      }
      const lg = Math.log10(a);
      if (lg < LOG_FLOOR) nFloor += 1;
      values[at] = lg < LOG_FLOOR ? LOG_FLOOR : lg;
      if (lg > maxLog) maxLog = lg;
    }
  }
  // 色标上端取半十倍程，且至少到 1（|元素| = 10）：既压住最大元素，刻度又是好看的数。
  const vmax = Number.isFinite(maxLog) ? Math.max(1, Math.ceil(maxLog * 2) / 2) : 1;
  return { values, vmax, nZero, nFloor, nNonFinite };
}

/** 下标刻度（0..n−1；n 大时按步进取样，避免刻线糊成一片）。 */
function integerTicks(n: number): number[] {
  if (n <= 0) return [];
  const step = Math.max(1, Math.ceil(n / 8));
  const out: number[] = [];
  for (let i = 0; i < n; i += step) out.push(i);
  return out;
}

/**
 * 面板宽度 → 显示宽度单位预算（size 11 时 1 单位 ≈ 7px；`textWidthUnits` 的 CJK=2/ASCII=1 口径）。
 * 留 12px 边距再折行，保证两个渲染器都不会自动折（见 `textBlock` 的注释）。
 */
function unitBudget(panelWidthPx: number): number {
  return Math.max(24, Math.floor((panelWidthPx - 12) / 7));
}

/** manifest 里该轴的标题（文案直接取 manifest，缺省才用回落值）。 */
function axisTitle(id: string, fallback: string): string {
  return SPEC?.axes.find((a) => a.id === id)?.title ?? fallback;
}

/** manifest 里该轴的 ylabel。 */
function axisYlabel(id: string, fallback: string): string {
  return SPEC?.axes.find((a) => a.id === id)?.ylabel ?? fallback;
}

/**
 * 一段说明文字 → 若干 `text` 层，**一行一个**（`render2d.ts::drawText` 不拆 `\n`，见文件头第 6 条）。
 *
 * 为什么要自己折行：两个渲染器都支持 `maxWidth` 自动折行，但折出来的续行**不占**模型这一侧
 * 的行距预算（光栅器的续行间距还是 `cell_h + 4`，比我排的 15px 大），于是续行会压到下一行文字上。
 * 所以模型先按"显示宽度单位"（CJK=2、ASCII=1，口径同 `model.ts::textWidthUnits`）把每段折到
 * `budget` 以内，再依次排开：一行就是一层，两个渲染器都不会再折。
 *
 * `anchor`：`"top"` 从 `top` 往下排；`"bottom"` 让整块贴着 `bottom`（行数由折行结果决定）。
 */
function textBlock(
  x: number,
  top: number,
  bottom: number,
  lines: readonly string[],
  color: string,
  budget: number,
  anchor: "top" | "bottom",
): TextSpec[] {
  const size = 11;
  const rowH = 15;
  let pieces = lines.flatMap((body) => wrapUnits(body, budget));
  // 画布小的时候宁可**截断并说明**，也不让文字叠在一起（叠字等于看不见，见规则 1 的精神）：
  // 能放下的行数由 top/bottom 之间的净空决定，最后一行留给"还有 N 行"。
  if (bottom > top) {
    const maxRows = Math.max(1, Math.floor((bottom - top) / rowH) + 1);
    if (pieces.length > maxRows) {
      const kept = Math.max(1, maxRows - 1);
      pieces = [...pieces.slice(0, kept), `…（另有 ${pieces.length - kept} 行，画布放不下）`];
    }
  }
  const y = anchor === "top" ? top : Math.max(top, bottom - (pieces.length - 1) * rowH);
  return pieces.map((body, i) => text(x, y + i * rowH, body, { role: "note", size, color, maxWidth: budget * 7 }));
}

/**
 * 按显示宽度单位折行（CJK=2、ASCII=1；口径同 `model.ts::textWidthUnits`）。
 *
 * 先按空白（含全角空格）切"词"再往行里塞：`step=5/11`、`ρ=0.9991` 这类字段不会被从中间劈开
 * （劈开在光栅器里读起来像两个数）。单个词比预算还长时才硬切。
 */
function wrapUnits(body: string, budget: number): string[] {
  const out: string[] = [];
  let line = "";
  let used = 0;
  const push = (): void => {
    if (line !== "") out.push(line);
    line = "";
    used = 0;
  };
  for (const word of body.split(/(?<=[\s　])/)) {
    // 用共享实现的同一把尺子（`model.ts::textWidthUnits`：CJK=2、ASCII=1），
    // 免得"模型以为放得下、渲染层却要折行"。
    const w = textWidthUnits(word);
    if (used > 0 && used + w > budget) push();
    if (w <= budget) {
      line += word;
      used += w;
      continue;
    }
    // 超长词：硬切成若干段
    let piece = "";
    let pieceUsed = 0;
    for (const ch of word) {
      const cw = textWidthUnits(ch);
      if (pieceUsed + cw > budget && piece !== "") {
        out.push(piece);
        piece = "";
        pieceUsed = 0;
      }
      piece += ch;
      pieceUsed += cw;
    }
    line = piece;
    used = pieceUsed;
  }
  push();
  return out;
}
