/**
 * `spy_ordering`（1.2 章）的 model：稀疏矩阵的填充长出来 + 三种排序的后向残差。
 *
 * 面板与图层（顺序即绘制顺序，测试按此断言）
 * ------------------------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | scatter | `main` | `pending_*`：后续还会产生的填充（最淡、最小） |
 * | 2 | scatter | `main` | `orig_*`：原矩阵非零（灰） |
 * | 3 | scatter | `main` | `fill_*`：已经产生的填充（红） |
 * | 4 | scatter | `main` | 当前消元步的主元格 `(k,k)`（一个点，标"走到哪了"） |
 * | 5 | bars | `aux` | `error`（三种排序的 LU 后向残差，log 纵轴） |
 * | 6 | scatter | `aux` | `error_mark`：当前排序的残差（高亮） |
 *
 * 角标槽位（固定，见 `SLOT`）：0 来源 / 1 主轴固定 / 2 副轴自适应 / 3 状态（排序名 + 帧 note）/
 * 4 前端预览（仅 `backend+preview`）/ 5 溢出 / 7 降级说明。
 *
 * 几条刻意的决定
 * --------------
 * 1. **行号翻转，而不是反向 `ylim`**：矩阵的习惯是行 0 在上，而 `validateScene` 要求
 *    `ylim[0] < ylim[1]`（反向轴会被判为"顺序颠倒"）。两条路都合法，这里选**翻转行号**
 *    （`y = n−1−row`），因为反向轴还会把热力图/散点的行序约定一起颠倒，日后加一层
 *    热力图就会踩坑；翻转只发生在这一处，其余代码一律用"显示坐标"。
 * 2. **`bars` 在 log 轴上必须给正 baseline**：两个渲染器都先算
 *    `dataToPixel(axis, x, baseline)`，`baseline = 0` 在对数轴上得到 NaN，整根柱子会被
 *    `continue` 跳过——图上什么都没有。所以 `baseline = auxYlim[0]`（轴底），柱子从轴底起画。
 * 3. **三种排序的名字只在本地表里**：帧只给"当前排序"的 `strings.ordering_name`，
 *    三个柱子的名字是 UI 文案（与后端 `_ordering_name` 同口径），顺序取自 `strategies_x`；
 *    帧 `strings.note` 原样显示在状态角标里，任何漂移都能一眼看出来。
 * 4. **后向残差与排序无关**：三根柱子应当都在 u 量级；这不是"图画得不好"，
 *    而是与 manifest pitfalls 一致的结论（排序只改运算量与访存，不改后向稳定性）。
 */

import { formatNumber, niceLimits } from "../../shared/src/axes.js";
import { countNonFinite, dataToPixel } from "../../shared/src/geometry.js";
import {
  applyOverflow,
  canvasSize,
  clamp,
  clipBadgeText,
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
  stackedPanels,
  stringOf,
  textWidthUnits,
  type Rect,
  type SceneModel,
} from "../../shared/src/model.js";
import {
  axisModeBadge,
  badge,
  barsLayer,
  makeAxis,
  previewBadge,
  scatterLayer,
  sourceBadge,
  text,
  type TextSpec,
} from "../../shared/src/scene.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "1.2";
const VIZ = "spy_ordering";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "稀疏 spy plot：填充是怎么长出来的";

/**
 * 固定角标槽位（见 `model.ts::BADGE_LAYOUT`）。
 * 长文本角标（溢出、降级）各占一行的左列，右侧留空，避免与邻居叠字。
 */
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
 * 三种消元顺序的显示名（UI 文案，与后端 `server/chapters/1.2.py::_ordering_name` 同口径）。
 * 顺序不是这里定的：柱子按帧里的 `strategies_x` 排，本地表只负责"第 i 根叫什么"。
 */
const ORDERING_KEYS: readonly string[] = ["natural", "rcm", "amd"];
const ORDERING_LABELS: Record<string, string> = {
  natural: "自然序",
  rcm: "RCM",
  amd: "最小度",
};

export const spyOrderingModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需字段：缺一个就抛 ModelInputError（宁可当场炸，也不画半张图）——
  const origRows = requireArray(frame, "orig_rows");
  const origCols = requireArray(frame, "orig_cols");
  const fillRows = requireArray(frame, "fill_rows");
  const fillCols = requireArray(frame, "fill_cols");
  const pendingRows = requireArray(frame, "pending_rows");
  const pendingCols = requireArray(frame, "pending_cols");
  const strategiesX = requireArray(frame, "strategies_x");
  const error = requireArray(frame, "error");
  const errorMark = requireArray(frame, "error_mark");
  const errorMarkX = requireArray(frame, "error_mark_x");
  requireSameLength(origRows, origCols, "帧数组 orig_rows 与 orig_cols");
  requireSameLength(fillRows, fillCols, "帧数组 fill_rows 与 fill_cols");
  requireSameLength(pendingRows, pendingCols, "帧数组 pending_rows 与 pending_cols");
  requireSameLength(strategiesX, error, "帧数组 strategies_x 与 error");
  const pitfalls = requirePitfalls(frame);

  // —— 可选字段：缺了就降级（记进 degraded，最终挂成一个 status 角标）——
  const degraded: string[] = [];
  const tMeta = metaNumber(frame, "t");
  if (tMeta === undefined) degraded.push("帧 meta 缺 t：scene.t 改用请求 t（两者可能不同）");
  const tNow = tMeta ?? opts.t;
  const nMeta = metaNumber(frame, "n");
  if (nMeta === undefined) degraded.push("帧 meta 缺 n：矩阵规模按非零坐标的最大值推断（坐标本身是数据，不是猜的）");
  const n = nMeta ?? Math.max(rowMax(origRows), rowMax(fillRows), rowMax(pendingRows)) + 1;
  const grid = metaNumber(frame, "grid");
  if (grid === undefined) degraded.push("帧 meta 缺 grid：参数摘要用请求值并标注");
  const k = metaNumber(frame, "step");
  if (k === undefined) degraded.push("帧 meta 缺 step：不标当前消元步（不猜）");
  const nSteps = metaNumber(frame, "n_steps");
  if (nSteps === undefined) degraded.push("帧 meta 缺 n_steps：不显示总步数");
  const nnzOriginal = metaNumber(frame, "nnz_original");
  const nnzTotal = metaNumber(frame, "nnz_total");
  const fillEdgesTotal = metaNumber(frame, "fill_edges_total");
  const fillEdgesSoFar = metaNumber(frame, "fill_edges_so_far");
  const fillRatio = metaNumber(frame, "fill_ratio");
  const bandwidth = metaNumber(frame, "bandwidth");
  const profile = metaNumber(frame, "profile");
  const maxClique = metaNumber(frame, "max_clique");
  const treewidth = metaNumber(frame, "treewidth_bound");
  const residualCurrent = metaNumber(frame, "residual_current");
  const orderingKey = stringOf(frame, "ordering");
  const orderingName = stringOf(frame, "ordering_name");
  if (orderingName === undefined) degraded.push("帧 strings 缺 ordering_name：状态角标只显示排序键名");
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

  // —— 排版：左侧方形 spy plot，右列 = 残差柱状 + 说明文字带 ——
  // 为什么不用 `columnsLayout`：它的左列高度是 `stackedPanels` 的主面板高度
  // （`mainShare` 缺省 0.56 → 只有 211px），spy plot 的格子会被压扁。
  // 这里直接拿内容区的 top/bottom 自己算方形（格子必须是方的，行列才对得上）。
  //
  // 为什么右列要分成"柱状面板 + 文字带"：spy plot 密到不能用文字压在上面
  // （红点是填充、灰点是原非零，任何文字都会糊在点阵里），所以图例与结论一律挪到
  // 右列下方那条带里；柱状面板因此只占右列上部 55%。
  const base = stackedPanels(width, height);
  // 方形边长扣掉 46px：轴下要放 x 刻度标签（约 18px）与一行指标（矩阵结构图的行列下标
  // 必须能读出来，所以这一行不能省）；主轴的 xlabel 因此置空，列的含义写进 head 行。
  const side = Math.max(110, Math.floor(Math.min(base.bottom - base.top - 46, base.contentWidth * 0.5)));
  const mainRect: Rect = [base.left, base.top, side, side];
  // 右列留出它自己的 y 轴刻度与 ylabel（render2d 把 ylabel 画在轴左侧 42px 处）。
  const auxX = base.left + side + 74;
  const auxW = Math.max(160, base.left + base.contentWidth - auxX);
  const auxH = Math.round((base.bottom - base.top) * 0.55);
  const auxRect: Rect = [auxX, base.top, auxW, auxH];
  // 文字带：从柱状面板的 x 类别标签下面开始（标签画在轴下 4px，约 14px 高）。
  // 底端只到 `stripTop + 104`（7 行）：再往下的 474~490 那条横带留给 spy 面板正下方的
  // 指标行——两条文字带都写满时会在那里撞车（真实踩过的坑）。
  const stripTop = base.top + auxH + 26;
  const stripBottom = stripTop + 104;

  const lim: [number, number] = [-0.5, n - 0.5];
  const mainAxis = makeAxis("main", mainRect, {
    xlim: lim,
    ylim: lim,
    xlabel: "",
    ylabel: "行 $i$（0 在上）",
    title: axisTitle("main", "矩阵结构（行 0 在上）"),
    autoscaled: false,
    xticks: indexTicks(n),
    yticks: indexTicks(n),
  });

  // 副轴（manifest 声明 autoscaled: true、scale: log）：三种排序的后向残差。
  const errPos = finitePositiveMinMax(error);
  const markPos = finitePositiveMinMax(errorMark);
  const auxLo = Math.min(errPos?.[0] ?? Number.POSITIVE_INFINITY, markPos?.[0] ?? Number.POSITIVE_INFINITY);
  const auxHi = Math.max(errPos?.[1] ?? Number.NEGATIVE_INFINITY, markPos?.[1] ?? Number.NEGATIVE_INFINITY);
  const auxYlim: [number, number] =
    Number.isFinite(auxLo) && Number.isFinite(auxHi) ? niceLimits(auxLo, auxHi, 7, { log: true }) : [1e-17, 1e-14];
  if (!(Number.isFinite(auxLo) && Number.isFinite(auxHi))) {
    degraded.push("error/error_mark 里没有 >0 的有限值：副轴用占位范围 [1e-17, 1e-14]，柱子画不出来");
  }
  const strategyRange = finiteRangeOr(strategiesX, [0, Math.max(0, error.length - 1)]);
  const auxAxis = makeAxis("aux", auxRect, {
    xlim: [(strategyRange[0] as number) - 0.6, (strategyRange[1] as number) + 0.6],
    ylim: auxYlim,
    xscale: "linear",
    yscale: "log",
    // xlabel 置空：类别名由模型手写在刻度行（`AxisOptions` 没有 xtickLabels），
    // 再让渲染层画一条 xlabel 就会与手写标签挤在同一带里（规则 8 会报）。
    xlabel: "",
    ylabel: axisYlabel("aux", "‖PA−LU‖_∞ / ‖A‖_∞"),
    title: axisTitle("aux", "各排序的 LU 后向残差"),
    autoscaled: true,
    // 刻度标签用 text 层手写（AxisOptions 没有 xtickLabels）：先清空数字刻度，免得两行叠字。
    xticks: [],
  });
  scene.axes.push(mainAxis, auxAxis);

  // —— 图层 1~4：spy plot（行号翻转：见文件头第 1 条）——
  const cellPx = Math.min(mainRect[2], mainRect[3]) / Math.max(1, n);
  const rOriginal = clamp(cellPx * 0.34, 1.1, 3.4);
  const show = (rows: ArrayLike<number>, cols: ArrayLike<number>): { x: number[]; y: number[] } => {
    const x: number[] = [];
    const y: number[] = [];
    const m = Math.min(rows.length, cols.length);
    for (let i = 0; i < m; i += 1) {
      const r = rows[i];
      const c = cols[i];
      if (!Number.isFinite(r) || !Number.isFinite(c)) continue;
      x.push(c);
      y.push(n - 1 - r);
    }
    return { x, y };
  };
  const pending = show(pendingRows, pendingCols);
  if (pending.x.length > 0) {
    scene.layers.push(
      scatterLayer("main", pending.x, pending.y, {
        r: Math.max(0.9, rOriginal * 0.62),
        color: pal.reference,
        alpha: 0.42,
        label: "后续还会产生的填充（帧 $pending_*$）",
      }),
    );
  }
  const orig = show(origRows, origCols);
  if (orig.x.length > 0) {
    scene.layers.push(
      scatterLayer("main", orig.x, orig.y, {
        r: rOriginal,
        color: pal.muted,
        alpha: 0.95,
        label: "原矩阵非零（帧 $orig_*$）",
      }),
    );
  }
  const fill = show(fillRows, fillCols);
  if (fill.x.length > 0) {
    scene.layers.push(
      scatterLayer("main", fill.x, fill.y, {
        r: Math.max(1.2, rOriginal * 0.85),
        color: pal.marker,
        alpha: 0.8,
        label: "已经产生的填充（帧 $fill_*$）",
      }),
    );
  }
  if (k !== undefined && k < n) {
    scene.layers.push(
      scatterLayer("main", [k], [n - 1 - k], {
        r: Math.max(2.4, rOriginal * 1.5),
        color: pal.note,
        alpha: 0.9,
        label: `当前消元步 k=${k} 的主元格`,
      }),
    );
  }

  // —— 图层 5~6：副轴（log 柱状 + 当前排序高亮）——
  // `baseline` 取轴底（>0）：log 轴上 baseline=0 会让整根柱子被渲染层跳过（见文件头第 2 条）。
  scene.layers.push(
    barsLayer("aux", strategiesX, error, {
      width: 0.62,
      color: pal.curves[0],
      baseline: auxAxis.ylim[0],
      label: "各排序的 LU 后向残差",
    }),
  );
  let errNaN = 0;
  for (let i = 0; i < error.length; i += 1) if (!Number.isFinite(error[i])) errNaN += 1;
  const mx = errorMarkX.length > 0 ? errorMarkX[0] : undefined;
  const my = errorMark.length > 0 ? errorMark[0] : undefined;
  if (mx !== undefined && my !== undefined && Number.isFinite(mx) && Number.isFinite(my) && my > 0) {
    scene.layers.push(scatterLayer("aux", [mx], [my], { r: 5, color: pal.marker, label: "当前排序的残差（$error_mark$）" }));
  } else {
    degraded.push(`帧 error_mark/error_mark_x 不可画（不可画 = 非有限或 ≤0）：当前排序没有高亮点`);
  }

  // —— 文本（一行一个 text 层：render2d 不拆 \n）——
  const summary = paramSummary(frame, opts, [{ name: "grid", label: "网格 $k$" }]);
  degraded.push(...summary.degraded);
  const head: string[] = [summary.text, `n=${n}`, "列 j 已按当前顺序重排"];
  if (k !== undefined) {
    head.push(nSteps === undefined ? `step=${formatNumber(k, 4)}` : `step=${formatNumber(k, 4)}/${formatNumber(nSteps, 4)}`);
  }
  if (orderingName !== undefined) head.push(`顺序=${orderingName}`);
  // 说明文字全在右列的文字带里（宽度 = 右列宽），一行约 52 个显示单位。
  const textBudget = unitBudget(auxW);
  let textTop = stripTop;
  const headSpecs = textBlock(auxX, textTop, stripBottom, [head.join("　")], pal.muted, textBudget, "top");
  scene.texts.push(...headSpecs);
  textTop += headSpecs.length * 15 + 6;

  const metrics: string[] = [];
  if (nnzTotal !== undefined) metrics.push(`nnz(L+U)=${formatNumber(nnzTotal, 6)}`);
  if (nnzOriginal !== undefined) metrics.push(`nnz(A)=${formatNumber(nnzOriginal, 6)}`);
  if (fillRatio !== undefined) metrics.push(`填充倍数=${formatNumber(fillRatio, 4)}`);
  if (fillEdgesSoFar !== undefined && fillEdgesTotal !== undefined) {
    metrics.push(`填充边 ${formatNumber(fillEdgesSoFar, 6)}/${formatNumber(fillEdgesTotal, 6)}`);
  } else if (fillEdgesTotal !== undefined) {
    metrics.push(`填充边=${formatNumber(fillEdgesTotal, 6)}`);
  }
  if (bandwidth !== undefined) metrics.push(`带宽=${formatNumber(bandwidth, 6)}`);
  if (profile !== undefined) metrics.push(`剖面=${formatNumber(profile, 8)}`);
  if (maxClique !== undefined) metrics.push(`最大团=${formatNumber(maxClique, 6)}`);
  if (treewidth !== undefined) metrics.push(`树宽上界=${formatNumber(treewidth, 6)}`);
  const mainLines = metrics.length > 0 ? [metrics.join("　")] : [];
  mainLines.push("填充只由结构与顺序决定，与数值无关：换顺序会改非零计数与访存，不会改后向稳定性。");
  // 指标行放在 spy 面板轴下（刻度标签下面），不占右列的说明带。
  // 这条带横跨整个内容区（spy 面板下面 46px），所以每行能放 ~120 个显示单位：
  // 指标行 + 一行结论，正好两行。
  scene.texts.push(
    ...textBlock(mainRect[0] + 6, mainRect[1] + mainRect[3] + 22, base.bottom + 4, mainLines, pal.muted, unitBudget(base.contentWidth), "top"),
  );

  const legend: string[] = [
    `灰=${orig.x.length} 个原非零　红=${fill.x.length} 条填充　淡=${pending.x.length} 条后续填充`,
    "行 0 在上：行号翻转成 y=n−1−row（不许反向 ylim，见文件头第 1 条）",
  ];
  if (pending.x.length === 0) legend.push("后续填充为空：本次排序的全部填充都已经长出来了（或没有填充）");
  const legendSpecs = textBlock(auxX, textTop, stripBottom, legend, pal.muted, textBudget, "top");
  scene.texts.push(...legendSpecs);
  textTop += legendSpecs.length * 15 + 6;

  const auxLines: string[] = [];
  if (note !== undefined) auxLines.push(`帧 note：${note}`);
  const names = orderingLabels(strategiesX, orderingKey);
  // 三根柱子的名字已经写在柱子下面（手写刻度标签），这里不再重复；
  // "都在 u 量级"的结论在 spy 面板下方那行（`mainLines`），一处说一件事。
  if (residualCurrent !== undefined) {
    auxLines.push(`当前排序的后向残差（meta.residual_current）= ${formatNumber(residualCurrent, 4)}`);
  }
  if (errNaN > 0) auxLines.push(`${errNaN} 个残差非有限：柱子被渲染层跳过（不是 0）。`);
  const auxSpecs = textBlock(auxX, textTop, stripBottom, auxLines, errNaN > 0 ? pal.warn : pal.muted, textBudget, "top");
  scene.texts.push(...auxSpecs);

  // x 刻度下的排序名：`AxisOptions` 只有 `xticks: number[]`、**没有** xtickLabels 字段
  // （见 scene.ts::AxisOptions），所以这里按"数据 → 像素"自己放 text 层；
  // 两个渲染器都画 text，浏览器与视觉回归因此一致。当前排序用强调色标出。
  const labelY = auxRect[1] + auxRect[3] + 4;
  numericTicks(strategiesX).forEach((s, i) => {
    const [px] = dataToPixel(auxAxis, s, auxAxis.ylim[0]);
    if (!Number.isFinite(px)) return;
    const isCurrent = orderingKey !== undefined && ORDERING_KEYS[i] === orderingKey;
    scene.texts.push(
      text(px, labelY, names[i] ?? `#${i}`, {
        role: "axis",
        size: 11,
        color: isCurrent ? pal.marker : pal.muted,
        align: "center",
      }),
    );
  });

  // —— 角标（固定槽位）——
  // 规则 5：恒挂 source；预览型数据源另挂 preview（kind 各表一意，见 scene.ts）。
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  if (scene.dataSource.startsWith("backend+preview")) {
    scene.badges.push(placeBadge(previewBadge("拖动时按显示空间插值"), SLOT.preview, width));
  }
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(
      badge("autoscale", "残差轴：自适应（log 纵轴按本帧 error 取整；自适应会掩盖量级变化）", 8, 8),
      SLOT.auxAxis,
      width,
    ),
  );
  const status: string[] = [];
  if (orderingName !== undefined) status.push(`顺序：${orderingName}`);
  if (fillRatio !== undefined) status.push(`填充倍数 ${formatNumber(fillRatio, 4)}`);
  if (note !== undefined) status.push(note);
  if (status.length === 0) status.push("帧未给出可判读的状态（预期有 strings.ordering_name 与 strings.note）");
  scene.badges.push(placeBadge(badge("status", clipBadgeText(status.join("；")), 8, 8), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  // 规则 1：图层越界（柱子的 baseline 取了轴底，不会越界）+ 帧内非有限值。
  const nBad =
    countNonFinite(origRows) +
    countNonFinite(origCols) +
    countNonFinite(fillRows) +
    countNonFinite(fillCols) +
    countNonFinite(pendingRows) +
    countNonFinite(pendingCols) +
    countNonFinite(strategiesX) +
    countNonFinite(error) +
    countNonFinite(errorMark) +
    countNonFinite(errorMarkX);
  applyOverflow(scene, nBad, { slot: SLOT.overflow, width });
  return scene;
};

// --------------------------------------------------------------------------- //
// 局部工具
// --------------------------------------------------------------------------- //

/**
 * 一段说明文字 → 若干 `text` 层，**一行一个**（`render2d.ts::drawText` 不拆 `\n`）。
 *
 * 为什么要自己折行：两个渲染器都支持 `maxWidth` 自动折行，但折出来的续行**不占**模型这一侧的
 * 行距预算（光栅器的续行间距是 `cell_h + 4`，比这里排的 15px 大），续行会压到下一行文字上。
 * 所以模型先按"显示宽度单位"折到 `budget` 以内（`model.ts::textWidthUnits`：CJK=2、ASCII=1），
 * 再逐行排开：一行就是一层，两个渲染器都不会再折。
 *
 * `anchor`：`"top"` 从 `top` 往下排；`"bottom"` 让整块贴着 `bottom`。返回的数组长度 = 实际行数。
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
 * 按显示宽度单位折行（同一把尺子：`model.ts::textWidthUnits`，CJK=2、ASCII=1）。
 *
 * 先按空白（含全角空格）切"词"再往行里塞：`ρ=0.9991`、`k=3/63` 这类字段不会被从中间劈开
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
    const w = textWidthUnits(word);
    if (used > 0 && used + w > budget) push();
    if (w <= budget) {
      line += word;
      used += w;
      continue;
    }
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

/**
 * 面板宽度 → 显示宽度单位预算（size 11 时 1 单位 ≈ 7px 光栅单元）。
 * 留 12px 边距再折算，保证两个渲染器都不会自动折行（见 `textBlock` 的注释）。
 */
function unitBudget(panelWidthPx: number): number {
  return Math.max(24, Math.floor((panelWidthPx - 12) / 7));
}

/** 有限值的最大行号（缺 meta.n 时用来定矩阵规模）；没有有限值时返回 −1。 */
function rowMax(rows: ArrayLike<number>): number {
  let hi = -1;
  for (let i = 0; i < rows.length; i += 1) {
    const v = rows[i];
    if (Number.isFinite(v) && v > hi) hi = v;
  }
  return hi;
}

/** 矩阵下标刻度（0..n−1，n 大时按步进取样）。 */
function indexTicks(n: number): number[] {
  if (n <= 0) return [];
  const step = Math.max(1, Math.ceil(n / 8));
  const out: number[] = [];
  for (let i = 0; i < n; i += step) out.push(i);
  return out;
}

/** 帧里给的策略横坐标（去重、升序、只留有限值）。 */
function numericTicks(values: ArrayLike<number>): number[] {
  const out: number[] = [];
  for (let i = 0; i < values.length; i += 1) {
    const v = values[i];
    if (!Number.isFinite(v)) continue;
    if (!out.includes(v)) out.push(v);
  }
  return out.sort((a, b) => a - b);
}

/** 三根柱子的显示名；当前排序用帧给的 `ordering_name` 覆盖本地文案（帧优先）。 */
function orderingLabels(values: ArrayLike<number>, currentKey: string | undefined): string[] {
  const out: string[] = [];
  for (let i = 0; i < values.length; i += 1) {
    const key = ORDERING_KEYS[i];
    if (key !== undefined && key === currentKey) out.push(`${ORDERING_LABELS[key] ?? key}（当前）`);
    else out.push(key === undefined ? `#${i}` : (ORDERING_LABELS[key] ?? key));
  }
  return out;
}

/** 有限值的 `[min, max]`；没有有限值时回落。 */
function finiteRangeOr(a: ArrayLike<number>, fallback: [number, number]): [number, number] {
  let lo = Number.POSITIVE_INFINITY;
  let hi = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i];
    if (!Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return Number.isFinite(lo) && Number.isFinite(hi) ? [lo, hi] : fallback;
}

/** manifest 里该轴的标题（文案直接取 manifest，缺省才用回落值）。 */
function axisTitle(id: string, fallback: string): string {
  return SPEC?.axes.find((a) => a.id === id)?.title ?? fallback;
}

/** manifest 里该轴的 ylabel。 */
function axisYlabel(id: string, fallback: string): string {
  return SPEC?.axes.find((a) => a.id === id)?.ylabel ?? fallback;
}
