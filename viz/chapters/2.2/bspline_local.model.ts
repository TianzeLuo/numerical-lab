/**
 * `bspline_local`（2.2 章）的 model：B 样条的局部支撑、节点插入与"哪些东西没变"。
 *
 * 面板与图层（顺序即绘制顺序，测试按此断言）
 * ------------------------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | line | `main` | 基函数（帧 `basis`，每行一条；只有支撑区间与受影响区间相交的那些） |
 * | 2 | line | `main` | 当前曲线 `curve_orig`（未移动控制点） |
 * | 3 | line | `main` | 移动控制点后的曲线 `curve_moved`（虚线） |
 * | 4 | line | `main` | 控制多边形（原）+（移动后，虚线），端点 = 帧的 `ctrl_x`/`ctrl_y(_moved)` |
 * | 5 | scatter | `main` | 控制点（含被移动的那一个，用强调色） |
 * | 6 | line | `main` | 受影响参数区间 `[span_left, span_right]` 的两条竖边界 |
 * | 7 | scatter | `main` | 插入点 `insert_marker` |
 * | 8 | line | `invariance` | `error`（插入前后的逐点差，log） |
 * | 9 | line | `support` | `residual`（支撑外的 max|B|，log） |
 * | 10 | line | `changed` | `changed`（改控制点前后的 |ΔC|，线性；同时看局部性与"到底改了多少"） |
 *
 * 角标槽位：0 来源 / 1 主轴固定 / 2 不变性轴自适应 log / 3 支撑轴自适应 log /
 * 4 状态（插入次数 + 受影响区间）/ 5 前端预览（仅 backend+preview）/ 7 状态与降级。
 *
 * 三条坑，以及各自在图上的落点
 * --------------------------
 * 1. **控制点不是曲线上的点**：控制多边形用虚线与散点单独画，曲线是另一条线；
 *    两者在图上明显不重合（凸包性质保证曲线在多边形**内**，`hull_gap` 就是它的量化）。
 * 2. **局部性的边界是 k+1 段**：主轴上的两条竖线就是帧给的 `[span_left, span_right]`，
 *    说明带里同时写出"几个非退化区间 / 上界 k+1 / 占参数域百分之几"——k=3、8 个控制点时
 *    它已经占了 59.6%，"局部"是相对 n 说的。
 * 3. **插节点不改曲线**：`invariance` 面板画的是 `error`（插入前后逐点差），
 *    它贴在 u/2 上；改的是**表示**（控制点多出来、重数涨），不是几何。
 */

import { niceTicks } from "../../shared/src/axes.js";
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
  makeAxis,
  previewBadge,
  scatterLayer,
  sourceBadge,
} from "../../shared/src/scene.js";
import { fmt, FLOOR, linRange, logRange, totalNonFinite } from "./derive.js";
import {
  contentBottom,
  finishTexts,
  hLine,
  matrixRow,
  panelRows,
  panelTop,
  rightColumnFits,
  bandText,
  packColumn,
  splitRect,
  twoColumnGap,
  vLine,
} from "./layout.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "2.2";
const VIZ = "bspline_local";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "B 样条：局部支撑与节点插入";

const SLOT = {
  source: 0,
  axisMode: 1,
  invarianceAxis: 2,
  supportAxis: 3,
  state: 4,
  preview: 5,
  /** 降级说明（规则 6：缺什么就说什么）。 */
  degraded: 6,
  /** 溢出徽标（`applyOverflow`）：与降级分开占槽，否则两者会叠在同一个盒子里。 */
  overflow: 7,
} as const;

/** 时间轴语义：`t` = 节点插入次数（整数，0..4），越界就夹住并如实写出来。 */
const T_MAX = 4;

export const bsplineLocalModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width } = canvasSize(opts);

  // —— 必需数组 ——
  const ctrlX = requireArray(frame, "ctrl_x");
  const ctrlY = requireArray(frame, "ctrl_y");
  const ctrlYMoved = requireArray(frame, "ctrl_y_moved");
  const knots = requireArray(frame, "knots");
  const knotsBase = requireArray(frame, "knots_base");
  const xGrid = requireArray(frame, "x_grid");
  const curveOrig = requireArray(frame, "curve_orig");
  const curveMoved = requireArray(frame, "curve_moved");
  const curveBase = requireArray(frame, "curve_base");
  const basis = requireArray(frame, "basis");
  const basisIndex = requireArray(frame, "basis_index");
  const supportLeft = requireArray(frame, "support_left");
  const supportRight = requireArray(frame, "support_right");
  const changed = requireArray(frame, "changed");
  const changedMask = requireArray(frame, "changed_mask");
  const invariance = requireArray(frame, "error");
  const supportRes = requireArray(frame, "residual");
  const affectedSpan = requireArray(frame, "affected_span");
  const insertMarker = requireArray(frame, "insert_marker");
  requireSameLength(ctrlX, ctrlY, "帧数组 ctrl_x 与 ctrl_y");
  requireSameLength(ctrlX, ctrlYMoved, "帧数组 ctrl_x 与 ctrl_y_moved");
  requireSameLength(xGrid, curveOrig, "帧数组 x_grid 与 curve_orig");
  requireSameLength(xGrid, curveMoved, "帧数组 x_grid 与 curve_moved");
  requireSameLength(xGrid, curveBase, "帧数组 x_grid 与 curve_base");
  requireSameLength(xGrid, changed, "帧数组 x_grid 与 changed");
  requireSameLength(xGrid, changedMask, "帧数组 x_grid 与 changed_mask");
  requireSameLength(xGrid, invariance, "帧数组 x_grid 与 error");
  requireSameLength(basisIndex, supportLeft, "帧数组 basis_index 与 support_left");
  requireSameLength(basisIndex, supportRight, "帧数组 basis_index 与 support_right");
  requireSameLength(basisIndex, supportRes, "帧数组 basis_index 与 residual");
  const pitfalls = requirePitfalls(frame);

  // —— 可选标注：缺了就降级 ——
  const degraded: string[] = [];
  const tMeta = metaNumber(frame, "t");
  if (tMeta === undefined) degraded.push("帧 meta 缺 t：插入次数改用请求 t（两者可能不同）");
  // **请求值优先**：`t` 是 scrub 的输入（用户拖到哪儿），帧里的 `t` 是"这一帧实际取到的次数"。
  // 反过来取（帧优先）会让"越界"这件事永远不可见——拖动 7 时帧仍是 t=4，图上什么都不说。
  const tRaw = opts.t ?? tMeta;
  // `scrub`：帧只覆盖 t∈[0,4] 的整数帧，越界就把真实取到的插入次数写进文本，不假装 t 就是次数。
  const clamped = tRaw < 0 || tRaw > T_MAX;
  const insertCount = metaNumber(frame, "insert_count");
  const nCtrl = metaNumber(frame, "n_ctrl");
  const degree = metaNumber(frame, "degree");
  const nBasis = metaNumber(frame, "n_basis");
  const nKnots = metaNumber(frame, "n_knots");
  const nKnotsBase = metaNumber(frame, "n_knots_base");
  const moveIndex = metaNumber(frame, "move_index");
  const moveDelta = metaNumber(frame, "move_delta");
  const insertAt = metaNumber(frame, "insert_at");
  const spanLeft = metaNumber(frame, "span_left");
  const spanRight = metaNumber(frame, "span_right");
  const affectedSpans = metaNumber(frame, "affected_spans");
  const affectedSpansMax = metaNumber(frame, "affected_spans_max");
  const changedMeasure = metaNumber(frame, "changed_measure");
  const changedMax = metaNumber(frame, "changed_max");
  const invarMax = metaNumber(frame, "invariance_error_max") ?? metaNumber(frame, "error");
  const supportViolationMax = metaNumber(frame, "support_violation_max");
  const partitionUnity = metaNumber(frame, "partition_unity_error");
  const hullGap = metaNumber(frame, "hull_gap");
  const note = stringOf(frame, "note");
  if (note === undefined) degraded.push("帧 strings 缺 note：结论不自行编造");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    // 时间轴的语义是"插入次数"：帧里的 `t` 是请求值、`insert_count` 才是这一帧真的插了几次。
    t: insertCount ?? tRaw,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  // —— 版式：**两列两行**（主图 | 不变性 / 支撑 | 改动量）——
  //
  // 为什么不是一列四行：一列的每行之间要塞下"上一格的说明文字 + 下一格的轴标题带"，
  // 净空至少 2×18px；四行就要多花 ~150px 行距，主轴只剩 60px 高（读不出曲线）。
  // 两列时横向空间本来就是多余的（轴只占 `AXIS_W`），用列换行更划算。
  const gapX = twoColumnGap("max|B| 支撑外", "|dC|");
  // 行高按"面板下方要写几行字"倒推：行距 = 上一格的 x 刻度带(15) + 下一格的轴标题带(16) + 4px 余量
  // = 35px，所以每行留 35px 就能在最下面挤出一行说明（2 行就要再减面板高度）。
  // 版式：三行主面板（主图 / 不变性 / 支撑+改动两列）。
  //
  // 竖直方向的可用区间**只有行距**：面板自己的 x 刻度带占 `[底+3, 底+18]`，
  // 下一格的轴标题带占 `[下一格顶−18, 下一格顶−2]`。于是说明文字只能落在
  // `[底+20, 底+gap−20]` 这条缝里 —— 要写两行（30px）就得 `gap ≥ 70`。
  // 三行版式：把「支撑」与「改动」并排在最下面一行（四行在 600px 高的画布上排不下：
  // 4×56 + 3×70 > 可用高度）。文字不放面板正下方，全部走 `packBands` 进右侧栏。
  const ROW_GAP_B = 40;
  const rowsLay = panelRows(width, [112, 112, 112], panelTop(width), contentBottom(scene), ROW_GAP_B, 24);
  const mainRect = rowsLay[0]!.rect;
  const invarRect = rowsLay[1]!.rect;
  const botRow = rowsLay[2]!;
  const supportRect = splitRect(botRow.rect, 0, 2, gapX);
  const changedRect = splitRect(botRow.rect, 1, 2, gapX);

  // —— 主轴：**固定参数域 [0,1]²**（manifest: autoscaled=false）——
  const yValues: number[] = [];
  for (let i = 0; i < ctrlY.length; i += 1) yValues.push(ctrlY[i] as number, ctrlYMoved[i] as number);
  for (let i = 0; i < curveOrig.length; i += 1) yValues.push(curveOrig[i] as number, curveMoved[i] as number);
  const yLim = linRange(yValues, [-1.5, 1.5], 5);
  const mainAxis = makeAxis("main", mainRect, {
    xlim: [-0.02, 1.02],
    ylim: yLim,
    xlabel: "",
    ylabel: axisYlabel("main", "y"),
    title:
      hullGap === undefined
        ? axisTitle("main", "曲线、控制多边形与基函数（固定参数域）")
        : `曲线、控制多边形与基函数（固定参数域；hull_gap=${fmt(hullGap, 3)}，曲线在控制多边形内）`,
    autoscaled: false,
    xticks: niceTicks(0, 1, Math.max(3, Math.round(mainRect[2] / 110))),
    yticks: niceTicks(yLim[0], yLim[1], 5),
  });

  // —— 不变性轴：log、自适应（插入节点前后逐点差，应当贴 u/2）——
  const invarLim = logRange([...toNumbers(invariance)], [FLOOR, 1e-12]);
  const invarAxis = makeAxis("invariance", invarRect, {
    xlim: [0, 1],
    ylim: invarLim,
    yscale: "log",
    xlabel: "",
    ylabel: axisYlabel("invariance", "|dC|"),
    title: axisTitle("invariance", "插入前后的逐点差（log）"),
    autoscaled: true,
    xticks: [],
    yticks: niceTicks(invarLim[0], invarLim[1], 4, { log: true }),
  });

  // —— 支撑轴：log、自适应；横轴是基函数下标（离散），所以清空 xticks 手写 ——
  const supportLim = logRange([...toNumbers(supportRes)], [FLOOR, 1e-12]);
  const supportAxis = makeAxis("support", supportRect, {
    xlim: [-0.6, Math.max(1, basisIndex.length - 1) + 0.6],
    ylim: supportLim,
    yscale: "log",
    xlabel: "",
    ylabel: "",
    title: axisTitle("support", "支撑外的基函数值（应恒为 0）"),
    autoscaled: true,
    xticks: [],
    // 中排两格的 y 刻度与轴名**故意留空**：它们的刻度带沿整条轴高延伸，会正好压住
    // 上排说明文字的位置（行距只有 40px）。数值改由说明栏给（见 `below(supportRect)`）。
    yticks: [],
  });

  // —— 改动量轴：线性、自适应（`changed` 有正有 0，log 画不出来）——
  const changedLim = linRange([...toNumbers(changed)], [0, 1], 5);
  const changedAxis = makeAxis("changed", changedRect, {
    xlim: [0, 1],
    ylim: changedLim,
    xlabel: "",
    ylabel: "",
    title: axisTitle("changed", "改一个控制点后的 |ΔC|（线性）"),
    autoscaled: true,
    xticks: [],
    yticks: [],
    // 同上：下排不挂 y 刻度，数值写在说明栏里。
  });

  scene.axes.push(mainAxis, invarAxis, supportAxis, changedAxis);

  // —— 图层的行/列约定：`basis` 是 (n_basis, n_grid) 的行主序矩阵 ——
  const nCols = xGrid.length;
  const nRows = nBasis !== undefined ? Math.min(Math.round(nBasis), Math.floor(basis.length / Math.max(1, nCols))) : Math.floor(basis.length / Math.max(1, nCols));

  // —— 图层 1：基函数。只画"支撑与受影响区间相交"的那些，避免十几条丘陵糊成一片；
  //     被移动的那个控制点对应的基函数用强调色（它才是 changed 的来源）。 ——
  const basisSeries: { x: number[]; y: number[]; color: string; width: number; label?: string }[] = [];
  let basisDrawn = 0;
  // meta 说的基函数个数与数组实际装得下的行数不一致 ⇒ 画出来的基函数会整体错位，
  // 宁可不画（下面 `basisOk` 会进降级角标与状态角标）。
  let basisOk = true;
  if (nBasis !== undefined && Math.round(nBasis) > nRows) {
    basisOk = false;
    degraded.push(`帧 basis 只有 ${fmt(nRows)} 行，meta.n_basis=${fmt(nBasis)}：行列不自洽，基函数层画不出来`);
  }
  for (let i = 0; i < nRows; i += 1) {
    const lo = supportLeft[i] as number | undefined;
    const hi = supportRight[i] as number | undefined;
    if (lo === undefined || hi === undefined || !Number.isFinite(lo) || !Number.isFinite(hi)) continue;
    if (!(hi > lo)) continue;
    const touches = spanLeft === undefined || spanRight === undefined ? true : hi >= spanLeft - 1e-12 && lo <= spanRight + 1e-12;
    const isMoved = moveIndex !== undefined && Math.abs(i - moveIndex) < 0.5;
    if (!touches && !isMoved) continue;
    const row = matrixRow(basis, i, nCols);
    if (!row.ok) {
      basisOk = false;
      degraded.push(`帧 basis 的行列与 meta.n_basis=${fmt(nBasis)}+x_grid 长度=${fmt(nCols)} 不自洽：基函数层画不出来`);
      break;
    }
    basisSeries.push({
      x: row.x.map((j) => xGrid[j] as number),
      y: row.y,
      color: isMoved ? pal.curves[0] : pal.reference,
      width: isMoved ? 1.8 : 1,
      label: isMoved ? `被移动控制点的基函数 B_${i}（支撑 [${fmt(lo, 3)}, ${fmt(hi, 3)}]）` : undefined,
    });
    basisDrawn += 1;
  }
  if (basisSeries.length > 0) {
    scene.layers.push({ kind: "line", axes: "main", series: basisSeries });
  } else {
    basisOk = false;
    degraded.push("帧 basis 里没有可画的基函数（行列不自洽或全长 0）：主轴只画曲线与控制多边形");
  }

  // —— 图层 2~3：曲线（未移动 / 移动后）——
  scene.layers.push({
    kind: "line",
    axes: "main",
    series: [
      { x: xGrid, y: curveOrig, color: pal.curves[1], width: 2, label: "曲线（$curve_orig$）" },
      { x: xGrid, y: curveMoved, color: pal.curves[0], width: 2, dash: [6, 4], label: "曲线（$curve_moved$）" },
    ],
  });

  // —— 图层 4~5：控制多边形 + 控制点 ——
  scene.layers.push({
    kind: "line",
    axes: "main",
    series: [
      { x: ctrlX, y: ctrlY, color: pal.muted, width: 1.2, dash: [2, 3], marker: "o", label: "控制多边形（$ctrl_y$）" },
      {
        x: ctrlX,
        y: ctrlYMoved,
        color: pal.curves[0],
        width: 1.2,
        dash: [2, 3],
        marker: "s",
        label: "多边形（$ctrl_y_moved$）",
      },
    ],
  });
  if (moveIndex !== undefined && moveIndex >= 0 && moveIndex < ctrlX.length) {
    const mx = ctrlX[moveIndex] as number;
    const my0 = ctrlY[moveIndex] as number;
    const my1 = ctrlYMoved[moveIndex] as number;
    scene.layers.push(
      scatterLayer("main", [mx, mx], [my0, my1], {
        r: 4.2,
        color: pal.marker,
        label: `被移动的第 ${fmt(moveIndex)} 个控制点（Δ=${fmt(moveDelta)}）`,
      }),
    );
  }

  // —— 图层 6：受影响参数区间（帧 affected_span 的两端）——
  if (affectedSpan.length >= 2) {
    const lo = affectedSpan[0] as number;
    const hi = affectedSpan[1] as number;
    if (Number.isFinite(lo) && Number.isFinite(hi) && hi > lo) {
      scene.layers.push(
        bandLayer("main", {
          x: [lo, hi],
          ylo: [yLim[0], yLim[0]],
          yhi: [yLim[1], yLim[1]],
          color: pal.curves[0],
          alpha: 0.08,
          label: "受影响区间",
        }),
      );
      vLine(scene, mainAxis, lo, pal.curves[0], { label: "`t_i`" });
      vLine(scene, mainAxis, hi, pal.curves[0], { label: "$t_{i+k+1}$" });
    }
  }

  // —— 图层 7：插入点（时间轴 = 插入次数；插在哪由 insert_at 给）——
  if (insertMarker.length >= 2) {
    const x = insertMarker[0] as number;
    const y = insertMarker[1] as number;
    if (Number.isFinite(x) && Number.isFinite(y)) {
      scene.layers.push(scatterLayer("main", [x], [y], { r: 4.5, color: pal.exact, label: "插入点 $(u,C(u))$" }));
      vLine(scene, mainAxis, x, pal.exact, { label: "插入参数 $u$" });
    }
  }

  // —— 图层 8~10：三格副轴 ——
  scene.layers.push({
    kind: "line",
    axes: "invariance",
    series: [{ x: xGrid, y: invariance, color: pal.residual, width: 1.6, label: "$|ΔC|$（error）" }],
  });
  hLine(scene, invarAxis, FLOOR, pal.note, { label: "$u/2$" });
  scene.layers.push({
    kind: "line",
    axes: "support",
    series: [{ x: basisIndex, y: supportRes, color: pal.curves[0], width: 1.4, marker: "o", label: "支撑外 $max|B_i|$（residual）" }],
  });
  hLine(scene, supportAxis, FLOOR, pal.note, { label: "$u/2$" });
  scene.layers.push({
    kind: "line",
    axes: "changed",
    series: [{ x: xGrid, y: changed, color: pal.curves[1], width: 1.6, label: "$|ΔC|$（changed）" }],
  });
  hLine(scene, changedAxis, 0, pal.reference, { label: "0" });
  // 受影响区间的两条边界在三格副轴上同样画出来：一眼看出"改动只发生在这个窗口里"。
  for (const axisId of ["invariance", "support", "changed"] as const) {
    const axis = scene.axes.find((a) => a.id === axisId);
    if (!axis || affectedSpan.length < 2) continue;
    const lo = affectedSpan[0] as number;
    const hi = affectedSpan[1] as number;
    if (axisId === "support") continue; // 支撑轴的横轴是基函数下标，与参数 u 不同轴
    if (Number.isFinite(lo)) vLine(scene, axis, lo, pal.curves[0], { label: undefined });
    if (Number.isFinite(hi)) vLine(scene, axis, hi, pal.curves[0], { label: undefined });
  }

  // —— 文本 ——
  // —— 说明文字：三块结论装进三行面板的右栏 ——
  // 文字**不放面板正下方**（那里只有行距 70px，要塞下自己的 x 刻度带 + 下一格的轴标题带，
  // 常常只剩 15px），而是放右侧那一栏：它在所有轴之外，整条高度都能用。
  const wide = rightColumnFits(width);
  const summary = paramSummary(frame, opts, [
    { name: "n_ctrl", label: "控制点" },
    { name: "degree", label: "$k$" },
    { name: "move_index", label: "移动 #" },
    { name: "move_delta", label: "$Δ$", digits: 3 },
    { name: "insert_at", label: "$u$", digits: 3 },
  ]);
  degraded.push(...summary.degraded);

  const mainLines = [
    `${summary.text}　控制点 ${fmt(nCtrl)} 个，k=${fmt(degree)}　u=${fmt(insertAt, 3)}`,
    `节点向量 ${fmt(nKnots)} 个（初始 ${fmt(nKnotsBase)}）——插节点只加自由度，不改曲线`,
  ];
  const invarLines = [`插入 ${fmt(insertCount)} 次后逐点差 max ${fmt(invarMax)}（曲线不变）`];
  const spanLines: string[] = [];
  if (changedMax !== undefined) {
    spanLines.push(
      `改 #${fmt(moveIndex)}：|ΔC|max ${fmt(changedMax)}，受影响 [${fmt(spanLeft, 3)}, ${fmt(spanRight, 3)}] ` +
        `（${fmt(affectedSpans)} 段，≤k+1=${fmt(affectedSpansMax)}）`,
    );
  } else {
    spanLines.push("帧 meta 缺 changed_max：改动量画不出结论");
  }
  spanLines.push(`支撑外 max|B| ${fmt(supportViolationMax)}（应恒 0）；Σ B_i − 1 = ${fmt(partitionUnity)}`);
  if (clamped) spanLines.push(`⚠ 请求 t=${fmt(tRaw)} 超出时间轴 [0, ${T_MAX}]：已夹到插入 ${fmt(insertCount)} 次`);
  if (note !== undefined) spanLines.push(`帧 note：${note}`);

  if (wide) {
    const lost = packColumn(
      scene,
      width,
      [mainLines, invarLines, spanLines],
      [pal.muted, pal.muted, clamped ? pal.warn : pal.muted],
      rowsLay.map((r) => r.rect[1]),
    );
    if (lost > 0) degraded.push(`${lost} 行结论在本画布下放不下（已并入上一栏或省略）`);
  } else {
    bandText(scene, rowsLay[0]!, width, mainLines, pal.muted, 2);
    bandText(scene, rowsLay[1]!, width, invarLines, pal.muted, 1);
    bandText(scene, rowsLay[2]!, width, spanLines, clamped ? pal.warn : pal.muted, 3);
  }

  // —— 角标 ——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  if (scene.dataSource.startsWith("backend+preview")) {
    scene.badges.push(placeBadge(previewBadge("拖动时按显示空间插值"), SLOT.preview, width));
  }
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(badge("autoscale", "|ΔC| 轴：自适应 log（u/2 参考线 = 「逐点不变」的判据）", 8, 8), SLOT.invarianceAxis, width),
  );
  scene.badges.push(
    placeBadge(badge("autoscale", "支撑轴：自适应 log（横轴是基函数下标，不是参数 u）", 8, 8), SLOT.supportAxis, width),
  );
  scene.badges.push(
    placeBadge(
      badge(
        "status",
        clipBadgeText(
          // 基函数层画不出来时必须放**最前面**：`clipBadgeText` 从尾部截断，
          // 放在末尾等于没写（只写进降级角标也会被后面的角标挤掉）。
          (basisOk ? "" : "⚠ basis 与 (n_basis, x_grid) 不自洽：基函数画不出来　") +
            `插入 ${fmt(insertCount)} 次　移动 #${fmt(moveIndex)}（Δ=${fmt(moveDelta)}）　` +
            `受影响 ${fmt(changedMeasure === undefined ? undefined : changedMeasure * 100, 3)}% 参数域`,
        ),
        8,
        8,
      ),
      SLOT.state,
      width,
    ),
  );
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  const bad = totalNonFinite([
    ctrlX,
    ctrlY,
    ctrlYMoved,
    knots,
    knotsBase,
    xGrid,
    curveOrig,
    curveMoved,
    curveBase,
    basis,
    basisIndex,
    supportLeft,
    supportRight,
    changed,
    changedMask,
    invariance,
    supportRes,
    affectedSpan,
    insertMarker,
  ]);
  applyOverflow(scene, bad, { slot: SLOT.overflow, width });
  // 文字层的两道新规（互相不重叠、不被陷阱面板压住）统一在这里收口。
  finishTexts(scene, width);
  return scene;
};

// --------------------------------------------------------------------------- //
// 局部工具
// --------------------------------------------------------------------------- //
function toNumbers(a: ArrayLike<number>): number[] {
  const out: number[] = [];
  for (let i = 0; i < a.length; i += 1) out.push(a[i] as number);
  return out;
}

/** manifest 里该轴的标题。 */
function axisTitle(id: string, fallback: string): string {
  return SPEC?.axes.find((a) => a.id === id)?.title ?? fallback;
}

/** manifest 里该轴的 ylabel。 */
function axisYlabel(id: string, fallback: string): string {
  return SPEC?.axes.find((a) => a.id === id)?.ylabel ?? fallback;
}
