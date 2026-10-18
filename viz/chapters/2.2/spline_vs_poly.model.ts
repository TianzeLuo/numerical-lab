/**
 * `spline_vs_poly`（2.2 章）的 model：同一组数据、同一个 n 上，样条与高阶多项式的对照。
 *
 * 面板与图层（顺序即绘制顺序，测试按此断言）
 * ------------------------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | line | `error` | 两条误差曲线（`error_spline` / `error_poly`，log）+ 舍入地板参考线 |
 * | 2 | line | `growth` | 误差随 n 的增长（`loglogfit`：实测斜率与理论斜率必须同图，规则 2） |
 * | 3 | line | `main` | 被插函数 `y_exact`、样条 `y_spline`、多项式 `y_poly`（**裁到主轴内**） |
 * | 4 | line | `main` | 主轴 0 基线与上下边界参考线（多项式越界的位置在这里读出来） |
 * | 5 | scatter | `main` | 节点（`node_x`/`node_y`） |
 *
 * 角标槽位：0 来源 / 1 主轴固定 / 2 误差轴自适应 log / 3 增长轴自适应 log /
 * 4 状态（n + Lebesgue + 误差比）/ 5 前端预览（仅 backend+preview）/ 7 溢出与降级。
 *
 * 三条坑，以及各自在图上的落点
 * --------------------------
 * 1. **多项式在节点上精确、误差全在节点之间**：`main` 上多项式穿过每个节点（散点在
 *    两条曲线上重合），而 `error` 面板里它的误差曲线在节点处掉到地板上、节点之间冲到
 *    1.0 量级——把"节点上的残差"（`residual`，贴 u）单独画出来就是为了让这个对比可读。
 * 2. **主轴固定 + 溢出显式报告**：多项式冲出画面时**不**自动放大坐标轴，
 *    而是把它裁到画面内、数清楚裁掉了几个点（`meta.poly_max_abs` / `poly_min` 一律进说明带），
 *    并挂 `overflow` 徽标。自动缩放会把"爆炸"画成"形状差不多"。
 * 3. **地板以下的下降是假的**：`error` 面板里那条 `u/2` 参考线是 `meta.error_floor`
 *    （= u × max|f|）；曲线贴上去之后再往下走纯属舍入噪声，图上的下界也压在它附近。
 */

import { fitOrder } from "../../shared/src/loglogfit.js";
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
  loglogFitLayer,
  makeAxis,
  previewBadge,
  scatterLayer,
  sourceBadge,
} from "../../shared/src/scene.js";
import { BC_KEY, BC_NAME, clipSeries, countOutside, enumName, fmt, FLOOR, linRange, logRange, NODES_KEY, totalNonFinite } from "./derive.js";
import {
  bandText,
  packColumn,
  contentBottom,
  finishTexts,
  hLine,
  panelRows,
  panelTop,
  rightColumnFits,
  rightX,
  tooNarrowNote,
} from "./layout.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "2.2";
const VIZ = "spline_vs_poly";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "样条 vs 高阶多项式：同一组数据的稳定性";

const SLOT = {
  source: 0,
  axisMode: 1,
  errorAxis: 2,
  growthAxis: 3,
  state: 4,
  preview: 5,
  /** 降级说明。 */
  degraded: 6,
  /** 溢出徽标（`applyOverflow` / 手写的裁剪计数）。 */
  overflow: 7,
} as const;

const NODES_NAME: Record<string, string> = {
  uniform: "等距节点",
  chebyshev: "Chebyshev--Lobatto 节点",
};

/** 理论上样条误差是 O(h⁴)：画在增长面板上与实测斜率对照（规则 2）。 */
const THEORY_SLOPE = 4;

export const splineVsPolyModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width } = canvasSize(opts);

  // —— 必需数组 ——
  const xGrid = requireArray(frame, "x_grid");
  const yExact = requireArray(frame, "y_exact");
  const ySpline = requireArray(frame, "y_spline");
  const yPoly = requireArray(frame, "y_poly");
  const nodeX = requireArray(frame, "node_x");
  const nodeY = requireArray(frame, "node_y");
  const errSpline = requireArray(frame, "error_spline");
  const errPoly = requireArray(frame, "error_poly");
  const floorLine = requireArray(frame, "error_floor_line");
  const maxErrors = requireArray(frame, "error");
  const nodalRes = requireArray(frame, "residual");
  const nHistory = requireArray(frame, "n_history");
  const polyHistory = requireArray(frame, "error_poly_history");
  const splineHistory = requireArray(frame, "error_spline_history");
  requireSameLength(xGrid, yExact, "帧数组 x_grid 与 y_exact");
  requireSameLength(xGrid, ySpline, "帧数组 x_grid 与 y_spline");
  requireSameLength(xGrid, yPoly, "帧数组 x_grid 与 y_poly");
  requireSameLength(xGrid, errSpline, "帧数组 x_grid 与 error_spline");
  requireSameLength(xGrid, errPoly, "帧数组 x_grid 与 error_poly");
  requireSameLength(xGrid, floorLine, "帧数组 x_grid 与 error_floor_line");
  requireSameLength(nodeX, nodeY, "帧数组 node_x 与 node_y");
  requireSameLength(nodeX, nodalRes, "帧数组 node_x 与 residual");
  requireSameLength(nHistory, polyHistory, "帧数组 n_history 与 error_poly_history");
  requireSameLength(nHistory, splineHistory, "帧数组 n_history 与 error_spline_history");
  const pitfalls = requirePitfalls(frame);

  // —— 可选标注 ——
  const degraded: string[] = [];
  // `time.mode="none"`：不引用 tMax，`scene.t` 恒为 0。
  if (metaNumber(frame, "t") !== undefined && (metaNumber(frame, "t") as number) !== 0) {
    degraded.push("这是无时间轴的 viz（time.mode=none）：帧 meta.t 非 0，已按 0 处理");
  }
  const n = metaNumber(frame, "n");
  const errorFloor = metaNumber(frame, "error_floor");
  const errorSplineMax = metaNumber(frame, "error_spline");
  const errorPolyMax = metaNumber(frame, "error_poly");
  const errorRatio = metaNumber(frame, "error_ratio");
  const nodalResidualMax = metaNumber(frame, "nodal_residual_max");
  const lebesgue = metaNumber(frame, "lebesgue_constant");
  const polyMaxAbs = metaNumber(frame, "poly_max_abs");
  const polyMin = metaNumber(frame, "poly_min");
  const energySpline = metaNumber(frame, "energy_spline");
  const energyPoly = metaNumber(frame, "energy_poly");
  const energyRatio = metaNumber(frame, "energy_ratio");
  const frameOverflow = metaNumber(frame, "overflow");
  const functionName = stringOf(frame, "function_name");
  if (functionName === undefined) degraded.push("帧 strings 缺 function_name：状态角标只用 meta 的编号");
  const bc = enumName(stringOf(frame, "bc_name"), metaNumber(frame, "bc_index"), BC_KEY, BC_NAME);
  const nodes = enumName(undefined, metaNumber(frame, "nodes_index"), NODES_KEY, NODES_NAME);
  if (!nodes.known) degraded.push("帧 meta 缺 nodes_index：不自行判断节点分布");
  const note = stringOf(frame, "note");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: 0,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  // —— 版式：误差 / 增长在上（先看结论），主面板在下（固定坐标、看形状） ——
  // 行高与说明文字的位置由 `panelRows` 统一算（保证不越过陷阱面板）。
  const rowsLay = panelRows(width, [128, 136, 136], panelTop(width), contentBottom(scene), undefined, 24);
  const errRect = rowsLay[0]!.rect;
  const growRect = rowsLay[1]!.rect;
  const mainRect = rowsLay[2]!.rect;

  // —— 主轴：**固定坐标**（manifest: autoscaled=false）。范围由被插函数与样条定：
  //     多项式被**显式裁到画面内**并计数，而不是让它把坐标轴撑开（见文件头第 2 条）。 ——
  const fixedValues: number[] = [...toNumbers(yExact), ...toNumbers(ySpline)];
  const yLim = linRange(fixedValues, [-1.2, 1.2], 5);
  const xLim: [number, number] = [xGrid[0] as number, xGrid[xGrid.length - 1] as number];
  const clipped = clipSeries(yPoly, yLim[0], yLim[1]);
  const outside = countOutside(yPoly, yLim[0], yLim[1]);
  const mainAxis = makeAxis("main", mainRect, {
    xlim: xLim,
    ylim: yLim,
    xlabel: "",
    ylabel: axisYlabel("main", "y"),
    title: axisTitle("main", "函数、样条与多项式（固定坐标）"),
    autoscaled: false,
    xticks: niceTicks(xLim[0], xLim[1], Math.max(3, Math.round(mainRect[2] / 110))),
    yticks: niceTicks(yLim[0], yLim[1], 4),
  });

  // —— 误差轴：log、自适应。下界压在**帧给的地板**上，不压到更低的假区域 ——
  const errLo = errorFloor !== undefined && errorFloor > 0 ? errorFloor : FLOOR;
  const errYLim = logRange([...toNumbers(maxErrors), errLo], [errLo, 1]);
  const errAxis = makeAxis("error", errRect, {
    xlim: xLim,
    ylim: errYLim,
    yscale: "log",
    xlabel: "",
    ylabel: axisYlabel("error", "|误差|"),
    title: axisTitle("error", "两条误差曲线与地板（log）"),
    autoscaled: true,
    xticks: [],
    yticks: niceTicks(errYLim[0], errYLim[1], 4, { log: true }),
  });

  // —— 增长轴：log-log、自适应（规则 2：实测斜率 + 理论斜率同时出现） ——
  const histErr: number[] = [];
  for (let i = 0; i < polyHistory.length; i += 1) histErr.push(polyHistory[i] as number);
  const growYLim = logRange(histErr, [FLOOR, 1], 4);
  const growAxis = makeAxis("growth", growRect, {
    xlim: [
      Math.max(1, Math.min(...toNumbers(nHistory)) - 1),
      Math.max(2, Math.max(...toNumbers(nHistory)) + 1),
    ],
    ylim: growYLim,
    xscale: "log",
    yscale: "log",
    xlabel: "",
    ylabel: axisYlabel("growth", "max|误差|"),
    title: axisTitle("growth", "误差随 n 的增长（log-log）"),
    autoscaled: true,
    xticks: [],
    yticks: niceTicks(growYLim[0], growYLim[1], 4, { log: true }),
  });

  scene.axes.push(errAxis, growAxis, mainAxis);

  // —— 图层 1：误差曲线（log）——
  scene.layers.push({
    kind: "line",
    axes: "error",
    series: [
      { x: xGrid, y: errPoly, color: pal.curves[1], width: 1.4, label: "|误差|（error_poly）" },
      { x: xGrid, y: errSpline, color: pal.curves[0], width: 1.8, label: "|误差|（error_spline）" },
      { x: xGrid, y: floorLine, color: pal.note, width: 1, dash: [5, 4], label: "地板（error_floor_line）" },
    ],
  });

  // —— 图层 2：增长面板（loglogfit；斜率必须有限，否则退化成普通折线并降级）——
  const histX: number[] = [];
  for (let i = 0; i < nHistory.length; i += 1) histX.push(nHistory[i] as number);
  const histSpline: number[] = [];
  for (let i = 0; i < splineHistory.length; i += 1) histSpline.push(splineHistory[i] as number);
  let measured = Number.NaN;
  if (histX.length >= 3 && histErr.length >= 3) {
    try {
      measured = fitOrder(histX, histErr).slope;
    } catch (err) {
      degraded.push(`增长面板：多项式历史的幂律拟合失败（${(err as Error).message.slice(0, 40)}…），只画点不画理论线`);
    }
  }
  if (Number.isFinite(measured)) {
    scene.layers.push(
      loglogFitLayer("growth", histX, histErr, measured, THEORY_SLOPE, {
        seriesLabel: `多项式实测斜率 ${fmt(measured, 3)}`,
        theoryLabel: `样条理论 O(h^${THEORY_SLOPE})`,
      }),
    );
  } else {
    scene.layers.push({
      kind: "line",
      axes: "growth",
      series: [{ x: histX, y: histErr, color: pal.curves[1], width: 1.6, marker: "o", label: "多项式" }],
    });
  }
  scene.layers.push({
    kind: "line",
    axes: "growth",
    series: [{ x: histX, y: histSpline, color: pal.curves[0], width: 1.6, marker: "o", label: "样条" }],
  });

  // —— 图层 3~5：主面板（固定坐标；多项式已裁到画面内并计数） ——
  scene.layers.push({
    kind: "line",
    axes: "main",
    series: [
      { x: xGrid, y: yExact, color: pal.ink, width: 1.2, dash: [6, 4], label: "f(x)（y_exact）" },
      { x: xGrid, y: clipped.values, color: pal.curves[1], width: 1.4, label: "Lagrange 多项式（已裁）" },
      { x: xGrid, y: ySpline, color: pal.curves[0], width: 2, label: "三次样条" },
    ],
  });
  hLine(scene, mainAxis, 0, pal.reference, { label: "y = 0" });
  scene.layers.push(scatterLayer("main", nodeX, nodeY, { r: 3.2, color: pal.marker, label: "节点" }));

  // —— 文本 ——
  const wide = rightColumnFits(width);
  const summary = paramSummary(frame, opts, [{ name: "n", label: "n" }]);
  degraded.push(...summary.degraded);
  const errorLines = [
    `最大误差：样条 ${fmt(errorSplineMax)}　多项式 ${fmt(errorPolyMax)}（${fmt(errorRatio, 3)} 倍）`,
    `地板 ${fmt(errorFloor)}（= u·max|f|）；节点残差最大 ${fmt(nodalResidualMax)}——多项式在节点上是精确的。`,
    "误差全长在节点之间：只看节点值会得出「多项式更准」的反结论。",
  ];
  const growLines: string[] = [
    `${functionName ?? "被插函数=?"}　${nodes.text}　bc=${bc.known ? bc.text : "?"}`,
    `多项式实测斜率 ${fmt(measured, 3)} vs 样条理论 ${THEORY_SLOPE}（O(h⁴)）；Lebesgue 常数 ${fmt(lebesgue, 3)}`,
    `max|误差| 随 n：样条 ${fmt(histSpline[0])}→${fmt(histSpline[histSpline.length - 1])}，` +
      `多项式 ${fmt(histErr[0])}→${fmt(histErr[histErr.length - 1])}`,
  ];
  const mainLines: string[] = [];
  const overflowNote = outside > 0 || frameOverflow === 1;
  mainLines.push(
    `多项式峰值 max|p| = ${fmt(polyMaxAbs)}，min = ${fmt(polyMin)}`,
    outside > 0
      ? `本图 ${fmt(outside)} 个采样点被裁到主轴边界：主轴**不**自动放大（固定坐标），这是 Runge 现象的本来面目。`
      : "多项式全部落在固定主轴范围内（本轮没有溢出）。",
  );
  mainLines.push(
    `能量：样条 ${fmt(energySpline)}　多项式 ${fmt(energyPoly)}` +
      (energyRatio !== undefined ? `（${fmt(energyRatio, 3)} 倍）` : "（比值无定义）") +
      "；折线能量 0 但不在 H² 里。",
  );
  if (note !== undefined) mainLines.push(`帧 note：${note}`);
  if (wide) {
    const lost = packColumn(
      scene,
      width,
      [errorLines, growLines, mainLines],
      [pal.muted, pal.muted, overflowNote ? pal.warn : pal.muted],
      rowsLay.map((r) => r.rect[1]),
    );
    if (lost > 0) degraded.push(`${lost} 行结论在本画布下放不下（已并入上一栏或省略）`);
  } else {
    bandText(scene, rowsLay[0]!, width, [`|误差|：样条 ${fmt(errorSplineMax)} / 多项式 ${fmt(errorPolyMax)}`], pal.muted, 2);
    bandText(scene, rowsLay[1]!, width, [growLines[1] as string], pal.muted, 1);
    bandText(scene, rowsLay[2]!, width, [mainLines[1] as string], overflowNote ? pal.warn : pal.muted, 2);
  }
  if (!wide) scene.texts.push(...tooNarrowNote([rightX(width), rowsLay[2]!.limit - 32, 160, 32], pal.warn, width));

  // 手写轴名（三条轴的 xticks 里，主轴保留数字刻度、其余两条清空）。

  // —— 角标 ——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  if (scene.dataSource.startsWith("backend+preview")) {
    scene.badges.push(placeBadge(previewBadge("拖动时按显示空间插值"), SLOT.preview, width));
  }
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(badge("autoscale", "误差轴：自适应 log（下界压在帧给的舍入地板上）", 8, 8), SLOT.errorAxis, width),
  );
  scene.badges.push(
    placeBadge(
      badge("autoscale", "增长轴：自适应 log-log（实测斜率与理论斜率同图，规则 2）", 8, 8),
      SLOT.growthAxis,
      width,
    ),
  );
  scene.badges.push(
    placeBadge(
      badge(
        "status",
        clipBadgeText(
          `n=${fmt(n)}　误差比 ${fmt(errorRatio, 3)}×　Lebesgue ${fmt(lebesgue, 3)}　` +
            `节点残差 ${fmt(nodalResidualMax)}`,
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
    xGrid,
    yExact,
    ySpline,
    yPoly,
    nodeX,
    nodeY,
    errSpline,
    errPoly,
    floorLine,
    maxErrors,
    nodalRes,
    nHistory,
    polyHistory,
    splineHistory,
  ]);
  // 规则 1：帧里的非有限值 + **被裁掉的多项式点数**（`checkOverflow` 看不见裁过的数据，
  // 所以这里由 model 自己把"裁了几个点"写进 note，再交给 `applyOverflow`）。
  applyOverflow(scene, bad, { slot: SLOT.overflow, width });
  // 文字层的两道新规（互相不重叠、不被陷阱面板压住）统一在这里收口。
  finishTexts(scene, width);
  if (outside > 0) {
    const extra = `多项式有 ${fmt(outside)} 个采样点超出固定主轴 [${fmt(yLim[0])}, ${fmt(yLim[1])}]（峰值 ${fmt(polyMaxAbs)}），已裁到边界并如实报告`;
    scene.overflow = { flag: true, note: scene.overflow.note ? `${scene.overflow.note}；${extra}` : extra };
    if (!scene.badges.some((b) => b.kind === "overflow")) {
      scene.badges.push(placeBadge(badge("overflow", `溢出：${extra}`, 8, 8), SLOT.overflow, width));
    }
  }
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
