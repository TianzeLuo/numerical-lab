/**
 * `smoothness_c0c1c2`（2.2 章）的 model：折线 C⁰ / PCHIP C¹ / 样条 C² 的接缝证据。
 *
 * 面板与图层（顺序即绘制顺序，测试按此断言）
 * ------------------------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | line | `main` | 折线（`y_lin`）、PCHIP（`y_pchip`）、样条（`y_spline`），x 用节点两侧加密网格 |
 * | 2 | scatter | `main` | 节点（`node_x`） |
 * | 3 | line | `d2` | 折线 S''≡0 + 样条/PCHIP 的「节点两侧」与「等距采样」各一对 |
 * | 4 | line | `jump` | `jump_d1_lin`（C⁰ 的破绽）/ `residual`（PCHIP 的 |dS''|）/ `jump_d2_spline` |
 * | 5 | line | `jump` | 参考线：`u/2` 地板 |
 *
 * 角标槽位：0 来源 / 1 主轴自适应 / 2 S'' 轴自适应 / 3 跳变轴自适应 log /
 * 4 数据与 BC 状态 / 5 前端预览（仅 backend+preview）/ 7 状态与降级。
 *
 * 三条坑，以及各自在图上的落点（这是本 viz 存在的理由）
 * --------------------------------------------------
 * 1. **等距采样把跳变连成斜坡**：`d2` 面板同时画两条 PCHIP 的二阶导——实线是
 *    「节点两侧分采」（跳变 = 竖直段），虚线是帧里那条 `d2_pchip_naive`（默认等距采样，
 *    跳变被相邻两点的连线抹成斜坡）。两条曲线**来自同一帧、同一个函数**，差别只在采样点。
 * 2. **折线的完美 0 线是假象**：`d2_lin` 恒为 0，所以折线的二阶导是一条贴在地板上的
 *    直线——它连 C¹ 都不是。它的破绽画在 `jump` 面板的 `jump_d1_lin` 上
 *    （实测 n=9 台阶数据、等距节点：|dS'| = 2.67，而 PCHIP 的 |dS''| = 64）。
 * 3. **跳变量级随节点分布变，但永远到不了 u**：`jump` 面板是 log 轴，
 *    `u/2` 参考线以下的点才可能是 C²；Chebyshev 节点会把 PCHIP 的跳变推到 85.1
 *    （更大），而不是更小——"更光滑"与"节点更密/更巧"是两件事。
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
  makeAxis,
  previewBadge,
  scatterLayer,
  sourceBadge,
} from "../../shared/src/scene.js";
import { BC_KEY, BC_NAME, enumName, fmt, FLOOR, linRange, logRange, NODES_KEY, totalNonFinite } from "./derive.js";
import {
  bandText,
  packColumn,
  contentBottom,
  finishTexts,
  hLine,
  panelRows,
  panelTop,
  rightColumnFits,
  tooNarrowNote,
  rightX,
} from "./layout.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "2.2";
const VIZ = "smoothness_c0c1c2";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "C0 / C1 / C2：二阶导的阶梯跳变";

const SLOT = {
  source: 0,
  axisMode: 1,
  d2Axis: 2,
  jumpAxis: 3,
  state: 4,
  preview: 5,
  /** 降级说明（规则 6：缺什么就说什么）。 */
  degraded: 6,
  /** 溢出徽标（`applyOverflow`）：与降级分开占槽，否则两者会叠在同一个盒子里。 */
  overflow: 7,
} as const;

const NODES_NAME: Record<string, string> = {
  uniform: "等距节点",
  chebyshev: "Chebyshev--Lobatto 节点",
};

export const smoothnessModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width } = canvasSize(opts);

  // —— 必需数组 ——
  const xStep = requireArray(frame, "x_step");
  const yLin = requireArray(frame, "y_lin");
  const yPchip = requireArray(frame, "y_pchip");
  const ySpline = requireArray(frame, "y_spline");
  const d2Lin = requireArray(frame, "d2_lin");
  const d2PchipStep = requireArray(frame, "d2_pchip_step");
  const d2SplineStep = requireArray(frame, "d2_spline_step");
  const d2PchipNaive = requireArray(frame, "d2_pchip_naive");
  const d2SplineNaive = requireArray(frame, "d2_spline_naive");
  const xNaive = requireArray(frame, "x_naive");
  const nodeX = requireArray(frame, "node_x");
  const jumpNodes = requireArray(frame, "jump_nodes");
  const jumpD1Lin = requireArray(frame, "jump_d1_lin");
  const jumpD2Pchip = requireArray(frame, "jump_d2_pchip");
  const jumpD2Spline = requireArray(frame, "jump_d2_spline");
  const residual = requireArray(frame, "residual");
  const nodalError = requireArray(frame, "error");
  requireSameLength(xStep, yLin, "帧数组 x_step 与 y_lin");
  requireSameLength(xStep, yPchip, "帧数组 x_step 与 y_pchip");
  requireSameLength(xStep, ySpline, "帧数组 x_step 与 y_spline");
  requireSameLength(xStep, d2Lin, "帧数组 x_step 与 d2_lin");
  requireSameLength(xStep, d2PchipStep, "帧数组 x_step 与 d2_pchip_step");
  requireSameLength(xStep, d2SplineStep, "帧数组 x_step 与 d2_spline_step");
  requireSameLength(xNaive, d2PchipNaive, "帧数组 x_naive 与 d2_pchip_naive");
  requireSameLength(xNaive, d2SplineNaive, "帧数组 x_naive 与 d2_spline_naive");
  requireSameLength(jumpNodes, jumpD1Lin, "帧数组 jump_nodes 与 jump_d1_lin");
  requireSameLength(jumpNodes, jumpD2Pchip, "帧数组 jump_nodes 与 jump_d2_pchip");
  requireSameLength(jumpNodes, jumpD2Spline, "帧数组 jump_nodes 与 jump_d2_spline");
  const pitfalls = requirePitfalls(frame);

  // —— 可选标注：缺了就降级 ——
  const degraded: string[] = [];
  // `time.mode="none"`：**不引用 tMax、不画时间轴控件**，scene.t 恒为 0。
  if (metaNumber(frame, "t") !== undefined && (metaNumber(frame, "t") as number) !== 0) {
    degraded.push("这是无时间轴的 viz（time.mode=none）：帧 meta.t 非 0，已按 0 处理");
  }
  const n = metaNumber(frame, "n");
  const epsUsed = metaNumber(frame, "eps");
  const epsRel = metaNumber(frame, "eps_rel");
  const datasetName = stringOf(frame, "dataset_name");
  if (datasetName === undefined) degraded.push("帧 strings 缺 dataset_name：状态角标只写 n 与 eps");
  const bc = enumName(stringOf(frame, "bc_name"), metaNumber(frame, "bc_index"), BC_KEY, BC_NAME);
  const nodes = enumName(
    undefined,
    metaNumber(frame, "nodes_index"),
    NODES_KEY,
    NODES_NAME,
  );
  if (!nodes.known) degraded.push("帧 meta 缺 nodes_index：不自行判断节点分布");
  const jumpPchipMax = metaNumber(frame, "jump_pchip_max");
  const jumpSplineMax = metaNumber(frame, "jump_spline_max");
  const jumpLinSlopeMax = metaNumber(frame, "jump_lin_slope_max");
  const errorSplineMax = metaNumber(frame, "error_spline_max");
  const errorPchipMax = metaNumber(frame, "error_pchip_max");
  const errorLinMax = metaNumber(frame, "error_lin_max");
  const energyLin = metaNumber(frame, "energy_lin");
  const energyPchip = metaNumber(frame, "energy_pchip");
  const energySpline = metaNumber(frame, "energy_spline");
  const nStepSamples = metaNumber(frame, "n_step_samples");
  if (nStepSamples === undefined) degraded.push("帧 meta 缺 n_step_samples：不报告加密采样点数");
  const naiveSamples = metaNumber(frame, "naive_samples");
  const note = stringOf(frame, "note");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: 0,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  // —— 版式：主曲线 + 二阶导 + 跳变量级（一列三行，行间 22px = 轴标题装饰带）——
  const rowsLay = panelRows(width, [214, 150, 110], panelTop(width), contentBottom(scene), undefined, 24);
  const mainRect = rowsLay[0]!.rect;
  const d2Rect = rowsLay[1]!.rect;
  const jumpRect = rowsLay[2]!.rect;

  // —— 主轴（manifest: autoscaled）——
  const allY: number[] = [...toNumbers(yLin), ...toNumbers(yPchip), ...toNumbers(ySpline)];
  const yLim = linRange(allY, [-1.2, 1.2], 5);
  const xLim: [number, number] = [Math.min(nodeX[0] as number, xStep[0] as number), Math.max(nodeX[nodeX.length - 1] as number, xStep[xStep.length - 1] as number)];
  const mainAxis = makeAxis("main", mainRect, {
    xlim: xLim,
    ylim: yLim,
    xlabel: "",
    ylabel: axisYlabel("main", "$y$"),
    title: axisTitle("main", "三条插值曲线"),
    autoscaled: true,
    xticks: niceTicks(xLim[0], xLim[1], Math.max(3, Math.round(mainRect[2] / 110))),
    yticks: niceTicks(yLim[0], yLim[1], 4),
  });

  // —— 二阶导轴（线性、自适应）。折线的 S''≡0，所以上下界由三条曲线一起定 ——
  const allD2: number[] = [
    ...toNumbers(d2Lin),
    ...toNumbers(d2PchipStep),
    ...toNumbers(d2SplineStep),
    ...toNumbers(d2PchipNaive),
    ...toNumbers(d2SplineNaive),
  ];
  const d2Lim = linRange(allD2, [-1, 1], 5);
  const d2Axis = makeAxis("d2", d2Rect, {
    xlim: xLim,
    ylim: d2Lim,
    xlabel: "",
    ylabel: axisYlabel("d2", "S''"),
    title: axisTitle("d2", "二阶导 S''（节点两侧分别采样）"),
    autoscaled: true,
    xticks: [],
    yticks: niceTicks(d2Lim[0], d2Lim[1], 4),
  });

  // —— 跳变轴（log、自适应）——
  const jumpLim = logRange([...toNumbers(jumpD1Lin), ...toNumbers(jumpD2Pchip), ...toNumbers(jumpD2Spline)], [FLOOR, 1e2]);
  const jumpAxis = makeAxis("jump", jumpRect, {
    xlim: xLim,
    ylim: jumpLim,
    yscale: "log",
    xlabel: "",
    ylabel: axisYlabel("jump", "|jump|"),
    title: axisTitle("jump", "节点处的跳变量级（log）"),
    autoscaled: true,
    xticks: [],
    yticks: niceTicks(jumpLim[0], jumpLim[1], 4, { log: true }),
  });

  scene.axes.push(mainAxis, d2Axis, jumpAxis);

  // —— 图层 1~2：主面板三条曲线 + 节点 ——
  scene.layers.push({
    kind: "line",
    axes: "main",
    series: [
      { x: xStep, y: yLin, color: pal.warn, width: 1.4, label: "折线（$y_lin$）" },
      { x: xStep, y: yPchip, color: pal.curves[1], width: 1.8, label: "PCHIP（$y_pchip$）" },
      { x: xStep, y: ySpline, color: pal.curves[0], width: 1.8, label: "样条（$y_spline$）" },
    ],
  });
  scene.layers.push(scatterLayer("main", nodeX, nodesSeriesY(nodeX, xStep, ySpline), { r: 3.2, color: pal.ink, label: "节点" }));

  // —— 图层 3：二阶导。实线=节点两侧分采（跳变是竖直段）；虚线=帧里的等距采样版本 ——
  scene.layers.push({
    kind: "line",
    axes: "d2",
    series: [
      { x: xStep, y: d2Lin, color: pal.warn, width: 1.6, label: "折线 $S''$≡0" },
      { x: xStep, y: d2SplineStep, color: pal.curves[0], width: 1.8, label: "样条 $S''$（节点两侧）" },
      { x: xNaive, y: d2SplineNaive, color: pal.curves[0], width: 1, dash: [3, 3], label: "样条 $S''$（等距采样）" },
      { x: xStep, y: d2PchipStep, color: pal.curves[1], width: 1.8, label: "PCHIP $S''$（节点两侧）" },
      { x: xNaive, y: d2PchipNaive, color: pal.curves[1], width: 1, dash: [3, 3], label: "PCHIP $S''$（等距采样）" },
    ],
  });
  hLine(scene, d2Axis, 0, pal.reference, { label: "$S''$ = 0" });

  // —— 图层 4~5：节点处的跳变量级（log）+ 地板参考线 ——
  scene.layers.push({
    kind: "line",
    axes: "jump",
    series: [
      {
        x: jumpNodes,
        y: jumpD1Lin,
        color: pal.warn,
        width: 1.6,
        marker: "o",
        label: "折线 $|dS'|$",
      },
      {
        x: jumpNodes,
        y: residual,
        color: pal.curves[1],
        width: 1.8,
        marker: "s",
        label: "PCHIP $|dS''|$（residual）",
      },
      {
        x: jumpNodes,
        y: jumpD2Spline,
        color: pal.curves[0],
        width: 1.8,
        marker: "s",
        label: "样条 $|dS''|$",
      },
    ],
  });
  hLine(scene, jumpAxis, FLOOR, pal.note, { label: "$u/2$" });

  // —— 文本：主轴与 S'' 之间的说明带 + 跳变轴下的结论带 ——
  // 说明文字一律写在所有轴右侧那一栏（见 `layout.ts`：轴间空隙里全是 y 刻度带）。
  const wide = rightColumnFits(width);
  const summary = paramSummary(frame, opts, [
    { name: "n", label: "$n$" },
    { name: "eps", label: "eps" },
  ]);
  degraded.push(...summary.degraded);

  const jumpLines: string[] = [
    `跳变：折线 ${fmt(jumpLinSlopeMax)}　PCHIP ${fmt(jumpPchipMax)}　样条 ${fmt(jumpSplineMax)}`,
    `地板线 u/2 = ${fmt(FLOOR)}：只有 C² 的 |dS''| 才可能落到它下面。`,
    `节点残差 ${fmt(errorLinMax)}/${fmt(errorPchipMax)}/${fmt(errorSplineMax)}　能量 ${fmt(energyLin)}/${fmt(energyPchip)}/${fmt(energySpline)}`,
  ];
  // 每行都压到 60 显示单位以内（= 一物理行）：折行会让"这块两行"的估算全错，
  // 而最后一块紧贴陷阱面板，一折行就要挤掉别的结论。
  const mainLines = [
    `${datasetName ?? "数据=?"}　${nodes.text}　n=${fmt(n)}　bc=${bc.known ? bc.text : "?"}`,
    `eps=${fmt(epsUsed)}（相对 ${fmt(epsRel, 3)}）：三条曲线过同一组点，差别只在接缝。`,
  ];
  const d2Lines = [
    `S''：实线 = 节点两侧分采，虚线 = 等距采样 ${fmt(naiveSamples)} 点`,
    "虚线把真跳变连成了斜坡：别把它读成「二阶导连续」。",
    "折线的 S''≡0 也一样：d2 恒为 0 却连 C¹ 都不是。",
  ];
  if (note !== undefined) d2Lines.push(`帧 note：${note}`);
  else d2Lines.push("帧未给出 note");
  if (wide) {
    const lost = packColumn(scene, width, [mainLines, d2Lines, jumpLines], [pal.muted, pal.muted, pal.muted], rowsLay.map((r) => r.rect[1]));
    if (lost > 0) degraded.push(`${lost} 行结论在本画布下放不下（已并入上一栏或省略）`);
  } else {
    bandText(scene, rowsLay[0]!, width, [`三条插值曲线：${datasetName ?? "数据=?"}`], pal.muted, 1);
    bandText(scene, rowsLay[1]!, width, ["S''：实线 = 分采，虚线 = 等距采样"], pal.muted, 1);
    bandText(scene, rowsLay[2]!, width, [jumpLines[0] as string], pal.muted, 2);
  }
  if (!wide) scene.texts.push(...tooNarrowNote([rightX(width), rowsLay[2]!.limit - 32, 160, 32], pal.warn, width));

  // 手写轴名（下面两条轴的 xticks 已清空，不会与数字刻度叠在同一条带里）。
  // —— 角标 ——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  if (scene.dataSource.startsWith("backend+preview")) {
    scene.badges.push(placeBadge(previewBadge("拖动时按显示空间插值"), SLOT.preview, width));
  }
  scene.badges.push(placeBadge(axisModeBadge(true), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(badge("autoscale", "S'' 轴：自适应线性（按本帧四条 d2 取整；自适应会掩盖量级变化）", 8, 8), SLOT.d2Axis, width),
  );
  scene.badges.push(
    placeBadge(badge("autoscale", "跳变轴：自适应 log（u/2 参考线是「C²」的判据）", 8, 8), SLOT.jumpAxis, width),
  );
  scene.badges.push(
    placeBadge(
      badge(
        "status",
        clipBadgeText(
          `n=${fmt(n)}　${nodes.text}　eps=${fmt(epsUsed)}（相对 ${fmt(epsRel, 3)}）　` +
            `PCHIP |dS''|=${fmt(jumpPchipMax)}　样条 |dS''|=${fmt(jumpSplineMax)}`,
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
    xStep,
    yLin,
    yPchip,
    ySpline,
    d2Lin,
    d2PchipStep,
    d2SplineStep,
    xNaive,
    d2PchipNaive,
    d2SplineNaive,
    nodeX,
    jumpNodes,
    jumpD1Lin,
    jumpD2Pchip,
    jumpD2Spline,
    residual,
    nodalError,
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

/**
 * 节点处曲线的 y：从**帧给的采样**里取与 `node_x` 最近的采样点，取不到就跳过。
 *
 * 为什么不去"插值出节点值"：`x_step` 本来就是节点两侧采样，节点本身不在网格上
 * （它是两侧的极限），所以这里只把"离节点最近的那个采样点"当作散点位置——
 * 散点的作用是标出节点在哪，不是提供数值（数值由曲线层负责）。
 */
function nodesSeriesY(nodeX: ArrayLike<number>, xStep: ArrayLike<number>, y: ArrayLike<number>): number[] {
  const out: number[] = [];
  const n = Math.min(xStep.length, y.length);
  for (let i = 0; i < nodeX.length; i += 1) {
    const target = nodeX[i] as number;
    if (!Number.isFinite(target)) continue;
    let best = -1;
    let bestD = Number.POSITIVE_INFINITY;
    for (let j = 0; j < n; j += 1) {
      const xv = xStep[j] as number;
      if (!Number.isFinite(xv)) continue;
      const d = Math.abs(xv - target);
      if (d < bestD) {
        bestD = d;
        best = j;
      }
    }
    if (best >= 0 && Number.isFinite(y[best] as number)) out.push(y[best] as number);
  }
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
