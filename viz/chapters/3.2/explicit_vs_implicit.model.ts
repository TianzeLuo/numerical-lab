/**
 * `explicit_vs_implicit`（3.2 章）的 model：步长 / 误差 / 计算量三向对比。
 *
 * 面板与图层（顺序即绘制顺序，测试按此断言）
 * ------------------------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | line | `main` | 精确解 cos t（橙虚线） |
 * | 2 | line | `main` | 显式解（蓝） |
 * | 3 | line | `main` | 隐式解（绿） |
 * | 4 | scatter | `main` | 显式方法的步点地毯（到此为止走了多少步） |
 * | 5 | scatter | `main` | 隐式方法的步点地毯 |
 * | 6 | bars | `cost` | 四组代价柱子：步数 / 函数求值 / LU / Newton（显式 vs 隐式） |
 * | 7 | line | `aux` | 显式误差随 t（对数） |
 * | 8 | line | `aux` | 隐式误差随 t（对数，虚线） |
 * | 9 | line | `aux` | 容差 tol（水平参考线，帧 meta.tol） |
 * | 10 | scatter | `aux` | 当前时刻的两个误差标记 |
 *
 * 时间轴语义：**真实时间 t**（manifest `time.mode="scrub"`，`tMax=2π`）。解数组与误差数组
 * 都只到 `meta.t_now`；柱子的高度也是"到目前为止"的累计量——拖着时间轴，它们会一起长大。
 *
 * 三条刻意的决定
 * --------------
 * 1. **柱子用对数纵轴**：步数 3141 与 LU 31 差两个数量级，线性轴会把后两根柱子压成一条线。
 *    代价是**显式方法的 LU/Newton 恒为 0**（帧数组里就是 0，不是缺省），对数轴上画不出来——
 *    这在文本里逐条写明，且"0"本身是结论（显式方法没有线性系统要解）。
 * 2. **`main` 固定 ±2 且截断要报账**：越界样本不画，第一个越界位置放一枚标记，
 *    点数与真实 `max|y|`（帧 meta）写进文本与溢出说明。截断的曲线不许画成贴着上边的水平线。
 * 3. **第三个面板（`cost`）是模型自加的**：manifest 只声明 `main`/`aux`，但"三向对比"里的
 *    "计算量"只有柱子说得清（`metric_*` 数组本来就是为它准备的）；它画的是帧里的数组，
 *    不是新物理量。窄画布下它退到下面，布局由 `chapterLayout` 统一处理。
 */

import { formatNumber } from "../../shared/src/axes.js";
import { countNonFinite, dataToPixel } from "../../shared/src/geometry.js";
import {
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
  axisModeBadge,
  badge,
  barsLayer,
  criticalBadge,
  lineLayer,
  makeAxis,
  scatterLayer,
  series,
  sourceBadge,
  text,
} from "../../shared/src/scene.js";
import {
  applyOverflowNote,
  axisTitleOf,
  axisYlabelOf,
  chapterLayout,
  clipRuns,
  logRange,
  textBlock,
  textBottom,
  textTop,
  unitBudget,
} from "./_panel.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "3.2";
const VIZ = "explicit_vs_implicit";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "显式 vs 隐式：步长 / 误差 / 计算量三向对比";

const SLOT = {
  source: 0,
  axisMode: 1,
  auxAxis: 2,
  critical: 3,
  status: 4,
  /** 第二个临界值（本章只有 viz 4 有两个：tol 与 stiffness_ratio）。 */
  critical2: 5,
  overflow: 6,
  degraded: 7,
} as const;

/** manifest 声明主轴"固定纵轴 ±2"。 */
const MAIN_LIM: [number, number] = [-2, 2];
const RUG_EXPLICIT = -1.90;
const RUG_IMPLICIT = -1.76;
/** 四个代价指标的类别名（写死在模型里：`metric_x` 只给下标，帧里没有类别字符串）。 */
const METRIC_NAMES = ["步数", "函数求值", "LU 分解", "Newton 迭代"];

export const explicitVsImplicitModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  const tExplicit = requireArray(frame, "t_explicit");
  const yExplicit = requireArray(frame, "y_explicit");
  const tImplicit = requireArray(frame, "t_implicit");
  const yImplicit = requireArray(frame, "y_implicit");
  const tExact = requireArray(frame, "t_exact");
  const yExact = requireArray(frame, "y_exact");
  const error = requireArray(frame, "error");
  const errorImplicit = requireArray(frame, "error_implicit");
  const errorMark = requireArray(frame, "error_mark");
  const errorImplicitMark = requireArray(frame, "error_implicit_mark");
  const metricX = requireArray(frame, "metric_x");
  const metricExplicit = requireArray(frame, "metric_explicit");
  const metricImplicit = requireArray(frame, "metric_implicit");
  const toleranceMark = frame.has("tolerance_mark") ? requireArray(frame, "tolerance_mark") : undefined;
  const costX = frame.has("cost_x") ? requireArray(frame, "cost_x") : undefined;
  const costExplicit = frame.has("cost_explicit") ? requireArray(frame, "cost_explicit") : undefined;
  const costImplicit = frame.has("cost_implicit") ? requireArray(frame, "cost_implicit") : undefined;
  const costCrossX = frame.has("cost_cross_x") ? requireArray(frame, "cost_cross_x") : undefined;
  const costCrossY = frame.has("cost_cross_y") ? requireArray(frame, "cost_cross_y") : undefined;

  requireSameLength(tExplicit, yExplicit, "帧数组 t_explicit 与 y_explicit");
  requireSameLength(tImplicit, yImplicit, "帧数组 t_implicit 与 y_implicit");
  requireSameLength(tExact, yExact, "帧数组 t_exact 与 y_exact");
  requireSameLength(tExplicit, error, "帧数组 t_explicit 与 error");
  requireSameLength(tImplicit, errorImplicit, "帧数组 t_implicit 与 error_implicit");
  requireSameLength(metricX, metricExplicit, "帧数组 metric_x 与 metric_explicit");
  requireSameLength(metricX, metricImplicit, "帧数组 metric_x 与 metric_implicit");
  const pitfalls = requirePitfalls(frame);

  const degraded: string[] = [];
  const tMeta = metaNumber(frame, "t");
  if (tMeta === undefined) degraded.push("帧 meta 缺 t：scene.t 用请求值代替（两者可能不同）");
  const tNow = tMeta ?? opts.t;
  const tEnd = metaNumber(frame, "t_end") ?? Math.max(lastOf(tExact), lastOf(tExplicit));
  const tol = metaNumber(frame, "tol");
  if (tol === undefined) degraded.push("帧 meta 缺 tol：不画容差参考线（不按参数默认值猜）");
  const ratio = metaNumber(frame, "stiffness_ratio");
  const lam = metaNumber(frame, "lam");
  const hExplicit = metaNumber(frame, "h_explicit");
  const hExplicitTheory = metaNumber(frame, "h_explicit_theory");
  const hImplicit = metaNumber(frame, "h_implicit");
  const hRatio = metaNumber(frame, "h_ratio");
  const nStepsExplicit = metaNumber(frame, "n_steps_explicit");
  const nStepsImplicit = metaNumber(frame, "n_steps_implicit");
  const stepsRatio = metaNumber(frame, "steps_ratio");
  const errExplicit = metaNumber(frame, "error_explicit");
  const errImplicit = metaNumber(frame, "error_implicit");
  const errExplicitSpan = metaNumber(frame, "error_explicit_span");
  const errImplicitSpan = metaNumber(frame, "error_implicit_span");
  const maxAbsY = metaNumber(frame, "max_abs_y_explicit");
  const truncFlag = (metaNumber(frame, "truncated") ?? 0) > 0.5;
  const displayCeiling = metaNumber(frame, "display_ceiling");
  const startupSteps = metaNumber(frame, "startup_steps");
  const tolFallback = (metaNumber(frame, "tolerance_fallback") ?? 0) > 0.5;
  const crossLogTol = metaNumber(frame, "cost_cross_log10_tol");
  const crossFound = (metaNumber(frame, "cost_cross_found") ?? 0) > 0.5;
  const costImplicitAtTol = metaNumber(frame, "cost_implicit_at_tol");
  const costExplicitConst = metaNumber(frame, "cost_explicit_const");
  const implicitErrAtPlanned = metaNumber(frame, "implicit_error_at_planned_step");
  const roundoffFloor = metaNumber(frame, "roundoff_floor");
  const note = stringOf(frame, "note");
  if (note === undefined) degraded.push("帧 strings 缺 note：状态角标不自行编造结论");
  const crossoverNote = stringOf(frame, "crossover_note");
  const explicitName = stringOf(frame, "explicit_method");
  const implicitName = stringOf(frame, "implicit_method");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: tNow,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  const xlim: [number, number] = tEnd > 0 ? [0, tEnd] : [0, 1];
  const layout = chapterLayout(width, height, { rightShares: [0.46, 0.54], gutter: 0, minRight: 220 });
  const mainRect = layout.main;
  const costRect = layout.right[0];
  const auxRect = layout.right[1];
  // 柱状面板底部留 24px 写类别名（"步数 / 函数求值 / LU / Newton"）：类别名不能放进刻度带
  // （`AxisOptions` 没有 xtickLabels），也不能放到面板外面——下面紧挨着 aux 面板的轴标题带。
  const COST_LABEL_STRIP = 24;
  const costAxisRect: [number, number, number, number] = [costRect[0], costRect[1], costRect[2], Math.max(60, costRect[3] - COST_LABEL_STRIP)];
  scene.axes.push(
    makeAxis("main", mainRect, {
      xlim,
      ylim: MAIN_LIM,
      xlabel: "t（一个周期；曲线只画到当前 t）",
      ylabel: axisYlabelOf(SPEC?.axes, "main", "y(t)"),
      title: axisTitleOf(SPEC?.axes, "main", "解曲线（固定纵轴 ±2）"),
      autoscaled: false,
    }),
  );
  // 代价柱：对数纵轴（步数 3141 与 LU 31 差两个数量级），基线取轴底——0 值柱子画不出来，
  // 由文本逐条说明（"显式方法没有 LU/Newton"本身是结论，不是缺数据）。
  const metricPos = positiveRange([metricExplicit, metricImplicit]);
  const costYlim: [number, number] = metricPos === undefined ? [1, 10] : logRange(metricPos[0], metricPos[1]);
  if (metricPos === undefined) degraded.push("metric_explicit/metric_implicit 里没有 >0 的值：代价柱画不出来");
  scene.axes.push(
    makeAxis("cost", costAxisRect, {
      xlim: [minOf(metricX) - 0.6, maxOf(metricX) + 0.6],
      ylim: costYlim,
      yscale: "log",
      xticks: [],
      xlabel: "",
      ylabel: "次数",
      title: "到此为止的计算量（对数纵轴，四组各自读数）",
      autoscaled: true,
    }),
  );
  const errPos = finitePositiveMinMax(error);
  const errImplPos = finitePositiveMinMax(errorImplicit);
  const floorPos = roundoffFloor !== undefined && roundoffFloor > 0 ? roundoffFloor : Number.POSITIVE_INFINITY;
  const errLoRaw = Math.min(errPos?.[0] ?? Number.POSITIVE_INFINITY, errImplPos?.[0] ?? Number.POSITIVE_INFINITY, floorPos);
  const errHiRaw = Math.max(errPos?.[1] ?? 0, errImplPos?.[1] ?? 0, tol ?? 0);
  const auxYlim: [number, number] =
    Number.isFinite(errLoRaw) && errHiRaw > 0 ? logRange(errLoRaw, errHiRaw) : [1e-18, 1];
  if (!(Number.isFinite(errLoRaw) && errHiRaw > 0)) degraded.push("error/error_implicit 里没有 >0 的有限值：副轴用占位范围，误差曲线画不出来");
  scene.axes.push(
    makeAxis("aux", auxRect, {
      xlim,
      ylim: auxYlim,
      yscale: "log",
      xlabel: "t（与主图同一条时间轴）",
      ylabel: axisYlabelOf(SPEC?.axes, "aux", "误差 / 次数"),
      title: axisTitleOf(SPEC?.axes, "aux", "显式/隐式误差（对数）与容差"),
      autoscaled: true,
    }),
  );
  const costAxis = scene.axes[1];
  const auxAxis = scene.axes[2];

  // —— 主面板：三条曲线 + 两排步点地毯 ——
  for (const run of clipRuns(tExact, yExact, MAIN_LIM[0], MAIN_LIM[1]).runs) {
    scene.layers.push(lineLayer("main", series(run.x, run.y, pal.exact, { width: 1.4, dash: [6, 3] })));
  }
  const explClip = clipRuns(tExplicit, yExplicit, MAIN_LIM[0], MAIN_LIM[1]);
  for (const run of explClip.runs) {
    scene.layers.push(lineLayer("main", series(run.x, run.y, pal.curves[0], { width: 1.2 })));
  }
  const implClip = clipRuns(tImplicit, yImplicit, MAIN_LIM[0], MAIN_LIM[1]);
  for (const run of implClip.runs) {
    scene.layers.push(lineLayer("main", series(run.x, run.y, pal.curves[1], { width: 1.6 })));
  }
  // 步点地毯：本帧没有 step_marks_*，但 `t_explicit`/`t_implicit` **本身就是**每步的落点时间，
  // 直接拿来当 x（y 是显示用的常量）——不重算、不插值。
  if (tExplicit.length > 0) {
    scene.layers.push(
      scatterLayer("main", tExplicit, new Float64Array(tExplicit.length).fill(RUG_EXPLICIT), {
        r: 0.7,
        color: pal.curves[0],
        alpha: 0.55,
        label: `显式步点（${tExplicit.length} 步）`,
      }),
    );
  }
  if (tImplicit.length > 0) {
    scene.layers.push(
      scatterLayer("main", tImplicit, new Float64Array(tImplicit.length).fill(RUG_IMPLICIT), {
        r: 1.7,
        color: pal.curves[1],
        alpha: 0.9,
        label: `隐式步点（${tImplicit.length} 步）`,
      }),
    );
  }

  // —— 代价柱（两组错开半个柱宽）——
  const halfBar = 0.19;
  // **只画 >0 的柱子**：对数轴上 0 落在轴外，`checkOverflow` 会把它记成"图层越界"
  // （而且那根柱子本来就画不出来）。恒 0 的项由文本与溢出说明逐条交代。
  const barsExplicit = positiveBars(metricX, metricExplicit, -halfBar);
  const barsImplicit = positiveBars(metricX, metricImplicit, halfBar);
  if (barsExplicit.x.length > 0) {
    scene.layers.push(
      barsLayer("cost", barsExplicit.x, barsExplicit.v, { width: 0.36, color: pal.curves[0], baseline: costYlim[0], label: explicitName ?? "显式" }),
    );
  }
  if (barsImplicit.x.length > 0) {
    scene.layers.push(
      barsLayer("cost", barsImplicit.x, barsImplicit.v, { width: 0.36, color: pal.curves[1], baseline: costYlim[0], label: implicitName ?? "隐式" }),
    );
  }

  // —— 副轴：两条误差曲线 + 容差线 + 当前标记 ——
  const errClip = clipRuns(tExplicit, error, auxAxis.ylim[0], auxAxis.ylim[1], (v) => v > 0);
  for (const run of errClip.runs) {
    scene.layers.push(lineLayer("aux", series(run.x, run.y, pal.residual, { width: 1.4 })));
  }
  const errImplClip = clipRuns(tImplicit, errorImplicit, auxAxis.ylim[0], auxAxis.ylim[1], (v) => v > 0);
  for (const run of errImplClip.runs) {
    scene.layers.push(lineLayer("aux", series(run.x, run.y, pal.curves[1], { width: 1.4, dash: [5, 3] })));
  }
  if (errClip.runs.length === 0 && errImplClip.runs.length === 0) degraded.push("两条误差曲线在对数范围内都没有可画的点");
  if (tol !== undefined && tol >= auxAxis.ylim[0] && tol <= auxAxis.ylim[1]) {
    scene.layers.push(lineLayer("aux", series([xlim[0], xlim[1]], [tol, tol], pal.reference, { width: 1, dash: [2, 3] })));
  } else if (tol !== undefined) {
    degraded.push(`帧 meta.tol=${formatNumber(tol, 3)} 落在副轴范围之外：容差线没画（数值写在文本里）`);
  }
  const markY = errorMark.length > 0 ? errorMark[0] : undefined;
  const markImpY = errorImplicitMark.length > 0 ? errorImplicitMark[0] : undefined;
  if (markY !== undefined && Number.isFinite(markY) && markY > 0 && tNow >= xlim[0] && tNow <= xlim[1]) {
    scene.layers.push(scatterLayer("aux", [tNow], [markY], { r: 4.5, color: pal.marker, label: "当前时刻显式误差" }));
  }
  if (markImpY !== undefined && Number.isFinite(markImpY) && markImpY > 0 && tNow >= xlim[0] && tNow <= xlim[1]) {
    scene.layers.push(scatterLayer("aux", [tNow], [markImpY], { r: 4.5, color: pal.curves[1], label: "当前时刻隐式误差" }));
  }

  // —— 文本 ——
  const mainHead: string[] = [];
  if (lam !== undefined) mainHead.push(`λ=${formatNumber(lam, 4)}`);
  if (ratio !== undefined) mainHead.push(`刚性比 r=${formatNumber(ratio, 4)}`);
  mainHead.push(`t=${formatNumber(tNow, 3)}/${formatNumber(tEnd, 3)}`);
  const mainLegend: string[] = [
    `固定纵轴 ±2：${explClip.nClipped + implClip.nClipped > 0 ? `有 ${explClip.nClipped + implClip.nClipped} 个样本越界已截断（真实 max|y_显式|=${maxAbsY === undefined ? "?" : formatNumber(maxAbsY, 5)}）` : `全部样本都在 ±2 内（真实 max|y_显式|=${maxAbsY === undefined ? "?" : formatNumber(maxAbsY, 5)}）`}`,
    `下沿两排短竖线=每一步的落点（x 直接取帧里的 t_explicit/t_implicit）：显式 ${tExplicit.length} 个、隐式 ${tImplicit.length} 个`,
  ];
  // 左列文字**不许**压到右列第一条轴（cost）的左侧装饰带（刻度数字 + ylabel，宽 56px）：
  // 校验器的"规则 8"就是拿这个盒子判的（面板本身不是它看得见的盒子）。
  const costLeftBand = layout.columns ? 56 + 4 : 0;
  const mainTextPx = layout.columns ? Math.max(120, costRect[0] - costLeftBand - (mainRect[0] + 6) - 2) : mainRect[2] - 12;
  // `textBlock` 的 budget 是**显示宽度单位**（CJK=2、ASCII=1），不是像素——换算见 `unitBudget`。
  const mainTextMax = Math.max(24, Math.floor(mainTextPx / 7));
  scene.texts.push(
    ...textBlock(mainRect[0] + 6, textTop(mainRect), textBottom(mainRect), [mainHead.join("　"), ...mainLegend], pal.ink, mainTextMax, "top"),
  );

  // 类别名：柱子轴刻意清空 xticks/xlabel，类别名用文本层手写（见布局契约）。
  const costLines: string[] = [];
  costLines.push(`${explicitName ?? "显式"}（蓝） vs ${implicitName ?? "隐式"}（绿）`);
  for (let i = 0; i < Math.min(metricX.length, METRIC_NAMES.length); i += 1) {
    const e = metricExplicit[i];
    const m = metricImplicit[i];
    costLines.push(`${METRIC_NAMES[i]}：${formatNumber(e, 4)} vs ${formatNumber(m, 4)}${e === 0 || m === 0 ? "（0 = 该方法没有这一步，不是缺数据）" : ""}`);
  }
  // 文字块的底线再抬 26px：把面板底部那条类别名带让出来（两条文本叠在一起校验器不报，但人会看到）。
  scene.texts.push(
    ...textBlock(costRect[0] + 6, textTop(costRect), textBottom(costRect) - 26, costLines, pal.muted, unitBudget(costRect[2]), "top"),
  );
  // 类别名画在柱子轴的**正下方**（该轴刻意清空 xticks 与 xlabel，那两条装饰带不存在）。
  for (let i = 0; i < Math.min(metricX.length, METRIC_NAMES.length); i += 1) {
    const [px] = dataToPixel(costAxis, metricX[i], costAxis.ylim[0]);
    const [px2] = dataToPixel(costAxis, metricX[i] + 1, costAxis.ylim[0]);
    const w = Math.abs(px2 - px);
    scene.texts.push(
      text(px, costAxisRect[1] + costAxisRect[3] + 8, METRIC_NAMES[i], {
        role: "note",
        size: 11,
        color: pal.muted,
        align: "center",
        maxWidth: Math.max(40, w * 2 - 8),
      }),
    );
  }

  const auxLines: string[] = []; // 曲线末端不写 label（会被渲染器画到面板外）；颜色由这几行说明
  auxLines.push(
    `显式 h=${formatNumber(hExplicit ?? 0, 4)}${hExplicitTheory === undefined ? "" : `（稳定天花板，理论 ${formatNumber(hExplicitTheory, 4)}）`}×${formatNumber(nStepsExplicit ?? 0, 5)} 步` +
      `；隐式 h=${formatNumber(hImplicit ?? 0, 4)}×${formatNumber(nStepsImplicit ?? 0, 4)} 步${hRatio === undefined ? "" : `（步长比 ${formatNumber(hRatio, 4)}×）`}`,
  );
  if (errExplicit !== undefined) auxLines.push(`端点误差：显式 ${formatNumber(errExplicit, 4)}（整段最大 ${formatNumber(errExplicitSpan ?? 0, 4)}）、隐式 ${formatNumber(errImplicit ?? 0, 4)}（整段最大 ${formatNumber(errImplicitSpan ?? 0, 4)}）`);
  if (tol !== undefined) {
    auxLines.push(
      `容差 tol=${formatNumber(tol, 3)}（帧 meta；临界 1e-06：更紧时隐式的精度受限步长会缩到显式的稳定受限步长同量级）` +
        (tolFallback ? "；⚠ 帧 meta.tolerance_fallback=True：容差比代价梯子最细一级还紧，隐式步长退到了梯底" : "") +
        (implicitErrAtPlanned === undefined ? "" : `；隐式在计划步长处的整段误差 ${formatNumber(implicitErrAtPlanned, 4)}`),
    );
  }
  if (crossFound && crossLogTol !== undefined) {
    const crossX = costCrossX?.[0];
    auxLines.push(
      `代价交叉点 log10(tol)=${formatNumber(crossLogTol, 4)}${crossX === undefined ? "" : `（x=${formatNumber(crossX, 4)}，y=${formatNumber(costCrossY?.[0] ?? 0, 4)}）`}：` +
        `容差更松时隐式便宜（求值 ${formatNumber(costImplicitAtTol ?? 0, 4)}，显式恒 ${formatNumber(costExplicitConst ?? 0, 4)}）；` +
        `更紧时显式的稳定性限制反而成了优势。`,
    );
  } else if (crossoverNote !== undefined) {
    auxLines.push(`帧 strings.crossover_note：${crossoverNote}`);
  }
  if (costX !== undefined && costImplicit !== undefined && costExplicit !== undefined) {
    auxLines.push(
      `代价—容差梯子（帧 cost_x/cost_implicit 各 9 点）：log10(tol) ${formatNumber(minOf(costX), 3)}…${formatNumber(maxOf(costX), 3)} 上隐式求值 ${formatNumber(minOf(costImplicit), 4)}…${formatNumber(maxOf(costImplicit), 4)}，显式恒 ${formatNumber(costExplicit[0], 4)}。`,
    );
  }
  if (startupSteps !== undefined && startupSteps > 0) auxLines.push(`多步法启动步 ${formatNumber(startupSteps, 4)} 步（污染精度，不改变稳定性）。`);
  if (roundoffFloor !== undefined) auxLines.push(`地板 ${formatNumber(roundoffFloor, 3)}=u/2：贴地板的样本可能是**精确 0**。`);
  scene.texts.push(...textBlock(auxRect[0] + 6, textTop(auxRect), textBottom(auxRect), auxLines, pal.muted, unitBudget(auxRect[2]), "top"));

  // —— 角标 ——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(placeBadge(badge("autoscale", "副轴：自适应 log（按本帧误差取整；会掩盖量级变化）", 8, 26), SLOT.auxAxis, width));
  const tolCrit = criticalOf(frame, "tol", 1e-6);
  if (tol !== undefined) {
    scene.badges.push(placeBadge(criticalBadge("tol", tol, tolCrit, { beyond: "below" }), SLOT.critical, width));
  }
  const rCrit = criticalOf(frame, "stiffness_ratio", 3183);
  if (ratio !== undefined) {
    scene.badges.push(placeBadge(criticalBadge("刚性比 r", ratio, rCrit, { beyond: "above" }), SLOT.critical2, width));
  }
  const status: string[] = [];
  if (nStepsExplicit !== undefined && nStepsImplicit !== undefined) {
    status.push(`步数 ${formatNumber(nStepsExplicit, 5)} vs ${formatNumber(nStepsImplicit, 4)}${stepsRatio === undefined ? "" : `（${formatNumber(stepsRatio, 4)}×）`}`);
  }
  if (markY !== undefined && markImpY !== undefined) {
    status.push(`当前误差 ${formatNumber(markY, 4)} vs ${formatNumber(markImpY, 4)}`);
  }
  if (note !== undefined) status.push(note);
  scene.badges.push(placeBadge(badge("status", clipBadgeText(status.join("；")), 8, 8), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);
  if (toleranceMark !== undefined && toleranceMark.length > 0) {
    // 帧给的 log10(tol) 与 meta.tol 必须自洽；不一致就降级说明（不静默取一个）。
    const implied = 10 ** toleranceMark[0];
    if (tol !== undefined && Math.abs(implied - tol) > Math.abs(tol) * 1e-9) {
      degraded.push(`帧 tolerance_mark=${formatNumber(toleranceMark[0], 4)}（=10^${formatNumber(implied, 3)}）与 meta.tol=${formatNumber(tol, 3)} 不一致：容差线按 meta.tol 画`);
    }
  }

  requireDrawableLayers(scene);
  const nBad =
    countNonFinite(tExplicit) +
    countNonFinite(yExplicit) +
    countNonFinite(tImplicit) +
    countNonFinite(yImplicit) +
    countNonFinite(error) +
    countNonFinite(errorImplicit) +
    countNonFinite(metricExplicit) +
    countNonFinite(metricImplicit);
  const extra: string[] = [];
  if (explClip.nClipped + implClip.nClipped > 0) extra.push(`解曲线有 ${explClip.nClipped + implClip.nClipped} 个样本超出固定纵轴 ±2，已截断（真实 max|y| 见 meta）`);
  const zeroBars = countZeros(metricExplicit) + countZeros(metricImplicit);
  if (zeroBars > 0) extra.push(`代价柱里有 ${zeroBars} 根的高度是 0（对数轴上画不出来）：显式方法没有 LU/Newton，这是结论不是缺数据`);
  if (truncFlag) extra.push(`帧 meta.truncated=True：|y| 超过显示上限 ${displayCeiling === undefined ? "?" : formatNumber(displayCeiling, 3)} 的部分在帧里就是截断值`);
  applyOverflowNote(scene, extra.join("；"), nBad, { slot: SLOT.overflow, width });
  return scene;
};

// --------------------------------------------------------------------------- //
// 局部工具
// --------------------------------------------------------------------------- //

function lastOf(a: ArrayLike<number>): number {
  return a.length > 0 ? a[a.length - 1] : 0;
}

function minOf(a: ArrayLike<number>): number {
  let best = Number.POSITIVE_INFINITY;
  for (let i = 0; i < a.length; i += 1) if (Number.isFinite(a[i]) && a[i] < best) best = a[i];
  return Number.isFinite(best) ? best : 0;
}

function maxOf(a: ArrayLike<number>): number {
  let best = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < a.length; i += 1) if (Number.isFinite(a[i]) && a[i] > best) best = a[i];
  return Number.isFinite(best) ? best : 0;
}

/** 柱子：只保留 >0 的项（对数轴上 0 在轴外），x 按组错开半个柱宽。 */
function positiveBars(x: ArrayLike<number>, v: ArrayLike<number>, offset: number): { x: number[]; v: number[] } {
  const xs: number[] = [];
  const vs: number[] = [];
  for (let i = 0; i < Math.min(x.length, v.length); i += 1) {
    if (!(v[i] > 0)) continue;
    xs.push(x[i] + offset);
    vs.push(v[i]);
  }
  return { x: xs, v: vs };
}

function countZeros(a: ArrayLike<number>): number {
  let n = 0;
  for (let i = 0; i < a.length; i += 1) if (a[i] === 0) n += 1;
  return n;
}

/** 若干数组里正的有限值范围（代价柱的对数轴用）；没有正值时 undefined。 */
function positiveRange(arrays: readonly ArrayLike<number>[]): [number, number] | undefined {
  let lo = Number.POSITIVE_INFINITY;
  let hi = 0;
  for (const a of arrays) {
    for (let i = 0; i < a.length; i += 1) {
      const v = a[i];
      if (!Number.isFinite(v) || v <= 0) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  return Number.isFinite(lo) && hi > 0 ? [lo, hi] : undefined;
}
