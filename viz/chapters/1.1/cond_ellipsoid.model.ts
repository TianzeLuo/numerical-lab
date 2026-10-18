/**
 * `cond_ellipsoid`（1.1 章）的 model：单位圆的像 + 右端项扰动把解推多远。
 *
 * 面板与图层（顺序即绘制顺序，测试按此断言）
 * ------------------------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | line | `geometry` | 单位圆（灰虚线参照：它必须看起来是圆的） |
 * | 2 | line | `geometry` | `A` 的像（椭圆；半轴比就是 κ₂，短半轴 = σmin） |
 * | 3 | line | `geometry` | `b`（右端项）、`δb`（扰动，这个尺度下不可见） |
 * | 4 | line | `geometry` | `x`（精确解）、`x̂`（数值解）、`x̂′`（扰动后的解），冲出 ±3 的按方向裁到轴边 |
 * | 5 | line | `error` | 实测放大比（随 log10 κ 的曲线，实线） |
 * | 6 | line | `error` | **理论上界 κ**（虚线；对数纵轴上它是斜率 +1 的直线） |
 * | 7 | scatter | `error` | 当前 log10 κ 处的实测放大比 |
 *
 * 角标槽位（固定，见 `SLOT`）：0 来源 / 1 主轴固定 ±3 / 2 副轴自适应 / 3 临界（δb 的
 * u = 1.1e-16）/ 4 状态（κu、note、前向/后向误差）/ 5 前端预览 / 6 溢出 / 7 降级说明。
 *
 * 几条刻意的决定
 * --------------
 * 1. **主轴固定 ±3 且面板是正方形**：单位圆必须看起来是圆的，否则"圆被压成椭圆"这件事
 *    本身就失真（与 1.2 章 `unit_ball` 同一条理由）。代价是 κ 大时椭圆与解向量会冲出画面
 *    ——**这是要看到的**：模型按方向把向量裁到轴边、把真实端点写进数值行，并在溢出角标里
 *    说明"看不到不等于不存在"。反过来若把轴改成自适应，κ=1e16 的细椭圆会被拉得和单位圆
 *    一样"圆"，条件数这种纯几何的性质在视觉上直接消失（manifest pitfalls 3a）。
 * 2. **规则 2 的"理论参考线"在这里就是 `error_bound`**：这条副轴的横轴是 `log10 κ`
 *    （线性），纵轴是对数；理论上界 `κ = 10^{log10 κ}` 在这张图上是一条**斜率 +1 的直线**，
 *    帧已经把它作为数组 `error_bound` 给出，模型只需照画，**不重算**任何物理量
 *    （P8：κ 由后端给）。实测曲线一般远低于它——那不是界松，是扰动方向不在最坏方向。
 * 3. **4 元"两段折线"是 `[x0, x1, y0, y1]`**：`b_vec`/`δb`/`x_*` 的排布前半段给 x、后半段给 y
 *    （后端就是 `np.array([0.0, v[0], 0.0, v[1]])` 拼的，`unit_ball` 的文件头第 2 条有同样的
 *    说明），所以 `x = [v[0], v[1]]`、`y = [v[2], v[3]]` 正是"从原点指向 (vx, vy)"的那一段。
 *    按交错 (x,y) 读会得到一条贴在 y 轴上的竖线，与 `‖b‖ = 1` 直接矛盾。
 * 4. **非有限的解向量只降级、不抛错**：κ ≳ 1/u 时后端可能给 `NaN`（`exact_solve` 退化），
 *    那是"这一帧没有有限解"的**如实标注**，不是缺数据。模型跳过该箭头、在状态里点名，
 *    其余图层照画——把整张图炸掉反而掩盖了"这里已经不可解"这个结论。
 */

import { formatNumber, niceLimits } from "../../shared/src/axes.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import type { Frame } from "../../shared/src/frame.js";
import type { LineSeries } from "../../shared/src/scene.js";
import {
  ModelInputError,
  applyOverflow,
  canvasSize,
  criticalOf,
  dataSourceOf,
  degradedBadge,
  finiteMinMax,
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
  criticalBadge,
  lineLayer,
  makeAxis,
  overflowBadge,
  previewBadge,
  scatterLayer,
  series,
  sourceBadge,
} from "../../shared/src/scene.js";
import {
  avoidTextOverlap,
  axisTitleOf,
  axisYlabelOf,
  textBlock,
  textBottom,
  textTop,
  unitBudget,
} from "./_panel.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "1.1";
const VIZ = "cond_ellipsoid";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "条件数椭球：单位圆的像与解 x 的跳变";

/** 固定角标槽位（见 `model.ts::BADGE_LAYOUT`）。 */
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

/** manifest 声明主轴固定 ±3（`ELLIPSE_HALF_WIDTH`）；单位圆必须看起来是圆的。 */
const MAIN_LIM: [number, number] = [-3, 3];
/** `frame.critical.delta_b` 缺失时的回落值（与 manifest 的 `critical[0]` = u 同源）。 */
const DELTA_B_CRITICAL = 1.1e-16;

/** 一条向量（帧里的 4 元数组解出来的"从原点出发"的那一段）。 */
interface Arrow {
  label: string;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export const condEllipsoidModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需数组 ——
  const circleX = requireArray(frame, "circle_x");
  const circleY = requireArray(frame, "circle_y");
  const imageX = requireArray(frame, "image_x");
  const imageY = requireArray(frame, "image_y");
  const bVec = requireArray(frame, "b_vec");
  const dbVec = requireArray(frame, "db_vec");
  const xExact = requireArray(frame, "x_exact");
  const xNum = requireArray(frame, "x_num");
  const xPert = requireArray(frame, "x_pert");
  const kappaGrid = requireArray(frame, "kappa_grid");
  const error = requireArray(frame, "error");
  const errorBound = requireArray(frame, "error_bound");
  requireSameLength(circleX, circleY, "帧数组 circle_x 与 circle_y");
  requireSameLength(imageX, imageY, "帧数组 image_x 与 image_y");
  requireSameLength(kappaGrid, error, "帧数组 kappa_grid 与 error");
  requireSameLength(kappaGrid, errorBound, "帧数组 kappa_grid 与 error_bound");
  for (const [name, arr] of [
    ["b_vec", bVec],
    ["db_vec", dbVec],
    ["x_exact", xExact],
    ["x_num", xNum],
    ["x_pert", xPert],
  ] as const) {
    if (arr.length !== 4) {
      throw ModelInputError4(name, arr.length);
    }
  }
  const pitfalls = requirePitfalls(frame);

  // —— 可选数组 ——
  const degraded: string[] = [];
  const errorMark = optionalFirst(frame, "error_mark");
  const kappaMark = optionalFirst(frame, "kappa_mark");
  if (errorMark === undefined || kappaMark === undefined) degraded.push("帧缺 error_mark/kappa_mark：副轴不画当前位置标记");

  // —— meta ——
  const tMeta = metaNumber(frame, "t");
  if (tMeta === undefined) degraded.push("帧 meta 缺 t：scene.t 改用请求 t（两者可能不同）");
  const shear = metaNumber(frame, "shear");
  const log10Kappa = metaNumber(frame, "log10_kappa");
  const kappaTarget = metaNumber(frame, "kappa_target");
  const kappaMeasured = metaNumber(frame, "kappa_measured");
  const kappa1 = metaNumber(frame, "kappa_1");
  const kappa2 = metaNumber(frame, "kappa_2");
  const kappaInf = metaNumber(frame, "kappa_inf");
  const kappaFro = metaNumber(frame, "kappa_fro");
  const sigmaMin = metaNumber(frame, "sigma_min");
  const sigmaMax = metaNumber(frame, "sigma_max");
  const detA = metaNumber(frame, "det_a");
  const deltaB = metaNumber(frame, "delta_b");
  const bAngle = metaNumber(frame, "b_angle");
  const amplification = metaNumber(frame, "amplification_measured");
  const ampOverKappa = metaNumber(frame, "amplification_over_kappa");
  const forwardError = metaNumber(frame, "forward_error");
  const backwardError = metaNumber(frame, "backward_error");
  const residual = metaNumber(frame, "residual");
  const boundRhs = metaNumber(frame, "bound_rhs");
  const kappaU = metaNumber(frame, "kappa_u");
  const nTheta = metaNumber(frame, "n_theta");
  const frameOverflow = (metaNumber(frame, "overflow") ?? 0) > 0.5;
  const family = stringOf(frame, "family");
  const normLabel = stringOf(frame, "norm");
  if (family === undefined) degraded.push("帧 strings 缺 family：不显示矩阵族名（不按参数猜）");
  if (normLabel === undefined) degraded.push("帧 strings 缺 norm：不显示范数名（κ 必须连范数一起读）");
  const note = stringOf(frame, "note");
  const floorNote = stringOf(frame, "floor_note");
  if (note === undefined) degraded.push("帧 strings 缺 note：状态角标不自行编造结论");
  if (floorNote === undefined) degraded.push("帧 strings 缺 floor_note：副轴的地板读法不自行编造（点数仍按 meta.error_floor 统计）");
  if (frame.critical["delta_b"] === undefined) {
    degraded.push(`帧 critical 缺 delta_b：临界值回落到 manifest 声明的 ${DELTA_B_CRITICAL}（不是帧给的）`);
  }
  const normNote = stringOf(frame, "norm_note");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: tMeta ?? opts.t,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  // —— 排版：左侧正方形画几何，右侧一整列画误差曲线（与 1.2 章 unit_ball 同一口径）——
  const base = stackedPanels(width, height);
  // 两种排布（都在 stackedPanels 给的同一个 [top, bottom] 里，规则 8 的装饰带仍然由 textBlock 让开）：
  //  · 宽画布：几何方块靠左，误差面板靠右（方块保持正方形，±3 的两个方向等比例，不看走眼）；
  //  · 窄画布（内容宽 < 620px）：肩并肩会把误差面板顶出画布，改成上下排。
  const narrow = base.contentWidth < 620;
  const side = narrow
    ? Math.max(120, Math.floor(Math.min(base.contentWidth * 0.62, (base.bottom - base.top) * 0.45)))
    : Math.max(110, Math.floor(Math.min(base.bottom - base.top, base.contentWidth * 0.5)));
  const mainRect: Rect = [base.left, base.top, side, side];
  const gap = narrow ? 0 : 74;
  const auxX = base.left + side + gap;
  const auxTop = narrow ? mainRect[1] + side + 30 : base.top;
  const auxRect: Rect = [
    auxX,
    auxTop,
    Math.max(120, base.left + base.contentWidth - auxX),
    base.bottom - auxTop,
  ];

  const mainAxis = makeAxis("geometry", mainRect, {
    xlim: MAIN_LIM,
    ylim: MAIN_LIM,
    xlabel: "x",
    ylabel: axisYlabelOf(SPEC?.axes, "geometry", "y"),
    title: axisTitleOf(SPEC?.axes, "geometry", "单位圆与它的像（固定 ±3）"),
    autoscaled: false,
  });

  // —— 副轴：实测放大比与理论上界共用的 log 纵轴（manifest: autoscaled=true）——
  const bErr = finitePositiveMinMax(error);
  const bBound = finitePositiveMinMax(errorBound);
  const markPos = errorMark === undefined ? undefined : errorMark > 0 ? [errorMark, errorMark] : undefined;
  const lo = Math.min(bErr?.[0] ?? Number.POSITIVE_INFINITY, bBound?.[0] ?? Number.POSITIVE_INFINITY, markPos?.[0] ?? Number.POSITIVE_INFINITY);
  const hi = Math.max(bErr?.[1] ?? 0, bBound?.[1] ?? 0, markPos?.[1] ?? 0);
  if (!(Number.isFinite(lo) && Number.isFinite(hi))) {
    degraded.push("error / error_bound 里没有 > 0 的有限值：副轴用占位范围 [1e-18, 1e18]，不画曲线");
  }
  const auxYlim: [number, number] =
    Number.isFinite(lo) && Number.isFinite(hi) ? niceLimits(lo, hi, 6, { log: true }) : [1e-18, 1e18];
  const kRange = finiteMinMax(kappaGrid) ?? [0, 16];
  const auxAxis = makeAxis("error", auxRect, {
    xlim: niceLimits(kRange[0], kRange[1], 6),
    ylim: auxYlim,
    xscale: "linear",
    yscale: "log",
    xlabel: "log10 κ（目标）",
    ylabel: axisYlabelOf(SPEC?.axes, "error", "相对误差放大比"),
    title: axisTitleOf(SPEC?.axes, "error", "实测放大比与理论上界 κ（对数）"),
    autoscaled: true,
    tickTarget: 5,
  });
  scene.axes.push(mainAxis, auxAxis);

  // —— 图层 1：单位圆（参照：它必须看起来是圆的）——
  const circle = trimNonFinite(circleX, circleY);
  if (circle !== undefined) {
    scene.layers.push(lineLayer("geometry", series(circle.x, circle.y, pal.reference, { width: 1.1, dash: [5, 4], label: "单位圆 S¹（参照）" })));
  } else {
    degraded.push("circle_x/circle_y 没有可绘的有限点：参照圆画不出来（椭圆的「扁」就没有对照了）");
  }

  // —— 图层 2：A 的像 ——
  const image = trimNonFinite(imageX, imageY);
  if (image !== undefined) {
    scene.layers.push(lineLayer("geometry", series(image.x, image.y, pal.curves[0] ?? "#2b6cb0", { width: 2, label: "A 的像：半轴比 = κ₂（短半轴 = σmin）" })));
  } else {
    degraded.push("image_x/image_y 没有可绘的有限点：A 的像画不出来");
  }

  // —— 图层 3~4：五条向量（4 元 [x0,x1,y0,y1]；非有限的跳过并点名）——
  const clipped: string[] = [];
  const skipped: string[] = [];
  const arrows: { raw: ArrayLike<number>; label: string; color: string; dash?: number[]; width: number }[] = [
    { raw: bVec, label: "b（右端项，‖b‖=1）", color: pal.note, dash: [7, 4], width: 2 },
    { raw: dbVec, label: "δb（扰动；这个尺度下与原点重合）", color: pal.curves[1] ?? "#2f855a", width: 2 },
    { raw: xExact, label: "x（精确解）", color: pal.ink, dash: [3, 3], width: 1.6 },
    { raw: xNum, label: "x̂（数值解）", color: pal.marker, width: 2 },
    { raw: xPert, label: "x̂′（扰动后的解）", color: pal.residual, dash: [6, 3], width: 1.6 },
  ];
  for (const item of arrows) {
    const arrow = toArrow(item.label, item.raw);
    if (arrow === undefined) {
      skipped.push(item.label);
      continue;
    }
    const drawn = clipArrow(arrow, MAIN_LIM);
    if (drawn.clipped) {
      clipped.push(`${arrow.label.split("（")[0]} 的真实端点 (${formatNumber(arrow.x1, 3)}, ${formatNumber(arrow.y1, 3)})`);
    }
    scene.layers.push(lineLayer("geometry", arrowSeries({ ...arrow, x1: drawn.x1, y1: drawn.y1 }, item.color, item.dash, item.width)));
  }
  if (skipped.length > 0) degraded.push(`帧里 ${skipped.join("、")} 含非有限分量（该 κ 上后端没有有限解）：这几条箭头不画，其余照画`);

  // —— 图层 5~7：副轴 ——
  const errSeries = segmentedSeries(kappaGrid, error, pal.residual, {
    width: 1.9,
    label: "实测放大比 (‖Δx‖/‖x‖)/(‖δb‖/‖b‖)：沿 log10 κ 逐点真跑一次",
    accept: (v) => v > 0,
  });
  if (errSeries.length > 0) scene.layers.push(lineLayer("error", errSeries));
  else degraded.push("error 的样本没有一个 > 0：log 轴上画不出实测曲线");
  const boundSeries = segmentedSeries(kappaGrid, errorBound, pal.curves[2] ?? "#b7791f", {
    width: 1.5,
    dash: [7, 4],
    label: "理论上界 κ（帧数组 error_bound）：对数纵轴上斜率 +1",
    accept: (v) => v > 0,
  });
  if (boundSeries.length > 0) scene.layers.push(lineLayer("error", boundSeries));
  else degraded.push("error_bound 的样本没有一个 > 0：log 轴上画不出理论上界");
  if (errorMark !== undefined && kappaMark !== undefined && errorMark > 0) {
    scene.layers.push(scatterLayer("error", [kappaMark], [errorMark], { r: 4.5, color: pal.marker, label: "当前 log10 κ 处的实测放大比" }));
  }

  // —— 文本 ——
  const summary = paramSummary(frame, opts, [
    { name: "shear", label: "剪切 s" },
    { name: "delta_b", label: "δb" },
    { name: "b_angle", label: "b 方向(度)" },
  ]);
  degraded.push(...summary.degraded);

  const head: string[] = [summary.text];
  if (family !== undefined) head.push(`族=${family}`);
  if (shear !== undefined) head.push(`剪切 s=${formatNumber(shear, 4)}`);
  if (normLabel !== undefined) head.push(`范数=${normLabel}`);
  if (log10Kappa !== undefined) head.push(`log10 κ=${formatNumber(log10Kappa, 4)}`);
  const mainBudget = unitBudget(mainRect[2]);
  scene.texts.push(...textBlock(mainRect[0] + 6, textTop(mainRect), textBottom(mainRect), [head.join("　")], pal.muted, mainBudget, "top"));

  const legend: string[] = [
    "灰虚=单位圆　蓝=像（半轴比 = κ₂）　蓝虚=b　黑虚=x　红=x̂　紫虚=x̂′",
  ];
  if (clipped.length > 0) legend.push(`⚠ ${clipped.length} 条向量冲出 ±3：${clipped.join("；")}——看不到不等于不存在`);
  legend.push("δb 在 ±3 的尺度下与原点重合（‖δb‖ ~ 1e-12）：它造成的位移要看 x 与 x̂′ 的夹角，不看箭头长度");
  scene.texts.push(...textBlock(mainRect[0] + 6, textTop(mainRect), textBottom(mainRect), legend, clipped.length > 0 ? pal.warn : pal.muted, mainBudget, "bottom"));

  const auxLines: string[] = [];
  const nums1: string[] = [];
  if (kappaTarget !== undefined) nums1.push(`目标 κ=${formatNumber(kappaTarget, 4)}`);
  if (kappaMeasured !== undefined) nums1.push(`实测 κ_${normLabel ?? "?"}=${formatNumber(kappaMeasured, 4)}`);
  if (nums1.length > 0) auxLines.push(`${nums1.join("　")}（target ≠ measured：族不同差得不同，两个都必须给）`);
  const nums2: string[] = [];
  if (kappa1 !== undefined) nums2.push(`κ₁=${formatNumber(kappa1, 3)}`);
  if (kappa2 !== undefined) nums2.push(`κ₂=${formatNumber(kappa2, 3)}`);
  if (kappaInf !== undefined) nums2.push(`κ∞=${formatNumber(kappaInf, 3)}`);
  if (kappaFro !== undefined) nums2.push(`κ_F=${formatNumber(kappaFro, 3)}`);
  if (nums2.length > 0) auxLines.push(`${nums2.join("　")}（不同范数可差一个 n 因子）`);
  const nums3: string[] = [];
  if (sigmaMin !== undefined && sigmaMax !== undefined) nums3.push(`σ ∈ [${formatNumber(sigmaMin, 3)}, ${formatNumber(sigmaMax, 3)}]`);
  if (detA !== undefined) nums3.push(`det A=${formatNumber(detA, 3)}`);
  if (nTheta !== undefined) nums3.push(`n_θ=${formatNumber(nTheta, 4)}`);
  if (nums3.length > 0) auxLines.push(nums3.join("　"));
  const nums4: string[] = [];
  if (amplification !== undefined) nums4.push(`实测放大比=${formatNumber(amplification, 3)}`);
  if (ampOverKappa !== undefined) nums4.push(`实测/κ=${formatNumber(ampOverKappa, 3)}`);
  if (forwardError !== undefined) nums4.push(`前向误差=${formatNumber(forwardError, 3)}`);
  if (backwardError !== undefined) nums4.push(`后向误差=${formatNumber(backwardError, 3)}`);
  if (residual !== undefined) nums4.push(`残差=${formatNumber(residual, 3)}`);
  if (nums4.length > 0) auxLines.push(nums4.join("　"));
  const nums5: string[] = [];
  if (deltaB !== undefined) nums5.push(`δb=${formatNumber(deltaB, 3)}`);
  if (bAngle !== undefined) nums5.push(`b 方向=${formatNumber(bAngle, 4)}°`);
  if (boundRhs !== undefined) nums5.push(`κ·δb=${formatNumber(boundRhs, 3)}`);
  if (kappaU !== undefined) nums5.push(`κu=${formatNumber(kappaU, 3)}`);
  if (nums5.length > 0) auxLines.push(nums5.join("　"));
  if (normNote !== undefined) auxLines.push(normNote);
  if (floorNote !== undefined) auxLines.push(floorNote);
  auxLines.push("实测曲线一般**低于**理论上界：界是对**所有**方向的保证，随机方向不在最坏方向上——这不是界太松");
  if (note !== undefined) auxLines.push(`帧 note：${note}`);
  const auxBudget = unitBudget(auxRect[2]);
  scene.texts.push(...textBlock(auxRect[0] + 6, textTop(auxRect), textBottom(auxRect), auxLines, pal.muted, auxBudget, "top"));

  // —— 角标（固定槽位）——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  if (scene.dataSource.startsWith("backend+preview")) {
    scene.badges.push(placeBadge(previewBadge("拖动时按显示空间插值"), SLOT.preview, width));
  }
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(
      badge("autoscale", "误差轴：自适应（log 纵轴按本帧 error/error_bound 取整；自适应会掩盖量级变化）", 8, 8),
      SLOT.auxAxis,
      width,
    ),
  );
  if (deltaB !== undefined) {
    scene.badges.push(
      placeBadge(criticalBadge("delta_b", deltaB, criticalOf(frame, "delta_b", DELTA_B_CRITICAL), { beyond: "below" }), SLOT.critical, width),
    );
  }
  const status: string[] = [];
  if (kappaU !== undefined && kappaU >= 1) {
    status.push(`⚠ κu = ${formatNumber(kappaU, 3)} ≥ 1：A 在 float64 里已与奇异矩阵无法区分，椭圆与曲线都只是舍入的产物`);
  } else if (kappaU !== undefined) {
    status.push(`κu = ${formatNumber(kappaU, 3)} < 1`);
  }
  if (clipped.length > 0) status.push(`${clipped.length} 条向量已按方向裁到 ±3 轴边`);
  if (note !== undefined) status.push(note);
  if (status.length === 0) status.push("帧未给出可判读的状态（预期有 meta.kappa_u 与 strings.note）");
  scene.badges.push(placeBadge(badge("status", status.join("；"), 8, 8), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  // 规则 1：图层越界 + 帧内非有限值 + "帧说像越界了" + "模型发现向量越界"，四者都要进 overflow。
  const nBad =
    countNonFinite(circleX) +
    countNonFinite(circleY) +
    countNonFinite(imageX) +
    countNonFinite(imageY) +
    countNonFinite(bVec) +
    countNonFinite(dbVec) +
    countNonFinite(xExact) +
    countNonFinite(xNum) +
    countNonFinite(xPert) +
    countNonFinite(kappaGrid) +
    countNonFinite(error) +
    countNonFinite(errorBound);
  applyOverflow(scene, nBad, { slot: SLOT.overflow, width });
  const extra: string[] = [];
  if (frameOverflow) extra.push("帧 meta.overflow=true：单位圆的像已经跑出固定的 ±3 坐标范围（看不到不等于不存在）");
  if (clipped.length > 0) extra.push(`${clipped.join("；")}——按方向裁到轴边显示`);
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

/** 4 元向量的长度契约错误（向量按 [x0, x1, y0, y1] 编码，见文件头第 3 条）。 */
function ModelInputError4(name: string, len: number): ModelInputError {
  return new ModelInputError(
    `帧 ${CHAPTER}/${VIZ} 的 '${name}' 长度 ${len} ≠ 4：向量按 [x0, x1, y0, y1] 编码（见文件头第 3 条）`,
  );
}

/** 4 元向量 `[x0, x1, y0, y1]` → 从原点出发的箭头；含非有限分量时返回 undefined（降级）。 */
function toArrow(label: string, v: ArrayLike<number>): Arrow | undefined {
  const x0 = v[0] as number;
  const x1 = v[1] as number;
  const y0 = v[2] as number;
  const y1 = v[3] as number;
  if (![x0, y0, x1, y1].every((t) => Number.isFinite(t))) return undefined;
  return { label, x0, y0, x1, y1 };
}

/** 箭头裁到固定坐标框里：原点在框内，取与框边最近的那个正向交点（纯显示几何）。 */
function clipArrow(a: Arrow, lim: [number, number]): { x1: number; y1: number; clipped: boolean } {
  const dx = a.x1 - a.x0;
  const dy = a.y1 - a.y0;
  let t = 1;
  const consider = (num: number, den: number): void => {
    if (den === 0 || !Number.isFinite(den)) return;
    const s = num / den;
    if (s > 0 && s < t) t = s;
  };
  if (dx > 0) consider(lim[1] - a.x0, dx);
  if (dx < 0) consider(lim[0] - a.x0, dx);
  if (dy > 0) consider(lim[1] - a.y0, dy);
  if (dy < 0) consider(lim[0] - a.y0, dy);
  const clipped = t < 1 - 1e-12;
  // 交点上再向内收一丝（相对量 1e-9），免得贴着轴框的坐标被判成"越界"。
  const shrink = clipped ? 1 - 1e-9 : 1;
  return { x1: a.x0 + dx * t * shrink, y1: a.y0 + dy * t * shrink, clipped };
}

/** 一条箭头 = 杆 + 两支箭头 + 端点圆点（与 1.2 章 `unit_ball` 同一画法）。 */
function arrowSeries(a: Arrow, color: string, dash: number[] | undefined, width: number): LineSeries[] {
  const dx = a.x1 - a.x0;
  const dy = a.y1 - a.y0;
  const len = Math.hypot(dx, dy);
  const out: LineSeries[] = [series([a.x0, a.x1], [a.y0, a.y1], color, { width, dash, label: a.label })];
  if (len > 0) {
    const span = MAIN_LIM[1] - MAIN_LIM[0];
    const head = Math.min(len * 0.35, span * 0.045);
    const ux = dx / len;
    const uy = dy / len;
    const bx = a.x1 - head * ux;
    const by = a.y1 - head * uy;
    const px = -uy * head * 0.45;
    const py = ux * head * 0.45;
    out.push(
      series([a.x1, bx + px], [a.y1, by + py], color, { width }),
      series([a.x1, bx - px], [a.y1, by - py], color, { width }),
      series([a.x1], [a.y1], color, { width: width + 0.4, marker: "o" }),
    );
  }
  return out;
}

/** 去掉非有限点；一个有限点都没有时 undefined（调用方记一条降级）。 */
function trimNonFinite(x: ArrayLike<number>, y: ArrayLike<number>): { x: Float64Array; y: Float64Array } | undefined {
  const xs: number[] = [];
  const ys: number[] = [];
  const n = Math.min(x.length, y.length);
  for (let i = 0; i < n; i += 1) {
    const xv = x[i] as number;
    const yv = y[i] as number;
    if (Number.isFinite(xv) && Number.isFinite(yv)) {
      xs.push(xv);
      ys.push(yv);
    }
  }
  if (xs.length === 0) return undefined;
  return { x: Float64Array.from(xs), y: Float64Array.from(ys) };
}

/** 取"长度 1 的数组"的第 0 个元素；缺数组、空数组、非有限值一律 undefined（降级，不猜）。 */
function optionalFirst(frame: Frame, name: string): number | undefined {
  if (!frame.has(name)) return undefined;
  const a = requireArray(frame, name);
  if (a.length === 0) return undefined;
  const v = a[0] as number;
  return Number.isFinite(v) ? v : undefined;
}
