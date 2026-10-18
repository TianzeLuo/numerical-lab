/**
 * `stiff_dilemma`（3.2 章）的 model：解那么平缓，步长却被 λ_max 掐住。
 *
 * 面板与图层（顺序即绘制顺序，测试按此断言）
 * ------------------------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | line | `main` | 精确解 cos t（橙色虚线；菜单里它是"平缓"的标尺） |
 * | 2 | line | `main` | 显式解（蓝） |
 * | 3 | line | `main` | 隐式解（绿） |
 * | 4 | scatter | `main` | 显式解的**每一步落点**（短竖线地毯，3142 个） |
 * | 5 | scatter | `main` | 隐式解的每一步落点（地毯，101 个） |
 * | 6 | line | `aux` | 显式方法误差（对数） |
 * | 7 | line | `aux` | 隐式方法误差（对数，虚线） |
 * | 8 | line | `aux` | 舍入地板 u/2（参考线） |
 * | 9 | scatter | `aux` | 当前时刻的误差标记（帧 `error_mark`） |
 *
 * 时间轴语义：**真实时间 t**（manifest `time.mode="scrub"`，`tMax=2π`）。帧里的解数组
 * 已经截到 `meta.t_now`；显示一律用 `t_now`，越界时夹住并写进文本，不假装 t 就是索引。
 *
 * 三条刻意的决定
 * --------------
 * 1. **"步长"用地毯画，不把它塞进误差轴**。帧的 `step_values`（0.002 与 0.0628）与误差
 *    （1e-16…2e-6）差 4 个数量级，硬画在同一条对数轴上会把误差曲线压成一条直线；
 *    而"显式 3142 步 / 隐式 101 步"这件事，**地毯的疏密**已经说完了。数值写在文本里。
 * 2. **误差是"到目前为止"的**：`error`/`error_implicit` 都只到 `t_now`，所以拖动时间轴时
 *    两条误差曲线一起变长——这不是动画，是切片。面板标题写清"截到 t_now"。
 * 3. **地板要标出来**：后端把 0 与非有限误差统一抬到 u/2（本帧 1.11e-16），贴地板的样本
 *    在图上与"误差恰好等于 1.11e-16"完全一样。所以贴地板的点会叠一枚空心标记（本模型用
 *    文本计数 + 参考线），"看不见"与"不存在"必须分开。
 */

import { formatNumber } from "../../shared/src/axes.js";
import { countNonFinite } from "../../shared/src/geometry.js";
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
  stackedPanels,
  stringOf,
  type SceneModel,
} from "../../shared/src/model.js";
import {
  axisModeBadge,
  badge,
  criticalBadge,
  lineLayer,
  makeAxis,
  scatterLayer,
  series,
  sourceBadge,
} from "../../shared/src/scene.js";
import {
  applyOverflowNote,
  axisTitleOf,
  axisYlabelOf,
  clipRuns,
  countAtCeiling,
  logRange,
  textBlock,
  textBottom,
  textTop,
  unitBudget,
} from "./_panel.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "3.2";
const VIZ = "stiff_dilemma";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "刚性困境：解那么平缓，步长却被 λ_max 掐住";

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

/** manifest 声明主轴"固定纵轴 ±2"：越界要标出来，不许偷偷把轴拉长。 */
const MAIN_LIM: [number, number] = [-2, 2];
/** 两排步点地毯的高度（在 ±2 里贴近下沿，不与曲线打架）。 */
const RUG_EXPLICIT = -1.90;
const RUG_IMPLICIT = -1.78;

export const stiffDilemmaModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  const tExact = requireArray(frame, "t_exact");
  const yExact = requireArray(frame, "y_exact");
  const tExplicit = requireArray(frame, "t_explicit");
  const yExplicit = requireArray(frame, "y_explicit");
  const tImplicit = requireArray(frame, "t_implicit");
  const yImplicit = requireArray(frame, "y_implicit");
  const error = requireArray(frame, "error");
  const errorMark = requireArray(frame, "error_mark");
  const errorImplicit = requireArray(frame, "error_implicit");
  const stepMarksExplicit = requireArray(frame, "step_marks_explicit");
  const stepMarksImplicit = requireArray(frame, "step_marks_implicit");
  const curvature = frame.has("curvature") ? requireArray(frame, "curvature") : undefined;

  requireSameLength(tExact, yExact, "帧数组 t_exact 与 y_exact");
  requireSameLength(tExplicit, yExplicit, "帧数组 t_explicit 与 y_explicit");
  requireSameLength(tImplicit, yImplicit, "帧数组 t_implicit 与 y_implicit");
  requireSameLength(tExplicit, error, "帧数组 t_explicit 与 error");
  requireSameLength(tImplicit, errorImplicit, "帧数组 t_implicit 与 error_implicit");
  const pitfalls = requirePitfalls(frame);

  const degraded: string[] = [];
  const tMeta = metaNumber(frame, "t");
  if (tMeta === undefined) degraded.push("帧 meta 缺 t：scene.t 用请求值代替（两者可能不同）");
  const tNow = tMeta ?? opts.t;
  const tEnd = metaNumber(frame, "t_end");
  if (tEnd === undefined) degraded.push("帧 meta 缺 t_end：横轴右端改用帧里最大的时间（不按 tMax 猜）");
  const lam = metaNumber(frame, "lam");
  const ratio = metaNumber(frame, "stiffness_ratio");
  const hCeiling = metaNumber(frame, "h_ceiling");
  const hCeilingTheory = metaNumber(frame, "h_ceiling_theory");
  const hImplicit = metaNumber(frame, "h_implicit");
  const stepsRatio = metaNumber(frame, "steps_ratio");
  const nStepsExplicit = metaNumber(frame, "n_steps_explicit");
  const nStepsExplicitPlanned = metaNumber(frame, "n_steps_explicit_planned");
  const nStepsImplicit = metaNumber(frame, "n_steps_implicit");
  const fEvalsExplicit = metaNumber(frame, "f_evals_explicit");
  const fEvalsImplicit = metaNumber(frame, "f_evals_implicit");
  const luImplicit = metaNumber(frame, "lu_implicit");
  const newtonImplicit = metaNumber(frame, "newton_implicit");
  const errFinal = metaNumber(frame, "error_explicit_final");
  const errImplFinal = metaNumber(frame, "error_implicit_final");
  const errMax = metaNumber(frame, "error_explicit_max");
  const maxAbsY = metaNumber(frame, "max_abs_y_explicit");
  const truncFlag = (metaNumber(frame, "truncated") ?? 0) > 0.5;
  const displayCeiling = metaNumber(frame, "display_ceiling");
  const curvatureMax = metaNumber(frame, "curvature_max");
  const roundoffFloor = metaNumber(frame, "roundoff_floor");
  const note = stringOf(frame, "note");
  if (note === undefined) degraded.push("帧 strings 缺 note：状态角标不自行编造结论");
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

  const tRight = Math.max(tEnd ?? Math.max(lastOf(tExplicit), lastOf(tImplicit)), Math.max(lastOf(tExplicit), lastOf(tImplicit)));
  const xlim: [number, number] = tRight > 0 ? [0, tRight] : [0, 1];
  // 面板间距 30（共享实现缺省 22）：上方面板的轴名装饰带要到 bottom+28，
  // 下方那一块顶部第一行文字在 top+4——间距不够就会被校验器的"规则 8"当场拦下。
  const base = stackedPanels(width, height, { gap: 30, mainShare: 0.6 });
  const mainRect = base.main;
  const auxRect = base.error;
  scene.axes.push(
    makeAxis("main", mainRect, {
      xlim,
      ylim: MAIN_LIM,
      xlabel: "t（一个周期 2π；曲线只画到当前 t）",
      ylabel: axisYlabelOf(SPEC?.axes, "main", "y(t)"),
      title: axisTitleOf(SPEC?.axes, "main", "解曲线（固定纵轴 ±2）与两种方法的步点"),
      autoscaled: false,
    }),
  );
  const errPos = finitePositiveMinMax(error);
  const errImplPos = finitePositiveMinMax(errorImplicit);
  const yLoRaw = Math.min(errPos?.[0] ?? Number.POSITIVE_INFINITY, errImplPos?.[0] ?? Number.POSITIVE_INFINITY);
  const yHiRaw = Math.max(errPos?.[1] ?? 0, errImplPos?.[1] ?? 0, roundoffFloor ?? 0);
  const auxYlim: [number, number] =
    Number.isFinite(yLoRaw) && Number.isFinite(yHiRaw) && yHiRaw > 0
      ? logRange(yLoRaw, yHiRaw)
      : [1e-18, 1];
  if (!(Number.isFinite(yLoRaw) && yHiRaw > 0)) degraded.push("error/error_implicit 里没有 >0 的有限值：副轴用占位范围，不画误差曲线");
  scene.axes.push(
    makeAxis("aux", auxRect, {
      xlim,
      ylim: auxYlim,
      yscale: "log",
      xlabel: "t（与上图同一条时间轴）",
      ylabel: axisYlabelOf(SPEC?.axes, "aux", "误差 / 步长"),
      title: axisTitleOf(SPEC?.axes, "aux", "两种方法的误差（对数，截到当前 t）"),
      autoscaled: true,
    }),
  );
  const auxAxis = scene.axes[1];

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
  if (stepMarksExplicit.length > 0) {
    const yRug = new Float64Array(stepMarksExplicit.length).fill(RUG_EXPLICIT);
    scene.layers.push(
      scatterLayer("main", stepMarksExplicit, yRug, {
        r: 0.7,
        color: pal.curves[0],
        alpha: 0.55,
        label: `显式方法的每一步落点（${stepMarksExplicit.length} 步）`,
      }),
    );
  }
  if (stepMarksImplicit.length > 0) {
    const yRug = new Float64Array(stepMarksImplicit.length).fill(RUG_IMPLICIT);
    scene.layers.push(
      scatterLayer("main", stepMarksImplicit, yRug, {
        r: 1.7,
        color: pal.curves[1],
        alpha: 0.9,
        label: `隐式方法的每一步落点（${stepMarksImplicit.length} 步）`,
      }),
    );
  }

  // —— 副轴：两条误差曲线 + 地板 + 当前标记 ——
  const errClip = clipRuns(tExplicit, error, auxAxis.ylim[0], auxAxis.ylim[1], (v) => v > 0);
  for (const run of errClip.runs) {
    scene.layers.push(lineLayer("aux", series(run.x, run.y, pal.residual, { width: 1.4 })));
  }
  const errImplClip = clipRuns(tImplicit, errorImplicit, auxAxis.ylim[0], auxAxis.ylim[1], (v) => v > 0);
  for (const run of errImplClip.runs) {
    scene.layers.push(lineLayer("aux", series(run.x, run.y, pal.curves[1], { width: 1.4, dash: [5, 3] })));
  }
  if (errClip.runs.length === 0 && errImplClip.runs.length === 0) {
    degraded.push("两条误差曲线在对数范围内都没有可画的点");
  }
  if (roundoffFloor !== undefined && roundoffFloor >= auxAxis.ylim[0] && roundoffFloor <= auxAxis.ylim[1]) {
    scene.layers.push(
      lineLayer(
        "aux",
        series([xlim[0], xlim[1]], [roundoffFloor, roundoffFloor], pal.reference, { width: 1, dash: [2, 3] }),
      ),
    );
  }
  const markX = tNow;
  const markY = errorMark.length > 0 ? errorMark[0] : undefined;
  if (markY !== undefined && Number.isFinite(markY) && markY > 0 && markX >= xlim[0] && markX <= xlim[1]) {
    scene.layers.push(scatterLayer("aux", [markX], [markY], { r: 4.5, color: pal.marker, label: "当前 t 处的显式误差（帧 error_mark）" }));
  }

  // —— 文本 ——
  const mainHead: string[] = [];
  if (lam !== undefined) mainHead.push(`λ=${formatNumber(lam, 4)}`);
  if (ratio !== undefined) mainHead.push(`刚性比 r=${formatNumber(ratio, 4)}`);
  if (hCeiling !== undefined) {
    mainHead.push(`显式 h≤${formatNumber(hCeiling, 4)}${hCeilingTheory === undefined ? "" : `（理论 ${formatNumber(hCeilingTheory, 4)}）`}`);
  }
  if (hImplicit !== undefined) mainHead.push(`隐式 h=${formatNumber(hImplicit, 4)}`);
  if (nStepsExplicit !== undefined && nStepsImplicit !== undefined) {
    mainHead.push(`步数 ${formatNumber(nStepsExplicit, 5)} vs ${formatNumber(nStepsImplicit, 4)}`);
  }
  const mainLegend: string[] = [
    `橙虚线=cos t（精确解，|y''|≤${curvatureMax === undefined ? "?" : formatNumber(curvatureMax, 2)}：解很平缓）　蓝=${explicitName ?? "显式"}　绿=${implicitName ?? "隐式"}`,
    `下沿两排短竖线=每一步的落点：显式 ${stepMarksExplicit.length} 个、隐式 ${stepMarksImplicit.length} 个（疏密就是步长）`,
    `固定纵轴 ±2：${explClip.nClipped + implClip.nClipped > 0 ? `有 ${explClip.nClipped + implClip.nClipped} 个样本越界已截断（真实 max|y|=${maxAbsY === undefined ? "?" : formatNumber(maxAbsY, 5)}）` : `全部样本都落在 ±2 内（真实 max|y|=${maxAbsY === undefined ? "?" : formatNumber(maxAbsY, 5)}）`}`,
  ];
  // 曲线末端不写 label：两个渲染器都会把它画在折线**最后一点右边**（主图右边缘之外 → 溢出画布）。
  scene.texts.push(
    ...textBlock(mainRect[0] + 6, textTop(mainRect), textBottom(mainRect), [`${mainHead.join("　")}`, ...mainLegend], pal.ink, unitBudget(mainRect[2]), "top"),
  );

  const auxLines: string[] = [];
  if (errMax !== undefined) auxLines.push(`显式误差 max=${formatNumber(errMax, 4)}（由**稳定性**限制：与容差无关）`);
  if (errFinal !== undefined && errImplFinal !== undefined) {
    auxLines.push(`当前 t 端点误差：显式 ${formatNumber(errFinal, 4)}、隐式 ${formatNumber(errImplFinal, 4)}`);
  }
  if (roundoffFloor !== undefined) {
    const onFloor = countAtCeiling(error, roundoffFloor) + countAtCeiling(errorImplicit, roundoffFloor);
    auxLines.push(
      `地板 ${formatNumber(roundoffFloor, 3)}=u/2：${onFloor} 个样本贴在地板上（可能是**精确 0**，不是"误差恰好等于它"）`,
    );
  }
  if (stepsRatio !== undefined) auxLines.push(`步长比 h_隐式/h_显式=${formatNumber(stepsRatio, 4)}×：误差轴只放误差，步长的疏密看主图下沿的地毯。`);
  if (curvature !== undefined) {
    let flat = 0;
    for (let i = 0; i < curvature.length; i += 1) if (Math.abs(curvature[i]) > 1e-12) flat += 1;
    if (flat !== curvature.length) {
      auxLines.push(`|y''| 有 ${curvature.length - flat} 个样本恰好为 0（解在该点曲率为零，不是缺数据）。`);
    }
  }
  scene.texts.push(...textBlock(auxRect[0] + 6, textTop(auxRect), textBottom(auxRect), auxLines, pal.muted, unitBudget(auxRect[2]), "top"));

  // —— 角标 ——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(badge("autoscale", "副轴：自适应 log（按本帧误差取整；会掩盖量级变化）", 8, 26), SLOT.auxAxis, width),
  );
  const crit = criticalOf(frame, "stiffness_ratio", 3183);
  if (ratio !== undefined) {
    scene.badges.push(placeBadge(criticalBadge("刚性比 r", ratio, crit, { beyond: "above" }), SLOT.critical, width));
  }
  const status: string[] = [];
  if (nStepsExplicit !== undefined && nStepsImplicit !== undefined) {
    status.push(
      `步数 ${formatNumber(nStepsExplicit, 5)} vs ${formatNumber(nStepsImplicit, 4)}` +
        (stepsRatio === undefined ? "" : `（${formatNumber(stepsRatio, 4)}×）`),
    );
  }
  if (fEvalsExplicit !== undefined && fEvalsImplicit !== undefined) {
    status.push(
      `求值 ${formatNumber(fEvalsExplicit, 5)} 次 vs ${formatNumber(fEvalsImplicit, 4)} 次` +
        (luImplicit === undefined ? "" : ` + ${formatNumber(luImplicit, 4)} 次 LU`) +
        (newtonImplicit === undefined ? "" : ` + ${formatNumber(newtonImplicit, 4)} 次 Newton`),
    );
  }
  if (nStepsExplicitPlanned !== undefined && nStepsExplicit !== undefined && nStepsExplicitPlanned > nStepsExplicit) {
    status.push(`注意：显式方法计划 ${formatNumber(nStepsExplicitPlanned, 5)} 步，只走到 ${formatNumber(nStepsExplicit, 5)} 步`);
  }
  if (maxAbsY !== undefined) status.push(`max|y_显式|=${formatNumber(maxAbsY, 5)}`);
  if (note !== undefined) status.push(note);
  scene.badges.push(placeBadge(badge("status", clipBadgeText(status.join("；")), 8, 8), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  const nBad =
    countNonFinite(tExact) +
    countNonFinite(yExact) +
    countNonFinite(tExplicit) +
    countNonFinite(yExplicit) +
    countNonFinite(tImplicit) +
    countNonFinite(yImplicit) +
    countNonFinite(error) +
    countNonFinite(errorImplicit);
  const extra: string[] = [];
  if (explClip.nClipped + implClip.nClipped > 0) {
    extra.push(`解曲线有 ${explClip.nClipped + implClip.nClipped} 个样本超出固定纵轴 ±2，已截断（真实 max|y| 见 meta.max_abs_y_explicit）`);
  }
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
