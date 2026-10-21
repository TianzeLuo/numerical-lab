/**
 * `area_convergence`（2.4 章）的 model：面积逼近的**阶**、面板与自适应细分。
 *
 * 左列（`main`，横轴 x）：被积函数 `f` 与当前 `k` 个面板上的分段逼近；
 * 右列（`err`，双对数）：实测误差曲线 + 实测拟合线 + **两条理论线**（公式阶 p 与
 * 光滑性上限 min(p, α+1)）。`method=adaptive` 时右列换成「估计误差 vs 真误差」。
 *
 * 面板与图层（顺序即绘制顺序，测试按此断言）
 * ------------------------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | line | `main` | 被积函数 `f(x)` |
 * | 2 | line | `main` | 当前 k 个面板上的分段逼近 `approx_*`（公式隐含的插值多项式，帧给） |
 * | 3 | line | `main` | 面板边界竖线（面板数 ≤ 129 时画；更密就只留文字报数） |
 * | 4 | scatter | `main` | 求值节点（同上，节点太多时只报数） |
 * | 5 | line | `main` | **adaptive**：每个已接受区间一根"深度梳"（高度 ∝ `interval_depth`）+ 梳顶点 |
 * | 6 | loglogfit | `err` | 实测误差 + 实测斜率线 + 理论斜率线（规则 2 的强制形态） |
 * | 7 | line | `err` | 实测拟合线 `fit_curve`（帧给的幂律，锚在窗口首点） |
 * | 8 | line | `err` | **光滑性上限** min(p, α+1) 参考线（斜率取自 `meta.slope_theory_smoothness`） |
 * | 9 | line | `err` | 舍入地板横线（只在地板离数据 < 1 个十倍程时画，否则只在文字里报） |
 * | 10 | line | `err` | 当前面板数 `k` 的游标竖线 + 该点标记（把左列与右列接起来） |
 * | 11 | line | `err` | **adaptive**：估计误差 `error_estimate_curve`（虚线）与容差 `tol` 参考线 |
 * | 12 | line | `err` | **adaptive**：真误差 `error_true_curve`（实线） |
 *
 * 角标槽位（固定，见 `common.ts::SLOT`）：0 来源 / 1 主轴固定 / 2 副轴自适应 /
 * 3 临界（本章这一格 `critical=[]`，如实说明）/ 4 状态 / 5 前端预览 / 6 溢出 / 7 降级。
 *
 * 这一页最容易骗人的两处，以及它们在图上的防法
 * --------------------------------------------
 * 1. **"双对数图上量到的斜率 = 求积公式的阶"**。`f=√x` 时三种公式的实测阶都被压到
 *    `min(p, α+1)=1.5`：只画公式阶那条线，读者会以为「Simpson 退化 / 这里算错了」。
 *    防法：`err` 面板**同时**画公式阶与光滑性上限两条理论线（后者只有斜率来自帧，
 *    见 `common.ts::powerLine` 的说明），并把 `α+1`、R²、拟合窗口写进同一块文字；
 *    `strings.smoothness_note` 原文照登。
 * 2. **"估计误差 < tol ⇒ 算对了"**。窄峰上估计器一开始看不见峰：`adaptive` 模式下
 *    估计与真误差两条曲线同屏，并把**已揭示区间里的最大低估倍数**算出来写在文字里
 *    （真误差/估计误差，只取两者都有限且 > 0 的 k）；帧自带的 `strings.estimator_note`
 *    （"4.3e-5 对 3.5e-2"）原样照登。
 *
 * 几条刻意的决定
 * --------------
 * 1. **左列的分段逼近是"公式隐含的插值多项式"**：`approx_x/approx_y` 直接来自帧，
 *    模型不重算积分（P8）。
 * 2. **误差曲线不随时间轴截断**：`error_curve`/`panel_count` 是**整条收敛研究**
 *    （后端与 `t` 无关地给出 20 个网格点，横坐标是**生效面板数**），时间轴揭示的是
 *    **左列的面板数 k**；两者的联系由 `err` 面板上那条 `k` 游标建立，
 *    不假装"右图也随时间揭示"。
 * 3. **没有可信窗口就不画拟合线**：`f=|x|` 上折点落在节点上、误差恒在地板，后端如实给
 *    `slope_measured=None`，`fit_curve/theory_curve` 填的是地板值。把地板值当"理论线"
 *    画出来会读成「理论说误差按地板下降」，所以这种情况只在文字里说明（宁可留白）。
 * 4. **`f=√x` 时 `theory_curve` 会冲到轴顶之上**（n=2 处 72.8）：纵轴范围把它一起纳入
 *    （这就是"公式阶 vs 实测"的对照本身），而不是把它裁掉——裁掉等于把
 *    「Simpson 是 4 阶」这句话从图上删了。
 */

import { niceLimits, niceTicks } from "../../shared/src/axes.js";
import type { Frame } from "../../shared/src/frame.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import {
  applyOverflow,
  canvasSize,
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
  toNumeric,
  segmentedSeries,
  type ModelOptions,
  type Rect,
  type SceneModel,
} from "../../shared/src/model.js";
import {
  axisModeBadge,
  badge,
  criticalBadge,
  lineLayer,
  loglogFitLayer,
  makeAxis,
  previewBadge,
  scatterLayer,
  series,
  sourceBadge,
} from "../../shared/src/scene.js";
import {
  CHAPTER,
  SLOT,
  axisTitle,
  axisYlabel,
  currentStep,
  fmt,
  fmtOpt,
  minMaxOfAll,
  noteOf,
  oneTwoFive,
  paramSpec,
  pickPairs,
  plainNote,
  positiveMinMaxOfAll,
  powerLine,
  sci,
  specTitle,
  splitColumns,
  textBlock,
  unitBudget,
  withZero,
} from "./common.js";

const VIZ = "area_convergence";
const TITLE = specTitle(VIZ, "面积逼近：阶、面板与自适应细分");

/** 面板边界/节点多到这个数以上就只报数不画（画出来是一团墨，不是信息）。 */
const EDGE_LIMIT = 129;
const NODE_LIMIT = 257;

export const areaConvergenceModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);
  const layout = splitColumns(width, height, { share: 0.5, gap: 64 });

  // —— 必需数组：两个模式共用 `x`/`f` ——
  const x = requireArray(frame, "x");
  const f = requireArray(frame, "f");
  requireSameLength(x, f, "帧数组 x 与 f");
  const pitfalls = requirePitfalls(frame);

  // —— 模式：按**数组的存在性**判断（结构化事实），meta 只用来交叉核对 ——
  const adaptive = frame.has("error_estimate_curve");
  const degraded: string[] = [];
  const isAdaptiveMeta = metaNumber(frame, "is_adaptive");
  if (isAdaptiveMeta === undefined) {
    degraded.push("帧 meta 缺 is_adaptive：模式按数组名判断（error_estimate_curve 在不在）");
  } else if ((isAdaptiveMeta > 0.5) !== adaptive) {
    degraded.push(`帧 meta.is_adaptive=${fmt(isAdaptiveMeta)} 与数组集合不一致：按数组画，并以本行说明`);
  }

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: metaNumber(frame, "t") ?? opts.t,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  const methodLabel = noteOf(frame, "method_label") ?? noteOf(frame, "method") ?? "?";
  const funcLabel = noteOf(frame, "func_label") ?? noteOf(frame, "func") ?? "?";
  const smoothness = noteOf(frame, "smoothness");
  const exact = metaNumber(frame, "exact");
  const value = metaNumber(frame, "value");
  const errorCurrent = metaNumber(frame, "error_current");
  const errorFloor = metaNumber(frame, "error_floor");
  const clipped = metaNumber(frame, "clipped_points");
  /** 整段的估计/真误差比值（后端算的标量；两个模式共用角标与状态行）。 */
  const estimatorOverTrue = metaNumber(frame, "estimator_over_true");

  // —— 左列：f 与分段逼近 ——
  const xRange = minMaxOfAll([x]) ?? [0, 1];
  const yRange = withZero(minMaxOfAll([f]));
  const xpad = 0.02 * (xRange[1] - xRange[0] || 1);
  const ypad = 0.08 * (yRange[1] - yRange[0] || 1);
  const mainAxis = makeAxis("main", layout.left, {
    xlim: niceLimits(xRange[0] - xpad, xRange[1] + xpad, 6),
    ylim: niceLimits(yRange[0], yRange[1] + ypad, 6),
    xlabel: "$x$",
    ylabel: axisYlabel(VIZ, "main", "f(x)"),
    title: axisTitle(VIZ, "main", "f 与分段逼近"),
    autoscaled: false,
  });
  scene.axes.push(mainAxis);
  scene.layers.push(lineLayer("main", segmentedSeries(x, f, pal.curves[0], { width: 2, label: "$f(x)$" })));

  const mainLines: string[] = [`${methodLabel}　${funcLabel}`];
  if (exact !== undefined) {
    mainLines.push(`真值 I=${fmt(exact, 8)}　当前 Q=${fmtOpt(value, 8)}　|Q−I|=${fmtOpt(errorCurrent, 3)}`);
  } else {
    degraded.push("帧 meta 缺 exact：不显示真值与当前误差（不按参数反推）");
  }
  if (smoothness !== undefined) mainLines.push(`光滑性：${smoothness}`);

  if (!adaptive) {
    // —— 经典复合规则：分段逼近 + 面板边界 + 节点 ——
    const approxX = requireArray(frame, "approx_x");
    const approxY = requireArray(frame, "approx_y");
    const nodeX = requireArray(frame, "node_x");
    const nodeY = requireArray(frame, "node_y");
    const panelEdge = requireArray(frame, "panel_edge");
    requireSameLength(approxX, approxY, "帧数组 approx_x 与 approx_y");
    requireSameLength(nodeX, nodeY, "帧数组 node_x 与 node_y");

    const panelsEff = metaNumber(frame, "panels");
    const panelsReq = metaNumber(frame, "panels_requested");
    const nMax = metaNumber(frame, "n_max");
    const step = currentStep(frame, opts, {
      lo: 1,
      hi: Math.max(1, Math.round(nMax ?? panelsReq ?? 1)),
      label: "面板数 $k$",
    });

    scene.layers.push(
      lineLayer(
        "main",
        segmentedSeries(approxX, approxY, pal.exact, {
          width: 1.7,
          label: "逼近",
        }),
      ),
    );
    if (panelEdge.length <= EDGE_LIMIT) {
      scene.layers.push(lineLayer("main", edgeSeries(mainAxis, panelEdge, pal.reference)));
    } else {
      mainLines.push(`面板边界 ${panelEdge.length} 条太密：不画竖线（误差曲线不受影响）`);
    }
    if (nodeX.length <= NODE_LIMIT) {
      scene.layers.push(scatterLayer("main", nodeX, nodeY, { r: 2, color: pal.marker, alpha: 0.85, label: "节点" }));
    } else {
      mainLines.push(`求值节点 ${nodeX.length} 个太密：不画点（曲线与边界已足够）`);
    }
    const nevals = metaNumber(frame, "n_evals");
    mainLines.push(
      `请求 k=${fmtOpt(panelsReq, 3)} → 生效 ${fmtOpt(panelsEff, 3)}（Simpson 取偶、3/8 取 3 的倍数）` +
        (nevals === undefined ? "" : `；求值 ${fmt(nevals, 4)} 次`),
    );
    if (panelsReq !== undefined && panelsEff !== undefined && Math.abs(panelsReq - panelsEff) > 0.5) {
      mainLines.push(`⚠ 生效面板数 ${fmt(panelsEff, 3)} ≠ 请求 ${fmt(panelsReq, 3)}：横坐标用的是生效值`);
    }
    mainLines.push(...step.notes);

    buildClassicErrorPanel(scene, frame, opts, layout.right, {
      degraded,
      errorFloor,
      clipped,
      panelsEff,
    });
  } else {
    // —— adaptive：区间划分（深度梳）+ 估计误差 vs 真误差 ——
    const intervalLo = requireArray(frame, "interval_lo");
    const intervalHi = requireArray(frame, "interval_hi");
    const intervalDepth = requireArray(frame, "interval_depth");
    const evalsCum = requireArray(frame, "evals_cum");
    const estCurve = requireArray(frame, "error_estimate_curve");
    const trueCurve = requireArray(frame, "error_true_curve");
    requireSameLength(intervalLo, intervalHi, "帧数组 interval_lo 与 interval_hi");
    requireSameLength(intervalLo, intervalDepth, "帧数组 interval_lo 与 interval_depth");
    requireSameLength(estCurve, trueCurve, "帧数组 error_estimate_curve 与 error_true_curve");

    const nIntervalsTotal = metaNumber(frame, "n_intervals_total");
    const nEvals = metaNumber(frame, "n_evals_total");
    const maxDepth = metaNumber(frame, "max_depth_used");
    const tol = metaNumber(frame, "tol");
    const estTotal = metaNumber(frame, "error_estimate_total");
    const trueTotal = metaNumber(frame, "error_true_total");
    const converged = metaNumber(frame, "converged");
    const step = currentStep(frame, opts, {
      lo: 1,
      hi: Math.max(1, Math.round(nIntervalsTotal ?? intervalLo.length)),
      label: "已接受区间数 $k$",
    });

    // 深度梳：每个已接受区间一根竖线，高度 ∝ 递归深度（显示几何，不是重算物理量）。
    const span = mainAxis.ylim[1] - mainAxis.ylim[0];
    const base = mainAxis.ylim[0] + 0.02 * span;
    const maxD = Math.max(1, maxDepth ?? 1);
    const depthScale = 0.32 * span;
    const combSeries = [];
    const combX: number[] = [];
    const combY: number[] = [];
    for (let i = 0; i < intervalLo.length; i += 1) {
      const lo = intervalLo[i] as number;
      const hi = intervalHi[i] as number;
      const d = intervalDepth[i] as number;
      if (!Number.isFinite(lo) || !Number.isFinite(hi) || !Number.isFinite(d) || !(hi > lo)) continue;
      const mid = 0.5 * (lo + hi);
      const top = base + (0.08 + 0.92 * (d / maxD)) * depthScale;
      combSeries.push(series([mid, mid], [base, top], pal.residual, { width: 1.6 }));
      combX.push(mid);
      combY.push(top);
    }
    if (combSeries.length > 0) {
      scene.layers.push(lineLayer("main", combSeries));
      scene.layers.push(scatterLayer("main", combX, combY, { r: 1.8, color: pal.residual, alpha: 0.9, label: "区间深度梳" }));
      mainLines.push(`区间深度梳（底部竖线）：${combSeries.length} 个已接受区间，高度 ∝ 递归深度`);
    } else {
      degraded.push("帧里没有可画的区间（interval_lo/hi 全非有限）：左列只画 f（并已说明）");
    }
    mainLines.push(
      `区间 ${fmtOpt(nIntervalsTotal, 4)} 个（预算 ${fmtOpt(metaNumber(frame, "budget"), 3)}）、` +
        `函数求值 ${fmtOpt(nEvals, 4)} 次、最深 ${fmtOpt(maxDepth, 3)} 层` +
        (converged === undefined ? "" : `；converged=${converged > 0.5 ? "true" : "false"}`),
    );
    mainLines.push(...step.notes);

    // —— 右列：估计误差 vs 真误差（规则 1 的核心：两者同屏） ——
    const evalVals = Array.from(evalsCum).filter((v) => Number.isFinite(v));
    const xLo = evalVals.length > 0 ? Math.min(...evalVals) : 1;
    const xHi = evalVals.length > 0 ? Math.max(...evalVals) : 2;
    const errAxis = makeAxis("err", layout.right, {
      xlim: niceLimits(xLo, Math.max(xHi, xLo + 1) * 1.3, 6),
      xscale: "linear",
      yscale: "log",
      xlabel: "累计函数求值次数",
      ylabel: axisYlabel(VIZ, "err", "误差"),
      title: "估计误差 vs 真误差（adaptive）",
      autoscaled: true,
    });
    const errRange = positiveMinMaxOfAll([estCurve, trueCurve, tol !== undefined && tol > 0 ? [tol] : undefined]) ?? [1e-16, 1];
    errAxis.ylim = niceLimits(errRange[0], errRange[1], 6, { log: true, pad: 0.02 });
    errAxis.yticks = niceTicks(errAxis.ylim[0], errAxis.ylim[1], 5, { log: true });
    scene.axes.push(errAxis);

    scene.layers.push(
      lineLayer(
        "err",
        segmentedSeries(evalsCum, estCurve, pal.warn, {
          width: 1.8,
          dash: [6, 4],
          accept: (v) => v > 0,
          label: "估计",
        }),
      ),
    );
    scene.layers.push(
      lineLayer(
        "err",
        segmentedSeries(evalsCum, trueCurve, pal.curves[0], { width: 2, accept: (v) => v > 0, label: "真误差" }),
      ),
    );
    const tolShown = tol !== undefined && tol > 0 && tol >= errAxis.ylim[0] && tol <= errAxis.ylim[1];
    if (tolShown) {
      scene.layers.push(
        lineLayer("err", series(errAxis.xlim, [tol as number, tol as number], pal.reference, { width: 1.1, dash: [5, 4], label: "tol" })),
      );
    }

    const ratio = ratioRange(estCurve, trueCurve, errorFloor);
    const errLines: string[] = ["两条曲线必须一起看：估计误差估的是嵌套法则自己那个模型的误差，不是真误差。"];
    if (ratio !== undefined) {
      errLines.push(
        `已揭示窗口内 估计/真误差 比：最小 ${fmt(ratio.min, 4)}（k=${ratio.minIndex}）、最大 ${fmt(ratio.max, 4)}（k=${ratio.maxIndex}）` +
          (ratio.skipped === 0 ? "" : `；${fmt(ratio.skipped, 3)} 个点因贴地板或为 0 不参与比值`),
      );
      if (ratio.max > 100) {
        errLines.push(
          `最大那一档出现在前段：估计已经掉到 ${sci(ratio.maxEst)} 而真误差还有 ${sci(ratio.maxTrue)}——` +
            `「估计 < tol」不足以证明算对了（manifest 的 pitfalls 里 814× 是同一现象的另一档）`,
        );
      }
    }
    errLines.push(
      `整段收尾：估计 ${fmtOpt(estTotal, 4)} vs 真 ${fmtOpt(trueTotal, 4)}` +
        (estimatorOverTrue === undefined ? "（帧未给比值）" : `，比值 ${fmt(estimatorOverTrue, 4)}`) +
        (tol === undefined ? "" : `；tol=${sci(tol)}`),
    );
    errLines.push(`截断到地板的点数 ${fmtOpt(clipped, 3)}（log 轴画不出 0，地板不是「完美收敛」）`);
    pushIf(errLines, plainNote(noteOf(frame, "estimator_note")));
    pushIf(errLines, plainNote(noteOf(frame, "worst_note")));
    scene.texts.push(
      ...textBlock(errAxis.rect[0] + 6, errAxis.rect[1] + 4, errAxis.rect[1] + errAxis.rect[3] - 4, errLines, pal.note, unitBudget(Math.round(errAxis.rect[2] * 0.8)), "bottom"),
    );
    mainLines.push("右列：虚线=估计误差，实线=真误差，点线=tol 停止判据");
  }

  scene.texts.push(
    ...textBlock(mainAxis.rect[0] + 6, mainAxis.rect[1] + 4, mainAxis.rect[1] + mainAxis.rect[3] - 4, mainLines, pal.note, unitBudget(Math.round(mainAxis.rect[2] * 0.8)), "top"),
  );

  // —— 角标 ——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(badge("autoscale", "右列 err：自适应（log 纵轴按本帧误差取整；自适应会掩盖量级变化）", 8, 26), SLOT.auxAxis, width),
  );
  const critList = paramSpec(VIZ, "n")?.critical ?? [];
  const nNow = metaNumber(frame, "n_max") ?? metaNumber(frame, "budget");
  scene.badges.push(
    placeBadge(
      critList.length === 0
        ? badge(
            "critical",
            "无参数临界值（manifest critical=[]）：复合规则的阶 p 随 f 的光滑性变，" +
              `没有诚实的滑块阈值${adaptive ? "；自适应模式的判据是 tol" : ""}`,
            8,
            62,
          )
        : criticalBadge("n", nNow ?? Number.NaN, critList, { beyond: "above" }),
      SLOT.critical,
      width,
    ),
  );
  const status: string[] = [];
  if (adaptive) {
    const r = estimatorOverTrue;
    status.push(r === undefined ? "自适应：帧未给出估计/真误差比值" : `自适应：估计/真误差 = ${fmt(r, 4)}${r < 1 ? "（偏乐观）" : "（偏保守）"}`);
  } else {
    status.push(`生效面板 ${fmtOpt(metaNumber(frame, "panels"), 3)}（请求 ${fmtOpt(metaNumber(frame, "panels_requested"), 3)}）`);
  }
  if (clipped !== undefined && clipped > 0) status.push(`⚠ ${fmt(clipped, 3)} 点被截到舍入地板`);
  scene.badges.push(placeBadge(badge("status", status.join("；"), 8, 80), SLOT.status, width));
  if (scene.dataSource.startsWith("backend+preview")) scene.badges.push(placeBadge(previewBadge(), SLOT.preview, width));

  let rawNonFinite = 0;
  for (const name of frame.names) rawNonFinite += countNonFinite(toNumeric(frame.array(name)));
  applyOverflow(scene, rawNonFinite, { slot: SLOT.overflow, width });
  const degradedObj = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedObj) scene.badges.push(degradedObj);

  requireDrawableLayers(scene);
  return scene;
};

/* --------------------------------------------------------------------------- */
/* 经典复合规则的右列（双对数 + 两条理论线）                                      */
/* --------------------------------------------------------------------------- */

interface ClassicOptions {
  degraded: string[];
  errorFloor: number | undefined;
  clipped: number | undefined;
  panelsEff: number | undefined;
}

function buildClassicErrorPanel(
  scene: ReturnType<typeof sceneShell>,
  frame: Frame,
  opts: ModelOptions,
  rect: Rect,
  o: ClassicOptions,
): void {
  const pal = palette(opts.theme);
  const panelCount = requireArray(frame, "panel_count");
  const errorCurve = requireArray(frame, "error_curve");
  requireSameLength(panelCount, errorCurve, "帧数组 panel_count 与 error_curve");

  const slopeMeasured = metaNumber(frame, "slope_measured");
  const slopeTheory = metaNumber(frame, "slope_theory");
  const slopeCap = metaNumber(frame, "slope_theory_smoothness");
  const fitStart = metaNumber(frame, "fit_start");
  const fitEnd = metaNumber(frame, "fit_end");
  const r2 = metaNumber(frame, "fit_r2");
  const orderCap = metaNumber(frame, "order_cap");

  const axis = makeAxis("err", rect, {
    xscale: "log",
    yscale: "log",
    xlabel: "生效面板数 $n$（对数轴）",
    ylabel: axisYlabel(VIZ, "err", "误差"),
    title: axisTitle(VIZ, "err", "|Q−I| vs 面板数（双对数）"),
    autoscaled: true,
  });
  scene.axes.push(axis);

  const positiveX = Array.from(panelCount).filter((v) => Number.isFinite(v) && v > 0);
  // 右端留 60%（log 轴）：拟合线/理论线/地板的 label 都画在最后可见点右侧，
  // 不留白就会被面板右边界裁成「地板 3.0」这种半截标签（实测余量只有 7px）。
  axis.xlim = [Math.max(Number.MIN_VALUE, Math.min(...positiveX) * 0.92), Math.max(...positiveX) * 1.6];
  const xt = oneTwoFive(positiveX);
  axis.xticks = xt.length >= 2 ? xt : niceTicks(axis.xlim[0], axis.xlim[1], 5, { log: true });

  // 拟合窗口（帧给的 [fit_start, fit_end]）：窗口内的点才上 loglogfit 层。
  const winIdx: number[] = [];
  if (fitStart !== undefined && fitEnd !== undefined) {
    for (let i = 0; i < panelCount.length; i += 1) {
      const v = panelCount[i] as number;
      if (v >= fitStart - 1e-9 && v <= fitEnd + 1e-9) winIdx.push(i);
    }
  }
  const winX = Float64Array.from(winIdx.map((i) => panelCount[i] as number));
  const winY = Float64Array.from(winIdx.map((i) => errorCurve[i] as number));
  const fitValid = slopeMeasured !== undefined && slopeTheory !== undefined && winIdx.length >= 2;

  // 锚点：优先用帧的拟合窗口首点（与后端 theory_curve 同一锚），否则用第一个可画点。
  const first = pickPairs(panelCount, errorCurve, (_xv, yv) => yv > 0);
  const anchorX = winIdx.length > 0 ? (panelCount[winIdx[0] as number] as number) : (first.x[0] ?? Number.NaN);
  const anchorY = winIdx.length > 0 ? (errorCurve[winIdx[0] as number] as number) : (first.y[0] ?? Number.NaN);

  // 理论线只在**拟合窗口内**画：斜率离开窗口就没有意义，而幂律外推到 n=1 会把纵轴顶高几个
  // 十倍程（exp/trapezoid 实测：外推到 n=1 得 14.3，整条误差曲线被压成一条细线）。
  const capX = winIdx.length >= 2 ? Float64Array.from(winIdx.map((i) => panelCount[i] as number)) : panelCount;
  const capLine =
    slopeCap !== undefined && Number.isFinite(anchorX) && anchorY > 0 && capX.length >= 2
      ? powerLine(capX, anchorX, anchorY, slopeCap)
      : undefined;
  const pLine =
    slopeTheory !== undefined && !fitValid && Number.isFinite(anchorX) && anchorY > 0 && capX.length >= 2
      ? powerLine(capX, anchorX, anchorY, slopeTheory)
      : undefined;

  const yRange = positiveMinMaxOfAll([errorCurve, capLine, pLine]) ?? [1e-16, 1];
  axis.ylim = niceLimits(yRange[0], yRange[1], 6, { log: true, pad: 0.03 });
  axis.yticks = niceTicks(axis.ylim[0], axis.ylim[1], 5, { log: true });

  // 图层 6：规则 2 的强制形态（实测 + 实测斜率 + 理论斜率），只覆盖拟合窗口。
  if (fitValid) {
    scene.layers.push(
      loglogFitLayer("err", winX, winY, slopeMeasured as number, slopeTheory as number, {
        seriesLabel: `实测 ${fmt(slopeMeasured as number, 5)}`,
        theoryLabel: `公式阶 ${fmt(slopeTheory as number, 5)}`,
        color: pal.curves[0],
        theoryColor: pal.reference,
      }),
    );
    const fitCurve = requireArray(frame, "fit_curve");
    scene.layers.push(
      lineLayer(
        "err",
        segmentedSeries(panelCount, fitCurve, pal.exact, {
          width: 1.3,
          dash: [5, 3],
          accept: (v) => v > 0 && v >= axis.ylim[0] && v <= axis.ylim[1],
          label: "拟合",
        }),
      ),
    );
  } else {
    o.degraded.push("帧没有可信的拟合窗口（slope_measured=null）：不画拟合线/理论线，文字里说明原因");
  }
  // 图层 8：光滑性上限（第二条理论线）。
  if (capLine !== undefined) {
    const same = slopeTheory !== undefined && Math.abs((slopeCap as number) - slopeTheory) < 1e-9;
    scene.layers.push(
      lineLayer(
        "err",
        segmentedSeries(capX, capLine, pal.warn, {
          width: 1.5,
          dash: [2, 3],
          accept: (v) => v > 0 && v >= axis.ylim[0] && v <= axis.ylim[1],
          // 标签画在最后可见点右侧：长标签在 log 轴右端必被裁（实测 252px 需求 / 7px 余量），
          // 值交给文字行「光滑性上限 min(p,α+1)=…」，屏上只留短标签。
          label: same ? "理论" : "上限",
        }),
      ),
    );
  } else if (noteOf(frame, "smoothness_note") !== undefined) {
    o.degraded.push("帧 meta 缺 slope_theory_smoothness：光滑性上限那条理论线画不出来（文字里保留 min(p,α+1)）");
  }
  if (pLine !== undefined) {
    scene.layers.push(
      lineLayer(
        "err",
        segmentedSeries(capX, pLine, pal.reference, {
          width: 1.3,
          dash: [6, 4],
          accept: (v) => v > 0 && v >= axis.ylim[0] && v <= axis.ylim[1],
          label: "公式阶",
        }),
      ),
    );
  }
  // 图层 9：地板横线（只在地板离数据一个十倍程以内时画）。
  if (o.errorFloor !== undefined && o.errorFloor > 0 && o.errorFloor >= axis.ylim[0] / 10 && o.errorFloor <= axis.ylim[1]) {
    scene.layers.push(
      lineLayer("err", series(axis.xlim, [o.errorFloor, o.errorFloor], pal.muted, { width: 1, dash: [3, 3], label: "地板" })),
    );
  }
  // 全量误差曲线**最后**画：拟合线/理论线都在它下面，重合的地方看到的是数据本身，
  // 只有它们分叉的地方才露出参考线（顺序反了会变成"用拟合线盖住数据"）。
  scene.layers.push(
    lineLayer(
      "err",
      segmentedSeries(panelCount, errorCurve, pal.curves[0], {
        width: 1.7,
        accept: (v) => v > 0,
        label: fitValid ? "" : "实测",
      }),
    ),
  );
  // 图层 10：当前面板数 k 的游标（把左列与右列接起来）。
  if (o.panelsEff !== undefined && Number.isFinite(o.panelsEff) && o.panelsEff >= axis.xlim[0] && o.panelsEff <= axis.xlim[1]) {
    scene.layers.push(
      lineLayer("err", series([o.panelsEff, o.panelsEff], axis.ylim, pal.marker, { width: 1.1, dash: [4, 3] })),
    );
    const kIdx = exactIndex(panelCount, o.panelsEff);
    if (kIdx >= 0) {
      scene.layers.push(
        scatterLayer("err", [panelCount[kIdx] as number], [errorCurve[kIdx] as number], { r: 3.6, color: pal.marker }),
      );
    }
  }

  const lines: string[] = [
    `实测斜率 ${fmtOpt(slopeMeasured, 6)}　公式阶 p=${fmtOpt(slopeTheory, 6)}　光滑性上限 min(p,α+1)=${fmtOpt(slopeCap, 6)}` +
      (orderCap === undefined ? "" : `（α+1=${fmt(orderCap, 4)}）`),
  ];
  const em = minMaxOfAll([errorCurve]);
  lines.push(
    `拟合窗口 n∈[${fmtOpt(fitStart, 4)}, ${fmtOpt(fitEnd, 4)}]，R²=${fmtOpt(r2, 6)}` +
      (em === undefined ? "" : `；实测误差 ${fmt(em[1], 4)} → ${fmt(em[0], 4)}`),
  );
  if (slopeCap !== undefined && slopeTheory !== undefined && Math.abs(slopeCap - slopeTheory) > 1e-9) {
    lines.push(
      `两条理论线（公式阶 ${fmt(slopeTheory, 4)} 与光滑性上限 ${fmt(slopeCap, 4)}）不重合——实测贴在后者上：` +
        `量到的是 f 光滑度的阶，不是公式退化`,
    );
  } else {
    lines.push("两条理论线重合（min(p,α+1) 不比 p 更紧）：实测应当贴着它，除非折点让逐段误差塌到地板");
  }
  if (noteOf(frame, "fit_note") === undefined) {
    lines.push("没有实测斜率 = 这一段没有可信的幂律（窗口触地板、误差不单调或 R²<0.98）：帧里留白而不是硬凑一条直线。");
  }
  if (o.errorFloor !== undefined) {
    lines.push(`地板 ${sci(o.errorFloor)}；被截断点数 ${fmtOpt(o.clipped, 3)}；红色竖线 = 当前 k，红点 = 该 k 的实测误差`);
  }
  pushIf(lines, plainNote(noteOf(frame, "floor_note")));
  scene.texts.push(...textBlock(rect[0] + 6, rect[1] + 4, rect[1] + rect[3] - 4, lines, pal.note, unitBudget(Math.round(rect[2] * 0.8)), "bottom"));
}

/* --------------------------------------------------------------------------- */
/* 局部工具                                                                      */
/* --------------------------------------------------------------------------- */

/** 条件式入队（`undefined` 不入队；顺带把"缺说明"的判据集中在一处）。 */
function pushIf(lines: string[], body: string | undefined): void {
  if (body !== undefined) lines.push(body);
}

/**
 * 已揭示窗口里 `真误差/估计误差` 的**比值区间**（>1 = 估计偏小/偏乐观）。
 *
 * 这是**两个帧数组相除**（显示空间的比值），不是重算误差：`est`、`tr` 都来自帧。
 * 贴地板（`≤ floor`）或为 0 的点**不参与**：那些点上估计值是被减法吃剩的噪声，
 * 拿它们做分母会得到一个只反映浮点地板的天文数字（实测 k=0 时 3.19e14，毫无信息量）。
 * 跳过的点数如实返回，由调用方写进面板文字。
 */
function ratioRange(
  est: ArrayLike<number>,
  tr: ArrayLike<number>,
  floor: number | undefined,
): { min: number; minIndex: number; max: number; maxIndex: number; maxEst: number; maxTrue: number; skipped: number } | undefined {
  const low = floor === undefined ? 0 : floor * 1.000001;
  const n = Math.min(est.length, tr.length);
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  let minIndex = -1;
  let maxIndex = -1;
  let maxEst = 0;
  let maxTrue = 0;
  let skipped = 0;
  for (let i = 0; i < n; i += 1) {
    const e = Math.abs(est[i] as number);
    const t = Math.abs(tr[i] as number);
    if (!Number.isFinite(e) || !Number.isFinite(t) || e <= low || t <= low) {
      skipped += 1;
      continue;
    }
    const r = t / e;
    if (r < min) min = r;
    if (r > max) {
      max = r;
      maxIndex = i;
      maxEst = e;
      maxTrue = t;
    }
    if (r === min) minIndex = i;
  }
  return maxIndex < 0 ? undefined : { min, minIndex, max, maxIndex, maxEst, maxTrue, skipped };
}

/** 精确命中的下标（找不到返回 −1）：把当前 k 标到误差曲线上用。 */
function exactIndex(values: ArrayLike<number>, target: number): number {
  for (let i = 0; i < values.length; i += 1) {
    if (Math.abs((values[i] as number) - target) < 1e-9) return i;
  }
  return -1;
}

/** 面板边界竖线：一个图层里塞 N 条两点折线（比 N 个图层省，也比逐条 stroke 稳）。 */
function edgeSeries(axis: { ylim: [number, number] }, edges: ArrayLike<number>, color: string) {
  const out = [];
  for (let i = 0; i < edges.length; i += 1) {
    const v = edges[i] as number;
    if (!Number.isFinite(v)) continue;
    out.push(series([v, v], [axis.ylim[0], axis.ylim[1]], color, { width: 0.7, alpha: 0.5 }));
  }
  return out;
}
