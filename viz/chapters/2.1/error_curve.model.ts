/**
 * `error_curve`（2.1 章）的 model：节点分布从等距**连续变形**到 Chebyshev，误差与 Λ 怎么跟着走。
 *
 * 面板与图层（顺序即绘制顺序，测试按此断言）
 * ------------------------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | line | `main` | `f_values`（被插函数）+ `p_morph`（当前变形 t 的插值多项式） |
 * | 2 | scatter | `main` | `nodes_x` / `nodes_y`（当前节点；拖时间轴就能看见它们往两端挪） |
 * | 3 | line | `aux` | `error`（当前误差）+ `error_equispaced` / `error_chebyshev`（同一 n 的两个端点，淡色） |
 * | 4 | line | `aux` | `envelope`（严格包络）+ `monic`（首一乘积 \|∏(x−x_i)\|） |
 * | 5 | line | `aux` | 地板参考线 `u/2`（节点上的精确 0 画在这里） |
 * | 6 | line | `lam` | `lambda_vs_t`（Λ 随变形参数 t） |
 * | 7 | line | `lam` | 当前 t 的竖直游标 + Λ 的全局最小点标记 + 当前点标记 |
 *
 * 时间轴语义
 * ----------
 * `mode="scrub"`、`dt=0.01`、`tMax=1`：这里的 `t` **不是时间**，而是**节点分布的变形参数**
 * （0 = 等距，1 = Chebyshev）。正文一律写 `t`（变形），并且用 `meta.morph` 而不是 `opts.t`。
 *
 * 防误导七条里本页的重点
 * ----------------------
 * * 规则 3：误差、包络、首一乘积三条曲线同屏（`requiredAux=["error"]`）；
 * * 规则 4：三个自适应轴各挂角标 + 正文写明"自适应会掩盖量级变化"；
 * * 规则 6：`n=12` 的临界值进 `criticalBadge`；地板值 `u/2` 画成参考线（规则 1 的"地板要标出来"）。
 *
 * 两条最容易被误读的地方，以及本页怎么防
 * --------------------------------------
 * 1. **包络被当成误差估计**：包络是**严格上界**，n=12、f=Runge 时它比实际误差大 1.2e6 倍。
 *    本页把包络画在同一条 log 轴上（差异是六个数量级，一眼可见），并在正文里给出
 *    `meta.envelope_looseness` 与"包络落到 100u 以下就没有意义"（`meta.tightness_points`
 *    正是后端为此统计的点数），还画了地板线——读者不会把"贴着包络"当成"误差就在那里"。
 * 2. **Λ(t) 被读成严格单调**：它整体下降，但末端会回升。`lam` 面板把 21 个采样点全部画出，
 *    并把**全局最小点**标出来（帧数组读数）：最小值在 t≈0.85 而不是 t=1，
 *    这件事只能靠"看见那个钩子"来传达。
 */

import { niceLimits } from "../../shared/src/axes.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import {
  applyOverflow,
  canvasSize,
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
  requireSameLength,
  sceneShell,
  segmentedSeries,
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
  type LineSeries,
} from "../../shared/src/scene.js";
import { axisTitleOf, axisYlabelOf, num, plain, rowLayout, textBlock, unitBudget } from "./scaffold.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "2.1";
const VIZ = "error_curve";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "误差曲线、包络与节点变形";
/** manifest 的 `n.critical = [12]`；`frame.critical.n` 缺失时的回落值。 */
const CRITICAL_N = 12;

const SLOT = {
  source: 0,
  axisMode: 1,
  auxAxis: 2,
  lamAxis: 3,
  critical: 4,
  status: 5,
  overflow: 6,
  degraded: 7,
} as const;

export const errorCurveModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需数组 ——
  const xs = requireArray(frame, "x_grid");
  const f = requireArray(frame, "f_values");
  const p = requireArray(frame, "p_morph");
  const err = requireArray(frame, "error");
  const env = requireArray(frame, "envelope");
  const monic = requireArray(frame, "monic");
  const errEq = requireArray(frame, "error_equispaced");
  const errCh = requireArray(frame, "error_chebyshev");
  const nodesX = requireArray(frame, "nodes_x");
  const nodesY = requireArray(frame, "nodes_y");
  const tAxis = requireArray(frame, "t_axis");
  const lamT = requireArray(frame, "lambda_vs_t");
  requireSameLength(xs, f, "x_grid 与 f_values");
  requireSameLength(xs, p, "x_grid 与 p_morph");
  requireSameLength(xs, err, "x_grid 与 error");
  requireSameLength(xs, env, "x_grid 与 envelope");
  requireSameLength(xs, monic, "x_grid 与 monic");
  requireSameLength(xs, errEq, "x_grid 与 error_equispaced");
  requireSameLength(xs, errCh, "x_grid 与 error_chebyshev");
  requireSameLength(nodesX, nodesY, "nodes_x 与 nodes_y");
  requireSameLength(tAxis, lamT, "t_axis 与 lambda_vs_t");
  const pitfalls = requirePitfalls(frame);

  // —— 可选量（缺了降级，不猜）——
  const degraded: string[] = [];
  const metaOf = (key: string, label: string): number | undefined => {
    const v = metaNumber(frame, key);
    if (v === undefined) degraded.push(`帧 meta 缺 ${key}：${label}`);
    return v;
  };
  const n = metaOf("n", "标题行不显示节点数 n");
  const morph = metaOf("morph", "不画当前变形位置的游标（不拿请求 t 冒充）");
  const maxErr = metaOf("max_error", "不显示当前最大误差（不重算）");
  const maxErrEq = metaOf("max_error_equispaced", "不显示等距端点的最大误差");
  const maxErrCh = metaOf("max_error_chebyshev", "不显示 Chebyshev 端点的最大误差");
  const lamMorph = metaOf("lambda_morph", "不显示当前 Λ");
  const lamEq = metaOf("lambda_equispaced", "不显示等距端点的 Λ");
  const lamCh = metaOf("lambda_chebyshev", "不显示 Chebyshev 端点的 Λ");
  const lamRatio = metaOf("lambda_ratio", "不显示两个端点的 Λ 之比");
  const envMax = metaOf("max_envelope", "不显示包络最大值");
  const envLoose = metaOf("envelope_looseness", "不显示包络的宽松倍数");
  const tightPoints = metaOf("tightness_points", "不显示「包络高于 100u」的点数");
  const tightMedian = metaOf("tightness_median", "不显示包络紧度的中位数");
  const monicMax = metaOf("monic_max", "不显示首一乘积的最大值");
  const monicTheory = metaOf("monic_sup_theory", "不显示首一乘积的理论下界 2^(1−n)");
  const monicRatio = metaOf("monic_ratio_to_theory", "不显示首一乘积与理论值之比");
  const errFloor = metaOf("error_floor", "不画地板参考线（不猜 u/2）");
  const nErrFloor = metaOf("n_error_at_floor", "不显示被夹到地板的误差点数");
  const nMonicFloor = metaOf("n_monic_at_floor", "不显示被夹到地板的首一乘积点数");
  const nZeroErr = metaOf("n_exact_zero_error", "不显示节点上精确 0 的误差点数");
  const nZeroMonic = metaOf("n_exact_zero_monic", "不显示节点上精确 0 的首一乘积点数");
  const funcLabel = plain(stringOf(frame, "function_label"));
  const note = plain(stringOf(frame, "note"));
  const tMeta = metaNumber(frame, "t");
  const tNow = tMeta ?? opts.t;
  if (tMeta === undefined) degraded.push("帧 meta 缺 t：scene.t 改用请求 t（两者可能不同）");
  if (morph !== undefined && tMeta !== undefined && Math.abs(tMeta - morph) > 0.005) {
    degraded.push(`请求 t=${num(tMeta, 5)} 与帧 meta.morph=${num(morph, 5)} 不一致：按帧里的 morph 画`);
  }

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: tNow,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  // —— 排版：主图（f 与变形后的 p）/ 误差-包络-首一乘积（log）/ Λ(t)（log）——
  // footer=144：pitfalls 实测 6 行 → 面板底 456，与 pitfalls 框留 6px。
  const layout = rowLayout(width, height, [1.45, 1.0, 0.72], { gap: 36, footer: 144 });
  const [mainRect, auxRect, lamRect] = layout.rects as [
    [number, number, number, number],
    [number, number, number, number],
    [number, number, number, number],
  ];

  const mainRange = rangeLinear([f, p], [-1, 1]);
  const mainAxis = makeAxis("main", mainRect, {
    xlim: [-1, 1],
    ylim: mainRange,
    xlabel: "",
    ylabel: axisYlabelOf(SPEC, "main", "y"),
    title: axisTitleOf(SPEC, "main", "f 与随 t 变形的插值多项式"),
    autoscaled: true,
  });
  const auxRange = rangeLog([err, env, monic, errEq, errCh], [1e-18, 1]);
  const auxAxis = makeAxis("aux", auxRect, {
    xlim: [-1, 1],
    ylim: auxRange,
    yscale: "log",
    xlabel: "",
    ylabel: axisYlabelOf(SPEC, "aux", "误差与首一乘积"),
    title: axisTitleOf(SPEC, "aux", "|f-p|、严格包络、|∏(x-x_i)|（log）"),
    autoscaled: true,
  });
  const lamRange = rangeLog([lamT], [1, 1e3]);
  const lamAxis = makeAxis("lam", lamRect, {
    xlim: [0, 1],
    ylim: lamRange,
    yscale: "log",
    xlabel: "$x$（上两行）与变形参数 $t$（本行：$0=$等距，$1=Chebyshev$）",
    ylabel: "$Λ(t)$",
    title: "Λ(t)：整体下降，但不是严格单调（末段会回升）",
    autoscaled: true,
  });
  scene.axes.push(mainAxis, auxAxis, lamAxis);

  // —— 图层 1：f 与当前变形下的插值多项式 ——
  scene.layers.push(
    lineLayer("main", [
      series(xs, f, pal.exact, { width: 2.2 }),
      series(xs, p, pal.curves[0], { width: 1.7 }),
    ]),
  );

  // —— 图层 2：当前节点（拖时间轴时它们会往两端挪——"变形"就是这件事）——
  const nodePx = Math.max(1.8, Math.min(mainRect[2], mainRect[3]) / 80);
  if (nodesX.length > 0) {
    scene.layers.push(scatterLayer("main", nodesX, nodesY, { r: nodePx, color: pal.curves[0], label: "当前节点（$t=1$ 时按 cos 分布挤向两端）" }));
  }

  // —— 图层 3~4：误差、包络、首一乘积（规则 3：三条同屏）——
  const auxSeries: LineSeries[] = [
    ...segmentedSeries(xs, errEq, pal.reference, { width: 1.1, dash: [4, 4], accept: (v) => v > 0 }),
    ...segmentedSeries(xs, errCh, pal.curves[1], { width: 1.1, dash: [4, 4], accept: (v) => v > 0 }),
    ...segmentedSeries(xs, env, pal.warn, { width: 1.5, dash: [3, 3], accept: (v) => v > 0 }),
    ...segmentedSeries(xs, monic, pal.residual, { width: 1.4, accept: (v) => v > 0 }),
  ];
  const curSeries = segmentedSeries(xs, err, pal.curves[0], { width: 1.9, accept: (v) => v > 0 });
  if (curSeries.length + auxSeries.length > 0) scene.layers.push(lineLayer("aux", [...auxSeries, ...curSeries]));
  else degraded.push("误差/包络/首一乘积里没有一个 >0 的有限值：log 轴上一段曲线都画不出来");

  // —— 图层 5：地板参考线（节点上的精确 0 就画在这条线上）——
  if (errFloor !== undefined && errFloor > 0) {
    scene.layers.push(lineLayer("aux", series([-1, 1], [errFloor, errFloor], pal.reference, { width: 1, dash: [2, 4] })));
  }

  // —— 图层 6：Λ(t) 的 21 个采样点 ——
  const lamSeries = segmentedSeries(tAxis, lamT, pal.curves[2] ?? pal.curves[0], { width: 1.9, accept: (v) => v > 0 });
  if (lamSeries.length > 0) scene.layers.push(lineLayer("lam", lamSeries));
  else degraded.push("lambda_vs_t 里没有 >0 的有限值：Λ(t) 面板画不出曲线");

  // —— 图层 7：游标 + 全局最小点 + 当前点 ——
  const markT = frame.has("mark_t") ? requireArray(frame, "mark_t") : undefined;
  const tNowMorph = markT !== undefined && markT.length > 0 ? (markT[0] as number) : morph;
  const lamNow = markT !== undefined && markT.length > 1 ? (markT[1] as number) : lamMorph;
  if (tNowMorph !== undefined) {
    scene.layers.push(lineLayer("lam", series([tNowMorph, tNowMorph], [lamAxis.ylim[0], lamAxis.ylim[1]], pal.cursor, { width: 1, dash: [3, 3] })));
  }
  const lamXY: number[] = [];
  const lamYY: number[] = [];
  if (tNowMorph !== undefined && lamNow !== undefined && lamNow > 0) {
    lamXY.push(tNowMorph);
    lamYY.push(lamNow);
  }
  const minIdx = argMin(lamT);
  const tMin = minIdx >= 0 ? valueAt(tAxis, minIdx) : undefined;
  const vMin = minIdx >= 0 ? valueAt(lamT, minIdx) : undefined;
  if (tMin !== undefined && vMin !== undefined && vMin > 0) {
    lamXY.push(tMin);
    lamYY.push(vMin);
  }
  if (lamXY.length > 0) {
    scene.layers.push(scatterLayer("lam", lamXY, lamYY, { r: 3.4, color: pal.marker, label: "当前 $t$ 与 $\\Lambda(t)$ 的全局最小点" }));
  }

  // —— 文本 ——
  const summary = paramSummary(frame, opts, [
    { name: "n", label: "$n$" },
    { name: "a", label: "$a$" },
  ]);
  degraded.push(...summary.degraded);
  const mainBudget = unitBudget(mainRect[2]);
  const mainLines: string[] = ["橙=f(x)　蓝=当前 t 的插值多项式　圆点=当前节点"];
  if (morph !== undefined) {
    mainLines.push(`变形 t=${num(morph, 4)}（0=等距，1=Chebyshev）　${summary.text}　${funcLabel ?? ""}`);
    if (Math.abs(tNow - morph) > 0.005) {
      mainLines.push(`请求 t=${num(tNow, 5)}，帧实到 morph=${num(morph, 5)}（时间轴不是时间，是节点分布的变形参数）。`);
    }
  } else {
    mainLines.push(`${summary.text}　${funcLabel ?? ""}`);
  }
  mainLines.push("t 变大不是「节点变密」：节点按 cos 分布被推向两端，区间中间反而变稀——所以两端的振荡先消失。");
  mainLines.push(`节点上的误差恰好为 0（本帧 ${nZeroErr === undefined ? "—" : num(nZeroErr, 3)} 个点），穿过节点不代表拟合好。`);
  if (note !== undefined) mainLines.push(`帧 note：${note}`);
  scene.texts.push(...textBlock(mainRect[0] + 8, mainRect[1] + 4, mainRect[1] + mainRect[3] - 3, mainLines, pal.ink, mainBudget, "top"));

  const auxBudget = unitBudget(auxRect[2]);
  const auxLines: string[] = ["蓝=当前误差　灰/绿虚线=等距与 Chebyshev 端点　红虚线=包络　紫=|∏(x−x_i)|　灰横线=地板"];
  if (maxErr !== undefined && maxErrEq !== undefined && maxErrCh !== undefined) {
    auxLines.push(`当前最大误差 ${num(maxErr, 4)}（同一 n 的两个端点：等距 ${num(maxErrEq, 4)}、Chebyshev ${num(maxErrCh, 4)}）。`);
  }
  if (envMax !== undefined) {
    auxLines.push(
      `严格包络（红虚线）max=${num(envMax, 4)}` +
        (envLoose === undefined ? "" : `，比实际误差宽松 ${num(envLoose, 3)} 倍`) +
        "——包络只是「不会超过」，不是误差估计。",
    );
  }
  if (tightPoints !== undefined) {
    auxLines.push(
      `包络 ≤100u 时量到的是舍入噪声：帧只统计包络 >100u 的 ${num(tightPoints, 5)} 个点` +
        (tightMedian === undefined ? "" : `（紧度中位数 ${num(tightMedian, 4)}）`) +
        "。",
    );
  }
  if (errFloor !== undefined) {
    auxLines.push(
      `地板 u/2=${num(errFloor, 3)}：误差 ${nErrFloor === undefined ? "—" : num(nErrFloor, 3)} 点、|∏| ${nMonicFloor === undefined ? "—" : num(nMonicFloor, 3)} 点被夹` +
        `（|∏| 在节点上恰好为 0：${nZeroMonic === undefined ? "—" : num(nZeroMonic, 3)} 个点）。`,
    );
  }
  scene.texts.push(...textBlock(auxRect[0] + 8, auxRect[1] + 4, auxRect[1] + auxRect[3] - 3, auxLines, pal.muted, auxBudget, "top"));

  const lamBudget = unitBudget(lamRect[2]);
  const lamLines: string[] = ["橙点=当前 t　红点=Λ(t) 全局最小点　灰竖线=当前 t 位置"];
  if (lamEq !== undefined && lamCh !== undefined) {
    lamLines.push(
      `Λ(t)：${num(lamEq, 5)} → ${num(lamCh, 5)}` +
        (lamRatio === undefined ? "" : `（${num(lamRatio, 3)} 倍）`) +
        (tMin !== undefined && vMin !== undefined ? `，但不是严格单调：全局最小在 t=${num(tMin, 3)} 处 ${num(vMin, 5)}，之后回升。` : "。"),
    );
  }
  if (monicMax !== undefined && monicTheory !== undefined) {
    lamLines.push(
      `|∏(x−x_i)| 最大值 ${num(monicMax, 4)}（理论极小 2^(1−n)=${num(monicTheory, 4)} 的 ${monicRatio === undefined ? "—" : num(monicRatio, 4)} 倍；t=0 时是 8.2 倍）。`,
    );
  }
  scene.texts.push(...textBlock(lamRect[0] + 8, lamRect[1] + 4, lamRect[1] + lamRect[3] - 3, lamLines, pal.note, lamBudget, "top"));

  // —— 角标 ——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(true), SLOT.axisMode, width));
  scene.badges.push(placeBadge(badge("autoscale", "误差面板：自适应 log 纵轴（误差、包络、|∏| 共用；自适应会掩盖量级变化）", 8, 8), SLOT.auxAxis, width));
  scene.badges.push(placeBadge(badge("autoscale", "Λ(t) 面板：自适应 log 纵轴（按本帧 21 个采样点取整）", 8, 8), SLOT.lamAxis, width));
  scene.badges.push(placeBadge(criticalBadge("n", n ?? Number.NaN, criticalOf(frame, "n", CRITICAL_N), { beyond: "above" }), SLOT.critical, width));
  const status: string[] = [];
  if (morph !== undefined && lamMorph !== undefined) status.push(`t=${num(morph, 3)}：Λ=${num(lamMorph, 4)}`);
  if (maxErr !== undefined) status.push(`最大误差 ${num(maxErr, 4)}`);
  if (status.length === 0) status.push("帧未给出可判读的误差（预期有 meta.morph / meta.max_error）");
  scene.badges.push(placeBadge(badge("status", status.join("；"), 8, 8), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  const nBad = countNonFinite(err) + countNonFinite(env) + countNonFinite(monic) + countNonFinite(p) + countNonFinite(lamT);
  applyOverflow(scene, nBad, { slot: SLOT.overflow, width });
  const metaOverflow = (metaNumber(frame, "overflow") ?? 0) > 0.5;
  if (metaOverflow && !scene.overflow.flag) {
    scene.overflow = { flag: true, note: "帧 meta.overflow=true：后端报告误差/包络越过对数轴上界（1e12），图上被裁到轴端" };
    scene.badges.push(placeBadge(badge("overflow", `溢出：${scene.overflow.note}`, 8, 44), SLOT.overflow, width));
  }
  return scene;
};

// --------------------------------------------------------------------------- //
// 局部工具
// --------------------------------------------------------------------------- //

/** 线性轴的取整范围（含有限值；一个都没有时用退化范围）。 */
function rangeLinear(arrays: readonly ArrayLike<number>[], fallback: [number, number]): [number, number] {
  let lo = Number.POSITIVE_INFINITY;
  let hi = Number.NEGATIVE_INFINITY;
  for (const a of arrays) {
    for (let i = 0; i < a.length; i += 1) {
      const v = a[i] as number;
      if (!Number.isFinite(v)) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return fallback;
  return niceLimits(lo, hi, 7, { pad: 0.08 });
}

/** 对数轴的取整范围（只取正值；一个正值都没有时用退化范围）。 */
function rangeLog(arrays: readonly ArrayLike<number>[], fallback: [number, number]): [number, number] {
  let lo = Number.POSITIVE_INFINITY;
  let hi = Number.NEGATIVE_INFINITY;
  for (const a of arrays) {
    const r = finitePositiveMinMax(a);
    if (r === undefined) continue;
    if (r[0] < lo) lo = r[0];
    if (r[1] > hi) hi = r[1];
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return fallback;
  return niceLimits(lo, hi, 7, { log: true, pad: 0.06 });
}

function valueAt(a: ArrayLike<number>, i: number): number | undefined {
  if (i < 0 || i >= a.length) return undefined;
  const v = a[i] as number;
  return Number.isFinite(v) ? v : undefined;
}

/** 最小有限值的下标；一个有限值都没有时 −1。 */
function argMin(a: ArrayLike<number>): number {
  let best = -1;
  let bestV = Number.POSITIVE_INFINITY;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i] as number;
    if (!Number.isFinite(v)) continue;
    if (v < bestV) {
      bestV = v;
      best = i;
    }
  }
  return best;
}
