/**
 * `barycentric_vs_vandermonde`（2.1 章）的 model：同一个 Vandermonde 系统上的**三条误差通道**。
 *
 * 面板与图层（顺序即绘制顺序，测试按此断言）
 * ------------------------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | line | `main` | `error_barycentric` / `error_vandermonde`（两条**函数值**通道） |
 * | 2 | line | `main` | `error_coefficients_vandermonde`（**系数**通道，真值是有理数精确解） |
 * | 3 | line | `main` | `kappa_bound_vandermonde` = κ(V)·u（系数误差的尺度参考线，虚线） |
 * | 4 | scatter | `main` | 当前 n 的四个标记（三条通道 + κu，取帧 meta） |
 * | 5 | loglogfit | `aux` | `kappa_vandermonde` + **实测斜率**与**理论斜率**两条参考线（规则 2） |
 * | 6 | line | `aux` | `kappa_theory` = 3^n（参考指数曲线，虚线） |
 * | 7 | line | `aux` | `residual_vandermonde`（后向残差，requiredAux 的 `residual`） |
 * | 8 | line | `slope` | `slope_kappa_measured` / `slope_kappa_theory`（log-log 斜率，实测 vs 理论） |
 * | 9 | line | `slope` | 当前 n 的竖直游标 + 两个斜率标记 |
 *
 * 时间轴语义
 * ----------
 * `mode="scrub"`、`tMax=36`：`t` 就是**节点数 n**（帧里写作 `meta.n_mark`）。
 * 所有 `*_n` 标量都在该下标处取值；不重算 κ、不重算误差。
 *
 * 防误导七条里本页的重点
 * ----------------------
 * * 规则 1：三条误差通道与 κu 参考线同屏（`requiredAux=["error","residual"]`，残差在 `aux`）；
 * * 规则 2：κ 面板给的是 `loglogfit` 的规范形态（实测斜率 + 理论斜率参考线 + 两条文字标注），
 *   第三个面板再把两个斜率逐点画出来；
 * * 规则 4：三个面板纵轴都自适应，各挂角标；正文写明"自适应会掩盖量级变化"。
 *
 * 两条最容易被误读的地方，以及本页怎么防
 * --------------------------------------
 * 1. **"κ(V) 涨到 1e14 = 公式不稳定"**：κ 是**系数基**里的放大因子，函数值误差的尺度是
 *    Λ_n·u。本页把函数值误差（两条）与系数误差（一条）画在**同一根 log 轴上**，
 *    再叠一条 κu 参考线：n=36 时函数值误差 1e-7、系数误差 1e0、κu≈28.6——谁跟谁同量级
 *    是一眼能看出来的；`aux` 面板里残差那条平线（1e-17 量级）与 κ 的指数上升同屏，
 *    "残差与 κ 无关"也一眼可见。
 * 2. **"残差小 ⇒ 解准"**：后向残差全程 ~1e-17，但它只说明后向稳定；前向误差还要乘 κu。
 *    正文直接把 `meta.coefficient_over_bound`（系数误差 / κu）写出来，
 *    免得把"最坏情况上界"当成"实际误差"。
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
  loglogFitLayer,
  makeAxis,
  scatterLayer,
  series,
  sourceBadge,
  type LineSeries,
} from "../../shared/src/scene.js";
import { axisTitleOf, axisYlabelOf, num, plain, rowLayout, textBlock, unitBudget } from "./scaffold.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "2.1";
const VIZ = "barycentric_vs_vandermonde";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "重心公式 vs 直接解 Vandermonde：三条误差通道";
/** manifest 的 `n_max.critical = [30]`；`frame.critical.n_max` 缺失时的回落值。 */
const CRITICAL_NMAX = 30;

const SLOT = {
  source: 0,
  axisMode: 1,
  auxAxis: 2,
  slopeAxis: 3,
  critical: 4,
  status: 5,
  overflow: 6,
  degraded: 7,
} as const;

export const barycentricVsVandermondeModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需数组 ——
  const nAxis = requireArray(frame, "n_axis");
  const errB = requireArray(frame, "error_barycentric");
  const errV = requireArray(frame, "error_vandermonde");
  const errC = requireArray(frame, "error_coefficients_vandermonde");
  const kap = requireArray(frame, "kappa_vandermonde");
  const kapBound = requireArray(frame, "kappa_bound_vandermonde");
  const kapTh = requireArray(frame, "kappa_theory");
  const resid = requireArray(frame, "residual_vandermonde");
  const slopeM = requireArray(frame, "slope_kappa_measured");
  const slopeT = requireArray(frame, "slope_kappa_theory");
  for (const [name, arr] of [
    ["error_barycentric", errB],
    ["error_vandermonde", errV],
    ["error_coefficients_vandermonde", errC],
    ["kappa_vandermonde", kap],
    ["kappa_bound_vandermonde", kapBound],
    ["kappa_theory", kapTh],
    ["residual_vandermonde", resid],
    ["slope_kappa_measured", slopeM],
    ["slope_kappa_theory", slopeT],
  ] as [string, ArrayLike<number>][]) {
    requireSameLength(nAxis, arr, `n_axis 与 ${name}`);
  }
  const pitfalls = requirePitfalls(frame);

  // —— 可选量（缺了降级，不猜）——
  const degraded: string[] = [];
  const metaOf = (key: string, label: string): number | undefined => {
    const v = metaNumber(frame, key);
    if (v === undefined) degraded.push(`帧 meta 缺 ${key}：${label}`);
    return v;
  };
  const nMax = metaOf("n_max", "标题不显示截断的 n_max");
  const nMark = metaOf("n_mark", "不画当前 n 的游标（不猜时间轴索引）");
  const kapN = metaOf("kappa_n", "不显示当前 n 的 κ（不重算）");
  const kapLog2 = metaOf("kappa_log2_per_n", "不显示实测 log2κ/n");
  const kapBase = metaOf("kappa_theory_base", "不显示参考指数曲线的底数");
  const kapRatio = metaOf("kappa_ratio_to_theory", "不显示实测 κ 与参考指数的比值");
  const kapBoundN = metaOf("kappa_bound_n", "不显示当前 n 的 κ·u");
  const errBN = metaOf("error_barycentric_n", "不显示当前 n 的重心函数值误差");
  const errVN = metaOf("error_vandermonde_n", "不显示当前 n 的 Vandermonde 函数值误差");
  const errCN = metaOf("error_coefficients_n", "不显示当前 n 的系数误差");
  const coefOverBound = metaOf("coefficient_over_bound", "不显示系数误差 / κu");
  const coefOverValue = metaOf("coefficient_over_value_error", "不显示系数误差 / 函数值误差");
  const residN = metaOf("residual_n", "不显示当前 n 的后向残差");
  const slopeMN = metaOf("slope_kappa_measured_n", "不显示当前 n 的实测斜率");
  const slopeTN = metaOf("slope_kappa_theory_n", "不显示当前 n 的理论斜率");
  const nClampedC = metaOf("n_coefficient_error_clamped", "不显示系数通道被夹取的点数");
  const nZeroResid = metaOf("n_exact_zero_residual", "不显示残差恰好为 0 的点数");
  const saturated = (metaNumber(frame, "kappa_saturated") ?? 0) > 0.5;
  const isChebyshev = (metaNumber(frame, "is_chebyshev") ?? 0) > 0.5;
  const family = plain(stringOf(frame, "family"));
  const funcLabel = plain(stringOf(frame, "function_label"));
  const note = plain(stringOf(frame, "note"));
  const tMeta = metaNumber(frame, "t");
  const tNow = tMeta ?? opts.t;
  if (tMeta === undefined) degraded.push("帧 meta 缺 t：scene.t 改用请求 t（两者可能不同）");
  if (nMark !== undefined && tMeta !== undefined && Math.abs(tMeta - nMark) > 0.5) {
    degraded.push(`请求 t=${num(tMeta, 4)} 与帧 meta.n_mark=${num(nMark, 4)} 不一致：按帧里的 n_mark 画`);
  }

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: tNow,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  // —— 排版：三条通道（log）/ κ 与残差（log-log）/ 斜率（线性）——
  // footer=160：pitfalls 实测 7 行（本章最长）→ 面板底 440，与 pitfalls 框留 6px。
  const layout = rowLayout(width, height, [1.4, 0.95, 0.72], { gap: 36, footer: 160 });
  const [mainRect, auxRect, slopeRect] = layout.rects as [
    [number, number, number, number],
    [number, number, number, number],
    [number, number, number, number],
  ];

  const nLo = minOf(nAxis, 1);
  const nHi = Math.max(maxOf(nAxis, nLo + 1), nLo + 1);
  const xLin: [number, number] = [nLo - 0.5, nHi + 0.5];
  const mainAxis = makeAxis("main", mainRect, {
    xlim: xLin,
    ylim: rangeLog([errB, errV, errC, kapBound], [1e-18, 1e3]),
    yscale: "log",
    xlabel: "",
    ylabel: axisYlabelOf(SPEC, "main", "误差（log）"),
    title: axisTitleOf(SPEC, "main", "三条误差通道（log）：函数值 ×2 + 系数 ×1，叠 κ(V)·u"),
    autoscaled: true,
  });
  const auxAxis = makeAxis("aux", auxRect, {
    xlim: [nLo, nHi * 1.06],
    xscale: "log",
    ylim: rangeLog([kap, kapTh, resid], [1e-18, 1e16]),
    yscale: "log",
    xlabel: "",
    ylabel: axisYlabelOf(SPEC, "aux", "κ₁(V) 与残差（log）"),
    title: axisTitleOf(SPEC, "aux", "κ₁(V) 与参考指数曲线、后向残差（log-log）"),
    autoscaled: true,
  });
  const slopeAxis = makeAxis("slope", slopeRect, {
    xlim: xLin,
    ylim: rangeLinear([slopeM, slopeT], [0, 40]),
    xlabel: "节点数 $n$（本行线性；上一行同轴但用对数刻度）",
    ylabel: "$d\\log\\kappa/d\\log n$",
    title: "log-log 斜率：实测 vs 理论（参考指数曲线只是量级参考）",
    autoscaled: true,
  });
  scene.axes.push(mainAxis, auxAxis, slopeAxis);

  // —— 图层 1~2：三条误差通道（`segmentedSeries` 的 accept 把 0/负值挡在 log 轴之外）——
  const mainSeries: LineSeries[] = [
    ...segmentedSeries(nAxis, errB, pal.curves[0], { width: 1.8, accept: (v) => v > 0 }),
    ...segmentedSeries(nAxis, errV, pal.curves[1], { width: 1.6, dash: [7, 3], accept: (v) => v > 0 }),
  ];
  // —— 图层 2b：系数通道单独一条（真值是有理数精确解）——
  mainSeries.push(...segmentedSeries(nAxis, errC, pal.exact, { width: 1.8, marker: "o", accept: (v) => v > 0 }));
  if (mainSeries.length > 0) scene.layers.push(lineLayer("main", mainSeries));
  else degraded.push("三条误差通道数组里没有 >0 的有限值：log 轴上一段曲线都画不出来");

  // —— 图层 3：κ(V)·u 参考线（系数误差的尺度；它才是"系数通道"该比的对象）——
  const boundSeries = segmentedSeries(nAxis, kapBound, pal.reference, { width: 1.2, dash: [2, 4], accept: (v) => v > 0 });
  if (boundSeries.length > 0) scene.layers.push(lineLayer("main", boundSeries));

  // —— 图层 4：当前 n 的标记（帧 meta 的四个标量）——
  const markN = frame.has("mark_n") ? requireArray(frame, "mark_n") : undefined;
  const nNow = markN !== undefined && markN.length > 0 ? (markN[0] as number) : nMark;
  const markXs: number[] = [];
  const markYs: number[] = [];
  for (const v of [errBN, errVN, errCN, kapBoundN]) {
    if (v !== undefined && v > 0 && nNow !== undefined) {
      markXs.push(nNow);
      markYs.push(v);
    }
  }
  if (markXs.length > 0) {
    scene.layers.push(scatterLayer("main", markXs, markYs, { r: 3.4, color: pal.marker, label: "当前 $n$ 的三条通道 + $\\kappa u$" }));
  }

  // —— 图层 5：规则 2 的规范形态——κ 的实测曲线 + 实测斜率 + 理论斜率参考线 ——
  const slopeMe = slopeMN ?? lastPositive(slopeM);
  const slopeTh = slopeTN ?? lastPositive(slopeT);
  scene.layers.push(
    loglogFitLayer("aux", nAxis, kap, slopeMe, slopeTh, {
      seriesLabel: `实测斜率 ${num(slopeMe, 4)}（log-log）`,
      theoryLabel: `理论斜率 ${num(slopeTh, 4)}（参考指数 ${num(kapBase ?? 3, 3)}^n）`,
      color: pal.curves[0],
      theoryColor: pal.marker,
    }),
  );
  if (slopeMN === undefined || slopeTN === undefined) {
    degraded.push("帧 meta 缺 slope_kappa_*_n：loglogfit 的斜率改用帧数组 slope_kappa_* 的末值（与理论线锚点同为 n_max，不重算）");
  }

  // —— 图层 6~7：参考指数曲线 + 后向残差（requiredAux 的 residual）——
  const auxSeries: LineSeries[] = [
    ...segmentedSeries(nAxis, kapTh, pal.marker, { width: 1.3, dash: [5, 3], accept: (v) => v > 0 }),
    ...segmentedSeries(nAxis, resid, pal.residual, { width: 1.8, accept: (v) => v > 0 }),
  ];
  if (auxSeries.length > 0) scene.layers.push(lineLayer("aux", auxSeries));
  else degraded.push("kappa_theory 与 residual_vandermonde 里没有 >0 的有限值：κ 面板只剩实测曲线");
  if (nNow !== undefined) {
    scene.layers.push(lineLayer("aux", series([nNow, nNow], [auxAxis.ylim[0], auxAxis.ylim[1]], pal.cursor, { width: 1, dash: [3, 3] })));
    const yMark: number[] = [];
    for (const v of [kapN, residN]) if (v !== undefined && v > 0) yMark.push(v);
    if (yMark.length > 0) scene.layers.push(scatterLayer("aux", yMark.map(() => nNow), yMark, { r: 3.2, color: pal.marker, label: "当前 $n$ 的 $\\kappa$ 与残差" }));
  }

  // —— 图层 8~9：斜率面板（实测 vs 理论逐点对照）——
  const slopeSeries: LineSeries[] = [
    ...segmentedSeries(nAxis, slopeM, pal.curves[0], { width: 1.8, accept: (v) => v !== 0 }),
    ...segmentedSeries(nAxis, slopeT, pal.marker, { width: 1.3, dash: [5, 3], accept: (v) => v !== 0 }),
  ];
  if (slopeSeries.length > 0) scene.layers.push(lineLayer("slope", slopeSeries));
  else degraded.push("两条斜率数组里没有有限值：斜率面板画不出曲线");
  if (nNow !== undefined) {
    scene.layers.push(lineLayer("slope", series([nNow, nNow], [slopeAxis.ylim[0], slopeAxis.ylim[1]], pal.cursor, { width: 1, dash: [3, 3] })));
    const yMark: number[] = [];
    for (const v of [slopeMN, slopeTN]) if (v !== undefined) yMark.push(v);
    if (yMark.length > 0) scene.layers.push(scatterLayer("slope", yMark.map(() => nNow), yMark, { r: 3.2, color: pal.marker, label: "当前 $n$ 的两个斜率" }));
  }

  // —— 文本 ——
  const summary = paramSummary(frame, opts, [
    { name: "n_max", label: "`n_max`" },
    { name: "distribution", label: "节点分布" },
  ]);
  degraded.push(...summary.degraded);
  const familyNote = `${family ?? (isChebyshev ? "Chebyshev 节点" : "等距节点")}　${funcLabel ?? ""}`;
  const mainBudget = unitBudget(mainRect[2]);
  const mainLines: string[] = ["蓝=重心　绿虚线=直接解　橙○=系数误差　灰虚线=κ(V)·u 参考线"];
  if (nNow !== undefined) mainLines.push(`当前 n=${num(nNow, 4)}　${familyNote}　${summary.text}`);
  else mainLines.push(`${familyNote}　${summary.text}`);
  if (errBN !== undefined && errVN !== undefined) {
    mainLines.push(`函数值误差：重心 ${num(errBN, 4)} vs 直接解 ${num(errVN, 4)}——同一个量级（都 ≈ Λ_n·u）。`);
  }
  if (errCN !== undefined) {
    mainLines.push(
      `系数误差（橙○）：${num(errCN, 4)}` +
        (kapBoundN === undefined ? "" : `，κu=${num(kapBoundN, 4)}` ) +
        (coefOverBound === undefined ? "" : `（占上界 ${num(coefOverBound, 3)}）`) +
        (coefOverValue === undefined ? "" : `，比函数值误差大 ${num(coefOverValue, 3)} 倍。`) +
        (nClampedC !== undefined && nClampedC > 0 ? `其中 ${num(nClampedC, 3)} 点被夹到对数地板；` : "") +
        "系数烂掉不影响 [−1,1] 上的取值。",
    );
  }
  if (note !== undefined) mainLines.push(`帧 note：${note}`);
  scene.texts.push(...textBlock(mainRect[0] + 8, mainRect[1] + 4, mainRect[1] + mainRect[3] - 3, mainLines, pal.ink, mainBudget, "top"));

  // κ 面板右上角被 `loglogfit` 的两条标注占着，正文只占左半边。
  const auxBudget = Math.max(30, Math.floor(unitBudget(auxRect[2]) * 0.6));
  // κ 面板只有 4 行：颜色写进数值行本身，省掉单独一行图例。
  const auxLines: string[] = [];
  if (kapN !== undefined) {
    auxLines.push(
      `κ₁(V)（蓝）=${num(kapN, 4)}` +
        (kapRatio === undefined ? "" : `　= 红虚线 ${num(kapBase ?? 3, 3)}^n 的 ${num(kapRatio, 3)} 倍`) +
        (kapLog2 === undefined ? "" : `　log2κ/n=${num(kapLog2, 4)}`),
    );
  }
  if (residN !== undefined) {
    auxLines.push(
      `后向残差（紫）全程 1e-17 量级（当前 ${num(residN, 3)}）` +
        (nZeroResid !== undefined && nZeroResid > 0 ? `，n=1、2 处恰为 0：log 轴画不出，曲线从 n=3 起` : "") +
        "。",
    );
  }
  auxLines.push("κ 大 ≠ 公式不稳：函数值误差的尺度是 Λ_n·u，残差小只说明后向稳定。");
  scene.texts.push(...textBlock(auxRect[0] + 8, auxRect[1] + 4, auxRect[1] + auxRect[3] - 3, auxLines, pal.muted, auxBudget, "top"));

  const slopeBudget = unitBudget(slopeRect[2]);
  const slopeLines: string[] = ["蓝=实测斜率　红虚线=理论斜率　灰竖线=当前 n"];
  if (slopeMN !== undefined && slopeTN !== undefined) {
    slopeLines.push(`当前 n：实测 ${num(slopeMN, 5)} vs 理论 ${num(slopeTN, 5)}（log-log 斜率，两者相差 ${num(Math.abs(slopeMN - slopeTN), 3)}）。`);
  }
  slopeLines.push(
    "参考指数只是量级参考：等距 log2κ/n 从 1.53 漂到 1.60，Chebyshev 稳定在 1.261" +
      (isChebyshev ? "（1+√2≈2.414 的底数）" : "") +
      "。",
  );
  scene.texts.push(...textBlock(slopeRect[0] + 8, slopeRect[1] + 4, slopeRect[1] + slopeRect[3] - 3, slopeLines, pal.note, slopeBudget, "top"));

  // —— 角标 ——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(true), SLOT.axisMode, width));
  scene.badges.push(placeBadge(badge("autoscale", "误差面板：自适应 log 纵轴（三条通道 + κu 共用；自适应会掩盖量级变化）", 8, 8), SLOT.auxAxis, width));
  scene.badges.push(placeBadge(badge("autoscale", "κ 面板：log-log 自适应（κ 与残差共用，跨 30 个数量级）", 8, 8), SLOT.slopeAxis, width));
  scene.badges.push(placeBadge(criticalBadge("n_max", nMax ?? Number.NaN, criticalOf(frame, "n_max", CRITICAL_NMAX), { beyond: "above" }), SLOT.critical, width));
  const status: string[] = [];
  if (nNow !== undefined && errBN !== undefined && errVN !== undefined) {
    status.push(`n=${num(nNow, 4)}：函数值 ${num(errBN, 4)} vs ${num(errVN, 4)}（同量级），系数 ${errCN === undefined ? "—" : num(errCN, 4)}`);
  }
  if (kapN !== undefined) status.push(`κ=${num(kapN, 4)}`);
  if (residN !== undefined) status.push(`残差 ${num(residN, 3)}`);
  if (saturated) status.push("⚠ 帧标记 kappa_saturated：κ 已经超出双精度可分辨范围");
  if (status.length === 0) status.push("帧未给出可判读的误差（预期有 meta.n_mark / meta.error_barycentric_n）");
  scene.badges.push(placeBadge(badge("status", status.join("；"), 8, 8), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  const nBad = countNonFinite(errB) + countNonFinite(errV) + countNonFinite(errC) + countNonFinite(kap) + countNonFinite(resid) + countNonFinite(slopeM);
  applyOverflow(scene, nBad, { slot: SLOT.overflow, width });
  const metaOverflow = (metaNumber(frame, "overflow") ?? 0) > 0.5;
  if (metaOverflow && !scene.overflow.flag) {
    scene.overflow = { flag: true, note: "帧 meta.overflow=true：后端报告有曲线越过对数轴上界，图上被裁到轴端" };
    scene.badges.push(placeBadge(badge("overflow", `溢出：${scene.overflow.note}`, 8, 44), SLOT.overflow, width));
  }
  return scene;
};

// --------------------------------------------------------------------------- //
// 局部工具
// --------------------------------------------------------------------------- //

function minOf(a: ArrayLike<number>, fallback: number): number {
  let lo = Number.POSITIVE_INFINITY;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i] as number;
    if (Number.isFinite(v) && v < lo) lo = v;
  }
  return Number.isFinite(lo) ? lo : fallback;
}

function maxOf(a: ArrayLike<number>, fallback: number): number {
  let hi = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i] as number;
    if (Number.isFinite(v) && v > hi) hi = v;
  }
  return Number.isFinite(hi) ? hi : fallback;
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

/** 线性轴的取整范围。 */
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

/** 数组里最后一个正有限值（`meta.slope_*_n` 缺失时的显示回落；锚点与 loglogfit 的末点一致）。 */
function lastPositive(y: ArrayLike<number>): number {
  for (let i = y.length - 1; i >= 0; i -= 1) {
    const v = y[i] as number;
    if (Number.isFinite(v) && v > 0) return v;
  }
  return 0;
}
