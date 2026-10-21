/**
 * `gauss_vs_equispaced`（2.4 章）的 model：Gauss–Legendre 与等距 Newton–Cotes。
 *
 * 三个面板（顺序即绘制顺序，测试按此断言）
 * ----------------------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | line | `nodes` | 权重 = 0 基准线（n≥9 时等距权重有负值，没有它读不出符号） |
 * | 2 | scatter | `nodes` | Gauss 节点 → 权重（节点在端点加密、权重全正） |
 * | 3 | scatter | `nodes` | 等距节点 → 权重（m≥9 起出现负值） |
 * | 4 | line | `exact` | Gauss 的矩误差 `error_moment_gauss`（已揭示到第 k 次） |
 * | 5 | line | `exact` | 等距的矩误差 `error_moment_equi`（同一窗口） |
 * | 6 | line | `exact` | 精确度边界 `k = 2n−1` 竖线（`meta.degree_exact_limit`） |
 * | 7 | line | `cond` | 等距的 Γ=Σ\|w\|/Σw（`cond_equi`） |
 * | 8 | line | `cond` | Gauss 的 Γ≡1（`cond_gauss`，与它同屏才有"两个方向"） |
 * | 9 | line | `cond` | 首次出现负权重的点数 `m = 9` 竖线（`frame.critical.n`） |
 * | 10 | line | `cond` | 当前 `n` 的游标竖线（把参数与曲线接起来） |
 *
 * 角标槽位（固定，见 `common.ts::SLOT`）：0 来源 / 1 主轴固定 / 2 副轴自适应 /
 * 3 临界（`n=9`，只对等距成立）/ 4 状态 / 5 前端预览 / 6 溢出 / 7 降级。
 *
 * 这一页最容易骗人的两处，以及它们在图上的防法
 * --------------------------------------------
 * 1. **「Gauss 很准」没有上界**。`k ≤ 2n−1` 精确、`k = 2n` 当场失效——两个方向都必须画：
 *    左下面板把两条矩误差曲线**一起**揭示（竖线标出 `2n−1`），并把 `error_at_limit`、
 *    `error_beyond_limit`、失效因子写进文字。**读图陷阱（本模型额外挡住的一条）**：
 *    等距曲线是**锯齿**（偶数 k 失败、奇数 k 落在舍入地板），那不是"等距有时精确"，
 *    而是奇数次上 `∫x^k=0` 与对称节点上的求和**同时为 0** 的平凡精确——文字里点名。
 * 2. **「Γ 大 = 不准确」要配放大实验**。`cond` 面板把等距的 Γ 与 Gauss 的 Γ≡1 同屏，
 *    并把"同一 ε 的相对扰动 → 积分相对误差 ≈ Γ·ε"这条推论用**帧里的 Γ 值**算出来
 *    （当前 n 与 m=19 的 175.46 各给一个数），而不是只写一句"条件数大"。
 *
 * 几条刻意的决定
 * --------------
 * 1. **矩误差取标准区间 [−1,1]**：`strings.exactness_note` 说明了为什么（在 [0,1] 上
 *    `x^{2n}` 的系数会把 k=2n 的失效证据压进地板，实测 n=20 时因子只有 0.52）。
 * 2. **横轴是次数 k 的整数刻度**（线性），纵轴是 log 的矩误差：阶梯状跳变在 log 轴上
 *    才是"跳了多少个数量级"，而不是"高了几个像素"。
 * 3. **时间轴 = 已测试到的次数 k**：曲线按 `meta.step` 截断，于是"拖时间轴"就是
 *    "逐次加大多项式次数"，`k=2n` 那一步的跳变会**当场**出现在图上。
 * 4. **`m=10` 是反例**（权重全正、Γ=1.000）：`critical` 竖线画在 `m=9`，
 *    文字里点名"判据是 Γ 不是 m"，免得读者把这条线当成"m≥9 一律完蛋"。
 */

import { niceLimits, niceTicks } from "../../shared/src/axes.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import {
  applyOverflow,
  canvasSize,
  columnsLayout,
  criticalOf,
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
  segmentedSeries,
  toNumeric,
  type SceneModel,
} from "../../shared/src/model.js";
import {
  axisModeBadge,
  badge,
  criticalBadge,
  lineLayer,
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
  head,
  minMaxOfAll,
  plainNote,
  noteOf,
  paramSpec,
  positiveMinMaxOfAll,
  sci,
  specTitle,
  textBlock,
  unitBudget,
  withZero,
} from "./common.js";

const VIZ = "gauss_vs_equispaced";
const TITLE = specTitle(VIZ, "Gauss 节点 vs 等距节点：精确度的两个方向");

export const gaussVsEquispacedModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需数组 ——
  const nodeGauss = requireArray(frame, "node_gauss");
  const weightGauss = requireArray(frame, "weight_gauss");
  const nodeEqui = requireArray(frame, "node_equi");
  const weightEqui = requireArray(frame, "weight_equi");
  const degree = requireArray(frame, "degree");
  const errGauss = requireArray(frame, "error_moment_gauss");
  const errEqui = requireArray(frame, "error_moment_equi");
  const mGrid = requireArray(frame, "m_grid");
  const condEqui = requireArray(frame, "cond_equi");
  const condGauss = requireArray(frame, "cond_gauss");
  requireSameLength(nodeGauss, weightGauss, "帧数组 node_gauss 与 weight_gauss");
  requireSameLength(nodeEqui, weightEqui, "帧数组 node_equi 与 weight_equi");
  requireSameLength(degree, errGauss, "帧数组 degree 与 error_moment_gauss");
  requireSameLength(degree, errEqui, "帧数组 degree 与 error_moment_equi");
  requireSameLength(mGrid, condEqui, "帧数组 m_grid 与 cond_equi");
  requireSameLength(mGrid, condGauss, "帧数组 m_grid 与 cond_gauss");
  const pitfalls = requirePitfalls(frame);

  const degraded: string[] = [];
  const n = metaNumber(frame, "n");
  const limit = metaNumber(frame, "degree_exact_limit") ?? (n === undefined ? undefined : 2 * n - 1);
  if (n === undefined) degraded.push("帧 meta 缺 n：节点数留「?」（不按数组长度反推）");
  if (limit === undefined) degraded.push("帧 meta 缺 degree_exact_limit：2n−1 竖线画不出来（文字里保留该边界）");
  const a = metaNumber(frame, "interval_a");
  const b = metaNumber(frame, "interval_b");
  const errFloor = metaNumber(frame, "error_floor");
  const errAtLimit = metaNumber(frame, "error_at_limit");
  const errBeyond = metaNumber(frame, "error_beyond_limit");
  const failureFactor = metaNumber(frame, "failure_factor");
  const errEquiAtLimit = metaNumber(frame, "error_equi_at_limit");
  const errEquiBeyond = metaNumber(frame, "error_equi_beyond");
  const gammaEquiN = metaNumber(frame, "gamma_equi_n");
  const gammaGauss = metaNumber(frame, "gamma_gauss");
  const sumWGauss = metaNumber(frame, "sum_w_gauss");
  const sumWEqui = metaNumber(frame, "sum_w_equi");
  const negEqui = metaNumber(frame, "n_negative_equi");
  const negGauss = metaNumber(frame, "n_negative_gauss");
  const minWGauss = metaNumber(frame, "min_weight_gauss");
  const minWEqui = metaNumber(frame, "min_weight_equi");
  const gammaM9 = metaNumber(frame, "knots_m9_gamma");
  const gammaM19 = metaNumber(frame, "knots_m19_gamma");
  const rateGauss = metaNumber(frame, "conv_rate_gauss");
  const orderEqui = metaNumber(frame, "conv_order_equi");
  const clipped = metaNumber(frame, "clipped_points");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: metaNumber(frame, "t") ?? opts.t,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  // —— 排版：左列上下两栏（节点权重 / 矩误差），右列通高（条件数 Γ）——
  // 左列上下两栏：上面板 46% / 间距 38（下面板要放 6 行说明，50px 间距会把它的高度压到 8 行以内）。
  const base = columnsLayout(width, height, { share: 0.56, columnGap: 64, mainShare: 0.40, gap: 32 });

  // 时间轴 = 已测试到的多项式次数 k（0 … 2n+2）。
  const kMax = degree.length > 0 ? (degree[degree.length - 1] as number) : 0;
  const step = currentStep(frame, opts, { lo: 0, hi: Math.max(0, Math.round(kMax)), label: "多项式次数 $k$" });
  const revealed = Math.max(1, Math.round(step.step ?? 0) + 1);

  /* ------------------------------ 左上：节点与权重 ------------------------------ */
  const xLo = a ?? (minMaxOfAll([nodeGauss, nodeEqui])?.[0] ?? -1);
  const xHi = b ?? (minMaxOfAll([nodeGauss, nodeEqui])?.[1] ?? 1);
  const wRange = withZero(minMaxOfAll([weightGauss, weightEqui]));
  const nodesAxis = makeAxis("nodes", base.main, {
    // 右端留 22%：节点散点/零线的 label 画在**最后一个可见点右侧**，不留白会被裁掉。
    xlim: niceLimits(xLo, xLo + (xHi - xLo) * 1.22, 6),
    ylim: niceLimits(wRange[0], wRange[1] + 0.05 * (wRange[1] - wRange[0] || 1), 6),
    xlabel: "节点位置 $x$",
    ylabel: axisYlabel(VIZ, "nodes", "w_k"),
    title: axisTitle(VIZ, "nodes", "节点与权重"),
    autoscaled: false,
  });
  scene.axes.push(nodesAxis);
  scene.layers.push(
    lineLayer(
      "nodes",
      // 参考线只画 72% 宽：label 落在最后可见点右侧，画满整轴会被右边界裁掉。
      series([nodesAxis.xlim[0], nodesAxis.xlim[0] + (nodesAxis.xlim[1] - nodesAxis.xlim[0]) * 0.72], [0, 0], pal.reference, {
        width: 1,
        dash: [5, 4],
        label: "$w=0$",
      }),
    ),
  );
  scene.layers.push(
    scatterLayer("nodes", nodeGauss, weightGauss, { r: 3.6, color: pal.curves[0], label: "Gauss" }),
  );
  scene.layers.push(scatterLayer("nodes", nodeEqui, weightEqui, { r: 3.6, color: pal.exact, label: "等距" }));

  const nodesLines: string[] = [];
  if (n !== undefined) nodesLines.push(`n = ${fmt(n, 4)} 点公式（${fmtOpt(a, 3)} … ${fmtOpt(b, 3)}）：Gauss 端点加密、权重全正`);
  nodesLines.push(`Σw：Gauss=${fmtOpt(sumWGauss, 8)}、等距=${fmtOpt(sumWEqui, 8)}（≈b−a，只说明常数被积对）`);
  if (a === undefined || b === undefined || Math.abs(a + 1) > 1e-9 || Math.abs(b - 1) > 1e-9) {
    nodesLines.push(`⚠ 权重在 [${fmtOpt(a, 3)}, ${fmtOpt(b, 3)}]、矩误差在 [−1,1]：区间不同是刻意的`);
  }
  nodesLines.push(
    `最小权重：Gauss=${fmtOpt(minWGauss, 4)}、等距=${fmtOpt(minWEqui, 4)}；负权重 ${fmtOpt(negGauss, 3)} / ${fmtOpt(negEqui, 3)}`,
  );
  if (negEqui !== undefined && negEqui > 0) {
    nodesLines.push(`⚠ 等距权重已有 ${fmt(negEqui, 3)} 个负值：Σ|w|>Σw，条件数 Γ 离开 1（见右列）`);
  }
  scene.texts.push(
    ...textBlock(nodesAxis.rect[0] + 6, nodesAxis.rect[1] + 4, nodesAxis.rect[1] + nodesAxis.rect[3] - 4, nodesLines, pal.note, unitBudget(Math.round(nodesAxis.rect[2] * 0.8)), "top"),
  );

  /* ------------------------------ 左下：矩误差 ------------------------------ */
  const exactAxis = makeAxis("exact", base.error, {
    xlim: [degree.length > 0 ? (degree[0] as number) : 0, kMax * 1.18],
    yscale: "log",
    xlabel: "多项式次数 $k$",
    ylabel: axisYlabel(VIZ, "exact", "误差"),
    title: axisTitle(VIZ, "exact", "矩误差 |Q(x^k)−∫x^k|"),
    autoscaled: true,
  });
  const revG = head(errGauss, revealed);
  const revE = head(errEqui, revealed);
  const yPool = positiveMinMaxOfAll([revG, revE, errFloor !== undefined && errFloor > 0 ? [errFloor] : undefined]) ?? [1e-16, 1];
  exactAxis.ylim = niceLimits(yPool[0], yPool[1], 6, { log: true, pad: 0.03 });
  exactAxis.yticks = niceTicks(exactAxis.ylim[0], exactAxis.ylim[1], 4, { log: true });
  exactAxis.xticks = niceTicks(exactAxis.xlim[0], exactAxis.xlim[1], 6);
  scene.axes.push(exactAxis);
  const degreeRevealed = head(degree, revealed);
  scene.layers.push(
    lineLayer("exact", segmentedSeries(degreeRevealed, revG, pal.curves[0], { width: 1.9, marker: "o", accept: (v) => v > 0, label: "Gauss" })),
  );
  scene.layers.push(
    lineLayer("exact", segmentedSeries(degreeRevealed, revE, pal.exact, { width: 1.9, marker: "o", accept: (v) => v > 0, label: "等距" })),
  );
  if (limit !== undefined && limit >= exactAxis.xlim[0] && limit <= exactAxis.xlim[1]) {
    scene.layers.push(
      lineLayer("exact", series([limit, limit], exactAxis.ylim, pal.warn, { width: 1.4, dash: [5, 4], label: "$k=2n$−1" })),
    );
  }
  const exactLines: string[] = [];
  if (limit !== undefined) {
    exactLines.push(
      `Gauss 精确到 k ≤ 2n−1 = ${fmt(limit, 4)}（帧实测 ${fmtOpt(errAtLimit, 4)}）；` +
        `k = 2n 失效 ${fmtOpt(errBeyond, 4)}（因子 ${fmtOpt(failureFactor, 4)}）`,
    );
  }
  exactLines.push(
    `等距：k=2n−1 处 ${fmtOpt(errEquiAtLimit, 4)}、k=2n 处 ${fmtOpt(errEquiBeyond, 4)}（锯齿从 k=n 起）`,
  );
  exactLines.push("⚠ 奇数次的精确是平凡的（∫x^k=0 与对称求和同时为 0）");
  pushIf(exactLines, plainNote(noteOf(frame, "exactness_note")));
  exactLines.push(
    `已揭示到 k = ${fmt((step.step ?? 0), 4)}（时间轴 = 加大多项式次数）；地板 ${sci(errFloor ?? Number.NaN)}、截断 ${fmtOpt(clipped, 3)}`,
  );
  exactLines.push(...step.notes);
  scene.texts.push(
    ...textBlock(exactAxis.rect[0] + 6, exactAxis.rect[1] + 4, exactAxis.rect[1] + exactAxis.rect[3] - 4, exactLines, pal.note, unitBudget(exactAxis.rect[2]), "top"),
  );

  /* ------------------------------ 右列：条件数 Γ ------------------------------ */
  const mRange = minMaxOfAll([mGrid]) ?? [2, 20];
  const condAxis = makeAxis("cond", base.side, {
    xlim: [mRange[0], mRange[0] + (mRange[1] - mRange[0]) * 1.18],
    yscale: "log",
    xlabel: "点数 $m$",
    ylabel: axisYlabel(VIZ, "cond", "Γ"),
    title: axisTitle(VIZ, "cond", "条件数 Γ=Σ|w|/Σw"),
    autoscaled: true,
  });
  const gRange = positiveMinMaxOfAll([condEqui, condGauss]) ?? [1, 10];
  condAxis.ylim = niceLimits(gRange[0], gRange[1], 6, { log: true, pad: 0.02 });
  condAxis.yticks = niceTicks(condAxis.ylim[0], condAxis.ylim[1], 5, { log: true });
  condAxis.xticks = niceTicks(condAxis.xlim[0], condAxis.xlim[1], 6);
  scene.axes.push(condAxis);
  scene.layers.push(
    lineLayer("cond", segmentedSeries(mGrid, condEqui, pal.exact, { width: 1.9, marker: "o", accept: (v) => v > 0, label: "等距" })),
  );
  scene.layers.push(
    lineLayer("cond", segmentedSeries(mGrid, condGauss, pal.curves[0], { width: 1.9, accept: (v) => v > 0, label: "Gauss" })),
  );
  const critical = criticalOf(frame, "n", paramSpec(VIZ, "n")?.critical[0] ?? 9);
  const critValue = Array.isArray(critical) ? (critical[0] as number) : critical;
  if (Number.isFinite(critValue) && critValue >= condAxis.xlim[0] && critValue <= condAxis.xlim[1]) {
    scene.layers.push(
      lineLayer("cond", series([critValue, critValue], condAxis.ylim, pal.warn, { width: 1.4, dash: [5, 4], label: `m = ${fmt(critValue, 4)}` })),
    );
  }
  if (n !== undefined && n >= condAxis.xlim[0] && n <= condAxis.xlim[1]) {
    scene.layers.push(
      lineLayer("cond", series([n, n], condAxis.ylim, pal.marker, { width: 1.2, dash: [4, 3], label: `n=${fmt(n, 4)}` })),
    );
  }
  const eps = 1e-8;
  const condLines: string[] = [];
  condLines.push("Γ = 函数值相对误差 → 积分相对误差的放大倍数（纵坐标就是放大倍数）");
  if (gammaGauss !== undefined && gammaEquiN !== undefined) {
    condLines.push(`当前 n：Gauss Γ=${fmt(gammaGauss, 6)}、等距 Γ=${fmt(gammaEquiN, 6)}`);
  }
  if (gammaM9 !== undefined && gammaM19 !== undefined) {
    condLines.push(`等距 m=9：Γ=${fmt(gammaM9, 6)}；m=19：Γ=${fmt(gammaM19, 6)}`);
    condLines.push(
      `同一 ε=${sci(eps)} 的扰动 → 积分相对误差：等距 m=19 放大 ${sci(gammaM19 * eps)}、` +
        `Gauss 只放大 ${sci((gammaGauss ?? 1) * eps)}（${fmt(gammaM19 / (gammaGauss ?? 1), 4)}×）`,
    );
  }
  if (gammaM9 !== undefined && gammaM19 !== undefined) {
    condLines.push("Γ 也带锯齿：m=10 与 m=20 明显低于相邻的奇数 m（偶数 m 的 Σ|w| 更小）");
  }
  condLines.push("⚠ 判据是 Γ 不是 m：m=10 权重全正、Γ=1.000 是反例（竖线只对等距成立）");
  pushIf(condLines, plainNote(noteOf(frame, "condition_note")));
  if (orderEqui !== undefined) {
    condLines.push(`同预算：Gauss 每节点速率 ${fmtOpt(rateGauss, 4)}（几何）、等距梯形阶 ${fmt(Math.abs(orderEqui), 4)}（代数）`);
  }
  scene.texts.push(
    ...textBlock(condAxis.rect[0] + 6, condAxis.rect[1] + 4, condAxis.rect[1] + condAxis.rect[3] - 4, condLines, pal.note, unitBudget(Math.round(condAxis.rect[2] * 0.8)), "top"),
  );

  /* ------------------------------ 角标 ------------------------------ */
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(badge("autoscale", "副轴 exact/cond：自适应（log 纵轴按本帧数据取整；自适应会掩盖量级变化）", 8, 26), SLOT.auxAxis, width),
  );
  scene.badges.push(
    placeBadge(
      n === undefined
        ? badge("critical", "帧 meta 缺 n：无法与临界值比较（留「?」，不猜）", 8, 62)
        : criticalBadge("n", n, critical, { beyond: "above" }),
      SLOT.critical,
      width,
    ),
  );
  const status: string[] = [];
  if (negEqui !== undefined && negEqui > 0) status.push(`⚠ 等距规则出现 ${fmt(negEqui, 3)} 个负权重（Γ=${fmtOpt(gammaEquiN, 4)}）`);
  else status.push(`等距规则权重全正（Γ=${fmtOpt(gammaEquiN, 4)}）：m 还没到 9`);
  if (revealed <= 1) status.push("时间轴在第 0 次：只测到 k=0，2n 处的失效还没揭示");
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
/* 局部工具                                                                      */
/* --------------------------------------------------------------------------- */

/** 条件式入队（`undefined` 不入队）。 */
function pushIf(lines: string[], body: string | undefined): void {
  if (body !== undefined) lines.push(body);
}
