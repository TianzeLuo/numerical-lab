/**
 * `lebesgue_growth`（2.1 章）的 model：把"发散"变成一个可测量的数 Λ_n，并给出它的斜率。
 *
 * 面板与图层（顺序即绘制顺序，测试按此断言）
 * ------------------------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | loglogfit | `main` | `lambda_equispaced` + **实测斜率** 与**理论斜率**两条参考线（规则 2） |
 * | 2 | line | `main` | `lambda_chebyshev`（实测）、`theory_equispaced` / `theory_chebyshev`（虚线渐近公式） |
 * | 3 | scatter | `main` | 当前 n 处两个家族的 Λ（`mark_n` / `mark_lambda`） |
 * | 4 | line | `aux` | `slope_measured_equispaced` / `slope_measured_chebyshev`（实测斜率） |
 * | 5 | line | `aux` | `slope_theory_equispaced` / `slope_theory_chebyshev`（理论斜率，虚线） |
 * | 6 | line | `aux` | 当前 n 的竖直游标 |
 * | 7 | line | `dev` | `error_asymptotic_equispaced` / `error_asymptotic_chebyshev`（相对偏差） |
 * | 8 | line | `dev` | 当前 n 的竖直游标 + 当前偏差标记 |
 *
 * 时间轴语义
 * ----------
 * `mode="scrub"`、`tMax=60`：`t` 就是**当前节点数 n**（帧里写作 `meta.n_mark`）。
 * 所有 `*_n` 标量都取该下标处的帧值，不重算、不外推；`n_mark` 越界时按帧自身的夹取结果显示。
 *
 * 防误导七条里本页的重点
 * ----------------------
 * * 规则 2（本页的主角）：主轴的 `loglogfit` 层把**实测斜率**与**理论斜率**同时画出来，
 *   `aux` 面板再把两个家族的实测/理论斜率成对画一遍——"看起来在涨"和"按 2^n 在涨"
 *   是两件事，两个斜率必须在同一张图上对照。
 * * 规则 4：三个面板的纵轴都是自适应的，各挂一枚角标，并在正文里写明"会掩盖量级变化"。
 * * 规则 6：$n_{\max}$ 的临界值 50 画成 `criticalBadge`（`beyond="above"`），
 *   越过时写明"Λ·u 已经让界失去意义"。
 *
 * 两条最容易被误读的地方，以及本页怎么防
 * --------------------------------------
 * 1. **"Λ_n 越大越糟"被当成函数/算法的属性**：Λ_n 只是**节点分布**的属性，与 f、
 *    与用哪条公式都无关。本页把两个家族画在同一根 log 纵轴上（n=30 差 2.1e6 倍），
 *    并在正文点明"两条线只差节点分布"。
 * 2. **把渐近公式当等式**：等距那条只准到 ~14%（n=30 处实测/公式 = 0.853）。`dev`
 *    面板直接把 `|Λ实测/Λ渐近 − 1|` 画出来（log 纵轴），把"公式不是等式"变成一条曲线；
 *    同时 `aux` 面板里的理论斜率是**求和公式的局部斜率**（`slope_theory_*`，帧给的），
 *    不是把 2^n 抄一遍。
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
const VIZ = "lebesgue_growth";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "Lebesgue 常数：指数爆炸与它的斜率";
/** manifest 的 `n_max.critical = [50]`；`frame.critical.n_max` 缺失时的回落值。 */
const CRITICAL_NMAX = 50;
/** Λ·u 超过这个量级时，插值误差的"界"已经被舍入项吃掉（见 manifest 的 criticalNote）。 */
const KAPPA_U_WARN = 1e-3;

const SLOT = {
  source: 0,
  axisMode: 1,
  auxAxis: 2,
  devAxis: 3,
  critical: 4,
  status: 5,
  overflow: 6,
  degraded: 7,
} as const;

export const lebesgueGrowthModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需数组 ——
  const nAxis = requireArray(frame, "n_axis");
  const lamEq = requireArray(frame, "lambda_equispaced");
  const lamCh = requireArray(frame, "lambda_chebyshev");
  const thEq = requireArray(frame, "theory_equispaced");
  const thCh = requireArray(frame, "theory_chebyshev");
  const slopeMq = requireArray(frame, "slope_measured_equispaced");
  const slopeMc = requireArray(frame, "slope_measured_chebyshev");
  const slopeTq = requireArray(frame, "slope_theory_equispaced");
  const slopeTc = requireArray(frame, "slope_theory_chebyshev");
  const devEq = requireArray(frame, "error_asymptotic_equispaced");
  const devCh = requireArray(frame, "error_asymptotic_chebyshev");
  for (const [name, arr] of [
    ["lambda_equispaced", lamEq],
    ["lambda_chebyshev", lamCh],
    ["theory_equispaced", thEq],
    ["theory_chebyshev", thCh],
    ["slope_measured_equispaced", slopeMq],
    ["slope_measured_chebyshev", slopeMc],
    ["slope_theory_equispaced", slopeTq],
    ["slope_theory_chebyshev", slopeTc],
    ["error_asymptotic_equispaced", devEq],
    ["error_asymptotic_chebyshev", devCh],
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
  const grid = metaOf("grid", "不显示网格点数（它决定 Λ 是否被低估）");
  const nMark = metaOf("n_mark", "不画当前 n 的游标（不猜时间轴索引）");
  const lamEqN = metaOf("lambda_equispaced_n", "不显示当前 n 的 Λ_等距");
  const lamChN = metaOf("lambda_chebyshev_n", "不显示当前 n 的 Λ_Chebyshev");
  const thEqN = metaOf("theory_equispaced_n", "不显示等距渐近公式的值");
  const thChN = metaOf("theory_chebyshev_n", "不显示 Chebyshev 渐近公式的值");
  const lamRatio = metaOf("lambda_ratio", "不显示两个家族的 Λ 之比");
  const lamEqU = metaOf("lambda_equispaced_times_u", "不显示等距的 Λ·u");
  const lamChU = metaOf("lambda_chebyshev_times_u", "不显示 Chebyshev 的 Λ·u");
  const slopeMeN = metaOf("slope_measured_equispaced_n", "不显示当前 n 的实测斜率（等距）");
  const slopeTeN = metaOf("slope_theory_equispaced_n", "不显示当前 n 的理论斜率（等距）");
  const slopeMcN = metaOf("slope_measured_chebyshev_n", "不显示当前 n 的实测斜率（Chebyshev）");
  const slopeTcN = metaOf("slope_theory_chebyshev_n", "不显示当前 n 的理论斜率（Chebyshev）");
  const gapEq = metaOf("slope_gap_equispaced", "不显示等距实测/理论斜率之差");
  const gapCh = metaOf("slope_gap_chebyshev", "不显示 Chebyshev 实测/理论斜率之差");
  const devEqN = metaOf("error_asymptotic_equispaced_n", "不显示当前 n 的渐近相对偏差（等距）");
  const devChN = metaOf("error_asymptotic_chebyshev_n", "不显示当前 n 的渐近相对偏差（Chebyshev）");
  const nAtFloor = metaOf("n_lambda_at_floor", "不显示被夹到对数地板的点数");
  const famBad = plain(stringOf(frame, "family_bad"));
  const famGood = plain(stringOf(frame, "family_good"));
  if (famBad === undefined || famGood === undefined) {
    degraded.push("帧 strings 缺 family_bad/family_good：图例改用「等距」/「Chebyshev」（不按 params 猜）");
  }
  // 本页不渲染 `frame.strings.note`：它的每一句（Λ 值、渐近偏差、grid 效应、n=1 的 0）都已经
  // 逐条进了三个面板的正文，而 8/5/3 行的净空塞不下它——截断成半句比不显示更糟。
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

  // —— 排版：Λ_n（log-log）/ 斜率（log 纵轴）/ 渐近偏差（log 纵轴）——
  // footer=128：pitfalls 实测 5 行 → 面板底 472，与 pitfalls 框留 6px。
  const layout = rowLayout(width, height, [1.5, 0.95, 0.72], { gap: 36, footer: 128 });
  const [mainRect, auxRect, devRect] = layout.rects as [
    [number, number, number, number],
    [number, number, number, number],
    [number, number, number, number],
  ];

  const nLo = minOf(nAxis, 1);
  const nHi = Math.max(maxOf(nAxis, nLo + 1), nLo + 1);
  // 横轴（线性面板）：留半格边距；纵轴一律 log（Λ 从 1 涨到 1e15，线性轴只会得到一条贴底的线）。
  const xLin: [number, number] = [nLo - 0.5, nHi + 0.5];
  // 横轴（log 面板）：显式给范围，避免 niceLimits 把 1..30 扩成 0.1..100（数据只剩一半宽度）。
  const xLog: [number, number] = [nLo, nHi * 1.06];
  const yMain = rangeOr([lamEq, lamCh, thEq, thCh], [1, 1e6], { log: true });
  const ySlope = rangeOr([slopeMq, slopeMc, slopeTq, slopeTc], [0.1, 1], { log: true });
  const yDev = rangeOr([devEq, devCh], [1e-6, 1], { log: true });

  const mainAxis = makeAxis("main", mainRect, {
    xlim: xLog,
    xscale: "log",
    ylim: yMain,
    yscale: "log",
    xlabel: "",
    ylabel: axisYlabelOf(SPEC, "main", "Λ_n"),
    title: axisTitleOf(SPEC, "main", "Λ_n（log）实测 vs 渐近公式"),
    autoscaled: true,
  });
  const auxAxis = makeAxis("aux", auxRect, {
    xlim: xLin,
    ylim: ySlope,
    yscale: "log",
    xlabel: "",
    ylabel: axisYlabelOf(SPEC, "aux", "d log Λ / d log n"),
    title: axisTitleOf(SPEC, "aux", "log-log 斜率：实测 vs 理论（规则 2）"),
    autoscaled: true,
  });
  const devAxis = makeAxis("dev", devRect, {
    xlim: xLin,
    ylim: yDev,
    yscale: "log",
    xlabel: "节点数 $n$（上面两个面板的横轴同样是 $n$；主轴用对数刻度）",
    ylabel: "相对偏差",
    title: "渐近公式的相对偏差：实测/渐近 $-1$（对数轴）",
    autoscaled: true,
  });
  scene.axes.push(mainAxis, auxAxis, devAxis);

  // —— 图层 1：规则 2 的规范形态——实测曲线 + 实测斜率线 + 理论斜率参考线 ——
  // 一条轴上只放一个 `loglogfit`：它的两条文字标注都画在面板右上角，两个层会叠字。
  // 两个家族的"实测 vs 理论"成对对照在 `aux` 面板里（图层 4/5），不会漏掉任何一条。
  // `loglogfit` 的理论线锚在 `err` 的**末点**（n=n_max）上，所以缺 meta 时用斜率数组的末值，
  // 而不是"随便取一个"：读的还是帧数组，逐点对应关系写在降级说明里。
  const slopeMe = slopeMeN ?? lastPositive(slopeMq);
  const slopeTe = slopeTeN ?? lastPositive(slopeTq);
  scene.layers.push(
    loglogFitLayer("main", nAxis, lamEq, slopeMe, slopeTe, {
      seriesLabel: `实测斜率 ${num(slopeMe, 4)}（${famBad ?? "等距"}）`,
      theoryLabel: `理论斜率 ${num(slopeTe, 4)}（同一根对数轴上）`,
      color: pal.curves[0],
      theoryColor: pal.marker,
    }),
  );
  if (slopeMeN === undefined || slopeTeN === undefined) {
    degraded.push("帧 meta 缺 slope_*_n：loglogfit 的两个斜率改用帧数组 slope_*_equispaced 的末值（与理论线的锚点同为 n_max，不重算）");
  }

  // —— 图层 2：另一族实测 + 两条渐近公式 ——
  const mainSeries: LineSeries[] = [];
  const chSeries = segmentedSeries(nAxis, lamCh, pal.curves[1], { width: 1.7, accept: (v) => v > 0 });
  mainSeries.push(...chSeries);
  mainSeries.push(
    ...segmentedSeries(nAxis, thEq, pal.warn, { width: 1.3, dash: [6, 3], accept: (v) => v > 0 }),
  );
  mainSeries.push(
    ...segmentedSeries(nAxis, thCh, pal.reference, { width: 1.2, dash: [3, 3], accept: (v) => v > 0 }),
  );
  if (mainSeries.length === 0) degraded.push("Λ 与渐近公式四个数组里没有 >0 的有限值：主轴画不出曲线");
  if (mainSeries.length > 0) scene.layers.push(lineLayer("main", mainSeries));

  // —— 图层 3：当前 n 的两个 Λ ——
  const markN = frame.has("mark_n") ? requireArray(frame, "mark_n") : undefined;
  const markLam = frame.has("mark_lambda") ? requireArray(frame, "mark_lambda") : undefined;
  const nNow = markN !== undefined && markN.length > 0 ? (markN[0] as number) : nMark;
  const markXY: number[] = [];
  const markYY: number[] = [];
  if (nNow !== undefined && markLam !== undefined) {
    for (let i = 0; i < markLam.length && i < 2; i += 1) {
      const v = markLam[i] as number;
      if (Number.isFinite(v) && v > 0) {
        markXY.push(nNow);
        markYY.push(v);
      }
    }
  }
  if (markXY.length > 0) {
    scene.layers.push(scatterLayer("main", markXY, markYY, { r: 3.4, color: pal.marker, label: "当前 $n$ 处两个家族的 $\\Lambda_n$" }));
  } else if (nNow !== undefined) {
    degraded.push("帧数组 mark_lambda 缺失或没有正值：主轴不画当前 n 的标记（不重算）");
  }

  // —— 图层 4~6：斜率面板（实测 vs 理论，两个家族都画）——
  const auxSeries: LineSeries[] = [
    ...segmentedSeries(nAxis, slopeMq, pal.curves[0], { width: 1.7, accept: (v) => v > 0 }),
    ...segmentedSeries(nAxis, slopeMc, pal.curves[1], { width: 1.7, accept: (v) => v > 0 }),
    ...segmentedSeries(nAxis, slopeTq, pal.marker, { width: 1.2, dash: [6, 3], accept: (v) => v > 0 }),
    ...segmentedSeries(nAxis, slopeTc, pal.reference, { width: 1.2, dash: [3, 3], accept: (v) => v > 0 }),
  ];
  if (auxSeries.length > 0) scene.layers.push(lineLayer("aux", auxSeries));
  else degraded.push("四条斜率数组里没有 >0 的有限值：斜率面板画不出曲线");
  if (nNow !== undefined) {
    scene.layers.push(lineLayer("aux", series([nNow, nNow], [auxAxis.ylim[0], auxAxis.ylim[1]], pal.cursor, { width: 1, dash: [3, 3], label: "当前 $n$" })));
    const yMark: number[] = [];
    if (slopeMeN !== undefined && slopeMeN > 0) yMark.push(slopeMeN);
    if (slopeMcN !== undefined && slopeMcN > 0) yMark.push(slopeMcN);
    if (yMark.length > 0) {
      scene.layers.push(scatterLayer("aux", yMark.map(() => nNow), yMark, { r: 3, color: pal.marker, label: "当前 $n$ 的实测斜率" }));
    }
  }

  // —— 图层 7~8：渐近公式的相对偏差（"公式不是等式"的那条曲线）——
  const devSeries: LineSeries[] = [
    ...segmentedSeries(nAxis, devEq, pal.curves[0], { width: 1.7, accept: (v) => v > 0 }),
    ...segmentedSeries(nAxis, devCh, pal.curves[1], { width: 1.7, dash: [5, 3], accept: (v) => v > 0 }),
  ];
  if (devSeries.length > 0) scene.layers.push(lineLayer("dev", devSeries));
  else degraded.push("两条 error_asymptotic 数组里没有 >0 的有限值：偏差面板画不出曲线");
  if (nNow !== undefined) {
    scene.layers.push(lineLayer("dev", series([nNow, nNow], [devAxis.ylim[0], devAxis.ylim[1]], pal.cursor, { width: 1, dash: [3, 3] })));
    const yDevMark: number[] = [];
    if (devEqN !== undefined && devEqN > 0) yDevMark.push(devEqN);
    if (devChN !== undefined && devChN > 0) yDevMark.push(devChN);
    if (yDevMark.length > 0) {
      scene.layers.push(scatterLayer("dev", yDevMark.map(() => nNow), yDevMark, { r: 3, color: pal.marker, label: "当前 $n$ 的渐近相对偏差" }));
    }
  }

  // —— 文本 ——
  const summary = paramSummary(frame, opts, [
    { name: "n_max", label: "$n_{\\max}$" },
    { name: "grid", label: "grid" },
  ]);
  degraded.push(...summary.degraded);
  // 主轴右上角被 `loglogfit` 的两条标注占着（两个渲染器都画在那里），正文只占左半边。
  const mainBudget = Math.max(30, Math.floor(unitBudget(mainRect[2]) * 0.62));
  const mainLines: string[] = ["蓝=Λ_等距　绿=Λ_Chebyshev　红虚线=渐近公式(等距)　灰点线=渐近公式(Chebyshev)"];
  if (nMark !== undefined) mainLines.push(`当前 n=${num(nMark, 4)}（时间轴）　${summary.text}`);
  else mainLines.push(`${summary.text}`);
  if (lamEqN !== undefined && lamChN !== undefined) {
    mainLines.push(
      `Λ_等距=${num(lamEqN, 4)}（公式 ${thEqN === undefined ? "—" : num(thEqN, 4)}，偏差 ${devEqN === undefined ? "—" : num(devEqN, 3)}）；` +
        `Λ_Chebyshev=${num(lamChN, 6)}（公式 ${thChN === undefined ? "—" : num(thChN, 6)}）。`,
    );
  }
  if (lamRatio !== undefined) {
    mainLines.push(
      `同一个 n 只换节点分布：Λ 相差 ${num(lamRatio, 3)} 倍（Λ_n 是节点分布的属性）；` +
        (lamEqU === undefined || lamChU === undefined ? "" : `Λ·u=${num(lamEqU, 3)} vs ${num(lamChU, 3)}。`),
    );
  }
  // 主面板 8 行已被上面五条占满；帧 note 与 pitfalls 面板的信息重复（Λ 值/偏差/grid 效应都已逐条列出），
  // 所以这里**不**再塞 note——叠不下时 textBlock 会截断，截断后的半句话比不显示更糟。
  scene.texts.push(...textBlock(mainRect[0] + 8, mainRect[1] + 4, mainRect[1] + mainRect[3] - 3, mainLines, pal.ink, mainBudget, "top"));

  const auxBudget = unitBudget(auxRect[2]);
  const auxLines: string[] = ["蓝/绿=实测斜率　红虚线/灰点线=对应理论斜率（n=1 处理论斜率恰为 0，log 轴画不出，曲线从 n=2 起）"];
  if (slopeMeN !== undefined && slopeTeN !== undefined && slopeMcN !== undefined && slopeTcN !== undefined) {
    auxLines.push(
      `实测 vs 理论：等距 ${num(slopeMeN, 5)} vs ${num(slopeTeN, 5)}` +
        `${gapEq === undefined ? "" : `（差 ${num(gapEq, 3)}）`}；Chebyshev ${num(slopeMcN, 5)} vs ${num(slopeTcN, 5)}` +
        `${gapCh === undefined ? "" : `（差 ${num(gapCh, 3)}）`}。`,
    );
  }
  auxLines.push("纵轴是 log：两族相差 100 倍，线性轴会把 Chebyshev 那一对压成贴底直线。");
  scene.texts.push(...textBlock(auxRect[0] + 8, auxRect[1] + 4, auxRect[1] + auxRect[3] - 3, auxLines, pal.muted, auxBudget, "top"));

  const devBudget = unitBudget(devRect[2]);
  const devLines: string[] = ["蓝=等距的渐近偏差　绿虚线=Chebyshev 的（纵轴 log）"];
  if (devEqN !== undefined && devChN !== undefined) {
    devLines.push(`渐近公式不是等式：等距偏差 ${num(devEqN, 3)}、Chebyshev ${num(devChN, 3)}（n≥15 时等距稳定在 0.14，只准到 ~14%）。`);
  }
  if (grid !== undefined) {
    devLines.push(`grid=${num(grid, 4)} 会系统性低估 Λ_n（峰值挤在端点 O(n^-2) 薄层里），加到 20001 时 n=30 的 Λ 还会变。`);
  }
  if (nAtFloor !== undefined && nAtFloor > 0) devLines.push(`帧 meta.n_lambda_at_floor=${num(nAtFloor, 3)}：有 Λ 值落到对数地板 −16 上。`);
  scene.texts.push(...textBlock(devRect[0] + 8, devRect[1] + 4, devRect[1] + devRect[3] - 3, devLines, pal.note, devBudget, "top"));

  // —— 角标 ——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(true), SLOT.axisMode, width));
  scene.badges.push(placeBadge(badge("autoscale", "主轴：log-log 自适应（按本帧 Λ 范围取整；自适应会掩盖量级变化）", 8, 8), SLOT.auxAxis, width));
  scene.badges.push(placeBadge(badge("autoscale", "斜率/偏差面板：log 纵轴自适应（同上）", 8, 8), SLOT.devAxis, width));
  scene.badges.push(placeBadge(criticalBadge("n_max", nMax ?? Number.NaN, criticalOf(frame, "n_max", CRITICAL_NMAX), { beyond: "above" }), SLOT.critical, width));
  const status: string[] = [];
  if (nMark !== undefined && lamEqN !== undefined && lamChN !== undefined) {
    status.push(`n=${num(nMark, 4)}：Λ_等距=${num(lamEqN, 4)} vs Λ_Chebyshev=${num(lamChN, 5)}`);
  }
  if (lamEqU !== undefined && lamEqU > KAPPA_U_WARN) {
    status.push(`⚠ Λ·u=${num(lamEqU, 3)} 已经大于 1e-3：等距插值在双精度下彻底失效（界被舍入项吃掉了）`);
  } else if (lamEqU !== undefined) {
    status.push(`Λ·u=${num(lamEqU, 3)}（还远小于 1e-3）`);
  }
  if (status.length === 0) status.push("帧未给出可判读的 Λ（预期有 meta.n_mark / meta.lambda_equispaced_n）");
  scene.badges.push(placeBadge(badge("status", status.join("；"), 8, 8), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  const nBad = countNonFinite(lamEq) + countNonFinite(lamCh) + countNonFinite(slopeMq) + countNonFinite(slopeMc) + countNonFinite(devEq) + countNonFinite(devCh);
  applyOverflow(scene, nBad, { slot: SLOT.overflow, width });
  const metaOverflow = (metaNumber(frame, "overflow") ?? 0) > 0.5;
  if (metaOverflow && !scene.overflow.flag) {
    scene.overflow = { flag: true, note: "帧 meta.overflow=true：后端报告 Λ 越过对数轴上界（1e16），图上被裁到轴端" };
    scene.badges.push(placeBadge(badge("overflow", `溢出：${scene.overflow.note}`, 8, 44), SLOT.overflow, width));
  }
  return scene;
};

// --------------------------------------------------------------------------- //
// 局部工具
// --------------------------------------------------------------------------- //

/** 有限值的 min（全非有限时用退化值）。 */
function minOf(a: ArrayLike<number>, fallback: number): number {
  let lo = Number.POSITIVE_INFINITY;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i] as number;
    if (Number.isFinite(v) && v < lo) lo = v;
  }
  return Number.isFinite(lo) ? lo : fallback;
}

/** 有限值的 max（全非有限时用退化值）。 */
function maxOf(a: ArrayLike<number>, fallback: number): number {
  let hi = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i] as number;
    if (Number.isFinite(v) && v > hi) hi = v;
  }
  return Number.isFinite(hi) ? hi : fallback;
}

/** 多个数组的**正有限值**并集 → 取整后的轴范围（对数轴要求 >0）。 */
function rangeOr(arrays: readonly ArrayLike<number>[], fallback: [number, number], opts: { log?: boolean } = {}): [number, number] {
  let lo = Number.POSITIVE_INFINITY;
  let hi = Number.NEGATIVE_INFINITY;
  for (const a of arrays) {
    const r = finitePositiveMinMax(a);
    if (r === undefined) continue;
    if (r[0] < lo) lo = r[0];
    if (r[1] > hi) hi = r[1];
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return fallback;
  return niceLimits(lo, hi, 7, { log: opts.log ?? false, pad: 0.06 });
}

/**
 * 数组里**最后一个**正有限值（帧 `meta.slope_*_n` 缺失时的显示回落）。
 *
 * 为什么是"末值"：`loglogFitLayer` 的理论参考线锚在 `err` 的末点上，斜率必须与那个锚点同源；
 * 取中间任何一个下标都会画出一条与锚点不匹配的线。读帧数组、不改数据。
 */
function lastPositive(y: ArrayLike<number>): number {
  for (let i = y.length - 1; i >= 0; i -= 1) {
    const v = y[i] as number;
    if (Number.isFinite(v) && v > 0) return v;
  }
  return 0;
}
