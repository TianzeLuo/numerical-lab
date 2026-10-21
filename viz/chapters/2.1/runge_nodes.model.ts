/**
 * `runge_nodes`（2.1 章）的 model：同一个函数、同样多的节点，**只换节点分布**。
 *
 * 面板与图层（顺序即绘制顺序，测试按此断言）
 * ------------------------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | line | `main` | `f_values`（Runge 函数，粗实线）+ `p_selected` + `p_other`（虚线） |
 * | 2 | scatter | `main` | `nodes_selected_*` / `nodes_other_*`（两套节点） |
 * | 3 | line | `aux` | `error_selected`（**整条**误差曲线，已夹地板）+ `error_other` + `envelope_selected` |
 * | 4 | line | `aux` | 地板参考线 `u/2`（节点上的精确 0 画在这里） |
 * | 5 | line | `sweep` | `max_error_vs_n_selected` 的全部 n（细实线）+ 偶数 n（○）+ 奇数 n（□） |
 * | 6 | line | `sweep` | `max_error_vs_n_other`（虚线） |
 * | 7 | line | `sweep` | `f` 自身量级参考线（`max|f|`，帧数组读数） |
 * | 8 | line | `sweep` | 临界值 `n=12` 的竖直参考线（规则 6） |
 * | 9 | scatter | `sweep` | 等距误差的全局最小点（帧数组读数）与当前 `k` 处两个标记 |
 *
 * 时间轴语义
 * ----------
 * `mode="scrub"`、`label="已用节点数 k"`：帧在 `t=k` 上给的是**用前 k 个节点**插值的结果。
 * `scene.t` 与所有文本里的 `k` 都取 `meta.k`（后端实到的节点数），**不拿 `opts.t` 冒充**；
 * 两者不等时（例如 `t=0` 被后端夹到 `k=1`）在正文里如实写出来。
 *
 * 防误导七条里本页的重点
 * ----------------------
 * * 规则 1/3：误差曲线与主图同屏（`aux` 是 log 误差轴，`requiredAux=["error"]`）；
 * * 规则 2/4：两个自适应轴各挂一枚角标，`sweep` 面板把"不单调"画成两条子列；
 * * 规则 6：`frame.critical.n = 12` 画成竖直参考线 + `criticalBadge`（`beyond="above"`）。
 *
 * 两条最容易被误读的地方，以及本页怎么防
 * --------------------------------------
 * 1. **"穿过节点 = 拟合好"**：节点上的误差**恰好为 0**，而对数轴上没有 0。帧已经把
 *    曲线夹到地板 `u/2` 并给出 `n_exact_zero_error` / `n_error_at_floor`；本页把这两个
 *    计数、地板值本身、以及"地板线画在哪"三件事同时摆在图上，并且**只用夹过的
 *    `error_selected` 画线**——从不把 0 悄悄丢给渲染层（那会让曲线断成 k 段）。
 * 2. **"n 越大越差（或越好）"**：等距与 Chebyshev 的误差都按奇偶分成两条子列，
 *    各自单调而合起来是锯齿。`sweep` 面板把偶数 n 与奇数 n 画成两条独立的线
 *    （○/□ 标记），并把 n=7 的全局最小点标出来——"不单调"是被**看见**的。
 */

import { niceLimits } from "../../shared/src/axes.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import type { Frame } from "../../shared/src/frame.js";
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
import {
  axisTitleOf,
  axisYlabelOf,
  nearestIndex,
  num,
  plain,
  rowLayout,
  textBlock,
  unitBudget,
} from "./scaffold.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "2.1";
const VIZ = "runge_nodes";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "Runge 现象：节点分布决定一切";
/** manifest 的 `n.critical = [12]`；`frame.critical.n` 缺失时的回落值。 */
const CRITICAL_N = 12;

/** 固定角标槽位（见 `model.ts::BADGE_LAYOUT`）：缺一枚时其余不挪位。 */
const SLOT = {
  source: 0,
  axisMode: 1,
  auxAxis: 2,
  sweepAxis: 3,
  critical: 4,
  status: 5,
  overflow: 6,
  degraded: 7,
} as const;

export const rungeNodesModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需数组：缺一个就抛 ModelInputError（宁可当场炸，也不画半张图）——
  const xs = requireArray(frame, "x_grid");
  const f = requireArray(frame, "f_values");
  const pSel = requireArray(frame, "p_selected");
  const pOth = requireArray(frame, "p_other");
  const errSel = requireArray(frame, "error_selected");
  const errOth = requireArray(frame, "error_other");
  const envSel = requireArray(frame, "envelope_selected");
  const ndSelX = requireArray(frame, "nodes_selected_x");
  const ndSelY = requireArray(frame, "nodes_selected_y");
  const ndOthX = requireArray(frame, "nodes_other_x");
  const ndOthY = requireArray(frame, "nodes_other_y");
  const nAxis = requireArray(frame, "n_axis");
  const sweepSel = requireArray(frame, "max_error_vs_n_selected");
  const sweepOth = requireArray(frame, "max_error_vs_n_other");
  requireSameLength(xs, f, "x_grid 与 f_values");
  requireSameLength(xs, pSel, "x_grid 与 p_selected");
  requireSameLength(xs, pOth, "x_grid 与 p_other");
  requireSameLength(xs, errSel, "x_grid 与 error_selected");
  requireSameLength(xs, errOth, "x_grid 与 error_other");
  requireSameLength(xs, envSel, "x_grid 与 envelope_selected");
  requireSameLength(ndSelX, ndSelY, "nodes_selected_x 与 nodes_selected_y");
  requireSameLength(ndOthX, ndOthY, "nodes_other_x 与 nodes_other_y");
  requireSameLength(nAxis, sweepSel, "n_axis 与 max_error_vs_n_selected");
  requireSameLength(nAxis, sweepOth, "n_axis 与 max_error_vs_n_other");
  const pitfalls = requirePitfalls(frame);

  // —— 可选量：缺了只降级（记进 degraded，最终收进一枚 status 角标）——
  const degraded: string[] = [];
  const metaOf = (key: string, label: string): number | undefined => {
    const v = metaNumber(frame, key);
    if (v === undefined) degraded.push(`帧 meta 缺 ${key}：${label}`);
    return v;
  };
  const k = metaOf("k", "不画当前位置标记（不猜是第几个节点）");
  const n = metaOf("n", "标题行不显示目标节点数 n");
  const errSelMax = metaOf("max_error_selected", "不显示当前分布的最大误差（不重算）");
  const errOthMax = metaOf("max_error_other", "不显示另一分布的最大误差（不重算）");
  const lamSelK = metaOf("lambda_selected_k", "不显示 Λ_k（不重算）");
  const lamOthK = metaOf("lambda_other_k", "不显示另一分布的 Λ_k");
  const envMax = metaOf("envelope_max", "不显示包络最大值");
  const errFloor = metaOf("error_floor", "不画地板参考线（不猜 u/2）");
  const envLooseness = metaOf("envelope_looseness", "不显示包络的宽松倍数");
  const nAtFloor = metaOf("n_error_at_floor", "不显示被夹到地板的点数");
  const nExactZero = metaOf("n_exact_zero_error", "不显示节点上精确 0 的个数");
  const maxErrEqN = metaOf("max_error_equispaced_n", "不显示 n_axis 末点处的等距误差");
  const maxErrChN = metaOf("max_error_chebyshev_n", "不显示 n_axis 末点处的 Chebyshev 误差");
  const divergence = metaOf("divergence_ratio", "不显示两种分布在末点的相差倍数");
  const isChebyshev = (metaNumber(frame, "is_chebyshev") ?? 0) > 0.5;
  const family = plain(stringOf(frame, "family"));
  const otherFamily = plain(stringOf(frame, "other_family"));
  if (family === undefined || otherFamily === undefined) {
    degraded.push("帧 strings 缺 family/other_family：曲线名改用帧数组 label（不按 params 猜分布）");
  }
  const note = plain(stringOf(frame, "note"));
  const tMeta = metaNumber(frame, "t");
  const tNow = tMeta ?? opts.t;
  if (tMeta === undefined) degraded.push("帧 meta 缺 t：scene.t 改用请求 t（两者可能不同）");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: tNow,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  // —— 排版：主图（f 与两条插值曲线）/ 误差（log）/ 误差随 n（log）——
  // footer=128：本章 pitfalls 实测 5 行（位图字体折行，比浏览器略宽）→ 面板底 472 上留 6px 净空。
  const layout = rowLayout(width, height, [1.65, 0.95, 0.62], { gap: 36, footer: 128 });
  const [mainRect, auxRect, sweepRect] = layout.rects as [
    [number, number, number, number],
    [number, number, number, number],
    [number, number, number, number],
  ];

  // 主图纵轴：由 f 与两条插值曲线的**有限值**决定（p 在两端会冲到 ±1e2 量级）。
  const mainLo = Math.min(minOf(f, -1), minOf(pSel, -1), minOf(pOth, -1));
  const mainHi = Math.max(maxOf(f, 1), maxOf(pSel, 1), maxOf(pOth, 1));
  const mainAxis = makeAxis("main", mainRect, {
    xlim: [-1, 1],
    ylim: niceLimits(mainLo, mainHi, 7, { pad: 0.06 }),
    // 上面还有面板：xlabel 置空（它的装饰带会把下一行的轴标题压住），刻度保留。
    xlabel: "",
    ylabel: axisYlabelOf(SPEC, "main", "y"),
    title: axisTitleOf(SPEC, "main", "f(x) 与两种节点分布的插值多项式"),
    autoscaled: true,
  });

  // 误差面板：log 纵轴，范围由三条曲线（已夹地板）的正值决定。
  const auxRange = combinedPositiveRange([errSel, errOth, envSel]);
  const auxAxis = makeAxis("aux", auxRect, {
    xlim: [-1, 1],
    ylim: auxRange === undefined ? [1e-16, 1] : niceLimits(auxRange[0], auxRange[1], 7, { log: true, pad: 0.06 }),
    yscale: "log",
    xlabel: "",
    ylabel: axisYlabelOf(SPEC, "aux", "误差"),
    title: axisTitleOf(SPEC, "aux", "插值误差 |f-p|（log）与严格包络"),
    autoscaled: true,
  });
  if (auxRange === undefined) degraded.push("error/envelope 三个数组里没有 >0 的有限值：误差面板用占位范围 [1e-16, 1]，不画曲线");

  // 误差随 n：横轴就是节点数（这条曲线**不随 time 变**，后端固定算到 n_axis 的末点）。
  const nLo = minOf(nAxis, 1);
  const nHi = Math.max(maxOf(nAxis, nLo + 1), nLo + 1);
  const fScale = maxOf(f, 1);
  const sweepRange = combinedPositiveRange([sweepSel, sweepOth, Float64Array.of(fScale)]);
  const sweepAxis = makeAxis("sweep", sweepRect, {
    xlim: [nLo - 0.5, nHi + 0.5],
    ylim: sweepRange === undefined ? [1e-16, 1] : niceLimits(sweepRange[0], sweepRange[1], 7, { log: true, pad: 0.08 }),
    yscale: "log",
    xlabel: "节点数 $n$",
    ylabel: "最大误差",
    title: "误差随 n：奇偶两条子列各自发散（不是单调）",
    autoscaled: true,
  });
  scene.axes.push(mainAxis, auxAxis, sweepAxis);

  // —— 图层 1：主图三条曲线 ——
  // 注意：`LineSeries.label` 不是元数据——两个渲染器都把它画在**曲线最后一个可见点**旁，
  // 而这些曲线都终止在面板右边缘（再往右 26px 就是画布边），写长了会被裁成半个词。
  // 所以颜色含义一律进正文图例（"橙=f、蓝=当前分布…"），`label` 只留给散点（渲染器不画它）。
  scene.layers.push(
    lineLayer("main", [
      series(xs, f, pal.exact, { width: 2.2 }),
      series(xs, pSel, pal.curves[0], { width: 1.7 }),
      series(xs, pOth, pal.curves[1], { width: 1.6, dash: [7, 4] }),
    ]),
  );

  // —— 图层 2：两套节点（节点上的误差恰好为 0，这正是"穿过节点不代表拟合好"的源头）——
  const nodePx = Math.max(1.6, Math.min(mainRect[2], mainRect[3]) / 90);
  if (ndSelX.length > 0) {
    scene.layers.push(scatterLayer("main", ndSelX, ndSelY, { r: nodePx, color: pal.curves[0], label: `节点：${family ?? "当前分布"}` }));
  }
  if (ndOthX.length > 0) {
    scene.layers.push(
      scatterLayer("main", ndOthX, ndOthY, { r: Math.max(1.2, nodePx * 0.7), color: pal.curves[1], alpha: 0.85, label: `节点：${otherFamily ?? "另一分布"}` }),
    );
  }

  // —— 图层 3：整条误差曲线 + 严格包络（规则 3：误差与主图同屏）——
  // 只用**夹过地板**的数组；`error_selected_raw` 里的精确 0 在对数轴上无定义，
  // 本页不把它画出来，而是把"有多少个 0"写进文本（见文件头第 1 条）。
  const selSeries = segmentedSeries(xs, errSel, pal.curves[0], { width: 1.8, accept: (v) => v > 0 });
  const othSeries = segmentedSeries(xs, errOth, pal.curves[1], { width: 1.5, dash: [6, 3], accept: (v) => v > 0 });
  const envSeries = segmentedSeries(xs, envSel, pal.warn, { width: 1.4, dash: [3, 3], accept: (v) => v > 0 });
  if (selSeries.length + othSeries.length + envSeries.length > 0) {
    scene.layers.push(lineLayer("aux", [...selSeries, ...othSeries, ...envSeries]));
  } else {
    degraded.push("误差与包络数组里没有一个 >0 的有限值：log 轴上一段曲线都画不出来");
  }
  const raw = frame.has("error_selected_raw") ? requireArray(frame, "error_selected_raw") : undefined;

  // —— 图层 4：地板参考线（节点上的精确 0 就画在这条线上）——
  if (errFloor !== undefined && errFloor > 0) {
    scene.layers.push(
      lineLayer("aux", series([-1, 1], [errFloor, errFloor], pal.reference, { width: 1, dash: [2, 4] })),
    );
  }

  // —— 图层 5~8：误差随 n（本页最要紧的一张图：不单调是"看"出来的）——
  scene.layers.push(lineLayer("sweep", buildSweepSeries(nAxis, sweepSel, sweepOth, pal, degraded)));
  scene.layers.push(
    lineLayer("sweep", [
      series([nLo, nHi], [fScale, fScale], pal.reference, { width: 1, dash: [2, 4] }),
    ]),
  );
  const critN = criticalOf(frame, "n", CRITICAL_N);
  const critFirst = Array.isArray(critN) ? (critN[0] as number) : critN;
  if (Number.isFinite(critFirst) && critFirst >= nLo && critFirst <= nHi) {
    scene.layers.push(
      lineLayer("sweep", [
        series([critFirst, critFirst], [sweepAxis.ylim[0], sweepAxis.ylim[1]], pal.marker, { width: 1.2, dash: [4, 3] }),
      ]),
    );
  }

  // —— 图层 9：等距误差的全局最小点（帧数组读数；它就是"触底之后不单调"的那个底）——
  const minIdx = argMin(sweepSel);
  const nMin = minIdx >= 0 ? valueAt(nAxis, minIdx) : undefined;
  const vMin = minIdx >= 0 ? valueAt(sweepSel, minIdx) : undefined;
  if (nMin !== undefined && vMin !== undefined && vMin > 0) {
    scene.layers.push(scatterLayer("sweep", [nMin], [vMin], { r: 3, color: pal.exact, label: "等距误差的全局最小点" }));
  }

  // —— 图层 9b：当前 k 的标记（两个分布各一枚；位置取帧里的 mark_* / meta）——
  const markK = frame.has("mark_k") ? requireArray(frame, "mark_k") : undefined;
  const kNow = markK !== undefined && markK.length > 0 ? (markK[0] as number) : k;
  const markX: number[] = [];
  const markY: number[] = [];
  if (kNow !== undefined) {
    const idx = nearestIndex(nAxis, kNow);
    if (idx >= 0) {
      const ySel = valueAt(sweepSel, idx);
      const yOth = valueAt(sweepOth, idx);
      if (ySel !== undefined && ySel > 0) {
        markX.push(kNow);
        markY.push(ySel);
      }
      if (yOth !== undefined && yOth > 0) {
        markX.push(kNow);
        markY.push(yOth);
      }
    } else {
      degraded.push("n_axis 里没有有限值：不画当前 k 的标记");
    }
  }
  if (markX.length > 0) {
    scene.layers.push(scatterLayer("sweep", markX, markY, { r: 3.4, color: pal.marker, label: "当前 $k$ 处两个分布的最大误差" }));
  }

  // —— 文本：一行一个 text 层（见 scaffold.ts 的约定）——
  const summary = paramSummary(frame, opts, [
    { name: "a", label: "$a$" },
    { name: "n", label: "$n$" },
  ]);
  degraded.push(...summary.degraded);
  const mainBudget = unitBudget(mainRect[2]);
  const mainLines: string[] = [];
  if (k !== undefined) {
    mainLines.push(`k=${num(k, 4)}${n === undefined ? "" : `/${num(n, 4)}`} 个节点　${summary.text}`);
    if (Math.abs(tNow - k) > 0.5) {
      mainLines.push(`请求 t=${num(tNow, 4)}，帧实到 k=${num(k, 4)}（帧里的 k 已按可用节点数夹住，不把 t 当索引）。`);
    }
  } else {
    mainLines.push(`${summary.text}`);
  }
  mainLines.push(`橙=f(x)　蓝=${family ?? "当前分布"}　绿虚线=${otherFamily ?? "另一分布"}　圆点=节点`);
  const lamLine = lambdaLine(frame);
  if (lamLine !== "") mainLines.push(lamLine);
  if (errSelMax !== undefined && errOthMax !== undefined) {
    mainLines.push(
      `最大误差 ${num(errSelMax, 4)} vs ${num(errOthMax, 4)}` +
        (lamSelK !== undefined && lamOthK !== undefined ? `　Λ_k=${num(lamSelK, 4)} vs ${num(lamOthK, 4)}` : ""),
    );
  }
  if (note !== undefined) mainLines.push(`帧 note：${note}`);
  scene.texts.push(...textBlock(mainRect[0] + 8, mainRect[1] + 4, mainRect[1] + mainRect[3] - 3, mainLines, pal.ink, mainBudget, "top"));

  const auxBudget = unitBudget(auxRect[2]);
  // 面板只有 5 行可用（142/82/54 三行排版 + 一行一字），所以正文按"每条一行"写：
  // 折叠掉的部分（例如"整条曲线而不是只画最大值"）由面板标题与 pitfalls 面板承担。
  // 图例只列**真的画了**的线：缺 meta.error_floor 时地板线不画，图例里就不能出现"灰虚线=地板"。
  const auxLines: string[] = [
    `蓝=${family ?? "当前分布"}误差　绿虚线=${otherFamily ?? "另一分布"}误差　红虚线=严格包络` +
      (errFloor === undefined ? "" : "　灰虚线=地板 u/2"),
  ];
  const rawZeros = raw === undefined ? undefined : countExactZero(raw);
  if (errFloor !== undefined && nExactZero !== undefined) {
    auxLines.push(
      `节点上误差恰好为 0（本帧 ${num(nExactZero, 3)} 个${rawZeros === undefined ? "" : `，error_selected_raw 实测 ${num(rawZeros, 3)} 个`}）：` +
        `log 轴没有 0，曲线先夹到地板 u/2=${num(errFloor, 3)}${nAtFloor === undefined ? "" : `（${num(nAtFloor, 3)} 点被夹）`}。`,
    );
  }
  if (envMax !== undefined) {
    auxLines.push(
      `严格包络（红虚线）max=${num(envMax, 4)}` +
        (envLooseness === undefined ? "" : `，本帧宽松 ${num(envLooseness, 3)} 倍`) +
        "：它只说「不会超过」。",
    );
  }
  scene.texts.push(...textBlock(auxRect[0] + 8, auxRect[1] + 4, auxRect[1] + auxRect[3] - 3, auxLines, pal.muted, auxBudget, "top"));

  const sweepBudget = unitBudget(sweepRect[2]);
  // 这一行只有 3 行文本可用：说两件事——两条子列、以及 n_max 处的量级对照。
  const sweepLines: string[] = ["蓝=全部 n　○=偶 n　□=奇 n　绿虚线=另一分布　橙点=最小　红竖线=临界 n"];
  if (nMin !== undefined && vMin !== undefined) {
    sweepLines.push(`两条子列各自单调，合起来才是锯齿（本帧最小 n=${num(nMin, 3)} 处 ${num(vMin, 4)}）。`);
  }
  if (maxErrEqN !== undefined && maxErrChN !== undefined) {
    sweepLines.push(
      `曲线不随时间轴变：n=${num(nHi, 3)} 时等距 ${num(maxErrEqN, 4)} vs Chebyshev ${num(maxErrChN, 4)}` +
        (divergence === undefined ? "" : `（相差 ${num(divergence, 3)} 倍）`) +
        "；红竖线是临界 n。",
    );
  }
  scene.texts.push(...textBlock(sweepRect[0] + 8, sweepRect[1] + 4, sweepRect[1] + sweepRect[3] - 3, sweepLines, pal.note, sweepBudget, "top"));

  // —— 角标（固定槽位；规则 4/5/6）——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(true), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(badge("autoscale", "误差面板：自适应 log 纵轴（按本帧误差与包络取整；自适应会掩盖量级变化）", 8, 8), SLOT.auxAxis, width),
  );
  scene.badges.push(
    placeBadge(badge("autoscale", "误差-n 面板：自适应 log 纵轴（横轴是节点数 n，不是时间）", 8, 8), SLOT.sweepAxis, width),
  );
  scene.badges.push(placeBadge(criticalBadge("n", n ?? k ?? Number.NaN, critN, { beyond: "above" }), SLOT.critical, width));
  const status: string[] = [];
  if (k !== undefined && errSelMax !== undefined && errOthMax !== undefined) {
    status.push(`k=${num(k, 4)}：${family ?? "当前"}最大误差 ${num(errSelMax, 4)} vs ${otherFamily ?? "另一"} ${num(errOthMax, 4)}`);
  }
  if (isChebyshev && errSelMax !== undefined) status.push("当前分布=Chebyshev：误差被摊平到整个区间（端点为节点，无 Runge 振荡）");
  if (status.length === 0) status.push("帧未给出可判读的误差（预期有 meta.k / meta.max_error_selected）");
  scene.badges.push(placeBadge(badge("status", status.join("；"), 8, 8), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  const nBad = countNonFinite(errSel) + countNonFinite(errOth) + countNonFinite(envSel) + countNonFinite(pSel) + countNonFinite(pOth);
  applyOverflow(scene, nBad, { slot: SLOT.overflow, width });
  // 帧自己声明了 overflow（后端判据：有曲线越过 log 轴上界 1e12）时也要如实报出来。
  const metaOverflow = (metaNumber(frame, "overflow") ?? 0) > 0.5;
  if (metaOverflow && !scene.overflow.flag) {
    scene.overflow = { flag: true, note: "帧 meta.overflow=true：后端报告有误差/包络点越过对数轴上界（1e12），图上被裁到轴端" };
    scene.badges.push(placeBadge(badge("overflow", `溢出：${scene.overflow.note}`, 8, 44), SLOT.overflow, width));
  }
  return scene;
};

// --------------------------------------------------------------------------- //
// 局部工具
// --------------------------------------------------------------------------- //

/** `Λ_n`（满节点）一行：`meta` 里三个量齐了才写，缺一个就不写（不重算、不猜）。 */
function lambdaLine(frame: Frame): string {
  const lam = metaNumber(frame, "lambda_selected_n");
  const other = metaNumber(frame, "lambda_other_n");
  if (lam === undefined || other === undefined) return "";
  return `　Λ_n（同 n 满节点）=${num(lam, 4)} vs ${num(other, 4)}`;
}

/** 有限值的 min（全非有限时用给定的退化值）。 */
function minOf(a: ArrayLike<number>, fallback: number): number {
  let lo = Number.POSITIVE_INFINITY;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i] as number;
    if (Number.isFinite(v) && v < lo) lo = v;
  }
  return Number.isFinite(lo) ? lo : fallback;
}

/** 有限值的 max（全非有限时用给定的退化值）。 */
function maxOf(a: ArrayLike<number>, fallback: number): number {
  let hi = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i] as number;
    if (Number.isFinite(v) && v > hi) hi = v;
  }
  return Number.isFinite(hi) ? hi : fallback;
}

/** 多个数组的**正有限值**并集范围（对数轴的定义域）；没有正值时 undefined。 */
function combinedPositiveRange(arrays: readonly ArrayLike<number>[]): [number, number] | undefined {
  let lo = Number.POSITIVE_INFINITY;
  let hi = Number.NEGATIVE_INFINITY;
  for (const a of arrays) {
    const r = finitePositiveMinMax(a);
    if (r === undefined) continue;
    if (r[0] < lo) lo = r[0];
    if (r[1] > hi) hi = r[1];
  }
  return Number.isFinite(lo) && Number.isFinite(hi) ? [lo, hi] : undefined;
}

function valueAt(a: ArrayLike<number>, i: number): number | undefined {
  if (i < 0 || i >= a.length) return undefined;
  const v = a[i] as number;
  return Number.isFinite(v) ? v : undefined;
}

/** 恰好等于 0 的元素个数（与帧 `meta.n_exact_zero_error` 同口径；本页只用它复核）。 */
function countExactZero(a: ArrayLike<number>): number {
  let n = 0;
  for (let i = 0; i < a.length; i += 1) if ((a[i] as number) === 0) n += 1;
  return n;
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

/**
 * 误差随 n 的曲线：全部 n（细实线）+ 偶数 n（○）+ 奇数 n（□）+ 另一分布（虚线）。
 *
 * 为什么要把奇偶**分开画**：等距节点的 Runge 误差按 n 的奇偶分成两条子列，
 * 每条都严格递增（实测偶数列 1.045→1.916→3.663，奇数列 0.2474→0.300→0.557），
 * 合起来才是"上上下下"的锯齿。只画一条线时，读者会把它读成"随机抖动"；
 * 分成两条之后，"各自发散"是直接可见的几何事实。
 */
function buildSweepSeries(
  nAxis: ArrayLike<number>,
  sel: ArrayLike<number>,
  oth: ArrayLike<number>,
  pal: ReturnType<typeof palette>,
  degraded: string[],
): LineSeries[] {
  const evX: number[] = [];
  const evY: number[] = [];
  const odX: number[] = [];
  const odY: number[] = [];
  for (let i = 0; i < nAxis.length; i += 1) {
    const nv = nAxis[i] as number;
    const yv = sel[i] as number;
    if (!Number.isFinite(nv) || !Number.isFinite(yv) || yv <= 0) continue;
    if (Math.round(nv) % 2 === 0) {
      evX.push(nv);
      evY.push(yv);
    } else {
      odX.push(nv);
      odY.push(yv);
    }
  }
  const out: LineSeries[] = [];
  const allSel = splitPositive(nAxis, sel);
  if (allSel.length === 0) degraded.push("max_error_vs_n_selected 里没有 >0 的有限值：log 轴上画不出这条曲线");
  for (const s of allSel) out.push(series(s.x, s.y, pal.curves[0], { width: 1.1 }));
  if (evX.length >= 2) out.push(series(evX, evY, pal.curves[0], { width: 1.6, dash: [1, 2], marker: "o" }));
  if (odX.length >= 2) out.push(series(odX, odY, pal.curves[0], { width: 1.6, dash: [1, 2], marker: "s" }));
  const allOth = splitPositive(nAxis, oth);
  if (allOth.length === 0) degraded.push("max_error_vs_n_other 里没有 >0 的有限值：log 轴上画不出这条曲线");
  for (const s of allOth) out.push(series(s.x, s.y, pal.curves[1], { width: 1.4, dash: [6, 3] }));
  return out;
}

/** 把 `(x, y)` 里 `y > 0` 的连续段切出来（log 轴上 `y ≤ 0` 无定义，不能连线跨过去）。 */
function splitPositive(x: ArrayLike<number>, y: ArrayLike<number>): { x: number[]; y: number[] }[] {
  const runs: { x: number[]; y: number[] }[] = [];
  let cx: number[] = [];
  let cy: number[] = [];
  const n = Math.min(x.length, y.length);
  for (let i = 0; i < n; i += 1) {
    const xv = x[i] as number;
    const yv = y[i] as number;
    if (Number.isFinite(xv) && Number.isFinite(yv) && yv > 0) {
      cx.push(xv);
      cy.push(yv);
    } else if (cx.length > 0) {
      runs.push({ x: cx, y: cy });
      cx = [];
      cy = [];
    }
  }
  if (cx.length > 0) runs.push({ x: cx, y: cy });
  return runs;
}
