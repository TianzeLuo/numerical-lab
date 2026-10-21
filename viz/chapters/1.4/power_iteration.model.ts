/**
 * `power_iteration`（1.4 章）的 model：单位圆上的迭代向量 + 残差/理论收敛率副轴。
 *
 * 面板与图层（顺序即绘制顺序，测试按此断言）
 * ------------------------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | line | `main` | 单位圆 `circle_x/circle_y`（虚线；所有迭代向量的 2-范数都是 1） |
 * | 2 | line | `main` | 迭代向量在**前两个坐标**上的投影轨迹 `vector_x/vector_y` |
 * | 3 | scatter | `main` | 轨迹上的步点（每 `stride` 步一个，标出"步"的走向） |
 * | 4 | line | `main` | 当前向量（原点 → 当前投影，红色实线） |
 * | 5 | scatter | `main` | 当前向量的端点 |
 * | 6 | line | `aux` | 实测残差 `residual`（对数纵轴） |
 * | 7 | line | `aux` | 理论参考线 `theory_rate = |λ₂/λ₁|^k` |
 * | 8 | line | `aux` | 理论参考线 `theory_rate_sq = |λ₂/λ₁|^(2k)`（Rayleigh 商的二次收敛） |
 * | 9 | line | `aux` | 当前步游标（竖虚线） |
 * | 10 | scatter | `aux` | 当前步的残差点 `residual_k` |
 *
 * 角标槽位（固定，见 `_shared.ts` 的约定）：0 来源 / 1 主轴固定 / 2 副轴自适应 /
 * 3 `ratio` 临界（1.0）/ 4 状态（converge / plateau / 复主特征值）/ 5 前端预览（仅
 * `backend+preview`）/ 6 溢出 / 7 降级。
 *
 * 几条刻意的决定
 * --------------
 * 1. **主轴固定 ±1.05，绝不自适应**：manifest 的轴标题就写着这一条。迭代向量是单位向量，
 *    把它们画进一个随数据缩放的小窗口会让"收敛到主特征方向"变成一个**位置不变的点**——
 *    图上就再也看不出它有没有在动（PROTOCOLS §6.2 规则 1/4）。
 * 2. **画的是投影，不是整个向量**：帧里的 `vector_x/vector_y` 是 `n` 维迭代向量的前两个分量
 *    （`server/chapters/1.4.py` 的 `head = vec[:, :2]`）。`n > 2` 时 ‖投影‖ ≤ 1，
 *    所以点都落在单位圆**内**——这不是"没收敛到圆上"，而是投影的必然结果；
 *    面板文本必须写明，否则"点跑到圆里面"会被读成发散。
 * 3. **两条理论参考线都画**（规则 2）：`|λ₂/λ₁|^k` 是残差上界的形状，`|λ₂/λ₁|^(2k)` 是
 *    Rayleigh 商的（二次）收敛率。它们不是拟合，是帧直接给的数组；实测曲线与它们的落差
 *    正是"shear 造成的前几十步瞬态"。
 * 4. **`plateau` 要说出来**：`meta.converged = false` 时曲线停在平台上，图上必须写
 *    「未收敛」，而不是让读者把平台读成收敛（pitfalls 第三条）。
 * 5. **曲线标签贴着线的末端画**（两个渲染层都是 `last + (4, −2)`），所以标签必须短；
 *    一段曲线只有一个点时干脆不挂标签（否则三四条标签会叠在同一处）——长句子一律进面板文本。
 * 6. **`ratio` 的临界值 1.0**：`|λ₂/λ₁| = 1` 时幂法完全不收敛，滑杆上限 0.99 是有意的；
 *    临界角标由 `frame.critical.ratio` 给（缺了才回落到 manifest 的 1.0）。
 */

import { formatNumber } from "../../shared/src/axes.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import {
  ModelInputError,
  canvasSize,
  clamp,
  clipBadgeText,
  criticalOf,
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
  sampleLinear,
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
  previewBadge,
  scatterLayer,
  series,
  sourceBadge,
} from "../../shared/src/scene.js";
import { manifest } from "./manifest.gen.js";
import {
  LOG_FLOOR,
  applyOverflowNote,
  axisTitleOf,
  axisYlabelOf,
  belowAxis,
  channelRange,
  chapterLayout,
  fmtSci,
  integerTicks,
  logRange,
  panelBudget,
  pitfallsGuard,
  replaceFloors,
  textBlock,
  textBottom,
  textTop,
} from "./_shared.js";

const CHAPTER = "1.4";
const VIZ = "power_iteration";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "幂法对齐：向量序列与 Rayleigh 商收敛";
const AXES = SPEC?.axes;

/** 固定角标槽位（见 `model.ts::BADGE_LAYOUT`）。 */
const SLOT = { source: 0, axisMode: 1, auxAxis: 2, critical: 3, status: 4, preview: 5, overflow: 6, degraded: 7 } as const;

/** 主特征方向的固定窗口（manifest 轴标题写的就是 ±1.05）。 */
const WINDOW = 1.05;
/** `frame.critical.ratio` 缺失时的回落值（manifest 里 `critical = [1.0]`）。 */
const RATIO_CRITICAL = 1.0;
/** 少于这么多个点时，曲线画不出"线"，改画点并且不挂标签（标签会叠在一起）。 */
const MIN_LINE_POINTS = 3;

export const powerIterationModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需字段：缺一个就抛 ModelInputError（宁可当场炸，也不画半张图）——
  const circleX = requireArray(frame, "circle_x");
  const circleY = requireArray(frame, "circle_y");
  const vectorX = requireArray(frame, "vector_x");
  const vectorY = requireArray(frame, "vector_y");
  const stepGrid = requireArray(frame, "step_grid");
  const residual = requireArray(frame, "residual");
  const theory = requireArray(frame, "theory_rate");
  const theorySq = requireArray(frame, "theory_rate_sq");
  requireSameLength(circleX, circleY, "circle_x/circle_y");
  requireSameLength(vectorX, vectorY, "vector_x/vector_y");
  requireSameLength(stepGrid, residual, "step_grid/residual");
  requireSameLength(stepGrid, theory, "step_grid/theory_rate");
  requireSameLength(stepGrid, theorySq, "step_grid/theory_rate_sq");
  if (vectorX.length === 0) {
    throw new ModelInputError(`帧 ${CHAPTER}/${VIZ} 的 vector_x 是空数组：没有任何一步可画（拒绝画空图）`);
  }
  const pitfalls = requirePitfalls(frame);

  // —— 可选字段：缺了就降级（记进 degraded，最终挂成一个 status 角标）——
  const degraded: string[] = [];
  const tMeta = metaNumber(frame, "t");
  if (tMeta === undefined) degraded.push("帧 meta 缺 t：scene.t 改用请求 t（两者可能不同）");
  const k = metaNumber(frame, "step");
  if (k === undefined) degraded.push("帧 meta 缺 step：不画当前步游标（不猜是第几步）");
  const nSteps = metaNumber(frame, "n_steps");
  if (nSteps === undefined) degraded.push("帧 meta 缺 n_steps：不显示总步数");
  const ratio = metaNumber(frame, "ratio");
  if (ratio === undefined) degraded.push("帧 meta 缺 ratio：不显示 |λ₂/λ₁|（不重算）");
  const lam1 = metaNumber(frame, "lambda1");
  const lam2 = metaNumber(frame, "lambda2");
  const rateMeasured = metaNumber(frame, "rate_measured");
  const ratioMeasured = metaNumber(frame, "ratio_measured");
  const converged = metaNumber(frame, "converged");
  const complexPair = (metaNumber(frame, "complex_pair") ?? 0) > 0.5;
  const angleStep = metaNumber(frame, "angle_step");
  const rayleighK = metaNumber(frame, "rayleigh_k");
  const rayleighError = metaNumber(frame, "rayleigh_error");
  const residualK = metaNumber(frame, "residual_k");
  const residualMin = metaNumber(frame, "residual_min");
  const residualMax = metaNumber(frame, "residual_max");
  const transientRatio = metaNumber(frame, "transient_ratio");
  const status = stringOf(frame, "status");
  if (status === undefined) degraded.push("帧 strings 缺 status：状态角标不自行编造结论");
  const rateNote = stringOf(frame, "rate_note");
  const family = stringOf(frame, "family");
  if (family === undefined) degraded.push("帧 strings 缺 family：不显示矩阵族名（不按参数猜）");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: tMeta ?? opts.t,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  // —— 排版：左方形单位圆（+ 轴下三行图例），右列残差曲线 ——
  const guard = pitfallsGuard(width, height, pitfalls);
  const layout = chapterLayout(width, height, {
    footer: guard.footer,
    rightShares: [1],
    gutter: 62, // 右列的 y 刻度/轴名过道（无热力图，不需要给光栅器色标留 162）
    mainReserve: 100, // 单位圆下方留给图例（4×15 + 装饰带 28 + 余量）
  });
  const mainRect = layout.main;
  const auxRect = layout.right[0] as [number, number, number, number];
  const mainBudget = panelBudget(mainRect, layout.columns ? [auxRect] : []);
  const auxBudget = panelBudget(auxRect);

  scene.axes.push(
    makeAxis("main", mainRect, {
      xlim: [-WINDOW, WINDOW],
      ylim: [-WINDOW, WINDOW],
      xlabel: "$x₁$（前两个坐标上的投影）",
      ylabel: axisYlabelOf(AXES, "main", "x₂"),
      title: axisTitleOf(AXES, "main", "单位圆与迭代向量（固定坐标 ±1.05）"),
      autoscaled: false,
      xticks: [-1, -0.5, 0, 0.5, 1],
      yticks: [-1, -0.5, 0, 0.5, 1],
    }),
  );

  // 副轴（manifest 声明 autoscaled: true、scale: "log"）：残差与两条理论参考线共用一条 log 纵轴。
  // 范围**排除地板哨兵**（后端把 0 抬到 1e-300，见 `_shared.ts::LOG_FLOOR`），并锚上 1.0。
  const chan = channelRange(residual, [1]);
  const theoryChan = channelRange(theory, [1]);
  const auxYlim = logRange(Math.min(chan.lo, theoryChan.lo, 1e-16), Math.max(chan.hi, theoryChan.hi, 1));
  const lastStep = Math.max(1, stepGrid.length - 1);
  scene.axes.push(
    makeAxis("aux", auxRect, {
      xlim: [0, lastStep],
      ylim: auxYlim,
      yscale: "log",
      xlabel: "迭代步 $k$（不是物理时间；拖到头会被夹住）",
      ylabel: axisYlabelOf(AXES, "aux", "‖Ax−ρ(x)x‖ 与 |λ₂/λ₁|^k"),
      title: axisTitleOf(AXES, "aux", "残差与理论收敛率"),
      autoscaled: true,
      xticks: integerTicks(auxRect[2], stepGrid.length),
    }),
  );

  // —— 图层 1：单位圆（虚线，所有迭代向量的 2-范数都是 1）——
  scene.layers.push(lineLayer("main", series(circleX, circleY, pal.reference, { dash: [5, 4], width: 1.2, label: "单位圆" })));

  // —— 图层 2~3：投影轨迹 + 步点 ——
  scene.layers.push(lineLayer("main", series(vectorX, vectorY, pal.curves[0], { width: 1.8, label: "迭代向量轨迹" })));
  const stride = Math.max(1, Math.ceil(vectorX.length / 24));
  const dotX: number[] = [];
  const dotY: number[] = [];
  for (let i = 0; i < vectorX.length; i += stride) {
    dotX.push(vectorX[i] as number);
    dotY.push(vectorY[i] as number);
  }
  scene.layers.push(scatterLayer("main", dotX, dotY, { r: 2.2, color: pal.curves[0], alpha: 0.85, label: "迭代落点" }));

  // —— 图层 4~5：当前向量（原点 → 当前投影）——
  const idxNow = k === undefined ? vectorX.length - 1 : clamp(Math.round(k), 0, vectorX.length - 1);
  const cx = vectorX[idxNow] as number;
  const cy = vectorY[idxNow] as number;
  scene.layers.push(lineLayer("main", series([0, cx], [0, cy], pal.marker, { width: 2.2, label: "当前向量" })));
  scene.layers.push(scatterLayer("main", [cx], [cy], { r: 4.5, color: pal.marker, label: "当前迭代点" }));

  // —— 图层 6~8：残差与两条理论参考线（规则 2：两条斜率都要在图上）——
  const pointLike = residual.length < MIN_LINE_POINTS;
  if (pointLike) {
    degraded.push(`本帧只有 ${residual.length} 个迭代点：曲线画不出「线」，参考斜率要等 k≥${MIN_LINE_POINTS - 1}（把时间轴往右拖）`);
  }
  // 地板样本（后端把恰好为 0 抬到 1e-300）**换到轴底再画**：原值在对数轴上等于画在
  // 轴外，会被渲染层裁掉——那样"画在轴底"就成了空话。个数照旧记账（`resFloor`）。
  const resFloor = replaceFloors(residual, auxYlim[0]);
  const resSeries = segmentedSeries(stepGrid, resFloor.values, pal.residual, {
    width: 1.8,
    marker: pointLike ? "o" : "none",
    accept: (v) => v > 0,
    label: "残差",
  });
  if (resSeries.length > 0) scene.layers.push(lineLayer("aux", resSeries));
  else degraded.push("residual 里没有一个 > 0 的值：log 轴上画不出残差曲线（只在文本里报数）");

  const theorySeries = segmentedSeries(stepGrid, theory, pal.reference, {
    width: 1.4,
    dash: [6, 3],
    marker: pointLike ? "o" : "none",
    accept: (v) => v > 0,
    label: "理论收敛率",
  });
  if (theorySeries.length > 0) scene.layers.push(lineLayer("aux", theorySeries));
  const theorySqSeries = segmentedSeries(stepGrid, theorySq, pal.reference, {
    width: 1.2,
    dash: [2, 3],
    marker: pointLike ? "o" : "none",
    accept: (v) => v > 0,
    label: "理论收敛率（平方项）",
  });
  if (theorySqSeries.length > 0) scene.layers.push(lineLayer("aux", theorySqSeries));

  // —— 图层 9~10：当前步游标 + 残差点 ——
  if (k !== undefined) {
    const kk = clamp(Math.round(k), 0, stepGrid.length - 1);
    const xNow = stepGrid[kk] as number;
    scene.layers.push(lineLayer("aux", series([xNow, xNow], auxYlim, pal.cursor, { dash: [3, 3], width: 1, label: "当前步" })));
    const rNow = sampleLinear(stepGrid, residual, xNow);
    if (rNow !== undefined && rNow > 0) {
      scene.layers.push(scatterLayer("aux", [xNow], [rNow], { r: 4, color: pal.marker, label: "当前残差" }));
    }
  }

  // —— 文本：一行一个 `text` 层（`render2d.ts::drawText` 不拆 `\n`）——
  const summary = paramSummary(frame, opts, [
    { name: "n", label: "$n$" },
    { name: "ratio", label: "$|\\lambda_2/\\lambda_1|$" },
    { name: "shear", label: "shear" },
  ]);
  degraded.push(...summary.degraded);

  const mainLines: string[] = [
    `${summary.text}${family === undefined ? "" : `　族=${family}`}`,
    k === undefined
      ? "当前步：帧未给 meta.step"
      : `第 ${formatNumber(k, 4)} 步${nSteps === undefined ? "" : ` / 共 ${formatNumber(nSteps, 4)} 步`}（时间轴 = 迭代步）`,
    "画的是迭代向量在前两个坐标上的投影：n>2 时 ‖投影‖≤1，所以点落在单位圆内（不是没收敛）",
  ];

  const legend: string[] = [];
  if (lam1 !== undefined && lam2 !== undefined) {
    legend.push(
      `谱 |λ₁|=${formatNumber(lam1, 4)}、|λ₂|=${formatNumber(lam2, 4)}（实测残差比 ${ratioMeasured === undefined ? "?" : formatNumber(ratioMeasured, 4)}）` +
        (rayleighK === undefined ? "" : `；Rayleigh 商 ρ(x_k)=${formatNumber(rayleighK, 6)}、与 λ₁ 距离 ${rayleighError === undefined ? "?" : formatNumber(rayleighError, 3)}`),
    );
  }
  legend.push(`单位圆=灰虚线；投影轨迹=蓝折线（点=每 ${stride} 步）；当前向量=红；实测残差=实线；两条理论收敛率=点划线（副轴）`);
  legend.push("Rayleigh 商是 O(θ²) 而向量只有 O(θ)：残差先贴地板 ≠ 特征向量已经准");

  // 宽画布：图例排在单位圆下方（那条轴有 xlabel，装饰带占到 bottom+28）；窄画布：并进面板内，
  // 由 `textBlock` 按净空截断并写明"另有 N 行未画"——绝不让它压住下面那块面板的标题带。
  if (layout.columns) {
    scene.texts.push(...textBlock(mainRect[0] + 6, belowAxis(mainRect), guard.textLimit, legend, pal.muted, mainBudget, "top"));
  } else {
    mainLines.push(...legend);
  }
  scene.texts.push(...textBlock(mainRect[0] + 6, textTop(mainRect), textBottom(mainRect), mainLines, pal.ink, mainBudget, "top"));

  const auxLines: string[] = ["实线=实测残差；长虚线=理论 |λ₂/λ₁|^k；短虚线=理论 |λ₂/λ₁|^(2k)（两条斜率都画出来了）"];
  if (rateNote !== undefined) auxLines.push(`帧 rate_note：${rateNote}`);
  if (residualK !== undefined) {
    auxLines.push(
      `当前残差 ${fmtSci(residualK)}（帧 meta.residual_k）` +
        (residualMin !== undefined && residualMax !== undefined
          ? `；整段 min=${fmtSci(residualMin)}、max=${fmtSci(residualMax)}`
          : ""),
    );
  }
  if (transientRatio !== undefined && transientRatio > 100) {
    auxLines.push(
      `瞬态比 max/min = ${fmtSci(transientRatio)}：shear 越大，开头几十步先涨后落（渐近斜率仍是 |λ₂/λ₁|，定理没失效）`,
    );
  }
  if (converged !== undefined && converged < 0.5) {
    auxLines.push("⚠ 未收敛（plateau）：曲线停在平台上——这张图不会把平台画成收敛，别把「不动」读成「到了」");
  }
  if (complexPair) {
    auxLines.push(
      `⚠ 复主特征值：实向量在 span{v, v̄} 里旋转（每步夹角 ${angleStep === undefined ? "?" : formatNumber(angleStep, 4)} rad），幂法不收敛`,
    );
  }
  if (chan.nFloor > 0) {
    auxLines.push(`${resFloor.nFloor}/${residual.length} 个残差样本落在地板 ${fmtSci(LOG_FLOOR)} 上（后端把恰好为 0 抬到地板）：画在轴底并计入溢出说明`);
  }
  // 贴近面板底排（曲线在左上角落点、标签也在左上角，错开）
  scene.texts.push(...textBlock(auxRect[0] + 6, textTop(auxRect), textBottom(auxRect), auxLines, pal.muted, auxBudget, "bottom"));

  // —— 角标（固定槽位）——
  scene.badges.push(placeBadge(sourceBadge(scene.dataSource), SLOT.source, width));
  if (scene.dataSource.startsWith("backend+preview")) {
    scene.badges.push(placeBadge(previewBadge("拖动时按显示空间插值"), SLOT.preview, width));
  }
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(badge("autoscale", "副轴：自适应（log 纵轴按本帧残差与理论线取整；自适应会掩盖量级变化）", 8, 8), SLOT.auxAxis, width),
  );
  scene.badges.push(
    placeBadge(criticalBadge("ratio", ratio ?? Number.NaN, criticalOf(frame, "ratio", RATIO_CRITICAL), { beyond: "above" }), SLOT.critical, width),
  );
  const statusBits: string[] = [];
  if (status !== undefined) statusBits.push(status);
  if (complexPair) statusBits.push("复主特征值 → 不收敛");
  if (converged !== undefined) statusBits.push(converged > 0.5 ? "帧判 converged" : "帧判未收敛");
  if (rateMeasured !== undefined) statusBits.push(`实测残差比 ${formatNumber(rateMeasured, 4)}`);
  if (statusBits.length === 0) statusBits.push("帧未给出可判读的状态（预期有 strings.status 与 meta.converged）");
  scene.badges.push(placeBadge(badge("status", clipBadgeText(statusBits.join("；")), 8, 8), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);

  // 规则 1：图层越界 + 帧内非有限值 + 地板哨兵，三者都要进 overflow。
  const extra: string[] = [];
  if (chan.nFloor > 0) extra.push(`residual 有 ${chan.nFloor} 个样本落在地板 ${fmtSci(LOG_FLOOR)}（恰好为 0，后端抬高以便上 log 轴）`);
  if (theoryChan.nFloor > 0) extra.push(`theory_rate 有 ${theoryChan.nFloor} 个样本在地板上（ratio^k 下溢）`);
  const nBad =
    countNonFinite(circleX) + countNonFinite(circleY) + countNonFinite(vectorX) + countNonFinite(vectorY) + countNonFinite(residual);
  applyOverflowNote(scene, extra.join("；"), nBad, { slot: SLOT.overflow, width });
  return scene;
};
