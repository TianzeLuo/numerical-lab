/**
 * `iteration_race`（1.5 章）的 model：κ 决定的椭圆等高线上，五条迭代轨迹赛跑。
 *
 * 面板与图层（顺序即绘制顺序，测试按此断言）
 * ------------------------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | line | `plane` | `contour_x/contour_y` 的 6 层椭圆等高线（每行一条闭合曲线） |
 * | 2 | line | `plane` | `b_vec` 折线（帧里存 [x0, x1, y0, y1]，即原点到 b） |
 * | 3 | line | `plane` | 5 条轨迹（**按 `race_len` 截断**，不画重复末点；当前方法加粗） |
 * | 4 | scatter | `plane` | 精确解 x*（`x_star`） |
 * | 5 | scatter | `plane` | 当前高亮方法在第 k 步的位置（`traj_mark_x/y`） |
 * | 6 | line | `convergence` | 每个方法一条相对残差曲线（截到第 k 步） |
 * | 7 | line | `convergence` | 理论 ρ^k 与实测因子^k 两条参考线（当前方法） |
 * | 8 | line | `convergence` | 第 k 步的游标竖线 |
 * | 9 | scatter | `convergence` | 第 k 步五条残差的快照点 |
 *
 * 角标槽位（固定，见 `SLOT`）：0 来源 / 1 主轴固定 / 2 副轴自适应 / 3 κ 临界 /
 * 4 ω 临界 / 5 状态 / 6 溢出 / 7 降级。
 *
 * 几条刻意的决定
 * --------------
 * 1. **主轴固定且等比例**（manifest 声明 `plane.autoscaled = false`）：xlim 与 ylim 的
 *    **跨度逐位相等**，面板是正方形，范围由本帧的等高线/轨迹/解/b 的并集按 1-2-5 取整。
 *    三件事缺一不可——只要 x/y 各自自适应，κ=200 的扁椭圆就会被拉成正圆，
 *    「条件数」这个几何事实当场消失（本章 pitfalls 第 1 条）。范围只由**帧内固定数组**
 *    （等高线与整段轨迹）算出，所以同一 params 下逐帧不变，拖动时间轴时画面不抖。
 * 2. **重复末点按 `race_len` 截断**：后端把每条轨迹补齐到 61 步用的是"重复最后一个点"
 *    （不是 NaN，见 `server/chapters/1.5.py::_pad`）。直接画整行会在图的右下角堆出一团
 *    "不动的点"，读起来像"方法卡住了"；按 `meta.race_len` 截断后，**每条轨迹真实的步数**
 *    才是画面里看到的长度。收敛后的重复点不画，而 `race_len` 的数值写在面板文本里。
 * 3. **当前方法的残差曲线用 `residual`（requiredAux），其余方法用 `race_residual`**：
 *    两者在后端是同一行数据（`residual = race_res[idx][:k+1]`），分开用是为了让
 *    "规则 3 的误差/残差同屏"落在声明的那个数组上，同时不重复画同一条曲线。
 *    注意 `residual` 是**已补齐后的前缀**：方法在第 2 步收敛时，第 3...k 步的值是重复点，
 *    所以还要再按 `meta.n_iter_current` 截一刀，否则会画出一段假的"平台"。
 * 4. **参考线的斜率来自帧 meta，不由数据重算**：ρ^k 的 ρ 取 `meta.rho_jacobi/rho_gs/rho_sor`
 *    （后端解析式），实测因子取 `meta.factor_measured`（后端尾部拟合）。前端只做"在对数轴上
 *    画一条斜率给定的直线"这件显示几何，并且**逐点裁到轴范围内**——折线图层不被渲染器裁剪，
 *    越界值会（正确地）触发溢出角标。
 * 5. **对数轴上的 0 与负值断开并点名**：残差恰为 0 的步在 log 轴上没有定义，
 *    `segmentedSeries` 的 `accept: v > 0` 会把它们断开（不是缺数据），断点数写进面板文本。
 * 6. **ω 的临界徽标是手写的**：`frame.critical.omega = [1.0]` 说的是"ω=1 退化为 GS"，
 *    而发散阈值是 `meta.threshold_omega = 2`——两面都是边界，用 `criticalBadge` 的单向
 *    `beyond` 语义会把 ω=1.3 误报成"已越界"。所以这里自己拼一枚 `kind="critical"` 徽标，
 *    把两个阈值与真实越界方向写清楚（κ 那枚仍然走 `criticalBadge(beyond="below")`）。
 */

import { formatNumber, niceLimits, niceTicks } from "../../shared/src/axes.js";
import type { Frame } from "../../shared/src/frame.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import {
  resolveTextOverlaps,
  ModelInputError,
  applyOverflow,
  canvasSize,
  criticalOf,
  dataSourceOf,
  degradedBadge,
  finitePositiveMinMax,
  metaNumber,
  optionalArray,
  palette,
  placeBadge,
  requireArray,
  requireDrawableLayers,
  requirePitfalls,
  sceneShell,
  segmentedSeries,
  stackedPanels,
  stringOf,
  textWidthUnits,
  type Rect,
  type SceneModel,
} from "../../shared/src/model.js";
import {
  pitfallsBand,
  axisModeBadge,
  badge,
  criticalBadge,
  lineLayer,
  makeAxis,
  scatterLayer,
  series,
  sourceBadge,
  text,
  type Badge,
  type TextSpec,
} from "../../shared/src/scene.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "1.5";
const VIZ = "iteration_race";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "迭代路径赛跑：椭圆等高线上的五条轨迹";
/** manifest 里 `kappa.critical = [1.0]`、`beyond = "below"`；`frame.critical.kappa` 缺失时的回落值。 */
const KAPPA_CRITICAL = 1.0;
/** manifest 里 `omega.critical = [1.0]`；发散阈值优先取 `meta.threshold_omega`（后端给 2）。 */
const OMEGA_DEGENERATE = 1.0;
const DEFAULT_OMEGA_THRESHOLD = 2.0;

/** 固定角标槽位（见 `model.ts::BADGE_LAYOUT`）。本帧数据源恒为 backend，故不占 preview 槽。 */
const SLOT = {
  source: 0,
  axisMode: 1,
  auxAxis: 2,
  criticalKappa: 3,
  criticalOmega: 4,
  status: 5,
  overflow: 6,
  degraded: 7,
} as const;

/** 五个方法在帧里的行序（`race_x/race_y/race_residual` 的行、`race_len` 的下标）。 */
const METHOD_KEYS = ["jacobi", "gs", "sor", "cg", "gmres"] as const;
/** 后端 `_method_name` 的回落名（`strings.race_order` 缺失时才用）。 */
const METHOD_FALLBACK = ["Jacobi", "Gauss–Seidel", "SOR", "CG（共轭梯度）", "GMRES(2)"] as const;
/** 五个方法的颜色（显示用；theme 只影响这一张表与背景）。 */
const RACE_COLORS: Record<"light" | "dark", readonly string[]> = {
  light: ["#2b6cb0", "#2f855a", "#b7791f", "#6b46c1", "#c53030"],
  dark: ["#63b3ed", "#68d391", "#f6ad55", "#d6bcfa", "#fc8181"],
};
const CONTOUR_NOTE = "等高线 6 层 (x−x*)^T·A·(x−x*)=c，c 外→内递减";
/** 曲线颜色顺序（与 `RACE_COLORS` 一一对应）：光栅器把 series 的 label 画在曲线末点，
 *  五条曲线末点几乎重合，所以不用末点标签当图例，改用这一行的文字图例。 */
const RACE_LEGEND = "颜色：Jacobi 蓝、GS 绿、SOR 黄、CG 紫、GMRES 红，当前加粗";

export const iterationRaceModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);
  const colors = RACE_COLORS[opts.theme === "dark" ? "dark" : "light"];

  // —— 必需数组：缺一个就抛 ModelInputError（宁可当场炸，也不画半张图）——
  const contourX = frameMatrix(frame, "contour_x");
  const contourY = frameMatrix(frame, "contour_y");
  const raceX = frameMatrix(frame, "race_x");
  const raceY = frameMatrix(frame, "race_y");
  const raceResidual = frameMatrix(frame, "race_residual");
  const residual = requireArray(frame, "residual");
  const stepGrid = requireArray(frame, "step_grid");
  if (contourX.rows !== contourY.rows || contourX.cols !== contourY.cols) {
    throw new ModelInputError(
      `帧 ${CHAPTER}/${VIZ} 的 contour_x 形状 ${contourX.rows}×${contourX.cols} 与 contour_y ` +
        `${contourY.rows}×${contourY.cols} 不一致：等高线必须逐点配对（拒绝画错位的椭圆）`,
    );
  }
  if (raceX.rows !== raceY.rows || raceX.cols !== raceY.cols || raceResidual.rows !== raceX.rows || raceResidual.cols !== raceX.cols) {
    throw new ModelInputError(
      `帧 ${CHAPTER}/${VIZ} 的 race_x/race_y/race_residual 形状不一致：` +
        `${raceX.rows}×${raceX.cols} / ${raceY.rows}×${raceY.cols} / ${raceResidual.rows}×${raceResidual.cols}`,
    );
  }
  if (residual.length > stepGrid.length) {
    throw new ModelInputError(
      `帧 ${CHAPTER}/${VIZ} 的 residual 长度 ${residual.length} 超过 step_grid 的 ${stepGrid.length}：` +
        "残差前缀必须落在步网格之内（拒绝画错位的曲线）",
    );
  }
  const pitfalls = requirePitfalls(frame);

  // —— 可选字段：缺了就降级（记进 degraded，最终挂成一个 status 角标）——
  const degraded: string[] = [];
  const tMeta = metaNumber(frame, "t");
  if (tMeta === undefined) degraded.push("帧 meta 缺 t：scene.t 改用请求 t（两者可能不同）");
  const tNow = tMeta ?? opts.t;
  const nMethods = raceX.rows;
  const k = metaNumber(frame, "step");
  if (k === undefined) degraded.push("帧 meta 缺 step：不画第 k 步游标与当前点标记（不猜是第几步）");
  const nSteps = metaNumber(frame, "n_steps");
  if (nSteps === undefined) degraded.push("帧 meta 缺 n_steps：不显示总步数");
  // 时间轴夹取：`opts.t` 与帧里的真实步 `meta.step` 不一致时如实写出（别假装 t 就是索引）。
  if (k !== undefined && Number.isFinite(opts.t) && Math.round(opts.t) !== Math.round(k)) {
    degraded.push(
      `请求 t=${formatNumber(opts.t, 4)} 夹到第 ${formatNumber(k, 4)} 步：帧给的是四舍五入并夹到 [0, n_steps] 的整数步`,
    );
  }
  const methodIdxRaw = metaNumber(frame, "method_index");
  const methodIdx = methodIdxRaw !== undefined && methodIdxRaw >= 0 && methodIdxRaw < nMethods ? methodIdxRaw : undefined;
  if (methodIdx === undefined && methodIdxRaw !== undefined) {
    degraded.push(`帧 meta.method_index=${formatNumber(methodIdxRaw, 4)} 越界（共 ${nMethods} 个方法）：不加粗任何轨迹`);
  }
  if (methodIdxRaw === undefined) degraded.push("帧 meta 缺 method_index：不画当前方法的加粗轨迹与当前点标记");
  const methodName = stringOf(frame, "method_name") ?? stringOf(frame, "method");
  if (methodName === undefined) degraded.push("帧 strings 缺 method/method_name：只按行序称呼五个方法");
  const raceOrder = stringOf(frame, "race_order");
  const labels = raceLabels(raceOrder, nMethods);
  if (raceOrder === undefined) degraded.push("帧 strings 缺 race_order：方法名用本章的固定表");
  const raceLen = optionalArray(frame, "race_len");
  if (raceLen === undefined) {
    degraded.push("帧 race_len 缺失：轨迹无法截断，会画出收敛后的重复末点（图上是一团不动的点）");
  }
  const markX = optionalArray(frame, "traj_mark_x");
  const markY = optionalArray(frame, "traj_mark_y");
  const xStar = optionalArray(frame, "x_star");
  if (xStar === undefined) degraded.push("帧 x_star 缺失：不画精确解标记（不猜解在哪）");
  const bVec = optionalArray(frame, "b_vec");
  if (bVec === undefined) degraded.push("帧 b_vec 缺失：不画 b 的折线");
  const note = stringOf(frame, "note");
  if (note === undefined) degraded.push("帧 strings 缺 note：状态角标不自行编造结论");

  const kappa = metaNumber(frame, "kappa");
  const kappaActual = metaNumber(frame, "kappa_actual");
  const omega = metaNumber(frame, "omega");
  const omegaOpt = metaNumber(frame, "omega_opt");
  const muJacobi = metaNumber(frame, "mu_jacobi");
  const rhoJacobi = metaNumber(frame, "rho_jacobi");
  const rhoGs = metaNumber(frame, "rho_gs");
  const rhoSor = metaNumber(frame, "rho_sor");
  const rhoSorMeasured = metaNumber(frame, "rho_sor_measured");
  const thresholdOmega = metaNumber(frame, "threshold_omega") ?? DEFAULT_OMEGA_THRESHOLD;
  const factorMeasured = metaNumber(frame, "factor_measured");
  const nIterCurrent = metaNumber(frame, "n_iter_current");
  const convergedCurrent = (metaNumber(frame, "converged_current") ?? 0) > 0.5;
  if (nIterCurrent === undefined) degraded.push("帧 meta 缺 n_iter_current：当前方法的残差曲线不在真实步数处截断");
  const paramsJson = paramsJsonOf(frame);
  if (stringOf(frame, "params_json") === undefined) degraded.push("帧 strings 缺 params_json：θ 等请求侧参数不显示（不猜）");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: tNow,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  // —— 排版：左侧正方形物理平面，右侧残差曲线 + 文本块 ——
  // 正方形是"等比例"的前提（见文件头第 1 条）：矩形面板配等跨度轴也会把圆画成椭圆。
  const base = stackedPanels(width, height);
  const contentH = base.bottom - base.top;
  const side = Math.max(140, Math.min(Math.round(base.contentWidth * 0.46), contentH));
  const planeRect: Rect = [base.left, base.top, side, side];
  const rightX = base.left + side + 58;
  const rightW = Math.max(180, base.left + base.contentWidth - rightX);
  // 面板只占 0.52：右下角的文本块要放下"这些数字是从哪来的 + 怎么看"，比多 30px 曲线更重要。
  const convH = Math.max(120, Math.round(contentH * 0.52));
  const convRect: Rect = [rightX, base.top, rightW, convH];

  // 主轴：x/y 同跨度（`equalSpanLimits`），范围只由帧内固定数组算出。
  const bounds = planeBounds(contourX, contourY, raceX, raceY, xStar, bVec);
  const { xlim, ylim, ticks } = equalSpanLimits(bounds, Math.max(3, Math.round(side / 90)));
  const planeAxis = makeAxis("plane", planeRect, {
    xlim,
    ylim,
    xlabel: "$x1$",
    ylabel: axisYlabel("plane", "$x_2$"),
    title: axisTitle("plane", "等高线与迭代轨迹（物理坐标，固定轴）"),
    autoscaled: false,
    xticks: ticks,
    yticks: ticks,
  });

  // 副轴（manifest 声明 `convergence.autoscaled = true`、`scale = "log"`）：
  // 范围按**整段 60 步**的全部残差取，不随 t 变（否则动画会抖）。
  const resPos = finitePositiveMinMax(raceResidual.values);
  const curPos = finitePositiveMinMax(residual);
  const lo = Math.min(resPos?.[0] ?? Number.POSITIVE_INFINITY, curPos?.[0] ?? Number.POSITIVE_INFINITY);
  const hi = Math.max(resPos?.[1] ?? Number.NEGATIVE_INFINITY, curPos?.[1] ?? Number.NEGATIVE_INFINITY);
  const hasPositive = Number.isFinite(lo) && Number.isFinite(hi);
  const convYlim: [number, number] = hasPositive ? niceLimits(lo, hi, 7, { log: true, pad: 0.2 }) : [1e-16, 1];
  if (!hasPositive) {
    degraded.push("race_residual/residual 里没有 >0 的有限值：副轴用占位范围 [1e-16, 1]，且不画曲线");
  }
  const stepMax = Math.max(1, nSteps ?? raceX.cols - 1);
  const convAxis = makeAxis("convergence", convRect, {
    xlim: [0, stepMax],
    ylim: convYlim,
    yscale: "log",
    xlabel: "迭代步 $k$",
    ylabel: axisYlabel("convergence", "相对残差"),
    title: axisTitle("convergence", "各方法的相对残差 ‖r_k‖/‖r_0‖"),
    autoscaled: true,
    xticks: niceTicks(0, stepMax, Math.max(3, Math.round(convRect[2] / 90))),
  });
  scene.axes.push(planeAxis, convAxis);

  // —— 图层 1：6 层椭圆等高线（每行一条闭合参数曲线）——
  const contourSeries: ReturnType<typeof series>[] = [];
  for (let i = 0; i < contourX.rows; i += 1) {
    // 不给 label：光栅器（`tools/raster.py`）把 series 的 label 画在**曲线末点**上，
    // 六条闭合椭圆各标一次会糊在椭圆上；这一句说明改由面板文本承担（见 CONTOUR_NOTE）。
    contourSeries.push(series(rowOf(contourX, i), rowOf(contourY, i), pal.reference, { width: 1 }));
  }
  scene.layers.push(lineLayer("plane", contourSeries));

  // —— 图层 2：b 的折线（帧里存 [x0, x1, y0, y1]：前一半是 x、后一半是 y，见文件头）——
  if (bVec !== undefined && bVec.length >= 4) {
    scene.layers.push(
      lineLayer(
        "plane",
        series([bVec[0], bVec[1]], [bVec[2], bVec[3]], pal.exact, {
          width: 1.4,
          dash: [5, 3],
          label: endLabel(bVec[1], planeAxis.xlim, planeRect[2], "b（帧 b_vec）"),
        }),
      ),
    );
  }

  // —— 图层 3：五条轨迹（按 race_len 截断；当前方法加粗）——
  const realSteps: number[] = [];
  for (let i = 0; i < nMethods; i += 1) {
    const nReal = raceLen === undefined ? raceX.cols - 1 : clampInt(raceLen[i], 0, raceX.cols - 1);
    realSteps.push(nReal);
    const end = nReal + 1;
    const current = methodIdx === i;
    scene.layers.push(
      lineLayer(
        "plane",
        series(rowSlice(raceX, i, end), rowSlice(raceY, i, end), colors[i % colors.length] as string, {
          width: current ? 2.6 : 1.1,
          alpha: current ? 1 : 0.75,
          // 只标当前方法：五条轨迹都收敛到 x* 附近，五个末点标签会叠成一团。
          label: current
            ? endLabel(rowSlice(raceX, i, end)[nReal] ?? Number.NaN, planeAxis.xlim, planeRect[2], `${labels[i]}（当前，${nReal} 步）`)
            : undefined,
        }),
      ),
    );
  }

  // —— 图层 4~5：精确解与当前步的位置 ——
  if (xStar !== undefined && xStar.length >= 2) {
    scene.layers.push(scatterLayer("plane", [xStar[0]], [xStar[1]], { r: 4.5, color: pal.exact, label: "精确解 $x*$" }));
  }
  if (methodIdx !== undefined && markX !== undefined && markY !== undefined && markX.length > 0 && markY.length > 0) {
    scene.layers.push(
      scatterLayer("plane", [markX[0]], [markY[0]], {
        r: 4,
        color: pal.marker,
        label: `${labels[methodIdx] ?? "当前方法"} 在第 ${formatNumber(k ?? 0, 3)} 步`,
      }),
    );
  }

  // —— 图层 6~9：副轴（规则 3：残差与主图同屏；规则 2：理论/实测两条参考线都画）——
  const cut = k === undefined ? stepGrid.length - 1 : clampInt(k, 0, stepGrid.length - 1);
  const zeroDropped: string[] = [];
  for (let i = 0; i < nMethods; i += 1) {
    const current = methodIdx === i;
    // 当前方法用 requiredAux 的 `residual`（还要再按真实步数截一刀，见文件头第 3 条）；
    // 其余方法用 `race_residual` 的对应行。
    let ys: ArrayLike<number>;
    let n: number;
    if (current) {
      const nReal = nIterCurrent === undefined ? raceX.cols - 1 : clampInt(nIterCurrent, 0, raceX.cols - 1);
      n = clampInt(Math.min(cut, nReal), 0, residual.length - 1) + 1;
      ys = residual;
    } else {
      const nReal = raceLen === undefined ? raceX.cols - 1 : clampInt(raceLen[i], 0, raceX.cols - 1);
      n = clampInt(Math.min(cut, nReal), 0, raceX.cols - 1) + 1;
      ys = rowOf(raceResidual, i);
    }
    const take = Math.min(n, ys.length);
    const xs = prefixOf(stepGrid, Math.min(take, stepGrid.length));
    const yv = prefixOf(ys, take);
    const seriesList = segmentedSeries(xs, yv, colors[i % colors.length] as string, {
      width: current ? 2.4 : 1,
      label: current
        ? endLabel(xs[xs.length - 1] ?? Number.NaN, convAxis.xlim, convRect[2], `${labels[i] ?? "当前方法"} 残差（帧 residual）`)
        : undefined,
      accept: (v) => v > 0,
    });
    const dropped = yv.length - countPositive(yv);
    if (dropped > 0) zeroDropped.push(`${labels[i] ?? `方法${i + 1}`} ${dropped}`);
    if (seriesList.length > 0) scene.layers.push(lineLayer("convergence", seriesList));
  }

  // 两条参考线（当前方法）：理论 ρ^k 与实测因子^k；斜率全部取自帧 meta。
  const refLines: string[] = [];
  if (methodIdx !== undefined) {
    const rhoTheory = theoryRho(methodIdx, { rhoJacobi, rhoGs, rhoSor });
    if (rhoTheory === undefined) {
      refLines.push(
        methodIdx === 3 || methodIdx === 4
          ? "参考线：CG/GMRES 没有定常衰减因子（至多 n 步精确终止），不画 ρ^k 直线"
          : "参考线：帧 meta 缺该方法的 ρ，不画理论直线（不猜）",
      );
    } else {
      const line = decayLine(rhoTheory, stepMax, convYlim[0]);
      if (line.x.length >= 2) {
        const refSeries = [
          series(line.x, line.y, pal.reference, {
            width: 1.2,
            dash: [6, 4],
            label: endLabel(line.x[line.x.length - 1] ?? Number.NaN, convAxis.xlim, convRect[2], `理论 ρ^k（${formatNumber(rhoTheory, 4)}）`),
          }),
        ];
        const meas = factorMeasured === undefined || !(factorMeasured > 0) ? undefined : decayLine(factorMeasured, stepMax, convYlim[0]);
        if (meas !== undefined && meas.x.length >= 2) {
          refSeries.push(
            series(meas.x, meas.y, pal.note, {
              width: 1.2,
              dash: [2, 3],
              label: endLabel(
                meas.x[meas.x.length - 1] ?? Number.NaN,
                convAxis.xlim,
                convRect[2],
                `实测因子^k（${formatNumber(factorMeasured as number, 4)}）`,
              ),
            }),
          );
        }
        scene.layers.push(lineLayer("convergence", refSeries));
        refLines.push(
          `参考线：理论 ρ^k（${formatNumber(rhoTheory, 4)}）、` +
            (meas !== undefined && meas.x.length >= 2
              ? `点线 = 实测因子^k（${formatNumber(factorMeasured as number, 4)}）`
              : "实测因子线不画（帧 meta.factor_measured 为 null 或不足两点）"),
        );
      } else {
        refLines.push(`参考线：ρ=${formatNumber(rhoTheory, 4)} 的 ρ^k 在本轴范围内不足两点，不画`);
      }
    }
  }

  // 游标 + 第 k 步快照。
  const snapshotX: number[] = [];
  const snapshotY: number[] = [];
  for (let i = 0; i < nMethods; i += 1) {
    const v = raceResidual.values[i * raceX.cols + cut];
    if (Number.isFinite(v) && v > 0) {
      snapshotX.push(cut);
      snapshotY.push(v);
    }
  }
  if (snapshotX.length > 0) {
    scene.layers.push(
      lineLayer("convergence", [series([cut, cut], [convYlim[0], convYlim[1]], pal.cursor, { dash: [3, 3], width: 1 })]),
    );
    scene.layers.push(
      scatterLayer("convergence", snapshotX, snapshotY, { r: 3.2, color: pal.marker, label: "第 $k$ 步五条残差的快照" }),
    );
  }

  // —— 文本（一行一个 text 层：render2d 不拆 \n）——
  const summary = summaryLines(paramsJson, {
    labels,
    realSteps,
    methodIdx,
    kappa,
    kappaActual,
    omega,
    omegaOpt,
    muJacobi,
    rhoJacobi,
    rhoGs,
    rhoSor,
    rhoSorMeasured,
    factorMeasured,
    nIterCurrent,
    convergedCurrent,
    note,
    zeroDropped,
  });
  const budget = unitBudget(rightW);
  const textTop = convRect[1] + convRect[3] + 28; // 让开副轴自己的刻度带(15)+轴名带(12)
  // 顺序 = 重要性：先数字（这些数从哪来），再读图辅助（参考线/图例/等高线）。
  // 这一块只有 10 行，排后面的会被丢掉，所以图例不能挤掉 race_len。
  scene.texts.push(
    ...textBlock(rightX, textTop, base.bottom - 4, [
      ...summary.filter((l) => !l.startsWith("收敛后的重复末点")),
      ...refLines,
      CONTOUR_NOTE,
      RACE_LEGEND,
      "收敛后的重复末点已按 race_len 截断：轨迹长度 = 真实步数",
    ], pal.muted, budget, "top"),
  );

  // —— 角标（固定槽位）——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  scene.badges.push(
    placeBadge(badge("autoscale", `${axisModeBadge(false).text}（plane：x/y 同跨度等比例）`, 8, 8), SLOT.axisMode, width),
  );
  scene.badges.push(
    placeBadge(
      badge("autoscale", "副轴 convergence：自适应（log 纵轴按整段残差取整；自适应会掩盖量级变化）", 8, 8),
      SLOT.auxAxis,
      width,
    ),
  );
  if (kappaActual !== undefined || kappa !== undefined) {
    scene.badges.push(
      placeBadge(
        criticalBadge("κ", (kappaActual ?? kappa) as number, criticalOf(frame, "kappa", KAPPA_CRITICAL), { beyond: "below" }),
        SLOT.criticalKappa,
        width,
      ),
    );
  }
  if (omega !== undefined) {
    scene.badges.push(placeBadge(omegaBadge(omega, thresholdOmega, criticalOf(frame, "omega", OMEGA_DEGENERATE)), SLOT.criticalOmega, width));
  }
  const status: string[] = [];
  if (methodName !== undefined) status.push(`当前方法 ${methodName}`);
  if (k !== undefined) {
    status.push(nSteps === undefined ? `第 ${formatNumber(k, 4)} 步` : `第 ${formatNumber(k, 4)}/${formatNumber(nSteps, 4)} 步`);
  }
  if (nIterCurrent !== undefined) {
    status.push(convergedCurrent ? `已在第 ${formatNumber(nIterCurrent, 4)} 步收敛` : `跑满 ${formatNumber(nIterCurrent, 4)} 步仍未收敛`);
  }
  if (note !== undefined) status.push(note);
  if (status.length === 0) status.push("帧未给出可判读的状态（预期有 meta.step 与 strings.note）");
  scene.badges.push(placeBadge(badge("status", status.join("；"), 8, 8), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  // 规则 1：图层越界（checkOverflow）+ 帧内非有限值，两者都要进 overflow 与角标。
  const nBad =
    countNonFinite(contourX.values) +
    countNonFinite(contourY.values) +
    countNonFinite(raceX.values) +
    countNonFinite(raceY.values) +
    countNonFinite(raceResidual.values) +
    countNonFinite(residual) +
    countNonFinite(stepGrid);
  applyOverflow(scene, nBad, { slot: SLOT.overflow, width });
  // 底部 pitfalls 面板（规则 7）由渲染层现画，模型看不见它：用共享的 pitfallsBand 拿到
  // 面板顶边，再让 resolveTextOverlaps 把放不下的文字如实计数后丢掉（窄画布下才生效）。
  const panelTop = pitfallsBand(scene)?.[1];
  scene.texts = resolveTextOverlaps(scene.texts, [8, 24, width - 16, (panelTop ?? height) - 28]);

  return scene;
};

// --------------------------------------------------------------------------- //
// 局部工具
// --------------------------------------------------------------------------- //

interface Matrix2D {
  rows: number;
  cols: number;
  values: ArrayLike<number>;
}

/** 二维数组（行主序）：缺数组抛 `ModelInputError`；形状不是二维也抛。 */
function frameMatrix(frame: Frame, name: string): Matrix2D {
  const values = requireArray(frame, name);
  const shape = frame.shapeOf(name);
  if (shape.length !== 2) {
    throw new ModelInputError(
      `帧 ${CHAPTER}/${VIZ} 的 '${name}' 形状 [${shape.join(", ")}] 不是二维：本图按「行 = 一条曲线」消费`,
    );
  }
  const rows = shape[0] as number;
  const cols = shape[1] as number;
  if (values.length !== rows * cols) {
    throw new ModelInputError(`帧 ${CHAPTER}/${VIZ} 的 '${name}' 有 ${values.length} 个元素，与 ${rows}×${cols} 不符`);
  }
  return { rows, cols, values };
}

/** 第 i 行的前 `end` 个元素（`[0, end)`）。 */
function rowSlice(m: Matrix2D, i: number, end: number): Float64Array {
  const n = Math.max(0, Math.min(end, m.cols));
  const out = new Float64Array(n);
  const base = i * m.cols;
  for (let j = 0; j < n; j += 1) out[j] = m.values[base + j] as number;
  return out;
}

/** 第 i 行整体。 */
function rowOf(m: Matrix2D, i: number): ArrayLike<number> {
  return rowSlice(m, i, m.cols);
}

/** 一维数组的前 n 个元素。 */
function prefixOf(a: ArrayLike<number>, n: number): Float64Array {
  const len = Math.max(0, Math.min(n, a.length));
  const out = new Float64Array(len);
  for (let i = 0; i < len; i += 1) out[i] = a[i] as number;
  return out;
}

function countPositive(a: ArrayLike<number>): number {
  let n = 0;
  for (let i = 0; i < a.length; i += 1) if ((a[i] as number) > 0) n += 1;
  return n;
}

function clampInt(v: number, lo: number, hi: number): number {
  const r = Math.round(v);
  if (!Number.isFinite(r)) return lo;
  return r < lo ? lo : r > hi ? hi : r;
}

/** `strings.race_order`（"Jacobi / Gauss–Seidel / SOR / CG（共轭梯度） / GMRES(2)"）→ 名字表。 */
function raceLabels(order: string | undefined, n: number): string[] {
  if (order !== undefined) {
    const parts = order
      .split(" / ")
      .map((s) => s.trim())
      .filter((s) => s !== "");
    if (parts.length === n) return parts;
  }
  return Array.from({ length: n }, (_, i) => METHOD_FALLBACK[i] ?? METHOD_KEYS[i] ?? `方法${i + 1}`);
}

/** `strings.params_json`（后端写的规范化参数）→ 对象；解析失败给空对象（不猜）。 */
function paramsJsonOf(frame: Frame): Record<string, number | string | boolean> {
  const raw = stringOf(frame, "params_json");
  if (raw === undefined) return {};
  try {
    const v: unknown = JSON.parse(raw);
    return typeof v === "object" && v !== null ? (v as Record<string, number | string | boolean>) : {};
  } catch {
    return {};
  }
}

/** 等跨度轴范围：xlim 与 ylim 的**宽度逐位相等**（主轴等比例的定义，见文件头第 1 条）。 */
function equalSpanLimits(
  b: { x0: number; x1: number; y0: number; y1: number },
  target: number,
): { xlim: [number, number]; ylim: [number, number]; ticks: number[] } {
  const cx = (b.x0 + b.x1) / 2;
  const cy = (b.y0 + b.y1) / 2;
  const span = Math.max(b.x1 - b.x0, b.y1 - b.y0, 1e-6) * 1.08;
  const [x0, x1] = niceLimits(cx - span / 2, cx + span / 2, target);
  const w = x1 - x0;
  const ylim: [number, number] = [cy - w / 2, cy + w / 2];
  return { xlim: [x0, x1], ylim, ticks: niceTicks(x0, x1, target) };
}

/** 主轴的显示范围：等高线 + 整段轨迹 + 解 + b 的并集（全部来自帧，不随 t 变化）。 */
function planeBounds(
  contourX: Matrix2D,
  contourY: Matrix2D,
  raceX: Matrix2D,
  raceY: Matrix2D,
  xStar: ArrayLike<number> | undefined,
  bVec: ArrayLike<number> | undefined,
): { x0: number; x1: number; y0: number; y1: number } {
  let x0 = Number.POSITIVE_INFINITY;
  let x1 = Number.NEGATIVE_INFINITY;
  let y0 = Number.POSITIVE_INFINITY;
  let y1 = Number.NEGATIVE_INFINITY;
  const eat = (x: number, y: number): void => {
    if (!Number.isFinite(x) || !Number.isFinite(y)) return;
    if (x < x0) x0 = x;
    if (x > x1) x1 = x;
    if (y < y0) y0 = y;
    if (y > y1) y1 = y;
  };
  for (let i = 0; i < contourX.values.length; i += 1) eat(contourX.values[i] as number, contourY.values[i] as number);
  for (let i = 0; i < raceX.values.length; i += 1) eat(raceX.values[i] as number, raceY.values[i] as number);
  if (xStar !== undefined && xStar.length >= 2) eat(xStar[0] as number, xStar[1] as number);
  if (bVec !== undefined && bVec.length >= 4) {
    eat(bVec[0] as number, bVec[2] as number);
    eat(bVec[1] as number, bVec[3] as number);
  }
  if (!Number.isFinite(x0)) return { x0: 0, x1: 1, y0: 0, y1: 1 };
  return { x0, x1, y0, y1 };
}

/** 当前方法的理论谱半径：Jacobi μ、GS μ²、SOR 的 Young 关系（全部来自帧 meta）。 */
function theoryRho(methodIdx: number, rho: { rhoJacobi?: number; rhoGs?: number; rhoSor?: number }): number | undefined {
  if (methodIdx === 0) return rho.rhoJacobi;
  if (methodIdx === 1) return rho.rhoGs;
  if (methodIdx === 2) return rho.rhoSor;
  return undefined;
}

/** 对数副轴上的 ρ^k 直线：逐点裁到轴内（折线不被渲染器裁剪，越界会触发溢出角标）。 */
function decayLine(rho: number, kMax: number, floor: number): { x: number[]; y: number[] } {
  const x: number[] = [];
  const y: number[] = [];
  if (!Number.isFinite(rho) || !(rho > 0)) return { x, y };
  for (let k = 0; k <= kMax; k += 1) {
    const v = Math.pow(rho, k);
    if (!Number.isFinite(v) || v < floor) break;
    x.push(k);
    y.push(v);
  }
  return { x, y };
}

/**
 * ω 的两面性徽标（手写，理由见文件头第 6 条）。
 *
 * `ω = 1` 是"退化为 GS"的**等式**边界，`ω ≥ threshold` 才是"发散"的**越过**边界；
 * `criticalBadge` 的单向语义无法同时表达这两个方向，所以这里自己拼 `kind="critical"`。
 */
function omegaBadge(omega: number, threshold: number, degenerate: number | number[]): Badge {
  const deg = Array.isArray(degenerate) ? (degenerate[0] ?? OMEGA_DEGENERATE) : degenerate;
  const parts = [`ω = ${formatNumber(omega, 4)}`, `ω=${formatNumber(deg, 4)} 退化为 GS`, `ω≥${formatNumber(threshold, 4)} 发散`];
  let warn = false;
  if (omega >= threshold) {
    parts.push("! 已越过发散阈值");
    warn = true;
  } else if (Math.abs(omega - deg) < 1e-9) {
    parts.push("（恰在退化点）");
    warn = true;
  }
  const b = badge("critical", parts.join("　"), 8, 8);
  b.color = warn ? "#c53030" : "#2f855a";
  return b;
}

/** 面板文本（一行一个 `text` 层；渲染层不拆 `\n`）。 */
function summaryLines(
  paramsJson: Record<string, number | string | boolean>,
  m: {
    labels: string[];
    realSteps: number[];
    methodIdx?: number;
    kappa?: number;
    kappaActual?: number;
    omega?: number;
    omegaOpt?: number;
    muJacobi?: number;
    rhoJacobi?: number;
    rhoGs?: number;
    rhoSor?: number;
    rhoSorMeasured?: number;
    factorMeasured?: number;
    nIterCurrent?: number;
    convergedCurrent: boolean;
    note?: string;
    zeroDropped: string[];
  },
): string[] {
  // 每一行都按「面板宽度 ÷ 7px」的单位预算裁过：这块地方只有 10 行，
  // 超出的行会被 `textBlock` 直接丢掉（丢掉的会被点名为「另有 N 行」）。
  const out: string[] = [];
  const head: string[] = [];
  if (m.kappaActual !== undefined || m.kappa !== undefined) {
    const requested =
      m.kappaActual !== undefined && m.kappa !== undefined && Math.abs(m.kappaActual - m.kappa) > 1e-9
        ? `（请求 ${formatNumber(m.kappa, 4)}）`
        : "";
    head.push(`κ=${formatNumber((m.kappaActual ?? m.kappa) as number, 4)}${requested}`);
  }
  if (m.omega !== undefined) head.push(`ω=${formatNumber(m.omega, 4)}`);
  if (m.omegaOpt !== undefined) head.push(`ω_opt=${formatNumber(m.omegaOpt, 4)}`);
  if (m.muJacobi !== undefined) head.push(`μ_Jacobi=${formatNumber(m.muJacobi, 4)}`);
  const angle = paramsJson["angle"];
  if (typeof angle === "number") head.push(`θ=${formatNumber(angle, 4)}°`);
  if (head.length > 0) out.push(head.join("　"));

  const rho: string[] = [];
  if (m.rhoJacobi !== undefined) rho.push(`Jacobi ${formatNumber(m.rhoJacobi, 4)}`);
  if (m.rhoGs !== undefined) rho.push(`GS ${formatNumber(m.rhoGs, 4)}`);
  if (m.rhoSor !== undefined) rho.push(`SOR ${formatNumber(m.rhoSor, 4)}`);
  if (rho.length > 0) out.push(`ρ(T)：${rho.join("、")}`);
  // ρ_sor_measured 与参考线里的「实测因子」是同一个数，不重复占行。

  const short = (name: string): string =>
    name.replace("（共轭梯度）", "").replace("Gauss–Seidel", "GS").replace("GMRES(2)", "GMRES");
  const steps = m.labels.map((name, i) => `${short(name)} ${m.realSteps[i] ?? "?"}`).join("、");
  out.push(`真实步数（meta.race_len）：${steps}`);

  if (m.methodIdx !== undefined) {
    const name = m.labels[m.methodIdx];
    out.push(
      `当前 ${name ?? "方法"}：` +
        (m.nIterCurrent === undefined ? "帧没给真实步数" : `${formatNumber(m.nIterCurrent, 4)} 步`) +
        `，${m.convergedCurrent ? "已收敛" : "未收敛"}` +
        (m.factorMeasured !== undefined ? `，实测因子 ${formatNumber(m.factorMeasured, 4)}` : ""),
    );
  }
  if (m.zeroDropped.length > 0) {
    out.push(`log 轴上恰为 0：${m.zeroDropped.join("、")}——无定义、曲线断开（不是缺数据）`);
  }
  if (m.note !== undefined) out.push(`帧 note：${m.note}`);
  return out;
}

/** manifest 里该轴的标题（文案直接取 manifest，缺省才用回落值）。 */
function axisTitle(id: string, fallback: string): string {
  return SPEC?.axes.find((a) => a.id === id)?.title ?? fallback;
}

/** manifest 里该轴的 ylabel。 */
function axisYlabel(id: string, fallback: string): string {
  return SPEC?.axes.find((a) => a.id === id)?.ylabel ?? fallback;
}

/**
 * 曲线标签现在只进网页的 HTML 图例：显示层会把 label 从图层里取走，
 * 不在画布末端绘制，所以不必再按剩余像素决定要不要保留。
 */
function endLabel(_xLast: number, _xlim: readonly [number, number], _panelW: number, label: string): string {
  return label;
}

/** 面板宽度 → 显示宽度单位预算（size 11 时 1 单位 ≈ 7px）。 */
function unitBudget(panelWidthPx: number): number {
  return Math.max(24, Math.floor((panelWidthPx - 12) / 7));
}

/** 一段说明文字 → 若干 `text` 层，**一行一个**（`render2d.ts::drawText` 不拆 `\n`）。 */
function textBlock(
  x: number,
  top: number,
  bottom: number,
  lines: readonly string[],
  color: string,
  budget: number,
  anchor: "top" | "bottom",
): TextSpec[] {
  const size = 11;
  const rowH = 15;
  // 折行后做一次「短行拼接」：把装得下的相邻短句并进同一行。
  // 不做这一步的话，一句 20 单位的短句会独占一行，白白吃掉一行的高度。
  let pieces: string[] = [];
  for (const body of lines) {
    for (const part of wrapUnits(body, budget)) {
      const last = pieces[pieces.length - 1];
      if (last !== undefined && textWidthUnits(last) + 1 + textWidthUnits(part) <= budget) {
        pieces[pieces.length - 1] = `${last}　${part}`;
      } else {
        pieces.push(part);
      }
    }
  }
  if (bottom > top) {
    const maxRows = Math.max(1, Math.floor((bottom - top) / rowH) + 1);
    if (pieces.length > maxRows) {
      const kept = Math.max(1, maxRows - 1);
      pieces = [...pieces.slice(0, kept), `...（另有 ${pieces.length - kept} 行，画布放不下）`];
    }
  }
  const y = anchor === "top" ? top : Math.max(top, bottom - (pieces.length - 1) * rowH);
  return pieces.map((body, i) => text(x, y + i * rowH, body, { role: "note", size, color, maxWidth: budget * 7 }));
}

/** 按显示宽度单位折行（CJK=2、ASCII=1；口径同 `model.ts::textWidthUnits`）。 */
function wrapUnits(body: string, budget: number): string[] {
  const out: string[] = [];
  let line = "";
  let used = 0;
  const push = (): void => {
    if (line !== "") out.push(line);
    line = "";
    used = 0;
  };
  // 折行点：空白、顿号/逗号、句读、右括号之后（CJK 长句没有空格，只在空白处折会把
  // 「v0(0.297, 0.493)」这种坐标从中间劈开）。
  for (const word of body.split(/(?<=[\s　、，,。；;：:）)])/)) {
    const w = textWidthUnits(word);
    if (used > 0 && used + w > budget) push();
    if (w <= budget) {
      line += word;
      used += w;
      continue;
    }
    let piece = "";
    let pieceUsed = 0;
    for (const ch of word) {
      const cw = textWidthUnits(ch);
      if (pieceUsed + cw > budget && piece !== "") {
        out.push(piece);
        piece = "";
        pieceUsed = 0;
      }
      piece += ch;
      pieceUsed += cw;
    }
    line = piece;
    used = pieceUsed;
  }
  push();
  return out;
}
