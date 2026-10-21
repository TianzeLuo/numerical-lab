/**
 * `gmres_pseudospec`（1.4 章）的 model：GMRES 残差平台与伪谱形状（1.5 章前哨）。
 *
 * 这一页要当场挡住"只看谱的界"这个坑（pitfalls 第六条）：
 * 只依赖**谱**的 GMRES 下降界在这类非正规矩阵上一路**上升**（本帧 k=10 时 3.24e4，
 * 比 k=0 的 1 还大；pitfalls 里量到 +0.956/步），而实测残差单调掉到 7.7e-33
 * （−3.5/步量级）。两条线在同一张对数图上一升一降——"谱说会变糟、实际在变好"。
 *
 * 面板与图层（顺序即绘制顺序）
 * ----------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | heat | `main` | `log10_sigma`（magma，色阶固定 [−3, 0.5]；地板格点掩成灰） |
 * | 2..k | contour | `main` | 每个 ε 一条 `inside[j]` 的 0.5 等值线（当前 ε 高亮） |
 * | k+1 | scatter | `main` | `spectrum_x/spectrum_y`：谱（固定窗口 ±1.5，越界的点要报数） |
 * | … | line | `aux` | `residual`：GMRES 相对残差（requiredAux） |
 * | … | line | `aux` | `spectral_bound`：只依赖谱的多项式下降界（规则 2 的第二条斜率） |
 * | … | line | `aux` | `residual_floor`：帧给的对数轴地板（画在轴底，作为"再往下没有意义"的标尺） |
 * | … | line | `aux` | 当前步游标 |
 * | … | scatter | `aux` | `error_mark` = 当前残差 |
 *
 * 角标槽位：0 来源 / 1 主轴固定（**固定窗口**是这一页的结论之一）/ 2 副轴自适应 / 4 状态 / 6 溢出 / 7 降级。
 *
 * 几条刻意的决定
 * --------------
 * 1. **主轴固定 ±1.5 且不许跟着谱缩放**：manifest 的轴标题就写着"伪谱（固定窗口）与谱"。
 *    grcar-10 的谱 Im 最远到 ±2.1，会有几个谱点落在窗口外——这时**必须报数**
 *    （规则 1：被裁掉的东西要写进 overflow），而不是把窗口拉大把谱装进去。
 * 2. **`inside` 掩码整片为空是常态**：ε ≤ 1e-2 时 28×28 的网格（间距 0.107）常常一个格点都
 *    抓不到。这不是"伪谱是空的"，是"网格太粗"（pitfalls 第四条：网格分辨率系统性低估 ρ_ε）。
 *    图上要写出来，并且不许把空掩码画成"什么都没有"了事。
 * 3. **`residual_floor` 画在轴底**：帧给的是一个恒为 1e-300 的数组（对数轴地板）。照原值参与
 *    取范围会把纵轴拉到 300 个数量级；这里把它当"地板标尺"画在轴底并注明。
 * 4. **`residual` 里的 1e-300 也是地板**（`np.maximum(x, 1e-300)`），同样按"地板"报数，
 *    不参与纵向取范围。
 * 5. **没有 `trusted` 掩码数组**：这一页的帧只给了 `meta.trusted_fraction`。地板格点用
 *    `log10_sigma ≤ −299.5` 这个哨兵识别（对数轴上 −300 只可能是被抬起来的 0），
 *    识别出的个数如实报出，并在文本里说明"这一页的掩码来自哨兵，不是帧里的掩码字段"。
 */

import { countNonFinite } from "../../shared/src/geometry.js";
import {
  ModelInputError,
  canvasSize,
  clamp,
  clipBadgeText,
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
  sceneShell,
  segmentedSeries,
  stringOf,
  type SceneModel,
} from "../../shared/src/model.js";
import {
  axisModeBadge,
  badge,
  contourLayer,
  heatLayer,
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
  channelRange,
  chapterLayout,
  countMask,
  fmtSci,
  integerTicks,
  log10Slope,
  logRange,
  panelBudget,
  pitfallsGuard,
  textBlock,
  textBottom,
  textTop,
} from "./_shared.js";

const CHAPTER = "1.4";
const VIZ = "gmres_pseudospec";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "GMRES 残差平台与伪谱形状（1.5 章前哨）";
const AXES = SPEC?.axes;

const SLOT = { source: 0, axisMode: 1, auxAxis: 2, critical: 3, status: 4, preview: 5, overflow: 6, degraded: 7 } as const;

/** 热力图色阶（本帧 σ_min 的量级在 1e-2~1 之间，固定窄色阶才看得出形状）。 */
const HEAT_VMIN = -3;
const HEAT_VMAX = 0.5;
/** `log10_sigma` 上的地板哨兵：−300 只可能是"把 0 抬到 1e-300"。 */
const LOG_FLOOR_SENTINEL = -299.5;

export const gmresPseudospecModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需字段 ——
  const gx = requireArray(frame, "grid_x");
  const gy = requireArray(frame, "grid_y");
  const logSigma = requireArray(frame, "log10_sigma");
  const inside = requireArray(frame, "inside");
  const insideShape = frame.shapeOf("inside");
  const epsGrid = requireArray(frame, "epsilon_grid");
  const radius = requireArray(frame, "radius");
  const specX = requireArray(frame, "spectrum_x");
  const specY = requireArray(frame, "spectrum_y");
  const residual = requireArray(frame, "residual");
  const bound = requireArray(frame, "spectral_bound");
  const floorRef = requireArray(frame, "residual_floor");
  const errMark = requireArray(frame, "error_mark");
  const nx = gx.length;
  const ny = gy.length;
  if (logSigma.length !== nx * ny) {
    throw new ModelInputError(`帧 ${CHAPTER}/${VIZ} 的 log10_sigma 长度 ${logSigma.length} ≠ nx*ny = ${nx * ny}`);
  }
  requireSameLength(specX, specY, "spectrum_x/spectrum_y");
  requireSameLength(residual, bound, "residual/spectral_bound");
  requireSameLength(residual, floorRef, "residual/residual_floor");
  requireSameLength(epsGrid, radius, "epsilon_grid/radius");
  if (insideShape.length !== 3 || insideShape[1] !== ny || insideShape[2] !== nx) {
    throw new ModelInputError(`帧 ${CHAPTER}/${VIZ} 的 inside 形状 [${insideShape.join(", ")}] 与网格 (${ny}, ${nx}) 不一致`);
  }
  const nEps = insideShape[0] as number;
  if (residual.length < 1) {
    throw new ModelInputError(`帧 ${CHAPTER}/${VIZ} 的 residual 是空数组：没有任何迭代步可画（拒绝画空图）`);
  }
  const pitfalls = requirePitfalls(frame);

  // —— 可选字段 ——
  const degraded: string[] = [];
  const tMeta = metaNumber(frame, "t");
  if (tMeta === undefined) degraded.push("帧 meta 缺 t：scene.t 改用请求 t（两者可能不同）");
  const k = metaNumber(frame, "step");
  if (k === undefined) degraded.push("帧 meta 缺 step：不画当前步游标（不猜）");
  const nSteps = metaNumber(frame, "n_steps");
  const epsSel = metaNumber(frame, "epsilon");
  const epsIndex = metaNumber(frame, "epsilon_index");
  const converged = metaNumber(frame, "converged");
  const breakdown = (metaNumber(frame, "breakdown") ?? 0) > 0.5;
  const plateau = (metaNumber(frame, "plateau") ?? 0) > 0.5;
  const krylovDim = metaNumber(frame, "krylov_dim");
  const resCurrent = metaNumber(frame, "residual_current");
  const resMin = metaNumber(frame, "residual_min");
  const resFinal = metaNumber(frame, "residual_final");
  const maxDrop = metaNumber(frame, "max_drop_ratio");
  const boundMeta = metaNumber(frame, "spectral_bound");
  const boundGap = metaNumber(frame, "bound_gap");
  const radiusCurrent = metaNumber(frame, "radius_current");
  const spectralRadius = metaNumber(frame, "spectral_radius");
  const trustedFraction = metaNumber(frame, "trusted_fraction");
  const status = stringOf(frame, "status");
  if (status === undefined) degraded.push("帧 strings 缺 status：状态角标不自行编造结论");
  const note = stringOf(frame, "note");
  const family = stringOf(frame, "family");
  const rhs = stringOf(frame, "rhs");
  if (family === undefined || rhs === undefined) degraded.push("帧 strings 缺 family/rhs：不显示矩阵族与右端项（不按参数猜）");
  const win = metaNumber(frame, "window") ?? 1.5;

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: tMeta ?? opts.t,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  // —— 排版：左方形复平面（留 162px 色标过道）+ 右列残差曲线 ——
  const guard = pitfallsGuard(width, height, pitfalls);
  const layout = chapterLayout(width, height, {
    footer: guard.footer,
    rightShares: [1],
    gutter: 162,
    minRight: 240,
  });
  const mainRect = layout.main;
  const auxRect = layout.right[0] as [number, number, number, number];
  const mainBudget = panelBudget(mainRect, layout.columns ? [auxRect] : []);
  const auxBudget = panelBudget(auxRect);

  const lim = win > 0 ? win : 1.5;

  // 步数相关的东西先算出来：下面的图层构建（残差/界的可见窗口）都要用
  const kMax = Math.max(1, residual.length - 1);
  const kk = k === undefined ? kMax : clamp(Math.round(k), 0, kMax);
  // k=0 的帧只有 1 个样本：改画点并说明（拖动时间轴才看得到下降）。
  const pointLike = residual.length < 3;
  if (pointLike) degraded.push(`本帧只有 ${residual.length} 个残差样本（k=0）：曲线画不出「线」，改画点；下降要等 k≥2`);

  // —— 主面板：固定窗口 ——
  scene.axes.push(
    makeAxis("main", mainRect, {
      xlim: [-lim, lim],
      ylim: [-lim, lim],
      xlabel: "$\\operatorname{Re} z$",
      ylabel: axisYlabelOf(AXES, "main", "Im z"),
      title: axisTitleOf(AXES, "main", "伪谱（固定窗口）与谱"),
      autoscaled: false,
    }),
  );

  // 图层 1：σ_min 场。地板格点（哨兵 −300）掩成 NaN：这一页帧里没有 `trusted` 数组，
  // 只能靠哨兵识别——文本里写明这一点，别让人以为掩码来自帧字段。
  const masked = new Float64Array(logSigma.length);
  let nFloorCells = 0;
  for (let i = 0; i < logSigma.length; i += 1) {
    const v = logSigma[i] as number;
    if (!Number.isFinite(v) || v <= LOG_FLOOR_SENTINEL) {
      masked[i] = Number.NaN;
      nFloorCells += 1;
      continue;
    }
    // 上下界交给图层的 vmin/vmax（契约里写得明明白白），这里不偷偷再夹一次
    masked[i] = v;
  }
  scene.layers.push(
    heatLayer("main", masked, nx, ny, [-lim, lim, -lim, lim], {
      cmap: "magma",
      vmin: HEAT_VMIN,
      vmax: HEAT_VMAX,
      clipBelow: HEAT_VMIN,
    }),
  );

  // 图层 2..k：逐 ε 的伪谱边界（掩码整片为空是常态，如实报数）
  let nEmptyMasks = 0;
  const emptyEps: string[] = [];
  for (let j = 0; j < nEps; j += 1) {
    const slice = new Float64Array(nx * ny);
    for (let i = 0; i < nx * ny; i += 1) slice[i] = inside[j * nx * ny + i] as number;
    if (countMask(slice) === 0) {
      nEmptyMasks += 1;
      const e = epsGrid[j];
      emptyEps.push(e === undefined ? `#${j}` : fmtSci(e));
      continue;
    }
    const selected = epsIndex !== undefined && j === Math.round(epsIndex);
    scene.layers.push(
      contourLayer("main", slice, nx, ny, [-lim, lim, -lim, lim], {
        levels: [0.5],
        color: selected ? pal.marker : pal.reference,
        width: selected ? 2 : 1,
        label: selected ? `当前 ε=${fmtSci(epsGrid[j] as number)} 的伪谱边界` : undefined,
      }),
    );
  }

  // 图层：谱（固定窗口，越界点必须报数）
  let nSpecOutside = 0;
  for (let i = 0; i < specX.length; i += 1) {
    const sx = specX[i] as number;
    const sy = specY[i] as number;
    if (!Number.isFinite(sx) || !Number.isFinite(sy) || Math.abs(sx) > lim || Math.abs(sy) > lim) nSpecOutside += 1;
  }
  scene.layers.push(scatterLayer("main", specX, specY, { r: 3.6, color: pal.background, label: "谱 $Λ(A)$" }));

  // —— 副轴：GMRES 残差 vs 只依赖谱的界 ——
  // 纵向范围要把**界**也包进去（界常常比残差大几十个数量级），并排除两处地板。
  const boundSalt = boundMeta === undefined ? [] : [boundMeta];
  const resChan = channelRange(residual, [1, ...boundSalt]);
  const boundChan = channelRange(bound, [1]);
  const auxYlim = logRange(Math.min(resChan.lo, boundChan.lo, 1e-16), Math.max(resChan.hi, boundChan.hi, 1));
  scene.axes.push(
    makeAxis("aux", auxRect, {
      xlim: [0, kMax],
      ylim: auxYlim,
      yscale: "log",
      xlabel: "GMRES 迭代步 $k$",
      ylabel: axisYlabelOf(AXES, "aux", "‖r_k‖/‖r_0‖"),
      title: axisTitleOf(AXES, "aux", "GMRES 相对残差（对数）"),
      autoscaled: true,
      xticks: integerTicks(auxRect[2], residual.length),
    }),
  );

  const resOk = segmentedSeries(
    Array.from({ length: residual.length }, (_, i) => i),
    residual,
    pal.residual,
    { width: 2, marker: "o", accept: (v) => v > 1e-299 },
  );
  let nResFloor = 0;
  for (let i = 0; i < residual.length; i += 1) if ((residual[i] as number) <= 1e-299) nResFloor += 1;
  if (resOk.length > 0) scene.layers.push(lineLayer("aux", resOk));
  else degraded.push("residual 全部落在地板 1e-300 上：残差曲线画不出来（只在文本里报数）");
  const boundSeries = segmentedSeries(
    Array.from({ length: bound.length }, (_, i) => i),
    bound,
    pal.reference,
    { width: 1.4, dash: [7, 4], marker: pointLike ? "o" : "none", accept: (v) => v > 0 },
  );
  if (boundSeries.length > 0) scene.layers.push(lineLayer("aux", boundSeries));
  // 地板标尺：帧给的 `residual_floor` 恒为 1e-300，画在轴底（参与取范围会把纵轴拉到 300 个数量级）
  scene.layers.push(
    lineLayer("aux", series([0, kMax], [auxYlim[0], auxYlim[0]], pal.cursor, { dash: [2, 4], width: 1.2, label: "对数轴地板 $1e-300$" })),
  );
  scene.layers.push(lineLayer("aux", series([kk, kk], auxYlim, pal.cursor, { dash: [3, 3], width: 1 })));
  const markVal = errMark.length > 0 ? (errMark[0] as number) : Number.NaN;
  if (Number.isFinite(markVal) && markVal > 0) {
    scene.layers.push(scatterLayer("aux", [kk], [Math.max(markVal, auxYlim[0])], { r: 4, color: pal.marker }));
  }

  // —— 文本 ——
  const summary = paramSummary(frame, opts, [{ name: "n", label: "$n$" }]);
  degraded.push(...summary.degraded);

  const mainLines: string[] = [
    `${summary.text}　族=${family ?? "?"}　右端项=${rhs ?? "?"}　谱半径 ${spectralRadius === undefined ? "?" : fmtSci(spectralRadius)}` +
      (nSteps === undefined ? "" : `　本帧 ${Math.round(nSteps)} 步`),
    epsSel === undefined
      ? "当前 ε：帧未给 meta.epsilon"
      : `当前 ε=${fmtSci(epsSel)}：伪谱半径 ρ_ε=${radiusCurrent === undefined ? "?" : fmtSci(radiusCurrent)}` +
        (radiusCurrent !== undefined && radiusCurrent <= 0 ? "（网格上抓不到：不是「伪谱是空的」，是网格太粗）" : ""),
    `σ_min 场：${nFloorCells}/${logSigma.length} 个格点在地板 −300 上（已掩成灰格）` +
      (trustedFraction === undefined ? "" : `；帧 meta.trusted_fraction=${fmtSci(trustedFraction)}`) +
      "。注意：本页帧里没有 trusted 掩码数组，掩码来自 −300 哨兵",
    `等高线 = 逐 ε 的掩码边界（不是插值）${nEmptyMasks > 0 ? `：${nEmptyMasks}/${nEps} 个 ε 整片为空（${emptyEps.join("、")}；网格间距 ${fmtSci(2 * lim / Math.max(1, nx - 1))}）` : ""}`,
    nSpecOutside > 0
      ? `⚠ ${nSpecOutside}/${specX.length} 个谱点落在固定窗口 ±${fmtSci(lim)} 外（grcar 的谱 Im 能到 ±2.1）：窗口是固定的，不跟着谱缩放`
      : `谱全部落在固定窗口 ±${fmtSci(lim)} 内`,
  ];
  scene.texts.push(...textBlock(mainRect[0] + 6, textTop(mainRect), textBottom(mainRect), mainLines, pal.ink, mainBudget, "top"));

  const ks: number[] = [];
  const resWindow: number[] = [];
  const boundWindow: number[] = [];
  for (let i = 0; i <= kk; i += 1) {
    ks.push(i);
    resWindow.push(residual[i] as number);
    boundWindow.push(bound[i] as number);
  }
  const resSlope = log10Slope(ks, resWindow);
  const boundSlope = log10Slope(ks, boundWindow);
  const auxLines: string[] = [
    "实线+点 = 实测相对残差；长虚线 = 只依赖谱的多项式下降界（规则 2 的第二条线）；点线 = 对数轴地板",
    resCurrent === undefined
      ? "当前残差：帧未给 meta.residual_current"
      : `当前残差 ${fmtSci(resCurrent)}（整段最小 ${resMin === undefined ? "?" : fmtSci(resMin)}、末步 ${resFinal === undefined ? "?" : fmtSci(resFinal)}）`,
    boundMeta === undefined
      ? "帧未给 meta.spectral_bound：界的位置只能从数组读"
      : `只依赖谱的界 = ${fmtSci(boundMeta)}：实测残差比它小 ${boundGap === undefined ? "?" : fmtSci(boundGap)} 倍（界越大越没用）`,
    resSlope === undefined
      ? "实测 log10 斜率：点太少，算不出来（不编）"
      : `实测 log10 残差斜率 ${resSlope.toFixed(3)}/步` +
        (boundSlope === undefined ? "" : `；本帧里界是常数（斜率 ${boundSlope.toFixed(3)}/步）——拖动时间轴能看到它随 k 变大（pitfalls 量到约 +0.96/步）`),
  ];
  if (plateau) {
    auxLines.push("⚠ plateau：残差连续多步几乎不降——这是 1.5 章要展开的现象，不是收敛");
  }
  if (breakdown) {
    auxLines.push("帧标 breakdown=True：Krylov 空间提前耗尽（残差掉到机器精度后无处可去），不是算法崩了");
  }
  auxLines.push(
    (converged === undefined ? "" : converged > 0.5 ? "帧判 converged；" : "帧判未收敛；") +
      (krylovDim === undefined ? "" : `Krylov 维数 ${fmtSci(krylovDim, 0)}；`) +
      `最大单步下降比 ${maxDrop === undefined ? "?" : fmtSci(maxDrop)}` +
      (note === undefined ? "" : `；帧 note：${note}`),
  );
  if (status !== undefined) auxLines.push(`帧 status：${status}`);
  if (nResFloor > 0) auxLines.push(`${nResFloor}/${residual.length} 个残差样本在地板 1e-300 上：画在轴底，不计入纵向范围`);
  scene.texts.push(...textBlock(auxRect[0] + 6, textTop(auxRect), textBottom(auxRect), auxLines, pal.muted, auxBudget, "top"));

  // —— 角标 ——
  scene.badges.push(placeBadge(sourceBadge(scene.dataSource), SLOT.source, width));
  if (scene.dataSource.startsWith("backend+preview")) {
    scene.badges.push(placeBadge(previewBadge("拖动时按显示空间插值"), SLOT.preview, width));
  }
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(badge("autoscale", "副轴：自适应（log 纵轴要同时装下 1e-33 的残差与 3e4 的界；自适应会掩盖量级变化）", 8, 8), SLOT.auxAxis, width),
  );
  const statusBits: string[] = [];
  if (status !== undefined) statusBits.push(status);
  statusBits.push(
    `谱半径 ${spectralRadius === undefined ? "?" : fmtSci(spectralRadius)} vs 伪谱半径 ρ_ε=${radiusCurrent === undefined ? "?" : fmtSci(radiusCurrent)}`,
  );
  if (boundMeta !== undefined && resCurrent !== undefined) {
    statusBits.push(`只看谱的界 ${fmtSci(boundMeta)} 对实测残差 ${fmtSci(resCurrent)}`);
  }
  scene.badges.push(placeBadge(badge("status", clipBadgeText(statusBits.join("；")), 8, 8), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);

  // 规则 1：掩掉的格点、空掩码、越界谱点、地板样本、非有限值。
  const extra: string[] = [];
  if (nFloorCells > 0) extra.push(`main 的 σ_min 场有 ${nFloorCells}/${logSigma.length} 个格点在地板 −300 上，已掩成「无数据」灰格`);
  if (nEmptyMasks > 0) extra.push(`${nEmptyMasks}/${nEps} 个 ε 的 inside 掩码整片为空：没有等高线可画（网格分辨率，不是「伪谱为空」）`);
  if (nSpecOutside > 0) extra.push(`${nSpecOutside}/${specX.length} 个谱点落在固定窗口 ±${fmtSci(lim)} 外：被裁掉（窗口不跟着谱缩放）`);
  if (nResFloor > 0) extra.push(`residual 有 ${nResFloor}/${residual.length} 个样本是地板 ${fmtSci(LOG_FLOOR)}，画在轴底 ${fmtSci(auxYlim[0])}`);
  const nBad = countNonFinite(logSigma) + countNonFinite(residual) + countNonFinite(bound) + countNonFinite(specX) + countNonFinite(specY);
  applyOverflowNote(scene, extra.join("；"), nBad, { slot: SLOT.overflow, width });
  return scene;
};
