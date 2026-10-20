/**
 * `pseudospectrum`（1.4 章）的 model：ε-伪谱云雾（σ_min 的等高线）+ 伪谱半径。
 *
 * 这一页是本章的旗舰，它要当场挡住本章第二个大坑：
 * **σ_min 在非正规/病态矩阵上不可信**（pitfalls 第二条：同一格点上本实现与 LAPACK 差 10³~10⁶ 倍，
 * 而且 LAPACK 自己也在浮点地板以下）。所以这一页**不允许**把 σ_min 当"一个数"来读：
 * 帧里带了 `trusted` 掩码（1 = 没掉进 √u·‖A‖ 的舍入地板），模型据此把不可信格点
 * **掩成"无数据"**（NaN → 渲染层的灰格）并**逐个报数**，绝不画成"σ_min 很小"。
 *
 * 面板与图层（顺序即绘制顺序）
 * ----------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | heat | `main` | `log10_sigma`（magma，色阶固定 [−16, 0]；不可信格点 = NaN 灰格） |
 * | 2..k | contour | `main` | 每个 ε 一条 `inside[j]` 的 0.5 等值线（当前 ε 高亮，其余灰） |
 * | k+1 | scatter | `main` | `spectrum_x/spectrum_y`：谱（真特征值） |
 * | k+2 | scatter | `main` | 不可信格点的中心（灰方块，与文本里的计数一一对应） |
 * | … | line/scatter | `aux` | `radius_ratio`+`radius_ratio_exact`（ρ>0）**或** `radius`（ρ=0 时改画绝对值） |
 * | … | heat | `err` | `error = max(σ_min − dist(z,Λ), 0)` 的场（帧里恒为地板，见下） |
 *
 * 角标槽位：0 来源 / 1 主轴固定 / 2 副轴自适应 / 4 状态 / 6 溢出 / 7 降级。
 *
 * 几条刻意的决定
 * --------------
 * 1. **主轴窗口固定 ±1.5**（manifest 的轴标题、帧的 `meta.window`）：自适应会把"云雾扩散"
 *    抹平成"图还是那么大"（PROTOCOLS §6.2 规则 1/4）。
 * 2. **色阶固定 [−16, 0]**：同理。自适应色阶下"jordan 整片亮 / normal 纹丝不动"这个对比
 *    在动画里会消失。
 * 3. **热力图与等高线的采样约定差半格**：热力图的 `extent` 是外框（第 i 格中心在
 *    `lo+(i+0.5)Δ`），等高线走 `grid_coord`（节点铺满 `[lo,hi]`）。帧的 `grid_x/grid_y`
 *    是**节点**坐标，所以 `extent` 取 `[-1.5,1.5]` 让**等高线与谱点对齐**（它们是这一页的
 *    主角），热力图整体偏半格（0.048，约 4px）——这是渲染层既有约定，模型层改不了，
 *    在这里写明以免被当成模型的错位 bug。
 * 4. **`error` 字段是信息量为零的**：`error = max(σ_min − dist(z,Λ), 0)`，而
 *    `σ_min(A−zI) ≤ dist(z,Λ(A))` 是恒等式，所以它必然恒为 0（帧里上 log 轴时抬到 1e-300）；
 *    只有 `normal` 族会有几百个格点顶到 2e-16 的舍入尺度。前端照帧画（这是必须的：
 *    requiredAux 里有它），但必须在图上说清"整块贴地板不是图坏了，是这个字段本身没信息"，
 *    并把它作为后端字段问题上报（见交付报告）。
 * 5. **ρ(A)=0 时副轴换口径**：manifest 的副轴是"ρ_ε / ρ(A)"，而 jordan/shifted_jordan 族
 *    的 ρ(A) 恰为 0——分母为零，帧里 8 个比值全是地板 1e-300。这不是"比值为 0"，
 *    是"比值没有定义"。此时改画绝对半径 ρ_ε（并在角标/文本/轴名里三处说明），
 *    否则默认帧的副轴会是一条贴在地板上的直线，等于什么都没画。
 */

import { countNonFinite } from "../../shared/src/geometry.js";
import {
  ModelInputError,
  canvasSize,
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
  countAbove,
  countMask,
  fmtSci,
  log10Slope,
  logRange,
  panelBudget,
  pitfallsGuard,
  replaceFloors,
  textBlock,
  textBottom,
  textTop,
} from "./_shared.js";

const CHAPTER = "1.4";
const VIZ = "pseudospectrum";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "ε-伪谱云雾：σ_min(A−zI) 的等高线";
const AXES = SPEC?.axes;

/** 固定角标槽位。 */
const SLOT = { source: 0, axisMode: 1, auxAxis: 2, critical: 3, status: 4, preview: 5, overflow: 6, degraded: 7 } as const;

/** 热力图色阶（固定，跨帧可比）。 */
const HEAT_VMIN = -16;
const HEAT_VMAX = 0;
/** `error` 场的色阶：地板 -300 夹到 -17，这样 `normal` 族那些 1e-16 的格点才看得见。 */
const ERR_VMIN = -17;

export const pseudospectrumModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需字段 ——
  const gx = requireArray(frame, "grid_x");
  const gy = requireArray(frame, "grid_y");
  const logSigma = requireArray(frame, "log10_sigma");
  const trusted = requireArray(frame, "trusted");
  const inside = requireArray(frame, "inside");
  const insideShape = frame.shapeOf("inside");
  const epsGrid = requireArray(frame, "epsilon_grid");
  const radius = requireArray(frame, "radius");
  const ratio = requireArray(frame, "radius_ratio");
  const ratioExact = requireArray(frame, "radius_ratio_exact");
  const specX = requireArray(frame, "spectrum_x");
  const specY = requireArray(frame, "spectrum_y");
  const errField = requireArray(frame, "error");
  const epsMark = requireArray(frame, "epsilon_mark");
  const radiusMark = requireArray(frame, "radius_mark");
  const nx = gx.length;
  const ny = gy.length;
  if (logSigma.length !== nx * ny) {
    throw new ModelInputError(`帧 ${CHAPTER}/${VIZ} 的 log10_sigma 长度 ${logSigma.length} ≠ nx*ny = ${nx * ny}`);
  }
  requireSameLength(trusted, logSigma, "trusted/log10_sigma");
  requireSameLength(specX, specY, "spectrum_x/spectrum_y");
  requireSameLength(epsGrid, ratio, "epsilon_grid/radius_ratio");
  requireSameLength(epsGrid, radius, "epsilon_grid/radius");
  if (errField.length !== nx * ny) {
    throw new ModelInputError(`帧 ${CHAPTER}/${VIZ} 的 error 长度 ${errField.length} ≠ nx*ny = ${nx * ny}（这一页要把它按场画出来）`);
  }
  if (insideShape.length !== 3 || insideShape[1] !== ny || insideShape[2] !== nx) {
    throw new ModelInputError(
      `帧 ${CHAPTER}/${VIZ} 的 inside 形状 [${insideShape.join(", ")}] 与网格 (${ny}, ${nx}) 不一致：等高线画不出来`,
    );
  }
  const nEps = insideShape[0] as number;
  const pitfalls = requirePitfalls(frame);

  // —— 可选字段 ——
  const degraded: string[] = [];
  const tMeta = metaNumber(frame, "t");
  if (tMeta === undefined) degraded.push("帧 meta 缺 t：scene.t 改用请求 t（两者可能不同）");
  const rho = metaNumber(frame, "rho");
  if (rho === undefined) degraded.push("帧 meta 缺 rho：不知道谱半径，副轴只能用比值口径（不重算 ρ）");
  const epsSel = metaNumber(frame, "epsilon");
  const epsIndex = metaNumber(frame, "epsilon_index");
  const radiusCurrent = metaNumber(frame, "radius_current");
  const ratioCurrent = metaNumber(frame, "radius_ratio_current");
  const sigmaMin = metaNumber(frame, "sigma_min");
  const sigmaMax = metaNumber(frame, "sigma_max");
  const kappaMax = metaNumber(frame, "kappa_max");
  const trustedFraction = metaNumber(frame, "trusted_fraction");
  const nNoise = metaNumber(frame, "n_noise_points");
  const nPoints = metaNumber(frame, "n_points");
  const window = metaNumber(frame, "window");
  const kreissLower = metaNumber(frame, "kreiss_lower_bound");
  const status = stringOf(frame, "status");
  if (status === undefined) degraded.push("帧 strings 缺 status：状态角标不自行编造结论");
  const family = stringOf(frame, "family");
  const method = stringOf(frame, "method");
  if (family === undefined) degraded.push("帧 strings 缺 family：不显示矩阵族名（不按参数猜）");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: tMeta ?? opts.t,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  // —— 排版：左方形复平面（留 162px 色标过道）+ 右列两块（比值/半径、误差场）——
  const guard = pitfallsGuard(width, height, pitfalls);
  const layout = chapterLayout(width, height, {
    footer: guard.footer,
    rightShares: [1.25, 1],
    gutter: 162,
    minRight: 220,
  });
  const mainRect = layout.main;
  const auxRect = layout.right[0] as [number, number, number, number];
  const errRect = layout.right[1] as [number, number, number, number];
  const mainBudget = panelBudget(mainRect, layout.columns ? [auxRect, errRect] : []);
  const auxBudget = panelBudget(auxRect);
  const errBudget = panelBudget(errRect);

  const win = window ?? 1.5;
  const lim = win > 0 ? win : 1.5;

  // —— 主面板：固定窗口 ±window ——
  scene.axes.push(
    makeAxis("main", mainRect, {
      xlim: [-lim, lim],
      ylim: [-lim, lim],
      xlabel: "Re z",
      ylabel: axisYlabelOf(AXES, "main", "Im z"),
      title: axisTitleOf(AXES, "main", "复平面（固定窗口）"),
      autoscaled: false,
    }),
  );

  // 图层 1：σ_min 场。不可信格点掩成 NaN（两个渲染层都画"无数据"灰）——这是"按掩码处理"，
  // 不是"丢掉"：个数会在文本、角标、overflow 里各出现一次。
  const masked = new Float64Array(logSigma.length);
  let nUntrusted = 0;
  for (let i = 0; i < logSigma.length; i += 1) {
    const v = logSigma[i] as number;
    const ok = (trusted[i] ?? 0) > 0.5;
    if (!ok) {
      masked[i] = Number.NaN;
      nUntrusted += 1;
      continue;
    }
    masked[i] = Number.isFinite(v) ? Math.max(v, HEAT_VMIN) : Number.NaN;
  }
  scene.layers.push(
    heatLayer("main", masked, nx, ny, [-lim, lim, -lim, lim], {
      cmap: "magma",
      vmin: HEAT_VMIN,
      vmax: HEAT_VMAX,
      clipBelow: HEAT_VMIN,
    }),
  );

  // 图层 2..k：逐 ε 的伪谱边界（0.5 等值线）。当前 ε 高亮，其余灰。
  // **不插值**：每条线都是"σ_min ≤ ε 且格点可信"的掩码边界；掩码整片为空时没有线可画，如实报数。
  let nEmptyMasks = 0;
  const emptyEps: string[] = [];
  for (let j = 0; j < nEps; j += 1) {
    const slice = new Float64Array(nx * ny);
    for (let i = 0; i < nx * ny; i += 1) slice[i] = inside[j * nx * ny + i] as number;
    const hits = countMask(slice);
    if (hits === 0) {
      nEmptyMasks += 1;
      const e = epsGrid[j];
      emptyEps.push(e === undefined ? `#${j}` : fmtSci(e));
      continue;
    }
    const selected = epsIndex !== undefined ? j === Math.round(epsIndex) : false;
    scene.layers.push(
      contourLayer("main", slice, nx, ny, [-lim, lim, -lim, lim], {
        levels: [0.5],
        color: selected ? pal.marker : pal.reference,
        width: selected ? 2 : 1,
        label: selected ? `当前 ε=${fmtSci(epsGrid[j] ?? Number.NaN)} 的伪谱边界` : undefined,
      }),
    );
  }

  // 图层：谱（真特征值）
  scene.layers.push(scatterLayer("main", specX, specY, { r: 3.6, color: pal.background, label: "谱 Λ(A)" }));

  // 图层：不可信格点（掩掉的那些）——让"哪些地方不能读"看得见
  if (nUntrusted > 0) {
    const ux: number[] = [];
    const uy: number[] = [];
    for (let iy = 0; iy < ny; iy += 1) {
      for (let ix = 0; ix < nx; ix += 1) {
        if ((trusted[iy * nx + ix] ?? 0) > 0.5) continue;
        ux.push(gx[ix] as number);
        uy.push(gy[iy] as number);
      }
    }
    scene.layers.push(scatterLayer("main", ux, uy, { r: 2.4, color: pal.muted, alpha: 0.95, label: "不可信格点" }));
    if (ux.length !== nUntrusted) {
      degraded.push(`trusted 掩码里有 ${nUntrusted} 个不可信格点，但只标出 ${ux.length} 个：两者不一致（帧自身矛盾）`);
    }
  }

  // —— 副轴：ρ(A) > 0 时是"ρ_ε/ρ(A)"，否则改画绝对半径 ρ_ε ——
  const ratioUsable = countAbove(ratio, 1e-299);
  const useAbsolute = rho === undefined || rho <= 0 || ratioUsable === 0;
  const epsX = Array.from({ length: epsGrid.length }, (_, i) => epsGrid[i] as number);
  const logEps = epsX.map((e) => Math.log10(e));
  const ratioChan = channelRange(ratio, [1]);
  const radiusChan = channelRange(radius, [1]);
  const auxYlim = useAbsolute ? logRange(radiusChan.lo, radiusChan.hi) : logRange(ratioChan.lo, ratioChan.hi);
  const auxTitle = axisTitleOf(AXES, "aux", "伪谱半径与谱半径之比");
  scene.axes.push(
    makeAxis("aux", auxRect, {
      xlim: [Math.min(...epsX) / 2, Math.max(...epsX) * 2],
      ylim: auxYlim,
      xscale: "log",
      yscale: "log",
      xlabel: "ε（对数；越小越难算）",
      ylabel: useAbsolute
        ? "ρ_ε（本帧 ρ(A)=0：比值无定义，改画绝对值）"
        : axisYlabelOf(AXES, "aux", "ρ_ε / ρ(A)"),
      title: auxTitle,
      autoscaled: true,
    }),
  );

  let measuredSlope: number | undefined;
  if (useAbsolute) {
    // ρ(A)=0（jordan/shifted_jordan）或比值整条在地板上：改画绝对半径。
    const raw = replaceFloors(radius, Number.NaN);
    const okX: number[] = [];
    const okY: number[] = [];
    let nZero = 0;
    for (let i = 0; i < raw.values.length; i += 1) {
      const v = raw.values[i] as number;
      if (!Number.isFinite(v) || v <= 0) {
        nZero += 1;
        continue;
      }
      okX.push(epsX[i] as number);
      okY.push(v);
    }
    const segs = segmentedSeries(okX, okY, pal.curves[0], { width: 1.8, marker: "o", label: "实测伪谱半径" });
    if (segs.length > 0) scene.layers.push(lineLayer("aux", segs));
    measuredSlope = log10Slope(okX.map((e) => Math.log10(e)), okY);
    // 解析参考线 ρ_ε = ε^(1/n)（Jordan 块；pitfalls 第二条给的就是这条）。它只用到帧里的
    // `meta.n` 与横轴值，属于"显示用的理论对照线"，与 `theory_rate` 那类帧内参考线同类；
    // 其它族没有这条闭式，就不画（不拿别的公式硬套）。
    if (family === "jordan" || family === "shifted_jordan") {
      const nDim = Math.max(1, Math.round(metaNumber(frame, "n") ?? 1));
      const refEps = [...epsX].sort((a, b) => a - b);
      scene.layers.push(
        lineLayer(
          "aux",
          series(
            refEps,
            refEps.map((e) => e ** (1 / nDim)),
            pal.reference,
            { dash: [6, 3], width: 1.2, label: "解析参考 ρ_ε = ε^(1/n)" },
          ),
        ),
      );
    }
    if (nZero > 0) {
      degraded.push(
        `${nZero}/${radius.length} 个 ε 的伪谱半径恰好是 0（网格上抓不到格点）：对数轴上画不出来，只能报数——这是网格分辨率问题，不是「伪谱是空的」`,
      );
    }
  } else {
    const mapped = replaceFloors(ratio, auxYlim[0]);
    const segs = segmentedSeries(epsX, mapped.values, pal.curves[0], { width: 1.8, marker: "o", label: "实测 ρ_ε / ρ(A)" });
    if (segs.length > 0) scene.layers.push(lineLayer("aux", segs));
    const exact = replaceFloors(ratioExact, auxYlim[0]);
    const exactSegs = segmentedSeries(epsX, exact.values, pal.reference, { width: 1.3, dash: [6, 3], label: "参考比值" });
    if (exactSegs.length > 0) scene.layers.push(lineLayer("aux", exactSegs));
    const fitX: number[] = [];
    const fitY: number[] = [];
    for (let i = 0; i < ratio.length; i += 1) {
      const v = ratio[i] as number;
      if (Number.isFinite(v) && v > 1e-299) {
        fitX.push(logEps[i] as number);
        fitY.push(v);
      }
    }
    measuredSlope = log10Slope(fitX, fitY);
    if (mapped.nFloor > 0) {
      degraded.push(
        `${mapped.nFloor}/${ratio.length} 个 ε 的 ρ_ε/ρ(A) 落在地板 ${fmtSci(LOG_FLOOR)} 上（后端把 0 抬到地板）：画在轴底 ${fmtSci(auxYlim[0])}，不是「比值真的等于它」`,
      );
    }
  }
  // 当前 ε 的游标：横线（x = 当前 ε）+ 点（当前纵值）
  const epsNow = epsMark.length > 0 ? (epsMark[0] as number) : epsSel;
  if (epsNow !== undefined && Number.isFinite(epsNow) && epsNow > 0) {
    scene.layers.push(
      lineLayer("aux", series([epsNow, epsNow], auxYlim, pal.cursor, { dash: [3, 3], width: 1, label: "当前 ε" })),
    );
    const rawMark = radiusMark.length > 0 ? (radiusMark[0] as number) : Number.NaN;
    const yMark = useAbsolute
      ? radiusCurrent
      : Number.isFinite(rawMark) && rawMark > 0
        ? rawMark <= 1e-299
          ? auxYlim[0]
          : rawMark
        : undefined;
    if (yMark !== undefined && Number.isFinite(yMark) && yMark > 0) {
      scene.layers.push(scatterLayer("aux", [epsNow], [yMark], { r: 4, color: pal.marker, label: "当前伪谱半径" }));
    }
  }

  // —— 误差场面板（requiredAux: error）——
  scene.axes.push(
    makeAxis("err", errRect, {
      xlim: [-lim, lim],
      ylim: [-lim, lim],
      xlabel: "Re z",
      ylabel: "Im z",
      // 刻意**不设 title**：右列两块面板是竖直切分的，标题带会落到上面那块轴的
      // `xlabel` 带上（`AXIS_DECO` 的两条带互相重叠）。面板名字由下面这行文本承担。
      autoscaled: false,
      xticks: [-1, 0, 1],
      yticks: [-1, 0, 1],
    }),
  );
  let nErrFloor = 0;
  let nErrReal = 0;
  const errLog = new Float64Array(errField.length);
  for (let i = 0; i < errField.length; i += 1) {
    const v = errField[i] as number;
    if (!Number.isFinite(v) || v <= 0) {
      errLog[i] = Number.NaN;
      continue;
    }
    if (v <= 1e-299) {
      nErrFloor += 1;
      errLog[i] = ERR_VMIN;
    } else {
      nErrReal += 1;
      errLog[i] = Math.max(Math.log10(v), ERR_VMIN);
    }
  }
  scene.layers.push(
    heatLayer("err", errLog, nx, ny, [-lim, lim, -lim, lim], {
      // 色标刻意用 gray：这个场要么整块在地板、要么只有零星几个格点亮，
      // 用发散色标（coolwarm）只会把"信息量为零"画成一大片抢眼的饱和色。
      cmap: "gray",
      vmin: ERR_VMIN,
      vmax: 0,
      clipBelow: ERR_VMIN,
    }),
  );

  // —— 文本 ——
  const summary = paramSummary(frame, opts, [{ name: "n", label: "n" }]);
  degraded.push(...summary.degraded);

  const mainLines: string[] = [
    `${summary.text}　族=${family ?? "?"}　ρ(A)=${rho === undefined ? "?" : fmtSci(rho)}　κ_max=${kappaMax === undefined ? "?" : fmtSci(kappaMax)}`,
    epsSel === undefined
      ? "当前 ε：帧未给 meta.epsilon"
      : `当前 ε=${fmtSci(epsSel)}：ρ_ε=${radiusCurrent === undefined ? "?" : fmtSci(radiusCurrent)}` +
        (useAbsolute ? "" : `、ρ_ε/ρ(A)=${ratioCurrent === undefined ? "?" : fmtSci(ratioCurrent)}`),
    `可信格点 ${trustedFraction === undefined ? "?" : `${(trustedFraction * 100).toFixed(1)}%`}` +
      (nNoise === undefined || nPoints === undefined ? "" : `（${fmtSci(nNoise, 0)}/${fmtSci(nPoints, 0)} 个已掉进舍入地板 ≈‖A‖√u）`) +
      `：灰格点已按「无数据」掩掉；等高线是逐 ε 的掩码边界（不是插值）` +
      (nEmptyMasks > 0 ? `；${nEmptyMasks}/${nEps} 个 ε 整片为空（网格间距 ${fmtSci(2 * lim / Math.max(1, nx - 1))}）：${emptyEps.join("、")}` : ""),
  ];
  if (method !== undefined) mainLines.push(`帧 method：${method}`);
  scene.texts.push(...textBlock(mainRect[0] + 6, textTop(mainRect), textBottom(mainRect), mainLines, pal.ink, mainBudget, "top"));

  const auxLines: string[] = [];
  if (useAbsolute) {
    auxLines.push(
      rho !== undefined && rho <= 0
        ? "⚠ 本帧 ρ(A)=0：ρ_ε/ρ(A) 的分母是 0，那个比值「没有定义」（帧里 8 个值全是地板 1e-300）——这里改画绝对半径 ρ_ε"
        : `⚠ 帧里 ${epsGrid.length} 个 ε 的 ρ_ε/ρ(A) 全是地板 ${fmtSci(LOG_FLOOR)}：比值的口径在这帧上给不出数，改画绝对半径 ρ_ε`,
    );
    auxLines.push(
      measuredSlope === undefined
        ? "实测 log-log 斜率：点太少或全在地板上，算不出来（不编）"
        : `实测 log-log 斜率 = ${measuredSlope.toFixed(3)}` +
          (family === "jordan" || family === "shifted_jordan"
            ? `（解析对照：Jordan 块 ρ_ε ≈ ε^(1/n)，斜率应为 1/n = ${(1 / Math.max(1, Math.round(metaNumber(frame, "n") ?? 1))).toFixed(3)}）`
            : "（斜率 = 伪谱半径随 ε 的幂次）"),
    );
    auxLines.push(
      `ρ_ε 从 ${fmtSci(radiusChan.lo)} 到 ${fmtSci(radiusChan.hi)}（${radius.length} 个 ε；注意「抓不到」的 ε 是网格分辨率问题，见 pitfalls）`,
    );
  } else {
    auxLines.push("实线+点 = 实测 ρ_ε/ρ(A)；长虚线 = 正规矩阵的解析对照 (ρ+ε)/ρ（两者重合 ⇒ 就是 ε-圆盘的并）");
    auxLines.push(
      `ρ(A)=${rho === undefined ? "?" : fmtSci(rho)}：比值从 ${fmtSci(ratioChan.lo)} 到 ${fmtSci(ratioChan.hi)}；` +
        (measuredSlope === undefined ? "斜率算不出来" : `实测 log-log 斜率 ${measuredSlope.toFixed(3)}`),
    );
    auxLines.push("正规矩阵这一页是一条水平线；非正规矩阵是往上翘的曲线——翘多少就是「谱之外还有多大一片」");
  }
  if (kreissLower !== undefined && status !== undefined) {
    auxLines.push(`帧 status：${status}；帧给的 Kreiss 下界 ${fmtSci(kreissLower)}（是下界不是上界：它小不代表没有大瞬态）`);
  } else if (status !== undefined) {
    auxLines.push(`帧 status：${status}`);
  } else if (kreissLower !== undefined) {
    auxLines.push(`帧给的 Kreiss 下界 ${fmtSci(kreissLower)}（是下界不是上界：它小不代表没有大瞬态）`);
  }
  scene.texts.push(...textBlock(auxRect[0] + 6, textTop(auxRect), textBottom(auxRect), auxLines, pal.muted, auxBudget, "bottom"));

  const errLines: string[] = [
    "误差场 σ_min − dist(z, Λ)（帧 error；固定窗口 ±" + fmtSci(lim) + "）",
    nErrReal === 0
      ? `这一格恒为 0：σ_min(A−zI) ≤ dist(z,Λ(A)) 是恒等式，所以 max(σ_min−dist,0) 整块贴在地板 ${fmtSci(LOG_FLOOR)}（色阶夹到 ${ERR_VMIN}）`
      : `这一格 ${nErrReal}/${errField.length} 个点真的顶到了舍入尺度（σ_min 与 dist 数值上分不开），其余贴在地板 ${fmtSci(LOG_FLOOR)}`,
    `它不能当「σ_min 的误差估计」用（字段定义如此）——后端字段问题，已上报；σ_min 本帧 min=${sigmaMin === undefined ? "?" : fmtSci(sigmaMin)}、max=${sigmaMax === undefined ? "?" : fmtSci(sigmaMax)}`,
  ];
  scene.texts.push(...textBlock(errRect[0] + 6, textTop(errRect), textBottom(errRect), errLines, pal.muted, errBudget, "top"));

  // —— 角标 ——
  scene.badges.push(placeBadge(sourceBadge(scene.dataSource), SLOT.source, width));
  if (scene.dataSource.startsWith("backend+preview")) {
    scene.badges.push(placeBadge(previewBadge("拖动时按显示空间插值"), SLOT.preview, width));
  }
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(
      badge(
        "autoscale",
        useAbsolute
          ? "副轴：自适应 log-log + 口径改为绝对半径 ρ_ε（ρ(A)=0，比值无定义）"
          : "副轴：自适应 log-log（ε 与比值都跨量级；自适应会掩盖量级变化）",
        8,
        8,
      ),
      SLOT.auxAxis,
      width,
    ),
  );
  const statusBits: string[] = [];
  if (status !== undefined) statusBits.push(status);
  statusBits.push(
    `不可信格点 ${nUntrusted}/${logSigma.length}（已掩成无数据）` + (nEmptyMasks > 0 ? `；${nEmptyMasks} 个 ε 的掩码为空` : ""),
  );
  if (statusBits.length === 0) statusBits.push("帧未给出可判读的状态（预期有 strings.status）");
  scene.badges.push(placeBadge(badge("status", clipBadgeText(statusBits.join("；")), 8, 8), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);

  // 规则 1：掩掉的格点、空掩码、地板值、非有限值，全部记账。
  const extra: string[] = [];
  if (nUntrusted > 0) {
    extra.push(`main 的 σ_min 场有 ${nUntrusted}/${logSigma.length} 个格点掉进舍入地板（trusted=0），已掩成「无数据」灰格并逐个标出`);
  }
  if (nEmptyMasks > 0) extra.push(`${nEmptyMasks}/${nEps} 个 ε 的 inside 掩码整片为空：这些 ε 没有等高线可画（网格分辨率，不是「伪谱为空」）`);
  if (nErrFloor > 0) extra.push(`error 场有 ${nErrFloor}/${errField.length} 个格点是地板 ${fmtSci(LOG_FLOOR)}（恒为 0 的字段），色阶夹到 ${ERR_VMIN}`);
  if (nErrReal > 0) extra.push(`error 场有 ${nErrReal}/${errField.length} 个格点在舍入尺度上（> 地板）`);
  const nBad =
    countNonFinite(logSigma) + countNonFinite(errField) + countNonFinite(ratio) + countNonFinite(radius) + countNonFinite(specX) + countNonFinite(specY);
  applyOverflowNote(scene, extra.join("；"), nBad, { slot: SLOT.overflow, width });
  return scene;
};
