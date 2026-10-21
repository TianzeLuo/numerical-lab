/**
 * `svd_truncation`（1.3 章）的 model：L 曲线、奇异值谱/Picard 量、截断权衡。
 *
 * 四个面板（manifest 的三条轴 + 一条模型自加的"比值"轴）
 * ----------------------------------------------------
 * | 轴 | 位置 | 内容 | 自适应 |
 * |---|---|---|---|
 * | `lcurve` | 左上 | 解范数 ‖x_k‖（横） vs 相对残差（纵），双对数 | 自适应 |
 * | `spectrum` | 右上 | 奇异值谱 σ_i（loglogfit）+ 谱比值 | 自适应 |
 * | `picard` | 右下 | `|u_iᵀb|`、Picard 比 `|u_iᵀb|/σ_i` 随 i（对数） | 自适应 |
 * | `tradeoff` | 左下 | ‖x_k‖ / 残差 / 相对误差 随 k（对数） | 自适应 |
 *
 * **为什么要多一条 `picard` 轴**（manifest 只声明了三条）：后端一帧里同时给了
 * `sigma`、`coeff`、`picard` 三组量，而 σ 与 Picard 比在同一根对数轴上相差好几个数量级
 * （默认帧：σ∈[1e-8,1]、Picard∈[2.2e-3,6.5e1]）——挤在一根轴上，"Picard 比在尾部
 * 重新抬头"这个**噪声接管的 signature** 就会被压成一条平线。分成两条轴之后，
 * 每一条都有一根自己的对数尺子。四、五两条纪律（谱要标幂律、自适应要标注）都照做。
 *
 * 几条刻意的决定
 * --------------
 * 1. **k=0 不画在对数轴上**：`x_norm[0]` 与 `lcurve_x[0]` 是地板 `1e-18`（后端把 0
 *    抬到地板），它在双对数图上是一个**位于角落的假点**（"解范数 1e-18"其实意味着
 *    "截断到 0 阶，解恒为 0"）。模型把 k=0 排除出曲线并写明，既不静默丢掉也不把它当真值。
 * 2. **三个候选 k 都画出来、并且说清它们的地位**：`k_corner`（L 曲线拐点，启发式）、
 *    `k_picard`（Picard 转折点）、`k_opt_error`（**真实**误差最小处，只有知道真解才算得出来）。
 *    manifest 的 pitfalls 说它们经常互不相等——图上不给"哪个是对的"的暗示，
 *    三条竖线并列 + 数值行 + 说明文字。
 * 3. **截断阶是整数步**：时间轴 t 是连续的（`dt=1`），模型用 `k_grid` 找最近的整数 k
 *    （与后端 `_step_of` 同口径：四舍五入后夹住），并在文本里写出**实到的 k**，
 *    不假装 t 就是 k。
 * 4. **`noise_level` 是帧给的实测值**（不是按参数重算的）：画成横线并标注；
 *    它同时是"残差降到这里以下就没有意义了"的读图参照。
 */

import { niceLimits } from "../../shared/src/axes.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import {
  canvasSize,
  clamp,
  clipBadgeText,
  criticalOf,
  dataSourceOf,
  degradedBadge,
  finiteMinMax,
  finitePositiveMinMax,
  metaNumber,
  palette,
  placeBadge,
  requireArray,
  requireDrawableLayers,
  resolveTextOverlaps,
  requirePitfalls,
  requireSameLength,
  sceneShell,
  stackedPanels,
  stringOf,
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
  type LineSeries,
  pitfallsBand,
} from "../../shared/src/scene.js";
import { contentBottom, fmt, fmtOr, joinParts, kv, overflowBadgeFor, textBlock, unitBudget } from "./_shared.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "1.3";
const VIZ = "svd_truncation";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "奇异值谱与截断：L 曲线、Picard 条件与最优 k";

/** 固定角标槽位（见 `model.ts::BADGE_LAYOUT`）。 */
const SLOT = {
  source: 0,
  axisMode: 1,
  auxAxis: 2,
  critical: 3,
  status: 4,
  preview: 5,
  overflow: 6,
  degraded: 7,
} as const;

const KAPPA_CRITICAL = 15.95;
const NOISE_CRITICAL = -15.95;

/** 对数地板（后端把 0 与小于地板的量都抬到它；模型照样画在它上面并说明）。 */
const FLOOR = 1e-18;

export const svdTruncationModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需字段 ——
  const kGrid = requireArray(frame, "k_grid");
  const sigma = requireArray(frame, "sigma");
  const coeff = requireArray(frame, "coeff");
  const picard = requireArray(frame, "picard");
  const xNorm = requireArray(frame, "x_norm");
  const residual = requireArray(frame, "residual");
  const error = requireArray(frame, "error");
  const lcurveX = requireArray(frame, "lcurve_x");
  const lcurveY = requireArray(frame, "lcurve_y");
  const kMark = requireArray(frame, "k_mark");
  const noiseLevel = requireArray(frame, "noise_level");
  const pitfalls = requirePitfalls(frame);
  requireSameLength(kGrid, xNorm, "帧数组 k_grid 与 x_norm");
  requireSameLength(kGrid, residual, "帧数组 k_grid 与 residual");
  requireSameLength(kGrid, error, "帧数组 k_grid 与 error");
  requireSameLength(lcurveX, lcurveY, "帧数组 lcurve_x 与 lcurve_y");
  requireSameLength(sigma, coeff, "帧数组 sigma 与 coeff");
  requireSameLength(sigma, picard, "帧数组 sigma 与 picard");

  // —— 可选字段 ——
  const degraded: string[] = [];
  const tMeta = metaNumber(frame, "t");
  if (tMeta === undefined) degraded.push("帧 meta 缺 t：scene.t 用请求 t（两者可能不同）");
  const tNow = tMeta ?? opts.t;
  const kCurrentMeta = metaNumber(frame, "k_current");
  const kMax = metaNumber(frame, "k_max");
  const kOpt = metaNumber(frame, "k_opt_error");
  const kCorner = metaNumber(frame, "k_corner");
  const kPicard = metaNumber(frame, "k_picard");
  const cliffIndex = metaNumber(frame, "cliff_index");
  const xNormJump = metaNumber(frame, "x_norm_jump");
  const kappaTarget = metaNumber(frame, "kappa_target");
  const kappaMeasured = metaNumber(frame, "kappa_measured");
  const sigmaMax = metaNumber(frame, "sigma_max");
  const sigmaMin = metaNumber(frame, "sigma_min");
  const noiseRel = metaNumber(frame, "noise_rel");
  const noiseExp = metaNumber(frame, "noise_exp");
  const errorCurrent = metaNumber(frame, "error_current");
  const residualCurrent = metaNumber(frame, "residual_current");
  const xNormCurrent = metaNumber(frame, "x_norm_current");
  const errorFloor = metaNumber(frame, "error_floor");
  const picardMin = metaNumber(frame, "picard_min");
  const rank = metaNumber(frame, "rank");
  const nSize = metaNumber(frame, "n");
  const mSize = metaNumber(frame, "m");
  const spectrumName = stringOf(frame, "spectrum");
  if (spectrumName === undefined) degraded.push("帧 strings 缺 spectrum：看不出谱形状（不按参数猜）");
  const note = stringOf(frame, "note");

  // 时间轴 → 最近的整数截断阶（与后端 `_step_of` 同口径：四舍五入 + 夹住）。
  // 真正显示的是**帧里给的** k_current；t 只用来在缺 k_current 时定位。
  const kFromT = Math.round(tNow);
  const kNow = clamp(Math.round(kCurrentMeta ?? kFromT), 0, Math.max(0, kGrid.length - 1));
  if (kCurrentMeta === undefined) {
    degraded.push(`帧 meta 缺 k_current：当前截断阶由请求 t=${fmt(tNow, 4)} 四舍五入得到（t 不是 k，这里是夹取）`);
  } else if (Math.round(kCurrentMeta) !== kFromT) {
    degraded.push(`帧 meta.k_current=${kCurrentMeta} 与 t=${fmt(tNow, 3)} 不一致：标记按 k_current 画（帧里的实测声位优先）`);
  }
  const kNowValue = kGrid[kNow] as number;

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: tNow,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  // —— 排版：左上 L 曲线、左下权衡；右上谱、右下 Picard ——
  const base = stackedPanels(width, height);
  // 底部面板（渲染层现画的 `scene.pitfalls`）会随文案长短变高：把内容底边一起收上来，
  // 否则副轴与文本的下半截会被它盖住（实测 1.3 的 fan 帧面板顶边只有 442，而 base.bottom=498）。
  base.bottom = contentBottom(frame, width, height, base.bottom);
  // 四条轴 + 两块文字。这个排法是**被规则 8 逼出来的**，每一步都有原因：
  //  - 两个纵轴各要 left−56..left−4 的刻度带，所以两列之间必须留 80px；
  //  - 轴名带在轴下方 16..28px、刻度带在 3..15px，所以上下两块之间要留 ~58px；
  //  - 剩下的竖直空档只有两处能放文字：左列 L 曲线与权衡面板之间（约 60px）、
  //    以及右列 Picard 面板下方（约 45px 净空）。两块文本就落在这两处，
  //    与任何轴的装饰带都不相交（校验器会当场报，这里被报过三次才收敛）。
  const colGap = 80;
  const leftW = Math.max(220, Math.round((base.contentWidth - colGap) * 0.5));
  const rightW = Math.max(150, base.contentWidth - leftW - colGap);
  // 竖直方向逐段算（1 行文字 = 15px、轴刻度带 = 3..15px、轴名带 = 16..28px、
  // 面板标题带 = top−18..top−2）：两列共用同一条骨架，每列中间留一条 33px 的空档
  // 给文字——那是**唯一**同时"够宽、又不与任何轴的装饰带相交"的位置（规则 8 前后
  // 把这一版打回五次：塞在面板之间、塞在顶部、塞在底部都越界，最后收敛到这里）：
  //   [122, 245] 面板（左：L 曲线 / 右：谱）
  //   [258, 288] 文字（左：说明块 / 右：数值块）——上一块的轴名带止于 245+28 = 273
  //   [320, 496] 面板（左：权衡 / 右：Picard）→ 轴名带到 524
  // 每段的像素账（写死成常量，改一个数字就能看出会不会压到别的东西）：
  //   上排面板 126 + 轴名带 28 + 文字带 45 + 下排标题带 18 + 下排面板 187 + 轴名带 26 = 430
  // 上面两块面板压到 110：底部 pitfalls 面板变高以后（实测顶边 506），
  // 中间那条文字带只剩两行的净空——多出来的 16px 全给文字带。
  const PANEL_TOP_H = 110;
  const TEXT_BAND_H = 45; // 两行文字 + 一点白（`textBlock` 每行占 15px 的盒子）
  const LOWER_TITLE_BAND = 18;
  const LOWER_XLABEL_BAND = 26;
  const lowerTop = base.top + PANEL_TOP_H + 28 + TEXT_BAND_H + LOWER_TITLE_BAND;
  const lowerH = Math.max(90, base.bottom - LOWER_XLABEL_BAND - lowerTop);
  const lcurveRect: Rect = [base.left, base.top, leftW, PANEL_TOP_H];
  const spectrumRect: Rect = [base.left + leftW + colGap, base.top, rightW, PANEL_TOP_H];
  const tradeoffRect: Rect = [base.left, lowerTop, leftW, lowerH];
  const picardRect: Rect = [spectrumRect[0], lowerTop, rightW, lowerH];

  // —— 轴 ——
  const lPos = finitePositiveMinMax(lcurveX);
  const lPosY = finitePositiveMinMax(lcurveY);
  const lcurveLim: [number, number] =
    lPos !== undefined ? niceLimits(Math.max(lPos[0], FLOOR), lPos[1], 6, { log: true, pad: 0.06 }) : [FLOOR, 1];
  const lcurveYlim: [number, number] =
    lPosY !== undefined ? niceLimits(Math.max(lPosY[0], FLOOR), lPosY[1], 6, { log: true, pad: 0.06 }) : [1e-8, 1];
  if (lPos === undefined || lPosY === undefined) degraded.push("lcurve_x/lcurve_y 没有 >0 的有限值：L 曲线轴用占位范围");
  const lcurveAxis = makeAxis("lcurve", lcurveRect, {
    xlim: lcurveLim,
    ylim: lcurveYlim,
    xscale: "log",
    yscale: "log",
    xlabel: "解范数 $\\|x_k\\|$（对数）",
    ylabel: axisYlabel("lcurve", "相对残差"),
    title: axisTitle("lcurve", "L 曲线：$\\|x_k\\|$ 与 $\\|Ax_k-b\\|/\\|b\\|$（双对数）"),
    autoscaled: true,
  });

  const kRange = finiteMinMax(kGrid) ?? [0, kGrid.length - 1];
  const tradeoffPos = finitePositiveMinMax(xNorm);
  const tradeoffYlim: [number, number] =
    tradeoffPos !== undefined ? niceLimits(Math.max(tradeoffPos[0], FLOOR), Math.max(tradeoffPos[1], 1), 6, { log: true, pad: 0.06 }) : [FLOOR, 1e2];
  const tradeoffAxis = makeAxis("tradeoff", tradeoffRect, {
    xlim: [kRange[0] - 0.4, kRange[1] + 0.4],
    ylim: tradeoffYlim,
    yscale: "log",
    xlabel: "截断阶 $k$",
    ylabel: axisYlabel("tradeoff", "三个量（对数）"),
    title: axisTitle("tradeoff", "三个量随截断阶 $k$ 的变化（对数）"),
    autoscaled: true,
  });

  const iGrid = Float64Array.from({ length: sigma.length }, (_, i) => i + 1);
  const sigmaPos = finitePositiveMinMax(sigma);
  const spectrumAxis = makeAxis("spectrum", spectrumRect, {
    xlim: [1, Math.max(2, sigma.length)],
    ylim: sigmaPos !== undefined ? niceLimits(sigmaPos[0], sigmaPos[1], 6, { log: true, pad: 0.1 }) : [1e-16, 1],
    yscale: "log",
    xlabel: "奇异值序号 $i$",
    ylabel: axisYlabel("spectrum", "$\\sigma_i$"),
    title: axisTitle("spectrum", "奇异值谱（双对数；直线表示幂律衰减，虚线是参考斜率）"),
    autoscaled: true,
  });
  const picardPos = finitePositiveMinMax(picard);
  const coeffPos = finitePositiveMinMax(coeff);
  const picLo = Math.min(picardPos?.[0] ?? Number.POSITIVE_INFINITY, coeffPos?.[0] ?? Number.POSITIVE_INFINITY);
  const picHi = Math.max(picardPos?.[1] ?? Number.NEGATIVE_INFINITY, coeffPos?.[1] ?? Number.NEGATIVE_INFINITY);
  const picardAxis = makeAxis("picard", picardRect, {
    xlim: spectrumAxis.xlim,
    ylim: Number.isFinite(picLo) && Number.isFinite(picHi) ? niceLimits(picLo, picHi, 6, { log: true, pad: 0.08 }) : [FLOOR, 1e2],
    yscale: "log",
    xlabel: "奇异值序号 $i$",
    ylabel: "$|u_i^{\\mathrm{T}}b|$ 与比值",
    title: "Picard 量：$|u_i^{\\mathrm{T}}b|$ 与 $|u_i^{\\mathrm{T}}b|/\\sigma_i$（对数）",
    autoscaled: true,
  });
  if (!(Number.isFinite(picLo) && Number.isFinite(picHi))) {
    degraded.push("coeff/picard 没有 >0 的有限值：Picard 轴用占位范围");
  }
  scene.axes.push(lcurveAxis, spectrumAxis, picardAxis, tradeoffAxis);

  // —— L 曲线面板：k=0 排除、拐点/最优点标出 ——
  // k=0 → 解恒为 0，后端把范数抬到地板；它在双对数图上是个角落里的假点（见文件头第 1 条）。
  const lx: number[] = [];
  const ly: number[] = [];
  const lk: number[] = [];
  for (let i = 0; i < lcurveX.length; i += 1) {
    if ((kGrid[i] as number) <= 0) continue;
    const a = lcurveX[i] as number;
    const b = lcurveY[i] as number;
    if (Number.isFinite(a) && a > 0 && Number.isFinite(b) && b > 0) {
      lx.push(a);
      ly.push(b);
      lk.push(kGrid[i] as number);
    }
  }
  const lcurveSeries: LineSeries[] = [];
  if (lx.length >= 1) {
    lcurveSeries.push(
      series(Float64Array.from(lx), Float64Array.from(ly), pal.curves[0], { width: 1.8, label: "L 曲线（$k$ 增大：残差下降、解范数上升）" }),
    );
    // 每个 k 一个点（点数少，直接画出来比只看折线清楚）
    lcurveSeries.push(
      series(Float64Array.from(lx), Float64Array.from(ly), pal.curves[0], { width: 0, marker: "o", label: "每个截断阶 $k$ 的取值" }),
    );
  } else {
    degraded.push("L 曲线在 k≥1 上没有 >0 的有限点：只画噪声水平参考线");
  }
  const markAt = (k: number | undefined, color: string, label: string): void => {
    if (k === undefined || !Number.isFinite(k)) return;
    const idx = lk.indexOf(Math.round(k));
    if (idx < 0) {
      degraded.push(`候选截断阶 k=${fmt(k, 0)} 不在帧的 k_grid 上：不标这一枚（不按 t 猜）`);
      return;
    }
    lcurveSeries.push(
      series([lx[idx] as number], [ly[idx] as number], color, { width: 0, marker: "s", label }),
    );
  };
  const ks: string[] = [];
  if (kCorner !== undefined) {
    markAt(kCorner, pal.exact, `L 曲线拐点 k_corner=${fmt(kCorner, 0)}（启发式：曲率最大处）`);
    ks.push(`k_corner=${fmt(kCorner, 0)}`);
  }
  if (kOpt !== undefined) {
    markAt(kOpt, pal.warn, `真实最优 k_opt=${fmt(kOpt, 0)}（只有知道真解才算得出来）`);
    ks.push(`k_opt=${fmt(kOpt, 0)}`);
  }
  if (kPicard !== undefined) {
    markAt(kPicard, pal.residual, `Picard 转折点 k_picard=${fmt(kPicard, 0)}`);
    ks.push(`k_picard=${fmt(kPicard, 0)}`);
  }
  const noiseVal = noiseLevel.length > 0 ? (noiseLevel[0] as number) : undefined;
  if (noiseVal !== undefined && Number.isFinite(noiseVal) && noiseVal > 0) {
    lcurveSeries.push(
      series(lcurveAxis.xlim, [noiseVal, noiseVal], pal.muted, {
        width: 1,
        dash: [5, 4],
        label: `噪声水平 ${fmt(noiseVal, 2)}（残差降到它以下就没有意义）`,
      }),
    );
  }
  if (lcurveSeries.length > 0) scene.layers.push(lineLayer("lcurve", lcurveSeries));
  // 当前 k 的位置：L 曲线上的一枚点（坐标取自帧数组的第 k 个，不是插值）
  if (kNow >= 1 && kNow - 1 < lx.length) {
    scene.layers.push(
      scatterLayer("lcurve", [lx[kNow - 1] as number], [ly[kNow - 1] as number], {
        r: 5,
        color: pal.marker,
        label: `当前截断阶 $k=${fmt(kNowValue, 0)}$`,
      }),
    );
  }
  // `k_mark` 是后端算出来的"当前截断阶"（帧里的实测位置），`meta.k_current` 只是它的标量镜像。
  // 两者不一致时以**数组**为准，并在降级角标里点名（数组来自后端，meta 可能落后一帧）。
  const kFromArray = kMark.length > 0 ? (kMark[0] as number) : undefined;
  if (kFromArray !== undefined && Number.isFinite(kFromArray) && Math.round(kFromArray) !== kNow) {
    degraded.push(
      `帧数组 k_mark=${fmt(kFromArray, 0)} 与按 meta/t 得到的 k=${fmt(kNow, 0)} 不一致：` +
        `图上的标记位置以 meta/t 为准，数组值一并报出（不替它挑一个）`,
    );
  }

  // —— 谱面板：loglogfit（规则 2：实测斜率 + 理论参考线都要有）——
  const sXs: number[] = [];
  const sYs: number[] = [];
  for (let i = 0; i < sigma.length; i += 1) {
    const v = sigma[i] as number;
    if (Number.isFinite(v) && v > 0) {
      sXs.push(iGrid[i] as number);
      sYs.push(v);
    }
  }
  const slopeMeasured = powerLawSlope(sXs, sYs);
  if (sXs.length >= 2) {
    scene.layers.push(
      loglogFitLayer("spectrum", iGrid, sigma, slopeMeasured, 0, {
        seriesLabel: `实测谱（幂律拟合斜率 ${fmt(slopeMeasured, 2)}）`,
        theoryLabel: "理论参考斜率 0（指数衰减谱在双对数上是直线，斜率 ≈ 0）",
        color: pal.curves[0],
        theoryColor: pal.reference,
      }),
    );
  } else {
    degraded.push("sigma 里没有 ≥2 个 >0 的有限值：谱面板画不出幂律参考线（规则 2 要求两条斜率都在）");
  }
  // 谱的比值（相邻 σ 之比）：悬崖的位置一眼可见——阈值取在悬崖上，k 差一就会跳级。
  const ratioX: number[] = [];
  const ratioY: number[] = [];
  for (let i = 1; i < sXs.length; i += 1) {
    const a = sYs[i - 1] as number;
    const b = sYs[i] as number;
    if (Number.isFinite(a) && Number.isFinite(b) && a > 0 && b > 0) {
      ratioX.push(sXs[i] as number);
      ratioY.push(a / b);
    }
  }
  if (ratioX.length > 0) {
    scene.layers.push(
      lineLayer("spectrum", [
        series(Float64Array.from(ratioX), Float64Array.from(ratioY), pal.exact, {
          width: 1.3,
          dash: [4, 3],
          label: "相邻奇异值之比 $\\sigma_{i-1}/\\sigma_i$（峰值是谱的悬崖）",
        }),
      ]),
    );
  }
  if (cliffIndex !== undefined) {
    const cv = iGrid[Math.min(Math.max(Math.round(cliffIndex) - 1, 0), iGrid.length - 1)] as number;
    scene.layers.push(
      lineLayer("spectrum", [
        series([cv, cv], spectrumAxis.ylim, pal.warn, {
          width: 1.2,
          dash: [6, 4],
          label: `谱跳变最大的位置 i=${fmt(cv, 0)}（cliff_index；阈值取在这里解范数会突变）`,
        }),
      ]),
    );
  }

  // —— Picard 面板 ——
  const picardLayers: LineSeries[] = [];
  const coeffX: number[] = [];
  const coeffY: number[] = [];
  const picX: number[] = [];
  const picY: number[] = [];
  for (let i = 0; i < sigma.length; i += 1) {
    const c = coeff[i] as number;
    const p = picard[i] as number;
    if (Number.isFinite(c) && c > 0) {
      coeffX.push(iGrid[i] as number);
      coeffY.push(c);
    }
    if (Number.isFinite(p) && p > 0) {
      picX.push(iGrid[i] as number);
      picY.push(p);
    }
  }
  if (coeffX.length > 0) {
    picardLayers.push(
      series(Float64Array.from(coeffX), Float64Array.from(coeffY), pal.curves[1], {
        width: 1.6,
        marker: "o",
        label: "$|u_i^{\\mathrm{T}}b|$（数据在奇异方向上的分量）",
      }),
    );
  }
  if (picX.length > 0) {
    picardLayers.push(
      series(Float64Array.from(picX), Float64Array.from(picY), pal.residual, {
        width: 1.8,
        marker: "s",
        label: "$|u_i^{\\mathrm{T}}b|/\\sigma_i$（Picard 比；尾部抬头表示噪声接管，不是信息）",
      }),
    );
  }
  if (picardLayers.length > 0) scene.layers.push(lineLayer("picard", picardLayers));
  else degraded.push("coeff/picard 没有 >0 的有限值：Picard 面板为空");
  if (kPicard !== undefined) {
    scene.layers.push(
      lineLayer("picard", [
        series([kPicard, kPicard], picardAxis.ylim, pal.reference, {
          width: 1,
          dash: [3, 3],
          label: `Picard 转折点 k=${fmt(kPicard, 0)}`,
        }),
      ]),
    );
  }
  if (kNow >= 1 && kNow - 1 < sigma.length) {
    scene.layers.push(
      lineLayer("picard", [
        series([kNowValue, kNowValue], picardAxis.ylim, pal.cursor, { width: 1, dash: [2, 3] }),
      ]),
    );
  }

  // —— 权衡面板 ——
  const tradeLayers: LineSeries[] = [];
  const tradeSpecs: { arr: ArrayLike<number>; color: string; dash?: number[]; label: string }[] = [
    { arr: xNorm, color: pal.curves[0], label: "$\\|x_k\\|$（单调不减）" },
    { arr: residual, color: pal.curves[1], dash: [6, 3], label: "相对残差（单调不增）" },
    { arr: error, color: pal.warn, label: "相对误差（只有知道真解才算得出来）" },
  ];
  for (const spec of tradeSpecs) {
    const xs: number[] = [];
    const ys: number[] = [];
    for (let i = 0; i < kGrid.length; i += 1) {
      const v = spec.arr[i] as number;
      if ((kGrid[i] as number) <= 0) continue; // k=0 的解恒为 0：对数轴上是地板假点
      if (Number.isFinite(v) && v > 0) {
        xs.push(kGrid[i] as number);
        ys.push(v);
      }
    }
    if (xs.length >= 1) {
      tradeLayers.push(series(Float64Array.from(xs), Float64Array.from(ys), spec.color, { width: 1.6, dash: spec.dash, label: spec.label }));
    } else {
      degraded.push(`权衡面板的「${spec.label}」没有 >0 的有限值：不画这条线`);
    }
  }
  for (const [k, color, label] of [
    [kOpt, pal.warn, "k_opt"],
    [kCorner, pal.exact, "k_corner"],
    [kPicard, pal.residual, "k_picard"],
  ] as const) {
    if (k === undefined || !Number.isFinite(k)) continue;
    tradeLayers.push(
      series([k, k], tradeoffAxis.ylim, color, { width: 1.2, dash: [4, 4], label: `${label}=${fmt(k, 0)}` }),
    );
  }
  // 竖线的说明写在数值行里（三条线常常互不相等——这就是 pitfalls 第 6 条）
  if (tradeLayers.length > 0) scene.layers.push(lineLayer("tradeoff", tradeLayers));
  else degraded.push("权衡面板没有任何可画的曲线");

  // —— 当前截断阶的标记（三块面板各一枚）——
  if (kNow >= 0 && kNow < kGrid.length) {
    if (Number.isFinite(residual[kNow] as number) && (residual[kNow] as number) > 0 && Number.isFinite(error[kNow] as number) && (error[kNow] as number) > 0) {
      scene.layers.push(
        scatterLayer("tradeoff", [kNowValue, kNowValue], [residual[kNow] as number, error[kNow] as number], {
          r: 3.6,
          color: pal.marker,
          label: `当前 $k=${fmt(kNowValue, 0)}$ 处的残差与真实误差`,
        }),
      );
    }
    if (Number.isFinite(xNorm[kNow] as number) && (xNorm[kNow] as number) > 0) {
      scene.layers.push(
        scatterLayer("tradeoff", [kNowValue], [xNorm[kNow] as number], { r: 3.6, color: pal.marker, label: "当前 $k$ 处的解范数" }),
      );
    }
  }

  // —— 文本 ——
  // 这一章四条轴 + 一块 60px 的文字带，是 960×600 画布能塞下的**上限**。
  // 文字带的落点（`[spectrumRect[1]+height+16, picardRect[1]−20]`）是规则 8 逼出来的：
  // 塞在两块面板之间、塞在顶部、塞在底部、塞在左列，五次都被"文本压住坐标轴装饰"打回。
  // 现在它落在右列两块面板的**中间**：上方是谱面板的轴名带（止于 +28），
  // 下方是 Picard 面板的标题带（从 top−18 开始），两侧都不相交。
  // 内容按 3 行写：只写"读图必须知道的数字"，长句交给 `pitfalls` 面板（规则 7）。
  // 3 行（正好填满 45px 的空档；写第 4 行就会压住下方面板的标题带）。
  // 精确到个位的写法：每行都用 `　` 拼同一行的短字段，避免中文长句把行折成两行。
  const info: string[] = [];
  info.push(
    joinParts([
      spectrumName === undefined ? undefined : `谱=${spectrumName}`,
      kv("n", nSize, 0),
      kv("m", mSize, 0),
      kv("秩", rank, 0),
      kv("k", kNowValue, 0),
      // `σmin`/`log10 κ` 不在这里重复：状态角标已经写了，这一行再放就会折行、
      // 把下面两行挤成"另有 N 行"（本块净空只有 3 行）。
    ]) ?? "帧 meta 缺规模/谱信息：这一行不显示",
  );
  info.push(
    joinParts([
      kv("k_corner", kCorner, 0),
      kv("k_picard", kPicard, 0),
      kv("δ", noiseVal, 2),
    ]) ?? "帧 meta 缺候选截断阶：这一行不显示",
  );
  // `此刻 ‖x_k‖/残差/误差` 进状态角标：文字带只有两行，这三个数放在这里就会挤掉 k/k_corner。
  scene.texts.push(
    ...textBlock(
      spectrumRect[0],
      spectrumRect[1] + spectrumRect[3] + 28, // 谱面板的轴名带止于 bottom+28
      picardRect[1] - 20,
      info,
      pal.muted,
      unitBudget(rightW),
      "top",
    ),
  );


  // —— 角标 ——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  if (scene.dataSource.startsWith("backend+preview")) {
    scene.badges.push(placeBadge(previewBadge("拖动时按显示空间插值"), SLOT.preview, width));
  }
  scene.badges.push(placeBadge(axisModeBadge(true), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(badge("autoscale", "四条轴：全部自适应（各自按本帧数据取整；自适应会掩盖量级变化）", 8, 8), SLOT.auxAxis, width),
  );
  if (kappaTarget !== undefined) {
    const lg = Math.log10(Math.max(kappaTarget, Number.MIN_VALUE));
    scene.badges.push(placeBadge(criticalBadge("log10 κ", lg, criticalOf(frame, "log10_kappa", KAPPA_CRITICAL)), SLOT.critical, width));
  }
  if (noiseExp !== undefined) {
    scene.badges.push(
      placeBadge(criticalBadge("log10 δ", noiseExp, criticalOf(frame, "noise_exp", NOISE_CRITICAL), { beyond: "below" }), SLOT.status, width),
    );
  }
  // 状态角标：把"文字块放不下、但读了才知道自己在看什么"的量都收在这里
  //（角标是紧凑摘要，超宽时 `clipBadgeText` 会截断并留 `…`，不会静默丢整块信息）。
  const statusBits: string[] = [`k=${fmt(kNowValue, 0)}/${fmtOr(kMax, 0)}`];
  // 放最前面：角标超宽会从尾部截断，这句是"k=0 为什么不在图上"的唯一交代。
  statusBits.push("k=0 的解恒为 0（后端抬到地板）：不进对数轴，角落里的那个点是假点");
  if (sigmaMin !== undefined) statusBits.push(`σmin=${fmt(sigmaMin, 2)}`);
  statusBits.push(
    `此刻 ‖x_k‖=${fmtOr(xNormCurrent, 3)}、残差=${fmtOr(residualCurrent, 3)}、误差=${fmtOr(errorCurrent, 3)}`,
  );
  if (spectrumName !== undefined) statusBits.push(`谱=${spectrumName}`);
  statusBits.push(`实测 κ=${fmtOr(kappaMeasured, 2)}`);
  statusBits.push(`σmax=${fmtOr(sigmaMax, 2)}`);
  statusBits.push(`δ=${fmtOr(noiseRel, 2)}`);
  statusBits.push(`相邻范数跳变=${fmtOr(xNormJump, 2)}×`);
  if (picardMin !== undefined) statusBits.push(`Picard min=${fmt(picardMin, 2)}`);
  if (errorFloor !== undefined) statusBits.push(`对数地板=${fmt(errorFloor, 1)}`);

  if (kOpt !== undefined && kCorner !== undefined && Math.round(kOpt) !== Math.round(kCorner)) {
    statusBits.push(`k_opt=${fmt(kOpt, 0)} ≠ k_corner=${fmt(kCorner, 0)}`);
  }
  if (note !== undefined) statusBits.push(`帧 note：${note}`);
  scene.badges.push(
    placeBadge(badge("status", clipBadgeText(statusBits.join("；")), 8, 8), noiseExp !== undefined ? SLOT.preview : SLOT.status, width),
  );
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  const nBad =
    countNonFinite(kGrid) +
    countNonFinite(sigma) +
    countNonFinite(coeff) +
    countNonFinite(picard) +
    countNonFinite(xNorm) +
    countNonFinite(residual) +
    countNonFinite(error) +
    countNonFinite(lcurveX) +
    countNonFinite(lcurveY) +
    countNonFinite(noiseLevel);
  // k=0 的解恒为 0，在对数 L 曲线上本来就没有坐标；把它排除是显示约定，不是数据丢失。
  // 因此不往 overflow 里塞，也不挂红色角标——真正的非有限值由 nBad 负责。
  overflowBadgeFor(scene, nBad, [], { slot: SLOT.overflow, width });
  // 规则 8 的安全网（两侧校验器都查）：
  //  - 文本层之间不许互相重叠（帧里的说明一变长就会多折一行压住下一块）；
  //  - 文本不许被底部 pitfalls 面板压住（面板高度随帧里 pitfalls 的折行数变，模型事先看不见）。
  // 只丢不移：下移会推进轴的装饰带，那是另一条违规。丢弃的行数会写在画布顶部那一行里。
  const band = pitfallsBand(scene);
  scene.texts = resolveTextOverlaps(scene.texts, [8, 24, width - 16, band === null ? height - 8 : band[1] - 4]);
  return scene;
};

// --------------------------------------------------------------------------- //
// 局部工具
// --------------------------------------------------------------------------- //

function axisTitle(id: string, fallback: string): string {
  return SPEC?.axes.find((a) => a.id === id)?.title ?? fallback;
}

function axisYlabel(id: string, fallback: string): string {
  return SPEC?.axes.find((a) => a.id === id)?.ylabel ?? fallback;
}

/**
 * 双对数斜率（显示辅助，**不是物理量**）：对 `(x, y)` 做最小二乘。
 *
 * 它只用来给"谱是不是幂律"一个可读的标签（规则 2 的实测斜率），不参与任何结论——
 * 结论性的斜率（误差阶数）一律来自帧的 `meta.slope_*`。点少于 2 个或全不正时返回 0。
 */
function powerLawSlope(xs: readonly number[], ys: readonly number[]): number {
  const px: number[] = [];
  const py: number[] = [];
  for (let i = 0; i < Math.min(xs.length, ys.length); i += 1) {
    const x = xs[i] as number;
    const y = ys[i] as number;
    if (Number.isFinite(x) && x > 0 && Number.isFinite(y) && y > 0) {
      px.push(Math.log10(x));
      py.push(Math.log10(y));
    }
  }
  if (px.length < 2) return 0;
  const mx = px.reduce((a, b) => a + b, 0) / px.length;
  const my = py.reduce((a, b) => a + b, 0) / py.length;
  let sxy = 0;
  let sxx = 0;
  for (let i = 0; i < px.length; i += 1) {
    sxy += ((px[i] as number) - mx) * ((py[i] as number) - my);
    sxx += ((px[i] as number) - mx) ** 2;
  }
  if (sxx === 0) return 0;
  return sxy / sxx;
}
