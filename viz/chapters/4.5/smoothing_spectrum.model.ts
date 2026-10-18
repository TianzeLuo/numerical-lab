/**
 * `smoothing_spectrum`（4.5 章）的 model：逐模态振幅 + 一步放大率 + 整体误差范数。
 *
 * 面板与图层（顺序即绘制顺序）
 * ----------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | line | `spectrum` | 竖直分界 θ/π=0.5（帧 `half_band_line`） |
 * | 2 | line | `factor` | 竖直分界 θ/π=0.5 |
 * | 3 | line | `spectrum` | `amplitude_initial`（灰虚线，初值） |
 * | 4 | line | `spectrum` | `amplitude_theory`（橙虚线，\|λ_k\|^k·\|c_k(0)\|） |
 * | 5 | line | `spectrum` | `amplitude_measured`（蓝实线，第 k 步实测） |
 * | 6 | line | `spectrum` | 地板线 y = `meta.floor` |
 * | 7 | scatter | `spectrum` | 已落到地板的模态（红点，掩码 `amplitude_at_floor`） |
 * | 8 | line | `factor` | 1/3 平滑因子参考线（帧 `smoothing_line`） |
 * | 9 | line | `factor` | `factor_theory`（橙虚线） |
 * | 10 | line | `factor` | `factor_measured`（蓝实线） |
 * | 11 | line | `error` | `error_norm_history`（整体误差范数，**含最低模态的尾巴**） |
 * | 12 | line | `error` | `error_norm_history_slowest`（只按最低模态预测 λ₁^m） |
 * | 13 | scatter | `error` | 当前步的位置 |
 *
 * 四条刻意的决定
 * --------------
 * 1. **两条 x 轴都是 θ/π，不是模态序号 k**：这样"高低频分界 θ/π=0.5"是一条**竖线**，
 *    而 `half_band_line`（帧里是一列常量 0.5）的语义就是这条分界的位置。把它画成横线
 *    （y=0.5）会让对数轴上的读者以为"0.5 是某个振幅阈值"——那是彻底的误读。
 * 2. **`mode_present = 0` 的模态一律不画**：帧的 `strings.nonfinite_policy` 写明了
 *    `factor_measured/theory` 在这些位置是 **1.0 占位符**（比值无定义）、振幅是地板占位。
 *    把占位符当数据画出来，`content=low/high` 时会出现一条假的"完全不衰减"的平线。
 * 3. **`error` 副轴是本条规则 3 的落点**：整体误差范数掉得慢（≈0.89 @ k=30）**完全**是被
 *    最低模态拖住的（高频只剩 3.3e-14）。把它与谱图同屏，第 1 个陷阱才挡得住。
 * 4. **因子轴固定 0–1.1**（manifest 声明 `autoscaled: false`）：越过 1 就是"这一步把该模态
 *    放大了"。自适应会正好藏住"ω 选错时最高频的放大率重新变大"这件事。
 *
 * 时间轴：`scrub`，`t = 迭代步 k`（**不是物理时间**，见帧 `strings.time_semantics`）。
 */

import { formatNumber } from "../../shared/src/axes.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import {
  applyOverflow,
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
import {
  axisTitleOf,
  axisYlabelOf,
  legendBlock,
  logRange,
  maskIndicesWhere,
  panel45Layout,
  pick,
  selectValues,
  textBlock,
  unitBudget,
} from "./_panel.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "4.5";
const VIZ = "smoothing_spectrum";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "频谱衰减：高频迅速消失，低频原地不动";
const T_MAX = SPEC?.time.mode === "none" ? 0 : SPEC?.time.tMax ?? 30;
/** 一维最优平滑权重（manifest 的 critical 值；帧 `critical.omega` 优先）。 */
const OMEGA_OPTIMAL = 0.6667;

/** 固定角标槽位（缺一枚时其余不挪位，动画不抖）。 */
const SLOT = { source: 0, axisMode: 1, auxAxis: 2, critical: 3, status: 4, preview: 5, overflow: 6, degraded: 7 } as const;

/** 平滑子 / 初值的显示名（与 `server/chapters/4.5.py::_smoother`、`_initial_coefficients` 同口径）。 */
const SMOOTHERS = ["jacobi", "gs", "sor"] as const;
const CONTENTS = ["mixed", "low", "high"] as const;

export const smoothingSpectrumModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需数组：缺一个就抛 ModelInputError（宁可当场炸，也不画半张图）——
  const theta = requireArray(frame, "theta_over_pi");
  const amp0 = requireArray(frame, "amplitude_initial");
  const ampNow = requireArray(frame, "amplitude_measured");
  const ampTheory = requireArray(frame, "amplitude_theory");
  const fMeas = requireArray(frame, "factor_measured");
  const fTheory = requireArray(frame, "factor_theory");
  const present = requireArray(frame, "mode_present");
  const normHist = requireArray(frame, "error_norm_history");
  for (const [name, arr] of [
    ["amplitude_initial", amp0],
    ["amplitude_measured", ampNow],
    ["amplitude_theory", ampTheory],
    ["factor_measured", fMeas],
    ["factor_theory", fTheory],
    ["mode_present", present],
  ] as const) {
    requireSameLength(theta, arr, `帧数组 theta_over_pi 与 ${name}`);
  }
  const pitfalls = requirePitfalls(frame);

  // —— 可选数组：缺了就降级（记进 degraded，最终挂成一个 status 角标）——
  const degraded: string[] = [];
  const atFloor = frame.has("amplitude_at_floor") ? requireArray(frame, "amplitude_at_floor") : undefined;
  if (atFloor === undefined) degraded.push("帧缺 amplitude_at_floor：不标「已到地板」的模态（不猜哪些贴了地板）");
  const slowest = frame.has("error_norm_history_slowest") ? requireArray(frame, "error_norm_history_slowest") : undefined;
  if (slowest === undefined) degraded.push("帧缺 error_norm_history_slowest：不画「只按最低模态预测」的对照线");
  const smoothLine = frame.has("smoothing_line") ? requireArray(frame, "smoothing_line") : undefined;
  const halfBand = frame.has("half_band_line") ? requireArray(frame, "half_band_line") : undefined;
  if (halfBand === undefined) degraded.push("帧缺 half_band_line：不画 θ/π=0.5 的分界（不按下标猜它落在哪）");

  // —— meta ——
  const stepsMeta = metaNumber(frame, "steps");
  if (stepsMeta === undefined) degraded.push("帧 meta 缺 steps：时间轴按请求 t 夹到 [0, tMax]，实到步数写进标题行");
  const steps = stepsMeta ?? clamp(Math.round(opts.t), 0, T_MAX);
  const tRequested = metaNumber(frame, "t_requested");
  if (tRequested !== undefined && stepsMeta !== undefined && Math.abs(tRequested - stepsMeta) > 1e-9) {
    degraded.push(`请求 t=${formatNumber(tRequested, 4)} 被夹到 k=${steps}（帧 meta.steps 是实到步数）`);
  }
  const n = metaNumber(frame, "n");
  if (n === undefined) degraded.push("帧 meta 缺 n：不显示内点数（不按请求参数猜）");
  const floor = metaNumber(frame, "floor");
  if (floor === undefined) degraded.push("帧 meta 缺 floor：不画对数轴地板线（1e-16 是帧的约定，不在这里写死）");
  const lowSurv = metaNumber(frame, "low_mode_survival");
  const highSurv = metaNumber(frame, "high_mode_survival");
  const surviveRatio = metaNumber(frame, "low_over_high_survival");
  const errRatio = metaNumber(frame, "error_norm_ratio");
  const nFloor = metaNumber(frame, "n_modes_at_floor");
  const fMeasSmooth = metaNumber(frame, "measured_smoothing_factor");
  const fTheorySmooth = metaNumber(frame, "theory_smoothing_factor");
  const fGap = metaNumber(frame, "smooth_factor_gap");
  const omega = metaNumber(frame, "omega");
  if (omega === undefined) degraded.push("帧 meta 缺 omega：不显示权重（不按请求参数猜）");
  if (fMeasSmooth === undefined || fTheorySmooth === undefined) {
    degraded.push("帧 meta 缺 measured_smoothing_factor / theory_smoothing_factor：高频带因子不写（不自己算）");
  }
  const smootherIdx = metaNumber(frame, "smoother_index");
  const contentIdx = metaNumber(frame, "content_index");
  const statusKey = stringOf(frame, "status");
  const timeSemantics = stringOf(frame, "time_semantics");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: steps,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  // —— 版式（底线钉在 pitfalls 面板之上，见 `_panel.ts` 文件头）——
  const L = panel45Layout(scene, width, height, { topShare: 0.6, auxH: 112 });

  // —— 逐模态数组：按 mode_present 掩码过滤（占位符一律不进图层）——
  const keepIdx: number[] = [];
  for (let i = 0; i < theta.length; i += 1) if ((present[i] ?? 0) > 0.5 && Number.isFinite(theta[i])) keepIdx.push(i);
  const nSkipped = theta.length - keepIdx.length;
  const th = selectValues(theta, keepIdx);
  const a0 = selectValues(amp0, keepIdx);
  const aNow = selectValues(ampNow, keepIdx);
  const aTh = selectValues(ampTheory, keepIdx);
  const fM = selectValues(fMeas, keepIdx);
  const fT = selectValues(fTheory, keepIdx);
  const nBadRaw = countNonFinite(theta) + countNonFinite(amp0) + countNonFinite(ampNow) + countNonFinite(fMeas);

  // —— 主轴：逐模态振幅（对数）——
  const ampLo = Math.min(
    minPositive(a0) ?? Number.POSITIVE_INFINITY,
    minPositive(aNow) ?? Number.POSITIVE_INFINITY,
    minPositive(aTh) ?? Number.POSITIVE_INFINITY,
    floor ?? Number.POSITIVE_INFINITY,
  );
  const ampHi = Math.max(maxFinite(a0) ?? 1, maxFinite(aNow) ?? 1, maxFinite(aTh) ?? 1);
  const spectrumY = Number.isFinite(ampLo) ? logRange(ampLo * 0.6, ampHi * 1.1) : ([1e-17, 10] as [number, number]);
  const spectrum = makeAxis("spectrum", [L.left, L.topY, L.leftW, L.topH], {
    xlim: [0, 1],
    ylim: spectrumY,
    yscale: "log",
    // 上下两块共用同一条 x 轴：上面板清空刻度与轴名，免得与下面板的轴标题挤在同一条带里
    // （`axisDecoBands` 只按「有没有刻度/轴名」占带，清空即不占）。
    xticks: [],
    xlabel: "",
    title: axisTitleOf(SPEC?.axes, "spectrum", "逐模态振幅（对数轴，地板 1e-16）"),
    ylabel: axisYlabelOf(SPEC?.axes, "spectrum", "|c_k|"),
    autoscaled: true,
  });
  const dividerX = firstFinite(halfBand) ?? 0.5;
  const slimRef = firstFinite(smoothLine) ?? 1 / 3;
  // 副图纵轴是**固定窗口 0–1.1**（manifest 声明）：越过 1 就是"这一步把该模态放大了"。
  const factor = makeAxis("factor", [L.left, L.botY, L.leftW, L.botH], {
    xlim: [0, 1],
    ylim: [0, 1.1],
    xticks: [0, 0.25, 0.5, 0.75, 1],
    xlabel: "θ/π（模态 k 的归一化频率）",
    title: axisTitleOf(SPEC?.axes, "factor", "一步放大率（固定窗口 0-1.1：越过 1 就是放大了）"),
    ylabel: axisYlabelOf(SPEC?.axes, "factor", "|lambda_k|"),
    autoscaled: false,
  });
  // 分界竖线画在**两块面板**上：频率分界在谱图上同样成立（右侧整片是高频带）。
  for (const ax of [spectrum, factor]) {
    scene.layers.push(
      lineLayer(
        ax.id,
        series([dividerX, dividerX], [ax.ylim[0], ax.ylim[1]], pal.reference, {
          dash: [6, 4],
          width: 1.2,
        }),
      ),
    );
  }
  // 图层顺序 = 绘制顺序：先参考线，再曲线（曲线压在参考线上才好读）。
  scene.layers.push(
    lineLayer("spectrum", series(th, a0, pal.reference, { dash: [4, 4], width: 1.2})),
    lineLayer("spectrum", series(th, aTh, pal.exact, { dash: [5, 4], width: 1.4})),
    lineLayer("spectrum", series(th, aNow, pal.curves[0], { width: 2.2})),
  );
  if (floor !== undefined && floor > 0) {
    scene.layers.push(
      lineLayer(
        "spectrum",
        series([spectrum.xlim[0], spectrum.xlim[1]], [floor, floor], pal.warn, {
          dash: [2, 3],
          width: 1,
        }),
      ),
    );
  }
  if (atFloor !== undefined) {
    const idx = maskIndicesWhere(atFloor, ampNow, (v) => v > 0);
    if (idx.length > 0) {
      const pts = pick(theta, ampNow, idx);
      scene.layers.push(
        scatterLayer("spectrum", pts.x, pts.y, { r: 3, color: pal.warn, alpha: 0.9, label: "已落到地板的模态" }),
      );
    }
  }

  // —— 副图：一步放大率（固定窗口）+ 1/3 平滑因子线 ——
  scene.layers.push(
    lineLayer(
      "factor",
      series([factor.xlim[0], factor.xlim[1]], [slimRef, slimRef], pal.reference, {
        dash: [5, 4],
        width: 1.2,
      }),
    ),
    lineLayer("factor", series(th, fT, pal.exact, { dash: [5, 4], width: 1.4})),
    lineLayer("factor", series(th, fM, pal.curves[0], { width: 2.2})),
  );

  // —— error 副轴（规则 3）：整体误差范数 vs 迭代步 ——
  const errAxis = makeAxis("error", [L.rightX, L.topY, L.rightW, L.auxH], {
    xlim: [0, Math.max(1, steps)],
    ylim: logRange(minPositive(normHist) ?? 1e-3, Math.max(maxFinite(normHist) ?? 1, 1) * 1.2),
    yscale: "log",
    xticks: Array.from(new Set([0, Math.round(steps / 2), steps])),
    xlabel: "迭代步 k",
    title: "整体误差范数：被最低模态拖住",
    ylabel: "‖e_k‖/‖e_0‖",
    autoscaled: true,
  });
  scene.axes.push(spectrum, factor, errAxis);
  const histX = new Float64Array(normHist.length);
  for (let i = 0; i < normHist.length; i += 1) histX[i] = i;
  scene.layers.push(lineLayer("error", series(histX, normHist, pal.curves[0], { width: 2})));
  if (slowest !== undefined && slowest.length === normHist.length) {
    scene.layers.push(
      lineLayer("error", series(histX, slowest, pal.reference, { dash: [5, 4], width: 1.2})),
    );
  }
  if (normHist.length > 0) {
    const last = normHist.length - 1;
    scene.layers.push(scatterLayer("error", [histX[last]], [normHist[last]], { r: 4, color: pal.marker, label: `当前 k=${last}` }));
  }

  // —— 文本：右列（副轴之下）——
  const smootherName = smootherIdx !== undefined ? SMOOTHERS[Math.round(smootherIdx)] ?? "?" : "?";
  const contentName = contentIdx !== undefined ? CONTENTS[Math.round(contentIdx)] ?? "?" : "?";
  if (L.columns) {
    const budget = unitBudget(L.rightW);
    const head = paramSummary(frame, opts, [
      { name: "n", label: "n" },
      { name: "omega", label: "ω" },
    ]);
    degraded.push(...head.degraded);
    // 每一行都按"显示宽度单位"手拆到 **一行的预算之内**：不靠渲染层再折行
    // （`textBoxOf` 按 maxWidth 折行估盒，手拆后估盒与实际逐行一致）。
    // 每一行都按"显示宽度单位"手拆到**一行的预算之内**：不靠渲染层再折行
    // （`textBoxOf` 按 maxWidth 折行估盒，手拆后估盒与实际逐行一致）。
    // 行数上限 = 右列净空 ÷ 15px；放不下的会被 `textBlock` 截断并如实标注"另有 N 行未画"。
    const lines: string[] = [];
    lines.push(`k=${steps}/${T_MAX}　${head.text}　${smootherName}`);
    lines.push(
      nSkipped > 0
        ? `初值 ${contentName}：只有 ${keepIdx.length}/${theta.length} 个模态非零`
        : `初值 ${contentName}：每个模态都被激发`,
    );
    if (lowSurv !== undefined) lines.push(`最低模态存活 ${formatNumber(lowSurv, 4)}`);
    if (highSurv !== undefined) lines.push(`最高频存活 ${formatNumber(highSurv, 4)}`);
    if (surviveRatio !== undefined && surviveRatio > 1.01) {
      lines.push(`两者相差 ${formatNumber(surviveRatio, 3)} 倍：高频整片塌掉`);
    }
    if (errRatio !== undefined) lines.push(`整体范数只降到 ${formatNumber(errRatio, 4)}：被最低模态拖住`);
    if (errRatio !== undefined) lines.push(`看总范数会以为「平滑几乎没用」`);
    if (errRatio !== undefined) lines.push(`→ 本页第 1 个陷阱（k=0 时两者相等）`);
    if (nFloor !== undefined) lines.push(`已到 ${formatNumber(floor ?? 1e-16, 3)} 地板：${formatNumber(nFloor, 4)}/${theta.length} 个模态`);
    if (nFloor !== undefined && nFloor > 0) lines.push(`贴平地板不是「停了」，是到了舍入尺度`);
    if (fMeasSmooth !== undefined && fTheorySmooth !== undefined) {
      lines.push(`高频带平滑因子 实测 ${formatNumber(fMeasSmooth, 4)} / 理论 ${formatNumber(fTheorySmooth, 4)}`);
      if (fGap !== undefined) lines.push(`差 ${formatNumber(fGap, 3)}（jacobi 下逐位相同）`);
    }
    if (omega !== undefined) lines.push(`ω=${formatNumber(omega, 4)}：高频放大率上限 1/3（灰线）`);
    if (timeSemantics !== undefined) lines.push(`时间轴：k 是迭代步，不是物理时间`);
    scene.texts.push(...textBlock(L.rightX, L.textTop, L.floor, lines, pal.muted, budget, "top"));
  } else {
    degraded.push("窄画布（单列）：右列说明未排下，关键数字见角标与 pitfalls 面板");
  }

  // —— 面板内的图例（按面板宽度折行，构造上不会横穿到右列副轴的装饰带里）——
  scene.texts.push(
    ...legendBlock(L.left + 6, L.topY + 6, L.leftW, `竖线 θ/π=${formatNumber(dividerX, 3)} 右侧是高频带`, pal.reference),
    ...legendBlock(L.left + 6, L.topY + 22, L.leftW, "灰虚=初值　橙虚=理论　蓝实=实测", pal.muted),
    ...legendBlock(L.left + 6, L.botY + 6, L.leftW, "灰虚=1/3 上限　橙虚=理论　蓝实=实测一步放大率", pal.muted),
  );

  // —— 角标（固定槽位）——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  if (scene.dataSource.startsWith("backend+preview")) {
    scene.badges.push(placeBadge(previewBadge("拖动时只重取帧，不在前端插值"), SLOT.preview, width));
  }
  scene.badges.push(placeBadge(axisModeBadge(true), SLOT.axisMode, width));
  scene.badges.push(placeBadge(badge("autoscale", "因子轴：固定 0–1.1（不随帧变）", 8, 26), SLOT.auxAxis, width));
  if (omega !== undefined) {
    scene.badges.push(
      // 值取帧（meta.omega），**理论值取 manifest 的 ω\*=2/3**：
      // 拿帧自己的 critical 当理论值就成了自证（帧是后端算的，这里要的是文档值）。
      placeBadge(criticalBadge("ω", omega, OMEGA_OPTIMAL, { beyond: "above" }), SLOT.critical, width),
    );
  }
  const status: string[] = [];
  if (errRatio !== undefined && lowSurv !== undefined) {
    status.push(`总范数 ${formatNumber(errRatio, 3)} vs 最低 ${formatNumber(lowSurv, 3)}`);
  }
  if (nFloor !== undefined && nFloor > 0) status.push(`（${formatNumber(nFloor, 4)}/${theta.length} 地板）`);
  if (statusKey !== undefined) status.push(`　${statusKey.slice(0, 4)}`);
  if (status.length === 0) status.push("帧未给出可判读的状态（预期有 meta.error_norm_ratio 与 strings.status）");
  scene.badges.push(placeBadge(badge("status", clipBadgeText(status.join("；")), 8, 8), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  // 规则 1：图层越界（因子越过 1.1 就是"这一步放大了"）+ 帧内非有限值，两者都要进 overflow。
  applyOverflow(scene, nBadRaw, { slot: SLOT.overflow, width });
  return scene;
};

// --------------------------------------------------------------------------- //
// 局部工具（只做显示几何，不算物理量）
// --------------------------------------------------------------------------- //

/** 有限正值的最小值；没有就 undefined（对数轴只接受 > 0 的有限值）。 */
function minPositive(a: ArrayLike<number>): number | undefined {
  let lo = Number.POSITIVE_INFINITY;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i];
    if (Number.isFinite(v) && v > 0 && v < lo) lo = v;
  }
  return Number.isFinite(lo) ? lo : undefined;
}

/** 有限值的最大值；没有就 undefined。 */
function maxFinite(a: ArrayLike<number>): number | undefined {
  let hi = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i];
    if (Number.isFinite(v) && v > hi) hi = v;
  }
  return Number.isFinite(hi) ? hi : undefined;
}

/** 数组里第一个有限值（参考线数组是一列常量，取一个就够）。 */
function firstFinite(a: ArrayLike<number> | undefined): number | undefined {
  if (a === undefined) return undefined;
  for (let i = 0; i < a.length; i += 1) if (Number.isFinite(a[i])) return a[i];
  return undefined;
}
