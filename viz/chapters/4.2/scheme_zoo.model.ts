/**
 * `scheme_zoo`（4.2 章）的 model：一次只画**一个**格式真跑出来的解，但所有窗口固定，
 * 所以切换格式时比较的是同一条尺子，而不是各自的自适应纵轴。
 *
 * 图层（顺序即绘制顺序）
 * ----------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | line | `solution` | 精确平移解（橙，参照） |
 * | 2 | line | `solution` | 初值（灰虚线，看它被抹平还是被弄振） |
 * | 3 | line | `solution` | 数值解（蓝，粗） |
 * | 4 | line | `error` | 逐点误差（对数，固定 1e-17–3） |
 * | 5 | line | `error` | 正向误差 u>精确（红，只在 >0 的点画） |
 * | 6 | line | `error` | 负向误差 u<精确（绿，只在 >0 的点画） |
 * | 7 | scatter | `error` | 已落到地板的点（灰点，来自 error_at_floor 掩码） |
 * | 8 | line | `residual` | 三级 RK 回代残差（紫） |
 * | 9 | line | `residual` | euler 与 rk3 的积分差（灰虚线） |
 * | 10 | line | `residual` | 代数残差标量（水平线，仅当 >0） |
 *
 * 三条防误导约定
 * --------------
 * 1. **解窗固定 −0.6–1.6、误差窗固定 1e-17–3**（manifest 声明 `autoscaled=false`）：
 *    发散时曲线冲出窗口是**事实**，用溢出角标与正文报出越窗点数，绝不用自适应把它压回来。
 * 2. **正向/负向误差只在 >0 的点画**：帧里 0 表示该点没有这个方向的误差，把它画到
 *    1e-17 地板上会被读成「有极小的误差」。注意这两个数组是**带符号误差**，不是
 *    区间外过冲；真正的过冲/欠冲取 `meta.overshoot/undershoot`，抹平也会让两侧误差都不为零。
 * 3. **残差副轴与误差轴分开**：`residual_profile` 是最后一步的 RK 回代残差（逐点），
 *    `algebraic_residual` 才是代数残差的 max；两者都不是解误差。把它们画进误差轴
 *    是本章的经典误读。副轴自适应，如实挂角标。
 *
 * 时间轴：`scrub`，`t` 是真实时间；`scene.t` 取 `meta.t_actual`（实到时刻），
 * 与请求值不同时在正文里点名。
 */

import { formatNumber } from "../../shared/src/axes.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import {
  applyOverflow,
  canvasSize,
  clamp,
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
  badge,
  lineLayer,
  makeAxis,
  scatterLayer,
  series,
  sourceBadge,
} from "../../shared/src/scene.js";
import {
  axisTitleOf,
  axisYlabelOf,
  columnsLayout,
  legendBlock,
  logRange,
  maskIndices,
  positiveMinMax,
  textBlock,
  unitBudget,
} from "../_common/panel.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "4.2";
const VIZ = "scheme_zoo";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "四种格式并排：谁在抹平、谁在起振";
const T_MAX = SPEC && SPEC.time.mode !== "none" ? SPEC.time.tMax ?? 0.6 : 0.6;

/** 固定窗口（manifest 的契约；不要跟着帧自适应）。 */
const Y_SOLUTION: [number, number] = [-0.6, 1.6];
const Y_ERROR: [number, number] = [1e-17, 3];
const FLOOR_FALLBACK = 1e-17;

/** 固定角标槽位（缺一枚时其余不挪位，动画不抖）。 */
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

export const schemeZooModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需数组：缺一个就抛 ModelInputError（宁可当场炸，也不画半张图）——
  const x = requireArray(frame, "x");
  const u = requireArray(frame, "u");
  const exact = requireArray(frame, "u_exact");
  const initial = requireArray(frame, "u_initial");
  const errCurve = requireArray(frame, "error_curve");
  const forward = requireArray(frame, "forward_error");
  const backward = requireArray(frame, "backward_error");
  const residual = requireArray(frame, "residual_profile");
  const gap = requireArray(frame, "integrator_gap");
  for (const [name, arr] of [
    ["u", u],
    ["u_exact", exact],
    ["u_initial", initial],
    ["error_curve", errCurve],
    ["forward_error", forward],
    ["backward_error", backward],
    ["residual_profile", residual],
    ["integrator_gap", gap],
  ] as const) {
    requireSameLength(x, arr, `帧数组 x 与 ${name}`);
  }
  const pitfalls = requirePitfalls(frame);

  // —— 可选数组：缺了只少画标注，绝不猜 ——
  const degraded: string[] = [];
  const uClipped = frame.has("u_clipped") ? requireArray(frame, "u_clipped") : undefined;
  if (uClipped === undefined) degraded.push("帧缺 u_clipped：不统计截断点数（不按 |u|>100 自己猜）");
  else requireSameLength(x, uClipped, "帧数组 x 与 u_clipped");
  const atFloor = frame.has("error_at_floor") ? requireArray(frame, "error_at_floor") : undefined;
  if (atFloor === undefined) degraded.push("帧缺 error_at_floor：不标贴地板的点（地板仍画）");
  else requireSameLength(x, atFloor, "帧数组 x 与 error_at_floor");
  const algRes = frame.has("algebraic_residual") ? requireArray(frame, "algebraic_residual") : undefined;
  if (algRes === undefined) degraded.push("帧缺 algebraic_residual：不画代数残差水平线");

  // —— meta（缺了只影响标注）——
  const tActualMeta = metaNumber(frame, "t_actual");
  const tRequested = metaNumber(frame, "t_requested") ?? opts.t;
  const tActual = tActualMeta ?? clamp(opts.t, 0, T_MAX);
  if (tActualMeta === undefined) degraded.push("帧 meta 缺 t_actual：时间取请求值并夹到 [0,tMax]");
  else if (Math.abs(tActual - clamp(tRequested, 0, T_MAX)) > 1e-9) {
    degraded.push(`请求 t=${formatNumber(tRequested, 4)} 实到 ${formatNumber(tActual, 4)}（步数取整）`);
  }
  const nu = metaNumber(frame, "nu");
  if (nu === undefined) degraded.push("帧 meta 缺 nu：不显示 CFL 数（不按请求参数猜）");
  const nx = metaNumber(frame, "nx");
  if (nx === undefined) degraded.push("帧 meta 缺 nx：不显示格点数");
  const maxG = metaNumber(frame, "max_abs_symbol");
  if (maxG === undefined) degraded.push("帧 meta 缺 max_abs_symbol：稳定性只在正文明说「未给出」");
  const l2 = metaNumber(frame, "l2_error");
  const linf = metaNumber(frame, "linf_error");
  const tvNow = metaNumber(frame, "total_variation");
  const tvInit = metaNumber(frame, "tv_initial");
  const over = metaNumber(frame, "overshoot");
  const under = metaNumber(frame, "undershoot");
  const amplified = frame.meta["amplified_per_step"] === true;
  const overflowMeta = frame.meta["overflow"] === true;
  const scheme = stringOf(frame, "scheme");
  const initialName = stringOf(frame, "initial");
  const integrator = stringOf(frame, "integrator");
  const floorText = stringOf(frame, "floor");
  const floor = numberOr(floorText, FLOOR_FALLBACK);
  if (floorText === undefined) degraded.push("帧 strings 缺 floor：地板标记按 1e-17（manifest 窗口下界）");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: tActual,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  // —— 版式：两列（左：解 + 误差；右：残差 + 文字）；窄画布退化成三块通栏面板 ——
  const L = columnsLayout(scene, width, height, { topShare: 0.6, auxH: 120, gap: 46 });
  const narrow = !L.columns;
  const panels = narrow ? narrowStack(L.floor, L.topY, width, 3) : undefined;

  const solution = makeAxis(
    "solution",
    panels ? panels[0] : [L.left, L.topY, L.leftW, L.topH],
    {
      xlim: [0, 1],
      ylim: Y_SOLUTION,
      // 上面板与误差面板共用 x 轴：清空刻度/轴名，免得两套标签挤在同一条带里。
      xticks: [],
      xlabel: "",
      title: axisTitleOf(SPEC?.axes, "solution", "解曲线与精确平移解"),
      ylabel: axisYlabelOf(SPEC?.axes, "solution", "$u(x,t)$"),
      autoscaled: false,
    },
  );
  const error = makeAxis(
    "error",
    panels ? panels[1] : [L.left, L.botY, L.leftW, L.botH],
    {
      xlim: [0, 1],
      ylim: Y_ERROR,
      yscale: "log",
      xticks: narrow ? [] : [0, 0.25, 0.5, 0.75, 1],
      xlabel: narrow ? "" : "x（格点，域长 L=1）",
      title: axisTitleOf(SPEC?.axes, "error", "逐点误差（对数纵轴）与过冲/欠冲"),
      ylabel: axisYlabelOf(SPEC?.axes, "error", "|u − u_exact|"),
      autoscaled: false,
    },
  );
  const resLo = positiveMinMax(residual)?.[0] ?? floor;
  const gapLo = positiveMinMax(gap)?.[0] ?? floor;
  const algMax = algRes !== undefined && algRes.length > 0 ? maxFinite(algRes) : undefined;
  const resHi = Math.max(
    positiveMinMax(residual)?.[1] ?? 1,
    positiveMinMax(gap)?.[1] ?? 1,
    algMax !== undefined && algMax > 0 ? algMax : 0,
  );
  const residualAxis = makeAxis(
    "residual",
    panels ? panels[2] : [L.rightX, L.topY, L.rightW, L.auxH],
    {
      xlim: [0, 1],
      ylim: logRange(Math.min(resLo, gapLo) * 0.5, resHi * 2),
      yscale: "log",
      xticks: narrow ? [] : [0, 0.25, 0.5, 0.75, 1],
      xlabel: narrow ? "" : "x",
      title: "残差：代数 − 积分器（对数）",
      ylabel: "$|r|$",
      autoscaled: true,
    },
  );
  if (narrow) {
    error.title = "";
    residualAxis.title = "";
  }
  scene.axes.push(solution, error, residualAxis);

  // —— 主面板：精确 / 初值 / 数值 ——
  scene.layers.push(
    lineLayer("solution", series(x, exact, pal.exact, { width: 2, label: "精确解" })),
    lineLayer("solution", series(x, initial, pal.reference, { width: 1.2, dash: [4, 4], label: "初值" })),
    lineLayer("solution", series(x, u, pal.curves[0], { width: 2.4, label: "数值解" })),
  );

  // —— 误差面板：逐点误差 + 只画 >0 的正/负向误差 + 地板散点 ——
  scene.layers.push(
    lineLayer("error", series(x, errCurve, pal.curves[0], { width: 2, label: "误差 $|u − u_exact|$" })),
    lineLayer("error", series([0, 1], [floor, floor], pal.reference, { width: 1, dash: [2, 3], label: "舍入地板" })),
  );
  const overSeries = segmentedSeries(x, forward, pal.warn, { accept: (v) => v >= floor, width: 1.6, label: "正向误差（数值偏高）" });
  if (overSeries.length > 0) scene.layers.push(lineLayer("error", overSeries));
  const underSeries = segmentedSeries(x, backward, pal.curves[1], { accept: (v) => v >= floor, width: 1.6, label: "负向误差（数值偏低）" });
  if (underSeries.length > 0) scene.layers.push(lineLayer("error", underSeries));
  if (atFloor !== undefined) {
    const idx = maskIndices(atFloor);
    if (idx.length > 0) {
      const xs = new Float64Array(idx.length);
      const ys = new Float64Array(idx.length);
      idx.forEach((i, k) => {
        xs[k] = x[i] as number;
        ys[k] = floor;
      });
      scene.layers.push(scatterLayer("error", xs, ys, { r: 2.4, color: pal.warn, alpha: 0.85 }));
    }
  }

  // —— 残差副轴：不是解误差，是最后一步的代数残差 ——
  scene.layers.push(
    lineLayer("residual", series(x, residual, pal.residual, { width: 1.8, label: "代数残差" })),
    lineLayer("residual", series(x, gap, pal.reference, { width: 1.2, dash: [5, 4], label: "积分器之差" })),
  );
  if (algMax !== undefined && algMax > 0) {
    scene.layers.push(
      lineLayer("residual", series([0, 1], [Math.max(algMax, floor), Math.max(algMax, floor)], pal.warn, {
        width: 1,
        dash: [4, 3],
      })),
    );
  }

  // —— 面板内图例（按面板宽度折行，绝不横穿到右列的装饰带）——
  if (!narrow) {
    scene.texts.push(
      ...legendBlock(L.left + 6, L.topY + 6, L.leftW, "橙=精确平移解　灰虚=初值　蓝=数值解", pal.muted),
      ...legendBlock(
        L.left + 6,
        L.botY + 6,
        L.leftW,
        "红=正向误差 u>精确　绿=负向误差 u<精确（只在 ≥地板 处画）　灰点=贴地板",
        pal.muted,
      ),
      ...legendBlock(
        L.rightX + 6,
        L.topY + 6,
        L.rightW,
        algMax !== undefined && algMax > 0 ? "紫=回代残差　灰虚=积分器之差　红虚=代数残差" : "紫=回代残差　灰虚=积分器之差",
        pal.muted,
      ),
      ...legendBlock(L.rightX + 6, L.topY + 38, L.rightW, "对数：贴地板=舍入级，不是 0", pal.muted),
    );
  }

  // —— 右列文字（窄画布放弃，由 degraded 点名）——
  const nFloor = atFloor !== undefined ? countMask(atFloor) : 0;
  const nClip = uClipped !== undefined ? countMask(uClipped) : 0;
  const nAbove = countAbove(errCurve, Y_ERROR[1]);
  const nOver = countAtLeast(forward, floor);
  const nUnder = countAtLeast(backward, floor);
  if (!narrow) {
    const head = paramSummary(frame, opts, [
      { name: "nu", label: "$ν$" },
      { name: "nx", label: "$nx$" },
    ]);
    degraded.push(...head.degraded);
    const lines: string[] = [];
    lines.push(`${scheme ?? "?"}/${initialName ?? "?"}/${integrator ?? "?"}　${head.text}`);
    lines.push(`t=${formatNumber(tActual, 4)}（请求 ${formatNumber(tRequested, 4)}）`);
    if (l2 !== undefined) lines.push(`L2 误差 ${formatNumber(l2, 4)}`);
    if (linf !== undefined) lines.push(`L∞ 误差 ${formatNumber(linf, 4)}`);
    if (tvNow !== undefined) {
      lines.push(`TV=${formatNumber(tvNow, 5)}（初值 ${fmt(tvInit, 4)}）`);
    }
    if (maxG !== undefined) {
      lines.push(`max|G|=${formatNumber(maxG, 5)}${amplified ? "：每步都在放大" : "：不放大"}`);
    }
    lines.push(`区间外过冲 ${fmt(over, 3)} / 欠冲 ${fmt(under, 3)}`);
    const smeared = (over ?? 0) <= 0 && (under ?? 0) <= 0;
    lines.push(
      nOver === 0 && nUnder === 0
        ? "没有 ≥ 地板的单向误差点"
        : smeared
          ? `正/负向误差 ${nOver}/${nUnder} 点（抹平，无区间外过冲）`
          : `正/负向误差 ${nOver}/${nUnder} 点（过冲/欠冲见上）`,
    );
    if (nClip > 0) lines.push(`解截断 ${nClip}/${x.length} 点（|u|>100）`);
    if (nAbove > 0) lines.push(`误差 ${nAbove} 点 >3（越窗，不是 3）`);
    if (nFloor > 0) lines.push(`${nFloor}/${x.length} 点贴地板：舍入级，不是 0`);
    lines.push("窗口固定：解 −0.6–1.6；误差 1e-17–3");
    scene.texts.push(...textBlock(L.rightX, L.textTop, L.floor, lines, pal.muted, unitBudget(L.rightW), "top"));
  } else {
    degraded.push("窄画布（单列）：右列说明与面板图例未排下，窗口仍固定");
  }

  // —— 角标 ——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  scene.badges.push(
    placeBadge(
      badge("autoscale", "主图固定 −0.6–1.6 / 1e-17–3；残差副轴自适应", 8, 26),
      SLOT.axisMode,
      width,
    ),
  );
  scene.badges.push(
    placeBadge(badge("autoscale", `解窗 ${Y_SOLUTION[0]}–${Y_SOLUTION[1]}；误差窗 1e-17–3（不随帧变）`, 8, 44), SLOT.auxAxis, width),
  );
  if (nu !== undefined) {
    // 0.5 只是某些格式的 a₃ 变号点，不是稳定界；真正的 CFL 界是 1.0。
    const cfl = nu > 1.0 + 1e-12 ? "⚠ 越过 CFL 界 1.0" : "CFL 界 1.0";
    scene.badges.push(
      placeBadge(badge("critical", `ν=${formatNumber(nu, 3)}　1/2 仅是迎风 a3 变号点；${cfl}`, 8, 62), SLOT.critical, width),
    );
  }
  const status: string[] = [`t=${formatNumber(tActual, 3)}`];
  if (l2 !== undefined) status.push(`L2 ${formatNumber(l2, 3)}`);
  if (tvNow !== undefined) status.push(`TV ${formatNumber(tvNow, 3)}`);
  if (maxG !== undefined) status.push(`max|G| ${formatNumber(maxG, 3)}`);
  status.push(`区间外过冲 ${fmt(over, 3)} / 欠冲 ${fmt(under, 3)}`);
  scene.badges.push(placeBadge(badge("status", status.join("；"), 8, 8), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  const nBad =
    countNonFinite(x) +
    countNonFinite(u) +
    countNonFinite(exact) +
    countNonFinite(errCurve) +
    countNonFinite(residual) +
    (uClipped !== undefined ? countNonFinite(uClipped) : 0) +
    (atFloor !== undefined ? countNonFinite(atFloor) : 0) +
    (algRes !== undefined ? countNonFinite(algRes) : 0);
  applyOverflow(scene, nBad, { slot: SLOT.overflow, width });
  if (overflowMeta && !scene.overflow.flag) {
    // 帧自己报了 overflow（例如解被截断后恰好落在窗口内）：也要挂角标，不能静默。
    scene.overflow = { flag: true, note: "帧 meta.overflow=true：解在显示上限处被截断，窗口内可能看不到真实幅度" };
    scene.badges.push(placeBadge(badge("overflow", `溢出：${scene.overflow.note}`, 8, 44), SLOT.overflow, width));
  }
  return scene;
};

// --------------------------------------------------------------------------- //
// 局部工具（只做显示几何与文案，不算物理量）
// --------------------------------------------------------------------------- //

/** 窄画布：把 n 块面板通栏堆在 [top, floor] 内（不做两列，说明文字由 degraded 点名）。 */
function narrowStack(
  floor: number,
  top: number,
  width: number,
  n: number,
): [number, number, number, number][] {
  const gap = 12;
  const avail = Math.max(40, floor - top);
  const h = Math.max(20, Math.floor((avail - gap * (n - 1)) / n));
  const w = Math.max(120, width - 88);
  const out: [number, number, number, number][] = [];
  for (let i = 0; i < n; i += 1) out.push([64, top + i * (h + gap), w, h]);
  return out;
}

/** 统计掩码（>0.5）的个数。 */
function countMask(a: ArrayLike<number>): number {
  let n = 0;
  for (let i = 0; i < a.length; i += 1) if ((a[i] ?? 0) > 0.5) n += 1;
  return n;
}

function countAtLeast(a: ArrayLike<number>, floor: number): number {
  let n = 0;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i];
    if (Number.isFinite(v) && v >= floor) n += 1;
  }
  return n;
}

function countAbove(a: ArrayLike<number>, limit: number): number {
  let n = 0;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i];
    if (Number.isFinite(v) && v > limit) n += 1;
  }
  return n;
}

function maxFinite(a: ArrayLike<number>): number | undefined {
  let hi = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i];
    if (Number.isFinite(v) && v > hi) hi = v;
  }
  return Number.isFinite(hi) ? hi : undefined;
}

function fmt(v: number | undefined, digits: number): string {
  return v === undefined || !Number.isFinite(v) ? "?" : formatNumber(v, digits);
}

function numberOr(text: string | undefined, fallback: number): number {
  if (text === undefined) return fallback;
  const v = Number(text);
  return Number.isFinite(v) && v > 0 ? v : fallback;
}
