/**
 * `stability_threshold`（4.1 章招牌 viz）的 model：真跑出来的解 + 放大因子 |G(φ)|。
 *
 * 三层证据同屏，缺一条就会回到"看图说话"
 * ----------------------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | line | `solution` | 数值解 `u`（实线）与参考解 `u_ref`（虚线）——**同一段代码真跑出来的** |
 * | 2 | scatter | `solution` | 被截断显示的格点（`u_clipped=1`）钉在窗口边界，用 × 标出 |
 * | 3 | line | `amplification` | 三条 `|G(φ)|` 曲线（当前格式 + 两个对照格式）与 `|G|=1` 界线 |
 * | 4 | band | `amplification` | `g_unstable_region=1` 的波数段：`1 ≤ |G| ≤ 2.4` 的放大区 |
 * | 5 | line | `amplification` | 幂迭代**实测**谱半径的水平线 + 符号翻转波数 φ* 的竖线 |
 * | 6 | line | `error` | `|u−参考解|`（对数轴）+ 地板线与上一步代数残差线 |
 *
 * 时间轴（manifest `time.mode="scrub"`，tMax=0.01，真实时间）
 * ---------------------------------------------------------
 * 滑块是**真实时间**：帧 `meta.t` 是请求时刻，`meta.t_actual = n_steps·dt` 才是数组真正
 * 到达的时刻（`n_steps = round(t/dt)`，所以两者一般不相等）。`scene.t` 用 `meta.t`，
 * 面板文字同时给出两个数；相差超过 `dt/2` 时用 ⚠ 明说"曲线是 t_actual 时刻的解"。
 *
 * 防误导七条里本章的重点
 * ----------------------
 * * **规则 1**：`solution` 轴窗口**写死 ±1.5**。发散时曲线冲出画面——而不是被自动缩放
 *   藏起来；超过 `meta.umax > 1e3` 的点在帧里已被截断到 ±1e3，`applyOverflow` 会把
 *   "有几处越界、多少倍于跨度"写进 `overflow.note` 并挂角标。
 * * **规则 6**：`r` 的理论临界值 0.5（`frame.critical.r`）恒挂 `criticalBadge`，
 *   越界时角标变红——**判据是 |G|，不是主图好不好看**。
 * * **规则 7**：两条最容易读反的写进面板文字：① `initial=sine` 时越界的高频模态根本没
 *   被激发，主图可以非常漂亮（实测 r=0.51 的误差比稳定的 r=0.45 只大一点）；
 *   ② 稳定（`|G|≤1`）与精确是两件事（BTCS 在 r=50 稳如泰山，误差是 r=0.4 的 86.2 倍）。
 */

import { formatNumber, niceLimits, niceTicks } from "../../shared/src/axes.js";
import type { Frame } from "../../shared/src/frame.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import {
  applyOverflow,
  canvasSize,
  criticalOf,
  dataSourceOf,
  degradedBadge,
  finiteMinMax,
  finitePositiveMinMax,
  metaNumber,
  palette,
  paramSummary,
  placeBadge,
  requireArray,
  requireDrawableLayers,
  requirePitfalls,
  requireSameLength,
  resolveTextOverlaps,
  sceneShell,
  segmentedSeries,
  stringOf,
  type SceneModel,
} from "../../shared/src/model.js";
import {
  axisModeBadge,
  badge,
  bandLayer,
  criticalBadge,
  lineLayer,
  makeAxis,
  pitfallsBand,
  previewBadge,
  scatterLayer,
  series,
  sourceBadge,
} from "../../shared/src/scene.js";
import { manifest } from "./manifest.gen.js";
import {
  arrayLabelOf,
  axisAutoscaledOf,
  bandLayout,
  axisTitleFitted,
  axisYlabelOf,
  endLabel,
  endLabelOf,
  countMask,
  contentBox,
  hLine,
  splitColumns,
  textBlock,
  vLine,
} from "./_panel.js";

const CHAPTER = "4.1";
const VIZ = "stability_threshold";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "稳定性阈值：显式 vs 隐式 vs Crank–Nicolson";
/** `frame.critical.r` 缺失时的回落值（manifest 里 `r.critical = [0.5]`）。 */
const R_CRITICAL = 0.5;
/** 主图窗口：manifest 的轴标题写着"固定 ±1.5"——发散时冲出去，而不是缩轴藏起来。 */
const SOLUTION_YLIM: [number, number] = [-1.5, 1.5];
/** 副图窗口：`|G|` 的上限 2.4（FTCS 在 r=0.8 时 |G|max = 2.2，留一点余量）。 */
const AMP_YLIM: [number, number] = [0, 2.4];
/** 帧 `strings.floor`（对数轴地板）缺失时的回落值，与后端 `_ERR_FLOOR` 同值。 */
const FLOOR_FALLBACK = 1e-17;

/** 固定角标槽位（`model.ts::BADGE_LAYOUT`：两列 4 行 × 2 列）。 */
const SLOT = {
  source: 0,
  axisMode: 1,
  auxAxis: 2,
  critical: 3,
  overflow: 4,
  preview: 5,
  status: 6,
  degraded: 7,
} as const;

export const stabilityThresholdModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需数组：画不出来就抛 ModelInputError，不画半张图 ——
  const x = requireArray(frame, "x");
  const u = requireArray(frame, "u");
  const uRef = requireArray(frame, "u_ref");
  const errCurve = requireArray(frame, "error_curve");
  const phi = requireArray(frame, "phi");
  const gCur = requireArray(frame, "g_abs_cur");
  const gA = requireArray(frame, "g_abs_other_a");
  const gB = requireArray(frame, "g_abs_other_b");
  const gOne = requireArray(frame, "g_one");
  requireSameLength(x, u, "帧数组 x 与 u");
  requireSameLength(x, uRef, "帧数组 x 与 u_ref");
  requireSameLength(x, errCurve, "帧数组 x 与 error_curve");
  requireSameLength(phi, gCur, "帧数组 phi 与 g_abs_cur");
  requireSameLength(phi, gA, "帧数组 phi 与 g_abs_other_a");
  requireSameLength(phi, gB, "帧数组 phi 与 g_abs_other_b");
  requireSameLength(phi, gOne, "帧数组 phi 与 g_one");
  const pitfalls = requirePitfalls(frame);

  // —— 可选数组：缺了降级，不猜 ——
  const degraded: string[] = [];
  const uClipped = optional(frame, "u_clipped");
  if (uClipped === undefined) degraded.push("帧缺数组 u_clipped：被截断显示的点不单独标出");
  const errAtFloor = optional(frame, "error_at_floor");
  if (errAtFloor === undefined) degraded.push("帧缺数组 error_at_floor：不报「贴地板」的点数");
  const gUnstable = optional(frame, "g_unstable_region");
  if (gUnstable === undefined) degraded.push("帧缺数组 g_unstable_region：不画放大区高亮（|G|=1 界线仍在）");

  // —— 标量 meta：缺了只少一条标注，不抛错（PROTOCOLS §1.3：null 是"没有有限值"）——
  const tMeta = metaNumber(frame, "t");
  if (tMeta === undefined) degraded.push("帧 meta 缺 t：改用请求 t（两者可能不同）");
  const tNow = tMeta ?? opts.t;
  const dt = metaNumber(frame, "dt");
  const tActual = metaNumber(frame, "t_actual");
  const nSteps = metaNumber(frame, "n_steps");
  if (dt === undefined && tActual === undefined) {
    degraded.push("帧 meta 缺 dt 与 t_actual：不判断请求时刻是否到达、也不比对时刻（不假设相等）");
  } else if (dt === undefined) {
    degraded.push("帧 meta 缺 dt：不判断请求时刻是否到达");
  } else if (tActual === undefined) {
    degraded.push("帧 meta 缺 t_actual：不比对时刻（不假设相等）");
  }
  const r = metaNumber(frame, "r");
  if (r === undefined) degraded.push("帧 meta 缺 r：不标 r 的临界值（不猜）");
  const theta = metaNumber(frame, "theta");
  const nx = metaNumber(frame, "nx");
  const measured = metaNumber(frame, "measured_radius");
  const analytic = metaNumber(frame, "analytic_radius");
  const umax = metaNumber(frame, "umax");
  const clipped = metaNumber(frame, "clipped_points");
  const unstableModes = metaNumber(frame, "unstable_modes");
  const totalModes = metaNumber(frame, "total_modes");
  const errLinf = metaNumber(frame, "err_linf");
  const errL2 = metaNumber(frame, "err_l2");
  const phiFlip = metaNumber(frame, "phi_flip");
  const hasFlip = frame.meta["has_flip"] === true;
  const overflowFlag = frame.meta["overflow"] === true;

  const scheme = stringOf(frame, "scheme");
  const bc = stringOf(frame, "bc");
  const initial = stringOf(frame, "initial");
  const floor = numberFromString(stringOf(frame, "floor")) ?? FLOOR_FALLBACK;
  const errorReference = stringOf(frame, "error_reference");
  const grad = paramSummary(frame, opts, [{ name: "r", label: "$r$" }]);
  degraded.push(...grad.degraded);

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: tNow,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  // —— 排版：上排「解 | 放大因子」，下排左「误差」；底部通栏数值块（见 `bandLayout`）——
  const box = contentBox(width, height, pitfallsBand(scene)?.[1] ?? null);
  const cols = splitColumns(box, 0.60);
  const band = bandLayout(box, { botH: Math.max(70, Math.round(box.height * 0.22)), rows: 6 });
  const solutionRect: [number, number, number, number] = [cols.left.left, band.topY, cols.left.width, band.topH];
  const ampRect: [number, number, number, number] = [cols.right.left, band.topY, cols.right.width, band.topH];
  const errorRect: [number, number, number, number] = [cols.left.left, band.botY, cols.left.width, band.botH];

  // —— 轴 ——
  const xRange = finiteMinMax(x) ?? [0, 1];
  const xlim = niceLimits(xRange[0], xRange[1], 6);
  // φ 的定义域就是 [0, π]：轴窗口取数据范围本身（不 nice 取整到 4，白占一截空白）。
  const phiRange = finiteMinMax(phi) ?? [0, Math.PI];
  const phiXlim: [number, number] =
    Number.isFinite(phiRange[0]) && Number.isFinite(phiRange[1]) && phiRange[1] > phiRange[0]
      ? [phiRange[0], phiRange[1]]
      : [0, Math.PI];

  const solutionAxis = makeAxis("solution", solutionRect, {
    xlim,
    ylim: SOLUTION_YLIM,
    ylabel: axisYlabelOf(SPEC, "solution", "u(x,t)"),
    title: axisTitleFitted(SPEC, "solution", "解 u(x,t)（窗口固定 ±1.5）", solutionRect[2]),
    autoscaled: axisAutoscaledOf(SPEC, "solution", false),
    // 上排清空 xticks：下面那块误差面板是同一根 x 轴，刻度由它出（省掉一整条装饰带，
    // 也避免上排刻度带与下排轴标题带对撞，见 `_panel.ts::bandLayout`）。
    xticks: [],
  });

  // 残差画在对数轴上：显式格式的代数残差**恰好是 0**（帧 meta 也这么说），
  // 对数轴画不出 0，所以把它抬到地板并如实标注——不是静默丢掉。
  const residual = optional(frame, "algebraic_residual");
  const residualRaw = residual !== undefined && residual.length > 0 ? (residual[0] as number) : undefined;
  const residualShown = residualRaw === undefined ? undefined : Math.max(residualRaw, floor);

  const errPos = finitePositiveMinMax(errCurve);
  const lo = Math.min(errPos?.[0] ?? Number.POSITIVE_INFINITY, floor, residualShown ?? Number.POSITIVE_INFINITY);
  const hi = Math.max(errPos?.[1] ?? Number.NEGATIVE_INFINITY, residualShown ?? 0);
  const errYlim: [number, number] =
    Number.isFinite(lo) && Number.isFinite(hi) && hi > lo ? niceLimits(lo, hi, 6, { log: true }) : [floor, 1];

  const errorAxis = makeAxis("error", errorRect, {
    xlim,
    ylim: errYlim,
    yscale: "log",
    ylabel: "$|u$−参考解|",
    title: "误差与上一步代数残差（对数轴）",
    autoscaled: true,
    xticks: niceTicks(xlim[0], xlim[1], 6),
  });

  const ampAxis = makeAxis("amplification", ampRect, {
    xlim: phiXlim,
    ylim: AMP_YLIM,
    ylabel: axisYlabelOf(SPEC, "amplification", "|G|"),
    title: axisTitleFitted(SPEC, "amplification", "放大因子 |G(φ)|（窗口 0–2.4）", ampRect[2]),
    autoscaled: axisAutoscaledOf(SPEC, "amplification", false),
    xticks: niceTicks(phiXlim[0], phiXlim[1], 5),
  });
  scene.axes.push(solutionAxis, errorAxis, ampAxis);

  // —— 图层 1~2：解与参考解；被截断的点钉在窗口边界（用 × 标出，不是丢掉）——
  const uSeries = segmentedSeries(x, u, pal.curves[0], {
    width: 1.8,
    label: endLabelOf(x, xlim, solutionRect[2], arrayLabelOf(frame, "u", `数值解 u（${scheme ?? "当前格式"}）`)),
  });
  const refSeries = segmentedSeries(x, uRef, pal.exact, {
    width: 1.3,
    dash: [6, 3],
    label: endLabelOf(x, xlim, solutionRect[2], arrayLabelOf(frame, "u_ref", "参考解")),
  });
  if (uSeries.length + refSeries.length > 0) scene.layers.push(lineLayer("solution", [...uSeries, ...refSeries]));
  else degraded.push("帧数组 u/u_ref 没有可绘的有限值：主面板不画曲线");

  if (uClipped !== undefined && clipped !== undefined && clipped > 0) {
    const cx: number[] = [];
    const cy: number[] = [];
    for (let i = 0; i < Math.min(x.length, uClipped.length); i += 1) {
      if ((uClipped[i] as number) > 0.5 && Number.isFinite(x[i] as number)) {
        cx.push(x[i] as number);
        cy.push((u[i] as number) >= 0 ? SOLUTION_YLIM[1] : SOLUTION_YLIM[0]);
      }
    }
    if (cx.length > 0) {
      scene.layers.push(
        scatterLayer("solution", cx, cy, {
          r: 3,
          color: pal.marker,
          label: endLabelOf(cx, xlim, solutionRect[2], `已截断显示（${cx.length} 点，真值见 meta.umax）`),
        }),
      );
    }
  }

  // —— 图层 3~5：三条 |G| 曲线 + 稳定界线 + 实测谱半径 + 符号翻转波数 ——
  const ampSeries = [
    ...segmentedSeries(phi, gCur, pal.curves[0], {
      width: 2,
      label: endLabelOf(phi, phiXlim, ampRect[2], arrayLabelOf(frame, "g_abs_cur", "|G(φ)|（当前格式）")),
    }),
    ...segmentedSeries(phi, gA, pal.muted, {
      width: 1,
      alpha: 0.85,
      label: endLabelOf(phi, phiXlim, ampRect[2], arrayLabelOf(frame, "g_abs_other_a", "|G(φ)|（对照）")),
    }),
    ...segmentedSeries(phi, gB, pal.reference, {
      width: 1,
      alpha: 0.85,
      label: endLabelOf(phi, phiXlim, ampRect[2], arrayLabelOf(frame, "g_abs_other_b", "|G(φ)|（对照）")),
    }),
    ...segmentedSeries(phi, gOne, pal.warn, {
      width: 1.2,
      dash: [5, 3],
      label: endLabelOf(phi, phiXlim, ampRect[2], arrayLabelOf(frame, "g_one", "稳定界线 |G| = 1")),
    }),
  ];
  scene.layers.push(lineLayer("amplification", ampSeries));

  if (gUnstable !== undefined) {
    const bx: number[] = [];
    const blo: number[] = [];
    const bhi: number[] = [];
    for (let i = 0; i < Math.min(phi.length, gUnstable.length, gCur.length); i += 1) {
      if ((gUnstable[i] as number) <= 0.5) continue;
      const g = gCur[i] as number;
      if (!Number.isFinite(g)) continue;
      // 画在 [max(1, |G|) 下界, min(2.4, |G|)] 上：只强调"越到 1 以上"的那一段。
      bx.push(phi[i] as number);
      blo.push(1);
      bhi.push(Math.min(AMP_YLIM[1], Math.max(1, g)));
    }
    if (bx.length > 0) {
      scene.layers.push(
        bandLayer("amplification", { x: bx, ylo: blo, yhi: bhi, color: pal.warn, alpha: 0.22, label: "$|G|>1$ 的放大区" }),
      );
    }
  }

  const markers: ReturnType<typeof series>[] = [];
  if (measured !== undefined && Number.isFinite(measured) && measured > 0) {
    const line = hLine(phiXlim[0], phiXlim[1], measured);
    markers.push(
      series(line.x, line.y, pal.marker, {
        width: 1.2,
        dash: [3, 3],
        label: endLabelOf(line.x, phiXlim, ampRect[2], `实测谱半径 ${formatNumber(measured, 6)}（幂迭代）`),
      }),
    );
  }
  if (hasFlip && phiFlip !== undefined && phiFlip >= phiXlim[0] && phiFlip <= phiXlim[1]) {
    const line = vLine(phiFlip, AMP_YLIM[0], AMP_YLIM[1]);
    markers.push(
      series(line.x, line.y, pal.note, {
        width: 1,
        dash: [2, 3],

      }),
    );
  }
  if (markers.length > 0) scene.layers.push(lineLayer("amplification", markers));

  // —— 图层 6：逐点误差 + 地板线 + 上一步代数残差线（requiredAux：error / residual）——
  const errSeries = segmentedSeries(x, errCurve, pal.residual, {
    width: 1.4,
    accept: (v) => v > 0,
    label: endLabelOf(x, xlim, errorRect[2], arrayLabelOf(frame, "error_curve", "|u−参考解|")),
  });
  const floorLine = hLine(xlim[0], xlim[1], floor);
  const errGuide = [
    series(floorLine.x, floorLine.y, pal.reference, {
      width: 1,
      dash: [2, 3],
      label: endLabel(floorLine.x[1], xlim, errorRect[2], `地板 ${formatNumber(floor, 2)}`),
    }),
  ];
  if (residualShown !== undefined && residualShown > 0) {
    const resLine = hLine(xlim[0], xlim[1], residualShown);
    const resLabel =
      residualRaw !== undefined && residualRaw === 0
        ? "上一步代数残差 = 0（画在地板处）"
        : `上一步代数残差 ${formatNumber(residualRaw ?? 0, 4)}`;
    errGuide.push(
      series(resLine.x, resLine.y, pal.note, {
        width: 1.2,
        dash: [6, 3],
        label: endLabel(resLine.x[1], xlim, errorRect[2], resLabel),
      }),
    );
  }
  if (errSeries.length > 0) scene.layers.push(lineLayer("error", errSeries));
  scene.layers.push(lineLayer("error", errGuide));

  // —— 文本：底部通栏（**先数字、后图例**；装不下时 `textBlock` 砍掉的总是最后那行图例）——
  const head: string[] = [];
  if (overflowFlag || (clipped !== undefined && clipped > 0)) {
    head.push(
      `⚠ 有 ${formatNumber(clipped ?? 0, 6)} 个点超出窗口被截断显示（真值 meta.umax=${formatNumber(umax ?? 0, 6)}）：` +
        "截断不等于没有发生，主图冲出去正是「发散」该有的样子",
    );
  }
  head.push(
    `参数：格式=${scheme ?? "?"}（θ=${formatNumber(theta ?? NaN, 3)}）　边界=${bc ?? "?"}　初值=${initial ?? "?"}` +
      `　${grad.text}　nx=${formatNumber(nx ?? 0, 6)}　dt=${formatNumber(dt ?? NaN, 4)}　n_steps=${formatNumber(nSteps ?? 0, 6)}`,
  );
  head.push(
    `时刻：请求 t=${formatNumber(tNow, 4)} / 实际到达 t_actual=${formatNumber(tActual ?? NaN, 4)}` +
      `（真实时间；曲线画的是 t_actual，两者一般不相等）`,
  );
  if (measured !== undefined) {
    head.push(
      `实测谱半径 ${formatNumber(measured, 6)}（幂迭代）　解析有限网格值 ${formatNumber(analytic ?? NaN, 6)}` +
        `　差 ${formatNumber(Math.abs(measured - (analytic ?? measured)), 3)}` +
        `　被放大的模态 ${formatNumber(unstableModes ?? 0, 6)}/${formatNumber(totalModes ?? 0, 6)}`,
    );
  }
  const nFloor = errAtFloor === undefined ? 0 : countMask(errAtFloor);
  const errs =
    `误差 L∞ ${formatNumber(errLinf ?? NaN, 4)} / L2 ${formatNumber(errL2 ?? NaN, 4)}（对${errorReference ?? "帧未说明的参考解"}）` +
    (nFloor > 0 ? `　${nFloor} 点贴地板 ${formatNumber(floor, 2)}` : "");
  head.push(
    hasFlip && phiFlip !== undefined
      ? `φ*=${formatNumber(phiFlip, 4)}：G<0 段每步翻符号（衰减≠不变号）　${errs}`
      : `无符号翻转波数（phi_flip=−1 哨兵：G 在 [0,π] 上不变号）　${errs}`,
  );
  // 图例放最后：它是唯一"砍掉也不损失数字"的一行。
  head.push(
    "图例：蓝=数值解、橙虚线=参考解；副图粗线=当前 |G|、细线=对照、红虚线=界线1、红点线=半径、蓝点线=φ*",
  );
  scene.texts.push(...textBlock(box.left, band.textTop, box.bottom, head, pal.muted, band.budget, "top"));

  // —— 角标 ——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  if (scene.dataSource.startsWith("backend+preview")) {
    scene.badges.push(placeBadge(previewBadge("拖动时按显示空间插值"), SLOT.preview, width));
  }
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(
      badge("autoscale", "解轴：固定 ±1.5（不随数据缩）；|G| 轴：固定 0–2.4；误差轴：对数、自适应", 8, 8),
      SLOT.auxAxis,
      width,
    ),
  );
  if (r !== undefined) {
    scene.badges.push(
      placeBadge(criticalBadge("r", r, criticalOf(frame, "r", R_CRITICAL), { beyond: "above" }), SLOT.critical, width),
    );
  }
  scene.badges.push(placeBadge(badge("status", stabilityStatus(measured, unstableModes, totalModes, initial), 8, 8), SLOT.status, width));
  const status = degradedBadge(degraded, SLOT.degraded, width);
  if (status !== undefined) scene.badges.push(status);

  requireDrawableLayers(scene);
  applyOverflow(scene, countNonFinite(u) + countNonFinite(uRef) + countNonFinite(errCurve), { slot: SLOT.overflow, width });

  // 文本层之间、以及与底部 pitfalls 面板之间都不许重叠（规则 8 / 规则 7）：
  // **第四个参数是高度**（`textBoxOf` 口径 `[x, y(顶), 宽, 高]`）。
  const panelTop = pitfallsBand(scene)?.[1];
  scene.texts = resolveTextOverlaps(scene.texts, [8, 24, width - 16, (panelTop ?? height) - 28]);

  return scene;
};

// --------------------------------------------------------------------------- //
// 局部工具
// --------------------------------------------------------------------------- //

/** 帧里可能缺的数组（缺了走降级路径，不抛错）。 */
function optional(frame: Frame, name: string): ArrayLike<number> | undefined {
  return frame.has(name) ? (frame.array(name) as ArrayLike<number>) : undefined;
}

/** 帧 `strings` 里的数字（`floor` 是字符串写的，按 PROTOCOLS §1.1 字符串不进 meta）。 */
function numberFromString(s: string | undefined): number | undefined {
  if (s === undefined) return undefined;
  const v = Number(s);
  return Number.isFinite(v) ? v : undefined;
}

/**
 * 状态角标：**判据只有 |G|**，与主图好不好看无关。
 *
 * 越界而初值是单模态时，这句话是本 viz 最要紧的一枚索引（pitfall 第 1 条）：
 * 主图会非常漂亮，而 |G|max 已经 > 1。
 */
function stabilityStatus(
  measured: number | undefined,
  unstableModes: number | undefined,
  totalModes: number | undefined,
  initial: string | undefined,
): string {
  if (measured === undefined) return "帧 meta 缺 measured_radius：不自行判断稳定性";
  const over = measured > 1 + 1e-12;
  const modes = unstableModes !== undefined && totalModes !== undefined ? `${unstableModes}/${totalModes} 模态被放大` : "被放大的模态数未标注";
  if (!over) return `稳定：实测谱半径 ${measured.toPrecision(6)} ≤ 1（${modes}）`;
  const single = initial === "sine" || initial === "cos" || initial === "cos_mean1" || initial === "cos_periodic_mean1";
  return single
    ? `⚠ 实测谱半径 ${measured.toPrecision(6)} > 1，但初值是单模态：主图可能看不出异常`
    : `⚠ 实测谱半径 ${measured.toPrecision(6)} > 1（${modes}）`;
}
