/**
 * `boundary_conservation`（4.1 章 viz 4）的 model：三种边界的解、总热量漂移与"结构性守恒"。
 *
 * 图层（顺序即绘制顺序）
 * ----------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | line | `solution` | 当前边界的数值解 `u` 与解析解 `u_exact`（虚线） |
 * | 2 | line | `mass` | 三种边界的 `ΔH(t)=H(t)−H(0)` **共用一根副轴** + 守恒基线 0 + 当前时刻竖线 |
 * | 3 | scatter | `mass` | `mass_residual` 的三个末刻漂移点（钉在当前时刻） |
 * | 4 | line | `error` | `|u−解析解|`（对数轴）+ 地板线（requiredAux: error） |
 * | 5 | line | `drift` | **\|ΔH\| 的对数轴**：结构性零（1e-14）与物理流失（0.24）相差 13 个数量级 |
 * | 6 | scatter | `drift` | `mass_residual` 的 \|值\|（requiredAux: residual）——三个边界都看得见 |
 *
 * 为什么要有 `drift` 这根额外的对数副轴：manifest 声明的 `mass` 轴窗口是**写死的**
 * −0.7…0.1，Neumann/周期的漂移（实测 3.2e-14）在那根轴上与 0 完全重合——"守恒"于是
 * 长得像"一条很小的曲线"，而这正是 pitfall 第 14 条要禁止的读法。对数副轴把
 * "舍入级"与"物理级"当场分开：它们不是"一个大一个小"，而是**两种不同的东西**。
 *
 * 时间轴（manifest `time.mode="scrub"`，tMax=0.05，真实时间）
 * ---------------------------------------------------------
 * 三条热量历史曲线都由后端**裁到当前时刻**（t=0 时退化为一个位于 0 的点）；
 * `meta.t_actual` 是实际到达时刻，与请求的 `meta.t` 可能不等，面板文字同时给出两者。
 *
 * 防误导七条里本章的重点
 * ----------------------
 * * **规则 7**：两条最容易被读反的都进面板文字：① 守恒是**结构性的**（离散算子行和恒为 0，
 *   与 θ、与 r 都无关），不是"数很小"；② 总热量不变 $\ne$ 解不变——Neumann 下常数模态
 *   永不衰减，数值解会冻结在一个与真解不同的常数上（主图那条固定的偏差带），
 *   `error` 副轴上的误差不随时间减小。
 * * **规则 1/4**：`solution`/`mass` 两根轴窗口写死（`autoscaled=false`），`drift`/`error`
 *   是自适应并按规则 4 标注——自适应轴只用在"量级本身就跨十几个数量级"的场合。
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
  absolute,
  arrayLabelOf,
  axisAutoscaledOf,
  axisTitleFitted,
  axisYlabelOf,
  bandLayout,
  endLabel,
  endLabelOf,
  contentBox,
  countZero,
  hLine,
  splitColumns,
  textBlock,
  vLine,
} from "./_panel.js";

const CHAPTER = "4.1";
const VIZ = "boundary_conservation";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "边界条件：总热量守恒 vs 边界流失";
/** manifest 的两根轴都写着"窗口固定"：解 −0.5…2.2、漂移 −0.7…0.1。 */
const SOLUTION_YLIM: [number, number] = [-0.5, 2.2];
const MASS_YLIM: [number, number] = [-0.7, 0.1];
const FLOOR_FALLBACK = 1e-17;

const SLOT = {
  source: 0,
  axisMode: 1,
  auxAxis: 2,
  critical: 3,
  overflow: 4,
  status: 5,
  preview: 6,
  degraded: 7,
} as const;

/** 三种边界的显示名（键就是后端 `set_strings` 用的枚举值）；面板文字按它排版。 */
const BC_LABELS: Record<string, string> = {
  dirichlet: "Dirichlet（端点钉住）",
  neumann: "Neumann（绝缘）",
  periodic: "周期",
};

export const boundaryConservationModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需数组 ——
  const x = requireArray(frame, "x");
  const u = requireArray(frame, "u");
  const uExact = requireArray(frame, "u_exact");
  const errCurve = requireArray(frame, "error_curve");
  const tHist = requireArray(frame, "t_history");
  const shifts = {
    dirichlet: requireArray(frame, "mass_shift_dirichlet"),
    neumann: requireArray(frame, "mass_shift_neumann"),
    periodic: requireArray(frame, "mass_shift_periodic"),
  };
  const massZero = requireArray(frame, "mass_zero");
  requireSameLength(x, u, "帧数组 x 与 u");
  requireSameLength(x, uExact, "帧数组 x 与 u_exact");
  requireSameLength(x, errCurve, "帧数组 x 与 error_curve");
  requireSameLength(tHist, massZero, "帧数组 t_history 与 mass_zero");
  const pitfalls = requirePitfalls(frame);

  // —— 可选数组 ——
  const degraded: string[] = [];
  const massResidual = optional(frame, "mass_residual");
  if (massResidual === undefined) degraded.push("帧缺数组 mass_residual：不画末刻漂移点（requiredAux: residual 只剩曲线）");
  const markT = optional(frame, "mark_t");
  if (markT === undefined) degraded.push("帧缺数组 mark_t：不画「当前时刻」竖线");

  // —— 标量 meta ——
  const tMeta = metaNumber(frame, "t");
  if (tMeta === undefined) degraded.push("帧 meta 缺 t：改用请求 t（两者可能不同）");
  const tNow = tMeta ?? opts.t;
  const tActual = metaNumber(frame, "t_actual");
  const dt = metaNumber(frame, "dt");
  const nSteps = metaNumber(frame, "n_steps");
  if (dt === undefined && tActual === undefined) {
    degraded.push("帧 meta 缺 dt 与 t_actual：不判断请求时刻是否到达、也不比对时刻（不假设相等）");
  } else if (dt === undefined) {
    degraded.push("帧 meta 缺 dt：不判断请求时刻是否到达");
  } else if (tActual === undefined) {
    degraded.push("帧 meta 缺 t_actual：不比对时刻（不假设相等）");
  }
  const r = metaNumber(frame, "r");
  const driftNow = metaNumber(frame, "drift_now");
  const driftDirichlet = metaNumber(frame, "drift_dirichlet_max");
  const driftNeumann = metaNumber(frame, "drift_neumann_max");
  const driftPeriodic = metaNumber(frame, "drift_periodic_max");
  const analyticLeak = metaNumber(frame, "analytic_leak");
  const leakNow = metaNumber(frame, "leak_now");
  const leakRatio = metaNumber(frame, "leak_ratio");
  const errLinf = metaNumber(frame, "err_linf");
  const errL2 = metaNumber(frame, "err_l2");
  const roundoff = metaNumber(frame, "roundoff");
  const mass0 = metaNumber(frame, "mass0");
  const massNow = metaNumber(frame, "mass_now");
  const conservationHolds = frame.meta["conservation_holds"] === true;

  const bc = stringOf(frame, "bc");
  const scheme = stringOf(frame, "scheme");
  const floor = numberFromString(stringOf(frame, "floor")) ?? FLOOR_FALLBACK;
  // 帧的 initial_note / drift_note / leak_note / reading_note 是完整句子（123/129/84/164 个单位）：
  // 面板通栏 6 行放不下，它们的内容按"数字 + 最短判读"重述在下面（三种相容初值、曲线裁到
  // 当前时刻、流失量的解析式都保留），完整口径留在帧里。
  const grad = paramSummary(frame, opts, [
    { name: "r", label: "$r$" },
    { name: "nx", label: "$nx$" },
  ]);
  degraded.push(...grad.degraded);

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: tNow,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  // —— 排版：上排「解 | 总热量漂移」，下排「误差 | |ΔH| 对数轴」；底部通栏数值块 ——
  // 上排**清空 xticks**（x 轴与下排共用），所以两排之间只要让开下排的轴标题带（`top−18`）。
  const box = contentBox(width, height, pitfallsBand(scene)?.[1] ?? null);
  const cols = splitColumns(box, 0.62);
  const band = bandLayout(box, { botH: Math.max(70, Math.round(box.height * 0.22)), rows: 6 });
  const solutionRect: [number, number, number, number] = [cols.left.left, band.topY, cols.left.width, band.topH];
  const massRect: [number, number, number, number] = [cols.right.left, band.topY, cols.right.width, band.topH];
  const errorRect: [number, number, number, number] = [cols.left.left, band.botY, cols.left.width, band.botH];
  const driftRect: [number, number, number, number] = [cols.right.left, band.botY, cols.right.width, band.botH];

  // —— 轴 ——
  const xRange = finiteMinMax(x) ?? [0, 1];
  const xlim = niceLimits(xRange[0], xRange[1], 6);
  // 热量历史的横轴用 manifest 的 tMax（历史被后端裁到当前时刻，用数据定范围会让轴随拖动伸缩）。
  const tMax = SPEC?.time.tMax ?? Math.max(0, lastOf(tHist));
  const tXlim: [number, number] = [0, tMax > 0 ? tMax : 1];

  const solutionAxis = makeAxis("solution", solutionRect, {
    xlim,
    ylim: SOLUTION_YLIM,
    ylabel: axisYlabelOf(SPEC, "solution", "u(x,t)"),
    title: axisTitleFitted(SPEC, "solution", "解 u(x,t)（窗口固定 −0.5–2.2）", solutionRect[2]),
    autoscaled: axisAutoscaledOf(SPEC, "solution", false),
    xticks: [],
  });
  const massAxis = makeAxis("mass", massRect, {
    xlim: tXlim,
    ylim: MASS_YLIM,
    ylabel: axisYlabelOf(SPEC, "mass", "ΔH"),
    title: axisTitleFitted(SPEC, "mass", "总热量漂移 ΔH（窗口固定 −0.7–0.1）", massRect[2]),
    autoscaled: axisAutoscaledOf(SPEC, "mass", false),
    xticks: [],
  });

  const errPos = finitePositiveMinMax(errCurve);
  const errYlim: [number, number] =
    errPos === undefined ? [floor, 1] : niceLimits(Math.min(errPos[0], floor), errPos[1], 6, { log: true });
  const errorAxis = makeAxis("error", errorRect, {
    xlim,
    ylim: errYlim,
    yscale: "log",
    ylabel: "$|u$−解析解|",
    title: "误差（对数轴）",
    autoscaled: true,
    xticks: niceTicks(xlim[0], xlim[1], 6),
  });

  // |ΔH| 对数轴：范围只由三条历史曲线定（1e-17 的地板不进范围——它在这根轴上不画线，
  // 硬塞进来会把 13 个数量级压成中间一条细缝）。
  const driftPos = finitePositiveMinMax(concatAbs(shifts.dirichlet, shifts.neumann, shifts.periodic));
  const driftYlim: [number, number] =
    driftPos === undefined ? [floor, 1] : niceLimits(driftPos[0] / 2, driftPos[1] * 2, 5, { log: true });
  const driftAxis = makeAxis("residual", driftRect, {
    xlim: tXlim,
    ylim: driftYlim,
    yscale: "log",
    ylabel: "$|ΔH|$",
    title: "|ΔH| 对数轴（结构性零 vs 物理流失）",
    autoscaled: true,
    xticks: niceTicks(tXlim[0], tXlim[1], 5),
  });
  scene.axes.push(solutionAxis, massAxis, errorAxis, driftAxis);

  // —— 图层 1：解与解析解 ——
  const solSeries = [
    ...segmentedSeries(x, u, pal.curves[0], {
      width: 1.8,
      label: endLabelOf(x, xlim, solutionRect[2], arrayLabelOf(frame, "u", `数值解（${bc ?? "?"}）`)),
    }),
    ...segmentedSeries(x, uExact, pal.exact, {
      width: 1.3,
      dash: [6, 3],
      label: endLabelOf(x, xlim, solutionRect[2], arrayLabelOf(frame, "u_exact", "解析解")),
    }),
  ];
  if (solSeries.length > 0) scene.layers.push(lineLayer("solution", solSeries));
  else degraded.push("帧数组 u/u_exact 没有可绘的有限值：主面板不画曲线");

  // —— 图层 2：三种边界的 ΔH 共用一根副轴 + 守恒基线 + 当前时刻 ——
  const massSeries = [
    ...segmentedSeries(tHist, massZero, pal.reference, {
      width: 1,
      dash: [3, 3],
      label: endLabelOf(tHist, tXlim, massRect[2], "守恒基线 0"),
    }),
    ...segmentedSeries(tHist, shifts.neumann, pal.curves[1], {
      width: 1.6,
      label: endLabelOf(tHist, tXlim, massRect[2], arrayLabelOf(frame, "mass_shift_neumann", "ΔH（Neumann：绝缘）")),
    }),
    ...segmentedSeries(tHist, shifts.periodic, pal.curves[2], {
      width: 1.6,
      label: endLabelOf(tHist, tXlim, massRect[2], arrayLabelOf(frame, "mass_shift_periodic", "ΔH（周期）")),
    }),
    ...segmentedSeries(tHist, shifts.dirichlet, pal.curves[0], {
      width: 1.8,
      label: endLabelOf(tHist, tXlim, massRect[2], arrayLabelOf(frame, "mass_shift_dirichlet", "ΔH（Dirichlet：边界流失）")),
    }),
  ];
  if (markT !== undefined && markT.length > 0) {
    const tm = markT[0] as number;
    if (Number.isFinite(tm)) {
      const line = vLine(tm, MASS_YLIM[0], MASS_YLIM[1]);
      massSeries.push(
        series(line.x, line.y, pal.cursor, {
          width: 1,
          dash: [2, 4],
          label: endLabel(tm, tXlim, massRect[2], `当前 t=${formatNumber(tm, 4)}`),
        }),
      );
    }
  }
  scene.layers.push(lineLayer("mass", massSeries));

  // —— 图层 3：mass_residual 的三个末刻漂移点（requiredAux: residual，线性轴上只有 Dirichlet 看得见）——
  if (massResidual !== undefined && markT !== undefined && massResidual.length > 0 && markT.length > 0) {
    const tm = markT[0] as number;
    const xs: number[] = [];
    const ys: number[] = [];
    for (let i = 0; i < massResidual.length; i += 1) {
      xs.push(tm);
      ys.push(massResidual[i] as number);
    }
    if (Number.isFinite(tm) && ys.every((v) => Number.isFinite(v))) {
      scene.layers.push(
        scatterLayer("mass", xs, ys, {
          r: 4,
          color: pal.marker,
          label: endLabelOf(xs, tXlim, massRect[2], arrayLabelOf(frame, "mass_residual", "末刻漂移（Neumann / 周期 / 当前边界）")),
        }),
      );
    }
  }

  // —— 图层 4：误差曲线（requiredAux: error）+ 地板线 ——
  const errSeries = segmentedSeries(x, errCurve, pal.residual, {
    width: 1.5,
    accept: (v) => v > 0,
    label: endLabelOf(x, xlim, errorRect[2], arrayLabelOf(frame, "error_curve", "|u−解析解|（对数轴）")),
  });
  if (errSeries.length > 0) scene.layers.push(lineLayer("error", errSeries));
  else degraded.push("帧 error_curve 没有 > 0 的有限值：对数轴画不出误差曲线");
  const floorLine = hLine(xlim[0], xlim[1], floor);
  scene.layers.push(
    lineLayer("error", [
      series(floorLine.x, floorLine.y, pal.reference, {
        width: 1,
        dash: [2, 3],
        label: endLabel(floorLine.x[1], xlim, errorRect[2], `地板 ${formatNumber(floor, 2)}`),
      }),
    ]),
  );

  // —— 图层 5：|ΔH| 对数轴（结构性零 vs 物理流失）——
  const driftSeries = [
    ...segmentedSeries(tHist, absolute(shifts.neumann), pal.curves[1], {
      width: 1.5,
      accept: (v) => v > 0,
      label: endLabelOf(tHist, tXlim, driftRect[2], "|ΔH|（Neumann）"),
    }),
    ...segmentedSeries(tHist, absolute(shifts.periodic), pal.curves[2], {
      width: 1.5,
      accept: (v) => v > 0,
      label: endLabelOf(tHist, tXlim, driftRect[2], "|ΔH|（周期）"),
    }),
    ...segmentedSeries(tHist, absolute(shifts.dirichlet), pal.curves[0], {
      width: 1.8,
      accept: (v) => v > 0,
      label: endLabelOf(tHist, tXlim, driftRect[2], "|ΔH|（Dirichlet）"),
    }),
  ];
  if (roundoff !== undefined && roundoff > 0 && roundoff >= driftAxis.ylim[0]) {
    const roLine = hLine(tXlim[0], tXlim[1], roundoff);
    driftSeries.push(
      series(roLine.x, roLine.y, pal.reference, {
        width: 1,
        dash: [4, 3],
        label: endLabel(roLine.x[1], tXlim, driftRect[2], `单位舍入 ${formatNumber(roundoff, 2)}（= eps/2）`),
      }),
    );
  }
  scene.layers.push(lineLayer("residual", driftSeries));

  // —— 图层 6：mass_residual 的 |值| 放到对数轴上（三个边界都看得见）——
  if (massResidual !== undefined && markT !== undefined && massResidual.length > 0 && markT.length > 0) {
    const tm = markT[0] as number;
    const xs: number[] = [];
    const ys: number[] = [];
    let nZero = 0;
    for (let i = 0; i < massResidual.length; i += 1) {
      const v = Math.abs(massResidual[i] as number);
      if (!Number.isFinite(v)) continue;
      if (v === 0) {
        nZero += 1;
        continue;
      }
      xs.push(tm);
      ys.push(v);
    }
    if (xs.length > 0 && Number.isFinite(tm)) {
      scene.layers.push(
        scatterLayer("residual", xs, ys, {
          r: 4,
          color: pal.marker,
          label: endLabelOf(xs, tXlim, driftRect[2], `末刻 |ΔH|（${xs.length} 个边界${nZero > 0 ? `；${nZero} 个精确为 0，对数轴画不出` : ""}）`),
        }),
      );
    }
  }

  // —— 文本：底部通栏（数字在前、判读在后、图例最后）——
  // 帧的 initial_note / drift_note / leak_note / reading_note 是完整句子，归底部 pitfalls 面板。
  const zeroPoints = countZero(shifts.dirichlet) + countZero(shifts.neumann) + countZero(shifts.periodic);
  const head: string[] = [
    `参数：边界=${bc ? `${bc}${BC_LABELS[bc] !== undefined ? `（${BC_LABELS[bc]}）` : ""}` : "?"}　格式=${scheme ?? "?"}　${grad.text}` +
      `　dt=${formatNumber(dt ?? NaN, 4)}　n_steps=${formatNumber(nSteps ?? 0, 6)}　请求 t=${formatNumber(tNow, 4)} / 到达 ${formatNumber(tActual ?? NaN, 4)}`,
    `末刻总热量 H(0)=${formatNumber(mass0 ?? NaN, 6)} → H(t)=${formatNumber(massNow ?? NaN, 6)}　漂移 ΔH=${formatNumber(driftNow ?? NaN, 4)}` +
      `　解析流失量 (2/π)(1−e^(−π²t))=${formatNumber(analyticLeak ?? NaN, 4)}（本步实测 ${formatNumber(leakNow ?? NaN, 4)}` +
      `${leakRatio !== undefined && leakRatio >= 0 ? `，实测/解析 ${formatNumber(leakRatio, 4)}` : ""}）`,
    `历史最大 |ΔH|：Dirichlet ${formatNumber(driftDirichlet ?? NaN, 3)}　Neumann ${formatNumber(driftNeumann ?? NaN, 3)}` +
      `　周期 ${formatNumber(driftPeriodic ?? NaN, 3)}（舍入级 = eps/2 × 步数）` +
      (errLinf !== undefined ? `　L∞ 误差 ${formatNumber(errLinf, 4)} / L2 ${formatNumber(errL2 ?? NaN, 4)}` : "") +
      (zeroPoints > 0 ? `　${zeroPoints} 点 = 0 已跳过` : ""),
  ];
  head.push(
    conservationHolds
      ? "⚠ 本帧边界守恒：漂移量级 = 舍入，与 r、与格式无关——这是「1ᵀD̃ = 0」的结构性事实，不是「数很小」。"
      : "本帧的边界不守恒：端点行不是零行，热量按边界通量流失。别把守恒判据读成「ΔH 小」——上面一条是舍入（1e-14），这一条是物理（1e-1）。",
  );
  head.push("⚠ 总热量不变 ≠ 解不变：Neumann 下常数模态永不衰减，解冻结在与真解不同的常数上，误差不随时间减小。");
  head.push(
    "图例：蓝=数值解、橙虚线=解析解；漂移轴蓝=Dirichlet、绿=Neumann、紫=周期、灰虚线=基线 0；|ΔH| 对数轴同色。",
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
      badge("autoscale", "解轴/漂移轴：固定窗口；误差轴与 |ΔH| 对数轴：自适应（会掩盖量级变化）", 8, 8),
      SLOT.auxAxis,
      width,
    ),
  );
  if (r !== undefined) {
    scene.badges.push(
      placeBadge(criticalBadge("r", r, criticalOf(frame, "r", 0.5), { beyond: "above" }), SLOT.critical, width),
    );
  }
  scene.badges.push(
    placeBadge(
      badge(
        "status",
        conservationHolds
          ? `守恒：末刻漂移 ${formatNumber(driftNow ?? 0, 3)}（舍入级，结构性；与 r/格式无关）`
          : `不守恒：末刻流失 ${formatNumber(-(driftNow ?? 0), 4)}（端点行不是零行）`,
        8,
        8,
      ),
      SLOT.status,
      width,
    ),
  );
  const status = degradedBadge(degraded, SLOT.degraded, width);
  if (status !== undefined) scene.badges.push(status);

  requireDrawableLayers(scene);
  applyOverflow(scene, countNonFinite(u) + countNonFinite(uExact) + countNonFinite(errCurve), {
    slot: SLOT.overflow,
    width,
  });

  const panelTop = pitfallsBand(scene)?.[1];
  scene.texts = resolveTextOverlaps(scene.texts, [8, 24, width - 16, (panelTop ?? height) - 28]);

  return scene;
};

// --------------------------------------------------------------------------- //
// 局部工具
// --------------------------------------------------------------------------- //

function optional(frame: Frame, name: string): ArrayLike<number> | undefined {
  return frame.has(name) ? (frame.array(name) as ArrayLike<number>) : undefined;
}

function numberFromString(s: string | undefined): number | undefined {
  if (s === undefined) return undefined;
  const v = Number(s);
  return Number.isFinite(v) ? v : undefined;
}

function lastOf(a: ArrayLike<number>): number {
  const v = a.length > 0 ? (a[a.length - 1] as number) : 0;
  return Number.isFinite(v) ? v : 0;
}

/** 若干数组（可能缺）里 > 0 的有限值并起来——只用于取对数轴的显示范围。 */
function concatAbs(...arrays: (ArrayLike<number> | undefined)[]): Float64Array {
  const out: number[] = [];
  for (const a of arrays) {
    if (a === undefined) continue;
    for (let i = 0; i < a.length; i += 1) {
      const v = Math.abs(a[i] as number);
      if (Number.isFinite(v) && v > 0) out.push(v);
    }
  }
  return Float64Array.from(out);
}
