/**
 * `sobol_bars`（5.4 章 viz 3）：一阶与总效应 Sobol 指数。
 *
 * 三条纪律
 * -------
 * 1. **三个都叫“一阶”的数**：独立输入的 pick-freeze `first`、相关样本下的条件方差
 *    `between_cond`、把相关样本打散后的 `first_indep`。三者相等当且仅当输入独立；
 *    相关时前两者之和可以大于 1（全方差公式只对单个变量成立）。
 * 2. **bootstrap 区间是估计量不确定性**，不是“真值一定在里面”。
 * 3. **估计器有偏**：副轴 sum S_i(N) 随 N 漂移 —— 这是偏差，不是运气。
 *
 * 帧 `time.mode="none"`。
 */

import { formatNumber } from "../../shared/src/axes.js";
import {
  applyOverflow,
  canvasSize,
  dataSourceOf,
  degradedBadge,
  metaNumber,
  palette,
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
  barsLayer,
  lineLayer,
  makeAxis,
  scatterLayer,
  sourceBadge,
} from "../../shared/src/scene.js";
import { TEXT_ROW_H, axisTitleOf, axisYlabelOf, columnsLayout, textBlock, unitBudget } from "../_common/panel.js";
import { CHAPTER, SLOT } from "./common.js";
import { manifest } from "./manifest.gen.js";

const VIZ = "sobol_bars";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "Sobol 指数：一阶 vs 总效应";

export const sobolBarsModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  const indexX = requireArray(frame, "index_x");
  const first = requireArray(frame, "first");
  const total = requireArray(frame, "total");
  const errFirst = requireArray(frame, "error_first");
  const errTotal = requireArray(frame, "error_total");
  const otherFirst = frame.has("first_other") ? requireArray(frame, "first_other") : undefined;
  const otherTotal = frame.has("total_other") ? requireArray(frame, "total_other") : undefined;
  const firstIndep = frame.has("first_indep") ? requireArray(frame, "first_indep") : undefined;
  const betweenCond = frame.has("between_cond") ? requireArray(frame, "between_cond") : undefined;
  const corrValid = frame.has("corr_valid") ? requireArray(frame, "corr_valid") : undefined;
  for (const [name, arr] of [
    ["first", first],
    ["total", total],
    ["error_first", errFirst],
    ["error_total", errTotal],
  ] as const) {
    requireSameLength(indexX, arr, `index_x 与 ${name}`);
  }
  for (const [name, arr] of [
    ["first_other", otherFirst],
    ["total_other", otherTotal],
    ["first_indep", firstIndep],
    ["between_cond", betweenCond],
    ["corr_valid", corrValid],
  ] as const) {
    if (arr !== undefined) requireSameLength(indexX, arr, `index_x 与 ${name}`);
  }
  const driftN = requireArray(frame, "drift_n");
  const driftFirst = requireArray(frame, "drift_sum_first");
  const driftTotal = requireArray(frame, "drift_sum_total");
  const driftAnaFirst = requireArray(frame, "drift_analytic_first");
  const driftAnaTotal = requireArray(frame, "drift_analytic_total");
  for (const [name, arr] of [
    ["drift_sum_first", driftFirst],
    ["drift_sum_total", driftTotal],
    ["drift_analytic_first", driftAnaFirst],
    ["drift_analytic_total", driftAnaTotal],
  ] as const) {
    requireSameLength(driftN, arr, `drift_n 与 ${name}`);
  }
  const pitfalls = requirePitfalls(frame);

  const degraded: string[] = [];
  if (corrValid === undefined) degraded.push("帧缺 corr_valid：不画相关输入的替代分解");
  const dim = metaNumber(frame, "dim") ?? indexX.length;
  const n = metaNumber(frame, "n");
  const rho = metaNumber(frame, "rho");
  const nEvals = metaNumber(frame, "n_evals");
  const sumFirst = metaNumber(frame, "sum_first");
  const sumTotal = metaNumber(frame, "sum_total");
  const sumFirstHw = metaNumber(frame, "sum_first_halfwidth");
  const sumTotalHw = metaNumber(frame, "sum_total_halfwidth");
  const interaction = metaNumber(frame, "interaction_share");
  const lowerOk = frame.meta["lower_bound_ok"] === true;
  const upperOk = frame.meta["upper_bound_ok"] === true;
  const anaFirst = metaNumber(frame, "analytic_sum_first");
  const anaTotal = metaNumber(frame, "analytic_sum_total");
  const driftFirstSpan = metaNumber(frame, "drift_span_first");
  const driftTotalSpan = metaNumber(frame, "drift_span_total");
  const corrApplicable = frame.meta["corr_applicable"] === true;
  const corrActive = frame.meta["corr_active"] === true;
  const overlapExcess = metaNumber(frame, "overlap_excess");
  const sumBetween = metaNumber(frame, "sum_between");
  const sumIndep = metaNumber(frame, "sum_indep");
  const modelLabel = stringOf(frame, "model_label") ?? stringOf(frame, "model") ?? "?";
  const estimator = stringOf(frame, "estimator") ?? "?";

  // 参数 i 从 1 开始；一组柱两根，柱宽 0.32。
  const xs: number[] = [];
  for (let i = 0; i < indexX.length; i += 1) xs.push(i + 1);
  let yHi = 1.05;
  let yLo = 0;
  for (let i = 0; i < first.length; i += 1) {
    yHi = Math.max(yHi, (first[i] as number) + (errFirst[i] as number), (total[i] as number) + (errTotal[i] as number));
    yLo = Math.min(yLo, (first[i] as number) - (errFirst[i] as number), (total[i] as number) - (errTotal[i] as number));
  }
  if (otherFirst !== undefined) for (let i = 0; i < otherFirst.length; i += 1) yHi = Math.max(yHi, otherFirst[i] as number);
  if (otherTotal !== undefined) for (let i = 0; i < otherTotal.length; i += 1) yHi = Math.max(yHi, otherTotal[i] as number);
  if (betweenCond !== undefined) for (let i = 0; i < betweenCond.length; i += 1) yHi = Math.max(yHi, betweenCond[i] as number);

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: 0,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  const L = columnsLayout(scene, width, height, { topShare: 0.58, auxH: 150, gap: 48 });
  const main = makeAxis("main", [L.left, L.topY, L.leftW, L.topH], {
    xlim: [0.5, dim + 0.5],
    ylim: [yLo < 0 ? yLo * 1.1 : 0, yHi * 1.08],
    xlabel: "参数 i",
    ylabel: axisYlabelOf(SPEC?.axes, "main", "S_i / S_Ti"),
    title: axisTitleOf(SPEC?.axes, "main", "Sobol 指数（柱=一阶/总效应，须线=bootstrap）"),
    autoscaled: true,
  });
  let dLo = Number.POSITIVE_INFINITY;
  let dHi = Number.NEGATIVE_INFINITY;
  for (const arr of [driftFirst, driftTotal, driftAnaFirst, driftAnaTotal]) {
    for (let i = 0; i < arr.length; i += 1) {
      dLo = Math.min(dLo, arr[i] as number);
      dHi = Math.max(dHi, arr[i] as number);
    }
  }
  if (!Number.isFinite(dLo)) {
    dLo = 0;
    dHi = 1;
  }
  const aux = makeAxis("aux", [L.left, L.botY, L.leftW, L.botH], {
    xlim: [driftN[0] as number, driftN[driftN.length - 1] as number],
    ylim: [dLo - 0.1, dHi + 0.1],
    xscale: "log",
    xlabel: "N（对数）",
    ylabel: axisYlabelOf(SPEC?.axes, "aux", "ΣS_i(N)"),
    title: axisTitleOf(SPEC?.axes, "aux", "估计器漂移：ΣS_i(N) 与 ΣS_Ti(N)"),
    autoscaled: true,
  });
  scene.axes.push(main, aux);

  const firstX = xs.map((x) => x - 0.19);
  const totalX = xs.map((x) => x + 0.19);
  scene.layers.push(
    barsLayer("main", firstX, first, { width: 0.3, color: pal.curves[0] as string, errors: errFirst, label: "一阶 S_i" }),
    barsLayer("main", totalX, total, { width: 0.3, color: pal.exact, errors: errTotal, label: "总效应 S_Ti" }),
  );
  // 相关输入下的替代分解：只画点，不塞进柱宽；corr_valid=0 时整块不画。
  if (corrValid !== undefined && betweenCond !== undefined && firstIndep !== undefined) {
    const bx: number[] = [];
    const by: number[] = [];
    const ix: number[] = [];
    const iy: number[] = [];
    for (let i = 0; i < corrValid.length; i += 1) {
      if ((corrValid[i] as number) <= 0.5) continue;
      bx.push(xs[i] as number);
      by.push(betweenCond[i] as number);
      ix.push(xs[i] as number);
      iy.push(firstIndep[i] as number);
    }
    if (bx.length > 0) {
      scene.layers.push(
        scatterLayer("main", bx, by, { r: 3.6, color: pal.residual, label: "条件方差口径 Var(E[Y|Xi])/V" }),
        scatterLayer("main", ix, iy, { r: 3.2, color: pal.curves[1] as string, label: "打散相关后的 pick-freeze" }),
      );
    }
  }
  scene.layers.push(
    lineLayer("main", { x: [0.5, dim + 0.5], y: [1, 1], color: pal.warn, width: 1, dash: [4, 3], label: "上界 1（独立输入）" }),
  );

  scene.layers.push(
    lineLayer("aux", { x: driftN, y: driftFirst, color: pal.curves[0] as string, width: 2, label: "ΣS_i(N)" }),
    lineLayer("aux", { x: driftN, y: driftTotal, color: pal.exact, width: 2, label: "ΣS_Ti(N)" }),
    lineLayer("aux", { x: driftN, y: driftAnaFirst, color: pal.curves[0] as string, width: 1.1, dash: [5, 3], label: "解析 ΣS_i" }),
    lineLayer("aux", { x: driftN, y: driftAnaTotal, color: pal.exact, width: 1.1, dash: [5, 3], label: "解析 ΣS_Ti" }),
  );

  const budget = unitBudget(L.rightW);
  if (L.rightW >= 200) {
    let y = L.topY;
    const push = (lines: readonly string[], color: string): void => {
      if (y + TEXT_ROW_H > L.floor) return;
      const block = textBlock(L.rightX, y, L.floor, lines, color, budget, "top");
      scene.texts.push(...block);
      y += block.length * TEXT_ROW_H;
    };
    push(
      [
        `模型：${modelLabel}`,
        `估计器：${estimator}；d=${formatNumber(dim, 0)}，N=${formatNumber(n ?? Number.NaN, 0)}。`,
        `模型求值 ${formatNumber(nEvals ?? Number.NaN, 0)} 次；ρ=${formatNumber(rho ?? Number.NaN, 2)}。`,
        `ΣS_i=${formatNumber(sumFirst ?? Number.NaN, 3)} ± ${formatNumber(sumFirstHw ?? Number.NaN, 3)}；ΣS_Ti=${formatNumber(sumTotal ?? Number.NaN, 3)} ± ${formatNumber(sumTotalHw ?? Number.NaN, 3)}。`,
      ],
      pal.ink,
    );
    push(
      [
        `解析 ΣS_i=${formatNumber(anaFirst ?? Number.NaN, 3)}，ΣS_Ti=${formatNumber(anaTotal ?? Number.NaN, 3)}。`,
        `交互份额 ${formatNumber(interaction ?? Number.NaN, 3)}。`,
        `漂移跨度 ΣS_i ${formatNumber(driftFirstSpan ?? Number.NaN, 3)}、ΣS_Ti ${formatNumber(driftTotalSpan ?? Number.NaN, 3)}。`,
        "漂移是偏差，不是运气。",
        lowerOk && upperOk ? "边界检查：S_i≤1 且 S_Ti≥S_i 成立。" : "边界检查失败：先查估计器/样本。",
      ],
      lowerOk && upperOk ? pal.muted : pal.warn,
    );
    push(
      [
        corrActive
          ? `相关输入：条件 Σ=${formatNumber(sumBetween ?? Number.NaN, 3)}，打散后 Σ=${formatNumber(sumIndep ?? Number.NaN, 3)}。`
          : corrApplicable
            ? "该模型支持相关输入，但当前 ρ=0。"
            : "该模型是独立输入：ρ 无效（不是 bug）。",
        corrActive ? `重叠超出 ${formatNumber(overlapExcess ?? Number.NaN, 3)}。` : "只有一套分解。",
        "三个“一阶”：first（独立）、",
        "between_cond（相关条件）、",
        "first_indep（打散后）。",
        "bootstrap 须线是估计量不确定性；",
        "它不保证真值一定落在区间内。",
      ],
      pal.muted,
    );
  } else {
    degraded.push("右侧说明列过窄：读数并入角标");
  }

  scene.badges.push(placeBadge(sourceBadge(dataSourceOf(frame)), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(true), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(
      badge(
        "status",
        `ΣS_i=${formatNumber(sumFirst ?? Number.NaN, 3)}｜ΣS_T=${formatNumber(sumTotal ?? Number.NaN, 3)}｜交互 ${formatNumber(interaction ?? Number.NaN, 3)}｜ρ=${formatNumber(rho ?? Number.NaN, 2)}`,
        8,
        8,
      ),
      SLOT.status,
      width,
    ),
  );
  if (!lowerOk || !upperOk) {
    scene.badges.push(placeBadge(badge("critical", "上界违约：S_i>1 或 S_Ti<S_i，估计器/样本有问题", 8, 8), SLOT.critical, width));
  }
  const dbg = degradedBadge(degraded, SLOT.degraded, width);
  if (dbg !== undefined) scene.badges.push(dbg);

  applyOverflow(scene, 0, { slot: SLOT.overflow, width });
  requireDrawableLayers(scene);
  return scene;
};
