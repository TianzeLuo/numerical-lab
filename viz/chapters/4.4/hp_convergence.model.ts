/**
 * `hp_convergence`（4.4 章 viz 3）的 model：h-加密 vs p-加密。
 *
 * 面板与图层
 * ----------
 * | 面板 | 轴 | 内容 |
 * |---|---|---|
 * | 左上 | `ladder` | h 阶梯 `error`、p 阶梯 `error_p`（两轴都按自由度）＋ `fit_h` ＋ 等自由度标记 |
 * | 左下 | `vs_p` | `error_vs_p`（requiredAux `error_p`）＋ 半对数/双对数两条拟合线 |
 * | 右上 | `condition` | `condition_p` vs p（`condition_unmeasured=1` 的点不进图层） |
 *
 * 三条防误导纪律
 * --------------
 * 1. **两个横轴**：同一份 p 数据在 `ladder` 上按自由度（与 h 可比）、在 `vs_p` 上按 p
 *    （才看得出指数/代数）。只画其中一张都会把结论说反。
 * 2. **地板点不画成数据**：`error_at_floor / error_p_at_floor` 为 1 的点被切开、不连线；
 *    地板线单独画并在图例里报数。
 * 3. **条件数是代价不是精度**：`condition_p=1e300` 是「未测量」的显示上限，
 *    `condition_unmeasured=1` 的点被排除；p=1 的未测点在文字里点名。
 *
 * 时间轴：manifest `mode="none"`；`scene.t` 恒为 0，不写 tMax。
 */

import { formatNumber } from "../../shared/src/axes.js";
import { countNonFinite } from "../../shared/src/geometry.js";
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
  lineLayer,
  makeAxis,
  scatterLayer,
  series,
  sourceBadge,
  type LineSeries,
} from "../../shared/src/scene.js";
import {
  SAFE_GAP,
  TEXT_ROW_H,
  axisTitleOf,
  axisYlabelOf,
  columnsLayout,
  logRange,
  textBlock,
  unitBudget,
} from "../_common/panel.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "4.4";
const VIZ = "hp_convergence";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "h-加密与 p-加密：阶是（方法，解的正则性）的函数";

const FLOOR_FALLBACK = 1e-17;

const SLOT = {
  source: 0,
  axisMode: 1,
  aux: 2,
  critical: 3,
  status: 4,
  preview: 5,
  overflow: 6,
  degraded: 7,
} as const;

function maskedSeries(
  x: ArrayLike<number>,
  y: ArrayLike<number>,
  mask: ArrayLike<number> | undefined,
  color: string,
  style: Partial<LineSeries>,
  accept: (v: number) => boolean = (v) => v > 0,
): LineSeries[] {
  const n = Math.min(x.length, y.length);
  const out: LineSeries[] = [];
  let start = -1;
  const usable = (i: number): boolean => {
    const v = y[i] as number;
    return Number.isFinite(v) && accept(v) && (mask === undefined || (mask[i] ?? 0) <= 0.5);
  };
  const push = (a: number, b: number): void => {
    if (b - a < 1) return;
    const xs = new Float64Array(b - a);
    const ys = new Float64Array(b - a);
    for (let i = a; i < b; i += 1) {
      xs[i - a] = x[i] as number;
      ys[i - a] = y[i] as number;
    }
    out.push(series(xs, ys, color, out.length === 0 ? style : { ...style, label: undefined }));
  };
  for (let i = 0; i < n; i += 1) {
    if (usable(i)) {
      if (start < 0) start = i;
    } else if (start >= 0) {
      push(start, i);
      start = -1;
    }
  }
  if (start >= 0) push(start, n);
  return out;
}

function countMask(mask: ArrayLike<number> | undefined): number {
  if (mask === undefined) return 0;
  let n = 0;
  for (let i = 0; i < mask.length; i += 1) if ((mask[i] ?? 0) > 0.5) n += 1;
  return n;
}

function maxFinite(a: ArrayLike<number>): number | undefined {
  let hi = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i] as number;
    if (Number.isFinite(v)) hi = Math.max(hi, v);
  }
  return Number.isFinite(hi) ? hi : undefined;
}

function firstFinite(a: ArrayLike<number> | undefined): number | undefined {
  if (a === undefined) return undefined;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i] as number;
    if (Number.isFinite(v)) return v;
  }
  return undefined;
}

function fmt(v: number | undefined, digits = 3): string {
  return v === undefined || !Number.isFinite(v) ? "?" : formatNumber(v, digits);
}

export const hpConvergenceModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需数组 ——
  const dofH = requireArray(frame, "dof_h");
  const errH = requireArray(frame, "error");
  const dofP = requireArray(frame, "dof_p");
  const errP = requireArray(frame, "error_p");
  const pAxis = requireArray(frame, "p_axis");
  const errVsP = requireArray(frame, "error_vs_p");
  const cond = requireArray(frame, "condition_p");
  const condMask = requireArray(frame, "condition_unmeasured");
  requireSameLength(dofH, errH, "帧数组 dof_h 与 error");
  requireSameLength(dofP, errP, "帧数组 dof_p 与 error_p");
  requireSameLength(dofP, pAxis, "帧数组 dof_p 与 p_axis");
  requireSameLength(dofP, errVsP, "帧数组 dof_p 与 error_vs_p");
  requireSameLength(dofP, cond, "帧数组 dof_p 与 condition_p");
  requireSameLength(dofP, condMask, "帧数组 dof_p 与 condition_unmeasured");
  const pitfalls = requirePitfalls(frame);

  // —— 可选数组 ——
  const degraded: string[] = [];
  const floorMaskH = frame.has("error_at_floor") ? requireArray(frame, "error_at_floor") : undefined;
  if (floorMaskH === undefined) degraded.push("帧缺 error_at_floor：h 阶梯的地板点不排除（不标已到地板）");
  else requireSameLength(dofH, floorMaskH, "帧数组 dof_h 与 error_at_floor");
  const floorMaskP = frame.has("error_p_at_floor") ? requireArray(frame, "error_p_at_floor") : undefined;
  if (floorMaskP === undefined) degraded.push("帧缺 error_p_at_floor：p 阶梯的地板点不排除");
  else requireSameLength(dofP, floorMaskP, "帧数组 dof_p 与 error_p_at_floor");

  const fitH = frame.has("fit_h") ? requireArray(frame, "fit_h") : undefined;
  if (fitH === undefined) degraded.push("帧缺 fit_h：不画 h 阶梯的拟合线");
  else if (fitH.length !== dofH.length) degraded.push("fit_h 长度与 dof_h 不符：不画该拟合线");
  const fitExp = frame.has("fit_p_exp") ? requireArray(frame, "fit_p_exp") : undefined;
  if (fitExp === undefined) degraded.push("帧缺 fit_p_exp：不画半对数拟合线");
  else if (fitExp.length !== pAxis.length) degraded.push("fit_p_exp 长度与 p_axis 不符：不画该拟合线");
  const fitAlg = frame.has("fit_p_alg") ? requireArray(frame, "fit_p_alg") : undefined;
  if (fitAlg === undefined) degraded.push("帧缺 fit_p_alg：不画双对数拟合线");
  else if (fitAlg.length !== pAxis.length) degraded.push("fit_p_alg 长度与 p_axis 不符：不画该拟合线");

  const markEqualP = frame.has("mark_equal_dof_p") ? requireArray(frame, "mark_equal_dof_p") : undefined;
  const markHLast = frame.has("mark_h_last") ? requireArray(frame, "mark_h_last") : undefined;
  const errHAtMark = frame.has("error_h_at_mark") ? requireArray(frame, "error_h_at_mark") : undefined;
  const errPAtMark = frame.has("error_p_at_mark") ? requireArray(frame, "error_p_at_mark") : undefined;
  const errHLastAtMark = frame.has("error_h_last_at_mark") ? requireArray(frame, "error_h_last_at_mark") : undefined;
  if (markEqualP === undefined || errHAtMark === undefined || errPAtMark === undefined) {
    degraded.push("帧缺等自由度标记：不画同自由度对照点（数字仍见右栏 meta）");
  }

  const floorText = stringOf(frame, "floor");
  const floor = floorText !== undefined && Number(floorText) > 0 ? Number(floorText) : FLOOR_FALLBACK;
  if (floorText === undefined) degraded.push("帧 strings 缺 floor：地板按 1e-17 显示");

  // —— meta ——
  const pmaxMeta = metaNumber(frame, "pmax");
  const pMax = Math.max(1, Math.round(pmaxMeta ?? maxFinite(pAxis) ?? 1));
  const solutionIdx = metaNumber(frame, "solution_index");
  const strategyIdx = metaNumber(frame, "strategy_index");
  const metricIdx = metaNumber(frame, "metric_index");
  const slopeH = metaNumber(frame, "slope_h_dof");
  const r2H = metaNumber(frame, "r2_h_dof");
  const slopeSemilog = metaNumber(frame, "slope_p_semilog");
  const r2Semilog = metaNumber(frame, "r2_p_semilog");
  const slopeLoglog = metaNumber(frame, "slope_p_loglog");
  const r2Loglog = metaNumber(frame, "r2_p_loglog");
  const expBetter = frame.meta["exponential_better"] === true;
  const equalDofMeta = metaNumber(frame, "equal_dof") ?? firstFinite(markEqualP);
  const errHMark = metaNumber(frame, "err_h_at_pdof") ?? firstFinite(errHAtMark);
  const errPMark = metaNumber(frame, "err_p_at_hdof") ?? firstFinite(errPAtMark);
  const gain = metaNumber(frame, "gain_p_over_h");
  const dofGap = metaNumber(frame, "h_dof_gap");
  const condFirst = metaNumber(frame, "cond_p_first");
  const condLast = metaNumber(frame, "cond_p_last");
  const condGrowth = metaNumber(frame, "cond_growth");
  const condMeasured = frame.meta["cond_at_pmax_measured"] === true;
  const errHLast = metaNumber(frame, "err_h_last");
  const errPLast = metaNumber(frame, "err_p_last");
  const solutionName = solutionIdx !== undefined ? (Math.round(solutionIdx) === 1 ? "kink" : "smooth") : "?";
  const strategyName = strategyIdx !== undefined ? (["both", "h", "p"][Math.round(strategyIdx)] ?? "?") : "?";
  const metricName = metricIdx !== undefined ? (Math.round(metricIdx) === 1 ? "L2" : "energy") : "?";

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: 0,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  const L = columnsLayout(scene, width, height, { topShare: 0.6, auxH: 150, gap: 56 });
  const narrow = !L.columns;
  const short = L.floor - L.topY < 250;
  const topH = narrow || short ? Math.max(60, L.floor - L.topY - 28) : L.topH;
  const mainW = narrow ? width - L.left - 24 : L.leftW;

  // —— 左上：误差 vs 自由度（双对数，窗口 1–1000 / 1e-17–10）——
  const ladder = makeAxis("ladder", [L.left, L.topY, mainW, topH], {
    xlim: [1, 300],
    ylim: [floor, 10],
    xscale: "log",
    yscale: "log",
    xlabel: "自由度 N（对数）",
    title: axisTitleOf(SPEC?.axes, "ladder", "误差 vs 自由度（双对数）"),
    ylabel: axisYlabelOf(SPEC?.axes, "ladder", "误差"),
    autoscaled: false,
  });
  scene.axes.push(ladder);
  scene.layers.push(
    lineLayer("ladder", series([1, 300], [floor, floor], pal.reference, { width: 1, dash: [3, 3], label: "误差地板" })),
  );
  if (fitH !== undefined && fitH.length === dofH.length) {
    const fitSeries = maskedSeries(dofH, fitH, undefined, pal.reference, { width: 1.2, dash: [5, 4], label: "h-加密拟合" });
    if (fitSeries.length > 0) scene.layers.push(lineLayer("ladder", fitSeries));
  }
  const hSeries = maskedSeries(dofH, errH, floorMaskH, pal.curves[0], { width: 2.3, marker: "o", label: "h-加密" });
  if (hSeries.length > 0) scene.layers.push(lineLayer("ladder", hSeries));
  const pLadderIdx: number[] = [];
  for (let i = 0; i < dofP.length; i += 1) {
    if ((dofP[i] as number) > 0) pLadderIdx.push(i);
  }
  if (pLadderIdx.length > 0) {
    const lx = new Float64Array(pLadderIdx.length);
    const ly = new Float64Array(pLadderIdx.length);
    const lm = new Float64Array(pLadderIdx.length);
    pLadderIdx.forEach((i, k) => {
      lx[k] = dofP[i] as number;
      ly[k] = errP[i] as number;
      lm[k] = floorMaskP !== undefined ? (floorMaskP[i] ?? 0) : 0;
    });
    const pSeries = maskedSeries(lx, ly, lm, pal.curves[1], { width: 2, marker: "s", dash: [6, 3], label: "p-加密" });
    if (pSeries.length > 0) scene.layers.push(lineLayer("ladder", pSeries));
  }

  // 等自由度与末端标记：竖线是读数的一部分，点只落在 frame 给的数值上。
  if (equalDofMeta !== undefined && equalDofMeta >= 1) {
    scene.layers.push(
      lineLayer("ladder", series([equalDofMeta, equalDofMeta], ladder.ylim, pal.marker, { width: 1, dash: [4, 4] })),
    );
    if (errHMark !== undefined && errHMark > 0) {
      scene.layers.push(scatterLayer("ladder", [equalDofMeta], [errHMark], { r: 4, color: pal.curves[0] }));
    }
    if (errPMark !== undefined && errPMark > 0) {
      scene.layers.push(scatterLayer("ladder", [equalDofMeta], [errPMark], { r: 4, color: pal.curves[1] }));
    }
  }
  const hLast = firstFinite(markHLast) ?? (dofH.length > 0 ? (dofH[dofH.length - 1] as number) : undefined);
  if (hLast !== undefined && hLast > 1 && hLast <= ladder.xlim[1] && errHLastAtMark !== undefined) {
    const e = firstFinite(errHLastAtMark);
    scene.layers.push(
      lineLayer("ladder", series([hLast, hLast], ladder.ylim, pal.reference, { width: 1, dash: [2, 4] })),
    );
    if (e !== undefined && e > 0) scene.layers.push(scatterLayer("ladder", [hLast], [e], { r: 3.6, color: pal.exact }));
  }

  // —— 左下：误差 vs p（半对数）——
  if (!narrow && !short) {
    const vsP = makeAxis("vs_p", [L.left, L.botY, L.leftW, L.botH], {
      xlim: [1, pMax],
      ylim: [floor, 10],
      yscale: "log",
      xlabel: "单元阶次 p",
      title: axisTitleOf(SPEC?.axes, "vs_p", "误差 vs p（半对数）"),
      ylabel: axisYlabelOf(SPEC?.axes, "vs_p", "误差"),
      autoscaled: false,
    });
    scene.axes.push(vsP);
    scene.layers.push(
      lineLayer("vs_p", series([1, pMax], [floor, floor], pal.reference, { width: 1, dash: [3, 3], label: "误差地板" })),
    );
    if (fitExp !== undefined && fitExp.length === pAxis.length) {
      const fitSeries = maskedSeries(pAxis, fitExp, undefined, pal.exact, { width: 1.4, dash: [6, 3], label: "指数收敛参考" });
      if (fitSeries.length > 0) scene.layers.push(lineLayer("vs_p", fitSeries));
    }
    if (fitAlg !== undefined && fitAlg.length === pAxis.length) {
      const fitSeries = maskedSeries(pAxis, fitAlg, undefined, pal.residual, { width: 1.3, dash: [2, 3], label: "代数收敛参考" });
      if (fitSeries.length > 0) scene.layers.push(lineLayer("vs_p", fitSeries));
    }
    const pSeries = maskedSeries(pAxis, errVsP, floorMaskP, pal.curves[1], { width: 2.3, marker: "s", label: "p-加密" });
    if (pSeries.length > 0) scene.layers.push(lineLayer("vs_p", pSeries));
    if (errPMark !== undefined && errPMark > 0) {
      scene.layers.push(scatterLayer("vs_p", [pMax], [errPMark], { r: 4, color: pal.curves[1] }));
    }
    scene.texts.push(
      ...textBlock(
        L.left + 6,
        L.botY + L.botH - SAFE_GAP - 6 * TEXT_ROW_H,
        L.botY + L.botH - SAFE_GAP - TEXT_ROW_H,
        ["绿=误差 vs p　橙虚=半对数拟合　紫虚=双对数拟合", "地板点不画成数据；灰虚=地板"],
        pal.muted,
        unitBudget(L.leftW),
        "bottom",
      ),
    );
  }

  // —— 右上：条件数 vs p（掩码点不进图层）——
  if (!narrow && !short) {
    const measured: number[] = [];
    for (let i = 0; i < cond.length; i += 1) {
      const v = cond[i] as number;
      if ((condMask[i] ?? 0) <= 0.5 && Number.isFinite(v) && v > 0 && v < 1e299) measured.push(v);
    }
    const lo = measured.length > 0 ? Math.min(...measured) : 1;
    const hi = measured.length > 0 ? Math.max(...measured) : 10;
    const condAxis = makeAxis("condition", [L.rightX, L.topY, L.rightW, L.auxH], {
      xlim: [1, pMax],
      ylim: logRange(lo * 0.7, Math.max(hi * 1.5, lo * 2)),
      yscale: "log",
      xlabel: "阶次 p",
      title: axisTitleOf(SPEC?.axes, "condition", "条件数 vs p（自动缩放）"),
      ylabel: axisYlabelOf(SPEC?.axes, "condition", "cond"),
      autoscaled: true,
    });
    scene.axes.push(condAxis);
    if (condAxis.ylim[0] <= 1 && condAxis.ylim[1] >= 1) {
      scene.layers.push(
        lineLayer("condition", series([1, pMax], [1, 1], pal.reference, { width: 1, dash: [2, 4] })),
      );
    }
    const condSeries = maskedSeries(
      pAxis,
      cond,
      condMask,
      pal.exact,
      { width: 2, marker: "o" },
      (v) => v > 0 && v < 1e299,
    );
    if (condSeries.length > 0) scene.layers.push(lineLayer("condition", condSeries));
    scene.texts.push(
      ...textBlock(
        L.rightX + 6,
        L.topY + 6,
        L.topY + 6 + 3 * TEXT_ROW_H,
        [`橙=条件数；未测 ${countMask(condMask)} 点不画`, "灰虚=cond=1（下界）"],
        pal.muted,
        unitBudget(L.rightW),
        "top",
      ),
    );
  }

  // —— 图例与右列文字 ——
  if (!narrow && !short) {
    scene.texts.push(
      ...textBlock(
        L.left + 6,
        L.topY + L.topH - SAFE_GAP - 8 * TEXT_ROW_H,
        L.topY + L.topH - SAFE_GAP - TEXT_ROW_H,
        [
          "蓝=h-加密　绿虚=p-加密（按自由度）　灰虚=h 拟合",
          `红竖线=等自由度 N=${fmt(equalDofMeta, 4)}；橙点=h 末端`,
        ],
        pal.muted,
        unitBudget(L.leftW),
        "bottom",
      ),
    );
  }

  if (L.columns && !short && L.floor > L.textTop + 8) {
    const lines: string[] = [];
    lines.push(`${solutionName} · ${strategyName} · ${metricName}（p≤${pMax}）`);
    lines.push(`末端误差 h=${fmt(errHLast)}　p=${fmt(errPLast)}`);
    if (gain !== undefined) lines.push(`同自由度优 ${formatNumber(gain, 3)}×`);
    if (slopeSemilog !== undefined && r2Semilog !== undefined) {
      lines.push(`p 阶梯半对数斜率 ${formatNumber(slopeSemilog, 4)}（R²=${formatNumber(r2Semilog, 4)}）`);
    }
    if (slopeLoglog !== undefined && r2Loglog !== undefined) {
      lines.push(`p 阶梯双对数斜率 ${formatNumber(slopeLoglog, 4)}（R²=${formatNumber(r2Loglog, 4)}）`);
    }
    lines.push(
      expBetter ? "指数拟合更优：这只对当前解的正则性成立" : "半对数并不更优：当前是代数收敛",
    );
    if (slopeH !== undefined && r2H !== undefined) {
      lines.push(`h 阶梯斜率 ${formatNumber(slopeH, 4)}（R²=${formatNumber(r2H, 4)}）`);
    }
    if (errHMark !== undefined && errPMark !== undefined && equalDofMeta !== undefined) {
      lines.push(
        `N=${formatNumber(equalDofMeta, 4)}：h=${formatNumber(errHMark, 3)}，p=${formatNumber(errPMark, 3)}`,
      );
    }
    if (dofGap !== undefined) lines.push(`h 用满 ${fmt(maxFinite(dofH), 3)} 自由度仍差 ${formatNumber(dofGap, 3)}×`);
    lines.push(
      `条件数 p: ${fmt(condFirst)}→${fmt(condLast)}` +
        (condGrowth !== undefined ? `（增长 ${fmt(condGrowth, 3)}×）` : "") +
        (condMeasured ? "" : "（末端未测）"),
    );
    lines.push(`未测条件数 ${countMask(condMask)} 点不画；地板点 ${countMask(floorMaskH) + countMask(floorMaskP)} 个`);
    scene.texts.push(...textBlock(L.rightX, L.textTop, L.floor, lines, pal.muted, unitBudget(L.rightW), "top"));
  } else if (narrow) {
    degraded.push("窄画布（单列）：只画误差-自由度主图；vs_p/条件数副轴与右列说明未排下");
  } else if (short) {
    degraded.push("画布过矮：副轴与右列说明未排下，关键读数见角标与 pitfalls");
  }

  // —— 角标 ——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(badge("autoscale", "ladder/vs_p 固定；条件数副轴自适应", 8, 26), SLOT.aux, width),
  );
  const status: string[] = [];
  status.push(`${solutionName}`);
  status.push(`${metricName}`);
  if (r2Semilog !== undefined && r2Loglog !== undefined) {
    status.push(`R²半/全 ${formatNumber(r2Semilog, 3)}/${formatNumber(r2Loglog, 3)}`);
  }
  scene.badges.push(placeBadge(badge("status", status.join("；")), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  const nBad =
    countNonFinite(dofH) +
    countNonFinite(errH) +
    countNonFinite(dofP) +
    countNonFinite(errP) +
    countNonFinite(pAxis) +
    countNonFinite(errVsP) +
    countNonFinite(cond) +
    countNonFinite(condMask) +
    (floorMaskH !== undefined ? countNonFinite(floorMaskH) : 0) +
    (floorMaskP !== undefined ? countNonFinite(floorMaskP) : 0);
  applyOverflow(scene, nBad, { slot: SLOT.overflow, width });
  return scene;
};
