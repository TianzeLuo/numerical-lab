/**
 * `adaptive_mesh`（4.4 章 viz 4）的 model：后验指示子与自适应加密。
 *
 * 面板与图层
 * ----------
 * | 面板 | 轴 | 内容 |
 * |---|---|---|
 * | 左上 | `mesh` | 一维网格：单元横线 + 节点 + 每单元 η_K 的竖线（高度/颜色 ∝ log η） |
 * | 左下 | `indicator` | `element_eta` 与 `error`（requiredAux，对数轴，地板线） |
 * | 右上 | `history` | 自适应/一致加密的能量误差 vs 自由度 + 总效率指数 + 当前层竖线 |
 *
 * 三条防误导纪律
 * --------------
 * 1. **η 是分布**：逐单元 η_K 与真误差并排画；效率指数的 median/spread 写进右栏，
 *    不把某一层的总值当常数。`ratio_undefined=1` 的占位值（1e-12）永不进图层。
 * 2. **网格质量与误差是两笔账**：主图让最小单元（压在界面附近）看得见，
 *    右栏同时报 max h/min h；加密到 1e-4 的网格"难看"但同自由度更准。
 * 3. **同自由度比较**：history 的横轴是自由度，不是加密层数；当前层竖线落在
 *    `dof_adaptive` 的读数上，`error_at_floor=1` 的层不画成数据。
 *
 * 时间轴：`scrub`，`t` 是加密层比例（帧 `meta.t` 优先）；实际层数在 `meta.level`。
 */

import { formatNumber } from "../../shared/src/axes.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import {
  applyOverflow,
  canvasSize,
  clamp,
  criticalOf,
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
  toNumeric,
  type SceneModel,
} from "../../shared/src/model.js";
import {
  axisModeBadge,
  badge,
  criticalBadge,
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
  textBlock,
  unitBudget,
} from "../_common/panel.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "4.4";
const VIZ = "adaptive_mesh";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "后验指示子与自适应网格：η 是分布，不是常数";

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

function firstFinite(a: ArrayLike<number> | undefined): number | undefined {
  if (a === undefined) return undefined;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i] as number;
    if (Number.isFinite(v)) return v;
  }
  return undefined;
}

function minMax(a: ArrayLike<number>): [number, number] | undefined {
  let lo = Number.POSITIVE_INFINITY;
  let hi = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i] as number;
    if (!Number.isFinite(v)) continue;
    lo = Math.min(lo, v);
    hi = Math.max(hi, v);
  }
  return Number.isFinite(lo) ? [lo, hi] : undefined;
}

function fmt(v: number | undefined, digits = 3): string {
  return v === undefined || !Number.isFinite(v) ? "?" : formatNumber(v, digits);
}

export const adaptiveMeshModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需数组 ——
  const meshX = requireArray(frame, "mesh_x");
  requireArray(frame, "elements");
  const elementsRaw = frame.matrix("elements");
  const elements = { rows: elementsRaw.rows, cols: elementsRaw.cols, values: toNumeric(elementsRaw.values) };
  if (elements.cols !== 2) throw new Error(`帧 ${CHAPTER}/${VIZ} 的 elements 第二维必须是 2，得到 ${elements.cols}`);
  const eta = requireArray(frame, "element_eta");
  const trueErr = requireArray(frame, "error");
  const levelAxis = requireArray(frame, "level_axis");
  const dofA = requireArray(frame, "dof_adaptive");
  const errA = requireArray(frame, "error_adaptive");
  const dofU = requireArray(frame, "dof_uniform");
  const errU = requireArray(frame, "error_uniform");
  const effectivity = requireArray(frame, "effectivity");
  const markLevel = requireArray(frame, "mark_level");
  if (eta.length !== elements.rows) throw new Error(`帧 ${CHAPTER}/${VIZ} 的 element_eta 长度 ≠ 单元数`);
  if (trueErr.length !== elements.rows) throw new Error(`帧 ${CHAPTER}/${VIZ} 的 error 长度 ≠ 单元数`);
  if (levelAxis.length === 0) throw new Error(`帧 ${CHAPTER}/${VIZ} 的 level_axis 为空：没有可画的层`);
  for (const [name, arr] of [
    ["error_adaptive", errA],
    ["dof_uniform", dofU],
    ["error_uniform", errU],
    ["effectivity", effectivity],
  ] as const) {
    requireSameLength(levelAxis, arr, `帧数组 level_axis 与 ${name}`);
  }
  requireSameLength(dofA, errA, "帧数组 dof_adaptive 与 error_adaptive");
  requireSameLength(dofA, effectivity, "帧数组 dof_adaptive 与 effectivity");
  for (let i = 0; i < elements.values.length; i += 1) {
    const k = elements.values[i] as number;
    if (!Number.isInteger(k) || k < 0 || k >= meshX.length) {
      throw new Error(`帧 ${CHAPTER}/${VIZ} 的 elements 含越界顶点编号 ${k}（n_node=${meshX.length}）`);
    }
  }
  const pitfalls = requirePitfalls(frame);

  // —— 可选掩码 ——
  const degraded: string[] = [];
  const floorMask = frame.has("error_at_floor") ? requireArray(frame, "error_at_floor") : undefined;
  if (floorMask === undefined) degraded.push("帧缺 error_at_floor：自适应误差的地板层不排除（不标已到地板）");
  else if (floorMask.length !== levelAxis.length) {
    degraded.push("error_at_floor 长度与 level_axis 不符：不排除地板层");
  }
  const ratioMask = frame.has("ratio_undefined") ? requireArray(frame, "ratio_undefined") : undefined;
  if (ratioMask === undefined) degraded.push("帧缺 ratio_undefined：逐单元效率指数的未定义计数只按 meta");

  const floorText = stringOf(frame, "floor");
  const floor = floorText !== undefined && Number(floorText) > 0 ? Number(floorText) : FLOOR_FALLBACK;
  if (floorText === undefined) degraded.push("帧 strings 缺 floor：地板按 1e-17 显示");

  // —— 时间 ——
  const tMeta = metaNumber(frame, "t_actual") ?? metaNumber(frame, "t");
  const t = clamp(tMeta ?? opts.t, 0, 1);
  if (tMeta === undefined) degraded.push("帧 meta 缺 t/t_actual：时间取请求值并夹到 [0,1]");
  const levelsMeta = metaNumber(frame, "levels");
  const levelMeta = metaNumber(frame, "level");
  const levMeta = levelMeta ?? firstFinite(markLevel);
  const lev = levMeta !== undefined ? Math.max(0, Math.min(levelAxis.length - 1, Math.round(levMeta))) : 0;
  if (levelMeta === undefined) degraded.push("帧 meta 缺 level：当前层按 mark_level 兜底（不猜）并夹到层范围");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  const L = columnsLayout(scene, width, height, { topShare: 0.6, auxH: 150, gap: 56 });
  const narrow = !L.columns;
  const short = L.floor - L.topY < 250;
  const topH = narrow || short ? Math.max(60, L.floor - L.topY - 28) : L.topH;
  const mainW = narrow ? width - L.left - 24 : L.leftW;
  const etaRange = minMax(eta);
  const etaMax = Math.max(etaRange?.[1] ?? 1, floor * 10);
  const logLo = Math.log10(floor);
  const logHi = Math.log10(Math.max(etaMax, floor * 10));

  // —— 左上：网格 + 逐单元指示子 ——
  const meshAxis = makeAxis("mesh", [L.left, L.topY, mainW, topH], {
    xlim: [0, 1],
    ylim: [0, 1],
    xlabel: "x（单元位置）",
    title: axisTitleOf(SPEC?.axes, "mesh", "当前网格与指示子 η_K（固定 0–1）"),
    ylabel: axisYlabelOf(SPEC?.axes, "mesh", "η_K 归一"),
    autoscaled: false,
  });
  scene.axes.push(meshAxis);
  const baseY = 0.42;
  scene.layers.push(
    lineLayer("mesh", series([0, 1], [baseY, baseY], pal.reference, { width: 1, dash: [3, 3] })),
  );
  const stems: LineSeries[] = [];
  const cells: LineSeries[] = [];
  const mid: number[] = [];
  for (let e = 0; e < elements.rows; e += 1) {
    const ia = elements.values[e * 2] as number;
    const ib = elements.values[e * 2 + 1] as number;
    const xa = meshX[ia] as number;
    const xb = meshX[ib] as number;
    if (!Number.isFinite(xa) || !Number.isFinite(xb)) continue;
    const xm = (xa + xb) / 2;
    mid.push(xm);
    const value = eta[e] as number;
    const lv = Math.log10(Math.max(Number.isFinite(value) ? value : floor, floor));
    const tEta = logHi > logLo ? clamp((lv - logLo) / (logHi - logLo), 0, 1) : 0;
    const color = tEta > 0.66 ? pal.warn : tEta > 0.33 ? pal.exact : pal.curves[0];
    const height = 0.06 + 0.4 * tEta;
    stems.push(series([xm, xm], [baseY, baseY + height], color, { width: 2.6 }));
    cells.push(series([xa, xb], [baseY, baseY], pal.muted, { width: 2 }));
  }
  if (cells.length > 0) scene.layers.push(lineLayer("mesh", cells));
  if (stems.length > 0) scene.layers.push(lineLayer("mesh", stems));
  if (mid.length > 0) {
    scene.layers.push(scatterLayer("mesh", mid, mid.map(() => baseY), { r: 2.1, color: pal.ink, alpha: 0.9 }));
  }
  scene.texts.push(
    ...textBlock(
      L.left + 6,
      L.topY + L.topH - SAFE_GAP - 6 * TEXT_ROW_H,
      L.topY + L.topH - SAFE_GAP - TEXT_ROW_H,
      ["横线=单元（宽度=h_K）　点=节点", "竖线高度/颜色=log η_K（蓝低→红高）"],
      pal.muted,
      unitBudget(mainW),
      "bottom",
    ),
  );

  // —— 左下：η_K 与真误差（对数）——
  if (!narrow && !short) {
    const indicator = makeAxis("indicator", [L.left, L.botY, L.leftW, L.botH], {
      xlim: [0, 1],
      ylim: [floor, 1],
      yscale: "log",
      xlabel: "单元中心 x",
      title: axisTitleOf(SPEC?.axes, "indicator", "η_K 与真误差（对数，地板 1e-17）"),
      ylabel: axisYlabelOf(SPEC?.axes, "indicator", "误差/指示子"),
      autoscaled: false,
    });
    scene.axes.push(indicator);
    scene.layers.push(
      lineLayer("indicator", series([0, 1], [floor, floor], pal.reference, { width: 1, dash: [3, 3] })),
    );
    if (mid.length > 0) {
      const etaSeries = maskedSeries(mid, eta, undefined, pal.curves[0], { width: 1.9, marker: "o" });
      const errSeries = maskedSeries(mid, trueErr, undefined, pal.exact, { width: 1.7, marker: "s", dash: [5, 3] });
      if (etaSeries.length > 0) scene.layers.push(lineLayer("indicator", etaSeries));
      if (errSeries.length > 0) scene.layers.push(lineLayer("indicator", errSeries));
    }
    const ibudget = Math.min(38, unitBudget(L.leftW));
    const legendTop = L.botY + Math.max(6, Math.floor((L.botH - 3 * TEXT_ROW_H) / 2));
    scene.texts.push(
      ...textBlock(
        L.left + L.leftW - 6 - ibudget * 7,
        legendTop,
        legendTop + 3 * TEXT_ROW_H,
        ["蓝=η_K（只用 u_h 与系数场）", "橙虚=逐单元真误差（要解析解）", "灰虚=地板"],
        pal.muted,
        ibudget,
        "top",
      ),
    );
  }

  // —— 右上：误差 vs 自由度 + 总效率指数 ——
  if (!narrow && !short) {
    const dofAll = [...Array.from(dofA), ...Array.from(dofU)];
    const dofRange = minMax(dofAll);
    const dLo = Math.max(1, dofRange?.[0] ?? 1);
    const dHi = Math.max(dLo * 1.5, dofRange?.[1] ?? 10);
    const hist = makeAxis("history", [L.rightX, L.topY, L.rightW, L.auxH], {
      xlim: [dLo * 0.8, dHi * 1.25],
      ylim: [floor, 10],
      xscale: "log",
      yscale: "log",
      xlabel: "自由度 N（对数）",
      title: axisTitleOf(SPEC?.axes, "history", "能量误差 vs 自由度（双对数）"),
      ylabel: axisYlabelOf(SPEC?.axes, "history", "能量误差"),
      autoscaled: false,
    });
    scene.axes.push(hist);
    scene.layers.push(
      lineLayer("history", series(hist.xlim, [floor, floor], pal.reference, { width: 1, dash: [3, 3] })),
    );
    const aSeries = maskedSeries(dofA, errA, floorMask, pal.curves[0], { width: 2.2, marker: "o" });
    if (aSeries.length > 0) scene.layers.push(lineLayer("history", aSeries));
    const uSeries = maskedSeries(dofU, errU, undefined, pal.curves[1], { width: 1.8, marker: "s", dash: [6, 3] });
    if (uSeries.length > 0) scene.layers.push(lineLayer("history", uSeries));
    const effX: number[] = [];
    const effY: number[] = [];
    for (let i = 0; i < Math.min(dofA.length, effectivity.length); i += 1) {
      const x = dofA[i] as number;
      const y = effectivity[i] as number;
      if (Number.isFinite(x) && x > 0 && Number.isFinite(y) && y > 0) {
        effX.push(x);
        effY.push(y);
      }
    }
    if (effX.length > 0) {
      scene.layers.push(scatterLayer("history", effX, effY, { r: 2.9, color: pal.exact, alpha: 0.9 }));
    }
    const dofNow = dofA[lev] as number;
    const errNow = errA[lev] as number;
    if (Number.isFinite(dofNow) && dofNow > 0) {
      scene.layers.push(
        lineLayer("history", series([dofNow, dofNow], hist.ylim, pal.marker, { width: 1, dash: [4, 4] })),
      );
      if (Number.isFinite(errNow) && errNow > 0 && (floorMask === undefined || (floorMask[lev] ?? 0) <= 0.5)) {
        scene.layers.push(scatterLayer("history", [dofNow], [errNow], { r: 3.8, color: pal.marker }));
      }
    }
    scene.texts.push(
      ...textBlock(
        L.rightX + 6,
        L.topY + L.auxH - SAFE_GAP - 6 * TEXT_ROW_H,
        L.topY + L.auxH - SAFE_GAP - TEXT_ROW_H,
        ["蓝=自适应　绿虚=一致加密　橙点=总效率指数", "红竖线/红点=当前层；地板层不画点"],
        pal.muted,
        unitBudget(L.rightW),
        "bottom",
      ),
    );
  }

  // —— 右列文字 ——
  const strategyIdx = metaNumber(frame, "strategy_index");
  const theta = metaNumber(frame, "theta");
  const nElem = metaNumber(frame, "n_elem") ?? elements.rows;
  const nNode = metaNumber(frame, "n_node") ?? meshX.length;
  const minH = metaNumber(frame, "min_h");
  const maxH = metaNumber(frame, "max_h");
  const hRatio = metaNumber(frame, "h_ratio");
  const etaSum = metaNumber(frame, "eta_sum");
  const etaNorm = metaNumber(frame, "eta_norm");
  const etaMaxMeta = metaNumber(frame, "eta_max");
  const etaShareMax = metaNumber(frame, "eta_share_max");
  const effMin = metaNumber(frame, "effectivity_min_defined");
  const effMax = metaNumber(frame, "effectivity_max_defined");
  const effMedian = metaNumber(frame, "effectivity_median");
  const effSpread = metaNumber(frame, "effectivity_spread");
  const effDefined = metaNumber(frame, "effectivity_defined_count");
  const ratioUndef = metaNumber(frame, "ratio_undefined_count");
  const errorNow = metaNumber(frame, "error_now");
  const errorL2 = metaNumber(frame, "error_l2_now");
  const logCorr = metaNumber(frame, "log_corr");
  const bestRatio = metaNumber(frame, "best_ratio");
  const medianRatio = metaNumber(frame, "median_ratio");
  const ratioNow = metaNumber(frame, "ratio_at_equal_dof");
  const uniformTruncated = frame.meta["uniform_truncated"] === true;
  const uniformDof = metaNumber(frame, "uniform_dof_last");
  const adaptiveDof = metaNumber(frame, "adaptive_dof_last");
  const kinkShare = metaNumber(frame, "kink_error_share");

  if (L.columns && !short && L.floor > L.textTop + 8) {
    const head = paramSummary(frame, opts, [
      { name: "theta", label: "θ" },
      { name: "ratio", label: "a₂/a₁" },
    ]);
    degraded.push(...head.degraded);
    const lines: string[] = [];
    lines.push(
      `${strategyIdx !== undefined && Math.round(strategyIdx) === 1 ? "uniform" : "adaptive"}　${head.text}` +
        (levelsMeta !== undefined ? `　ℓ=${lev}/${Math.round(levelsMeta)}` : `　ℓ=${lev}`),
    );
    lines.push(`单元 ${fmt(nElem, 0)}　节点 ${fmt(nNode, 0)}　h ${fmt(minH)}…${fmt(maxH)}（比 ${fmt(hRatio)}）`);
    if (etaSum !== undefined && etaNorm !== undefined) {
      lines.push(`η 总和 ${formatNumber(etaSum, 3)}　范数 ${formatNumber(etaNorm, 3)}　最大 ${fmt(etaMaxMeta)}`);
    }
    lines.push(
      `效率指数 ${fmt(effMin)}…${fmt(effMax)}　中位 ${fmt(effMedian)}` +
        (effSpread !== undefined ? `　散布 ${formatNumber(effSpread, 3)}` : ""),
    );
    lines.push(
      `已定义 ${fmt(effDefined, 0)}　未定义 ${fmt(ratioUndef, 0)}（1e-12 占位不画）`,
    );
    if (etaShareMax !== undefined) lines.push(`η 份额 max ${formatNumber(etaShareMax, 3)}`);
    lines.push(`当前能量误差 ${fmt(errorNow)}　L2 ${fmt(errorL2)}`);
    if (bestRatio !== undefined || medianRatio !== undefined) {
      lines.push(`同自由度比：最优 ${fmt(bestRatio)}　中位 ${fmt(medianRatio)}`);
      if (ratioNow !== undefined) lines.push(`当前 ${fmt(ratioNow)}`);
    }
    if (logCorr !== undefined) lines.push(`log η 与 log 真误差相关 ${formatNumber(logCorr, 3)}：不成比例`);
    if (kinkShare !== undefined) lines.push(`界面附近单元误差份额 ${formatNumber(kinkShare, 3)}`);
    lines.push(
      uniformTruncated
        ? `一致阶梯被截断（到 ${fmt(uniformDof, 0)} 自由度）`
        : `末端自由度：自适应 ${fmt(adaptiveDof, 0)} / 一致 ${fmt(uniformDof, 0)}`,
    );
    scene.texts.push(...textBlock(L.rightX, L.textTop, L.floor, lines, pal.muted, unitBudget(L.rightW), "top"));
  } else if (narrow) {
    degraded.push("窄画布（单列）：只画网格主图；指示子/历史副面板与右列说明未排下");
  } else if (short) {
    degraded.push("画布过矮：副面板与右列说明未排下，关键读数见角标与 pitfalls");
  }

  // —— 角标 ——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(badge("autoscale", "网格/指示子/历史窗固定；颜色=log η（归一）", 8, 26), SLOT.aux, width),
  );
  if (theta !== undefined) {
    scene.badges.push(
      placeBadge(criticalBadge("θ", theta, criticalOf(frame, "theta", 1), { beyond: "above" }), SLOT.critical, width),
    );
  }
  const status: string[] = [];
  status.push(`ℓ=${lev}`);
  if (nElem !== undefined) status.push(`单元 ${formatNumber(nElem, 0)}`);
  if (hRatio !== undefined) status.push(`h比 ${formatNumber(hRatio, 3)}`);
  if (bestRatio !== undefined) status.push(`最优比 ${formatNumber(bestRatio, 3)}`);
  if (ratioUndef !== undefined) status.push(`η未定义 ${formatNumber(ratioUndef, 0)}`);
  scene.badges.push(placeBadge(badge("status", status.join("；")), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  const nBad =
    countNonFinite(meshX) +
    countNonFinite(elements.values) +
    countNonFinite(eta) +
    countNonFinite(trueErr) +
    countNonFinite(levelAxis) +
    countNonFinite(dofA) +
    countNonFinite(errA) +
    countNonFinite(dofU) +
    countNonFinite(errU) +
    countNonFinite(effectivity) +
    countNonFinite(markLevel) +
    (floorMask !== undefined ? countNonFinite(floorMask) : 0) +
    (ratioMask !== undefined ? countNonFinite(ratioMask) : 0);
  applyOverflow(scene, nBad, { slot: SLOT.overflow, width });
  return scene;
};
