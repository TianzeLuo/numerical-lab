/**
 * `energy_drift_duel`（3.3 章）的 model：辛格式 vs 非辛基线的长时能量误差。
 *
 * 这张图最容易撒的三个谎，以及本模型的对应做法：
 *
 * 1. **用一个自适应纵轴把两条曲线“拉平”**：主轴按**两条曲线共同**的 |误差| 上界取对称窗口
 *    （`autoscaled: true`，manifest 如此声明），不按辛方法自己那条带取——否则对照曲线的
 *    单调趋势会被裁到画外。对照曲线在 h 越过稳定边界后会到 1e23 量级，那时辛曲线确实被压成
 *    一条线：这是**真实量级差**，文本里会点名，绝不用截断把它藏掉。
 * 2. **把掩码后的占位值当数据画**：积分停止后 `mask_* = 0` 的位置是重复的末值。模型按
 *    `runsBy(mask)` 切段，只画 `mask > 0.5` 的连续段（`sliceRun`），占位值一个都不进图层。
 * 3. **只看“谁更陡”**：辛曲线在带宽里来回振荡，斜率没有意义。主判据（带宽比、趋势/带宽、
 *    后半段/前半段 sup 比、短期自收敛误差）全部从 `meta` 读出、写进右侧文字，不靠目测。
 *
 * 图层顺序（=绘制顺序）：误差带 → 零线/带边 → 趋势线 → 两条误差曲线（按掩码分段）→
 * 爆点竖线。副轴是一张“量级对照”点图：带宽 / 趋势 / 短期误差六个数共用同一对数刻度。
 */

import { formatNumber } from "../../shared/src/axes.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import {
  applyOverflow,
  canvasSize,
  criticalOf,
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
  bandLayer,
  criticalBadge,
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
  indexAxis,
  legendBlock,
  logRange,
  runsBy,
  sliceRun,
  textBlock,
  unitBudget,
  type Run,
} from "../_common/panel.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "3.3";
const VIZ = "energy_drift_duel";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "能量漂移对决：10⁶ 步的辛 vs 非辛";
const H_CRITICAL = SPEC?.params?.find((p) => p.name === "h")?.critical?.[0] ?? 2.0;
/** 对数副轴的地板：低于它的值按地板画（不是“恰好等于 1e-16”）。 */
const FLOOR = 1e-16;

const SLOT = {
  source: 0,
  axisMode: 1,
  aux: 2,
  critical: 3,
  status: 4,
  overflow: 6,
  degraded: 7,
} as const;

/** 掩码 > 0.5 的连续段（占位值段不返回）。 */
function trustedRuns(mask: ArrayLike<number>): Run[] {
  return runsBy(mask).filter((r) => (mask[r.start] ?? 0) > 0.5);
}

/** 对数轴上的可绘值：有限且 > 0，否则落到地板。 */
function logValue(v: number): number {
  return Number.isFinite(v) && v > 0 ? Math.max(v, FLOOR) : FLOOR;
}

/** 方法标签的短名：去掉帧文案里的括注（完整名称仍可从帧 strings 读到）。 */
function shortLabel(v: string | undefined): string {
  return v === undefined ? "?" : v.split("（")[0]!.trim();
}

/** 按掩码分段画一条曲线（每段一个 series；孤立点用散点补上）。 */
function maskedCurves(
  axis: string,
  x: ArrayLike<number>,
  y: ArrayLike<number>,
  mask: ArrayLike<number>,
  color: string,
  width: number,
  dash?: number[],
): { lines: ReturnType<typeof lineLayer>[]; dots: ReturnType<typeof scatterLayer>[] } {
  const lines: ReturnType<typeof lineLayer>[] = [];
  const dots: ReturnType<typeof scatterLayer>[] = [];
  const n = Math.min(x.length, y.length, mask.length);
  for (const run of trustedRuns(mask)) {
    const end = Math.min(run.end, n);
    if (end - run.start < 2) {
      if (run.start < n) dots.push(scatterLayer(axis, [x[run.start]], [y[run.start]], { r: 2.5, color }));
      continue;
    }
    const seg = sliceRun(x, y, { start: run.start, end });
    lines.push(lineLayer(axis, series(seg.x, seg.y, color, { width, dash })));
  }
  return { lines, dots };
}

export const energyDriftDuelModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需数组 ——
  const tArr = requireArray(frame, "t");
  const errS = requireArray(frame, "energy_error_symplectic");
  const errB = requireArray(frame, "energy_error_baseline");
  const maskS = requireArray(frame, "mask_symplectic");
  const maskB = requireArray(frame, "mask_baseline");
  requireSameLength(errS, maskS, "帧数组 energy_error_symplectic 与 mask_symplectic");
  requireSameLength(errB, maskB, "帧数组 energy_error_baseline 与 mask_baseline");
  const pitfalls = requirePitfalls(frame);

  // —— 可选数组：缺了只少画一层，但必须点名 ——
  const degraded: string[] = [];
  const bandHi = frame.has("energy_band_hi") ? requireArray(frame, "energy_band_hi") : undefined;
  const bandLo = frame.has("energy_band_lo") ? requireArray(frame, "energy_band_lo") : undefined;
  if (bandHi === undefined || bandLo === undefined) {
    degraded.push("帧缺 energy_band_hi/lo：不画辛方法的误差带（不按误差曲线自己算带宽）");
  } else if (bandHi.length !== errS.length || bandLo.length !== errS.length) {
    degraded.push("energy_band_hi/lo 与辛方法误差曲线长度不一致：误差带不画（拒绝错位）");
  }
  const driftS = frame.has("drift_line_symplectic") ? requireArray(frame, "drift_line_symplectic") : undefined;
  const driftB = frame.has("drift_line_baseline") ? requireArray(frame, "drift_line_baseline") : undefined;
  if (driftS === undefined || driftB === undefined) {
    degraded.push("帧缺 drift_line_symplectic/baseline：不画窗口内的最小二乘趋势线");
  }

  // —— meta ——
  const tAxis = metaNumber(frame, "t_axis") ?? metaNumber(frame, "t") ?? opts.t;
  const h = metaNumber(frame, "h");
  const nObs = metaNumber(frame, "n_steps_observed");
  const nMax = metaNumber(frame, "n_steps_max");
  const periods = metaNumber(frame, "n_periods_observed");
  const bandSMeta = metaNumber(frame, "band_symplectic");
  const bandBMeta = metaNumber(frame, "band_baseline");
  const trendSMeta = metaNumber(frame, "trend_symplectic");
  const trendBMeta = metaNumber(frame, "trend_baseline");
  const ratioSMeta = metaNumber(frame, "trend_over_band_symplectic");
  const ratioBMeta = metaNumber(frame, "trend_over_band_baseline");
  const growthSMeta = metaNumber(frame, "band_growth_symplectic");
  const growthBMeta = metaNumber(frame, "band_growth_baseline");
  const shortSMeta = metaNumber(frame, "short_term_error_symplectic");
  const shortBMeta = metaNumber(frame, "short_term_error_baseline");
  const shortWindow = metaNumber(frame, "short_term_window");
  const endBMeta = metaNumber(frame, "end_error_baseline");
  const blowS = frame.meta["blowup_symplectic"] === true;
  const blowB = frame.meta["blowup_baseline"] === true;
  const tBlowS = metaNumber(frame, "t_blowup_symplectic");
  const tBlowB = metaNumber(frame, "t_blowup_baseline");
  const nMasks = metaNumber(frame, "n_masked_symplectic");
  const nMaskB = metaNumber(frame, "n_masked_baseline");
  const status = stringOf(frame, "status");
  const methodS = stringOf(frame, "method_symplectic");
  const methodB = stringOf(frame, "method_baseline");
  const system = stringOf(frame, "system");
  if (tAxis === undefined) degraded.push("帧 meta 缺 t_axis/t：时间轴按请求 t 显示");
  if (h === undefined) degraded.push("帧 meta 缺 h：不显示步长（不按请求参数猜）");
  if (bandSMeta === undefined || bandBMeta === undefined) {
    degraded.push("帧 meta 缺 band_symplectic/band_baseline：量级对照图不画带宽两项");
  }
  if (trendSMeta === undefined || trendBMeta === undefined) {
    degraded.push("帧 meta 缺 trend_symplectic/trend_baseline：量级对照图不画趋势两项");
  }
  if (shortSMeta === undefined || shortBMeta === undefined) {
    degraded.push("帧 meta 缺 short_term_error_*：量级对照图不画短期误差两项");
  }

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: tAxis ?? opts.t,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  const L = columnsLayout(scene, width, height, { topShare: 0.62, auxH: 146, gap: 44 });
  const mainH = Math.max(70, L.floor - 28 - L.topY);
  const auxH = Math.max(60, Math.min(120, Math.round(mainH * 0.46)));

  // —— 横轴：帧的 t 数组覆盖两种方法时才用它；长度不一致就退回采样下标（并说明）——
  const sameLen = tArr.length === errS.length && errB.length === errS.length;
  if (!sameLen) {
    degraded.push(
      `帧 t 数组 ${tArr.length} 点与误差曲线（辛 ${errS.length} / 对照 ${errB.length}）长度不一致：` +
        "横轴改用采样下标（不按索引外推时间）",
    );
  }
  const xS = sameLen ? tArr : indexAxis(errS.length);
  const xB = sameLen ? tArr : indexAxis(errB.length);
  let x0 = Number.POSITIVE_INFINITY;
  let x1 = Number.NEGATIVE_INFINITY;
  for (const arr of [xS, xB]) {
    for (let i = 0; i < arr.length; i += 1) {
      const v = arr[i] as number;
      if (!Number.isFinite(v)) continue;
      if (v < x0) x0 = v;
      if (v > x1) x1 = v;
    }
  }
  if (!Number.isFinite(x0) || !Number.isFinite(x1) || !(x1 > x0)) {
    x0 = 0;
    x1 = 1;
  }

  // —— 主轴：两条曲线共同的 |误差| 上界（对称窗口，绝不只框住辛方法）——
  let maxAbs = 0;
  for (const [y, mask] of [
    [errS, maskS],
    [errB, maskB],
  ] as const) {
    for (const run of trustedRuns(mask)) {
      for (let i = run.start; i < Math.min(run.end, y.length); i += 1) {
        const v = y[i] as number;
        if (Number.isFinite(v)) maxAbs = Math.max(maxAbs, Math.abs(v));
      }
    }
  }
  if (bandHi !== undefined && bandHi.length === errS.length) {
    for (let i = 0; i < bandHi.length; i += 1) {
      const v = bandHi[i] as number;
      if (Number.isFinite(v)) maxAbs = Math.max(maxAbs, Math.abs(v));
    }
  }
  if (!(maxAbs > 0)) maxAbs = FLOOR;

  const main = makeAxis("main", [L.left, L.topY, L.leftW, mainH], {
    xlim: [x0, x1],
    ylim: [-1.35 * maxAbs, 1.35 * maxAbs],
    xlabel: sameLen ? "观测时间 t（数组只覆盖已观测步数）" : "采样下标（帧 t 与对照曲线长度不一致）",
    title: axisTitleOf(SPEC?.axes, "main", "能量相对误差 (H−H₀)/|H₀|"),
    ylabel: axisYlabelOf(SPEC?.axes, "main", "ΔE/E₀"),
    autoscaled: true,
    tickTarget: Math.max(3, Math.round(L.leftW / 100)),
  });

  // 1) 辛方法的误差带（帧里是一对常量数组；只画可信长度）
  if (bandHi !== undefined && bandLo !== undefined && bandHi.length === errS.length && bandLo.length === errS.length) {
    const bandX = sameLen ? tArr : indexAxis(errS.length);
    scene.layers.push(
      bandLayer("main", {
        x: bandX,
        ylo: bandLo,
        yhi: bandHi,
        color: pal.reference,
        alpha: 0.18,
        label: "辛方法的误差带",
      }),
    );
  }
  // 2) 零线与带边（带边是“有界”这句话的可读位置，必须真画）
  scene.layers.push(
    lineLayer("main", series([x0, x1], [0, 0], pal.reference, { width: 1, dash: [4, 4] })),
  );
  if (bandHi !== undefined && bandLo !== undefined && bandHi.length > 0 && bandLo.length > 0) {
    const hi = bandHi[0] as number;
    const lo = bandLo[0] as number;
    if (Number.isFinite(hi)) {
      scene.layers.push(
        lineLayer("main", series([x0, x1], [hi, hi], pal.reference, { width: 1, dash: [2, 3] })),
      );
    }
    if (Number.isFinite(lo)) {
      scene.layers.push(
        lineLayer("main", series([x0, x1], [lo, lo], pal.reference, { width: 1, dash: [2, 3] })),
      );
    }
  }
  // 3) 趋势线（窗口内的最小二乘拟合；只画与各自横轴等长的那条）
  if (driftS !== undefined && driftS.length === xS.length) {
    scene.layers.push(
      lineLayer("main", series(xS, driftS, pal.muted, { width: 1.2, dash: [7, 4] })),
    );
  } else if (driftS !== undefined) {
    degraded.push("drift_line_symplectic 与辛方法横轴长度不一致：趋势线不画");
  }
  if (driftB !== undefined && driftB.length === xB.length) {
    scene.layers.push(
      lineLayer("main", series(xB, driftB, pal.exact, { width: 1.2, dash: [7, 4] })),
    );
  } else if (driftB !== undefined) {
    degraded.push("drift_line_baseline 与对照横轴长度不一致：趋势线不画");
  }
  // 4) 两条误差曲线（掩码分段：占位值不进图层）
  const segS = maskedCurves("main", xS, errS, maskS, pal.curves[0] as string, 2.2);
  const segB = maskedCurves("main", xB, errB, maskB, pal.exact, 2.2);
  scene.layers.push(...segS.lines, ...segB.lines);
  scene.layers.push(...segS.dots, ...segB.dots);
  // 5) 爆点位置：竖线 + 末个可信点（占位值不画，但“停在哪”必须看得见）
  const runS = trustedRuns(maskS);
  const runB = trustedRuns(maskB);
  const lastS = runS.length > 0 ? runS[runS.length - 1].end - 1 : -1;
  const lastB = runB.length > 0 ? runB[runB.length - 1].end - 1 : -1;
  if (blowS && lastS >= 0) {
    scene.layers.push(
      lineLayer("main", series([xS[lastS], xS[lastS]], [main.ylim[0], main.ylim[1]], pal.warn, { width: 1, dash: [3, 3] })),
    );
    scene.layers.push(scatterLayer("main", [xS[lastS]], [errS[lastS]], { r: 3.5, color: pal.warn }));
  }
  if (blowB && lastB >= 0) {
    scene.layers.push(
      lineLayer("main", series([xB[lastB], xB[lastB]], [main.ylim[0], main.ylim[1]], pal.warn, { width: 1, dash: [3, 3] })),
    );
    scene.layers.push(scatterLayer("main", [xB[lastB]], [errB[lastB]], { r: 3.5, color: pal.warn }));
  }

  // —— 副轴：带宽 / 趋势 / 短期误差的“量级对照”（同一对数刻度）——
  const entries: Array<{ label: string; value: number | undefined; color: string }> = [
    { label: "带宽·辛", value: bandSMeta, color: pal.curves[0] as string },
    { label: "带宽·对照", value: bandBMeta, color: pal.exact },
    { label: "趋势·辛", value: trendSMeta, color: pal.curves[0] as string },
    { label: "趋势·对照", value: trendBMeta, color: pal.exact },
    { label: "短期·辛", value: shortSMeta, color: pal.curves[0] as string },
    { label: "短期·对照", value: shortBMeta, color: pal.exact },
  ].map((e) => ({ ...e, value: e.value === undefined ? undefined : Math.abs(e.value) }));
  const defined = entries.filter((e) => e.value !== undefined && Number.isFinite(e.value));
  let ledHi = FLOOR;
  for (const e of defined) ledHi = Math.max(ledHi, logValue(e.value as number));
  const ledX0 = -0.5;
  const ledX1 = entries.length - 0.5;
  const aux = makeAxis("aux", [L.rightX, L.topY, L.rightW, auxH], {
    xlim: [ledX0, ledX1],
    ylim: logRange(FLOOR / 2, ledHi * 2),
    yscale: "log",
    xticks: [],
    xlabel: "",
    title: axisTitleOf(SPEC?.axes, "aux", "带宽 / 趋势 / 短期误差（量级对照）"),
    ylabel: axisYlabelOf(SPEC?.axes, "aux", "量级（对数）"),
    autoscaled: true,
  });
  scene.axes.push(main, aux);
  // 地板线（对数轴必须有地板；低于它的值按地板画）
  scene.layers.push(
    lineLayer("aux", series([ledX0, ledX1], [FLOOR, FLOOR], pal.warn, { width: 1, dash: [2, 3] })),
  );
  // 带宽·辛 作为横向参考：短期误差比它小多少，一眼可见
  if (bandSMeta !== undefined && bandSMeta > 0) {
    const ref = logValue(bandSMeta);
    scene.layers.push(
      lineLayer("aux", series([ledX0, ledX1], [ref, ref], pal.reference, { width: 1, dash: [5, 4] })),
    );
  }
  for (const [i, e] of entries.entries()) {
    if (e.value === undefined || !Number.isFinite(e.value)) continue;
    const y = logValue(e.value);
    scene.layers.push(
      lineLayer("aux", series([i, i], [FLOOR, y], pal.muted, { width: 1 })),
    );
    scene.layers.push(scatterLayer("aux", [i], [y], { r: 3.2, color: e.color }));
  }

  // —— 面板内图例 ——
  const budgetW = Math.max(12, Math.floor((L.leftW - 12) / 7));
  scene.texts.push(
    ...legendBlock(
      L.left + 6,
      L.topY + 6,
      L.leftW,
      "蓝=辛方法　橙=非辛对照　灰带=辛的误差带（±sup）",
      pal.muted,
    ),
    ...textBlock(
      L.left + 6,
      L.topY + 23,
      L.topY + 3 * 15,
      ["虚线=窗口内最小二乘趋势；红虚线=积分停止处"],
      pal.muted,
      budgetW,
      "top",
    ),
  );
  const auxBudget = Math.max(8, Math.floor((L.rightW - 12) / 7));
  scene.texts.push(
    ...legendBlock(
      L.rightX + 6,
      L.topY + 6,
      L.rightW,
      "1 带宽辛　2 带宽对照　3 趋势辛",
      pal.muted,
    ),
    ...textBlock(
      L.rightX + 6,
      L.topY + 23,
      L.topY + Math.max(24, auxH - 3),
      ["4 趋势对照　5 短期辛　6 短期对照"],
      pal.muted,
      auxBudget,
      "top",
    ),
  );

  // —— 右列文字 ——
  const textTop = L.topY + auxH + 28 + 8;
  if (L.columns) {
    const lines: string[] = [];
    lines.push(`${shortLabel(system)}　${shortLabel(methodS)} vs ${shortLabel(methodB)}`);
    if (h !== undefined) {
      lines.push(
        `h=${formatNumber(h, 3)}　t=${formatNumber(scene.t, 3)}` +
          (periods !== undefined ? `　${formatNumber(periods, 4)} 个周期` : ""),
      );
    }
    if (nObs !== undefined) {
      lines.push(`观测 ${Math.round(nObs)} 步` + (nMax !== undefined ? `（上限 ${Math.round(nMax)}）` : ""));
    }
    if (bandSMeta !== undefined && bandBMeta !== undefined) {
      lines.push(`带宽：辛 ${formatNumber(bandSMeta, 3)} / 对照 ${formatNumber(bandBMeta, 3)}`);
    }
    if (ratioSMeta !== undefined && ratioBMeta !== undefined) {
      lines.push(`趋势/带宽：辛 ${formatNumber(ratioSMeta, 3)} / 对照 ${formatNumber(ratioBMeta, 3)}（≈1 才是线性漂移）`);
    }
    if (growthSMeta !== undefined && growthBMeta !== undefined) {
      lines.push(`后半/前半 sup 比：辛 ${formatNumber(growthSMeta, 3)} / 对照 ${formatNumber(growthBMeta, 3)}`);
    }
    if (shortSMeta !== undefined && shortBMeta !== undefined) {
      lines.push(
        `短期自收敛误差` +
          (shortWindow !== undefined ? `（窗口 ${formatNumber(shortWindow, 3)}）` : "") +
          `：辛 ${formatNumber(shortSMeta, 3)} / 对照 ${formatNumber(shortBMeta, 3)}`,
      );
      lines.push("短期准 ≠ 长期守规矩");
    }
    if (blowB) {
      lines.push(
        `对照在 t≈${tBlowB !== undefined ? formatNumber(tBlowB, 4) : "?"} 停止（掩码 ${nMaskB !== undefined ? Math.round(nMaskB) : "?"} 点）`,
      );
    }
    if (blowS) {
      lines.push(
        `辛方法在 t≈${tBlowS !== undefined ? formatNumber(tBlowS, 4) : "?"} 停止（掩码 ${nMasks !== undefined ? Math.round(nMasks) : "?"} 点）`,
      );
    }
    if (!sameLen) lines.push("横轴是采样下标：帧的 t 数组只覆盖辛方法那一段");
    if (status !== undefined) lines.push(`状态：${status}`);
    scene.texts.push(...textBlock(L.rightX, textTop, L.floor, lines, pal.muted, unitBudget(L.rightW), "top"));
  } else {
    degraded.push("窄画布（单列）：右列说明未排下，关键数字见角标与 pitfalls 面板");
  }

  // —— 角标 ——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(true), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(badge("autoscale", "副轴：对数自适应（地板 1e-16）", 8, 26), SLOT.aux, width),
  );
  if (h !== undefined) {
    scene.badges.push(
      placeBadge(criticalBadge("h", h, criticalOf(frame, "h", H_CRITICAL), { beyond: "above" }), SLOT.critical, width),
    );
  }
  const statusBits: string[] = [];
  if (bandSMeta !== undefined) statusBits.push(`带宽辛 ${formatNumber(bandSMeta, 3)}`);
  if (growthBMeta !== undefined) statusBits.push(`对照增长比 ${formatNumber(growthBMeta, 3)}`);
  if (blowB) statusBits.push(`停 t=${tBlowB !== undefined ? formatNumber(tBlowB, 3) : "?"}/掩 ${nMaskB !== undefined ? Math.round(nMaskB) : "?"}`);
  else if (endBMeta !== undefined) statusBits.push(`对照末端 ${formatNumber(endBMeta, 3)}`);
  if (status !== undefined) statusBits.push(status);
  scene.badges.push(placeBadge(badge("status", statusBits.join("；") || "帧未给出可判读的状态", 8, 8), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  let nBad = 0;
  for (const arr of [tArr, errS, errB, maskS, maskB, bandHi, bandLo, driftS, driftB]) {
    if (arr !== undefined) nBad += countNonFinite(arr);
  }
  applyOverflow(scene, nBad, { slot: SLOT.overflow, width });
  return scene;
};
