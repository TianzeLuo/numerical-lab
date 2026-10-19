/**
 * `riemann_fan`（4.3 章）的 model：Euler Riemann 扇的密度/速度/压强与误差。
 *
 * 三块物理面板共用 x 轴、上下对齐，才能一眼看出同一位置上的三种行为：
 * 跨激波**三条量都跳**，跨接触**只有密度跳**（p、u 连续），
 * 稀疏波扇面内三条量连续变化。五条波的位置由帧的 `wave_x/wave_kind` 给出，
 * 前端只负责按 kind 上色（激波红实、接触绿虚、稀疏边界灰点）。
 *
 * t=0 时自相似变量 x/t 未定义（帧把 t 抬到 1e-12 并置 `t_clamped=1`）：
 * 此时精确解是初始间断，**误差整条无效**，模型不画误差曲线。
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
  paramSummary,
  placeBadge,
  requireArray,
  requireDrawableLayers,
  requirePitfalls,
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
  countMask,
  legendBlock,
  logRange,
  positiveMinMax,
  textBlock,
  unitBudget,
} from "../_common/panel.js";
import { dedupeWaves, drawWaves, leftStack, rightStack, SLOT, widenRightColumn } from "./common.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "4.3";
const VIZ = "riemann_fan";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "Riemann 解扇：激波、接触间断与稀疏波";
const FLOOR = 1e-17;
const T_MAX = Number(SPEC?.time?.tMax ?? 0.2);

export const riemannFanModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  const x = requireArray(frame, "x");
  const rhoExact = requireArray(frame, "rho_exact");
  const uExact = requireArray(frame, "u_exact");
  const pExact = requireArray(frame, "p_exact");
  const rhoNum = requireArray(frame, "rho_num");
  const uNum = requireArray(frame, "u_num");
  const pNum = requireArray(frame, "p_num");
  const overflowMask = requireArray(frame, "overflow_mask");
  const exactValid = requireArray(frame, "exact_valid");
  const errRho = requireArray(frame, "error_rho");
  const errU = requireArray(frame, "error_u");
  const errP = requireArray(frame, "error_p");
  const errMask = requireArray(frame, "error_floor_mask");
  const waveX = requireArray(frame, "wave_x");
  const waveKind = requireArray(frame, "wave_kind");
  const residualTime = requireArray(frame, "residual_time");
  const residualMass = requireArray(frame, "residual_mass");
  const residualMask = requireArray(frame, "residual_floor_mask");
  const pitfalls = requirePitfalls(frame);

  const n = Math.min(
    x.length,
    rhoExact.length,
    uExact.length,
    pExact.length,
    rhoNum.length,
    uNum.length,
    pNum.length,
    errRho.length,
    errU.length,
    errP.length,
  );
  if (n < 2) throw new Error(`帧 ${CHAPTER}/${VIZ} 的采样点 ${n} < 2`);
  for (const [name, arr] of [
    ["rho_exact", rhoExact],
    ["u_exact", uExact],
    ["p_exact", pExact],
    ["rho_num", rhoNum],
    ["u_num", uNum],
    ["p_num", pNum],
    ["error_rho", errRho],
    ["error_u", errU],
    ["error_p", errP],
    ["error_floor_mask", errMask],
    ["overflow_mask", overflowMask],
    ["exact_valid", exactValid],
  ] as const) {
    if (arr.length !== n) throw new Error(`帧 ${CHAPTER}/${VIZ} 的 ${name} 长度 ${arr.length} ≠ n=${n}`);
  }

  const degraded: string[] = [];
  const t = metaNumber(frame, "t") ?? opts.t;
  const tRequested = metaNumber(frame, "t_requested") ?? opts.t;
  const tClamped = (metaNumber(frame, "t_clamped") ?? 0) > 0.5;
  const pStar = metaNumber(frame, "p_star");
  const uStar = metaNumber(frame, "u_star");
  const rhoStarL = metaNumber(frame, "rho_star_l");
  const rhoStarR = metaNumber(frame, "rho_star_r");
  const contactX = metaNumber(frame, "contact_x");
  const sLeft = metaNumber(frame, "s_left");
  const sRight = metaNumber(frame, "s_right");
  const vacuum = metaNumber(frame, "vacuum");
  const l1Rho = metaNumber(frame, "l1_rho");
  const l1U = metaNumber(frame, "l1_u");
  const l1P = metaNumber(frame, "l1_p");
  const minRho = metaNumber(frame, "min_rho");
  const minP = metaNumber(frame, "min_p");
  const maxDefect = metaNumber(frame, "max_defect");
  const entropyRatio = metaNumber(frame, "entropy_ratio_shock");
  const winRhoLo = metaNumber(frame, "window_rho_lo");
  const winRhoHi = metaNumber(frame, "window_rho_hi");
  const winPLo = metaNumber(frame, "window_p_lo");
  const winPHi = metaNumber(frame, "window_p_hi");
  const problem = stringOf(frame, "problem");
  const scheme = stringOf(frame, "scheme");
  const limiter = stringOf(frame, "limiter");
  const leftWave = stringOf(frame, "left_wave");
  const rightWave = stringOf(frame, "right_wave");
  const schemeNote = stringOf(frame, "scheme_note");
  const waveNote = stringOf(frame, "wave_note");
  const theoryNote = stringOf(frame, "theory_note");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  const L = widenRightColumn(columnsLayout(scene, width, height, { topShare: 0.56, auxH: 104, gap: 50 }), width);
  const [rhoRect, velRect, pRect] = leftStack(L, 3, 48);
  const right = rightStack(L);

  const rhoAxis = makeAxis("rho", rhoRect, {
    xlim: [0, 1],
    ylim: [winRhoLo ?? 0, winRhoHi ?? 1],
    title: axisTitleOf(SPEC?.axes, "rho", "密度"),
    ylabel: axisYlabelOf(SPEC?.axes, "rho", "rho"),
    xlabel: "",
    autoscaled: false,
  });
  const velAxis = makeAxis("velocity", velRect, {
    xlim: [0, 1],
    ylim: [-0.6, 1.0],
    title: axisTitleOf(SPEC?.axes, "velocity", "速度"),
    ylabel: axisYlabelOf(SPEC?.axes, "velocity", "u"),
    xlabel: "",
    autoscaled: false,
  });
  const pAxis = makeAxis("pressure", pRect, {
    xlim: [0, 1],
    ylim: [winPLo ?? 0, winPHi ?? 1],
    title: axisTitleOf(SPEC?.axes, "pressure", "压强"),
    ylabel: axisYlabelOf(SPEC?.axes, "pressure", "p"),
    xlabel: "x（三块图共享横轴）",
    autoscaled: false,
  });
  const errAxis = makeAxis("error", right.error, {
    xlim: [0, 1],
    ylim: [FLOOR, 1],
    yscale: "log",
    title: axisTitleOf(SPEC?.axes, "error", "三条误差（对数）"),
    ylabel: axisYlabelOf(SPEC?.axes, "error", "|num - exact|"),
    xlabel: "x",
    autoscaled: false,
  });
  const residAxis = makeAxis("residual", right.residual, {
    xlim: [0, Math.max(T_MAX, lastFinite(residualTime) ?? 0)],
    ylim: logRange(FLOOR, Math.max(1e-13, (positiveMinMax(residualMass)?.[1] ?? FLOOR) * 4)),
    yscale: "log",
    title: "Euler 质量账本残差（对数轴）",
    ylabel: "|delta M|",
    xlabel: "t",
    autoscaled: true,
  });
  scene.axes.push(rhoAxis, velAxis, pAxis, errAxis, residAxis);

  const waves = dedupeWaves(waveX, waveKind);
  drawWaves(scene, "rho", waves, rhoAxis.ylim, pal);
  drawWaves(scene, "velocity", waves, velAxis.ylim, pal);
  drawWaves(scene, "pressure", waves, pAxis.ylim, pal);

  scene.layers.push(
    lineLayer("rho", series(x, rhoExact, pal.exact, { width: 1.4, dash: tClamped ? [5, 4] : [] })),
    lineLayer("rho", series(x, rhoNum, pal.curves[0], { width: 2.1 })),
    lineLayer("velocity", series(x, uExact, pal.exact, { width: 1.3, dash: tClamped ? [5, 4] : [] })),
    lineLayer("velocity", series(x, uNum, pal.curves[0], { width: 2.1 })),
    lineLayer("pressure", series(x, pExact, pal.exact, { width: 1.3, dash: tClamped ? [5, 4] : [] })),
    lineLayer("pressure", series(x, pNum, pal.curves[0], { width: 2.1 })),
  );

  // —— 三条误差：参照无效（t=0）时整条不画 ——
  const exactOk = !tClamped && countMask(exactValid) > 0;
  if (exactOk) {
    scene.layers.push(
      lineLayer("error", [
        ...segmentedSeries(x, errRho, pal.curves[0], {
          width: 1.6,
          accept: (_v, i) => (errMask[i] ?? 0) < 0.5,
        }),
        ...segmentedSeries(x, errU, pal.curves[1], {
          width: 1.4,
          accept: (_v, i) => (errMask[i] ?? 0) < 0.5,
        }),
        ...segmentedSeries(x, errP, pal.exact, {
          width: 1.4,
          accept: (_v, i) => (errMask[i] ?? 0) < 0.5,
        }),
      ]),
    );
  }
  scene.layers.push(lineLayer("error", series([0, 1], [FLOOR, FLOOR], pal.reference, { width: 1, dash: [3, 4] })));
  const residualSegments = segmentedSeries(residualTime, residualMass, pal.curves[2], {
    width: 1.6,
    accept: (v) => v > 0,
  });
  if (residualSegments.length > 0) scene.layers.push(lineLayer("residual", residualSegments));
  if (residualTime.length === 1) {
    scene.layers.push(scatterLayer("residual", [residualTime[0] as number], [residualMass[0] as number], { r: 2.8, color: pal.curves[2] }));
  }

  // —— 图例 ——
  scene.texts.push(
    ...legendBlock(
      L.left + 6,
      rhoRect[1] + rhoRect[3] - 21,
      L.leftW,
      `橙=精确　蓝=数值`,
      pal.muted,
    ),
    ...legendBlock(
      L.left + 6,
      velRect[1] + velRect[3] - 21,
      L.leftW,
      "接触处 u、p 连续，rho 跳",
      pal.muted,
    ),
    ...legendBlock(
      L.left + 6,
      pRect[1] + pRect[3] - 21,
      L.leftW,
      "红=激波　绿=接触　灰点=稀疏边界",
      pal.muted,
    ),
    ...legendBlock(
      right.error[0] + 6,
      right.error[1] + 6,
      right.error[2],
      exactOk ? "蓝=rho 绿=u 橙=p（贴地板点断开）" : "t=0：自相似解未定义，误差整条不画",
      exactOk ? pal.muted : pal.warn,
    ),
    ...legendBlock(
      right.residual[0] + 6,
      right.residual[1] + 6,
      right.residual[2],
      `橙=质量账本残差（${countMask(residualMask)} 步贴 1e-17 地板）`,
      pal.muted,
    ),
  );

  // —— 右列文字 ——
  if (L.columns && right.textTop !== undefined) {
    const head = paramSummary(frame, opts, [
      { name: "n", label: "n" },
    ]);
    degraded.push(...head.degraded);
    const lines: string[] = [];
    lines.push(`${problem ?? "?"}　${scheme ?? "?"}+${limiter ?? "?"}　${head.text}`);
    lines.push(
      `t=${formatNumber(t, 4)}（请求 ${formatNumber(tRequested, 4)}${tClamped ? "，t=0 自相似解未定义" : ""}）`,
    );
    if (pStar !== undefined && uStar !== undefined) {
      lines.push(`星区 p*=${formatNumber(pStar, 6)}　u*=${formatNumber(uStar, 6)}`);
    }
    if (exactOk && l1Rho !== undefined && l1U !== undefined && l1P !== undefined) {
      lines.push(
        `L1 rho/u/p = ${formatNumber(l1Rho, 4)} / ${formatNumber(l1U, 4)} / ${formatNumber(l1P, 4)}`,
      );
    } else if (!exactOk) {
      lines.push("t=0：误差整条无效（不是“数值解精确到零”）");
    }
    if (minRho !== undefined && minP !== undefined) {
      lines.push(
        `min rho ${formatNumber(minRho, 4)}　min p ${formatNumber(minP, 4)}（必须为正）` +
          (maxDefect !== undefined ? `　账本 ${formatNumber(maxDefect, 2)}` : ""),
      );
    }
    if (entropyRatio !== undefined) lines.push(`激波两侧熵比 p/rho^gamma ${formatNumber(entropyRatio, 4)}（激波上增大）`);
    if (rhoStarL !== undefined && rhoStarR !== undefined) {
      lines.push(`rho*L/R=${formatNumber(rhoStarL, 5)} / ${formatNumber(rhoStarR, 5)}（接触处密度跳）`);
    }
    if (leftWave !== undefined || rightWave !== undefined) {
      lines.push(
        `波型 左 ${leftWave ?? "?"} / 右 ${rightWave ?? "?"}${vacuum === 1 ? "（真空）" : ""}` +
          (contactX !== undefined ? `　接触 x=${formatNumber(contactX, 3)}` : ""),
      );
    }
    if (sLeft !== undefined && sRight !== undefined) {
      lines.push(`HLLC 波速 s_L/s_R=${formatNumber(sLeft, 3)} / ${formatNumber(sRight, 3)}`);
    }
    const nClip = countMask(overflowMask);
    if (nClip > 0) lines.push(`${nClip} 个点被显示窗口截断（overflow_mask=1）`);
    if (schemeNote !== undefined) lines.push(schemeNote);
    if (waveNote !== undefined) lines.push(`波型：${waveNote}`);
    if (theoryNote !== undefined) lines.push(`理论：${theoryNote}`);
    scene.texts.push(...textBlock(L.rightX, right.textTop, L.floor, lines, pal.muted, unitBudget(L.rightW), "top"));
  } else {
    degraded.push("画布过窄：右列说明未排下，关键数字见角标与 pitfalls 面板");
  }

  // —— 角标 ——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  scene.badges.push(
    placeBadge(badge("autoscale", "坐标轴：rho/u/p/误差固定窗口；账本残差轴自适应（对数）", 8, 26), SLOT.axisMode, width),
  );
  const status = [
    `${problem ?? "?"} ${scheme ?? "?"}+${limiter ?? "?"}`,
    `t=${formatNumber(t, 3)}${tClamped ? "（t=0）" : ""}`,
    pStar !== undefined ? `p*=${formatNumber(pStar, 4)}` : "",
    exactOk && l1Rho !== undefined ? `L1 rho ${formatNumber(l1Rho, 3)}` : "误差无效",
    minRho !== undefined ? `min rho ${formatNumber(minRho, 3)}` : "",
  ].filter((s) => s !== "");
  scene.badges.push(placeBadge(badge("status", status.join("；"), 8, 8), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  const nBad =
    countNonFinite(rhoExact) +
    countNonFinite(uExact) +
    countNonFinite(pExact) +
    countNonFinite(rhoNum) +
    countNonFinite(uNum) +
    countNonFinite(pNum) +
    countNonFinite(errRho) +
    countNonFinite(errU) +
    countNonFinite(errP);
  applyOverflow(scene, nBad, { slot: SLOT.overflow, width });
  return scene;
};

function lastFinite(a: ArrayLike<number>): number | undefined {
  for (let i = a.length - 1; i >= 0; i -= 1) if (Number.isFinite(a[i])) return a[i];
  return undefined;
}
