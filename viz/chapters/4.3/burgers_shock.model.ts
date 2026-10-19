/**
 * `burgers_shock`（4.3 章）的 model：解、逐单元通量进出、逐点误差、账本残差。
 *
 * 这张图要同时挡住三个误读：
 * 1. **纯迎风在跨声速稀疏波上是错解，不是"黏性大了点"**：解会冻成驻定间断，
 *    而误差曲线可以很小。帧里的 `entropy_production`、`max_jump` 与右列文字
 *    把"误差"和"物理正确"分开报。
 * 2. **参照物可能失效**：sine 破散后（`exact_valid=0`）不画精确解、也不画误差曲线——
 *    拿多值的特征回溯解当参照，会得到漂亮的假收敛。
 * 3. **通量窗口与账本**：`flux_net=-(F_out-F_in)` 的柱状图与
 *    `residual_mass`（对数轴）是守恒性的两条独立证据；贴 1e-17 地板表示
 *    "低于地板"，不是"残差恰好 1e-17"。
 *
 * 轴口径与 manifest 一致：解/通量/误差三轴固定窗口（同一参数下拖动 t 不缩放），
 * 账本残差轴是本章额外加的（`requiredAux=residual`），自适应并在角标里说明。
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
  barsLayer,
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
  countMask,
  firstFinite,
  legendBlock,
  logRange,
  positiveMinMax,
  textBlock,
  unitBudget,
} from "../_common/panel.js";
import { leftStack, rightStack, SLOT, widenRightColumn } from "./common.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "4.3";
const VIZ = "burgers_shock";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "Burgers 激波形成与单元通量进出";
const CFL_CRITICAL = SPEC?.params?.find((p) => p.name === "cfl")?.critical ?? [1.0];
const FLOOR = 1e-17;
const U_WINDOW: [number, number] = [-1.2, 1.33];
const FLUX_WINDOW: [number, number] = [-0.75, 0.75];
const ERROR_HI = 3;

export const burgersShockModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  const cellX = requireArray(frame, "cell_x");
  const uNum = requireArray(frame, "u");
  const uClipped = requireArray(frame, "u_clipped");
  const uExact = requireArray(frame, "u_exact");
  const exactValid = requireArray(frame, "exact_valid");
  const exactTClamped = requireArray(frame, "exact_t_clamped");
  const fluxIn = requireArray(frame, "flux_in");
  const fluxOut = requireArray(frame, "flux_out");
  const fluxNet = requireArray(frame, "flux_net");
  const fluxClipped = requireArray(frame, "flux_clipped");
  const error = requireArray(frame, "error_abs");
  const errorMask = requireArray(frame, "error_floor_mask");
  const residualTime = requireArray(frame, "residual_time");
  const residualMass = requireArray(frame, "residual_mass");
  const residualMask = requireArray(frame, "residual_floor_mask");
  const pitfalls = requirePitfalls(frame);

  const n = Math.min(
    cellX.length,
    uNum.length,
    uExact.length,
    exactValid.length,
    fluxIn.length,
    fluxOut.length,
    fluxNet.length,
    error.length,
  );
  if (cellX.length < 2) {
    throw new Error(`帧 ${CHAPTER}/${VIZ} 的单元数 ${cellX.length} < 2：无法画通量差`);
  }
  for (const [name, arr] of [
    ["u", uNum],
    ["u_clipped", uClipped],
    ["u_exact", uExact],
    ["exact_valid", exactValid],
    ["flux_in", fluxIn],
    ["flux_out", fluxOut],
    ["flux_net", fluxNet],
    ["flux_clipped", fluxClipped],
    ["error_abs", error],
    ["error_floor_mask", errorMask],
  ] as const) {
    if (arr.length !== n) {
      throw new Error(`帧 ${CHAPTER}/${VIZ} 的 ${name} 长度 ${arr.length} ≠ 单元数 ${n}`);
    }
  }

  const degraded: string[] = [];
  const t = metaNumber(frame, "t_actual") ?? metaNumber(frame, "t") ?? opts.t;
  const tRequested = metaNumber(frame, "t") ?? opts.t;
  const dx = metaNumber(frame, "dx");
  const cfl = metaNumber(frame, "cfl");
  const scheme = stringOf(frame, "scheme");
  const schemeCn = stringOf(frame, "scheme_cn");
  const limiter = stringOf(frame, "limiter_effective");
  const limiterRequested = stringOf(frame, "limiter_requested");
  const limiterNote = stringOf(frame, "limiter_note");
  const ic = stringOf(frame, "ic");
  const bc = stringOf(frame, "bc");
  const readingNote = stringOf(frame, "reading_note");

  const rhSpeed = metaNumber(frame, "rh_speed");
  const speedMeasured = metaNumber(frame, "speed_measured");
  const speedRelErr = metaNumber(frame, "speed_rel_err");
  const speedR2 = metaNumber(frame, "speed_fit_r2");
  const mass0 = metaNumber(frame, "mass0");
  const massEnd = metaNumber(frame, "mass_end");
  const massIn = metaNumber(frame, "mass_in_cum");
  const massOut = metaNumber(frame, "mass_out_cum");
  const maxDefect = metaNumber(frame, "max_defect");
  const relDefect = metaNumber(frame, "rel_defect");
  const entropyProduction = metaNumber(frame, "entropy_production");
  const tv0 = metaNumber(frame, "tv0");
  const tvEnd = metaNumber(frame, "tv_end");
  const maxU = metaNumber(frame, "max_u");
  const minU = metaNumber(frame, "min_u");
  const maxJump = metaNumber(frame, "max_jump");
  const errorL1 = metaNumber(frame, "error_l1");
  const errorMax = metaNumber(frame, "error_max");
  const exactBreak = metaNumber(frame, "exact_t_break");
  const exactOk = (metaNumber(frame, "exact_valid") ?? 0) > 0.5;
  const nClipped = metaNumber(frame, "n_clipped");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  const L = widenRightColumn(columnsLayout(scene, width, height, { topShare: 0.58, auxH: 104, gap: 60 }), width);
  const [solRect, fluxRect] = leftStack(L, 2);
  const right = rightStack(L);

  const solAxis = makeAxis("solution", solRect, {
    xlim: [0, 1],
    ylim: U_WINDOW,
    title: axisTitleOf(SPEC?.axes, "solution", "解与精确解"),
    ylabel: axisYlabelOf(SPEC?.axes, "solution", "u(x,t)"),
    xlabel: "",
    autoscaled: false,
  });
  const fluxAxis = makeAxis("flux", fluxRect, {
    xlim: [0, 1],
    ylim: FLUX_WINDOW,
    title: axisTitleOf(SPEC?.axes, "flux", "逐单元界面通量"),
    ylabel: axisYlabelOf(SPEC?.axes, "flux", "F_(j±1/2)"),
    xlabel: "x（域长 L=1，时间轴见右列）",
    autoscaled: false,
  });
  const errAxis = makeAxis("error", right.error, {
    xlim: [0, 1],
    ylim: [FLOOR, ERROR_HI],
    yscale: "log",
    title: axisTitleOf(SPEC?.axes, "error", "逐点误差（对数）"),
    ylabel: axisYlabelOf(SPEC?.axes, "error", "|u - u_exact|"),
    xlabel: "x（单元中心）",
    autoscaled: false,
  });
  const residY = logRange(FLOOR, Math.max(1e-13, (positiveMinMax(residualMass)?.[1] ?? FLOOR) * 4));
  const residAxis = makeAxis("residual", right.residual, {
    xlim: [0, Math.max(0.6, lastFinite(residualTime) ?? 0)],
    ylim: residY,
    yscale: "log",
    title: "质量账本残差 |M-M0+out-in|（对数轴）",
    ylabel: "|delta M|",
    xlabel: "t",
    autoscaled: true,
  });
  scene.axes.push(solAxis, fluxAxis, errAxis, residAxis);

  const drawExact = countMask(exactValid) > 0 || (firstFinite(exactTClamped) ?? 0) > 0.5;
  scene.layers.push(
    lineLayer("solution", series([0, 1], [0, 0], pal.reference, { width: 1, dash: [4, 4], label: "u=0" })),
    lineLayer("solution", series(cellX, uNum, pal.curves[0], { width: 2.2 })),
  );
  if (drawExact) {
    scene.layers.push(
      lineLayer(
        "solution",
        series(cellX, uExact, pal.exact, { width: 1.4 }),
      ),
    );
  }

  // —— 通量：两条界面通量 + 净流入柱 ——
  const barW = Math.max(1e-4, (dx ?? 1 / n) * 0.9);
  scene.layers.push(
    barsLayer("flux", cellX, fluxNet, { width: barW, color: pal.curves[0], label: "净流入" }),
    lineLayer("flux", series([0, 1], [0, 0], pal.reference, { width: 1, dash: [4, 4] })),
    lineLayer("flux", series(cellX, fluxIn, pal.exact, { width: 1.2 })),
    lineLayer("flux", series(cellX, fluxOut, pal.curves[1], { width: 1.2 })),
  );

  // —— 误差：只在参照有效且未触地板处画；无效时不画整条 ——
  const nErrorValid = countMask(errorMask) < errorMask.length;
  const errSegments = segmentedSeries(cellX, error, pal.residual, {
    width: 1.8,
    accept: (_v, i) => (errorMask[i] ?? 0) < 0.5,
  });
  if (errSegments.length > 0) scene.layers.push(lineLayer("error", errSegments));
  scene.layers.push(
    lineLayer("error", series([0, 1], [FLOOR, FLOOR], pal.reference, { width: 1, dash: [3, 4] })),
  );
  const residualSegments = segmentedSeries(residualTime, residualMass, pal.curves[2], {
    width: 1.6,
    accept: (v) => v > 0,
  });
  if (residualSegments.length > 0) scene.layers.push(lineLayer("residual", residualSegments));
  if (residualTime.length === 1) {
    scene.layers.push(scatterLayer("residual", [residualTime[0]], [residualMass[0]], { r: 2.8, color: pal.curves[2] }));
  }

  // —— 面板图例 ——
  scene.texts.push(
    ...legendBlock(
      L.left + 6,
      solRect[1] + solRect[3] - 21,
      L.leftW,
      drawExact ? "蓝=数值解　橙=精确解　灰虚=u=0" : "蓝=数值解　（精确解参照无效：不画）　灰虚=u=0",
      pal.muted,
    ),
    ...legendBlock(
      L.left + 6,
      fluxRect[1] + fluxRect[3] - 21,
      L.leftW,
      "橙=左界面　绿=右界面　蓝柱=净流入 -(F_out-F_in)",
      pal.muted,
    ),
    ...legendBlock(
      right.error[0] + 6,
      right.error[1] + 6,
      right.error[2],
      nErrorValid ? "紫=逐点误差；贴 1e-17 地板/参照无效的点不画" : "参照无效：误差曲线整条不画",
      nErrorValid ? pal.muted : pal.warn,
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
      { name: "cfl", label: "cfl" },
    ]);
    degraded.push(...head.degraded);
    const lines: string[] = [];
    lines.push(`${scheme ?? "?"} / ${limiter ?? "?"} / ${ic ?? "?"} / ${bc ?? "?"}`);
    if (limiterRequested !== undefined && limiterRequested !== limiter) {
      lines.push(limiterNote ?? `limiter=${limiterRequested} 被格式覆盖为 ${limiter}`);
    }
    lines.push(`${head.text}　t=${formatNumber(t, 4)}（请求 ${formatNumber(tRequested, 4)}）`);
    if (speedMeasured !== undefined && rhSpeed !== undefined) {
      lines.push(
        `激波速度 ${formatNumber(speedMeasured, 6)} vs RH ${formatNumber(rhSpeed, 3)}` +
          (speedRelErr !== undefined ? `（相对 ${formatNumber(speedRelErr, 2)}，r² ${formatNumber(speedR2 ?? 0, 4)}）` : ""),
      );
    } else if (rhSpeed !== undefined) {
      lines.push(`RH 参考速度 ${formatNumber(rhSpeed, 4)}（本帧未给出拟合读数）`);
    }
    if (mass0 !== undefined && massEnd !== undefined) {
      lines.push(
        `质量 ${formatNumber(mass0, 5)} → ${formatNumber(massEnd, 5)}` +
          (massIn !== undefined && massOut !== undefined
            ? `（流入 ${formatNumber(massIn, 3)}，流出 ${formatNumber(massOut, 3)}）`
            : ""),
      );
    }
    if (maxDefect !== undefined) {
      lines.push(
        `账本残差 ${formatNumber(maxDefect, 3)}` +
          (relDefect !== undefined ? `（相对 ${formatNumber(relDefect, 2)}）` : ""),
      );
    }
    if (errorL1 !== undefined && errorMax !== undefined && exactOk) {
      lines.push(`误差 L1 ${formatNumber(errorL1, 3)}　max ${formatNumber(errorMax, 3)}`);
    } else if (!exactOk) {
      lines.push(
        `参照无效${exactBreak !== undefined ? `（t_b=${formatNumber(exactBreak, 5)}）` : ""}：误差曲线不画`,
      );
    }
    if (entropyProduction !== undefined) {
      lines.push(
        `熵产 ${formatNumber(entropyProduction, 3)}` + (entropyProduction > 0 ? "（正：违反熵条件）" : ""),
      );
    }
    if (tv0 !== undefined && tvEnd !== undefined) {
      lines.push(
        `TV ${formatNumber(tv0, 5)} → ${formatNumber(tvEnd, 5)}` +
          (maxJump !== undefined ? `　maxJump ${formatNumber(maxJump, 4)}` : "") +
          (minU !== undefined && maxU !== undefined ? `　u∈[${formatNumber(minU, 3)}, ${formatNumber(maxU, 3)}]` : ""),
      );
    }
    if (nClipped !== undefined && nClipped > 0) lines.push(`显示裁剪 ${Math.round(nClipped)} 个单元（见 overflow）`);
    const nFluxClipped = countMask(fluxClipped);
    if (nFluxClipped > 0) lines.push(`${nFluxClipped} 个界面通量被截到 ±${FLUX_WINDOW[1]}（flux_clipped=1）`);
    if (readingNote !== undefined) lines.push(`读图：${readingNote}`);
    scene.texts.push(...textBlock(L.rightX, right.textTop, L.floor, lines, pal.muted, unitBudget(L.rightW), "top"));
  } else {
    degraded.push("画布过窄：右列说明未排下，关键数字见角标与 pitfalls 面板");
  }

  // —— 角标 ——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  scene.badges.push(
    placeBadge(
      badge("autoscale", "坐标轴：解/通量/误差固定窗口；账本残差轴自适应（对数）", 8, 26),
      SLOT.axisMode,
      width,
    ),
  );
  if (cfl !== undefined) scene.badges.push(placeBadge(criticalBadge("cfl", cfl, CFL_CRITICAL), SLOT.critical, width));
  const status = [
    schemeCn ?? scheme ?? "?",
    `t=${formatNumber(t, 3)}`,
    speedMeasured !== undefined ? `s=${formatNumber(speedMeasured, 5)}` : "",
    errorL1 !== undefined && exactOk ? `L1=${formatNumber(errorL1, 3)}` : "误差参照无效",
    maxJump !== undefined ? `maxJump=${formatNumber(maxJump, 3)}` : "",
  ].filter((s) => s !== "");
  scene.badges.push(placeBadge(badge("status", status.join("；"), 8, 8), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  const nBad =
    countNonFinite(uNum) +
    countNonFinite(uExact) +
    countNonFinite(fluxIn) +
    countNonFinite(fluxOut) +
    countNonFinite(fluxNet) +
    countNonFinite(error) +
    countNonFinite(residualMass);
  applyOverflow(scene, nBad, { slot: SLOT.overflow, width });
  return scene;
};

function lastFinite(a: ArrayLike<number>): number | undefined {
  for (let i = a.length - 1; i >= 0; i -= 1) if (Number.isFinite(a[i])) return a[i];
  return undefined;
}
