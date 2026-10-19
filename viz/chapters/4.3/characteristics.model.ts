/**
 * `characteristics`（4.3 章）的 model：特征线平面 + 当前时刻剖面 + 回溯解误差。
 *
 * 三条纪律：
 * 1. **破散点来自帧的掩码**：`cross_mask=1` 的线对才有有限交点；发散线对的
 *    `cross_t` 是巨大的占位值（实测 4.5e14），模型只画掩码命中且
 *    `cross_t<=t_view` 的点。把占位值当数据画会得到"所有线交于 t=4.5e14"的假图。
 * 2. **t>t_b 的"精确解"不是精确解**：`exact_valid=0` 时回溯解多值，
 *    模型不画参考线、也不画误差曲线，只在图上写明原因。
 * 3. **误差地板不连线**：`error_floor_mask=1` 的点是 1e-17 占位，
 *    折线在这些点断开；贴底不等于"误差恰好 1e-17"。
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
  firstFinite,
  legendBlock,
  logRange,
  maskIndices,
  pick,
  positiveMinMax,
  textBlock,
  unitBudget,
} from "../_common/panel.js";
import { leftStack, rightStack, SLOT, widenRightColumn } from "./common.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "4.3";
const VIZ = "characteristics";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "特征线：相交即激波、发散即稀疏波";
const FLOOR = 1e-17;
const T_VIEW = Number(SPEC?.time?.tMax ?? 0.8);
const U_WINDOW: [number, number] = [-1.1, 1.1];

export const characteristicsModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  const lineX0 = requireArray(frame, "line_x0");
  const lineU0 = requireArray(frame, "line_u0");
  const lineXEnd = requireArray(frame, "line_x_end");
  const crossT = requireArray(frame, "cross_t");
  const crossX = requireArray(frame, "cross_x");
  const crossMask = requireArray(frame, "cross_mask");
  const breakLineT = requireArray(frame, "break_line_t");
  const sliceT = requireArray(frame, "slice_t");
  const profileX = requireArray(frame, "profile_x");
  const profileU = requireArray(frame, "profile_u");
  const profileExact = requireArray(frame, "profile_exact");
  const exactValid = requireArray(frame, "exact_valid");
  const error = requireArray(frame, "error_abs");
  const errorMask = requireArray(frame, "error_floor_mask");
  const residualTime = requireArray(frame, "residual_time");
  const residualMass = requireArray(frame, "residual_mass");
  const pitfalls = requirePitfalls(frame);

  const nLines = Math.min(lineX0.length, lineU0.length, lineXEnd.length);
  const nProfile = Math.min(profileX.length, profileU.length, profileExact.length, exactValid.length, error.length);
  if (nLines < 2) throw new Error(`帧 ${CHAPTER}/${VIZ} 的特征线数 ${nLines} < 2`);
  if (nProfile < 2) throw new Error(`帧 ${CHAPTER}/${VIZ} 的剖面点数 ${nProfile} < 2`);
  if (crossT.length !== nLines - 1 || crossX.length !== nLines - 1 || crossMask.length !== nLines - 1) {
    throw new Error(
      `帧 ${CHAPTER}/${VIZ} 的交点数组长度 ${crossT.length}/${crossX.length}/${crossMask.length} ` +
        `≠ n_lines-1=${nLines - 1}`,
    );
  }

  const degraded: string[] = [];
  const t = metaNumber(frame, "t") ?? opts.t;
  const amplitude = metaNumber(frame, "amplitude");
  const tBreakClosed = metaNumber(frame, "t_break_closed");
  const tBreakScan = metaNumber(frame, "t_break_scan");
  const tBreakLines = metaNumber(frame, "t_break_lines");
  const tBreakScanRel = metaNumber(frame, "t_break_scan_rel");
  const firstPair = metaNumber(frame, "first_pair");
  const xBreak = metaNumber(frame, "x_break");
  const nCrossing = metaNumber(frame, "n_crossing");
  const nDiverging = metaNumber(frame, "n_diverging");
  const exactOk = (metaNumber(frame, "exact_valid") ?? 0) > 0.5;
  const tvNow = metaNumber(frame, "tv_now");
  const maxU = metaNumber(frame, "max_u");
  const minU = metaNumber(frame, "min_u");
  const maxDefect = metaNumber(frame, "max_defect");
  const errorL1 = metaNumber(frame, "error_l1");
  const errorMax = metaNumber(frame, "error_max");
  const profileName = stringOf(frame, "profile");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  const L = widenRightColumn(columnsLayout(scene, width, height, { topShare: 0.56, auxH: 104, gap: 60 }), width);
  const [planeRect, profileRect] = leftStack(L, 2);
  const right = rightStack(L);

  const planeAxis = makeAxis("characteristics", planeRect, {
    xlim: [0, 1],
    ylim: [0, T_VIEW],
    title: axisTitleOf(SPEC?.axes, "characteristics", "特征线平面 (x,t)"),
    ylabel: axisYlabelOf(SPEC?.axes, "characteristics", "t"),
    xlabel: "",
    autoscaled: false,
  });
  const profileAxis = makeAxis("profile", profileRect, {
    xlim: [0, 1],
    ylim: U_WINDOW,
    title: axisTitleOf(SPEC?.axes, "profile", "当前时刻的解"),
    ylabel: axisYlabelOf(SPEC?.axes, "profile", "u(x,t)"),
    xlabel: "x（时间 t 见左图切线）",
    autoscaled: false,
  });
  const errAxis = makeAxis("error", right.error, {
    xlim: [0, 1],
    ylim: [FLOOR, 1],
    yscale: "log",
    title: axisTitleOf(SPEC?.axes, "error", "回溯解与数值解之差（对数）"),
    ylabel: axisYlabelOf(SPEC?.axes, "error", "|u - u_exact|"),
    xlabel: "x",
    autoscaled: false,
  });
  const residAxis = makeAxis("residual", right.residual, {
    xlim: [0, T_VIEW],
    ylim: logRange(FLOOR, Math.max(1e-13, (positiveMinMax(residualMass)?.[1] ?? FLOOR) * 4)),
    yscale: "log",
    title: "质量账本残差（对数轴）",
    ylabel: "|delta M|",
    xlabel: "t",
    autoscaled: true,
  });
  scene.axes.push(planeAxis, profileAxis, errAxis, residAxis);

  // —— (x,t) 平面：每条特征线按 u0 的符号上色 ——
  for (let i = 0; i < nLines; i += 1) {
    const u0 = lineU0[i] as number;
    if (!Number.isFinite(lineX0[i]) || !Number.isFinite(lineXEnd[i]) || !Number.isFinite(u0)) continue;
    const color = u0 > 1e-12 ? pal.curves[0] : u0 < -1e-12 ? pal.exact : pal.reference;
    scene.layers.push(
      lineLayer(
        "characteristics",
        series([lineX0[i] as number, lineXEnd[i] as number], [0, T_VIEW], color, {
          width: 0.9,
          alpha: 0.75,
          label: i === 0 ? (u0 > 0 ? "u0>0" : "u0<0") : undefined,
        }),
      ),
    );
  }
  const crossingIdx = maskIndices(crossMask).filter((i) => {
    const tc = crossT[i] as number;
    return Number.isFinite(tc) && tc >= 0 && tc <= T_VIEW;
  });
  if (crossingIdx.length > 0) {
    const pts = pick(crossX, crossT, crossingIdx);
    scene.layers.push(scatterLayer("characteristics", pts.x, pts.y, { r: 2.6, color: pal.marker }));
  }
  if (xBreak !== undefined && tBreakScan !== undefined && tBreakScan <= T_VIEW) {
    scene.layers.push(
      scatterLayer("characteristics", [xBreak], [tBreakScan], { r: 4.5, color: pal.warn }),
    );
  }
  const tBreakDraw = tBreakLines ?? firstFinite(breakLineT) ?? tBreakScan;
  if (tBreakDraw !== undefined && tBreakDraw > 0 && tBreakDraw <= T_VIEW) {
    scene.layers.push(
      lineLayer(
        "characteristics",
        series([0, 1], [tBreakDraw, tBreakDraw], pal.warn, { width: 1.1, dash: [5, 4] }),
      ),
    );
  }
  const sliceNow = firstFinite(sliceT) ?? t;
  if (Number.isFinite(sliceNow)) {
    scene.layers.push(
      lineLayer(
        "characteristics",
        series([0, 1], [sliceNow, sliceNow], pal.marker, { width: 1.5, label: "当前 t" }),
      ),
    );
  }

  // —— 当前时刻剖面：参照无效时不画精确解 ——
  const drawExact = countMask(exactValid) > 0;
  scene.layers.push(
    lineLayer("profile", series([0, 1], [0, 0], pal.reference, { width: 1, dash: [4, 4] })),
    lineLayer("profile", series(profileX, profileU, pal.curves[0], { width: 2.2 })),
  );
  if (drawExact) {
    scene.layers.push(
      lineLayer("profile", series(profileX, profileExact, pal.exact, { width: 1.4 })),
    );
  }

  // —— 误差与账本残差 ——
  if (drawExact) {
    const segments = segmentedSeries(profileX, error, pal.residual, {
      width: 1.8,
      accept: (_v, i) => (errorMask[i] ?? 0) < 0.5,
    });
    if (segments.length > 0) scene.layers.push(lineLayer("error", segments));
  }
  scene.layers.push(lineLayer("error", series([0, 1], [FLOOR, FLOOR], pal.reference, { width: 1, dash: [3, 4] })));
  const residualSegments = segmentedSeries(residualTime, residualMass, pal.curves[2], {
    width: 1.6,
    accept: (v) => v > 0,
  });
  if (residualSegments.length > 0) scene.layers.push(lineLayer("residual", residualSegments));
  if (residualTime.length === 1) {
    scene.layers.push(scatterLayer("residual", [residualTime[0]], [residualMass[0]], { r: 2.8, color: pal.curves[2] }));
  }

  // —— 图例（只留在剖面面板内；(x,t) 平面的配色改在右列说明里给出） ——
  scene.texts.push(
    ...legendBlock(
      L.left + 6,
      profileRect[1] + 6,
      L.leftW,
      drawExact ? "蓝=数值解　橙=回溯精确解　灰虚=u=0" : "蓝=数值解（回溯解多值：参照无效，不画）",
      drawExact ? pal.muted : pal.warn,
    ),
  );

  // —— 右列文字 ——
  if (L.columns && right.textTop !== undefined) {
    const head = paramSummary(frame, opts, [
      { name: "amplitude", label: "A" },
      { name: "n_lines", label: "线数" },
    ]);
    degraded.push(...head.degraded);
    const lines: string[] = [];
    lines.push(`${profileName ?? "?"} 初值　${head.text}　t=${formatNumber(t, 4)} / 视窗 ${formatNumber(T_VIEW, 3)}`);
    if (tBreakClosed !== undefined && tBreakScan !== undefined) {
      lines.push(
        `t_b 闭式/扫描 ${formatNumber(tBreakClosed, 8)} / ${formatNumber(tBreakScan, 8)}` +
          (tBreakScanRel !== undefined ? `（相对 ${formatNumber(tBreakScanRel, 2)}）` : ""),
      );
    }
    if (tBreakLines !== undefined) lines.push(`离散线首交 ${formatNumber(tBreakLines, 8)}（线间距的代价）`);
    if (drawExact && errorL1 !== undefined && errorMax !== undefined) {
      lines.push(`误差 L1 ${formatNumber(errorL1, 3)}　max ${formatNumber(errorMax, 3)}`);
    } else if (!exactOk) {
      lines.push(`t>t_b 回溯解多值${tBreakScan !== undefined ? `（t_b=${formatNumber(tBreakScan, 5)}）` : ""}：误差参照无效`);
    }
    if (nCrossing !== undefined && nDiverging !== undefined) {
      lines.push(
        `相交 ${Math.round(nCrossing)} / 发散 ${Math.round(nDiverging)} 对` +
          (firstPair !== undefined ? `　首对 #${Math.round(firstPair)}` : "") +
          (xBreak !== undefined ? ` @x=${formatNumber(xBreak, 3)}` : ""),
      );
    }
    if (tvNow !== undefined) {
      lines.push(
        `TV ${formatNumber(tvNow, 5)}` +
          (minU !== undefined && maxU !== undefined
            ? `　u∈[${formatNumber(minU, 3)}, ${formatNumber(maxU, 3)}]`
            : "") +
          (maxDefect !== undefined ? `　账本 ${formatNumber(maxDefect, 2)}` : "")
      );
    }
    lines.push("紫=误差 橙=账本；蓝/橙=u0 正负；红点=交点，红实=t。");
    scene.texts.push(...textBlock(L.rightX, right.textTop, L.floor, lines, pal.muted, unitBudget(L.rightW), "top"));
  } else {
    degraded.push("画布过窄：右列说明未排下，关键数字见角标与 pitfalls 面板");
  }

  // —— 角标 ——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  scene.badges.push(
    placeBadge(badge("autoscale", "坐标轴：平面/剖面/误差固定；残差轴自适应", 8, 26), SLOT.axisMode, width),
  );
  const status = [
    `${profileName ?? "?"} A=${formatNumber(amplitude ?? 0, 3)}`,
    `t=${formatNumber(t, 3)}/${formatNumber(T_VIEW, 3)}`,
    tBreakScan !== undefined ? `t_b=${formatNumber(tBreakScan, 4)}` : "",
    nCrossing !== undefined ? `相交 ${Math.round(nCrossing)} 对` : "",
  ].filter((s) => s !== "");
  scene.badges.push(placeBadge(badge("status", status.join("；"), 8, 8), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  const nBad =
    countNonFinite(lineX0) +
    countNonFinite(lineU0) +
    countNonFinite(profileU) +
    countNonFinite(profileExact) +
    countNonFinite(error) +
    countNonFinite(residualMass);
  applyOverflow(scene, nBad, { slot: SLOT.overflow, width });
  return scene;
};
