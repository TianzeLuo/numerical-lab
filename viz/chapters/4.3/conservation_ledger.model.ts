/**
 * `conservation_ledger`（4.3 章）的 model：三本账的质量、账本残差与激波位置。
 *
 * 这一页的核心读法是**两条独立的证据**：
 * 1. 左图的总质量**该不该变**由边界条件决定（出流边界下质量从 0.5 涨到 0.8 是正常的）；
 * 2. 右图的账本残差 `|M(t)-M(0)+(out-in)|` 才判格式守不守恒：
 *    两条守恒支路压在 1e-16、非守恒支路在 3e-1。
 *
 * 底部是激波位置-时间：守恒格式的斜率收敛到 RH 值 0.5，非守恒的波前原地不动。
 * 三张图的窗口都是固定的（manifest 声明 `autoscaled=false`），拖动时间只让曲线变长。
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
  maskIndices,
  pick,
  textBlock,
  unitBudget,
} from "../_common/panel.js";
import { leftStack, SLOT, widenRightColumn, type Rect } from "./common.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "4.3";
const VIZ = "conservation_ledger";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "守恒账本：机器零 vs 漏水";
const T_MAX = Number(SPEC?.time?.tMax ?? 0.6);
const DEFECT_FLOOR = 1e-18;
const DEFECT_CEIL = 1e1;

export const conservationLedgerModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  const timeGod = requireArray(frame, "time_godunov");
  const defectGod = requireArray(frame, "defect_godunov");
  const maskGod = requireArray(frame, "defect_floor_mask_godunov");
  const timeMuscl = requireArray(frame, "time_muscl");
  const defectMuscl = requireArray(frame, "defect_muscl");
  const maskMuscl = requireArray(frame, "defect_floor_mask_muscl");
  const timeAdv = requireArray(frame, "time_advective");
  const defectAdv = requireArray(frame, "defect_advective");
  const maskAdv = requireArray(frame, "defect_floor_mask_advective");
  const massTime = requireArray(frame, "mass_time");
  const massGod = requireArray(frame, "mass_godunov");
  const massMuscl = requireArray(frame, "mass_muscl");
  const massAdv = requireArray(frame, "mass_advective");
  const inflow = requireArray(frame, "inflow_cum");
  const outflow = requireArray(frame, "outflow_cum");
  const shockTGod = requireArray(frame, "shock_t_godunov");
  const shockXGod = requireArray(frame, "shock_x_godunov");
  const shockMaskGod = requireArray(frame, "shock_mask_godunov");
  const shockTAdv = requireArray(frame, "shock_t_advective");
  const shockXAdv = requireArray(frame, "shock_x_advective");
  const shockMaskAdv = requireArray(frame, "shock_mask_advective");
  const rhT = requireArray(frame, "rh_line_t");
  const rhX = requireArray(frame, "rh_line_x");
  const pitfalls = requirePitfalls(frame);

  for (const [name, a, b] of [
    ["mass_godunov", massTime, massGod],
    ["mass_muscl", massTime, massMuscl],
    ["mass_advective", massTime, massAdv],
    ["inflow_cum", massTime, inflow],
    ["outflow_cum", massTime, outflow],
  ] as const) {
    if (a.length !== b.length) throw new Error(`帧 ${CHAPTER}/${VIZ} 的 ${name} 与 mass_time 长度不一致`);
  }

  const degraded: string[] = [];
  const t = metaNumber(frame, "t") ?? opts.t;
  const mass0 = metaNumber(frame, "mass0");
  const massEndGod = metaNumber(frame, "mass_end_godunov");
  const massEndMuscl = metaNumber(frame, "mass_end_muscl");
  const massEndAdv = metaNumber(frame, "mass_end_advective");
  const defectGodMeta = metaNumber(frame, "defect_godunov");
  const defectMusclMeta = metaNumber(frame, "defect_muscl");
  const defectAdvMeta = metaNumber(frame, "defect_advective");
  const relDefectGod = metaNumber(frame, "rel_defect_godunov");
  const relDefectAdv = metaNumber(frame, "rel_defect_advective");
  const leakTotal = metaNumber(frame, "leak_total");
  const inflowCum = metaNumber(frame, "inflow_cum");
  const outflowCum = metaNumber(frame, "outflow_cum");
  const rhSpeed = metaNumber(frame, "rh_speed");
  const speedGod = metaNumber(frame, "speed_godunov");
  const speedAdv = metaNumber(frame, "speed_advective");
  const entropyGod = metaNumber(frame, "entropy_production_godunov");
  const entropyAdv = metaNumber(frame, "entropy_production_advective");
  const bc = stringOf(frame, "bc");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  const L = widenRightColumn(columnsLayout(scene, width, height, { topShare: 0.56, auxH: 128, gap: 60 }), width);
  const [massRect, shockRect] = leftStack(L, 2);
  const defectRect: Rect = [L.rightX, L.topY, L.rightW, 120];
  const textTop = defectRect[1] + defectRect[3] + 28 + 6;

  const massAxis = makeAxis("mass", massRect, {
    xlim: [0, T_MAX],
    ylim: [0, 1.05],
    title: axisTitleOf(SPEC?.axes, "mass", "总质量 M(t)"),
    ylabel: axisYlabelOf(SPEC?.axes, "mass", "M = sum u_j dx"),
    xlabel: "",
    autoscaled: false,
  });
  const shockAxis = makeAxis("shock", shockRect, {
    xlim: [0, T_MAX],
    ylim: [-0.05, 1.05],
    title: axisTitleOf(SPEC?.axes, "shock", "激波位置-时间"),
    ylabel: axisYlabelOf(SPEC?.axes, "shock", "x_shock(t)"),
    xlabel: "t（激波位置作纵轴）",
    autoscaled: false,
  });
  const defectAxis = makeAxis("defect", defectRect, {
    xlim: [0, T_MAX],
    ylim: [DEFECT_FLOOR, DEFECT_CEIL],
    yscale: "log",
    title: axisTitleOf(SPEC?.axes, "defect", "账本残差（对数）"),
    ylabel: axisYlabelOf(SPEC?.axes, "defect", "|M-M0+(out-in)|"),
    xlabel: "t",
    autoscaled: false,
  });
  scene.axes.push(massAxis, shockAxis, defectAxis);

  // —— 质量：三条曲线 + M(0) 参考线 ——
  scene.layers.push(
    lineLayer("mass", series(massTime, massGod, pal.curves[0], { width: 2 })),
    lineLayer("mass", series(massTime, massMuscl, pal.curves[1], { width: 1.6 })),
    lineLayer("mass", series(massTime, massAdv, pal.warn, { width: 1.8, dash: [5, 3] })),
  );
  if (mass0 !== undefined) {
    scene.layers.push(
      lineLayer("mass", series([0, T_MAX], [mass0, mass0], pal.reference, { width: 1, dash: [3, 4] })),
    );
  }

  // —— 账本残差：各支路自己的时间轴 ——
  scene.layers.push(
    lineLayer("defect", [
      ...segmentedSeries(timeGod, defectGod, pal.curves[0], { width: 1.8, accept: (v) => v > 0 }),
      ...segmentedSeries(timeMuscl, defectMuscl, pal.curves[1], { width: 1.5, accept: (v) => v > 0 }),
      ...segmentedSeries(timeAdv, defectAdv, pal.warn, { width: 1.8, dash: [5, 3], accept: (v) => v > 0 }),
    ]),
  );
  scene.layers.push(
    lineLayer("defect", series([0, T_MAX], [1e-17, 1e-17], pal.reference, { width: 1, dash: [3, 4] })),
  );
  for (const [time, defect, color] of [
    [timeGod, defectGod, pal.curves[0]],
    [timeMuscl, defectMuscl, pal.curves[1]],
    [timeAdv, defectAdv, pal.warn],
  ] as const) {
    if (time.length === 1) {
      scene.layers.push(scatterLayer("defect", [time[0] as number], [defect[0] as number], { r: 2.8, color }));
    }
  }

  // —— 激波位置：掩码选点 + RH 参考线 + 当前时刻切线 ——
  const drawTrack = (
    tArr: ArrayLike<number>,
    xArr: ArrayLike<number>,
    mask: ArrayLike<number>,
    color: string,
    width2: number,
    dash: number[],
    label: string,
  ): void => {
    const idx = maskIndices(mask);
    if (idx.length === 0) return;
    if (idx.length === 1) {
      scene.layers.push(
        scatterLayer("shock", [tArr[idx[0] as number] as number], [xArr[idx[0] as number] as number], {
          r: 2.8,
          color,
        }),
      );
      return;
    }
    const pts = pick(tArr, xArr, idx);
    scene.layers.push(lineLayer("shock", series(pts.x, pts.y, color, { width: width2, dash, label })));
  };
  drawTrack(shockTGod, shockXGod, shockMaskGod, pal.curves[0], 2, [], "");
  drawTrack(shockTAdv, shockXAdv, shockMaskAdv, pal.warn, 1.8, [5, 3], "");
  scene.layers.push(
    lineLayer("shock", series(rhT, rhX, pal.reference, { width: 1.3, dash: [4, 4] })),
    lineLayer("shock", series([t, t], [-0.05, 1.05], pal.marker, { width: 1.1, label: "当前 t" })),
  );

  // —— 图例（只留在左列面板内；右列账本轴图例并入说明末行） ——
  scene.texts.push(
    ...legendBlock(
      L.left + 6,
      massRect[1] + massRect[3] - 21,
      L.leftW,
      "蓝=Godunov 绿=MUSCL 红虚=advective 灰点=M(0)",
      pal.muted,
    ),
    ...legendBlock(
      L.left + 6,
      shockRect[1] + shockRect[3] - 21,
      L.leftW,
      "蓝=守恒波前　红虚=非守恒　灰虚=RH　红实=t",
      pal.muted,
    ),
  );

  // —— 右列文字 ——
  if (L.columns && textTop <= L.floor - 3 * 15) {
    const head = paramSummary(frame, opts, [
      { name: "n", label: "n" },
      { name: "cfl", label: "cfl" },
    ]);
    degraded.push(...head.degraded);
    const lines: string[] = [];
    lines.push(`${head.text}　bc=${bc ?? "?"}　t=${formatNumber(t, 4)}`);
    if (mass0 !== undefined) {
      lines.push(
        `M(0)=${formatNumber(mass0, 6)}　流入 ${formatNumber(inflowCum ?? 0, 4)}　流出 ${formatNumber(outflowCum ?? 0, 4)}`,
      );
    }
    if (massEndGod !== undefined && massEndMuscl !== undefined && massEndAdv !== undefined) {
      lines.push(
        `M_end G/M/A = ${formatNumber(massEndGod, 5)} / ${formatNumber(massEndMuscl, 5)} / ${formatNumber(massEndAdv, 5)}`,
      );
    }
    if (defectGodMeta !== undefined && defectMusclMeta !== undefined && defectAdvMeta !== undefined) {
      lines.push(
        `账本残差 G/M/A = ${formatNumber(defectGodMeta, 2)} / ${formatNumber(defectMusclMeta, 2)} / ${formatNumber(defectAdvMeta, 2)}`,
      );
    }
    if (leakTotal !== undefined) {
      lines.push(
        `非守恒漏水 ${formatNumber(leakTotal, 4)}　相对残差 G ${formatNumber(relDefectGod ?? 0, 2)} / A ${formatNumber(relDefectAdv ?? 0, 2)}`,
      );
    }
    lines.push(
      `贴地板步数 G/M/A = ${countMask(maskGod)} / ${countMask(maskMuscl)} / ${countMask(maskAdv)}`,
    );
    if (rhSpeed !== undefined) {
      lines.push(
        `激波速度 RH ${formatNumber(rhSpeed, 3)}` +
          (speedGod !== undefined ? `　G ${formatNumber(speedGod, 6)}` : "") +
          (speedAdv !== undefined ? `　A ${formatNumber(speedAdv, 2)}` : ""),
      );
    }
    if (entropyGod !== undefined && entropyAdv !== undefined) {
      lines.push(`熵产 G ${formatNumber(entropyGod, 3)}　A ${formatNumber(entropyAdv, 3)}`);
    }
    lines.push("红虚=非守恒；灰虚=RH 线；红实=当前 t；蓝/绿=两条守恒支路。");
    scene.texts.push(...textBlock(L.rightX, textTop, L.floor, lines, pal.muted, unitBudget(L.rightW), "top"));
  } else {
    degraded.push("画布过窄：右列说明未排下，关键数字见角标与 pitfalls 面板");
  }

  // —— 角标 ——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  scene.badges.push(
    placeBadge(badge("autoscale", "坐标轴：三轴全部固定窗口（账本对数轴 1e-18–10）", 8, 26), SLOT.axisMode, width),
  );
  const status = [
    `bc=${bc ?? "?"}`,
    `t=${formatNumber(t, 3)}`,
    defectGodMeta !== undefined ? `G 残差 ${formatNumber(defectGodMeta, 2)}` : "",
    speedGod !== undefined ? `s=${formatNumber(speedGod, 4)}` : "",
  ].filter((s) => s !== "");
  scene.badges.push(placeBadge(badge("status", status.join("；"), 8, 8), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  const nBad =
    countNonFinite(massGod) +
    countNonFinite(massMuscl) +
    countNonFinite(massAdv) +
    countNonFinite(defectGod) +
    countNonFinite(defectMuscl) +
    countNonFinite(defectAdv) +
    countNonFinite(shockXGod) +
    countNonFinite(shockXAdv);
  applyOverflow(scene, nBad, { slot: SLOT.overflow, width });
  return scene;
};
