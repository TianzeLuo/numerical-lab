/**
 * `convergence_race`（4.6 章 viz 3）的 model：同一个平流 PDE，谱方法 vs 二阶/四阶差分。
 *
 * 判据不是「谁更小」，而是「每加倍误差比是不是常数」
 * -------------------------------------------------
 * * 主图 `error`：三条相对误差-自由度曲线，**同一个横轴口径**（谱=模态数、差分=格点数），
 *   双对数。两条差分的理论阶 N^-2 / N^-4 用虚线参考线画在右半段；谱方法不画幂律线
 *   （指数收敛没有固定阶），只报每加倍比值。
 * * 副图 `ratio`：固定窗口 0–1（manifest）。二阶 ≈0.25、四阶 ≈0.0625 是水平线；
 *   谱方法的比值一路变小。`meta.degenerate` 为真（t=0）时**整段比值无信息**，
 *   模型明确不画并说明；谱误差撞地板（`n_at_floor`）之后的比值也被剔除，
 *   因为那里只剩舍入噪声（`floor_note` 的口径）。窗口外的比值由溢出角标点名。
 * * 右图 `pointwise`：最大自由度上的逐点误差（y 对数、x 线性——x 含 0，不能上对数）。
 *
 * 帧 `time.mode="scrub"`：`scene.t` 取帧的 `meta.t_actual`（请求 t 可能被夹到 [0, tMax]）。
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
  requireSameLength,
  sceneShell,
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
} from "../../shared/src/scene.js";
import {
  SAFE_GAP,
  TEXT_ROW_H,
  axisTitleOf,
  axisYlabelOf,
  columnsLayout,
  selectValues,
  textBlock,
  unitBudget,
} from "../_common/panel.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "4.6";
const VIZ = "convergence_race";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "同一个 PDE：谱方法 vs 二阶/四阶差分";

const ERR_FLOOR_FALLBACK = 1e-17;
const T_MAX_FALLBACK = 1.0;

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

export const convergenceRaceModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需数组 ——
  const errN = requireArray(frame, "error_n");
  const errSp = requireArray(frame, "error_spectral");
  const err2 = requireArray(frame, "error_fd2");
  const err4 = requireArray(frame, "error_fd4");
  const ratioN = requireArray(frame, "ratio_n");
  const ratioSp = requireArray(frame, "ratio_spectral");
  const ratio2 = requireArray(frame, "ratio_fd2");
  const ratio4 = requireArray(frame, "ratio_fd4");
  const px = requireArray(frame, "point_x");
  const pSp = requireArray(frame, "point_error_spectral");
  const p2 = requireArray(frame, "point_error_fd2");
  const p4 = requireArray(frame, "point_error_fd4");
  for (const [name, arr] of [
    ["error_spectral", errSp],
    ["error_fd2", err2],
    ["error_fd4", err4],
  ] as const) {
    requireSameLength(errN, arr, `帧数组 error_n 与 ${name}`);
  }
  for (const [name, arr] of [
    ["ratio_spectral", ratioSp],
    ["ratio_fd2", ratio2],
    ["ratio_fd4", ratio4],
  ] as const) {
    requireSameLength(ratioN, arr, `帧数组 ratio_n 与 ${name}`);
  }
  for (const [name, arr] of [
    ["point_error_spectral", pSp],
    ["point_error_fd2", p2],
    ["point_error_fd4", p4],
  ] as const) {
    requireSameLength(px, arr, `帧数组 point_x 与 ${name}`);
  }
  const pitfalls = requirePitfalls(frame);

  // —— 可选 meta ——
  const degraded: string[] = [];
  const floorMeta = metaNumber(frame, "error_floor");
  if (floorMeta === undefined) degraded.push("帧 meta 缺 error_floor：误差地板用登记常量 1e-17");
  const errFloor = floorMeta !== undefined && floorMeta > 0 ? floorMeta : ERR_FLOOR_FALLBACK;
  const nAtFloor = metaNumber(frame, "n_at_floor");
  if (nAtFloor === undefined) {
    degraded.push("帧 meta 缺 n_at_floor：谱误差的每加倍比值不做地板段剔除（可能画出舍入噪声）");
  }
  const degenerateRaw = frame.meta["degenerate"];
  const degenerate = degenerateRaw === true || (metaNumber(frame, "degenerate") ?? 0) > 0.5;
  if (degenerateRaw === undefined) degraded.push("帧 meta 缺 degenerate：按非退化处理（比值可能含舍入段）");
  const slope2 = metaNumber(frame, "slope_fd2");
  const slope4 = metaNumber(frame, "slope_fd4");
  const fitLo = metaNumber(frame, "fit_lo");
  const fitHi = metaNumber(frame, "fit_hi");
  const ratioSpFirst = metaNumber(frame, "ratio_spectral_first");
  const ratioSpLast = metaNumber(frame, "ratio_spectral_last");
  const tActual = metaNumber(frame, "t_actual");
  const shift = metaNumber(frame, "shift");
  const cAdv = metaNumber(frame, "c");

  const tMax = SPEC !== undefined && SPEC.time.mode === "scrub" ? SPEC.time.tMax ?? T_MAX_FALLBACK : T_MAX_FALLBACK;
  const t = tActual !== undefined ? Math.max(0, Math.min(tActual, tMax)) : Math.max(0, Math.min(opts.t, tMax));

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });
  const L = columnsLayout(scene, width, height, { topShare: 0.55, auxH: 120, gap: 46 });
  const panelTextFits = L.floor > L.botY + L.botH + 8;

  // —— 主图：误差-自由度（双对数）——
  let peakPoint = 0;
  for (const arr of [pSp, p2, p4]) {
    for (let i = 0; i < arr.length; i += 1) {
      const v = arr[i] as number;
      if (Number.isFinite(v) && v > peakPoint) peakPoint = v;
    }
  }
  const nLo = errN[0] as number;
  const nHi = errN[errN.length - 1] as number;
  const errorAxis = makeAxis("error", [L.left, L.topY, L.leftW, L.topH], {
    xlim: [nLo, Math.max(nLo * 1.2, nHi)],
    ylim: [errFloor, 1],
    xscale: "log",
    yscale: "log",
    xlabel: "自由度 N（谱=模态数，差分=格点数）",
    title: axisTitleOf(SPEC?.axes, "error", "相对误差 vs N（双对数，1e-17–1）"),
    ylabel: axisYlabelOf(SPEC?.axes, "error", "||u_num-u_exact||_inf/||u_exact||_inf"),
    autoscaled: false,
  });
  const ratioAxis = makeAxis("ratio", [L.left, L.botY, L.leftW, L.botH], {
    xlim: [
      ratioN.length > 0 ? (ratioN[0] as number) : nLo,
      Math.max((ratioN.length > 0 ? (ratioN[0] as number) : nLo) * 1.2, ratioN.length > 0 ? (ratioN[ratioN.length - 1] as number) : nHi),
    ],
    ylim: [0, 1],
    xlabel: "自由度 N",
    title: axisTitleOf(SPEC?.axes, "ratio", "每加倍误差比（窗口 0–1）"),
    ylabel: axisYlabelOf(SPEC?.axes, "ratio", "e(2N)/e(N)"),
    autoscaled: false,
  });
  const pointAxis = makeAxis("pointwise", [L.rightX, L.topY, L.rightW, L.auxH], {
    xlim: [0, Math.max(px[px.length - 1] as number, 1e-6)],
    ylim: [errFloor, 1],
    yscale: "log",
    xlabel: "网格点 x（线性轴：含 0，不能上对数）",
    title: axisTitleOf(SPEC?.axes, "pointwise", "逐点误差 @ 最大 N（双对数）"),
    ylabel: axisYlabelOf(SPEC?.axes, "pointwise", "|u_num-u_exact|"),
    autoscaled: false,
  });
  scene.axes.push(errorAxis, ratioAxis, pointAxis);

  // —— 主图：三条误差曲线 + 地板 + 两条理论阶参考线（右半段锚定）——
  scene.layers.push(
    lineLayer("error", series([nLo, nHi], [errFloor, errFloor], pal.warn, { dash: [3, 3], width: 1, label: "误差地板" })),
    lineLayer("error", series(errN, errSp, pal.residual, { width: 2.2, label: "谱方法" })),
    lineLayer("error", series(errN, err2, pal.curves[0], { width: 1.8, label: "二阶差分" })),
    lineLayer("error", series(errN, err4, pal.curves[1], { width: 1.8, dash: [6, 3], label: "四阶差分" })),
  );
  const theoryAnchor = (yEnd: number, order: number): { x: [number, number]; y: [number, number] } | undefined => {
    if (!Number.isFinite(yEnd) || yEnd <= 0) return undefined;
    const x0 = Math.max(nLo, nHi / 4);
    const y0 = yEnd * Math.pow(nHi / x0, order);
    return { x: [x0, nHi], y: [Math.max(y0, errFloor), Math.max(yEnd, errFloor)] };
  };
  const ref2 = err2.length > 0 ? theoryAnchor(err2[err2.length - 1] as number, 2) : undefined;
  const ref4 = err4.length > 0 ? theoryAnchor(err4[err4.length - 1] as number, 4) : undefined;
  if (ref2 !== undefined) {
    scene.layers.push(lineLayer("error", series(ref2.x, ref2.y, pal.reference, { dash: [5, 4], width: 1.1, label: "二阶斜率参考" })));
  }
  if (ref4 !== undefined) {
    scene.layers.push(lineLayer("error", series(ref4.x, ref4.y, pal.reference, { dash: [2, 3], width: 1.1, label: "四阶斜率参考" })));
  }

  // —— 每加倍比值：0–1 固定窗口；退化帧/地板段不画（并说明）——
  scene.layers.push(
    lineLayer("ratio", series([ratioAxis.xlim[0], ratioAxis.xlim[1]], [0.25, 0.25], pal.reference, { dash: [6, 4], width: 1 })),
    lineLayer("ratio", series([ratioAxis.xlim[0], ratioAxis.xlim[1]], [0.0625, 0.0625], pal.reference, { dash: [2, 3], width: 1 })),
  );
  const ratioFilter = (vals: ArrayLike<number>, dropFloorTail: boolean): number[] => {
    const out: number[] = [];
    if (degenerate) return out;
    for (let i = 0; i < ratioN.length; i += 1) {
      const y = vals[i] as number;
      if (!Number.isFinite(y) || y <= 0) continue;
      if (dropFloorTail && nAtFloor !== undefined) {
        const right = ratioN[i] as number;
        if (Number.isFinite(right) && right >= nAtFloor) continue;
      }
      out.push(i);
    }
    return out;
  };
  const spIdx = ratioFilter(ratioSp, true);
  const fd2Idx = ratioFilter(ratio2, false);
  const fd4Idx = ratioFilter(ratio4, false);
  const pushRatio = (vals: ArrayLike<number>, idx: number[], color: string, label: string, dash?: number[]): void => {
    if (idx.length >= 2) {
      scene.layers.push(
        lineLayer("ratio", series(selectValues(ratioN, idx), selectValues(vals, idx), color, { width: 1.8, dash, label })),
      );
    } else if (idx.length === 1) {
      scene.layers.push(
        scatterLayer("ratio", selectValues(ratioN, idx), selectValues(vals, idx), { r: 3, color, label }),
      );
    }
  };
  pushRatio(ratio2, fd2Idx, pal.curves[0], "二阶差分");
  pushRatio(ratio4, fd4Idx, pal.curves[1], "四阶差分", [6, 3]);
  pushRatio(ratioSp, spIdx, pal.residual, "谱方法");

  // —— 逐点误差（y 对数）——
  scene.layers.push(
    lineLayer("pointwise", series([pointAxis.xlim[0], pointAxis.xlim[1]], [errFloor, errFloor], pal.warn, { dash: [3, 3], width: 1, label: "误差地板" })),
    lineLayer("pointwise", series(px, pSp, pal.residual, { width: 2.2, label: "谱方法" })),
    lineLayer("pointwise", series(px, p2, pal.curves[0], { width: 1.6, label: "二阶差分" })),
    lineLayer("pointwise", series(px, p4, pal.curves[1], { width: 1.6, dash: [6, 3], label: "四阶差分" })),
  );

  // —— 面板内图例 ——
  if (panelTextFits) {
    scene.texts.push(
      ...textBlock(
        L.left + 6,
        L.topY + L.topH - SAFE_GAP - 7 * TEXT_ROW_H,
        L.topY + L.topH - SAFE_GAP - TEXT_ROW_H,
        [
          "紫实=谱　蓝实=二阶差分　绿虚=四阶差分　红线=地板",
          "灰虚参考线：N^-2 / N^-4 理论阶（锚在最右点，只画右半段）",
        ],
        pal.muted,
        unitBudget(L.leftW),
        "bottom",
      ),
    );
    const ratioLines = [
      "灰虚=0.25（二阶理论）　点线=0.0625（四阶理论）",
      degenerate
        ? `t=${formatNumber(t, 3)} 退化帧：初值即精确解，三种误差都在舍入级，比值无信息（不画）`
        : nAtFloor !== undefined
          ? `谱比值只画 N<n_at_floor=${formatNumber(nAtFloor, 4)} 的段（地板以下只剩舍入噪声）`
          : "谱比值未做地板段剔除（meta.n_at_floor 缺失，已降级）",
    ];
    scene.texts.push(...textBlock(L.left + 6, L.botY + 6, L.botY + L.botH - 4, ratioLines, pal.muted, unitBudget(L.leftW), "top"));
    const pointLines = ["紫实=谱　蓝实=二阶差分　绿虚=四阶差分", "y 对数、x 线性（x 含 0）"];
    if (peakPoint > 1) pointLines.push(`最大逐点误差 ${formatNumber(peakPoint, 3)} > 1：超出固定窗口，溢出点名`);
    scene.texts.push(...textBlock(L.rightX + 4, L.topY + L.auxH - SAFE_GAP - 8 * TEXT_ROW_H, L.topY + L.auxH - SAFE_GAP - TEXT_ROW_H, pointLines, pal.muted, unitBudget(L.rightW), "bottom"));
  } else {
    degraded.push("画布过矮/过窄：面板说明与右列文字放不下（pitfalls 面板占满底部），关键读数见角标");
  }

  // —— 右列文字 ——
  if (L.columns && L.floor - L.textTop >= 32) {
    const head = paramSummary(frame, opts, [
      { name: "n_max", label: "N_max" },
      { name: "rho", label: "ρ", digits: 3 },
    ]);
    degraded.push(...head.degraded);
    const lines: string[] = [];
    lines.push(
      `t=${formatNumber(t, 3)} c=${formatNumber(cAdv ?? 0.85, 2)}　${head.text}` +
        (shift !== undefined ? `　位移 ${formatNumber(shift, 3)} 周期` : ""),
    );
    if (peakPoint > 1) lines.push(`最大逐点误差 ${formatNumber(peakPoint, 3)} > 1：溢出点名`);
    if (degenerate) {
      lines.push("退化帧：t=0 初值就是精确解，误差全在舍入级");
      lines.push("招牌数字请拖到 t≈0.3–0.7 再看");
    } else if (slope2 !== undefined && slope4 !== undefined) {
      lines.push(
        `FD2 ${formatNumber(slope2, 3)}（理论 -2）　FD4 ${formatNumber(slope4, 3)}（理论 -4）`,
      );
      if (fitLo !== undefined && fitHi !== undefined) {
        lines.push(`拟合窗口 [${formatNumber(fitLo, 3)}, ${formatNumber(fitHi, 3)}]`);
      }
    } else {
      degraded.push("帧 meta 缺 slope_fd2/slope_fd4：不写差分实测阶（退化帧或拟合失败）");
    }
    if (ratioSpFirst !== undefined && ratioSpLast !== undefined) {
      lines.push(`谱每加倍比 ${formatNumber(ratioSpFirst, 3)}→${formatNumber(ratioSpLast, 3)}（在变小=指数）`);
    }
    if (nAtFloor !== undefined && !degenerate) lines.push(`谱在 N=${formatNumber(nAtFloor, 4)} 撞 1e-13 地板`);
    if (!degenerate) lines.push("比值在 0–1 窗口外由溢出角标点名");
    scene.texts.push(...textBlock(L.rightX, L.textTop, L.floor, lines, pal.muted, unitBudget(L.rightW), "top"));
  } else if (L.columns) {
    degraded.push("画布过矮：右列文字放不下（pitfalls 面板之下），关键读数见角标");
  } else {
    degraded.push("窄画布（单列）：右列说明未排下，关键读数见角标与 pitfalls 面板");
  }

  // —— 角标 ——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(badge("autoscale", "固定窗口：error/pointwise 1e-17–1 · ratio 0–1", 8, 26), SLOT.aux, width),
  );
  const status: string[] = [];
  if (degenerate) {
    status.push("退化帧（t=0）：误差全在舍入级");
  } else {
    if (slope2 !== undefined) status.push(`FD2 ${formatNumber(slope2, 3)}`);
    if (slope4 !== undefined) status.push(`FD4 ${formatNumber(slope4, 3)}`);
    if (ratioSpFirst !== undefined && ratioSpLast !== undefined) {
      status.push(`谱比值 ${formatNumber(ratioSpFirst, 3)}→${formatNumber(ratioSpLast, 3)}`);
    }
  }
  scene.badges.push(placeBadge(badge("status", status.join("；")), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  const nBad =
    countNonFinite(errSp) +
    countNonFinite(err2) +
    countNonFinite(err4) +
    countNonFinite(ratioSp) +
    countNonFinite(ratio2) +
    countNonFinite(ratio4) +
    countNonFinite(pSp) +
    countNonFinite(p2) +
    countNonFinite(p4);
  applyOverflow(scene, nBad, { slot: SLOT.overflow, width });
  return scene;
};
