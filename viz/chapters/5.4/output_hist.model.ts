/**
 * `output_hist`（5.4 章 viz 2）：输入分布经过前向模型后的输出分布。
 *
 * 三条纪律
 * -------
 * 1. **参考密度可能是 CLT 近似**：`strings.ref_kind` / `meta.ref_is_exact` 说明它是精确卷积
 *    还是同均值同方差的正态；若是近似，密度差里同时含抽样误差与 CLT 自己的偏差。
 * 2. **直方图依赖 bin 数**：偏差 ~h²、方差 ~1/(Nh)；帧里的 bins_error 是这条 U 形曲线的实测，
 *    侧栏报最优 bin 数，不把某一个 bin 数当“真相”。
 * 3. **QQ 图比“看起来像正态”定量**：点偏离 y=x 的位置是分位数差；KS 距离另给。
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
import { CHAPTER, SLOT, argMinLog } from "./common.js";
import { manifest } from "./manifest.gen.js";

const VIZ = "output_hist";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "输出分布直方图";

export const outputHistModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  const centers = requireArray(frame, "bin_center");
  const density = requireArray(frame, "density");
  const densityRef = requireArray(frame, "density_ref");
  const error = requireArray(frame, "error");
  const errorValid = requireArray(frame, "error_valid");
  const qqRef = requireArray(frame, "qq_ref");
  const qqEmp = requireArray(frame, "qq_emp");
  requireSameLength(centers, density, "bin_center 与 density");
  requireSameLength(centers, densityRef, "bin_center 与 density_ref");
  requireSameLength(centers, error, "bin_center 与 error");
  requireSameLength(centers, errorValid, "bin_center 与 error_valid");
  requireSameLength(qqRef, qqEmp, "qq_ref 与 qq_emp");
  const binsCurve = frame.has("bins_curve") ? requireArray(frame, "bins_curve") : undefined;
  const binsError = frame.has("bins_error") ? requireArray(frame, "bins_error") : undefined;
  if (binsCurve !== undefined && binsError !== undefined) requireSameLength(binsCurve, binsError, "bins_curve 与 bins_error");
  const pitfalls = requirePitfalls(frame);

  const degraded: string[] = [];
  if (binsCurve === undefined) degraded.push("帧缺 bins_curve/bins_error：不报 bin 数权衡的最优点");
  const n = metaNumber(frame, "n");
  const nBins = metaNumber(frame, "n_bins");
  const binWidth = metaNumber(frame, "bin_width");
  const mean = metaNumber(frame, "mean");
  const std = metaNumber(frame, "std");
  const skew = metaNumber(frame, "skewness");
  const kurt = metaNumber(frame, "excess_kurtosis");
  const ksNormal = metaNumber(frame, "ks_to_normal");
  const ksRef = metaNumber(frame, "ks_to_ref");
  const within = metaNumber(frame, "within_1p96_sigma");
  const refExact = frame.meta["ref_is_exact"] === true;
  const spread = metaNumber(frame, "spread");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: 0,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  const L = columnsLayout(scene, width, height, { topShare: 0.58, auxH: 150, gap: 56 });
  let xLo = Number.POSITIVE_INFINITY;
  let xHi = Number.NEGATIVE_INFINITY;
  let yHi = 0;
  for (let i = 0; i < centers.length; i += 1) {
    xLo = Math.min(xLo, centers[i] as number);
    xHi = Math.max(xHi, centers[i] as number);
    yHi = Math.max(yHi, density[i] as number, densityRef[i] as number);
  }
  if (!Number.isFinite(xLo)) {
    xLo = -1;
    xHi = 1;
  }
  const half = (binWidth ?? (xHi - xLo) / Math.max(1, centers.length)) / 2;
  const main = makeAxis("main", [L.left, L.topY, L.leftW, L.topH], {
    xlim: [xLo - half, xHi + half],
    ylim: [0, Math.max(0.05, yHi * 1.15)],
    xlabel: "$y$",
    ylabel: axisYlabelOf(SPEC?.axes, "main", "$p_Y(y)$"),
    title: axisTitleOf(SPEC?.axes, "main", "输出密度（直方图 $\\hat{p}_Y$ vs 参考 $p_Y$）"),
    autoscaled: true,
  });
  const aux = makeAxis("aux", [L.left, L.botY, L.leftW, L.botH], {
    xlim: [xLo - half, xHi + half],
    ylim: [-18, 1],
    xlabel: "$y$",
    ylabel: axisYlabelOf(SPEC?.axes, "aux", "$\\log_{10}|\\hat{p}-p_{\\mathrm{ref}}|$"),
    title: axisTitleOf(SPEC?.axes, "aux", "$\\log_{10}$ 密度差（地板 $-17$）"),
    autoscaled: true,
  });
  scene.axes.push(main, aux);

  scene.layers.push(
    barsLayer("main", centers, density, { width: (binWidth ?? (xHi - xLo) / centers.length) * 0.9, color: pal.curves[0] as string, label: "直方图密度 $\\hat{p}$" }),
    lineLayer("main", { x: centers, y: densityRef, color: pal.marker, width: 2 }),
  );

  const eX: number[] = [];
  const eY: number[] = [];
  const eBadX: number[] = [];
  for (let i = 0; i < centers.length; i += 1) {
    if ((errorValid[i] as number) > 0.5 && Number.isFinite(error[i] as number)) {
      eX.push(centers[i] as number);
      eY.push(error[i] as number);
    } else {
      eBadX.push(centers[i] as number);
    }
  }
  if (eX.length >= 2) {
    scene.layers.push(lineLayer("aux", { x: eX, y: eY, color: pal.residual, width: 1.8 }));
  }
  if (eBadX.length > 0) {
    scene.layers.push(scatterLayer("aux", eBadX, eBadX.map(() => -17), { r: 2.2, color: pal.muted, label: "无效（地板 $-17$）" }));
  }
  scene.layers.push(lineLayer("aux", { x: [xLo, xHi], y: [-17, -17], color: pal.muted, width: 1, dash: [3, 3] }));

  // 右上：QQ 图（分位数 vs 分位数）；y=x 是“完全一致”。
  const qqH = Math.min(150, L.auxH);
  const qq = makeAxis("qq", [L.rightX, L.topY, L.rightW, qqH], {
    xlim: [Math.min(...Array.from(qqRef)), Math.max(...Array.from(qqRef))],
    ylim: [Math.min(...Array.from(qqEmp)), Math.max(...Array.from(qqEmp))],
    xlabel: "$q_{\\mathrm{ref}}$",
    ylabel: "$q_{\\mathrm{emp}}$",
    title: "Q-Q 对照（偏离 $y=x$ 表示分布差）",
    autoscaled: true,
  });
  scene.axes.push(qq);
  scene.layers.push(
    lineLayer("qq", { x: qq.xlim, y: qq.ylim, color: pal.reference, width: 1, dash: [4, 3] }),
    scatterLayer("qq", qqRef, qqEmp, { r: 2, color: pal.curves[0] as string, label: "$q_{\\mathrm{emp}}$ vs $q_{\\mathrm{ref}}$" }),
  );

  const budget = unitBudget(L.rightW);
  let y = L.topY + qqH + 34;
  if (L.rightW >= 200 && L.floor - y >= 3 * TEXT_ROW_H) {
    let dropped = false;
    const push = (lines: readonly string[], color: string): void => {
      if (y + 2 * TEXT_ROW_H > L.floor) {
        if (!dropped) degraded.push("右列空间不足：部分读数未画（角标保留关键值）");
        dropped = true;
        return;
      }
      const block = textBlock(L.rightX, y, L.floor, lines, color, budget, "top");
      scene.texts.push(...block);
      y += block.length * TEXT_ROW_H;
    };
    push(
      [
        `$n=${formatNumber(n ?? Number.NaN, 0)}$，$B=${formatNumber(nBins ?? Number.NaN, 0)}$（$h=${formatNumber(binWidth ?? Number.NaN, 3)}$）`,
        `均值 $\\mu=${formatNumber(mean ?? Number.NaN, 3)}$，$\\sigma=${formatNumber(std ?? Number.NaN, 3)}$。`,
        `偏度 $g_1=${formatNumber(skew ?? Number.NaN, 3)}$，超额峰度 $g_2=${formatNumber(kurt ?? Number.NaN, 3)}$。`,
      ],
      pal.ink,
    );
    push(["蓝=直方图，红=参考；紫=密度差，灰虚=地板（$\\hat{p}$ vs $p_{\\mathrm{ref}}$）。"], pal.muted);
    push(
      [
        refExact ? "参考：精确卷积密度；密度差主要是抽样误差。" : "参考：CLT 近似，不是真值。",
        `$D_{\\mathrm{KS}}$ vs 正态 ${formatNumber(ksNormal ?? Number.NaN, 3)}；vs 参考 ${formatNumber(ksRef ?? Number.NaN, 3)}。`,
        `$\\pm1.96\\sigma$ 内 ${formatNumber(within !== undefined ? within * 100 : Number.NaN, 2)}%。` +
          (spread !== undefined ? ` 输入 $\\sigma_X=${formatNumber(spread, 2)}$。` : ""),
      ],
      refExact ? pal.muted : pal.warn,
    );
    if (binsCurve !== undefined && binsError !== undefined) {
      const j = argMinLog(binsError);
      push(
        [
          `实测最小 RMS 在 bin=${formatNumber(binsCurve[j] ?? Number.NaN, 0)}；不是“越多越好”。`,
          `$\\log_{10}\\mathrm{RMS}$ ${formatNumber(binsError[j] ?? Number.NaN, 2)}；曲线呈 U 形。`,
        ],
        pal.muted,
      );
    }
  } else {
    degraded.push("右列过窄/过矮：数值读数并入角标");
  }

  scene.badges.push(placeBadge(sourceBadge(dataSourceOf(frame)), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(true), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(
      badge(
        "status",
        `μ=${formatNumber(mean ?? Number.NaN, 3)}｜σ=${formatNumber(std ?? Number.NaN, 3)}｜KS(ref)=${formatNumber(ksRef ?? Number.NaN, 3)}｜${refExact ? "精确参考" : "CLT 参考"}`,
        8,
        8,
      ),
      SLOT.status,
      width,
    ),
  );
  const dbg = degradedBadge(degraded, SLOT.degraded, width);
  if (dbg !== undefined) scene.badges.push(dbg);

  applyOverflow(scene, 0, { slot: SLOT.overflow, width });
  requireDrawableLayers(scene);
  return scene;
};
