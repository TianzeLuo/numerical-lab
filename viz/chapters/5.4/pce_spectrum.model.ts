/**
 * `pce_spectrum`（5.4 章 viz 4）：多项式混沌的谱系数、截断误差与外推。
 *
 * 三条纪律
 * -------
 * 1. **系数 RMS 是 log10 值**：主图纵轴线性（值本身是 log10），不是“再取一次对数”。
 *    参考线的斜率是 `meta.decay_slope`，只在“解析函数”上才是几何衰减；折点只按 p^(-3/2)。
 * 2. **训练误差到 1e-16 什么也不能说明**：n_train=8、基规模超过样本数时训练误差可以
 *    是 4e-16 而留出误差 0.45；欠定区必须显式标出。
 * 3. **外推没有收敛保证**：帧里 exp 外推 2 倍放大 709 倍；只有能被精确表示的函数
 *    才能外推，而那不是代理模型的功劳。
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
  lineLayer,
  makeAxis,
  scatterLayer,
  sourceBadge,
} from "../../shared/src/scene.js";
import { TEXT_ROW_H, axisTitleOf, axisYlabelOf, columnsLayout, textBlock, unitBudget } from "../_common/panel.js";
import { CHAPTER, SLOT, nearestIndex } from "./common.js";
import { manifest } from "./manifest.gen.js";

const VIZ = "pce_spectrum";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "多项式混沌：谱系数与截断误差";

export const pceSpectrumModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  const degree = requireArray(frame, "degree");
  const coeff = requireArray(frame, "coeff_rms");
  const coeffValid = requireArray(frame, "coeff_valid");
  const count = frame.has("coeff_count") ? requireArray(frame, "coeff_count") : undefined;
  const orderCurve = requireArray(frame, "order_curve");
  const err = requireArray(frame, "error");
  const errValid = requireArray(frame, "error_valid");
  const errTrain = requireArray(frame, "error_train");
  const nBasis = requireArray(frame, "n_basis_curve");
  const under = requireArray(frame, "under_flag");
  const extrapX = frame.has("extrap_x") ? requireArray(frame, "extrap_x") : undefined;
  const extrapErr = frame.has("error_extrap") ? requireArray(frame, "error_extrap") : undefined;
  const decayRef = frame.has("decay_ref") ? requireArray(frame, "decay_ref") : undefined;
  requireSameLength(degree, coeff, "degree 与 coeff_rms");
  requireSameLength(degree, coeffValid, "degree 与 coeff_valid");
  if (count !== undefined) requireSameLength(degree, count, "degree 与 coeff_count");
  requireSameLength(orderCurve, err, "order_curve 与 error");
  requireSameLength(orderCurve, errValid, "order_curve 与 error_valid");
  requireSameLength(orderCurve, errTrain, "order_curve 与 error_train");
  requireSameLength(orderCurve, nBasis, "order_curve 与 n_basis_curve");
  requireSameLength(orderCurve, under, "order_curve 与 under_flag");
  if (extrapX !== undefined && extrapErr !== undefined) requireSameLength(extrapX, extrapErr, "extrap_x 与 error_extrap");
  if (decayRef !== undefined) requireSameLength(degree, decayRef, "degree 与 decay_ref");
  const pitfalls = requirePitfalls(frame);

  const degraded: string[] = [];
  if (extrapX === undefined) degraded.push("帧缺 extrap_x/error_extrap：右上不画外推扫描");
  const order = metaNumber(frame, "order");
  const nBasisCur = metaNumber(frame, "n_basis");
  const nTrain = metaNumber(frame, "n_train");
  const underdetermined = frame.meta["underdetermined"] === true;
  const errTrainCur = metaNumber(frame, "error_train_current");
  const errOutCur = metaNumber(frame, "error_outsample_current");
  const errExtrapCur = metaNumber(frame, "error_extrap_current");
  const extrapRatio = metaNumber(frame, "extrap_ratio");
  const extrapFactor = metaNumber(frame, "extrap_factor");
  const decaySlope = metaNumber(frame, "decay_slope");
  const residualL2 = metaNumber(frame, "residual_train_l2");
  const exactAt = metaNumber(frame, "exact_at_order");
  const minOut = metaNumber(frame, "min_outsample");
  const overfit = metaNumber(frame, "overfit_factor");
  const nUnder = metaNumber(frame, "n_underdetermined");
  const basis = stringOf(frame, "basis") ?? "?";
  const target = stringOf(frame, "target") ?? "?";
  const fit = stringOf(frame, "fit") ?? "?";

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: 0,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  const L = columnsLayout(scene, width, height, { topShare: 0.6, auxH: 140, gap: 48 });
  const main = makeAxis("main", [L.left, L.topY, L.leftW, L.topH], {
    xlim: [0, Math.max(1, degree[degree.length - 1] as number)],
    ylim: [-18, 1],
    xlabel: "阶 $k$",
    ylabel: axisYlabelOf(SPEC?.axes, "main", "log10 系数 RMS"),
    title: axisTitleOf(SPEC?.axes, "main", "log10 逐阶系数 RMS（地板 −17）"),
    autoscaled: true,
  });
  const aux = makeAxis("aux", [L.left, L.botY, L.leftW, L.botH], {
    xlim: [1, Math.max(2, orderCurve[orderCurve.length - 1] as number)],
    ylim: [-18, 1],
    xlabel: "截断阶 $p$",
    ylabel: axisYlabelOf(SPEC?.axes, "aux", "log10 相对误差"),
    title: axisTitleOf(SPEC?.axes, "aux", "log10 误差 vs 截断阶（训练/留出）"),
    autoscaled: true,
  });
  scene.axes.push(main, aux);

  // 主图：有效点连线；无效点（系数 0）落到地板，单独标出。
  const vx: number[] = [];
  const vy: number[] = [];
  const bx: number[] = [];
  for (let i = 0; i < degree.length; i += 1) {
    if ((coeffValid[i] as number) > 0.5 && Number.isFinite(coeff[i] as number)) {
      vx.push(degree[i] as number);
      vy.push(coeff[i] as number);
    } else {
      bx.push(degree[i] as number);
    }
  }
  if (vx.length >= 2) scene.layers.push(lineLayer("main", { x: vx, y: vy, color: pal.curves[0] as string, width: 1.8 }));
  if (vx.length > 0) scene.layers.push(scatterLayer("main", vx, vy, { r: 3, color: pal.curves[0] as string, label: "逐阶 RMS" }));
  if (bx.length > 0) scene.layers.push(scatterLayer("main", bx, bx.map(() => -17), { r: 2.4, color: pal.warn, label: "系数为 0（写地板）" }));
  if (decayRef !== undefined) {
    scene.layers.push(lineLayer("main", { x: degree, y: decayRef, color: pal.exact, width: 1.3, dash: [5, 3] }));
  }

  // 误差：训练 / 留出；欠定区用竖线 + 顶部标记。
  scene.layers.push(
    lineLayer("aux", { x: orderCurve, y: errTrain, color: pal.curves[0] as string, width: 1.8 }),
    lineLayer("aux", { x: orderCurve, y: err, color: pal.marker, width: 2.2 }),
  );
  const underPts: number[] = [];
  for (let i = 0; i < under.length; i += 1) if ((under[i] as number) > 0.5) underPts.push(orderCurve[i] as number);
  if (underPts.length > 0) {
    const first = underPts[0] as number;
    scene.layers.push(
      lineLayer("aux", { x: [first, first], y: [aux.ylim[0], aux.ylim[1]], color: pal.warn, width: 1.2, dash: [4, 3] }),
      scatterLayer("aux", underPts, underPts.map(() => -17), { r: 2.6, color: pal.warn, label: "欠定点" }),
    );
  }
  if (order !== undefined) {
    const j = nearestIndex(orderCurve, order);
    if (j >= 0) {
      scene.layers.push(scatterLayer("aux", [orderCurve[j] as number], [err[j] as number], { r: 4, color: pal.exact, label: "当前 order 的留出误差" }));
    }
  }

  // 右上：外推扫描。
  let extrapAxis: ReturnType<typeof makeAxis> | undefined;
  let extH = 0;
  if (extrapX !== undefined && extrapErr !== undefined && extrapX.length >= 2) {
    extH = Math.min(120, L.auxH);
    extrapAxis = makeAxis("extrap", [L.rightX, L.topY, L.rightW, extH], {
      xlim: [extrapX[0] as number, extrapX[extrapX.length - 1] as number],
      ylim: [-18, 2],
      xlabel: "外推倍率",
      ylabel: "$\\log_{10}$ 误差",
      title: "外推：误差随倍率",
      autoscaled: true,
    });
    scene.axes.push(extrapAxis);
    scene.layers.push(lineLayer("extrap", { x: extrapX, y: extrapErr, color: pal.residual, width: 1.8 }));
    if (extrapFactor !== undefined) {
      const j = nearestIndex(extrapX, extrapFactor);
      if (j >= 0) {
        scene.layers.push(
          lineLayer("extrap", { x: [extrapFactor, extrapFactor], y: [extrapAxis.ylim[0], extrapAxis.ylim[1]], color: pal.marker, width: 1, dash: [2, 2] }),
          scatterLayer("extrap", [extrapFactor], [extrapErr[j] as number], { r: 3.2, color: pal.marker }),
        );
      }
    }
  }

  const budget = unitBudget(L.rightW);
  let textTop = extrapAxis !== undefined ? L.topY + extH + 34 : L.topY;
  if (L.rightW >= 200 && L.floor - textTop >= 4 * TEXT_ROW_H) {
    let y = textTop;
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
        `${target} / ${basis} / ${fit}`,
        "上：蓝=系数 RMS；橙虚=衰减参考。",
        "下：蓝=训练，红=留出；右：紫=外推。",
        `order=${formatNumber(order ?? Number.NaN, 0)}，基规模 ${formatNumber(nBasisCur ?? Number.NaN, 0)}，n_train=${formatNumber(nTrain ?? Number.NaN, 0)}`,
        underdetermined ? "⚠ 欠定：基规模 > 样本数，训练误差可以到 1e-16 而留出很差。" : "非欠定：基规模 ≤ 样本数。",
      ],
      underdetermined ? pal.warn : pal.ink,
    );
    push(
      [
        `训练 ${formatNumber(errTrainCur ?? Number.NaN, 2)}，留出 ${formatNumber(errOutCur ?? Number.NaN, 2)}`,
        `最小留出 ${formatNumber(minOut ?? Number.NaN, 2)}；过拟合因子 ${formatNumber(overfit ?? Number.NaN, 2)}`,
        `训练残差 L2 ${formatNumber(residualL2 ?? Number.NaN, 2)}。`,
      ],
      pal.muted,
    );
    push(
      [
        `外推（×${formatNumber(extrapFactor ?? Number.NaN, 2)}）误差 ${formatNumber(errExtrapCur ?? Number.NaN, 2)}，放大 ${formatNumber(extrapRatio ?? Number.NaN, 2)} 倍。`,
        exactAt !== undefined ? `在 order=${formatNumber(exactAt, 0)} 处可精确表示（该目标的多项式阶有限）。` : "目标不是有限阶多项式：外推没有收敛保证。",
        `衰减斜率 ${formatNumber(decaySlope ?? Number.NaN, 2)}；解析函数几何衰减。`,
        "折点按 p^(-3/2)。",
        `欠定点 ${formatNumber(nUnder ?? Number.NaN, 0)} 个；横轴是截断阶，不是样本数。`,
      ],
      pal.muted,
    );
  } else {
    degraded.push("右列过窄/过矮：外推与读数并入角标");
  }

  scene.badges.push(placeBadge(sourceBadge(dataSourceOf(frame)), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(true), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(
      badge(
        "status",
        `p=${formatNumber(order ?? Number.NaN, 0)}｜留出 ${formatNumber(errOutCur ?? Number.NaN, 2)}｜外推 ×${formatNumber(extrapRatio ?? Number.NaN, 1)}｜${underdetermined ? "欠定" : "非欠定"}`,
        8,
        8,
      ),
      SLOT.status,
      width,
    ),
  );
  if (underdetermined) {
    scene.badges.push(placeBadge(badge("critical", `基规模 ${formatNumber(nBasisCur ?? Number.NaN, 0)} > 样本 ${formatNumber(nTrain ?? Number.NaN, 0)}：训练误差不是泛化误差`, 8, 8), SLOT.critical, width));
  }
  const dbg = degradedBadge(degraded, SLOT.degraded, width);
  if (dbg !== undefined) scene.badges.push(dbg);

  applyOverflow(scene, 0, { slot: SLOT.overflow, width });
  requireDrawableLayers(scene);
  return scene;
};
