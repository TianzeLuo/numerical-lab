/**
 * `gradient_check`（5.3 章 viz 3）：有限差分梯度检验的 V 形曲线与代价。
 *
 * 三条纪律
 * -------
 * 1. **V 形的两臂是两个不同的误差源**：左臂是舍入误差（∝ u/ε），右臂是截断误差
 *    （中心差分 ∝ ε²，前向 ∝ ε）。最小值位置因此不同：中心 ε*∝u^(1/3)，
 *    前向 ε*∝u^(1/2)。图上把两条理论参考线同屏，斜率写进文字。
 * 2. **“方向差不多”不是检验**：在这里比的是梯度向量本身的相对误差，阈值 1e-6。
 * 3. **代价**：伴随 1 次求解（加一次反向）与有限差分的 2k 次求解是两笔账；
 *    横轴是参数个数 k，不是 ε。保存的浮点数也要算进去（检查点把内存压下来）。
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
import {
  TEXT_ROW_H,
  axisTitleOf,
  axisYlabelOf,
  columnsLayout,
  textBlock,
  unitBudget,
} from "../_common/panel.js";
import { CHAPTER, SLOT } from "./common.js";
import { manifest } from "./manifest.gen.js";

const VIZ = "gradient_check";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "梯度检验：截断 vs 舍入";

export const gradientCheckModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  const eps = requireArray(frame, "eps");
  const eCentral = requireArray(frame, "error_central");
  const eForward = requireArray(frame, "error_forward");
  const truncRef = requireArray(frame, "truncation_reference");
  const roundRef = requireArray(frame, "roundoff_reference");
  for (const [name, arr] of [
    ["error_central", eCentral],
    ["error_forward", eForward],
    ["truncation_reference", truncRef],
    ["roundoff_reference", roundRef],
  ] as const) {
    requireSameLength(eps, arr, `eps 与 ${name}`);
  }
  const nParams = requireArray(frame, "cost_n_params");
  const adjEvals = requireArray(frame, "cost_adjoint_evals");
  const fdEvals = requireArray(frame, "cost_fd_evals");
  const ckEvals = requireArray(frame, "cost_checkpoint_evals");
  requireSameLength(nParams, adjEvals, "cost_n_params 与 cost_adjoint_evals");
  requireSameLength(nParams, fdEvals, "cost_n_params 与 cost_fd_evals");
  requireSameLength(nParams, ckEvals, "cost_n_params 与 cost_checkpoint_evals");
  const pitfalls = requirePitfalls(frame);

  const degraded: string[] = [];
  const epsOptC = metaNumber(frame, "eps_opt_central");
  const epsOptF = metaNumber(frame, "eps_opt_forward");
  const bestC = metaNumber(frame, "error_best_central");
  const bestF = metaNumber(frame, "error_best_forward");
  const slopeDownC = metaNumber(frame, "slope_down_central");
  const slopeUpC = metaNumber(frame, "slope_up_central");
  const slopeDownF = metaNumber(frame, "slope_down_forward");
  const slopeUpF = metaNumber(frame, "slope_up_forward");
  const unit = metaNumber(frame, "unit_roundoff");
  const ratioC = metaNumber(frame, "eps_opt_central_over_u13");
  const ratioF = metaNumber(frame, "eps_opt_forward_over_sqrtu");
  const paramIndex = metaNumber(frame, "param_index");
  const kScanned = metaNumber(frame, "k_scanned");
  const adjGrad = metaNumber(frame, "adjoint_grad_scanned");
  const fdGrad = metaNumber(frame, "fd_grad_at_best");
  const interiorC = frame.meta["eps_opt_central_interior"] === true;
  const interiorF = frame.meta["eps_opt_forward_interior"] === true;
  const method = stringOf(frame, "method") ?? (typeof opts.params.method === "string" ? (opts.params.method as string) : "?");
  if (bestC === undefined) degraded.push("帧 meta 缺 error_best_central：无法给最小值读数");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: 0,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  const L = columnsLayout(scene, width, height, { topShare: 0.6, auxH: 140, gap: 48 });
  const epsLo = eps[0] as number;
  const epsHi = eps[eps.length - 1] as number;
  let yLo = Number.POSITIVE_INFINITY;
  let yHi = 0;
  for (const arr of [eCentral, eForward, truncRef, roundRef]) {
    for (let i = 0; i < arr.length; i += 1) {
      const v = arr[i] as number;
      if (v > 0 && Number.isFinite(v)) {
        yLo = Math.min(yLo, v);
        yHi = Math.max(yHi, v);
      }
    }
  }
  if (!(yLo > 0)) yLo = 1e-18;
  const vcurve = makeAxis("vcurve", [L.left, L.topY, L.leftW, L.topH], {
    xlim: [epsLo, epsHi],
    ylim: [yLo * 0.5, yHi * 3],
    xscale: "log",
    yscale: "log",
    xlabel: "$ε$（对数）",
    ylabel: axisYlabelOf(SPEC?.axes, "vcurve", "|g_fd − g_adj| 相对误差"),
    title: axisTitleOf(SPEC?.axes, "vcurve", "梯度检验 V 曲线（双对数）"),
    autoscaled: true,
  });

  const nLo = nParams[0] as number;
  const nHi = nParams[nParams.length - 1] as number;
  const costLo = Math.min(...Array.from(adjEvals), ...Array.from(fdEvals), ...Array.from(ckEvals));
  const costHi = Math.max(...Array.from(adjEvals), ...Array.from(fdEvals), ...Array.from(ckEvals));
  const cost = makeAxis("cost", [L.left, L.botY, L.leftW, L.botH], {
    xlim: [nLo * 0.8, nHi * 1.25],
    ylim: [Math.max(1, costLo * 0.6), costHi * 2],
    xscale: "log",
    yscale: "log",
    xlabel: "参数个数 $k$",
    ylabel: axisYlabelOf(SPEC?.axes, "cost", "求值次数"),
    title: axisTitleOf(SPEC?.axes, "cost", "代价：伴随 vs 有限差分"),
    autoscaled: true,
  });
  scene.axes.push(vcurve, cost);

  scene.layers.push(
    lineLayer("vcurve", { x: eps, y: eCentral, color: pal.curves[0] as string, width: 2.2 }),
    lineLayer("vcurve", { x: eps, y: eForward, color: pal.curves[2] as string, width: 2 }),
    lineLayer("vcurve", { x: eps, y: truncRef, color: pal.curves[0] as string, width: 1, dash: [2, 4] }),
    lineLayer("vcurve", { x: eps, y: roundRef, color: pal.residual, width: 1.1, dash: [5, 3] }),
  );
  if (bestC !== undefined && epsOptC !== undefined && bestC > 0) {
    scene.layers.push(
      lineLayer("vcurve", { x: [epsOptC, epsOptC], y: [vcurve.ylim[0], vcurve.ylim[1]], color: pal.curves[0] as string, width: 1.1, dash: [4, 3], label: "中心差分 $ε*$" }),
      scatterLayer("vcurve", [epsOptC], [bestC], { r: 3.8, color: pal.curves[0] as string, label: "中心差分最优" }),
    );
  }
  if (bestF !== undefined && epsOptF !== undefined && bestF > 0) {
    scene.layers.push(
      lineLayer("vcurve", { x: [epsOptF, epsOptF], y: [vcurve.ylim[0], vcurve.ylim[1]], color: pal.curves[2] as string, width: 1.1, dash: [2, 3], label: "前向差分 $ε*$" }),
      scatterLayer("vcurve", [epsOptF], [bestF], { r: 3.4, color: pal.curves[2] as string, label: "前向差分最优" }),
    );
  }

  scene.layers.push(
    lineLayer("cost", { x: nParams, y: adjEvals, color: pal.marker, width: 2.2 }),
    lineLayer("cost", { x: nParams, y: fdEvals, color: pal.curves[2] as string, width: 2.2 }),
    lineLayer("cost", { x: nParams, y: ckEvals, color: pal.residual, width: 1.6, dash: [4, 3] }),
  );
  // 在最后一个 k 上标出比值，避免只看两条线的视觉间距。
  if (nParams.length > 0 && fdEvals.length > 0 && adjEvals.length > 0) {
    const last = nParams.length - 1;
    const ratio = (fdEvals[last] as number) / Math.max(1, adjEvals[last] as number);
    scene.layers.push(
      scatterLayer("cost", [nParams[last] as number], [fdEvals[last] as number], { r: 3.6, color: pal.curves[2] as string, label: `k=${formatNumber(nParams[last] as number, 0)}：FD/伴随=${formatNumber(ratio, 2)}×` }),
    );
  }

  const budget = unitBudget(L.rightW);
  if (L.rightW >= 200) {
    let y = L.topY;
    const push = (lines: readonly string[], color: string): void => {
      if (y + TEXT_ROW_H > L.floor) return;
      const block = textBlock(L.rightX, y, L.floor, lines, color, budget, "top");
      scene.texts.push(...block);
      y += block.length * TEXT_ROW_H;
    };
    push(
      [
        `参数分量 j=${formatNumber(paramIndex ?? Number.NaN, 0)}，k=${formatNumber(kScanned ?? Number.NaN, 0)}，${method}`,
        "蓝=中心差分，橙=前向差分。",
        "蓝虚=截断参考，紫虚=舍入参考。",
        `中心 ε*=ε_opt=${formatNumber(epsOptC ?? Number.NaN, 2)}，最小误差 ${formatNumber(bestC ?? Number.NaN, 2)}`,
        `前向 ε*=${formatNumber(epsOptF ?? Number.NaN, 2)}，最小误差 ${formatNumber(bestF ?? Number.NaN, 2)}`,
      ],
      pal.ink,
    );
    push(
      [
        `中心斜率：左 ${formatNumber(slopeDownC ?? Number.NaN, 3)}，右 ${formatNumber(slopeUpC ?? Number.NaN, 3)}。`,
        `前向斜率：左 ${formatNumber(slopeDownF ?? Number.NaN, 3)}，右 ${formatNumber(slopeUpF ?? Number.NaN, 3)}。`,
        "理论斜率：中心 −1/+2，前向 −1/+1。",
        `ε*/u^(1/3)=${formatNumber(ratioC ?? Number.NaN, 2)}，ε*/u^(1/2)=${formatNumber(ratioF ?? Number.NaN, 2)}。`,
        `u=${formatNumber(unit ?? Number.NaN, 2)}（机器精度量级）。`,
      ],
      pal.muted,
    );
    push(
      [
        interiorC && interiorF ? "两个最优都在网格内部（不是端点假象）。" : "至少一个最优落在网格端点：V 形没有真正探到谷底。",
        `梯度读数：伴随 ${formatNumber(adjGrad ?? Number.NaN, 4)}，FD@ε* ${formatNumber(fdGrad ?? Number.NaN, 4)}。`,
        "阈值：相对误差 < 1e-6；余弦接近 1 不算。",
        "左臂舍入、右臂截断；参考线是理论斜率。",
      ],
      pal.muted,
    );
    const last = nParams.length - 1;
    if (last >= 0) {
      push(
        [
          `k=${formatNumber(nParams[last] as number, 0)}：伴随 ${formatNumber(adjEvals[last] as number, 0)} 次、FD ${formatNumber(fdEvals[last] as number, 0)} 次求值。`,
          "红=伴随，橙=FD，紫虚=检查点。",
          "FD 随 k 线性增长。",
          "但伴随要把状态存下来（检查点能压内存）。",
          `停止：${stringOf(frame, "status") ?? "?"}。`,
        ],
        pal.muted,
      );
    }
  } else {
    degraded.push("右侧说明列过窄：读数并入角标");
  }

  scene.badges.push(placeBadge(sourceBadge(dataSourceOf(frame)), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(true), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(
      badge(
        "status",
        `中心 ε*=${formatNumber(epsOptC ?? Number.NaN, 2)}，误差 ${formatNumber(bestC ?? Number.NaN, 2)}｜前向 ${formatNumber(bestF ?? Number.NaN, 2)}`,
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
