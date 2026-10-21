/**
 * `filter_shapes`（5.2 章 viz 2）：滤波因子形状与“同自由度谁更准”。
 *
 * 三条纪律
 * -------
 * 1. **滤波因子不是正则化方法的全部**：TSVD 是阶跃、Tikhonov 是平滑过渡、
 *    Landweber 是 `1-(1-ωσ²)^k`；TV 根本不对应任何谱滤波（图上的是用 b=u_i
 *    逐个探出来的有效读数，换一个 b 就变）。`filter_tv_probe_mixed` 就是反例。
 * 2. **比较精度必须在同一个有效自由度上**：`matched_dof` 处三条误差曲线同屏，
 *    “形状不同 ≠ 精度不同”。横轴是 Σf_i，不是 λ、不是 k。
 * 3. **ω_rel=2 是 Landweber 的发散线**：越线后滤波因子出界，帧里用夹值 + 掩码
 *    标出无效段；这类点的 y 不是“真实因子”，不能当数据读。
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
  criticalBadge,
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
  positiveMinMax,
  textBlock,
  unitBudget,
} from "../_common/panel.js";
import { CHAPTER, SLOT } from "./common.js";
import { manifest } from "./manifest.gen.js";

const VIZ = "filter_shapes";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "滤波因子：同样自由度下谁更准";

export const filterShapesModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  const index = requireArray(frame, "index");
  const fTsvd = requireArray(frame, "filter_tsvd");
  const fTik = requireArray(frame, "filter_tikhonov");
  const fLw = requireArray(frame, "filter_landweber");
  const fLwValid = requireArray(frame, "filter_landweber_valid");
  const fTv = requireArray(frame, "filter_tv_probe");
  const dofTsvd = requireArray(frame, "dof_tsvd_curve");
  const errTsvd = requireArray(frame, "error_tsvd_curve");
  const dofTik = requireArray(frame, "dof_tikhonov_curve");
  const errTik = requireArray(frame, "error_tikhonov_curve");
  const dofLw = requireArray(frame, "dof_landweber_curve");
  const errLw = requireArray(frame, "error_landweber_curve");
  const errMatched = requireArray(frame, "error_matched");
  requireSameLength(index, fTsvd, "index 与 filter_tsvd");
  requireSameLength(index, fTik, "index 与 filter_tikhonov");
  requireSameLength(index, fLw, "index 与 filter_landweber");
  requireSameLength(index, fLwValid, "index 与 filter_landweber_valid");
  requireSameLength(index, fTv, "index 与 filter_tv_probe");
  requireSameLength(dofTsvd, errTsvd, "dof_tsvd_curve 与 error_tsvd_curve");
  requireSameLength(dofTik, errTik, "dof_tikhonov_curve 与 error_tikhonov_curve");
  requireSameLength(dofLw, errLw, "dof_landweber_curve 与 error_landweber_curve");
  if (errMatched.length !== 3) throw new Error(`error_matched 长度 ${errMatched.length} ≠ 3（tsvd/tikhonov/landweber）`);
  const fTvMixed = frame.has("filter_tv_probe_mixed") ? requireArray(frame, "filter_tv_probe_mixed") : undefined;
  if (fTvMixed !== undefined) requireSameLength(index, fTvMixed, "index 与 filter_tv_probe_mixed");
  const pitfalls = requirePitfalls(frame);

  const degraded: string[] = [];
  if (fTvMixed === undefined) degraded.push("帧缺 filter_tv_probe_mixed：TV 的 b-依赖性只能看侧栏数字");
  const lam = metaNumber(frame, "lam");
  const kTsvd = metaNumber(frame, "k_tsvd");
  const nIter = metaNumber(frame, "n_iter");
  const omegaRel = metaNumber(frame, "omega_rel");
  const omegaLine = metaNumber(frame, "omega_divergence_threshold_rel");
  const diverges = frame.meta["landweber_diverges"] === true;
  const maxFactor = metaNumber(frame, "landweber_max_abs_factor");
  const dofT = metaNumber(frame, "dof_tikhonov");
  const dofS = metaNumber(frame, "dof_tsvd");
  const dofL = metaNumber(frame, "dof_landweber");
  const errT = metaNumber(frame, "error_tikhonov");
  const errS = metaNumber(frame, "error_tsvd");
  const errL = metaNumber(frame, "error_landweber");
  const matchedDof = metaNumber(frame, "matched_dof");
  const winner = stringOf(frame, "winner");
  const tvDependence = metaNumber(frame, "tv_probe_dependence");
  const tvLeakage = metaNumber(frame, "tv_probe_max_leakage");
  const halfTik = metaNumber(frame, "half_power_index_tikhonov");
  const halfLw = metaNumber(frame, "half_power_index_landweber");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: 0,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  const L = columnsLayout(scene, width, height, { topShare: 0.52, auxH: 150, gap: 72 });

  const iHi = Math.max(2, index[index.length - 1] as number);
  const filters = makeAxis("filters", [L.left, L.topY, L.leftW, L.topH], {
    xlim: [1, iHi],
    ylim: [0, 1.15],
    xlabel: "",
    ylabel: axisYlabelOf(SPEC?.axes, "filters", "f_i"),
    title: "滤波因子 $f_i$ 随模态下标 $i$（$\\sigma_i$ 递减）",
    autoscaled: false,
  });

  let dofLo = Number.POSITIVE_INFINITY;
  let dofHi = 0;
  for (const arr of [dofTsvd, dofTik, dofLw]) {
    for (let i = 0; i < arr.length; i += 1) {
      const v = arr[i] as number;
      if (v > 0 && Number.isFinite(v)) {
        dofLo = Math.min(dofLo, v);
        dofHi = Math.max(dofHi, v);
      }
    }
  }
  if (!(dofLo > 0)) dofLo = 0.05;
  const errPos = positiveMinMax(errTsvd) ?? [1e-17, 1];
  const errPosT = positiveMinMax(errTik);
  const errPosL = positiveMinMax(errLw);
  const errLoAll = Math.max(1e-17, Math.min(errPos[0], errPosT?.[0] ?? errPos[0], errPosL?.[0] ?? errPos[0]) * 0.5);
  const errHiAll = Math.max(
    errLoAll * 10,
    Math.max(errPos[1], errPosT?.[1] ?? 0, errPosL?.[1] ?? 0) * 1.5,
  );
  const dofError = makeAxis("dof_error", [L.left, L.botY, L.leftW, L.botH], {
    xlim: [dofLo * 0.8, Math.max(dofLo * 4, dofHi * 1.25, (matchedDof ?? 0) * 1.15)],
    ylim: [errLoAll, errHiAll],
    xscale: "log",
    yscale: "log",
    xlabel: "有效自由度 $\\sum f_i$（横轴必须同口径）",
    ylabel: axisYlabelOf(SPEC?.axes, "dof_error", "相对误差"),
    title: axisTitleOf(SPEC?.axes, "dof_error", "误差随自由度（对数轴）"),
    autoscaled: true,
  });
  scene.axes.push(filters, dofError);

  // 滤波因子：TSVD（阶跃）、Tikhonov（平滑）、Landweber（几何衰减）、TV（探针）。
  // 线一律不带内联 label：7 条曲线在右端叠字是 5.2 第一版基线的事故；颜色由侧栏解释。
  scene.layers.push(lineLayer("filters", { x: index, y: fTsvd, color: pal.curves[2] as string, width: 1.8 }));
  scene.layers.push(lineLayer("filters", { x: index, y: fTik, color: pal.curves[0] as string, width: 1.8 }));
  // Landweber 的无效段用红色点标在边界上，不参与连线。
  const lwX: number[] = [];
  const lwY: number[] = [];
  const lwBadX: number[] = [];
  for (let i = 0; i < fLw.length; i += 1) {
    const v = fLw[i] as number;
    if ((fLwValid[i] as number) > 0.5 && Number.isFinite(v)) {
      lwX.push(index[i] as number);
      lwY.push(Math.max(-0.05, Math.min(1.15, v)));
    } else {
      lwBadX.push(index[i] as number);
    }
  }
  if (lwX.length >= 2) scene.layers.push(lineLayer("filters", { x: lwX, y: lwY, color: pal.residual, width: 1.6 }));
  if (lwBadX.length > 0) {
    scene.layers.push(scatterLayer("filters", lwBadX, lwBadX.map(() => 1.08), { r: 2.6, color: pal.warn, label: "因子出界（掩码为 0，夹值不是数据）" }));
  }
  scene.layers.push(lineLayer("filters", { x: index, y: fTv, color: pal.curves[1] as string, width: 1.5, dash: [5, 3] }));
  // 混合 b 的 TV 探针读数可以是负的（不是 0）；固定窗口装不下，按“在窗内”切段画，
  // 断口本身就是信息。若把它连成一条跨窗线，读图的人会以为中间没有数据点。
  let tvMixedOut = 0;
  if (fTvMixed !== undefined) {
    let runX: number[] = [];
    let runY: number[] = [];
    const flush = (): void => {
      if (runX.length >= 2) {
        scene.layers.push(lineLayer("filters", { x: runX, y: runY, color: pal.exact, width: 1.4, dash: [2, 3] }));
      }
      runX = [];
      runY = [];
    };
    for (let i = 0; i < fTvMixed.length; i += 1) {
      const v = fTvMixed[i] as number;
      if (Number.isFinite(v) && v >= 0 && v <= 1.15) {
        runX.push(index[i] as number);
        runY.push(v);
      } else {
        tvMixedOut += 1;
        flush();
      }
    }
    flush();
  }
  // 半功率交点：只看形状相等的 Tikhonov / Landweber 之间才有“同指数”可比性。
  if (halfTik !== undefined) {
    scene.layers.push(scatterLayer("filters", [halfTik], [0.5], { r: 3, color: pal.curves[0] as string, label: `Tikhonov f=0.5 @ i=${formatNumber(halfTik, 0)}` }));
  }
  if (halfLw !== undefined) {
    scene.layers.push(scatterLayer("filters", [halfLw], [0.5], { r: 3, color: pal.residual, label: `Landweber f=0.5 @ i=${formatNumber(halfLw, 0)}` }));
  }

  // 误差-自由度：三条曲线必须同横轴；同样不画内联曲线名（会在右端叠字）。
  scene.layers.push(
    lineLayer("dof_error", { x: dofTsvd, y: errTsvd, color: pal.curves[2] as string, width: 1.8 }),
    lineLayer("dof_error", { x: dofTik, y: errTik, color: pal.curves[0] as string, width: 1.8 }),
    lineLayer("dof_error", { x: dofLw, y: errLw, color: pal.residual, width: 1.8 }),
  );
  if (matchedDof !== undefined && matchedDof > 0) {
    scene.layers.push(
      lineLayer("dof_error", {
        x: [matchedDof, matchedDof],
        y: [dofError.ylim[0], dofError.ylim[1]],
        color: pal.reference,
        width: 1.2,
        dash: [4, 3],
      }),
    );
    const matchedColors = [pal.curves[2] as string, pal.curves[0] as string, pal.residual];
    const matchedNames = ["TSVD", "Tikhonov", "Landweber"];
    for (let i = 0; i < 3; i += 1) {
      scene.layers.push(
        scatterLayer("dof_error", [matchedDof], [errMatched[i] as number], {
          r: 3.6,
          color: matchedColors[i] as string,
          label: `${matchedNames[i]} @ 同自由度`,
        }),
      );
    }
  }

  const budget = unitBudget(L.rightW);
  if (L.rightW >= 200) {
    let y = L.topY;
    const push = (body: string, color: string): void => {
      if (y + TEXT_ROW_H > L.floor) return;
      const block = textBlock(L.rightX, y, L.floor, [body], color, budget, "top");
      scene.texts.push(...block);
      y += block.length * TEXT_ROW_H;
    };
    push(`λ=10^${formatNumber(lam !== undefined ? Math.log10(lam) : Number.NaN, 2)}｜TSVD k=${formatNumber(kTsvd ?? Number.NaN, 0)}｜Landweber ${formatNumber(nIter ?? Number.NaN, 0)} 步`, pal.ink);
    push(`同自由度 dof=${formatNumber(matchedDof ?? Number.NaN, 2)}；最准 ${winner ?? "?"}`, pal.ink);
    push(`Tikhonov（蓝）：err=${formatNumber(errT ?? Number.NaN, 3)}，dof=${formatNumber(dofT ?? Number.NaN, 2)}`, pal.curves[0] as string);
    push(`TSVD（绿）：err=${formatNumber(errS ?? Number.NaN, 3)}，dof=${formatNumber(dofS ?? Number.NaN, 2)}`, pal.curves[2] as string);
    push(`Landweber（红）：err=${formatNumber(errL ?? Number.NaN, 3)}，dof=${formatNumber(dofL ?? Number.NaN, 2)}`, pal.residual);
    if (halfTik !== undefined || halfLw !== undefined) {
      push(`半功率下标：Tik ${formatNumber(halfTik ?? Number.NaN, 0)}、LW ${formatNumber(halfLw ?? Number.NaN, 0)}；只对同形状可比。`, pal.muted);
    }
    push("横轴是有效自由度 Σf_i；拿 λ 或迭代数当横轴，三条曲线不可比。", pal.muted);
    push("滤波因子只是形状；同样的形状在不同谱上精度仍可能不同。", pal.muted);
    push("TV 不是谱滤波：橙=单模态 b=u_i，绿=混合 b。", pal.curves[1] as string);
    push(
      tvDependence !== undefined
        ? `换 b 后同一读数最大改变 ${formatNumber(tvDependence, 3)}，泄漏 ${formatNumber(tvLeakage ?? Number.NaN, 3)}。`
        : "两条探针曲线的差就是 b-依赖性。",
      pal.curves[1] as string,
    );
    if (tvMixedOut > 0) {
      push(`混合 b 有 ${tvMixedOut} 个探针读数在窗 [0,1.15] 外（负值不是 0），线按断口画。`, pal.muted);
    }
    if (omegaRel !== undefined) {
      push(
        `ω_rel=${formatNumber(omegaRel, 3)}（发散线 ${formatNumber(omegaLine ?? Number.NaN, 2)}）：` +
          (diverges
            ? `已发散：最大 |因子| ${formatNumber(maxFactor ?? Number.NaN, 3)}，红点是夹值不是读数。`
            : "因子全程在 [0,1]。"),
        diverges ? pal.warn : pal.muted,
      );
    } else {
      push("帧缺 omega_rel：Landweber 步长口径不明。", pal.warn);
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
        `同自由度 dof=${formatNumber(matchedDof ?? Number.NaN, 2)}｜最准 ${winner ?? "?"}｜ω_rel=${formatNumber(omegaRel ?? Number.NaN, 2)}`,
        8,
        8,
      ),
      SLOT.status,
      width,
    ),
  );
  if (omegaRel !== undefined && omegaLine !== undefined) {
    scene.badges.push(placeBadge(criticalBadge("ω_rel", omegaRel, omegaLine, { beyond: "above" }), SLOT.critical, width));
  }
  const dbg = degradedBadge(degraded, SLOT.degraded, width);
  if (dbg !== undefined) scene.badges.push(dbg);

  // TV mixed 的负值会越出固定窗口；applyOverflow 会如实点名（不是把负值裁掉）。
  applyOverflow(scene, 0, { slot: SLOT.overflow, width });
  requireDrawableLayers(scene);
  return scene;
};
