/**
 * `bias_variance`（5.2 章 viz 4）：λ 两侧各自的失败方式。
 *
 * 防误导纪律
 * ----------
 * 1. **偏差与方差是两笔账**：偏差只与无噪右端项 b_exact 有关；把带噪 b 代进偏差
 *    里量到的是噪声。图上实线是解析分解，虚线是同一批噪声上的蒙特卡洛复核；
 *    两者在偏差^2 很小的地方用相对差会得到假差异（侧栏报绝对差 / 总误差）。
 * 2. **oracle 用了 x***：曲线最低点与现实可算的判据（GCV/L 曲线/偏差原则）不是一回事；
 *    这里把 oracle 的 λ 画出来只作基准。
 * 3. **斜率副轴是方差的斜率**（不是 MSE）：右段理论 -4（f_i~σ_i²/λ²），
 *    左段理论 0（f_i 饱和）；0 段是否出现在窗口内由 σ_min² 与 λ_min 的关系决定，
 *    没看到就是没看到，不能把 -1.5 硬读成 0。
 * 4. **地板 1e-17**：方差可以低到画不出来，写地板值不是“精确为 0”。
 *
 * 帧 `time.mode="none"`。
 */

import { formatNumber } from "../../shared/src/axes.js";
import {
  applyOverflow,
  canvasSize,
  dataSourceOf,
  degradedBadge,
  finiteMinMax,
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
  positiveMinMax,
  textBlock,
  unitBudget,
} from "../_common/panel.js";
import { CHAPTER, SLOT, nearestIndex } from "./common.js";
import { manifest } from "./manifest.gen.js";

const VIZ = "bias_variance";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "偏差-方差分解：λ 两侧各自的失败方式";
const FLOOR_FALLBACK = 1e-17;

export const biasVarianceModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  const lgLam = requireArray(frame, "log10_lam");
  const bias2 = requireArray(frame, "bias2_analytic");
  const variance = requireArray(frame, "variance_analytic");
  const mse = requireArray(frame, "mse_analytic");
  const biasMc = requireArray(frame, "bias2_mc");
  const varMc = requireArray(frame, "variance_mc");
  const mseMc = requireArray(frame, "mse_mc");
  for (const [name, arr] of [
    ["bias2_analytic", bias2],
    ["variance_analytic", variance],
    ["mse_analytic", mse],
    ["bias2_mc", biasMc],
    ["variance_mc", varMc],
    ["mse_mc", mseMc],
  ] as const) {
    requireSameLength(lgLam, arr, `log10_lam 与 ${name}`);
  }
  const hasVarSlope = frame.has("local_slope_variance");
  const slope = hasVarSlope ? requireArray(frame, "local_slope_variance") : requireArray(frame, "local_slope");
  requireSameLength(lgLam, slope, "log10_lam 与 local_slope_variance");
  const pitfalls = requirePitfalls(frame);

  const degraded: string[] = [];
  if (!hasVarSlope) {
    degraded.push("帧缺 local_slope_variance：副轴暂用 local_slope（MSE 的斜率），口径在正文注明");
  }
  const floorMeta = metaNumber(frame, "floor");
  const floor = floorMeta !== undefined && floorMeta > 0 ? floorMeta : FLOOR_FALLBACK;
  if (floorMeta === undefined) degraded.push("帧 meta 缺 floor：地板用 1e-17");

  // 帧里的 lam_current 是**λ 本身**（不是指数）；画在与 log10_lam 同一条横轴上要先取对数。
  const lamValue = metaNumber(frame, "lam_current");
  const lamLogMeta = metaNumber(frame, "log10_lam_current");
  const lamNow =
    lamValue !== undefined && lamValue > 0 ? Math.log10(lamValue) : lamLogMeta;
  const errNow = metaNumber(frame, "error_current");
  const dofNow = metaNumber(frame, "dof_current");
  const lgOracle = metaNumber(frame, "log10_lam_oracle");
  const lgOpt = metaNumber(frame, "log10_lam_opt_total");
  const errMin = metaNumber(frame, "error_min");
  const errLo = metaNumber(frame, "error_at_lam_min");
  const errHi = metaNumber(frame, "error_at_lam_max");
  const biasHi = metaNumber(frame, "bias2_at_lam_max");
  const varLo = metaNumber(frame, "variance_at_lam_min");
  const varFracLo = metaNumber(frame, "variance_over_mse_at_lam_min");
  const biasFracHi = metaNumber(frame, "bias2_over_mse_at_lam_max");
  const slopeLo = metaNumber(frame, "variance_slope_low");
  const slopeHi = metaNumber(frame, "variance_slope_high");
  const mcRel = metaNumber(frame, "mc_vs_analytic_max_rel");
  const mcBias = metaNumber(frame, "mc_bias_max_abs_over_mse");
  const mcMseNear = metaNumber(frame, "mc_mse_max_rel_near_min");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: 0,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  const L = columnsLayout(scene, width, height, { topShare: 0.55, auxH: 150, gap: 72 });
  const xr: [number, number] = [(lgLam[0] as number) - 0.2, (lgLam[lgLam.length - 1] as number) + 0.2];

  // 主轴窗只包住三条解析曲线；地板线若落在窗外就不画（不能把轴拉到 1e-17 把曲线压成一条）。
  const curveLo = Math.min(
    positiveMinMax(bias2)?.[0] ?? 1,
    positiveMinMax(variance)?.[0] ?? 1,
    positiveMinMax(mse)?.[0] ?? 1,
  );
  const decompLo = Math.max(floor, curveLo * 0.5);
  const decomp = makeAxis("decomposition", [L.left, L.topY, L.leftW, L.topH], {
    xlim: xr,
    ylim: [decompLo, 3],
    yscale: "log",
    xlabel: "log10 λ",
    ylabel: axisYlabelOf(SPEC?.axes, "decomposition", "平方误差量级"),
    title: axisTitleOf(SPEC?.axes, "decomposition", "偏差²、方差与总误差（对数轴）"),
    autoscaled: true,
  });

  // 副轴：加入 0 与 -4 两条理论参考，让“没出现的那一段”也可见。
  const sm = finiteMinMax(slope);
  const yLo = Math.min(-4.5, (sm?.[0] ?? -4) - 0.2);
  const yHi = Math.max(0.3, (sm?.[1] ?? 0) + 0.2);
  const slopeAxis = makeAxis("variance_slope", [L.left, L.botY, L.leftW, L.botH], {
    xlim: xr,
    ylim: [yLo, yHi],
    xlabel: "log10 λ",
    ylabel: axisYlabelOf(SPEC?.axes, "variance_slope", "d log10(方差) / d log10 λ"),
    title: axisTitleOf(SPEC?.axes, "variance_slope", "方差的局部斜率（参考 0 与 −4）"),
    autoscaled: true,
  });
  scene.axes.push(decomp, slopeAxis);

  // 解析实线 + 蒙特卡洛虚线（同一批噪声）。虚线不抢实线的语义：它们不该有两条结论。
  // 线不带内联 label：曲线末端挤在右边缘时六条标签会叠成一团；颜色/线型由侧栏解释。
  scene.layers.push(
    lineLayer("decomposition", { x: lgLam, y: bias2, color: pal.exact, width: 2 }),
    lineLayer("decomposition", { x: lgLam, y: variance, color: pal.curves[0] as string, width: 2 }),
    lineLayer("decomposition", { x: lgLam, y: mse, color: pal.residual, width: 2.4 }),
    lineLayer("decomposition", { x: lgLam, y: biasMc, color: pal.exact, width: 1.1, dash: [2, 4] }),
    lineLayer("decomposition", { x: lgLam, y: varMc, color: pal.curves[0] as string, width: 1.1, dash: [2, 4] }),
    lineLayer("decomposition", { x: lgLam, y: mseMc, color: pal.residual, width: 1.1, dash: [2, 4] }),
  );
  if (floor >= (decomp.ylim[0] as number)) {
    scene.layers.push(lineLayer("decomposition", { x: [xr[0], xr[1]], y: [floor, floor], color: pal.warn, width: 1, dash: [3, 3] }));
  }

  if (lgOpt !== undefined) {
    scene.layers.push(
      lineLayer("decomposition", { x: [lgOpt, lgOpt], y: [decomp.ylim[0], decomp.ylim[1]], color: pal.reference, width: 1.2, dash: [4, 3] }),
      lineLayer("variance_slope", { x: [lgOpt, lgOpt], y: [slopeAxis.ylim[0], slopeAxis.ylim[1]], color: pal.reference, width: 1.1, dash: [4, 3] }),
    );
  }
  if (lamNow !== undefined) {
    const j = nearestIndex(lgLam, lamNow);
    scene.layers.push(
      lineLayer("decomposition", { x: [lamNow, lamNow], y: [decomp.ylim[0], decomp.ylim[1]], color: pal.marker, width: 1.3, dash: [2, 2] }),
      lineLayer("variance_slope", { x: [lamNow, lamNow], y: [slopeAxis.ylim[0], slopeAxis.ylim[1]], color: pal.marker, width: 1.1, dash: [2, 2] }),
    );
    if (j >= 0) {
      scene.layers.push(
        scatterLayer("decomposition", [lamNow, lamNow, lamNow], [bias2[j] as number, variance[j] as number, mse[j] as number], { r: 3, color: pal.marker, label: "当前 λ 的三条解析曲线" }),
        scatterLayer("variance_slope", [lamNow], [slope[j] as number], { r: 3.2, color: pal.marker, label: "当前斜率" }),
      );
    }
  }

  scene.layers.push(
    lineLayer("variance_slope", { x: [xr[0], xr[1]], y: [0, 0], color: pal.exact, width: 1.2, dash: [5, 3] }),
    lineLayer("variance_slope", { x: [xr[0], xr[1]], y: [-4, -4], color: pal.curves[0] as string, width: 1.2, dash: [5, 3] }),
    lineLayer("variance_slope", { x: lgLam, y: slope, color: pal.residual, width: 2 }),
  );
  const slopeBelowZero = (() => {
    let n = 0;
    for (let i = 0; i < slope.length; i += 1) if ((slope[i] as number) > -0.5) n += 1;
    return n;
  })();

  const budget = unitBudget(L.rightW);
  if (L.rightW >= 200) {
    let y = L.topY;
    const push = (body: string, color: string): void => {
      if (y + TEXT_ROW_H > L.floor) return;
      const block = textBlock(L.rightX, y, L.floor, [body], color, budget, "top");
      scene.texts.push(...block);
      y += block.length * TEXT_ROW_H;
    };
    push(`当前 λ=${formatNumber(lamValue ?? Number.NaN, 4)}（log10=${formatNumber(lamNow ?? Number.NaN, 2)}），dof=${formatNumber(dofNow ?? Number.NaN, 2)}。`, pal.ink);
    push(`本次实现误差 ${formatNumber(errNow ?? Number.NaN, 3)}（单次噪声实现）。`, pal.ink);
    push(`oracle：期望 √MSE 最小 ${formatNumber(errMin ?? Number.NaN, 3)} @ λ=10^${formatNumber(lgOracle ?? Number.NaN, 2)}。`, pal.muted);
    push(`λ→0：方差占 ${formatNumber(varFracLo ?? Number.NaN, 3)}，误差 ${formatNumber(errLo ?? Number.NaN, 3)}（噪声接管）。`, pal.muted);
    push(`λ→∞：偏差² 占 ${formatNumber(biasFracHi ?? Number.NaN, 3)}，误差 ${formatNumber(errHi ?? Number.NaN, 3)}（解被压平）。`, pal.muted);
    push(`两端：方差 ${formatNumber(varLo ?? Number.NaN, 3)}，偏差² ${formatNumber(biasHi ?? Number.NaN, 3)}。`, pal.muted);
    push(
      `方差斜率：右段 ${formatNumber(slopeHi ?? Number.NaN, 4)}（理论 −4）；左段 ${slopeLo === undefined ? "没测到（null ≠ 0）" : formatNumber(slopeLo, 4)}。`,
      pal.muted,
    );
    push(
      slopeBelowZero === 0
        ? "本窗口没有出现 0 段：λ_min 仍大于 σ_min²，f_i 还没饱和；这不是“理论错了”。"
        : `窗口内有 ${slopeBelowZero} 个点落在 0 附近（f_i 饱和段）。`,
      slopeBelowZero === 0 ? pal.warn : pal.muted,
    );
    push(
      hasVarSlope
        ? "副轴画的是方差（不是 MSE）的斜率；MSE 的斜率是另一条曲线。"
        : "副轴画的是 MSE 的斜率（local_slope_variance 缺失的降级，不能当方差读）。",
      hasVarSlope ? pal.muted : pal.warn,
    );
    push(`MC 复核：MSE 最大相对差 ${formatNumber(mcRel ?? Number.NaN, 3)}，最小点附近 ${formatNumber(mcMseNear ?? Number.NaN, 3)}。`, pal.muted);
    push(`偏差用绝对差/总量 ${formatNumber(mcBias ?? Number.NaN, 3)}；相对差在偏差→0 时不能读。`, pal.muted);
    push("实线=解析，虚线=MC；绿虚线=oracle λ，橙点线=当前 λ。", pal.muted);
    push(`oracle 用了 x*，只作基准；现实判据见 L 曲线页。${stringOf(frame, "status") === "ok" ? "" : ` 状态：${stringOf(frame, "status")}`}`, pal.muted);
  } else {
    degraded.push("右侧说明列过窄：读数并入角标");
  }

  scene.badges.push(placeBadge(sourceBadge(dataSourceOf(frame)), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(true), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(
      badge(
        "status",
        `λ=${formatNumber(lamValue ?? Number.NaN, 4)}｜dof=${formatNumber(dofNow ?? Number.NaN, 2)}｜err=${formatNumber(errNow ?? Number.NaN, 3)}｜方差斜率 ${formatNumber(slopeHi ?? Number.NaN, 2)}`,
        8,
        8,
      ),
      SLOT.status,
      width,
    ),
  );
  const lamSpec = SPEC?.params.find((p) => p.name === "log10_lam");
  const lamCritical = lamSpec?.critical?.[0];
  if (lamNow !== undefined && lamCritical !== undefined && lamNow >= lamCritical) {
    scene.badges.push(
      placeBadge(badge("critical", `λ=10^${formatNumber(lamNow, 2)} ≥ 1：偏差² 主导，解被压平`, 8, 8), SLOT.critical, width),
    );
  }
  const dbg = degradedBadge(degraded, SLOT.degraded, width);
  if (dbg !== undefined) scene.badges.push(dbg);

  applyOverflow(scene, 0, { slot: SLOT.overflow, width });
  requireDrawableLayers(scene);
  return scene;
};
