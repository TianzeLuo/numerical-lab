/**
 * `forward_adjoint_playback`（5.3 章 viz 4）：正向轨迹与反向伴随轨迹的叠加播放。
 *
 * 三条纪律
 * -------
 * 1. **同一张相平面只是视觉安排**：y 是状态、μ 是余状态（对偶空间）。图上“相遇”
 *    不代表两个向量在同一空间里相等；有意义的陈述是 `sensitivity_residual`
 *    （μ(t)=dJ/dy(t) 的离散残差）。
 * 2. **正向画到播放头、伴随从尾部倒着画回播放头**：两个“已发生/将回流”的方向
 *    必须在图例里写清楚，否则会被读成同一条轨迹的两半。
 * 3. **增长比 |μ(0)|/|μ(T)| 不是数值不稳定**：它是正向压缩率的倒数（转置不改变
 *    奇异值），随耗散参数过零换向。
 *
 * 帧 `time.mode="scrub"`：`scene.t` 是播放比例 t/T。
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
import { CHAPTER, SLOT, finiteExtent, prefix, sliceRange } from "./common.js";
import { manifest } from "./manifest.gen.js";

const VIZ = "forward_adjoint_playback";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "正向轨迹与伴随轨迹叠加播放";
const FLOOR_FALLBACK = 1e-17;

export const forwardAdjointPlaybackModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  const time = requireArray(frame, "time");
  const fy1 = requireArray(frame, "forward_y1");
  const fy2 = requireArray(frame, "forward_y2");
  const mu1 = requireArray(frame, "adjoint_mu1");
  const mu2 = requireArray(frame, "adjoint_mu2");
  const fNorm = requireArray(frame, "forward_norm");
  const aNorm = requireArray(frame, "adjoint_norm");
  const sensResidual = requireArray(frame, "sensitivity_residual");
  const sensFd = requireArray(frame, "sensitivity_fd");
  const sensAdj = requireArray(frame, "sensitivity_adjoint");
  const contGap = requireArray(frame, "continuous_gap");
  for (const [name, arr] of [
    ["forward_y1", fy1],
    ["forward_y2", fy2],
    ["adjoint_mu1", mu1],
    ["adjoint_mu2", mu2],
    ["forward_norm", fNorm],
    ["adjoint_norm", aNorm],
  ] as const) {
    requireSameLength(time, arr, `time 与 ${name}`);
  }
  requireSameLength(sensResidual, sensFd, "sensitivity_residual 与 sensitivity_fd");
  requireSameLength(sensResidual, sensAdj, "sensitivity_residual 与 sensitivity_adjoint");
  requireSameLength(sensResidual, contGap, "sensitivity_residual 与 continuous_gap");
  if (sensResidual.length !== 2) throw new Error(`sensitivity_residual 长度 ${sensResidual.length} ≠ 2`);
  const pitfalls = requirePitfalls(frame);

  const degraded: string[] = [];
  const floor = FLOOR_FALLBACK;
  const fracMeta = metaNumber(frame, "play_fraction");
  const frac = fracMeta ?? metaNumber(frame, "t") ?? opts.t;
  const idxMeta = metaNumber(frame, "play_index");
  const kMax = time.length - 1;
  const playIdx = Math.max(0, Math.min(kMax, idxMeta ?? Math.round(frac * kMax)));
  const tPlay = metaNumber(frame, "t_play") ?? (time[playIdx] as number);
  const tauPlay = metaNumber(frame, "tau_play");
  const horizon = metaNumber(frame, "horizon");
  const nSteps = metaNumber(frame, "n_steps");
  const normY = metaNumber(frame, "norm_y_play");
  const normMu = metaNumber(frame, "norm_mu_play");
  const growth = metaNumber(frame, "adjoint_growth_ratio");
  const muMin = metaNumber(frame, "norm_mu_min");
  const muMax = metaNumber(frame, "norm_mu_max");
  const sensMax = metaNumber(frame, "sensitivity_residual_max");
  const gradStiff = metaNumber(frame, "grad_stiffness");
  const gradDamp = metaNumber(frame, "grad_damping");
  const contMax = metaNumber(frame, "continuous_gap_max");
  const nRhs = metaNumber(frame, "n_rhs_forward");
  const nFloats = metaNumber(frame, "n_floats_stored");
  if (sensMax === undefined) degraded.push("帧 meta 缺 sensitivity_residual_max：对偶残差只能逐点读");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: frac,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  const L = columnsLayout(scene, width, height, { topShare: 0.62, auxH: 140, gap: 48 });
  const extent = finiteExtent([fy1, fy2, mu1, mu2], 0.08);
  const phase = makeAxis("phase", [L.left, L.topY, L.leftW, L.topH], {
    xlim: extent,
    ylim: extent,
    xlabel: "第一分量（$y1 / μ1$，不同空间）",
    ylabel: axisYlabelOf(SPEC?.axes, "phase", "第二分量（y2 / μ2）"),
    title: axisTitleOf(SPEC?.axes, "phase", "相平面：正向（已发生）与伴随（将回流）"),
    autoscaled: true,
  });

  let yLo = Number.POSITIVE_INFINITY;
  let yHi = 0;
  for (const arr of [fNorm, aNorm]) {
    for (let i = 0; i < arr.length; i += 1) {
      const v = arr[i] as number;
      if (v > 0 && Number.isFinite(v)) {
        yLo = Math.min(yLo, v);
        yHi = Math.max(yHi, v);
      }
    }
  }
  if (!(yLo > 0)) yLo = 1e-6;
  const norms = makeAxis("norms", [L.left, L.botY, L.leftW, L.botH], {
    xlim: [time[0] as number, time[time.length - 1] as number],
    ylim: [yLo * 0.5, yHi * 2],
    yscale: "log",
    xlabel: "物理时间 $t$",
    ylabel: axisYlabelOf(SPEC?.axes, "norms", "‖y(t)‖ / ‖μ(t)‖"),
    title: axisTitleOf(SPEC?.axes, "norms", "范数（半对数）"),
    autoscaled: true,
  });
  scene.axes.push(phase, norms);

  // 正向：0..播放头；伴随：从尾部倒着画回播放头（同一数组按时间升序，视觉上反向铺开）。
  const fx = prefix(fy1, playIdx + 1);
  const fy = prefix(fy2, playIdx + 1);
  const ax = sliceRange(mu1, playIdx, kMax);
  const ay = sliceRange(mu2, playIdx, kMax);
  if (fx.length >= 2) scene.layers.push(lineLayer("phase", { x: fx, y: fy, color: pal.curves[0] as string, width: 2.2 }));
  if (ax.length >= 2) scene.layers.push(lineLayer("phase", { x: ax, y: ay, color: pal.residual, width: 1.9, dash: [5, 3] }));
  if (fx.length > 0) {
    scene.layers.push(scatterLayer("phase", [fx[fx.length - 1] as number], [fy[fy.length - 1] as number], { r: 4, color: pal.curves[0] as string, label: "播放头 $y(t)$" }));
  }
  if (ax.length > 0) {
    scene.layers.push(scatterLayer("phase", [ax[0] as number], [ay[0] as number], { r: 4, color: pal.residual, label: "播放头 $μ(t)$" }));
  }
  scene.layers.push(
    lineLayer("norms", { x: prefix(time, playIdx + 1), y: prefix(fNorm, playIdx + 1), color: pal.curves[0] as string, width: 2, label: "$‖y(t)‖$（已发生）" }),
    lineLayer("norms", { x: sliceRange(time, playIdx, kMax), y: sliceRange(aNorm, playIdx, kMax), color: pal.residual, width: 1.8, dash: [5, 3], label: "$‖μ(t)‖$（将回流）" }),
  );
  if (tPlay !== undefined && Number.isFinite(tPlay)) {
    scene.layers.push(
      lineLayer("norms", { x: [tPlay, tPlay], y: [norms.ylim[0], norms.ylim[1]], color: pal.cursor, width: 1.2, dash: [2, 2] }),
    );
  }

  // 右列：对偶残差（requiredAux） + 文字。
  let resAxis: ReturnType<typeof makeAxis> | undefined;
  let resH = 0;
  if (L.rightW >= 200) {
    resH = Math.min(110, L.auxH);
    resAxis = makeAxis("residual", [L.rightX, L.topY, L.rightW, resH], {
      xlim: [-0.2, 1.2],
      ylim: [floor, Math.max(1e-6, (sensMax ?? 1e-9) * 100)],
      yscale: "log",
      xlabel: "",
      ylabel: "残差",
      title: "对偶残差 |μ − dJ/dy|",
      autoscaled: true,
      tickTarget: 3,
    });
    scene.axes.push(resAxis);
    scene.layers.push(
      lineLayer("residual", { x: [-0.2, 1.2], y: [floor, floor], color: pal.muted, width: 1, dash: [3, 3] }),
      lineLayer("residual", { x: [-0.2, 1.2], y: [1e-8, 1e-8], color: pal.warn, width: 1.1, dash: [4, 3] }),
      scatterLayer("residual", [0, 1], sensResidual, { r: 3.4, color: pal.marker, label: "两个参数分量的相对残差" }),
    );
  }

  const budget = unitBudget(L.rightW);
  let textTop = resAxis !== undefined ? L.topY + resH + 34 : L.topY;
  if (L.rightW >= 200 && L.floor - textTop >= 4 * TEXT_ROW_H) {
    let y = textTop;
    const push = (lines: readonly string[], color: string): void => {
      const block = textBlock(L.rightX, y, L.floor, lines, color, budget, "top");
      scene.texts.push(...block);
      y += block.length * TEXT_ROW_H;
    };
    push(
      [
        `播放头 t=${formatNumber(tPlay ?? Number.NaN, 3)}（τ=${formatNumber(tauPlay ?? Number.NaN, 3)}），index ${playIdx}/${kMax}`,
        `‖y‖=${formatNumber(normY ?? Number.NaN, 4)}，‖μ‖=${formatNumber(normMu ?? Number.NaN, 4)}`,
        horizon !== undefined ? `T=${formatNumber(horizon, 2)}，${formatNumber(nSteps ?? Number.NaN, 0)} 步。` : `${formatNumber(nSteps ?? Number.NaN, 0)} 步。`,
        "蓝实=正向 y；紫虚=伴随 μ。",
      ],
      pal.ink,
    );
    push(
      [
        `增长比 |μ(0)|/|μ(T)|=${formatNumber(growth ?? Number.NaN, 4)}`,
        `（<1，正向流收缩）。`,
        `‖μ(t)‖ 范围 [${formatNumber(muMin ?? Number.NaN, 3)}, ${formatNumber(muMax ?? Number.NaN, 3)}]。`,
        `ΔJ/Δθ：p1 ${formatNumber(gradStiff ?? Number.NaN, 3)}，p2 ${formatNumber(gradDamp ?? Number.NaN, 3)}。`,
      ],
      pal.muted,
    );
    push(
      [
        `对偶残差最大 ${formatNumber(sensMax ?? Number.NaN, 3)}。`,
        `连续极限差最大 ${formatNumber(contMax ?? Number.NaN, 3)}。`,
        "残差图：灰虚=地板，橙虚=合格线 1e-8。",
        "同屏不同空间；只有残差是定量陈述。",
        `工作：RHS ${formatNumber(nRhs ?? Number.NaN, 0)} 次，存 ${formatNumber(nFloats ?? Number.NaN, 0)} 个浮点数。`,
        `停止：${stringOf(frame, "status") ?? "?"}。`,
      ],
      pal.muted,
    );
  } else {
    degraded.push("右列过窄/过矮：残差与读数并入角标");
  }

  scene.badges.push(placeBadge(sourceBadge(dataSourceOf(frame)), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(true), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(
      badge(
        "status",
        `t/T=${formatNumber(frac, 2)}｜‖y‖=${formatNumber(normY ?? Number.NaN, 3)}｜‖μ‖=${formatNumber(normMu ?? Number.NaN, 3)}｜对偶残差 ${formatNumber(sensMax ?? Number.NaN, 2)}`,
        8,
        8,
      ),
      SLOT.status,
      width,
    ),
  );
  scene.badges.push(placeBadge(badge("critical", `|μ(0)|/|μ(T)|=${formatNumber(growth ?? Number.NaN, 3)}：正向压缩率的倒数，不是数值不稳定`, 8, 8), SLOT.critical, width));
  const dbg = degradedBadge(degraded, SLOT.degraded, width);
  if (dbg !== undefined) scene.badges.push(dbg);

  applyOverflow(scene, 0, { slot: SLOT.overflow, width });
  requireDrawableLayers(scene);
  return scene;
};
