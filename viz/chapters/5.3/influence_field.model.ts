/**
 * `influence_field`（5.3 章 viz 2）：伴随/影响场随**反向时间**扩散。
 *
 * 三条纪律
 * -------
 * 1. **横轴是位置、纵轴是反向时间**：`tau=0` 是终端观测时刻（影响场=观测权重），
 *    `tau=T` 回到初始时刻；时间轴滑的是 tau/T，不是正向物理时间。
 * 2. **宽度律是连续律**：Var(tau)=σ0²+2α tau；实测斜率比与“长 tau 处的偏差量级”
 *    分开报。tau 很大时偏差 1e-2，不等于“扩散律完全不对”。
 * 3. **质心漂移不是扩散**：图上质心几乎不动（机器精度）；宽度才在长。
 *    支持接触边界的那一段用红点标出，此时连续宽度律本来就不该继续成立。
 *
 * 帧 `time.mode="scrub"`：`scene.t` 是 tau/T。
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
  heatLayer,
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
import { CHAPTER, HEAT_RIGHT_GUTTER, SLOT } from "./common.js";
import { manifest } from "./manifest.gen.js";

const VIZ = "influence_field";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "影响场：随反向时间扩散";
const FLOOR_FALLBACK = 1e-16;

export const influenceFieldModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  const influence = requireArray(frame, "influence");
  const position = requireArray(frame, "position");
  const tau = requireArray(frame, "tau");
  const widthMeasured = requireArray(frame, "width_measured");
  const widthTheory = requireArray(frame, "width_theory");
  const widthError = requireArray(frame, "width_error");
  const centerError = requireArray(frame, "center_error");
  const boundary = requireArray(frame, "boundary_flag");
  const now = requireArray(frame, "influence_now");
  const nTau = metaNumber(frame, "n_tau_rows");
  const nX = position.length;
  if (nTau === undefined || nTau * nX !== influence.length) {
    throw new Error(`influence 长度 ${influence.length} 与 n_tau_rows=${nTau}×${nX} 不自洽`);
  }
  requireSameLength(tau, widthMeasured, "tau 与 width_measured");
  requireSameLength(tau, widthTheory, "tau 与 width_theory");
  requireSameLength(tau, widthError, "tau 与 width_error");
  requireSameLength(tau, centerError, "tau 与 center_error");
  requireSameLength(tau, boundary, "tau 与 boundary_flag");
  requireSameLength(position, now, "position 与 influence_now");
  const pitfalls = requirePitfalls(frame);

  const degraded: string[] = [];
  const floor = FLOOR_FALLBACK;
  const tauNow = metaNumber(frame, "tau_current");
  const tauMaxMeta = metaNumber(frame, "tau_max");
  const tauMax = tauMaxMeta ?? (tau[0] as number);
  const frac = metaNumber(frame, "tau_fraction") ?? metaNumber(frame, "t") ?? opts.t;
  const widthNow = metaNumber(frame, "width_current");
  const widthTheoryNow = metaNumber(frame, "width_theory_current");
  const widthRatio = metaNumber(frame, "width_ratio_at_max_tau");
  const widthClean = metaNumber(frame, "width_ratio_clean_max");
  const varSlope = metaNumber(frame, "var_slope_measured");
  const varTheory = metaNumber(frame, "var_slope_theory");
  const varRatio = metaNumber(frame, "var_slope_ratio");
  const centerDrift = metaNumber(frame, "center_drift_max");
  const semiErr = metaNumber(frame, "semidiscrete_rel_error_max");
  const boundaryRows = metaNumber(frame, "boundary_rows");
  const alpha = metaNumber(frame, "alpha");
  const sigma0 = metaNumber(frame, "sigma0");
  const obsCenter = metaNumber(frame, "obs_center");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: frac,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  const L = columnsLayout(scene, width, height, { topShare: 0.62, auxH: 150, gap: 48 });
  const x0 = position[0] as number;
  const x1 = position[position.length - 1] as number;
  const field = makeAxis("field", [L.left, L.topY, Math.max(200, L.leftW - HEAT_RIGHT_GUTTER), L.topH], {
    xlim: [x0, x1],
    ylim: [0, tauMax],
    xlabel: "位置 $x$",
    ylabel: axisYlabelOf(SPEC?.axes, "field", "反向时间 τ"),
    title: axisTitleOf(SPEC?.axes, "field", "影响场 μ(x,τ)（τ=0 为终端）"),
    autoscaled: false,
  });
  const widthAxis = makeAxis("width", [L.left, L.botY, L.leftW, L.botH], {
    xlim: [0, tauMax],
    ylim: [floor, 0.1],
    yscale: "log",
    xlabel: "反向时间 $τ$",
    ylabel: axisYlabelOf(SPEC?.axes, "width", "宽度相对偏差 / 质心漂移"),
    title: axisTitleOf(SPEC?.axes, "width", "宽度偏差与质心漂移（对数）"),
    autoscaled: true,
  });
  scene.axes.push(field, widthAxis);

  // 帧的行序是 tau 递减；heatLayer 的行 0 在 y 最小处，所以翻转行序。
  const flipped = new Float64Array(influence.length);
  for (let j = 0; j < nTau; j += 1) {
    for (let i = 0; i < nX; i += 1) {
      flipped[(nTau - 1 - j) * nX + i] = influence[j * nX + i] as number;
    }
  }
  const vmax = Math.max(1e-12, finiteMinMax(influence)?.[1] ?? 1);
  scene.layers.push(
    heatLayer("field", flipped, nX, nTau, [x0, x1, 0, tauMax], {
      cmap: "viridis",
      vmin: 0,
      vmax,
    }),
  );
  // 当前切片位置：水平线（振幅在右侧 profile 轴里读，不在这条线的长度里）。
  if (tauNow !== undefined && Number.isFinite(tauNow)) {
    scene.layers.push(
      lineLayer("field", { x: [x0, x1], y: [tauNow, tauNow], color: pal.marker, width: 1.4, dash: [4, 3], label: "当前 $τ$ 切片" }),
    );
  }
  // 边界行：支持已经接触定义域边界，连续宽度律在这里不适用。
  const bndX: number[] = [];
  const bndY: number[] = [];
  for (let j = 0; j < tau.length; j += 1) {
    if ((boundary[j] as number) > 0.5) {
      bndX.push(tau[j] as number);
      bndY.push(Math.max(floor, widthError[j] as number));
    }
  }
  if (bndX.length > 0) {
    scene.layers.push(scatterLayer("width", bndX, bndY, { r: 2.2, color: pal.warn, label: "支持接触边界（连续律不适用）" }));
  }
  const posTau = (a: ArrayLike<number>) => {
    const out: number[] = [];
    for (let j = 0; j < a.length; j += 1) out.push(Math.max(floor, a[j] as number));
    return out;
  };
  const centerAbs: number[] = [];
  for (let j = 0; j < centerError.length; j += 1) centerAbs.push(Math.max(floor, Math.abs(centerError[j] as number)));
  scene.layers.push(
    lineLayer("width", { x: tau, y: posTau(widthError), color: pal.curves[0] as string, width: 1.8 }),
    lineLayer("width", { x: tau, y: centerAbs, color: pal.residual, width: 1.5 }),
    lineLayer("width", { x: [0, tauMax], y: [floor, floor], color: pal.muted, width: 1, dash: [3, 3] }),
  );

  // 右上：当前切片剖面。
  let prof: ReturnType<typeof makeAxis> | undefined;
  let profH = 0;
  if (L.rightW >= 200) {
    profH = Math.min(140, L.auxH);
    prof = makeAxis("profile", [L.rightX, L.topY, L.rightW, profH], {
      xlim: [x0, x1],
      ylim: [0, Math.max(1e-12, (finiteMinMax(now)?.[1] ?? 1) * 1.15)],
      xlabel: "位置 $x$",
      ylabel: "$μ$",
      title: `当前切片 τ=${formatNumber(tauNow ?? Number.NaN, 3)}`,
      autoscaled: true,
    });
    scene.axes.push(prof);
    scene.layers.push(lineLayer("profile", { x: position, y: now, color: pal.marker, width: 2 }));
    if (obsCenter !== undefined) {
      scene.layers.push(
        lineLayer("profile", { x: [obsCenter, obsCenter], y: [prof.ylim[0], prof.ylim[1]], color: pal.reference, width: 1, dash: [4, 3] }),
      );
    }
  }

  const budget = unitBudget(L.rightW);
  let textTop = prof !== undefined ? L.topY + profH + 34 : L.topY;
  if (L.rightW >= 200 && L.floor - textTop >= 3 * TEXT_ROW_H) {
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
        `τ/T=${formatNumber(frac, 3)}，τ=${formatNumber(tauNow ?? Number.NaN, 3)}/${formatNumber(tauMax, 3)}`,
        `宽度 ${formatNumber(widthNow ?? Number.NaN, 4)}（连续律 ${formatNumber(widthTheoryNow ?? Number.NaN, 4)}）`,
        `σ0=${formatNumber(sigma0 ?? Number.NaN, 3)}；α=${formatNumber(alpha ?? Number.NaN, 3)}，观测中心 ${formatNumber(obsCenter ?? Number.NaN, 3)}。`,
      ],
      pal.ink,
    );
    push(
      [
        `方差斜率 ${formatNumber(varSlope ?? Number.NaN, 4)}（理论 2α=${formatNumber(varTheory ?? Number.NaN, 4)}）。`,
        `宽度偏差：干净窗口 ${formatNumber(widthClean ?? Number.NaN, 2)}，τ=T 时 ${formatNumber(widthRatio ?? Number.NaN, 2)}。`,
        "长 τ 处才开始偏，量级要照实说。",
        "蓝=宽度偏差，紫=质心漂移；灰虚=地板。",
        "橙点=支持触界；红=切片 μ，灰虚=观测中心。",
        `质心漂移最大 ${formatNumber(centerDrift ?? Number.NaN, 2)}（≈0，扩散在宽度）。`,
        `半离散相对误差 ${formatNumber(semiErr ?? Number.NaN, 2)}；边界行 ${formatNumber(boundaryRows ?? Number.NaN, 0)}。`,
      ],
      pal.muted,
    );
    push(
      [
        "纵轴反向：从上往下=正向时间的逆过程。",
        "宽度律只在支持未触边界时有效。",
        `停止：${stringOf(frame, "status") ?? "?"}。`,
      ],
      pal.muted,
    );
  } else {
    degraded.push("右列过窄/过矮：剖面与读数并入角标");
  }

  scene.badges.push(placeBadge(sourceBadge(dataSourceOf(frame)), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(
      badge(
        "status",
        `τ/T=${formatNumber(frac, 2)}｜宽 ${formatNumber(widthNow ?? Number.NaN, 3)}｜斜率比 ${formatNumber(varRatio ?? Number.NaN, 3)}｜边界 ${formatNumber(boundaryRows ?? Number.NaN, 0)} 行`,
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
