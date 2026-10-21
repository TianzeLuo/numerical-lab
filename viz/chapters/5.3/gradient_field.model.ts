/**
 * `gradient_field`（5.3 章 viz 1）：参数平面上的目标函数与伴随梯度方向。
 *
 * 三条纪律
 * -------
 * 1. **箭头长度不是灵敏度**：`arrow_length` 是对梯度范数做对数压缩后的显示长度；
 *    真正的量级在 `grad_norm` 与侧栏里。跨量级比较请用弹性 (dJ/dp)(p/J)，
 *    它才是无量纲的“哪个参数更灵敏”。
 * 2. **方向不是证据，误差才是**：伴随方向与中心差分方向在图上几乎重合是必然的；
 *    证据是逐点相对误差（底栏，必须 < 1e-6）。
 * 3. **共振脊**：p1=1 附近目标函数骤增；等高线用对数级别，否则脊会糊成一片。
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
  contourLayer,
  criticalBadge,
  heatLayer,
  lineLayer,
  makeAxis,
  scatterLayer,
  sourceBadge,
  vectorFieldLayer,
} from "../../shared/src/scene.js";
import {
  TEXT_ROW_H,
  axisTitleOf,
  axisYlabelOf,
  columnsLayout,
  textBlock,
  unitBudget,
} from "../_common/panel.js";
import { CHAPTER, HEAT_RIGHT_GUTTER, SLOT, logArrowLength } from "./common.js";
import { manifest } from "./manifest.gen.js";

const VIZ = "gradient_field";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "参数平面上的梯度场";

export const gradientFieldModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  const objective = requireArray(frame, "objective");
  const elasticity = requireArray(frame, "elasticity");
  const gradCenter = requireArray(frame, "gradient_center");
  const gp1 = requireArray(frame, "grid_p1");
  const gp2 = requireArray(frame, "grid_p2");
  const ap1 = requireArray(frame, "arrow_p1");
  const ap2 = requireArray(frame, "arrow_p2");
  const d1 = requireArray(frame, "adjoint_dir1");
  const d2 = requireArray(frame, "adjoint_dir2");
  const fd1 = requireArray(frame, "fd_dir1");
  const fd2 = requireArray(frame, "fd_dir2");
  const aLen = requireArray(frame, "arrow_length");
  const gNorm = requireArray(frame, "grad_norm");
  const lgErr = requireArray(frame, "rel_error_log10");
  const relErr = requireArray(frame, "rel_error");
  requireSameLength(ap1, ap2, "arrow_p1 与 arrow_p2");
  for (const [name, arr, ref] of [
    ["adjoint_dir1", d1, ap1],
    ["adjoint_dir2", d2, ap1],
    ["fd_dir1", fd1, ap1],
    ["fd_dir2", fd2, ap1],
    ["arrow_length", aLen, ap1],
    ["grad_norm", gNorm, ap1],
    ["rel_error_log10", lgErr, ap1],
    ["rel_error", relErr, ap1],
  ] as const) {
    requireSameLength(ap1, arr, `arrow_p1 与 ${name}`);
    void ref;
  }
  const n1 = metaNumber(frame, "n_grid_p1");
  const n2 = metaNumber(frame, "n_grid_p2");
  if (n1 === undefined || n2 === undefined) throw new Error("帧 meta 缺 n_grid_p1/n_grid_p2：二维网格不能猜");
  if (objective.length !== n1 * n2) throw new Error(`objective 长度 ${objective.length} ≠ ${n1}×${n2}`);
  if (elasticity.length !== 2 || gradCenter.length !== 2) throw new Error("elasticity/gradient_center 必须是 2 分量");
  const nArrows = metaNumber(frame, "n_arrows");
  const m = nArrows ?? ap1.length;
  const aN1 = Math.round(Math.sqrt(m));
  const aN2 = aN1 > 0 && aN1 * aN1 === m ? aN1 : m;
  if (aN1 * aN2 !== m) throw new Error(`箭头数 ${m} 不是可重构的二维网格`);
  const pitfalls = requirePitfalls(frame);

  const degraded: string[] = [];
  const maxErr = metaNumber(frame, "max_rel_error");
  const medErr = metaNumber(frame, "median_rel_error");
  const minErr = metaNumber(frame, "min_rel_error");
  const gMax = metaNumber(frame, "grad_norm_max");
  const gMin = metaNumber(frame, "grad_norm_min");
  const objMin = metaNumber(frame, "objective_min");
  const objMax = metaNumber(frame, "objective_max");
  const centerP1 = metaNumber(frame, "stiffness_center");
  const centerP2 = metaNumber(frame, "damping_center");
  const resonanceP1 = metaNumber(frame, "resonance_p1");
  const nGridSolves = metaNumber(frame, "n_grid_solves");
  const nArrowSolves = metaNumber(frame, "n_arrow_solves");
  const rhsEvals = metaNumber(frame, "rhs_evals");
  const dominantRaw = frame.meta["dominant_raw"];
  const dominantElasticity = frame.meta["dominant_elasticity"];
  const paramName = (v: unknown): string => (v === 0 ? "p1" : v === 1 ? "p2" : "?");
  if (maxErr === undefined) degraded.push("帧 meta 缺 max_rel_error：无法给伴随梯度正确性下结论");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: 0,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  const L = columnsLayout(scene, width, height, { topShare: 0.64, auxH: 120, gap: 48 });
  const p1lo = gp1[0] as number;
  const p1hi = gp1[gp1.length - 1] as number;
  const p2lo = gp2[0] as number;
  const p2hi = gp2[gp2.length - 1] as number;

  const plane = makeAxis("plane", [L.left, L.topY, Math.max(200, L.leftW - HEAT_RIGHT_GUTTER), L.topH], {
    xlim: [p1lo, p1hi],
    ylim: [p2lo, p2hi],
    xlabel: "$p1$（刚度）",
    ylabel: axisYlabelOf(SPEC?.axes, "plane", "p2（阻尼）"),
    title: axisTitleOf(SPEC?.axes, "plane", "目标函数等高线与伴随梯度方向"),
    autoscaled: false,
  });
  const errAxis = makeAxis("error", [L.left, L.botY, L.leftW, L.botH], {
    xlim: [1, Math.max(2, m)],
    ylim: [-12, -4],
    xlabel: "箭头序号（$4×4$ 子网格）",
    ylabel: axisYlabelOf(SPEC?.axes, "error", "log10 相对误差"),
    title: axisTitleOf(SPEC?.axes, "error", "伴随 vs 中心差分（必须 < −6）"),
    autoscaled: true,
  });
  scene.axes.push(plane, errAxis);

  scene.layers.push(
    heatLayer("plane", objective, n1, n2, [p1lo, p1hi, p2lo, p2hi], {
      cmap: "viridis",
      logScale: true,
    }),
  );
  const levels: number[] = [];
  const lmin = Math.log10(Math.max(objMin ?? 1e-4, 1e-4));
  const lmax = Math.log10(Math.max(objMax ?? 1, 1e-3));
  for (let i = 0; i < 14; i += 1) levels.push(10 ** (lmin + ((lmax - lmin) * i) / 13));
  scene.layers.push(contourLayer("plane", objective, n1, n2, [p1lo, p1hi, p2lo, p2hi], { levels, color: pal.muted, width: 0.9 }));

  // 箭头：方向是伴随梯度，长度取帧里的对数压缩值（不是灵敏度大小）。
  const u = new Float64Array(m);
  const v = new Float64Array(m);
  const scale = 0.05 * (p1hi - p1lo);
  for (let i = 0; i < m; i += 1) {
    const len = Number.isFinite(aLen[i] as number) ? (aLen[i] as number) : logArrowLength(gNorm[i] as number, gMin ?? 1e-6, gMax ?? 1);
    u[i] = (d1[i] as number) * len * scale;
    v[i] = (d2[i] as number) * len * scale;
  }
  scene.layers.push(vectorFieldLayer("plane", u, v, aN1, aN2, [ap1[0] as number, ap1[ap1.length - 1] as number, ap2[0] as number, ap2[ap2.length - 1] as number], { color: pal.ink, scale: 1 }));
  // FD 方向只画一小段虚线对照，避免完全盖住伴随箭头（方向应重合，证据在底栏）。
  const uf = new Float64Array(m);
  const vf = new Float64Array(m);
  for (let i = 0; i < m; i += 1) {
    uf[i] = (fd1[i] as number) * scale * 0.5;
    vf[i] = (fd2[i] as number) * scale * 0.5;
  }
  scene.layers.push(vectorFieldLayer("plane", uf, vf, aN1, aN2, [ap1[0] as number, ap1[ap1.length - 1] as number, ap2[0] as number, ap2[ap2.length - 1] as number], { color: pal.exact, scale: 1 }));

  if (resonanceP1 !== undefined && resonanceP1 > p1lo && resonanceP1 < p1hi) {
    scene.layers.push(
      lineLayer("plane", { x: [resonanceP1, resonanceP1], y: [p2lo, p2hi], color: pal.warn, width: 1.2, dash: [4, 3] }),
    );
  }
  if (centerP1 !== undefined && centerP2 !== undefined) {
    scene.layers.push(
      scatterLayer("plane", [centerP1], [centerP2], { r: 4.2, color: pal.marker, label: "滑块工作点（参数中心）" }),
    );
  }

  // 底栏：逐点相对误差必须低于 1e-6，且不能只看“方向差不多”。
  const idx: number[] = [];
  for (let i = 0; i < m; i += 1) idx.push(i + 1);
  scene.layers.push(
    scatterLayer("error", idx, lgErr, { r: 3, color: pal.residual, label: "$\\log_{10}$ 相对误差" }),
    lineLayer("error", { x: [1, Math.max(2, m)], y: [-6, -6], color: pal.warn, width: 1.4, dash: [4, 3] }),
  );
  if (medErr !== undefined && medErr > 0) {
    scene.layers.push(
      lineLayer("error", { x: [1, Math.max(2, m)], y: [Math.log10(medErr), Math.log10(medErr)], color: pal.exact, width: 1.1, dash: [5, 3] }),
    );
  }

  const budget = unitBudget(L.rightW);
  if (L.rightW >= 200) {
    let y = L.topY;
    const push = (lines: readonly string[], color: string): void => {
      const block = textBlock(L.rightX, y, L.floor, lines, color, budget, "top");
      scene.texts.push(...block);
      y += block.length * TEXT_ROW_H;
    };
    push(
      [
        `工作点 (p1,p2)=(${formatNumber(centerP1 ?? Number.NaN, 3)}, ${formatNumber(centerP2 ?? Number.NaN, 3)})`,
        `J 范围 [${formatNumber(objMin ?? Number.NaN, 3)}, ${formatNumber(objMax ?? Number.NaN, 3)}]；p1=1 是共振脊。`,
        `工作点梯度 (${formatNumber(gradCenter[0] as number, 3)}, ${formatNumber(gradCenter[1] as number, 3)})。`,
        "热图=J（对数色标）；灰线=等值线。",
        "黑箭头=伴随方向；橙短线=中心差分。",
      ],
      pal.ink,
    );
    push(
      [
        `弹性（无量纲）：p1 ${formatNumber(elasticity[0] as number, 3)}，p2 ${formatNumber(elasticity[1] as number, 3)}`,
        `最灵敏参数：原始 ${paramName(dominantRaw)}，弹性 ${paramName(dominantElasticity)}。`,
        "两个答案可以不同，差的是量纲。",
        `梯度范数 ${formatNumber(gMin ?? Number.NaN, 2)}…${formatNumber(gMax ?? Number.NaN, 2)}；箭头长度已对数压缩。`,
        "底栏：橙虚=及格线 −6；橙实=实测中位数。",
      ],
      pal.muted,
    );
    push(
      [
        `伴随 vs 中心差分：最小 ${formatNumber(minErr ?? Number.NaN, 2)}。`,
        `中位 ${formatNumber(medErr ?? Number.NaN, 2)}；最大 ${formatNumber(maxErr ?? Number.NaN, 2)}。`,
        `网格 ${formatNumber(nGridSolves ?? Number.NaN, 0)} 次 + 箭头 ${formatNumber(nArrowSolves ?? Number.NaN, 0)} 次求解；RHS ${formatNumber(rhsEvals ?? Number.NaN, 0)}。`,
        "方向重合不算证据；看底栏误差。",
        `停止：${stringOf(frame, "status") ?? "?"}。`,
      ],
      maxErr !== undefined && maxErr > 1e-6 ? pal.warn : pal.muted,
    );
  } else {
    degraded.push("右侧说明列过窄：读数并入角标");
  }

  scene.badges.push(placeBadge(sourceBadge(dataSourceOf(frame)), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(
      badge(
        "status",
        `max 相对误差 ${formatNumber(maxErr ?? Number.NaN, 2)}｜中位 ${formatNumber(medErr ?? Number.NaN, 2)}｜箭头 ${m} 个`,
        8,
        8,
      ),
      SLOT.status,
      width,
    ),
  );
  if (maxErr !== undefined && maxErr > 0) {
    scene.badges.push(placeBadge(criticalBadge("log10 误差", Math.log10(maxErr), -6, { beyond: "above" }), SLOT.critical, width));
  }
  const dbg = degradedBadge(degraded, SLOT.degraded, width);
  if (dbg !== undefined) scene.badges.push(dbg);

  applyOverflow(scene, 0, { slot: SLOT.overflow, width });
  requireDrawableLayers(scene);
  return scene;
};
