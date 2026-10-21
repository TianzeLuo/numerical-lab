/**
 * `saddle_escape`（5.1 章 viz 3）：鞍点附近的扰动轨迹。
 *
 * 这张图要防的误读是“随机扰动让它逃出来了”：
 * - 逃逸只取决于扰动在**负曲率方向** u_min 上的投影。图上同时画出沿 u_min 的对照轨迹
 *   （`stable_*`，只做正曲率方向的收缩）与当前扰动轨迹；侧栏给出实测重叠 |⟨u_min,d⟩|。
 * - 阈值是**测出来**的（二分），不是解析式抄回来的一个数；`meta.threshold_*` 为 null
 *   表示“放大到 1e12 也没找到逃逸”，不是阈值 0。
 * - 轨迹与残差都只画到 `meta.k_actual`；residual=0 的步写地板值，不是“梯度精确为零”。
 *
 * 帧 `time.mode="scrub"`：`scene.t` 取帧 `meta.t`。
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
  contourLayer,
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
import { CHAPTER, SLOT, cropGrid, indexAxis, prefix, stepIndex } from "./common.js";
import { manifest } from "./manifest.gen.js";

const VIZ = "saddle_escape";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "鞍点逃逸：负曲率方向决定一切";
const FLOOR_FALLBACK = 1e-17;

export const saddleEscapeModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  const cx = requireArray(frame, "contour_x");
  const cy = requireArray(frame, "contour_y");
  const cz = requireArray(frame, "contour_z");
  const trajX = requireArray(frame, "traj_x");
  const trajY = requireArray(frame, "traj_y");
  const residual = requireArray(frame, "residual");
  const distance = requireArray(frame, "error_to_saddle");
  const stableX = requireArray(frame, "stable_x");
  const stableY = requireArray(frame, "stable_y");
  requireSameLength(trajX, trajY, "traj_x 与 traj_y");
  requireSameLength(trajX, residual, "traj_x 与 residual");
  requireSameLength(trajX, distance, "traj_x 与 error_to_saddle");
  requireSameLength(stableX, stableY, "stable_x 与 stable_y");
  if (cz.length !== cx.length * cy.length) {
    throw new Error(`contour_z 长度 ${cz.length} ≠ ${cx.length}×${cy.length}`);
  }
  const pitfalls = requirePitfalls(frame);

  const degraded: string[] = [];
  const t = metaNumber(frame, "t") ?? opts.t;
  const kMeta = metaNumber(frame, "k_actual");
  if (kMeta === undefined) degraded.push("帧 meta 缺 k_actual：按请求时刻取整步");
  const k = Math.max(0, Math.min(trajX.length - 1, kMeta ?? stepIndex(t, trajX.length - 1)));
  const xNow = prefix(trajX, k + 1);
  const yNow = prefix(trajY, k + 1);
  const resNow = prefix(residual, k + 1);
  const distNow = prefix(distance, k + 1);

  const lambdaMin = metaNumber(frame, "lambda_min");
  const overlap = metaNumber(frame, "direction_overlap");
  const delta = metaNumber(frame, "delta");
  const direction = stringOf(frame, "direction") ?? (opts.params.direction as string | undefined) ?? "?";
  const thresholdNeg = metaNumber(frame, "threshold_neg_curv");
  const thresholdRandom = metaNumber(frame, "threshold_random");
  const thresholdRatio = metaNumber(frame, "threshold_ratio");
  const theoryThreshold = metaNumber(frame, "theory_threshold");
  const escapeSteps = metaNumber(frame, "escape_steps");
  const escapedNeg = frame.meta["escaped_neg_curv"] === true;
  const escapedRandom = frame.meta["escaped_random"] === true;
  const finalDist = distNow.length > 0 ? (distNow[distNow.length - 1] as number) : Number.NaN;
  const currentThreshold =
    direction === "neg_curv" ? thresholdNeg : direction === "random" ? thresholdRandom : undefined;
  const deltaAboveThreshold =
    currentThreshold !== undefined && delta !== undefined ? delta >= currentThreshold : undefined;
  // 阈值判据是“N 步内到达半径 R=1”；当前帧可能还没走到 N 步，必须分开报。
  const escapedCurrent = Number.isFinite(finalDist) ? finalDist >= 1 : undefined;

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  const L = columnsLayout(scene, width, height, { topShare: 0.6, auxH: 150, gap: 64 });
  // 视窗对准负曲率方向（x_1）与原点附近的鞍点结构：±2.5 的方窗会把轨迹压成一个点。
  const main = makeAxis("main", [L.left, L.topY, L.leftW, L.topH], {
    xlim: [-0.5, 2.5],
    ylim: [-0.8, 0.8],
    xlabel: "$x_1$（负曲率方向）",
    ylabel: axisYlabelOf(SPEC?.axes, "main", "x_2"),
    title: "鞍点等高线（双曲线族）与扰动轨迹",
    autoscaled: false,
  });

  let auxLo = Number.POSITIVE_INFINITY;
  let auxHi = 0;
  for (let i = 0; i < resNow.length; i += 1) {
    for (const v of [resNow[i] as number, distNow[i] as number]) {
      if (Number.isFinite(v) && v > 0) {
        auxLo = Math.min(auxLo, v);
        auxHi = Math.max(auxHi, v);
      }
    }
  }
  const auxYLo = Math.max(FLOOR_FALLBACK, (Number.isFinite(auxLo) ? auxLo : 1) * 0.3);
  const auxYHi = Math.max(auxYLo * 10, (auxHi > 0 ? auxHi : 1) * 1.5);
  const aux = makeAxis("aux", [L.left, L.botY, L.leftW, L.botH], {
    xlim: [0, Math.max(1, trajX.length - 1)],
    ylim: [auxYLo, auxYHi],
    yscale: "log",
    xlabel: "GD 步 $k$",
    ylabel: axisYlabelOf(SPEC?.axes, "aux", "‖∇f‖ / ‖x - x*‖"),
    title: axisTitleOf(SPEC?.axes, "aux", "梯度范数与到驻点距离（对数）"),
    autoscaled: true,
  });
  scene.axes.push(main, aux);

  // 小量级的对称等高线：鞍点附近 f = 0.5(λ- x_1² + λ+ x_2²)，λ-<0；
  // 只用线性 ±3 的粗级别时，负曲率那一侧的线全落在窗内近乎水平，根本看不出“鞍”。
  const levelMags = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 3];
  const zLo = finiteMinMax(cz)?.[0] ?? -1;
  const zHi = finiteMinMax(cz)?.[1] ?? 1;
  const levels = [...levelMags.map((v) => -v).reverse(), 0, ...levelMags].filter((v) => v >= zLo && v <= zHi);
  const crop = cropGrid(cx, cy, cz, main.xlim, main.ylim);
  if (crop !== undefined) {
    scene.layers.push(
      contourLayer("main", crop.values, crop.nx, crop.ny, [crop.x0, crop.x1, crop.y0, crop.y1], {
        levels,
        color: pal.reference,
        width: 0.8,
        label: "$f(x)=const$（双曲线族）",
      }),
    );
  } else {
    degraded.push("等高线网格与视窗没有交集：底图省略（轨迹仍画）");
  }
  if (stableX.length >= 2) {
    scene.layers.push(
      lineLayer("main", { x: stableX, y: stableY, color: pal.muted, width: 1.4, dash: [3, 3] }),
    );
    scene.layers.push(scatterLayer("main", [stableX[stableX.length - 1] as number], [stableY[stableY.length - 1] as number], { r: 2.4, color: pal.muted, label: "稳定轨迹终点" }));
  }
  if (xNow.length >= 2) {
    scene.layers.push(lineLayer("main", { x: xNow, y: yNow, color: pal.marker, width: 2 }));
  }
  const last = xNow.length - 1;
  if (last >= 0 && Number.isFinite(xNow[last] as number) && Number.isFinite(yNow[last] as number)) {
    scene.layers.push(scatterLayer("main", [xNow[last] as number], [yNow[last] as number], { r: 4, color: pal.marker, label: "当前点" }));
  }

  const step = indexAxis(resNow.length);
  scene.layers.push(
    lineLayer("aux", { x: step, y: resNow, color: pal.residual, width: 1.8 }),
    lineLayer("aux", { x: step, y: distNow, color: pal.marker, width: 1.6 }),
  );
  if (FLOOR_FALLBACK >= auxYLo && FLOOR_FALLBACK <= auxYHi) {
    scene.layers.push(
      lineLayer("aux", { x: [aux.xlim[0], aux.xlim[1]], y: [FLOOR_FALLBACK, FLOOR_FALLBACK], color: pal.warn, width: 1, dash: [3, 3] }),
    );
  }
  {
    const lastAux = resNow.length - 1;
    const xa = lastAux >= 0 ? (step[lastAux] as number) : Number.NaN;
    if (Number.isFinite(xa)) {
      if (Number.isFinite(resNow[lastAux] as number)) {
        scene.layers.push(scatterLayer("aux", [xa], [resNow[lastAux] as number], { r: 3, color: pal.residual, label: "当前 $‖∇f‖$" }));
      }
      if (Number.isFinite(distNow[lastAux] as number)) {
        scene.layers.push(scatterLayer("aux", [xa], [distNow[lastAux] as number], { r: 3, color: pal.marker, label: "当前 $‖x-x*‖$" }));
      }
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
    push(`λ_min=${formatNumber(lambdaMin ?? Number.NaN, 4)}（Hessian 特征值）`, pal.ink);
    push(`u_min=(${formatNumber(metaNumber(frame, "u_min_x") ?? Number.NaN, 2)}, ${formatNumber(metaNumber(frame, "u_min_y") ?? Number.NaN, 2)})`, pal.ink);
    push(`方向 ${direction}，|⟨u_min,d⟩|=${formatNumber(overlap ?? Number.NaN, 3)}`, pal.ink);
    push(`δ=${formatNumber(delta ?? Number.NaN, 4)}，已走 ${k} 步`, pal.ink);
    push(
      currentThreshold === undefined
        ? "当前方向与负曲率正交："
        : `阈值 δ*=${formatNumber(currentThreshold, 4)}（N 步判据）`,
      deltaAboveThreshold === true ? pal.warn : pal.muted,
    );
    push(
      currentThreshold === undefined
        ? "阈值测不到（null≠0）。"
        : `δ=${formatNumber(delta ?? Number.NaN, 4)} ${deltaAboveThreshold === true ? "≥" : "<"} δ*。`,
      deltaAboveThreshold === true ? pal.warn : pal.muted,
    );
    push(`当前 ‖x−x*‖=${formatNumber(finalDist, 3)}（R=1）`, pal.muted);
    push(`→ ${escapedCurrent === true ? "已逃逸" : "尚未逃逸"}`, escapedCurrent === true ? pal.warn : pal.muted);
    if (thresholdNeg !== undefined && thresholdRandom !== undefined) {
      push(`沿 u_min δ*=${formatNumber(thresholdNeg, 4)}；随机 ${formatNumber(thresholdRandom, 4)}。`, pal.muted);
      if (thresholdRatio !== undefined) {
        push(`两者之比 ${formatNumber(thresholdRatio, 3)} = 1/|⟨u_min,d⟩|。`, pal.muted);
        push(`解析 δ*=${formatNumber(theoryThreshold ?? Number.NaN, 4)}。`, pal.muted);
      }
    } else {
      push("沿 u_min / 随机阈值：没测到（null，不是 0）。", pal.muted);
    }
    if (escapeSteps !== undefined) {
      push(`N=${escapeSteps} 步、R=1 的二分测得。`, pal.muted);
      push(`u_min ${escapedNeg ? "能" : "不能"}逃逸，随机方向${escapedRandom ? "能" : "不能"}逃逸。`, pal.muted);
    }
    push("红=轨迹/距离，紫=‖∇f‖；灰虚线=稳定流形。", pal.muted);
    push("特征分解是数值证据，不是参数回抄。", pal.muted);
  } else {
    degraded.push("右侧说明列过窄：阈值读数并入角标");
  }

  scene.badges.push(placeBadge(sourceBadge(dataSourceOf(frame)), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(
      badge(
        "status",
        `k=${k}｜λ_min=${formatNumber(lambdaMin ?? Number.NaN, 3)}｜|⟨u,d⟩|=${formatNumber(overlap ?? Number.NaN, 3)}｜‖x-x*‖=${formatNumber(finalDist, 3)}`,
        8,
        8,
      ),
      SLOT.status,
      width,
    ),
  );
  if (currentThreshold !== undefined) {
    scene.badges.push(
      placeBadge(badge("status", `δ*（${direction}）=${formatNumber(currentThreshold, 4)}；δ=${formatNumber(delta ?? Number.NaN, 4)}`, 8, 8), SLOT.aux, width),
    );
  }
  const dbg = degradedBadge(degraded, SLOT.degraded, width);
  if (dbg !== undefined) scene.badges.push(dbg);

  applyOverflow(scene, 0, { slot: SLOT.overflow, width });
  requireDrawableLayers(scene);
  return scene;
};
