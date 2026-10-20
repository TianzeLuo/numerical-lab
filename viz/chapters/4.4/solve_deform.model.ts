/**
 * `solve_deform`（4.4 章 viz 2）的 model：拖动载荷位置的准静态求解。
 *
 * 面板与图层
 * ----------
 * | 面板 | 轴 | 内容 |
 * |---|---|---|
 * | 左上 | `deformed` | 梁：FE 解曲线 + 精确节点 + 未变形参考虚线；膜：单元边线（变形/未变形）+ 精确节点 |
 * | 左下 | `scan` | `error_energy_scan`（requiredAux）+ 梁的 `nodal_error_scan` + 当前载荷标记 + 地板线 |
 * | 右上 | `error` | 当前节点误差 `error`（requiredAux；对数轴） |
 *
 * 三条防误导纪律
 * --------------
 * 1. **奇异就承认**：`singular_mask=1` 的点不进位移/误差图层，`meta.singular` 为真时
 *    只画参考网格与精确解，并显式写出 `fail_reason`——不拿 0 位移冒充算出来的解。
 * 2. **地板点不画成数据**：`scan_at_floor=1` 的节点误差点被排除；地板线单独画，
 *    图例报告被排除的点数。
 * 3. **误差轴与解轴分开**：节点误差是 ±0.02 位移上的 1e-15 量级，混在位移轴上只会变成
 *    一条贴轴的线；这里给它自己的对数轴（requiredAux 必须可见）。
 *
 * 时间轴：`scrub`，`t` = 加载因子 λ（`meta.load_factor` / `t_actual` 优先）。
 */

import { formatNumber } from "../../shared/src/axes.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import {
  applyOverflow,
  canvasSize,
  clamp,
  criticalOf,
  dataSourceOf,
  degradedBadge,
  metaNumber,
  ModelInputError,
  palette,
  paramSummary,
  placeBadge,
  requireArray,
  requireDrawableLayers,
  requirePitfalls,
  requireSameLength,
  sceneShell,
  stringOf,
  toNumeric,
  type SceneModel,
} from "../../shared/src/model.js";
import {
  axisModeBadge,
  badge,
  criticalBadge,
  lineLayer,
  makeAxis,
  scatterLayer,
  series,
  sourceBadge,
  type LineSeries,
} from "../../shared/src/scene.js";
import {
  SAFE_GAP,
  TEXT_ROW_H,
  axisTitleOf,
  axisYlabelOf,
  columnsLayout,
  linearRange,
  textBlock,
  unitBudget,
} from "../_common/panel.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "4.4";
const VIZ = "solve_deform";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "拖动载荷：梁与膜的实时求解与误差锯齿";

const FLOOR_FALLBACK = 1e-17;

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

interface MatrixView {
  rows: number;
  cols: number;
  values: ArrayLike<number>;
}

/** `frame.matrix` 的数值视图：`i8` 走 `toNumeric`，避免 BigInt 进图层。 */
function numericMatrix(frame: Parameters<SceneModel>[0], name: string): MatrixView {
  requireArray(frame, name);
  const m = frame.matrix(name);
  return { rows: m.rows, cols: m.cols, values: toNumeric(m.values) };
}

function finiteSmall(a: ArrayLike<number>, accept: (v: number) => boolean): number[] {
  const out: number[] = [];
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i] as number;
    if (Number.isFinite(v) && accept(v)) out.push(i);
  }
  return out;
}

function pick(a: ArrayLike<number>, idx: readonly number[]): Float64Array {
  const out = new Float64Array(idx.length);
  idx.forEach((i, k) => {
    out[k] = a[i] as number;
  });
  return out;
}

/** 一条多段折线（每段闭合到起点）；用于把二维单元边线画成一条 line 图层。 */
function meshEdges(
  mesh: MatrixView,
  elements: MatrixView,
  offset: ArrayLike<number> | undefined,
  color: string,
  width: number,
  dash?: number[],
  alpha = 1,
): LineSeries[] {
  const out: LineSeries[] = [];
  const cols = elements.cols;
  for (let e = 0; e < elements.rows; e += 1) {
    const xs: number[] = [];
    const ys: number[] = [];
    for (let k = 0; k <= cols; k += 1) {
      const kk = k === cols ? 0 : k; // 闭合：最后一个点回到第一个顶点
      const node = elements.values[e * cols + kk] as number;
      if (!Number.isInteger(node) || node < 0 || node >= mesh.rows) {
        xs.length = 0;
        break;
      }
      const x = mesh.values[node * 2] as number;
      const y = mesh.values[node * 2 + 1] as number;
      const d = offset !== undefined ? (offset[node] as number) : 0;
      if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(d)) {
        xs.length = 0;
        break;
      }
      xs.push(x);
      ys.push(y + d);
    }
    if (xs.length === cols + 1) {
      out.push(series(Float64Array.from(xs), Float64Array.from(ys), color, { width, dash, alpha }));
    }
  }
  return out;
}

export const solveDeformModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需数组 ——
  const mesh = numericMatrix(frame, "mesh_xy");
  if (mesh.cols !== 2) throw new ModelInputError(`帧 ${CHAPTER}/${VIZ} 的 mesh_xy 第二维必须是 2，得到 ${mesh.cols}`);
  const nNode = mesh.rows;
  const elements = numericMatrix(frame, "elements");
  const solution = requireArray(frame, "solution");
  const exact = requireArray(frame, "solution_exact");
  const error = requireArray(frame, "error");
  const scanX = requireArray(frame, "scan_x0");
  const scanEnergy = requireArray(frame, "error_energy_scan");
  const singularMask = requireArray(frame, "singular_mask");
  const markX0 = requireArray(frame, "mark_x0");
  const markEnergy = requireArray(frame, "mark_energy");
  requireSameLength(solution, exact, "帧数组 solution 与 solution_exact");
  requireSameLength(solution, error, "帧数组 solution 与 error");
  requireSameLength(solution, singularMask, "帧数组 solution 与 singular_mask");
  if (solution.length !== nNode) {
    throw new ModelInputError(`帧 ${CHAPTER}/${VIZ} 的 solution 长度 ${solution.length} ≠ mesh_xy 顶点数 ${nNode}`);
  }
  requireSameLength(scanX, scanEnergy, "帧数组 scan_x0 与 error_energy_scan");
  requireSameLength(markX0, markEnergy, "帧数组 mark_x0 与 mark_energy");
  if (markX0.length === 0) throw new ModelInputError(`帧 ${CHAPTER}/${VIZ} 的 mark_x0 为空：无法标当前载荷位置`);
  const pitfalls = requirePitfalls(frame);

  // —— 模型类型：strings.kind 优先，其次 meta.model_index，最后请求参数 ——
  const degraded: string[] = [];
  const kindText = stringOf(frame, "kind");
  const modelIdx = metaNumber(frame, "model_index");
  const membrane =
    kindText !== undefined
      ? kindText === "membrane"
      : modelIdx !== undefined
        ? Math.round(modelIdx) === 1
        : opts.params.model === "membrane";
  if (kindText === undefined && modelIdx === undefined) {
    degraded.push("帧缺 strings.kind / meta.model_index：模型类型按请求参数判定");
  }
  // 梁的 elements 是自由度编号（0..2·n_elem），膜的才是顶点编号；只在膜上做越界检查。
  if (membrane) {
    for (let i = 0; i < elements.values.length; i += 1) {
      const k = elements.values[i] as number;
      if (!Number.isInteger(k) || k < 0 || k >= nNode) {
        throw new ModelInputError(`帧 ${CHAPTER}/${VIZ} 的 elements 含越界顶点编号 ${k}（n_node=${nNode}）`);
      }
    }
  }
  const singularFlag = frame.meta["singular"] === true;
  let maskedNodes = 0;
  for (let i = 0; i < singularMask.length; i += 1) if ((singularMask[i] ?? 0) > 0.5) maskedNodes += 1;
  const singular = singularFlag || maskedNodes > 0;
  if (singularFlag !== maskedNodes > 0 && !singularFlag) {
    degraded.push("singular_mask 全为 0 但 meta.singular=true：按奇异处理（不画位移）");
  }

  // —— 可选数组 ——
  let curve: MatrixView | undefined;
  if (!membrane) {
    requireArray(frame, "solution_curve");
    const shape = frame.shapeOf("solution_curve");
    if (shape.length !== 2 || shape[1] !== 2) {
      throw new ModelInputError(`帧 ${CHAPTER}/${VIZ} 的 solution_curve 形状 [${shape.join(", ")}] 不是 (m, 2)`);
    }
    curve = numericMatrix(frame, "solution_curve");
    if (curve.values.length === 0) degraded.push("solution_curve 为空：只画节点位移");
  }
  const nodalScan = membrane ? undefined : requireArray(frame, "nodal_error_scan");
  if (nodalScan !== undefined) requireSameLength(scanX, nodalScan, "帧数组 scan_x0 与 nodal_error_scan");
  const scanFloorMask = frame.has("scan_at_floor") ? requireArray(frame, "scan_at_floor") : undefined;
  if (scanFloorMask !== undefined) requireSameLength(scanX, scanFloorMask, "帧数组 scan_x0 与 scan_at_floor");
  else if (nodalScan !== undefined) degraded.push("帧缺 scan_at_floor：节点误差地板点仍画在曲线上（不标已到地板）");
  const rotation = frame.has("rotation") ? requireArray(frame, "rotation") : undefined;
  if (rotation !== undefined) requireSameLength(solution, rotation, "帧数组 solution 与 rotation");
  const errorElement = frame.has("error_element") ? requireArray(frame, "error_element") : undefined;
  if (errorElement !== undefined && errorElement.length !== elements.rows) {
    degraded.push("error_element 长度与单元数不符：不画逐单元误差");
  }

  const floorText = stringOf(frame, "floor");
  const floor = floorText !== undefined && Number(floorText) > 0 ? Number(floorText) : FLOOR_FALLBACK;
  if (floorText === undefined) degraded.push("帧 strings 缺 floor：地板按 1e-17 显示");

  // —— 时间 ——
  const tMeta = metaNumber(frame, "t_actual") ?? metaNumber(frame, "load_factor") ?? metaNumber(frame, "t");
  const lam = clamp(tMeta ?? opts.t, 0, 1);
  if (tMeta === undefined) degraded.push("帧 meta 缺 t_actual/load_factor：加载因子取请求值并夹到 [0,1]");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: lam,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  const L = columnsLayout(scene, width, height, { topShare: 0.6, auxH: 150, gap: 56 });
  const narrow = !L.columns;
  const short = L.floor - L.topY < 250;
  const topH = narrow || short ? Math.max(60, L.floor - L.topY - 28) : L.topH;
  const mainW = narrow ? width - L.left - 24 : L.leftW;

  // —— 左上：变形构型 ——
  const xVals: number[] = [];
  const yVals: number[] = [0];
  for (let i = 0; i < nNode; i += 1) {
    const x = mesh.values[i * 2] as number;
    const y = mesh.values[i * 2 + 1] as number;
    if (Number.isFinite(x)) xVals.push(x);
    if (Number.isFinite(y)) yVals.push(y);
    const u = solution[i] as number;
    const e = exact[i] as number;
    if (Number.isFinite(y) && Number.isFinite(u) && !singular) yVals.push(y + u);
    if (Number.isFinite(y) && Number.isFinite(e)) yVals.push(y + e);
  }
  if (curve !== undefined) {
    for (let i = 0; i < curve.values.length; i += 2) {
      const y = curve.values[i + 1] as number;
      if (Number.isFinite(y)) yVals.push(y);
    }
  }
  const xLo = xVals.length > 0 ? Math.min(...xVals) : 0;
  const xHi = xVals.length > 0 ? Math.max(...xVals) : 1;
  const yLo = yVals.length > 0 ? Math.min(...yVals) : 0;
  const yHi = yVals.length > 0 ? Math.max(...yVals) : 1;
  const xPad = Math.max(1e-6, xHi - xLo) * 0.02;
  const deformedAxis = makeAxis("deformed", [L.left, L.topY, mainW, topH], {
    xlim: [xLo - xPad, Math.max(xLo + 1e-6, xHi) + xPad],
    ylim: linearRange(yLo, yHi, 0.1),
    xlabel: "x",
    title: axisTitleOf(SPEC?.axes, "deformed", "变形后的构型（自动缩放）"),
    ylabel: membrane ? "y+u_h（位移）" : axisYlabelOf(SPEC?.axes, "deformed", "位移"),
    autoscaled: true,
  });
  scene.axes.push(deformedAxis);

  // 未变形参考：梁是 y=0；膜是原始网格的单元边线（灰虚线）。
  if (!membrane) {
    const refX = new Float64Array(nNode);
    const refY = new Float64Array(nNode);
    for (let i = 0; i < nNode; i += 1) {
      refX[i] = mesh.values[i * 2] as number;
      refY[i] = mesh.values[i * 2 + 1] as number;
    }
    scene.layers.push(
      lineLayer("deformed", series(refX, refY, pal.reference, { width: 1.2, dash: [5, 4], label: "未变形" })),
    );
  } else {
    const ref = meshEdges(mesh, elements, undefined, pal.reference, 0.9, [4, 4], 0.55);
    if (ref.length > 0) scene.layers.push(lineLayer("deformed", ref));
  }

  // 精确解（始终画）：奇异时它是唯一可信的参照。
  if (membrane) {
    const exX: number[] = [];
    const exY: number[] = [];
    for (let i = 0; i < nNode; i += 1) {
      const x = mesh.values[i * 2] as number;
      const y = mesh.values[i * 2 + 1] as number;
      const e = exact[i] as number;
      if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(e)) {
        exX.push(x);
        exY.push(y + e);
      }
    }
    if (exX.length > 0) scene.layers.push(scatterLayer("deformed", exX, exY, { r: 1.1, color: pal.exact, alpha: 0.45 }));
  } else {
    const good: number[] = [];
    for (let i = 0; i < nNode; i += 1) {
      if (Number.isFinite(exact[i] as number)) good.push(i);
    }
    if (good.length > 0) {
      scene.layers.push(scatterLayer("deformed", pick(meshX(mesh), good), pick(exact, good), { r: 2.6, color: pal.exact, alpha: 0.9 }));
    }
  }

  // FE 解：奇异时不画（singular_mask 的诚实用法）。
  if (!singular) {
    if (membrane) {
      const lines = meshEdges(mesh, elements, solution, pal.curves[0], 1.05, undefined, 0.9);
      if (lines.length > 0) scene.layers.push(lineLayer("deformed", lines));
    } else if (curve !== undefined) {
      const xs = new Float64Array(curve.rows);
      const ys = new Float64Array(curve.rows);
      for (let i = 0; i < curve.rows; i += 1) {
        xs[i] = curve.values[i * 2] as number;
        ys[i] = curve.values[i * 2 + 1] as number;
      }
      scene.layers.push(lineLayer("deformed", series(xs, ys, pal.curves[0], { width: 2.4, label: "FE 解" })));
      const good: number[] = [];
      for (let i = 0; i < nNode; i += 1) {
        if ((singularMask[i] ?? 0) <= 0.5 && Number.isFinite(solution[i] as number)) good.push(i);
      }
      if (good.length > 0) {
        scene.layers.push(scatterLayer("deformed", pick(meshX(mesh), good), pick(solution, good), { r: 2.2, color: pal.curves[0] }));
      }
    }
  }

  const loadX = markX0[0] as number;
  if (Number.isFinite(loadX) && loadX >= deformedAxis.xlim[0] && loadX <= deformedAxis.xlim[1]) {
    scene.layers.push(
      lineLayer("deformed", series([loadX, loadX], deformedAxis.ylim, pal.marker, { width: 1.1, dash: [4, 4], label: "载荷位置" })),
    );
    if (!membrane) {
      scene.layers.push(scatterLayer("deformed", [loadX], [0], { r: 3.4, color: pal.marker }));
    }
  }

  // 面板图例：梁放在左上（曲线下方、上方为空）；膜放在左下。
  if (!membrane && !narrow && !short) {
    const lines = singular
      ? ["橙=精确解　灰虚=未变形梁　红虚=载荷位置（FE 位移不画）"]
      : ["蓝=FE 解　橙=精确节点　灰虚=未变形　红虚=载荷位置"];
    scene.texts.push(
      ...textBlock(
        L.left + 6,
        L.topY + 6,
        L.topY + 6 + 3 * TEXT_ROW_H,
        lines,
        pal.muted,
        unitBudget(mainW),
        "top",
      ),
    );
  }

  // —— 左下：扫载荷位置（宽面板，锯齿）——
  let scanAxis: ReturnType<typeof makeAxis> | undefined;
  if (!narrow && !short) {
    scanAxis = makeAxis("scan", [L.left, L.botY, L.leftW, L.botH], {
      xlim: [0, 1],
      ylim: [floor, 1],
      yscale: "log",
      xlabel: "载荷位置 x0",
      title: axisTitleOf(SPEC?.axes, "scan", "扫载荷位置的误差（对数，地板 1e-17）"),
      ylabel: axisYlabelOf(SPEC?.axes, "scan", "误差"),
      autoscaled: false,
    });
    scene.axes.push(scanAxis);
    scene.layers.push(
      lineLayer("scan", series([0, 1], [floor, floor], pal.reference, { width: 1, dash: [3, 3], label: "能量误差地板" })),
    );
    if (!singular) {
      scene.layers.push(
        lineLayer("scan", series(scanX, scanEnergy, pal.curves[0], { width: 2.1, marker: "o", label: "能量误差" })),
      );
      if (nodalScan !== undefined) {
        const maskIdx = finiteSmall(scanX, () => true);
        const kept: number[] = [];
        for (const i of maskIdx) {
          const masked = scanFloorMask !== undefined && (scanFloorMask[i] ?? 0) > 0.5;
          if (!masked) kept.push(i);
        }
        if (kept.length > 0) {
          scene.layers.push(
            lineLayer("scan", series(pick(scanX, kept), pick(nodalScan, kept), pal.exact, { width: 1.6, marker: "s", dash: [4, 3], label: "节点误差" })),
          );
        }
      }
      if (Number.isFinite(loadX) && Number.isFinite(markEnergy[0] as number)) {
        const energyNow = markEnergy[0] as number;
        if (energyNow > 0) {
          scene.layers.push(
            lineLayer("scan", series([loadX, loadX], [floor, Math.max(floor, Math.min(energyNow, 1))], pal.marker, { width: 1, dash: [4, 4] })),
          );
          scene.layers.push(scatterLayer("scan", [loadX], [Math.max(floor, energyNow)], { r: 3.4, color: pal.marker }));
        }
      }
    } else {
      scene.texts.push(
        ...textBlock(
          L.left + 6,
          L.botY + 6,
          L.botY + 6 + 2 * TEXT_ROW_H,
          ["奇异：零解的扫描误差不画（singular_mask）"],
          pal.muted,
          unitBudget(L.leftW),
          "top",
        ),
      );
    }
    if (!singular) {
      const lines = ["蓝=能量误差　橙=节点误差", "红=当前 x0　灰虚=地板"];
      const legendBottom = membrane;
      const legendTop = legendBottom
        ? L.botY + L.botH - SAFE_GAP - 6 * TEXT_ROW_H
        : L.botY + 6;
      const legendEnd = legendBottom ? L.botY + L.botH - SAFE_GAP - TEXT_ROW_H : L.botY + 6 + 3 * TEXT_ROW_H;
      scene.texts.push(
        ...textBlock(
          L.left + 6,
          legendTop,
          legendEnd,
          lines,
          pal.muted,
          legendBottom ? unitBudget(L.leftW) : Math.min(34, unitBudget(L.leftW)),
          legendBottom ? "bottom" : "top",
        ),
      );
    }
  }

  // —— 右上：节点误差（requiredAux；独立对数轴）——
  let errorAxis: ReturnType<typeof makeAxis> | undefined;
  if (!narrow && !short) {
    errorAxis = makeAxis("error", [L.rightX, L.topY, L.rightW, L.auxH], {
      xlim: [0, Math.max(1, nNode - 1)],
      ylim: [floor, 1],
      yscale: "log",
      xlabel: "节点序号 i",
      title: "节点误差 |u_h−u|（对数）",
      ylabel: "|u_h−u|",
      autoscaled: false,
    });
    scene.axes.push(errorAxis);
    scene.layers.push(
      lineLayer("error", series([0, Math.max(1, nNode - 1)], [floor, floor], pal.reference, { width: 1, dash: [3, 3], label: "节点误差地板" })),
    );
    const good: number[] = [];
    for (let i = 0; i < nNode; i += 1) {
      if ((singularMask[i] ?? 0) > 0.5) continue;
      const v = error[i] as number;
      if (Number.isFinite(v) && v > 0) good.push(i);
    }
    if (good.length > 0 && !singular) {
      const ex = new Float64Array(good);
      if (membrane) {
        scene.layers.push(scatterLayer("error", ex, pick(error, good), { r: 1.8, color: pal.exact, alpha: 0.75 }));
      } else {
        scene.layers.push(
          lineLayer("error", series(ex, pick(error, good), pal.exact, { width: 1.6, marker: "o", label: "节点误差 |u_h − u|" })),
        );
      }
    }
    scene.texts.push(
      ...textBlock(
        L.rightX + 6,
        L.topY + L.auxH - SAFE_GAP - 3 * TEXT_ROW_H,
        L.topY + L.auxH - SAFE_GAP - TEXT_ROW_H,
        [singular ? "奇异：解不可用，误差不画" : "橙=节点误差（对数）　灰虚=地板"],
        pal.muted,
        unitBudget(L.rightW),
        "bottom",
      ),
    );
  }

  // —— 右列文字 ——
  const x0 = metaNumber(frame, "x0");
  const nodeDistance = metaNumber(frame, "node_distance");
  const loadOnNode = frame.meta["load_on_node"] === true;
  const h2 = singular ? undefined : membrane ? metaNumber(frame, "h1") : metaNumber(frame, "h2");
  const errLabel = membrane ? "H¹" : "H²";
  const energyNow = singular ? undefined : metaNumber(frame, "energy");
  const l2Now = singular ? undefined : metaNumber(frame, "l2");
  const nodalNow = singular ? undefined : metaNumber(frame, "nodal_error");
  const cond = metaNumber(frame, "condition");
  const residual = metaNumber(frame, "residual");
  const wMax = metaNumber(frame, "w_max");
  const scanMin = metaNumber(frame, "scan_min_error");
  const scanMax = metaNumber(frame, "scan_max_error");
  const peak = metaNumber(frame, "scan_peak_to_peak");
  const floorCount = metaNumber(frame, "scan_at_floor_count");
  const failReason = stringOf(frame, "fail_reason");

  if (L.columns && !short && L.floor > L.textTop + 8) {
    const head = paramSummary(frame, opts, [
      { name: "x0", label: "x0" },
      { name: "load_factor", label: "λ" },
    ]);
    degraded.push(...head.degraded);
    const lines: string[] = [];
    lines.push(`${membrane ? "membrane" : "beam"}　${head.text}`);
    lines.push(
      membrane
        ? `n_grid=${fmt(metaNumber(frame, "n_grid"))}　单元 ${elements.rows}　节点 ${nNode}`
        : `n_elem=${fmt(metaNumber(frame, "n_elem"))}　${frame.meta["hermite"] === true ? "Hermite" : "P1"}　节点 ${nNode}`,
    );
    if (membrane) lines.push("主图：蓝=变形网格　灰虚=未变形　橙点=精确");
    if (x0 !== undefined) {
      lines.push(
        loadOnNode
          ? "载荷在节点上（误差锯齿谷底）"
          : `载荷距节点 ${formatNumber(nodeDistance ?? 0, 3)}（单元内部）`,
      );
    }
    if (singular) {
      lines.push("误差范数不可用：奇异（不写 0、不写 NaN）");
    } else {
      lines.push(`${errLabel}=${fmt(h2)}　L2=${fmt(l2Now)}`);
      lines.push(`节点误差=${fmt(nodalNow)}`);
    }
    if (membrane) {
      lines.push(
        `σ=${fmt(metaNumber(frame, "sigma"), 3)}　畸变 ${fmt(metaNumber(frame, "distortion"), 3)}` +
          `　J 比 ${fmt(metaNumber(frame, "jac_ratio"), 3)}`,
      );
    } else if (!singular) {
      lines.push(`w_max=${fmt(wMax)}　θ_max=${fmt(metaNumber(frame, "theta_max"))}`);
    }
    if (!singular && scanMin !== undefined && scanMax !== undefined) {
      lines.push(`扫 x0：误差 ${formatNumber(scanMin, 3)}…${formatNumber(scanMax, 3)}`);
      if (peak !== undefined && peak > 0) lines.push(`峰谷比 ${formatNumber(peak, 3)}`);
    }
    if (!singular) {
      lines.push(
        floorCount !== undefined
          ? `节点误差触地板 ${formatNumber(floorCount, 3)}/${scanX.length} 点：不画成数据`
          : "节点误差地板掩码缺失：地板点仍画在曲线上",
      );
    }
    if (!membrane && !singular) {
      lines.push(`能量随 λ²：当前 ${fmt(energyNow)}`);
      lines.push(`λ=1 时 ${fmt(h2)}`);
    }
    if (!singular && cond !== undefined && cond > 0) {
      lines.push(`条件数 ${formatNumber(cond, 4)}　残差 ${fmt(residual)}`);
    }
    if (singular) {
      lines.push(`⚠ 系统奇异：${failReason ?? "位移不画（singular_mask）"}`);
    }
    scene.texts.push(...textBlock(L.rightX, L.textTop, L.floor, lines, pal.muted, unitBudget(L.rightW), "top"));
  } else if (narrow) {
    degraded.push("窄画布（单列）：只画变形构型；扫描/误差副面板与右列说明未排下");
  } else if (short) {
    degraded.push("画布过矮：副面板与右列说明未排下，关键读数见角标与 pitfalls");
  }

  // —— 角标 ——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(true), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(badge("autoscale", "扫描/误差副轴固定（地板 1e-17）；主图自动缩放", 8, 26), SLOT.aux, width),
  );
  const elementIdx = metaNumber(frame, "element_index");
  const useDistortion = membrane && elementIdx !== undefined && Math.round(elementIdx) === 1 && frame.critical["distortion"] !== undefined;
  if (useDistortion) {
    const dist = metaNumber(frame, "distortion") ?? 0;
    scene.badges.push(
      placeBadge(criticalBadge("distortion", dist, criticalOf(frame, "distortion", 0.16), { beyond: "above" }), SLOT.critical, width),
    );
  } else {
    const x0Crit = x0 ?? (Number.isFinite(loadX) ? loadX : 0);
    scene.badges.push(
      placeBadge(criticalBadge("x0", x0Crit, criticalOf(frame, "x0", 0.5), { beyond: "above" }), SLOT.critical, width),
    );
  }
  const status: string[] = [];
  status.push(`λ=${formatNumber(lam, 3)}`);
  if (x0 !== undefined) status.push(`x0=${formatNumber(x0, 3)}`);
  if (!singular) status.push(`${errLabel}=${fmt(h2)}`);
  if (nodalNow !== undefined && nodalNow >= 0) status.push(`节点=${formatNumber(nodalNow, 2)}`);
  if (singular) status.push("奇异");
  scene.badges.push(placeBadge(badge("status", status.join("；")), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  const nBad =
    countNonFinite(mesh.values) +
    countNonFinite(elements.values) +
    countNonFinite(solution) +
    countNonFinite(exact) +
    countNonFinite(error) +
    countNonFinite(scanX) +
    countNonFinite(scanEnergy) +
    countNonFinite(singularMask) +
    countNonFinite(markX0) +
    countNonFinite(markEnergy) +
    (curve !== undefined ? countNonFinite(curve.values) : 0) +
    (nodalScan !== undefined ? countNonFinite(nodalScan) : 0) +
    (scanFloorMask !== undefined ? countNonFinite(scanFloorMask) : 0);
  applyOverflow(scene, nBad, { slot: SLOT.overflow, width });
  return scene;
};

function meshX(mesh: MatrixView): Float64Array {
  const out = new Float64Array(mesh.rows);
  for (let i = 0; i < mesh.rows; i += 1) out[i] = mesh.values[i * 2] as number;
  return out;
}

function fmt(v: number | undefined, digits = 3): string {
  return v === undefined || !Number.isFinite(v) ? "?" : formatNumber(v, digits);
}
