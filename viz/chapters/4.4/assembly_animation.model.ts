/**
 * `assembly_animation`（4.4 章 viz 1）的 model：单元矩阵逐格装配成全局 K。
 *
 * 面板与图层（顺序即绘制顺序）
 * ----------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | heat | `matrix` | K_k 的已定型元素；`entry_pending=1` 的格子置哨兵并低于 `clipBelow`（中性灰=未定型） |
 * | 2 | scatter | `matrix` | `K_defect≠0` 的格子（路线偏差；full 路线为空） |
 * | 3 | scatter | `matrix` | 当前单元的散射目标（`local_dofs` 在对角线上的白点） |
 * | 4 | heat | `local` | 当前单元的局部矩阵 ke（独立色标） |
 * | 5 | line | `residuals` | 地板线 1e-17 |
 * | 6 | line | `residuals` | `residual`（requiredAux，掩码 `residual_at_floor` 的点不进图层） |
 * | 7 | line | `residuals` | `symmetry_residual`（requiredAux；0 没有对数，不画点只说明） |
 * | 8 | line | `residuals` | `pivot_condition`（只在 `pivot_unmeasured=0` 的步上画） |
 *
 * 三条防误导纪律
 * --------------
 * 1. **未定型不是 0**：`entry_pending=1` 的格子不画成 0 色，而是置一个低于色标窗口的哨兵，
 *    由 `clipBelow` 画成中性灰（无数据）。否则「还没装」会被读成「恰好抵消」。
 * 2. **对称残差 0 不是合格证**：P1 上 `symmetry_residual` 恒为精确 0（对数轴画不出），
 *    图上用文字说明；真正区分路线的是 `K_defect` 的红色格子与右栏的 `route_defect`。
 * 3. **主元比不是条件数**：`pivot_condition` 在奇异步上是显示上限（1e300），
 *    这些点被 `pivot_unmeasured` 排除，不冒充实测条件数。
 *
 * 时间轴：`scrub`，`t` 是装配进度（帧 `meta.t` / `t_actual` 优先）。
 */

import { formatNumber } from "../../shared/src/axes.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import {
  applyOverflow,
  canvasSize,
  clamp,
  dataSourceOf,
  degradedBadge,
  metaNumber,
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
  heatLayer,
  lineLayer,
  makeAxis,
  scatterLayer,
  series,
  sourceBadge,
  type LineSeries,
} from "../../shared/src/scene.js";
import {
  TEXT_ROW_H,
  axisTitleOf,
  axisYlabelOf,
  columnsLayout,
  textBlock,
  unitBudget,
} from "../_common/panel.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "4.4";
const VIZ = "assembly_animation";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "装配动画：单元矩阵怎么变成全局矩阵";

const FLOOR_FALLBACK = 1e-17;
const COND_CEIL = 1e300;
const MIRRORS = ["full", "half", "double"] as const;
const DEGREES = ["P1", "P2"] as const;
const COEFFICIENTS = ["const", "jump"] as const;

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

/** 一列里是不是有限值（heat 的哨兵与 NaN 都在这里被换掉）。 */
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

function finiteOr(v: number, fallback: number): number {
  return Number.isFinite(v) ? v : fallback;
}

/** 把 (x,y) 按掩码切成连续可绘段：被掩掉的位置不进图层，也不跨过去连线。 */
function maskedSeries(
  x: ArrayLike<number>,
  y: ArrayLike<number>,
  mask: ArrayLike<number> | undefined,
  color: string,
  style: Partial<LineSeries>,
): LineSeries[] {
  const n = Math.min(x.length, y.length);
  const out: LineSeries[] = [];
  let start = -1;
  const usable = (i: number): boolean => {
    const v = y[i] as number;
    return Number.isFinite(v) && (mask === undefined || (mask[i] ?? 0) <= 0.5);
  };
  const push = (a: number, b: number): void => {
    if (b - a < 1) return;
    const xs = new Float64Array(b - a);
    const ys = new Float64Array(b - a);
    for (let i = a; i < b; i += 1) {
      xs[i - a] = x[i] as number;
      ys[i - a] = y[i] as number;
    }
    out.push(series(xs, ys, color, out.length === 0 ? style : { ...style, label: undefined }));
  };
  for (let i = 0; i < n; i += 1) {
    if (usable(i)) {
      if (start < 0) start = i;
    } else if (start >= 0) {
      push(start, i);
      start = -1;
    }
  }
  if (start >= 0) push(start, n);
  return out;
}

function maxAbsOf(values: ArrayLike<number> | undefined): number {
  if (values === undefined) return 0;
  let m = 0;
  for (let i = 0; i < values.length; i += 1) {
    const v = values[i] as number;
    if (Number.isFinite(v)) m = Math.max(m, Math.abs(v));
  }
  return m;
}

export const assemblyAnimationModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需数组：缺一个就抛 ModelInputError ——
  const partial = numericMatrix(frame, "K_partial");
  const n = partial.rows;
  if (!(n > 0) || partial.cols !== n || partial.values.length !== n * n) {
    throw new Error(`帧 ${CHAPTER}/${VIZ} 的 K_partial 不是方阵：${partial.rows}×${partial.cols}`);
  }
  const local = numericMatrix(frame, "k_local");
  const npe = local.rows;
  if (!(npe > 0) || local.cols !== npe || local.values.length !== npe * npe) {
    throw new Error(`帧 ${CHAPTER}/${VIZ} 的 k_local 不是方阵：${local.rows}×${local.cols}`);
  }
  const localDofs = requireArray(frame, "local_dofs");
  const pending = requireArray(frame, "entry_pending");
  const kAxis = requireArray(frame, "k_axis");
  const residual = requireArray(frame, "residual");
  const symmetry = requireArray(frame, "symmetry_residual");
  if (pending.length !== n * n) {
    throw new Error(`帧 ${CHAPTER}/${VIZ} 的 entry_pending 长度 ${pending.length} ≠ n²=${n * n}`);
  }
  requireSameLength(kAxis, residual, "帧数组 k_axis 与 residual");
  requireSameLength(kAxis, symmetry, "帧数组 k_axis 与 symmetry_residual");
  const pitfalls = requirePitfalls(frame);

  // —— 可选数组 / meta：缺了降级，不猜 ——
  const degraded: string[] = [];
  const final = frame.has("K_final") ? numericMatrix(frame, "K_final") : undefined;
  if (final === undefined) degraded.push("帧缺 K_final：矩阵色标按当前 K_k 定（跨 t 可能变色）");
  else if (final.rows !== n || final.cols !== n) {
    degraded.push(`K_final 形状 [${final.rows}, ${final.cols}] 与 K_partial 不符：色标按当前 K_k 定`);
  }
  const defect = frame.has("K_defect") ? numericMatrix(frame, "K_defect") : undefined;
  if (defect === undefined) degraded.push("帧缺 K_defect：不标路线偏差的格子");
  else if (defect.rows !== n || defect.cols !== n) {
    degraded.push("K_defect 形状与 K_partial 不符：不标路线偏差的格子");
  }
  const pivot = frame.has("pivot_condition") ? requireArray(frame, "pivot_condition") : undefined;
  const pivotMask = frame.has("pivot_unmeasured") ? requireArray(frame, "pivot_unmeasured") : undefined;
  if (pivot === undefined) degraded.push("帧缺 pivot_condition：残差轴不画主元比（条件数代理）");
  else requireSameLength(kAxis, pivot, "帧数组 k_axis 与 pivot_condition");
  if (pivot !== undefined && pivotMask === undefined) {
    degraded.push("帧缺 pivot_unmeasured：不排除奇异步的显示上限，主元比可能冒充实测");
  } else if (pivotMask !== undefined) {
    requireSameLength(kAxis, pivotMask, "帧数组 k_axis 与 pivot_unmeasured");
  }
  const residualMask = frame.has("residual_at_floor") ? requireArray(frame, "residual_at_floor") : undefined;
  if (residualMask === undefined) degraded.push("帧缺 residual_at_floor：地板步仍画在残差曲线上（不标已到地板）");
  else requireSameLength(kAxis, residualMask, "帧数组 k_axis 与 residual_at_floor");

  const floorText = stringOf(frame, "floor");
  const floor = floorText !== undefined && Number(floorText) > 0 ? Number(floorText) : FLOOR_FALLBACK;
  if (floorText === undefined) degraded.push("帧 strings 缺 floor：地板按 1e-17 显示");

  // —— 时间：帧内 meta 优先；scene.t 必须如实 ——
  const tMeta = metaNumber(frame, "t_actual") ?? metaNumber(frame, "t");
  const t = clamp(tMeta ?? opts.t, 0, 1);
  if (tMeta === undefined) degraded.push("帧 meta 缺 t/t_actual：时间取请求值并夹到 [0,1]");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  const L = columnsLayout(scene, width, height, { topShare: 0.6, auxH: 150, gap: 56 });
  const narrow = !L.columns;
  const short = L.floor - L.topY < 250;
  const HEAT_GAP = 112;
  const topH = narrow || short ? Math.max(60, L.floor - L.topY - 28) : L.topH;
  const matrixW = Math.max(120, (narrow ? width - L.left - 24 : L.leftW) - HEAT_GAP);

  // —— 主面板：K_k 热力图（固定窗口；灰=未定型）——
  const nDofMeta = metaNumber(frame, "n_dof");
  if (nDofMeta !== undefined && Math.round(nDofMeta) !== n) {
    degraded.push(`meta.n_dof=${Math.round(nDofMeta)} 与 K_partial 阶 ${n} 不一致：按矩阵画`);
  }
  const matrixAxis = makeAxis("matrix", [L.left, L.topY, matrixW, topH], {
    xlim: [-0.5, n - 0.5],
    ylim: [-0.5, n - 0.5],
    xlabel: "自由度 $j$",
    title: axisTitleOf(SPEC?.axes, "matrix", "全局刚度阵 K_k（固定窗口，灰=未定型）"),
    ylabel: axisYlabelOf(SPEC?.axes, "matrix", "自由度 i"),
    autoscaled: false,
  });
  const matrixScale = maxAbsOf(final !== undefined && final.rows === n && final.cols === n ? final.values : partial.values);
  const scaleAbs = Math.max(matrixScale, 1e-300);
  const sentinel = -2.5 * scaleAbs;
  const clipBelow = -1.5 * scaleAbs;
  const display = new Float64Array(n * n);
  let pendingCount = 0;
  for (let i = 0; i < n * n; i += 1) {
    const isPending = (pending[i] ?? 0) > 0.5;
    if (isPending) pendingCount += 1;
    display[i] = isPending ? sentinel : finiteOr(partial.values[i] as number, sentinel);
  }
  scene.axes.push(matrixAxis);

  // 未定型格子的本地计数与 meta 可能不同（entry_pending 把结构零也算未定型）；
  // 两个口径都报出来，免得把「还没来」与「结构性 0」混成一笔账。
  const metaPending = metaNumber(frame, "entries_pending");

  scene.layers.push(
    heatLayer("matrix", display, n, n, [-0.5, n - 0.5, -0.5, n - 0.5], {
      cmap: "rdbu",
      vmin: -scaleAbs,
      vmax: scaleAbs,
      clipBelow,
      label: "`K_ij`",
    }),
  );

  // 路线缺陷：只标非零格（full 路线自然为空）。红色点叠在热力图上，
  // 「矩阵仍然对称、仍然正定」时这是唯一能看见的错。
  if (defect !== undefined && defect.rows === n && defect.cols === n) {
    const dMax = maxAbsOf(defect.values);
    if (dMax > 0) {
      const xs: number[] = [];
      const ys: number[] = [];
      for (let r = 0; r < n; r += 1) {
        for (let c = 0; c < n; c += 1) {
          const v = defect.values[r * n + c] as number;
          if (Number.isFinite(v) && Math.abs(v) > 1e-9 * dMax) {
            xs.push(c);
            ys.push(r);
          }
        }
      }
      if (xs.length > 0) {
        scene.layers.push(scatterLayer("matrix", xs, ys, { r: 2.6, color: pal.warn, alpha: 0.95 }));
      }
    }
  }

  // 当前单元的自由度在对角线上的落点（散射目标）。
  const diag: number[] = [];
  for (let i = 0; i < localDofs.length; i += 1) {
    const d = localDofs[i] as number;
    if (Number.isFinite(d) && d >= 0 && d < n) diag.push(d);
  }
  if (diag.length > 0) {
    scene.layers.push(scatterLayer("matrix", diag, diag, { r: 3.6, color: "#f7fafc", alpha: 0.9 }));
  }

  // —— 右列：当前单元矩阵（独立色标；右侧留 112px 给色标）——
  let localAxisW = 0;
  if (!narrow && !short) {
    localAxisW = Math.max(76, L.rightW - HEAT_GAP);
    const localAxis = makeAxis("local", [L.rightX, L.topY, localAxisW, L.auxH], {
      xlim: [-0.5, npe - 0.5],
      ylim: [-0.5, npe - 0.5],
      xlabel: "局部自由度",
      title: axisTitleOf(SPEC?.axes, "local", "当前单元矩阵 ke（色标自适应）"),
      ylabel: axisYlabelOf(SPEC?.axes, "local", "局部自由度"),
      xticks: Array.from({ length: npe }, (_, i) => i),
      yticks: Array.from({ length: npe }, (_, i) => i),
      autoscaled: true,
    });
    scene.axes.push(localAxis);
    let lo = Number.POSITIVE_INFINITY;
    let hi = Number.NEGATIVE_INFINITY;
    for (let i = 0; i < local.values.length; i += 1) {
      const v = local.values[i] as number;
      if (!Number.isFinite(v)) continue;
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    }
    if (!Number.isFinite(lo) || !Number.isFinite(hi)) {
      lo = -1;
      hi = 1;
    }
    if (!(hi > lo)) {
      const pad = Math.abs(hi) > 0 ? Math.abs(hi) * 0.1 : 1;
      lo -= pad;
      hi += pad;
    }
    scene.layers.push(
      heatLayer("local", local.values, npe, npe, [-0.5, npe - 0.5, -0.5, npe - 0.5], {
        cmap: "rdbu",
        vmin: lo,
        vmax: hi,
        label: "`ke_ij`",
      }),
    );
  }

  // —— 下方面板：装配/对称性残差与主元比（对数）——
  let residualAxis: ReturnType<typeof makeAxis> | undefined;
  if (!narrow && !short) {
    let measuredPivot = 0;
    if (pivot !== undefined && pivotMask !== undefined) {
      for (let i = 0; i < Math.min(pivot.length, pivotMask.length); i += 1) {
        if ((pivotMask[i] ?? 0) > 0.5) continue;
        const v = pivot[i] as number;
        if (Number.isFinite(v) && v > 0 && v < COND_CEIL) measuredPivot = Math.max(measuredPivot, v);
      }
    }
    const resMax = Math.max(maxAbsOf(residual), maxAbsOf(symmetry), measuredPivot, floor);
    const upper = Math.pow(10, Math.ceil(Math.log10(Math.max(resMax, 1) * 5)));
    residualAxis = makeAxis("residuals", [L.left, L.botY, L.leftW, L.botH], {
      xlim: [0, Math.max(1, kAxis.length - 1)],
      ylim: [floor, Math.max(upper, floor * 10)],
      yscale: "log",
      xlabel: "已装入的单元数 $k$",
      title: axisTitleOf(SPEC?.axes, "residuals", "残差与主元比（对数，地板 1e-17）"),
      ylabel: axisYlabelOf(SPEC?.axes, "residuals", "max|ΔK|"),
      autoscaled: false,
    });
    scene.axes.push(residualAxis);
    scene.layers.push(
      lineLayer(
        "residuals",
        series([0, Math.max(1, kAxis.length - 1)], [floor, floor], pal.reference, {
          width: 1,
          dash: [3, 3],
        }),
      ),
    );
    const resSeries = maskedSeries(kAxis, residual, residualMask, pal.curves[0], {
      width: 2.2,
      marker: "o",
    });
    if (resSeries.length > 0) scene.layers.push(lineLayer("residuals", resSeries));
    const symSeries = maskedSeries(kAxis, symmetry, undefined, pal.curves[1], {
      width: 1.8,
      marker: "s",
      dash: [5, 3],
    }).filter((s) => Array.from(s.y).some((v) => (v as number) > 0));
    if (symSeries.length > 0) scene.layers.push(lineLayer("residuals", symSeries));
    if (pivot !== undefined && pivotMask !== undefined) {
      const pivSeries = maskedSeries(kAxis, pivot, pivotMask, pal.exact, {
        width: 1.6,
        marker: "x",
        dash: [2, 3],
      }).filter((s) => Array.from(s.y).every((v) => (v as number) < COND_CEIL));
      if (pivSeries.length > 0) scene.layers.push(lineLayer("residuals", pivSeries));
    }
  }

  // —— 面板内图例 ——
  if (residualAxis !== undefined && L.botH >= 48) {
    const lines = [
      "蓝=一致性残差　绿虚=对称残差　橙虚=主元比",
      `灰虚=地板 ${floorText ?? "1e-17"}；主元比掩码步不进图层`,
    ];
    const legendTop = L.botY + Math.max(6, Math.floor((L.botH - 3 * TEXT_ROW_H) / 2));
    scene.texts.push(
      ...textBlock(
        L.left + 6,
        legendTop,
        legendTop + 3 * TEXT_ROW_H,
        lines,
        pal.muted,
        unitBudget(L.leftW),
        "top",
      ),
    );
  }

  // —— 右列文字 ——
  const kDone = metaNumber(frame, "k_done");
  const nElem = metaNumber(frame, "n_elem");
  const mirrorIdx = metaNumber(frame, "mirror_index");
  const degreeIdx = metaNumber(frame, "degree_index");
  const coeffIdx = metaNumber(frame, "coefficient_index");
  const routeDefect = metaNumber(frame, "route_defect");
  const solveDefect = metaNumber(frame, "solve_defect");
  const condFinal = metaNumber(frame, "condition_final");
  const lambdaMin = metaNumber(frame, "lambda_min_final");
  const lambdaMax = metaNumber(frame, "lambda_max_final");
  const symmetryMax = metaNumber(frame, "symmetry_max_over_history");
  const routeName = mirrorIdx !== undefined ? MIRRORS[Math.round(mirrorIdx)] ?? "?" : stringOf(frame, "mirror") ?? "?";
  const degreeName = degreeIdx !== undefined ? DEGREES[Math.round(degreeIdx)] ?? "?" : "?";
  const coeffName = coeffIdx !== undefined ? COEFFICIENTS[Math.round(coeffIdx)] ?? "?" : "?";

  if (L.columns && !short && L.floor > L.textTop + 8) {
    const head = paramSummary(frame, opts, [
      { name: "n_elem", label: "`n_elem`" },
      { name: "n_dof", label: "`n_dof`" },
    ]);
    degraded.push(...head.degraded);
    const lines: string[] = [];
    lines.push(`${degreeName} / ${coeffName} / 路线 ${routeName}　${head.text}`);
    lines.push(
      `已装 ${kDone !== undefined ? formatNumber(kDone, 4) : "?"}/${nElem !== undefined ? formatNumber(nElem, 4) : "?"} 单元；` +
        `未定型 ${formatNumber(metaPending ?? pendingCount, 3)}/${n * n} 格（灰）`,
    );
    lines.push(
      `路线偏差 max|K_route−K|=${routeDefect !== undefined ? formatNumber(routeDefect, 4) : "?"}` +
        `；解差=${solveDefect !== undefined && solveDefect >= 0 ? formatNumber(solveDefect, 3) : "?"}`,
    );
    if (lambdaMin !== undefined && lambdaMax !== undefined && lambdaMin > 0) {
      lines.push(
        `λ_min=${formatNumber(lambdaMin, 4)}　λ_max=${formatNumber(lambdaMax, 4)}` +
          (condFinal !== undefined && condFinal > 0 ? `　cond=${formatNumber(condFinal, 4)}` : ""),
      );
    } else {
      lines.push("特征值未精确测量（自由度数超过上限）：只报主元比代理");
    }
    lines.push(
      `对称残差最大 ${symmetryMax !== undefined ? formatNumber(symmetryMax, 3) : "?"}：` +
        (symmetryMax !== undefined && symmetryMax === 0 ? "P1 上恒 0，不能当合格证" : "非零步见副图"),
    );
    lines.push("灰格=未定型（结构零也不与 0 混读）");
    if (symmetryMax !== undefined && symmetryMax === 0 && routeDefect !== undefined && routeDefect > 0) {
      lines.push("⚠ 对称残差 0 而路线偏差非零：这项「体检」通过了，矩阵却是错的");
    }
    scene.texts.push(...textBlock(L.rightX, L.textTop, L.floor, lines, pal.muted, unitBudget(L.rightW), "top"));
  } else if (!narrow && short) {
    degraded.push("画布过矮：副面板与右列说明未排下，关键读数见角标与 pitfalls");
  } else if (narrow) {
    degraded.push("窄画布（单列）：只画主矩阵；局部矩阵/残差副面板与右列说明未排下");
  }

  // —— 角标 ——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(
      badge("autoscale", "矩阵/残差窗固定；局部矩阵色标随帧（独立色标）", 8, 26),
      SLOT.aux,
      width,
    ),
  );
  const status: string[] = [];
  if (kDone !== undefined && nElem !== undefined) status.push(`已装 ${formatNumber(kDone, 3)}/${formatNumber(nElem, 3)}`);
  status.push(`灰=未定型 ${formatNumber(metaPending ?? pendingCount, 3)} 格`);
  status.push(routeName);
  if (routeDefect !== undefined) status.push(`Δ=${formatNumber(routeDefect, 3)}`);
  if (condFinal !== undefined && condFinal > 0) status.push(`κ=${formatNumber(condFinal, 4)}`);
  scene.badges.push(placeBadge(badge("status", status.join("；")), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  const nBad =
    countNonFinite(partial.values) +
    countNonFinite(local.values) +
    countNonFinite(localDofs) +
    countNonFinite(pending) +
    countNonFinite(kAxis) +
    countNonFinite(residual) +
    countNonFinite(symmetry) +
    (final !== undefined ? countNonFinite(final.values) : 0) +
    (defect !== undefined ? countNonFinite(defect.values) : 0) +
    (pivot !== undefined ? countNonFinite(pivot) : 0) +
    (pivotMask !== undefined ? countNonFinite(pivotMask) : 0) +
    (residualMask !== undefined ? countNonFinite(residualMask) : 0);
  applyOverflow(scene, nBad, { slot: SLOT.overflow, width });
  return scene;
};
