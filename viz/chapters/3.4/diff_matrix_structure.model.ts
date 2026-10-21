/**
 * `diff_matrix_structure`（3.4 章）的 model：Chebyshev 谱微分矩阵的稠密/奇异结构。
 *
 * 面板与图层（顺序即绘制顺序）
 * ----------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | heat | `main` | `log_abs_matrix` 热力图（行主序，(N+1)²） |
 * | 2 | scatter | `main` | 对角元：正/负两色（D_00=(2N²+1)/6，**不是 0**） |
 * | 3 | line | `aux` | 行和残差 `residual`（对数轴；零空间是常数） |
 * | 4 | line | `aux` | 奇异值谱 `sigma`（末位 1e-17 是零空间噪声） |
 * | 5 | line | `aux` | σmin+（挖掉零空间后的最小值）参考线 |
 * | 6 | line | `derivative` | 对解析解求导的逐点误差 `error_derivative` |
 * | 7 | line | `derivative` | N^4·u 的预期舍入地板 |
 *
 * 三条防误导决定
 * --------------
 * 1. **稠密/奇异必须画出来**：热力图是整块 (N+1)² 的彩色格（没有零对角线的稀疏假象），
 *    副轴的奇异值谱从 σmax≈N² 一路落到 1e-17——那个 1e-17 不是「精度」，是零空间噪声；
 *    同一张图上画 σmin+ 参考线，读者才能看见 `κ+=σmax/σmin+` 与朴素 `κ` 的区别。
 * 2. **对角元不是 0**：主图上用两色点叠出对角元；帧的 `diagonal` 缺了就降级，不补 0。
 * 3. **N^4·u 的舍入放大**：逐点误差轴画出 `meta.n4_roundoff` 参考线；误差高于它说明
 *    当前 N 仍在截断区，低于它是舍入区，两者不能混读。
 *
 * 时间轴：manifest `mode="none"`，`scene.t` 恒为 0。
 */

import { formatNumber } from "../../shared/src/axes.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import {
  applyOverflow,
  canvasSize,
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
  ModelInputError,
  type SceneModel,
} from "../../shared/src/model.js";
import {
  badge,
  criticalBadge,
  heatLayer,
  lineLayer,
  makeAxis,
  scatterLayer,
  series,
  sourceBadge,
} from "../../shared/src/scene.js";
import {
  axisTitleOf,
  axisYlabelOf,
  columnsLayout,
  legendBlock,
  logRange,
  positiveMinMax,
  textBlock,
  unitBudget,
} from "../_common/panel.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "3.4";
const VIZ = "diff_matrix_structure";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "谱微分矩阵：稠密、奇异、条件数 N^2";
const CRITICAL = SPEC?.params?.find((p) => p.name === "n")?.critical ?? [24];

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

export const diffMatrixStructureModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需数组 ——
  const matrix = frame.matrix("matrix");
  const logAbs = frame.matrix("log_abs_matrix");
  const residual = requireArray(frame, "residual");
  const sigma = requireArray(frame, "sigma");
  const nodes = requireArray(frame, "nodes");
  const derr = requireArray(frame, "error_derivative");
  const pitfalls = requirePitfalls(frame);
  const size = matrix.rows;
  if (!(size > 0) || matrix.cols !== size || matrix.values.length !== size * size) {
    throw new ModelInputError(`帧 ${CHAPTER}/${VIZ} 的 matrix 不是方阵：${matrix.rows}×${matrix.cols}`);
  }
  if (logAbs.rows !== size || logAbs.cols !== size) {
    throw new ModelInputError(`帧 ${CHAPTER}/${VIZ} 的 log_abs_matrix 形状 [${logAbs.rows}, ${logAbs.cols}] 与 matrix [${size}, ${size}] 不符`);
  }
  requireSameLength(residual, sigma, "帧数组 residual 与 sigma");
  requireSameLength(nodes, derr, "帧数组 nodes 与 error_derivative");
  if (residual.length !== size) throw new ModelInputError(`帧 ${CHAPTER}/${VIZ} 的 residual 长度 ${residual.length} ≠ size=${size}`);

  // —— 可选数组 ——
  const degraded: string[] = [];
  const diagonal = frame.has("diagonal") ? requireArray(frame, "diagonal") : undefined;
  if (diagonal === undefined) degraded.push("帧缺 diagonal：不叠对角元标记（不补 0）");
  else if (diagonal.length !== size) degraded.push("diagonal 长度与 matrix 阶不符：不叠对角元标记");
  const sigmaIndex = frame.has("sigma_index") ? requireArray(frame, "sigma_index") : undefined;

  // —— meta / strings ——
  const nMeta = metaNumber(frame, "n");
  const n = nMeta ?? size - 1;
  if (nMeta !== undefined && Math.round(nMeta) !== size - 1) {
    degraded.push(`帧 meta.n=${Math.round(nMeta)} 与 matrix 阶-1=${size - 1} 不一致：按矩阵画`);
  }
  const normInf = metaNumber(frame, "norm_inf");
  const normInfTheory = metaNumber(frame, "norm_inf_theory");
  const d00 = metaNumber(frame, "d00");
  const d00Theory = metaNumber(frame, "d00_theory");
  const sigmaMax = metaNumber(frame, "sigma_max");
  const sigmaMin = metaNumber(frame, "sigma_min");
  const sigmaMinNonzero = metaNumber(frame, "sigma_min_nonzero");
  const kappaNaive = metaNumber(frame, "kappa_naive");
  const kappaNonzero = metaNumber(frame, "kappa_nonzero");
  const n4Roundoff = metaNumber(frame, "n4_roundoff");
  const derrMax = metaNumber(frame, "derivative_error_max");
  const derrMid = metaNumber(frame, "derivative_error_mid");
  const which = stringOf(frame, "which") ?? (typeof opts.params.which === "string" ? opts.params.which : "?");

  let nnz = 0;
  for (let i = 0; i < matrix.values.length; i += 1) if ((matrix.values[i] as number) !== 0) nnz += 1;

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: 0,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  const L = columnsLayout(scene, width, height, { topShare: 0.6, auxH: 150, gap: 54, contentLeft: 120 });
  const short = L.floor - L.topY < 300;
  const topH = short ? Math.max(60, L.floor - L.topY - 28) : L.topH;
  const auxH = Math.max(52, Math.min(L.auxH, topH));

  // —— 主面板：热力图 ——（右侧留 112px 给色标条与它的数字，避免压到右列轴）
  const heatW = Math.max(180, L.leftW - 112);
  const heat = makeAxis("main", [L.left, L.topY, heatW, topH], {
    xlim: [0, size - 1],
    ylim: [0, size - 1],
    title: L.columns
      ? axisTitleOf(SPEC?.axes, "main", "log10|D_N| 热力图（行主序，含零对角线与零空间）")
      : "log10|D| 热力图",
    ylabel: axisYlabelOf(SPEC?.axes, "main", "行 i"),
    xlabel: "列 $j$",
    autoscaled: false,
  });
  scene.axes.push(heat);
  let lo = Number.POSITIVE_INFINITY;
  let hi = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < logAbs.values.length; i += 1) {
    const v = logAbs.values[i] as number;
    if (!Number.isFinite(v)) continue;
    lo = Math.min(lo, v);
    hi = Math.max(hi, v);
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) {
    lo = -16;
    hi = 2;
  }
  if (!(hi > lo)) hi = lo + 1;
  scene.layers.push(
    heatLayer("main", toNumeric(logAbs.values), size, size, [0, size - 1, 0, size - 1], {
      cmap: "magma",
      vmin: Math.floor(lo),
      vmax: Math.ceil(hi),
      label: `log10|${which}|`,
    }),
  );
  if (diagonal !== undefined && diagonal.length === size) {
    const posIdx: number[] = [];
    const negIdx: number[] = [];
    for (let i = 0; i < size; i += 1) {
      const v = diagonal[i] as number;
      if (v > 0) posIdx.push(i);
      else if (v < 0) negIdx.push(i);
    }
    if (posIdx.length > 0) {
      scene.layers.push(
        scatterLayer("main", posIdx, posIdx, { r: 1.6, color: "#f7fafc", alpha: 0.85 }),
      );
    }
    if (negIdx.length > 0) {
      scene.layers.push(
        scatterLayer("main", negIdx, negIdx, { r: 1.6, color: "#63b3ed", alpha: 0.85 }),
      );
    }
  }

  // —— 副轴：行和残差 + 奇异值谱 ——
  const pos = positiveMinMax([...Array.from(residual), ...Array.from(sigma), sigmaMinNonzero ?? Number.NaN]);
  const auxLo = pos?.[0] ?? 1e-18;
  const auxHi = pos?.[1] ?? 1;
  const aux = makeAxis("aux", [L.rightX, L.topY, L.rightW, auxH], {
    xlim: [0, Math.max(1, size - 1)],
    ylim: logRange(auxLo * 0.5, auxHi * 2),
    yscale: "log",
    title: L.columns
      ? axisTitleOf(SPEC?.axes, "aux", "行和残差 |D_N·1| 与奇异值谱（对数轴）")
      : "行和/奇异值",
    ylabel: "$\\sigma$ / 残差",
    xlabel: "行 $i$ / 奇异值序号",
    autoscaled: false,
  });
  scene.axes.push(aux);
  const rowIdx = new Float64Array(size);
  for (let i = 0; i < size; i += 1) rowIdx[i] = i;
  const sigmaX = sigmaIndex !== undefined && sigmaIndex.length === sigma.length ? sigmaIndex : rowIdx;
  if (sigmaMinNonzero !== undefined && sigmaMinNonzero > 0) {
    scene.layers.push(
      lineLayer(
        "aux",
        series([aux.xlim[0], aux.xlim[1]], [sigmaMinNonzero, sigmaMinNonzero], pal.reference, {
          width: 1,
          dash: [3, 3],
        }),
      ),
    );
  }
  scene.layers.push(
    lineLayer("aux", series(rowIdx, residual, pal.residual, { width: 1.8, marker: "s" })),
  );
  scene.layers.push(
    lineLayer("aux", series(sigmaX, sigma, pal.curves[0], { width: 2, marker: "o" })),
  );
  if (sigmaMin !== undefined && sigmaMin > 0) {
    const last = sigma.length - 1;
    scene.layers.push(
      scatterLayer("aux", [sigmaX[last] as number], [sigmaMin], {
        r: 4,
        color: pal.marker,
      }),
    );
  }

  // —— 下方面板：谱微分的逐点误差 ——
  if (!short) {
    const derrPos = positiveMinMax([...Array.from(derr), n4Roundoff ?? Number.NaN]);
    const dLo = derrPos?.[0] ?? 1e-16;
    const dHi = derrPos?.[1] ?? 1;
    const derivative = makeAxis("derivative", [L.left, L.botY, heatW, L.botH], {
      xlim: [-1, 1],
      ylim: logRange(dLo * 0.5, dHi * 2),
      yscale: "log",
      title: "谱微分的逐点误差 $|D_Nu-u\\'|$ 与 $N^4u$ 地板",
      ylabel: "误差",
      xlabel: "节点 $x$",
      autoscaled: false,
    });
    scene.axes.push(derivative);
    if (n4Roundoff !== undefined && n4Roundoff > 0) {
      scene.layers.push(
        lineLayer(
          "derivative",
          series([derivative.xlim[0], derivative.xlim[1]], [n4Roundoff, n4Roundoff], pal.reference, {
            width: 1,
            dash: [3, 4],
          }),
        ),
      );
    }
    scene.layers.push(
      lineLayer("derivative", series(nodes, derr, pal.curves[2], { width: 1.8, marker: "o" })),
    );
    scene.texts.push(
      ...legendBlock(L.left + 6, L.botY + 6, L.leftW, "橙=逐点误差　灰虚=N^4·u 舍入地板（再加大 N 没有意义）", pal.muted),
    );
  } else {
    degraded.push(
      short
        ? "窄画布：逐点误差面板与右列说明未排下（热力图/奇异值仍在）"
        : "窄画布（单列）：右列说明未排下，关键数字见角标与 pitfalls 面板",
    );
  }

  // —— 面板图例 ——
  scene.texts.push(
    ...legendBlock(
      L.left + 6,
      L.topY + topH - 32,
      L.leftW,
      `热力图 log10|${which}|：色标 ${formatNumber(Math.floor(lo), 3)}…${formatNumber(Math.ceil(hi), 3)}；白/蓝点=对角元（D_00=${formatNumber(d00 ?? d00Theory ?? 0, 4)}）`,
      pal.muted,
    ),
  );
  scene.texts.push(...legendBlock(L.rightX + 4, L.topY + auxH - 34, L.rightW, "紫=行和残差　蓝=σ；灰虚=σmin+", pal.muted));

  // —— 右列文字 ——
  if (L.columns) {
    const head = paramSummary(frame, opts, [{ name: "n", label: "$N$" }]);
    degraded.push(...head.degraded);
    const lines: string[] = [];
    lines.push(`N=${n}　${which}　${size}×${size} 稠密（${nnz}/${size * size} 非零）`);
    if (d00 !== undefined || d00Theory !== undefined) {
      lines.push(`D_00=${formatNumber(d00 ?? 0, 5)}（理论 ${formatNumber(d00Theory ?? 0, 5)}）：对角元≠0`);
    }
    if (normInf !== undefined) {
      lines.push(`||${which}||∞=${formatNumber(normInf, 5)}` + (normInfTheory !== undefined ? `（理论 ${formatNumber(normInfTheory, 5)}）` : ""));
    }
    if (sigmaMax !== undefined && sigmaMin !== undefined) {
      lines.push(`σmax=${formatNumber(sigmaMax, 4)}，σmin=${formatNumber(sigmaMin, 3)}（零空间噪声）`);
    }
    if (sigmaMinNonzero !== undefined && kappaNonzero !== undefined) {
      lines.push(`σmin+=${formatNumber(sigmaMinNonzero, 4)} → κ+=${formatNumber(kappaNonzero, 4)}`);
    }
    if (kappaNaive !== undefined) lines.push(`朴素 κ=σmax/σmin=${formatNumber(kappaNaive, 3)}（零空间噪声，无意义）`);
    if (derrMax !== undefined) {
      lines.push(
        `逐点误差 max=${formatNumber(derrMax, 3)}` +
          (derrMid !== undefined ? `（中位 ${formatNumber(derrMid, 3)}）` : "") +
          (n4Roundoff !== undefined ? `；N^4·u=${formatNumber(n4Roundoff, 3)}` : ""),
      );
    }
    scene.texts.push(...textBlock(L.rightX, L.textTop, L.floor, lines, pal.muted, unitBudget(L.rightW), "top"));
  } else if (!short) {
    degraded.push("窄画布（单列）：右列说明未排下，关键数字见角标与 pitfalls 面板");
  }

  // —— 角标 ——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  scene.badges.push(
    placeBadge(badge("autoscale", "坐标轴：固定；色标由帧的 log10|D| 定（不随 t 变）", 8, 26), SLOT.axisMode, width),
  );
  scene.badges.push(
    placeBadge(badge("status", "副轴：行和残差与奇异值共用对数标尺", 8, 44), SLOT.aux, width),
  );
  const rawCritical = frame.critical["n"];
  const critical = typeof rawCritical === "number" ? rawCritical : Array.isArray(rawCritical) ? rawCritical : CRITICAL;
  scene.badges.push(placeBadge(criticalBadge("N", n, critical, { beyond: "above" }), SLOT.critical, width));
  const status = [
    `N=${n}`,
    kappaNonzero !== undefined ? `κ+=${formatNumber(kappaNonzero, 3)}` : "",
    `稠密 ${nnz}/${size * size}`,
  ].filter((s) => s !== "");
  scene.badges.push(placeBadge(badge("status", status.join("；"), 8, 8), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  const nBad =
    countNonFinite(toNumeric(matrix.values)) +
    countNonFinite(toNumeric(logAbs.values)) +
    countNonFinite(residual) +
    countNonFinite(sigma) +
    countNonFinite(nodes) +
    countNonFinite(derr) +
    (diagonal !== undefined ? countNonFinite(diagonal) : 0);
  applyOverflow(scene, nBad, { slot: SLOT.overflow, width });
  return scene;
};
