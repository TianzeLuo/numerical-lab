/**
 * `convergence_rate`（3.4 章）的 model：有限差分的代数收敛 vs 谱配置的指数收敛。
 *
 * 面板与图层（顺序即绘制顺序）
 * ----------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | line | `main` | 舍入地板（灰虚线） |
 * | 2 | line | `main` | 实测半对数拟合 `fit_exponential`（橙虚线） |
 * | 3 | line | `main` | 谱配置误差 `cheb_error`（蓝实线 + 点） |
 * | 4 | line | `main` | 插值误差 `interp_error`（紫虚线，负对照） |
 * | 5 | scatter | `main` | 最小误差点与达到地板的 N |
 * | 6 | loglogfit | `aux` | 有限差分误差 + **实测阶**与**理论阶**两条参考（规则 2） |
 * | 7 | line | `aux` | 内部误差 `fd_error_interior`（同阶，对照边界污染） |
 * | 8 | line | `aux` | 真实边界残差 `fd_residual` 与离散行残差（两个口径） |
 * | 9 | line | `aux` | 幂律竞争模型 `fit_power`（只在拟合窗口里画） |
 *
 * 三条防误导决定
 * --------------
 * 1. **规则 2 的落点是 `loglogfit`**：`measuredSlope=-fd_p`、`theorySlope=-bc_order`
 *    同时画出来。理论阶取帧 `meta.bc_order`（有限差分整体阶 = min(内部 2, 边界格式阶)），
 *    不是照抄 manifest；缺它时明确降级，不偷偷画一条没有对照的误差线。
 * 2. **指数收敛不能只看 R²**：同窗口里幂律拟合常常也很高（本章 pitfalls 第一条），
 *    所以图上同时给出幂律竞争线，并在文字里报两个 R² 与误差比走向。
 * 3. **地板不是「误差很小」**：`floor` 是实测平台；`N≥n_floor` 后误差不再下降，
 *    地板线必须画出来。插值曲线是负对照——它不是配置解误差。
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
  paramNumber,
  paramSummary,
  placeBadge,
  requireArray,
  requireDrawableLayers,
  requirePitfalls,
  requireSameLength,
  sceneShell,
  segmentedSeries,
  stringOf,
  ModelInputError,
  type SceneModel,
} from "../../shared/src/model.js";
import {
  badge,
  lineLayer,
  loglogFitLayer,
  makeAxis,
  scatterLayer,
  series,
  sourceBadge,
  type LineSeries,
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
const VIZ = "convergence_rate";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "有限差分 vs 谱配置：误差衰减的代数与指数之别";

const SLOT = {
  source: 0,
  axisMode: 1,
  aux: 2,
  status: 3,
  preview: 4,
  overflow: 6,
  degraded: 7,
} as const;

export const convergenceRateModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需数组 ——
  const nFd = requireArray(frame, "n_fd");
  const fdError = requireArray(frame, "fd_error");
  const nCheb = requireArray(frame, "n_cheb");
  const chebError = requireArray(frame, "cheb_error");
  const fdResidual = requireArray(frame, "fd_residual");
  const pitfalls = requirePitfalls(frame);
  requireSameLength(nFd, fdError, "帧数组 n_fd 与 fd_error");
  requireSameLength(nCheb, chebError, "帧数组 n_cheb 与 cheb_error");
  if (nFd.length < 2 || nCheb.length < 2) {
    throw new ModelInputError(`帧 ${CHAPTER}/${VIZ} 的误差曲线少于 2 个点：无法画收敛图`);
  }

  // —— 可选数组：缺了只是少一条对照 / 少一条拟合线，如实降级 ——
  const degraded: string[] = [];
  const fdInterior = frame.has("fd_error_interior") ? requireArray(frame, "fd_error_interior") : undefined;
  if (fdInterior === undefined) degraded.push("帧缺 fd_error_interior：不画内部误差对照（不按整体阶自己造）");
  const fdResidualDiscrete = frame.has("fd_residual_discrete") ? requireArray(frame, "fd_residual_discrete") : undefined;
  if (fdResidualDiscrete === undefined) degraded.push("帧缺 fd_residual_discrete：不画离散行残差（真实残差仍在）");
  const chebResidual = frame.has("cheb_residual") ? requireArray(frame, "cheb_residual") : undefined;
  const interpError = frame.has("interp_error") ? requireArray(frame, "interp_error") : undefined;
  if (interpError === undefined) degraded.push("帧缺 interp_error：不画插值负对照（不许拿它冒充配置误差）");
  const fitExp = frame.has("fit_exponential") ? requireArray(frame, "fit_exponential") : undefined;
  if (fitExp === undefined) degraded.push("帧缺 fit_exponential：不画实测半对数拟合线");
  const fitPower = frame.has("fit_power") ? requireArray(frame, "fit_power") : undefined;
  if (fitPower === undefined) degraded.push("帧缺 fit_power：不画幂律竞争线");
  const floorArr = frame.has("floor") ? requireArray(frame, "floor") : undefined;
  if (floorArr === undefined) degraded.push("帧缺 floor：不画舍入地板（地板不是 0）");

  // —— meta ——
  const bcOrderMeta = metaNumber(frame, "bc_order");
  const bcOrder = bcOrderMeta ?? paramNumber(opts.params, "bc_order");
  if (bcOrderMeta === undefined) {
    degraded.push(
      bcOrder === undefined
        ? "帧 meta 缺 bc_order：规则 2 的理论阶无法画（不猜 2）"
        : "帧 meta 缺 bc_order：理论阶用请求参数并标注",
    );
  }
  const fdP = metaNumber(frame, "fd_p");
  if (fdP === undefined) degraded.push("帧 meta 缺 fd_p：不画 loglogfit（规则 2 不许只画一条理论线）");
  const fdR2 = metaNumber(frame, "fd_r2");
  const expSlope = metaNumber(frame, "exp_slope");
  const expR2 = metaNumber(frame, "exp_r2");
  const expDigits = metaNumber(frame, "exp_digits_per_node");
  const powerP = metaNumber(frame, "power_p");
  const powerR2 = metaNumber(frame, "power_r2");
  const interpP = metaNumber(frame, "interp_p");
  const floorMeta = metaNumber(frame, "floor");
  const nFloor = metaNumber(frame, "n_floor");
  const nAtMin = metaNumber(frame, "n_at_min");
  const errorMin = metaNumber(frame, "error_min");
  const fdWinLo = metaNumber(frame, "fd_window_lo");
  const fdWinHi = metaNumber(frame, "fd_window_hi");
  const chebWinLo = metaNumber(frame, "cheb_window_lo");
  const chebWinHi = metaNumber(frame, "cheb_window_hi");
  const expHead = metaNumber(frame, "exp_fit_head");
  const problem = stringOf(frame, "problem");

  const floorVal = floorArr !== undefined && floorArr.length > 0 ? (floorArr[0] as number) : floorMeta;
  const maxN = Math.max(...Array.from(nCheb, (v) => (Number.isFinite(v) ? v : 0)), 1);
  const minN = Math.min(...Array.from(nCheb, (v) => (Number.isFinite(v) && v > 0 ? v : Number.POSITIVE_INFINITY)));
  const mainX0 = Number.isFinite(minN) ? Math.max(0, minN * 0.5) : 0;
  const mainX1 = maxN * 1.05;

  const mainPos = positiveMinMax([
    ...Array.from(chebError),
    ...(interpError !== undefined ? Array.from(interpError) : []),
    ...(fitExp !== undefined ? Array.from(fitExp) : []),
    floorVal ?? Number.NaN,
  ]);
  const mainLo = mainPos?.[0] ?? 1e-16;
  const mainHi = mainPos?.[1] ?? 1;

  const auxPos = positiveMinMax([
    ...Array.from(fdError),
    ...Array.from(fdResidual),
    ...(fdInterior !== undefined ? Array.from(fdInterior) : []),
    ...(fdResidualDiscrete !== undefined ? Array.from(fdResidualDiscrete) : []),
    ...(fitPower !== undefined ? Array.from(fitPower) : []),
    floorVal ?? Number.NaN,
  ]);
  const auxLo = auxPos?.[0] ?? 1e-16;
  const auxHi = auxPos?.[1] ?? 1;
  const xAll = [...Array.from(nFd), ...Array.from(nCheb)].filter((v) => Number.isFinite(v) && v > 0);
  const xLo = xAll.length > 0 ? Math.min(...xAll) : 1;
  const xHi = xAll.length > 0 ? Math.max(...xAll) : 2;

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: 0,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  const L = columnsLayout(scene, width, height, { topShare: 0.999, auxH: 158, gap: 0, contentLeft: 120 });
  const short = L.floor - L.topY < 300;
  const topH = short ? Math.max(60, L.floor - L.topY - 28) : L.topH;
  const auxH = Math.max(52, Math.min(L.auxH, topH));

  const main = makeAxis("main", [L.left, L.topY, L.leftW, topH], {
    xlim: [mainX0, mainX1],
    ylim: logRange(mainLo * 0.5, mainHi * 2),
    yscale: "log",
    title: L.columns
      ? axisTitleOf(SPEC?.axes, "main", "误差 vs N：半对数（指数收敛是直线）")
      : "误差 vs N（半对数）",
    ylabel: axisYlabelOf(SPEC?.axes, "main", "max|u_num - u_exact|"),
    xlabel: "N（Chebyshev 节点数）",
    autoscaled: false,
  });
  const aux = makeAxis("aux", [L.rightX, L.topY, L.rightW, auxH], {
    xlim: logRange(xLo * 0.9, xHi * 1.1),
    ylim: logRange(auxLo * 0.5, auxHi * 2),
    xscale: "log",
    yscale: "log",
    title: L.columns
      ? axisTitleOf(SPEC?.axes, "aux", "误差 vs N：双对数（代数收敛是直线）+ 真实边界残差")
      : "双对数 + 边界残差",
    ylabel: axisYlabelOf(SPEC?.axes, "aux", "误差 / 边界残差"),
    xlabel: "n / N",
    autoscaled: false,
  });
  scene.axes.push(main, aux);

  // —— 图层 1：地板 ——
  if (floorVal !== undefined && floorVal > 0) {
    scene.layers.push(
      lineLayer(
        "main",
        series([main.xlim[0], main.xlim[1]], [floorVal, floorVal], pal.reference, {
          width: 1,
          dash: [2, 4],
        }),
      ),
    );
  }

  // —— 图层 2~4：实测指数拟合、谱误差、插值负对照 ——
  if (fitExp !== undefined && fitExp.length === nCheb.length) {
    scene.layers.push(
      lineLayer(
        "main",
        segmentedPositive(nCheb, fitExp, pal.exact, { width: 1.5, dash: [6, 4], label: "实测半对数拟合" }),
      ),
    );
  }
  const mainSeries: LineSeries[] = [];
  mainSeries.push(...segmentedPositive(nCheb, chebError, pal.curves[0], { width: 2.2, marker: "o" }));
  if (interpError !== undefined && interpError.length === nCheb.length) {
    mainSeries.push(
      ...segmentedPositive(nCheb, interpError, pal.residual, {
        width: 1.3,
        dash: [5, 4],
        marker: "s",
      }),
    );
  }
  scene.layers.push(lineLayer("main", mainSeries));

  // —— 图层 5：最小误差点（地板上的那一个）——
  if (errorMin !== undefined && errorMin > 0 && nAtMin !== undefined) {
    const idx = Array.from(nCheb).findIndex((v, i) => Math.abs((v as number) - nAtMin) < 1e-9 && Math.abs((chebError[i] as number) - errorMin) < 1e-30);
    const xAt = idx >= 0 ? (nCheb[idx] as number) : nAtMin;
    scene.layers.push(
      scatterLayer("main", [xAt], [Math.max(errorMin, floorVal ?? errorMin)], {
        r: 4,
        color: pal.marker,
        label: `最小误差 ${formatNumber(errorMin, 3)} @ N=${formatNumber(nAtMin, 3)}`,
      }),
    );
  }

  // —— 图层 6：规则 2 —— 实测阶与理论阶必须同图 ——
  const fdPairs = positivePairs(nFd, fdError);
  if (fdP !== undefined && bcOrder !== undefined && fdPairs.h.length >= 2) {
    scene.layers.push(
      loglogFitLayer("aux", fdPairs.h, fdPairs.err, -fdP, -bcOrder, {
        seriesLabel:
          L.rightW >= 220
            ? `实测斜率 -${formatNumber(fdP, 4)}（最小二乘，窗口见文字）`
            : `实测 -${formatNumber(fdP, 3)}`,
        theoryLabel:
          L.rightW >= 220
            ? `理论 -${formatNumber(bcOrder, 3)}（整体阶=min(2,边界格式阶)）`
            : `理论 -${formatNumber(bcOrder, 3)}`,
        color: pal.curves[0],
        theoryColor: pal.marker,
      }),
    );
  } else {
    scene.layers.push(lineLayer("aux", segmentedPositive(nFd, fdError, pal.curves[0], { width: 1.8, marker: "o" })));
  }

  // —— 图层 7~9：内部误差、两种边界残差、幂律竞争线 ——
  const auxSeries: LineSeries[] = [];
  if (fdInterior !== undefined && fdInterior.length === nFd.length) {
    auxSeries.push(...segmentedPositive(nFd, fdInterior, pal.curves[2], { width: 1.2, dash: [4, 3] }));
  }
  auxSeries.push(...segmentedPositive(nFd, fdResidual, pal.residual, { width: 1.6, marker: "s" }));
  if (fdResidualDiscrete !== undefined && fdResidualDiscrete.length === nFd.length) {
    auxSeries.push(
      ...segmentedPositive(nFd, fdResidualDiscrete, pal.reference, {
        width: 1,
        dash: [2, 4],
      }),
    );
  }
  if (chebResidual !== undefined && chebResidual.length === nCheb.length) {
    auxSeries.push(...segmentedPositive(nCheb, chebResidual, pal.exact, { width: 1.1, dash: [1, 3] }));
  }
  if (fitPower !== undefined && powerP !== undefined && powerP > 0) {
    const head = Math.max(2, Math.min(Math.round(expHead ?? nCheb.length), nCheb.length));
    auxSeries.push(
      ...segmentedPositive(
        firstN(nCheb, head),
        firstN(fitPower, head),
        pal.exact,
        { width: 1.2, dash: [7, 4] },
      ),
    );
  }
  if (auxSeries.length > 0) scene.layers.push(lineLayer("aux", auxSeries));

  // —— 面板图例 ——
  scene.texts.push(
    ...legendBlock(
      L.left + 6,
      L.topY + 6,
      L.leftW,
      `蓝=谱配置　紫虚=插值负对照　橙虚=半对数拟合；斜率 ${formatNumber(expSlope ?? 0, 3)}`,
      pal.muted,
    ),
  );
  scene.texts.push(
    ...legendBlock(
      L.rightX + 4,
      L.topY + auxH - 40,
      L.rightW,
      "双对数：蓝=FD 误差+斜率　紫=边界残差",
      pal.muted,
    ),
  );

  // —— 右列文字 ——
  if (L.columns) {
    const head = paramSummary(frame, opts, [
      { name: "nmax", label: "nmax" },
      { name: "bc_order", label: "bc_order" },
    ]);
    degraded.push(...head.degraded);
    const lines: string[] = [];
    lines.push(`${head.text}　${problem ?? "?"}`);
    if (fdP !== undefined) {
      lines.push(
        `差分实测 p=${formatNumber(fdP, 4)}（R²=${formatNumber(fdR2 ?? 0, 4)}）` +
          (bcOrder !== undefined ? ` vs 理论 ${formatNumber(bcOrder, 3)}` : ""),
      );
    }
    if (fdWinLo !== undefined && fdWinHi !== undefined) {
      lines.push(
        `窗口：n=${formatNumber(fdWinLo, 0)}..${formatNumber(fdWinHi, 0)}` +
          (chebWinLo !== undefined && chebWinHi !== undefined
            ? `；N=${formatNumber(chebWinLo, 0)}..${formatNumber(chebWinHi, 0)}`
            : ""),
      );
    }
    if (expSlope !== undefined) {
      lines.push(
        `谱斜率 ${formatNumber(expSlope, 3)}（每节点 ${formatNumber(expDigits ?? 0, 2)} 位，R²=${formatNumber(expR2 ?? 0, 3)}）`,
      );
    }
    if (powerP !== undefined && powerR2 !== undefined) {
      lines.push(`幂律竞争 p=${formatNumber(powerP, 3)}（R²=${formatNumber(powerR2, 3)}）：R² 高≠幂律，判据看误差比`);
    }
    if (interpP !== undefined) lines.push(`插值 p=${formatNumber(interpP, 3)}（负对照，不是配置误差）`);
    if (floorVal !== undefined) {
      lines.push(
        `地板 ${formatNumber(floorVal, 3)}` + (nFloor !== undefined ? `，N≥${formatNumber(nFloor, 0)} 起不再下降` : ""),
      );
    }
    scene.texts.push(...textBlock(L.rightX, L.textTop, L.floor, lines, pal.muted, unitBudget(L.rightW), "top"));
  } else {
    degraded.push(
      short
        ? "窄画布：双对数副轴保留、右列说明与长标题缩写（关键数字见角标）"
        : "窄画布（单列）：右列说明未排下，关键数字见角标与 pitfalls 面板",
    );
  }

  // —— 角标 ——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  scene.badges.push(
    placeBadge(badge("autoscale", "坐标轴：固定（误差/残差轴不随参数回缩）", 8, 26), SLOT.axisMode, width),
  );
  scene.badges.push(
    placeBadge(badge("status", "副轴：双对数；实测与理论斜率同图（规则 2）", 8, 44), SLOT.aux, width),
  );
  const status = [
    fdP !== undefined ? `差分 p=${formatNumber(fdP, 3)}` : "差分 p=未给",
    expSlope !== undefined ? `谱斜率 ${formatNumber(expSlope, 3)}` : "谱斜率未给",
    floorVal !== undefined ? `地板 ${formatNumber(floorVal, 2)}` : "",
  ].filter((s) => s !== "");
  scene.badges.push(placeBadge(badge("status", status.join("；"), 8, 8), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  const nBad =
    countNonFinite(nFd) +
    countNonFinite(fdError) +
    countNonFinite(nCheb) +
    countNonFinite(chebError) +
    countNonFinite(fdResidual) +
    (interpError !== undefined ? countNonFinite(interpError) : 0) +
    (fitExp !== undefined ? countNonFinite(fitExp) : 0);
  applyOverflow(scene, nBad, { slot: SLOT.overflow, width });
  return scene;
};

// --------------------------------------------------------------------------- //
// 局部工具（只做显示几何：把非正值从对数轴曲线上摘掉，不重算任何物理量）
// --------------------------------------------------------------------------- //

function segmentedPositive(
  x: ArrayLike<number>,
  y: ArrayLike<number>,
  color: string,
  style: { width?: number; dash?: number[]; marker?: "none" | "o" | "s" | "x" | "+"; label?: string },
): LineSeries[] {
  return segmentedSeries(x, y, color, { ...style, accept: (v) => v > 0 });
}

function firstN(a: ArrayLike<number>, n: number): Float64Array {
  const out = new Float64Array(Math.max(0, Math.min(n, a.length)));
  for (let i = 0; i < out.length; i += 1) out[i] = a[i] as number;
  return out;
}

function positivePairs(x: ArrayLike<number>, y: ArrayLike<number>): { h: Float64Array; err: Float64Array } {
  const xs: number[] = [];
  const ys: number[] = [];
  const n = Math.min(x.length, y.length);
  for (let i = 0; i < n; i += 1) {
    const xv = x[i] as number;
    const yv = y[i] as number;
    if (Number.isFinite(xv) && xv > 0 && Number.isFinite(yv) && yv > 0) {
      xs.push(xv);
      ys.push(yv);
    }
  }
  return { h: Float64Array.from(xs), err: Float64Array.from(ys) };
}
