/**
 * `normal_vs_qr`（1.3 章）的 model：κ 扫描下三条路线的前向误差、残差与"分道扬镳"比值。
 *
 * 三个面板（manifest 声明的三条轴**都在**，顺序即声明顺序）
 * -------------------------------------------------------
 * | 轴 | 位置 | 内容 | 自适应 |
 * |---|---|---|---|
 * | `errors` | 左上 | 前向误差 vs log10 κ（双对数），三条路线 + 两条理论参考线 + 上限线 | **固定** 1e-17..1e2 |
 * | `residual` | 右整列 | 相对残差（对数）——三条路线都贴着 u，看不出差别 | 自适应 |
 * | `divergence` | 左下 | 误差之比 正规方程/QR（对数） | **固定** 1e-18..1e12 |
 *
 * 四条刻意的决定
 * --------------
 * 1. **上限 `error_ceiling = 1.0` 是"没有有效数字"的上限，不是测量值**（与 1.2 章
 *    `error_valid` 同一条纪律）。某条路线在某个 κ 上解不出来时，后端把误差写成这个上限
 *    并以 `singular_normal` / `singular_qr` 标出。模型据此把曲线**在该点截断**
 *    （画到最后一个有效点为止），画一条上限参考线并**显式说明它是上限**——
 *    若把 1.0 当真值连线，读图的人会得到"误差饱和在 1 附近"这个错误结论。
 *    `error_normal_ir` 没有对应的 singular 数组（Python 层未给），模型**借用**
 *    `singular_normal` 作保守掩码并在文本里说明这一点（不静默）。
 * 2. **理论参考线用帧里的数组画**：`theory_kappa2`/`theory_kappa1` 是后端算好的，
 *    前端不重算 κu 与 κ²u（P8）。实测斜率取 `meta.slope_*`（后端在**未饱和窗口**内拟合），
 *    窗口上界的对数写在 `meta.slope_window_*` 里——斜率不写窗口就没法读（pitfall 2）。
 * 3. **主轴固定、副轴自适应都要标出来**（规则 1/4）：`errors` 轴固定是为了让
 *    "正规方程在某个 κ 之后彻底崩掉"这件事在屏幕上真的看得见；`residual` 轴自适应
 *    只为了让三条贴地板的曲线分得开，它的角标明确写"自适应会掩盖量级变化"。
 * 4. **时间轴是 log10 κ**：`t` 就是当前扫描位置，夹在 `kappa_grid` 的实际范围内，
 *    真正被标注的实测值来自 `kappa_mark`/`error_mark`/`residual_mark`（帧里给好的）。
 */

import { niceLimits } from "../../shared/src/axes.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import {
  ModelInputError,
  canvasSize,
  clipBadgeText,
  criticalOf,
  dataSourceOf,
  degradedBadge,
  finitePositiveMinMax,
  metaNumber,
  palette,
  placeBadge,
  requireArray,
  requireDrawableLayers,
  resolveTextOverlaps,
  requirePitfalls,
  requireSameLength,
  sceneShell,
  stackedPanels,
  stringOf,
  type Rect,
  type SceneModel,
} from "../../shared/src/model.js";
import {
  axisModeBadge,
  badge,
  criticalBadge,
  lineLayer,
  makeAxis,
  previewBadge,
  scatterLayer,
  series,
  sourceBadge,
  type LineSeries,
  pitfallsBand,
} from "../../shared/src/scene.js";
import { contentBottom, fmt, fmtOr, joinParts, kv, overflowBadgeFor, textBlock, textFloor, unitBudget } from "./_shared.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "1.3";
const VIZ = "normal_vs_qr";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "精度对决：正规方程的 κ² 与 QR 的 κ";

/** 固定角标槽位（见 `model.ts::BADGE_LAYOUT`）。 */
const SLOT = {
  source: 0,
  axisMode: 1,
  auxAxis: 2,
  critical: 3,
  status: 4,
  preview: 5,
  overflow: 6,
  degraded: 7,
} as const;

/** manifest 声明的固定纵轴范围（`errors` 轴）：图上看不见的那一段必须由数字说出来。 */
const ERRORS_YLIM: [number, number] = [1e-17, 1e2];
/** `divergence` 轴固定范围：后端把比值抬到地板 `_LOG_FLOOR = 1e-18`，上界取 1e12。 */
const DIVERGENCE_YLIM: [number, number] = [1e-18, 1e12];
/** `log10(1/u)` 的临界值（manifest；`frame.critical.exp_max` 缺失时的回落值）。 */
const EXP_CRITICAL = 15.95;

/** 一条误差曲线的画法（颜色、线型、标签）。名字与帧里的数组一致。 */
interface Route {
  key: string;
  label: string;
  color: string;
  width: number;
  dash?: number[];
  /** 掩码数组名（后端逐路线标注"这条路线在该 κ 上没有有限解"）；缺省表示后端没给。 */
  mask?: string;
}

const ROUTES: readonly Route[] = [
  { key: "error_normal", label: "正规方程 AᵀA", color: "#2b6cb0", width: 1.9, mask: "singular_normal" },
  { key: "error_qr", label: "QR（Householder）", color: "#2f855a", width: 1.9, mask: "singular_qr" },
  { key: "error_svd", label: "SVD 截断", color: "#b7791f", width: 1.5, dash: [5, 3] },
  { key: "error_normal_ir", label: "正规方程 + 迭代精化", color: "#805ad5", width: 1.4, dash: [2, 2], mask: "singular_normal" },
];

export const normalVsQrModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需字段 ——
  const kappaGrid = requireArray(frame, "kappa_grid");
  const errorNormal = requireArray(frame, "error_normal");
  const errorQr = requireArray(frame, "error_qr");
  const errorSvd = requireArray(frame, "error_svd");
  const residualNormal = requireArray(frame, "residual_normal");
  const residualQr = requireArray(frame, "residual_qr");
  const theory2 = requireArray(frame, "theory_kappa2");
  const theory1 = requireArray(frame, "theory_kappa1");
  const divergence = requireArray(frame, "divergence");
  const kappaMark = requireArray(frame, "kappa_mark");
  const errorMark = requireArray(frame, "error_mark");
  const residualMark = requireArray(frame, "residual_mark");
  const pitfalls = requirePitfalls(frame);
  for (const [name, arr] of [
    ["error_normal", errorNormal],
    ["error_qr", errorQr],
    ["error_svd", errorSvd],
    ["residual_normal", residualNormal],
    ["residual_qr", residualQr],
    ["theory_kappa2", theory2],
    ["theory_kappa1", theory1],
    ["divergence", divergence],
  ] as const) {
    requireSameLength(kappaGrid, arr, `帧数组 kappa_grid 与 ${name}`);
  }
  if (errorMark.length < 3 || residualMark.length < 2) {
    throw new ModelInputError(
      `帧 ${CHAPTER}/${VIZ} 的 error_mark/residual_mark 长度不对：` +
        `error_mark=${errorMark.length}（期望 ≥3）、residual_mark=${residualMark.length}（期望 ≥2）`,
    );
  }

  // —— 可选字段（有就画、没有就降级）——
  const degraded: string[] = [];
  const errorNormalIr = frame.has("error_normal_ir") ? requireArray(frame, "error_normal_ir") : undefined;
  const errorMarkIr = frame.has("error_mark_ir") ? requireArray(frame, "error_mark_ir") : undefined;
  if (errorNormalIr !== undefined) requireSameLength(kappaGrid, errorNormalIr, "帧数组 kappa_grid 与 error_normal_ir");
  const masks: Record<string, ArrayLike<number> | undefined> = {};
  for (const name of ["singular_normal", "singular_qr", "singular_svd", "singular_normal_ir"]) {
    if (frame.has(name)) {
      masks[name] = requireArray(frame, name);
      requireSameLength(kappaGrid, masks[name] as ArrayLike<number>, `帧数组 kappa_grid 与 ${name}`);
    }
  }
  const tMeta = metaNumber(frame, "t");
  if (tMeta === undefined) degraded.push("帧 meta 缺 t：scene.t 用请求 t（两者可能不同）");
  const tNow = tMeta ?? opts.t;
  const errorCeiling = metaNumber(frame, "error_ceiling");
  if (errorCeiling === undefined) {
    degraded.push("帧 meta 缺 error_ceiling：曲线不按上限截断（本帧若含失败点会把上限画成实测值）");
  }
  const errorFloor = metaNumber(frame, "error_floor");
  const residualFloor = metaNumber(frame, "residual_floor");
  const slopeNormal = metaNumber(frame, "slope_normal");
  const slopeQr = metaNumber(frame, "slope_qr");
  const slopeSvd = metaNumber(frame, "slope_svd");
  const slopeNormalIr = metaNumber(frame, "slope_normal_ir");
  const winNormal = metaNumber(frame, "slope_window_normal");
  const winQr = metaNumber(frame, "slope_window_qr");
  const nSingNormal = metaNumber(frame, "n_singular_normal");
  const nSingQr = metaNumber(frame, "n_singular_qr");
  const nPoints = metaNumber(frame, "n_points");
  const mSize = metaNumber(frame, "m");
  const nSize = metaNumber(frame, "n");
  const nSeeds = metaNumber(frame, "n_seeds");
  const refinementSteps = metaNumber(frame, "refinement_steps");
  const kappaCurrent = metaNumber(frame, "kappa_current");
  const log10KappaCurrent = metaNumber(frame, "log10_kappa_current");
  const errorNormalCur = metaNumber(frame, "error_normal_current");
  const errorQrCur = metaNumber(frame, "error_qr_current");
  const errorSvdCur = metaNumber(frame, "error_svd_current");
  const errorIrCur = metaNumber(frame, "error_ir_current");
  const ratioCur = metaNumber(frame, "error_ratio_current");
  const defectMax = metaNumber(frame, "defect_max");
  const refinementName = stringOf(frame, "refinement");
  if (refinementName === undefined) degraded.push("帧 strings 缺 refinement：看不出精化步数（不按参数猜）");
  const status = stringOf(frame, "status");
  const note = stringOf(frame, "note");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: tNow,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  // —— 排版：左列上=误差、左列下=分道扬镳；右列整列=残差 ——
  const base = stackedPanels(width, height);
  // 底部面板（渲染层现画的 `scene.pitfalls`）会随文案长短变高：把内容底边一起收上来，
  // 否则副轴与文本的下半截会被它盖住（实测 1.3 的 fan 帧面板顶边只有 442，而 base.bottom=498）。
  base.bottom = contentBottom(frame, width, height, base.bottom);
  const gapY = 60; // 左列两块之间：上面那块的 x 刻度带 + 轴名 + 下面那块的轴标题
  const colGap = 78; // 右列的 y 刻度 + ylabel
  const leftW = Math.max(220, Math.round((base.contentWidth - colGap) * 0.5));
  const rightW = Math.max(160, base.contentWidth - leftW - colGap);
  const plotH = base.bottom - base.top;
  const topH = Math.max(90, Math.round((plotH - gapY) * 0.52));
  const errorsRect: Rect = [base.left, base.top, leftW, topH];
  const divRect: Rect = [base.left, base.top + topH + gapY, leftW, Math.max(70, plotH - topH - gapY)];
  // 右列：残差轴只占上半（220px），下半留给两块文字（见下面的文本段）
  const residualRect: Rect = [base.left + leftW + colGap, base.top, rightW, Math.min(190, plotH)];

  const kappaRange = finiteMinMaxOr(kappaGrid, [2, 14]);
  const errorsAxis = makeAxis("errors", errorsRect, {
    xlim: niceLimits(kappaRange[0], kappaRange[1], 7, { pad: 0.02 }),
    ylim: ERRORS_YLIM,
    xscale: "linear",
    yscale: "log",
    xlabel: "log10 κ",
    ylabel: axisYlabel("errors", "‖x̂−x*‖/‖x*‖"),
    title: axisTitle("errors", "前向误差 vs κ（双对数、固定轴 1e-17..1e2）"),
    autoscaled: false,
  });
  const resPos = finitePositiveMinMax(residualNormal);
  const resPosQr = finitePositiveMinMax(residualQr);
  const rLo = Math.min(resPos?.[0] ?? Number.POSITIVE_INFINITY, resPosQr?.[0] ?? Number.POSITIVE_INFINITY);
  const rHi = Math.max(resPos?.[1] ?? Number.NEGATIVE_INFINITY, resPosQr?.[1] ?? Number.NEGATIVE_INFINITY);
  const residualYlim: [number, number] =
    Number.isFinite(rLo) && Number.isFinite(rHi) ? niceLimits(rLo, rHi, 7, { log: true, pad: 0.08 }) : [1e-18, 1e-12];
  if (!(Number.isFinite(rLo) && Number.isFinite(rHi))) {
    degraded.push("residual_normal/residual_qr 里没有 >0 的有限值：残差轴用占位范围 [1e-18,1e-12]");
  }
  const residualAxis = makeAxis("residual", residualRect, {
    xlim: errorsAxis.xlim,
    ylim: residualYlim,
    yscale: "log",
    xlabel: "log10 κ",
    ylabel: axisYlabel("residual", "相对残差"),
    title: axisTitle("residual", "相对残差 ‖Ax̂−b‖/‖b‖（对数）"),
    autoscaled: true,
  });
  const divPos = finitePositiveMinMax(divergence);
  const divergenceAxis = makeAxis("divergence", divRect, {
    xlim: errorsAxis.xlim,
    ylim: DIVERGENCE_YLIM,
    yscale: "log",
    xlabel: "log10 κ",
    ylabel: axisYlabel("divergence", "比值"),
    title: axisTitle("divergence", "分道扬镳：误差之比 正规方程 / QR（对数）"),
    autoscaled: false,
  });
  if (divPos === undefined) degraded.push("divergence 里没有 >0 的有限值：比值面板只有 1 的参考线");
  scene.axes.push(errorsAxis, residualAxis, divergenceAxis);

  // —— 误差曲线：按掩码与上限**截断** ——
  const truncated: string[] = [];
  /** 误差曲线里被丢掉的非有限值个数（NaN/∞ 不进图层，最后拼进溢出角标）。 */
  let nNonFiniteCurve = 0;
  const errSeries: LineSeries[] = [];
  const drawRoute = (route: Route, values: ArrayLike<number>): void => {
    const maskName = route.mask;
    const mask = maskName !== undefined ? masks[maskName] : undefined;
    if (route.mask !== undefined && mask === undefined) {
      degraded.push(`帧里没有 ${route.mask} 掩码：${route.label} 的失败点无法与"误差恰好很大"区分`);
    }
    const xs: number[] = [];
    const ys: number[] = [];
    let firstFailed = -1;
    let firstCeil = -1;
    for (let i = 0; i < kappaGrid.length; i += 1) {
      const failed = mask !== undefined && (mask[i] as number) > 0.5;
      if (failed) {
        if (firstFailed < 0) firstFailed = i;
        break;
      }
      const v = values[i] as number;
      // 非有限值**不进图层**：`series` 里带 NaN 会被 `validateScene` 判为"图层含非有限坐标"，
      // 而浏览器与光栅器对 NaN 的处理并不一致（一个是整段不画，一个是画到原点）。
      // 丢掉的同时用 `nNonFiniteCurve` 计数，最后拼进溢出角标——不静默。
      if (!Number.isFinite(v)) {
        nNonFiniteCurve += 1;
        continue;
      }
      if (errorCeiling !== undefined && v >= errorCeiling) {
        if (firstCeil < 0) firstCeil = i;
        break;
      }
      xs.push(kappaGrid[i] as number);
      ys.push(v);
    }
    if (firstFailed >= 0) {
      truncated.push(
        `${route.label}：曲线在 log10 κ=${fmt(kappaGrid[firstFailed] as number, 4)} 处截断（该点上没有有限解）`,
      );
    } else if (firstCeil >= 0) {
      truncated.push(
        `${route.label}：曲线在 log10 κ=${fmt(kappaGrid[firstCeil] as number, 4)} 处截断（该点起误差已达上限 ${fmt(errorCeiling ?? 1, 2)}）`,
      );
    }
    if (xs.length >= 1) {
      errSeries.push(series(Float64Array.from(xs), Float64Array.from(ys), route.color, {
        width: route.width,
        dash: route.dash,
        label: route.label,
      }));
    } else {
      degraded.push(`${route.label} 的第一个扫描点就不可用：这条路线在本帧没有可画的前缀`);
    }
    if (route.mask !== undefined && mask === undefined && firstCeil >= 0) {
      degraded.push(`${route.label} 的截断依据只有上限值（缺掩码）：上限与"测量到 1"无法区分`);
    }
  };
  for (const route of ROUTES) {
    if (route.key === "error_svd") drawRoute({ ...route, mask: "singular_svd" }, errorSvd);
    else if (route.key === "error_normal_ir") {
      if (errorNormalIr === undefined) continue;
      drawRoute(route, errorNormalIr);
      if (masks.singular_normal_ir === undefined) {
        degraded.push("error_normal_ir 借用了 singular_normal 掩码（后端没有给 singular_normal_ir）：精化路线的失败点可能更早");
      }
    } else {
      drawRoute(route, route.key === "error_normal" ? errorNormal : errorQr);
    }
  }
  if (errSeries.length > 0) scene.layers.push(lineLayer("errors", errSeries));
  else degraded.push("三条误差曲线都没有可画的前缀：误差面板为空");

  // 理论参考线：**帧里的数组**（后端算好的 κ²u 与 κu），不重算（P8）。
  const theorySeries: LineSeries[] = [
    series(kappaGrid, theory2, pal.reference, { width: 1, dash: [6, 4], label: "理论参考线 κ²u（正规方程的预期阶）" }),
    series(kappaGrid, theory1, pal.reference, { width: 1, dash: [2, 3], label: "理论参考线 κu（QR/SVD 的预期阶）" }),
  ];
  scene.layers.push(lineLayer("errors", theorySeries));

  // 上限线：**不是测量值**，画成醒目的红色虚线并写明"一位有效数字都不剩"。
  if (errorCeiling !== undefined && errorCeiling > ERRORS_YLIM[0] && errorCeiling < ERRORS_YLIM[1]) {
    scene.layers.push(
      lineLayer("errors", [
        series(errorsAxis.xlim, [errorCeiling, errorCeiling], pal.warn, {
          width: 2,
          dash: [8, 5],
          label: `误差上限 ${fmt(errorCeiling, 2)}（解不出来时的占位上限，不是测量值）`,
        }),
        series(errorsAxis.xlim, [errorFloor ?? 1e-16, errorFloor ?? 1e-16], pal.muted, {
          width: 1,
          dash: [4, 4],
          label: `误差地板 ${fmt(errorFloor ?? 1e-16, 2)}（帧 meta.error_floor）`,
        }),
      ]),
    );
  } else if (errorCeiling !== undefined) {
    degraded.push(`误差上限 ${fmt(errorCeiling, 2)} 落在固定纵轴 ${ERRORS_YLIM[0]}..${ERRORS_YLIM[1]} 之外：不画上限线`);
  }

  // 当前位置标记（kappa_mark / error_mark 由后端给）
  const markXs = kappaMark.length > 0 ? (kappaMark[0] as number) : Number.NaN;
  const markPoints: { x: number[]; y: number[]; color: string; label: string }[] = [];
  const mk = (i: number, color: string, label: string): void => {
    const v = errorMark[i] as number;
    if (Number.isFinite(markXs) && Number.isFinite(v) && v > 0) markPoints.push({ x: [markXs], y: [v], color, label });
  };
  mk(0, ROUTES[0]!.color, "当前 κ 处：正规方程");
  mk(1, ROUTES[1]!.color, "当前 κ 处：QR");
  mk(2, ROUTES[2]!.color, "当前 κ 处：SVD");
  if (errorMarkIr !== undefined && errorMarkIr.length > 0) {
    const v = errorMarkIr[0] as number;
    if (Number.isFinite(markXs) && Number.isFinite(v) && v > 0) {
      markPoints.push({ x: [markXs], y: [v], color: ROUTES[3]!.color, label: "当前 κ 处：正规方程+精化" });
    }
  }
  for (const p of markPoints) {
    scene.layers.push(scatterLayer("errors", p.x, p.y, { r: 3.2, color: p.color, label: p.label }));
  }
  scene.layers.push(
    lineLayer("errors", series([markXs, markXs], ERRORS_YLIM, pal.cursor, { width: 1, dash: [3, 3] })),
  );

  // —— 残差面板：三条路线都贴着地板（这正是"残差小"骗人的地方）——
  const resSeries: LineSeries[] = [];
  const residRoutes: { arr: ArrayLike<number>; label: string; color: string; dash?: number[] }[] = [
    { arr: residualNormal, label: "正规方程", color: ROUTES[0]!.color },
    { arr: residualQr, label: "QR", color: ROUTES[1]!.color, dash: [5, 3] },
  ];
  if (frame.has("residual_svd")) {
    const rs = requireArray(frame, "residual_svd");
    requireSameLength(kappaGrid, rs, "帧数组 kappa_grid 与 residual_svd");
    residRoutes.push({ arr: rs, label: "SVD", color: ROUTES[2]!.color, dash: [2, 2] });
  }
  for (const r of residRoutes) {
    const xs: number[] = [];
    const ys: number[] = [];
    for (let i = 0; i < kappaGrid.length; i += 1) {
      const v = r.arr[i] as number;
      if (Number.isFinite(v) && v > 0) {
        xs.push(kappaGrid[i] as number);
        ys.push(v);
      }
    }
    if (xs.length >= 1) {
      resSeries.push(series(Float64Array.from(xs), Float64Array.from(ys), r.color, { width: 1.6, dash: r.dash, label: r.label }));
    } else {
      degraded.push(`相对残差（${r.label}）没有 >0 的有限值：log 轴上画不出`);
    }
  }
  if (resSeries.length > 0) scene.layers.push(lineLayer("residual", resSeries));
  if (residualFloor !== undefined && residualFloor > 0) {
    scene.layers.push(
      lineLayer("residual", [
        series(residualAxis.xlim, [residualFloor, residualFloor], pal.muted, {
          width: 1,
          dash: [4, 4],
          label: `残差地板 ${fmt(residualFloor, 2)}（帧 meta.residual_floor）`,
        }),
      ]),
    );
  }
  const resMarkX: number[] = [];
  const resMarkY: number[] = [];
  for (let i = 0; i < residualMark.length; i += 1) {
    const r = residualMark[i] as number;
    if (Number.isFinite(markXs) && Number.isFinite(r) && r > 0) {
      resMarkX.push(markXs);
      resMarkY.push(r);
    }
  }
  if (resMarkX.length > 0) {
    scene.layers.push(scatterLayer("residual", resMarkX, resMarkY, { r: 3.2, color: pal.marker, label: "当前 κ 处的相对残差" }));
  }

  // —— 分道扬镳面板 ——
  const divX: number[] = [];
  const divY: number[] = [];
  for (let i = 0; i < kappaGrid.length; i += 1) {
    const v = divergence[i] as number;
    if (Number.isFinite(v) && v > 0) {
      divX.push(kappaGrid[i] as number);
      divY.push(v);
    }
  }
  if (divX.length >= 1) {
    scene.layers.push(
      lineLayer("divergence", [
        series(Float64Array.from(divX), Float64Array.from(divY), pal.curves[0], {
          width: 1.8,
          label: "误差之比 正规方程 / QR（实测；比值 ≥1 表示正规方程更差）",
        }),
        series(divergenceAxis.xlim, [1, 1], pal.reference, { width: 1, dash: [5, 4], label: "比值 = 1：两条路线一样准" }),
      ]),
    );
    if (ratioCur !== undefined && ratioCur > 0 && Number.isFinite(markXs)) {
      scene.layers.push(
        scatterLayer("divergence", [markXs], [ratioCur], { r: 3.4, color: pal.marker, label: "当前 κ 处的比值" }),
      );
    }
  } else {
    degraded.push("divergence 全为 0 或非有限：比值面板画不出曲线（只留 1 的参考线）");
  }

  // —— 文本：**残差轴下方**那一块空档 ——
  // 原来这两块文字被画在残差轴内部（视觉上盖住整条曲线；规则 8 只查"文字压住坐标轴装饰"，
  // 不查"文字压住绘图区"，所以校验器不会报——这里按"别把图画没了"的常识改成：
  // 残差轴缩到 220px，文字占它下方的空档；随帧变长的 status/note 交给状态角标。
  const textX = residualRect[0];
  const textBudget = unitBudget(residualRect[2]);
  // 11 行（正好填满 `residualRect` 下方 342..494 的空档：`maxRows = floor(152/15)+1 = 11`）。
  // 每行都按 `unitBudget(rightW)` 量过显示宽度（≤50 单位 = 350px）：超宽会被 `wrapUnits`
  // 折成两行，整块就会溢出成"另有 N 行，画布放不下"——那句话会正好吃掉最要紧的区分。
  const info: string[] = [];
  info.push(
    joinParts([
      kv("log10 κ", log10KappaCurrent, 3),
      kv("κ", kappaCurrent, 3),
      kv("n", nSize, 0),
      kv("m", mSize, 0),
      kv("扫描点", nPoints, 0),
    ]) ?? "帧 meta 缺当前扫描位置：这一行不显示（不按 t 猜）",
  );
  info.push(
    joinParts([
      refinementName === undefined ? undefined : `精化=${refinementName}（${fmtOr(refinementSteps, 0)} 步）`,
      kv("每点种子数", nSeeds, 0),
    ]) ?? "帧 meta 缺精化设置：这一行不显示",
  );
  info.push(
    joinParts([
      kv("斜率 normal", slopeNormal, 3),
      kv("qr", slopeQr, 3),
      kv("svd", slopeSvd, 3),
    ]) ?? "帧 meta 缺 slope_*：不显示实测斜率（不重算，P8）",
  );
  info.push(
    joinParts([
      slopeNormalIr === undefined ? undefined : `normal+精化=${fmt(slopeNormalIr, 3)}`,
      kv("理论", undefined, 0),
      "理论：正规方程 2、QR/SVD 1",
    ]) ?? "帧 meta 缺 slope_*：这一行不显示（不重算）",
  );
  info.push(
    `窗口：normal ≤ ${winNormal === undefined ? "?" : fmt(winNormal, 3)}、` +
      `qr ≤ ${winQr === undefined ? "?" : fmt(winQr, 3)}。`,
  );
  info.push(
    joinParts([
      kv("误差 normal", errorNormalCur, 3),
      kv("qr", errorQrCur, 3),
      errorSvdCur === undefined ? undefined : `svd=${fmt(errorSvdCur, 3)}`,
    ]) ?? "帧 meta 缺当前误差：这一行不显示（不重算）",
  );
  info.push(
    joinParts([
      kv("残差 normal", residualMark[0] as number | undefined, 3),
      kv("qr", residualMark[1] as number | undefined, 3),
    ]) ?? "帧数组 residual_mark 缺分量：不显示当前残差",
  );
  info.push(`**残差小 ≠ 解准**；纵轴固定，不随数据缩放。`);
  // `normal+精化` 的当前误差放角标（上面那一行放下它就超宽被折行，会把整块挤爆）
  if (errorCeiling !== undefined) {
    info.push(`误差上限 ${fmt(errorCeiling, 2)} 是占位（不是测量）；曲线到它就截断。`);
  }
  scene.texts.push(
    ...textBlock(textX, residualRect[1] + residualRect[3] + 30, textFloor(scene, base.bottom - 4), info, pal.muted, textBudget, "top"),
  );

  // —— 角标 ——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  if (scene.dataSource.startsWith("backend+preview")) {
    scene.badges.push(placeBadge(previewBadge("拖动时按显示空间插值"), SLOT.preview, width));
  }
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(
      badge("autoscale", "残差轴：自适应（log 纵轴按本帧残差取整；自适应会掩盖量级变化）", 8, 8),
      SLOT.auxAxis,
      width,
    ),
  );
  scene.badges.push(
    placeBadge(criticalBadge("log10 κ", log10KappaCurrent ?? tNow, criticalOf(frame, "exp_max", EXP_CRITICAL)), SLOT.critical, width),
  );
  const statusBits: string[] = [];
  if (log10KappaCurrent !== undefined) statusBits.push(`当前 log10 κ=${fmt(log10KappaCurrent, 3)}`);
  if (nSingNormal !== undefined && nSingNormal > 0) {
    // 这条必须出现在画面上：`解不出来` 与 `误差很大` 是两种不同的失败（pitfalls 第 13 条）。
    statusBits.push(`⚠ 正规方程 ${fmt(nSingNormal, 0)} 个扫描点没有有限解："解不出来" ≠ "误差很大"`);
  }
  if (nSingQr !== undefined && nSingQr > 0) statusBits.push(`⚠ QR ${fmt(nSingQr, 0)} 个扫描点没有有限解`);
  if (errorIrCur !== undefined) statusBits.push(`normal+精化误差=${fmt(errorIrCur, 3)}`);
  if (defectMax !== undefined) statusBits.push(`正交性缺陷 ${fmt(defectMax, 3)}（与"解准不准"是两件事）`);
  if (truncated.length > 0) statusBits.push(`截断：${truncated.join("；")}`);
  if (status !== undefined && status !== "") statusBits.push(status);
  if (note !== undefined && note !== "") statusBits.push(`note：${note}`);
  scene.badges.push(
    placeBadge(
      badge("status", clipBadgeText(statusBits.length > 0 ? statusBits.join("；") : "帧未给出可判读的状态"), 8, 8),
      SLOT.status,
      width,
    ),
  );
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  let nBad =
    countNonFinite(kappaGrid) +
    countNonFinite(errorNormal) +
    countNonFinite(errorQr) +
    countNonFinite(errorSvd) +
    countNonFinite(residualNormal) +
    countNonFinite(residualQr) +
    countNonFinite(theory2) +
    countNonFinite(theory1) +
    countNonFinite(divergence) +
    countNonFinite(errorMark) +
    countNonFinite(residualMark);
  if (errorNormalIr !== undefined) nBad += countNonFinite(errorNormalIr);
  const extra: string[] = [];
  if (nNonFiniteCurve > 0) {
    extra.push(`${nNonFiniteCurve} 个非有限误差值（NaN/∞）没有进图层——曲线在该点断开，不画成 0`);
  }
  if (errorNormalIr !== undefined && masks.singular_normal_ir === undefined) {
    extra.push("error_normal_ir 缺 singular_normal_ir 掩码：借用 singular_normal 作保守截断");
  }
  if (masks.singular_svd === undefined && frame.has("error_svd")) {
    extra.push("error_svd 缺 singular_svd 掩码：只按上限值截断");
  }
  if (nSingNormal !== undefined && (nSingNormal > 0 || (nSingQr ?? 0) > 0)) {
    extra.push(`${fmt(nSingNormal, 0)}/${nPoints === undefined ? "?" : fmt(nPoints, 0)} 个扫描点上正规方程无有限解（误差数组里是上限占位）`);
  }
  overflowBadgeFor(scene, nBad, extra, { slot: SLOT.overflow, width }, "本帧没有点越出固定纵轴窗口");
  // 规则 8 的安全网（两侧校验器都查）：
  //  - 文本层之间不许互相重叠（帧里的说明一变长就会多折一行压住下一块）；
  //  - 文本不许被底部 pitfalls 面板压住（面板高度随帧里 pitfalls 的折行数变，模型事先看不见）。
  // 只丢不移：下移会推进轴的装饰带，那是另一条违规。丢弃的行数会写在画布顶部那一行里。
  const band = pitfallsBand(scene);
  scene.texts = resolveTextOverlaps(scene.texts, [8, 24, width - 16, band === null ? height - 8 : band[1] - 4]);
  return scene;
};

// --------------------------------------------------------------------------- //
// 局部工具
// --------------------------------------------------------------------------- //

function axisTitle(id: string, fallback: string): string {
  return SPEC?.axes.find((a) => a.id === id)?.title ?? fallback;
}

function axisYlabel(id: string, fallback: string): string {
  return SPEC?.axes.find((a) => a.id === id)?.ylabel ?? fallback;
}

/** 有限值的 min/max；一个有限值都没有时用回落范围。 */
function finiteMinMaxOr(a: ArrayLike<number>, fallback: [number, number]): [number, number] {
  let lo = Number.POSITIVE_INFINITY;
  let hi = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i] as number;
    if (!Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return Number.isFinite(lo) && Number.isFinite(hi) ? [lo, hi] : fallback;
}
