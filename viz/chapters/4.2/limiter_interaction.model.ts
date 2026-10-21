/**
 * `limiter_interaction`（4.2 章）的 model：限制器不是「装了就更准」，而是「用一阶换不振荡」。
 *
 * 版式（2×2，四块面板都是证据）
 * ----------------------------
 * | 面板 | 轴 | 内容 |
 * |---|---|---|
 * | 左上 | `solution` | 阶跃解 / 精确解 / 初值 + 正向误差（红带）与负向误差（绿带） |
 * | 右上 | `sweby` | **Sweby 允许区**（绿带 0≤φ≤min(2r,2)）+ 边界线 + φ(r) + 当前解的 (r,φ) 点 |
 * | 左下 | `tv` | TV(t)，固定窗口 1.8–2.2（TVD ⟺ 这条线不上升） |
 * | 右下 | `residual` | TV 的逐步增量 \|ΔTV\|（对数，自适应） |
 *
 * 三条防误导约定
 * --------------
 * 1. **Sweby 区域画成有边界的带**，不是一句「落在区域内」：上界直接取帧的 `sweby_upper`
 *    （min(2r,2)），下界取 `sweby_lower`（0）；r<0 的竖带单独画并标注「要求 φ=0」。
 *    区域内的点**不代表二阶**（limiter=none 的 φ≡0 也在区域内）。
 * 2. **正向/负向误差带画在解面板上（带高 = 误差）**。帧里 0 表示该点没有这个方向的
 *    误差，不画到地板上；区间外过冲/欠冲另取 `meta.overshoot/undershoot` 报数，
 *    两者不是一回事（抹平会让两侧误差都不为零，但区间外过冲仍是 0）。
 * 3. **TV 窗固定 1.8–2.2、Sweby 窗固定**（manifest 声明 `autoscaled=false`）：
 *    ν>1 时 TV 冲出窗口，用溢出角标点名，不用自适应把它压回来。
 *
 * 时间轴：`scrub`，`t` 是真实时间；TV 的横轴是步号 k（每步 Δt=νΔx）。
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
  type SceneModel,
} from "../../shared/src/model.js";
import {
  badge,
  criticalBadge,
  lineLayer,
  makeAxis,
  scatterLayer,
  series,
  sourceBadge,
} from "../../shared/src/scene.js";
import {
  TEXT_ROW_H,
  axisTitleOf,
  axisYlabelOf,
  columnsLayout,
  logRange,
  positiveMinMax,
  sliceRun,
  textBlock,
  unitBudget,
} from "../_common/panel.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "4.2";
const VIZ = "limiter_interaction";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "限制器交互：TVD 区域、激波削波与 CFL 的必要性";
const T_MAX = SPEC && SPEC.time.mode !== "none" ? SPEC.time.tMax ?? 0.6 : 0.6;

const Y_SOLUTION: [number, number] = [-0.5, 1.5];
const X_SWEBY: [number, number] = [-0.5, 4.5];
const Y_SWEBY: [number, number] = [-0.5, 2.5];
const Y_TV: [number, number] = [1.8, 2.2];
const FLOOR_FALLBACK = 1e-17;

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

export const limiterInteractionModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需数组 ——
  const x = requireArray(frame, "x");
  const u = requireArray(frame, "u");
  const exact = requireArray(frame, "u_exact");
  const initial = requireArray(frame, "u_initial");
  const rGrid = requireArray(frame, "r_grid");
  const phiCurve = requireArray(frame, "phi_curve");
  const upper = requireArray(frame, "sweby_upper");
  const lower = requireArray(frame, "sweby_lower");
  const tvHist = requireArray(frame, "tv_history");
  const forward = requireArray(frame, "forward_error");
  const backward = requireArray(frame, "backward_error");
  const tvResidual = requireArray(frame, "tv_residual");
  for (const [name, arr] of [
    ["u", u],
    ["u_exact", exact],
    ["u_initial", initial],
    ["forward_error", forward],
    ["backward_error", backward],
  ] as const) {
    requireSameLength(x, arr, `帧数组 x 与 ${name}`);
  }
  for (const [name, arr] of [
    ["phi_curve", phiCurve],
    ["sweby_upper", upper],
    ["sweby_lower", lower],
  ] as const) {
    requireSameLength(rGrid, arr, `帧数组 r_grid 与 ${name}`);
  }
  const pitfalls = requirePitfalls(frame);

  // —— 可选数组 ——
  const degraded: string[] = [];
  const inSweby = frame.has("in_sweby") ? requireArray(frame, "in_sweby") : undefined;
  if (inSweby === undefined) degraded.push("帧缺 in_sweby：比例只取 meta.sweby_fraction");
  else requireSameLength(rGrid, inSweby, "帧数组 r_grid 与 in_sweby");
  const rSol = frame.has("r_at_solution") ? requireArray(frame, "r_at_solution") : undefined;
  const phiSol = frame.has("phi_at_solution") ? requireArray(frame, "phi_at_solution") : undefined;
  const rValid = frame.has("r_valid") ? requireArray(frame, "r_valid") : undefined;
  if (rSol === undefined || phiSol === undefined || rValid === undefined) {
    degraded.push("帧缺 r_at_solution/phi_at_solution/r_valid：不画当前解上的 (r,φ) 点");
  } else {
    requireSameLength(x, rSol, "帧数组 x 与 r_at_solution");
    requireSameLength(x, phiSol, "帧数组 x 与 phi_at_solution");
    requireSameLength(x, rValid, "帧数组 x 与 r_valid");
  }
  const algRes = frame.has("algebraic_residual") ? requireArray(frame, "algebraic_residual") : undefined;
  if (algRes === undefined) degraded.push("帧缺 algebraic_residual：不画代数残差参考线");
  const uClipped = frame.has("u_clipped") ? requireArray(frame, "u_clipped") : undefined;
  if (uClipped === undefined) degraded.push("帧缺 u_clipped：不统计截断点数");
  else requireSameLength(x, uClipped, "帧数组 x 与 u_clipped");

  // —— meta ——
  const tActualMeta = metaNumber(frame, "t_actual");
  const tActual = tActualMeta ?? clamp(opts.t, 0, T_MAX);
  if (tActualMeta === undefined) degraded.push("帧 meta 缺 t_actual：时间按请求值并夹到 [0,tMax]");
  const tRequested = metaNumber(frame, "t_requested") ?? opts.t;
  if (Math.abs(tActual - clamp(tRequested, 0, T_MAX)) > 1e-9) {
    degraded.push(`请求 t=${formatNumber(tRequested, 4)} 实到 ${formatNumber(tActual, 4)}（步数取整）`);
  }
  const nu = metaNumber(frame, "nu");
  if (nu === undefined) degraded.push("帧 meta 缺 nu：不显示 CFL 数（不按请求参数猜）");
  const nx = metaNumber(frame, "nx");
  if (nx === undefined) degraded.push("帧 meta 缺 nx：不显示格点数");
  const nSteps = metaNumber(frame, "n_steps");
  const tvNow = metaNumber(frame, "tv_now");
  const tvInit = metaNumber(frame, "tv_initial");
  const tvRel = metaNumber(frame, "tv_max_increase_rel");
  if (tvRel === undefined) degraded.push("帧 meta 缺 tv_max_increase_rel：不报 TV 的最大相对增量");
  const swebyFrac = metaNumber(frame, "sweby_fraction");
  if (swebyFrac === undefined) degraded.push("帧 meta 缺 sweby_fraction：比例改由 in_sweby 掩码统计");
  const maxG = metaNumber(frame, "max_abs_symbol");
  const cflExceeded = frame.meta["cfl_exceeded"] === true;
  const tvdHoldsMeta = frame.meta["tvd_holds"];
  const tvdHolds = typeof tvdHoldsMeta === "boolean" ? tvdHoldsMeta : undefined;
  if (tvdHolds === undefined) degraded.push("帧 meta 缺 tvd_holds：不判 TVD（不自己复算总变差）");
  const nRUndefined = metaNumber(frame, "n_r_undefined");
  const swebyFracShown = swebyFrac ?? (inSweby !== undefined ? meanMask(inSweby) : undefined);
  const secondOrderFrac = metaNumber(frame, "second_order_fraction");
  const overMeta = metaNumber(frame, "overshoot");
  const underMeta = metaNumber(frame, "undershoot");
  const limiter = stringOf(frame, "limiter");
  const floorText = stringOf(frame, "floor");
  const floor = numberOr(floorText, FLOOR_FALLBACK);
  if (floorText === undefined) degraded.push("帧 strings 缺 floor：地板标记按 1e-17");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: tActual,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  // —— 2×2 版式：左右两列由 columnsLayout 定 x，行高自己分 ——
  const L = columnsLayout(scene, width, height, { topShare: 0.6, auxH: 110, gap: 46 });
  const equal = !L.columns;
  const left = L.left;
  const colGap = 56;
  const colW = equal ? Math.max(80, Math.floor((width - left - 24 - colGap) / 2)) : L.leftW;
  const rightX = equal ? left + colW + colGap : L.rightX;
  const rightW = equal ? colW : L.rightW;
  const top = L.topY;
  // 上行 xlabel（面板底 +29px）与下行标题（行顶 −19px）之间需要 ≥8px：46+7=53，取 56。
  const gapY = 56;
  const rowH = Math.max(24, Math.floor((L.floor - top - gapY) / 2));
  const row2 = top + rowH + gapY;
  const solutionAxis = makeAxis("solution", [left, top, colW, rowH], {
    xlim: [0, 1],
    ylim: Y_SOLUTION,
    title: axisTitleOf(SPEC?.axes, "solution", "阶跃解与精确解（固定窗口）"),
    ylabel: axisYlabelOf(SPEC?.axes, "solution", "$u(x,t)$"),
    xlabel: "$x$（格点，域长 $L=1$）",
    autoscaled: false,
  });
  const swebyAxis = makeAxis("sweby", [rightX, top, rightW, rowH], {
    xlim: X_SWEBY,
    ylim: Y_SWEBY,
    title: axisTitleOf(SPEC?.axes, "sweby", "Sweby 图 φ(r) 与允许区"),
    ylabel: axisYlabelOf(SPEC?.axes, "sweby", "φ(r)"),
    xlabel: "斜率比 $r$",
    autoscaled: false,
  });
  const tvAxis = makeAxis("tv", [left, row2, colW, rowH], {
    xlim: [0, Math.max(1, tvHist.length - 1)],
    ylim: Y_TV,
    title: axisTitleOf(SPEC?.axes, "tv", "总变差 TV(t)（固定窗口：不上升即 TVD）"),
    ylabel: axisYlabelOf(SPEC?.axes, "tv", "TV"),
    xlabel: "步号 $k$（每步 $Δt=νΔx$）",
    autoscaled: false,
  });
  const resLo = positiveMinMax(tvResidual)?.[0] ?? floor;
  const resHi = positiveMinMax(tvResidual)?.[1] ?? 1;
  const algMax = algRes !== undefined && algRes.length > 0 ? maxFinite(algRes) : undefined;
  const residualAxis = makeAxis("residual", [rightX, row2, rightW, rowH], {
    xlim: [0, Math.max(1, tvResidual.length - 1)],
    ylim: logRange(resLo * 0.5, Math.max(resHi, algMax ?? 0, floor) * 2),
    yscale: "log",
    title: "TV 增量 |ΔTV|（对数）",
    ylabel: "$|ΔTV|$",
    xlabel: "步号 $k$",
    autoscaled: true,
  });
  scene.axes.push(solutionAxis, swebyAxis, tvAxis, residualAxis);

  // —— 左上：解 + 正/负向误差带（带高就是误差）——
  scene.layers.push(
    lineLayer("solution", series(x, initial, pal.reference, { width: 1.1, dash: [4, 4] })),
    lineLayer("solution", series(x, exact, pal.exact, { width: 2 })),
    lineLayer("solution", series(x, u, pal.curves[0], { width: 2.3 })),
  );
  pushErrorBands(scene, "solution", x, exact, u, forward, pal.warn);
  pushErrorBands(scene, "solution", x, u, exact, backward, pal.curves[1]);

  // —— 右上：Sweby 允许区（先画带，再画边界与曲线）——
  scene.layers.push({
    kind: "band",
    axes: "sweby",
    band: {
      x: rGrid,
      ylo: lower,
      yhi: upper,
      color: pal.curves[1],
      alpha: 0.12,
      label: "Sweby 允许区",
    },
  });
  // r<0：φ 必须为 0（不是「区域外可以随便」）。
  scene.layers.push({
    kind: "band",
    axes: "sweby",
    band: {
      x: [X_SWEBY[0], 0],
      ylo: [Y_SWEBY[0], Y_SWEBY[0]],
      yhi: [Y_SWEBY[1], Y_SWEBY[1]],
      color: pal.reference,
      alpha: 0.1,
      label: "$r<0$：要求 $φ=0$",
    },
  });
  scene.layers.push(
    lineLayer("sweby", series([1, 1], [Y_SWEBY[0], Y_SWEBY[1]], pal.reference, { width: 0.9, dash: [3, 3] })),
    lineLayer("sweby", series([X_SWEBY[0], X_SWEBY[1]], [1, 1], pal.reference, { width: 0.9, dash: [3, 3] })),
    lineLayer("sweby", series(rGrid, upper, pal.curves[1], { width: 1.6, dash: [5, 3] })),
    lineLayer("sweby", series(rGrid, lower, pal.reference, { width: 1, dash: [2, 3] })),
    lineLayer("sweby", series(rGrid, phiCurve, pal.curves[0], { width: 2.2 })),
  );
  if (rSol !== undefined && phiSol !== undefined && rValid !== undefined) {
    let n = 0;
    for (let i = 0; i < x.length; i += 1) if ((rValid[i] ?? 0) > 0.5) n += 1;
    if (n > 0) {
      const xs = new Float64Array(n);
      const ys = new Float64Array(n);
      let k = 0;
      for (let i = 0; i < x.length; i += 1) {
        if ((rValid[i] ?? 0) <= 0.5) continue;
        const rv = rSol[i] as number;
        const pv = phiSol[i] as number;
        if (!Number.isFinite(rv) || !Number.isFinite(pv)) continue;
        xs[k] = Math.max(X_SWEBY[0], Math.min(X_SWEBY[1], rv));
        ys[k] = Math.max(Y_SWEBY[0], Math.min(Y_SWEBY[1], pv));
        k += 1;
      }
      scene.layers.push(scatterLayer("sweby", xs.subarray(0, k), ys.subarray(0, k), { r: 2.4, color: pal.marker, alpha: 0.8 }));
    }
  }

  // —— 左下：TV(t) ——
  const tvX = new Float64Array(tvHist.length);
  for (let i = 0; i < tvX.length; i += 1) tvX[i] = i;
  if (tvHist.length > 0) {
    scene.layers.push(
      lineLayer("tv", series(tvX, tvHist, pal.curves[0], { width: 2.2 })),
      scatterLayer("tv", [tvX[tvX.length - 1] as number], [tvHist[tvHist.length - 1] as number], {
        r: 3.4,
        color: pal.marker,
      }),
    );
  }
  if (tvInit !== undefined) {
    scene.layers.push(
      lineLayer("tv", series([0, Math.max(1, tvHist.length - 1)], [tvInit, tvInit], pal.reference, {
        width: 1,
        dash: [5, 4],
      })),
    );
  }

  // —— 右下：|ΔTV|（残差副轴）——
  const resX = new Float64Array(tvResidual.length);
  for (let i = 0; i < resX.length; i += 1) resX[i] = i;
  if (tvResidual.length > 0) {
    scene.layers.push(
      lineLayer("residual", series(resX, tvResidual, pal.residual, { width: 1.8 })),
      lineLayer(
        "residual",
        series([0, Math.max(1, tvResidual.length - 1)], [floor, floor], pal.reference, { width: 1, dash: [2, 3] }),
      ),
    );
  }
  if (algMax !== undefined && algMax > 0) {
    scene.layers.push(
      lineLayer(
        "residual",
        series([0, Math.max(1, tvResidual.length - 1)], [Math.max(algMax, floor), Math.max(algMax, floor)], pal.warn, {
          width: 1,
          dash: [4, 3],
        }),
      ),
    );
  }

  // —— 面板内图例（窄面板只放一行短图例）——
  const nOver = countPositive(forward);
  const nUnder = countPositive(backward);
  const nClip = uClipped !== undefined ? countMask(uClipped) : 0;
  const tvLast = tvHist.length > 0 ? (tvHist[tvHist.length - 1] as number) : undefined;
  const tvShown = tvNow ?? tvLast;
  if (rowH >= 60) {
    // 一个面板的图例一次性交给 textBlock：折行后逐行下移，绝不按固定 y 偏移叠字。
    scene.texts.push(
      ...textBlock(
        left + 6,
        top + 6,
        top + 6 + 4 * TEXT_ROW_H,
        ["蓝=MUSCL　橙=精确　灰虚=初值", `红=正向误差（${nOver} 点）　绿=负向（${nUnder} 点）；带高=误差`],
        pal.muted,
        unitBudget(colW),
        "top",
      ),
      ...textBlock(
        rightX + 6,
        top + 6,
        top + 6 + 4 * TEXT_ROW_H,
        [
          "绿带=允许区 0≤φ≤min(2r,2)",
          `r<0 灰带=φ 必须为 0；区内 ${swebyFracShown !== undefined ? formatNumber(swebyFracShown * 100, 4) : "?"}%；二阶 ${fmt(secondOrderFrac, 3)}`,
        ],
        pal.muted,
        unitBudget(rightW),
        "top",
      ),
      ...textBlock(
        left + 6,
        row2 + 6,
        row2 + 6 + 4 * TEXT_ROW_H,
        [
          "窗口 1.8–2.2：TVD 要求这条线不上升",
          `TV=${fmt(tvShown, 8)}（初值 ${fmt(tvInit, 4)}）；区间外过冲 ${fmt(overMeta, 3)} / 欠冲 ${fmt(underMeta, 3)}`,
          `TVD ${tvdHolds === undefined ? "未判" : tvdHolds ? "成立" : "破坏"}；r 未定义 ${fmt(nRUndefined, 0)}/${x.length}；代数残差 0 不画`,
        ],
        pal.muted,
        unitBudget(colW),
        "top",
      ),
      ...textBlock(
        rightX + 6,
        row2 + 6,
        row2 + 6 + 4 * TEXT_ROW_H,
        ["紫=|ΔTV|（对数轴）", `最大增量 ${fmt(tvRel, 3)}（地板=舍入级）`],
        pal.muted,
        unitBudget(rightW),
        "top",
      ),
    );
  }

  // —— 两列版式的窄右列放不下通栏说明：关键数字进图例与状态角标，长句在 pitfalls ——
  const head = paramSummary(frame, opts, [
    { name: "nu", label: "$ν$" },
    { name: "nx", label: "$nx$" },
  ]);
  degraded.push(...head.degraded);
  if (equal) degraded.push("窄画布（单列）：四块面板等宽排布，长说明未排下");

  // —— 角标 ——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  scene.badges.push(
    placeBadge(badge("autoscale", "解/Sweby/TV 固定；TV 增量副轴自适应", 8, 26), SLOT.axisMode, width),
  );
  scene.badges.push(
    placeBadge(badge("autoscale", "Sweby 窗 r∈−0.5–4.5；TV 窗 1.8–2.2（不随帧变）", 8, 44), SLOT.auxAxis, width),
  );
  if (nu !== undefined) {
    scene.badges.push(
      placeBadge(criticalBadge("ν", nu, [criticalValue(frame, 1.0)], { beyond: "above" }), SLOT.critical, width),
    );
  }
  const status: string[] = [];
  if (limiter !== undefined) status.push(limiter);
  if (tvdHolds !== undefined) status.push(tvdHolds ? "TVD 成立" : "TVD 破坏");
  if (tvNow !== undefined) status.push(`TV ${formatNumber(tvNow, 4)}`);
  if (nSteps !== undefined) status.push(`k ${formatNumber(nSteps, 0)}`);
  if (cflExceeded) status.push("CFL 越界");
  if (maxG !== undefined) status.push(`max|G| ${formatNumber(maxG, 3)}`);
  status.push(`区间外过冲/欠冲 ${fmt(overMeta, 3)}/${fmt(underMeta, 3)}`);
  if (nClip > 0) status.push(`解截断 ${nClip}/${x.length}`);
  scene.badges.push(placeBadge(badge("status", status.join("；"), 8, 8), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  const nBad =
    countNonFinite(x) +
    countNonFinite(u) +
    countNonFinite(phiCurve) +
    countNonFinite(tvHist) +
    countNonFinite(tvResidual) +
    (uClipped !== undefined ? countNonFinite(uClipped) : 0) +
    (inSweby !== undefined ? countNonFinite(inSweby) : 0) +
    (rSol !== undefined ? countNonFinite(rSol) : 0) +
    (phiSol !== undefined ? countNonFinite(phiSol) : 0) +
    (rValid !== undefined ? countNonFinite(rValid) : 0) +
    (algRes !== undefined ? countNonFinite(algRes) : 0);
  applyOverflow(scene, nBad, { slot: SLOT.overflow, width });
  return scene;
};

// --------------------------------------------------------------------------- //
// 局部工具（显示几何与计数，不算物理量）
// --------------------------------------------------------------------------- //

/**
 * 过冲/欠冲带：`mask = hi − lo > 0` 的连续段画成带，带的两端就是 `lo`/`hi`。
 * 调用方把 (lo,hi) 换成 (exact,u) 或 (u,exact)，于是红带/绿带的高度**就是**误差。
 */
function pushErrorBands(
  scene: ReturnType<SceneModel>,
  axis: string,
  x: ArrayLike<number>,
  lo: ArrayLike<number>,
  hi: ArrayLike<number>,
  mask: ArrayLike<number>,
  color: string,
): void {
  let start = -1;
  const push = (end: number): void => {
    if (start < 0 || end - start < 1) return;
    const sx = sliceRun(x, x, { start, end });
    const sy = sliceRun(lo, hi, { start, end });
    scene.layers.push({
      kind: "band",
      axes: axis,
      band: { x: sx.x, ylo: sy.x, yhi: sy.y, color, alpha: 0.22 },
    });
  };
  for (let i = 0; i < mask.length; i += 1) {
    const on = (mask[i] ?? 0) > 0;
    if (on && start < 0) start = i;
    if (!on && start >= 0) {
      push(i);
      start = -1;
    }
  }
  push(mask.length);
  // 单点段（带没有宽度）也要看得见：把 >0 的点画成散点。
  let n = 0;
  for (let i = 0; i < mask.length; i += 1) if ((mask[i] ?? 0) > 0) n += 1;
  if (n === 0) return;
  const xs = new Float64Array(n);
  const ys = new Float64Array(n);
  let k = 0;
  for (let i = 0; i < mask.length; i += 1) {
    if ((mask[i] ?? 0) <= 0) continue;
    xs[k] = x[i] as number;
    ys[k] = hi[i] as number;
    k += 1;
  }
  scene.layers.push(scatterLayer(axis, xs, ys, { r: 1.8, color, alpha: 0.75 }));
}

function meanMask(a: ArrayLike<number>): number | undefined {
  if (a.length === 0) return undefined;
  let n = 0;
  for (let i = 0; i < a.length; i += 1) if ((a[i] ?? 0) > 0.5) n += 1;
  return n / a.length;
}

function countMask(a: ArrayLike<number>): number {
  let n = 0;
  for (let i = 0; i < a.length; i += 1) if ((a[i] ?? 0) > 0.5) n += 1;
  return n;
}

function countPositive(a: ArrayLike<number>): number {
  let n = 0;
  for (let i = 0; i < a.length; i += 1) if ((a[i] ?? 0) > 0) n += 1;
  return n;
}

function maxFinite(a: ArrayLike<number>): number | undefined {
  let hi = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i];
    if (Number.isFinite(v) && v > hi) hi = v;
  }
  return Number.isFinite(hi) ? hi : undefined;
}

function criticalValue(frame: Parameters<SceneModel>[0], fallback: number): number {
  const raw = frame.critical["nu"];
  return typeof raw === "number" && Number.isFinite(raw) ? raw : fallback;
}

function fmt(v: number | undefined, digits: number): string {
  return v === undefined || !Number.isFinite(v) ? "?" : formatNumber(v, digits);
}

function numberOr(text: string | undefined, fallback: number): number {
  if (text === undefined) return fallback;
  const v = Number(text);
  return Number.isFinite(v) && v > 0 ? v : fallback;
}
