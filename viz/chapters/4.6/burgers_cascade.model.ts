/**
 * `burgers_cascade`（4.6 章 viz 4）的 model：Burgers 的能谱、能流与激波。
 *
 * 三条不同量必须分开读（对应三块面板 + 一条副轴）
 * ------------------------------------------------
 * 1. **能谱斜率**（`spectrum`）：双对数、固定 1e-18–1。初值能谱（灰虚）、当前能谱（蓝）、
 *    帧给的 k^-2 参考线（橙虚）同屏；拟合窗口 [fit_lo, fit_hi] 用两条细竖线标出，
 *    斜率与 R² 写进图例。`slope_theory=-2` 是理论值，`slope_fit` 是实测值，两者都报。
 * 2. **能量通量**（`flux`）：固定 -0.3–0.3。Pi=0 参考线、耗散率 eps、Pi_max 标记与
 *    k_d 竖线同屏。`Pi_max/eps<1` 说明这里不是惯性区；末端低于 -0.3（欠分辨）会由
 *    溢出角标点名——这正是「能流不闭合」的数值证据，不能藏。
 * 3. **激波位置**（`solution`）：解曲线固定 ±1.6；服务器把超过 ±1.6 的点截断并配
 *    `u_clipped` 掩码（meta.n_clipped）。模型用**相邻最大落差**示踪激波，只在
 *    落差 > 0.2·max|u| 时画竖线（t=0 的平滑正弦不会被误标成激波）；这是标注几何，
 *    不是物理量。
 * 4. **误差/残差副轴**（`residual`）：能量漂移 `energy_error` 与收支残差
 *    `budget_residual` 两条真实曲线，固定 1e-18–10。低于 1e-18 的点由服务器放在
 *    地板上并配掩码（=完美闭合），本轴不再重算任何量。
 *
 * 帧 `time.mode="scrub"`：`scene.t` 取 `meta.t_actual`（请求 t 可能被夹到 [0, 1.5]）。
 */

import { formatNumber } from "../../shared/src/axes.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import {
  applyOverflow,
  canvasSize,
  criticalOf,
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
  criticalBadge,
  lineLayer,
  makeAxis,
  scatterLayer,
  series,
  sourceBadge,
} from "../../shared/src/scene.js";
import {
  SAFE_GAP,
  TEXT_ROW_H,
  axisTitleOf,
  axisYlabelOf,
  columnsLayout,
  selectValues,
  textBlock,
  unitBudget,
} from "../_common/panel.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "4.6";
const VIZ = "burgers_cascade";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "Burgers 湍流化：能谱、能流与激波";

const ENERGY_FLOOR_FALLBACK = 1e-18;
const U_CLIP_FALLBACK = 1.6;
const NU_CRITICAL_FALLBACK = 0.015;
const T_MAX_FALLBACK = 1.5;

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

export const burgersCascadeModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需数组 ——
  const k = requireArray(frame, "k");
  const eNow = requireArray(frame, "energy_now");
  const eIni = requireArray(frame, "energy_initial");
  const eRef = requireArray(frame, "energy_ref_slope");
  const flux = requireArray(frame, "flux");
  const x = requireArray(frame, "x");
  const uNow = requireArray(frame, "u_now");
  const uIni = requireArray(frame, "u_initial");
  const energyT = requireArray(frame, "energy_t");
  const energyErr = requireArray(frame, "energy_error");
  const budgetT = requireArray(frame, "budget_t");
  const budgetRes = requireArray(frame, "budget_residual");
  for (const [name, arr] of [
    ["energy_now", eNow],
    ["energy_initial", eIni],
    ["energy_ref_slope", eRef],
    ["flux", flux],
  ] as const) {
    requireSameLength(k, arr, `帧数组 k 与 ${name}`);
  }
  requireSameLength(x, uNow, "帧数组 x 与 u_now");
  requireSameLength(x, uIni, "帧数组 x 与 u_initial");
  requireSameLength(energyT, energyErr, "帧数组 energy_t 与 energy_error");
  requireSameLength(budgetT, budgetRes, "帧数组 budget_t 与 budget_residual");
  const pitfalls = requirePitfalls(frame);

  // —— 可选掩码 / meta ——
  const degraded: string[] = [];
  const eFloorMask = frame.has("energy_floor_mask") ? requireArray(frame, "energy_floor_mask") : undefined;
  if (eFloorMask === undefined) degraded.push("帧缺 energy_floor_mask：不标「能谱该模低于地板」");
  const eErrMask = frame.has("energy_error_mask") ? requireArray(frame, "energy_error_mask") : undefined;
  if (eErrMask === undefined) degraded.push("帧缺 energy_error_mask：不标能量漂移的地板点（=完美闭合）");
  const bResMask = frame.has("budget_residual_mask") ? requireArray(frame, "budget_residual_mask") : undefined;
  if (bResMask === undefined) degraded.push("帧缺 budget_residual_mask：不标收支残差的地板点");
  const uClipMask = frame.has("u_clipped") ? requireArray(frame, "u_clipped") : undefined;
  if (uClipMask === undefined) degraded.push("帧缺 u_clipped：不标被显示上限截断的网格点（计数仍在 meta.n_clipped）");

  const floorMeta = metaNumber(frame, "energy_floor");
  if (floorMeta === undefined) degraded.push("帧 meta 缺 energy_floor：地板用登记常量 1e-18");
  const floor = floorMeta !== undefined && floorMeta > 0 ? floorMeta : ENERGY_FLOOR_FALLBACK;
  const uClipMeta = metaNumber(frame, "u_clip");
  const uClip = uClipMeta !== undefined && uClipMeta > 0 ? uClipMeta : U_CLIP_FALLBACK;
  if (uClipMeta === undefined) degraded.push("帧 meta 缺 u_clip：解曲线窗口用登记常量 ±1.6");

  const n = metaNumber(frame, "n");
  const nu = metaNumber(frame, "nu");
  const dealiasRaw = frame.meta["dealias"];
  const dealias = dealiasRaw === true || (metaNumber(frame, "dealias") ?? 0) > 0.5;
  if (dealiasRaw === undefined) degraded.push("帧 meta 缺 dealias：按未去混叠显示（不猜开关）");
  const tActual = metaNumber(frame, "t_actual");
  const tMax = SPEC !== undefined && SPEC.time.mode === "scrub" ? SPEC.time.tMax ?? T_MAX_FALLBACK : T_MAX_FALLBACK;
  const t = tActual !== undefined ? Math.max(0, Math.min(tActual, tMax)) : Math.max(0, Math.min(opts.t, tMax));
  const dt = metaNumber(frame, "dt");
  const nSteps = metaNumber(frame, "n_steps");
  const kDiss = metaNumber(frame, "k_diss");
  const fitLo = metaNumber(frame, "fit_lo");
  const fitHi = metaNumber(frame, "fit_hi");
  const slopeFit = metaNumber(frame, "slope_fit");
  const slopeR2 = metaNumber(frame, "slope_r2");
  const slopeTheory = metaNumber(frame, "slope_theory");
  const energy0 = metaNumber(frame, "energy_0");
  const energyNow = metaNumber(frame, "energy_now");
  const energyDrift = metaNumber(frame, "energy_drift");
  const energyErrMax = metaNumber(frame, "energy_error_max");
  const enstrophy = metaNumber(frame, "enstrophy");
  const dissipation = metaNumber(frame, "dissipation");
  const fluxMax = metaNumber(frame, "flux_max");
  const fluxEnd = metaNumber(frame, "flux_at_end");
  const fluxOverEps = metaNumber(frame, "flux_over_eps");
  const budgetErrMax = metaNumber(frame, "budget_error_max");
  const maxU = metaNumber(frame, "max_u");
  const nClipped = metaNumber(frame, "n_clipped");
  const initial = stringOf(frame, "initial");
  const floorText = stringOf(frame, "floor");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });
  const L = columnsLayout(scene, width, height, { topShare: 0.56, auxH: 120, gap: 50 });
  const panelTextFits = L.floor > L.botY + L.botH + 8;

  // 右列：解曲线（上）+ 误差/残差副轴（下）。副轴底线同样钉在 L.floor。
  const solH = Math.max(70, Math.min(120, Math.round(Math.max(0, L.floor - L.topY) * 0.36)));
  const solY = L.topY;
  // 右列中缝放 4–5 行读数：解轴 xlabel 止于 solY+solH+29，残差轴标题起于 resY-19。
  const resY = solY + solH + 130;
  const resH = Math.max(56, Math.min(120, L.floor - resY - 28));

  // —— 能谱：只保留 k>0（对数 x 轴）——
  const keep: number[] = [];
  for (let i = 0; i < k.length; i += 1) {
    const kv = k[i] as number;
    if (Number.isFinite(kv) && kv > 0) keep.push(i);
  }
  if (keep.length === 0) {
    throw new Error(`帧 ${CHAPTER}/${VIZ} 的 k 没有正波数：双对数能谱无法成立`);
  }
  const kx = selectValues(k, keep);
  const nowY = selectValues(eNow, keep);
  const iniY = selectValues(eIni, keep);
  const refY = selectValues(eRef, keep);
  const kLo = kx[0] as number;
  const kHi = kx[kx.length - 1] as number;
  const xHi = x.length > 0 ? (x[x.length - 1] as number) : 2 * Math.PI;

  const spectrum = makeAxis("spectrum", [L.left, L.topY, L.leftW, L.topH], {
    xlim: [kLo, Math.max(kLo * 1.2, kHi)],
    ylim: [floor, 1],
    xscale: "log",
    yscale: "log",
    xlabel: "波数 k（对数）",
    title: axisTitleOf(SPEC?.axes, "spectrum", "能谱 E(k)（双对数，1e-18–1）"),
    ylabel: axisYlabelOf(SPEC?.axes, "spectrum", "E(k)"),
    autoscaled: false,
  });
  const fluxAxis = makeAxis("flux", [L.left, L.botY, L.leftW, L.botH], {
    xlim: [0, Math.max(kHi, 1)],
    ylim: [-0.3, 0.3],
    xlabel: "波数 k",
    title: axisTitleOf(SPEC?.axes, "flux", "能流 Pi(k)（窗口 ±0.3）"),
    ylabel: axisYlabelOf(SPEC?.axes, "flux", "Pi(k)"),
    autoscaled: false,
  });
  const solution = makeAxis("solution", [L.rightX, solY, L.rightW, solH], {
    xlim: [0, Math.max(xHi, 2 * Math.PI)],
    ylim: [-uClip, uClip],
    xlabel: "x",
    title: axisTitleOf(SPEC?.axes, "solution", "解 u(x,t)（窗口 ±1.6）"),
    ylabel: axisYlabelOf(SPEC?.axes, "solution", "u"),
    autoscaled: false,
  });
  const residual = makeAxis("residual", [L.rightX, resY, L.rightW, resH], {
    xlim: [0, tMax],
    ylim: [floor, 10],
    yscale: "log",
    xlabel: "t",
    title: "漂移与收支残差（对数，1e-18–10）",
    ylabel: "相对量",
    autoscaled: false,
  });
  scene.axes.push(spectrum, fluxAxis, solution, residual);

  // —— 能谱图层：地板线 → 初值 → k^-2 → 当前 → 地板散点；拟合窗口竖线 ——
  scene.layers.push(
    lineLayer("spectrum", series([spectrum.xlim[0], spectrum.xlim[1]], [floor, floor], pal.warn, { dash: [3, 3], width: 1, label: "能谱地板" })),
    lineLayer("spectrum", series(kx, iniY, pal.reference, { dash: [5, 4], width: 1.3, label: "初始能谱" })),
    lineLayer("spectrum", series(kx, refY, pal.exact, { dash: [6, 3], width: 1.4, label: "参考能谱" })),
    lineLayer("spectrum", series(kx, nowY, pal.curves[0], { width: 2.2, label: "当前能谱" })),
  );
  if (fitLo !== undefined && fitHi !== undefined && fitHi > fitLo) {
    for (const kv of [fitLo, fitHi]) {
      if (kv >= spectrum.xlim[0] && kv <= spectrum.xlim[1]) {
        scene.layers.push(
          lineLayer("spectrum", series([kv, kv], [floor, 1], pal.reference, { dash: [2, 4], width: 1 })),
        );
      }
    }
  }
  if (eFloorMask !== undefined && eFloorMask.length === k.length) {
    const idx: number[] = [];
    for (const i of keep) if ((eFloorMask[i] ?? 0) > 0.5) idx.push(i);
    if (idx.length > 0) {
      scene.layers.push(
        scatterLayer("spectrum", selectValues(k, idx), selectValues(eNow, idx), {
          r: 2.4,
          color: pal.warn,
          label: "能谱低于地板",
        }),
      );
    }
  }

  // —— 能流：Pi=0、eps、k_d、Pi_max 标记 ——
  let fluxPeak = -1;
  let fluxPeakVal = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < flux.length; i += 1) {
    const v = flux[i] as number;
    if (Number.isFinite(v) && v > fluxPeakVal) {
      fluxPeakVal = v;
      fluxPeak = i;
    }
  }
  scene.layers.push(
    lineLayer("flux", series([fluxAxis.xlim[0], fluxAxis.xlim[1]], [0, 0], pal.reference, { dash: [4, 4], width: 1, label: "零通量" })),
    lineLayer("flux", series(k, flux, pal.curves[0], { width: 2.2, label: "能流 Π(k)" })),
  );
  if (dissipation !== undefined && dissipation > 0 && dissipation <= fluxAxis.ylim[1]) {
    scene.layers.push(
      lineLayer(
        "flux",
        series([fluxAxis.xlim[0], fluxAxis.xlim[1]], [dissipation, dissipation], pal.exact, { dash: [6, 3], width: 1.2, label: "耗散" }),
      ),
    );
  }
  if (kDiss !== undefined && kDiss > 0 && kDiss <= fluxAxis.xlim[1]) {
    scene.layers.push(
      lineLayer("flux", series([kDiss, kDiss], [fluxAxis.ylim[0], fluxAxis.ylim[1]], pal.reference, { dash: [1, 3], width: 1, label: "耗散波数" })),
    );
  }
  if (fluxPeak >= 0) {
    scene.layers.push(
      scatterLayer("flux", [k[fluxPeak] as number], [flux[fluxPeak] as number], {
        r: 3.2,
        color: pal.marker,
      }),
    );
  }

  // —— 解曲线：初值（灰虚）、当前（蓝）、截断点（红点）+ 激波示踪 ——
  scene.layers.push(
    lineLayer("solution", series(x, uIni, pal.reference, { dash: [5, 4], width: 1.2, label: "初始解" })),
    lineLayer("solution", series(x, uNow, pal.curves[0], { width: 2.2, label: "当前解" })),
  );
  if (uClipMask !== undefined && uClipMask.length === x.length) {
    const idx: number[] = [];
    for (let i = 0; i < uClipMask.length; i += 1) if ((uClipMask[i] ?? 0) > 0.5) idx.push(i);
    if (idx.length > 0) {
      scene.layers.push(
        scatterLayer("solution", selectValues(x, idx), selectValues(uNow, idx), {
          r: 2.8,
          color: pal.warn,
          label: "超过 ±u_clip 被截断显示",
        }),
      );
    }
  }
  // 激波示踪：相邻最大落差。**只用于标注位置**，不改写任何物理量；
  // 阈值 0.2·max|u| 是显示判据（平滑正弦的落差约 0.05，成形激波约 0.8）。
  let shockIdx = -1;
  let maxDrop = 0;
  for (let i = 0; i + 1 < uNow.length; i += 1) {
    const a = uNow[i] as number;
    const b = uNow[i + 1] as number;
    const d = a - b;
    if (Number.isFinite(d) && d > maxDrop) {
      maxDrop = d;
      shockIdx = i;
    }
  }
  let uAbsMax = 0;
  for (let i = 0; i < uNow.length; i += 1) {
    const v = Math.abs(uNow[i] as number);
    if (Number.isFinite(v) && v > uAbsMax) uAbsMax = v;
  }
  // 初值的最大相邻落差：只有**相对初值显著变陡**才叫「激波成形」。
  // 随机初值本身就有 0.29 的落差（t=0），只用绝对阈值会把它误标成激波。
  let initialDrop = 0;
  for (let i = 0; i + 1 < uIni.length; i += 1) {
    const d = (uIni[i] as number) - (uIni[i + 1] as number);
    if (Number.isFinite(d) && d > initialDrop) initialDrop = d;
  }
  const uScale = maxU !== undefined && maxU > 0 ? maxU : uAbsMax;
  const shockShown =
    shockIdx >= 0 &&
    t > 0 &&
    maxDrop > 0.2 * Math.max(uScale, 1e-12) &&
    maxDrop > 1.5 * Math.max(initialDrop, 1e-12);
  const shockX = shockShown ? (x[shockIdx] as number) : undefined;
  if (shockX !== undefined && Number.isFinite(shockX)) {
    scene.layers.push(
      lineLayer("solution", series([shockX, shockX], [-uClip, uClip], pal.marker, { dash: [3, 3], width: 1.2 })),
      scatterLayer("solution", [shockX], [uNow[shockIdx + 1] as number], { r: 3.4, color: pal.marker, label: "激波示踪" }),
    );
  }

  // —— 误差/残差副轴（规则 3）：两条真实曲线 + 地板线 ——
  scene.layers.push(
    lineLayer("residual", series([residual.xlim[0], residual.xlim[1]], [floor, floor], pal.warn, { dash: [3, 3], width: 1, label: "残差地板" })),
  );
  if (energyErr.length >= 2) {
    scene.layers.push(lineLayer("residual", series(energyT, energyErr, pal.curves[0], { width: 1.8, label: "能量收支误差" })));
  } else if (energyErr.length === 1) {
    scene.layers.push(scatterLayer("residual", energyT, energyErr, { r: 3, color: pal.curves[0] }));
  }
  if (budgetRes.length >= 2) {
    scene.layers.push(lineLayer("residual", series(budgetT, budgetRes, pal.residual, { width: 1.8, label: "预算残差" })));
  } else if (budgetRes.length === 1) {
    scene.layers.push(scatterLayer("residual", budgetT, budgetRes, { r: 3, color: pal.residual }));
  }
  if (eErrMask !== undefined && eErrMask.length === energyErr.length) {
    const idx: number[] = [];
    for (let i = 0; i < eErrMask.length; i += 1) if ((eErrMask[i] ?? 0) > 0.5) idx.push(i);
    if (idx.length > 0) {
      scene.layers.push(
        scatterLayer("residual", selectValues(energyT, idx), selectValues(energyErr, idx), {
          r: 2.2,
          color: pal.warn,
          alpha: 0.8,
        }),
      );
    }
  }

  // —— 面板内图例/数值（单块 textBlock，窄画布整块放弃）——
  if (panelTextFits) {
    const spLines: string[] = [];
    spLines.push(`初值=灰虚　当前=蓝　k^-2=橙虚　地板红线=${floorText ?? formatNumber(floor, 3)}`);
    if (slopeFit !== undefined && fitLo !== undefined && fitHi !== undefined) {
      spLines.push(
        `拟合窗口 k∈[${formatNumber(fitLo, 3)}, ${formatNumber(fitHi, 3)}]：斜率 ${formatNumber(slopeFit, 3)}` +
          `（理论 ${formatNumber(slopeTheory ?? -2, 2)}` +
          (slopeR2 !== undefined ? `，R²=${formatNumber(slopeR2, 5)}` : "") +
          "）",
      );
    } else {
      degraded.push("帧 meta 缺 slope_fit/fit_lo/fit_hi：不写能谱拟合斜率（不猜）");
    }
    spLines.push(`N=${n !== undefined ? Math.round(n) : "?"}　nu=${nu !== undefined ? formatNumber(nu, 3) : "?"}　${dealias ? "2/3+3/2" : "朴素乘积"}　初值 ${initial ?? "?"}`);
    scene.texts.push(...textBlock(L.left + 6, L.topY + L.topH - SAFE_GAP - 8 * TEXT_ROW_H, L.topY + L.topH - SAFE_GAP - TEXT_ROW_H, spLines, pal.muted, unitBudget(L.leftW), "bottom"));

    const fxLines: string[] = [];
    fxLines.push("蓝=Pi(k)　灰虚=0　橙虚=耗散 eps　点线=k_d");
    if (fluxMax !== undefined && dissipation !== undefined) {
      fxLines.push(
        `Pi_max=${formatNumber(fluxMax, 3)}　eps=${formatNumber(dissipation, 3)}` +
          (fluxOverEps !== undefined ? `　Pi/eps=${formatNumber(fluxOverEps, 3)}` : "") +
          (dissipation > fluxAxis.ylim[1] ? "；eps 超出窗口（未画）" : ""),
      );
    }
    if (fluxEnd !== undefined) {
      fxLines.push(
        `末端 ${formatNumber(fluxEnd, 3)}` +
          (fluxEnd < fluxAxis.ylim[0] ? "：低于窗口（欠分辨，溢出点名）" : "：应回落到 0"),
      );
    }
    fxLines.push(`窗口 -0.3–0.3；欠分辨时通量不闭合是数值现象`);
    scene.texts.push(...textBlock(L.left + 6, L.botY + L.botH - SAFE_GAP - 8 * TEXT_ROW_H, L.botY + L.botH - SAFE_GAP - TEXT_ROW_H, fxLines, pal.muted, unitBudget(L.leftW), "bottom"));

    const solLines: string[] = [];
    solLines.push("蓝=当前解　灰虚=初值　红=激波示踪/截断");
    solLines.push(
      shockX !== undefined
        ? `激波 x=${formatNumber(shockX, 3)}，最大相邻落差 ${formatNumber(maxDrop, 3)}`
        : `激波未形成：最大相邻落差 ${formatNumber(maxDrop, 3)}（初值 ${formatNumber(initialDrop, 3)}）`,
    );
    solLines.push(
      (nClipped !== undefined && nClipped > 0
        ? `max|u|=${formatNumber(maxU ?? uAbsMax, 3)}，截断 ${Math.round(nClipped)} 点`
        : `max|u|=${formatNumber(maxU ?? uAbsMax, 3)}，未截断`) +
        (energy0 !== undefined && energyNow !== undefined
          ? `；E ${formatNumber(energy0, 3)}→${formatNumber(energyNow, 3)}`
          : ""),
    );
    if (enstrophy !== undefined || (dt !== undefined && nSteps !== undefined) || energyErrMax !== undefined) {
      solLines.push(
        (enstrophy !== undefined ? `Enstrophy ${formatNumber(enstrophy, 3)}` : "") +
          (dt !== undefined && nSteps !== undefined ? `；dt=${formatNumber(dt, 4)}，${Math.round(nSteps)} 步` : "") +
          (energyErrMax !== undefined ? `；漂移 ${formatNumber(energyErrMax, 3)}` : ""),
      );
    }
    scene.texts.push(
      ...textBlock(L.rightX + 4, solY + solH + 32, resY - 22, solLines, pal.muted, unitBudget(L.rightW), "top"),
    );
    scene.texts.push(
      ...textBlock(
        L.rightX + 4,
        resY + resH - SAFE_GAP - 6 * TEXT_ROW_H,
        resY + resH - SAFE_GAP - TEXT_ROW_H,
        [`蓝=能量漂移　紫=收支残差${budgetErrMax !== undefined ? `（max ${formatNumber(budgetErrMax, 3)}）` : ""}　红线=地板`],
        pal.muted,
        unitBudget(L.rightW),
        "bottom",
      ),
    );
  } else {
    degraded.push("画布过矮/过窄：面板说明与数值放不下（pitfalls 面板占满底部），关键读数见角标");
  }

  // —— 角标 ——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(
      badge("autoscale", "窗口固定：E 1e-18–1 · Π ±0.3 · u ±1.6 · 残差 1e-18–10", 8, 26),
      SLOT.aux,
      width,
    ),
  );
  if (nu !== undefined) {
    scene.badges.push(
      placeBadge(
        criticalBadge("nu", nu, criticalOf(frame, "nu", NU_CRITICAL_FALLBACK), { beyond: "below" }),
        SLOT.critical,
        width,
      ),
    );
  }
  const status: string[] = [];
  if (slopeFit !== undefined) status.push(`斜率 ${formatNumber(slopeFit, 3)}/理论 -2`);
  if (fluxOverEps !== undefined) status.push(`Π/ε=${formatNumber(fluxOverEps, 3)}`);
  if (energyDrift !== undefined) status.push(`E 漂移 ${formatNumber(energyDrift, 3)}`);
  if (nClipped !== undefined && nClipped > 0) status.push(`${Math.round(nClipped)} 点截断`);
  scene.badges.push(placeBadge(badge("status", status.join("；")), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  const nBad =
    countNonFinite(eNow) +
    countNonFinite(eIni) +
    countNonFinite(eRef) +
    countNonFinite(flux) +
    countNonFinite(uNow) +
    countNonFinite(uIni) +
    countNonFinite(energyErr) +
    countNonFinite(budgetRes);
  applyOverflow(scene, nBad, { slot: SLOT.overflow, width });
  return scene;
};
