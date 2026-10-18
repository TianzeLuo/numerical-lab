/**
 * `convergence_rates`（2.3 章）的 model：三种光滑性的 L² 投影误差 + 每条曲线的实测/理论斜率。
 *
 * 面板与图层（顺序即绘制顺序，测试按此断言）
 * ------------------------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | line | `main` | 解析函数（Bernstein 椭圆有限）误差 `error_analytic`（实线） |
 * | 2 | line | `main` | 尖点 `error_cusp`（实线，另一色） |
 * | 3 | line | `main` | 跳点 `error_jump`（实线，第三色） |
 * | 4 | line | `main` | 三条**理论**参考曲线 `theory_*`（同色虚线，锚在窗口首点） |
 * | 5 | line | `main` | （仅在有点触地板时）一条 `u` 地板横线 |
 * | 6 | scatter | `main` | 三个**拟合窗口起点**标记 |
 * | 7 | loglogfit | `rate` | 尖点：窗口内的实测拟合线 + 理论斜率直线（规则 2 的强制形态） |
 * | 8 | loglogfit | `rate` | 跳点：同上 |
 * | 9 | line | `rate` | 解析情形：窗口内的 `fit_analytic`（实线）与 `theory_analytic`（虚线） |
 *
 * 角标槽位（固定，见 `common.ts::SLOT`）：0 来源 / 1 主轴固定 / 2 副轴自适应 /
 * 3 状态（无参数临界值，如实说明）/ 4 地板与截断 / 5 前端预览 / 6 溢出 / 7 降级。
 *
 * 这个 viz 最容易骗人的两处，以及它们在图上的防法
 * ------------------------------------------------
 * 1. **把理论参考线当成拟合结果**。理论线只固定**斜率**，常数项是锚在拟合窗口第一个实测点上的：
 *    两条线重合说明斜率对，不说明常数对、更不说明误差真的按这个率下降。防法：`rate` 面板把
 *    实测拟合线与理论参考线画成实/虚一对，并把 measured/theory/相对差/R²/窗口起点四个数写在
 *    同一块文字里；`strings.fit_note` 原文照登。s=200（ρ=1.0732）时实测与理论差 2.4%——
 *    这不是理论错，而是 n≤40 还没进渐近区，"看起来是直线"不足以支撑结论（pitfalls §12）。
 * 2. **在半对数收敛上画双对数理论直线**。解析情形的误差按 ρ^(−n) 衰减，双对数下**不是直线**；
 *    若照搬 `loglogfit` 的直线形态会画出一条完全错误的理论线。防法：解析情形用后端给的
 *    `theory_analytic` 曲线（正确的函数形态），只有两条代数（幂律）曲线用 `loglogfit` 层画
 *    理论斜率直线——并在 `rate` 面板文字里写明这个区分。
 *
 * 几条刻意的决定
 * --------------
 * 1. **两条线都画，来源各自写明**：`fit_*`（实测拟合，锚在窗口首点）与 `theory_*`（理论，
 *    同样锚在窗口首点）在 `main` 面板成对出现；`rate` 面板是它们的窗口放大版。
 * 2. **窗口是"人选的"，所以画出来**：三个拟合窗口起点（n=2/3/8）用散点标在 `main` 上，
 *    并在文字里给出具体数字——"实测斜率"这个数离开了窗口就没有意义（pitfalls §6）。
 * 3. **地板只在真的有东西掉下去时才画横线**（`meta.clipped_*` > 0）：s=25 时三条曲线都没触底，
 *    画一条 1e-16 的横线会把纵轴硬拉下 12 个数量级，反而把三条曲线压成平的。触底与否、
 *    触底几个点，文字里始终写。
 * 4. **没有时间轴**（`time.mode="none"`）：不引用 `tMax`、不画游标，`scene.t` 取帧 `meta.t`。
 */

import { niceLimits, niceTicks } from "../../shared/src/axes.js";
import { countNonFinite } from "../../shared/src/geometry.js";
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
  segmentedSeries,
  stackedPanels,
  stringOf,
  type SceneModel,
} from "../../shared/src/model.js";
import {
  axisModeBadge,
  badge,
  lineLayer,
  loglogFitLayer,
  makeAxis,
  previewBadge,
  scatterLayer,
  sourceBadge,
} from "../../shared/src/scene.js";
import {
  CHAPTER,
  SLOT,
  axisTitle,
  axisYlabel,
  countAtOrBelow,
  fmt,
  fmtOpt,
  minMaxOfAll,
  oneTwoFive,
  noteOf,
  sci,
  specTitle,
  textBlock,
  unitBudget,
} from "./common.js";

const VIZ = "convergence_rates";
const TITLE = specTitle(VIZ, "误差衰减率：实测斜率与理论斜率");

/** 三个 regime 的固定顺序（颜色、meta 键前缀、数组名一一对应）。 */
const REGIMES = ["analytic", "cusp", "jump"] as const;
type Regime = (typeof REGIMES)[number];
/** 后端存的是"正的衰减指数"或半对数斜率；图上习惯写负号：取负只是显示约定（缺失仍显示 `?`）。 */
const neg = (v: number | undefined): number | undefined => (v === undefined ? undefined : -v);

export const convergenceRatesModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需数组 ——
  const n = requireArray(frame, "n");
  const errs: Record<Regime, ArrayLike<number>> = {
    analytic: requireArray(frame, "error_analytic"),
    cusp: requireArray(frame, "error_cusp"),
    jump: requireArray(frame, "error_jump"),
  };
  const theory: Record<Regime, ArrayLike<number>> = {
    analytic: requireArray(frame, "theory_analytic"),
    cusp: requireArray(frame, "theory_cusp"),
    jump: requireArray(frame, "theory_jump"),
  };
  const fit: Record<Regime, ArrayLike<number>> = {
    analytic: requireArray(frame, "fit_analytic"),
    cusp: requireArray(frame, "fit_cusp"),
    jump: requireArray(frame, "fit_jump"),
  };
  for (const r of REGIMES) {
    requireSameLength(n, errs[r], `帧数组 n 与 error_${r}`);
    requireSameLength(n, theory[r], `帧数组 n 与 theory_${r}`);
    requireSameLength(n, fit[r], `帧数组 n 与 fit_${r}`);
  }
  const pitfalls = requirePitfalls(frame);

  // —— 可选/降级字段 ——
  const degraded: string[] = [];
  const slope: Record<Regime, { measured?: number; theory?: number; r2?: number; rel?: number; start?: number }> = {
    analytic: {},
    cusp: {},
    jump: {},
  };
  const META_KEYS: Record<Regime, { m: string; t: string; r2: string; rel: string; start: string }> = {
    analytic: {
      m: "rate_analytic_measured",
      t: "rate_analytic_theory",
      r2: "rate_analytic_r2",
      rel: "rate_analytic_rel_error",
      start: "fit_window_analytic_start",
    },
    cusp: {
      m: "alpha_cusp_measured",
      t: "alpha_cusp_theory",
      r2: "alpha_cusp_r2",
      rel: "alpha_cusp_rel_error",
      start: "fit_window_cusp_start",
    },
    jump: {
      m: "alpha_jump_measured",
      t: "alpha_jump_theory",
      r2: "alpha_jump_r2",
      rel: "alpha_jump_rel_error",
      start: "fit_window_jump_start",
    },
  };
  for (const r of REGIMES) {
    const keys = META_KEYS[r];
    slope[r] = {
      measured: metaNumber(frame, keys.m),
      theory: metaNumber(frame, keys.t),
      r2: metaNumber(frame, keys.r2),
      rel: metaNumber(frame, keys.rel),
      start: metaNumber(frame, keys.start),
    };
    if (slope[r].measured === undefined) degraded.push(`帧 meta 缺 ${keys.m}：${r} 的实测斜率留「?」，不猜`);
    if (slope[r].start === undefined) degraded.push(`帧 meta 缺 ${keys.start}：${r} 的拟合窗口按全部 n 处理并已点明`);
  }
  const rho = metaNumber(frame, "rho");
  const errorFloor = metaNumber(frame, "error_floor");
  if (rho === undefined) degraded.push("帧 meta 缺 rho：解析情形的几何率 ρ 留「?」（不按 s 反推）");
  if (errorFloor === undefined) degraded.push("帧 meta 缺 error_floor：地板位置留「?」");
  const clipped: Record<Regime, number> = {
    analytic: metaNumber(frame, "clipped_analytic") ?? 0,
    cusp: metaNumber(frame, "clipped_cusp") ?? 0,
    jump: metaNumber(frame, "clipped_jump") ?? 0,
  };
  if (metaNumber(frame, "clipped_analytic") === undefined && metaNumber(frame, "clipped_cusp") === undefined) {
    degraded.push("帧 meta 缺 clipped_*：不声称曲线是否触底（图上不画地板横线）");
  }
  const clippedTotal = clipped.analytic + clipped.cusp + clipped.jump;
  /** log 轴画不到 0/负数：这类点会被 `accept` 跳过，跳过的个数必须写出来（规则 4）。 */
  const nonPositive = REGIMES.reduce((acc, r) => acc + countAtOrBelow(errs[r], 0) + countAtOrBelow(theory[r], 0), 0);
  const analyticNote = stringOf(frame, "analytic");
  const regimeNote: Record<Regime, string | undefined> = {
    analytic: noteOf(frame, "regime_analytic"),
    cusp: noteOf(frame, "regime_cusp"),
    jump: noteOf(frame, "regime_jump"),
  };
  const floorNote = noteOf(frame, "floor_note");
  const fitNote = noteOf(frame, "fit_note");
  if (fitNote === undefined) degraded.push("帧 strings 缺 fit_note：锚点约定（锚在窗口首点）按模型说明显示");

  // —— 面板 ——
  const base = stackedPanels(width, height, { mainShare: 0.48, gap: 50 });

  // —— 场景外壳（无时间轴：scene.t 取帧 meta.t）——
  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: metaNumber(frame, "t") ?? opts.t,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  // —— 坐标范围：横轴 n 用对数（图题写明"双对数"），纵轴误差用对数 ——
  const nRange = minMaxOfAll([n]) ?? [2, 40];
  const xlim: [number, number] = [Math.max(1, nRange[0] * 0.9), nRange[1] * 1.12];
  /** 横轴刻度：帧 `n` 里的 1-2-5 值（[2,5,10,20,40]），比 niceTicks 的 [5,10] 可读。 */
  const nTicks = oneTwoFive(n);
  const allMain = REGIMES.flatMap((r) => [errs[r], theory[r]]);
  const yMain = minMaxOfAll(allMain) ?? [1e-3, 1];
  const ylimMain = niceLimits(yMain[0] > 0 ? yMain[0] : 1e-16, yMain[1], 5, { log: true, pad: 0.02 });

  scene.axes.push(
    makeAxis("main", base.main, {
      xlim,
      ylim: ylimMain,
      xscale: "log",
      yscale: "log",
      xlabel: "维数 n（对数轴）",
      ylabel: axisYlabel(VIZ, "main", "‖f − P_n f‖"),
      title: axisTitle(VIZ, "main", "L² 误差 vs 维数（双对数）"),
      autoscaled: false,
      xticks: nTicks.length > 1 ? nTicks : niceTicks(xlim[0], xlim[1], 7, { log: true }),
      yticks: niceTicks(ylimMain[0], ylimMain[1], 6, { log: true }),
    }),
  );
  for (const r of REGIMES) {
    const color = pal.curves[REGIMES.indexOf(r)] as string;
    scene.layers.push(lineLayer("main", segmentedSeries(n, errs[r], color, { width: 1.9, label: `${r}：实测 L² 误差`, accept: (v) => v > 0 })));
  }
  for (const r of REGIMES) {
    const color = pal.curves[REGIMES.indexOf(r)] as string;
    scene.layers.push(
      lineLayer("main", segmentedSeries(n, theory[r], color, { width: 1.2, dash: [6, 4], label: `${r}：理论参考线（锚在窗口首点）`, accept: (v) => v > 0 })),
    );
  }
  if (clippedTotal > 0 && errorFloor !== undefined) {
    scene.layers.push(
      lineLayer(
        "main",
        segmentedSeries([xlim[0], xlim[1]], [errorFloor, errorFloor], pal.warn, { width: 1.2, dash: [2, 3], label: `舍入地板 u = ${sci(errorFloor)}` }),
      ),
    );
  }
  // 拟合窗口起点：三个横坐标（n=2/3/8），纵坐标取该点的实测误差（帧数组，不插值猜测）。
  const winX: number[] = [];
  const winY: number[] = [];
  for (const r of REGIMES) {
    const start = slope[r].start;
    if (start === undefined) continue;
    for (let i = 0; i < n.length; i += 1) {
      if ((n[i] as number) >= start) {
        winX.push(n[i] as number);
        winY.push(errs[r][i] as number);
        break;
      }
    }
  }
  if (winX.length > 0) {
    scene.layers.push(scatterLayer("main", winX, winY, { r: 4.2, color: pal.marker, label: "拟合窗口起点" }));
  }

  // —— rate 面板：拟合窗口内的实测拟合线 vs 理论参考线 ——
  const slices: Record<Regime, number[]> = { analytic: [], cusp: [], jump: [] };
  for (const r of REGIMES) {
    const start = slope[r].start ?? nRange[0];
    for (let i = 0; i < n.length; i += 1) if ((n[i] as number) >= start) slices[r].push(i);
  }
  const winN = (r: Regime): Float64Array => Float64Array.from(slices[r].map((i) => n[i] as number));
  const winOf = (a: ArrayLike<number>, r: Regime): Float64Array => Float64Array.from(slices[r].map((i) => a[i] as number));
  const allRate = REGIMES.flatMap((r) => [winOf(fit[r], r), winOf(theory[r], r)]);
  const yRate = minMaxOfAll(allRate) ?? [1e-3, 1];
  const ylimRate = niceLimits(yRate[0] > 0 ? yRate[0] : 1e-16, yRate[1], 5, { log: true, pad: 0.02 });
  // 横轴左端取**最早**的那个窗口起点（用 max 会把晚开窗的 regime 的实测点挤出轴外：
  // 跳点从 n=2 起、尖点从 n=8 起，取 max 会让跳点的前几个点落在轴外 → 规则 1 的溢出）。
  const rateX0 = Math.min(...REGIMES.map((r) => (slices[r].length > 0 ? (winN(r)[0] as number) : nRange[1])));
  const xlimRate: [number, number] = [Math.max(1, rateX0 * 0.88), nRange[1] * 1.12];

  scene.axes.push(
    makeAxis("rate", base.error, {
      xlim: xlimRate,
      ylim: ylimRate,
      xscale: "log",
      yscale: "log",
      xlabel: "维数 n（拟合窗口内，对数轴）",
      ylabel: axisYlabel(VIZ, "rate", "误差（对数轴）"),
      title: axisTitle(VIZ, "rate", "实测/理论参考线（拟合窗口内）"),
      autoscaled: true,
      xticks: nTicks.length > 1 ? nTicks : niceTicks(xlimRate[0], xlimRate[1], 6, { log: true }),
      yticks: niceTicks(ylimRate[0], ylimRate[1], 5, { log: true }),
    }),
  );
  // 两条代数（幂律）曲线：双对数下理论线是直线 → 用 loglogfit 层（规则 2 的强制形态）。
  // 解析情形不能用它：ρ^(−n) 在双对数下不是直线，硬画一条直线是错的。
  for (const r of ["cusp", "jump"] as const) {
    const m = slope[r].measured;
    const t = slope[r].theory;
    if (m === undefined || t === undefined || slices[r].length < 2) {
      degraded.push(`${r}：缺实测/理论斜率或窗口内点不足，rate 面板改用 ${r} 的拟合/理论曲线（不画直线参考）`);
      scene.layers.push(lineLayer("rate", segmentedSeries(winN(r), winOf(fit[r], r), pal.curves[REGIMES.indexOf(r)], { width: 1.6 })));
      scene.layers.push(
        lineLayer("rate", segmentedSeries(winN(r), winOf(theory[r], r), pal.reference, { width: 1.2, dash: [6, 4] })),
      );
      continue;
    }
    scene.layers.push(
      loglogFitLayer("rate", winN(r), winOf(fit[r], r), -Math.abs(m), -Math.abs(t), {
        // 光栅渲染器会把这两个标签画在轴的右上角（`.cache` 之外的 `tools/raster.py`），
        // 标签太长会互相压住、还会顶出轴外——所以这里只写"是谁 + 斜率"，不写整句。
        seriesLabel: `${r} 实测 ${fmt(-Math.abs(m), 4)}`,
        theoryLabel: `理论 ${fmt(-Math.abs(t), 4)}`,
        color: pal.curves[REGIMES.indexOf(r)],
        theoryColor: pal.reference,
      }),
    );
  }
  // 解析情形：后端给的曲线（几何收敛在双对数下是弯的）。
  if (slices.analytic.length >= 2) {
    scene.layers.push(
      lineLayer("rate", segmentedSeries(winN("analytic"), winOf(fit.analytic, "analytic"), pal.curves[0], {
        width: 1.6,
        accept: (v: number) => v > 0,
        label: `analytic 实测拟合线（半对数斜率 ${slope.analytic.measured === undefined ? "?" : fmt(slope.analytic.measured, 4)}）`,
      })),
    );
    scene.layers.push(
      lineLayer("rate", segmentedSeries(winN("analytic"), winOf(theory.analytic, "analytic"), pal.reference, {
        width: 1.2,
        dash: [6, 4],
        accept: (v: number) => v > 0,
        label: "analytic 理论曲线 ρ^(−n)（双对数下不是直线）",
      })),
    );
  }
  // 窗口起点锚点（拟合线与理论线都锚在这里）。
  const anchorX: number[] = [];
  const anchorY: number[] = [];
  for (const r of REGIMES) {
    if (slices[r].length === 0) continue;
    anchorX.push(winN(r)[0] as number);
    anchorY.push(winOf(fit[r], r)[0] as number);
  }
  if (anchorX.length > 0) {
    scene.layers.push(scatterLayer("rate", anchorX, anchorY, { r: 4, color: pal.marker, label: "锚点（窗口首个实测点）" }));
  }

  // —— 文本 ——
  // 两条曲线都是从左上往右下衰减，面板上半部正好被数据占着；文字块统一贴在**底边**
  // （`anchor="bottom"`），只压住衰减曲线已经走过去的那一角。
  const mainLines: string[] = [
    `实线 = 实测 L² 误差（解析 / 尖点 / 跳点三族）　虚线 = 理论参考线（锚在拟合窗口的首个实测点）`,
    `解析 f = ${analyticNote ?? "?"}（ρ = ${rho === undefined ? "?" : fmt(rho, 7)}）：${regimeNote.analytic ?? "帧未给解析情形的说明"}。` +
      `实测 ${fmtOpt(slope.analytic.measured, 6)} vs 理论 ${fmtOpt(neg(slope.analytic.theory), 6)}` +
      `（相对差 ${slope.analytic.rel === undefined ? "?" : `${fmt(slope.analytic.rel * 100, 3)}%`}，R² = ${fmtOpt(slope.analytic.r2, 5)}）`,
    `尖点：${regimeNote.cusp ?? "帧未给说明"} → 实测 ${fmtOpt(neg(slope.cusp.measured), 6)} vs 理论 ${fmtOpt(neg(slope.cusp.theory), 3)}` +
      `（R² = ${fmtOpt(slope.cusp.r2, 5)}）　跳点：${regimeNote.jump ?? "帧未给说明"} → 实测 ${fmtOpt(neg(slope.jump.measured), 6)}` +
      ` vs 理论 ${fmtOpt(neg(slope.jump.theory), 3)}（R² = ${fmtOpt(slope.jump.r2, 5)}）`,
    `拟合窗口起点：解析 n ≥ ${fmtOpt(slope.analytic.start, 3)}、尖点 n ≥ ${fmtOpt(slope.cusp.start, 3)}、` +
      `跳点 n ≥ ${fmtOpt(slope.jump.start, 3)}（圆点标出；斜率离开窗口就没有意义）`,
    `地板 u = ${errorFloor === undefined ? "?" : sci(errorFloor)}；触底点数 解析/尖点/跳点 = ` +
      `${clipped.analytic}/${clipped.cusp}/${clipped.jump}${clippedTotal > 0 ? "（>0 才画地板横线）" : "（都没触底，故不画横线）"}`,
  ];
  if (nonPositive > 0) mainLines.push(`⚠ 有 ${nonPositive} 个误差值 ≤ 0：对数轴画不到，已跳过（**不是**当成 0 画在轴上）`);
  scene.texts.push(...textBlock(base.main[0] + 6, base.main[1] + 4, base.main[1] + base.main[3] - 4, mainLines, pal.note, unitBudget(base.main[2]), "bottom"));

  const rateLines: string[] = [
    `本面板只画拟合窗口内的点（横轴左端 = 最早的窗口起点）；实线 = 实测拟合线 fit_*，虚线 = 理论参考线。` +
      `幂律（尖点/跳点）的双对数理论线是直线 → 由 loglogfit 层画；解析按 ρ^(−n) 衰减，双对数下是弯的 → 用 theory_analytic 曲线`,
    `地板 u = ${errorFloor === undefined ? "?" : sci(errorFloor)}，被截断到地板的点数 ${clippedTotal}（meta.clipped_*）`,
  ];
  // 帧自带的说明优先照登（原样链）：它们的措辞是后端/笔记的原话，比我复述的可靠。
  if (fitNote !== undefined) rateLines.push(fitNote);
  if (floorNote !== undefined) rateLines.push(floorNote);
  scene.texts.push(...textBlock(base.error[0] + 6, base.error[1] + 4, base.error[1] + base.error[3] - 4, rateLines, pal.note, unitBudget(base.error[2]), "bottom"));

  // —— 角标 ——
  scene.badges.push(placeBadge(sourceBadge(scene.dataSource), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(badge("autoscale", "副轴 rate：自适应（拟合窗口内），会掩盖窗口外的量级变化", 8, 26), SLOT.auxAxis, width),
  );
  scene.badges.push(
    placeBadge(
      badge("critical", `无参数临界值（manifest critical=[]）；斜率是拟合量，不是阈值：相对差 ${slope.analytic.rel === undefined ? "?" : fmt(slope.analytic.rel * 100, 3)}%（解析）`),
      SLOT.critical,
      width,
    ),
  );
  scene.badges.push(
    placeBadge(
      badge(
        "status",
        clippedTotal > 0
          ? `⚠ 误差已触地板：共 ${clippedTotal} 点按 u 截断`
          : `未触地板（u=${errorFloor === undefined ? "?" : sci(errorFloor)}）：三条曲线全程在舍入噪声之上`,
      ),
      SLOT.status,
      width,
    ),
  );
  if (scene.dataSource.startsWith("backend+preview")) scene.badges.push(placeBadge(previewBadge(), SLOT.preview, width));

  let rawNonFinite = 0;
  for (const r of REGIMES) {
    rawNonFinite += countNonFinite(errs[r]) + countNonFinite(theory[r]) + countNonFinite(fit[r]);
  }
  applyOverflow(scene, rawNonFinite, { slot: SLOT.overflow, width });
  const degradedObj = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedObj) scene.badges.push(degradedObj);

  requireDrawableLayers(scene);
  return scene;
};
