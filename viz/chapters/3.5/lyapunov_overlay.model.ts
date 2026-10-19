/**
 * `lyapunov_overlay`（3.5 章）的 model：λ(r) 叠在分岔图上 + 分块标准误 + 三个解析对拍点。
 *
 * 面板与图层（顺序即绘制顺序，测试按此断言）
 * ------------------------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | scatter | `main` | 分岔图点云（淡化，用来对齐周期窗口） |
 * | 2 | band | `main` | λ 的分块标准误带（`sem_band_lo`/`sem_band_hi`） |
 * | 3 | line | `main` | λ=0 与 λ=ln2 两条参考线 |
 * | 4 | line | `main` | λ(r) 曲线 |
 * | 5 | line | `main` | 当前参数 r 的竖线 |
 * | 6 | scatter | `main` | 解析 λ（r=2.5/3.2/4）与同参数实测值 |
 * | 7 | line | `err` | 分块标准误本身（对数纵轴） |
 *
 * 三条刻意的决定
 * --------------
 * 1. **λ 轴固定 `[-6, 1]`**（manifest 声明 `autoscaled: false`）。理由不是省事：
 *    λ 有理论硬上界 ln2=0.693，而下界在周期窗口里可以很深（乘子趋近 ±1 时 λ→−∞ 的方向）；
 *    自适应纵轴会把"窗口里 λ 掉下去"这件事抹平，正好抹掉本图最想让人看见的那件事。
 *    自适应的是**副轴**（标准误），它按数据取整并显式标注。
 * 2. **"λ>0 所以混沌"必须被挡住**：λ 的有限时间估计在周期窗口边缘随迭代次数漂移
 *    （乘子趋近 ±1 时收敛是代数级的），所以副轴画标准误本身，文本里写明"它在哪里大，
 *    哪里的 λ 就不能当准数用"。标准误是**有限时间涨落**，不是舍入误差。
 * 3. **三个解析对拍点按"解析 vs 实测"成对出现**（r=2.5、r=3.2 的乘子有闭式，
 *    r=4 是 ln2）：图上两个标记（圈/叉）几乎重合，数字写进文本——对拍的意义在于
 *    把"估计器没写错"和"有限时间涨落有多大"分开。

 */

import { formatNumber, niceLimits } from "../../shared/src/axes.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import {
  applyOverflow,
  canvasSize,
  clipBadgeText,
  criticalOf,
  dataSourceOf,
  degradedBadge,
  finitePositiveMinMax,
  metaNumber,
  palette,
  paramSummary,
  placeBadge,
  requireArray,
  requireDrawableLayers,
  requirePitfalls,
  requireSameLength,
  resolveTextOverlaps,
  sceneShell,
  segmentedSeries,
  type SceneModel,
} from "../../shared/src/model.js";
import {
  axisModeBadge,
  badge,
  bandLayer,
  criticalBadge,
  lineLayer,
  makeAxis,
  pitfallsBand,
  scatterLayer,
  sourceBadge,
  text,
} from "../../shared/src/scene.js";
import { badgeUnitCap, layout35, stackTexts } from "./common.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "3.5";
const VIZ = "lyapunov_overlay";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "Lyapunov 指数叠在分岔图上";

/** 后端参数网格（`server/chapters/3.5.py` 里 lyapunov 的 r 网格就是这一段）；固定轴用。 */
const R_LO = 2.4;
const R_HI = 4.0;
/** λ 固定纵轴：上界 1 留出 ln2=0.693 的余量，下界 −6 覆盖默认网格的最小值。 */
const LAMBDA_YLIM: [number, number] = [-6, 1];
const LAMBDA_YTICKS = [-6, -4, -2, 0];
/** 后端给标准误数组的地板（`_floor(sem, 1e-18)`）：对数轴上"0"的占位，必须画出来并说明。 */
const SEM_FLOOR = 1e-18;
/** `manifest.gen.ts` 里 `r_probe.critical` 的**最紧**一条（3 周期窗口起点）：帧 `critical.r_probe` 缺失时的回落值。 */
const R_PROBE_CRITICAL = 3.8284271247461903;

const SLOT = {
  source: 0,
  mainAxis: 1,
  errAxis: 2,
  critical: 3,
  overflow: 4,
  status: 5,
  degraded: 6,
} as const;

export const lyapunovOverlayModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需数组 ——
  const lyapR = requireArray(frame, "lyap_r");
  const lambda = requireArray(frame, "lyap_lambda");
  const sem = requireArray(frame, "error_sem");
  const bifR = requireArray(frame, "bif_r");
  const bifX = requireArray(frame, "bif_x");
  const pitfalls = requirePitfalls(frame);
  requireSameLength(lyapR, lambda, "lyap_r 与 lyap_lambda");
  requireSameLength(lyapR, sem, "lyap_r 与 error_sem");
  requireSameLength(bifR, bifX, "bif_r 与 bif_x");

  // —— 可选字段 ——
  const degraded: string[] = [];
  const summary = paramSummary(frame, opts, [
    { name: "n_iter", label: "迭代次数" },
    { name: "n_block", label: "分块数" },
    { name: "n_transient", label: "暂态" },
  ]);
  degraded.push(...summary.degraded);
  const bandLo = frame.has("sem_band_lo") ? requireArray(frame, "sem_band_lo") : undefined;
  const bandHi = frame.has("sem_band_hi") ? requireArray(frame, "sem_band_hi") : undefined;
  const zeroX = frame.has("zero_x") ? requireArray(frame, "zero_x") : undefined;
  const zeroY = frame.has("zero_y") ? requireArray(frame, "zero_y") : undefined;
  const ln2X = frame.has("ln2_x") ? requireArray(frame, "ln2_x") : undefined;
  const ln2Y = frame.has("ln2_y") ? requireArray(frame, "ln2_y") : undefined;
  const anaR = frame.has("analytic_r") ? requireArray(frame, "analytic_r") : undefined;
  const anaL = frame.has("analytic_lambda") ? requireArray(frame, "analytic_lambda") : undefined;
  const meaL = frame.has("measured_lambda") ? requireArray(frame, "measured_lambda") : undefined;
  const markL = frame.has("mark_lambda") ? requireArray(frame, "mark_lambda") : undefined;
  if (bandLo === undefined || bandHi === undefined) degraded.push("帧缺 sem_band_lo/sem_band_hi：不画标准误带（不猜区间）");
  if (zeroX === undefined || zeroY === undefined) degraded.push("帧缺 zero_x/zero_y：λ=0 参考线不画（不猜位置）");
  if (ln2X === undefined || ln2Y === undefined) degraded.push("帧缺 ln2_x/ln2_y：λ=ln2 参考线不画（不猜位置）");
  if (anaR === undefined || anaL === undefined || meaL === undefined) {
    degraded.push("帧缺 analytic_r/analytic_lambda/measured_lambda：三个解析对拍点不画（不猜）");
  }

  const rProbe = metaNumber(frame, "r_probe");
  const lambdaProbe = metaNumber(frame, "lambda_probe");
  const semProbe = metaNumber(frame, "sem_probe");
  const periodProbe = metaNumber(frame, "period_probe");
  const residualProbe = metaNumber(frame, "residual_probe");
  const nPositive = metaNumber(frame, "n_positive");
  const nR = metaNumber(frame, "n_r");
  const fracPositive = metaNumber(frame, "frac_positive");
  const lambdaMax = metaNumber(frame, "lambda_max");
  const lambdaArgmax = metaNumber(frame, "lambda_argmax");
  const rAccum = metaNumber(frame, "r_accumulation");
  if (rProbe === undefined) degraded.push("帧 meta 缺 r_probe：不画当前参数竖线、不标临界值（不猜）");
  if (lambdaProbe === undefined) degraded.push("帧 meta 缺 lambda_probe：当前参数的 λ 写「?」（不猜）");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: metaNumber(frame, "t") ?? 0,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  // ------------------------------------------------------------------------- //
  // 排版：主图 + 标准误副轴（间距 50 让出主轴的刻度带与轴名）
  // ------------------------------------------------------------------------- //
  const panels = layout35(width, height, { mainShare: 0.62, gap: 50 });
  scene.axes.push(
    makeAxis("main", panels.main, {
      xlim: [R_LO, R_HI],
      ylim: LAMBDA_YLIM,
      xticks: [2.5, 3, 3.5, 4],
      yticks: LAMBDA_YTICKS,
      xlabel: "参数 r",
      ylabel: "λ(r)",
      title: "λ(r) 与分岔图",
      autoscaled: false,
    }),
  );

  // —— 图层 1：分岔图点云（淡化；同一套参数网格，用来对齐周期窗口）——
  scene.layers.push(
    scatterLayer("main", bifR, bifX, { r: 0.5, color: pal.reference, alpha: 0.55 }),
  );
  // —— 图层 2：标准误带 ——
  if (bandLo !== undefined && bandHi !== undefined && bandLo.length === lyapR.length && bandHi.length === lyapR.length) {
    scene.layers.push(
      bandLayer("main", { x: lyapR, ylo: bandLo, yhi: bandHi, color: pal.residual, alpha: 0.25 }),
    );
  }
  // —— 图层 3：两条参考线 ——
  const refs = [];
  if (zeroX !== undefined && zeroY !== undefined) {
    refs.push({ x: zeroX, y: zeroY, color: pal.muted, width: 1, dash: [4, 4] });
  }
  if (ln2X !== undefined && ln2Y !== undefined) {
    refs.push({ x: ln2X, y: ln2Y, color: pal.exact, width: 1, dash: [2, 3] });
  }
  if (refs.length > 0) scene.layers.push(lineLayer("main", refs));
  // —— 图层 4：λ(r) ——
  scene.layers.push(
    lineLayer("main", segmentedSeries(lyapR, lambda, pal.curves[0], { width: 1.8 })),
  );
  // —— 图层 5：当前参数 ——
  if (rProbe !== undefined) {
    scene.layers.push(
      lineLayer("main", [{ x: [rProbe, rProbe], y: LAMBDA_YLIM, color: pal.marker, width: 1.4, dash: [5, 3] }]),
    );
    if (markL !== undefined && markL.length > 0) {
      scene.layers.push(scatterLayer("main", [rProbe], [markL[0] as number], { r: 3.4, color: pal.marker }));
    }
  }
  // —— 图层 6：三个解析对拍点（解析=圈，实测=叉；两者几乎重合）——
  if (anaR !== undefined && anaL !== undefined && meaL !== undefined && anaL.length === meaL.length) {
    scene.layers.push(scatterLayer("main", anaR, anaL, { r: 4.4, color: pal.exact }));
    scene.layers.push(scatterLayer("main", anaR, meaL, { r: 3, color: pal.ink }));
  }

  // ------------------------------------------------------------------------- //
  // 副轴 `err`：分块标准误（对数）
  // ------------------------------------------------------------------------- //
  const semPositive = finitePositiveMinMax(concat(sem, Float64Array.from([SEM_FLOOR])));
  const semYlim: [number, number] =
    semPositive === undefined ? [1e-18, 1] : niceLimits(semPositive[0], semPositive[1], 5, { log: true, pad: 0.05 });
  scene.axes.push(
    makeAxis("err", panels.error, {
      xlim: [R_LO, R_HI],
      ylim: semYlim,
      yscale: "log",
      xticks: [2.5, 3, 3.5, 4],
      xlabel: "参数 r",
      ylabel: "标准误",
      title: "λ 的分块标准误",
      autoscaled: true,
    }),
  );
  scene.layers.push(
    lineLayer("err", [
      { x: [R_LO, R_HI], y: [SEM_FLOOR, SEM_FLOOR], color: pal.muted, width: 1, dash: [1, 3] },
    ]),
  );
  scene.layers.push(
    lineLayer("err", segmentedSeries(lyapR, sem, pal.residual, { width: 1.3, accept: (v) => v > 0 })),
  );

  let nAtFloor = 0;
  for (let i = 0; i < sem.length; i += 1) {
    const v = sem[i] as number;
    if (Number.isFinite(v) && v <= SEM_FLOOR * (1 + 1e-12)) nAtFloor += 1;
  }

  // ------------------------------------------------------------------------- //
  // 文本
  // ------------------------------------------------------------------------- //
  const topBlocks: { body: string; color: string; size: number; role: "label" | "note" }[] = [
    {
      body: "灰点=分岔图　蓝=λ(r)　紫带=±标准误　橙虚=ln2　灰虚=0　圈/叉=解析/实测",
      color: pal.muted,
      size: 10,
      role: "label" as const,
    },
  ];
  if (nPositive !== undefined && nR !== undefined) {
    topBlocks.push({
      body:
        `${formatNumber(nPositive, 4)}/${formatNumber(nR, 4)} 个参数 λ>0（${fmt(fracPositive === undefined ? undefined : fracPositive * 100, 2)}%）；` +
        `max λ=${fmt(lambdaMax, 5)}（r=${fmt(lambdaArgmax, 5)}）${rAccum === undefined ? "" : `；r∞≈${formatNumber(rAccum, 10)}`}`,
      color: pal.muted,
      size: 10,
      role: "note" as const,
    });
  }
  const analyticText =
    anaR !== undefined && anaL !== undefined && meaL !== undefined && anaR.length === anaL.length && anaR.length === meaL.length
      ? "解析/实测：" +
        Array.from(anaR).map((r, i) => `r=${fmt(r, 3)} ${fmt(anaL[i] as number, 5)}/${fmt(meaL[i] as number, 5)}`).join("；")
      : "解析/实测对拍点未给出";
  topBlocks.push({ body: analyticText, color: pal.ink, size: 10, role: "note" });
  {
    // λ 曲线在 r<3.6 都在 −0.7 以下，面板最底部（λ<−4）是空区；说明放这里不压数据。
    const st = stackTexts(
      panels.main[0] + 6,
      panels.main[1] + panels.main[3] - 4 * 15 - 6,
      panels.main[2] * 0.72,
      panels.main[1] + panels.main[3] - 6,
      topBlocks.map((b) => ({ body: b.body, color: b.color, size: b.size, role: b.role })),
    );
    scene.texts.push(...st.texts);
    if (st.dropped.length > 0) degraded.push(`主图图例放不下 ${st.dropped.length} 段：未显示`);
  }
  {
    scene.texts.push(
      text(
        panels.error[0] + 6,
        panels.error[1] + panels.error[3] - 6 - 14,
        `分块标准误（8 块/√8）：度量有限时间涨落，不是舍入误差；地板 ${formatNumber(SEM_FLOOR, 2)}=占位（${nAtFloor}/${sem.length}）`,
        { role: "note", size: 10, color: pal.warn, maxWidth: panels.error[2] * 0.92 },
      ),
    );
  }

  // ------------------------------------------------------------------------- //
  // 角标（规则 4/5/6）
  // ------------------------------------------------------------------------- //
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.mainAxis, width));
  scene.badges.push(placeBadge(badge("autoscale", "标准误轴：自适应对数（按数据取整）", 8, 8), SLOT.errAxis, width));
  if (rProbe !== undefined) {
    scene.badges.push(
      placeBadge(criticalBadge("r", rProbe, criticalOf(frame, "r_probe", R_PROBE_CRITICAL), { beyond: "above" }), SLOT.critical, width),
    );
  }
  scene.badges.push(
    placeBadge(
      badge(
        "status",
        clipBadgeText(
          `当前 r=${fmt(rProbe, 5)}：λ=${fmt(lambdaProbe, 5)}±${fmt(semProbe, 3)}、最小周期 ${fmt(periodProbe, 4)}（0=混沌）、闭合残差 ${fmt(residualProbe, 3)}`,
          badgeUnitCap(width),
        ),
        8,
        8,
      ),
      SLOT.status,
      width,
    ),
  );
  const status = degradedBadge(degraded, SLOT.degraded, width);
  if (status !== undefined) scene.badges.push(status);

  requireDrawableLayers(scene);
  applyOverflow(scene, countNonFinite(lambda) + countNonFinite(sem) + countNonFinite(bifX), { slot: SLOT.overflow, width });

  const panelTop = pitfallsBand(scene)?.[1];
  scene.texts = resolveTextOverlaps(scene.texts, [8, 24, width - 16, (panelTop ?? height) - 28]);
  return scene;
};

// --------------------------------------------------------------------------- //
// 局部工具
// --------------------------------------------------------------------------- //

function fmt(v: number | undefined, sig: number): string {
  return v === undefined ? "?" : formatNumber(v, sig);
}

function concat(a: ArrayLike<number>, b: ArrayLike<number>): Float64Array {
  const out = new Float64Array(a.length + b.length);
  for (let i = 0; i < a.length; i += 1) out[i] = a[i] as number;
  for (let i = 0; i < b.length; i += 1) out[a.length + i] = b[i] as number;
  return out;
}

