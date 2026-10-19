/**
 * `lcurve_tradeoff`（5.2 章 viz 1）：L 曲线、四个判据与 Picard 条件。
 *
 * 防误导纪律
 * ----------
 * 1. **L 曲线的拐点不是良定义**：三个拐点判据（曲率、两段直线交点、离散二阶差分）
 *    可能给出不同 λ；图上把它们的 λ 位置一起画出，侧栏报下标跨度。不要把某一个
 *    拐点当成“正确 λ”。
 * 2. **oracle 只作基准**：它用了真解 x*，现实中拿不到。其余判据的代价用
 *    `inflation_* = error_criterion / error_oracle` 表示，不是“谁更接近拐点”。
 * 3. **Picard 条件要分两段读**：系数高于噪声的那一段斜率决定 Picard 是否成立；
 *    噪声接管之后整段必然变平。`picard_below_noise` 标出被噪声吞掉的模态。
 * 4. **偏差原则可能没有交点**：帧里写 null/valid=0 是“没测到”，不是 λ=0。
 *
 * 读图纪律：所有线都不带内联文字。判据一多，曲线末端的 `label` 会叠成一团
 * （这正是 5.2 第一版基线上的事故）；颜色只有在侧栏用同色文字才能对上号。
 * 两个主面板之间留 72px：上图的刻度、下图的标题、σ 与 λ 的单位各自占用一条带。
 *
 * 帧 `time.mode="none"`：横轴是 λ 或模态序号，不读 `opts.t`。
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
  type SceneModel,
} from "../../shared/src/model.js";
import {
  axisModeBadge,
  badge,
  lineLayer,
  makeAxis,
  scatterLayer,
  sourceBadge,
} from "../../shared/src/scene.js";
import {
  TEXT_ROW_H,
  columnsLayout,
  positiveMinMax,
  textBlock,
  unitBudget,
} from "../_common/panel.js";
import {
  CHAPTER,
  CRITERIA,
  CRITERION_LABEL,
  SLOT,
  criterionColors,
  lambdaAt,
  nearestIndex,
  type Criterion,
} from "./common.js";
import { manifest } from "./manifest.gen.js";

const VIZ = "lcurve_tradeoff";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "L 曲线：λ 的四个判据";

const SHORT_LABEL: Record<Criterion, string> = {
  oracle: "oracle",
  lcurve: "L 拐点",
  gcv: "GCV",
  discrepancy: "偏差原则",
};

export const lcurveTradeoffModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  const lgLam = requireArray(frame, "log10_lam");
  const lgX = requireArray(frame, "log10_x_norm");
  const lgRes = requireArray(frame, "log10_residual");
  const errCurve = requireArray(frame, "error_curve");
  const err = errCurve;
  requireSameLength(lgLam, lgX, "log10_lam 与 log10_x_norm");
  requireSameLength(lgLam, lgRes, "log10_lam 与 log10_residual");
  requireSameLength(lgLam, errCurve, "log10_lam 与 error_curve");
  const critLam = requireArray(frame, "criterion_log10_lam");
  const critErr = requireArray(frame, "criterion_error");
  const critValid = requireArray(frame, "criterion_valid");
  requireSameLength(critLam, critErr, "criterion_log10_lam 与 criterion_error");
  requireSameLength(critLam, critValid, "criterion_log10_lam 与 criterion_valid");
  if (critLam.length !== CRITERIA.length) {
    throw new Error(`criterion_* 长度 ${critLam.length} ≠ ${CRITERIA.length}（顺序：oracle/lcurve/gcv/discrepancy）`);
  }
  const cornerLam = frame.has("corner_log10_lam") ? requireArray(frame, "corner_log10_lam") : undefined;
  const pSigma = frame.has("picard_sigma") ? requireArray(frame, "picard_sigma") : undefined;
  const pCoeff = frame.has("picard_coeff") ? requireArray(frame, "picard_coeff") : undefined;
  const pBelow = frame.has("picard_below_noise") ? requireArray(frame, "picard_below_noise") : undefined;
  if (pSigma !== undefined && pCoeff !== undefined) requireSameLength(pSigma, pCoeff, "picard_sigma 与 picard_coeff");
  const pitfalls = requirePitfalls(frame);

  const degraded: string[] = [];
  if (cornerLam === undefined) degraded.push("帧缺 corner_log10_lam：不画三个拐点判据的位置");
  if (pSigma === undefined || pCoeff === undefined) degraded.push("帧缺 Picard 数组：右上角不画 Picard 图");
  if (pBelow === undefined && pSigma !== undefined) degraded.push("帧缺 picard_below_noise：不标被噪声吞掉的模态");

  const lamNow = metaNumber(frame, "log10_lam_current");
  const dofNow = metaNumber(frame, "dof_current");
  const errNow = metaNumber(frame, "error_current");
  const resNow = metaNumber(frame, "residual_current");
  const kappa = metaNumber(frame, "kappa_measured");
  const noiseRel = metaNumber(frame, "noise_rel");
  const slopeSignal = metaNumber(frame, "picard_slope_signal");
  const slopeAll = metaNumber(frame, "picard_slope_all");
  const spread = metaNumber(frame, "corner_index_spread");
  const discrepancyFound = frame.meta["discrepancy_found"] === true;
  const oracleEdge = frame.meta["oracle_at_grid_edge"] === true;

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: 0,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  const L = columnsLayout(scene, width, height, { topShare: 0.52, auxH: 140, gap: 72 });

  const lxMinMax = finiteMinMax(lgX);
  const lyMinMax = finiteMinMax(lgRes);
  const lo = (v: [number, number] | undefined, pad: number): [number, number] =>
    v === undefined ? [-5, 5] : [v[0] - pad, v[1] + pad];
  const xr = lo(lxMinMax, 0.08);
  const yr = lo(lyMinMax, 0.08);
  const lcurve = makeAxis("lcurve", [L.left, L.topY, L.leftW, L.topH], {
    xlim: xr,
    ylim: yr,
    xlabel: "",
    ylabel: "log10 残差",
    title: "L 曲线（横=log10‖x‖，纵=log10 残差）",
    autoscaled: true,
  });

  // 真误差与 GCV 的纵轴只包住数据本身（0 在对数轴上仍不可画，用正值下限）。
  const errPos = positiveMinMax(err) ?? [1e-12, 1];
  const errLo = Math.max(1e-17, errPos[0] * 0.5);
  const errHi = Math.max(errLo * 10, errPos[1] * 1.5);
  const errAxis = makeAxis("error", [L.left, L.botY, L.leftW, L.botH], {
    xlim: [lgLam[0] as number, lgLam[lgLam.length - 1] as number],
    ylim: [errLo, errHi],
    yscale: "log",
    xlabel: "log10 λ",
    ylabel: "相对误差",
    title: "真误差随 λ（对数轴）",
    autoscaled: true,
  });
  scene.axes.push(lcurve, errAxis);

  const colors = criterionColors(pal);

  scene.layers.push(lineLayer("lcurve", { x: lgX, y: lgRes, color: pal.ink, width: 2 }));
  scene.layers.push(lineLayer("error", { x: lgLam, y: err, color: pal.ink, width: 1.8 }));

  // 三个拐点判据的位置：只画点，不假装它们一致；线不带文字。
  if (cornerLam !== undefined) {
    const cxs: number[] = [];
    const cys: number[] = [];
    for (let i = 0; i < cornerLam.length; i += 1) {
      const j = nearestIndex(lgLam, cornerLam[i] as number);
      if (j < 0) continue;
      cxs.push(lgX[j] as number);
      cys.push(lgRes[j] as number);
      scene.layers.push(
        lineLayer("error", {
          x: [cornerLam[i] as number, cornerLam[i] as number],
          y: [errAxis.ylim[0], errAxis.ylim[1]],
          color: pal.muted,
          width: 0.9,
          dash: [1, 3],
          alpha: 0.8,
        }),
      );
    }
    if (cxs.length > 0) {
      scene.layers.push(scatterLayer("lcurve", cxs, cys, { r: 2.8, color: pal.muted, label: "拐点判据（3 个）" }));
    }
  }

  // 四个判据：只在 valid=1 时画；无效项的 λ 不参与任何“最优”比较。
  for (let i = 0; i < CRITERIA.length; i += 1) {
    const name = CRITERIA[i]!;
    const valid = (critValid[i] as number) > 0.5 && Number.isFinite(critLam[i] as number);
    if (!valid) continue;
    const lamC = critLam[i] as number;
    const j = nearestIndex(lgLam, lamC);
    if (j >= 0) {
      scene.layers.push(
        scatterLayer("lcurve", [lgX[j] as number], [lgRes[j] as number], {
          r: name === "oracle" ? 4.2 : 3.6,
          color: colors[name],
          label: CRITERION_LABEL[name],
        }),
      );
    }
    if (lamC >= errAxis.xlim[0] && lamC <= errAxis.xlim[1]) {
      scene.layers.push(
        lineLayer("error", {
          x: [lamC, lamC],
          y: [errAxis.ylim[0], errAxis.ylim[1]],
          color: colors[name],
          width: name === "oracle" ? 1.5 : 1.1,
          dash: name === "oracle" ? [5, 3] : [3, 3],
        }),
      );
      const e = critErr[i] as number;
      if (Number.isFinite(e) && e >= errAxis.ylim[0] && e <= errAxis.ylim[1]) {
        scene.layers.push(scatterLayer("error", [lamC], [e], { r: 3, color: colors[name], label: CRITERION_LABEL[name] }));
      }
    }
  }
  if (lamNow !== undefined) {
    const j = nearestIndex(lgLam, lamNow);
    if (j >= 0) {
      scene.layers.push(
        scatterLayer("lcurve", [lgX[j] as number], [lgRes[j] as number], { r: 5, color: pal.marker, label: "当前 λ（滑块）" }),
        scatterLayer("error", [lamNow], [err[j] as number], { r: 4.4, color: pal.marker, label: "当前 λ（滑块）" }),
      );
    }
    scene.layers.push(
      lineLayer("error", {
        x: [lamNow, lamNow],
        y: [errAxis.ylim[0], errAxis.ylim[1]],
        color: pal.marker,
        width: 1.2,
        dash: [2, 2],
      }),
    );
  }

  // Picard 条件：右列上方；同样不带内联标签，颜色约定写在侧栏。
  let picardAxis: ReturnType<typeof makeAxis> | undefined;
  if (pSigma !== undefined && pCoeff !== undefined) {
    let sigLo = Number.POSITIVE_INFINITY;
    let sigHi = 0;
    let cLo = Number.POSITIVE_INFINITY;
    let cHi = 0;
    for (let i = 0; i < pSigma.length; i += 1) {
      const s = pSigma[i] as number;
      const c = pCoeff[i] as number;
      if (s > 0 && Number.isFinite(s)) {
        sigLo = Math.min(sigLo, s);
        sigHi = Math.max(sigHi, s);
      }
      if (c > 0 && Number.isFinite(c)) {
        cLo = Math.min(cLo, c);
        cHi = Math.max(cHi, c);
      }
    }
    const picardH = Math.max(110, Math.min(140, Math.floor((L.floor - L.topY) * 0.38)));
    picardAxis = makeAxis("picard", [L.rightX, L.topY, L.rightW, picardH], {
      xlim: [sigLo * 0.7, sigHi * 1.4],
      ylim: [cLo * 0.5, cHi * 2],
      xscale: "log",
      yscale: "log",
      xlabel: "σ_i（对数）",
      ylabel: "|u_i^T b|（对数）",
      title: "Picard：系数 vs σ",
      autoscaled: true,
    });
    scene.axes.push(picardAxis);
    scene.layers.push(lineLayer("picard", { x: pSigma, y: pCoeff, color: pal.curves[0] as string, width: 1.8 }));
    if (pSigma.length >= 2 && (pSigma[0] as number) > 0 && (pCoeff[0] as number) > 0) {
      const s0 = pSigma[0] as number;
      const c0 = pCoeff[0] as number;
      scene.layers.push(
        lineLayer("picard", {
          x: [s0, sigHi],
          y: [c0, c0 * (sigHi / s0)],
          color: pal.exact,
          width: 1.2,
          dash: [5, 3],
        }),
      );
    }
    if (pBelow !== undefined) {
      const bx: number[] = [];
      const by: number[] = [];
      for (let i = 0; i < Math.min(pBelow.length, pSigma.length); i += 1) {
        if ((pBelow[i] as number) > 0.5 && (pCoeff[i] as number) > 0) {
          bx.push(pSigma[i] as number);
          by.push(pCoeff[i] as number);
        }
      }
      if (bx.length > 0) {
        scene.layers.push(scatterLayer("picard", bx, by, { r: 2.8, color: pal.warn, label: "系数已低于噪声" }));
      }
    }
  }

  // 右列文字：每个判据一行，颜色与图上的点/竖线一致；短行避免折行。
  const budget = unitBudget(L.rightW);
  const picardH = picardAxis !== undefined ? (picardAxis.rect[3] as number) : 0;
  const textTop = picardAxis !== undefined ? L.topY + picardH + 34 : L.topY;
  if (L.rightW >= 200 && L.floor - textTop >= 3 * TEXT_ROW_H) {
    let y = textTop;
    const push = (body: string, color: string): void => {
      const block = textBlock(L.rightX, y, L.floor, [body], color, budget, "top");
      scene.texts.push(...block);
      y += block.length * TEXT_ROW_H;
    };
    push(`κ=${formatNumber(kappa ?? Number.NaN, 2)}，噪声相对 ${formatNumber(noiseRel ?? Number.NaN, 2)}`, pal.muted);
    const oracleE = critErr[0] as number;
    for (let i = 0; i < CRITERIA.length; i += 1) {
      const name = CRITERIA[i]!;
      const valid = (critValid[i] as number) > 0.5;
      const e = critErr[i] as number;
      if (name === "discrepancy" && (!valid || !discrepancyFound)) {
        push("偏差原则整条网格没有交点（null ≠ λ=0）", pal.muted);
        continue;
      }
      if (!valid) {
        push(`${SHORT_LABEL[name]}：没测到（valid=0）`, pal.muted);
        continue;
      }
      const infl = name === "oracle" ? 1 : e > 0 && oracleE > 0 ? e / oracleE : Number.NaN;
      const suffix = name === "oracle" ? (oracleEdge ? "；落在端点" : "；仅基准") : `；×${formatNumber(infl, 2)}`;
      push(`${SHORT_LABEL[name]}：err=${formatNumber(e, 3)}${suffix}`, colors[name]);
    }
    push(`拐点下标跨度 ${formatNumber(spread ?? Number.NaN, 0)}：三个拐点判据不保证一致。`, pal.warn);
    push("黑线=L 曲线；红大点=当前 λ；橙点=oracle；灰点=三个拐点判据。", pal.muted);
    if (slopeSignal !== undefined || slopeAll !== undefined) {
      push(
        `Picard：信号段斜率 ${formatNumber(slopeSignal ?? Number.NaN, 2)}（>1），整段 ${formatNumber(slopeAll ?? Number.NaN, 2)}；橙点=已低于噪声。`,
        pal.muted,
      );
    }
    push("两轴都是 log10 值；轴本身线性，别再取对数。", pal.muted);
    push("判据顺序：oracle→L 拐点→GCV→偏差原则。", pal.muted);
  } else {
    degraded.push("右列过窄/过矮：判据读数并入角标，口径见 pitfalls");
  }

  scene.badges.push(placeBadge(sourceBadge(dataSourceOf(frame)), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(true), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(
      badge(
        "status",
        `${lambdaAt(lamNow)}｜dof=${formatNumber(dofNow ?? Number.NaN, 2)}｜err=${formatNumber(errNow ?? Number.NaN, 2)}｜res=${formatNumber(resNow ?? Number.NaN, 2)}`,
        8,
        8,
      ),
      SLOT.status,
      width,
    ),
  );
  const lamSpec = SPEC?.params.find((p) => p.name === "log10_lam");
  const lamCritical = lamSpec?.critical?.[0];
  if (lamNow !== undefined && lamCritical !== undefined && lamNow >= lamCritical) {
    scene.badges.push(
      placeBadge(badge("critical", `λ=10^${formatNumber(lamNow, 2)} ≥ 1：解被系统性压小（滤波因子 ≤ 1/2）`, 8, 8), SLOT.critical, width),
    );
  }
  const dbg = degradedBadge(degraded, SLOT.degraded, width);
  if (dbg !== undefined) scene.badges.push(dbg);

  applyOverflow(scene, 0, { slot: SLOT.overflow, width });
  requireDrawableLayers(scene);
  return scene;
};
