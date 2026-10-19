/**
 * `modal_decay`（4.6 章 viz 1）的 model：模态系数衰减 + 局部斜率 + 逼近误差阶梯。
 *
 * 三条防误导纪律（都落在具体图层/角标上）
 * ----------------------------------------
 * 1. **主图必须双对数**：`decay` 的 x/y 都是对数轴（manifest 固定窗口 1e-18–1）。
 *    代数衰减是直线，指数衰减是凹向下的曲线；换成线性轴两者都看不出来。
 *    `coeff_theory`（帧算好的 C·rho^k 或 C·k^-p）作为**理论参考线**同屏，
 *    理论斜率同时写进图例与右列文字（解析剖面没有固定幂律斜率，如实说明）。
 * 2. **「指数收敛」看斜率是否在变，不看某个拟合 p**：副图的 `local_slope` 逐点画，
 *    窗口固定 -24–0；代数衰减是水平线，指数衰减一路变陡。越窗的点由溢出角标点名。
 * 3. **误差轴是规则 3 的落点**：光滑/尖角/跳变三条误差-自由度阶梯同屏。跳变那条
 *    **不随 N 下降**（Gibbs），水平参考线画出 N=8 的首值；尖角那条按 N^-1 下降，
 *    参考线斜率固定 -1。地板线与掩码都显式画出。
 *
 * 帧 `time.mode="none"`：`scene.t` 恒为 0，模型不读 `opts.t`。
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
  type SceneModel,
} from "../../shared/src/model.js";
import {
  axisModeBadge,
  badge,
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
const VIZ = "modal_decay";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "模态衰减：指数、代数、以及根本不降";

/** 方案约定的两个地板（帧 meta 优先；缺 meta 时用这两个**登记过的**常量并降级）。 */
const COEF_FLOOR_FALLBACK = 1e-18;
const ERR_FLOOR_FALLBACK = 1e-17;

/** 固定角标槽位：缺一枚时其余不挪位。 */
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

export const modalDecayModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需数组：缺一个就抛 ModelInputError ——
  const modeK = requireArray(frame, "mode_k");
  const coeff = requireArray(frame, "coeff_abs");
  const theory = requireArray(frame, "coeff_theory");
  const slopeK = requireArray(frame, "slope_k");
  const slope = requireArray(frame, "local_slope");
  const errN = requireArray(frame, "error_n");
  const errSmooth = requireArray(frame, "error_smooth");
  const errCorner = requireArray(frame, "error_corner");
  const errJump = requireArray(frame, "error_jump");
  requireSameLength(modeK, coeff, "帧数组 mode_k 与 coeff_abs");
  requireSameLength(modeK, theory, "帧数组 mode_k 与 coeff_theory");
  requireSameLength(slopeK, slope, "帧数组 slope_k 与 local_slope");
  for (const [name, arr] of [
    ["error_smooth", errSmooth],
    ["error_corner", errCorner],
    ["error_jump", errJump],
  ] as const) {
    requireSameLength(errN, arr, `帧数组 error_n 与 ${name}`);
  }
  const pitfalls = requirePitfalls(frame);

  // —— 可选掩码与 meta：缺了只降级，不抛错 ——
  const degraded: string[] = [];
  const coeffFloorMask = frame.has("coeff_floor_mask") ? requireArray(frame, "coeff_floor_mask") : undefined;
  if (coeffFloorMask === undefined) degraded.push("帧缺 coeff_floor_mask：不标「已到系数地板」的模态（不猜哪些贴了地板）");
  const slopeFloorMask = frame.has("slope_floor_mask") ? requireArray(frame, "slope_floor_mask") : undefined;
  if (slopeFloorMask === undefined) degraded.push("帧缺 slope_floor_mask：不标「斜率不可测」的点");
  const errFloorMask = frame.has("error_floor_mask") ? requireArray(frame, "error_floor_mask") : undefined;
  if (errFloorMask === undefined) degraded.push("帧缺 error_floor_mask：不标误差阶梯里的地板点");

  const coefFloorMeta = metaNumber(frame, "coeff_floor");
  const errFloorMeta = metaNumber(frame, "error_floor");
  if (coefFloorMeta === undefined) degraded.push("帧 meta 缺 coeff_floor：系数地板用登记常量 1e-18");
  if (errFloorMeta === undefined) degraded.push("帧 meta 缺 error_floor：误差地板用登记常量 1e-17");
  const coefFloor = coefFloorMeta !== undefined && coefFloorMeta > 0 ? coefFloorMeta : COEF_FLOOR_FALLBACK;
  const errFloor = errFloorMeta !== undefined && errFloorMeta > 0 ? errFloorMeta : ERR_FLOOR_FALLBACK;

  const basis = stringOf(frame, "basis");
  const profile = stringOf(frame, "profile");

  let slopeOut = 0;
  for (let i = 0; i < slope.length; i += 1) {
    const v = slope[i] as number;
    if (Number.isFinite(v) && (v < -24 || v > 0)) slopeOut += 1;
  }
  const fitP = metaNumber(frame, "fit_p");
  const fitR2 = metaNumber(frame, "fit_r2");
  const fitLo = metaNumber(frame, "fit_lo");
  const fitHi = metaNumber(frame, "fit_hi");
  const slopeFirst = metaNumber(frame, "slope_first");
  const slopeLast = metaNumber(frame, "slope_last");
  const slopeSteepens = frame.meta["slope_steepens"] === true || (metaNumber(frame, "slope_steepens") ?? 0) > 0.5;
  const theoryPMeta = metaNumber(frame, "theory_p");
  // 帧的 meta.theory_p 只按 Fourier 侧给出（server 用 _FOURIER_KIND 映射）；
  // Chebyshev 的方波剖面是 |x|^3，理论斜率是 k^-4（MANIFEST theory 第 4 条），
  // 这里按帧 strings 的 basis/profile 校正显示标签（参考线 coeff_theory 本身就是 k^-4）。
  const theoryP =
    theoryPMeta !== undefined && basis === "chebyshev" && profile !== undefined && profile.includes("方波")
      ? 4
      : theoryPMeta;
  const errCornerSlope = metaNumber(frame, "err_corner_slope");
  const errJumpFirst = metaNumber(frame, "err_jump_first");
  const errJumpLast = metaNumber(frame, "err_jump_last");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: 0,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  const L = columnsLayout(scene, width, height, { topShare: 0.58, auxH: 130, gap: 46 });
  // 窄/矮画布：pitfalls 面板会占满底部，任何文本层都会压住它（规则 7/8）。
  // 这里宁可少说也不叠着说——放弃面板说明并显式降级（角标仍保留）。
  const panelTextFits = L.floor > L.botY + L.botH + 8;

  // —— 主图：只保留 k>0（对数 x 轴的定义域；k=0 的均值模态不在本图）——
  const keep: number[] = [];
  for (let i = 0; i < modeK.length; i += 1) {
    const kv = modeK[i] as number;
    if (Number.isFinite(kv) && kv > 0) keep.push(i);
  }
  if (keep.length === 0) {
    throw new Error(`帧 ${CHAPTER}/${VIZ} 的 mode_k 没有正模态：双对数主图无法成立`);
  }
  const kx = selectValues(modeK, keep);
  const amp = selectValues(coeff, keep);
  const env = selectValues(theory, keep);
  const kLo = kx[0] as number;
  const kHi = kx[kx.length - 1] as number;
  let peakAmp = 0;
  for (let i = 0; i < amp.length; i += 1) {
    const v = amp[i] as number;
    if (Number.isFinite(v) && v > peakAmp) peakAmp = v;
  }

  const decay = makeAxis("decay", [L.left, L.topY, L.leftW, L.topH], {
    xlim: [kLo, Math.max(kLo * 1.2, kHi)],
    ylim: [coefFloor, 1],
    xscale: "log",
    yscale: "log",
    xlabel: "k（模态序号，对数）",
    title: axisTitleOf(SPEC?.axes, "decay", "模态系数 |a_k|（双对数，窗口 1e-18–1）"),
    ylabel: axisYlabelOf(SPEC?.axes, "decay", "|a_k|"),
    autoscaled: false,
  });

  // —— 局部斜率：窗口固定 -24–0（manifest）。越窗的点由 applyOverflow 点名 ——
  const slopeXHi = slopeK.length > 0 ? (slopeK[slopeK.length - 1] as number) : kHi;
  const slopeXLo = slopeK.length > 0 ? (slopeK[0] as number) : kLo;
  const slopeAxis = makeAxis("slope", [L.left, L.botY, L.leftW, L.botH], {
    xlim: [slopeXLo, Math.max(slopeXLo * 1.2, slopeXHi)],
    ylim: [-24, 0],
    xlabel: "k（模态序号）",
    title: axisTitleOf(SPEC?.axes, "slope", "局部斜率（窗口 −24–0）"),
    ylabel: axisYlabelOf(SPEC?.axes, "slope", "dlog|a_k|/dlog k"),
    autoscaled: false,
  });

  // —— error 副轴（规则 3）：三类光滑性的逼近误差阶梯，固定 1e-17–10 ——
  const nLo = errN[0] as number;
  const nHi = errN[errN.length - 1] as number;
  const errAxis = makeAxis("error", [L.rightX, L.topY, L.rightW, L.auxH], {
    xlim: [nLo, Math.max(nLo * 1.2, nHi)],
    ylim: [errFloor, 10],
    xscale: "log",
    yscale: "log",
    xlabel: "自由度 N",
    title: axisTitleOf(SPEC?.axes, "error", "误差 vs N（双对数，1e-17–10）"),
    ylabel: axisYlabelOf(SPEC?.axes, "error", "||f - S_N f||_inf"),
    autoscaled: false,
  });
  scene.axes.push(decay, slopeAxis, errAxis);

  // —— 主图图层：地板线 → 理论包络 → 实测曲线 → 地板散点 ——
  scene.layers.push(
    lineLayer("decay", series([decay.xlim[0], decay.xlim[1]], [coefFloor, coefFloor], pal.warn, { dash: [3, 3], width: 1 })),
    lineLayer("decay", series(kx, env, pal.exact, { dash: [5, 4], width: 1.5 })),
    lineLayer("decay", series(kx, amp, pal.curves[0], { width: 2.2 })),
  );
  if (coeffFloorMask !== undefined && coeffFloorMask.length === modeK.length) {
    const atFloor: number[] = [];
    for (const i of keep) if ((coeffFloorMask[i] ?? 0) > 0.5) atFloor.push(i);
    const pts = selectValues(modeK, atFloor);
    const vals = selectValues(coeff, atFloor);
    if (pts.length > 0) {
      scene.layers.push(scatterLayer("decay", pts, vals, { r: 2.8, color: pal.warn, label: "已到系数地板" }));
    }
  }
  // 拟合窗口（局部斜率拟合 p 的窗口）用两条细竖线标出：窗口是读数的一部分。
  if (fitLo !== undefined && fitHi !== undefined && fitLo > 0 && fitHi > fitLo) {
    for (const x of [fitLo, fitHi]) {
      if (x >= decay.xlim[0] && x <= decay.xlim[1]) {
        scene.layers.push(
          lineLayer("decay", series([x, x], [decay.ylim[0], decay.ylim[1]], pal.reference, { dash: [2, 4], width: 1 })),
        );
      }
    }
  }

  // —— 斜率副图：y=0（不衰减）与理论斜率水平线 ——
  scene.layers.push(
    lineLayer("slope", series([slopeAxis.xlim[0], slopeAxis.xlim[1]], [0, 0], pal.reference, { dash: [3, 3], width: 1 })),
  );
  if (theoryP !== undefined && theoryP > 0) {
    const y = -theoryP;
    if (y >= slopeAxis.ylim[0] && y <= slopeAxis.ylim[1]) {
      scene.layers.push(
        lineLayer(
          "slope",
          series([slopeAxis.xlim[0], slopeAxis.xlim[1]], [y, y], pal.exact, { dash: [5, 4], width: 1.3 }),
        ),
      );
    }
  }
  if (slope.length >= 2) {
    scene.layers.push(lineLayer("slope", series(slopeK, slope, pal.curves[0], { width: 2 })));
  } else if (slope.length === 1) {
    scene.layers.push(scatterLayer("slope", slopeK, slope, { r: 3, color: pal.curves[0] }));
  }
  if (slopeFloorMask !== undefined && slopeFloorMask.length === slope.length) {
    const bad: number[] = [];
    for (let i = 0; i < slopeFloorMask.length; i += 1) if ((slopeFloorMask[i] ?? 0) > 0.5) bad.push(i);
    if (bad.length > 0) {
      scene.layers.push(
        scatterLayer("slope", selectValues(slopeK, bad), selectValues(slope, bad), {
          r: 2.6,
          color: pal.warn,
          label: "斜率不可测（系数在地板）",
        }),
      );
    }
  }

  // —— 误差阶梯：三条曲线 + 地板线 + 两条理论参考线 ——
  scene.layers.push(
    lineLayer("error", series([errAxis.xlim[0], errAxis.xlim[1]], [errFloor, errFloor], pal.warn, { dash: [3, 3], width: 1 })),
    lineLayer("error", series(errN, errSmooth, pal.curves[0], { width: 2.2 })),
    lineLayer("error", series(errN, errCorner, pal.exact, { width: 1.8, dash: [6, 3] })),
    lineLayer("error", series(errN, errJump, pal.residual, { width: 1.8, dash: [2, 3] })),
  );
  // 尖角误差 ~ N^-1：过首个数据点画一条 -1 参考线（只覆盖轴的 x 范围）。
  if (errCorner.length > 0 && Number.isFinite(errCorner[0] as number) && (errCorner[0] as number) > 0) {
    const y0 = Math.max(errCorner[0] as number, errFloor);
    const yEnd = Math.max(y0 * Math.pow(nHi / nLo, -1), errFloor);
    scene.layers.push(
      lineLayer("error", series([nLo, nHi], [y0, yEnd], pal.reference, { dash: [5, 4], width: 1.2 })),
    );
  }
  // 跳变误差不降：水平参考线放在首值（Gibbs 只压窄、不压平）。
  if (errJump.length > 0 && Number.isFinite(errJump[0] as number) && (errJump[0] as number) > 0) {
    const y = errJump[0] as number;
    scene.layers.push(
      lineLayer("error", series([errAxis.xlim[0], errAxis.xlim[1]], [y, y], pal.residual, { dash: [4, 3], width: 1 })),
    );
  }
  if (errFloorMask !== undefined && errFloorMask.length === errN.length) {
    const fl: number[] = [];
    for (let i = 0; i < errFloorMask.length; i += 1) if ((errFloorMask[i] ?? 0) > 0.5) fl.push(i);
    if (fl.length > 0) {
      scene.layers.push(
        scatterLayer("error", selectValues(errN, fl), selectValues(errSmooth, fl), {
          r: 2.6,
          color: pal.warn,
          label: "误差已到地板",
        }),
      );
    }
  }

  // —— 面板内图例（构造上不越过面板宽度；窄画布下整块放弃）——
  if (panelTextFits) {
    scene.texts.push(
      ...textBlock(
        L.left + 6,
        L.topY + L.topH - SAFE_GAP - 7 * TEXT_ROW_H,
        L.topY + L.topH - SAFE_GAP - TEXT_ROW_H,
        [
          "灰虚=理论包络　蓝实=实测 |a_k|　红虚线=系数地板",
          "红点=贴地板；灰竖线=拟合窗口 [k_lo, k_hi]",
        ],
        pal.muted,
        unitBudget(L.leftW),
        "bottom",
      ),
      ...textBlock(
        L.left + 6,
        L.botY + L.botH - SAFE_GAP - 7 * TEXT_ROW_H,
        L.botY + L.botH - SAFE_GAP - TEXT_ROW_H,
        ["蓝实=局部斜率　灰虚 y=0（不衰减）　橙虚=理论斜率", "越窗=零点邻域的对数差商（噪声，不是物理斜率）"],
        pal.muted,
        unitBudget(L.leftW),
        "bottom",
      ),
      ...textBlock(
        L.rightX + 6,
        L.topY + L.auxH - SAFE_GAP - 7 * TEXT_ROW_H,
        L.topY + L.auxH - SAFE_GAP - TEXT_ROW_H,
        ["蓝=光滑　橙虚=尖角　紫虚=跳变", "灰虚=N^-1 理论；红线=误差地板；红点=贴地板"],
        pal.muted,
        unitBudget(L.rightW),
        "bottom",
      ),
    );
  } else {
    degraded.push("画布过矮/过窄：面板说明与右列文字放不下（pitfalls 面板占满底部），关键读数见角标");
  }

  // —— 右列文字 ——
  if (L.columns && L.floor - L.textTop >= 32) {
    const head = paramSummary(frame, opts, [
      { name: "n", label: "N" },
      { name: "rho", label: "ρ", digits: 3 },
    ]);
    degraded.push(...head.degraded);
    const lines: string[] = [];
    const shortProfile = profile !== undefined ? profile.split("（")[0] : "?";
    lines.push(`${basis ?? "?"}/${shortProfile}　${head.text}`);
    if (fitP !== undefined && fitR2 !== undefined) {
      lines.push(
        `拟合 p=${formatNumber(fitP, 4)}　R²=${formatNumber(fitR2, 3)}` +
          (fitLo !== undefined && fitHi !== undefined ? `（k∈[${formatNumber(fitLo, 3)}, ${formatNumber(fitHi, 3)}]）` : ""),
      );
    } else {
      degraded.push("帧 meta 缺 fit_p/fit_r2：不写拟合 p（单看 p 不是指数收敛的证据）");
      lines.push("拟合 p 不可用：只报局部斜率与每加倍比值");
    }
    if (slopeFirst !== undefined && slopeLast !== undefined) {
      if (slopeFirst < -24 || slopeLast < -24) {
        lines.push(
          `斜率 ${formatNumber(slopeFirst, 3)}→${formatNumber(slopeLast, 3)} 越窗（${slopeOut} 点）：零点邻域噪声，不据此判阶`,
        );
      } else {
        lines.push(
          `斜率 ${formatNumber(slopeFirst, 3)}→${formatNumber(slopeLast, 3)}：` +
            (slopeSteepens ? "在变=指数" : "近似常数=代数"),
        );
      }
    }
    if (theoryP !== undefined && theoryP > 0) {
      lines.push(`理论 a_k~k^-${formatNumber(theoryP, 2)}（橙虚参考线）`);
    } else if (profile !== undefined && profile.includes("解析")) {
      lines.push("理论 a_k~ρ^k（几何，无固定幂律）");
    } else {
      degraded.push("帧 meta 缺 theory_p 且剖面不是解析型：不写理论幂律（只画 coeff_theory 包络）");
      lines.push("理论幂律未给出：以橙虚包络为准");
    }
    if (errCornerSlope !== undefined) {
      lines.push(`尖角：系数 -2，误差 ${formatNumber(errCornerSlope, 3)}（差一阶）`);
    }
    if (errJumpFirst !== undefined && errJumpLast !== undefined) {
      lines.push(
        `跳变误差 ${formatNumber(errJumpFirst, 3)}→${formatNumber(errJumpLast, 3)}：` +
          (errJumpLast >= 0.8 * errJumpFirst ? "基本不降" : "下降"),
      );
    }
    if (peakAmp > 1) lines.push(`系数峰值 ${formatNumber(peakAmp, 3)} > 1：溢出点名`);
    scene.texts.push(...textBlock(L.rightX, L.textTop, L.floor, lines, pal.muted, unitBudget(L.rightW), "top"));
  } else if (L.columns) {
    degraded.push("画布过矮：右列文字放不下（pitfalls 面板之下），关键读数见角标");
  } else {
    degraded.push("窄画布（单列）：右列说明未排下，关键读数见角标与 pitfalls 面板");
  }

  // —— 角标 ——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(
      badge("autoscale", "窗口固定：三轴均不随帧变（范围见标题）", 8, 26),
      SLOT.aux,
      width,
    ),
  );
  const status: string[] = [];
  if (theoryP !== undefined && theoryP > 0) status.push(`k^-${formatNumber(theoryP, 2)}`);
  else status.push("ρ^k");
  const ratioFirst = metaNumber(frame, "ratio_smooth_first");
  const ratioLast = metaNumber(frame, "ratio_smooth_last");
  if (ratioFirst !== undefined && ratioLast !== undefined) {
    status.push(`光滑比 ${formatNumber(ratioFirst, 2)}→${formatNumber(ratioLast, 1)}`);
  }
  if (slopeLast !== undefined) {
    status.push(slopeLast < -24 ? `斜率越窗 ${slopeOut} 点` : `末斜率 ${formatNumber(slopeLast, 3)}`);
  }
  if (errJumpFirst !== undefined && errJumpLast !== undefined && errJumpLast >= 0.8 * errJumpFirst) status.push("跳变不降");
  scene.badges.push(placeBadge(badge("status", status.join("；")), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  const nBad =
    countNonFinite(modeK) +
    countNonFinite(coeff) +
    countNonFinite(theory) +
    countNonFinite(slope) +
    countNonFinite(errSmooth) +
    countNonFinite(errCorner) +
    countNonFinite(errJump);
  applyOverflow(scene, nBad, { slot: SLOT.overflow, width });
  return scene;
};
