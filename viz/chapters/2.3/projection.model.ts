/**
 * `projection`（2.3 章）的 model：把 f 与它的两个投影画在一起，并把「系数」与「误差」同时端出来。
 *
 * 面板与图层（顺序即绘制顺序，测试按此断言）
 * ------------------------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | line | `main` | `f`（橙实线，被逼近的函数） |
 * | 2 | line | `main` | `proj_target`（灰虚线：目标 n 维投影 P_n f） |
 * | 3 | line | `main` | `proj`（蓝实线：当前 k 维投影 P_k f） |
 * | 4 | bars | `coef` | 前 k 个系数 `c_0..c_{k−1}`（`i ≥ k` 的位置**不画**） |
 * | 5 | line | `err` | `error_target`（灰虚线：f − P_n f） |
 * | 6 | line | `err` | `error`（紫实线：f − P_k f） |
 * | 7 | line | `err` | `error_target − error` = P_k f − P_n f（红实线，仅 0<k<n 时） |
 *
 * 角标槽位（固定，见 `common.ts::SLOT`）：0 来源 / 1 主轴固定 / 2 副轴自适应 /
 * 3 κu 临界 / 4 状态（κu 是否越过 1）/ 5 前端预览（仅 `backend+preview`）/
 * 6 溢出 / 7 降级说明。
 *
 * 这个 viz 最容易骗人的两处，以及它们在图上的防法
 * ------------------------------------------------
 * 1. **「残差小 = 解对了」**（本章最贵的坑）。单项式基 n=24 时 κ(G)u≈21.7>1、系数已全是噪声，
 *    而正规方程残差只有 1.0e-11——只看残差会得出"投影没问题"。防法：`err` 面板的数值块把
 *    **残差**、**κ(G)**、**κ(G)u** 与**投影偏差 max|P_k f − P_n f|**（k<n 时可度量）排在同一屏，
 *    并写明判据是 κu 而不是残差；κu>1 时另挂状态角标。k=n 时偏差恒为 0（同一个多项式），
 *    这时**明说**「跨基对照偏差（1.17e-3）不在这一帧里」——不拿 0 冒充实测对照。
 * 2. **「换个基结果就不同」**。三个基张成同一个子空间，投影作为函数是同一个（实测 Legendre 与
 *    Chebyshev 的最大差 ≤2.2e-15）；变的是系数与条件数。防法：`coef` 面板的柱状图 + 标题里的
 *    `basis_label` 原文 + 状态角标同时给 κ(G)。单项式基的系数会到 1e5 量级（大数相消），
 *    而 Legendre 只有 0.39——柱高本身就是证据。
 *
 * 几条刻意的决定
 * --------------
 * 1. **`coef` 是补零到长度 n 的**（后端 `np.zeros(n)` 后填前 k 项）。把 `i ≥ k` 的零画成柱子
 *    等于宣称"系数是 0"；这里只画 `i < k`，并在面板上写明补零这件事。k=0 时**不画** bars 层。
 * 2. **`gram` 数组不画**。原因有二：(a) manifest 只声明了 main/coef/err 三个轴，κ(G) 的信息
 *    已在 meta 里（数值 + 角标），再加一个热力图面板会挤掉误差/系数面板；(b) 单项式基的
 *    Gram 元素只跨两个数量级（Hilbert 型 1/(i+j+1)），画出来会给人"矩阵很温和"的错觉，
 *    与 κ=1e17 并列时必须配大段注解，反而更容易误导。条件数用**数值**说，不用元素的深浅说。
 * 3. **没有竖着的临界线**：本章的临界值是**参数 n** 的阈值（n≈23 处 κ(G) 越过 1/u），
 *    而本页没有任何以 n 为横轴的坐标轴（coef 的横轴是系数编号 i，也不是 n）。硬把线画到
 *    x 轴上就是编造几何。所以临界以 `criticalBadge` + 文本行出现，并写明"为什么不画线"。
 * 4. **时间轴是"已展开的基函数个数 k"**，不是时间：后端已把 t 夹到 [0, n]，模型这边以
 *    `meta.step` 为准，并在请求值与实到值不同时明说（`common.ts::currentStep`）。
 * 5. **差值只在显示空间里做**：`error_target − error` 是同一坐标下两条帧曲线的差（P_k f − P_n f），
 *    不引入任何新物理量（P8）；它的最大值也只用来说明"偏差有多大"。
 */

import { niceLimits, niceTicks } from "../../shared/src/axes.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import {
  applyOverflow,
  canvasSize,
  clamp,
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
  barsLayer,
  criticalBadge,
  lineLayer,
  makeAxis,
  previewBadge,
  sourceBadge,
} from "../../shared/src/scene.js";
import {
  CHAPTER,
  SLOT,
  axisTitle,
  axisYlabel,
  currentStep,
  difference,
  fmt,
  maxAbs,
  minMaxOfAll,
  noteOf,
  sci,
  specTitle,
  textBlock,
  unitBudget,
  withZero,
} from "./common.js";

const VIZ = "projection";
const TITLE = specTitle(VIZ, "函数空间投影：把函数画成向量");
/** manifest 里 `n.critical = [22, 23, 24]`；帧 `critical.n` 缺失时的回落值（κ(G)u 越过 1 在 n≈23）。 */
const N_CRITICAL_FALLBACK = 23;

export const projectionModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需数组（缺一个就抛 ModelInputError：画不出来就不画半张图）——
  const x = requireArray(frame, "x");
  const f = requireArray(frame, "f");
  const proj = requireArray(frame, "proj");
  const projTarget = requireArray(frame, "proj_target");
  const error = requireArray(frame, "error");
  const errorTarget = requireArray(frame, "error_target");
  const coef = requireArray(frame, "coef");
  requireSameLength(x, f, "帧数组 x 与 f");
  requireSameLength(x, proj, "帧数组 x 与 proj");
  requireSameLength(x, projTarget, "帧数组 x 与 proj_target");
  requireSameLength(x, error, "帧数组 x 与 error");
  requireSameLength(x, errorTarget, "帧数组 x 与 error_target");
  const pitfalls = requirePitfalls(frame);

  // —— 可选/降级字段（只影响标注：缺了就明说，不猜 0）——
  const degraded: string[] = [];
  const nTerms = metaNumber(frame, "n_terms");
  const n = nTerms !== undefined ? Math.max(1, Math.round(nTerms)) : coef.length;
  if (nTerms === undefined) degraded.push(`帧 meta 缺 n_terms：按 coef 长度取 n=${n}（图像自洽，但不是帧说的）`);

  const { step, notes: stepNotes } = currentStep(frame, opts, { lo: 0, hi: n, label: "已展开项数 $k$" });
  const k = clamp(step ?? 0, 0, n);
  const kMeta = clamp(Math.round(metaNumber(frame, "n_current") ?? k), 0, n);
  if (metaNumber(frame, "n_current") === undefined) degraded.push("帧 meta 缺 n_current：当前项数按 meta.step 显示");
  if (kMeta !== k) degraded.push(`帧 meta.n_current=${kMeta} 与 meta.step=${k} 不一致：以 step 为准，不一致已点明`);

  const gramCond = metaNumber(frame, "gram_cond");
  const gramCondU = metaNumber(frame, "gram_cond_times_u");
  const normalResidual = metaNumber(frame, "normal_residual");
  const l2Current = metaNumber(frame, "l2_error_current");
  const l2Target = metaNumber(frame, "l2_error_target");
  const linfCurrent = metaNumber(frame, "linf_error_current");
  const linfTarget = metaNumber(frame, "linf_error_target");
  const exceeds = frame.meta["condition_exceeds_roundoff"];
  const errorFloor = metaNumber(frame, "error_floor");
  for (const [key, label] of [
    ["gram_cond", "κ(G)"],
    ["normal_residual", "正规方程残差"],
    ["l2_error_current", "L² 误差（当前）"],
    ["linf_error_target", "L∞ 误差（目标）"],
  ] as const) {
    if (metaNumber(frame, key) === undefined) degraded.push(`帧 meta 缺 ${key}（${label}）：该数值留「?」，不猜`);
  }
  if (typeof exceeds !== "boolean") degraded.push("帧 meta 缺 condition_exceeds_roundoff：不作「κu 是否越过 1」的结论");

  const basis = stringOf(frame, "basis");
  const basisLabel = stringOf(frame, "basis_label");
  const funcName = stringOf(frame, "func");
  const note = noteOf(frame, "note");
  const singularNote = noteOf(frame, "singular_note");
  const isMonomial = basis === "monomial";
  const nCritical = (() => {
    const raw = frame.critical["n"];
    if (typeof raw === "number" && Number.isFinite(raw)) return raw;
    if (Array.isArray(raw) && raw.length > 0) return raw[0] as number;
    return N_CRITICAL_FALLBACK;
  })();

  // —— 面板排版（纯几何：与帧无关，动画时面板不跳）——
  // `stackedPanels` 给出「主轴 + 副轴」的上下两块；下排再横着切成 coef | err 两栏。
  // gap=50：主面板的 x 刻度带（bottom+3..+18）与轴名带（bottom+16..+28）走完之后，
  // 下排两个面板的标题带（top−18..top−2）才开始，两套装饰不叠。
  const base = stackedPanels(width, height, { mainShare: 0.48, gap: 50 });
  const colGap = 62; // 右面板的 y 刻度带（41px）+ ylabel 带（56px）都在这个缝里
  const rowY = base.error[1];
  const rowH = base.error[3];
  const coefW = Math.max(80, Math.round((base.contentWidth - colGap) * 0.5));
  const coefRect: [number, number, number, number] = [base.left, rowY, coefW, rowH];
  const errRect: [number, number, number, number] = [
    base.left + coefW + colGap,
    rowY,
    Math.max(80, base.contentWidth - coefW - colGap),
    rowH,
  ];

  // —— 场景外壳 ——
  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: metaNumber(frame, "t") ?? opts.t,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  // —— 主轴：f 与两个投影 ——
  const xRange = minMaxOfAll([x]) ?? [0, 1];
  const xlim = niceLimits(xRange[0], xRange[1], 7, { pad: 0.02 });
  const yMain = minMaxOfAll([f, proj, projTarget]) ?? [0, 1];
  const ylimMain = niceLimits(yMain[0], yMain[1], 6, { pad: 0.08 });
  const mainAxis = makeAxis("main", base.main, {
    xlim,
    ylim: ylimMain,
    xlabel: "$x$",
    ylabel: axisYlabel(VIZ, "main", "y"),
    title: axisTitle(VIZ, "main", "f 与两种投影"),
    autoscaled: false,
    xticks: niceTicks(xlim[0], xlim[1], 7),
    yticks: niceTicks(ylimMain[0], ylimMain[1], 5),
  });
  scene.axes.push(mainAxis);
  scene.layers.push(
    lineLayer("main", segmentedSeries(x, f, pal.exact, { width: 1.5, label: "$f$（被逼近的函数）" })),
    lineLayer("main", segmentedSeries(x, projTarget, pal.reference, { width: 1.5, dash: [6, 4], label: `目标 ${n} 维投影 P_n f` })),
    lineLayer("main", segmentedSeries(x, proj, pal.curves[0], { width: 2, label: `当前 ${k} 维投影 P_k f` })),
  );

  // —— 系数面板：只画已展开的前 k 个（i ≥ k 是后端补的零，不是"系数为 0"）——
  const coefShown = clamp(k, 0, coef.length);
  const coefIndex = new Float64Array(coefShown);
  const coefValue = new Float64Array(coefShown);
  for (let i = 0; i < coefShown; i += 1) {
    coefIndex[i] = i;
    coefValue[i] = coef[i] as number;
  }
  const coefRange = withZero(minMaxOfAll([coefValue]));
  const coefSpan = coefRange[1] - coefRange[0] || 1;
  // 顶部多留 35%：面板左上角要放一行说明文字，柱子不上去（同时仍在 ylim 之内，不会误报溢出）。
  const coefYlim = niceLimits(coefRange[0], coefRange[1] + 0.35 * coefSpan, 4, { pad: 0.06 });
  const coefXlim: [number, number] = [-0.5, Math.max(1.5, coef.length - 0.5)];
  scene.axes.push(
    makeAxis("coef", coefRect, {
      xlim: coefXlim,
      ylim: coefYlim,
      xlabel: "基函数编号 $i$",
      ylabel: axisYlabel(VIZ, "coef", "c_k"),
      title: axisTitle(VIZ, "coef", "投影系数"),
      autoscaled: true,
      xticks: niceTicks(coefXlim[0], coefXlim[1], 8).filter((v) => Number.isInteger(v)),
      yticks: niceTicks(coefYlim[0], coefYlim[1], 4),
    }),
  );
  if (coefShown > 0) {
    scene.layers.push(
      barsLayer("coef", coefIndex, coefValue, {
        width: 0.7,
        color: pal.curves[0],
        baseline: 0,
        label: `前 ${coefShown} 个系数`,
      }),
    );
  }

  // —— 误差面板：f − P_k f、f − P_n f，以及两者之差（= P_k f − P_n f，仅 0<k<n 可度量）——
  const deviation = difference(errorTarget, error);
  const showDeviation = k > 0 && k < n;
  const devMax = showDeviation ? maxAbs(deviation) : undefined;
  const errArrays = showDeviation ? [error, errorTarget, deviation] : [error, errorTarget];
  const yErr = minMaxOfAll(errArrays) ?? [0, 1];
  const ylimErr = niceLimits(yErr[0], yErr[1], 5, { pad: 0.12 });
  scene.axes.push(
    makeAxis("err", errRect, {
      xlim,
      ylim: ylimErr,
      xlabel: "$x$",
      ylabel: axisYlabel(VIZ, "err", "误差"),
      title: axisTitle(VIZ, "err", "误差 f − P_k f"),
      autoscaled: true,
      xticks: niceTicks(xlim[0], xlim[1], 6),
      yticks: niceTicks(ylimErr[0], ylimErr[1], 4),
    }),
  );
  scene.layers.push(
    lineLayer("err", segmentedSeries(x, errorTarget, pal.reference, { width: 1.4, dash: [6, 4], label: `f − P_n f（目标）` })),
    lineLayer("err", segmentedSeries(x, error, pal.residual, { width: 1.8, label: `f − P_k f（当前 ${k} 维）` })),
  );
  if (showDeviation) {
    scene.layers.push(lineLayer("err", segmentedSeries(x, deviation, pal.warn, { width: 1.4, label: "$P_k $f − P_n $f$（两者之差）" })));
  }

  // —— 文本：主面板说明 + 误差面板数值块（一行一个 text 层）——
  // 版面预算：主面板 872px ≈ 122 字（11 行）、coef/err 各 405px ≈ 56 字（9 行）。
  // 误差面板只有半个横轴宽，所以"必须在同一屏"的那几个数（残差 / κ(G) / κ(G)u / 偏差）
  // 压成短行；解释性的话放到主面板——文字块一律贴在面板**底边**（`anchor="bottom"`），
  // 因为自适应轴上曲线多在面板上半部走，贴底能少压住数据。
  const budgetMain = unitBudget(base.main[2]);
  const budgetErr = unitBudget(errRect[2]);
  const mainLines: string[] = [
    `k = ${k} / n = ${n}　基底：${basisLabel ?? basis ?? "?"}　f = ${funcName ?? "?"}　` +
      `时间轴 = 已展开的基函数个数 k（meta.step = ${step === undefined ? "?" : fmt(step)}）`,
    `橙实线 = f　蓝实线 = 当前 ${k} 维投影　灰虚线 = 目标 ${n} 维投影；k = n 时两条投影重合，k = 0 时当前投影是零函数`,
    note ?? "帧 strings 未给 note：这台基的性质按下面这行说明",
    `临界：manifest 阈值 n = 22/23/24（帧 critical.n = ${fmt(nCritical)}），κ(G) 在 n ≈ 23 处越过 1/u ≈ 9.0e15；` +
      `该阈值只对单项式基成立。本页没有以 n 为横轴的轴，所以不画竖线——临界只能以角标和这行文字出现`,
  ];
  mainLines.push(...stepNotes);
  scene.texts.push(...textBlock(base.left + 6, base.main[1] + 4, base.main[1] + base.main[3] - 4, mainLines, pal.note, budgetMain, "bottom"));

  const errLines: string[] = [
    `残差 ‖Gc−b‖ = ${normalResidual === undefined ? "?" : sci(normalResidual)}　` +
      `κ(G) = ${gramCond === undefined ? "?" : sci(gramCond)}　` +
      `κ(G)u = ${gramCondU === undefined ? "?" : fmt(gramCondU, 3)}${gramCondU === undefined ? "" : gramCondU > 1 ? "（>1）" : "（≤1）"}`,
    showDeviation
      ? `k < n：红线 = P_k f − P_n f，max = ${devMax === undefined ? "?" : sci(devMax)}（同屏对照）`
      : `k = n：P_k f ≡ P_n f，偏差恒为 0；跨基对照 1.17e-3 不在本帧（帧只有选中的那一个基）`,
    singularNote ?? `残差小 ≠ 解对：判据是 κ(G)u 是否越过 1，不是残差（笔记定理 6 推论）`,
    `L² 误差 当前/目标 = ${l2Current === undefined ? "?" : sci(l2Current)} / ${l2Target === undefined ? "?" : sci(l2Target)}` +
      `　L∞ = ${linfCurrent === undefined ? "?" : sci(linfCurrent)} / ${linfTarget === undefined ? "?" : sci(linfTarget)}`,
    `误差地板 u = ${errorFloor === undefined ? "?" : sci(errorFloor)}（低于它的差别没有意义）`,
  ];
  scene.texts.push(...textBlock(errRect[0] + 6, errRect[1] + 4, errRect[1] + errRect[3] - 4, errLines, pal.note, budgetErr, "bottom"));

  const coefLines = [
    `柱 = 前 ${coefShown} 个系数 c_i；i ≥ k 的 ${Math.max(0, coef.length - coefShown)} 个位置未展开（帧里补零，不是"系数为 0"）`,
    isMonomial
      ? `单项式基：系数可到 1e5 量级、正负交替——大数相消正是 κ(G) 变坏的样子`
      : `正交基：系数就是函数在这组基下的坐标，量级随阶数下降`,
  ];
  if (coefShown === 0) coefLines.push("k = 0：投影是零函数，一个系数都没有（不画空柱）");
  scene.texts.push(...textBlock(coefRect[0] + 6, coefRect[1] + 4, coefRect[1] + coefRect[3] - 4, coefLines, pal.note, unitBudget(coefRect[2]), "bottom"));

  // —— 角标 ——
  scene.badges.push(placeBadge(sourceBadge(scene.dataSource), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(
      badge("autoscale", "副轴 coef/err：自适应（随 k 变），只看形状别看绝对高度", 8, 26),
      SLOT.auxAxis,
      width,
    ),
  );
  if (gramCondU === undefined) {
    scene.badges.push(placeBadge(badge("critical", "κ(G)u 未给出（null）：不与 1 比较，也不猜"), SLOT.critical, width));
  } else if (isMonomial) {
    scene.badges.push(placeBadge(criticalBadge("κ(G)u", gramCondU, [1], { beyond: "above" }), SLOT.critical, width));
  } else {
    scene.badges.push(
      placeBadge(
        badge("critical", `正交基：κ(G)u = ${fmt(gramCondU, 3)} ≤ 1（该临界只对单项式基成立）`),
        SLOT.critical,
        width,
      ),
    );
  }
  scene.badges.push(
    placeBadge(
      badge(
        "status",
        exceeds === true
          ? "⚠ κ(G)u > 1：系数已不可信（残差仍很小）"
          : exceeds === false
            ? "κ(G)u ≤ 1：系数在舍入意义下可信"
            : "condition_exceeds_roundoff 未给出：不下结论",
      ),
      SLOT.status,
      width,
    ),
  );
  if (scene.dataSource.startsWith("backend+preview")) scene.badges.push(placeBadge(previewBadge(), SLOT.preview, width));

  const rawNonFinite =
    countNonFinite(f) + countNonFinite(proj) + countNonFinite(projTarget) + countNonFinite(error) + countNonFinite(errorTarget);
  applyOverflow(scene, rawNonFinite, { slot: SLOT.overflow, width });
  const degradedBadgeObj = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedBadgeObj) scene.badges.push(degradedBadgeObj);

  requireDrawableLayers(scene);
  return scene;
};
