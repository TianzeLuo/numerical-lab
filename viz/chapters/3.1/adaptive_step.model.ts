/**
 * `adaptive_step`（3.1 章 viz 2）的 model：每次**试步**的步长、归一化误差估计与真误差。
 *
 * 面板与图层（顺序即绘制顺序）
 * ----------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | line | `solution` | 自适应数值解（实线）+ 精确解（虚线，同一根轴上） |
 * | 2 | line | `solution` | 已接受的步点（每步一个圆点） |
 * | 3 | line | `solution` | **被拒绝的试步**的起点（一个点；"拒绝"不是错误，是判据在起作用） |
 * | 4 | line | `hseq` | 步长地板 `1e-16` 参考线（对数轴的下沿） |
 * | 5 | bars | `hseq` | 每次试步的 h（对数轴的 stem；baseline = 轴底，0 在对数轴上画不出来） |
 * | 6 | line | `errseq` | 接受阈值 ε = 1（**穿过它必被拒绝**，这不是抖动） |
 * | 7 | line | `errseq` | 归一化误差估计 ε（实线）与同一尺度下的真局部误差（虚线 + 圆点） |
 * | 8 | bars | `errseq` | 被拒绝的试步的 ε（红色竖线，与接受的分开画） |
 *
 * 时间轴语义（manifest `time.mode="scrub"`，`tMax=100`）
 * ---------------------------------------------------
 * 滑块是**试步进度**（0–100：0 = 还没开始，100 = 整段积分走完），不是时间：
 * 不同 tol 下试步数从 5 变到 3000，绝对序号当不了横轴。帧 `meta.t` 是进度坐标、
 * `meta.attempt_index` 是当前试步序号、`meta.t_left` 是**当前积分时刻**（后端如此命名），
 * 三者都在面板文字里分开写。
 *
 * 两条最容易读反的地方（图上怎么防）
 * ----------------------------------
 * 1. **被拒绝的步也花了一次函数求值**：拒绝点单独画（红点），状态角标里同时给
 *    "接受/拒绝/总求值"三个数——"步数少"不等于"便宜"。
 * 2. **ε 是归一化的**：ε=1 恰好等于容差，所以柱子穿过 1 必被拒绝；而"真局部误差"
 *    是拿同方法 6 个子步的参考解量的，它自己也有误差。两者在地板附近贴合只说明都到地板了。
 */

import { formatNumber } from "../../shared/src/axes.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import {
  metaNumber,
  optionalArray,
  palette,
  placeBadge,
  requireArray,
  requireDrawableLayers,
  requirePitfalls,
  requireSameLength,
  sceneShell,
  stackedPanels,
  stringOf,
  type SceneModel,
} from "../../shared/src/model.js";
import {
  axisModeBadge,
  badge,
  lineLayer,
  makeAxis,
  previewBadge,
  scatterLayer,
  series,
  sourceBadge,
  type LineSeries,
} from "../../shared/src/scene.js";
import { manifest } from "./manifest.gen.js";
import {
  applyOverflowNote,
  axisTitleOf,
  axisYlabelOf,
  countMask,
  finiteMinMaxOf,
  logRange,
  positiveMinMax,
  stackPanels,
  stems,
  textBlock,
  textBottom,
  textTop,
  unitBudget,
  type PanelStack,
} from "./_panel.js";

const CHAPTER = "3.1";
const VIZ = "adaptive_step";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "自适应步长序列：估计器、接受阈值与真误差";
const TIME_MAX = SPEC?.time.tMax ?? 100;

/** 固定角标槽位（见 `model.ts::BADGE_LAYOUT`）。 */
const SLOT = {
  source: 0,
  axisMode: 1,
  auxAxis: 2,
  status: 3,
  preview: 4,
  overflow: 5,
  critical: 6,
  degraded: 7,
} as const;

export const adaptiveStepModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const width = Math.max(320, Math.round(opts.width));
  const height = Math.max(240, Math.round(opts.height));

  // —— 必需：画不出来就抛 ModelInputError ——
  const tAttempt = requireArray(frame, "t_attempt");
  const hAttempt = requireArray(frame, "h_attempt");
  const errEst = requireArray(frame, "error_estimate");
  const accepted = requireArray(frame, "accepted");
  requireSameLength(hAttempt, errEst, "帧数组 h_attempt 与 error_estimate");
  requireSameLength(hAttempt, accepted, "帧数组 h_attempt 与 accepted");
  const pitfalls = requirePitfalls(frame);

  // —— 可选 ——
  const degraded: string[] = [];
  const tMeta = metaNumber(frame, "t");
  if (tMeta === undefined) degraded.push("帧 meta 缺 t：scene.t 改用请求 t");
  const tol = metaNumber(frame, "tol");
  if (tol === undefined) degraded.push("帧 meta 缺 tol：不显示容差（不拿参数顶替帧值）");
  const errTrue = optionalArray(frame, "error_true");
  if (errTrue === undefined) degraded.push("帧缺数组 error_true：不画真局部误差（估计器有多准就看不见了）");
  const trueClipped = optionalArray(frame, "error_true_clipped");
  const threshold = optionalArray(frame, "threshold");
  const tSol = optionalArray(frame, "t_sol");
  const ySol = optionalArray(frame, "y_sol");
  const yExact = optionalArray(frame, "y_exact_sol");
  if (tSol === undefined || ySol === undefined) degraded.push("帧缺数组 t_sol/y_sol：主轴不画解曲线");
  if (yExact === undefined) degraded.push("帧缺数组 y_exact_sol：主轴没有精确解可比");
  const attemptIndex = metaNumber(frame, "attempt_index");
  const nAttempts = metaNumber(frame, "n_attempts");
  const nAccepted = metaNumber(frame, "n_accepted");
  const nRejected = metaNumber(frame, "n_rejected");
  const nFEvals = metaNumber(frame, "n_f_evals");
  const workPerTime = metaNumber(frame, "work_per_time");
  const errOverTol = metaNumber(frame, "err_over_tol");
  const hMedian = metaNumber(frame, "h_median");
  const hMin = metaNumber(frame, "h_min");
  const hMax = metaNumber(frame, "h_max");
  const maxEps = metaNumber(frame, "max_eps");
  const errFloor = metaNumber(frame, "err_floor");
  const hStab = metaNumber(frame, "h_stability_limit");
  const ratioMedian = metaNumber(frame, "estimator_ratio_median");
  const orderProp = metaNumber(frame, "order_propagated");
  const orderEst = metaNumber(frame, "order_estimate");
  const tLeft = metaNumber(frame, "t_left");
  const stiffNote = stringOf(frame, "stiff_note");
  const extNote = stringOf(frame, "ext_note");
  if (extNote === undefined) degraded.push("帧 strings 缺 ext_note：不声明推进的是哪个解（tol 是不是精度就失去依据）");
  const timeSemantics = stringOf(frame, "time_semantics");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: tMeta ?? opts.t,
    dataSource: stringOf(frame, "data_source") ?? "backend",
    pitfalls,
  });

  // —— 排版：解曲线 / 步长序列 / 误差序列，竖直堆叠 ——
  const absolute = stackedPanels(width, height);
  const stack: PanelStack = stackPanels(absolute, [0.34, 0.31, 0.35], { minHeights: [96, 88, 96] });
  const mainRect = stack.rects[0] as [number, number, number, number];
  const hRect = stack.rects[1] as [number, number, number, number];
  const eRect = stack.rects[2] as [number, number, number, number];
  const budget = unitBudget(stack.contentWidth);

  const n = Math.min(hAttempt.length, errEst.length, accepted.length, tAttempt.length);
  const idx: number[] = [];
  for (let i = 0; i < n; i += 1) idx.push(i);
  const accIdx = idx.filter((i) => (accepted[i] as number) > 0.5);
  const rejIdx = idx.filter((i) => (accepted[i] as number) <= 0.5);

  // —— 主轴：解曲线（自适应节点 + 精确解）——
  const solRange = finiteMinMaxOf(ySol ?? hAttempt) ?? [0, 1];
  const exactRange = yExact !== undefined ? finiteMinMaxOf(yExact) : undefined;
  const ylo = Math.min(solRange[0], exactRange?.[0] ?? solRange[0]);
  const yhi = Math.max(solRange[1], exactRange?.[1] ?? solRange[1]);
  const tEnd = tSol !== undefined && tSol.length > 0 ? (tSol[tSol.length - 1] as number) : attemptLastTime(tAttempt);
  const mainAxis = makeAxis("solution", mainRect, {
    xlim: [0, Number.isFinite(tEnd) && tEnd > 0 ? tEnd : 1],
    ylim: pad(ylo, yhi, 0.08),
    xlabel: "",
    ylabel: axisYlabelOf(SPEC?.axes, "solution", "y"),
    title: axisTitleOf(SPEC?.axes, "solution", "解曲线（自适应节点）"),
    autoscaled: true,
  });

  // —— 副轴 1：步长序列（对数）——
  // 窗口取**整段积分**的 h_min/h_max（meta），不取当前帧数组的可见范围：拖动滑块时数组只包含
  // 已走过的试步，按数组定界会让窗口随动画缩放——同一个 h 在不同进度下看起来不一样高。
  const hPosArr = positiveMinMax(hAttempt);
  const hPos: [number, number] | undefined =
    hMin !== undefined && hMax !== undefined && hMin > 0 && hMax > hMin ? [hMin, hMax] : hPosArr;
  const hAxis = makeAxis("hseq", hRect, {
    xlim: indexRange(n),
    ylim: logRange(hPos?.[0] ?? 1e-16, hPos?.[1] ?? 1),
    yscale: "log",
    xlabel: "",
    ylabel: axisYlabelOf(SPEC?.axes, "hseq", "h"),
    title: axisTitleOf(SPEC?.axes, "hseq", "试步的步长 h（对数）"),
    autoscaled: false,
  });

  // —— 副轴 2：归一化误差（对数，阈值 1）——
  const errPos = positiveMinMax(errEst);
  const truePos = errTrue !== undefined ? positiveMinMax(errTrue) : undefined;
  // 同样：下沿用地板（双精度下限），上沿取**整段**最大 ε（meta.max_eps）与阈值 1 的较大者。
  const eLo = Math.min(errPos?.[0] ?? Number.POSITIVE_INFINITY, truePos?.[0] ?? Number.POSITIVE_INFINITY, errFloor ?? 1e-17, 1);
  const eHi = Math.max(errPos?.[1] ?? 1, truePos?.[1] ?? 0, maxEps ?? 0, 1);
  const eAxis = makeAxis("errseq", eRect, {
    xlim: indexRange(n),
    ylim: logRange(Number.isFinite(eLo) ? eLo : 1e-16, Number.isFinite(eHi) ? eHi : 1),
    yscale: "log",
    xlabel: "",
    ylabel: axisYlabelOf(SPEC?.axes, "errseq", "ε"),
    title: axisTitleOf(SPEC?.axes, "errseq", "归一化误差：估计 ε 与真误差（阈值 1）"),
    autoscaled: false,
  });
  scene.axes.push(mainAxis, hAxis, eAxis);

  // —— 图层 1~3：解曲线、接受点、拒绝点 ——
  if (tSol !== undefined && ySol !== undefined && tSol.length === ySol.length && tSol.length > 0) {
    const solSeries: LineSeries[] = [
      series(toArray(tSol), toArray(ySol), pal.curves[0], { width: 2, label: "自适应数值解" }),
    ];
    if (yExact !== undefined && yExact.length === tSol.length) {
      solSeries.push(series(toArray(tSol), toArray(yExact), pal.exact, { width: 1.3, dash: [6, 3], label: "精确解" }));
    }
    scene.layers.push(lineLayer("solution", solSeries));
    scene.layers.push(
      lineLayer("solution", [
        series(toArray(tSol), toArray(ySol), pal.curves[0], { width: 1, marker: "o", alpha: 0.85, label: "被接受的节点" }),
      ]),
    );
  }
  const rejX = rejIdx.map((i) => tAttempt[i] as number);
  if (rejX.length > 0) {
    const rejY = rejX.map((x) => sampleAt(tSol, ySol, x));
    const okX: number[] = [];
    const okY: number[] = [];
    rejY.forEach((v, k) => {
      if (v !== undefined) {
        okX.push(rejX[k] as number);
        okY.push(v);
      }
    });
    if (okX.length > 0) {
      scene.layers.push(
        scatterLayer("solution", okX, okY, { r: 4.2, color: pal.marker, label: "被拒绝的试步起点（也花了一次函数求值）" }),
      );
    }
  }

  // —— 图层 4~5：步长 stem（对数轴的基线必须是正数：0 在对数轴上画不出来）——
  const hBase = hAxis.ylim[0];
  const hAccepted = stems(idx, hAttempt, hBase, (_v, i) => (accepted[i] as number) > 0.5);
  const hRejected = stems(idx, hAttempt, hBase, (_v, i) => (accepted[i] as number) <= 0.5);
  const width1 = stemWidth(hAxis);
  if (hAccepted.length > 0) {
    scene.layers.push(lineLayer("hseq", hAccepted.map((s, k) => series(s.x, s.y, pal.curves[0], { width: width1, label: k === 0 ? "被接受的步长 h" : undefined }))));
  }
  if (hRejected.length > 0) {
    scene.layers.push(lineLayer("hseq", hRejected.map((s, k) => series(s.x, s.y, pal.warn, { width: width1 * 1.8, label: k === 0 ? "被拒绝的步长（立刻缩小重试）" : undefined }))));
  }
  if (hMin !== undefined && hMin > 0) {
    scene.layers.push(
      lineLayer("hseq", series(hAxis.xlim, [hMin, hMin], pal.reference, { width: 1, dash: [4, 3], label: `实测最小 h = ${formatNumber(hMin, 3)}` })),
    );
  }

  // —— 图层 6~8：阈值线、ε 与真误差、被拒绝的 ε ——
  if (threshold !== undefined && threshold.length > 0) {
    scene.layers.push(
      lineLayer("errseq", series(eAxis.xlim, [1, 1], pal.marker, { width: 1.2, dash: [5, 3], label: "接受阈值 ε = 1（越过它必被拒绝，这不是抖动）" })),
    );
  }
  scene.layers.push(lineLayer("errseq", series(idx, errEst, pal.curves[0], { width: 1.8, marker: "s", label: "归一化误差估计 ε（1 = 容差）" })));
  if (errTrue !== undefined && errTrue.length >= n && n > 0) {
    const tru = toArray(errTrue).slice(0, n);
    scene.layers.push(
      lineLayer("errseq", series(idx, tru, pal.curves[1], { width: 1.4, dash: [5, 3], marker: "o", label: "真局部误差（同一尺度，6 子步参考解）" })),
    );
  }
  const eRej = stems(rejIdx, errEst, eAxis.ylim[0]);
  if (eRej.length > 0) {
    scene.layers.push(lineLayer("errseq", eRej.map((s, k) => series(s.x, s.y, pal.warn, { width: 2.4, label: k === 0 ? "被拒绝的步（ε > 1）" : undefined }))));
  }

  // —— 面板文字 ——
  const mainLines: string[] = [];
  mainLines.push(
    `时间轴是**试步进度** 0–${formatNumber(TIME_MAX, 2)}（帧 meta.t=${fmt(tMeta)}）：拖到 k% 就画到第 ${attemptIndex !== undefined ? Math.round(attemptIndex) : "?"} 次试步。` +
      (timeSemantics !== undefined ? `帧里的语义声明：${timeSemantics}。` : ""),
  );
  mainLines.push(
    `解曲线只画到"已被接受"的最后一个节点（t=${fmt(tSol !== undefined && tSol.length > 0 ? (tSol[tSol.length - 1] as number) : undefined)}）` +
      (tLeft !== undefined ? `；帧 meta.t_left=${formatNumber(tLeft, 4)} 是当前的积分时刻（后端就这么命名，不是"剩余时间"）` : "") +
      "——拒绝点画在曲线上是红点，它们是**被判据挡下**的试步，不是解的一部分。",
  );
  scene.texts.push(...textBlock(stack.left + 6, textTop(mainRect), textBottom(mainRect), mainLines, pal.muted, budget, "bottom"));

  const hLines: string[] = [];
  hLines.push(
    `每次试步一根竖线（h 对数）：接受 ${accIdx.length} 次、拒绝 ${rejIdx.length} 次（帧 meta.n_rejected=${fmt(nRejected)}）。` +
      (hMedian !== undefined ? `步长中位数 ${formatNumber(hMedian, 4)}。` : ""),
  );
  hLines.push(
    hStab !== undefined && hStab > 0
      ? `刚性算例：稳定极限 h*=${formatNumber(hStab, 4)}（|R(x*)|=1 除以 |λ|）。` +
          (hMedian !== undefined ? `实测步长中位数是它的 ${formatNumber(hMedian / hStab, 3)} 倍——` : "") +
          "步长被**稳定**钉住，不是被精度要求放开。"
      : stiffNote ?? "非刚性算例：步长由精度要求决定。",
  );
  if (workPerTime !== undefined || nFEvals !== undefined) {
    hLines.push(
      `代价记账：总函数求值 ${fmt(nFEvals)} 次` +
        (workPerTime !== undefined ? `、每单位时间 ${formatNumber(workPerTime, 4)} 次（被拒绝的步也占一次求值）` : "（被拒绝的步也占一次求值）") +
        "——「步数少」不等于「便宜」。",
    );
  }
  scene.texts.push(...textBlock(stack.left + 6, textTop(hRect), textBottom(hRect), hLines, pal.muted, budget, "bottom"));

  const eLines: string[] = [];
  eLines.push(
    `ε 是**归一化**的：ε=1 恰好等于容差${tol !== undefined ? `（tol=${formatNumber(tol, 3)}）` : ""}，所以柱子穿过 1 必被拒绝。`,
  );
  if (errOverTol !== undefined) {
    eLines.push(
      `本帧整段实到最大误差 / 请求容差 = ${formatNumber(errOverTol, 4)}（` +
        (errOverTol <= 3 ? "开局部外推时 ~1 量级；" : "大于 1 说明容差没控制住这个量；") +
        `推进阶 p̂=${fmt(orderProp)}、估计阶 ${fmt(orderEst)}）。`,
    );
  }
  if (ratioMedian !== undefined) {
    eLines.push(
      `估计/真误差的中位数 = ${formatNumber(ratioMedian, 4)}（只有真误差为 0 的步被筛掉）。` +
        `这**不是**"估计器说谎"：估计量 ~O(h^{p̂+1}) 估的是两个解之差，真误差 ~O(h^p)，比值 ~1/h；` +
        `越细越悲观 = 白花的工作量。`,
    );
  }
  if (trueClipped !== undefined) {
    const clipN = countMask(trueClipped);
    if (clipN > 0) {
      eLines.push(`⚠ 有 ${clipN} 个真误差样本超过显示上限（帧用占位值 + error_true_clipped 掩码标出），那一段的真值不在图上。`);
    }
  }
  eLines.push(
    `真局部误差是拿同方法 6 个子步的参考解量的，**它自己也有误差**；两者在地板附近贴合只说明都到地板了，不说明估计器准。`,
  );
  scene.texts.push(...textBlock(stack.left + 6, textTop(eRect), textBottom(eRect), eLines, pal.muted, budget, "bottom"));

  // —— 角标 ——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  if (scene.dataSource.startsWith("backend+preview")) {
    scene.badges.push(placeBadge(previewBadge("拖动时按显示空间插值"), SLOT.preview, width));
  }
  scene.badges.push(placeBadge(axisModeBadge(true), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(badge("autoscale", "步长轴/误差轴：固定对数窗口（阈值 1 与地板线始终可见）；主轴自适应", 8, 8), SLOT.auxAxis, width),
  );
  const statusBits: string[] = [];
  statusBits.push(
    `${stringOf(frame, "method") ?? "嵌入对"}@${stringOf(frame, "problem") ?? "本算例"}　tol=${fmt(tol)}` +
      (extNote !== undefined ? `　${metaNumber(frame, "local_extrapolation") !== undefined && metaNumber(frame, "local_extrapolation")! > 0.5 ? "开外推" : "关外推"}` : ""),
  );
  statusBits.push(
    `试步 ${fmt(nAttempts)}：接受 ${fmt(nAccepted)} / 拒绝 ${fmt(nRejected)}　求值 ${fmt(nFEvals)}`,
  );
  if (errOverTol !== undefined) statusBits.push(`实到/请求 = ${formatNumber(errOverTol, 3)}`);
  scene.badges.push(placeBadge(badge("status", statusBits.join("　"), 8, 8), SLOT.status, width));

  requireDrawableLayers(scene);
  const rawNonFinite =
    countNonFinite(tAttempt) + countNonFinite(hAttempt) + countNonFinite(errEst) + countNonFinite(accepted) + (errTrue !== undefined ? countNonFinite(errTrue) : 0);
  const extras: string[] = [];
  if (rejIdx.length > 0) extras.push(`${rejIdx.length} 次被拒绝的试步（也各占一次函数求值）`);
  if (trueClipped !== undefined && countMask(trueClipped) > 0) extras.push(`${countMask(trueClipped)} 个真误差样本按显示上限占位（见掩码数组）`);
  const eZero = countZero(errEst);
  if (eZero > 0) extras.push(`${eZero} 个 ε 恰好为 0（对数轴上未绘制，与"未计算"不同）`);
  applyOverflowNote(scene, extras.join("；"), rawNonFinite, { slot: SLOT.overflow, width });

  const deg = degradedBadgeLocal(degraded, SLOT.degraded, width);
  if (deg !== undefined) scene.badges.push(deg);
  return scene;
};

// --------------------------------------------------------------------------- //
// 局部工具
// --------------------------------------------------------------------------- //

function toArray(a: ArrayLike<number>): number[] {
  const out = new Array<number>(a.length);
  for (let i = 0; i < a.length; i += 1) out[i] = a[i] as number;
  return out;
}

/** 序号轴的范围（两端各留半个槽位，柱子不会压在轴线上）。 */
function indexRange(n: number): [number, number] {
  return n <= 1 ? [-0.5, 1.5] : [-0.5, n - 0.5];
}

/** stem 的线宽（像素量级；渲染层按 points 画，与数据单位无关）。 */
function stemWidth(axis: { rect: [number, number, number, number] }): number {
  return axis.rect[2] > 420 ? 1.1 : 0.9;
}

/** 线性轴 y 范围两端留余量（退化时给 ±1）。 */
function pad(lo: number, hi: number, frac: number): [number, number] {
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [0, 1];
  if (hi <= lo) {
    const d = Math.max(1e-12, Math.abs(lo) * 0.1);
    return [lo - d, hi + d];
  }
  const p = frac * (hi - lo);
  return [lo - p, hi + p];
}

/** 试步序列里最后一个时刻（主轴 x 右端用）。 */
function attemptLastTime(tAttempt: ArrayLike<number>): number {
  for (let i = tAttempt.length - 1; i >= 0; i -= 1) {
    const v = tAttempt[i] as number;
    if (Number.isFinite(v)) return v;
  }
  return 1;
}

/** 在解曲线上插值取样（只做显示几何；区间外不外推）。 */
function sampleAt(xs: ArrayLike<number> | undefined, ys: ArrayLike<number> | undefined, x: number): number | undefined {
  if (xs === undefined || ys === undefined) return undefined;
  const n = Math.min(xs.length, ys.length);
  for (let i = 0; i + 1 < n; i += 1) {
    const a = xs[i] as number;
    const b = xs[i + 1] as number;
    if (!Number.isFinite(a) || !Number.isFinite(b)) continue;
    const lo = Math.min(a, b);
    const hi = Math.max(a, b);
    if (x < lo || x > hi) continue;
    const ya = ys[i] as number;
    const yb = ys[i + 1] as number;
    if (!Number.isFinite(ya) || !Number.isFinite(yb)) return undefined;
    if (a === b) return ya;
    return ya + ((x - a) / (b - a)) * (yb - ya);
  }
  return undefined;
}

/** 恰好为 0 的样本个数（对数轴上画不出，与"非有限"含义不同）。 */
function countZero(a: ArrayLike<number>): number {
  let k = 0;
  for (let i = 0; i < a.length; i += 1) if (a[i] === 0) k += 1;
  return k;
}

/** 数值的紧凑显示（undefined 显示为「—」而不是 0）。 */
function fmt(v: number | undefined): string {
  return v === undefined || !Number.isFinite(v) ? "—" : Number.isInteger(v) ? String(v) : formatNumber(v, 4);
}

/** 降级角标（槽位固定在本章的 SLOT.degraded）。 */
function degradedBadgeLocal(items: readonly string[], slot: number, width: number) {
  if (items.length === 0) return undefined;
  const head = items.slice(0, 2).join("；");
  const rest = items.length > 2 ? `；等 ${items.length} 项` : "";
  return placeBadge(badge("status", `⚠ 降级：${head}${rest}`, 8, 8), slot, width);
}
