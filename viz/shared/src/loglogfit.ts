/**
 * 双对数收敛阶拟合（本模块与 Python `numcore/verify/order.py` **同算法、同约定**，
 * 是门禁 G7 对拍清单里"显示辅助"一侧的实现）。
 *
 * 数学定义
 * --------
 * 若误差满足 `e(h) = C·h^p + o(h^p)`，则 `log e = log C + p·log h` 是直线，斜率即阶 `p`：
 *
 * * **全局拟合**（{@link fitOrder}）：对（渐近区的）点做最小二乘；
 * * **局部阶**（{@link consecutiveOrders}）：`p_i = log(e_i/e_{i+1}) / log(h_i/h_{i+1})`；
 * * **窗口诊断**（{@link diagnoseOrder}）：只看证据、不自动挑窗口——自动挑窗口会在
 *   "舍入地板"与"预渐近区"两种污染下悄悄给出一个漂亮但错误的斜率；
 * * **理论参考线**（{@link referenceLine}）：`e = e0·(h/h0)^p_theory`，满足
 *   `docs/CHAPTERS.md` §2 防误导规则 2（实测斜率与理论斜率必须同图）。
 *
 * 约定与 Python 的逐字段对应
 * --------------------------
 * | Python | 本模块 | 说明 |
 * |---|---|---|
 * | `slope` | `slope` | `log(h)`、`log(err)` 上的最小二乘斜率（用自然对数，与 `np.log` 一致） |
 * | `intercept` | `intercept` | `log C`（自然对数底） |
 * | `r2` | `r2` | `1 − SSres/SStot`；`SStot = 0` 时取 `1` |
 * | `stderr` | `stderr` | `sqrt(SSres / max(n−2,1) / Sxx)` |
 * | `n_used` / `n_dropped` | `nUsed` / `nDropped` | 有效点数 / 被丢弃点数（含 `skip`、`tail` 丢掉的） |
 * | `h`、`err` | — | TS 侧不随拟合结果返回数组；调用方自己按同一掩码切片 |
 *
 * 入参约定：`h` 按**递减**给出（从最粗网格到最细网格），与 Python 一致；本模块不排序，
 * 只按给定顺序做"丢前 `skip` 个 / 留最后 `tail` 个"。
 *
 * 全部为纯函数，无 IO / DOM / 第三方依赖。
 */

/** 双对数最小二乘的拟合结果（字段与 Python `OrderFit` 对齐）。 */
export interface OrderFit {
  /** 拟合斜率（实测收敛阶）。 */
  slope: number;
  /** 截距 `log C`（自然对数底，与 `np.log` 口径一致）。 */
  intercept: number;
  /** 决定系数 `1 − SSres/SStot`（`SStot = 0` 时取 1）。 */
  r2: number;
  /** 斜率标准误 `sqrt(SSres / max(n−2,1) / Sxx)`。 */
  stderr: number;
  /** 参与拟合的点数。 */
  nUsed: number;
  /** 被丢弃的点数（`skip`/`tail`/非正/非有限）。 */
  nDropped: number;
}

/** 拟合选项。 */
export interface FitOptions {
  /** 丢弃最前面的 `skip` 个点（尚未进入渐近区的大步长）。缺省 0。 */
  skip?: number;
  /** 只用（`skip` 之后的）最后 `tail` 个点；缺省不截断。 */
  tail?: number;
  /** 最少有效点数（缺省 3，且至少 2）。 */
  minPoints?: number;
}

/** 窗口诊断选项。 */
export interface DiagnoseOptions {
  /** 局部阶"彼此一致"的容差（缺省 0.1，与 A 类断言的 ±0.1 同口径）。 */
  tol?: number;
  /** 判定"触舍入地板"的尾部阶阈值（缺省 0.25）。 */
  floorP?: number;
  /** "一致段"被采纳为建议所需的最短长度（缺省 2）。 */
  minRun?: number;
}

/**
 * 幂律窗口诊断结果（**诊断**，不是自动决策）。
 *
 * `suggestSkip` / `suggestTail` 的语义与 Python 一致：`suggestSkip` 是"局部阶一致段"的
 * 起始**点**序号，`suggestTail` 是从该点起连续取的**点数**（因为第 `i` 个局部阶由点
 * `i`、`i+1` 给出，长度 `L` 的一致段对应 `L+1` 个点，且被 `h.length − suggestSkip` 截顶）。
 * 直接用于 `fitOrder(h, err, { skip: d.suggestSkip, tail: d.suggestTail })`。
 */
export interface WindowDiagnosis {
  /** 相邻点局部阶（长度 `n−1`）。 */
  localOrders: Float64Array;
  /** 前 1/3 局部阶的中位数（大 h 端）。 */
  headP: number;
  /** 后 1/3 局部阶的中位数（小 h 端，即渐近方向）。 */
  tailP: number;
  /** 尾部被舍入地板淹没：`tailP < floorP && headP > tailP + tol`。 */
  floored: boolean;
  /** 局部阶整体不服从单一幂律：`|headP − tailP| > tol && !floored`。 */
  noisy: boolean;
  /** 建议丢弃的前导点数。 */
  suggestSkip: number;
  /** 建议保留的点数。 */
  suggestTail: number;
}

/** 形状校验：长度一致且非空（对应 Python `_sanitize`）。 */
function sanitize(h: ArrayLike<number>, err: ArrayLike<number>): number {
  const n = h.length;
  if (n !== err.length) {
    throw new Error(`h 与 err 长度不一致：${n} vs ${err.length}`);
  }
  if (n === 0) {
    throw new Error("h/err 为空，无法拟合阶数");
  }
  return n;
}

/** 非负整数选项。 */
function nonNegativeInt(v: number | undefined, fallback: number): number {
  if (v === undefined) return fallback;
  if (!Number.isFinite(v)) throw new RangeError(`选项必须是有限数，收到 ${String(v)}`);
  return Math.max(0, Math.floor(v));
}

/**
 * 对 `err ≈ C·h^slope` 做双对数最小二乘（与 Python `fit_order` 同算法）。
 *
 * 步骤严格对齐 Python：
 * 1. 丢弃最前面的 `skip` 个点；
 * 2. 再用（`skip` 之后的）最后 `tail` 个点；
 * 3. 再丢弃 `err ≤ 0`、`h ≤ 0` 或非有限的点（`nDropped` 记录**所有**被丢弃的点数）；
 * 4. 在 `(log h, log err)` 上最小二乘，`r2 = 1 − SSres/SStot`（`SStot = 0` 取 1），
 *    `stderr = sqrt(SSres / max(n−2,1) / Sxx)`。
 *
 * 抛错条件（与 Python 相同的三种）：形状不一致、有效点少于 `max(minPoints, 2)`、
 * 所有 `h` 相同（`Sxx = 0`）。
 */
export function fitOrder(h: ArrayLike<number>, err: ArrayLike<number>, opts: FitOptions = {}): OrderFit {
  const n0 = sanitize(h, err);
  const skip = nonNegativeInt(opts.skip, 0);
  const tail = opts.tail === undefined ? undefined : nonNegativeInt(opts.tail, 0);
  const minPoints = Math.max(2, nonNegativeInt(opts.minPoints, 3));

  const keep = new Array<boolean>(n0).fill(true);
  const s = Math.min(skip, n0);
  for (let i = 0; i < s; i++) keep[i] = false;
  if (tail !== undefined) {
    const idx: number[] = [];
    for (let i = 0; i < n0; i++) if (keep[i]) idx.push(i);
    if (tail < idx.length) {
      for (let k = 0; k < idx.length - tail; k++) keep[idx[k]] = false;
    }
  }
  for (let i = 0; i < n0; i++) {
    if (!keep[i]) continue;
    const hv = h[i];
    const ev = err[i];
    if (!Number.isFinite(hv) || !Number.isFinite(ev) || !(hv > 0) || !(ev > 0)) keep[i] = false;
  }

  const xs: number[] = [];
  const ys: number[] = [];
  const usedIdx: number[] = [];
  for (let i = 0; i < n0; i++) {
    if (!keep[i]) continue;
    xs.push(Math.log(h[i]));
    ys.push(Math.log(err[i]));
    usedIdx.push(i);
  }
  const n = xs.length;
  if (n < minPoints) {
    throw new Error(
      `有效点不足：给定 ${n0} 点，过滤后仅 ${n} 点（需要 ≥${minPoints}）。` +
        ` 有效阶数=[${usedIdx.join(", ")}]`,
    );
  }

  let xbar = 0;
  let ybar = 0;
  for (let i = 0; i < n; i++) {
    xbar += xs[i];
    ybar += ys[i];
  }
  xbar /= n;
  ybar /= n;

  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - xbar;
    sxx += dx * dx;
    sxy += dx * (ys[i] - ybar);
  }
  if (sxx === 0) {
    throw new Error("所有 h 相同，无法拟合阶数");
  }
  const slope = sxy / sxx;
  const intercept = ybar - slope * xbar;

  let ssRes = 0;
  let ssTot = 0;
  for (let i = 0; i < n; i++) {
    const r = ys[i] - (intercept + slope * xs[i]);
    ssRes += r * r;
    const dy = ys[i] - ybar;
    ssTot += dy * dy;
  }
  const r2 = ssTot === 0 ? 1 : 1 - ssRes / ssTot;
  const dof = Math.max(n - 2, 1);
  const stderr = Math.sqrt(ssRes / dof / sxx);

  return { slope, intercept, r2, stderr, nUsed: n, nDropped: n0 - n };
}

/**
 * 相邻点局部阶 `p_i = log(e_i/e_{i+1}) / log(h_i/h_{i+1})`（返回长度 `n−1`）。
 *
 * 与 Python 一致：**不做**正性/有限性过滤（先算比值再取对数），`h_i = h_{i+1}` 处得到
 * `±Infinity`、退化处得到 `NaN`——由 {@link diagnoseOrder} 负责把这些值排除在证据之外。
 * 点数少于 2 → 抛错。
 */
export function consecutiveOrders(h: ArrayLike<number>, err: ArrayLike<number>): Float64Array {
  const n = sanitize(h, err);
  if (n < 2) throw new Error("至少需要 2 个点才能计算局部阶");
  const out = new Float64Array(n - 1);
  for (let i = 0; i < n - 1; i++) {
    const num = Math.log(err[i] / err[i + 1]);
    const den = Math.log(h[i] / h[i + 1]);
    out[i] = num / den;
  }
  return out;
}

/** 中位数（偶数个取中间两数平均；区间内出现 `NaN` 则返回 `NaN`，与 `np.median` 一致）。 */
function median(values: Float64Array, start: number, end: number): number {
  const seg: number[] = [];
  for (let i = start; i < end; i++) {
    if (Number.isNaN(values[i])) return NaN;
    seg.push(values[i]);
  }
  if (seg.length === 0) return NaN;
  seg.sort((a, b) => a - b);
  const m = seg.length;
  return m % 2 === 1 ? seg[(m - 1) / 2] : 0.5 * (seg[m / 2 - 1] + seg[m / 2]);
}

/**
 * 诊断误差序列的幂律质量（与 Python `diagnose_order` 同算法）。
 *
 * * `headP` / `tailP` = 前 / 后 1/3 局部阶的中位数（`k = max(1, ⌊n/3⌋)`）；
 * * `floored = tailP < floorP && headP > tailP + tol`（尾部触舍入地板）；
 * * `noisy = |headP − tailP| > tol && !floored`；
 * * `suggestSkip` / `suggestTail` = 最长"局部阶彼此落在中位数 ± tol 内"的连续段
 *   （长度 `< minRun` 时退化为整段 `(0, n)`；平手取起点更小者，与 Python 的 `>` 比较一致）；
 * * 任何局部阶都非有限（或 `n = 0`）→ `headP = tailP = NaN`、`noisy = true`、
 *   `suggestSkip = 0`、`suggestTail = h.length`；此时 `floored = false`（NaN 比较恒为 false）。
 *
 * 本函数只给证据，不替测试作者做决定——这正是 Python 侧 `WindowDiagnosis` 的设计立场。
 */
export function diagnoseOrder(
  h: ArrayLike<number>,
  err: ArrayLike<number>,
  opts: DiagnoseOptions = {},
): WindowDiagnosis {
  const n0 = sanitize(h, err);
  const tol = opts.tol === undefined ? 0.1 : opts.tol;
  const floorP = opts.floorP === undefined ? 0.25 : opts.floorP;
  const minRun = Math.max(1, nonNegativeInt(opts.minRun, 2));
  const orders = consecutiveOrders(h, err);
  const n = orders.length;

  let anyFinite = false;
  for (let i = 0; i < n; i++) {
    if (Number.isFinite(orders[i])) {
      anyFinite = true;
      break;
    }
  }
  if (n === 0 || !anyFinite) {
    return {
      localOrders: orders,
      headP: NaN,
      tailP: NaN,
      floored: false,
      noisy: true,
      suggestSkip: 0,
      suggestTail: n0,
    };
  }

  const k = Math.max(1, Math.floor(n / 3));
  const headP = median(orders, 0, k);
  const tailP = median(orders, n - k, n);

  // 最长一致段（平手保留起点更小者：只在严格更长时替换）
  let bestStart = 0;
  let bestLen = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j <= n; j++) {
      let finite = true;
      for (let t = i; t < j; t++) {
        if (!Number.isFinite(orders[t])) {
          finite = false;
          break;
        }
      }
      if (!finite) break;
      const med = median(orders, i, j);
      let ok = true;
      for (let t = i; t < j; t++) {
        if (!(Math.abs(orders[t] - med) <= tol)) {
          ok = false;
          break;
        }
      }
      if (ok && j - i > bestLen) {
        bestStart = i;
        bestLen = j - i;
      }
    }
  }
  let start = bestStart;
  let length = bestLen;
  if (length < minRun) {
    start = 0;
    length = n;
  }

  const floored = tailP < floorP && headP > tailP + tol;
  const noisy = Math.abs(headP - tailP) > tol && !floored;
  return {
    localOrders: orders,
    headP,
    tailP,
    floored,
    noisy,
    suggestSkip: start,
    suggestTail: Math.min(length + 1, n0 - start),
  };
}

/**
 * 构造理论斜率参考线 `e = e0·(h/h0)^p`（防误导规则 2：必须与实测拟合线同图）。
 *
 * `anchor` 给出参考线穿过的点 `(h0, e0)`；缺省 `h0 = min(h)`、`e0 = 1`
 * （与 Python `reference_line(scale=1.0)` 一致）。
 *
 * 与 Python 的两点差异（有意为之，已在汇报中说明）：空 `h` 返回空数组而不抛错
 * （渲染层不该因为一条参考线崩掉）；其余逐元素语义完全一致——含 `h ≤ 0` 时照原样
 * 计算（可能得到 `NaN`/`Infinity`），不做过滤。
 */
export function referenceLine(
  h: ArrayLike<number>,
  theorySlope: number,
  anchor?: { h0: number; e0: number },
): Float64Array {
  const n = h.length;
  const out = new Float64Array(n);
  if (n === 0) return out;
  let h0: number;
  let e0: number;
  if (anchor === undefined) {
    // 与 np.min 一致：出现 NaN 即整体 NaN
    let mn = Infinity;
    for (let i = 0; i < n; i++) {
      const v = h[i];
      if (Number.isNaN(v)) {
        mn = NaN;
        break;
      }
      if (v < mn) mn = v;
    }
    h0 = mn;
    e0 = 1;
  } else {
    h0 = anchor.h0;
    e0 = anchor.e0;
  }
  for (let i = 0; i < n; i++) out[i] = e0 * Math.pow(h[i] / h0, theorySlope);
  return out;
}
