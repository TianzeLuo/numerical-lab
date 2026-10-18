/**
 * 坐标轴刻度、数值格式化与"数据 ↔ 像素"映射（纯函数，无 DOM / canvas / 第三方依赖）。
 *
 * 数学定义
 * --------
 * * 线性刻度步长取 ``1·10^k``、``2·10^k``、``5·10^k``（1-2-5 规则）；
 * * 对数刻度主刻度取 ``10^k``；当区间跨度不足 ``target/2`` 个十倍程时，依次补
 *   ``5·10^k``（必要时再补 ``2·10^k``）作为次刻度，使刻度始终是"好看的数"；
 * * 数据到像素是仿射映射 ``p = p0 + (v − lo)/(hi − lo)·(p1 − p0)``，
 *   对数刻度下把 ``v`` 换成 ``log10 v``。
 *
 * 覆盖口径（可机器验证）：**任何原始区间**下，1-2-5 格点 + "刻度数 ≤ 2·target"这两条
 * 硬约束决定了首末刻度未必能压住端点（例如 8 个十倍程的区间只有十几个格点可用）；本模块
 * 的选择策略是"在可行集里取覆盖最大者"（见 {@link niceTicks} 的选择口径）。要 100% 压住
 * 两端，就先过一遍 {@link niceLimits}——它会把轴范围取整到与刻度同一格点，测试里对此有断言。
 *
 * 契约与退化行为（全部写进各函数 docstring，不靠"猜"）
 * ------------------------------------------------
 * | 输入情形 | 行为 |
 * |---|---|
 * | `lo === hi`（线性） | `niceTicks` 退化为 `[lo]`；`niceLimits` 向两侧各扩 `|lo|`（`lo=0` 时扩 1）；`mapToPixel` 退化到像素中点、`mapToData` 返回 `lo` |
 * | `lo === hi`（对数，`lo>0`） | `niceTicks` 退化为 `[lo]`；`niceLimits` 扩成 `[lo/10, lo·10]` |
 * | `lo > hi` | `niceTicks`/`niceLimits` 抛 `RangeError`（区间方向是契约的一部分）；`mapToPixel`/`mapToData` 允许（等价于反向轴） |
 * | log 刻度遇到 `lo ≤ 0`/`hi ≤ 0` | `niceTicks`/`niceLimits`/`mapTo*` 抛 `RangeError`（不静默给 NaN 轴） |
 * | log 刻度遇到数据 `v ≤ 0` | `mapToPixel` 返回 `NaN`（渲染层据此跳过该点，不抛错） |
 * | `p0 === p1` | `mapToPixel`/`mapToData` 抛 `RangeError`（零长轴没有映射可言） |
 * | 非有限输入 | 抛 `RangeError`（`mapToPixel` 的数据 `v` 除外：`NaN` 进 `NaN` 出） |
 */

/** 刻度类型：线性 / 对数（以 10 为底）。 */
export type ScaleKind = "linear" | "log";

/** 缺省目标刻度数（契约要求缺省落在 6~8）。 */
const DEFAULT_TARGET = 7;
/** 1-2-5 规则的步长尾数。 */
const MANTISSAS: readonly number[] = [1, 2, 5];
/** 浮点"落在端点上"的相对判别阈值（只用于吸收 1~2 ulp 的表示误差）。 */
const EPS_REL = 1e-12;

/** 校验有限数。 */
function requireFinite(v: number, name: string): number {
  if (typeof v !== "number" || !Number.isFinite(v)) {
    throw new RangeError(`${name} 必须是有限数，收到 ${String(v)}`);
  }
  return v;
}

/** 校验目标刻度数：缺省 7，至少 2，向下取整。 */
function checkTarget(target?: number): number {
  const t = target === undefined ? DEFAULT_TARGET : requireFinite(target, "target");
  if (t < 2) throw new RangeError(`target 至少为 2，收到 ${t}`);
  return Math.floor(t);
}

/** 校验点数：非负、向下取整。 */
function checkCount(n: number): number {
  requireFinite(n, "n");
  if (n < 0) throw new RangeError(`点数必须非负，收到 ${n}`);
  return Math.floor(n);
}

/** `10^e`，在次正规/上溢边界处饱和，避免得到 0 或 Infinity。 */
function pow10(e: number): number {
  if (e > 308) return Number.MAX_VALUE;
  if (e < -323) return Number.MIN_VALUE;
  const v = Math.pow(10, e);
  if (v === 0) return Number.MIN_VALUE;
  if (!Number.isFinite(v)) return Number.MAX_VALUE;
  return v;
}

/** 最大的整数 `e` 使 `10^e ≤ x`（要求 `x > 0`）。 */
function decadeExpDown(x: number): number {
  let e = Math.floor(Math.log10(x) + EPS_REL);
  while (pow10(e) > x) e -= 1;
  while (pow10(e + 1) <= x) e += 1;
  return e;
}

/** 最小的整数 `e` 使 `10^e ≥ x`（要求 `x > 0`）。 */
function decadeExpUp(x: number): number {
  let e = Math.ceil(Math.log10(x) - EPS_REL);
  while (pow10(e) < x) e += 1;
  while (pow10(e - 1) >= x) e -= 1;
  return e;
}

/* ------------------------------------------------------------------ *
 * 均匀 / 等比序列
 * ------------------------------------------------------------------ */

/**
 * 返回 `a … b` 的 `n` 点等距序列（含两端，端点精确）。
 *
 * * `n = 0` → 空；`n = 1` → `[a]`；`n ≥ 2` → 首元素 `a`、末元素 `b`（精确赋值）；
 * * `a > b` 合法（递减序列）；
 * * `a`/`b` 非有限或 `n` 非有限/为负 → 抛 `RangeError`。
 */
export function linspace(a: number, b: number, n: number): Float64Array {
  requireFinite(a, "a");
  requireFinite(b, "b");
  const len = checkCount(n);
  const out = new Float64Array(len);
  if (len === 0) return out;
  if (len === 1) {
    out[0] = a;
    return out;
  }
  const step = (b - a) / (len - 1);
  for (let i = 0; i < len - 1; i++) out[i] = a + i * step;
  out[len - 1] = b;
  return out;
}

/**
 * 返回 `10^a … 10^b` 的 `n` 点等比序列（含两端）。
 *
 * **参数是十进制指数**（与 `numpy.logspace` 一致）：`logspace(-3, 3, 7)` 给出
 * `1e-3 … 1e3` 的 7 个整十次幂。退化行为同 {@link linspace}（`n=0` 空、`n=1` 取 `10^a`、
 * `a > b` 递减、非有限参数抛错）。
 */
export function logspace(a: number, b: number, n: number): Float64Array {
  const exps = linspace(a, b, n);
  const out = new Float64Array(exps.length);
  for (let i = 0; i < exps.length; i++) out[i] = Math.pow(10, exps[i]);
  return out;
}

/* ------------------------------------------------------------------ *
 * 线性刻度
 * ------------------------------------------------------------------ */

/** 生成"以 `step` 为公差的等差数列 ∩ [lo, hi]"；端点处只做 ≤1e-12 相对量的钳位。 */
function ticksForStep(lo: number, hi: number, step: number): number[] {
  const out: number[] = [];
  if (!(step > 0) || !Number.isFinite(step)) return out;
  const tol = EPS_REL * Math.max(Math.abs(lo), Math.abs(hi), step);
  const i0 = Math.ceil(lo / step - 1e-9);
  const i1 = Math.floor(hi / step + 1e-9);
  if (!Number.isFinite(i0) || !Number.isFinite(i1) || i1 < i0) return out;
  if (i1 - i0 > 1e6) return out; // 病态输入（步长相对区间小到失去意义）→ 视为无候选
  for (let i = i0; i <= i1; i++) {
    let v = i * step;
    // 与端点相差 ≤1e-12 相对量时钳到端点：既保证"严格在区间内"，又保住覆盖
    if (v < lo && v >= lo - tol) v = lo;
    if (v > hi && v <= hi + tol) v = hi;
    if (v < lo || v > hi) continue;
    const last = out.length > 0 ? out[out.length - 1] : Number.NEGATIVE_INFINITY;
    if (v <= last) continue; // 去重 + 保序（极端量级下 i*step 可能失去分辨力）
    out.push(v);
  }
  return out;
}

/** 1-2-5 候选步长（覆盖"步长 ≈ span/target"附近的若干数量级）。 */
function candidateSteps(span: number): number[] {
  const e0 = Math.floor(Math.log10(span)) - 2;
  const out: number[] = [];
  for (let e = e0; e <= e0 + 4; e++) {
    for (const m of MANTISSAS) out.push(m * pow10(e));
  }
  return out;
}

/** 刻度占区间的比例（首尾跨度 / 区间长度）。 */
function coverage(ticks: number[], lo: number, hi: number): number {
  if (ticks.length < 2) return 0;
  const span = hi - lo;
  if (!(span > 0)) return 1;
  return (ticks[ticks.length - 1] - ticks[0]) / span;
}

/**
 * 按 1-2-5 规则选线性步长。
 *
 * 选择口径（可机器验证）：
 * 1. 只在"刻度数 ∈ [⌈target/2⌉, 2·target]"的候选里挑；
 * 2. 优先挑覆盖 ≥ 90% 的候选，其中取刻度数最接近 `target` 的（再平手取更稀疏的步长）；
 * 3. 若没有任何候选能覆盖 ≥ 90%（1-2-5 格点 + 数量上限的硬约束，见模块头注释），
 *    退化为"覆盖最大、其次数量最接近 target"。
 */
function selectLinearStep(lo: number, hi: number, target: number): { step: number; ticks: number[] } {
  const span = hi - lo;
  const minCount = Math.max(2, Math.ceil(target / 2));
  const maxCount = Math.max(minCount, Math.floor(target * 2));
  let bestStep = 0;
  let bestTicks: number[] = [];
  let bestKey: [number, number, number] | null = null;

  for (const step of candidateSteps(span)) {
    const ticks = ticksForStep(lo, hi, step);
    const n = ticks.length;
    if (n < minCount || n > maxCount) continue;
    const cov = coverage(ticks, lo, hi);
    const key: [number, number, number] =
      cov >= 0.9 ? [0, Math.abs(n - target), n] : [1, -cov, Math.abs(n - target)];
    if (bestKey === null || key[0] < bestKey[0] ||
        (key[0] === bestKey[0] && (key[1] < bestKey[1] ||
          (key[1] === bestKey[1] && key[2] < bestKey[2])))) {
      bestKey = key;
      bestStep = step;
      bestTicks = ticks;
    }
  }
  if (bestKey !== null) return { step: bestStep, ticks: bestTicks };

  // 兜底：没有候选满足数量约束（区间极窄/极宽）→ 经典 1-2-5 取步长
  const raw = span / target;
  const e = Math.floor(Math.log10(raw));
  let step = pow10(e + 1);
  for (const m of MANTISSAS) {
    const s = m * pow10(e);
    if (s >= raw) {
      step = s;
      break;
    }
  }
  let ticks = ticksForStep(lo, hi, step);
  if (ticks.length < 2) ticks = [lo, hi];
  return { step, ticks };
}

/* ------------------------------------------------------------------ *
 * 对数刻度
 * ------------------------------------------------------------------ */

/** 对数主刻度的十倍程步进：保证刻度数 ≤ target（跨度极大时按 10^stride 取样）。 */
function decadeStride(decadeCount: number, target: number): number {
  return Math.max(1, Math.ceil((decadeCount - 1) / Math.max(1, target - 1)));
}

/** 对数刻度：[10^e]（必要时补 5×10^e），全部严格落在 [lo, hi] 内。 */
function logTicks(lo: number, hi: number, target: number): number[] {
  const minCount = Math.max(2, Math.ceil(target / 2));
  const eLo = decadeExpUp(lo);
  const eHi = decadeExpDown(hi);
  const count = eHi - eLo + 1;
  if (count >= minCount) {
    const stride = decadeStride(count, target);
    // 步进可以整体平移：挑"两端缺口最大值最小"的那个相位，避免刻度全挤在一侧
    let bestOffset = 0;
    let bestGap = Infinity;
    for (let off = 0; off < stride; off++) {
      const first = eLo + off;
      if (first > eHi) break;
      const last = eHi - ((eHi - first) % stride);
      const gap = Math.max(
        Math.log10(pow10(first) / lo),
        Math.log10(hi / pow10(last)),
      );
      if (gap < bestGap - 1e-12) {
        bestGap = gap;
        bestOffset = off;
      }
    }
    const out: number[] = [];
    for (let e = eLo + bestOffset; e <= eHi; e += stride) out.push(pow10(e));
    return out;
  }
  // 整十次幂太少（区间跨度不足若干十倍程）：补 5×10^e 次刻度，把最大间隙压到 5 倍；
  // 若连 2 个刻度都凑不出（区间窄于一个 5 倍格点），再补 2×10^e；仍不足则退化为端点。
  const build = (mantissas: readonly number[]): number[] => {
    const out: number[] = [];
    for (let e = eLo - 1; e <= eHi + 1; e++) {
      for (const m of mantissas) {
        const v = m * pow10(e);
        if (v < lo || v > hi) continue;
        const last = out.length > 0 ? out[out.length - 1] : Number.NEGATIVE_INFINITY;
        if (v > last) out.push(v);
      }
    }
    return out;
  };
  const coarse = build([1, 5]);
  if (coarse.length >= 2) return coarse;
  const fine = build([1, 2, 5]);
  if (fine.length >= 2) return fine;
  return [lo, hi]; // 极窄区间：退化为端点刻度（仍然严格落在区间内）
}

/* ------------------------------------------------------------------ *
 * 对外：刻度与区间
 * ------------------------------------------------------------------ */

/**
 * 给出 `[lo, hi]` 内的"好看"刻度（升序、严格落在区间内、无重复）。
 *
 * * 线性：1-2-5 步长，刻度是 `step` 的整数倍；
 * * 对数（`opts.log = true`）：主刻度为整十次幂；当区间跨度不足 `target/2` 个十倍程时
 *   补 `5×10^k` 次刻度（连 2 个刻度都凑不出时再补 `2×10^k`；仍然不足则退化为端点），
 *   因此对数刻度恒为 1×/2×/5×10^k，**十倍程足够时相邻比值才是 10 的幂**——这是"必要时"
 *   的唯一代价，见模块头注释；
 * * 数量目标：落在 `[⌈target/2⌉, 2·target]`；对数刻度在"十倍程数不足"或"极窄区间"时
 *   无法达标，此时以"刻度合法且尽量贴近端点"为先；
 * * `lo > hi` → `RangeError`；`lo === hi` → `[lo]`（退化）；`log` 且 `lo ≤ 0` → `RangeError`。
 *
 * @param target 目标刻度数（缺省 7，须 ≥ 2）
 * @param opts.log 是否对数刻度
 */
export function niceTicks(lo: number, hi: number, target?: number, opts: { log?: boolean } = {}): number[] {
  requireFinite(lo, "lo");
  requireFinite(hi, "hi");
  const tgt = checkTarget(target);
  if (lo > hi) throw new RangeError(`niceTicks 要求 lo ≤ hi，收到 [${lo}, ${hi}]`);
  if (opts.log) {
    if (!(lo > 0) || !(hi > 0)) {
      throw new RangeError(`对数刻度要求 lo > 0 且 hi > 0，收到 [${lo}, ${hi}]`);
    }
    if (lo === hi) return [lo];
    return logTicks(lo, hi, tgt);
  }
  if (lo === hi) return [lo];
  return selectLinearStep(lo, hi, tgt).ticks;
}

/**
 * 把 `[lo, hi]` 扩成"好看"的轴范围 `[lo', hi'] ⊇ [lo, hi]`。
 *
 * * 线性：用与 {@link niceTicks} 相同的 1-2-5 步长，把两端取整到该步长的整数倍；
 * * 对数：向上/向下取到整十次幂（`10^⌊log10 lo'⌋`、`10^⌈log10 hi'⌉`）；
 * * `pad > 0`：先在两端各留 `pad × span`（对数刻度在 log10 空间留 `pad` 倍跨度）再取整；
 * * 退化 `lo === hi`：线性扩成 `[lo − d, lo + d]`（`d = |lo|`，`lo = 0` 时 `d = 1`），
 *   对数扩成 `[lo/10, lo·10]`；因此返回值**恒满足** `lo' ≤ lo ≤ hi ≤ hi'`；
 * * `lo > hi` / log 且非正 / 非有限参数 / `pad < 0` → `RangeError`。
 *
 * 与 {@link niceTicks} 配合使用即可获得"压住两端"的刻度（覆盖 100%）。
 */
export function niceLimits(
  lo: number,
  hi: number,
  target?: number,
  opts: { log?: boolean; pad?: number } = {},
): [number, number] {
  requireFinite(lo, "lo");
  requireFinite(hi, "hi");
  const tgt = checkTarget(target);
  const pad = opts.pad === undefined ? 0 : requireFinite(opts.pad, "pad");
  if (pad < 0) throw new RangeError(`pad 必须非负，收到 ${pad}`);
  if (lo > hi) throw new RangeError(`niceLimits 要求 lo ≤ hi，收到 [${lo}, ${hi}]`);

  if (opts.log) {
    if (!(lo > 0) || !(hi > 0)) {
      throw new RangeError(`对数刻度要求 lo > 0 且 hi > 0，收到 [${lo}, ${hi}]`);
    }
    let a = lo;
    let b = hi;
    if (a === b) {
      a /= 10;
      b *= 10;
    }
    if (pad > 0) {
      const k = pad * Math.log10(b / a);
      a /= pow10(k);
      b *= pow10(k);
    }
    let ea = decadeExpDown(a);
    let eb = decadeExpUp(b);
    // 十倍程跨度极大时主刻度按 stride 取样；把端点扩到 stride 的整数倍上，
    // 这样 niceTicks 的首末刻度正好落在轴端（覆盖 100%）。最多 8 轮，只向外扩。
    const minCount = Math.max(2, Math.ceil(tgt / 2));
    for (let iter = 0; iter < 8; iter++) {
      const count = eb - ea + 1;
      if (count < minCount) break; // 走 5×10^k 次刻度分支：端点本身就会被取到
      const stride = decadeStride(count, tgt);
      const rem = (eb - ea) % stride;
      if (rem === 0) break;
      const need = stride - rem;
      ea -= Math.floor(need / 2);
      eb += Math.ceil(need / 2);
    }
    return [pow10(ea), pow10(eb)];
  }

  let a = lo;
  let b = hi;
  if (a === b) {
    const d = a === 0 ? 1 : Math.abs(a);
    a -= d;
    b += d;
  }
  if (pad > 0) {
    const d = pad * (b - a);
    a -= d;
    b += d;
  }
  // 取整到"该区间自己会选中的那个步长"的整数倍；扩大区间可能让步长变化，
  // 故迭代到不动点（最多 8 轮，只向外扩，必然收敛），保证 niceTicks 能压住两端。
  for (let iter = 0; iter < 8; iter++) {
    const { step, ticks } = selectLinearStep(a, b, tgt);
    if (coverage(ticks, a, b) >= 1 - 1e-9) break;
    const na = Math.floor(a / step + 1e-12) * step;
    const nb = Math.ceil(b / step - 1e-12) * step;
    a = Math.min(a, na);
    b = Math.max(b, nb);
  }
  return [a, b];
}

/* ------------------------------------------------------------------ *
 * 数值格式化
 * ------------------------------------------------------------------ */

/** 去掉定点字符串尾部的 0 与小数点。 */
function stripFixedZeros(s: string): string {
  if (!s.includes(".")) return s;
  return s.replace(/\.?0+$/, "");
}

/** 去掉科学计数法尾数部分的尾随 0。 */
function stripSciZeros(mant: string): string {
  if (!mant.includes(".")) return mant;
  return mant.replace(/\.?0+$/, "");
}

/**
 * 把 `v` 格式化为 `sig` 位有效数字的紧凑字符串。
 *
 * * `|v| ∈ [1e-3, 1e5]`（闭区间）用定点（如 `0.001`、`123.456`）；
 * * 其余用 `e` 记法（如 `1.23e-7`、`6.02e23`，指数不带 `+` 号）；
 * * `0` → `"0"`；`NaN` → `"NaN"`；`±Infinity` → `"∞"` / `"-∞"`；
 * * 尾随 0 一律去掉（`1.20000e-7` → `1.2e-7`）。
 */
export function formatNumber(v: number, sig = 6): string {
  const digits = Math.max(1, Math.floor(requireFinite(sig, "sig")));
  if (Number.isNaN(v)) return "NaN";
  if (v === Infinity) return "∞";
  if (v === -Infinity) return "-∞";
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e-3 && a <= 1e5) {
    const decimals = Math.min(100, Math.max(0, digits - 1 - Math.floor(Math.log10(a))));
    return stripFixedZeros(v.toFixed(decimals));
  }
  const s = v.toExponential(digits - 1);
  const [mant, exp] = s.split("e");
  return `${stripSciZeros(mant)}e${exp.startsWith("+") ? exp.slice(1) : exp}`;
}

/** 上标数字/负号映射（用于 `10⁻⁷` 这样的标签）。 */
const SUP: Record<string, string> = {
  "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
  "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹", "-": "⁻",
};

/** 把整数指数写成上标串，如 `-7` → `⁻⁷`。 */
function superscript(exp: number): string {
  return String(exp).split("").map((c) => SUP[c] ?? c).join("");
}

/**
 * 科学计数法字符串 `m×10^k`（例如 `sig = 2` 时 `1.2e-7` → `"1.2×10⁻⁷"`）。
 *
 * * 恒为 `×10^k` 形式（不像 {@link formatNumber} 那样在小量级改用定点）；
 * * `0` → `"0"`；`NaN` → `"NaN"`；`±Infinity` → `"∞"` / `"-∞"`；
 * * 尾数舍入进位到 10 时自动进一位（`9.99e-7`, `sig = 2` → `"1×10⁻⁶"`）。
 */
export function formatSci(v: number, sig = 6): string {
  const digits = Math.max(1, Math.floor(requireFinite(sig, "sig")));
  if (Number.isNaN(v)) return "NaN";
  if (v === Infinity) return "∞";
  if (v === -Infinity) return "-∞";
  if (v === 0) return "0";
  const a = Math.abs(v);
  let e = Math.floor(Math.log10(a));
  if (a < pow10(e)) e -= 1;
  if (a >= pow10(e + 1)) e += 1;
  let mant = v / pow10(e);
  let mantStr = mant.toPrecision(digits);
  if (Math.abs(Number(mantStr)) >= 10) {
    e += 1;
    mant /= 10;
    mantStr = mant.toPrecision(digits);
  }
  return `${stripSciZeros(mantStr)}×10${superscript(e)}`;
}

/**
 * 刻度标签：`opts.log` 时把整十次幂写成 `10^k`（`k = 0` → `"1"`，`k = 1` → `"10"`），
 * 把 `5·10^k` 写成 `5×10^k`，其余交给 {@link formatNumber}；线性刻度直接
 * {@link formatNumber}。
 *
 * 注意：`log` 模式下非正数据不会出现（对数轴定义域），若传入 `v ≤ 0` 则按
 * {@link formatNumber} 原样输出。
 */
export function tickLabel(v: number, opts: { log?: boolean; sig?: number } = {}): string {
  const sig = opts.sig === undefined ? 6 : Math.max(1, Math.floor(requireFinite(opts.sig, "sig")));
  if (!opts.log || !Number.isFinite(v) || v <= 0) return formatNumber(v, sig);
  const k = Math.log10(v);
  const kr = Math.round(k);
  if (Math.abs(v - pow10(kr)) <= 1e-9 * v) {
    if (kr === 0) return "1";
    if (kr === 1) return "10";
    return `10${superscript(kr)}`;
  }
  const k5 = Math.round(k - Math.log10(5));
  if (Math.abs(v - 5 * pow10(k5)) <= 1e-9 * v) {
    if (k5 === 0) return "5";
    return `5×10${superscript(k5)}`;
  }
  return formatNumber(v, sig);
}

/* ------------------------------------------------------------------ *
 * 数据 ↔ 像素
 * ------------------------------------------------------------------ */

/**
 * 数据值 → 像素坐标（仿射映射；对数刻度下对 `v`、`lo`、`hi` 取 `log10`）。
 *
 * * `p0 > p1` 合法（屏幕 y 轴向下），`lo > hi` 亦合法（相当于反向轴）；
 * * `lo === hi`：退化为像素中点 `(p0 + p1)/2`（没有可比的距离，画在中间最不误导）；
 * * `p0 === p1` → `RangeError`；log 且 `lo ≤ 0`/`hi ≤ 0` → `RangeError`；
 * * log 且 `v ≤ 0` → `NaN`（渲染层跳过该点）；`v = NaN` → `NaN`。
 */
export function mapToPixel(
  v: number,
  lo: number,
  hi: number,
  p0: number,
  p1: number,
  scale: ScaleKind = "linear",
): number {
  requireFinite(lo, "lo");
  requireFinite(hi, "hi");
  requireFinite(p0, "p0");
  requireFinite(p1, "p1");
  if (p0 === p1) throw new RangeError("像素区间退化（p0 === p1），无法映射");
  if (scale === "log") {
    if (!(lo > 0) || !(hi > 0)) {
      throw new RangeError(`对数刻度要求 lo > 0 且 hi > 0，收到 [${lo}, ${hi}]`);
    }
    if (lo === hi) return (p0 + p1) / 2;
    if (!(v > 0)) return NaN;
    const t = (Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo));
    return p0 + t * (p1 - p0);
  }
  if (lo === hi) return (p0 + p1) / 2;
  return p0 + ((v - lo) / (hi - lo)) * (p1 - p0);
}

/**
 * 像素坐标 → 数据值（{@link mapToPixel} 的逆；log 刻度下走 `10^(…)`）。
 *
 * * `p0 === p1` → `RangeError`；
 * * `lo === hi` → 返回 `lo`（退化轴只有这一个数据值）；
 * * log 且 `lo ≤ 0`/`hi ≤ 0` → `RangeError`；`p = NaN` → `NaN`。
 */
export function mapToData(
  p: number,
  lo: number,
  hi: number,
  p0: number,
  p1: number,
  scale: ScaleKind = "linear",
): number {
  requireFinite(lo, "lo");
  requireFinite(hi, "hi");
  requireFinite(p0, "p0");
  requireFinite(p1, "p1");
  if (p0 === p1) throw new RangeError("像素区间退化（p0 === p1），无法反解");
  const t = (p - p0) / (p1 - p0);
  if (scale === "log") {
    if (!(lo > 0) || !(hi > 0)) {
      throw new RangeError(`对数刻度要求 lo > 0 且 hi > 0，收到 [${lo}, ${hi}]`);
    }
    if (lo === hi) return lo;
    const a = Math.log10(lo);
    const b = Math.log10(hi);
    return Math.pow(10, a + t * (b - a));
  }
  if (lo === hi) return lo;
  return lo + t * (hi - lo);
}

/**
 * 对数轴下"下一个可表示的正数"：`v > 0` 时返回 `v` 沿 +∞ 方向的相邻浮点数，
 * 否则（`0`、负数、`-Infinity`）返回最小正次正规数 `Number.MIN_VALUE ≈ 4.94e-324`
 * （`log10(MIN_VALUE) ≈ -323.3`，有限）。用途：把"零/负"映射成可安全取对数的位置，
 * 而不必在渲染层写 `Math.max(v, 1e-300)` 这类魔法常数。
 *
 * `NaN` → `NaN`；`+Infinity` → `+Infinity`。
 */
export function nextafter_log(v: number): number {
  if (Number.isNaN(v)) return NaN;
  if (v === Infinity) return Infinity;
  if (!(v > 0)) return Number.MIN_VALUE;
  if (v === Number.MAX_VALUE) return Infinity;
  const view = new DataView(new ArrayBuffer(8));
  view.setFloat64(0, v, false);
  // 正数：位模式 +1 即"下一个更大的可表示数"（IEEE-754 数量级随位模式单调）
  const bits = view.getBigUint64(0, false) + 1n;
  view.setBigUint64(0, bits, false);
  return view.getFloat64(0, false);
}
