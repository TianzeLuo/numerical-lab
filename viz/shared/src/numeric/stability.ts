/**
 * 稳定性函数与线性多步法根模的 TS 侧对拍实现（PROTOCOLS §7 覆盖清单里的 `stability`，对应 3.2 章）。
 *
 * **本目录只服务 G7 双实现对拍，不是渲染路径。**
 * `viz/chapters/3.2/**` MUST NOT 直接 import 本目录（P8）：稳定域曲面的每一条等值线、
 * 每一根色标刻度都来自后端帧（`numcore/ode/stability.py` 算好、`server/chapters/3.2.py` 装配）。
 * 这里再写一遍的唯一理由，是让"两侧同算法同结果"可执行地被证明——
 * 例如把 `bᵀA^{j-1}e` 换成"直接抄 `1+z+z²/2+…`"、或把 Durand–Kerner 的初值圆半径
 * 从 `1+max|c_j|` 改成 `1`，帧依然"看起来正常"。
 *
 * 与 `numcore/ode/stability.py` 的逐字对应
 * --------------------------------------
 * | Python | 本模块 | 说明 |
 * |---|---|---|
 * | `rk_tableau` + `rk_stability_coefficients` | {@link rkStabilityCoefficients} | 系数**现场算**：`c_j = bᵀA^{j-1}e`（不是抄 `1/j!`） |
 * | `taylor_stability_coefficients` | {@link taylorStabilityCoefficients} | `c_j = 1/j!`，**另一条代码路径**（`explicit_taylor` 用） |
 * | `bdf_coefficients` | {@link bdfCoefficients} | `α` 升幂表（Hairer & Wanner §III.1）、`β_k = 1` |
 * | `polyval_ascending` | `polyvalAscending`（内部） | 升幂 Horner；**实系数也走复数乘法**（与 numpy 的 dtype 提升一致） |
 * | `stability_function` | {@link stabilityFunction} | `bdf` 分支的 k=1/2 走闭式，k≥3 退化为稳定模（见下） |
 * | `lmm_root_modulus` | {@link lmmRootModulus} | Durand–Kerner 同时迭代 48 次，初值圆半径 `1+max|monic_j|` |
 *
 * `stability_function` 在 `bdf, k ≥ 3` 时返回的是**稳定模**而不是复数 `R`
 * ------------------------------------------------------------------
 * 多步法（k≥3）根本不存在单一的 `R`；Python 侧此时返回 `lmm_root_modulus` 的
 * `max_i|r_i(z)|`（实数组），本模块照抄这个口径：返回 `{re: 模, im: 0}`。
 * **调用方不应把它当成"一步的放大因子"**——这不是 TS 侧的发明，是两侧共同的有意退化。
 *
 * 已知的、被容差显式覆盖的差异（写进 `tests/parity/tolerances.yaml` 的 reason）
 * ----------------------------------------------------------------------
 * 1. `exp/cos/sin`：Durand–Kerner 初值圆上的 `e^{i(θ+0.35)}` 走 glibc 与 V8 两套实现（1–2 ulp）；
 * 2. 复数除法：本模块按 Smith/C99 口径（`ratio = b/a`、`s = 1/d`），
 *    numpy 的 `nc_div` 同源但可能差 1 ulp；除法在 DK 迭代里出现 48 次，是主导误差项；
 * 3. `np.sum` 成对求和 vs 本模块左折叠（残差里的 `Σ_j |monic_j|·|r|^j`）；
 * 4. `np.abs(complex)` 与 `Math.hypot` 都是"防上溢的斜边"，可差 1 ulp；
 * 5. `x**j`：numpy 的 `npy_pow` 对 `j = 0/1/2` 有特例（`1`、`x`、`x*x`），本模块逐条照抄，
 *    `j ≥ 3` 才落到 `Math.pow`。
 */

/** 复数网格：与 Python `np.ndarray[complex128]` 对应，拆成两条实数组（跨语言 JSON 只认实数）。 */
export interface ComplexGrid {
  /** 实部。 */
  re: number[];
  /** 虚部。 */
  im: number[];
}

/** `lmm_root_modulus` 的结果；`residual` 只在 `withResidual` 时非 `null`。 */
export interface LmmRootResult {
  /** 稳定模 `max_i |r_i(z)|`（极点处为 `+∞`，绝不返回 NaN）。 */
  mod: number[];
  /** Durand–Kerner 的相对残差 `max_i |p(r_i)| / Σ_j |c_j||r_i|^j`（极点处记 0）。 */
  residual: number[] | null;
}

/** `lmm_root_modulus` 的选项。 */
export interface LmmRootOptions {
  /** DK 迭代次数（Python 缺省 48；实测与 240 次差 < 1e-15）。 */
  iters?: number;
  /** 是否同时算残差（与 Python `return_residual=True` 同义，多一次 Horner + 一次尺度求和）。 */
  withResidual?: boolean;
}

// --------------------------------------------------------------------------- //
// 系数表（与 numcore/ode/stability.py 的 _TABLEAUX / _RATIONAL / _BDF_ALPHA 逐字一致）
// --------------------------------------------------------------------------- //

/** 显式 RK 的 Butcher 表：`a` 严格下三角（只存用到的行）、`b` 权重。 */
const TABLEAUX: Record<string, { a: number[][]; b: number[] }> = {
  forward_euler: { a: [[0.0]], b: [1.0] },
  midpoint: { a: [[0.0, 0.0], [0.5, 0.0]], b: [0.0, 1.0] },
  heun: { a: [[0.0, 0.0], [1.0, 0.0]], b: [0.5, 0.5] },
  rk4: {
    a: [
      [0.0, 0.0, 0.0, 0.0],
      [0.5, 0.0, 0.0, 0.0],
      [0.0, 0.5, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0],
    ],
    b: [1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0],
  },
};

/** 隐式单步法的有理稳定函数 `R = N/D`（升幂系数）。 */
const RATIONAL: Record<string, { num: number[]; den: number[] }> = {
  implicit_euler: { num: [1.0], den: [1.0, -1.0] },
  trapezoidal: { num: [1.0, 0.5], den: [1.0, -0.5] },
};

/** BDF-k 的 `ρ(r) = Σ_j α_j r^j`（升幂）与 `β_k = 1`。 */
const BDF_ALPHA: Record<number, number[]> = {
  1: [-1.0, 1.0],
  2: [0.5, -2.0, 1.5],
  3: [-1.0 / 3.0, 1.5, -3.0, 11.0 / 6.0],
  4: [0.25, -4.0 / 3.0, 3.0, -4.0, 25.0 / 12.0],
  5: [-0.2, 1.25, -10.0 / 3.0, 5.0, -5.0, 137.0 / 60.0],
  6: [1.0 / 6.0, -1.2, 3.75, -20.0 / 3.0, 7.5, -6.0, 49.0 / 20.0],
};

/**
 * 极点判据的相对阈值 `8·eps`（与 Python `_POLE_RTOL` 同值）。
 *
 * 退化点（`α_k − z` 小到该量级）就是极点：多项式降次、一个根跑到无穷远，
 * 那里 `|R| = +∞` 是有意义的极限值而不是错误。用**相对**判据是为了不把
 * "离极点很近但完全正常"的点（`z = 1.5 − 1e-13`，`|R| ≈ 2e12` 的有限真值）误判成极点。
 */
const POLE_RTOL = 8.0 * Number.EPSILON;

// --------------------------------------------------------------------------- //
// 复数原语（只在本模块内部用；跨语言接口一律是 `{re, im}` 两条实数组）
// --------------------------------------------------------------------------- //

/** Smith/C99 复数除法：`(ar + i·ai)/(br + i·bi)`（与 numpy `nc_div` 同口径）。 */
function cdiv(ar: number, ai: number, br: number, bi: number): [number, number] {
  if (br === 0.0 && bi === 0.0) {
    // 除零：C99 复数除法给 ±Inf + NaN·i。本 case 的 z 网格避开所有极点，
    // 这条分支只为"函数在任何输入下都不返回谎话"而存在（Python 侧同样靠 errstate 静音）。
    return [Math.sign(ar) * Number.POSITIVE_INFINITY, Number.NaN];
  }
  if (Math.abs(br) >= Math.abs(bi)) {
    const ratio = bi / br;
    const d = br + ratio * bi;
    const s = 1.0 / d;
    return [(ar + ai * ratio) * s, (ai - ar * ratio) * s];
  }
  const ratio = br / bi;
  const d = bi + ratio * br;
  const s = 1.0 / d;
  return [(ar * ratio + ai) * s, (ai * ratio - ar) * s];
}

/** 主平方根（C99 `csqrt` 口径：负实轴取 `+i√|x|`，与 `np.sqrt` 的主支一致）。 */
function csqrt(ar: number, ai: number): [number, number] {
  if (ar === 0.0 && ai === 0.0) return [0.0, ai];
  const mag = Math.hypot(ar, ai);
  const t = Math.sqrt((mag + Math.abs(ar)) / 2.0);
  if (ar >= 0.0) return [t, ai / (2.0 * t)];
  return [Math.abs(ai) / (2.0 * t), ai >= 0.0 ? t : -t];
}

/** `numpy` 的 `npy_pow` 对整数指数的特例（`0/1/2`），其余落 `Math.pow`。 */
function powInt(x: number, j: number): number {
  if (j === 0) return 1.0;
  if (j === 1) return x;
  if (j === 2) return x * x;
  return Math.pow(x, j);
}

/** `|z|`（`Math.hypot` 与 numpy 的 `npy_hypot` 同为防上溢实现，可差 1 ulp）。 */
export function complexAbs(z: ComplexGrid): number[] {
  const n = z.re.length;
  const out: number[] = new Array<number>(n);
  for (let i = 0; i < n; i++) out[i] = Math.hypot(z.re[i] as number, z.im[i] as number);
  return out;
}

// --------------------------------------------------------------------------- //
// 系数
// --------------------------------------------------------------------------- //

/**
 * 显式 RK 的稳定多项式系数（升幂）：`R(z) = Σ_j c_j z^j`，`c_j = bᵀA^{j-1}e`。
 *
 * 显式方法的 `A` 严格下三角，`A^s = 0`，级数在第 `s` 项自动截断——
 * "显式 RK 的 `R` 是多项式"是显式性的推论，不是抄来的公式。
 */
export function rkStabilityCoefficients(name: string): number[] {
  const tab = TABLEAUX[name];
  if (tab === undefined) {
    throw new Error(`未知显式 RK 表 ${JSON.stringify(name)}（可用：${Object.keys(TABLEAUX).sort().join(", ")}）`);
  }
  const s = tab.b.length;
  const coeffs: number[] = new Array<number>(s + 1).fill(0);
  coeffs[0] = 1.0;
  let u: number[] = new Array<number>(s).fill(1.0); // A^{j-1} e
  for (let j = 1; j <= s; j++) {
    let acc = 0.0;
    for (let i = 0; i < s; i++) acc += (tab.b[i] as number) * (u[i] as number);
    coeffs[j] = acc;
    const next: number[] = new Array<number>(s).fill(0);
    for (let i = 0; i < s; i++) {
      let row = 0.0;
      for (let jj = 0; jj < s; jj++) row += ((tab.a[i] as number[])[jj] as number) * (u[jj] as number);
      next[i] = row;
    }
    u = next;
  }
  return coeffs;
}

/** `p` 级 `p` 阶显式 RK 的稳定多项式 `R_p(z) = Σ_{j≤p} z^j/j!`（阶条件的推论）。 */
export function taylorStabilityCoefficients(order: number): number[] {
  const p = Math.trunc(order);
  if (p < 1) throw new Error(`阶数至少为 1，得到 ${String(order)}`);
  const out: number[] = new Array<number>(p + 1);
  let fact = 1.0; // j!（j ≤ 22 时在 double 里精确）
  out[0] = 1.0 / fact;
  for (let j = 1; j <= p; j++) {
    fact *= j;
    out[j] = 1.0 / fact;
  }
  return out;
}

/** BDF-`k` 的 `(α, β_k)`：`Σ_j α_j y_{n+j} = h β_k f_{n+k}`，`α` 升幂、`β_k = 1`。 */
export function bdfCoefficients(k: number): { alpha: number[]; betaK: number } {
  const kk = Math.trunc(k);
  const alpha = BDF_ALPHA[kk];
  if (alpha === undefined) throw new Error(`BDF 阶数只实现 1..6（k≥7 连零稳定都没有），得到 ${String(k)}`);
  return { alpha: alpha.slice(), betaK: 1.0 };
}

// --------------------------------------------------------------------------- //
// 求值
// --------------------------------------------------------------------------- //

/** 升幂 Horner `Σ_j c_j z^j`（实系数也走复数乘法，与 numpy 的 dtype 提升逐字一致）。 */
function polyvalAscending(coeffs: number[], z: ComplexGrid): ComplexGrid {
  const n = z.re.length;
  const c = coeffs;
  if (c.length === 0) throw new Error("系数不能为空");
  const outRe: number[] = new Array<number>(n).fill(c[c.length - 1] as number);
  const outIm: number[] = new Array<number>(n).fill(0.0);
  for (let j = c.length - 2; j >= 0; j--) {
    const cj = c[j] as number;
    for (let p = 0; p < n; p++) {
      const or = outRe[p] as number;
      const oi = outIm[p] as number;
      const zr = z.re[p] as number;
      const zi = z.im[p] as number;
      outRe[p] = or * zr - oi * zi + cj;
      outIm[p] = or * zi + oi * zr + 0.0;
    }
  }
  return { re: outRe, im: outIm };
}

/**
 * 单步法的稳定函数 `R(z)`（复数值）。
 *
 * `bdf` 的 `k = 1, 2` 分别就是隐式 Euler 与 BDF2 的闭式主根；`k ≥ 3` 时
 * **退化为稳定模** `max_i|r_i(z)|`（`im = 0`）——见文件头说明。
 */
export function stabilityFunction(method: string, order: number, z: ComplexGrid): ComplexGrid {
  const n = z.re.length;
  if (z.im.length !== n) throw new Error(`z 的实部/虚部长度不一致：${n} vs ${z.im.length}`);
  if (method === "bdf") {
    const { alpha, betaK } = bdfCoefficients(order);
    const k = alpha.length - 1;
    if (k === 1) {
      // z=1 是极点（1/(1−z) 的分母零点）：|R|=+∞ 是极限值，不是错误
      const re: number[] = new Array<number>(n);
      const im: number[] = new Array<number>(n);
      for (let p = 0; p < n; p++) {
        const [r, i] = cdiv(1.0, 0.0, 1.0 - (z.re[p] as number), -(z.im[p] as number));
        re[p] = r;
        im[p] = i;
      }
      return { re, im };
    }
    if (k === 2) {
      // (3/2 − z)r² − 2r + 1/2 = 0  ⟹  r = (2 ± sqrt(1+2z)) / (3 − 2z)；z=1.5 处是极点
      const re: number[] = new Array<number>(n);
      const im: number[] = new Array<number>(n);
      for (let p = 0; p < n; p++) {
        const zr = z.re[p] as number;
        const zi = z.im[p] as number;
        const [sr, si] = csqrt(1.0 + 2.0 * zr, 2.0 * zi);
        const [r, i] = cdiv(2.0 + sr, si, 3.0 - 2.0 * zr, -2.0 * zi);
        re[p] = r;
        im[p] = i;
      }
      return { re, im };
    }
    return { re: lmmRootModulus(alpha, betaK, z).mod, im: new Array<number>(n).fill(0.0) };
  }
  const rat = RATIONAL[method];
  if (rat !== undefined) {
    const num = polyvalAscending(rat.num, z);
    const den = polyvalAscending(rat.den, z);
    const re: number[] = new Array<number>(n);
    const im: number[] = new Array<number>(n);
    for (let p = 0; p < n; p++) {
      const [r, i] = cdiv(num.re[p] as number, num.im[p] as number, den.re[p] as number, den.im[p] as number);
      re[p] = r;
      im[p] = i;
    }
    return { re, im };
  }
  if (TABLEAUX[method] !== undefined || method === "explicit_taylor") {
    const coeffs = method === "explicit_taylor" ? taylorStabilityCoefficients(order) : rkStabilityCoefficients(method);
    return polyvalAscending(coeffs, z);
  }
  throw new Error(`未知方法 ${JSON.stringify(method)}`);
}

/**
 * 线性多步法的稳定模 `max_i|r_i(z)|`：`r_i` 是 `ρ(r) − zσ(r)` 的根。
 *
 * 根用**自写的 Durand–Kerner 迭代**（Weierstrass 同时迭代）求：
 * `r_i ← r_i − p(r_i)/∏_{j≠i}(r_i − r_j)`，初值取半径 `1+max_j|c_j|` 的圆上等分布点
 * （相位角再偏 0.35）——确定性、无随机、无 `np.roots`。
 *
 * **更新必须是 Jacobi 式（同时）的**：先把所有 `p(r_i)` 与所有 `∏_{j≠i}` 算完，
 * 再统一改 `roots`。边算边改会让第 `i+1` 个根的乘积读到已经动过的第 `i` 个根，
 * 那就是另一个算法了。
 *
 * 退化点（`r^k` 的系数为 0，即 `z = α_k/β_k`）是极点：多项式降次、一个根跑到无穷远，
 * 直接给 `+∞` 并跳过 DK——否则首一化会除以 0，迭代出来的全是 NaN。
 */
export function lmmRootModulus(
  alpha: ArrayLike<number>,
  betaK: number,
  z: ComplexGrid,
  opts: LmmRootOptions = {},
): LmmRootResult {
  const a: number[] = Array.from(alpha);
  const k = a.length - 1;
  if (k < 1) throw new Error("多步法至少是一步的");
  const n = z.re.length;
  if (z.im.length !== n) throw new Error(`z 的实部/虚部长度不一致：${n} vs ${z.im.length}`);
  const iters = Math.max(1, Math.trunc(opts.iters === undefined ? 48 : opts.iters));
  const wantResidual = opts.withResidual === true;
  if (n === 0) return { mod: [], residual: wantResidual ? [] : null };

  const w = k + 1; // 每个点的系数个数
  const coeffRe = new Float64Array(n * w);
  const coeffIm = new Float64Array(n * w);
  for (let p = 0; p < n; p++) {
    const base = p * w;
    for (let j = 0; j <= k; j++) {
      coeffRe[base + j] = a[j] as number;
      coeffIm[base + j] = 0.0;
    }
    // ρ(r) − z·β_k·r^k
    coeffRe[base + k] = (a[k] as number) - (z.re[p] as number) * betaK;
    coeffIm[base + k] = 0.0 - (z.im[p] as number) * betaK;
  }

  // 极点掩码：最高次系数的模相对整行尺度退化（_pole_where）
  const pole = new Uint8Array(n);
  for (let p = 0; p < n; p++) {
    const base = p * w;
    const lead = Math.hypot(coeffRe[base + k] as number, coeffIm[base + k] as number);
    let scale = 0.0;
    for (let j = 0; j <= k; j++) {
      const v = Math.hypot(coeffRe[base + j] as number, coeffIm[base + j] as number);
      if (v > scale) scale = v;
    }
    pole[p] = lead <= POLE_RTOL * Math.max(scale, 1e-300) ? 1 : 0;
  }

  // 首一化：极点处用占位 1.0（结果稍后被 +∞ 覆盖，只为让除法不炸）
  const monicRe = new Float64Array(n * w);
  const monicIm = new Float64Array(n * w);
  for (let p = 0; p < n; p++) {
    const base = p * w;
    const isPole = pole[p] === 1;
    const lead = isPole ? 1.0 : (coeffRe[base + k] as number);
    const leadIm = isPole ? 0.0 : (coeffIm[base + k] as number);
    for (let j = 0; j <= k; j++) {
      const cr = isPole && j === k ? 1.0 : (coeffRe[base + j] as number);
      const ci = isPole && j === k ? 0.0 : (coeffIm[base + j] as number);
      const [qr, qi] = cdiv(cr, ci, lead, leadIm);
      monicRe[base + j] = qr;
      monicIm[base + j] = qi;
    }
  }

  // 初值：半径 1 + max_j|monic_j| 的圆上等分布点，相位再偏 0.35
  const rootsRe = new Float64Array(n * k);
  const rootsIm = new Float64Array(n * k);
  for (let p = 0; p < n; p++) {
    const base = p * w;
    let mx = 0.0;
    for (let j = 0; j < k; j++) {
      const v = Math.hypot(monicRe[base + j] as number, monicIm[base + j] as number);
      if (v > mx) mx = v;
    }
    const radius = 1.0 + mx;
    for (let j = 0; j < k; j++) {
      const theta = (2.0 * Math.PI * (j + 0.5)) / k + 0.35;
      rootsRe[p * k + j] = radius * Math.cos(theta);
      rootsIm[p * k + j] = radius * Math.sin(theta);
    }
  }

  const valsRe = new Float64Array(n * k);
  const valsIm = new Float64Array(n * k);
  const prodRe = new Float64Array(n * k);
  const prodIm = new Float64Array(n * k);

  /** 首一多项式 `monic` 在全部 `roots` 处求值（Horner，与 `_eval_monic` 同序）。 */
  const evalMonic = (): void => {
    for (let p = 0; p < n; p++) {
      for (let i = 0; i < k; i++) {
        const idx = p * k + i;
        const base = p * w;
        let vr = monicRe[base + k] as number;
        let vi = monicIm[base + k] as number;
        const rr = rootsRe[idx] as number;
        const ri = rootsIm[idx] as number;
        for (let j = k - 1; j >= 0; j--) {
          const nr = vr * rr - vi * ri + (monicRe[base + j] as number);
          const ni = vr * ri + vi * rr + (monicIm[base + j] as number);
          vr = nr;
          vi = ni;
        }
        valsRe[idx] = vr;
        valsIm[idx] = vi;
      }
    }
  };

  for (let it = 0; it < iters; it++) {
    evalMonic();
    // ∏_{j≠i}(r_i − r_j)：Python 把自因子置 1 后连乘，乘 1.0 逐位不变，故这里直接跳过 j = i
    for (let p = 0; p < n; p++) {
      for (let i = 0; i < k; i++) {
        const idx = p * k + i;
        let pr = 1.0;
        let pi = 0.0;
        const ri = rootsRe[idx] as number;
        const ii = rootsIm[idx] as number;
        for (let j = 0; j < k; j++) {
          if (j === i) continue;
          const jdx = p * k + j;
          const dr = ri - (rootsRe[jdx] as number);
          const di = ii - (rootsIm[jdx] as number);
          const nr = pr * dr - pi * di;
          const ni = pr * di + pi * dr;
          pr = nr;
          pi = ni;
        }
        prodRe[idx] = pr;
        prodIm[idx] = pi;
      }
    }
    // 统一更新（Jacobi 式）
    for (let idx = 0; idx < n * k; idx++) {
      const [qr, qi] = cdiv(valsRe[idx] as number, valsIm[idx] as number, prodRe[idx] as number, prodIm[idx] as number);
      rootsRe[idx] = (rootsRe[idx] as number) - qr;
      rootsIm[idx] = (rootsIm[idx] as number) - qi;
    }
  }

  const mod: number[] = new Array<number>(n);
  for (let p = 0; p < n; p++) {
    let best = 0.0;
    for (let i = 0; i < k; i++) {
      const idx = p * k + i;
      const v = Math.hypot(rootsRe[idx] as number, rootsIm[idx] as number);
      if (v > best) best = v;
    }
    // 库的约定：非有限一律折成 +∞（"模无界"），绝不返回 NaN
    mod[p] = Number.isFinite(best) ? best : Number.POSITIVE_INFINITY;
    if (pole[p] === 1) mod[p] = Number.POSITIVE_INFINITY;
  }

  let residual: number[] | null = null;
  if (wantResidual) {
    evalMonic();
    residual = new Array<number>(n);
    for (let p = 0; p < n; p++) {
      const base = p * w;
      let worst = 0.0;
      for (let i = 0; i < k; i++) {
        const idx = p * k + i;
        const absR = Math.hypot(rootsRe[idx] as number, rootsIm[idx] as number);
        let scale = 0.0;
        for (let j = 0; j <= k; j++) {
          scale += Math.hypot(monicRe[base + j] as number, monicIm[base + j] as number) * powInt(absR, j);
        }
        const v = Math.hypot(valsRe[idx] as number, valsIm[idx] as number) / Math.max(scale, 1e-300);
        if (v > worst) worst = v;
      }
      residual[p] = Number.isFinite(worst) ? worst : Number.POSITIVE_INFINITY;
      if (pole[p] === 1) residual[p] = 0.0; // 极点处"根在无穷远"是解析事实，残差记 0
    }
  }

  return { mod, residual };
}
