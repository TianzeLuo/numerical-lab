/**
 * 幂法的 TS 侧对拍实现（PROTOCOLS §7 覆盖清单里的 `power_iter`，对应 1.4 章）。
 *
 * **本目录只服务 G7 双实现对拍，不是渲染路径。**
 * `viz/chapters/**` MUST NOT 直接 import 本目录（P8）：1.4 章前端消费的是后端帧，
 * 帧里的 λ₁ / 残差 / 夹角全部由 `numcore/linalg/eigen.py` 算好。
 * 这里再写一遍的唯一理由，是让"两侧同算法同结果"可执行地被证明——
 * 帧协议只传数据不传公式，把 `x_{k+1} = Ax_k/‖Ax_k‖` 换成 `Aᵀx_k`、
 * 或把停止判据从相对残差换成 Rayleigh 商的变化量，帧依然"看起来正常"。
 *
 * 与 `numcore/linalg/eigen.py::power_iteration`（第 269 行起）的逐字对应
 * ------------------------------------------------------------------
 * | Python | 本模块 | 说明 |
 * |---|---|---|
 * | `_as_square(A)` | `a`（行主序展平）+ `n` | 只支持实方阵（本 case 全是实矩阵） |
 * | `normalize(x0)` | {@link normalize} | 同一 `_norm2`：**先按 `max|x_i|` 缩放**再求 `√Σ(v/m)²`，零向量抛错 |
 * | `rayleigh_quotient(A, x)` | `rayleighQuotient` | `Σx_i·(Ax)_i / Σx_i·x_i`（实数分支，无共轭开销） |
 * | `eigen_residual(A, λ, x)` | `eigenResidual` | `‖Ax − λx‖₂`，**重新做一次** matvec（与 Python 同结构） |
 * | `vectors[k+1] = y/ny` | 同 | 归一化后的迭代向量 |
 * | `angle_step[k+1] = arccos(clamp(|⟨x, x_new⟩|))` | 同 | `clamp` 到 `[0,1]` 之后再 `arccos` |
 * | `residual ≤ tol·max(1, |ρ|)` | 同 | **唯一的停止判据**（不是 `‖x_{k+1}−x_k‖`） |
 * | `ny == 0 → used = k; break` | 同 | `Ax = 0`：主特征值为 0（或起点落在零空间） |
 * | `ratios = pos[1:]/pos[:-1]`，再丢一个 | 同 | `pos` 是严格正残差；不足 3 个则为空 |
 * | `is_complex_pair`：末 1/4 段夹角平台 > 1e-3 | 同 | 复主特征值的诊断标志 |
 *
 * **初值为什么不在 TS 侧再造一遍**（"同一初值构造"的落点）
 * ------------------------------------------------------
 * Python 侧的起点是 `normalize(_random_start(n, seed=seed))`，而 `_random_start` 走
 * `numcore.utils.rng` = `np.random.Generator(PCG64(seed))` 的 `standard_normal`（ziggurat）。
 * 把 PCG64+ziggurat 在 TS 里抄一遍，等于把"两个 RNG 是否逐位一致"变成对拍的前提，
 * 而且直接违反 PROTOCOLS §7 的纪律：**输入一律由 Python 生成落盘，避免输入构造这一环
 * 被两套实现的差异污染**（`lu` case 的矩阵、`heat` case 的 `u0` 都是这个口径）。
 * 所以本模块 **要求显式传入 `x0`**：case 的 `inputs.x0` 就是 Python 用
 * `_random_start(n, seed=seed)` 生成的 double，TS 只做 `normalize` ——
 * 归一化口径（max 缩放 + 平方和 + sqrt）与 Python 逐字相同。
 *
 * 已知的、被容差显式覆盖的差异（写进 `tests/parity/tolerances.yaml` 的 reason）
 * ----------------------------------------------------------------------
 * 1. `A @ x`：numpy 走 BLAS dgemv（SIMD/FMA、分块求和），本模块是朴素行内左折叠；
 * 2. `np.sum`：n ≥ 8 时 numpy 用 8 路展开的成对求和，本模块仍是左折叠；
 * 3. `acos`：glibc 与 V8 的 fdlibm 移植差 1–2 ulp；**夹角趋 0 时 `arccos` 的条件数
 *    是 `1/sin θ`**，`inner` 的 1 ulp 差最多放大成 `√(2u) ≈ 1.5e-8` 的绝对差
 *    （`inner` 一侧舍入到恰好 1.0 时另一侧不是）——这是 `angleStep` 的容差必须比
 *    `eigenvalue` 宽几个数量级的唯一原因，与算法无关。
 */

/** 幂法一次运行的对拍输出（字段与 Python `PowerResult` 一一对应）。 */
export interface ParityPowerIter {
  /** 最后一次迭代的 Rayleigh 商（`nIter = 0` 时是起点处的商）。 */
  eigenvalue: number;
  /** 末迭代向量 `x`（已归一化；`nIter = 0` 时就是归一化后的起点）。 */
  vector: number[];
  /** 每一步的 Rayleigh 商，长度 `nIter + 1`（第 0 个是起点）。 */
  rayleigh: number[];
  /** 对应的残差 `‖Ax − ρ(x)x‖₂`，长度同 `rayleigh`。 */
  residual: number[];
  /** 相邻两向量的夹角 `arccos|⟨x_k, x_{k+1}⟩|`，长度同 `rayleigh`（第 0 个恒为 0）。 */
  angleStep: number[];
  /** 实际迭代步数（未收敛时等于 `maxIter`）。 */
  nIter: number;
  /** 是否由 `residual ≤ tol·max(1, |ρ|)` 提前停止。 */
  converged: boolean;
  /** 正残差的一阶比值（丢掉第一步瞬态）；正残差不足 3 个时为空数组。 */
  ratios: number[];
  /** 末 1/4 段夹角平台 > 1e-3（复主特征值诊断；本 case 的两族都是实谱）。 */
  isComplexPair: boolean;
}

/** {@link powerIteration} 的选项；`x0` 必填（见文件头"初值为什么不在 TS 侧再造一遍"）。 */
export interface PowerIterOptions {
  /** 起点（未归一化也可，内部会 `normalize`）——由 Python 侧的 `_random_start(n, seed)` 生成落盘。 */
  x0: ArrayLike<number>;
  /** 迭代上限（Python 默认 200）；必须 ≥ 1，否则与 Python 一样抛错。 */
  maxIter?: number;
  /** 相对停止阈值（Python 默认 1e-12）。 */
  tol?: number;
}

/** 行主序方阵 → 二维数组（跨语言约定的存储顺序，`lu.ts` 同款）。 */
function toRows(a: ArrayLike<number>, n: number): number[][] {
  const m: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = new Array<number>(n);
    for (let j = 0; j < n; j++) row[j] = a[i * n + j] as number;
    m.push(row);
  }
  return m;
}

/**
 * `‖x‖₂`（与 Python `_norm2_real` 逐字同序：**先按最大元缩放**，再 `√Σ(v/m)²`）。
 *
 * 缩放不是为了好看：`Σ v_i²` 在 `v` 的量级到 1e150 时会直接上溢成 `Inf`，
 * 而 max 缩放把中间量钉在 `[0,1]`，只在最后一步乘回去。
 */
function norm2(x: ArrayLike<number>): number {
  const n = x.length;
  if (n === 0) return 0;
  let m = 0;
  for (let i = 0; i < n; i++) {
    const v = Math.abs(x[i] as number);
    if (v > m) m = v;
  }
  if (m === 0) return 0;
  let s = 0;
  for (let i = 0; i < n; i++) {
    const t = (x[i] as number) / m;
    s += t * t;
  }
  return m * Math.sqrt(s);
}

/** `x/‖x‖₂`；零向量抛错（对齐 Python `normalize` 的 `ValueError`）。 */
function normalize(x: ArrayLike<number>): number[] {
  const nrm = norm2(x);
  if (nrm === 0) {
    throw new Error("零向量无法归一化（幂法的起点必须在主特征方向上有分量）");
  }
  const out: number[] = new Array<number>(x.length);
  for (let i = 0; i < x.length; i++) out[i] = (x[i] as number) / nrm;
  return out;
}

/** `A @ x`（按行左折叠；Python 侧是 `arr @ x`，见文件头"已知差异"第 1 条）。 */
function matvec(m: number[][], x: ArrayLike<number>): number[] {
  const n = m.length;
  const out: number[] = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    const row = m[i] as number[];
    let s = 0;
    for (let j = 0; j < n; j++) s += (row[j] as number) * (x[j] as number);
    out[i] = s;
  }
  return out;
}

/** Rayleigh 商 `ρ(x) = xᵀAx / xᵀx`（实矩阵实向量分支；零向量抛错，对齐 Python）。 */
function rayleighQuotient(m: number[][], x: ArrayLike<number>): number {
  let denom = 0;
  for (let i = 0; i < x.length; i++) {
    const v = x[i] as number;
    denom += v * v;
  }
  if (denom === 0) throw new Error("零向量的 Rayleigh 商无定义");
  const y = matvec(m, x);
  let num = 0;
  for (let i = 0; i < x.length; i++) num += (x[i] as number) * (y[i] as number);
  return num / denom;
}

/** 特征对残差 `‖Ax − λx‖₂`（**不做**归一化：`x` 按给定量纲算，与 Python 同）。 */
function eigenResidual(m: number[][], lam: number, x: ArrayLike<number>): number {
  const y = matvec(m, x);
  const w: number[] = new Array<number>(x.length);
  for (let i = 0; i < x.length; i++) w[i] = (y[i] as number) - lam * (x[i] as number);
  return norm2(w);
}

/**
 * 幂法：`x_{k+1} = A x_k/‖A x_k‖₂`，用 Rayleigh 商估 `λ₁`（`numcore/linalg/eigen.py::power_iteration` 的逐字移植）。
 *
 * 收敛分支与停止点必须与 Python **完全一致**，因为 `rayleigh` / `residual` / `ratios`
 * 是**变长**数组：停止点差一步，长度就对不上，G7 会直接报"长度不一致"而不是数值超差。
 * 这正是本 case 的价值所在——它把"停止判据写没写对"变成可判定的。
 *
 * @param a 行主序方阵，长度必须是 `n*n`
 * @param n 阶数
 * @param opts `x0` 必填；`maxIter` / `tol` 缺省与 Python 一致（200 / 1e-12）
 */
export function powerIteration(a: ArrayLike<number>, n: number, opts: PowerIterOptions): ParityPowerIter {
  if (!Number.isInteger(n) || n <= 0) throw new Error(`阶数 n 必须是正整数，得到 ${String(n)}`);
  if (a.length !== n * n) throw new Error(`矩阵长度 ${a.length} ≠ n*n = ${n * n}（行主序展平？）`);
  const maxIter = opts.maxIter === undefined ? 200 : opts.maxIter;
  if (!Number.isInteger(maxIter) || maxIter < 1) {
    throw new Error(`maxIter 至少为 1，得到 ${String(maxIter)}`);
  }
  const tol = opts.tol === undefined ? 1e-12 : opts.tol;
  const m = toRows(a, n);
  let x: number[] = normalize(opts.x0);
  if (x.length !== n) throw new Error(`起点长度 ${x.length} ≠ n = ${n}`);

  // Python 侧预分配 (max_iter+1) 再按 used 截断；未走到的位置保持 0 —— 那里
  // `ny == 0` 的提前 break 会把"预分配的 0"留在历史里，这里用同样语义的定长数组 + 截断。
  const rayleigh: number[] = new Array<number>(maxIter + 1).fill(0);
  const residual: number[] = new Array<number>(maxIter + 1).fill(0);
  const angleStep: number[] = new Array<number>(maxIter + 1).fill(0);
  rayleigh[0] = rayleighQuotient(m, x);
  residual[0] = eigenResidual(m, rayleigh[0] as number, x);

  let converged = false;
  let used = 0;
  for (let k = 0; k < maxIter; k++) {
    const y = matvec(m, x);
    const ny = norm2(y);
    if (ny === 0) {
      // A x = 0：主特征值为 0（或起点落在零空间里）；与 Python 一样把 used 停在 k
      used = k;
      break;
    }
    const xNew: number[] = new Array<number>(n);
    for (let i = 0; i < n; i++) xNew[i] = (y[i] as number) / ny;
    // 夹角：用 |⟨x, x_new⟩| 反余弦；复主特征值时它会停在 O(1) 而不是 → 0
    let inner = 0;
    for (let i = 0; i < n; i++) inner += (x[i] as number) * (xNew[i] as number);
    inner = Math.abs(inner);
    inner = Math.min(1, Math.max(0, inner));
    angleStep[k + 1] = Math.acos(inner);
    x = xNew;
    rayleigh[k + 1] = rayleighQuotient(m, x);
    residual[k + 1] = eigenResidual(m, rayleigh[k + 1] as number, x);
    used = k + 1;
    const scale = Math.max(1, Math.abs(rayleigh[k + 1] as number));
    if ((residual[k + 1] as number) <= tol * scale) {
      converged = true;
      break;
    }
  }

  const len = used + 1;
  const rq = rayleigh.slice(0, len);
  const rs = residual.slice(0, len);
  const ang = angleStep.slice(0, len);

  // 实测收敛因子：正残差的一阶比值（丢掉第一步的瞬态），与 Python 的 pos[1:]/pos[:-1] 同序
  const pos = rs.filter((v) => v > 0);
  let ratios: number[] = [];
  if (pos.length >= 3) {
    ratios = new Array<number>(pos.length - 1);
    for (let i = 1; i < pos.length; i++) ratios[i - 1] = (pos[i] as number) / (pos[i - 1] as number);
    if (ratios.length > 1) ratios = ratios.slice(1);
  }

  // 复主特征值的标志：末 1/4 段的夹角平台仍大于 1e-3 弧度
  const start = Math.max(1, ang.length - Math.max(2, Math.floor(ang.length / 4)));
  const tail = ang.slice(start);
  let plateau = 0;
  if (tail.length > 0) {
    let s = 0;
    for (const v of tail) s += v;
    plateau = s / tail.length;
  }

  return {
    eigenvalue: rq[rq.length - 1] as number,
    vector: x.slice(0, n),
    rayleigh: rq,
    residual: rs,
    angleStep: ang,
    nIter: used,
    converged,
    ratios,
    isComplexPair: plateau > 1e-3,
  };
}
