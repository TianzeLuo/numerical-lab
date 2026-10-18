/**
 * 一维热传导显式格式（FTCS）的 TS 侧对拍实现。
 *
 * **本目录只服务 G7 双实现对拍，不是渲染路径。**
 * `viz/chapters/**` MUST NOT 直接 import 本目录（P8）：章节前端只能消费后端帧；
 * 本目录存在的唯一理由，是让 `tools/parity.py` 能把同一批用例在两侧各算一遍并逐点比较。
 * 因此这里**只放纯函数**：无 IO、无 DOM、无第三方依赖，也不做任何"渲染友好"的加工。
 *
 * 数学定义
 * --------
 * 一维热传导 `u_t = α u_xx`，齐次 Dirichlet 边条件 `u(0,t) = u(1,t) = 0`。
 * 网格 `Δx = 1/(nx−1)`，记 `r = αΔt/Δx²`，显式格式（FTCS）为
 *
 * ```
 * u_i^{n+1} = u_i^n + r·(u_{i+1}^n − 2u_i^n + u_{i−1}^n),  i = 1 … nx−2
 * u_0^{n+1} = u_{nx−1}^{n+1} = 0
 * ```
 *
 * 放大因子（对第 `mode` 个 Fourier 模 `sin(kπx)`，`k = mode`）
 *
 * ```
 * g = 1 − 4r·sin²(kπΔx/2)，θ = kπ/(nx−1)
 * ```
 *
 * 与 Python 侧的对应关系（`numcore/smoke.py`）
 * ------------------------------------------
 * | Python | 本模块 | 说明 |
 * |---|---|---|
 * | `heat_ftcs(...)` 的步进循环 | {@link ftcsStep} / {@link ftcsRun} | 表达式与结合顺序逐字对齐 |
 * | `ftcs_amplification(r, mode, nx)` | {@link ftcsAmplification} | 同一公式、同一求值顺序 |
 *
 * **为什么要逐字对齐表达式**：浮点加减乘不满足结合律，`(a − 2b) + c` 与 `(c − 2b) + a`
 * 可以差 1 ulp。对拍的意义就在于把"两侧是不是同一套算法"钉死，而不是"看起来差不多"。
 */

/** `r ≥ 0` 且有限（对齐 `numcore.smoke.ftcs_amplification` 的 `ValueError`）。 */
function checkR(r: number): void {
  if (!Number.isFinite(r) || r < 0) {
    throw new RangeError(`r 必须是非负有限数，得到 ${String(r)}`);
  }
}

/**
 * 一步 FTCS：`u' = u + r·(u_{i+1} − 2u_i + u_{i−1})`，两端固定 0。
 *
 * 纯函数：**不修改入参**，返回新数组（Python 侧 `heat_ftcs` 是原地更新，
 * 但两侧对拍的是"同一份输入 → 同一份输出"，原地与否不影响数值）。
 */
export function ftcsStep(u: Float64Array, r: number): Float64Array {
  checkR(r);
  const n = u.length;
  const out = new Float64Array(n);
  if (n === 0) return out;
  if (n === 1) {
    // 只有一个点：齐次 Dirichlet 两端重合，值恒为 0
    out[0] = 0;
    return out;
  }
  for (let i = 1; i < n - 1; i++) {
    // 与 numpy 的 `u[interior] + r * (u[2:] - 2.0 * u[interior] + u[:-2])` 同序
    out[i] = u[i] + r * (u[i + 1] - 2.0 * u[i] + u[i - 1]);
  }
  out[0] = 0;
  out[n - 1] = 0;
  return out;
}

/**
 * 连跑 `nSteps` 步（`nSteps = 0` 时返回 `u0` 的副本）。
 *
 * 不修改 `u0`：每一步都产生新数组，便于调用方保留中间状态做差分定位。
 */
export function ftcsRun(u0: Float64Array, r: number, nSteps: number): Float64Array {
  checkR(r);
  if (!Number.isInteger(nSteps) || nSteps < 0) {
    throw new RangeError(`nSteps 必须是非负整数，得到 ${String(nSteps)}`);
  }
  let u: Float64Array = Float64Array.from(u0);
  for (let k = 0; k < nSteps; k++) u = ftcsStep(u, r);
  return u;
}

/**
 * FTCS 对第 `mode` 个 Fourier 模的放大因子 `g = 1 − 4r·sin²(θ/2)`，`θ = mode·π/(nx−1)`。
 *
 * 有效模态是 `mode = 1 … nx−2`（内部有 `nx−2` 个未知量）；`mode = nx−1` 是网格分辨不出的
 * "每格反号"幽灵模态——这里与 Python 一致地**不禁止**它（`1 ≤ mode ≤ nx−1`），
 * 因为它正是"r\* (nx) 为什么严格大于 1/2"的证据。
 */
export function ftcsAmplification(r: number, mode: number, nx: number): number {
  checkR(r);
  if (!Number.isInteger(mode) || mode < 1 || mode > nx - 1) {
    throw new RangeError(`mode 必须在 1..${nx - 1} 之间，得到 ${String(mode)}`);
  }
  const theta = (mode * Math.PI) / (nx - 1);
  const s = Math.sin(theta / 2.0);
  // 结合顺序对齐 Python：1.0 - (4.0 * r) * (s ** 2)
  return 1.0 - 4.0 * r * (s * s);
}
