/**
 * 部分主元 LU 分解的 TS 侧对拍薄包装（PROTOCOLS §7 覆盖清单里的 `lu`）。
 *
 * **本目录只服务 G7 双实现对拍，不是渲染路径。**
 * `viz/chapters/**` MUST NOT 直接 import 本目录（P8）：章节目录里的 `lu_heatmap.model.ts`
 * 消费的是后端帧，帧里的 L/U/置换/ρ 全部由 `numcore/linalg/lu.py` 算好。
 *
 * 为什么还要在 TS 里再写一遍
 * --------------------------
 * 这一层存在的唯一理由是**可执行地证明"两侧同算法同结果"**：帧协议只传数据、不传公式，
 * 如果有人把 Python 侧的选主元规则改成"选绝对值最大的**列**"或把增长因子改成
 * `max|L|`，帧依然"看起来正常"，只有逐点对拍会立刻炸。
 *
 * 与 `numcore/linalg/lu.py::lu_decompose(pivot="partial")` 的对应关系
 * ----------------------------------------------------------------
 * | Python | 本模块 | 说明 |
 * |---|---|---|
 * | `L` | `l` | 行主序展平；严格下三角 + 单位对角 |
 * | `U` | `u` | 行主序展平；上三角（含对角） |
 * | `perm` | `perm` | `perm[k]` = 第 k 步之后的原始行号 |
 * | `pivots` | `pivots` | 每一步选中的主元**行**号（未交换前的行号） |
 * | `row_swaps` | `rowSwaps` | 实际发生的行交换，展平成 `[k0, p0, k1, p1, …]`（无交换则为空） |
 * | `rho` | `rho` | 增长因子 `max\|U\| / max\|A\|`（`max\|A\| = 0` 时取 0） |
 *
 * **消元顺序必须与 Python 侧逐字一致**：那边是
 * `M[k+1:, k+1:] -= outer(m, M[k, k+1:])`，即"先整列算乘子、再整块外积更新"。
 * 换成逐元素 `a[i][j] -= m[i]*a[k][j]` 在 float64 下也能过容差，但一旦
 * 容差收到 1 ulp 量级就会露馅——两边都按外积写法来，理由写在
 * `tests/parity/tolerances.yaml` 的 `reason` 里。
 */

/** 分解结果（跨语言比对用的字段少而稳：全是扁平数组与标量）。 */
export interface ParityLu {
  /** 行主序展平的 L（下三角 + 单位对角）。 */
  l: number[];
  /** 行主序展平的 U（上三角）。 */
  u: number[];
  /** `perm[k]`：第 k 步之后的原始行号。 */
  perm: number[];
  /** 每一步选中的主元行号（该步交换前的行号）。 */
  pivots: number[];
  /** 实际行交换：`[k0, p0, k1, p1, …]`（没有交换时为空数组）。 */
  rowSwaps: number[];
  /** 增长因子 `max|U| / max|A|`。 */
  rho: number;
}

/** 行主序方阵 → 二维数组（只在本模块内部用，行主序是跨语言约定的存储顺序）。 */
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
 * 部分主元 LU 分解（**与 Python 侧同一套运算顺序**）。
 *
 * @param a 行主序方阵，长度必须是 `n*n`
 * @param n 阶数
 */
export function luPartialPivot(a: ArrayLike<number>, n: number): ParityLu {
  if (!Number.isInteger(n) || n <= 0) throw new Error(`阶数 n 必须是正整数，得到 ${String(n)}`);
  if (a.length !== n * n) throw new Error(`矩阵长度 ${a.length} ≠ n*n = ${n * n}（行主序展平？）`);
  const m = toRows(a, n);
  const perm: number[] = [];
  for (let i = 0; i < n; i++) perm.push(i);
  const pivots: number[] = [];
  const rowSwaps: number[] = [];

  for (let k = 0; k < n - 1; k++) {
    // 选主元：|M[i][k]| 最大者，并列取**最小行号**（与 np.argmax 同语义，可复现）
    let piv = k;
    let best = Math.abs(m[k]![k] as number);
    for (let i = k + 1; i < n; i++) {
      const v = Math.abs(m[i]![k] as number);
      if (v > best) {
        best = v;
        piv = i;
      }
    }
    if (piv !== k) {
      const tmp = m[k] as number[];
      m[k] = m[piv] as number[];
      m[piv] = tmp;
      const t = perm[k] as number;
      perm[k] = perm[piv] as number;
      perm[piv] = t;
      rowSwaps.push(k, piv);
    }
    pivots.push(piv);

    const pivotValue = m[k]![k] as number;
    if (pivotValue === 0) continue; // 精确零主元：跳过该列（与 Python 侧一致）

    // 乘子（写回第 k 列的下三角）
    const mult: number[] = new Array<number>(n - k - 1);
    for (let i = k + 1; i < n; i++) {
      const v = (m[i]![k] as number) / pivotValue;
      m[i]![k] = v;
      mult[i - k - 1] = v;
    }
    // 外积更新：M[k+1:, k+1:] -= outer(mult, M[k, k+1:])
    for (let i = k + 1; i < n; i++) {
      const mi = mult[i - k - 1] as number;
      for (let j = k + 1; j < n; j++) {
        m[i]![j] = (m[i]![j] as number) - mi * (m[k]![j] as number);
      }
    }
  }

  const l: number[] = new Array<number>(n * n);
  const u: number[] = new Array<number>(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const v = m[i]![j] as number;
      l[i * n + j] = i > j ? v : i === j ? 1 : 0;
      u[i * n + j] = i <= j ? v : 0;
    }
  }

  let maxA = 0;
  let maxU = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const va = Math.abs(a[i * n + j] as number);
      if (va > maxA) maxA = va;
      const vu = Math.abs(u[i * n + j] as number);
      if (vu > maxU) maxU = vu;
    }
  }
  const rho = maxA === 0 ? 0 : maxU / maxA;

  return { l, u, perm, pivots, rowSwaps, rho };
}
