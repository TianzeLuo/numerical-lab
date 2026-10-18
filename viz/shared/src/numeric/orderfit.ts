/**
 * 收敛阶拟合的 TS 侧对拍薄包装。
 *
 * **本目录只服务 G7 双实现对拍，不是渲染路径。**
 * `viz/chapters/**` MUST NOT 直接 import 本目录（P8）；双对数图的渲染路径是
 * `viz/shared/src/loglogfit.ts` 的 `fitOrder`（本模块就是它的薄包装）。
 *
 * 为什么要有这一层
 * ----------------
 * `fitOrder` 返回的 `OrderFit` 带 `stderr`，而 Python 的 `OrderFit` 还带 `h` / `err` 数组。
 * 跨语言逐字段比较时，两边都只暴露**同一小组标量**最不容易漂移：
 *
 * ```
 * {slope, intercept, r2, nUsed, nDropped}
 * ```
 *
 * 与 `numcore/verify/order.py::fit_order` 的字段对应：
 *
 * | Python | 本模块 |
 * |---|---|
 * | `slope` | `slope` |
 * | `intercept` | `intercept`（自然对数底） |
 * | `r2` | `r2`（`SStot = 0` 时取 1） |
 * | `n_used` | `nUsed` |
 * | `n_dropped` | `nDropped`（含 `skip` / `head` / `tail` 丢掉的） |
 *
 * 选项名映射：Python `min_points=` ↔ TS `minPoints=`；`skip` / `head` / `tail` 同名。
 * 语义（含"`head` 保留粗网格端、`tail` 保留细网格端"的区别）全部由 `loglogfit.fitOrder` 承担。
 */

import { fitOrder } from "../loglogfit.js";
import type { FitOptions } from "../loglogfit.js";

/** 跨语言比对用的拟合结果（字段少而稳，便于逐字段对拍）。 */
export interface ParityOrderFit {
  /** 拟合斜率（实测收敛阶）。 */
  slope: number;
  /** 截距 `log C`（自然对数底）。 */
  intercept: number;
  /** 决定系数 `1 − SSres/SStot`（`SStot = 0` 时取 1）。 */
  r2: number;
  /** 参与拟合的点数。 */
  nUsed: number;
  /** 被丢弃的点数（`skip`/`head`/`tail`/非正/非有限）。 */
  nDropped: number;
}

/**
 * 对 `err ≈ C·h^slope` 做双对数最小二乘，只返回跨语言可比的 5 个字段。
 *
 * 抛错条件与 `fitOrder`（以及 Python `fit_order`）完全一致：形状不一致、空输入、
 * 有效点少于 `max(minPoints, 2)`、所有 `h` 相同。
 */
export function fitOrderParity(h: ArrayLike<number>, err: ArrayLike<number>, opts: FitOptions = {}): ParityOrderFit {
  const fit = fitOrder(h, err, opts);
  return {
    slope: fit.slope,
    intercept: fit.intercept,
    r2: fit.r2,
    nUsed: fit.nUsed,
    nDropped: fit.nDropped,
  };
}
