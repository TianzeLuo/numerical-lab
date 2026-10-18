/**
 * 阻尼振子解析解的 TS 侧对拍实现。
 *
 * **本目录只服务 G7 双实现对拍，不是渲染路径。**
 * `viz/chapters/**` MUST NOT 直接 import 本目录（P8）。这里只有纯函数：无 IO、无 DOM、无第三方依赖。
 *
 * 数学定义
 * --------
 * `x'' + 2ζω x' + ω² x = 0`，初值 `x(0) = x0`、`x'(0) = v0`（缺省 `x0 = 1, v0 = 0`）。
 *
 * * **欠阻尼** `ζ < 1`：`x = e^{−ζωt}[x0·cos(ω_d t) + b·sin(ω_d t)]`，
 *   `ω_d = ω√(1−ζ²)`，`b = (v0 + ζω x0)/ω_d`；
 * * **临界** `ζ = 1`：`x = e^{−ωt}[x0 + (v0 + ω x0)t]`；
 * * **过阻尼** `ζ > 1`：`λ± = ω(−ζ ± √(ζ²−1))`，`x = c₊e^{λ₊t} + c₋e^{λ₋t}`。
 *
 * 与 Python 侧的对应关系（`numcore/smoke.py`）
 * ------------------------------------------
 * | Python | 本模块 |
 * |---|---|
 * | `damped_oscillator(t, ω, ζ, x0, v0)` | {@link dampedOscillator} |
 * | `oscillator_energy(x, v, ω)` | {@link oscillatorEnergy} |
 *
 * 分支顺序也必须一致：Python 先判 `|ζ−1| < 1e-12`（临界），再判 `ζ < 1`。
 * 于是 `ζ = 1 − 1e-13` 走的是**临界**分支——这不是笔误，而是"临界附近的解析解本身就病态"
 * 的显式表达；两侧若各自选分支，对拍就会在这一点上永久性地对不上。
 */

/** 一条采样曲线的样本（`x`、`v` 定义在同一时间网格 `t` 上）。 */
export interface OscillatorSample {
  /** 位移 `x(t)`。 */
  x: Float64Array;
  /** 速度 `v(t)`。 */
  v: Float64Array;
}

/**
 * 阻尼振子的**精确解**，返回 `{x, v}`（与 `numcore.smoke.damped_oscillator` 逐分支一致）。
 *
 * 参数校验也与 Python 一致：`ω ≤ 0` 或 `ζ < 0` 抛错（这里用 `RangeError`）。
 * 注意 Python 写的是 `if omega <= 0`，NaN 会**穿透**校验并在结果里变成 NaN——
 * 本实现刻意保持同样的口径，因为"非法输入的处理方式"也是契约的一部分。
 */
export function dampedOscillator(
  t: Float64Array,
  omega = 1.0,
  zeta = 0.05,
  x0 = 1.0,
  v0 = 0.0,
): OscillatorSample {
  if (omega <= 0) {
    throw new RangeError(`omega 必须为正，得到 ${String(omega)}`);
  }
  if (zeta < 0) {
    throw new RangeError(`zeta 不能为负，得到 ${String(zeta)}`);
  }
  const n = t.length;
  const x = new Float64Array(n);
  const v = new Float64Array(n);

  if (Math.abs(zeta - 1.0) < 1e-12) {
    // 临界阻尼：重根 λ = −ω，解为 (x0 + (v0 + ωx0)t)·e^{−ωt}
    const c = v0 + omega * x0;
    for (let i = 0; i < n; i++) {
      const tt = t[i];
      const env = Math.exp(-omega * tt);
      const poly = x0 + c * tt;
      x[i] = env * poly;
      v[i] = env * (c - omega * poly);
    }
    return { x, v };
  }

  if (zeta < 1.0) {
    // 欠阻尼：ω_d = ω√(1−ζ²)
    const wd = omega * Math.sqrt(1.0 - zeta * zeta);
    const a = x0;
    const b = (v0 + zeta * omega * x0) / wd;
    const c1 = -zeta * omega * a + wd * b;
    const c2 = -zeta * omega * b - wd * a;
    for (let i = 0; i < n; i++) {
      const tt = t[i];
      const env = Math.exp(-zeta * omega * tt);
      const cs = Math.cos(wd * tt);
      const sn = Math.sin(wd * tt);
      x[i] = env * (a * cs + b * sn);
      v[i] = env * (c1 * cs + c2 * sn);
    }
    return { x, v };
  }

  // 过阻尼：两个实根，按初值定系数
  const root = omega * Math.sqrt(zeta * zeta - 1.0);
  const lamP = -zeta * omega + root;
  const lamM = -zeta * omega - root;
  const cP = (v0 - lamM * x0) / (lamP - lamM);
  const cM = x0 - cP;
  for (let i = 0; i < n; i++) {
    const tt = t[i];
    const eP = Math.exp(lamP * tt);
    const eM = Math.exp(lamM * tt);
    x[i] = cP * eP + cM * eM;
    v[i] = cP * lamP * eP + cM * lamM * eM;
  }
  return { x, v };
}

/**
 * 比能量 `E = ½(v² + ω²x²)`（无阻尼时是守恒量，B 类断言的靶子）。
 *
 * 对阻尼振子它不是守恒量——本函数只负责**算这个表达式**，是否守恒由断言负责。
 * 与 Python 的 `0.5 * (v**2 + omega**2 * x**2)` 同序（`omega**2` 在此按 `omega*omega` 求值）。
 */
export function oscillatorEnergy(x: Float64Array, v: Float64Array, omega = 1.0): Float64Array {
  if (x.length !== v.length) {
    throw new RangeError(`x 与 v 长度不一致：${x.length} vs ${v.length}`);
  }
  const n = x.length;
  const out = new Float64Array(n);
  const w2 = omega * omega;
  for (let i = 0; i < n; i++) {
    const xx = x[i];
    const vv = v[i];
    out[i] = 0.5 * (vv * vv + w2 * (xx * xx));
  }
  return out;
}
