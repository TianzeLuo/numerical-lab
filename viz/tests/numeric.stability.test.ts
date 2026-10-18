/**
 * `viz/shared/src/numeric/stability.ts` 的单元测试（G7 对拍之外的本地护栏）。
 *
 * 对拍（G7）证明的是"两侧同结果"，但它依赖 golden 与 TS 报告都在场；
 * 这里补上**不依赖 Python** 的性质检查，分四类：
 *
 * 1. **系数真源**：`rk_stability_coefficients` 走 Butcher 表现场算（`bᵀA^{j-1}e`）、
 *    `taylor_stability_coefficients` 走 `1/j!` —— 两条路径必须给出同一组系数，
 *    否则"RK4 的 R 等于截断指数"就只是个假设；
 * 2. **闭式对拍**：`R(z)` 在 golden 的 24 个 z 上必须等于独立写出的闭式
 *    （`1+z`、`1/(1−z)`、`(1+z/2)/(1−z/2)`、BDF2 的判别式），且全 1 矩阵/全 0 矩阵那种
 *    "退化信号"不许悄悄溜过去；
 * 3. **多步法根模**：BDF-1 的根模必须等于 `1/|1−z|`、BDF-2 必须等于闭式判别式
 *    （同一模块里两条独立代码路径互拍）；极点（`z = α_k`）必须是 `+∞` 而不是 NaN；
 * 4. **确定性与契约**：连跑两次逐位相同、按点分块求值与整块求值逐位相同、
 *    退化输入响亮报错。
 *
 * 复数一律写成 `{re, im}` 两条实数组（跨语言 JSON 只认实数，见模块头）。
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";

import {
  bdfCoefficients,
  complexAbs,
  lmmRootModulus,
  rkStabilityCoefficients,
  stabilityFunction,
  taylorStabilityCoefficients,
  type ComplexGrid,
} from "../shared/src/numeric/stability.js";

// --------------------------------------------------------------------------- //
// 测试自带的复数原语（**故意与实现不同**：除法用共轭法、开方用极坐标，
// 这样"独立对拍"才不是把同一段代码抄两遍）
// --------------------------------------------------------------------------- //

/** `(a+bi)/(c+di)`：分子分母同乘共轭（与实现的 Smith 算法是两条路）。 */
function cdivConj(ar: number, ai: number, br: number, bi: number): [number, number] {
  const d = br * br + bi * bi;
  return [(ar * br + ai * bi) / d, (ai * br - ar * bi) / d];
}

/** 主平方根（极坐标形式：`√r·e^{iθ/2}`，与实现的代数公式是两条路）。 */
function csqrtPolar(ar: number, ai: number): [number, number] {
  const r = Math.hypot(ar, ai);
  const th = Math.atan2(ai, ar);
  return [Math.sqrt(r) * Math.cos(th / 2), Math.sqrt(r) * Math.sin(th / 2)];
}

/** 从仓库根读一份对拍 golden（向上找含 tests/parity/cases.json 的那一层）。 */
function rootDir(): string {
  let dir = process.cwd();
  for (let i = 0; i < 8; i++) {
    try {
      readFileSync(join(dir, "tests", "parity", "cases.json"));
      return dir;
    } catch {
      dir = join(dir, "..");
    }
  }
  throw new Error("找不到仓库根（含 tests/parity/cases.json 的那一层）");
}

interface StabilityGolden {
  inputs: { op: string; z_re: number[]; z_im: number[]; orders?: number[]; iters?: number };
}

function golden(caseName: string): StabilityGolden {
  return JSON.parse(
    readFileSync(join(rootDir(), "tests", "parity", "golden", `${caseName}.json`), "utf8"),
  ) as StabilityGolden;
}

/** golden 里的 24 点 z 网格（由 Python 生成落盘，两侧读回同一个 double）。 */
function goldenGrid(): ComplexGrid {
  const g = golden("stability_function_grid");
  return { re: g.inputs.z_re, im: g.inputs.z_im };
}

/** `|a − b| ≤ atol + rtol|b|`（与 tools/parity.py 同口径）。 */
function close(actual: number, expected: number, rtol = 1e-12, atol = 1e-15, what = ""): void {
  const limit = atol + rtol * Math.abs(expected);
  assert.ok(Math.abs(actual - expected) <= limit, `${what}: ${actual} vs ${expected}（limit=${limit}）`);
}

// --------------------------------------------------------------------------- //
// 1. 系数真源：Butcher 表 vs 1/j!
// --------------------------------------------------------------------------- //

test("rk4 的稳定多项式系数（由 Butcher 表现场算）等于 1/j!（阶条件的推论）", () => {
  const fromTable = rkStabilityCoefficients("rk4");
  const fromTaylor = taylorStabilityCoefficients(4);
  assert.equal(fromTable.length, 5);
  for (let j = 0; j <= 4; j++) {
    close(fromTable[j] as number, fromTaylor[j] as number, 1e-15, 1e-16, `c${j}`);
  }
  // 1/1, 1/1, 1/2, 1/6, 1/24 在 float64 里都是"整除以阶乘"，逐位可比
  assert.deepEqual(fromTaylor, [1, 1, 0.5, 1 / 6, 1 / 24]);
});

test("低阶表的系数：forward_euler [1,1]、midpoint/heun [1,1,1/2]", () => {
  assert.deepEqual(rkStabilityCoefficients("forward_euler"), [1, 1]);
  assert.deepEqual(rkStabilityCoefficients("midpoint"), [1, 1, 0.5]);
  assert.deepEqual(rkStabilityCoefficients("heun"), [1, 1, 0.5]);
});

test("显式 RK 的 R 是多项式：系数表长度 = 级数 + 1，末项非零", () => {
  for (const [name, stages] of [["forward_euler", 1], ["midpoint", 2], ["heun", 2], ["rk4", 4]] as const) {
    const c = rkStabilityCoefficients(name);
    assert.equal(c.length, stages + 1, name);
    assert.notEqual(c[c.length - 1], 0, `${name} 的最高次系数不该是 0（A^s = 0 决定截断位置）`);
  }
});

test("BDF 系数表：α_k 的高次项 = 1, 3/2, 11/6, 25/12, 137/60, 49/20（Hairer & Wanner §III.1）", () => {
  const leading = [1.0, 1.5, 11 / 6, 25 / 12, 137 / 60, 49 / 20];
  for (let k = 1; k <= 6; k++) {
    const { alpha, betaK } = bdfCoefficients(k);
    assert.equal(alpha.length, k + 1);
    assert.equal(alpha[k], leading[k - 1], `BDF-${k} 的首项系数`);
    assert.equal(betaK, 1.0);
    // 归一化约定：Σα_j = 0（常值解必须精确满足），BDF-k 满足 ρ(1) = 0
    const sum = alpha.reduce((s, v) => s + v, 0);
    assert.ok(Math.abs(sum) < 1e-15, `BDF-${k} 的 Σα_j = ${sum}（应当为 0）`);
  }
});

test("未知方法与越界阶数响亮报错（`dopri5` 不在真源里，TS 侧也不许自造）", () => {
  // numcore/ode/stability.py::METHODS 没有 dopri5（那是 3.1 章的自适应 RK 表，没有解析稳定多项式）。
  // 这里把它钉成"明确不支持"，免得有人为了凑 case 在 TS 侧编一个 Python 没有的公式。
  assert.throws(() => rkStabilityCoefficients("dopri5"), /未知显式 RK 表/);
  assert.throws(() => stabilityFunction("dopri5", 5, { re: [0], im: [0] }), /未知方法/);
  assert.throws(() => stabilityFunction("bdf", 7, { re: [0], im: [0] }), /BDF 阶数只实现 1\.\.6/);
  assert.throws(() => taylorStabilityCoefficients(0), /阶数至少为 1/);
});

// --------------------------------------------------------------------------- //
// 2. R(z) 的闭式对拍（在 golden 的 z 网格上）
// --------------------------------------------------------------------------- //

test("stability_function_grid：R(z) 逐点等于独立写出的闭式", () => {
  const z = goldenGrid();
  const n = z.re.length;
  assert.equal(n, 24);

  const fe = stabilityFunction("forward_euler", 1, z); // 1 + z
  const ie = stabilityFunction("implicit_euler", 1, z); // 1/(1−z)
  const trap = stabilityFunction("trapezoidal", 2, z); // (1+z/2)/(1−z/2)
  const bdf1 = stabilityFunction("bdf", 1, z);
  const bdf2 = stabilityFunction("bdf", 2, z); // (2+√(1+2z))/(3−2z)

  for (let i = 0; i < n; i++) {
    const zr = z.re[i] as number;
    const zi = z.im[i] as number;

    close(fe.re[i] as number, 1 + zr, 1e-15, 1e-16, `fe.re[${i}]`);
    close(fe.im[i] as number, zi, 1e-15, 1e-16, `fe.im[${i}]`);

    // 隐式 Euler / BDF-1：与 1−z 共轭同乘（不用实现里的 Smith 除法）
    const [ir, ii] = cdivConj(1, 0, 1 - zr, -zi);
    close(ie.re[i] as number, ir, 1e-13, 1e-15, `ie.re[${i}]`);
    close(ie.im[i] as number, ii, 1e-13, 1e-15, `ie.im[${i}]`);
    close(bdf1.re[i] as number, ir, 1e-13, 1e-15, `bdf1.re[${i}]`);
    close(bdf1.im[i] as number, ii, 1e-13, 1e-15, `bdf1.im[${i}]`);

    // 梯形法：(1+z/2)/(1−z/2)
    const [tr, ti] = cdivConj(1 + zr / 2, zi / 2, 1 - zr / 2, -zi / 2);
    close(trap.re[i] as number, tr, 1e-13, 1e-15, `trap.re[${i}]`);
    close(trap.im[i] as number, ti, 1e-13, 1e-15, `trap.im[${i}]`);

    // BDF-2：(2 + √(1+2z))/(3 − 2z)，开方走极坐标
    const [sr, si] = csqrtPolar(1 + 2 * zr, 2 * zi);
    const [br, bi] = cdivConj(2 + sr, si, 3 - 2 * zr, -2 * zi);
    close(bdf2.re[i] as number, br, 1e-12, 1e-14, `bdf2.re[${i}]`);
    close(bdf2.im[i] as number, bi, 1e-12, 1e-14, `bdf2.im[${i}]`);
  }
});

test("rk4 与 explicit_taylor(4) 在网格上给出同一个 R（两条系数路径互相印证）", () => {
  const z = goldenGrid();
  const a = stabilityFunction("rk4", 4, z);
  const b = stabilityFunction("explicit_taylor", 4, z);
  for (let i = 0; i < z.re.length; i++) {
    close(a.re[i] as number, b.re[i] as number, 1e-14, 1e-16, `re[${i}]`);
    close(a.im[i] as number, b.im[i] as number, 1e-14, 1e-16, `im[${i}]`);
  }
});

test("|R| 与 re/im 自洽，且 z = 0 处 R ≡ 1", () => {
  const z = goldenGrid();
  for (const [method, order] of [
    ["forward_euler", 1],
    ["heun", 2],
    ["rk4", 4],
    ["explicit_taylor", 4],
    ["implicit_euler", 1],
    ["trapezoidal", 2],
    ["bdf", 1],
    ["bdf", 2],
  ] as const) {
    const r = stabilityFunction(method, order, z);
    const abs = complexAbs(r);
    assert.equal(r.re.length, z.re.length);
    for (let i = 0; i < z.re.length; i++) {
      close(abs[i] as number, Math.hypot(r.re[i] as number, r.im[i] as number), 1e-15, 1e-16, "三者");
    }
    close(r.re[0] as number, 1.0, 1e-15, 1e-16, `${method}/${order} 的 R(0).re`);
    close(r.im[0] as number, 0.0, 1e-15, 1e-16, `${method}/${order} 的 R(0).im`);
  }
});

test("RK4 实轴稳定边界：z = −2.785293563405782 处 |R| 恰好跨过 1", () => {
  const z = goldenGrid();
  const i = z.re.indexOf(-2.785293563405782);
  assert.ok(i > 0, "网格里必须有 RK4 实轴边界点");
  const r = stabilityFunction("rk4", 4, z);
  const abs = complexAbs(r);
  const onBoundary = abs[i] as number;
  assert.ok(Math.abs(onBoundary - 1) < 1e-9, `|R(z*)| = ${onBoundary}`);
  // 再往左一点就冲出稳定域：|R| > 1
  const beyond = stabilityFunction("rk4", 4, { re: [-3.0], im: [0] });
  assert.ok((complexAbs(beyond)[0] as number) > 1, "z = −3 已在 RK4 稳定域外");
});

// --------------------------------------------------------------------------- //
// 3. 极点：|R| = +∞ 是有意义的极限值，不许是 NaN
// --------------------------------------------------------------------------- //

test("极点处 |R| = +∞（隐式 Euler/BDF-1 在 z=1、BDF-2 在 z=1.5、梯形法在 z=2）", () => {
  const cases: [string, number, number][] = [
    ["implicit_euler", 1, 1.0],
    ["trapezoidal", 2, 2.0],
    ["bdf", 1, 1.0],
    ["bdf", 2, 1.5],
  ];
  for (const [method, order, pole] of cases) {
    const r = stabilityFunction(method, order, { re: [pole], im: [0] });
    const mod = complexAbs(r)[0] as number;
    assert.equal(mod, Number.POSITIVE_INFINITY, `${method}/${order} 在 z=${pole} 的 |R|`);
    assert.ok(!Number.isNaN(mod), "极点必须是 +∞，不能是 NaN");
  }
  // 离极点很近但完全正常的点仍然是有限真值（相对判据要能区分这两者）
  const near = stabilityFunction("bdf", 1, { re: [1 - 1e-13], im: [0] });
  const nearMod = complexAbs(near)[0] as number;
  assert.ok(Number.isFinite(nearMod) && nearMod > 1e12, `z = 1 − 1e-13 处 |R| = ${nearMod}`);
});

// --------------------------------------------------------------------------- //
// 4. 多步法根模：同一模块里的两条独立路径互拍
// --------------------------------------------------------------------------- //

test("BDF-1 的根模等于闭式 1/|1−z|（Durand–Kerner 路径 vs 有理式路径）", () => {
  const { alpha, betaK } = bdfCoefficients(1);
  const z: ComplexGrid = { re: [-0.5, -1, -2, 0.5, 1.0, 2.5], im: [0, 0, 0, 1.5, -0.5, 0.75] };
  const { mod } = lmmRootModulus(alpha, betaK, z);
  for (let i = 0; i < z.re.length; i++) {
    const expected = 1 / Math.hypot(1 - (z.re[i] as number), -(z.im[i] as number));
    close(mod[i] as number, expected, 1e-12, 1e-15, `mod[${i}]`);
  }
});

test("BDF-2 的根模等于闭式 max|2±√(1+2z)|/|3−2z|（重根点除外，见下一条）", () => {
  const { alpha, betaK } = bdfCoefficients(2);
  const z = goldenGrid();
  const { mod, residual } = lmmRootModulus(alpha, betaK, z, { withResidual: true });
  assert.ok(residual !== null);
  for (let i = 0; i < z.re.length; i++) {
    // z = −0.5 处判别式 1+2z = 0：多项式是 (r−1/2)²，**二重根**。
    // DK 在重根处只有线性收敛，48 次迭代后模还是 0.5+8.6e-12（见下一条测试），
    // 那不是实现错误，是重根处的固有退化——所以闭式对照要跳过这个点。
    if ((z.re[i] as number) === -0.5 && (z.im[i] as number) === 0) continue;
    const [sr, si] = csqrtPolar(1 + 2 * (z.re[i] as number), 2 * (z.im[i] as number));
    const den = Math.hypot(3 - 2 * (z.re[i] as number), -2 * (z.im[i] as number));
    const expected = Math.max(Math.hypot(2 + sr, si), Math.hypot(2 - sr, si)) / den;
    close(mod[i] as number, expected, 1e-11, 1e-14, `mod[${i}]`);
    assert.ok((residual[i] as number) < 1e-12, `DK 残差[${i}] = ${residual[i]}（48 次迭代应当到舍入地板）`);
  }
});

test("重根处的 DK：残差小 ≠ 根准（误差被平方掉），且模对 1 ulp 的 z 扰动就跳", () => {
  const { alpha, betaK } = bdfCoefficients(2);
  const { mod, residual } = lmmRootModulus(alpha, betaK, { re: [-0.5], im: [0] }, { withResidual: true });
  // 根在 1/2 附近（真值恰好 0.5），但 DK 在重根处只有线性收敛，48 次只到 ~1e-11
  assert.ok(Math.abs((mod[0] as number) - 0.5) < 1e-9, `重根处 mod = ${mod[0]}`);
  assert.notEqual(mod[0], 0.5, "不该假装精确命中（真值 1/2，DK 只到 1e-11）");
  // 残差 = |p(r)|/scale，而 p 在重根附近 ~ (r−1/2)²：误差被平方 → 残差 1e-20 时根还差 1e-11
  assert.ok((residual?.[0] as number) < 1e-18, `重根处残差 = ${residual?.[0]}（这正是"残差会骗你"的地方）`);
  // 重根让迭代对舍入混沌：z 挪 2 ulp，模就跳 ~1e-11（Python 侧实测同量级）
  const shifted = lmmRootModulus(alpha, betaK, { re: [-0.5 - 2 * Number.EPSILON], im: [0] });
  assert.ok(
    Math.abs((mod[0] as number) - (shifted.mod[0] as number)) > 1e-12,
    `z 挪 2 ulp 后模应当明显跳动：${mod[0]} vs ${shifted.mod[0]}`,
  );
});

test("稳定域的定性事实：z = −1 在 BDF-3/4 域内（模 < 1），z = 0.5+1.5i / 1+i 在域外（模 > 1）", () => {
  for (const k of [3, 4]) {
    const { alpha, betaK } = bdfCoefficients(k);
    const { mod } = lmmRootModulus(alpha, betaK, { re: [-1, 0.5, 1], im: [0, 1.5, 1] });
    assert.ok((mod[0] as number) < 1, `BDF-${k} 在 z=−1 的模 = ${mod[0]}`);
    assert.ok((mod[1] as number) > 1, `BDF-${k} 在 z=0.5+1.5i 的模 = ${mod[1]}`);
    assert.ok((mod[2] as number) > 1, `BDF-${k} 在 z=1+i 的模 = ${mod[2]}`);
  }
  // BDF-3 的极点在 z = 11/6：那里多项式降次、一个根跑到无穷远 → +∞，且残差记 0
  const { alpha, betaK } = bdfCoefficients(3);
  const poleRes = lmmRootModulus(alpha, betaK, { re: [11 / 6], im: [0] }, { withResidual: true });
  assert.equal(poleRes.mod[0], Number.POSITIVE_INFINITY);
  assert.equal(poleRes.residual?.[0], 0);
});

test("stabilityFunction(bdf, k≥3) 退化为稳定模：im ≡ 0 且与 lmmRootModulus 同值", () => {
  const z = goldenGrid();
  const r = stabilityFunction("bdf", 4, z);
  const { alpha, betaK } = bdfCoefficients(4);
  const { mod } = lmmRootModulus(alpha, betaK, z);
  for (let i = 0; i < z.re.length; i++) {
    assert.equal(r.im[i], 0, "多步法没有单个 R，退化的那条路径虚部必须恰好是 0");
    assert.equal(r.re[i], mod[i], "两条路径必须逐位相同（同一次 DK 迭代）");
  }
});

// --------------------------------------------------------------------------- //
// 5. 确定性与契约
// --------------------------------------------------------------------------- //

test("确定性：同一输入连跑两次逐位相同；按点分块与整块结果逐位相同", () => {
  const z = goldenGrid();
  const { alpha, betaK } = bdfCoefficients(3);
  const first = lmmRootModulus(alpha, betaK, z, { withResidual: true });
  const second = lmmRootModulus(alpha, betaK, z, { withResidual: true });
  assert.deepEqual(first.mod, second.mod);
  assert.deepEqual(first.residual, second.residual);

  // 每个 z 只依赖自己（DK 是逐点迭代，不跨点共享状态）
  for (let i = 0; i < z.re.length; i++) {
    const single = lmmRootModulus(alpha, betaK, { re: [z.re[i] as number], im: [z.im[i] as number] });
    assert.equal(single.mod[0], first.mod[i], `z[${i}] 单点与整块的结果必须逐位相同`);
  }
  const r1 = stabilityFunction("rk4", 4, z);
  const r2 = stabilityFunction("rk4", 4, z);
  assert.deepEqual(r1, r2);
});

test("契约：空网格返回空数组、退化 α 与长度不符的 z 响亮报错", () => {
  assert.deepEqual(lmmRootModulus([-1, 1], 1, { re: [], im: [] }).mod, []);
  assert.deepEqual(lmmRootModulus([-1, 1], 1, { re: [], im: [] }, { withResidual: true }).residual, []);
  assert.throws(() => lmmRootModulus([1], 1, { re: [0], im: [0] }), /至少是一步/);
  assert.throws(() => stabilityFunction("rk4", 4, { re: [0, 1], im: [0] }), /长度不一致/);
  assert.throws(() => lmmRootModulus([-1, 1], 1, { re: [0], im: [0, 1] }), /长度不一致/);
});

test("iters 越大越准（但 48 次已经到舍入地板）——BDF-4 在 z = −5 处", () => {
  const { alpha, betaK } = bdfCoefficients(4);
  const z: ComplexGrid = { re: [-5], im: [0] };
  const m48 = lmmRootModulus(alpha, betaK, z, { iters: 48, withResidual: true });
  const m240 = lmmRootModulus(alpha, betaK, z, { iters: 240, withResidual: true });
  close(m48.mod[0] as number, m240.mod[0] as number, 1e-14, 1e-16, "48 次 vs 240 次");
  assert.ok((m48.residual?.[0] as number) < 1e-14, `48 次的残差 = ${m48.residual?.[0]}`);
});
