/**
 * `viz/shared/src/numeric/power_iter.ts` 的单元测试（G7 对拍之外的本地护栏）。
 *
 * 对拍（G7）证明的是"两侧同结果"，但它依赖 golden 与 TS 报告都在场；
 * 这里补上**不依赖 Python** 的性质检查，分四类：
 *
 * 1. **契约/边界**：零向量起点、`A x = 0`、`max_iter < 1`、长度不符——全部要响亮报错或走对分支；
 * 2. **退化**：零矩阵（`nx == 0` 的提前 break）、幂零矩阵（`A x = 0` 但起点不在零空间）、
 *    全 1 矩阵（起点恰好落在零空间）——三条都把"预分配的 0 留在历史里"这个语义钉死；
 * 3. **物理/结构不变量**：返回值必须自洽（末向量归一化、Rayleigh 商 = vᵀAv/vᵀv、
 *    残差 = ‖Av − λv‖）、收敛族残差单调下降、`ratios` 收敛到 |λ₂/λ₁|、
 *    旋转块的夹角平台停在 2π/5（复主特征值）；
 * 4. **确定性**：同一输入连跑两次逐位相同（无隐藏随机/时间依赖）。
 *
 * 矩阵一律行主序展平（跨语言约定，见 `numeric/power_iter.ts` 顶部的对应表）。
 * 边界用例的"Python 侧应当是什么样"都用 `numcore/linalg/eigen.py::power_iteration`
 * 实测过一遍（结果写在注释里），避免把 TS 的 bug 写成"契约"。
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { powerIteration, type ParityPowerIter } from "../shared/src/numeric/power_iter.js";

/** 从仓库根读一份对拍 golden（`--root` 之外的测试不猜路径：向上找 tests/parity）。 */
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

interface PowerGolden {
  inputs: { n: number; a: number[]; x0: number[]; max_iter: number; tol: number; kind: string };
  meta: { n_iter: number; converged: boolean; is_complex_pair: boolean };
}

function golden(caseName: string): PowerGolden {
  return JSON.parse(
    readFileSync(join(rootDir(), "tests", "parity", "golden", `${caseName}.json`), "utf8"),
  ) as PowerGolden;
}

/** 按 golden 的输入跑一遍（矩阵、起点、上限、阈值全部来自 Python 落盘的 double）。 */
function runGolden(caseName: string): ParityPowerIter {
  const g = golden(caseName);
  return powerIteration(g.inputs.a, g.inputs.n, {
    x0: g.inputs.x0,
    maxIter: g.inputs.max_iter,
    tol: g.inputs.tol,
  });
}

// --------------------------------------------------------------------------- //
// 测试自带的小工具（与实现**故意写得不一样**：这里用朴素定义，实现走 max 缩放范数）
// --------------------------------------------------------------------------- //

const norm2 = (v: number[]): number => Math.sqrt(v.reduce((s, x) => s + x * x, 0));

/** `A @ x`（行主序密集写法，只给测试用）。 */
function matvec(a: number[], n: number, x: number[]): number[] {
  return Array.from({ length: n }, (_, i) => {
    let s = 0;
    for (let j = 0; j < n; j++) s += (a[i * n + j] as number) * (x[j] as number);
    return s;
  });
}

/** 任意向量的 Rayleigh 商 `xᵀAx/xᵀx`（测试侧独立实现）。 */
function rayleigh(a: number[], n: number, x: number[]): number {
  const ax = matvec(a, n, x);
  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i++) {
    num += (x[i] as number) * (ax[i] as number);
    den += (x[i] as number) * (x[i] as number);
  }
  return num / den;
}

/** `‖Ax − λx‖₂`（测试侧独立实现）。 */
function residualOf(a: number[], n: number, lam: number, x: number[]): number {
  const ax = matvec(a, n, x);
  return norm2(Array.from({ length: n }, (_, i) => (ax[i] as number) - lam * (x[i] as number)));
}

// --------------------------------------------------------------------------- //
// 1. golden 驱动的性质检查
// --------------------------------------------------------------------------- //

test("power_iter_symmetric：对角族 λ₁ = 1、实测比值收敛到 |λ₂/λ₁| = 0.35", () => {
  const g = golden("power_iter_symmetric");
  const res = runGolden("power_iter_symmetric");
  // A = diag(1, 0.35, …, 0.35^11)：主特征值**精确**是 1（对角元），Rayleigh 商必须咬住它
  assert.ok(Math.abs(res.eigenvalue - 1.0) < 1e-12, `λ₁ = ${res.eigenvalue}`);
  assert.equal(res.converged, true);
  assert.equal(res.nIter, g.meta.n_iter, "停止点必须与 Python 一致（变长数组差一步就长度不符）");
  // 最后一跳的实测比值 = |λ₂/λ₁| = 0.35（这条族的残差没有相消，比值是干净量）
  const last = res.ratios[res.ratios.length - 1] as number;
  assert.ok(Math.abs(last - 0.35) < 1e-9, `最后一跳比值 = ${last}（应 ≈ 0.35）`);
  // 收敛族的残差在**瞬态之后**单调下降；但第 0 → 1 步会先涨（实测 0.185 → 0.332）：
  // 幂法第一步先把向量转向主方向，Rayleigh 商跳得快、‖Ax − λx‖ 反而可能变大。
  // "看起来应该单调"不是契约，所以只钉住瞬态之后的尾部。
  assert.ok((res.residual[1] as number) > (res.residual[0] as number), "第 1 步残差先涨（瞬态）");
  for (let i = 2; i < res.residual.length; i++) {
    assert.ok((res.residual[i] as number) <= (res.residual[i - 1] as number), `残差第 ${i} 步反弹`);
  }
});

test("power_iter_tridiag：三对角族收敛、比值落在 0.6 附近（尾部被相消噪声主导）", () => {
  const g = golden("power_iter_tridiag");
  const res = runGolden("power_iter_tridiag");
  assert.equal(res.converged, true);
  assert.equal(res.nIter, g.meta.n_iter);
  assert.ok(res.nIter < g.inputs.max_iter, "应当在 max_iter 之前停");
  const last = res.ratios[res.ratios.length - 1] as number;
  // 只做量级断言：深收敛区的末几个比值带 ~1e-6 的相消噪声（见 tolerances.yaml 的 reason）
  assert.ok(Math.abs(last - 0.60047) < 1e-3, `最后一跳比值 = ${last}`);
});

test("power_iter_shear：非正规亏格族 200 步不收敛，残差停在 1e-4 量级", () => {
  const g = golden("power_iter_shear");
  const res = runGolden("power_iter_shear");
  assert.equal(res.converged, false, "shear_matrix 的特征值全是 1（几何重数 1），幂法不该收敛");
  assert.equal(res.nIter, g.inputs.max_iter, "没收敛就必须跑满 max_iter");
  assert.equal(res.isComplexPair, false, "实谱不该报复主特征值");
  const last = res.residual[res.residual.length - 1] as number;
  assert.ok(last > 1e-5 && last < 1e-3, `末残差 = ${last}（shear(6, 0.4) 实测 2.66e-4）`);
  assert.ok(res.residual.every((v) => Number.isFinite(v)));
});

test("power_iter_angle_plateau：夹角序列长度与 nIter 自洽、平台远低于复特征值判据", () => {
  const g = golden("power_iter_angle_plateau");
  const res = runGolden("power_iter_angle_plateau");
  assert.equal(res.angleStep.length, g.inputs.max_iter + 1);
  assert.equal(res.angleStep[0], 0, "第 0 个夹角是起点与自己，恒为 0");
  const tail = res.angleStep.slice(res.angleStep.length - 50);
  const plateau = tail.reduce((s, v) => s + v, 0) / tail.length;
  assert.ok(plateau < 1e-3, `末段夹角平台 = ${plateau}（shear 是实谱：夹角应当趋于 0）`);
  assert.equal(res.isComplexPair, false);
});

test("power_iter_ratio_floor：比值序列与另一次运行逐位一致（同一输入 → 同一输出）", () => {
  const a = runGolden("power_iter_ratio_floor");
  const b = runGolden("power_iter_ratio_floor");
  assert.deepEqual(a.ratios, b.ratios);
  assert.equal(a.ratios.length, 55, "nIter = 56 且全部残差为正 → 55 个一阶比值");
});

// --------------------------------------------------------------------------- //
// 2. 结构不变量（不依赖 golden 的输出）
// --------------------------------------------------------------------------- //

for (const name of ["power_iter_symmetric", "power_iter_tridiag", "power_iter_shear"]) {
  test(`power_iter(${name})：返回值自洽（归一化 / Rayleigh 商 / 残差 / 长度）`, () => {
    const g = golden(name);
    const res = runGolden(name);
    const n = g.inputs.n;
    assert.equal(res.vector.length, n);
    assert.ok(Math.abs(norm2(res.vector) - 1) < 1e-15, `‖v‖₂ = ${norm2(res.vector)}（必须归一化）`);
    assert.equal(res.rayleigh.length, res.nIter + 1);
    assert.equal(res.residual.length, res.nIter + 1);
    assert.equal(res.angleStep.length, res.nIter + 1);
    assert.equal(res.ratios.length, res.nIter - 1, "正残差有 nIter+1 个 → 丢掉第一步瞬态后剩 nIter−1 个");
    // 末向量的 Rayleigh 商与残差必须与返回值一致（独立实现，容差给到 ~45 ulp）
    const lam = rayleigh(g.inputs.a, n, res.vector);
    assert.ok(Math.abs(lam - res.eigenvalue) < 1e-12, `重算 λ = ${lam} ≠ ${res.eigenvalue}`);
    const rs = residualOf(g.inputs.a, n, res.eigenvalue, res.vector);
    const reported = res.residual[res.residual.length - 1] as number;
    assert.ok(Math.abs(rs - reported) <= 1e-12 * Math.max(1, Math.abs(reported)), `重算残差 ${rs} ≠ ${reported}`);
    assert.ok(res.vector.every((v) => Number.isFinite(v)));
  });
}

test("确定性：同一输入连跑两次逐位相同（无隐藏随机 / 时间依赖）", () => {
  for (const name of ["power_iter_symmetric", "power_iter_tridiag", "power_iter_shear"]) {
    const a = runGolden(name);
    const b = runGolden(name);
    assert.deepEqual(a, b, name);
  }
});

// --------------------------------------------------------------------------- //
// 3. 退化与边界（"Python 侧是什么样"都已实测过，写在注释里）
// --------------------------------------------------------------------------- //

test("零矩阵：A x₀ = 0 走提前 break，历史长度 1 且不谎报收敛", () => {
  // numcore 实测：n_iter = 0、converged = False、rayleigh = [0.0]、residual = [0.0]、ratios = []
  // （停止判据只在循环里判，不判起点——所以"残差 0 但没收敛"是有意保留的语义）
  const res = powerIteration([0, 0, 0, 0], 2, { x0: [1, 2] });
  assert.equal(res.nIter, 0);
  assert.equal(res.converged, false);
  assert.deepEqual(res.rayleigh, [0]);
  assert.deepEqual(res.residual, [0]);
  assert.deepEqual(res.angleStep, [0]);
  assert.deepEqual(res.ratios, []);
  assert.ok(Math.abs(norm2(res.vector) - 1) < 1e-15, "起点仍然被归一化");
});

test("幂零矩阵 [[0,1],[0,0]]、起点 [0,1]：第一步就把残差压到 0", () => {
  // numcore 实测：n_iter = 1、converged = True、rayleigh = [0, 0]、residual = [1, 0]、
  // angleStep = [0, π/2]、is_complex_pair = True（夹角平台 90° 触发诊断标志）
  const res = powerIteration([0, 1, 0, 0], 2, { x0: [0, 1] });
  assert.equal(res.nIter, 1);
  assert.equal(res.converged, true);
  assert.deepEqual(res.rayleigh, [0, 0]);
  assert.deepEqual(res.residual, [1, 0]);
  assert.deepEqual(res.vector, [1, 0]);
  assert.ok(Math.abs((res.angleStep[1] as number) - Math.PI / 2) < 1e-15);
  assert.equal(res.isComplexPair, true);
});

test("全 1 矩阵 + 起点 [1,−1]：起点正好落在零空间，与零矩阵同分支", () => {
  const res = powerIteration([1, 1, 1, 1], 2, { x0: [1, -1] });
  assert.equal(res.nIter, 0);
  assert.equal(res.converged, false);
  assert.deepEqual(res.residual, [0]);
});

test("旋转块：实向量在 span{v, v̄} 里绕圈 —— 复主特征值的活标本", () => {
  // 2π/5 旋转：复特征值 e^{±iθ}，实向量永远转不出去，夹角平台恰好停在 θ = 2π/5
  const th = (2 * Math.PI) / 5;
  const c = Math.cos(th);
  const s = Math.sin(th);
  const res = powerIteration([c, -s, s, c], 2, { x0: [1, 0], maxIter: 12 });
  assert.equal(res.converged, false, "Rayleigh 商恒为 cos θ，残差停在 sin θ，判据永远不满足");
  assert.equal(res.isComplexPair, true);
  for (let i = 1; i < res.angleStep.length; i++) {
    assert.ok(Math.abs((res.angleStep[i] as number) - th) < 1e-12, `第 ${i} 步夹角 ≠ 2π/5`);
  }
  assert.ok(Math.abs(res.eigenvalue - c) < 1e-15, "Rayleigh 商 = cos θ（不是特征值——图上要看得出来）");
});

test("输入契约：零向量起点 / 阶数 / 矩阵长度 / max_iter 全部响亮报错", () => {
  assert.throws(() => powerIteration([1, 0, 0, 1], 2, { x0: [0, 0] }), /零向量无法归一化/);
  assert.throws(() => powerIteration([1], 0, { x0: [1] }), /阶数 n 必须是正整数/);
  assert.throws(() => powerIteration([1], 1.5, { x0: [1] }), /阶数 n 必须是正整数/);
  assert.throws(() => powerIteration([1, 2, 3], 2, { x0: [1, 0] }), /矩阵长度 3 ≠ n\*n = 4/);
  assert.throws(() => powerIteration([1], 1, { x0: [1], maxIter: 0 }), /maxIter 至少为 1/);
  assert.throws(() => powerIteration([1, 0, 0, 1], 2, { x0: [1, 0, 0] }), /起点长度 3 ≠ n = 2/);
});

test("maxIter 缺省与 Python 一致（200），tol 缺省 1e-12", () => {
  const g = golden("power_iter_symmetric");
  // 不传 maxIter/tol：对角族自己会在 27 步收敛（阈值不同则停止点不同，这里两处都缺省）
  const res = powerIteration(g.inputs.a, g.inputs.n, { x0: g.inputs.x0 });
  assert.equal(res.nIter, 27);
  assert.equal(res.converged, true);
});
