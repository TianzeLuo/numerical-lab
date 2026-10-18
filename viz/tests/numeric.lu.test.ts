/**
 * `viz/shared/src/numeric/lu.ts` 的单元测试（G7 对拍之外的本地护栏）。
 *
 * 对拍（G7）证明的是"两侧同结果"，但它依赖 golden 与 TS 报告都在场；
 * 这里补上**不依赖 golden** 的性质检查：重构 `PA = LU`、ρ 的经典值、置换合法性、
 * 零主元不产生 NaN。这样即使 `tests/parity/golden/**` 被人删了，
 * 选主元规则被改错也还是会在 `npm test` 里立刻红。
 *
 * 矩阵一律行主序展平（跨语言约定，见 `numeric/lu.ts` 顶部的对应表）。
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { luPartialPivot, type ParityLu } from "../shared/src/numeric/lu.js";
import { readFileSync } from "node:fs";
import { join } from "node:path";

/** 从仓库根读一份对拍 golden 的输入矩阵（`--root` 之外的测试不猜路径：向上找 tests/parity）。 */
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

function goldenMatrix(caseName: string): { a: number[]; n: number } {
  const doc = JSON.parse(readFileSync(join(rootDir(), "tests", "parity", "golden", `${caseName}.json`), "utf8"));
  return { a: doc.inputs.a as number[], n: doc.inputs.n as number };
}

/** 行主序 → 二维（只给测试用）。 */
function rows(a: number[], n: number): number[][] {
  return Array.from({ length: n }, (_, i) => a.slice(i * n, i * n + n));
}

/** `P`：`perm[k]` 表示分解后第 k 行来自原始第 perm[k] 行（Python 侧同义）。 */
function permutedA(a: number[], n: number, perm: number[]): number[][] {
  const m = rows(a, n);
  return perm.map((src) => m[src] as number[]);
}

/** 最大绝对差（`max|X − Y|`）。 */
function maxAbsDiff(x: number[][], y: number[][]): number {
  let worst = 0;
  for (let i = 0; i < x.length; i++) {
    for (let j = 0; j < (x[i] as number[]).length; j++) {
      worst = Math.max(worst, Math.abs((x[i]![j] as number) - (y[i]![j] as number)));
    }
  }
  return worst;
}

function matmul(x: number[][], y: number[][]): number[][] {
  const n = x.length;
  const m = (y[0] as number[]).length;
  const k = y.length;
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: m }, (_, j) => {
      let s = 0;
      for (let p = 0; p < k; p++) s += (x[i]![p] as number) * (y[p]![j] as number);
      return s;
    }),
  );
}

const CASES = ["lu_partial_pivot", "lu_wilkinson_growth", "lu_tiny_pivot"] as const;

for (const name of CASES) {
  test(`lu(${name})：P·A = L·U`, () => {
    const { a, n } = goldenMatrix(name);
    const res: ParityLu = luPartialPivot(a, n);
    const pa = permutedA(a, n, res.perm);
    const lu = matmul(rows(res.l, n), rows(res.u, n));
    // ‖A‖∞ 量级约 1，重构误差应当在 n·u 量级；取 1e-11 是宽松上限，
    // 但只要选主元/消元顺序写错，误差会到 O(1) 甚至 NaN，这里立刻报警。
    const worst = maxAbsDiff(pa, lu);
    assert.ok(worst < 1e-11, `${name}：max|PA − LU| = ${worst}（应 ≤ n·u ≈ 1e-15）`);
  });

  test(`lu(${name})：结构不变量（L 单位下三角、U 上三角、perm 是置换）`, () => {
    const { a, n } = goldenMatrix(name);
    const res = luPartialPivot(a, n);
    assert.equal(res.l.length, n * n);
    assert.equal(res.u.length, n * n);
    for (let i = 0; i < n; i++) {
      assert.equal(res.l[i * n + i], 1, `L[${i}][${i}] 必须是 1`);
      for (let j = 0; j < n; j++) {
        if (i < j) assert.equal(res.l[i * n + j], 0, `L[${i}][${j}] 必须是 0`);
        if (i > j) assert.equal(res.u[i * n + j], 0, `U[${i}][${j}] 必须是 0`);
      }
    }
    assert.deepEqual([...res.perm].sort((x, y) => x - y), Array.from({ length: n }, (_, i) => i));
    assert.ok(res.l.every((v) => Number.isFinite(v)) && res.u.every((v) => Number.isFinite(v)));
  });
}

test("Wilkinson(12) 的部分主元增长因子恰好是 2^11 = 2048", () => {
  const { a, n } = goldenMatrix("lu_wilkinson_growth");
  const res = luPartialPivot(a, n);
  // 这是本章的标志性数字：教科书上"部分主元的最坏情形"就是 ρ = 2^{n-1}。
  // 用严格相等而不是容差：整数 2048 在 float64 里是精确值，允许容差就等于允许"ρ 少一半"。
  assert.equal(res.rho, 2048);
});

test("首元 1e-14 的矩阵必须换行（否则把小主元当主元）", () => {
  const { a, n } = goldenMatrix("lu_tiny_pivot");
  const res = luPartialPivot(a, n);
  assert.notEqual(res.pivots[0], 0, "第一步就该换行：|A[0][0]| = 1e-14 不可能是该列最大元");
  assert.ok(res.rowSwaps.length >= 2, "至少发生过一次行交换（rowSwaps 是 [k, p] 对）");
  assert.ok(Number.isFinite(res.rho) && res.rho < 10, `ρ = ${res.rho}：换行之后不该有增长`);
});

test("Hilbert(6) 不需要换行时也要给出正确的 ρ = 1", () => {
  const { a, n } = goldenMatrix("lu_partial_pivot");
  const res = luPartialPivot(a, n);
  // Hilbert 矩阵首元 1 已是该列最大元；U 的最大元仍来自原始矩阵，故 ρ 恰好为 1。
  assert.equal(res.rho, 1);
  assert.equal(res.pivots[0], 0);
});

test("精确零主元：跳过该列而不是产生 NaN/Inf", () => {
  // [[1,2,3],[2,4,6],[1,1,1]]：第 2 行是第 1 行的 2 倍，第二步的主元列全零。
  const a = [1, 2, 3, 2, 4, 6, 1, 1, 1];
  const res = luPartialPivot(a, 3);
  assert.ok(res.l.every((v) => Number.isFinite(v)), "L 里不许有 NaN/Inf");
  assert.ok(res.u.every((v) => Number.isFinite(v)), "U 里不许有 NaN/Inf");
  assert.equal(res.pivots.length, 2, "n−1 步都要记录主元（哪怕该步是零主元）");
});

test("输入契约：长度与阶数不符时响亮报错", () => {
  assert.throws(() => luPartialPivot([1, 2, 3], 2), /矩阵长度 3 ≠ n\*n = 4/);
  assert.throws(() => luPartialPivot([1], 0), /阶数 n 必须是正整数/);
  assert.throws(() => luPartialPivot([1], 1.5), /阶数 n 必须是正整数/);
});
