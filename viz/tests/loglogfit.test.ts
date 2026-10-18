import { test } from "node:test";
import assert from "node:assert/strict";

import {
  fitOrder,
  consecutiveOrders,
  diagnoseOrder,
  referenceLine,
  type OrderFit,
  type WindowDiagnosis,
} from "../shared/src/loglogfit.js";

/** 确定性 PRNG（测试内自备，不引入第三方；种子固定 → 失败可复现）。 */
function makeRng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** `h_k = 2^-k`（从最粗到最细，符合"h 递减"约定）。 */
function dyadicGrid(n: number): number[] {
  return Array.from({ length: n }, (_, k) => Math.pow(2, -k));
}

/** 断言：与期望值的相对误差 ≤ rtol，附加 1e-15 绝对地板（机器零量级只比绝对差）。 */
function assertClose(actual: number, expected: number, msg: string, rtol = 1e-12): void {
  if (Number.isNaN(expected)) {
    assert.ok(Number.isNaN(actual), `${msg}：期望 NaN，实际 ${actual}`);
    return;
  }
  if (!Number.isFinite(expected)) {
    assert.equal(actual, expected, `${msg}：期望 ${expected}，实际 ${actual}`);
    return;
  }
  const tol = rtol * Math.max(Math.abs(expected), Math.abs(actual)) + 1e-15;
  assert.ok(
    Math.abs(actual - expected) <= tol,
    `${msg}：期望 ${expected}，实际 ${actual}（差 ${Math.abs(actual - expected)} > ${tol}）`,
  );
}

/* ------------------------------------------------------------------ *
 * 基本性质
 * ------------------------------------------------------------------ */

test("loglogfit: err = C·h^p 的拟合斜率精确等于 p，r2 = 1", () => {
  for (const p of [1, 2, 3, 4, 1.5]) {
    const h = dyadicGrid(14);
    const err = h.map((v) => 0.37 * Math.pow(v, p));
    const fit = fitOrder(h, err);
    assertClose(fit.slope, p, `p=${p} 斜率`, 1e-10);
    assert.ok(fit.r2 > 1 - 1e-12, `p=${p} 的 r2 = ${fit.r2} 应 ≈ 1`);
    assertClose(fit.intercept, Math.log(0.37), `p=${p} 截距 = log C`, 1e-12);
    assert.equal(fit.nUsed, 14);
    assert.equal(fit.nDropped, 0);
    assert.ok(fit.stderr < 1e-14, `完美幂律的斜率标准误应 ≈ 0，实际 ${fit.stderr}`);
  }
});

test("loglogfit: 1% 乘性噪声下斜率误差 < 0.04（多组随机种子）", () => {
  for (const seed of [1, 2, 3, 12345, 987654321]) {
    const rng = makeRng(seed);
    const p = 1 + 3 * rng();
    const h = dyadicGrid(40);
    const err = h.map((v) => 0.37 * Math.pow(v, p) * (1 + 0.01 * (2 * rng() - 1)));
    const fit = fitOrder(h, err);
    assert.ok(
      Math.abs(fit.slope - p) < 0.04,
      `seed=${seed} p=${p} 实测 ${fit.slope}，误差 ${Math.abs(fit.slope - p)}`,
    );
    assert.ok(fit.r2 > 0.9999, `seed=${seed} r2 = ${fit.r2} 过低`);
  }
});

test("loglogfit: skip / tail 精确裁剪（nUsed / nDropped 逐点核对）", () => {
  const h = dyadicGrid(16);
  const err = h.map((v) => 0.37 * v * v);
  assert.deepEqual(
    (() => {
      const f = fitOrder(h, err, { skip: 3 });
      return [f.nUsed, f.nDropped];
    })(),
    [13, 3],
  );
  assert.deepEqual(
    (() => {
      const f = fitOrder(h, err, { skip: 3, tail: 5 });
      return [f.nUsed, f.nDropped];
    })(),
    [5, 11],
  );
  assert.deepEqual(
    (() => {
      const f = fitOrder(h, err, { tail: 7 });
      return [f.nUsed, f.nDropped];
    })(),
    [7, 9],
  );
  // skip 吃掉全部点 → 明确抛错
  assert.throws(() => fitOrder(h, err, { skip: 16 }), /有效点不足/);
  assert.throws(() => fitOrder(h, err, { tail: 0 }), /有效点不足/);
  // tail 大于剩余点数 → 等价于不截断
  assert.equal(fitOrder(h, err, { skip: 2, tail: 999 }).nUsed, 14);
});

test("loglogfit: err ≤ 0 / h ≤ 0 / 非有限点被丢弃并计入 nDropped", () => {
  const h = [1, 0.5, 0.25, 0.125, 0, -0.0625, 0.03125, Number.NaN, 0.015625];
  const err = [0.37, 0.0925, 0, 0.00578125, 1e-3, 1e-3, 3.6e-4, 9e-5, Infinity];
  const fit = fitOrder(h, err);
  // 保留 (1,0.37)、(0.5,0.0925)、(0.125,0.00578125) 三点：err = 0.37·h² 精确成立
  assert.equal(fit.nUsed, 4, "只有第 1、2、4、7 个点既正又有限");
  assert.equal(fit.nDropped, 5);
  // 第 7 点 (0.03125, 3.6e-4) 的 C 略偏（0.3686 vs 0.37），故斜率只到 ~1e-3 精度
  assertClose(fit.slope, 2, "保留下来的点仍满足 err ≈ 0.37·h²", 1e-2);
});

test("loglogfit: consecutiveOrders 对 err = h² 恒等于 2（含降序网格）", () => {
  const h = dyadicGrid(20);
  const err = h.map((v) => v * v);
  const orders = consecutiveOrders(h, err);
  assert.equal(orders.length, h.length - 1);
  for (let i = 0; i < orders.length; i++) {
    assert.ok(Math.abs(orders[i] - 2) <= 1e-12, `局部阶[${i}] = ${orders[i]} ≠ 2`);
  }
  assert.equal(orders[0], 2, "2 的幂网格上比值正好是整数幂，可精确等于 2");
  // 少于 2 点 → 抛错
  assert.throws(() => consecutiveOrders([1], [1]), /至少需要 2 个点/);
  // 非正/重复 h 不抛错，只产生非有限局部阶（由诊断负责排除）
  const dirty = consecutiveOrders([1, 1, 0, -1], [1, 1, 0, -1]);
  assert.equal(dirty.length, 3);
});

/* ------------------------------------------------------------------ *
 * 诊断
 * ------------------------------------------------------------------ */

test("loglogfit: 舍入地板用例必须判为 floored（err = h² + 1e-3）", () => {
  const h = dyadicGrid(14);
  const err = h.map((v) => v * v + 1e-3);
  const diag = diagnoseOrder(h, err);
  assert.equal(diag.floored, true, `应判为触舍入地板：head=${diag.headP}, tail=${diag.tailP}`);
  assert.equal(diag.noisy, false, "floored 时不再报 noisy（二者互斥）");
  assert.ok(diag.headP > 1.9, `头部局部阶≈2，实际 ${diag.headP}`);
  assert.ok(diag.tailP < 0.01, `尾部局部阶应贴近 0，实际 ${diag.tailP}`);
  assert.ok(diag.suggestSkip > 0 && diag.suggestTail > 0, "应给出可用的裁剪窗口");
  // 全量拟合会被地板显著拉低（这就是"看起来收敛"的陷阱）
  assert.ok(fitOrder(h, err).slope < 1, "全量拟合斜率被地板污染，应明显偏小");
  // 注意：建议窗口取的是"局部阶彼此最一致"的连续段；地板区的局部阶全都≈0，
  // 反而最"一致"，所以 suggestSkip/suggestTail 指向的正是地板段（与 Python 完全一致）。
  // 这是设计立场"只给证据、不替你决定"的直接后果：floored=true 时不能盲用建议窗口。
  assert.ok(
    fitOrder(h, err, { skip: diag.suggestSkip, tail: diag.suggestTail }).slope < 0.1,
    "地板段自身最一致，建议窗口会落在细网格端（必须在报告里显式说明）",
  );
  // 真正恢复阶数要丢掉细网格端；skip/tail 只能"丢前缀 + 保后缀"，
  // 因此这一步需要调用方自己切片（截取最粗的前 4 个点 → 1.97 ≈ 2）
  const coarse = fitOrder(h.slice(0, 4), err.slice(0, 4));
  assert.ok(
    coarse.slope > 1.9 && coarse.slope < 2.05,
    `粗网格端的斜率 ${coarse.slope} 应回到 2 附近`,
  );
});

test("loglogfit: 局部阶分段时给出 head/tail 中位数与最长一致段", () => {
  // 构造局部阶恰为 [4,4,4,1,1,1] 的误差序列：e_{i+1} = e_i / 2^{p_i}
  const want = [4, 4, 4, 1, 1, 1];
  const h: number[] = [1];
  const err: number[] = [1];
  for (const p of want) {
    h.push(h[h.length - 1] / 2);
    err.push(err[err.length - 1] / Math.pow(2, p));
  }
  const diag = diagnoseOrder(h, err);
  assert.deepEqual(Array.from(diag.localOrders), want, "构造出的局部阶应精确复现");
  assertClose(diag.headP, 4, "前 1/3 中位数");
  assertClose(diag.tailP, 1, "后 1/3 中位数");
  assert.equal(diag.noisy, true, "首尾差 3 > tol=0.1 → noisy");
  assert.equal(diag.floored, false, "尾部阶 1 远高于 floorP=0.25，不算地板");
  // 最长一致段 = 前三个点 [4,4,4] → skip=0，保留 3+1=4 个点
  assert.equal(diag.suggestSkip, 0);
  assert.equal(diag.suggestTail, 4);

  // minRun 大于最长段 → 退化为整段
  const loose = diagnoseOrder(h, err, { minRun: 5 });
  assert.equal(loose.suggestSkip, 0);
  assert.equal(loose.suggestTail, h.length);

  // 一致序列：floored / noisy 都为假
  const clean = diagnoseOrder(dyadicGrid(10), dyadicGrid(10).map((v) => v * v));
  assert.equal(clean.floored, false);
  assert.equal(clean.noisy, false);
  assertClose(clean.headP, 2, "干净序列头部阶");
  assertClose(clean.tailP, 2, "干净序列尾部阶");
});

/* ------------------------------------------------------------------ *
 * 参考线（防误导规则 2）
 * ------------------------------------------------------------------ */

test("loglogfit: referenceLine 的斜率等于理论斜率，anchor 决定穿过点", () => {
  const h = dyadicGrid(12);
  for (const theory of [1, 2, 2.5, 3, 4]) {
    const line = referenceLine(h, theory, { h0: 1, e0: 3 });
    assert.equal(line.length, h.length);
    const fit = fitOrder(h, Array.from(line));
    assertClose(fit.slope, theory, `theory=${theory} 参考线斜率`, 1e-10);
    assertClose(line[0], 3, "anchor 处应精确穿过 e0"); // h[0] = 1
    for (let i = 0; i < h.length; i++) {
      assertClose(line[i], 3 * Math.pow(h[i], theory), `theory=${theory} 第 ${i} 点`, 1e-14);
    }
  }
  // 缺省 anchor：h0 = min(h)，e0 = 1
  const hh = [8, 4, 2, 1];
  const def = referenceLine(hh, 3);
  assert.deepEqual(Array.from(def), [512, 64, 8, 1]);
  // 实测线与理论线必须能同图对照：两者在锚点处相交、斜率不同
  const err = hh.map((v) => 0.5 * v * v);
  const measured = fitOrder(hh, err);
  assertClose(measured.slope, 2, "实测斜率");
  const theoryLine = referenceLine(hh, 3, { h0: hh[hh.length - 1], e0: 0.5 });
  assertClose(theoryLine[theoryLine.length - 1], 0.5, "锚点重合");
  assert.notEqual(measured.slope, 3, "实测与理论必须可区分（否则参考线没意义）");
  // 空数组：退化为空结果（渲染层不该因为一条参考线崩掉）
  assert.equal(referenceLine([], 2).length, 0);
});

/* ------------------------------------------------------------------ *
 * 抛错与边界
 * ------------------------------------------------------------------ */

test("loglogfit: 形状不一致 / 空数组 / 点不足 / h 全相同 都有明确抛错", () => {
  assert.throws(() => fitOrder([1, 2], [1]), /长度不一致/);
  assert.throws(() => fitOrder([], []), /为空/);
  assert.throws(() => fitOrder([1, 2], [1, 2]), /有效点不足/); // 默认 minPoints = 3
  assert.throws(() => fitOrder([1, 1, 1], [1, 2, 3]), /所有 h 相同/);
  assert.throws(() => fitOrder([1, 2, 3], [1, 2, 3], { minPoints: 99 }), /有效点不足/);
  assert.throws(() => fitOrder([1, 2, 3], [1, 2, 3], { skip: Number.NaN }), RangeError);
  assert.throws(() => diagnoseOrder([1], [1]), /至少需要 2 个点/);
});

test("loglogfit: 常数误差序列给出 slope = 0、r2 = 1（SStot = 0 的约定）", () => {
  const fit: OrderFit = fitOrder([1, 2, 4, 8], [5, 5, 5, 5]);
  assert.equal(fit.slope, 0);
  assert.equal(fit.r2, 1, "SStot = 0 时 r2 取 1（与 Python 一致）");
  assert.equal(fit.stderr, 0);
  assertClose(fit.intercept, Math.log(5), "截距 = log 常数");
});

test("loglogfit: 局部阶全非有限时给出 NaN 诊断且不抛错", () => {
  const diag: WindowDiagnosis = diagnoseOrder([1, 1, 1], [1, 1, 1]);
  assert.equal(diag.localOrders.length, 2);
  assert.ok(Number.isNaN(diag.headP) && Number.isNaN(diag.tailP));
  assert.equal(diag.floored, false);
  assert.equal(diag.noisy, true, "无任何有限局部阶 → 视为 noisy");
  assert.equal(diag.suggestSkip, 0);
  assert.equal(diag.suggestTail, 3, "退化时建议保留整段");
});

/* ------------------------------------------------------------------ *
 * 与 Python numcore/verify/order.py 的逐字段对拍（G7 的 TS 侧锚点）
 * ------------------------------------------------------------------ *
 * 下面的期望值由 Python 侧原样算出（同算法、同约定），精度按 float64 完整给出：
 *
 *     python3 - <<'PY'
 *     from numcore.verify.order import fit_order, diagnose_order
 *     fit = fit_order(h, err, skip=..., tail=...)
 *     print(fit.slope, fit.intercept, fit.r2, fit.stderr, fit.n_used, fit.n_dropped)
 *     print(diagnose_order(h, err))
 *     PY
 *
 * 逐字段比较（slope/intercept/r2/stderr/nUsed/nDropped/headP/tailP/floored/noisy/
 * suggestSkip/suggestTail/localOrders/referenceLine，共 17 组 case）最大相对差 1.6e-14。
 */

/** Python 侧给出的基准（`numcore/verify/order.py`，种子 20251018 的 numpy 随机点云）。 */
const PY_GOLDEN = [
  {
    name: "clean_p2",
    h: dyadicGrid(14),
    err: dyadicGrid(14).map((v) => 0.37 * v * v),
    opts: { skip: 0, tail: null, minPoints: 3 },
    fit: { slope: 2, intercept: -0.9942522733438679, r2: 1, stderr: 8.32218161504953e-17, nUsed: 14, nDropped: 0 },
    diag: { headP: 2, tailP: 2, floored: false, noisy: false, suggestSkip: 0, suggestTail: 14 },
  },
  {
    name: "floor",
    h: dyadicGrid(14),
    err: dyadicGrid(14).map((v) => v * v + 1e-3),
    opts: { skip: 0, tail: null, minPoints: 3 },
    fit: {
      slope: 0.6903095972159902,
      intercept: -2.228600192959242,
      r2: 0.7236678857089698,
      stderr: 0.12314004787808838,
      nUsed: 14,
      nDropped: 0,
    },
    diag: {
      headP: 1.9581305592247913,
      tailP: 0.0006446083995886359,
      floored: true,
      noisy: false,
      suggestSkip: 7,
      suggestTail: 7,
    },
  },
  {
    name: "window_0_7",
    h: dyadicGrid(16),
    err: dyadicGrid(16).map((v) => 0.37 * v * v),
    opts: { skip: 0, tail: 7, minPoints: 3 },
    fit: { slope: 2.0000000000000004, intercept: -0.9942522733438643, r2: 1, stderr: 2.16591163152742e-16, nUsed: 7, nDropped: 9 },
    diag: { headP: 2, tailP: 2, floored: false, noisy: false, suggestSkip: 0, suggestTail: 16 },
  },
  {
    name: "drop_nonpositive",
    h: [1, 0.5, 0.25, 0.125, 0, -0.0625, 0.03125, 0.015625],
    err: [0.37, 0.0925, 0, 0.00578125, 0.001, 0.001, 0.00036, 0.00009],
    opts: { skip: 0, tail: null, minPoints: 3 },
    fit: {
      slope: 2.00102166324014,
      intercept: -0.9936007633890327,
      r2: 0.9999999353507327,
      stderr: 0.0002937466893489458,
      nUsed: 5,
      nDropped: 3,
    },
    // headP = +Infinity（前 1/3 局部阶含 +Inf）、tailP = NaN（含 NaN）——与 numpy.median 同口径
    diag: { headP: Infinity, tailP: NaN, floored: false, noisy: false, suggestSkip: 3, suggestTail: 3 },
  },
  {
    name: "random_2",
    h: [
      0.041037875060659444, 0.02886502625934506, 0.000015110504949398192,
      0.000009099871922537235, 0.0000019107402819221746, 1.6186001348643222e-8,
    ],
    err: [
      0.00018746999684128687, 0.00007594159430985928, 5.992384303928495e-13,
      1.737840772850646e-13, 3.3105634193979726e-15, 2.4851024446422118e-20,
    ],
    opts: { skip: 0, tail: null, minPoints: 3 },
    fit: {
      slope: 2.4777833839474845,
      intercept: -0.6729348291955439,
      r2: 0.9999933341414283,
      stderr: 0.0031986213784216335,
      nUsed: 6,
      nDropped: 0,
    },
    diag: {
      headP: 2.568186593284652,
      tailP: 2.4731679842252015,
      floored: false,
      noisy: false,
      suggestSkip: 0,
      suggestTail: 6,
    },
  },
];

test("loglogfit: 与 Python numcore/verify/order.py 逐字段对齐（金标）", () => {
  for (const c of PY_GOLDEN) {
    const fit = fitOrder(c.h, c.err, c.opts);
    assertClose(fit.slope, c.fit.slope, `${c.name}.slope`);
    assertClose(fit.intercept, c.fit.intercept, `${c.name}.intercept`);
    assertClose(fit.r2, c.fit.r2, `${c.name}.r2`);
    assertClose(fit.stderr, c.fit.stderr, `${c.name}.stderr`);
    assert.equal(fit.nUsed, c.fit.nUsed, `${c.name}.nUsed`);
    assert.equal(fit.nDropped, c.fit.nDropped, `${c.name}.nDropped`);

    const diag = diagnoseOrder(c.h, c.err);
    assertClose(diag.headP, c.diag.headP, `${c.name}.headP`);
    assertClose(diag.tailP, c.diag.tailP, `${c.name}.tailP`);
    assert.equal(diag.floored, c.diag.floored, `${c.name}.floored`);
    assert.equal(diag.noisy, c.diag.noisy, `${c.name}.noisy`);
    assert.equal(diag.suggestSkip, c.diag.suggestSkip, `${c.name}.suggestSkip`);
    assert.equal(diag.suggestTail, c.diag.suggestTail, `${c.name}.suggestTail`);
  }
});

// --------------------------------------------------------------------------- //
// head 选项：对付舍入地板的正确工具（与 numcore/verify/order.py 对齐）
// --------------------------------------------------------------------------- //
test("fitOrder 的 head 只保留粗网格端，能从舍入地板里救回阶数", () => {
  const h = [1, 0.5, 0.25, 0.125, 0.0625, 0.03125];
  // 前四个点严格是 h²，后两个点被 1e-3 的舍入地板压平
  const err = [1, 0.25, 0.0625, 0.015625, 1e-3, 1e-3];
  const full = fitOrder(h, err);
  const head = fitOrder(h, err, { head: 4 });
  assert.ok(Math.abs(full.slope - 2) > 0.05, `全窗口应当被地板带偏，实测 ${full.slope}`);
  assert.ok(full.r2 < 0.99, `R² 应当暴露地板污染，实测 ${full.r2}`);
  assert.ok(Math.abs(head.slope - 2) < 1e-9, `head=4 应当精确救回 2，实测 ${head.slope}`);
  assert.ok(head.r2 > 0.999999);
  // tail 方向相反：保留细端恰好保留被污染的点，窗口里几乎全是地板点 → R² 崩掉
  const tail = fitOrder(h, err, { tail: 3 });
  assert.ok(tail.r2 < 0.9, `tail=3 的窗口不可信，R²=${tail.r2}`);
  // head 与 skip 可以叠加：丢前 1 个、保随后 3 个
  const combo = fitOrder(h, err, { skip: 1, head: 3 });
  assert.ok(Math.abs(combo.slope - 2) < 1e-9, `skip=1,head=3 实测 ${combo.slope}`);
});
