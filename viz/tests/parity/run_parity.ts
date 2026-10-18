/**
 * G7 双实现对拍的 **TypeScript 侧 runner**（PROTOCOLS §7）。
 *
 * 职责（只做三件事）
 * ----------------
 * 1. 读 `tests/parity/cases.json`（case 索引：`name` / `algo` / `golden` / `tolerance` / `outputs`）；
 * 2. 对每个 case 读 `tests/parity/golden/<case>.json` 里的**输入**（输入一律由 Python 生成落盘：
 *    连 `u0`、时间网格 `t`、`h`/`err` 都是 Python 算好的 double，TS 侧只做 JSON 精确回读——
 *    这样"输入构造"这一环不会被两套三角/幂函数实现的差异污染）；
 * 3. 调用 `viz/shared/src/numeric/<algo>.ts` 的纯函数，把输出写 `dist/reports/parity_ts.json`。
 *
 * 报告结构（Python `tools/parity.py` 按此读取，缺键即判失败）
 * --------------------------------------------------------
 * ```json
 * {
 *   "generated_at": "2026-10-18T15:36:17.000Z",
 *   "runner": "viz/tests/parity/run_parity.ts",
 *   "node": "v24.21.0",
 *   "root": "<仓库根绝对路径>",
 *   "n_cases": 5,
 *   "cases": {
 *     "heat_explicit_step": {
 *       "algo": "heat",
 *       "outputs": {"u": [0, 0.078…, …]}
 *     },
 *     "某个跑挂的 case": {"algo": "…", "error": "有效点不足：…"}
 *   }
 * }
 * ```
 *
 * 数值一律以 JSON number 写出：`JSON.stringify` 给出的是 double 的**最短往返表示**，
 * Python `json.loads` 能逐位还原，不存在"打印精度不足导致的假通过"。
 *
 * 运行：`node dist/viz/tests/parity/run_parity.js`（先 `npx tsc -p tsconfig.json`）。
 * 退出码：0 = 全部 case 算完；1 = 至少一个 case 抛错（报告仍会写盘，便于 Python 侧报明细）。
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

// PROTOCOLS §7：对每个 case 调用 `viz/shared/src/numeric/<algo>.ts`。这里按模块逐个 import，
// 而不是走 `numeric/index.js` 桶文件——为的是让"case.algo → 实现模块"的映射在源码里可 grep。
// （`index.ts` 仍是对外汇总出口，供其它消费者使用。）
import { ftcsAmplification, ftcsRun, ftcsStep } from "../../shared/src/numeric/heat.js";
import { luPartialPivot } from "../../shared/src/numeric/lu.js";
import { dampedOscillator, oscillatorEnergy } from "../../shared/src/numeric/oscillator.js";
import { fitOrderParity } from "../../shared/src/numeric/orderfit.js";
import { powerIteration } from "../../shared/src/numeric/power_iter.js";
import {
  bdfCoefficients,
  complexAbs,
  lmmRootModulus,
  stabilityFunction,
  type ComplexGrid,
} from "../../shared/src/numeric/stability.js";
import type { FitOptions } from "../../shared/src/loglogfit.js";

/** `tests/parity/cases.json` 里的一个 case 条目。 */
interface CaseEntry {
  name: string;
  algo: string;
  golden: string;
  tolerance: string;
  outputs: string[];
  params?: Record<string, unknown>;
  note?: string;
}

interface CasesFile {
  version: number;
  cases: CaseEntry[];
}

/** `tests/parity/golden/<case>.json`：Python 生成的输入 + 标准输出。 */
interface GoldenDoc {
  case: string;
  algo: string;
  inputs: Record<string, unknown>;
  outputs?: Record<string, unknown>;
  meta?: Record<string, unknown>;
  note?: string;
}

type Scalar = number;
type Vector = number[];
type Outputs = Record<string, Scalar | Vector>;

// --------------------------------------------------------------------------- //
// 小工具
// --------------------------------------------------------------------------- //

function argValue(args: string[], flag: string): string | undefined {
  const i = args.indexOf(flag);
  if (i >= 0 && i + 1 < args.length) return args[i + 1];
  const prefixed = args.find((a) => a.startsWith(`${flag}=`));
  return prefixed === undefined ? undefined : prefixed.slice(flag.length + 1);
}

/** 从 runner 所在目录向上找仓库根（含 `tests/parity/cases.json` 的那一层）。 */
function findRoot(startDir: string): string {
  let dir = startDir;
  for (let i = 0; i < 16; i++) {
    if (existsSync(join(dir, "tests", "parity", "cases.json"))) return dir;
    const parent = dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
  return process.cwd();
}

function asRecord(v: unknown, what: string): Record<string, unknown> {
  if (v === null || typeof v !== "object" || Array.isArray(v)) {
    throw new Error(`${what} 必须是 JSON 对象`);
  }
  return v as Record<string, unknown>;
}

function asNumber(v: unknown, what: string): number {
  if (typeof v !== "number") throw new Error(`${what} 必须是 number，得到 ${JSON.stringify(v)}`);
  return v;
}

function asNumberArray(v: unknown, what: string): number[] {
  if (!Array.isArray(v)) throw new Error(`${what} 必须是数组`);
  return v.map((x, i) => {
    if (typeof x !== "number") throw new Error(`${what}[${i}] 必须是 number，得到 ${JSON.stringify(x)}`);
    return x;
  });
}

function asString(v: unknown, what: string): string {
  if (typeof v !== "string") throw new Error(`${what} 必须是 string`);
  return v;
}

/** 按 `entry.outputs` 的声明顺序重排输出键，并校验一个都不少。 */
function pickOutputs(name: string, wanted: string[], got: Outputs): Outputs {
  const out: Outputs = {};
  for (const key of wanted) {
    if (!(key in got)) throw new Error(`case ${name} 缺少输出键 ${key}（实现与 cases.json 漂移？）`);
    out[key] = got[key] as Scalar | Vector;
  }
  return out;
}

// --------------------------------------------------------------------------- //
// 各算法分支（键名必须与 tools/parity.py 逐字一致）
// --------------------------------------------------------------------------- //

function computeHeat(inputs: Record<string, unknown>): Outputs {
  const op = asString(inputs.op, "inputs.op");
  const r = asNumber(inputs.r, "inputs.r");
  const u0 = Float64Array.from(asNumberArray(inputs.u0, "inputs.u0"));
  if (op === "step") return { u: Array.from(ftcsStep(u0, r)) };
  if (op === "run") {
    const nSteps = asNumber(inputs.n_steps, "inputs.n_steps");
    return { u: Array.from(ftcsRun(u0, r, nSteps)) };
  }
  throw new Error(`未知的 heat op=${op}（只有 step / run）`);
}

function computeAmplification(inputs: Record<string, unknown>): Outputs {
  const nx = asNumber(inputs.nx, "inputs.nx");
  const r = asNumber(inputs.r, "inputs.r");
  const modes = asNumberArray(inputs.modes, "inputs.modes");
  return { g: modes.map((m) => ftcsAmplification(r, m, nx)) };
}

function computeOscillator(inputs: Record<string, unknown>): Outputs {
  const omega = asNumber(inputs.omega, "inputs.omega");
  const t = Float64Array.from(asNumberArray(inputs.t, "inputs.t"));
  const x0 = inputs.x0 === undefined ? 1.0 : asNumber(inputs.x0, "inputs.x0");
  const v0 = inputs.v0 === undefined ? 0.0 : asNumber(inputs.v0, "inputs.v0");
  const branches = inputs.branches;
  if (!Array.isArray(branches)) throw new Error("inputs.branches 必须是数组");
  const out: Outputs = {};
  for (const raw of branches) {
    const b = asRecord(raw, "inputs.branches[i]");
    const tag = asString(b.tag, "branch.tag");
    const zeta = asNumber(b.zeta, "branch.zeta");
    const sample = dampedOscillator(t, omega, zeta, x0, v0);
    out[`x_${tag}`] = Array.from(sample.x);
    out[`v_${tag}`] = Array.from(sample.v);
    out[`E_${tag}`] = Array.from(oscillatorEnergy(sample.x, sample.v, omega));
  }
  return out;
}

function computeOrderFit(inputs: Record<string, unknown>): Outputs {
  const groups = inputs.groups;
  if (!Array.isArray(groups)) throw new Error("inputs.groups 必须是数组");
  const out: Outputs = {};
  for (const raw of groups) {
    const g = asRecord(raw, "inputs.groups[i]");
    const name = asString(g.name, "group.name");
    const h = asNumberArray(g.h, "group.h");
    const err = asNumberArray(g.err, "group.err");
    const rawOpts = g.opts === undefined ? {} : asRecord(g.opts, "group.opts");
    const opts: FitOptions = {};
    if (rawOpts.skip !== undefined && rawOpts.skip !== null) opts.skip = asNumber(rawOpts.skip, "opts.skip");
    if (rawOpts.head !== undefined && rawOpts.head !== null) opts.head = asNumber(rawOpts.head, "opts.head");
    if (rawOpts.tail !== undefined && rawOpts.tail !== null) opts.tail = asNumber(rawOpts.tail, "opts.tail");
    if (rawOpts.min_points !== undefined && rawOpts.min_points !== null) {
      opts.minPoints = asNumber(rawOpts.min_points, "opts.min_points");
    }
    const fit = fitOrderParity(h, err, opts);
    out[`slope_${name}`] = fit.slope;
    out[`intercept_${name}`] = fit.intercept;
    out[`r2_${name}`] = fit.r2;
    out[`nUsed_${name}`] = fit.nUsed;
    out[`nDropped_${name}`] = fit.nDropped;
  }
  return out;
}

function computeLu(inputs: Record<string, unknown>): Outputs {
  const n = asNumber(inputs.n, "inputs.n");
  const a = asNumberArray(inputs.a, "inputs.a");
  const res = luPartialPivot(a, n);
  return {
    l: res.l,
    u: res.u,
    perm: res.perm,
    pivots: res.pivots,
    rowSwaps: res.rowSwaps,
    rho: res.rho,
  };
}

/**
 * 幂法（1.4 章）：矩阵 `a` 与初值 `x0` 都是 Python 侧生成落盘的 double，
 * 这里只按 `max_iter` / `tol` 跑迭代。布尔量写成 0/1 —— 对拍协议只认数值
 * （`tools/parity.py::_as_vector` 把 `bool` 判成非法），所以转换发生在这一层。
 */
function computePowerIter(inputs: Record<string, unknown>): Outputs {
  const n = asNumber(inputs.n, "inputs.n");
  const a = asNumberArray(inputs.a, "inputs.a");
  const x0 = asNumberArray(inputs.x0, "inputs.x0");
  const maxIter = inputs.max_iter === undefined ? undefined : asNumber(inputs.max_iter, "inputs.max_iter");
  const tol = inputs.tol === undefined ? undefined : asNumber(inputs.tol, "inputs.tol");
  const res = powerIteration(a, n, { x0, maxIter, tol });
  return {
    eigenvalue: res.eigenvalue,
    vector: res.vector,
    rayleigh: res.rayleigh,
    residual: res.residual,
    ratios: res.ratios,
    nIter: res.nIter,
    converged: res.converged ? 1 : 0,
    angleStep: res.angleStep,
    isComplexPair: res.isComplexPair ? 1 : 0,
  };
}

/** 3.2 章：`stability_function` 网格（`op="function"`）与 LMM 根模（`op="lmm_roots"`）。 */
function computeStability(inputs: Record<string, unknown>): Outputs {
  const op = asString(inputs.op, "inputs.op");
  const z: ComplexGrid = {
    re: asNumberArray(inputs.z_re, "inputs.z_re"),
    im: asNumberArray(inputs.z_im, "inputs.z_im"),
  };
  const out: Outputs = {};
  if (op === "function") {
    const methods = inputs.methods;
    if (!Array.isArray(methods)) throw new Error("inputs.methods 必须是数组");
    for (const raw of methods) {
      const m = asRecord(raw, "inputs.methods[i]");
      const tag = asString(m.tag, "methods.tag");
      const method = asString(m.method, "methods.method");
      const order = asNumber(m.order, "methods.order");
      const r = stabilityFunction(method, order, z);
      out[`re_${tag}`] = r.re;
      out[`im_${tag}`] = r.im;
      out[`abs_${tag}`] = complexAbs(r);
    }
    return out;
  }
  if (op === "lmm_roots") {
    const orders = asNumberArray(inputs.orders, "inputs.orders");
    const iters = inputs.iters === undefined ? undefined : asNumber(inputs.iters, "inputs.iters");
    for (const k of orders) {
      const { alpha, betaK } = bdfCoefficients(k);
      const res = lmmRootModulus(alpha, betaK, z, { iters, withResidual: true });
      out[`mod_bdf${k}`] = res.mod;
      out[`resid_bdf${k}`] = res.residual ?? [];
    }
    return out;
  }
  throw new Error(`未知的 stability op=${op}（只有 function / lmm_roots）`);
}

const ALGOS: Record<string, (inputs: Record<string, unknown>) => Outputs> = {
  heat: computeHeat,
  lu: computeLu,
  amplification: computeAmplification,
  oscillator: computeOscillator,
  orderfit: computeOrderFit,
  power_iter: computePowerIter,
  stability: computeStability,
};

// --------------------------------------------------------------------------- //
// 主流程
// --------------------------------------------------------------------------- //

function main(argv: string[]): number {
  const here = dirname(fileURLToPath(import.meta.url));
  const rootArg = argValue(argv, "--root");
  const root = resolve(rootArg ?? findRoot(here));
  const casesPath = resolve(argValue(argv, "--cases") ?? join(root, "tests", "parity", "cases.json"));
  const outPath = resolve(argValue(argv, "--out") ?? join(root, "dist", "reports", "parity_ts.json"));

  const casesFile = JSON.parse(readFileSync(casesPath, "utf8")) as CasesFile;
  if (!Array.isArray(casesFile.cases)) throw new Error(`${casesPath} 缺少 cases 数组`);
  if (new Set(casesFile.cases.map((c) => c.name)).size !== casesFile.cases.length) {
    throw new Error(`${casesPath} 里有重名 case`);
  }

  const report: {
    generated_at: string;
    runner: string;
    node: string;
    root: string;
    cases_file: string;
    n_cases: number;
    n_errors: number;
    cases: Record<string, { algo: string; outputs?: Outputs; error?: string }>;
  } = {
    generated_at: new Date().toISOString(),
    runner: "viz/tests/parity/run_parity.ts",
    node: process.version,
    root,
    cases_file: casesPath,
    n_cases: casesFile.cases.length,
    n_errors: 0,
    cases: {},
  };

  for (const entry of casesFile.cases) {
    const goldenPath = resolve(root, entry.golden);
    try {
      if (!existsSync(goldenPath)) throw new Error(`golden 文件不存在：${entry.golden}（先跑 python tools/parity.py --update-golden）`);
      const golden = JSON.parse(readFileSync(goldenPath, "utf8")) as GoldenDoc;
      if (golden.case !== entry.name) {
        throw new Error(`golden 里的 case=${golden.case} 与索引 ${entry.name} 不一致`);
      }
      const compute = ALGOS[entry.algo];
      if (compute === undefined) throw new Error(`未知 algo=${entry.algo}`);
      const outputs = pickOutputs(entry.name, entry.outputs, compute(asRecord(golden.inputs, "inputs")));
      report.cases[entry.name] = { algo: entry.algo, outputs };
      process.stderr.write(`[parity] ${entry.name}: ${entry.outputs.length} 个输出键\n`);
    } catch (err) {
      // 一个 case 挂掉不能拖垮整份报告：Python 侧需要看到"哪个 case 为什么挂"
      report.cases[entry.name] = { algo: entry.algo, error: err instanceof Error ? err.message : String(err) };
      report.n_errors += 1;
      process.stderr.write(`[parity] ${entry.name}: 失败 —— ${report.cases[entry.name].error}\n`);
    }
  }

  mkdirSync(dirname(outPath), { recursive: true });
  writeFileSync(outPath, `${JSON.stringify(report, null, 2)}\n`, { encoding: "utf8" });
  process.stderr.write(`[parity] 已写 ${outPath}（${report.n_cases} 个 case，${report.n_errors} 个失败）\n`);
  return report.n_errors === 0 ? 0 : 1;
}

if (process.argv.includes("--help") || process.argv.includes("-h")) {
  process.stdout.write(
    "用法：node dist/viz/tests/parity/run_parity.js [--root DIR] [--cases FILE] [--out FILE]\n" +
      "  --root   仓库根（缺省：从本文件向上找含 tests/parity/cases.json 的目录）\n" +
      "  --cases  case 索引（缺省 <root>/tests/parity/cases.json）\n" +
      "  --out    报告输出（缺省 <root>/dist/reports/parity_ts.json）\n",
  );
} else {
  try {
    process.exitCode = main(process.argv.slice(2));
  } catch (err) {
    process.stderr.write(`[parity] runner 失败：${err instanceof Error ? err.message : String(err)}\n`);
    process.exitCode = 1;
  }
}
