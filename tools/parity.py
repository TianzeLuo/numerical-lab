"""``tools/parity.py`` —— Python ↔ TypeScript 双实现对拍（CI 门禁 G7，规格见 ``docs/PROTOCOLS.md`` §7）。

为什么需要这个脚本
------------------
双实现策略（同一算法在 Python 与 TS 各写一遍）只有在"可执行地证明两侧一致"时才算数，
否则前端会静默撒谎。本脚本把这件事变成一条门禁：

* **golden 由 Python 生成并落盘**（``tests/parity/golden/<case>.json``，小数据用 JSON 便于 diff），
  输入与输出都在里面，且**逐字节确定性**：无时间戳、键排序、UTF-8、LF 行尾、``repr`` 精度；
* **TS 侧 runner**（``viz/tests/parity/run_parity.ts`` → ``dist/viz/tests/parity/run_parity.js``）
  读同一批 case 的**输入**、调用 ``viz/shared/src/numeric/**``、写 ``dist/reports/parity_ts.json``；
* 本脚本重算 Python 侧（``numcore``），与 TS 输出**逐点**比较，容差取自
  ``tests/parity/tolerances.yaml``，报告落 ``dist/reports/parity.json``；
* 失败时 MUST 打印 ``case / 索引 / python 值 / ts 值 / 相对差 / 容差`` 六项，并给出"最大超差 Top 5"。

三种模式
--------
======================  ==================================================================
``--update-golden``     重算并写盘 golden 与 ``cases.json``（后者含每个 golden 的 sha256 校验和）
``--check-golden``      重算 Python 侧并与磁盘 golden **逐字节**比对（防"golden 过期"静默通过）
（默认）                对拍 TS 报告；缺失时自动跑 ``node dist/viz/tests/parity/run_parity.js``
======================  ==================================================================

退出码：0 = 全过 / 生成成功；1 = 有超差、缺失或 golden 过期。

case 的**定义**（算法、参数、容差键、输出键）只有一处真源：本文件的 :data:`CASE_SPECS`。
``cases.json`` 与 ``golden/**`` 都是它的生成物——"索引与基准数据漂移"这类问题因此在结构上不存在。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

from numcore.smoke import (  # noqa: E402
    damped_oscillator,
    ftcs_amplification,
    heat_ftcs,
    initial_profile,
    oscillator_energy,
)
from numcore.linalg.lu import lu_decompose  # noqa: E402
from numcore.verify.order import fit_order  # noqa: E402

try:  # pyyaml 在 pyproject 的 dev 依赖里；缺了就明确报错，不静默降级
    import yaml  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - 环境缺依赖
    yaml = None  # type: ignore[assignment]

# --------------------------------------------------------------------------- #
# 路径常量（一律相对仓库根，便于 --root 指向临时目录做"可重生成"测试）
# --------------------------------------------------------------------------- #
GOLDEN_REL = pathlib.Path("tests/parity/golden")
CASES_REL = pathlib.Path("tests/parity/cases.json")
TOL_REL = pathlib.Path("tests/parity/tolerances.yaml")
TS_REPORT_REL = pathlib.Path("dist/reports/parity_ts.json")
REPORT_REL = pathlib.Path("dist/reports/parity.json")
RUNNER_REL = pathlib.Path("dist/viz/tests/parity/run_parity.js")
BUILD_HINT = "npx tsc -p tsconfig.json"

#: 本目录下**不属于** G7 对拍的 golden 文件前缀：``nlf_*.json`` 是 PROTOCOLS §1 的
#: 解码基准（由 ``tools/gen_fixtures.py`` 生成、``viz/tests/nlf.test.ts`` 消费）。
#: 显式列出而不是"忽略所有不认识的 json"，是为了让"golden 目录里多了个没人认领的文件"
#: 仍然能被 :func:`check_artifacts` 抓到。
FOREIGN_GOLDEN_PREFIXES: tuple[str, ...] = ("nlf_",)


class ParityError(RuntimeError):
    """对拍链路本身出错（缺报告、缺依赖、golden 过期……），区别于"数值超差"。"""


# --------------------------------------------------------------------------- #
# 容差
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Tolerance:
    """一个 case 的容差：``|py − ts| ≤ atol + rtol·|py|``（与 ``numpy.allclose`` 同口径）。"""

    rtol: float
    atol: float
    reason: str = ""
    source: str = "default"

    def limit(self, py_value: float) -> float:
        """给定 python 侧的参考值，返回本次比较的绝对容差上限。"""
        return self.atol + self.rtol * abs(py_value)

    def to_dict(self) -> dict[str, Any]:
        return {"rtol": self.rtol, "atol": self.atol, "reason": self.reason, "source": self.source}


def _parse_tolerance(raw: Any, *, where: str, source: str, require_reason: bool) -> Tolerance:
    """把 YAML 里的一小段映射解析成 :class:`Tolerance`（键名/类型/正负都在这里把关）。"""
    if not isinstance(raw, dict):
        raise ParityError(f"{where} 必须是映射（rtol/atol/reason），得到 {type(raw).__name__}")
    extra = set(raw) - {"rtol", "atol", "reason"}
    if extra:
        raise ParityError(f"{where} 出现未知键 {sorted(extra)}（只允许 rtol / atol / reason）")
    for key in ("rtol", "atol"):
        if key not in raw:
            raise ParityError(f"{where} 缺少 {key}")
    rtol = float(raw["rtol"])
    atol = float(raw["atol"])
    if not math.isfinite(rtol) or not math.isfinite(atol) or rtol < 0 or atol < 0:
        raise ParityError(f"{where} 的 rtol/atol 必须是非负有限数，得到 rtol={rtol!r}, atol={atol!r}")
    if rtol == 0.0 and atol == 0.0:
        raise ParityError(f"{where} 不允许 rtol=atol=0：那等于要求逐位相同，任何跨语言实现都过不了")
    reason = str(raw.get("reason", "") or "").strip()
    if require_reason and not reason:
        raise ParityError(f"{where} 必须写 reason（说清误差来源：求和顺序 / 三角查表 / 舍入地板……）")
    return Tolerance(rtol=rtol, atol=atol, reason=reason, source=source)


def load_tolerances(path: pathlib.Path) -> tuple[Tolerance, dict[str, Tolerance]]:
    """读 ``tolerances.yaml``，返回 ``(默认容差, {case 名: 覆盖容差})``。

    结构（PROTOCOLS §7）::

        default: {rtol: 1e-12, atol: 1e-14, reason: "..."}
        cases:
          heat_explicit_step: {rtol: 1e-15, atol: 1e-16, reason: "..."}
    """
    if yaml is None:  # pragma: no cover - 环境缺依赖
        raise ParityError("读取 tolerances.yaml 需要 pyyaml（pyproject 的 dev 依赖）：pip install pyyaml")
    if not path.exists():
        raise ParityError(f"容差文件不存在：{path}（PROTOCOLS §7 要求容差写在仓库里）")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ParityError(f"{path} 顶层必须是映射")
    extra = set(raw) - {"default", "cases"}
    if extra:
        raise ParityError(f"{path} 顶层出现未知键 {sorted(extra)}（只允许 default / cases）")
    default = _parse_tolerance(raw.get("default"), where=f"{path}: default", source="default", require_reason=True)
    cases_raw = raw.get("cases", {}) or {}
    if not isinstance(cases_raw, dict):
        raise ParityError(f"{path}: cases 必须是映射")
    overrides: dict[str, Tolerance] = {}
    for name, item in cases_raw.items():
        overrides[str(name)] = _parse_tolerance(
            item, where=f"{path}: cases.{name}", source=str(name), require_reason=True
        )
    return default, overrides


def resolve_tolerance(name: str, default: Tolerance, overrides: dict[str, Tolerance]) -> Tolerance:
    """取某 case 的容差：有覆盖用覆盖，否则用默认（默认的 ``source`` 保持 ``"default"``）。"""
    return overrides.get(name, default)


# --------------------------------------------------------------------------- #
# case 定义（唯一真源）
# --------------------------------------------------------------------------- #
def _power_law_grid(n: int) -> list[float]:
    """``h = 2^-1 … 2^-n``（2 的负整数次幂可被 double 精确表示，两侧回读后完全相同）。"""
    return [2.0**-k for k in range(1, n + 1)]


def _order_groups() -> list[dict[str, Any]]:
    """5 组人造幂律：4 组纯幂律（含 skip/head/tail 三种窗口）+ 1 组带舍入地板。

    为什么要有"带舍入地板"那一组：误差序列尾部一旦被舍入误差淹没，双对数斜率会**系统性偏小**，
    而这种污染在两条实现上都存在、且随求和顺序略有不同——它是对拍里最需要显式覆盖的形态。
    """
    g1_h = _power_law_grid(12)
    g2_h = _power_law_grid(10)
    g3_h = _power_law_grid(12)
    g4_h = _power_law_grid(14)
    g5_h = _power_law_grid(14)
    return [
        {
            "name": "pure_p2",
            "h": g1_h,
            "err": [0.37 * h**2 for h in g1_h],
            "opts": {},
            "note": "纯 p=2 幂律，全窗口",
        },
        {
            "name": "preasym_p2_skip3",
            "h": g2_h,
            "err": [5.0 * h**2 + 0.02 * h for h in g2_h],
            "opts": {"skip": 3},
            "note": "p=2 主项 + O(h) 预渐近污染，丢前 3 点",
        },
        {
            "name": "p3_head7",
            "h": g3_h,
            "err": [1.7 * h**3 for h in g3_h],
            "opts": {"head": 7},
            "note": "纯 p=3，只保留最粗的 7 个点（head 保留粗网格端）",
        },
        {
            "name": "p05_skip1_tail9",
            "h": g4_h,
            "err": [0.9 * h**0.5 for h in g4_h],
            "opts": {"skip": 1, "tail": 9},
            "note": "p=0.5，skip=1 + tail=9（tail 保留细网格端）",
        },
        {
            "name": "rounding_floor_head6",
            "h": g5_h,
            "err": [2.5 * h**2 + 1e-6 for h in g5_h],
            "opts": {"head": 6},
            "note": "p=2 主项 + 1e-6 舍入地板，只保留粗端 6 点，否则斜率被地板拖低",
        },
    ]


#: 对拍覆盖清单（PROTOCOLS §7 的"覆盖清单（M0 起逐步补齐）"在本仓库的第一批落地）。
CASE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": "heat_explicit_step",
        "algo": "heat",
        "tolerance": "heat_explicit_step",
        "note": "单步 FTCS（nx=41, r=0.4, u0=sin(πx)）：u_i' = u_i + r(u_{i+1} − 2u_i + u_{i−1})，两端固定 0",
        "outputs": ["u"],
        "params": {"op": "step", "nx": 41, "r": 0.4, "n_steps": 1, "initial": "sine", "k": 1},
    },
    {
        "name": "heat_ftcs_run",
        "algo": "heat",
        "tolerance": "heat_ftcs_run",
        "note": "40 步之后的 u（同网格同 r）：检验误差是否会随步数累积出可见漂移",
        "outputs": ["u"],
        "params": {"op": "run", "nx": 41, "r": 0.4, "n_steps": 40, "initial": "sine", "k": 1},
    },
    {
        "name": "ftcs_amplification_modes",
        "algo": "amplification",
        "tolerance": "ftcs_amplification_modes",
        "note": "nx=31、r=0.3 时 mode=1..8 的放大因子 g = 1 − 4r sin²(kπ/(2(nx−1)))",
        "outputs": ["g"],
        "params": {"nx": 31, "r": 0.3, "modes": list(range(1, 9))},
    },
    {
        "name": "oscillator_sample",
        "algo": "oscillator",
        "tolerance": "oscillator_sample",
        "note": "ω=1.3、ζ=0.05/1.0/1.5 三条分支在 t ∈ linspace(0,5,200) 上的 x、v 与比能量 E",
        "outputs": [
            "x_under",
            "v_under",
            "E_under",
            "x_crit",
            "v_crit",
            "E_crit",
            "x_over",
            "v_over",
            "E_over",
        ],
        "params": {
            "omega": 1.3,
            "x0": 1.0,
            "v0": 0.0,
            "t_spec": [0.0, 5.0, 200],
            "branches": [
                {"tag": "under", "zeta": 0.05},
                {"tag": "crit", "zeta": 1.0},
                {"tag": "over", "zeta": 1.5},
            ],
        },
    },
    {
        "name": "order_fit_powerlaw",
        "algo": "orderfit",
        "tolerance": "order_fit_powerlaw",
        "note": "4 组人造幂律 + 1 组带舍入地板；比 slope / intercept / r2 / nUsed / nDropped",
        "outputs": [
            f"{field}_{group['name']}"
            for group in _order_groups()
            for field in ("slope", "intercept", "r2", "nUsed", "nDropped")
        ],
        "params": {
            "groups": [
                {"name": g["name"], "n": len(g["h"]), "opts": g["opts"], "note": g["note"]} for g in _order_groups()
            ]
        },
    },
    {
        "name": "lu_partial_pivot",
        "algo": "lu",
        "tolerance": "lu_partial_pivot",
        "note": "1.2 章：Hilbert(6) 的部分主元 LU（乘子写回下三角 + 外积更新），比 L/U/置换/主元/行交换/ρ",
        "outputs": ["l", "u", "perm", "pivots", "rowSwaps", "rho"],
        "params": {"n": 6, "kind": "hilbert"},
    },
    {
        "name": "lu_wilkinson_growth",
        "algo": "lu",
        "tolerance": "lu_partial_pivot",
        "note": "1.2 章：Wilkinson(12) 的部分主元 LU——ρ = 2^11 的经典最坏情形，专门盯住选主元规则",
        "outputs": ["l", "u", "perm", "pivots", "rowSwaps", "rho"],
        "params": {"n": 12, "kind": "wilkinson"},
    },
    {
        "name": "lu_tiny_pivot",
        "algo": "lu",
        "tolerance": "lu_partial_pivot",
        "note": "1.2 章：首元 1e-14 的矩阵——不换行就会把小主元当主元，ρ 会爆炸；这条 case 守住'到底换没换行'",
        "outputs": ["l", "u", "perm", "pivots", "rowSwaps", "rho"],
        "params": {"n": 5, "kind": "mixed"},
    },
)

#: 算法名 → Python 侧构造器（TS 侧 ``run_parity.ts`` 里必须有同名的分支）。
ALGOS: tuple[str, ...] = ("heat", "amplification", "oscillator", "orderfit", "lu")


def case_by_name(name: str) -> dict[str, Any]:
    """按名字取 case 定义（找不到即抛 :class:`ParityError`）。"""
    for spec in CASE_SPECS:
        if spec["name"] == name:
            return spec
    raise ParityError(f"未知 case：{name}")


# --------------------------------------------------------------------------- #
# Python 侧：由 case 定义算出输入与标准输出
# --------------------------------------------------------------------------- #
def _floats(values: Any) -> list[float]:
    """numpy 数组 / 列表 → 纯 float 列表（``float()`` 保证写出的是 double 本体）。"""
    return [float(v) for v in np.asarray(values, dtype=float).ravel()]


def _build_heat(params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """FTCS：初值用 ``initial_profile``，步进调用 ``numcore.smoke.heat_ftcs``。

    ``heat_ftcs`` 自己按 ``n_steps = ceil(t_end/Δt)`` 决定步数。为了让 TS 侧的
    ``ftcsStep`` / ``ftcsRun`` 与 Python 侧**步数完全一致**，这里把 ``t_end`` 反算成
    ``n_steps·Δt``，并在生成 golden 时断言 ceil 之后仍是 ``n_steps``——
    宁可在这里响亮地失败，也不要写出一份"步数对不上"的基准数据。
    """
    nx = int(params["nx"])
    r = float(params["r"])
    k = int(params["k"])
    n_steps = int(params["n_steps"])
    alpha = 1.0
    dx = 1.0 / (nx - 1)
    dt = r * dx * dx / alpha
    t_end = n_steps * dt
    x = np.linspace(0.0, 1.0, nx)
    u0 = initial_profile(x, str(params["initial"]), k)
    run = heat_ftcs(nx=nx, alpha=alpha, t_end=t_end, r=r, k=k, initial=str(params["initial"]))
    if run.n_steps != n_steps:
        raise ParityError(
            f"heat_ftcs 实际步数 {run.n_steps} ≠ 目标 {n_steps}（t_end={t_end!r}, dt={dt!r}）："
            "ceil 取整把步数改了，请调整 n_steps 或显式传入 t_end"
        )
    inputs = {
        "op": str(params["op"]),
        "nx": nx,
        "r": r,
        "n_steps": n_steps,
        "u0": _floats(u0),
    }
    outputs = {"u": _floats(run.u)}
    meta = {"alpha": alpha, "dt": float(run.dt), "n_steps": run.n_steps, "t_final": float(run.n_steps * run.dt)}
    return inputs, outputs, meta


def _build_amplification(params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """放大因子数组：``ftcs_amplification(r, mode, nx)`` 逐个模态求值。"""
    nx = int(params["nx"])
    r = float(params["r"])
    modes = [int(m) for m in params["modes"]]
    inputs = {"nx": nx, "r": r, "modes": modes}
    outputs = {"g": [float(ftcs_amplification(r, mode, nx)) for mode in modes]}
    return inputs, outputs, {"formula": "g = 1 - 4 r sin^2(mode*pi/(2(nx-1)))"}


def _build_oscillator(params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """三条阻尼分支的解析采样：``damped_oscillator`` + ``oscillator_energy``。"""
    omega = float(params["omega"])
    x0 = float(params.get("x0", 1.0))
    v0 = float(params.get("v0", 0.0))
    t0, t1, num = params["t_spec"]
    t = np.linspace(float(t0), float(t1), int(num))
    branches = [{"tag": str(b["tag"]), "zeta": float(b["zeta"])} for b in params["branches"]]
    outputs: dict[str, Any] = {}
    for branch in branches:
        x, v = damped_oscillator(t, omega, branch["zeta"], x0, v0)
        tag = branch["tag"]
        outputs[f"x_{tag}"] = _floats(x)
        outputs[f"v_{tag}"] = _floats(v)
        outputs[f"E_{tag}"] = _floats(oscillator_energy(x, v, omega))
    inputs = {"omega": omega, "x0": x0, "v0": v0, "t": _floats(t), "branches": branches}
    meta = {"n_samples": int(num), "t_span": [float(t0), float(t1)]}
    return inputs, outputs, meta


def _build_orderfit(params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """收敛阶拟合：``numcore.verify.order.fit_order``（与 TS ``fitOrderParity`` 同窗口同选项）。"""
    groups: list[dict[str, Any]] = []
    outputs: dict[str, Any] = {}
    for group in _order_groups():
        opts = dict(group["opts"])
        fit = fit_order(
            group["h"],
            group["err"],
            skip=int(opts.get("skip", 0)),
            head=opts.get("head"),
            tail=opts.get("tail"),
            min_points=int(opts.get("min_points", 3)),
        )
        groups.append(
            {
                "name": group["name"],
                "h": [float(v) for v in group["h"]],
                "err": [float(v) for v in group["err"]],
                "opts": opts,
            }
        )
        outputs[f"slope_{group['name']}"] = float(fit.slope)
        outputs[f"intercept_{group['name']}"] = float(fit.intercept)
        outputs[f"r2_{group['name']}"] = float(fit.r2)
        outputs[f"nUsed_{group['name']}"] = int(fit.n_used)
        outputs[f"nDropped_{group['name']}"] = int(fit.n_dropped)
    return {"groups": groups}, outputs, {"groups": [g["name"] for g in groups]}


def _build_lu(params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """部分主元 LU：输入是行主序展平的方阵，输出 L/U/置换/主元/行交换/增长因子。

    矩阵由**参数生成**（而不是在 case 里写死 36 个数字）：`kind` 决定矩阵族，
    这样"换成 Wilkinson 矩阵"只是改一个字符串，不必手抄一堆浮点数。
    生成规则必须在 TS 侧可复现——不，恰恰相反：**生成规则不复制到 TS 侧**，
    生成好的矩阵作为 `inputs.a` 落盘，TS 只读回 double 再分解（PROTOCOLS §7：
    "输入一律由 Python 生成落盘，避免输入构造这一环被两套实现的差异污染"）。
    """
    n = int(params["n"])
    kind = str(params["kind"])
    if kind == "hilbert":
        i = np.arange(1, n + 1, dtype=float)
        a = 1.0 / (i[:, None] + i[None, :] - 1.0)
    elif kind == "wilkinson":
        from numcore.linalg.lu import wilkinson_matrix

        a = wilkinson_matrix(n)
    elif kind == "mixed":
        # 主元会被"逼"出来的矩阵：小主元 + 一般项，避免第一个主元就恰好最大
        idx = np.arange(n, dtype=float)
        a = np.cos(idx[:, None] * 1.7 + idx[None, :] * 0.9) + 0.3 * idx[:, None] - 0.2 * idx[None, :]
        a[0, 0] = 1e-14
    else:
        raise ParityError(f"未知的 lu kind={kind!r}（可用：hilbert / wilkinson / mixed）")

    res = lu_decompose(a, pivot="partial")
    inputs = {"n": n, "kind": kind, "a": _floats(a)}
    outputs = {
        "l": _floats(res.L),
        "u": _floats(res.U),
        "perm": _floats(res.perm),
        "pivots": _floats(res.pivots),
        "rowSwaps": _floats([x for pair in res.row_swaps for x in pair]),
        "rho": float(res.rho),
    }
    meta = {
        "pivot": "partial",
        "row_swaps": [list(pair) for pair in res.row_swaps],
        "ops": int(res.ops),
        "singular": bool(res.singular),
    }
    return inputs, outputs, meta


_BUILDERS = {
    "lu": _build_lu,
    "heat": _build_heat,
    "amplification": _build_amplification,
    "oscillator": _build_oscillator,
    "orderfit": _build_orderfit,
}


def build_golden_document(spec: dict[str, Any]) -> dict[str, Any]:
    """把一个 case 定义算成 golden 文档（``{case, algo, inputs, meta, outputs, note}``）。"""
    algo = str(spec["algo"])
    if algo not in _BUILDERS:
        raise ParityError(f"case {spec['name']} 的 algo={algo!r} 没有 Python 侧实现（可用：{list(ALGOS)}）")
    inputs, outputs, meta = _BUILDERS[algo](spec["params"])
    missing = [key for key in spec["outputs"] if key not in outputs]
    if missing:
        raise ParityError(f"case {spec['name']} 声明了输出 {missing}，但 Python 侧没有产出")
    extra = [key for key in outputs if key not in spec["outputs"]]
    if extra:
        raise ParityError(f"case {spec['name']} 的 Python 侧多产出 {extra}，未在 outputs 里声明")
    return {
        "case": str(spec["name"]),
        "algo": algo,
        "inputs": inputs,
        "meta": meta,
        "outputs": {key: outputs[key] for key in spec["outputs"]},
        "note": str(spec.get("note", "")),
    }


def python_outputs(spec: dict[str, Any]) -> dict[str, Any]:
    """只取 Python 侧输出（对拍用）。"""
    return build_golden_document(spec)["outputs"]


# --------------------------------------------------------------------------- #
# 确定性序列化 / 生成物
# --------------------------------------------------------------------------- #
def dumps_canonical(obj: Any) -> str:
    """确定性 JSON 文本：键排序、缩进 1、``ensure_ascii=False``、结尾恰好一个 LF。

    ``json.dumps`` 对 float 用 ``repr``（最短往返表示，17 位有效数字以内），
    因此写出去的 double 能被两侧逐位读回——这是"输入不引入额外误差"的前提。
    """
    return json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=True) + "\n"


def write_text_lf(path: pathlib.Path, text: str) -> None:
    """以 UTF-8 + LF 写文件（在 WSL/Windows 共享的仓库里必须锁死行尾）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def golden_rel_path(name: str) -> str:
    """golden 文件相对仓库根的 POSIX 路径（写进 cases.json，跨平台一致）。"""
    return (GOLDEN_REL / f"{name}.json").as_posix()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_cases_index(golden_texts: dict[str, str]) -> dict[str, Any]:
    """由 case 定义 + golden 文本生成 ``cases.json``（含每个 golden 的 sha256 校验和）。"""
    entries: list[dict[str, Any]] = []
    for spec in CASE_SPECS:
        name = str(spec["name"])
        entries.append(
            {
                "name": name,
                "algo": str(spec["algo"]),
                "golden": golden_rel_path(name),
                "sha256": _sha256(golden_texts[name]),
                "tolerance": str(spec["tolerance"]),
                "outputs": list(spec["outputs"]),
                "params": spec["params"],
                "note": str(spec.get("note", "")),
            }
        )
    return {
        "_generated": "本文件由 tools/parity.py --update-golden 生成（请勿手改）；case 定义见 tools/parity.py::CASE_SPECS",
        "version": 1,
        "golden_dir": GOLDEN_REL.as_posix(),
        "tolerance_file": TOL_REL.as_posix(),
        "cases": entries,
    }


def build_artifacts() -> tuple[dict[str, dict[str, Any]], dict[str, str], dict[str, Any], str]:
    """生成全部产物文本：``(golden 文档, golden 文本, cases 索引, cases 文本)``。"""
    docs: dict[str, dict[str, Any]] = {}
    texts: dict[str, str] = {}
    for spec in CASE_SPECS:
        doc = build_golden_document(spec)
        docs[str(spec["name"])] = doc
        texts[str(spec["name"])] = dumps_canonical(doc)
    cases_doc = build_cases_index(texts)
    return docs, texts, cases_doc, dumps_canonical(cases_doc)


def update_golden(root: pathlib.Path, *, quiet: bool = False) -> int:
    """``--update-golden``：重算并写盘 golden 与 ``cases.json``。"""
    _, texts, _, cases_text = build_artifacts()
    for name, text in texts.items():
        write_text_lf(root / GOLDEN_REL / f"{name}.json", text)
    write_text_lf(root / CASES_REL, cases_text)
    if not quiet:
        print(f"已生成 {len(texts)} 份 golden → {(root / GOLDEN_REL).as_posix()}")
        for name, text in texts.items():
            print(f"  · {name + '.json':32s} {len(text.encode('utf-8')):>7d} B  sha256={_sha256(text)[:12]}…")
        print(f"已更新 case 索引 → {(root / CASES_REL).as_posix()}（含 sha256 校验和）")
    return 0


def check_artifacts(root: pathlib.Path) -> list[str]:
    """重算 Python 侧并与磁盘比对；返回"过期/缺失/多余"的文件路径列表（空 = 一致）。"""
    _, texts, _, cases_text = build_artifacts()
    stale: list[str] = []
    for name, text in texts.items():
        path = root / GOLDEN_REL / f"{name}.json"
        rel = path.relative_to(root).as_posix()
        if not path.exists():
            stale.append(f"{rel}（缺失）")
        elif path.read_text(encoding="utf-8") != text:
            stale.append(f"{rel}（内容过期）")
    cases_path = root / CASES_REL
    if not cases_path.exists():
        stale.append(f"{CASES_REL.as_posix()}（缺失）")
    elif cases_path.read_text(encoding="utf-8") != cases_text:
        stale.append(f"{CASES_REL.as_posix()}（内容过期：校验和或 case 定义已变）")
    # "不多不少"：golden 目录里属于 G7 的 json 必须都能被某个 case 认领
    golden_dir = root / GOLDEN_REL
    owned = {f"{name}.json" for name in texts}
    if golden_dir.exists():
        for path in sorted(golden_dir.glob("*.json")):
            if path.name in owned or path.name.startswith(FOREIGN_GOLDEN_PREFIXES):
                continue
            stale.append(f"{path.relative_to(root).as_posix()}（多余：没有任何 case 引用它）")
    return stale


# --------------------------------------------------------------------------- #
# TS 报告
# --------------------------------------------------------------------------- #
def _tail(text: str, limit: int = 1500) -> str:
    text = text.strip()
    return text if len(text) <= limit else "…" + text[-limit:]


def load_ts_report(
    root: pathlib.Path, *, auto_run: bool = True, timeout: float = 900.0
) -> tuple[dict[str, Any], pathlib.Path, str]:
    """读 ``dist/reports/parity_ts.json``；缺失时（可选）自动跑 node runner。

    返回 ``(报告, 报告路径, 说明)``。任何"拿不到报告"的情况都抛 :class:`ParityError`，
    且错误信息里必须给出构建命令——否则使用者只会看到一句"文件不存在"。
    """
    path = root / TS_REPORT_REL
    runner = root / RUNNER_REL
    note = "磁盘上已有的报告"
    if not path.exists() and auto_run:
        if not runner.exists():
            raise ParityError(
                f"未找到 TS 对拍报告 {TS_REPORT_REL.as_posix()}，且编译产物 {RUNNER_REL.as_posix()} 也不存在。\n"
                f"  请先构建：{BUILD_HINT}（或 npm run build）\n"
                f"  再运行：node {RUNNER_REL.as_posix()}"
            )
        try:
            proc = subprocess.run(
                ["node", str(runner), "--root", str(root)],
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ParityError(
                f"自动运行 node {RUNNER_REL.as_posix()} 失败：{exc}\n  请先构建：{BUILD_HINT}"
            ) from exc
        note = f"自动运行 node {RUNNER_REL.as_posix()}（退出码 {proc.returncode}）"
        if not path.exists():
            raise ParityError(
                f"自动运行 node {RUNNER_REL.as_posix()} 未产出 {TS_REPORT_REL.as_posix()}（退出码 {proc.returncode}）：\n"
                f"{_tail(proc.stderr or proc.stdout)}\n  请先构建：{BUILD_HINT}"
            )
        if proc.returncode != 0:
            note += "；runner 报了 case 级错误，明细见报告"
    if not path.exists():
        raise ParityError(
            f"未找到 TS 对拍报告 {TS_REPORT_REL.as_posix()}。\n"
            f"  请先构建并运行：{BUILD_HINT} && node {RUNNER_REL.as_posix()}\n"
            f"  或去掉 --no-auto-run 让本脚本自动调用 node"
        )
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ParityError(f"{path} 不是合法 JSON：{exc}（重新构建：{BUILD_HINT}）") from exc
    if not isinstance(doc, dict) or not isinstance(doc.get("cases"), dict):
        raise ParityError(f"{path} 结构不对：顶层缺少 cases 映射（重新构建：{BUILD_HINT}）")
    return doc, path, note


# --------------------------------------------------------------------------- #
# 逐点比较
# --------------------------------------------------------------------------- #
def _as_vector(value: Any) -> list[float] | None:
    """标量/数组 → float 列表；``bool``、字符串、嵌套数组一律视为非法（返回 None）。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        out: list[float] = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                return None
            out.append(float(item))
        return out
    return None


@dataclass(frozen=True)
class Mismatch:
    """一个超差点（或在比较阶段就无法成立的点）。"""

    case: str
    key: str
    index: str
    py: float
    ts: float
    rel_diff: float
    ratio: float
    tol: Tolerance
    limit: float
    note: str = ""

    def line(self) -> str:
        """PROTOCOLS §7 要求的六项：case / 索引 / python 值 / ts 值 / 相对差 / 容差。"""
        return (
            f"case={self.case}  index={self.key}[{self.index}]  python={self.py!r}  ts={self.ts!r}  "
            f"rel_diff={self.rel_diff!r}  tol=rtol={self.tol.rtol:g},atol={self.tol.atol:g}"
            f"(limit={self.limit!r})"
            + (f"  note={self.note}" if self.note else "")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case": self.case,
            "key": self.key,
            "index": self.index,
            "python": self.py,
            "ts": self.ts,
            "rel_diff": self.rel_diff,
            "ratio": self.ratio,
            "tolerance": self.tol.to_dict(),
            "limit": self.limit,
            "note": self.note,
        }


def _rel_diff(py: float, ts: float) -> float:
    """相对差 ``|py − ts| / |py|``；``py = 0`` 时退化为 0 或 ``inf``（不假装是有限数）。"""
    diff = abs(py - ts)
    if py != 0.0 and math.isfinite(py) and math.isfinite(ts):
        return diff / abs(py)
    if diff == 0.0:
        return 0.0
    return math.inf


def _ratio(diff: float, limit: float) -> float:
    """超差倍数：``> 1`` 才算失败（恰好等于容差 = 通过，与 ``<=`` 判定一致）。"""
    if limit > 0.0:
        return diff / limit
    return 0.0 if diff == 0.0 else math.inf


def compare_case(
    spec: dict[str, Any],
    py_outputs: dict[str, Any],
    ts_outputs: dict[str, Any] | None,
    tol: Tolerance,
) -> tuple[list[Mismatch], dict[str, Any]]:
    """逐点比较一个 case，返回 ``(超差列表, 统计)``。

    判定口径：``|py − ts| ≤ atol + rtol·|py|``（**恰好等于容差算通过**）。
    非有限值（NaN/Inf）**一律判失败**并给出专门说明——绝不因为"两边都是 NaN"就放行：
    NaN 参与任何比较都返回 False，若照搬 ``allclose`` 的写法会得到"静默通过"。
    """
    case = str(spec["name"])
    stats: dict[str, Any] = {
        "n_points": 0,
        "n_mismatch": 0,
        "max_abs_diff": 0.0,
        "max_rel_diff": 0.0,
        "max_ratio": 0.0,
    }
    mismatches: list[Mismatch] = []
    if ts_outputs is None:
        mismatches.append(
            Mismatch(
                case=case,
                key="-",
                index="-",
                py=math.nan,
                ts=math.nan,
                rel_diff=math.nan,
                ratio=math.inf,
                tol=tol,
                limit=math.nan,
                note="TS 报告里没有这个 case（runner 未跑 / case 改名 / runner 崩了）",
            )
        )
        stats["n_mismatch"] = 1
        return mismatches, stats
    if not isinstance(ts_outputs, dict):
        raise ParityError(f"case {case} 的 TS 输出必须是映射，得到 {type(ts_outputs).__name__}")

    for key in spec["outputs"]:
        if key not in py_outputs:
            raise ParityError(f"case {case} 的 Python 侧缺少输出键 {key}（golden 与 case 定义漂移）")
        if key not in ts_outputs:
            mismatches.append(
                Mismatch(
                    case=case,
                    key=key,
                    index="-",
                    py=math.nan,
                    ts=math.nan,
                    rel_diff=math.nan,
                    ratio=math.inf,
                    tol=tol,
                    limit=math.nan,
                    note="TS 报告缺少该输出键（runner 与 cases.json 漂移？）",
                )
            )
            continue
        py_vec = _as_vector(py_outputs[key])
        ts_vec = _as_vector(ts_outputs[key])
        if py_vec is None:
            raise ParityError(f"case {case} 的 Python 输出 {key} 不是数值数组（golden 坏了）")
        if ts_vec is None:
            mismatches.append(
                Mismatch(
                    case=case,
                    key=key,
                    index="-",
                    py=math.nan,
                    ts=math.nan,
                    rel_diff=math.nan,
                    ratio=math.inf,
                    tol=tol,
                    limit=math.nan,
                    note=f"TS 输出 {key} 不是数值（{type(ts_outputs[key]).__name__}）",
                )
            )
            continue
        if len(py_vec) != len(ts_vec):
            mismatches.append(
                Mismatch(
                    case=case,
                    key=key,
                    index="-",
                    py=math.nan,
                    ts=math.nan,
                    rel_diff=math.nan,
                    ratio=math.inf,
                    tol=tol,
                    limit=math.nan,
                    note=f"长度不一致：python {len(py_vec)} vs ts {len(ts_vec)}",
                )
            )
            continue
        scalar = len(py_vec) == 1
        for i, (py_value, ts_value) in enumerate(zip(py_vec, ts_vec)):
            index = "-" if scalar else str(i)
            stats["n_points"] += 1
            if not (math.isfinite(py_value) and math.isfinite(ts_value)):
                bad = []
                if not math.isfinite(py_value):
                    bad.append("python")
                if not math.isfinite(ts_value):
                    bad.append("ts")
                mismatches.append(
                    Mismatch(
                        case=case,
                        key=key,
                        index=index,
                        py=py_value,
                        ts=ts_value,
                        rel_diff=math.nan,
                        ratio=math.inf,
                        tol=tol,
                        limit=math.nan,
                        note=f"非有限值（{'/'.join(bad)} 侧为 NaN/Inf）：不参与数值比较，直接判失败",
                    )
                )
                continue
            diff = abs(py_value - ts_value)
            limit = tol.limit(py_value)
            rel = _rel_diff(py_value, ts_value)
            ratio = _ratio(diff, limit)
            stats["max_abs_diff"] = max(stats["max_abs_diff"], diff)
            if math.isfinite(rel):
                stats["max_rel_diff"] = max(stats["max_rel_diff"], rel)
            if math.isfinite(ratio):
                stats["max_ratio"] = max(stats["max_ratio"], ratio)
            if diff > limit:  # 恰好等于容差 → 通过
                mismatches.append(
                    Mismatch(
                        case=case,
                        key=key,
                        index=index,
                        py=py_value,
                        ts=ts_value,
                        rel_diff=rel,
                        ratio=ratio,
                        tol=tol,
                        limit=limit,
                    )
                )
    stats["n_mismatch"] = len(mismatches)
    return mismatches, stats


def worst_mismatches(mismatches: Iterable[Mismatch], limit: int = 5) -> list[Mismatch]:
    """按"超差倍数"降序取前 ``limit`` 条（并列时按 case/key/index 稳定排序）。"""
    ordered = sorted(mismatches, key=lambda m: (-m.ratio, m.case, m.key, m.index))
    return ordered[:limit]


# --------------------------------------------------------------------------- #
# 主判定
# --------------------------------------------------------------------------- #
def evaluate(root: pathlib.Path, *, auto_run: bool = True, timeout: float = 900.0) -> dict[str, Any]:
    """跑完整条对拍链路，返回报告字典（不打印、不退出，便于测试直接调用）。"""
    default_tol, overrides = load_tolerances(root / TOL_REL)
    stale = check_artifacts(root)
    ts_doc, ts_path, ts_note = load_ts_report(root, auto_run=auto_run, timeout=timeout)
    ts_cases = ts_doc.get("cases", {})

    case_reports: list[dict[str, Any]] = []
    all_mismatches: list[Mismatch] = []
    n_points = 0
    for spec in CASE_SPECS:
        name = str(spec["name"])
        tol = resolve_tolerance(str(spec["tolerance"]), default_tol, overrides)
        py_outputs = python_outputs(spec)
        ts_entry = ts_cases.get(name)
        ts_outputs = ts_entry.get("outputs") if isinstance(ts_entry, dict) else None
        ts_error = ts_entry.get("error") if isinstance(ts_entry, dict) else None
        mismatches, stats = compare_case(spec, py_outputs, ts_outputs, tol)
        if ts_error:
            mismatches = list(mismatches) + [
                Mismatch(
                    case=name,
                    key="-",
                    index="-",
                    py=math.nan,
                    ts=math.nan,
                    rel_diff=math.nan,
                    ratio=math.inf,
                    tol=tol,
                    limit=math.nan,
                    note=f"TS runner 报错：{ts_error}",
                )
            ]
            stats["n_mismatch"] = len(mismatches)
        n_points += int(stats["n_points"])
        all_mismatches.extend(mismatches)
        case_reports.append(
            {
                "name": name,
                "algo": spec["algo"],
                "tolerance_key": spec["tolerance"],
                "tolerance": tol.to_dict(),
                "outputs": list(spec["outputs"]),
                "ok": not mismatches,
                "n_points": stats["n_points"],
                "n_mismatch": stats["n_mismatch"],
                "max_abs_diff": stats["max_abs_diff"],
                "max_rel_diff": stats["max_rel_diff"],
                "max_ratio": stats["max_ratio"],
                "ts_error": ts_error,
                "mismatches": [m.to_dict() for m in worst_mismatches(mismatches, limit=10)],
            }
        )

    missing_ts_cases = [str(s["name"]) for s in CASE_SPECS if str(s["name"]) not in ts_cases]
    ok = not all_mismatches and not stale and not missing_ts_cases
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ok": ok,
        "root": str(root),
        "protocol": "docs/PROTOCOLS.md §7",
        "cases_file": CASES_REL.as_posix(),
        "tolerance_file": TOL_REL.as_posix(),
        "ts_report": ts_path.relative_to(root).as_posix() if ts_path.is_relative_to(root) else str(ts_path),
        "ts_report_note": ts_note,
        "ts_generated_at": ts_doc.get("generated_at"),
        "default_tolerance": default_tol.to_dict(),
        "n_cases": len(CASE_SPECS),
        "n_points": n_points,
        "n_mismatch": len(all_mismatches),
        "missing_ts_cases": missing_ts_cases,
        "golden_stale": stale,
        "cases": case_reports,
        "worst": [m.to_dict() for m in worst_mismatches(all_mismatches, limit=5)],
    }
    return report


def _fmt_float(value: Any) -> str:
    if isinstance(value, float):
        return repr(value)
    return str(value)


def print_report_human(report: dict[str, Any]) -> None:
    """人读输出：失败时按 PROTOCOLS §7 打印六项明细 + 最大超差 Top 5。"""
    n_cases = report["n_cases"]
    if report["golden_stale"]:
        print("golden 已过期（请运行 `python tools/parity.py --update-golden`）：")
        for item in report["golden_stale"]:
            print(f"  · {item}")
    if report["missing_ts_cases"]:
        print(f"TS 报告缺少 case：{', '.join(report['missing_ts_cases'])}（先跑 node {RUNNER_REL.as_posix()}）")
    if report["ok"]:
        print(
            f"对拍通过：{n_cases} 个 case、{report['n_points']} 个点全部落在容差内"
            f"（容差：{report['tolerance_file']}；TS 报告：{report['ts_report']}）"
        )
        for case in report["cases"]:
            tol = case["tolerance"]
            print(
                f"  · {case['name']:26s} {case['n_points']:>5d} 点  max|Δ|={_fmt_float(case['max_abs_diff']):<10s}"
                f" max_rel={_fmt_float(case['max_rel_diff']):<10s} 余量比={_fmt_float(case['max_ratio'])}"
                f"  (rtol={tol['rtol']:g}, atol={tol['atol']:g})"
            )
        return
    print(f"对拍失败：{n_cases} 个 case、{report['n_points']} 个点中出现 {report['n_mismatch']} 处超差/缺失")
    for case in report["cases"]:
        if case["ok"]:
            continue
        print(f"  ── case={case['name']}（{case['n_mismatch']} 处，最多列 10 条）")
        for item in case["mismatches"]:
            mismatch = Mismatch(
                case=item["case"],
                key=item["key"],
                index=item["index"],
                py=item["python"],
                ts=item["ts"],
                rel_diff=item["rel_diff"],
                ratio=item["ratio"],
                tol=Tolerance(
                    rtol=item["tolerance"]["rtol"],
                    atol=item["tolerance"]["atol"],
                    reason=item["tolerance"]["reason"],
                    source=item["tolerance"]["source"],
                ),
                limit=item["limit"],
                note=item["note"],
            )
            print(f"    {mismatch.line()}")
    if report["worst"]:
        print("  最大超差 Top 5（按 超差倍数 = |Δ| / 容差 排序）：")
        for rank, item in enumerate(report["worst"], start=1):
            print(
                f"    {rank}. case={item['case']}  index={item['key']}[{item['index']}]  "
                f"python={item['python']!r}  ts={item['ts']!r}  rel_diff={item['rel_diff']!r}  "
                f"tol=rtol={item['tolerance']['rtol']:g},atol={item['tolerance']['atol']:g}(limit={item['limit']!r})"
                f"  ratio={item['ratio']!r}"
            )
    print("  提示：--json 可拿机器可读版本")


def write_report(root: pathlib.Path, report: dict[str, Any], path: pathlib.Path | None = None) -> pathlib.Path:
    """把报告写到磁盘（缺省 ``<root>/dist/reports/parity.json``）。"""
    out = path or (root / REPORT_REL)
    write_text_lf(out, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return out


def run_gate(
    root: pathlib.Path,
    *,
    auto_run: bool = True,
    json_mode: bool = False,
    report_path: pathlib.Path | None = None,
    quiet: bool = False,
) -> int:
    """默认模式：对拍并落报告，返回退出码（0 全过 / 1 有超差或缺失）。"""
    try:
        report = evaluate(root, auto_run=auto_run)
    except ParityError as exc:
        if json_mode:
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        else:
            print(f"对拍无法进行：{exc}", file=sys.stderr)
        return 1
    out = write_report(root, report, report_path)
    if json_mode:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif not quiet:
        print_report_human(report)
        print(f"  报告：{out}")
    return 0 if report["ok"] else 1

def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口（三种模式，见模块 docstring）。"""
    parser = argparse.ArgumentParser(
        prog="tools/parity.py",
        description="Python ↔ TypeScript 双实现对拍（CI 门禁 G7，PROTOCOLS §7）",
    )
    parser.add_argument("--root", default=str(_ROOT), help="仓库根（缺省：本文件所在仓库）")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--update-golden", action="store_true", help="重算并写盘 golden 与 cases.json")
    mode.add_argument("--check-golden", action="store_true", help="重算 Python 侧并与磁盘 golden 逐字节比对")
    parser.add_argument("--json", action="store_true", help="把报告以 JSON 打到 stdout（机器可读）")
    parser.add_argument("--report", default=None, help=f"报告输出路径（缺省 {REPORT_REL.as_posix()}）")
    parser.add_argument("--no-auto-run", action="store_true", help="TS 报告缺失时不自动调用 node runner")
    parser.add_argument("--quiet", action="store_true", help="只输出错误与退出码")
    args = parser.parse_args(argv)
    root = pathlib.Path(args.root).resolve()

    if args.update_golden:
        return update_golden(root, quiet=args.quiet)
    if args.check_golden:
        stale = check_artifacts(root)
        if stale:
            print("golden 与 Python 侧重算结果不一致（已过期）：")
            for item in stale:
                print(f"  · {item}")
            print("  修复：python tools/parity.py --update-golden")
            return 1
        if not args.quiet:
            print(f"golden 与 Python 侧一致：{len(CASE_SPECS)} 个 case（{(root / GOLDEN_REL).as_posix()}）")
        return 0
    report_path = pathlib.Path(args.report).resolve() if args.report else None
    return run_gate(
        root,
        auto_run=not args.no_auto_run,
        json_mode=args.json,
        report_path=report_path,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    raise SystemExit(main())
