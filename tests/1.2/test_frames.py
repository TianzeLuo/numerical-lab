"""1.2 章的**帧级**回归测试：把前端 agent 真踩到的两个 bug 钉在 pytest 里。

为什么帧级测试要单独一个文件
----------------------------
`test_linalg.py` / `test_sparse_ordering.py` 测的是 `numcore` 的算法；而下面这两条
bug 都只在"把算法接到帧上"的那一层才暴露：

1. **前导块残差用了最终的置换**。第 j 步之后的后续交换还会改动位置 ≥ j 的行列，
   因此最终的 :math:`P`（甚至 :math:`PAQ`）都不是当时的 :math:`P_j`。
   症状：`pivot="complete"` 时 `residual_history` 是 O(1)（正确量级是 :math:`O(u)`），
   而 `pivot="none"`、非占优矩阵、n≥8 时同样会错——热力图上那条"残差曲线"于是在撒谎。
   修法：`LUResult.ref_history`（只交换、不消元）给出**当步**的参照块。

2. **κ ≳ 1/u 时 `unit_ball` 直接 500**。`shear ∈ {0, 2}` 且 `t = 17`（manifest 声明的
   时间轴右端）时 A 在 float64 下真的奇异，回代抛 `ZeroDivisionError`。
   现在它被当成一种**要显示的状态**：`meta.singular`、`meta.*=null`（JSON 没有 NaN）、
   `strings.status` 给出原因，而不是让整帧失败。
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from numcore.linalg.lu import (  # noqa: E402
    ROUNDOFF_HINT,
    example_matrix,
    lu_decompose,
    leading_block_residuals,
    wilkinson_matrix,
)
from numcore.verify import assertion, record  # noqa: E402
from server import nlf as nlf_mod  # noqa: E402
from server.frames import build_frame  # noqa: E402
from server.registry import Registry  # noqa: E402


def _entry():
    reg = Registry(ROOT)
    reg.discover(strict=False)  # 并发开发时兄弟章节可能是半成品；本章必须能加载
    return reg.get("1.2")


def _defaults(viz_id: str) -> dict:
    spec = next(v for v in _entry().module.MANIFEST["vizzes"] if v["id"] == viz_id)
    return {p["name"]: p["default"] for p in spec["params"]}


# --------------------------------------------------------------------------- #
# 1. P·A·Q = L·U：完全主元会换列，只做行置换是不够的
# --------------------------------------------------------------------------- #
@assertion("A", "P·A·Q = L·U（三种主元策略）", chapter="1.2")
def test_permuted_factorization_identity_holds_for_all_pivots() -> None:
    """``P @ A @ Q == L @ U``；奇异/极端增长时按 :math:`n u \\rho` 量级判定。

    这条测试的价值在于它**同时钉住了 Q 的约定**：`Q` 的写法与 `P` 互为转置
    （行置换在左、列置换在右），照抄 `P` 会得到 ``Wilkinson(12)`` 上 2.0 的偏差。
    """
    u = float(np.finfo(float).eps) / 2.0
    worst = 0.0
    n_cases = 0
    skipped_singular = 0
    for name in ("swap", "tiny_pivot", "wilkinson", "spd", "hilbert", "diag_dominant", "banded"):
        A = example_matrix(name, 6 if name != "wilkinson" else 12)
        for pivot in ("none", "partial", "complete"):
            res = lu_decompose(A, pivot=pivot)
            if res.singular:
                # 无主元 + 精确零主元时分解被定义为"跳过该列"，L·U 不再等于 A 的任何置换
                skipped_singular += 1
                continue
            err = float(np.max(np.abs(res.P() @ A @ res.Q() - res.L @ res.U)))
            bound = 16.0 * u * max(res.rho, 1.0) * max(1.0, float(np.max(np.abs(A))))
            ratio = err / bound
            worst = max(worst, ratio)
            assert ratio <= 1.0, f"{name}/{pivot}: |PAQ − LU| = {err:.3e} 超出 n·u·ρ 界 {bound:.3e}"
            n_cases += 1
    record(
        label="max|PAQ − LU| / (16·u·ρ·max|A|)",
        theory=1.0,
        measured=worst,
        tol="≤ 1",
        passed=worst <= 1.0,
        detail=f"{n_cases} 个非奇异情形；{skipped_singular} 个奇异情形按定义跳过",
    )


# --------------------------------------------------------------------------- #
# 2. 前导块残差必须用"当步"的参照矩阵（ref_history），不能用最终置换
# --------------------------------------------------------------------------- #
@assertion("B", "前导块残差 ≤ n·u·ρ（逐步、三种主元）", chapter="1.2")
def test_leading_block_residuals_are_roundoff_sized() -> None:
    """逐步残差 :math:`\\|A_{\\rm lead} - L_{\\rm lead}U_{\\rm lead}\\|_\\infty` 落在 :math:`n u \\rho` 量级。

    回归对象：曾经用最终的 :math:`PAQ` 取前导块，非占优矩阵上得到 O(1) 的假残差
    （部分主元 n=8 时 0.78、完全主元时 1.07）——那条曲线会直接被画进热力图的副轴。
    """
    u = float(np.finfo(float).eps) / 2.0
    worst = 0.0
    n_steps = 0
    for name in ("hilbert", "wilkinson", "diag_dominant", "banded", "spd"):
        A = example_matrix(name, 8 if name != "wilkinson" else 12)
        scale = max(1.0, float(np.max(np.abs(A))))
        for pivot in ("none", "partial", "complete"):
            res = lu_decompose(A, pivot=pivot, history=True)
            if res.singular:
                continue
            diffs = leading_block_residuals(res)
            assert len(diffs) == len(res.history), "每一步都应当有一条残差"
            for j, err in enumerate(diffs):
                m = j + 1
                bound = 8.0 * u * max(res.rho, 1.0) * scale * m
                ratio = err / bound if bound > 0 else 0.0
                worst = max(worst, ratio)
                n_steps += 1
                assert ratio <= 1.0, (
                    f"{name}/{pivot} 第 {j} 步：残差 {err:.3e} 超出 {bound:.3e}"
                    "（是不是又用了最终置换去取前导块？）"
                )
    record(
        label="max 前导块残差 / (8·m·u·ρ·max|A|)",
        theory=1.0,
        measured=worst,
        tol="≤ 1",
        passed=worst <= 1.0,
        detail=f"{n_steps} 个（步, 策略）组合；地板 u/2 = {ROUNDOFF_HINT:.2e}",
    )


@assertion("B", "热力图残差曲线 ≤ 16·n·u·ρ_max（帧级）", chapter="1.2")
def test_heatmap_residual_history_is_roundoff_sized_for_all_pivots() -> None:
    r"""帧里的 `residual_history` 必须落在 :math:`n u \rho` 量级，三种策略都不许出现 O(1)。

    这条是"用户实际看到的那个数组"的回归：前端 agent 拿 ``pivot="complete"`` 跑出 1.065，
    说明"算法层面对、帧里错"是真实存在的失败模式——所以必须钉住帧里的数字。

    判据用**相对**尺度：除以本帧的 :math:`\max\rho`。无主元 + 非占优矩阵会合法地长出
    :math:`\rho \sim 10^3`，绝对阈值会把它误判成 bug（这正是"不要太紧也不要太松"的分寸）。
    """
    u = float(np.finfo(float).eps) / 2.0
    entry = _entry()
    worst = 0.0
    detail: list[str] = []
    for pivot in ("none", "partial", "complete"):
        for n, dominance in ((4, 1.6), (8, 0.1), (12, 0.5)):
            params = {**_defaults("lu_heatmap"), "pivot": pivot, "n": n, "dominance": dominance}
            frame = entry.module.compute("lu_heatmap", params, 0.0)
            hist = np.asarray(frame["residual_history"], dtype=float)
            rho_hist = np.asarray(frame["rho_history"], dtype=float)
            peak = float(np.max(hist))
            rho_max = max(1.0, float(np.max(rho_hist)))
            bound = 16.0 * n * u * rho_max
            worst = max(worst, peak / bound)
            detail.append(f"{pivot}/n={n}/dom={dominance}: {peak:.1e}（ρ_max={rho_max:.3g}）")
            assert peak <= bound, (
                f"{pivot} n={n} dominance={dominance}: 残差峰值 {peak:.3e} 超出 n·u·ρ 界 {bound:.3e}"
                "（是不是又用了最终置换去取前导块？）"
            )
    record(
        label="max residual_history / (16·n·u·ρ_max)",
        theory=1.0,
        measured=worst,
        tol="≤ 1",
        passed=worst <= 1.0,
        detail="；".join(detail),
    )


# --------------------------------------------------------------------------- #
# 3. κ ≳ 1/u：奇异是一种状态，不是崩溃
# --------------------------------------------------------------------------- #
@assertion("B", "κ ≳ 1/u 时 unit_ball 仍可编码且如实报告奇异", chapter="1.2")
def test_unit_ball_reports_singular_state_instead_of_crashing() -> None:
    """`t = tMax`（κ = 1e17）时必须能编码，并给出 `singular` 与状态字符串。

    背景：`shear ∈ {0, 2}` 时 A 在 t ≥ 16.5 处真的奇异，回代抛 ZeroDivisionError，
    时间轴右端 8 个点直接 500。修法不是"把 tMax 调小"——那会藏掉本章最想讲的一件事：
    **κ > 1/u 时问题在这个精度下不可解**。"""
    entry = _entry()
    spec = next(v for v in entry.module.MANIFEST["vizzes"] if v["id"] == "unit_ball")
    t_max = float(spec["time"]["tMax"])
    u = float(np.finfo(float).eps) / 2.0
    singular_cases = 0
    finite_cases = 0
    invariants: list[bool] = []
    for shear in (0.0, 0.8, 2.0):
        for t in (0.0, t_max):
            params = {**_defaults("unit_ball"), "shear": shear}
            frame = build_frame(entry, "unit_ball", params, t, strict=True)
            blob = nlf_mod.pack_frame(frame)  # 编码本身就是一条不变量（meta 不许有 NaN）
            invariants.append(len(blob) > 0)
            if frame.meta.get("singular"):
                singular_cases += 1
                # 奇异时：没有有限值 → 必须是 null，而不是 NaN/0
                for key in ("forward_error", "backward_error", "residual", "x_num_absmax"):
                    assert frame.meta.get(key) is None, f"奇异帧的 meta[{key}] 应当是 null"
                assert "奇异" in frame.strings.get("status", ""), "奇异帧必须给出状态字符串"
                assert frame.meta["overflow"] is True, "解向量冲出 ±3 画面：overflow 必须为真"
            else:
                finite_cases += 1
                assert np.isfinite(frame.meta["forward_error"])
                if float(t) >= 16.0:
                    # 没奇异但 κu ≫ 1：前向误差应当已经 O(1)，且必须报溢出
                    assert frame.meta["forward_error"] > 1e-3
                    assert frame.meta["overflow"] is True
            # κ 的两种口径都要在：解析（σ 比）与实测（单位圆采样）
            assert frame.meta["kappa"] > 0
            assert frame.meta["sigma_min"] >= 0
    assert singular_cases >= 1, "至少要有一种参数组合真的走到奇异，否则这条测试在空转"
    assert finite_cases >= 1, "也要有非奇异的对照，否则这条测试在空转"
    assert all(invariants)
    record(
        label="奇异帧的 meta 契约（null + status + overflow）",
        theory=1.0,
        measured=float(singular_cases),
        tol="≥ 1 个奇异组合且对照存在",
        passed=True,
        detail=f"奇异 {singular_cases} 组 / 非奇异 {finite_cases} 组（共 6 组，t∈{{0, {t_max:g}}}）",
    )


@assertion("A", "单位球的像不越界、解向量越界（几何对照）", chapter="1.2")
def test_overflow_flag_tracks_the_drawn_objects() -> None:
    """`overflow` 必须跟着"画出来的东西"走：单位圆的像 ≤ σmax ≤ 3，而 x̂ 可以到 1e15。

    回归对象：早先 `meta.overflow` 只看单位圆的像（构造上恒为 false），
    而真正冲出 ±3 的 `x_num` / `x_exact` 无人标记——图上"箭头不见了"、账上也没说。"""
    entry = _entry()
    small = entry.module.compute("unit_ball", {**_defaults("unit_ball"), "shear": 0.8}, 0.0)
    large = entry.module.compute("unit_ball", {**_defaults("unit_ball"), "shear": 0.8}, 16.0)
    image_peak = float(np.max(np.abs(np.asarray(small["image_x"]))))
    image_peak = max(image_peak, float(np.max(np.abs(np.asarray(small["image_y"])))))
    x_peak = float(np.max(np.abs(np.asarray(large["x_num"]))))
    details = [
        f"κ=1: 像的峰值 {image_peak:.3f}（≤ 2.24），overflow={small.meta['overflow']}",
        f"κ=1e16: |x̂|∞ = {x_peak:.3e}，overflow={large.meta['overflow']}",
    ]
    assert image_peak <= 3.0 and small.meta["overflow"] is False
    assert x_peak > 3.0 and large.meta["overflow"] is True
    record(
        label="overflow 跟随被画对象（像 vs 解向量）",
        theory=1.0,
        measured=float(large.meta["overflow"]),
        tol="κ=1 假 / κ=1e16 真",
        passed=bool(large.meta["overflow"]) and not bool(small.meta["overflow"]),
        detail="；".join(details),
    )


@assertion("B", "wilkinson_matrix 的经典增长因子 ρ = 2^(n−1)", chapter="1.2")
def test_wilkinson_growth_is_two_to_the_n_minus_one() -> None:
    """``pivot="partial"`` 下 ρ(Wilkinson(n)) = 2^{n−1}；完全主元把它压到 2。

    这个数字是本章"部分主元的最坏情形"那句话的**可执行证据**，也是 G7 对拍 case
    `lu_wilkinson_growth` 里 pin 住的那个值。"""
    rows: list[str] = []
    ok = True
    for n in (6, 12, 16):
        partial = lu_decompose(wilkinson_matrix(n), pivot="partial").rho
        complete = lu_decompose(wilkinson_matrix(n), pivot="complete").rho
        rows.append(f"n={n}: ρ_partial={partial:g}（理论 2^{n - 1}={2 ** (n - 1):g}），ρ_complete={complete:g}")
        ok = ok and partial == float(2 ** (n - 1)) and complete <= 2.0
    assert ok, "；".join(rows)
    record(
        label="ρ(Wilkinson(n)) 部分 / 完全主元",
        theory=float(2 ** 11),
        measured=float(lu_decompose(wilkinson_matrix(12), pivot="partial").rho),
        tol="=",
        passed=True,
        detail="；".join(rows),
    )


if __name__ == "__main__":  # pragma: no cover - 手工排查用
    raise SystemExit(pytest.main([__file__, "-q"]))
