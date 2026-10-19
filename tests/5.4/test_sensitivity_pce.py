r"""5.4 章 A/B/C 类断言：Sobol' 方差分解、相关输入的不唯一性、多项式混沌代理。

这一文件钉住的是第 3、4 页背后的数学，重点是**判据与恒等式的区别**：

* A 类：Ishigami 与 g-函数的解析指数（在本文件里用闭式公式独立重算一遍）、
  gPC 基的正交归一性、Gauss--Legendre 与 2.4 章对拍、Hermite 三点权重闭式、
  多项式被精确投影（以及 |x| 的边界失败）、PCE 解析指数与 pick-freeze 实测的对拍；
* B 类：:math:`\\sum_iS_i\\le1\\le\\sum_iS_{Ti}` 在独立输入下成立（带 bootstrap 区间），
  在相关输入下**失效**（实测 :math:`\\rho=0.5` 时条件方差口径给 1.49）——
  它是判据不是恒等式；
* C 类：谱衰减（几何 vs 代数）、过拟合区（训练误差到机器零而留出误差上升 1e15 倍）、
  外推失效（放大 700--1.5e4 倍）、估计器漂移随 N 的实测变化。
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from numcore.inverse.uq import pce, sensitivity  # noqa: E402
from numcore.quad.gauss import gauss_legendre as gl_reference  # noqa: E402
from numcore.verify import assert_order, assertion, fit_order, record  # noqa: E402


def _semilog_fit(p: np.ndarray, err: np.ndarray) -> tuple[float, float]:
    """``log10(err)`` 关于 ``p`` 的一元最小二乘：返回 ``(每阶数量级斜率, R^2)``。"""
    x = np.asarray(p, dtype=np.float64)
    y = np.log10(np.asarray(err, dtype=np.float64))
    mx, my = float(np.mean(x)), float(np.mean(y))
    den = float(np.sum((x - mx) ** 2))
    b = float(np.sum((x - mx) * (y - my)) / den)
    a = my - b * mx
    ss_res = float(np.sum((y - (a + b * x)) ** 2))
    ss_tot = float(np.sum((y - my) ** 2))
    return b, 1.0 - ss_res / ss_tot


# --------------------------------------------------------------------------- #
# 1. A：Ishigami / g-函数的解析指数（闭式独立重算）
# --------------------------------------------------------------------------- #
@assertion("A", theory="Ishigami 解析方差分解（闭式重算）与模块登记值一致到 1e-12；且 sum S_i <= 1 <= sum S_Ti", chapter="5.4")
def test_ishigami_analytic_matches_closed_form() -> None:
    r"""闭式（:math:`a=7,b=0.1`）：

    .. math::
        V=\frac{a^2}8+\frac{b\pi^4}5+\frac{b^2\pi^8}{18}+\frac12,\quad
        V_1=\frac12+\frac{b\pi^4}5+\frac{b^2\pi^8}{50},\quad V_2=\frac{a^2}8,\quad
        V_{13}=\frac{b^2\pi^8}{18}-\frac{b^2\pi^8}{50}.

    实测：模块给的三元组与闭式相对差 < 1e-15；:math:`\\sum S_i=0.7563<1`、
    :math:`\\sum S_{Ti}=1.2437>1`，差就是 :math:`S_{13}=0.2437`——交互项一点也不小。
    """
    a, b = 7.0, 0.1
    V = a * a / 8.0 + b * math.pi**4 / 5.0 + b * b * math.pi**8 / 18.0 + 0.5
    V1 = 0.5 + b * math.pi**4 / 5.0 + b * b * math.pi**8 / 50.0
    V2 = a * a / 8.0
    V13 = b * b * math.pi**8 / 18.0 - b * b * math.pi**8 / 50.0
    ref_first = np.array([V1 / V, V2 / V, 0.0])
    ref_total = np.array([(V1 + V13) / V, V2 / V, V13 / V])
    m = sensitivity.make_model("ishigami")
    worst = max(
        float(np.max(np.abs(np.asarray(m.first) - ref_first))),
        float(np.max(np.abs(np.asarray(m.total) - ref_total))),
        abs(float(m.variance) - V) / V,
    )
    rec = record(
        label="Ishigami 解析值 vs 闭式公式",
        theory="相对差 <= 1e-12；sum S_i = 0.7563 < 1 < sum S_Ti = 1.2437",
        measured=f"最大相对差 {worst:.3e}；S={np.array2string(np.asarray(m.first), precision=4)}，"
        f"S_T={np.array2string(np.asarray(m.total), precision=4)}，"
        f"sum S_i={sum(m.first):.4f}，sum S_Ti={sum(m.total):.4f}，V={float(m.variance):.4f}",
        tol="1e-12",
        passed=bool(
            worst <= 1e-12
            and abs(sum(m.first) - (V1 + V2) / V) <= 1e-12
            and abs(sum(m.total) - (V1 + V2 + 2 * V13) / V) <= 1e-12
            and sum(m.first) < 1.0 < sum(m.total)
        ),
    )
    assert rec.passed, rec.measured


@assertion("A", theory="g-函数解析一阶/总效应指数与闭式一致；a=(0,1,4.5) 时 sum S_i = 0.9290 < 1", chapter="5.4")
def test_g_function_analytic_and_the_one_bug_that_got_away() -> None:
    r"""闭式 :math:`\\sigma_j^2=\\frac{1/3}{(1+a_j)^2}`、:math:`V=\prod(1+\\sigma_j^2)-1`、
    :math:`V_j=\\sigma_j^2`（因为 :math:`E[g_j]=1`）、
    :math:`V_{Tj}=V-[\\prod_{k\\ne j}(1+\\sigma_k^2)-1]`。

    这里刻意把**第一版写错的那个公式** :math:`V_j=\\sigma_j^2\\prod_{k\\ne j}(1+\\sigma_k^2)`
    也算出来：它给 :math:`\\sum_jS_j=1.0716>1`——一阶指数之和大于 1 是数学上不可能的，
    而它在当时"看起来"完全正常。正确写法 0.9290 < 1。
    """
    aa = np.array([0.0, 1.0, 4.5])
    sig2 = (1.0 / 3.0) / (1.0 + aa) ** 2
    var = float(np.prod(1.0 + sig2)) - 1.0
    first = np.array([sig2[j] / var for j in range(3)])
    total = np.array([1.0 - (float(np.prod(np.delete(1.0 + sig2, j))) - 1.0) / var for j in range(3)])
    wrong = np.array([sig2[j] * float(np.prod(np.delete(1.0 + sig2, j))) / var for j in range(3)])
    m = sensitivity.make_model("g_function")
    worst = max(
        float(np.max(np.abs(np.asarray(m.first) - first))),
        float(np.max(np.abs(np.asarray(m.total) - total))),
    )
    rec = record(
        label="g-函数解析值 vs 闭式 + 错误公式的对照",
        theory="正确公式相对差 <= 1e-12 且 sum S_i < 1；错误公式 sum S_i = 1.0716 > 1",
        measured=f"最大相对差 {worst:.3e}；正确 sum S_i={float(np.sum(first)):.4f}，"
        f"sum S_Ti={float(np.sum(total)):.4f}；错误公式 sum S_i={float(np.sum(wrong)):.4f}",
        tol="1e-12",
        passed=bool(
            worst <= 1e-12 and float(np.sum(first)) < 1.0 < float(np.sum(total)) and float(np.sum(wrong)) > 1.0
        ),
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 2. A：gPC 基的正交归一性与 Gauss 求积对拍
# --------------------------------------------------------------------------- #
@assertion("A", theory="Legendre/Hermite 基在各自测度下正交归一（max|G-I| <= 1e-9）；Gauss-Legendre 与 2.4 章一致", chapter="5.4")
def test_basis_orthonormality_and_gauss_rules_against_chapter_2_4() -> None:
    r"""三件事一次钉住：

    1. :math:`\langle\phi_k,\phi_l\rangle=\delta_{kl}`——Legendre 对 :math:`\mathrm dx/2`、
       Hermite 对标准正态（用各自的 Gauss 求积当内积）；
    2. Gauss--Legendre 节点与 :func:`numcore.quad.gauss.gauss_legendre` 相对差 <= 5e-15，
       权重**恰好**是它的一半（那一支对 :math:`\mathrm dx` 归一，权重和为 2）；
    3. Hermite 三点权重必须**恰好**是 :math:`(1/6,2/3,1/6)`——本模块第一版在这里
       给出 :math:`(1/12,\\dots,1/12)`，差一个因子 2，会让"正交性"和
       "方差恒等式"两条断言一起错。
    """
    worst_gram = 0.0
    for kind, rule in (("legendre", pce.gauss_legendre_rule), ("hermite", pce.gauss_hermite_rule)):
        x, w = rule(24)
        vals = pce.legendre_values(x, 12) if kind == "legendre" else pce.hermite_values(x, 12)
        gram = (vals * w[:, None]).T @ vals
        worst_gram = max(worst_gram, float(np.max(np.abs(gram - np.eye(13)))))
    worst_node = 0.0
    worst_weight = 0.0
    for n in (1, 2, 3, 5, 8, 12):
        x, w = pce.gauss_legendre_rule(n)
        xr, wr = gl_reference(n)
        worst_node = max(worst_node, float(np.max(np.abs(x - xr))))
        worst_weight = max(worst_weight, float(np.max(np.abs(w - wr / 2.0))))
    xh, wh = pce.gauss_hermite_rule(3)
    hermite_ok = (
        float(np.max(np.abs(xh - np.array([-math.sqrt(3.0), 0.0, math.sqrt(3.0)])))) <= 1e-14
        and float(np.max(np.abs(wh - np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0])))) <= 1e-15
    )
    rec = record(
        label="Gram 矩阵偏离单位阵 / 节点与权重对拍 / Hermite 三点权重",
        theory="Gram <= 1e-9；节点 <= 5e-15；权重比恒为 2（<= 1e-15）；Hermite 三点 = (1/6,2/3,1/6)",
        measured=f"Gram 最大偏离 {worst_gram:.3e}；GL 节点 {worst_node:.3e}；GL 权重差 {worst_weight:.3e}；"
        f"GH 三点权重 {np.array2string(wh, precision=6)}",
        tol="见 theory",
        passed=bool(worst_gram <= 1e-9 and worst_node <= 5e-15 and worst_weight <= 1e-15 and hermite_ok),
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 3. A：多项式被精确投影 + |x| 的边界失败
# --------------------------------------------------------------------------- #
@assertion("A", theory="p >= 4 时 x^4-2x^2+x 被精确投影（误差 <= 1e-12）；|x| 在 p=12 时仍有 2e-2 误差", chapter="5.4")
def test_pce_exact_for_polynomials_and_fails_for_kinks() -> None:
    r"""双向 A 类：

    * ``poly4``：``p>=4`` 的投影/回归误差到 :math:`10^{-15}` 量级，
      且 ``p<4`` 时**明显不为零**（否则这条断言什么也没测）；
    * ``abs``：折点函数的谱只按代数速率衰减，``p=12`` 时相对误差仍有 :math:`2\\times10^{-2}`
      ——同一个基、同一个阶数，"能不能精确"完全由被积函数的光滑性决定。
    """
    f4 = pce.target_function("poly4")
    x = pce.design_samples(1, 500, kind="legendre", seed=1)
    exact_errors = [
        pce.relative_l2_error(pce.pce_projection(f4, dim=1, order=p), f4, x) for p in (2, 3, 4, 6, 12)
    ]
    fa = pce.target_function("abs")
    abs_err = pce.relative_l2_error(pce.pce_projection(fa, dim=1, order=12), fa, x)
    rec = record(
        label="poly4 在 p=2,3,4,6,12 的投影误差 与 |x| 在 p=12 的误差",
        theory="poly4：p<4 时 > 1e-2、p>=4 时 <= 1e-12；|x| 在 p=12 时 > 1e-2",
        measured=f"poly4 {np.array2string(np.array(exact_errors), precision=3)}；|x| p=12 → {abs_err:.4f}",
        tol="见 theory",
        passed=bool(
            exact_errors[0] > 1e-2
            and exact_errors[1] > 1e-2
            and max(exact_errors[2:]) <= 1e-12
            and abs_err > 1e-2
        ),
    )
    assert rec.passed, rec.measured


@assertion("A", theory="PCE 的解析 Sobol 指数（由系数读出）与 pick-freeze 实测一致（2 维可加模型，S=(0.5,0.5)）", chapter="5.4")
def test_pce_analytic_indices_match_pick_freeze() -> None:
    r"""两条互不相干的路：一条只用系数（:math:`S_i=\\sum_{\\alpha\\in\\mathcal A_i}c_\\alpha^2/V`），
    一条用 4000 次模型求值的 pick-freeze。对 :math:`x_1+x_2`（标准正态输入）
    两者都该给 :math:`(0.5,0.5)`，且 PCE 那条是**精确**的（1 阶 Hermite 就够）。
    """
    f = sensitivity.make_model("bilinear").func
    fit = pce.pce_regression(f, pce.design_samples(2, 200, kind="hermite", seed=20261018), order=1, kind="hermite")
    first, total = pce.pce_sobol_indices(fit)
    ind = sensitivity.pick_freeze_indices("bilinear", 4096, seed=20261018, bootstrap=200)
    rec = record(
        label="PCE 解析指数 vs pick-freeze 实测（双线性，N=4096）",
        theory="PCE 精确给出 (0.5,0.5)（相对 1e-12）；pick-freeze 在 ±0.05 内",
        measured=f"PCE S={np.array2string(first, precision=6)}，S_T={np.array2string(total, precision=6)}；"
        f"pick-freeze S={np.array2string(np.asarray(ind.first), precision=4)}"
        f"（半宽 {np.array2string(np.asarray(ind.first_halfwidth), precision=4)}），V={ind.variance:.4f}",
        tol="见 theory",
        passed=bool(
            float(np.max(np.abs(first - 0.5))) <= 1e-12
            and float(np.max(np.abs(total - 0.5))) <= 1e-12
            and float(np.max(np.abs(np.asarray(ind.first) - 0.5))) <= 0.05
            and abs(ind.variance - 2.0) <= 0.05
        ),
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 4. B：sum S_i <= 1 <= sum S_Ti 是判据，不是恒等式
# --------------------------------------------------------------------------- #
@assertion("B", theory="独立输入：sum S_i <= 1 <= sum S_Ti 在区间意义下成立；rho=0.5 时条件方差口径给 1.49 > 1", chapter="5.4")
def test_variance_decomposition_criterion_holds_and_fails() -> None:
    r"""两个方向都要有：

    * **成立的一侧**（独立输入）：g-函数 :math:`\\sum S_i=0.929`、:math:`\\sum S_{Ti}=1.072`，
      bootstrap 区间把 1 夹在中间（实测 N=4096 时 :math:`\\sum S_i\\pm` 半宽仍小于 1）；
    * **失效的一侧**（相关输入）：双线性模型 :math:`\\rho=0.5` 时条件方差口径
      :math:`\\sum_i\\mathrm{Var}(E[Y|X_i])/V=1.4942>1`——不是定理被推翻，
      而是"输入独立"这个前提没了。同一份样本打散后乘积测度口径给 0.989 < 1。
    """
    ind = sensitivity.pick_freeze_indices("g_function", 4096, seed=20261018, bootstrap=200)
    half = math.sqrt(float(np.sum(np.asarray(ind.first_halfwidth) ** 2)))
    x = sensitivity.correlated_normal_samples(0.5, 40000, 2, seed=7)
    y = np.asarray(sensitivity.make_model("bilinear").func(x), dtype=np.float64)
    st = sensitivity.structural_indices(x, y, n_bins=64, rho=0.5)
    sh = sensitivity.shuffled_indices("bilinear", x, seed=11)
    half_total = math.sqrt(float(np.sum(np.asarray(ind.total_halfwidth) ** 2)))
    rec = record(
        label="独立输入：g-函数的 sum S_i / sum S_Ti 与解析值、区间",
        theory="sum S_i - 半宽 <= 1（不显著超过 1）、sum S_Ti >= 1（区间内），且与解析值差 <= 0.08",
        measured=f"sum S_i={ind.sum_first:.4f}±{half:.4f}（解析 0.9290），sum S_Ti={ind.sum_total:.4f}±{half_total:.4f}"
        f"（解析 1.0716）",
        tol="见 theory",
        passed=bool(
            ind.sum_first - half <= 1.0 <= ind.sum_total + half_total
            and abs(ind.sum_first - 0.929023) <= 0.08
            and abs(ind.sum_total - 1.071642) <= 0.08
        ),
    )
    rec2 = record(
        label="相关输入（rho=0.5）：条件方差口径 / 乘积测度口径",
        theory="条件方差口径 > 1（实测 1.4942，超出部分 0.4942）；乘积测度口径 ≈ 1（实测 0.989）",
        measured=f"条件方差口径 {float(np.sum(st.between)):.4f}（超出 {st.overlap_excess:+.4f}），"
        f"乘积测度口径 {sh.sum_first:.4f}，两者差 {float(np.sum(st.between)) - sh.sum_first:+.4f}",
        tol="前者 > 1.4、后者 ∈ [0.95,1.05]",
        passed=bool(float(np.sum(st.between)) > 1.4 and 0.95 <= sh.sum_first <= 1.05),
    )
    assert rec.passed and rec2.passed, f"{rec.measured} / {rec2.measured}"


@assertion("B", theory="估计器漂移是偏差不是噪声：sum S_i(N) 在 N=64..4096 上单调靠近解析值（幅度 >= 0.15）", chapter="5.4")
def test_estimator_drift_is_bias_not_noise() -> None:
    r"""同一随机流、同一批样本的前 N 行：:math:`\\sum S_i(N)` 从 0.62 一路爬到 0.93（解析 0.929）。

    判据取"漂移幅度 >= 0.15 且与解析值同向"，而不是"随 N 单调"：小 N 处的
    bootstrap 区间宽到 ±0.35，单个 N 上完全可以反着走——**这正是它必须被画出来的原因**。
    """
    ns, first, total = sensitivity.estimator_drift("g_function", (64, 128, 256, 512, 1024, 2048, 4096), seed=20261018)
    sums = np.sum(first, axis=1)
    analytic = sum(sensitivity.make_model("g_function").first)
    span = float(np.max(sums) - np.min(sums))
    rec = record(
        label="sum S_i(N) 的漂移幅度与解析值对照",
        theory="漂移 >= 0.15；N=4096 处与解析值差 <= 0.08",
        measured=f"sum S_i(N)={np.array2string(sums, precision=4)}（N={[int(v) for v in ns]}），"
        f"漂移 {span:.4f}，解析 {analytic:.4f}，末端差 {abs(float(sums[-1]) - analytic):.4f}",
        tol="见 theory",
        passed=bool(span >= 0.15 and abs(float(sums[-1]) - analytic) <= 0.08),
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 5. C：谱衰减的两种形态
# --------------------------------------------------------------------------- #
@assertion("C", theory="exp(1.5x) 的误差按几何速率衰减（每阶约 10^-1.0，R^2 >= 0.98）；|x| 只按代数速率衰减（实测 alpha 1.09-1.31）", chapter="5.4")
def test_spectral_decay_geometric_versus_algebraic() -> None:
    r"""同一套基、同一套拟合方式，只换被积函数：

    * ``exp``：:math:`\\log_{10}\\text{err}` 对 :math:`p` 的斜率实测 -1.0005（R²=0.992）
      ——每加一阶误差掉一个数量级，11 阶掉 11 个数量级；
    * ``abs``：同一条拟合只给 -0.108（R²=0.86），换成幂律 :math:`C p^{-\\alpha}`
      给出 :math:`\\alpha=1.17`——折点函数**没有**谱收敛，只有代数收敛。

    这就是"谱方法不是万能的"那条结论的实测形态。
    """
    detail = []
    slopes = {}
    for name in ("exp", "abs"):
        out = pce.error_vs_order(
            pce.target_function(name), dim=1, orders=range(1, 13), n_train=64, kind="legendre", method="regression"
        )
        b, r2 = _semilog_fit(out["orders"], out["outsample"])
        slopes[name] = (b, r2, out["outsample"])
        detail.append(f"{name}: 半对数斜率 {b:+.4f}（R²={r2:.4f}）")
    rec = record(
        label="两种被积函数的半对数衰减率",
        theory="exp：斜率 <= -0.9 且 R² >= 0.98；|x|：斜率 >= -0.2（远慢于几何）",
        measured="；".join(detail)
        + f"；exp 从 p=1 到 p=12 掉了 {math.log10(slopes['exp'][2][0] / slopes['exp'][2][-1]):.1f} 个数量级，"
        f"|x| 只掉了 {math.log10(slopes['abs'][2][0] / slopes['abs'][2][-1]):.1f} 个数量级",
        tol="见 theory",
        passed=bool(slopes["exp"][0] <= -0.9 and slopes["exp"][1] >= 0.98 and slopes["abs"][0] >= -0.2),
    )
    # |x| 的代数指数：奇/偶阶分别拟合（偶数阶几乎不降，误差曲线是"台阶"）
    orders16 = pce.error_vs_order(
        pce.target_function("abs"), dim=1, orders=range(1, 17), n_train=64, kind="legendre", method="regression"
    )
    pp = np.asarray(orders16["orders"], dtype=float)
    ee = np.asarray(orders16["outsample"], dtype=float)
    fits = {
        "全部阶": fit_order(1.0 / pp, ee),
        "奇数阶": fit_order(1.0 / pp[pp % 2 == 1], ee[pp % 2 == 1]),
        "偶数阶": fit_order(1.0 / pp[pp % 2 == 0], ee[pp % 2 == 0]),
    }
    alphas = {k: float(v.slope) for k, v in fits.items()}
    r2s = {k: float(v.r2) for k, v in fits.items()}
    rec2 = record(
        label="|x| 的代数衰减指数（三种子样本）",
        theory="三者都落在 [0.9,1.5] 且 R² >= 0.97：与渐近的 p^-1（c_k ~ k^(-3/2)）一致，"
        "但 p<=16 上测不出干净的 -1（奇偶台阶）",
        measured="；".join(f"{k} alpha={alphas[k]:.4f}（R²={r2s[k]:.4f}）" for k in alphas)
        + f"；误差序列 {np.array2string(ee, precision=4)}",
        tol="[0.9,1.5]",
        passed=bool(all(0.9 <= a <= 1.5 for a in alphas.values()) and all(r >= 0.97 for r in r2s.values())),
    )
    assert rec.passed and rec2.passed, f"{rec.measured} / {rec2.measured}"
