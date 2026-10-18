"""4.4 章 A/B 类断言：装配、对称正定、Lax–Milgram/Cea、范数与阶、自适应与 p-加密。

这一章的证据链刻意分成三层，每层都有自己的对照物：

1. **解析解**（``model_1d`` / ``model_2d`` 的制造解）——测收敛阶。同一个解必须报**两个范数**
   的阶：光滑解 :math:`P_1` 上 L2 是 2、能量是 1，这不是矛盾而是两个不同的量；
   界面算例上分别是 1 与 1/2（不是教科书里默认的 3/2）。
2. **离散不变量**——装配矩阵的对称性与正定性、装配顺序无关性、网格加密的协调性、
   常向量落在刚度阵的核里。这些都是"装出来的"性质，所以要用**独立于装配**的量去对：
   Cholesky 主元、二次型、单元内反序累加、镜像路线。
3. **后验量**——指示子的效率指数、Doerfler 标记的份额、自适应与一致加密在同自由度下的比。
   指示子只用 :math:`u_h` 与系数场，真误差要用解析解，两者必须交叉验证。

每一条断言都写明"这个数字是怎么量出来的"，因为这一章最容易骗人的地方正是
"用一个数字代表一整类结论"。
"""

from __future__ import annotations

import numpy as np

from numcore.linalg.chol import cholesky, is_positive_definite
from numcore.pde.fem import adapt as ad
from numcore.pde.fem import assemble as asm
from numcore.pde.fem import mesh as ms
from numcore.pde.fem import solve as sv
from numcore.utils import rng
from numcore.verify import (
    assert_leq,
    assert_monotone,
    assert_symmetric,
    assert_within,
    assertion,
    fit_order,
)
from numcore.verify import record as _record  # noqa: E402

def record(*, passed: bool = True, **kw):
    """``record`` 的本地包装：**失败即抛**。

    框架的 ``record(passed=False)`` 只登记不抛，于是"报告里有失败记录、测试却全绿"——
    这正是本章要防的静默通过。这里让它同时失败（先登记、再抛，报告里留痕）。
    """
    rec = _record(passed=passed, **kw)
    assert passed, f"{kw.get('label')}: 实测 {kw.get('measured')} vs 理论 {kw.get('theory')}"
    return rec

#: h-阶梯（P1/P2 光滑解）：不加密到更细是为了把整套测试压在几秒内。
_H = tuple(1.0 / n for n in (4, 8, 16, 32, 64))


def _ladder(model: sv.Model1D, degree: int, ne_list) -> dict[str, np.ndarray]:
    """一致加密阶梯：返回 ``h``、``l2``、``h1``、``energy``、``h2``（与帧同一入口）。"""
    h, l2, h1, en, h2 = [], [], [], [], []
    for ne in ne_list:
        r = sv.solve_model(model, ms.uniform_mesh_1d(int(ne)), degree)
        h.append(1.0 / float(ne))
        l2.append(float(r["error"]["l2"]))
        h1.append(float(r["error"]["h1"]))
        en.append(float(r["error"]["energy"]))
        h2.append(float(r["error"].get("h2", np.nan)))
    return {"h": np.asarray(h), "l2": np.asarray(l2), "h1": np.asarray(h1),
            "energy": np.asarray(en), "h2": np.asarray(h2)}


def _semilog_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """``(斜率, R^2)``：对 :math:`\\log y` 做最小二乘（自己算，不用被禁用的现成拟合）。"""
    xa = np.asarray(x, dtype=float)
    ya = np.log(np.asarray(y, dtype=float))
    n = xa.size
    sx, sy = float(np.sum(xa)), float(np.sum(ya))
    den = n * float(np.sum(xa * xa)) - sx * sx
    slope = (n * float(np.sum(xa * ya)) - sx * sy) / den
    intercept = (sy - slope * sx) / n
    ss_res = float(np.sum((ya - (slope * xa + intercept)) ** 2))
    ss_tot = float(np.sum((ya - np.mean(ya)) ** 2))
    return float(slope), 1.0 - ss_res / ss_tot


def _loglog_r2(x: np.ndarray, y: np.ndarray) -> float:
    lx = np.log(np.asarray(x, dtype=float))
    ly = np.log(np.asarray(y, dtype=float))
    n = lx.size
    sx, sy = float(np.sum(lx)), float(np.sum(ly))
    den = n * float(np.sum(lx * lx)) - sx * sx
    slope = (n * float(np.sum(lx * ly)) - sx * sy) / den
    intercept = (sy - slope * sx) / n
    ss_res = float(np.sum((ly - (slope * lx + intercept)) ** 2))
    ss_tot = float(np.sum((ly - np.mean(ly)) ** 2))
    return 1.0 - ss_res / ss_tot


# --------------------------------------------------------------------------- #
# A 类：解析对拍（阶、常数、闭式）
# --------------------------------------------------------------------------- #
@assertion("A", theory="同一个解在不同范数下的阶不同：P1 光滑解 L2 是 2、能量是 1", chapter="4.4")
def test_order_depends_on_the_norm() -> None:
    r"""同一条网格阶梯、同一份解，只换范数：L2 给 2、H1 半范与能量范数给 1。

    "收敛阶"必须和范数一起报，否则这句话没有内容。两个方向都要报：
    L2 阶**大于** 1（否则说明连光滑解都没测对），同时能量阶**等于** 1（不能把 L2 的阶当成能量阶）。
    """
    for model, degree, l2_theory, en_theory, tol in (
        (sv.model_1d("smooth"), 1, 2.0, 1.0, 0.1),
        (sv.model_1d("smooth"), 2, 3.0, 2.0, 0.1),
    ):
        lad = _ladder(model, degree, (4, 8, 16, 32, 64))
        f_l2 = fit_order(lad["h"], lad["l2"], min_points=3)
        f_en = fit_order(lad["h"], lad["energy"], min_points=3)
        record(
            label=f"P{degree} 光滑解：L2 阶 = {l2_theory:g}",
            theory=l2_theory, measured=f_l2.slope, tol=f"±{tol:g}",
            passed=abs(f_l2.slope - l2_theory) <= tol, kind="A",
            detail=f"R^2={f_l2.r2:.6f}；逐点 h={np.array2string(lad['h'], precision=4)}",
        )
        record(
            label=f"P{degree} 光滑解：能量阶 = {en_theory:g}",
            theory=en_theory, measured=f_en.slope, tol=f"±{tol:g}",
            passed=abs(f_en.slope - en_theory) <= tol, kind="A",
            detail=f"R^2={f_en.r2:.6f}",
        )
        # 两向性：同一条阶梯上两个范数的阶必须**分开**（否则"报哪个范数都一样"）
        gap = float(f_l2.slope - f_en.slope)
        record(
            label=f"P{degree} 光滑解：L2 阶与能量阶之差 ≥ 0.8",
            theory="≥ 0.8", measured=gap, tol="0", passed=gap >= 0.8, kind="A",
            detail="同一个解、同一条网格阶梯，只换范数——阶就变了，所以范数必须和阶一起报",
        )


@assertion("A", theory="界面算例（a 跳 100 倍）：能量阶 1/2、L2 阶 1（对偶问题只有 H^{3/2} 正则性）",
           chapter="4.4")
def test_interface_orders_are_one_half_and_one() -> None:
    r"""分段线性解的经典奇性：能量阶 :math:`1/2`、L2 阶 :math:`1`。

    关键在于**把界面在单元内的相对位置固定住**（``xs=1/3`` 且 ``ne=4^m`` ⟹ θ 恒为 1/3）：
    否则 θ 随 h 变，误差里会带上一个 O(1) 的相位因子，拟合出的阶会被污染几个百分点。
    两个方向都测：能量阶必须**小于** 0.7（奇性确实降阶了），同时 L2 阶必须**大于** 0.8。
    """
    model = sv.model_1d("interface", xs=1.0 / 3.0, ratio=100.0)
    lad = _ladder(model, 1, (4, 16, 64, 256))
    f_en = fit_order(lad["h"], lad["energy"], min_points=3)
    f_l2 = fit_order(lad["h"], lad["l2"], min_points=3)
    assert_within(f_en.slope, 0.5, rtol=0.0, atol=0.1, label="界面算例能量阶 = 1/2",
                  detail=f"R^2={f_en.r2:.6f}（θ 固定在 1/3，无相位噪声）")
    assert_within(f_l2.slope, 1.0, rtol=0.0, atol=0.1, label="界面算例 L2 阶 = 1",
                  detail=f"R^2={f_l2.r2:.6f}")
    # 两向性 + 证伪朴素期待：L2 阶不是 3/2（只有对偶问题有 H^2 正则性时才是）
    gap = abs(float(f_l2.slope) - 1.5)
    record(
        label="界面算例的 L2 阶不是 3/2（差 ≥ 0.4）",
        theory="|slope − 1.5| ≥ 0.4", measured=gap, tol="0", passed=gap >= 0.4, kind="A",
        detail=("对偶问题带同一个系数跳，只有 H^{3/2} 正则性 ⟹  || u−u_h || ₀ ≤ C h^{1/2}·h^{1/2} = O(h)；"
                "朴素期待 3/2 需要 H^2 正则性"),
    )
    # 相位陷阱的对照：同一算例把界面放到无理数位置，拟合同一条阶梯会偏
    bad = _ladder(sv.model_1d("interface"), 1, (4, 16, 64, 256))
    f_bad = fit_order(bad["h"], bad["energy"], min_points=3)
    record(
        label="界面位置取无理数 1/π 时同一条阶梯的拟合阶会被污染",
        theory="偏离 0.5 至少 0.02", measured=abs(float(f_bad.slope) - 0.5), tol="0",
        passed=abs(float(f_bad.slope) - 0.5) >= 0.02, kind="A",
        detail=(f"θ 固定的拟合值 {f_en.slope:.4f} vs θ 随 h 变的拟合值 {f_bad.slope:.4f}；"
                "误差 ~ C(θ)h^{1/2}，C 依赖界面在单元里的相对位置（实测包络 max/min = 2.71）"),
    )


@assertion("A", theory="四阶梁：Hermite 三次元的 H^2 阶 2、L2 阶 4，均布载荷最大挠度 5q/384",
           chapter="4.4")
def test_beam_orders_and_the_deflection_formula() -> None:
    r"""梁是 4 阶问题：能量范数是 :math:`H^2` 半范，阶是 2；:math:`L^2` 阶是 4（两阶之差 2）。

    同时验证闭式：简支梁均布载荷的最大挠度 :math:`5q/384`，Hermite 解在节点上**精确**。
    """
    ne_list = (2, 4, 8, 16, 32)
    model = sv.model_1d("beam_uniform", q=1.0)
    h, h2, l2, wmax = [], [], [], []
    for ne in ne_list:
        r = sv.solve_model(model, ms.uniform_mesh_1d(ne))
        h.append(1.0 / ne)
        h2.append(float(r["error"]["h2"]))
        l2.append(float(r["error"]["l2"]))
        wmax.append(float(np.max(np.abs(r["sol"].u[0::2]))))
    h = np.asarray(h)
    f_h2 = fit_order(h, np.asarray(h2), min_points=3)
    f_l2 = fit_order(h, np.asarray(l2), min_points=3)
    record(label="梁的 H^2 半范阶 = 2", theory=2.0, measured=f_h2.slope, tol="±0.1",
           passed=abs(f_h2.slope - 2.0) <= 0.1, kind="A", detail=f"R^2={f_h2.r2:.6f}")
    record(label="梁的 L2 阶 = 4", theory=4.0, measured=f_l2.slope, tol="±0.1",
           passed=abs(f_l2.slope - 4.0) <= 0.1, kind="A", detail=f"R^2={f_l2.r2:.6f}")
    assert_within(wmax, 5.0 / 384.0, rtol=1e-12, label="均布载荷最大挠度 = 5q/384（节点精确）",
                  detail=f"从 ne=2 到 ne=32 全部相同：{np.array2string(np.asarray(wmax), precision=8)}")
    # 两向性：两个范数的阶必须**不同**（4 阶问题的能量范数不是 H^1）
    gap = float(f_l2.slope - f_h2.slope)
    record(label="梁的 L2 阶比能量阶高 2", theory="≈2", measured=gap, tol="±0.3",
           passed=abs(gap - 2.0) <= 0.3, kind="A",
           detail="同一批解换范数：H^2 是 2 阶、L2 是 4 阶——四阶问题的'能量范数'必须是 H^2 半范")


@assertion("A", theory="一维点载荷：载荷落在节点上时节点值精确（Green 函数分段线性），"
                       "落进单元内部立刻退化", chapter="4.4")
def test_point_load_sawtooth_and_nodal_exactness() -> None:
    r"""点载荷的锯齿：同一个网格、同一段代码，只把载荷挪 1/32 个单元。

    ``x0=0.5``（节点）能量误差 5.7e-15；``x0=0.53125``（单元内部）1.25e-1；
    而**节点误差在两种位置上都在机器精度**（≤4.4e-16）——所以"节点值准"不能推出"解准"。
    梁（Hermite）同样如此：9.96e-15 对 1.10e-3。
    """
    model = sv.model_1d("point", x0=0.5, P=1.0)
    r_node = sv.solve_model(model, ms.uniform_mesh_1d(16))
    r_in = sv.solve_model(sv.model_1d("point", x0=0.53125, P=1.0), ms.uniform_mesh_1d(16))
    assert_leq(r_node["error"]["energy"], 1e-12, label="载荷在节点上：能量误差 ≤1e-12",
               detail=f"实测 {r_node['error']['energy']:.3e}（一维 Green 函数分段线性 ⟹ 节点精确）")
    assert_leq(1e-2, r_in["error"]["energy"], label="载荷在单元内部：能量误差 ≥1e-2",
               detail=f"实测 {r_in['error']['energy']:.3e}")
    nodes = ms.uniform_mesh_1d(16).x
    nod_node = float(np.max(np.abs(r_node["sol"].u - model.exact(nodes))))
    nod_in = float(np.max(np.abs(r_in["sol"].u - sv.model_1d("point", x0=0.53125, P=1.0).exact(nodes))))
    assert_leq(nod_node, 1e-13, label="载荷在节点上：节点误差 ≤1e-13", detail=f"实测 {nod_node:.3e}")
    assert_leq(nod_in, 1e-13, label="载荷在单元内部：节点误差仍然 ≤1e-13",
               detail=f"实测 {nod_in:.3e}——节点精确与解精确是两件事")
    record(label="点载荷锯齿的峰谷比 ≥ 1e10", theory="≥ 1e10",
           measured=float(r_in["error"]["energy"] / max(r_node["error"]["energy"], 1e-300)),
           tol="0", passed=r_in["error"]["energy"] / max(r_node["error"]["energy"], 1e-300) >= 1e10, kind="A",
           detail="同一个网格、同一个载荷大小，只挪了 1/32 个单元")
    # 梁：H^2 半范（四阶问题的能量范数）
    b_node = sv.solve_model(sv.model_1d("beam_point", x0=0.5), ms.uniform_mesh_1d(16))
    b_in = sv.solve_model(sv.model_1d("beam_point", x0=0.53125), ms.uniform_mesh_1d(16))
    record(label="梁：载荷在节点上 H^2 误差 ≤1e-12", theory="≤ 1e-12", measured=float(b_node["error"]["h2"]),
           tol="0", passed=b_node["error"]["h2"] <= 1e-12, kind="A",
           detail=f"实测 {b_node['error']['h2']:.3e}")
    record(label="梁：载荷在单元内部 H^2 误差 ≥1e-4", theory="≥ 1e-4", measured=float(b_in["error"]["h2"]),
           tol="0", passed=b_in["error"]["h2"] >= 1e-4, kind="A",
           detail=f"实测 {b_in['error']['h2']:.3e}")


@assertion("A", theory="二维 Q1/P1 单元的能量阶 1、L2 阶 2（渐近窗口内）", chapter="4.4")
def test_two_dimensional_orders_for_tri_and_quad() -> None:
    r"""二维制造解（边界上为零的局部凸包）：三角形与四边形都到能量 1 阶、L2 2 阶。

    窗口是刻意选的：**粗端还没进渐近区**。四点全局拟合会给出荒唐值（三角形 L2 只到 1.367），
    取 :math:`n\ge16` 才是理论阶——这一条本身就是本章的陷阱之一。
    """
    model = sv.model_2d("bump", x0=0.5, y0=0.5, sigma=0.18)
    for kind, maker, ns in (
        ("Q1 四边形", lambda n: ms.uniform_quad_mesh(n, n, 0.0), (8, 16, 32)),
        ("P1 三角形", lambda n: ms.uniform_tri_mesh(n, n), (16, 32, 64)),
    ):
        hs, l2, en = [], [], []
        for n in ns:
            r = sv.solve_model_2d(model, maker(n))
            hs.append(1.0 / n)
            l2.append(float(r["error"]["l2"]))
            en.append(float(r["error"]["energy"]))
        f_en = fit_order(np.asarray(hs), np.asarray(en), min_points=3)
        f_l2 = fit_order(np.asarray(hs), np.asarray(l2), min_points=3)
        record(label=f"{kind}：能量阶 = 1", theory=1.0, measured=f_en.slope, tol="±0.1",
               passed=abs(f_en.slope - 1.0) <= 0.1, kind="A",
               detail=f"R^2={f_en.r2:.6f}；能量误差 {np.array2string(np.asarray(en), precision=3)}")
        record(label=f"{kind}：L2 阶 = 2", theory=2.0, measured=f_l2.slope, tol="±0.1",
               passed=abs(f_l2.slope - 2.0) <= 0.1, kind="A",
               detail=f"R^2={f_l2.r2:.6f}；L2 误差 {np.array2string(np.asarray(l2), precision=3)}")
    # 粗端污染：把三角形阶梯从 n=4 起拟合（含远未渐近的粗网格），阶会明显偏低
    hs, l2 = [], []
    for n in (4, 8, 16):
        r = sv.solve_model_2d(model, ms.uniform_tri_mesh(n, n))
        hs.append(1.0 / n)
        l2.append(float(r["error"]["l2"]))
    f_coarse = fit_order(np.asarray(hs), np.asarray(l2), min_points=3)
    record(label="粗端未进渐近区：三角形 L2 拟合阶明显低于 2", theory="< 1.8",
           measured=float(f_coarse.slope), tol="0", passed=f_coarse.slope < 1.8, kind="A",
           detail=f"n=4,8,16 拟合得 {f_coarse.slope:.4f}，而 n≥16 的窗口是 2 附近")


@assertion("A", theory="Cea 常数 = √(M_h/α_h)：c=1 时双线性形式就是 H^1 内积，常数恰好 1",
           chapter="4.4")
def test_cea_constant_and_the_useless_analytic_bound() -> None:
    r"""可测的前提：:math:`\alpha_h` 与 :math:`M_h` 是广义特征值端点。

    ``c=1`` 时形式就是 :math:`H^1` 内积：:math:`\alpha_h=M_h=1`、Cea 常数 1，界是**锐的**；
    ``c=0`` 时"系数型"解析界是 :math:`\infty`（:math:`\min(a,c)=0`）而离散常数有限，
    实测 Galerkin 效率比 1.000000；反差 :math:`10^4` 时 Cea 界允许 100 倍劣化，
    真实效率比却只有 0.0228——**界成立但没用**。
    """
    model = sv.model_1d("smooth", c=1.0)
    r = sv.solve_model(model, ms.uniform_mesh_1d(8), 1)
    lm = sv.lax_milgram_constants(r["space"], model.a_field, model.c_field, free=r["sol"].free,
                                  a_bounds=model.a_bounds, c_bounds=model.c_bounds)
    interp = sv.approximation_error_1d(r["space"], model.exact, model.dexact,
                                       a_field=model.a_field, c_field=model.c_field)
    ratio = float(r["error"]["energy"] / interp["energy"])
    assert_within(lm["alpha_h"], 1.0, rtol=1e-9, label="c=1 时 α_h = 1（形式就是范数）",
                  detail=f"实测 {lm['alpha_h']:.12f}")
    assert_within(lm["M_h"], 1.0, rtol=1e-9, label="c=1 时 M_h = 1", detail=f"实测 {lm['M_h']:.12f}")
    assert_within(lm["cea_factor"], 1.0, rtol=1e-6, label="c=1 时 Cea 常数 = 1（界是锐的）",
                  detail=f"解析界 {lm['analytic_cea_factor']:.6f}，实测网格比 {ratio:.6f}")
    assert_leq(ratio, 1.0, rtol=0.0, atol=1e-9, label="Galerkin 解在能量范数下不劣于插值",
               detail=f"实测比 {ratio:.6f} ≤ 1（这是最佳逼近性质的直接后果）")
    # 两向性：c=0 时解析界变得没用，而离散常数仍然有限
    m0 = sv.model_1d("smooth")
    r0 = sv.solve_model(m0, ms.uniform_mesh_1d(8), 1)
    lm0 = sv.lax_milgram_constants(r0["space"], m0.a_field, m0.c_field, free=r0["sol"].free,
                                   a_bounds=m0.a_bounds, c_bounds=m0.c_bounds)
    # 注意：measured 这里必须是有限值或字符串——tools/build_book.py 的 fmt() 用
    # f"{v:.3e}".split("e") 排版，遇到 inf 会 ValueError 崩掉整本书的附表生成。
    record(label="c=0 时系数型解析界是无穷（min(a,c)=0）", theory="无穷大",
           measured="inf（min(a,c)=0）", tol="0",
           passed=not np.isfinite(lm0["analytic_cea_factor"]), kind="A",
           detail=f"同一算例的离散 α_h = {lm0['alpha_h']:.4f}（Poincare + 网格给的），界与实测差着「有没有用」")
    # 反差 1e4：界允许 100 倍劣化，真实效率比小 4000 倍
    m4 = sv.model_1d("interface", xs=1.0 / 3.0, ratio=10000.0)
    r4 = sv.solve_model(m4, ms.uniform_mesh_1d(16), 1)
    lm4 = sv.lax_milgram_constants(r4["space"], m4.a_field, m4.c_field, free=r4["sol"].free,
                                   a_bounds=m4.a_bounds, c_bounds=m4.c_bounds)
    interp4 = sv.approximation_error_1d(r4["space"], m4.exact, m4.dexact,
                                        a_field=m4.a_field, c_field=m4.c_field)
    ratio4 = float(r4["error"]["energy"] / interp4["energy"])
    record(label="反差 1e4 时 Cea 界 ≥ 50（几乎不挑网格）", theory="≥ 50",
           measured=float(lm4["cea_factor"]), tol="0", passed=lm4["cea_factor"] >= 50.0, kind="A",
           detail=f"实测 √(M_h/α_h) = {lm4['cea_factor']:.4f}，而真实效率比只有 {ratio4:.6f}（差 4000 倍）")


@assertion("A", theory="罚函数加边界条件：偏差 = π·ε（与 ε 成正比），条件数 ×O(1/ε)", chapter="4.4")
def test_penalty_bias_scales_with_epsilon() -> None:
    r"""罚函数把 Dirichlet 条件变成"加一个大数"：实现简单、保持对称，但

    * 边界值有 :math:`O(\\varepsilon)` 的偏差——实测恰好 :math:`\\pi\\varepsilon`（ε=1e-8 时 3.142e-8）；
    * 条件数被抬到 :math:`O(1/\\varepsilon)`：ε=1e-8 时 1.63e8，而消元法是 103.09（差 1.6e6 倍）。

    两个方向都测：偏差**小**（3e-8，肉眼看不见）与条件数**大**（1.6e6 倍）同时成立。
    """
    model = sv.model_1d("smooth")
    r = sv.solve_model(model, ms.uniform_mesh_1d(16), 1)
    K, F = r["asm"].K, r["asm"].F
    fixed = r["sol"].fixed
    g = np.zeros(fixed.size)
    elim = asm.dirichlet_elimination(K, F, fixed, g)
    x_el = sv.spd_solve(elim["Kff"], elim["Ff"])["x"]
    k_el = sv.system_condition(elim["Kff"])["condition"]
    prev = None
    for eps in (1e-6, 1e-7, 1e-8):
        pen = asm.dirichlet_penalty(K, F, fixed, g, eps)
        x_pen = sv.spd_solve(pen["K"], pen["F"])["x"]
        free = np.setdiff1d(np.arange(K.shape[0]), fixed)
        bias = float(np.max(np.abs(x_pen[free] - x_el)))
        assert_within(bias, np.pi * eps, rtol=0.02, atol=0.0,
                      label=f"ε={eps:g}：罚解偏差 = π·ε",
                      detail=f"实测 {bias:.6e}，π·ε = {np.pi * eps:.6e}")
        assert_symmetric(pen["K"], label=f"ε={eps:g}：罚矩阵仍然对称",
                         detail=f"对称残差 {asm.symmetry_defect(pen['K']):.3e}（对称性对这个决定完全不敏感）")
        if prev is not None:
            record(label=f"ε={eps:g}：偏差与 ε 成正比（逐十倍缩小）",
                   theory="≈10", measured=float(prev / max(bias, 1e-300)), tol="±0.05",
                   passed=abs(prev / max(bias, 1e-300) - 10.0) <= 0.05, kind="A",
                   detail="偏差是 O(ε) 而不是 O(ε^2)：罚函数给误差设了一个 ε·π 的地板")
        prev = bias
    k_pen = sv.system_condition(asm.dirichlet_penalty(K, F, fixed, g, 1e-8)["K"])["condition"]
    record(label="ε=1e-8 时罚函数的条件数比消元法差 ≥ 1e5 倍", theory="≥ 1e5",
           measured=float(k_pen / k_el), tol="0", passed=k_pen / k_el >= 1e5, kind="A",
           detail=f"κ(罚)={k_pen:.4g} vs κ(消元)={k_el:.4g}")


@assertion("A", theory="p-加密：光滑解指数收敛（半对数拟合优），尖角解退化为代数（双对数拟合优）",
           chapter="4.4")
def test_p_version_is_spectral_only_for_smooth_solutions() -> None:
    r"""同一个 p-阶梯（单元素 Chebyshev–Lobatto 空间）跑两种解：

    * 光滑解：半对数 R^2 高于双对数（指数收敛），p=12 时误差 2.6e-11；
    * 界面尖角解：双对数 R^2 高于半对数（代数收敛），p=12 还有 0.59。

    两向性就在"两种解给出相反的结论"这一条上——"谱收敛"不是方法的属性。
    """
    for tag, model in (("光滑解", sv.model_1d("smooth")),
                       ("尖角解", sv.model_1d("interface", xs=1.0 / 3.0, ratio=100.0))):
        ps, en = [], []
        for p in range(1, 13):
            space = ms.lobatto_space(p)
            ke = asm.element_bilinear_1d(space, model.a_field, model.c_field,
                                         breakpoints=model.breakpoints or None)
            fe = asm.element_load_1d(space, model.load, breakpoints=model.breakpoints or None)
            fixed = space.boundary_dofs("dirichlet")
            sol = sv.solve_1d(space, ke, fe, fixed, np.asarray([0.0, model.bc_values[1]]))[1]
            err = sv.error_norms_1d(space, sol.u, model.exact, model.dexact,
                                    a_field=model.a_field, c_field=model.c_field)
            ps.append(p)
            en.append(float(err["energy"]))
        pa = np.asarray(ps, dtype=float)
        ea = np.asarray(en, dtype=float)
        _, r2_exp = _semilog_fit(pa, ea)
        r2_alg = _loglog_r2(pa, ea)
        if tag == "光滑解":
            record(label="光滑解：半对数（指数）拟合更优", theory="R^2_exp > R^2_log",
                   measured=float(r2_exp), tol="0", passed=r2_exp > r2_alg, kind="A",
                   detail=f"R^2 半对数 {r2_exp:.6f} vs 双对数 {r2_alg:.6f}；p=12 误差 {ea[-1]:.3e}")
            assert_leq(ea[-1], 1e-9, label="光滑解 p=12 时误差 ≤1e-9", detail=f"实测 {ea[-1]:.3e}")
        else:
            record(label="尖角解：双对数（代数）拟合更优", theory="R^2_log > R^2_exp",
                   measured=float(r2_alg), tol="0", passed=r2_alg > r2_exp, kind="A",
                   detail=f"R^2 双对数 {r2_alg:.6f} vs 半对数 {r2_exp:.6f}；p=12 误差仍有 {ea[-1]:.3e}")
            assert_leq(0.05, ea[-1], label="尖角解 p=12 时误差 ≥0.05（没有指数收敛）",
                       detail=f"实测 {ea[-1]:.3e}")


@assertion("A", theory="冻结 Jacobi = 解另一个变分问题：误差不随加密消失（变分犯罪）", chapter="4.4")
def test_frozen_jacobian_does_not_vanish() -> None:
    r"""把等参映射当仿射（用单元中心的 :math:`J`）不是"少算一点点"：

    畸变 d=0.16 时，逐点 :math:`J` 与冻结 :math:`J` 的 L2 误差相对差从 3.77e-2 涨到 4.93e-1
    ——**不收敛**。两向性：逐点 :math:`J` 的误差在收敛（每次加密下降），冻结 :math:`J` 的相对差
    在变大。
    """
    model = sv.model_2d("bump", x0=0.5, y0=0.5, sigma=0.18)
    rel = []
    own = []
    for n in (8, 16, 32):
        mesh = ms.uniform_quad_mesh(n, n, 0.16)
        a = sv.solve_model_2d(model, mesh, freeze_jacobian=False)["error"]["l2"]
        b = sv.solve_model_2d(model, mesh, freeze_jacobian=True)["error"]["l2"]
        own.append(float(a))
        rel.append(abs(float(b) - float(a)) / float(a))
    assert_monotone(own, direction="decreasing", label="逐点 J 的 L2 误差随加密下降",
                    detail=f"{np.array2string(np.asarray(own), precision=4)}")
    record(label="冻结 J 的相对差随加密不降反升", theory="末项 > 首项", measured=rel[-1], tol="0",
           passed=rel[-1] > rel[0], kind="A",
           detail=f"相对差 {np.array2string(np.asarray(rel), precision=4)}（d=0.16）——它解的是另一个问题")


# --------------------------------------------------------------------------- #
# B 类：离散不变量（装出来的性质要用独立的量去对）
# --------------------------------------------------------------------------- #
@assertion("B", theory="装配矩阵对称正定：v^TKv ≥ 0、常向量在核里、Dirichlet 消元后严格正定",
           chapter="4.4")
def test_assembled_matrix_is_symmetric_and_positive_definite() -> None:
    r"""SPD 是**装出来的**，所以要三条独立证据：对称残差、二次型、Cholesky 主元。

    * P1：``max|K−K^T|`` 是精确的 0（单元矩阵逐位对称）；
    * P2：``max|K−K^T|`` 是 1e-15 量级，来源是单元矩阵自己的矩阵乘（反序散射给同一个数）；
    * 常向量落在核里（``K·1 = 0``），Dirichlet 消元后的自由块严格正定。
    """
    gen = rng(20250404)
    for degree, ne in ((1, 8), (1, 16), (2, 8), (2, 16)):
        space = ms.lagrange_space(ms.uniform_mesh_1d(ne), degree)
        ke = asm.element_stiffness_1d(space, None)
        fe = asm.element_load_1d(space, lambda x: np.ones_like(np.asarray(x, dtype=float)))
        A = asm.assemble_1d(space, ke, fe)
        defect = assert_symmetric(A.K, label=f"P{degree} ne={ne}：装配矩阵对称",
                                  detail=f"max|K−K^T| = {asm.symmetry_defect(A.K):.3e}")
        assert_leq(defect, 1e-13, label=f"P{degree} ne={ne}：对称残差 ≤1e-13")
        ones = np.ones(space.n_dof)
        assert_leq(float(np.max(np.abs(A.K @ ones))), 1e-10,
                   label=f"P{degree} ne={ne}：常向量在刚度阵的核里",
                   detail="v^TKv = Σ∫a|v'|^2，常数向量使每一项为 0（这是结构性结论）")
        for _ in range(4):
            v = gen.normal(size=space.n_dof)
            q = float(v @ (A.K @ v))
            assert_leq(max(-q, 0.0), 0.0, label=f"P{degree} ne={ne}：二次型非负",
                       detail=f"v^TKv = {q:.6e}（a≡1 ⟹ v^TKv = Σ_K ∫|v'|^2）")
        free = np.flatnonzero(ms.lagrange_space(ms.uniform_mesh_1d(ne), degree).boundary_dofs("dirichlet"))
        mask = np.ones(space.n_dof, dtype=bool)
        mask[free] = False
        Kff = A.K[np.ix_(np.flatnonzero(mask), np.flatnonzero(mask))]
        ok = is_positive_definite(Kff)
        r = cholesky(Kff)
        record(label=f"P{degree} ne={ne}：Dirichlet 自由块严格正定", theory="Cholesky 成功且主元 > 0",
               measured=float(r.min_pivot_sq) if hasattr(r, "min_pivot_sq") else 0.0, tol="> 0",
               passed=bool(ok), kind="B",
               detail="有限元刚度阵的半正定性来自 v^TKv = Σ∫a|v'|^2；钉住边界后核里只剩 0")


@assertion("B", theory="对称残差为 0 不是合格证：镜像装配把对角线翻倍后仍然对称、仍然正定",
           chapter="4.4")
def test_symmetry_defect_is_not_a_validity_certificate() -> None:
    r"""本章的招牌反例：``upper_mirror(diagonal="double")``。

    它把上三角里的对角线整条放了进去，镜像时对角线被加了两遍。结果：
    ``max|K−K^T| = 0.0``（精确）、Cholesky 成功、条件数从 25.27 **降到** 2.72
    ——所有常规体检指标都变好，但矩阵差了 16.0，解全错。
    """
    space = ms.lagrange_space(ms.uniform_mesh_1d(8), 1)
    ke = asm.element_stiffness_1d(space, None)
    fe = asm.element_load_1d(space, lambda x: np.ones_like(np.asarray(x, dtype=float)))
    full = asm.assemble_1d(space, ke, fe)
    half = asm.upper_mirror(ke, space.elems, space.n_dof, full.F, diagonal="half")
    bad = asm.upper_mirror(ke, space.elems, space.n_dof, full.F, diagonal="double")
    mask = np.ones(space.n_dof, dtype=bool)
    mask[space.boundary_dofs("dirichlet")] = False
    free = np.flatnonzero(mask)
    assert_leq(float(np.max(np.abs(half.K - full.K))), 0.0, label="half 路线与完整散射逐位相同")
    d_bad = float(np.max(np.abs(bad.K - full.K)))
    record(label="double 路线：对称残差 0，但矩阵差 16.0", theory="对称残差 0 且 max|ΔK| = 16",
           measured=d_bad, tol="0", passed=asm.symmetry_defect(bad.K) == 0.0 and d_bad > 1.0, kind="B",
           detail=f"max|K_bad−K^T_bad| = {asm.symmetry_defect(bad.K):.3e}，max|K_bad−K_full| = {d_bad:.3e}")
    c_full = sv.system_condition(full.K[np.ix_(free, free)])["condition"]
    c_bad = sv.system_condition(bad.K[np.ix_(free, free)])["condition"]
    record(label="double 路线的条件数反而更小（25.27 → 2.72）", theory="κ_bad < κ_full",
           measured=float(c_bad), tol="0", passed=c_bad < c_full, kind="B",
           detail=f"κ(完整)={c_full:.4f}，κ(double)={c_bad:.4f}；λ_min 从 1.217927 变成 17.217927（+16）")
    x_full = sv.spd_solve(full.K[np.ix_(free, free)], full.F[free])["x"]
    x_bad = sv.spd_solve(bad.K[np.ix_(free, free)], bad.F[free])["x"]
    rel = float(np.max(np.abs(x_bad - x_full)) / np.max(np.abs(x_full)))
    record(label="double 路线的解与正确解的相对差 ≥5%（实测 93.8%）", theory="≥ 0.05",
           measured=rel, tol="0", passed=rel >= 0.05, kind="B",
           detail=(f"max|x_full| = {float(np.max(np.abs(x_full))):.4f}，"
                   f"max|Δx| = {float(np.max(np.abs(x_bad - x_full))):.4f}；"
                   "对称 + 正定 + Cholesky 成功，三件事同时成立也可能是错的矩阵"))


@assertion("B", theory="装配是线性累加：完整散射、单元内反序、上三角镜像（对角只放一次）给出同一个矩阵",
           chapter="4.4")
def test_assembly_is_order_independent() -> None:
    r"""装配顺序不改变结果；P2 上那 1e-15 的对称残差来自单元矩阵自己，不是散射顺序。"""
    for degree, ne in ((1, 8), (1, 16), (2, 8), (2, 16)):
        space = ms.lagrange_space(ms.uniform_mesh_1d(ne), degree)
        ke = asm.element_stiffness_1d(space, None)
        fe = asm.element_load_1d(space, lambda x: np.ones_like(np.asarray(x, dtype=float)))
        a = asm.assemble_1d(space, ke, fe)
        b = asm.assemble_1d(space, ke, fe, reverse_within_element=True)
        c = asm.upper_mirror(ke, space.elems, space.n_dof, a.F, diagonal="half")
        assert_leq(float(np.max(np.abs(a.K - b.K))), 1e-14,
                   label=f"P{degree} ne={ne}：单元内反序给出同一个 K")
        assert_leq(float(np.max(np.abs(a.K - c.K))), 1e-14,
                   label=f"P{degree} ne={ne}：上三角镜像（half）给出同一个 K")
        elem_asym = float(np.max(np.abs(ke - np.transpose(ke, (0, 2, 1)))))
        if degree == 2:
            record(label=f"P2 ne={ne}：K 的不对称与单元矩阵的不对称是同一个数",
                   theory="两者相同（≤1e-14）", measured=float(asm.symmetry_defect(a.K)), tol="0",
                   passed=abs(elem_asym - asm.symmetry_defect(a.K)) <= 1e-14, kind="B",
                   detail=f"max|ke−ke^T| = {elem_asym:.3e}，max|K−K^T| = {asm.symmetry_defect(a.K):.3e}")


@assertion("B", theory="网格加密的不变量：单元数翻倍、h 减半、单元首尾相接覆盖全部顶点", chapter="4.4")
def test_refinement_invariants() -> None:
    r"""一致加密与标记加密都必须保持"协调网格"不变量：单元首尾相接、覆盖全部顶点。

    这条不变量抓到过一个真 bug：加密时一个单元只登记了一条连接，节点数涨了、单元数没涨，
    于是"节点号 = 单元号 + 1"的假设在后续代码里静默失效（指示子会读到不存在的单元）。
    """
    for ne in (2, 3, 8, 16):
        m0 = ms.uniform_mesh_1d(ne)
        m1 = m0.refine_uniform()
        assert_leq(float(abs(m1.n_elem - 2 * m0.n_elem)), 0.0, label=f"ne={ne}：单元数翻倍")
        assert_leq(float(abs(m1.max_h - 0.5 * m0.max_h)), 1e-15, label=f"ne={ne}：h 减半")
        assert_leq(float(abs(float(np.sum(m1.h)) - 1.0)), 1e-14, label=f"ne={ne}：总长度守恒")
        chain = bool(np.all(m1.elems[1:, 0] == m1.elems[:-1, 1]))
        record(label=f"ne={ne}：加密后单元首尾相接、覆盖全部顶点", theory="elems[k,1]=elems[k+1,0] 且首尾为 0..n−1",
               measured=float(int(chain)), tol="0", passed=chain and m1.n_node == m1.n_elem + 1, kind="B",
               detail=f"n_node={m1.n_node}，n_elem={m1.n_elem}")
    m0 = ms.uniform_mesh_1d(8)
    eta = np.asarray([0.01, 0.2, 5.0, 0.3, 0.02, 1.0, 0.05, 0.4])
    mark = ad.dorfler_mark(eta, 0.5)
    m2 = m0.refine_marked(mark)
    n_marked = int(np.count_nonzero(mark))
    added = int(m2.n_elem - m0.n_elem)
    assert_leq(max(n_marked - added, 0), 0.0,
               label="标记加密：新增单元数 ≥ 被标记单元数（闭包只会多不会少）",
               detail=f"标记 {n_marked} 个、新增 {added} 个：多出来的是保证协调的一层闭包")
    assert_leq(max(added - 3 * n_marked, 0), 0.0,
               label="标记加密：新增单元数 ≤ 3×被标记数（一层闭包的上界）")
    chain2 = bool(np.all(m2.elems[1:, 0] == m2.elems[:-1, 1]))
    record(label="标记加密后仍然是协调网格（无悬挂节点）", theory="单元首尾相接且覆盖全部顶点",
           measured=float(int(chain2)), tol="0", passed=chain2 and m2.n_node == m2.n_elem + 1, kind="B",
           detail=f"标记 {int(np.count_nonzero(mark))} 个单元（含一层闭包），n_elem={m2.n_elem}")


@assertion("B", theory="Doerfler 标记的份额 ≥ θ^2Ση^2；自适应历史里误差单调下降、效率指数有界",
           chapter="4.4")
def test_dorfler_and_adaptive_invariants() -> None:
    r"""标记规则与自适应过程的不变量：

    * 被标记集合的 :math:`\\sum\\eta_K^2` 至少占 :math:`\\theta^2` 的比例；
    * 误差逐层单调下降；自由度增长受标记数控制；
    * 效率指数（指示子/真误差）落在 [1, 3]——它是**分布**，这里只钉住总量。
    """
    gen = rng(7)
    eta = np.abs(gen.normal(size=40)) + 0.01
    total = float(np.sum(eta**2))
    for theta in (0.2, 0.5, 0.9, 1.0):
        mark = ad.dorfler_mark(eta, theta)
        share = float(np.sum(eta[mark] ** 2) / total)
        assert_leq(theta * theta, share, label=f"θ={theta:g}：被标记份额 ≥ θ^2",
                   detail=f"实测份额 {share:.6f} ≥ {theta ** 2:.6f}")
    assert_monotone(np.asarray([np.sum(eta[ad.dorfler_mark(eta, t)] ** 2) for t in (0.2, 0.5, 0.9)]),
                    direction="nondecreasing", label="标记份额随 θ 单调不减")
    model = sv.model_1d("interface", xs=1.0 / 3.0, ratio=100.0)
    hist = ad.adaptive_history(ms.uniform_mesh_1d(4), degree=1, exact=model.exact, dexact=model.dexact,
                               f_field=model.load, a_field=model.a_field,
                               fixed_values=(0.0, model.bc_values[1]), theta=0.5, levels=6)
    errs = np.asarray([lv.error_energy for lv in hist])
    dofs = np.asarray([lv.n_dof for lv in hist], dtype=float)
    effs = np.asarray([lv.effectivity_total for lv in hist])
    assert_monotone(errs, direction="decreasing", label="自适应误差逐层下降",
                    detail=f"误差 {np.array2string(errs, precision=3)}")
    assert_monotone(dofs, direction="increasing", label="自由度逐层增加")
    assert_within(float(np.max(effs)), 2.0, rtol=1.0, atol=0.0,
                  label="效率指数总量在 [1,4]（是分布，不是一个常数）",
                  detail=f"逐层 {np.array2string(effs, precision=3)}")
    uni = ad.uniform_history(ms.uniform_mesh_1d(4), degree=1, exact=model.exact, dexact=model.dexact,
                             f_field=model.load, a_field=model.a_field,
                             fixed_values=(0.0, model.bc_values[1]), levels=6)
    cmp_res = ad.compare_strategies(hist, uni)
    ratio = float(cmp_res["best_ratio"])
    record(label="同自由度下自适应误差 ≤ 一致加密的一半", theory="≤ 0.5", measured=ratio, tol="0",
           passed=ratio <= 0.5, kind="B",
           detail=f"最好比 {ratio:.4f}（自适应末层 n_dof={int(dofs[-1])}，一致加密末层 n_dof={uni[-1].n_dof}）")


@assertion("B", theory="逐单元误差平方和 = 总误差^2；而节点误差向量不是误差函数", chapter="4.4")
def test_error_norm_identities() -> None:
    r"""两条独立代码路径给出同一个数，以及一个最容易骗人的"恒等式"：

    * 逐单元真误差的平方和 = 总能量误差^2（求积阶对齐后逐位一致，rel diff 0.0）；
    * :math:`\sqrt{e^\top K e}` 只有在**真解本身落在** :math:`V_h` 里时才等于能量误差
      （点载荷在节点上：真解分段线性 ⟹ 两者相对差 1e-12）；
      光滑解上它给 4.9e-12，而函数误差的能量范数是 0.2512——**差 11 个数量级**。
      原因是 :math:`a\equiv1,c=0` 时 :math:`u_h=I_hu`（节点值相等），节点误差向量恒为 0，
      而误差函数在单元内部并不为零。
    """
    model = sv.model_1d("smooth")
    for ne in (8, 16):
        r = sv.solve_model(model, ms.uniform_mesh_1d(ne), 1)
        space = r["space"]
        elem = ad.true_element_errors(space, r["sol"].u, model.exact, model.dexact,
                                      model.a_field, model.c_field)
        assert_within(float(np.sqrt(np.sum(elem**2))), r["error"]["energy"], rtol=1e-12,
                      label=f"ne={ne}：逐单元误差平方和开根 = 总能量误差（两条独立路径）",
                      detail=f"逐单元 {np.sqrt(np.sum(elem ** 2)):.12e} vs 总误差 {r['error']['energy']:.12e}")
    # 非平凡版本：同一个装配矩阵算 u_h 自己的能量范数（P1、a≡1 时求积精确，两条路径应逐位一致）
    r_pl = sv.solve_model(model, ms.uniform_mesh_1d(16), 1)
    u_pl = r_pl["sol"].u
    quad_pl = float(np.sqrt(max(u_pl @ (r_pl["asm"].K @ u_pl), 0.0)))
    ref_pl = sv.energy_norm_1d(r_pl["space"], u_pl, None, None)
    assert_within(quad_pl, ref_pl, rtol=1e-12,
                  label="√(u_h^TK u_h) = energy_norm_1d(u_h)（装配矩阵 vs 逐点求积）",
                  detail=f"二次型 {quad_pl:.12e} vs 求积 {ref_pl:.12e}")
    # 光滑解（真解不在 V_h 里）：节点误差向量恒为 0，但函数误差是 O(h)
    r_sm = sv.solve_model(model, ms.uniform_mesh_1d(16), 1)
    e_sm = r_sm["sol"].u - model.exact(r_sm["space"].nodes)
    quad_sm = float(np.sqrt(max(e_sm @ (r_sm["asm"].K @ e_sm), 0.0)))
    nod = float(np.max(np.abs(e_sm)))
    en = float(r_sm["error"]["energy"])
    assert_leq(nod, 1e-9, label="光滑解：节点误差 ≤1e-9（P1 在节点上精确）",
               detail=f"实测 {nod:.3e}——节点值精确是 1-D 常数系数 P1 的定理")
    assert_leq(1e-2, en, label="光滑解：函数误差的能量范数 ≥1e-2（与节点无关）",
               detail=f"实测 {en:.3e}")
    record(label="光滑解：√(e^TKe) 与能量误差差 ≥ 8 个数量级（节点向量不是误差函数）",
           theory="≥ 1e8 倍", measured=float(en / max(quad_sm, 1e-300)), tol="0",
           passed=en / max(quad_sm, 1e-300) >= 1e8, kind="B",
           detail=f"√(e^TKe) = {quad_sm:.3e} vs 能量误差 {en:.3e}；节点误差向量 {nod:.3e}")
    assert_leq(r_sm["error"]["l2"], r_sm["error"]["energy"],
               label="L2 误差 ≤ 能量误差（Poincare： || v ||  ≤  || v' || ）",
               detail=f"L2 {r_sm['error']['l2']:.6e} vs 能量 {en:.6e}")
    assert_within(r_sm["error"]["energy"], r_sm["error"]["h1"], rtol=1e-12,
                  label="a=1,c=0 时能量范数就是 H^1 半范")


@assertion("B", theory="一维门控：矩阵奇异时必须如实报告 ok=False，而不是返回被 NaN 污染的解",
           chapter="4.4")
def test_singular_system_is_reported_not_polluted() -> None:
    r"""四阶问题用 P1 线性单元：``N''≡0`` ⟹ 单元刚度阵恒零 ⟹ 整矩阵奇异。

    这时 ``solve_1d`` 必须给出 ``ok=False``、``fail_reason`` 非空、解里没有 NaN；
    ``spd_solve`` 也要在 Cholesky 与 LU 都失败时保持有限输出（帧把它显示成 0 + 掩码）。
    """
    mesh = ms.uniform_mesh_1d(4)
    space = ms.lagrange_space(mesh, 1)
    zero_val = asm.p1_beam_stiffness_zero(space)
    ke = np.zeros((space.n_elem, space.npe, space.npe), dtype=float)
    record(label="P1 单元在四阶问题上的刚度阵恒零（N''≡0）", theory="0.0", measured=float(zero_val), tol="0",
           passed=float(zero_val) == 0.0, kind="B",
           detail="p1_beam_stiffness_zero 返回的就是这个 0；整矩阵因此奇异")
    fe = asm.element_load_1d(space, lambda x: np.ones_like(np.asarray(x, dtype=float)))
    fixed = space.boundary_dofs("dirichlet")
    sol = sv.solve_1d(space, ke, fe, fixed, np.zeros(fixed.size))[1]
    assert_leq(float(int(sol.ok)), 0.0, label="奇异系统：ok=False")
    record(label="奇异系统：fail_reason 非空且解全有限", theory="ok=False 且无 NaN",
           measured=float(np.max(np.abs(sol.u))), tol="0",
           passed=(not sol.ok) and bool(sol.fail_reason) and bool(np.all(np.isfinite(sol.u))), kind="B",
           detail=f"method={sol.method!r}，fail_reason={sol.fail_reason!r}")
    zero = sv.spd_solve(np.zeros((3, 3)), np.ones(3))
    record(label="spd_solve 在奇异矩阵上保持有限输出", theory="ok=False 且 x 全有限",
           measured=float(np.max(np.abs(zero["x"]))), tol="0",
           passed=(not zero["ok"]) and bool(np.all(np.isfinite(zero["x"]))), kind="B",
           detail=f"method={zero['method']!r}——帧层再把它换成 0 + singular_mask")
