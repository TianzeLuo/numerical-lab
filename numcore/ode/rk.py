r"""3.1 章（单步法、阶条件与自适应步长）的**方法族**真源。

数学对象
--------

显式 Runge--Kutta 方法由 Butcher 表 :math:`(A,b)` 给出（:math:`A` 严格下三角、:math:`c=Ae`）：

.. math::
    k_i=f\Bigl(t+c_ih,\;y_n+h\sum_{j<i}a_{ij}k_j\Bigr),\qquad
    y_{n+1}=y_n+h\sum_i b_ik_i .

**阶条件**写成有根树：每棵树 :math:`\tau` 对应一个初等权
:math:`\Phi(\tau)=b^{\mathsf T}\Phi(\tau)`，其中
:math:`\Phi_i(\bullet)=1`、:math:`\Phi_i([\tau_1,\dots,\tau_m])=\prod_j(A\Phi(\tau_j))_i`；
方法 :math:`p` 阶当且仅当对一切 :math:`\rho(\tau)\le p` 有 :math:`\Phi(\tau)=1/\gamma(\tau)`
（:math:`\gamma` 是密度）。本模块**现场枚举**根树、**现场**用表算这些和——
既不查 :math:`1/\gamma` 表，也不把 :math:`1+z+z^2/2+\cdots` 抄进来：阶是**算出来的结论**。

**嵌入对** :math:`(b,\hat b)` 给出误差估计 :math:`\hat e=y_{n+1}-\hat y_{n+1}`，
它估的是**哪一个**解的局部误差，是本章最容易被读错的地方（见 :func:`local_error_ladder`）。

**稠密输出**用三次 Hermite（:math:`y_n,f_n,y_{n+1},f_{n+1}`）：它的步内精度是
:math:`\min(4,p+1)`，对 :math:`p\ge4` 的方法**比方法本身低阶**——这件事由断言实测，
而不是由「插值一般都很好」的直觉担保。

自研边界（P2/P3）
-----------------
本章只做逐元素运算与矩阵乘法，**不解线性系统**（那是隐式方法的事，见 3.2 章），
不碰任何 numpy 成品算法。幂律斜率在 :func:`loglog_slope` 里自写最小二乘，
它与 :func:`numcore.verify.fit_order` 的口径一致（由 ``tests/3.1`` 逐位对拍）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import numpy as np

__all__ = [
    "RHS",
    "Tableau",
    "StepResult",
    "Trajectory",
    "Problem",
    "OrderConditions",
    "Ladder",
    "METHODS",
    "PAIR_METHODS",
    "PROBLEM_NAMES",
    "tableau",
    "as_tableau",
    "tableaux",
    "pair_methods",
    "rk_step",
    "scaled_error_measure",
    "plain_error_measure",
    "rooted_trees",
    "tree_order",
    "tree_density",
    "tree_label",
    "tree_layout",
    "elementary_weight",
    "order_conditions",
    "measured_order",
    "hermite_dense",
    "dense_step",
    "dense_error_ladder",
    "problem",
    "problem_names",
    "loglog_slope",
    "asymptotic_window",
    "local_orders",
    "pointwise_error",
    "stability_polynomial",
    "evaluate_polynomial",
    "negative_real_axis_endpoint",
    "critical_step_for_decay",
    "integrate_fixed",
    "error_ladder",
    "local_error_ladder",
    "estimate_local_error",
    "step_error_reference",
]

#: 右端函数 ``f(t, y) -> dy`` 的签名（与 :mod:`numcore.ode.implicit` 同款）。
RHS = Callable[[float, np.ndarray], np.ndarray]

#: 未爆炸的显示上限：超过它一律停止并如实标注（``inf`` 进不了帧）。
BLOWUP = 1.0e12


# --------------------------------------------------------------------------- #
# 1. Butcher 表
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Tableau:
    r"""一张显式 Runge--Kutta 表：:math:`(A,b)`，可选嵌入低阶权 :math:`\hat b`。

    ``order`` 是 :math:`b` 的阶，``order_hat`` 是 :math:`\hat b` 的阶；
    ``fsal``（first-same-as-last）为真时最后一级就是 :math:`f(t+h,y_{n+1})`，
    于是下一步的第一级与稠密输出的右端点导数都**不必再求值**——这是「每步代价」里
    最容易被忽略的一项。
    """

    name: str
    label: str
    order: int
    A: tuple[tuple[float, ...], ...]
    b: tuple[float, ...]
    b_hat: tuple[float, ...] | None = None
    order_hat: int = 0
    fsal: bool = False
    note: str = ""

    @property
    def stages(self) -> int:
        """级数 :math:`s`（= 一次试探步的 :math:`f` 求值次数上界）。"""
        return len(self.b)

    def A_array(self) -> np.ndarray:
        return np.asarray(self.A, dtype=float)

    def b_array(self) -> np.ndarray:
        return np.asarray(self.b, dtype=float)

    def b_hat_array(self) -> np.ndarray:
        if self.b_hat is None:
            raise ValueError(f"{self.name} 没有嵌入低阶权 b_hat（不是嵌入对）")
        return np.asarray(self.b_hat, dtype=float)

    def c_array(self) -> np.ndarray:
        A = self.A_array()
        return A @ np.ones(A.shape[0], dtype=float)

    def pair_label(self) -> str:
        """形如 ``Bogacki–Shampine 3(2)`` 的显示名（无嵌入对时只给方法名）。"""
        if self.b_hat is None:
            return f"{self.label}（{self.order} 阶，无嵌入对）"
        return f"{self.label} {self.order}({self.order_hat})"


#: 全部表。**不查表**指的是不从「已知方法的系数表」里读 :math:`1/\gamma`；
#: 方法系数本身当然是定义的一部分。每个方法的阶由 :func:`measured_order` 现场验证。
_TABLES: tuple[Tableau, ...] = (
    Tableau(
        name="euler",
        label="显式 Euler",
        order=1,
        A=((0.0,),),
        b=(1.0,),
        note="一级一阶；局部误差 O(h²)、全局误差 O(h)。",
    ),
    Tableau(
        name="midpoint",
        label="中点法",
        order=2,
        A=((0.0, 0.0), (0.5, 0.0)),
        b=(0.0, 1.0),
        note="两级二阶；b 有一个零权，故不是 FSAL。",
    ),
    Tableau(
        name="heun",
        label="Heun（改进 Euler）",
        order=2,
        A=((0.0, 0.0), (1.0, 0.0)),
        b=(0.5, 0.5),
        b_hat=(1.0, 0.0),
        order_hat=1,
        note="与显式 Euler 配成 Heun–Euler 2(1) 对：估计量 O(h²)，被推进的解局部误差 O(h³)。",
    ),
    Tableau(
        name="rk3",
        label="Kutta 三阶",
        order=3,
        A=((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (-1.0, 2.0, 0.0)),
        b=(1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0),
        note="三级三阶；c₃=1 但第三级不是 f(t+h,y_{n+1})，故非 FSAL。",
    ),
    Tableau(
        name="rk4",
        label="经典 RK4",
        order=4,
        A=(
            (0.0, 0.0, 0.0, 0.0),
            (0.5, 0.0, 0.0, 0.0),
            (0.0, 0.5, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
        ),
        b=(1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0),
        note="四级四阶；稳定多项式的系数正是 bᵀA^{j−1}e = 1/j!（与 3.2 章对拍）。",
    ),
    Tableau(
        name="bs23",
        label="Bogacki–Shampine",
        order=3,
        A=(
            (0.0, 0.0, 0.0, 0.0),
            (0.5, 0.0, 0.0, 0.0),
            (0.0, 0.75, 0.0, 0.0),
            (2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0, 0.0),
        ),
        b=(2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0, 0.0),
        b_hat=(7.0 / 24.0, 0.25, 1.0 / 3.0, 0.125),
        order_hat=2,
        fsal=True,
        note="三级三阶 + FSAL：第四级恰好是 f(t+h,y_{n+1})，下一步第一级免费。",
    ),
    Tableau(
        name="rkf45",
        label="Fehlberg",
        order=4,
        A=(
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.25, 0.0, 0.0, 0.0, 0.0, 0.0),
            (3.0 / 32.0, 9.0 / 32.0, 0.0, 0.0, 0.0, 0.0),
            (1932.0 / 2197.0, -7200.0 / 2197.0, 7296.0 / 2197.0, 0.0, 0.0, 0.0),
            (439.0 / 216.0, -8.0, 3680.0 / 513.0, -845.0 / 4104.0, 0.0, 0.0),
            (-8.0 / 27.0, 2.0, -3544.0 / 2565.0, 1859.0 / 4104.0, -11.0 / 40.0, 0.0),
        ),
        b=(25.0 / 216.0, 0.0, 1408.0 / 2565.0, 2197.0 / 4104.0, -0.2, 0.0),
        b_hat=(
            16.0 / 135.0,
            0.0,
            6656.0 / 12825.0,
            28561.0 / 56430.0,
            -9.0 / 50.0,
            2.0 / 55.0,
        ),
        order_hat=5,
        fsal=False,
        note=(
            "Fehlberg 4(5)：推进四阶解、用五阶解估误差。两个解共享全部级，"
            "估计量的**阶**对得上，但常数可以差几个量级——教科书里最有名的不可靠估计器。"
        ),
    ),
    Tableau(
        name="dopri5",
        label="Dormand–Prince",
        order=5,
        A=(
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0, 0.0),
            (19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0, 0.0, 0.0, 0.0),
            (
                9017.0 / 3168.0,
                -355.0 / 33.0,
                46732.0 / 5247.0,
                49.0 / 176.0,
                -5103.0 / 18656.0,
                0.0,
                0.0,
            ),
            (35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0),
        ),
        b=(35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0),
        b_hat=(
            5179.0 / 57600.0,
            0.0,
            7571.0 / 16695.0,
            393.0 / 640.0,
            -92097.0 / 339200.0,
            187.0 / 2100.0,
            1.0 / 40.0,
        ),
        order_hat=4,
        fsal=True,
        note="五阶 + 四阶嵌入 + FSAL；ODE 求解器的默认工作马。",
    ),
)

#: 方法名（按阶数升序，同阶按表内顺序）。
METHODS: tuple[str, ...] = tuple(t.name for t in _TABLES)

#: 带嵌入低阶解、可做误差估计的方法名。
PAIR_METHODS: tuple[str, ...] = tuple(t.name for t in _TABLES if t.b_hat is not None)


def tableaux() -> tuple[Tableau, ...]:
    """全部 Butcher 表（不可变元组，调用方拿到的是只读数据）。"""
    return _TABLES


def pair_methods() -> tuple[Tableau, ...]:
    """全部嵌入对（能做误差估计与自适应步长的方法）。"""
    return tuple(t for t in _TABLES if t.b_hat is not None)


def tableau(name: str) -> Tableau:
    """按名字取 Butcher 表；名字不认识就抛 :class:`ValueError`（不返回默认值）。"""
    for t in _TABLES:
        if t.name == name:
            return t
    raise ValueError(f"未知方法 {name!r}（可用：{list(METHODS)}）")


def as_tableau(method: "str | Tableau") -> Tableau:
    """把"方法名或一张现成的表"统一成 :class:`Tableau`。

    为什么允许传表：本章的核心结论是"阶条件破坏 ⟹ 阶掉"，要证明它就必须能拿一张
    **被扰动过的表**去真跑积分。测试里用 :func:`dataclasses.replace` 造出这样的表，
    与"注册表里只有 8 个合法方法"并不矛盾——注册表管的是交付物的方法族，
    阶条件管的是任意一张表。
    """
    return method if isinstance(method, Tableau) else tableau(method)


# --------------------------------------------------------------------------- #
# 2. 单步
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StepResult:
    """一次 Rung--Kutta 试探步的全部可观测物。"""

    method: str
    t: float
    h: float
    y: np.ndarray
    y_low: np.ndarray
    error: np.ndarray
    stages: np.ndarray
    c: np.ndarray
    n_f_evals: int
    fsal_ready: bool

    def has_embedding(self) -> bool:
        """``y_low`` 是否真的来自嵌入低阶权（否则它与 ``y`` 逐位相同）。"""
        return bool(np.any(self.error != 0.0))


def rk_step(
    name: "str | Tableau",
    f: RHS,
    t: float,
    y: Any,
    h: float,
    *,
    f_start: np.ndarray | None = None,
) -> StepResult:
    r"""一次显式 RK 步：:math:`y^+=y+h\sum_ib_ik_i`（同时给出嵌入低阶解 :math:`\hat y^+`）。

    ``f_start`` 允许调用方把上一步末尾已经算过的 :math:`f(t,y)` 递进来（FSAL 复用），
    这样「每步代价」里省下的那次求值是**算进账**的，而不是嘴上说说。
    """
    tab = as_tableau(name)
    A = tab.A_array()
    b = tab.b_array()
    s = tab.stages
    c = tab.c_array()
    yy = np.asarray(y, dtype=float).reshape(-1)
    m = int(yy.size)
    if not h > 0.0:
        raise ValueError(f"步长必须为正，得到 {h}")

    ks = np.zeros((s, m), dtype=float)
    n_ev = 0
    for i in range(s):
        if i == 0:
            if f_start is None:
                ks[0] = np.asarray(f(t, yy), dtype=float)
                n_ev += 1
            else:
                ks[0] = np.asarray(f_start, dtype=float).reshape(-1)
        else:
            stage = yy + h * (A[i, :i] @ ks[:i])
            ks[i] = np.asarray(f(t + float(c[i]) * h, stage), dtype=float)
            n_ev += 1
    y_new = yy + h * (b @ ks)
    if tab.b_hat is None:
        y_low = y_new.copy()
    else:
        y_low = yy + h * (tab.b_hat_array() @ ks)
    fsal_ready = bool(tab.fsal and np.array_equal(b, A[-1]))
    return StepResult(
        method=tab.name,
        t=float(t),
        h=float(h),
        y=y_new,
        y_low=y_low,
        error=y_new - y_low,
        stages=ks,
        c=c,
        n_f_evals=n_ev,
        fsal_ready=fsal_ready,
    )


def plain_error_measure(error: Any) -> float:
    r"""无权 RMS :math:`\sqrt{\frac1m\sum_i e_i^2}`——用于**量级对拍**（如估计器 vs 真误差）。

    与 :func:`scaled_error_measure` 的分工：后者带 ``atol/rtol`` 尺度，是**接受判据**；
    前者不带任何尺度，是「两个误差谁更大」的公平尺子。拿带容差的度量去比值，
    得到的数字里混着 :math:`1/\mathrm{atol}` 这个因子（实测凭空多出 :math:`10^{14}` 倍），
    这一点在 ``tests/3.1`` 里被专门记了一笔。
    """
    e = np.asarray(error, dtype=float).reshape(-1)
    return float(np.sqrt(np.mean(e * e)))


def scaled_error_measure(
    error: Any,
    y_new: Any,
    y_old: Any,
    *,
    atol: float,
    rtol: float,
) -> float:
    r"""误差的**加权 RMS** 度量 :math:`\|\hat e\|=\sqrt{\frac1m\sum_i(\hat e_i/s_i)^2}`。

    :math:`s_i=\mathrm{atol}+\mathrm{rtol}\max(|y_i^{new}|,|y_i^{old}|)`——
    这个「混合绝对/相对」的尺度是 Hairer--Wanner 的标准口径：单用相对容差，
    解穿过 0 时误差会被自己的小分母放大成假警报；单用绝对容差，解到 :math:`10^6`
    时又完全失去意义。度量值是**接受判据的输入**：``≤ 1`` 才算通过。
    """
    e = np.asarray(error, dtype=float).reshape(-1)
    yn = np.asarray(y_new, dtype=float).reshape(-1)
    yo = np.asarray(y_old, dtype=float).reshape(-1)
    if not (atol > 0.0 or rtol > 0.0):
        raise ValueError("atol 与 rtol 不能同时为 0：误差度量会退化成 0/0")
    scale = float(atol) + float(rtol) * np.maximum(np.abs(yn), np.abs(yo))
    r = e / scale
    return float(np.sqrt(np.mean(r * r)))


# --------------------------------------------------------------------------- #
# 3. 阶条件：有根树、密度、初等权
# --------------------------------------------------------------------------- #
#: 有根树的表示：嵌套元组；**空元组是叶子**（单节点树）。
#: 例：``()`` = ``*``；``((),)`` = ``[*]``；``((), ())`` = ``[*,*]``。
Tree = tuple


def tree_order(t: Tree) -> int:
    r"""树的阶 = 节点数 :math:`\rho(\tau)`。"""
    return 1 + sum(tree_order(c) for c in t)


def tree_density(t: Tree) -> int:
    r"""树的密度 :math:`\gamma(\tau)=\rho(\tau)\prod_i\gamma(\tau_i)`。"""
    d = tree_order(t)
    for c in t:
        d *= tree_density(c)
    return d


def tree_label(t: Tree) -> str:
    r"""Butcher 括号记法（纯 ASCII，便于进帧与 LaTeX）：叶子 ``*``，否则 ``[τ₁,…,τ_m]``。"""
    if not t:
        return "*"
    return "[" + ",".join(tree_label(c) for c in t) + "]"


def _subtree_multisets(pool: tuple[Tree, ...], total: int, start: int = 0) -> Iterable[tuple[Tree, ...]]:
    """枚举「节点数之和为 ``total``」的子树的**多重集**（按池中下标非降序，保证不重复）。"""
    if total == 0:
        yield ()
        return
    for i in range(start, len(pool)):
        sub = pool[i]
        k = tree_order(sub)
        if k > total:
            continue
        for rest in _subtree_multisets(pool, total - k, i):
            yield (sub,) + rest


def rooted_trees(max_order: int) -> tuple[Tree, ...]:
    r"""阶数 :math:`\le` ``max_order`` 的全部**有根树**（按阶数升序、同阶按规范序）。

    递推枚举：一棵 :math:`n` 节点的树 = 根 + 一个「子树多重集」，其节点数之和为
    :math:`n-1`。多重集按下标非降序生成，所以每棵同构树**只出现一次**
    （计数 1, 1, 2, 4, 9 是断言里的第一条自检）。
    """
    n_max = int(max_order)
    if n_max < 1:
        raise ValueError(f"阶数至少为 1，得到 {max_order}")
    out: list[Tree] = []
    pool: list[Tree] = []
    for n in range(1, n_max + 1):
        fresh: list[Tree] = []
        for children in _subtree_multisets(tuple(pool), n - 1):
            fresh.append(tuple(children))
        out.extend(fresh)
        pool.extend(fresh)
    return tuple(out)


def elementary_weight(A: Any, b: Any, tree: Tree) -> float:
    r"""初等权 :math:`\Phi(\tau)=b^{\mathsf T}\Phi(\tau)`（由 Butcher 表现场求和）。

    :math:`\Phi_i(\bullet)=1`，:math:`\Phi_i([\tau_1,\dots,\tau_m])=\prod_j(A\Phi(\tau_j))_i`：
    每一层子树就是乘一次 :math:`A`，叶子的乘积是 1。**这就是阶条件的全部内容**——
    不存在「查表得到的阶」。
    """
    Am = np.asarray(A, dtype=float)
    bv = np.asarray(b, dtype=float).reshape(-1)
    if Am.shape[0] != bv.size or Am.shape[1] != bv.size:
        raise ValueError(f"A 与 b 的形状不匹配：A{Am.shape}，b{bv.shape}")

    def phi_vec(t: Tree) -> np.ndarray:
        if not t:
            return np.ones(bv.size, dtype=float)
        acc = np.ones(bv.size, dtype=float)
        for child in t:
            acc = acc * (Am @ phi_vec(child))
        return acc

    return float(bv @ phi_vec(tree))


@dataclass(frozen=True)
class OrderConditions:
    r"""一个方法的阶条件表：逐棵树给出 :math:`\gamma`、:math:`\Phi` 与残差 :math:`1-\gamma\Phi`。"""

    method: str
    weights: str
    max_order: int
    orders: np.ndarray
    densities: np.ndarray
    values: np.ndarray
    residuals: np.ndarray
    labels: tuple[str, ...]
    declared_order: int
    measured_order: int

    @property
    def n_failed(self) -> int:
        """残差超出机器精度量级的条件条数（= 该方法**没有**满足的条件）。"""
        return int(np.count_nonzero(np.abs(self.residuals) > 1e-12))

    def passing_mask(self, tol: float = 1e-12) -> np.ndarray:
        return (np.abs(self.residuals) <= tol).astype(float)


def order_conditions(name: "str | Tableau", max_order: int = 5, *, use_hat: bool = False) -> OrderConditions:
    r"""逐棵树算出 :math:`1-\gamma(\tau)\Phi(\tau)`。

    ``use_hat=True`` 时换成嵌入低阶权 :math:`\hat b`——这正是「误差估计器**估计的是谁**」
    的答案：估计量 :math:`\hat e=(b-\hat b)^{\mathsf T}\Phi` 的阶由 :math:`\hat b`
    满足到第几阶条件决定。
    """
    tab = as_tableau(name)
    b = tab.b_hat_array() if use_hat else tab.b_array()
    if use_hat and tab.b_hat is None:
        raise ValueError(f"{tab.name} 没有嵌入低阶权，无法给出 b_hat 的阶条件")
    trees = rooted_trees(int(max_order))
    n = len(trees)
    orders = np.zeros(n, dtype=float)
    dens = np.zeros(n, dtype=float)
    vals = np.zeros(n, dtype=float)
    for k, tr in enumerate(trees):
        orders[k] = tree_order(tr)
        dens[k] = tree_density(tr)
        vals[k] = elementary_weight(tab.A_array(), b, tr)
    resid = 1.0 - dens * vals
    tol = 1e-10 * max(1.0, float(np.max(np.abs(b))))
    measured = 0
    for p in range(1, int(max_order) + 1):
        sel = np.abs(resid[orders <= p]) <= tol
        if bool(np.all(sel)):
            measured = p
        else:
            break
    return OrderConditions(
        method=tab.name,
        weights="b_hat" if use_hat else "b",
        max_order=int(max_order),
        orders=orders,
        densities=dens,
        values=vals,
        residuals=resid,
        labels=tuple(tree_label(tr) for tr in trees),
        declared_order=tab.order_hat if use_hat else tab.order,
        measured_order=measured,
    )


def measured_order(name: "str | Tableau", *, max_order: int = 5, tol: float = 1e-10, use_hat: bool = False) -> int:
    r"""**实测阶**：由阶条件残差得到的最大 :math:`p`（残差全在 ``tol`` 以内）。

    这是「阶」的第一条独立证据链：它只用 Butcher 表与 :math:`1/\gamma` 的定义，
    与「跑一个算例量误差斜率」完全独立。两条链必须给出同一个 :math:`p`。
    """
    res = order_conditions(name, max_order=max_order, use_hat=use_hat)
    tab = as_tableau(name)
    b = tab.b_hat_array() if use_hat else tab.b_array()
    scale = max(1.0, float(np.max(np.abs(b))))
    p = 0
    for k in range(1, int(max_order) + 1):
        sel = np.abs(res.residuals[res.orders <= k]) <= tol * scale
        if bool(np.all(sel)):
            p = k
        else:
            break
    return p


def tree_layout(tree: Tree) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""把一棵有根树摆进单位方框：返回 ``(x, y, parent)``（节点 0 是根）。

    布局规则是「叶子等距、父节点居中、深度即高度」——纯几何，无随机、无迭代：
    同一棵树在任何时候都得到同一张图（帧与讲义里的图像必须一致）。
    :math:`y` 向上生长（根在 :math:`y=1`），未归一化前叶子的 :math:`y=0`。
    """
    xs: list[float] = []
    ys: list[float] = []
    parents: list[int] = []
    counter = [0]

    def layout(t: Tree, parent: int) -> tuple[float, float]:
        idx = len(xs)
        xs.append(0.0)
        ys.append(0.0)
        parents.append(parent)
        if not t:
            xs[idx] = float(counter[0])
            counter[0] += 1
            return float(xs[idx]), 0.0
        centres: list[float] = []
        depth = 0.0
        for child in t:
            cx, cd = layout(child, idx)
            centres.append(cx)
            depth = max(depth, cd + 1.0)
        xs[idx] = float(np.mean(centres))
        ys[idx] = depth
        return float(xs[idx]), depth

    layout(tree, -1)
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    width = float(np.max(x) - np.min(x))
    height = float(np.max(y)) if y.size else 1.0
    x = (x - float(np.min(x))) / width if width > 0.0 else np.full_like(x, 0.5)
    y = y / height if height > 0.0 else y
    return x, y, np.asarray(parents, dtype=float)


# --------------------------------------------------------------------------- #
# 4. 稠密输出（步内插值）
# --------------------------------------------------------------------------- #
def hermite_dense(
    y0: Any,
    y1: Any,
    f0: Any,
    f1: Any,
    h: float,
    theta: Any,
) -> np.ndarray:
    r"""三次 Hermite 插值 :math:`H(\theta)`（:math:`\theta\in[0,1]` 是步内参数）。

    .. math::
        H=(1-3\theta^2+2\theta^3)y_n+h(\theta-2\theta^2+\theta^3)f_n
          +(3\theta^2-2\theta^3)y_{n+1}+h(-\theta^2+\theta^3)f_{n+1}

    端点值与端点导数（:math:`H'(0)=hf_n`、:math:`H'(1)=hf_{n+1}`）全部精确匹配，
    所以它对三次多项式是**恒等**的；一般光滑解上步内误差 :math:`O(h^4)`。
    """
    th = np.asarray(theta, dtype=float).reshape(-1, 1)
    a0 = np.asarray(y0, dtype=float).reshape(1, -1)
    a1 = np.asarray(y1, dtype=float).reshape(1, -1)
    b0 = np.asarray(f0, dtype=float).reshape(1, -1)
    b1 = np.asarray(f1, dtype=float).reshape(1, -1)
    h00 = 1.0 - 3.0 * th**2 + 2.0 * th**3
    h10 = th - 2.0 * th**2 + th**3
    h01 = 3.0 * th**2 - 2.0 * th**3
    h11 = -th**2 + th**3
    return h00 * a0 + (float(h) * h10) * b0 + h01 * a1 + (float(h) * h11) * b1


def dense_step(
    name: "str | Tableau",
    f: RHS,
    t: float,
    y: Any,
    h: float,
    theta: Any,
    *,
    f_start: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int, StepResult]:
    r"""走一步并给出步内稠密输出：``(t_dense, y_dense, n_extra_f_evals, step)``。

    右端点导数 :math:`f(t+h,y_{n+1})`：FSAL 方法已经在最后一级里算过（额外代价 **0**），
    其余方法要补一次求值——这一项在「工作-精度」里必须记进账。
    """
    step = rk_step(name, f, t, y, h, f_start=f_start)
    tab = as_tableau(name)
    f0 = step.stages[0]
    if step.fsal_ready:
        f1 = step.stages[-1]
        n_extra = 0
    else:
        f1 = np.asarray(f(t + h, step.y), dtype=float)
        n_extra = 1
    th = np.asarray(theta, dtype=float).reshape(-1)
    y_dense = hermite_dense(y, step.y, f0, f1, h, th)
    return float(t) + th * float(h), y_dense, n_extra, step


# --------------------------------------------------------------------------- #
# 5. 测试问题（解析解）
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Problem:
    r"""一个初值问题 :math:`y'=f(t,y)`、:math:`y(0)=y_0` 与它的**解析解**。

    解析解不是「好看的装饰」：本章全部误差数字都相对它测量。没有解析解就没有
    实测阶——这也是本章只选这四个问题的原因。
    """

    name: str
    label: str
    y0: tuple[float, ...]
    t_end: float
    rhs: RHS
    exact: Callable[[Any], np.ndarray]
    stiff: bool = False
    note: str = ""
    hs: tuple[float, ...] = ()

    def dim(self) -> int:
        return len(self.y0)

    def y0_array(self) -> np.ndarray:
        return np.asarray(self.y0, dtype=float)


def _exp_decay_rhs(t: float, y: np.ndarray) -> np.ndarray:  # noqa: ARG001 - 签名统一
    return -y


def _exp_decay_exact(t: Any) -> np.ndarray:
    tt = np.atleast_1d(np.asarray(t, dtype=float))
    return np.exp(-tt)[:, None]


def _logistic_rhs(t: float, y: np.ndarray) -> np.ndarray:  # noqa: ARG001
    return y * (1.0 - y)


def _logistic_exact(t: Any) -> np.ndarray:
    tt = np.atleast_1d(np.asarray(t, dtype=float))
    return (1.0 / (1.0 + np.exp(-tt)))[:, None]


def _oscillator_rhs(t: float, y: np.ndarray) -> np.ndarray:  # noqa: ARG001
    return np.array([y[1], -y[0]])


def _oscillator_exact(t: Any) -> np.ndarray:
    tt = np.atleast_1d(np.asarray(t, dtype=float))
    return np.stack([np.cos(tt), -np.sin(tt)], axis=-1)


def _stiff_cos_rhs_factory(lam: float) -> RHS:
    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        return np.array([lam * (y[0] - math.cos(t)) - math.sin(t)])

    return rhs


def _stiff_cos_exact_factory(lam: float, c0: float) -> Callable[[Any], np.ndarray]:
    def exact(t: Any) -> np.ndarray:
        tt = np.atleast_1d(np.asarray(t, dtype=float))
        return (np.cos(tt) + c0 * np.exp(lam * tt))[:, None]

    return exact


def _blowup_rhs(t: float, y: np.ndarray) -> np.ndarray:  # noqa: ARG001
    return y * y


def _blowup_exact(t: Any) -> np.ndarray:
    tt = np.atleast_1d(np.asarray(t, dtype=float))
    return (1.0 / (1.0 - tt))[:, None]


#: 测试问题集。``hs`` 是建议的步长阶梯（粗→细，供双对数拟合用）。
_PROBLEMS: tuple[Problem, ...] = (
    Problem(
        name="exp_decay",
        label="指数衰减 y' = −y",
        y0=(1.0,),
        t_end=2.0,
        rhs=_exp_decay_rhs,
        exact=_exp_decay_exact,
        note="最干净的一维问题：误差不增长、解量级 O(1)，适合量阶。",
        hs=(0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625),
    ),
    Problem(
        name="exp_decay_long",
        label="指数衰减（长时程 T=10）",
        y0=(1.0,),
        t_end=10.0,
        rhs=_exp_decay_rhs,
        exact=_exp_decay_exact,
        note="同一个 y'=−y 拉到 T=10：误差在长时程上累积，端点误差与整段最大误差能差三个数量级。",
        hs=(0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625),
    ),
    Problem(
        name="logistic",
        label="logistic 增长 y' = y(1−y)",
        y0=(0.5,),
        t_end=4.0,
        rhs=_logistic_rhs,
        exact=_logistic_exact,
        note="解从 0.5 单调爬到 0.982：误差在拐点附近最大，端点误差会**小于**整段最大误差。",
        hs=(0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625),
    ),
    Problem(
        name="oscillator",
        label="谐振子 y'' = −y",
        y0=(1.0, 0.0),
        t_end=6.283185307179586,
        rhs=_oscillator_rhs,
        exact=_oscillator_exact,
        note="二维、解在单位圆上：相位误差让整段最大误差远大于终点误差。",
        hs=(0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625),
    ),
    Problem(
        name="stiff_cos",
        label="刚性问题 y' = λ(y − cos t) − sin t（λ = −1000）",
        y0=(1.0,),
        t_end=1.0,
        rhs=_stiff_cos_rhs_factory(-1000.0),
        exact=_stiff_cos_exact_factory(-1000.0, 0.0),
        stiff=True,
        note=(
            "真解就是 cos t，导数的「慢」部分完全平缓；但显式方法一旦 h > 2/|λ| 就会被快模态吃掉。"
            "误差估计器看不见这件事（见 pitfalls）。"
        ),
        hs=(0.02, 0.01, 0.005, 0.0025, 0.00125, 0.000625),
    ),
    Problem(
        name="blowup",
        label="有限时刻爆破 y' = y²",
        y0=(1.0,),
        t_end=0.9,
        rhs=_blowup_rhs,
        exact=_blowup_exact,
        note="真解 1/(1−t) 在 t=1 爆破：光滑性假设全部失效，任何阶都无从谈起。",
        hs=(0.05, 0.025, 0.0125, 0.00625, 0.003125),
    ),
)


def problem_names() -> tuple[str, ...]:
    """全部测试问题名。"""
    return tuple(p.name for p in _PROBLEMS)


#: 问题名（供 manifest 的 enum 使用）。
PROBLEM_NAMES: tuple[str, ...] = problem_names()


def problem(name: str) -> Problem:
    """按名字取测试问题；名字不认识就抛 :class:`ValueError`。"""
    for p in _PROBLEMS:
        if p.name == name:
            return p
    raise ValueError(f"未知问题 {name!r}（可用：{list(PROBLEM_NAMES)}）")


def pointwise_error(y: np.ndarray, exact: np.ndarray) -> np.ndarray:
    """逐时刻的 :math:`\\|y-y_{exact}\\|_\\infty`（对分量取最大，兼容一维/二维状态）。"""
    diff = np.abs(np.asarray(y, dtype=float) - np.asarray(exact, dtype=float))
    if diff.ndim == 1:
        return diff
    return np.max(diff, axis=1)


# --------------------------------------------------------------------------- #
# 6. 定步长积分与误差阶梯
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Trajectory:
    """一次定步长积分的全部可观测物（含整段最大误差与终点误差）。"""

    method: str
    problem: str
    t: np.ndarray
    y: np.ndarray
    h: float
    n_steps: int
    n_f_evals: int
    max_error: float
    endpoint_error: float
    error_curve: np.ndarray
    ok: bool
    blowup: bool
    t_blowup: float

    def final(self) -> np.ndarray:
        arr = np.asarray(self.y, dtype=float)
        return arr[-1] if arr.ndim > 1 else np.asarray([arr[-1]], dtype=float)

    def work_per_time(self) -> float:
        """单位时间所需的 :math:`f` 求值次数（= 方法级数 / 步长，实测账）。"""
        span = float(self.t[-1] - self.t[0])
        return float(self.n_f_evals) / span if span > 0.0 else float("nan")


def integrate_fixed(
    method: "str | Tableau",
    f: RHS,
    t_span: tuple[float, float],
    y0: Any,
    h: float,
    *,
    exact: Callable[[Any], np.ndarray] | None = None,
    max_steps: int = 400000,
    label: str = "",
) -> Trajectory:
    r"""定步长积分（显式方法）：返回节点、函数求值次数与（有解析解时的）误差曲线。

    三条不撒谎的约定，与 :func:`numcore.ode.implicit.integrate` 同款：

    1. **爆炸显式停止**：:math:`\max|y|>` :data:`BLOWUP` 立刻收工，``blowup=True``、
       给出 ``t_blowup``，绝不让 ``inf``/``nan`` 流进帧；
    2. **末步截断**：最后一步缩到 ``t1``（步长不再均匀时如实记录 ``h`` 是名义步长）；
    3. **误差曲线是整段的**：``max_error`` 与 ``endpoint_error`` 同时给出——
       端点误差可以比整段最大误差小好几个数量级（本章 pitfalls 的第一条）。
    """
    t0, t1 = float(t_span[0]), float(t_span[1])
    if not t1 > t0:
        raise ValueError(f"需要 t1 > t0，得到 {t_span}")
    if not h > 0.0:
        raise ValueError(f"步长必须为正，得到 {h}")
    yy = np.asarray(y0, dtype=float).reshape(-1)
    ts: list[float] = [t0]
    ys: list[np.ndarray] = [yy.copy()]
    n_f = 0
    t = t0
    blowup = False
    t_blow = float("nan")
    while t < t1 - 1e-15 and len(ts) <= int(max_steps):
        hh = min(float(h), t1 - t)
        step = rk_step(method, f, t, yy, hh)
        n_f += step.n_f_evals
        yy = step.y
        t = t + hh
        ts.append(t)
        ys.append(yy.copy())
        if not np.all(np.isfinite(yy)) or float(np.max(np.abs(yy))) > BLOWUP:
            blowup = True
            t_blow = t
            break
    tt = np.asarray(ts, dtype=float)
    Y = np.asarray(ys, dtype=float)
    if exact is not None:
        curve = pointwise_error(Y, np.asarray(exact(tt), dtype=float))
        max_err = float(np.max(curve))
        end_err = float(curve[-1])
    else:
        curve = np.zeros(tt.size, dtype=float)
        max_err = float("nan")
        end_err = float("nan")
    return Trajectory(
        method=as_tableau(method).name,
        problem=label,
        t=tt,
        y=Y,
        h=float(h),
        n_steps=int(tt.size - 1),
        n_f_evals=int(n_f),
        max_error=max_err,
        endpoint_error=end_err,
        error_curve=curve,
        ok=not blowup,
        blowup=blowup,
        t_blowup=t_blow,
    )


def local_orders(h: Any, err: Any) -> np.ndarray:
    r"""相邻点的局部阶 :math:`\log(e_i/e_{i+1})/\log(h_i/h_{i+1})`（长度 ``n−1``）。

    本章不 import :mod:`numcore.verify`（分层规矩），这个六行函数就是
    :func:`numcore.verify.consecutive_orders` 的本地版；``tests/3.1`` 里有一条断言
    专门核对两者给出同一个数。全正输入才调用——含 0 的误差序列会触发除零警告，
    而仓库把警告当错误。
    """
    hh = np.asarray(h, dtype=float).reshape(-1)
    ee = np.asarray(err, dtype=float).reshape(-1)
    if hh.size != ee.size or hh.size < 2:
        raise ValueError(f"需要长度相同的两组数据（≥2），得到 {hh.size} 与 {ee.size}")
    with np.errstate(divide="ignore", invalid="ignore"):
        num = np.log(ee[:-1] / ee[1:])
        den = np.log(hh[:-1] / hh[1:])
        return num / den


def asymptotic_window(
    h: Any,
    err: Any,
    theory: float,
    *,
    loose: float = 0.2,
    floor: float | None = None,
    min_points: int = 3,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    r"""按**局部阶**自动裁剪渐近窗口：返回 ``(h_win, err_win, 丢头点数, 丢尾点数)``。

    规则（写死，不随数据调）：
    1. 丢掉前缀，直到所有剩余点的相邻局部阶 :math:`\log(e_i/e_{i+1})/\log(h_i/h_{i+1})`
       都落在 ``theory ± loose``——那是"还没进渐近区"的粗网格；
    2. 再丢掉误差低于 ``floor``（缺省 :math:`200u`，:math:`u` 是单位舍入）的尾巴——
       那是舍入地板，那里斜率是噪声；
    3. 剩余点少于 ``min_points`` 就抛错：**点不够时的斜率是编的**。

    为什么值得写成一个函数：同一个窗口规则被测试、断言表与 viz 三处共用，
    谁都没机会"为了让斜率好看"临时挪窗口。实测教训：rkf45 在 exp_decay 上
    全窗口斜率是 4.16（预渐近的 :math:`h^5` 项还很大），裁剪后才回到 4.0 附近。
    """
    hh = np.asarray(h, dtype=float).reshape(-1)
    ee = np.asarray(err, dtype=float).reshape(-1)
    if hh.size != ee.size:
        raise ValueError(f"长度不同：{hh.size} vs {ee.size}")
    keep = np.isfinite(hh) & np.isfinite(ee) & (hh > 0.0) & (ee > 0.0)
    hh, ee = hh[keep], ee[keep]
    if hh.size < min_points:
        raise ValueError(f"有效点不足：{hh.size} < {min_points}")
    fl = float(200.0 * (np.finfo(float).eps / 2.0) if floor is None else floor)
    local = local_orders(hh, ee)
    k = 0
    while k < local.size and abs(float(local[k]) - float(theory)) > float(loose):
        k += 1
    m = hh.size
    while m - 1 > k + min_points - 1 and ee[m - 1] < fl:
        m -= 1
    if m - k < min_points:
        raise ValueError(f"渐近窗口不足 {min_points} 点（丢头 {k}、丢尾 {hh.size - m}）")
    return hh[k:m], ee[k:m], k, hh.size - m


def loglog_slope(h: Any, err: Any) -> float:
    """双对数最小二乘斜率 :math:`p`（:math:`\\mathrm{err}\\approx Ch^p`）。

    与 :func:`numcore.verify.fit_order` 同一口径（先取对数再做一元最小二乘），
    这样「帧里画出来的斜率」与「断言里拟合的斜率」是同一个数。
    只接受正数点；非正或非有限的点会被丢弃。
    """
    hh = np.asarray(h, dtype=float).ravel()
    ee = np.asarray(err, dtype=float).ravel()
    if hh.size != ee.size:
        raise ValueError(f"h 与 err 长度不同：{hh.size} vs {ee.size}")
    keep = np.isfinite(hh) & np.isfinite(ee) & (hh > 0) & (ee > 0)
    if int(np.count_nonzero(keep)) < 2:
        raise ValueError("有效点不足 2 个，无法拟合幂律斜率")
    x = np.log(hh[keep])
    y = np.log(ee[keep])
    xb = float(np.mean(x))
    yb = float(np.mean(y))
    sxx = float(np.sum((x - xb) ** 2))
    if sxx == 0.0:
        raise ValueError("所有 h 相同，无法拟合幂律斜率")
    return float(np.sum((x - xb) * (y - yb)) / sxx)


@dataclass(frozen=True)
class Ladder:
    r"""误差阶梯：同一问题的同一方法在一串步长上的**全局 / 终点 / 局部 / 插值**误差。

    四条曲线放在一起才看得出本章的核心事实：

    * ``error_global`` :math:`\\sim h^p`（阶 :math:`p`）；
    * ``error_endpoint`` 同阶但常数可能小几个量级（**别拿它当误差**）；
    * ``error_local`` :math:`\\sim h^{p+1}`（局部比全局高一阶）；
    * ``error_dense`` :math:`\\sim h^{\\min(4,p+1)}`（稠密输出可能比方法低阶）。
    """

    problem: str
    method: str
    order: int
    h: np.ndarray
    error_global: np.ndarray
    error_endpoint: np.ndarray
    error_local: np.ndarray
    error_dense: np.ndarray
    work: np.ndarray
    n_f_evals: np.ndarray

    def slopes(self) -> dict[str, float]:
        """四条曲线的实测幂律斜率（对数-对数最小二乘）。"""
        out: dict[str, float] = {}
        for key, arr in (
            ("global", self.error_global),
            ("endpoint", self.error_endpoint),
            ("local", self.error_local),
            ("dense", self.error_dense),
        ):
            try:
                out[key] = loglog_slope(self.h, arr)
            except ValueError:
                out[key] = float("nan")
        return out


def estimate_local_error(
    method: "str | Tableau",
    f: RHS,
    t: float,
    y: Any,
    h: float,
    *,
    n_sub: int = 8,
) -> np.ndarray:
    r"""一步的**真局部误差**估计：与同一方法的 ``n_sub`` 个子步之差的 :math:`\infty` 范数。

    为什么不用解析解：局部误差是「这一步相对流动的解」的偏差，用解析解去量会把
    前面积累的误差也算进来。子步参考解自身的误差是 :math:`O((h/n)^{p})`，
    取 :math:`n=8` 时比被测量的量小 :math:`8^{p}` 倍（四阶方法约 4000 倍），够用；
    这一约定写进 docstring，免得被读成「精确局部误差」。
    """
    n = int(n_sub)
    if n < 2:
        raise ValueError(f"子步数至少 2，得到 {n_sub}")
    yy = np.asarray(y, dtype=float).reshape(-1)
    ref = yy.copy()
    hh = float(h) / n
    for k in range(n):
        ref = rk_step(method, f, float(t) + k * hh, ref, hh).y
    step = rk_step(method, f, t, yy, h)
    return np.abs(step.y - ref)


def step_error_reference(
    method: "str | Tableau",
    f: RHS,
    t: float,
    y: Any,
    h: float,
    *,
    n_sub: int = 8,
) -> float:
    """一步真局部误差的标量值 :math:`\\|y_1-y_{ref}\\|_\\infty`。"""
    return float(np.max(estimate_local_error(method, f, t, y, h, n_sub=n_sub)))


def local_error_ladder(
    problem_name: str,
    method: "str | Tableau",
    hs: Sequence[float] | None = None,
    *,
    n_sub: int = 8,
    t0: float = 0.0,
) -> Ladder:
    """局部误差阶梯：每个步长只走**一步**，量相对子步参考解的真局部误差。"""
    prob = problem(problem_name)
    grid = np.asarray(list(hs) if hs is not None else prob.hs, dtype=float)
    n = grid.size
    err_local = np.zeros(n, dtype=float)
    work = np.zeros(n, dtype=float)
    n_f = np.zeros(n, dtype=float)
    y0 = prob.y0_array()
    for k, h in enumerate(grid):
        err_local[k] = step_error_reference(method, prob.rhs, t0, y0, float(h), n_sub=n_sub)
        n_f[k] = float(as_tableau(method).stages * n_sub + as_tableau(method).stages)
        work[k] = n_f[k] / float(h)
    nan = np.full(n, float("nan"))
    return Ladder(
        problem=problem_name,
        method=as_tableau(method).name,
        order=as_tableau(method).order,
        h=grid,
        error_global=nan.copy(),
        error_endpoint=nan.copy(),
        error_local=err_local,
        error_dense=nan.copy(),
        work=work,
        n_f_evals=n_f,
    )


def dense_error_ladder(
    problem_name: str,
    method: "str | Tableau",
    hs: Sequence[float] | None = None,
    *,
    n_theta: int = 9,
    t_end: float | None = None,
) -> Ladder:
    r"""稠密输出误差阶梯：整段积分中**步内**（不含端点）插值误差的最大值。

    :math:`\theta` 取 ``n_theta`` 个内点（缺省 9 个，含 0.5），端点误差另算——
    「插值在端点当然准」这件事没有信息量，必须看步内。
    """
    prob = problem(problem_name)
    grid = np.asarray(list(hs) if hs is not None else prob.hs, dtype=float)
    t1 = float(prob.t_end if t_end is None else t_end)
    theta = np.linspace(0.0, 1.0, int(n_theta) + 2)[1:-1]
    n = grid.size
    err_dense = np.zeros(n, dtype=float)
    work = np.zeros(n, dtype=float)
    n_f = np.zeros(n, dtype=float)
    for k, h in enumerate(grid):
        traj = integrate_fixed(method, prob.rhs, (0.0, t1), prob.y0_array(), float(h), exact=prob.exact)
        worst = 0.0
        extra = 0
        for i in range(traj.t.size - 1):
            t_d, y_d, n_extra, _ = dense_step(
                method, prob.rhs, float(traj.t[i]), traj.y[i], float(traj.t[i + 1] - traj.t[i]), theta
            )
            extra += n_extra
            ref = np.asarray(prob.exact(t_d), dtype=float)
            worst = max(worst, float(np.max(pointwise_error(y_d, ref))))
        err_dense[k] = worst
        n_f[k] = float(traj.n_f_evals + extra)
        work[k] = n_f[k] / t1
    nan = np.full(n, float("nan"))
    return Ladder(
        problem=problem_name,
        method=as_tableau(method).name,
        order=as_tableau(method).order,
        h=grid,
        error_global=nan.copy(),
        error_endpoint=nan.copy(),
        error_local=nan.copy(),
        error_dense=err_dense,
        work=work,
        n_f_evals=n_f,
    )


def error_ladder(
    problem_name: str,
    method: "str | Tableau",
    hs: Sequence[float] | None = None,
    *,
    n_sub: int = 8,
    n_theta: int = 9,
    t_end: float | None = None,
) -> Ladder:
    r"""完整的四条误差曲线（全局 / 终点 / 局部 / 稠密）在同一条步长阶梯上。

    一次调用就把本章 viz 1 需要的全部量算齐：同一串 ``h`` 上的
    :math:`\\max_t|y_h-y|`、:math:`|y_h(T)-y(T)|`、首步真局部误差与步内插值误差。
    """
    prob = problem(problem_name)
    grid = np.asarray(list(hs) if hs is not None else prob.hs, dtype=float)
    t1 = float(prob.t_end if t_end is None else t_end)
    theta = np.linspace(0.0, 1.0, int(n_theta) + 2)[1:-1]
    n = grid.size
    eg = np.zeros(n, dtype=float)
    ee = np.zeros(n, dtype=float)
    el = np.zeros(n, dtype=float)
    ed = np.zeros(n, dtype=float)
    work = np.zeros(n, dtype=float)
    n_f = np.zeros(n, dtype=float)
    y0 = prob.y0_array()
    for k, h in enumerate(grid):
        hh = float(h)
        traj = integrate_fixed(method, prob.rhs, (0.0, t1), y0, hh, exact=prob.exact)
        eg[k] = traj.max_error
        ee[k] = traj.endpoint_error
        el[k] = step_error_reference(method, prob.rhs, 0.0, y0, hh, n_sub=n_sub)
        worst = 0.0
        extra = 0
        for i in range(traj.t.size - 1):
            t_d, y_d, n_extra, _ = dense_step(
                method, prob.rhs, float(traj.t[i]), traj.y[i], float(traj.t[i + 1] - traj.t[i]), theta
            )
            extra += n_extra
            worst = max(worst, float(np.max(pointwise_error(y_d, np.asarray(prob.exact(t_d), dtype=float)))))
        ed[k] = worst
        n_f[k] = float(traj.n_f_evals + extra + as_tableau(method).stages * n_sub)
        work[k] = n_f[k] / t1
    return Ladder(
        problem=problem_name,
        method=as_tableau(method).name,
        order=as_tableau(method).order,
        h=grid,
        error_global=eg,
        error_endpoint=ee,
        error_local=el,
        error_dense=ed,
        work=work,
        n_f_evals=n_f,
    )


# --------------------------------------------------------------------------- #
# 7. 稳定多项式与负实轴边界（3.1 只用到"临界步长"这一个推论）
# --------------------------------------------------------------------------- #
def stability_polynomial(name: str) -> np.ndarray:
    r"""显式 RK 的稳定多项式**系数**（升幂）：:math:`R(z)=\sum_j c_jz^j`、:math:`c_j=b^{\mathsf T}A^{j-1}e`。

    它和阶条件是同一个和：方法 :math:`p` 阶时 :math:`c_j=1/j!`（:math:`j\le p`），
    于是"阶"这件事在复平面上留下指纹。3.2 章从 :math:`R(z)` 出发谈稳定域，
    本章只用它回答一个问题：**多小的步长才不炸**（刚性算例的临界步长）。
    """
    tab = tableau(name)
    A = tab.A_array()
    b = tab.b_array()
    s = tab.stages
    coeffs = np.zeros(s + 1, dtype=float)
    coeffs[0] = 1.0
    u = np.ones(s, dtype=float)
    for j in range(1, s + 1):
        coeffs[j] = float(b @ u)
        u = A @ u
    return coeffs


def evaluate_polynomial(coeffs: Any, x: Any) -> np.ndarray:
    r"""按升幂系数用自写 Horner 求值（``np.polyval`` 属 P3 黑名单）。"""
    c = np.asarray(coeffs, dtype=float).reshape(-1)
    xx = np.asarray(x, dtype=float)
    out = np.full(xx.shape, c[-1], dtype=float)
    for j in range(c.size - 2, -1, -1):
        out = out * xx + c[j]
    return out


def negative_real_axis_endpoint(name: str, *, n_scan: int = 20001, x_lo: float = -60.0) -> float:
    r"""绝对稳定域与**负实轴**的左端点 :math:`x^\*`（:math:`|R(x^\*)|=1`、:math:`x^\*<0`）。

    算法：从 :math:`x=0`（显式方法总有 :math:`R(0)=1`，故 0 在稳定域内）向左细扫，
    找到第一个 :math:`|R|>1` 的格点，再在该格点与上一个稳定格点之间二分。
    返回 :math:`x^\*`；若整个扫描区间都稳定，返回 ``x_lo`` 并**由调用方**判断是否可信。

    为什么不能用"一步就炸"来实测：:math:`|R|>1` 只是"每步放大"，放大到看得见需要
    时间——有限观测时长给出的临界步长会**偏大**（实测 T=20 时偏 19%），
    这条偏差本身是本章 pitfalls 的一条。
    """
    coeffs = stability_polynomial(name)
    grid = np.linspace(0.0, float(x_lo), int(n_scan))
    vals = np.abs(evaluate_polynomial(coeffs, grid))
    bad = np.flatnonzero(vals > 1.0)
    if bad.size == 0:
        return float(x_lo)
    k = int(bad[0])
    lo = float(grid[k - 1])  # 稳定端（更靠近 0）
    hi = float(grid[k])  # 不稳定端
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if float(np.abs(evaluate_polynomial(coeffs, np.array([mid]))[0])) <= 1.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def critical_step_for_decay(name: str, lam: float) -> float:
    r"""刚性衰减率 :math:`\lambda<0` 下显式方法的**临界步长** :math:`h^\*=x^\*/|\lambda|`。

    :math:`z=\lambda h` 必须落在绝对稳定域里，于是 :math:`h\le|\lambda|^{-1}` 乘上
    负实轴端点。这一个数就是 3.1 与 3.2 的接缝：方法族在本章，域在 3.2 章，
    而"刚性问题被迫走多小的步"由两者相乘决定。
    """
    if not lam < 0.0:
        raise ValueError(f"需要衰减率 λ<0，得到 {lam}")
    return negative_real_axis_endpoint(name) / float(lam)
