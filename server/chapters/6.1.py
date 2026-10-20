r"""6.1 章：二维综合仿真（4 个可视化）。

这一章把前面各章的零件装成三个完整求解器，并且只讲一件事：**"能出图"与"算对了"之间
还差一整套证据**。四个可视化的分工：

1. ``cavity_vorticity`` —— 顶盖驱动方腔的稳态涡量场与流线。Re 与网格是参数；
   副轴是耦合迭代的残差历史。侧栏同时给出"残差 1e-7"与"中心线 RMSE 0.2"——
   前者量离散方程，后者量离散误差，两者不能互相替代。
2. ``ghia_benchmark`` —— Ghia--Ghia--Shin (1982) 公开基准：竖直中心线 u(y) 与
   水平中心线 v(x) 逐点对拍，副轴是 RMSE 随网格 n 的收敛曲线。默认 Re=100；
   Re=1000 的粗网格上误差会明显变大，页面不会把它藏起来。
3. ``turing_pattern`` —— Gray--Scott 反应扩散。主图是斑图随显式步数的演化，
   副轴是五点模板的色散误差随 n 的收敛，右上是解析色散关系 λ(φ)。
   ``turing`` 预设的 λ*>0 是真图灵；``spots/stripes/waves`` 的最快模都在 φ=0
   （spots/waves 无稳态、stripes 底态均匀失稳），**没有正波数图灵带**。
4. ``dam_break`` —— 一维干底溃坝 vs Ritter 精确解。主图是水深剖面，
   副轴是逐点误差；侧栏报扇内/全域 L1、干底质量漂移（地板注入的水）与 CFL。

帧只做搬运：所有求解、误差与诊断都来自 :mod:`numcore.apps`。参数默认值按"能在
1 秒内出一帧"选取；测试与笔记里另跑更细的网格/更长的推进。
"""

from __future__ import annotations

import functools
import math
from typing import Any, Mapping

import numpy as np

from numcore.apps import cavity, reaction_diffusion as rd, shallow_water as sw
from server.nlf import Frame

__all__ = ["MANIFEST", "compute"]

CHAPTER = "6.1"

#: 对数轴/误差图的地板（与前端 scene 的 floor 一致）。
FLOOR = 1.0e-14

#: 方腔耦合迭代：tol=1e-7 足够让"残差 vs 离散误差"的对比成立，又不至于太慢。
_CAVITY_TOL = 1.0e-7
_CAVITY_MAX_SWEEPS = 12000

#: Ghia 网格收敛曲线用的四档网格（也是本章断言里用的那组）。
_GHIA_NS = (17, 25, 33, 49)

#: 浅水默认推进到 t=0.06：此时干锋在 x≈0.88，扇区与干区同时留在域内。
_DAM_T_MAX = 0.06

_PITFALLS = {
    "cavity_vorticity": (
        "残差小不等于解准：这里的残差只量『离离散方程有多远』，而中心线 RMSE 量的是"
        "『离 Ghia 基准有多远』。$n=41$、$\\mathrm{Re}=1000$ 时残差降到 $10^{-7}$，RMSE 仍有 0.2 量级——"
        "粗网格一阶迎风的数值扩散把主涡位置拖偏了。第二件事：网格与时间步的限制来自"
        "两个机制，$\\Delta t_{\\mathrm{diffusion}}\\sim\\mathrm{Re}\\cdot h^2$、"
        "$\\Delta t_{\\mathrm{CFL}}\\sim h/|u|$，它们的交叉点在 $\\mathrm{Re}_x=4c_{\\mathrm{cfl}}(n-1)$；"
        "本页 $\\mathrm{Re}=1000$ 已经在 $\\mathrm{CFL}$ 一侧，所以『$\\mathrm{Re}$ 越大越难算』这句话在交叉点两侧是反的。"
        "第三：中心差分的格点雷诺数 $\\mathrm{Re}_h=\\mathrm{Re}|u|h$ 超过 $2$ 时会失去对角占优，"
        "$n=41$、$\\mathrm{Re}=1000$ 的 $\\mathrm{Re}_h\\approx25$，换二阶中心格式会先失稳而不是先变准。"
    ),
    "ghia_benchmark": (
        "对拍要先把『基准本身的抄写分歧』算进去：Ghia 表在 $\\mathrm{Re}=1000$ 的 $v$ 剖面上，"
        "不同转载在 $x=0.9063$ 处就有 $-0.51550$ 与 $-0.51500$ 两种写法，跨转载最大分歧"
        "$10^{-2}$ 量级。把容差压到 $10^{-3}$ 的『完美对拍』量的是抄写版本，不是数值误差。"
        "第二：插值方向要固定——把基准插到网格上会把离散误差抹平，本页统一把数值解"
        "插到 Ghia 原表节点。第三：$\\mathrm{Re}=1000$ 的 RMSE 在 $n=49$ 仍有 0.2 左右，这不是"
        "代码错，而是粗网格一阶迎风的离散误差；$R=100$ 的同一套代码能到 0.006。"
        "第四：网格收敛阶要用同一窗口与同一范数报，换窗口能把阶改掉。"
    ),
    "turing_pattern": (
        "经典 Gray--Scott 斑图不是图灵斑图：spots 没有非平凡稳态，最快模 $\\lambda^\\ast=-0.035$ 在 "
        "$\\phi=0$；stripes 的底态存在但 $\\lambda(0)=+0.0054>0$，均匀模先失稳；waves 的 $\\lambda^\\ast=-0.014$ 也在 "
        "$\\phi=0$。三者都没有正波数增长带；真正的图灵预设 $\\lambda^\\ast=+0.015$ 且 $\\phi^\\ast=53.9>0$。判据必须"
        "同时满足『底态无扩散稳定』与『正波数 $\\lambda^\\ast>0$』两条：只看图案长得好看会把整类"
        "自复制斑图误认成图灵失稳。第二：$\\lambda^\\ast>0$ 也不一定是图灵——最大值在 $\\phi=0$ 时，"
        "那是整片均匀（同相）失稳。第三：显式稳定性上限只看最大扩散系数，"
        "$\\Delta t$ 不能超过 $\\Delta x^2/(4\\max(D_u,D_v))$ 的一半；turing 预设请求 $\\Delta t=0.25$ 会被夹到 0.095，"
        "帧里如实标出 dt_clamped。第四：域长固定时允许波数 $\\phi=2\\pi m$ 与 $n$ 无关，"
        "加密网格只抬高截止；真正按 $O(\\Delta x^2)$ 收敛的是五点模板的有效波数与增长率。"
    ),
    "dam_break": (
        "对拍窗口决定成败：干底 Ritter 解只在扇区 $-c_L\\le\\xi\\le2c_L$ 内光滑，"
        "干锋与头部折点都在窗口外。把整个域喂进 L1 误差，量到的主要是干锋位置误差；"
        "本页把扇内、全域两个数并排报，且不把『全域更差』说成格式精度。"
        "第二：干底的质量漂移本来就不为零——正性地板每步至少注入 $\\text{min_depth}\\cdot\\Delta x$ 的水，"
        "拿它当『守恒性证据』是假证据；真正可测的是湿域/周期域的守恒。"
        "第三：$\\mathrm{CFL}$ 不是抄来的常数，本页记录实际 max_cfl 与步数；$\\mathrm{cfl}=0.8$ 稳，"
        "但把 $\\mathrm{cfl}$ 推到 2 会出非有限值，页面不替它兜底。第四：Rusanov 的波速上界"
        "让通量失去逐点单调的直觉，别把『看起来单调』当定理。"
    ),
}


def _viz(**kwargs: Any) -> dict[str, Any]:
    return kwargs


MANIFEST: dict[str, Any] = {
    "chapter": CHAPTER,
    "title": "二维综合仿真",
    "part": "VI",
    "summary": (
        "把前五部分的零件装成三个完整求解器：顶盖驱动方腔的流函数--涡量、"
        "Gray--Scott 反应扩散的图灵判据、浅水方程的干底溃坝。每个应用都配公开基准、"
        "守恒/收敛证据与失效边界，尤其是『残差小≠解准』『经典斑图≠图灵』"
        "『干底守恒漂移≠格式不守恒』这三类最容易读错的量。"
    ),
    "vizzes": [
        _viz(
            id="cavity_vorticity",
            title="顶盖驱动方腔：涡量、流线与残差",
            description=(
                "单位方腔顶盖以 $U=1$ 拖动。流函数--涡量形式：DST 直接解 Poisson，"
                "红黑 SOR 扫涡量方程。主图是稳态涡量与流函数等值线，副轴是耦合迭代的"
                "相对残差历史；右侧报 $\\mathrm{Re}$、网格、主涡位置、与 Ghia 基准的 RMSE、"
                "以及扩散/$\\mathrm{CFL}$ 两条时间步限制。"
            ),
            dataSource="backend",
            params=[
                {
                    "name": "re",
                    "label": "雷诺数 Re",
                    "type": "enum",
                    "options": [100, 400, 1000],
                    "default": 1000,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "n",
                    "label": "每边节点数 n",
                    "type": "enum",
                    "options": [25, 33, 41, 49],
                    "default": 41,
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            time={"mode": "none"},
            requiredAux=["residual"],
            axes=[
                {"id": "main", "autoscaled": False, "title": "稳态涡量与流函数等值线", "ylabel": "y"},
                {"id": "residual", "autoscaled": True, "scale": "log", "title": "耦合迭代残差", "ylabel": "相对残差"},
            ],
            pitfalls=_PITFALLS["cavity_vorticity"],
            theory=[
                "流函数--涡量：$\\Delta\\psi=-\\omega$，$\\omega_t+u\\cdot\\nabla\\omega=\\mathrm{Re}^{-1}\\Delta\\omega$；壁面涡量用 Thom 公式",
                "DST 直接解 Poisson：$\\Delta$ 在 Dirichlet 下的特征值 $-(4/h^2)(\\sin^2(i\\pi/2m)+\\sin^2(j\\pi/2m))$",
                "残差只量离散方程；中心线剖面与 Ghia 基准的差是离散误差，二者不能互替",
            ],
        ),
        _viz(
            id="ghia_benchmark",
            title="Ghia 基准：中心线剖面与网格收敛",
            description=(
                "把数值解插值到 Ghia--Ghia--Shin (1982) 的原表节点上，分别对拍竖直中心线"
                "$u(y)$ 与水平中心线 $v(x)$。副轴是 RMSE 随网格 $n$ 的收敛曲线；"
                "侧栏固定报出基准抄写分歧（不得低于 $10^{-2}$）与主涡位置。"
            ),
            dataSource="backend",
            params=[
                {
                    "name": "re",
                    "label": "雷诺数 Re",
                    "type": "enum",
                    "options": [100, 400, 1000],
                    "default": 100,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "n",
                    "label": "主剖面的网格 n",
                    "type": "enum",
                    "options": [25, 33, 49, 65],
                    "default": 49,
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            time={"mode": "none"},
            requiredAux=["error"],
            axes=[
                {"id": "main", "autoscaled": False, "title": "竖直中心线 u(y) vs Ghia", "ylabel": "y"},
                {"id": "error", "autoscaled": True, "scale": "log", "title": "RMSE 随网格 n 的收敛", "ylabel": "RMSE"},
                {"id": "vprofile", "autoscaled": True, "title": "水平中心线 v(x) vs Ghia", "ylabel": "v"},
            ],
            pitfalls=_PITFALLS["ghia_benchmark"],
            theory=[
                "Ghia 表是公开基准，不是真值；转载分歧 $10^{-2}$ 量级，容差不能低于它",
                "数值解插值到基准节点，而不是把基准插到数值网格上（后者抹平离散误差）",
                "一阶迎风的网格收敛阶接近 1，但 $\\mathrm{Re}=1000$ 在粗网格上被数值扩散主导",
            ],
        ),
        _viz(
            id="turing_pattern",
            title="Gray--Scott：图灵斑图、色散关系与网格误差",
            description=(
                "单位正方形周期域上的 Gray--Scott 系统，显式 FTCS 从中心种子出发。"
                "主图是 $v(x,y)$ 随步数的演化；右上是解析色散关系 $\\lambda(\\phi)$ 与最不稳定波数；"
                "副轴是五点模板增长率误差随 $n$ 的收敛。turing 预设是真图灵，"
                "spots/stripes/waves 的最快模都在 $\\phi=0$，各缺一条判据，是对照组。"
            ),
            dataSource="backend",
            params=[
                {
                    "name": "preset",
                    "label": "参数预设",
                    "type": "enum",
                    "options": ["turing", "spots", "stripes", "waves", "uniform"],
                    "default": "turing",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "n",
                    "label": "每边格点数 n",
                    "type": "enum",
                    "options": [64, 96, 128],
                    "default": 128,
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            time={"mode": "scrub", "tMax": 10000.0, "dt": 2000.0, "label": "显式步数"},
            requiredAux=["error"],
            axes=[
                {"id": "main", "autoscaled": False, "title": "v(x,y)：Gray--Scott 斑图", "ylabel": "y"},
                {"id": "error", "autoscaled": True, "scale": "log", "title": "五点模板增长率误差 vs n", "ylabel": "相对误差"},
                {"id": "dispersion", "autoscaled": True, "title": "解析色散关系 λ(φ)", "ylabel": "λ_max"},
            ],
            pitfalls=_PITFALLS["turing_pattern"],
            theory=[
                "Gray--Scott：$u_t=D_u\\Delta u-uv^2+F(1-u)$，$v_t=D_v\\Delta v+uv^2-(F+k)v$（周期边界）",
                "图灵判据：底态无扩散稳定，且 $M(\\phi)=J-\\phi^2\\operatorname{diag}(D_u,D_v)$ 的最大特征值 $\\lambda(\\phi)$ 在某 $\\phi^\\ast>0$ 变正",
                "显式 FTCS 稳定上限 $\\Delta t\\le\\Delta x^2/(4\\max(D_u,D_v))$；五点模板有效波数 $(2/\\Delta x)\\sin(\\phi\\Delta x/2)$",
            ],
        ),
        _viz(
            id="dam_break",
            title="干底溃坝：Ritter 精确解、误差窗口与 CFL",
            description=(
                "一维浅水方程，Rusanov 通量 + 显式推进。初始为 $x=0.5$ 左侧 $h=1$、右侧干底。"
                "主图是水深剖面与 Ritter 精确解，副轴是逐点绝对误差（对数）；"
                "时间轴是物理时间。侧栏把扇内 $L^1$、全域 $L^1$、质量漂移与 $\\mathrm{CFL}$ 分开报。"
            ),
            dataSource="backend",
            params=[
                {
                    "name": "n",
                    "label": "格数 n",
                    "type": "enum",
                    "options": [100, 200, 400],
                    "default": 200,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "cfl",
                    "label": "目标 CFL",
                    "type": "float",
                    "min": 0.3,
                    "max": 0.9,
                    "step": 0.1,
                    "default": 0.8,
                    "critical": [1.0],
                    "criticalNote": "$\\mathrm{CFL}>1$ 不保证稳定；本模块的实测临界值在 $1.2$–$1.5$ 之间，但页面只允许到 $0.9$。",
                },
            ],
            time={"mode": "scrub", "tMax": _DAM_T_MAX, "dt": 0.01, "label": "物理时间 t"},
            requiredAux=["error"],
            axes=[
                {"id": "main", "autoscaled": False, "title": "水深 h(x) 与 Ritter 精确解", "ylabel": "h"},
                {"id": "error", "autoscaled": True, "scale": "log", "title": "|h_num - h_exact|（逐点）", "ylabel": "绝对误差"},
            ],
            pitfalls=_PITFALLS["dam_break"],
            theory=[
                "Ritter 干底解：$h=(2c_L-\\xi)^2/(9g)$、$u=(2/3)(\\xi+c_L)$（$-c_L<\\xi<2c_L$，$\\xi=(x-x_0)/t$）",
                "Rusanov 通量：$F=\\frac{1}{2}(F_L+F_R)-\\frac{1}{2}a(U_R-U_L)$，$a=\\max(|u|+\\sqrt{gh})$",
                "干底质量漂移来自正性地板 $\\text{min_depth}\\cdot\\Delta x$ 的注入，不是格式不守恒",
            ],
        ),
    ],
}


# --------------------------------------------------------------------------- #
# 缓存：帧预计算与前端拖动共用同一份结果
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=24)
def _cavity_case(n: int, re: float) -> cavity.SteadyResult:
    """稳态方腔（按 n/Re 缓存；一张 49×49 的场只占几十 KB）。"""
    return cavity.steady_solve(
        int(n),
        float(re),
        scheme="upwind",
        max_sweeps=_CAVITY_MAX_SWEEPS,
        tol=_CAVITY_TOL,
        record_every=1,
    )


@functools.lru_cache(maxsize=8)
def _ghia_curve(re: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Ghia RMSE 网格收敛曲线：``(n, rmse, rmse_u, rmse_v)``。"""
    ns = np.asarray(_GHIA_NS, dtype=float)
    rmse = np.empty(ns.size, dtype=float)
    ru = np.empty(ns.size, dtype=float)
    rv = np.empty(ns.size, dtype=float)
    for i, n in enumerate(_GHIA_NS):
        res = _cavity_case(int(n), float(re))
        gr = cavity.cavity_grid(int(n))
        cmp = cavity.compare_with_benchmark(res.u, res.v, gr, float(re))
        rmse[i] = cmp.rmse
        ru[i] = cmp.rmse_u
        rv[i] = cmp.rmse_v
    return ns, rmse, ru, rv


@functools.lru_cache(maxsize=24)
def _rd_case(preset: str, n: int, steps: int) -> tuple[rd.RDResult, np.ndarray]:
    """Gray--Scott 推进到 ``steps`` 步；返回 ``(结果, 初始 u)``。"""
    du, dv, feed, kill = rd.RD_PRESETS[str(preset)]
    kind = {"turing": "spots", "spots": "spots", "stripes": "stripes", "waves": "waves"}.get(str(preset), "uniform")
    if str(preset) == "uniform":
        u0 = np.ones((int(n), int(n)), dtype=float)
        v0 = np.zeros((int(n), int(n)), dtype=float)
    else:
        u0, v0 = rd.initial_condition(int(n), kind=kind, seed=20261019)
    params = rd.RDParams(n=int(n), du=du, dv=dv, feed=feed, kill=kill, dt=0.25, steps=int(steps))
    res = rd.gray_scott_solve(params, u0=u0, v0=v0, seed=20261019)
    return res, u0


@functools.lru_cache(maxsize=16)
def _swe_case(n: int, cfl: float, t_end: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, sw.SWEResult]:
    """干底溃坝推进到 ``t_end``（含初始剖面）。"""
    x, h0, hu0 = sw.dam_break_1d(int(n), right=0.0)
    params = sw.SWEParams(n=int(n), length=1.0, t_end=float(t_end), cfl=float(cfl), g=sw.G_DEFAULT, scheme="rusanov")
    res = sw.swe_solve_1d(params, h0=h0, hu0=hu0, bc="outflow")
    return x, h0, hu0, res


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def _finite(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float(default)
    return x if math.isfinite(x) else float(default)


def _floor_error(values: Any, floor: float = FLOOR) -> tuple[np.ndarray, np.ndarray]:
    """对数误差：返回 ``(max(|e|, floor), valid)``；invalid 表示该点不是真误差。"""
    v = np.abs(np.asarray(values, dtype=float))
    valid = np.isfinite(v) & (v > floor)
    out = np.where(valid, v, floor)
    return np.asarray(out, dtype=float), valid.astype(float)


def _log_slope(x: Any, y: Any) -> float | None:
    """双对数最小二乘斜率（自己写，不引入成品的 polyfit）。"""
    xa = np.log(np.asarray(x, dtype=float))
    ya = np.log(np.asarray(y, dtype=float))
    ok = np.isfinite(xa) & np.isfinite(ya)
    if np.count_nonzero(ok) < 2:
        return None
    xa, ya = xa[ok], ya[ok]
    xm = float(np.mean(xa))
    ym = float(np.mean(ya))
    den = float(np.sum((xa - xm) ** 2))
    if den <= 0.0:
        return None
    slope = float(np.sum((xa - xm) * (ya - ym)) / den)
    return slope if math.isfinite(slope) else None


def _ghia_centers(re: int) -> dict[str, tuple[float, float]]:
    return cavity.ghia_vortex_centers(int(re))


# --------------------------------------------------------------------------- #
# viz 1：方腔涡量
# --------------------------------------------------------------------------- #
def _cavity_vorticity(params: Mapping[str, Any]) -> Frame:
    re = int(params["re"])
    n = int(params["n"])
    res = _cavity_case(n, re)
    gr = cavity.cavity_grid(n)
    cmp = cavity.compare_with_benchmark(res.u, res.v, gr, float(re))
    centers = cavity.vortex_centers(res.psi, gr, threshold=1.0e-4)
    primary = centers[0] if centers else None
    dt = cavity.dt_limits(n, float(re), cfl=0.8)
    cell_re = cavity.cell_reynolds(float(re), n)
    threshold_re = cavity.central_scheme_threshold_re(n)

    # 残差历史最多保留 256 个点（首尾必留），避免 n=49 时把 1600 点原样发到前端。
    hist = np.asarray(res.residual_history, dtype=float)
    hist = np.maximum(hist, np.finfo(float).tiny)
    if hist.size > 256:
        idx = np.unique(np.linspace(0, hist.size - 1, 256).round().astype(int))
        hist = hist[idx]
        rec_idx = idx.astype(float)
    else:
        rec_idx = np.arange(hist.size, dtype=float)
    residual = np.log10(hist)

    # 粗箭头网格（约 8×8）：只画方向，不画量级。
    stride = max(1, (n - 1) // 8)
    take = np.arange(0, n, stride, dtype=int)
    arrow_u = np.asarray(res.u[np.ix_(take, take)], dtype=float)
    arrow_v = np.asarray(res.v[np.ix_(take, take)], dtype=float)

    fr = Frame(chapter=CHAPTER, viz="cavity_vorticity", kind="frame")
    fr.add("x", gr.x, label="x")
    fr.add("y", gr.y, label="y")
    fr.add("vorticity", res.omega, label="涡量 ω")
    fr.add("psi", res.psi, label="流函数 ψ")
    fr.add("arrow_x", gr.x[take], label="箭头 x")
    fr.add("arrow_y", gr.y[take], label="箭头 y")
    fr.add("arrow_u", arrow_u, label="箭头 u（速度方向）")
    fr.add("arrow_v", arrow_v, label="箭头 v（速度方向）")
    fr.add("residual", residual, label="log10 相对残差")
    fr.add("residual_index", rec_idx, label="迭代记录序号")
    omega_max = max(1.0e-12, float(np.max(np.abs(res.omega))))
    fr.set_meta(
        re=float(re),
        n=int(n),
        h=float(gr.h),
        sweeps=int(res.n_sweeps),
        poisson=int(res.n_poisson),
        converged=1.0 if res.converged else 0.0,
        finite=1.0 if res.finite else 0.0,
        diverged=1.0 if res.diverged else 0.0,
        residual_initial=_finite(res.residual_initial),
        residual_final=_finite(res.residual),
        residual_tol=_CAVITY_TOL,
        max_divergence=_finite(res.max_divergence),
        enstrophy=_finite(cavity.enstrophy(res.omega, gr)),
        kinetic_energy=_finite(cavity.kinetic_energy(res.u, res.v, gr)),
        psi_min=_finite(float(np.min(res.psi))),
        psi_max=_finite(float(np.max(res.psi))),
        omega_abs_max=_finite(omega_max),
        center_valid=1.0 if primary is not None else 0.0,
        center_x=_finite(primary["x"]) if primary is not None else 0.0,
        center_y=_finite(primary["y"]) if primary is not None else 0.0,
        center_psi=_finite(primary["psi"]) if primary is not None else 0.0,
        rmse=_finite(cmp.rmse),
        rmse_u=_finite(cmp.rmse_u),
        rmse_v=_finite(cmp.rmse_v),
        transcription_floor=cavity.BENCHMARK_TRANSCRIPTION_SPREAD,
        dt_cfl=_finite(dt["dt_cfl"]),
        dt_diffusion=_finite(dt["dt_diffusion"]),
        re_cross=_finite(dt["re_cross"]),
        dt_binding=_finite(dt["binding"]),
        cell_re=_finite(cell_re),
        central_threshold_re=_finite(threshold_re),
        overflow=False,
    )
    fr.set_strings(
        status="ok" if res.converged and res.finite else "degraded",
        scheme="upwind",
        note=(
            f"Re={re}、n={n}：{res.n_sweeps} 次 SOR + {res.n_poisson} 次 DST Poisson；"
            f"残差 {res.residual:.2e}，中心线 RMSE {cmp.rmse:.4f}（抄写地板 1e-2）。"
        ),
        residual_note="残差量的是离散方程的离差；中心线 RMSE 量的是离散误差。两者不是一回事。",
        center_note=(
            f"主涡 ({primary['x']:.4f}, {primary['y']:.4f})；Ghia Re={re} 的主涡是 "
            f"({_ghia_centers(re).get('primary', (float('nan'), float('nan')))[0]:.4f}, "
            f"{_ghia_centers(re).get('primary', (float('nan'), float('nan')))[1]:.4f})。"
            if primary is not None else "本网格未找到主涡（阈值 1e-4）。"
        ),
        dt_note=(
            f"dt_diffusion={dt['dt_diffusion']:.4f}，dt_CFL={dt['dt_cfl']:.4f}，"
            f"交叉 Re_x={dt['re_cross']:.1f}；当前由{'CFL' if dt['binding'] > 0.5 else '扩散'}限制。"
        ),
        central_note=(
            f"格点雷诺数 Re_h={cell_re:.2f}；中心格式失去对角占优的阈值 Re={threshold_re:.1f}，"
            + ("所以本网格不能用中心差分。" if cell_re > 2.0 else "本网格仍在中心格式的稳定侧。")
        ),
        benchmark_note=f"与 Ghia 的 RMSE={cmp.rmse:.4f}；基准转载分歧本身就有 1e-2 量级。",
    )
    fr.pitfalls = _PITFALLS["cavity_vorticity"]
    return fr


# --------------------------------------------------------------------------- #
# viz 2：Ghia 基准
# --------------------------------------------------------------------------- #
def _ghia_benchmark(params: Mapping[str, Any]) -> Frame:
    re = int(params["re"])
    n = int(params["n"])
    res = _cavity_case(n, re)
    gr = cavity.cavity_grid(n)
    cmp = cavity.compare_with_benchmark(res.u, res.v, gr, float(re))
    y_grid, u_profile, x_grid, v_profile = cavity.centerline_profiles(res.u, res.v, gr)
    ns, rmse, rmse_u, rmse_v = _ghia_curve(re)
    slope = _log_slope(1.0 / (ns - 1.0), rmse)
    slope_u = _log_slope(1.0 / (ns - 1.0), rmse_u)
    slope_v = _log_slope(1.0 / (ns - 1.0), rmse_v)
    centers = cavity.vortex_centers(res.psi, gr, threshold=1.0e-4)
    primary = centers[0] if centers else None
    ghia_primary = _ghia_centers(re).get("primary")
    center_dist = (
        math.hypot(primary["x"] - ghia_primary[0], primary["y"] - ghia_primary[1])
        if primary is not None and ghia_primary is not None
        else 0.0
    )
    best = int(np.argmin(rmse))

    fr = Frame(chapter=CHAPTER, viz="ghia_benchmark", kind="frame")
    fr.add("y_ref", cmp.y, label="Ghia 竖直中心线采样 y")
    fr.add("u_ref", cmp.u_ref, label="Ghia u_ref")
    fr.add("u_num_ref", cmp.u_num, label="数值解在 Ghia 节点上的 u")
    fr.add("x_ref", cmp.x, label="Ghia 水平中心线采样 x")
    fr.add("v_ref", cmp.v_ref, label="Ghia v_ref")
    fr.add("v_num_ref", cmp.v_num, label="数值解在 Ghia 节点上的 v")
    fr.add("y_profile", y_grid, label="数值网格 y")
    fr.add("u_profile", u_profile, label="数值 u(y)")
    fr.add("x_profile", x_grid, label="数值网格 x")
    fr.add("v_profile", v_profile, label="数值 v(x)")
    fr.add("n_grid", ns, label="网格 n")
    fr.add("error", np.maximum(rmse, FLOOR), label="RMSE（requiredAux）")
    fr.add("error_u", np.maximum(rmse_u, FLOOR), label="u 剖面 RMSE")
    fr.add("error_v", np.maximum(rmse_v, FLOOR), label="v 剖面 RMSE")
    fr.set_meta(
        re=float(re),
        n=int(n),
        rmse=_finite(cmp.rmse),
        rmse_u=_finite(cmp.rmse_u),
        rmse_v=_finite(cmp.rmse_v),
        centerline_offset=_finite(cmp.centerline_offset),
        transcription_floor=float(cmp.transcription_floor),
        grid_slope=None if slope is None else float(slope),
        grid_slope_u=None if slope_u is None else float(slope_u),
        grid_slope_v=None if slope_v is None else float(slope_v),
        best_n=float(ns[best]),
        best_rmse=_finite(rmse[best]),
        residual=_finite(res.residual),
        sweeps=int(res.n_sweeps),
        converged=1.0 if res.converged else 0.0,
        center_dist=_finite(center_dist),
        overflow=False,
    )
    fr.set_strings(
        status="ok" if res.converged and res.finite else "degraded",
        scheme="upwind",
        note=(
            f"Re={re}、n={n}：RMSE={cmp.rmse:.4f}（u {cmp.rmse_u:.4f} / v {cmp.rmse_v:.4f}）；"
            f"基准抄写分歧地板 {cmp.transcription_floor:.2e}。"
        ),
        grid_note=(
            "网格收敛（n=17/25/33/49）："
            + (f"实测阶 {slope:.2f}" if slope is not None else "测不出斜率")
            + f"；最优点 n={ns[best]:.0f}，RMSE={rmse[best]:.4f}。"
        ),
        transcription_note=(
            "Ghia 表在不同转载之间至少有 1e-2 量级的分歧；容差低于它时对拍的是抄写版本。"
        ),
        center_note=(
            f"主涡数值 ({primary['x']:.3f}, {primary['y']:.3f})，Ghia "
            f"({ghia_primary[0]:.3f}, {ghia_primary[1]:.3f})，距离 {center_dist:.3f}（网格分辨率 O(h)）。"
            if primary is not None and ghia_primary is not None
            else "未找到主涡。"
        ),
    )
    fr.pitfalls = _PITFALLS["ghia_benchmark"]
    return fr


# --------------------------------------------------------------------------- #
# viz 3：Gray--Scott / 图灵
# --------------------------------------------------------------------------- #
def _turing_metrics(du: float, dv: float, feed: float, kill: float) -> dict[str, float]:
    phi, lam = rd.most_unstable_mode(du, dv, feed, kill)
    base = rd.turing_base_state(feed, kill)
    if base is None:
        fu, fv, gu, gv = -feed, 0.0, 0.0, -(feed + kill)
        u0, v0 = 1.0, 0.0
        base_valid = 0.0
    else:
        u0, v0 = base
        fu = -v0 * v0 - feed
        fv = -2.0 * u0 * v0
        gu = v0 * v0
        gv = 2.0 * u0 * v0 - (feed + kill)
        base_valid = 1.0
    prod = float(du) * float(dv)
    a2 = float(du) * gv + float(dv) * fu
    z_det = max(0.0, a2 / (2.0 * prod)) if prod > 0.0 else 0.0
    phi_det = math.sqrt(z_det)
    lam_det = float(rd.turing_growth_rate(phi_det, du, dv, feed, kill))
    uniform_rate = float(rd.turing_growth_rate(0.0, du, dv, feed, kill))
    return {
        "phi_star": float(phi),
        "lambda_star": float(lam),
        "uniform_rate": uniform_rate,
        "turing_band": 1.0 if (lam > 0.0 and phi > 1.0e-6 and base_valid > 0.5) else 0.0,
        "base_valid": base_valid,
        "base_u": float(u0),
        "base_v": float(v0),
        "phi_det": float(phi_det),
        "lambda_det": float(lam_det),
    }


def _stencil_errors(du: float, dv: float, feed: float, kill: float, phi_star: float) -> tuple[np.ndarray, np.ndarray, float | None]:
    """五点模板有效波数导致的增长率相对误差，随 n 按 O(Δx²) 收敛。"""
    ns = np.asarray([64.0, 128.0, 256.0, 512.0])
    m = max(1, int(round(phi_star / (2.0 * math.pi))))
    phi = 2.0 * math.pi * m
    rate = max(abs(float(rd.turing_growth_rate(phi, du, dv, feed, kill))), 1.0e-30)
    err = np.empty(ns.size, dtype=float)
    for i, n in enumerate(ns):
        dx = 1.0 / n
        phi_eff = 2.0 / dx * math.sin(phi * dx / 2.0)
        lam_eff = float(rd.turing_growth_rate(phi_eff, du, dv, feed, kill))
        lam = float(rd.turing_growth_rate(phi, du, dv, feed, kill))
        err[i] = abs(lam_eff - lam) / rate
    err = np.maximum(err, FLOOR)
    slope = _log_slope(1.0 / ns, err)
    return ns, err, None if slope is None else float(slope)


def _turing_pattern(params: Mapping[str, Any], t: float) -> Frame:
    preset = str(params["preset"])
    n = int(params["n"])
    steps = int(max(0, min(10000, round(float(t)))))
    res, u0 = _rd_case(preset, n, steps)
    du, dv, feed, kill = rd.RD_PRESETS[preset]
    metrics = _turing_metrics(du, dv, feed, kill)
    phi, rate = rd.turing_dispersion(du, dv, feed, kill, n_modes=256)
    n_err, err, err_slope = _stencil_errors(du, dv, feed, kill, metrics["phi_star"])
    wavelength_raw = float(rd.dominant_wavelength(res.u, res.dx))
    wavelength_valid = bool(math.isfinite(wavelength_raw) and wavelength_raw > 0.0)
    wavelength = wavelength_raw if wavelength_valid else 0.0
    mass_delta = float(res.total_u - float(np.sum(u0)))
    balance = float(rd.reaction_balance(res.u, res.v, feed, kill))
    overflow = False
    vmax = max(1.0e-9, float(np.max(res.v)))

    fr = Frame(chapter=CHAPTER, viz="turing_pattern", kind="frame")
    fr.add("x", (np.arange(n, dtype=float) + 0.5) / n, label="x")
    fr.add("y", (np.arange(n, dtype=float) + 0.5) / n, label="y")
    fr.add("v", res.v, label="v(x,y)")
    fr.add("phi", phi, label="波数 φ")
    fr.add("rate", rate, label="解析增长率 λ(φ)")
    fr.add("n_error", n_err, label="网格 n")
    fr.add("error", err, label="五点模板增长率相对误差（requiredAux）")
    fr.set_meta(
        t=float(steps),
        steps=int(steps),
        n=int(n),
        du=float(du),
        dv=float(dv),
        feed=float(feed),
        kill=float(kill),
        dx=float(res.dx),
        dt=float(res.dt),
        dt_requested=float(res.dt_requested),
        dt_limit=float(res.dt_limit),
        dt_clamped=1.0 if res.dt_clamped else 0.0,
        time_physical=float(res.time),
        amplitude=float(res.amplitude),
        spot_count=float(res.spot_count),
        wavelength=wavelength,
        wavelength_valid=1.0 if wavelength_valid else 0.0,
        total_u=float(res.total_u),
        total_v=float(res.total_v),
        mass_delta=mass_delta,
        balance=balance,
        vmax=vmax,
        phi_star=metrics["phi_star"],
        lambda_star=metrics["lambda_star"],
        phi_det=metrics["phi_det"],
        lambda_det=metrics["lambda_det"],
        base_valid=metrics["base_valid"],
        base_u=metrics["base_u"],
        base_v=metrics["base_v"],
        uniform_rate=metrics["uniform_rate"],
        turing_band=metrics["turing_band"],
        grid_error_slope=None if err_slope is None else float(err_slope),
        finite=1.0 if res.finite else 0.0,
        overflow=False,
    )
    fr.set_strings(
        status="ok" if res.finite else "degraded",
        preset=preset,
        note=(
            f"{preset}：φ*={metrics['phi_star']:.3f}，λ*={metrics['lambda_star']:.5f}；"
            f"底态 {'有' if metrics['base_valid'] > 0.5 else '无'}（非平凡稳态）。"
        ),
        turing_note=(
            "λ*>0、φ*>0 且底态无扩散稳定：这是真图灵失稳。"
            if metrics["turing_band"] > 0.5
            else (
                "平凡态中性：λ 恒为 0，没有失稳。"
                if preset == "uniform"
                else (
                    f"最快模在 φ=0（λ(0)={metrics['uniform_rate']:+.5f}）：底态均匀失稳，不是图灵带。"
                    if metrics["base_valid"] > 0.5
                    else "没有非平凡稳态（底态退回平凡态）：经典斑图来自自复制瞬态，不是图灵分岔。"
                )
            )
        ),
        dispersion_note=(
            f"det 顶点 φ={metrics['phi_det']:.3f} 给 λ={metrics['lambda_det']:.5f}，"
            f"最不稳定点 φ*={metrics['phi_star']:.3f} 给 λ*={metrics['lambda_star']:.5f}；"
            "只信 det 顶点会系统性偏大波长。"
            if metrics["turing_band"] > 0.5
            else (
                "反应项为零：λ(φ)=-Dφ²，只有纯扩散衰减，无增长模。"
                if preset == "uniform"
                else "没有正波数图灵带：最快模在 φ=0，det 顶点只对真图灵带有比较意义。"
            )
        ),
        dt_note=(
            f"请求 dt={res.dt_requested:g}；显式上限 {res.dt_limit:.5f}；"
            + (f"已夹到 {res.dt:g}。" if res.dt_clamped else "未触发夹紧。")
        ),
        pattern_note=(
            f"t={res.time:.2f}（{steps} 步），振幅 {res.amplitude:.3f}，局部极大 {res.spot_count} 个，"
            + (f"实测主波长 {wavelength:.3f}（瞬态期会受种子尺度影响）。" if wavelength_valid else "无空间结构：主波长无定义（不是 0）。")
        ),
        mass_note=(
            f"Σu 变化 {mass_delta:.3e}；终态反应项平衡 {balance:.3e}。"
            "Gray--Scott 不守恒，Σu 只能由反应项改变；扩散项在周期域上逐点和为零。"
        ),
        grid_note=(
            f"五点模板增长率误差（n=64/128/256/512）实测阶 "
            + (f"{err_slope:.2f}" if err_slope is not None else "测不出")
            + "；这是空间模板误差，不是完整时间推进误差。"
        ),
    )
    fr.pitfalls = _PITFALLS["turing_pattern"]
    return fr


# --------------------------------------------------------------------------- #
# viz 4：干底溃坝
# --------------------------------------------------------------------------- #
def _dam_break(params: Mapping[str, Any], t: float) -> Frame:
    n = int(params["n"])
    cfl = float(params["cfl"])
    t_end = float(max(0.0, min(_DAM_T_MAX, t)))
    x, h0, hu0, res = _swe_case(n, cfl, t_end)
    if t_end <= 0.0:
        h_num = h0.copy()
        hu_num = hu0.copy()
        h_exact, u_exact = sw.ritter_dam_break(x, 0.0)
        max_cfl = 0.0
        n_steps = 0
        dt_last = 0.0
        mass = sw.total_mass(x, h_num)
        mass0 = mass
        energy = sw.total_energy(x, h_num, hu_num)
        reached = True
    else:
        h_num = np.asarray(res.h, dtype=float)
        hu_num = np.asarray(res.hu, dtype=float)
        h_exact, u_exact = sw.ritter_dam_break(x, res.t)
        max_cfl = float(res.max_cfl)
        n_steps = int(res.n_steps)
        dt_last = float(res.dt)
        mass = float(res.mass)
        mass0 = sw.total_mass(x, h0)
        energy = float(res.energy)
        reached = bool(res.reached_t_end)
    u_num = np.where(h_num > 1.0e-12, hu_num / np.maximum(h_num, 1.0e-12), 0.0)
    err = np.abs(h_num - h_exact)
    err_floor, err_valid = _floor_error(err)
    c_l = math.sqrt(sw.G_DEFAULT * 1.0)
    xi = np.full(x.shape, -1.0e9) if t_end <= 0.0 else (x - 0.5) / t_end
    fan = (xi > -c_l) & (xi < 2.0 * c_l)
    if t_end <= 0.0:
        fan = np.zeros(x.shape, dtype=bool)
    base = max(float(np.sum(np.maximum(h_exact, 0.0)) * (x[1] - x[0])), 1.0e-30)
    l1_fan = float(np.sum(err[fan]) * (x[1] - x[0]) / base) if np.any(fan) else 0.0
    l1_all = float(np.sum(err) * (x[1] - x[0]) / base)
    mass_drift = abs(mass - mass0) / max(abs(mass0), 1.0e-30)
    finite = bool(np.all(np.isfinite(h_num)) and np.all(np.isfinite(hu_num)))

    fr = Frame(chapter=CHAPTER, viz="dam_break", kind="frame")
    fr.add("x", x, label="x")
    fr.add("h", h_num, label="数值水深 h")
    fr.add("hu", hu_num, label="数值动量 hu")
    fr.add("u", u_num, label="数值速度 u")
    fr.add("h_exact", h_exact, label="Ritter 精确水深")
    fr.add("u_exact", u_exact, label="Ritter 精确速度")
    fr.add("h_initial", h0, label="初始水深")
    fr.add("error", err_floor, label="|h-h_exact|（requiredAux）")
    fr.add("error_valid", err_valid, label="误差有效掩码")
    fr.add("fan", fan.astype(float), label="Ritter 扇区掩码")
    fr.set_meta(
        t=float(t_end),
        n=int(n),
        dx=float(x[1] - x[0]),
        cfl=float(cfl),
        dt=float(dt_last),
        n_steps=int(n_steps),
        max_cfl=float(max_cfl),
        mass=float(mass),
        mass0=float(mass0),
        mass_drift=float(mass_drift),
        energy=float(energy),
        h_min=float(np.min(h_num)),
        reached=1.0 if reached else 0.0,
        finite=1.0 if finite else 0.0,
        l1_fan=l1_fan,
        l1_all=l1_all,
        l1_ratio=(l1_all / l1_fan) if l1_fan > 0 else 0.0,
        fan_lo=float(0.5 - c_l * t_end) if t_end > 0 else 0.0,
        fan_hi=float(0.5 + 2.0 * c_l * t_end) if t_end > 0 else 0.0,
        overflow=False,
    )
    fr.set_strings(
        status="ok" if finite and reached else "degraded",
        scheme="rusanov",
        note=(
            f"n={n}、cfl={cfl:g}：{n_steps} 步到 t={t_end:.4f}，实际 max CFL={max_cfl:.3f}。"
        ),
        window_note=(
            "t=0：尚未推进，Ritter 扇区无定义。"
            if t_end <= 0.0
            else (
                f"Ritter 扇区 ξ∈(-c_L,2c_L)，本帧窗口 x∈[{0.5 - c_l * t_end:.3f}, "
                f"{0.5 + 2.0 * c_l * t_end:.3f}]；扇内与全域 L1 分开报。"
            )
        ),
        mass_note=(
            f"质量相对漂移 {mass_drift:.2e}：干底上正性地板每步注入 min_depth·Δx 的水，"
            "这个数不是格式不守恒的证据。"
        ),
        cfl_note=(
            f"目标 CFL={cfl:g}，实测最大值 {max_cfl:.3f}；cfl<=0.9 时本算例稳定，"
            "页面不把越界值悄悄夹回去。"
        ),
        error_note=(
            f"扇内 L1={l1_fan:.2e}，全域 L1={l1_all:.2e}；"
            "全域多出来的是干锋位置误差，不是格式阶的干净估计。"
        ),
    )
    fr.pitfalls = _PITFALLS["dam_break"]
    return fr


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def compute(viz_id: str, params: Mapping[str, Any], t: float) -> Frame:
    """章节计算入口：``params`` 已由 :mod:`server.frames` 规范化并校验过区间。"""
    if viz_id == "cavity_vorticity":
        return _cavity_vorticity(params)
    if viz_id == "ghia_benchmark":
        return _ghia_benchmark(params)
    if viz_id == "turing_pattern":
        return _turing_pattern(params, t)
    if viz_id == "dam_break":
        return _dam_break(params, t)
    raise KeyError(f"章节 6.1 没有可视化 {viz_id!r}")
