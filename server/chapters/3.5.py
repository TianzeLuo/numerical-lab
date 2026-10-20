r"""3.5 章：动力系统、分岔与混沌（5 个可视化）。

帧只做搬运：所有数学量都来自 :mod:`numcore.ode.dynamics`、:mod:`numcore.ode.lyapunov`
与 :mod:`numcore.ode.continuation`，本文件不写数学。五个可视化的分工是
「先看见轨道，再看见分岔，再看见指数，最后看见截面与相图」：

1. ``lorenz_butterfly`` —— **招牌**：Lorenz 吸引子上两条相差 ``1e-8`` 的孪生轨道。
   柱子不是「画得像」：帧里给出**实测**的分离曲线 :math:`d(t)`、它的拟合增长率
   （与 Lyapunov 指数并排）、饱和距离，以及「1e-8 的差要多久才看得见」——
   并同时给出 :math:`h` 与 :math:`2h` 两条曲线，让「这不是数值噪声」当场可查。
2. ``bifurcation_logistic`` —— logistic 映射的分岔图：拖动参数 :math:`r` 实时生成，
   周期窗口用**检测出来的最小周期**着色，混沌区用「最接近闭合的残差」标出来。
3. ``lyapunov_overlay`` —— :math:`\lambda(r)` 曲线叠在分岔图上，副轴给每个参数点的
   分块标准误：一条没有误差棒的 :math:`\lambda` 曲线不足以说明任何事。
4. ``poincare_section`` —— 三维轨道打在平面上的点集（降维后的真正对象），
   每点都带「离平面多远」的残差，另有线性插值位移作为定位误差的实测。
5. ``phase_portrait`` —— 向量场 + 零倾线 + 平衡点分类：数值零倾线与闭式零倾线
   同时画，两者的差就是这张图的可信度。

三维数据怎么办（与 1.5 章 ``krylov_growth`` 同一条约定）
--------------------------------------------------------
``viz/shared/src/scene.ts`` 只有二维图层，所以本帧**同时**给：

* 原始三维数组（``traj1_x/y/z``、``section_x/y/z`` 或 ``orbit_x/y/z``），
* 按 ``azimuth``/``elevation`` 参数投影好的二维数组（``*_u/*_v``），
* ``strings.projection`` 里的投影定义（公式 + 当前角度），前端可以**换角度重画**
  （自己算 :math:`(u,v)` 即可，不必回后端）。

投影是正交投影，没有透视除法：远近不改变尺寸，深度在图上不可读——这句话写在
``strings.projection_note`` 里，不能让人误以为近处的点更大是「更亮/更强」。

帧契约（与 1.2/3.2 章一致）
--------------------------
``meta`` 只放有限标量（``allow_nan=False``，非有限值一律换成哨兵 + 显式标志），
字符串走 ``strings``，数组走 ``add``。时间轴语义逐个写清：viz 1/4/5 是真实时间
:math:`t`，viz 2/3 没有时间轴（参数由滑块给，图是整条参数网格一次算出来的）。
"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping

import numpy as np

from numcore.ode import continuation as cont
from numcore.ode import dynamics as dyn
from numcore.ode import lyapunov as lyap
from numcore.utils import rng
from server.nlf import Frame

__all__ = ["MANIFEST", "compute"]

#: 单位舍入（与 1.2/3.2 章同口径）：对数轴的地板与「精确 0」的显示都用它。
_ROUNDOFF = float(np.finfo(float).eps) / 2.0

#: 显示分辨率：屏幕上一个像素在单位尺度下大约就是这个量级。
#: 「1e-8 的初值差要多久才看得见」这句话里的「看得见」用它定义（= 1e-3 相对吸引子直径）。
_DISPLAY_RES = 1.0e-3

#: 孪生实验的固定种子：初值必须可复现，绝不掷骰子。
_RNG_SEED = 20261018

#: Lorenz 孪生实验的统一口径（写进 strings.twin_protocol，也在 tests 里被引用）。
_SETTLE_T = 20.0
_SETTLE_H = 0.01
_TWIN_T = 30.0
_TWIN_H = 0.005
_TWIN_H_COARSE = 0.01
_TWIN_STRIDE = 10
_TWIN_STRIDE_COARSE = 5

#: 分岔图的固定参数窗口：越过 4 之后 logistic 映射跑出 [0,1]（真解跑到负无穷）。
_R_LO, _R_HI = 2.4, 4.0

#: 时间轴的固定上限（与 manifest 的 time.tMax 一致，写在这里避免两处各写一遍）。
_POINCARE_T_END = 60.0
_PHASE_T_END = 12.0

#: 公认值与临界值（全部有出处，写进 strings 与 notes.tex）。
_LAMBDA_LORENZ = lyap.BENETTIN_REFERENCE_LORENZ
_R_DOUBLING = (3.0, 1.0 + math.sqrt(6.0), 3.5440903595519225, 3.564407266095432)
_R_WINDOW3 = 1.0 + math.sqrt(8.0)
_R_ACCUM = 3.569945672
_R_SUPERSTABLE2 = 1.0 + math.sqrt(5.0)

#: 平衡点分类的编码（数值写进数组，名字走 strings——数组里不能放字符串）。
_KIND_CODES = dict(dyn.KIND_CODES)

#: 帧里「非有限值怎么办」的统一口径，写进每帧的 ``strings.nonfinite_policy``。
_NONFINITE_POLICY = (
    "本帧不含任何非有限值：所有数组在入帧前都经过 _finite_or（占位值 + 掩码 + 文案三件套），"
    "meta 一律是有限标量（.nlf 头用 allow_nan=False 编码，塞一个 NaN 整帧就写不出去）。"
    "Lyapunov 指数在 r=2（超稳定不动点）处理论上是 −∞，本帧只画 r≥2.4，"
    "并把「−∞ 的位置」写成 strings 里的一句话，而不是画一个假的有限值。"
)


def _fin(x: Any, default: float = 0.0) -> float:
    """把非有限标量换成哨兵：``.nlf`` 头用 ``allow_nan=False``，NaN/inf 直接写不进去。"""
    v = float(x)
    return v if math.isfinite(v) else float(default)


def _floor(arr: Any, floor: float = _ROUNDOFF) -> np.ndarray:
    """对数轴的「软着陆」：把 0（以及非有限值）抬到明确的地板上。"""
    a = np.asarray(arr, dtype=float).copy()
    a[~np.isfinite(a)] = floor
    return np.maximum(a, floor)


def _finite_or(arr: Any, fill: float = 0.0) -> tuple[np.ndarray, np.ndarray, int]:
    """把非有限元显式换成 ``fill``，返回 ``(干净数组, 掩码, 个数)``（三件套）。"""
    a = np.asarray(arr, dtype=float)
    bad = ~np.isfinite(a)
    n = int(np.count_nonzero(bad))
    if n:
        a = np.where(bad, float(fill), a)
    return a, bad.astype(float), n


def _clip(t: float, lo: float, hi: float) -> float:
    """把时间轴夹到区间 [lo, hi]（拖到头就不动了，不是卡住）。"""
    return max(float(lo), min(float(hi), float(t)))


def _step_of(t: float, n_steps: int) -> int:
    """把连续时间轴四舍五入到整数步并夹住（与 1.2/3.2 章同口径）。"""
    k = int(math.floor(float(t) + 0.5))
    return max(0, min(k, int(n_steps)))


def _lerp(xp: Any, fp: Any, x: Any) -> np.ndarray:
    """逐元素线性插值（自写；不引 numpy.interp，保持与 numcore 同一口径）。

    ``xp`` 必须单调递增。区间外按端点取值（不外推——外推出来的点没有意义）。
    """
    xp_a = np.asarray(xp, dtype=float).reshape(-1)
    fp_a = np.asarray(fp, dtype=float).reshape(-1)
    x_a = np.asarray(x, dtype=float)
    if xp_a.size < 2:
        return np.full(x_a.shape, float(fp_a[0]) if fp_a.size else 0.0)
    i = np.clip(np.searchsorted(xp_a, x_a, side="right") - 1, 0, xp_a.size - 2)
    x0 = xp_a[i]
    x1 = xp_a[i + 1]
    w = np.where(x1 > x0, (x_a - x0) / np.where(x1 > x0, x1 - x0, 1.0), 0.0)
    return fp_a[i] + w * (fp_a[i + 1] - fp_a[i])


def _seg_ids(sizes: Any) -> np.ndarray:
    """把若干条折线的点数变成「段号」数组（折线不用 NaN 分隔，NaN 会让前端静默丢段）。"""
    out = [np.full(int(n), float(k)) for k, n in enumerate(sizes)]
    return np.concatenate(out) if out else np.zeros(0)


def _pack_segments(branches: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把 ``[(x, y), ...]`` 分支列表拼成 ``(x, y, 段号)`` 三个数组（空分支安全）。"""
    xs = [np.asarray(b[0], dtype=float).reshape(-1) for b in branches]
    ys = [np.asarray(b[1], dtype=float).reshape(-1) for b in branches]
    if not xs:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    return np.concatenate(xs), np.concatenate(ys), _seg_ids([a.size for a in xs])


def _projection_strings(az: float, el: float) -> dict[str, str]:
    """三维帧的投影说明：把公式与当前角度一起交给前端（换角度不必回后端）。"""
    return {
        "projection": json.dumps(
            {
                "kind": "orthographic",
                "azimuth_deg": _fin(az),
                "elevation_deg": _fin(el),
                "u": "-sin(az)*x + cos(az)*y",
                "v": "-cos(az)*sin(el)*x - sin(az)*sin(el)*y + cos(el)*z",
                "note": "原始三维数组同时给出，前端可用同一公式换角度重算",
            },
            ensure_ascii=False,
            allow_nan=False,
        ),
        "projection_note": (
            "正交投影（无透视除法）：远近不改变尺寸，深度在二维图上不可读——"
            "不要从点的疏密读「远近」，疏密只反映轨道在相空间里停留的久暂（这恰恰是我们想要的）。"
        ),
    }


# --------------------------------------------------------------------------- #
# 固定的初值（全部写死，不用随机数——帧必须逐次可复现）
# --------------------------------------------------------------------------- #
def _lorenz_base(n_pairs: int) -> np.ndarray:
    """孪生实验的 ``n_pairs`` 个基准初值：种子写死，落在吸引子上（先积分掉暂态）。"""
    gen = rng(_RNG_SEED)
    return gen.uniform([-15.0, -20.0, 5.0], [15.0, 20.0, 40.0], size=(int(n_pairs), 3))


def _circle_ics(center: tuple[float, float], radius: float, n: int) -> np.ndarray:
    """相图上的初始点：以 ``center`` 为心、``radius`` 为半径的圆周上均匀取 ``n`` 个。"""
    th = np.linspace(0.0, 2.0 * math.pi, int(n), endpoint=False) + 0.35
    return np.stack([center[0] + radius * np.cos(th), center[1] + radius * np.sin(th)], axis=1)


# --------------------------------------------------------------------------- #
# manifest（PROTOCOLS §5：每个 viz 的参数、时间轴、副轴与防误导文案）
# --------------------------------------------------------------------------- #
MANIFEST: dict[str, Any] = {
    "chapter": "3.5",
    "title": "动力系统、分岔与混沌",
    "part": "III",
    "summary": (
        "相空间里的轨道、分岔图上的窗口、Lyapunov 指数的正负，三件事说的是同一件事："
        "相邻轨道分不分开。本章每一步都给实测：1e-8 的初值差多久分开（以及步长减半后"
        "这个数是否还站得住）、周期窗口的闭合残差、$\\lambda$ 估计的分块标准误、截面点离平面多远。"
    ),
    "vizzes": [
        # ------------------------------------------------------------------ #
        {
            "id": "lorenz_butterfly",
            "title": "Lorenz 孪生轨道：1e-8 的差多久看得见",
            "description": (
                "两条初值只差 1e-8 的轨道在 Lorenz 吸引上同时推进。主图是三维相空间的正交投影"
                "（原始 x/y/z 同帧给出，角度可调）；副图是对数坐标的分离距离 $d(t)$，"
                "细线是每一对孪生轨迹、粗线是平均，另有 $h$ 与 $2h$ 两次积分的对照曲线。"
                "帧里同时给出：实测拟合增长率、跨孪生对的散布、饱和距离、"
                "以及「以显示分辨率 1e-3 为准要多久才看得见」的实测与解析两个答案。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "sigma",
                    "label": "σ（Prandtl 数）",
                    "type": "float",
                    "min": 4.0,
                    "max": 20.0,
                    "step": 0.5,
                    "default": 10.0,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "rho",
                    "label": "ρ（Rayleigh 数）",
                    "type": "float",
                    "min": 5.0,
                    "max": 60.0,
                    "step": 0.5,
                    "default": 28.0,
                    "critical": [1.0, 24.736842105263158],
                    "criticalNote": (
                        "$\\rho=1$：原点的一对实特征值穿过 $0$，原点失稳；"
                        "$\\rho=\\rho_H=470/19\\approx 24.7368$：$C_\\pm$ 发生 Hopf 分岔（本仓二分实测 24.736842105253，"
                        "与闭式差 1e-11）。$\\rho<\\rho_H$ 时吸引子是稳定平衡点，根本没有混沌可看。"
                    ),
                    "beyond": "above",
                },
                {
                    "name": "beta",
                    "label": "β（几何因子）",
                    "type": "float",
                    "min": 1.0,
                    "max": 5.0,
                    "step": 0.05,
                    "default": 2.6666666666666665,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "log10_eps",
                    "label": "初值差（10 的幂）",
                    "type": "float",
                    "min": -14.0,
                    "max": -2.0,
                    "step": 0.5,
                    "default": -8.0,
                    "critical": [-14.3522],
                    "criticalNote": (
                        "$\\mathrm{log}_{10}(20u)\\approx -14.35$ 是 float64 在 $|x|\\approx 20$ 上的分辨极限："
                        "初值差再小，加在 $x$ 上就四舍五入回同一个数，两条轨道从第一步起完全相同，"
                        "分离恒等于 $0$——那是「没有差」，不是「差得慢」。"
                    ),
                    "beyond": "below",
                },
                {
                    "name": "n_pairs",
                    "label": "孪生对数量",
                    "type": "int",
                    "min": 1,
                    "max": 12,
                    "default": 8,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "azimuth",
                    "label": "方位角（度）",
                    "type": "float",
                    "min": -180.0,
                    "max": 180.0,
                    "step": 5.0,
                    "default": -60.0,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "elevation",
                    "label": "仰角（度）",
                    "type": "float",
                    "min": -89.0,
                    "max": 89.0,
                    "step": 5.0,
                    "default": 20.0,
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "scrub", "tMax": 30.0, "dt": 0.1, "fps": 20, "label": "时间 t"},
            "axes": [
                {"id": "scene", "title": "三维相空间（正交投影，等比例）", "autoscaled": False},
                {"id": "sep", "title": "孪生轨道分离距离 d(t)", "autoscaled": True, "scale": "log"},
            ],
            "requiredAux": ["error"],
            "theory": [
                "确定性系统的初值敏感性由最大 Lyapunov 指数定量：两条轨道初始相差 $\\delta_0$，"
                "在饱和之前按 $\\delta(t)\\approx\\delta_0 e^{\\lambda_1 t}$ 分开。",
                "Lorenz 系统在 $\\sigma=10$、$\\rho=28$、$\\beta=8/3$ 处的散度 $\\nabla\\cdot f=-(\\sigma+1+\\beta)=-13.6667$ 处处为常数，"
                "所以三个 Lyapunov 指数之和必须等于它（本帧用孪生实验的增长率单独校验最大那个）。",
                "有限时间增长率不等于 Lyapunov 指数：它依赖拟合窗口与初值，所以帧里给出窗口、"
                "用点数、$R^2$ 与跨孪生对的散布。",
            ],
            "pitfalls": (
                "「看起来分开了」不是证据：主图只有两个像素宽，任何数值噪声都能让两条曲线错开。"
                "本帧的判据是副图里的三条线——实测 $d(t)$ 在 $10^{-6}\\le d\\le 1$ 的窗口内与指数增长线平行"
                "（$R^2$ 写在 meta 里）、$h$ 与 $2h$ 两条曲线在该窗口内相差不到 1%（meta.rel_step_max）、"
                "并且饱和距离与吸引子直径同量级。缺任何一条，这张图就只能说明「积分器在舍入」。"
            ),
        },
        # ------------------------------------------------------------------ #
        {
            "id": "bifurcation_logistic",
            "title": "logistic 分岔图：拖动参数实时生成",
            "description": (
                "参数网格上每个 $r$ 丢掉暂态后留下 n_keep 个迭代值，画成一列点：周期轨道是几条线，"
                "混沌是一片。每个参数点的最小周期是**检测**出来的（$|f^p(x)-x|\\le 10^{-9}$），"
                "副轴给每个参数的闭合残差（周期窗口 1e-15、混沌区 1e-2，两类点在同一个对数图上是分开的）。"
                "插图是当前 $r$ 的映射曲线与蛛网图。拖动 $r$ 越过 $3$、$1+\\sqrt 6$、$1+\\sqrt 8$ 看周期窗口如何出现。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "r",
                    "label": "参数 r",
                    "type": "float",
                    "min": 2.4,
                    "max": 4.0,
                    "step": 0.001,
                    "default": 3.9,
                    "critical": [3.0, 3.449489742783178, 3.5440903595519225, 3.564407266095432,
                                 3.8284271247461903],
                    "criticalNote": (
                        "$r_1=3$（1 周期→2 周期）、$r_2=1+\\sqrt 6=3.449489742783$（2→4）、"
                        "$r_3=3.5440903595519225$（4→8）、$r_4=3.564407266095432$（8→16，本仓二分实测逐位相符）、"
                        "$1+\\sqrt 8=3.8284271247461903$（3 周期窗口起点，切分岔）。"
                    ),
                    "beyond": "above",
                },
                {
                    "name": "n_r",
                    "label": "参数网格点数",
                    "type": "int",
                    "min": 200,
                    "max": 1200,
                    "default": 700,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "n_keep",
                    "label": "每参数留下的迭代点数",
                    "type": "int",
                    "min": 16,
                    "max": 128,
                    "default": 64,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "n_transient",
                    "label": "丢掉的暂态迭代数",
                    "type": "int",
                    "min": 100,
                    "max": 4000,
                    "default": 800,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "max_period",
                    "label": "周期检测上限",
                    "type": "int",
                    "min": 8,
                    "max": 64,
                    "default": 32,
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "none"},
            "axes": [
                {"id": "main", "title": "分岔图：迭代值 vs 参数 r", "autoscaled": False},
                {"id": "residual", "title": "周期闭合残差 |f^p(x)−x|", "autoscaled": True, "scale": "log"},
                {"id": "cobweb", "title": "映射与蛛网图", "autoscaled": False},
            ],
            "requiredAux": ["residual"],
            "theory": [
                "周期 $p$ 轨道是 $f^p(x)=x$ 的根；稳定性由乘子 $M=\\prod f'(x_i)$ 决定，$|M|<1$ 稳定。",
                "周期倍增分岔发生在 $M=-1$：$r_1=3$、$r_2=1+\\sqrt 6$ 有闭式，更高阶由二分实测。",
                "3 周期窗口的起点是切分岔（$M=+1$），有闭式 $1+\\sqrt 8$；窗口内部的倍周期级联"
                "按 Feigenbaum 常数 $\\delta=4.6692016\\ldots$ 收缩。",
            ],
            "pitfalls": (
                "分辨率不足会把周期窗口画糊：$\\Delta r$ 与窗口宽度同量级时，"
                "整串倍周期级联被压成一条灰带，看起来「这里就是混沌」；"
                "竖直方向留下的点数不够时，本来分开的几条周期线又会糊成一条粗线。"
                "判据不看图：副轴的闭合残差把周期点（1e-15 量级）与混沌点（1e-2 量级）分在对数轴两端，"
                "再把 max_period 调小，原本判为「周期 $32$」的参数会掉进「混沌」桶里——"
                "所以 meta 里同时给出用了多大的 max_period。"
            ),
        },
        # ------------------------------------------------------------------ #
        {
            "id": "lyapunov_overlay",
            "title": "Lyapunov 指数叠在分岔图上",
            "description": (
                "同一条参数网格上同时算 $\\lambda(r)=\\mathrm{lim}(1/N)\\sum\\mathrm{ln}|f'(x_n)|$ 与分岔图："
                "$\\lambda>0$ 的区间恰好是分岔图上「成片」的区间，$\\lambda<0$ 的区间恰好是周期窗口。"
                "副轴给 $\\lambda$ 估计的分块标准误（有限时间涨落），三个有闭式解的参数点上"
                "给出解析值与实测值的对照。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "r_probe",
                    "label": "探针参数 r",
                    "type": "float",
                    "min": 2.4,
                    "max": 4.0,
                    "step": 0.001,
                    "default": 3.9,
                    "critical": [3.0, 3.449489742783178, 3.5440903595519225, 3.564407266095432,
                                 3.8284271247461903, 3.5699456718],
                    "criticalNote": (
                        "前四个是周期倍增点（$r_2=1+\\sqrt 6$ 有闭式）、$1+\\sqrt 8$ 是 3 周期窗口起点，"
                        "$3.5699456718$ 是 Feigenbaum 聚点 $r_\\infty$（本仓由超稳定参数外推实测 3.569945672）。"
                    ),
                    "beyond": "above",
                },
                {
                    "name": "n_r",
                    "label": "参数网格点数",
                    "type": "int",
                    "min": 200,
                    "max": 1200,
                    "default": 600,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "n_iter",
                    "label": "每个参数的迭代次数",
                    "type": "int",
                    "min": 5000,
                    #: 上限由**帧预算**定：参数批迭代 60000 次 × 600 个参数 ≈ 0.7 s，
                    #: 再加三个解析对拍点与探针就接近 1 s。要更高的精度请把 n_r 调小。
                    "max": 60000,
                    "default": 20000,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "n_transient",
                    "label": "丢掉的暂态迭代数",
                    "type": "int",
                    "min": 200,
                    "max": 5000,
                    "default": 2000,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "n_block",
                    "label": "分块数（标准误）",
                    "type": "int",
                    "min": 2,
                    "max": 16,
                    "default": 8,
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "none"},
            "axes": [
                {"id": "main", "title": "λ(r) 与分岔图", "autoscaled": False},
                {"id": "err", "title": "λ 的分块标准误", "autoscaled": True, "scale": "log"},
            ],
            "requiredAux": ["error"],
            "theory": [
                "$\\lambda=\\mathrm{lim}(1/N)\\sum\\mathrm{ln}|f'(x_n)|$；对 $r=4$ 共轭到 Bernoulli 移位，$\\lambda=\\mathrm{ln}(2)$ 是精确值。",
                "2 周期窗口内乘子有闭式 $M=4+2r-r^2$，$\\lambda=\\frac{1}{2}\\mathrm{ln}|M|$；$r=3.2$ 处 $M=0.16$，$\\lambda=-0.9162907\\ldots$。",
                "周期窗口内 $\\lambda<0$、混沌区 $\\lambda>0$；判定边界是 $\\lambda=0$（切分岔点）。",
            ],
            "pitfalls": (
                "「$\\lambda$ 是正的所以混沌」这句话在有限迭代下不成立："
                "周期窗口边缘的 $\\lambda$ 估计会随迭代次数漂移（乘子趋近 $\\pm 1$ 时收敛是代数级的），"
                "所以副轴必须画标准误。另一个坑是把 $r=2$ 附近画进去——那里不动点是超稳定的，"
                "$\\lambda$ 理论上是 $-\\infty$，任何有限值都是假的。本帧参数范围从 2.4 开始，"
                "并在 strings 里说明这条 $-\\infty$ 为什么不画。"
            ),
        },
        # ------------------------------------------------------------------ #
        {
            "id": "poincare_section",
            "title": "Poincaré 截面：三维轨道打在平面上的点集",
            "description": (
                "轨道穿过一个平面（Lorenz 用 $z=\\rho-1$，Rössler 用 $x=0$）时按方向记一个点，"
                "把这些点按时间顺序连起来就是返回映射。主图是三维轨道的正交投影与平面上的落点，"
                "副图是第 $k$ 次穿越到第 $k+1$ 次穿越的返回映射。每个点都带离平面的残差。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "system",
                    "label": "系统",
                    "type": "enum",
                    "options": ["lorenz", "rossler"],
                    "default": "lorenz",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "rho",
                    "label": "ρ（Lorenz）",
                    "type": "float",
                    "min": 15.0,
                    "max": 60.0,
                    "step": 0.5,
                    "default": 28.0,
                    "critical": [24.736842105263158],
                    "criticalNote": (
                        "$\\rho_H=470/19$：$C_\\pm$ 的 Hopf 分岔；截面取在 $z=\\rho-1$（经过两个非平凡平衡点的平面）。"
                        "$\\rho<\\rho_H$ 时轨道收敛到平衡点，截面点会退化成两三个不动点。"
                    ),
                    "beyond": "above",
                },
                {
                    "name": "sigma",
                    "label": "σ（Lorenz）",
                    "type": "float",
                    "min": 4.0,
                    "max": 20.0,
                    "step": 0.5,
                    "default": 10.0,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "c",
                    "label": "c（Rössler）",
                    "type": "float",
                    "min": 3.0,
                    "max": 12.0,
                    "step": 0.1,
                    "default": 5.7,
                    "critical": [4.2],
                    "criticalNote": (
                        "$a=b=0.2$ 的 Rössler：$c\\approx 4.2$ 处最大 Lyapunov 指数由负转正（本仓实测 $T=800$："
                        "$\\lambda_1(4.2)=-0.0003$、$\\lambda_1(4.3)=+0.0330$，二分 12 步收敛到 $c\\approx 4.2032$，"
                        "那里的 $\\lambda_1$ 已经降到 $\\pm 3\\times 10^{-5}$；$c=5.7$ 处 $\\lambda_1\\approx 0.069$）。"
                        "截面点的形状从有限个簇变成一条曲线。"
                    ),
                    "beyond": "above",
                },
                {
                    "name": "n_cross",
                    "label": "最多显示多少穿越点",
                    "type": "int",
                    "min": 20,
                    "max": 200,
                    "default": 80,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "azimuth",
                    "label": "方位角（度）",
                    "type": "float",
                    "min": -180.0,
                    "max": 180.0,
                    "step": 5.0,
                    "default": -60.0,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "elevation",
                    "label": "仰角（度）",
                    "type": "float",
                    "min": -89.0,
                    "max": 89.0,
                    "step": 5.0,
                    "default": 20.0,
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "scrub", "tMax": 60.0, "dt": 0.25, "fps": 20, "label": "时间 t"},
            "axes": [
                {"id": "scene", "title": "三维相空间与截面平面", "autoscaled": False},
                {"id": "section", "title": "截面内的落点（原始坐标）", "autoscaled": True},
                {"id": "retmap", "title": "返回映射", "autoscaled": True},
            ],
            "requiredAux": ["residual"],
            "theory": [
                "连续流的 Poincaré 映射把 $n$ 维流降到 $n-1$ 维映射，且保持周期轨与稳定性结构。",
                "定向截取（只收单向穿越）保证返回映射是单值的；双向都收会出现重影。",
                "截面点的定位误差是 $\\mathcal{O}(h^2)$（线性插值）加上 Newton 精化的舍入，两者都在帧里给出。",
            ],
            "pitfalls": (
                "截面点集合看起来像一条曲线不等于它是映射的图像："
                "如果只收一个方向的穿越却把平面的两侧都算进去，会得到两团重叠的点"
                "（返回映射变成双值）。另一个坑是拿插值点当精确点："
                "$h=0.01$ 时线性插值点在相空间里离真穿越点约 0.06（meta.max_shift），"
                "相对吸引子直径 0.2%——把它当成「零误差」去和理论曲线比对，"
                "误差就会被算到理论头上。"
            ),
        },
        # ------------------------------------------------------------------ #
        {
            "id": "phase_portrait",
            "title": "相图：向量场、零倾线与平衡点",
            "description": (
                "平面系统的箭头场 + 两条零倾线（数值追踪的实线与闭式解的虚线）+ 若干条轨迹。"
                "平衡点按线性化分类着色（稳定结点/焦点、鞍点、不稳定），"
                "副轴给零倾线上每一点的 $|f|$ 残差。Hopf 标准型下另画闭式的极限环 $\\sqrt{\\mu}$。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "system",
                    "label": "系统",
                    "type": "enum",
                    "options": ["van_der_pol", "damped_pendulum", "hopf_normal_form"],
                    "default": "van_der_pol",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "mu",
                    "label": "μ（van der Pol / Hopf）",
                    "type": "float",
                    "min": -0.5,
                    "max": 3.0,
                    "step": 0.05,
                    "default": 0.5,
                    "critical": [0.0],
                    "criticalNote": (
                        "$\\mu=0$ 是 Hopf 标准型的分岔点：$\\mu<0$ 原点稳定、$\\mu>0$ 原点失稳并生出半径 $\\sqrt{\\mu}$ 的极限环。"
                        "van der Pol 在 $\\mu=0$ 退化成线性中心（一圈连续的中性闭轨，不是孤立极限环），"
                        "$\\mu>0$ 才有孤立极限环，所以同一个 $0$ 对两个系统的含义**不同**。"
                    ),
                    "beyond": "above",
                },
                {
                    "name": "c",
                    "label": "c（仅阻尼摆：阻尼系数）",
                    "type": "float",
                    "min": 0.05,
                    "max": 2.0,
                    "step": 0.05,
                    "default": 0.3,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "n_traj",
                    "label": "轨迹条数",
                    "type": "int",
                    "min": 4,
                    "max": 16,
                    "default": 8,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "grid_n",
                    "label": "向量场网格（每边）",
                    "type": "int",
                    "min": 9,
                    "max": 31,
                    "default": 19,
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "scrub", "tMax": 12.0, "dt": 0.25, "fps": 20, "label": "时间 t"},
            "axes": [
                {"id": "phase", "title": "相平面（等比例）", "autoscaled": False},
                {"id": "residual", "title": "零倾线上的 |f| 残差", "autoscaled": True, "scale": "log"},
            ],
            "requiredAux": ["residual"],
            "theory": [
                "平衡点由 $f(x)=0$ 给出，其类型由 Jacobian 的特征值决定（实部符号 + 虚部是否为 $0$）。",
                "零倾线 $f_1=0$ 与 $f_2=0$ 的交点是平衡点；两条零倾线在交点处的切向决定鞍点或结点的几何。",
                "Hopf 分岔后极限环的半径按 $\\sqrt{\\mu}$ 增长（标准型），周期趋于 $2\\pi/\\omega$；"
                "van der Pol 的极限环在 $\\mu\\to 0$ 时周期趋于 $2\\pi$、振幅趋于 $2$。",
            ],
            "pitfalls": (
                "零倾线不是「随手画的曲线」：它必须与向量场来自同一次求值，"
                "否则会出现箭头往右、零倾线却说 $\\mathrm{d}x/\\mathrm{d}t=0$ 的自相矛盾。"
                "更隐蔽的是分支连接：van der Pol 的 $\\mathrm{d}y/\\mathrm{d}t=0$ 零倾线在 $x=\\pm 1$ 有竖直渐近线，"
                "只按 $y$ 方向的距离连接会把两支接成一条假曲线（本仓实测踩过），"
                "所以连接判据用归一化二维距离，并把数值分支数与闭式分支数一起写进 meta。"
            ),
        },
    ],
}


# --------------------------------------------------------------------------- #
# 可视化 1：Lorenz 孪生轨道
# --------------------------------------------------------------------------- #
def _viz_butterfly(p: Mapping[str, Any], t: float) -> Frame:
    sigma = float(p["sigma"])
    rho = float(p["rho"])
    beta = float(p["beta"])
    eps = float(10.0 ** float(p["log10_eps"]))
    n_pairs = int(p["n_pairs"])
    az = float(p["azimuth"])
    el = float(p["elevation"])

    system = dyn.flow_system("lorenz", sigma=sigma, rho=rho, beta=beta)
    base = _lorenz_base(n_pairs)
    settled = dyn.integrate_ensemble(system, base, (0.0, _SETTLE_T), _SETTLE_H).final()

    fine = lyap.twin_separation(system, settled, eps=eps, h=_TWIN_H, t_end=_TWIN_T, stride=_TWIN_STRIDE)
    coarse = lyap.twin_separation(
        system, settled, eps=eps, h=_TWIN_H_COARSE, t_end=_TWIN_T, stride=_TWIN_STRIDE_COARSE
    )
    coarse_on_fine = _lerp(coarse.t, coarse.d_mean, fine.t)

    rate, n_fit, r2 = lyap.fit_growth_rate(fine.t, fine.d_mean, lo=1e-6, hi=1.0)
    rate_coarse, n_fit_c, _ = lyap.fit_growth_rate(coarse.t, coarse.d_mean, lo=1e-6, hi=1.0)
    per_pair_raw = np.asarray(
        [lyap.fit_growth_rate(fine.t, fine.d[:, i], lo=1e-6, hi=1.0)[0] for i in range(n_pairs)],
        dtype=float,
    )
    per_pair, pair_mask, n_pair_bad = _finite_or(per_pair_raw, -1.0)
    sat = fine.saturation()
    m_diff = lyap.fit_window(fine.t, fine.d_mean, 1e-6, 1.0)
    rel_raw = np.abs(fine.d_mean[m_diff] - coarse_on_fine[m_diff]) / np.maximum(
        coarse_on_fine[m_diff], np.finfo(float).tiny
    )
    rel_step, rel_mask, n_rel_bad = _finite_or(rel_raw, 0.0)
    rel_max = float(np.max(rel_step)) if rel_step.size else 0.0
    rel_med = float(np.median(rel_step)) if rel_step.size else 0.0
    rel_mean = float(np.mean(rel_step)) if rel_step.size else 0.0
    # 拟合失败（窗口里点太少，例如初值差已经大到一开始就饱和）时不画一条假的实测线：
    # 改用公认指数给的解析线，并把 fit_ok=0 写进 meta。
    fit_ok = bool(n_fit >= 3 and math.isfinite(rate) and math.isfinite(r2))
    rate_line = rate if fit_ok else _LAMBDA_LORENZ
    # 注意：m_diff 是布尔掩码，.size 是掩码长度而不是 True 的个数——
    # 这里要用 np.any（初版写成 m_diff.size，rho<rho_H 时窗口为空、直接 IndexError）。
    win_t = fine.t[m_diff] if bool(np.any(m_diff)) else np.array([fine.t[0], fine.t[-1]])

    vis = fine.d_mean >= _DISPLAY_RES
    t_visible = float(fine.t[int(np.argmax(vis))]) if bool(np.any(vis)) else float("nan")
    t_visible_theory = lyap.resolvable_time(eps, _LAMBDA_LORENZ, _DISPLAY_RES)

    n = n_pairs
    A = fine.Y[:, 0, :]
    B = fine.Y[:, n, :]
    u1, v1 = dyn.project3d(A, az, el)
    u2, v2 = dyn.project3d(B, az, el)

    k = _step_of(_clip(t, 0.0, _TWIN_T) / fine.t[-1] * (fine.t.size - 1), fine.t.size - 1)
    t_now = float(fine.t[k])

    fr = Frame(chapter="3.5", viz="lorenz_butterfly")
    fr.add("t_traj", fine.t, unit="t", label="时间")
    fr.add("traj1_x", A[:, 0], unit="x", label="轨道 1 的 x")
    fr.add("traj1_y", A[:, 1], unit="y", label="轨道 1 的 y")
    fr.add("traj1_z", A[:, 2], unit="z", label="轨道 1 的 z")
    fr.add("traj2_x", B[:, 0], unit="x", label="轨道 2 的 x")
    fr.add("traj2_y", B[:, 1], unit="y", label="轨道 2 的 y")
    fr.add("traj2_z", B[:, 2], unit="z", label="轨道 2 的 z")
    fr.add("traj1_u", u1, unit="px", label="轨道 1 投影 u")
    fr.add("traj1_v", v1, unit="px", label="轨道 1 投影 v")
    fr.add("traj2_u", u2, unit="px", label="轨道 2 投影 u")
    fr.add("traj2_v", v2, unit="px", label="轨道 2 投影 v")
    fr.add("sep_t", fine.t, unit="t", label="分离曲线的时间")
    fr.add("sep_pairs", fine.d, unit="d", label="每对孪生轨道的距离")
    fr.add("sep_mean", fine.d_mean, unit="d", label="平均分离")
    fr.add("sep_coarse", coarse_on_fine, unit="d", label="步长 2h 的平均分离（插到同一时间格点）")
    fr.add("error_stepsize", rel_step, unit="1", label="两种步长的相对差（误差场）")
    t_scale = lyap.resolvable_time(eps, _LAMBDA_LORENZ, max(sat, 1e-12))
    if not math.isfinite(t_scale):
        t_scale = _TWIN_T
    t_scale = min(_TWIN_T, max(t_scale, 1e-9))
    fr.add("fit_x", np.array([float(win_t[0]), float(win_t[-1])]), unit="t", label="拟合窗口")
    fr.add(
        "fit_y",
        _floor(np.array([eps * math.exp(rate_line * float(win_t[0])),
                         eps * math.exp(rate_line * float(win_t[-1]))]), 1e-300),
        unit="d",
        label="拟合出的指数增长线（拟合失败时退回公认指数）",
    )
    fr.add("growth_line_t", np.array([0.0, t_scale]), unit="t",
           label="公认 Lyapunov 指数给出的增长线")
    fr.add(
        "growth_line_d",
        _floor(np.array([eps, eps * math.exp(_LAMBDA_LORENZ * t_scale)]), 1e-300),
        unit="d",
        label="公认 Lyapunov 指数给出的增长线",
    )
    fr.add("floor_line_t", np.array([0.0, fine.t[-1]]), unit="t", label="显示分辨率地板")
    fr.add("floor_line_d", np.array([_DISPLAY_RES, _DISPLAY_RES]), unit="d", label="显示分辨率地板")
    fr.add("float_line_d", np.array([20.0 * float(np.finfo(float).eps), 20.0 * float(np.finfo(float).eps)]),
           unit="d", label="float64 在 |x|~20 上的分辨极限")
    fr.add("mark1_u", np.array([u1[k]]), unit="px", label="当前时刻轨道 1")
    fr.add("mark1_v", np.array([v1[k]]), unit="px", label="当前时刻轨道 1")
    fr.add("mark2_u", np.array([u2[k]]), unit="px", label="当前时刻轨道 2")
    fr.add("mark2_v", np.array([v2[k]]), unit="px", label="当前时刻轨道 2")
    fr.add("mark_t", np.array([t_now]), unit="t", label="当前时刻")
    fr.add("mark_sep", np.array([float(fine.d_mean[k])]), unit="d", label="当前时刻的平均分离")
    fr.add("start_x", settled[:, 0], unit="x", label="孪生对的落点 x")
    fr.add("start_y", settled[:, 1], unit="y", label="孪生对的落点 y")
    fr.add("start_z", settled[:, 2], unit="z", label="孪生对的落点 z")
    fr.add("rate_pairs", per_pair, unit="1/t", label="每对各自的拟合增长率")
    fr.add("rate_pair_valid", pair_mask, unit="1", label="该对的增长率是否可用（0=拟合失败，值填 -1）")
    fr.add("pair_index", np.arange(n_pairs, dtype=float), unit="1", label="孪生对编号")

    fr.set_meta(
        sigma=sigma,
        rho=rho,
        beta=beta,
        rho_hopf=_fin(dyn.lorenz_hopf_rho(sigma, beta), -1.0) if dyn.lorenz_hopf_rho(sigma, beta) is not None else -1.0,
        rho_hopf_exists=bool(dyn.lorenz_hopf_rho(sigma, beta) is not None),
        divergence=dyn.lorenz_divergence(sigma, beta),
        eps=eps,
        log10_eps=float(p["log10_eps"]),
        n_pairs=n_pairs,
        t_now=t_now,
        settle_t=_SETTLE_T,
        settle_h=_SETTLE_H,
        t_end=_TWIN_T,
        h_fine=_TWIN_H,
        h_coarse=_TWIN_H_COARSE,
        stride=_TWIN_STRIDE,
        n_samples=int(fine.t.size),
        rate=_fin(rate, -1.0),
        rate_ok=fit_ok,
        rate_fill=-1.0,
        rate_coarse=_fin(rate_coarse, -1.0),
        rate_diff=_fin(abs(rate - rate_coarse), -1.0),
        rate_r2=_fin(r2, -1.0),
        rate_n_fit=float(n_fit),
        rate_n_fit_coarse=float(n_fit_c),
        rate_pair_min=float(np.min(per_pair)) if n_pairs else 0.0,
        rate_pair_max=float(np.max(per_pair)) if n_pairs else 0.0,
        rate_pair_std=float(np.std(per_pair, ddof=1)) if n_pairs > 1 else 0.0,
        rate_pair_bad=n_pair_bad,
        lambda_reference=_LAMBDA_LORENZ,
        lambda_delta=_fin(rate - _LAMBDA_LORENZ, -1.0),
        saturation=_fin(sat, -1.0),
        t_visible=_fin(t_visible, -1.0),
        t_visible_nan=bool(not math.isfinite(t_visible)),
        t_visible_theory=_fin(t_visible_theory, -1.0),
        display_res=_DISPLAY_RES,
        rel_step_max=rel_max,
        rel_step_median=rel_med,
        rel_step_mean=rel_mean,
        rel_step_bad=n_rel_bad,
        sep_now=float(fine.d_mean[k]),
        sep_min=float(np.min(fine.d_mean)),
        sep_max=float(np.max(fine.d_mean)),
        sep_zero=bool(float(np.max(fine.d_mean)) <= 0.0),
        fit_lo=1e-6,
        fit_hi=1.0,
        azimuth=az,
        elevation=el,
        ref_nan=0,
        overflow=0,
    )
    fr.set_strings(
        note=(
            "两条轨道的初值只差 "
            f"{eps:.3g}（沿 x 方向），其余完全相同；两条曲线在 t≈{_fin(t_visible):.1f} 之前"
            "**逐像素重合**，之后分开。这不是画出来的效果："
            f"实测平均分离在 t=30 时达到 {sat:.2f}，与吸引子直径同量级。"
        ),
        display_policy=(
            "主图是三维相空间的正交投影（原始 x/y/z 数组同帧给出，角度是参数）；"
            "副图是分离距离 d(t) 的对数坐标曲线：细线是每一对孪生轨迹，"
            "粗线是平均，虚线是 h 与 2h 两次积分的对照（两者若差很多，说明看到的是数值噪声）。"
        ),
        nonfinite_policy=_NONFINITE_POLICY,
        twin_protocol=(
            f"孪生实验口径：先以 h={_SETTLE_H:g} 把 {n_pairs} 个随机初值积分 {_SETTLE_T:g} 个时间单位"
            f"落到吸引子上（落点即 base，逐次可复现：种子 {_RNG_SEED}）；"
            f"然后**同一个 base**、同一步长跑两条轨道（第二条只在 x 上加 {eps:.3g}），"
            f"方法与步长完全一致（RK4，h={_TWIN_H:g}，每 {_TWIN_STRIDE} 步采样一次）。"
            "两条曲线共用同一个数值格式，所以「分离」不会被单条的截断误差污染。"
        ),
        fit_protocol=(
            f"增长率拟合：在 1e-6 ≤ d ≤ 1 的窗口内对 ln d 做最小二乘（自写，不是 np.polyfit），"
            f"用点 {n_fit} 个，R²={r2:.4f}。窗口上限取 1 是因为 d 到 1 以后开始饱和，"
            "下限取 1e-6 是因为再小就接近 float64 在 |x|~20 上的分辨极限（见 float_line_d）。"
        ),
        step_size_evidence=(
            f"步长证据：h={_TWIN_H:g} 与 h={_TWIN_H_COARSE:g}（减半）的两条平均分离曲线，"
            f"拟合增长率 {rate:.4f} 与 {rate_coarse:.4f}（差 {abs(rate - rate_coarse):.4f}）；"
            f"在拟合窗口内逐点相对差的中位数 {rel_med:.2%}、最大 {rel_max:.1%}。"
            "两个都报是有原因的：分离曲线**不是光滑的**，它在有限时间里会平台化再突然跳一下"
            f"（实测 t≈8 时 d≈2.0e-5，到 t=10 还只有 2.6e-5），两次积分落在跳变的两侧时"
            "逐点相对差就会很大（最大那一项就是这么来的）。"
            "稳健的读数是**增长率**（差 0.02）与逐点差的中位数；只看最大值会把一个正常的"
            "有限时间涨落误读成数值失稳。"
        ),
        resolution_note=(
            f"「1e-8 的差要多久才看得见」：以显示分辨率 {_DISPLAY_RES:g}（吸引子直径的千分之一）为准，"
            f"实测 t≈{_fin(t_visible):.2f}，用公认 Lyapunov 指数 {_LAMBDA_LORENZ} 的解析估计是 "
            f"t=ln(1e-3/1e-8)/{_LAMBDA_LORENZ}={_fin(t_visible_theory):.2f}。"
            f"float64 在 |x|≈20 上的分辨极限是 20u≈{20.0 * float(np.finfo(float).eps):.2e}，"
            f"初值差 {eps:.1e} 比它大 {eps / (20.0 * float(np.finfo(float).eps)):.1e} 倍，"
            "所以这两条轨道从第一步起就是**不同的数**，不是同一个数的舍入回声。"
        ),
        lambda_compare=(
            f"实测拟合增长率 {rate:.4f}（跨 {n_pairs} 对的散布 "
            f"{float(np.min(per_pair)):.4f}…{float(np.max(per_pair)):.4f}，标准差 "
            f"{float(np.std(per_pair, ddof=1)) if n_pairs > 1 else 0.0:.4f}）；"
            f"公认值 {_LAMBDA_LORENZ}（Sprott 2003 表 5.1）。"
            "有限时间增长率**不是** Lyapunov 指数：它依赖窗口与初值，"
            "所以本帧把窗口、点数、R² 与散布一起给出，让读者自己判断差多少算一致。"
        ),
        **_projection_strings(az, el),
    )
    return fr


# --------------------------------------------------------------------------- #
# 可视化 2：logistic 分岔图
# --------------------------------------------------------------------------- #
def _viz_bifurcation(p: Mapping[str, Any], t: float) -> Frame:
    r_now = float(p["r"])
    n_r = int(p["n_r"])
    n_keep = int(p["n_keep"])
    n_transient = int(p["n_transient"])
    max_period = int(p["max_period"])

    grid = np.linspace(_R_LO, _R_HI, n_r)
    cloud = cont.bifurcation_diagram(grid, n_transient=n_transient, n_keep=n_keep)
    periods, resid_min = cont.detect_periods(grid, max_period=max_period, transient=n_transient)
    resid_cycle = cont.cycle_residuals(grid, periods, transient=n_transient)

    bif_r = np.repeat(grid, n_keep)
    bif_x = cloud.T.reshape(-1)

    xs = np.linspace(0.0, 1.0, 201)
    ys = r_now * xs * (1.0 - xs)
    cob_x, cob_y = cont.cobweb(dyn.logistic(r_now), 0.4, n_steps=120, transient=400)
    cyc_x, cyc_y, cyc_period, cyc_res = _cycle_points(r_now, max_period=max_period)

    n_chaotic = int(np.count_nonzero(periods == 0))
    fr = Frame(chapter="3.5", viz="bifurcation_logistic")
    fr.add("bif_r", bif_r, unit="r", label="参数 r")
    fr.add("bif_x", bif_x, unit="x", label="吸引子上的迭代值")
    fr.add("par_grid", grid, unit="r", label="参数网格")
    fr.add("period_grid", periods.astype(float), unit="p", label="检测出的最小周期")
    fr.add("residual_cycle", _floor(resid_cycle, 1e-18), unit="1", label="周期闭合残差 |f^p(x)-x|")
    fr.add("residual_min", _floor(resid_min, 1e-18), unit="1", label="最接近闭合的残差")
    fr.add("map_x", xs, unit="x", label="映射自变量 x")
    fr.add("map_y", ys, unit="f(x)", label="logistic 映射")
    fr.add("cobweb_x", cob_x, unit="x", label="蛛网图 x")
    fr.add("cobweb_y", cob_y, unit="y", label="蛛网图 y")
    fr.add("cycle_x", cyc_x, unit="x", label="检测到的周期轨（x）")
    fr.add("cycle_y", cyc_y, unit="y", label="检测到的周期轨（y）")
    fr.add("return_map_x", xs, unit="x", label="对角线 x")
    fr.add("return_map_y", xs, unit="y", label="对角线 y")
    fr.add("mark_r", np.array([r_now]), unit="r", label="当前参数")
    fr.add("mark_period", np.array([float(cyc_period)]), unit="p", label="当前周期")
    fr.add("critical_r", np.asarray(_R_DOUBLING + (_R_WINDOW3, _R_ACCUM)), unit="r",
           label="临界参数（倍周期点 / 3 周期窗口起点 / 聚点）")

    fr.set_meta(
        r_now=r_now,
        period_now=float(cyc_period),
        residual_now=_fin(cyc_res, -1.0),
        residual_finite=bool(math.isfinite(cyc_res)),
        n_r=n_r,
        n_keep=n_keep,
        n_transient=n_transient,
        max_period=max_period,
        dr=(_R_HI - _R_LO) / (n_r - 1),
        n_points=int(bif_r.size),
        n_chaotic=n_chaotic,
        n_periodic=int(n_r - n_chaotic),
        frac_chaotic=n_chaotic / float(n_r),
        r_doubling1=_R_DOUBLING[0],
        r_doubling2=_R_DOUBLING[1],
        r_doubling3=_R_DOUBLING[2],
        r_doubling4=_R_DOUBLING[3],
        r_window3=_R_WINDOW3,
        r_accumulation=_R_ACCUM,
        residual_min=float(np.min(resid_cycle)),
        residual_max=float(np.max(resid_cycle)),
        azimuth=0.0,
        elevation=0.0,
        ref_nan=0,
        overflow=0,
    )
    fr.set_strings(
        note=(
            f"参数网格 {n_r} 个点（Δr={(_R_HI - _R_LO) / (n_r - 1):.2e}），每个参数丢掉前 "
            f"{n_transient} 次迭代（暂态），留下 {n_keep} 个点画成一条竖线："
            f"共 {bif_r.size} 个点、检测到 {n_r - n_chaotic} 个参数落在周期窗口里"
            f"（占比 {1.0 - n_chaotic / float(n_r):.1%}）。"
            "竖线在一个参数上是一串点而不是一个点，这正是「周期 vs 混沌」的图形差别。"
        ),
        display_policy=(
            "主图：横轴 r、纵轴迭代值，点按检测到的最小周期着色（0 = 混沌，用灰色）；"
            f"每个点的周期 = repeat(period_grid, {n_keep})（同帧给出 period_grid，未单独发一份重复数组）。"
            "副图（对数轴）：每个参数的周期闭合残差 |f^p(x)−x|——周期窗口里是 1e-15 量级，"
            "混沌区是 1e-2 量级，两类点在同一个图上是**分开的**，不靠肉眼猜。"
            "插图：当前 r 的映射曲线与蛛网图（迭代的几何意义）。"
        ),
        nonfinite_policy=_NONFINITE_POLICY,
        resolution_note=(
            f"分辨率不足会把周期窗口画糊：窗口宽度与 Δr={(_R_HI - _R_LO) / (n_r - 1):.2e} 同量级时，"
            "整串倍周期级联会被压成一条灰带。3 周期窗口宽约 0.013"
            f"（{_R_WINDOW3:.6f} 到 ≈3.8415），窗口内部的倍周期级联宽度只有 1e-2…1e-4 量级；"
            f"当前 Δr 下窗口里只有约 {int(0.013 / ((_R_HI - _R_LO) / (n_r - 1)))} 个参数点。"
            "把 n_r 拖大到 1200 能看到窗口内部再分岔，但 `n_keep` 不够时竖直方向又会糊。"
        ),
        period_protocol=(
            f"周期是**检测**出来的，不是画出来的：从 x0=0.4 迭代 {n_transient} 次后，"
            f"逐个测试 |f^p(x)−x| ≤ 1e-9（p ≤ {max_period}），第一个闭合的 p 就是最小周期。"
            "混沌轨道在 32 步内不闭合，此时给出「最接近闭合」的残差作为量化读数。"
            "判据里**没有**用「乘子是否小于 1」：窗口里同时存在稳定与不稳定周期轨。"
        ),
        critical_note=(
            f"临界值：r1={_R_DOUBLING[0]:g}（1 周期失稳）、r2=1+√6={_R_DOUBLING[1]:.12f}、"
            f"r3={_R_DOUBLING[2]:.10f}、r4={_R_DOUBLING[3]:.10f}、"
            f"3 周期窗口起点 1+√8={_R_WINDOW3:.12f}、聚点 r∞≈{_R_ACCUM}。"
            "前两个是本仓二分实测值与闭式逐位相符的（见 asserts.yaml），其余取自标准文献值。"
        ),
        cobweb_note=(
            f"蛛网图从 x0=0.4 迭代 {400} 次暂态后画 120 步：当前 r={r_now:.4f} 的最小周期是 "
            f"{cyc_period}（0 表示混沌），闭合残差 {_fin(cyc_res):.3e}。"
            "混沌时蛛网填满一段区间而不是落进方框——这是「周期」与「混沌」最直观的几何差别。"
        ),
    )
    return fr


def _cycle_points(r: float, *, max_period: int = 32, transient: int = 4000) -> tuple[np.ndarray, np.ndarray, int, float]:
    """当前参数下轨道落在哪个周期上：返回 ``(x, y, p, 残差)``（混沌时给最近的迭代点列）。"""
    g = dyn.logistic(r)
    x = 0.4
    for _ in range(int(transient)):
        x = r * x * (1.0 - x)
    p, res = cont.detect_period(g, x, max_period=max_period, transient=0)
    if p == 0:
        xs = []
        y = x
        for _ in range(64):
            y = r * y * (1.0 - y)
            xs.append(y)
        arr = np.asarray(xs, dtype=float)
        return arr, g(arr), 0, res
    xs = []
    y = x
    for _ in range(p):
        xs.append(y)
        y = r * y * (1.0 - y)
    arr = np.asarray(xs, dtype=float)
    return arr, g(arr), p, res


# --------------------------------------------------------------------------- #
# 可视化 3：Lyapunov 指数叠在分岔图上
# --------------------------------------------------------------------------- #
def _viz_lyapunov(p: Mapping[str, Any], t: float) -> Frame:
    n_r = int(p["n_r"])
    n_iter = int(p["n_iter"])
    n_transient = int(p["n_transient"])
    n_block = int(p["n_block"])
    r_probe = float(p["r_probe"])

    grid = np.linspace(_R_LO, _R_HI, n_r)
    lam, sem = lyap.lyapunov_curve(grid, n_iter=n_iter, transient=n_transient, n_block=n_block)
    cloud = cont.bifurcation_diagram(grid, n_transient=n_transient, n_keep=16)

    ana_r = np.array([2.5, 3.2, 4.0])
    ana_lam = np.array([lyap.logistic_analytic_lyapunov(v) for v in ana_r])
    # 三个解析对拍点用固定次数（20000）：r=4 上误差已是 1e-6 量级，
    # 再往上加迭代只是把有限时间涨落的随机游走走得更久，不改变"和 ln2 差多少"这个结论。
    n_check = min(n_iter, 20000)
    mea_lam = np.array([lyap.logistic_lyapunov(float(v), n_iter=n_check, transient=n_transient, n_block=n_block)[0]
                        for v in ana_r])
    probe_lam, probe_sem = lyap.logistic_lyapunov(
        r_probe, n_iter=n_iter, transient=n_transient, n_block=n_block
    )
    probe_period, probe_res = cont.detect_period(
        dyn.logistic(r_probe), 0.4, max_period=32, transient=n_transient, tol=1e-9
    )

    n_pos = int(np.count_nonzero(lam > 0.0))
    fr = Frame(chapter="3.5", viz="lyapunov_overlay")
    fr.add("bif_r", np.repeat(grid, 16), unit="r", label="参数 r")
    fr.add("bif_x", cloud.T.reshape(-1), unit="x", label="吸引子（淡化显示）")
    fr.add("lyap_r", grid, unit="r", label="参数 r")
    fr.add("lyap_lambda", lam, unit="1/t", label="Lyapunov 指数")
    fr.add("error_sem", _floor(sem, 1e-18), unit="1/t", label="分块标准误（误差场）")
    fr.add("sem_band_lo", lam - sem, unit="1/t", label="λ 下界")
    fr.add("sem_band_hi", lam + sem, unit="1/t", label="λ 上界")
    fr.add("zero_x", np.array([_R_LO, _R_HI]), unit="r", label="λ=0")
    fr.add("zero_y", np.array([0.0, 0.0]), unit="1/t", label="λ=0")
    fr.add("ln2_x", np.array([_R_LO, _R_HI]), unit="r", label="ln2")
    fr.add("ln2_y", np.array([math.log(2.0), math.log(2.0)]), unit="1/t", label="ln2")
    fr.add("analytic_r", ana_r, unit="r", label="有闭式解的参数")
    fr.add("analytic_lambda", ana_lam, unit="1/t", label="解析 Lyapunov 指数")
    fr.add("measured_lambda", mea_lam, unit="1/t", label="同参数下的实测值")
    fr.add("mark_r", np.array([r_probe]), unit="r", label="当前参数")
    fr.add("mark_lambda", np.array([probe_lam]), unit="1/t", label="当前参数的 λ")

    fr.set_meta(
        r_probe=r_probe,
        lambda_probe=probe_lam,
        sem_probe=probe_sem,
        period_probe=float(probe_period),
        residual_probe=_fin(probe_res, -1.0),
        n_r=n_r,
        n_iter=n_iter,
        n_transient=n_transient,
        n_block=n_block,
        dr=(_R_HI - _R_LO) / (n_r - 1),
        n_positive=n_pos,
        frac_positive=n_pos / float(n_r),
        lambda_max=float(np.max(lam)),
        lambda_argmax=float(grid[int(np.argmax(lam))]),
        sem_max=float(np.max(sem)),
        analytic_r2_5=lyap.logistic_analytic_lyapunov(2.5),
        measured_r2_5=float(mea_lam[0]),
        analytic_r3_2=lyap.logistic_analytic_lyapunov(3.2),
        measured_r3_2=float(mea_lam[1]),
        analytic_r4=math.log(2.0),
        measured_r4=float(mea_lam[2]),
        delta_r4=float(mea_lam[2]) - math.log(2.0),
        r_accumulation=_R_ACCUM,
        azimuth=0.0,
        elevation=0.0,
        ref_nan=0,
        overflow=0,
    )
    fr.set_strings(
        note=(
            "λ(r) 的公式是 λ=lim (1/N)Σln|f'(x_n)|：混沌区为正（相邻点指数分开），"
            "周期窗口内为负（收敛到周期轨）。"
            f"当前网格上 {n_pos}/{n_r} 个参数 λ>0；最大 λ≈{float(np.max(lam)):.4f}"
            f"（在 r≈{float(grid[int(np.argmax(lam))]):.4f} 处，接近理论值 ln2={math.log(2.0):.6f}）。"
        ),
        display_policy=(
            "主图：灰色细点是分岔图（同一套参数网格，用来对齐周期窗口），"
            "深色曲线是 λ(r)，浅色带是分块标准误；虚线是 λ=0 与 ln2。"
            "副图（对数轴）：标准误本身——它在哪里大，哪里的 λ 就不能当准数用。"
        ),
        nonfinite_policy=_NONFINITE_POLICY,
        estimate_protocol=(
            f"估计口径：丢弃 {n_transient} 次暂态，再迭代 {n_iter} 次；"
            f"分成 {n_block} 块各求均值，块间标准差除以 √{n_block} 作为标准误。"
            "块标准误度量的是**有限时间涨落**，不是舍入误差——它随迭代次数的增加变慢"
            "（周期窗口附近尤其慢：乘子趋近 ±1 时收敛是代数级的）。"
        ),
        analytic_check=(
            f"三个有闭式的点：r=2.5 解析 {lyap.logistic_analytic_lyapunov(2.5):.8f} / 实测 {mea_lam[0]:.8f}；"
            f"r=3.2（2 周期，乘子 4+2r−r²=0.16）解析 {lyap.logistic_analytic_lyapunov(3.2):.8f} / "
            f"实测 {mea_lam[1]:.8f}；r=4 解析 ln2={math.log(2.0):.8f} / 实测 {mea_lam[2]:.8f}"
            f"（差 {abs(float(mea_lam[2]) - math.log(2.0)):.2e}；"
            f"这三个点用 {n_check} 次迭代，与曲线上的 n_iter={n_iter} 分开记）。"
        ),
        r2_special_note=(
            "r=2 处不动点是超稳定的（f'(1/2)=0），Lyapunov 指数是 −∞；"
            "本图的参数范围从 2.4 开始，所以那条 −∞ 不出现在图上——"
            "不是被截断，是本帧根本不画它（画一个「很负的有限值」才是撒谎）。"
        ),
    )
    return fr


# --------------------------------------------------------------------------- #
# 可视化 4：Poincaré 截面
# --------------------------------------------------------------------------- #
def _viz_poincare(p: Mapping[str, Any], t: float) -> Frame:
    kind = str(p["system"])
    rho = float(p["rho"])
    sigma = float(p["sigma"])
    c = float(p["c"])
    n_cross = int(p["n_cross"])
    az = float(p["azimuth"])
    el = float(p["elevation"])
    t_end = _POINCARE_T_END

    if kind == "lorenz":
        system = dyn.flow_system("lorenz", sigma=sigma, rho=rho, beta=8.0 / 3.0)
        y0 = np.array([1.0, 0.0, 20.0])
        plane_index, level = 2, rho - 1.0
        axes_labels = ("x", "y", "z")
    else:
        system = dyn.flow_system("rossler", a=0.2, b=0.2, c=c)
        y0 = np.array([1.0, 1.0, 0.0])
        plane_index, level = 0, 0.0
        axes_labels = ("x", "y", "z")

    t_now = _clip(float(t), 0.0, t_end)
    sec = dyn.poincare_crossings(
        system,
        y0,
        plane_index=plane_index,
        level=level,
        direction=1,
        t_span=(0.0, max(t_now, 1e-9)),
        h=0.01,
        max_crossings=n_cross,
        refine=3,
    )
    orbit = sec.orbit_y
    ou, ov = dyn.project3d(orbit, az, el)
    if sec.points.size:
        su, sv = dyn.project3d(sec.points, az, el)
        ru = sec.points[:, (plane_index + 1) % 3]
        rv = sec.points[:, (plane_index + 2) % 3]
        rx = sec.points[:, (plane_index + 1) % 3]
        ry = sec.points[:, (plane_index + 2) % 3]
    else:
        su = sv = ru = rv = rx = ry = np.zeros(0)

    fr = Frame(chapter="3.5", viz="poincare_section")
    fr.add("orbit_t", sec.orbit_t, unit="t", label="轨道时间")
    fr.add("orbit_x", orbit[:, 0], unit=axes_labels[0], label="轨道 x")
    fr.add("orbit_y", orbit[:, 1], unit=axes_labels[1], label="轨道 y")
    fr.add("orbit_z", orbit[:, 2], unit=axes_labels[2], label="轨道 z")
    fr.add("orbit_u", ou, unit="px", label="轨道投影 u")
    fr.add("orbit_v", ov, unit="px", label="轨道投影 v")
    fr.add("section_x", sec.points[:, 0] if sec.points.size else np.zeros(0), unit="x", label="截面点 x")
    fr.add("section_y", sec.points[:, 1] if sec.points.size else np.zeros(0), unit="y", label="截面点 y")
    fr.add("section_z", sec.points[:, 2] if sec.points.size else np.zeros(0), unit="z", label="截面点 z")
    fr.add("section_u", su, unit="px", label="截面点投影 u")
    fr.add("section_v", sv, unit="px", label="截面点投影 v")
    fr.add("section_inplane_x", ru, unit="u", label="截面内坐标（平面内第一轴）")
    fr.add("section_inplane_y", rv, unit="v", label="截面内坐标（平面内第二轴）")
    fr.add("section_t", sec.t, unit="t", label="穿越时刻")
    fr.add("residual_plane", _floor(sec.plane_residual, 1e-18), unit="1", label="离平面的残差（误差场）")
    fr.add("residual_shift", _floor(sec.refine_shift, 1e-18), unit="1", label="线性插值点到精化点的位移")
    fr.add("lin_x", sec.lin_points[:, 0] if sec.points.size else np.zeros(0), unit="x", label="线性插值点 x")
    fr.add("lin_y", sec.lin_points[:, 1] if sec.points.size else np.zeros(0), unit="y", label="线性插值点 y")
    fr.add("lin_z", sec.lin_points[:, 2] if sec.points.size else np.zeros(0), unit="z", label="线性插值点 z")
    fr.add("return_x", rx[:-1], unit="x", label="第 k 次穿越（平面内坐标）")
    fr.add("return_y", ry[1:], unit="y", label="第 k+1 次穿越（平面内坐标）")
    fr.add("plane_corner_x", np.array([float(np.min(orbit[:, 0])), float(np.max(orbit[:, 0]))]),
           unit="x", label="平面框")
    fr.add("plane_corner_y", np.array([float(np.min(orbit[:, 1])), float(np.max(orbit[:, 1]))]),
           unit="y", label="平面框")

    fr.set_meta(
        system_code=float(dyn.FLOW_NAMES.index(kind)),
        rho=rho,
        sigma=sigma,
        c=c,
        plane_index=float(plane_index),
        level=level,
        direction=1.0,
        h=0.01,
        t_now=t_now,
        t_end=t_end,
        n_cross=int(sec.t.size),
        n_cross_max=n_cross,
        n_steps=int(sec.n_steps),
        n_f_evals=int(sec.n_f_evals),
        mean_return=_fin(sec.mean_return_time(), -1.0),
        max_residual=float(np.max(sec.plane_residual)) if sec.plane_residual.size else 0.0,
        max_shift=float(np.max(sec.refine_shift)) if sec.refine_shift.size else 0.0,
        has_crossings=bool(sec.t.size > 0),
        blowup=bool(sec.blowup),
        azimuth=az,
        elevation=el,
        ref_nan=0,
        overflow=0,
    )
    fr.set_strings(
        note=(
            f"{'Lorenz' if kind == 'lorenz' else 'Rössler'} 系统的轨道被平面 "
            f"{axes_labels[plane_index]}={level:g} 按**方向**截取：只收由负变正的那次穿越"
            f"（双向都收会让返回映射变成双值的重影）。"
            f"到 t={t_now:.2f} 为止收到 {int(sec.t.size)} 个点，平均返回时间 "
            f"{_fin(sec.mean_return_time()):.3f}。"
        ),
        display_policy=(
            "主图：三维轨道的正交投影（原始 x/y/z 同帧给出）+ 截面平面上的落点；"
            "副图：返回映射（第 k 次穿越 → 第 k+1 次穿越）——截面点若是 1 维曲线，"
            "返回映射就是 1 维映射，这正是「用低维映射研究高维流」的那一步。"
        ),
        nonfinite_policy=_NONFINITE_POLICY,
        residual_note=(
            f"每个截面点都带两个残差：离平面的距离（Newton 精化后最大 "
            f"{float(np.max(sec.plane_residual)) if sec.plane_residual.size else float('nan'):.2e}）"
            f"与「线性插值点 → 精化点」的位移（最大 "
            f"{float(np.max(sec.refine_shift)) if sec.refine_shift.size else float('nan'):.2e}，"
            "单位是相空间长度，不是无量纲量）。"
            "后者是**定位误差的实测**：步长 h=0.01 时它约 0.06，与吸引子直径（~30）相比是 0.2%。"
        ),
        protocol=(
            "口径：RK4 定步长 h=0.01；穿越判定用方向敏感的符号变化；"
            "定位先做线性插值（O(h²)），再以局部导数 f(y)[平面分量] 做 3 次 Newton 精化"
            "（用整步平均斜率当割线只能做到 3e-3，实测收敛比约 1/3.5——这是本模块的实测记录）。"
        ),
        **_projection_strings(az, el),
    )
    return fr


# --------------------------------------------------------------------------- #
# 可视化 5：相图 / 向量场 / 零倾线
# --------------------------------------------------------------------------- #
def _viz_phase(p: Mapping[str, Any], t: float) -> Frame:
    kind = str(p["system"])
    mu = float(p["mu"])
    c = float(p["c"])
    n_traj = int(p["n_traj"])
    grid_n = int(p["grid_n"])
    t_end = _PHASE_T_END

    if kind == "van_der_pol":
        system = dyn.flow_system("van_der_pol", mu=mu)
        fixed = [np.array([0.0, 0.0])]
    elif kind == "damped_pendulum":
        system = dyn.flow_system("damped_pendulum", c=c)
        fixed = [np.array([0.0, 0.0]), np.array([math.pi, 0.0])]
    else:
        system = dyn.flow_system("hopf_normal_form", mu=mu)
        fixed = [np.array([0.0, 0.0])]

    (xlo, xhi), (ylo, yhi) = system.window[0], system.window[1]
    gx = np.linspace(xlo, xhi, grid_n)
    gy = np.linspace(ylo, yhi, grid_n)
    U, V = dyn.vector_field(system, gx, gy)
    ggx, ggy = np.meshgrid(gx, gy, indexing="ij")

    ncl = dyn.nullcline_branches(system, nx=max(81, grid_n * 8), ny=max(81, grid_n * 8), min_points=4)
    ana = dyn.analytic_nullclines(system)
    nx_, ny_, ns_ = _pack_segments(ncl.get("x", []))
    nyx, nyy, nys = _pack_segments(ncl.get("y", []))
    ax_, ay_, as_ = _pack_segments(ana.get("x", []))
    ayx, ayy, ays = _pack_segments(ana.get("y", []))
    res_nx = _nullcline_residual(system, nx_, ny_, 0)
    res_ny = _nullcline_residual(system, nyx, nyy, 1)

    fx = np.array([q[0] for q in fixed])
    fy = np.array([q[1] for q in fixed])
    reports = [dyn.fixed_point_report(system, q) for q in fixed]
    kinds = np.array([float(_KIND_CODES.get(rep.kind, -1)) for rep in reports])
    res_fix = np.array([_fin(rep.residual, 0.0) for rep in reports])
    eig_max = np.array([_fin(np.max(rep.eigenvalues.real), 0.0) for rep in reports])

    t_now = _clip(float(t), 0.0, t_end)
    ics = _phase_ics(kind, system, n_traj)
    run = dyn.integrate_ensemble(system, ics, (0.0, max(t_now, 1e-9)), 0.005, stride=4)
    k = run.Y.shape[0] - 1
    radius, radius_theory = _limit_metrics(kind, system, run, mu)

    fr = Frame(chapter="3.5", viz="phase_portrait")
    fr.add("vf_x", ggx.reshape(-1), unit="x", label="向量场格点 x")
    fr.add("vf_y", ggy.reshape(-1), unit="y", label="向量场格点 y")
    fr.add("vf_u", U.reshape(-1), unit="dx/dt", label="向量场 u 分量")
    fr.add("vf_v", V.reshape(-1), unit="dy/dt", label="向量场 v 分量")
    fr.add("null_x_x", nx_, unit="x", label="数值 dx/dt=0 零倾线")
    fr.add("null_x_y", ny_, unit="y", label="数值 dx/dt=0 零倾线")
    fr.add("null_x_seg", ns_, unit="1", label="分段号")
    fr.add("null_y_x", nyx, unit="x", label="数值 dy/dt=0 零倾线")
    fr.add("null_y_y", nyy, unit="y", label="数值 dy/dt=0 零倾线")
    fr.add("null_y_seg", nys, unit="1", label="分段号")
    fr.add("analytic_x_x", ax_, unit="x", label="闭式 dx/dt=0 零倾线")
    fr.add("analytic_x_y", ay_, unit="y", label="闭式 dx/dt=0 零倾线")
    fr.add("analytic_x_seg", as_, unit="1", label="分段号")
    fr.add("analytic_y_x", ayx, unit="x", label="闭式 dy/dt=0 零倾线")
    fr.add("analytic_y_y", ayy, unit="y", label="闭式 dy/dt=0 零倾线")
    fr.add("analytic_y_seg", ays, unit="1", label="分段号")
    fr.add("residual_null", np.concatenate([res_nx, res_ny]) if res_nx.size + res_ny.size else np.zeros(0),
           unit="1", label="零倾线上的 |f| 残差（误差场）")
    fr.add("residual_fixed", res_fix, unit="1", label="平衡点上的 |f| 残差")
    fr.add("fixed_x", fx, unit="x", label="平衡点 x")
    fr.add("fixed_y", fy, unit="y", label="平衡点 y")
    fr.add("fixed_kind", kinds, unit="1", label="平衡点分类编码")
    fr.add("fixed_max_real", eig_max, unit="1/t", label="平衡点最大特征值实部")
    fr.add("traj_t", run.t, unit="t", label="轨迹时间")
    fr.add("traj_x", run.Y[:, :, 0], unit="x", label="轨迹 x")
    fr.add("traj_y", run.Y[:, :, 1], unit="y", label="轨迹 y")
    fr.add("mark_x", run.Y[k, :, 0], unit="x", label="当前时刻轨迹点 x")
    fr.add("mark_y", run.Y[k, :, 1], unit="y", label="当前时刻轨迹点 y")
    fr.add("start_x", ics[:, 0], unit="x", label="初始点 x")
    fr.add("start_y", ics[:, 1], unit="y", label="初始点 y")
    fr.add("traj_index", np.arange(n_traj, dtype=float), unit="1", label="轨迹编号")
    if kind == "hopf_normal_form":
        th = np.linspace(0.0, 2.0 * math.pi, 361)
        rr = math.sqrt(mu) if mu > 0.0 else 0.0
        fr.add("cycle_x", rr * np.cos(th), unit="x", label="闭式极限环 x（半径 √μ）")
        fr.add("cycle_y", rr * np.sin(th), unit="y", label="闭式极限环 y")

    n_stable = sum(1 for rep in reports if rep.kind.startswith("stable"))
    n_unstable = sum(1 for rep in reports if rep.kind.startswith("unstable"))
    n_saddle = sum(1 for rep in reports if rep.kind in ("saddle", "saddle_focus"))
    fr.set_meta(
        system_code=float(dyn.FLOW_NAMES.index(kind)) if kind in dyn.FLOW_NAMES else -1.0,
        mu=mu,
        c=c,
        mu_critical=0.0,
        n_traj=n_traj,
        grid_n=grid_n,
        t_now=t_now,
        t_end=t_end,
        x_lo=xlo, x_hi=xhi, y_lo=ylo, y_hi=yhi,
        n_stable=float(n_stable),
        n_unstable=float(n_unstable),
        n_saddle=float(n_saddle),
        max_residual_null=float(np.max(np.concatenate([res_nx, res_ny]))) if res_nx.size + res_ny.size else 0.0,
        max_residual_fixed=float(np.max(res_fix)) if res_fix.size else 0.0,
        limit_radius=radius,
        limit_radius_theory=(radius_theory if math.isfinite(radius_theory) else None),
        limit_radius_theory_null=bool(not math.isfinite(radius_theory)),
        n_null_branches=float(len(ncl.get("x", [])) + len(ncl.get("y", []))),
        n_analytic_branches=float(len(ana.get("x", [])) + len(ana.get("y", []))),
        n_f_evals=float(run.n_f_evals),
        blowup=bool(run.blowup),
        azimuth=0.0,
        elevation=0.0,
        ref_nan=0,
        overflow=0,
    )
    fr.set_strings(
        note=(
            f"{kind} 的相图：箭头是向量场（{grid_n}×{grid_n} 个格点），"
            "蓝/绿曲线是两条零倾线（数值追踪），虚线是同一批零倾线的闭式解。"
            f"平衡点 {len(fixed)} 个：稳定 {n_stable}、不稳定 {n_unstable}、鞍点 {n_saddle}。"
        ),
        display_policy=(
            "主图：向量场（箭头长度按格点间距归一，只表示方向）+ 零倾线 + 若干条轨迹，"
            "轨迹随时间推进（时间轴是真实时间）。"
            "副图（对数轴）：零倾线上的 |f| 残差——数值追踪出来的每一点离「f=0」有多远。"
        ),
        nonfinite_policy=_NONFINITE_POLICY,
        nullcline_protocol=(
            "零倾线的数值追踪口径：在窗口网格上逐列找 f 分量的符号变化，"
            "每个变化里二分 60 次把根压到机器精度，再按归一化二维距离连成分支"
            "（只比较 y 差会把跨渐近线的两支接成一条假曲线——van der Pol 的 "
            "dy/dt=0 零倾线在 x=±1 有竖直渐近线，本仓实测踩过这个坑）。"
            f"当前数值分支 {len(ncl.get('x', [])) + len(ncl.get('y', []))} 条、"
            f"闭式分支 {len(ana.get('x', [])) + len(ana.get('y', []))} 条，"
            f"数值点上的最大 |f|={float(np.max(np.concatenate([res_nx, res_ny]))) if res_nx.size + res_ny.size else 0.0:.2e}。"
        ),
        limit_cycle_note=(
            (
                f"极限环：Hopf 标准型的闭式半径 √μ={math.sqrt(mu):.6f}，"
                f"实测轨迹尾段平均半径 {radius:.6f}（差 {abs(radius - math.sqrt(mu)):.2e}）。"
            )
            if kind == "hopf_normal_form" and mu > 0.0
            else (
                f"μ≤0：原点是稳定焦点、没有极限环。轨迹尾段平均半径 {radius:.4f} 不是在量极限环，"
                "是在量它收敛到原点附近的速度（收敛率 = 特征值实部 μ）。"
            )
            if kind == "hopf_normal_form"
            else (
                f"阻尼摆没有极限环：轨迹尾段平均半径 {radius:.4f} 是它耗散到原点附近的读数，"
                "衰减率由阻尼 c 决定（能量 E=½y²+c²(1−cos x) 单调下降）。"
            )
            if kind == "damped_pendulum"
            else (
                f"van der Pol 的极限环不是圆：轨迹尾段平均半径 {radius:.4f} 只是量级读数，"
                "振幅、周期与 μ→0 的极限（振幅 2、周期 2π）由 asserts.yaml 的 A/C 类断言给出。"
            )
        ),
        kind_codes="平衡点分类编码：" + "、".join(
            f"{code}={dyn.KIND_NAMES[key]}" for key, code in sorted(dyn.KIND_CODES.items(), key=lambda kv: kv[1])
        ),
    )
    return fr


def _nullcline_residual(system: dyn.FlowSystem, x: Any, y: Any, comp: int) -> np.ndarray:
    r"""零倾线点上的 :math:`|f_{\text{comp}}|`（**对应那一分量**，不是矢量长度）。

    这一点很容易写错：dx/dt=0 的零倾线上只有 f1 是 0，f2 一般不为 0，
    所以拿 :math:`\sqrt{f_1^2+f_2^2}` 当残差会得到 O(1) 的数（本仓实测 van der Pol 上是 4.44），
    看着像"零倾线全错了"。残差必须取**定义这条零倾线的那一个分量**。
    """
    xs = np.asarray(x, dtype=float).reshape(-1)
    ys = np.asarray(y, dtype=float).reshape(-1)
    if xs.size == 0:
        return np.zeros(0)
    pts = np.stack([xs, ys], axis=1)
    F = np.asarray(system.rhs(0.0, pts), dtype=float).reshape(-1, 2)
    return np.abs(F[:, int(comp)])


def _phase_ics(kind: str, system: dyn.FlowSystem, n_traj: int) -> np.ndarray:
    """相图上的初始点：写死的圆形分布（不用随机数，帧必须逐次可复现）。"""
    if kind == "damped_pendulum":
        return _circle_ics((0.0, 0.0), 1.2, n_traj)
    if kind == "hopf_normal_form":
        return _circle_ics((0.0, 0.0), 0.9, n_traj)
    (xlo, xhi), _ = system.window[0], system.window[1]
    return _circle_ics((0.5 * (xlo + xhi), 0.0), 1.1, n_traj)


def _limit_metrics(kind: str, system: dyn.FlowSystem, run: dyn.EnsembleRun, mu: float) -> tuple[float, float]:
    """尾段平均半径（极限环的量级读数）与它的闭式值（只有 Hopf 标准型有）。"""
    Z = run.Y
    tail = Z[max(0, int(0.8 * Z.shape[0])):, 0, :]
    r = float(np.mean(np.sqrt(np.sum(tail * tail, axis=1))))
    if kind == "hopf_normal_form" and mu > 0.0:
        return r, math.sqrt(mu)
    return r, float("nan")


# --------------------------------------------------------------------------- #
# 分发
# --------------------------------------------------------------------------- #
_DISPATCH = {
    "lorenz_butterfly": _viz_butterfly,
    "bifurcation_logistic": _viz_bifurcation,
    "lyapunov_overlay": _viz_lyapunov,
    "poincare_section": _viz_poincare,
    "phase_portrait": _viz_phase,
}


def compute(viz_id: str, params: Mapping[str, Any], t: float) -> Frame:
    """按 ``viz_id`` 分发到具体的帧构造器（与 1.2/3.2 章同签名）。"""
    fn = _DISPATCH.get(str(viz_id))
    if fn is None:
        raise KeyError(f"未知的可视化 id：{viz_id!r}（本 chapter 有 {sorted(_DISPATCH)}）")
    merged = dict(params or {})
    for spec in MANIFEST["vizzes"]:
        if spec["id"] == str(viz_id):
            for par in spec["params"]:
                merged.setdefault(par["name"], par["default"])
            break
    return fn(merged, float(t))
