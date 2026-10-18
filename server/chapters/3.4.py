"""3.4 章：边值问题与谱方法（4 个可视化）。

帧只做搬运：所有数学量都来自 :mod:`numcore.ode.bvp`，本文件不写数学。
四个可视化的分工是"从打靶到微分矩阵"：

1. ``shooting_scan`` —— **招牌**：一族初始斜率的轨迹（同一灵敏度 S，两种命运）。
   时间轴是斜率序号 j：s_j = s_ref·(1 + span·(2j/(m-1) - 1))，扫过去时
   "打靶函数在右端被抵消掉多少位"看得见。副轴是打靶残差 |F(s_j)|。
2. ``convergence_rate`` —— 有限差分与谱配置的误差衰减对比：代数（N^-p）vs 指数
   （半对数直线），外加两条**实测**拟合线与舍入地板。时间轴无（纯参数图）。
3. ``node_clustering`` —— Chebyshev 节点的端点聚集：节点、间距、密度、以及
   "节点上插值误差为 0、节点之间才是误差"的那条曲线。时间轴无。
4. ``diff_matrix_structure`` —— 谱微分矩阵的结构热力图 + 行和残差 + 奇异值谱：
   它稠密、奇异（零空间是常数）、||D_N||_inf = N^2，**不是用来解方程组的**。

帧契约（与 1.2/3.2 章一致）：``meta`` 只放有限标量（``allow_nan=False``，非有限值一律
换成哨兵 + 显式布尔标志），字符串走 ``strings``，数组走 ``add``。本章所有"指数收敛"
的结论都必须带**实测半对数斜率 + 舍入地板 + 撞上地板的 N**，所有"阶"的结论都必须带
**拟合窗口**——这三样写进 ``meta`` 与 ``strings``，并由 ``tests/3.4`` 逐帧核对。
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from numcore.ode import bvp
from server.nlf import Frame

__all__ = ["MANIFEST", "compute"]

#: 单位舍入（与 1.2/3.2 章同口径）；打靶残差的理论地板 = 这个数乘灵敏度。
_ROUNDOFF = float(np.finfo(float).eps) / 2.0
U_FLOOR = _ROUNDOFF

#: 轨迹/误差曲线的显示上限：超过就截断**显示**并配掩码，真值写在 meta 里。
_DISPLAY_CEIL = 1.0e6

#: 统一的口径声明：任何"非有限值"都必须显式占位 + 掩码，不许静默变成 0。
_NONFINITE_POLICY = (
    "本帧不含任何非有限值：打靶飞出 BLOWUP 的轨迹用 strings.clip_fill（={fill:g}）占位并配 *_clip 掩码；"
    "超上限的轨迹值截断显示并配 *_clip 掩码（真值写在 meta 的 *_max 里）；"
    "meta 一律是有限标量（.nlf 头用 allow_nan=False 编码），NaN/inf 用哨兵 + 布尔标志代替"
)


def _fin(x: Any, default: float = 0.0) -> float:
    """把非有限标量换成哨兵：``.nlf`` 的 header 用 ``allow_nan=False``，塞不进 NaN。"""
    v = float(x)
    return v if math.isfinite(v) else float(default)


def _clip_display(arr: Any, ceil: float = _DISPLAY_CEIL) -> tuple[np.ndarray, np.ndarray, float, int]:
    """把超上限/非有限的元素截断到 ``ceil``，返回 ``(干净数组, 掩码, 截断前最大值, 个数)``。"""
    a = np.asarray(arr, dtype=float).copy()
    finite = np.isfinite(a)
    raw_max = float(np.max(np.abs(a[finite]))) if np.any(finite) else 0.0
    bad = (~finite) | (np.abs(a) > ceil)
    n_bad = int(np.count_nonzero(bad))
    a = np.where(bad, float(ceil), a)
    return a, bad.astype(float), raw_max, n_bad


def _soft_floor(arr: Any, floor: float = _ROUNDOFF) -> np.ndarray:
    """对数轴的软着陆：把 0 与非有限值抬到明确的地板上（半对数图上就是那条地板线）。"""
    a = np.asarray(arr, dtype=float).copy()
    a[~np.isfinite(a)] = floor
    return np.maximum(a, floor)


def _fmt(x: Any, nd: int = 3) -> str:
    """把可能非有限的拟合量格式化成字符串（NaN 写成 ``n/a``，绝不输出 ``nan``）。"""
    v = float(x)
    return f"{v:.{nd}f}" if math.isfinite(v) else "n/a"


def _clip_scan(t: float, lo: float, hi: float) -> float:
    """把时间轴夹到扫描区间（拖到头就不动了）。"""
    return max(float(lo), min(float(hi), float(t)))


def _prob(params: Mapping[str, Any]) -> bvp.BVPProblem:
    """按 viz 参数构造问题（个别问题不吃 k/L，多给的参数直接忽略）。"""
    name = str(params["problem"])
    kw: dict[str, float] = {}
    if name in ("layer", "decay", "tanh"):
        kw["k"] = float(params.get("k", 10.0))
    if name in ("layer", "decay"):
        kw["L"] = float(params.get("L", 1.0))
    return bvp.problem(name, **kw)


MANIFEST: dict[str, Any] = {
    "chapter": "3.4",
    "title": "边值问题与谱方法",
    "part": "III",
    "summary": (
        "两点边值问题有两种性格完全不同的解法：打靶法把边值问题变成初值问题，"
        "灵敏度 S=|du(b)/ds| 决定它能不能用；谱方法用 N+1 个 Chebyshev 节点换指数收敛，"
        "代价是一个稠密、条件数按 N^2 长的微分矩阵。这一章把两者的「能用/不能用」"
        "都用实测数字钉住：打靶在衰减边界层上失效的长度、有限差分的整体阶被边界格式拉到几阶、"
        "指数收敛的半对数斜率与舍入地板、以及 N^4·u 的谱微分舍入放大。"
    ),
    "vizzes": [
        # ------------------------------------------------------------------ #
        {
            "id": "shooting_scan",
            "title": "打靶法：初始斜率扫描与轨迹族",
            "description": (
                "固定一个边值问题，在参考斜率 s_ref 附近扫一族初始斜率，"
                "画出整族轨迹 u(x; s_j) 与右端打靶函数值 F(s_j)=u(b;s_j)-β。"
                "时间轴是斜率序号 j（相对扫描 s_j = s_ref(1+span(2j/(m-1)-1))）。"
                "选 decay（衰减边界层）时，真解是指数小的那一支，打靶要在两个大数相减里"
                "把它抠出来：kL 一大，整族轨迹就与真解无关了——这不是 bug，是舍入被 S 放大。"
                "选 layer（增长边界层）时同一个 S 却完全能用，副轴的残差会贴着舍入地板。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "problem",
                    "label": "算例",
                    "type": "enum",
                    "options": ["layer", "decay", "tanh", "smooth"],
                    "default": "decay",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "k",
                    "label": "波数 k",
                    "type": "float",
                    "min": 1.0,
                    "max": 40.0,
                    "step": 1.0,
                    "default": 10.0,
                    "critical": [18.0, 36.0],
                    "criticalNote": (
                        "kL≈36 是 roundoff 分界：舍入放大 e^{kL} 到 1/u 之后，"
                        "衰减模态的打靶解就再也回不来了（本章二分实测 L*≈1.875，k=20）。"
                    ),
                    "beyond": "above",
                },
                {
                    "name": "L",
                    "label": "区间长度 L",
                    "type": "float",
                    "min": 0.5,
                    "max": 3.0,
                    "step": 0.1,
                    "default": 1.0,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "span",
                    "label": "相对扫描半宽",
                    "type": "float",
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01,
                    "default": 0.2,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "m",
                    "label": "轨迹条数",
                    "type": "int",
                    "min": 3,
                    "max": 15,
                    "step": 1,
                    "default": 9,
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "scrub", "tMax": 8.0, "dt": 0.25, "label": "斜率序号 j（整族轨迹同时显示；tMax 对应 m=9）"},
            "requiredAux": ["error"],
            "axes": [
                {
                    "id": "main",
                    "autoscaled": True,
                    "title": "轨迹族 u(x; s_j) 与真解",
                    "ylabel": "u",
                },
                {
                    "id": "aux",
                    "autoscaled": False,
                    "title": "右端打靶残差 |F(s_j)| = |u(b;s_j) - beta|（对数轴）",
                    "ylabel": "|F|",
                    "scale": "log",
                },
            ],
            "pitfalls": (
                "第一，轨迹族的「发散」分两种：打靶真的飞出去了（截断显示 + 掩码），"
                "和打靶值本身没错但解在右端偏离真解（衰减模态就是这样，误差 1e1 而轨迹仍然 O(1)），"
                "两者必须分开读。第二，参考斜率 s_ref 是**再用打靶求一次**得到的对照量，"
                "不是解析公式，所以它自己也带同样的舍入误差；把它当「真值」去比较是循环论证。"
                "第三，副轴的对数残差看起来「贴着地板」并不代表解准：衰减模态在 kL=30 时"
                "残差已经停在 5e-4（舍入被 S 放大），此时真解误差也是 5e-4。"
            ),
            "theory": [
                "打靶法：把 u'(a)=s 当参数积分到 b，解 F(s)=u(b;s)-β=0；F'(s)=∂u(b)/∂s 由变分方程 z'=J(x,u)z、z(a)=(0,1) 给出",
                "u''=k^2u、u(a)=0 的闭式灵敏度 S=sinh(kL)/k；衰减算例 u(0)=1,u(L)=0 有同一个 S",
                "线性问题 F 对 s 仿射（Newton 一步）；非线性问题二次收敛（Bratu 实测残差 2.7e1→8.5→1.9→1.8e-1→2.1e-3→3.0e-7→7.3e-13）",
                "衰减模态的失效机制是舍入放大而非截断：误差 ~ eps·e^{kL}，与步长 h 无关（实测 h 缩小 32 倍误差不降）",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "convergence_rate",
            "title": "有限差分 vs 谱配置：误差衰减的代数与指数之别",
            "description": (
                "同一批 N 上并列画两条误差曲线：有限差分（二阶中心差商 + 边界格式 bc_order）"
                "与 Chebyshev 谱配置。左轴半对数（指数收敛是直线），右轴双对数（代数收敛是直线），"
                "两条**实测**拟合线叠加在数据上，地板线单独画。选 smooth（解析解）时谱误差"
                "以每节点约 0.9 位数的速率掉到 1e-14 的地板；选 abs3/abs5（C^2/C^4）时"
                "退化成幂律，而且插值误差比配置解误差**好一阶**——图里同时给出插值曲线作负对照。"
                "有限差分那条曲线带边界格式开关：一阶边界会把整体阶从 2 拉到 1。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "problem",
                    "label": "算例",
                    "type": "enum",
                    "options": ["smooth", "abs3", "abs5", "tanh", "neumann"],
                    "default": "smooth",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "bc_order",
                    "label": "边界差商阶",
                    "type": "enum",
                    "options": ["1", "2"],
                    "default": "2",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "nmax",
                    "label": "最大 N",
                    "type": "int",
                    "min": 16,
                    "max": 512,
                    "step": 16,
                    "default": 256,
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "none"},
            "requiredAux": ["error", "residual"],
            "axes": [
                {
                    "id": "main",
                    "autoscaled": False,
                    "title": "误差 vs N：半对数（指数收敛是直线）",
                    "ylabel": "max|u_num - u_exact|",
                    "scale": "log",
                },
                {
                    "id": "aux",
                    "autoscaled": False,
                    "title": "误差 vs N：双对数（代数收敛是直线）+ 真实边界残差",
                    "ylabel": "误差 / 边界残差",
                    "scale": "log",
                },
            ],
            "pitfalls": (
                "第一，两种收敛的判别不能只看 R^2：有限窗口里幂律拟合解析解也能给出 R^2≈0.98"
                "（本章实测 tanh k=3 的窗口内两种模型难分）。判据要用**误差比**"
                "err(N)/err(N+2)：指数收敛的比值整体上抬（smooth 实测 39→76→197），"
                "幂律收敛的比值压向常数（abs3 实测 1.56→1.14）。第二，报阶必须写窗口："
                "同一批 mixed 边界数据，n=16..128 拟合出 1.86，n=256..1024 才是 1.98。"
                "第三，幂律拟合别拿解析解的**插值**误差冒充配置解误差（|x|^3 实测 3.00 vs 1.99）。"
            ),
            "theory": [
                "有限差分：内部二阶中心差商；整体阶 = min(内部阶, 边界格式阶)——实测一阶边界下整体 0.998 且内部也是 0.998",
                "谱配置：u_N 是在 N+1 个 Chebyshev-Lobatto 节点上满足 ODE 与边界条件的 N 次多项式；解析解误差 e^{-lambda N}（实测斜率 -2.09，每节点约 0.9 位数）",
                "地板：微分矩阵 ||D^2||_inf ~ N^4/3，把舍入放大到 N^4·u（实测 N=32 时 1.1e-10），误差在 N≈18 后不再下降",
                "非解析解降级：配置解率 = 插值率 - 1（|x|^3 实测 3.00-1=1.99；|x|^5 实测 5.01-1=4.02）",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "node_clustering",
            "title": "Chebyshev 节点：端点聚集与插值误差的位置",
            "description": (
                "上排画 N+1 个 Chebyshev-Lobatto 节点的位置、相邻间距、以及解析密度"
                "rho(x)=N/(pi sqrt(1-x^2))：节点在端点附近以 1/sqrt(1-x^2) 聚集，"
                "最大/最小间距比 ≈ (2/pi)N（N=128 实测 81.5）。"
                "下排是同一批节点的**插值**误差曲线：节点上恒为 0，节点之间才是误差，"
                "端点附近的聚集正是为了压住那里的插值误差。"
                "同一张图上再叠一条**配置解**误差作对照：非解析算例下它比插值误差差一阶。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "problem",
                    "label": "算例",
                    "type": "enum",
                    "options": ["smooth", "abs3", "abs5", "tanh"],
                    "default": "abs3",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "n",
                    "label": "节点数 N",
                    "type": "int",
                    "min": 4,
                    "max": 128,
                    "step": 2,
                    "default": 24,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "k",
                    "label": "tanh 的 k",
                    "type": "float",
                    "min": 1.0,
                    "max": 20.0,
                    "step": 1.0,
                    "default": 3.0,
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "none"},
            "requiredAux": ["error", "collocation_error"],
            "axes": [
                {
                    "id": "main",
                    "autoscaled": False,
                    "title": "节点、间距与密度",
                    "ylabel": "x_j 与间距",
                },
                {
                    "id": "aux",
                    "autoscaled": False,
                    "title": "插值误差与配置解误差（对数轴）",
                    "ylabel": "误差",
                    "scale": "log",
                },
            ],
            "pitfalls": (
                "第一，「节点密」不等于「那里误差小」：插值误差在**节点上**恒为 0，"
                "图上那条为 0 的曲线不是「精度高」，而是「没测到」。第二，端点聚集的代价是"
                "条件数：节点间距最小 ~N^-2，微分矩阵 ||D||_inf=N^2，"
                "所以 N 不能一味加大（smooth 算例 N≥24 后误差反而变大）。"
                "第三，密度曲线是**解析式** N/(pi sqrt(1-x^2))，不是从节点间距统计出来的，"
                "两者在 N 小的时候差得很明显（端点处解析值发散，实际间距有限）。"
            ),
            "theory": [
                "Chebyshev-Lobatto 节点 x_j=cos(j pi/N)，j=0..N（降序、含两端），是 theta 均匀网格的余弦投影",
                "密度 rho(x)=N/(pi sqrt(1-x^2))；rho(0.9)/rho(0)=1/sqrt(0.19)=2.294（N=64 实测 2.2936）",
                "间距比 max/min ≈ (2/pi)N：N=8..128 实测 5.03→81.5，/N 单调趋于 0.63662",
                "插值误差与配置解误差是两回事：|x|^3 实测插值 N^-3.00、配置解 N^-1.99",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "diff_matrix_structure",
            "title": "谱微分矩阵：稠密、奇异、条件数 N^2",
            "description": (
                "热力图画出 N+1 阶 Chebyshev 微分矩阵 D_N（用 log10|D| 着色，符号单独给掩码），"
                "对角元由行和为零反解（D_00=(2N^2+1)/6，不是 0）；右边两条副轴分别是行和 D_N·1（恒为 0，"
                "零空间就是常数向量）与奇异值谱（sigma_min 是 1e-17 级的噪声，"
                "真正有意义的是挖掉零空间后的 sigma_min+ 与 kappa+ ≈ N^2）。"
                "再叠一条「对解析解求导的逐点误差」，展示谱微分在节点上的精度与"
                "在 N 变大时 N^4·u 的舍入放大。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "n",
                    "label": "矩阵阶 N",
                    "type": "int",
                    "min": 4,
                    "max": 64,
                    "step": 2,
                    "default": 16,
                    "critical": [24],
                    "criticalNote": (
                        "N≈24 起 N^4·u 的舍入放大（24^4·u≈1e-12）开始盖过截断误差，"
                        "这就是「再加大 N 没有意义」的实测位置。"
                    ),
                    "beyond": "above",
                },
                {
                    "name": "which",
                    "label": "矩阵",
                    "type": "enum",
                    "options": ["D", "D2"],
                    "default": "D",
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "none"},
            "requiredAux": ["residual", "error_derivative"],
            "axes": [
                {
                    "id": "main",
                    "autoscaled": False,
                    "title": "log10|D_N| 热力图（行主序，含零对角线与零空间）",
                    "ylabel": "行 i",
                },
                {
                    "id": "aux",
                    "autoscaled": False,
                    "title": "行和残差 |D_N·1| 与奇异值谱（对数轴）",
                    "ylabel": "残差 / sigma",
                    "scale": "log",
                },
            ],
            "pitfalls": (
                "第一，D_N 是**奇异**矩阵，直接把 sigma_max/sigma_min 当条件数会得到 1e17"
                "这种毫无意义的数（那是零空间的舍入噪声）；正确口径是挖掉零空间后的 kappa+，"
                "它按 N^1.95 增长。第二，D_N 稠密、条件数 N^2，不能拿它去「解方程组」："
                "配置法解的是 L=D_N^2+diag(q) 再加两行边界，kappa(L) 按 N^4.4 增长"
                "（N=256 实测 2.8e9），所以 N 的上限由条件数而不是内存决定。"
                "第三，矩阵没有零对角线可省：对角元由行和为零反解出 D_00=(2N^2+1)/6"
                "（N=16 实测 85.5），把它当稀疏结构去优化会写错。"
            ),
            "theory": [
                "D_N 的元素有闭式（Trefethen 公式）；对角元 D_ii=-sum_{j!=i}D_ij，故 D_00=(2N^2+1)/6（N=16 实测 85.5）；||D_N||_inf = N^2（实测 N=4..64 逐位吻合）",
                "D_N·1 = 0：常数函数的导数处处为 0，这就是零空间；sigma_min ≈ 1e-17 是噪声，sigma_min+ ≈ 1.05~1.14 有界",
                "kappa+ = sigma_max/sigma_min+ ~ N^1.95（实测 N=8..64：35.1→1988）；||D_N^2||_inf ≈ N^4/3",
                "配置矩阵 kappa_inf ~ N^4.4（实测 N=8 5.5e2 → N=256 2.8e9）；tau 法 kappa ~ N^3.9 且矩阵更稀疏",
            ],
        },
    ],
}


def compute(viz_id: str, params: Mapping[str, Any], t: float) -> Frame:
    """章节计算入口：``params`` 已由 :mod:`server.frames` 规范化并校验过区间。"""
    if viz_id == "shooting_scan":
        return _shooting_scan(params, t)
    if viz_id == "convergence_rate":
        return _convergence_rate(params, t)
    if viz_id == "node_clustering":
        return _node_clustering(params, t)
    if viz_id == "diff_matrix_structure":
        return _diff_matrix_structure(params, t)
    raise KeyError(f"章节 3.4 没有可视化 {viz_id!r}")


# --------------------------------------------------------------------------- #
# viz 1：打靶法斜率扫描
# --------------------------------------------------------------------------- #
def _shooting_scan(params: Mapping[str, Any], t: float) -> Frame:
    prob = _prob(params)
    m = int(params["m"])
    span = float(params["span"])
    j = _clip_scan(t, 0.0, float(max(m - 1, 1)))
    n_steps = 800 if prob.length <= 1.5 else 2000

    ref = bvp.shooting(prob, n_steps=n_steps, max_iter=20)
    s_ref = ref.slope if math.isfinite(ref.slope) else 1.0
    frac = np.linspace(-1.0, 1.0, m)
    slopes = s_ref * (1.0 + span * frac)
    x, family, matched, ok = bvp.shooting_scan(prob, slopes, n_steps=n_steps)
    exact = prob.exact(x)

    # 每一条轨迹的诊断：与真解的最大偏差、右端残差、是否飞出 BLOWUP
    errors = np.zeros(m)
    for k in range(m):
        errors[k] = float(np.max(np.abs(family[k] - exact))) if ok[k] else float("nan")
    family_disp, family_clip, family_raw_max, n_family_clip = _clip_display(family)
    errors_disp, errors_clip, errors_raw_max, n_errors_clip = _clip_display(errors)
    residual = np.where(ok, matched - prob.beta, np.nan)
    residual_disp, residual_clip, residual_raw_max, n_res_clip = _clip_display(np.abs(residual))
    node = int(min(max(round(j), 0), m - 1))

    sens = bvp.shooting_sensitivity(prob, s_ref, n_steps=n_steps)
    frame = Frame(chapter="3.4", viz="shooting_scan", kind="frame")
    frame.add("x", x, unit="1", label="x（打靶轨迹的积分网格）")
    frame.add("family", family_disp, unit="1", label="轨迹族 u(x; s_k)，k=0..m-1（超上限处 = strings.clip_fill）")
    frame.add("family_clip", family_clip, unit="1", label="轨迹截断掩码（1 = 该格被截断显示）")
    frame.add("u_exact", exact, unit="1", label="真解 u(x)")
    frame.add("slopes", slopes, unit="1", label="本族初始斜率 s_k（相对参考斜率的等比扫描）")
    matched_clean = _soft_floor(np.where(ok, matched, math.inf), _DISPLAY_CEIL)
    matched_clean = np.minimum(matched_clean, _DISPLAY_CEIL)
    frame.add("matched", matched_clean, unit="1", label="右端匹配量 u(b;s_k)（飞出 BLOWUP 处 = strings.clip_fill）")
    frame.add("residual", residual_disp, unit="1", label="打靶残差 |u(b;s_k) - beta|（对数轴用）")
    frame.add("residual_clip", residual_clip, unit="1", label="残差截断掩码")
    frame.add("error", errors_disp, unit="1", label="每条轨迹与真解的最大偏差（*_clip 标记截断）")
    frame.add("error_clip", errors_clip, unit="1", label="误差截断掩码")
    frame.add("trace_x", np.array([x[0], x[-1]], dtype=float), unit="1", label="区间端点（供前端画参考线）")
    frame.set_meta(
        m=m,
        span=_fin(span),
        j=_fin(j),
        node_index=node,
        s_ref=_fin(s_ref),
        slope_node=_fin(float(slopes[node])),
        span_lo=_fin(float(slopes[0])),
        span_hi=_fin(float(slopes[-1])),
        sensitivity=_fin(sens),
        digits_lost=_fin(bvp.amplification_digits(sens)),
        converged=bool(ref.converged),
        stagnated=bool(ref.stagnated),
        iterations=int(ref.iterations),
        residual_ref=_fin(abs(ref.residual)),
        error_ref=_fin(ref.error_max),
        n_steps=int(n_steps),
        n_family_clip=n_family_clip,
        n_errors_clip=n_errors_clip,
        n_residual_clip=n_res_clip,
        n_blowup=int(np.count_nonzero(~ok)),
        family_max_abs=_fin(family_raw_max),
        error_max_raw=_fin(errors_raw_max),
        residual_max_raw=_fin(residual_raw_max),
        residual_min=_fin(float(np.min(residual_disp))),
        residual_floor=_fin(max(U_FLOOR * sens, U_FLOOR)),
        k=_fin(prob.param("k")),
        L=_fin(prob.length),
        kL=_fin(prob.param("k") * prob.length),
        clip_fill=_DISPLAY_CEIL,
        roundoff_floor=_ROUNDOFF,
        overflow=bool(n_family_clip > 0 or n_errors_clip > 0 or np.count_nonzero(~ok) > 0),
    )
    frame.set_strings(
        problem=str(params["problem"]),
        label=prob.label,
        nonfinite_policy=_NONFINITE_POLICY.format(fill=_DISPLAY_CEIL),
        clip_fill=f"{_DISPLAY_CEIL:g}",
        axis_note="主轴是真解与轨迹族；副轴对数显示打靶残差 |F(s_k)|，地板由 S 决定：|F| 分辨不出来时曲线会变平",
        failure_note=(
            "衰减边界层（decay）：解含一个指数小的模态，打靶要在 (s/k)sinh(kx) 与 (s/k)cosh(kx) 的"
            "组合里把它相减出来；舍入被 S 放大 e^{kL} 倍，误差 ~ eps·e^{kL}。"
            "增长边界层（layer）：同一 S，但真解是指数大的那支，相减不丢位数。"
        ),
        sensitivity_note=(
            f"实测灵敏度 S={sens:.4e}（变分方程积分，不是差分近似），吃掉 {bvp.amplification_digits(sens):.2f} 位十进制；"
            "s_ref 由再一次打靶给出，因此它自己也带同一量级的误差"
        ),
        overflow=(
            "有轨迹飞出 BLOWUP 或超上限：数组里已换成 clip_fill 并配掩码，"
            "meta 里的 *_max 是被截断前的真实量级"
        )
        if (n_family_clip or n_errors_clip or np.count_nonzero(~ok))
        else "本帧无截断",
    )
    return frame


# --------------------------------------------------------------------------- #
# viz 2：收敛率对比
# --------------------------------------------------------------------------- #
def _convergence_rate(params: Mapping[str, Any], t: float) -> Frame:
    prob = _prob(params)
    bc_order = int(params["bc_order"])
    nmax = int(params["nmax"])
    # 差分窗口必须是"从 nmax 往回按比例取"的一组点：写死一串会让 nmax=16 时只剩 1 个点，
    # 拟合函数返回 NaN，元数据里就出现非有限值（本章实测撞过一次）。
    fd_ns = sorted({max(8, int(round(nmax * f))) for f in (0.1, 0.2, 0.35, 0.55, 0.8, 1.0)})
    fd_ns = [n for n in fd_ns if n <= nmax] or [nmax]
    if len(fd_ns) < 3:
        fd_ns = [max(4, nmax // 2), max(6, (3 * nmax) // 4), nmax]
    fd = bvp.convergence_study(prob, fd_ns, method="fd", bc_order=bc_order)
    cheb_ns = [n for n in (4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32) if n <= max(20, nmax // 4)]
    if len(cheb_ns) < 5:
        cheb_ns = [4, 6, 8, 10, 12]
    cheb = bvp.convergence_study(prob, cheb_ns, method="collocation")
    interp = np.array([bvp.interpolation_error(prob, n) for n in cheb_ns], dtype=float)
    cheb_x = np.asarray(cheb["n"], dtype=float)
    cheb_err = np.asarray(cheb["error"], dtype=float)

    # 实测拟合：谱侧取「尚未触地板」的粗端窗口，差分侧取细端窗口（窗口写进 meta 与 strings）
    floor_info = bvp.roundoff_floor(cheb["error"], cheb["n"])
    head = int(np.count_nonzero(cheb["error"] > 100.0 * floor_info["floor"]))
    head = max(4, min(head, len(cheb_ns) - 1))
    fit_exp = bvp.fit_exponential(cheb["n"], cheb["error"], head=head)
    # 幂律拟合必须与指数拟合**同窗口**：拿细端（已触地板）做幂律拟合会得到 p=0.5
    # 这种毫无意义的数，而且画出来的直线会横穿地板区——两条竞争模型必须站在同一段数据上。
    fit_pow = bvp.fit_power(cheb_x[:head], cheb_err[:head], tail=head)
    fit_fd = bvp.fit_power(fd["n"], fd["error"], tail=min(4, len(fd_ns)))
    fit_int = bvp.fit_power(cheb_x[:head], interp[:head], tail=head)
    fit_interior = bvp.fit_power(fd["n"], fd["error_interior"], tail=min(4, len(fd_ns)))

    line_exp = np.exp(fit_exp["intercept"] + fit_exp["slope"] * cheb_x) if math.isfinite(fit_exp["slope"]) else np.full_like(cheb_x, floor_info["floor"])
    line_pow = (
        np.exp(fit_pow["intercept"]) * cheb_x ** (-fit_pow["p"])
        if math.isfinite(fit_pow["p"])
        else np.full_like(cheb_x, floor_info["floor"])
    )
    # 竞品线只画在拟合窗口里：窗口外是外推，画出来就是另一种撒谎
    line_pow = np.where(cheb_x <= cheb_x[head - 1], line_pow, floor_info["floor"])

    frame = Frame(chapter="3.4", viz="convergence_rate", kind="frame")
    frame.add("n_fd", np.asarray(fd["n"], dtype=float), unit="1", label="有限差分的区间数 n")
    frame.add("fd_error", _soft_floor(fd["error"]), unit="1", label="有限差分误差 max|u_num-u_exact|")
    frame.add("fd_error_interior", _soft_floor(fd["error_interior"]), unit="1", label="有限差分内部误差（掐掉两端 25%）")
    frame.add("fd_residual", _soft_floor(fd["bc_true"]), unit="1", label="有限差分真实边界残差（不是离散行残差）")
    frame.add("fd_residual_discrete", _soft_floor(fd["bc_discrete"]), unit="1", label="有限差分离散边界行残差（恒为舍入级）")
    frame.add("n_cheb", cheb_x, unit="1", label="Chebyshev 节点数 N")
    frame.add("cheb_error", _soft_floor(cheb["error"]), unit="1", label="谱配置误差")
    frame.add("cheb_error_interior", _soft_floor(cheb["error_interior"]), unit="1", label="谱配置内部误差")
    frame.add("cheb_residual", _soft_floor(cheb["bc_true"]), unit="1", label="谱配置边界残差（与误差同量级）")
    frame.add("interp_error", _soft_floor(interp), unit="1", label="同一批节点上的插值误差（负对照：比配置解好一阶）")
    frame.add("fit_exponential", _soft_floor(line_exp), unit="1", label="实测半对数拟合线")
    frame.add("fit_power", _soft_floor(line_pow), unit="1", label="实测幂律拟合线（与指数拟合同窗口，窗口外不画）")
    frame.add("floor", np.array([floor_info["floor"]], dtype=float), unit="1", label="实测舍入地板")
    frame.add("bc_true_fd", _soft_floor(fd["bc_true"]), unit="1", label="有限差分真实边界残差（副轴用）")
    frame.set_meta(
        bc_order=bc_order,
        nmax=nmax,
        fd_window_lo=float(fd_ns[0]),
        fd_window_hi=float(fd_ns[-1]),
        fd_p=_fin(fit_fd["p"]),
        fd_r2=_fin(fit_fd["r2"]),
        fd_p_interior=_fin(fit_interior["p"]),
        cheb_window_lo=float(cheb_ns[0]),
        cheb_window_hi=float(cheb_ns[-1]),
        exp_fit_head=head,
        exp_slope=_fin(fit_exp["slope"]),
        exp_r2=_fin(fit_exp["r2"]),
        exp_digits_per_node=_fin(-float(fit_exp["slope"]) / math.log(10.0) if math.isfinite(fit_exp["slope"]) else 0.0),
        power_p=_fin(fit_pow["p"]),
        power_r2=_fin(fit_pow["r2"]),
        interp_p=_fin(fit_int["p"]),
        interp_r2=_fin(fit_int["r2"]),
        floor=_fin(floor_info["floor"]),
        n_floor=_fin(floor_info["n_floor"]),
        n_at_min=_fin(floor_info["n_at_min"]),
        error_min=_fin(float(np.min(cheb["error"]))),
        cheb_error_first=_fin(float(cheb["error"][0])),
        cheb_error_last=_fin(float(cheb["error"][-1])),
        fd_error_first=_fin(float(fd["error"][0])),
        fd_error_last=_fin(float(fd["error"][-1])),
        n4_roundoff=_fin(U_FLOOR * float(max(cheb_ns)) ** 4),
        roundoff_floor=_ROUNDOFF,
        overflow=False,
    )
    frame.set_strings(
        problem=str(params["problem"]),
        label=prob.label,
        nonfinite_policy=_NONFINITE_POLICY.format(fill=_DISPLAY_CEIL),
        clip_fill=f"{_DISPLAY_CEIL:g}",
        window_note=(
            f"差分拟合窗口 n={fd_ns[0]}..{fd_ns[-1]}（细端），谱拟合窗口 N={cheb_ns[0]}..{cheb_ns[head - 1]}（粗端，"
            f"取到误差仍高于 100 倍地板为止）；插值曲线与谱曲线同一窗口"
        ),
        rate_note=(
            f"实测：谱侧半对数斜率 {_fmt(fit_exp['slope'], 3)}（每节点 {_fmt(-fit_exp['slope'] / math.log(10.0), 2)} 位，R^2={_fmt(fit_exp['r2'], 4)}），"
            f"幂律 p={_fmt(fit_pow['p'], 3)}（R^2={_fmt(fit_pow['r2'], 4)}）；差分侧 p={_fmt(fit_fd['p'], 4)}（内部 {_fmt(fit_interior['p'], 4)}）；"
            f"插值 p={_fmt(fit_int['p'], 3)}（与配置解差 {_fmt(fit_int['p'] - fit_pow['p'], 3)} 阶）；"
            "三个幂律指数都在**同一个**粗端窗口上拟合，是直接竞争关系；"
            f"注意两个模型在同一窗口上的 R² 可以同时很高（指数 {_fmt(fit_exp['r2'], 3)} vs 幂律 {_fmt(fit_pow['r2'], 3)}），"
            "所以判别指数/幂律要看误差比 err(N)/err(N+2) 的走向，不是比 R²"
        ),
        floor_note=(
            f"舍入地板实测 {floor_info['floor']:.2e}，N>={floor_info['n_floor']:.0f} 起不再下降"
            f"（最小误差出现在 N={floor_info['n_at_min']:.0f}）；N={max(cheb_ns)} 时 N^4·u={U_FLOOR * float(max(cheb_ns)) ** 4:.1e}"
        ),
        bc_note=(
            "边界残差有两个口径：离散行残差恒为舍入级（图上那条平的），"
            "真实违反量才反映边界格式的阶——一阶边界下它是 O(h) 并与整体误差同量级"
        ),
        overflow="本帧无截断：所有曲线取对数前已抬到 u/2 地板",
    )
    return frame


# --------------------------------------------------------------------------- #
# viz 3：节点聚集
# --------------------------------------------------------------------------- #
def _node_clustering(params: Mapping[str, Any], t: float) -> Frame:
    prob = _prob(params)
    n = int(params["n"])
    x = bvp.cheb_nodes_on(prob, n)
    xi = bvp.cheb_nodes(n)
    spacing = bvp.node_spacing(n)

    fine = bvp.default_grid(prob, 601)
    rho_fine = bvp.node_density(n, bvp.to_reference(prob, fine))
    rho_nodes = bvp.node_density(n, xi)
    interp = bvp.clenshaw(bvp.cheb_transform(prob.exact(x)), bvp.to_reference(prob, fine))
    interp_err = np.abs(interp - prob.exact(fine))
    sol = bvp.cheb_collocation(prob, n)
    coll_err = np.abs(bvp.sample_solution(sol.x, sol.u, fine, kind="barycentric") - prob.exact(fine))

    frame = Frame(chapter="3.4", viz="node_clustering", kind="frame")
    frame.add("nodes", x, unit="1", label="Chebyshev-Lobatto 节点 x_j（降序，含两端）")
    frame.add("node_theta", np.pi * np.arange(n + 1) / n, unit="1", label="对应的 theta_j = j pi/N（均匀）")
    frame.add("spacing", spacing, unit="1", label="相邻间距 |x_{j+1} - x_j|（长度 N）")
    frame.add("spacing_mid", 0.5 * (x[:-1] + x[1:]), unit="1", label="间距对应的中点位置")
    frame.add("density_fine", rho_fine, unit="1", label="解析密度 rho(x)=N/(pi sqrt(1-x^2))（细网格）")
    frame.add("density_nodes", rho_nodes, unit="1", label="节点处的解析密度")
    frame.add("x_fine", fine, unit="1", label="细网格（不含端点）")
    frame.add("u_exact_fine", prob.exact(fine), unit="1", label="真解（细网格）")
    frame.add("u_interp_fine", interp, unit="1", label="节点插值多项式（细网格）")
    frame.add("error", _soft_floor(interp_err), unit="1", label="插值误差 |I_N u - u|（节点上恒为 0）")
    frame.add("collocation_error", _soft_floor(coll_err), unit="1", label="配置解误差（负对照：非解析算例下差一阶）")
    frame.add("density_ratio_mark", np.array([rho_nodes[0], rho_nodes[n // 2]], dtype=float), unit="1", label="端点/中点密度（供副轴读聚集比）")
    frame.set_meta(
        n=n,
        n_nodes=n + 1,
        k=_fin(prob.param("k")),
        cluster_ratio=_fin(bvp.cluster_ratio(n)),
        ratio_over_n=_fin(bvp.cluster_ratio(n) / float(n)),
        ratio_over_n_theory=_fin(2.0 / math.pi),
        max_gap=_fin(float(np.max(spacing))),
        min_gap=_fin(float(np.min(spacing))),
        outer_nodes=_fin(float(bvp.outer_node_count(n, 0.1))),
        interp_error_max=_fin(float(np.max(interp_err))),
        interp_error_mid=_fin(float(interp_err[interp_err.size // 2])),
        collocation_error_max=_fin(sol.error_max),
        order_gap=_fin(math.log10(max(float(np.max(interp_err)), 1e-300) / max(sol.error_max, 1e-300))),
        density_ratio=_fin(float(bvp.node_density(n, np.array([0.9]))[0] / bvp.node_density(n, np.array([0.0]))[0])),
        density_ratio_theory=_fin(1.0 / math.sqrt(1.0 - 0.9**2)),
        cond_inf=_fin(sol.cond_inf),
        roundoff_floor=_ROUNDOFF,
        overflow=False,
    )
    frame.set_strings(
        problem=str(params["problem"]),
        label=prob.label,
        nonfinite_policy=_NONFINITE_POLICY.format(fill=_DISPLAY_CEIL),
        clip_fill=f"{_DISPLAY_CEIL:g}",
        cluster_note=(
            f"最大/最小间距比 {bvp.cluster_ratio(n):.3f}（= {bvp.cluster_ratio(n) / n:.4f}·N，渐近常数 2/pi=0.63662）；"
            f"两端各 10% 宽度内有 {bvp.outer_node_count(n, 0.1)} 个节点，共 {n + 1} 个"
        ),
        zero_note=(
            "插值误差曲线在**节点上**恰好为 0（多项式插值的定义），所以那条「掉到地板」的曲线"
            "不是精度证据；节点之间的峰值才是有意义的量"
        ),
        contrast_note=(
            f"同一批节点：插值误差峰值 {float(np.max(interp_err)):.2e}，配置解误差 {sol.error_max:.2e}——"
            "非解析解下配置解差一阶（微分吃掉一阶光滑度）"
        ),
        overflow="本帧无截断",
    )
    return frame


# --------------------------------------------------------------------------- #
# viz 4：微分矩阵结构
# --------------------------------------------------------------------------- #
def _diff_matrix_structure(params: Mapping[str, Any], t: float) -> Frame:
    n = int(params["n"])
    which = str(params["which"])
    d1 = bvp.cheb_diff_matrix(n)
    matrix = d1 if which == "D" else bvp.cheb_second_matrix(n)
    diag = bvp.diff_matrix_diagnostics(n)

    abs_m = np.abs(matrix)
    log_m = np.log10(np.maximum(abs_m, _ROUNDOFF))
    sign_m = (matrix > 0.0).astype(float) - (matrix < 0.0).astype(float)
    rowsum = matrix.sum(axis=1)  # D 的零空间：这一行应当恒为 0
    sigma = np.asarray(_svd_values(matrix), dtype=float)

    # 对解析解求导的逐点误差（谱微分的精度）：只在 D 上做
    prob = bvp.problem("smooth")
    u_vals = prob.exact(bvp.cheb_nodes(n))
    du_num = d1 @ u_vals
    du_exact = prob.derivative(bvp.cheb_nodes(n))
    derr = np.abs(du_num - du_exact)

    coll_cond = bvp.cheb_collocation(prob, min(max(n, 4), 48)).cond_inf
    frame = Frame(chapter="3.4", viz="diff_matrix_structure", kind="frame")
    frame.add("matrix", matrix, unit="1", label="微分矩阵（行主序，(N+1)x(N+1)）")
    frame.add("log_abs_matrix", log_m, unit="1", label="log10|matrix|（热力图着色用）")
    frame.add("sign_matrix", sign_m, unit="1", label="符号（+1/0/-1）")
    frame.add("diagonal", np.diag(matrix), unit="1", label="对角元（D_00=(2N^2+1)/6；由行和为零反解，不是 0）")
    frame.add("diag_zero", (np.abs(np.diag(matrix)) <= 1e-12 * max(float(np.max(np.abs(matrix))), 1.0)).astype(float), unit="1", label="数值上为零的对角元掩码（只有 x=0 那个节点可能命中）")
    frame.add("rowsum", rowsum, unit="1", label="行和（D·1 = 0：零空间是常数）")
    frame.add("residual", _soft_floor(np.abs(rowsum) + 0.0), unit="1", label="行和残差 |matrix·1|（对数轴；D 与 D^2 都应恒为 0，零空间是常数）")
    frame.add("sigma", sigma, unit="1", label=f"{which} 的奇异值（降序；sigma_min ~ 1e-17 是零空间噪声）")
    frame.add("sigma_index", np.arange(sigma.size, dtype=float), unit="1", label="奇异值序号")
    frame.add("null_vector", np.ones(n + 1), unit="1", label="零空间向量（常数 1）")
    frame.add("error_derivative", _soft_floor(derr), unit="1", label="谱微分对解析解的逐点误差")
    frame.add("nodes", bvp.cheb_nodes(n), unit="1", label="节点（误差曲线的横坐标）")
    frame.set_meta(
        n=n,
        size=n + 1,
        norm_inf=_fin(diag["norm_inf"]),
        norm_inf_theory=_fin(float(n * n) if which == "D" else float(n**4) / 3.0),
        rowsum_residual=_fin(diag["rowsum_residual"]),
        density=_fin(diag["density"]),
        d00=_fin(float(matrix[0, 0])),
        d00_theory=_fin(float((2.0 * n * n + 1.0) / 6.0) if which == "D" else float(n**4) / 3.0),
        n_nonzero=_fin(diag.get("n_nonzero", 0.0)),
        sigma_max=_fin(diag.get("sigma_max", 0.0)),
        sigma_min=_fin(diag.get("sigma_min", 0.0)),
        sigma_min_nonzero=_fin(diag.get("sigma_min_nonzero", 0.0)),
        kappa_naive=_fin(diag.get("kappa_naive", 0.0)),
        kappa_nonzero=_fin(diag.get("kappa_nonzero", 0.0)),
        d2_norm_inf=_fin(diag.get("D2_norm_inf", 0.0)),
        collocation_cond_inf=_fin(coll_cond),
        derivative_error_max=_fin(float(np.max(derr))),
        derivative_error_mid=_fin(float(derr[derr.size // 2])),
        n4_roundoff=_fin(U_FLOOR * float(n) ** 4),
        roundoff_floor=_ROUNDOFF,
        overflow=False,
    )
    frame.set_strings(
        which=which,
        nonfinite_policy=_NONFINITE_POLICY.format(fill=_DISPLAY_CEIL),
        clip_fill=f"{_DISPLAY_CEIL:g}",
        singular_note=(
            f"D_N 奇异（零空间是常数向量）：朴素 kappa = {diag.get('kappa_naive', float('nan')):.2e} 是零空间里的舍入噪声，"
            f"挖掉后 kappa+ = {diag.get('kappa_nonzero', float('nan')):.2f}（sigma_min+ 有界）"
        ),
        use_note=(
            "微分矩阵不是用来解方程组的：配置法解的是 L = s^2 D_N^2 + diag(q) 再加两行边界，"
            f"kappa_inf(L) 实测 {coll_cond:.2e}（按 N^4.4 增长），"
            "tau 法把方程写在系数空间，矩阵更稀疏但条件数同量级"
        ),
        roundoff_note=(
            f"谱微分对解析解的逐点误差最大 {float(np.max(derr)):.2e}；N^4·u = {U_FLOOR * float(n) ** 4:.1e} 是"
            "微分放大舍入的预期地板（把精确解插值后求二阶导，残差也在这个量级）"
        ),
        overflow="本帧无截断",
    )
    return frame


def _svd_values(matrix: np.ndarray) -> np.ndarray:
    """奇异值（走自研 Jacobi SVD，P3 不许 numpy.linalg）。"""
    from numcore.linalg.svd import jacobi_svd

    fit = jacobi_svd(np.asarray(matrix, dtype=float))
    return np.asarray(fit.s, dtype=float)
