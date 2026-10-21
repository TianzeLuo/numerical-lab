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
        "灵敏度 $S=|\\mathrm{d}u(b)/\\mathrm{d}s|$ 决定它能不能用；谱方法用 $N+1$ 个 Chebyshev 节点换指数收敛，"
        "代价是一个稠密、条件数按 $N^2$ 长的微分矩阵。这一章把两者的「能用/不能用」"
        "都用实测数字钉住：打靶在衰减边界层上失效的长度、有限差分的整体阶被边界格式拉到几阶、"
        "指数收敛的半对数斜率与舍入地板、以及 $N^4\\cdot u$ 的谱微分舍入放大。"
    ),
    "vizzes": [
        # ------------------------------------------------------------------ #
        {
            "id": "shooting_scan",
            "title": "打靶法：初始斜率扫描与轨迹族",
            "description": (
                "固定一个边值问题，在参考斜率 $s_{\\mathrm{ref}}$ 附近扫一族初始斜率，"
                "画出整族轨迹 $u(x;s_j)$ 与右端打靶函数值 $F(s_j)=u(b;s_j)-\\beta$。"
                "时间轴是斜率序号 $j$（相对扫描 $s_j=s_{\\mathrm{ref}}(1+\\mathrm{span}(2j/(m-1)-1))$）。"
                "选 decay（衰减边界层）时，真解是指数小的那一支，打靶要在两个大数相减里"
                "把它抠出来：$kL$ 一大，整族轨迹就与真解无关了——这不是 bug，是舍入被 $S$ 放大。"
                "选 layer（增长边界层）时同一个 $S$ 却完全能用，副轴的残差会贴着舍入地板。"
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
                    "help": (
                        "选择边值算例。layer 是增长边界层，decay 是衰减边界层，二者灵敏度相同但命运相反；"
                        "tanh 与 smooth 用来观察非线性与光滑解。切换后先看副轴残差能否贴到地板，"
                        "再看整族轨迹是否包住真解。"
                    ),
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
                        "$kL\\approx 36$ 是 roundoff 分界：舍入放大 $e^{kL}$ 到 $1/u$ 之后，"
                        "衰减模态的打靶解就再也回不来了（本章二分实测 $L^*\\approx 1.875$，$k=20$）。"
                    ),
                    "beyond": "above",
                    "help": (
                        "边界层的波数（无量纲）。调大后 $kL$ 增大，衰减算例的舍入放大按 $e^{kL}$ 增长，"
                        "打靶很快失效，增长算例却仍然可用。它对 tanh 算例只改变过渡层陡度，"
                        "对 smooth 算例没有影响；请同时看右列的灵敏度与失效位数。"
                    ),
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
                    "help": (
                        "区间长度。它与 $k$ 通过乘积 $kL$ 共同决定难度：$L$ 越大，边界层相对越厚，"
                        "衰减模态被放大的倍数越高；$L$ 很小则问题接近光滑，打靶容易成功。"
                        "调 $L$ 时请同时看右列的 $kL$ 读数，而不是只看单个参数。"
                    ),
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
                    "help": (
                        "初始斜率围绕参考值展开的相对半宽。调大后整族轨迹散得更开，副轴上的残差跨度变大，"
                        "容易看出哪一条最接近命中；调小则所有轨迹挤在一起，更接近线性问题的局部行为。"
                        "它不改变参考斜率本身。"
                    ),
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
                    "help": (
                        "画多少条轨迹（3 到 15）。条数多时能看清整族如何随斜率变化，但每条线更密、图例更挤；"
                        "条数少时单条轨迹更醒目，时间轴上的斜率序号范围也更短。"
                        "它与扫描半宽一起决定相邻斜率的间隔。"
                    ),
                },
            ],
            "time": {"mode": "scrub", "tMax": 8.0, "dt": 0.25, "label": "斜率序号 $j$（整族轨迹同时显示；$t_{\\max}$ 对应 $m=9$）"},
            "requiredAux": ["error"],
            "axes": [
                {
                    "id": "main",
                    "autoscaled": False,
                    "title": "打靶轨迹族（真解窗口）",
                    "ylabel": "$u$",
                },
                {
                    "id": "aux",
                    "autoscaled": False,
                    "title": "右端打靶残差 $|F(s_j)|$（对数轴）",
                    "ylabel": "$|F|$",
                    "scale": "log",
                },
            ],
            "pitfalls": (
                "第一，轨迹族的「发散」分两种：打靶真的飞出去了（截断显示 + 掩码），"
                "和打靶值本身没错但解在右端偏离真解（衰减模态就是这样，误差 1e1 而轨迹仍然 $\\mathcal{O}(1)$），"
                "两者必须分开读。第二，参考斜率 $s_{\\mathrm{ref}}$ 是**再用打靶求一次**得到的对照量，"
                "不是解析公式，所以它自己也带同样的舍入误差；把它当「真值」去比较是循环论证。"
                "第三，副轴的对数残差看起来「贴着地板」并不代表解准：衰减模态在 $kL=30$ 时"
                "残差已经停在 5e-4（舍入被 $S$ 放大），此时真解误差也是 5e-4。"
            ),
            "theory": [
                (
                    "**边值问题不能从一端直接积分。** 方程 $u''=f(x,u,u')$ 在 $[a,b]$ 上给定两端条件，"
                    "例如 $u(a)=\\alpha$、$u(b)=\\beta$。初值问题只需要 $u(a)$ 与 $u'(a)$，所以打靶法把未知的左端斜率记作 $s$，"
                    "先解初值问题 $u(a)=\\alpha$、$u'(a)=s$，得到右端值 $u(b;s)$，再解一个标量方程 "
                    "$F(s)=u(b;s)-\\beta=0$。每算一次 $F$ 要积分一遍整个区间，求根方法决定总共要打几枪。"
                    "如果 $F$ 在区间两端异号，连续函数保证至少有一个根；打靶的困难不在“有没有根”，"
                    "而在“根对 $s$ 的微小扰动有多敏感”。"
                ),
                (
                    "**灵敏度由变分方程给出。** 把状态写成 $y=(u,u')^{\\mathrm T}$，右端 $y'=f(x,y)$。"
                    "对 $s$ 求导并交换求导次序，得到 $z'=J(x,u)z$，其中 $J=\\partial f/\\partial y$，初值 "
                    "$z(a)=(0,1)^{\\mathrm T}$。它的第一个分量就是 $\\partial u(b;s)/\\partial s$，绝对值记作 $S$。"
                    "这一步不需要数值差分：变分方程和主方程共用同一个积分器，既省一次积分，也避免差分步长带来的额外误差。"
                ),
                (
                    "**一个能手算的例子。** 取 $u''=k^2u$、$u(a)=0$，通解是 $Ae^{kx}+Be^{-kx}$。"
                    "由 $u(a)=0$ 得 $A=-Be^{-2ka}$，再代入 $u'(a)=s$ 可解出 $u(b;s)=\\frac{s}{k}\\sinh(kL)$，"
                    "于是 $S=\\frac{\\sinh(kL)}{k}\\approx\\frac{e^{kL}}{2k}$。衰减算例 $u(0)=1$、$u(L)=0$ "
                    "的真解是 $\\frac{\\sinh(k(L-x))}{\\sinh(kL)}$，它的灵敏度是同一个 $S$。"
                    "所以“灵敏度大”本身不说明打靶会失败，还要看真解的大小。这个例子里 $S$ 随 $kL$ 指数增长，"
                    "而真解本身是指数衰减的；两者的比值才是打靶要面对的有效放大倍数。"
                ),
                (
                    "**线性问题一步 Newton，非线性问题二次收敛。** 线性方程里 $u(b;s)=u_0+s\\,z_1(b)$ 是 $s$ 的一次函数，"
                    "$F$ 是仿射的，Newton 一步就落到根上，程序再算一次只是确认。非线性问题里 $F''\\ne0$，"
                    "Newton 的残差按 $|F_{k+1}|\\le C|F_k|^2$ 下降，通常五六步就到 $10^{-12}$。"
                    "收敛判据用相对步长而不是绝对残差，因为残差本身有 $u\\,S$ 量级的地板。"
                ),
                (
                    "**衰减模态的失效是舍入放大。** 真解是两个指数 $e^{kx}$ 与 $e^{-kx}$ 相减，$e^{kL}$ 远大于解本身。"
                    "左端斜率有一个小扰动 $\\delta$，传到右端变成 $S\\delta$；要把右端压到解的量级，$\\delta$ 必须小于 $u/S$，"
                    "而 $s$ 本身只能分辨到 $u|s|$。两者之比约为 $e^{kL}$，所以相对误差按 $\\varepsilon e^{kL}$ 增长，"
                    "与步长 $h$ 无关。$kL\\approx\\ln(1/u)\\approx36$ 时，$e^{kL}$ 已经和 $1/u$ 同量级，再缩小步长也救不回来。"
                    "增长算例的真解是指数大的那一支，两个模态相减不丢位数，所以同样的 $S$ 却能算得很准。"
                    "这也解释了为什么缩小步长没有用：误差来自浮点表示 $s$ 时的相对精度，而不是来自时间离散；"
                    "把积分器换得更准，只会让两个大模态相减得更精确地抵消，仍然救不回被丢掉的位数。"
                ),
                (
                    "**参数与读图。** 扫描斜率取 $s_j=s_{\\mathrm{ref}}(1+\\mathrm{span}(2j/(m-1)-1))$，"
                    "拖时间轴只是选中第 $j$ 条；$\\mathrm{span}$ 控制整族散开的宽度，$m$ 控制条数。"
                    "副轴同时画残差 $|F(s_j)|$ 和整条轨迹与真解的偏差：衰减算例里两者一起被 $S$ 放大，"
                    "残差贴地板并不等于解准。参考斜率 $s_{\\mathrm{ref}}$ 是再打靶一次得到的，不是解析真值，"
                    "拿它做“精确对照”是循环论证。轨迹飞出上限时会被截断并标记，读图时不要把占位值当成真实值。"
                ),
                (
                    "**什么时候该换方法。** 当 $kL$ 接近 36，衰减算例的打靶已经不可救；此时应改用有限差分或谱配置，"
                    "它们把边界条件直接写进方程组，不经过“大数相减”。增长算例则仍可打靶，因为放大的是真解本身。"
                    "判断依据不是 $S$ 的大小，而是 $S$ 与真解量级之比：这个比值才决定舍入放大后还剩几位有效数字。"
                ),
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "convergence_rate",
            "title": "有限差分 vs 谱配置：误差衰减的代数与指数之别",
            "description": (
                "同一批 $N$ 上并列画两条误差曲线：有限差分（二阶中心差商 + 边界格式 bc_order）"
                "与 Chebyshev 谱配置。左轴半对数（指数收敛是直线），右轴双对数（代数收敛是直线），"
                "两条**实测**拟合线叠加在数据上，地板线单独画。选 smooth（解析解）时谱误差"
                "以每节点约 0.9 位数的速率掉到 1e-14 的地板；选 abs3/abs5（$C^2/C^4$）时"
                "退化成幂律，而且插值误差比配置解误差**好一阶**——图里同时给出插值曲线作负对照。"
                "有限差分那条曲线带边界格式开关：一阶边界会把整体阶从 $2$ 拉到 $1$。"
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
                    "help": (
                        "选择测试解。smooth 是解析函数，谱方法呈指数收敛；abs3 与 abs5 只有有限阶光滑度，"
                        "会退化成幂律；tanh 的过渡层和 neumann 边界用来观察边界格式的影响。"
                        "切换后先看曲线是直线还是弯的，再看误差地板。"
                    ),
                },
                {
                    "name": "bc_order",
                    "label": "边界差商阶",
                    "type": "enum",
                    "options": ["1", "2"],
                    "default": "2",
                    "critical": [],
                    "criticalNote": "",
                    "help": (
                        "边界差商用一阶还是二阶。内部始终是二阶中心差商，所以整体阶由边界格式决定："
                        "一阶边界会把整体阶拉到 1，二阶边界才回到 2。切到一阶后，差分曲线整体抬高，"
                        "副轴上的实测斜率也会跟着变。"
                    ),
                },
                {
                    "name": "nmax",
                    "label": "最大 $N$",
                    "type": "int",
                    "min": 16,
                    "max": 512,
                    "step": 16,
                    "default": 256,
                    "critical": [],
                    "criticalNote": "",
                    "help": (
                        "有限差分允许的最大网格数，也决定误差曲线的右端。调大能看到更细的网格和更低的误差，"
                        "但也更容易撞上舍入地板；调小则窗口变短，拟合出的阶更不稳定。"
                        "谱方法的节点数按比例随它变化。"
                    ),
                },
            ],
            "time": {"mode": "none"},
            "requiredAux": ["error", "residual"],
            "axes": [
                {
                    "id": "main",
                    "autoscaled": False,
                    "title": "误差 vs $N$：半对数（指数收敛是直线）",
                    "ylabel": "$\\max|u_{\\mathrm{num}}-u_{\\mathrm{exact}}|$",
                    "scale": "log",
                },
                {
                    "id": "aux",
                    "autoscaled": False,
                    "title": "双对数误差与边界残差",
                    "ylabel": "误差 / 边界残差",
                    "scale": "log",
                },
            ],
            "pitfalls": (
                "第一，两种收敛的判别不能只看 $R^2$：有限窗口里幂律拟合解析解也能给出 $R^2\\approx 0.98$"
                "（本章实测 tanh $k=3$ 的窗口内两种模型难分）。判据要用**误差比**"
                "$\\mathrm{err}(N)/\\mathrm{err}(N+2)$：指数收敛的比值整体上抬（smooth 实测 39→76→197），"
                "幂律收敛的比值压向常数（abs3 实测 1.56→1.14）。第二，报阶必须写窗口："
                "同一批 mixed 边界数据，$n=16\\ldots 128$ 拟合出 1.86，$n=256\\ldots 1024$ 才是 1.98。"
                "第三，幂律拟合别拿解析解的**插值**误差冒充配置解误差（$|x|^3$ 实测 3.00 vs 1.99）。"
            ),
            "theory": [
                (
                    "**两种误差来源。** 有限差分把导数换成差商，每个网格点都留下 $O(h^p)$ 的截断误差；"
                    "谱方法则用一个 $N$ 次多项式去逼近解，误差来自“多项式空间装不下这个函数”。"
                    "前者随网格变细按幂律下降，后者在函数解析时按指数下降。"
                    "图上左轴是半对数、右轴是双对数，就是为了让这两种直线分别显形。"
                ),
                (
                    "**有限差分的整体阶由边界决定。** 内部用二阶中心差商，截断误差是 $\\frac{h^2}{12}u^{(4)}$；"
                    "两端各补一行边界条件。若边界用一阶单侧差商，边界行的截断误差是 $O(h)$，它作为右端扰动进入解，"
                    "经过离散算子的逆（范数约 $n^2$）铺满整个区间，最终整体误差仍是 $O(h)$。"
                    "所以整体阶是内部阶与边界阶的较小者；把边界换成二阶格式，整体才回到 $O(h^2)$。"
                ),
                (
                    "**谱配置在节点上满足方程。** 取 $N+1$ 个 Chebyshev–Lobatto 节点，构造 $N$ 次多项式 $u_N$，"
                    "要求它在 $N-1$ 个内点满足微分方程、在两个端点满足边界条件。导数用谱微分矩阵 $D_N$ 计算，"
                    "于是离散方程是 $D_N^2u+\\operatorname{diag}(q)u=r$ 再加两行边界。"
                    "它和有限差分一样是线性系统，但未知数少得多，矩阵是稠密的。"
                ),
                (
                    "**解析函数的指数收敛。** 若 $u$ 在包含 $[-1,1]$ 的某个椭圆内解析，Chebyshev 系数按 "
                    "$\\rho^{-N}$ 衰减，$\\rho>1$ 由最靠近区间的奇点决定。插值误差因此像 $C\\rho^{-N}$，"
                    "在半对数图上是一条斜率 $-\\log\\rho$ 的直线。配置解的误差同阶，因为微分方程只把系数乘上有界因子。"
                    "图上实测的半对数斜率就是 $-\\log\\rho$，每增加一个节点大约多一位有效数字。"
                ),
                (
                    "**非解析函数退化成幂律。** 若 $u$ 只有 $s$ 阶连续导数，插值误差是 $O(N^{-s-1})$。"
                    "配置解要对插值多项式求两次导数，每求一次导数损失一阶，所以误差是 $O(N^{-s})$。"
                    "例如 $|x|^3$ 的插值率约为 3、配置解率约为 2；$|x|^5$ 是 5 与 4。"
                    "图上把插值误差画成负对照，就是为了提醒：看到更漂亮的幂律，先确认它是不是插值而不是配置解。"
                ),
                (
                    "**舍入地板是 $N^4u$。** 谱微分矩阵的无穷范数约为 $N^2$，二阶导数的离散算子约为 $N^4/3$。"
                    "把解的舍入误差 $u$ 乘上这个放大因子，得到 $N^4u$：$N=32$ 时约 $10^{-10}$，"
                    "已经高于很多截断误差。继续增大 $N$，截断误差还在降，舍入误差却按 $N^4$ 涨，"
                    "总误差在某个 $N$ 之后反而上升。图上的地板线与最小误差点就是这条抛物线的谷底。"
                ),
                (
                    "**怎样判断指数还是幂律。** 不要只看 $R^2$：在一个不大的窗口里，幂律也能把指数曲线拟合得很像。"
                    "更稳的判据是误差比 $\\mathrm{err}(N)/\\mathrm{err}(N+2)$：指数收敛时这个比值随 $N$ 增大而增大，"
                    "幂律收敛时它趋于常数。拟合还必须写窗口，同一批数据在粗端和细端可以给出不同的阶；"
                    "细端一旦碰到地板，拟合出的指数就没有意义。切到一阶边界格式，差分曲线会整体抬高，"
                    "实测阶也应跟着变成 1。"
                ),
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "node_clustering",
            "title": "Chebyshev 节点：端点聚集与插值误差的位置",
            "description": (
                "上排画 $N+1$ 个 Chebyshev-Lobatto 节点的位置、相邻间距、以及解析密度"
                "$\\rho(x)=N/(\\pi\\sqrt{1-x^2})$：节点在端点附近以 $1/\\sqrt{1-x^2}$ 聚集，"
                "最大/最小间距比 $\\approx(2/\\pi)N$（$N=128$ 实测 81.5）。"
                "下排是同一批节点的**插值**误差曲线：节点上恒为 $0$，节点之间才是误差，"
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
                    "help": (
                        "选择被插值的函数。smooth 解析，增加节点会让误差指数下降；abs3 与 abs5 在原点只有有限阶光滑度，"
                        "误差按幂律下降；tanh 的陡峭程度由下一个参数控制。"
                        "切换后比较插值误差与配置解误差的差距。"
                    ),
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
                    "help": (
                        "节点数 $N$，实际画 $N+1$ 个点。调大后端点间距按 $N^{-2}$ 变小，间距比按 $2N/\\pi$ 变大，"
                        "插值误差下降；但微分矩阵的条件数也按 $N^2$ 增长，超过舍入地板后继续加密反而更差。"
                    ),
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
                    "help": (
                        "只对 tanh 算例生效的陡度参数。调大后过渡层变窄，函数更接近不解析，谱收敛退化成幂律，"
                        "需要更多节点才能压住误差；调小则接近光滑函数。其他算例忽略它，"
                        "切换算例后这个滑块不会改变曲线。"
                    ),
                },
            ],
            "time": {"mode": "none"},
            "requiredAux": ["error", "collocation_error"],
            "axes": [
                {
                    "id": "main",
                    "autoscaled": False,
                    "title": "节点、间距与密度",
                    "ylabel": "$x_j$ 与间距",
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
                "条件数：节点间距最小 $\\sim N^{-2}$，微分矩阵 $\\|D\\|_\\infty=N^2$，"
                "所以 $N$ 不能一味加大（smooth 算例 $N\\ge 24$ 后误差反而变大）。"
                "第三，密度曲线是**解析式** $N/(\\pi\\sqrt{1-x^2})$，不是从节点间距统计出来的，"
                "两者在 $N$ 小的时候差得很明显（端点处解析值发散，实际间距有限）。"
            ),
            "theory": [
                (
                    "**节点是余弦投影。** Chebyshev–Lobatto 节点定义为 $x_j=\\cos\\frac{j\\pi}{N}$，"
                    "$j=0,\\ldots,N$，按降序排列并包含两个端点。它等价于在 $\\theta\\in[0,\\pi]$ 上取均匀网格 "
                    "$\\theta_j=\\frac{j\\pi}{N}$，再做 $x=\\cos\\theta$ 的投影。投影把均匀的 $\\theta$ 挤压到 $x$ 的两端，"
                    "所以节点在端点密、在中心疏。"
                ),
                (
                    "**间距的闭式。** 相邻节点之差为 $x_j-x_{j+1}=\\cos\\frac{j\\pi}{N}-\\cos\\frac{(j+1)\\pi}{N}"
                    "=2\\sin\\frac{(2j+1)\\pi}{2N}\\sin\\frac{\\pi}{2N}$。最靠端点的一格 $j=0$ 约为 "
                    "$\\frac{\\pi^2}{2N^2}$，中心附近 $j\\approx N/2$ 约为 $\\frac{\\pi}{N}$。"
                    "两者之比约为 $\\frac{2N}{\\pi}$，这就是图上“间距比随 $N$ 线性增长”的来源；$N=128$ 时约为 81.5。"
                ),
                (
                    "**密度是间距的倒数。** 在 $\\theta$ 上节点均匀，密度是 $\\frac{N}{\\pi}$；换到 $x$ 坐标要用 "
                    "$|\\mathrm{d}\\theta/\\mathrm{d}x|=\\frac{1}{\\sin\\theta}=\\frac{1}{\\sqrt{1-x^2}}$，"
                    "所以 $\\rho(x)=\\frac{N}{\\pi\\sqrt{1-x^2}}$。它在端点发散，但积分 "
                    "$\\int_{-1}^{1}\\rho\\,\\mathrm{d}x=N$ 仍然有限。图上的橙虚线画的是 $1/\\rho$，"
                    "它和实测间距同量纲，便于直接比较；小 $N$ 时两者在端点附近差别明显，因为解析式是连续极限。"
                ),
                (
                    "**为什么要把节点堆在端点。** 多项式插值误差是 "
                    "$f(x)-I_Nf(x)=\\frac{f^{(N+1)}(\\xi)}{(N+1)!}\\prod_j(x-x_j)$。在等距节点上，"
                    "乘积在端点附近增长得很快，可能出现 Runge 现象；Chebyshev 节点让这个乘积的无穷范数最小，"
                    "等于 $2^{-N}$，把误差均匀地压下去。代价是端点的间距变成 $O(N^{-2})$，"
                    "微分矩阵的条件数随之变大。"
                ),
                (
                    "**节点上误差为零，不代表精度。** 插值多项式按定义经过每个节点，所以在 $x_j$ 上 "
                    "$f-I_Nf=0$；图上那些贴在地板上的点不是“测到很小的误差”，而是“这里没有误差可测”。"
                    "真正的误差在节点之间，峰值才是有意义的量。配置解误差又是另一回事：它不经过真解，"
                    "非解析算例下通常比插值误差差一阶。"
                ),
                (
                    "**Lebesgue 常数与条件数。** 插值算子范数对 Chebyshev 节点按 "
                    "$\\frac{2}{\\pi}\\log N+O(1)$ 增长，所以增加节点不会让插值爆炸；但微分矩阵的范数是 $N^2$，"
                    "配置矩阵的条件数还要再乘 $N^2$ 量级。$N$ 太大时，舍入误差 $N^4u$ 盖过截断误差，"
                    "误差曲线反而抬头。这就是“节点不是越多越好”的定量原因。"
                ),
                (
                    "**读图。** 上排蓝点是节点位置，灰虚线是同样点数的均匀网格，两者在端点的分离就是聚集；"
                    "绿方块是相邻间距，放在另一个标尺上，端点的小格和中心的大格一眼可见。下排左边是间距随位置的变化，"
                    "橙虚线是 $1/\\rho$，灰虚线是均匀间距 $2/N$；右边是插值误差与配置解误差，"
                    "节点位置画在地板上表示误差为零。$k$ 只对 tanh 算例生效；其他算例下拖动它不会改变曲线，"
                    "这不是程序没反应。"
                ),
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "diff_matrix_structure",
            "title": "谱微分矩阵：稠密、奇异、条件数 $N^2$",
            "description": (
                "热力图画出 $N+1$ 阶 Chebyshev 微分矩阵 $D_N$（用 $\\mathrm{log}_{10}|D|$ 着色，符号单独给掩码），"
                "对角元由行和为零反解（$D_{00}=(2N^2+1)/6$，不是 $0$）；右边两条副轴分别是行和 $D_N\\cdot 1$（恒为 $0$，"
                "零空间就是常数向量）与奇异值谱（$\\sigma_{\\mathrm{min}}$ 是 1e-17 级的噪声，"
                "真正有意义的是挖掉零空间后的 $\\sigma_{\\mathrm{min}}^+$ 与 $\\kappa^+\\approx N^2$）。"
                "再叠一条「对解析解求导的逐点误差」，展示谱微分在节点上的精度与"
                "在 $N$ 变大时 $N^4\\cdot u$ 的舍入放大。"
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
                        "$N\\approx 24$ 起 $N^4\\cdot u$ 的舍入放大（$24^4\\cdot u\\approx 10^{-12}$）开始盖过截断误差，"
                        "这就是「再加大 N 没有意义」的实测位置。"
                    ),
                    "beyond": "above",
                    "help": (
                        "矩阵阶 $N$，实际是 $(N+1)\\times(N+1)$。调大后热力图更密，$\\|D_N\\|_\\infty=N^2$ 与条件数一起增长，"
                        "谱微分的舍入误差按 $N^4$ 放大；$N$ 超过二十几以后，逐点误差往往不再下降。"
                    ),
                },
                {
                    "name": "which",
                    "label": "矩阵",
                    "type": "enum",
                    "options": ["D", "D2"],
                    "default": "D",
                    "critical": [],
                    "criticalNote": "",
                    "help": (
                        "选择看一阶微分矩阵 $D$ 还是二阶矩阵 $D^2$。$D$ 的零空间是常数，行和恒为零；"
                        "$D^2$ 的零空间是常数与线性函数，条件数更大、对角元也不同。"
                        "切换后奇异值谱与理论参考线会一起换。"
                    ),
                },
            ],
            "time": {"mode": "none"},
            "requiredAux": ["residual", "error_derivative"],
            "axes": [
                {
                    "id": "main",
                    "autoscaled": False,
                    "title": "$\\log_{10}|D_N|$ 热力图（行主序，含零对角线与零空间）",
                    "ylabel": "行 $i$",
                },
                {
                    "id": "aux",
                    "autoscaled": False,
                    "title": "行和残差与奇异值谱（对数）",
                    "ylabel": "$\\sigma$ / 残差",
                    "scale": "log",
                },
            ],
            "pitfalls": (
                "第一，$D_N$ 是**奇异**矩阵，直接把 $\\sigma_{\\mathrm{max}}/\\sigma_{\\mathrm{min}}$ 当条件数会得到 1e17"
                "这种毫无意义的数（那是零空间的舍入噪声）；正确口径是挖掉零空间后的 $\\kappa^+$，"
                "它按 $N^{1.95}$ 增长。第二，$D_N$ 稠密、条件数 $N^2$，不能拿它去「解方程组」："
                "配置法解的是 $L=D_N^2+\\mathrm{diag}(q)$ 再加两行边界，$\\kappa(L)$ 按 $N^{4.4}$ 增长"
                "（$N=256$ 实测 2.8e9），所以 $N$ 的上限由条件数而不是内存决定。"
                "第三，矩阵没有零对角线可省：对角元由行和为零反解出 $D_{00}=(2N^2+1)/6$"
                "（$N=16$ 实测 85.5），把它当稀疏结构去优化会写错。"
            ),
            "theory": [
                (
                    "**谱微分矩阵的定义。** 给定 $N+1$ 个节点和节点值 $u_j$，存在唯一的 $N$ 次插值多项式 $p$。"
                    "谱微分矩阵 $D_N$ 的作用是：把 $u$ 映成 $p'$ 在节点上的值，$(D_Nu)_i=p'(x_i)$。"
                    "因此它精确地微分“节点数据对应的多项式”，误差只来自多项式本身与真函数的差别，"
                    "而不是差商近似。"
                ),
                (
                    "**非对角元与对角元。** 对 Lagrange 基函数求导并在节点取值，得到 "
                    "$D_{ij}=\\frac{c_i}{c_j}\\frac{(-1)^{i+j}}{x_i-x_j}$，其中 $c_0=c_N=2$、其余 $c_k=1$。"
                    "对角元不能由这个公式直接取，因为分母为零，而是用“常数函数的导数为零”反解："
                    "$D_{ii}=-\\sum_{j\\ne i}D_{ij}$。于是每行之和为零，$D_N\\mathbf{1}=0$。"
                    "特别地 $D_{00}=\\frac{2N^2+1}{6}$，$N=16$ 时约为 85.5，绝不是零。"
                ),
                (
                    "**零空间与奇异性。** $D_N\\mathbf{1}=0$ 说明常数向量在零空间里；$N$ 次多项式的导数最高 $N-1$ 次，"
                    "所以 $D_N$ 的秩是 $N$，它是一个 $(N+1)\\times(N+1)$ 的奇异矩阵。奇异值谱里最小的那个约 "
                    "$10^{-17}$，那是零空间的舍入噪声，不是“精度很高”。直接把 $\\sigma_{\\max}/\\sigma_{\\min}$ "
                    "当条件数会得到 $10^{17}$ 这种没有意义的数；正确的做法是挖掉零空间，再看 "
                    "$\\kappa^+=\\sigma_{\\max}/\\sigma_{\\min}^+$，它按 $N^2$ 量级增长。"
                ),
                (
                    "**为什么不能拿 $D_N$ 解方程组。** 配置法解的是 $L=D_N^2+\\operatorname{diag}(q)$ 再加两行边界，"
                    "不是 $D_N$ 本身。$D_N^2$ 的无穷范数约为 $N^4/3$，条件数随 $N$ 增长得更快，"
                    "实测约按 $N^{4.4}$ 增长，所以 $N$ 的上限由条件数决定，而不是内存。"
                    "把 $D_N$ 当稀疏矩阵优化也会写错：它稠密，而且对角元非零。"
                ),
                (
                    "**谱微分的舍入放大。** 用 $D_N$ 对精确解求导，误差由两部分组成：截断误差随 $N$ 指数下降，"
                    "舍入误差随 $N^4u$ 上升。两者相交处就是最佳 $N$。$N=24$ 时 $N^4u$ 约为 $10^{-12}$，"
                    "已经接近双精度的实用地板；再加大 $N$，逐点误差不再下降，甚至变大。"
                    "图上画出的 $N^4u$ 参考线就是这条预期地板，实际误差高于它说明还在截断区，"
                    "低于它说明已经进入舍入区。"
                ),
                (
                    "**热力图怎么读。** 颜色是 $\\log_{10}|D_{ij}|$，所以深色格表示元素接近零，"
                    "但“接近零”不等于“结构上的零”。矩阵几乎处处非零，对角线上也不空；行和残差恒为 "
                    "$10^{-16}$ 量级，正是因为对角元按行和反解。右侧奇异值从 $\\sigma_{\\max}\\approx N^2$ "
                    "一路降到 $10^{-17}$，中间没有真正的“小但可用”的区间；挖掉零空间后的 "
                    "$\\sigma_{\\min}^+$ 才是可用的最小奇异值。"
                ),
                (
                    "**切换 $D$ 与 $D^2$。** $D$ 的零空间是常数，$D^2$ 的零空间是常数与线性函数，"
                    "所以 $D^2$ 有两个接近零的奇异值；它的对角元公式与 $D$ 不同，热力图的色标范围也会变。"
                    "无论看哪一个，行和都应为零，这是“微分算子吃掉常数”的直接结果。"
                    "$N$ 的临界值约在二十几：再大，舍入放大开始盖过截断误差，右下方的逐点误差曲线会抬头。"
                ),
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
