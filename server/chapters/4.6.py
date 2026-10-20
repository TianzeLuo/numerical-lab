r"""4.6 章：谱方法与伪谱（4 个可视化）。

四个可视化共用一条证据链：**同一张模态系数图，换一个地方看它会不会骗你**。

1. ``modal_decay`` —— 招牌。双对数模态衰减曲线 + 局部斜率 + 逼近误差-自由度阶梯。
   ``analytic`` 剖面的系数**恰好**是 :math:`\rho^{|k|}`（Poisson 核，不用拟合就能对拍），
   ``corner``/``jump`` 只按代数衰减；同一个「斜率」读数在三类光滑性上分别是
   「每步更负」、「常数」、「根本不降」。Chebyshev 侧照同一口径做一遍：
   光滑（Runge）指数、尖角（|x|）代数、跳变（sign）不降。
2. ``dealias_rule`` —— 3/2 规则的三项账：**去掉多少混叠**、**代价是多少截断**、
   **污染落在哪**。三条参照物分开算（朴素乘积 / 2/3 截断 / 高分辨率参照），
   因为把「混叠」和「截断」混成一条曲线就说不清 3/2 规则到底做了什么。
3. ``convergence_race`` —— 同一个线性平流 PDE、同一个初值，谱方法 vs 二阶/四阶中心差分
   的误差-自由度阶梯。谱误差每加倍**越降越快**（指数收敛），差分每加倍**恒为 1/4、1/16**
   （代数收敛）——判据是「每加倍比值是否在变」，不是「谁更小」。
4. ``burgers_cascade`` —— Burgers 湍流化：能谱随时间从单模态长成 :math:`k^{-2}` 段、
   能流 :math:`\Pi(k)` 向高波数爬、解里长出激波。所有数字（斜率、能流平台、耗散率、
   能量收支残差）都进 meta，``dealias`` 开关就在滑块上。

帧契约（与 4.1 章一致）：``meta`` 只放有限标量（``allow_nan=False``，非有限值一律换成
哨兵 + 显式掩码），字符串走 ``strings``，数组走 ``add``；对数轴的地板与掩码在 ``strings``
里逐条说明。**帧里的数学全部来自** :mod:`numcore.pde.spectral`，本文件只做搬运与记账。
"""

from __future__ import annotations

import functools
import math
from typing import Any, Mapping

import numpy as np

from numcore.pde import spectral as sp
from server.nlf import Frame

__all__ = ["MANIFEST", "compute"]

#: 单位舍入（= eps/2）：对数轴地板与「精确 0」的显示约定都用它。
_ROUNDOFF = float(np.finfo(float).eps) / 2.0

#: 对数轴地板：比这更小的值一律显示为地板，掩码同时置位。
_ERR_FLOOR = 1.0e-17

#: 模态系数地板：比这更小的系数（含恒为 0 的偶次谐波）显示为地板。
_COEF_FLOOR = 1.0e-18

#: 解曲线的显示上限：激波附近超过它就地截断显示，**不隐藏**（meta.u_clip 置位）。
_U_CLIP = 1.6

#: 平流速度：取 0.85 而不是 1，避免 t=0.5、1.0 处位移正好是半周期/整周期——
#: 那两个时刻所有格式的误差都会退化到舍入级（实测 t=1.0 时谱误差 5.7e-15，
#: 与 N 无关），曲线会看起来「谱方法永远精确」。
_ADV_C = 0.85

_T_MAX = {"convergence_race": 1.0, "burgers_cascade": 1.5}

_NONFINITE_POLICY = (
    "本帧不含任何非有限值：对数轴上的 0 用 strings.floor（={floor:g}）占位并配掩码"
    "（网格值本身按实数写入，不用 NaN 分隔折线——NaN 会让前端静默丢段）；"
    "解超过显示上限 ±{uc:g} 的位置截断并配 *_clipped 掩码、meta.u_clip 置位；"
    "meta 一律是有限标量（.nlf 头用 allow_nan=False 编码，塞一个 NaN 整帧就写不出去）"
)

_TIME_SEMANTICS = {
    "modal_decay": "无时间轴（一次性扫整条衰减曲线与误差阶梯）",
    "dealias_rule": "无时间轴（对静态测试场做一次乘积，看混叠落在哪）",
    "convergence_race": "真实时间 t：同一条平流解推进到 t（t=0 时所有格式都精确，误差全 0）",
    "burgers_cascade": "真实时间 t：从初值推进到 t，能谱/能流/解曲线都取当前时刻",
}


def _fin(x: Any, default: float = 0.0) -> float:
    """把非有限标量（含 ``None``）换成哨兵（``.nlf`` 头用 ``allow_nan=False``）。"""
    if x is None:
        return float(default)
    v = float(x)
    return v if math.isfinite(v) else float(default)


def _num_or_none(x: Any) -> float | None:
    """可用就返回有限浮点，否则返回 ``None``（meta 允许 null，但**不允许** NaN/Inf）。"""
    if x is None:
        return None
    v = float(x)
    return v if math.isfinite(v) else None


def _floor(arr: Any, floor: float = _ERR_FLOOR) -> np.ndarray:
    """对数轴用的地板占位（**不是**把值改成 0：图上是「量不出来」）。"""
    return np.maximum(np.asarray(arr, dtype=float), float(floor))


def _mask(arr: Any, floor: float = _ERR_FLOOR) -> np.ndarray:
    """地板掩码：1 表示该点被地板占位（或非有限），0 表示是真值。"""
    a = np.asarray(arr, dtype=float)
    return np.where(np.isfinite(a) & (a > float(floor)), 0.0, 1.0)


def _clip(arr: Any, limit: float) -> np.ndarray:
    return np.clip(np.asarray(arr, dtype=float), -float(limit), float(limit))


def _clip_mask(arr: Any, limit: float) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    return np.where(np.isfinite(a) & (np.abs(a) <= float(limit)), 0.0, 1.0)


def _fold_diff(diff: np.ndarray, n: int) -> np.ndarray:
    """把标准 FFT 顺序的偏差谱折成单边 ``|k|`` 曲线（``k=0..n/2``）。"""
    d = np.asarray(diff, dtype=float).reshape(-1)
    half = n // 2
    out = np.zeros(half + 1)
    out[0] = d[0]
    for j in range(1, half):
        out[j] = max(d[j], d[n - j])
    out[half] = d[half]
    return out


# --------------------------------------------------------------------------- #
# viz 1：模态衰减与逼近误差
# --------------------------------------------------------------------------- #
#: 四类剖面的名字（两个基共用一套语义，方便对照）
_PROFILE_KEYS = ("analytic", "corner", "jump", "square")

_FOURIER_KIND = {"analytic": "analytic", "corner": "corner", "jump": "jump", "square": "square"}
_CHEB_KIND = {"analytic": "runge", "corner": "abs", "jump": "sign", "square": "abs3"}

_PROFILE_CN = {
    "analytic": "解析（Poisson 核 / Runge 函数）",
    "corner": "C⁰ 尖角（tent / |x|）",
    "jump": "跳变（sawtooth / sign）",
    "square": "方波（Gibbs 过冲 / |x|³）",
}

#: 误差阶梯用的自由度（固定，不随 n 滑块变：副图要能横着比）
_ERR_NS = (8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)

#: 模态衰减拟合窗口的上界（下界固定 4）
_FIT_HI = 48.0


@functools.lru_cache(maxsize=256)
def _coeff_curve(basis: str, profile: str, n: int, rho: float) -> tuple[np.ndarray, np.ndarray]:
    """当前基/剖面在 ``n`` 个自由度上的系数模 ``(k, |a_k|)``（k 从 1 起，去掉均值模态）。"""
    if basis == "fourier":
        kind = _FOURIER_KIND[profile]
        x = sp.fourier_grid(n)
        a = sp.fourier_coeffs(sp.fourier_profile(kind, x, rho=rho))
        half = n // 2
        k = np.arange(1, half + 1, dtype=float)
        return k, np.abs(a[1 : half + 1])
    kind = _CHEB_KIND[profile]
    c = sp.chebyshev_coeffs(sp.cheb_profile(kind, sp.chebyshev_grid(n)))
    k = np.arange(1, n + 1, dtype=float)
    return k, np.abs(c[1 : n + 1])


def _envelope(basis: str, profile: str, k: np.ndarray, amp: np.ndarray, rho: float) -> np.ndarray:
    """理论包络线：解析剖面走 :math:`C\\rho^k`，其余走 :math:`Ck^{-p}`。

    常数 C 由**窗口内第一个非零模态**定（只对齐量级做参照，不是拟合结果）；
    平滑掉零点模态（方波偶次谐波恒 0）避免包络被 1e-18 的地板拖平。
    """
    kv = np.asarray(k, dtype=float)
    av = np.asarray(amp, dtype=float)
    pos = av > max(1e-14 * float(np.max(av)) if av.size else 1.0, 0.0)
    if not np.any(pos):
        return np.full(kv.shape, _COEF_FLOOR)
    k0 = float(kv[pos][0])
    a0 = float(av[pos][0])
    if profile == "analytic":
        r = rho if basis == "fourier" else (3.0 - 2.0 * math.sqrt(2.0))
        return a0 * r ** (kv - k0)
    if basis == "fourier":
        p = {"corner": 2.0, "jump": 1.0, "square": 1.0}[profile]
    else:
        p = {"corner": 2.0, "jump": 1.0, "square": 4.0}[profile]
    return a0 * (kv / k0) ** (-p)


@functools.lru_cache(maxsize=128)
def _error_ladder(basis: str, rho: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """三类光滑性的误差-自由度阶梯 ``(ns, err_smooth, err_corner, err_jump)``。"""
    ns = np.asarray([v for v in _ERR_NS], dtype=float)
    if basis == "fourier":
        _, e_s = sp.fourier_convergence("analytic", ns, rho=rho)
        _, e_c = sp.fourier_convergence("corner", ns, rho=rho)
        _, e_j = sp.fourier_convergence("jump", ns, rho=rho)
    else:
        _, e_s = sp.chebyshev_convergence("runge", ns)
        _, e_c = sp.chebyshev_convergence("abs", ns)
        _, e_j = sp.chebyshev_convergence("sign", ns)
    return ns, e_s, e_c, e_j


def _viz_modal_decay(p: Mapping[str, Any], t: float) -> Frame:
    del t
    basis = str(p["basis"])
    profile = str(p["profile"])
    n = int(p["n"])
    rho = float(p["rho"])
    fr = Frame()

    k, amp = _coeff_curve(basis, profile, n, round(rho, 6))
    env = _envelope(basis, profile, k, amp, rho)
    lo = 4.0
    hi = min(_FIT_HI, float(n) // 2.0)
    fit = sp.modal_decay_fit(k, amp, lo, hi)
    # 局部斜率**只在非零模态上算**：零点模态在对数轴上会产生 ±inf 的假斜率
    # （实测 Chebyshev/Runge 是偶函数、奇次系数恰好为 0，不筛就得到 +56 这种正斜率）
    alive = amp > max(1e-12 * float(np.max(amp)), _COEF_FLOOR)
    k_alive = k[alive]
    amp_alive = amp[alive]
    slope = sp.local_slopes(k_alive, amp_alive)

    ns, e_s, e_c, e_j = _error_ladder(basis, round(rho, 6))
    ratio = sp.doubling_ratios(ns, e_s)
    # 每加倍比值只在**两步都在地板以上**时有意义（地板上的比值恒为 1，反而是上涨的假象）
    ratio_alive = ratio[(e_s[:-1] > 1e-13) & (e_s[1:] > 1e-13)]
    at_floor = e_s <= 1e-14
    n_floor = float(ns[int(np.argmax(at_floor))]) if bool(np.any(at_floor)) else float("nan")
    gibbs = sp.gibbs_overshoot(512) if profile == "square" else float("nan")

    fr.add("mode_k", k, unit="", label="模态序号 k")
    fr.add("coeff_abs", _floor(amp, _COEF_FLOOR), unit="", label="|a_k|（地板占位）")
    fr.add("coeff_theory", _floor(env, _COEF_FLOOR), unit="", label="理论包络 C·rho^k 或 C·k^-p")
    fr.add("coeff_floor_mask", _mask(amp, _COEF_FLOOR), unit="", label="1=该模态低于地板（含恒为 0 的谐波）")
    fr.add("slope_k", k_alive[:-1], unit="", label="局部斜率的横坐标（只含非零模态）")
    fr.add("local_slope", slope, unit="", label="dlog|a_k|/dlog k（窗口内）")
    fr.add("slope_floor_mask", _mask(np.abs(slope), 1e-12), unit="", label="1=斜率不可测（系数在地板上）")
    fr.add("error_n", ns, unit="", label="自由度 N")
    fr.add("error_smooth", _floor(e_s), unit="", label="光滑剖面逼近误差")
    fr.add("error_corner", _floor(e_c), unit="", label="尖角剖面逼近误差")
    fr.add("error_jump", _floor(e_j), unit="", label="跳变剖面逼近误差")
    stacked = np.vstack([_mask(e_s), _mask(e_c), _mask(e_j)])
    fr.add("error_floor_mask", np.max(stacked, axis=0), unit="", label="1=该 N 上三条误差里有落在地板的")
    fr.add("ratio_n", ns[1:], unit="", label="每加倍比值曲线的横坐标")
    fr.add("ratio_smooth", _floor(ratio, 1e-12), unit="", label="光滑剖面每加倍误差比（恒定=代数，递减=指数）")
    fr.add("ratio_smooth_mask", _mask(ratio, 1e-12), unit="", label="1=该比值落在地板（无信息）")

    fr.set_meta(
        n=n,
        rho=rho,
        n_modes=int(k.size),
        fit_lo=lo,
        fit_hi=hi,
        fit_n_used=int(fit["n_used"]),
        fit_p=_num_or_none(fit["p"]),
        fit_r2=_num_or_none(fit["r2"]),
        fit_dropped=int(fit.get("n_zero_dropped", 0)),
        slope_first=_num_or_none(float(slope[0]) if slope.size else None),
        slope_last=_num_or_none(float(slope[-1]) if slope.size else None),
        slope_steepens=bool(slope.size > 2 and float(slope[-1]) < float(slope[0]) - 1.0),
        coeff_floor=_COEF_FLOOR,
        error_floor=_ERR_FLOOR,
        err_smooth_min=_fin(float(np.min(e_s))),
        err_smooth_last=_fin(float(e_s[-1])),
        err_corner_slope=_num_or_none(float(sp.fit_power_law(ns, e_c)["p"])),
        err_jump_first=_fin(float(e_j[0])),
        err_jump_last=_fin(float(e_j[-1])),
        err_jump_drop=_num_or_none(float(e_j[0] / e_j[-1]) if e_j[-1] > 0 else None),
        n_at_floor=_num_or_none(n_floor),
        ratio_smooth_first=_num_or_none(float(ratio_alive[0]) if ratio_alive.size else None),
        ratio_smooth_last=_num_or_none(float(ratio_alive[-1]) if ratio_alive.size else None),
        ratio_smooth_n=int(ratio_alive.size),
        gibbs_overshoot=_num_or_none(gibbs),
        theory_p=_num_or_none(sp.profile_decay_exponent(_FOURIER_KIND[profile]) or float("nan")),
        roundoff=_ROUNDOFF,
    )
    fr.set_strings(
        time_semantics=_TIME_SEMANTICS["modal_decay"],
        nonfinite_policy=_NONFINITE_POLICY.format(floor=_ERR_FLOOR, uc=_U_CLIP),
        basis=basis,
        profile=_PROFILE_CN[profile],
        floor=f"{_COEF_FLOOR:g}",
        reading_note=(
            "主图：|a_k| 与理论包络（虚线）。副图：局部斜率——**它是否在变**才是指数收敛的判据，"
            "单个拟合 p 值不是（meta.fit_p / fit_r2 一起报）。右图：三类光滑性的逼近误差阶梯，"
            "跳变那条**随 N 不下降**（Gibbs 振荡只被压窄、幅度不变），"
            "尖角那条按幂律下降（实测斜率见 meta.err_corner_slope）。"
        ),
        mask_note=(
            "coeff_floor_mask=1 表示该模态低于 1e-18 地板。**零点模态必须先筛掉**："
            "N 点网格上采样的方波，跳变正好落在节点上，偶次谐波不是 0 而是常数 1/N"
            "（实测 N=64 时 0.03125），它把双对数拟合拖成 p=-0.848、R²=0.346 的假斜率；"
            "Chebyshev 侧的 Runge 函数是偶函数，奇次系数**恰好**为 0，"
            "不筛就直接算局部斜率会得到 +56 这种正斜率（实测）。"
            "meta.fit_dropped 是窗口里被筛掉的模态个数。"
        ),
        theory_note=(
            "C^∞ 解析：a_k ~ rho^{|k|}（Poisson 核系数是精确的 rho^{|k|}，不是拟合）；"
            "C^0 尖角：a_k ~ k^-2；跳变：a_k ~ k^-1；逼近误差各再快一阶。"
            "Chebyshev 侧同口径：Runge 走 (3-2sqrt2)^k=0.1716^k，|x| 走 k^-2，sign 走 k^-1。"
        ),
    )
    return fr


# --------------------------------------------------------------------------- #
# viz 2：3/2 规则与混叠
# --------------------------------------------------------------------------- #
#: 扫描用的输入谱指数（正数：|a_k| = |k|^-decay）
_SCAN_DECAY = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5)


@functools.lru_cache(maxsize=128)
def _alias_scan(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """四项账随输入谱指数的走向 ``(decay, alias, truncation, pad_only)``。

    ``alias``：朴素乘积（不补零不截断）相对**全带**参照的偏差——真·混叠误差；
    ``truncation``：2/3 截断相对全带参照的偏差——它主动丢掉的；
    ``pad_only``：只补零（精确 3N/2、不截断）相对全带参照的偏差；
    ``dealias``：2/3+3/2 相对截断参照的偏差（应当贴舍入，图上压在 1e-16 一条线上）。
    """
    xs = np.asarray(_SCAN_DECAY, dtype=float)
    alias = np.zeros(xs.size)
    trunc = np.zeros(xs.size)
    pad = np.zeros(xs.size)
    deal = np.zeros(xs.size)
    for i, s in enumerate(xs):
        u = sp.synthetic_field(n, float(s), seed=20261018)
        r = sp.alias_error(u, u)
        alias[i] = float(r["alias_rel"])
        trunc[i] = float(r["truncation_rel"])
        pad[i] = float(r["pad_only_rel"])
        deal[i] = float(r["dealias_rel"])
    return xs, alias, trunc, pad, deal


def _viz_dealias_rule(p: Mapping[str, Any], t: float) -> Frame:
    del t
    n = int(p["n"])
    decay = float(p["decay"])
    mode = str(p["mode"])
    exact_pad = bool(p["exact_pad"])
    fr = Frame()

    u = sp.synthetic_field(n, decay, seed=20261018)
    A = sp.fourier_coeffs(u)
    mask = sp.two_thirds_mask(n)
    u_t = np.real(sp.fourier_synthesize(np.where(mask, A, 0.0)))

    r = sp.alias_error(u, u)
    ref_full = sp.product_reference(u, u)
    ref_trunc = sp.product_reference(u_t, u_t)
    if mode == "naive":
        cur = sp.product_naive(u, u, length=1.0)
        cur_ref = ref_full
    elif mode == "pad_only":
        cur = sp.dealias_product(u, u, keep_two_thirds=False, exact=exact_pad)
        cur_ref = ref_full
    else:
        cur = sp.dealias_product(u, u, exact=exact_pad)
        cur_ref = ref_trunc

    half = n // 2
    k_abs = np.arange(half + 1, dtype=float)
    # 注意：product_* 返回的是**系数**（标准 FFT 顺序），不能再做一次 fourier_coeffs
    prod_ref = np.abs(np.asarray(ref_full)[: half + 1])
    prod_cur = np.abs(np.asarray(cur)[: half + 1])
    scale_cur = max(float(np.max(np.abs(cur_ref))), 1e-300)
    cur_rel = float(np.max(np.abs(cur - cur_ref))) / scale_cur
    # 保留带 |k|<=N/3 内的偏差（"污染了有用频段"的那部分）+ 边界模态单独的读数
    kk = np.abs(sp.fourier_wavenumbers(n))
    d_cur = np.abs(np.asarray(cur) - np.asarray(cur_ref)) / scale_cur
    cur_low_rel = float(np.max(d_cur[kk <= float(n) / 3.0]))
    cur_nyq_rel = float(d_cur[half])

    # 能量恒等式残差：朴素乘积 vs 去混叠乘积（口径必须写清楚：都对截断后的场）
    a_t = np.where(mask, A, 0.0)
    resid_naive = sp.nonlinear_energy_residual(a_t, sp.product_naive(u_t, u_t))
    resid_deal = sp.nonlinear_energy_residual(a_t, sp.dealias_product(u_t, u_t))
    resid_full_naive = sp.nonlinear_energy_residual(A, sp.product_naive(u, u))

    xs, s_alias, s_trunc, s_pad, s_deal = _alias_scan(n)

    fr.add("k_abs", k_abs, unit="", label="|k|（单边）")
    fr.add("prod_ref", _floor(prod_ref, _COEF_FLOOR), unit="", label="全带参照乘积谱 |W_ref|")
    fr.add("prod_cur", _floor(prod_cur, _COEF_FLOOR), unit="", label="当前模式乘积谱 |W_cur|")
    fr.add("prod_floor_mask", _mask(np.maximum(prod_ref, prod_cur), _COEF_FLOOR), unit="", label="1=两条谱都低于地板")
    fr.add("alias_error_spectrum", _floor(_fold_diff(r["spectrum"], n), _COEF_FLOOR), unit="", label="|W_naive-W_ref,full| 折叠到 |k|")
    fr.add("alias_error_mask", _mask(_fold_diff(r["spectrum"], n), _COEF_FLOOR), unit="", label="1=该模偏差低于地板")
    fr.add("scan_decay", xs, unit="", label="输入谱指数 decay（|a_k|=|k|^-decay）")
    fr.add("scan_alias", _floor(s_alias), unit="", label="混叠误差（朴素 vs 全带参照）")
    fr.add("scan_truncation", _floor(s_trunc), unit="", label="2/3 截断代价（截断参照 vs 全带参照）")
    fr.add("scan_pad_only", _floor(s_pad), unit="", label="只补零到 3N/2、不截断的偏差")
    fr.add("scan_dealias", _floor(s_deal), unit="", label="2/3+3/2 对截断参照的偏差（贴舍入）")
    fr.add(
        "scan_floor_mask",
        np.max(np.vstack([_mask(s_alias), _mask(s_trunc), _mask(s_pad), _mask(s_deal)]), axis=0),
        unit="",
        label="1=该衰减指数上四条曲线里有落在地板的",
    )
    fr.add("k_keep_line", np.asarray([float(n) / 3.0, float(n) / 3.0]), unit="", label="2/3 带边缘 k=N/3")

    fr.set_meta(
        n=n,
        decay=decay,
        exact_pad=bool(exact_pad),
        m_used=int(sp.extended_size(n, 1.5, exact=exact_pad)),
        k_keep_max=_fin(float(n) / 3.0),
        alias_rel=_fin(r["alias_rel"]),
        alias_low_rel=_fin(r["alias_low_rel"]),
        truncation_rel=_fin(r["truncation_rel"]),
        dealias_rel=_fin(r["dealias_rel"]),
        pad_only_rel=_fin(r["pad_only_rel"]),
        naive_vs_dealias_rel=_fin(r["naive_vs_dealias_rel"]),
        cur_rel=_fin(cur_rel),
        cur_low_rel=_fin(cur_low_rel),
        cur_nyquist_rel=_fin(cur_nyq_rel),
        folded_energy_fraction=_fin(r["folded_energy_fraction"]),
        truncated_input_energy_fraction=_fin(r["truncated_input_energy_fraction"]),
        energy_resid_naive=_fin(resid_naive),
        energy_resid_dealias=_fin(resid_deal),
        energy_resid_naive_full=_fin(resid_full_naive),
        energy_resid_scale=_fin(float(np.sum(np.abs(A) ** 2))),
        error_floor=_COEF_FLOOR,
        roundoff=_ROUNDOFF,
    )
    fr.set_strings(
        time_semantics=_TIME_SEMANTICS["dealias_rule"],
        nonfinite_policy=_NONFINITE_POLICY.format(floor=_COEF_FLOOR, uc=_U_CLIP),
        mode=mode,
        floor=f"{_COEF_FLOOR:g}",
        reading_note=(
            "主图：全带参照乘积谱与当前模式乘积谱（竖线是 2/3 带边缘 k=N/3）。"
            "副图：|W_naive-W_ref,full| 的折叠谱——混叠**落在哪些模态**一眼可见："
            "它铺满整个保留带（不是只碰高波数）。"
            "扫描图四条线：真·混叠误差、2/3 截断代价、只补零不截断的偏差、"
            "2/3+3/2 对截断参照的偏差（压在最下面）。"
        ),
        energy_note=(
            "能量恒等式残差 meta.energy_resid_*：**别把它当成去混叠的证明**。"
            "对已经 2/3 截断的场，折返落在保留带之外，朴素乘积与去混叠乘积的残差都是舍入级"
            "（实测 1e-17 量级）；把场放宽到全带宽，朴素乘积的残差才升到 1e-3~1e-1，"
            "而且强烈依赖相位（同一张谱换相位实测 1e-17 到 2e-1）。"
            "去混叠真正修好的是**非线性项自身的取值**（meta.alias_low_rel），不是总能量账。"
        ),
        truncation_note=(
            "三笔账必须分开读：meta.alias_rel 是朴素乘积相对全带参照的偏差（真混叠）；"
            "meta.truncation_rel 是 2/3 截断自己丢掉的（比混叠还大，实测 decay=1 时 3.7e-2 vs 1.2e-2）；"
            "meta.naive_vs_dealias_rel 是两条算法路线的总差（含两项，不能单独当混叠读）。"
            "meta.pad_only_rel 是只补零到 3N/2、不截断的偏差——保留带内的折返最小落点是 "
            "M-N>=N/2，所以只补零已经够用，残差只剩 |k|=N/2 那条边界模态。"
        ),
    )
    return fr


# --------------------------------------------------------------------------- #
# viz 3：谱方法 vs 有限差分
# --------------------------------------------------------------------------- #
#: 误差阶梯用的自由度（≤ n_max）
_RACE_NS = (16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)


@functools.lru_cache(maxsize=128)
def _race(n_max: int, rho: float, t: float) -> dict[str, Any]:
    """误差-自由度曲线：**必须用当前时刻**（t=0 时所有格式都精确，曲线全在地板上）。"""
    ns = [v for v in _RACE_NS if v <= int(n_max)]
    if len(ns) < 4:
        ns = list(_RACE_NS[:4])
    return sp.advection_error_curve(ns, float(t), c=_ADV_C, rho=float(rho), orders=(2, 4))


def _viz_convergence_race(p: Mapping[str, Any], t: float) -> Frame:
    rho = float(p["rho"])
    n_max = int(p["n_max"])
    t = max(0.0, min(float(t), _T_MAX["convergence_race"]))
    fr = Frame()

    curve = _race(n_max, round(rho, 6), round(t, 4))
    ns = np.asarray(curve["n"], dtype=float)
    err_sp = np.asarray(curve["spectral"], dtype=float)
    err_fd2 = np.asarray(curve["fd2"], dtype=float)
    err_fd4 = np.asarray(curve["fd4"], dtype=float)

    hi = float(np.max(ns))
    # 拟合窗口：取阶梯的上半段（差分格式在小 N 还没进渐近区），且至少 3 个点
    lo_fit = max(64.0, hi / 2.0)
    if int(np.count_nonzero(ns >= lo_fit)) < 3:
        lo_fit = float(ns[max(int(ns.size) - 3, 0)])
    degenerate = bool(max(float(np.max(err_sp)), float(np.max(err_fd2)), float(np.max(err_fd4))) <= 1e-13)
    fit2 = sp.fit_power_law(ns, err_fd2, lo=lo_fit, hi=hi)
    fit4 = sp.fit_power_law(ns, err_fd4, lo=lo_fit, hi=hi)
    ratio_sp = sp.doubling_ratios(ns, err_sp)
    ratio_2 = sp.doubling_ratios(ns, err_fd2)
    ratio_4 = sp.doubling_ratios(ns, err_fd4)
    at_floor = err_sp <= 1e-13
    n_floor = float(ns[int(np.argmax(at_floor))]) if bool(np.any(at_floor)) else float("nan")
    # 谱曲线拟合**只能吃地板以上的点**：拿地板段的假陡峭去拟合会得到 p=-12 这种数
    pre = (~at_floor) & (ns >= lo_fit)
    n_pre = int(np.count_nonzero(pre))
    lo_sp = float(np.min(ns[pre])) if n_pre else float("nan")
    hi_sp = float(np.max(ns[pre])) if n_pre else float("nan")
    span_ok = bool(n_pre >= 3 and hi_sp >= 1.4 * lo_sp)
    fit_sp = (
        sp.fit_power_law(ns, err_sp, lo=lo_sp, hi=hi_sp)
        if span_ok
        else {"p": float("nan"), "r2": float("nan"), "n_used": n_pre}
    )
    # 陷阱数字：把地板段也喂进最小二乘，能得到一条"很陡"的假幂律
    fit_sp_trap = sp.fit_power_law(ns, err_sp, lo=lo_fit, hi=hi)
    # 每加倍比值：只报"两步都在地板以上"的那一段（地板上的比值恒为 1，没有信息）
    ok_ratio = (err_sp[:-1] > 1e-13) & (err_sp[1:] > 1e-13)
    ratio_sp_alive = ratio_sp[ok_ratio] if np.any(ok_ratio) else np.asarray([], dtype=float)

    # 当前时刻的定点误差（用最大自由度画，横坐标是网格点）
    n_big = int(np.max(ns))
    x = sp.fourier_grid(n_big)
    ref = sp.advection_exact(x, t, c=_ADV_C)
    u_sp = sp.advection_spectral(n_big, t, c=_ADV_C, rho=rho)
    u_2 = sp.advection_fd(n_big, t, c=_ADV_C, rho=rho, order=2)
    u_4 = sp.advection_fd(n_big, t, c=_ADV_C, rho=rho, order=4)
    e_sp = np.abs(u_sp - ref)
    e_2 = np.abs(u_2 - ref)
    e_4 = np.abs(u_4 - ref)

    fr.add("error_n", ns, unit="", label="自由度 N（谱=模态数，差分=格点数）")
    fr.add("error_spectral", _floor(err_sp), unit="", label="谱方法相对误差（地板占位）")
    fr.add("error_fd2", _floor(err_fd2), unit="", label="二阶中心差分相对误差")
    fr.add("error_fd4", _floor(err_fd4), unit="", label="四阶中心差分相对误差")
    fr.add(
        "error_floor_mask",
        np.max(np.vstack([_mask(err_sp), _mask(err_fd2), _mask(err_fd4)]), axis=0),
        unit="",
        label="1=该 N 上三条误差里有落在地板的",
    )
    fr.add("ratio_n", ns[1:], unit="", label="每加倍比值曲线的横坐标")
    fr.add("ratio_spectral", _floor(ratio_sp, 1e-12), unit="", label="谱误差每加倍比值（递减=指数收敛）")
    fr.add("ratio_fd2", _floor(ratio_2, 1e-12), unit="", label="二阶差分每加倍比值（预期 0.25）")
    fr.add("ratio_fd4", _floor(ratio_4, 1e-12), unit="", label="四阶差分每加倍比值（预期 0.0625）")
    fr.add("point_x", x, unit="", label="网格点 x")
    fr.add("point_error_spectral", _floor(e_sp), unit="", label="逐点误差 |u_谱-u_精确|")
    fr.add("point_error_fd2", _floor(e_2), unit="", label="逐点误差 |u_FD2-u_精确|")
    fr.add("point_error_fd4", _floor(e_4), unit="", label="逐点误差 |u_FD4-u_精确|")
    fr.add(
        "point_floor_mask",
        np.max(np.vstack([_mask(e_sp), _mask(e_2), _mask(e_4)]), axis=0),
        unit="",
        label="1=该点三条误差里有落在地板的",
    )
    fr.add("point_u_exact", ref, unit="", label="精确解 u(x,t)")

    fr.set_meta(
        rho=rho,
        n_max=n_max,
        t_actual=t,
        c=_ADV_C,
        shift=c_frac(t),
        n_points=int(ns.size),
        fit_lo=lo_fit,
        fit_hi=hi,
        slope_spectral=_num_or_none(fit_sp["p"]),
        slope_spectral_r2=_num_or_none(fit_sp["r2"]),
        slope_spectral_trap=_num_or_none(fit_sp_trap["p"]),
        slope_spectral_trap_r2=_num_or_none(fit_sp_trap["r2"]),
        fit_spectral_lo=_num_or_none(lo_sp),
        fit_spectral_hi=_num_or_none(hi_sp),
        fit_spectral_points=n_pre,
        slope_fd2=_num_or_none(None if degenerate else fit2["p"]),
        slope_fd2_r2=_num_or_none(None if degenerate else fit2["r2"]),
        slope_fd4=_num_or_none(None if degenerate else fit4["p"]),
        slope_fd4_r2=_num_or_none(None if degenerate else fit4["r2"]),
        degenerate=degenerate,
        ratio_spectral_last=_num_or_none(float(ratio_sp[-1]) if ratio_sp.size else None),
        ratio_spectral_first=_num_or_none(float(ratio_sp[0]) if ratio_sp.size else None),
        ratio_fd2_last=_num_or_none(float(ratio_2[-1]) if ratio_2.size else None),
        ratio_fd4_last=_num_or_none(float(ratio_4[-1]) if ratio_4.size else None),
        err_spectral_min=_num_or_none(None if degenerate else float(np.min(err_sp))),
        err_spectral_at_max_n=_fin(float(err_sp[-1])),
        err_fd2_at_max_n=_fin(float(err_fd2[-1])),
        err_fd4_at_max_n=_fin(float(err_fd4[-1])),
        gain_over_fd2=_num_or_none(float(err_fd2[-1] / err_sp[-1]) if err_sp[-1] > 0 else None),
        n_at_floor=_num_or_none(n_floor),
        error_floor=_ERR_FLOOR,
        roundoff=_ROUNDOFF,
    )
    fr.set_strings(
        time_semantics=_TIME_SEMANTICS["convergence_race"],
        nonfinite_policy=_NONFINITE_POLICY.format(floor=_ERR_FLOOR, uc=_U_CLIP),
        floor=f"{_ERR_FLOOR:g}",
        reading_note=(
            "主图：三条误差-自由度曲线（双对数）。谱方法那条**越来越陡**，两条差分是直线。"
            "副图：每加倍误差比——差分的比值是常数（二阶≈0.25、四阶≈0.0625），"
            "谱方法的比值**每步再乘一个更小的数**（meta.ratio_spectral_first→last）。"
            "这才是「指数收敛」的可执行判据；单看某个 N 上谱误差更小，只能说明常数小。"
        ),
        shift_note=(
            "平流速度取 c=0.85 不是随手取的：c=1 时 t=0.5、1.0 处位移正好是半周期/整周期，"
            "所有格式的误差都会退化到舍入级（实测 t=1.0 时谱误差 5.7e-15、与 N 无关），"
            "曲线会看起来「谱方法永远精确」。meta.shift 是当前时刻的实际位移（周期数）。"
            "t=0 时误差恒为 0：初值就是精确解。"
        ),
        floor_note=(
            "谱曲线在 N 足够大后撞上 1e-13 以下的双精度地板（meta.n_at_floor 是第一个撞上的 N），"
            "地板之下误差不再随 N 下降——这不是收敛停滞，是算不动了。"
        ),
    )
    return fr


def c_frac(t: float) -> float:
    """当前时刻的位移（周期数，取小数部分），写进 meta 供前端标注。"""
    s = (_ADV_C * float(t)) % 1.0
    return float(s)


# --------------------------------------------------------------------------- #
# viz 4：Burgers 湍流化
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=64)
def _burgers(n: int, nu: float, dealias: bool, initial: str, t_end: float):
    return sp.burgers_solve(
        int(n),
        float(nu),
        float(t_end),
        dealias=bool(dealias),
        initial=str(initial),
        n_snapshots=9,
    )


def _viz_burgers_cascade(p: Mapping[str, Any], t: float) -> Frame:
    n = int(p["n"])
    nu = float(p["nu"])
    dealias = bool(p["dealias"])
    initial = str(p["initial"])
    t = max(0.0, min(float(t), _T_MAX["burgers_cascade"]))
    fr = Frame()

    res = _burgers(n, round(nu, 6), dealias, initial, round(t, 4))
    k = res.k
    e_now = res.spectra[-1]
    e_ini = res.spectra[0]
    kk = np.maximum(k, 1.0)
    ref_line = e_ini[1] * kk ** (-2.0) if e_ini.size > 1 else np.zeros_like(k)
    lo, hi = sp.cascade_window(nu, max(t, 1e-6), n)
    fit = sp.fit_power_law(k[lo : hi + 1], e_now[lo : hi + 1])
    eps = float(res.dissipation[-1])
    flux_max = float(np.max(res.flux))
    x = sp.fourier_grid(n, length=sp.TWO_PI)
    u_now = res.u[-1]
    frac_high = float(np.sum(e_now[int(np.ceil(n / 3.0)) :]) / max(float(np.sum(e_now)), 1e-300))

    fr.add("k", k, unit="", label="波数 k（单边，±k 合并）")
    fr.add("energy_now", _floor(e_now, _COEF_FLOOR), unit="", label="当前时刻能谱 E(k)（地板占位）")
    fr.add("energy_initial", _floor(e_ini, _COEF_FLOOR), unit="", label="初值能谱 E(k,0)")
    fr.add("energy_ref_slope", _floor(ref_line, _COEF_FLOOR), unit="", label="k^-2 参考线（过初值第一模态）")
    fr.add("energy_floor_mask", _mask(e_now, _COEF_FLOOR), unit="", label="1=当前能谱该模低于地板")
    fr.add("flux", res.flux, unit="", label="能流 Pi(k)（正=流向高波数）")
    fr.add("x", x, unit="", label="网格点 x")
    fr.add("u_now", _clip(u_now, _U_CLIP), unit="", label="当前解 u(x,t)（超出 ±1.6 截断显示）")
    fr.add("u_initial", _clip(res.u[0], _U_CLIP), unit="", label="初值 u(x,0)")
    fr.add("u_clipped", _clip_mask(u_now, _U_CLIP), unit="", label="1=该点被显示上限截断")
    e0 = float(res.energy[0])
    en_err = np.abs(res.energy - e0) / max(e0, 1e-300)
    fr.add("energy_t", res.times, unit="", label="能谱快照时刻")
    fr.add("energy_error", _floor(en_err, 1e-18), unit="", label="|E(t)-E(0)|/E(0)（相对能量漂移，地板占位）")
    fr.add("energy_error_mask", _mask(en_err, 1e-18), unit="", label="1=该时刻漂移低于地板")
    fr.add("budget_t", res.budget_times, unit="", label="能量收支记账时刻")
    fr.add("budget_residual", _floor(res.budget_residual, 1e-18), unit="", label="|E(t)-E(0)+2nu∫Omega| 相对残差")
    fr.add("budget_residual_mask", _mask(res.budget_residual, 1e-18), unit="", label="1=该时刻残差低于地板（=完美闭合）")

    fr.set_meta(
        n=n,
        nu=nu,
        dealias=dealias,
        t_actual=t,
        dt=_fin(res.dt),
        dt_cfl=_fin(res.dt_cfl),
        n_steps=int(res.n_steps),
        k_diss=_fin(1.0 / math.sqrt(max(nu * max(t, 1e-6), 1e-300))),
        fit_lo=lo,
        fit_hi=hi,
        fit_window_points=int(max(hi - lo + 1, 0)),
        slope_fit=_num_or_none(fit["p"]),
        slope_r2=_num_or_none(fit["r2"]),
        slope_theory=-2.0,
        energy_0=_fin(float(np.sum(e_ini))),
        energy_now=_fin(float(np.sum(e_now))),
        energy_drift=_fin(float(np.sum(e_now)) - float(np.sum(e_ini))),
        energy_error_max=_fin(float(np.max(en_err))),
        enstrophy=_fin(res.enstrophy[-1]),
        dissipation=_fin(eps),
        flux_max=_fin(flux_max),
        flux_at_end=_fin(float(res.flux[-1])),
        flux_over_eps=_num_or_none(flux_max / eps if eps > 0 else None),
        budget_error_max=_fin(float(np.max(res.budget_residual))),
        frac_energy_above_2_3=frac_high,
        shock_width_ratio=_fin(nu / (2.0 * (sp.TWO_PI / n))),
        max_u=_fin(float(np.max(np.abs(u_now)))),
        u_clip=_U_CLIP,
        n_clipped=int(np.count_nonzero(_clip_mask(u_now, _U_CLIP))),
        energy_floor=_COEF_FLOOR,
        error_floor=_ERR_FLOOR,
        roundoff=_ROUNDOFF,
    )
    fr.set_strings(
        time_semantics=_TIME_SEMANTICS["burgers_cascade"],
        nonfinite_policy=_NONFINITE_POLICY.format(floor=_ERR_FLOOR, uc=_U_CLIP),
        floor=f"{_COEF_FLOOR:g}",
        initial=initial,
        reading_note=(
            "主图：当前能谱与初值能谱（虚线是 k^-2 参考）。副图：能流 Pi(k)，"
            "k=0 处为 0、在高波数端回落到 0（去混叠下非线性项守恒总能量）；"
            "右图：解曲线——单模态初值长出激波，谱的 k^-2 段就是这条激波的谱。"
        ),
        cascade_note=(
            "**别把它读成 3D 湍流的惯性区**：1D Burgers 的 k^-2 段是激波的谱，"
            "范围由 nu 与 N 共同决定。惯性区判据 Pi(k)≈eps 在这里并不成立"
            "（meta.flux_over_eps 通常 <1）：欠分辨时 2/3 截断直接从高波数拿走能量，"
            "绕过了能流。分辨是否够用看 meta.shock_width_ratio = (nu/2)/dx 与 "
            "meta.frac_energy_above_2_3：前者 <0.1 或后者不小，就说明高波数由网格决定。"
        ),
        energy_note=(
            "meta.budget_error_max 是 E(t)+2nu∫Omega=E(0) 的相对残差，"
            "同时被三件事影响：时间离散、2/3 截断这个**能量汇**、以及欠分辨时的高波数堆积。"
            "实测 n=128,t=1.5 时它从 nu=0.05 的 8e-6 升到 nu=0.002 的 1.6（能量反而涨了 68%）——"
            "低 nu 端的「级联」数字是数值产物，不是物理。"
        ),
    )
    return fr


# --------------------------------------------------------------------------- #
# MANIFEST
# --------------------------------------------------------------------------- #
MANIFEST: dict[str, Any] = {
    "chapter": "4.6",
    "title": "谱方法与伪谱",
    "part": "IV",
    "summary": (
        "谱方法把「解 PDE」换成「算模态系数」，于是所有结论都挂在一张双对数图上："
        "系数怎么衰减、斜率是不是常数、误差每加倍降多少。"
        "光滑数据指数收敛（Poisson 核的系数恰是 $\\rho^{|k|}$）、尖角只按代数收敛、"
        "跳变连收敛都没有（Gibbs 振荡只被压窄）；伪谱法还要多一笔账——"
        "非线性项在 $N$ 点网格上相乘会把高频折回低波数，$2/3$ 截断 + $3/2$ 补零才把它修干净。"
        "四个可视化把这条链拆成：模态衰减（谁的斜率在变）、$3/2$ 规则的三项账"
        "（去掉多少混叠、代价多少截断、污染落在哪）、谱 vs 差分的误差阶梯"
        "（每加倍比值是常数还是在变小）、Burgers 能谱级联（$k^{-2}$ 段与能流从哪来）。"
    ),
    "vizzes": [
        # ------------------------------------------------------------------ #
        {
            "id": "modal_decay",
            "title": "模态衰减：指数、代数、以及根本不降",
            "description": (
                "主图是当前基/剖面在 $N$ 个自由度上的 $|a_k|$ 与理论包络（双对数）。"
                "副图是局部斜率 $d\\log|a_k|/d\\log k$——光滑剖面**斜率本身在变**（越来越负），"
                "尖角是常数，跳变在地板上乱跳。右图把三类光滑性的逼近误差-自由度阶梯"
                "画在一起：跳变那条随 $N$ 不下降（Gibbs），尖角按幂律降，光滑指数降。"
                "basis 换成 chebyshev 是同一套口径的另一个基：Runge 指数、$|x|$ 代数、sign 不降。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "basis",
                    "label": "基",
                    "type": "enum",
                    "options": ["fourier", "chebyshev"],
                    "default": "fourier",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "profile",
                    "label": "剖面光滑性",
                    "type": "enum",
                    "options": ["analytic", "corner", "jump", "square"],
                    "default": "analytic",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "n",
                    "label": "自由度 N",
                    "type": "int",
                    "min": 16,
                    "max": 256,
                    "step": 8,
                    "default": 64,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "rho",
                    "label": "解析剖面的衰减率 rho",
                    "type": "float",
                    "min": 0.3,
                    "max": 0.85,
                    "step": 0.01,
                    "default": 0.6,
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "none"},
            "requiredAux": ["error"],
            "axes": [
                {
                    "id": "decay",
                    "autoscaled": False,
                    "title": "模态系数 |a_k|（双对数，窗口 1e-18–1）",
                    "ylabel": "|a_k|",
                    "scale": "log",
                },
                {
                    "id": "slope",
                    "autoscaled": False,
                    "title": "局部斜率（窗口 −24–0）",
                    "ylabel": "dlog|a_k|/dlog k",
                },
                {
                    "id": "error",
                    "autoscaled": False,
                    "title": "误差 vs N（双对数，1e-17–10）",
                    "ylabel": "||f - S_N f||_inf",
                    "scale": "log",
                },
            ],
            "pitfalls": (
                "第一，「指数收敛」不能只看某一条曲线陡，也不能只看拟合出的 $p$。默认帧"
                "（fourier/analytic/$n=64$/$\\rho=0.6$）在窗口 $k\\in[4,32]$ 上拟出 $p=-7.129$、$R^2=0.938$，"
                "看着比任何代数衰减都陡；但它的**局部斜率**从 $-0.737$ 一路走到 $-3.942$"
                "（meta.slope_first→slope_last），**每段都在变**才是指数收敛的证据，"
                "代数衰减的斜率是常数。判据是 meta.ratio_smooth_first→last（实测 $0.143\\to4.1\\times10^{-4}$，"
                "每加倍再乘一个更小的数），不是某个 $p$ 值。"
                "第二，跳变剖面的误差**不随 $N$ 下降**：实测 Fourier 锯齿的 $L_\\infty$ 逼近误差 $N=16$ 时 $1.956$、"
                "$N=256$ 时 $1.802$（Chebyshev 的 sign 是 $1.997\\to1.905$），Gibbs 过冲把误差钉在跳变附近，"
                "过冲极限 $0.178980$（实测 $K=1024$ 时 $0.178391$）——「加密网格」对间断数据没有指数红利，"
                "振荡只被压窄、幅度不变。"
                "第三，零点模态会伪造斜率：$N$ 点网格上的方波偶次谐波是常数 $1/N=0.03125$（$N=64$ 实测），"
                "直接拟合得到 $p=-0.848$、$R^2=0.346$；Chebyshev 的 Runge 函数奇次系数恰好为 $0$，"
                "不筛就得到正斜率。第四，尖角剖面的**系数**按 $k^{-2}$ 衰减，但**逼近误差**只按 $N^{-1}$"
                "下降（实测 Fourier $-1.008$、Chebyshev $-0.999$）——系数的指数与误差的指数差一阶，"
                "拿系数斜率当误差收敛率会高估一阶。"
            ),
            "theory": [
                "Poisson 核 $(1-\\rho^2)/(1-2\\rho\\cos\\theta+\\rho^2)$ 的 Fourier 系数恰为 $a_k=\\rho^{|k|}$（几何衰减，非拟合）",
                "$C^{p-1}$ 分段光滑（第 $p$ 阶导数跳变）$\\Rightarrow a_k\\sim k^{-(p+1)}$：尖角 $p=1$ 得 $k^{-2}$，跳变 $p=0$ 得 $k^{-1}$",
                "逼近误差比系数快一阶：$\\|f-S_N f\\|_\\infty\\sim N^{-(p+1)}$（尖角实测 $-1$、光滑按 $N^{-1}\\rho^N$ 指数）",
                "Chebyshev：解析函数误差 $\\sim\\rho^n$ 且 $\\rho$ 由 Bernstein 椭圆参数给出（Runge 为 $3-2\\sqrt2=0.1716$）；$C^0$ 尖角退化为 $n^{-2}$",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "dealias_rule",
            "title": "3/2 规则：混叠去哪了、代价是什么",
            "description": (
                "对一张 $|a_k|=|k|^{-\\mathrm{decay}}$ 的确定性测试场做一次伪谱乘积。"
                "主图是参照乘积谱与当前模式的乘积谱（竖线标出 $2/3$ 保留带边缘 $k=N/3$）；"
                "副图是 $|W_{\\mathrm{naive}}-W_{\\mathrm{ref}}|$ 的折叠谱——混叠污染的模态一眼可见；"
                "右图把三条账随输入谱指数的走向并排画：朴素乘积偏差、$3/2$ 规则偏差（贴舍入）、"
                "$2/3$ 截断的代价。mode=pad_only 用来验证「只补零不截断」是白补。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "n",
                    "label": "网格点数 N",
                    "type": "int",
                    "min": 32,
                    "max": 256,
                    "step": 8,
                    "default": 128,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "decay",
                    "label": "输入谱指数 decay（|a_k|=|k|^-decay）",
                    "type": "float",
                    "min": 0.5,
                    "max": 2.5,
                    "step": 0.05,
                    "default": 1.0,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "mode",
                    "label": "乘积算法",
                    "type": "enum",
                    "options": ["naive", "dealias", "pad_only"],
                    "default": "dealias",
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "exact_pad",
                    "label": "补到精确 3N/2（否则补到 2 的幂）",
                    "type": "bool",
                    "default": False,
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "none"},
            "requiredAux": ["error"],
            "axes": [
                {
                    "id": "spectrum",
                    "autoscaled": False,
                    "title": "乘积谱 |W_k|（双对数，1e-18–10）",
                    "ylabel": "|W_k|",
                    "scale": "log",
                },
                {
                    "id": "alias",
                    "autoscaled": False,
                    "title": "混叠偏差 |dW_k| 折叠（1e-18–10）",
                    "ylabel": "|dW_k|",
                    "scale": "log",
                },
                {
                    "id": "scan",
                    "autoscaled": False,
                    "title": "相对偏差 vs decay（双对数）",
                    "ylabel": "相对偏差",
                    "scale": "log",
                },
            ],
            "pitfalls": (
                "第一，**别把能量恒等式残差当成去混叠的功劳**：对已经 $2/3$ 截断的场，折返落在保留带之外，"
                "朴素乘积与去混叠乘积的残差都是 $10^{-17}$ 量级（实测）；把场放宽到全带宽，"
                "朴素乘积的残差才升到 $10^{-3}\\sim2\\times10^{-1}$，而且强烈依赖相位（同一张 $k^{-1}$ 谱换相位，"
                "二次相位 $1.2\\times10^{-17}$、线性相位 $2.0\\times10^{-1}$）。去混叠真正修好的是**非线性项自身的取值**"
                "（meta.alias_rel / alias_low_rel），不是总能量账。"
                "第二，混叠与截断是两笔账，混在一起读会得出相反的结论：$N=128$、$\\mathrm{decay}=1$ 时"
                "混叠误差 $1.2\\times10^{-2}$（低带 $2.7\\times10^{-3}$），而 $2/3$ 截断自己丢掉的偏差是 $3.7\\times10^{-2}$——"
                "**截断的代价比它顺手消掉的混叠还大 3 倍**；两条算法路线的总差 $4.0\\times10^{-2}$ 里"
                "大部分来自截断而不是混叠。"
                "第三，「$3/2$ 必须配 $2/3$」是错的说法：只补零到精确 $3N/2$、完全不截断时，"
                "相对全带参照的偏差只有 $7.8\\times10^{-5}$（保留带内 $6\\times10^{-7}$），残差**全部集中在 $|k|=N/2$ 这一条边界模态**"
                "（meta.cur_nyquist_rel 与 cur_low_rel 相等就是证据）；"
                "$2/3$ 截断的真正价值是不用补零也能消混叠（省 FFT/内存）。"
                "本内核默认把 $M$ 补到不小于 $3N/2$ 的 2 的幂（$N=128$ 时是 $2N$，实测偏差贴 $8\\times10^{-18}$），"
                "meta.m_used 是实际用的 $M$，别把它当成 $3N/2$。"
                "第四，decay 越大一切都越小，但**两者之比**几乎不变（截断/混叠 $\\approx3$）："
                "$\\mathrm{decay}=0.5$ 时 $1.4\\times10^{-1}$ vs $6.6\\times10^{-2}$，$\\mathrm{decay}=2.5$ 时 $1.4\\times10^{-4}$ vs $2.1\\times10^{-5}$；"
                "所以「去混叠改善了精度」不能只报一个数，必须说清报的是哪一笔。"
            ),
            "theory": [
                "$N$ 点网格上的朴素乘积把乘积模 $p+q$ 中 $|p+q|>N/2$ 的部分折回 $|p+q-N|$：这些落点铺满保留带，所以混叠不只碰高波数",
                "只补零到 $M\\ge3N/2$：乘积模超过 $M/2$ 的部分折回后的最小落点是 $M-N\\ge N/2$，因此保留带 $|k|<N/2$ 内无混叠，残差只剩 $|k|=N/2$ 这一条边界模态（$p=q=N/2$ 的自乘积）",
                "$2/3$ 规则是另一条路：把输入截到 $|k|\\le N/3$ 后，$N$ 点朴素乘积的折返落点 $|k|\\ge N-N/3=2N/3$ 落在保留带之外——不补零也能消混叠，代价是丢掉 $|k|>N/3$ 的输入",
                "$2/3$ 截断 + $3/2$ 补零一起做：对截断后的场逐位精确（实测偏差 $0$，贴 $10^{-16}$ 舍入）",
                "四项账（$N=128$、$\\mathrm{decay}=1$ 实测）：混叠 $1.2\\times10^{-2}$ / 截断代价 $3.7\\times10^{-2}$ / 只补零 $7.8\\times10^{-5}$ / 两条路线总差 $4.0\\times10^{-2}$——混叠与截断必须分开报",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "convergence_race",
            "title": "同一个 PDE：谱方法 vs 二阶/四阶差分",
            "description": (
                "线性平流（精确时间推进，隔离空间误差）用三种空间离散解同一个初值。"
                "主图是相对误差-自由度阶梯（双对数）：谱方法那条越来越陡，差分是直线；"
                "副图是每加倍误差比——差分的比值恒定（二阶 $\\approx0.25$、四阶 $\\approx0.0625$），"
                "谱方法的比值每一步再乘一个更小的数。右图是最大自由度上的逐点误差。"
                "$\\rho$ 控制初值（Poisson 核）的解析性，越小越光滑、指数收敛越快。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "rho",
                    "label": "初值衰减率 rho（Poisson 核）",
                    "type": "float",
                    "min": 0.3,
                    "max": 0.7,
                    "step": 0.01,
                    "default": 0.6,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "n_max",
                    "label": "最大自由度 N_max",
                    "type": "int",
                    "min": 64,
                    "max": 512,
                    "step": 32,
                    "default": 256,
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "scrub", "tMax": 1.0, "dt": 0.1, "label": "t（真实时间）"},
            "requiredAux": ["error"],
            "axes": [
                {
                    "id": "error",
                    "autoscaled": False,
                    "title": "相对误差 vs N（双对数，1e-17–1）",
                    "ylabel": "||u_num-u_exact||_inf/||u_exact||_inf",
                    "scale": "log",
                },
                {
                    "id": "ratio",
                    "autoscaled": False,
                    "title": "每加倍误差比（窗口 0–1）",
                    "ylabel": "e(2N)/e(N)",
                },
                {
                    "id": "pointwise",
                    "autoscaled": False,
                    "title": "逐点误差 @ 最大 N（双对数）",
                    "ylabel": "|u_num-u_exact|",
                    "scale": "log",
                },
            ],
            "pitfalls": (
                "第一，选错时刻会把招牌结论换成假结论：平流速度若取 $1$，$t=0.5$、$1.0$ 处位移正好是"
                "半周期/整周期，三种格式的误差全部退化到舍入级（实测 $c=1$、$t=1.0$ 时 $N=16\\ldots128$ 的"
                "谱误差全部 $\\le10^{-13}$ 且与 $N$ 无关），图上看像「谱方法永远精确、差分也精确」。"
                "本帧取 $c=0.85$ 并把位移写进 meta.shift；$t=0$ 时三种格式也都精确（初值就是解），"
                "所以时间轴两端都是退化帧，招牌数字在 $t\\approx0.3$–$0.7$。"
                "第二，拟合出的斜率不是收敛阶的证据：谱曲线撞地板后整段拟合能给出 "
                "meta.slope_spectral_trap（实测 $t=0.6$ 时 $-1.60$、$R^2=0.09$；$t=1.0$ 时 $-1.33$、$R^2=0.06$——"
                "$R^2$ 低到自曝）。更危险的是只把地板上方的点拿出来拟合：内核层面 "
                "fit_power_law(N, err_spectral, lo=64, hi=256) 在 $t=1.0$ 给出 $-12.05$、$R^2=0.80$，"
                "像一条很陡的幂律——本帧因此拒绝报这个数（meta.slope_spectral 为 null），"
                "只报每加倍比值。"
                "差分比值恒为 $0.246/0.0611$（meta.ratio_fd2_last/ratio_fd4_last），"
                "谱方法的每加倍比值一路缩到 $1.3\\times10^{-4}$——「比值是不是常数」才是阶的判据。"
                "第三，差分曲线在可达的 $N$ 上并不总是 $-2/-4$：窗口取上半段时 $\\rho=0.6$ 实测 "
                "$-1.970/-3.968$（$R^2>0.999$），$\\rho=0.7$ 掉到 $-1.640/-3.705$，"
                "把 $\\rho$ 拖到 $0.8$ 以外（超出本帧范围）实测只剩 $-1.11/-2.92$——"
                "初值不够光滑时，误差被 $k\\,dx\\sim1$ 的最短波主导，阶就测不出来。"
                "窗口是断言的一部分（meta.fit_lo/fit_hi）。"
                "第四，谱误差撞上双精度地板后不再下降（meta.n_at_floor），那不是收敛停滞。"
            ),
            "theory": [
                "谱方法：$u_N$ 是三角函数插值，误差由系数尾巴决定，$C^\\infty$ 数据下 $e(N)\\sim N^{-1}\\rho^N$（指数）",
                "二阶中心差分：修正波数 $\\tilde{\\kappa}=\\sin(k\\,dx)/dx$，$e\\sim(k\\,dx)^2/6$，每加倍降 $1/4$",
                "四阶中心差分：$e\\sim(k\\,dx)^4/30$，每加倍降 $1/16$",
                "同一 PDE、同一初值、同一精确时间推进下比较，横轴 $N$ 对两者都是自由度（谱=模态数，差分=格点数）",
            ],
        },
        # ------------------------------------------------------------------ #
        {
            "id": "burgers_cascade",
            "title": "Burgers 湍流化：能谱、能流与激波",
            "description": (
                "$u_t+u u_x=\\nu u_{xx}$ 的伪谱解（积分因子 RK4，步长只受对流 CFL 限制）。"
                "主图是当前能谱与初值能谱（虚线是 $k^{-2}$ 参考）；副图是能流 $\\Pi(k)$；"
                "右图是解曲线——单模态初值在 $t\\sim1$ 附近长成激波，$k^{-2}$ 段就是这条激波的谱。"
                "dealias 开关直接控制非线性项用不用 $2/3+3/2$；$\\nu$ 越大耗散截断越靠左、"
                "惯性区越短，$\\nu$ 太小则高波数由网格决定（余量都在 meta 里）。"
            ),
            "dataSource": "backend",
            "params": [
                {
                    "name": "n",
                    "label": "网格点数 N",
                    "type": "int",
                    "min": 64,
                    "max": 256,
                    "step": 16,
                    "default": 128,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "nu",
                    "label": "黏性 nu",
                    "type": "float",
                    "min": 0.002,
                    "max": 0.05,
                    "step": 0.001,
                    "default": 0.02,
                    "critical": [0.015],
                    "criticalNote": (
                        "实测临界：$N=128$、$t=1.5$ 时能量收支残差随 $\\nu$ 单调下降，"
                        "$\\nu=0.015$ 处约 $2.5\\times10^{-2}$、$\\nu=0.01$ 处 $1.1\\times10^{-1}$、$\\nu=0.002$ 处 $1.6$（能量反而涨 $68$%）。"
                        "物理上对应 $(\\nu/2)/dx\\approx0.15$：激波厚度开始小于一个网格间距，"
                        "高波数堆积由 $2/3$ 截断与网格决定，$k^{-2}$ 段仍在但数字不再是物理量。"
                        "不是凭空取的阈值：判据 $1/\\sqrt{\\nu t}$ 与 $2/3$ 带边缘 $N/3$ 同量级。"
                    ),
                    "beyond": "below",
                },
                {
                    "name": "dealias",
                    "label": "非线性项去混叠（2/3+3/2）",
                    "type": "bool",
                    "default": True,
                    "critical": [],
                    "criticalNote": "",
                },
                {
                    "name": "initial",
                    "label": "初值",
                    "type": "enum",
                    "options": ["sine", "bump", "random"],
                    "default": "sine",
                    "critical": [],
                    "criticalNote": "",
                },
            ],
            "time": {"mode": "scrub", "tMax": 1.5, "dt": 0.05, "label": "t（真实时间）"},
            "requiredAux": ["error", "residual"],
            "axes": [
                {
                    "id": "spectrum",
                    "autoscaled": False,
                    "title": "能谱 E(k)（双对数，1e-18–1）",
                    "ylabel": "E(k)",
                    "scale": "log",
                },
                {
                    "id": "flux",
                    "autoscaled": False,
                    "title": "能流 Pi(k)（窗口 ±0.3）",
                    "ylabel": "Pi(k)",
                },
                {
                    "id": "solution",
                    "autoscaled": False,
                    "title": "解 u(x,t)（窗口 ±1.6）",
                    "ylabel": "u",
                },
            ],
            "pitfalls": (
                "第一，别把 $k^{-2}$ 段读成 3D 湍流的惯性区：1D Burgers 的 $k^{-2}$ 谱是**激波**的谱，"
                "范围由 $\\nu$ 与 $N$ 共同决定。默认帧（$N=128$、$\\nu=0.02$、$t=1.5$）窗口 $[3,8]$ 的斜率是 $-2.199$"
                "（$R^2=0.99998$，很漂亮），但能流峰值与耗散率之比只有 $0.85$、能流末端 $-1.7\\times10^{-2}$"
                "（本该精确回落到 $0$）——「$\\Pi\\approx\\varepsilon$ 才是惯性区」这条判据在这里读到的是假的不守恒。"
                "第二，$\\nu$ 太小时能量会**涨**：$n=128$、$t=1.5$、$\\nu=0.002$ 的收支残差 $1.6$，"
                "$E$ 从 $0.500$ 涨到 $0.840$，能流末端 $-4.28$——这是网格激波 + 高波数堆积的数值现象，不是物理；"
                "判据是 meta.shock_width_ratio=$(\\nu/2)/dx$ 与 meta.frac_energy_above_2_3，"
                "前者 $<0.1$ 或后者不小就说明高波数由网格决定。"
                "第三，去混叠在分辨良好时对**总能量**几乎没影响（$N=128$、$\\nu=0.02$、$t=1.5$ 实测："
                "两种算法的 $E(T)$ 差 $2\\times10^{-4}$ 相对量、收支残差 $5.9\\times10^{-3}$ vs $2.8\\times10^{-3}$——朴素的那个还更小），"
                "但在小尺度量上现形（同算例 Enstrophy 差 $3.5$%、$\\max|du|/|u|=3.9\\times10^{-2}$）；"
                "欠分辨时才在能量账上现形（$N=128$、$\\nu=0.005$、$t=1.5$：Enstrophy 差 $6.5$ 倍）。"
                "所以「去混叠改善能量守恒」这句话在本章算例里是**有条件**的："
                "分辨良好时改进体现在非线性项与谱细节，不体现在总能量。"
                "第四，时间步在积分因子格式下只受对流 CFL 限制：显式 RK4 的黏性稳定界 "
                "$dt<2.78/(\\nu k_{\\max}^2)$ 会把 $\\nu$ 大、$N$ 大的算例卡死（$N=256$、$\\nu=0.02$ 实测 656 步 vs "
                "本帧 175 步），meta.dt/dt_cfl/n_steps 都在帧里可以自己验。"
            ),
            "theory": [
                "Burgers：$u_t+u u_x=\\nu u_{xx}$，激波形成时间 $t_s=1/(\\pi\\max|u_x|)$，单模态初值在 $t\\sim1$ 长出激波",
                "能谱分解：$E(t)=\\sum_k E_k$，$dE_k/dt=2T_k-2\\nu k^2E_k$，能流 $\\Pi(k)=-2\\sum_{|p|\\le k}T_p$",
                "能流在 $k_{\\max}$ 处回落到 $0$（去混叠的非线性项守恒总能量），$E(t)+2\\nu\\int\\Omega\\,dt=E(0)$",
                "激波能谱：$E(k)\\sim k^{-2}$（间断的谱），耗散截断 $k_d\\sim(\\nu t)^{-1/2}$",
            ],
        },
    ],
}


_DISPATCH = {
    "modal_decay": _viz_modal_decay,
    "dealias_rule": _viz_dealias_rule,
    "convergence_race": _viz_convergence_race,
    "burgers_cascade": _viz_burgers_cascade,
}


def compute(viz_id: str, params: Mapping[str, Any], t: float = 0.0) -> Frame:
    """按 ``viz_id`` 分发到四个可视化之一（帧里的数学全部来自 numcore）。"""
    try:
        fn = _DISPATCH[str(viz_id)]
    except KeyError:
        raise KeyError(f"4.6 章没有可视化 {viz_id!r}（可用：{sorted(_DISPATCH)}）") from None
    return fn(params, float(t))
