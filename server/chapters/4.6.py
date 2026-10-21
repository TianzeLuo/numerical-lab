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
                    "help": (
                        "选择展开用的基函数：fourier 用周期三角函数，chebyshev 用 Chebyshev 多项式。"
                        "Fourier 适合周期剖面，Chebyshev 把节点聚在区间两端，适合非周期函数；"
                        "切换后主图的系数曲线、理论包络和右图误差阶梯会同时换一套数据，"
                        "不能把两条曲线的纵坐标直接相减，应该各自和自己的理论包络比较。"
                    ),
                },
                {
                    "name": "profile",
                    "label": "剖面光滑性",
                    "type": "enum",
                    "options": ["analytic", "corner", "jump", "square"],
                    "default": "analytic",
                    "critical": [],
                    "criticalNote": "",
                    "help": (
                        "选择被展开剖面的光滑程度：analytic 是无限光滑的 Poisson 核或 Runge 函数，"
                        "corner 是 $C^0$ 尖角，jump 是跳变，square 是方波（Chebyshev 侧为 $|x|^3$）。"
                        "越不光滑，系数衰减越慢、误差阶梯越平；jump 那条随 $N$ 几乎不降，"
                        "square 在 Fourier 侧还会出现 Gibbs 过冲。先看主图斜率是否随 $k$ 改变，"
                        "再看右图误差随 $N$ 的走向。"
                    ),
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
                    "help": (
                        "网格点数（自由度）$N$，也就是保留的模态数上限。"
                        "调大 $N$ 会把主图向右延伸、让更多小系数露出来，并把误差阶梯推向更低的误差；"
                        "但跳变剖面的误差最终只压窄振荡、不降低幅度。$N$ 还决定局部斜率可用的窗口："
                        "$N$ 太小时只剩几个模态，拟合和斜率都不可信。"
                    ),
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
                    "help": (
                        "解析剖面的几何衰减率 $\\rho$，取值在 $0$ 到 $1$ 之间，越接近 $1$ 衰减越慢。"
                        "它只对 Fourier 侧的 analytic 剖面生效；Chebyshev 侧的 Runge 函数固定为 "
                        "$3-2\\sqrt{2}\\approx0.1716$，改 $\\rho$ 不会改变那条曲线。"
                        "$\\rho$ 调大时主图变平、需要更大的 $N$ 才能达到同样误差；调小则系数和误差都掉得更快。"
                        "看主图理论包络与实测系数是否重合，以及右图光滑误差阶梯的陡峭程度。"
                    ),
                },
            ],
            "time": {"mode": "none"},
            "requiredAux": ["error"],
            "axes": [
                {
                    "id": "decay",
                    "autoscaled": False,
                    "title": "模态系数 $|a_k|$（双对数，窗口 $10^{-18}$–$1$）",
                    "ylabel": "$|a_k|$",
                    "scale": "log",
                },
                {
                    "id": "slope",
                    "autoscaled": False,
                    "title": "局部斜率（窗口 $-24$–$0$）",
                    "ylabel": "$\\mathrm{d}\\log|a_k|/\\mathrm{d}\\log k$",
                },
                {
                    "id": "error",
                    "autoscaled": False,
                    "title": "误差 vs $N$（双对数，$10^{-17}$–$10$）",
                    "ylabel": "$\\|f-S_Nf\\|_\\infty$",
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
                "**配置点与离散系数**。取周期长度 $2\\pi$ 的 $N$ 个等距节点 $x_j=2\\pi j/N$，"
                "把解写成三角插值 $u_N(x)=\\sum_{|k|\\le N/2}\\hat u_k e^{ikx}$。要求 $u_N(x_j)=u(x_j)$，"
                "两边乘 $e^{-imx_j}$ 并对 $j$ 求和，利用 $\\frac1N\\sum_j e^{i(k-m)x_j}$ 在 $k-m$ 为 $N$ 的整数倍时等于 $1$、"
                "否则等于 $0$，就得到 $\\hat u_k=\\frac1N\\sum_j u(x_j)e^{-ikx_j}$。这就是配置法："
                "系数不是拟合出来的，而是插值条件唯一确定的离散 Fourier 系数；当 $u$ 的谱宽超过 $N/2$ 时，"
                "高频会折进低频，后面讨论的混叠在系数层就已经发生。节点越多，能表示的模态越多；"
                "但系数只由离散采样决定，采样不足时高频信息不会自动消失，只会换一个位置出现。",
                "**Poisson 核给出精确几何**。对 $|\\rho|<1$，Poisson 核 "
                "$P_\\rho(\\theta)=\\frac{1-\\rho^2}{1-2\\rho\\cos\\theta+\\rho^2}$ 可以展开为 "
                "$1+\\sum_{k\\ge1}\\rho^k(e^{ik\\theta}+e^{-ik\\theta})$，所以它的 Fourier 系数恰好是 "
                "$a_k=\\rho^{|k|}$。这不是拟合：$\\rho$ 直接就是相邻模态之比，主图的理论包络因此能逐点对上实测系数。"
                "$\\rho$ 越接近 $1$，系数衰减越慢、同一点的谱越宽；$\\rho$ 越小，曲线掉得越快。"
                "Chebyshev 侧的解析剖面是 Runge 函数，奇点在 $x=\\pm i/5$，对应的 Bernstein 椭圆参数固定为 "
                "$3-2\\sqrt2\\approx0.1716$，所以拖动 $\\rho$ 不会改变那条曲线。",
                "**代数衰减来自分部积分**。若 $u$ 的第 $m$ 阶导数有幅度为 $\\Delta$ 的跳变，"
                "就对 Fourier 系数分部积分 $m$ 次：$\\hat u_k=(ik)^{-m}\\hat u^{(m)}_k$，"
                "而跳变函数的变换按 $1/k$ 衰减，于是 $\\hat u_k\\sim Ck^{-(m+1)}$。"
                "$u$ 本身跳变（$m=0$）给 $k^{-1}$，$C^0$ 尖角的 $u'$ 跳变（$m=1$）给 $k^{-2}$，"
                "四阶导数跳变的 $|x|^3$ 给 $k^{-4}$。主图双对数上的局部斜率 "
                "$\\frac{d\\log|a_k|}{d\\log k}$ 对幂律是常数 $-(m+1)$；对几何衰减则等于 $k\\log\\rho$，"
                "会随 $k$ 一路变负——这正是「斜率在不在变」能区分两类衰减的原因。",
                "**误差比系数快一阶**。截断误差是系数尾巴："
                "$\\|u-u_N\\|_\\infty\\le\\sum_{|k|>N/2}|a_k|$。代入 $|a_k|\\le Ck^{-(m+1)}$ 并用积分判别法，"
                "尾巴按 $N^{-m}$ 下降；系数指数是 $m+1$，误差指数少一阶。尖角剖面因此出现「系数 $k^{-2}$、"
                "误差 $N^{-1}$」的错位，拿系数斜率当误差收敛阶会高估一阶。几何衰减的尾巴是等比级数，"
                "$\\sum_{k>N/2}\\rho^k=\\rho^{N/2+1}/(1-\\rho)$，误差随 $N$ 指数下降；"
                "在双对数图上它没有固定斜率，只会越来越陡。",
                "**每加倍比值才是判据**。定义 $R(N)=e(2N)/e(N)$。若 $e(N)\\sim N^{-m}$，则 "
                "$R(N)\\to2^{-m}$ 是常数；若 $e(N)\\sim C\\rho^{N}$，则 $R(N)\\approx\\rho^{N}$ 随 $N$ 变小且没有正下界。"
                "所以「指数收敛」的可执行判据是每加倍比值是否持续递减，而不是某次拟合出的 $p$ 有多负。"
                "默认帧里光滑剖面的比值从 $0.143$ 降到 $4\\times10^{-4}$，而拟合 $p=-7.129$、$R^2=0.938$ "
                "只说明窗口内看起来陡；同一曲线的局部斜率从 $-0.737$ 走到 $-3.942$，每段都在变，"
                "这才是几何衰减的指纹。若只看某一个 $N$ 上的误差大小，常数因子也会骗人；"
                "必须看误差随 $N$ 变化的整条趋势。",
                "**跳变与 Gibbs 极限**。方波的 $K$ 项部分和在跳变两侧产生过冲，过冲幅度趋于 "
                "$\\frac2\\pi\\mathrm{Si}(\\pi)-1\\approx0.17898$，其中 "
                "$\\mathrm{Si}(\\pi)=\\int_0^\\pi\\frac{\\sin t}{t}dt$；$K$ 增大只把振荡压窄到 $O(1/K)$，"
                "不降低幅度。因此 $L^\\infty$ 误差趋于一个非零常数，图上跳变那条误差曲线基本水平，"
                "只有 $L^2$ 误差会下降。把「加密网格」当成对间断数据也有效的万能药，是本章最常见的误读；"
                "右图的跳变曲线正是用来拆穿它的：水平参考线画在首值附近，提醒读者它不会随 $N$ 归零。",
                "**零点模态与拟合陷阱**。在 $N$ 点网格上采样方波时，跳变若正好落在节点上，"
                "偶次谐波不是 $0$ 而是常数 $1/N$；Chebyshev 侧的 Runge 函数是偶函数，"
                "奇次系数在精确算术下为 $0$，只剩舍入噪声。对 $\\log|a_k|$ 做拟合或差商时，"
                "这些零点会制造正斜率或虚假的陡峭段，所以必须先筛掉低于地板的模态。"
                "主图的拟合窗口固定为 $[4,\\min(48,N/2)]$，$N$ 太小时窗口里只有几个点；"
                "读图时应同时看窗口位置、$R^2$ 和局部斜率，不能只抄一个 $p$。"
                "窗口和筛点规则本身也是结论的一部分，不写出来就无法复现同一条斜率。",
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
                    "help": (
                        "网格点数 $N$，同时决定可保留的波数上限 $N/2$ 和 2/3 带边缘 $N/3$。"
                        "调大 $N$ 会把主图与扫描图的保留带整体右移，混叠和截断的相对偏差通常下降；"
                        "调小则保留带变窄、边界效应更明显。读图时先看竖线 $N/3$ 与 Nyquist 的位置，"
                        "再比较主图两条谱在保留带内是否重合。"
                    ),
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
                    "help": (
                        "输入谱的衰减指数 $\\mathrm{decay}$，满足 $|a_k|=|k|^{-\\mathrm{decay}}$，没有单位。"
                        "$\\mathrm{decay}$ 越大，高频输入越弱，混叠与截断的绝对偏差都变小，但两者之比变化不大；"
                        "$\\mathrm{decay}$ 越小，谱越「白」，混叠会明显铺开。扫描图上紫色竖线标出当前值，"
                        "可对照四条曲线读数；它也会改变主图中当前谱与参照谱的形状。"
                    ),
                },
                {
                    "name": "mode",
                    "label": "乘积算法",
                    "type": "enum",
                    "options": ["naive", "dealias", "pad_only"],
                    "default": "dealias",
                    "critical": [],
                    "criticalNote": "",
                    "help": (
                        "选择非线性乘积的算法：naive 直接逐点相乘，dealias 用 2/3 截断加 3/2 补零，"
                        "pad_only 只补零不截断。naive 的混叠会铺满保留带；dealias 在保留带内最干净，"
                        "但丢掉 $|k|>N/3$ 的输入；pad_only 保留全部输入，残差主要集中在 $|k|=N/2$ 的边界模态。"
                        "切换后主图蓝线与副图偏差谱会换，右图扫描曲线不变，可用来对照三种路线。"
                    ),
                },
                {
                    "name": "exact_pad",
                    "label": "补到精确 3N/2（否则补到 2 的幂）",
                    "type": "bool",
                    "default": False,
                    "critical": [],
                    "criticalNote": "",
                    "help": (
                        "控制补零后的网格长度：关闭时补到不小于 $3N/2$ 的 2 的幂（常见是 $2N$），"
                        "打开时精确补到 $3N/2$。精确补零更省内存，但在 $N$ 为偶数时 Nyquist 边界模态会留下"
                        "约 $10^{-4}$ 量级的残差；补到 $2N$ 时这条边界也被推开，偏差降到舍入级。"
                        "它只影响 dealias 与 pad_only 两条路线，对 naive 没有作用；看右列「只补零」"
                        "和「3/2 残差」两个读数。"
                    ),
                },
            ],
            "time": {"mode": "none"},
            "requiredAux": ["error"],
            "axes": [
                {
                    "id": "spectrum",
                    "autoscaled": False,
                    "title": "乘积谱 $|W_k|$（双对数，$10^{-18}$–$10$）",
                    "ylabel": "$|W_k|$",
                    "scale": "log",
                },
                {
                    "id": "alias",
                    "autoscaled": False,
                    "title": "混叠偏差 $|\\Delta W_k|$ 折叠（$10^{-18}$–$10$）",
                    "ylabel": "$|\\Delta W_k|$",
                    "scale": "log",
                },
                {
                    "id": "scan",
                    "autoscaled": False,
                    "title": "相对偏差 vs $\\mathrm{decay}$（双对数）",
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
                "**逐点乘积就是循环卷积**。设 $u_j=\\sum_p\\hat u_p e^{ipx_j}$、"
                "$v_j=\\sum_q\\hat v_q e^{iqx_j}$，则 $w_j=u_jv_j=\\sum_{p,q}\\hat u_p\\hat v_q e^{i(p+q)x_j}$。"
                "对 $w_j$ 做 DFT 时，正交性只留下 $p+q=k+mN$ 的项，于是 "
                "$\\hat w_k=\\sum_{p+q=k+mN}\\hat u_p\\hat v_q$。这就是循环卷积：真实乘积中波数超过 $N/2$ 的分量"
                "不会消失，而是折到 $p+q-N$ 的位置，和低频模态叠加在一起。",
                "**折返落点铺满保留带**。若输入的带宽是 $K=N/2$，乘积的真实带宽是 $N$。"
                "一个真实波数 $r\\in(N/2,N]$ 折回后落在 $r-N\\in(-N/2,0]$，所以混叠不是只污染最高几个模态，"
                "而是可以落到保留带内的任何位置。图上把 $2N/3$ 的「本尊」画在 Nyquist 右侧的灰带里"
                "（那里没有网格数据），它折回后正好落在 $N/3$ 的「别名」位置；副图的偏差谱因此铺满整个保留带，"
                "而不是只在右端翘起。",
                "**补零到 3N/2 的保证**。把 $N$ 点数据补零到 $M$ 点再相乘，真实带宽仍是 $2K$，"
                "但折返的最小落点变成 $M-2K$。要让保留带 $|k|\\le K$ 干净，需要 $M-2K>K$，即 $M>3K$。"
                "取 $K=N/2$ 得 $M>3N/2$：精确补到 $M=3N/2$ 时，只有 $p=q=N/2$ 的自乘积落到边界 "
                "$|k|=N/2$，低带已经完全干净；补到 $2N$ 时连这条边界也被推开，偏差降到舍入级。"
                "右图「只补零」那条线在 $10^{-5}$ 量级、且残差集中在 Nyquist，就是这条边界模态的证据。",
                "**2/3 截断是另一条路**。先把输入截到 $|k|\\le N/3$，再做 $N$ 点朴素乘积。"
                "此时乘积带宽是 $2N/3>N/2$，折返落点是 $2N/3-N=-N/3$，其绝对值不小于 $N/3$，"
                "正好落在保留带之外。于是不补零也能在 $|k|\\le N/3$ 内消掉混叠，代价是丢掉 "
                "$N/3<|k|\\le N/2$ 的输入。这个代价不是小量：默认 $N=128$、$\\mathrm{decay}=1$ 时"
                "截断造成的相对偏差约 $3.7\\times10^{-2}$，比它顺手消掉的混叠 $1.2\\times10^{-2}$ 还大。",
                "**两条路线可以分开用**。补零路线保留全部输入，靠加大 $M$ 把折返推出保留带；"
                "2/3 路线不补零，靠缩小输入带宽把折返推出保留带。两者没有「必须配对」的硬约束："
                "只补零不截断已经能把低带做干净，只是留下 Nyquist 边界；只截断不补零也能在 $N/3$ 带内干净，"
                "只是丢掉了高频输入。把两者一起做，等价于对截断后的场做精确乘积，偏差贴到 $10^{-16}$ 的舍入级。",
                "**三项账必须分开报**。混叠误差是朴素乘积相对全带参照的偏差；截断代价是 2/3 截断相对全带参照的偏差；"
                "只补零偏差是补零但不截断相对全带参照的偏差；两条路线总差则同时包含混叠和截断。"
                "$\\mathrm{decay}$ 越大，四项都变小，但截断与混叠的比值大致保持在 $3$ 附近，"
                "所以「去混叠改善了精度」这句话必须说明报的是哪一笔。主图看保留带内的两条谱是否重合，"
                "副图看偏差落在哪些模态，扫描图看四条账随 $\\mathrm{decay}$ 的走向。",
                "**能量残差不是判据**。对已经 2/3 截断的场，折返落在保留带之外，"
                "朴素乘积和去混叠乘积的能量恒等式残差都只有 $10^{-17}$ 量级；把场放宽到全带宽，"
                "朴素乘积的残差才升到 $10^{-3}$ 至 $10^{-1}$，而且同一张谱只换相位就能差很多个数量级。"
                "去混叠真正修好的是非线性项在保留带里的取值，不是总能量账。读右列时要把「混叠」「截断」"
                "「只补零」「3/2 残差」四个数分开看，不能把它们合成一个「精度」。",
                "**参数怎样改变图形**。$N$ 同时移动 2/3 带边缘 $N/3$ 与 Nyquist $N/2$，调大后主图向右延伸、"
                "混叠通常变小；$\\mathrm{decay}$ 越小谱越白，混叠越明显，扫描图上紫色竖线标出当前值；"
                "算法开关只改变主图和副图，不改变扫描曲线，可用来对照三条路线；精确补零开关只影响补零长度，"
                "打开后 $M=3N/2$，边界模态的残差会重新出现。看图时先确认竖线位置，再比较保留带内的偏差，"
                "不要把 Nyquist 右侧灰带里的「本尊」当成数据。",
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
                    "help": (
                        "初值 Poisson 核的衰减率 $\\rho$，越接近 $1$ 初值越不光滑、谱越宽。"
                        "调大 $\\rho$ 会让谱误差在同样 $N$ 下更大，也让差分的渐近阶更晚出现；"
                        "调小 $\\rho$ 则初值更光滑，谱方法更快撞到舍入地板。"
                        "它同时影响三条误差曲线，比较时不能只看谱方法一条。"
                    ),
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
                    "help": (
                        "误差阶梯的最大自由度 $N_{\\max}$，决定曲线能画到多远。"
                        "调大能把谱方法推过双精度地板，也能让差分的渐近段更完整；"
                        "调小则可能只看到预渐近区，拟合出的阶不可信。"
                        "看主图折线与副图每加倍比值时，先确认当前 $N_{\\max}$ 是否覆盖了地板转折点。"
                    ),
                },
            ],
            "time": {"mode": "scrub", "tMax": 1.0, "dt": 0.1, "label": "t（真实时间）"},
            "requiredAux": ["error"],
            "axes": [
                {
                    "id": "error",
                    "autoscaled": False,
                    "title": "相对误差 vs $N$（双对数，$10^{-17}$–$1$）",
                    "ylabel": "$\\|u_{\\mathrm{num}}-u_{\\mathrm{exact}}\\|_\\infty/\\|u_{\\mathrm{exact}}\\|_\\infty$",
                    "scale": "log",
                },
                {
                    "id": "ratio",
                    "autoscaled": False,
                    "title": "每加倍误差比（窗口 $0$–$1$）",
                    "ylabel": "$e(2N)/e(N)$",
                },
                {
                    "id": "pointwise",
                    "autoscaled": False,
                    "title": "逐点误差 @ 最大 $N$（双对数）",
                    "ylabel": "$|u_{\\mathrm{num}}-u_{\\mathrm{exact}}|$",
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
                "**模型问题与精确解**。取周期线性平流 $u_t+cu_x=0$，初值 $u_0$ 是 Poisson 核剖面，"
                "Fourier 系数为 $\\hat u_k(0)=\\rho^{|k|}$。精确解只是把每个模态的相位旋转 $e^{-ikct}$："
                "$u(x,t)=u_0(x-ct)$。本页取 $c=0.85$ 而不是 $1$，因为 $c=1$ 时 $t=0.5$ 或 $1.0$ 的位移"
                "正好是半周期或整周期，三种格式的误差都会退化到舍入级，图上会误以为「谱方法永远精确」。"
                "时间推进用精确相位，所以比较的只是空间离散误差；三种格式的差别全部来自空间导数怎么近似。",
                "**谱方法的误差是系数尾巴**。Fourier 配置解为 "
                "$u_N(x,t)=\\sum_{|k|\\le N/2}\\hat u_k(0)e^{ik(x-ct)}$，与精确解只差被截掉的模态："
                "$\\|u-u_N\\|_\\infty\\le\\sum_{|k|>N/2}\\rho^{|k|}=\\frac{\\rho^{N/2+1}}{1-\\rho}$。"
                "因此误差按 $\\rho^{N/2}$ 指数下降，双对数图上是一条向下弯的曲线，没有固定幂律斜率；"
                "$\\rho$ 越接近 $1$，同一条曲线越靠上、越晚撞到舍入地板。几何尾部的收敛速度由最慢的那个"
                "未解析模态决定，所以高波数系数哪怕很小，也会在 $N$ 不够时主导误差。",
                "**二阶中心差分的相位误差**。半离散格式 "
                "$\\frac{du_j}{dt}+c\\frac{u_{j+1}-u_{j-1}}{2\\Delta x}=0$ 对模态 $e^{ikx_j}$ 给出修正波数 "
                "$\\tilde k=\\frac{\\sin(k\\Delta x)}{\\Delta x}=k\\left(1-\\frac{(k\\Delta x)^2}{6}+\\ldots\\right)$。"
                "相速度偏慢，误差累积到时间 $t$ 后约为 $ctk^3\\Delta x^2/6$；对固定光滑初值，"
                "主导误差按 $\\Delta x^2=N^{-2}$ 下降，$N$ 加倍时误差乘 $1/4$。"
                "这就是副图里二阶差分比值稳定在 $0.25$ 附近的原因。",
                "**四阶中心差分的常数**。四阶格式的符号是 "
                "$\\frac{8\\sin\\theta-\\sin2\\theta}{6\\Delta x}$，其中 $\\theta=k\\Delta x$。"
                "把两个正弦展开到五阶，$\\theta$ 的三阶项正好抵消，剩下 "
                "$\\tilde k=k\\left(1-\\frac{(k\\Delta x)^4}{30}+\\ldots\\right)$，所以误差按 $N^{-4}$ 下降，"
                "$N$ 加倍时乘 $1/16$。图上四阶曲线在 $N$ 足够大时与 $N^{-4}$ 参考线平行；"
                "$N$ 太小时高阶项还没占主导，拟合出的阶会偏低，所以拟合窗口只取阶梯的上半段。",
                "**每加倍比值的判据**。定义 $R(N)=e(2N)/e(N)$。代数收敛 $e\\sim N^{-p}$ 给 "
                "$R\\to2^{-p}$，是一个常数；谱方法的 $R\\approx\\rho^{N/2}$ 随 $N$ 减小，没有正下界。"
                "双对数斜率是 $\\frac{d\\log e}{d\\log N}\\approx\\frac{N}{2}\\log\\rho$，本身随 $N$ 变负，"
                "所以谱曲线越来越陡。只比较某个 $N$ 上谁更小，只能说明常数大小；要看副图三条比值曲线："
                "两条差分是水平线，谱方法那条持续下降。若比值在若干步后基本不变，才说明已经进入该收敛"
                "形态的渐近区；否则读到的只是预渐近段。",
                "**退化时刻与舍入地板**。$t=0$ 时初值就是精确解，三种误差全在舍入级，比值没有信息；"
                "$t$ 取整周期时也会出现同样的假象。谱误差降到 $10^{-14}$ 附近后不再随 $N$ 下降，"
                "这不是收敛停滞，而是双精度算不动了；此时每加倍比值变成 $1$ 附近或乱跳，不能拿来判断阶。"
                "代码只在误差高于 $10^{-13}$ 且拟合点不少于三个、跨度足够时报告谱斜率，否则明确写「不可用」，"
                "避免把地板段的假陡峭拟成 $p=-12$。所以看到曲线突然变陡时，先检查纵轴是否已经接近 $10^{-14}$。",
                "**参数与读图**。$\\rho$ 越大初值越不光滑，谱误差整体抬高，差分也更容易受高波数影响，"
                "实测阶可能从 $-2$、$-4$ 偏离；$N_{\\max}$ 决定阶梯能走多远，调大后谱方法会撞到地板、"
                "差分进入更干净的渐近段。主图看三条曲线的形状与参考线，副图看比值是否恒定，"
                "右图看最大 $N$ 下误差在空间上的分布：谱误差是全局的小振荡，差分误差是沿途的色散波纹。"
                "不要把「曲线陡」当成阶，也不要把「谱误差更小」当成指数收敛的证据。同一张图在不同 "
                "$\\rho$ 下不能直接横比，因为初值本身已经换了。",
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
                    "help": (
                        "网格点数 $N$，决定能表示的最高波数与 2/3 带边缘 $N/3$。"
                        "调大 $N$ 会把能谱和能流曲线向右延伸，让激波谱的 $k^{-2}$ 段更长、耗散截断更靠右；"
                        "调小则高波数很快被截断，$k^{-2}$ 段变短。读图时把 $N$、$\\nu$ 一起看："
                        "只有激波宽度远大于网格间距时，高波数才是物理的。"
                    ),
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
                    "help": (
                        "黏性系数 $\\nu$，越大耗散越强。调大 $\\nu$ 会让激波更厚、耗散波数 "
                        "$k_d\\approx(\\nu t)^{-1/2}$ 更靠左，能谱的 $k^{-2}$ 段缩短；调小 $\\nu$ 会让激波变薄，"
                        "若薄过一个网格间距，高波数由网格和 2/3 截断决定，能量收支会明显变差。"
                        "默认临界值 $0.015$ 附近开始欠分辨，看能流末端是否回落到 $0$、能量漂移是否变大。"
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
                    "help": (
                        "是否对非线性项 $uu_x$ 使用 2/3 截断加 3/2 补零。打开后非线性项在保留带内没有混叠，"
                        "能流在高波数端更接近 $0$；关闭后混叠会污染小尺度量，Enstrophy 和逐点解先现形，"
                        "分辨良好时总能量差别反而很小。把它与 $\\nu$、$N$ 联动看："
                        "欠分辨时开关对能量账的影响才会放大。"
                    ),
                },
                {
                    "name": "initial",
                    "label": "初值",
                    "type": "enum",
                    "options": ["sine", "bump", "random"],
                    "default": "sine",
                    "critical": [],
                    "criticalNote": "",
                    "help": (
                        "初值类型：sine 是单模态正弦，bump 是局部高斯包，random 是若干随机相位模态的叠加。"
                        "sine 最干净、激波在 $t\\approx1$ 形成；bump 和 random 一开始就有较宽的谱，"
                        "激波更早、$k^{-2}$ 段更难辨认。切换后解曲线和能谱初值都会变，"
                        "比较不同时刻时要先确认激波是否已经形成。"
                    ),
                },
            ],
            "time": {"mode": "scrub", "tMax": 1.5, "dt": 0.05, "label": "t（真实时间）"},
            "requiredAux": ["error", "residual"],
            "axes": [
                {
                    "id": "spectrum",
                    "autoscaled": False,
                    "title": "能谱 $E(k)$（双对数，$10^{-18}$–$1$）",
                    "ylabel": "$E(k)$",
                    "scale": "log",
                },
                {
                    "id": "flux",
                    "autoscaled": False,
                    "title": "能流 $\\Pi(k)$（窗口 $\\pm0.3$）",
                    "ylabel": "$\\Pi(k)$",
                },
                {
                    "id": "solution",
                    "autoscaled": False,
                    "title": "解 $u(x,t)$（窗口 $\\pm1.6$）",
                    "ylabel": "$u$",
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
                "**方程与谱形式**。周期 Burgers 方程 $u_t+uu_x=\\nu u_{xx}$ 中，非线性项可以写成 "
                "$\\frac12(u^2)_x$。把 $u=\\sum_k\\hat u_k e^{ikx}$ 代入，得到系数空间的常微分方程组 "
                "$\\frac{d\\hat u_k}{dt}=-\\frac{ik}{2}\\hat w_k-\\nu k^2\\hat u_k$，其中 $w=u^2$。"
                "伪谱法在每一步用 FFT 把 $\\hat u_k$ 变到网格上，逐点平方，再变回系数；麻烦全在「逐点平方」"
                "这一步，因为 $N$ 点网格上的乘积是循环卷积，高频会折回低频。这也说明伪谱法为什么既快，"
                "又必须额外处理混叠。",
                "**能量收支的推导**。取能量 $E=\\sum_k|\\hat u_k|^2=\\frac1{2\\pi}\\int u^2dx$、"
                "Enstrophy $\\Omega=\\sum_k k^2|\\hat u_k|^2=\\frac1{2\\pi}\\int u_x^2dx$。用 $u$ 乘方程并积分，"
                "非线性项 $\\int u^2u_x/2\\,dx=\\int (u^3)_x/6\\,dx=0$（周期性），只剩 "
                "$\\frac{dE}{dt}=-2\\nu\\Omega$。从 $0$ 到 $t$ 积分得 "
                "$E(t)+2\\nu\\int_0^t\\Omega(s)ds=E(0)$。图右下的收支残差就是这条恒等式的相对误差；"
                "它同时受时间离散、2/3 截断和欠分辨的高波数堆积影响，不是去混叠的单独证据。",
                "**逐带能量与能流**。对每个模态定义 $E_k=|\\hat u_k|^2$，非线性项把它写成 "
                "$\\frac{dE_k}{dt}=2T_k-2\\nu k^2E_k$，其中 $T_k=\\mathrm{Re}(\\bar{\\hat u}_k\\hat w_k)$ "
                "来自 $-\\frac{ik}{2}\\hat w_k$ 与 $\\bar{\\hat u}_k$ 的乘积。对 $|k|\\le K$ 求和，得到 "
                "$\\Pi(K)=-2\\sum_{|k|\\le K}T_k$，于是 "
                "$\\frac{dE_{\\le K}}{dt}=-\\Pi(K)-2\\nu\\sum_{|k|\\le K}k^2E_k$。"
                "$\\Pi(K)>0$ 表示能量从低波数流向高波数；在理想惯性区它应近似等于耗散率 "
                "$\\varepsilon=2\\nu\\Omega$，在 $k_{\\max}$ 处回落到 $0$。",
                "**激波形成时间**。沿特征线 $u$ 不变，$x=\\xi+tu_0(\\xi)$，所以 "
                "$u_x=u_0'(\\xi)/(1+tu_0'(\\xi))$。当 $1+t\\min u_0'=0$ 时导数发散，形成激波；"
                "对 $u_0=\\sin x$，$\\min u_0'=-1$，所以 $t_s=1$。默认时间轴到 $t=1.5$，"
                "已经越过激波形成时刻，解曲线出现陡峭的跳跃层；在这之前能谱仍接近初值的单模态，"
                "之后才长出宽频段。初值换成 bump 或 random 时，初始谱更宽、激波更早，"
                "$k^{-2}$ 段也更难辨认。激波形成前解仍光滑，谱方法的指数收敛还成立；形成后不光滑性"
                "开始主导，误差和能谱都换了一种形态。",
                "**k^{-2} 段从哪来**。激波附近的定常剖面近似 "
                "$u=-U\\tanh(U(x-x_s)/(2\\nu))$，它的 Fourier 幅度按 "
                "$1/\\cosh(\\pi\\nu k/(2U))$ 衰减：当 $k$ 远小于 $U/(\\pi\\nu)$ 时幅度按 $1/k$ 衰减，"
                "能量谱 $E_k\\sim k^{-2}$；当 $k$ 超过耗散波数 $k_d\\approx(\\nu t)^{-1/2}$ 后指数衰减。"
                "有限 $N$ 还会在 $N/3$ 处切断，所以可用的 $k^{-2}$ 段只有 "
                "$3\\le k\\le\\min(1.5k_d,N/3)$。默认帧拟合斜率约 $-2.2$，不是精确的 $-2$，"
                "因为窗口、有限黏性和离散都在改变局部斜率。",
                "**能流平台不是三维惯性区**。三维湍流的惯性区判据是 $\\Pi(k)\\approx\\varepsilon$ 且平台宽；"
                "一维 Burgers 的 $k^{-2}$ 段只是激波谱，能量在激波处直接耗散，不存在真正的正向级联惯性区。"
                "默认帧 $\\Pi_{\\max}/\\varepsilon$ 通常小于 $1$，能流先升后降，末端应回到 $0$；"
                "如果末端明显为负，说明高波数堆积或欠分辨，而不是「级联更强」。把 $k^{-2}$ 读成 "
                "$k^{-5/3}$ 的惯性区，或者把 $\\Pi$ 的峰值当成耗散率，都是常见误读。",
                "**分辨率的数值极限**。激波厚度约 $\\nu/U$，要与网格间距 $\\Delta x=2\\pi/N$ 比较；"
                "代码用 $(\\nu/2)/\\Delta x$ 作为宽度比，小于 $0.1$ 就说明激波薄于一个网格，"
                "$k^{-2}$ 段的高端由网格和 2/3 截断决定。$\\nu$ 太小或 $N$ 太小都会让能量收支残差变大，"
                "甚至出现能量反向增长；默认临界 $\\nu=0.015$ 附近开始欠分辨。$\\nu$ 调大则激波变厚、"
                "$k_d$ 左移、$k^{-2}$ 段变短；$N$ 调大则带边缘右移、可分辨的 $k^{-2}$ 段变长。",
                "**去混叠与时间步**。非线性项用 2/3+3/2 去混叠时，保留带内没有混叠，能流末端更接近 $0$；"
                "关闭后混叠先污染 Enstrophy 和逐点解，分辨良好时总能量差别反而很小，"
                "欠分辨时才会在能量账上放大。时间推进用积分因子 RK4，扩散项被指数因子精确吸收，"
                "步长只受对流 CFL 限制 $\\Delta t\\le0.35\\Delta x/\\max|u|$；否则显式黏性稳定界 "
                "$\\Delta t<2.78/(\\nu k_{\\max}^2)$ 会把大 $N$、大 $\\nu$ 的算例卡死。"
                "看图时把 $N$、$\\nu$、去混叠开关和 $t$ 一起读，任何单个数字都不能单独证明「级联正确」。"
                "这也是为什么同一组参数下，开关对总能量和对小尺度量的影响可以完全不同。",
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
