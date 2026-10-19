"""5.4 章的**帧级**测试：把"图上看到的东西"与 ``numcore`` 的实测值钉在一起。

帧是唯一会被前端直接画出来的东西，所以这里检查的是**呈现纪律**：

1. **没有非有限值**。四个可视化在默认参数、每个参数的两端、以及 ``t=0``/``t=tMax``
   的全组合下都必须能 ``pack_frame``（JSON 头里不许出现 ``NaN``/``Infinity``）。
2. **占位与真值分开**。"没有误差棒"（确定性点集）、"斜率测不出来"（误差曲线整条贴地板）、
   "相关分解不适用"（均匀输入方体上拖 rho）三件事都必须用掩码 + 文字说清楚，
   而不是画一条看起来正常的曲线。
3. **诚实字段真的会亮**：``band_available``、``sum_between``、``underdetermined``、
   ``ref_is_exact`` 在各自该亮的时候亮，且 ``strings`` 里带着**实测数字**。
4. **装配层只许搬运**：帧里的数组必须逐位等于直接调用 ``numcore`` 的结果。
"""

from __future__ import annotations

import itertools
import json
import math
import struct
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from numcore.inverse.uq import mc as mcmod  # noqa: E402
from numcore.inverse.uq import pce, qmc, sensitivity, sobol_seq  # noqa: E402
from numcore.verify import assertion, record  # noqa: E402
from server.frames import build_frame  # noqa: E402
from server.nlf import pack_frame  # noqa: E402
from server.registry import Registry  # noqa: E402

_ENTRY_CACHE: list = []
VIZ_IDS = ("point_fill", "output_hist", "sobol_bars", "pce_spectrum")


def _entry():
    if not _ENTRY_CACHE:
        reg = Registry(ROOT)
        reg.discover(strict=False)  # 并发开发时兄弟章节可能是半成品；本章必须能加载
        _ENTRY_CACHE.append(reg.get("5.4"))
    return _ENTRY_CACHE[0]


def _viz(viz_id: str):
    return next(v for v in _entry().manifest.vizzes if v.id == viz_id)


def _defaults(viz_id: str) -> dict:
    return {p.name: p.default for p in _viz(viz_id).params}


def _frame(viz_id: str, **over):
    t = float(over.pop("t", 0.0))
    params = _defaults(viz_id)
    params.update(over)
    return build_frame(_entry(), viz_id, params, t)


def _corners(viz_id: str):
    spec = _viz(viz_id)
    base = {p.name: p.default for p in spec.params}
    out = [dict(base)]
    for p in spec.params:
        if p.type == "enum":
            out.extend({**base, p.name: o} for o in p.options)
        elif p.type in ("int", "float"):
            out.append({**base, p.name: p.min})
            out.append({**base, p.name: p.max})
        elif p.type == "bool":
            out.append({**base, p.name: not base[p.name]})
    return out


# --------------------------------------------------------------------------- #
# 1. 全组合下的有限性与 JSON 可序列化
# --------------------------------------------------------------------------- #
@assertion("A", theory="帧在参数端点与时间端点全组合下都只有有限值，且能装进 JSON", chapter="5.4")
def test_all_frames_finite_and_packable_at_corners() -> None:
    r"""参数端点 x 时间端点的全组合：数组全有限、``meta`` 无 NaN、JSON 里没有 ``Infinity``。

    ``point_fill`` 的时间轴是"已铺开的点数"，最容易炸的是 ``t=tMax`` 那一帧（点最多、
    误差曲线最长）；``output_hist`` 的 ``n=1e5`` 与 ``n_bins=80`` 组合是另一头。
    """
    def _boom(token: str) -> None:
        raise AssertionError(f"JSON 里出现非有限常量 {token!r}")

    n_cases = 0
    bad_tokens = 0
    detail: list[str] = []
    for viz_id in VIZ_IDS:
        spec = _viz(viz_id)
        times = [0.0] if spec.time.mode == "none" else [0.0, float(spec.time.t_max)]
        for params, t in itertools.product(_corners(viz_id), times):
            fr = build_frame(_entry(), viz_id, params, t)
            for name in fr.names:
                arr = np.asarray(fr[name], dtype=float)
                assert np.all(np.isfinite(arr)), f"{viz_id}/{name} 在 {params} t={t} 处有非有限值"
            for key, val in fr.meta.items():
                if isinstance(val, float):
                    assert math.isfinite(val), f"{viz_id} meta.{key} 非有限：{val}"
            blob = pack_frame(fr)
            try:
                # 只解析头部 JSON，并且只认真正的非有限**常量**
                # （文档字符串里写「没有 NaN」这三个字母不算违规）
                (hlen,) = struct.unpack_from("<I", blob, 4)
                json.loads(blob[8 : 8 + hlen].decode("utf-8"), parse_constant=_boom)
            except AssertionError:
                bad_tokens += 1
            n_cases += 1
            if not detail:
                detail.append(f"{viz_id}: {len(fr.names)} 个数组、{len(fr.meta)} 个 meta 字段")
    rec = record(
        label="端点组合帧数与 JSON 里真正的非有限常量个数",
        theory="非有限常量个数 = 0（字符串里出现 NaN 三个字母不算），帧数 >= 60",
        measured=f"{n_cases} 帧，计数 {bad_tokens}（" + "；".join(detail) + "）",
        tol="= 0",
        passed=bool(bad_tokens == 0 and n_cases >= 60),
    )
    assert rec.passed, rec.measured


@assertion("A", theory="对数轴曲线一律 >= log10(1e-17) 且留有真实数据（max 明显高于地板）；贴地板的点由掩码标 0", chapter="5.4")
def test_log_axis_curves_are_floored_with_masks() -> None:
    r"""对数轴上画不出 0：帧里存的**就是** :math:`\log_{10}` 值，且恒有下界
    :math:`\log_{10}10^{-17}=-17`（``strings.error_floor`` 写明）。检查两件事：

    1. 每条对数曲线的 ``min`` 都不小于地板、``max`` 高于地板至少一个数量级
       （否则说明整条曲线都是假的，比如"所有误差都恰好是 0"）；
    2. 真的贴到地板的点必须有掩码 0——Sobol' 网精确积分对齐的间断函数就是这种情形
       （误差恒为 0，``error_valid`` 全 0），而偏移边界的对照帧掩码全 1。
    """
    floor = -17.0
    detail = []
    ok = True
    for viz_id, names in (
        ("point_fill", ("error", "error_mc", "error_sobol", "error_ref", "band_lo", "band_hi")),
        ("output_hist", ("error", "bins_error")),
        ("pce_spectrum", ("error", "error_train", "coeff_rms")),
    ):
        fr = _frame(viz_id)
        for name in names:
            arr = np.asarray(fr[name], dtype=float)
            ok = ok and bool(np.all(arr >= floor - 1e-9)) and float(np.max(arr)) > floor + 1.0
            detail.append(f"{viz_id}.{name}∈[{arr.min():.1f},{arr.max():.1f}]")
    fr_pf = _frame("point_fill")
    ok = ok and ("1e-17" in fr_pf.strings["error_floor"])
    # 贴地板的情形：Sobol' 对 2^-1 对齐的间断函数精确积分，误差恒为 0
    fr0 = _frame("point_fill", integrand="discontinuous", method="sobol", randomize=False)
    fr1 = _frame("point_fill", integrand="offset_box", method="sobol", randomize=False)
    v0 = np.asarray(fr0["error_valid"], dtype=float)
    v1 = np.asarray(fr1["error_valid"], dtype=float)
    rec = record(
        label="对数曲线范围、地板说明、以及贴地板处掩码确实为 0",
        theory="全部曲线 >= -17 且 max > -16；discontinuous 掩码全 0、offset_box 掩码全 1",
        measured="；".join(detail) + f"；discontinuous 掩码 {np.unique(v0)}，offset_box 掩码 {np.unique(v1)}",
        tol="全 True",
        passed=bool(ok and np.all(v0 == 0.0) and np.all(v1 == 1.0)),
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 2. 没有随机化就没有误差棒
# --------------------------------------------------------------------------- #
@assertion("A", theory="randomize=False（或 method=grid）时 band_valid 全 0、band_available=False，strings 说明原因", chapter="5.4")
def test_point_fill_declares_missing_error_band() -> None:
    r"""三种情形必须**分别**说清楚，而不是共用一个"无数据"：

    * ``randomize=False`` + Sobol：确定性点集的复制标准差恒为 0 → 没有误差棒；
    * ``method=grid``：张量格点**没有**随机化版本（``point_set`` 会拒绝），
      即使 ``randomize=True`` 也不该画出带宽；
    * ``randomize=True`` + Sobol：误差棒存在（``band_valid`` 全 1）。
    """
    fr_off = _frame("point_fill", randomize=False, method="sobol")
    fr_grid = _frame("point_fill", randomize=True, method="grid")
    fr_on = _frame("point_fill", randomize=True, method="sobol")
    band_off = np.asarray(fr_off["band_valid"], dtype=float)
    band_grid = np.asarray(fr_grid["band_valid"], dtype=float)
    band_on = np.asarray(fr_on["band_valid"], dtype=float)
    rec = record(
        label="三种情形下的 band_valid / band_available / 文字说明",
        theory="关随机化与 grid：全 0 且 band_available=False；开随机化：全 1 且 True",
        measured=f"关：valid={np.unique(band_off)}，available={fr_off.meta['band_available']}；"
        f"grid：valid={np.unique(band_grid)}，available={fr_grid.meta['band_available']}；"
        f"开：valid={np.unique(band_on)}，available={fr_on.meta['band_available']}；"
        f"说明含「不存在」={'不存在' in fr_off.strings['band_note']}",
        tol="全 True",
        passed=bool(
            np.all(band_off == 0.0)
            and np.all(band_grid == 0.0)
            and np.all(band_on == 1.0)
            and fr_off.meta["band_available"] is False
            and fr_grid.meta["band_available"] is False
            and fr_on.meta["band_available"] is True
            and "不存在" in fr_off.strings["band_note"]
        ),
    )
    assert rec.passed, rec.measured


@assertion("A", theory="误差曲线整条贴地板时 slope_current 为 None（不允许在地板上拟合斜率）", chapter="5.4")
def test_point_fill_refuses_to_report_slope_on_the_floor() -> None:
    r"""Sobol' 网对 :math:`1[x_1<0.5\\wedge x_2<0.5]` 是**精确**积分（误差恒为 0，见 C 类断言），
    于是误差曲线整条贴在地板上。此时：

    * ``slope_current`` 必须是 ``None``（不是 -0.5 之类的编出来的数）；
    * ``strings.note`` 里要直接说明"误差精确为 0"；
    * ``error_valid`` 全 0（掩码如实标注）。
    对照：同一个点集在 0.3 边界的间断函数上有正常斜率（实测 -0.85 量级）。
    """
    fr = _frame("point_fill", integrand="discontinuous", method="sobol", randomize=False)
    note = fr.strings["note"]
    fr2 = _frame("point_fill", integrand="offset_box", method="sobol", randomize=True)
    rec = record(
        label="贴地板时的 slope_current 与说明文字",
        theory="slope_current = None；note 里出现「精确为 0」；偏移边界上斜率 ∈ [-1.2,-0.6]",
        measured=f"slope_current={fr.meta['slope_current']!r}，note 片段={note[-40:]!r}，"
        f"offset_box 的 slope_current={fr2.meta['slope_current']!r}",
        tol="见 theory",
        passed=bool(
            fr.meta["slope_current"] is None
            and "精确为 0" in note
            and fr2.meta["slope_current"] is not None
            and -1.2 <= float(fr2.meta["slope_current"]) <= -0.6
        ),
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 3. 相关输入：三种口径同时呈现
# --------------------------------------------------------------------------- #
@assertion("A", theory="rho=0 时相关分解字段为 None + corr_valid 全 0；rho=0.5 时 sum_between=1.49 > 1 且文字带数字", chapter="5.4")
def test_sobol_bars_correlation_states() -> None:
    r"""``rho`` 是本页的临界参数（``critical=[0.0]``）：

    * ``rho=0``：只报独立分解，``overlap_excess``/``sum_between``/``rho_measured`` 必须是
      ``None``（不是 0——0 会被读成"实测到没有重叠"），``corr_valid`` 全 0；
    * ``rho=0.5``（双线性模型）：``sum_between`` 实测 :math:`1.49>1`，
      ``sum_indep`` 实测 :math:`0.99`，两者差 0.5 就是"被重复计算的方差"；
    * 均匀方体上的模型（g-函数）拖 rho 不改变结论，但 ``corr_applicable=False``
      且文字里要说明"rho 对它无效"。
    """
    fr0 = _frame("sobol_bars", model="bilinear", rho=0.0, n=1024)
    fr5 = _frame("sobol_bars", model="bilinear", rho=0.5, n=1024)
    frg = _frame("sobol_bars", model="g_function", rho=0.8, n=1024)
    rec = record(
        label="rho=0 与 rho=0.5 的字段对照（双线性模型）",
        theory="rho=0：三个字段为 None、corr_valid 全 0；rho=0.5：sum_between > 1.4 且 sum_indep ∈ [0.95,1.05]",
        measured=f"rho=0：corr_valid={np.unique(np.asarray(fr0['corr_valid']))}，"
        f"overlap={fr0.meta['overlap_excess']!r}，sum_between={fr0.meta['sum_between']!r}；"
        f"rho=0.5：corr_valid={np.unique(np.asarray(fr5['corr_valid']))}，"
        f"sum_between={fr5.meta['sum_between']:.4f}，sum_indep={fr5.meta['sum_indep']:.4f}，"
        f"rho_measured={fr5.meta['rho_measured']:.4f}，corr_note 带数字={('1.4' in fr5.strings['corr_note'])}",
        tol="见 theory",
        passed=bool(
            np.all(np.asarray(fr0["corr_valid"]) == 0.0)
            and fr0.meta["overlap_excess"] is None
            and fr0.meta["sum_between"] is None
            and np.all(np.asarray(fr5["corr_valid"]) == 1.0)
            and fr5.meta["sum_between"] > 1.4
            and 0.95 <= fr5.meta["sum_indep"] <= 1.05
            and abs(fr5.meta["rho_measured"] - 0.5) < 0.01
        ),
    )
    rec2 = record(
        label="均匀输入方体上的模型拖 rho",
        theory="corr_applicable=False、corr_active=False，且文字说明 rho 无效",
        measured=f"corr_applicable={frg.meta['corr_applicable']}，corr_active={frg.meta['corr_active']}，"
        f"说明含「无效」={'无效' in frg.strings['corr_note']}",
        tol="全 True",
        passed=bool(
            frg.meta["corr_applicable"] is False
            and frg.meta["corr_active"] is False
            and "无效" in frg.strings["corr_note"]
        ),
    )
    assert rec.passed and rec2.passed, f"{rec.measured} / {rec2.measured}"


@assertion("A", theory="SOBOL 指数之和是判据：独立时 sum S_i <= 1 <= sum S_Ti，相关时条件方差口径可以 > 1", chapter="5.4")
def test_sobol_bars_sum_criterion_and_truth_flags() -> None:
    r"""``lower_bound_ok``/``upper_bound_ok`` 两个布尔必须由**实测值 + 区间**算出来，
    而不是恒为 True：双线性 + rho=0.5 时 :math:`\\sum S_i \\le 1` 这条在条件方差口径下
    本来就不成立，帧里必须让读者看到它不成立（数字就在 meta 里）。"""
    fr_ind = _frame("sobol_bars", model="g_function", rho=0.0, n=4096)
    fr_corr = _frame("sobol_bars", model="bilinear", rho=0.5, n=1024)
    rec = record(
        label="独立/相关两种情形下的和与判据布尔",
        theory="独立：sum S_i <= 1 <= sum S_Ti 且布尔为真；相关：sum_between > 1（判据在该口径下失效）",
        measured=f"独立 g-函数：sum S_i={fr_ind.meta['sum_first']:.4f}，sum S_Ti={fr_ind.meta['sum_total']:.4f}，"
        f"lower={fr_ind.meta['lower_bound_ok']}，upper={fr_ind.meta['upper_bound_ok']}，"
        f"解析={fr_ind.meta['analytic_sum_first']:.4f}；"
        f"相关双线性：sum_between={fr_corr.meta['sum_between']:.4f}（>1）",
        tol="见 theory",
        passed=bool(
            fr_ind.meta["sum_first"] <= 1.0
            and fr_ind.meta["sum_total"] >= 1.0
            and fr_ind.meta["analytic_sum_first"] is not None
            and fr_corr.meta["sum_between"] > 1.0
        ),
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 4. PCE：过拟合、外推、精确阶
# --------------------------------------------------------------------------- #
@assertion("A", theory="n_train=8 且 p>n_basis 时 underdetermined=True、训练误差到 0 而留出误差 > 0.1、under_flag 有 1", chapter="5.4")
def test_pce_spectrum_marks_overfitting_region() -> None:
    r"""过拟合区必须在帧里**看得见**：

    * ``under_flag`` 在基规模超过样本数的那些阶上为 1；
    * ``meta.underdetermined`` 对当前阶为 True；
    * ``error_train_current`` <= 1e-14 而 ``error_outsample_current`` > 0.1——
      两个数放在同一帧里，才能读出"训练误差什么也不能说明"。
    """
    fr = _frame("pce_spectrum", target="exp", fit="regression", order=10, n_train=8)
    under = np.asarray(fr["under_flag"], dtype=float)
    nb = np.asarray(fr["n_basis_curve"], dtype=float)
    oc = np.asarray(fr["order_curve"], dtype=float)
    rec = record(
        label="过拟合区的三个字段 + 训练/留出误差",
        theory="underdetermined=True；under_flag = (n_basis > 8) 逐点成立；训练 <= 1e-14 且留出 > 0.1",
        measured=f"underdetermined={fr.meta['underdetermined']}，欠定阶数 {int(np.sum(under))}，"
        f"训练 {fr.meta['error_train_current']:.2e}，留出 {fr.meta['error_outsample_current']:.3e}，"
        f"外推 {fr.meta['error_extrap_current']:.3e}（放大 {fr.meta['extrap_ratio']:.3g} 倍）",
        tol="见 theory",
        passed=bool(
            fr.meta["underdetermined"] is True
            and np.array_equal(under == 1.0, nb > 8)
            and fr.meta["error_train_current"] <= 1e-14
            and fr.meta["error_outsample_current"] > 0.1
            and oc.size == 12
        ),
    )
    assert rec.passed, rec.measured


@assertion("A", theory="poly4 在 p>=4 后误差到 1e-14 且 exact_at_order=4；p<4 时误差 > 1e-1", chapter="5.4")
def test_pce_spectrum_reports_exact_order_and_floor() -> None:
    r"""``exact_at_order`` 给出第一次达到 :math:`10^{-12}` 的阶：实测 4，与"四次多项式用四阶
    基就精确"完全一致；``p<4`` 时误差 :math:`10^{-0.36}` 到 :math:`10^{-1.05}`（明显不为零，
    否则这条断言什么也没测）。这一帧**没有**触地板（1.65e-14 > 1e-17），
    所以 ``error_valid`` 全 1——掩码只在真的贴地板时才为 0（见点集那一页）。"""
    fr = _frame("pce_spectrum", target="poly4", fit="projection", order=12, n_train=64)
    valid = np.asarray(fr["error_valid"], dtype=float)
    rec = record(
        label="poly4 的 exact_at_order / 掩码 / 当前阶误差",
        theory="exact_at_order = 4；p<4 时 log10(err) >= -1.1；当前阶误差 <= 1e-12；掩码全 1",
        measured=f"exact_at_order={fr.meta['exact_at_order']}，log10 误差曲线 {np.array2string(np.asarray(fr['error']), precision=2)}，"
        f"掩码取值 {np.unique(valid)}，当前留出误差 {fr.meta['error_outsample_current']:.3e}",
        tol="见 theory",
        passed=bool(
            fr.meta["exact_at_order"] == 4
            and float(np.max(np.asarray(fr["error"], dtype=float)[:3])) >= -1.1
            and fr.meta["error_outsample_current"] <= 1e-12
            and np.all(valid == 1.0)
        ),
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 5. 直方图：参考线的身份
# --------------------------------------------------------------------------- #
@assertion("A", theory="正态+线性时 ref_is_exact=True 且 KS(参考) ≈ KS(正态)；均匀+线性时两者相差 5 倍以上", chapter="5.4")
def test_output_hist_declares_reference_identity() -> None:
    r"""``ref_is_exact`` 与 ``strings.ref_kind`` 决定读者怎么解释 |密度差| 那条曲线：

    * 正态输入 + 线性模型：精确卷积**就是**正态密度 → 两个 KS 必须几乎相等
      （本模块第一版在 40 个 bin 中心上累加 CDF，给出 0.0506 vs 0.0054，
      把"参考本身就是真值"的两种度量算出 10 倍差——那是装配层的错，不是统计现象）；
    * 均匀输入 + 线性模型：输出是三角分布，精确参考明显优于正态近似。
    """
    fr_n = _frame("output_hist", dist="normal", model="linear", n=20000)
    fr_u = _frame("output_hist", dist="uniform", model="linear", n=20000)
    fr_e = _frame("output_hist", dist="exponential", model="linear", n=20000)
    rec = record(
        label="三种输入分布下 ref_is_exact / KS 对照",
        theory="正态+线性：exact=True 且两个 KS 相差 <= 2e-3（精确参考就是正态）；"
        "均匀+线性：精确三角密度，ks_to_normal >= 3 * ks_to_ref；"
        "指数+线性：没有精确参考，exact=False 且 ref_kind 里写明是 CLT 近似",
        measured=f"正态：exact={fr_n.meta['ref_is_exact']}，ks_ref={fr_n.meta['ks_to_ref']:.4f}，"
        f"ks_normal={fr_n.meta['ks_to_normal']:.4f}；"
        f"均匀：exact={fr_u.meta['ref_is_exact']}，ks_ref={fr_u.meta['ks_to_ref']:.4f}，"
        f"ks_normal={fr_u.meta['ks_to_normal']:.4f}；"
        f"指数：exact={fr_e.meta['ref_is_exact']}，ref_kind={fr_e.strings['ref_kind'][:24]!r}",
        tol="见 theory",
        passed=bool(
            fr_n.meta["ref_is_exact"] is True
            and abs(fr_n.meta["ks_to_ref"] - fr_n.meta["ks_to_normal"]) <= 2e-3
            and fr_u.meta["ref_is_exact"] is True
            and fr_u.meta["ks_to_normal"] >= 3.0 * fr_u.meta["ks_to_ref"]
            and fr_e.meta["ref_is_exact"] is False
            and "CLT" in fr_e.strings["ref_kind"]
        ),
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 6. 装配层只许搬运
# --------------------------------------------------------------------------- #
@assertion("A", theory="帧里的数组逐位等于直接调用 numcore 的结果（点集/指数/误差/系数）", chapter="5.4")
def test_frames_match_kernel_calls_bitwise() -> None:
    r"""四个可视化各抽一个数组与核心对拍（**逐位相等**，不是"接近"）：

    * ``point_fill`` 的 ``sobol_x`` = ``sobol_seq.point_set("sobol", 4096, 2, randomize=True)`` 的前 k 个；
    * ``sobol_bars`` 的 ``first`` = ``sensitivity.pick_freeze_indices`` 的 ``first_alt``（estimator=jansen）；
    * ``output_hist`` 的 ``counts`` 之和 = N；
    * ``pce_spectrum`` 的 ``error``（对数）与 ``pce.relative_l2_error`` 的 log10 一致（未触地板处）。
    """
    fr1 = _frame("point_fill", method="sobol", randomize=True, n_max=256, t=256.0)
    pts = sobol_seq.point_set("sobol", 4096, 2, seed=20261018, randomize=True)
    same_x = float(np.max(np.abs(np.asarray(fr1["sobol_x"]) - pts[:256, 0])))
    fr3 = _frame("sobol_bars", model="ishigami", n=512, rho=0.0, estimator="jansen")
    ind = sensitivity.pick_freeze_indices("ishigami", 512, seed=20261018, bootstrap=200)
    same_first = float(np.max(np.abs(np.asarray(fr3["first"]) - np.asarray(ind.first_alt))))
    fr2 = _frame("output_hist", dist="beta", model="quadratic", n=20000, n_bins=40)
    total_counts = float(np.sum(np.asarray(fr2["counts"], dtype=float)))
    fr4 = _frame("pce_spectrum", target="exp", fit="regression", order=6, n_train=64)
    curve = pce.error_vs_order(
        pce.target_function("exp"), dim=1, orders=range(1, 13), n_train=64,
        kind="legendre", method="regression", n_test=2000,
    )
    ref = float(np.log10(np.asarray(curve["outsample"], dtype=float)[5]))
    got = float(np.asarray(fr4["error"], dtype=float)[5])
    rec = record(
        label="四处数组与核心调用的逐位/对数一致性",
        theory="点集与指数逐位相等（差 = 0）；计数和 = N；误差对数差 <= 1e-12",
        measured=f"点集差 {same_x:.1e}，指数差 {same_first:.1e}，计数和 {total_counts:.0f}，"
        f"误差 log10 差 {abs(got - ref):.3e}（帧的曲线来自同一个 pce.error_vs_order 调用）",
        tol="见 theory",
        passed=bool(
            same_x == 0.0 and same_first == 0.0 and total_counts == 20000.0 and abs(got - ref) <= 1e-9
        ),
    )
    assert rec.passed, rec.measured


@pytest.mark.parametrize("viz_id", VIZ_IDS)
def test_manifest_declares_aux_axis_arrays(viz_id: str) -> None:
    """``requiredAux`` 里声明的名字（含 error/residual）必须真的出现在帧里。"""
    spec = _viz(viz_id)
    fr = _frame(viz_id)
    names = set(fr.names)
    missing = [req for req in spec.required_aux if not any(req in n for n in names)]
    assert not missing, f"{viz_id} 的 requiredAux {missing} 在帧里找不到（现有：{sorted(names)}）"


@pytest.mark.parametrize("viz_id", VIZ_IDS)
def test_frame_strings_carry_measured_numbers(viz_id: str) -> None:
    """``strings`` 里必须出现**实测数字**（不是只有定性描述）：至少一个 ``note`` 字段含小数。"""
    fr = _frame(viz_id if viz_id != "point_fill" else "point_fill", t=512.0)
    notes = [v for k, v in fr.strings.items() if k.endswith("note")]
    joined = " ".join(notes)
    has_number = any(ch.isdigit() for ch in joined) and ("." in joined or "e-" in joined or "e+" in joined)
    assert notes and has_number, f"{viz_id} 的 note 字段缺少实测数字：{notes!r}"
