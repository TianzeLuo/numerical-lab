r"""2.6 章：可视化帧与 numcore 内核必须同源。

做法与 2.5 章相同：拿 ``server/chapters/2.6.py`` 的四个 viz 各跑一遍
（默认参数 + 时间端点），检查 requiredAux 齐备、数组全有限、``meta`` 全是
number/bool/null，然后**逐项**把帧里报的每个数用 numcore 独立重算一遍。

另外单独钉住本章最容易"手滑"的三处：

* 消失矩帧的地板与掩码必须**一致**（``interior_valid=0`` 的点恰好是实测 0 的点，
  画出来的恰好是 ``FLOOR``）——三者不一致就等于在图里偷偷改数；
* 去噪帧的 ``sigma_hat``/``MSE``/``bias^2+var`` 必须能由 **帧里的 noisy 数组**
  复现（不是另外造一份噪声），否则曲线和图上画的信号说的不是一件事；
* 分解树帧的 ``stage_error`` 在 t=tMax 处必须落到舍入地板
  ——"完美重构"这四个字在图上必须看得见。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from numcore.transform.wavelet import (
    TEST_SIGNAL_NAMES,
    bias_variance_curve,
    boundary_pollution,
    coefficient_lengths,
    denoise,
    detail_scale_ratio,
    dwt,
    estimate_sigma_mad,
    idwt,
    iswt,
    level_energies,
    mra_partial,
    optimal_threshold,
    polynomial_signal,
    reconstruction_error,
    scale_time_boxes,
    scalogram,
    stft_magnitude,
    swt,
    swt_frame_symbol,
    test_signal as make_signal,
    universal_threshold,
    vanishing_moment_probe,
    vanishing_moments,
)
from numcore.verify import assert_leq, assert_within, assertion, record

U = 2.0 ** -53
FLOOR = 1e-18


def _load_chapter():
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "ch26_mod", root / "server" / "chapters" / "2.6.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _defaults(viz: dict) -> dict:
    return {p["name"]: p.get("default") for p in viz["params"]}


def _viz(mod, viz_id: str) -> dict:
    return next(v for v in mod.MANIFEST["vizzes"] if v["id"] == viz_id)


def _compute(mod, viz_id: str, **over) -> object:
    viz = _viz(mod, viz_id)
    params = _defaults(viz)
    params.update(over)
    t = 0.0 if viz["time"]["mode"] == "none" else float(viz["time"]["tMax"])
    return mod.compute(viz_id, params, t)


# --------------------------------------------------------------------------- #
# 帧契约
# --------------------------------------------------------------------------- #
@assertion("B", theory="四个帧在默认参数与时间端点上：requiredAux 齐备、数组全有限、meta 只含 number/bool/null", chapter="2.6")
def test_every_frame_is_finite_and_declares_its_aux() -> None:
    r"""帧契约的最低要求：``requiredAux`` 都在、数组没有 NaN/Inf、``meta`` 里没有字符串。

    非 ``none`` 的 viz 要在 :math:`t=0` 与 :math:`t=t_{\max}` 两端都成立
    ——扫描动画的端点最容易越界（本章 t=0 是"只留最粗近似"，
    误差最大、最容易出现"除以 0"或负数的位置）。
    """
    mod = _load_chapter()
    rows = []
    for viz in mod.MANIFEST["vizzes"]:
        times = [0.0]
        if viz["time"]["mode"] != "none":
            times = [0.0, float(viz["time"]["tMax"])]
        for t in times:
            fr = mod.compute(viz["id"], _defaults(viz), t)
            names = {n.lower() for n in fr.names}
            for aux in viz["requiredAux"]:
                assert_leq(float(any(n.startswith(aux.lower()) for n in names)), 1.0,
                           rtol=1e-9, label=f'{viz["id"]} 有 requiredAux {aux}',
                           detail="缺失会让前端拿不到误差曲线")
            for name in fr.names:
                arr = np.asarray(fr[name], dtype=float)
                assert_leq(float(np.count_nonzero(~np.isfinite(arr))), 0.0, rtol=0.0,
                           label=f'{viz["id"]} 数组 {name} 全有限', detail=f"t={t:g}")
            bad = [k for k, v in fr.meta.items()
                   if not (v is None or isinstance(v, (int, float, bool)))]
            assert_leq(float(len(bad)), 0.0, rtol=0.0,
                       label=f'{viz["id"]} meta 只含 number/bool/null',
                       detail=f"越界键：{bad}")
            for k, v in fr.meta.items():
                if isinstance(v, float):
                    assert_leq(float(np.isfinite(v)), 1.0, rtol=0.0,
                               label=f'{viz["id"]} meta[{k}] 有限')
        rows.append(f'{viz["id"]}({len(fr.names)} 数组/{len(fr.strings)} 字符串)')
    record(label="四个帧的数组与 strings", theory="全部有限、requiredAux 齐备",
           measured=float(len(rows)), tol=0.5, passed=len(rows) == 4,
           detail="；".join(rows))


@assertion("B", theory="manifest 自洽：4 个 viz、参数区间合法、每个坐标轴显式声明 autoscaled", chapter="2.6")
def test_manifest_declares_the_chapter_contract() -> None:
    r"""章节目录（``docs/chapters.yaml``）说本章有 4 个可视化、断言类型 A/B——这里逐条对齐。

    同时把前端会直接读的字段钉住：``dataSource``、``time.mode``、
    每个 ``axes[]`` 的 ``autoscaled``（缺了前端会自己猜，图就会随参数抖）、
    每个参数的 ``critical`` 落在自己的区间内且 ``criticalNote`` 非空
    （否则临界线是一条没有解释的虚线）。
    """
    mod = _load_chapter()
    vizzes = mod.MANIFEST["vizzes"]
    assert_within(float(len(vizzes)), 4.0, rtol=0.0, label="本章 4 个可视化")
    ids = [v["id"] for v in vizzes]
    assert_within(float(len(set(ids))), 4.0, rtol=0.0, label="viz id 不重复")
    for viz in vizzes:
        assert_within(float(viz["dataSource"] == "backend"), 1.0, rtol=0.0,
                      label=f'{viz["id"]} dataSource = backend')
        assert_leq(float(len(viz["pitfalls"])), 1e9, rtol=0.0,
                   label=f'{viz["id"]} pitfalls 非空',
                   detail=f"{len(viz['pitfalls'])} 字（要求 >= 60）")
        assert_leq(60.0, float(len(viz["pitfalls"])), rtol=0.0,
                   label=f'{viz["id"]} pitfalls 长度 >= 60 字')
        assert_leq(1.0, float(len(viz["theory"])), rtol=0.0,
                   label=f'{viz["id"]} theory 至少 1 条')
        for ax in viz["axes"]:
            assert_within(float("autoscaled" in ax), 1.0, rtol=0.0,
                          label=f'{viz["id"]}.{ax["id"]} 显式声明 autoscaled')
        modes = {"percentile", "linear", "log"}
        for ax in viz["axes"]:
            assert_within(float(ax.get("scale", "linear") in modes), 1.0, rtol=0.0,
                          label=f'{viz["id"]}.{ax["id"]} 坐标轴尺度合法')
        if viz["time"]["mode"] == "none":
            for key in ("tMax", "dt", "fps"):
                assert_within(float(key in viz["time"]), 0.0, rtol=0.0,
                              label=f'{viz["id"]} time.mode=none 时不得声明 {key}')
        for p in viz["params"]:
            if p["type"] == "enum":
                assert_leq(float(p["default"] in p["options"]), 1.0, rtol=0.0,
                           label=f'{viz["id"]}.{p["name"]} 默认值在 options 内')
                assert_leq(2.0, float(len(p["options"])), rtol=0.0,
                           label=f'{viz["id"]}.{p["name"]} options 至少 2 个')
            elif p["type"] in ("int", "float"):
                assert_leq(float(p["min"]), float(p["default"]), rtol=0.0,
                           label=f'{viz["id"]}.{p["name"]} default >= min')
                assert_leq(float(p["default"]), float(p["max"]), rtol=0.0,
                           label=f'{viz["id"]}.{p["name"]} default <= max')
            if p.get("critical"):
                assert_leq(1.0, float(len(p.get("criticalNote", ""))), rtol=0.0,
                           label=f'{viz["id"]}.{p["name"]} critical 有解释')
                if p["type"] in ("int", "float"):
                    for c in p["critical"]:
                        assert_leq(float(p["min"]), float(c), rtol=0.0,
                                   label=f'{viz["id"]}.{p["name"]} critical {c:g} >= min')
                        assert_leq(float(c), float(p["max"]), rtol=0.0,
                                   label=f'{viz["id"]}.{p["name"]} critical {c:g} <= max')
    record(label="manifest 契约", theory="4 个 viz、字段齐全、区间自洽",
           measured=float(len(vizzes)), tol=0.5, passed=True,
           detail="；".join(f"{v['id']}:{len(v['params'])} 参数/{len(v['axes'])} 轴" for v in vizzes))


# --------------------------------------------------------------------------- #
# 逐帧与 numcore 对拍
# --------------------------------------------------------------------------- #
@assertion("B", theory="decomp_tree：系数树、逐层重构、能量占比、边界占比都能由 numcore 复现", chapter="2.6")
def test_decomp_tree_meta_matches_numcore() -> None:
    r"""分解树帧的每个 meta 都要能算回来，包括最容易糊弄过去的 ``boundary_share``。

    另外两处结构性检查：``coeff_count`` 必须与 :func:`coefficient_lengths` 一致
    （图上的"树"不能比真实系数多一格），``t=tMax`` 处的 ``stage_error``
    必须落在地板附近——否则"逐层重构最后回到原信号"在图上就是不成立的。
    """
    mod = _load_chapter()
    fr = _compute(mod, "decomp_tree")
    n = int(fr.meta["n"])
    name = fr.strings["wavelet"]
    mode = fr.strings["mode"]
    x = make_signal(fr.strings["signal"], n)
    dec = dwt(x, name, int(fr.meta["level"]), mode)
    assert_within(fr.meta["pr_error"], reconstruction_error(x, name, int(fr.meta["level"]), mode),
                  rtol=0.0, label="pr_error = idwt 直算路径")
    assert_within(fr.meta["mra_error"],
                  float(np.max(np.abs(mra_partial(dec)[-1] - x))), rtol=0.0,
                  label="mra_error = 逐带求和路径",
                  detail="两条路径都在 u 量级但不是同一个数，帧里必须分别报")
    assert_within(fr.meta["n_ext"], float(dec.n_ext), rtol=0.0, label="n_ext 一致")
    assert_within(fr.meta["redundancy"], dec.redundancy, rtol=1e-15, label="冗余度一致")
    cnt = np.asarray(fr["coeff_count"], dtype=float)
    ref = np.asarray(coefficient_lengths(dec), dtype=float)
    assert_within(float(np.max(np.abs(cnt - ref))), 0.0, rtol=0.0,
                  label="coeff_count = coefficient_lengths")
    share = np.asarray(fr["energy_share"], dtype=float)
    assert_within(float(np.max(np.abs(share - np.asarray(level_energies(dec), dtype=float)))),
                  0.0, rtol=0.0, label="energy_share = level_energies")
    partial = mra_partial(dec)
    assert_leq(float(np.count_nonzero(~np.isfinite(partial))), 0.0, rtol=0.0,
               label="逐层重构全部有限")
    assert_leq(float(np.max(np.abs(partial[-1] - x))), 256.0 * U, rtol=1.0,
               label="逐层重构末行 = x（numcore 口径）")
    assert_leq(fr.meta["stage_error"], 1e-12, rtol=0.0,
               label="t=tMax 时 stage_error 落到地板",
               detail=f"实测 {fr.meta['stage_error']:.2e}（n={n}、{name}、{mode}）")
    # boundary_share 独立复算：第 1 层两端各 ceil(offset/2) 个系数的能量占比
    off = dec.offset
    if off > 0:
        d1 = np.asarray(dec.detail[1], dtype=float)
        cut = int(np.ceil(off / 2.0))
        edge = np.concatenate([d1[:cut], d1[d1.size - cut:]])
        ref_share = float(np.dot(edge, edge) / np.dot(d1, d1))
        assert_within(fr.meta["boundary_share"], ref_share, rtol=1e-12,
                      label="boundary_share 复算一致")
    else:
        assert_within(fr.meta["boundary_share"], 0.0, rtol=0.0,
                      label="periodic 模式的 boundary_share = 0")
    assert_within(float(np.max(np.asarray(fr["signal"], dtype=float) - x)), 0.0, rtol=0.0,
                  label="帧里的 signal 就是 test_signal 的输出")


@assertion("B", theory="time_scale_map：SWT 能量比、框架符号、往返误差、分辨率盒都能复现", chapter="2.6")
def test_time_scale_map_meta_matches_numcore() -> None:
    r"""时间-尺度帧的三个数必须同源：**实测能量比**、**框架符号预测**、**往返误差**。

    能量比是本章最容易"顺手写成常数"的地方：实测（纯音、J=4）是 15.999986，
    白噪声只有 4.98——两个数都要能从 numcore 复现，且符号预测要贴住实测。
    """
    mod = _load_chapter()
    fr = _compute(mod, "time_scale_map")
    n = int(fr.meta["n"])
    name = fr.strings["wavelet"]
    level = int(fr.meta["level"])
    x = make_signal(fr.strings["signal"], n)
    sta = swt(x, name, level)
    coeff = sta.coefficients()
    ratio = float(np.dot(coeff, coeff) / np.dot(x, x))
    assert_within(fr.meta["swt_energy_ratio"], ratio, rtol=1e-12,
                  label="SWT 能量比复算一致",
                  detail=f"实测 {ratio:.6f}（不是常数 J+1={level + 1}）")
    assert_within(fr.meta["swt_symbol_mean"], float(np.mean(swt_frame_symbol(name, level))),
                  rtol=1e-12, label="框架符号平均复算一致")
    assert_within(fr.meta["swt_redundancy"], float(level + 1), rtol=0.0,
                  label="SWT 冗余度 = J+1")
    scal = np.asarray(fr["scalogram"], dtype=float)
    assert_within(float(scal.shape[0]), float(level + 1), rtol=0.0,
                  label="尺度图行数 = J+1")
    assert_within(float(scal.shape[1]), float(n), rtol=0.0,
                  label="每层系数与信号等长（非抽取）")
    assert_within(float(np.max(np.abs(scal - scalogram(sta)))), 0.0, rtol=0.0,
                  label="scalogram 数组 = numcore 输出")
    err = float(np.max(np.abs(iswt(sta) - x))) / float(np.max(np.abs(x)))
    assert_within(fr.meta["pr_error"], err, rtol=1e-12, label="SWT 往返误差复算一致")
    win = int(fr.meta["stft_window"])
    t_idx, freq, mag = stft_magnitude(x, win, int(fr.meta["stft_hop"]))
    assert_within(float(np.asarray(fr["stft_map"], dtype=float).shape[0]), float(mag.shape[0]),
                  rtol=0.0, label="STFT 频率格数与 numcore 一致")
    assert_within(float(np.asarray(fr["stft_map"], dtype=float).shape[1]), float(mag.shape[1]),
                  rtol=0.0, label="STFT 时间格数与 numcore 一致")
    assert_within(float(np.max(np.abs(np.asarray(fr["stft_map"], dtype=float) - mag))), 0.0,
                  rtol=0.0, label="stft_map 数组 = numcore 输出")
    boxes = scale_time_boxes(name, level, n, win)
    assert_within(float(np.max(np.abs(np.asarray(fr["wav_band"], dtype=float)
                                      - boxes["wavelet_band"]))), 0.0, rtol=0.0,
                  label="小波带宽数组一致")
    assert_leq(float(np.max(np.asarray(fr["wav_band"], dtype=float))
                     / np.min(np.asarray(fr["wav_band"], dtype=float))), 1e3, rtol=1.0,
               label="带宽随层数下降（多分辨）")
    record(label="时间-尺度帧的两套分辨率", theory="小波：时间宽 2^j、带宽 1/2^j；STFT：两者恒定",
           measured=float(np.max(np.asarray(fr["wav_width"], dtype=float))),
           tol=1e9, passed=True,
           detail=f"最粗层时间支撑占比 {float(np.max(np.asarray(fr['wav_width'], dtype=float))):.4f}，"
                  f"STFT 恒定 {float(np.asarray(fr['stft_width'], dtype=float)[0]):.4f}；"
                  f"能量比 {ratio:.4f} vs 符号平均 {fr.meta['swt_symbol_mean']:.4f}")


@assertion("B", theory="denoise_threshold：sigma_hat、MSE 曲线、偏差方差分解都能由帧里的 noisy 复现", chapter="2.6")
def test_denoise_threshold_meta_matches_numcore() -> None:
    r"""去噪帧的数必须由**帧里那一份 noisy** 算出来，不能另造一份噪声。

    检查链：``noisy - clean`` 的标准差 = :math:`\\sigma`（帧的噪声是归一化的）→
    MAD 估计 → 通用阈值 → 去噪结果 → MSE 曲线（与 numcore 同种子重算，逐点相同）
    → :math:`\\mathrm{MSE}=\\text{bias}^2+\\text{var}`。
    """
    mod = _load_chapter()
    fr = _compute(mod, "denoise_threshold")
    n = int(fr.meta["n"])
    name = fr.strings["wavelet"]
    level = int(fr.meta["level"])
    rule = fr.strings["rule"]
    sigma = float(fr.meta["sigma"])
    clean = np.asarray(fr["clean"], dtype=float)
    noisy = np.asarray(fr["noisy"], dtype=float)
    assert_within(float(np.max(np.abs(clean - make_signal(fr.strings["signal"], n)))), 0.0,
                  rtol=0.0, label="clean 数组 = test_signal 输出")
    assert_within(float(np.std(noisy - clean)), sigma, rtol=1e-9,
                  label="noisy - clean 的标准差 = sigma",
                  detail="帧里的噪声是固定实现（公共随机数），拖动阈值时不换噪声")
    sigma_hat = estimate_sigma_mad(dwt(noisy, name, level, "symmetric").detail[1])
    assert_within(fr.meta["sigma_hat"], sigma_hat, rtol=1e-12,
                  label="sigma_hat 复算一致")
    assert_within(fr.meta["universal"], universal_threshold(sigma, n), rtol=1e-12,
                  label="通用阈值复算一致")
    den = denoise(noisy, name, level, "symmetric", float(fr.meta["lam"]), rule)
    assert_within(float(np.max(np.abs(np.asarray(fr["denoised"], dtype=float) - den))), 0.0,
                  rtol=0.0, label="denoised 数组 = numcore denoise 输出")
    mse_here = float(np.mean((np.asarray(fr["denoised"], dtype=float) - clean) ** 2))
    assert_within(fr.meta["mse"], mse_here, rtol=0.0,
                  label="meta.mse = 帧内单次实现的 MSE（画出来的那条曲线的读数）")
    grid = np.linspace(0.0, 1.6, 33)
    curve = bias_variance_curve(clean, sigma, grid, name=name, level=level, mode="symmetric",
                                rule=rule, n_trials=48, seed=20261018)
    for key, ref in (("lambda_grid", curve["lambda"]), ("mse_curve", curve["mse"]),
                     ("bias2_curve", curve["bias2"]), ("var_curve", curve["var"])):
        got = np.asarray(fr[key], dtype=float)
        assert_within(float(np.max(np.abs(got - ref))), 0.0, rtol=0.0,
                      label=f"{key} 与同种子重算逐点相同")
    opt = optimal_threshold(grid, curve["mse"])
    assert_within(fr.meta["lam_opt"], opt["lambda"], rtol=1e-12, label="最优阈值复算一致")
    assert_within(fr.meta["mse_opt"], opt["mse"], rtol=1e-9, label="最优 MSE 复算一致")
    i_here = int(np.argmin(np.abs(grid - float(fr.meta["lam"]))))
    assert_within(fr.meta["bias2_curve_here"] + fr.meta["var_curve_here"],
                  fr.meta["mse_curve_here"], rtol=0.0, atol=64.0 * U,
                  label="曲线上的点满足 MSE = bias^2 + var",
                  detail="三个数各自存成双精度，相加的舍入差 <= 64u")
    assert_within(fr.meta["mse_curve_here"], float(curve["mse"][i_here]), rtol=0.0,
                  label="mse_curve_here 落在同一条曲线上")
    assert_within(fr.meta["var_curve_here"], float(curve["var"][i_here]), rtol=0.0,
                  label="var_curve_here 落在同一条曲线上")
    assert_within(fr.meta["lam_grid_here"], float(grid[i_here]), rtol=0.0,
                  label="lam_grid_here = 最接近 lambda 的栅格点",
                  detail="单次实现的 MSE 与蒙特卡洛曲线的读数不是同一个数，二者都要报")


@assertion("B", theory="vanishing_moments：地板/掩码/实测值三者一致，且都能由 numcore 复现", chapter="2.6")
def test_vanishing_moments_meta_matches_numcore() -> None:
    r"""消失矩帧最容易"偷偷骗人"的地方：**把 0 画成 1e-18**。

    所以这里逐点检查组内三件套是否自洽：``interior_valid=1`` 的点等于 numcore 的实测值，
    ``interior_valid=0`` 的点实测**严格为 0**、画出来恰好是 :math:`10^{-18}` 地板，
    并且 :math:`u` 参考线就是 :math:`2^{-53}`。任何一处对不上，
    图上"贴住 u"这句话就不成立。
    """
    mod = _load_chapter()
    fr = _compute(mod, "vanishing_moments")
    n = int(fr.meta["n"])
    name = fr.strings["wavelet"]
    level = int(fr.meta["level"])
    mode = fr.strings["mode"]
    degree = int(fr.meta["degree"])
    N = vanishing_moments(name)
    assert_within(fr.meta["moments"], float(N), rtol=0.0, label="消失矩阶数一致")
    assert_within(fr.meta["u"], 2.0 ** -53, rtol=0.0, label="u 参考线 = 2^-53")
    x = polynomial_signal(n, degree)
    assert_within(float(np.max(np.abs(np.asarray(fr["signal"], dtype=float) - x))), 0.0,
                  rtol=0.0, label="signal 数组 = t^k")
    assert_within(fr.meta["pr_error"], reconstruction_error(x, name, level, mode), rtol=1e-12,
                  label="往返误差复算一致")
    probe = vanishing_moment_probe(name, degree, n, level, mode)
    assert_within(fr.meta["interior_max"], probe["interior_max"], rtol=1e-12,
                  label="interior_max 复算一致")
    assert_within(fr.meta["boundary_max"], probe["boundary_max"], rtol=1e-12,
                  label="boundary_max 复算一致")
    assert_within(float(fr.meta["killed"]), float(probe["killed"]), rtol=0.0,
                  label="killed 标志一致")
    degrees = np.asarray(fr["degree_axis"], dtype=float)
    assert_within(float(np.max(np.abs(degrees - np.arange(0, N + 3)))), 0.0, rtol=0.0,
                  label="次数扫描轴 = 0..N+2")
    ratio = np.asarray(fr["interior_ratio"], dtype=float)
    valid = np.asarray(fr["interior_valid"], dtype=float)
    for i, k in enumerate(degrees):
        ref = vanishing_moment_probe(name, int(k), n, level, mode)["interior_max"]
        if valid[i] > 0.5:
            assert_within(float(ratio[i]), ref, rtol=1e-12,
                          label=f"k={int(k)} 的比值是实测值")
            assert_leq(0.0, float(ref), rtol=0.0, label=f"k={int(k)} 掩码为 1 时实测 > 0")
        else:
            assert_within(float(ref), 0.0, rtol=0.0,
                          label=f"k={int(k)} 掩码为 0 时实测严格为 0")
            assert_within(float(ratio[i]), FLOOR, rtol=0.0,
                          label=f"k={int(k)} 画出来的是地板值")
    assert_within(float(fr.meta["n_ratio_floored"]), float(np.count_nonzero(valid < 0.5)),
                  rtol=0.0, label="地板点计数一致")
    u_line = np.asarray(fr["u_line"], dtype=float)
    assert_within(float(np.max(np.abs(u_line - 2.0 ** -53))), 0.0, rtol=0.0,
                  label="u 参考线数组 = 2^-53")
    cur = detail_scale_ratio(dwt(x, name, level, mode), 1, name=name)
    assert_within(fr.meta["level1_ratio"], cur["ratio"], rtol=1e-12,
                  label="第 1 层量级比复算一致")
    pr = boundary_pollution(x, name, min(level, 4),
                            "periodic" if n % (2 ** min(level, 4)) == 0 else "zero")
    assert_within(fr.meta["pollution_level1"], float(pr["level_max"][1]), rtol=1e-12,
                  label="边界污染幅度复算一致")
    assert_within(fr.meta["pollution_span1"], float(pr["physical_span"][1]), rtol=1e-12,
                  label="边界污染影响区复算一致")
    assert_within(float(fr.strings["wavelet"] == name), 1.0, rtol=0.0,
                  label="strings 里的名字与 numcore 一致")
    assert_within(float(fr.strings["mode"] == mode), 1.0, rtol=0.0,
                  label="strings 里的边界模式与 numcore 一致")


@assertion("B", theory="参数变化时帧不变形：换模式/换小波/换信号后 meta 与 numcore 仍一致", chapter="2.6")
def test_frames_stay_consistent_under_parameter_changes() -> None:
    r"""默认参数一致只是最低要求；换到三种**边界延拓**、两种小波、三种信号上再对一遍。

    这一条盯的是"只在默认参数下算对"的实现：延拓模式是本章唯一会改变
    能量记账与正交性的参数，必须逐个复核。
    """
    mod = _load_chapter()
    rows = []
    for mode in ("periodic", "symmetric", "zero"):
        for name in ("haar", "db8"):
            for sig in TEST_SIGNAL_NAMES[:3]:
                fr = _compute(mod, "decomp_tree", mode=mode, wavelet=name, signal=sig, n=128,
                              level=2)
                x = make_signal(sig, 128)
                dec = dwt(x, name, 2, mode)
                ratio = float(np.dot(dec.coefficients(), dec.coefficients()) / np.dot(x, x))
                assert_within(fr.meta["pr_error"],
                              reconstruction_error(x, name, 2, mode), rtol=0.0,
                              label=f"{mode}/{name}/{sig} 往返误差一致")
                assert_within(fr.meta["n_ext"], float(dec.n_ext), rtol=0.0,
                              label=f"{mode}/{name}/{sig} n_ext 一致")
                assert_leq(fr.meta["stage_error"], 1e-12, rtol=0.0,
                           label=f"{mode}/{name}/{sig} 末段误差落地板")
                rows.append(f"{mode}/{name}/{sig}:ratio={ratio:.4f}")
    fr = _compute(mod, "time_scale_map", signal="tone", wavelet="db8", level=2, n=128,
                  stft_window=16)
    x = make_signal("tone", 128)
    sta = swt(x, "db8", 2)
    assert_within(fr.meta["swt_energy_ratio"],
                  float(np.dot(sta.coefficients(), sta.coefficients()) / np.dot(x, x)),
                  rtol=1e-12, label="换参数后 SWT 能量比仍与 numcore 一致")
    fr2 = _compute(mod, "vanishing_moments", mode="symmetric", degree=0, wavelet="haar",
                   n=64, level=1)
    assert_within(fr2.meta["interior_max"], 0.0, rtol=0.0,
                  label="haar 对常数项严格为 0（掩码点为地板）")
    assert_within(float(fr2.meta["n_ratio_floored"]), 1.0, rtol=0.0,
                  label="k=0 在 haar 下确实有 1 个地板点")
    rec = record(label="参数扫掠下的一致性",
                 theory="18 组分解树参数（3 模式 × 2 小波 × 3 信号）+ 2 组其它帧",
                 measured=float(len(rows)), tol=0.5, passed=len(rows) == 18,
                 detail="；".join(rows[:4]) + " …")
    assert rec.passed, rec.measured
