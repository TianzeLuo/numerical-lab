"""3.4 章：四个可视化的帧契约与"诚实性"断言（A/B 类）。

这一组测试做两件事：

* **帧契约**：四个 viz 在默认参数、参数角点、以及时间轴两端都能建出帧，
  全部数组有限、meta 全是有限标量、``requiredAux`` 真的在、口径字符串非空；
* **帧里的数字必须与直接计算一致**：把 ``meta`` 里的灵敏度、收敛斜率、舍入地板、
  条件数逐个拿 :mod:`numcore.ode.bvp` 重算一遍对拍。帧是"搬运工"，
  一旦搬运途中被截断/换成哨兵/口径漂移，这里当场炸。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from numcore.ode import bvp
from numcore.verify import assert_leq, assertion, record
from server.frames import build_frame
from server.registry import Registry, RegistryError

U = float(np.finfo(float).eps) / 2.0

CHAPTER = "3.4"
VIZ_IDS = ("shooting_scan", "convergence_rate", "node_clustering", "diff_matrix_structure")


@pytest.fixture(scope="module")
def entry():
    """章节注册表。

    先按 ``strict=True`` 走：那是本地能拿到的最严标准。但别的章节同时在写，
    他们的 manifest 坏了会让注册表整体抛错（本章真撞上过 4.4 的 requiredAux），
    所以退化成 ``strict=False`` 再确认**本章**在不在——别人坏不该把本章的测试带红，
    而本章自己坏了这里照样当场报错（错误原文一起打出来，不让原因藏在别人身上）。
    """
    try:
        return Registry(".").discover(strict=True)[CHAPTER]
    except RegistryError as exc:
        reg = Registry(".").discover(strict=False)
        assert CHAPTER in reg, f"章节 {CHAPTER} 无法注册；注册表原始错误：{exc}"
        return reg[CHAPTER]


def _corners(viz) -> list[dict]:
    """参数角点：每个参数取 min/max 一轮（其余取默认），外加默认整组。"""
    out = [{}]
    for p in viz.params:
        for val in (p.min, p.max):
            if val is None:
                continue
            if p.type == "int":
                out.append({p.name: int(val)})
            elif p.type == "float":
                out.append({p.name: float(val)})
            elif p.type == "enum":
                for opt in p.options:
                    out.append({p.name: opt})
    return out


# --------------------------------------------------------------------------- #
# A 类：帧契约 + 数字对拍
# --------------------------------------------------------------------------- #
@assertion("A", "四个可视化在默认参数、参数角点与时间轴两端都能建出合法帧，且所有数组/标量有限", chapter="3.4")
def test_frames_are_valid_everywhere(entry) -> None:
    checked = 0
    n_arrays = 0
    for viz in entry.manifest.vizzes:
        assert viz.id in VIZ_IDS, f"出现了计划外的 viz：{viz.id}"
        for params in _corners(viz):
            for t in (0.0, 0.5 * float(viz.time.t_max or 0.0), float(viz.time.t_max or 0.0)):
                fr = build_frame(entry, viz.id, params, t)
                fr.validate()
                checked += 1
                n_arrays += len(fr.names)
                for name in fr.names:
                    arr = np.asarray(fr[name], dtype=float)
                    assert np.all(np.isfinite(arr)), f"{viz.id} 的数组 {name} 含非有限值（params={params}, t={t}）"
                for key, val in fr.meta.items():
                    assert isinstance(val, (int, float, bool)) or val is None, f"{viz.id}: meta[{key}] 不是标量"
                    if isinstance(val, float):
                        assert math.isfinite(val), f"{viz.id}: meta[{key}] 非有限（params={params}, t={t}）"
                assert fr.pitfalls and len(fr.pitfalls) >= 60, f"{viz.id} 的 pitfalls 太短"
                assert fr.strings.get("nonfinite_policy"), f"{viz.id} 缺少非有限值口径字符串"
    assert checked >= 20, f"角点覆盖太少：只查了 {checked} 帧"
    record(
        label="帧契约：角点 × 时间端点的合法帧数",
        theory="所有参数角点与时间端点都必须建出有限、合规的帧",
        measured=f"{len(VIZ_IDS)} 个 viz，共 {checked} 帧、{n_arrays} 个数组段，全部有限且 meta 全是有限标量",
        tol="0 个非有限值",
        passed=True,
        kind="A",
        extra={"frames": checked, "arrays": n_arrays},
    )


@assertion("A", "viz1（打靶扫描）meta 里的灵敏度、参考斜率与打靶误差与直接重算一致", chapter="3.4")
def test_shooting_frame_numbers_match_direct_computation(entry) -> None:
    fr = build_frame(entry, "shooting_scan", {"problem": "decay", "k": 10.0, "L": 1.0}, 0.0)
    prob = bvp.problem("decay", k=10.0, L=1.0)
    s_direct = bvp.shooting_sensitivity(prob, n_steps=int(fr.meta["n_steps"]))
    closed = math.sinh(10.0) / 10.0
    rel = abs(float(fr.meta["sensitivity"]) - s_direct) / s_direct
    rel_closed = abs(s_direct - closed) / closed
    ref = bvp.shooting(prob, n_steps=int(fr.meta["n_steps"]), max_iter=20)
    slope_rel = abs(float(fr.meta["s_ref"]) - ref.slope) / abs(ref.slope)
    assert_leq(rel, 1e-12, label="帧内灵敏度 vs 直接重算（相对）")
    assert_leq(rel_closed, 1e-6, label="灵敏度 vs 闭式 sinh(kL)/k（相对）")
    assert_leq(slope_rel, 1e-12, label="帧内参考斜率 vs 直接重算（相对）")
    # 斜率是负的（衰减解在左端往下走），且 |s_ref| -> k*coth(kL)
    assert float(fr.meta["s_ref"]) < 0.0, "衰减算例的参考斜率应当为负"
    record(
        label="viz1 的数字对拍（decay, k=10, L=1）",
        theory="S=sinh(kL)/k=1101.32；s_ref=-k*coth(kL)",
        measured=f"帧内 S={float(fr.meta['sensitivity']):.6e}，直接重算 {s_direct:.6e}（相对 {rel:.1e}）；"
        f"闭式 {closed:.6e}（相对 {rel_closed:.1e}）；s_ref={float(fr.meta['s_ref']):.6f}",
        tol="灵敏度相对 1e-12，斜率相对 1e-12",
        passed=rel <= 1e-12 and slope_rel <= 1e-12 and rel_closed <= 1e-6,
        kind="A",
        extra={"sensitivity": float(fr.meta["sensitivity"]), "closed_form": closed, "s_ref": float(fr.meta["s_ref"])},
    )


@assertion("A", "viz2（收敛率）meta 里的指数斜率、幂律指数、舍入地板与直接重算一致，且窗口写进了字符串", chapter="3.4")
def test_convergence_frame_numbers_match_direct_computation(entry) -> None:
    fr = build_frame(entry, "convergence_rate", {"problem": "smooth", "nmax": 256}, 0.0)
    ns = [4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32]
    study = bvp.convergence_study(bvp.problem("smooth"), ns)
    info = bvp.roundoff_floor(study["error"], study["n"])
    head = int(np.count_nonzero(study["error"] > 100.0 * info["floor"]))
    head = max(4, min(head, len(ns) - 1))
    fit = bvp.fit_exponential(study["n"], study["error"], head=head)
    d_floor = abs(float(fr.meta["floor"]) - info["floor"]) / info["floor"]
    d_slope = abs(float(fr.meta["exp_slope"]) - float(fit["slope"]))
    d_r2 = abs(float(fr.meta["exp_r2"]) - float(fit["r2"]))
    assert_leq(d_floor, 1e-12, label="帧内地板 vs 直接重算（相对）")
    assert_leq(max(d_slope, d_r2), 1e-9, label="帧内半对数斜率/R^2 vs 直接重算")
    assert fit["slope"] < 0.0 and fit["r2"] > 0.99, f"解析算例应当是干净直线：斜率 {fit['slope']:.3f}，R^2={fit['r2']:.5f}"
    assert str(fr.meta["exp_fit_head"]) == str(head), "帧里的拟合窗口与重算不一致"
    text = " ".join(fr.strings.values())
    assert "窗口" in text and f"N={ns[0]}..{ns[head - 1]}" in text, "窗口必须写进 strings"
    assert "地板" in text, "舍入地板必须写进 strings"
    record(
        label="viz2 的数字对拍（smooth）",
        theory="半对数斜率由未触地板的粗端窗口拟合；地板=曲线最小误差",
        measured=f"斜率 {float(fr.meta['exp_slope']):.4f}（重算 {float(fit['slope']):.4f}），R^2={float(fr.meta['exp_r2']):.5f}，"
        f"地板 {float(fr.meta['floor']):.3e}（重算 {info['floor']:.3e}），窗口 N={ns[0]}..{ns[head - 1]}（head={head}）",
        tol="地板相对 1e-12；斜率/R^2 绝对 1e-9",
        passed=d_floor <= 1e-12 and max(d_slope, d_r2) <= 1e-9,
        kind="A",
        extra={"slope": float(fr.meta["exp_slope"]), "floor": float(fr.meta["floor"]), "head": head},
    )


@assertion("A", "viz3（节点聚集）meta 里的聚集比、密度比与直接重算一致，且插值误差与配置解误差的关系正确", chapter="3.4")
def test_node_clustering_frame_numbers_match(entry) -> None:
    fr = build_frame(entry, "node_clustering", {"problem": "abs3", "n": 24}, 0.0)
    ratio = bvp.cluster_ratio(24)
    dens_ratio = float(bvp.node_density(24, np.array([0.9]))[0] / bvp.node_density(24, np.array([0.0]))[0])
    assert_leq(abs(float(fr.meta["cluster_ratio"]) - ratio) / ratio, 1e-12, label="聚集比 vs 直接重算")
    assert_leq(abs(float(fr.meta["density_ratio"]) - dens_ratio) / dens_ratio, 1e-12, label="密度比 vs 直接重算")
    assert_leq(abs(dens_ratio - 1.0 / math.sqrt(1.0 - 0.81)), 1e-12, label="密度比 vs 解析 1/sqrt(0.19)")
    # 插值误差与配置解误差：节点上插值为 0，节点之间才是误差；非解析算例下配置解更差
    err = np.asarray(fr["error"], dtype=float)
    coll = np.asarray(fr["collocation_error"], dtype=float)
    assert err.size == coll.size and np.all(np.isfinite(err))
    assert float(fr.meta["interp_error_max"]) > 1e-6, "abs3 的插值误差不该是舍入级"
    assert float(np.max(coll)) > 0.0 and float(fr.meta["collocation_error_max"]) >= float(fr.meta["interp_error_max"]), (
        "非解析算例下配置解误差应当不小于插值误差（实测差一阶）"
    )
    record(
        label="viz3 的数字对拍（abs3, N=24）",
        theory="聚集比 0.6366N；密度比 1/sqrt(0.19)=2.294；配置解比插值差一阶",
        measured=f"聚集比 {float(fr.meta['cluster_ratio']):.4f}（重算 {ratio:.4f}）；密度比 {float(fr.meta['density_ratio']):.6f}（解析 {1 / math.sqrt(0.19):.6f}）；"
        f"插值峰值 {float(fr.meta['interp_error_max']):.2e}，配置解误差 {float(fr.meta['collocation_error_max']):.2e}",
        tol="两个比值相对 1e-12；配置解误差 ≥ 插值峰值",
        passed=True,
        kind="A",
        extra={"cluster_ratio": ratio, "density_ratio": dens_ratio},
    )


@assertion("A", "viz4（微分矩阵）meta 的 ||D||=N^2、行和残差、朴素 kappa 与 kappa+ 与直接重算一致", chapter="3.4")
def test_diff_matrix_frame_numbers_match(entry) -> None:
    fr = build_frame(entry, "diff_matrix_structure", {"n": 16, "which": "D"}, 0.0)
    diag = bvp.diff_matrix_diagnostics(16)
    assert_leq(abs(float(fr.meta["norm_inf"]) - 16.0 * 16.0) / 256.0, 1e-12, label="||D||_inf vs N^2（相对）")
    assert_leq(abs(float(fr.meta["rowsum_residual"]) - diag["rowsum_residual"]), 1e-18, label="行和残差 vs 直接重算")
    assert_leq(abs(float(fr.meta["kappa_nonzero"]) - diag["kappa_nonzero"]) / diag["kappa_nonzero"], 1e-12, label="kappa+ vs 直接重算")
    assert float(fr.meta["kappa_naive"]) > 1e12, "朴素 kappa 应当被零空间噪声支配（>1e12）"
    assert float(fr.meta["kappa_nonzero"]) < 1e4, "kappa+ 应当是有界的（N=16 时 ~130）"
    d00 = abs(float(fr.meta["d00"]) - (2.0 * 256.0 + 1.0) / 6.0) / ((2.0 * 256.0 + 1.0) / 6.0)
    assert_leq(d00, 1e-13, label="D_00 vs (2N^2+1)/6（相对）")
    text = " ".join(fr.strings.values())
    assert "不是用来解方程组" in text, "必须写明微分矩阵不是用来解方程组的"
    assert "条件数" in text, "必须写明条件数口径"
    record(
        label="viz4 的数字对拍（N=16, D）",
        theory="||D||_inf=N^2；D·1=0；kappa+ ~ N^2；朴素 kappa 无意义",
        measured=f"||D||={float(fr.meta['norm_inf']):.6f}（N^2=256），行和残差 {float(fr.meta['rowsum_residual']):.2e}，"
        f"朴素 kappa={float(fr.meta['kappa_naive']):.2e}，kappa+={float(fr.meta['kappa_nonzero']):.3f}",
        tol="范数相对 1e-12；kappa+ 相对 1e-12",
        passed=True,
        kind="A",
        extra={"kappa_nonzero": float(fr.meta["kappa_nonzero"]), "kappa_naive": float(fr.meta["kappa_naive"])},
    )
