"""6.1 章的**帧级**测试：把"页面上看到的数字"和核心求解器钉在一起。

这一章最容易出的错不是"图不漂亮"，而是**图与文字说的不是同一件事**：

1. **有限性与装帧**：四页在默认参数与时间两端都必须没有非有限值，JSON 头里不许
   出现 ``NaN``/``Infinity``。图灵页在 ``t=10000`` 才会把 dt 夹紧后的饱和图案
   露出来，浅水页在 ``t=0`` 必须给出"零步"而不是一条假装算过的曲线。
2. **残差不是误差**：方腔页同时报 SOR 残差与 Ghia RMSE，两者差五个数量级；
   帧里必须保留这条对比，而不是只挑一个好看的数字。
3. **基准的身份**：Ghia 表的抄写分歧是 ``1e-2``，误差曲线未必随 n 单调到零；
   帧里报的收敛阶必须与四个网格点的实际 RMSE 对得上。
4. **图灵判据**：``turing`` 必须有正波数增长带；``stripes`` 的底态存在但
   ``λ(0)>0``（均匀失稳），``spots`` 没有非平凡稳态，``uniform`` 没有空间结构
   （波长无定义，帧里用 ``wavelength_valid=0`` + 0 占位）。
5. **干底守恒**：质量漂移是正性地板的注入，不是格式不守恒；扇内/全域 L1 必须
   分栏报，帧里的 ``fan``/``error_valid`` 掩码要能重建这两个窗口。
6. **装配只许搬运**：帧数组与直接调用 ``numcore`` 的结果逐位相等。
"""

from __future__ import annotations

import itertools
import json
import math
import pathlib
import struct
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from numcore.apps import cavity, reaction_diffusion as rd, shallow_water as sw  # noqa: E402
from numcore.verify import assertion, record  # noqa: E402
from server.frames import build_frame  # noqa: E402
from server.nlf import pack_frame  # noqa: E402
from server.registry import Registry  # noqa: E402

_ENTRY_CACHE: list = []
VIZ_IDS = ("cavity_vorticity", "ghia_benchmark", "turing_pattern", "dam_break")


def _entry():
    if not _ENTRY_CACHE:
        reg = Registry(ROOT)
        reg.discover(strict=False)  # 兄弟章节可处于半成品；本章必须能独立加载
        _ENTRY_CACHE.append(reg.get("6.1"))
    return _ENTRY_CACHE[0]


def _viz(viz_id: str):
    return next(v for v in _entry().manifest.vizzes if v.id == viz_id)


def _defaults(viz_id: str) -> dict:
    return {p.name: p.default for p in _viz(viz_id).params}


def _frame(viz_id: str, t: float = 0.0, **over):
    params = _defaults(viz_id)
    params.update(over)
    return build_frame(_entry(), viz_id, params, float(t))


# --------------------------------------------------------------------------- #
# 1. 四页全帧：有限、可装、requiredAux 真的存在
# --------------------------------------------------------------------------- #
@assertion("A", theory="四页在默认参数与 t=0/t=tMax 下数组全有限、meta 无 NaN、JSON 无 Infinity，requiredAux 可定位", chapter="6.1")
def test_all_frames_finite_packable_and_declare_aux() -> None:
    r"""``turing_pattern`` 的 ``t=tMax`` 是最重的一帧（10000 步、n=128），
    ``dam_break`` 的 ``t=0`` 是最容易露出占位错误的一帧（0 步、无扇区）。
    两页都必须能装帧；``requiredAux`` 里的名字必须在数组名集合里找得到。"""

    def _boom(token: str) -> None:
        raise AssertionError(f"JSON 里出现非有限常量 {token!r}")

    n_frames = 0
    n_arrays = 0
    bad_tokens = 0
    detail: list[str] = []
    for viz_id in VIZ_IDS:
        spec = _viz(viz_id)
        times = [0.0] if spec.time.mode == "none" else [0.0, float(spec.time.t_max)]
        for t in times:
            fr = _frame(viz_id, t)
            for name in fr.names:
                arr = np.asarray(fr[name], dtype=float)
                assert np.all(np.isfinite(arr)), f"{viz_id}/{name} 在 t={t} 处有非有限值"
            for key, val in fr.meta.items():
                if isinstance(val, float):
                    assert math.isfinite(val), f"{viz_id} meta.{key} 非有限：{val!r}"
            for req in spec.required_aux:
                assert any(req in name for name in fr.names), f"{viz_id} 缺 requiredAux={req}"
            blob = pack_frame(fr)
            try:
                (hlen,) = struct.unpack_from("<I", blob, 4)
                json.loads(blob[8 : 8 + hlen].decode("utf-8"), parse_constant=_boom)
            except AssertionError:
                bad_tokens += 1
            n_frames += 1
            n_arrays += len(fr.names)
            if t == times[0]:
                detail.append(f"{viz_id}: {len(fr.names)} 数组 / {len(fr.meta)} meta")
    rec = record(
        label="四页端点帧的有限性、JSON 常量与 requiredAux",
        theory="非有限常量 = 0；帧数 = 6；每页 requiredAux 都在数组名里",
        measured=f"{n_frames} 帧、{n_arrays} 个数组，非有限常量计数 {bad_tokens}（" + "；".join(detail) + "）",
        tol="计数 = 0 且帧数 >= 6",
        passed=bool(bad_tokens == 0 and n_frames >= 6),
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 2. 方腔：残差与离散误差必须同时在场
# --------------------------------------------------------------------------- #
@assertion("A", theory="Re=1000/n=41：残差 <=1e-6、RMSE >0.15、CFL 绑定；Re=100/n=49：RMSE <0.02；两处四舍五入都对得上", chapter="6.1")
def test_cavity_residual_and_discretization_error_are_separate() -> None:
    r"""默认帧的残差已经到 ``1e-7``，但与 Ghia 的 RMSE 仍有 0.227——这是本页的
    核心对照。同时钉住：

    * ``residual`` 曲线末点等于 ``log10(residual_final)``，不是另算的；
    * Re=1000/n=41 由 CFL 限制（``dt_binding=1``），格点雷诺数 25 > 2；
    * Re=100/n=49 的 RMSE 降到 0.006，说明同一套格式在低 Re 是准的；
    * 中心格式阈值恰为 ``2(n-1)=80``，侧栏文字带实测数字。
    """
    fr_hi = _frame("cavity_vorticity", re=1000, n=41)
    fr_lo = _frame("cavity_vorticity", re=100, n=49)
    hist = np.asarray(fr_hi["residual"], dtype=float)
    last_log = float(hist[-1])
    rec = record(
        label="方腔高/低 Re 的残差、RMSE 与时间步绑定",
        theory="高 Re：残差 <=1e-6、RMSE>0.15、CFL 绑定、Re_h>2；低 Re：RMSE<0.02；residual[-1]=log10(final)",
        measured=(
            f"Re=1000/n=41：残差 {fr_hi.meta['residual_final']:.2e}，RMSE {fr_hi.meta['rmse']:.4f}，"
            f"binding={fr_hi.meta['dt_binding']:.0f}，Re_h={fr_hi.meta['cell_re']:.1f}；"
            f"Re=100/n=49：RMSE {fr_lo.meta['rmse']:.4f}；"
            f"residual[-1]={last_log:.6f} vs log10(final)={math.log10(fr_hi.meta['residual_final']):.6f}；"
            f"中心阈值={fr_hi.meta['central_threshold_re']:.1f}"
        ),
        tol="见 theory",
        passed=bool(
            fr_hi.meta["converged"] == 1.0
            and fr_hi.meta["residual_final"] <= 1.0e-6
            and fr_hi.meta["rmse"] > 0.15
            and fr_hi.meta["dt_binding"] == 1.0
            and fr_hi.meta["cell_re"] > 2.0
            and fr_hi.meta["central_threshold_re"] == 80.0
            and fr_lo.meta["rmse"] < 0.02
            and abs(last_log - math.log10(fr_hi.meta["residual_final"])) < 1.0e-12
            and "25.00" in fr_hi.strings["central_note"]
            and "CFL" in fr_hi.strings["dt_note"]
        ),
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 3. Ghia：四个网格点、收敛阶、抄写地板
# --------------------------------------------------------------------------- #
@assertion("A", theory="Ghia Re=100：n=17/25/33/49 的 RMSE 递减，收敛阶 ~1.10，最优 n=49；n=25 的 RMSE 至少是 n=49 的 1.8 倍", chapter="6.1")
def test_ghia_curve_matches_grid_evidence() -> None:
    r"""帧里的 ``n_grid``/``error`` 必须与四个网格的实际 RMSE 一致，且页面报的
    ``grid_slope≈1.10`` 真的来自这四个点。抄写地板 1e-2 提醒读者：Re=1000 时
    RMSE 0.2 是离散误差，但 RMSE 0.006 与 0 的差别不能全算成格式误差。"""
    fr = _frame("ghia_benchmark", re=100, n=49)
    fr25 = _frame("ghia_benchmark", re=100, n=25)
    ns = np.asarray(fr["n_grid"], dtype=float)
    err = np.asarray(fr["error"], dtype=float)
    y_ref = np.asarray(fr["y_ref"], dtype=float)
    u_ref = np.asarray(fr["u_ref"], dtype=float)
    u_num = np.asarray(fr["u_num_ref"], dtype=float)
    rec = record(
        label="Ghia 四网格 RMSE 曲线与抄写地板",
        theory="n_grid=[17,25,33,49]；RMSE 严格递减；斜率∈[1.0,1.2]；n=25 的 RMSE >=1.8×n=49；中心距 <0.05",
        measured=(
            f"n={ns.astype(int).tolist()}，RMSE={np.array2string(err, precision=4)}；"
            f"斜率={fr.meta['grid_slope']:.3f}，best_n={fr.meta['best_n']:.0f}；"
            f"n=25 RMSE={fr25.meta['rmse']:.4f} vs n=49 {fr.meta['rmse']:.4f}；"
            f"抄写地板={fr.meta['transcription_floor']:.1e}，中心距={fr.meta['center_dist']:.4f}；"
            f"基准点 {y_ref.size} 个，u 数组同长={y_ref.size == u_ref.size == u_num.size}"
        ),
        tol="见 theory",
        passed=bool(
            np.array_equal(ns, np.array([17.0, 25.0, 33.0, 49.0]))
            and np.all(np.diff(err) < 0.0)
            and 1.0 <= float(fr.meta["grid_slope"]) <= 1.2
            and fr.meta["best_n"] == 49.0
            and fr25.meta["rmse"] >= 1.8 * fr.meta["rmse"]
            and fr.meta["transcription_floor"] == 0.01
            and fr.meta["center_dist"] < 0.05
            and y_ref.size >= 15
            and y_ref.size == u_ref.size == u_num.size
        ),
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 4. 图灵：四种"像斑图但不是图灵带"的预设
# --------------------------------------------------------------------------- #
@assertion("A", theory="turing 的 φ*=53.9、λ*=0.015、turing_band=1、dt 夹紧；stripes 的 λ(0)=+0.0054 但 turing_band=0；uniform 波长无效", chapter="6.1")
def test_turing_band_flags_distinguish_four_reasons() -> None:
    r"""同一个页面要同时说清四种情形：

    * ``turing``：底态稳定 + 正波数 λ*>0 → ``turing_band=1``；
    * ``stripes``：底态存在但 ``λ(0)=+0.0054>0``，最快模在 φ=0 → 均匀失稳，不是图灵带；
    * ``spots``：没有非平凡稳态（``base_valid=0``），最快模 λ<0 → 自复制瞬态；
    * ``uniform``：λ≡0，主波长**无定义**，必须 ``wavelength_valid=0`` 且存 0 占位。

    另外默认 turing 帧必须真的触发 dt 夹紧（请求 0.25，上限 0.0954）。
    """
    tu = _frame("turing_pattern", 10000.0)
    st = _frame("turing_pattern", 0.0, preset="stripes")
    sp = _frame("turing_pattern", 0.0, preset="spots")
    un = _frame("turing_pattern", 0.0, preset="uniform")
    theory_wavelength = 2.0 * math.pi / tu.meta["phi_star"]
    rel = abs(tu.meta["wavelength"] - theory_wavelength) / theory_wavelength
    rec = record(
        label="四预设的 turing_band / base_valid / λ(0) / 波长有效性",
        theory="turing：band=1、base=1、dt_clamped=1、波长差<12%；stripes：band=0、base=1、λ(0)>0；"
        "spots：band=0、base=0；uniform：band=0、wavelength_valid=0 且 wavelength=0",
        measured=(
            f"turing：φ*={tu.meta['phi_star']:.3f}，λ*={tu.meta['lambda_star']:.5f}，band={tu.meta['turing_band']:.0f}，"
            f"dt {tu.meta['dt_requested']:.2f}->{tu.meta['dt']:.5f}（limit {tu.meta['dt_limit']:.5f}），"
            f"波长 {tu.meta['wavelength']:.4f} vs {theory_wavelength:.4f}（差 {rel * 100:.1f}%）；"
            f"stripes：band={st.meta['turing_band']:.0f}，base={st.meta['base_valid']:.0f}，λ(0)={st.meta['uniform_rate']:+.5f}；"
            f"spots：band={sp.meta['turing_band']:.0f}，base={sp.meta['base_valid']:.0f}；"
            f"uniform：band={un.meta['turing_band']:.0f}，wavelength_valid={un.meta['wavelength_valid']:.0f}，"
            f"wavelength={un.meta['wavelength']:.3f}"
        ),
        tol="见 theory",
        passed=bool(
            tu.meta["turing_band"] == 1.0
            and tu.meta["base_valid"] == 1.0
            and tu.meta["dt_clamped"] == 1.0
            and abs(tu.meta["dt"] - tu.meta["dt_limit"]) < 1.0e-14
            and tu.meta["dt_requested"] == 0.25
            and rel < 0.12
            and tu.meta["spot_count"] >= 40.0
            and tu.meta["amplitude"] > 0.2
            and st.meta["turing_band"] == 0.0
            and st.meta["base_valid"] == 1.0
            and st.meta["uniform_rate"] > 0.0
            and "均匀失稳" in st.strings["turing_note"]
            and sp.meta["turing_band"] == 0.0
            and sp.meta["base_valid"] == 0.0
            and un.meta["turing_band"] == 0.0
            and un.meta["wavelength_valid"] == 0.0
            and un.meta["wavelength"] == 0.0
            and "平凡态" in un.strings["turing_note"]
            and "纯扩散" in un.strings["dispersion_note"]
        ),
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 5. 干底溃坝：窗口、CFL、质量漂移
# --------------------------------------------------------------------------- #
@assertion("A", theory="t=0 帧为 0 步且 L1=0；t=0.06 时 fan⊂全域、l1_all>l1_fan、max_cfl<=目标、质量漂移 >0 但 <1e-6，n=100→400 的扇内 L1 递减", chapter="6.1")
def test_dam_break_windows_and_cfl_are_reported_separately() -> None:
    r"""``fan`` 掩码与 ``error_valid`` 掩码必须能重建页面上两个 L1：

    * ``t=0``：``n_steps=0``、``l1_fan=l1_all=0``、``h`` 与 ``h_initial`` 逐位相等；
    * ``t=0.06``：扇区是 ``(-c_L,2c_L)``，``fan`` 非空，``l1_all>l1_fan>0``；
    * 质量漂移来自正性地板注入：非零但极小（实测 ``9.9e-9``），不能当"守恒好"的唯一证据；
    * ``n=100/400`` 的扇内 L1 从 0.0304 降到 0.0125，同时 max CFL 都 <= 0.8。
    """
    f0 = _frame("dam_break", 0.0)
    fr = _frame("dam_break", 0.06)
    f100 = _frame("dam_break", 0.06, n=100)
    f400 = _frame("dam_break", 0.06, n=400)
    fan = np.asarray(fr["fan"], dtype=float)
    valid = np.asarray(fr["error_valid"], dtype=float)
    h0 = np.asarray(f0["h"], dtype=float)
    hi = np.asarray(f0["h_initial"], dtype=float)
    rec = record(
        label="干底溃坝的零步帧、扇区 L1、质量漂移与 CFL",
        theory="t=0：n_steps=0、L1=0、h==h_initial；t=0.06：l1_all>l1_fan>0、0<mass_drift<1e-6、max_cfl<=0.8；"
        "n=100→400 扇内 L1 递减",
        measured=(
            f"t=0：n_steps={f0.meta['n_steps']:.0f}，L1={f0.meta['l1_all']:.1e}，h 逐位相等={np.array_equal(h0, hi)}；"
            f"t=0.06：fan={int(np.sum(fan))} 点，valid={int(np.sum(valid))} 点，"
            f"L1_fan={fr.meta['l1_fan']:.4e}，L1_all={fr.meta['l1_all']:.4e}，比值={fr.meta['l1_ratio']:.3f}；"
            f"质量漂移={fr.meta['mass_drift']:.2e}，max_cfl={fr.meta['max_cfl']:.3f}；"
            f"n=100/400 扇内 L1={f100.meta['l1_fan']:.4e}/{f400.meta['l1_fan']:.4e}"
        ),
        tol="见 theory",
        passed=bool(
            f0.meta["n_steps"] == 0
            and f0.meta["l1_all"] == 0.0
            and np.array_equal(h0, hi)
            and np.sum(fan) > 50
            and np.sum(valid) > 100
            and fr.meta["l1_all"] > fr.meta["l1_fan"] > 0.0
            and 0.0 < fr.meta["mass_drift"] < 1.0e-6
            and fr.meta["max_cfl"] <= 0.8 + 1.0e-12
            and fr.meta["n_steps"] > 0
            and np.all(fr["h"] >= 0.0)
            and f100.meta["l1_fan"] > f400.meta["l1_fan"]
            and f400.meta["max_cfl"] <= 0.8 + 1.0e-12
        ),
    )
    assert rec.passed, rec.measured


# --------------------------------------------------------------------------- #
# 6. 装配层只许搬运
# --------------------------------------------------------------------------- #
@assertion("B", theory="四页各抽一组数组与直接调用 numcore 逐位相等（方腔场/Ghia 基准/图灵 v/溃坝 h）", chapter="6.1")
def test_frames_match_kernel_calls_bitwise() -> None:
    r"""方腔的重算必须用同一组参数（``upwind``、tol=1e-7、max_sweeps=12000）；
    图灵必须复用同一种子；溃坝必须用同一 ``t_end`` 与边界。差应为 0，而不是 1e-12。"""
    ch61 = _entry().module

    fr_c = _frame("cavity_vorticity", re=400, n=33)
    res_c = cavity.steady_solve(
        33, 400.0, scheme="upwind",
        max_sweeps=ch61._CAVITY_MAX_SWEEPS, tol=ch61._CAVITY_TOL, record_every=1,
    )
    gr = cavity.cavity_grid(33)
    d_vort = float(np.max(np.abs(np.asarray(fr_c["vorticity"]) - res_c.omega)))
    d_psi = float(np.max(np.abs(np.asarray(fr_c["psi"]) - res_c.psi)))
    d_x = float(np.max(np.abs(np.asarray(fr_c["x"]) - gr.x)))

    fr_g = _frame("ghia_benchmark", re=400, n=33)
    cmp = cavity.compare_with_benchmark(res_c.u, res_c.v, gr, 400.0)
    d_ghia = float(np.max(np.abs(np.asarray(fr_g["u_num_ref"]) - cmp.u_num)))
    d_ghia_ref = float(np.max(np.abs(np.asarray(fr_g["u_ref"]) - cmp.u_ref)))

    fr_t = _frame("turing_pattern", 1000.0, preset="spots", n=64)
    du, dv, feed, kill = rd.RD_PRESETS["spots"]
    u0, v0 = rd.initial_condition(64, kind="spots", seed=20261019)
    res_t = rd.gray_scott_solve(
        rd.RDParams(n=64, du=du, dv=dv, feed=feed, kill=kill, dt=0.25, steps=1000),
        u0=u0, v0=v0, seed=20261019,
    )
    d_turing = float(np.max(np.abs(np.asarray(fr_t["v"]) - res_t.v)))

    fr_d = _frame("dam_break", 0.04, n=200)
    x, h0, hu0 = sw.dam_break_1d(200, right=0.0)
    res_d = sw.swe_solve_1d(
        sw.SWEParams(n=200, length=1.0, t_end=0.04, cfl=0.8, g=sw.G_DEFAULT, scheme="rusanov"),
        h0=h0, hu0=hu0, bc="outflow",
    )
    d_dam = float(np.max(np.abs(np.asarray(fr_d["h"]) - res_d.h)))
    d_dam_x = float(np.max(np.abs(np.asarray(fr_d["x"]) - x)))

    rec = record(
        label="四页数组与直接调用 numcore 的最大差",
        theory="方腔场/Ghia 基准/图灵 v/溃坝 h 的最大差都 = 0",
        measured=(
            f"方腔 ω {d_vort:.1e}、ψ {d_psi:.1e}、x {d_x:.1e}；"
            f"Ghia u_num {d_ghia:.1e}、u_ref {d_ghia_ref:.1e}；"
            f"图灵 v {d_turing:.1e}；溃坝 h {d_dam:.1e}、x {d_dam_x:.1e}"
        ),
        tol="全部 = 0",
        passed=bool(
            d_vort == 0.0 and d_psi == 0.0 and d_x == 0.0
            and d_ghia == 0.0 and d_ghia_ref == 0.0
            and d_turing == 0.0 and d_dam == 0.0 and d_dam_x == 0.0
        ),
    )
    assert rec.passed, rec.measured


@pytest.mark.parametrize("viz_id", VIZ_IDS)
def test_manifest_entry_contract(viz_id: str) -> None:
    """每个 viz 必须有唯一 id、非空 pitfalls、数据源 backend，且参数默认值合法。"""
    spec = _viz(viz_id)
    assert spec.data_source == "backend"
    assert len(spec.pitfalls) >= 200, f"{viz_id} 的 pitfalls 太短：{len(spec.pitfalls)}"
    names = [p.name for p in spec.params]
    assert len(names) == len(set(names))
    assert spec.axes[0].id == "main"
    assert len({axis.id for axis in spec.axes}) == len(spec.axes)
    assert len(spec.required_aux) >= 1
    # 参数默认值必须落在声明的区间/选项里
    for p in spec.params:
        if p.type == "enum":
            assert p.default in p.options
        elif p.type in ("int", "float"):
            assert p.min <= p.default <= p.max
