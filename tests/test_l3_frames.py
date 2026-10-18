"""帧装配层的测试。

这里守的是"前端要什么、后端必须给什么"那几件事：
参数越界要报错而不是静默夹取；声明了 error/residual 就必须真有这个数组；
临界值、pitfalls、数据来源徽标必须随帧走；缓存有界；静态预计算可复现。
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from server.frames import (
    FrameCache,
    ParamError,
    build_frame,
    governing_critical,
    normalize_params,
    params_hash,
    precompute_chapter,
    static_frame_path,
)
from server.manifest import ParamSpec, load_manifest
from server.nlf import Frame, NlfError, read_frame
from server.registry import ChapterEntry, Registry

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture()
def entry() -> ChapterEntry:
    reg = Registry(ROOT)
    reg.discover()
    return reg.get("0.0")


def spec(**kw) -> ParamSpec:
    base = dict(name="x", label="x", type="float", default=0.5, critical=())
    base.update(kw)
    return ParamSpec(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# 参数规范化
# --------------------------------------------------------------------------- #
def test_defaults_fill_in(entry: ChapterEntry) -> None:
    viz = entry.manifest.viz("oscillator_regimes")
    got = normalize_params(viz, {})
    assert got == {"omega": 1.0, "zeta": 0.05, "n_periods": 3}
    assert normalize_params(viz, None) == got


def test_partial_params_keep_defaults(entry: ChapterEntry) -> None:
    viz = entry.manifest.viz("heat_ftcs")
    got = normalize_params(viz, {"nx": 51})
    assert got == {"nx": 51, "r": 0.4, "t_end": 0.05}
    assert isinstance(got["nx"], int)


def test_unknown_param_lists_available(entry: ChapterEntry) -> None:
    viz = entry.manifest.viz("heat_ftcs")
    with pytest.raises(ParamError) as ei:
        normalize_params(viz, {"nope": 1})
    msg = str(ei.value)
    assert "nope" in msg and "nx" in msg and "r" in msg


def test_type_and_range_errors(entry: ChapterEntry) -> None:
    viz = entry.manifest.viz("oscillator_regimes")
    with pytest.raises(ParamError, match="整数"):
        normalize_params(viz, {"n_periods": 2.5})
    with pytest.raises(ParamError, match="数字"):
        normalize_params(viz, {"zeta": "0.1"})
    with pytest.raises(ParamError, match="有限数"):
        normalize_params(viz, {"zeta": float("nan")})
    with pytest.raises(ParamError, match="有限数"):
        normalize_params(viz, {"zeta": float("inf")})
    with pytest.raises(ParamError, match="下界"):
        normalize_params(viz, {"zeta": -0.5})
    with pytest.raises(ParamError, match="上界"):
        normalize_params(viz, {"zeta": 3.0})
    with pytest.raises(ParamError, match="整数"):
        normalize_params(viz, {"n_periods": True})  # bool 不是 int


def test_enum_and_bool_paths() -> None:
    man = load_manifest(
        {
            "chapter": "9.9",
            "title": "t",
            "vizzes": [
                {
                    "id": "demo",
                    "title": "d",
                    "dataSource": "backend",
                    "params": [
                        {"name": "mode", "label": "m", "type": "enum", "options": ["a", "b"], "default": "a", "critical": []},
                        {"name": "flag", "label": "f", "type": "bool", "default": False, "critical": []},
                    ],
                    "time": {"mode": "none"},
                    "requiredAux": ["error"],
                    "axes": [{"id": "main", "autoscaled": False}],
                    "pitfalls": "自适应纵轴会把发散画成一条平线，必须固定纵轴；对数误差轴又会让 1e-16 和 1e-3 看着一样深，两件事必须同时说清，否则读者会把「稳定」读成「精确」。",
                    "theory": ["t"],
                }
            ],
        }
    )
    viz = man.viz("demo")
    assert normalize_params(viz, {}) == {"mode": "a", "flag": False}
    assert normalize_params(viz, {"mode": "b", "flag": True}) == {"mode": "b", "flag": True}
    with pytest.raises(ParamError, match="之一"):
        normalize_params(viz, {"mode": "c"})
    with pytest.raises(ParamError, match="布尔"):
        normalize_params(viz, {"flag": 1})


def test_endpoints_are_inclusive(entry: ChapterEntry) -> None:
    """区间是闭区间：正好等于上下界必须通过，越出一丁点必须报错。"""
    viz = entry.manifest.viz("heat_ftcs")
    assert normalize_params(viz, {"r": 0.7})["r"] == 0.7
    assert normalize_params(viz, {"r": 0.05})["r"] == 0.05
    assert normalize_params(viz, {"nx": 11})["nx"] == 11
    assert normalize_params(viz, {"nx": 201})["nx"] == 201
    with pytest.raises(ParamError, match="上界"):
        normalize_params(viz, {"r": 0.70001})
    with pytest.raises(ParamError, match="上界"):
        normalize_params(viz, {"nx": 202})


# --------------------------------------------------------------------------- #
# 指纹
# --------------------------------------------------------------------------- #
def test_params_hash_is_order_independent(entry: ChapterEntry) -> None:
    a = params_hash("0.0", "heat_ftcs", {"nx": 41, "r": 0.4, "t_end": 0.05})
    b = params_hash("0.0", "heat_ftcs", {"t_end": 0.05, "r": 0.4, "nx": 41})
    assert a == b and len(a) == 16


def test_params_hash_separates_everything(entry: ChapterEntry) -> None:
    base = {"nx": 41, "r": 0.4}
    assert params_hash("0.0", "heat_ftcs", base) != params_hash("0.0", "other", base)
    assert params_hash("0.0", "heat_ftcs", base) != params_hash("1.1", "heat_ftcs", base)
    assert params_hash("0.0", "heat_ftcs", base) != params_hash("0.0", "heat_ftcs", {"nx": 42, "r": 0.4})


def test_params_hash_handles_nonascii_and_unicode_keys() -> None:
    h = params_hash("0.0", "v", {"模式": "严格"})
    assert h == params_hash("0.0", "v", {"模式": "严格"})
    assert h.isalnum()


# --------------------------------------------------------------------------- #
# 支配临界值
# --------------------------------------------------------------------------- #
def test_governing_critical_directions() -> None:
    above = spec(critical=(0.5, 1.0), beyond="above")
    assert governing_critical(above) == 0.5
    assert governing_critical(above, 0.2) == 0.5
    assert governing_critical(above, 0.7) == 0.5  # 只越过 0.5 → 报最近越过的那条
    assert governing_critical(above, 1.5) == 1.0
    assert governing_critical(above, 2.0) == 1.0
    below = spec(critical=(0.1, 0.01), beyond="below")
    assert governing_critical(below) == 0.1
    assert governing_critical(below, 0.05) == 0.1  # 只越过 0.1
    assert governing_critical(below, 0.005) == 0.01  # 两条都越过 → 报最近越过的
    mag = spec(critical=(-2.0, 1.0), beyond="magnitude")
    assert governing_critical(mag, 0.9) == 1.0
    assert governing_critical(mag, -1.9) == -2.0
    assert governing_critical(spec(critical=())) is None


# --------------------------------------------------------------------------- #
# 出帧
# --------------------------------------------------------------------------- #
def test_build_frame_sets_everything_the_frontend_needs(entry: ChapterEntry) -> None:
    fr = build_frame(entry, "heat_ftcs", {"nx": 31, "r": 0.45}, 0.03)
    assert fr.chapter == "0.0" and fr.viz == "heat_ftcs"
    assert fr.meta["t"] == pytest.approx(0.03)
    assert fr.pitfalls and len(fr.pitfalls) >= 60
    assert fr.critical["r"] == pytest.approx(0.45) or fr.critical["r"] == 0.5
    assert fr.critical["r"] == pytest.approx(0.5)  # 未越过 → 给最小阈值
    assert fr.strings["data_source"] == "backend"
    assert json.loads(fr.strings["params_json"]) == {"nx": 31, "r": 0.45, "t_end": 0.05}
    assert set(fr.names) == {"x", "u", "exact", "error"}


def test_build_frame_critical_follows_param_value(entry: ChapterEntry) -> None:
    crossed = build_frame(entry, "heat_ftcs", {"r": 0.6}, 0.0)
    assert crossed.critical["r"] == pytest.approx(0.5)
    assert crossed.strings["stability"] == "unstable"
    stable = build_frame(entry, "heat_ftcs", {"r": 0.4}, 0.0)
    assert stable.strings["stability"] == "stable"


def test_build_frame_detects_missing_required_aux(entry: ChapterEntry) -> None:
    """声明了 requiredAux 却没给对应数组 → 必须当场炸，不能画一张空图。"""
    man = load_manifest(
        {
            "chapter": "0.0",
            "title": "t",
            "vizzes": [
                {
                    "id": "liar",
                    "title": "d",
                    "dataSource": "backend",
                    "params": [{"name": "n", "label": "n", "type": "int", "default": 3, "critical": []}],
                    "time": {"mode": "none"},
                    "requiredAux": ["error", "residual"],
                    "axes": [{"id": "main", "autoscaled": False}],
                    "pitfalls": "自适应纵轴会把发散画成一条平线，必须固定纵轴；对数误差轴又会让 1e-16 和 1e-3 看着一样深，两件事必须同时说清，否则读者会把「稳定」读成「精确」。",
                    "theory": ["t"],
                }
            ],
        }
    )
    fake = ChapterEntry(id="0.0", path=entry.path, manifest=man, module=None)

    def compute(viz_id: str, params: dict, t: float) -> Frame:
        fr = Frame(chapter="0.0", viz=viz_id, kind="frame")
        fr.add("x", np.zeros(4))
        fr.set_meta(n=params["n"])
        return fr

    fake.module = type("M", (), {"compute": staticmethod(compute)})()
    with pytest.raises(NlfError, match="规则 3"):
        build_frame(fake, "liar", {}, 0.0)

    def compute_ok(viz_id: str, params: dict, t: float) -> Frame:
        fr = Frame(chapter="0.0", viz=viz_id, kind="frame")
        fr.add("x", np.zeros(4))
        fr.add("residual", np.zeros(4))
        fr.add("residual_error", np.zeros(4))  # 名字里含 error 也算数（大小写不敏感）
        return fr

    fake.module = type("M", (), {"compute": staticmethod(compute_ok)})()
    assert build_frame(fake, "liar", {}, 0.0).names == ["x", "residual", "residual_error"]


def test_build_frame_rejects_wrong_return_type(entry: ChapterEntry) -> None:
    fake = ChapterEntry(
        id="0.0", path=entry.path, manifest=entry.manifest, module=type("M", (), {"compute": staticmethod(lambda *a: {"x": 1})})()
    )
    with pytest.raises(NlfError, match="必须返回 Frame"):
        build_frame(fake, "heat_ftcs", {}, 0.0)


def test_build_frame_rejects_container_meta(entry: ChapterEntry) -> None:
    def compute(viz_id: str, params: dict, t: float) -> Frame:
        fr = Frame(chapter="0.0", viz=viz_id, kind="frame")
        fr.add("error", np.zeros(4))
        fr.meta["bag"] = {"a": 1}  # 绕过 set_meta 直接塞容器
        return fr

    fake = ChapterEntry(id="0.0", path=entry.path, manifest=entry.manifest, module=type("M", (), {"compute": staticmethod(compute)})())
    with pytest.raises(NlfError, match="只允许标量"):
        build_frame(fake, "heat_ftcs", {}, 0.0)


def test_build_frame_strict_false_skips_audit(entry: ChapterEntry) -> None:
    def compute(viz_id: str, params: dict, t: float) -> Frame:
        fr = Frame(chapter="0.0", viz=viz_id, kind="frame")
        fr.add("x", np.zeros(4))
        return fr

    fake = ChapterEntry(id="0.0", path=entry.path, manifest=entry.manifest, module=type("M", (), {"compute": staticmethod(compute)})())
    fr = build_frame(fake, "heat_ftcs", {}, 0.0, strict=False)
    assert fr.names == ["x"]


def test_build_frame_is_deterministic(entry: ChapterEntry) -> None:
    p = {"nx": 41, "r": 0.4, "t_end": 0.05}
    a = build_frame(entry, "heat_ftcs", p, 0.02)
    b = build_frame(entry, "heat_ftcs", p, 0.02)
    assert np.array_equal(a["u"], b["u"])
    assert a.meta == b.meta


def test_oscillator_frame_has_residual_and_regime_note(entry: ChapterEntry) -> None:
    fr = build_frame(entry, "oscillator_regimes", {"zeta": 1.0}, 0.0)
    # 字符串一律在 strings 里：meta 只允许 number/bool/null（PROTOCOLS §1.1）
    assert fr.strings["regime"] == "critical"
    assert "临界阻尼" in fr.strings["regime_note"]
    for key, val in fr.meta.items():
        assert not isinstance(val, str), key
    assert 0.0 < float(np.max(fr["residual"])) < 1e-4  # 中心差分截断误差的量级
    assert fr.meta["residual_max"] == pytest.approx(float(np.max(fr["residual"])))
    assert fr.critical["zeta"] == pytest.approx(1.0)
    assert fr["energy"].shape == fr["t"].shape


# --------------------------------------------------------------------------- #
# 缓存
# --------------------------------------------------------------------------- #
def test_frame_cache_lru_by_entries() -> None:
    c = FrameCache(max_entries=2, max_bytes=10**6)
    c.put("a", b"1")
    c.put("b", b"2")
    assert c.get("a") == b"1"
    c.put("c", b"3")  # 触发淘汰：最久未用的 "a" 被淘汰（get 不会刷新顺序）
    assert c.n_entries == 2 and c.n_bytes == 2
    c.clear()
    assert c.n_entries == 0 and c.n_bytes == 0 and c.get("a") is None


def test_frame_cache_respects_byte_budget() -> None:
    c = FrameCache(max_entries=100, max_bytes=10)
    c.put("a", b"x" * 6)
    c.put("b", b"y" * 6)
    assert c.n_bytes <= 10
    assert c.get("a") is None and c.get("b") is not None
    c.put("huge", b"z" * 100)
    assert c.get("huge") is None  # 单帧超预算：不缓存，而不是把缓存清空
    assert c.get("b") == b"y" * 6


def test_frame_cache_key_includes_params_and_time() -> None:
    c = FrameCache()
    k1 = c.key("0.0", "heat_ftcs", {"nx": 41}, 0.0)
    k2 = c.key("0.0", "heat_ftcs", {"nx": 41}, 0.1)
    k3 = c.key("0.0", "heat_ftcs", {"nx": 42}, 0.0)
    assert len({k1, k2, k3}) == 3
    assert c.key("0.0", "heat_ftcs", {"nx": 41}, 0.0) == k1


# --------------------------------------------------------------------------- #
# 静态预计算
# --------------------------------------------------------------------------- #
def test_static_frame_path_layout() -> None:
    p = static_frame_path(ROOT, "0.0", "heat_ftcs", {"nx": 41, "r": 0.4}, 0.05)
    assert p.parent == ROOT / "data" / "frames" / "0.0" / "heat_ftcs"
    assert p.name.endswith("_t0.05.nlf")
    assert p == static_frame_path(ROOT, "0.0", "heat_ftcs", {"r": 0.4, "nx": 41}, 0.05)


def test_precompute_writes_readable_frames(tmp_path: pathlib.Path) -> None:
    files = precompute_chapter(Registry(ROOT), "0.0", out_root=tmp_path, frames_per_viz=3)
    assert len(files) == 6  # oscillator(stream) 3 帧 + heat_ftcs(scrub) 3 帧
    assert {p.parent.name for p in files} == {"oscillator_regimes", "heat_ftcs"}
    assert all(p.exists() for p in files)
    times = sorted(float(read_frame(p).meta["t"]) for p in files if p.parent.name == "heat_ftcs")
    assert times[0] == 0.0 and times[-1] == pytest.approx(0.2)
    for p in files:
        fr = read_frame(p)
        assert fr.chapter == "0.0"
        assert fr.pitfalls and fr.strings["params_json"]
    # 第二次跑必须字节一致（静态预计算要能被 git 之外的地方重现）
    first = {p: p.read_bytes() for p in files}
    again = precompute_chapter(Registry(ROOT), "0.0", out_root=tmp_path, frames_per_viz=3)
    assert [first[p] for p in again] == [p.read_bytes() for p in again]


def test_precompute_limit_and_single_frame_mode(tmp_path: pathlib.Path) -> None:
    files = precompute_chapter(Registry(ROOT), "0.0", out_root=tmp_path, frames_per_viz=5, limit=2)
    assert len(files) == 2
    names = {p.parent.name for p in files}
    assert names == {"oscillator_regimes"}  # 只跑到第一个可视化就被 limit 截断


def test_precompute_uses_registry_root_by_default(tmp_path: pathlib.Path) -> None:
    reg = Registry(ROOT)
    path = static_frame_path(reg.root, "0.0", "heat_ftcs", {"nx": 11, "r": 0.4, "t_end": 0.01}, 0.0)
    assert path.is_relative_to(ROOT / "data" / "frames")
