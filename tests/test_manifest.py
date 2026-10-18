"""Manifest 校验器的测试。

重点不是"合法的能过"，而是**每一条防误导规则都真的会拦人**：
少写 critical、criticalNote、autoscaled、requiredAux 里的 error/residual、pitfalls 太短……
每一种都要能被抓住，并且错误信息里说清是哪条规则。
"""

from __future__ import annotations

import copy
import json

import pytest

from server.manifest import (
    MIN_PITFALL_CHARS,
    ChapterManifest,
    ManifestError,
    VizSpec,
    load_manifest,
    validate_manifest,
)

LONG_PITFALL = (
    "自适应纵轴会把发散画成一条平线，必须固定纵轴；对数色标让 1e-16 与 1e-3 看着一样深，"
    "必须同时读数量级；曲线在 NaN 处若被直线连起来，会凭空造出一段根本不存在的解。"
)


def good_viz(viz_id: str = "demo") -> dict:
    return {
        "id": viz_id,
        "title": "演示",
        "dataSource": "backend",
        "params": [
            {"name": "n", "label": "阶数", "type": "int", "min": 4, "max": 64, "default": 8, "critical": [], "criticalNote": ""},
            {
                "name": "tol",
                "label": "容差",
                "type": "float",
                "min": 1e-12,
                "max": 1.0,
                "default": 1e-6,
                "critical": [0.5],
                "criticalNote": "超过 0.5 迭代不再收敛",
            },
            {"name": "mode", "label": "策略", "type": "enum", "options": ["a", "b"], "default": "a", "critical": []},
        ],
        "time": {"mode": "none"},
        "requiredAux": ["error"],
        "axes": [{"id": "main", "autoscaled": False}, {"id": "error", "autoscaled": True}],
        "pitfalls": LONG_PITFALL,
        "theory": ["后向误差界 ‖ΔA‖ ≤ ρ·u·‖A‖"],
    }


def good_manifest() -> dict:
    return {"chapter": "9.9", "title": "测试章", "vizzes": [good_viz()]}


def problems_of(m: dict) -> list[str]:
    return validate_manifest(m)


def test_good_manifest_passes_and_loads() -> None:
    raw = good_manifest()
    assert problems_of(raw) == []
    man = load_manifest(raw)
    assert isinstance(man, ChapterManifest)
    assert man.chapter == "9.9" and man.viz_ids == ("demo",)
    viz = man.viz("demo")
    assert isinstance(viz, VizSpec)
    assert viz.time.mode == "none" and viz.time.n_frames == 1
    assert viz.param("tol").critical == (0.5,)
    assert viz.defaults() == {"n": 8, "tol": 1e-6, "mode": "a"}
    assert viz.axes[1].autoscaled is True
    assert viz.required_aux == ("error",)


def test_chapter_id_and_title_are_checked() -> None:
    m = good_manifest()
    m["chapter"] = "1"
    assert any("chapter" in p for p in problems_of(m))
    m["chapter"] = "1.2.3"
    assert any("chapter" in p for p in problems_of(m))
    m = good_manifest()
    m["title"] = "   "
    assert any("title" in p for p in problems_of(m))


def test_vizzes_must_be_nonempty() -> None:
    m = good_manifest()
    m["vizzes"] = []
    assert any("vizzes" in p for p in problems_of(m))
    assert validate_manifest({"chapter": "1.1", "title": "t"}) != []


def test_duplicate_and_bad_viz_ids() -> None:
    m = good_manifest()
    m["vizzes"].append(copy.deepcopy(m["vizzes"][0]))
    assert any("唯一" in p for p in problems_of(m))
    m = good_manifest()
    m["vizzes"][0]["id"] = "Bad-Id"
    assert any("id 必须是" in p for p in problems_of(m))


def test_data_source_must_be_known() -> None:
    m = good_manifest()
    m["vizzes"][0]["dataSource"] = "frontend"
    assert any("dataSource" in p for p in problems_of(m))
    m["vizzes"][0]["dataSource"] = "backend+preview"
    assert problems_of(m) == []


# --------------------------------------------------------------------------- #
# 规则 6：临界值
# --------------------------------------------------------------------------- #
def test_critical_must_be_present_even_if_empty() -> None:
    m = good_manifest()
    del m["vizzes"][0]["params"][0]["critical"]
    got = problems_of(m)
    assert any("缺少 critical" in p and "规则 6" in p for p in got)


def test_critical_note_required_when_critical_nonempty() -> None:
    m = good_manifest()
    m["vizzes"][0]["params"][1]["criticalNote"] = "  "
    assert any("criticalNote" in p for p in problems_of(m))
    m["vizzes"][0]["params"][1]["criticalNote"] = "超过 0.5 迭代不再收敛"
    assert problems_of(m) == []


def test_critical_elements_must_be_numbers() -> None:
    m = good_manifest()
    m["vizzes"][0]["params"][1]["critical"] = ["半"]
    assert any("critical 的元素" in p for p in problems_of(m))
    m["vizzes"][0]["params"][1]["critical"] = "0.5"
    assert any("critical 必须是列表" in p for p in problems_of(m))


def test_beyond_direction_is_validated() -> None:
    m = good_manifest()
    m["vizzes"][0]["params"][1]["beyond"] = "sideways"
    assert any("beyond" in p for p in problems_of(m))


# --------------------------------------------------------------------------- #
# 参数
# --------------------------------------------------------------------------- #
def test_enum_requires_options_and_matching_default() -> None:
    m = good_manifest()
    m["vizzes"][0]["params"][2]["options"] = ["a"]
    assert any("两个 options" in p for p in problems_of(m))
    m = good_manifest()
    m["vizzes"][0]["params"][2]["default"] = "c"
    assert any("不在 options" in p for p in problems_of(m))


@pytest.mark.parametrize(
    "field,value,frag",
    [
        ("min", 100, "default"),
        ("max", 1, "default"),
        ("max", 4, "min < max"),
    ],
)
def test_numeric_bounds_are_consistent(field: str, value: float, frag: str) -> None:
    m = good_manifest()
    m["vizzes"][0]["params"][0][field] = value
    assert any(frag in p for p in problems_of(m)), problems_of(m)


def test_int_param_requires_int_default() -> None:
    m = good_manifest()
    m["vizzes"][0]["params"][0]["default"] = 8.5
    assert any("整数" in p for p in problems_of(m))


def test_bool_param_requires_bool_default() -> None:
    m = good_manifest()
    m["vizzes"][0]["params"].append({"name": "flag", "label": "开关", "type": "bool", "default": 1, "critical": []})
    assert any("true/false" in p for p in problems_of(m))


def test_unknown_param_type_and_missing_label() -> None:
    m = good_manifest()
    m["vizzes"][0]["params"][0]["type"] = "complex"
    assert any("type 必须是" in p for p in problems_of(m))
    m = good_manifest()
    del m["vizzes"][0]["params"][0]["label"]
    assert any("label" in p for p in problems_of(m))


def test_duplicate_and_empty_param_names() -> None:
    m = good_manifest()
    m["vizzes"][0]["params"].append(copy.deepcopy(m["vizzes"][0]["params"][0]))
    assert any("参数名重复" in p for p in problems_of(m))
    m = good_manifest()
    m["vizzes"][0]["params"][0]["name"] = ""
    assert any("name 必须" in p for p in problems_of(m))


def test_params_must_be_nonempty_list() -> None:
    m = good_manifest()
    m["vizzes"][0]["params"] = []
    assert any("params" in p for p in problems_of(m))


# --------------------------------------------------------------------------- #
# 时间轴（规则 4）
# --------------------------------------------------------------------------- #
def test_time_none_forbids_time_controls() -> None:
    m = good_manifest()
    m["vizzes"][0]["time"] = {"mode": "none", "tMax": 1.0}
    assert any("不得声明时间控件" in p for p in problems_of(m))


def test_stream_mode_requires_positive_tmax_and_dt() -> None:
    m = good_manifest()
    m["vizzes"][0]["time"] = {"mode": "stream", "tMax": 0, "dt": 0.1}
    assert any("tMax" in p for p in problems_of(m))
    m = good_manifest()
    m["vizzes"][0]["time"] = {"mode": "stream", "tMax": 1.0, "dt": 0}
    assert any("dt" in p for p in problems_of(m))
    m = good_manifest()
    m["vizzes"][0]["time"] = {"mode": "stream", "tMax": 1.0, "dt": 0.01, "fps": 1000}
    assert any("fps" in p for p in problems_of(m))
    m = good_manifest()
    m["vizzes"][0]["time"] = {"mode": "scrub", "tMax": 1.0, "dt": 2.0}
    assert any("dt 不应大于 tMax" in p for p in problems_of(m))


def test_missing_time_field() -> None:
    m = good_manifest()
    del m["vizzes"][0]["time"]
    assert any("缺少 time" in p for p in problems_of(m))


def test_time_n_frames() -> None:
    m = good_manifest()
    m["vizzes"][0]["time"] = {"mode": "scrub", "tMax": 1.0, "dt": 0.1}
    man = load_manifest(m)
    assert man.viz("demo").time.n_frames == 11


# --------------------------------------------------------------------------- #
# 规则 1/3/7
# --------------------------------------------------------------------------- #
def test_required_aux_must_contain_error_or_residual() -> None:
    m = good_manifest()
    m["vizzes"][0]["requiredAux"] = ["mesh"]
    assert any("规则 3" in p for p in problems_of(m))
    m["vizzes"][0]["requiredAux"] = ["residual"]
    assert problems_of(m) == []


def test_axes_must_declare_autoscaled() -> None:
    m = good_manifest()
    del m["vizzes"][0]["axes"][0]["autoscaled"]
    assert any("autoscaled" in p and "规则 1/4" in p for p in problems_of(m))
    m = good_manifest()
    m["vizzes"][0]["axes"][0]["autoscaled"] = "no"
    assert any("autoscaled" in p for p in problems_of(m))


def test_axes_ids_unique_and_scale_limited() -> None:
    m = good_manifest()
    m["vizzes"][0]["axes"].append({"id": "main", "autoscaled": True})
    assert any("id 重复" in p for p in problems_of(m))
    m = good_manifest()
    m["vizzes"][0]["axes"][0]["scale"] = "sqrt"
    assert any("scale" in p for p in problems_of(m))


def test_axes_must_be_nonempty() -> None:
    m = good_manifest()
    m["vizzes"][0]["axes"] = []
    assert any("axes 必须非空" in p for p in problems_of(m))


def test_pitfalls_minimum_length() -> None:
    m = good_manifest()
    m["vizzes"][0]["pitfalls"] = "很短"
    got = problems_of(m)
    assert any("pitfalls" in p and "规则 7" in p for p in got)
    m["vizzes"][0]["pitfalls"] = "恰" * (MIN_PITFALL_CHARS - 1)
    assert any("pitfalls" in p for p in problems_of(m))
    m["vizzes"][0]["pitfalls"] = "恰" * MIN_PITFALL_CHARS
    assert problems_of(m) == []


def test_theory_must_be_nonempty() -> None:
    m = good_manifest()
    m["vizzes"][0]["theory"] = []
    assert any("theory" in p for p in problems_of(m))


# --------------------------------------------------------------------------- #
# 其它
# --------------------------------------------------------------------------- #
def test_manifest_error_lists_all_problems_and_where() -> None:
    m = good_manifest()
    m["vizzes"][0]["pitfalls"] = "短"
    del m["vizzes"][0]["axes"][0]["autoscaled"]
    with pytest.raises(ManifestError) as ei:
        load_manifest(m, where="章节 9.9")
    assert len(ei.value.problems) >= 2
    assert "章节 9.9" in str(ei.value)
    assert "pitfalls" in str(ei.value)


def test_non_mapping_input() -> None:
    assert validate_manifest([]) == ["manifest 必须是 dict"]  # type: ignore[arg-type]


def test_to_json_is_json_serializable_and_roundtrips() -> None:
    man = load_manifest(good_manifest())
    text = json.dumps(man.to_json(), ensure_ascii=False)
    back = json.loads(text)
    assert back["chapter"] == "9.9"
    assert back["vizzes"][0]["params"][1]["critical"] == [0.5]
    assert back["vizzes"][0]["axes"][0]["autoscaled"] is False
    # 生成的 JSON 必须能再次通过校验（否则 G6 的"生成物 == 真源"就没意义）
    assert validate_manifest(back) == []


def test_to_json_omits_optional_empty_fields() -> None:
    man = load_manifest(good_manifest())
    params = man.to_json()["vizzes"][0]["params"]
    assert "options" not in params[0]
    assert "unit" not in params[0]
    assert "beyond" not in params[0]  # 缺省 above 不写出来
