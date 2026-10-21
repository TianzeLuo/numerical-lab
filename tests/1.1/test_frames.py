"""1.1 帧层回归：动态范围里的“必然坏点”必须被显式表示，而不是 500 / NaN。

用户实际踩到的三类：
* 灾难性抵消：损失位数 t 接近 16 时，目标差可能小于一个 ulp，`a - gap` 四舍五入回
  `a`，相对误差变成 0/0；整帧因数组含 Inf 被拒收。
* 条件数椭球：graded 族在 κ≈1e16 时矩阵真的变成 [[1,1],[1,1]]，条件数函数抛错。
* 这两者都应该在**默认与极端参数**下都能返回一帧合法数据（占位值 + 掩码 + 说明）。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _module() -> Any:
    spec = importlib.util.spec_from_file_location("chapter_1_1_frames", ROOT / "server" / "chapters" / "1.1.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _default_params(mod: Any, viz: str) -> dict[str, Any]:
    spec = next(v for v in mod.MANIFEST["vizzes"] if v["id"] == viz)
    return {p["name"]: p["default"] for p in spec["params"]}


@pytest.mark.parametrize("a", [0.5, 1.0, 1.000001, 1.05, 1.1, 1.2, 2.0])
def test_cancellation_grid_is_finite_for_all_loss_digits(a: float) -> None:
    mod = _module()
    params = _default_params(mod, "cancellation")
    params["a"] = a
    frame = mod.compute("cancellation", params, 0.0)
    frame.validate()
    for name in ("error", "error_theory", "roundoff", "digits_remaining"):
        assert np.all(np.isfinite(frame[name])), f"{name} 在 a={a} 时出现非有限值"
    assert frame["error"].size == 33
    # 末端被夹到“最小可表示差”时，必须自报家门。
    if frame.meta.get("grid_clamped_steps", 0) > 0:
        assert "最小" in frame.strings["note"]


@pytest.mark.parametrize("family", ["shear", "diagonal", "rotation", "graded", "triangular"])
@pytest.mark.parametrize("t", [0.0, 8.0, 16.0])
def test_cond_ellipsoid_never_500s(family: str, t: float) -> None:
    mod = _module()
    params = _default_params(mod, "cond_ellipsoid")
    params["family"] = family
    frame = mod.compute("cond_ellipsoid", params, t)
    frame.validate()
    assert frame["matrix_A"].size == 4
    for name in ("x_exact", "x_num", "x_pert"):
        assert np.all(np.isfinite(frame[name])), f"{name} 必须用占位值而不是 NaN"
        assert float(frame[f"{name}_defined"][0]) in (0.0, 1.0)
    if float(frame["x_num_defined"][0]) == 0.0:
        assert "无有限解" in frame.strings["note"] or "奇异" in frame.strings["note"]


def test_cond_graded_at_kappa_limit_is_explicit_not_silent() -> None:
    mod = _module()
    params = _default_params(mod, "cond_ellipsoid")
    params["family"] = "graded"
    frame = mod.compute("cond_ellipsoid", params, 16.0)
    frame.validate()
    assert frame.meta["kappa_measured"] is None, "奇异矩阵的条件数必须写成 null，而不是 inf/nan"
    assert float(frame["x_num_defined"][0]) == 0.0
    assert "奇异" in frame.strings["note"]
