"""``tools/raster.py`` / ``tools/visual_regression.py`` / ``tools/export_video.py`` 的测试。

覆盖口径（对应交付要求）
------------------------
* 8 种图层各至少一次，含退化输入（空数据 / 单点 / 全 NaN）；
* 热力图 NaN → 中性灰 ``(130,130,130,255)`` **不是透明**；折线 NaN → 断线；
* 对数轴、色标（含对数色标）、徽标（source/autoscale/critical/overflow）、pitfalls 面板；
* 确定性：同输入两次 → 同一 bytes；
* 帧 → 标准场景的稳定性（``default_scene_for_frame``）；
* **跨语言刻度对拍**：用 ``node`` 从已编译的 ``dist/viz/shared/src/axes.js`` 取
  ``niceLimits/niceTicks/formatNumber/tickLabel`` 的真实输出，与 Python 侧逐值比对
  （跨 6 个数量级、极窄区间、负区间、对数区间）；
* ``visual_regression --check/--update`` 的端到端（tmp 目录 + 0.0 章）。

测试**不写仓库里的任何产物**：所有落盘都进 ``tmp_path``。
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import shutil
import struct
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from tools import export_video as ev
from tools import visual_regression as vr
from tools.gif import decode_gif_frames, read_gif
from tools.png import read_png, write_rgb
from tools.raster import (
    RasterError,
    apply_cmap,
    badge_box_width,
    build_heat_rgba,
    check_overflow,
    data_to_pixel,
    default_scene_for_frame,
    format_number,
    format_sci,
    layer_to_pixel_path,
    lookup_cmap,
    marching_squares,
    nice_limits,
    nice_ticks,
    normalize_value,
    render_scene,
    render_scene_to_bytes,
    render_scene_to_file,
    tick_label,
    validate_scene,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
BG = (255, 255, 255)
GRAY_NAN = (130, 130, 130)

# --------------------------------------------------------------------------- #
# 夹具
# --------------------------------------------------------------------------- #
PITFALL = "固定纵轴时它看起来在收敛；请同时看副轴的残差量级，别把「看起来」当成「收敛」。"


def make_axis(**over) -> dict:
    axis = {
        "id": "main",
        "rect": [60.0, 60.0, 400.0, 240.0],
        "xlim": [0.0, 1.0],
        "ylim": [0.0, 1.0],
        "xscale": "linear",
        "yscale": "linear",
        "xlabel": "x",
        "ylabel": "y",
        "title": "",
        "autoscaled": False,
        "xticks": [0.0, 0.5, 1.0],
        "yticks": [0.0, 0.5, 1.0],
        "grid": True,
        "frame": True,
    }
    axis.update(over)
    return axis


def make_scene(layers, *, axes=None, texts=(), badges=None, pitfalls=PITFALL, overflow=None,
               canvas=(480.0, 360.0), title="回归探针") -> dict:
    if badges is None:
        badges = [{"kind": "source", "text": "数据：后端 .nlf 帧", "x": 8.0, "y": 8.0}]
    return {
        "version": 1,
        "chapter": "0.0",
        "viz": "probe",
        "title": title,
        "t": 0.0,
        "canvas": {"w": canvas[0], "h": canvas[1], "background": "#ffffff"},
        "axes": list(axes) if axes is not None else [make_axis()],
        "layers": list(layers),
        "texts": list(texts),
        "badges": list(badges),
        "overflow": overflow or {"flag": False, "note": ""},
        "pitfalls": pitfalls,
        "dataSource": "backend",
    }


def line_layer(x, y, **over) -> dict:
    series = {"x": list(x), "y": list(y), "color": "#1f77b4", "width": 1.6}
    series.update(over)
    return {"kind": "line", "axes": "main", "series": [series]}


def heat_layer(values, nx, ny, **over) -> dict:
    layer = {
        "kind": "heat", "axes": "main", "nx": nx, "ny": ny, "values": list(values),
        "extent": [0.0, 1.0, 0.0, 1.0], "cmap": "gray", "vmin": 0.0, "vmax": 1.0, "logScale": False,
    }
    layer.update(over)
    return layer


def pixels(scene, **kw) -> np.ndarray:
    return render_scene(scene, **kw).array()


GRID_RGB = np.array([232, 232, 232], dtype=np.int16)


def ink(arr: np.ndarray, box: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """"有内容"的像素掩码：既不是背景白、也不是网格线灰（box = (x0, y0, x1, y1)）。"""
    rgb = arr[:, :, :3].astype(np.int16)
    bg = np.array(BG, dtype=np.int16)
    mask = np.any(np.abs(rgb - bg) > 0, axis=2) & np.any(np.abs(rgb - GRID_RGB) > 0, axis=2)
    if box is None:
        return mask
    x0, y0, x1, y1 = box
    out = np.zeros_like(mask)
    out[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    return out


# =========================================================================== #
# 1. 渲染入口
# =========================================================================== #
def test_render_scene_returns_rgba_png_image():
    img = render_scene(make_scene([line_layer([0, 1], [0, 1])]))
    assert (img.width, img.height, img.channels) == (480, 360, 4)
    assert len(img.pixels) == 480 * 360 * 4


def test_render_scene_scale_multiplies_canvas():
    img = render_scene(make_scene([line_layer([0, 1], [0, 1])]), scale=2.0)
    assert (img.width, img.height) == (960, 720)


def test_render_scene_to_bytes_is_valid_png(tmp_path):
    blob = render_scene_to_bytes(make_scene([line_layer([0, 1], [0, 1])]))
    assert blob[:8] == b"\x89PNG\r\n\x1a\n"
    path = tmp_path / "a.png"
    path.write_bytes(blob)
    back = read_png(path)
    assert (back.width, back.height, back.channels) == (480, 360, 4)


def test_render_is_byte_deterministic():
    scene = make_scene([line_layer([0.0, 0.3, 0.7, 1.0], [1.0, 0.5, 0.25, 0.125])])
    assert render_scene_to_bytes(scene) == render_scene_to_bytes(scene)
    assert render_scene_to_bytes(scene, scale=1.5) == render_scene_to_bytes(scene, scale=1.5)


def test_render_scene_to_file_creates_parents(tmp_path):
    out = render_scene_to_file(tmp_path / "deep" / "x.png", make_scene([line_layer([0, 1], [0, 1])]))
    assert out.exists() and out.stat().st_size > 0


def test_render_scene_rejects_invalid_scene():
    bad = make_scene([line_layer([0, 1], [0, 1])], badges=[])
    with pytest.raises(RasterError):
        render_scene(bad)
    assert render_scene(bad, validate=False).width == 480  # 关掉校验仍然能画


# =========================================================================== #
# 2. 校验器
# =========================================================================== #
def test_validate_scene_accepts_minimal_scene():
    assert validate_scene(make_scene([line_layer([0, 1], [0, 1])])) == []


def test_validate_scene_requires_source_badge():
    scene = make_scene([line_layer([0, 1], [0, 1])], badges=[{"kind": "critical", "text": "x", "x": 1, "y": 1}])
    issues = validate_scene(scene)
    assert any("source" in i for i in issues)


def test_validate_scene_flags_missing_autoscaled_and_short_pitfalls():
    scene = make_scene([line_layer([0, 1], [0, 1])], pitfalls="太短")
    del scene["axes"][0]["autoscaled"]
    issues = validate_scene(scene)
    assert any("autoscaled" in i for i in issues)
    assert any("pitfalls" in i for i in issues)


def test_validate_scene_requires_overflow_note_when_flagged():
    scene = make_scene([line_layer([0, 1], [0, 1])], overflow={"flag": True, "note": "  "})
    assert any("overflow.note" in i for i in validate_scene(scene))


def test_validate_scene_flags_bad_layer_shapes():
    scene = make_scene([heat_layer([1.0, 2.0, 3.0], nx=3, ny=2)])  # 长度 ≠ nx*ny
    assert any("values 长度" in i for i in validate_scene(scene))


def test_validate_scene_flags_unknown_cmap_and_log_vmin():
    scene = make_scene([heat_layer([1.0, 2.0, 3.0, 4.0], nx=2, ny=2, cmap="jet")])
    assert any("色标" in i for i in validate_scene(scene))
    scene = make_scene([heat_layer([1.0, 2.0, 3.0, 4.0], nx=2, ny=2, logScale=True, vmin=-1.0)])
    assert any("vmin > 0" in i for i in validate_scene(scene))


def test_validate_scene_flags_nan_in_line_and_axes_mismatch():
    scene = make_scene([line_layer([0, 1, 2], [0, math.nan, 2])])
    assert any("非有限纵坐标" in i for i in validate_scene(scene))
    scene = make_scene([{"kind": "line", "axes": "nope", "series": []}])
    assert any("不存在的轴" in i for i in validate_scene(scene))


# =========================================================================== #
# 3. 轴：1-2-5 / 对数 / 格式化
# =========================================================================== #
def test_nice_ticks_linear_uses_1_2_5():
    # 注意 0.6000000000000001：刻度是 i*step 逐点算出来的（与 axes.ts 一字不差），
    # 这正是"跨语言对拍"要守住的东西——不要为了好看把它改成 0.6。
    assert nice_ticks(0.0, 1.0, 7) == [0.0, 0.2, 0.4, 0.6000000000000001, 0.8, 1.0]
    assert nice_ticks(0.1, 0.9, 7)[:3] == [0.1, 0.2, 0.30000000000000004]
    assert nice_ticks(0.0, 7.0, 7) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]


def test_nice_ticks_degenerate_and_reversed():
    assert nice_ticks(3.0, 3.0) == [3.0]
    with pytest.raises(ValueError):
        nice_ticks(1.0, 0.0)
    with pytest.raises(ValueError):
        nice_ticks(-1.0, 1.0, log=True)


def test_nice_ticks_log_decades_and_five_stride():
    ticks = nice_ticks(1e-3, 1.0, 7, log=True)
    assert ticks == [1e-3, 1e-2, 1e-1, 1.0]
    # 跨度不足一个十倍程时退化为 1/2/5 格点（这里只有 10^0 落在区间内）
    assert nice_ticks(0.7, 3.0, 7, log=True) == [1.0, 2.0]
    assert nice_ticks(2.0, 20.0, 7, log=True) == [5.0, 10.0]


def test_nice_limits_covers_input_for_many_intervals():
    cases = [(0.1, 0.9), (1.0, 100.0), (-5.0, 5.0), (-0.003, 0.007), (1e-6, 1e-5),
             (0.0, 1.0), (123.4, 987.6), (1e5, 1e6), (1e-12, 1e-9), (2.5, 2.5000001)]
    for lo, hi in cases:
        nlo, nhi = nice_limits(lo, hi, 7)
        assert nlo <= lo and nhi >= hi, (lo, hi, nlo, nhi)
        ticks = nice_ticks(nlo, nhi, 7)
        assert len(ticks) >= 2
        assert ticks[0] >= nlo and ticks[-1] <= nhi


def test_nice_limits_log_and_degenerate():
    lo, hi = nice_limits(0.7, 300.0, 7, log=True)
    assert lo <= 0.7 and hi >= 300.0 and math.log10(hi / lo) % 1 == 0
    a, b = nice_limits(5.0, 5.0, 7, log=True)
    assert (a, b) == (0.1, 100.0)  # 5→[0.5,50] 后再把十倍程数扩到 ≥4（与 axes.ts 同口径）


def test_format_number_uses_ecmascript_half_up_tie_rule():
    assert format_number(2.5, 1) == "3"       # JS: (2.5).toFixed(0) === "3"
    assert format_number(-2.5, 1) == "-3"     # JS: (-2.5).toFixed(0) === "-3"（逢半远离 0）
    assert format_number(-0.4, 1) == "-0.4"
    assert format_number(0.0) == "0"
    assert format_number(math.nan) == "NaN"
    assert format_number(math.inf) == "∞"
    assert format_number(-math.inf) == "-∞"


def test_format_number_fixed_and_scientific():
    assert format_number(0.1 + 0.2) == "0.3"
    assert format_number(1.0 / 3.0) == "0.333333"
    assert format_number(1234.5678) == "1234.57"
    assert format_number(1.2345e-7) == "1.2345e-7"  # 6 位有效数字：尾随 0 去掉
    assert format_number(-6.02e23) == "-6.02e23"
    assert format_number(1e5) == "100000"


def test_format_sci_and_tick_label():
    assert format_sci(0.0) == "0"
    assert format_sci(1.5e-7) == "1.5×10⁻⁷"
    assert tick_label(1e-7, log=True) == "10⁻⁷"
    assert tick_label(10.0, log=True) == "10"
    assert tick_label(1.0, log=True) == "1"
    assert tick_label(5e-3, log=True) == "5×10⁻³"
    assert tick_label(0.25, log=True) == "0.25"


def test_map_and_data_to_pixel_log_nonpositive_is_nan():
    px, py = data_to_pixel(make_axis(yscale="log", ylim=[1e-6, 1.0]), 0.5, 1e-3)
    assert py == pytest.approx(60.0 + 240.0 / 2.0, abs=1e-9)
    _, py0 = data_to_pixel(make_axis(yscale="log", ylim=[1e-6, 1.0]), 0.5, 0.0)
    assert math.isnan(py0)
    del px


# =========================================================================== #
# 4. 跨语言对拍（node → dist/viz/shared/src/axes.js）
# =========================================================================== #
_NODE_SCRIPT = r"""
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
const m = await import(pathToFileURL(process.env.AXES_JS).href);
const payload = JSON.parse(readFileSync(0, "utf8"));
const out = { ticks: [], limits: [], fmt: [], labels: [], sci: [] };
for (const c of payload.ticks) out.ticks.push(Array.from(m.niceTicks(c[0], c[1], c[2], { log: c[3] })));
for (const c of payload.limits) out.limits.push(Array.from(m.niceLimits(c[0], c[1], c[2], { log: c[3], pad: c[4] })));
for (const c of payload.fmt) out.fmt.push(m.formatNumber(c[0], c[1]));
for (const c of payload.labels) out.labels.push(m.tickLabel(c[0], { log: c[1] }));
for (const c of payload.sci) out.sci.push(m.formatSci(c[0], c[1]));
process.stdout.write(JSON.stringify(out));
"""

LINEAR_INTERVALS = [
    (0.1, 0.9), (1.0, 100.0), (-5.0, 5.0), (-0.003, 0.007), (1e-6, 1e-5),
    (0.0, 1.0), (0.0, 7.0), (123.4, 987.6), (1e5, 1e6), (1e-12, 1e-9),
    (-1e4, -1.0), (2.5, 2.5000001), (1e-3, 1e3), (0.000123, 0.000987),
    (999.0, 1001.0), (-0.5, 0.5), (1e8, 1.0000001e8), (3.0, 3.0001),
    (7.5e-5, 2.5e-3), (1e-300, 1e-299), (5e-7, 5e-6), (0.9999, 1.0001),
    (-1e-5, -1e-7), (1e-16, 1e-15), (42.0, 42.5), (1e20, 1e21),
]
LOG_INTERVALS = [
    (1e-6, 1e-1), (0.7, 300.0), (1e-20, 1.0), (2.0, 8.0), (0.05, 0.5),
    (1e-3, 1e-3 * 1.0000001), (1e5, 1e9), (1.5, 2.5), (3e-4, 7e-2),
    (1e-300, 1e-290), (1.0, 1e12), (0.9, 1.1), (1e-9, 1e-8), (4.0, 40.0),
]
FORMAT_VALUES = [
    0.0, 1.0, -1.0, 0.5, 2.5, -2.5, 3.5, 1.25, 1e-7, 1.2345e-7, 123456.789,
    1e5, 99999.5, 1e-3, 0.001, 42.0, -0.0001234, 1e21, 5e-324,
    1.7976931348623157e308, 1234.5, 0.1 + 0.2, 1.0 / 3.0, -1.0 / 3.0,
    6.02214076e23, 2.0 ** -20, 123456789.0, 0.0009999999, -7.5e-11, 999.999,
]
SIGS = [1, 2, 3, 4, 6, 8]


def _axes_js() -> pathlib.Path:
    """已编译的 axes.js；缺失时用仓库本地的 tsc 编译一次（不联网）。"""
    target = ROOT / "dist" / "viz" / "shared" / "src" / "axes.js"
    if target.exists():
        return target
    local = ROOT / "node_modules" / "typescript" / "bin" / "tsc"
    cmd = ["node", str(local), "-p", "tsconfig.json"] if local.exists() else ["npx", "--no-install", "tsc", "-p", "tsconfig.json"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if not target.exists():  # 并发写 dist 的瞬时失败：按纪律重试一次，不改代码
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert target.exists(), f"编译 dist 失败：{proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    return target


def _bits(v) -> bytes:
    """IEEE-754 位模式（能同时抓住 1 ulp 差异与 -0.0/0.0 的符号差异）。"""
    return struct.pack("<d", float(v))


def _node_results(payload: dict) -> dict:
    env = dict(os.environ, AXES_JS=str(_axes_js()))
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", _NODE_SCRIPT],
        input=json.dumps(payload), capture_output=True, text=True, env=env, cwd=ROOT,
    )
    assert proc.returncode == 0, f"node 调用失败：{proc.stderr}"
    return json.loads(proc.stdout)


def test_cross_language_nice_ticks_and_limits_match_ts():
    payload = {
        "ticks": [[lo, hi, 7, False] for lo, hi in LINEAR_INTERVALS]
        + [[lo, hi, tgt, log] for lo, hi in LINEAR_INTERVALS for tgt in (5,)
           for log in (False,)]
        + [[lo, hi, 7, True] for lo, hi in LOG_INTERVALS]
        + [[lo, hi, tgt, True] for lo, hi in LOG_INTERVALS[:6] for tgt in (2, 10)],
        "limits": [[lo, hi, 7, False, 0.0] for lo, hi in LINEAR_INTERVALS]
        + [[lo, hi, 7, False, 0.1] for lo, hi in LINEAR_INTERVALS[:12]]
        + [[lo, hi, 7, True, 0.0] for lo, hi in LOG_INTERVALS]
        + [[lo, hi, 7, True, 0.2] for lo, hi in LOG_INTERVALS[:8]],
        "fmt": [[v, sig] for v in FORMAT_VALUES for sig in SIGS],
        "labels": [[1e-7, True], [1e-3, True], [5e-3, True], [0.25, True], [1.0, True],
                   [10.0, True], [1e6, True], [0.0, False], [1.5, False], [-2.5, False]],
        "sci": [[float(v), 6] for v in FORMAT_VALUES if v == v and abs(v) != float("inf") and v != 0.0][:20],
    }
    ts = _node_results(payload)

    total_ticks = 0
    for (lo, hi, tgt, log), got in zip(payload["ticks"], ts["ticks"]):
        mine = nice_ticks(lo, hi, tgt, log=log)
        assert len(mine) == len(got), f"刻度个数不一致 ticks({lo},{hi},{tgt},{log})：{mine} vs {got}"
        for a, b in zip(mine, got):
            assert _bits(a) == _bits(b), f"刻度值不一致 ticks({lo},{hi},{tgt},{log})：{a!r} vs {b!r}"
        total_ticks += len(mine)
    assert total_ticks >= 300  # ≥20 组区间 × 多档 target，逐值比对
    for (lo, hi, tgt, log, pad), got in zip(payload["limits"], ts["limits"]):
        mine = nice_limits(lo, hi, tgt, log=log, pad=pad)
        assert [_bits(v) for v in mine] == [_bits(v) for v in got], f"轴范围不一致 limits({lo},{hi},{tgt},{log},{pad})"
    for (v, sig), got in zip(payload["fmt"], ts["fmt"]):
        assert format_number(v, sig) == got, f"formatNumber({v!r},{sig}) → {format_number(v, sig)!r} vs {got!r}"
    for (v, log), got in zip(payload["labels"], ts["labels"]):
        assert tick_label(v, log=log) == got, f"tickLabel({v!r},{log}) 不一致"
    for (v, sig), got in zip(payload["sci"], ts["sci"]):
        assert format_sci(v, sig) == got, f"formatSci({v!r},{sig}) → {format_sci(v, sig)!r} vs {got!r}"


# =========================================================================== #
# 5. 色标 / 热力图
# =========================================================================== #
def test_build_heat_rgba_nan_is_neutral_gray_not_transparent():
    # 行主序：row0 = [nan, 0, 1]，row1 = [+inf, -inf, 0.5]
    rgba = build_heat_rgba([math.nan, 0.0, 1.0, math.inf, -math.inf, 0.5], 3, 2, "gray", 0.0, 1.0)
    for row, col in ((0, 0), (1, 0), (1, 1)):
        assert tuple(int(v) for v in rgba[row, col]) == (*GRAY_NAN, 255), (row, col)
    assert tuple(int(v) for v in rgba[0, 1]) == (0, 0, 0, 255)
    assert tuple(int(v) for v in rgba[0, 2]) == (255, 255, 255, 255)
    assert tuple(int(v) for v in rgba[1, 2])[:3] == lookup_cmap("gray", 0.5)


def test_build_heat_rgba_log_and_clipbelow_are_gray():
    log_rgba = build_heat_rgba([0.0, -1.0, 1e-4, 1.0], 2, 2, "gray", 1e-4, 1.0, log_scale=True)
    assert tuple(log_rgba[0, 0]) == (*GRAY_NAN, 255)
    assert tuple(log_rgba[0, 1]) == (*GRAY_NAN, 255)
    clip_rgba = build_heat_rgba([0.0, 5.0], 2, 1, "gray", 0.0, 10.0, clip_below=1.0)
    assert tuple(clip_rgba[0, 0]) == (*GRAY_NAN, 255)


def test_apply_cmap_constant_field_uses_midpoint_and_errors():
    rgba = apply_cmap("gray", [3.0, 3.0], 3.0, 3.0)
    assert tuple(rgba[0][:3]) == lookup_cmap("gray", 0.5)
    with pytest.raises(RasterError):
        apply_cmap("gray", [1.0], 0.0, 1.0, log_scale=True)
    with pytest.raises(RasterError):
        apply_cmap("nope", [1.0], 0.0, 1.0)
    assert math.isnan(normalize_value(0.0, 1e-3, 1.0, True))


def test_render_heat_nan_pixels_are_gray_in_image():
    scene = make_scene([heat_layer([math.nan, 0.0, 0.0, math.nan], 2, 2, cmap="viridis")])
    arr = pixels(scene)
    gray = np.all(np.abs(arr[:, :, :3].astype(int) - np.array(GRAY_NAN)) == 0, axis=2)
    assert gray.sum() > 1000  # 两个 NaN 格子 ≈ 一半的热力区
    assert arr[gray][:, 3].min() == 255  # 不是透明


def test_render_heat_row0_is_at_bottom():
    """节点采样约定：第 0 行 = y 最小 = 画在**底部**（见 raster.py docstring 的差异说明）。"""
    axis = make_axis(rect=[60.0, 60.0, 200.0, 200.0], xlim=[0.0, 1.0], ylim=[0.0, 1.0], grid=False)
    scene = make_scene([heat_layer([0.0, 0.0, 1.0, 1.0], 2, 2, cmap="gray")], axes=[axis])
    arr = pixels(scene)
    top = arr[70, 160, 0]
    bottom = arr[250, 160, 0]
    assert top > 200 and bottom < 60


def test_render_heat_log_colorbar_is_drawn_and_ordered():
    axis = make_axis(rect=[40.0, 40.0, 200.0, 200.0])
    scene = make_scene(
        [heat_layer([1e-4, 1e-3, 1e-2, 1e-1], 2, 2, cmap="viridis", vmin=1e-4, vmax=1e-1, logScale=True)],
        axes=[axis], canvas=(400.0, 300.0),
    )
    assert validate_scene(scene) == []
    arr = pixels(scene)
    bar_x = int(40 + 200 + 46 + 7)
    colors = {tuple(arr[y, bar_x, :3]) for y in range(50, 240)}
    assert len(colors) > 8, "色标条应当是连续渐变的色带"
    assert arr[50, bar_x, 0] != arr[235, bar_x, 0]


# =========================================================================== #
# 6. 折线断开 / 图层退化输入
# =========================================================================== #
def test_layer_to_pixel_path_inserts_none_on_nan():
    path = layer_to_pixel_path(make_axis(), [0.0, 0.5, 1.0], [0.0, math.nan, 1.0])
    assert path[2] is None and path[3] is None
    assert path[0] is not None


def test_render_line_breaks_at_nan():
    axis = make_axis(grid=False, xticks=[0.0, 1.0], yticks=[0.0, 1.0])
    scene = make_scene([line_layer([0.0, 0.25, 0.5, 0.75, 1.0], [0.5, 0.5, math.nan, 0.5, 0.5])], axes=[axis])
    arr = pixels(scene, validate=False)
    mask = ink(arr)
    # x=0.5 处的列（像素 260）必须没有线（只有轴框在 y=60/300）
    column = mask[70:290, 259:262]
    assert not column.any(), "NaN 处必须断开，不能用直线连过去"
    assert mask[70:290, 70:90].any() and mask[70:290, 430:450].any()


def test_render_clips_line_to_axis_rect():
    """y 远在窗外的折线段不能画到轴外（标题/角标/相邻面板上）。"""
    axis = make_axis(grid=False, xticks=[0.0, 1.0], yticks=[0.0, 1.0])
    scene = make_scene([line_layer([0.5, 0.5, 0.6], [0.5, 100.0, 0.5])], axes=[axis], title="", badges=[])
    arr = pixels(scene, validate=False)
    assert not ink(arr, (70, 0, 450, 58)).any(), "窗外的折线段必须被裁在轴框内"
    assert ink(arr, (70, 70, 450, 290)).any(), "窗内的折线段仍要画出来"


def test_render_clips_scatter_to_axis_rect():
    axis = make_axis(grid=False, xticks=[0.0, 1.0], yticks=[0.0, 1.0])
    scene = make_scene(
        [{"kind": "scatter", "axes": "main", "x": [0.5, 0.5], "y": [0.5, 100.0],
          "r": 4.0, "color": "#dd6b20"}],
        axes=[axis], title="", badges=[],
    )
    arr = pixels(scene, validate=False)
    assert not ink(arr, (70, 0, 450, 58)).any()


def test_render_empty_line_layer_is_blank_inside_axis():
    axis = make_axis(grid=False)
    arr = pixels(make_scene([line_layer([], [])], axes=[axis]))
    assert not ink(arr, (70, 70, 450, 290)).any()


def test_render_single_point_line_layer():
    axis = make_axis(grid=False)
    bare = ink(pixels(make_scene([line_layer([0.5], [0.5])], axes=[axis])), (70, 70, 450, 290))
    assert not bare.any(), "单点且无 marker：canvas 语义下一个点构不成线段（与 render2d.ts 一致）"
    marked = ink(pixels(make_scene([line_layer([0.5], [0.5], marker="o")], axes=[axis])), (70, 70, 450, 290))
    assert 0 < marked.sum() < 120  # 只是一个小圆点


def test_render_all_nan_line_layer_is_blank_inside_axis():
    axis = make_axis(grid=False)
    arr = pixels(make_scene([line_layer([0.0, 0.5, 1.0], [math.nan] * 3)], axes=[axis]), validate=False)
    assert not ink(arr, (70, 70, 450, 290)).any()


def test_render_band_layer_and_nan_degradation():
    band = {"kind": "band", "axes": "main",
            "band": {"x": [0.0, 0.5, 1.0], "ylo": [0.2, 0.3, 0.4], "yhi": [0.8, 0.7, 0.6],
                     "color": "#4c78a8", "alpha": 0.3}}
    arr = pixels(make_scene([band]))
    assert ink(arr, (70, 70, 450, 290)).any()
    band_nan = {"kind": "band", "axes": "main",
                "band": {"x": [0.0, 0.5, 1.0], "ylo": [0.2, math.nan, 0.4], "yhi": [0.8, math.nan, 0.6],
                         "color": "#4c78a8", "alpha": 0.3}}
    pixels(make_scene([band_nan]), validate=False)  # 不崩即可


def test_render_scatter_layer_skips_nan():
    scene = make_scene([{"kind": "scatter", "axes": "main", "x": [0.2, 0.5, math.nan],
                         "y": [0.2, math.nan, 0.5], "r": 4.0, "color": "#dd6b20"}], axes=[make_axis(grid=False)])
    arr = pixels(scene, validate=False)
    mask = ink(arr, (70, 70, 450, 290))
    assert mask.any()
    assert not mask[70:290, 250:290].any()  # 两个 NaN 点都没画


def test_render_bars_layer_with_errors():
    bars = {"kind": "bars", "axes": "main", "x": [0.2, 0.5, 0.8], "values": [0.3, 0.6, 0.4],
            "width": 0.1, "color": "#2f855a", "errors": [0.05, 0.1, 0.02], "baseline": 0.0}
    axis = make_axis(ylim=[0.0, 1.0], yticks=[0.0, 0.5, 1.0])
    arr = pixels(make_scene([bars], axes=[axis]))
    assert ink(arr, (70, 70, 450, 290)).any()


def test_render_contour_layer_draws_levels():
    values = [0.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 0.0]  # 3×3 峰
    contour = {"kind": "contour", "axes": "main", "nx": 3, "ny": 3, "values": values,
               "extent": [0.0, 1.0, 0.0, 1.0], "levels": [0.25, 0.75], "color": "#222", "width": 1.0}
    arr = pixels(make_scene([contour]))
    assert ink(arr, (70, 70, 450, 290)).sum() > 50


def test_render_vectorfield_layer_draws_arrows():
    nx = ny = 4
    u = [1.0] * (nx * ny)
    v = [0.0] * (nx * ny)
    field = {"kind": "vectorfield", "axes": "main", "nx": nx, "ny": ny, "u": u, "v": v,
             "extent": [0.0, 1.0, 0.0, 1.0], "stride": 1, "color": "#666", "scale": 0.1}
    arr = pixels(make_scene([field]))
    assert ink(arr, (70, 70, 450, 290)).sum() > 200


def test_render_loglogfit_draws_both_slopes():
    h = [1e-3, 1e-2, 1e-1]
    err = [1e-6, 1e-4, 1e-2]
    axis = make_axis(xlim=[1e-3, 1e-1], ylim=[1e-6, 1e-2], xscale="log", yscale="log",
                     xticks=[1e-3, 1e-2, 1e-1], yticks=[1e-6, 1e-4, 1e-2])
    fit = {"kind": "loglogfit", "axes": "main", "h": h, "err": err, "measuredSlope": 2.0,
           "theorySlope": 2.0, "seriesLabel": "实测斜率 2.0", "theoryLabel": "理论斜率 2",
           "refX": [1e-3, 1e-1]}
    arr = pixels(make_scene([fit], axes=[axis]))
    region = arr[70:290, 70:450, :3]
    has_blue = bool(np.any(np.all(region == np.array([43, 108, 176]), axis=2)))
    has_red = bool(np.any(np.all(region == np.array([197, 48, 48]), axis=2)))
    assert has_blue and has_red, "实测曲线与理论参考线必须同时出现（规则 2）"


def test_render_loglogfit_labels_follow_chrome_texts_flag():
    """网页显示层把两条标注交给 HTML 图例，光栅器不应再在画布上重复写一遍。"""
    h = [1e-3, 1e-2, 1e-1]
    err = [1e-6, 1e-4, 1e-2]
    axis = make_axis(xlim=[1e-3, 1e-1], ylim=[1e-6, 1e-2], xscale="log", yscale="log",
                     xticks=[1e-3, 1e-2, 1e-1], yticks=[1e-6, 1e-4, 1e-2])
    fit = {"kind": "loglogfit", "axes": "main", "h": h, "err": err, "measuredSlope": 2.0,
           "theorySlope": 2.0, "seriesLabel": "实测斜率 2.00", "theoryLabel": "理论斜率 2",
           "refX": [1e-3, 1e-1]}
    raw = make_scene([fit], axes=[axis])
    without = json.loads(json.dumps(raw))
    without["chrome"] = {"texts": False}
    assert validate_scene(raw) == []
    assert validate_scene(without) == []
    region = (300, 64, 456, 110)
    with_labels = ink(pixels(raw), region).sum()
    no_labels = ink(pixels(without), region).sum()
    assert with_labels - no_labels > 80, (
        f"关闭 chrome.texts 后标注应消失：带标注 {with_labels} 像素，无标注 {no_labels} 像素"
    )
    # 曲线本身不受影响：整幅图的差异像素应当集中在标注区域
    diff = np.any(pixels(raw)[:, :, :3] != pixels(without)[:, :, :3], axis=2)
    ys, xs = np.nonzero(diff)
    assert len(xs) > 0
    assert xs.min() >= 300 and xs.max() < 456 and ys.min() >= 64 and ys.max() < 110


def test_marching_squares_single_cell():
    seg = marching_squares([0.0, 1.0, 0.0, 1.0], 2, 2, 0.5)
    assert len(seg) == 4  # 一条线段
    assert all(0.0 <= v <= 1.0 for v in seg)


# =========================================================================== #
# 7. 装饰：徽标 / pitfalls / 文本 / 标题
# =========================================================================== #
def test_render_four_badge_kinds():
    badges = [
        {"kind": "source", "text": "数据：后端 .nlf 帧", "x": 8.0, "y": 8.0},
        {"kind": "autoscale", "text": "坐标轴：自适应（可能掩盖发散）", "x": 8.0, "y": 26.0},
        {"kind": "overflow", "text": "溢出：图层0 的 y 超出轴范围 0.2 倍", "x": 8.0, "y": 44.0},
        {"kind": "critical", "text": "r = 0.6　理论临界 0.5　⚠ 已越界", "x": 8.0, "y": 62.0},
    ]
    scene = make_scene([line_layer([0, 1], [0, 1])], badges=badges, canvas=(560.0, 360.0))
    assert validate_scene(scene) == []
    arr = pixels(scene)
    for i in range(4):
        band = ink(arr, (6, 6 + 18 * i, 460, 24 + 18 * i))
        assert band.any(), f"第 {i + 1} 个徽标没有画出来"


def test_pitfalls_panel_grows_with_text():
    short = make_scene([line_layer([0, 1], [0, 1])], pitfalls="短陷阱说明也要超过十个字。")
    long = make_scene([line_layer([0, 1], [0, 1])], pitfalls="这是一段很长的陷阱说明，" * 6)
    scene_short, scene_long = pixels(short), pixels(long)

    def panel_top(arr: np.ndarray) -> int:
        border = np.all(arr[:, :, :3] == np.array([221, 107, 32]), axis=2)
        rows = np.nonzero(border.any(axis=1))[0]
        return int(rows.min())

    assert panel_top(scene_long) < panel_top(scene_short), "更长的 pitfalls 应该有更高的面板"


def test_render_texts_and_titles():
    plain = make_scene([line_layer([0, 1], [0, 1])], title="无注解")
    rich = make_scene(
        [line_layer([0, 1], [0, 1])], title="0.0 · 标题",
        texts=[{"x": 300.0, "y": 120.0, "text": "注解文本 note", "size": 12.0, "color": "#2d3748",
                "align": "left", "role": "note"}],
    )
    assert not np.array_equal(pixels(plain), pixels(rich))
    assert validate_scene(rich) == []


def test_render_vertical_axis_label():
    with_label = make_scene([line_layer([0, 1], [0, 1])], axes=[make_axis(ylabel="位移 x")])
    without = make_scene([line_layer([0, 1], [0, 1])], axes=[make_axis(ylabel="")])
    assert not np.array_equal(pixels(with_label), pixels(without))


def test_vertical_axis_label_is_centered_on_axis():
    """旋转文本必须按逐字字格求总高；按横排宽度估会把标签整体上移半段。"""
    axis = make_axis(grid=False, xticks=[], yticks=[], ylabel="abcdef")
    arr = pixels(make_scene([], axes=[axis], title="", badges=[]), validate=False)
    mask = ink(arr, (0, 40, 20, 280))
    ys, _xs = np.where(mask)
    assert ys.size > 0, "y 轴名必须画出来"
    center = (ys.min() + ys.max()) / 2.0
    assert abs(center - (axis["rect"][1] + axis["rect"][3] / 2.0)) <= 6.0


def test_render_log_axis_labels_present():
    axis = make_axis(yscale="log", ylim=[1e-6, 1.0], yticks=nice_ticks(1e-6, 1.0, 7, log=True))
    scene = make_scene([line_layer([0.0, 1.0], [1e-6, 1.0])], axes=[axis])
    arr = pixels(scene)
    assert ink(arr, (0, 60, 55, 300)).any(), "对数刻度标签必须画出来"


# =========================================================================== #
# 8. 帧 → 标准探针场景
# =========================================================================== #
@pytest.fixture(scope="module")
def chapter_entry():
    from server.registry import default_registry

    registry = default_registry()
    if "0.0" not in registry.ids():
        pytest.fail("注册表里没有 0.0 章")
    return registry.get("0.0")


@pytest.mark.parametrize("viz_id", ["oscillator_regimes", "heat_ftcs"])
def test_default_scene_for_frame_is_valid_and_complete(chapter_entry, viz_id):
    from server.frames import build_frame

    frame = build_frame(chapter_entry, viz_id, None, 0.0)
    viz = chapter_entry.manifest.viz(viz_id).to_json()
    scene = default_scene_for_frame(viz, frame)
    assert validate_scene(scene) == []
    kinds = {b["kind"] for b in scene["badges"]}
    assert "source" in kinds and "autoscale" in kinds and "critical" in kinds
    assert [a["id"] for a in scene["axes"]] == ["main", "error"]
    assert any(l["kind"] == "line" for l in scene["layers"])
    assert scene["pitfalls"] == frame.pitfalls
    assert render_scene_to_bytes(scene)


def test_default_scene_for_frame_is_deterministic(chapter_entry):
    from server.frames import build_frame

    frame = build_frame(chapter_entry, "heat_ftcs", None, 0.0)
    viz = chapter_entry.manifest.viz("heat_ftcs").to_json()
    a = default_scene_for_frame(viz, frame)
    b = default_scene_for_frame(viz, frame)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert render_scene_to_bytes(a) == render_scene_to_bytes(b)


def test_default_scene_error_axis_is_log_and_badged(chapter_entry):
    from server.frames import build_frame

    frame = build_frame(chapter_entry, "oscillator_regimes", None, 0.0)
    viz = chapter_entry.manifest.viz("oscillator_regimes").to_json()
    scene = default_scene_for_frame(viz, frame)
    err_axis = next(a for a in scene["axes"] if a["id"] == "error")
    assert err_axis["yscale"] == "log" and err_axis["autoscaled"] is True
    assert err_axis["ylim"][0] > 0
    assert any(b["kind"] == "autoscale" for b in scene["badges"])


def test_default_scene_handles_2d_field_and_colorbar():
    from server.nlf import Frame

    ny, nx = 4, 5
    values = np.linspace(0.0, 1.0, nx * ny)
    frame = Frame(chapter="9.9", viz="fake_heat")
    frame.add("x", np.linspace(0.0, 1.0, nx), label="x")
    frame.add("y", np.linspace(0.0, 1.0, ny), label="y")
    frame.add("U", values.reshape(ny, nx), label="U")
    frame.set_meta(t=0.0)
    frame.set_strings(data_source="backend")
    frame.pitfalls = "二维场的探针：热力图第 0 行必须在底部，NaN 必须画成中性灰而不是透明。"
    frame.critical = {"r": 0.5}
    viz = {"id": "fake_heat", "title": "假热力图", "dataSource": "backend",
           "params": [{"name": "r", "type": "float", "default": 0.6, "critical": [0.5], "beyond": "above"}],
           "requiredAux": [], "axes": [{"id": "main", "autoscaled": False, "title": "U", "ylabel": "U"}],
           "pitfalls": frame.pitfalls}
    scene = default_scene_for_frame(viz, frame)
    assert validate_scene(scene) == []
    heat = [l for l in scene["layers"] if l["kind"] == "heat"]
    assert heat and heat[0]["nx"] == nx and heat[0]["ny"] == ny
    kinds = {b["kind"] for b in scene["badges"]}
    assert "critical" in kinds
    assert render_scene_to_bytes(scene)


def test_default_scene_survives_all_nan_frame():
    from server.nlf import Frame

    frame = Frame(chapter="9.9", viz="nan_viz")
    frame.add("x", np.linspace(0.0, 1.0, 5))
    frame.add("u", np.full(5, np.nan))
    frame.add("error", np.full(5, np.nan))
    frame.set_strings(data_source="backend")
    frame.pitfalls = "全 NaN 的帧：探针场景仍要能构造并画出来（灰色表示无数据）。"
    viz = {"id": "nan_viz", "title": "NaN", "dataSource": "backend", "params": [],
           "requiredAux": ["error"], "axes": [{"id": "main", "autoscaled": True}], "pitfalls": frame.pitfalls}
    scene = default_scene_for_frame(viz, frame)
    assert validate_scene(scene) == []
    render_scene_to_bytes(scene)


def test_check_overflow_reports_offenders_and_non_finite():
    scene = make_scene([line_layer([0.0, 0.5, 1.5], [0.0, 0.5, math.nan])])
    report = check_overflow(scene)
    assert report["flag"] is True
    assert report["offenders"] and report["offenders"][0]["axis"] == "x"
    assert report["nonFinite"] == 1
    assert "超出轴" in report["note"]
    clean = check_overflow(make_scene([line_layer([0.0, 1.0], [0.0, 1.0])]))
    assert clean["flag"] is False and clean["note"] == ""


# =========================================================================== #
# 9. 视觉回归端到端
# =========================================================================== #
def _write_scene_json(path: pathlib.Path, scene: dict) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scene, ensure_ascii=False), encoding="utf-8")
    return path


def test_visual_regression_update_then_check_roundtrip(tmp_path):
    from server.frames import build_frame
    from server.registry import default_registry

    entry = default_registry().get("0.0")
    frame = build_frame(entry, "heat_ftcs", None, 0.0)
    scene = default_scene_for_frame(entry.manifest.viz("heat_ftcs").to_json(), frame)
    scenes = tmp_path / "scenes"
    _write_scene_json(scenes / "0.0" / "heat_ftcs__probe__t0.json", scene)
    kwargs = dict(scenes_dir=scenes, baseline_dir=tmp_path / "baselines", dist_visual=tmp_path / "visual",
                  chapters=["0.0"], vizzes=["heat_ftcs"], quiet=True)

    up = vr.run(mode="update", **kwargs)
    assert up["ok"] and up["totals"]["created"] == 1
    baseline = tmp_path / "baselines" / "0.0" / "heat_ftcs__probe__t0.png"
    assert baseline.exists()

    chk = vr.run(mode="check", **kwargs)
    assert chk["ok"] and chk["totals"]["ok"] == 1
    assert chk["entries"][0]["source"] == "scene"
    assert chk["entries"][0]["diff"]["sameBytes"] is True


def test_visual_regression_check_reports_pixel_diff(tmp_path):
    from server.frames import build_frame
    from server.registry import default_registry

    entry = default_registry().get("0.0")
    frame = build_frame(entry, "heat_ftcs", None, 0.0)
    scene = default_scene_for_frame(entry.manifest.viz("heat_ftcs").to_json(), frame)
    scenes = tmp_path / "scenes"
    _write_scene_json(scenes / "0.0" / "probe.json", scene)
    baselines = tmp_path / "baselines"
    kwargs = dict(scenes_dir=scenes, baseline_dir=baselines, dist_visual=tmp_path / "visual",
                  chapters=["0.0"], quiet=True)
    vr.run(mode="update", **kwargs)
    # 人为改一个像素：报告必须给出可诊断的四件套
    baseline = baselines / "0.0" / "probe.png"
    arr = read_png(baseline).array().copy()
    arr[100:110, 200:230, :3] = 0
    write_rgb(baseline, arr[:, :, :3])
    out = vr.run(mode="check", write_diff=True, **kwargs)
    assert out["ok"] is False and out["totals"]["failed"] == 1
    diff = out["entries"][0]["diff"]
    assert diff["differentPixels"] > 200 and diff["maxChannelDelta"] > 100
    assert diff["bbox"] is not None and diff["bbox"][0] <= 200 and diff["bbox"][2] >= 230
    assert out["entries"][0]["status"] == "mismatch"
    assert (tmp_path / "visual" / "diff" / "0.0" / "probe.png").exists()


def test_visual_regression_fallback_does_not_fail(tmp_path, capsys):
    out = vr.run(mode="check", scenes_dir=tmp_path / "no-scenes", baseline_dir=tmp_path / "baselines",
                 dist_visual=tmp_path / "visual", chapters=["0.0"], quiet=True)
    assert out["totals"]["fallback"] == 2 and out["totals"]["scene"] == 0
    assert out["ok"] is True, "兜底路径（没有 TS 场景）不应让 --check 变红"
    assert all(e["reason"] for e in out["entries"])


def test_visual_regression_require_scenes_fails_without_dumps(tmp_path):
    """兜底路径默认不算失败；CI 用 --require-scenes 把"必须有 TS 转储"钉死。"""
    kwargs = dict(scenes_dir=tmp_path / "no-scenes", baseline_dir=tmp_path / "baselines",
                  dist_visual=tmp_path / "visual", chapters=["0.0"], quiet=True)
    assert vr.run(mode="check", **kwargs)["ok"] is True
    strict = vr.run(mode="check", require_scenes=True, **kwargs)
    assert strict["ok"] is False and strict["totals"]["scene"] == 0
    assert strict["totals"]["failed"] == 1


def test_visual_regression_cli_json_output(tmp_path, capsys):
    code = vr.main(["--check", "--scenes-dir", str(tmp_path / "none"), "--baseline-dir", str(tmp_path / "base"),
                    "--dist-visual", str(tmp_path / "visual"), "--chapter", "0.0", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0 and payload["mode"] == "check"
    assert payload["totals"]["fallback"] == 2 and payload["ok"] is True


def test_compare_png_detects_shape_and_content(tmp_path):
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    write_rgb(a, np.zeros((10, 10, 3), dtype=np.uint8))
    write_rgb(b, np.zeros((10, 10, 3), dtype=np.uint8))
    assert vr.compare_png(a, b).same_bytes is True
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    arr[4, 5] = 255
    write_rgb(b, arr)
    report = vr.compare_png(a, b)
    assert report.different_pixels == 1 and report.bbox == (5, 4, 6, 5)
    write_rgb(b, np.zeros((12, 10, 3), dtype=np.uint8))
    assert vr.compare_png(a, b).shape_mismatch is True


# =========================================================================== #
# 10. GIF 导出
# =========================================================================== #
def test_export_video_writes_animated_gif(tmp_path):
    out = tmp_path / "demo.gif"
    code = ev.main(["--chapter", "0.0", "--viz", "heat_ftcs", "--out", str(out), "--n", "5",
                    "--fps", "20", "--width", "240", "--height", "180"])
    assert code == 0 and out.exists()
    info = read_gif(out)
    assert info.n_frames == 5 and info.delays_cs == (5,) * 5 and info.version == "GIF89a"
    frames = decode_gif_frames(out)
    assert len(frames) == 5 and frames[0].shape == (180, 240, 3)
    assert not np.array_equal(frames[0], frames[-1]), "时间参数扫过之后帧必须真的变化"


def test_export_video_png_dir(tmp_path):
    out = tmp_path / "demo.gif"
    png_dir = tmp_path / "frames"
    ev.main(["--chapter", "0.0", "--viz", "heat_ftcs", "--out", str(out), "--n", "3",
             "--width", "200", "--height", "160", "--png-dir", str(png_dir)])
    assert len(list(png_dir.glob("*.png"))) == 3


def test_sweep_values_prefers_time_param(chapter_entry):
    viz = chapter_entry.manifest.viz("heat_ftcs").to_json()
    values, name, note = ev.sweep_values(viz, 4)
    assert name == "t_end" and len(values) == 4
    assert values[0] == pytest.approx(0.005) and values[-1] == pytest.approx(0.2)
    assert "t_end" in note
    no_time = {"params": [{"name": "alpha", "type": "float"}], "time": {"mode": "none"}}
    assert ev.sweep_values(no_time, 8)[0] == [0.0]


def test_choose_palette_paths():
    frames = [np.full((4, 4, 3), 7, dtype=np.uint8), np.full((4, 4, 3), 9, dtype=np.uint8)]
    palette, note = ev.choose_palette(frames, [{"layers": []}], "auto")
    assert palette is not None and len(palette) == 2 and "无损" in note
    scene = {"layers": [{"kind": "heat", "cmap": "viridis"}]}
    palette, note = ev.choose_palette(frames, [scene], "auto")
    assert palette is not None and palette.shape[1] == 3 and len(palette) <= 256 and "viridis" in note
    assert any(np.all(palette == np.array([7, 7, 7]), axis=1)), "非热力颜色必须保留在调色板里"
    none_palette, note = ev.choose_palette(frames, [scene], "lossless")
    assert none_palette is None and "无损" in note


def test_export_video_rejects_unknown_viz(capsys):
    with pytest.raises(SystemExit):
        ev.main(["--chapter", "0.0", "--viz", "nope", "--out", "/tmp/nope.gif"])


# --------------------------------------------------------------------------- #
# 规则 8（布局）：角标盒的估算宽度与重叠/越界判定
# --------------------------------------------------------------------------- #
def test_badge_box_width_matches_the_shared_metric():
    """角标宽度用共享尺子（CJK=2、ASCII=1，1 单位 = 7px）。

    这组数字同时钉在 `viz/tests/scene.test.ts` 里（同一条样例），
    于是两侧一旦漂移就会各红一条 —— 这正是"模型以为放得下、校验器以为放不下"的解药。
    """
    assert badge_box_width("数据：后端 .nlf 帧") == pytest.approx(18 * 7 + 10)  # 18 单位
    assert badge_box_width("坐标轴：固定") == pytest.approx(12 * 7 + 10)
    # 纯 ASCII 只算 1 单位/字符
    assert badge_box_width("abcdefghij") == pytest.approx(10 * 7 + 10)


def test_validate_scene_flags_overlapping_badges():
    """同一行里的两枚长角标必须被判违规（1.2 与 0.0 章真出现过）。"""
    scene = make_scene(
        [line_layer([0.0, 1.0], [0.0, 1.0])],
        canvas=(960.0, 600.0),
        badges=[
            {"kind": "source", "text": "数据：后端 .nlf 帧", "x": 8.0, "y": 34.0},
            {"kind": "autoscale", "text": "副轴：自适应（log 纵轴按本帧 ρ 与残差取整；自适应会掩盖量级变化）", "x": 8.0, "y": 52.0},
            {"kind": "critical", "text": "dominance = 1.600　理论临界 1.000*", "x": 352.0, "y": 52.0},
        ],
    )
    issues = validate_scene(scene)
    assert any("规则 8" in i and "重叠" in i for i in issues), issues
    # 挪到独占一行 → 立刻合法（证明判据不是"有多枚角标就报错"）
    scene["badges"][2]["y"] = 70.0
    assert [i for i in validate_scene(scene) if "规则 8" in i] == []


def test_validate_scene_flags_badge_outside_canvas():
    """角标盒必须落在画布内：宽画布上的长文案挪到窄画布就会越界。"""
    scene = make_scene(
        [line_layer([0.0, 1.0], [0.0, 1.0])],
        canvas=(320.0, 240.0),
        badges=[
            {"kind": "source", "text": "数据：后端 .nlf 帧", "x": 8.0, "y": 34.0},
            {"kind": "status", "text": "很长很长" * 20, "x": 8.0, "y": 52.0},
        ],
    )
    issues = validate_scene(scene)
    assert any("超出画布" in i for i in issues), issues


def test_validate_scene_survives_malformed_badges():
    """畸形角标（缺坐标、text 非字符串）要给出诊断，而不是让校验器自己炸。"""
    scene = make_scene([line_layer([0.0, 1.0], [0.0, 1.0])])
    scene["badges"] = [{"kind": "source", "text": "数据"}, {"kind": "status", "text": None, "x": "x", "y": None}]
    issues = validate_scene(scene)
    assert any("徽标文本不得为空" in i for i in issues)
    assert any("坐标非有限" in i for i in issues)
