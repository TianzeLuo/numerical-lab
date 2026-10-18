"""全仓帧契约测试：每个章节的每个 viz 都必须能**编码成 .nlf**。

为什么需要一条全仓测试
----------------------
章节自己的测试只看自己那一章，而"帧能不能编码"这件事的失败模式是**跨章同构**的：

* `meta` 里塞了 NaN/Infinity → JSON 头用 ``allow_nan=False`` 编码，**整帧**失败；
* `requiredAux` 声明的名字在帧里不存在 → 前端的"必备对照量"面板静默空着（防误导规则 3）；
* `time.mode="none"` 却声明了 ``tMax`` → 前端会画出一条根本不存在的进度条；
* 参数取到边界就跑飞（除零、奇异、溢出）→ 用户一拖滑块就白屏。

这些都在 `server/frames.py::build_frame(strict=True)` 与 `server/nlf.py::Frame.validate()`
里被判，但**必须有东西真的去调用它们**。这个文件就是那个"东西"：它不重复数值断言，
只保证"每一帧都真的能落地"。

采样策略（速度与覆盖的折中）
----------------------------
每个 viz 只取 **默认参数** 与 **时间轴两端**（``t=0`` / ``t=tMax``），外加每个参数各取一次
最小/最大边界（枚举取第一个与最后一个选项）。整仓 100 多个 viz 走下来是秒级，
而它能抓到的正是"用户一上来就会碰到"的那些点。
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server import nlf as nlf_mod  # noqa: E402
from server.frames import build_frame  # noqa: E402
from server.registry import Registry  # noqa: E402


def _registry() -> Registry:
    reg = Registry(ROOT)
    reg.discover()  # 严格：CI 里任何一章坏了都该红
    return reg


def _param_corners(spec: dict) -> list[dict]:
    """默认参数 + 每个参数各取一次边界（组合爆炸就到这里为止）。"""
    base = {p["name"]: p["default"] for p in spec.get("params", [])}
    out: list[dict] = [dict(base)]
    for p in spec.get("params", []):
        name = p["name"]
        kind = p.get("type")
        if kind == "enum":
            options = list(p.get("options", []))
            if len(options) >= 2:
                out.append({**base, name: options[0]})
                out.append({**base, name: options[-1]})
        elif kind in ("int", "float") and "min" in p and "max" in p:
            out.append({**base, name: p["min"]})
            out.append({**base, name: p["max"]})
        elif kind == "bool":
            out.append({**base, name: not base.get(name, False)})
    # 去重（用 repr 做键：参数都是标量）
    seen: set[str] = set()
    uniq: list[dict] = []
    for item in out:
        key = repr(sorted(item.items()))
        if key not in seen:
            seen.add(key)
            uniq.append(item)
    return uniq


def _times(spec: dict) -> list[float]:
    time = spec.get("time") or {}
    if time.get("mode") == "none":
        return [0.0]
    t_max = float(time.get("tMax", 0.0) or 0.0)
    return [0.0, t_max] if t_max > 0.0 else [0.0]


def _iter_viz():
    """产出 ``(章节号, 注册表条目, viz 的**原始字典**)``。

    用 ``entry.module.MANIFEST`` 而不是 ``entry.manifest.vizzes``（后者是解析后的
    `VizSpec` 对象）：本文件要检查的正是"作者写的那些字段"，用原始字典才能保证
    检查的是真源本身，而不是解析器愿意暴露给我的那一部分。
    """
    reg = _registry()
    for cid in reg.ids():
        entry = reg.get(cid)
        raw = getattr(entry.module, "MANIFEST", None) or {}
        for spec in raw.get("vizzes", []):
            yield cid, entry, spec

    return None


@pytest.fixture(scope="module")
def viz_list() -> list[tuple[str, object, dict]]:
    return list(_iter_viz())


def test_registry_has_vizzes(viz_list: list[tuple[str, object, dict]]) -> None:
    """先确认采样器真的拿到了东西（否则下面的测试会"零个 case 全绿"）。"""
    assert viz_list, "注册表里一个 viz 都没有：测试会静默通过，必须在这里失败"
    chapters = {cid for cid, _, _ in viz_list}
    assert "0.0" in chapters


def test_every_viz_packs_at_default_and_corners(viz_list: list[tuple[str, object, dict]]) -> None:
    """每个 viz 在默认参数与各参数边界上都必须能 build_frame + pack_frame。"""
    problems: list[str] = []
    n_frames = 0
    for cid, entry, spec in viz_list:
        for params in _param_corners(spec):
            for t in _times(spec):
                n_frames += 1
                label = f"{cid}/{spec['id']} params={params} t={t}"
                try:
                    frame = build_frame(entry, spec["id"], params, t, strict=True)
                    nlf_mod.pack_frame(frame)
                    # 数组里也不能有 NaN/Inf：`.nlf` 的数组段是裸 float64，**编码得进去**，
                    # 于是问题会跑到前端——一条 NaN 折线在画布上就是"这段数据凭空消失"，
                    # 而且不报错（2.1 章 agent 提醒的正是这一点：strict 只查 meta 的有限性）。
                    for spec_arr, arr in nlf_mod.iter_arrays(frame):
                        if arr.size and not np.all(np.isfinite(arr)):
                            n_bad = int(np.count_nonzero(~np.isfinite(arr)))
                            problems.append(f"{label}: 数组 {spec_arr.name} 有 {n_bad}/{arr.size} 个非有限值")
                except Exception as exc:  # noqa: BLE001 - 要把每一处失败都收集起来
                    problems.append(f"{label}: {type(exc).__name__}: {exc}")
    assert not problems, f"{len(problems)}/{n_frames} 帧无法编码：\n  - " + "\n  - ".join(problems[:20])


def test_required_aux_names_exist_in_frames(viz_list: list[tuple[str, object, dict]]) -> None:
    """`requiredAux` 声明的名字必须在帧里真的存在（防误导规则 3）。

    判定口径与 `server/frames.py::build_frame(strict=True)` 一致：**大小写无关的子串匹配**
    （声明的 ``error`` 可以由 ``error_curve`` 满足）。这里额外要求"至少有一帧真的带上它"，
    避免"声明了但只在某个参数角落才出现"这种半残状态。
    """
    problems: list[str] = []
    for cid, entry, spec in viz_list:
        wanted = list(spec.get("requiredAux") or [])
        if not wanted:
            problems.append(f"{cid}/{spec['id']}: requiredAux 为空（规则 3）")
            continue
        names: set[str] = set()
        for params in _param_corners(spec):
            frame = build_frame(entry, spec["id"], params, _times(spec)[0], strict=True)
            names |= {n.lower() for n in frame.names}
        for aux in wanted:
            if not any(aux.lower() in n for n in names):
                problems.append(f"{cid}/{spec['id']}: requiredAux 里的 {aux!r} 在任何一帧里都没有对应数组")
    assert not problems, "\n  - ".join(["requiredAux 与帧不一致："] + problems)


def test_time_axis_declarations_are_consistent(viz_list: list[tuple[str, object, dict]]) -> None:
    """`time.mode="none"` 不许声明 tMax/dt/fps；其余模式必须给出正的 tMax。"""
    problems: list[str] = []
    for cid, _, spec in viz_list:
        time = spec.get("time") or {}
        mode = time.get("mode")
        if mode == "none":
            for key in ("tMax", "dt", "fps"):
                if time.get(key) is not None:
                    problems.append(f"{cid}/{spec['id']}: mode=none 却声明了 {key}")
        else:
            t_max = float(time.get("tMax", 0.0) or 0.0)
            if t_max <= 0.0:
                problems.append(f"{cid}/{spec['id']}: mode={mode} 的 tMax 必须是正数，得到 {t_max}")
    assert not problems, "\n  - ".join(["时间轴声明不一致："] + problems)
