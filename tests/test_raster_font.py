"""内嵌位图字体的覆盖率：PNG 基线里不许出现"缺字框"。

为什么值得一条测试（而不是靠肉眼看基线）：`tools/raster.py` 的画布没有系统字体可用，
汉字子集是**烘焙时收集到的字符**。收集范围漏掉一整个目录（`viz/chapters/**`）时，
每一章前端新写的中文都会在 PNG 里变成方框——浏览器 canvas 却完全正常，于是"基线全绿"
这件事本身会骗人。实测漏过 220 个字符（2.1 章一节就 40 余字）。

这条测试的口径与 `tools/bake_font.py --check` 一致：**场景里真正会被画出来的字符**
都必须有字形（场景由 `npm run scene` 转储到 `dist/scenes/`；没转储时跳过，CI 里 G8 先跑）。
缺字时的修法写在 `tools/bake_font.py` 的模块 docstring 里：`--write` 重烘焙 + 重画基线。
"""

from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import bake_font  # noqa: E402


def test_embedded_font_parses_and_has_both_kinds_of_strikes() -> None:
    ascii_strikes, cjk_strikes = bake_font.parse_blob(bake_font.read_blob(ROOT))
    assert sorted(ascii_strikes) == [11, 12, 13, 18], "ASCII 四档几何是排版契约的一部分（字号→档位最近邻）"
    assert sorted(cjk_strikes) == [14], "汉字档只有 14px 一档（渲染时按最近邻缩放到 ASCII 档）"
    for size, (w, h, base, ink) in ascii_strikes.items():
        assert ink, f"{size}px 档没有字形"
        assert 0 < base <= h and w > 0
    for size, (w, h, base, ink) in cjk_strikes.items():
        assert ink and w >= 2 * 7, "汉字全宽（≥ 2 个 ASCII 单元）"
    # 常用字必须真在里面（否则基线里那一章的中文全是框）
    cjk_ink = cjk_strikes[14][3]
    for ch in "数值方法实验室误差收敛稳定":
        assert ord(ch) in cjk_ink, f"汉字档缺 {ch!r}——请重烘焙字体"


def test_scene_text_has_no_missing_glyphs() -> None:
    scenes = sorted((ROOT / "dist" / "scenes").rglob("*.json"))
    if not scenes:
        pytest.skip("还没有转储场景（先跑 npm run scene / G8）")
    used = bake_font.used_codepoints(ROOT)
    bad = bake_font.missing(used, ROOT)
    assert not bad, (
        "场景里出现了内嵌字体没有字形的字符（PNG 里会画成缺字框）："
        + "".join(chr(cp) for cp in bad[:40])
        + f"（共 {len(bad)} 个）——修法见 tools/bake_font.py 的模块 docstring"
    )


def test_every_committed_baseline_has_a_scene_or_is_documented() -> None:
    """基线目录与场景目录要对得上：没有场景的基线只可能是"别章的兜底路径"（这里不许有）。"""
    baseline_root = ROOT / "tests" / "baselines"
    if not baseline_root.exists():
        pytest.skip("还没有基线")
    scene_root = ROOT / "dist" / "scenes"
    if not scene_root.exists():
        pytest.skip("还没有转储场景")
    missing: list[str] = []
    for chapter_dir in sorted(p for p in baseline_root.iterdir() if p.is_dir()):
        if not (scene_root / chapter_dir.name).exists():
            missing.append(chapter_dir.name)
    assert not missing, f"这些章有基线却没有转储下来的场景（基线无法复现）：{missing}"
