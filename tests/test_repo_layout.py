"""仓库布局不变量：几件"一旦破了就会让整套门禁失灵"的事。

为什么要单独一个文件：这些约束不属于某一章，而是**仓库级**的——
最典型的是重名测试模块。pytest 默认按模块名导入（`rootdir` 下没有 `__init__.py` 的目录
不会变成包），于是 `tests/1.2/test_frames.py` 与 `tests/test_frames.py`
会在 `pytest tests` 时报
``import file mismatch: imported module 'test_frames' has this __file__ attribute ...``
——整轮收集直接失败（真踩过：1.2 章的帧测试一落地，全仓 pytest 就红了）。
"""

from __future__ import annotations

import collections
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server.registry import Registry  # noqa: E402


def test_no_duplicate_test_module_basenames() -> None:
    """`tests/**` 下不得有两个同名的 `test_*.py`（pytest 会撞模块名）。

    唯一的重名豁免：目录里带 `__init__.py`（那才会被当成包，模块名带上前缀）。
    """
    seen: dict[str, list[str]] = collections.defaultdict(list)
    for path in sorted((ROOT / "tests").rglob("test_*.py")):
        if (path.parent / "__init__.py").exists():
            continue
        seen[path.name].append(str(path.relative_to(ROOT)))
    dupes = {name: files for name, files in seen.items() if len(files) > 1}
    assert not dupes, (
        "以下测试模块重名，`pytest tests` 会因 import file mismatch 整轮失败："
        + "; ".join(f"{name} → {files}" for name, files in sorted(dupes.items()))
        + "。改名（例如给仓库级文件加限定词）或在目录里放 __init__.py。"
    )


def test_tests_directories_match_registry_chapters() -> None:
    """`tests/<章号>/` 必须对应一个真实存在的章节；反过来，章节不强制要求本目录。

    "章节不强制"是刻意的：Python 层的测试可能全写在 `tests/<章号>/` 里，
    也可能暂时整章未开工；这里只拦"目录名拼错、指向不存在的章"这种错。
    """
    reg = Registry(ROOT)
    reg.discover(strict=False)
    known = set(reg.ids())
    dirs = {
        p.name
        for p in (ROOT / "tests").iterdir()
        if p.is_dir() and p.name not in {"__pycache__", "baselines", "fixtures", "parity", "policy"}
    }
    unknown = sorted(d for d in dirs if d not in known)
    assert not unknown, f"tests/ 下有目录不对应任何已加载章节：{unknown}（已加载：{sorted(known)}）"


def test_generated_artifacts_are_not_tracked_sources() -> None:
    """`data/frames/**` 是缓存，不能进版本库（`.gitignore` 必须挡住它）。

    帧缓存动辄几百 MB，而且可由 `tools/precompute.py` 逐字节重建；
    一旦被 `git add` 进来，仓库会瞬间膨胀且与生成器不同步。
    """
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "data/frames" in ignore, ".gitignore 必须忽略 data/frames（预计算缓存，可重建）"
