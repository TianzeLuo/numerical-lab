"""仓库布局不变量：几件"一旦破了就会让整套门禁失灵"的事。

为什么要单独一个文件：这些约束不属于某一章，而是**仓库级**的。

最典型的是重名测试模块：pytest 的默认导入方式（`prepend`，按模块名导入，`rootdir` 下
没有 `__init__.py` 的目录不会变成包）会让 `tests/1.2/test_frames.py` 与
`tests/4.1/test_frames.py` 撞在同一个模块名 `test_frames` 上，报
``import file mismatch``，**整轮收集直接失败**（真踩过两次：2.4/1.2 与 4.1/1.2）。

两条修法只能选一条，本文件把"两条都没做"变成红灯：
1. `pyproject.toml` 的 `addopts` 里带 `--import-mode=importlib`（**本仓库的选择**）：
   pytest 直接按文件路径导入，重名无害（章节目录名是 `1.2` 这种带点的编号，
   刚好也不能当包名，所以塞 `__init__.py` 那条路走不通）；
2. 或者给测试目录加 `__init__.py`（代价见上：编号目录名不是合法包名）。
"""

from __future__ import annotations

import collections
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server.registry import Registry  # noqa: E402


def _pytest_addopts() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    hit = re.search(r'^\s*addopts\s*=\s*"([^"]*)"', text, re.MULTILINE)
    return hit.group(1) if hit else ""


def test_pytest_import_mode_allows_duplicate_basenames() -> None:
    """要么开 `--import-mode=importlib`，要么 `tests/**` 里不许有重名测试模块。"""
    importlib_mode = "--import-mode=importlib" in _pytest_addopts()
    seen: dict[str, list[str]] = collections.defaultdict(list)
    for path in sorted((ROOT / "tests").rglob("test_*.py")):
        if (path.parent / "__init__.py").exists():
            continue
        seen[path.name].append(str(path.relative_to(ROOT)))
    dupes = {name: files for name, files in seen.items() if len(files) > 1}
    if importlib_mode:
        return  # 重名无害：pytest 按路径导入
    assert not dupes, (
        "以下测试模块重名，`pytest tests` 会因 import file mismatch 整轮失败："
        + "; ".join(f"{name} → {files}" for name, files in sorted(dupes.items()))
        + "。修法：在 pyproject.toml 的 addopts 里加 `--import-mode=importlib`（推荐），"
        "或给这些目录加 __init__.py。"
    )


def test_duplicate_basenames_are_actually_collectible() -> None:
    """守"importlib 模式真的开着"这件事本身：重名存在时也必须能收集（而不是假定）。"""
    if "--import-mode=importlib" not in _pytest_addopts():
        return
    seen: dict[str, list[str]] = collections.defaultdict(list)
    for path in sorted((ROOT / "tests").rglob("test_*.py")):
        seen[path.name].append(str(path.relative_to(ROOT)))
    dupes = {name: files for name, files in seen.items() if len(files) > 1}
    assert dupes, (
        "本测试是为「重名 + importlib」这条组合写的；现在没有重名，说明它失去了守的对象"
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
