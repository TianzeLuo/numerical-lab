"""章节注册表：发现 ``server/chapters/<id>.py``、校验 manifest、对照 ``docs/chapters.yaml`` 报进度。

设计要点：
* **章节模块名带点（``1.2.py``）无法直接 import**，所以用 ``importlib.util.spec_from_file_location``
  以 ``numlab_chapter_1_2`` 这样的合法名字加载；
* 注册表是"进度真相"的唯一来源：`tools/gen_progress.py`、CI 门禁、派发器都读它；
* 缺什么就报什么（缺模块 / 缺 manifest / viz 数不符 / 缺 viz 文件 / 缺测试目录 / 缺笔记），
  这样"完成度"不是靠人拍脑袋，而是可计算的数。
"""

from __future__ import annotations

import functools
import importlib.util
import pathlib
import re
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping

from server.manifest import ChapterManifest, ManifestError, load_manifest

__all__ = [
    "ChapterEntry",
    "Registry",
    "RegistryError",
    "ChapterProgress",
    "default_registry",
    "load_chapter_yaml",
]

CHAPTER_FILE_RE = re.compile(r"^(\d+\.\d+)\.py$")


class RegistryError(RuntimeError):
    """注册表层面的错误（文件缺失、模块加载失败、manifest 不合规…）。"""


@dataclass
class ChapterEntry:
    """一个已加载的章节。"""

    id: str
    path: pathlib.Path
    manifest: ChapterManifest
    module: Any = None

    @property
    def title(self) -> str:
        return self.manifest.title

    @property
    def viz_ids(self) -> tuple[str, ...]:
        return self.manifest.viz_ids

    def compute(self, viz_id: str, params: Mapping[str, Any] | None = None, t: float = 0.0) -> Any:
        """调用章节模块的 ``compute`` 生成帧（延迟到真正需要时才调用）。"""
        if self.module is None or not hasattr(self.module, "compute"):
            raise RegistryError(f"章节 {self.id} 的模块没有 compute()")
        return self.module.compute(viz_id, dict(params or {}), float(t))


@dataclass
class ChapterProgress:
    """一章的完成度（每个布尔都可以在 CI 里被强制）。"""

    id: str
    title: str
    part: str = ""
    viz_declared: int = 0
    viz_implemented: int = 0
    has_module: bool = False
    manifest_ok: bool = False
    manifest_problems: list[str] = field(default_factory=list)
    has_tests: bool = False
    has_notes: bool = False
    modules_present: list[str] = field(default_factory=list)
    modules_missing: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return (
            self.has_module
            and self.manifest_ok
            and self.viz_implemented >= self.viz_declared
            and self.has_tests
            and self.has_notes
            and not self.modules_missing
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "part": self.part,
            "vizDeclared": self.viz_declared,
            "vizImplemented": self.viz_implemented,
            "hasModule": self.has_module,
            "manifestOk": self.manifest_ok,
            "hasTests": self.has_tests,
            "hasNotes": self.has_notes,
            "modulesMissing": list(self.modules_missing),
            "complete": self.complete,
        }


def load_chapter_yaml(root: pathlib.Path) -> dict[str, dict[str, Any]]:
    """读 ``docs/chapters.yaml``（章节清单），返回 ``id → 条目``。

    没有 PyYAML 时退化为"只读 id/viz/title"的极简行解析——但**绝不静默**：
    退化路径会在结果里塞 ``"_degraded": True``，让调用方知道少了字段。
    """
    path = root / "docs" / "chapters.yaml"
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
    except Exception:  # pragma: no cover - 环境无 PyYAML 时的兜底
        out: dict[str, dict[str, Any]] = {}
        for m in re.finditer(r"-\s*\{id:\s*\"([^\"]+)\",\s*title:\s*([^,]+),\s*part:\s*([^,]+)", text):
            out[m.group(1)] = {"id": m.group(1), "title": m.group(2).strip(), "part": m.group(3).strip(), "_degraded": True}
        if not out:
            raise RegistryError(f"无法解析 {path}（既没有 PyYAML，简易解析也没匹配到内容）") from None
        return out
    chapters = data.get("chapters") or []
    return {c["id"]: c for c in chapters if isinstance(c, dict) and "id" in c}


class Registry:
    """章节注册表。"""

    def __init__(self, root: pathlib.Path | str | None = None) -> None:
        self.root = pathlib.Path(root or pathlib.Path(__file__).resolve().parents[1]).resolve()
        self._entries: dict[str, ChapterEntry] = {}
        self._loaded = False

    # ------------------------------------------------------------------ #
    @property
    def chapters_dir(self) -> pathlib.Path:
        return self.root / "server" / "chapters"

    def chapter_files(self) -> list[pathlib.Path]:
        if not self.chapters_dir.exists():
            return []
        out: list[pathlib.Path] = []
        for p in sorted(self.chapters_dir.glob("*.py")):
            if p.name.startswith("_"):
                continue
            if CHAPTER_FILE_RE.match(p.name):
                out.append(p)
        return out

    def discover(self, *, strict: bool = True) -> dict[str, ChapterEntry]:
        """加载全部章节模块。``strict=True`` 时任何 manifest 问题都会抛错。"""
        entries: dict[str, ChapterEntry] = {}
        problems: list[str] = []
        for path in self.chapter_files():
            cid = CHAPTER_FILE_RE.match(path.name).group(1)  # type: ignore[union-attr]
            mod_name = f"numlab_chapter_{cid.replace('.', '_')}"
            spec = importlib.util.spec_from_file_location(mod_name, path)
            if spec is None or spec.loader is None:
                problems.append(f"{cid}: 无法为 {path} 创建 import spec")
                continue
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
            except Exception as exc:  # noqa: BLE001 - 要把原始错误原样呈现
                problems.append(f"{cid}: 导入 {path.name} 失败：{type(exc).__name__}: {exc}")
                continue
            raw = getattr(module, "MANIFEST", None)
            if raw is None:
                problems.append(f"{cid}: {path.name} 缺少模块级 MANIFEST")
                continue
            try:
                manifest = load_manifest(raw, where=f"章节 {cid}")
            except ManifestError as exc:
                problems.extend(f"{cid}: {p}" for p in exc.problems)
                continue
            if manifest.chapter != cid:
                problems.append(f"{cid}: MANIFEST['chapter']={manifest.chapter!r} 与文件名不一致")
                continue
            entries[cid] = ChapterEntry(id=cid, path=path, manifest=manifest, module=module)
        if problems and strict:
            raise RegistryError("章节注册失败：\n  - " + "\n  - ".join(problems))
        self._entries = entries
        self._loaded = True
        return entries

    # ------------------------------------------------------------------ #
    def _ensure(self) -> dict[str, ChapterEntry]:
        if not self._loaded:
            self.discover()
        return self._entries

    def ids(self) -> list[str]:
        return sorted(self._ensure(), key=_chapter_sort_key)

    def all(self) -> list[ChapterEntry]:
        return [self._ensure()[cid] for cid in self.ids()]

    def get(self, chapter_id: str) -> ChapterEntry:
        entries = self._ensure()
        if chapter_id not in entries:
            raise KeyError(f"没有章节 {chapter_id}（已加载：{', '.join(self.ids()) or '（空）'}）")
        return entries[chapter_id]

    def viz(self, chapter_id: str, viz_id: str):
        return self.get(chapter_id).manifest.viz(viz_id)

    def iter_viz(self) -> Iterator[tuple[str, str]]:
        for entry in self.all():
            for vid in entry.viz_ids:
                yield entry.id, vid

    # ------------------------------------------------------------------ #
    def viz_file(self, chapter_id: str, viz_id: str) -> pathlib.Path:
        return self.root / "viz" / "chapters" / chapter_id / f"{viz_id}.viz.ts"

    def model_file(self, chapter_id: str, viz_id: str) -> pathlib.Path:
        return self.root / "viz" / "chapters" / chapter_id / f"{viz_id}.model.ts"

    def tests_dir(self, chapter_id: str) -> pathlib.Path:
        return self.root / "tests" / chapter_id

    def notes_file(self, chapter_id: str) -> pathlib.Path:
        return self.root / "chapters" / chapter_id / "notes.tex"

    def progress(self) -> list[ChapterProgress]:
        """对照 ``docs/chapters.yaml`` 与磁盘现状，算出每章完成度。"""
        declared = load_chapter_yaml(self.root)
        out: list[ChapterProgress] = []
        for cid, info in sorted(declared.items(), key=lambda kv: _chapter_sort_key(kv[0])):
            prog = ChapterProgress(id=cid, title=str(info.get("title", "")), part=str(info.get("part", "")))
            prog.viz_declared = int(info.get("viz", 0) or 0)
            path = self.chapters_dir / f"{cid}.py"
            prog.has_module = path.exists()
            if prog.has_module:
                try:
                    spec = importlib.util.spec_from_file_location(f"numlab_probe_{cid.replace('.', '_')}", path)
                    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
                    spec.loader.exec_module(module)  # type: ignore[union-attr]
                    raw = getattr(module, "MANIFEST", None)
                    if raw is None:
                        prog.manifest_problems = ["缺少 MANIFEST"]
                    else:
                        from server.manifest import validate_manifest

                        prog.manifest_problems = validate_manifest(raw, where=f"章节 {cid}")
                        if not prog.manifest_problems:
                            manifest = load_manifest(raw)
                            prog.viz_implemented = sum(
                                1 for vid in manifest.viz_ids if self.viz_file(cid, vid).exists()
                            )
                except Exception as exc:  # noqa: BLE001
                    prog.manifest_problems = [f"导入失败：{type(exc).__name__}: {exc}"]
                prog.manifest_ok = prog.has_module and not prog.manifest_problems
            prog.has_tests = self.tests_dir(cid).is_dir() and any(self.tests_dir(cid).glob("test_*.py"))
            prog.has_notes = self.notes_file(cid).exists()
            for mod in info.get("modules", []) or []:
                rel = str(mod)
                target = self.root / (rel if rel.endswith(".py") else f"{rel}.py")
                if not rel.endswith(".py") and "/" in rel and (self.root / rel).is_dir():
                    target = self.root / rel
                if target.exists():
                    prog.modules_present.append(rel)
                else:
                    prog.modules_missing.append(rel)
            out.append(prog)
        return out

    def summary(self) -> dict[str, Any]:
        """全项目进度汇总（``tools/gen_progress.py`` 与 CI 用）。"""
        prog = self.progress()
        total_viz = sum(p.viz_declared for p in prog)
        done_viz = sum(min(p.viz_implemented, p.viz_declared) for p in prog)
        return {
            "chapters_declared": len(prog),
            "chapters_complete": sum(1 for p in prog if p.complete),
            "vizzes_declared": total_viz,
            "vizzes_implemented": done_viz,
            "chapters": [p.to_json() for p in prog],
        }


def _chapter_sort_key(cid: str) -> tuple[int, int]:
    major, minor = cid.split(".")
    return int(major), int(minor)


@functools.lru_cache(maxsize=1)
def default_registry() -> Registry:
    """进程内共享的注册表（发现一次，缓存）。"""
    reg = Registry()
    reg.discover()
    return reg
