#!/usr/bin/env python3
"""把章节模块里的 ``MANIFEST``（真源）生成两份产物：

* ``data/manifests/<id>.json``            —— 给静态降级路径与工具读；
* ``viz/chapters/<id>/manifest.gen.ts``   —— 给前端 import（带类型）；
* ``data/manifests/index.json``           —— 全书目录（28 章 / 115 可视化）；
* ``viz/shared/src/chapters.gen.ts``      —— 侧边栏用的目录常量。

G6 门：``--check`` 重新生成一遍并与磁盘比对，任何漂移都算失败。
生成必须是**字节可复现**的：键排序、浮点用 Python repr（最短往返）、行尾 LF、末尾换行。

用法::

    python3 tools/gen_manifests.py            # 写入
    python3 tools/gen_manifests.py --check    # 只检查漂移（CI）
    python3 tools/gen_manifests.py --json     # 机器可读结果
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Iterable

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.manifest import ChapterManifest  # noqa: E402
from server.registry import Registry  # noqa: E402

GEN_HEADER = "// 本文件由 tools/gen_manifests.py 从章节模块的 MANIFEST 生成，请勿手改。\n"


def _json_text(obj: Any) -> str:
    """确定性的 JSON 文本：键排序、缩进 2、结尾换行、非 ASCII 原样。"""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def _ts_literal(obj: Any, indent: int = 0) -> str:
    """把 JSON 值渲染成 TS 字面量（对象键加引号，保证任何键名都合法）。"""
    pad = "  " * indent
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        items = []
        for key in sorted(obj):
            items.append(f"{pad}  {json.dumps(key, ensure_ascii=False)}: {_ts_literal(obj[key], indent + 1)}")
        return "{\n" + ",\n".join(items) + f",\n{pad}}}"
    if isinstance(obj, list):
        if not obj:
            return "[]"
        if all(isinstance(x, str) for x in obj):
            return "[" + ", ".join(json.dumps(x, ensure_ascii=False) for x in obj) + "]"
        items = [f"{pad}  {_ts_literal(x, indent + 1)}" for x in obj]
        return "[\n" + ",\n".join(items) + f",\n{pad}]"
    return json.dumps(obj, ensure_ascii=False)


def manifest_ts(man: ChapterManifest) -> str:
    body = _ts_literal(man.to_json(), 0)
    return (
        GEN_HEADER
        + 'import type { ChapterManifestJson } from "../../shared/src/manifest.js";\n\n'
        + f"/** {man.title} */\n"
        + f"export const manifest: ChapterManifestJson = {body};\n\n"
        + "export default manifest;\n"
    )


def index_json(manifests: list[ChapterManifest], registry: Registry) -> dict[str, Any]:
    chapters = []
    for man in manifests:
        chapters.append(
            {
                "chapter": man.chapter,
                "title": man.title,
                "part": man.part or _part_of(man.chapter, registry),
                "summary": man.summary or "",
                "nViz": len(man.vizzes),
                "vizzes": [v.id for v in man.vizzes],
                "hasModule": True,
                "notesFile": f"chapters/{man.chapter}/notes.tex",
            }
        )
    return {
        "count": len(chapters),
        "totalViz": sum(c["nViz"] for c in chapters),
        "chapters": chapters,
    }


def models_registry_ts(root: pathlib.Path, manifests: list[ChapterManifest]) -> str:
    """生成 ``viz/chapters/index.gen.ts``：章节 id → { manifest, models } 的静态登记表。

    为什么必须是**生成的静态 import**：浏览器里没有 require/动态文件枚举，
    而 `dump_scenes.ts` 与站点都按 id 找 model。只登记"已经有 ``index.ts``"的章节——
    否则一个还没写可视化的章节会让整包编译失败。
    """
    ready = [m for m in manifests if (root / "viz" / "chapters" / m.chapter / "index.ts").exists()]
    lines = [
        GEN_HEADER,
        'import type { ChapterManifestJson } from "../shared/src/manifest.js";',
        'import type { SceneModel } from "../shared/src/model.js";',
    ]
    for m in ready:
        alias = "c_" + m.chapter.replace(".", "_")
        lines.append(
            f'import {{ manifest as manifest_{alias}, models as models_{alias} }} from "./{m.chapter}/index.js";'
        )
    lines += [
        "",
        "/** 一章的前端装配：manifest（契约）+ models（viz id → Frame→Scene） */",
        "export interface ChapterModels {",
        "  manifest: ChapterManifestJson;",
        "  models: Record<string, SceneModel>;",
        "}",
        "",
        "/** 已经实现可视化的章节（未实现的不在表里，调用方要处理 undefined） */",
        "export const chapterModels: Record<string, ChapterModels> = {",
    ]
    for m in ready:
        alias = "c_" + m.chapter.replace(".", "_")
        lines.append(f'  "{m.chapter}": {{ manifest: manifest_{alias}, models: models_{alias} }},')
    lines += ["};", "", "export default chapterModels;", ""]
    return "\n".join(lines)


def chapters_ts(idx: dict[str, Any]) -> str:
    return (
        GEN_HEADER
        + 'import type { ChapterIndexEntry } from "./chapters.js";\n\n'
        + "export const chapterIndex: ChapterIndexEntry[] = "
        + _ts_literal(
            [
                {
                    "chapter": c["chapter"],
                    "title": c["title"],
                    "part": c["part"],
                    "summary": c["summary"],
                    "nViz": c["nViz"],
                    "vizzes": c["vizzes"],
                }
                for c in idx["chapters"]
            ],
            0,
        )
        + ";\n\nexport default chapterIndex;\n"
    )


def _part_of(chapter: str, registry: Registry) -> str:
    for row in registry.progress():
        if row.id == chapter:
            return row.part
    return ""


def build(registry: Registry) -> dict[pathlib.Path, str]:
    """返回 {相对路径: 内容}；不写盘。"""
    manifests: list[ChapterManifest] = []
    out: dict[pathlib.Path, str] = {}
    for cid in registry.ids():
        entry = registry.get(cid)
        man = entry.manifest
        manifests.append(man)
        rel = pathlib.Path("data") / "manifests" / f"{cid}.json"
        out[rel] = _json_text(man.to_json())
        rel_ts = pathlib.Path("viz") / "chapters" / cid / "manifest.gen.ts"
        out[rel_ts] = manifest_ts(man)
    idx = index_json(manifests, registry)
    out[pathlib.Path("data/manifests/index.json")] = _json_text(idx)
    out[pathlib.Path("viz/shared/src/chapters.gen.ts")] = chapters_ts(idx)
    out[pathlib.Path("viz/chapters/index.gen.ts")] = models_registry_ts(registry.root, manifests)
    return out


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成章节 manifest 产物（G6 真源单一）")
    ap.add_argument("--check", action="store_true", help="只检查是否漂移，不写盘")
    ap.add_argument("--json", action="store_true", help="机器可读输出")
    ap.add_argument("--root", type=pathlib.Path, default=ROOT, help="仓库根（测试用）")
    args = ap.parse_args(list(argv) if argv is not None else None)

    root: pathlib.Path = args.root
    registry = Registry(root)
    files = build(registry)

    drift: list[str] = []
    missing: list[str] = []
    for rel, text in sorted(files.items()):
        path = root / rel
        if args.check:
            if not path.exists():
                missing.append(str(rel))
            elif path.read_text(encoding="utf-8") != text:
                drift.append(str(rel))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8", newline="\n")

    ok = not drift and not missing
    if args.json:
        print(json.dumps({"ok": ok, "written": [str(p) for p in sorted(files)], "drift": drift, "missing": missing}, ensure_ascii=False))
    else:
        if args.check:
            if ok:
                print(f"manifest 生成物与真源一致：{len(files)} 个文件")
            else:
                for p in missing:
                    print(f"  缺失：{p}")
                for p in drift:
                    print(f"  漂移：{p}")
                print(f"请运行 python3 tools/gen_manifests.py 重新生成（{len(drift)} 漂移 / {len(missing)} 缺失）")
        else:
            print(f"已生成 {len(files)} 个 manifest 产物")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
