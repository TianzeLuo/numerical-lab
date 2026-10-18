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
import time
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
    """生成 ``viz/chapters/index.gen.ts``：章节 id → { manifest, models } 的**惰性**登记表。

    为什么是**动态 import** 而不是静态 import（这里踩过真坑）：静态 import 会让
    "某一章的 model 文件还没写完"变成**整包加载失败**——`import` 断链时
    `npm run scene -- --chapter 1.1` 也起不来（`ERR_MODULE_NOT_FOUND` 发生在
    `index.gen.ts` 求值那一刻，`--chapter` 已经来不及救了）。并发开发下每个前端
    agent 都会撞上这堵墙。改成动态 import + try/catch 之后：

    * 访问哪一章才加载哪一章，坏掉的兄弟章节只会让**它自己**加载失败并打印一行警告；
    * `chapterIds` 仍然是静态常量（从磁盘上"有 ``index.ts``"的章节生成），
      站点侧边栏不需要加载任何 model 就能知道哪些章可点。

    ``loadChapter`` 会缓存结果（同一章只 import 一次）；失败不缓存（下次可以重试）。
    """
    ready = [m for m in manifests if (root / "viz" / "chapters" / m.chapter / "index.ts").exists()]
    ids = ", ".join(f'"{m.chapter}"' for m in ready)
    lines = [
        GEN_HEADER,
        'import type { ChapterManifestJson } from "../shared/src/manifest.js";',
        'import type { SceneModel } from "../shared/src/model.js";',
        "",
        "/** 一章的前端装配：manifest（契约）+ models（viz id → Frame→Scene） */",
        "export interface ChapterModels {",
        "  manifest: ChapterManifestJson;",
        "  models: Record<string, SceneModel>;",
        "}",
        "",
        "/** 已经实现可视化的章节 id（未实现的不在表里；判断「能不能点」看它，不需要加载 model） */",
        f"export const chapterIds: string[] = [{ids}];",
        "",
        "/** 章节 id → 加载器（动态 import：坏掉的兄弟章节不会拖垮整包） */",
        "const loaders: Record<string, () => Promise<ChapterModels>> = {",
    ]
    for m in ready:
        lines.append(f'  "{m.chapter}": () => import("./{m.chapter}/index.js"),')
    lines += [
        "};",
        "",
        "const cache = new Map<string, ChapterModels>();",
        "",
        "/** 加载一章的前端装配；没有实现或加载失败时返回 undefined（并打印一行警告）。 */",
        "export async function loadChapter(id: string): Promise<ChapterModels | undefined> {",
        "  const hit = cache.get(id);",
        "  if (hit) return hit;",
        "  const load = loaders[id];",
        "  if (!load) return undefined;",
        "  try {",
        "    const bundle = await load();",
        "    cache.set(id, bundle);",
        "    return bundle;",
        "  } catch (err) {",
        "    console.warn(`[chapters] 章节 ${id} 的前端装配加载失败：${(err as Error).message}`);",
        "    return undefined;",
        "  }",
        "}",
        "",
        "/** 按 id 升序加载所有已实现章节（失败的跳过，返回值只含成功的）。 */",
        "export async function loadAllChapters(): Promise<Map<string, ChapterModels>> {",
        "  const out = new Map<string, ChapterModels>();",
        "  for (const id of chapterIds) {",
        "    const bundle = await loadChapter(id);",
        "    if (bundle) out.set(id, bundle);",
        "  }",
        "  return out;",
        "}",
        "",
        "export default loadChapter;",
        "",
    ]
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
    ap.add_argument(
        "--report",
        type=pathlib.Path,
        default=ROOT / "dist" / "reports" / "manifests.json",
        help="证据文件（docs/PROGRESS.md 的门禁表读它）",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    root: pathlib.Path = args.root
    registry = Registry(root)
    # 生成模式用 strict=False 并**打印**跳过的章节：写完一章的 agent 不该被兄弟章节
    # 的语法错误挡住；但 --check 仍然严格（那才是门禁要的"一个都不能坏"）。
    skipped: list[str] = []
    if args.check:
        registry.discover()
    else:
        registry.discover(strict=False)
        known = set(registry.ids())
        for path in registry.chapter_files():
            cid = path.name[: -len(".py")]
            if cid not in known:
                skipped.append(cid)
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

    if skipped and not args.json:
        print(f"[警告] 以下章节当前无法加载、已跳过（修好后重跑即可）：{', '.join(skipped)}")
    ok = not drift and not missing
    # 证据落盘：PROTOCOLS §9 把 dist/reports/manifests.json 列为 G6 的证据，
    # 但之前**没有任何写入者**，于是 docs/PROGRESS.md 的门禁表永远显示"未运行"。
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {
                    "ok": ok,
                    "mode": "check" if args.check else "write",
                    "files": [str(p) for p in sorted(files)],
                    "drift": drift,
                    "missing": missing,
                    "skipped": skipped,
                    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
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
