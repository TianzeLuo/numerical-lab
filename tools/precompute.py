#!/usr/bin/env python3
"""L3 静态预计算：把章节的帧预先算好写到 ``data/frames/...``（PROTOCOLS §4 的第三层降级）。

用法::

    python3 tools/precompute.py [--chapter 0.0] [--n 60] [--limit N] [--out-root DIR] [--prune]

* ``--chapter`` 可重复；缺省生成**注册表里的全部章节**。
* ``--n`` 每个可视化沿时间轴生成的帧数（缺省 1 = 只生成关键参数点/首帧；
  ``time.mode="none"`` 的 viz 只会有 1 帧，与 ``server.frames.precompute_chapter`` 一致）。
* ``--limit`` 总帧数上限（CI 用：一个手滑不至于写出几万个文件）。
* ``--out-root`` 输出根，布局为 ``<out-root>/data/frames/<chapter>/<viz>/<hash>_t<t>.nlf``
  （缺省仓库根；``server.frames.static_frame_path`` 就是这么拼的）。
* ``--prune`` 删除没有对应章节/可视化的**陈旧帧**目录（只清理 ``data/frames`` 之下）。

为什么要单独一个工具：浏览器端算不动 PDE/FEM（``docs/CHAPTERS.md`` §8 风险 1），
而"降级路径上禁止重算物理量"（P8）意味着这些帧必须**由后端算好、原样落盘**。
落盘文件名是参数指纹（``params_hash``），因此重复运行是幂等的、可以只补不删。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys
from typing import Any, Sequence

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.frames import precompute_chapter  # noqa: E402
from server.registry import Registry, RegistryError  # noqa: E402


def data_frames_dir(out_root: pathlib.Path) -> pathlib.Path:
    """``<out_root>/data/frames`` —— 与 ``server.frames.static_frame_path`` 同一约定。"""
    return out_root / "data" / "frames"


def prune_stale(
    out_root: pathlib.Path,
    registry: Registry,
    chapters: Sequence[str] | None = None,
    *,
    dry_run: bool = False,
) -> tuple[list[pathlib.Path], int]:
    """删掉 ``data/frames`` 下没有对应章节/可视化的目录，返回 ``(被删目录, 文件数)``。

    判据只有一条：目录名在**注册表**里存在吗（``chapters``/``viz``）。``chapters`` 参数
    用于把清理范围再收窄到给定章节（测试与"只收拾这一章"的场景用），``None`` = 全库。

    删除前逐个校验：目标必须**严格位于** ``data/frames`` 之内。这条检查不是形式主义——
    清理工具一旦算错路径就是不可逆的数据丢失。
    """
    base = data_frames_dir(out_root)
    if not base.is_dir():
        return [], 0
    base_real = base.resolve()
    known: dict[str, set[str]] = {cid: set(registry.get(cid).viz_ids) for cid in registry.ids()}
    scope = set(chapters) if chapters else None

    doomed: list[pathlib.Path] = []
    for chapter_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        cid = chapter_dir.name
        if scope is not None and cid not in scope:
            continue
        if cid not in known:
            doomed.append(chapter_dir)
            continue
        for viz_dir in sorted(p for p in chapter_dir.iterdir() if p.is_dir()):
            if viz_dir.name not in known[cid]:
                doomed.append(viz_dir)

    removed_files = 0
    for path in doomed:
        real = path.resolve()
        if real == base_real or base_real not in real.parents:
            raise RegistryError(f"拒绝删除 {real}：不在 {base_real} 之内")
        removed_files += sum(1 for p in real.rglob("*") if p.is_file())
        if not dry_run:
            shutil.rmtree(real)
    if not dry_run:
        # 清掉被删空的章节目录与空的 data/frames
        for candidate in sorted((p for p in base.iterdir() if p.is_dir()), key=lambda p: -len(p.parts)):
            try:
                if not any(candidate.iterdir()):
                    candidate.rmdir()
            except OSError:  # pragma: no cover - 并发写盘的竞态
                pass
    return doomed, removed_files


def _human_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} GiB"  # pragma: no cover - 循环里已覆盖


def run(args: argparse.Namespace) -> dict[str, Any]:
    """执行一次预计算（可被测试直接调用，不经过 argparse）。"""
    out_root = pathlib.Path(args.out_root).resolve() if args.out_root else ROOT
    registry = Registry(ROOT)
    registry.discover()
    chapters: list[str] = list(args.chapter) if args.chapter else registry.ids()
    unknown = [c for c in chapters if c not in set(registry.ids())]
    if unknown:
        raise RegistryError(f"未知章节 {unknown}；注册表里有：{', '.join(registry.ids()) or '（空）'}")

    written: list[pathlib.Path] = []
    per_chapter: list[dict[str, Any]] = []
    for cid in chapters:
        remaining = None if args.limit is None else max(0, args.limit - len(written))
        if remaining == 0:
            break
        paths = precompute_chapter(
            registry, cid, out_root=out_root, frames_per_viz=args.n, limit=remaining
        )
        written.extend(paths)
        total = sum(p.stat().st_size for p in paths)
        per_chapter.append({"chapter": cid, "files": len(paths), "bytes": total})
        if not args.quiet:
            print(f"{cid}: {len(paths)} 帧，{_human_bytes(total)}")

    pruned: list[str] = []
    pruned_files = 0
    if args.prune:
        # 全局清理（不随 --chapter 收窄）：判据是"这个章节/可视化在注册表里存在吗"，
        # 与本次生成哪几章无关——陈旧产物恰恰常常来自已被改名/删除的 viz。
        doomed, pruned_files = prune_stale(out_root, registry, None)
        pruned = [str(p) for p in doomed]
        if not args.quiet:
            print(f"prune: 删除 {len(pruned)} 个陈旧目录，{pruned_files} 个文件")

    total_bytes = sum(p.stat().st_size for p in written)
    if not args.quiet:
        print(f"合计：{len(written)} 个文件，{total_bytes} 字节（{_human_bytes(total_bytes)}）")
        print(f"输出根：{out_root}（{data_frames_dir(out_root)}）")
    return {
        "ok": True,
        "outRoot": str(out_root),
        "written": [str(p) for p in written],
        "files": len(written),
        "bytes": total_bytes,
        "chapters": per_chapter,
        "pruned": pruned,
        "prunedFiles": pruned_files,
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python3 tools/precompute.py",
        description="L3 静态预计算：生成 data/frames/<chapter>/<viz>/<hash>_t<t>.nlf",
    )
    parser.add_argument("--chapter", action="append", default=None, help="章节 id（可重复；缺省全部）")
    parser.add_argument("--n", type=int, default=1, help="每个 viz 沿时间轴的帧数（缺省 1）")
    parser.add_argument("--limit", type=int, default=None, help="总帧数上限")
    parser.add_argument("--out-root", default=None, help="输出根（缺省仓库根）")
    parser.add_argument("--prune", action="store_true", help="删除没有对应章节/可视化的陈旧帧")
    parser.add_argument("--json", action="store_true", help="把结果以 JSON 打到 stdout（CI 用）")
    parser.add_argument("--quiet", action="store_true", help="不打印人类可读摘要")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口：成功返回 0，参数/注册表错误返回 1（不抛 traceback 给 CI 日志）。"""
    args = _parse_args(argv)
    if args.n < 1:
        print("--n 必须 ≥ 1", file=sys.stderr)
        return 1
    try:
        result = run(args)
    except (RegistryError, OSError, ValueError) as exc:
        print(f"预计算失败：{exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
