"""视觉回归（CI 门禁 G8）：把 Scene 光栅化成 PNG，与基线逐字节 + 逐像素比对。

输入优先级（这条规则是**硬约定**，见 ``docs/PROTOCOLS.md`` §6/§9）
------------------------------------------------------------------
1. **首选：TS 模型转储的 Scene**——``dist/scenes/<chapter>/<viz>__<params指纹>__t<时刻>.json``
   （由 ``viz/tests/scene/dump_scenes.ts`` → ``npm run scene`` 产出）。它才是"用户真正会看到
   的那张图"；基线落在 ``tests/baselines/<chapter>/<同名>.png``。
   场景 JSON 里没有 ``canvas``（或没有 w/h）时用 ``--width/--height``（缺省 960×600，与前端画布一致）。
2. **兜底：``tools/raster.py::default_scene_for_frame``**——某章/某可视化在 ``dist/scenes/`` 里
   没有对应文件时才用，并且**必须在报告里标出来**（``"source": "fallback"`` + ``reason``）。
   兜底路径只生成报告、**不算失败**：否则 TS 还没编译就会让 G8 莫名其妙变红。

为什么要有这条优先级：Python 侧的"标准场景"只是回归探针，不是教学内容；如果它变成了主路径，
视觉回归就会退化成"探针自己跟自己比"，用户看到的图反而没人守。

``--check`` 只把**首选路径**的不一致算作失败；像素差异报告给出可诊断的四件套：
不同像素数 / 最大通道差 / 差异包围盒 / 平均绝对差（外加可选的差异图）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from server.frames import build_frame  # noqa: E402
from server.registry import default_registry  # noqa: E402
from tools.png import read_png, write_rgb  # noqa: E402
from tools.raster import (  # noqa: E402
    default_scene_for_frame,
    render_scene,
    render_scene_to_bytes,
    validate_scene,
)

__all__ = ["main", "compare_png", "discover_scenes", "DiffReport"]

DEFAULT_SCENES_DIR = "dist/scenes"
DEFAULT_BASELINE_DIR = "tests/baselines"
DEFAULT_DIST_VISUAL = "dist/visual"
DEFAULT_WIDTH = 960.0
DEFAULT_HEIGHT = 600.0


# --------------------------------------------------------------------------- #
# 像素差异报告
# --------------------------------------------------------------------------- #
@dataclass
class DiffReport:
    """两张 PNG 的差异（可诊断：不只说"不一样"）。"""

    same_bytes: bool
    shape_mismatch: bool = False
    different_pixels: int = 0
    total_pixels: int = 0
    max_channel_delta: int = 0
    mean_abs_delta: float = 0.0
    bbox: tuple[int, int, int, int] | None = None  # (x0, y0, x1, y1)，右下开区间
    note: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "sameBytes": self.same_bytes,
            "shapeMismatch": self.shape_mismatch,
            "differentPixels": self.different_pixels,
            "totalPixels": self.total_pixels,
            "maxChannelDelta": self.max_channel_delta,
            "meanAbsDelta": round(self.mean_abs_delta, 6),
            "bbox": list(self.bbox) if self.bbox else None,
            "note": self.note,
        }

    def summary(self) -> str:
        if self.same_bytes:
            return "逐字节一致"
        if self.shape_mismatch:
            return f"尺寸不一致：{self.note}"
        if self.different_pixels == 0:
            return "像素一致（容器字节不同）"
        x0, y0, x1, y1 = self.bbox or (0, 0, 0, 0)
        return (
            f"{self.different_pixels}/{self.total_pixels} 像素不同"
            f"（{100.0 * self.different_pixels / max(1, self.total_pixels):.3f}%），"
            f"最大通道差 {self.max_channel_delta}，平均绝对差 {self.mean_abs_delta:.4f}，"
            f"差异区域 x[{x0},{x1}) y[{y0},{y1})"
        )


def _normalize_rgb(arr: np.ndarray) -> np.ndarray:
    """任意通道形态 → 白底合成后的 ``(H, W, 3)`` uint8（视觉回归比的是肉眼所见）。"""
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.shape[2] == 4:
        alpha = arr[:, :, 3:4].astype(np.float64) / 255.0
        rgb = arr[:, :, :3].astype(np.float64) * alpha + 255.0 * (1.0 - alpha)
        arr = np.clip(np.floor(rgb + 0.5), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr[:, :, :3])


def _pixels(path: pathlib.Path) -> np.ndarray:
    return _normalize_rgb(read_png(path).array())


def compare_arrays(a: np.ndarray, b: np.ndarray, *, same_bytes: bool = False) -> DiffReport:
    """比对两份像素：不同像素数 / 最大通道差 / 包围盒 / 平均绝对差（可诊断）。"""
    a = _normalize_rgb(a)
    b = _normalize_rgb(b)
    if a.shape != b.shape:
        return DiffReport(
            same_bytes=same_bytes,
            shape_mismatch=True,
            total_pixels=int(a.shape[0] * a.shape[1]),
            note=f"{a.shape[1]}x{a.shape[0]} vs {b.shape[1]}x{b.shape[0]}",
        )
    total = int(a.shape[0] * a.shape[1])
    if same_bytes:
        return DiffReport(same_bytes=True, total_pixels=total)
    diff = np.abs(a.astype(np.int32) - b.astype(np.int32))
    per_pixel = diff.max(axis=2)
    mask = per_pixel > 0
    count = int(np.count_nonzero(mask))
    report = DiffReport(
        same_bytes=False,
        different_pixels=count,
        total_pixels=total,
        max_channel_delta=int(diff.max()) if count else 0,
        mean_abs_delta=float(diff.mean()) if count else 0.0,
    )
    if count:
        ys, xs = np.nonzero(mask)
        report.bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    else:
        report.note = "像素完全相同，仅容器字节不同（如 zlib 级别/元数据差异）"
    return report


def compare_png(path_a: pathlib.Path, path_b: pathlib.Path) -> DiffReport:
    """比对两个 PNG 文件：字节 + 像素四件套（不同像素数 / 最大通道差 / 包围盒 / 平均差）。"""
    blob_a = path_a.read_bytes()
    blob_b = path_b.read_bytes()
    if blob_a == blob_b:
        return compare_arrays(_pixels(path_a), _pixels(path_a), same_bytes=True)
    return compare_arrays(_pixels(path_a), _pixels(path_b))


def diff_image(a: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    """把差异画成一张图：相同处压暗，不同处标红（便于人眼定位）。"""
    a = _normalize_rgb(a)
    b = _normalize_rgb(b)
    if a.shape != b.shape:
        return None
    diff = np.abs(a.astype(np.int32) - b.astype(np.int32)).max(axis=2)
    out = (a.astype(np.float64) * 0.25 + 191.25).astype(np.uint8)
    out[diff > 0] = (255, 0, 0)
    return out


def write_diff_image(path_a: pathlib.Path, path_b: pathlib.Path, out: pathlib.Path) -> pathlib.Path | None:
    """差异图落盘（两个 PNG 文件版）。"""
    arr = diff_image(_pixels(path_a), _pixels(path_b))
    if arr is None:
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    write_rgb(out, arr)
    return out


# --------------------------------------------------------------------------- #
# 输入发现
# --------------------------------------------------------------------------- #
@dataclass
class Entry:
    """一个待回归的可视化条目。"""

    chapter: str
    viz: str
    source: str  # "scene"（TS 转储）/ "fallback"（Python 标准场景）
    reason: str = ""
    scene_path: pathlib.Path | None = None
    baseline_name: str = ""
    scene: dict[str, Any] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.chapter}/{self.viz}"


def _inject_canvas(scene: dict[str, Any], width: float, height: float) -> dict[str, Any]:
    """场景 JSON 缺 ``canvas``（或 w/h）时补上前端画布尺寸。"""
    canvas = scene.get("canvas")
    if not isinstance(canvas, dict):
        canvas = {}
    if not isinstance(canvas.get("w"), (int, float)) or canvas.get("w", 0) <= 0:
        canvas["w"] = width
    if not isinstance(canvas.get("h"), (int, float)) or canvas.get("h", 0) <= 0:
        canvas["h"] = height
    canvas.setdefault("background", "#ffffff")
    scene["canvas"] = canvas
    return scene


def discover_scenes(scenes_dir: pathlib.Path, *, width: float, height: float) -> dict[str, Entry]:
    """扫描 ``dist/scenes/**/*.json``（TS 模型转储），返回 ``chapter/viz → Entry``。

    目录约定 ``<scenes_dir>/<chapter>/<viz>…json``；名字里第一段 ``__`` 之前是 viz id。
    """
    out: dict[str, Entry] = {}
    if not scenes_dir.is_dir():
        return out
    for path in sorted(scenes_dir.rglob("*.json")):
        if path.name.startswith("."):
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            chapter = path.parent.name
            out[f"{chapter}/{path.stem}"] = Entry(
                chapter=chapter, viz=path.stem, source="scene", reason=f"场景 JSON 解析失败：{exc}",
                scene_path=path, baseline_name=path.stem,
            )
            continue
        if not isinstance(raw, dict):
            continue
        chapter = str(raw.get("chapter") or path.parent.name)
        viz = str(raw.get("viz") or path.stem.split("__")[0])
        scene = _inject_canvas(raw, width, height)
        key = f"{chapter}/{path.stem}"  # 同名文件互不覆盖：一个 viz 可以有多个参数/时刻的转储
        out[key] = Entry(
            chapter=chapter,
            viz=viz,
            source="scene",
            scene_path=path,
            baseline_name=path.stem,
            scene=scene,
            issues=validate_scene(scene),
        )
    return out


def fallback_entries(
    registry: Any,
    covered: Mapping[str, Entry],
    *,
    width: float,
    height: float,
    t: float,
) -> list[Entry]:
    """对 ``dist/scenes`` 没覆盖到的 (chapter, viz) 用标准探针场景兜底。"""
    covered_pairs = {(e.chapter, e.viz) for e in covered.values()}
    out: list[Entry] = []
    for entry in registry.all():
        for viz_id in entry.viz_ids:
            if (entry.id, viz_id) in covered_pairs:
                continue
            viz = entry.manifest.viz(viz_id).to_json()
            frame = build_frame(entry, viz_id, None, t)
            scene = default_scene_for_frame(viz, frame, width=width, height=height)
            out.append(
                Entry(
                    chapter=entry.id,
                    viz=viz_id,
                    source="fallback",
                    reason="dist/scenes 里没有该可视化的 TS 场景转储（npm run scene 未产出）",
                    baseline_name=viz_id,
                    scene=scene,
                    issues=validate_scene(scene),
                )
            )
    return out


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def _sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def make_relative(path: pathlib.Path) -> str:
    """相对仓库根的可读路径（临时目录下的路径就原样返回）。"""
    try:
        return str(path.relative_to(_ROOT))
    except ValueError:
        return str(path)


def run(
    *,
    mode: str,
    scenes_dir: pathlib.Path,
    baseline_dir: pathlib.Path,
    dist_visual: pathlib.Path,
    chapters: Sequence[str] | None = None,
    vizzes: Sequence[str] | None = None,
    width: float = DEFAULT_WIDTH,
    height: float = DEFAULT_HEIGHT,
    scale: float = 1.0,
    t: float = 0.0,
    write_diff: bool = False,
    diff_dir: pathlib.Path | None = None,
    quiet: bool = False,
    require_scenes: bool = False,
) -> dict[str, Any]:
    """执行一轮视觉回归；返回机器可读结果（``--json`` 直接打印它）。

    ``require_scenes=True`` 时"一个 TS 场景转储都没有"本身就是失败：CI 里应该这么用
    （否则 ``npm run scene`` 一没跑，G8 就退化成"只出兜底报告"，看起来还是绿的）。
    """
    registry = default_registry()
    entries: list[Entry] = []
    primary = discover_scenes(scenes_dir, width=width, height=height)
    entries.extend(primary.values())
    entries.extend(fallback_entries(registry, primary, width=width, height=height, t=t))
    if chapters:
        entries = [e for e in entries if e.chapter in set(chapters)]
    if vizzes:
        entries = [e for e in entries if e.viz in set(vizzes)]
    entries.sort(key=lambda e: (e.chapter, e.viz, e.baseline_name))

    results: list[dict[str, Any]] = []
    totals = {"entries": 0, "scene": 0, "fallback": 0, "ok": 0, "updated": 0, "mismatch": 0,
              "missing": 0, "invalid": 0, "failed": 0, "created": 0}
    for entry in entries:
        totals["entries"] += 1
        totals["scene" if entry.source == "scene" else "fallback"] += 1
        record: dict[str, Any] = {
            "chapter": entry.chapter,
            "viz": entry.viz,
            "source": entry.source,
            "reason": entry.reason,
            "scene": make_relative(entry.scene_path) if entry.scene_path else None,
            "baseline": make_relative(baseline_dir / entry.chapter / f"{entry.baseline_name}.png"),
            "blocking": entry.source == "scene",
        }
        if entry.issues:
            record["status"] = "invalid"
            record["issues"] = list(entry.issues)
            totals["invalid"] += 1
            if entry.source == "scene":  # 兜底场景不合法只报告，不让门禁变红（TS 还没转储）
                totals["failed"] += 1
            results.append(record)
            if not quiet:
                print(f"[invalid] {entry.chapter}/{entry.viz}（{entry.source}）：{len(entry.issues)} 处不合契约")
                for issue in entry.issues[:5]:
                    print(f"          - {issue}")
            continue
        image = render_scene(entry.scene, scale=scale, validate=False)
        png = render_scene_to_bytes(entry.scene, scale=scale, validate=False)
        record["sha256"] = _sha256(png)
        record["bytes"] = len(png)
        record["size"] = [image.width, image.height]
        baseline = baseline_dir / entry.chapter / f"{entry.baseline_name}.png"
        exists = baseline.exists()
        if mode == "update":
            baseline.parent.mkdir(parents=True, exist_ok=True)
            before = baseline.read_bytes() if exists else None
            baseline.write_bytes(png)
            record["status"] = "ok" if before == png else ("created" if before is None else "updated")
            totals["ok" if record["status"] == "ok" else ("created" if before is None else "updated")] += 1
        else:
            if not exists:
                record["status"] = "missing"
                totals["missing"] += 1
                if entry.source == "scene":
                    totals["failed"] += 1
            else:
                base_pixels = _pixels(baseline)
                report = compare_arrays(base_pixels, image.array(), same_bytes=baseline.read_bytes() == png)
                record["diff"] = report.to_json()
                if report.same_bytes or report.different_pixels == 0:
                    record["status"] = "ok"
                    totals["ok"] += 1
                else:
                    record["status"] = "mismatch"
                    totals["mismatch"] += 1
                    if entry.source == "scene":
                        totals["failed"] += 1
                    if write_diff:
                        arr = diff_image(base_pixels, image.array())
                        if arr is not None:
                            target = (diff_dir or dist_visual / "diff") / entry.chapter / f"{entry.baseline_name}.png"
                            target.parent.mkdir(parents=True, exist_ok=True)
                            write_rgb(target, arr)
                            record["diffImage"] = make_relative(target)
        # PROTOCOLS §9：dist/visual/ 下留一份场景快照与光栅（供人翻查）
        scene_out = dist_visual / "scenes" / f"{entry.chapter}-{entry.baseline_name}.json"
        png_out = dist_visual / "png" / f"{entry.chapter}-{entry.baseline_name}.png"
        scene_out.parent.mkdir(parents=True, exist_ok=True)
        scene_out.write_text(json.dumps(entry.scene, ensure_ascii=False, sort_keys=True, indent=1), encoding="utf-8")
        png_out.parent.mkdir(parents=True, exist_ok=True)
        png_out.write_bytes(png)
        results.append(record)
        if not quiet:
            _print_record(record)

    if require_scenes and totals["scene"] == 0 and totals["entries"]:
        totals["failed"] += 1  # 兜底路径不能悄悄变成主路径：CI 用 --require-scenes 把这条钉死
    ok = totals["failed"] == 0
    return {
        "mode": mode,
        "scenesDir": str(scenes_dir),
        "baselineDir": str(baseline_dir),
        "scale": scale,
        "totals": totals,
        "ok": ok,
        "entries": results,
    }


def _print_record(record: Mapping[str, Any]) -> None:
    status = record["status"]
    tag = "ok" if status == "ok" else status
    extra = ""
    diff = record.get("diff")
    if isinstance(diff, Mapping) and not diff.get("sameBytes") and status == "mismatch":
        x0, y0, x1, y1 = diff.get("bbox") or (0, 0, 0, 0)
        extra = (
            f"  →  {diff['differentPixels']}/{diff['totalPixels']} 像素不同，"
            f"最大通道差 {diff['maxChannelDelta']}，包围盒 x[{x0},{x1}) y[{y0},{y1})"
        )
    src = "scene" if record["source"] == "scene" else "fallback"
    mark = "  " if record.get("blocking", True) else "· "
    print(f"{mark}[{tag:>7}] {record['chapter']}/{record['viz']}（{src}）{extra}")
    if status == "invalid":
        for issue in record.get("issues", [])[:3]:
            print(f"             - {issue}")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="视觉回归：Scene → PNG 与基线比对（CI 门禁 G8）")
    ap.add_argument("--check", action="store_true", help="与基线比对（缺省动作）")
    ap.add_argument("--update", action="store_true", help="生成/更新基线")
    ap.add_argument("--json", action="store_true", help="输出机器可读结果（stdout）")
    ap.add_argument("--scene-dir", "--scenes-dir", dest="scenes_dir", default=DEFAULT_SCENES_DIR,
                    help=f"TS 场景转储目录（缺省 {DEFAULT_SCENES_DIR}）")
    ap.add_argument("--baseline-dir", default=DEFAULT_BASELINE_DIR, help=f"基线目录（缺省 {DEFAULT_BASELINE_DIR}）")
    ap.add_argument("--dist-visual", default=DEFAULT_DIST_VISUAL, help=f"场景/光栅快照目录（缺省 {DEFAULT_DIST_VISUAL}）")
    ap.add_argument("--chapter", action="append", help="只跑指定章节（可重复）")
    ap.add_argument("--viz", action="append", help="只跑指定可视化（可重复）")
    ap.add_argument("--width", type=float, default=DEFAULT_WIDTH, help="场景缺 canvas 时的宽度")
    ap.add_argument("--height", type=float, default=DEFAULT_HEIGHT, help="场景缺 canvas 时的高度")
    ap.add_argument("--scale", type=float, default=1.0, help="光栅缩放（等价 devicePixelRatio）")
    ap.add_argument("--t", type=float, default=0.0, help="兜底路径取帧的时刻")
    ap.add_argument("--diff-dir", default=None, help="差异图输出目录（缺省 dist/visual/diff）")
    ap.add_argument("--no-diff-image", action="store_true", help="不写差异图")
    ap.add_argument("--quiet", action="store_true", help="只输出汇总")
    ap.add_argument("--require-scenes", action="store_true",
                    help="没有 TS 场景转储即判失败（CI 用；兜底路径默认不算失败）")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 入口：``--update`` 写基线，``--check`` 比对；返回进程退出码。"""
    args = _parse_args(argv)
    if args.update and args.check:
        print("--update 与 --check 不能同时用", file=sys.stderr)
        return 2
    mode = "update" if args.update else "check"
    result = run(
        mode=mode,
        scenes_dir=_ROOT / args.scenes_dir,
        baseline_dir=_ROOT / args.baseline_dir,
        dist_visual=_ROOT / args.dist_visual,
        chapters=args.chapter,
        vizzes=args.viz,
        width=args.width,
        height=args.height,
        scale=args.scale,
        t=args.t,
        write_diff=not args.no_diff_image,
        diff_dir=_ROOT / args.diff_dir if args.diff_dir else None,
        # --json 时关掉逐条打印：stdout 必须只有一份机器可读的 JSON。
        quiet=args.quiet or args.json,
        require_scenes=args.require_scenes,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=1))
    else:
        totals = result["totals"]
        print(
            f"[visual] {mode}：{totals['entries']} 个条目"
            f"（TS 场景 {totals['scene']} / 兜底 {totals['fallback']}）· "
            f"ok {totals['ok']} · 更新 {totals['updated']} · 新建 {totals['created']} · "
            f"不一致 {totals['mismatch']} · 缺基线 {totals['missing']} · 不合法 {totals['invalid']}"
        )
        if result["totals"]["scene"] == 0 and result["totals"]["fallback"]:
            print(
                f"[visual] ⚠ {result['totals']['fallback']} 个条目走了**兜底**路径："
                f"{args.scenes_dir} 里没有 TS 场景转储（先跑 npm run build && npm run scene）；"
                "兜底只出报告、不算失败，要在 CI 里钉死请加 --require-scenes",
                file=sys.stderr,
            )
        if not result["ok"]:
            print("[visual] 失败：首选路径（TS 场景）存在不一致/缺基线/不合契约，或 --require-scenes 下没有转储", file=sys.stderr)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
