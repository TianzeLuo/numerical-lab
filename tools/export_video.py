"""把一章/一个可视化的**时间序列**导出成 GIF（环境里没有 ffmpeg，GIF 是唯一能动的交付格式）。

链路
----
``server.frames.build_frame``（逐时刻取帧）→ ``tools.raster.default_scene_for_frame``（标准探针场景）
→ ``tools.raster.render_scene``（纯 Python 光栅化）→ ``tools.gif.write_gif``（自研 LZW / 中位切分）。

时间轴怎么取（**这一条决定了 GIF 到底动不动**）
----------------------------------------------
1. 先把时间轴映射到一个"**时间参数**"：manifest 里类型为 float/int、名字匹配
   ``t / t_end / t_max / tmax / time / period`` 的参数（如 0.0 章的 ``t_end``）。
   有它就把该参数从它自己的 ``min`` 扫到 ``min(max, time.t_max)``——这是"帧真的在变"的唯一可靠来源。
2. 没有时间参数时才把 ``t`` 直接传给 ``compute``（此时帧变不变取决于章节实现）。
3. ``time.mode == "none"`` 且没有时间参数 → 只会得到 1 帧；工具会**显式警告**"帧序列退化了"，
   而不是假装导出了一段动画。

调色板（``--palette``）
-----------------------
* ``auto``（缺省）：场景里有 heat 图层 → 用该图层 cmap 的 256 级 LUT
  （``tools/colorscale_tables.py``，与 TS 侧同源）当调色板——热力图动画因此逐帧配色稳定；
  其余颜色（背景、文字、坐标轴）不会被"最近邻到色标里"：LUT 会让出若干格给它们
  （LUT 均匀抽样 + 帧内其它颜色，总数 ≤ 256）。
  没有 heat 图层 → 用帧内**实际出现过的颜色**（≤256 时无损，一个像素都不许变）。
* ``--palette <cmap>``：强制用该色标的 LUT（同样会给非热力颜色让格）。
* ``--palette lossless``：强制用帧内实际颜色（超过 256 色时交给 GIF 的中位切分）。

``--png-dir`` 可把逐帧 PNG 一并存下来（便于挑图或做讲义插图）。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from server.frames import build_frame  # noqa: E402
from server.registry import default_registry  # noqa: E402
from tools.colorscale_tables import NAMES as CMAP_NAMES  # noqa: E402
from tools.colorscale_tables import TABLES as CMAP_TABLES  # noqa: E402
from tools.gif import write_gif  # noqa: E402
from tools.png import write_rgb  # noqa: E402
from tools.raster import default_scene_for_frame, render_scene, linspace  # noqa: E402

__all__ = ["main", "build_frames", "choose_palette", "sweep_values"]

DEFAULT_WIDTH = 640.0
DEFAULT_HEIGHT = 480.0
_MAX_PALETTE = 256
_TIME_PARAM_RE = re.compile(r"(^t$|^t_|_t$|time|tmax|t_max|period)", re.IGNORECASE)


def _pack(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0].astype(np.int64)
    g = rgb[..., 1].astype(np.int64)
    b = rgb[..., 2].astype(np.int64)
    return (r << 16) | (g << 8) | b


def _unpack(packed: np.ndarray) -> np.ndarray:
    p = np.asarray(packed, dtype=np.int64)
    return np.stack([(p >> 16) & 0xFF, (p >> 8) & 0xFF, p & 0xFF], axis=1).astype(np.uint8)


def time_param_name(viz: Mapping[str, Any]) -> str | None:
    """挑出"时间参数"：类型是数值、名字像时间的那一个（没有就返回 None）。"""
    for p in viz.get("params", []):
        if p.get("type") not in ("float", "int"):
            continue
        if _TIME_PARAM_RE.search(str(p.get("name", ""))):
            return str(p["name"])
    return None


def sweep_values(viz: Mapping[str, Any], n: int) -> tuple[list[float], str | None, str]:
    """给出 n 个"时刻"与该时刻对应的参数名；返回 (值列表, 参数名, 说明)。"""
    if n < 1:
        raise ValueError("--n 必须 ≥ 1")
    name = time_param_name(viz)
    spec = None
    if name is not None:
        spec = next(p for p in viz.get("params", []) if p.get("name") == name)
    time_spec = viz.get("time") or {}
    t_max = time_spec.get("tMax") or time_spec.get("t_max")
    if name is not None and spec is not None:
        lo = spec.get("min")
        hi = spec.get("max")
        if lo is None or hi is None:
            lo, hi = 0.0, float(t_max or 1.0)
        if t_max is not None:
            hi = min(float(hi), float(t_max))
        lo, hi = float(lo), float(hi)
        if lo == hi:
            return [lo], name, f"时间参数 {name} 的区间退化（{lo}），只有 1 帧"
        return [float(v) for v in linspace(lo, hi, n)], name, f"扫时间参数 {name} ∈ [{lo}, {hi}]"
    hi = float(t_max or 1.0)
    if time_spec.get("mode") in (None, "none"):
        return [0.0], None, "该可视化没有时间轴（time.mode=none）→ 只有 1 帧"
    return [float(v) for v in linspace(0.0, hi, n)], None, f"把 t ∈ [0, {hi}] 直接传给 compute"


def build_frames(
    entry: Any,
    viz_id: str,
    *,
    n: int = 60,
    params: Mapping[str, Any] | None = None,
    width: float = DEFAULT_WIDTH,
    height: float = DEFAULT_HEIGHT,
    scale: float = 1.0,
) -> tuple[list[np.ndarray], list[dict[str, Any]], str]:
    """取时间序列 → 标准场景 → RGB 帧。返回 (帧, 场景列表, 说明)。"""
    viz = entry.manifest.viz(viz_id).to_json()
    values, t_param, note = sweep_values(viz, n)
    frames: list[np.ndarray] = []
    scenes: list[dict[str, Any]] = []
    for value in values:
        base = dict(params or {})
        t = 0.0
        if t_param is not None:
            base[t_param] = value
            t = float(value)
        else:
            t = float(value)
        frame = build_frame(entry, viz_id, base, t)
        scene = default_scene_for_frame(viz, frame, width=width, height=height, t=t)
        image = render_scene(scene, scale=scale, validate=True)
        rgba = np.frombuffer(image.pixels, dtype=np.uint8).reshape(image.height, image.width, 4)
        # 画布背景不透明（alpha=255），直接丢 alpha 即可；这里仍走一次白底合成，防止将来出现透明画布。
        alpha = rgba[:, :, 3:4].astype(np.float64) / 255.0
        rgb = np.clip(np.floor(rgba[:, :, :3].astype(np.float64) * alpha + 255.0 * (1.0 - alpha) + 0.5), 0, 255)
        frames.append(rgb.astype(np.uint8))
        scenes.append(scene)
    return frames, scenes, note


def choose_palette(frames: Sequence[np.ndarray], scenes: Sequence[Mapping[str, Any]], mode: str) -> tuple[np.ndarray | None, str]:
    """按 ``--palette`` 选全局调色板；返回 (palette 或 None, 说明)。"""
    if not frames:
        return None, "没有帧"
    packed = np.unique(np.concatenate([_pack(f).reshape(-1) for f in frames]))
    heat_cmaps = [str(l.get("cmap", "viridis")) for s in scenes for l in s.get("layers", []) if l.get("kind") == "heat"]
    cmap = None
    if mode == "lossless":
        return (None, f"无损优先：帧里出现 {packed.size} 种颜色" + ("（>256，交给中位切分）" if packed.size > _MAX_PALETTE else "（≤256，逐像素无损）"))
    if mode != "auto":
        if mode not in CMAP_TABLES:
            raise SystemExit(f"未知调色板 {mode!r}；可用：auto / lossless / {', '.join(CMAP_NAMES)}")
        cmap = mode
    elif heat_cmaps:
        cmap = heat_cmaps[0]
    elif packed.size <= _MAX_PALETTE:
        return _unpack(packed), f"帧内实际颜色（{packed.size} 种，逐像素无损）"

    if cmap is None:
        return None, f"帧内颜色 {packed.size} 种 > 256，交给 GIF 中位切分"

    lut = np.asarray(CMAP_TABLES[cmap], dtype=np.uint8)
    lut_packed = np.sort(_pack(lut))
    extras = packed[~np.isin(packed, lut_packed)]
    if extras.size > 128:  # 非热力颜色太多：让位不如让色标完整，交给中位切分
        return None, f"非热力颜色多达 {extras.size} 种，改用 GIF 中位切分"
    keep = max(2, _MAX_PALETTE - extras.size)
    idx = np.unique(np.floor(np.linspace(0, 255, keep) + 1e-9).astype(np.int64))
    palette = np.concatenate([_unpack(extras), lut[idx]], axis=0)
    return palette, f"{cmap} 色标 LUT（{idx.size} 级）+ 帧内其它 {extras.size} 色"


def _parse_params(text: str | None) -> dict[str, Any]:
    if not text:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--params 不是合法 JSON：{exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit("--params 必须是 JSON 对象，例如 '{\"nx\": 41}'")
    return data


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="把可视化的时间序列导出为 GIF（纯 Python，无 ffmpeg）")
    ap.add_argument("--chapter", required=True, help="章节 id，如 0.0")
    ap.add_argument("--viz", required=True, help="可视化 id，如 heat_ftcs")
    ap.add_argument("--out", required=True, help="输出 .gif 路径")
    ap.add_argument("--n", type=int, default=60, help="帧数（缺省 60）")
    ap.add_argument("--fps", type=float, default=25.0, help="帧率（缺省 25；GIF 的 delay 以 1/100 s 为单位）")
    ap.add_argument("--scale", type=float, default=1.0, help="光栅缩放（缺省 1.0）")
    ap.add_argument("--width", type=float, default=DEFAULT_WIDTH, help=f"画布宽（缺省 {DEFAULT_WIDTH:g}）")
    ap.add_argument("--height", type=float, default=DEFAULT_HEIGHT, help=f"画布高（缺省 {DEFAULT_HEIGHT:g}）")
    ap.add_argument("--params", default=None, help="参数覆盖（JSON 对象，会覆盖 manifest 缺省值）")
    ap.add_argument("--palette", default="auto", help="auto | lossless | 色标名（viridis/magma/coolwarm/gray/rdbu）")
    ap.add_argument("--loop", type=int, default=0, help="NETSCAPE 循环次数，0 = 无限（缺省 0）")
    ap.add_argument("--png-dir", default=None, help="额外把逐帧 PNG 存到该目录")
    ap.add_argument("--dither", action="store_true", help="调色板映射开启 Floyd–Steinberg 抖动")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 入口：返回进程退出码（0 = 成功）。"""
    args = _parse_args(argv)
    if args.fps <= 0:
        raise SystemExit("--fps 必须为正")
    delay_cs = max(1, int(round(100.0 / args.fps)))

    registry = default_registry()
    try:
        entry = registry.get(args.chapter)
    except KeyError:
        raise SystemExit(f"没有章节 {args.chapter}（已加载：{', '.join(registry.ids()) or '（空）'}）")
    if args.viz not in entry.viz_ids:
        raise SystemExit(f"章节 {args.chapter} 没有可视化 {args.viz}（可用：{', '.join(entry.viz_ids)}）")

    frames, scenes, note = build_frames(
        entry,
        args.viz,
        n=args.n,
        params=_parse_params(args.params),
        width=args.width,
        height=args.height,
        scale=args.scale,
    )
    if not frames:
        raise SystemExit("没有可导出的帧")
    print(f"[video] {note}；共 {len(frames)} 帧，{frames[0].shape[1]}×{frames[0].shape[0]}，delay={delay_cs}cs")

    changed = sum(1 for i in range(1, len(frames)) if not np.array_equal(frames[i], frames[i - 1]))
    if changed == 0 and len(frames) > 1:
        print(
            "[video] ⚠ 帧序列退化了：所有帧逐像素相同——该可视化的 compute 既没有用到时间参数、也没有用到 t。"
            "这不是导出失败，但 GIF 不会动。",
            file=sys.stderr,
        )

    palette, palette_note = choose_palette(frames, scenes, args.palette)
    print(f"[video] 调色板：{palette_note}")

    out = pathlib.Path(args.out)
    write_gif(
        out,
        frames,
        delay_cs=delay_cs,
        loop=int(args.loop),
        palette=palette,
        dither=bool(args.dither),
        comment=f"numerical-lab {args.chapter}/{args.viz} n={len(frames)} fps={args.fps:g}",
    )
    print(f"[video] 写出 {out}（{out.stat().st_size} 字节）")

    if args.png_dir:
        png_dir = pathlib.Path(args.png_dir)
        png_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames):
            t = float(scenes[i].get("t", 0.0))
            name = f"{args.chapter}-{args.viz}-{i:04d}-t{t:.6g}.png"
            write_rgb(png_dir / name, frame)
        print(f"[video] 逐帧 PNG：{png_dir}（{len(frames)} 张）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
