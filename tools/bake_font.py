"""重新烘焙 ``tools/raster.py`` 里内嵌的位图字体，并**自检覆盖率**（开发期工具）。

为什么需要它
------------
`tools/raster.py` 是"没有浏览器也能做视觉回归"的那一半：它把 `Scene` 画成 PNG。它不依赖
Pillow、不依赖字体文件，代价是**字体要以位图形式内嵌**（`_FONT_BLOB_B64`）。位图字体的汉字
子集是"烘焙那一刻收集到的字符"，而**收集范围**决定了 PNG 里会不会出现"缺字框"：

* 最初的收集范围只有 `server/chapters/*.py`、`docs/chapters.yaml`、`viz/shared/src/*.ts`、
  `tools/raster.py`——**不含 `viz/chapters/**`**，也就是"前端 model 里手写的中文"。
* 于是每一章的前端装配层一落地，它新写的中文（"蓝/橙/紫/渐/奇/偶/恰/估/夹/薄/竖"…）在
  PNG 基线里全是方框；浏览器 canvas 正常，所以只有栅格基线受影响（2.1 章一次性缺 40 余字，
  0.0/1.1/1.2/1.5/2.2/2.3 都缺，全仓性的问题）。

现在的口径
----------
需要字形的字符集 = **现有 blob 的全部码位** ∪ ASCII 可见字符 ∪
`dist/scenes/**/*.json`（真正会被画出来的文本：文本层的 `text`、轴标题、图例、角标）∪
源码扫描（`server/**/*.py`、`viz/**/*.ts`、`docs/chapters.yaml`、`tools/raster.py`）。
前两项保证"只增不减"，后两项保证"以后新写的文案也会被算进去"。

用法::

    python3 tools/bake_font.py --check     # 覆盖率自检（缺字则 exit 1，信息里带字符清单）
    python3 tools/bake_font.py --write     # 重新烘焙并写回 tools/raster.py

烘焙参数是**标定过的**：DejaVu Sans Mono（ASCII/符号）+ Droid Sans Fallback（汉字，
Apache-2.0），灰度渲染后按 96/255 取阈值，基线锚点 ``anchor="ls"``——用 blob 里已有的字形
反推出来的（ASCII 档 13px 下逐像素一致率 99.9%，残差来自 FreeType 版本差异）。
仓里已有的老基线是按老 blob 画的，所以 `--write` 之后需要对所有章节
``visual_regression.py --update`` 重画一遍基线（字形更全，属于预期变化）。
"""

from __future__ import annotations

import argparse
import base64
import json
import pathlib
import re
import struct
import sys
import zlib
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = pathlib.Path(__file__).resolve().parent.parent
_VERBOSE = True

DEJAVU = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
DROID = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
#: 回退链：等宽 DejaVu 缺的符号（`ℓ ≪ ≫ ⟹ ① ② ③`）从别的字体借字形——**只借字形**，
#: cell 几何与基线锚点不变，所以排版仍然等宽。找不到任何字体提供字形的字符会被跳过
#: 并打印警告（`--check` 会把它算成缺字）。
FALLBACK_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
    "/usr/share/fonts/truetype/arphic/gbsn00lp.ttf",
    "/usr/share/fonts/truetype/batang/batang.ttf",
]

#: 灰度阈值：标定结果（>= 96 视为墨点）
THRESHOLD = 96
#: 汉字档在 16px 宽的 cell 里向右偏 1px 居中（与 blob 里的既有字形对齐）
CJK_DX = 1

#: 各档几何：(size, w, h, baseline)——与既有 blob 完全一致，改了会让所有排版错位
ASCII_STRIKES = [(11, 7, 14, 11), (12, 7, 15, 12), (13, 8, 17, 13), (18, 11, 22, 17)]
CJK_STRIKE = (14, 16, 17, 13)

#: 源码扫描范围（相对仓库根）
SOURCE_GLOBS = [
    "server/**/*.py",
    "viz/**/*.ts",
    "tools/raster.py",
    "docs/chapters.yaml",
]
TEXT_SUFFIXES = {".py", ".ts", ".yaml", ".yml", ".json", ".md"}


def read_blob(root: pathlib.Path = ROOT) -> bytes:
    """从 ``tools/raster.py`` 里抠出 blob（base64 → 原始压缩字节）。"""
    text = (root / "tools" / "raster.py").read_text(encoding="utf-8")
    m = re.search(r"_FONT_BLOB_B64 = \(\n(.*?)\n\)\n", text, re.S)
    if not m:
        raise SystemExit("tools/raster.py 里找不到 _FONT_BLOB_B64 字面量")
    b64 = "".join(re.findall(r'"([^"]*)"', m.group(1)))
    return base64.b64decode(b64)


def parse_blob(blob: bytes) -> tuple[dict[int, tuple[int, int, int, dict[int, np.ndarray]]], dict[int, tuple[int, int, int, dict[int, np.ndarray]]]]:
    """解析 blob → (ascii 档, cjk 档)：``{size: (w, h, baseline, {码位: 位图})}``。"""
    raw = zlib.decompress(blob)
    pos = 0

    def read_strikes(pos: int, count: int) -> tuple[dict[int, tuple[int, int, int, dict[int, np.ndarray]]], int]:
        out: dict[int, tuple[int, int, int, dict[int, np.ndarray]]] = {}
        for _ in range(count):
            size, w, h, base, n = struct.unpack_from("<BBBBH", raw, pos)
            pos += 6
            nbytes = (w + 7) // 8
            ink: dict[int, np.ndarray] = {}
            for _g in range(n):
                (cp,) = struct.unpack_from("<I", raw, pos)
                pos += 4
                bits = np.zeros((h, w), dtype=np.uint8)
                for y in range(h):
                    row = int.from_bytes(raw[pos + y * nbytes : pos + (y + 1) * nbytes], "big")
                    for x in range(w):
                        if row & (1 << (w - 1 - x)):
                            bits[y, x] = 1
                pos += h * nbytes
                ink[cp] = bits
            out[int(size)] = (int(w), int(h), int(base), ink)
        return out, pos

    version, n_ascii = struct.unpack_from("<BB", raw, pos)
    pos += 2
    if version != 1:
        raise SystemExit(f"内嵌字体版本 {version} 未知")
    ascii_strikes, pos = read_strikes(pos, n_ascii)
    version2, n_cjk = struct.unpack_from("<BB", raw, pos)
    pos += 2
    if version2 != 1:
        raise SystemExit(f"内嵌字体版本 {version2} 未知")
    cjk_strikes, pos = read_strikes(pos, n_cjk)
    return ascii_strikes, cjk_strikes


def needed_codepoints(root: pathlib.Path, *, include_existing: bool = True) -> set[int]:
    """需要字形的全部码位（见模块 docstring 的口径）。"""
    cps: set[int] = set(range(0x20, 0x7F))
    if include_existing:
        ascii_strikes, cjk_strikes = parse_blob(read_blob(root))
        for table in list(ascii_strikes.values()) + list(cjk_strikes.values()):
            cps |= set(table[3])
    # 真正会被画出来的文本：场景里的 text / 轴标题 / 图例 / 角标
    for p in (root / "dist" / "scenes").rglob("*.json"):
        cps |= {ord(ch) for ch in p.read_text(encoding="utf-8")}
    # 源码（新写的文案在变成场景之前也要算进来，否则 --check 会漏报）
    for pat in SOURCE_GLOBS:
        for p in root.glob(pat):
            if p.is_file():
                cps |= {ord(ch) for ch in p.read_text(encoding="utf-8", errors="replace")}
    return {cp for cp in cps if cp >= 0x20 and cp not in (0x7F,)}


def _render(font: ImageFont.FreeTypeFont, ch: str, w: int, h: int, base: int, dx: int) -> np.ndarray:
    img = Image.new("L", (w, h), 0)
    ImageDraw.Draw(img).text((dx, base), ch, fill=255, font=font, anchor="ls")
    return (np.asarray(img, dtype=np.uint8) >= THRESHOLD).astype(np.uint8)


#: 私有区码位：任何字体都没有它的字形，于是它的位图就是".notdef 缺字框"
_NOTDEF_CP = 0xE123


def _primary_path(cp: int) -> str:
    return DEJAVU if cp < 0x2E80 else DROID


def _render_with_fallback(cp: int, size: int, w: int, h: int, base: int, dx: int, *, cache: dict) -> np.ndarray | None:
    """渲染一个码位；主字体缺字形时按回退链借。全都没有 → 返回 None（调用方决定怎么办）。"""
    key = (cp, size, w, h, base, dx)
    if key in cache:
        return cache[key]
    chain = [_primary_path(cp)] + [p for p in FALLBACK_FONTS if pathlib.Path(p).exists()]
    notdef: np.ndarray | None = None
    out: np.ndarray | None = None
    for path in chain:
        fk = (path, size)
        font = cache.get(fk)
        if font is None:
            try:
                font = ImageFont.truetype(path, size)
            except Exception:
                continue
            cache[fk] = font
        if notdef is None:
            notdef = _render(font, chr(_NOTDEF_CP), w, h, base, 0)
        bits = _render(font, chr(cp), w, h, base, dx)
        if not np.array_equal(bits, notdef):
            out = bits
            break
    cache[key] = out
    return out


def bake(root: pathlib.Path = ROOT) -> bytes:
    """按标定参数重新烘焙全部档位（保留既有码位 + 新增需要的码位）。"""
    cps = needed_codepoints(root)
    ascii_cps = sorted(cp for cp in cps if cp < 0x2E80)
    cjk_cps = sorted(cp for cp in cps if cp >= 0x2E80)

    cache: dict = {}
    skipped: list[int] = []
    out = bytearray()
    out += struct.pack("<BB", 1, len(ASCII_STRIKES))
    for size, w, h, base in ASCII_STRIKES:
        kept = [cp for cp in ascii_cps if _render_with_fallback(cp, size, w, h, base, 0, cache=cache) is not None]
        skipped += [cp for cp in ascii_cps if cp not in kept]
        out += struct.pack("<BBBBH", size, w, h, base, len(kept))
        for cp in kept:
            out += struct.pack("<I", cp)
            out += _pack(_render_with_fallback(cp, size, w, h, base, 0, cache=cache), w, h)
    out += struct.pack("<BB", 1, 1)
    size, w, h, base = CJK_STRIKE
    kept = [cp for cp in cjk_cps if _render_with_fallback(cp, size, w, h, base, CJK_DX, cache=cache) is not None]
    skipped += [cp for cp in cjk_cps if cp not in kept]
    out += struct.pack("<BBBBH", size, w, h, base, len(kept))
    for cp in kept:
        out += struct.pack("<I", cp)
        out += _pack(_render_with_fallback(cp, size, w, h, base, CJK_DX, cache=cache), w, h)
    if skipped and _VERBOSE:
        uniq = sorted(set(skipped))
        shown = "".join(chr(c) for c in uniq[:40])
        print(f"  ⚠ 有 {len(uniq)} 个码位在所有字体里都没有字形，已跳过：{shown}")
    return zlib.compress(bytes(out), 9)


def _pack(bits: np.ndarray, w: int, h: int) -> bytes:
    nbytes = (w + 7) // 8
    buf = bytearray()
    for y in range(h):
        row = 0
        for x in range(w):
            if bits[y, x]:
                row |= 1 << (w - 1 - x)
        buf += row.to_bytes(nbytes, "big")
    return bytes(buf)


def write_blob(blob: bytes, root: pathlib.Path = ROOT) -> None:
    """把 blob 写回 ``tools/raster.py``（保持原有的大括号与缩进风格）。"""
    path = root / "tools" / "raster.py"
    text = path.read_text(encoding="utf-8")
    b64 = base64.b64encode(blob).decode("ascii")
    chunks = [b64[i : i + 118] for i in range(0, len(b64), 118)]
    body = "\n".join(f'    "{c}"' for c in chunks)
    new = f"_FONT_BLOB_B64 = (\n{body}\n)\n"
    m = re.search(r"_FONT_BLOB_B64 = \(\n.*?\n\)\n", text, re.S)
    if not m:
        raise SystemExit("tools/raster.py 里找不到可替换的 _FONT_BLOB_B64 字面量")
    path.write_text(text[: m.start()] + new + text[m.end() :], encoding="utf-8")


def missing(cps: Iterable[int], root: pathlib.Path = ROOT) -> list[int]:
    """返回 blob 里**没有**（或只有缺字框）字形的码位（按码位排序）。"""
    ascii_strikes, cjk_strikes = parse_blob(read_blob(root))
    have: set[int] = set()
    for table in list(ascii_strikes.values()) + list(cjk_strikes.values()):
        have |= set(table[3])
    cache: dict = {}
    out: list[int] = []
    for cp in sorted(set(cps)):
        if cp not in have:
            out.append(cp)
            continue
        # 有码位但位图与 .notdef 相同 = 还是缺字框
        if cp < 0x2E80:
            size, w, h, base = ASCII_STRIKES[2]
            dx = 0
        else:
            size, w, h, base = CJK_STRIKE
            dx = CJK_DX
        ref = _render_with_fallback(_NOTDEF_CP, size, w, h, base, dx, cache=cache)
        got = _render_with_fallback(cp, size, w, h, base, dx, cache=cache)
        if ref is not None and got is not None and np.array_equal(ref, got):
            out.append(cp)
    return out


def used_codepoints(root: pathlib.Path = ROOT) -> set[int]:
    """**只**统计真正会被画出来的字符（场景 JSON）——`--check` 的严格口径。"""
    cps: set[int] = set()
    for p in (root / "dist" / "scenes").rglob("*.json"):
        cps |= {ord(ch) for ch in p.read_text(encoding="utf-8")}
    return {cp for cp in cps if cp >= 0x20}


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="重新烘焙/自检 tools/raster.py 的内嵌位图字体")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true", help="覆盖率自检：场景里出现的字符必须都有字形")
    g.add_argument("--write", action="store_true", help="重新烘焙并写回 tools/raster.py")
    ap.add_argument("--json", action="store_true", help="机器可读输出")
    ap.add_argument("--root", type=pathlib.Path, default=ROOT, help="仓库根（测试用）")
    args = ap.parse_args(list(argv) if argv is not None else None)
    root: pathlib.Path = args.root

    if args.write:
        blob = bake(root)
        write_blob(blob, root)
        cps = needed_codepoints(root)
        if args.json:
            print(json.dumps({"written": True, "codepoints": len(cps), "bytes": len(blob)}, ensure_ascii=False))
        else:
            print(f"已重新烘焙：{len(cps)} 个码位，blob {len(blob)} 字节（写回 tools/raster.py）")
            print("提示：字形集合变了，请对每一章跑 `visual_regression.py --update --chapter <id>` 重画基线。")
        return 0

    bad = missing(used_codepoints(root), root)
    if args.json:
        print(json.dumps({"missing": [chr(cp) for cp in bad], "ok": not bad}, ensure_ascii=False))
    elif bad:
        shown = "".join(chr(cp) for cp in bad[:60])
        print(f"❌ 场景里出现了 {len(bad)} 个没有字形的字符：{shown}")
        print("   （PNG 里它们会画成缺字框；浏览器 canvas 不受影响）")
        print("   修法：python3 tools/bake_font.py --write 之后重画基线")
    else:
        print("内嵌字体覆盖率自检通过：场景里出现的字符全部有字形")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
