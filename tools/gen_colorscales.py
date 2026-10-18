"""生成色标的**双侧共享**数据：``viz/shared/src/colorscale_data.gen.ts`` 与 ``viz/shared/colorscales.json``。

为什么要有这个生成器：色标必须在前端（TypeScript 画热力图）与后端（Python 光栅化、视觉回归、
讲义插图）之间**逐字节一致**，否则"同一份 Scene 在浏览器和基线里长得不一样"，
视觉回归立刻变成噪声。生成物入库 + CI 门禁（G6）比对，是唯一能长期守住这件事的做法。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.colorscale_tables import NAMES, TABLES  # noqa: E402

TS_PATH = _ROOT / "viz" / "shared" / "src" / "colorscale_data.gen.ts"
JSON_PATH = _ROOT / "viz" / "shared" / "colorscales.json"

_HEADER = """// ⚠️ 本文件由 tools/gen_colorscales.py 生成，请勿手改。
// 修改色标请编辑 tools/colorscale_tables.py，然后重跑生成器。
// 门禁 G6（tools/gen_colorscales.py --check）会检查它与真源一致。

"""


def _ts_source() -> str:
    lines = [_HEADER]
    names = ", ".join(f'"{n}"' for n in NAMES)
    lines.append(f"export const CMAP_NAMES = [{names}] as const;\n")
    lines.append("export type CmapName = (typeof CMAP_NAMES)[number];\n\n")
    lines.append("/** 名称 → 256×3 的 RGB 查找表（行主序，取值 0..255）。 */\n")
    lines.append("export const CMAP_LUT: Record<CmapName, Uint8Array> = {\n")
    for name in NAMES:
        flat = []
        for r, g, b in TABLES[name]:
            flat.extend((r, g, b))
        body = ", ".join(str(v) for v in flat)
        lines.append(f'  "{name}": new Uint8Array([{body}]),\n')
    lines.append("};\n")
    return "".join(lines)


def _json_doc() -> str:
    doc = {
        "generator": "tools/gen_colorscales.py",
        "levels": 256,
        "names": list(NAMES),
        "tables": {n: [list(rgb) for rgb in TABLES[n]] for n in NAMES},
    }
    return json.dumps(doc, ensure_ascii=False, separators=(",", ":")) + "\n"


def build(*, check: bool) -> int:
    """生成（或校验）共享色标数据；返回不一致的文件数。"""
    ts = _ts_source()
    js = _json_doc()
    stale: list[str] = []
    targets = [(TS_PATH, ts), (JSON_PATH, js)]
    for path, content in targets:
        if check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                stale.append(str(path.relative_to(_ROOT)))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    if check:
        if stale:
            print("以下生成物已过期（请运行 `python tools/gen_colorscales.py`）：")
            for s in stale:
                print(f"  · {s}")
            return len(stale)
        print(f"色标数据一致：{len(NAMES)} 个色标 × 256 级")
        return 0
    print(f"已生成色标数据：{', '.join(NAMES)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成前端/后端共享的色标数据")
    ap.add_argument("--check", action="store_true", help="只校验（CI 门禁 G6）")
    args = ap.parse_args(argv)
    return build(check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
