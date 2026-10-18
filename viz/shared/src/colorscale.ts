/**
 * 色标（colormap）：256 级 RGB 查找表的查表、归一化、RGBA 打包与 CSS 文本。
 *
 * 数据源
 * ------
 * 真源是 `viz/shared/colorscales.json`（由 `tools/gen_colorscales.py` 生成，Python 侧
 * `tools/raster.py` 用同名同值 LUT，门禁 G6 检查一致）；本模块只 import 生成的
 * {@link ./colorscale_data.gen.js}，**不手改、不重算**，从而保证 TS 光栅与 Python 光栅
 * 逐像素同色。
 *
 * 数学定义
 * --------
 * * 线性归一化：`t = (v − vmin) / (vmax − vmin)`；
 * * 对数归一化：`t = log10(v / vmin) / log10(vmax / vmin)`（要求 `vmin > 0`、`vmax > vmin`）；
 * * 查表：`idx = round(t · 255)`，`rgb = LUT[3·idx : 3·idx+3]`；
 * * 相对亮度（Rec.709 / sRGB）：`Y = 0.2126·R̂ + 0.7152·Ĝ + 0.0722·B̂`，
 *   其中 `Ĉ` 是 sRGB 反伽马后的线性分量（见 {@link luminance}）。
 *
 * 不可见（透明）口径：`NaN` / `±Infinity`、对数刻度下的非正值、以及低于 `clipBelow`
 * 的值一律输出 `alpha = 0`——"没有数据"必须看起来像没有数据，而不是像某个颜色。
 *
 * 全部为纯函数，无 DOM / canvas / 第三方依赖。
 */

import { CMAP_NAMES, CMAP_LUT, type CmapName } from "./colorscale_data.gen.js";

export { CMAP_NAMES };
export type { CmapName };

/** 色标级数（LUT 行数）；LUT 字节数 = 3 × LEVELS = 768。 */
const LEVELS = 256;

/** 运行时校验色标名（场景 JSON 里的 `cmap` 字段可能来自外部输入）。 */
function requireCmap(name: CmapName): Uint8Array {
  const table = CMAP_LUT[name];
  if (table === undefined) {
    throw new RangeError(`未知色标：${String(name)}（可用：${CMAP_NAMES.join(", ")}）`);
  }
  return table;
}

/**
 * 取某个色标的 256×3 RGB 查找表（长度 768，行主序，取值 0..255）。
 *
 * 返回的是模块内共享常量，**调用方不得修改**（只读使用；需要可变副本请自行 `slice()`）。
 * 名称非法 → `RangeError`。
 */
export function cmapTable(name: CmapName): Uint8Array {
  return requireCmap(name);
}

/**
 * 查表：`t` 截断到 `[0, 1]` 后按 `round(t · 255)` 取 RGB（取值 0..255）。
 *
 * `NaN` 按 0 处理（取色标起点），`+Infinity` → 末色，`-Infinity` → 首色。
 * `gray` 是恒等映射：`lookupCmap("gray", t) = [round(255t)] × 3`。
 */
export function lookupCmap(name: CmapName, t: number): [number, number, number] {
  const table = requireCmap(name);
  let tt: number;
  if (Number.isNaN(t)) tt = 0;
  else if (t === Infinity) tt = 1;
  else if (t === -Infinity) tt = 0;
  else tt = t < 0 ? 0 : t > 1 ? 1 : t;
  const idx = Math.min(LEVELS - 1, Math.max(0, Math.round(tt * (LEVELS - 1))));
  const base = idx * 3;
  return [table[base], table[base + 1], table[base + 2]];
}

/**
 * 查表并格式化为 CSS `rgb(r,g,b)` 字符串（`t` 截断规则同 {@link lookupCmap}）。
 */
export function cmapToCss(name: CmapName, t: number): string {
  const [r, g, b] = lookupCmap(name, t);
  return `rgb(${r},${g},${b})`;
}

/**
 * 把数据值归一化到 `[0, 1]`（越界截断）。
 *
 * * 线性：`(v − vmin)/(vmax − vmin)`；`vmin === vmax`（常数场）→ 返回 `0.5`
 *   （色标中点：在发散色标上不会被误读成某一侧的极值）；
 * * 对数（`logScale = true`）：`log10(v/vmin)/log10(vmax/vmin)`；要求
 *   `vmin > 0` 且 `vmax > 0`（否则 `RangeError`）、`vmin !== vmax`（否则 `RangeError`）；
 *   数据 `v ≤ 0` 在对数刻度下无定义 → 返回 `NaN`；
 * * `v` 为 `NaN` 或 `±Infinity` → `NaN`；`vmin`/`vmax` 非有限 → `RangeError`。
 */
export function normalizeValue(v: number, vmin: number, vmax: number, logScale = false): number {
  if (!Number.isFinite(vmin) || !Number.isFinite(vmax)) {
    throw new RangeError(`vmin/vmax 必须是有限数，收到 [${vmin}, ${vmax}]`);
  }
  if (!Number.isFinite(v)) return NaN;
  if (logScale) {
    if (!(vmin > 0) || !(vmax > 0)) {
      throw new RangeError(`对数色标要求 vmin > 0 且 vmax > 0，收到 [${vmin}, ${vmax}]`);
    }
    if (vmin === vmax) {
      throw new RangeError(`对数色标要求 vmin ≠ vmax，收到 vmin = vmax = ${vmin}`);
    }
    if (!(v > 0)) return NaN;
    const t = Math.log10(v / vmin) / Math.log10(vmax / vmin);
    return t < 0 ? 0 : t > 1 ? 1 : t;
  }
  if (vmin === vmax) return 0.5;
  const t = (v - vmin) / (vmax - vmin);
  return t < 0 ? 0 : t > 1 ? 1 : t;
}

/**
 * 把一维数据场映射成 RGBA 字节（长度 `4N`，供 `ImageData` 直接使用）。
 *
 * * RGB 来自 {@link lookupCmap}；不透明像素 `alpha = 255`；
 * * `NaN` / `±Infinity` → 四字节全 0（**完全透明**），不画成"某个颜色"；
 * * `opts.clipBelow`：以**数据单位**给出的下阈值，`v < clipBelow` 的像素透明
 *   （用于"低于噪声地板/阈值不显示"的场景）；
 * * `opts.logScale`：对数归一化；`vmin ≤ 0` 或 `vmax ≤ 0` 或 `vmin === vmax` → `RangeError`
 *   （宁可抛错，也不悄悄给出一张 NaN 图）；此时数据 `v ≤ 0` 一并透明；
 * * 线性且 `vmin === vmax`（常数场）→ 全部取色标中点（与 {@link normalizeValue} 一致）。
 */
export function applyCmap(
  name: CmapName,
  values: ArrayLike<number>,
  vmin: number,
  vmax: number,
  opts: { logScale?: boolean; clipBelow?: number } = {},
): Uint8ClampedArray {
  const table = requireCmap(name);
  const logScale = opts.logScale === true;
  const clipBelow = opts.clipBelow;
  if (clipBelow !== undefined && !Number.isFinite(clipBelow)) {
    throw new RangeError(`clipBelow 必须是有限数，收到 ${String(clipBelow)}`);
  }
  // 归一化参数的合法性先整体校验一次，避免"每个像素都抛错"的半成品输出
  if (!Number.isFinite(vmin) || !Number.isFinite(vmax)) {
    throw new RangeError(`vmin/vmax 必须是有限数，收到 [${vmin}, ${vmax}]`);
  }
  if (logScale && (!(vmin > 0) || !(vmax > 0))) {
    throw new RangeError(`对数色标要求 vmin > 0 且 vmax > 0，收到 [${vmin}, ${vmax}]`);
  }
  if (logScale && vmin === vmax) {
    throw new RangeError(`对数色标要求 vmin ≠ vmax，收到 vmin = vmax = ${vmin}`);
  }

  const n = values.length;
  const out = new Uint8ClampedArray(4 * n);
  for (let i = 0; i < n; i++) {
    const v = values[i];
    const o = 4 * i;
    if (!Number.isFinite(v)) continue; // alpha 已为 0
    if (clipBelow !== undefined && v < clipBelow) continue;
    const t = normalizeValue(v, vmin, vmax, logScale);
    if (Number.isNaN(t)) continue; // 对数刻度下的非正值
    const idx = Math.min(LEVELS - 1, Math.max(0, Math.round(t * (LEVELS - 1))));
    const base = idx * 3;
    out[o] = table[base];
    out[o + 1] = table[base + 1];
    out[o + 2] = table[base + 2];
    out[o + 3] = 255;
  }
  return out;
}

/**
 * sRGB 分量反伽马：`0..255` → 线性光 `0..1`（IEC 61966-2-1 折线，阈 0.04045）。
 */
function srgbToLinear(channel: number): number {
  const s = channel / 255;
  return s <= 0.04045 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4);
}

/**
 * Rec.709 / sRGB **相对亮度** `Y ∈ [0, 1]`：
 * `Y = 0.2126·R̂ + 0.7152·Ĝ + 0.0722·B̂`，`Ĉ` 为反伽马后的线性分量。
 *
 * 输入是 `0..255` 的 sRGB 通道值（与 {@link lookupCmap} 的输出同量纲），
 * 越界先截断到 `[0, 255]`；任一通道非有限 → `NaN`。
 * 参考：`luminance(0,0,0) = 0`，`luminance(255,255,255) = 1`。
 */
export function luminance(r: number, g: number, b: number): number {
  if (!Number.isFinite(r) || !Number.isFinite(g) || !Number.isFinite(b)) return NaN;
  const clamp = (c: number): number => (c < 0 ? 0 : c > 255 ? 255 : c);
  return (
    0.2126 * srgbToLinear(clamp(r)) +
    0.7152 * srgbToLinear(clamp(g)) +
    0.0722 * srgbToLinear(clamp(b))
  );
}

/** 百分比文本：最多 4 位小数并去掉尾随 0（`6.6667%`、`0%`、`100%`）。 */
function percent(t: number): string {
  const s = (t * 100).toFixed(4);
  return `${s.includes(".") ? s.replace(/\.?0+$/, "") : s}%`;
}

/**
 * 生成图例用 CSS 渐变串（水平方向，采样 LUT，不做插值/线性光混合）。
 *
 * `stops` 为采样点数（缺省 16）：向下取整，小于 2 时按 2 处理（`0%` 与 `100%` 两端
 * 必在）。返回形如
 * `linear-gradient(to right, rgb(68,1,84) 0%, …, rgb(253,231,37) 100%)`。
 */
export function cmapCssGradient(name: CmapName, stops = 16): string {
  const k = Math.max(2, Math.floor(Number.isFinite(stops) ? stops : 16));
  const parts: string[] = [];
  for (let i = 0; i < k; i++) {
    const t = i / (k - 1);
    parts.push(`${cmapToCss(name, t)} ${percent(t)}`);
  }
  return `linear-gradient(to right, ${parts.join(", ")})`;
}
