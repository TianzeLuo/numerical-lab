/**
 * 1.3 章四个 model 共用的**显示几何**小工具（纯函数，无 DOM、无第三方依赖、不读帧）。
 *
 * 为什么单独一个文件：四个 viz 都要做同样三件事——把一段说明折成"一行一个 `text` 层"、
 * 把帧里的数排成紧凑的一行、把面板宽度换算成显示宽度单位预算。1.2 章把它们各自复制
 * 在 4 个 model 里（`unit_ball.model.ts::textBlock/wrapUnits/unitBudget`），本章有 4 个
 * viz 且文本更多，复制四份必然漂移（"同样的折行规则三处不同"正是 1,2 章踩过的坑）。
 *
 * 三条纪律（与 `unit_ball.model.ts` 的文件头第 6 条同源）：
 *
 * 1. **一段话不写成一个多行字符串**：`render2d.ts::drawText` 不拆 `\n`（`raster.py` 拆），
 *    多行字符串会让浏览器与光栅器的排版分叉。这里统一先折行、再一行一个 `text` 层。
 * 2. **折行用与校验器同一把尺子**：`model.ts::textWidthUnits`（CJK=2、ASCII=1、1 单位 ≈ 7px）。
 *    自己数 `String.length` 会让角标/文本盒的估算与校验器对不上。
 * 3. **放不下就截断并说明**：`bottom` 之间净空不够时宁可写"另有 N 行"也不让两行文字叠在一起
 *    （叠字等于看不见，见规则 1 的精神）。
 */

import { applyOverflow, clipBadgeText, placeBadge, textWidthUnits, type Rect } from "../../shared/src/model.js";
import { badge, pitfallsBand, text, type Scene, type TextSpec } from "../../shared/src/scene.js";

/** 面板排版用的固定行高与字号（与 1.2 章一致；两个渲染器都不会再折行）。 */
export const TEXT_SIZE = 11;
export const TEXT_ROW_H = 15;

/**
 * 面板宽度 → 显示宽度单位预算（size 11 时 1 单位 ≈ 7px）。
 *
 * 留 12px 边距再折算，保证两个渲染器都不会自动折行——`raster.py` 的续行间距
 * （`cell_h + 4`）比这里的 15px 行距大，一旦让渲染器自己折行，续行就会压到下一行文字上。
 */
export function unitBudget(panelWidthPx: number): number {
  return Math.max(24, Math.floor((panelWidthPx - 12) / 7));
}

/**
 * 按显示宽度单位折行：先按空白（含全角空格）切词，再往行里塞。
 *
 * 切词是为了不让 `ρ=0.9991`、`k=3/63` 这类字段被从中间劈开——劈开在光栅器里读起来像两个数。
 * 单个词比预算还长时才硬切。
 */
export function wrapUnits(body: string, budget: number): string[] {
  const out: string[] = [];
  let line = "";
  let used = 0;
  const push = (): void => {
    if (line !== "") out.push(line);
    line = "";
    used = 0;
  };
  for (const word of body.split(/(?<=[\s　])/)) {
    const w = textWidthUnits(word);
    if (used > 0 && used + w > budget) push();
    if (w <= budget) {
      line += word;
      used += w;
      continue;
    }
    let piece = "";
    let pieceUsed = 0;
    for (const ch of word) {
      const cw = textWidthUnits(ch);
      if (pieceUsed + cw > budget && piece !== "") {
        out.push(piece);
        piece = "";
        pieceUsed = 0;
      }
      piece += ch;
      pieceUsed += cw;
    }
    line = piece;
    used = pieceUsed;
  }
  push();
  return out;
}

/**
 * 一段说明 → 若干 `text` 层（**一行一个**）。
 *
 * `anchor="top"` 从 `top` 往下排；`anchor="bottom"` 让整块贴着 `bottom`。
 * 返回数组的长度 = 实际行数（调用方据此继续往下排版）。`bottom > top` 时按净空截断。
 */
export function textBlock(
  x: number,
  top: number,
  bottom: number,
  lines: readonly string[],
  color: string,
  budget: number,
  anchor: "top" | "bottom" = "top",
): TextSpec[] {
  let pieces = lines.flatMap((body) => wrapUnits(body, budget));
  if (bottom > top) {
    // 每行占 **15px 的盒子**（`size + 4`），不是"基线间距"：`maxRows` 必须是
    // `floor((bottom-top)/15)`，多给一行就会让最后一行的盒底越过 `bottom`——
    // 而 `bottom` 现在等于底部 pitfalls 面板的顶边（见 `textFloor`），越过去就是"压住"（校验器判违规）。
    const maxRows = Math.max(1, Math.floor((bottom - top) / TEXT_ROW_H));
    if (pieces.length > maxRows) {
      const kept = Math.max(1, maxRows - 1);
      pieces = [...pieces.slice(0, kept), `…（另有 ${pieces.length - kept} 行，画布放不下）`];
    }
  }
  const y = anchor === "top" ? top : Math.max(top, bottom - (pieces.length - 1) * TEXT_ROW_H);
  return pieces.map((body, i) =>
    text(x, y + i * TEXT_ROW_H, body, { role: "note", size: TEXT_SIZE, color, maxWidth: budget * 7 }),
  );
}

/**
 * 紧凑数字：先试定点（最多 `sig` 位有效数字），太大/太小才转科学记数。
 *
 * 为什么不用 `Number.prototype.toPrecision` 一把梭：`1e-18` 会印成 `1.000e-18`、
 * `0.377` 会印成 `0.3770`，在一行要放五六个量的面板里太占地方（角标/文本都按显示宽度算预算）。
 * 也不用 `axes.ts::formatNumber`：它把 `[1e-3, 1e5]` 之外一律转 `e` 记法，
 * 而这里 `1e-9` 与 `1e-18` 同屏出现，需要的是"能一眼比大小"的紧凑写法。
 */
export function fmt(v: number | undefined, sig = 3): string {
  if (v === undefined || !Number.isFinite(v)) return "—"; // NaN/±∞ 与"缺值"同等待遇：不写 NaN
  const p = Math.min(100, Math.max(1, Math.round(sig)));
  const a = Math.abs(v);
  if (v !== 0 && (a >= 1e5 || a < 1e-3)) {
    const e = v.toExponential(p - 1);
    const [mant, exp] = e.split("e");
    return `${mant}e${exp}`;
  }
  return v.toPrecision(p);
}

export function fmtOr(v: number | undefined, sig = 3): string {
  return fmt(v ?? Number.NaN, sig);
}

/** `标签=值` 的一行（缺值时**整项省略**，绝不写 0 冒充——见 PROTOCOLS §1.3）。 */
export function kv(label: string, value: number | undefined, sig = 3, suffix = ""): string | undefined {
  return value === undefined ? undefined : `${label}=${fmt(value, sig)}${suffix}`;
}

/** 把若干可缺项拼成一行（全缺时返回 `undefined`，调用方据此决定要不要加这一层）。 */
export function joinParts(parts: readonly (string | undefined)[], sep = "　"): string | undefined {
  const kept = parts.filter((p): p is string => p !== undefined && p !== "");
  return kept.length > 0 ? kept.join(sep) : undefined;
}

/** 一个轴的**装饰带之外**的可用顶部（正文不要再往上跑，否则挤进轴标题带）。 */
export function belowTitle(rect: Rect): number {
  return rect[1] + 4;
}

/**
 * 规则 5 的落地：**溢出/地板角标永远存在**，而且说清是"没有越界"还是"越了多少"。
 *
 * 为什么不用 `applyOverflow` 一把梭：它只在 `n_bad > 0` 时立角标，于是一个"本帧
 * 干净"的场景会**一个溢出角标都没有**——读图的人无从判断"没画出来的点"到底是不存在，
 * 还是被悄悄丢了。这里把两种情况都写成角标文字：干净时明说"无越界"，
 * 有情况时把 `extra` 里每条具体原因拼进去（哪条路线没有有限解、几段几何被裁掉）。
 */
export function overflowBadgeFor(
  scene: Scene,
  nBad: number,
  extra: string[],
  opts: { slot: number; width: number },
): void {
  applyOverflow(scene, nBad, { slot: opts.slot, width: opts.width });
  // 「没有越界」是一种正常状态，不该占一枚红色角标。只有真的丢了数据才挂 overflow；
  // 调用方的 `extra` 必须是**本帧特有**的数据问题（哪条路线没有有限解、哪个值非有限），
  // 不能把「地板点没有画进对数轴」这种显示约定塞进来。
  if (!scene.overflow.flag) {
    scene.overflow = { flag: false, note: "" };
    return;
  }
  // 顺序有意：具体原因在前，通用范围提示在后；角标超宽会从尾部截断。
  const detail = [...extra, scene.overflow.note].filter((t) => t.length > 0).join("；");
  scene.overflow = { flag: true, note: detail };
  const idx = scene.badges.findIndex((b) => b.kind === "overflow");
  const node = placeBadge(badge("overflow", clipBadgeText(`溢出：${detail}`), 8, 8), opts.slot, opts.width);
  if (idx >= 0) scene.badges[idx] = node;
  else scene.badges.push(node);
}

/**
 * 文本块的可用底边：**渲染层会在画布底部现画一块不透明的 pitfalls 面板**，
 * 模型看不见它（面板高度随 `scene.pitfalls` 的折行数变），所以文本块的底边不能直接用
 * `base.bottom`——那正是"文字被面板压住"的来源（校验器两个方向 >2px 就判违规）。
 *
 * 这里按 `pitfallsBand(scene)` 的真实盒子把底边收上来 4px（留一点白）。
 * 行数因此少一行时，`textBlock` 自己会写"另有 N 行，画布放不下"——宁可少画一行并说明，
 * 也不让最后一行的下半截被面板吃掉。
 */
export function textFloor(scene: Scene, fallbackBottom: number): number {
  const band = pitfallsBand(scene);
  const top = band === null ? Number.POSITIVE_INFINITY : band[1];
  return Math.max(60, Math.min(fallbackBottom, top - 4));
}

/**
 * 内容底边：`stackedPanels` 按固定 102px 页脚留白算 `bottom`，但**帧里的 pitfalls 文案
 * 长短不一**——1.3 的 4 条帧里最长的那条会把底部面板撑到 150px 高（实测 band top=442，
 * 而 `base.bottom=498`），于是副轴与文本的下半截会被面板盖住"。
 * 这里用 `pitfallsBand` 的真实盒子把底边一起收上来：轴、文本、面板三者的底边从此一致。
 */
export function contentBottom(
  frame: { pitfalls?: string },
  width: number,
  height: number,
  baseBottom: number,
): number {
  const band = pitfallsBand({ canvas: { w: width, h: height }, pitfalls: frame.pitfalls });
  const top = band === null ? Number.POSITIVE_INFINITY : band[1];
  return Math.max(120, Math.min(baseBottom, top - 6));
}
