/**
 * 1.1 章四个 model 共用的**排版工具**（纯几何 / 纯文本，不含任何物理量）。
 *
 * 为什么不复制四份：`textBlock` / `wrapUnits` / `unitBudget` 是"模型自己折行"的同一套实现
 * （`render2d.ts::drawText` 不拆 `\n`，光栅器的续行行距是 `size+4` 而模型按 15px 排行，
 * 把折行交给渲染层就会让续行压到下一行文字上）。四份拷贝里只要有一份常量漂了，
 * 就会有一张图与另外三张排版不同——1.2 章的四份拷贝已经埋着这个隐患，本章不重复它。
 *
 * 这些函数只做两件事：按共享尺子（`textWidthUnits`：CJK=2、ASCII=1）折行，
 * 以及把折好的行放进 `[top, bottom]` 的净空里。**不碰帧、不碰轴范围、不算物理量。**
 *
 * `SAFE_GAP` 的口径（规则 8）：`axisDecoBands` 给面板下沿的装饰带是
 * `[bottom+3, bottom+15]`（刻度标签）与 `[bottom+16, bottom+28]`（轴名）。
 * 一段文字的盒子高度按 `行数 × (size+4)` 估（`scene.ts::textBoxOf`），
 * 所以"贴边放"必须让最后一行的盒子底 ≤ `bottom−3`：本模块的约定是**留 18px**，
 * 这样即使某条轴上既有刻度又有轴名，文字也不会压进装饰带（校验器当场就会报"规则 8"）。
 */

import type { AxisSpecJson } from "../../shared/src/manifest.js";
import { text, textBoxOf, type TextSpec } from "../../shared/src/scene.js";
import { textWidthUnits } from "../../shared/src/model.js";

/** 面板内说明文字的字号（与 1.2 / 0.0 章一致）。 */
export const TEXT_SIZE = 11;
/** 行距：`textBoxOf` 按 `size + 4 = 15` 估盒，模型自己排行也必须用同一个数。 */
export const TEXT_ROW_H = TEXT_SIZE + 4;
/** 面板底边之上留出的净空（见文件头：避开 `bottom+3..bottom+28` 的两条装饰带）。 */
export const SAFE_GAP = 18;

/**
 * 面板宽度 → 显示宽度单位预算（1 单位 = 7px，与 `textWidthUnits` 同一把尺子）。
 *
 * 留 12px 边距再折算：折出来的行一定装得进面板，两个渲染器都不会再自动折行。
 */
export function unitBudget(panelWidthPx: number): number {
  return Math.max(24, Math.floor((panelWidthPx - 12) / 7));
}

/**
 * 按显示宽度单位折行（CJK=2、ASCII=1；口径同 `model.ts::textWidthUnits`）。
 *
 * 先按空白（含全角空格）切"词"再往行里塞：`k=64`、`ulp=5.551e-17` 这类字段不会被从中间
 * 劈开（劈开在光栅器里读起来像两个数）。单个词比预算还长时才硬切。
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
 * 一段说明文字 → 若干 `text` 层，**一行一个**（`render2d.ts::drawText` 不拆 `\n`）。
 *
 * `anchor`：`"top"` 从 `top` 往下排；`"bottom"` 让整块贴着 `bottom`（盒子底 = 最后一行 y + 15）。
 * 返回数组的长度 = 实际排出的行数（调用方据此把下一块往下推）。
 *
 * 画布小的时候宁可**截断并说明**，也不让文字叠在一起：能放下的行数由 `[top, bottom]` 的
 * 净空决定，最后一行留给"还有 N 行"。这条与 1.2 章 `textBlock` 同一口径。
 */
export function textBlock(
  x: number,
  top: number,
  bottom: number,
  lines: readonly string[],
  color: string,
  budget: number,
  anchor: "top" | "bottom",
): TextSpec[] {
  let pieces = lines.flatMap((body) => wrapUnits(body, budget));
  if (bottom > top) {
    const maxRows = Math.max(1, Math.floor((bottom - top) / TEXT_ROW_H) + 1);
    if (pieces.length > maxRows) {
      const total = pieces.length;
      // 截断提示行**自己也可能折行**（画布越窄越容易），所以按实际折行后的行数回退，
      // 直到"内容 + 提示"正好装进 maxRows 为止——否则提示行会顶到面板的装饰带里。
      for (let kept = Math.max(1, maxRows - 1); kept >= 0; kept -= 1) {
        const tail = kept === 0 ? "…（画布放不下，内容见 pitfalls）" : `…（另有 ${total - kept} 行，画布放不下）`;
        const head = pieces.slice(0, kept);
        const rows = [...head, ...wrapUnits(tail, budget)];
        if (rows.length <= maxRows) {
          pieces = rows;
          break;
        }
      }
    }
  }
  const y = anchor === "top" ? top : Math.max(top, bottom - (pieces.length - 1) * TEXT_ROW_H);
  return pieces.map((body, i) => text(x, y + i * TEXT_ROW_H, body, { role: "note", size: TEXT_SIZE, color, maxWidth: budget * 7 }));
}

/** 面板内文字块的上边界（面板矩形 → 第一行 y）。 */
export function textTop(rect: readonly [number, number, number, number]): number {
  return rect[1] + 4;
}

/** 面板内文字块的下边界（面板矩形 → 最后一行允许的 y；`SAFE_GAP` 见文件头）。 */
export function textBottom(rect: readonly [number, number, number, number]): number {
  return rect[1] + rect[3] - SAFE_GAP;
}

/** manifest 里该轴的标题（文案直接取 manifest，缺省才用回落值——不许模型自己发明轴名）。 */
export function axisTitleOf(axes: readonly AxisSpecJson[] | undefined, id: string, fallback: string): string {
  return axes?.find((a) => a.id === id)?.title ?? fallback;
}

/** manifest 里该轴的 ylabel（同上）。 */
export function axisYlabelOf(axes: readonly AxisSpecJson[] | undefined, id: string, fallback: string): string {
  return axes?.find((a) => a.id === id)?.ylabel ?? fallback;
}

/** 有限值的绝对值上界（只是显示范围，不是物理量）；一个有限值都没有时 undefined。 */
export function absMax(a: ArrayLike<number>): number | undefined {
  let hi = 0;
  let seen = false;
  for (let i = 0; i < a.length; i += 1) {
    const v = Math.abs(a[i] as number);
    if (!Number.isFinite(v)) continue;
    seen = true;
    if (v > hi) hi = v;
  }
  return seen ? hi : undefined;
}

/** 整数刻度（`0, step, 2·step, …`，含两端；用于"剩余有效位数"这类只能整位读的轴）。 */
export function integerTicks(lo: number, hi: number, step: number): number[] {
  const out: number[] = [];
  const s = Math.max(1, Math.round(step));
  for (let v = Math.ceil(lo); v <= Math.floor(hi); v += s) out.push(v);
  return out;
}

/**
 * 统计"贴在地板上的采样点"的**下标**（显示几何，不是物理量）。
 *
 * 对数轴画不出 0，后端把 0 与低于地板的量统一抬到地板值（见 `server/chapters/1.1.py`
 * 的 `np.maximum(raw, floor)`）。到了前端，"贴地板"与"真的等于地板那个数"在像素上
 * 完全一样——这正是本章最容易误读的地方，所以模型必须把这两类点**画出来**：
 * 返回下标供调用方画空心标记，计数供文本如实说明。
 */
export function floorIndices(values: ArrayLike<number>, floor: ArrayLike<number>): number[] {
  const out: number[] = [];
  const n = Math.min(values.length, floor.length);
  for (let i = 0; i < n; i += 1) {
    const v = values[i] as number;
    const f = floor[i] as number;
    if (!Number.isFinite(v) || !Number.isFinite(f) || f <= 0) continue;
    if (v <= f * (1 + 1e-12)) out.push(i);
  }
  return out;
}

/** 把下标数组映射成 `(x, y)` 两列，供 `scatter` 图层使用。 */
export function pick(x: ArrayLike<number>, y: ArrayLike<number>, idx: readonly number[]): { x: Float64Array; y: Float64Array } {
  const xs = new Float64Array(idx.length);
  const ys = new Float64Array(idx.length);
  idx.forEach((i, k) => {
    xs[k] = x[i] as number;
    ys[k] = y[i] as number;
  });
  return { x: xs, y: ys };
}

/**
 * 把帧给的一个**标量地板**铺成长度 `n` 的数组（只用于"哪些点贴在地板上"的显示判断）。
 *
 * 为什么这不是"前端造物理量"：值来自帧 `meta.*_floor`（后端算的），这里做的只是
 * 把它铺成与曲线同长的向量，好让 {@link floorIndices} 能逐点比较。前端不产生新的数。
 */
export function constantArray(n: number, value: number): Float64Array {
  const out = new Float64Array(Math.max(0, n));
  out.fill(value);
  return out;
}

/**
 * 消解文本块重叠：按 y 序保留先到者，与已保留块重叠的**丢弃**并在画布顶部如实计数。
 *
 * 为什么需要它：模型里几块说明各自按"自己的行数"估算下一块的起点（例如
 * `midTop + mid.length * 15 + 4`），而**真实占几行**取决于 `maxWidth` 处的折行——
 * 画布变窄（480px）时每块都会多折出一两行，于是下一块就压上来了。规则 8 的
 * "文本互相压住"在 960px 下全绿、在 480px 下报 3 处，就是这么来的。
 *
 * 这里用**共享的盒子量法**（`textBoxOf`：折行 + `size + 4` 行高）而不是自己数行：
 * 与校验器同一把尺子，判完就一定不再重叠。为什么不"下移"而是"丢弃"：下移会推进
 * 下一条轴的装饰带（刻度/轴名由渲染层自己画，模型看不见），那是另一条规则 8 违规。
 * 诚实口径：宁可少说，不可叠着说，也不可压着轴说。
 */
export function avoidTextOverlap(layers: TextSpec[], rect: readonly [number, number, number, number]): TextSpec[] {
  // 只丢、不移：往下推会推进下一条轴的装饰带（刻度与轴名由渲染层自己画，模型看不见），
  // 那是另一条规则 8 违规。丢掉的块数如实写在画布顶部一行短注里（不静默）。
  const kept: TextSpec[] = [];
  let dropped = 0;
  const sorted = [...layers].sort((a, b) => (a.y ?? 0) - (b.y ?? 0));
  for (const layer of sorted) {
    const box = textBoxOf({ ...layer, y: layer.y ?? 0 });
    if (!box.ok || box.y + box.h > rect[1] + rect[3]) {
      dropped += 1;
      continue;
    }
    const hit = kept.some((prev) => {
      const a = textBoxOf({ ...prev, y: prev.y ?? 0 });
      const ox = Math.min(a.x + a.w, box.x + box.w) - Math.max(a.x, box.x);
      const oy = Math.min(a.y + a.h, box.y + box.h) - Math.max(a.y, box.y);
      return ox > 2 && oy > 2;
    });
    if (hit) {
      dropped += 1;
      continue;
    }
    kept.push(layer);
  }
  if (dropped > 0) {
    kept.push(
      text(rect[0], rect[1], `文本放不下：另有 ${dropped} 行未显示（完整口径见章节笔记与角标）`, {
        role: "note",
        size: 11,
        color: "#718096",
      }),
    );
  }
  return kept;
}
