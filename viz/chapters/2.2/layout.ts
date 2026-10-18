/**
 * 2.2 章四个 model 共用的**版式**与小工具。
 *
 * 为什么单独一层：本章四个 viz 的差别在"画什么量"，不在"怎么排"——都是
 * 「一列主面板 + 若干自适应副轴（log/线性）」，且都要在轴下放两行数值/结论。
 * 版式算一次、四个模型共用，才不会出现"同一个 viz 里两处留白口径不同"。
 *
 * 这一层只做**显示几何**（矩形、折行、数据→像素），不解释任何数值含义：
 * 所有数值都来自帧（P8），这里最"物理"的动作是把数组里的数取出来摆到像素上。
 */

import { niceLimits } from "../../shared/src/axes.js";
import { dataToPixel } from "../../shared/src/geometry.js";
import { clamp, headerBand, resolveTextOverlaps, textWidthUnits, type Rect } from "../../shared/src/model.js";
import { pitfallsBand, text, textBoxOf, type Axis, type Scene, type TextSpec } from "../../shared/src/scene.js";

/**
 * 2.2 章四个 viz 共用的**图幅契约**（这一章最容易踩的一条几何约束）。
 *
 * 为什么需要它：`axisDecoBands` 里 y 刻度标签带是 `[left−41, y, 35, h]`、
 * y 轴名带是 `[left−56, y, 14, h]`——它们**沿整条轴的高度**延伸；而轴标题带
 * `[x, top−18, w, 16]` 沿**整条轴的宽度**延伸。于是"两行轴之间那条空隙"里：
 *   * 写进轴的正下方（x 落在轴的宽度内）→ 撞下一格的轴标题带；
 *   * 写在轴右侧的空栏里 → 竖直方向仍与下一格的标题带同高，横向却越过了轴的右边界。
 * 结论：文字只能落在**所有轴矩形之外**，或者落在**轴标题带之外的那一段竖直区间**里。
 *
 * 本层的做法：
 *   ① 轴只占左边 `AXIS_W`；
 *   ② 说明文字全都写在 `RIGHT_X = 74 + AXIS_W + 34` 往右那一栏（越过最宽的装饰带）；
 *   ③ 每个面板自己的说明写在它**下方那条 5px 起、最多 60px 高**的区间里
 *      （`bandRect`），下限由 `tailBand` 兜住——不会压到下方的陷阱面板。
 */
export const AXIS_W = 430;
/** 说明栏的左边界：必须 ≥ 轴右边界 + 2×`AXIS_DECO.top`（轴标题带与 xlabel 带同宽）。 */
export const RIGHT_X = 74 + AXIS_W + 14;

/**
 * 轴的实际宽度：默认 `AXIS_W`，画布太窄时**按画布收缩**（窄画布上轴不许越出画面）。
 *
 * 这是 `validateScene` 之外的一条硬要求（也是 web 端窄窗口下的现实）：轴越出画布
 * 时右边那一栏说明也没地方放了，于是整块结论消失。宁可轴窄一点。
 */
export function axisWidth(canvasW: number): number {
  return Math.max(120, Math.min(AXIS_W, Math.round(canvasW - 74 - 16)));
}

/** 说明栏的左边界（窄画布时跟着轴一起左移）。 */
export function rightX(canvasW: number): number {
  return Math.min(RIGHT_X, 74 + axisWidth(canvasW) + 14);
}
/** 面板区顶端：角标区之下（`headerBand(960) = 114`）再让 8px。 */
export const PANEL_TOP = 122;
/** 不存在下一格时（最后一行）说明文字的下限与陷阱面板之间的安全边距。 */
export const TAIL_SAFE = 10;

/** 说明栏在给定宽度下有多少像素可用（扣掉左右留白）。 */
export function textColWidth(width: number): number {
  return Math.max(80, width - 8 - rightX(width));
}

/** 右栏是否放得下（窄画布时逐格说明只能省略，见 `tooNarrowNote`）。 */
export function rightColumnFits(width: number): boolean {
  return textColWidth(width) >= 150;
}

/** 面板区顶端：角标区之下（`headerBand(960) = 114`）再让 `HEADER_GAP`。 */
export function panelTop(width: number, headerExtra = 14): number {
  // `headerExtra = 14` 而不是 8：角标第 4 行占 `[88, 106]`，而轴标题带是
  // `[top − 18, top − 2]`。`+8` 时标题带上沿 104 会与第 4 行角标叠 2px
  // （校验器放行，但看上去就是两行字压在一起）。
  return headerBand(width) + headerExtra;
}

/** 面板行之间必须留出的净空：一行的 x 刻度带（15px）+ 下一行的轴标题带（16px）+ 4px。 */
export const ROW_GAP22 = 34;

/**
 * 面板区底端。
 *
 * 用**陷阱面板的上沿**当硬边界，而不是"再让 10px"：一行说明 15px，让掉 10px 之后
 * 最后一行往往只剩 5px 可用空间——于是最后一块说明整体消失（`bandRect` 返回 0 行），
 * 图上就少一栏结论，而且不报错。宁可让最后一块文字贴着陷阱面板，也不能悄悄不画。
 */
export function contentBottom(scene: Scene, extra = 0): number {
  return pitfallsBoxTop(scene) - 2 - extra;
}

/**
 * 收尾：把文字层交给共享层的 `resolveTextOverlaps` 过一遍。
 *
 * 两个新规则都靠它：
 *   * 文本**互相不许重叠**（>2px）——按 y 序保留先到的、丢弃后来的，并在画布顶部留一行
 *     「窄画布：另有 N 行说明未显示」（只丢不移：下移会推进轴的装饰带，那是规则 8）；
 *   * 文本**不许被陷阱面板压住**——`limit` 取陷阱面板上沿，超出的整块直接丢。
 *
 * 为什么放最后而不是"算准了就不需要"：画布宽度、数据长度、折行后的字数都会变，
 * 手算的净空只对默认 960×600 成立。交给共享层是**唯一**不随版本漂移的做法。
 */
export function finishTexts(scene: Scene, width: number, limit = contentBottom(scene)): void {
  scene.texts = resolveTextOverlaps(scene.texts, [8, 24, Math.max(80, width - 16), Math.max(0, limit - 24)]);
}

export interface PanelRow {
  rect: Rect;
  /** 这一行说明文字能写到哪里（下一行轴标题带上沿，或陷阱面板上沿 − 安全边距）。 */
  limit: number;
}

/**
 * 竖直排列 `heights.length` 个面板行。
 *
 * 与 `columnLayout` 的区别：**行高由调用方直接给**（面板需要多高是模型自己最清楚的），
 * 这里只负责"从哪开始、到哪结束、说明文字写到哪"这三件事，并且
 * 「所有面板都落在 `[panelTop, contentBottom]` 内」是硬保证（放不下就一起等比缩）。
 */
export function panelRows(
  width: number,
  heights: readonly number[],
  top: number,
  bottom: number,
  gap = ROW_GAP22,
  /**
   * 最后一行面板下面要给说明文字留的高度（px）。
   *
   * 为什么必须显式留：最后一行下面就是陷阱面板，`bandRect` 的 `limit` 直接取
   * `contentBottom`——不留的话那一块说明会被 `finishTexts` 整体丢掉，
   * 而它通常是全图最要紧的一句（「折线能量恒 0」「多项式溢出」……）。
   */
  tailRoom = 0,
): PanelRow[] {
  const n = heights.length;
  const avail = Math.max(120, bottom - top - gap * (n - 1) - tailRoom);
  const want = heights.reduce((a, b) => a + Math.max(56, b), 0);
  const scale = want > avail ? avail / want : 1;
  const out: PanelRow[] = [];
  let y = top;
  for (let i = 0; i < n; i += 1) {
    // 最后一行吃掉剩余高度：它下面就是陷阱面板，不需要再留说明文字的净空
    // （那一行的说明走 `bandRect` 的 `contentBottom` 分支）。其余行**必须**留够
    // `gap`：说明文字只有落在行距里才不压到下一格的轴标题带。
    const h =
      i === n - 1
        ? Math.max(20, Math.round(bottom - y - tailRoom))
        : Math.max(56, Math.round((heights[i] as number) * scale));
    out.push({ rect: [74, Math.round(y), axisWidth(width), h], limit: 0 });
    y += h + gap;
  }
  for (let i = 0; i < n; i += 1) {
    // `limit` = 本行说明文字的**下沿**（`bandRect` 再按 15px 行高折算行数）。
    //
    // 取"下一行的**底边**"而不是"下一行轴标题带的上沿"：说明文字全写在右侧那一栏
    // （`RIGHT_X` 在所有轴的右边界之外），轴标题带只占轴的宽度，横向根本不相交——
    // 之前按标题带让位，白白吃掉一个行距，最后一块结论只能被 `finishTexts` 丢掉。
    out[i]!.limit = i + 1 < n ? (out[i + 1] as PanelRow).rect[1] + (out[i + 1] as PanelRow).rect[3] : bottom;
  }
  return out;
}

/** 一行的说明矩形（顶锚书写）：`rect[1]` 是行底 +5，高度按"到 `limit` 放得下几行"给。 */
export function bandRect(row: PanelRow, width: number, maxRows = 4): { rect: Rect; maxRows: number } {
  const y = row.rect[1] + row.rect[3] + 5;
  // 行数按"到 `limit` 还剩多少竖直空间"算。
  //
  // 注意 `textBoxOf` 里文本盒是 `[y, y + size + 4)`——`y` 就是**基线**，
  // 所以第 k 行的下沿是 `y + 11 + 15k + 15`，不是 `y + 15k`。
  // 早先按 `(limit − y − 4)/15` 算，最后一行刚好探出 `limit` 15px，
  // 于是**整行结论被 `finishTexts` 丢掉**（而且只丢最后一行，很难看出来）。
  const room = Math.max(0, Math.floor((row.limit - y - 26) / 15) + 1);
  const rows = Math.max(0, Math.min(maxRows, room));
  return { rect: [rightX(width), y, textColWidth(width), Math.max(0, rows * 15 - 1)], maxRows: rows };
}

/**
 * 右栏文字流：把若干"结论块"从上往下排，块首尽量对齐它那一行面板的顶边。
 *
 * 为什么不用"每块绑在自己面板下方"：右栏（x ≥ `rightX`）在所有轴之外，
 * 整条画布高度都能用；绑在面板下方的话，最后一块只剩陷阱面板前面那几十像素，
 * 一折行就丢掉一整句结论。这里让文字**顺流而下**：上一块排完，下一块接着排，
 * 但不会早于它自己那一行的顶边（这样"哪块说的是哪格"还能看出来）。
 *
 * 返回没排下的行数（调用方应写进降级角标，绝不静默丢）。
 */
export function packColumn(
  scene: Scene,
  width: number,
  blocks: readonly (readonly string[])[],
  color: string | readonly string[],
  starts: readonly number[],
  top = panelTop(width),
  bottom = contentBottom(scene),
): number {
  const colorOf = (i: number): string => (typeof color === "string" ? color : ((color[i] as string) ?? "#4a5568"));
  const colW = textColWidth(width);
  const budget = unitBudget(colW);
  const size = 11;
  const rowH = size + 4;
  let cursor = top;
  let dropped = 0;
  for (let i = 0; i < blocks.length; i += 1) {
    const start = Math.max(cursor, Math.round(starts[i] ?? top));
    const lines = (blocks[i] ?? []).flatMap((line) => wrapUnits(line, budget));
    lines.forEach((body, k) => {
      const y = start + k * rowH + (rowH - 4);
      // `textBoxOf` 里盒子的下沿是 `y + size + 4`：越界就整行不画（画了也会被校验器丢掉）。
      if (y + size + 4 > bottom) {
        dropped += 1;
        return;
      }
      scene.texts.push(text(rightX(width), y, body, { role: "note", size, color: colorOf(i), maxWidth: budget * 7 }));
    });
    cursor = start + lines.length * rowH + 4;
  }
  return dropped;
}

/**
 * 把若干"结论块"装进若干行面板的右栏——**按物理行从最后一块往前装，放不下的行上移**。
 *
 * 为什么需要它：每块说明能写几行由"行距"和"陷阱面板上沿"共同决定（`bandRect`），
 * 而这两件事只对默认画布成立。硬写"这块 3 行、那块 2 行"，换个画布高度就会有一整块
 * 结论被 `finishTexts` 静默丢掉——被丢的往往是全图最要紧的那句。
 *
 * 关键细节：装的是**折行之后的物理行**（`wrapUnits`），不是"逻辑行"。
 * 一条长句折成两行时，逻辑行计数会骗人——实测就这样丢过一整句（「折线能量恒 0」）。
 * 上移是安全的：文字都在右栏（所有轴之外），块与块之间只按先后顺序排，不指向具体像素。
 *
 * 返回被丢弃的行数（只有连第一块都装不下时才会 >0；调用方应该把它写进降级角标）。
 */
export function packBands(
  scene: Scene,
  rows: readonly PanelRow[],
  width: number,
  blocks: readonly (readonly string[])[],
  color: string | readonly string[],
  perRowMax = 6,
): number {
  const colorOf = (i: number): string => (typeof color === "string" ? color : ((color[i] as string) ?? "#4a5568"));
  const caps = rows.map((r) => bandRect(r, width, perRowMax).maxRows);
  const out: string[][] = rows.map(() => []);
  let carry: string[] = [];
  for (let i = rows.length - 1; i >= 0; i -= 1) {
    const { rect } = bandRect(rows[i] as PanelRow, width, perRowMax);
    const budget = unitBudget(rect[2]);
    const phys = [...(blocks[i] ?? []), ...carry].flatMap((line) => wrapUnits(line, budget));
    carry = [];
    const cap = Math.max(0, caps[i] as number);
    out[i] = phys.slice(0, cap);
    if (phys.length > cap) carry = phys.slice(cap);
  }
  for (let i = 0; i < rows.length; i += 1) {
    const lines = out[i] as string[];
    if (lines.length === 0) continue;
    const { rect } = bandRect(rows[i] as PanelRow, width, perRowMax);
    const size = 11;
    const rowH = size + 4;
    lines.forEach((body, k) => {
      scene.texts.push(
        text(rect[0], rect[1] + rowH - 4 + k * rowH, body, {
          role: "note",
          size,
          color: colorOf(i),
          maxWidth: unitBudget(rect[2]) * 7,
        }),
      );
    });
  }
  return carry.length;
}

/** 把说明写进一行的右栏（`bandRect` + `textInRect` 的组合；`rows=0` 时什么都不画）。 */
export function bandText(
  scene: Scene,
  row: PanelRow,
  width: number,
  lines: readonly string[],
  color: string,
  maxRows = 4,
): void {
  const { rect, maxRows: rows } = bandRect(row, width, maxRows);
  if (rows <= 0) return;
  scene.texts.push(...textInRect(rect, lines, color, { anchor: "top", maxRows: rows }));
}

/** 画布太窄时的替代说明（写在陷阱面板之上）。 */
export function tooNarrowNote(rect: Rect, color: string, width: number): TextSpec[] {
  return textInRect(
    rect,
    [
      `画布宽 ${Math.round(width)}px：右侧说明栏放不下（需要 ≥ ${rightX(width) + 150 + 8}px），` +
        "本图的逐格数值说明已省略；曲线与参考线本身不受影响，陷阱面板仍在最下方。",
    ],
    color,
    { anchor: "top" },
  );
}

/** 版式常量（像素）。轴高按"信息量"分档，别让竖排文字挤成一条缝。 */
export const AXIS_H = {
  /** 主面板（曲线 + 梳齿 + 控制多边形）。 */
  main: 190,
  /** 只有一条曲线 + 两条参考线的副轴。 */
  aux: 86,
  /** 要同时画两条曲线的副轴（S''、误差）。 */
  plot: 118,
  /** 概览轴（误差随 n 的增长）。 */
  compact: 118,
  /** 固定参数域的示意轴（B 样条）。 */
  fixed: 150,
} as const;

/** 行间竖直间隔：必须 ≥ 轴标题装饰带（18px），否则下一行的标题会贴到上一行的刻度上。 */
export const ROW_GAP = 22;

/** 面板下方一行说明的推荐高度（size 11 → 行高 15，再留 4px 缝）。 */
export const LABEL_BAND = 19;

/**
 * 把行主序矩阵的第 `row` 行取成一条曲线（`frame.matrix()` 的 `rows × cols` 约定）。
 *
 * 单独一个函数是因为它要把"行列 ↔ 扁平下标"这件事只写一次：B 样条的基函数矩阵
 * 与损失函数一样，最经典的错法是行列写反（`values[i*cols+j]` 写成 `values[j*rows+i]`），
 * 而画出来的图**看起来仍然像一堆基函数**。
 */
export function matrixRow(
  values: ArrayLike<number>,
  row: number,
  cols: number,
): { x: number[]; y: number[]; ok: boolean } {
  const x: number[] = [];
  const y: number[] = [];
  const lo = row * cols;
  if (!(cols > 0) || lo < 0 || lo + cols > values.length) return { x, y, ok: false };
  for (let j = 0; j < cols; j += 1) {
    x.push(j);
    y.push(values[lo + j] as number);
  }
  return { x, y, ok: true };
}

/**
 * 取一条一列版式的排布：`main` 在上，其余副轴依次向下，最后留一条说明带。
 *
 * `prefs` 里每项要给**期望高度**；总高不够时按比例压缩（最小 56px，保证还能看出形状），
 * 多出来的高度平分给各行（不搞"主面板吃掉全部余量"——本章四个 viz 里
 * 副轴上的量（跳变、支撑、地板）与主轴同等重要）。
 */
export interface ColumnLayout {
  /** 各行的矩形（与 `prefs` 一一对应）。 */
  rows: Rect[];
  /** 每行下方那一条说明带的矩形（与 `prefs` 一一对应；高度可能为 0）。 */
  bands: Rect[];
  /** 说明带之后的剩余空间（放"图例/结论"块，`null` = 一点都没剩）。 */
  tail: Rect | null;
}

/** 把一个矩形按 `split` 切成左右两块（第 `i` 块，共 `n` 块），留出 `gap`。 */
export function splitRect(rect: Rect, i: number, n: number, gap = 24): Rect {
  const w = Math.max(60, (rect[2] - gap * (n - 1)) / n);
  return [rect[0] + i * (w + gap), rect[1], w, rect[3]];
}

/**
 * 两列网格的**必须间隙**：左格的 y 刻度标签带在 `left−41 … left−6`、轴名在 `left−56 … left−42`，
 * 右格同理。所以两格之间的净空最少要"左格的左装饰 + 右格的左装饰 + 余量"。
 * 这个函数就是把这句几何写成数：`max(56, leftYLabel 宽度) + max(56, maxYLabel 宽度) + 8`。
 * 实测（bspline_local）：左格 `max|B|` 要 35px、右格 `|dC|` 要 61px → 间隙 104px 才干净。
 */
export function twoColumnGap(leftYLabel: string, rightYLabel: string): number {
  const w = (t: string): number => Math.max(56, 7 * t.length + 8);
  return Math.round(w(leftYLabel) + w(rightYLabel) + 8);
}

export function columnLayout(
  left: number,
  right: number,
  top: number,
  bottom: number,
  prefs: readonly number[],
  opts: { gap?: number; band?: number; bands?: readonly number[] } = {},
): ColumnLayout {
  const gapPref = opts.gap ?? ROW_GAP;
  const bandOf = (i: number): number => opts.bands?.[i] ?? opts.band ?? LABEL_BAND;
  const n = prefs.length;
  const width = Math.max(80, right - left);
  const height = Math.max(120, bottom - top);

  // 预算顺序（**先保证画得出来，再谈说明文字**）：
  //   ① 每行至少 `minRow`（下面还要压标题带与刻度带）；② 每行不高于 `prefs[i]`；
  //   ③ 剩下的分给说明带（每行最多 `bandPref`）；④ 还有剩才加大行间距。
  // 早期版本把"说明带"当成必得项、把行高压到 56 以下，结果 `bands` 全被夹成 0、
  // 文本块直接落在轴上（校验器报规则 8）——所以这里改成显式预算而不是"够就减"。
  const minRow = 56;
  const rowsPref = prefs.map((v) => Math.max(minRow, v));
  const bandWant = prefs.map((_, i) => Math.max(0, bandOf(i)));
  const sum = (a: readonly number[]): number => a.reduce((s2, v) => s2 + v, 0);
  // 预算顺序（**先把面板给足，再谈说明文字**）：
  //   ① 每个面板按 prefs 给（不足就一起等比缩，最小 56px）；
  //   ② 剩下的分给说明带（每格最多 bandOf(i)）；
  //   ③ 还有剩就继续加面板高度（上限 240px）。
  // 早期版本把说明带当"必得项"、面板当"余量"，结果主轴只剩 56px 一条缝（实测）。
  const wantTotal = sum(rowsPref) + gapPref * (n - 1) + sum(bandWant);
  const shrink = wantTotal > height ? height / wantTotal : 1;
  const rowsOut = rowsPref.map((v) => Math.max(minRow, v * shrink));
  const wantBands = bandWant.map((v) => v * shrink);
  const fixedRows = sum(rowsOut);
  const used = fixedRows + gapPref * (n - 1);
  const leftover = height - used;
  let pool = Math.max(0, leftover - sum(wantBands));
  const bandsOut = wantBands.map((v) => v + pool / n);
  if (pool > 0) {
    // 说明带已经拿满，剩下的加面板高
    let k2 = 0;
    while (pool > 1 && k2 < n * 60) {
      rowsOut[k2] = Math.min(240, (rowsOut[k2] as number) + 1);
      pool -= 1;
      k2 += 1;
    }
  }
  const gap = gapPref;

  const rows: Rect[] = [];
  const bands: Rect[] = [];
  let y = top;
  for (let i = 0; i < n; i += 1) {
    const h = Math.max(minRow, Math.round(rowsOut[i] as number));
    rows.push([left, Math.round(y), width, h]);
    y += h + gap;
    const bh = Math.round(clamp(bandsOut[i] as number, 0, Math.max(0, bottom - y)));
    bands.push([left, Math.round(y), width, bh]);
    y += bh + gap;
  }
  const tailH = Math.round(Math.max(0, bottom - (y - gap)));
  return { rows, bands, tail: tailH > 0 ? [left, Math.round(y - gap), width, tailH] : null };
}

// --------------------------------------------------------------------------- //
// 文本折行（与 scene.ts::textBoxOf 同一把尺子）
// --------------------------------------------------------------------------- //
/**
 * 按显示宽度单位折行：CJK=2、ASCII=1（`scene.ts::textWidthUnits`）。
 *
 * 为什么模型要自己折：两个渲染层都会在 `maxWidth` 处**再**折一次，而续行的高度
 * 两个渲染层算得不一样（浏览器按比例字体、光栅器按等宽位图字体）。模型先折到
 * 一行 = 一个 `text` 层，续行就不再出现——版式与"校验器估的盒子"于是逐像素一致。
 *
 * 先按空白切词再塞行，`-1.27e-16`、`|dS''|=64` 这类字段不会被从中间劈开；
 * 单个词比整行还长时才硬切。
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

/** 面板宽度 → 一行能放多少个显示宽度单位（size 11 时 1 单位 ≈ 7px，留 12px 边距）。 */
export function unitBudget(panelWidthPx: number): number {
  return Math.max(16, Math.floor((panelWidthPx - 12) / 7));
}

/**
 * `renderScene` 会把 `pitfalls` 面板画在画布底部：`y0 = h − 10 − 行数 × 行高`（size 12）。
 *
 * 模型这一侧必须在它之前收住内容——否则说明文字会压在陷阱面板上（`validateScene`
 * 只查"文本 ↔ 坐标轴装饰带"，查不出"文本 ↔ 陷阱面板"，所以这件事只能由模型自己算准）。
 * 行数按两个渲染层的**保守**口径估：每行 `floor((w−30)/7)` 个显示单位（比例字体比
 * 7px/单位窄，光栅器的等宽字体比它宽，取中间偏保守的值）。
 */
/** 陷阱面板的上沿：**直接问共享层**（校验器用的就是 `pitfallsBand`；自己再折一次行必然差几像素）。 */
export function pitfallsBoxTop(scene: Scene): number {
  const band = pitfallsBand(scene);
  return band === null ? scene.canvas.h - 10 : band[1];
}

/** 内容区底端：陷阱面板之上再让出 `margin` 像素。 */
export function reservePanelBottom(scene: Scene, margin = 6): number {
  return pitfallsBoxTop(scene) - margin;
}

/** 拿一条文本带的**真实**盒子（与校验器同一把尺子）去撞 `[from, to)` 这段纵带。 */
export function textClearOf(scene: Scene, from: number, to: number): boolean {
  if (!(to > from)) return true;
  return scene.texts.every((t) => {
    const box = textBoxOf(t);
    return box.y + box.h <= from || box.y >= to;
  });
}

/**
 * 一条说明带里的**分栏**文本：每栏有自己的 x 与宽度，栏内按 `textInRect` 折行。
 *
 * 为什么需要它：一条整幅宽（比如 860px、120 单位）的说明横跨两三个面板，
 * 右端会伸进右边那条轴的 y 刻度标签带（`[left−41, left−6]`）——校验器按
 * 真实盒子报规则 8，而且图上确实就是压着刻度数字。按面板切栏之后，
 * 每栏的文字只落在自己面板的正下方。
 *
 * `bottom` 是整条带的底边（各栏共用）；`minY` 是各栏共同的顶边上界。
 */
export function textTailBlock(
  band: Rect,
  columns: readonly { x: number; width: number; lines: readonly string[]; color: string }[],
  minY = 0,
): TextSpec[] {
  const out: TextSpec[] = [];
  for (const col of columns) {
    if (col.lines.length === 0) continue;
    const rect: Rect = [col.x, band[1], col.width, band[3]];
    out.push(...textInRect(rect, col.lines, col.color, { anchor: "top", minY }));
  }
  return out;
}

/**
 * 一个矩形里排若干行文本（一行 = 一个 `text` 层）。
 *
 * `anchor` 决定这一块贴哪一边：`"bottom"`（缺省）让整块贴着矩形底边往上排，
 * `"top"` 从矩形顶边往下排。**说明带的语义是"这块地方就是留给它的"**，所以本章
 * 一律用 `"top"`：底锚会让"行数少于预算"的文本浮到预留区的上半部分，撞进上一条轴的
 * 刻度带（真实踩过：两行文本浮进 `main` 的 xticks 带，`validateScene` 当场报规则 8）。
 *
 * 放不下时**截断并写明"另有 N 行"**（叠字等于看不见，规则 1 的精神）；
 * `minY` 是这块文本的上界（不许往上顶进别的面板的装饰带）。
 */
export function textInRect(
  rect: Rect,
  lines: readonly string[],
  color: string,
  opts: { size?: number; minY?: number; maxRows?: number; anchor?: "top" | "bottom" } = {},
): TextSpec[] {
  const size = opts.size ?? 11;
  const rowH = size + 4;
  const budget = unitBudget(rect[2]);
  const [x, y, , h] = rect;
  const top = Math.max(y, opts.minY ?? y);
  // 行数上限**由调用方给**（默认按矩形高度算），而不是"放不下就加一行计数"。
  // 早期版本在放不下时补一行「另有 N 行放不下」，那一行本身就超出矩形 → 压住下一格的
  // 轴标题带，校验器报规则 8（本章 4 个 viz 全部踩过）。现在超出的行**直接不画**，
  // 想多写就自己把带加高（或者写进陷阱面板——它本来就在画布底部）。
  const maxRows = Math.max(1, opts.maxRows ?? Math.floor((y + h - top) / rowH) + 1);
  const pieces = lines.flatMap((body) => wrapUnits(body, budget)).slice(0, maxRows);
  const anchor = opts.anchor ?? "bottom";
  const baseline = anchor === "top" ? top + rowH - 4 : y + h - (pieces.length - 1) * rowH - rowH + 4;
  return pieces.map((body, i) => text(x, baseline + i * rowH, body, { role: "note", size, color, maxWidth: budget * 7 }));
}

/**
 * 在某个轴上画一段**竖直参考线**（临界值、地板、0 基线、探针位置）。
 *
 * 用两点的 `line` 层而不是别的图层：两个渲染层对 `line` 的语义完全一致，
 * 且 `checkOverflow` 会把越界的参考线如实报出来（规则 1）。
 */
export function vLine(
  scene: Scene,
  axis: Axis,
  x: number,
  color: string,
  opts: { dash?: number[]; width?: number; label?: string } = {},
): void {
  if (!Number.isFinite(x) || x < axis.xlim[0] || x > axis.xlim[1]) return;
  scene.layers.push({
    kind: "line",
    axes: axis.id,
    series: [
      {
        x: [x, x],
        y: [axis.ylim[0], axis.ylim[1]],
        color,
        width: opts.width ?? 1,
        dash: opts.dash ?? [4, 3],
        label: opts.label,
      },
    ],
  });
}

/** 在某个轴上画一条**水平参考线**（地板、单位分解…）。 */
export function hLine(
  scene: Scene,
  axis: Axis,
  y: number,
  color: string,
  opts: { dash?: number[]; width?: number; label?: string } = {},
): void {
  if (!Number.isFinite(y) || y < axis.ylim[0] || y > axis.ylim[1]) return;
  scene.layers.push({
    kind: "line",
    axes: axis.id,
    series: [
      {
        x: [axis.xlim[0], axis.xlim[1]],
        y: [y, y],
        color,
        width: opts.width ?? 1,
        dash: opts.dash ?? [4, 3],
        label: opts.label,
      },
    ],
  });
}

/**
 * 一根**梳齿**：从根部到齿尖的直线段（曲率梳的每一根都单独成为一个两点的 series）。
 *
 * 不用 `vectorfield` 的理由：那个图层的 `u/v` 是**数据单位**的向量，梳齿长度
 * 在数据坐标里各向异性（x 与 y 的尺度不同），画出来方向是错的；直接用帧给的
 * `comb_base` / `comb_tip` 两个端点连线，方向与长度都由后端算好。
 */
export function combSeries(
  base: ArrayLike<number>,
  tip: ArrayLike<number>,
  color: string,
  width: number,
): { x: number[]; y: number[]; color: string; width: number }[] {
  const n = Math.min(Math.floor(base.length / 2), Math.floor(tip.length / 2));
  const out: { x: number[]; y: number[]; color: string; width: number }[] = [];
  for (let i = 0; i < n; i += 1) {
    const bx = base[2 * i];
    const by = base[2 * i + 1];
    const tx = tip[2 * i];
    const ty = tip[2 * i + 1];
    if (!Number.isFinite(bx) || !Number.isFinite(by) || !Number.isFinite(tx) || !Number.isFinite(ty)) continue;
    out.push({ x: [bx, tx], y: [by, ty], color, width });
  }
  return out;
}

/** 二维数组的第 `k` 个分量的连续视图（帧里的 `(n,2)` 数组一律行主序）。 */
export function component(values: ArrayLike<number>, k: number, stride = 2): number[] {
  const out: number[] = [];
  for (let i = k; i < values.length; i += stride) out.push(values[i] as number);
  return out;
}

/** 数据点 → 像素（给"手写刻度标签"用；`AxisOptions` 没有 `xtickLabels`）。 */
export function pixelX(axis: Axis, x: number): number | undefined {
  const [px] = dataToPixel(axis, x, axis.ylim[0]);
  return Number.isFinite(px) ? px : undefined;
}

/** 一列数值的 1-2-5 取整范围；没有有限正值时返回 `undefined`。 */
export function logLimitsOf(values: readonly number[], ticks = 7): [number, number] | undefined {
  let lo = Number.POSITIVE_INFINITY;
  let hi = Number.NEGATIVE_INFINITY;
  for (const v of values) {
    if (!Number.isFinite(v) || v <= 0) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return undefined;
  const lim = niceLimits(lo, hi, ticks, { log: true });
  return lim[0] > 0 && lim[1] > lim[0] ? lim : undefined;
}

/** 线性范围（0 打底）；全为非有限值时返回回落范围。 */
export function linearLimitsOf(values: readonly number[], fallback: [number, number], ticks = 5): [number, number] {
  let lo = Number.POSITIVE_INFINITY;
  let hi = Number.NEGATIVE_INFINITY;
  for (const v of values) {
    if (!Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return fallback;
  const lim = niceLimits(lo, hi, ticks);
  return lim[1] > lim[0] ? lim : fallback;
}
