/**
 * Canvas 2D 的轻量数学排版：把 `$...$` / `\(...\)` 里的一小撮 TeX 画成
 * 带上下标、分式、根号、重音与常见算符的排版文本。
 *
 * 为什么不用 MathML：画布上没有 DOM，`renderRichText` 的 MathML 结果无法直接绘制。
 * 这个模块只覆盖图表里真正会用到的记法（不是完整 TeX），并且**绝不抛异常**：
 * 解析失败时按原文字面量画出来，至少让读者看到“这里有一个没排出来的公式”，
 * 而不是让整张图消失。
 *
 * 无第三方依赖；测量、字体与绘制全部走调用方给的 `CanvasRenderingContext2D`。
 */

export interface CanvasRichStyle {
  /** 基础字号（px）。数学段与正文同字号。 */
  size: number;
  /** 正文字体族（不含字号/字重），例如 `-apple-system, sans-serif`。 */
  fontFamily: string;
  color: string;
  align?: "left" | "center" | "right";
  baseline?: "top" | "middle" | "bottom" | "alphabetic";
  weight?: number | string;
}

interface Box {
  w: number;
  /** 基线之上的高度。 */
  h: number;
  /** 基线之下的深度。 */
  d: number;
  draw: (ctx: CanvasRenderingContext2D, x: number, baseline: number) => void;
}

const MATH_FAMILY = '"Latin Modern Math", "STIX Two Math", "Cambria Math", "DejaVu Serif", serif';

export function fontOf(size: number, family: string, weight: number | string = 400, italic = false): string {
  const w = String(weight);
  return `${italic ? "italic " : ""}${w === "400" || w === "normal" ? "" : `${w} `}${size}px ${family}`;
}

function measured(ctx: CanvasRenderingContext2D, text: string, font: string): { w: number; h: number; d: number; size: number } {
  ctx.save();
  ctx.font = font;
  const m = ctx.measureText(text);
  const size = Number(/([\d.]+)px/.exec(font)?.[1] ?? 12);
  const h = Number.isFinite(m.actualBoundingBoxAscent) ? m.actualBoundingBoxAscent : size * 0.75;
  const d = Number.isFinite(m.actualBoundingBoxDescent) ? m.actualBoundingBoxDescent : size * 0.2;
  ctx.restore();
  return { w: m.width, h, d, size };
}

function textBox(
  ctx: CanvasRenderingContext2D,
  text: string,
  font: string,
  color: string,
  opts: { pad?: number; dx?: number } = {},
): Box {
  const m = measured(ctx, text, font);
  const pad = opts.pad ?? 0;
  return {
    w: m.w + pad * 2,
    h: m.h,
    d: m.d,
    draw: (c, x, baseline) => {
      c.save();
      c.font = font;
      if (color !== "math") c.fillStyle = color;
      c.textAlign = "left";
      c.textBaseline = "alphabetic";
      c.fillText(text, x + pad, baseline);
      c.restore();
    },
  };
}

function hbox(boxes: Box[]): Box {
  const children = boxes.filter((b) => b !== null);
  if (children.length === 0) return { w: 0, h: 0, d: 0, draw: () => undefined };
  let x = 0;
  const placed = children.map((b) => {
    const at = x;
    x += b.w;
    return { b, at };
  });
  return {
    w: x,
    h: Math.max(...children.map((b) => b.h)),
    d: Math.max(...children.map((b) => b.d)),
    draw: (c, origin, baseline) => {
      for (const { b, at } of placed) b.draw(c, origin + at, baseline);
    },
  };
}

/** 行内空白（TeX 的 `\,` / `\;` / `\quad` 等）。 */
function spaceBox(width: number): Box {
  return { w: width, h: 0, d: 0, draw: () => undefined };
}

// --------------------------------------------------------------------------- //
// 符号表
// --------------------------------------------------------------------------- //

const GREEK: Record<string, string> = {
  alpha: "α", beta: "β", gamma: "γ", delta: "δ", epsilon: "ϵ", varepsilon: "ε",
  zeta: "ζ", eta: "η", theta: "θ", vartheta: "ϑ", iota: "ι", kappa: "κ",
  lambda: "λ", mu: "μ", nu: "ν", xi: "ξ", omicron: "ο", pi: "π", varpi: "ϖ",
  rho: "ρ", varrho: "ϱ", sigma: "σ", varsigma: "ς", tau: "τ", upsilon: "υ",
  phi: "ϕ", varphi: "φ", chi: "χ", psi: "ψ", omega: "ω",
  Gamma: "Γ", Delta: "Δ", Theta: "Θ", Lambda: "Λ", Xi: "Ξ", Pi: "Π",
  Sigma: "Σ", Upsilon: "Υ", Phi: "Φ", Psi: "Ψ", Omega: "Ω",
};

const SYMBOLS: Record<string, string> = {
  times: "×", cdot: "·", div: "÷", pm: "±", mp: "∓",
  le: "≤", leq: "≤", ge: "≥", geq: "≥", ne: "≠", neq: "≠",
  approx: "≈", sim: "∼", simeq: "≃", equiv: "≡", propto: "∝",
  ll: "≪", gg: "≫", in: "∈", notin: "∉", subset: "⊂", subseteq: "⊆",
  cup: "∪", cap: "∩", setminus: "∖", emptyset: "∅", varnothing: "∅",
  to: "→", rightarrow: "→", leftarrow: "←", leftrightarrow: "↔",
  Rightarrow: "⇒", Leftarrow: "⇐", Leftrightarrow: "⇔", implies: "⟹",
  mapsto: "↦", uparrow: "↑", downarrow: "↓",
  infty: "∞", partial: "∂", nabla: "∇", forall: "∀", exists: "∃",
  neg: "¬", land: "∧", lor: "∨", oplus: "⊕", otimes: "⊗",
  sum: "∑", prod: "∏", int: "∫", iint: "∬", oint: "∮",
  ldots: "…", dots: "…", cdots: "⋯", vdots: "⋮", ddots: "⋱",
  prime: "′", degree: "°", angle: "∠", perp: "⊥", parallel: "∥",
  lvert: "|", rvert: "|", lVert: "‖", rVert: "‖",
  langle: "⟨", rangle: "⟩", lfloor: "⌊", rfloor: "⌋", lceil: "⌈", rceil: "⌉",
  top: "⊤", bot: "⊥", star: "⋆", ast: "∗", circ: "∘", bullet: "•",
  Re: "ℜ", Im: "ℑ", hbar: "ℏ", ell: "ℓ",
  "{": "{", "}": "}", "%": "%", "#": "#", "&": "&", "_": "_",
  "|": "‖", "\\": "\n", " ": " ",
};

const UPRIGHT_FUNCTIONS = new Set([
  "sin", "cos", "tan", "cot", "sec", "csc", "arcsin", "arccos", "arctan",
  "sinh", "cosh", "tanh", "log", "ln", "lg", "exp", "max", "min", "arg",
  "det", "rank", "Tr", "tr", "diag", "span", "dim", "deg", "gcd", "sup", "inf",
  "lim", "limsup", "liminf", "Pr", "mod", "bmod", "sgn", "sign", "cond",
]);

const BLACKBOARD: Record<string, string> = {
  R: "ℝ", N: "ℕ", Z: "ℤ", Q: "ℚ", C: "ℂ", P: "ℙ", E: "𝔼", H: "ℍ",
};

const ACCENTS: Record<string, string> = {
  hat: "ˆ", widehat: "ˆ", bar: "¯", overline: "¯", tilde: "˜", widetilde: "˜",
  vec: "→", dot: "˙", ddot: "¨", check: "ˇ", breve: "˘",
};

// --------------------------------------------------------------------------- //
// 解析
// --------------------------------------------------------------------------- //

interface ParseState {
  src: string;
  i: number;
  ctx: CanvasRenderingContext2D;
  size: number;
  family: string;
  weight: number | string;
  upright: boolean;
}

function skipSpaces(s: ParseState): void {
  while (s.i < s.src.length && /\s/.test(s.src[s.i] as string)) s.i += 1;
}

function readGroupRaw(s: ParseState): string {
  skipSpaces(s);
  if (s.src[s.i] !== "{") {
    const ch = s.src[s.i] ?? "";
    s.i += ch.length;
    return ch;
  }
  s.i += 1;
  let depth = 1;
  let out = "";
  while (s.i < s.src.length && depth > 0) {
    const ch = s.src[s.i] as string;
    if (ch === "{") depth += 1;
    else if (ch === "}") depth -= 1;
    if (depth > 0) out += ch;
    s.i += 1;
  }
  return out;
}

function readCommand(s: ParseState): string {
  s.i += 1; // backslash
  if (s.i >= s.src.length) return "";
  if (/[A-Za-z]/.test(s.src[s.i] as string)) {
    let name = "";
    while (s.i < s.src.length && /[A-Za-z]/.test(s.src[s.i] as string)) {
      name += s.src[s.i];
      s.i += 1;
    }
    return name;
  }
  const ch = s.src[s.i] as string;
  s.i += 1;
  return ch;
}

/**
 * 解析一个“原子 + 上下标”。返回的 Box 使用**基线坐标**。
 * 任何未覆盖的命令都不抛错：画成等宽的命令名，方便发现并补支持。
 */
function parseAtom(s: ParseState): Box {
  skipSpaces(s);
  if (s.i >= s.src.length) return spaceBox(0);
  const color = "math";
  void color;
  const mathFont = (italic: boolean, weight = s.weight): string =>
    fontOf(s.size, MATH_FAMILY, weight, italic);

  let base: Box;
  const ch = s.src[s.i] as string;
  if (ch === "{") {
    s.i += 1;
    base = parseSequence(s, "}");
    if (s.src[s.i] === "}") s.i += 1;
  } else if (ch === "\\") {
    const saved = s.i;
    const name = readCommand(s);
    if (name === "frac" || name === "dfrac" || name === "tfrac") {
      const numSrc = readGroupRaw(s);
      const denSrc = readGroupRaw(s);
      base = fractionBox(
        s,
        parseFragment(s, numSrc, { size: s.size * 0.92, upright: false }),
        parseFragment(s, denSrc, { size: s.size * 0.92, upright: false }),
      );
    } else if (name === "sqrt") {
      const content = readGroupRaw(s);
      base = sqrtBox(s, parseFragment(s, content, { size: s.size, upright: false }));
    } else if (name in ACCENTS) {
      const content = readGroupRaw(s);
      base = accentBox(
        s,
        parseFragment(s, content, { size: s.size, upright: false }),
        ACCENTS[name] as string,
      );
    } else if (name === "text" || name === "textrm" || name === "mbox") {
      const raw = readGroupRaw(s);
      base = textBox(s.ctx, raw, fontOf(s.size, s.family, 400, false), "math");
    } else if (name === "mathrm" || name === "operatorname") {
      const raw = readGroupRaw(s);
      base = parseFragment(s, raw, { size: s.size, upright: true });
    } else if (name === "mathbf") {
      const raw = readGroupRaw(s);
      base = parseFragment(s, raw, { size: s.size, weight: 700, upright: true });
    } else if (name === "mathit") {
      const raw = readGroupRaw(s);
      base = parseFragment(s, raw, { size: s.size, upright: false });
    } else if (name === "mathbb") {
      const raw = readGroupRaw(s);
      const mapped = [...raw].map((c) => BLACKBOARD[c] ?? c).join("");
      base = textBox(s.ctx, mapped, mathFont(false), "math");
    } else if (name === "mathcal") {
      const raw = readGroupRaw(s);
      base = textBox(s.ctx, raw, `${fontOf(s.size, MATH_FAMILY, 400, false)}`, "math");
    } else if (name === "left" || name === "right" || name === "middle") {
      skipSpaces(s);
      const delim = s.src[s.i] === "\\" ? readCommand(s) : (s.src[s.i++] ?? "");
      const shown = delim === "." ? "" : (SYMBOLS[delim] ?? delim);
      base = shown === "" ? spaceBox(0) : textBox(s.ctx, shown, mathFont(false), "math");
    } else if (name === "begin" || name === "end") {
      readGroupRaw(s);
      base = spaceBox(0);
    } else if (name === "quad") {
      base = spaceBox(s.size * 1.0);
    } else if (name === "qquad") {
      base = spaceBox(s.size * 2.0);
    } else if (name === "," || name === ";" || name === ":" || name === "!" || name === " ") {
      base = spaceBox(s.size * (name === "," ? 0.17 : name === ";" ? 0.28 : name === ":" ? 0.22 : name === "!" ? -0.17 : 0.25));
    } else if (name === "\\") {
      base = spaceBox(s.size * 0.4);
    } else if (name in GREEK) {
      base = textBox(s.ctx, GREEK[name] as string, mathFont(!s.upright), "math");
    } else if (name in SYMBOLS) {
      const glyph = SYMBOLS[name] as string;
      const isOp = /^[=+\-<>≤≥≠≈∼≃≡×·÷±∓→←↔⇒⇐⇔⟹∈∉⊂⊆∪∩≤≥≪≫]$/.test(glyph);
      base = textBox(s.ctx, glyph, mathFont(false), "math", { pad: isOp ? s.size * 0.12 : 0 });
    } else if (UPRIGHT_FUNCTIONS.has(name)) {
      base = textBox(s.ctx, name, mathFont(false), "math");
    } else if (name === "textwidth" || name === "displaystyle") {
      base = spaceBox(0);
    } else {
      // 未覆盖的命令：画出 `\name`，不静默丢掉。
      s.i = saved;
      const raw = s.src.slice(saved, Math.min(s.src.length, saved + name.length + 1));
      s.i = Math.min(s.src.length, saved + name.length + 1);
      base = textBox(s.ctx, raw, mathFont(false), "math");
    }
  } else {
    s.i += ch.length;
    if (ch === " ") {
      base = spaceBox(s.size * 0.25);
    } else if (/[A-Za-z]/.test(ch)) {
      const combined = /[A-Za-z]{2,}/.exec(s.src.slice(s.i - 1));
      const word = combined?.[0] ?? ch;
      if (word.length > 1 && UPRIGHT_FUNCTIONS.has(word)) {
        s.i = s.i - 1 + word.length;
        base = textBox(s.ctx, word, mathFont(false), "math");
      } else {
        base = textBox(s.ctx, ch, mathFont(!s.upright), "math");
      }
    } else if (/[0-9.]/.test(ch)) {
      let num = ch;
      while (s.i < s.src.length && /[0-9.]/.test(s.src[s.i] as string)) {
        num += s.src[s.i];
        s.i += 1;
      }
      base = textBox(s.ctx, num, mathFont(false), "math");
    } else if (ch === "=" || ch === "+" || ch === "<" || ch === ">" || ch === "-") {
      const glyph = ch === "-" ? "−" : ch;
      base = textBox(s.ctx, glyph, mathFont(false), "math", { pad: s.size * 0.1 });
    } else if (ch === "'" ) {
      base = textBox(s.ctx, "′", mathFont(false), "math");
    } else if (ch === "~") {
      base = spaceBox(s.size * 0.3);
    } else {
      base = textBox(s.ctx, ch, mathFont(false), "math");
    }
  }

  // 上下标（可以同时出现，顺序任意）。
  let sub: Box | undefined;
  let sup: Box | undefined;
  for (let guard = 0; guard < 4; guard += 1) {
    skipSpaces(s);
    const mark = s.src[s.i];
    if (mark !== "_" && mark !== "^") break;
    s.i += 1;
    const arg = readGroupRaw(s);
    const script = parseFragment(s, arg, {
      size: s.size * 0.68,
      upright: mark === "_" ? s.upright : false,
    });
    if (mark === "_") sub = script;
    else sup = script;
  }
  if (sub !== undefined || sup !== undefined) return scriptBox(s, base, sub, sup);
  return base;
}

function parseSequence(s: ParseState, stop?: string): Box {
  const parts: Box[] = [];
  while (s.i < s.src.length) {
    skipSpaces(s);
    if (stop !== undefined && s.src[s.i] === stop) break;
    if (s.i >= s.src.length) break;
    const before = s.i;
    parts.push(parseAtom(s));
    if (s.i === before) s.i += 1; // 防御：任何情况下都前进
  }
  return hbox(parts);
}

interface FragmentStyle {
  size?: number;
  weight?: number | string;
  upright?: boolean;
}

/** 用新的字号/字形解析一段独立的 TeX 片段（分式、根号、下标等）。 */
function parseFragment(s: ParseState, tex: string, style: FragmentStyle = {}): Box {
  const sub: ParseState = {
    ...s,
    src: tex,
    i: 0,
    size: style.size ?? s.size,
    weight: style.weight ?? s.weight,
    upright: style.upright ?? s.upright,
  };
  return parseSequence(sub);
}

function scriptBox(s: ParseState, base: Box, sub: Box | undefined, sup: Box | undefined): Box {
  const gap = s.size * 0.06;
  const up = s.size * 0.42;
  const down = s.size * 0.22;
  const w = base.w + gap + Math.max(sub?.w ?? 0, sup?.w ?? 0);
  const h = Math.max(base.h, up + (sup?.h ?? 0) + (sup?.d ?? 0));
  const d = Math.max(base.d, down + (sub?.d ?? 0) + (sub?.h ?? 0) * 0.25);
  return {
    w,
    h,
    d,
    draw: (ctx, x, baseline) => {
      base.draw(ctx, x, baseline);
      if (sup !== undefined) sup.draw(ctx, x + base.w + gap, baseline - up);
      if (sub !== undefined) sub.draw(ctx, x + base.w + gap, baseline + down);
    },
  };
}

function fractionBox(s: ParseState, num: Box, den: Box): Box {
  const pad = s.size * 0.16;
  const gap = s.size * 0.18;
  const w = Math.max(num.w, den.w) + pad * 2;
  const ruleY = s.size * 0.28;
  const h = num.h + num.d + gap + ruleY;
  const d = den.h + den.d + gap - ruleY * 0.25;
  return {
    w,
    h,
    d,
    draw: (ctx, x, baseline) => {
      const cx = x + w / 2;
      num.draw(ctx, cx - num.w / 2, baseline - ruleY - gap - num.d);
      den.draw(ctx, cx - den.w / 2, baseline - ruleY + gap + den.h);
      ctx.save();
      ctx.strokeStyle = "math";
      ctx.lineWidth = Math.max(0.8, s.size * 0.05);
      ctx.beginPath();
      ctx.moveTo(x + pad * 0.3, baseline - ruleY);
      ctx.lineTo(x + w - pad * 0.3, baseline - ruleY);
      ctx.stroke();
      ctx.restore();
    },
  };
}

function sqrtBox(s: ParseState, content: Box): Box {
  const pad = s.size * 0.1;
  const glyphSize = Math.max(s.size * 0.9, content.h + content.d);
  const font = fontOf(glyphSize, MATH_FAMILY, 400, false);
  const radical = textBox(s.ctx, "√", font, "math");
  const barH = Math.max(1, s.size * 0.055);
  const w = radical.w + content.w + pad * 2;
  return {
    w,
    h: Math.max(radical.h, content.h + barH * 2),
    d: Math.max(radical.d, content.d),
    draw: (ctx, x, baseline) => {
      radical.draw(ctx, x, baseline);
      const contentX = x + radical.w + pad;
      content.draw(ctx, contentX, baseline);
      ctx.save();
      ctx.strokeStyle = "math";
      ctx.lineWidth = barH;
      ctx.beginPath();
      ctx.moveTo(contentX - pad * 0.3, baseline - content.h - barH);
      ctx.lineTo(contentX + content.w + pad * 0.5, baseline - content.h - barH);
      ctx.stroke();
      ctx.restore();
    },
  };
}

function accentBox(s: ParseState, base: Box, accent: string): Box {
  const size = s.size * 0.9;
  const accentBox = textBox(s.ctx, accent, fontOf(size, MATH_FAMILY, 400, false), "math");
  const gap = s.size * 0.05;
  return {
    w: Math.max(base.w, accentBox.w),
    h: base.h + gap + accentBox.h,
    d: base.d,
    draw: (ctx, x, baseline) => {
      base.draw(ctx, x, baseline);
      accentBox.draw(ctx, x + (base.w - accentBox.w) / 2, baseline - base.h - gap);
    },
  };
}

/** 一行富文本：`$...$` 与普通文字混排。 */
function layoutRich(
  ctx: CanvasRenderingContext2D,
  text: string,
  style: CanvasRichStyle,
): Box {
  const source = text.replace(/\r\n?/g, "\n");
  const boxes: Box[] = [];
  const pushPlain = (raw: string): void => {
    if (raw === "") return;
    const font = fontOf(style.size, style.fontFamily, style.weight ?? 400, false);
    for (const piece of raw.split("\n")) {
      if (piece !== "") boxes.push(textBox(ctx, piece, font, style.color));
      // 换行在本模块里不处理；调用方按行拆分。保留换行符为空隙。
    }
  };
  let i = 0;
  while (i < source.length) {
    // 先找下一个数学定界符；两段数学之间的普通文字一次画成一段。
    let next = -1;
    for (let j = i; j < source.length; j += 1) {
      if (source[j] === "$" || source.startsWith("\\(", j)) {
        next = j;
        break;
      }
    }
    if (next < 0) {
      pushPlain(source.slice(i));
      break;
    }
    if (next > i) {
      pushPlain(source.slice(i, next));
      i = next;
      continue;
    }
    let start = -1;
    let end = -1;
    let skip = 0;
    if (source[i] === "$") {
      if (source[i + 1] === "$") {
        start = i + 2;
        end = source.indexOf("$$", start);
        skip = 2;
      } else {
        start = i + 1;
        end = source.indexOf("$", start);
        skip = 1;
      }
    } else if (source.startsWith("\\(", i)) {
      start = i + 2;
      end = source.indexOf("\\)", start);
      skip = 2;
    }
    if (start < 0 || end < 0) {
      pushPlain(source[i] as string);
      i += 1;
      continue;
    }
    const tex = source.slice(start, end);
    const state: ParseState = {
      src: tex,
      i: 0,
      ctx,
      size: style.size,
      family: style.fontFamily,
      weight: style.weight ?? 400,
      upright: false,
    };
    const math = parseSequence(state);
    boxes.push({
      w: math.w,
      h: math.h,
      d: math.d,
      draw: (c, x, baseline) => {
        c.save();
        c.fillStyle = style.color;
        c.strokeStyle = style.color;
        math.draw(c, x, baseline);
        c.restore();
      },
    });
    i = end + skip;
  }
  return hbox(boxes);
}

/** 量一段可能含 `$...$` 的文本宽度（与绘制共用同一套排版）。 */
export function measureRichText(ctx: CanvasRenderingContext2D, text: string, style: CanvasRichStyle): { w: number; h: number; d: number } {
  const box = layoutRich(ctx, text, style);
  return { w: box.w, h: box.h, d: box.d };
}

/**
 * 画一段可能含 `$...$` 的文本；位置语义与 Canvas `fillText` 的
 * `textAlign`/`textBaseline` 对齐。普通（不含 `$`）文本仍然按原字体绘制。
 */
export function drawRichText(
  ctx: CanvasRenderingContext2D,
  text: string,
  x: number,
  y: number,
  style: CanvasRichStyle,
): { w: number; h: number; d: number } {
  const box = layoutRich(ctx, text, style);
  const align = style.align ?? "left";
  const ox = align === "center" ? x - box.w / 2 : align === "right" ? x - box.w : x;
  const baseline = style.baseline ?? "alphabetic";
  const by =
    baseline === "top"
      ? y + box.h
      : baseline === "middle"
        ? y + (box.h - box.d) / 2
        : baseline === "bottom"
          ? y - box.d
          : y;
  ctx.save();
  ctx.fillStyle = style.color;
  box.draw(ctx, ox, by);
  ctx.restore();
  return { w: box.w, h: box.h, d: box.d };
}

/** 富文本按字符折行（只在没有更好断点时使用；图表文本通常已经预先折行）。 */
export function wrapRichText(
  ctx: CanvasRenderingContext2D,
  text: string,
  maxWidth: number,
  style: CanvasRichStyle,
): string[] {
  if (!text.includes("$") && !text.includes("\\(")) {
    const out: string[] = [];
    let line = "";
    for (const ch of text) {
      const probe = line + ch;
      if (measureRichText(ctx, probe, style).w > maxWidth && line !== "") {
        out.push(line);
        line = ch;
      } else {
        line = probe;
      }
    }
    if (line !== "") out.push(line);
    return out;
  }
  // 数学段不拆开：先按 `$...$` 切块，再在块之间折行；块本身超宽时硬切。
  const chunks = text.split(/(\$\$?[^$]*\$\$?|\\\([^]*?\\\))/g).filter((s) => s !== "");
  const out: string[] = [];
  let line = "";
  for (const chunk of chunks) {
    const probe = line + chunk;
    if (line !== "" && measureRichText(ctx, probe, style).w > maxWidth) {
      out.push(line);
      line = chunk;
    } else {
      line = probe;
    }
    while (measureRichText(ctx, line, style).w > maxWidth && line.length > 1) {
      let cut = line.length - 1;
      while (cut > 1 && measureRichText(ctx, line.slice(0, cut), style).w > maxWidth) cut -= 1;
      out.push(line.slice(0, cut));
      line = line.slice(cut);
    }
  }
  if (line !== "") out.push(line);
  return out;
}
