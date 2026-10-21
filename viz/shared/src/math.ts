/**
 * 极简 TeX → MathML 渲染器 + manifest 富文本助手。
 *
 * 为什么不用 KaTeX/MathJax：站点运行期要求零第三方依赖；manifest 里的公式也很窄。
 * 这里刻意做成"词法 → 递归下降 → 中间树 → 序列化"的小编译器：
 * - 词法阶段保留每个 token 的原文，`\text{}` 才能原样保留中文与空格；
 * - 语法阶段遇到坏输入（缺 `}`、悬空 `^`、孤立 `\right`）只做可见降级，绝不抛异常；
 * - 序列化阶段逐个转义文本，数学源码只进入 `aria-label`，因此不会产生 HTML 注入。
 *
 * 输出是确定性的：同样输入两次得到完全相同的字符串，方便测试与快照。
 * 代码运行在 node 与浏览器中，不访问 DOM。
 */

export interface TexToMathMLOptions {
  display?: boolean;
}

const MATHML_NS = "http://www.w3.org/1998/Math/MathML";

/* -------------------------------------------------------------------------
 * XML 基础工具
 * ---------------------------------------------------------------------- */

function escapeXml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

/**
 * 属性里的反斜杠改写成数字实体：解析后 aria-label 仍是原始 TeX，
 * 但序列化文本里不会出现 "\frac" 这类命令字面量（内容区域必须是纯 MathML）。
 */
function escapeXmlAttribute(value: string): string {
  return escapeXml(value).replace(/\\/g, "&#92;");
}

/** 只查自有键，避免 `\constructor` 之类的命令撞上 Object.prototype。 */
function lookup<T>(table: Readonly<Record<string, T>>, key: string): T | undefined {
  return Object.prototype.hasOwnProperty.call(table, key) ? table[key] : undefined;
}

/* -------------------------------------------------------------------------
 * 中间树：只包含我们要输出的 MathML 子集
 * ---------------------------------------------------------------------- */

type Variant = "normal" | "bold" | "double-struck" | "script";

type MNode =
  | { kind: "mi"; text: string; variant?: Variant }
  | { kind: "mn"; text: string; variant?: Variant }
  | { kind: "mo"; text: string; variant?: Variant; fence?: boolean; stretchy?: boolean; largeop?: boolean; movablelimits?: boolean }
  | { kind: "mtext"; text: string; variant?: Variant }
  | { kind: "mrow"; children: MNode[] }
  | { kind: "mfrac"; num: MNode; den: MNode }
  | { kind: "msqrt"; body: MNode }
  | { kind: "mroot"; body: MNode; index: MNode }
  | { kind: "msup"; base: MNode; sup: MNode }
  | { kind: "msub"; base: MNode; sub: MNode }
  | { kind: "msubsup"; base: MNode; sub: MNode; sup: MNode }
  | { kind: "munder"; base: MNode; under: MNode }
  | { kind: "mover"; base: MNode; over: MNode; accent?: boolean }
  | { kind: "munderover"; base: MNode; under: MNode; over: MNode }
  | { kind: "mspace"; width?: string; linebreak?: "newline" };

interface MoOptions {
  variant?: Variant;
  fence?: boolean;
  stretchy?: boolean;
  largeop?: boolean;
  movablelimits?: boolean;
}

function mi(text: string, variant?: Variant): MNode {
  return { kind: "mi", text, variant };
}

function mn(text: string, variant?: Variant): MNode {
  return { kind: "mn", text, variant };
}

function mo(text: string, options: MoOptions = {}): MNode {
  return { kind: "mo", text, ...options };
}

function mtext(text: string): MNode {
  return { kind: "mtext", text };
}

function mrow(children: MNode[]): MNode {
  return { kind: "mrow", children };
}

function emptyRow(): MNode {
  return mrow([]);
}

function mspace(width: string): MNode {
  return { kind: "mspace", width };
}

function lineBreak(): MNode {
  return { kind: "mspace", linebreak: "newline" };
}

function variantAttribute(variant: Variant | undefined): string {
  return variant === undefined ? "" : ` mathvariant="${variant}"`;
}

function serialize(node: MNode): string {
  switch (node.kind) {
    case "mi":
    case "mn":
      return `<${node.kind}${variantAttribute(node.variant)}>${escapeXml(node.text)}</${node.kind}>`;
    case "mo": {
      let attrs = variantAttribute(node.variant);
      if (node.fence) attrs += ' fence="true"';
      // MathML Core 默认让括号/竖线随整行高度拉伸；普通 () 在 TeX 里并不伸缩，
      // 不显式关掉的话，同一行里有分式时 f(x) 的括号会涨到分式那么高。
      if (node.largeop) attrs += ' largeop="true"';
      else attrs += node.stretchy ? ' stretchy="true"' : ' stretchy="false"';
      if (node.movablelimits) attrs += ' movablelimits="true"';
      return `<mo${attrs}>${escapeXml(node.text)}</mo>`;
    }
    case "mtext":
      return `<mtext${variantAttribute(node.variant)}>${escapeXml(node.text)}</mtext>`;
    case "mrow":
      return `<mrow>${node.children.map(serialize).join("")}</mrow>`;
    case "mfrac":
      return `<mfrac>${serialize(node.num)}${serialize(node.den)}</mfrac>`;
    case "msqrt":
      return `<msqrt>${serialize(node.body)}</msqrt>`;
    case "mroot":
      return `<mroot>${serialize(node.body)}${serialize(node.index)}</mroot>`;
    case "msup":
      return `<msup>${serialize(node.base)}${serialize(node.sup)}</msup>`;
    case "msub":
      return `<msub>${serialize(node.base)}${serialize(node.sub)}</msub>`;
    case "msubsup":
      return `<msubsup>${serialize(node.base)}${serialize(node.sub)}${serialize(node.sup)}</msubsup>`;
    case "munder":
      return `<munder>${serialize(node.base)}${serialize(node.under)}</munder>`;
    case "mover": {
      const accent = node.accent ? ' accent="true"' : "";
      return `<mover${accent}>${serialize(node.base)}${serialize(node.over)}</mover>`;
    }
    case "munderover":
      return `<munderover>${serialize(node.base)}${serialize(node.under)}${serialize(node.over)}</munderover>`;
    case "mspace":
      if (node.linebreak === "newline") return '<mspace linebreak="newline"/>';
      return `<mspace width="${node.width ?? "0.25em"}"/>`;
  }
  return "";
}

/**
 * 把 \mathrm/\mathbf/\mathbb/\mathcal 作用到整棵子树上。
 * 在 token 节点上落 mathvariant（而不是套 mstyle），测试与读屏都能直接看到语义。
 */
function applyVariant(node: MNode, variant: Variant): MNode {
  switch (node.kind) {
    case "mi":
    case "mn":
    case "mo":
    case "mtext":
      return { ...node, variant };
    case "mrow":
      return { ...node, children: node.children.map((child) => applyVariant(child, variant)) };
    case "mfrac":
      return { ...node, num: applyVariant(node.num, variant), den: applyVariant(node.den, variant) };
    case "msqrt":
      return { ...node, body: applyVariant(node.body, variant) };
    case "mroot":
      return { ...node, body: applyVariant(node.body, variant), index: applyVariant(node.index, variant) };
    case "msup":
      return { ...node, base: applyVariant(node.base, variant), sup: applyVariant(node.sup, variant) };
    case "msub":
      return { ...node, base: applyVariant(node.base, variant), sub: applyVariant(node.sub, variant) };
    case "msubsup":
      return {
        ...node,
        base: applyVariant(node.base, variant),
        sub: applyVariant(node.sub, variant),
        sup: applyVariant(node.sup, variant),
      };
    case "munder":
      return { ...node, base: applyVariant(node.base, variant), under: applyVariant(node.under, variant) };
    case "mover":
      return { ...node, base: applyVariant(node.base, variant), over: applyVariant(node.over, variant) };
    case "munderover":
      return {
        ...node,
        base: applyVariant(node.base, variant),
        under: applyVariant(node.under, variant),
        over: applyVariant(node.over, variant),
      };
    case "mspace":
      return node;
  }
  return node;
}

/* -------------------------------------------------------------------------
 * 符号表
 * ---------------------------------------------------------------------- */

const GREEK: Readonly<Record<string, string>> = {
  alpha: "α", beta: "β", gamma: "γ", delta: "δ", epsilon: "ϵ", varepsilon: "ε",
  zeta: "ζ", eta: "η", theta: "θ", vartheta: "ϑ", iota: "ι", kappa: "κ", lambda: "λ",
  mu: "μ", nu: "ν", xi: "ξ", omicron: "ο", pi: "π", varpi: "ϖ", rho: "ρ", varrho: "ϱ",
  sigma: "σ", varsigma: "ς", tau: "τ", upsilon: "υ", phi: "ϕ", varphi: "φ", chi: "χ",
  psi: "ψ", omega: "ω",
  Alpha: "Α", Beta: "Β", Gamma: "Γ", Delta: "Δ", Epsilon: "Ε", Zeta: "Ζ", Eta: "Η",
  Theta: "Θ", Iota: "Ι", Kappa: "Κ", Lambda: "Λ", Mu: "Μ", Nu: "Ν", Xi: "Ξ",
  Omicron: "Ο", Pi: "Π", Rho: "Ρ", Sigma: "Σ", Tau: "Τ", Upsilon: "Υ", Phi: "Φ",
  Chi: "Χ", Psi: "Ψ", Omega: "Ω",
};

const SYMBOLS: Readonly<Record<string, string>> = {
  le: "≤", leq: "≤", ge: "≥", geq: "≥", ne: "≠", neq: "≠",
  approx: "≈", sim: "∼", simeq: "≃", equiv: "≡", cong: "≅", propto: "∝",
  ll: "≪", gg: "≫", in: "∈", notin: "∉", ni: "∋", subset: "⊂", subseteq: "⊆",
  supset: "⊃", supseteq: "⊇", cup: "∪", cap: "∩", setminus: "∖",
  times: "×", cdot: "⋅", pm: "±", mp: "∓", ast: "∗", star: "⋆", circ: "∘",
  bullet: "•", oplus: "⊕", otimes: "⊗", perp: "⊥", parallel: "∥",
  to: "→", rightarrow: "→", leftarrow: "←", Rightarrow: "⇒", Leftarrow: "⇐",
  leftrightarrow: "↔", Leftrightarrow: "⇔", mapsto: "↦", uparrow: "↑", downarrow: "↓",
  infinity: "∞", infty: "∞", partial: "∂", nabla: "∇", forall: "∀", exists: "∃",
  neg: "¬", land: "∧", lor: "∨", emptyset: "∅", varnothing: "∅",
  ldots: "…", cdots: "⋯", dots: "…", vdots: "⋮", ddots: "⋱",
  "|": "‖", "{": "{", "}": "}", langle: "⟨", rangle: "⟩",
  lfloor: "⌊", rfloor: "⌋", lceil: "⌈", rceil: "⌉",
  lvert: "|", rvert: "|", lVert: "‖", rVert: "‖", Vert: "‖",
  ell: "ℓ", prime: "′", mid: "|", "%": "%",
};

const SPACES: Readonly<Record<string, string>> = {
  ",": "0.1667em", ":": "0.2222em", ";": "0.2778em", "!": "-0.1667em",
  " ": "0.25em", quad: "1em", qquad: "2em",
  enspace: "0.5em", thinspace: "0.1667em", negthinspace: "-0.1667em",
};

type LimitMode = "underover" | "subsup";

interface BigOpDef {
  ch: string;
  limits: LimitMode;
  movable?: boolean;
}

const BIG_OPS: Readonly<Record<string, BigOpDef>> = {
  sum: { ch: "∑", limits: "underover", movable: true },
  prod: { ch: "∏", limits: "underover", movable: true },
  coprod: { ch: "∐", limits: "underover", movable: true },
  int: { ch: "∫", limits: "subsup" },
  iint: { ch: "∬", limits: "subsup" },
  iiint: { ch: "∭", limits: "subsup" },
  oint: { ch: "∮", limits: "subsup" },
  bigcup: { ch: "⋃", limits: "underover", movable: true },
  bigcap: { ch: "⋂", limits: "underover", movable: true },
  bigoplus: { ch: "⨁", limits: "underover", movable: true },
  bigotimes: { ch: "⨂", limits: "underover", movable: true },
  bigvee: { ch: "⋁", limits: "underover", movable: true },
  bigwedge: { ch: "⋀", limits: "underover", movable: true },
};

interface NamedFunctionDef {
  limits?: boolean;
}

const NAMED_FUNCTIONS: Readonly<Record<string, NamedFunctionDef>> = {
  sin: {}, cos: {}, tan: {}, cot: {}, sec: {}, csc: {},
  arcsin: {}, arccos: {}, arctan: {}, sinh: {}, cosh: {}, tanh: {},
  log: {}, ln: {}, exp: {}, arg: {}, deg: {}, dim: {}, hom: {}, ker: {},
  lim: { limits: true }, max: { limits: true }, min: { limits: true },
  sup: { limits: true }, inf: { limits: true }, det: { limits: true },
  gcd: { limits: true }, Pr: { limits: true },
};

const ACCENTS: Readonly<Record<string, string>> = {
  hat: "^", widehat: "^", bar: "¯", overline: "¯", tilde: "~",
  widetilde: "~", vec: "→", dot: "˙", ddot: "¨", acute: "´",
  grave: "`", check: "ˇ", breve: "˘",
};

const DELIMITERS: Readonly<Record<string, string>> = {
  "{": "{", "}": "}", "|": "‖", "\\": "‖", ".": "",
  Vert: "‖", vert: "|", lbrace: "{", rbrace: "}", langle: "⟨", rangle: "⟩",
  lfloor: "⌊", rfloor: "⌋", lceil: "⌈", rceil: "⌉",
  lvert: "|", rvert: "|", lVert: "‖", rVert: "‖",
  "(": "(", ")": ")", "[": "[", "]": "]", "/": "/",
};

const MO_CHARS = "+-=<>()[]{}|,.;:!?/";

/* -------------------------------------------------------------------------
 * 词法：命令名按字母连续段切分，控制符号各自成 token
 * ---------------------------------------------------------------------- */

type Token =
  | { kind: "cmd"; name: string; raw: string }
  | { kind: "char"; value: string; raw: string }
  | { kind: "lbrace"; raw: string }
  | { kind: "rbrace"; raw: string }
  | { kind: "sup"; raw: string }
  | { kind: "sub"; raw: string }
  | { kind: "space"; raw: string }
  | { kind: "eof"; raw: string };

function tokenize(source: string): Token[] {
  const tokens: Token[] = [];
  let i = 0;
  while (i < source.length) {
    const ch = source[i];
    if (ch === "\\") {
      const start = i;
      i += 1;
      if (i >= source.length) {
        tokens.push({ kind: "cmd", name: "", raw: "\\" });
        break;
      }
      if (/[A-Za-z]/.test(source[i])) {
        let end = i;
        while (end < source.length && /[A-Za-z]/.test(source[end])) end += 1;
        tokens.push({ kind: "cmd", name: source.slice(i, end), raw: source.slice(start, end) });
        i = end;
      } else {
        tokens.push({ kind: "cmd", name: source[i], raw: source.slice(start, i + 1) });
        i += 1;
      }
      continue;
    }
    if (ch === "%") {
      // TeX 注释：吃掉整行（\text 里的 % 也需要写 \%）
      while (i < source.length && source[i] !== "\n") i += 1;
      continue;
    }
    if (ch === "{") {
      tokens.push({ kind: "lbrace", raw: "{" });
      i += 1;
      continue;
    }
    if (ch === "}") {
      tokens.push({ kind: "rbrace", raw: "}" });
      i += 1;
      continue;
    }
    if (ch === "^") {
      tokens.push({ kind: "sup", raw: "^" });
      i += 1;
      continue;
    }
    if (ch === "_") {
      tokens.push({ kind: "sub", raw: "_" });
      i += 1;
      continue;
    }
    if (/\s/.test(ch)) {
      let end = i;
      while (end < source.length && /\s/.test(source[end])) end += 1;
      tokens.push({ kind: "space", raw: source.slice(i, end) });
      i = end;
      continue;
    }
    tokens.push({ kind: "char", value: ch, raw: ch });
    i += 1;
  }
  tokens.push({ kind: "eof", raw: "" });
  return tokens;
}

/* -------------------------------------------------------------------------
 * 语法：递归下降。每个 atom 带上 limits 元信息，决定竖排还是横排上下标。
 * ---------------------------------------------------------------------- */

interface Atom {
  node: MNode;
  limits?: LimitMode;
}

interface Delimiter {
  raw: string;
  node?: MNode;
}

interface StopContext {
  stopAtBrace?: boolean;
  stopAtRight?: boolean;
  stopAtBracket?: boolean;
}

interface SequenceResult {
  nodes: MNode[];
  right?: Delimiter;
}

function isDigit(value: string): boolean {
  return value >= "0" && value <= "9";
}

function isCjk(value: string): boolean {
  return /[\u3400-\u9FFF\uF900-\uFAFF\u3040-\u30FF\uAC00-\uD7AF]/u.test(value);
}

class TexParser {
  private position = 0;

  constructor(private readonly tokens: Token[]) {}

  parse(): MNode {
    const { nodes } = this.parseSequence({});
    return mrow(nodes);
  }

  private peek(): Token {
    return this.tokens[this.position] ?? { kind: "eof", raw: "" };
  }

  private next(): Token {
    const token = this.peek();
    if (token.kind !== "eof") this.position += 1;
    return token;
  }

  private skipSpaces(): void {
    while (this.peek().kind === "space") this.position += 1;
  }

  private parseSequence(stop: StopContext): SequenceResult {
    const nodes: MNode[] = [];
    while (true) {
      const token = this.peek();
      if (token.kind === "eof") return { nodes };
      if (stop.stopAtBrace && token.kind === "rbrace") {
        this.position += 1;
        return { nodes };
      }
      if (stop.stopAtBracket && token.kind === "char" && token.value === "]") {
        this.position += 1;
        return { nodes };
      }
      if (stop.stopAtRight && token.kind === "cmd" && token.name === "right") {
        this.position += 1;
        return { nodes, right: this.readDelimiter() };
      }
      if (token.kind === "space") {
        this.position += 1;
        continue;
      }
      if (token.kind === "rbrace") {
        // 顶层多余的 }：转成可见文本，保证不中断后续解析
        this.position += 1;
        nodes.push(mtext("}"));
        continue;
      }
      nodes.push(this.parseAtom().node);
    }
  }

  private parseAtom(): Atom {
    const primary = this.parsePrimary();
    let sub: MNode | undefined;
    let sup: MNode | undefined;
    let limitsOverride: LimitMode | undefined;

    while (true) {
      this.skipSpaces();
      const token = this.peek();
      if (token.kind === "cmd" && (token.name === "limits" || token.name === "nolimits" || token.name === "displaylimits")) {
        this.position += 1;
        limitsOverride = token.name === "nolimits" ? "subsup" : "underover";
        continue;
      }
      if (token.kind !== "sup" && token.kind !== "sub") break;
      this.position += 1;
      const argument = this.parseScriptArgument();
      if (token.kind === "sup") sup = argument;
      else sub = argument;
    }

    if (sub === undefined && sup === undefined) return primary;
    const mode = limitsOverride ?? primary.limits ?? "subsup";
    const base = primary.node;

    if (sub !== undefined && sup !== undefined) {
      return {
        node: mode === "underover"
          ? { kind: "munderover", base, under: sub, over: sup }
          : { kind: "msubsup", base, sub, sup },
      };
    }
    if (sub !== undefined) {
      return { node: mode === "underover" ? { kind: "munder", base, under: sub } : { kind: "msub", base, sub } };
    }
    return { node: mode === "underover" ? { kind: "mover", base, over: sup! } : { kind: "msup", base, sup: sup! } };
  }

  private parsePrimary(): Atom {
    this.skipSpaces();
    const token = this.next();
    switch (token.kind) {
      case "lbrace": {
        const inner = this.parseSequence({ stopAtBrace: true });
        return { node: mrow(inner.nodes) };
      }
      case "rbrace":
        return { node: mtext("}") };
      case "sup":
        return { node: mtext("^") };
      case "sub":
        return { node: mtext("_") };
      case "space":
        return { node: mspace("0.25em") };
      case "eof":
        return { node: emptyRow() };
      case "char":
        return { node: this.parseCharacter(token.value) };
      case "cmd":
        return this.parseCommand(token.name);
    }
    return { node: emptyRow() };
  }

  private parseCharacter(value: string): MNode {
    if (value === "~") return mspace("0.25em");
    if (value === "&") return mspace("0.5em");
    if (isDigit(value)) return this.parseNumber(value);
    if (value === ".") {
      const next = this.peek();
      if (next.kind === "char" && isDigit(next.value)) return this.parseNumber(".");
      return mo(".");
    }
    if (/[A-Za-z]/.test(value)) return mi(value);
    if (isCjk(value)) return mtext(value);
    if (/^\p{L}$/u.test(value)) return mi(value);
    if (MO_CHARS.includes(value) || /[\u2200-\u22FF\u27C0-\u27EF\u2A00-\u2AFF]/u.test(value)) {
      return mo(value);
    }
    return mtext(value);
  }

  private readDigits(): string {
    let out = "";
    while (true) {
      const token = this.peek();
      if (token.kind === "char" && isDigit(token.value)) {
        out += token.value;
        this.position += 1;
      } else {
        break;
      }
    }
    return out;
  }

  private parseNumber(first: string): MNode {
    if (first === ".") return mn("." + this.readDigits());
    let text = first + this.readDigits();
    const dot = this.peek();
    if (dot.kind === "char" && dot.value === ".") {
      const after = this.tokens[this.position + 1];
      if (after !== undefined && after.kind === "char" && isDigit(after.value)) {
        this.position += 1; // 小数点
        text += "." + this.readDigits();
      }
    }
    return mn(text);
  }

  private parseCommand(name: string): Atom {
    switch (name) {
      case "frac":
      case "dfrac":
      case "tfrac":
        return { node: this.parseFraction() };
      case "sqrt":
        return { node: this.parseRoot() };
      case "left":
        return { node: this.parseLeftRight() };
      case "right":
        // 孤立的 \right：可见降级，不消费后面的分隔符
        return { node: mtext("\\right") };
      case "text":
        return { node: mtext(this.readRawGroup()) };
      case "mathrm":
        return { node: applyVariant(this.parseArgument(), "normal") };
      case "mathbf":
        return { node: applyVariant(this.parseArgument(), "bold") };
      case "mathbb":
        return { node: applyVariant(this.parseArgument(), "double-struck") };
      case "mathcal":
        return { node: applyVariant(this.parseArgument(), "script") };
      case "operatorname": {
        const starred = this.consumeStar();
        const raw = this.readRawGroup().trim();
        const node = /^[A-Za-z]+$/.test(raw) ? mi(raw, "normal") : mtext(raw);
        return { node, limits: starred ? "underover" : "subsup" };
      }
      case "mathop":
        return { node: this.parseArgument(), limits: "underover" };
      case "limits":
      case "nolimits":
      case "displaylimits":
      case "displaystyle":
      case "textstyle":
      case "scriptstyle":
      case "scriptscriptstyle":
      case "relax":
        return { node: emptyRow() };
      default:
        break;
    }

    const accent = lookup(ACCENTS, name);
    if (accent !== undefined) {
      const base = this.parseArgument();
      return { node: { kind: "mover", base, over: mo(accent, { stretchy: true }), accent: true } };
    }

    const big = lookup(BIG_OPS, name);
    if (big !== undefined) {
      return {
        node: mo(big.ch, { largeop: true, movablelimits: big.movable === true }),
        limits: big.limits,
      };
    }

    const fn = lookup(NAMED_FUNCTIONS, name);
    if (fn !== undefined) {
      return { node: mi(name, "normal"), limits: fn.limits === true ? "underover" : undefined };
    }

    const greek = lookup(GREEK, name);
    if (greek !== undefined) return { node: mi(greek) };

    const symbol = lookup(SYMBOLS, name);
    if (symbol !== undefined) return { node: mo(symbol) };

    const spacing = lookup(SPACES, name);
    if (spacing !== undefined) return { node: mspace(spacing) };

    if (name === "\\") return { node: lineBreak() };

    // 未知命令（含孤立的反斜杠）：保留为可见文本，且全部经过 XML 转义
    return { node: mtext(name === "" ? "\\" : `\\${name}`) };
  }

  private parseScriptArgument(): MNode {
    this.skipSpaces();
    const token = this.peek();
    if (
      token.kind === "eof" ||
      token.kind === "rbrace" ||
      (token.kind === "cmd" && token.name === "right")
    ) {
      return emptyRow();
    }
    return this.parsePrimary().node;
  }

  private parseFraction(): MNode {
    const num = this.parseArgument();
    const den = this.parseArgument();
    return { kind: "mfrac", num, den };
  }

  private parseRoot(): MNode {
    this.skipSpaces();
    let index: MNode | undefined;
    const first = this.peek();
    if (first.kind === "char" && first.value === "[") {
      this.position += 1;
      const inner = this.parseSequence({ stopAtBracket: true });
      index = mrow(inner.nodes);
    }
    const body = this.parseArgument();
    return index === undefined
      ? { kind: "msqrt", body }
      : { kind: "mroot", body, index };
  }

  private parseLeftRight(): MNode {
    const left = this.readDelimiter();
    const inner = this.parseSequence({ stopAtRight: true });
    const children: MNode[] = [];
    if (left.node !== undefined) children.push(left.node);
    children.push(...inner.nodes);
    if (inner.right?.node !== undefined) children.push(inner.right.node);
    return mrow(children);
  }

  private readDelimiter(): Delimiter {
    this.skipSpaces();
    const token = this.peek();
    if (token.kind === "eof" || token.kind === "rbrace") return { raw: "" };
    if (token.kind === "lbrace") {
      this.position += 1;
      return { raw: "{", node: mo("{", { fence: true, stretchy: true }) };
    }
    if (token.kind === "char") {
      this.position += 1;
      if (token.value === ".") return { raw: token.value };
      return { raw: token.value, node: mo(token.value, { fence: true, stretchy: true }) };
    }
    if (token.kind === "cmd") {
      this.position += 1;
      const mapped = lookup(DELIMITERS, token.name);
      if (mapped !== undefined) {
        return mapped === ""
          ? { raw: token.raw }
          : { raw: mapped, node: mo(mapped, { fence: true, stretchy: true }) };
      }
      return { raw: token.raw, node: mtext(token.raw) };
    }
    this.position += 1;
    return { raw: token.raw };
  }

  /** 取一个参数：花括号组，或单个 primary（TeX 的 `\frac12` 也合法）。 */
  private parseArgument(): MNode {
    this.skipSpaces();
    const token = this.peek();
    if (
      token.kind === "eof" ||
      token.kind === "rbrace" ||
      (token.kind === "cmd" && token.name === "right")
    ) {
      return emptyRow();
    }
    if (token.kind === "lbrace") {
      this.position += 1;
      const inner = this.parseSequence({ stopAtBrace: true });
      return mrow(inner.nodes);
    }
    return this.parsePrimary().node;
  }

  private consumeStar(): boolean {
    this.skipSpaces();
    const token = this.peek();
    if (token.kind === "char" && token.value === "*") {
      this.position += 1;
      return true;
    }
    return false;
  }

  /**
   * 原样读取花括号内容，供 \text / \operatorname 使用。
   * 嵌套花括号只作为分组，不输出；控制符号 \{ \} \% 等还原成可见字符。
   */
  private readRawGroup(): string {
    this.skipSpaces();
    const start = this.peek();
    if (start.kind !== "lbrace") {
      if (start.kind === "eof") return "";
      this.position += 1;
      return this.rawTokenAsText(start);
    }
    this.position += 1;
    let depth = 1;
    let out = "";
    while (true) {
      const token = this.next();
      if (token.kind === "eof") break;
      if (token.kind === "lbrace") {
        depth += 1;
        continue;
      }
      if (token.kind === "rbrace") {
        depth -= 1;
        if (depth === 0) break;
        continue;
      }
      out += this.rawTokenAsText(token);
    }
    return out;
  }

  private rawTokenAsText(token: Token): string {
    if (token.kind === "cmd") {
      if (token.name === "\\") return " ";
      if (token.name === "," || token.name === ";" || token.name === ":" || token.name === " ") return " ";
      if (token.name.length === 1 && "{}%$&#_".includes(token.name)) return token.name;
      return token.raw;
    }
    return token.raw;
  }
}

/* -------------------------------------------------------------------------
 * 公开 API：TeX → MathML
 * ---------------------------------------------------------------------- */

/**
 * 把 TeX 子集转成 MathML 字符串。任何情况下都不抛异常：
 * 解析器内部的防御之外，这里还有一层兜底，坏输入退化为一段转义文本。
 */
export function texToMathML(tex: string, opts: TexToMathMLOptions = {}): string {
  const source = typeof tex === "string" ? tex : String(tex);
  const display = opts?.display === true;
  let body: string;
  try {
    body = serialize(new TexParser(tokenize(source)).parse());
  } catch {
    body = `<mtext>${escapeXml(source)}</mtext>`;
  }
  const displayMode = display ? "block" : "inline";
  return `<math xmlns="${MATHML_NS}" display="${displayMode}" aria-label="${escapeXmlAttribute(source)}">${body}</math>`;
}

/* -------------------------------------------------------------------------
 * 富文本：manifest prose → 安全 HTML
 *
 * 块级规则（可预测、便于测试）：
 * - 空行分段，每段输出一个 <p>；
 * - 段内单个换行输出 <br>；
 * - $$...$$ 与 \[...\] 独占一个 <div class="math-display"> 块；
 * - 未闭合的定界符按普通文本显示，不吞掉后文。
 * 行内支持 **粗体**、`代码`、$...$ 与 \(...\)。
 * 所有非数学文本在拼接前逐字符做 HTML 转义。
 * ---------------------------------------------------------------------- */

type RichPart = { kind: "text"; value: string } | { kind: "display"; value: string };

/** 先切出展示公式；反引号内的 $$ 不当作公式定界符。 */
function splitDisplayMath(source: string): RichPart[] {
  const parts: RichPart[] = [];
  let textStart = 0;
  let i = 0;
  let inCode = false;
  while (i < source.length) {
    if (source[i] === "`") {
      inCode = !inCode;
      i += 1;
      continue;
    }
    if (inCode) {
      i += 1;
      continue;
    }
    if (source.startsWith("$$", i)) {
      const end = source.indexOf("$$", i + 2);
      if (end !== -1) {
        if (i > textStart) parts.push({ kind: "text", value: source.slice(textStart, i) });
        parts.push({ kind: "display", value: source.slice(i + 2, end) });
        i = end + 2;
        textStart = i;
        continue;
      }
    }
    if (source.startsWith("\\[", i)) {
      const end = source.indexOf("\\]", i + 2);
      if (end !== -1) {
        if (i > textStart) parts.push({ kind: "text", value: source.slice(textStart, i) });
        parts.push({ kind: "display", value: source.slice(i + 2, end) });
        i = end + 2;
        textStart = i;
        continue;
      }
    }
    i += 1;
  }
  if (textStart < source.length) parts.push({ kind: "text", value: source.slice(textStart) });
  return parts;
}

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function findInlineMathEnd(text: string, from: number): number {
  for (let i = from; i < text.length; i += 1) {
    const ch = text[i];
    if (ch === "\\") {
      i += 1;
      continue;
    }
    if (ch === "$") return i;
    if (ch === "\n") return -1;
  }
  return -1;
}

function renderInline(text: string): string {
  let out = "";
  let i = 0;
  while (i < text.length) {
    if (text.startsWith("\\(", i)) {
      const end = text.indexOf("\\)", i + 2);
      if (end !== -1) {
        out += texToMathML(text.slice(i + 2, end), { display: false });
        i = end + 2;
        continue;
      }
      out += escapeHtml("\\(");
      i += 2;
      continue;
    }

    const ch = text[i];

    if (ch === "`") {
      const end = text.indexOf("`", i + 1);
      if (end !== -1) {
        out += `<code>${escapeHtml(text.slice(i + 1, end))}</code>`;
        i = end + 1;
        continue;
      }
      out += "`";
      i += 1;
      continue;
    }

    if (ch === "*" && text[i + 1] === "*") {
      const end = text.indexOf("**", i + 2);
      if (end !== -1) {
        out += `<strong>${renderInline(text.slice(i + 2, end))}</strong>`;
        i = end + 2;
        continue;
      }
      out += "**";
      i += 2;
      continue;
    }

    if (ch === "$") {
      if (text[i + 1] === "$") {
        // splitDisplayMath 已经尝试配对；走到这里说明未闭合，按普通文本显示
        out += escapeHtml("$$");
        i += 2;
        continue;
      }
      const end = findInlineMathEnd(text, i + 1);
      if (end !== -1) {
        out += texToMathML(text.slice(i + 1, end), { display: false });
        i = end + 1;
        continue;
      }
      out += "$";
      i += 1;
      continue;
    }

    if (ch === "\\") {
      const next = text[i + 1];
      if (next !== undefined && "$`*\\[]{}()_#%&".includes(next)) {
        out += escapeHtml(next);
        i += 2;
        continue;
      }
      out += "\\";
      i += 1;
      continue;
    }

    out += escapeHtml(ch);
    i += 1;
  }
  return out;
}

function renderParagraph(text: string): string {
  const lines = text.split("\n").map((line) => line.trim());
  while (lines.length > 0 && lines[0] === "") lines.shift();
  while (lines.length > 0 && lines[lines.length - 1] === "") lines.pop();
  if (lines.length === 0) return "";
  return `<p>${lines.map(renderInline).join("<br>")}</p>`;
}

/**
 * manifest 文本 → 安全 HTML。文本仍被视为可信输入（不接受用户 HTML），
 * 但所有非数学内容都转义，`<script>` 只会以文本出现。
 */
export function renderRichText(text: string): string {
  const source = (typeof text === "string" ? text : String(text)).replace(/\r\n?/g, "\n");
  const blocks: string[] = [];
  for (const part of splitDisplayMath(source)) {
    if (part.kind === "display") {
      blocks.push(`<div class="math-display">${texToMathML(part.value, { display: true })}</div>`);
      continue;
    }
    for (const paragraph of part.value.split(/\n[ \t]*\n+/)) {
      const html = renderParagraph(paragraph);
      if (html !== "") blocks.push(html);
    }
  }
  return blocks.join("\n");
}
