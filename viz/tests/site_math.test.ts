/**
 * 站点公式渲染的验收测试：TeX → MathML 与 manifest 富文本。
 *
 * 全部测试只碰字符串，不建 DOM；结构断言优先于完整字符串，
 * 但确定性、注入、坏输入与真实公式会做整串级检查。
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import { renderRichText, texToMathML } from "../shared/src/math.js";

const MATH_OPEN = '<math xmlns="http://www.w3.org/1998/Math/MathML"';

function occurrences(haystack: string, needle: string): number {
  return haystack.split(needle).length - 1;
}

test("texToMathML：根节点、display 模式与 aria-label 转义", () => {
  const inline = texToMathML("x<y&z");
  assert.ok(inline.startsWith(`${MATH_OPEN} display="inline"`), "默认应为行内公式");
  assert.ok(inline.endsWith("</math>"));
  assert.ok(inline.includes('aria-label="x&lt;y&amp;z"'), "aria-label 必须是转义后的原始 TeX");
  assert.ok(!inline.includes("<script>"));

  const block = texToMathML("x", { display: true });
  assert.ok(block.startsWith(`${MATH_OPEN} display="block"`));
  assert.ok(block.includes("<mi>x</mi>"));
});

test("texToMathML：符号、数字与普通空格", () => {
  const out = texToMathML("x+1=2.5");
  assert.ok(
    out.includes('<mrow><mi>x</mi><mo stretchy="false">+</mo><mn>1</mn><mo stretchy="false">=</mo><mn>2.5</mn></mrow>'),
    "普通运算符应显式 stretchy=false，避免被同行的分式拉高",
  );
  // 数学模式里的空白不产生节点
  assert.ok(!out.includes("<mtext> </mtext>"));
  const spaced = texToMathML("a   b\n\tc");
  assert.ok(spaced.includes("<mrow><mi>a</mi><mi>b</mi><mi>c</mi></mrow>"));
});

test("texToMathML：上下标、组合上下标与花括号分组", () => {
  assert.ok(texToMathML("x^2").includes("<msup><mi>x</mi><mn>2</mn></msup>"));
  assert.ok(texToMathML("x_i").includes("<msub><mi>x</mi><mi>i</mi></msub>"));
  const both = texToMathML("x_{i}^{2}");
  assert.ok(
    both.includes("<msubsup><mi>x</mi><mrow><mi>i</mi></mrow><mrow><mn>2</mn></mrow></msubsup>"),
    "先下后上应合成 msubsup",
  );
  const reversed = texToMathML("x^{2}_{i}");
  assert.ok(reversed.includes("<msubsup>"), "先上后下也应合成 msubsup");
  assert.ok(texToMathML("{a+b}").includes('<mrow><mi>a</mi><mo stretchy="false">+</mo><mi>b</mi></mrow>'));
});

test("texToMathML：分式与根式", () => {
  const frac = texToMathML(String.raw`\frac{a}{b}`);
  assert.ok(frac.includes("<mfrac><mrow><mi>a</mi></mrow><mrow><mi>b</mi></mrow></mfrac>"));
  assert.ok(!frac.includes("\\frac"), "命令字面量不应出现在 MathML 内容里");

  const sqrt = texToMathML(String.raw`\sqrt{x}`);
  assert.ok(sqrt.includes("<msqrt><mrow><mi>x</mi></mrow></msqrt>"));
  assert.ok(!sqrt.includes("\\sqrt"));

  const root = texToMathML(String.raw`\sqrt[3]{x}`);
  assert.ok(root.includes("<mroot><mrow><mi>x</mi></mrow><mrow><mn>3</mn></mrow></mroot>"));
  assert.ok(!root.includes("<msqrt>"));
});

test("texToMathML：求和/求积用 munderover，积分用 msubsup", () => {
  const sum = texToMathML(String.raw`\sum_{i=0}^{n} i`);
  assert.ok(sum.includes("<munderover><mo largeop=\"true\" movablelimits=\"true\">∑</mo>"));
  assert.ok(sum.includes('<mi>i</mi><mo stretchy="false">=</mo><mn>0</mn>'));
  assert.ok(sum.includes("<mi>n</mi>"));

  const prod = texToMathML(String.raw`\prod_{i=1}^{n} i`);
  assert.ok(prod.includes("<munderover>"));
  assert.ok(prod.includes("∏"));

  const integral = texToMathML(String.raw`\int_{a}^{b} f(x)\,dx`);
  assert.ok(integral.includes("<msubsup><mo largeop=\"true\">∫</mo>"));
  assert.ok(integral.includes('width="0.1667em"'), "\\, 应变成细空格");

  assert.ok(texToMathML(String.raw`\sum_{i}`).includes("<munder><mo"));
  assert.ok(texToMathML(String.raw`\sum^{n}`).includes("<mover><mo"));
});

test("texToMathML：\\text / 字体变体 / 算子名 / 标准函数", () => {
  const text = texToMathML(String.raw`\text{中文 与 \{x\}}`);
  assert.ok(text.includes("<mtext>中文 与 {x}</mtext>"));

  assert.ok(texToMathML(String.raw`\mathrm{d}`).includes('<mi mathvariant="normal">d</mi>'));
  assert.ok(texToMathML(String.raw`\mathbf{1}`).includes('<mn mathvariant="bold">1</mn>'));
  assert.ok(texToMathML(String.raw`\mathbb{R}`).includes('<mi mathvariant="double-struck">R</mi>'));
  assert.ok(texToMathML(String.raw`\mathcal{O}`).includes('<mi mathvariant="script">O</mi>'));
  assert.ok(
    texToMathML(String.raw`\operatorname{argmax}_{x}`).includes('<mi mathvariant="normal">argmax</mi>'),
  );
  assert.ok(texToMathML(String.raw`\operatorname*{argmax}_{x}`).includes("<munder>"));

  const sin = texToMathML(String.raw`\sin x`);
  assert.ok(sin.includes('<mi mathvariant="normal">sin</mi>'));
  assert.ok(sin.includes("<mi>x</mi>"));
  const lim = texToMathML(String.raw`\lim_{n\to\infty} \frac{1}{n}=0`);
  assert.ok(lim.includes('<munder><mi mathvariant="normal">lim</mi>'));
  assert.ok(lim.includes("<mfrac>"));
});

test("texToMathML：希腊字母完整映射", () => {
  const greek: Array<[string, string]> = [
    ["alpha", "α"], ["beta", "β"], ["gamma", "γ"], ["delta", "δ"],
    ["epsilon", "ϵ"], ["varepsilon", "ε"], ["zeta", "ζ"], ["eta", "η"],
    ["theta", "θ"], ["vartheta", "ϑ"], ["iota", "ι"], ["kappa", "κ"],
    ["lambda", "λ"], ["mu", "μ"], ["nu", "ν"], ["xi", "ξ"], ["pi", "π"],
    ["rho", "ρ"], ["sigma", "σ"], ["tau", "τ"], ["upsilon", "υ"],
    ["phi", "ϕ"], ["varphi", "φ"], ["chi", "χ"], ["psi", "ψ"], ["omega", "ω"],
    ["Gamma", "Γ"], ["Delta", "Δ"], ["Theta", "Θ"], ["Lambda", "Λ"], ["Xi", "Ξ"],
    ["Pi", "Π"], ["Sigma", "Σ"], ["Upsilon", "Υ"], ["Phi", "Φ"], ["Psi", "Ψ"],
    ["Omega", "Ω"],
  ];
  for (const [name, expected] of greek) {
    assert.ok(texToMathML(`\\${name}`).includes(`<mi>${expected}</mi>`), `\\${name} → ${expected}`);
  }
});

test("texToMathML：关系符、算子、定界符与省略号", () => {
  const cases: Array<[string, string]> = [
    ["le", "≤"], ["leq", "≤"], ["ge", "≥"], ["geq", "≥"], ["ne", "≠"],
    ["neq", "≠"], ["approx", "≈"], ["sim", "∼"], ["simeq", "≃"],
    ["equiv", "≡"], ["times", "×"], ["cdot", "⋅"], ["pm", "±"],
    ["mp", "∓"], ["to", "→"], ["rightarrow", "→"], ["leftarrow", "←"],
    ["Rightarrow", "⇒"], ["Leftarrow", "⇐"], ["leftrightarrow", "↔"],
    ["infty", "∞"], ["partial", "∂"], ["nabla", "∇"],
    ["ldots", "…"], ["cdots", "⋯"], ["dots", "…"],
    ["|", "‖"], ["{", "{"], ["}", "}"], ["langle", "⟨"], ["rangle", "⟩"],
  ];
  for (const [name, expected] of cases) {
    assert.ok(texToMathML(`\\${name}`).includes(`>${expected}</mo>`), `\\${name} → ${expected}`);
  }
  assert.ok(texToMathML(String.raw`\sum`).includes("∑"));
  assert.ok(texToMathML(String.raw`\prod`).includes("∏"));
  assert.ok(texToMathML(String.raw`\int`).includes("∫"));
  assert.ok(texToMathML(String.raw`\oint`).includes("∮"));
});

test("texToMathML：重音与 \\left … \\right", () => {
  const accents: Array<[string, string]> = [
    ["hat", "^"], ["bar", "¯"], ["tilde", "~"], ["vec", "→"], ["dot", "˙"],
  ];
  for (const [name, accent] of accents) {
    const out = texToMathML(`\\${name}{x}`);
    assert.ok(out.includes('<mover accent="true">'), `\\${name} 应使用 mover`);
    assert.ok(out.includes(`<mo stretchy="true">${accent}</mo>`));
  }

  const fenced = texToMathML(String.raw`\left( \frac{a}{b} \right)`);
  assert.ok(fenced.includes('<mo fence="true" stretchy="true">(</mo>'));
  assert.ok(fenced.includes('<mo fence="true" stretchy="true">)</mo>'));
  assert.ok(fenced.includes("<mfrac>"));

  const plain = texToMathML("(x)");
  assert.ok(plain.includes('<mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo>'));
});

test("texToMathML：空白命令、换行与 \\text 内空格", () => {
  const spaced = texToMathML(String.raw`a\,b\;c\!d\quad e\qquad f`);
  assert.equal(occurrences(spaced, "<mspace width="), 5);
  assert.ok(spaced.includes('width="0.1667em"'));
  assert.ok(spaced.includes('width="0.2778em"'));
  assert.ok(spaced.includes('width="-0.1667em"'));
  assert.ok(spaced.includes('width="1em"'));
  assert.ok(spaced.includes('width="2em"'));

  const broken = texToMathML(String.raw`a \\ b`);
  assert.ok(broken.includes('<mspace linebreak="newline"/>'));

  assert.ok(texToMathML(String.raw`\text{a b}`).includes("<mtext>a b</mtext>"));
});

test("texToMathML：坏输入不抛异常，且仍输出闭合的 math", () => {
  const malformed: string[] = [
    String.raw`\frac{a}`,
    String.raw`\frac`,
    String.raw`\sqrt[3`,
    "x^",
    "x_",
    "{",
    "}",
    "\\",
    String.raw`\left( x`,
    String.raw`\right)`,
    String.raw`\text{未闭合`,
    "^",
  ];
  for (const input of malformed) {
    let output = "";
    assert.doesNotThrow(() => {
      output = texToMathML(input);
    }, `坏输入不应抛异常：${JSON.stringify(input)}`);
    assert.ok(output.startsWith(MATH_OPEN), `坏输入应仍返回 math 根：${JSON.stringify(input)}`);
    assert.ok(output.endsWith("</math>"), `math 根应闭合：${JSON.stringify(input)}`);
  }
  assert.ok(texToMathML(String.raw`\frac{a}`).includes("<mfrac>"), "缺分母仍应保留分式结构");
  assert.ok(texToMathML(String.raw`\left( x`).includes('<mo fence="true"'));
});

test("texToMathML：未知命令降级为可见且转义的文本", () => {
  const out = texToMathML(String.raw`\foobar + x`);
  assert.ok(out.includes("<mtext>\\foobar</mtext>"), "未知命令应显示命令名");
  assert.ok(out.includes('aria-label="&#92;foobar + x"'), "aria 中的反斜杠用实体编码");
  assert.doesNotThrow(() => texToMathML(String.raw`\__proto__ \constructor`));

  const injection = texToMathML(String.raw`\text{<script>alert("x")</script>}`);
  assert.ok(!injection.includes("<script>"), "mtext 不得产生真实 script 标签");
  assert.ok(injection.includes("&lt;script&gt;"));
  assert.ok(injection.includes("&quot;x&quot;"));
  assert.ok(!injection.includes('aria-label="\\text{<script>'), "属性也必须转义");
});

test("texToMathML：真实公式结构、无命令字面量与确定性", () => {
  const tex = String.raw`f(x)-p_n(x)=\frac{f^{(n+1)}(\xi)}{(n+1)!}\,\omega(x)`;
  const out = texToMathML(tex);
  assert.ok(out.includes("<mfrac>"), "必须真的生成分式");
  assert.ok(out.includes("<msup>"), "f^{(n+1)} 必须生成上标");
  assert.ok(out.includes("<msub>"), "p_n 必须生成下标");
  assert.ok(out.includes("<mi>ω</mi>"));
  assert.ok(out.includes('<mspace width="0.1667em"/>'));
  assert.ok(!out.includes("\\frac"), "序列化结果不应含 TeX 源命令（aria 已实体化）");
  assert.equal(texToMathML(tex), texToMathML(tex), "同一输入必须确定性输出");

  const block = texToMathML(tex, { display: true });
  assert.ok(block.includes('display="block"'));
  assert.ok(block.includes("<mfrac>"));
});

test("renderRichText：粗体与行内代码，内容全部转义", () => {
  assert.equal(
    renderRichText("**bold** and `x < 1`"),
    "<p><strong>bold</strong> and <code>x &lt; 1</code></p>",
  );
  assert.equal(
    renderRichText("**<b>bold</b>**"),
    "<p><strong>&lt;b&gt;bold&lt;/b&gt;</strong></p>",
  );
  assert.equal(renderRichText("`<img src=x onerror=1>`"), "<p><code>&lt;img src=x onerror=1&gt;</code></p>");
});

test("renderRichText：$...$ 与 \\(...\\) 行内公式", () => {
  const out = renderRichText(String.raw`面积 $x^2$ 与 \(y\)`);
  assert.ok(out.startsWith("<p>"));
  assert.equal(occurrences(out, "<math "), 2);
  assert.ok(out.includes('display="inline"'));
  assert.ok(out.includes("<msup><mi>x</mi><mn>2</mn></msup>"));
  assert.ok(out.includes("<mi>y</mi>"));
  assert.ok(!out.includes("$x^2$"));
  assert.ok(!out.includes("\\("));

  // 未闭合的 $ 按普通文本处理，不吞后文
  const unmatched = renderRichText("价格 $5");
  assert.ok(unmatched.includes("<p>价格 $5</p>"));
});

test("renderRichText：$$...$$ 与 \\[...\\] 展示公式", () => {
  const dollars = renderRichText(String.raw`$$\frac{a}{b}$$`);
  assert.equal(
    dollars,
    `<div class="math-display">${texToMathML(String.raw`\frac{a}{b}`, { display: true })}</div>`,
  );
  assert.ok(dollars.includes('display="block"'));
  assert.ok(dollars.includes("<mfrac>"));
  assert.ok(!dollars.includes("$$"));

  const brackets = renderRichText(String.raw`\[y=1\]`);
  assert.ok(brackets.startsWith('<div class="math-display">'));
  assert.ok(brackets.includes('display="block"'));
  assert.ok(brackets.includes("<mi>y</mi>"));
  assert.ok(!brackets.includes("\\["));

  const withProse = renderRichText("前\n\n$$x$$\n\n后");
  assert.match(withProse, /^<p>前<\/p>\n<div class="math-display">.*<\/div>\n<p>后<\/p>$/s);
});

test("renderRichText：空行分段、段内换行与空白清理", () => {
  assert.equal(
    renderRichText("第一段\n\n第二段\n换行"),
    "<p>第一段</p>\n<p>第二段<br>换行</p>",
  );
  assert.equal(renderRichText("   \n\n  "), "");
  assert.equal(renderRichText(""), "");
  assert.equal(
    renderRichText("一\n\n\n二"),
    "<p>一</p>\n<p>二</p>",
  );
});

test("renderRichText：HTML 注入在正文、代码与公式文本里都被转义", () => {
  const plain = renderRichText("<script>alert('x')</script>");
  assert.ok(!plain.includes("<script>"));
  assert.ok(plain.includes("&lt;script&gt;"));
  assert.ok(plain.includes("&#39;"));

  const inCode = renderRichText("`<script>alert(1)</script>`");
  assert.ok(!inCode.includes("<script>"));
  assert.ok(inCode.includes("&lt;script&gt;"));

  const inMath = renderRichText(String.raw`$\text{<script>alert(1)</script>}$`);
  assert.ok(!inMath.includes("<script>"));
  assert.ok(inMath.includes("&lt;script&gt;"));

  const inBold = renderRichText("**<script>alert(1)</script>**");
  assert.ok(!inBold.includes("<script>"));
});

test("renderRichText：富文本输出确定性", () => {
  const text = "**标题**\n\n行内 $x_1$ 与 `code`\n\n$$\\sum_{i=1}^{n} i$$";
  assert.equal(renderRichText(text), renderRichText(text));
});


/* ------------------------------------------------------------------------- */
/* 站点正文数据中的 $...$：不能有未闭合定界符或未支持命令                     */
/* ------------------------------------------------------------------------- */

interface GuideTerm {
  term?: unknown;
  desc?: unknown;
}
interface GuideEntry {
  intro?: unknown;
  glossary?: GuideTerm[];
}
interface VizProse {
  id?: unknown;
  description?: unknown;
  theory?: unknown;
  pitfalls?: unknown;
  params?: Array<{ name?: unknown; criticalNote?: unknown }>;
}
interface ChapterManifestProse {
  chapter?: unknown;
  summary?: unknown;
  vizzes?: VizProse[];
}

const ROOT = process.cwd();

function readJson(rel: string): unknown {
  return JSON.parse(readFileSync(`${ROOT}/${rel}`, "utf8")) as unknown;
}

/** 取出所有成对的 `$...$`；未配对的 `$` 单独报告。 */
function inlineMath(source: string, where: string, errors: string[]): string[] {
  const out: string[] = [];
  let i = 0;
  while (i < source.length) {
    if (source[i] !== "$") {
      i += 1;
      continue;
    }
    if (source[i + 1] === "$") {
      errors.push(`${where}: 出现了 $$ 展示公式定界符；正文公式请用单个 $...$`);
      i += 2;
      continue;
    }
    const end = source.indexOf("$", i + 1);
    if (end < 0) {
      errors.push(`${where}: 未闭合的 $...$`);
      break;
    }
    const body = source.slice(i + 1, end);
    if (body.trim() === "") errors.push(`${where}: 空的 $...$`);
    out.push(body);
    i = end + 1;
  }
  return out;
}

/** 正文里除 $...$ 与 `代码` 外不应再出现 TeX 反斜杠命令（会原样显示在页面上）。 */
function checkNoBareCommands(source: string, where: string, errors: string[]): void {
  const plain = source
    .replace(/\$[^$]*\$/g, "")
    .replace(/`[^`]*`/g, "")
    .replace(/\\([\[\](){}])/g, "$1");
  for (const match of plain.matchAll(/\\[A-Za-z]+/g)) {
    errors.push(`${where}: 公式外的 TeX 命令 ${match[0]}（会原样显示）`);
  }
}

function checkInlineMath(tex: string, where: string, errors: string[]): void {
  let braces = 0;
  let left = 0;
  let right = 0;
  for (let i = 0; i < tex.length; i += 1) {
    if (tex[i] === "\\") {
      let j = i + 1;
      if (/[A-Za-z]/.test(tex[j] ?? "")) {
        while (j < tex.length && /[A-Za-z]/.test(tex[j] as string)) j += 1;
      } else if (j < tex.length) {
        j += 1;
      }
      const name = tex.slice(i + 1, j);
      if (name === "left") left += 1;
      else if (name === "right") right += 1;
      i = j - 1;
      continue;
    }
    if (tex[i] === "{") braces += 1;
    else if (tex[i] === "}") {
      braces -= 1;
      if (braces < 0) {
        errors.push(`${where}: 多余的 }`);
        return;
      }
    }
  }
  if (braces !== 0) errors.push(`${where}: 花括号不平衡（${braces > 0 ? "缺 }" : "多 }"}）`);
  if (left !== right) errors.push(`${where}: \\left 与 \\right 数量不一致（${left} vs ${right}）`);
  const html = texToMathML(tex);
  if (/<mtext>\\/.test(html)) {
    const shown = /<mtext>(\\[^<]*)<\/mtext>/.exec(html)?.[1] ?? "?";
    errors.push(`${where}: 渲染器不支持命令 ${shown}（会原样显示在页面上）`);
  }
}

test("站点数据：guide intro / glossary 的每个 $...$ 都能完整渲染", () => {
  const raw = readJson("data/site/guides.json") as { chapters?: Record<string, GuideEntry> };
  const errors: string[] = [];
  for (const [chapter, entry] of Object.entries(raw.chapters ?? {})) {
    const texts: Array<[string, unknown]> = [[`guides.${chapter}.intro`, entry.intro]];
    for (const [i, item] of (entry.glossary ?? []).entries()) {
      texts.push([`guides.${chapter}.glossary[${i}].desc`, item.desc]);
    }
    for (const [where, value] of texts) {
      if (typeof value !== "string") continue;
      checkNoBareCommands(value, where, errors);
      for (const tex of inlineMath(value, where, errors)) checkInlineMath(tex, where, errors);
      assert.doesNotThrow(() => renderRichText(value));
    }
  }
  assert.deepEqual(errors, []);
});

test("站点数据：章节 MANIFEST 的正文公式都能完整渲染", () => {
  const errors: string[] = [];
  const index = readJson("data/manifests/index.json") as { chapters?: Array<{ chapter?: string }> };
  for (const entry of index.chapters ?? []) {
    if (typeof entry.chapter !== "string") continue;
    const manifest = readJson(`data/manifests/${entry.chapter}.json`) as ChapterManifestProse;
    const texts: Array<[string, unknown]> = [[`${entry.chapter}.summary`, manifest.summary]];
    for (const viz of manifest.vizzes ?? []) {
      const tag = `${entry.chapter}/${String(viz.id ?? "?")}`;
      texts.push([`${tag}.description`, viz.description], [`${tag}.pitfalls`, viz.pitfalls]);
      if (Array.isArray(viz.theory)) {
        viz.theory.forEach((line, i) => texts.push([`${tag}.theory[${i}]`, line]));
      }
      for (const [i, param] of (viz.params ?? []).entries()) {
        texts.push([`${tag}.params[${i}].criticalNote`, param.criticalNote]);
      }
    }
    for (const [where, value] of texts) {
      if (typeof value !== "string") continue;
      checkNoBareCommands(value, where, errors);
      for (const tex of inlineMath(value, where, errors)) checkInlineMath(tex, where, errors);
    }
  }
  assert.deepEqual(errors, []);
});
