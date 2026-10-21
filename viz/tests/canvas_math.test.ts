/**
 * Canvas 数学排版的冒烟测试：图表里的 `$...$` 必须被画成上下标/分式，
 * 而不是把美元符号和反斜杠命令原样画到画布上。
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import { drawRichText, measureRichText, wrapRichText } from "../shared/src/canvas_math.js";

interface Call {
  text: string;
  x: number;
  y: number;
  font: string;
  color: string;
}

interface FakeCtx {
  ctx: CanvasRenderingContext2D;
  calls: Call[];
  strokes: () => number;
}

function fakeCtx(): FakeCtx {
  const calls: Call[] = [];
  const state = { strokes: 0 };
  const ctx = {
    font: "",
    fillStyle: "",
    strokeStyle: "",
    lineWidth: 1,
    save() {},
    restore() {},
    beginPath() {},
    moveTo() {},
    lineTo() {},
    stroke() {
      state.strokes += 1;
    },
    measureText(text: string) {
      const size = Number(/([\d.]+)px/.exec(this.font ?? "12px")?.[1] ?? 12);
      return { width: text.length * size * 0.55, actualBoundingBoxAscent: size * 0.72, actualBoundingBoxDescent: size * 0.2 };
    },
    fillText(text: string, x: number, y: number) {
      calls.push({ text, x, y, font: this.font, color: this.fillStyle });
    },
  };
  return { ctx: ctx as unknown as CanvasRenderingContext2D, calls, strokes: () => state.strokes };
}

const style = { size: 13, fontFamily: "sans-serif", color: "#123456" };

test("普通文字不经过数学排版，颜色/宽度正常", () => {
  const { ctx, calls } = fakeCtx();
  const size = drawRichText(ctx, "普通文字", 10, 20, style);
  assert.ok(size.w > 0);
  assert.equal(calls.length, 1);
  assert.equal(calls[0]?.text, "普通文字");
  assert.equal(calls[0]?.color, "#123456");
});

test("$...$ 中的希腊字母、上下标与分式被绘制，且不出现 $ 或反斜杠", () => {
  const { ctx, calls, strokes } = fakeCtx();
  const text = "误差 $\\kappa_2^2=\\frac{\\sigma_1}{\\sigma_2}$ 与 $\\|x\\|$";
  drawRichText(ctx, text, 10, 40, style);
  const joined = calls.map((c) => c.text).join("");
  assert.ok(joined.includes("κ"), "应画出 κ");
  assert.ok(joined.includes("σ"), "应画出 σ");
  assert.ok(!joined.includes("$"), "不应画出美元符号");
  assert.ok(!joined.includes("\\"), "不应画出 TeX 命令");
  assert.ok(calls.some((c) => c.text === "2" && Number(/([\d.]+)px/.exec(c.font)?.[1] ?? 99) < 13), "上标应使用更小字号");
  assert.ok(calls.some((c) => c.text === "1"), "分式分子应被画出");
  assert.ok(strokes() >= 1, "分式横线应被画出");
  assert.ok(calls.every((c) => c.color === "#123456"), "数学段应继承文本颜色");
});

test("度量与绘制使用同一套排版；折行不拆开公式块", () => {
  const { ctx } = fakeCtx();
  const m = measureRichText(ctx, "$x_1$ 与 $x_2$", style);
  assert.ok(m.w > 0);
  const lines = wrapRichText(ctx, "前 $\\kappa$ 后 $\\sigma$", 40, style);
  assert.ok(lines.length >= 2);
  for (const line of lines) {
    const dollars = (line.match(/\$/g) ?? []).length;
    assert.equal(dollars % 2, 0, `折行后公式定界符必须成对：${line}`);
  }
});

test("未知命令按字面量画出，绝不抛异常", () => {
  const { ctx, calls } = fakeCtx();
  assert.doesNotThrow(() => drawRichText(ctx, "$\\notARealCommand{x}$", 0, 0, style));
  assert.ok(calls.length > 0);
});
