import assert from "node:assert/strict";
import { test } from "node:test";

import { renderScene } from "../shared/src/render2d.js";
import { makeAxis, newScene } from "../shared/src/scene.js";

interface TextCall {
  text: string;
  color: string;
}

function probeCtx(calls: TextCall[]): CanvasRenderingContext2D {
  const target: Record<string, unknown> = {
    canvas: { width: 600, height: 400 },
    measureText: (text: string) => ({ width: String(text).length * 7 }),
    fillText: (text: string) => {
      calls.push({ text: String(text), color: String(target["fillStyle"]) });
    },
    save: () => undefined,
    restore: () => undefined,
  };
  return new Proxy(target, {
    get(obj, prop: string) {
      if (prop in obj) return obj[prop];
      return () => undefined;
    },
    set(obj, prop: string, value) {
      obj[prop] = value;
      return true;
    },
  }) as unknown as CanvasRenderingContext2D;
}

function sceneWithAxis() {
  const scene = newScene({
    chapter: "0.0",
    viz: "theme_probe",
    title: "主题探针",
    pitfalls: "坐标轴装饰必须随主题换色，否则深色背景上看不见刻度。",
    width: 600,
    height: 400,
  });
  scene.axes.push(
    makeAxis("main", [70, 40, 480, 300], {
      xlim: [0, 1],
      ylim: [0, 1],
      xlabel: "横轴名称",
      ylabel: "纵轴名称",
    }),
  );
  return scene;
}

test("深色主题下刻度和轴名换成浅色，不沿用以黑字为主的浅色配色", () => {
  const calls: TextCall[] = [];
  renderScene(probeCtx(calls), sceneWithAxis(), { theme: "dark", showBadges: false });
  assert.ok(calls.some((c) => c.text === "0.2" && c.color === "#cbd5e0"), "刻度应为浅色");
  assert.ok(calls.some((c) => c.text === "横轴名称" && c.color === "#edf2f7"), "轴名应为浅色");
  assert.ok(calls.every((c) => c.color !== "#333333" && c.color !== "#222222"), "不应残留浅色主题的深色文字");
});

test("浅色主题保持原有深色刻度", () => {
  const calls: TextCall[] = [];
  renderScene(probeCtx(calls), sceneWithAxis(), { theme: "light", showBadges: false });
  assert.ok(calls.some((c) => c.text === "0.2" && c.color === "#333333"));
  assert.ok(calls.some((c) => c.text === "横轴名称" && c.color === "#222222"));
});
