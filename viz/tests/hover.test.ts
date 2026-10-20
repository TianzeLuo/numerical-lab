import assert from "node:assert/strict";
import { test } from "node:test";

import { dataToPixel } from "../shared/src/geometry.js";
import { hoverAt } from "../shared/src/hover.js";
import {
  barsLayer,
  heatLayer,
  lineLayer,
  makeAxis,
  newScene,
  scatterLayer,
  series,
  sourceBadge,
  vectorFieldLayer,
  type Scene,
} from "../shared/src/scene.js";

function baseScene(): Scene {
  const scene = newScene({
    chapter: "0.0",
    viz: "hover_probe",
    title: "悬停",
    pitfalls: "悬停提示只描述图元本身，不替读者猜测因果。",
  });
  scene.axes.push(makeAxis("main", [60, 40, 400, 240], { xlim: [0, 1], ylim: [0, 1] }));
  scene.badges.push(sourceBadge(scene.dataSource));
  return scene;
}

function at(scene: Scene, x: number, y: number): [number, number] {
  return dataToPixel(scene.axes[0]!, x, y);
}

test("hover：命中最近的曲线，返回系列名与两个坐标", () => {
  const scene = baseScene();
  scene.layers.push(
    lineLayer("main", [
      series([0, 0.5, 1], [0.2, 0.5, 0.8], "#2b6cb0", { label: "上界", dash: [4, 3] }),
      series([0, 0.5, 1], [0.1, 0.4, 0.7], "#2b6cb0", { label: "下界" }),
    ]),
  );
  const [px, py] = at(scene, 0.5, 0.5);
  const hit = hoverAt(scene, px, py, { labels: { main: { x: "时间 t", y: "位移 u" } } });
  assert.ok(hit);
  assert.equal(hit.label, "上界");
  assert.equal(hit.dash, true);
  assert.match(hit.xText, /时间 t = 0\.5/);
  assert.match(hit.yText, /位移 u = 0\.5/);
  assert.ok(Math.abs(hit.x - 0.5) < 1e-9 && Math.abs(hit.y - 0.5) < 1e-9);
});

test("hover：远离图元与轴外都不命中", () => {
  const scene = baseScene();
  scene.layers.push(lineLayer("main", series([0, 1], [0, 1], "#333", { label: "对角线" })));
  const [px, py] = at(scene, 0.5, 0.5);
  assert.equal(hoverAt(scene, px, py - 80), null);
  assert.equal(hoverAt(scene, 10, 10), null);
  assert.ok(hoverAt(scene, px, py, { radius: 2 }));
});

test("hover：散点、柱、热图与向量场各自给出可读数值", () => {
  const scatter = baseScene();
  scatter.layers.push(scatterLayer("main", [0.2], [0.8], { label: "采样点" }));
  const [sx, sy] = at(scatter, 0.2, 0.8);
  const dot = hoverAt(scatter, sx + 2, sy + 1);
  assert.equal(dot?.label, "采样点");
  assert.equal(dot?.shape, "point");

  const bars = baseScene();
  bars.layers.push(barsLayer("main", [0.75], [0.6], { width: 0.16, label: "分箱计数", errors: [0.05] }));
  const [bx, by] = at(bars, 0.75, 0.3);
  const bar = hoverAt(bars, bx, by);
  assert.equal(bar?.label, "分箱计数");
  assert.equal(bar?.shape, "bar");
  assert.match(bar?.detail ?? "", /误差 ±0\.05/);

  const heat = baseScene();
  heat.layers.push(heatLayer("main", [0, 0.5, 1, 0.25, 0.75, 0.2], 3, 2, [0, 1, 0, 1], { label: "温度场" }));
  const [hx, hy] = at(heat, 0.5, 0.5);
  const cell = hoverAt(heat, hx, hy);
  assert.equal(cell?.label, "温度场");
  assert.equal(cell?.shape, "cell");
  assert.match(cell?.detail ?? "", /数值 = 0\.75/);

  const field = baseScene();
  field.layers.push(vectorFieldLayer("main", [1, 2, 3, 4], [0, 1, 0, -1], 2, 2, [0, 1, 0, 1], { color: "#555" }));
  const [vx, vy] = at(field, 1, 1);
  const vec = hoverAt(field, vx, vy);
  assert.equal(vec?.shape, "vector");
  assert.match(vec?.detail ?? "", /u = 4/);
});

test("hover：双对数拟合给出实测与理论斜率", () => {
  const scene = baseScene();
  scene.axes[0] = makeAxis("main", [60, 40, 400, 240], {
    xlim: [1e-3, 1e-1],
    ylim: [1e-6, 1e-2],
    xscale: "log",
    yscale: "log",
  });
  scene.layers.push({
    kind: "loglogfit",
    axes: "main",
    h: [1e-3, 1e-2, 1e-1],
    err: [1e-6, 1e-4, 1e-2],
    measuredSlope: 2,
    theorySlope: 2,
    refX: [1e-3, 1e-1],
    seriesLabel: "实测误差",
    theoryLabel: "理论斜率 2",
  });
  const [px, py] = at(scene, 1e-2, 1e-4);
  const hit = hoverAt(scene, px, py);
  assert.equal(hit?.label, "实测误差");
  assert.match(hit?.detail ?? "", /实测斜率 2/);
  assert.match(hit?.detail ?? "", /理论斜率 2/);
});
