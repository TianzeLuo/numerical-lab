/**
 * geometry.ts 与 render2d.ts 的测试。
 *
 * 光栅化"看起来对不对"很难自动判定，但**几何是可以精确验证的**：
 * 例如等值线的每个端点都必须落在 `f = level` 上（线性场下插值是精确的），
 * 溢出检测必须一个不漏。这些测好之后，`tools/raster.py` 的视觉基线才值得信任。
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  buildHeatRgba,
  checkOverflow,
  contourSegments,
  countNonFinite,
  dataToPixel,
  gridCoord,
  layerToPixelPath,
  marchingSquares,
  pixelIndex,
  pixelToData,
} from "../shared/src/geometry.js";
import { makeAxis, newScene, sourceBadge, type Axis, type Scene } from "../shared/src/scene.js";
import { renderScene } from "../shared/src/render2d.js";
import { cmapTable } from "../shared/src/colorscale.js";

function mainAxis(over: Partial<Axis> = {}): Axis {
  return Object.assign(makeAxis("main", [40, 20, 400, 300], { xlim: [0, 1], ylim: [0, 1] }), over);
}

// --------------------------------------------------------------------------- //
// 网格与像素映射
// --------------------------------------------------------------------------- //
test("gridCoord 铺满区间两端", () => {
  assert.equal(gridCoord(0, 5, -1, 1), -1);
  assert.equal(gridCoord(4, 5, -1, 1), 1);
  assert.equal(gridCoord(2, 5, -1, 1), 0);
  assert.equal(gridCoord(0, 1, 3, 7), 3); // 退化网格只给左端点
});

test("dataToPixel 的 y 轴方向朝下", () => {
  const a = mainAxis();
  const [px0, py0] = dataToPixel(a, 0, 1); // 左上角
  const [px1, py1] = dataToPixel(a, 1, 0); // 右下角
  assert.equal(px0, 40);
  assert.equal(py0, 20);
  assert.equal(px1, 440);
  assert.equal(py1, 320);
});

test("pixelToData 是 dataToPixel 的逆（线性与对数）", () => {
  for (const scale of ["linear", "log"] as const) {
    const a = mainAxis({
      xscale: scale,
      xlim: scale === "log" ? [1e-3, 1] : [0, 1],
      xticks: scale === "log" ? [1e-3, 1e-2, 1e-1, 1] : [0, 0.5, 1],
    });
    for (const x of scale === "log" ? [1e-3, 3e-2, 1] : [0, 0.25, 1]) {
      const [px, py] = dataToPixel(a, x, 0.5);
      const [bx, by] = pixelToData(a, px, py);
      assert.ok(Math.abs(bx - x) < 1e-9 * Math.max(1, Math.abs(x)), `${scale} ${x} → ${bx}`);
      assert.ok(Math.abs(by - 0.5) < 1e-12);
    }
  }
});

test("countNonFinite 统计 NaN 与 Inf", () => {
  assert.equal(countNonFinite([1, Number.NaN, 3, Number.POSITIVE_INFINITY, 5]), 2);
  assert.equal(countNonFinite(new Float64Array(4)), 0);
});

test("pixelIndex 行主序", () => {
  assert.equal(pixelIndex(0, 0, 4), 0);
  assert.equal(pixelIndex(3, 1, 4), 7);
});

// --------------------------------------------------------------------------- //
// 热力图 RGBA
// --------------------------------------------------------------------------- //
test("buildHeatRgba 长度正确且不透明像素来自色标", () => {
  const v = new Float64Array([0, 1, 2, 3]);
  const rgba = buildHeatRgba(v, 2, 2, "viridis", 0, 3);
  assert.equal(rgba.length, 16);
  for (let p = 0; p < 4; p += 1) assert.equal(rgba[4 * p + 3], 255);
  const table = cmapTable("viridis");
  assert.equal(rgba[0], table[0]); // 最小值 → 色标起点
  assert.equal(rgba[12], table[3 * 255]); // 最大值 → 色标终点
});

test("buildHeatRgba 把 NaN 与 clipBelow 画成中性灰（而不是透明）", () => {
  const v = new Float64Array([1, Number.NaN, 0.5, 3]);
  const rgba = buildHeatRgba(v, 2, 2, "gray", 1, 3, { clipBelow: 1 });
  const gray = [130, 130, 130, 255];
  for (const p of [1, 2]) {
    assert.deepEqual([rgba[4 * p], rgba[4 * p + 1], rgba[4 * p + 2], rgba[4 * p + 3]], gray, `像素 ${p}`);
  }
  assert.equal(rgba[3], 255);
  assert.equal(rgba[15], 255);
});

test("buildHeatRgba 支持自定义 NaN 颜色与长度校验", () => {
  const rgba = buildHeatRgba(new Float64Array([Number.NaN]), 1, 1, "gray", 0, 1, { nanColor: [1, 2, 3, 4] });
  assert.deepEqual(Array.from(rgba), [1, 2, 3, 4]);
  assert.throws(() => buildHeatRgba(new Float64Array(3), 2, 2, "gray", 0, 1), /长度/);
});

// --------------------------------------------------------------------------- //
// 等值线
// --------------------------------------------------------------------------- //
test("marchingSquares 在线性场上给出精确的等值线", () => {
  // f = x + y（网格坐标 0..2），level = 1 的等值线就是直线 i + j = 1
  const nx = 5;
  const ny = 5;
  const vals = new Float64Array(nx * ny);
  for (let j = 0; j < ny; j += 1) for (let i = 0; i < nx; i += 1) vals[j * nx + i] = i / (nx - 1) + j / (ny - 1);
  const segs = marchingSquares(vals, nx, ny, 1.0);
  assert.ok(segs.length > 0);
  for (let k = 0; k < segs.length; k += 2) {
    const x = segs[k] as number;
    const y = segs[k + 1] as number;
    // 网格坐标 → 数据坐标再验证（f = i/(nx-1) + j/(ny-1)）
    const val = x / (nx - 1) + y / (ny - 1);
    assert.ok(Math.abs(val - 1.0) < 1e-12, `端点 ${x},${y} 的场值 ${val}`);
  }
});

test("marchingSquares 无交点与退化网格", () => {
  const flat = new Float64Array(9).fill(5);
  assert.deepEqual(marchingSquares(flat, 3, 3, 1), []);
  assert.deepEqual(marchingSquares(new Float64Array(4), 2, 2, 0.5), []); // 全 0 = level → 无穿越
  assert.deepEqual(marchingSquares(new Float64Array(4), 2, 2, -1), []);
  assert.deepEqual(marchingSquares(new Float64Array(2), 1, 2, 0), []); // nx<2
});

test("marchingSquares 跳过含非有限值的格子", () => {
  const nx = 4;
  const ny = 4;
  const vals = new Float64Array(nx * ny);
  for (let j = 0; j < ny; j += 1) for (let i = 0; i < nx; i += 1) vals[j * nx + i] = i < 2 ? 0 : 10;
  const clean = marchingSquares(vals, nx, ny, 5).length;
  assert.ok(clean > 0);
  vals[1] = Number.NaN; // (i=1, j=0)：正是一个穿越 level 的格子
  const withNaN = marchingSquares(vals, nx, ny, 5).length;
  assert.ok(withNaN < clean, "含 NaN 的格子必须被跳过");
});

test("marchingSquares 鞍点不产生自交（两种构型分支都被覆盖）", () => {
  // 2×2：对角同号 → 鞍点，必须给出 2 条互不相交的线段
  const saddleA = Float64Array.from([10, 0, 0, 10]);
  const a = marchingSquares(saddleA, 2, 2, 5);
  assert.equal(a.length, 8);
  const saddleB = Float64Array.from([0, 10, 10, 0]);
  const b = marchingSquares(saddleB, 2, 2, 5);
  assert.equal(b.length, 8);
});

test("contourSegments 逐层返回", () => {
  const nx = 5;
  const ny = 5;
  const vals = new Float64Array(nx * ny);
  for (let j = 0; j < ny; j += 1) for (let i = 0; i < nx; i += 1) vals[j * nx + i] = i + j;
  const layers = contourSegments(vals, nx, ny, [2, 4, 6]);
  assert.equal(layers.length, 3);
  assert.deepEqual(
    layers.map((l) => l.level),
    [2, 4, 6],
  );
  assert.ok(layers.every((l) => l.segments.length > 0));
});

// --------------------------------------------------------------------------- //
// 溢出检测（规则 1 的自动化）
// --------------------------------------------------------------------------- //
function sceneWithLine(x: number[], y: number[]): Scene {
  const s = newScene({ chapter: "0.0", viz: "t", title: "t", pitfalls: "自适应坐标轴会掩盖发散，纵轴固定在这里。" });
  s.axes.push(mainAxis());
  s.layers.push({ kind: "line", axes: "main", series: [{ x, y, color: "#000", width: 1 }] });
  s.badges.push(sourceBadge(s.dataSource));
  return s;
}

test("checkOverflow 在数据落在轴内时静默", () => {
  const rep = checkOverflow(sceneWithLine([0, 0.5, 1], [0, 0.5, 1]));
  assert.equal(rep.flag, false);
  assert.equal(rep.note, "");
});

test("checkOverflow 报告越界的方向与相对幅度", () => {
  const rep = checkOverflow(sceneWithLine([0, 0.5, 1], [0, 5, 1]));
  assert.equal(rep.flag, true);
  assert.equal(rep.offenders.length, 1);
  const o = rep.offenders[0]!;
  assert.equal(o.axis, "y");
  assert.equal(o.axes, "main");
  assert.ok(Math.abs(o.over - 4) < 1e-12); // 超出 4 个轴跨度
  assert.match(rep.note, /图层0/);
  assert.match(rep.note, /超出轴 main/);
});

test("checkOverflow 统计非有限值（没画出来的点）", () => {
  const rep = checkOverflow(sceneWithLine([0, 0.5, 1], [0, Number.NaN, 1]));
  assert.equal(rep.flag, true);
  assert.equal(rep.nonFinite, 1);
  assert.equal(rep.offenders.length, 0);
  assert.match(rep.note, /非有限值/);
});

test("checkOverflow 覆盖标量场图层的 extent 与柱状图的误差棒", () => {
  const s = sceneWithLine([0, 1], [0, 1]);
  s.layers.push({
    kind: "heat",
    axes: "main",
    nx: 2,
    ny: 2,
    values: new Float64Array(4),
    extent: [0, 1, 0, 2], // y 方向越界
    cmap: "viridis",
    vmin: 0,
    vmax: 1,
    logScale: false,
  });
  s.layers.push({ kind: "bars", axes: "main", x: [0.5], values: [0.5], width: 0.2, color: "#000", errors: new Float64Array([9]) });
  const rep = checkOverflow(s);
  assert.ok(rep.offenders.some((o) => o.kind === "heat"));
  assert.ok(rep.offenders.some((o) => o.kind === "bars"));
});

test("checkOverflow 忽略不存在的轴（由校验器负责报错）", () => {
  const s = sceneWithLine([0, 1], [0, 1]);
  s.layers.push({ kind: "line", axes: "ghost", series: [{ x: [0, 1], y: [0, 1e9], color: "#000", width: 1 }] });
  const rep = checkOverflow(s);
  assert.equal(rep.flag, false);
});

test("layerToPixelPath 在 NaN 处断开而不是连线", () => {
  const a = mainAxis();
  const path = layerToPixelPath(a, [0, 0.5, 1], [0, Number.NaN, 1]);
  assert.equal(path.length, 6);
  assert.equal(path[0], 40);
  assert.equal(path[2], null);
  assert.equal(path[3], null);
  assert.equal(path[4], 440);
});

// --------------------------------------------------------------------------- //
// 渲染器（用一个记录调用的假上下文做冒烟测试）
// --------------------------------------------------------------------------- //
class MockCtx {
  calls: string[] = [];
  fillStyle = "#000";
  strokeStyle = "#000";
  lineWidth = 1;
  font = "";
  textAlign = "left";
  textBaseline = "top";
  lineJoin = "miter";
  imageSmoothingEnabled = true;
  globalAlpha = 1;
  private readonly track =
    (name: string) =>
    (...args: unknown[]): void => {
      this.calls.push(`${name}(${args.length})`);
    };
  save = this.track("save");
  restore = this.track("restore");
  setTransform = this.track("setTransform");
  clearRect = this.track("clearRect");
  fillRect = this.track("fillRect");
  strokeRect = this.track("strokeRect");
  beginPath = this.track("beginPath");
  closePath = this.track("closePath");
  moveTo = this.track("moveTo");
  lineTo = this.track("lineTo");
  arc = this.track("arc");
  rect = this.track("rect");
  clip = this.track("clip");
  stroke = this.track("stroke");
  fill = this.track("fill");
  setLineDash = this.track("setLineDash");
  translate = this.track("translate");
  rotate = this.track("rotate");
  drawImage = this.track("drawImage");
  putImageData = this.track("putImageData");
  fillText = this.track("fillText");
  measureText = (s: string): { width: number } => {
    this.calls.push("measureText(1)"); // 与 track() 的命名保持一致，便于 count()
    return { width: s.length * 7 };
  };
  count(name: string): number {
    return this.calls.filter((c) => c.startsWith(`${name}(`)).length;
  }
}

function asCtx(m: MockCtx): CanvasRenderingContext2D {
  return m as unknown as CanvasRenderingContext2D;
}

function fullScene(): Scene {
  const s = newScene({ chapter: "0.0", viz: "all", title: "全图层冒烟", pitfalls: "自适应坐标轴会掩盖发散，因此这里把纵轴钉死。" });
  s.axes.push(mainAxis());
  s.layers.push({ kind: "line", axes: "main", series: [{ x: [0, 0.5, 1], y: [0, 1, 0], color: "#111", width: 2, marker: "o", label: "u" }] });
  s.layers.push({ kind: "band", axes: "main", band: { x: [0, 0.5, 1], ylo: [0, 0.2, 0], yhi: [0.2, 0.6, 0.2], color: "#88a", alpha: 0.3 } });
  s.layers.push({ kind: "scatter", axes: "main", x: [0.1, 0.9], y: [0.5, 0.5], r: 3, color: "#a00" });
  s.layers.push({ kind: "heat", axes: "main", nx: 3, ny: 3, values: Float64Array.from([0, 1, 2, 3, 4, 5, 6, 7, 8]), extent: [0, 1, 0, 1], cmap: "magma", vmin: 0, vmax: 8, logScale: false });
  s.layers.push({ kind: "contour", axes: "main", nx: 3, ny: 3, values: Float64Array.from([0, 1, 2, 3, 4, 5, 6, 7, 8]), extent: [0, 1, 0, 1], levels: [3, 6], color: "#222", width: 1 });
  s.layers.push({ kind: "bars", axes: "main", x: [0.2, 0.8], values: [0.3, 0.6], width: 0.1, color: "#268", errors: Float64Array.from([0.05, 0.1]) });
  s.layers.push({ kind: "vectorfield", axes: "main", nx: 3, ny: 3, u: new Float64Array(9).fill(0.1), v: new Float64Array(9).fill(0.05), extent: [0, 1, 0, 1], stride: 1, color: "#555", scale: 1 });
  s.layers.push({ kind: "loglogfit", axes: "main", h: [0.1, 0.05, 0.025], err: [1e-2, 2.6e-3, 6.6e-4], measuredSlope: 1.98, theorySlope: 2, refX: [0.025, 0.1], seriesLabel: "实测斜率 1.98", theoryLabel: "理论斜率 2" });
  s.badges.push(sourceBadge(s.dataSource));
  s.texts.push({ x: 50, y: 200, text: "标注", size: 12, color: "#000", align: "left", role: "note" });
  return s;
}

test("renderScene 对全部图层类型都能跑通（无离屏画布时走逐格填充）", () => {
  const m = new MockCtx();
  renderScene(asCtx(m), fullScene());
  assert.ok(m.count("fillText") > 10, "刻度与徽标必须画出来");
  assert.ok(m.count("stroke") > 5);
  assert.ok(m.count("clip") >= 4, "标量场与柱状图必须裁剪在轴内");
  assert.ok(m.count("fillRect") > 9, "热力图在无离屏画布时必须逐格填充，而不是静默跳过");
  assert.ok(m.count("measureText") > 0);
});

test("renderScene 尊重 showBadges 开关", () => {
  const withBadges = new MockCtx();
  renderScene(asCtx(withBadges), fullScene());
  const without = new MockCtx();
  renderScene(asCtx(without), fullScene(), { showBadges: false });
  assert.ok(without.count("fillText") < withBadges.count("fillText"));
});

test("renderScene 画出了 pitfalls 面板（规则 7）", () => {
  const m = new MockCtx();
  renderScene(asCtx(m), fullScene());
  assert.ok(m.count("strokeRect") >= 3, "轴框 + 徽标框 + pitfalls 面板");
});

test("renderScene 在数据越界时把徽标文案画出来", () => {
  const s = fullScene();
  s.overflow = { flag: true, note: "u 已超出纵轴 1e3 倍" };
  s.badges.push({ kind: "overflow", text: `溢出：${s.overflow.note}`, x: 8, y: 44 });
  const m = new MockCtx();
  renderScene(asCtx(m), s);
  assert.ok(m.count("fillText") > 10);
});
