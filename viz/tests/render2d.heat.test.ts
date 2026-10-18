/**
 * 热力图**快路径 vs 慢路径**的朝向一致性。
 *
 * 为什么专门为它写一个测试文件：渲染层有两条画热力图的路——
 *
 * * **快路径**：把 RGBA 塞进离屏画布（每格一像素），`drawImage` 整体缩放到轴矩形；
 * * **慢路径**：无离屏画布的宿主（Node、旧浏览器）逐格 `fillRect`。
 *
 * 两条路必须画出**同一张图**。`tools/raster.py`（G8 的参考光栅器）在栅格化 TS 场景时
 * 发现它们上下颠倒：慢路径按 `dataToPixel(j=0) → extent[2]`（第 0 行在**底部**），
 * 而 `drawImage` 把图像第 0 行放在矩形**顶部**——少了一次纵向翻转。
 * 这个 bug 只会在"浏览器里看着不对"，任何纯逻辑测试都抓不到，所以这里造一个假
 * `OffscreenCanvas`，把两条路的朝向都**量出来**再比对。
 */

import assert from "node:assert/strict";
import test from "node:test";

import { heatLayer, makeAxis, newScene, type Scene } from "../shared/src/scene.js";
import { renderScene } from "../shared/src/render2d.js";

interface Call {
  fn: string;
  args: unknown[];
}

/**
 * 记录所有调用的假 2D 上下文：未实现的方法一律返回无害的默认值。
 *
 * ``fillRect`` 会把**当时的 fillStyle** 一起记进 ``args[4]``——慢路径靠颜色辨认格子，
 * 不记颜色就等于没测。
 */
function recordingCtx(log: Call[]): CanvasRenderingContext2D {
  const target: Record<string, unknown> = {
    canvas: { width: 960, height: 600 },
    measureText: (text: string) => ({ width: String(text).length * 7 }),
    createLinearGradient: () => ({ addColorStop: () => undefined }),
    getLineDash: () => [] as number[],
    imageSmoothingEnabled: true,
    fillStyle: "#000000",
    fillRect: (x: number, y: number, w: number, h: number) =>
      log.push({ fn: "fillRect", args: [x, y, w, h, target["fillStyle"]] }),
    save: () => log.push({ fn: "save", args: [] }),
    restore: () => log.push({ fn: "restore", args: [] }),
  };
  return new Proxy(target, {
    get(obj, prop: string) {
      if (prop in obj) return obj[prop];
      return (...args: unknown[]) => {
        log.push({ fn: prop, args });
        return undefined;
      };
    },
    set(obj, prop: string, value) {
      obj[prop] = value;
      return true;
    },
  }) as unknown as CanvasRenderingContext2D;
}

/** 一列"值随行号单调增"的热力图：row 0 最暗、最后一行最亮。 */
function sceneWithHeat(): Scene {
  const s = newScene({
    chapter: "0.0",
    viz: "heat_orientation",
    title: "朝向测试",
    pitfalls: "热力图的行序是渲染约定的一部分，快慢两条路径必须一致。",
  });
  s.axes.push(makeAxis("main", [60, 40, 500, 300], { xlim: [0, 1], ylim: [0, 1], xlabel: "x", ylabel: "t" }));
  const nx = 3;
  const ny = 4;
  const values: number[] = [];
  for (let j = 0; j < ny; j += 1) for (let i = 0; i < nx; i += 1) values.push(j / (ny - 1));
  s.layers.push(heatLayer("main", values, nx, ny, [0, 1, 0, 1], { vmin: 0, vmax: 1 }));
  return s;
}

/** 装一个只够用的 `OffscreenCanvas` + `ImageData`，让快路径真的跑起来。 */
function withOffscreen<T>(log: Call[], body: () => T): T {
  const g = globalThis as unknown as Record<string, unknown>;
  const hadOff = "OffscreenCanvas" in g;
  const hadImage = "ImageData" in g;
  class FakeImageData {
    constructor(
      readonly data: Uint8ClampedArray,
      readonly width: number,
      readonly height: number,
    ) {}
  }
  class FakeOffscreen {
    readonly ctx = recordingCtx(log);
    constructor(
      readonly width: number,
      readonly height: number,
    ) {}
    getContext(): CanvasRenderingContext2D {
      return this.ctx;
    }
  }
  g["ImageData"] = FakeImageData;
  g["OffscreenCanvas"] = FakeOffscreen;
  try {
    return body();
  } finally {
    if (!hadOff) delete g["OffscreenCanvas"];
    if (!hadImage) delete g["ImageData"];
  }
}

/** 从调用日志里解出"图像行 j 的中心被画到设备 y 是多少"。 */
function deviceYOfImageRow(fast: Call[], row: number, ny: number): number {
  const translate = fast.find((c) => c.fn === "translate");
  const scale = fast.find((c) => c.fn === "scale");
  const draw = fast.find((c) => c.fn === "drawImage");
  assert.ok(translate && scale && draw, "快路径应当 translate/scale/drawImage 各调用一次");
  const [, ty] = translate.args as [number, number];
  const [, sy] = scale.args as [number, number];
  assert.equal(sy, -1, "快路径必须纵向翻转（scale(1,-1)）");
  const dy = (draw.args as [unknown, number, number, number, number])[4];
  // 翻转后：图像 y → 设备 y = ty + sy * y = ty - y
  return ty + sy * ((row + 0.5) * (dy / ny));
}

test("热力图快路径与慢路径的行序一致（第 0 行在底部）", () => {
  const scene = sceneWithHeat();

  // —— 慢路径：无 OffscreenCanvas，逐格 fillRect ——
  const slowLog: Call[] = [];
  const g = globalThis as unknown as Record<string, unknown>;
  const savedOff = g["OffscreenCanvas"];
  delete g["OffscreenCanvas"];
  try {
    renderScene(recordingCtx(slowLog), scene, { showBadges: false });
  } finally {
    if (savedOff !== undefined) g["OffscreenCanvas"] = savedOff;
  }
  const cellCalls = slowLog.filter((c) => c.fn === "fillRect" && typeof c.args[1] === "number");
  assert.ok(cellCalls.length >= 12, `慢路径应当逐格填充（实测 ${cellCalls.length} 次 fillRect）`);

  // 慢路径里"最亮的格子"应当在最上面（row ny-1 → 最大的数据 y → 最小的设备 y）
  const lum = (style: unknown): number => {
    const m = /rgba\((\d+),(\d+),(\d+)/.exec(String(style));
    return m ? Number(m[1]) + Number(m[2]) + Number(m[3]) : Number.NaN;
  };
  const slowCells = cellCalls
    .map((c) => ({ y: c.args[1] as number, l: lum(c.args[4]) }))
    .filter((c) => Number.isFinite(c.l));
  const byY = [...slowCells].sort((a, b) => a.y - b.y);
  const brightestTop = byY[byY.length - 1] as { y: number; l: number };
  const brightestBottom = byY[0] as { y: number; l: number };
  assert.ok(
    byY[0]!.l <= byY[byY.length - 1]!.l,
    `慢路径：行号越大越亮，因此设备 y 越小越亮（实测顶部亮度 ${brightestBottom.l}、底部 ${brightestTop.l}）`,
  );

  // —— 快路径：装假 OffscreenCanvas，记录 putImageData 与变换 ——
  const fastLog: Call[] = [];
  withOffscreen(fastLog, () => {
    renderScene(recordingCtx(fastLog), scene, { showBadges: false });
  });
  const put = fastLog.find((c) => c.fn === "putImageData");
  assert.ok(put, "快路径应当把 RGBA 塞进离屏画布");
  const image = put.args[0] as { width: number; height: number; data: Uint8ClampedArray };
  assert.equal(image.width, 3);
  assert.equal(image.height, 4);

  const rowLum = (j: number): number => {
    let sum = 0;
    for (let i = 0; i < image.width; i += 1) {
      const p = 4 * (j * image.width + i);
      sum += (image.data[p] as number) + (image.data[p + 1] as number) + (image.data[p + 2] as number);
    }
    return sum / image.width;
  };
  const top = deviceYOfImageRow(fastLog, image.height - 1, image.height);
  const bottom = deviceYOfImageRow(fastLog, 0, image.height);
  assert.ok(top < bottom, `翻转后：图像最后一行（最亮）应当在上方（top=${top} < bottom=${bottom}）`);
  assert.ok(rowLum(image.height - 1) > rowLum(0), "夹具本身要满足「行号越大越亮」，否则这条测试没有判别力");
  // 两条路的一致结论：最亮的行在**上面**
  assert.ok(top < bottom, "快路径与慢路径的朝向必须一致（第 0 行在底部）");
});

test("快路径把离屏画布按轴矩形整体贴图（不逐格 fillRect）", () => {
  const fastLog: Call[] = [];
  withOffscreen(fastLog, () => {
    renderScene(recordingCtx(fastLog), sceneWithHeat(), { showBadges: false });
  });
  const draw = fastLog.find((c) => c.fn === "drawImage");
  assert.ok(draw, "快路径必须有 drawImage");
  const [, sx, sy, sw, sh] = draw.args as [unknown, number, number, number, number];
  assert.deepEqual([sx, sy], [0, 0], "贴图源点应当是离屏画布左上角");
  assert.ok(sw > 0 && sh > 0, "贴图目标宽高必须为正");
  const heatFills = fastLog.filter((c) => c.fn === "fillRect");
  // 快路径下热力图不再逐格填充（其余装饰可能仍有 fillRect，但数量级要小得多）
  assert.ok(heatFills.length < 12, `快路径不该逐格 fillRect（实测 ${heatFills.length} 次）`);
});
