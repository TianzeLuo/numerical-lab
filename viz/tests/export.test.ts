/**
 * 导出的测试：PNG 字节的两条路（`convertToBlob` / `toDataURL`）、Scene JSON 的稳定性、
 * 下载文件名的确定性。
 *
 * 无浏览器环境拿不到真 canvas，所以这里传的是**假 canvas 对象**——`canvasPngBytes` 只调用
 * 那两个方法，这正是它被写成"依赖最小接口"的原因。真正容易出事的是**退化路径**：
 * `toDataURL` 在画布被污染或尺寸为 0 时会返回 `data:,` 或半截数据，所以 base64 头与 PNG
 * 魔术都必须校验，测试逐条钉住。
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  PNG_DATA_URL_PREFIX,
  PNG_SIGNATURE,
  canvasPngBytes,
  downloadName,
  isPngBytes,
  parsePngDataUrl,
  sceneToJsonText,
} from "../shared/src/export.js";
import type { CanvasPngSource } from "../shared/src/export.js";
import { paramsHash } from "../shared/src/framesource.js";
import { base64Encode } from "../shared/src/share.js";
import { newScene, makeAxis, lineLayer, series, sourceBadge } from "../shared/src/scene.js";
import type { Scene } from "../shared/src/scene.js";

/** 一段"看起来像 PNG"的字节（真 PNG 头 + 任意负载）。 */
const PNG_BYTES = Uint8Array.from([...PNG_SIGNATURE, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52, 1, 2, 3]);
const PNG_DATA_URL = PNG_DATA_URL_PREFIX + base64Encode(PNG_BYTES);

function fakeScene(): Scene {
  const scene = newScene({ chapter: "0.0", viz: "demo", title: "演示", pitfalls: "自适应纵轴会把发散画成平的。" });
  scene.axes.push(makeAxis("main", [60, 40, 400, 300], { xlim: [0, 1], ylim: [-1, 1], xlabel: "x", ylabel: "u" }));
  scene.layers.push(lineLayer("main", series([0, 0.5, 1], [0, 1, 0], "#333")));
  scene.badges.push(sourceBadge(scene.dataSource));
  return scene;
}

// --------------------------------------------------------------------------- //
// canvasPngBytes
// --------------------------------------------------------------------------- //

test("canvasPngBytes：优先 convertToBlob，且不碰 toDataURL", async () => {
  let toDataUrlCalls = 0;
  const canvas: CanvasPngSource = {
    convertToBlob: async (opts) => {
      assert.equal(opts?.type, "image/png");
      return new Blob([PNG_BYTES], { type: "image/png" });
    },
    toDataURL: () => {
      toDataUrlCalls++;
      return PNG_DATA_URL;
    },
  };
  const bytes = await canvasPngBytes(canvas as unknown as HTMLCanvasElement);
  assert.deepEqual(bytes, PNG_BYTES);
  assert.equal(toDataUrlCalls, 0, "首选路径成功时不该走退化路径");
});

test("canvasPngBytes：convertToBlob 抛错 → 退化到 toDataURL", async () => {
  const canvas: CanvasPngSource = {
    convertToBlob: async () => {
      throw new Error("假：GPU 上下文丢了");
    },
    toDataURL: () => PNG_DATA_URL,
  };
  assert.deepEqual(await canvasPngBytes(canvas as unknown as HTMLCanvasElement), PNG_BYTES);
});

test("canvasPngBytes：convertToBlob 返回空数据/非 PNG → 退化到 toDataURL", async () => {
  const empty = await canvasPngBytes({
    convertToBlob: async () => new Blob([]),
    toDataURL: () => PNG_DATA_URL,
  } as unknown as HTMLCanvasElement);
  assert.deepEqual(empty, PNG_BYTES);

  const junk = await canvasPngBytes({
    convertToBlob: async () => new Blob([Uint8Array.from([1, 2, 3])]),
    toDataURL: () => PNG_DATA_URL,
  } as unknown as HTMLCanvasElement);
  assert.deepEqual(junk, PNG_BYTES);
});

test("canvasPngBytes：没有 convertToBlob 时直接用 toDataURL", async () => {
  const bytes = await canvasPngBytes({ toDataURL: () => PNG_DATA_URL } as unknown as HTMLCanvasElement);
  assert.deepEqual(bytes, PNG_BYTES);
});

test("canvasPngBytes：坏 data URL（空、MIME 不对、base64 截断、非 PNG）都必须报错且说清原因", async () => {
  const cases: Array<[string, RegExp]> = [
    ["data:,", /data URL 不可用/],
    ["data:image/jpeg;base64,AAAA", /data URL 不可用/],
    [`${PNG_DATA_URL_PREFIX}!!!not-base64!!!`, /data URL 不可用/],
    [PNG_DATA_URL_PREFIX, /data URL 不可用/],
    [PNG_DATA_URL_PREFIX + base64Encode(Uint8Array.from([1, 2, 3, 4, 5, 6, 7, 8, 9])), /data URL 不可用/],
  ];
  for (const [url, pattern] of cases) {
    await assert.rejects(
      () => canvasPngBytes({ toDataURL: () => url } as unknown as HTMLCanvasElement),
      pattern,
      `应拒绝 ${JSON.stringify(url.slice(0, 32))}`,
    );
  }
  // 报错消息里要能看出"头是什么"，方便排查
  await assert.rejects(
    () => canvasPngBytes({ toDataURL: () => "data:," } as unknown as HTMLCanvasElement),
    /"data:,"/,
  );
});

test("canvasPngBytes：toDataURL 抛错 / 两个方法都没有 → 报错", async () => {
  await assert.rejects(
    () =>
      canvasPngBytes({
        convertToBlob: async () => {
          throw new Error("没有 GPU");
        },
        toDataURL: () => {
          throw new Error("画布被污染");
        },
      } as unknown as HTMLCanvasElement),
    /画布被污染/,
  );
  await assert.rejects(() => canvasPngBytes({} as unknown as HTMLCanvasElement), /既没有 convertToBlob 也没有 toDataURL/);
});

test("parsePngDataUrl：只认规范的 PNG data URL", () => {
  assert.deepEqual(parsePngDataUrl(PNG_DATA_URL), PNG_BYTES);
  assert.deepEqual(parsePngDataUrl(`  ${PNG_DATA_URL}  `), PNG_BYTES, "两端空白可容忍");
  assert.deepEqual(parsePngDataUrl(PNG_DATA_URL_PREFIX.toUpperCase() + base64Encode(PNG_BYTES)), PNG_BYTES, "MIME 大小写不敏感");
  assert.equal(parsePngDataUrl("data:image/png;base64,"), null);
  assert.equal(parsePngDataUrl("data:image/png,abc"), null);
  assert.equal(parsePngDataUrl(123), null);
  assert.equal(parsePngDataUrl(null), null);
  assert.equal(isPngBytes(PNG_BYTES), true);
  assert.equal(isPngBytes(Uint8Array.from([0x89, 0x50])), false);
});

// --------------------------------------------------------------------------- //
// sceneToJsonText
// --------------------------------------------------------------------------- //

test("sceneToJsonText：确定性（键序无关、TypedArray 归一、两次调用逐字节相同）", () => {
  const scene = fakeScene();
  const a = sceneToJsonText(scene);
  const b = sceneToJsonText(scene);
  assert.equal(a, b);
  // 曲线里的 TypedArray 与普通数组必须序列化成同一串（跨平台基线靠这条）
  const typed = fakeScene();
  (typed.layers[0] as { series: unknown }).series = [
    { x: Float64Array.from([0, 0.5, 1]), y: Float64Array.from([0, 1, 0]), color: "#333", width: 1.6 },
  ];
  assert.equal(sceneToJsonText(typed), a, "TypedArray 与普通数组的序列化必须一致");
  assert.match(a, /"viz": "demo"/);
});

// --------------------------------------------------------------------------- //
// downloadName
// --------------------------------------------------------------------------- //

test("downloadName：确定性、含参数指纹与 %.6g 时刻", () => {
  const name = downloadName("0.0", "heat_ftcs", { nx: 41, r: 0.4, t_end: 0.05 }, 0.15);
  assert.equal(name, "nl-0.0-heat_ftcs-8e44fdf2f53dceb1-t0.15.png");
  assert.equal(downloadName("0.0", "heat_ftcs", { r: 0.4, t_end: 0.05, nx: 41 }, 0.15), name, "参数键序无关");
  assert.match(name, /-t0\.15\.png$/);
  // 浮点噪声不该产生第二个文件名（0.1+0.2 与 0.3 的 %.6g 写法相同）
  assert.equal(downloadName("0.0", "demo", { r: 0.1 }, 0.1 + 0.2), downloadName("0.0", "demo", { r: 0.1 }, 0.3));
  assert.ok(downloadName("0.0", "demo", { r: 0.1 }, 0.1 + 0.2).endsWith("-t0.3.png"));
  // 参数指纹与 L3 静态帧文件名用的是同一个算法
  assert.ok(name.includes(paramsHash("0.0", "heat_ftcs", { nx: 41, r: 0.4, t_end: 0.05 })));
});

test("downloadName：参数不同 → 指纹不同；扩展名可换；非法字符被清掉", () => {
  const a = downloadName("0.0", "demo", { r: 0.4 }, 0);
  const b = downloadName("0.0", "demo", { r: 0.5 }, 0);
  assert.notEqual(a, b, "参数变了名字必须变，否则会互相覆盖");
  assert.ok(a.endsWith(".png"));
  assert.ok(a.includes("-t0.png"));
  assert.equal(downloadName("0.0", "demo", { r: 0.4 }, 0, "gif").endsWith(".gif"), true);
  const slugged = downloadName("1/2", "a b", { r: 0.4 }, 0);
  assert.equal(slugged, `nl-1-2-a-b-${paramsHash("1/2", "a b", { r: 0.4 })}-t0.png`);
  assert.ok(!slugged.includes("/") && !slugged.includes(" "), "文件名里不能出现路径分隔符与空格");
});
