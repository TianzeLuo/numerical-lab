/**
 * 导出：canvas → PNG 字节、Scene → 稳定 JSON、下载文件名。
 *
 * 三条要求，各有对应的测试：
 *
 * 1. **`canvasPngBytes` 优先 `convertToBlob`**（无损、不经过 base64 字符串、大图不炸内存），
 *    失败或不存在时退化到 `toDataURL`，并且**校验 base64 头**——浏览器在某些状态下会返回
 *    `data:,`（空）或 `data:image/png;base64,` 后跟半个图，直接 `atob` 会得到一个"能解码但
 *    不是 PNG"的字节串，渲染出来是一张空白图，比报错难查得多。
 * 2. **`sceneToJsonText` 复用 `scene.ts::sceneToJson`**：视觉回归基线
 *    （`dist/visual/scenes/*.json`，PROTOCOLS §9）用的就是那个序列化器。导出和基线必须是
 *    同一份字节，否则"导出的图"与"回归基线"会在某次重构后悄悄分叉。
 * 3. **`downloadName` 确定性**：章节 + viz + 参数指纹（与 L3 静态帧同名算法）+ `%.6g` 时刻。
 *    同一屏导出两次得到同一个文件名；换个参数必然换名，不会互相覆盖。
 *
 * 本模块不碰 DOM 全局：`canvasPngBytes` 只调用传入对象上的两个方法，因此测试里
 * 传一个假 canvas 就能跑（真 canvas 在无浏览器环境下拿不到）。
 */

import { formatPyG, paramsHash } from "./framesource.js";
import type { ParamValues } from "./share.js";
import { base64Decode } from "./share.js";
import { sceneToJson } from "./scene.js";
import type { Scene } from "./scene.js";

/** 只用到这两个方法的"canvas"（`HTMLCanvasElement` 结构上满足它）。 */
export interface CanvasPngSource {
  convertToBlob?: (options?: { type?: string }) => Promise<Blob>;
  toDataURL?: (type?: string) => string;
}

/** PNG 文件头（`\x89PNG\r\n\x1a\n`）。 */
export const PNG_SIGNATURE: Uint8Array = Uint8Array.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

/** `toDataURL("image/png")` 的 MIME 头。 */
export const PNG_DATA_URL_PREFIX = "data:image/png;base64,";

/** 字节串是否以 PNG 魔术开头。 */
export function isPngBytes(bytes: Uint8Array): boolean {
  if (bytes.byteLength < PNG_SIGNATURE.byteLength) return false;
  for (let i = 0; i < PNG_SIGNATURE.byteLength; i++) {
    if (bytes[i] !== PNG_SIGNATURE[i]) return false;
  }
  return true;
}

/**
 * 解析 PNG data URL。
 *
 * 返回 `null`（而不是抛）的三种情况：MIME 头不是 `data:image/png;base64,`、
 * base64 段缺失或非法、解出来的字节不是 PNG。调用方据此给出"为什么不能导出"。
 */
export function parsePngDataUrl(text: unknown): Uint8Array | null {
  if (typeof text !== "string") return null;
  const trimmed = text.trim();
  if (trimmed.length <= PNG_DATA_URL_PREFIX.length) return null;
  if (trimmed.slice(0, PNG_DATA_URL_PREFIX.length).toLowerCase() !== PNG_DATA_URL_PREFIX) return null;
  const payload = trimmed.slice(PNG_DATA_URL_PREFIX.length);
  if (payload === "") return null;
  const bytes = base64Decode(payload);
  if (bytes === null || !isPngBytes(bytes)) return null;
  return bytes;
}

/**
 * 把 canvas 导出成 PNG 字节。
 *
 * 顺序：`convertToBlob({type:"image/png"})` → `toDataURL("image/png")`；两条路都走不通时抛错，
 * 错误消息里带上两条路各自的失败原因（UI 可以直接显示给用户）。
 */
export async function canvasPngBytes(canvas: HTMLCanvasElement | CanvasPngSource): Promise<Uint8Array> {
  // 统一成"只要求这两个方法"的视图：某些 TS lib 版本的 HTMLCanvasElement 还没有
  // convertToBlob（浏览器早就有），直接访问联合类型会报错。
  const source: CanvasPngSource = canvas;
  const reasons: string[] = [];

  if (typeof source.convertToBlob === "function") {
    try {
      const blob = await source.convertToBlob({ type: "image/png" });
      const bytes = new Uint8Array(await blob.arrayBuffer());
      if (bytes.byteLength === 0) throw new Error("返回 0 字节");
      if (!isPngBytes(bytes)) throw new Error("返回的字节不是 PNG");
      return bytes;
    } catch (err) {
      reasons.push(`convertToBlob：${err instanceof Error ? err.message : String(err)}`);
    }
  } else {
    reasons.push("convertToBlob：不可用");
  }

  if (typeof canvas.toDataURL === "function") {
    let url: string;
    try {
      url = canvas.toDataURL("image/png");
    } catch (err) {
      throw new Error(
        `canvas 导出 PNG 失败（toDataURL 抛错：${err instanceof Error ? err.message : String(err)}；${reasons.join("；")}）`,
      );
    }
    const bytes = parsePngDataUrl(url);
    if (bytes !== null) return bytes;
    const head = url.slice(0, 32);
    throw new Error(`canvas 导出 PNG 失败：toDataURL 返回的 data URL 不可用（开头是 ${JSON.stringify(head)}）；${reasons.join("；")}`);
  }

  throw new Error(`canvas 导出 PNG 失败：既没有 convertToBlob 也没有 toDataURL（${reasons.join("；")}）`);
}

/**
 * Scene → 稳定 JSON 文本（键排序、数值 12 位有效数字截断）。
 *
 * 直接复用 `sceneToJson`：这是 `dist/visual/scenes/*.json` 视觉基线的同一份序列化，
 * 导出与回归必须是同一个字节流（见文件头第 2 条）。
 */
export function sceneToJsonText(scene: Scene): string {
  return sceneToJson(scene);
}

/** 文件名里只保留 `[A-Za-z0-9._-]`，其余压成 `-`（章节号里的 `.` 保留）。 */
function slug(text: string): string {
  return text.replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^-+|-+$/g, "");
}

/**
 * 下载文件名：`nl-<chapter>-<viz>-<参数指纹>-t<%.6g>.<ext>`。
 *
 * 参数指纹 = `paramsHash`（与 L3 静态帧文件名同一个算法），所以"图上的参数"和"文件名的参数"
 * 永远对得上；时刻用 `%.6g`，`0.1+0.2` 这种浮点噪声不会写出两种文件名。
 */
export function downloadName(
  chapter: string,
  viz: string,
  params: ParamValues,
  t: number,
  ext = "png",
): string {
  const fingerprint = paramsHash(chapter, viz, params);
  return `nl-${slug(chapter)}-${slug(viz)}-${fingerprint}-t${formatPyG(t, 6)}.${slug(ext) || "bin"}`;
}
