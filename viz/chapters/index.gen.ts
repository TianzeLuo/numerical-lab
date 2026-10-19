// 本文件由 tools/gen_manifests.py 从章节模块的 MANIFEST 生成，请勿手改。

import type { ChapterManifestJson } from "../shared/src/manifest.js";
import type { SceneModel } from "../shared/src/model.js";

/** 一章的前端装配：manifest（契约）+ models（viz id → Frame→Scene） */
export interface ChapterModels {
  manifest: ChapterManifestJson;
  models: Record<string, SceneModel>;
}

/** 已经实现可视化的章节 id（未实现的不在表里；判断「能不能点」看它，不需要加载 model） */
export const chapterIds: string[] = ["0.0", "1.1", "1.2", "1.3", "1.4", "1.5", "2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "3.1", "3.2", "3.3", "3.4", "3.5", "4.1", "4.2", "4.3", "4.4", "4.5", "4.6", "5.1", "5.2", "5.3", "5.4", "6.1", "6.2"];

/** 章节 id → 加载器（动态 import：坏掉的兄弟章节不会拖垮整包） */
const loaders: Record<string, () => Promise<ChapterModels>> = {
  "0.0": () => import("./0.0/index.js"),
  "1.1": () => import("./1.1/index.js"),
  "1.2": () => import("./1.2/index.js"),
  "1.3": () => import("./1.3/index.js"),
  "1.4": () => import("./1.4/index.js"),
  "1.5": () => import("./1.5/index.js"),
  "2.1": () => import("./2.1/index.js"),
  "2.2": () => import("./2.2/index.js"),
  "2.3": () => import("./2.3/index.js"),
  "2.4": () => import("./2.4/index.js"),
  "2.5": () => import("./2.5/index.js"),
  "2.6": () => import("./2.6/index.js"),
  "3.1": () => import("./3.1/index.js"),
  "3.2": () => import("./3.2/index.js"),
  "3.3": () => import("./3.3/index.js"),
  "3.4": () => import("./3.4/index.js"),
  "3.5": () => import("./3.5/index.js"),
  "4.1": () => import("./4.1/index.js"),
  "4.2": () => import("./4.2/index.js"),
  "4.3": () => import("./4.3/index.js"),
  "4.4": () => import("./4.4/index.js"),
  "4.5": () => import("./4.5/index.js"),
  "4.6": () => import("./4.6/index.js"),
  "5.1": () => import("./5.1/index.js"),
  "5.2": () => import("./5.2/index.js"),
  "5.3": () => import("./5.3/index.js"),
  "5.4": () => import("./5.4/index.js"),
  "6.1": () => import("./6.1/index.js"),
  "6.2": () => import("./6.2/index.js"),
};

const cache = new Map<string, ChapterModels>();

/** 加载一章的前端装配；没有实现或加载失败时返回 undefined（并打印一行警告）。 */
export async function loadChapter(id: string): Promise<ChapterModels | undefined> {
  const hit = cache.get(id);
  if (hit) return hit;
  const load = loaders[id];
  if (!load) return undefined;
  try {
    const bundle = await load();
    cache.set(id, bundle);
    return bundle;
  } catch (err) {
    console.warn(`[chapters] 章节 ${id} 的前端装配加载失败：${(err as Error).message}`);
    return undefined;
  }
}

/** 按 id 升序加载所有已实现章节（失败的跳过，返回值只含成功的）。 */
export async function loadAllChapters(): Promise<Map<string, ChapterModels>> {
  const out = new Map<string, ChapterModels>();
  for (const id of chapterIds) {
    const bundle = await loadChapter(id);
    if (bundle) out.set(id, bundle);
  }
  return out;
}

export default loadChapter;
