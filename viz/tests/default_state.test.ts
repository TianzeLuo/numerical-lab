/**
 * 默认状态的“第一眼”审计：每个可视化的初始时刻、默认参数下，
 * 画面不能亮红色“数据异常/溢出”角标。
 *
 * 固定视窗裁剪、地板点、掩码等显示约定可以用中性的“显示范围”提示，
 * 但初始画面本身必须是可读的。这个测试直接读 `data/frames` 的预计算帧，
 * 与用户打开页面时走的是同一条 Frame → Scene 路径。
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { readdirSync, readFileSync, existsSync } from "node:fs";
import { join } from "node:path";

import { chapterIds, loadChapter } from "../chapters/index.gen.js";
import { decodeNlf } from "../shared/src/nlf.js";
import { formatPyG } from "../shared/src/framesource.js";
import type { Frame } from "../shared/src/frame.js";
import type { SceneModel } from "../shared/src/model.js";

const ROOT = process.cwd();
const FRAMES = join(ROOT, "data/frames");

function frameFor(chapter: string, viz: string, t: number): Frame {
  const dir = join(FRAMES, chapter, viz);
  if (!existsSync(dir)) throw new Error(`缺帧目录 ${dir}；先运行 tools/precompute.py --n 6`);
  const files = readdirSync(dir);
  const exact = files.find((name) => name.endsWith(`_t${formatPyG(t, 6)}.nlf`));
  const fallback = t === 0 ? undefined : files.find((name) => name.endsWith("_t0.nlf"));
  const name = exact ?? fallback;
  if (name === undefined) throw new Error(`${chapter}/${viz} 没有 t=${t} 的预计算帧（${files.slice(0, 4).join(", ")}…）`);
  return decodeNlf(new Uint8Array(readFileSync(join(dir, name))));
}

test("默认状态：没有红色数据异常/意外溢出", async () => {
  const bad: string[] = [];
  for (const chapter of chapterIds) {
    const bundle = await loadChapter(chapter);
    if (bundle === undefined) continue;
    for (const viz of bundle.manifest.vizzes) {
      const model = bundle.models[viz.id] as SceneModel | undefined;
      if (model === undefined) continue;
      const t = viz.time.mode === "none" ? 0 : (viz.time.initial ?? 0);
      const frame = frameFor(chapter, viz.id, t);
      const params = Object.fromEntries(viz.params.map((p) => [p.name, p.default]));
      const scene = model(frame, { width: 960, height: 600, t, params, theme: "light" });
      for (const badge of scene.badges) {
        if (badge.kind !== "overflow") continue;
        const text = badge.detail ?? badge.text;
        const dataLike = badge.tone === "danger" || /非有限|NaN|∞|无有限解|未定义/.test(text);
        if (dataLike) bad.push(`${chapter}/${viz.id}: ${text.slice(0, 140)}`);
      }
    }
  }
  assert.deepEqual(bad, [], "默认初始状态出现数据异常/溢出");
});
