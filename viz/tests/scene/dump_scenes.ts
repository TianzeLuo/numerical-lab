/**
 * 把每章每个可视化的 Scene **用真前端模型**算出来并落盘成 JSON。
 *
 * 为什么要有这个脚本：视觉回归如果只栅格化"Python 侧拼出来的标准场景"，那它
 * 只能证明渲染器没坏，证明不了**模型**没坏。这里走的是用户真正会看到的那条路：
 *
 *     data/frames/<章>/<viz>/*.nlf  →  decodeNlf  →  章节 model（Frame→Scene）  →  dist/scenes/**.json
 *
 * 之后 `tools/visual_regression.py` 只负责把这份 JSON 栅格化并与基线比对。
 * 场景 JSON 是"契约"（进 git 的只有基线 PNG），文件名必须稳定：
 *
 *     dist/scenes/<章>/<viz>__<参数指纹 8 位>__t<时刻>.json
 *
 * 用法：
 *     node dist/viz/tests/scene/dump_scenes.js [--frames data/frames] [--out dist/scenes]
 *                                              [--width 960] [--height 600] [--json]
 *
 * 前置条件：先 `python3 tools/precompute.py`（或 `make precompute`）生成帧。
 * 没有帧的章节/可视化会被跳过并计入 `skipped`，不会静默当成"通过"。
 */

import { mkdirSync, readdirSync, readFileSync, writeFileSync, existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";

import { chapterModels } from "../../chapters/index.gen.js";
import { decodeNlf } from "../../shared/src/nlf.js";
import type { Frame } from "../../shared/src/frame.js";
import { sceneToJson } from "../../shared/src/scene.js";
import { validateScene } from "../../shared/src/scene.js";
import type { FrameHistoryEntry, ModelOptions } from "../../shared/src/model.js";

interface Args {
  frames: string;
  out: string;
  width: number;
  height: number;
  json: boolean;
  maxHistory: number;
  chapter: string;
}

function parseArgs(argv: string[]): Args {
  const get = (name: string, dflt: string): string => {
    const i = argv.indexOf(`--${name}`);
    return i >= 0 && i + 1 < argv.length ? (argv[i + 1] as string) : dflt;
  };
  return {
    frames: get("frames", "data/frames"),
    out: get("out", "dist/scenes"),
    width: Number.parseInt(get("width", "960"), 10),
    height: Number.parseInt(get("height", "600"), 10),
    json: argv.includes("--json"),
    maxHistory: Number.parseInt(get("max-history", "60"), 10),
    chapter: get("chapter", ""),
  };
}

/** FNV-1a 32 位：稳定、短、无依赖（只用于文件名，不用于任何安全用途）。 */
function fnv1a(text: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h.toString(16).padStart(8, "0");
}

/** 时刻写进文件名：整数不带小数点，小数最多 6 位（与 Python 侧 `%g` 对齐）。 */
function timeTag(t: number): string {
  if (!Number.isFinite(t)) return "tNaN";
  return "t" + (Number.isInteger(t) ? String(t) : t.toFixed(6).replace(/0+$/, "").replace(/\.$/, ""));
}

interface Dumped {
  chapter: string;
  viz: string;
  file: string;
  width: number;
  height: number;
  t: number;
  layers: number;
  problems: string[];
}

function main(): number {
  const args = parseArgs(process.argv.slice(2));
  const framesDir = resolve(args.frames);
  const outDir = resolve(args.out);
  const dumped: Dumped[] = [];
  const skipped: { chapter: string; viz: string; reason: string }[] = [];
  const problems: string[] = [];

  // `--chapter 1.2`：只转储指定章节。并发开发时别的章节随时可能处于半成品状态，
  // 而"我只想看我这一章的图"不该被别人的中间态拖成一次失败（与 tools/visual_regression.py、
  // tools/precompute.py 的 `--chapter` 同一条规矩）。
  const only = args.chapter ? new Set(args.chapter.split(",").map((c) => c.trim()).filter(Boolean)) : null;
  for (const [chapter, bundle] of Object.entries(chapterModels)) {
    if (only && !only.has(chapter)) continue;
    for (const viz of bundle.manifest.vizzes) {
      const model = bundle.models[viz.id];
      if (!model) {
        skipped.push({ chapter, viz: viz.id, reason: "章节没有为这个 viz 提供 model" });
        continue;
      }
      const vizDir = join(framesDir, chapter, viz.id);
      if (!existsSync(vizDir)) {
        skipped.push({ chapter, viz: viz.id, reason: `没有帧目录 ${vizDir}（先跑 tools/precompute.py）` });
        continue;
      }
      const files = readdirSync(vizDir)
        .filter((f) => f.endsWith(".nlf"))
        .sort();
      if (files.length === 0) {
        skipped.push({ chapter, viz: viz.id, reason: "帧目录是空的" });
        continue;
      }
      const history: FrameHistoryEntry[] = [];
      for (const file of files) {
        const bytes = new Uint8Array(readFileSync(join(vizDir, file)));
        let frame: Frame;
        try {
          frame = decodeNlf(bytes);
        } catch (err) {
          problems.push(`${chapter}/${viz.id}/${file}: 解码失败 ${(err as Error).message}`);
          continue;
        }
        const t = Number(frame.meta["t"] ?? 0);
        const paramsJson = String(frame.strings["params_json"] ?? "{}");
        const opts: ModelOptions = {
          width: args.width,
          height: args.height,
          t,
          params: JSON.parse(paramsJson) as ModelOptions["params"],
          history: history.slice(-args.maxHistory),
        };
        let scene;
        try {
          scene = model(frame, opts);
        } catch (err) {
          problems.push(`${chapter}/${viz.id}/${file}: model 抛错 ${(err as Error).name}: ${(err as Error).message}`);
          continue;
        }
        const sceneProblems = validateScene(scene);
        if (sceneProblems.length > 0) {
          problems.push(`${chapter}/${viz.id}/${file}: Scene 不合规 → ${sceneProblems.join("；")}`);
          continue;
        }
        const name = `${viz.id}__${fnv1a(`${chapter}/${viz.id}/${paramsJson}`)}__${timeTag(t)}.json`;
        const target = join(outDir, chapter, name);
        mkdirSync(dirname(target), { recursive: true });
        // sceneToJson 做 12 位有效数字的稳定序列化：同一帧两次必须逐字节相同
        writeFileSync(target, sceneToJson(scene) + "\n", "utf8");
        dumped.push({
          chapter,
          viz: viz.id,
          file: target.slice(outDir.length + 1),
          width: args.width,
          height: args.height,
          t,
          layers: scene.layers.length,
          problems: [],
        });
        history.push({ t, frame });
        if (history.length > args.maxHistory) history.shift();
      }
    }
  }

  const summary = {
    ok: problems.length === 0,
    dumped: dumped.length,
    skipped: skipped.length,
    problems,
    skippedDetail: skipped,
    scenes: dumped,
  };
  const reportPath = resolve("dist/reports/scenes.json");
  mkdirSync(dirname(reportPath), { recursive: true });
  writeFileSync(reportPath, JSON.stringify(summary, null, 2) + "\n", "utf8");

  if (args.json) {
    process.stdout.write(JSON.stringify(summary) + "\n");
  } else {
    process.stdout.write(`场景转储：${dumped.length} 个 → ${args.out}\n`);
    for (const s of skipped) process.stdout.write(`  跳过 ${s.chapter}/${s.viz}：${s.reason}\n`);
    for (const p of problems) process.stdout.write(`  ❌ ${p}\n`);
    process.stdout.write(`报告：dist/reports/scenes.json\n`);
  }
  return problems.length === 0 ? 0 : 1;
}

process.exitCode = main();
