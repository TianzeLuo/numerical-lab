/**
 * 站点外壳：目录（侧边栏）+ 路由（hash）+ 挂载可视化。
 *
 * 这一层只做三件事：**选章节、找 model、交给 `mountViz`**。所有可见的判断
 * （参数范围、临界值、陷阱文案、降级状态）都在共享层里，且都有测试。
 *
 * 路由形如 `#v1.<base64url(JSON)>`（见 `shared/src/share.ts`）。没有 hash 时
 * 落到 `data/manifests/index.json` 里第一章的第一个可视化。
 */

import { chapterIndex } from "../shared/src/chapters.gen.js";
import { chapterIds, loadChapter } from "../chapters/index.gen.js";
import type { ChapterIndexEntry } from "../shared/src/chapters.js";
import { groupByPart } from "../shared/src/chapters.js";
import { buildHash, decodeHash, type ShareState } from "../shared/src/share.js";
import { mountViz, type VizView } from "../shared/src/view.js";
import { validateManifestJson } from "../shared/src/manifest.js";
import type { ChapterManifestJson } from "../shared/src/manifest.js";

const DEFAULT_BASE = "";

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  props: Partial<HTMLElementTagNameMap[K]> = {},
  ...children: (Node | string)[]
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  Object.assign(node, props);
  node.append(...children);
  return node;
}

export interface SiteHandle {
  goto(state: ShareState): Promise<void>;
  readonly current: ShareState | null;
}

/** 找到第一个"有 model"的可视化（目录里可能有尚未实现的章节）。 */
async function firstImplemented(): Promise<ShareState | null> {
  for (const entry of chapterIndex) {
    const bundle = await loadChapter(entry.chapter);
    if (!bundle) continue;
    for (const viz of bundle.manifest.vizzes) {
      if (bundle.models[viz.id]) {
        return { chapter: entry.chapter, viz: viz.id, params: {}, t: 0, playing: false };
      }
    }
  }
  return null;
}

export async function startSite(root: HTMLElement, opts: { baseUrl?: string } = {}): Promise<SiteHandle> {
  const baseUrl = opts.baseUrl ?? DEFAULT_BASE;
  const nav = el("nav", { className: "nl-nav" });
  const main = el("main", { className: "nl-content" });
  root.append(el("header", { className: "nl-top" }, el("h1", {}, "数值方法实验室")), nav, main, el("footer", {
    className: "nl-foot",
  }, "每一章：推导（LaTeX）· 实现（numcore）· 可视化（帧→Scene→canvas）· 三类断言 · 防误导。"));

  let view: VizView | null = null;
  let current: ShareState | null = null;

  const renderNav = (): void => {
    nav.innerHTML = "";
    for (const group of groupByPart(chapterIndex as ChapterIndexEntry[])) {
      const box = el("section", { className: "nl-part" });
      box.append(el("h2", {}, group.title));
      const ul = el("ul");
      for (const entry of group.items) {
        // 只查 id 表（静态常量）：侧边栏不该为了知道"能不能点"而加载 12 个 model
        const implemented = chapterIds.includes(entry.chapter);
        const li = el("li");
        const a = el("a", {
          href: `#${entry.chapter}`,
          className: current?.chapter === entry.chapter ? "active" : "",
          title: entry.summary || entry.title,
        }, `${entry.chapter} ${entry.title}`);
        if (!implemented) a.classList.add("pending");
        a.addEventListener("click", (ev) => {
          ev.preventDefault();
          void (async () => {
          const bundle = await loadChapter(entry.chapter);
          const viz = bundle ? bundle.manifest.vizzes.find((v) => bundle.models[v.id]) : undefined;
          if (!viz) {
            main.innerHTML = "";
            main.append(
              el("div", { className: "nl-todo" },
                el("h2", {}, `${entry.chapter} ${entry.title}`),
                el("p", {}, `这一章的 ${entry.nViz} 个可视化还没实现（目录里先占位）。`),
                el("p", {}, entry.summary || "")),
            );
            return;
          }
          void goto({ chapter: entry.chapter, viz: viz.id, params: {}, t: 0, playing: false });
          })();
        });
        li.append(a);
        ul.append(li);
      }
      box.append(ul);
      nav.append(box);
    }
  };

  const goto = async (state: ShareState): Promise<void> => {
    const bundle = await loadChapter(state.chapter);
    if (!bundle) {
      main.innerHTML = "";
      main.append(el("div", { className: "nl-todo" }, el("h2", {}, "这一章还没有可视化"), el("p", {}, state.chapter)));
      return;
    }
    const problems = validateManifestJson(bundle.manifest);
    if (problems.length > 0) {
      main.innerHTML = "";
      main.append(
        el("div", { className: "nl-todo" },
          el("h2", {}, "manifest 不合规"),
          el("ul", {}, ...problems.map((p) => el("li", {}, p)))),
      );
      return;
    }
    const manifest: ChapterManifestJson = bundle.manifest;
    const viz = manifest.vizzes.find((v) => v.id === state.viz) ?? manifest.vizzes[0];
    if (!viz) return;
    const model = bundle.models[viz.id];
    if (!model) {
      main.innerHTML = "";
      main.append(el("div", { className: "nl-todo" }, el("h2", {}, "这个可视化还没实现"), el("p", {}, viz.id)));
      return;
    }
    view?.dispose();
    main.innerHTML = "";
    view = await mountViz({
      manifest,
      viz,
      model,
      host: main,
      baseUrl,
      params: state.params,
      t: state.t,
      onState: (s) => {
        current = s;
        // 编码只用 share.ts 里那一份实现（有 20 组随机往返测试），这里不重复造
        history.replaceState(null, "", buildHash(s));
      },
    });
    current = state;
    renderNav();
  };

  const fromHash = (): ShareState | null => decodeHash(location.hash);
  window.addEventListener("hashchange", () => {
    const s = fromHash();
    if (s) void goto(s);
  });

  const initial = fromHash() ?? (await firstImplemented());
  if (initial) {
    await goto(initial);
  } else {
    main.append(el("div", { className: "nl-todo" }, el("h2", {}, "还没有任何可视化"), el("p", {}, "先跑 make precompute 与 make build。")));
  }
  renderNav();

  return {
    goto,
    get current() {
      return current;
    },
  };
}

const host = document.getElementById("app");
if (host) {
  void startSite(host, { baseUrl: "" }).catch((err: unknown) => {
    host.textContent = `启动失败：${(err as Error).message}`;
  });
}
