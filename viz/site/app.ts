/**
 * 站点外壳：目录、路由、章节页与主题。
 *
 * 目录只列真正有交互实验的章节（模板章 0.0 不进入产品）。路由形如
 * `#/2.1/runge_nodes`；带完整参数与时刻的分享链接仍由 `share.ts` 的
 * `#v1.…` 编码承载。页面外壳本身不碰数值，只决定「展示哪一章、哪一个实验」。
 */

import { chapterIndex } from "../shared/src/chapters.gen.js";
import { loadChapter } from "../chapters/index.gen.js";
import type { ChapterIndexEntry } from "../shared/src/chapters.js";
import { groupByPart, PART_TITLES } from "../shared/src/chapters.js";
import { decodeHash, type ShareState } from "../shared/src/share.js";
import { VizView, type SiteFormula } from "../shared/src/view.js";
import { validateManifestJson } from "../shared/src/manifest.js";
import type { ChapterManifestJson } from "../shared/src/manifest.js";
import type { SceneModel } from "../shared/src/model.js";
import { renderRichText } from "../shared/src/math.js";
import { el, icon, type IconName } from "../shared/src/dom.js";

const DEFAULT_CHAPTER = "1.1";
const THEME_KEY = "nl-theme";
const BASE_URL = "";

type ThemePref = "light" | "dark" | "system";
type ThemeName = "light" | "dark";
type FormulaMap = Record<string, SiteFormula[]>;

const CHAPTERS: ChapterIndexEntry[] = chapterIndex.filter((entry) => entry.chapter !== "0.0");

interface ChapterBundleLike {
  manifest: ChapterManifestJson;
  models: Record<string, unknown>;
}

interface GlossaryItem {
  term: string;
  desc: string;
}

interface GuideEntry {
  intro?: string;
  note?: string;
  glossary?: GlossaryItem[];
}

interface GuideFile {
  chapters?: Record<string, GuideEntry>;
}

/* -------------------------------------------------------------------------- */
/* 文本与图标                                                                */
/* -------------------------------------------------------------------------- */

function soften(text: string): string {
  return text
    .replace(/[（(]\s*规则\s*[0-9]+(?:\s*\/\s*[0-9]+)?[^）)]*[）)]/g, "")
    .replace(/[（(]\s*(?:PROTOCOLS|POLICY)\s*[^）)]*[）)]/g, "")
    .replace(/规则\s*[0-9]+(?:\s*\/\s*[0-9]+)?/g, "")
    .replace(/[ \t]{2,}/g, " ")
    .replace(/\s+([，。；：、）】])/g, "$1")
    .trim();
}

/** 富文本 → 纯文本（浏览器内）：用于标签副标题这类不能放 MathML 的位置。 */
function plainRichText(text: string): string {
  const box = document.createElement("div");
  box.innerHTML = renderRichText(soften(text));
  return box.textContent ?? "";
}

function tabSummary(text: string, max = 42): string {
  // 简介里现在大量使用 $...$。标签副标题是纯文本，先渲染成 MathML 再取 textContent，
  // 这样读者看到的是 κ、n、a 等符号，而不是一对裸露的美元号或 TeX 命令。
  const plain = plainRichText(text).replace(/\s+/g, " ").trim();
  if (plain === "") return "";
  const end = plain.search(/[。！？]/);
  let first = end >= 0 ? plain.slice(0, end) : plain;
  first = first.replace(/[。！？]$/, "").trim();
  if ([...first].length <= max) return first;
  return `${[...first].slice(0, max).join("").replace(/[，、；：\s]+$/, "")}…`;
}

function partName(part: string): string {
  const title = PART_TITLES[part] ?? part;
  const tail = title.split("·").slice(1).join("·").trim();
  return tail === "" ? title : tail;
}

function routeHash(chapter: string, viz?: string): string {
  return viz === undefined || viz === "" ? `#/${chapter}` : `#/${chapter}/${viz}`;
}

function hydrateIcons(root: ParentNode): void {
  for (const node of root.querySelectorAll<HTMLElement>("[data-icon]")) {
    const name = node.dataset["icon"] as IconName | undefined;
    if (name === undefined) continue;
    node.append(icon(name));
    node.removeAttribute("data-icon");
  }
}

/* -------------------------------------------------------------------------- */
/* 主题                                                                      */
/* -------------------------------------------------------------------------- */

function readThemePref(): ThemePref {
  try {
    const stored = localStorage.getItem(THEME_KEY);
    if (stored === "light" || stored === "dark" || stored === "system") return stored;
  } catch {
    // 隐私模式下 localStorage 可能不可用；跟随系统即可。
  }
  return "system";
}

function systemTheme(): ThemeName {
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function effectiveTheme(pref: ThemePref): ThemeName {
  return pref === "system" ? systemTheme() : pref;
}

/* -------------------------------------------------------------------------- */
/* 正文                                                                      */
/* -------------------------------------------------------------------------- */

function skeleton(): HTMLElement {
  const box = el("div", { className: "chapter-skeleton", attrs: { "aria-busy": "true", "aria-label": "正在打开章节" } });
  box.innerHTML = `
    <div class="sk sk-eyebrow"></div>
    <div class="sk sk-title"></div>
    <div class="sk sk-title sk-title-short"></div>
    <div class="sk sk-lede"></div>
    <div class="sk sk-lede sk-lede-short"></div>
    <div class="sk-row"><div class="sk sk-tab"></div><div class="sk sk-tab"></div><div class="sk sk-tab"></div><div class="sk sk-tab"></div></div>
    <div class="sk sk-canvas"></div>`;
  return box;
}

function errorCard(title: string, message: string, detail?: string): HTMLElement {
  const card = el("div", { className: "nl-empty" });
  const mark = el("div", { className: "nl-empty-mark" });
  mark.append(icon("alert"));
  const h = el("h2", { text: title });
  const p = el("p", { text: message });
  const retry = el("button", { className: "nl-btn nl-btn-primary", attrs: { type: "button" } });
  retry.append(icon("reset"), el("span", { text: "刷新页面" }));
  retry.addEventListener("click", () => location.reload());
  card.append(mark, h, p, retry);
  if (detail !== undefined && detail.trim() !== "") {
    const more = el("details", { className: "nl-empty-detail" });
    more.append(el("summary", { text: "技术细节" }), el("p", { text: detail }));
    card.append(more);
  }
  return card;
}

async function loadFormulas(): Promise<FormulaMap> {
  try {
    const response = await fetch(`${BASE_URL}data/site/formulas.json`, { cache: "no-cache" });
    if (!response.ok) return {};
    const parsed = (await response.json()) as unknown;
    if (typeof parsed !== "object" || parsed === null) return {};
    const out: FormulaMap = {};
    for (const [chapter, value] of Object.entries(parsed as Record<string, unknown>)) {
      if (!Array.isArray(value)) continue;
      const items: SiteFormula[] = [];
      for (const raw of value) {
        if (typeof raw !== "object" || raw === null) continue;
        const item = raw as Record<string, unknown>;
        if (typeof item["tex"] === "string" && typeof item["caption"] === "string") {
          items.push({ tex: item["tex"], caption: item["caption"] });
        }
      }
      if (items.length > 0) out[chapter] = items;
    }
    return out;
  } catch {
    return {};
  }
}

async function loadGuides(): Promise<Record<string, GuideEntry>> {
  try {
    const response = await fetch(`${BASE_URL}data/site/guides.json`, { cache: "no-cache" });
    if (!response.ok) return {};
    const parsed = (await response.json()) as GuideFile;
    return parsed.chapters ?? {};
  } catch {
    return {};
  }
}

/* -------------------------------------------------------------------------- */
/* 站点                                                                      */
/* -------------------------------------------------------------------------- */

export interface SiteHandle {
  goto(state: ShareState): Promise<void>;
  readonly current: ShareState | null;
}

export async function startSite(root: HTMLElement, opts: { baseUrl?: string } = {}): Promise<SiteHandle> {
  void opts;
  const formulas = await loadFormulas();
  const guides = await loadGuides();
  const chapterMap = new Map(CHAPTERS.map((entry) => [entry.chapter, entry]));
  const totalViz = CHAPTERS.reduce((sum, entry) => sum + entry.nViz, 0);

  root.innerHTML = `
    <a class="skip-link" href="#main">跳到正文</a>
    <div class="app-shell">
      <aside class="sidebar" id="sidebar" aria-label="章节目录">
        <div class="sidebar-head">
          <a class="brand" href="#/${DEFAULT_CHAPTER}" data-home>
            <span class="brand-mark" aria-hidden="true">
              <svg viewBox="0 0 32 32" focusable="false">
                <rect x="1.3" y="1.3" width="29.4" height="29.4" rx="9" fill="currentColor" opacity=".1"/>
                <path d="M7 22.2c4.1-1.3 4.6-11.3 9-11.3s4.6 8.8 9 6.5" fill="none" stroke="currentColor" stroke-width="2.1" stroke-linecap="round"/>
                <circle cx="16" cy="16" r="2.15" fill="currentColor"/>
              </svg>
            </span>
            <span class="brand-copy">
              <span class="brand-name">数值方法实验室</span>
              <span class="brand-sub">NUMERICAL METHODS LAB</span>
            </span>
          </a>
          <button class="icon-btn sidebar-close" type="button" data-close-nav aria-label="关闭目录"><span data-icon="close"></span></button>
        </div>
        <div class="nav-search">
          <span class="nav-search-icon" data-icon="search"></span>
          <input id="navSearch" type="search" placeholder="搜索章节或主题" autocomplete="off" aria-label="搜索章节">
          <kbd>/</kbd>
        </div>
        <nav class="toc" id="toc" aria-label="章节列表"></nav>
        <div class="sidebar-foot">
          <p class="sidebar-stat">${CHAPTERS.length} 个章节 · ${totalViz} 个交互实验</p>
          <p>从浮点误差到偏微分方程</p>
        </div>
      </aside>
      <button class="sidebar-scrim" type="button" data-close-nav aria-label="关闭目录" tabindex="-1"></button>
      <div class="workspace">
        <header class="topbar">
          <button class="icon-btn menu-btn" type="button" data-open-nav aria-label="打开目录" aria-expanded="false" aria-controls="sidebar"><span data-icon="menu"></span></button>
          <nav class="breadcrumbs" aria-label="面包屑">
            <a class="crumb-home" href="#/${DEFAULT_CHAPTER}">目录</a>
            <span class="crumb-sep" aria-hidden="true"><span data-icon="chevron-right"></span></span>
            <span class="crumb-part" id="crumbPart"></span>
            <span class="crumb-sep" aria-hidden="true"><span data-icon="chevron-right"></span></span>
            <span class="crumb-current" id="crumbCurrent"></span>
          </nav>
          <div class="topbar-actions">
            <div class="theme-switch" role="group" aria-label="主题">
              <button type="button" data-theme-choice="light" aria-label="浅色主题" title="浅色"><span data-icon="sun"></span></button>
              <button type="button" data-theme-choice="dark" aria-label="深色主题" title="深色"><span data-icon="moon"></span></button>
              <button type="button" data-theme-choice="system" aria-label="跟随系统" title="跟随系统"><span data-icon="monitor"></span></button>
            </div>
          </div>
        </header>
        <main class="content" id="main" tabindex="-1"></main>
        <footer class="site-foot">
          <span>数值方法实验室</span>
          <span class="site-foot-sep" aria-hidden="true">·</span>
          <span>${CHAPTERS.length} 章 / ${totalViz} 个交互实验</span>
          <span class="site-foot-sep" aria-hidden="true">·</span>
          <span>每一帧都来自可复现的数值计算</span>
        </footer>
      </div>
    </div>`;
  hydrateIcons(root);

  const sidebar = root.querySelector<HTMLElement>("#sidebar") as HTMLElement;
  const toc = root.querySelector<HTMLElement>("#toc") as HTMLElement;
  const search = root.querySelector<HTMLInputElement>("#navSearch") as HTMLInputElement;
  const main = root.querySelector<HTMLElement>("#main") as HTMLElement;
  const crumbPart = root.querySelector<HTMLElement>("#crumbPart") as HTMLElement;
  const crumbCurrent = root.querySelector<HTMLElement>("#crumbCurrent") as HTMLElement;
  const menuButton = root.querySelector<HTMLButtonElement>("[data-open-nav]") as HTMLButtonElement;
  const themeMeta = document.querySelector<HTMLMetaElement>('meta[name="theme-color"]');

  let view: VizView | null = null;
  let current: ShareState | null = null;
  let currentRoute = "";
  let themePref = readThemePref();
  let renderedTheme = effectiveTheme(themePref);
  main.replaceChildren(skeleton());
  const getCurrent = (): ShareState | null => current;

  // ------------------------------------------------------------------ 主题
  const applyTheme = (pref: ThemePref, persist: boolean): void => {
    themePref = pref;
    const effective = effectiveTheme(pref);
    renderedTheme = effective;
    document.documentElement.dataset["theme"] = effective;
    document.documentElement.style.colorScheme = effective;
    if (themeMeta !== null) themeMeta.content = effective === "dark" ? "#0d1216" : "#f6f4ee";
    for (const button of root.querySelectorAll<HTMLButtonElement>("[data-theme-choice]")) {
      const active = button.dataset["themeChoice"] === pref;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", active ? "true" : "false");
    }
    if (persist) {
      try {
        localStorage.setItem(THEME_KEY, pref);
      } catch {
        // 存不下就算了，主题仍然生效。
      }
    }
    view?.setTheme(effective);
  };
  for (const button of root.querySelectorAll<HTMLButtonElement>("[data-theme-choice]")) {
    button.addEventListener("click", () => applyTheme(button.dataset["themeChoice"] as ThemePref, true));
  }
  const media = window.matchMedia("(prefers-color-scheme: dark)");
  const onSystemTheme = (): void => {
    if (themePref === "system") applyTheme("system", false);
  };
  media.addEventListener("change", onSystemTheme);
  applyTheme(themePref, false);

  // ------------------------------------------------------------------ 抽屉
  const setNav = (open: boolean): void => {
    document.body.classList.toggle("nav-open", open);
    menuButton.setAttribute("aria-expanded", open ? "true" : "false");
    if (open) {
      const active = toc.querySelector<HTMLElement>("a.is-active");
      active?.focus({ preventScroll: true });
    }
  };
  menuButton.addEventListener("click", () => setNav(!document.body.classList.contains("nav-open")));
  for (const button of root.querySelectorAll<HTMLElement>("[data-close-nav]")) {
    button.addEventListener("click", () => setNav(false));
  }

  // ------------------------------------------------------------------ 目录
  interface NavEntry {
    entry: ChapterIndexEntry;
    link: HTMLAnchorElement;
    part: HTMLDetailsElement;
    searchText: string;
  }
  const navEntries: NavEntry[] = [];

  const buildNav = (): void => {
    toc.replaceChildren();
    for (const group of groupByPart(CHAPTERS)) {
      const details = el("details", { className: "nav-part" });
      details.open = true;
      const summary = el("summary", { className: "nav-part-head" });
      summary.append(
        el("span", { className: "nav-part-roman", text: group.part === "" ? "·" : group.part }),
        el("span", { className: "nav-part-title", text: partName(group.part) }),
        el("span", { className: "nav-part-count", text: String(group.items.length) }),
      );
      const list = el("ul", { className: "nav-list" });
      for (const entry of group.items) {
        const item = el("li", { className: "nav-item" });
        const link = el("a", {
          className: "nav-link",
          attrs: { href: routeHash(entry.chapter), "data-chapter": entry.chapter },
        });
        link.append(
          el("span", { className: "nav-num", text: entry.chapter }),
          el("span", { className: "nav-text", text: entry.title }),
        );
        link.addEventListener("click", (ev) => {
          ev.preventDefault();
          setNav(false);
          void openChapter(entry.chapter, "chapter", "push");
        });
        item.append(link);
        list.append(item);
        navEntries.push({
          entry,
          link,
          part: details,
          searchText: `${entry.chapter} ${entry.title} ${entry.summary}`.toLowerCase(),
        });
      }
      details.append(summary, list);
      toc.append(details);
    }
  };

  const filterNav = (): void => {
    const query = search.value.trim().toLowerCase();
    let shown = 0;
    const partHits = new Map<HTMLDetailsElement, number>();
    for (const item of navEntries) {
      const hit = query === "" || item.searchText.includes(query);
      item.link.parentElement!.hidden = !hit;
      if (hit) {
        shown += 1;
        partHits.set(item.part, (partHits.get(item.part) ?? 0) + 1);
        if (query !== "") item.part.open = true;
      }
    }
    for (const part of toc.querySelectorAll<HTMLDetailsElement>(".nav-part")) {
      part.hidden = query !== "" && !partHits.has(part);
    }
    toc.dataset["search"] = query;
    if (query !== "" && shown === 0) {
      let empty = toc.querySelector<HTMLElement>(".nav-empty");
      if (empty === null) {
        empty = el("p", { className: "nav-empty", text: "没有找到匹配的章节" });
        toc.append(empty);
      }
      empty.hidden = false;
    } else {
      const empty = toc.querySelector<HTMLElement>(".nav-empty");
      if (empty !== null) empty.hidden = true;
    }
  };
  search.addEventListener("input", filterNav);
  search.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape") {
      search.value = "";
      filterNav();
      search.blur();
    }
  });

  const updateActiveNav = (chapter: string): void => {
    for (const item of navEntries) {
      const active = item.entry.chapter === chapter;
      item.link.classList.toggle("is-active", active);
      if (active) {
        item.part.open = true;
        const linkBox = item.link.getBoundingClientRect();
        const navBox = sidebar.getBoundingClientRect();
        if (linkBox.top < navBox.top + 90 || linkBox.bottom > navBox.bottom - 20) {
          item.link.scrollIntoView({ block: "nearest" });
        }
      }
    }
  };

  // ------------------------------------------------------------------ 路由
  const parseRoute = (): ShareState | null => {
    const shared = decodeHash(location.hash);
    if (shared !== null) return shared;
    const raw = decodeURIComponent(location.hash.replace(/^#/, ""));
    const match = /^\/?([0-9]+\.[0-9]+)(?:\/([a-z][a-z0-9_]*))?$/.exec(raw);
    if (match === null) return null;
    const chapter = chapterMap.has(match[1] as string) ? (match[1] as string) : DEFAULT_CHAPTER;
    return { chapter, viz: match[2] ?? "", params: {}, t: 0, playing: false };
  };

  const defaultState = async (): Promise<ShareState | null> => {
    const entry = chapterMap.get(DEFAULT_CHAPTER) ?? CHAPTERS[0];
    if (entry === undefined) return null;
    return { chapter: entry.chapter, viz: "", params: {}, t: 0, playing: false };
  };

  const chooseBundle = async (chapter: string): Promise<{ bundle: ChapterBundleLike | null; error?: unknown }> => {
    try {
      const bundle = (await loadChapter(chapter)) as ChapterBundleLike | null;
      return { bundle };
    } catch (error) {
      return { bundle: null, error };
    }
  };

  const implementedVizzes = (bundle: ChapterBundleLike): ChapterManifestJson["vizzes"] =>
    bundle.manifest.vizzes.filter((viz) => typeof bundle.models[viz.id] === "function");

  const renderChapter = (
    manifest: ChapterManifestJson,
    vizzes: ChapterManifestJson["vizzes"],
    selected: ChapterManifestJson["vizzes"][number],
  ): { vizHost: HTMLElement; tabs: HTMLElement | null } => {
    main.replaceChildren();
    const entry = chapterMap.get(manifest.chapter);
    const part = manifest.part ?? entry?.part ?? "";
    const guide = guides[manifest.chapter];
    const hero = el("section", { className: "chapter-hero" });
    const glyph = el("span", { className: "chapter-glyph", text: manifest.chapter, attrs: { "aria-hidden": "true" } });
    const eyebrow = el("p", { className: "chapter-eyebrow" });
    eyebrow.append(
      el("span", { className: "chapter-eyebrow-dot", attrs: { "aria-hidden": "true" } }),
      el("span", { text: `${partName(part)} / 第 ${manifest.chapter.split(".")[0]} 部分` }),
    );
    const h1 = el("h1", { className: "chapter-title", attrs: { tabindex: "-1" } });
    h1.append(el("span", { className: "chapter-number", text: manifest.chapter }), el("span", { text: manifest.title }));
    const lede = el("div", { className: "chapter-lede nl-prose" });
    const intro = guide?.intro?.trim() ?? "";
    lede.innerHTML = renderRichText(soften(intro === "" ? (manifest.summary ?? "") : intro));

    const glossary = Array.isArray(guide?.glossary) ? guide.glossary : [];
    const glossaryBox = el("section", { className: "nl-glossary", attrs: { "aria-label": "本章术语" } });
    if (glossary.length > 0) {
      glossaryBox.append(el("p", { className: "nl-glossary-title", text: "先认识这些词" }));
      const list = el("dl", { className: "nl-glossary-list" });
      for (const item of glossary) {
        if (typeof item?.term !== "string" || typeof item?.desc !== "string") continue;
        const row = el("div", { className: "nl-glossary-item" });
        const desc = el("dd");
        desc.innerHTML = renderRichText(soften(item.desc));
        row.append(el("dt", { text: item.term }), desc);
        list.append(row);
      }
      glossaryBox.append(list);
    }

    const paramsCount = vizzes.reduce((sum, viz) => sum + viz.params.length, 0);
    const formulaCount = (formulas[manifest.chapter] ?? []).length;
    const meta = el("div", { className: "chapter-meta" });
    const stat = (text: string): HTMLElement => el("span", { className: "chapter-stat", text });
    meta.append(stat(`${vizzes.length} 个交互实验`), stat(`${paramsCount} 个可调参数`));
    if (formulaCount > 0) meta.append(stat(`${formulaCount} 条核心公式`));
    hero.append(glyph, eyebrow, h1, lede);
    if (glossary.length > 0) hero.append(glossaryBox);
    hero.append(meta);
    main.append(hero);

    let tabs: HTMLElement | null = null;
    if (vizzes.length > 1) {
      tabs = el("nav", { className: "viz-tabs", attrs: { "aria-label": "本章实验" } });
      vizzes.forEach((viz, index) => {
        const link = el("a", {
          className: `viz-tab${viz.id === selected.id ? " is-active" : ""}`,
          attrs: { href: routeHash(manifest.chapter, viz.id), "aria-current": viz.id === selected.id ? "page" : "false" },
        });
        const titleBox = el("span", { className: "viz-tab-copy" });
        const tabTitle = el("span", { className: "viz-tab-title" });
        tabTitle.innerHTML = renderRichText(viz.title);
        titleBox.append(tabTitle);
        const summary = tabSummary(viz.description ?? "");
        if (summary !== "") titleBox.append(el("span", { className: "viz-tab-sub", text: summary }));
        link.append(el("span", { className: "viz-tab-index", text: String(index + 1).padStart(2, "0") }), titleBox);
        link.addEventListener("click", (ev) => {
          ev.preventDefault();
          if (viz.id === selected.id) return;
          void openState(
            { chapter: manifest.chapter, viz: viz.id, params: {}, t: 0, playing: false },
            { scroll: "viz", sync: "push", restore: false },
          );
        });
        tabs!.append(link);
      });
      main.append(tabs);
    }

    const vizHost = el("div", { className: "viz-host" });
    main.append(vizHost);

    // 窄屏上实验卡是横向滚动的；把当前实验滚到可见区域，否则深链/切章后
    // 用户看到的可能是另外几项，误以为没有选中。
    const tabStrip = tabs;
    const activeTab = tabStrip?.querySelector<HTMLElement>(".viz-tab.is-active");
    if (tabStrip && activeTab) {
      requestAnimationFrame(() => {
        const left = activeTab.offsetLeft - Math.max(0, (tabStrip.clientWidth - activeTab.offsetWidth) / 2);
        tabStrip.scrollTo({ left: Math.max(0, left), behavior: "auto" });
      });
    }

    const pager = el("nav", { className: "chapter-pager", attrs: { "aria-label": "翻章" } });
    const index = CHAPTERS.findIndex((item) => item.chapter === manifest.chapter);
    const prev = index > 0 ? CHAPTERS[index - 1] : undefined;
    const next = index >= 0 && index < CHAPTERS.length - 1 ? CHAPTERS[index + 1] : undefined;
    const pagerLink = (entryItem: ChapterIndexEntry, direction: "prev" | "next"): HTMLAnchorElement => {
      const link = el("a", { className: `pager-link pager-${direction}`, attrs: { href: routeHash(entryItem.chapter) } });
      const label = el("span", { className: "pager-label" });
      label.append(icon(direction === "prev" ? "chevron-left" as IconName : "chevron-right"));
      label.append(el("span", { text: direction === "prev" ? "上一章" : "下一章" }));
      link.append(label, el("strong", { text: `${entryItem.chapter} ${entryItem.title}` }));
      link.addEventListener("click", (ev) => {
        ev.preventDefault();
        void openChapter(entryItem.chapter, "chapter", "push");
      });
      return link;
    };
    if (prev !== undefined) pager.append(pagerLink(prev, "prev"));
    else pager.append(el("span", { className: "pager-spacer" }));
    if (next !== undefined) pager.append(pagerLink(next, "next"));
    main.append(pager);

    crumbPart.textContent = partName(part);
    crumbCurrent.textContent = `${manifest.chapter} ${manifest.title}`;
    document.title = `${manifest.chapter} ${manifest.title} · 数值方法实验室`;
    updateActiveNav(manifest.chapter);
    filterNav();
    return { vizHost, tabs };
  };

  interface OpenOptions {
    scroll: "none" | "chapter" | "viz";
    sync: "none" | "push" | "replace";
    /** 分享链接与历史回退要恢复原时刻；普通进入章节时落在时间轴末端，先看到完整现象。 */
    restore: boolean;
  }

  const openState = async (state: ShareState, options: OpenOptions): Promise<void> => {
    const entry = chapterMap.get(state.chapter) ?? chapterMap.get(DEFAULT_CHAPTER) ?? CHAPTERS[0];
    if (entry === undefined) {
      main.replaceChildren(errorCard("还没有可读的章节", "章节数据没有加载成功。", "CHAPTERS 为空"));
      return;
    }
    if (current === null || current.chapter !== entry.chapter) {
      main.replaceChildren(skeleton());
    }
    const { bundle, error } = await chooseBundle(entry.chapter);
    if (bundle === null) {
      main.replaceChildren(
        errorCard(
          "这一章没有打开",
          "章节脚本加载失败。刷新页面通常可以恢复；如果一直失败，请检查本地服务。",
          error instanceof Error ? error.message : String(error),
        ),
      );
      return;
    }
    const problems = validateManifestJson(bundle.manifest);
    if (problems.length > 0) {
      main.replaceChildren(errorCard("这一章的数据不完整", "manifest 没有通过校验，暂时无法绘图。", problems.join("\n")));
      return;
    }
    const manifest = bundle.manifest;
    const vizzes = implementedVizzes(bundle);
    if (vizzes.length === 0) {
      main.replaceChildren(errorCard("这一章还没有可交互的实验", "章节数据存在，但还没有挂上图形模型。"));
      return;
    }
    const selected = vizzes.find((viz) => viz.id === state.viz) ?? vizzes[0];
    if (selected === undefined) return;

    if (options.scroll === "chapter") {
      window.scrollTo({ top: 0, behavior: "auto" });
    }

    const previous = view;
    view = null;
    previous?.dispose();
    const rendered = renderChapter(manifest, vizzes, selected);
    // 直接打开时从 t=0 开始，而不是从 tMax 开始：端点常常是「条件数拉满 / 扰动最大」
    // 的极端状态，默认就亮红色溢出既吓人又掩盖了正常状态。时间轴仍然可以拖到端点。
    const startT = options.restore ? state.t : (selected.time.initial ?? 0);
    current = {
      chapter: manifest.chapter,
      viz: selected.id,
      params: state.params,
      t: startT,
      playing: options.restore ? state.playing : false,
    };
    currentRoute = `${manifest.chapter}/${selected.id}`;

    if (options.sync !== "none") {
      const hash = routeHash(manifest.chapter, selected.id);
      if (location.hash !== hash) history[options.sync === "push" ? "pushState" : "replaceState"](null, "", hash);
    }

    const ordinal = vizzes.findIndex((viz) => viz.id === selected.id) + 1;
    const instance = new VizView({
      manifest,
      viz: selected,
      model: bundle.models[selected.id] as SceneModel,
      host: rendered.vizHost,
      baseUrl: BASE_URL,
      theme: renderedTheme,
      params: state.params,
      t: startT,
      playing: options.restore ? state.playing : false,
      formulas: formulas[manifest.chapter] ?? [],
      ordinal,
      total: vizzes.length,
      onState: (next) => {
        current = next;
      },
    });
    view = instance;
    const mounting = instance.mount().catch((err: unknown) => {
      instance.dispose();
      rendered.vizHost.replaceChildren(
        errorCard("图形没有启动", "浏览器没能初始化这张图。可以换一个浏览器或刷新页面。", err instanceof Error ? err.message : String(err)),
      );
    });
    if (options.scroll === "viz") {
      requestAnimationFrame(() => {
        rendered.vizHost.scrollIntoView({ behavior: "smooth", block: "start" });
      });
    }
    await mounting;
    if (options.scroll === "viz") {
      rendered.vizHost.querySelector<HTMLElement>(".nl-viz-title")?.focus({ preventScroll: true });
    } else if (options.scroll === "chapter") {
      main.querySelector<HTMLElement>(".chapter-title")?.focus({ preventScroll: true });
    }
  };

  const openChapter = async (chapter: string, scroll: "chapter" | "viz", sync: "push" | "replace"): Promise<void> => {
    await openState({ chapter, viz: "", params: {}, t: 0, playing: false }, { scroll, sync, restore: false });
  };

  const handleLocation = (): void => {
    const state = parseRoute();
    if (state === null) return;
    const restore = decodeHash(location.hash) !== null;
    const key = `${state.chapter}/${state.viz}`;
    if (key === currentRoute) {
      const sameParams = JSON.stringify(state.params) === JSON.stringify(current?.params ?? {});
      if (sameParams && (restore ? Math.abs(state.t - (current?.t ?? 0)) < 1e-12 : true)) return;
    }
    void openState(state, { scroll: "none", sync: "none", restore });
  };

  window.addEventListener("hashchange", handleLocation);
  window.addEventListener("popstate", handleLocation);

  const onKey = (ev: KeyboardEvent): void => {
    const target = ev.target as HTMLElement | null;
    const typing = target !== null && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable);
    if (ev.key === "Escape") {
      if (document.body.classList.contains("nav-open")) setNav(false);
      return;
    }
    if (typing) return;
    if (ev.key === "/" || ((ev.ctrlKey || ev.metaKey) && ev.key.toLowerCase() === "k")) {
      ev.preventDefault();
      if (window.matchMedia("(max-width: 1180px)").matches) setNav(true);
      search.focus();
      search.select();
    }
  };
  document.addEventListener("keydown", onKey);

  buildNav();
  const initial = parseRoute();
  if (initial === null) {
    const fallback = await defaultState();
    if (fallback === null) {
      main.replaceChildren(errorCard("还没有章节", "目录是空的，无法打开实验。"));
      return {
        goto: async () => undefined,
        get current() {
          return current;
        },
      };
    }
    await openState(fallback, { scroll: "none", sync: "replace", restore: false });
    const settled = getCurrent();
    if (location.hash === "" && settled !== null) {
      history.replaceState(null, "", routeHash(settled.chapter, settled.viz));
    }
  } else {
    const restore = decodeHash(location.hash) !== null;
    await openState(initial, {
      scroll: "none",
      sync: initial.chapter === "0.0" ? "replace" : "none",
      restore,
    });
  }
  const finalState = getCurrent();
  if (!location.hash.startsWith("#v1.") && finalState !== null) {
    const desired = routeHash(finalState.chapter, finalState.viz);
    if (location.hash === "" || location.hash !== desired) history.replaceState(null, "", desired);
  }

  return {
    async goto(state: ShareState): Promise<void> {
      await openState(state, { scroll: "none", sync: "push", restore: true });
    },
    get current() {
      return getCurrent();
    },
  };
}

/* -------------------------------------------------------------------------- */
/* 启动                                                                      */
/* -------------------------------------------------------------------------- */

const host = document.getElementById("app");
if (host) {
  void startSite(host, { baseUrl: "" }).catch((err: unknown) => {
    host.replaceChildren(
      errorCard(
        "页面没有启动",
        "浏览器在初始化站点时遇到了问题。刷新一次通常就能恢复。",
        err instanceof Error ? `${err.name}: ${err.message}` : String(err),
      ),
    );
  });
}

const globalWindow = window as Window & {
  numericalLab?: { startSite: typeof startSite };
};
globalWindow.numericalLab = { startSite };
