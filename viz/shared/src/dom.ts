/**
 * 站点用的小型 DOM 工具：元素构造、线性图标、轻提示。
 *
 * 这里刻意不引入任何框架：整站运行期零依赖，图标也用内联 SVG（不请求字体或图标库）。
 * 图标只服务于界面语义，因此统一 24×24、1.7 描边、`currentColor`，随主题自动变色。
 */

export type IconName =
  | "menu"
  | "close"
  | "search"
  | "sun"
  | "moon"
  | "monitor"
  | "chevron-down"
  | "chevron-right"
  | "chevron-left"
  | "play"
  | "pause"
  | "step-back"
  | "step-forward"
  | "download"
  | "link"
  | "expand"
  | "reset"
  | "info"
  | "sliders"
  | "chart"
  | "book"
  | "check"
  | "alert"
  | "arrow-right";

const PATHS: Readonly<Record<IconName, string>> = {
  menu: '<path d="M4 7h16M4 12h16M4 17h16"/>',
  close: '<path d="M6 6l12 12M18 6L6 18"/>',
  search: '<circle cx="10.8" cy="10.8" r="6.4"/><path d="M15.6 15.6L21 21"/>',
  sun: '<circle cx="12" cy="12" r="4.1"/><path d="M12 2.4v2.2M12 19.4v2.2M2.4 12h2.2M19.4 12h2.2M5.2 5.2l1.6 1.6M17.2 17.2l1.6 1.6M18.8 5.2l-1.6 1.6M6.8 17.2l-1.6 1.6"/>',
  moon: '<path d="M20.1 14.4A8.4 8.4 0 0 1 9.6 3.9a8.6 8.6 0 1 0 10.5 10.5Z"/>',
  monitor: '<rect x="3" y="4.5" width="18" height="12.5" rx="1.8"/><path d="M9 21h6M12 17v4"/>',
  "chevron-down": '<path d="M6.5 9.5 12 15l5.5-5.5"/>',
  "chevron-right": '<path d="M9.5 6.5 15 12l-5.5 5.5"/>',
  "chevron-left": '<path d="M14.5 6.5 9 12l5.5 5.5"/>',
  play: '<path d="M8.4 5.6 18 12l-9.6 6.4Z" fill="currentColor" stroke="none"/>',
  pause: '<path d="M8.8 5.8v12.4M15.2 5.8v12.4" stroke-width="2.6"/>',
  "step-back": '<path d="M18.4 6.4 9.6 12l8.8 5.6Z" fill="currentColor" stroke="none"/><path d="M6 5.8v12.4"/>',
  "step-forward": '<path d="M5.6 6.4 14.4 12l-8.8 5.6Z" fill="currentColor" stroke="none"/><path d="M18 5.8v12.4"/>',
  download: '<path d="M12 3.8v11.4M7.4 10.8 12 15.4l4.6-4.6M4.8 20.2h14.4"/>',
  link: '<path d="M10.2 13.8a4.2 4.2 0 0 0 6 0l2.8-2.8a4.2 4.2 0 1 0-6-6l-1.5 1.5"/><path d="M13.8 10.2a4.2 4.2 0 0 0-6 0l-2.8 2.8a4.2 4.2 0 1 0 6 6l1.5-1.5"/>',
  expand: '<path d="M9 3.8H3.8V9M15 3.8h5.2V9M9 20.2H3.8V15M15 20.2h5.2V15"/>',
  reset: '<path d="M4.6 9.4A8 8 0 1 1 4 14.4"/><path d="M4.4 4.4v5.2h5.2"/>',
  info: '<circle cx="12" cy="12" r="8.6"/><path d="M12 11v5.4M12 7.9v.1"/>',
  sliders: '<path d="M4 7h10M18 7h2M4 17h2M10 17h10"/><circle cx="16" cy="7" r="2.1"/><circle cx="8" cy="17" r="2.1"/>',
  chart: '<path d="M4 4v16h16"/><path d="M7.4 15.4l3.2-3.6 2.7 2.1 4.5-5.3"/>',
  book: '<path d="M4.2 5.4c2.7-.8 5.3-.8 7.8.6 2.5-1.4 5.1-1.4 7.8-.6v13.2c-2.7-.8-5.3-.8-7.8.6-2.5-1.4-5.1-1.4-7.8-.6Z"/><path d="M12 6v13.2"/>',
  check: '<path d="m5.4 12.2 4.2 4.2 9-9"/>',
  alert: '<path d="M12 3.8 21 19.6H3Z"/><path d="M12 9v4.6M12 16.4v.1"/>',
  "arrow-right": '<path d="M4.5 12h14M13.5 7l5 5-5 5"/>',
};

/** 生成一枚图标；颜色与尺寸都由 CSS 决定。 */
export function icon(name: IconName, className = ""): SVGSVGElement {
  const SVG_NS = "http://www.w3.org/2000/svg";
  const node = document.createElementNS(SVG_NS, "svg");
  node.setAttribute("viewBox", "0 0 24 24");
  node.setAttribute("aria-hidden", "true");
  node.setAttribute("focusable", "false");
  node.setAttribute("fill", "none");
  node.setAttribute("stroke", "currentColor");
  node.setAttribute("stroke-width", "1.7");
  node.setAttribute("stroke-linecap", "round");
  node.setAttribute("stroke-linejoin", "round");
  node.classList.add("nl-icon");
  if (className !== "") node.classList.add(...className.split(/\s+/).filter(Boolean));
  node.innerHTML = PATHS[name];
  return node;
}

export interface ElementOptions {
  className?: string;
  text?: string;
  html?: string;
  title?: string;
  attrs?: Record<string, string>;
}

/** 建元素并一次性设置常用属性（children 由调用方 append）。 */
export function el<K extends keyof HTMLElementTagNameMap>(tag: K, opts: ElementOptions = {}): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (opts.className !== undefined) node.className = opts.className;
  if (opts.text !== undefined) node.textContent = opts.text;
  if (opts.html !== undefined) node.innerHTML = opts.html;
  if (opts.title !== undefined) node.title = opts.title;
  if (opts.attrs !== undefined) {
    for (const [key, value] of Object.entries(opts.attrs)) node.setAttribute(key, value);
  }
  return node;
}

let toastTimer = 0;

/** 页面底部的一行轻提示；同一时刻只保留一条。 */
export function toast(message: string, tone: "ok" | "warn" = "ok"): void {
  let host = document.getElementById("nl-toast-host");
  if (!host) {
    host = document.createElement("div");
    host.id = "nl-toast-host";
    host.className = "nl-toast-host";
    host.setAttribute("aria-live", "polite");
    document.body.append(host);
  }
  const item = document.createElement("div");
  item.className = `nl-toast nl-toast-${tone}`;
  const mark = document.createElement("span");
  mark.className = "nl-toast-mark";
  mark.append(icon(tone === "ok" ? "check" : "alert"));
  const text = document.createElement("span");
  text.textContent = message;
  item.append(mark, text);
  host.replaceChildren(item);
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => {
    item.classList.add("is-leaving");
    window.setTimeout(() => item.remove(), 220);
  }, 2200);
}
