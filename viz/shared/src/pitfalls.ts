/**
 * "它会怎样骗你"面板（防误导规则 7）。
 *
 * manifest 里的 `pitfalls` 是一整段话；直接铺在页面角落没人读。这里把它拆成**可扫读**的
 * 结构：一句话一条、附上该可视化的临界值提示与数据来源徽标，让"这张图会怎么骗我"和
 * "数据是哪来的"永远同时出现在视野里。
 *
 * 与 `controls.ts` 同样的分工：`pitfallsModel(viz)` 是纯函数（无浏览器也能测），
 * `mountPitfalls(model, parent)` 只负责拼元素。
 *
 * 徽标文案刻意与 `scene.ts::sourceBadge` 用词一致——画布上的徽标和旁边面板里的说法
 * 不一样，是"两处声明漂移"的经典开端。
 */

import { BEYOND_TEXT } from "./controls.js";
import { governingCritical } from "./manifest.js";
import type { VizSpecJson } from "./manifest.js";

/** 面板模型：标题 + 段落 + 徽标。 */
export interface PitfallsModel {
  title: string;
  paragraphs: string[];
  badges: string[];
}

/** 与 `scene.ts::sourceBadge` 同一套话术（规则 5）。 */
export function dataSourceBadgeText(dataSource: string): string {
  return dataSource.startsWith("backend+preview")
    ? "数据：后端帧 + 前端近似预览"
    : "数据：后端 .nlf 帧";
}

/**
 * 按标点切成句子（保留标点，便于独立阅读）。
 *
 * 规则：
 * - 换行（一个或多个）一律当分隔；
 * - `。！？；!?;` 之后断句，并把连续的终止符（`……！？`）吞进同一句；
 * - `.` 只在**后面是空白或行尾**且**不夹在数字之间**时断句——否则 `r=0.5` 会被切成
 *   `r=0.` 和 `5`、`meta.t_actual` 会被切成两半，这是中文技术文本里最容易踩的坑。
 */
export function splitSentences(text: string): string[] {
  const out: string[] = [];
  const terminator = "。！？；!?;";
  for (const block of text.replace(/\r\n?/g, "\n").split(/\n+/)) {
    const line = block.trim();
    if (line === "") continue;
    let start = 0;
    for (let i = 0; i < line.length; i++) {
      const ch = line[i];
      let cut = terminator.includes(ch);
      if (!cut && ch === ".") {
        const prev = i > 0 ? line[i - 1] : "";
        const next = i + 1 < line.length ? line[i + 1] : "";
        cut = (next === "" || /\s/.test(next)) && !(/\d/.test(prev) && /\d/.test(next));
      }
      if (!cut) continue;
      let end = i + 1;
      while (end < line.length && terminator.includes(line[end])) end++;
      const sentence = line.slice(start, end).trim();
      if (sentence !== "") out.push(sentence);
      start = end;
      i = end - 1;
    }
    const rest = line.slice(start).trim();
    if (rest !== "") out.push(rest);
  }
  return out;
}

/** 临界值提示段落（每个有临界值的参数一段）。 */
function criticalParagraphs(viz: VizSpecJson): string[] {
  const out: string[] = [];
  for (const param of viz.params) {
    const critical = [...(param.critical ?? [])].sort((a, b) => a - b);
    if (critical.length === 0) continue;
    const beyond = param.beyond ?? "above";
    const numeric = typeof param.default === "number" ? param.default : undefined;
    const governing = governingCritical(param, numeric);
    const note = param.criticalNote !== undefined && param.criticalNote !== "" ? param.criticalNote : "越过它之后本图的结论可能不再成立";
    const state =
      governing === null
        ? ""
        : numeric === undefined
          ? ""
          : beyond === "below"
            ? numeric < governing
              ? `；⚠ 默认值 ${numeric} 已低于 ${governing}`
              : `；默认值 ${numeric} 尚在阈值内`
            : beyond === "magnitude"
              ? Math.abs(numeric) > Math.abs(governing)
                ? `；⚠ 默认值 ${numeric} 的绝对值已超过 ${governing}`
                : `；默认值 ${numeric} 尚在阈值内`
              : numeric > governing
                ? `；⚠ 默认值 ${numeric} 已超过 ${governing}`
                : `；默认值 ${numeric} 尚在阈值内`;
    out.push(
      `${param.label}（${param.name}）的理论临界 ${critical.join(" / ")}：${BEYOND_TEXT[beyond]}；${note}${state}。`,
    );
  }
  return out;
}

/**
 * 生成"它会怎样骗你"面板模型。
 *
 * `paragraphs` = `pitfalls` 的句子 + 临界值提示 + 坐标轴自适应提示 + 数据来源/误差场说明；
 * `badges` = 数据来源、误差场、坐标轴模式、临界参数。
 */
export function pitfallsModel(viz: VizSpecJson): PitfallsModel {
  const sentences = splitSentences(viz.pitfalls ?? "");
  const paragraphs: string[] = sentences.length > 0 ? sentences : ["（manifest 没有给出 pitfalls——这本身就是个问题）"];

  paragraphs.push(...criticalParagraphs(viz));

  const autoscaled = viz.axes.filter((a) => a.autoscaled).map((a) => a.id);
  if (autoscaled.length > 0) {
    paragraphs.push(
      `坐标轴 ${autoscaled.join("、")} 是自适应的：纵轴跟着数据缩放，发散被画成"看起来是平的"是这张图最常见的骗法（规则 1/4）。`,
    );
  } else {
    paragraphs.push("该可视化的坐标轴是固定的：越界会被标出来，而不是被悄悄缩放掩盖（规则 1/4）。");
  }

  const aux = (viz.requiredAux ?? []).join("、");
  paragraphs.push(
    `必须同时给出误差/残差场（${aux}）：没有它，"收敛"只能靠眼睛判断（规则 3）。${dataSourceBadgeText(viz.dataSource)}——前端只做显示空间插值，不重算物理量（P8）。`,
  );

  const badges: string[] = [dataSourceBadgeText(viz.dataSource)];
  if (aux !== "") badges.push(`含误差场：${aux}`);
  badges.push(autoscaled.length > 0 ? "坐标轴：自适应（可能掩盖发散）" : "坐标轴：固定");
  const criticalParams = viz.params.filter((p) => (p.critical ?? []).length > 0).map((p) => p.name);
  if (criticalParams.length > 0) badges.push(`临界参数：${criticalParams.join("、")}`);
  if (viz.time.mode !== "none") badges.push(`时间轴：${viz.time.mode}`);

  return { title: `${viz.title}：它会怎样骗你`, paragraphs, badges };
}

/**
 * 把面板挂到 `parent` 下（先清空）。
 *
 * 结构：`<section class="nl-pitfalls">` → 标题 / 徽标行 / 每段一个 `<p>`。
 * 返回创建出来的 `<section>`。
 */
export function mountPitfalls(model: PitfallsModel, parent: HTMLElement): HTMLElement {
  const section = document.createElement("section");
  section.className = "nl-pitfalls";

  const title = document.createElement("h3");
  title.className = "nl-pitfalls-title";
  title.textContent = model.title;
  section.appendChild(title);

  const badgeRow = document.createElement("div");
  badgeRow.className = "nl-pitfalls-badges";
  for (const text of model.badges) {
    const span = document.createElement("span");
    span.className = "nl-badge";
    span.textContent = text;
    badgeRow.appendChild(span);
  }
  section.appendChild(badgeRow);

  for (const paragraph of model.paragraphs) {
    const p = document.createElement("p");
    p.className = "nl-pitfalls-line";
    p.textContent = paragraph;
    section.appendChild(p);
  }

  parent.replaceChildren(section);
  return section;
}
