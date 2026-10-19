/**
 * 参数控件：从 manifest 生成**纯描述**，再把它变成 DOM。
 *
 * 分工是刻意的：
 *
 * - `describeControls(viz)` / `clampParam(param, value)` 是**纯函数**，没有浏览器也能测，
 *   它们承载全部"防误导"义务——规则 6 的临界值标记、规则 1/4 的自适应坐标轴提示、
 *   单位与步长的展示口径都在这里定；
 * - `mountControls(...)` 只做一件事：把一个 `ControlSpec` 变成一个 `<input>`/`<select>`，
 *   薄到读一遍就敢信。所有取值都过 `clampParam`，所以 UI 不可能提交越界参数。
 *
 * 关于"临界值标记"：`criticalMarks` 给出每个理论临界值在滑块上的位置（0–1 比例），
 * UI 可以直接画刻度；`governing` 是与 `server/frames.py::governing_critical` 同一口径的
 * "当前起支配作用的那一个"，`crossed` 说明当前值是否已经越过它。
 */

import { governingCritical } from "./manifest.js";
import type { BeyondDirection, ParamSpecJson, ParamType, VizSpecJson } from "./manifest.js";
import type { ParamValue, ParamValues } from "./share.js";

/** 一个理论临界值在控件上的位置（规则 6 的视觉提示义务）。 */
export interface CriticalMark {
  value: number;
  /** 在 `[min, max]` 上的位置（0–1）；范围未知时为 `null`。 */
  fraction: number | null;
  /** 刻度上的短标签（如 `0.5`）。 */
  label: string;
  /** 是否是"当前起支配作用"的那一个。 */
  governing: boolean;
  /** 当前值是否已经越过它。 */
  crossed: boolean;
}

/** 一个参数的完整控件描述（UI 只需要读这个结构）。 */
export interface ControlSpec {
  name: string;
  label: string;
  type: ParamType;
  /** 当前值（= 调用方给的 `params[name]`，缺失或非法时回落到默认值）。 */
  value: ParamValue;
  defaultValue: ParamValue;
  min: number | null;
  max: number | null;
  step: number | null;
  unit: string;
  /** `enum` 的可选项（其它类型为空数组）。 */
  options: string[];
  /** 理论临界值（升序副本；无临界值为空数组——规则 6 要求显式给出）。 */
  critical: number[];
  criticalNote: string;
  criticalMarks: CriticalMark[];
  /** 当前起支配作用的临界值；无临界值为 `null`。 */
  governing: number | null;
  beyond: BeyondDirection;
  /** 当前值是否已越过支配临界值。 */
  crossed: boolean;
  /** 该可视化是否存在自适应坐标轴（规则 1/4）。 */
  autoscaled: boolean;
  /** 面板要显示的全部提示文案（单位 / 临界 / 越界 / 自适应）。 */
  hints: string[];
}

/** `beyond` 的中文说法（"越过它会怎样"的方向）。 */
export const BEYOND_TEXT: Readonly<Record<BeyondDirection, string>> = {
  above: "不得超过",
  below: "不得低于",
  magnitude: "按绝对值比较",
};

function finiteOrNull(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function defaultNumber(param: ParamSpecJson): number {
  if (typeof param.default === "number" && Number.isFinite(param.default)) return param.default;
  return finiteOrNull(param.min) ?? 0;
}

/**
 * 把一个参数值夹进 manifest 声明的合法域，并做类型归一。
 *
 * `bool`：只认布尔与 `"true"/"false"`；`enum`：非法选项回落到默认值（或第一个选项）；
 * `int`/`float`：非数字 / NaN 回落到默认值，然后**先取整再夹取**（`int` 参数不会拿到 3.7）。
 * 这是前端唯一的参数防线——下游（`ChapterFrameSource`）只会拿到这里处理过的值。
 */
export function clampParam(param: ParamSpecJson, value: unknown): ParamValue {
  if (param.type === "bool") {
    if (typeof value === "boolean") return value;
    if (value === "true") return true;
    if (value === "false") return false;
    return typeof param.default === "boolean" ? param.default : false;
  }
  if (param.type === "enum") {
    // 选项可以是数字（Re=1000 这类离散档位）。UI 的 <select> 只能提交字符串，
    // 这里按显示文本匹配后**返回 manifest 里的原值**，否则后端会拒绝 "1000"。
    const options = param.options ?? [];
    const text = typeof value === "string" ? value : typeof value === "number" || typeof value === "boolean" ? String(value) : "";
    const hit = options.find((o) => String(o) === text);
    if (hit !== undefined) return hit;
    const def = options.find((o) => String(o) === String(param.default));
    if (def !== undefined) return def;
    return options[0] ?? "";
  }
  const raw = typeof value === "number" ? value : typeof value === "string" ? Number(value) : Number.NaN;
  let out = Number.isFinite(raw) ? raw : defaultNumber(param);
  if (param.type === "int") out = Math.round(out);
  const min = finiteOrNull(param.min);
  const max = finiteOrNull(param.max);
  if (min !== null) out = Math.max(min, out);
  if (max !== null) out = Math.min(max, out);
  return out;
}

/** 是否已经越过临界值（方向由 `beyond` 决定，与 `governingCritical` 同口径）。 */
export function isBeyond(beyond: BeyondDirection, value: number, critical: number): boolean {
  if (!Number.isFinite(value)) return false;
  if (beyond === "below") return value < critical;
  if (beyond === "magnitude") return Math.abs(value) > Math.abs(critical);
  return value > critical;
}

/** 临界值在 `[min, max]` 上的位置；范围缺失或退化时为 `null`。 */
function markFraction(value: number, min: number | null, max: number | null): number | null {
  if (min === null || max === null || !(max > min)) return null;
  return Math.min(1, Math.max(0, (value - min) / (max - min)));
}

/** 该可视化是否有自适应坐标轴；有则返回提示文案（规则 1/4）。 */
export function autoscaleNote(viz: VizSpecJson): string | null {
  const autoscaled = viz.axes.filter((a) => a.autoscaled).map((a) => a.id);
  if (autoscaled.length === 0) return null;
  return (
    `坐标轴 ${autoscaled.join("、")} 是自适应的：纵轴会跟着数据缩放，` +
    "发散/衰减在这种轴上可能看起来是平的（规则 1/4）"
  );
}

/** 单个参数的提示文案（单位 / 临界 / 越界）。 */
function paramHints(param: ParamSpecJson, value: ParamValue, governing: number | null): string[] {
  const hints: string[] = [];
  if (param.unit !== undefined && param.unit !== "") hints.push(`单位：${param.unit}`);
  const critical = [...(param.critical ?? [])].sort((a, b) => a - b);
  if (critical.length > 0) {
    const beyond = param.beyond ?? "above";
    hints.push(
      `理论临界 ${critical.join(" / ")}（${BEYOND_TEXT[beyond]}` +
        (governing !== null ? `；当前支配 ${governing}` : "") +
        "）",
    );
    if (param.criticalNote !== undefined && param.criticalNote !== "") hints.push(param.criticalNote);
    if (governing !== null && typeof value === "number" && isBeyond(beyond, value, governing)) {
      hints.push(`⚠ 当前值 ${value} 已越过 ${governing}：${param.criticalNote ?? "结论可能不再成立"}`);
    }
  }
  return hints;
}

/**
 * 从一个 viz 的 manifest 生成控件描述表（顺序与 manifest 一致）。
 *
 * `params` 给出当前值（通常来自 `defaultParams(viz)` 或分享链接）；缺省用默认值。
 * 返回的每个 `ControlSpec` 都自洽：`value` 一定在域内、`hints` 一定非空（至少有单位或临界说明）。
 */
export function describeControls(viz: VizSpecJson, params?: ParamValues): ControlSpec[] {
  const autoscaled = viz.axes.some((a) => a.autoscaled);
  const autoNote = autoscaleNote(viz);
  return viz.params.map((param) => {
    const raw = params === undefined ? undefined : params[param.name];
    const value = clampParam(param, raw === undefined ? param.default : raw);
    const defaultValue = clampParam(param, param.default);
    const critical = [...(param.critical ?? [])].sort((a, b) => a - b);
    const beyond = param.beyond ?? "above";
    const numeric = typeof value === "number" ? value : Number.NaN;
    // governingCritical 要的是"当前值"；bool/enum 没有数值语义，按 undefined 处理（取最紧的那条）。
    const governing =
      param.type === "int" || param.type === "float"
        ? governingCritical(param, Number.isFinite(numeric) ? numeric : undefined)
        : governingCritical(param);
    const min = finiteOrNull(param.min);
    const max = finiteOrNull(param.max);
    const marks: CriticalMark[] = critical.map((c) => ({
      value: c,
      fraction: markFraction(c, min, max),
      label: String(c),
      governing: governing !== null && c === governing,
      crossed: Number.isFinite(numeric) && isBeyond(beyond, numeric, c),
    }));

    const hints = paramHints(param, value, governing);
    if (autoscaled && autoNote !== null) hints.push(autoNote);

    return {
      name: param.name,
      label: param.label,
      type: param.type,
      value,
      defaultValue,
      min,
      max,
      step: finiteOrNull(param.step),
      unit: param.unit ?? "",
      options: (param.options ?? []).map(String),
      critical,
      criticalNote: param.criticalNote ?? "",
      criticalMarks: marks,
      governing,
      beyond,
      crossed: governing !== null && Number.isFinite(numeric) && isBeyond(beyond, numeric, governing),
      autoscaled,
      hints,
    };
  });
}

// --------------------------------------------------------------------------- //
// DOM 部分（薄壳：一个 ControlSpec 一个控件）
// --------------------------------------------------------------------------- //

/** 参数变化回调：`(参数名, 新值, 全量参数)`。 */
export type ControlsChangeHandler = (name: string, value: ParamValue, params: ParamValues) => void;

function inputKind(spec: ControlSpec): "range" | "number" {
  return spec.min !== null && spec.max !== null && spec.max > spec.min ? "range" : "number";
}

function stepOf(spec: ControlSpec): number {
  if (spec.step !== null && spec.step > 0) return spec.step;
  if (spec.type === "int") return 1;
  if (spec.min !== null && spec.max !== null && spec.max > spec.min) return (spec.max - spec.min) / 200;
  return 0.001;
}

/**
 * 把控件挂到 `parent` 下（先清空）。
 *
 * 每个 `ControlSpec` 生成一行：`<label>` + 一个 `<input>`（range/number/checkbox）或 `<select>`，
 * 加上提示文案。任何输入都先过 `clampParam` 再回调，所以 `onChange` 拿到的值一定合法。
 * 返回创建出来的容器元素（方便调用方再挂样式或断言）。
 */
export function mountControls(
  viz: VizSpecJson,
  parent: HTMLElement,
  onChange: ControlsChangeHandler,
  params?: ParamValues,
): HTMLElement {
  const specs = describeControls(viz, params);
  const current: ParamValues = {};
  for (const s of specs) current[s.name] = s.value;

  const root = document.createElement("form");
  root.className = "nl-controls";
  root.setAttribute("data-viz", viz.id);
  root.addEventListener("submit", (ev) => ev.preventDefault());

  for (const spec of specs) {
    const row = document.createElement("div");
    row.className = "nl-control";
    row.setAttribute("data-param", spec.name);

    const label = document.createElement("label");
    label.textContent = spec.label;
    label.htmlFor = `nl-${viz.id}-${spec.name}`;
    row.appendChild(label);

    const param = viz.params.find((p) => p.name === spec.name);
    if (param === undefined) continue;
    const commit = (raw: unknown): void => {
      const value = clampParam(param, raw);
      current[spec.name] = value;
      onChange(spec.name, value, { ...current });
    };

    if (spec.type === "enum") {
      const select = document.createElement("select");
      select.id = label.htmlFor;
      for (const option of spec.options) {
        const el = document.createElement("option");
        el.value = option;
        el.textContent = option;
        el.selected = option === String(spec.value);
        select.appendChild(el);
      }
      select.addEventListener("change", () => commit(select.value));
      row.appendChild(select);
    } else if (spec.type === "bool") {
      const input = document.createElement("input");
      input.type = "checkbox";
      input.id = label.htmlFor;
      input.checked = spec.value === true;
      input.addEventListener("change", () => commit(input.checked));
      row.appendChild(input);
    } else {
      const input = document.createElement("input");
      input.type = inputKind(spec);
      input.id = label.htmlFor;
      if (spec.min !== null) input.min = String(spec.min);
      if (spec.max !== null) input.max = String(spec.max);
      input.step = String(stepOf(spec));
      input.value = String(spec.value);
      const readout = document.createElement("output");
      readout.textContent = `${spec.value}${spec.unit}`;
      input.addEventListener("input", () => {
        const value = clampParam(param, Number(input.value));
        readout.textContent = `${value}${spec.unit}`;
        commit(value);
      });
      row.appendChild(input);
      row.appendChild(readout);
    }

    if (spec.hints.length > 0) {
      const hint = document.createElement("p");
      hint.className = "nl-hint";
      hint.textContent = spec.hints.join("；");
      row.appendChild(hint);
    }
    root.appendChild(row);
  }

  parent.replaceChildren(root);
  return root;
}
