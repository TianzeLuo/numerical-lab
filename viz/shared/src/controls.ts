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
import { renderRichText } from "./math.js";
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
  /** 面向初学者的展开解释（manifest `help`）。 */
  help: string;
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
  return "纵轴会自动缩放（自适应）：发散或衰减在这种轴上可能看起来是平的；比较绝对大小时先看刻度。";
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
      help: param.help ?? "",
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
 * manifest 没写 `help` 时的兜底说明（仍然给出范围与操作方式，不冒充物理解释）。
 * 真源章节应当逐个参数提供 `help`；这里只防止旧数据出现“点了没反应”的空展开框。
 */
function fallbackHelp(spec: ControlSpec): string {
  if (spec.type === "enum") {
    return `在 ${spec.options.join(" / ")} 中选择一种方案。不同选项会改变计算路线或矩阵族，图表和数值会一起更新。默认是 ${String(spec.defaultValue)}。`;
  }
  if (spec.type === "bool") return "打开或关闭这一选项；图与数值会随开关重新计算。";
  const lo = spec.min === null ? "未限定" : numberText(spec.min);
  const hi = spec.max === null ? "未限定" : numberText(spec.max);
  return `允许范围 ${lo} 到 ${hi}${spec.unit === "" ? "" : `，单位 ${spec.unit}`}。拖动滑轨可以快速扫过整个范围；右侧数字框可以键入精确值，按 Enter 生效。默认是 ${numberText(Number(spec.defaultValue))}。`;
}

/** 控件里的数字写法：整数直出，小数最多 6 位有效数字。 */
function numberText(value: number): string {
  if (!Number.isFinite(value)) return "—";
  if (Number.isInteger(value)) return String(value);
  return String(Number(value.toPrecision(6)));
}

function unitText(value: number, unit: string): string {
  if (unit === "") return numberText(value);
  const tight = unit === "%" || unit === "°" || unit === "‰";
  return `${numberText(value)}${tight ? "" : " "}${unit}`;
}

function pctOf(spec: ControlSpec, value: number): number | null {
  if (spec.min === null || spec.max === null || !(spec.max > spec.min)) return null;
  return Math.min(100, Math.max(0, ((value - spec.min) / (spec.max - spec.min)) * 100));
}

interface RenderedControl {
  spec: ControlSpec;
  set(value: ParamValue, emit: boolean): void;
}

/**
 * 把控件挂到 `parent` 下（先清空）。
 *
 * 视觉层是「标题 + 当前值 + 轨道 + 临界刻度」：滑块拖动时只更新读数与刻度状态，
 * 不重建 DOM（重建会让拖动中断）。`spec.hints` 仍然保留在纯函数接口里供测试与其它
 * 调用方使用；这里只挑最必要的信息展示，长解释收进可展开的说明，避免侧栏变成一堵墙。
 */
export interface ControlsHandle {
  root: HTMLElement;
  /** 把所有参数恢复到 manifest 的默认值，并只提交一次变更。 */
  reset(): void;
}

export function mountControls(
  viz: VizSpecJson,
  parent: HTMLElement,
  onChange: ControlsChangeHandler,
  params?: ParamValues,
): ControlsHandle {
  const specs = describeControls(viz, params);
  const current: ParamValues = {};
  for (const s of specs) current[s.name] = s.value;

  const root = document.createElement("form");
  root.className = "nl-controls";
  root.setAttribute("data-viz", viz.id);
  root.addEventListener("submit", (ev) => ev.preventDefault());

  const rows: RenderedControl[] = [];

  for (const spec of specs) {
    const param = viz.params.find((p) => p.name === spec.name);
    if (param === undefined) continue;

    const row = document.createElement("div");
    row.className = "nl-control";
    row.setAttribute("data-param", spec.name);

    const head = document.createElement("div");
    head.className = "nl-control-head";
    const label = document.createElement("label");
    label.className = "nl-control-label";
    label.textContent = spec.label;
    label.htmlFor = `nl-${viz.id}-${spec.name}`;
    const valueBox = document.createElement("span");
    valueBox.className = "nl-control-value-box";

    // 数值参数直接给可编辑输入框：滑轨适合快速扫范围，尾数/相邻 ulp 这类
    // 需要很多有效数字的值只能靠键入。输入框与滑轨共用同一份 `current`，
    // 键入走 clampParam，非法输入恢复为上一次的值，不会把 NaN 发给后端。
    const editable = spec.type === "int" || spec.type === "float";
    const editor: HTMLInputElement | null = editable ? document.createElement("input") : null;
    let output: HTMLElement;
    if (editor !== null) {
      editor.type = "text";
      editor.inputMode = "decimal";
      editor.spellcheck = false;
      editor.autocomplete = "off";
      editor.className = "nl-control-value nl-control-value-input";
      editor.id = label.htmlFor;
      editor.setAttribute("aria-label", `${spec.label}：可直接输入精确数值`);
      output = editor;
    } else {
      const textOut = document.createElement("output");
      textOut.className = "nl-control-value";
      output = textOut;
    }
    if (editor !== null) editor.setAttribute("aria-describedby", `${label.htmlFor}-note`);
    valueBox.append(output);
    if (spec.unit !== "") {
      valueBox.append(Object.assign(document.createElement("span"), {
        className: "nl-control-unit",
        textContent: spec.unit,
      }));
    }
    // 枚举与开关的状态已经由分段按钮 / 开关本身显示，右上角再重复一遍只是噪声。
    if (!editable) valueBox.hidden = true;
    head.append(label, valueBox);
    row.append(head);

    const showText = (text: string): void => {
      if (editor !== null) editor.value = text;
      else output.textContent = text;
    };

    const commit = (raw: unknown): void => {
      const value = clampParam(param, raw);
      current[spec.name] = value;
      onChange(spec.name, value, { ...current });
    };

    let setVisual = (_value: ParamValue): void => {};

    if (spec.type === "enum") {
      const group = document.createElement("div");
      group.className = "nl-segmented";
      group.setAttribute("role", "group");
      group.setAttribute("aria-label", spec.label);
      const buttons: HTMLButtonElement[] = [];
      for (const option of spec.options) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "nl-segment";
        btn.textContent = option;
        btn.title = option;
        btn.setAttribute("aria-label", `${spec.label}：${option}`);
        btn.addEventListener("click", () => {
          const value = clampParam(param, option);
          current[spec.name] = value;
          setVisual(value);
          commit(option);
        });
        buttons.push(btn);
        group.append(btn);
      }
      row.append(group);
      setVisual = (value) => {
        for (const [i, btn] of buttons.entries()) {
          const on = spec.options[i] === String(value);
          btn.classList.toggle("is-active", on);
          btn.setAttribute("aria-pressed", on ? "true" : "false");
        }
        showText(String(value));
      };
      showText(String(spec.value));
    } else if (spec.type === "bool") {
      const wrap = document.createElement("label");
      wrap.className = "nl-switch";
      const input = document.createElement("input");
      input.type = "checkbox";
      input.id = label.htmlFor;
      input.checked = spec.value === true;
      const track = document.createElement("span");
      track.className = "nl-switch-track";
      track.setAttribute("aria-hidden", "true");
      const knob = document.createElement("span");
      knob.className = "nl-switch-knob";
      track.append(knob);
      const state = document.createElement("span");
      state.className = "nl-switch-state";
      wrap.append(input, track, state);
      input.addEventListener("change", () => {
        const value = input.checked;
        current[spec.name] = value;
        setVisual(value);
        commit(value);
      });
      row.append(wrap);
      setVisual = (value) => {
        const on = value === true;
        input.checked = on;
        state.textContent = on ? "开启" : "关闭";
        showText(on ? "开" : "关");
      };
      showText(spec.value === true ? "开" : "关");
    } else {
      const kind = inputKind(spec);
      const trackWrap = document.createElement("div");
      trackWrap.className = "nl-slider-wrap";

      const input = document.createElement("input");
      input.className = kind === "range" ? "nl-range" : "nl-number";
      input.type = kind;
      if (spec.min !== null) input.min = String(spec.min);
      if (spec.max !== null) input.max = String(spec.max);
      input.step = String(stepOf(spec));
      input.value = String(spec.value);
      input.setAttribute("aria-label", `${spec.label}（滑轨）`);
      input.setAttribute("aria-describedby", `${label.htmlFor}-note`);
      trackWrap.append(input);

      const marks: { value: number; node: HTMLButtonElement }[] = [];
      if (kind === "range") {
        const markLayer = document.createElement("div");
        markLayer.className = "nl-marks";
        markLayer.setAttribute("aria-hidden", "true");
        for (const mark of spec.criticalMarks) {
          if (mark.fraction === null) continue;
          const tick = document.createElement("button");
          tick.type = "button";
          tick.className = "nl-mark";
          tick.tabIndex = -1;
          tick.style.left = `${(mark.fraction * 100).toFixed(3)}%`;
          tick.title = `理论临界 ${mark.label}${spec.unit === "" ? "" : ` ${spec.unit}`}（点击跳到该值）`;
          tick.setAttribute("aria-label", `设为理论临界值 ${mark.label}`);
          tick.addEventListener("click", () => {
            const value = clampParam(param, mark.value);
            current[spec.name] = value;
            input.value = String(value);
            setVisual(value);
            commit(value);
          });
          markLayer.append(tick);
          marks.push({ value: mark.value, node: tick });
        }
        trackWrap.append(markLayer);
      }
      row.append(trackWrap);

      if (kind === "range" && spec.min !== null && spec.max !== null) {
        const limits = document.createElement("div");
        limits.className = "nl-range-limits";
        limits.setAttribute("aria-hidden", "true");
        limits.append(
          Object.assign(document.createElement("span"), { textContent: numberText(spec.min) }),
          Object.assign(document.createElement("span"), { textContent: numberText(spec.max) }),
        );
        row.append(limits);
      }

      const note = document.createElement("p");
      note.className = "nl-control-note";
      note.id = `${label.htmlFor}-note`;
      const criticalText = spec.critical.length > 0 ? `理论临界 ${spec.critical.map((c) => numberText(c)).join(" / ")}` : "";
      note.textContent = criticalText;
      if (criticalText === "") note.hidden = true;
      row.append(note);

      if (spec.criticalNote !== "") {
        const more = document.createElement("details");
        more.className = "nl-control-more";
        const summary = document.createElement("summary");
        summary.textContent = "这个临界值意味着什么";
        const body = document.createElement("div");
        body.className = "nl-control-more-body";
        body.innerHTML = renderRichText(spec.criticalNote);
        more.append(summary, body);
        row.append(more);
      }

      input.addEventListener("input", () => {
        const value = clampParam(param, input.value);
        current[spec.name] = value;
        setVisual(value);
        commit(value);
      });

      if (editor !== null) {
        editor.addEventListener("focus", () => editor.select());
        editor.addEventListener("change", () => {
          const raw = editor.value.trim();
          const parsed = raw === "" ? Number.NaN : Number(raw);
          if (!Number.isFinite(parsed)) {
            setVisual(current[spec.name] ?? spec.value);
            return;
          }
          const value = clampParam(param, parsed);
          setVisual(value);
          commit(value);
        });
        editor.addEventListener("keydown", (ev) => {
          if (ev.key === "Enter") {
            ev.preventDefault();
            editor.blur();
          } else if (ev.key === "Escape") {
            ev.preventDefault();
            setVisual(current[spec.name] ?? spec.value);
            editor.blur();
          } else if (ev.key === "ArrowUp" || ev.key === "ArrowDown") {
            ev.preventDefault();
            const now = Number(current[spec.name]);
            if (!Number.isFinite(now)) return;
            const delta = stepOf(spec) * (ev.key === "ArrowUp" ? 1 : -1);
            const value = clampParam(param, now + delta);
            setVisual(value);
            commit(value);
          }
        });
      }

      setVisual = (value) => {
        const numeric = typeof value === "number" ? value : Number(value);
        if (!(typeof value === "number") || !Number.isFinite(value)) {
          showText(String(value));
          return;
        }
        input.value = String(value);
        // 输入框保留完整往返写法（例如 0.999999、1e-12），不用 6 位有效数字截断；
        // 参数值本身仍由 clampParam 校验，显示不改变数值。
        showText(String(value));
        if (editor !== null) editor.title = unitText(value, spec.unit);
        const pct = pctOf(spec, numeric);
        if (pct !== null) input.style.setProperty("--fill", `${pct.toFixed(3)}%`);
        const governing = governingCritical(param, numeric);
        for (const mark of marks) {
          const crossed = governing !== null && isBeyond(spec.beyond, numeric, mark.value);
          mark.node.classList.toggle("is-crossed", crossed);
          mark.node.classList.toggle("is-governing", governing !== null && mark.value === governing);
        }
        const crossedNow = governing !== null && isBeyond(spec.beyond, numeric, governing);
        row.classList.toggle("is-crossed", crossedNow);
        if (criticalText !== "") {
          note.hidden = false;
          note.textContent = crossedNow
            ? `已越过临界值 ${numberText(governing as number)}`
            : criticalText;
        }
      };
    }

    if (spec.type !== "bool" && spec.type !== "enum") setVisual(spec.value);
    const helpText = spec.help.trim() !== "" ? spec.help : fallbackHelp(spec);
    if (helpText.trim() !== "") {
      const help = document.createElement("details");
      help.className = "nl-control-more nl-control-help";
      const summary = document.createElement("summary");
      summary.textContent = "这个参数是做什么的";
      const body = document.createElement("div");
      body.className = "nl-control-more-body";
      body.innerHTML = renderRichText(helpText);
      help.append(summary, body);
      row.append(help);
    }
    rows.push({ spec, set: (value, emit) => {
      current[spec.name] = value;
      setVisual(value);
      if (emit) onChange(spec.name, value, { ...current });
    } });
    root.append(row);
  }

  if (rows.length === 0) {
    root.append(Object.assign(document.createElement("p"), {
      className: "nl-controls-empty",
      textContent: "这个实验没有可调参数。",
    }));
  }

  const resetAll = (): void => {
    for (const row of rows) row.set(row.spec.defaultValue, false);
    const first = rows[0];
    if (first !== undefined) onChange(first.spec.name, current[first.spec.name], { ...current });
  };

  const foot = document.createElement("div");
  foot.className = "nl-controls-foot";
  const reset = document.createElement("button");
  reset.type = "button";
  reset.className = "nl-text-btn";
  reset.textContent = "恢复全部默认值";
  reset.addEventListener("click", resetAll);
  const hint = document.createElement("span");
  hint.className = "nl-controls-hint";
  const numeric = specs.some((s) => s.type === "int" || s.type === "float");
  hint.textContent = [
    numeric ? "滑轨用于快速扫范围；右侧数字框可直接键入精确值，↑↓ 微调。" : "",
    specs.some((s) => s.critical.length > 0) ? "滑轨上的细线标出理论临界值，点击可直接跳过去。" : "",
  ]
    .filter((line) => line !== "")
    .join(" ");
  foot.append(reset, hint);
  root.append(foot);

  if (specs.some((s) => s.autoscaled)) {
    const auto = document.createElement("p");
    auto.className = "nl-autoscale-note";
    const mark = document.createElement("span");
    mark.className = "nl-autoscale-mark";
    mark.textContent = "↕";
    const text = document.createElement("span");
    text.textContent = "误差副图的纵轴会随数据自动缩放，适合看趋势；比较不同参数下的绝对误差时，请留意刻度变化。";
    auto.append(mark, text);
    root.append(auto);
  }

  parent.replaceChildren(root);
  return { root, reset: resetAll };
}
