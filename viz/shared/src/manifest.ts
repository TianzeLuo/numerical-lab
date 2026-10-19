/**
 * 章节 manifest 的前端类型与运行时校验（与 `server/manifest.py` 一一对应）。
 *
 * 真源是 Python 章节模块里的 `MANIFEST`；`viz/chapters/<id>/manifest.gen.ts` 由
 * `tools/gen_manifests.py` 生成，CI 的 G6 门会检查生成物与真源不漂移。
 * 这里的类型只用于前端读，**不重复实现校验** —— 但凡前端要相信 manifest 的地方，
 * 都应当先过一遍 `validateManifestJson`，因为浏览器可能拿到的是旧的静态文件。
 */

export type ParamType = "int" | "float" | "bool" | "enum";
export type TimeMode = "none" | "stream" | "scrub";
export type DataSource = "backend" | "backend+preview";
export type AxisScale = "linear" | "log";
export type BeyondDirection = "above" | "below" | "magnitude";

export interface ParamSpecJson {
  name: string;
  label: string;
  type: ParamType;
  default: number | boolean | string;
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
  options?: Array<string | number>;
  /** 理论临界值（规则 6：显式给出，没有就写 `[]`）。 */
  critical: number[];
  criticalNote?: string;
  beyond?: BeyondDirection;
}

export interface AxisSpecJson {
  id: string;
  title?: string;
  xlabel?: string;
  ylabel?: string;
  unit?: string;
  scale?: AxisScale;
  /** 规则 1：纵轴是否自适应缩放——自适应会让发散"看起来是平的"，必须显式声明。 */
  autoscaled: boolean;
}

export interface TimeSpecJson {
  mode: TimeMode;
  tMax?: number;
  dt?: number;
  fps?: number;
  label?: string;
}

export interface VizSpecJson {
  id: string;
  title: string;
  description?: string;
  dataSource: DataSource;
  params: ParamSpecJson[];
  time: TimeSpecJson;
  /** 规则 3：必须含 error 或 residual。 */
  requiredAux: string[];
  axes: AxisSpecJson[];
  /** 规则 7：非空、≥60 字，说清"它会怎样骗你"。 */
  pitfalls: string;
  theory: string[];
}

export interface ChapterManifestJson {
  chapter: string;
  title: string;
  part?: string;
  summary?: string;
  vizzes: VizSpecJson[];
}

export const MIN_PITFALL_CHARS = 60;

const PARAM_TYPES: readonly string[] = ["int", "float", "bool", "enum"];
const TIME_MODES: readonly string[] = ["none", "stream", "scrub"];
const DATA_SOURCES: readonly string[] = ["backend", "backend+preview"];
const AXIS_SCALES: readonly string[] = ["linear", "log"];
const BEYOND: readonly string[] = ["above", "below", "magnitude"];

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

/**
 * 校验一份 manifest JSON，返回问题列表（空数组 = 合规）。
 * 规则与 `server/manifest.py::validate_manifest` 相同，措辞也尽量一致，便于对照排查。
 */
export function validateManifestJson(raw: unknown): string[] {
  const out: string[] = [];
  if (!isRecord(raw)) return ["manifest 必须是 dict"];
  const chapter = raw["chapter"];
  if (typeof chapter !== "string" || !/^\d+\.\d+$/.test(chapter)) {
    out.push(`chapter 必须是 "主.次" 形式，得到 ${JSON.stringify(chapter)}`);
  }
  if (typeof raw["title"] !== "string" || (raw["title"] as string).trim() === "") {
    out.push("title 必须是非空字符串");
  }
  const vizzes = raw["vizzes"];
  if (!Array.isArray(vizzes) || vizzes.length === 0) {
    out.push("vizzes 必须是非空列表");
    return out;
  }
  const seen = new Set<string>();
  vizzes.forEach((v, i) => {
    const where = `第 ${i + 1} 个可视化`;
    if (!isRecord(v)) {
      out.push(`${where}：必须是对象`);
      return;
    }
    const id = v["id"];
    if (typeof id !== "string" || !/^[a-z][a-z0-9_]*$/.test(id)) {
      out.push(`${where}：id 必须是 snake_case，得到 ${JSON.stringify(id)}`);
    } else if (seen.has(id)) {
      out.push(`${where}：可视化 id 重复：${id}`);
    } else {
      seen.add(id);
    }
    const tag = typeof id === "string" ? id : where;
    if (typeof v["title"] !== "string" || (v["title"] as string).trim() === "") out.push(`${tag}：title 不能为空`);
    if (typeof v["dataSource"] !== "string" || !DATA_SOURCES.includes(v["dataSource"])) {
      out.push(`${tag}：dataSource 必须是 ${DATA_SOURCES.join(" / ")} 之一`);
    }
    const params = v["params"];
    if (!Array.isArray(params) || params.length === 0) {
      out.push(`${tag}：params 必须是非空列表`);
    } else {
      const names = new Set<string>();
      for (const p of params) {
        if (!isRecord(p)) {
          out.push(`${tag}：参数必须是对象`);
          continue;
        }
        const pn = typeof p["name"] === "string" ? (p["name"] as string) : "";
        if (pn === "") out.push(`${tag}：参数 name 必须是非空字符串`);
        else if (names.has(pn)) out.push(`${tag}：参数名重复：${pn}`);
        else names.add(pn);
        if (typeof p["type"] !== "string" || !PARAM_TYPES.includes(p["type"])) {
          out.push(`${tag}.${pn}：type 必须是 ${PARAM_TYPES.join(" / ")} 之一`);
        }
        if (typeof p["label"] !== "string" || (p["label"] as string).trim() === "") {
          out.push(`${tag}.${pn}：label 不能为空`);
        }
        if (!("default" in p)) out.push(`${tag}.${pn}：缺少 default`);
        // 规则 6：critical 必须显式给出（空也写 []），非空时必须有 criticalNote
        if (!Array.isArray(p["critical"])) {
          out.push(`${tag}.${pn}：缺少 critical 列表——规则 6 要求显式给出（没有就写 []）`);
        } else {
          for (const c of p["critical"] as unknown[]) {
            if (typeof c !== "number" || !Number.isFinite(c)) out.push(`${tag}.${pn}：critical 的元素必须是有限数`);
          }
          if ((p["critical"] as unknown[]).length > 0) {
            const note = p["criticalNote"];
            if (typeof note !== "string" || note.trim() === "") {
              out.push(`${tag}.${pn}：critical 非空时必须写 criticalNote（说清越过它会怎样）`);
            }
          }
        }
        if (p["beyond"] !== undefined && (typeof p["beyond"] !== "string" || !BEYOND.includes(p["beyond"]))) {
          out.push(`${tag}.${pn}：beyond 必须是 ${BEYOND.join(" / ")} 之一`);
        }
      }
    }
    const time = v["time"];
    if (!isRecord(time)) {
      out.push(`${tag}：缺少 time`);
    } else if (typeof time["mode"] !== "string" || !TIME_MODES.includes(time["mode"])) {
      out.push(`${tag}：time.mode 必须是 ${TIME_MODES.join(" / ")} 之一`);
    } else if (time["mode"] === "none") {
      if (time["tMax"] !== undefined || time["dt"] !== undefined) {
        out.push(`${tag}：time.mode === "none" 时不得声明 tMax/dt`);
      }
    } else {
      const tMax = time["tMax"];
      const dt = time["dt"];
      if (typeof tMax !== "number" || !(tMax > 0)) out.push(`${tag}：time.tMax 必须为正数`);
      if (typeof dt !== "number" || !(dt > 0)) out.push(`${tag}：time.dt 必须为正数`);
      if (typeof tMax === "number" && typeof dt === "number" && dt > tMax) {
        out.push(`${tag}：time.dt 不应大于 tMax`);
      }
    }
    const aux = v["requiredAux"];
    if (!Array.isArray(aux) || aux.length === 0) out.push(`${tag}：requiredAux 必须是非空列表（规则 3）`);
    else if (!aux.some((a) => a === "error" || a === "residual")) {
      out.push(`${tag}：requiredAux 必须包含 error 或 residual——规则 3 不允许画一条没有误差信息的曲线`);
    }
    const axes = v["axes"];
    if (!Array.isArray(axes) || axes.length === 0) {
      out.push(`${tag}：axes 必须非空`);
    } else {
      const ids = new Set<string>();
      for (const ax of axes) {
        if (!isRecord(ax)) {
          out.push(`${tag}：axis 必须是对象`);
          continue;
        }
        const aid = typeof ax["id"] === "string" ? (ax["id"] as string) : "";
        if (aid === "") out.push(`${tag}：axis.id 必须是非空字符串`);
        else if (ids.has(aid)) out.push(`${tag}：axis id 重复：${aid}`);
        else ids.add(aid);
        if (typeof ax["autoscaled"] !== "boolean") {
          out.push(`${tag}.${aid}：必须显式声明 autoscaled（规则 1/4）`);
        }
        if (ax["scale"] !== undefined && (typeof ax["scale"] !== "string" || !AXIS_SCALES.includes(ax["scale"]))) {
          out.push(`${tag}.${aid}：scale 必须是 ${AXIS_SCALES.join(" / ")} 之一`);
        }
      }
    }
    const pitfalls = v["pitfalls"];
    if (typeof pitfalls !== "string" || pitfalls.trim().length < MIN_PITFALL_CHARS) {
      out.push(
        `${tag}：pitfalls 至少 ${MIN_PITFALL_CHARS} 字（现在 ${typeof pitfalls === "string" ? pitfalls.trim().length : 0} 字）——规则 7`,
      );
    }
    const theory = v["theory"];
    if (!Array.isArray(theory) || theory.length === 0 || !theory.every((t) => typeof t === "string" && t.trim() !== "")) {
      out.push(`${tag}：theory 必须是非空字符串列表`);
    }
  });
  return out;
}

/** 参数默认值表（前端控件初始化用）。 */
export function defaultParams(viz: VizSpecJson): Record<string, number | boolean | string> {
  const out: Record<string, number | boolean | string> = {};
  for (const p of viz.params) out[p.name] = p.default;
  return out;
}

/**
 * 该可视化的"支配临界值"：**最近越过的那条线**（与 `server/frames.py::governing_critical` 同一口径）。
 *
 * - `above`：已越过的阈值里最大的那个；一条都没越过 → 最小的阈值（下一个目标）；
 * - `below`：已越过的阈值里最小的那个；一条都没越过 → 最大的阈值；
 * - `magnitude`：绝对值最接近 `value` 的那个。
 */
export function governingCritical(param: ParamSpecJson, value?: number): number | null {
  const crit = [...param.critical].sort((a, b) => a - b);
  if (crit.length === 0) return null;
  const beyond = param.beyond ?? "above";
  if (value === undefined || !Number.isFinite(value)) return beyond === "below" ? crit[crit.length - 1]! : crit[0]!;
  if (beyond === "above") {
    let best: number | null = null;
    for (const c of crit) if (value > c) best = c;
    return best ?? crit[0]!;
  }
  if (beyond === "below") {
    for (const c of crit) if (value < c) return c;
    return crit[crit.length - 1]!;
  }
  let best = crit[0]!;
  for (const c of crit) if (Math.abs(Math.abs(c) - Math.abs(value)) < Math.abs(Math.abs(best) - Math.abs(value))) best = c;
  return best;
}
