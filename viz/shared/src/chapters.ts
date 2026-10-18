/**
 * 目录（侧边栏 / 路由）的类型与工具。
 *
 * 数据来自 `chapters.gen.ts`（由 `tools/gen_manifests.py` 生成），真源是
 * `docs/chapters.yaml` + 各章模块的 `MANIFEST`。这里只放类型与纯函数。
 */

/** 生成物里的一条目录项（刻意不含 hasModule / notesFile 这类构建期信息）。 */
export interface ChapterIndexEntry {
  chapter: string;
  title: string;
  part: string;
  summary: string;
  nViz: number;
  vizzes: string[];
}

export const PART_TITLES: Readonly<Record<string, string>> = {
  "": "自检",
  I: "第一部分 · 线性代数与矩阵计算",
  II: "第二部分 · 插值、逼近与积分",
  III: "第三部分 · 微分方程与时间推进",
  IV: "第四部分 · 非线性方程与优化",
  V: "第五部分 · 特征值、随机与谱方法",
  VI: "第六部分 · 工程打包与发布",
};

/** 按分部分组，保持原顺序。 */
export function groupByPart(entries: readonly ChapterIndexEntry[]): { part: string; title: string; items: ChapterIndexEntry[] }[] {
  const out: { part: string; title: string; items: ChapterIndexEntry[] }[] = [];
  for (const e of entries) {
    const last = out[out.length - 1];
    if (last && last.part === e.part) last.items.push(e);
    else out.push({ part: e.part, title: PART_TITLES[e.part] ?? e.part, items: [e] });
  }
  return out;
}

/** 章节号排序键（"1.10" 排在 "1.9" 之后，而不是字典序）。 */
export function compareChapter(a: string, b: string): number {
  const pa = a.split(".").map((x) => Number.parseInt(x, 10));
  const pb = b.split(".").map((x) => Number.parseInt(x, 10));
  for (let i = 0; i < Math.max(pa.length, pb.length); i++) {
    const d = (pa[i] ?? 0) - (pb[i] ?? 0);
    if (d !== 0) return d;
  }
  return 0;
}
