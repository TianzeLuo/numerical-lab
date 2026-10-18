/**
 * `.nlf` 帧容器（`docs/PROTOCOLS.md` §1）的 TypeScript 数据模型。
 *
 * 本文件只回答"一帧是什么"，字节级编解码在 `nlf.ts`。把它单独拆出来是为了让
 * 各章节 model / view 只依赖一个**不含解码逻辑**的纯数据类（可被 canvas 直接消费）。
 *
 * 三条设计决定（写下来是因为它们会被调用方看见）：
 *
 * 1. **内部一律按 C 序（行主序）连续存储**。`.nlf` 头部里每个数组保留它写盘时的
 *    `order` 字段（`specs[i].order`），列主序（`"F"`）数据在**解码时重排一次**，
 *    因此 `array()` 返回的永远是行主序缓冲；`matrix()` 的行列也据此给出。
 *    取这个决定的理由：渲染与物理代码只需要一种轴序，把 F→C 的重排压在解码边界上，
 *    下游（heatmap/contour/vectorfield）就不必各自处理两种顺序，也就不会各自出错。
 * 2. 数组数据**零拷贝**暴露（`array()` 直接返回内部缓冲区）。一帧可能是 8 MB，
 *    每次访问都复制一遍是无法接受的；作为交换，调用方 MUST NOT 就地修改返回值
 *    （TypeScript 的 `TypedArray` 无法在类型上表达只读）。
 * 3. `specs` / `shape` 在构造时 `Object.freeze`：布局元信息一旦被下游改掉，
 *    `nbytes`/`offset` 与数据的对应关系就悄悄失效了，冻结让它立刻抛错而不是静默出错。
 */

/** 数组元素类型代号，与 `server/nlf.py::DTYPE_CODES` 一一对应。 */
export type Dtype = "f8" | "f4" | "i8" | "i4" | "i2" | "u1" | "u2" | "u4" | "bool";

/**
 * dtype → 每元素字节数。
 *
 * 写成表而不是散落在各处的 switch：解码端要用它做 `nbytes == size × itemsize`
 * 的自洽校验，编码端要用它算 payload 布局；两处必须永远一致，共用一张表最省心。
 */
export const DTYPE_ITEMSIZE: Record<Dtype, number> = {
  f8: 8,
  f4: 4,
  i8: 8,
  i4: 4,
  i2: 2,
  u1: 1,
  u2: 2,
  u4: 4,
  bool: 1,
};

/** 一帧里单个数组的元信息（不含数据本体）。 */
export interface ArraySpec {
  name: string;
  dtype: Dtype;
  shape: number[];
  order: "C" | "F";
  offset: number;
  nbytes: number;
  unit: string;
  label: string;
}

/**
 * 帧内数组的运行时表示。
 * `bool` 与 `u1` 都是 `Uint8Array`（bool 只允许 0/1），`i8` 用 `BigInt64Array`
 * （JS 的 number 装不下 int64 全量程，用 Float64 兜会把大整数悄悄改值）。
 */
export type TypedArray =
  | Float64Array
  | Float32Array
  | BigInt64Array
  | Int32Array
  | Int16Array
  | Uint8Array
  | Uint16Array
  | Uint32Array;

/** 帧种类：`dataset` 用于对拍基准文件，`manifest` 用于把 manifest 走同一条通道。 */
export type FrameKind = "frame" | "dataset" | "manifest";

/** payload 的压缩方式；`header` 恒为明文。 */
export type FrameCodec = "none" | "zlib" | "gzip";

/** `.nlf` 容器格式错误——解码端**唯一**会抛出的异常类型（绝不返回可疑数据）。 */
export class NlfError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "NlfError";
  }
}

/** 构造一帧所需的全部字段（全部可选，缺省给空值，便于测试只造关心的部分）。 */
export interface FrameInit {
  chapter?: string;
  viz?: string;
  kind?: FrameKind;
  codec?: FrameCodec;
  meta?: Record<string, number | boolean | null>;
  strings?: Record<string, string>;
  critical?: Record<string, number | number[]>;
  pitfalls?: string;
  /** 写帧者信息；解码时原样保留，重新编码时不丢（便于 G7 对拍与追溯）。 */
  generator?: Record<string, string>;
  /** `nlf.py` 支持的附加标注；本实现只透传，不解释。 */
  annotations?: Array<Record<string, unknown>>;
  /** 与 `arrays` 一一对应的元信息（顺序即 payload 中的排列顺序）。 */
  specs?: ArraySpec[];
  /** 数组数据本体，键为数组名；值必须是 C 序（行主序）连续缓冲。 */
  arrays?: Map<string, TypedArray>;
}

/**
 * 一帧数值数据：若干命名数组 + 标量元数据 + 防误导文本。
 *
 * 构造函数对所有调用方开放（解码器与测试都需要造帧），但业务代码只应**消费**
 * 后端帧：前端自造数值会直接违反 `docs/POLICY.md` P8/P9。
 */
export class Frame {
  readonly chapter: string;
  readonly viz: string;
  readonly kind: FrameKind;
  readonly codec: FrameCodec;
  /** 只放标量（number / boolean / null）；数组一律走 `arrays`（PROTOCOLS §1.1）。 */
  readonly meta: Record<string, number | boolean | null>;
  readonly strings: Record<string, string>;
  readonly critical: Record<string, number | number[]>;
  readonly pitfalls: string;
  readonly generator: Record<string, string>;
  readonly annotations: Array<Record<string, unknown>>;
  readonly specs: ArraySpec[];

  /** 数组数据本体；私有，# 前缀保证外部只能经 `array()` 取。 */
  #arrays: Map<string, TypedArray>;

  constructor(init: FrameInit = {}) {
    this.chapter = init.chapter ?? "";
    this.viz = init.viz ?? "";
    this.kind = init.kind ?? "frame";
    this.codec = init.codec ?? "none";
    this.meta = { ...(init.meta ?? {}) };
    this.strings = { ...(init.strings ?? {}) };
    this.critical = { ...(init.critical ?? {}) };
    this.pitfalls = init.pitfalls ?? "";
    this.generator = { ...(init.generator ?? {}) };
    this.annotations = (init.annotations ?? []).map((a) => ({ ...a }));
    this.specs = Object.freeze(
      (init.specs ?? []).map((s) =>
        Object.freeze({ ...s, shape: Object.freeze([...s.shape]) as number[] }),
      ),
    ) as ArraySpec[];
    this.#arrays = new Map(init.arrays ?? []);
  }

  /** 数组名列表（顺序即 payload 中的排列顺序）。 */
  get names(): string[] {
    return this.specs.map((s) => s.name);
  }

  /**
   * 取数组数据。
   *
   * 返回值是**内部 C 序缓冲区的零拷贝视图**：F 序数据在解码时已经重排完毕，
   * 这里拿到的永远是行主序。调用方 MUST NOT 就地修改它（会改掉整帧）。
   * 找不到名字时抛 `NlfError`，消息里列出可用数组名——前端拼错名字时应当立刻知道。
   */
  array(name: string): TypedArray {
    const arr = this.#arrays.get(name);
    if (arr === undefined) {
      throw new NlfError(`帧中没有数组 '${name}'；现有：[${this.names.map((n) => `'${n}'`).join(", ")}]`);
    }
    return arr;
  }

  /**
   * 把 2 维数组按行主序取出：`rows × cols`，`values[r * cols + c]`。
   * 仅支持 2 维（热力图/等高线/向量场的输入形态）；其它维度抛 `NlfError`，
   * 避免调用方拿到一个语义不明的扁平缓冲。
   */
  matrix(name: string): { rows: number; cols: number; values: TypedArray } {
    const shape = this.shapeOf(name);
    if (shape.length !== 2) {
      throw new NlfError(
        `matrix() 只接受 2 维数组：'${name}' 的 shape=[${shape.join(", ")}] 是 ${shape.length} 维`,
      );
    }
    return { rows: shape[0], cols: shape[1], values: this.array(name) };
  }

  /**
   * 取 `meta` 里的标量（曲线量、残差、阶数……）。
   *
   * 与 `nlf.py::Frame.scalar` 同语义：键不存在时返回 `fallback`；键存在但值为
   * `null` 时返回 `undefined`；`boolean` 按 `1/0` 处理（与 Python `float(True)` 一致）。
   */
  scalar(key: string, fallback?: number): number | undefined {
    const raw = Object.prototype.hasOwnProperty.call(this.meta, key) ? this.meta[key] : fallback;
    if (raw === undefined || raw === null) return undefined;
    return typeof raw === "number" ? raw : Number(raw);
  }

  /** 是否存在某个数组（供可选的辅助量探测，例如 `has("error")`）。 */
  has(name: string): boolean {
    return this.#arrays.has(name);
  }

  /** 取数组形状；不存在时抛 `NlfError`（与 `nlf.py::Frame.shape_of` 同语义）。 */
  shapeOf(name: string): number[] {
    for (const s of this.specs) {
      if (s.name === name) return s.shape;
    }
    throw new NlfError(`帧中没有数组 '${name}'`);
  }
}
