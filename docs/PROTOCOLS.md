# PROTOCOLS —— 帧格式 / 场景契约 / 可视化契约 / 对拍协议

> 本文件定义**跨语言、跨进程、跨时间的接口**。任何一端改动都必须同步本文件与两侧实现，
> 并由 `tests/test_nlf_roundtrip.py`、`viz/tests/nlf.test.ts`、`tests/parity/` 三处守住。

目录：
1. [.nlf 帧容器](#1-nlf-帧容器)
2. [WebSocket 流式协议](#2-websocket-流式协议)
3. [HTTP API](#3-http-api)
4. [静态预计算与降级](#4-静态预计算与降级)
5. [Manifest（可视化契约源）](#5-manifest可视化契约源)
6. [Scene 契约（渲染与视觉回归）](#6-scene-契约渲染与视觉回归)
7. [Python ↔ TypeScript 对拍协议](#7-python--typescript-对拍协议)
8. [断言报告协议](#8-断言报告协议)
9. [产物目录](#9-产物目录)

---

## 1. `.nlf` 帧容器

**NLF = Numerical Lab Frame**，v1。设计目标：零依赖、可 mmap 式随机读、头部人类可读、字节序固定。

```
偏移 0   : 4 字节   magic = b"NLF1"
偏移 4   : uint32LE  header_len
偏移 8   : header_len 字节  UTF-8 JSON（无 BOM，无尾随空白）
偏移 8+H : payload     = 各 array 的连续缓冲（little-endian，按 arrays 顺序紧排）
```

### 1.1 header JSON

```json
{
  "version": 1,
  "kind": "frame",
  "chapter": "1.2",
  "viz": "lu_heatmap",
  "codec": "none",
  "endian": "little",
  "generator": {"tool": "server/nlf.py", "python": "3.13.9", "numpy": "2.3.5"},
  "meta":    {"t": 0.3, "h": 0.01, "residual": 1.2e-13},
  "strings": {"status": "ok", "theory_order": "2"},
  "critical": {"r": 0.5},
  "pitfalls": "固定纵轴时它看起来在收敛……",
  "arrays": [
    {"name": "U", "dtype": "f8", "shape": [64, 64], "order": "C",
     "offset": 0, "nbytes": 32768, "unit": "", "label": "U 因子"}
  ]
}
```

约束（解码端 MUST 校验，违反即抛 `NlfError`）：

| 字段 | 取值 | 说明 |
|---|---|---|
| `version` | `1` | 不兼容升级才改 |
| `kind` | `frame` \| `dataset` \| `manifest` | `dataset` 用于对拍基准文件 |
| `codec` | `none` \| `zlib` \| `gzip` | 只压缩 payload，header 恒为明文 |
| `endian` | `little` | 固定 |
| `dtype` | `f8 f4 i8 i4 i2 u1 u2 u4 bool` | **无 `f2`**：浏览器侧没有统一的半精度类型（`Float16Array` 尚未普及），半精度只会带来"两侧行为不一致"的风险；要省带宽请用 `codec`，不要新造 dtype |
| `shape` | 长度 1–3 的整数数组 | 3 维仅用于 RGB/体数据 |
| `order` | `C` \| `F` | 仅对 ≥2 维有意义 |
| `offset`+`nbytes` | 必须与 payload 长度自洽、互不重叠 | 解码时校验 |
| `meta` | 仅允许 JSON number / bool / null | 标量曲线量（残差、阶、时间…） |
| `strings` | 字符串标量 | 供 UI 徽标、状态提示 |
| `critical` | 参数名 → 理论临界值 | 规则 6 |
| `pitfalls` | 非空字符串 | 规则 7 |

**禁止**在 `meta` 里塞数组；数组一律走 `arrays`。这是为了让"前端自算物理量"无处藏身。

### 1.2 参考实现

- Python（写读双端，权威实现）：`server/nlf.py` —— `pack_frame(fr: Frame) -> bytes`、`unpack_frame(buf) -> Frame`、
  `write_frame(path, fr)`、`read_frame(path)`、`Frame.add(name, array, unit, label)`。
- TypeScript（读端 + 测试用写端）：`viz/shared/src/nlf.ts` ——
  `decodeNlf(bytes): Frame`（**仅** `codec === "none"`）、`decodeNlfAsync(bytes): Promise<Frame>`（任意 codec，
  内部用 `DecompressionStream`）、`encodeNlf(frame, opts?): Uint8Array`、`frameSummary(frame): string`。
  调用方一律用 `decodeNlfAsync`（对未压缩帧无额外开销），这样可以不必在业务代码里分支。
- 两侧的**报错顺序**必须一致（`version → endian → codec → dtype → order → nbytes 自洽 → 越界 → payload 总长 → 区间重叠`），
  便于把"同一份坏字节"的判定写成跨语言对拍。
- 有意的差异（只影响**非法输入**）：TS 端更严——`kind`、`meta` 标量性、`shape` 元素整数性、
  `strings`/`critical`/`pitfalls` 类型都会抛 `NlfError`，而 Python 写端本来就不可能产出这些形态，
  因此 Python 写出的帧 100% 可被 TS 解码（已用 6 夹具 × 3 codec 双向验证）。
- 基准文件：`tests/fixtures/nlf/*.nlf`（每个 ≤ 4 KB，随仓库提交）。两侧都必须能逐点还原，
  还原结果写入 `tests/parity/golden/nlf_*.json` 供 G7 逐值比对。

---

## 2. WebSocket 流式协议

端点：`ws://<host>/ws/chapter/<chapter_id>`。**二进制帧 = `.nlf`**，**文本帧 = JSON 控制消息**。

### 2.1 客户端 → 服务端

```json
{"type": "subscribe", "chapter": "3.2", "viz": "stability_domain",
 "params": {"method": "rk4", "h": 0.05},
 "time":   {"mode": "stream", "dt": 0.01, "frames": 120, "fps": 30}}
```
| type | 载荷 | 语义 |
|---|---|---|
| `subscribe` | `chapter, viz, params, time` | 建立订阅并请求首帧 |
| `params` | `params`（增量合并） | 参数变化：服务端重算 **1 帧**（默认）或按 `time` 重放 |
| `play` / `pause` / `seek` | `t`（seek 用） | 时间轴控制 |
| `step` | `n` | 单步前进 n 帧（便于"逐步点亮"） |
| `cancel` | — | 停止当前流，丢弃队列 |
| `ping` | — | 保活（服务端回 `pong`） |

### 2.2 服务端 → 客户端

| type | 载荷 | 语义 |
|---|---|---|
| `ready` | `manifest`（见 §5） | 订阅成功，附 manifest（前端据此建控件） |
| `frames` | `n`（二进制帧数） | 预告接下来会推送 n 个二进制帧 |
| `error` | `message, code` | 计算失败（如参数越界） |
| `pong` | `t` | 保活应答 |
| `done` | `frames_sent` | 流结束（静态终态，前端可停在最后一帧） |

二进制帧与 JSON 交替出现；客户端 MUST 以 `frame.meta.t` 对齐时间轴，不得依赖到达顺序推时间。

### 2.3 服务端实现

`server/ws.py` 为**自研 RFC 6455 子集**（`fastapi/uvicorn/websockets` 不在允许依赖内）：

- 支持：握手（`Sec-WebSocket-Accept`）、文本/二进制帧、客户端掩码解码、分片（continuation）、
  ping/pong、close 握手、`Sec-WebSocket-Protocol` 回显。
- 不支持（并显式以 close code 1003/1002 拒绝）：扩展协商、压缩（`permessage-deflate`）、超过 `MAX_FRAME`（默认 16 MiB）的帧。
- 单连接单线程处理；计算在订阅循环内同步执行（帧小、算得快；重活走预计算）。
- 测试：`tests/test_ws.py` 用自研客户端（`server/wsclient.py`，同样零依赖）跑握手/分片/大帧/关闭/半包。

---

## 3. HTTP API

| 方法 | 路径 | 返回 |
|---|---|---|
| GET | `/` | 站点壳 `viz/site/index.html` |
| GET | `/viz/...`, `/dist/viz/...` | 前端静态资源（JS/CSS/JSON） |
| GET | `/data/...` | 预计算帧与数据集 |
| GET | `/api/health` | `{"ok": true, "chapters": N, "version": "..."}`|
| GET | `/api/chapters` | 章节清单（合并 `docs/chapters.yaml` + 进度 + 依赖）|
| GET | `/api/chapter/<id>/manifest` | 该章全部 viz 的 manifest |
| GET | `/api/chapter/<id>/frame?viz=..&params=<base64url json>` | 单个 `.nlf`（`Content-Type: application/x-nlf`）|
| GET | `/api/chapter/<id>/frames?...&n=60` | 多帧流：`multipart/mixed` 或长度前缀拼接（降级通道）|
| POST | `/api/asserts/<id>` | 触发该章断言（仅 `--allow-write` 时启用）|

`params` 传输：JSON 经 UTF-8 → base64url，避免 URL 编码歧义；也接受 `params_json`（简单调试用）。

---

## 4. 静态预计算与降级

**问题**：浏览器端实时 PDE/FEM 算不动（`docs/CHAPTERS.md` §8 风险 1）。

**方案**：三层递降，前端永远只做渲染与**显示空间**插值。

1. **L1 WebSocket 流**：实时算，所有参数量连续可拖。
2. **L2 REST 单帧**：无 WS 时按需取帧（拖动节流 60 ms）。
3. **L3 静态预计算**：`data/precomputed/<chapter>/index.json` + `*.nlf`。
   - `index.json`：`{"viz": "...", "grid": {"r": [0.1,0.2,...]}, "entries": [{"key": {...}, "file": "r0.10.nlf"}]}`
   - 查表策略：先精确命中，再在声明为 `interpolable: true` 的轴上取最近两点做**显示插值**，
     并在 UI 上打出 `precomputed+interp` 徽标（规则 4 的视觉提示义务）。
   - **禁止**在降级路径上重算物理量（P8）。

`tools/precompute.py --chapter 4.1 --grid spec.yaml` 生成 L3 数据；`make precompute-all` 生成全套（体积可控，缺省只生成"关键参数点"）。

---

## 5. Manifest（可视化契约源）

**单一真源在 Python**：`server/chapters/<id>.py` 里声明 `MANIFEST`，
`tools/gen_manifests.py` 生成
`viz/chapters/<id>/manifest.gen.ts`（TS 字面量 + 类型）与 `data/manifests/<id>.json`（运行时）。
CI 门禁 G6 保证"生成物 == 真源"，从根上消灭"前后端两处声明漂移"。

```python
# server/chapters/<id>.py
MANIFEST = {
  "chapter": "1.2",
  "title": "线性方程组直接法",
  "vizzes": [
    {
      "id": "lu_heatmap",
      "title": "消元过程热力图",
      "dataSource": "backend",              # backend | backend+preview
      "params": [
        {"name": "n", "label": "阶数 n", "type": "int", "min": 4, "max": 128, "default": 24,
         "critical": [], "criticalNote": ""},
        {"name": "pivot", "label": "主元策略", "type": "enum",
         "options": ["none","partial","complete"], "default": "partial", "critical": []},
        {"name": "rho", "label": "增长因子阈值", "type": "float", "min": 1, "max": 1e4,
         "default": 10, "critical": [128.0], "criticalNote": "完全主元下 ρ 上界 2^{n-1}·…"}
      ],
      "time": {"mode": "none"},             # none | stream | scrub
      "requiredAux": ["error"],             # 规则 3：必须给出误差/残差场
      "axes": [{"id": "main", "autoscaled": false}, {"id": "error", "autoscaled": true}],
      "pitfalls": "热力图用对数色标会把 1e-16 和 1e-3 画得一样深……",
      "theory": ["后向误差界 ‖ΔA‖ ≤ ρ·u·‖A‖", "Cholesky 存在 ⟺ A 对称正定"]
    }
  ]
}
```

字段约束（`server/manifest.py::validate_manifest` 强制，违反即 pytest 失败）：

- `params[*].critical` MUST 存在（无临界值也要写 `[]`）——规则 6 的机器化。
- `params[*].criticalNote` 在 `critical` 非空时 MUST 非空。
- `requiredAux` MUST 含 `error` 或 `residual`——规则 3。
- `axes[*].autoscaled` MUST 显式给出——规则 1/4。
- `pitfalls` MUST ≥ 60 字（中文按字符数计）——规则 7。
- `time.mode="none"` 的 viz MUST NOT 声明时间轴控件。
- 每个 viz 的 `id` MUST 唯一，且与 `viz/chapters/<id>/<viz_id>.viz.ts` 文件同名。

---

## 6. Scene 契约（渲染与视觉回归）

前端 `model.ts` 把 `Frame` 映射为**纯 JSON** 的 `Scene`；`view.ts` 把 `Scene` 画到 canvas；
Python `tools/raster.py` 把**同一个** `Scene` 光栅化为 PNG。于是"视觉回归"在没有浏览器的环境下也可执行。

```ts
type Scene = {
  version: 1; chapter: string; viz: string; title: string; t: number;
  canvas: { w: number; h: number; background: string };
  axes: Axis[];
  layers: Layer[];
  texts: Text[];
  badges: Badge[];          // 数据来源 / 自适应轴 / 溢出 / 预览 徽标
  overflow: { flag: boolean; note: string };
};
type Axis = {
  id: string; rect: [number, number, number, number];        // px: x,y,w,h
  xlim: [number, number]; ylim: [number, number];
  xscale: "linear" | "log"; yscale: "linear" | "log";
  xlabel: string; ylabel: string; title: string;
  autoscaled: boolean; xticks: number[]; yticks: number[];
  grid: boolean; frame: boolean;
};
type Layer =
  | { kind: "line"; axes: string; series: { x: number[]; y: number[]; color: string;
      width: number; dash?: number[]; label?: string; marker?: "none"|"o"|"s" }[] }
  | { kind: "band"; axes: string; x: number[]; ylo: number[]; yhi: number[]; color: string; alpha: number }
  | { kind: "scatter"; axes: string; x: number[]; y: number[]; r: number; color: string; label?: string }
  | { kind: "heat"; axes: string; nx: number; ny: number; values: number[];   // 行主序，长度 nx*ny
      extent: [number, number, number, number]; cmap: "viridis"|"magma"|"coolwarm"|"gray"|"rdbu";
      vmin: number; vmax: number; logScale: boolean; clipBelow?: number }
  | { kind: "contour"; axes: string; nx: number; ny: number; values: number[];
      extent: [number, number, number, number]; levels: number[]; color: string; width: number }
  | { kind: "bars"; axes: string; x: number[]; values: number[]; width: number; color: string;
      errors?: number[]; baseline?: number }
  | { kind: "vectorfield"; axes: string; nx: number; ny: number; u: number[]; v: number[];
      extent: [number, number, number, number]; stride: number; color: string; scale: number }
  | { kind: "loglogfit"; axes: string; h: number[]; err: number[];
      measuredSlope: number; theorySlope: number; refX: [number, number];
      seriesLabel: string; theoryLabel: string };
type Text = { x: number; y: number; text: string; size: number; color: string;
              align: "left"|"center"|"right"; role: "title"|"label"|"note"|"axis" };
type Badge = { x: number; y: number; text: string; kind: "source"|"autoscale"|"overflow"|"preview"|"critical" };
```

`validateScene(scene)`（TS）与 `tools/raster.py::validate_scene`（Python）实现同一套校验，
规则：层引用的 `axes` 必须存在；`heat.values.length == nx*ny`；`loglogfit` 两斜率必须有限；
`overflow.flag == true` 时 `overflow.note` 非空；`badges` 至少含一个 `source` 徽标。
两侧校验器都对 `viz/tests/scene/golden/*.json` 的同一批场景跑，结论必须一致（G7 的一部分）。

**配色与色标**：`viz/shared/src/colorscale.ts` 内置 5 个色标（256 级 LUT 常量，**不含第三方依赖**），
Python `tools/raster.py` 用同名同值的 LUT（`tools/colorscales.py` 生成一次，两侧共用 JSON：
`viz/shared/colorscales.json` 由 `tools/gen_colorscales.py` 产出，G6 检查）。

---

## 7. Python ↔ TypeScript 对拍协议

目的：`docs/CHAPTERS.md` §3 的"双实现策略"必须**可执行**，否则前端会撒谎。

- 基准数据由 Python 生成并落盘：`tests/parity/golden/<case>.nlf`（或 `.json`，小数据用 JSON 便于 diff）。
- TS 侧 runner：`viz/tests/parity/run_parity.ts` → 编译产物 `node dist/viz/tests/parity/run_parity.js`，
  读取 `tests/parity/cases.json`，对每个 case 调用 `viz/shared/src/numeric/<algo>.ts`，
  输出 `dist/reports/parity_ts.json`。
- Python 侧 runner：`tools/parity.py` 读同一批 case，调用 `numcore` 实现，与 TS 输出**逐点**比较。
- 容差**写在仓库里**：`tests/parity/tolerances.yaml`：
  ```yaml
  default: {rtol: 1e-12, atol: 1e-14}
  cases:
    lu_partial_pivot: {rtol: 1e-13, atol: 1e-15, reason: "TS 用 float64 同一算法，误差来源仅求和顺序"}
    fft_radix2:      {rtol: 1e-10, atol: 1e-12, reason: "三角查表 vs 递推，累积差"}
  ```
- 门禁 G7 失败时 MUST 打印 `case / 索引 / python 值 / ts 值 / 相对差 / 容差`。

覆盖清单（M0 起逐步补齐）：`lu`、`qr`（Householder）、`chol`、`svd_jacobi`、`power_iter`、
`barycentric`、`spline_natural`、`gauss_legendre_nodes`、`fft_radix2`、`dwt_haar`、
`rk4`、`dopri5_step`、`verlet`、`heat_explicit_step`、`upwind_step`、`sobol_seq`、`pce_basis`。

---

## 8. 断言报告协议

`tools/run_asserts.py` 产出 `dist/reports/assertions.json`：

```json
{"generated_at": "...", "totals": {"A": 12, "B": 9, "C": 7, "passed": 28, "failed": 0},
 "chapters": {"1.2": {"declared": 7, "collected": 7, "passed": 7,
   "items": [{"id": "lu_backward_error", "kind": "B", "theory": "≤ ρ·u·‖A‖",
              "measured": 3.1e-16, "tol": 1e-15, "passed": true,
              "nodeid": "tests/1.2/test_lu.py::test_backward_error_bound"}]}}}
```

每章的 `chapters/<id>/asserts_report.md` 由同一脚本生成（理论值 / 实测值 / 容差 / 判定），
并作为 `notes.tex` 的附表 `\input` 进去——**纸面结论与机器结论同源**。

---

## 9. 产物目录

```
dist/
├── viz/                 tsc 编译产物（浏览器原生 ESM）
├── export/
│   ├── gif/<chapter>-<viz>.gif
│   ├── mp4/<chapter>-<viz>.mp4        （需 ffmpeg，缺失时降级为 .frames + README）
│   └── png/<chapter>-<viz>-<t>.png
├── reports/
│   ├── assertions.json / asserts_<id>.md
│   ├── parity_ts.json / parity.json
│   ├── policy.json
│   └── progress.json
├── visual/
│   ├── scenes/<chapter>-<viz>.json    （场景快照，随仓库提交，是回归基线）
│   └── png/<chapter>-<viz>.png        （光栅基线）
└── book/
    ├── chapters/<id>.pdf
    └── numerical-lab.pdf              （合并本）
```
