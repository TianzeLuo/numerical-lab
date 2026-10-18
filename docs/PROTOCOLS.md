# PROTOCOLS —— 帧格式 / 场景契约 / 可视化契约 / 对拍协议

> 本文件定义**跨语言、跨进程、跨时间的接口**。任何一端改动都必须同步本文件与两侧实现，
> 并由 `tests/test_nlf.py`、`viz/tests/nlf.test.ts`、`tests/parity/` 三处守住。

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
  `decodeNlf(bytes): Frame`（**同步**，**仅** `codec === "none"`）、
  `decodeNlfAsync(bytes): Promise<Frame>`（**异步**，任意 codec，内部用 `DecompressionStream`）、
  `encodeNlf(frame, opts?): Promise<Uint8Array>`（**异步**，`opts.codec` 缺省 `"none"`，调用方必须 `await`）、
  `frameSummary(frame): string`。
  调用方一律用 `decodeNlfAsync`（对未压缩帧无额外开销），这样可以不必在业务代码里分支。
  `encodeNlf` 之所以是异步的：`codec="zlib"` 要走 `CompressionStream`，而它只有异步 API——
  把 `none` 分支写成同步、`zlib` 分支写成异步会让调用方被迫分叉，所以统一成 `Promise`。
- 两侧的**报错顺序**必须一致（`version → endian → codec → dtype → order → nbytes 自洽 → 越界 → payload 总长 → 区间重叠`），
  便于把"同一份坏字节"的判定写成跨语言对拍。
- 有意的差异（只影响**非法输入**）：TS 端更严——`kind`、`meta` 标量性、`shape` 元素整数性、
  `strings`/`critical`/`pitfalls` 类型都会抛 `NlfError`，而 Python 写端本来就不可能产出这些形态，
  因此 Python 写出的帧 100% 可被 TS 解码。实测口径（`tests/fixtures/nlf/` 的 **6 个夹具**，
  用 `server/nlf.py::read_frame` 逐个读 header 得到）：**5 个 `codec=none` + 1 个 `zlib`（`zlib_2d.nlf`），
  没有 gzip 夹具**；`gzip` 分支只在 `viz/tests/nlf.test.ts` 里用**合成帧**做往返，不在这 6 个夹具之列。
- 基准文件：`tests/fixtures/nlf/*.nlf`（随仓库提交）。实测体积 **618 B – 10049 B**：
  `zlib_2d.nlf` 约 **9.8 KiB**（压缩帧的 payload 是大数组，压缩后仍比其余夹具大一个量级），
  其余 5 个 653–890 B。两侧都必须能逐点还原，
  还原结果写入 `tests/parity/golden/nlf_*.json`（6 份）供 G7 逐值比对。

### 1.3 `Frame.set_meta` 的硬约束（写入前自检）

`meta` 原样进 header JSON，所以只收 **number / bool / null**，且 number 必须是**有限数**。
三条都不是"风格建议"，违反时 `server/nlf.py::Frame.validate()` 当场抛 `NlfError`：

| 写法 | 报错（原文摘录） |
|---|---|
| `fr.set_meta(tag="partial")` | `meta['tag'] 是字符串——请改放进 strings（PROTOCOLS §1.1）` |
| `fr.set_meta(hist=[1, 2])`（list / tuple / ndarray / dict 同理） | `meta['hist'] 不是 number/bool/null（PROTOCOLS §1.1）` |
| `fr.set_meta(kappa_u=float("nan"))`（`±inf` 同理） | `meta['kappa_u'] = nan 不是有限数——JSON 没有 NaN/Infinity，请用 None（null）表示"无有限值"，并在 strings 里说明原因` |

最后一条是**硬错误**，不是"写的时候顺手清洗一下"：header 由 `json.dumps(..., allow_nan=False)`
编码（`server/nlf.py:315`），NaN 会在**写盘那一刻**才炸，报的是
`Out of range float values are not JSON compliant`——与"哪个量没定义"毫无关系（1.1 章真踩过：
`pack_frame` 抛在预计算落盘路径上，报错完全指不到元凶）。`Frame.validate()` 宁可提前拒绝，
也不让一个 NaN 拖垮整帧。

**规定写法**：每个装配帧的章节文件里放一个 `_finite_or_none(x)` 小工具
（1.1 章叫 `_scalar`、1.2/1.5 章叫 `_finite_or_none`、2.2 章叫 `_finite`），
"没有有限值"一律写 `None`，并在 `strings` 里如实说明**为什么**。已落地的四处：

| 位置 | 写进 meta | `strings` 里的说明 |
|---|---|---|
| 1.1 `cond_ellipsoid`：`kappa_u`（κ 非有限时） | `None` | `note`：`κu = … ≥ 1：A 在 float64 里已经和奇异矩阵无法区分…` |
| 1.2 `unit_ball`：`forward_error` / `backward_error` / `residual` / `x_num_absmax` | `None`（`singular=True` 时） | `status`：`数值奇异：κ > 1/u，A 的浮点分解出现零主元`；另有 `note` 说明 |
| 2.2 曲率能量：`energy_ratio`（τ=1 时能量恰为 0） | `None` | `note`：`tau=1：弯矩为 0，曲线就是折线——C1 已经断了，曲率能量恒为 0（比值因此无定义）` |
| 1.5 `preconditioner`：`ilu_dropped`（预条件不是 ILU(0) 时） | `-1` 哨兵 | **必须在 `strings` 里写明 `-1` = 不适用**；当前只在 manifest 描述里提了"ILU(0) 丢弃的填充次数"，见 `docs/ROADMAP.md` §4 的已知缺口 |

**共同纪律**：帧里出现 `None` 或 `-1` 这类哨兵值，`strings` 里就必须有一句解释它为什么不是数字；
枚举/字符串参数（主元策略、边界条件、方法名）一律走 `fr.set_strings(...)`，不进 `meta`。
自检：`python3 -m pytest tests/test_nlf.py tests/test_frames_contract.py -q`
（前者钉 `meta` 的类型契约，后者把每一章的每个 viz 在默认参数与边界参数上**真的编码一遍**——
NaN 会在那里当场炸）；单章还可以跑 `python3 tools/precompute.py --chapter <id>`，
同样在这一步抛 `NlfError`，而不是等到前端解码那一帧。

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
- 不支持（并显式拒绝）：扩展协商与压缩（`permessage-deflate`，RSV1 非零 → **1002**）、
  未知 opcode → **1002**、非法 UTF-8 文本 → **1007**、**超过 `MAX_FRAME`（默认 16 MiB）的帧 → 1009**。
  注意"超长"用的是 **1009**（message too big），不是 1003（unsupported data）；1002 只用于协议违规。
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
| GET | `/api/chapter/<id>/frame?viz=..&params=<base64url json>&t=<%.6g>` | 单个 `.nlf`（`Content-Type: application/x-nlf`）|
| GET | `/api/chapter/<id>/frames?...&n=60` | 多帧流（降级通道）：**只有 `NLFS/1` 一种格式**——长度前缀容器，`Content-Type: application/octet-stream`，见下；**没有 multipart** |
| GET \| POST | `/api/asserts/<id>` | **只读**返回 `dist/reports/assertions.json` 里该章的记录（**不会跑 pytest**）。报告文件不存在 → **404**；未以 `--allow-write` 启动 → **403** |

`params` 传输：JSON 经 UTF-8 → base64url，避免 URL 编码歧义；也接受 `params_json`（简单调试用）。

`t` **是 `/frame` 的必需语义参数**（缺省 `t=0`）：没有它，L2 降级通道就无法响应时间轴拖动
（拖动/单步都会拿到同一帧）。数字写法与 L3 文件名一致，用 `%.6g`（`t=0.05`、`t=1.5e-05`）；
服务端解析后 MUST 与 `TimeSpec.dt` 对齐到最近的整步，并在响应帧的 `meta.t_actual` 里写回**实到时刻**。

`/frames`（多帧流）格式**已定稿为 NLFS/1**，自描述长度前缀容器，**所有 u32 一律大端**：

```
offset 0 : magic   = b"NLFS"
offset 4 : u32BE   version = 1
offset 8 : u32BE   header_len
offset 12: header_len 字节 UTF-8 JSON，形如
           {"format":"nlf-stream","version":1,"chapter":"0.0","viz":"heat_ftcs",
            "n":5,"params":{...},"t":[0.0,0.05,0.1,0.15,0.2]}
之后 n 条记录，每条 = u32BE frame_len | frame_len 字节的完整 .nlf
```

响应头：`Content-Type: application/octet-stream`、`X-Nlf-Stream: NLFS/1`、`X-Frame-Count: n`。
参考实现 `server.app.pack_frame_stream` / `read_frame_stream`（测试里另有一份独立的手工解析）。
时间栅格与 `server.frames.precompute_chapter` 同口径（`tMax*i/(n-1)`，**不做 dt 对齐**），
因此同参数下 `/frames` 的每一帧与 L3 静态文件**逐字节相同**。
注意 `.nlf` v1 自己的 `header_len` 是**小端**，两个容器别混。

---

## 4. 静态预计算与降级

**问题**：浏览器端实时 PDE/FEM 算不动（`docs/CHAPTERS.md` §8 风险 1）。

**方案**：三层递降，前端永远只做渲染与**显示空间**插值。

1. **L1 WebSocket 流**：实时算，所有参数量连续可拖。
2. **L2 REST 单帧**：无 WS 时按需取帧（拖动节流 60 ms）。
3. **L3 静态预计算**：`data/frames/<chapter>/<viz>/<params_hash>_t<%.6g>.nlf`（平铺，无 index.json）。
   - 文件名里的 `params_hash` 与 `server.frames.params_hash` **逐字节一致**（sha256 前 16 位十六进制，
     参数键排序、`json.dumps(..., sort_keys=True, separators=(",", ":"), ensure_ascii=False)` 之后取哈希）。
   - `t` 用 `%.6g`（与 `/frame` 的 `t` 同款写法）；`t=0` 写作 `t0`。
   - 为什么不做 index.json：文件名本身就是索引。前端在 spawn 之前就能算出候选路径，
     命中失败只是"少一帧"，不会因为索引与文件不同步而整章黑屏。**候选路径的生成规则只有一份实现**
     （Python `server.frames.static_frame_path` 与 TS `framesource.staticFrameRelPath`，由 G7 对拍钉住）。
   - 查表策略：先精确命中；没有再退到 `none`（发一条 `status`，UI 显示"无数据源"）。
     显示空间插值只在**同一次会话已有的两帧**之间做，并打 `interp` 徽标（规则 4 的视觉提示义务）。
   - **禁止**在降级路径上重算物理量（P8）。

`tools/precompute.py --chapter 4.1 [--n N]` 生成 L3 数据；不带 `--chapter` 时生成全套
（体积可控，缺省每个可视化只生成 1 帧）。
**`--n` 是视觉回归的公共契约**：基线的文件名里带时刻，所以 CI 固定用 `--n 6`
（`tools/ci.sh` 的 G8 前置步骤），改它等于改基线命名，必须与 `tests/baselines/` 一起提交。生成完用
`node dist/viz/tests/scene/dump_scenes.js` 把帧过一遍真前端模型，产物 `dist/scenes/**.json`
就是视觉回归（G8）的输入。

---

## 5. Manifest（可视化契约源）

**单一真源在 Python**：`server/chapters/<id>.py` 里声明 `MANIFEST`，
`tools/gen_manifests.py` 生成
`viz/chapters/<id>/manifest.gen.ts`（TS 字面量 + 类型）与 `data/manifests/<id>.json`（运行时）。
CI 门禁 G6 保证"生成物 == 真源"，从根上消灭"前后端两处声明漂移"；
G6 是同一条命令里的两步：`gen_manifests.py --check`（本节）**与** `gen_colorscales.py --check`（§6 的色标表），
两者都无漂移才算 G6 通过。

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

前端 `viz/chapters/<id>/<viz>.model.ts` 把 `Frame` 映射为**纯 JSON** 的 `Scene`；
共享的 `viz/shared/src/render2d.ts` 把 `Scene` 画到 canvas（DOM 装配在 `viz/shared/src/view.ts`）；
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

### 6.1 角标盒与文本盒：两侧共用的一把尺子

模型层是纯函数、拿不到 `measureText`，所以"角标放不放得下、两个盒子会不会相交"只能靠一把**写死的尺子**。
这把尺子同时被 TS 的模型/校验器与 Python 的光栅器/校验器使用，任何一侧偷改都会让另一侧误判
（"模型以为放得下、校验器以为放不下"是无解的扯皮）：

| 量 | TS（`viz/shared/src/scene.ts`） | Python（`tools/raster.py`） | 取值 / 口径 |
|---|---|---|---|
| 显示宽度单位 | `textWidthUnits(text)` | `_text_width_units(text)` | CJK/全角（`codePointAt > 0x2e80` / `ord > 0x2E80`）记 **2**，其余记 **1** |
| 像素 / 单位 | `BADGE_UNIT_PX` | `BADGE_UNIT_PX` | **7**：内嵌位图字体实测（角标字号 = `base_font_size − 2` = 11，实测 10 个 ASCII = 70px、10 个汉字 = 140px，即 7.00 px/单位）；浏览器 sans-serif 略窄，7 当保守上界 |
| 左右内边距之和 | `BADGE_PAD_PX` | `BADGE_PAD_PX` | **10**（两个渲染层都是左右各 5px） |
| 角标盒高度 | `BADGE_BOX_H` | `BADGE_BOX_H` | **15**：校验器用的名义盒高；绘制高度是 `base_font_size + 4`，槽位行距 18，所以相邻行不会撞 |
| 角标盒宽度 | `badgeBoxWidth(text)` | `badge_box_width(text)` | `units × 7 + 10` |
| 单行预算 | `badgeUnitBudget(width)` | `badge_unit_budget(width)` | 见 §6.2 |
| 角标裁剪 | `clipBadgeText(text, units)`（`model.ts`） | `clip_badge_text(text, units)` | 超出部分砍掉并用尾部 `…` 收尾，两侧同一算法 |

钉住两侧的是同一组样例：`tests/test_raster.py::test_badge_box_width_matches_the_shared_metric`
（`"数据：后端 .nlf 帧" == 18 单位 × 7 + 10`）与 `viz/tests/scene.test.ts` 的
`placeBadge` / `badgeUnitBudget` 用例（断言裁剪后 `textWidthUnits(text) ≤ badgeUnitBudget(960)`）。
**改这把尺子 = 两侧一起改**，否则 G8 的像素基线会告诉你哪里对不上。

### 6.2 规则 8（布局）：校验器查什么、模型该怎么写

`validateScene(scene)`（TS）与 `tools/raster.py::validate_scene`（Python）都实现"规则 8（布局）"，
报错文本里直接写着 `规则 8`：

**① 角标盒必须落在画布内、两两不相交**（两侧都查）。
`placeBadge(b, slot, width)`（`viz/shared/src/model.ts`）负责把角标放进固定槽位并**自动裁剪**：

| 常量 / 函数 | 值 | 语义 |
|---|---|---|
| `BADGE_LAYOUT` | `{x: 8, top: 34, rowH: 18, colW: 344, maxSlots: 8}` | 顶部角标区；`top=34` 是为了不压住画在 (12, 8) 的 `scene.title` |
| `badgeColumns(width)` | `width ≥ 2×344 + 24 = 712` 才是 **2** 列，否则 1 列 | 窄画布（含 `--width 200` 的导出）退化成单列 |
| `badgeRows(width)` | `ceil(8 / columns)`：两列 **4** 行、单列 **8** 行 | 排版函数据此让出顶部 |
| `headerBand(width)` | `34 + rows×18 + 8`：两列 **114px**、单列 **186px** | 面板的 `top` 必须从它之后开始，角标与面板互不重叠 |
| `badgeUnitBudget(width)` | 两列：`floor((344 − 10) / 7)` = **47** 个显示单位；单列：`floor((max(60, width − 16) − 10) / 7)`（960 宽 → 47、200 宽 → 24） | `placeBadge` 用它裁文案 |

**角标是索引级标签**：`placeBadge` 的裁剪是静默的（只留尾部 `…`），所以角标里只放"这是什么"
（数据来源、自适应、溢出、临界值、降级）；完整句子请放面板文本、`strings` 或 `pitfalls`。

**② 文本层不得压住坐标轴的装饰带**（TS 侧 `validateScene` 当场报；Python 侧目前只做第 ① 条，
见下面的判据）。装饰带常量 `AXIS_DECO = {top: 18, bottom: 28, left: 56}`，
`axisDecoBands(axis)` **按需生成**——`xticks` 为空就不占刻度带、没有 `xlabel` / `title` / `ylabel`
就不占那一条：

| 有装饰才占带 | 矩形（相对画布） | 谁画的 |
|---|---|---|
| `title` 非空 | `[x, y−18, w, 16]` | 轴标题基线在 `top − 6` |
| `xticks` 非空 | `[x, y+h+3, w, 15]` | 刻度标签基线在 `bottom + 4` |
| `xlabel` 非空 | `[x, y+h+16, w, 12]` | 轴名基线在 `bottom + base + 16` |
| `yticks` 非空 | `[x−41, y, 35, h]` | y 刻度标签右对齐于 `left − 6` |
| `ylabel` 非空 | `[x−56, y, 14, h]` | 轴名竖排在 `left − 42` |

**清空 `xticks` 的轴不占刻度带**——好几个模型刻意清空 `xticks` 改用手写类别名
（`AxisOptions` 没有 `xtickLabels`），若把空刻度也当装饰，这些模型会被误报。

**③ 文本盒按"块"算，不按横条算**：`textBoxOf(t)`（TS）与光栅器的排版同口径——按 `maxWidth`
折行（每行 `floor(maxWidth / 7)` 个单位）、行高 `size + 4`、`baseline = top`、按 `align` 平移起点。
一句长说明占的是**几行的块**；早先按单行估盒，0.0 章被误报过。

**三起真实事故（各写清怎么修）**：

| 事故 | 现象 | 修法 |
|---|---|---|
| 1.2 / 0.0 章的一枚 **65 单位**角标 | 横穿到右列、压住临界值角标 | `placeBadge` 统一裁到列宽预算（`clipBadgeText`）：宁可少显示几个字，也不许两枚角标叠在一起 |
| `lu_heatmap` 的数值块 | 压进副轴 `xlabel` 带 | 副轴高度多扣 **62px**（轴自己的两条装饰带 + 模型自绘的两行文本块），并把该轴 `xlabel` 置空（`viz/chapters/1.2/lu_heatmap.model.ts:203`） |
| `spy_ordering` 的手写类别标签 | 与 `xlabel` 挤在同一条带 | 清空该轴 `xticks`、**不设** `xlabel`，类别名全部由 `text` 层画 |

判据（可执行）：`node dist/viz/tests/scene/dump_scenes.js` 会用 TS `validateScene` 校验每个真场景，
`python3 tools/visual_regression.py --check --require-scenes` 再用 Python `validate_scene` 校验同一批文件；
两侧报告里都不出现 `规则 8` 字样才算过。

两侧**不是**各自跑一批 golden JSON 场景（仓库里没有 `viz/tests/scene/golden/`）：真正被钉住的是
**同一批真场景**——`dump_scenes.ts` 先用 TS `validateScene` 校验再写 `dist/scenes/**.json`，
`tools/visual_regression.py` 再用 Python `validate_scene` 校验同一批文件并光栅化（G8）。
两侧各自另有单测：TS 在 `viz/tests/scene.test.ts`，Python 在 `tests/test_raster.py::test_validate_scene_*`。

**配色与色标**：`viz/shared/src/colorscale.ts` 内置 5 个色标（256 级 LUT 常量，**不含第三方依赖**），
Python `tools/raster.py` 用同名同值的 LUT（`tools/colorscale_tables.py` 是真源，`tools/gen_colorscales.py`
生成两侧共享的 `viz/shared/colorscales.json` 与 `viz/shared/src/colorscale_data.gen.ts`，由 G6 检查）。

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
    lu_partial_pivot: {rtol: 1e-14, atol: 1e-16, reason: "同一套运算顺序（先整列算乘子、再整块 outer 更新），实测逐位相同；不取 0 是留 FMA 收缩的 1 ulp 余量"}
    fft_radix2:      {rtol: 1e-10, atol: 1e-12, reason: "三角查表 vs 递推，累积差"}
  ```
- 门禁 G7 失败时 MUST 打印 `case / 索引 / python 值 / ts 值 / 相对差 / 容差`，
  并给出跨 case 的"最大超差 Top 5"（按 `|Δ|/容差` 降序）。
- **判定与生成分离**：`--update-golden` 重新生成基准与 `cases.json`（含每份 golden 的 sha256）；
  `--check-golden` 只验证"盘上的 golden == 现在 Python 算出来的"（谁偷偷改了 `numcore`，这里立刻红）。
- **容差必须写理由**：`tests/parity/tolerances.yaml` 的每条覆盖都要带 `reason` 与实测值，
  例如 heat 系"实测逐位相同，但跨平台 FMA 可能差 1 ulp，故留 1e-14≈45 ulp"。

当前覆盖（截至 2026-10-18）：`heat_explicit_step`、`heat_ftcs_run`、`ftcs_amplification_modes`、
`oscillator_sample`、`order_fit_powerlaw`、`lu_partial_pivot`（Hilbert(6)）、`lu_wilkinson_growth`（Wilkinson(12)，ρ = 2^11）、
`lu_tiny_pivot`（首元 1e-14）—— **8 个 case / 2387 个点**（逐 case 点数：1800 / 312 / 92 / 68 / 41 / 41 / 25 / 8，
读数来自 `dist/reports/parity.json`；新增的 3 条 LU case 实测 `max|Δ| = 0`）。
覆盖清单（M0 起逐步补齐，加一个 case = `tools/parity.py::CASE_SPECS` 加一条 +
`viz/shared/src/numeric/<algo>.ts` 加一个模块 + `--update-golden`）：~~`lu`~~（已完成：`luPartialPivot`，
见 `viz/shared/src/numeric/lu.ts` + `viz/tests/numeric.lu.test.ts` 的本地护栏）、
`qr`（Householder）、`chol`、`svd_jacobi`、`power_iter`、
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

### 8.1 每条断言至少要产生一条 `record(...)`

`@assertion("A"|"B"|"C", theory=..., chapter=...)` 装饰的每个测试，运行期**必须**至少登记一条记录：
调用 `numcore.verify` 的 `assert_*` helper、显式 `record(...)`、或 `return Measured(value=..., tol=...)`。
一条都没有 → 插件（`numcore/verify/report.py`，`strict_measure=True` 缺省，**不要关**）直接判失败：

```
测试未登记任何实测值：请调用 numcore.verify 的 assert_* helper，或 return Measured(value=..., tol=...)。禁止静默通过（POLICY §3）。
```

### 8.2 helper 的类别固定是 `"B"`：`@assertion("A")` 不能只调 helper

`invariants.py` / `rates.py` 里的不变量 helper（`assert_leq` / `assert_within` / `assert_monotone` /
`assert_conserved` …）登记的 `kind` **写死为 `"B"`**。于是 `@assertion("A")` 的测试若只调 helper，
就会出现"声明 A 却一条 A 记录都没有"——插件会报**类别不符**并失败：

```
声明为 A 类，但本测试登记的全是 B 类记录：A 类必须与解析解/参考实现对拍、B 类是不变量、C 类是收敛率。
要用 helper 记 A 类，请显式调用 record(kind="A", ...) 并 assert 它；helper 的缺省类别是 B。
```

规范写法（自己算完残差/误差后登记）：

```python
rec = record(label=..., theory=..., measured=..., tol=..., passed=cond)
assert rec.passed, rec.measured
```

一行 `record` + 一行 `assert`：既保证类别正确，又保证失败真的失败。

### 8.3 `record(passed=False)` 后面必须跟 `assert`

只 `record(passed=False)` 而不 `assert`：pytest **全绿**，但 `tools/run_asserts.py::analyse`
先看 `totals.failed`，会报 `有 1 条断言失败`。
两种失败方式分工明确：`assert` 让 pytest 当场红（定位到用例），报告的 `failed` 让 G3 红（定位到章节）。
**统一写成 `assert rec.passed, rec.measured`**，两处一起红，不留"记录里失败、进程却退出 0"的缝。

### 8.4 `assert_order` / `fit_order` 是幂律拟合：递减、渐近区、窗口写进 `theory`

`numcore/verify/order.py` 拟合的是 `err ≈ C·h^p`（双对数最小二乘），四个硬条件：

1. `h` / `err` **必须按 `h` 递减**传（从最粗网格到最细网格）。`head=` 保留**粗端**、`tail=` 保留**细端**——
   "尾部触舍入地板"要用 `head=`，方向别记反（写反了等于把被污染的那一端留下）。
2. **只喂幂律关系**。对数型关系（例："位数 = log10(gap) + C"）拟合出来的 `p` 没有意义（1.1 章实测 **0.047**）。
3. **必须确认处在渐近区**，窗口写进 `theory` 串与 `meta`：1.3 章正规方程把 21 个点全塞进去得 **1.52**，
   只取 `err < 0.1` 的窗口才回到 **1.981**；3.2 章 BDF2 粗窗口（h = 1/8…1/64）**1.8843**、
   细窗口（1/128…1/4096）**1.9862**。
4. **饱和段 / 舍入地板会污染拟合**：`err` 到机器精度后不再随 h 下降，全局拟合必然偏小。
   `numcore.verify.diagnose_order` 会给出 `skip=` / `head=` 的建议，但它是**诊断、不自动改窗口**——
   窗口是你写进 `theory` 的断言内容的一部分。

### 8.5 A 类要双向：既测"精度对"，也测"到边界当场失效"

"误差很小"单独不能证明阶或精度，必须配上**边界反例**：

- 2.3 章 Gauss 求积：断言 `k ≤ 2n−1` 的矩误差在舍入水平，**同时**断言 `k = 2n` 处误差显著非零
  （否则"精度"没有上界，等于什么都没说）；交错定理：交错点个数 **= n+2**、振幅比 → 1。
- 1.1 章 Kahan：除了"`Kahan 误差 ~ u`"，还测**分段并行 43 倍退化**
  （`1e16 + 1e-3×4095`：串行 0.095 → 8 段并行 4.095）——"稳定"这句话的边界必须被实测划出来。

### 8.6 矩阵分解类章节的 A 类必须覆盖 `m<n` / `m=n` / `m>n`

形状假设不写进断言，就会在宽矩阵上悄悄失效：1.3 章 `bidiagonalize` 曾在 `(5, 8)` 上给出
**0.935** 的重建误差（相对 ‖A‖ 的 4.65e14 倍），因为 `m<n` 时 `r×r` 上双对角**不存在**、
规范形式是下双对角。现在的断言在 `(8,5)` / `(5,8)` / `(12,12)` / `(20,4)` 四种形状上同时测
重建误差、"只有两条对角线非零"与两翼正交性
（`tests/1.3/test_qr_svd.py::test_bidiagonalization_structure_and_error`）。
1.3 与 2.2 两次被"宽矩阵的规范形式不同"咬到——`m<n / m=n / m>n` 是最低覆盖。

### 8.7 反例要交代数据分布

"我试了没出问题"不是证据：1.1 章 3000 组 `uniform(−1,1)` 三元组里 **2940 组**加法结合律的结果逐位相同，
把量级铺到 `2^−30 … 2^0` 才有 **654 组**不同、偏差最大到界的 **0.9969 倍**。
反例类断言的 `what` / `theory` 里要写清样本分布与规模，否则测出来的只是"数据还不够刁"。

### 8.8 报告文件是全仓共享的：单章验收要与 pytest 串在同一条命令里

`dist/reports/assertions.json` **每次 pytest 会话都被整体覆盖**（插件在 `pytest_configure` 清表、
在 `pytest_sessionfinish` 落盘；`tools/run_asserts.py` 用 `NUMCORE_REPORT` 指定同一个文件）。
所以单独跑 `pytest tests/<id>` 之后，报告里只剩这一章；紧接着 `run_asserts.py --no-run`
会报其它已实现章节"在报告里没有任何断言记录"（实测：只跑 `tests/1.3` 后，0.0/1.1/1.2/1.4/1.5/2.1/2.2/2.3/3.2 共 9 章全部报缺）。
单章排查请指定一份自己的报告，并把两条命令**串在同一条命令里**跑：

```bash
NUMCORE_REPORT=/tmp/assertions-1.3.json python3 -m pytest tests/1.3 -q \
  && python3 tools/run_asserts.py --no-run --report /tmp/assertions-1.3.json
```

G3 用的是整跑出来的那一份：G2 `pytest tests -q` 写、G3 `run_asserts.py --no-run` 读（`tools/ci.sh`），
所以"过门禁"只能靠不带 pytest 参数的 `python3 tools/run_asserts.py`（或 `make asserts`）。
注意 `run_asserts.py` 无论报告路径是什么，都会按该报告**重写** `chapters/<id>/asserts_report.md`
（报告里没有的章节不重写）——临时报告只影响被它覆盖到的那几章。

---

## 9. 产物目录

```
dist/                                # 全部可重新生成（.gitignore 覆盖 dist/**）
├── viz/                 tsc 编译产物（浏览器原生 ESM）
├── scenes/<chapter>/<viz>__<params指纹>__t<时刻>.json   # 真模型跑出来的 Scene（G8 的**首选**输入）
├── visual/
│   ├── scenes/<chapter>-<name>.json   # 光栅器侧的场景快照（诊断用）
│   ├── png/<chapter>-<name>.png       # 本次光栅结果
│   └── diff/<chapter>-<name>.png      # 与基线的差异图（不一致时）
├── reports/
│   ├── assertions.json（`tools/run_asserts.py`；每章的 markdown 附表落在 chapters/<id>/asserts_report.md，不在 dist/）
│   ├── parity_ts.json / parity.json
│   ├── policy.json / manifests.json / scenes.json / visual_regression.json
│   ├── book.json / ci.json
├── logs/G1.log … G10.log              # `make ci` 每道门的完整输出
├── book/                              # one_<id>.tex/pdf、book.tex/pdf、*.log
└── export/
    ├── gif/<chapter>-<viz>.gif
    └── png/<chapter>-<viz>-<t>.png

tests/baselines/<chapter>/<场景文件名>.png   # **进 git**：视觉回归的契约
tests/parity/golden/<case>.json             # **进 git**：对拍基准（含 sha256 索引 cases.json）
data/manifests/<id>.json + index.json       # **进 git**：manifest 生成物（G6 保证与真源一致）
data/frames/<chapter>/<viz>/<hash>_t<t>.nlf # 生成物，可 `make precompute` 重建（不进 git）
```

**基线命名规则**：基线文件名 = 场景转储文件名（只换扩展名），所以
"改 `dump_scenes.ts` 的采样时刻/参数 → 文件名全变 → 全部缺基线"是**有意**的：
想让新场景进基线，必须显式跑 `make visual-update` 并在提交信息里写明原因。
