# ARCHITECTURE —— 数据流、模块地图与测试策略

> 读完 `docs/CHAPTERS.md`（要做什么）+ `docs/POLICY.md`（红线）+ `docs/PROTOCOLS.md`（接口）后，
> 本文件回答"代码怎么摆、一次点击的数据怎么流、怎么证明它没撒谎"。

---

## 1. 一张图的数据流

```
              ┌────────────────────────── numcore/（纯函数，无 IO）──────────────────────────┐
              │  linalg  interp  quad  transform  ode  pde  inverse  uq  apps                 │
              │  ── 唯一数学真源；reference/ 是唯一可 import scipy 的对拍基准 ──              │
              └───────────────────────────────┬───────────────────────────────────────────┘
                                              │  (纯数组进出)
                       ┌──────────────────────▼───────────────────────┐
                       │ server/chapters/<id>.py                      │
                       │   MANIFEST（参数/临界值/轴/陷阱文本）         │
                       │   compute(viz, params, t) -> Frame           │
                       └───────┬───────────────────────┬──────────────┘
     tools/gen_manifests.py    │                       │ server/nlf.py
   （契约的单一真源→生成物）    │                       │ 打包 .nlf
        ┌──────────────────────┴────────┐       ┌──────▼───────────────────────────┐
        ▼                               ▼       │ server/app.py（零依赖 http.server）│
 viz/chapters/<id>/manifest.gen.ts   data/manifests/<id>.json                  │
 viz/chapters/index.gen.ts           （运行时契约）  │  L1 WS /ws/chapter/<id>（流式）  │
        │                                           │  L2 GET …/frame?viz&params&t    │
        │                                           │  L3 data/frames/<ch>/<viz>/*.nlf│
        │                                           └──────┬───────────────────────────┘
        │                                                  │ 二进制 .nlf
        │                    ┌─────────────────────────────▼───────────────────────┐
        │                    │ viz/shared/src/framesource.ts：L1→L2→L3→none 递降   │
        │                    │ （节流 / 退避 / 降级徽标，每一层都发 status）        │
        │                    └─────────────────────────────┬───────────────────────┘
        ▼                                                  ▼
 viz/chapters/<id>/<viz>.model.ts  ◄───────────────  Frame（解码后）
        │  纯函数：Frame → Scene（无 DOM，可在 node 里直接测）
        ▼
   ┌────┴─────────────────────────────────────────────┐
   │ Scene（唯一的可视化中间表示）                      │
   └────┬───────────────────────────────┬─────────────┘
        │                               │
        ▼                               ▼
 viz/shared/src/view.ts           tools/raster.py
 render2d.ts → canvas（浏览器）    同一 Scene → PNG（无浏览器）
        │                               │
        ▼                               ├─► tools/visual_regression.py（G8，基线比对）
 viz/site/app.ts（目录/路由/分享）      ├─► tools/export_video.py（帧序列 → GIF）
                                        └─► tools/build_book.py（讲义插图 / PDF）
```

**关键不变量**：`Scene` 是两侧唯一的可视化中间表示。
浏览器用 canvas 画它，Python 用光栅器画它，视觉回归对后者做像素级基线，
node 单测对前者做结构快照。于是"没有浏览器也能做视觉回归"。

三个由此推出的工程结论（都会被门禁强制执行）：

1. **模型是纯函数**（`(Frame, opts) → Scene`），所以 `node dist/viz/tests/scene/dump_scenes.js`
   能把"用户真正会看到的那张图"批量转储成 JSON——视觉回归测的是**真模型**，不是另写一套探针。
2. **`Scene` 必须自描述**：轴是否自适应、有没有溢出、数据来源、临界值、陷阱文案都在里面，
   `validateScene()`（TS/Python 各一份）把"防误导七条"变成契约错误而不是风格建议。
3. **降级要留下痕迹**：L1→L2→L3→none 每降一层都发 `status`，UI 上显示徽标；
   `precomputed`/`none` 时的图**不准**冒充实时计算结果（规则 4/5）。

---

## 2. 模块地图与职责

### 2.1 `numcore/`（数学真源）

- 只依赖 `numpy`（容器 + 原语），函数纯、无 IO、无全局状态（POLICY P5/P6）。
- 每个模块 docstring 给出**数学定义**与**算法出处**（教材章节 / 论文），
  例如 `lu.py`：`PA = LU`；`pseudospec.py`：`Λ_ε(A) = {z : σ_min(zI − A) ≤ ε}`。
- `numcore/verify/` 是"裁判"，被所有测试 import：
  | 模块 | 提供 |
  |---|---|
  | `order.py` | `fit_order`（双对数最小二乘）、`consecutive_orders`、`assert_order`（±0.1） |
  | `rates.py` | `estimate_linear_factor`、`assert_linear_convergence`、`find_threshold`（二分实测临界值） |
  | `invariants.py` | 守恒量、正交性、对称性、单调性、有界漂移 |
  | `tolerance.py` | `unit_roundoff`、`scaled_tol`、`backward_error_ratio`、`safe_log` |
  | `report.py` | `@assertion(...)` 装饰器 + 注册表 + JSON/Markdown 报告 |
  | `stats.py` | 均值/方差/置信区间/KS 检验（UQ 章用，避免 tests 里手搓统计） |
- `numcore/reference/`：解析解、精确解、高精度（mpmath）、SciPy 对拍。
  **这是全仓库唯一允许 import scipy 的目录**（POLICY P1）。

### 2.2 `server/`（通道，不含数学）

| 文件 | 职责 |
|---|---|
| `nlf.py` | `.nlf` 编解码（PROTOCOLS §1） |
| `manifest.py` | `validate_manifest`：规则 1/3/4/6/7 的机器化 |
| `registry.py` | 读 `docs/chapters.yaml`，动态加载 `server/chapters/<id>.py`，缓存 |
| `chapters/<id>.py` | 每章的 `MANIFEST` + `compute`（把 numcore 的纯函数包成帧） |
| `ws.py` | 自研 RFC 6455 服务端子集（握手 / 掩码 / 分片 / ping-pong / close） |
| `wsclient.py` | 自研客户端，供测试与 `tools/` 使用 |
| `app.py` | `http.server` 上的 HTTP + WS 路由、静态资源、REST API（PROTOCOLS §3）；`t` 对齐 + 帧缓存命名空间 |
| `frames.py` | 帧装配：`normalize_params`、`params_hash`、`build_frame`、`FrameCache`（线程安全 LRU）、`static_frame_path`（L3 路径的权威定义）、`precompute_chapter` |

`server/` 里**没有数学**：所有物理量都来自 `numcore`，这里只做参数校验、时间对齐、
打包与传输。`tools/precompute.py`（CLI）只是 `frames.precompute_chapter` 的壳。

### 2.3 `viz/`（渲染，不做物理）

```
viz/shared/src/
  nlf.ts            帧解码（含 zlib 经 DecompressionStream）
  frame.ts          帧类型（数组视图零拷贝）
  scene.ts          Scene 类型 + validateScene（防误导七条的机器化）
  geometry.ts       轴 ↔ 像素映射、节点采样（热力图行序约定在这里）
  render2d.ts       Scene → CanvasRenderingContext2D（含热力图快/慢两条路径）
  axes.ts           刻度（1-2-5 规则）、数字/科学计数法格式、双对数刻度
  colorscale.ts     LUT（与 Python 共享 colorscale_data.gen.ts）
  colorscale_data.gen.ts  由 tools/gen_colorscales.py 生成，勿手改
  loglogfit.ts      双对数拟合（实测斜率 + 理论斜率参考线，规则 2；`head=`/`tail=` 窗口语义）
  framesource.ts    L1→L2→L3→none 递降的数据源（节流、退避、代际取消、降级徽标）
  timeline.ts       时间轴（stream/scrub/none，可注入时钟，丢帧计数）
  controls.ts       参数控件：滑块 + 数值框 + 临界值标记 + 越界警告（规则 6）
  pitfalls.ts       "它会怎样骗你"面板 + 徽标（规则 7/4/1）
  export.ts         canvas → PNG；Scene → JSON
  share.ts          链接状态编解码（`v1.` + base64url）、base64 原语
  view.ts           DOM 装配层：把上面这些接到 canvas/控件/时间轴上（章节无关）
  manifest.ts       manifest 的 TS 侧类型 + 校验（与 server/manifest.py 对称）
  chapters.ts       章节索引类型 + 分册（Part）分组
  chapters.gen.ts   由 tools/gen_manifests.py 生成，勿手改
  model.ts          章节模型的公共工具层（配色、缺字段策略、角标槽位、排版）
  numeric/*.ts      与 Python 对拍的算法实现（P8 白名单，仅供 parity 使用）
viz/chapters/<id>/
  manifest.gen.ts   由 Python 生成，勿手改
  <viz>.model.ts    纯：Frame → Scene（可在 node 里直接测）
  <viz>.viz.ts      挂载入口：把 manifest+model 交给 shared/view.ts（**注册表按它统计"已实现"**）
  index.ts          viz id → model 的登记表（+ 再导出 manifest）
viz/chapters/index.gen.ts   id → { manifest, models }，由 tools/gen_manifests.py 生成
viz/site/           index.html / site.css / app.ts（目录树、路由、分享链接）
viz/tests/scene/    dump_scenes.ts：把真模型跑出来的 Scene 批量写成 dist/scenes/**.json
```

### 2.4 `tools/`（工程与产物）

| 工具 | 用途 | 门禁 |
|---|---|---|
| `check_policy.py` | AST 扫描 POLICY P1–P13（含"五件套完整性"P13） | G1 |
| `run_asserts.py` | 断言声明↔实跑双向核对 + `asserts_report.md` + `asserts_table.tex` 源 | G3/G4 |
| `gen_manifests.py` | Python manifest → `manifest.gen.ts` / `data/manifests/**.json` / `index.gen.ts` | G6 |
| `gen_colorscales.py` + `colorscale_tables.py` | 色标 LUT → 双侧共享 TS 常量 | G6 |
| `gen_fixtures.py` | `.nlf` 边界样本（解码器对拍用） | — |
| `parity.py` | Python↔TS 逐点对拍（golden 生成/校验 + 容差判定） | G7 |
| `precompute.py` | L3 静态帧（`data/frames/…`）+ `--prune` | G8 前置 |
| `visual_regression.py` | TS 场景转储 → 光栅基线比对（`--check/--update/--require-scenes`） | G8 |
| `raster.py` | 参考光栅器：Scene → PNG（纯 stdlib + 内嵌位图字体） | — |
| `export_video.py` + `gif.py` | 帧序列 → GIF（自研 LZW） | — |
| `png.py` | 纯 stdlib PNG 编解码（栅格器/基线/差异图共用） | — |
| `build_book.py` | 逐章 xelatex + 合并本 + 断言附表 | G9 |
| `gen_progress.py` | 五件套完成度 → `docs/PROGRESS.md`（易变段与确定性段分离） | G10 |
| `ci.sh` | 串联 G1–G10（失败不早停，逐门汇总 → `dist/reports/ci.json`） | `make ci` |

**自研而非依赖**的理由：`fastapi/uvicorn/websockets` 不在允许依赖内（POLICY P7 精神同样适用于 Python 侧的"重依赖"），
且 WebSocket 子集 + PNG + GIF 三类编码器各有 ~200–400 行、可被单测完全覆盖的规模，
自研反而**更可控、更可验证**，并顺带成为教学内容（帧协议本身就是数值实验的一部分）。

---

## 3. 一次交互的完整时序（以 1.2 的 LU 热力图为例）

1. 站点路由 `#/1.2/lu_heatmap` → 加载 `viz/chapters/1.2/lu_heatmap.viz.ts` 与 `manifest.gen.ts`。
2. `FrameSource` 连接 `ws://…/ws/chapter/1.2`，发 `subscribe{viz, params}`。
3. 服务端 `registry.get("1.2").compute(params, t)` → `numcore.linalg.lu.lu_with_history(A)` →
   把每步的矩阵快照、增长因子、残差打包成 `Frame`（meta 里带 `rho/backward_error/step`）。
4. `nlf.pack_frame` → WebSocket 二进制帧 → 浏览器 `decodeNlf` → `Frame`。
5. `<viz>.model.ts::buildScene(frame, state)` → `Scene`（含 `heat` 层 + `loglogfit` 层 + 误差副轴 + 徽标）。
6. `render2d.ts` 画到 canvas；`controls.ts` 的滑块标注 `ρ` 的理论临界值；越界时 `pitfalls.ts` 弹出提示。
7. 同一 `Scene` 由 `visual_regression.py` 用 `raster.py` 画成 PNG，与基线比对（G8）。
8. 断言在 `tests/1.2/` 里独立跑，理论值与实测值写进 `chapters/1.2/asserts_report.md`，并被 `notes.tex` 引用。

**任何一步都不允许"前端自己算一遍物理量"**（P8）；前端只允许在 `numeric/` 内实现"与方法无关的显示辅助"，
且必须通过对拍。

---

## 4. 测试策略（金字塔）

```
        ┌────────────────────────────────────────────┐
   极少 │ 端到端：真 HTTP + 真 WS + 真帧              │  tests/test_app.py（117 条）
        ├────────────────────────────────────────────┤
        │ 视觉回归：真模型 → Scene → PNG 基线         │  tests/test_raster.py + G8
        ├────────────────────────────────────────────┤
        │ 双实现对拍：Python ↔ TS 逐点                │  tests/test_parity.py + G7
        ├────────────────────────────────────────────┤
        │ 章节断言：A/B/C 三类 + 声明↔实跑对账        │  tests/<id>/ + G3
        ├────────────────────────────────────────────┤
        │ 工具链元测试：检查器自己必须可测            │  tests/test_tools.py, tests/policy/
        ├────────────────────────────────────────────┤
   最多 │ 单元测试：numcore/server/viz                │  tests/test_*.py, viz/tests/*.test.ts
        └────────────────────────────────────────────┘
```

**不写"看起来对"的测试**：每个数值测试都要么对拍解析解（A），要么验证不变量（B），
要么测量收敛率/阈值（C）。禁止 `assert result is not None` 式占位（POLICY §3）。

---

## 5. M0 的模板章（0.0）

`chapters/0.0/` + `server/chapters/0.0.py` + `viz/chapters/0.0/` + `tests/0.0/` 是**不写进
`docs/chapters.yaml`** 的额外章节（所以 28 章 / 115 可视化的账目不受影响，进度表里带 `*`）。
它必须走完整条管道：manifest → 生成物 → `compute` → `.nlf` → WS/REST/L3 →
前端解码 → 模型 → Scene → canvas/光栅 → 视觉基线 → 断言报告 → 讲义编译 → 合并本。

两个例子都是"有闭式解、且陷阱真实存在"的问题，因此每个门禁都能拿到真值：

| 可视化 | 内容 | 陷阱（`chapters/0.0/pitfalls.md` 展开） |
|---|---|---|
| `oscillator_regimes` | 阻尼振子三分支 + ODE 残差副轴 | 自适应纵轴把发散画成收敛；残差到机器精度后加密网格只会测出噪声 |
| `heat_ftcs` | 一维热传导显式格式 + 误差副轴 + 时空热力图 | **用最低模态测稳定性是假阴性**；有限网格阈值 $r^*(n_x)>1/2$ |

它的存在意义：**在写任何真算法之前，先把"管道 + 门禁"证明一遍**——0.0 章跑通之后，
剩下 27 章都是"把真算法插进这条已经验证过的管道"。
