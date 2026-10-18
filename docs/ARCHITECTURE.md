# ARCHITECTURE —— 数据流、模块地图与测试策略

> 读完 `docs/CHAPTERS.md`（要做什么）+ `docs/POLICY.md`（红线）+ `docs/PROTOCOLS.md`（接口）后，
> 本文件回答"代码怎么摆、一次点击的数据怎么流、怎么证明它没撒谎"。

---

## 1. 一张图的数据流

```
              ┌────────────────────────── numcore/（纯函数，无 IO）──────────────────────────┐
              │  fp  linalg  interp  quad  transform  ode  pde  inverse  apps              │
              │  ── 唯一数学真源；reference/ 是唯一可 import scipy 的对拍基准 ──            │
              └───────────────────────────────┬───────────────────────────────────────────┘
                                              │  (纯数组进出)
                       ┌──────────────────────▼───────────────────────┐
                       │ server/chapters/<id>.py                      │
                       │   MANIFEST（参数/临界值/轴/陷阱文本）         │
                       │   compute(params, t) -> Frame                │
                       └───────┬───────────────────────┬──────────────┘
             tools/gen_manifests.py                    │ server/nlf.py
                       │                               │ 打包 .nlf
      viz/chapters/<id>/manifest.gen.ts          ┌─────▼──────────────────────────┐
      data/manifests/<id>.json                   │ WS /ws/...（流式，二进制帧）    │
                       │                         │ GET /api/.../frame（单帧）      │
                       │                         │ data/precomputed/（静态降级）   │
                       │                         └─────┬──────────────────────────┘
                       └──────────────┬────────────────┘
                                      ▼
                      viz/chapters/<id>/<viz>.model.ts    Frame → Scene（纯函数，无 DOM）
                                      │                    ↑ 可被 node 直接测试
                                      ▼
                      viz/chapters/<id>/<viz>.view.ts     Scene → canvas（浏览器）
                      viz/shared/src/render2d.ts
                                      │
                                      ▼
                      tools/raster.py   同一 Scene → PNG（Python 侧，供视觉回归 / 讲义插图 / GIF）
```

**关键不变量**：`Scene` 是两侧唯一的可视化中间表示。
浏览器用 canvas 画它，Python 用光栅器画它，视觉回归对后者做像素级基线，
node 单测对前者做结构快照。于是"没有浏览器也能做视觉回归"。

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
| `app.py` | `http.server` 上的 HTTP + WS 路由、静态资源、REST API |
| `precompute.py` | L3 预计算网格生成与查表 |
| `frames.py` | 时间轴推进：把 `compute` 生成器包装成"按 t 取帧" |

### 2.3 `viz/`（渲染，不做物理）

```
viz/shared/src/
  nlf.ts            帧解码（含 zlib 经 DecompressionStream）
  framesource.ts    L1→L2→L3 递降的数据源（含节流、取消、插值徽标）
  render2d.ts       Scene → CanvasRenderingContext2D
  scene.ts          Scene 类型 + validateScene
  colorscale.ts     LUT（与 Python 共享 colorscales.json）
  axes.ts           刻度（1-2-5 规则）、符号、双对数刻度
  timeline.ts       时间轴控件（播放/暂停/单步/拖动）
  controls.ts       参数控件：滑块 + 数值框 + 临界值标记 + 越界警告（规则 6）
  loglogfit.ts      双对数拟合（实测斜率 + 理论斜率参考线，规则 2）
  pitfalls.ts       "它会怎样骗你"面板 + 徽标（规则 7/4/1）
  export.ts         canvas → PNG；帧序列 → GIF（浏览器端动画录制）
  share.ts          参数预设分享链接（URL hash 编解码）
  numeric/*.ts      与 Python 对拍的算法实现（P8 白名单）
viz/chapters/<id>/
  manifest.gen.ts   由 Python 生成，勿手改
  <viz>.model.ts    纯：Frame → Scene
  <viz>.view.ts     DOM：Scene → canvas + 控件装配
  <viz>.viz.ts      导出 { manifest, mount(el, source) }（站点按此发现）
viz/site/           目录树、进度、搜索、路由、主题
```

### 2.4 `tools/`（工程与产物）

| 工具 | 用途 | 门禁 |
|---|---|---|
| `check_policy.py` | AST 扫描 POLICY P1–P6 | G1 |
| `run_asserts.py` | 断言声明↔实跑双向核对 + 报告 | G3 |
| `gen_manifests.py` | Python manifest → TS/JSON 生成物 | G6 |
| `gen_colorscales.py` | 色标 LUT → 双侧共享 JSON | G6 |
| `parity.py` | Python↔TS 逐点对拍 | G7 |
| `visual_regression.py` | 场景快照 + 光栅基线 | G8 |
| `build_book.py` | 每章 xelatex 编译 + 合并本 | G9 |
| `gen_progress.py` | 五件套完成度 → `docs/PROGRESS.md` | G10 |
| `export_video.py` / `export_gif.py` | 帧序列 → GIF（自研 LZW）/ MP4（有 ffmpeg 才用） | — |
| `png.py` / `raster.py` / `colorscales.py` | 纯 stdlib PNG 写出器、Scene 光栅器 | — |
| `precompute.py` | L3 网格预计算 | — |
| `ci.sh` | 串联 G1–G10 | `make ci` |

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
        ┌───────────────────────────────┐
   极少 │ 端到端：假章节 0.0 全链路       │  tests/e2e/test_fake_chapter.py
        ├───────────────────────────────┤
        │ 视觉回归：Scene 快照 + PNG 基线 │  tools/visual_regression.py (G8)
        ├───────────────────────────────┤
        │ 双实现对拍：Python ↔ TS         │  tools/parity.py (G7)
        ├───────────────────────────────┤
        │ 章节断言：A/B/C 三类            │  tests/<id>/（约 1000 条）
        ├───────────────────────────────┤
   最多 │ 单元测试：numcore/server/tools  │  tests/test_*.py, viz/tests/*.test.ts
        └───────────────────────────────┘
```

**不写"看起来对"的测试**：每个数值测试都要么对拍解析解（A），要么验证不变量（B），
要么测量收敛率/阈值（C）。禁止 `assert result is not None` 式占位（POLICY §3）。

---

## 5. M0 的"假章节"（0.0）

`chapters/0.0/` + `viz/chapters/0.0/` + `tests/0.0/` 是一个**故意平凡**的章节：
核心只有一条 `y = sin(2πft)·exp(−at)` 的解析曲线（无算法），
但它必须走完：manifest → 生成物 → 服务端 compute → `.nlf` → WS 推送 → 前端解码 → Scene → 光栅化 → GIF 导出 → 断言报告 → 讲义编译。

它的存在意义：**在写任何真算法之前，先把"管道"证明一遍**。之后每章都是"把真算法插进这条管道"。
