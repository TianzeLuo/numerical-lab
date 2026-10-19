# numerical-lab · 交互式数值方法实验室

> 把数值分析从纸面公式变成**可以拖着看它收敛、看它发散、看它撒谎**的东西。

这是一个「交互式数值分析教材 + 可复现实验平台」：**28 章 / 6 个 Part / 115 个交互可视化**。
每章产出五件套——推导笔记（LaTeX）、自研数值内核（纯函数）、交互可视化、可执行数值断言、
防误导分析。规模与逐章清单见 [`docs/CHAPTERS.md`](docs/CHAPTERS.md)，完成度见
[`docs/PROGRESS.md`](docs/PROGRESS.md)。

当前状态（逐章数字以 [`docs/PROGRESS.md`](docs/PROGRESS.md) 为准，那份文件由 `tools/gen_progress.py`
生成、G10 守着不许漂移）：**28 章 / 115 个可视化全部完成**，0.0 模板章端到端跑通，
G1–G10 十道门禁全绿。全仓实测：3212 条数值断言（A 720 / B 2273 / C 219）全部通过、
502 个场景逐像素基线一致、合并本 1466 页且 0 未定义引用 / 0 缺字形。
里程碑与完成历史见 [`docs/ROADMAP.md`](docs/ROADMAP.md)。

每章的验收口径不是"跑起来了"，而是：`.nlf` 帧契约测试（全仓 500+ 帧，含每个参数角落与
非有限值扫描）、断言报告与 `asserts.yaml` 的**对账**（声明数与实测数必须一致）、
LaTeX 单章编译（**0 缺字形**、0 未定义引用）、以及**无浏览器**的逐像素视觉回归
（Python 光栅器画同一份 `Scene`，与基线比对）。

---

## 1. 这个仓库特殊在哪

**① 数值真值与讲解分离。** 数学只允许长在 `numcore/`（纯函数、无 IO、无全局状态），
讲解只长在 `chapters/<id>/notes.tex`，两者之间隔着一条可执行的管道：`numcore` → `server/chapters/<id>.py`
（`compute` 把纯函数包成帧）→ `.nlf` 二进制帧 → 前端 `model.ts`（`Frame → Scene` 纯函数）→ canvas。
`tools/check_policy.py` 用 AST 静态检查把这条边界钉死（`docs/POLICY.md` P1–P6）。

**② 「防误导七条」被机器化，不是风格建议。** `axes[].autoscaled`、`overflow{flag,note}`、
`requiredAux` 必含 `error`/`residual`、`ParamSpec.critical[]` + `criticalNote`、`pitfalls`（非空且 ≥ 60 字）
都是 manifest 的必填字段，`server/manifest.py::validate_manifest` 与 `viz/shared/src/scene.ts::validateScene`
两侧独立校验——漏填会让 pytest 失败，而不是让图悄悄骗人（`docs/POLICY.md` §4）。

**③ 没有浏览器也能做视觉回归。** `Scene` 是两侧唯一的可视化中间表示：浏览器用 canvas 画它，
Python 用 `tools/raster.py`（纯 stdlib + 内嵌位图字体）画它。于是「用户真正会看到的那张图」
可以在无浏览器环境下逐像素与基线比对（G8），而场景本身由真前端模型转储（`npm run scene`），
不是另写一套探针（`docs/ARCHITECTURE.md` §1）。

---

## 2. 快速开始

### 2.1 环境要求

| 组件 | 本仓库实测版本 | 硬性要求 |
|---|---|---|
| Python | 3.13.9（numpy 2.3.5） | `pyproject.toml`：`requires-python = ">=3.11"`，运行必需 `numpy>=1.24` |
| Node / npm | v24.21.0 / 11.19.0 | `package.json`：`engines.node >= 20` |
| LaTeX | XeTeX 3.141592653-2.6-0.999995（TeX Live 2023/Debian）+ `ctex` | 缺 xelatex 时 G9 只降级为结构检查 |

Python 依赖按需装（`pyproject.toml` 的 extras）：`dev` = pytest + pyyaml（跑测试必需）；
`reference` = scipy/sympy/mpmath（只给 `numcore/reference/` 与对拍用）；`docs` = matplotlib
（只在重新烘焙色标表时用）。TypeScript 侧只有两个 devDependency：`typescript` + `@types/node`
（运行期零第三方依赖，见 §7）。`make setup` **只检查、不安装任何东西**。

### 2.2 两个必须知道的坑（都已经踩过）

**坑 1：`MPLCONFIGDIR`。** 在 HOME 受限的环境里，matplotlib 建不出 `~/.config/matplotlib`，
会退回临时目录并打印一大段警告、拖慢 import：

```
mkdir -p failed for path /home/user/.config/matplotlib: [Errno 13] Permission denied
Matplotlib created a temporary cache directory at /tmp/matplotlib-... because there was an issue
with the default path (/home/user/.config/matplotlib); it is highly recommended to set the
MPLCONFIGDIR environment variable to a writable directory
```

仓库约定把它指向仓库内的 `.cache/mpl`（`tools/ci.sh` 与 `tools/run_asserts.py` 已自动设置）。
手动跑任何可能 import matplotlib 的命令前：

```bash
export MPLCONFIGDIR="$PWD/.cache/mpl"
```

**坑 2：`npm_config_cache`。** 默认缓存 `~/.npm` 可能不可写，npm 会直接报错：

```
npm error To permanently fix this problem, please run:
npm error   sudo chown -R 1000:1000 "/home/user/.npm"
npm error Log files were not written due to an error writing to the directory: /home/user/.npm/_logs
```

仓库约定把 npm 缓存也放进仓库内，保持环境密闭且可清理：

```bash
export npm_config_cache="$PWD/.cache/npm"     # make ci 会自己设
```

### 2.3 自检、起服务、跑门禁

```console
$ make setup
python 3.13.9 numpy 2.3.5
v24.21.0
11.19.0
XeTeX 3.141592653-2.6-0.999995 (TeX Live 2023/Debian)

$ make serve            # HTTP + WebSocket，默认 127.0.0.1:8080
数值方法实验室服务已启动：http://127.0.0.1:8080/
  WebSocket：ws://127.0.0.1:8080/ws/chapter/<chapter_id>
```

浏览器打开 **<http://127.0.0.1:8080/>** 就是站点壳。端口用 `make serve PORT=9000` 改。
（`server/app.py` 默认监听 `127.0.0.1`；加 `--allow-write` 才开放 `POST /api/asserts/<id>`，
`make serve` 已带。）

```console
$ make ci               # 十道门禁一把跑完（失败不早停，逐门汇总 → dist/reports/ci.json）
  ✅ G1 策略红线（POLICY P1–P13 中的 11 条机器规则）  4.41 s
  ✅ G2 Python 测试（pytest，严格模式）     74.35 s
  ✅ G6 生成物无漂移（manifest + 色标）            2.00 s
  ✅ G4 TypeScript 类型检查                       8.02 s
  ✅ G5 TypeScript 测试                             9.64 s
  ✅ G3 数值断言对账（声明↔实测）       2.13 s
  ✅ G7 Python ↔ TS 对拍                          1.09 s
  ✅ G8 视觉回归（帧→TS 场景→栅格基线）   6.42 s
  ✅ G9 LaTeX（逐章 + 合并本）                14.54 s
  ✅ G10 进度文档无漂移                        2.39 s

全部门禁通过。 报告：dist/reports/ci.json
```

**约 2 分钟**跑完一轮；实测规模：Python 侧 **740 条**（`python3 -m pytest tests --collect-only -q`）、
TypeScript 侧 **296 条**（`node --test "dist/viz/tests/**/*.test.js"`，全绿）——数字会随章节增长，
以这两条命令的输出为准。写完一章后该跑哪些、
失败了先看哪里，见 [`docs/AUTHORING.md`](docs/AUTHORING.md) §3。

---

## 3. 目录结构

| 目录 / 文件 | 一句话职责 |
|---|---|
| `numcore/` | 数学真源：纯函数、无 IO（`fp` `linalg` `interp` `quad` `transform` `ode` `pde` `inverse` `apps`）；`verify/` 是断言裁判；`reference/` 是全仓库**唯一**可 import scipy 的目录 |
| `server/` | 数据通道，**不含数学**：`registry.py`（读 `docs/chapters.yaml`）、`manifest.py`（契约校验）、`frames.py`（帧装配 + 缓存）、`nlf.py`（帧编解码）、`ws.py`（自研 RFC 6455 子集）、`app.py`（HTTP + WS 路由）、`chapters/<id>.py`（每章的 `MANIFEST` + `compute`） |
| `viz/` | 渲染层，**不做物理**：`shared/src/`（`scene` `render2d` `framesource` `controls` `timeline` `loglogfit` `numeric/` …）、`chapters/<id>/`（每章一个包，`<viz>.model.ts` + `<viz>.viz.ts`）、`site/`（目录树/路由/分享链接）、`tests/`（296 条 TS 单测 + 场景转储） |
| `chapters/<id>/` | 讲义与文档：`notes.tex`（推导）、`pitfalls.md`（防误导分析，≥ 800 字）、`asserts.yaml`（断言声明）；`asserts_report.md` / `asserts_table.tex` 是**生成物** |
| `tests/` | Python 测试：`<id>/`（三类数值断言）、`test_*.py`（协议/工具/端到端）、`parity/`（对拍 golden 与容差）、`baselines/`（视觉回归基线 PNG，进 git）、`policy/`（检查器的元测试） |
| `tools/` | 工程与产物：策略检查、断言对账、manifest 生成、对拍、光栅器、视觉回归、GIF 导出、讲义编译、进度生成、`ci.sh` |
| `data/` | `manifests/`（生成物，进 git）、`frames/`（L3 静态帧，可 `make precompute` 重建，**不进 git**） |
| `docs/` | 规范与状态：`CHAPTERS.md`（要做什么）、`POLICY.md`（红线）、`PROTOCOLS.md`（接口）、`ARCHITECTURE.md`（怎么摆）、`chapters.yaml`（机器可读清单）、`PROGRESS.md`（生成）、`AUTHORING.md` / `ROADMAP.md` |
| `dist/` | 全部可重新生成：`viz/`（tsc 产物）、`scenes/`、`visual/`、`reports/`、`logs/G1.log…G10.log`、`book/`（PDF）、`export/`。**不进 git** |
| `Makefile` | 所有入口；`make help` 是权威列表 |

---

## 4. 一次点击的数据流（简版）

```
numcore（纯函数，唯一数学真源）
   └─► server/chapters/<id>.py::compute(viz, params, t) ─► Frame
          ├─► .nlf 打包（server/nlf.py）─► L1 WebSocket /ws/chapter/<id>（流式）
          │                                L2 GET /api/chapter/<id>/frame?viz&params&t
          │                                L3 data/frames/<ch>/<viz>/<hash>_t<%.6g>.nlf（静态）
          │                                        │
          │                    viz/shared/src/framesource.ts：L1→L2→L3→none 逐级降级，每级发 status
          ▼                                        ▼
   tools/gen_manifests.py ──► viz/chapters/<id>/manifest.gen.ts / data/manifests/<id>.json
                                              │
                                  viz/chapters/<id>/<viz>.model.ts（Frame → Scene，纯函数、无 DOM）
                                              │
                       ┌──────────────────────┴───────────────────────┐
                       ▼                                              ▼
              render2d.ts → canvas（浏览器）              tools/raster.py → PNG（无浏览器）
                                                              ├─ visual_regression.py（G8 基线）
                                                              ├─ export_video.py（GIF）
                                                              └─ build_book.py（讲义插图 / PDF）
```

关键是那个 `Scene`：它自描述（轴是否自适应、有没有溢出、数据来源、临界值、陷阱文案都在里面），
所以「浏览器里画的」和「回归测的」不可能不是同一张图。完整版（含时序与不变量）见
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

---

## 5. 十道门禁（`make ci`）

命令、通过标准的规范性定义在 [`docs/POLICY.md`](docs/POLICY.md) §5；下表补一列
**真实抓捕记录**（证据截至提交 `1b0c6c8`，都能在仓库里查到）。

| 门 | 命令 | 防住什么 | 抓到过什么 |
|---|---|---|---|
| **G1** | `python3 tools/check_policy.py` | 用现成求解器替代自研实现（P1–P3）、绕过 `rng()` 用 `numpy.random`（P4）、`numcore` 里出现 IO/print（P6）、未登记的豁免注释（`tests/policy/allowlist.txt`） | **一次「假绿」**：`tools/build_book.py` 的 docstring 里 `\input` 是无效转义，`pytest` 每次都打 `SyntaxWarning`，而检查器仍报「0 违规」——警告没人接住就等于没检查 |
| **G2** | `python3 -m pytest tests -q` | 数值内核、帧协议、WS、HTTP、导出链路的回归 | `tools/gif.py` 的两处真 bug 由测试锁死：LZW 收尾码漏判码长增长（只在「最后一枚数据码正好越过阈值」时炸）、`1 << 16 // n` 被解析成 `1 << (16 // n)` 导致量化退化 |
| **G3** | `python3 tools/run_asserts.py --no-run` | 「声明了却没测」「测了却没声明」「靠静默通过混过去」：`asserts.yaml` ↔ pytest 收集**双向**对账 | 诊断文案曾建议用 `tail=` 截掉细网格端，但 `tail=` 保留的恰是被舍入地板污染的**细端**；这条教训被写成测试，并新增 `head=` 参数（`tests/0.0/test_smoke.py::test_fit_order_head_option_escapes_roundoff_floor`） |
| **G4** | `npm run typecheck` | 接口/类型层面的前后端漂移（`tsc --noEmit`，`strict: true`） | **暂无抓捕记录**。反而有一条盲区值得记住：0.0 章曾把字符串写进 `meta`，`pytest` 全绿、`tsc` 全绿，直到前端真解码那一帧才炸（`server/nlf.py` 的 `set_strings` 注释记下了这次事故） |
| **G5** | `npm test` | `.nlf` 解码、`validateScene`、时间轴/控件/分享、渲染与数值 helpers | 「viridis 亮度单调不减」用现成 8 位 LUT 不可能成立（i=95/146 单通道 −1 LSB），测试因此改成逐条可解释的断言；热力图快慢两条路径曾画出**镜像**（由 G8 发现） |
| **G6** | `python3 tools/gen_manifests.py --check && python3 tools/gen_colorscales.py --check` | manifest 生成物（5 个文件）与 Python 真源漂移（从根上消灭「前后端两处声明」），以及**色标 LUT**（`colorscale_data.gen.ts` / `colorscales.json`）与真源 `tools/colorscale_tables.py` 漂移 | **暂无抓捕记录**。这条门曾经只跑 manifest：`tools/gen_colorscales.py --check` 有实现却没人调用，色标漂移无人守；现已并进本门（`tools/ci.sh` 里是一条 `&&` 命令） |
| **G7** | `python3 tools/parity.py` | Python ↔ TS 双实现漂移（前端自己算一遍物理量） | 基准 golden 曾可能「随时间悄悄漂移而门禁照样绿灯」（要求 `--update-golden` 逐字节可重生成）；容差纪律也来自教训：`tolerances.yaml` 明写「绝不靠把 rtol 放大到能过——那是把门禁做成了装饰」 |
| **G8** | `python3 tools/precompute.py --n 6 --quiet && npm run scene --silent && python3 tools/visual_regression.py --check --require-scenes` | 「图好看但算错」：真前端模型 → `Scene` → 光栅 PNG，与基线逐像素比对 | 参考光栅器发现前端热力图快路径**少了一次纵向翻转**——这个 bug 只会在浏览器里「看着不对」，任何纯逻辑测试都抓不到；另外若不加 `--require-scenes`，「一张图都没比」也能绿 |
| **G9** | `python3 tools/build_book.py --check` | 讲义编译失败、未定义引用、章末断言附表与最新报告不一致（PDF 撒谎） | 未转义的下划线在正文数学模式下直接变成 `Missing $ inserted`（工具注释：「这个坑真的踩过一次」）。**已知盲区**：字体缺字（`Missing character`）不算失败，`book.pdf` 里 `≤` 之类字符会被静默吞掉 |
| **G10** | `python3 tools/gen_progress.py --check` | `docs/PROGRESS.md` 与仓库状态漂移——专治「文档说做了、其实没做」 | 反过来的坑：若把「门禁运行记录」段（时间戳/结论）也纳入比较，**每跑一次 CI 都会判成漂移**；于是 `PROGRESS.md` 用一条分界线把易变段隔离在外，只有分界线以上参与判定 |

> **G6/G8/G10 的「漂移」类门禁是本项目的良心**：它们专治「文档说做了、其实没做」与「图好看但算错」。

两点读表须知：

- G1 的规则编号**只认检查器**：`docs/POLICY.md` §1 现已按 `tools/check_policy.py` 的 `RULE_DOC`
  逐条对齐，覆盖 **P1–P8、P10、P12、P13 共 11 条机器规则**；P9 保留为评审项、P11 是空号
  （`tools/ci.sh` 的门名与之一致）。改规则时先改检查器，再同步 POLICY——反过来会立刻漂移。
- **argparse 的前缀匹配是个真陷阱**：`tools/run_asserts.py` 没有 `--all`，写 `--all` 会被当成
  `--allow-empty`（"报告缺失也不算失败"）而**静默变绿**。上表与 `tools/ci.sh` 用的是真正生效的
  `--no-run`。给 `tools/` 里的脚本加选项时，别让两个选项名互为前缀。

---

## 6. 常用 `make` 目标

（`make help` 的真实输出；`make` 不带参数即打印这份列表。）

| 目标 | 用途 | 何时用 |
|---|---|---|
| `make setup` | 检查 python/node/依赖/字体，不装任何东西 | 换机器后第一件事 |
| `make serve` | 起本地服务（HTTP + WebSocket），默认 8080 | 看可视化 / 手工验证接口 |
| `make test` | `pytest` + `jstest` | 提交前 |
| `make pytest` | Python 测试（740 条） | 改了 `numcore/` `server/` `tests/` |
| `make typecheck` | `tsc --noEmit`（strict） | 改了 `viz/**/*.ts` |
| `make jstest` | 先 `tsc` 编译再 `npm test`（296 条） | 改了前端 |
| `make build` | `tsc` 编译到 `dist/viz` | 只想要产物 |
| `make policy` | G1 策略红线静态检查 | 改了 `numcore/` 或加了豁免 |
| `make asserts` | G3 跑断言、生成报告与 `chapters/<id>/asserts_report.md` | 改了 `tests/<id>/` 或 `asserts.yaml` |
| `make parity` | G7 Python ↔ TS 对拍 | 改了 `viz/shared/src/numeric/` 或 `numcore` 对拍口径 |
| `make visual` | G8 视觉回归 `--check` | 改了 `model.ts` / 渲染层 / manifest |
| `make visual-update` | G8 显式更新基线（**必须在提交信息里写明原因**） | 只想有意改图时 |
| `make manifests` | G6 重新生成 manifest 产物 | 改了 `MANIFEST` 或 `docs/chapters.yaml` |
| `make book` | G9 逐章 + 合并本 LaTeX | 改了 `notes.tex` |
| `make progress` | G10 重新生成 `docs/PROGRESS.md` | 开/关一章、改了 `chapters.yaml` |
| `make precompute` | 生成 L3 静态帧到 `data/frames/`（`CHAPTER=<id>` 指定章） | 跑 G8 / 导出视频前 |
| `make ci` | 十道门禁一把跑完（失败不早停） | 提交前最后一道 |
| `make clean` | 删 `dist/` 与缓存（保留 `data/` 契约产物） | 排除构建缓存干扰 |
| `make clean-all` | 连 `data/frames` 一起删（可重新生成） | 想要「全新检出」状态 |

---

## 7. 章节总账与依赖政策

**总账**：`docs/chapters.yaml` 声明 **28 章 / 115 个交互可视化**，分 6 个 Part
（I 基础与数值线性代数 5 章、II 逼近积分与变换 6 章、III 常微分方程 5 章、
IV 偏微分方程与离散化 6 章、V 反问题优化与不确定性 4 章、VI 综合与产出 2 章）。
**模板章 0.0 不记账**：它故意不写进 `chapters.yaml`，所以不影响 28/115 这两个数；
在进度表里它带 `*`（`*` = 未在 `chapters.yaml` 里声明的额外章节）。

**当前完成度以 [`docs/PROGRESS.md`](docs/PROGRESS.md) 为准**（由 `tools/gen_progress.py`
从注册表与断言报告算出，手写数字会漂移，G10 会检查）。截至 `1b0c6c8`：章节 **0 / 28**，
可视化 **0 / 115**，模板章 0.0 五件套齐全（另有 2/2 个可视化）。

**依赖政策（一句话）**：**运行期零第三方依赖**——TypeScript 侧 `node_modules` 只允许
`typescript` + `@types/node` 两个 devDependency；Python 侧只允许 `numpy` 与标准库，
且 `numpy` 只能当容器与原语用（`np.dot`/`@` 可以，`np.linalg.solve` 不行），
`scipy` **只允许**出现在 `numcore/reference/`（全仓库范围，含 `tests/`、`tools/`、`server/`），
并由 G1 用 AST 静态检查强制执行。判定口径很简单：
**如果一个调用直接给出了本章要教的那个算法结果，它就是被禁的。**

---

## 8. 版本与引用

- 版本 `0.1.0`（`docs/chapters.yaml` / `package.json` / `pyproject.toml` 三处一致）。
- 当前提交 `1b0c6c8`（M0 骨架 + 模板章 0.0）。
- 引用本项目时，请写明**章节 id + 提交哈希**（例：`numerical-lab 0.0 @ 1b0c6c8`），
  因为断言数值与视觉基线都会随提交变化。
- 仓库当前**没有** `LICENSE` / `CITATION` 文件，对外复用与引用的授权方式尚未确定——
  正式复用前请先与维护者确认。

---

## 9. 想动手写一章？

从 [`docs/AUTHORING.md`](docs/AUTHORING.md) 开始：它按「先读什么 → 五件套模板与检查清单 →
本地验证循环 → 加一个新可视化 → 双实现对拍 → 常见拒绝原因 → 提交纪律」的顺序，
给出可以照抄的最小样例（都摘自模板章 0.0）。章节排期与验收标准见
[`docs/ROADMAP.md`](docs/ROADMAP.md)。
