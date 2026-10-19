# ROADMAP —— 里程碑、验收标准与复现

> 本文回答三个问题：**现在做到哪了**、**下一个里程碑要交付什么、怎么算做完**、
> **换一台机器怎么复现出同样的进度表**。
>
> 里程碑的原始定义在 [`docs/chapters.yaml`](chapters.yaml) 的 `milestones:` 与
> [`docs/CHAPTERS.md`](CHAPTERS.md) §6；完成度真源是 [`docs/PROGRESS.md`](PROGRESS.md)。

---

## 0. 口径（先读这段，能省掉很多争论）

1. **完成度只有一个真源**：`docs/PROGRESS.md`。它由 `python3 tools/gen_progress.py`
   从注册表（`server/registry.py`）与断言报告算出，G10 会检查「文档 ≠ 仓库状态」的漂移。
   **不要在任何文档里手写完成度数字**——手写数字会漂移，而且漂移了没人知道。
2. **「完成」= 五件套齐全**：`numcore` 模块、`server/chapters/<id>.py` + manifest、
   可视化数达标、`tests/<id>/` + `asserts.yaml`、`notes.tex`、`pitfalls.md`（≥ 800 字）。
   判定逻辑在 `server/registry.py::ChapterProgress.complete`，可逐项查。
3. **28 章是上限，不是承诺**（`docs/CHAPTERS.md` §6：*「建议：先只做 M0 + M1。样板三章跑通后再决定
   是否全做——28 章是上限而不是承诺。」*）。本文列出的 M2–M6 是**排期意图**，不是已承诺的交付。
4. 工作区可能同时有进行中的改动。**判断「做到哪了」以 `git status` + `docs/PROGRESS.md` 为准**，
   本文的状态段落一律锚定到具体提交。

---

## 1. 当前状态（最终交付）

| 项 | 状态 | 证据 / 怎么自己核 |
|---|---|---|
| **M0 骨架** | ✅ **已完成** | 十道门禁全绿（`make ci`）；模板章 0.0 端到端跑通 |
| 模板章 0.0（不记账） | ✅ 五件套齐全 | 注册表：`vizImplemented 2/2`、`hasModule/manifestOk/hasTests/hasNotes/hasPitfalls/hasAssertsDecl` 全 `true`、`pitfallChars 1970`、`complete true` |
| 28 章正文 | ✅ 28 / 28 | `docs/PROGRESS.md`：章节 `28 / 28`，进度条 `████████████████████` |
| 115 个交互可视化 | ✅ 115 / 115 | `docs/PROGRESS.md`：`115 / 115`（模板章另有 `2 / 2`） |
| 数值断言 | ✅ 3212 条（A 720 / B 2273 / C 219）全绿 | `make asserts` 输出；`dist/reports/assertions.json` |
| 双实现对拍 | ✅ 15 个 case / 4088 个点 | `make parity` 输出 → `dist/reports/parity.json` 的 `n_cases` / `n_points` |
| 视觉回归基线 | ✅ 502 张（29 章、全部场景，0 缺基线） | `make visual` 输出：`ok 502 · 不一致 0 · 缺基线 0` |
| 讲义 | ✅ 合并本 1466 页，未定义引用 0、缺字形 0 | `make book` 输出；`dist/book/book.pdf` |
| **M1 – M6** | ✅ 全部完成 | `docs/PROGRESS.md` 的章节状态表 28 章全为 `✅` |

> **一处提醒**：`docs/PROGRESS.md` 的「门禁状态」表格是**易变段**（每次生成都刷新），
> 且它只覆盖 7 道门（G1/G3/G6/G7/G8/G9/CI）；要确认「十道门是否真的都跑过」，
> 看 `dist/reports/ci.json`（`{"ok": true, "failed": 0, "gates": [...]}` 含每道门的耗时与日志路径）。

---

## 2. 里程碑总表

| 里程碑 | 目标 | 包含章节 | 验收标准（对应门禁） | 状态 |
|---|---|---|---|---|
| **M0 骨架** | 数值核心协议、前端渲染协议、断言框架、WS 通道、导出工具全部就位 | 模板章 0.0（不记账） | G1–G10 全绿；0.0 章五件套齐全（§3.1） | ✅ |
| **M1 样板三章** | 三种最典型的可视化范式成型，模板可复制 | 1.2（消元与稀疏）、1.4（伪谱）、3.2（稳定域） | 三章各自五件套齐全 + G7 覆盖对应算法 + 基线入库（§3.2） | ✅ |
| **M2 Part I 完成** | 线性代数地基可用 | 1.1 – 1.5 | Part I 全部章节 `complete`；跨章依赖（1.2→1.3→1.4→1.5）不断链（§3.3） | ✅ |
| **M3 Part III 完成** | ODE 章可对外展示（视觉冲击最强） | 3.1 – 3.5 | Part III 全部章节 `complete`；3.3 的长期能量漂移、3.5 的 Lyapunov 指数有 A/C 类断言（§3.4） | ✅ |
| **M4 Part IV 完成** | PDE 与 FEM 收尾（工作量峰值） | 4.1 – 4.6 | Part IV 全部章节 `complete`；4.4（FEM）与 4.5（多重网格）工作量与前置依赖已消化（§3.5） | ✅ |
| **M5 Part V + 6.1** | 反问题与综合仿真 | 5.1 – 5.4、6.1 | Part V 全部 `complete`；6.1 与公开基准算例对齐（§3.6） | ✅ |
| **M6 打包** | 交互教材与视频产出 | 6.2 | 目录树/搜索/分享链接可用；一键导出 GIF/MP4；讲义合并本无未定义引用（§3.7） | ✅ |

---

## 3. 逐个里程碑

### 3.1 M0 骨架 —— ✅ 已完成

- **目标**：在写任何真算法之前，先把「数值内核 → 帧 → 前端场景 → 断言 → 视觉回归 → 讲义」
  这条管道和十道门禁**证明一遍**。
- **交付**：`numcore/verify/`（断言裁判）、`server/`（nlf / manifest / registry / frames / ws / app）、
  `viz/shared/src/`（Scene / render2d / framesource / controls / timeline / loglogfit）、
  `tools/`（策略检查、生成器、对拍、光栅器、视觉回归、GIF、讲义、进度、`ci.sh`）、
  模板章 0.0（阻尼振子 + FTCS 热传导，两个都有闭式解，所以每个门禁都能拿到真值）。
- **验收标准（已完成，可复核）**：
  - `make ci` 十道门全绿（约 2 分钟）；
  - 0.0 章注册表 `complete: true`（2/2 可视化、11 条断言 A5/B4/C2、12 张基线、笔记编译通过）；
  - `python3 tools/gen_progress.py --check` 与 `python3 tools/gen_manifests.py --check` 无漂移。
- **已知风险**：无（这是已完成态）。遗留的小缺口见 §4 的「已知缺口」。

### 3.2 M1 样板三章 —— ✅ 已完成

- **目标**：把三种最典型的范式定型，后面 25 章都是「把真算法插进已经验证过的管道」。
- **包含章节**：`1.2` 线性方程组直接法（热力图 + spy plot 范式）、
  `1.4` 特征值/伪谱（等高线 + 收敛曲线范式）、`3.2` 稳定性与刚性（复平面域 + 边界越界告警范式）。
- **验收标准**：每章都要过——
  - **G1**：`numcore/linalg/{lu,chol,sparse,ordering}.py`、`eigen.py`、`pseudospec.py`、
    `ode/{implicit,stability}.py` 过 P1–P13 静态检查（自研、无 scipy、纯函数）；
  - **G3**：`asserts.yaml` ↔ pytest 双向一致，且 `chapters.yaml` 要求的类别齐全
    （1.2 `[A,B]`、1.4 `[A,C]`、3.2 `[A,C]`）；
  - **G6**：三章的 `manifest.gen.ts` / `data/manifests/*.json` 与真源一致；
  - **G7**：`tools/parity.py::CASE_SPECS` 覆盖 `lu`、`power_iter`、`stability` 等 TS 侧实现，
    每个 case 在 `tolerances.yaml` 里带 `reason`；
  - **G8**：三章的可视化进入 `tests/baselines/`，且 `dump_scenes.ts` 的场景数随之增长；
  - **G9/G10**：讲义编译通过、进度表显示三章 `complete`。
- **已知风险**：
  - **范围**：1.2 的图论视角（填充 = 弦化、最小度排序）容易做成「看不懂的图」，
    `docs/CHAPTERS.md` §8 要求这类概念给**逐步点亮**的分步动画；
  - **双实现漂移**：1.4 的伪谱、3.2 的稳定域都涉及复平面上的稠密计算，
    TS 侧若重复实现必须先过 G7，否则前端会撒谎；
  - **基线噪声**：3.2 的稳定域图会因 `autoscaled` 与色标变化整体重刷基线，
    必须按「有意的 bless」流程走（提交信息写明原因）。

### 3.3 M2 Part I 完成 —— ✅

- **包含章节**：`1.1` 浮点数与误差传播、`1.2`、`1.3` 最小二乘与正交化、`1.4`、`1.5` 稀疏迭代与预条件。
- **验收标准**：Part I 五章在 `docs/PROGRESS.md` 中「完成」列为 ✅；
  依赖链 `1.1 → 1.2 → {1.3, 1.4} → 1.5` 的每一条边都有前置产物可 import；
  每章至少一条 C 类断言（收敛因子/条件数/阈值由脚本判定）。
- **已知风险**：1.1 的「灾难性抵消 / 求和顺序」可视化容易出现「图好看但没有理论参考线」，
  违反防误导规则 2——双对数图必须同时画实测斜率与理论斜率。

### 3.4 M3 Part III 完成 —— ✅

- **包含章节**：`3.1` 单步法与自适应步长、`3.2`、`3.3` 几何数值积分、`3.4` 边值问题与谱方法、
  `3.5` 动力系统与混沌。
- **验收标准**：Part III 五章 `complete`；`docs/CHAPTERS.md` 点名的两条硬指标落地——
  3.3 的「辛方法能量误差有界、非辛方法线性漂移」、3.5 的「Lyapunov 指数与公认值一致」
  都必须由断言给出数字（不能只看图）。
- **已知风险**：3.5 的实时 3D 渲染（Lorenz + 分岔图）在浏览器里性能吃紧；
  `docs/CHAPTERS.md` §8 的对策是「后端预计算帧 + 前端插值回放」或 WASM，
  前者已由 L3 通道（`data/frames/`）支持。

### 3.5 M4 Part IV 完成（工作量峰值）—— ✅

- **包含章节**：`4.1` 抛物型、`4.2` 双曲型、`4.3` 守恒律与有限体积、`4.4` 有限元、
  `4.5` 多重网格、`4.6` 谱方法与伪谱。
- **验收标准**：Part IV 六章 `complete`；4.1 的实测稳定阈值（`r = 1/2` 的有限网格修正）、
  4.2 的 TVD 性质、4.3 的离散守恒账本（机器零）都必须有 B/C 类断言；
  4.4/4.5 的模块位于 `numcore/pde/fem/`、`numcore/pde/multigrid.py`，且没有触碰 numpy 成品求解器。
- **已知风险**：**这是全项目最重的一段**（4.4 被标注为「工作量最大的一章」，4.5「概念最难，靠可视化救」）；
  实时 PDE/FEM 在浏览器里算不动，必须走 L3 预计算降级；
  建议宽度：M2 之后 8–20 个并行子任务（`docs/chapters.yaml` 的 `parallelism.suggested_width`）。

### 3.6 M5 Part V + 6.1 —— ✅

- **包含章节**：`5.1` 无约束优化、`5.2` 反问题与正则化、`5.3` 伴随方法、`5.4` 不确定性量化、`6.1` 二维综合仿真。
- **验收标准**：Part V 四章 `complete`；6.1 是**全项目唯一以论文基准数值验收**的章节
  （如方腔 Re = 100 / 1000 的中心线速度剖面），需要网格收敛性研究；
  5.4 的随机实验必须经 `numcore.utils.rng(seed)` 保持确定性可复现。
- **已知风险**：6.1 依赖 4.1–4.6 全部就绪（`chapters.yaml` 的 `deps` 列了六个前置），
  是关键的串行瓶颈；5.3 的伴随梯度检验（相对误差 < 1e-6）容易写成「自己验证自己」，
  必须与有限差分对拍。

### 3.7 M6 打包 —— ✅

- **包含章节**：`6.2` 交互式教材打包与视频导出。
- **验收标准**：站点目录树/进度/搜索/参数预设分享链接可用；每章能一键导出 GIF/MP4
  （`tools/export_video.py`，已可跑：`python3 tools/export_video.py --chapter 0.0 --viz heat_ftcs --out /tmp/x.gif --n 6`）；
  讲义合并本编译无未定义引用；视觉回归覆盖全部已实现可视化。
- **已知风险**：导出链路对帧数与时间轴一致性的要求（`docs/CHAPTERS.md` §5 的 6.2 断言），
  以及 `tools/export_video.py` 的帧率/调色板参数组合带来的产物体积膨胀。

---

## 4. 28 章是上限，不是承诺

- 规模上限写在 `docs/chapters.yaml` 的 `totals:`（`chapters: 28` / `visualizations: 115`）
  与 `docs/CHAPTERS.md` §7 的量级估算里（核心代码 6–10 万行、前端 4–9 万行、笔记 500–1000 页、
  断言约 1000 条、112 个可并行子任务）。
- `docs/CHAPTERS.md` §6 的建议是**先只做 M0 + M1**，然后设检查点按兴趣续做，**允许砍章**。
- 关键路径（`docs/chapters.yaml` 的 `critical_path`）：`1.1 → 1.2 → 1.4 → 1.5 → 4.1 → 4.2 → 4.3 → 6.1`；
  这条链上的任何一章延期，都会顺延 M4/M5。
- **缺口清单（2026-10-18 第二轮复核：✅ 已闭合 5 条、⬜ 未闭合 1 条、⚠ 已知取舍 1 组）**：
  - ✅ 已闭合：`tools/gen_colorscales.py --check` 曾经「有实现但没有门禁调用」，色标漂移无人守；
    现已并入 **G6**（`tools/ci.sh` 里与 `gen_manifests.py --check` 用 `&&` 串成一条命令，
    见 `docs/POLICY.md` §5）。
  - ✅ 已闭合：`dist/reports/{manifests,visual_regression}.json` 曾经没有写入者，
    导致 `docs/PROGRESS.md` 的 G6/G8 恒显示「未运行」；`gen_manifests.py` 与 `visual_regression.py`
    现在跑门禁时会把证据落盘（提交 `54c51d9`）。
  - ✅ 已闭合：`tools/raster.py::validate_scene` 没有复刻规则 8 的第二条（文本层压住
    坐标轴装饰带）。现在 `axis_deco_bands` / `text_box_of` / `_text_deco_issues` 与 TS 侧
    `axisDecoBands` / `textBoxOf` 同口径（含"按 `maxWidth` 折行、装饰带按需生成"两个细节），
    31 个场景在两侧都报 0 条布局问题。
  - ✅ 已闭合：1.5 章 `ilu_dropped = -1` 哨兵没有在 `strings` 里说明含义；现在帧里
    多了 `sentinel_note` 与 `ilu_dropped_applicable` / `ilu_dropped_sentinel`。
  - ✅ 已闭合：`Frame.validate()` 拒绝非有限 `meta` 没有单测；`tests/test_nlf.py` 现在
    有三条：meta 的 NaN/Inf、**数组**的 NaN/Inf（新规则，见下）、以及 `pack_frame` 端到端拒绝。
  - ⬜ 仍未闭合：`dist/reports/progress.json` 既没有写入者、也没有读取者。文档里已不再引用它
    （`docs/PROTOCOLS.md` §9 那一行已删）——要恢复这个文件，先得有一个真实的消费方。
  - ⚠ 已知取舍（有意如此，不是缺口）：**两个渲染层各自实现同一套几何**——装饰带/角标盒的常量由
    同一组样例数字钉住；色标只有光栅器画；场景以 TS 转储为首选、Python 探针场景兜底
    （CI 用 `visual_regression.py --require-scenes` 钉住首选路径）；`viz/shared/src/numeric/**`
    只用于 parity 对拍，章节要 import 必须标 `// preview-approx`（详见 `docs/ARCHITECTURE.md` §2.3）。

---

## 5. 新机器复现：从零到「进度表与今天一致」

前置：Python ≥ 3.11（本仓库在 3.13.9 上验证）、Node ≥ 20（验证于 v24.21.0）、
xelatex + `ctex` 宏包（缺了 G9 只降级为结构检查）。

```bash
# 1) 取代码
git clone <repo-url> numerical-lab && cd numerical-lab
git checkout 1b0c6c8                  # 本文的状态锚点；跟最新则省略

# 2) 两个必须的环境变量（见 README §2.2）
export MPLCONFIGDIR="$PWD/.cache/mpl"
export npm_config_cache="$PWD/.cache/npm"

# 3) 依赖（这两个命令都真跑过：npm 报 up to date，pip 报 already satisfied）
python3 -m pip install numpy pytest pyyaml
npm install

# 4) 环境自检：应当打印 python 3.13.9 / numpy / node / xelatex
make setup

# 5) 按依赖顺序自检（任一步红就先修，别往下走）
python3 -m pytest tests -q            # 699 passed
npm run typecheck                     # 0 错误
npm run build --silent && npm test    # 285 条 TS 单测全过

# 6) 生成物 / 断言 / 对拍 / 视觉 / 讲义 / 进度
python3 tools/gen_manifests.py --check     # 5 个文件与真源一致
python3 tools/run_asserts.py               # 11 条断言（A5/B4/C2）全过
python3 tools/parity.py                    # 锚点提交上为 5 case / 1915 点（工作区现为 8 / 2387）
python3 tools/precompute.py --n 6 --quiet  # G8 前置：L3 帧
npm run scene --silent                     # 真模型转储 12 个场景
python3 tools/visual_regression.py --check --require-scenes   # ok 12
python3 tools/build_book.py --check        # 单章 7 页 + 合并本 11 页
python3 tools/gen_progress.py --check      # 进度表无漂移

# 7) 一把梭（等价于上面 5–6 步，失败不早停）
make ci
```

**判定「与今天一致」的标准**：`make ci` 的汇总里十道门全 ✅，
且 `docs/PROGRESS.md` 的确定性段落显示「章节 0 / 28、可视化 0 / 115、模板章 0.0 五件套齐全」。
若数字不同，先看 `git status`（是不是带了别的改动），再看 `dist/reports/*.json`。

---

## 6. 状态查询速查

```bash
# 人类可读：章节状态表 + 断言总览 + 门禁状态
python3 tools/gen_progress.py            # 重新生成 docs/PROGRESS.md
python3 tools/gen_progress.py --check    # 只检查漂移（G10 用的就是它）
python3 tools/gen_progress.py --json     # {"ok": true, "path": "docs/PROGRESS.md", "bytes": …}

# 机器可读：某一章的完成度字段（complete = 五件套齐全）
python3 - <<'EOF'
from server.registry import Registry
import pathlib, json
reg = Registry(pathlib.Path(".")); reg.discover(strict=False)
p = [c for c in reg.summary()["chapters"] if c["id"] == "0.0"][0]
print(json.dumps({k: p[k] for k in ("vizImplemented","vizDeclared","hasModule","manifestOk",
                                    "hasTests","hasNotes","hasPitfalls","hasAssertsDecl","complete")},
                 ensure_ascii=False))
EOF

# 门禁最近一次运行的汇总与逐门日志
cat dist/reports/ci.json
ls dist/logs/                            # G1.log … G10.log
```
