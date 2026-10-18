# AUTHORING —— 写一章的完整手册

> 面向「要写第 X 章的人」（人或子 agent）。照着做就能产出**合规章节**：
> 五件套齐全、断言可执行、可视化不撒谎、门禁全绿。
>
> 本文所有模板都摘自**真实文件**（模板章 0.0），复制粘贴即可跑；所有命令都真跑过。
> 规范冲突时以 `docs/POLICY.md` 与 `tools/check_policy.py` 为准（检查器优先）。

---

## 0. 你会产出什么

一章 = 五个交付物，路径固定、缺一不可（`docs/POLICY.md` §2）：

| # | 交付物 | 路径 | 机器判定 |
|---|---|---|---|
| 1 | 推导笔记 | `chapters/<id>/notes.tex` | 存在 + 至少一个 `\begin{theorem}` + xelatex 编译过 |
| 2 | 核心实现 | `numcore/<area>/…`（见 `docs/chapters.yaml` 的 `modules`） | 模块存在、可 import、过 G1 静态检查 |
| 3 | 交互可视化 | `viz/chapters/<id>/<viz>.viz.ts` + `manifest.gen.ts` | 实现数 ≥ `chapters.yaml` 的 `viz:` 声明 |
| 4 | 数值断言 | `tests/<id>/` + `chapters/<id>/asserts.yaml` | 声明与实跑**双向**一致（G3） |
| 5 | 防误导分析 | `chapters/<id>/pitfalls.md` | 存在且 ≥ 800 字 |

`tools/gen_progress.py` 按这五条算完成度，`docs/PROGRESS.md` 的「完成」列就是它。

**顺序建议**：先写 `numcore/`（有真值才谈得上断言）→ `server/chapters/<id>.py`（把真值包成帧）→
`tests/<id>/`（先证明它是对的）→ `viz/`（再让它好看）→ `notes.tex` / `pitfalls.md`（最后写清为什么）。
反过来做（先画图）几乎必然会在断言阶段推倒重来。

---

## 1. 前置：先读三份文档，再改 `docs/chapters.yaml`

### 1.1 先读这三份（顺序别换）

| 顺序 | 文档 | 你要从里面拿到什么 |
|---|---|---|
| 1 | [`docs/POLICY.md`](POLICY.md) | 红线：P1–P13 各管什么、五件套完整性、断言纪律、防误导七条、提交信息格式。**任何一条都不能破。** |
| 2 | [`docs/PROTOCOLS.md`](PROTOCOLS.md) | 跨语言接口：`.nlf` 帧容器与 `meta` 硬约束（§1）、WS/HTTP 协议（§2–4）、manifest 字段约束（§5）、`Scene` 契约 + 布局契约（§6）、对拍协议（§7）、断言纪律与报告（§8）、产物目录（§9） |
| 3 | [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) | 代码怎么摆、一次点击的数据怎么流、`Scene` 为什么是唯一中间表示、测试金字塔 |

你这一章的**内容要求**（推导要点、该做哪几个可视化、核心模块清单、断言方向）在
[`docs/CHAPTERS.md`](CHAPTERS.md) §5 的对应小节里，别自己发明范围。

### 1.2 `docs/chapters.yaml` 的字段（逐条）

一章一行，机器可读真源（`server/registry.py` 读它，`tools/gen_progress.py` 按它算进度）：

```yaml
- {id: "1.2", title: 线性方程组直接法, part: I, deps: ["1.1"], viz: 4,
   difficulty: mid, burn: 3,
   modules: [numcore/linalg/lu.py, numcore/linalg/chol.py, numcore/linalg/sparse.py, numcore/linalg/ordering.py],
   asserts: [A, B]}
```

| 字段 | 含义 | 写错会怎样 |
|---|---|---|
| `id` | 章节号 `主.次`，必须与目录名、文件名一致（`1.2` → `chapters/1.2/`、`server/chapters/1.2.py`、`tests/1.2/`、`viz/chapters/1.2/`） | 注册表认不出这一章，进度表里不出现 |
| `title` | 中文章名，与 `MANIFEST["title"]`、`notes.tex` 的标题保持一致 | 只在文档里不一致，门禁抓不到（但人会看晕） |
| `part` | `I`…`VI`，决定站点的分册分组 | 分组错乱 |
| `deps` | 前置章节 id 列表，用于拓扑排序与派发 | 只影响排期，不影响门禁 |
| `viz` | **交互可视化个数**（如 `4`）。这是硬契约：`viz/chapters/<id>/` 里实际实现的 viz 数必须 ≥ 它 | G10 进度表显示 `2/4`，`complete` 为假 |
| `difficulty` | `low` / `mid` / `high` / `very_high` | 只影响排期 |
| `burn` | 燃烧量 `2` / `3`（投入档位，非精确人日） | 只影响排期 |
| `modules` | 该章核心实现要落的文件/目录（相对仓库根）。注册表逐个检查存在性 | 少一个文件 → 该章 `has_module` 为假 → `完成` 为假 |
| `asserts` | 该章**必须出现**的断言类别（`A` 解析对拍 / `B` 不变量 / `C` 收敛率与阈值） | G3 报「缺少 C 类断言」 |

**改完 `chapters.yaml` 必须跑这三条**（顺序无所谓，但三条都要）：

```bash
python3 tools/gen_manifests.py      # 重新生成 data/manifests/index.json 与 viz/shared/src/chapters.gen.ts
python3 tools/gen_progress.py       # 重新生成 docs/PROGRESS.md（G10 会检查漂移）
python3 tools/gen_manifests.py --check && python3 tools/gen_progress.py --check   # 确认无漂移
```

---

## 2. 五件套：最小可复制模板 + 检查清单

### 2.1 `numcore/**` —— 数学真源

**模板**（真实文件 `numcore/smoke.py` 的开头，注意 docstring 第一行给的是**数学对象**而不是功能描述）：

```python
"""0.0 章（管线自检）的数值内核。

这一章不教新东西，它的存在是为了**证明整条流水线是真的**：
自研实现 → 后端帧 → 前端 Scene → 三类断言 → 视觉回归 → 笔记。
只用 numpy 的基础 ufunc（P3：不得使用任何现成求解器）。
"""

from __future__ import annotations

import numpy as np

from numcore.utils import rng


def damped_oscillator(t: Any, omega: float = 1.0, zeta: float = 0.05,
                      x0: float = 1.0, v0: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """阻尼振子的**精确解**，返回 ``(x(t), v(t))``。

    x'' + 2ζω x' + ω² x = 0，按 ζ<1 / ζ=1 / ζ>1 三支给出闭式解
    （出处：Boyce & DiPrima, *Elementary Differential Equations*, §3.4）。
    """
    ...
```

**检查清单**

- [ ] 文件在 `chapters.yaml` 的 `modules` 里，且路径一字不差。
- [ ] docstring 第一行是**数学定义**（`PA = LU`、`Λ_ε(A) = {z : σ_min(zI − A) ≤ ε}`），并给出**出处**（教材章节/论文）。
- [ ] 公开函数是纯函数：无 IO、无 `print`、无 `global`、无 `open/os/sys/subprocess/socket`（P6，`numcore/verify/{plugin,report}.py` 是唯二登记豁免）。
- [ ] 只用 numpy 的容器与原语（`ndarray`、逐元素算术、切片/广播、`np.dot`/`@`、`concatenate/linspace/where/clip/abs/sqrt/exp/sin/sum/mean/max/argmax/isfinite/cumsum/outer/einsum`…）。
- [ ] **没碰** `np.linalg.*`、`np.fft.*`、`np.polyfit/polyval/interp/trapezoid/gradient/roots/lstsq/solve/inv/det/eig/svd/qr/cholesky/norm/…`（P2/P3，完整名单在 `tools/check_policy.py`）。
- [ ] **没碰** scipy / matplotlib / pandas / sympy / mpmath / sklearn / numba（P1/P5）；要对拍就把它写进 `numcore/reference/`。
- [ ] 需要随机数时用 `numcore.utils.rng(seed)`，**不直接用** `np.random`（P4）。
- [ ] 公开函数签名统一 `(输入数组, 关键字参数) -> 结果`。
- [ ] 顺手写边界校验（`nx 至少 11`、`zeta 必须 ≥ 0`…）并用 `pytest.raises(ValueError, match=...)` 测掉。

跑：`make pytest`，或只跑本章 `python3 -m pytest tests/<id> -q`。

### 2.2 `server/chapters/<id>.py` —— 把纯函数包成帧

**模板**（真实文件 `server/chapters/0.0.py`，删去第二个 viz 的细节）：

```python
"""1.2 章：线性方程组直接法。

帧只做搬运：所有物理量来自 numcore，本文件不写数学。
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from numcore.linalg.lu import lu_with_history
from server.nlf import Frame

MANIFEST: dict[str, Any] = {
    "chapter": "1.2",
    "title": "线性方程组直接法",
    "part": "I",
    "summary": "一句话说清这一章要用帧讲什么。",
    "vizzes": [
        {
            "id": "lu_heatmap",                      # 必须与 viz/chapters/1.2/lu_heatmap.viz.ts 同名
            "title": "消元过程热力图",
            "description": "每一步的行操作 + 增长因子与当前残差。",
            "dataSource": "backend",                 # backend | backend+preview
            "params": [
                {"name": "n", "label": "阶数 n", "type": "int",
                 "min": 4, "max": 128, "default": 24,
                 "critical": [], "criticalNote": ""},
                {"name": "rho", "label": "增长因子阈值", "type": "float",
                 "min": 1.0, "max": 1e4, "step": 0.1, "default": 10.0,
                 "critical": [128.0],
                 "criticalNote": "部分主元下 ρ ≤ 2^{n−1}；这里给 n=8 的上界作为参考线。",
                 "beyond": "above"},
            ],
            "time": {"mode": "scrub", "tMax": 1.0, "dt": 0.02, "fps": 30, "label": "step"},
            "requiredAux": ["error"],                # 规则 3：必须含 error 或 residual
            "axes": [
                {"id": "main", "autoscaled": False, "title": "U 因子", "ylabel": "列"},
                {"id": "error", "autoscaled": True, "title": "后向误差", "ylabel": "‖δA‖", "scale": "log"},
            ],
            "pitfalls": "热力图用对数色标会把 1e-16 和 1e-3 画得一样深；……（≥ 60 字，写清它会怎样骗你）",
            "theory": ["后向误差界 ‖ΔA‖ ≤ ρ·u·‖A‖", "Cholesky 存在 ⟺ A 对称正定"],
        },
    ],
}


def compute(viz_id: str, params: Mapping[str, Any], t: float) -> Frame:
    """章节计算入口：``params`` 已经过 :mod:`server.frames` 规范化与区间校验。"""
    if viz_id == "lu_heatmap":
        return _lu_frame(params, t)
    raise KeyError(f"章节 1.2 没有可视化 {viz_id!r}")


def _lu_frame(params: Mapping[str, Any], t: float) -> Frame:
    n = int(params["n"])
    out = lu_with_history(np.asarray(..., dtype=float))

    fr = Frame(chapter="1.2", viz="lu_heatmap", kind="frame")
    fr.add("U", out.U, unit="", label="U 因子")          # 数组一律走 arrays
    fr.add("error", out.backward_error, unit="", label="后向误差")
    fr.set_meta(rho=out.rho, n=n, t_actual=float(t))     # meta 只允许 number/bool/null
    fr.set_strings(status="ok", pivot="partial")         # 字符串一律走 strings
    return fr
```

**`MANIFEST` 字段含义**（校验器是 `server/manifest.py::validate_manifest`，位点是 `server/manifest.py:318`）：

| 字段 | 必填 | 约束 / 含义 |
|---|---|---|
| `chapter` / `title` / `vizzes` | ✅ | `chapter` 形如 `1.2`；`title` 非空；`vizzes` 非空 |
| `vizzes[].id` | ✅ | 小写下划线标识符，**同章唯一**，且与 `<viz>.viz.ts` 文件同名 |
| `vizzes[].title` / `description` | ✅ / 建议 | 控件面板上显示 |
| `vizzes[].dataSource` | ✅ | `backend`（数据全部来自后端）或 `backend+preview`（前端另有近似预览，必须打徽标，见 §5） |
| `vizzes[].params[]` | ✅ 非空 | `name`（唯一）、`label`、`type`（`int/float/enum/bool`）、`default`；数值型给 `min/max/step`；`enum` 给 `options`（≥ 2）且 `default` 在其中 |
| `params[].critical[]` | ✅ **必写** | 理论临界值列表；没有也要写 `[]`（规则 6） |
| `params[].criticalNote` | `critical` 非空时必填 | 写清临界值的**依据**（哪条定理/哪个界） |
| `params[].beyond` | 可选 | `above` / `below` / `magnitude`，越界提示的方向，默认 `above` |
| `vizzes[].time` | ✅ | `{"mode": "none"}` / `stream` / `scrub`；非 `none` 时必须给正的 `tMax`、`dt`（`dt ≤ tMax`）、可选 `fps`（1–120）。`mode="none"` 时**不得**出现 `tMax/dt/fps`（规则 4） |
| `vizzes[].requiredAux[]` | ✅ | **必须含 `error` 或 `residual`**（规则 3） |
| `vizzes[].axes[]` | ✅ 非空 | 每项要有 `id`（唯一）、显式的 `autoscaled`（bool，规则 1/4）、可选 `scale: linear/log` |
| `vizzes[].pitfalls` | ✅ | ≥ **60 字**（脚本按字符数算，规则 7） |
| `vizzes[].theory[]` | ✅ 非空 | 至少一条理论依据，`notes.tex` 里要有对应推导 |

**`compute` 的签名与返回值**：`compute(viz_id: str, params: Mapping[str, Any], t: float) -> Frame`。
`params` 已经过 `server/frames.normalize_params` 规范化（缺省值填充 + 区间校验 + 类型转换），
**不需要**再校验一遍；`t` 是请求时刻。

**帧的规矩**（`docs/PROTOCOLS.md` §1.1）：

- 数组一律 `fr.add(name, array, unit="", label="")`；`dtype` 只允许 `f8 f4 i8 i4 i2 u1 u2 u4 bool`（**没有 f2**）。
- 标量曲线量（残差、阶、时间…）走 `fr.set_meta(**scalars)`，**只允许 number / bool / null**。
- 字符串走 `fr.set_strings(**strings)`。把字符串塞进 `meta` 是最经典的翻车方式：
  `pytest` 全绿、`tsc` 全绿，直到前端真解码那一帧才炸。
- 请求时刻 ≠ 到达时刻：显式格式只能整步推进，把**实到时刻**写进 `meta.t_actual`（0.0 章的做法）。
- `server/` 里**没有数学**。任何物理量都必须来自 `numcore`。

**检查清单**

- [ ] `MANIFEST["chapter"] == "<id>"`，`vizzes[].id` 与文件名一一对应。
- [ ] 每个 `params[]` 都有 `critical`（哪怕 `[]`）；`critical` 非空时有 `criticalNote`。
- [ ] `requiredAux` 含 `error` 或 `residual`；副轴在 `axes` 里有对应条目。
- [ ] 每个 `axes[]` 都有显式 `autoscaled`。
- [ ] `pitfalls` ≥ 60 字，且**具体**（说清哪个图在什么参数下会骗人）。
- [ ] 帧里没有把字符串塞进 `meta`；没有在 `server/` 里写数学。
- [ ] `meta` 里没有 NaN/Inf（"没有有限值"写 `None`），**数组里也没有**（占位值 + 掩码 + `strings`
      说明三件套，见 `PROTOCOLS.md` §1.3.1；`Frame.validate()` 现在会当场拒绝）。
- [ ] **参数上界真的扫到过**：每个参数的 `min`/`max`、每个 enum 的两端、`t ∈ {0, tMax}`
      都要能构建并编码（`tests/test_frames_contract.py` 会逐个跑；单章可自己再扫一遍）。
- [ ] **数值路线失败要变成可见状态，不许冒充**：某条路线没有有限解时，帧里要说得清
      "哪条路线失败 + 其余路线照常"（1.3 的 `method_singular`、1.2 的 `error_valid`、
      3.2 的 `abs_R_pole` 是三个范例）。用 0、上限值或 NaN 冒充失败是最严重的误导。

跑：`python3 tools/gen_manifests.py --check`（G6）、`make pytest`（`tests/test_manifest.py`、`tests/test_registry.py`）。

### 2.3 `viz/chapters/<id>/` —— 帧变成 Scene

一个 viz 拆成三块（真实文件在 `viz/chapters/0.0/`）：

```ts
// 1) lu_heatmap.model.ts —— 纯函数：Frame → Scene（无 DOM、无 IO、无第三方依赖）
import type { Frame } from "../../shared/src/frame.js";
import type { ModelOptions, SceneModel } from "../../shared/src/model.js";
import type { Scene } from "../../shared/src/scene.js";
import { manifest } from "./manifest.gen.js";

const VIZ = "lu_heatmap";
const TITLE = manifest.vizzes.find((v) => v.id === VIZ)?.title ?? "消元过程热力图";

export const luHeatmapModel: SceneModel = (frame, opts) => {
  const U = frame.arrays("U");          // 缺数组要抛 ModelInputError，别静默画空图
  const err = frame.arrays("error");
  const scene: Scene = {
    version: 1, chapter: "1.2", viz: VIZ, title: TITLE, t: frame.meta.t ?? 0,
    canvas: { w: opts.width, h: opts.height, background: "#ffffff" },
    axes: [/* main + error 副轴，autoscaled 如实标注 */],
    layers: [/* heat / line / loglogfit … */],
    texts: [], badges: [/* source 徽标必须有 */],
    overflow: { flag: false, note: "" },
  };
  return scene;                          // 交给 validateScene 与渲染层
};
```

```ts
// 2) lu_heatmap.viz.ts —— 挂载入口：把 manifest spec、model、宿主元素交给 mountViz
import type { VizView, VizViewOptions } from "../../shared/src/view.js";
import { mountViz } from "../../shared/src/view.js";
import { manifest } from "./manifest.gen.js";
import { luHeatmapModel } from "./lu_heatmap.model.js";

export const vizId = "lu_heatmap";

export function vizSpec() {
  const spec = manifest.vizzes.find((v) => v.id === vizId);
  if (!spec) throw new Error(`manifest 里没有 ${vizId}（manifest.gen.ts 需要重新生成？）`);
  return spec;
}

export function mount(host: HTMLElement, opts: Partial<VizViewOptions> = {}): Promise<VizView> {
  return mountViz({ manifest, viz: vizSpec(), model: luHeatmapModel, host, ...opts });
}
```

```ts
// 3) index.ts —— viz id → model 的登记表（不认识 DOM；按它统计"已实现"）
import type { SceneModel } from "../../shared/src/model.js";
import { luHeatmapModel } from "./lu_heatmap.model.js";

export const models: Record<string, SceneModel> = { lu_heatmap: luHeatmapModel };
export { manifest } from "./manifest.gen.js";

export function modelFor(vizId: string): SceneModel | null {
  return Object.prototype.hasOwnProperty.call(models, vizId) ? models[vizId] : null;
}
```

**为什么 model 必须无 DOM**：视觉回归走的就是这条路——
`data/frames/**.nlf → decodeNlf → model（Frame→Scene）→ dist/scenes/**.json → tools/raster.py → PNG`。
模型一旦碰了 `document`/`canvas`，`node dist/viz/tests/scene/dump_scenes.js` 就跑不起来，
G8 就只能退化成「Python 自己拼一个标准场景来比」，用户真正看到的那张图反而没人守
（`docs/ARCHITECTURE.md` §1）。

**布局契约（Scene 的「规则 8」）**：`validateScene`（TS）与 `tools/raster.py::validate_scene`（Python）
会当场拒绝两类布局错误，光栅器那一侧还会让 G8 变红（细节见 `docs/PROTOCOLS.md` §6.1–6.2）。
写模型时照这四条做：

1. **角标一律走 `placeBadge(b, slot, width)`**，不要手写 `x/y`：它按 `BADGE_LAYOUT`
   （`x=8, top=34, rowH=18, colW=344`）放进固定槽位，并把文本裁到 `badgeUnitBudget(width)`
   （两列画布 **47** 个显示单位）。槽位固定是为了"少一枚角标时其余不挪位"，裁剪是因为
   1.2/0.0 章真出现过 65 单位的长角标横穿到右列、压住临界值角标。越界/重叠的报错长这样：
   `徽标 xxx… 超出画布：盒子 [x, y, w×15]，画布 W×H（规则 8：先用 placeBadge/clipBadgeText 收进列宽）`、
   `徽标重叠（规则 8）：…`。
2. **面板 `top` 从 `headerBand(width)` 之后开始**：两列画布预留 **114px**（`34 + 4×18 + 8`），
   单列（宽度 < 712）是 **186px**。
3. **轴下的位置要自己算够**：渲染层会在 `rect` 外侧画装饰带（`AXIS_DECO = {top:18, bottom:28, left:56}`），
   文本层落进去就报 `文本压住坐标轴装饰（规则 8）`。要放图例/数值块时，把该轴 `xlabel` 置空、
   高度多留出 `28px + 自绘文本的高度`（`lu_heatmap` 的副轴按这个口径扣了 62px）。
4. **手写类别名（`AxisOptions` 没有 `xtickLabels`）时**：清空该轴 `xticks` 并**不要**再设 `xlabel`，
   否则数字刻度/轴名会与手写标签叠在同一条带里（`spy_ordering` 踩过）。

**角标是索引级标签**：`placeBadge` 的裁剪是静默的（只留尾部 `…`），所以角标里只放"这是什么"；
长句子请进面板文本、`strings` 或 `pitfalls`。

**检查清单**

- [ ] `<viz>.model.ts` 里没有 `document` / `window` / `canvas` / `node:*` / 第三方 import。
- [ ] 帧里缺必需数组时抛 `ModelInputError`；缺的只是标注信息时降级（不抛错）。
- [ ] `Scene` 至少含一枚 `kind="source"` 徽标；`overflow.flag === true` 时 `note` 非空。
- [ ] 层引用的 `axes` id 都存在；`heat.values.length === nx*ny`；`loglogfit` 的两个斜率都有限。
- [ ] 角标是用 `placeBadge(b, slot, width)` 摆的（没有手写坐标、没有被裁掉关键信息）；
      面板 `top ≥ headerBand(width)`。
- [ ] 有图例/数值块的轴：`xlabel` 置空、高度多让出 `28px + 文本高度`；手写类别名的轴清空了
      `xticks` 且未设 `xlabel`（否则 `validateScene` 报「规则 8」）。
- [ ] 章节前端**不得** import `viz/shared/src/numeric/**`（P8）；确实需要前端近似时，按 §5 走白名单并打 `preview-approx` 徽标。
- [ ] `index.ts` 登记了新 viz；`manifest.gen.ts` 是生成物，**不要手改**。

跑：`make typecheck && make jstest`，然后 `make manifests`。

### 2.4 `tests/<id>/` + `chapters/<id>/asserts.yaml` —— 数值断言

三类断言（`docs/POLICY.md` §3、`docs/CHAPTERS.md` §2）：

| 类 | 内容 | 判定方式 |
|---|---|---|
| **A 解析对拍** | 与解析解/精确解比较，误差随 `h`/`N` 的**阶数必须等于理论阶** | `numcore.verify.assert_order`，理论阶 ± 0.1 |
| **B 不变量** | 守恒量、正交性、对称性、单调性、有界漂移 | 机器精度界或**声明来源**的容差 |
| **C 收敛率/阈值** | 迭代收敛因子、稳定阈值、特征值界 | 脚本判定；阈值 SHOULD 二分实测 |

**模板**（真实文件 `tests/0.0/test_smoke.py` 的三段，各取一条）：

```python
from numcore.verify import assertion, record, assert_order, assert_leq, find_threshold, assert_threshold


@assertion("A", theory="空间二阶精度", chapter="1.2")
def test_ftcs_spatial_order_is_two() -> None:
    """固定 r，加密网格：L2 误差应当以 O(Δx²) 下降（理论阶 2）。"""
    nxs = [11, 21, 41, 81]
    errs = [heat_ftcs(nx=nx, r=0.4, t_end=0.05).l2_error for nx in nxs]
    fit = assert_order([1.0 / nx for nx in nxs], errs, theory=2.0, tol=0.1, label="FTCS 空间收敛阶")
    assert fit.r2 > 0.99


@assertion("B", theory="ζ=0 时比能量严格守恒", chapter="1.2")
def test_energy_conserved_without_damping() -> None:
    tt = np.linspace(0.0, 50.0, 4001)
    x, v = damped_oscillator(tt, omega=1.0, zeta=0.0)
    assert_conserved(oscillator_energy(x, v, omega=1.0), quantity="比能量", label="无阻尼能量守恒")


@assertion("C", theory="有限网格阈值 1/(2sin²(θmax/2))", chapter="1.2")
def test_ftcs_stability_threshold_measured() -> None:
    """对归一化幂迭代实测的增长比做二分，得到阈值后与**有限网格**理论值比对。"""
    result = find_threshold(lambda r: ftcs_growth(r, nx=21) > 1.0 + 1e-6,
                            0.4, 0.6, assume="false_at_lo", tol=1e-7, note="幂迭代测 |λ|max")
    assert_threshold(result, ftcs_critical_r(21), rtol=5e-4, label="FTCS 实测稳定阈值（nx=21）")
```

**`@assertion` 装饰器**（`numcore/verify/report.py:230`）：

```python
assertion(kind: "A"|"B"|"C", theory: Any, *, id: str | None = None,
          chapter: str | None = None, tol: Any = None, strict_measure: bool = True)
```

- `kind`：断言类别（必须与 `asserts.yaml` 里声明的一致）。
- `theory`：理论值/理论阶/理论界，会写进报告与失败信息。
- `id`：缺省用函数名；`asserts.yaml` 用函数名引用它。
- `chapter`：缺省从 `nodeid`（`tests/<id>/`）推断。
- `strict_measure=True`（**不要关掉**）：函数必须在执行过程中产生**至少一条登记**——
  调用 `numcore.verify` 的 `assert_*` helper，或 `return Measured(value=..., tol=...)`，
  或显式 `record(...)`。**静默通过即失败**：

  ```
  测试未登记任何实测值：请调用 numcore.verify 的 assert_* helper，或 return Measured(value=..., tol=...)。
  禁止静默通过（POLICY §3）。
  ```

**实测值必须来自 bisect / 拟合，不能抄理论值**：C 类阈值用 `find_threshold`（二分）实测，
A 类阶数用 `fit_order`/`assert_order`（双对数最小二乘）拟合。
反面教材就是 0.0 章的 FTCS：连续极限是 `r ≤ 1/2`，但有限网格上的真实阈值是
`1/(2sin²(θ_max/2)) = 0.503097…`——把 `1/2` 抄成「实测值」会得到一条永远通过的断言。
0.0 章连「实测阈值与 1/2 的偏差按 O(Δx²) 收敛」这件事也一起测了。

**`chapters/<id>/asserts.yaml` 模板**（真实文件 `chapters/0.0/asserts.yaml`）：

```yaml
# 1.2 章断言清单（真源：`tests/1.2/test_lu.py`）
chapter: "1.2"
title: 线性方程组直接法
required_kinds: [A, B, C]     # 少一类 → G3 直接红

min_counts:                   # 各类断言条数下限
  A: 2
  B: 2
  C: 1

tests:                        # 逐条列出测试函数名 + 类别 + 一句话说明 + 理论依据
  - name: test_lu_backward_error_bound
    kind: B
    what: 后向误差落在 ρ·u·‖A‖ 内
    theory: ‖δA‖ ≤ ρ·u·‖A‖
  - name: test_partial_pivot_growth_is_small
    kind: C
    what: 部分主元下增长因子远小于 2^{n−1}
    theory: ρ ≤ 2^{n−1}
```

双向对账由 `tools/run_asserts.py` 执行（**声明↔实跑两向都要一致**）：

- 「声明了 `X` 但报告里没有这条断言」→ 失败；
- 「`X` 有断言记录但没在 `asserts.yaml` 里声明」→ 失败；
- 「声明为 A 类，实测类别是 B」→ 失败；
- `min_counts` / `required_kinds` 不满足 → 失败；
- 已实现（有 `server/chapters/<id>.py`）却没有 `asserts.yaml` → 失败。

**检查清单**

- [ ] 每条断言都被 `@assertion` 装饰，且有 `theory`（不许 `assert True` 占位）。
- [ ] 每个 `@assertion` 的测试至少产生一条 `record`；A 类不能只调 helper（helper 的类别固定是 `B`）；
      `record(passed=False)` 后面必须跟 `assert`（`docs/PROTOCOLS.md` §8.1–8.3）。
- [ ] A 类声明了收敛阶，实测阶落在理论阶 ± 0.1（`assert_order` 的 `tol` 别偷偷放宽）；
      `h`/`err` 按 h **递减**传，拟合窗口写进 `theory` 串与 `meta`（§8.4）。
- [ ] B 类容差**说得出出处**（机器精度界 `8nu·max|E|`、解析容差、问题规模推导），不是「试出来的」。
- [ ] C 类阈值是二分实测的；若与教科书常数不同，把「向常数收敛」也测掉。
- [ ] `asserts.yaml` 的 `tests[]` 与 `tests/<id>/` 里的函数名逐条对应；`required_kinds` 与实际类别一致。
- [ ] 矩阵分解类章节的 A 类覆盖 `m<n / m=n / m>n`（§8.6）；反例类断言写清样本分布与规模（§8.7）。
- [ ] 拟合窗口的舍入地板处理过（用 `head=` 保留粗端；`tail=` 保留的是**细端**，正是被污染的那一端）。

跑：`make asserts`（会重跑 pytest 并生成 `chapters/<id>/asserts_report.md`）。

### 2.5 `chapters/<id>/`：`notes.tex` + `pitfalls.md` + `asserts.yaml`

**`notes.tex` 骨架**（真实文件 `chapters/0.0/notes.tex`）：

```latex
% 1.2 章 · 线性方程组直接法
% 断言附表由工具生成（\input 进来，不手抄）。

\numlabchapter{线性方程组直接法}{1.2}

\section{问题与记号}
……数学定义、记号、假设……

\begin{theorem}[LU 分解的存在性]
\label{thm:lu}
……
\end{theorem}

\begin{proof}
……完整证明……
\end{proof}

……

\input{chapters/1.2/asserts_table}

\section{它会怎样骗你（本章版）}
\begin{enumerate}
  \item ……
\end{enumerate}
```

- `\numlabchapter{标题}{章号}` 由 `tools/build_book.py:58` 定义：`\newcommand{\numlabchapter}[2]{\chapter{#1}\label{ch:#2}}`。
  它自己生成章标题与 `\label`，**不要在 notes 里再写 `\chapter`**（否则合并本里会出现「空章 + 真章」两页）。
- 至少一个 `\begin{theorem}`（P13 与 G9 都检查）；关键定理要有完整证明。
- 必须有一行 `\input{chapters/<id>/asserts_table}`（G9 检查），它由工具生成，**不许手改**。
- 作者名/标题里的下划线要走转义：正文数学模式下的 `_` 会直接变成 `Missing $ inserted`。

**`asserts_table.tex` 的生成链路**（三步，顺序不能反）：

```bash
make asserts                              # 1) 跑断言 → dist/reports/assertions.json + chapters/<id>/asserts_report.md
python3 tools/build_book.py --no-compile  # 2) 由 assertions.json 生成 chapters/<id>/asserts_table.tex
make book                                 # 3) 编译；附表与最新报告不一致时 G9 直接红
```

**`pitfalls.md`（≥ 800 字）**：不要写泛泛的「注意数值稳定性」。写**这个仓库里真实踩过或差点踩到**的坑，
每条都要能落到某个字段或某行代码上。0.0 章的结构可以照抄：标题 → 每节一个坑（现象 + 为什么会骗人）
→ **本项目的做法**（对应到 `axes[].autoscaled`、`overflow{flag,note}`、`meta.t_actual` 这类具体字段）。

字数按 `tools/check_policy.py` 与 `server/registry.py` 的同一口径统计：
`pitfalls.md` 全文 `strip()` 后的字符数 ≥ 800（`MIN_PITFALL_DOC_CHARS = 800`）。

**检查清单**

- [ ] `notes.tex` 有 `\numlabchapter`、≥ 1 个 `theorem` + `proof`、`\input{chapters/<id>/asserts_table}`。
- [ ] 每个结论都能被某条断言验证（纸面结论与机器结论同源）。
- [ ] `pitfalls.md` ≥ 800 字，条目落到具体字段/函数。
- [ ] `asserts.yaml` 与 `tests/<id>/` 双向一致。

跑：`make book`（G9）。

### 2.6 源码书写约定（中文、引号、枚举与哨兵）

- **中文字符串里不要用裸 ASCII 双引号（`"`）**：它会把 Python 字符串提前截断，实测多次当场
  `SyntaxError`（1.3 / 3.2 / 2.3 章都中过）；LaTeX 的字符串里同理（`"` 不是引号）。
  统一用「」或 “”。G1 的 SYNTAX 规则把这类问题报成 `语法错误：…`；同一规则还管另一类相关报错：
  `警告当作错误：invalid escape sequence '\i'（无效转义请写成原始字符串 r""" 或双反斜杠）`——
  docstring 里写 LaTeX（`\input`、`\int`）必须用 `r"""` 或双反斜杠。
- **枚举/字符串参数走 `set_strings`**（`pivot="partial"`、`bc="natural"`、`method="bdf2"`）；
  帧里出现哨兵值（`None`、`-1`）必须在 `strings` 里说明它为什么不是数字
  （`docs/PROTOCOLS.md` §1.3，那里列了 1.1/1.2/2.2 三处正例与 1.5 的待补处）。
- **PNG 基线里的中文要有字形**：`tools/raster.py` 的位图字体是**烘焙**进去的，子集按
  "场景里真正出现的字符"收集（`tools/bake_font.py`）。新写的中文如果不在里面，PNG 里就是方框
  （浏览器 canvas 正常），`pytest tests/test_raster_font.py` 与 `python3 tools/bake_font.py --check`
  会报出缺字清单；修法是 `--write` 重烘焙 + 重画基线。**别改回旧的 `glyph_offsets(...) is None` 探针**
  ——它永远为假（缺字时返回的是缺字框轮廓），这正是缺字溜进基线的原因。
- **`@assertion(theory=...)` / `record(label=...)` / `asserts.yaml` 的 `what:` 会原样进 LaTeX 附表**：
  它们不是普通注释，而是印在 PDF 正文里的字。两条纪律：
  ① **不许写组合变音符号**（`x̂` = `x` + U+0302、`H̄` = `H` + U+0304）：xelatex 会**静默吞掉**
  组合符号，纸上就只剩一个孤零零的 `x`；写成 `x_hat` / `H_bar`。
  ② 白名单之外的符号同样缺字形（`⇔ ≳ ⊥ ‖ ⱼ ᵀ ⁻¹ ① ₀` 都缺；`≈ ≠ ≤ ≥ ± × ÷ ∞ → ← ∂ ∇ ∈ ⊂ ∪ ∩ ⋅ ° √ ∑ ∏ ∫` 不缺）。
  判定代码是 `tools/build_book.py::latex_risky_chars()`，G9 在写附表前与 `--check` 时都会报
  （1.4 章两次踩到：`⇔ ≳` 一次、`H̄` 一次；1.3 / 3.1 章各一次 `x̂`）。
- **没有 `xtickLabels` 这个东西**：`AxisOptions` 没有这个字段，类别名只能自己用 `text` 层手写；
  并且要**把该轴的 `xticks` 清空**（否则数字刻度与手写标签叠在同一行）、
  **不要再给该轴设 `xlabel`**（否则与手写标签挤在同一条装饰带，规则 8 当场报违规）。
- **几个反复踩到的具体坑**（各一句）：
  - `np.roll` 平移采样网格是**空操作**：它只把数组循环置换，`min`/`max`/`sum` 这类置换不变量
    逐位不变（判据：对同一数组 `np.roll(x, k)` 与 `x` 的 `min/max/sum` 完全相同，只有 `argmin` 会动），
    1.1 章差点据此写出假证据；真要平移网格请写 `x + k*dx`。
  - `next_float(±0, direction=−1)` 要特判：位模式键把 `+0`/`−0` 当相邻，但数轴上它们是同一个点，
    实现里必须先跳到另一个零的键上（`numcore/fp.py` 的 `next_float`）。
  - **C¹ 函数的数值导数必须用单侧差商**：跨节点的中心差商带 `O(h·ΔS'')` 项，
    2.2 章 PCHIP 实测卡在 `1e-5` 量级、缩小 h 也不改善（实测最大相对差 `2.6e-6`）；
    换成同一段内的三点单侧二阶差商立刻降到 `1.4e-10`。
  - **Newton 收敛判据要用相对残差**：绝对判据在解衰减到 `1e-11` 时"零次迭代就宣布收敛"
    （那一刻残差已是当前解的 50%）；`numcore.ode.implicit.newton_solve` 用
    `‖F‖∞ ≤ tol·max(‖y‖∞, ‖y₀‖∞) + atol`（3.2 章真踩过）。

---

## 3. 本地验证循环

按「改动面」选命令，**提交前一律 `make ci`**（十道门禁，约 2 分钟，失败不早停）。

| 命令 | 什么时候需要跑 | 失败了先看哪里 |
|---|---|---|
| `make typecheck` | 改了 `viz/**/*.ts` | `tsc` 输出（`strict: true`）；常见是 `Scene` 字段漏写或类型不匹配 |
| `make jstest` | 改了前端（会先 `tsc` 编译到 `dist/viz`） | `node:test` 的用例名；`viz/tests/*.test.ts` 里同名断言 |
| `make pytest` | 改了 `numcore/` `server/` `tests/` | 失败用例的 `theory/measured/差值`（断言失败信息会同时打印三者） |
| `make asserts` | 改了 `tests/<id>/` 或 `asserts.yaml` | 先看输出的「N 个对账问题」列表；再看 `dist/reports/assertions.json` |
| `make manifests` | 改了 `MANIFEST` 或 `docs/chapters.yaml` | `tools/gen_manifests.py` 报的问题（字段不合规会指出具体 viz/param） |
| `make parity` | 改了 `viz/shared/src/numeric/` 或对拍口径 | 报告会打印 `case / 索引 / python 值 / ts 值 / 相对差 / 容差` 与超差 Top 5 |
| `make visual` | 改了 `model.ts`、渲染层、manifest 参数 | `dist/visual/diff/*.png` 差异图；`dist/reports/visual_regression.json` |
| `make visual-update` | 图**有意**变了（必须在提交信息里写明原因） | 更新后 `git status tests/baselines/` 应只多出你预期的那几张 |
| `make book` | 改了 `notes.tex` / `asserts_table.tex` | `dist/book/one_<id>.log`（缺字、未定义引用、Overfull 都在这） |
| `make progress` | 开/关一章、改了 `chapters.yaml`、完成度变了 | 重生成 `docs/PROGRESS.md`；门禁状态段读 `dist/reports/*.json`，缺哪个就显示「未运行」（不是失败） |
| `make ci` | 提交前 | `dist/logs/G1.log … G10.log`（每道门的完整输出）+ `dist/reports/ci.json`（汇总） |

两个只读的「体检」命令，排查时很好用：

```bash
python3 tools/gen_manifests.py --check    # 生成物有没有漂移（不写盘）
python3 tools/parity.py --check-golden    # 盘上的 golden 是否仍等于 Python 现在算出来的
```

---

## 4. 加一个新可视化的完整步骤

假设给 1.2 章加第 5 个 viz（`lu_sparsity`），从改数到基线：

1. **改 `docs/chapters.yaml`**：把 1.2 的 `viz: 4` 改成 `viz: 5`。
2. **写 `MANIFEST`**：在 `server/chapters/1.2.py` 的 `vizzes` 里加一条，`id: "lu_sparsity"`，
   补齐 `params` / `time` / `requiredAux` / `axes` / `pitfalls` / `theory`。
3. **写前端三件**：`viz/chapters/1.2/lu_sparsity.model.ts`（纯 Frame→Scene）、
   `lu_sparsity.viz.ts`（`mountViz` 装配）、在 `viz/chapters/1.2/index.ts` 的 `models` 里登记。
4. **重新生成 manifest**：`make manifests`。
   它会重写 `viz/chapters/1.2/manifest.gen.ts`、`data/manifests/1.2.json`、
   `data/manifests/index.json`、`viz/shared/src/chapters.gen.ts`，以及
   **`viz/chapters/index.gen.ts`**（`章节 id → { manifest, models }` 的静态登记表——
   站点路由、`dump_scenes.ts` 都从这里取模型）。生成物一律不手改。
5. **类型与单测**：`make typecheck && make jstest`。
6. **生成 L3 帧**：`python3 tools/precompute.py --chapter 1.2`（或 `make precompute CHAPTER=1.2`）。
   帧落在 `data/frames/1.2/lu_sparsity/<params_hash>_t<%.6g>.nlf`（不进 git）。
7. **转储场景**：`npm run scene`（即 `node dist/viz/tests/scene/dump_scenes.js`）。
   它会遍历 `data/frames/`，用**真模型**把每帧算成 Scene，写到
   `dist/scenes/<chapter>/<viz>__<参数指纹 8 位>__t<时刻>.json`。
   **新可视化会让场景数变多**（当前模板章是 12 个）——这一步没有新帧就会「跳过并计入 skipped」，
   不会静默当成通过。
8. **补基线**：`make visual` 此时会报「缺基线」；确认图是对的以后跑 `make visual-update`
   生成 `tests/baselines/1.2/lu_sparsity__<指纹>__t<时刻>.png`（**进 git**），
   并在提交信息里写明为什么更新基线。基线命名 = 场景转储文件名换扩展名，
   所以改了采样时刻/参数 → 文件名全变 → 全部缺基线，这是**有意**的。
9. **收尾**：`make progress`（进度表里 1.2 的可视化数变成 5/5）、`make ci`。

---

## 5. 双实现对拍（Python ↔ TypeScript）

**什么时候必须写 TS 侧 `viz/shared/src/numeric/<algo>.ts`**：只有两种正当理由——

1. 前端要做**与方法无关的显示辅助**（拟合斜率、采样、坐标变换），并且要保证与后端同口径；
2. 前端要在不连后端时给**明确的近似预览**（`dataSource: "backend+preview"`），
   且必须在 UI 上打 `preview-approx` 徽标。

白名单由 P8 检查：`viz/chapters/**` 一旦 import `shared/numeric/**`，
文件里必须有 `// preview-approx` 标注，否则 G1 报：

```
章节前端调用了 shared/numeric：必须在文件中标注 // preview-approx（P8）
```

**加一个 case 的完整步骤**

1. `tools/parity.py` 的 `CASE_SPECS` 加一条（`name` / `algo` / `tolerance` / `outputs` / `params` / `note`）：
   这是 case 定义的**唯一真源**，`tests/parity/cases.json` 与 `tests/parity/golden/**` 都是它的生成物。
2. `viz/shared/src/numeric/<algo>.ts` 实现 TS 侧（纯函数，无 IO/DOM/第三方），
   并在 `viz/tests/parity/run_parity.ts` 的 `ALGOS` 映射里登记 `algo` → 计算函数
   （runner 按模块逐个 import，为的是「`case.algo` → 实现模块」在源码里可 grep）。
3. `tests/parity/tolerances.yaml` 给这个 case 写 `rtol` / `atol` / **`reason`**（必填）：
   说清误差来源（求和顺序 / 三角查表 / 舍入地板 / 回读精度……）并附实测值。
   没有 `reason`，`tools/parity.py` 直接判配置错误。
4. 生成基准并校验：

```bash
python3 tools/parity.py --update-golden   # 重算并写盘 golden 与 cases.json（含每份 golden 的 sha256）
python3 tools/parity.py                   # G7：Python ↔ TS 逐点比对
python3 tools/parity.py --check-golden    # 额外确认「盘上的 golden == 现在 Python 算出来的」
```

**容差怎么定**（口径写在 `tests/parity/tolerances.yaml` 顶部）：

- 两侧同为 float64、同一套公式、同一求值顺序时，**实测逐位相同也不取 0 容差**。理由有三条，
  都要写进 `reason`：
  1. **跨平台余量**：numpy 在 x86-64 上的 SIMD 路径可能做 FMA 收缩，单步就能差 1 ulp（≈2.2e-16）；
  2. **超越函数**：`sin/cos/exp/log` 在 numpy（glibc）与 Node（V8 的 fdlibm 移植）可差 1–2 ulp；
  3. **求和顺序**：numpy 的 `np.sum` 成对求和，TS 侧朴素左折叠，`n` 个点差 ~`n·eps`。
- 容差要「容得下 1 ulp 级实现差异，又能在算法被改错时立刻报警（那种错至少 1e-3 量级）」。
  例如实测逐位相同的单步 case 取 `rtol=1e-14`（≈45 ulp）。
- 靠近零点的分量相对差会自然放大，靠 `atol` 兜底；**绝不靠把 `rtol` 放大到能过**——
  「那是把门禁做成了装饰」。

**当前覆盖**（实测 2026-10-18）：**15 个 case / 4088 个点**——`heat_explicit_step`、`heat_ftcs_run`、
`ftcs_amplification_modes`、`oscillator_sample`、`order_fit_powerlaw`，
1.2 章的三条 LU 对拍：`lu_partial_pivot`（Hilbert(6)，92 点）、`lu_wilkinson_growth`（Wilkinson(12)，312 点）、
`lu_tiny_pivot`（首元 1e-14，68 点），三条均实测 `max|Δ| = 0`；
1.4 章的幂法五条（`power_iter_symmetric` 97 / `power_iter_tridiag` 125 / `power_iter_shear` 610 /
`power_iter_angle_plateau` 202 / `power_iter_ratio_floor` 55 点——同一个 `PowerResult` 按**条件数**拆开比，
夹角与深收敛比值的容差放宽写在各自 reason 里，那是量本身的条件数，不是实现差异）；
3.2 章的稳定域两条（`stability_function_grid` 576 点、`lmm_root_modulus_bdf` 36 点）。
读数命令：`python3 tools/parity.py` → `dist/reports/parity.json` 的 `n_cases` / `n_points`；
计划清单见 `docs/PROTOCOLS.md` §7。

---

## 6. 常见拒绝原因（照着 POLICY 与 `check_policy.py` 的 RULE_DOC 写）

| 违规写法 | 报错信息（原文摘录） | 正确写法 |
|---|---|---|
| `numcore/` 或 `tests/` 里 `import scipy`（P1） | `禁止 import scipy（scipy 只允许出现在 numcore/reference/）` | 自研实现；要对拍就把参照物写进 `numcore/reference/`，测试从那里调用 |
| `np.linalg.solve(...)` / `np.fft.fft(...)`（P2） | `禁止 numpy.linalg.*（numpy.linalg / numpy.fft 只允许出现在 numcore/reference/）` | 用 `@` 等原语自己写（`PA = LU`、radix-2 FFT…） |
| `np.polyfit/polyval/interp/trapezoid/roots/lstsq/eig/svd/…`（P3） | `禁止 numpy.polyfit：这是现成求解器，必须自研（P3）` | 同一算法自己实现；测试/工具里做交叉验证可以用 |
| `np.random.default_rng(...)`（P4） | `禁止 numpy.random：请用 numcore.utils.rng（numcore 不得直接使用 numpy.random）` | `from numcore.utils import rng; rng(seed)` |
| `numcore/` 里 `import matplotlib/pandas/sympy/mpmath/sklearn/numba`（P5） | `禁止 import matplotlib（numcore 不得 import …）` | 这些只允许出现在 `numcore/reference/` 或 `tools/` |
| `numcore/` 里 `print(...)` / `open(...)` / `import os` / `global x`（P6） | `numcore 内禁止 print()（纯函数约束，P6）` | 计算函数只收数组、只返回结果；IO 留给 `tools/` 与 `server/` |
| `viz/` 里 import 第三方包（P7） | `viz 运行期禁止第三方 import：'lodash'（viz 运行期零第三方依赖…）` | 只用标准库/浏览器 API；`node_modules` 只留 `typescript` + `@types/node` |
| 章节前端 import `shared/numeric/**` 却没标注（P8） | `章节前端调用了 shared/numeric：必须在文件中标注 // preview-approx（P8）` | 要么改成消费后端帧，要么在文件里写 `// preview-approx` 并在 UI 打「前端近似预览」徽标 |
| `numcore/`、`server/`、`viz/` 里 `import numcore.reference`（P10） | `禁止 import numcore.reference：参照物只能出现在测试与工具里` | 参照物只在 `tests/` 与 `tools/` 中使用 |
| 手改生成物、或生成物缺头部（P12） | `生成物文件缺少「本文件由…生成 / 请勿手改」头部声明（P12）` | 生成物一律由工具产出（`manifest.gen.ts`、`chapters.gen.ts`、`index.gen.ts`、`*.gen.ts`…） |
| 开了章却缺 `notes.tex` / `pitfalls.md` / `asserts.yaml`（P13） | `章节 1.2 缺推导笔记 notes.tex（POLICY §2 第 1 项）` / `章节 1.2 的 pitfalls.md 只有 512 字，少于 800 字` | 五件套补齐；`notes.tex` 至少一个 `\begin{theorem}` |
| 行内豁免没登记（ALLOWLIST） | `未登记的豁免 numcore/linalg/lu.py:P4（请在 tests/policy/allowlist.txt 登记，理由：…）` | 源码写 `# policy: allow P4 <理由>`，并在 `tests/policy/allowlist.txt` 加一行 `路径:规则` |
| docstring 里写 LaTeX 导致无效转义 | `警告当作错误：invalid escape sequence '\i'（无效转义请写成原始字符串 r""" 或双反斜杠）` | 用 `r"""…"""` 或 `\\input` |
| manifest 少字段 | `lu_heatmap.rho: 缺少 critical（没有理论临界值也要显式写 []）——规则 6` / `requiredAux 必须含 'error' 或 'residual'` / `必须显式声明 autoscaled（true/false）——规则 1/4` | 按 §2.2 的字段表补齐 |
| `pitfalls` 太短 | `lu_heatmap: pitfalls 至少 60 字（现在 12 字）——规则 7 要求说清'它会怎样骗你'` | 写具体：哪个参数、哪张图、会读出什么错误结论 |
| 断言静默通过 | `测试未登记任何实测值：请调用 numcore.verify 的 assert_* helper，或 return Measured(...)。禁止静默通过（POLICY §3）。` | 用 `assert_order/assert_leq/assert_conserved/find_threshold/record` 登记实测值 |
| 声明与实跑不一致 | `1.2: 声明了 test_lu_bound（B 类）但报告里没有这条断言` / `1.2: test_lu_bound 有断言记录但没在 asserts.yaml 里声明` | 改 `asserts.yaml` 或改测试，让两边逐条对齐 |
| 断言类别缺失 | `1.2: 缺少 C 类断言（asserts.yaml 要求必须有）` | 按 `chapters.yaml` 的 `asserts:` 补齐类别 |

---

## 7. 提交纪律

- **提交信息格式**（`docs/POLICY.md` §6）：`<area>: <what>`，例如
  `core(linalg): 加入带部分主元的 LU 与增长因子`、`viz(1.2): 稀疏 spy plot 与排序切换`。
  视觉基线更新时，在正文里**写明原因**（`tools/ci.sh` 与 `Makefile` 都强调这一点）。
- **显式 stage**：一次提交只包含这一章的改动，逐路径 `git add <具体文件>`。
- **不许 `git add -A` / `git add .`**：仓库里有大量生成物与基线，
  一把梭会把别人的改动、临时产物、未完成的实验一起提交。
- 生成物该不该进 git，看 `docs/PROTOCOLS.md` §9：`manifest.gen.ts`、`data/manifests/**`、
  `tests/baselines/**`、`tests/parity/golden/**` 要进；`dist/**`、`data/frames/**`、`.cache/**` 不进。
- 提交前跑 `make ci`；只要有一道门红就不要提交（G6/G8/G10 的漂移是**内容问题**，
  不是「重新生成一下就好」——先确认新状态是你想要的，再更新生成物/基线）。

---

## 8. 一页速查

```bash
make pytest && make typecheck && make jstest   # 核心实现 + 断言 + 前端
make asserts && make manifests && make progress
make book && make parity && make visual
make ci                                        # 提交前最后一道

# 排错三件套
python3 tools/gen_manifests.py --check         # 生成物漂移？
python3 tools/parity.py --check-golden         # golden 过期？
python3 tools/run_asserts.py --no-run --json   # 断言对账问题清单
```
