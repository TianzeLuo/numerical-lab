# POLICY —— 仓库红线与 CI 门禁

> 本文件是**规范性**文档（RFC 2119 语义：MUST / MUST NOT / SHOULD）。
> 任何章节、任何子 agent、任何未来贡献者都不得例外。`tools/check_policy.py` 是它的**可执行化身**：
> 文档与检查器不一致时，以检查器为准，并必须同步修改本文件。

---

## 1. 依赖红线（最重要的一条）

`docs/CHAPTERS.md` §0 写死了本项目的存在理由：

> **不用现成求解器替代自研实现**。SciPy / NumPy / PyTorch 只作为对拍参照物存在。

为了让这条口号变成**机器可验证的约束**，规定如下：

| 规则 | 内容 | 强制方式 |
|---|---|---|
| **P1** | `import scipy` / `from scipy ...` **只允许**出现在 `numcore/reference/` 下的文件 | AST 静态检查 |
| **P2** | `numcore/`（`reference/` 除外）MUST NOT 使用 `numpy.linalg.*`、`numpy.fft.*` | AST 静态检查（属性链） |
| **P3** | `numcore/`（`reference/` 除外）MUST NOT 使用 `numpy.polyfit / polyval / interp / trapezoid / trapz / gradient / convolve / correlate / corrcoef / cov / histogram / histogram2d / bincount / diff / ediff1d / roots / lstsq / solve / inv / det / eig / eigvals / svd / qr / cholesky / norm / cond / matrix_rank / pinv / expm / kron / tensordot` 等"成品算法" | AST 静态检查（属性白名单见 `tools/check_policy.py`） |
| **P4** | `numcore/` MUST NOT 直接使用 `numpy.random`；随机数一律经 `numcore.utils.rng(seed)` 取用（集中确定性 + 可复现） | AST 静态检查 |
| **P5** | `numcore/` MUST NOT import `matplotlib / pandas / sympy / mpmath / sklearn / numba`（`reference/` 可有 sympy、mpmath，用于精确解与高精度对拍） | AST 静态检查 |
| **P6** | `numcore/` 的公开函数 MUST 是纯函数：无 IO、无全局可变状态、无 `print`、无 `open/os/sys/subprocess/socket` | AST 静态检查（IO 名称黑名单） |
| **P7** | `viz/` MUST NOT 引入任何运行期第三方依赖（`node_modules` 只允许 `typescript`、`@types/node` 两个 devDependency） | 检查 `package.json` + import 扫描 |
| **P8** | `viz/chapters/**` MUST NOT 自行实现物理/数值算法；只能消费后端帧，或调用 `viz/shared/src/numeric/` 中**已通过对拍**的函数（且 UI 必须打上"前端近似预览"徽标） | 白名单检查 + 对拍测试 |
| **P9** | 章节前端 MUST NOT 硬编码数值结果充当"数据"；场景必须由帧驱动 | 评审 + 视觉回归快照（快照由帧生成） |

**允许用 numpy 做什么**：`ndarray` 作为容器、逐元素算术、切片/广播、`np.dot`/`@`（当作 BLAS 原语）、`np.concatenate/stack/zeros/empty/arange/linspace/meshgrid/where/clip/abs/sqrt/exp/log/sin/cos/sum/mean/max/min/argmax/all/any/isfinite/isnan/cumsum/prod/outer/einsum`（缩并原语）、`np.linalg` 之外的 `np.errstate`。这些是"语言的一部分"，不是"求解器"。

> 判定口径：**如果一个调用直接给出了本章要教的那个算法结果，它就是被禁的。**
> 例如 2.5 章要教 FFT，则 `np.fft.fft` 被禁；1.2 章要教 LU，则 `np.linalg.solve/lu` 被禁。
> 反过来，用 `@` 做矩阵乘法实现自己的 LU 是允许且鼓励的。

### 例外登记

确需例外时，必须在源码处写 `# policy: allow <规则号> <理由>`，并在 `tests/policy/test_allowlist.py` 中登记该文件；
检查器只承认**已登记**的豁免。禁止行内豁免不登记。

---

## 2. 五件套完整性（每章）

一章 = 五个交付物，缺一不可，路径固定：

| # | 交付物 | 路径 | 缺失判定 |
|---|---|---|---|
| 1 | 推导笔记 | `chapters/<id>/notes.tex` | 文件不存在，或 `\begin{theorem}` 数为 0，或编译失败 |
| 2 | 核心实现 | `numcore/<area>/...`（见 `docs/chapters.yaml` 的 `modules`） | 模块不存在或 import 失败 |
| 3 | 交互可视化 | `viz/chapters/<id>/*.viz.ts` + `manifest.gen.ts` | 可视化数 ≠ `chapters.yaml` 的 `viz:` 声明 |
| 4 | 数值断言 | `tests/<id>/` + `chapters/<id>/asserts.yaml` | 声明与实跑**双向**不匹配 |
| 5 | 防误导分析 | `chapters/<id>/pitfalls.md` | 文件不存在或少于 800 字 |

`tools/gen_progress.py` 依此生成 `docs/PROGRESS.md` 与站点进度条。

---

## 3. 断言纪律

- A 类（解析对拍）：**必须**给出收敛阶，且实测阶 ∈ 理论阶 ± 0.1（`numcore.verify.assert_order`）。
- B 类（不变量）：**必须**声明容差来源（机器精度 / 解析容差 / 问题规模推导），不得"试出来的容差"。
- C 类（收敛率/阈值）：**必须**由脚本判定；阈值类断言 SHOULD 用 `numcore.verify.find_threshold` 二分实测，而不是把理论值抄回来当实测值。
- 每条断言 MUST 通过 `@assertion(...)` 装饰器登记，声明 `theory`（理论值/理论阶）与 `measured`（实测），
  失败信息 MUST 同时打印两者与差值。禁止 `assert True` 式占位。
- 每章 `asserts.yaml` 中声明的断言 id 与 pytest 收集到的 id MUST 双向一致；
  跳过（skip/xfail）不算通过，CI 视为失败（`--strict` 模式）。

---

## 4. 防误导七条规则（可视化硬约束）

来自 `docs/CHAPTERS.md` §2，落地为**可检查的字段**：

| # | 规则 | 落地字段（`Scene` / manifest） | 检查点 |
|---|---|---|---|
| 1 | 坐标轴不得自动裁剪以隐藏发散，越界显式标记 | `axes.autoscaled: boolean`、`overflow: {flag, note}` | `validateScene()` 必填 |
| 2 | 双对数图必须同时画实测拟合斜率与理论斜率 | `layer.kind="loglogfit"` 含 `measuredSlope` 与 `theorySlope` | 二者都渲染成参考线 |
| 3 | "看起来很准"的图旁必须给出误差场/残差场 | manifest `requiredAux` 必含 `error` 或 `residual`；`axes` 中必须有对应副轴 | 契约测试 |
| 4 | 自适应轴与固定轴必须明确标注 | `axes.autoscaled` + UI 角标 | 契约测试 |
| 5 | 数据一律来自 `numcore`，前端只渲染/插值 | `dataSource` 字段 + P8/P9 | 策略检查 + 对拍 |
| 6 | 每个可拖参数标注理论临界值 | `ParamSpec.critical[]` + `ParamSpec.criticalNote` | manifest 校验（无临界值须显式写 `critical: []`） |
| 7 | 每个可视化附"它会怎样骗你" | `manifest.pitfalls`（非空）+ `chapters/<id>/pitfalls.md` | 契约测试 |

---

## 5. CI 门禁（`make ci` 全绿才算完成）

| 门 | 命令 | 通过标准 |
|---|---|---|
| G1 | `python tools/check_policy.py` | 0 违规 |
| G2 | `python -m pytest tests -q` | 全绿，且无 skip（`--strict`） |
| G3 | `python tools/run_asserts.py --all` | 声明↔实跑双向一致，全绿 |
| G4 | `npm run typecheck` | `tsc --noEmit` 0 错误（`strict: true`） |
| G5 | `npm test` | TS 单测（含 `.nlf` 解码、scene 校验、数值 helpers）全绿 |
| G6 | `python tools/gen_manifests.py --check` | 生成物与仓库一致（无漂移） |
| G7 | `python tools/parity.py` | Python↔TS 基准数据逐点一致（容差取自 `tests/parity/tolerances.yaml`） |
| G8 | `python tools/visual_regression.py --check` | 场景/像素快照与基线一致（或已显式 `--update` 并附原因） |
| G9 | `python tools/build_book.py --check` | 每章 `notes.tex` 编译通过，无未定义引用 |
| G10 | `python tools/gen_progress.py --check` | `docs/PROGRESS.md` 与仓库状态一致 |

> **G6/G8/G10 的"漂移"类门禁是本项目的良心**：它们专治"文档说做了、其实没做"与"图好看但算错"。

---

## 6. 命名与风格

- Python：`snake_case`；公开 API 必须有 docstring，**docstring 第一行给数学定义**（不是功能描述）。
  例：`def lu(A): """PA = LU，其中 L 单位下三角，U 上三角。"""`。
- 数值核心函数签名统一为 `(输入数组, 关键字参数) -> 结果`，不在函数内做输入校验以外的副作用。
- TypeScript：`camelCase`；每个 viz 拆成 `model.ts`（纯，无 DOM）/ `view.ts`（DOM）/ `viz.ts`（装配 + manifest 绑定）。
- 中文注释解释**为什么**，英文标识符承载**是什么**。
- 提交信息：`<area>: <what>`（例：`core(linalg): 加入带部分主元的 LU 与增长因子`）。
