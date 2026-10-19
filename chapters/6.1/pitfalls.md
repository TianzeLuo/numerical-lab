# 6.1 章防误导清单（二维综合仿真）

本章是前五部分的“总装车间”：方腔、Ghia 基准、Gray--Scott 与干底溃坝各自都是
一个完整的求解器，图也都能画得很漂亮。正因为如此，最容易犯的错不是写不出代码，
而是**把一张好看的图当成证据**。下面 14 条里，第 1、4、6、9 条是“我几乎把它写成
假证据”的现场记录；所有数字都来自 `tests/6.1/` 的断言记录。

---

## 1. 残差小 ≠ 解对了：两者差四个数量级

**现象**：看到残差压到 $10^{-7}$ 就宣布“方腔算准了”。
**实测**（$Re=1000,n=41$）：相对残差 $1.00\times10^{-7}$，而同一帧解对 Ghia
中心线的 RMSE 是 $0.2271$（$u$ 剖面 $0.1833$、$v$ 剖面 $0.2637$）。反过来，
$Re=100,n=49$ 残差 $9.91\times10^{-8}$，RMSE 只有 $0.00603$。同样小的残差，
误差相差 37 倍。
**正确判据**：残差量的是“离散方程解得有多狠”，误差量的是“格式与网格离真解多远”。
页面必须把 `residual` 与 `rmse` 分成两栏，并给出各自的收敛研究。
（断言：`test_small_residual_does_not_imply_benchmark_accuracy`、`test_cavity_residual_and_discretization_error_are_separate`）

## 2. 涡量壁面符号写反，会得到“镜像但漂亮”的解

**现象**：用内部中心差分外推壁面涡量，或把顶盖的 Thom 项符号写反。
**实测**：$n=129$、$Re=100$ 时主涡心跑到 $x=0.383$，而 Ghia 是 $0.6172$；
解依然光滑、残差依然能压到 $10^{-7}$，因为方程本身是自洽的，只是边界条件错了。
**正确判据**：Thom 公式 $\omega_w=-2\psi_{\text{邻}}/h^2-2U_\tau/h$ 里顶壁
$U_\tau=+1$ 给出负涡量；用主涡心的 $x$ 坐标做独立校验，不要只看残差。
（断言：`test_dst_poisson_residual_and_helmholtz_manufactured_solution`、
`test_ghia_vortex_centers_and_transcription_floor`）

## 3. DST 解的是离散方程，不是连续方程

**现象**：制造解误差 $10^{-14}$，于是说“Poisson 求解器没有误差”。
**实测**：五点 Laplace 的制造解残差 $4.885\times10^{-14}$、Helmholtz 制造解误差
$1.110\times10^{-15}$、离散散度 $8.882\times10^{-16}$——这些都是**代数恒等式**
级别的验证，证明变换实现没有和算子互相抵消，但不证明网格离散误差小。
**正确判据**：A 类断言只能证明“代码解的是它声称的那个方程”；精度阶必须另做网格实验。
（断言：`test_velocity_is_divergence_free_and_interp_is_conservative`）

## 4. 「我几乎写成假证据」之一：四网格斜率不是精度阶

**现象**：把 Ghia 误差的四个点拟合出 $1.10$，写成“本章格式是一阶”。
**实测**（$Re=100$）：$n=17/25/33/49$ 的 RMSE 为
$0.02024/0.01211/0.00870/0.00603$；前三段斜率 $2.10$、$1.73$、$1.10$。
$v$ 剖面的误差已经接近 Ghia 表本身的抄写分歧，最后一段被地板压低。
**正确判据**：斜率必须附窗口；本章一阶迎风的理论阶是 1，但 $v$ 的局部 2.10
不能读成二阶精度。到 $n=49$ 时 RMSE 仍高于 $10^{-2}$ 的转载地板。
（断言：`test_ghia_grid_convergence_re100`、`test_ghia_tables_complete_and_endpoints_exact`）

## 5. Ghia 表是“转载数据”，不是无限精度真值

**现象**：用 $0.0001$ 的容差去对拍 Ghia 数值，或直接把表值当解析解。
**实测**：Ghia--Ghia--Shin (1982) 的表在文献转载中有 $10^{-2}$ 量级的末位分歧；
本章把 $0.01$ 作为**抄写地板**，$n=49$ 的 RMSE $0.00603$ 已经低于它。
$u(y)$ 表的端点必须精确为 $0$（壁面无滑移），$v(x)$ 在左右壁也必须为 $0$。
**正确判据**：对拍要给地板；插值方向要明确（把数值解插到基准节点，而不是反过来）。
（断言：`test_ghia_tables_complete_and_endpoints_exact`）

## 6. 「我几乎写成假证据」之二：经典斑图不是图灵失稳

**现象**：看到 spots 长出斑点，就说“图灵判据验证成功”。
**实测**：spots 预设的均匀底态**不存在**（$v_0$ 无正实根），最快模的
$\lambda^\ast=-0.035<0$，但数值上仍长出振幅 $0.729$ 的斑图——那是双稳态/
初值驱动的图案，不是图灵失稳。stripes 预设的底态存在，可 $\lambda(0)=+0.0053949$，
是**均匀模**先失稳，也不是图灵。只有真正满足“底态稳定 + 非零波数增长”的 turing
预设才配叫图灵：$\varphi^\ast=53.8988$、$\lambda^\ast=0.015017$。
**正确判据**：帧里用四个互斥原因标记预设（turing / stripes / spots / uniform），
不能用一个 `pattern=True` 混过去。
（断言：`test_turing_base_state_selects_stable_branch`、`test_classic_pattern_grows_without_turing_instability`、`test_turing_band_flags_distinguish_four_reasons`）

## 7. det 顶点不是最快模

**现象**：令 $\partial_\varphi\det J=0$ 求“最不稳定波数”。
**实测**：det 的顶点在 $\varphi=62.7495$，给出 $\lambda=0.013392$；真正的最快模在
$\varphi^\ast=53.8988$，$\lambda^\ast=0.015017$，比顶点高 $10.8\%$。
**正确判据**：最快模要解 $\partial_\varphi\lambda=0$（本章用扫描 + 局部细化），
不能拿行列式的驻点代替增长率的驻点。
（断言：`test_most_unstable_mode_beats_det_vertex`）

## 8. dt 上限是夹取还是发散，必须显式写

**现象**：Gray--Scott 默认 $\Delta t=0.25$ 超过稳定性上限，代码静默截断后
页面仍显示用户填的 0.25。
**实测**：turing 预设的稳定上限为 $0.0953674$，实际用 $\Delta t=0.0953674/0.25$
的比例夹取；若按 0.25 硬推，几步内出现负浓度或非有限值。
**正确判据**：帧的 metrics 要同时给“请求 dt / 实收 dt / 为何夹取”，
状态栏显示夹取后的值；不许静默改参数。
（断言：`test_dt_limit_clamps_instead_of_silently_diverging`）

## 9. 「我几乎写成假证据」之三：质量漂移不是守恒误差

**现象**：看到干底溃坝的质量漂移 $9.9\times10^{-9}$，说“Rusanov 格式不守恒”。
**实测**：周期边界下质量漂移是机器精度；干底溃坝的 $9.9\times10^{-9}$ 全部来自
正性地板（$h\leftarrow\max(h,0)$）在干锋附近注入的微量水，方向单一、随网格加密
减小。通量形式本身是守恒的，问题在“湿/干判据”不是通量。
**正确判据**：把“地板注入”和“通量守恒”分成两项报；质量漂移要和地板阈值一起报。
（断言：`test_periodic_mass_is_conserved`、`test_ritter_solution_formula_and_positivity`）

## 10. 报溃坝误差必须报窗口

**现象**：只给一个“全域 L1”，或拿全域 L1 估格式阶。
**实测**（$t=0.06,n=200$）：扇内 L1 $2.02\times10^{-2}$，全域 L1
$2.21\times10^{-2}$，比值 $1.093$；$n=100\to400$ 时扇内 L1 从
$3.04\times10^{-2}$ 降到 $1.25\times10^{-2}$（表值），实测阶 $0.669$、$0.694$。
全域 L1 混着干锋位置误差与激波抹平，阶会退化。
**正确判据**：格式阶用 Ritter 扇区内的窗口；全域值只作辅助，并注明它包含干锋。
（断言：`test_dam_break_error_windows_and_convergence`、`test_dam_break_windows_and_cfl_are_reported_separately`）

## 11. t=0 时 Ritter 扇区无定义

**现象**：在 $t=0$ 的帧里画“Ritter 精确解”和误差曲线，得到除零或 $0/0$。
**实测**：$t=0$ 时 $\xi=(x-x_0)/t$ 无定义，误差 arrays 必须是占位/掩码，
错误面板要显示“尚未推进”，而不是一条贴地的零线。
**正确判据**：帧的 `error_valid` 全 0，窗口说明明确写“t=0：Ritter 扇区无定义”。
（断言：`test_dam_break_error_windows_and_convergence` 的 t=0 分支、
`test_dam_break_windows_and_cfl_are_reported_separately`）

## 12. well-balanced 不是“源项随便加”

**现象**：湖面静止时用中心差分源项，水面出现 $10^{-4}$ 量级伪振荡。
**实测**：well-balanced（静水重构）水面漂移 $1.89\times10^{-16}$；朴素中心源项
$3.30\times10^{-4}$，相差 $1.7\times10^{12}$ 倍。后者不是“精度差一点”，
而是破坏了 $h+b=\text{const}$ 的离散平衡。
**正确判据**：静水问题上，源项离散必须和通量离散匹配；断言直接量水面漂移。
（断言：`test_lake_at_rest_well_balanced_vs_naive`）

## 13. CFL 越界要真出非有限值

**现象**：为了“稳健”给所有 $\Delta t$ 加 clip，于是 $cfl=2.0$ 也能跑。
**实测**：$cfl=0.8$ 时到达 $t_{\text{end}}$、$h_{\min}=0.426$、maxCFL $0.800$；
$cfl=2.0$ 时出现 NaN，帧必须把 `finite=False` 原样报出，不许兜底。
**正确判据**：稳定性边界要由实测 bracket 决定（$0.8$ 稳定、$2.0$ 崩溃），
页面用状态徽章显示“非有限”，而不是静默夹取到稳定区。
（断言：`test_cfl_stability_bracket`）

## 14. 波长对比差 7.5% 是“接近”还是“没对上”？

**现象**：turing 预设的末态斑点间距与线性理论对比，只报一个“符合”。
**实测**：末态主波长 $0.1253$，线性理论 $0.1166$，相差 $7.5\%$；斑点计数 96。
差的来源包括非线性饱和、离散波数与连续波数的差、以及有限域上的模式选择。
**正确判据**：把理论波长、实测波长、相对差与斑点计数一起报；二阶模板误差
$1.81$ 的收敛实验要和完整时间推进分开说，不能拿它冒充“整章二阶”。
（断言：`test_turing_pattern_saturates_near_analytic_wavelength`、`test_stencil_dispersion_error_converges_second_order`）

---

**一句话总结**：本章的每个可视化都要回答“这条证据排除了哪种错误解释”。残差排除
“方程没解完”，Ghia 排除“解错了方程”，图灵判据排除“图案等于失稳”，溃坝的窗口与
质量账本排除“误差小等于格式好”。任何一条只报好看的数字、不报窗口与地板的说法，
都应被视为未验证。
