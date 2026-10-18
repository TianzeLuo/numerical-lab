## 章节 0.0 数值断言报告

> 本表由 `tools/run_asserts.py` 从实际运行结果生成，**不是手写的**。

- 断言条数：2（A 1 / B 0 / C 1）
- 通过：1，失败：1
- 声明文件：`chapters/0.0/asserts.yaml`

| 类别 | 断言 | 理论值 | 实测值 | 容差 | 判定 |
|---|---|---|---|---|---|
| A | 阈值偏差的收敛阶 | 2 | 2.0894 | ±0.2 | ✅ |
| C | test_ftcs_threshold_converges_to_continuum_limit | 阈值随 nx 收敛到 1/2 | — | — | ❌ |

<details><summary>逐条细节（拟合窗口、数据点数、残差量级等）</summary>

- **阈值偏差的收敛阶**：R²=0.999817，斜率标准误=2.00e-02，用点 4 个（丢弃 0 个），局部阶=[2.163 2.077 2.037]
- **test_ftcs_threshold_converges_to_continuum_limit**：声明为 C 类，但本测试登记的全是 A 类记录：A 类必须与解析解/参考实现对拍、B 类是不变量、C 类是收敛率。要用 helper 记 A 类，请显式调用 record(kind="A", ...) 并 assert 它；helper 的缺省类别是 B。

</details>

**对账问题**：

- 0.0: 声明了 test_ftcs_spatial_order_is_two（A 类）但报告里没有这条断言
- 0.0: 声明了 test_oscillator_residual_scales_as_h_squared（A 类）但报告里没有这条断言
- 0.0: 声明了 test_underdamped_envelope_matches_theory（A 类）但报告里没有这条断言
- 0.0: 声明了 test_critical_damping_is_the_limit_of_underdamped（A 类）但报告里没有这条断言
- 0.0: 声明了 test_energy_conserved_without_damping（B 类）但报告里没有这条断言
- 0.0: 声明了 test_discrete_maximum_principle（B 类）但报告里没有这条断言
- 0.0: 声明了 test_heat_boundary_and_initial_conditions（B 类）但报告里没有这条断言
- 0.0: 声明了 test_heat_solution_stays_symmetric（B 类）但报告里没有这条断言
- 0.0: 声明了 test_ftcs_stability_threshold_measured（C 类）但报告里没有这条断言
- 0.0: 声明了 test_analytic_and_measured_stability_agree（C 类）但报告里没有这条断言
- 0.0: A 类断言只有 1 条，声明要求 ≥ 4
- 0.0: B 类断言只有 0 条，声明要求 ≥ 4
- 0.0: C 类断言只有 1 条，声明要求 ≥ 2
- 0.0: 缺少 B 类断言（asserts.yaml 要求必须有）
