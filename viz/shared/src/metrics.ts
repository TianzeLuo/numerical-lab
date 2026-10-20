/**
 * 关键读数：从帧的 `meta` 里挑出对当前画面最重要、含义明确的量，供 HTML 实时展示。
 *
 * 这里不是把 meta 原样倒出来。后端 meta 常含大量中间量（下标、采样数、内部开关），
 * 直接显示会变成一堵密码墙。只有下表中的键会进入界面，并且允许按可视化覆盖顺序与
 * 文案——例如同一个 `k` 在“求和”里是项数，在“1.1 放大镜”里是累加次数。
 */

import { formatNumber } from "./axes.js";

export interface MetricChip {
  key: string;
  label: string;
  value: string;
  unit: string;
}

interface MetricSpec {
  label: string;
  unit?: string;
  priority: number;
  /** 计数一律显示整数（四舍五入而不是截断）。 */
  integer?: boolean;
}

const META_LABELS: Readonly<Record<string, MetricSpec>> = {
  n_points: { label: "采样点数", priority: 6, integer: true },
  residual_0102_03_ulps: { label: "0.1+0.2 与 0.3 相差", unit: "ulp", priority: 9 },
  residual_0102_03: { label: "0.1+0.2 的表示误差", priority: 8 },
  gap: { label: "a−b", priority: 10 },
  digits_remaining: { label: "剩余有效位", unit: "位", priority: 9 },
  digits_lost: { label: "已损失有效位", unit: "位", priority: 8 },
  loss_digits: { label: "已损失有效位", unit: "位", priority: 7 },
  perturbation_amplification: { label: "输入误差放大", unit: "倍", priority: 8 },
  subtraction_relative_error: { label: "减法自身舍入误差", priority: 6 },
  kappa_measured: { label: "实测条件数 κ", priority: 10 },
  kappa_target: { label: "目标条件数 κ", priority: 8 },
  amplification_measured: { label: "当前方向放大比", priority: 9 },
  forward_error: { label: "前向误差", priority: 7 },
  backward_error: { label: "后向误差", priority: 6 },
  spectral_radius: { label: "谱半径", priority: 8 },
  sigma_min: { label: "最小奇异值", priority: 7 },
  sigma_max: { label: "最大奇异值", priority: 6 },
  residual: { label: "残差", priority: 8 },
  residual_max: { label: "最大残差", priority: 8 },
  residual_final: { label: "最终残差", priority: 7 },
  residual_current: { label: "当前残差", priority: 7 },
  defect_max: { label: "最大缺陷", priority: 7 },
  roundoff: { label: "舍入误差", priority: 6 },
  roundoff_floor: { label: "舍入地板", priority: 5 },
  error_floor: { label: "误差地板", priority: 5 },
  max_error: { label: "最大误差", priority: 9 },
  l2_error: { label: "L2 误差", priority: 9 },
  error_current: { label: "当前误差", priority: 9 },
  error_max: { label: "最大误差", priority: 8 },
  rel_error: { label: "相对误差", priority: 8 },
  measured_order: { label: "实测收敛阶", priority: 8 },
  declared_order: { label: "方法阶数", priority: 6 },
  order_hat: { label: "估计阶数", priority: 6 },
  slope: { label: "实测斜率", priority: 8 },
  slope_theory: { label: "理论斜率", priority: 6 },
  h: { label: "步长 / 网格 h", priority: 7 },
  dt: { label: "时间步长", priority: 6 },
  cfl: { label: "CFL 数", priority: 7 },
  n_iter: { label: "迭代次数", priority: 6, integer: true },
  n_steps: { label: "总步数", priority: 5, integer: true },
  n_elem: { label: "单元数", priority: 6, integer: true },
  n_dof: { label: "自由度", priority: 6, integer: true },
  tol: { label: "容差", priority: 5 },
  k_actual: { label: "实际迭代步", priority: 7, integer: true },
  k_current: { label: "当前截断阶", priority: 6, integer: true },
  rank: { label: "数值秩", priority: 5, integer: true },
  kappa: { label: "条件数 κ", priority: 8 },
  stiffness_ratio: { label: "刚性比", priority: 6 },
  n_rejected: { label: "被拒绝的步", priority: 5, integer: true },
  n_accepted: { label: "被接受的步", priority: 5, integer: true },
  n_f_evals: { label: "函数求值次数", priority: 4, integer: true },
  energy_drift: { label: "能量漂移", priority: 7 },
  max_defect: { label: "最大通量缺陷", priority: 7 },
  n_clipped: { label: "被截断的点", priority: 4, integer: true },
  noise_rel: { label: "噪声水平", priority: 5 },
  iterations: { label: "迭代次数", priority: 5, integer: true },
  error_naive: { label: "朴素求和误差", priority: 9 },
  error_kahan: { label: "Kahan 求和误差", priority: 9 },
  error_pairwise: { label: "成对求和误差", priority: 7 },
  bound_naive: { label: "朴素求和上界", priority: 6 },
  error_kahan_serial: { label: "串行 Kahan 误差", priority: 7 },
  max_error_equispaced_n: { label: "等距节点最大误差", priority: 7 },
  max_error_chebyshev_n: { label: "Chebyshev 最大误差", priority: 7 },
  divergence_ratio: { label: "两端发散比", priority: 6 },
  lambda_selected_n: { label: "当前 Λ_n", priority: 6 },
  error_spline: { label: "样条误差", priority: 7 },
  error_poly: { label: "多项式误差", priority: 7 },
  error_ratio: { label: "误差比（多项式 / 样条）", priority: 6 },
  lebesgue_constant: { label: "Lebesgue 常数", priority: 5 },
  jump_pchip_max: { label: "PCHIP 跳变", priority: 7 },
  jump_spline_max: { label: "样条跳变", priority: 7 },
  error_spline_max: { label: "样条最大误差", priority: 6 },
  rate_analytic_measured: { label: "解析函数收敛率", priority: 7 },
  alpha_cusp_measured: { label: "尖点收敛指数", priority: 7 },
  alpha_jump_measured: { label: "跳点收敛指数", priority: 7 },
  error_at_limit: { label: "代数精度极限处误差", priority: 7 },
  error_beyond_limit: { label: "超出精度后的误差", priority: 6 },
  failure_factor: { label: "高一次后误差放大", priority: 6 },
  degree_exact_limit: { label: "可精确积出的最高次数", priority: 5, integer: true },
  band_symplectic: { label: "辛方法能量带宽", priority: 8 },
  band_baseline: { label: "基线方法能量带宽", priority: 7 },
  ratio_band_over_baseline: { label: "能量带宽比", priority: 6 },
  short_term_error_symplectic: { label: "辛方法短期误差", priority: 5 },
  band_measured: { label: "实测能量带宽", priority: 7 },
  band_theory: { label: "理论带宽", priority: 6 },
  band_ratio_theory: { label: "实测 / 理论带宽", priority: 5 },
  invariant_residual: { label: "修正量守恒残差", priority: 5 },
  ecc_measured: { label: "实测偏心率", priority: 7 },
  ecc_error_sup: { label: "偏心率最大误差", priority: 6 },
  angmom_drift: { label: "角动量漂移", priority: 5 },
  varpi_drift_per_orbit: { label: "近日点每圈进动", priority: 6 },
  tv_max_increase_rel: { label: "总变差最大相对增长", priority: 7 },
  overshoot: { label: "过冲", priority: 6 },
  undershoot: { label: "欠冲", priority: 6 },
  second_order_fraction: { label: "二阶精度点比例", priority: 5 },
  mg_factor_mean: { label: "平均收敛因子", priority: 8 },
  speedup_max: { label: "最大加速比", priority: 7 },
  iters_smoother_max: { label: "光滑迭代最多步数", priority: 5, integer: true },
  iters_mg_max: { label: "多重网格最多步数", priority: 5, integer: true },
  max_rel_error: { label: "最大相对误差", priority: 7 },
  grad_norm_max: { label: "最大梯度范数", priority: 6 },
  objective_min: { label: "最小目标值", priority: 5 },
  semidiscrete_rel_error_max: { label: "半离散最大相对误差", priority: 7 },
  width_ratio_at_max_tau: { label: "影响宽度比", priority: 6 },
  boundary_layer_ratio: { label: "边界层内外比", priority: 5 },
  grad_fd_rel_error: { label: "梯度有限差分核对误差", priority: 5 },
  error_best_central: { label: "中心差分最小误差", priority: 7 },
  eps_opt_central: { label: "中心差分最优步长", priority: 6 },
  slope_down_central: { label: "小步长侧斜率", priority: 5 },
  slope_up_central: { label: "大步长侧斜率", priority: 5 },
  adjoint_growth_ratio: { label: "伴随变量增长比", priority: 7 },
  sensitivity_residual_max: { label: "灵敏度残差", priority: 6 },
  grad_stiffness: { label: "刚度参数梯度", priority: 5 },
  grad_damping: { label: "阻尼参数梯度", priority: 5 },
  amplitude: { label: "斑图振幅", priority: 7 },
  spot_count: { label: "斑点数", priority: 6, integer: true },
  wavelength: { label: "斑图波长", priority: 6 },
  dt_limit: { label: "显式稳定步长上限", priority: 5 },
  n_issues: { label: "构建问题数", priority: 8, integer: true },
  n_labels: { label: "标签数", priority: 6, integer: true },
  n_refs: { label: "引用次数", priority: 5, integer: true },
  n_table_rows: { label: "断言表行数", priority: 4, integer: true },
  n_chapters: { label: "章节数", priority: 7, integer: true },
  n_edges: { label: "依赖边数", priority: 6, integer: true },
  max_depth: { label: "最大依赖深度", priority: 6, integer: true },
  n_violations: { label: "依赖环路数", priority: 7, integer: true },
  lambda_chebyshev_n: { label: "Chebyshev 的 Λ_n", priority: 7 },
  lambda_equispaced_n: { label: "等距节点的 Λ_n", priority: 7 },
  lambda_ratio: { label: "两节点分布的 Λ 比", priority: 6 },
  kappa_n: { label: "当前条件数 κ(V)", priority: 7 },
  error_barycentric_n: { label: "重心公式误差", priority: 7 },
  error_vandermonde_n: { label: "Vandermonde 误差", priority: 7 },
  n_roots_current: { label: "当前零点数", priority: 6, integer: true },
  max_root_error: { label: "最大零点误差", priority: 7 },
  newton_iter_max: { label: "牛顿迭代上限", priority: 5, integer: true },
  rho_max: { label: "最大增长因子", priority: 8 },
  forward_max: { label: "最大前向误差", priority: 7 },
  n_singular: { label: "奇异主元次数", priority: 6, integer: true },
  max_level_residual: { label: "本层最大残差", priority: 7 },
  spectrum_error: { label: "频谱误差", priority: 6 },
  stage: { label: "当前级数", priority: 5, integer: true },
  max_wrap: { label: "最大卷绕比例", priority: 7 },
  wrap_rel: { label: "卷绕相对误差", priority: 6 },
  level: { label: "当前层", priority: 4, integer: true },
  pr_error: { label: "完美重构误差", priority: 6 },
  mra_error: { label: "多分辨误差", priority: 6 },
  interior_max: { label: "内部最大系数", priority: 6 },
  boundary_max: { label: "边界最大系数", priority: 5 },
  ratio_over_u: { label: "相对舍入放大", priority: 6 },
  low_mode_survival: { label: "低频存活率", priority: 7 },
  high_mode_survival: { label: "高频存活率", priority: 7 },
  measured_smoothing_factor: { label: "实测平滑因子", priority: 6 },
  factor_measured: { label: "实测收敛因子", priority: 8 },
  cycles_to_1e8: { label: "降到 1e-8 的周期", priority: 6, integer: true },
  error_norm_now: { label: "当前误差范数", priority: 6 },
  loss_ratio: { label: "误差保留比例", priority: 7 },
  hf_ratio_fine: { label: "细网格高频比例", priority: 6 },
  hf_ratio_coarse_visible: { label: "粗网格可见高频", priority: 5 },
  err_h_last: { label: "h-加密末级误差", priority: 7 },
  err_p_last: { label: "p-加密末级误差", priority: 7 },
  gain_p_over_h: { label: "p 相对 h 的增益", priority: 6 },
  eps: { label: "ε", priority: 6 },
  rate: { label: "速率", priority: 7 },
  settle_t: { label: "可见分歧时刻", priority: 7 },
  n_pairs: { label: "孪生对数量", priority: 4, integer: true },
  c_at_phi_max: { label: "最大波数相速度比", priority: 6 },
  decay_at_phi_max: { label: "最大波数衰减", priority: 5 },
  n_shown: { label: "当前样本数", priority: 5, integer: true },
  error_last: { label: "当前误差", priority: 8 },
  error_ratio_vs_mc: { label: "相对蒙特卡洛的误差比", priority: 6 },
  mean: { label: "均值", priority: 6 },
  std: { label: "标准差", priority: 5 },
  within_1p96_sigma: { label: "95% 区间内比例", priority: 4 },
  sum_first: { label: "一阶 Sobol' 指数", priority: 7 },
  sum_total: { label: "总效应指数", priority: 7 },
  interaction_share: { label: "交互作用占比", priority: 6 },
  error_outsample_current: { label: "样本外误差", priority: 8 },
  overfit_factor: { label: "过拟合因子", priority: 6 },
};

/** 每个可视化的读数顺序与（必要时）覆盖文案。 */
const VIZ_METRICS: Readonly<Record<string, readonly string[]>> = {
  "1.1/float_zoom": ["center", "spacing_center", "k", "residual_0102_03_ulps", "spacing_ratio"],
  "1.1/cancellation": ["gap", "digits_remaining", "loss_digits", "perturbation_amplification"],
  "1.1/cond_ellipsoid": ["kappa_measured", "kappa_target", "amplification_measured", "forward_error"],
  "1.1/summation_order": ["k", "error_naive", "error_kahan", "bound_naive"],
  "1.2/pivot_strategy": ["rho_max", "forward_max", "n_singular"],
  "2.1/lebesgue_growth": ["lambda_chebyshev_n", "lambda_equispaced_n", "lambda_ratio"],
  "2.1/barycentric_vs_vandermonde": ["kappa_n", "error_barycentric_n", "error_vandermonde_n"],
  "2.3/zeros_evolution": ["n_roots_current", "max_root_error", "newton_iter_max"],
  "2.5/butterfly_stages": ["stage", "max_level_residual", "spectrum_error"],
  "2.5/conv_theorem": ["max_wrap", "wrap_rel"],
  "2.6/decomp_tree": ["level", "pr_error", "mra_error"],
  "2.6/vanishing_moments": ["interior_max", "boundary_max", "ratio_over_u"],
  "3.5/lorenz_butterfly": ["eps", "rate", "settle_t", "n_pairs"],
  "4.2/phase_error": ["c_at_phi_max", "decay_at_phi_max"],
  "4.4/hp_convergence": ["err_h_last", "err_p_last", "gain_p_over_h"],
  "4.5/smoothing_spectrum": ["high_mode_survival", "low_mode_survival", "measured_smoothing_factor"],
  "4.5/vcycle_hierarchy": ["factor_measured", "cycles_to_1e8", "error_norm_now"],
  "4.5/residual_transfer": ["loss_ratio", "hf_ratio_fine", "hf_ratio_coarse_visible"],
  "5.4/point_fill": ["n_shown", "error_last", "error_ratio_vs_mc"],
  "5.4/output_hist": ["mean", "std", "within_1p96_sigma"],
  "5.4/sobol_bars": ["sum_first", "sum_total", "interaction_share"],
  "5.4/pce_spectrum": ["error_outsample_current", "overfit_factor"],
  "2.1/runge_nodes": ["max_error_equispaced_n", "max_error_chebyshev_n", "divergence_ratio", "lambda_selected_n"],
  "2.2/spline_vs_poly": ["error_spline", "error_poly", "error_ratio", "lebesgue_constant"],
  "2.2/smoothness_c0c1c2": ["jump_pchip_max", "jump_spline_max", "error_spline_max"],
  "2.3/convergence_rates": ["rate_analytic_measured", "alpha_cusp_measured", "alpha_jump_measured", "error_floor"],
  "2.4/gauss_vs_equispaced": ["error_at_limit", "error_beyond_limit", "failure_factor", "degree_exact_limit"],
  "3.3/energy_drift_duel": ["band_symplectic", "band_baseline", "ratio_band_over_baseline", "short_term_error_symplectic"],
  "3.3/modified_hamiltonian": ["band_measured", "band_theory", "band_ratio_theory", "invariant_residual"],
  "3.3/long_term_orbit": ["ecc_measured", "ecc_error_sup", "angmom_drift", "varpi_drift_per_orbit"],
  "4.2/limiter_interaction": ["tv_max_increase_rel", "overshoot", "undershoot", "second_order_fraction"],
  "4.5/grid_independence": ["mg_factor_mean", "speedup_max", "iters_smoother_max", "iters_mg_max"],
  "5.3/gradient_field": ["max_rel_error", "grad_norm_max", "objective_min"],
  "5.3/influence_field": ["semidiscrete_rel_error_max", "width_ratio_at_max_tau", "boundary_layer_ratio", "grad_fd_rel_error"],
  "5.3/gradient_check": ["error_best_central", "eps_opt_central", "slope_down_central", "slope_up_central"],
  "5.3/forward_adjoint_playback": ["adjoint_growth_ratio", "sensitivity_residual_max", "grad_stiffness", "grad_damping"],
  "6.1/turing_pattern": ["amplitude", "spot_count", "wavelength", "dt_limit"],
  "6.1/ghia_benchmark": ["residual", "max_error", "l2_error"],
  "6.2/catalog_nav": ["n_chapters", "n_edges", "max_depth", "n_violations"],
  "6.2/book_build": ["n_issues", "n_labels", "n_refs", "n_table_rows"],
};

const VIZ_LABELS: Readonly<Record<string, Readonly<Record<string, Partial<MetricSpec>>>>> = {
  "1.1/float_zoom": {
    k: { label: "累加次数" },
    n_points: { label: "窗口内浮点数" },
    center: { label: "窗口中心" },
    spacing_center: { label: "相邻 double 间距" },
    spacing_ratio: { label: "右侧 / 左侧间距" },
    residual_0102_03_ulps: { label: "0.1+0.2 比 0.3 大" },
  },
  "1.1/summation_order": { k: { label: "当前项数" } },
  "1.1/cancellation": {
    gap: { label: "a 与 b 的差" },
    digits_remaining: { label: "剩余有效位" },
    loss_digits: { label: "抵消损失位数 t", unit: "位" },
    perturbation_amplification: { label: "实测相对误差", unit: "" },
  },
  "3.5/lorenz_butterfly": { eps: { label: "初始差异 ε" }, rate: { label: "实测分离速率" } },
};

/** 文本化一个 meta 数值：计数取整，其它保留 4 位有效数字。 */
function formatMetricValue(value: number, spec: MetricSpec): string {
  if (!Number.isFinite(value)) return "—";
  if (spec.integer) return String(Math.round(value));
  if (Object.is(value, -0)) value = 0;
  return formatNumber(value, 4);
}

/**
 * 取当前帧的关键读数（最多 `limit` 条）。
 *
 * @param meta 帧的数值 meta
 * @param chapter 章号（用于按可视化覆盖）
 * @param viz 可视化 id
 */
export function keyMetrics(
  meta: Readonly<Record<string, number | boolean | null>>,
  chapter: string,
  viz: string,
  limit = 4,
): MetricChip[] {
  const picked: string[] = [];
  const overridden = VIZ_METRICS[`${chapter}/${viz}`];
  if (overridden !== undefined) {
    picked.push(...overridden);
  } else {
    const candidates = Object.keys(META_LABELS)
      .filter((key) => typeof meta[key] === "number" && Number.isFinite(meta[key] as number))
      .sort((a, b) => (META_LABELS[b]?.priority ?? 0) - (META_LABELS[a]?.priority ?? 0));
    picked.push(...candidates);
  }
  const out: MetricChip[] = [];
  const labels = VIZ_LABELS[`${chapter}/${viz}`] ?? {};
  for (const key of picked) {
    if (out.length >= limit) break;
    const base = META_LABELS[key];
    const raw = meta[key];
    if ((base === undefined && labels[key] === undefined) || typeof raw !== "number" || !Number.isFinite(raw)) continue;
    const spec: MetricSpec = {
      label: labels[key]?.label ?? base?.label ?? "当前值",
      unit: labels[key]?.unit ?? base?.unit,
      priority: base?.priority ?? 0,
      integer: labels[key]?.integer ?? base?.integer,
    };
    out.push({
      key,
      label: spec.label,
      value: formatMetricValue(raw, spec),
      unit: spec.unit ?? "",
    });
  }
  return out;
}
