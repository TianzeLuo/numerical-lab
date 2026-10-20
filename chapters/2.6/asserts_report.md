## 章节 2.6 数值断言报告

> 本表由 `tools/run_asserts.py` 从实际运行结果生成，**不是手写的**。

- 断言条数：878（A 27 / B 851 / C 0）
- 通过：878，失败：0
- 声明文件：`chapters/2.6/asserts.yaml`

| 类别 | 断言 | 理论值 | 实测值 | 容差 | 判定 |
|---|---|---|---|---|---|
| B | decomp_tree 有 requiredAux error | ≤ 1 | 1 | +1.00000e-09 | ✅ |
| B | decomp_tree 数组 signal 全有限 | ≤ 0 | 0 | +0 | ✅ |
| B | decomp_tree 数组 approx_tree 全有限 | ≤ 0 | 0 | +0 | ✅ |
| B | decomp_tree 数组 detail_tree 全有限 | ≤ 0 | 0 | +0 | ✅ |
| B | decomp_tree 数组 coeff_length 全有限 | ≤ 0 | 0 | +0 | ✅ |
| B | decomp_tree 数组 recon 全有限 | ≤ 0 | 0 | +0 | ✅ |
| B | decomp_tree 数组 error 全有限 | ≤ 0 | 0 | +0 | ✅ |
| B | decomp_tree 数组 energy_share 全有限 | ≤ 0 | 0 | +0 | ✅ |
| B | decomp_tree 数组 coeff_count 全有限 | ≤ 0 | 0 | +0 | ✅ |
| B | decomp_tree meta 只含 number/bool/null | ≤ 0 | 0 | +0 | ✅ |
| B | decomp_tree meta[redundancy] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | decomp_tree meta[pr_error] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | decomp_tree meta[mra_error] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | decomp_tree meta[stage_error] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | decomp_tree meta[rel_stage_error] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | decomp_tree meta[boundary_share] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | decomp_tree meta[energy_top] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | decomp_tree meta[energy_detail_sum] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | decomp_tree meta[max_detail] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | decomp_tree meta[t_actual] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | decomp_tree 有 requiredAux error | ≤ 1 | 1 | +1.00000e-09 | ✅ |
| B | decomp_tree 数组 signal 全有限 | ≤ 0 | 0 | +0 | ✅ |
| B | decomp_tree 数组 approx_tree 全有限 | ≤ 0 | 0 | +0 | ✅ |
| B | decomp_tree 数组 detail_tree 全有限 | ≤ 0 | 0 | +0 | ✅ |
| B | decomp_tree 数组 coeff_length 全有限 | ≤ 0 | 0 | +0 | ✅ |
| B | decomp_tree 数组 recon 全有限 | ≤ 0 | 0 | +0 | ✅ |
| B | decomp_tree 数组 error 全有限 | ≤ 0 | 0 | +0 | ✅ |
| B | decomp_tree 数组 energy_share 全有限 | ≤ 0 | 0 | +0 | ✅ |
| B | decomp_tree 数组 coeff_count 全有限 | ≤ 0 | 0 | +0 | ✅ |
| B | decomp_tree meta 只含 number/bool/null | ≤ 0 | 0 | +0 | ✅ |
| B | decomp_tree meta[redundancy] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | decomp_tree meta[pr_error] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | decomp_tree meta[mra_error] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | decomp_tree meta[stage_error] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | decomp_tree meta[rel_stage_error] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | decomp_tree meta[boundary_share] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | decomp_tree meta[energy_top] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | decomp_tree meta[energy_detail_sum] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | decomp_tree meta[max_detail] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | decomp_tree meta[t_actual] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | time_scale_map 有 requiredAux error | ≤ 1 | 1 | +1.00000e-09 | ✅ |
| B | time_scale_map 数组 signal 全有限 | ≤ 0 | 0 | +0 | ✅ |
| B | time_scale_map 数组 scalogram 全有限 | ≤ 0 | 0 | +0 | ✅ |
| B | time_scale_map 数组 stft_map 全有限 | ≤ 0 | 0 | +0 | ✅ |
| B | time_scale_map 数组 stft_time 全有限 | ≤ 0 | 0 | +0 | ✅ |
| B | time_scale_map 数组 stft_freq 全有限 | ≤ 0 | 0 | +0 | ✅ |
| B | time_scale_map 数组 wav_width 全有限 | ≤ 0 | 0 | +0 | ✅ |
| B | time_scale_map 数组 wav_band 全有限 | ≤ 0 | 0 | +0 | ✅ |
| B | time_scale_map 数组 stft_width 全有限 | ≤ 0 | 0 | +0 | ✅ |
| B | time_scale_map 数组 stft_band 全有限 | ≤ 0 | 0 | +0 | ✅ |
| B | time_scale_map 数组 cell_area 全有限 | ≤ 0 | 0 | +0 | ✅ |
| B | time_scale_map 数组 error 全有限 | ≤ 0 | 0 | +0 | ✅ |
| B | time_scale_map meta 只含 number/bool/null | ≤ 0 | 0 | +0 | ✅ |
| B | time_scale_map meta[n_stft_time] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | time_scale_map meta[n_stft_freq] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | time_scale_map meta[pr_error] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | time_scale_map meta[swt_energy_ratio] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | time_scale_map meta[swt_symbol_mean] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | time_scale_map meta[swt_redundancy] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | time_scale_map meta[cell_area_min] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | time_scale_map meta[cell_area_max] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | time_scale_map meta[t_actual] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | denoise_threshold 有 requiredAux error | ≤ 1 | 1 | +1.00000e-09 | ✅ |
| B | denoise_threshold 数组 clean 全有限 | ≤ 0 | 0 | +0 | ✅ |
| B | denoise_threshold 数组 noisy 全有限 | ≤ 0 | 0 | +0 | ✅ |
| B | denoise_threshold 数组 denoised 全有限 | ≤ 0 | 0 | +0 | ✅ |
| B | denoise_threshold 数组 error 全有限 | ≤ 0 | 0 | +0 | ✅ |
| B | denoise_threshold 数组 lambda_grid 全有限 | ≤ 0 | 0 | +0 | ✅ |
| B | denoise_threshold 数组 mse_curve 全有限 | ≤ 0 | 0 | +0 | ✅ |
| B | denoise_threshold 数组 bias2_curve 全有限 | ≤ 0 | 0 | +0 | ✅ |
| B | denoise_threshold 数组 var_curve 全有限 | ≤ 0 | 0 | +0 | ✅ |
| B | denoise_threshold 数组 mse_noisy_line 全有限 | ≤ 0 | 0 | +0 | ✅ |
| B | denoise_threshold 数组 universal_line 全有限 | ≤ 0 | 0 | +0 | ✅ |
| B | denoise_threshold meta 只含 number/bool/null | ≤ 0 | 0 | +0 | ✅ |
| B | denoise_threshold meta[sigma] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | denoise_threshold meta[sigma_hat] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | denoise_threshold meta[sigma_hat_bias] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | denoise_threshold meta[lam] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | denoise_threshold meta[mse] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | denoise_threshold meta[mse_curve_here] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | denoise_threshold meta[lam_grid_here] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | denoise_threshold meta[mse_noisy] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | denoise_threshold meta[mse_opt] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | denoise_threshold meta[lam_opt] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | denoise_threshold meta[lam_grid] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | denoise_threshold meta[universal] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | denoise_threshold meta[universal_over_opt] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | denoise_threshold meta[mse_at_universal] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | denoise_threshold meta[bias2_curve_here] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | denoise_threshold meta[var_curve_here] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | denoise_threshold meta[snr_in] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | denoise_threshold meta[snr_out] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | denoise_threshold meta[rel_error] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | denoise_threshold meta[n_trials] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | denoise_threshold meta[t_actual] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | vanishing_moments 有 requiredAux error | ≤ 1 | 1 | +1.00000e-09 | ✅ |
| B | vanishing_moments 数组 signal 全有限 | ≤ 0 | 0 | +0 | ✅ |
| B | vanishing_moments 数组 detail_tree 全有限 | ≤ 0 | 0 | +0 | ✅ |
| B | vanishing_moments 数组 detail_length 全有限 | ≤ 0 | 0 | +0 | ✅ |
| B | vanishing_moments 数组 degree_axis 全有限 | ≤ 0 | 0 | +0 | ✅ |
| B | vanishing_moments 数组 interior_ratio 全有限 | ≤ 0 | 0 | +0 | ✅ |
| B | vanishing_moments 数组 interior_valid 全有限 | ≤ 0 | 0 | +0 | ✅ |
| B | vanishing_moments 数组 boundary_ratio 全有限 | ≤ 0 | 0 | +0 | ✅ |
| B | vanishing_moments 数组 u_line 全有限 | ≤ 0 | 0 | +0 | ✅ |
| B | vanishing_moments 数组 error 全有限 | ≤ 0 | 0 | +0 | ✅ |
| B | vanishing_moments 数组 level_ratio 全有限 | ≤ 0 | 0 | +0 | ✅ |
| B | vanishing_moments 数组 level_boundary 全有限 | ≤ 0 | 0 | +0 | ✅ |
| B | vanishing_moments meta 只含 number/bool/null | ≤ 0 | 0 | +0 | ✅ |
| B | vanishing_moments meta[interior_max] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | vanishing_moments meta[boundary_max] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | vanishing_moments meta[ratio_over_u] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | vanishing_moments meta[level1_ratio] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | vanishing_moments meta[n_interior] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | vanishing_moments meta[n_ratio_floored] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | vanishing_moments meta[u] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | vanishing_moments meta[pr_error] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | vanishing_moments meta[pollution_level1] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | vanishing_moments meta[pollution_span1] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | vanishing_moments meta[t_actual] 有限 | ≤ 1 | 1 | +0 | ✅ |
| B | 四个帧的数组与 strings | 全部有限、requiredAux 齐备 | 4 | 0.5 | ✅ |
| B | 本章 4 个可视化 | 4 | 4 | ±(0·\|t\|+0) | ✅ |
| B | viz id 不重复 | 4 | 4 | ±(0·\|t\|+0) | ✅ |
| B | decomp_tree dataSource = backend | 1 | 1 | ±(0·\|t\|+0) | ✅ |
| B | decomp_tree pitfalls 非空 | ≤ 1.00000e+09 | 140 | +0 | ✅ |
| B | decomp_tree pitfalls 长度 >= 60 字 | ≤ 140 | 60 | +0 | ✅ |
| B | decomp_tree theory 至少 1 条 | ≤ 4 | 1 | +0 | ✅ |
| B | decomp_tree.tree 显式声明 autoscaled | 1 | 1 | ±(0·\|t\|+0) | ✅ |
| B | decomp_tree.recon 显式声明 autoscaled | 1 | 1 | ±(0·\|t\|+0) | ✅ |
| B | decomp_tree.error 显式声明 autoscaled | 1 | 1 | ±(0·\|t\|+0) | ✅ |
| B | decomp_tree.tree 坐标轴尺度合法 | 1 | 1 | ±(0·\|t\|+0) | ✅ |
| B | decomp_tree.recon 坐标轴尺度合法 | 1 | 1 | ±(0·\|t\|+0) | ✅ |
| B | decomp_tree.error 坐标轴尺度合法 | 1 | 1 | ±(0·\|t\|+0) | ✅ |
| B | decomp_tree.signal 默认值在 options 内 | ≤ 1 | 1 | +0 | ✅ |
| B | decomp_tree.signal options 至少 2 个 | ≤ 6 | 2 | +0 | ✅ |
| B | decomp_tree.n default >= min | ≤ 256 | 64 | +0 | ✅ |
| B | decomp_tree.n default <= max | ≤ 1024 | 256 | +0 | ✅ |
| B | decomp_tree.n critical 有解释 | ≤ 60 | 1 | +0 | ✅ |
| B | decomp_tree.n critical 256 >= min | ≤ 256 | 64 | +0 | ✅ |
| B | decomp_tree.n critical 256 <= max | ≤ 1024 | 256 | +0 | ✅ |
| B | decomp_tree.wavelet 默认值在 options 内 | ≤ 1 | 1 | +0 | ✅ |
| B | decomp_tree.wavelet options 至少 2 个 | ≤ 5 | 2 | +0 | ✅ |
| B | decomp_tree.level default >= min | ≤ 4 | 1 | +0 | ✅ |
| B | decomp_tree.level default <= max | ≤ 6 | 4 | +0 | ✅ |
| B | decomp_tree.level critical 有解释 | ≤ 93 | 1 | +0 | ✅ |
| B | decomp_tree.level critical 4 >= min | ≤ 4 | 1 | +0 | ✅ |
| B | decomp_tree.level critical 4 <= max | ≤ 6 | 4 | +0 | ✅ |
| B | decomp_tree.mode 默认值在 options 内 | ≤ 1 | 1 | +0 | ✅ |
| B | decomp_tree.mode options 至少 2 个 | ≤ 3 | 2 | +0 | ✅ |
| B | time_scale_map dataSource = backend | 1 | 1 | ±(0·\|t\|+0) | ✅ |
| B | time_scale_map pitfalls 非空 | ≤ 1.00000e+09 | 153 | +0 | ✅ |
| B | time_scale_map pitfalls 长度 >= 60 字 | ≤ 153 | 60 | +0 | ✅ |
| B | time_scale_map theory 至少 1 条 | ≤ 4 | 1 | +0 | ✅ |
| B | time_scale_map.scale 显式声明 autoscaled | 1 | 1 | ±(0·\|t\|+0) | ✅ |
| B | time_scale_map.stft 显式声明 autoscaled | 1 | 1 | ±(0·\|t\|+0) | ✅ |
| B | time_scale_map.error 显式声明 autoscaled | 1 | 1 | ±(0·\|t\|+0) | ✅ |
| B | time_scale_map.scale 坐标轴尺度合法 | 1 | 1 | ±(0·\|t\|+0) | ✅ |
| B | time_scale_map.stft 坐标轴尺度合法 | 1 | 1 | ±(0·\|t\|+0) | ✅ |
| B | time_scale_map.error 坐标轴尺度合法 | 1 | 1 | ±(0·\|t\|+0) | ✅ |
| B | time_scale_map time.mode=none 时不得声明 tMax | 0 | 0 | ±(0·\|t\|+0) | ✅ |
| B | time_scale_map time.mode=none 时不得声明 dt | 0 | 0 | ±(0·\|t\|+0) | ✅ |
| B | time_scale_map time.mode=none 时不得声明 fps | 0 | 0 | ±(0·\|t\|+0) | ✅ |
| B | time_scale_map.signal 默认值在 options 内 | ≤ 1 | 1 | +0 | ✅ |
| B | time_scale_map.signal options 至少 2 个 | ≤ 4 | 2 | +0 | ✅ |
| B | time_scale_map.n default >= min | ≤ 256 | 128 | +0 | ✅ |
| B | time_scale_map.n default <= max | ≤ 512 | 256 | +0 | ✅ |
| B | time_scale_map.wavelet 默认值在 options 内 | ≤ 1 | 1 | +0 | ✅ |
| B | time_scale_map.wavelet options 至少 2 个 | ≤ 3 | 2 | +0 | ✅ |
| B | time_scale_map.level default >= min | ≤ 4 | 2 | +0 | ✅ |
| B | time_scale_map.level default <= max | ≤ 5 | 4 | +0 | ✅ |
| B | time_scale_map.level critical 有解释 | ≤ 74 | 1 | +0 | ✅ |
| B | time_scale_map.level critical 5 >= min | ≤ 5 | 2 | +0 | ✅ |
| B | time_scale_map.level critical 5 <= max | ≤ 5 | 5 | +0 | ✅ |
| B | time_scale_map.stft_window default >= min | ≤ 32 | 8 | +0 | ✅ |
| B | time_scale_map.stft_window default <= max | ≤ 64 | 32 | +0 | ✅ |
| B | time_scale_map.stft_window critical 有解释 | ≤ 83 | 1 | +0 | ✅ |
| B | time_scale_map.stft_window critical 16 >= min | ≤ 16 | 8 | +0 | ✅ |
| B | time_scale_map.stft_window critical 16 <= max | ≤ 64 | 16 | +0 | ✅ |
| B | denoise_threshold dataSource = backend | 1 | 1 | ±(0·\|t\|+0) | ✅ |
| B | denoise_threshold pitfalls 非空 | ≤ 1.00000e+09 | 165 | +0 | ✅ |
| B | denoise_threshold pitfalls 长度 >= 60 字 | ≤ 165 | 60 | +0 | ✅ |
| B | denoise_threshold theory 至少 1 条 | ≤ 4 | 1 | +0 | ✅ |
| B | denoise_threshold.signals 显式声明 autoscaled | 1 | 1 | ±(0·\|t\|+0) | ✅ |
| B | denoise_threshold.curves 显式声明 autoscaled | 1 | 1 | ±(0·\|t\|+0) | ✅ |
| B | denoise_threshold.error 显式声明 autoscaled | 1 | 1 | ±(0·\|t\|+0) | ✅ |
| B | denoise_threshold.signals 坐标轴尺度合法 | 1 | 1 | ±(0·\|t\|+0) | ✅ |
| B | denoise_threshold.curves 坐标轴尺度合法 | 1 | 1 | ±(0·\|t\|+0) | ✅ |
| B | denoise_threshold.error 坐标轴尺度合法 | 1 | 1 | ±(0·\|t\|+0) | ✅ |
| B | denoise_threshold time.mode=none 时不得声明 tMax | 0 | 0 | ±(0·\|t\|+0) | ✅ |
| B | denoise_threshold time.mode=none 时不得声明 dt | 0 | 0 | ±(0·\|t\|+0) | ✅ |
| B | denoise_threshold time.mode=none 时不得声明 fps | 0 | 0 | ±(0·\|t\|+0) | ✅ |
| B | denoise_threshold.signal 默认值在 options 内 | ≤ 1 | 1 | +0 | ✅ |
| B | denoise_threshold.signal options 至少 2 个 | ≤ 3 | 2 | +0 | ✅ |
| B | denoise_threshold.n default >= min | ≤ 256 | 128 | +0 | ✅ |
| B | denoise_threshold.n default <= max | ≤ 512 | 256 | +0 | ✅ |
| B | denoise_threshold.sigma default >= min | ≤ 0.3 | 0.05 | +0 | ✅ |
| B | denoise_threshold.sigma default <= max | ≤ 0.6 | 0.3 | +0 | ✅ |
| B | denoise_threshold.lam default >= min | ≤ 0.28 | 0 | +0 | ✅ |
| B | denoise_threshold.lam default <= max | ≤ 1.6 | 0.28 | +0 | ✅ |
| B | denoise_threshold.lam critical 有解释 | ≤ 126 | 1 | +0 | ✅ |
| B | denoise_threshold.lam critical 0.28 >= min | ≤ 0.28 | 0 | +0 | ✅ |
| B | denoise_threshold.lam critical 0.28 <= max | ≤ 1.6 | 0.28 | +0 | ✅ |
| B | denoise_threshold.lam critical 1 >= min | ≤ 1 | 0 | +0 | ✅ |
| B | denoise_threshold.lam critical 1 <= max | ≤ 1.6 | 1 | +0 | ✅ |
| B | denoise_threshold.rule 默认值在 options 内 | ≤ 1 | 1 | +0 | ✅ |
| B | denoise_threshold.rule options 至少 2 个 | ≤ 2 | 2 | +0 | ✅ |
| B | denoise_threshold.wavelet 默认值在 options 内 | ≤ 1 | 1 | +0 | ✅ |
| B | denoise_threshold.wavelet options 至少 2 个 | ≤ 3 | 2 | +0 | ✅ |
| B | denoise_threshold.level default >= min | ≤ 4 | 1 | +0 | ✅ |
| B | denoise_threshold.level default <= max | ≤ 6 | 4 | +0 | ✅ |
| B | vanishing_moments dataSource = backend | 1 | 1 | ±(0·\|t\|+0) | ✅ |
| B | vanishing_moments pitfalls 非空 | ≤ 1.00000e+09 | 154 | +0 | ✅ |
| B | vanishing_moments pitfalls 长度 >= 60 字 | ≤ 154 | 60 | +0 | ✅ |
| B | vanishing_moments theory 至少 1 条 | ≤ 4 | 1 | +0 | ✅ |
| B | vanishing_moments.poly 显式声明 autoscaled | 1 | 1 | ±(0·\|t\|+0) | ✅ |
| B | vanishing_moments.ratio 显式声明 autoscaled | 1 | 1 | ±(0·\|t\|+0) | ✅ |
| B | vanishing_moments.error 显式声明 autoscaled | 1 | 1 | ±(0·\|t\|+0) | ✅ |
| B | vanishing_moments.poly 坐标轴尺度合法 | 1 | 1 | ±(0·\|t\|+0) | ✅ |
| B | vanishing_moments.ratio 坐标轴尺度合法 | 1 | 1 | ±(0·\|t\|+0) | ✅ |
| B | vanishing_moments.error 坐标轴尺度合法 | 1 | 1 | ±(0·\|t\|+0) | ✅ |
| B | vanishing_moments time.mode=none 时不得声明 tMax | 0 | 0 | ±(0·\|t\|+0) | ✅ |
| B | vanishing_moments time.mode=none 时不得声明 dt | 0 | 0 | ±(0·\|t\|+0) | ✅ |
| B | vanishing_moments time.mode=none 时不得声明 fps | 0 | 0 | ±(0·\|t\|+0) | ✅ |
| B | vanishing_moments.wavelet 默认值在 options 内 | ≤ 1 | 1 | +0 | ✅ |
| B | vanishing_moments.wavelet options 至少 2 个 | ≤ 5 | 2 | +0 | ✅ |
| B | vanishing_moments.degree default >= min | ≤ 3 | 0 | +0 | ✅ |
| B | vanishing_moments.degree default <= max | ≤ 6 | 3 | +0 | ✅ |
| B | vanishing_moments.degree critical 有解释 | ≤ 100 | 1 | +0 | ✅ |
| B | vanishing_moments.degree critical 4 >= min | ≤ 4 | 0 | +0 | ✅ |
| B | vanishing_moments.degree critical 4 <= max | ≤ 6 | 4 | +0 | ✅ |
| B | vanishing_moments.n default >= min | ≤ 256 | 64 | +0 | ✅ |
| B | vanishing_moments.n default <= max | ≤ 512 | 256 | +0 | ✅ |
| B | vanishing_moments.level default >= min | ≤ 3 | 1 | +0 | ✅ |
| B | vanishing_moments.level default <= max | ≤ 5 | 3 | +0 | ✅ |
| B | vanishing_moments.mode 默认值在 options 内 | ≤ 1 | 1 | +0 | ✅ |
| B | vanishing_moments.mode options 至少 2 个 | ≤ 3 | 2 | +0 | ✅ |
| B | manifest 契约 | 4 个 viz、字段齐全、区间自洽 | 4 | 0.5 | ✅ |
| B | pr_error = idwt 直算路径 | 3.000e-16 | 3.000e-16 | ±(0·\|t\|+0) | ✅ |
| B | mra_error = 逐带求和路径 | 1.776e-15 | 1.776e-15 | ±(0·\|t\|+0) | ✅ |
| B | n_ext 一致 | 288 | 288 | ±(0·\|t\|+0) | ✅ |
| B | 冗余度一致 | 1.125 | 1.125 | ±(1e-15·\|t\|+0) | ✅ |
| B | coeff_count = coefficient_lengths | 0 | 0 | ±(0·\|t\|+0) | ✅ |
| B | energy_share = level_energies | 0 | 0 | ±(0·\|t\|+0) | ✅ |
| B | 逐层重构全部有限 | ≤ 0 | 0 | +0 | ✅ |
| B | 逐层重构末行 = x（numcore 口径） | ≤ 2.84217e-14 | 1.776e-15 | +2.84217e-14 | ✅ |
| B | t=tMax 时 stage_error 落到地板 | ≤ 1.00000e-12 | 1.776e-15 | +0 | ✅ |
| B | boundary_share 复算一致 | 0.000291707 | 0.000291707 | ±(1e-12·\|t\|+0) | ✅ |
| B | 帧里的 signal 就是 test_signal 的输出 | 0 | 0 | ±(0·\|t\|+0) | ✅ |
| B | SWT 能量比复算一致 | 9.03172 | 9.03172 | ±(1e-12·\|t\|+0) | ✅ |
| B | 框架符号平均复算一致 | 5.01367 | 5.01367 | ±(1e-12·\|t\|+0) | ✅ |
| B | SWT 冗余度 = J+1 | 5 | 5 | ±(0·\|t\|+0) | ✅ |
| B | 尺度图行数 = J+1 | 5 | 5 | ±(0·\|t\|+0) | ✅ |
| B | 每层系数与信号等长（非抽取） | 256 | 256 | ±(0·\|t\|+0) | ✅ |
| B | scalogram 数组 = numcore 输出 | 0 | 0 | ±(0·\|t\|+0) | ✅ |
| B | SWT 往返误差复算一致 | 4.441e-16 | 4.441e-16 | ±(1e-12·\|t\|+0) | ✅ |
| B | STFT 频率格数与 numcore 一致 | 17 | 17 | ±(0·\|t\|+0) | ✅ |
| B | STFT 时间格数与 numcore 一致 | 29 | 29 | ±(0·\|t\|+0) | ✅ |
| B | stft_map 数组 = numcore 输出 | 0 | 0 | ±(0·\|t\|+0) | ✅ |
| B | 小波带宽数组一致 | 0 | 0 | ±(0·\|t\|+0) | ✅ |
| B | 带宽随层数下降（多分辨） | ≤ 1000 | 8 | +1000 | ✅ |
| B | 时间-尺度帧的两套分辨率 | 小波：时间宽 2^j、带宽 1/2^j；STFT：两者恒定 | 0.222656 | 1.000e+09 | ✅ |
| B | clean 数组 = test_signal 输出 | 0 | 0 | ±(0·\|t\|+0) | ✅ |
| B | noisy - clean 的标准差 = sigma | 0.3 | 0.3 | ±(1e-09·\|t\|+0) | ✅ |
| B | sigma_hat 复算一致 | 0.454763 | 0.454763 | ±(1e-12·\|t\|+0) | ✅ |
| B | 通用阈值复算一致 | 0.999066 | 0.999066 | ±(1e-12·\|t\|+0) | ✅ |
| B | denoised 数组 = numcore denoise 输出 | 0 | 0 | ±(0·\|t\|+0) | ✅ |
| B | meta.mse = 帧内单次实现的 MSE（画出来的那条曲线的读数） | 0.0388936 | 0.0388936 | ±(0·\|t\|+0) | ✅ |
| B | lambda_grid 与同种子重算逐点相同 | 0 | 0 | ±(0·\|t\|+0) | ✅ |
| B | mse_curve 与同种子重算逐点相同 | 0 | 0 | ±(0·\|t\|+0) | ✅ |
| B | bias2_curve 与同种子重算逐点相同 | 0 | 0 | ±(0·\|t\|+0) | ✅ |
| B | var_curve 与同种子重算逐点相同 | 0 | 0 | ±(0·\|t\|+0) | ✅ |
| B | 最优阈值复算一致 | 0.278475 | 0.278475 | ±(1e-12·\|t\|+0) | ✅ |
| B | 最优 MSE 复算一致 | 0.0468315 | 0.0468315 | ±(1e-09·\|t\|+0) | ✅ |
| B | 曲线上的点满足 MSE = bias^2 + var | 0.0468315 | 0.0468315 | ±(0·\|t\|+7.10543e-15) | ✅ |
| B | mse_curve_here 落在同一条曲线上 | 0.0468315 | 0.0468315 | ±(0·\|t\|+0) | ✅ |
| B | var_curve_here 落在同一条曲线上 | 0.0292461 | 0.0292461 | ±(0·\|t\|+0) | ✅ |
| B | lam_grid_here = 最接近 lambda 的栅格点 | 0.3 | 0.3 | ±(0·\|t\|+0) | ✅ |
| B | 消失矩阶数一致 | 4 | 4 | ±(0·\|t\|+0) | ✅ |
| B | u 参考线 = 2^-53 | 1.110e-16 | 1.110e-16 | ±(0·\|t\|+0) | ✅ |
| B | signal 数组 = t^k | 0 | 0 | ±(0·\|t\|+0) | ✅ |
| B | 往返误差复算一致 | 5.551e-16 | 5.551e-16 | ±(1e-12·\|t\|+0) | ✅ |
| B | interior_max 复算一致 | 9.302e-17 | 9.302e-17 | ±(1e-12·\|t\|+0) | ✅ |
| B | boundary_max 复算一致 | 0.644509 | 0.644509 | ±(1e-12·\|t\|+0) | ✅ |
| B | killed 标志一致 | 1 | 1 | ±(0·\|t\|+0) | ✅ |
| B | 次数扫描轴 = 0..N+2 | 0 | 0 | ±(0·\|t\|+0) | ✅ |
| B | k=0 掩码为 0 时实测严格为 0 | 0 | 0 | ±(0·\|t\|+0) | ✅ |
| B | k=0 画出来的是地板值 | 1.000e-18 | 1.000e-18 | ±(0·\|t\|+0) | ✅ |
| B | k=1 的比值是实测值 | 2.077e-16 | 2.077e-16 | ±(1e-12·\|t\|+0) | ✅ |
| B | k=1 掩码为 1 时实测 > 0 | ≤ 2.07680e-16 | 0 | +0 | ✅ |
| B | k=2 的比值是实测值 | 8.791e-17 | 8.791e-17 | ±(1e-12·\|t\|+0) | ✅ |
| B | k=2 掩码为 1 时实测 > 0 | ≤ 8.79054e-17 | 0 | +0 | ✅ |
| B | k=3 的比值是实测值 | 9.302e-17 | 9.302e-17 | ±(1e-12·\|t\|+0) | ✅ |
| B | k=3 掩码为 1 时实测 > 0 | ≤ 9.30169e-17 | 0 | +0 | ✅ |
| B | k=4 的比值是实测值 | 1.061e-05 | 1.061e-05 | ±(1e-12·\|t\|+0) | ✅ |
| B | k=4 掩码为 1 时实测 > 0 | ≤ 1.06099e-05 | 0 | +0 | ✅ |
| B | k=5 的比值是实测值 | 4.734e-05 | 4.734e-05 | ±(1e-12·\|t\|+0) | ✅ |
| B | k=5 掩码为 1 时实测 > 0 | ≤ 4.73380e-05 | 0 | +0 | ✅ |
| B | k=6 的比值是实测值 | 0.000126871 | 0.000126871 | ±(1e-12·\|t\|+0) | ✅ |
| B | k=6 掩码为 1 时实测 > 0 | ≤ 1.26871e-04 | 0 | +0 | ✅ |
| B | 地板点计数一致 | 1 | 1 | ±(0·\|t\|+0) | ✅ |
| B | u 参考线数组 = 2^-53 | 0 | 0 | ±(0·\|t\|+0) | ✅ |
| B | 第 1 层量级比复算一致 | 8.038e-17 | 8.038e-17 | ±(1e-12·\|t\|+0) | ✅ |
| B | 边界污染幅度复算一致 | 0.968302 | 0.968302 | ±(1e-12·\|t\|+0) | ✅ |
| B | 边界污染影响区复算一致 | 6 | 6 | ±(1e-12·\|t\|+0) | ✅ |
| B | strings 里的名字与 numcore 一致 | 1 | 1 | ±(0·\|t\|+0) | ✅ |
| B | strings 里的边界模式与 numcore 一致 | 1 | 1 | ±(0·\|t\|+0) | ✅ |
| B | periodic/haar/bumps 往返误差一致 | 4.000e-16 | 4.000e-16 | ±(0·\|t\|+0) | ✅ |
| B | periodic/haar/bumps n_ext 一致 | 128 | 128 | ±(0·\|t\|+0) | ✅ |
| B | periodic/haar/bumps 末段误差落地板 | ≤ 1.00000e-12 | 1.332e-15 | +0 | ✅ |
| B | periodic/haar/blocks 往返误差一致 | 2.961e-16 | 2.961e-16 | ±(0·\|t\|+0) | ✅ |
| B | periodic/haar/blocks n_ext 一致 | 128 | 128 | ±(0·\|t\|+0) | ✅ |
| B | periodic/haar/blocks 末段误差落地板 | ≤ 1.00000e-12 | 1.776e-15 | +0 | ✅ |
| B | periodic/haar/heavisine 往返误差一致 | 3.907e-16 | 3.907e-16 | ±(0·\|t\|+0) | ✅ |
| B | periodic/haar/heavisine n_ext 一致 | 128 | 128 | ±(0·\|t\|+0) | ✅ |
| B | periodic/haar/heavisine 末段误差落地板 | ≤ 1.00000e-12 | 1.332e-15 | +0 | ✅ |
| B | periodic/db8/bumps 往返误差一致 | 4.000e-16 | 4.000e-16 | ±(0·\|t\|+0) | ✅ |
| B | periodic/db8/bumps n_ext 一致 | 128 | 128 | ±(0·\|t\|+0) | ✅ |
| B | periodic/db8/bumps 末段误差落地板 | ≤ 1.00000e-12 | 1.110e-15 | +0 | ✅ |
| B | periodic/db8/blocks 往返误差一致 | 5.921e-16 | 5.921e-16 | ±(0·\|t\|+0) | ✅ |
| B | periodic/db8/blocks n_ext 一致 | 128 | 128 | ±(0·\|t\|+0) | ✅ |
| B | periodic/db8/blocks 末段误差落地板 | ≤ 1.00000e-12 | 3.109e-15 | +0 | ✅ |
| B | periodic/db8/heavisine 往返误差一致 | 7.814e-16 | 7.814e-16 | ±(0·\|t\|+0) | ✅ |
| B | periodic/db8/heavisine n_ext 一致 | 128 | 128 | ±(0·\|t\|+0) | ✅ |
| B | periodic/db8/heavisine 末段误差落地板 | ≤ 1.00000e-12 | 1.776e-15 | +0 | ✅ |
| B | symmetric/haar/bumps 往返误差一致 | 4.000e-16 | 4.000e-16 | ±(0·\|t\|+0) | ✅ |
| B | symmetric/haar/bumps n_ext 一致 | 136 | 136 | ±(0·\|t\|+0) | ✅ |
| B | symmetric/haar/bumps 末段误差落地板 | ≤ 1.00000e-12 | 1.332e-15 | +0 | ✅ |
| B | symmetric/haar/blocks 往返误差一致 | 2.961e-16 | 2.961e-16 | ±(0·\|t\|+0) | ✅ |
| B | symmetric/haar/blocks n_ext 一致 | 136 | 136 | ±(0·\|t\|+0) | ✅ |
| B | symmetric/haar/blocks 末段误差落地板 | ≤ 1.00000e-12 | 1.776e-15 | +0 | ✅ |
| B | symmetric/haar/heavisine 往返误差一致 | 3.907e-16 | 3.907e-16 | ±(0·\|t\|+0) | ✅ |
| B | symmetric/haar/heavisine n_ext 一致 | 136 | 136 | ±(0·\|t\|+0) | ✅ |
| B | symmetric/haar/heavisine 末段误差落地板 | ≤ 1.00000e-12 | 1.332e-15 | +0 | ✅ |
| B | symmetric/db8/bumps 往返误差一致 | 4.000e-16 | 4.000e-16 | ±(0·\|t\|+0) | ✅ |
| B | symmetric/db8/bumps n_ext 一致 | 160 | 160 | ±(0·\|t\|+0) | ✅ |
| B | symmetric/db8/bumps 末段误差落地板 | ≤ 1.00000e-12 | 1.110e-15 | +0 | ✅ |
| B | symmetric/db8/blocks 往返误差一致 | 5.921e-16 | 5.921e-16 | ±(0·\|t\|+0) | ✅ |
| B | symmetric/db8/blocks n_ext 一致 | 160 | 160 | ±(0·\|t\|+0) | ✅ |
| B | symmetric/db8/blocks 末段误差落地板 | ≤ 1.00000e-12 | 3.109e-15 | +0 | ✅ |
| B | symmetric/db8/heavisine 往返误差一致 | 7.814e-16 | 7.814e-16 | ±(0·\|t\|+0) | ✅ |
| B | symmetric/db8/heavisine n_ext 一致 | 160 | 160 | ±(0·\|t\|+0) | ✅ |
| B | symmetric/db8/heavisine 末段误差落地板 | ≤ 1.00000e-12 | 1.776e-15 | +0 | ✅ |
| B | zero/haar/bumps 往返误差一致 | 4.000e-16 | 4.000e-16 | ±(0·\|t\|+0) | ✅ |
| B | zero/haar/bumps n_ext 一致 | 136 | 136 | ±(0·\|t\|+0) | ✅ |
| B | zero/haar/bumps 末段误差落地板 | ≤ 1.00000e-12 | 1.332e-15 | +0 | ✅ |
| B | zero/haar/blocks 往返误差一致 | 2.961e-16 | 2.961e-16 | ±(0·\|t\|+0) | ✅ |
| B | zero/haar/blocks n_ext 一致 | 136 | 136 | ±(0·\|t\|+0) | ✅ |
| B | zero/haar/blocks 末段误差落地板 | ≤ 1.00000e-12 | 1.776e-15 | +0 | ✅ |
| B | zero/haar/heavisine 往返误差一致 | 3.907e-16 | 3.907e-16 | ±(0·\|t\|+0) | ✅ |
| B | zero/haar/heavisine n_ext 一致 | 136 | 136 | ±(0·\|t\|+0) | ✅ |
| B | zero/haar/heavisine 末段误差落地板 | ≤ 1.00000e-12 | 1.332e-15 | +0 | ✅ |
| B | zero/db8/bumps 往返误差一致 | 4.000e-16 | 4.000e-16 | ±(0·\|t\|+0) | ✅ |
| B | zero/db8/bumps n_ext 一致 | 160 | 160 | ±(0·\|t\|+0) | ✅ |
| B | zero/db8/bumps 末段误差落地板 | ≤ 1.00000e-12 | 1.110e-15 | +0 | ✅ |
| B | zero/db8/blocks 往返误差一致 | 5.181e-16 | 5.181e-16 | ±(0·\|t\|+0) | ✅ |
| B | zero/db8/blocks n_ext 一致 | 160 | 160 | ±(0·\|t\|+0) | ✅ |
| B | zero/db8/blocks 末段误差落地板 | ≤ 1.00000e-12 | 3.109e-15 | +0 | ✅ |
| B | zero/db8/heavisine 往返误差一致 | 7.814e-16 | 7.814e-16 | ±(0·\|t\|+0) | ✅ |
| B | zero/db8/heavisine n_ext 一致 | 160 | 160 | ±(0·\|t\|+0) | ✅ |
| B | zero/db8/heavisine 末段误差落地板 | ≤ 1.00000e-12 | 1.776e-15 | +0 | ✅ |
| B | 换参数后 SWT 能量比仍与 numcore 一致 | 3.99955 | 3.99955 | ±(1e-12·\|t\|+0) | ✅ |
| B | haar 对常数项严格为 0（掩码点为地板） | 0 | 0 | ±(0·\|t\|+0) | ✅ |
| B | k=0 在 haar 下确实有 1 个地板点 | 1 | 1 | ±(0·\|t\|+0) | ✅ |
| B | 参数扫掠下的一致性 | 18 组分解树参数（3 模式 × 2 小波 × 3 信号）+ 2 组其它帧 | 18 | 0.5 | ✅ |
| B | haar k=0 J=1 内部比值（应贴 u） | ≤ 1.00000e-15 | 0 | +1.00000e-15 | ✅ |
| B | haar k=0 J=2 内部比值（应贴 u） | ≤ 1.00000e-15 | 0 | +1.00000e-15 | ✅ |
| B | haar k=0 J=3 内部比值（应贴 u） | ≤ 1.00000e-15 | 0 | +1.00000e-15 | ✅ |
| B | haar k=0 J=4 内部比值（应贴 u） | ≤ 1.00000e-15 | 0 | +1.00000e-15 | ✅ |
| A | haar k=N=1 的残差远高于地板（下界） | >= 100u（否则会被舍入噪声埋掉） | 0.00392157 | 1.110e-14 | ✅ |
| B | haar k=N 残差第 2 层 / 第 1 层 = 2^N | 2 | 2.01581 | ±(0.3·\|t\|+0) | ✅ |
| B | db2 k=0 J=1 内部比值（应贴 u） | ≤ 1.00000e-15 | 3.925e-17 | +1.00000e-15 | ✅ |
| B | db2 k=0 J=2 内部比值（应贴 u） | ≤ 1.00000e-15 | 3.925e-17 | +1.00000e-15 | ✅ |
| B | db2 k=0 J=3 内部比值（应贴 u） | ≤ 1.00000e-15 | 3.925e-17 | +1.00000e-15 | ✅ |
| B | db2 k=0 J=4 内部比值（应贴 u） | ≤ 1.00000e-15 | 3.925e-17 | +1.00000e-15 | ✅ |
| B | db2 k=1 J=1 内部比值（应贴 u） | ≤ 1.00000e-15 | 7.890e-17 | +1.00000e-15 | ✅ |
| B | db2 k=1 J=2 内部比值（应贴 u） | ≤ 1.00000e-15 | 1.127e-16 | +1.00000e-15 | ✅ |
| B | db2 k=1 J=3 内部比值（应贴 u） | ≤ 1.00000e-15 | 1.626e-16 | +1.00000e-15 | ✅ |
| B | db2 k=1 J=4 内部比值（应贴 u） | ≤ 1.00000e-15 | 1.626e-16 | +1.00000e-15 | ✅ |
| A | db2 k=N=2 的残差远高于地板（下界） | >= 100u（否则会被舍入噪声埋掉） | 5.339e-05 | 1.110e-14 | ✅ |
| B | db2 k=N 残差第 2 层 / 第 1 层 = 2^N | 4 | 4.08085 | ±(0.3·\|t\|+0) | ✅ |
| B | db3 k=0 J=1 内部比值（应贴 u） | ≤ 1.00000e-15 | 0 | +1.00000e-15 | ✅ |
| B | db3 k=0 J=2 内部比值（应贴 u） | ≤ 1.00000e-15 | 0 | +1.00000e-15 | ✅ |
| B | db3 k=0 J=3 内部比值（应贴 u） | ≤ 1.00000e-15 | 0 | +1.00000e-15 | ✅ |
| B | db3 k=0 J=4 内部比值（应贴 u） | ≤ 1.00000e-15 | 0 | +1.00000e-15 | ✅ |
| B | db3 k=1 J=1 内部比值（应贴 u） | ≤ 1.00000e-15 | 7.901e-17 | +1.00000e-15 | ✅ |
| B | db3 k=1 J=2 内部比值（应贴 u） | ≤ 1.00000e-15 | 1.698e-16 | +1.00000e-15 | ✅ |
| B | db3 k=1 J=3 内部比值（应贴 u） | ≤ 1.00000e-15 | 1.698e-16 | +1.00000e-15 | ✅ |
| B | db3 k=1 J=4 内部比值（应贴 u） | ≤ 1.00000e-15 | 1.698e-16 | +1.00000e-15 | ✅ |
| B | db3 k=2 J=1 内部比值（应贴 u） | ≤ 1.00000e-15 | 7.952e-17 | +1.00000e-15 | ✅ |
| B | db3 k=2 J=2 内部比值（应贴 u） | ≤ 1.00000e-15 | 1.154e-16 | +1.00000e-15 | ✅ |
| B | db3 k=2 J=3 内部比值（应贴 u） | ≤ 1.00000e-15 | 1.154e-16 | +1.00000e-15 | ✅ |
| B | db3 k=2 J=4 内部比值（应贴 u） | ≤ 1.00000e-15 | 1.154e-16 | +1.00000e-15 | ✅ |
| A | db3 k=N=3 的残差远高于地板（下界） | >= 100u（否则会被舍入噪声埋掉） | 1.153e-06 | 1.110e-14 | ✅ |
| B | db3 k=N 残差第 2 层 / 第 1 层 = 2^N | 8 | 8.31658 | ±(0.3·\|t\|+0) | ✅ |
| B | db4 k=0 J=1 内部比值（应贴 u） | ≤ 1.00000e-15 | 0 | +1.00000e-15 | ✅ |
| B | db4 k=0 J=2 内部比值（应贴 u） | ≤ 1.00000e-15 | 0 | +1.00000e-15 | ✅ |
| B | db4 k=0 J=3 内部比值（应贴 u） | ≤ 1.00000e-15 | 0 | +1.00000e-15 | ✅ |
| B | db4 k=0 J=4 内部比值（应贴 u） | ≤ 1.00000e-15 | 0 | +1.00000e-15 | ✅ |
| B | db4 k=1 J=1 内部比值（应贴 u） | ≤ 1.00000e-15 | 7.913e-17 | +1.00000e-15 | ✅ |
| B | db4 k=1 J=2 内部比值（应贴 u） | ≤ 1.00000e-15 | 1.137e-16 | +1.00000e-15 | ✅ |
| B | db4 k=1 J=3 内部比值（应贴 u） | ≤ 1.00000e-15 | 2.077e-16 | +1.00000e-15 | ✅ |
| B | db4 k=1 J=4 内部比值（应贴 u） | ≤ 1.00000e-15 | 2.077e-16 | +1.00000e-15 | ✅ |
| B | db4 k=2 J=1 内部比值（应贴 u） | ≤ 1.00000e-15 | 7.975e-17 | +1.00000e-15 | ✅ |
| B | db4 k=2 J=2 内部比值（应贴 u） | ≤ 1.00000e-15 | 7.975e-17 | +1.00000e-15 | ✅ |
| B | db4 k=2 J=3 内部比值（应贴 u） | ≤ 1.00000e-15 | 8.791e-17 | +1.00000e-15 | ✅ |
| B | db4 k=2 J=4 内部比值（应贴 u） | ≤ 1.00000e-15 | 8.791e-17 | +1.00000e-15 | ✅ |
| B | db4 k=3 J=1 内部比值（应贴 u） | ≤ 1.00000e-15 | 8.038e-17 | +1.00000e-15 | ✅ |
| B | db4 k=3 J=2 内部比值（应贴 u） | ≤ 1.00000e-15 | 8.038e-17 | +1.00000e-15 | ✅ |
| B | db4 k=3 J=3 内部比值（应贴 u） | ≤ 1.00000e-15 | 9.302e-17 | +1.00000e-15 | ✅ |
| B | db4 k=3 J=4 内部比值（应贴 u） | ≤ 1.00000e-15 | 9.302e-17 | +1.00000e-15 | ✅ |
| A | db4 k=N=4 的残差远高于地板（下界） | >= 100u（否则会被舍入噪声埋掉） | 3.412e-08 | 1.110e-14 | ✅ |
| B | db4 k=N 残差第 2 层 / 第 1 层 = 2^N | 16 | 17.0546 | ±(0.3·\|t\|+0) | ✅ |
| B | db5 k=0 J=1 内部比值（应贴 u） | ≤ 1.00000e-15 | 0 | +1.00000e-15 | ✅ |
| B | db5 k=0 J=2 内部比值（应贴 u） | ≤ 1.00000e-15 | 5.551e-17 | +1.00000e-15 | ✅ |
| B | db5 k=0 J=3 内部比值（应贴 u） | ≤ 1.00000e-15 | 5.551e-17 | +1.00000e-15 | ✅ |
| B | db5 k=0 J=4 内部比值（应贴 u） | ≤ 1.00000e-15 | 5.551e-17 | +1.00000e-15 | ✅ |
| B | db5 k=1 J=1 内部比值（应贴 u） | ≤ 1.00000e-15 | 7.924e-17 | +1.00000e-15 | ✅ |
| B | db5 k=1 J=2 内部比值（应贴 u） | ≤ 1.00000e-15 | 1.713e-16 | +1.00000e-15 | ✅ |
| B | db5 k=1 J=3 内部比值（应贴 u） | ≤ 1.00000e-15 | 1.713e-16 | +1.00000e-15 | ✅ |
| B | db5 k=1 J=4 内部比值（应贴 u） | ≤ 1.00000e-15 | 1.713e-16 | +1.00000e-15 | ✅ |
| B | db5 k=2 J=1 内部比值（应贴 u） | ≤ 1.00000e-15 | 7.999e-17 | +1.00000e-15 | ✅ |
| B | db5 k=2 J=2 内部比值（应贴 u） | ≤ 1.00000e-15 | 7.999e-17 | +1.00000e-15 | ✅ |
| B | db5 k=2 J=3 内部比值（应贴 u） | ≤ 1.00000e-15 | 8.986e-17 | +1.00000e-15 | ✅ |
| B | db5 k=2 J=4 内部比值（应贴 u） | ≤ 1.00000e-15 | 8.986e-17 | +1.00000e-15 | ✅ |
| B | db5 k=3 J=1 内部比值（应贴 u） | ≤ 1.00000e-15 | 8.074e-17 | +1.00000e-15 | ✅ |
| B | db5 k=3 J=2 内部比值（应贴 u） | ≤ 1.00000e-15 | 8.074e-17 | +1.00000e-15 | ✅ |
| B | db5 k=3 J=3 内部比值（应贴 u） | ≤ 1.00000e-15 | 8.074e-17 | +1.00000e-15 | ✅ |
| B | db5 k=3 J=4 内部比值（应贴 u） | ≤ 1.00000e-15 | 8.074e-17 | +1.00000e-15 | ✅ |
| B | db5 k=4 J=1 内部比值（应贴 u） | ≤ 1.00000e-15 | 4.075e-17 | +1.00000e-15 | ✅ |
| B | db5 k=4 J=2 内部比值（应贴 u） | ≤ 1.00000e-15 | 9.328e-17 | +1.00000e-15 | ✅ |
| B | db5 k=4 J=3 内部比值（应贴 u） | ≤ 1.00000e-15 | 9.328e-17 | +1.00000e-15 | ✅ |
| B | db5 k=4 J=4 内部比值（应贴 u） | ≤ 1.00000e-15 | 9.328e-17 | +1.00000e-15 | ✅ |
| A | db5 k=N=5 的残差远高于地板（下界） | >= 100u（否则会被舍入噪声埋掉） | 1.284e-09 | 1.110e-14 | ✅ |
| B | db5 k=N 残差第 2 层 / 第 1 层 = 2^N | 32 | 35.1896 | ±(0.3·\|t\|+0) | ✅ |
| B | db6 k=0 J=1 内部比值（应贴 u） | ≤ 1.00000e-15 | 7.850e-17 | +1.00000e-15 | ✅ |
| B | db6 k=0 J=2 内部比值（应贴 u） | ≤ 1.00000e-15 | 7.850e-17 | +1.00000e-15 | ✅ |
| B | db6 k=0 J=3 内部比值（应贴 u） | ≤ 1.00000e-15 | 7.850e-17 | +1.00000e-15 | ✅ |
| B | db6 k=0 J=4 内部比值（应贴 u） | ≤ 1.00000e-15 | 7.850e-17 | +1.00000e-15 | ✅ |
| B | db6 k=1 J=1 内部比值（应贴 u） | ≤ 1.00000e-15 | 7.936e-17 | +1.00000e-15 | ✅ |
| B | db6 k=1 J=2 内部比值（应贴 u） | ≤ 1.00000e-15 | 1.721e-16 | +1.00000e-15 | ✅ |
| B | db6 k=1 J=3 内部比值（应贴 u） | ≤ 1.00000e-15 | 1.721e-16 | +1.00000e-15 | ✅ |
| B | db6 k=1 J=4 内部比值（应贴 u） | ≤ 1.00000e-15 | 1.721e-16 | +1.00000e-15 | ✅ |
| B | db6 k=2 J=1 内部比值（应贴 u） | ≤ 1.00000e-15 | 8.023e-17 | +1.00000e-15 | ✅ |
| B | db6 k=2 J=2 内部比值（应贴 u） | ≤ 1.00000e-15 | 8.023e-17 | +1.00000e-15 | ✅ |
| B | db6 k=2 J=3 内部比值（应贴 u） | ≤ 1.00000e-15 | 8.023e-17 | +1.00000e-15 | ✅ |
| B | db6 k=2 J=4 内部比值（应贴 u） | ≤ 1.00000e-15 | 8.023e-17 | +1.00000e-15 | ✅ |
| B | db6 k=3 J=1 内部比值（应贴 u） | ≤ 1.00000e-15 | 8.110e-17 | +1.00000e-15 | ✅ |
| B | db6 k=3 J=2 内部比值（应贴 u） | ≤ 1.00000e-15 | 8.110e-17 | +1.00000e-15 | ✅ |
| B | db6 k=3 J=3 内部比值（应贴 u） | ≤ 1.00000e-15 | 8.110e-17 | +1.00000e-15 | ✅ |
| B | db6 k=3 J=4 内部比值（应贴 u） | ≤ 1.00000e-15 | 8.110e-17 | +1.00000e-15 | ✅ |
| B | db6 k=4 J=1 内部比值（应贴 u） | ≤ 1.00000e-15 | 4.099e-17 | +1.00000e-15 | ✅ |
| B | db6 k=4 J=2 内部比值（应贴 u） | ≤ 1.00000e-15 | 4.099e-17 | +1.00000e-15 | ✅ |
| B | db6 k=4 J=3 内部比值（应贴 u） | ≤ 1.00000e-15 | 4.099e-17 | +1.00000e-15 | ✅ |
| B | db6 k=4 J=4 内部比值（应贴 u） | ≤ 1.00000e-15 | 4.099e-17 | +1.00000e-15 | ✅ |
| B | db6 k=5 J=1 内部比值（应贴 u） | ≤ 1.00000e-15 | 8.288e-17 | +1.00000e-15 | ✅ |
| B | db6 k=5 J=2 内部比值（应贴 u） | ≤ 1.00000e-15 | 8.288e-17 | +1.00000e-15 | ✅ |
| B | db6 k=5 J=3 内部比值（应贴 u） | ≤ 1.00000e-15 | 8.288e-17 | +1.00000e-15 | ✅ |
| B | db6 k=5 J=4 内部比值（应贴 u） | ≤ 1.00000e-15 | 8.288e-17 | +1.00000e-15 | ✅ |
| A | db6 k=N=6 的残差远高于地板（下界） | >= 100u（否则会被舍入噪声埋掉） | 5.868e-11 | 1.110e-14 | ✅ |
| B | db6 k=N 残差第 2 层 / 第 1 层 = 2^N | 64 | 73.0574 | ±(0.3·\|t\|+0) | ✅ |
| B | db7 k=0 J=1 内部比值（应贴 u） | ≤ 1.00000e-15 | 3.925e-17 | +1.00000e-15 | ✅ |
| B | db7 k=0 J=2 内部比值（应贴 u） | ≤ 1.00000e-15 | 3.925e-17 | +1.00000e-15 | ✅ |
| B | db7 k=0 J=3 内部比值（应贴 u） | ≤ 1.00000e-15 | 3.925e-17 | +1.00000e-15 | ✅ |
| B | db7 k=0 J=4 内部比值（应贴 u） | ≤ 1.00000e-15 | 5.551e-17 | +1.00000e-15 | ✅ |
| B | db7 k=1 J=1 内部比值（应贴 u） | ≤ 1.00000e-15 | 5.961e-17 | +1.00000e-15 | ✅ |
| B | db7 k=1 J=2 内部比值（应贴 u） | ≤ 1.00000e-15 | 5.961e-17 | +1.00000e-15 | ✅ |
| B | db7 k=1 J=3 内部比值（应贴 u） | ≤ 1.00000e-15 | 5.961e-17 | +1.00000e-15 | ✅ |
| B | db7 k=1 J=4 内部比值（应贴 u） | ≤ 1.00000e-15 | 5.961e-17 | +1.00000e-15 | ✅ |
| B | db7 k=2 J=1 内部比值（应贴 u） | ≤ 1.00000e-15 | 6.035e-17 | +1.00000e-15 | ✅ |
| B | db7 k=2 J=2 内部比值（应贴 u） | ≤ 1.00000e-15 | 1.197e-16 | +1.00000e-15 | ✅ |
| B | db7 k=2 J=3 内部比值（应贴 u） | ≤ 1.00000e-15 | 1.197e-16 | +1.00000e-15 | ✅ |
| B | db7 k=2 J=4 内部比值（应贴 u） | ≤ 1.00000e-15 | 1.197e-16 | +1.00000e-15 | ✅ |
| B | db7 k=3 J=1 内部比值（应贴 u） | ≤ 1.00000e-15 | 4.073e-17 | +1.00000e-15 | ✅ |
| B | db7 k=3 J=2 内部比值（应贴 u） | ≤ 1.00000e-15 | 4.073e-17 | +1.00000e-15 | ✅ |
| B | db7 k=3 J=3 内部比值（应贴 u） | ≤ 1.00000e-15 | 4.496e-17 | +1.00000e-15 | ✅ |
| B | db7 k=3 J=4 内部比值（应贴 u） | ≤ 1.00000e-15 | 4.496e-17 | +1.00000e-15 | ✅ |
| B | db7 k=4 J=1 内部比值（应贴 u） | ≤ 1.00000e-15 | 4.124e-17 | +1.00000e-15 | ✅ |
| B | db7 k=4 J=2 内部比值（应贴 u） | ≤ 1.00000e-15 | 4.124e-17 | +1.00000e-15 | ✅ |
| B | db7 k=4 J=3 内部比值（应贴 u） | ≤ 1.00000e-15 | 4.124e-17 | +1.00000e-15 | ✅ |
| B | db7 k=4 J=4 内部比值（应贴 u） | ≤ 1.00000e-15 | 4.124e-17 | +1.00000e-15 | ✅ |
| B | db7 k=5 J=1 内部比值（应贴 u） | ≤ 1.00000e-15 | 4.175e-17 | +1.00000e-15 | ✅ |
| B | db7 k=5 J=2 内部比值（应贴 u） | ≤ 1.00000e-15 | 5.021e-17 | +1.00000e-15 | ✅ |
| B | db7 k=5 J=3 内部比值（应贴 u） | ≤ 1.00000e-15 | 5.021e-17 | +1.00000e-15 | ✅ |
| B | db7 k=5 J=4 内部比值（应贴 u） | ≤ 1.00000e-15 | 5.021e-17 | +1.00000e-15 | ✅ |
| B | db7 k=6 J=1 内部比值（应贴 u） | ≤ 1.00000e-15 | 4.227e-17 | +1.00000e-15 | ✅ |
| B | db7 k=6 J=2 内部比值（应贴 u） | ≤ 1.00000e-15 | 4.227e-17 | +1.00000e-15 | ✅ |
| B | db7 k=6 J=3 内部比值（应贴 u） | ≤ 1.00000e-15 | 4.227e-17 | +1.00000e-15 | ✅ |
| B | db7 k=6 J=4 内部比值（应贴 u） | ≤ 1.00000e-15 | 4.227e-17 | +1.00000e-15 | ✅ |
| A | db7 k=N=7 的残差远高于地板（下界） | >= 100u（否则会被舍入噪声埋掉） | 3.159e-12 | 1.110e-14 | ✅ |
| B | db7 k=N 残差第 2 层 / 第 1 层 = 2^N | 128 | 152.612 | ±(0.3·\|t\|+0) | ✅ |
| B | db8 k=0 J=1 内部比值（应贴 u） | ≤ 1.00000e-15 | 2.944e-17 | +1.00000e-15 | ✅ |
| B | db8 k=0 J=2 内部比值（应贴 u） | ≤ 1.00000e-15 | 1.006e-16 | +1.00000e-15 | ✅ |
| B | db8 k=0 J=3 内部比值（应贴 u） | ≤ 1.00000e-15 | 1.006e-16 | +1.00000e-15 | ✅ |
| B | db8 k=0 J=4 内部比值（应贴 u） | ≤ 1.00000e-15 | 1.006e-16 | +1.00000e-15 | ✅ |
| B | db8 k=1 J=1 内部比值（应贴 u） | ≤ 1.00000e-15 | 1.393e-16 | +1.00000e-15 | ✅ |
| B | db8 k=1 J=2 内部比值（应贴 u） | ≤ 1.00000e-15 | 1.393e-16 | +1.00000e-15 | ✅ |
| B | db8 k=1 J=3 内部比值（应贴 u） | ≤ 1.00000e-15 | 1.393e-16 | +1.00000e-15 | ✅ |
| B | db8 k=1 J=4 内部比值（应贴 u） | ≤ 1.00000e-15 | 1.393e-16 | +1.00000e-15 | ✅ |
| B | db8 k=2 J=1 内部比值（应贴 u） | ≤ 1.00000e-15 | 9.079e-17 | +1.00000e-15 | ✅ |
| B | db8 k=2 J=2 内部比值（应贴 u） | ≤ 1.00000e-15 | 1.019e-16 | +1.00000e-15 | ✅ |
| B | db8 k=2 J=3 内部比值（应贴 u） | ≤ 1.00000e-15 | 1.019e-16 | +1.00000e-15 | ✅ |
| B | db8 k=2 J=4 内部比值（应贴 u） | ≤ 1.00000e-15 | 1.019e-16 | +1.00000e-15 | ✅ |
| B | db8 k=3 J=1 内部比值（应贴 u） | ≤ 1.00000e-15 | 7.160e-17 | +1.00000e-15 | ✅ |
| B | db8 k=3 J=2 内部比值（应贴 u） | ≤ 1.00000e-15 | 1.181e-16 | +1.00000e-15 | ✅ |
| B | db8 k=3 J=3 内部比值（应贴 u） | ≤ 1.00000e-15 | 1.181e-16 | +1.00000e-15 | ✅ |
| B | db8 k=3 J=4 内部比值（应贴 u） | ≤ 1.00000e-15 | 1.181e-16 | +1.00000e-15 | ✅ |
| B | db8 k=4 J=1 内部比值（应贴 u） | ≤ 1.00000e-15 | 5.056e-17 | +1.00000e-15 | ✅ |
| B | db8 k=4 J=2 内部比值（应贴 u） | ≤ 1.00000e-15 | 5.056e-17 | +1.00000e-15 | ✅ |
| B | db8 k=4 J=3 内部比值（应贴 u） | ≤ 1.00000e-15 | 5.056e-17 | +1.00000e-15 | ✅ |
| B | db8 k=4 J=4 内部比值（应贴 u） | ≤ 1.00000e-15 | 5.056e-17 | +1.00000e-15 | ✅ |
| B | db8 k=5 J=1 内部比值（应贴 u） | ≤ 1.00000e-15 | 6.835e-17 | +1.00000e-15 | ✅ |
| B | db8 k=5 J=2 内部比值（应贴 u） | ≤ 1.00000e-15 | 6.850e-17 | +1.00000e-15 | ✅ |
| B | db8 k=5 J=3 内部比值（应贴 u） | ≤ 1.00000e-15 | 6.850e-17 | +1.00000e-15 | ✅ |
| B | db8 k=5 J=4 内部比值（应贴 u） | ≤ 1.00000e-15 | 6.850e-17 | +1.00000e-15 | ✅ |
| B | db8 k=6 J=1 内部比值（应贴 u） | ≤ 1.00000e-15 | 4.798e-17 | +1.00000e-15 | ✅ |
| B | db8 k=6 J=2 内部比值（应贴 u） | ≤ 1.00000e-15 | 4.798e-17 | +1.00000e-15 | ✅ |
| B | db8 k=6 J=3 内部比值（应贴 u） | ≤ 1.00000e-15 | 4.798e-17 | +1.00000e-15 | ✅ |
| B | db8 k=6 J=4 内部比值（应贴 u） | ≤ 1.00000e-15 | 4.798e-17 | +1.00000e-15 | ✅ |
| B | db8 k=7 J=1 内部比值（应贴 u） | ≤ 1.00000e-15 | 5.945e-17 | +1.00000e-15 | ✅ |
| B | db8 k=7 J=2 内部比值（应贴 u） | ≤ 1.00000e-15 | 5.945e-17 | +1.00000e-15 | ✅ |
| B | db8 k=7 J=3 内部比值（应贴 u） | ≤ 1.00000e-15 | 5.945e-17 | +1.00000e-15 | ✅ |
| B | db8 k=7 J=4 内部比值（应贴 u） | ≤ 1.00000e-15 | 5.945e-17 | +1.00000e-15 | ✅ |
| A | db8 k=N=8 的残差远高于地板（下界） | >= 100u（否则会被舍入噪声埋掉） | 1.958e-13 | 1.110e-14 | ✅ |
| B | db8 k=N 残差第 2 层 / 第 1 层 = 2^N | 256 | 320.775 | ±(0.3·\|t\|+0) | ✅ |
| B | db4 k=4 残差随 n 按 n^N 衰减 | 256 | 281.808 | ±(0.2·\|t\|+0) | ✅ |
| A | db4 被抹平的比值与 u 之比 | 0.0 <= ratio/u <= 1.0（是地板级，不是某个小数） | 0.837822 | 1 | ✅ |
| B | haar 逐层内部比值（k=N-1） | ≤ 1.00000e-15 | 0 | +1.00000e-15 | ✅ |
| B | haar 有内部系数可测的层数 | >= 1（否则这一条无从验证） | 4 | 1 | ✅ |
| B | haar 逐层内部比值（k=N）显著高于地板（下界） | >= 100u | 0.00392157 | 1.110e-14 | ✅ |
| B | db4 逐层内部比值（k=N-1） | ≤ 1.00000e-15 | 9.302e-17 | +1.00000e-15 | ✅ |
| B | db4 有内部系数可测的层数 | >= 1（否则这一条无从验证） | 4 | 1 | ✅ |
| B | db4 逐层内部比值（k=N）显著高于地板（下界） | >= 100u | 3.412e-08 | 1.110e-14 | ✅ |
| B | db8 逐层内部比值（k=N-1） | ≤ 1.00000e-15 | 5.945e-17 | +1.00000e-15 | ✅ |
| B | db8 有内部系数可测的层数 | >= 1（否则这一条无从验证） | 3 | 1 | ✅ |
| B | db8 逐层内部比值（k=N）显著高于地板（下界） | >= 100u | 1.958e-13 | 1.110e-14 | ✅ |
| B | 边界系数（同一条多项式）实测 O(1)（下界） | >= 1e-2 | 0.953551 | 0.01 | ✅ |
| B | MSE = bias^2 + var（逐点恒等式） | ≤ 1.00000e-15 | 2.776e-17 | +1.00000e-15 | ✅ |
| B | 软阈值 偏差^2 随 lambda 单调不减 | nondecreasing | 违反 0 处 | 5.89741e-15 | ✅ |
| B | 软阈值 方差随 lambda 单调不增 | nonincreasing | 违反 0 处 | 2.54332e-15 | ✅ |
| B | 抛物线细化的最优点不差于栅格最优点 | ≤ 0.0468315 | 0.0468315 | +0 | ✅ |
| A | 硬阈值 偏差^2 的非单调段 | >= 1 段（趋势对、逐点不对） | 2 | 1 | ✅ |
| B | 最优点不差于不去噪（lambda=0） | ≤ 0.0896338 | 0.0473287 | +0 | ✅ |
| B | 实测最优阈值 <= 通用阈值 | ≤ 0.999066 | 0.280955 | +0 | ✅ |
| A | 通用阈值处的 MSE（与不去噪对照） | 偏保守：可高于不去噪 | 0.129939 | 0.0896338 | ✅ |
| B | lambda=0.3 处软阈值 MSE <= 硬阈值 | ≤ 0.0900402 | 0.0411659 | +0 | ✅ |
| B | 各自最优点上软阈值更优 | ≤ 0.0537851 | 0.0473287 | +0 | ✅ |
| B | 软/硬最优阈值之比 | 硬阈值的最优 lambda 更大（不连续、方差大） | 2.72182 | 0.5 | ✅ |
| B | haar/J=1 lambda=0 恒等 | ≤ 7.10543e-15 | 4.441e-16 | +7.10543e-15 | ✅ |
| B | haar/J=3 lambda=0 恒等 | ≤ 2.13163e-14 | 8.882e-16 | +2.13163e-14 | ✅ |
| B | haar/J=4 lambda=0 恒等 | ≤ 2.84217e-14 | 1.110e-15 | +2.84217e-14 | ✅ |
| B | db4/J=1 lambda=0 恒等 | ≤ 7.10543e-15 | 6.661e-16 | +7.10543e-15 | ✅ |
| B | db4/J=3 lambda=0 恒等 | ≤ 2.13163e-14 | 4.441e-16 | +2.13163e-14 | ✅ |
| B | db4/J=4 lambda=0 恒等 | ≤ 2.84217e-14 | 1.110e-15 | +2.84217e-14 | ✅ |
| B | 通用阈值随 sigma_hat 线性缩放 | 1.06669 | 1.06669 | ±(1e-12·\|t\|+0) | ✅ |
| B | 单次 MAD 读数在真值 30% 以内 | ≤ 0.3 | 0.066693 | +0.3 | ✅ |
| B | 阈值只作用于细节系数（最粗近似原样保留） | 0 | 0 | ±(0·\|t\|+0) | ✅ |
| B | SNR 增益（单次实现） | 去噪后 SNR 提升 | 2.0324 | 0 | ✅ |
| B | w=16 峰值定位误差 <= 半格 | ≤ 0.03125 | 0.0195312 | +0.03125 | ✅ |
| B | w=32 峰值定位误差 <= 半格 | ≤ 0.015625 | 0.0117188 | +0.015625 | ✅ |
| B | w=64 峰值定位误差 <= 半格 | ≤ 0.0078125 | 0.00390625 | +0.0078125 | ✅ |
| B | 峰值定位误差随窗长单调下降 | decreasing | 违反 0 处 | 5.20417e-17 | ✅ |
| A | 低频音在 w=32 时的峰值误差（反面） | > 1/(2w) = 0.0156（主瓣把峰吞了） | 0.0195312 | 0.015625 | ✅ |
| B | 第 1 层时间支撑 = L-1+1 = 8 个样本 | 8 | 8 | ±(0·\|t\|+0) | ✅ |
| B | 小波时间支撑随层数增长 | increasing | 违反 0 处 | 2.02505e-13 | ✅ |
| B | 分辨率元面积极差（近似守恒） | ≤ 1.2 | 1.12281 | +0 | ✅ |
| B | STFT 时间窗宽恒定 | 0 | 0 | ±(0·\|t\|+0) | ✅ |
| B | STFT 带宽恒定 | 0 | 0 | ±(0·\|t\|+0) | ✅ |
| B | 两种分辨率元的面积对照 | 小波近似守恒、STFT 恒定 | 1.12281 | 0.2 | ✅ |
| B | bumps 两次调用逐位相同 | 0 | 0 | ±(0·\|t\|+0) | ✅ |
| B | bumps 全部有限 | ≤ 0 | 0 | +0 | ✅ |
| B | bumps/haar 往返误差 | ≤ 1.00000e-13 | 1.776e-15 | +1.00000e-13 | ✅ |
| B | bumps/db4 往返误差 | ≤ 1.00000e-13 | 1.332e-15 | +1.00000e-13 | ✅ |
| B | bumps/db8 往返误差 | ≤ 1.00000e-13 | 1.332e-15 | +1.00000e-13 | ✅ |
| B | bumps SWT 往返误差 | ≤ 1.00000e-13 | 8.882e-16 | +1.00000e-13 | ✅ |
| B | blocks 两次调用逐位相同 | 0 | 0 | ±(0·\|t\|+0) | ✅ |
| B | blocks 全部有限 | ≤ 0 | 0 | +0 | ✅ |
| B | blocks/haar 往返误差 | ≤ 1.00000e-13 | 2.665e-15 | +1.00000e-13 | ✅ |
| B | blocks/db4 往返误差 | ≤ 1.00000e-13 | 5.329e-15 | +1.00000e-13 | ✅ |
| B | blocks/db8 往返误差 | ≤ 1.00000e-13 | 5.329e-15 | +1.00000e-13 | ✅ |
| B | blocks SWT 往返误差 | ≤ 1.00000e-13 | 2.665e-15 | +1.00000e-13 | ✅ |
| B | heavisine 两次调用逐位相同 | 0 | 0 | ±(0·\|t\|+0) | ✅ |
| B | heavisine 全部有限 | ≤ 0 | 0 | +0 | ✅ |
| B | heavisine/haar 往返误差 | ≤ 1.00000e-13 | 2.220e-15 | +1.00000e-13 | ✅ |
| B | heavisine/db4 往返误差 | ≤ 1.00000e-13 | 1.776e-15 | +1.00000e-13 | ✅ |
| B | heavisine/db8 往返误差 | ≤ 1.00000e-13 | 2.220e-15 | +1.00000e-13 | ✅ |
| B | heavisine SWT 往返误差 | ≤ 1.00000e-13 | 1.110e-15 | +1.00000e-13 | ✅ |
| B | doppler 两次调用逐位相同 | 0 | 0 | ±(0·\|t\|+0) | ✅ |
| B | doppler 全部有限 | ≤ 0 | 0 | +0 | ✅ |
| B | doppler/haar 往返误差 | ≤ 1.00000e-13 | 3.331e-16 | +1.00000e-13 | ✅ |
| B | doppler/db4 往返误差 | ≤ 1.00000e-13 | 3.331e-16 | +1.00000e-13 | ✅ |
| B | doppler/db8 往返误差 | ≤ 1.00000e-13 | 3.886e-16 | +1.00000e-13 | ✅ |
| B | doppler SWT 往返误差 | ≤ 1.00000e-13 | 2.220e-16 | +1.00000e-13 | ✅ |
| B | tone 两次调用逐位相同 | 0 | 0 | ±(0·\|t\|+0) | ✅ |
| B | tone 全部有限 | ≤ 0 | 0 | +0 | ✅ |
| B | tone/haar 往返误差 | ≤ 1.00000e-13 | 1.110e-15 | +1.00000e-13 | ✅ |
| B | tone/db4 往返误差 | ≤ 1.00000e-13 | 1.110e-15 | +1.00000e-13 | ✅ |
| B | tone/db8 往返误差 | ≤ 1.00000e-13 | 1.110e-15 | +1.00000e-13 | ✅ |
| B | tone SWT 往返误差 | ≤ 1.00000e-13 | 6.661e-16 | +1.00000e-13 | ✅ |
| B | chirp 两次调用逐位相同 | 0 | 0 | ±(0·\|t\|+0) | ✅ |
| B | chirp 全部有限 | ≤ 0 | 0 | +0 | ✅ |
| B | chirp/haar 往返误差 | ≤ 1.00000e-13 | 8.882e-16 | +1.00000e-13 | ✅ |
| B | chirp/db4 往返误差 | ≤ 1.00000e-13 | 6.661e-16 | +1.00000e-13 | ✅ |
| B | chirp/db8 往返误差 | ≤ 1.00000e-13 | 8.882e-16 | +1.00000e-13 | ✅ |
| B | chirp SWT 往返误差 | ≤ 1.00000e-13 | 4.441e-16 | +1.00000e-13 | ✅ |
| B | 六个基准信号的取值范围 | 有界（避免量级混进来） | 6 | 8 | ✅ |
| B | 各层细节/近似量级比有界 | ≤ 10 | 1.01015 | +0 | ✅ |
| B | 用原信号做分母 vs 用同层近似（层 4） | 同层口径更大（不受 2^{-j} 尺度衰减影响） | 1.02468 | 0 | ✅ |
| B | haar 正交性缺陷 | ≤ 8.88178e-16 | 2.220e-16 | +8.88178e-16 | ✅ |
| B | haar 归一化 \|sum h - sqrt2\|/sqrt2 | ≤ 8.88178e-16 | 0 | +8.88178e-16 | ✅ |
| B | haar 完美重构条件 | ≤ 3.55271e-15 | 1.332e-15 | +3.55271e-15 | ✅ |
| B | haar 前 1 阶消失矩（相对缺陷） | ≤ 1.00000e-14 | 0 | +1.00000e-14 | ✅ |
| B | db2 正交性缺陷 | ≤ 8.88178e-16 | 1.388e-17 | +8.88178e-16 | ✅ |
| B | db2 归一化 \|sum h - sqrt2\|/sqrt2 | ≤ 8.88178e-16 | 0 | +8.88178e-16 | ✅ |
| B | db2 完美重构条件 | ≤ 3.55271e-15 | 1.554e-15 | +3.55271e-15 | ✅ |
| B | db2 前 2 阶消失矩（相对缺陷） | ≤ 1.00000e-14 | 3.318e-17 | +1.00000e-14 | ✅ |
| B | db3 正交性缺陷 | ≤ 8.88178e-16 | 2.082e-17 | +8.88178e-16 | ✅ |
| B | db3 归一化 \|sum h - sqrt2\|/sqrt2 | ≤ 8.88178e-16 | 1.570e-16 | +8.88178e-16 | ✅ |
| B | db3 完美重构条件 | ≤ 3.55271e-15 | 1.998e-15 | +3.55271e-15 | ✅ |
| B | db3 前 3 阶消失矩（相对缺陷） | ≤ 1.00000e-14 | 1.817e-17 | +1.00000e-14 | ✅ |
| B | db4 正交性缺陷 | ≤ 8.88178e-16 | 1.388e-17 | +8.88178e-16 | ✅ |
| B | db4 归一化 \|sum h - sqrt2\|/sqrt2 | ≤ 8.88178e-16 | 0 | +8.88178e-16 | ✅ |
| B | db4 完美重构条件 | ≤ 3.55271e-15 | 1.776e-15 | +3.55271e-15 | ✅ |
| B | db4 前 4 阶消失矩（相对缺陷） | ≤ 1.00000e-14 | 2.685e-17 | +1.00000e-14 | ✅ |
| B | db5 正交性缺陷 | ≤ 8.88178e-16 | 2.776e-17 | +8.88178e-16 | ✅ |
| B | db5 归一化 \|sum h - sqrt2\|/sqrt2 | ≤ 8.88178e-16 | 0 | +8.88178e-16 | ✅ |
| B | db5 完美重构条件 | ≤ 3.55271e-15 | 2.220e-15 | +3.55271e-15 | ✅ |
| B | db5 前 5 阶消失矩（相对缺陷） | ≤ 1.00000e-14 | 3.865e-17 | +1.00000e-14 | ✅ |
| B | db6 正交性缺陷 | ≤ 8.88178e-16 | 2.776e-17 | +8.88178e-16 | ✅ |
| B | db6 归一化 \|sum h - sqrt2\|/sqrt2 | ≤ 8.88178e-16 | 1.570e-16 | +8.88178e-16 | ✅ |
| B | db6 完美重构条件 | ≤ 3.55271e-15 | 3.553e-15 | +3.55271e-15 | ✅ |
| B | db6 前 6 阶消失矩（相对缺陷） | ≤ 1.00000e-14 | 3.588e-17 | +1.00000e-14 | ✅ |
| B | db7 正交性缺陷 | ≤ 8.88178e-16 | 1.110e-16 | +8.88178e-16 | ✅ |
| B | db7 归一化 \|sum h - sqrt2\|/sqrt2 | ≤ 8.88178e-16 | 0 | +8.88178e-16 | ✅ |
| B | db7 完美重构条件 | ≤ 3.55271e-15 | 4.441e-15 | +3.55271e-15 | ✅ |
| B | db7 前 7 阶消失矩（相对缺陷） | ≤ 1.00000e-14 | 4.827e-17 | +1.00000e-14 | ✅ |
| B | db8 正交性缺陷 | ≤ 8.88178e-16 | 1.110e-16 | +8.88178e-16 | ✅ |
| B | db8 归一化 \|sum h - sqrt2\|/sqrt2 | ≤ 8.88178e-16 | 0 | +8.88178e-16 | ✅ |
| B | db8 完美重构条件 | ≤ 3.55271e-15 | 4.219e-15 | +3.55271e-15 | ✅ |
| B | db8 前 8 阶消失矩（相对缺陷） | ≤ 1.00000e-14 | 8.873e-17 | +1.00000e-14 | ✅ |
| A | 最坏正交性缺陷 | <= 8u | 2.220e-16 | 8.882e-16 | ✅ |
| A | m = N 处消失矩不再成立（下界） | >= 1e-6（远高于 u=2.2e-16） | 0.00178094 | 1.000e-06 | ✅ |
| B | db2 J=1 周期模式系数个数 = n | 64 | 64 | ±(0·\|t\|+0) | ✅ |
| B | db2 J=1 正交性缺陷 | ≤ 7.10543e-15 | 2.487e-17 | +7.10543e-15 | ✅ |
| B | db2 J=2 周期模式系数个数 = n | 64 | 64 | ±(0·\|t\|+0) | ✅ |
| B | db2 J=2 正交性缺陷 | ≤ 7.10543e-15 | 2.220e-16 | +7.10543e-15 | ✅ |
| B | db2 J=4 周期模式系数个数 = n | 64 | 64 | ±(0·\|t\|+0) | ✅ |
| B | db2 J=4 正交性缺陷 | ≤ 7.10543e-15 | 2.220e-16 | +7.10543e-15 | ✅ |
| B | db4 J=1 周期模式系数个数 = n | 64 | 64 | ±(0·\|t\|+0) | ✅ |
| B | db4 J=1 正交性缺陷 | ≤ 7.10543e-15 | 2.220e-16 | +7.10543e-15 | ✅ |
| B | db4 J=2 周期模式系数个数 = n | 64 | 64 | ±(0·\|t\|+0) | ✅ |
| B | db4 J=2 正交性缺陷 | ≤ 7.10543e-15 | 4.441e-16 | +7.10543e-15 | ✅ |
| B | db4 J=4 周期模式系数个数 = n | 64 | 64 | ±(0·\|t\|+0) | ✅ |
| B | db4 J=4 正交性缺陷 | ≤ 7.10543e-15 | 4.441e-16 | +7.10543e-15 | ✅ |
| B | db8 J=1 周期模式系数个数 = n | 64 | 64 | ±(0·\|t\|+0) | ✅ |
| B | db8 J=1 正交性缺陷 | ≤ 7.10543e-15 | 2.220e-16 | +7.10543e-15 | ✅ |
| B | db8 J=2 周期模式系数个数 = n | 64 | 64 | ±(0·\|t\|+0) | ✅ |
| B | db8 J=2 正交性缺陷 | ≤ 7.10543e-15 | 4.441e-16 | +7.10543e-15 | ✅ |
| B | db8 J=4 周期模式系数个数 = n | 64 | 64 | ±(0·\|t\|+0) | ✅ |
| B | db8 J=4 正交性缺陷 | ≤ 7.10543e-15 | 8.882e-16 | +7.10543e-15 | ✅ |
| B | zero 模式 M^T M = I（等距嵌入） | ≤ 7.10543e-15 | 2.220e-16 | +7.10543e-15 | ✅ |
| A | zero 模式 M M^T != I（冗余） | >= 0.5（远大于 u） | 1 | 0.5 | ✅ |
| A | symmetric 模式 M^T M != I（反射复制样本） | >= 0.5 | 1 | 0.5 | ✅ |
| B | n=16 periodic 递归 vs 显式矩阵 | ≤ 1.13687e-13 | 4.797e-17 | +1.13687e-13 | ✅ |
| B | n=16 symmetric 递归 vs 显式矩阵 | ≤ 1.13687e-13 | 1.892e-16 | +1.13687e-13 | ✅ |
| B | n=16 zero 递归 vs 显式矩阵 | ≤ 1.13687e-13 | 0 | +1.13687e-13 | ✅ |
| B | n=32 periodic 递归 vs 显式矩阵 | ≤ 2.27374e-13 | 2.635e-17 | +2.27374e-13 | ✅ |
| B | n=32 symmetric 递归 vs 显式矩阵 | ≤ 2.27374e-13 | 1.054e-16 | +2.27374e-13 | ✅ |
| B | n=32 zero 递归 vs 显式矩阵 | ≤ 2.27374e-13 | 0 | +2.27374e-13 | ✅ |
| B | n=64 periodic 递归 vs 显式矩阵 | ≤ 4.54747e-13 | 8.405e-17 | +4.54747e-13 | ✅ |
| B | n=64 symmetric 递归 vs 显式矩阵 | ≤ 4.54747e-13 | 8.405e-17 | +4.54747e-13 | ✅ |
| B | n=64 zero 递归 vs 显式矩阵 | ≤ 4.54747e-13 | 0 | +4.54747e-13 | ✅ |
| A | Haar 闭式解 | a_k=(x_2k+x_2k+1)/sqrt2，逐位 | 4.441e-16 | 8.882e-16 | ✅ |
| B | periodic/haar/J=1 往返误差 | ≤ 4.44089e-16 | 2.808e-16 | +4.44089e-16 | ✅ |
| B | periodic/haar/J=2 往返误差 | ≤ 8.88178e-16 | 5.411e-16 | +8.88178e-16 | ✅ |
| B | periodic/haar/J=3 往返误差 | ≤ 1.33227e-15 | 7.021e-16 | +1.33227e-15 | ✅ |
| B | periodic/haar/J=4 往返误差 | ≤ 1.77636e-15 | 8.425e-16 | +1.77636e-15 | ✅ |
| B | periodic/haar/J=6 往返误差 | ≤ 2.66454e-15 | 8.425e-16 | +2.66454e-15 | ✅ |
| B | periodic/haar/J=8 往返误差 | ≤ 3.55271e-15 | 8.425e-16 | +3.55271e-15 | ✅ |
| B | periodic/db2/J=1 往返误差 | ≤ 4.44089e-16 | 4.058e-16 | +4.44089e-16 | ✅ |
| B | periodic/db2/J=2 往返误差 | ≤ 8.88178e-16 | 4.058e-16 | +8.88178e-16 | ✅ |
| B | periodic/db2/J=3 往返误差 | ≤ 1.33227e-15 | 5.411e-16 | +1.33227e-15 | ✅ |
| B | periodic/db2/J=4 往返误差 | ≤ 1.77636e-15 | 5.617e-16 | +1.77636e-15 | ✅ |
| B | periodic/db2/J=6 往返误差 | ≤ 2.66454e-15 | 5.617e-16 | +2.66454e-15 | ✅ |
| B | periodic/db2/J=8 往返误差 | ≤ 3.55271e-15 | 5.617e-16 | +3.55271e-15 | ✅ |
| B | periodic/db4/J=1 往返误差 | ≤ 4.44089e-16 | 2.808e-16 | +4.44089e-16 | ✅ |
| B | periodic/db4/J=2 往返误差 | ≤ 8.88178e-16 | 4.213e-16 | +8.88178e-16 | ✅ |
| B | periodic/db4/J=3 往返误差 | ≤ 1.33227e-15 | 7.021e-16 | +1.33227e-15 | ✅ |
| B | periodic/db4/J=4 往返误差 | ≤ 1.77636e-15 | 7.021e-16 | +1.77636e-15 | ✅ |
| B | periodic/db4/J=6 往返误差 | ≤ 2.66454e-15 | 7.021e-16 | +2.66454e-15 | ✅ |
| B | periodic/db4/J=8 往返误差 | ≤ 3.55271e-15 | 7.021e-16 | +3.55271e-15 | ✅ |
| B | periodic/db8/J=1 往返误差 | ≤ 4.44089e-16 | 4.213e-16 | +4.44089e-16 | ✅ |
| B | periodic/db8/J=2 往返误差 | ≤ 8.88178e-16 | 6.764e-16 | +8.88178e-16 | ✅ |
| B | periodic/db8/J=3 往返误差 | ≤ 1.33227e-15 | 8.425e-16 | +1.33227e-15 | ✅ |
| B | periodic/db8/J=4 往返误差 | ≤ 1.77636e-15 | 8.425e-16 | +1.77636e-15 | ✅ |
| B | periodic/db8/J=6 往返误差 | ≤ 2.66454e-15 | 8.425e-16 | +2.66454e-15 | ✅ |
| B | periodic/db8/J=8 往返误差 | ≤ 3.55271e-15 | 8.425e-16 | +3.55271e-15 | ✅ |
| B | symmetric/haar/J=1 往返误差 | ≤ 4.44089e-16 | 2.808e-16 | +4.44089e-16 | ✅ |
| B | symmetric/haar/J=2 往返误差 | ≤ 8.88178e-16 | 5.411e-16 | +8.88178e-16 | ✅ |
| B | symmetric/haar/J=3 往返误差 | ≤ 1.33227e-15 | 7.021e-16 | +1.33227e-15 | ✅ |
| B | symmetric/haar/J=4 往返误差 | ≤ 1.77636e-15 | 8.425e-16 | +1.77636e-15 | ✅ |
| B | symmetric/haar/J=6 往返误差 | ≤ 2.66454e-15 | 8.425e-16 | +2.66454e-15 | ✅ |
| B | symmetric/haar/J=8 往返误差 | ≤ 3.55271e-15 | 8.425e-16 | +3.55271e-15 | ✅ |
| B | symmetric/db2/J=1 往返误差 | ≤ 4.44089e-16 | 4.058e-16 | +4.44089e-16 | ✅ |
| B | symmetric/db2/J=2 往返误差 | ≤ 8.88178e-16 | 4.058e-16 | +8.88178e-16 | ✅ |
| B | symmetric/db2/J=3 往返误差 | ≤ 1.33227e-15 | 5.411e-16 | +1.33227e-15 | ✅ |
| B | symmetric/db2/J=4 往返误差 | ≤ 1.77636e-15 | 6.087e-16 | +1.77636e-15 | ✅ |
| B | symmetric/db2/J=6 往返误差 | ≤ 2.66454e-15 | 5.617e-16 | +2.66454e-15 | ✅ |
| B | symmetric/db2/J=8 往返误差 | ≤ 3.55271e-15 | 6.764e-16 | +3.55271e-15 | ✅ |
| B | symmetric/db4/J=1 往返误差 | ≤ 4.44089e-16 | 2.808e-16 | +4.44089e-16 | ✅ |
| B | symmetric/db4/J=2 往返误差 | ≤ 8.88178e-16 | 4.213e-16 | +8.88178e-16 | ✅ |
| B | symmetric/db4/J=3 往返误差 | ≤ 1.33227e-15 | 7.021e-16 | +1.33227e-15 | ✅ |
| B | symmetric/db4/J=4 往返误差 | ≤ 1.77636e-15 | 7.021e-16 | +1.77636e-15 | ✅ |
| B | symmetric/db4/J=6 往返误差 | ≤ 2.66454e-15 | 7.021e-16 | +2.66454e-15 | ✅ |
| B | symmetric/db4/J=8 往返误差 | ≤ 3.55271e-15 | 7.021e-16 | +3.55271e-15 | ✅ |
| B | symmetric/db8/J=1 往返误差 | ≤ 4.44089e-16 | 4.213e-16 | +4.44089e-16 | ✅ |
| B | symmetric/db8/J=2 往返误差 | ≤ 8.88178e-16 | 6.764e-16 | +8.88178e-16 | ✅ |
| B | symmetric/db8/J=3 往返误差 | ≤ 1.33227e-15 | 8.425e-16 | +1.33227e-15 | ✅ |
| B | symmetric/db8/J=4 往返误差 | ≤ 1.77636e-15 | 8.425e-16 | +1.77636e-15 | ✅ |
| B | symmetric/db8/J=6 往返误差 | ≤ 2.66454e-15 | 8.425e-16 | +2.66454e-15 | ✅ |
| B | symmetric/db8/J=8 往返误差 | ≤ 3.55271e-15 | 9.009e-16 | +3.55271e-15 | ✅ |
| B | zero/haar/J=1 往返误差 | ≤ 4.44089e-16 | 2.808e-16 | +4.44089e-16 | ✅ |
| B | zero/haar/J=2 往返误差 | ≤ 8.88178e-16 | 5.411e-16 | +8.88178e-16 | ✅ |
| B | zero/haar/J=3 往返误差 | ≤ 1.33227e-15 | 7.021e-16 | +1.33227e-15 | ✅ |
| B | zero/haar/J=4 往返误差 | ≤ 1.77636e-15 | 8.425e-16 | +1.77636e-15 | ✅ |
| B | zero/haar/J=6 往返误差 | ≤ 2.66454e-15 | 8.425e-16 | +2.66454e-15 | ✅ |
| B | zero/haar/J=8 往返误差 | ≤ 3.55271e-15 | 8.425e-16 | +3.55271e-15 | ✅ |
| B | zero/db2/J=1 往返误差 | ≤ 4.44089e-16 | 4.058e-16 | +4.44089e-16 | ✅ |
| B | zero/db2/J=2 往返误差 | ≤ 8.88178e-16 | 4.058e-16 | +8.88178e-16 | ✅ |
| B | zero/db2/J=3 往返误差 | ≤ 1.33227e-15 | 5.411e-16 | +1.33227e-15 | ✅ |
| B | zero/db2/J=4 往返误差 | ≤ 1.77636e-15 | 5.411e-16 | +1.77636e-15 | ✅ |
| B | zero/db2/J=6 往返误差 | ≤ 2.66454e-15 | 5.617e-16 | +2.66454e-15 | ✅ |
| B | zero/db2/J=8 往返误差 | ≤ 3.55271e-15 | 5.617e-16 | +3.55271e-15 | ✅ |
| B | zero/db4/J=1 往返误差 | ≤ 4.44089e-16 | 2.808e-16 | +4.44089e-16 | ✅ |
| B | zero/db4/J=2 往返误差 | ≤ 8.88178e-16 | 4.213e-16 | +8.88178e-16 | ✅ |
| B | zero/db4/J=3 往返误差 | ≤ 1.33227e-15 | 7.021e-16 | +1.33227e-15 | ✅ |
| B | zero/db4/J=4 往返误差 | ≤ 1.77636e-15 | 7.021e-16 | +1.77636e-15 | ✅ |
| B | zero/db4/J=6 往返误差 | ≤ 2.66454e-15 | 7.021e-16 | +2.66454e-15 | ✅ |
| B | zero/db4/J=8 往返误差 | ≤ 3.55271e-15 | 7.021e-16 | +3.55271e-15 | ✅ |
| B | zero/db8/J=1 往返误差 | ≤ 4.44089e-16 | 4.213e-16 | +4.44089e-16 | ✅ |
| B | zero/db8/J=2 往返误差 | ≤ 8.88178e-16 | 6.764e-16 | +8.88178e-16 | ✅ |
| B | zero/db8/J=3 往返误差 | ≤ 1.33227e-15 | 8.425e-16 | +1.33227e-15 | ✅ |
| B | zero/db8/J=4 往返误差 | ≤ 1.77636e-15 | 1.123e-15 | +1.77636e-15 | ✅ |
| B | zero/db8/J=6 往返误差 | ≤ 2.66454e-15 | 1.123e-15 | +2.66454e-15 | ✅ |
| B | zero/db8/J=8 往返误差 | ≤ 3.55271e-15 | 1.123e-15 | +3.55271e-15 | ✅ |
| A | 最大往返误差 | <= 4·J·u | 1.123e-15 | 3.553e-15 | ✅ |
| A | 裁到临界采样后的往返误差（负对照） | O(1)，远大于 u | 1.37918 | 0.5 | ✅ |
| B | periodic/haar/J=1 能量比 = 1 | ≤ 2.27374e-13 | 2.220e-16 | +2.27374e-13 | ✅ |
| B | periodic/haar/J=1 逐层能量占比求和 = 总能量比 | 1 | 1 | ±(1e-12·\|t\|+0) | ✅ |
| B | periodic/haar/J=3 能量比 = 1 | ≤ 2.27374e-13 | 2.220e-16 | +2.27374e-13 | ✅ |
| B | periodic/haar/J=3 逐层能量占比求和 = 总能量比 | 1 | 1 | ±(1e-12·\|t\|+0) | ✅ |
| B | periodic/haar/J=4 能量比 = 1 | ≤ 2.27374e-13 | 2.220e-16 | +2.27374e-13 | ✅ |
| B | periodic/haar/J=4 逐层能量占比求和 = 总能量比 | 1 | 1 | ±(1e-12·\|t\|+0) | ✅ |
| B | periodic/db2/J=1 能量比 = 1 | ≤ 2.27374e-13 | 0 | +2.27374e-13 | ✅ |
| B | periodic/db2/J=1 逐层能量占比求和 = 总能量比 | 1 | 1 | ±(1e-12·\|t\|+0) | ✅ |
| B | periodic/db2/J=3 能量比 = 1 | ≤ 2.27374e-13 | 2.220e-16 | +2.27374e-13 | ✅ |
| B | periodic/db2/J=3 逐层能量占比求和 = 总能量比 | 1 | 1 | ±(1e-12·\|t\|+0) | ✅ |
| B | periodic/db2/J=4 能量比 = 1 | ≤ 2.27374e-13 | 2.220e-16 | +2.27374e-13 | ✅ |
| B | periodic/db2/J=4 逐层能量占比求和 = 总能量比 | 1 | 1 | ±(1e-12·\|t\|+0) | ✅ |
| B | periodic/db4/J=1 能量比 = 1 | ≤ 2.27374e-13 | 2.220e-16 | +2.27374e-13 | ✅ |
| B | periodic/db4/J=1 逐层能量占比求和 = 总能量比 | 1 | 1 | ±(1e-12·\|t\|+0) | ✅ |
| B | periodic/db4/J=3 能量比 = 1 | ≤ 2.27374e-13 | 0 | +2.27374e-13 | ✅ |
| B | periodic/db4/J=3 逐层能量占比求和 = 总能量比 | 1 | 1 | ±(1e-12·\|t\|+0) | ✅ |
| B | periodic/db4/J=4 能量比 = 1 | ≤ 2.27374e-13 | 2.220e-16 | +2.27374e-13 | ✅ |
| B | periodic/db4/J=4 逐层能量占比求和 = 总能量比 | 1 | 1 | ±(1e-12·\|t\|+0) | ✅ |
| B | periodic/db8/J=1 能量比 = 1 | ≤ 2.27374e-13 | 2.220e-16 | +2.27374e-13 | ✅ |
| B | periodic/db8/J=1 逐层能量占比求和 = 总能量比 | 1 | 1 | ±(1e-12·\|t\|+0) | ✅ |
| B | periodic/db8/J=3 能量比 = 1 | ≤ 2.27374e-13 | 0 | +2.27374e-13 | ✅ |
| B | periodic/db8/J=3 逐层能量占比求和 = 总能量比 | 1 | 1 | ±(1e-12·\|t\|+0) | ✅ |
| B | periodic/db8/J=4 能量比 = 1 | ≤ 2.27374e-13 | 1.110e-16 | +2.27374e-13 | ✅ |
| B | periodic/db8/J=4 逐层能量占比求和 = 总能量比 | 1 | 1 | ±(1e-12·\|t\|+0) | ✅ |
| B | zero/haar/J=1 能量比 = 1 | ≤ 2.27374e-13 | 0 | +2.27374e-13 | ✅ |
| B | zero/haar/J=1 逐层能量占比求和 = 总能量比 | 1 | 1 | ±(1e-12·\|t\|+0) | ✅ |
| B | zero/haar/J=3 能量比 = 1 | ≤ 2.27374e-13 | 4.441e-16 | +2.27374e-13 | ✅ |
| B | zero/haar/J=3 逐层能量占比求和 = 总能量比 | 1 | 1 | ±(1e-12·\|t\|+0) | ✅ |
| B | zero/haar/J=4 能量比 = 1 | ≤ 2.27374e-13 | 0 | +2.27374e-13 | ✅ |
| B | zero/haar/J=4 逐层能量占比求和 = 总能量比 | 1 | 1 | ±(1e-12·\|t\|+0) | ✅ |
| B | zero/db2/J=1 能量比 = 1 | ≤ 2.27374e-13 | 0 | +2.27374e-13 | ✅ |
| B | zero/db2/J=1 逐层能量占比求和 = 总能量比 | 1 | 1 | ±(1e-12·\|t\|+0) | ✅ |
| B | zero/db2/J=3 能量比 = 1 | ≤ 2.27374e-13 | 0 | +2.27374e-13 | ✅ |
| B | zero/db2/J=3 逐层能量占比求和 = 总能量比 | 1 | 1 | ±(1e-12·\|t\|+0) | ✅ |
| B | zero/db2/J=4 能量比 = 1 | ≤ 2.27374e-13 | 2.220e-16 | +2.27374e-13 | ✅ |
| B | zero/db2/J=4 逐层能量占比求和 = 总能量比 | 1 | 1 | ±(1e-12·\|t\|+0) | ✅ |
| B | zero/db4/J=1 能量比 = 1 | ≤ 2.27374e-13 | 0 | +2.27374e-13 | ✅ |
| B | zero/db4/J=1 逐层能量占比求和 = 总能量比 | 1 | 1 | ±(1e-12·\|t\|+0) | ✅ |
| B | zero/db4/J=3 能量比 = 1 | ≤ 2.27374e-13 | 0 | +2.27374e-13 | ✅ |
| B | zero/db4/J=3 逐层能量占比求和 = 总能量比 | 1 | 1 | ±(1e-12·\|t\|+0) | ✅ |
| B | zero/db4/J=4 能量比 = 1 | ≤ 2.27374e-13 | 2.220e-16 | +2.27374e-13 | ✅ |
| B | zero/db4/J=4 逐层能量占比求和 = 总能量比 | 1 | 1 | ±(1e-12·\|t\|+0) | ✅ |
| B | zero/db8/J=1 能量比 = 1 | ≤ 2.27374e-13 | 0 | +2.27374e-13 | ✅ |
| B | zero/db8/J=1 逐层能量占比求和 = 总能量比 | 1 | 1 | ±(1e-12·\|t\|+0) | ✅ |
| B | zero/db8/J=3 能量比 = 1 | ≤ 2.27374e-13 | 2.220e-16 | +2.27374e-13 | ✅ |
| B | zero/db8/J=3 逐层能量占比求和 = 总能量比 | 1 | 1 | ±(1e-12·\|t\|+0) | ✅ |
| B | zero/db8/J=4 能量比 = 1 | ≤ 2.27374e-13 | 2.220e-16 | +2.27374e-13 | ✅ |
| B | zero/db8/J=4 逐层能量占比求和 = 总能量比 | 1 | 1 | ±(1e-12·\|t\|+0) | ✅ |
| A | symmetric 模式能量比（50 个实现均值） | n_ext/n = 1.1250 | 1.12802 | 0.02 | ✅ |
| B | 边界污染影响区（样本数）随层数增长 | increasing | 违反 0 处 | 3.41061e-13 | ✅ |
| B | 受污染的系数个数（各层） | ≤ 8 | 6 | +8 | ✅ |
| A | 偏差幅度随层数增长 | 各层最大偏差单调增（下层代表的尺度更大） | 3.47967 | 0.0532079 | ✅ |
| B | periodic/haar 各带之和 = x | ≤ 2.84217e-14 | 6.769e-16 | +2.84217e-14 | ✅ |
| B | periodic/haar 逐层累加末行 = x | ≤ 2.84217e-14 | 9.476e-16 | +2.84217e-14 | ✅ |
| B | periodic/haar 第一层细节把误差降下来 | ≤ 1.371 | 1.00306 | +0 | ✅ |
| B | periodic/db4 各带之和 = x | ≤ 2.84217e-14 | 5.415e-16 | +2.84217e-14 | ✅ |
| B | periodic/db4 逐层累加末行 = x | ≤ 2.84217e-14 | 6.769e-16 | +2.84217e-14 | ✅ |
| B | periodic/db4 第一层细节把误差降下来 | ≤ 0.849582 | 0.709429 | +0 | ✅ |
| B | periodic/db8 各带之和 = x | ≤ 2.84217e-14 | 6.769e-16 | +2.84217e-14 | ✅ |
| B | periodic/db8 逐层累加末行 = x | ≤ 2.84217e-14 | 6.769e-16 | +2.84217e-14 | ✅ |
| B | periodic/db8 第一层细节把误差降下来 | ≤ 0.707023 | 0.689836 | +0 | ✅ |
| B | symmetric/haar 各带之和 = x | ≤ 2.84217e-14 | 6.769e-16 | +2.84217e-14 | ✅ |
| B | symmetric/haar 逐层累加末行 = x | ≤ 2.84217e-14 | 9.476e-16 | +2.84217e-14 | ✅ |
| B | symmetric/haar 第一层细节把误差降下来 | ≤ 1.371 | 1.00306 | +0 | ✅ |
| B | symmetric/db4 各带之和 = x | ≤ 2.84217e-14 | 6.769e-16 | +2.84217e-14 | ✅ |
| B | symmetric/db4 逐层累加末行 = x | ≤ 2.84217e-14 | 6.769e-16 | +2.84217e-14 | ✅ |
| B | symmetric/db4 第一层细节把误差降下来 | ≤ 1.24511 | 0.709429 | +0 | ✅ |
| B | symmetric/db8 各带之和 = x | ≤ 2.84217e-14 | 8.122e-16 | +2.84217e-14 | ✅ |
| B | symmetric/db8 逐层累加末行 = x | ≤ 2.84217e-14 | 9.476e-16 | +2.84217e-14 | ✅ |
| B | symmetric/db8 第一层细节把误差降下来 | ≤ 1.20498 | 0.689825 | +0 | ✅ |
| B | zero/haar 各带之和 = x | ≤ 2.84217e-14 | 6.769e-16 | +2.84217e-14 | ✅ |
| B | zero/haar 逐层累加末行 = x | ≤ 2.84217e-14 | 9.476e-16 | +2.84217e-14 | ✅ |
| B | zero/haar 第一层细节把误差降下来 | ≤ 1.371 | 1.00306 | +0 | ✅ |
| B | zero/db4 各带之和 = x | ≤ 2.84217e-14 | 6.769e-16 | +2.84217e-14 | ✅ |
| B | zero/db4 逐层累加末行 = x | ≤ 2.84217e-14 | 5.415e-16 | +2.84217e-14 | ✅ |
| B | zero/db4 第一层细节把误差降下来 | ≤ 0.849582 | 0.709429 | +0 | ✅ |
| B | zero/db8 各带之和 = x | ≤ 2.84217e-14 | 6.769e-16 | +2.84217e-14 | ✅ |
| B | zero/db8 逐层累加末行 = x | ≤ 2.84217e-14 | 8.122e-16 | +2.84217e-14 | ✅ |
| B | zero/db8 第一层细节把误差降下来 | ≤ 0.699126 | 0.689844 | +0 | ✅ |
| B | 软阈值不放大（1-Lipschitz 的推论） | ≤ 4 | 3.3 | +0 | ✅ |
| B | 软阈值斜率不超过 1 | ≤ 1 | 0.02 | +0 | ✅ |
| B | \|c\| <= lambda 时软阈值为 0（硬阈值同样） | ≤ 0 | 0 | +0 | ✅ |
| B | \|c\| <= lambda 时硬阈值为 0 | ≤ 0 | 0 | +0 | ✅ |
| B | 硬阈值原样保留大系数（无偏） | 0 | 0 | ±(0·\|t\|+0) | ✅ |
| B | 软阈值对大系数收缩 lambda | 0 | 0 | ±(1e-12·\|t\|+0) | ✅ |
| B | \|soft\| <= \|hard\| | ≤ 0 | 0 | +0 | ✅ |
| B | 硬阈值在 \|c\|=lambda 处的跳变 | = lambda（不连续） | 0.7 | 7.000e-10 | ✅ |
| B | MAD 的相对偏差 | ≤ 0.03 | 0.01409 | +0.03 | ✅ |
| B | MAD 的相对标准差 | 0.09 | 0.107236 | ±(0.35·\|t\|+0) | ✅ |
| B | 细节系数方差 = sigma^2 | 0.09 | 0.0889298 | ±(0.03·\|t\|+0) | ✅ |
| B | periodic 模式的 MAD 均值（对照） | 同样 ~ sigma | 0.299396 | 0.009 | ✅ |
| B | SWT J=1 平移一格后系数变化 | ≤ 0 | 0 | +0 | ✅ |
| A | DWT J=1 平移残留（允许每带整体挪一格后） | O(1)：抽取带来相位错位 | 2.01446 | 0.1 | ✅ |
| B | SWT J=2 平移一格后系数变化 | ≤ 0 | 0 | +0 | ✅ |
| A | DWT J=2 平移残留（允许每带整体挪一格后） | O(1)：抽取带来相位错位 | 8.39099 | 0.1 | ✅ |
| B | SWT J=3 平移一格后系数变化 | ≤ 0 | 0 | +0 | ✅ |
| A | DWT J=3 平移残留（允许每带整体挪一格后） | O(1)：抽取带来相位错位 | 36.2351 | 0.1 | ✅ |
| B | SWT J=4 平移一格后系数变化 | ≤ 0 | 0 | +0 | ✅ |
| A | DWT J=4 平移残留（允许每带整体挪一格后） | O(1)：抽取带来相位错位 | 51.201 | 0.1 | ✅ |
| B | SWT J=1 往返误差 | ≤ 7.10543e-15 | 2.221e-16 | +7.10543e-15 | ✅ |
| B | SWT J=3 往返误差 | ≤ 2.13163e-14 | 4.441e-16 | +2.13163e-14 | ✅ |
| B | SWT J=4 往返误差 | ≤ 2.84217e-14 | 5.551e-16 | +2.84217e-14 | ✅ |
| A | DWT 与 SWT 的平移敏感度之比 | DWT 相对变化 O(1)、SWT 恰好 0 | 51.201 | 0.1 | ✅ |
| B | J=1 纯音能量比 = 框架符号 | ≤ 1.00000e-10 | 0 | +1.00000e-10 | ✅ |
| B | J=2 纯音能量比 = 框架符号 | ≤ 1.00000e-10 | 1.110e-16 | +1.00000e-10 | ✅ |
| B | J=3 纯音能量比 = 框架符号 | ≤ 1.00000e-10 | 1.110e-16 | +1.00000e-10 | ✅ |
| B | J=4 纯音能量比 = 框架符号 | ≤ 1.00000e-10 | 1.110e-16 | +1.00000e-10 | ✅ |
| A | 白噪声能量比 vs 框架符号平均（J=3） | 符号平均 4.0059（有限样本下 ~1% 偏差） | 3.97052 | 0.06 | ✅ |
| B | periodic/J=1 系数总数 = n_ext | 256 | 256 | ±(0·\|t\|+0) | ✅ |
| B | periodic/J=1 冗余度记账 | 1 | 1 | ±(1e-15·\|t\|+0) | ✅ |
| B | periodic/J=4 系数总数 = n_ext | 256 | 256 | ±(0·\|t\|+0) | ✅ |
| B | periodic/J=4 冗余度记账 | 1 | 1 | ±(1e-15·\|t\|+0) | ✅ |
| B | periodic/J=6 系数总数 = n_ext | 256 | 256 | ±(0·\|t\|+0) | ✅ |
| B | periodic/J=6 冗余度记账 | 1 | 1 | ±(1e-15·\|t\|+0) | ✅ |
| B | symmetric/J=1 系数总数 = n_ext | 288 | 288 | ±(0·\|t\|+0) | ✅ |
| B | symmetric/J=1 冗余度记账 | 1.125 | 1.125 | ±(1e-15·\|t\|+0) | ✅ |
| B | symmetric/J=4 系数总数 = n_ext | 288 | 288 | ±(0·\|t\|+0) | ✅ |
| B | symmetric/J=4 冗余度记账 | 1.125 | 1.125 | ±(1e-15·\|t\|+0) | ✅ |
| B | symmetric/J=6 系数总数 = n_ext | 384 | 384 | ±(0·\|t\|+0) | ✅ |
| B | symmetric/J=6 冗余度记账 | 1.5 | 1.5 | ±(1e-15·\|t\|+0) | ✅ |
| B | zero/J=1 系数总数 = n_ext | 288 | 288 | ±(0·\|t\|+0) | ✅ |
| B | zero/J=1 冗余度记账 | 1.125 | 1.125 | ±(1e-15·\|t\|+0) | ✅ |
| B | zero/J=4 系数总数 = n_ext | 288 | 288 | ±(0·\|t\|+0) | ✅ |
| B | zero/J=4 冗余度记账 | 1.125 | 1.125 | ±(1e-15·\|t\|+0) | ✅ |
| B | zero/J=6 系数总数 = n_ext | 384 | 384 | ±(0·\|t\|+0) | ✅ |
| B | zero/J=6 冗余度记账 | 1.5 | 1.5 | ±(1e-15·\|t\|+0) | ✅ |
| B | 边界模式与参数校验 | 非法输入当场 ValueError，不静默兜底 | 4 | 4 | ✅ |
| B | haar \|G(w)\| = \|H(w+pi)\| | ≤ 1.00000e-12 | 7.772e-16 | +1.00000e-12 | ✅ |
| B | haar \|H\|^2+\|G\|^2 = 2（逐点） | ≤ 1.00000e-12 | 1.776e-15 | +1.00000e-12 | ✅ |
| B | db4 \|G(w)\| = \|H(w+pi)\| | ≤ 1.00000e-12 | 2.831e-15 | +1.00000e-12 | ✅ |
| B | db4 \|H\|^2+\|G\|^2 = 2（逐点） | ≤ 1.00000e-12 | 3.553e-15 | +1.00000e-12 | ✅ |
| B | db8 \|G(w)\| = \|H(w+pi)\| | ≤ 1.00000e-12 | 6.911e-15 | +1.00000e-12 | ✅ |
| B | db8 \|H\|^2+\|G\|^2 = 2（逐点） | ≤ 1.00000e-12 | 6.217e-15 | +1.00000e-12 | ✅ |

<details><summary>逐条细节（拟合窗口、数据点数、残差量级等）</summary>

- **decomp_tree 有 requiredAux error**：松弛 rtol=1e-09, atol=0，余量 = 1.00000e-09；缺失会让前端拿不到误差曲线
- **decomp_tree 数组 signal 全有限**：松弛 rtol=0, atol=0，余量 = 0；t=0
- **decomp_tree 数组 approx_tree 全有限**：松弛 rtol=0, atol=0，余量 = 0；t=0
- **decomp_tree 数组 detail_tree 全有限**：松弛 rtol=0, atol=0，余量 = 0；t=0
- **decomp_tree 数组 coeff_length 全有限**：松弛 rtol=0, atol=0，余量 = 0；t=0
- **decomp_tree 数组 recon 全有限**：松弛 rtol=0, atol=0，余量 = 0；t=0
- **decomp_tree 数组 error 全有限**：松弛 rtol=0, atol=0，余量 = 0；t=0
- **decomp_tree 数组 energy_share 全有限**：松弛 rtol=0, atol=0，余量 = 0；t=0
- **decomp_tree 数组 coeff_count 全有限**：松弛 rtol=0, atol=0，余量 = 0；t=0
- **decomp_tree meta 只含 number/bool/null**：松弛 rtol=0, atol=0，余量 = 0；越界键：[]
- **decomp_tree meta[redundancy] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **decomp_tree meta[pr_error] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **decomp_tree meta[mra_error] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **decomp_tree meta[stage_error] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **decomp_tree meta[rel_stage_error] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **decomp_tree meta[boundary_share] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **decomp_tree meta[energy_top] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **decomp_tree meta[energy_detail_sum] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **decomp_tree meta[max_detail] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **decomp_tree meta[t_actual] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **decomp_tree 有 requiredAux error**：松弛 rtol=1e-09, atol=0，余量 = 1.00000e-09；缺失会让前端拿不到误差曲线
- **decomp_tree 数组 signal 全有限**：松弛 rtol=0, atol=0，余量 = 0；t=1
- **decomp_tree 数组 approx_tree 全有限**：松弛 rtol=0, atol=0，余量 = 0；t=1
- **decomp_tree 数组 detail_tree 全有限**：松弛 rtol=0, atol=0，余量 = 0；t=1
- **decomp_tree 数组 coeff_length 全有限**：松弛 rtol=0, atol=0，余量 = 0；t=1
- **decomp_tree 数组 recon 全有限**：松弛 rtol=0, atol=0，余量 = 0；t=1
- **decomp_tree 数组 error 全有限**：松弛 rtol=0, atol=0，余量 = 0；t=1
- **decomp_tree 数组 energy_share 全有限**：松弛 rtol=0, atol=0，余量 = 0；t=1
- **decomp_tree 数组 coeff_count 全有限**：松弛 rtol=0, atol=0，余量 = 0；t=1
- **decomp_tree meta 只含 number/bool/null**：松弛 rtol=0, atol=0，余量 = 0；越界键：[]
- **decomp_tree meta[redundancy] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **decomp_tree meta[pr_error] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **decomp_tree meta[mra_error] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **decomp_tree meta[stage_error] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **decomp_tree meta[rel_stage_error] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **decomp_tree meta[boundary_share] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **decomp_tree meta[energy_top] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **decomp_tree meta[energy_detail_sum] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **decomp_tree meta[max_detail] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **decomp_tree meta[t_actual] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **time_scale_map 有 requiredAux error**：松弛 rtol=1e-09, atol=0，余量 = 1.00000e-09；缺失会让前端拿不到误差曲线
- **time_scale_map 数组 signal 全有限**：松弛 rtol=0, atol=0，余量 = 0；t=0
- **time_scale_map 数组 scalogram 全有限**：松弛 rtol=0, atol=0，余量 = 0；t=0
- **time_scale_map 数组 stft_map 全有限**：松弛 rtol=0, atol=0，余量 = 0；t=0
- **time_scale_map 数组 stft_time 全有限**：松弛 rtol=0, atol=0，余量 = 0；t=0
- **time_scale_map 数组 stft_freq 全有限**：松弛 rtol=0, atol=0，余量 = 0；t=0
- **time_scale_map 数组 wav_width 全有限**：松弛 rtol=0, atol=0，余量 = 0；t=0
- **time_scale_map 数组 wav_band 全有限**：松弛 rtol=0, atol=0，余量 = 0；t=0
- **time_scale_map 数组 stft_width 全有限**：松弛 rtol=0, atol=0，余量 = 0；t=0
- **time_scale_map 数组 stft_band 全有限**：松弛 rtol=0, atol=0，余量 = 0；t=0
- **time_scale_map 数组 cell_area 全有限**：松弛 rtol=0, atol=0，余量 = 0；t=0
- **time_scale_map 数组 error 全有限**：松弛 rtol=0, atol=0，余量 = 0；t=0
- **time_scale_map meta 只含 number/bool/null**：松弛 rtol=0, atol=0，余量 = 0；越界键：[]
- **time_scale_map meta[n_stft_time] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **time_scale_map meta[n_stft_freq] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **time_scale_map meta[pr_error] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **time_scale_map meta[swt_energy_ratio] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **time_scale_map meta[swt_symbol_mean] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **time_scale_map meta[swt_redundancy] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **time_scale_map meta[cell_area_min] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **time_scale_map meta[cell_area_max] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **time_scale_map meta[t_actual] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **denoise_threshold 有 requiredAux error**：松弛 rtol=1e-09, atol=0，余量 = 1.00000e-09；缺失会让前端拿不到误差曲线
- **denoise_threshold 数组 clean 全有限**：松弛 rtol=0, atol=0，余量 = 0；t=0
- **denoise_threshold 数组 noisy 全有限**：松弛 rtol=0, atol=0，余量 = 0；t=0
- **denoise_threshold 数组 denoised 全有限**：松弛 rtol=0, atol=0，余量 = 0；t=0
- **denoise_threshold 数组 error 全有限**：松弛 rtol=0, atol=0，余量 = 0；t=0
- **denoise_threshold 数组 lambda_grid 全有限**：松弛 rtol=0, atol=0，余量 = 0；t=0
- **denoise_threshold 数组 mse_curve 全有限**：松弛 rtol=0, atol=0，余量 = 0；t=0
- **denoise_threshold 数组 bias2_curve 全有限**：松弛 rtol=0, atol=0，余量 = 0；t=0
- **denoise_threshold 数组 var_curve 全有限**：松弛 rtol=0, atol=0，余量 = 0；t=0
- **denoise_threshold 数组 mse_noisy_line 全有限**：松弛 rtol=0, atol=0，余量 = 0；t=0
- **denoise_threshold 数组 universal_line 全有限**：松弛 rtol=0, atol=0，余量 = 0；t=0
- **denoise_threshold meta 只含 number/bool/null**：松弛 rtol=0, atol=0，余量 = 0；越界键：[]
- **denoise_threshold meta[sigma] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **denoise_threshold meta[sigma_hat] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **denoise_threshold meta[sigma_hat_bias] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **denoise_threshold meta[lam] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **denoise_threshold meta[mse] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **denoise_threshold meta[mse_curve_here] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **denoise_threshold meta[lam_grid_here] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **denoise_threshold meta[mse_noisy] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **denoise_threshold meta[mse_opt] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **denoise_threshold meta[lam_opt] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **denoise_threshold meta[lam_grid] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **denoise_threshold meta[universal] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **denoise_threshold meta[universal_over_opt] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **denoise_threshold meta[mse_at_universal] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **denoise_threshold meta[bias2_curve_here] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **denoise_threshold meta[var_curve_here] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **denoise_threshold meta[snr_in] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **denoise_threshold meta[snr_out] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **denoise_threshold meta[rel_error] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **denoise_threshold meta[n_trials] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **denoise_threshold meta[t_actual] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **vanishing_moments 有 requiredAux error**：松弛 rtol=1e-09, atol=0，余量 = 1.00000e-09；缺失会让前端拿不到误差曲线
- **vanishing_moments 数组 signal 全有限**：松弛 rtol=0, atol=0，余量 = 0；t=0
- **vanishing_moments 数组 detail_tree 全有限**：松弛 rtol=0, atol=0，余量 = 0；t=0
- **vanishing_moments 数组 detail_length 全有限**：松弛 rtol=0, atol=0，余量 = 0；t=0
- **vanishing_moments 数组 degree_axis 全有限**：松弛 rtol=0, atol=0，余量 = 0；t=0
- **vanishing_moments 数组 interior_ratio 全有限**：松弛 rtol=0, atol=0，余量 = 0；t=0
- **vanishing_moments 数组 interior_valid 全有限**：松弛 rtol=0, atol=0，余量 = 0；t=0
- **vanishing_moments 数组 boundary_ratio 全有限**：松弛 rtol=0, atol=0，余量 = 0；t=0
- **vanishing_moments 数组 u_line 全有限**：松弛 rtol=0, atol=0，余量 = 0；t=0
- **vanishing_moments 数组 error 全有限**：松弛 rtol=0, atol=0，余量 = 0；t=0
- **vanishing_moments 数组 level_ratio 全有限**：松弛 rtol=0, atol=0，余量 = 0；t=0
- **vanishing_moments 数组 level_boundary 全有限**：松弛 rtol=0, atol=0，余量 = 0；t=0
- **vanishing_moments meta 只含 number/bool/null**：松弛 rtol=0, atol=0，余量 = 0；越界键：[]
- **vanishing_moments meta[interior_max] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **vanishing_moments meta[boundary_max] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **vanishing_moments meta[ratio_over_u] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **vanishing_moments meta[level1_ratio] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **vanishing_moments meta[n_interior] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **vanishing_moments meta[n_ratio_floored] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **vanishing_moments meta[u] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **vanishing_moments meta[pr_error] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **vanishing_moments meta[pollution_level1] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **vanishing_moments meta[pollution_span1] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **vanishing_moments meta[t_actual] 有限**：松弛 rtol=0, atol=0，余量 = 0
- **四个帧的数组与 strings**：decomp_tree(8 数组/8 字符串)；time_scale_map(11 数组/6 字符串)；denoise_threshold(10 数组/8 字符串)；vanishing_moments(11 数组/7 字符串)
- **本章 4 个可视化**：|实测−理论| = 0，容差 rtol=0, atol=0
- **viz id 不重复**：|实测−理论| = 0，容差 rtol=0, atol=0
- **decomp_tree dataSource = backend**：|实测−理论| = 0，容差 rtol=0, atol=0
- **decomp_tree pitfalls 非空**：松弛 rtol=0, atol=0，余量 = 1.00000e+09；140 字（要求 >= 60）
- **decomp_tree pitfalls 长度 >= 60 字**：松弛 rtol=0, atol=0，余量 = 80
- **decomp_tree theory 至少 1 条**：松弛 rtol=0, atol=0，余量 = 3
- **decomp_tree.tree 显式声明 autoscaled**：|实测−理论| = 0，容差 rtol=0, atol=0
- **decomp_tree.recon 显式声明 autoscaled**：|实测−理论| = 0，容差 rtol=0, atol=0
- **decomp_tree.error 显式声明 autoscaled**：|实测−理论| = 0，容差 rtol=0, atol=0
- **decomp_tree.tree 坐标轴尺度合法**：|实测−理论| = 0，容差 rtol=0, atol=0
- **decomp_tree.recon 坐标轴尺度合法**：|实测−理论| = 0，容差 rtol=0, atol=0
- **decomp_tree.error 坐标轴尺度合法**：|实测−理论| = 0，容差 rtol=0, atol=0
- **decomp_tree.signal 默认值在 options 内**：松弛 rtol=0, atol=0，余量 = 0
- **decomp_tree.signal options 至少 2 个**：松弛 rtol=0, atol=0，余量 = 4
- **decomp_tree.n default >= min**：松弛 rtol=0, atol=0，余量 = 192
- **decomp_tree.n default <= max**：松弛 rtol=0, atol=0，余量 = 768
- **decomp_tree.n critical 有解释**：松弛 rtol=0, atol=0，余量 = 59
- **decomp_tree.n critical 256 >= min**：松弛 rtol=0, atol=0，余量 = 192
- **decomp_tree.n critical 256 <= max**：松弛 rtol=0, atol=0，余量 = 768
- **decomp_tree.wavelet 默认值在 options 内**：松弛 rtol=0, atol=0，余量 = 0
- **decomp_tree.wavelet options 至少 2 个**：松弛 rtol=0, atol=0，余量 = 3
- **decomp_tree.level default >= min**：松弛 rtol=0, atol=0，余量 = 3
- **decomp_tree.level default <= max**：松弛 rtol=0, atol=0，余量 = 2
- **decomp_tree.level critical 有解释**：松弛 rtol=0, atol=0，余量 = 92
- **decomp_tree.level critical 4 >= min**：松弛 rtol=0, atol=0，余量 = 3
- **decomp_tree.level critical 4 <= max**：松弛 rtol=0, atol=0，余量 = 2
- **decomp_tree.mode 默认值在 options 内**：松弛 rtol=0, atol=0，余量 = 0
- **decomp_tree.mode options 至少 2 个**：松弛 rtol=0, atol=0，余量 = 1
- **time_scale_map dataSource = backend**：|实测−理论| = 0，容差 rtol=0, atol=0
- **time_scale_map pitfalls 非空**：松弛 rtol=0, atol=0，余量 = 1.00000e+09；153 字（要求 >= 60）
- **time_scale_map pitfalls 长度 >= 60 字**：松弛 rtol=0, atol=0，余量 = 93
- **time_scale_map theory 至少 1 条**：松弛 rtol=0, atol=0，余量 = 3
- **time_scale_map.scale 显式声明 autoscaled**：|实测−理论| = 0，容差 rtol=0, atol=0
- **time_scale_map.stft 显式声明 autoscaled**：|实测−理论| = 0，容差 rtol=0, atol=0
- **time_scale_map.error 显式声明 autoscaled**：|实测−理论| = 0，容差 rtol=0, atol=0
- **time_scale_map.scale 坐标轴尺度合法**：|实测−理论| = 0，容差 rtol=0, atol=0
- **time_scale_map.stft 坐标轴尺度合法**：|实测−理论| = 0，容差 rtol=0, atol=0
- **time_scale_map.error 坐标轴尺度合法**：|实测−理论| = 0，容差 rtol=0, atol=0
- **time_scale_map time.mode=none 时不得声明 tMax**：|实测−理论| = 0，容差 rtol=0, atol=0
- **time_scale_map time.mode=none 时不得声明 dt**：|实测−理论| = 0，容差 rtol=0, atol=0
- **time_scale_map time.mode=none 时不得声明 fps**：|实测−理论| = 0，容差 rtol=0, atol=0
- **time_scale_map.signal 默认值在 options 内**：松弛 rtol=0, atol=0，余量 = 0
- **time_scale_map.signal options 至少 2 个**：松弛 rtol=0, atol=0，余量 = 2
- **time_scale_map.n default >= min**：松弛 rtol=0, atol=0，余量 = 128
- **time_scale_map.n default <= max**：松弛 rtol=0, atol=0，余量 = 256
- **time_scale_map.wavelet 默认值在 options 内**：松弛 rtol=0, atol=0，余量 = 0
- **time_scale_map.wavelet options 至少 2 个**：松弛 rtol=0, atol=0，余量 = 1
- **time_scale_map.level default >= min**：松弛 rtol=0, atol=0，余量 = 2
- **time_scale_map.level default <= max**：松弛 rtol=0, atol=0，余量 = 1
- **time_scale_map.level critical 有解释**：松弛 rtol=0, atol=0，余量 = 73
- **time_scale_map.level critical 5 >= min**：松弛 rtol=0, atol=0，余量 = 3
- **time_scale_map.level critical 5 <= max**：松弛 rtol=0, atol=0，余量 = 0
- **time_scale_map.stft_window default >= min**：松弛 rtol=0, atol=0，余量 = 24
- **time_scale_map.stft_window default <= max**：松弛 rtol=0, atol=0，余量 = 32
- **time_scale_map.stft_window critical 有解释**：松弛 rtol=0, atol=0，余量 = 82
- **time_scale_map.stft_window critical 16 >= min**：松弛 rtol=0, atol=0，余量 = 8
- **time_scale_map.stft_window critical 16 <= max**：松弛 rtol=0, atol=0，余量 = 48
- **denoise_threshold dataSource = backend**：|实测−理论| = 0，容差 rtol=0, atol=0
- **denoise_threshold pitfalls 非空**：松弛 rtol=0, atol=0，余量 = 1.00000e+09；165 字（要求 >= 60）
- **denoise_threshold pitfalls 长度 >= 60 字**：松弛 rtol=0, atol=0，余量 = 105
- **denoise_threshold theory 至少 1 条**：松弛 rtol=0, atol=0，余量 = 3
- **denoise_threshold.signals 显式声明 autoscaled**：|实测−理论| = 0，容差 rtol=0, atol=0
- **denoise_threshold.curves 显式声明 autoscaled**：|实测−理论| = 0，容差 rtol=0, atol=0
- **denoise_threshold.error 显式声明 autoscaled**：|实测−理论| = 0，容差 rtol=0, atol=0
- **denoise_threshold.signals 坐标轴尺度合法**：|实测−理论| = 0，容差 rtol=0, atol=0
- **denoise_threshold.curves 坐标轴尺度合法**：|实测−理论| = 0，容差 rtol=0, atol=0
- **denoise_threshold.error 坐标轴尺度合法**：|实测−理论| = 0，容差 rtol=0, atol=0
- **denoise_threshold time.mode=none 时不得声明 tMax**：|实测−理论| = 0，容差 rtol=0, atol=0
- **denoise_threshold time.mode=none 时不得声明 dt**：|实测−理论| = 0，容差 rtol=0, atol=0
- **denoise_threshold time.mode=none 时不得声明 fps**：|实测−理论| = 0，容差 rtol=0, atol=0
- **denoise_threshold.signal 默认值在 options 内**：松弛 rtol=0, atol=0，余量 = 0
- **denoise_threshold.signal options 至少 2 个**：松弛 rtol=0, atol=0，余量 = 1
- **denoise_threshold.n default >= min**：松弛 rtol=0, atol=0，余量 = 128
- **denoise_threshold.n default <= max**：松弛 rtol=0, atol=0，余量 = 256
- **denoise_threshold.sigma default >= min**：松弛 rtol=0, atol=0，余量 = 0.25
- **denoise_threshold.sigma default <= max**：松弛 rtol=0, atol=0，余量 = 0.3
- **denoise_threshold.lam default >= min**：松弛 rtol=0, atol=0，余量 = 0.28
- **denoise_threshold.lam default <= max**：松弛 rtol=0, atol=0，余量 = 1.32
- **denoise_threshold.lam critical 有解释**：松弛 rtol=0, atol=0，余量 = 125
- **denoise_threshold.lam critical 0.28 >= min**：松弛 rtol=0, atol=0，余量 = 0.28
- **denoise_threshold.lam critical 0.28 <= max**：松弛 rtol=0, atol=0，余量 = 1.32
- **denoise_threshold.lam critical 1 >= min**：松弛 rtol=0, atol=0，余量 = 1
- **denoise_threshold.lam critical 1 <= max**：松弛 rtol=0, atol=0，余量 = 0.6
- **denoise_threshold.rule 默认值在 options 内**：松弛 rtol=0, atol=0，余量 = 0
- **denoise_threshold.rule options 至少 2 个**：松弛 rtol=0, atol=0，余量 = 0
- **denoise_threshold.wavelet 默认值在 options 内**：松弛 rtol=0, atol=0，余量 = 0
- **denoise_threshold.wavelet options 至少 2 个**：松弛 rtol=0, atol=0，余量 = 1
- **denoise_threshold.level default >= min**：松弛 rtol=0, atol=0，余量 = 3
- **denoise_threshold.level default <= max**：松弛 rtol=0, atol=0，余量 = 2
- **vanishing_moments dataSource = backend**：|实测−理论| = 0，容差 rtol=0, atol=0
- **vanishing_moments pitfalls 非空**：松弛 rtol=0, atol=0，余量 = 1.00000e+09；154 字（要求 >= 60）
- **vanishing_moments pitfalls 长度 >= 60 字**：松弛 rtol=0, atol=0，余量 = 94
- **vanishing_moments theory 至少 1 条**：松弛 rtol=0, atol=0，余量 = 3
- **vanishing_moments.poly 显式声明 autoscaled**：|实测−理论| = 0，容差 rtol=0, atol=0
- **vanishing_moments.ratio 显式声明 autoscaled**：|实测−理论| = 0，容差 rtol=0, atol=0
- **vanishing_moments.error 显式声明 autoscaled**：|实测−理论| = 0，容差 rtol=0, atol=0
- **vanishing_moments.poly 坐标轴尺度合法**：|实测−理论| = 0，容差 rtol=0, atol=0
- **vanishing_moments.ratio 坐标轴尺度合法**：|实测−理论| = 0，容差 rtol=0, atol=0
- **vanishing_moments.error 坐标轴尺度合法**：|实测−理论| = 0，容差 rtol=0, atol=0
- **vanishing_moments time.mode=none 时不得声明 tMax**：|实测−理论| = 0，容差 rtol=0, atol=0
- **vanishing_moments time.mode=none 时不得声明 dt**：|实测−理论| = 0，容差 rtol=0, atol=0
- **vanishing_moments time.mode=none 时不得声明 fps**：|实测−理论| = 0，容差 rtol=0, atol=0
- **vanishing_moments.wavelet 默认值在 options 内**：松弛 rtol=0, atol=0，余量 = 0
- **vanishing_moments.wavelet options 至少 2 个**：松弛 rtol=0, atol=0，余量 = 3
- **vanishing_moments.degree default >= min**：松弛 rtol=0, atol=0，余量 = 3
- **vanishing_moments.degree default <= max**：松弛 rtol=0, atol=0，余量 = 3
- **vanishing_moments.degree critical 有解释**：松弛 rtol=0, atol=0，余量 = 99
- **vanishing_moments.degree critical 4 >= min**：松弛 rtol=0, atol=0，余量 = 4
- **vanishing_moments.degree critical 4 <= max**：松弛 rtol=0, atol=0，余量 = 2
- **vanishing_moments.n default >= min**：松弛 rtol=0, atol=0，余量 = 192
- **vanishing_moments.n default <= max**：松弛 rtol=0, atol=0，余量 = 256
- **vanishing_moments.level default >= min**：松弛 rtol=0, atol=0，余量 = 2
- **vanishing_moments.level default <= max**：松弛 rtol=0, atol=0，余量 = 2
- **vanishing_moments.mode 默认值在 options 内**：松弛 rtol=0, atol=0，余量 = 0
- **vanishing_moments.mode options 至少 2 个**：松弛 rtol=0, atol=0，余量 = 1
- **manifest 契约**：decomp_tree:5 参数/3 轴；time_scale_map:5 参数/3 轴；denoise_threshold:7 参数/3 轴；vanishing_moments:5 参数/3 轴
- **pr_error = idwt 直算路径**：|实测−理论| = 0，容差 rtol=0, atol=0
- **mra_error = 逐带求和路径**：|实测−理论| = 0，容差 rtol=0, atol=0；两条路径都在 u 量级但不是同一个数，帧里必须分别报
- **n_ext 一致**：|实测−理论| = 0，容差 rtol=0, atol=0
- **冗余度一致**：|实测−理论| = 0，容差 rtol=1e-15, atol=0
- **coeff_count = coefficient_lengths**：|实测−理论| = 0，容差 rtol=0, atol=0
- **energy_share = level_energies**：|实测−理论| = 0，容差 rtol=0, atol=0
- **逐层重构全部有限**：松弛 rtol=0, atol=0，余量 = 0
- **逐层重构末行 = x（numcore 口径）**：松弛 rtol=1, atol=0，余量 = 5.50671e-14
- **t=tMax 时 stage_error 落到地板**：松弛 rtol=0, atol=0，余量 = 9.98224e-13；实测 1.78e-15（n=256、db4、symmetric）
- **boundary_share 复算一致**：|实测−理论| = 0，容差 rtol=1e-12, atol=0
- **帧里的 signal 就是 test_signal 的输出**：|实测−理论| = 0，容差 rtol=0, atol=0
- **SWT 能量比复算一致**：|实测−理论| = 0，容差 rtol=1e-12, atol=0；实测 9.031719（不是常数 J+1=5）
- **框架符号平均复算一致**：|实测−理论| = 0，容差 rtol=1e-12, atol=0
- **SWT 冗余度 = J+1**：|实测−理论| = 0，容差 rtol=0, atol=0
- **尺度图行数 = J+1**：|实测−理论| = 0，容差 rtol=0, atol=0
- **每层系数与信号等长（非抽取）**：|实测−理论| = 0，容差 rtol=0, atol=0
- **scalogram 数组 = numcore 输出**：|实测−理论| = 0，容差 rtol=0, atol=0
- **SWT 往返误差复算一致**：|实测−理论| = 0，容差 rtol=1e-12, atol=0
- **STFT 频率格数与 numcore 一致**：|实测−理论| = 0，容差 rtol=0, atol=0
- **STFT 时间格数与 numcore 一致**：|实测−理论| = 0，容差 rtol=0, atol=0
- **stft_map 数组 = numcore 输出**：|实测−理论| = 0，容差 rtol=0, atol=0
- **小波带宽数组一致**：|实测−理论| = 0，容差 rtol=0, atol=0
- **带宽随层数下降（多分辨）**：松弛 rtol=1, atol=0，余量 = 1992
- **时间-尺度帧的两套分辨率**：最粗层时间支撑占比 0.2227，STFT 恒定 0.1250；能量比 9.0317 vs 符号平均 5.0137
- **clean 数组 = test_signal 输出**：|实测−理论| = 0，容差 rtol=0, atol=0
- **noisy - clean 的标准差 = sigma**：|实测−理论| = 0，容差 rtol=1e-09, atol=0；帧里的噪声是固定实现（公共随机数），拖动阈值时不换噪声
- **sigma_hat 复算一致**：|实测−理论| = 0，容差 rtol=1e-12, atol=0
- **通用阈值复算一致**：|实测−理论| = 0，容差 rtol=1e-12, atol=0
- **denoised 数组 = numcore denoise 输出**：|实测−理论| = 0，容差 rtol=0, atol=0
- **meta.mse = 帧内单次实现的 MSE（画出来的那条曲线的读数）**：|实测−理论| = 0，容差 rtol=0, atol=0
- **lambda_grid 与同种子重算逐点相同**：|实测−理论| = 0，容差 rtol=0, atol=0
- **mse_curve 与同种子重算逐点相同**：|实测−理论| = 0，容差 rtol=0, atol=0
- **bias2_curve 与同种子重算逐点相同**：|实测−理论| = 0，容差 rtol=0, atol=0
- **var_curve 与同种子重算逐点相同**：|实测−理论| = 0，容差 rtol=0, atol=0
- **最优阈值复算一致**：|实测−理论| = 0，容差 rtol=1e-12, atol=0
- **最优 MSE 复算一致**：|实测−理论| = 0，容差 rtol=1e-09, atol=0
- **曲线上的点满足 MSE = bias^2 + var**：|实测−理论| = 1.38778e-17，容差 rtol=0, atol=7.10543e-15；三个数各自存成双精度，相加的舍入差 <= 64u
- **mse_curve_here 落在同一条曲线上**：|实测−理论| = 0，容差 rtol=0, atol=0
- **var_curve_here 落在同一条曲线上**：|实测−理论| = 0，容差 rtol=0, atol=0
- **lam_grid_here = 最接近 lambda 的栅格点**：|实测−理论| = 0，容差 rtol=0, atol=0；单次实现的 MSE 与蒙特卡洛曲线的读数不是同一个数，二者都要报
- **消失矩阶数一致**：|实测−理论| = 0，容差 rtol=0, atol=0
- **u 参考线 = 2^-53**：|实测−理论| = 0，容差 rtol=0, atol=0
- **signal 数组 = t^k**：|实测−理论| = 0，容差 rtol=0, atol=0
- **往返误差复算一致**：|实测−理论| = 0，容差 rtol=1e-12, atol=0
- **interior_max 复算一致**：|实测−理论| = 0，容差 rtol=1e-12, atol=0
- **boundary_max 复算一致**：|实测−理论| = 0，容差 rtol=1e-12, atol=0
- **killed 标志一致**：|实测−理论| = 0，容差 rtol=0, atol=0
- **次数扫描轴 = 0..N+2**：|实测−理论| = 0，容差 rtol=0, atol=0
- **k=0 掩码为 0 时实测严格为 0**：|实测−理论| = 0，容差 rtol=0, atol=0
- **k=0 画出来的是地板值**：|实测−理论| = 0，容差 rtol=0, atol=0
- **k=1 的比值是实测值**：|实测−理论| = 0，容差 rtol=1e-12, atol=0
- **k=1 掩码为 1 时实测 > 0**：松弛 rtol=0, atol=0，余量 = 2.07680e-16
- **k=2 的比值是实测值**：|实测−理论| = 0，容差 rtol=1e-12, atol=0
- **k=2 掩码为 1 时实测 > 0**：松弛 rtol=0, atol=0，余量 = 8.79054e-17
- **k=3 的比值是实测值**：|实测−理论| = 0，容差 rtol=1e-12, atol=0
- **k=3 掩码为 1 时实测 > 0**：松弛 rtol=0, atol=0，余量 = 9.30169e-17
- **k=4 的比值是实测值**：|实测−理论| = 0，容差 rtol=1e-12, atol=0
- **k=4 掩码为 1 时实测 > 0**：松弛 rtol=0, atol=0，余量 = 1.06099e-05
- **k=5 的比值是实测值**：|实测−理论| = 0，容差 rtol=1e-12, atol=0
- **k=5 掩码为 1 时实测 > 0**：松弛 rtol=0, atol=0，余量 = 4.73380e-05
- **k=6 的比值是实测值**：|实测−理论| = 0，容差 rtol=1e-12, atol=0
- **k=6 掩码为 1 时实测 > 0**：松弛 rtol=0, atol=0，余量 = 1.26871e-04
- **地板点计数一致**：|实测−理论| = 0，容差 rtol=0, atol=0
- **u 参考线数组 = 2^-53**：|实测−理论| = 0，容差 rtol=0, atol=0
- **第 1 层量级比复算一致**：|实测−理论| = 0，容差 rtol=1e-12, atol=0
- **边界污染幅度复算一致**：|实测−理论| = 0，容差 rtol=1e-12, atol=0
- **边界污染影响区复算一致**：|实测−理论| = 0，容差 rtol=1e-12, atol=0
- **strings 里的名字与 numcore 一致**：|实测−理论| = 0，容差 rtol=0, atol=0
- **strings 里的边界模式与 numcore 一致**：|实测−理论| = 0，容差 rtol=0, atol=0
- **periodic/haar/bumps 往返误差一致**：|实测−理论| = 0，容差 rtol=0, atol=0
- **periodic/haar/bumps n_ext 一致**：|实测−理论| = 0，容差 rtol=0, atol=0
- **periodic/haar/bumps 末段误差落地板**：松弛 rtol=0, atol=0，余量 = 9.98668e-13
- **periodic/haar/blocks 往返误差一致**：|实测−理论| = 0，容差 rtol=0, atol=0
- **periodic/haar/blocks n_ext 一致**：|实测−理论| = 0，容差 rtol=0, atol=0
- **periodic/haar/blocks 末段误差落地板**：松弛 rtol=0, atol=0，余量 = 9.98224e-13
- **periodic/haar/heavisine 往返误差一致**：|实测−理论| = 0，容差 rtol=0, atol=0
- **periodic/haar/heavisine n_ext 一致**：|实测−理论| = 0，容差 rtol=0, atol=0
- **periodic/haar/heavisine 末段误差落地板**：松弛 rtol=0, atol=0，余量 = 9.98668e-13
- **periodic/db8/bumps 往返误差一致**：|实测−理论| = 0，容差 rtol=0, atol=0
- **periodic/db8/bumps n_ext 一致**：|实测−理论| = 0，容差 rtol=0, atol=0
- **periodic/db8/bumps 末段误差落地板**：松弛 rtol=0, atol=0，余量 = 9.98890e-13
- **periodic/db8/blocks 往返误差一致**：|实测−理论| = 0，容差 rtol=0, atol=0
- **periodic/db8/blocks n_ext 一致**：|实测−理论| = 0，容差 rtol=0, atol=0
- **periodic/db8/blocks 末段误差落地板**：松弛 rtol=0, atol=0，余量 = 9.96891e-13
- **periodic/db8/heavisine 往返误差一致**：|实测−理论| = 0，容差 rtol=0, atol=0
- **periodic/db8/heavisine n_ext 一致**：|实测−理论| = 0，容差 rtol=0, atol=0
- **periodic/db8/heavisine 末段误差落地板**：松弛 rtol=0, atol=0，余量 = 9.98224e-13
- **symmetric/haar/bumps 往返误差一致**：|实测−理论| = 0，容差 rtol=0, atol=0
- **symmetric/haar/bumps n_ext 一致**：|实测−理论| = 0，容差 rtol=0, atol=0
- **symmetric/haar/bumps 末段误差落地板**：松弛 rtol=0, atol=0，余量 = 9.98668e-13
- **symmetric/haar/blocks 往返误差一致**：|实测−理论| = 0，容差 rtol=0, atol=0
- **symmetric/haar/blocks n_ext 一致**：|实测−理论| = 0，容差 rtol=0, atol=0
- **symmetric/haar/blocks 末段误差落地板**：松弛 rtol=0, atol=0，余量 = 9.98224e-13
- **symmetric/haar/heavisine 往返误差一致**：|实测−理论| = 0，容差 rtol=0, atol=0
- **symmetric/haar/heavisine n_ext 一致**：|实测−理论| = 0，容差 rtol=0, atol=0
- **symmetric/haar/heavisine 末段误差落地板**：松弛 rtol=0, atol=0，余量 = 9.98668e-13
- **symmetric/db8/bumps 往返误差一致**：|实测−理论| = 0，容差 rtol=0, atol=0
- **symmetric/db8/bumps n_ext 一致**：|实测−理论| = 0，容差 rtol=0, atol=0
- **symmetric/db8/bumps 末段误差落地板**：松弛 rtol=0, atol=0，余量 = 9.98890e-13
- **symmetric/db8/blocks 往返误差一致**：|实测−理论| = 0，容差 rtol=0, atol=0
- **symmetric/db8/blocks n_ext 一致**：|实测−理论| = 0，容差 rtol=0, atol=0
- **symmetric/db8/blocks 末段误差落地板**：松弛 rtol=0, atol=0，余量 = 9.96891e-13
- **symmetric/db8/heavisine 往返误差一致**：|实测−理论| = 0，容差 rtol=0, atol=0
- **symmetric/db8/heavisine n_ext 一致**：|实测−理论| = 0，容差 rtol=0, atol=0
- **symmetric/db8/heavisine 末段误差落地板**：松弛 rtol=0, atol=0，余量 = 9.98224e-13
- **zero/haar/bumps 往返误差一致**：|实测−理论| = 0，容差 rtol=0, atol=0
- **zero/haar/bumps n_ext 一致**：|实测−理论| = 0，容差 rtol=0, atol=0
- **zero/haar/bumps 末段误差落地板**：松弛 rtol=0, atol=0，余量 = 9.98668e-13
- **zero/haar/blocks 往返误差一致**：|实测−理论| = 0，容差 rtol=0, atol=0
- **zero/haar/blocks n_ext 一致**：|实测−理论| = 0，容差 rtol=0, atol=0
- **zero/haar/blocks 末段误差落地板**：松弛 rtol=0, atol=0，余量 = 9.98224e-13
- **zero/haar/heavisine 往返误差一致**：|实测−理论| = 0，容差 rtol=0, atol=0
- **zero/haar/heavisine n_ext 一致**：|实测−理论| = 0，容差 rtol=0, atol=0
- **zero/haar/heavisine 末段误差落地板**：松弛 rtol=0, atol=0，余量 = 9.98668e-13
- **zero/db8/bumps 往返误差一致**：|实测−理论| = 0，容差 rtol=0, atol=0
- **zero/db8/bumps n_ext 一致**：|实测−理论| = 0，容差 rtol=0, atol=0
- **zero/db8/bumps 末段误差落地板**：松弛 rtol=0, atol=0，余量 = 9.98890e-13
- **zero/db8/blocks 往返误差一致**：|实测−理论| = 0，容差 rtol=0, atol=0
- **zero/db8/blocks n_ext 一致**：|实测−理论| = 0，容差 rtol=0, atol=0
- **zero/db8/blocks 末段误差落地板**：松弛 rtol=0, atol=0，余量 = 9.96891e-13
- **zero/db8/heavisine 往返误差一致**：|实测−理论| = 0，容差 rtol=0, atol=0
- **zero/db8/heavisine n_ext 一致**：|实测−理论| = 0，容差 rtol=0, atol=0
- **zero/db8/heavisine 末段误差落地板**：松弛 rtol=0, atol=0，余量 = 9.98224e-13
- **换参数后 SWT 能量比仍与 numcore 一致**：|实测−理论| = 0，容差 rtol=1e-12, atol=0
- **haar 对常数项严格为 0（掩码点为地板）**：|实测−理论| = 0，容差 rtol=0, atol=0
- **k=0 在 haar 下确实有 1 个地板点**：|实测−理论| = 0，容差 rtol=0, atol=0
- **参数扫掠下的一致性**：periodic/haar/bumps:ratio=1.0000；periodic/haar/blocks:ratio=1.0000；periodic/haar/heavisine:ratio=1.0000；periodic/db8/bumps:ratio=1.0000 …
- **haar k=0 J=1 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 2.00000e-15；N=1；实测 0.00e+00 = 0.00 u
- **haar k=0 J=2 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 2.00000e-15；N=1；实测 0.00e+00 = 0.00 u
- **haar k=0 J=3 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 2.00000e-15；N=1；实测 0.00e+00 = 0.00 u
- **haar k=0 J=4 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 2.00000e-15；N=1；实测 0.00e+00 = 0.00 u
- **haar k=N=1 的残差远高于地板（下界）**：k<N 的最大读数 0.00e+00（0.00 u），k=N 在 J=1 处 3.92e-03（35322350018592 u）
- **haar k=N 残差第 2 层 / 第 1 层 = 2^N**：|实测−理论| = 0.0158103，容差 rtol=0.3, atol=0；实测 2.016 倍，理论 2^1=2
- **db2 k=0 J=1 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.96075e-15；N=2；实测 3.93e-17 = 0.35 u
- **db2 k=0 J=2 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.96075e-15；N=2；实测 3.93e-17 = 0.35 u
- **db2 k=0 J=3 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.96075e-15；N=2；实测 3.93e-17 = 0.35 u
- **db2 k=0 J=4 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.96075e-15；N=2；实测 3.93e-17 = 0.35 u
- **db2 k=1 J=1 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.92110e-15；N=2；实测 7.89e-17 = 0.71 u
- **db2 k=1 J=2 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.88730e-15；N=2；实测 1.13e-16 = 1.02 u
- **db2 k=1 J=3 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.83735e-15；N=2；实测 1.63e-16 = 1.47 u
- **db2 k=1 J=4 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.83735e-15；N=2；实测 1.63e-16 = 1.47 u
- **db2 k=N=2 的残差远高于地板（下界）**：k<N 的最大读数 1.63e-16（1.47 u），k=N 在 J=1 处 5.34e-05（480853973452 u）
- **db2 k=N 残差第 2 层 / 第 1 层 = 2^N**：|实测−理论| = 0.0808465，容差 rtol=0.3, atol=0；实测 4.081 倍，理论 2^2=4
- **db3 k=0 J=1 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 2.00000e-15；N=3；实测 0.00e+00 = 0.00 u
- **db3 k=0 J=2 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 2.00000e-15；N=3；实测 0.00e+00 = 0.00 u
- **db3 k=0 J=3 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 2.00000e-15；N=3；实测 0.00e+00 = 0.00 u
- **db3 k=0 J=4 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 2.00000e-15；N=3；实测 0.00e+00 = 0.00 u
- **db3 k=1 J=1 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.92099e-15；N=3；实测 7.90e-17 = 0.71 u
- **db3 k=1 J=2 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.83021e-15；N=3；实测 1.70e-16 = 1.53 u
- **db3 k=1 J=3 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.83021e-15；N=3；实测 1.70e-16 = 1.53 u
- **db3 k=1 J=4 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.83021e-15；N=3；实测 1.70e-16 = 1.53 u
- **db3 k=2 J=1 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.92048e-15；N=3；实测 7.95e-17 = 0.72 u
- **db3 k=2 J=2 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.88460e-15；N=3；实测 1.15e-16 = 1.04 u
- **db3 k=2 J=3 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.88460e-15；N=3；实测 1.15e-16 = 1.04 u
- **db3 k=2 J=4 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.88460e-15；N=3；实测 1.15e-16 = 1.04 u
- **db3 k=N=3 的残差远高于地板（下界）**：k<N 的最大读数 1.70e-16（1.53 u），k=N 在 J=1 处 1.15e-06（10384074604 u）
- **db3 k=N 残差第 2 层 / 第 1 层 = 2^N**：|实测−理论| = 0.316576，容差 rtol=0.3, atol=0；实测 8.317 倍，理论 2^3=8
- **db4 k=0 J=1 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 2.00000e-15；N=4；实测 0.00e+00 = 0.00 u
- **db4 k=0 J=2 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 2.00000e-15；N=4；实测 0.00e+00 = 0.00 u
- **db4 k=0 J=3 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 2.00000e-15；N=4；实测 0.00e+00 = 0.00 u
- **db4 k=0 J=4 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 2.00000e-15；N=4；实测 0.00e+00 = 0.00 u
- **db4 k=1 J=1 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.92087e-15；N=4；实测 7.91e-17 = 0.71 u
- **db4 k=1 J=2 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.88630e-15；N=4；实测 1.14e-16 = 1.02 u
- **db4 k=1 J=3 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.79232e-15；N=4；实测 2.08e-16 = 1.87 u
- **db4 k=1 J=4 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.79232e-15；N=4；实测 2.08e-16 = 1.87 u
- **db4 k=2 J=1 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.92025e-15；N=4；实测 7.98e-17 = 0.72 u
- **db4 k=2 J=2 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.92025e-15；N=4；实测 7.98e-17 = 0.72 u
- **db4 k=2 J=3 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.91209e-15；N=4；实测 8.79e-17 = 0.79 u
- **db4 k=2 J=4 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.91209e-15；N=4；实测 8.79e-17 = 0.79 u
- **db4 k=3 J=1 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.91962e-15；N=4；实测 8.04e-17 = 0.72 u
- **db4 k=3 J=2 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.91962e-15；N=4；实测 8.04e-17 = 0.72 u
- **db4 k=3 J=3 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.90698e-15；N=4；实测 9.30e-17 = 0.84 u
- **db4 k=3 J=4 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.90698e-15；N=4；实测 9.30e-17 = 0.84 u
- **db4 k=N=4 的残差远高于地板（下界）**：k<N 的最大读数 2.08e-16（1.87 u），k=N 在 J=1 处 3.41e-08（307307595 u）
- **db4 k=N 残差第 2 层 / 第 1 层 = 2^N**：|实测−理论| = 1.05456，容差 rtol=0.3, atol=0；实测 17.055 倍，理论 2^4=16
- **db5 k=0 J=1 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 2.00000e-15；N=5；实测 0.00e+00 = 0.00 u
- **db5 k=0 J=2 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.94449e-15；N=5；实测 5.55e-17 = 0.50 u
- **db5 k=0 J=3 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.94449e-15；N=5；实测 5.55e-17 = 0.50 u
- **db5 k=0 J=4 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.94449e-15；N=5；实测 5.55e-17 = 0.50 u
- **db5 k=1 J=1 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.92076e-15；N=5；实测 7.92e-17 = 0.71 u
- **db5 k=1 J=2 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.82867e-15；N=5；实测 1.71e-16 = 1.54 u
- **db5 k=1 J=3 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.82867e-15；N=5；实测 1.71e-16 = 1.54 u
- **db5 k=1 J=4 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.82867e-15；N=5；实测 1.71e-16 = 1.54 u
- **db5 k=2 J=1 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.92001e-15；N=5；实测 8.00e-17 = 0.72 u
- **db5 k=2 J=2 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.92001e-15；N=5；实测 8.00e-17 = 0.72 u
- **db5 k=2 J=3 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.91014e-15；N=5；实测 8.99e-17 = 0.81 u
- **db5 k=2 J=4 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.91014e-15；N=5；实测 8.99e-17 = 0.81 u
- **db5 k=3 J=1 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.91926e-15；N=5；实测 8.07e-17 = 0.73 u
- **db5 k=3 J=2 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.91926e-15；N=5；实测 8.07e-17 = 0.73 u
- **db5 k=3 J=3 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.91926e-15；N=5；实测 8.07e-17 = 0.73 u
- **db5 k=3 J=4 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.91926e-15；N=5；实测 8.07e-17 = 0.73 u
- **db5 k=4 J=1 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.95925e-15；N=5；实测 4.08e-17 = 0.37 u
- **db5 k=4 J=2 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.90672e-15；N=5；实测 9.33e-17 = 0.84 u
- **db5 k=4 J=3 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.90672e-15；N=5；实测 9.33e-17 = 0.84 u
- **db5 k=4 J=4 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.90672e-15；N=5；实测 9.33e-17 = 0.84 u
- **db5 k=N=5 的残差远高于地板（下界）**：k<N 的最大读数 1.71e-16（1.54 u），k=N 在 J=1 处 1.28e-09（11563889 u）
- **db5 k=N 残差第 2 层 / 第 1 层 = 2^N**：|实测−理论| = 3.18959，容差 rtol=0.3, atol=0；实测 35.190 倍，理论 2^5=32
- **db6 k=0 J=1 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.92150e-15；N=6；实测 7.85e-17 = 0.71 u
- **db6 k=0 J=2 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.92150e-15；N=6；实测 7.85e-17 = 0.71 u
- **db6 k=0 J=3 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.92150e-15；N=6；实测 7.85e-17 = 0.71 u
- **db6 k=0 J=4 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.92150e-15；N=6；实测 7.85e-17 = 0.71 u
- **db6 k=1 J=1 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.92064e-15；N=6；实测 7.94e-17 = 0.71 u
- **db6 k=1 J=2 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.82789e-15；N=6；实测 1.72e-16 = 1.55 u
- **db6 k=1 J=3 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.82789e-15；N=6；实测 1.72e-16 = 1.55 u
- **db6 k=1 J=4 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.82789e-15；N=6；实测 1.72e-16 = 1.55 u
- **db6 k=2 J=1 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.91977e-15；N=6；实测 8.02e-17 = 0.72 u
- **db6 k=2 J=2 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.91977e-15；N=6；实测 8.02e-17 = 0.72 u
- **db6 k=2 J=3 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.91977e-15；N=6；实测 8.02e-17 = 0.72 u
- **db6 k=2 J=4 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.91977e-15；N=6；实测 8.02e-17 = 0.72 u
- **db6 k=3 J=1 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.91890e-15；N=6；实测 8.11e-17 = 0.73 u
- **db6 k=3 J=2 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.91890e-15；N=6；实测 8.11e-17 = 0.73 u
- **db6 k=3 J=3 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.91890e-15；N=6；实测 8.11e-17 = 0.73 u
- **db6 k=3 J=4 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.91890e-15；N=6；实测 8.11e-17 = 0.73 u
- **db6 k=4 J=1 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.95901e-15；N=6；实测 4.10e-17 = 0.37 u
- **db6 k=4 J=2 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.95901e-15；N=6；实测 4.10e-17 = 0.37 u
- **db6 k=4 J=3 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.95901e-15；N=6；实测 4.10e-17 = 0.37 u
- **db6 k=4 J=4 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.95901e-15；N=6；实测 4.10e-17 = 0.37 u
- **db6 k=5 J=1 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.91712e-15；N=6；实测 8.29e-17 = 0.75 u
- **db6 k=5 J=2 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.91712e-15；N=6；实测 8.29e-17 = 0.75 u
- **db6 k=5 J=3 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.91712e-15；N=6；实测 8.29e-17 = 0.75 u
- **db6 k=5 J=4 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.91712e-15；N=6；实测 8.29e-17 = 0.75 u
- **db6 k=N=6 的残差远高于地板（下界）**：k<N 的最大读数 1.72e-16（1.55 u），k=N 在 J=1 处 5.87e-11（528555 u）
- **db6 k=N 残差第 2 层 / 第 1 层 = 2^N**：|实测−理论| = 9.05737，容差 rtol=0.3, atol=0；实测 73.057 倍，理论 2^6=64
- **db7 k=0 J=1 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.96075e-15；N=7；实测 3.93e-17 = 0.35 u
- **db7 k=0 J=2 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.96075e-15；N=7；实测 3.93e-17 = 0.35 u
- **db7 k=0 J=3 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.96075e-15；N=7；实测 3.93e-17 = 0.35 u
- **db7 k=0 J=4 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.94449e-15；N=7；实测 5.55e-17 = 0.50 u
- **db7 k=1 J=1 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.94039e-15；N=7；实测 5.96e-17 = 0.54 u
- **db7 k=1 J=2 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.94039e-15；N=7；实测 5.96e-17 = 0.54 u
- **db7 k=1 J=3 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.94039e-15；N=7；实测 5.96e-17 = 0.54 u
- **db7 k=1 J=4 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.94039e-15；N=7；实测 5.96e-17 = 0.54 u
- **db7 k=2 J=1 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.93965e-15；N=7；实测 6.03e-17 = 0.54 u
- **db7 k=2 J=2 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.88033e-15；N=7；实测 1.20e-16 = 1.08 u
- **db7 k=2 J=3 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.88033e-15；N=7；实测 1.20e-16 = 1.08 u
- **db7 k=2 J=4 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.88033e-15；N=7；实测 1.20e-16 = 1.08 u
- **db7 k=3 J=1 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.95927e-15；N=7；实测 4.07e-17 = 0.37 u
- **db7 k=3 J=2 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.95927e-15；N=7；实测 4.07e-17 = 0.37 u
- **db7 k=3 J=3 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.95504e-15；N=7；实测 4.50e-17 = 0.40 u
- **db7 k=3 J=4 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.95504e-15；N=7；实测 4.50e-17 = 0.40 u
- **db7 k=4 J=1 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.95876e-15；N=7；实测 4.12e-17 = 0.37 u
- **db7 k=4 J=2 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.95876e-15；N=7；实测 4.12e-17 = 0.37 u
- **db7 k=4 J=3 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.95876e-15；N=7；实测 4.12e-17 = 0.37 u
- **db7 k=4 J=4 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.95876e-15；N=7；实测 4.12e-17 = 0.37 u
- **db7 k=5 J=1 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.95825e-15；N=7；实测 4.18e-17 = 0.38 u
- **db7 k=5 J=2 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.94979e-15；N=7；实测 5.02e-17 = 0.45 u
- **db7 k=5 J=3 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.94979e-15；N=7；实测 5.02e-17 = 0.45 u
- **db7 k=5 J=4 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.94979e-15；N=7；实测 5.02e-17 = 0.45 u
- **db7 k=6 J=1 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.95773e-15；N=7；实测 4.23e-17 = 0.38 u
- **db7 k=6 J=2 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.95773e-15；N=7；实测 4.23e-17 = 0.38 u
- **db7 k=6 J=3 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.95773e-15；N=7；实测 4.23e-17 = 0.38 u
- **db7 k=6 J=4 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.95773e-15；N=7；实测 4.23e-17 = 0.38 u
- **db7 k=N=7 的残差远高于地板（下界）**：k<N 的最大读数 1.20e-16（1.08 u），k=N 在 J=1 处 3.16e-12（28452 u）
- **db7 k=N 残差第 2 层 / 第 1 层 = 2^N**：|实测−理论| = 24.6125，容差 rtol=0.3, atol=0；实测 152.612 倍，理论 2^7=128
- **db8 k=0 J=1 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.97056e-15；N=8；实测 2.94e-17 = 0.27 u
- **db8 k=0 J=2 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.89939e-15；N=8；实测 1.01e-16 = 0.91 u
- **db8 k=0 J=3 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.89939e-15；N=8；实测 1.01e-16 = 0.91 u
- **db8 k=0 J=4 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.89939e-15；N=8；实测 1.01e-16 = 0.91 u
- **db8 k=1 J=1 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.86070e-15；N=8；实测 1.39e-16 = 1.25 u
- **db8 k=1 J=2 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.86070e-15；N=8；实测 1.39e-16 = 1.25 u
- **db8 k=1 J=3 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.86070e-15；N=8；实测 1.39e-16 = 1.25 u
- **db8 k=1 J=4 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.86070e-15；N=8；实测 1.39e-16 = 1.25 u
- **db8 k=2 J=1 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.90921e-15；N=8；实测 9.08e-17 = 0.82 u
- **db8 k=2 J=2 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.89810e-15；N=8；实测 1.02e-16 = 0.92 u
- **db8 k=2 J=3 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.89810e-15；N=8；实测 1.02e-16 = 0.92 u
- **db8 k=2 J=4 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.89810e-15；N=8；实测 1.02e-16 = 0.92 u
- **db8 k=3 J=1 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.92840e-15；N=8；实测 7.16e-17 = 0.64 u
- **db8 k=3 J=2 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.88192e-15；N=8；实测 1.18e-16 = 1.06 u
- **db8 k=3 J=3 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.88192e-15；N=8；实测 1.18e-16 = 1.06 u
- **db8 k=3 J=4 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.88192e-15；N=8；实测 1.18e-16 = 1.06 u
- **db8 k=4 J=1 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.94944e-15；N=8；实测 5.06e-17 = 0.46 u
- **db8 k=4 J=2 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.94944e-15；N=8；实测 5.06e-17 = 0.46 u
- **db8 k=4 J=3 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.94944e-15；N=8；实测 5.06e-17 = 0.46 u
- **db8 k=4 J=4 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.94944e-15；N=8；实测 5.06e-17 = 0.46 u
- **db8 k=5 J=1 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.93165e-15；N=8；实测 6.83e-17 = 0.62 u
- **db8 k=5 J=2 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.93150e-15；N=8；实测 6.85e-17 = 0.62 u
- **db8 k=5 J=3 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.93150e-15；N=8；实测 6.85e-17 = 0.62 u
- **db8 k=5 J=4 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.93150e-15；N=8；实测 6.85e-17 = 0.62 u
- **db8 k=6 J=1 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.95202e-15；N=8；实测 4.80e-17 = 0.43 u
- **db8 k=6 J=2 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.95202e-15；N=8；实测 4.80e-17 = 0.43 u
- **db8 k=6 J=3 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.95202e-15；N=8；实测 4.80e-17 = 0.43 u
- **db8 k=6 J=4 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.95202e-15；N=8；实测 4.80e-17 = 0.43 u
- **db8 k=7 J=1 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.94055e-15；N=8；实测 5.95e-17 = 0.54 u
- **db8 k=7 J=2 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.94055e-15；N=8；实测 5.95e-17 = 0.54 u
- **db8 k=7 J=3 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.94055e-15；N=8；实测 5.95e-17 = 0.54 u
- **db8 k=7 J=4 内部比值（应贴 u）**：松弛 rtol=1, atol=0，余量 = 1.94055e-15；N=8；实测 5.95e-17 = 0.54 u
- **db8 k=N=8 的残差远高于地板（下界）**：k<N 的最大读数 1.39e-16（1.25 u），k=N 在 J=1 处 1.96e-13（1764 u）
- **db8 k=N 残差第 2 层 / 第 1 层 = 2^N**：|实测−理论| = 64.7754，容差 rtol=0.3, atol=0；实测 320.775 倍，理论 2^8=256
- **db4 k=4 残差随 n 按 n^N 衰减**：|实测−理论| = 25.8076，容差 rtol=0.2, atol=0；实测 281.8 倍（n 翻两番 = 2^8 = 256）
- **db4 被抹平的比值与 u 之比**：对数轴上它落在 u 参考线附近，而不是 -inf；同一读数在 J=1..4、n=256 上逐层不变
- **haar 逐层内部比值（k=N-1）**：松弛 rtol=1, atol=0，余量 = 2.00000e-15；各层都贴 u，不是只有第 1 层
- **haar 有内部系数可测的层数**：各层内部系数个数 [126, 62, 30, 14]（0 表示该层没有不碰延拓区的系数）
- **haar 逐层内部比值（k=N）显著高于地板（下界）**：有内部系数的层读数 3.92e-03、7.91e-03、1.61e-02、3.32e-02（逐层上升；db2/db4 的 J=1 读数可以低到 1e-13）
- **db4 逐层内部比值（k=N-1）**：松弛 rtol=1, atol=0，余量 = 1.90698e-15；各层都贴 u，不是只有第 1 层
- **db4 有内部系数可测的层数**：各层内部系数个数 [120, 56, 24, 8]（0 表示该层没有不碰延拓区的系数）
- **db4 逐层内部比值（k=N）显著高于地板（下界）**：有内部系数的层读数 3.41e-08、5.82e-07、1.06e-05、2.23e-04（逐层上升；db2/db4 的 J=1 读数可以低到 1e-13）
- **db8 逐层内部比值（k=N-1）**：松弛 rtol=1, atol=0，余量 = 1.94055e-15；各层都贴 u，不是只有第 1 层
- **db8 有内部系数可测的层数**：各层内部系数个数 [112, 48, 16, 0]（0 表示该层没有不碰延拓区的系数）
- **db8 逐层内部比值（k=N）显著高于地板（下界）**：有内部系数的层读数 1.96e-13、6.28e-11、2.57e-08（逐层上升；db2/db4 的 J=1 读数可以低到 1e-13）
- **边界系数（同一条多项式）实测 O(1)（下界）**：实测边界比值 0.954——只看 max|d| 会误判消失矩失效
- **MSE = bias^2 + var（逐点恒等式）**：松弛 rtol=1, atol=0，余量 = 1.97224e-15；最大偏差 2.78e-17
- **软阈值 偏差^2 随 lambda 单调不减**：最大相邻增量 0.00799872，松弛 5.89741e-15（machine:33·u·scale）
- **软阈值 方差随 lambda 单调不增**：最大相邻增量 -2.40285e-04，松弛 2.54332e-15（machine:33·u·scale）
- **抛物线细化的最优点不差于栅格最优点**：松弛 rtol=0, atol=0，余量 = 9.99201e-16
- **硬阈值 偏差^2 的非单调段**：33 个阈值里有 2 处偏差^2 下降；软阈值处为 0 处
- **最优点不差于不去噪（lambda=0）**：松弛 rtol=0, atol=0，余量 = 0.042305；最优 0.04733 vs 不去噪 0.08963
- **实测最优阈值 <= 通用阈值**：松弛 rtol=0, atol=0，余量 = 0.718111；0.2810 <= 0.9991（比值 3.56）
- **通用阈值处的 MSE（与不去噪对照）**：通用阈值 0.999 处 MSE 0.12994，不去噪 0.08963（差 45.0%）
- **lambda=0.3 处软阈值 MSE <= 硬阈值**：松弛 rtol=0, atol=0，余量 = 0.0488742；软 0.04117、硬 0.09004、不去噪 0.09806
- **各自最优点上软阈值更优**：松弛 rtol=0, atol=0，余量 = 0.00645632；软 0.04733（lam=0.281） vs 硬 0.05379（lam=0.765）
- **软/硬最优阈值之比**：硬 0.765 / 软 0.281
- **haar/J=1 lambda=0 恒等**：松弛 rtol=1, atol=0，余量 = 1.37668e-14；阈值算子的零点锚
- **haar/J=3 lambda=0 恒等**：松弛 rtol=1, atol=0，余量 = 4.17444e-14；阈值算子的零点锚
- **haar/J=4 lambda=0 恒等**：松弛 rtol=1, atol=0，余量 = 5.57332e-14；阈值算子的零点锚
- **db4/J=1 lambda=0 恒等**：松弛 rtol=1, atol=0，余量 = 1.35447e-14；阈值算子的零点锚
- **db4/J=3 lambda=0 恒等**：松弛 rtol=1, atol=0，余量 = 4.21885e-14；阈值算子的零点锚
- **db4/J=4 lambda=0 恒等**：松弛 rtol=1, atol=0，余量 = 5.57332e-14；阈值算子的零点锚
- **通用阈值随 sigma_hat 线性缩放**：|实测−理论| = 2.22045e-16，容差 rtol=1e-12, atol=0
- **单次 MAD 读数在真值 30% 以内**：松弛 rtol=1, atol=0，余量 = 0.533307；它的标准差约 9%，30% 是 3 sigma 的宽松界
- **阈值只作用于细节系数（最粗近似原样保留）**：|实测−理论| = 0，容差 rtol=0, atol=0
- **SNR 增益（单次实现）**：含噪 SNR 22.10 dB -> 去噪后 24.13 dB
- **w=16 峰值定位误差 <= 半格**：松弛 rtol=1, atol=0，余量 = 0.0429688；实测 0.01953，半格 0.03125；时间栅格 61 个窗
- **w=32 峰值定位误差 <= 半格**：松弛 rtol=1, atol=0，余量 = 0.0195313；实测 0.01172，半格 0.01562；时间栅格 29 个窗
- **w=64 峰值定位误差 <= 半格**：松弛 rtol=1, atol=0，余量 = 0.0117188；实测 0.00391，半格 0.00781；时间栅格 13 个窗
- **峰值定位误差随窗长单调下降**：最大相邻增量 -0.0078125，松弛 5.20417e-17（machine:3·u·scale）；0.0195 -> 0.0117 -> 0.0039
- **低频音在 w=32 时的峰值误差（反面）**：峰落在 f=0.00000（真值 0.01953），误差 0.01953——固定窗长在低频处就是这样骗人的
- **第 1 层时间支撑 = L-1+1 = 8 个样本**：|实测−理论| = 0，容差 rtol=0, atol=0
- **小波时间支撑随层数增长**：最大相邻增量 28，松弛 2.02505e-13（machine:4·u·scale）
- **分辨率元面积极差（近似守恒）**：松弛 rtol=0, atol=0，余量 = 0.077193；max/min = 1.123（0.01392 .. 0.01562）
- **STFT 时间窗宽恒定**：|实测−理论| = 0，容差 rtol=0, atol=0
- **STFT 带宽恒定**：|实测−理论| = 0，容差 rtol=0, atol=0
- **两种分辨率元的面积对照**：小波面积极差 1.123；STFT 面积 = 0.0078（恒定）
- **bumps 两次调用逐位相同**：|实测−理论| = 0，容差 rtol=0, atol=0
- **bumps 全部有限**：松弛 rtol=0, atol=0，余量 = 0
- **bumps/haar 往返误差**：松弛 rtol=1, atol=0，余量 = 1.98224e-13；四条路径都要能还原
- **bumps/db4 往返误差**：松弛 rtol=1, atol=0，余量 = 1.98668e-13；四条路径都要能还原
- **bumps/db8 往返误差**：松弛 rtol=1, atol=0，余量 = 1.98668e-13；四条路径都要能还原
- **bumps SWT 往返误差**：松弛 rtol=1, atol=0，余量 = 1.99112e-13
- **blocks 两次调用逐位相同**：|实测−理论| = 0，容差 rtol=0, atol=0
- **blocks 全部有限**：松弛 rtol=0, atol=0，余量 = 0
- **blocks/haar 往返误差**：松弛 rtol=1, atol=0，余量 = 1.97335e-13；四条路径都要能还原
- **blocks/db4 往返误差**：松弛 rtol=1, atol=0，余量 = 1.94671e-13；四条路径都要能还原
- **blocks/db8 往返误差**：松弛 rtol=1, atol=0，余量 = 1.94671e-13；四条路径都要能还原
- **blocks SWT 往返误差**：松弛 rtol=1, atol=0，余量 = 1.97335e-13
- **heavisine 两次调用逐位相同**：|实测−理论| = 0，容差 rtol=0, atol=0
- **heavisine 全部有限**：松弛 rtol=0, atol=0，余量 = 0
- **heavisine/haar 往返误差**：松弛 rtol=1, atol=0，余量 = 1.97780e-13；四条路径都要能还原
- **heavisine/db4 往返误差**：松弛 rtol=1, atol=0，余量 = 1.98224e-13；四条路径都要能还原
- **heavisine/db8 往返误差**：松弛 rtol=1, atol=0，余量 = 1.97780e-13；四条路径都要能还原
- **heavisine SWT 往返误差**：松弛 rtol=1, atol=0，余量 = 1.98890e-13
- **doppler 两次调用逐位相同**：|实测−理论| = 0，容差 rtol=0, atol=0
- **doppler 全部有限**：松弛 rtol=0, atol=0，余量 = 0
- **doppler/haar 往返误差**：松弛 rtol=1, atol=0，余量 = 1.99667e-13；四条路径都要能还原
- **doppler/db4 往返误差**：松弛 rtol=1, atol=0，余量 = 1.99667e-13；四条路径都要能还原
- **doppler/db8 往返误差**：松弛 rtol=1, atol=0，余量 = 1.99611e-13；四条路径都要能还原
- **doppler SWT 往返误差**：松弛 rtol=1, atol=0，余量 = 1.99778e-13
- **tone 两次调用逐位相同**：|实测−理论| = 0，容差 rtol=0, atol=0
- **tone 全部有限**：松弛 rtol=0, atol=0，余量 = 0
- **tone/haar 往返误差**：松弛 rtol=1, atol=0，余量 = 1.98890e-13；四条路径都要能还原
- **tone/db4 往返误差**：松弛 rtol=1, atol=0，余量 = 1.98890e-13；四条路径都要能还原
- **tone/db8 往返误差**：松弛 rtol=1, atol=0，余量 = 1.98890e-13；四条路径都要能还原
- **tone SWT 往返误差**：松弛 rtol=1, atol=0，余量 = 1.99334e-13
- **chirp 两次调用逐位相同**：|实测−理论| = 0，容差 rtol=0, atol=0
- **chirp 全部有限**：松弛 rtol=0, atol=0，余量 = 0
- **chirp/haar 往返误差**：松弛 rtol=1, atol=0，余量 = 1.99112e-13；四条路径都要能还原
- **chirp/db4 往返误差**：松弛 rtol=1, atol=0，余量 = 1.99334e-13；四条路径都要能还原
- **chirp/db8 往返误差**：松弛 rtol=1, atol=0，余量 = 1.99112e-13；四条路径都要能还原
- **chirp SWT 往返误差**：松弛 rtol=1, atol=0，余量 = 1.99556e-13
- **六个基准信号的取值范围**：bumps 4.44、blocks 6、heavisine 2.27、doppler 0.5、tone 1.5、chirp 1
- **各层细节/近似量级比有界**：松弛 rtol=0, atol=0，余量 = 8.98985；bumps 的比值在 1 附近，不会因为换层数而爆炸
- **用原信号做分母 vs 用同层近似（层 4）**：同层 1.0101 vs 原信号 0.98581，比 1.0
- **haar 正交性缺陷**：松弛 rtol=1, atol=0，余量 = 1.55431e-15；sum_j h_j h_{j-2k} = delta_k（定义式）
- **haar 归一化 |sum h - sqrt2|/sqrt2**：松弛 rtol=1, atol=0，余量 = 1.77636e-15
- **haar 完美重构条件**：松弛 rtol=1, atol=0，余量 = 5.77316e-15；|H(w)|^2 + |H(w+pi)|^2 = 2，512 点扫描
- **haar 前 1 阶消失矩（相对缺陷）**：松弛 rtol=1, atol=0，余量 = 2.00000e-14；m < N 时 sum_j (-1)^j j^m h_j = 0
- **db2 正交性缺陷**：松弛 rtol=1, atol=0，余量 = 1.76248e-15；sum_j h_j h_{j-2k} = delta_k（定义式）
- **db2 归一化 |sum h - sqrt2|/sqrt2**：松弛 rtol=1, atol=0，余量 = 1.77636e-15
- **db2 完美重构条件**：松弛 rtol=1, atol=0，余量 = 5.55112e-15；|H(w)|^2 + |H(w+pi)|^2 = 2，512 点扫描
- **db2 前 2 阶消失矩（相对缺陷）**：松弛 rtol=1, atol=0，余量 = 1.99668e-14；m < N 时 sum_j (-1)^j j^m h_j = 0
- **db3 正交性缺陷**：松弛 rtol=1, atol=0，余量 = 1.75554e-15；sum_j h_j h_{j-2k} = delta_k（定义式）
- **db3 归一化 |sum h - sqrt2|/sqrt2**：松弛 rtol=1, atol=0，余量 = 1.61935e-15
- **db3 完美重构条件**：松弛 rtol=1, atol=0，余量 = 5.10703e-15；|H(w)|^2 + |H(w+pi)|^2 = 2，512 点扫描
- **db3 前 3 阶消失矩（相对缺陷）**：松弛 rtol=1, atol=0，余量 = 1.99818e-14；m < N 时 sum_j (-1)^j j^m h_j = 0
- **db4 正交性缺陷**：松弛 rtol=1, atol=0，余量 = 1.76248e-15；sum_j h_j h_{j-2k} = delta_k（定义式）
- **db4 归一化 |sum h - sqrt2|/sqrt2**：松弛 rtol=1, atol=0，余量 = 1.77636e-15
- **db4 完美重构条件**：松弛 rtol=1, atol=0，余量 = 5.32907e-15；|H(w)|^2 + |H(w+pi)|^2 = 2，512 点扫描
- **db4 前 4 阶消失矩（相对缺陷）**：松弛 rtol=1, atol=0，余量 = 1.99732e-14；m < N 时 sum_j (-1)^j j^m h_j = 0
- **db5 正交性缺陷**：松弛 rtol=1, atol=0，余量 = 1.74860e-15；sum_j h_j h_{j-2k} = delta_k（定义式）
- **db5 归一化 |sum h - sqrt2|/sqrt2**：松弛 rtol=1, atol=0，余量 = 1.77636e-15
- **db5 完美重构条件**：松弛 rtol=1, atol=0，余量 = 4.88498e-15；|H(w)|^2 + |H(w+pi)|^2 = 2，512 点扫描
- **db5 前 5 阶消失矩（相对缺陷）**：松弛 rtol=1, atol=0，余量 = 1.99613e-14；m < N 时 sum_j (-1)^j j^m h_j = 0
- **db6 正交性缺陷**：松弛 rtol=1, atol=0，余量 = 1.74860e-15；sum_j h_j h_{j-2k} = delta_k（定义式）
- **db6 归一化 |sum h - sqrt2|/sqrt2**：松弛 rtol=1, atol=0，余量 = 1.61935e-15
- **db6 完美重构条件**：松弛 rtol=1, atol=0，余量 = 3.55271e-15；|H(w)|^2 + |H(w+pi)|^2 = 2，512 点扫描
- **db6 前 6 阶消失矩（相对缺陷）**：松弛 rtol=1, atol=0，余量 = 1.99641e-14；m < N 时 sum_j (-1)^j j^m h_j = 0
- **db7 正交性缺陷**：松弛 rtol=1, atol=0，余量 = 1.66533e-15；sum_j h_j h_{j-2k} = delta_k（定义式）
- **db7 归一化 |sum h - sqrt2|/sqrt2**：松弛 rtol=1, atol=0，余量 = 1.77636e-15
- **db7 完美重构条件**：松弛 rtol=1, atol=0，余量 = 2.66454e-15；|H(w)|^2 + |H(w+pi)|^2 = 2，512 点扫描
- **db7 前 7 阶消失矩（相对缺陷）**：松弛 rtol=1, atol=0，余量 = 1.99517e-14；m < N 时 sum_j (-1)^j j^m h_j = 0
- **db8 正交性缺陷**：松弛 rtol=1, atol=0，余量 = 1.66533e-15；sum_j h_j h_{j-2k} = delta_k（定义式）
- **db8 归一化 |sum h - sqrt2|/sqrt2**：松弛 rtol=1, atol=0，余量 = 1.77636e-15
- **db8 完美重构条件**：松弛 rtol=1, atol=0，余量 = 2.88658e-15；|H(w)|^2 + |H(w+pi)|^2 = 2，512 点扫描
- **db8 前 8 阶消失矩（相对缺陷）**：松弛 rtol=1, atol=0，余量 = 1.99113e-14；m < N 时 sum_j (-1)^j j^m h_j = 0
- **最坏正交性缺陷**：归一化 1.6e-16、重构条件 4.4e-15、消失矩 8.9e-17
- **m = N 处消失矩不再成立（下界）**：各族 m=N 的相对缺陷：haar=1.0e+00、db2=4.2e-01、db3=1.9e-01、db4=8.5e-02、db5=3.5e-02、db6=1.3e-02、db7=4.7e-03、db8=1.8e-03
- **db2 J=1 周期模式系数个数 = n**：|实测−理论| = 0，容差 rtol=0, atol=0
- **db2 J=1 正交性缺陷**：松弛 rtol=1, atol=0，余量 = 1.41860e-14；M M^T 与 M^T M 两个方向都查
- **db2 J=2 周期模式系数个数 = n**：|实测−理论| = 0，容差 rtol=0, atol=0
- **db2 J=2 正交性缺陷**：松弛 rtol=1, atol=0，余量 = 1.39888e-14；M M^T 与 M^T M 两个方向都查
- **db2 J=4 周期模式系数个数 = n**：|实测−理论| = 0，容差 rtol=0, atol=0
- **db2 J=4 正交性缺陷**：松弛 rtol=1, atol=0，余量 = 1.39888e-14；M M^T 与 M^T M 两个方向都查
- **db4 J=1 周期模式系数个数 = n**：|实测−理论| = 0，容差 rtol=0, atol=0
- **db4 J=1 正交性缺陷**：松弛 rtol=1, atol=0，余量 = 1.39888e-14；M M^T 与 M^T M 两个方向都查
- **db4 J=2 周期模式系数个数 = n**：|实测−理论| = 0，容差 rtol=0, atol=0
- **db4 J=2 正交性缺陷**：松弛 rtol=1, atol=0，余量 = 1.37668e-14；M M^T 与 M^T M 两个方向都查
- **db4 J=4 周期模式系数个数 = n**：|实测−理论| = 0，容差 rtol=0, atol=0
- **db4 J=4 正交性缺陷**：松弛 rtol=1, atol=0，余量 = 1.37668e-14；M M^T 与 M^T M 两个方向都查
- **db8 J=1 周期模式系数个数 = n**：|实测−理论| = 0，容差 rtol=0, atol=0
- **db8 J=1 正交性缺陷**：松弛 rtol=1, atol=0，余量 = 1.39888e-14；M M^T 与 M^T M 两个方向都查
- **db8 J=2 周期模式系数个数 = n**：|实测−理论| = 0，容差 rtol=0, atol=0
- **db8 J=2 正交性缺陷**：松弛 rtol=1, atol=0，余量 = 1.37668e-14；M M^T 与 M^T M 两个方向都查
- **db8 J=4 周期模式系数个数 = n**：|实测−理论| = 0，容差 rtol=0, atol=0
- **db8 J=4 正交性缺陷**：松弛 rtol=1, atol=0，余量 = 1.33227e-14；M M^T 与 M^T M 两个方向都查
- **zero 模式 M^T M = I（等距嵌入）**：松弛 rtol=1, atol=0，余量 = 1.39888e-14
- **zero 模式 M M^T != I（冗余）**：n_ext=96，多出的 32 个系数不是噪声
- **symmetric 模式 M^T M != I（反射复制样本）**：这就是同一份系数能量比 > 1 的代数原因
- **n=16 periodic 递归 vs 显式矩阵**：松弛 rtol=1, atol=0，余量 = 2.27326e-13；c·n·u，c=64
- **n=16 symmetric 递归 vs 显式矩阵**：松弛 rtol=1, atol=0，余量 = 2.27185e-13；c·n·u，c=64
- **n=16 zero 递归 vs 显式矩阵**：松弛 rtol=1, atol=0，余量 = 2.27374e-13；c·n·u，c=64
- **n=32 periodic 递归 vs 显式矩阵**：松弛 rtol=1, atol=0，余量 = 4.54721e-13；c·n·u，c=64
- **n=32 symmetric 递归 vs 显式矩阵**：松弛 rtol=1, atol=0，余量 = 4.54642e-13；c·n·u，c=64
- **n=32 zero 递归 vs 显式矩阵**：松弛 rtol=1, atol=0，余量 = 4.54747e-13；c·n·u，c=64
- **n=64 periodic 递归 vs 显式矩阵**：松弛 rtol=1, atol=0，余量 = 9.09411e-13；c·n·u，c=64
- **n=64 symmetric 递归 vs 显式矩阵**：松弛 rtol=1, atol=0，余量 = 9.09411e-13；c·n·u，c=64
- **n=64 zero 递归 vs 显式矩阵**：松弛 rtol=1, atol=0，余量 = 9.09495e-13；c·n·u，c=64
- **Haar 闭式解**：近似误差 4.44e-16、细节误差 3.33e-16（n=32）
- **periodic/haar/J=1 往返误差**：松弛 rtol=1, atol=0，余量 = 6.07337e-16；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **periodic/haar/J=2 往返误差**：松弛 rtol=1, atol=0，余量 = 1.23526e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **periodic/haar/J=3 往返误差**：松弛 rtol=1, atol=0，余量 = 1.96243e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **periodic/haar/J=4 往返误差**：松弛 rtol=1, atol=0，余量 = 2.71019e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **periodic/haar/J=6 往返误差**：松弛 rtol=1, atol=0，余量 = 4.48655e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **periodic/haar/J=8 往返误差**：松弛 rtol=1, atol=0，余量 = 6.26290e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **periodic/db2/J=1 往返误差**：松弛 rtol=1, atol=0，余量 = 4.82352e-16；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **periodic/db2/J=2 往返误差**：松弛 rtol=1, atol=0，余量 = 1.37053e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **periodic/db2/J=3 往返误差**：松弛 rtol=1, atol=0，余量 = 2.12343e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **periodic/db2/J=4 往返误差**：松弛 rtol=1, atol=0，余量 = 2.99103e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **periodic/db2/J=6 往返误差**：松弛 rtol=1, atol=0，余量 = 4.76739e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **periodic/db2/J=8 往返误差**：松弛 rtol=1, atol=0，余量 = 6.54374e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **periodic/db4/J=1 往返误差**：松弛 rtol=1, atol=0，余量 = 6.07337e-16；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **periodic/db4/J=2 往返误差**：松弛 rtol=1, atol=0，余量 = 1.35509e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **periodic/db4/J=3 往返误差**：松弛 rtol=1, atol=0，余量 = 1.96243e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **periodic/db4/J=4 往返误差**：松弛 rtol=1, atol=0，余量 = 2.85061e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **periodic/db4/J=6 往返误差**：松弛 rtol=1, atol=0，余量 = 4.62697e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **periodic/db4/J=8 往返误差**：松弛 rtol=1, atol=0，余量 = 6.40332e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **periodic/db8/J=1 往返误差**：松弛 rtol=1, atol=0，余量 = 4.66916e-16；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **periodic/db8/J=2 往返误差**：松弛 rtol=1, atol=0，余量 = 1.09998e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **periodic/db8/J=3 往返误差**：松弛 rtol=1, atol=0，余量 = 1.82201e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **periodic/db8/J=4 往返误差**：松弛 rtol=1, atol=0，余量 = 2.71019e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **periodic/db8/J=6 往返误差**：松弛 rtol=1, atol=0，余量 = 4.48655e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **periodic/db8/J=8 往返误差**：松弛 rtol=1, atol=0，余量 = 6.26290e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **symmetric/haar/J=1 往返误差**：松弛 rtol=1, atol=0，余量 = 6.07337e-16；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **symmetric/haar/J=2 往返误差**：松弛 rtol=1, atol=0，余量 = 1.23526e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **symmetric/haar/J=3 往返误差**：松弛 rtol=1, atol=0，余量 = 1.96243e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **symmetric/haar/J=4 往返误差**：松弛 rtol=1, atol=0，余量 = 2.71019e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **symmetric/haar/J=6 往返误差**：松弛 rtol=1, atol=0，余量 = 4.48655e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **symmetric/haar/J=8 往返误差**：松弛 rtol=1, atol=0，余量 = 6.26290e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **symmetric/db2/J=1 往返误差**：松弛 rtol=1, atol=0，余量 = 4.82352e-16；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **symmetric/db2/J=2 往返误差**：松弛 rtol=1, atol=0，余量 = 1.37053e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **symmetric/db2/J=3 往返误差**：松弛 rtol=1, atol=0，余量 = 2.12343e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **symmetric/db2/J=4 往返误差**：松弛 rtol=1, atol=0，余量 = 2.94397e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **symmetric/db2/J=6 往返误差**：松弛 rtol=1, atol=0，余量 = 4.76739e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **symmetric/db2/J=8 往返误差**：松弛 rtol=1, atol=0，余量 = 6.42905e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **symmetric/db4/J=1 往返误差**：松弛 rtol=1, atol=0，余量 = 6.07337e-16；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **symmetric/db4/J=2 往返误差**：松弛 rtol=1, atol=0，余量 = 1.35509e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **symmetric/db4/J=3 往返误差**：松弛 rtol=1, atol=0，余量 = 1.96243e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **symmetric/db4/J=4 往返误差**：松弛 rtol=1, atol=0，余量 = 2.85061e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **symmetric/db4/J=6 往返误差**：松弛 rtol=1, atol=0，余量 = 4.62697e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **symmetric/db4/J=8 往返误差**：松弛 rtol=1, atol=0，余量 = 6.40332e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **symmetric/db8/J=1 往返误差**：松弛 rtol=1, atol=0，余量 = 4.66916e-16；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **symmetric/db8/J=2 往返误差**：松弛 rtol=1, atol=0，余量 = 1.09998e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **symmetric/db8/J=3 往返误差**：松弛 rtol=1, atol=0，余量 = 1.82201e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **symmetric/db8/J=4 往返误差**：松弛 rtol=1, atol=0，余量 = 2.71019e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **symmetric/db8/J=6 往返误差**：松弛 rtol=1, atol=0，余量 = 4.48655e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **symmetric/db8/J=8 往返误差**：松弛 rtol=1, atol=0，余量 = 6.20449e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **zero/haar/J=1 往返误差**：松弛 rtol=1, atol=0，余量 = 6.07337e-16；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **zero/haar/J=2 往返误差**：松弛 rtol=1, atol=0，余量 = 1.23526e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **zero/haar/J=3 往返误差**：松弛 rtol=1, atol=0，余量 = 1.96243e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **zero/haar/J=4 往返误差**：松弛 rtol=1, atol=0，余量 = 2.71019e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **zero/haar/J=6 往返误差**：松弛 rtol=1, atol=0，余量 = 4.48655e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **zero/haar/J=8 往返误差**：松弛 rtol=1, atol=0，余量 = 6.26290e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **zero/db2/J=1 往返误差**：松弛 rtol=1, atol=0，余量 = 4.82352e-16；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **zero/db2/J=2 往返误差**：松弛 rtol=1, atol=0，余量 = 1.37053e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **zero/db2/J=3 往返误差**：松弛 rtol=1, atol=0，余量 = 2.12343e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **zero/db2/J=4 往返误差**：松弛 rtol=1, atol=0，余量 = 3.01161e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **zero/db2/J=6 往返误差**：松弛 rtol=1, atol=0，余量 = 4.76739e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **zero/db2/J=8 往返误差**：松弛 rtol=1, atol=0，余量 = 6.54374e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **zero/db4/J=1 往返误差**：松弛 rtol=1, atol=0，余量 = 6.07337e-16；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **zero/db4/J=2 往返误差**：松弛 rtol=1, atol=0，余量 = 1.35509e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **zero/db4/J=3 往返误差**：松弛 rtol=1, atol=0，余量 = 1.96243e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **zero/db4/J=4 往返误差**：松弛 rtol=1, atol=0，余量 = 2.85061e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **zero/db4/J=6 往返误差**：松弛 rtol=1, atol=0，余量 = 4.62697e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **zero/db4/J=8 往返误差**：松弛 rtol=1, atol=0，余量 = 6.40332e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **zero/db8/J=1 往返误差**：松弛 rtol=1, atol=0，余量 = 4.66916e-16；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **zero/db8/J=2 往返误差**：松弛 rtol=1, atol=0，余量 = 1.09998e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **zero/db8/J=3 往返误差**：松弛 rtol=1, atol=0，余量 = 1.82201e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **zero/db8/J=4 往返误差**：松弛 rtol=1, atol=0，余量 = 2.42935e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **zero/db8/J=6 往返误差**：松弛 rtol=1, atol=0，余量 = 4.20571e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **zero/db8/J=8 往返误差**：松弛 rtol=1, atol=0，余量 = 5.98206e-15；容差 4·J·u（每层两次滤波，舍入按层数累积）
- **最大往返误差**：出现在 zero/db8/J=4；n=256、三个信号取最大
- **裁到临界采样后的往返误差（负对照）**：裁掉的正是延拓撑出来的系数——边界处理骗人就在这里
- **periodic/haar/J=1 能量比 = 1**：松弛 rtol=1, atol=0，余量 = 4.54525e-13；正交（或等距）变换的 Parseval 等式
- **periodic/haar/J=1 逐层能量占比求和 = 总能量比**：|实测−理论| = 2.22045e-16，容差 rtol=1e-12, atol=0
- **periodic/haar/J=3 能量比 = 1**：松弛 rtol=1, atol=0，余量 = 4.54525e-13；正交（或等距）变换的 Parseval 等式
- **periodic/haar/J=3 逐层能量占比求和 = 总能量比**：|实测−理论| = 2.22045e-16，容差 rtol=1e-12, atol=0
- **periodic/haar/J=4 能量比 = 1**：松弛 rtol=1, atol=0，余量 = 4.54525e-13；正交（或等距）变换的 Parseval 等式
- **periodic/haar/J=4 逐层能量占比求和 = 总能量比**：|实测−理论| = 0，容差 rtol=1e-12, atol=0
- **periodic/db2/J=1 能量比 = 1**：松弛 rtol=1, atol=0，余量 = 4.54747e-13；正交（或等距）变换的 Parseval 等式
- **periodic/db2/J=1 逐层能量占比求和 = 总能量比**：|实测−理论| = 2.22045e-16，容差 rtol=1e-12, atol=0
- **periodic/db2/J=3 能量比 = 1**：松弛 rtol=1, atol=0，余量 = 4.54525e-13；正交（或等距）变换的 Parseval 等式
- **periodic/db2/J=3 逐层能量占比求和 = 总能量比**：|实测−理论| = 2.22045e-16，容差 rtol=1e-12, atol=0
- **periodic/db2/J=4 能量比 = 1**：松弛 rtol=1, atol=0，余量 = 4.54525e-13；正交（或等距）变换的 Parseval 等式
- **periodic/db2/J=4 逐层能量占比求和 = 总能量比**：|实测−理论| = 0，容差 rtol=1e-12, atol=0
- **periodic/db4/J=1 能量比 = 1**：松弛 rtol=1, atol=0，余量 = 4.54525e-13；正交（或等距）变换的 Parseval 等式
- **periodic/db4/J=1 逐层能量占比求和 = 总能量比**：|实测−理论| = 2.22045e-16，容差 rtol=1e-12, atol=0
- **periodic/db4/J=3 能量比 = 1**：松弛 rtol=1, atol=0，余量 = 4.54747e-13；正交（或等距）变换的 Parseval 等式
- **periodic/db4/J=3 逐层能量占比求和 = 总能量比**：|实测−理论| = 0，容差 rtol=1e-12, atol=0
- **periodic/db4/J=4 能量比 = 1**：松弛 rtol=1, atol=0，余量 = 4.54525e-13；正交（或等距）变换的 Parseval 等式
- **periodic/db4/J=4 逐层能量占比求和 = 总能量比**：|实测−理论| = 2.22045e-16，容差 rtol=1e-12, atol=0
- **periodic/db8/J=1 能量比 = 1**：松弛 rtol=1, atol=0，余量 = 4.54525e-13；正交（或等距）变换的 Parseval 等式
- **periodic/db8/J=1 逐层能量占比求和 = 总能量比**：|实测−理论| = 0，容差 rtol=1e-12, atol=0
- **periodic/db8/J=3 能量比 = 1**：松弛 rtol=1, atol=0，余量 = 4.54747e-13；正交（或等距）变换的 Parseval 等式
- **periodic/db8/J=3 逐层能量占比求和 = 总能量比**：|实测−理论| = 0，容差 rtol=1e-12, atol=0
- **periodic/db8/J=4 能量比 = 1**：松弛 rtol=1, atol=0，余量 = 4.54636e-13；正交（或等距）变换的 Parseval 等式
- **periodic/db8/J=4 逐层能量占比求和 = 总能量比**：|实测−理论| = 1.11022e-16，容差 rtol=1e-12, atol=0
- **zero/haar/J=1 能量比 = 1**：松弛 rtol=1, atol=0，余量 = 4.54747e-13；正交（或等距）变换的 Parseval 等式
- **zero/haar/J=1 逐层能量占比求和 = 总能量比**：|实测−理论| = 0，容差 rtol=1e-12, atol=0
- **zero/haar/J=3 能量比 = 1**：松弛 rtol=1, atol=0，余量 = 4.54303e-13；正交（或等距）变换的 Parseval 等式
- **zero/haar/J=3 逐层能量占比求和 = 总能量比**：|实测−理论| = 0，容差 rtol=1e-12, atol=0
- **zero/haar/J=4 能量比 = 1**：松弛 rtol=1, atol=0，余量 = 4.54747e-13；正交（或等距）变换的 Parseval 等式
- **zero/haar/J=4 逐层能量占比求和 = 总能量比**：|实测−理论| = 2.22045e-16，容差 rtol=1e-12, atol=0
- **zero/db2/J=1 能量比 = 1**：松弛 rtol=1, atol=0，余量 = 4.54747e-13；正交（或等距）变换的 Parseval 等式
- **zero/db2/J=1 逐层能量占比求和 = 总能量比**：|实测−理论| = 4.44089e-16，容差 rtol=1e-12, atol=0
- **zero/db2/J=3 能量比 = 1**：松弛 rtol=1, atol=0，余量 = 4.54747e-13；正交（或等距）变换的 Parseval 等式
- **zero/db2/J=3 逐层能量占比求和 = 总能量比**：|实测−理论| = 0，容差 rtol=1e-12, atol=0
- **zero/db2/J=4 能量比 = 1**：松弛 rtol=1, atol=0，余量 = 4.54525e-13；正交（或等距）变换的 Parseval 等式
- **zero/db2/J=4 逐层能量占比求和 = 总能量比**：|实测−理论| = 0，容差 rtol=1e-12, atol=0
- **zero/db4/J=1 能量比 = 1**：松弛 rtol=1, atol=0，余量 = 4.54747e-13；正交（或等距）变换的 Parseval 等式
- **zero/db4/J=1 逐层能量占比求和 = 总能量比**：|实测−理论| = 0，容差 rtol=1e-12, atol=0
- **zero/db4/J=3 能量比 = 1**：松弛 rtol=1, atol=0，余量 = 4.54747e-13；正交（或等距）变换的 Parseval 等式
- **zero/db4/J=3 逐层能量占比求和 = 总能量比**：|实测−理论| = 0，容差 rtol=1e-12, atol=0
- **zero/db4/J=4 能量比 = 1**：松弛 rtol=1, atol=0，余量 = 4.54525e-13；正交（或等距）变换的 Parseval 等式
- **zero/db4/J=4 逐层能量占比求和 = 总能量比**：|实测−理论| = 2.22045e-16，容差 rtol=1e-12, atol=0
- **zero/db8/J=1 能量比 = 1**：松弛 rtol=1, atol=0，余量 = 4.54747e-13；正交（或等距）变换的 Parseval 等式
- **zero/db8/J=1 逐层能量占比求和 = 总能量比**：|实测−理论| = 0，容差 rtol=1e-12, atol=0
- **zero/db8/J=3 能量比 = 1**：松弛 rtol=1, atol=0，余量 = 4.54525e-13；正交（或等距）变换的 Parseval 等式
- **zero/db8/J=3 逐层能量占比求和 = 总能量比**：|实测−理论| = 0，容差 rtol=1e-12, atol=0
- **zero/db8/J=4 能量比 = 1**：松弛 rtol=1, atol=0，余量 = 4.54525e-13；正交（或等距）变换的 Parseval 等式
- **zero/db8/J=4 逐层能量占比求和 = 总能量比**：|实测−理论| = 1.11022e-16，容差 rtol=1e-12, atol=0
- **symmetric 模式能量比（50 个实现均值）**：单次实测区间 [1.0377, 1.1943]；同一变换在 periodic 下比值恒为 1
- **边界污染影响区（样本数）随层数增长**：最大相邻增量 48，松弛 3.41061e-13（machine:4·u·scale）；实测 6 -> 20 -> 48 -> 96（每层约翻倍）
- **受污染的系数个数（各层）**：松弛 rtol=1, atol=0，余量 = 10；每层都只有边界那几个系数，不是整层
- **偏差幅度随层数增长**：各层最大偏差 0.053、0.221、0.766、3.480
- **periodic/haar 各带之和 = x**：松弛 rtol=1, atol=0，余量 = 5.61665e-14；MRA 恒等式，逐位
- **periodic/haar 逐层累加末行 = x**：松弛 rtol=1, atol=0，余量 = 5.58958e-14
- **periodic/haar 第一层细节把误差降下来**：松弛 rtol=0, atol=0，余量 = 0.36794；只留最粗近似误差最大
- **periodic/db4 各带之和 = x**：松弛 rtol=1, atol=0，余量 = 5.63019e-14；MRA 恒等式，逐位
- **periodic/db4 逐层累加末行 = x**：松弛 rtol=1, atol=0，余量 = 5.61665e-14
- **periodic/db4 第一层细节把误差降下来**：松弛 rtol=0, atol=0，余量 = 0.140153；只留最粗近似误差最大
- **periodic/db8 各带之和 = x**：松弛 rtol=1, atol=0，余量 = 5.61665e-14；MRA 恒等式，逐位
- **periodic/db8 逐层累加末行 = x**：松弛 rtol=1, atol=0，余量 = 5.61665e-14
- **periodic/db8 第一层细节把误差降下来**：松弛 rtol=0, atol=0，余量 = 0.0171875；只留最粗近似误差最大
- **symmetric/haar 各带之和 = x**：松弛 rtol=1, atol=0，余量 = 5.61665e-14；MRA 恒等式，逐位
- **symmetric/haar 逐层累加末行 = x**：松弛 rtol=1, atol=0，余量 = 5.58958e-14
- **symmetric/haar 第一层细节把误差降下来**：松弛 rtol=0, atol=0，余量 = 0.36794；只留最粗近似误差最大
- **symmetric/db4 各带之和 = x**：松弛 rtol=1, atol=0，余量 = 5.61665e-14；MRA 恒等式，逐位
- **symmetric/db4 逐层累加末行 = x**：松弛 rtol=1, atol=0，余量 = 5.61665e-14
- **symmetric/db4 第一层细节把误差降下来**：松弛 rtol=0, atol=0，余量 = 0.53568；只留最粗近似误差最大
- **symmetric/db8 各带之和 = x**：松弛 rtol=1, atol=0，余量 = 5.60312e-14；MRA 恒等式，逐位
- **symmetric/db8 逐层累加末行 = x**：松弛 rtol=1, atol=0，余量 = 5.58958e-14
- **symmetric/db8 第一层细节把误差降下来**：松弛 rtol=0, atol=0，余量 = 0.515158；只留最粗近似误差最大
- **zero/haar 各带之和 = x**：松弛 rtol=1, atol=0，余量 = 5.61665e-14；MRA 恒等式，逐位
- **zero/haar 逐层累加末行 = x**：松弛 rtol=1, atol=0，余量 = 5.58958e-14
- **zero/haar 第一层细节把误差降下来**：松弛 rtol=0, atol=0，余量 = 0.36794；只留最粗近似误差最大
- **zero/db4 各带之和 = x**：松弛 rtol=1, atol=0，余量 = 5.61665e-14；MRA 恒等式，逐位
- **zero/db4 逐层累加末行 = x**：松弛 rtol=1, atol=0，余量 = 5.63019e-14
- **zero/db4 第一层细节把误差降下来**：松弛 rtol=0, atol=0，余量 = 0.140153；只留最粗近似误差最大
- **zero/db8 各带之和 = x**：松弛 rtol=1, atol=0，余量 = 5.61665e-14；MRA 恒等式，逐位
- **zero/db8 逐层累加末行 = x**：松弛 rtol=1, atol=0，余量 = 5.60312e-14
- **zero/db8 第一层细节把误差降下来**：松弛 rtol=0, atol=0，余量 = 0.00928238；只留最粗近似误差最大
- **软阈值不放大（1-Lipschitz 的推论）**：松弛 rtol=0, atol=0，余量 = 0.7
- **软阈值斜率不超过 1**：松弛 rtol=0, atol=0，余量 = 0.98；1-Lipschitz
- **|c| <= lambda 时软阈值为 0（硬阈值同样）**：松弛 rtol=0, atol=0，余量 = 0
- **|c| <= lambda 时硬阈值为 0**：松弛 rtol=0, atol=0，余量 = 0
- **硬阈值原样保留大系数（无偏）**：|实测−理论| = 0，容差 rtol=0, atol=0
- **软阈值对大系数收缩 lambda**：|实测−理论| = 0，容差 rtol=1e-12, atol=0
- **|soft| <= |hard|**：松弛 rtol=0, atol=0，余量 = 0；同一 lambda 下软阈值收缩更多
- **硬阈值在 |c|=lambda 处的跳变**：输入差 2e-9、输出差 0.7——方差就是这么来的
- **MAD 的相对偏差**：松弛 rtol=1, atol=0，余量 = 0.04591；200 个实现（n=256、db4、symmetric）
- **MAD 的相对标准差**：|实测−理论| = 0.017236，容差 rtol=0.35, atol=0；实测 ~9%，单次实现的读数可以偏 30%
- **细节系数方差 = sigma^2**：|实测−理论| = 0.00107016，容差 rtol=0.03, atol=0；正交小波把白噪声搬到系数域，方差不变
- **periodic 模式的 MAD 均值（对照）**：symmetric 的边界会复制样本，两个模式都必须无偏
- **SWT J=1 平移一格后系数变化**：松弛 rtol=0, atol=0，余量 = 0；逐位相同：同一批浮点加法作用在平移后的输入上
- **DWT J=1 平移残留（允许每带整体挪一格后）**：实测相对变化 2.014，不是 0
- **SWT J=2 平移一格后系数变化**：松弛 rtol=0, atol=0，余量 = 0；逐位相同：同一批浮点加法作用在平移后的输入上
- **DWT J=2 平移残留（允许每带整体挪一格后）**：实测相对变化 8.391，不是 0
- **SWT J=3 平移一格后系数变化**：松弛 rtol=0, atol=0，余量 = 0；逐位相同：同一批浮点加法作用在平移后的输入上
- **DWT J=3 平移残留（允许每带整体挪一格后）**：实测相对变化 36.235，不是 0
- **SWT J=4 平移一格后系数变化**：松弛 rtol=0, atol=0，余量 = 0；逐位相同：同一批浮点加法作用在平移后的输入上
- **DWT J=4 平移残留（允许每带整体挪一格后）**：实测相对变化 51.201，不是 0
- **SWT J=1 往返误差**：松弛 rtol=1, atol=0，余量 = 1.39888e-14；c·J·u
- **SWT J=3 往返误差**：松弛 rtol=1, atol=0，余量 = 4.21885e-14；c·J·u
- **SWT J=4 往返误差**：松弛 rtol=1, atol=0，余量 = 5.62883e-14；c·J·u
- **DWT 与 SWT 的平移敏感度之比**：DWT 的系数不是「挪个位置」，这就是抽取的代价
- **J=1 纯音能量比 = 框架符号**：松弛 rtol=1, atol=0，余量 = 2.00000e-10；预测 2.000000、实测 2.000000
- **J=2 纯音能量比 = 框架符号**：松弛 rtol=1, atol=0，余量 = 2.00000e-10；预测 4.000000、实测 4.000000
- **J=3 纯音能量比 = 框架符号**：松弛 rtol=1, atol=0，余量 = 2.00000e-10；预测 8.000000、实测 8.000000
- **J=4 纯音能量比 = 框架符号**：松弛 rtol=1, atol=0，余量 = 2.00000e-10；预测 16.000000、实测 16.000000
- **白噪声能量比 vs 框架符号平均（J=3）**：同一 J 的纯音比值是 8.0——差一倍，所以 J+1 不是框架界
- **periodic/J=1 系数总数 = n_ext**：|实测−理论| = 0，容差 rtol=0, atol=0；n_ext=256、滤波器长 L=16
- **periodic/J=1 冗余度记账**：|实测−理论| = 0，容差 rtol=1e-15, atol=0
- **periodic/J=4 系数总数 = n_ext**：|实测−理论| = 0，容差 rtol=0, atol=0；n_ext=256、滤波器长 L=16
- **periodic/J=4 冗余度记账**：|实测−理论| = 0，容差 rtol=1e-15, atol=0
- **periodic/J=6 系数总数 = n_ext**：|实测−理论| = 0，容差 rtol=0, atol=0；n_ext=256、滤波器长 L=16
- **periodic/J=6 冗余度记账**：|实测−理论| = 0，容差 rtol=1e-15, atol=0
- **symmetric/J=1 系数总数 = n_ext**：|实测−理论| = 0，容差 rtol=0, atol=0；n_ext=288、滤波器长 L=16
- **symmetric/J=1 冗余度记账**：|实测−理论| = 0，容差 rtol=1e-15, atol=0
- **symmetric/J=4 系数总数 = n_ext**：|实测−理论| = 0，容差 rtol=0, atol=0；n_ext=288、滤波器长 L=16
- **symmetric/J=4 冗余度记账**：|实测−理论| = 0，容差 rtol=1e-15, atol=0
- **symmetric/J=6 系数总数 = n_ext**：|实测−理论| = 0，容差 rtol=0, atol=0；n_ext=384、滤波器长 L=16
- **symmetric/J=6 冗余度记账**：|实测−理论| = 0，容差 rtol=1e-15, atol=0
- **zero/J=1 系数总数 = n_ext**：|实测−理论| = 0，容差 rtol=0, atol=0；n_ext=288、滤波器长 L=16
- **zero/J=1 冗余度记账**：|实测−理论| = 0，容差 rtol=1e-15, atol=0
- **zero/J=4 系数总数 = n_ext**：|实测−理论| = 0，容差 rtol=0, atol=0；n_ext=288、滤波器长 L=16
- **zero/J=4 冗余度记账**：|实测−理论| = 0，容差 rtol=1e-15, atol=0
- **zero/J=6 系数总数 = n_ext**：|实测−理论| = 0，容差 rtol=0, atol=0；n_ext=384、滤波器长 L=16
- **zero/J=6 冗余度记账**：|实测−理论| = 0，容差 rtol=1e-15, atol=0
- **边界模式与参数校验**：periodic 不整除、层数过大、未知小波、未知模式四种都测过
- **haar |G(w)| = |H(w+pi)|**：松弛 rtol=1, atol=0，余量 = 1.99922e-12；本模块的约定 g_j=(-1)^j h_{L-1-j} 给出的 QMF 关系（差一个相位因子）
- **haar |H|^2+|G|^2 = 2（逐点）**：松弛 rtol=1, atol=0，余量 = 1.99822e-12；完美重构条件；与 perfect_reconstruction_defect 互为独立路径
- **db4 |G(w)| = |H(w+pi)|**：松弛 rtol=1, atol=0，余量 = 1.99717e-12；本模块的约定 g_j=(-1)^j h_{L-1-j} 给出的 QMF 关系（差一个相位因子）
- **db4 |H|^2+|G|^2 = 2（逐点）**：松弛 rtol=1, atol=0，余量 = 1.99645e-12；完美重构条件；与 perfect_reconstruction_defect 互为独立路径
- **db8 |G(w)| = |H(w+pi)|**：松弛 rtol=1, atol=0，余量 = 1.99309e-12；本模块的约定 g_j=(-1)^j h_{L-1-j} 给出的 QMF 关系（差一个相位因子）
- **db8 |H|^2+|G|^2 = 2（逐点）**：松弛 rtol=1, atol=0，余量 = 1.99378e-12；完美重构条件；与 perfect_reconstruction_defect 互为独立路径

</details>
