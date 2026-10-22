# Problem index

Every problem in the curriculum, one line each, with the variables each
part asks for. Use this to work out which problem the student is on and
what it wants. The full text of a week is in tutor_weekNN.md.

## Week 1, numpy, pandas and matplotlib

### W1D1. Clean one plasma profile and get the terminal half-life

- topic: numpy: 1D arrays, nan masking, log-linear terminal slope
- notebook: problems/week01/w1d1_numpy_pk_profile.ipynb
- time: 20-30 min
- part a asks for: `measurable`, `n_blq`, `t_meas`, `c_meas`
- part b asks for: `cmax`, `tmax`
- part c asks for: `t_term`, `c_term`, `lambda_z`, `t_half`
- introduces: np.random.default_rng, np.array, .size, np.isnan, ~, boolean indexing, np.sum, np.max / np.min, np.argmax, np.log, np.polyfit, len

### W1D2. A subject by time matrix: axis reductions, SD, and broadcasting

- topic: numpy: 2D arrays, axis reductions, sample SD, broadcasting
- notebook: problems/week01/w1d2_numpy_subject_matrix.ipynb
- time: 20-30 min
- part a asks for: `mean_by_time`, `cmax_by_subject`
- part b asks for: `sd_by_time`, `cv_by_time`
- part c asks for: `norm_conc`
- introduces: .shape, np.mean, np.std, keepdims, .reshape, broadcasting, np.round

### W1D3. Rat PK study: filter, summarise by group, and join the dosing record

- topic: pandas: filtering, derived columns, groupby.agg, left merge
- notebook: problems/week01/w1d3_pandas_wrangle_merge.ipynb
- time: 20-30 min
- part a asks for: `n_animals`, `df_post`
- part b asks for: `summary`
- part c asks for: `pk_merged`, `n_missing_weight`
- introduces: pd.DataFrame, .head, .columns, selecting a column, boolean filtering, adding a column, .nunique / .unique, .isna, .groupby, .agg, .reset_index, .rename, pd.merge, np.repeat / np.tile

### W1D4. A collaborator's spreadsheet: reshape wide to tidy and back again

- topic: pandas: melt, string cleaning, duplicates, pivot
- notebook: problems/week01/w1d4_pandas_tidy_wide.ipynb
- time: 20-30 min
- part a asks for: `tidy`
- part b asks for: `tidy_clean`
- part c asks for: `n_dup_rows`, `wide_again`
- introduces: .melt, .str accessor, .astype, .drop, .sort_values, .duplicated, .drop_duplicates, .pivot, pd.concat

### W1D5. Cells inside animals: aggregate first, then plot mean with SD

- topic: matplotlib and pandas: nested data, animal level aggregation, SD error bars
- notebook: problems/week01/w1d5_matplotlib_nested_sd.ipynb
- time: 20-30 min
- part a asks for: `animal_level`
- part b asks for: `group_stats`
- part c asks for: see the notebook
- introduces: plt.subplots, ax.errorbar, ax.scatter, zorder, axis labels, ax.set_xticks, ax.legend, plt.show, np.where, np.sqrt

## Week 2, pandas in depth

### W2D1. Baseline correction: transform, and why it is not agg

- topic: pandas: groupby.transform, baseline correction, alignment
- notebook: problems/week02/w2d1_pandas_transform_baseline.ipynb
- time: 20-30 min
- part a asks for: `n_subjects`, `df`, `baseline_ok`
- part b asks for: `df_post`
- part c asks for: `summary`
- introduces: .transform, .all / .any, ignore_index

### W2D2. Joins that lie: validate the merge before you trust it

- topic: pandas: join types, validate, indicator, duplicate keys
- notebook: problems/week02/w2d2_pandas_merge_validate.ipynb
- time: 20-30 min
- part a asks for: `n_pk_rows`, `n_dup_rows`, `dup_subjects`
- part b asks for: `dosing_clean`, `merged`
- part c asks for: `checked`, `n_left_only`, `unmatched_subjects`
- introduces: validate=, indicator=, .value_counts, sorted

### W2D3. Nominal versus actual sampling times, and the samples that never arrived

- topic: pandas: time deviations, building a full grid, finding missing records
- notebook: problems/week02/w2d3_pandas_nominal_actual.ipynb
- time: 20-30 min
- part a asks for: `df`, `n_out_of_window`
- part b asks for: `grid`, `missing`, `n_missing`
- part c asks for: `per_subject`
- introduces: pd.MultiIndex.from_product, .abs, .fillna, .assign

### W2D4. Adverse events: counting events is not counting patients

- topic: pandas: crosstab, pivot_table, normalize, subject level incidence
- notebook: problems/week02/w2d4_pandas_crosstab_incidence.ipynb
- time: 20-30 min
- part a asks for: `ae_counts`
- part b asks for: `ae_pct`
- part c asks for: `incidence`
- introduces: pd.crosstab, pd.pivot_table, size versus nunique, .reindex, .map

### W2D5. Two panels and a spaghetti plot: the figure every PK report opens with

- topic: matplotlib: subplots, semilog axes, per subject overlays
- notebook: problems/week02/w2d5_matplotlib_panels_semilog.ipynb
- time: 20-30 min
- part a asks for: `summary`
- part b asks for: see the notebook
- part c asks for: see the notebook
- introduces: looping over groupby, ax.set_yscale, plt.subplots(rows, cols), fig.tight_layout

## Week 3, scipy

### W3D1. Comparing two groups when the data is skewed

- topic: scipy.stats: t-test, Welch, Mann-Whitney, and choosing between them
- notebook: problems/week03/w3d1_scipy_group_comparison.ipynb
- time: 20-30 min
- part a asks for: `desc`
- part b asks for: `t_student`, `p_student`, `t_welch`, `p_welch`
- part c asks for: `u_stat`, `p_mw`, `t_log`, `p_log`
- introduces: from scipy import stats, stats.ttest_ind, stats.mannwhitneyu, stats.iqr, result objects

### W3D2. AUC is lognormal, so report the geometric mean

- topic: scipy.stats: lognormal data, geometric mean and CV, normality testing
- notebook: problems/week03/w3d2_scipy_geometric_stats.ipynb
- time: 20-30 min
- part a asks for: `arith_mean`, `arith_sd`, `arith_cv`
- part b asks for: `geo_mean`, `sd_log`, `geo_cv`
- part c asks for: `p_raw`, `p_log`, `use_log`
- introduces: geometric mean, geometric CV, stats.shapiro, np.exp

### W3D3. Fitting a decay curve, and how sure you are of the parameters

- topic: scipy.optimize: curve_fit, starting values, parameter standard errors
- notebook: problems/week03/w3d3_scipy_curve_fit.ipynb
- time: 20-30 min
- part a asks for: `popt`, `pcov`
- part b asks for: `c0_hat`, `k_hat`, `se_c0`, `se_k`
- part c asks for: `t_half`, `pred`, `rmse`
- introduces: curve_fit, p0, np.diag, argument unpacking

### W3D4. Solving the compartment model as an ODE, and surviving a discontinuity

- topic: scipy.integrate: solve_ivp, t_eval, checking against an analytic solution
- notebook: problems/week03/w3d4_scipy_solve_ivp.ipynb
- time: 20-30 min
- part a asks for: `sol`, `conc_num`
- part b asks for: `conc_exact`, `max_abs_err`
- part c asks for: `sol_inf`, `conc_inf`, `cmax_inf`, `t_cmax`
- introduces: solve_ivp, t_eval, max_step, sol.y

### W3D5. Emax and sigmoid Emax: fitting a concentration response curve

- topic: scipy.optimize: Emax models, bounds, and comparing nested fits
- notebook: problems/week03/w3d5_scipy_emax_hill.ipynb
- time: 20-30 min
- part a asks for: `agg`
- part b asks for: `emax1`, `ec50_1`, `rmse1`
- part c asks for: `emax2`, `ec50_2`, `hill`, `rmse2`
- introduces: Emax model, sigmoid Emax, bounds, np.logspace

## Week 4, scikit-image and OpenCV

### W4D1. From a microscope image to a binary mask

- topic: scikit-image: images as arrays, Otsu thresholding, smoothing before you threshold
- notebook: problems/week04/w4d1_skimage_thresholding.ipynb
- time: 20-30 min
- part a asks for: `otsu_t`, `mask`, `n_fg`, `frac_fg`
- part b asks for: `otsu_t2`, `frac_otsu_2`, `frac_fixed_2`
- part c asks for: `smoothed`, `mask_smooth`, `n_obj_raw`, `n_obj_smooth`
- introduces: images as arrays, filters.threshold_otsu, filters.gaussian, measure.label, ax.imshow

### W4D2. Cleaning up a mask: opening, filling holes, and dropping edge cells

- topic: scikit-image: morphological opening, filling holes, clearing the border
- notebook: problems/week04/w4d2_skimage_morphology.ipynb
- time: 20-30 min
- part a asks for: `mask_open`, `n_obj_raw`, `n_obj_open`
- part b asks for: `mask_filled`, `n_hole_px`
- part c asks for: `mask_clean`, `n_obj_clean`, `n_border_objects`
- introduces: morphology.opening, morphology.closing, morphology.disk, binary_fill_holes, clear_border

### W4D3. Labelling objects and measuring them one by one

- topic: scikit-image: connected components, regionprops_table, filtering by measurement
- notebook: problems/week04/w4d3_skimage_regionprops.ipynb
- time: 20-30 min
- part a asks for: `labels`, `n_objects`, `props`
- part b asks for: `cells`, `n_cells`, `n_dropped`
- part c asks for: `mean_area`, `mean_ecc`, `mean_intensity`
- introduces: measure.regionprops_table, connectivity, eccentricity

### W4D4. The same segmentation in OpenCV, and where the two libraries disagree

- topic: OpenCV: blurring, Otsu thresholding, findContours and contour areas
- notebook: problems/week04/w4d4_opencv_contours.ipynb
- time: 20-30 min
- part a asks for: `blurred`, `t_cv`, `binary`
- part b asks for: `contours`, `n_contours`
- part c asks for: `areas`, `n_cells_cv`, `area_ratio`
- introduces: cv2.GaussianBlur, cv2.threshold, cv2.findContours, cv2.contourArea

### W4D5. A whole imaging study: fields inside animals inside groups

- topic: scikit-image pipeline: per field, per animal, per group, with the right n
- notebook: problems/week04/w4d5_imaging_pipeline_animals.ipynb
- time: 20-30 min
- part a asks for: `per_field`
- part b asks for: `per_animal`
- part c asks for: `group_stats`
- introduces: building a DataFrame from rows, three level aggregation

## Week 5, scikit-learn

### W5D1. Predicting clearance: the train/test split and why it exists

- topic: scikit-learn: train_test_split, fit and predict, R2 and RMSE, optimism
- notebook: problems/week05/w5d1_sklearn_train_test.ipynb
- time: 20-30 min
- part a asks for: `X_train`, `X_test`, `y_train`, `y_test`, `n_train`, `n_test`
- part b asks for: `model`, `pred_test`, `r2_test`, `rmse_test`
- part c asks for: `r2_train`, `rmse_train`, `optimism`
- introduces: train_test_split, fit and predict, r2_score, mean_squared_error, random_state

### W5D2. Cross-validation, and the leak that invents a result from noise

- topic: scikit-learn: KFold, cross_val_score, Pipeline, and data leakage
- notebook: problems/week05/w5d2_sklearn_cv_leakage.ipynb
- time: 20-30 min
- part a asks for: `cv_scores`, `cv_mean`, `cv_sd`
- part b asks for: `X_leaky`, `cv_leaky_mean`
- part c asks for: `pipe`, `cv_clean_mean`, `leak_inflation`
- introduces: KFold, cross_val_score, Pipeline, SelectKBest, data leakage

### W5D3. Random forests, and two different answers to which feature matters

- topic: scikit-learn: RandomForestRegressor, impurity importance, permutation importance
- notebook: problems/week05/w5d3_sklearn_random_forest.ipynb
- time: 20-30 min
- part a asks for: `rf`, `r2_test_rf`, `r2_train_rf`, `gap_rf`
- part b asks for: `imp_impurity`
- part c asks for: `perm`, `imp_perm`, `noise_rank_impurity`, `noise_rank_perm`
- introduces: RandomForestRegressor, feature_importances_, permutation_importance

### W5D4. Elastic net: why you must standardise, and what the penalty buys

- topic: scikit-learn: ElasticNet, standardisation, regularisation paths, ElasticNetCV
- notebook: problems/week05/w5d4_sklearn_elastic_net.ipynb
- time: 20-30 min
- part a asks for: `coef_raw`, `coef_scaled`, `n_zero_raw`, `n_zero_scaled`
- part b asks for: `path`
- part c asks for: `encv`, `best_alpha`, `n_nonzero_cv`, `r2_cv_test`
- introduces: ElasticNet, StandardScaler, make_pipeline, ElasticNetCV

### W5D5. PCA on a correlated covariate panel

- topic: scikit-learn: PCA, explained variance, loadings, projecting onto components
- notebook: problems/week05/w5d5_sklearn_pca.ipynb
- time: 20-30 min
- part a asks for: `pca`, `evr`, `evr_unscaled_first`
- part b asks for: `cumulative`, `n_for_90`, `scores`
- part c asks for: `loadings`, `pc1_top3`, `pc2_top3`
- introduces: PCA, explained_variance_ratio_, components_, np.cumsum

## Week 6, PyTorch

### W6D1. Tensors: numpy with two things added

- topic: PyTorch: tensors, dtypes, shapes, numpy interop, the autograd flag
- notebook: problems/week06/w6d1_torch_tensors.ipynb
- time: 20-30 min
- part a asks for: `t_tensor`, `c_tensor`, `dtype_default`, `dtype_asked`
- part b asks for: `cmax`, `tmax`, `log_c`, `mask`, `c_measurable`
- part c asks for: `plain`, `tracked`, `result_plain`, `result_tracked`, `has_grad_fn_plain`, `has_grad_fn_tracked`, `shares_memory`
- introduces: torch.tensor, tensor dtypes, .item(), tensor to numpy, requires_grad

### W6D2. Autograd: derivatives you did not have to derive

- topic: PyTorch: backward, .grad, gradient accumulation, no_grad, autograd vs finite differences
- notebook: problems/week06/w6d2_torch_autograd.ipynb
- time: 20-30 min
- part a asks for: `k_a`, `c_pred`, `resid`, `loss`, `loss_a`, `grad_auto`, `grad_manual`
- part b asks for: `v_b`, `k_b`, `loss_b`, `grad_v`, `grad_k`, `fd_v`, `fd_k`
- part c asks for: `k_c`, `grad_once`, `grad_twice`, `grad_after_zero`, `k_d`, `second_backward_error`
- introduces: backward, tensor .grad, gradient accumulation, torch.no_grad, finite differences

### W6D3. Gradient descent, written out by hand

- topic: PyTorch: the fitting loop, learning rate, log parameters, divergence
- notebook: problems/week06/w6d3_torch_gradient_descent.ipynb
- time: 20-30 min
- part a asks for: `v1`, `k1`, `loss_before`, `loss_after_small`, `loss_after_big`, `inplace_error`
- part b asks for: `log_v`, `log_k`, `loss_hist`, `v_hat`, `k_hat`, `v_scipy`, `k_scipy`
- part c asks for: `loss_big`, `v_big`, `zero_model_loss`, `loss_hist_huge`, `first_nan_index`
- introduces: gradient descent loop, learning rate, log parameters, np.isnan, np.argmax on booleans

### W6D4. nn.Module and an optimiser: the loop everyone actually writes

- topic: PyTorch: nn.Module, nn.Parameter, torch.optim, Adam against plain SGD
- notebook: problems/week06/w6d4_torch_module_optimizer.ipynb
- time: 20-30 min
- part a asks for: `v`, `k`, `n_params`, `param_names`, `all_require_grad`, `loss_start`
- part b asks for: `opt`, `loss_hist_adam`, `v_hat`, `k_hat`, `loss_hist_sgd`
- part c asks for: `cl_hat`, `t_half_hat`, `auc_inf`, `resid_sd`
- introduces: torch.nn.Module, torch.nn.Parameter, torch.optim, zero_grad and step, state_dict

### W6D5. Twelve subjects, twenty four parameters, one backward pass

- topic: PyTorch: vectorised parameters, pooled against individual fitting, recovering between subject variability
- notebook: problems/week06/w6d5_torch_population_fit.ipynb
- time: 25-35 min
- part a asks for: `v`, `k`, `v_pool`, `k_pool`, `sse_pool`, `subject_bias`
- part b asks for: `v_ind`, `k_ind`, `sse_ind`, `v_scipy`
- part c asks for: `v_gm`, `cv_v_pct`, `cv_k_pct`, `sse_ratio`
- introduces: torch.full, parameter vectors, None axis indexing, pooled against individual fitting, between subject CV

## Week 7, PK and PD applications, part 1

### W7D1. Non-compartmental analysis: Cmax, Tmax and AUC by trapezoid

- topic: PK: linear and linear-up log-down trapezoidal AUC, partial AUC
- notebook: problems/week07/w7d1_nca_auc_trapezoid.ipynb
- time: 20-30 min
- part a asks for: `cmax`, `tmax`, `auc_linear`
- part b asks for: `auc_linlog`, `pct_diff`
- part c asks for: `auc_0_12`, `frac_in_12h`
- introduces: np.diff, linear trapezoid, linear up log down, np.errstate, partial AUC

### W7D2. Finishing the NCA: lambda_z, AUC to infinity, clearance and MRT

- topic: PK: terminal slope selection, extrapolated AUC, CL/F, Vz/F, mean residence time
- notebook: problems/week07/w7d2_nca_lambdaz_clearance.ipynb
- time: 20-30 min
- part a asks for: `n_points_used`, `lambda_z`, `t_half`, `adj_r2`
- part b asks for: `auc_inf`, `pct_extrap`, `pct_extrap_12h`
- part c asks for: `cl_f`, `vz_f`, `mrt`
- introduces: lambda_z selection, adjusted R2, AUC extrapolation, mean residence time

### W7D3. Fitting a one compartment oral model, and checking it against NCA

- topic: PK modelling: depot and central compartments, fitting an ODE model, model versus NCA
- notebook: problems/week07/w7d3_pk_fit_one_compartment.ipynb
- time: 20-30 min
- part a asks for: see the notebook
- part b asks for: `popt`, `ka_hat`, `v_hat`, `cl_hat`, `rmse`
- part c asks for: `t_half_model`, `cl_ratio`, `flip_flop`
- introduces: depot and central, fitting an ODE model, flip-flop kinetics

### W7D4. Two compartments, and what a one compartment model gets wrong

- topic: PK modelling: distribution phase, alpha and beta, choosing model structure
- notebook: problems/week07/w7d4_pk_two_compartment.ipynb
- time: 20-30 min
- part a asks for: `popt2`
- part b asks for: `popt1`, `rmse1`, `rmse2`
- part c asks for: `vss`, `late_bias`, `auc_one`, `auc_two`, `auc_bias`
- introduces: two compartment model, alpha and beta phases, Vss, residual structure

### W7D5. Multiple dosing: accumulation, steady state, and how long it takes

- topic: PK: repeated dosing by simulation, accumulation ratio, time to steady state
- notebook: problems/week07/w7d5_pk_multiple_dosing.ipynb
- time: 20-30 min
- part a asks for: `times`
- part b asks for: `cmax_first`, `cmax_ss`, `r_sim`, `r_theory`
- part c asks for: `cavg_ss`, `t_to_90pct`, `n_half_lives`
- introduces: carrying state forward, accumulation ratio, time to steady state, Cavg at steady state

## Week 8, PK and PD applications, part 2

### W8D1. PK/PD link: why the effect peaks hours after the concentration

- topic: PK/PD: hysteresis, direct Emax fitting and its residual structure, the effect compartment
- notebook: problems/week08/w8d1_pkpd_hysteresis.ipynb
- time: 25-35 min
- part a asks for: `peak_idx`, `tmax_c`, `tmax_e`, `e_rising`, `e_falling`
- part b asks for: `emax_direct`, `ec50_direct`, `sse_direct`, `resid_early`, `resid_late`
- part c asks for: `ke0_hat`, `emax_hat`, `ec50_hat`, `link_pred`, `sse_link`
- introduces: np.interp, hysteresis, effect compartment, residuals against time, curve_fit bounds

### W8D2. Indirect response: when the biomarker, not the drug, sets the clock

- topic: PK/PD: turnover models, inhibition of production, dose dependent time to nadir
- notebook: problems/week08/w8d2_indirect_response.ipynb
- time: 25-35 min
- part a asks for: `kin`, `r_single`, `nadir`, `t_nadir`, `pct_suppression`
- part b asks for: `t_nadir_by_dose`, `nadir_by_dose`, `t_above_by_dose`, `t_link_by_dose`
- part c asks for: `auec`, `t_recover`, `t_nadir_fast`, `t_nadir_slow`
- introduces: indirect response model, turnover half life, baseline pins kin, AUEC, np.where first index

### W8D3. A virtual population: simulating variability and reading it back

- topic: Population PK: lognormal parameters, exposure percentiles, how precisely a CV can be estimated
- notebook: problems/week08/w8d3_population_variability.ipynb
- time: 25-35 min
- part a asks for: `cl_gm`, `cl_cv_pct`, `cl_mean`, `mean_over_median`, `mean_over_median_theory`
- part b asks for: `k_i`, `auc_i`, `tmax_i`, `cmax_i`, `auc_gm`, `auc_p5`, `auc_p95`, `fold_range`
- part c asks for: `draws_12`, `draws_120`, `cv_hat_12`, `cv_hat_120`, `cv12_lo`, `cv12_hi`, `cv120_lo`, `cv120_hi`
- introduces: rng.lognormal, np.percentile, geometric mean and CV, arithmetic against geometric mean, Monte Carlo study simulation

### W8D4. Building a NONMEM ready dataset, and auditing it

- topic: Population PK: event records, EVID and MDV, BLQ handling, per subject NCA with groupby
- notebook: problems/week08/w8d4_nonmem_dataset.ipynb
- time: 25-35 min
- part a asks for: `dose_rows`, `obs_rows`, `nm`, `n_rows`, `n_dose_rows`, `first_evid_per_id`
- part b asks for: `n_blq`, `n_mdv`, `records_per_id`, `time_monotone`
- part c asks for: `nca`, `auc_dn_low`, `auc_dn_high`, `auc_dn_ratio`
- introduces: NONMEM event records, EVID and MDV, BLQ handling, sort_values mixed ascending, groupby apply several outputs

### W8D5. Capstone: choosing the phase 2 dose

- topic: PK/PD: steady state exposure, probability of target attainment, joint efficacy and safety
- notebook: problems/week08/w8d5_dose_selection.ipynb
- time: 30-40 min
- part a asks for: `k_typ`, `c_typ`, `cmax_ss`, `cmin_ss`, `cavg_ss`, `cavg_identity`
- part b asks for: `k`, `cmax_i`, `mean_inh_i`, `p_eff`, `p_tox`, `p_success`
- part c asks for: `p_eff_by_dose`, `p_tox_by_dose`, `p_success_by_dose`, `best_dose`
- introduces: steady state closed form, Cavg identity, probability of target attainment, joint criteria, boolean mean as a fraction
