"""Feed deliberately wrong answers to each check() and confirm it catches them.

    python3 tools/negative_test.py

A checker that passes everything is worse than no checker, so this starts from
the correct solution, breaks exactly one thing, and insists that the right
diagnostic comes back.
"""

import contextlib
import importlib.util
import io
import os
import pathlib
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

import nbformat

ROOT = pathlib.Path(__file__).resolve().parent.parent
SPEC_DIR = ROOT / "tools" / "specs"

# (problem id, part, code that breaks one thing, text the message must contain)
CASES = [
    ("W1D1", "a", "measurable = np.isnan(conc)", "flip"),
    ("W1D1", "a", "n_blq = int(np.sum(~np.isnan(conc)))", "Count the entries where"),
    ("W1D1", "a", "t_meas = time_h", "still has every sample"),
    ("W1D1", "b", "cmax = np.max(conc)", "nan"),
    ("W1D1", "b", "tmax = int(np.argmax(c_meas))", "index of the peak"),
    ("W1D1", "b", "cmax, tmax = tmax, cmax", "swapped"),
    ("W1D1", "c", "lambda_z = -lambda_z", "negative"),
    ("W1D1", "c", "t_half = np.log(2) / -np.polyfit(t_meas, np.log(c_meas), 1)[0]", "whole profile"),
    ("W1D1", "c", "t_term = t_meas[-3:]\nc_term = c_meas[-3:]", "last 4"),

    ("W1D2", "a", "mean_by_time = conc.mean(axis=1)", "one value per subject"),
    ("W1D2", "a", "cmax_by_subject = conc.max(axis=0)", "one value per timepoint"),
    ("W1D2", "a", "cmax_by_subject = conc.mean(axis=1)", "mean, not each subject's peak"),
    ("W1D2", "b", "sd_by_time = conc.std(axis=0)", "ddof=1"),
    ("W1D2", "b", "sd_by_time = conc.std(axis=0, ddof=1) / np.sqrt(conc.shape[0])", "SEM"),
    ("W1D2", "b", "cv_by_time = sd_by_time / mean_by_time", "percent"),
    ("W1D2", "c", "norm_conc = conc / conc.max()", "only one subject peaks"),
    ("W1D2", "c", "norm_conc = conc / conc.max(axis=0, keepdims=True)", "down the columns"),

    ("W1D3", "a", "n_animals = len(pk)", "number of rows in pk"),
    ("W1D3", "a", "df_post = pk", "still has every row"),
    ("W1D3", "b", "summary = (pk.groupby(['GROUP','TIME_H'])['CONC_NG_ML'].agg(['mean','std'])"
                  ".reset_index().rename(columns={'mean':'CONC_MEAN','std':'CONC_SD'}))", "predose"),
    ("W1D3", "b", "summary = summary.rename(columns={'CONC_SD': 'SD'})", "named exactly"),
    ("W1D3", "c", "pk_merged = pd.merge(pk, demo, on='ANIMAL', how='inner')\n"
                  "pk_merged['DOSE_MG_PER_KG'] = pk_merged['DOSE_MG'] / (pk_merged['BODYWEIGHT_G']/1000)",
     "inner join"),
    ("W1D3", "c", "pk_merged['DOSE_MG_PER_KG'] = pk_merged['DOSE_MG'] / pk_merged['BODYWEIGHT_G']",
     "milligrams per gram"),
    ("W1D3", "c", "n_missing_weight = 1", "number of animals missing"),

    ("W1D4", "a", "tidy = wide.melt(id_vars=['ANIMAL','GROUP'])", "variable"),
    ("W1D4", "a", "tidy = wide", "not been reshaped"),
    ("W1D4", "b", "tidy_clean = tidy_clean.sort_values(['ANIMAL','INHIB_PCT'])"
                  ".reset_index(drop=True)", "not in the right order"),
    ("W1D4", "b", "tidy_clean = (tidy.assign(TIME_H=tidy['TIME_LABEL'].str.replace("
                  "'T_','',regex=False).astype(float)).drop(columns=['TIME_LABEL'])"
                  ".sort_values(['ANIMAL','TIME_H']))", "index reset"),
    ("W1D4", "c", "n_dup_rows = int(tidy_clean.duplicated(keep=False).sum())", "double"),
    ("W1D4", "c", "n_dup_rows = 1", "number of animals that got re-sent"),
    ("W1D4", "c", "wide_again = wide_again.reset_index()", "ANIMAL as its index"),

    ("W1D5", "a", "animal_level = cells.assign(MEAN_INTENSITY=cells['INTENSITY'], N_CELLS=1)",
     "one row per cell"),
    ("W1D5", "a", "animal_level['MEAN_INTENSITY'] = ("
                  "cells.groupby(['ANIMAL'])['INTENSITY'].median().reindex("
                  "animal_level['ANIMAL']).to_numpy())", "median"),
    ("W1D5", "b", "group_stats = (cells.groupby('GROUP').agg(MEAN=('INTENSITY','mean'),"
                  "SD=('INTENSITY','std'), N_ANIMALS=('INTENSITY','size')).reset_index())",
     "counted cells, not animals"),
    ("W1D5", "b", "group_stats['MEAN'] = (cells.groupby('GROUP')['INTENSITY'].mean()"
                  ".reindex(group_stats['GROUP']).to_numpy())", "weights each animal"),
    ("W1D5", "b", "group_stats['SD'] = group_stats['SD'] / np.sqrt(group_stats['N_ANIMALS'])", "SEM"),
    ("W1D5", "b", "group_stats['SD'] = (cells.groupby('GROUP')['INTENSITY'].std()"
                  ".reindex(group_stats['GROUP']).to_numpy())", "spread of individual cells"),
    ("W1D5", "c", "_SEM = group_stats['SD'] / np.sqrt(group_stats['N_ANIMALS'])\n"
                  "fig, ax = plt.subplots()\n"
                  "ax.errorbar([0,1], group_stats['MEAN'], yerr=_SEM, fmt='o')\n"
                  "ax.scatter([0]*6+[1]*6, animal_level['MEAN_INTENSITY'])\n"
                  "ax.set_xticks([0,1]); ax.set_xticklabels(list(group_stats['GROUP']))\n"
                  "ax.set_xlabel('g'); ax.set_ylabel('i'); ax.set_title('t')", "SEM"),
    ("W1D5", "c", "_CELLSD = cells.groupby('GROUP')['INTENSITY'].std().reindex("
                  "group_stats['GROUP']).to_numpy()\n"
                  "fig, ax = plt.subplots()\n"
                  "ax.errorbar([0,1], group_stats['MEAN'], yerr=_CELLSD, fmt='o')\n"
                  "ax.scatter([0]*6+[1]*6, animal_level['MEAN_INTENSITY'])\n"
                  "ax.set_xticks([0,1]); ax.set_xticklabels(list(group_stats['GROUP']))\n"
                  "ax.set_xlabel('g'); ax.set_ylabel('i'); ax.set_title('t')",
     "SD of individual cells"),
    ("W1D5", "c", "fig, ax = plt.subplots()\n"
                  "ax.errorbar([0,1], group_stats['MEAN'], yerr=group_stats['SD'], fmt='o')\n"
                  "ax.set_xticks([0,1]); ax.set_xticklabels(list(group_stats['GROUP']))\n"
                  "ax.set_xlabel('g'); ax.set_ylabel('i'); ax.set_title('t')",
     "No scatter points found"),
    ("W1D5", "c", "fig, ax = plt.subplots()\n"
                  "ax.errorbar([0,1], group_stats['MEAN'], yerr=group_stats['SD'], fmt='o')\n"
                  "ax.scatter(np.where(cells['GROUP']=='Drug',0,1), cells['INTENSITY'])\n"
                  "ax.set_xticks([0,1]); ax.set_xticklabels(list(group_stats['GROUP']))\n"
                  "ax.set_xlabel('g'); ax.set_ylabel('i'); ax.set_title('t')", "one point per cell"),
    ("W1D5", "c", "fig, ax = plt.subplots()\n"
                  "ax.errorbar([0,1], group_stats['MEAN'], yerr=group_stats['SD'], fmt='o')\n"
                  "ax.scatter([0]*6+[1]*6, animal_level['MEAN_INTENSITY'])\n"
                  "ax.set_xticks([0,1]); ax.set_xticklabels(list(group_stats['GROUP']))",
     "x axis has no label"),
    ("W1D5", "c", "fig, ax = plt.subplots()", "axes is still empty"),
    ("W1D5", "c", "_POOL = cells.groupby('GROUP')['INTENSITY'].mean().reindex("
                  "group_stats['GROUP']).to_numpy()\n"
                  "fig, ax = plt.subplots()\n"
                  "ax.errorbar([0,1], _POOL, yerr=group_stats['SD'], fmt='o')\n"
                  "ax.scatter([0]*6+[1]*6, animal_level['MEAN_INTENSITY'])\n"
                  "ax.set_xticks([0,1]); ax.set_xticklabels(list(group_stats['GROUP']))\n"
                  "ax.set_xlabel('g'); ax.set_ylabel('i'); ax.set_title('t')",
     "pooling every cell"),
    ("W1D5", "c", "fig, ax = plt.subplots()\n"
                  "ax.errorbar([0,1], [1.0, 2.0], yerr=group_stats['SD'], fmt='o')\n"
                  "ax.scatter([0]*6+[1]*6, animal_level['MEAN_INTENSITY'])\n"
                  "ax.set_xticks([0,1]); ax.set_xticklabels(list(group_stats['GROUP']))\n"
                  "ax.set_xlabel('g'); ax.set_ylabel('i'); ax.set_title('t')",
     "do not match the mean of the animal means"),
# ------------------------------------------------------------ week 2
    ("W2D1", "a", "df = raw.copy()\n"
                  "df['BASELINE'] = df.groupby('SUBJECT')['BIOMARKER'].transform('first')",
     "not sorted by SUBJECT"),
    ("W2D1", "a", "df['BASELINE'] = df.groupby('SUBJECT')['BIOMARKER'].transform('mean')",
     "MEAN, not their time zero"),
    ("W2D1", "b", "df_post['PCT_CHANGE'] = df_post['PCT_CHANGE'] / 100.0", "fraction"),
    ("W2D1", "b", "df_post['PCT_CHANGE'] = -df_post['PCT_CHANGE']", "sign the wrong way"),
    ("W2D1", "c", "summary['PCT_SD'] = summary['PCT_SD'] / np.sqrt(summary['N'])", "SEM"),

    ("W2D2", "a", "n_dup_rows = int(dosing.duplicated(subset=['SUBJECT'], keep=False).sum())",
     "keep=False"),
    ("W2D2", "b", "merged = pd.merge(pk, dosing, on='SUBJECT', how='left')", "doubled"),
    ("W2D2", "b", "merged = pd.merge(pk, dosing_clean, on='SUBJECT', how='inner')",
     "inner join dropped"),
    ("W2D2", "c", "n_left_only = int((checked['_merge'] == 'both').sum())", "did match"),
    ("W2D2", "c", "n_left_only = 1", "unmatched SUBJECTS"),

    ("W2D3", "a", "df['OUT_OF_WINDOW'] = df['DEV_MIN'].abs() > 5.0\n"
                  "n_out_of_window = int(df['OUT_OF_WINDOW'].sum())", "flat 5 minute"),
    ("W2D3", "a", "df['OUT_OF_WINDOW'] = df['DEV_MIN'].abs() > 30.0\n"
                  "n_out_of_window = int(df['OUT_OF_WINDOW'].sum())", "flat 30 minute"),
    ("W2D3", "a", "df['DEV_MIN'] = df['ACTUAL_H'] - df['NOMINAL_H']", "in hours"),
    ("W2D3", "b", "grid = samples[['SUBJECT','NOMINAL_H']].copy()", "not the full scheduled"),
    ("W2D3", "c", "per_subject = per_subject.iloc[:-1]", "rows but there are"),

    ("W2D4", "a", "ae_counts = pd.crosstab(ae['GROUP'], ae['SEVERITY']).reindex(GROUP_ORDER)",
     "no Total row"),
    ("W2D4", "b", "ae_pct = (pd.crosstab(ae['GROUP'], ae['SEVERITY'], normalize='columns')"
                  "*100).reindex(GROUP_ORDER).round(1)", "down the columns"),
    ("W2D4", "b", "ae_pct = pd.crosstab(ae['GROUP'], ae['SEVERITY'], normalize='index')"
                  ".reindex(GROUP_ORDER)", "still a fraction"),
    ("W2D4", "c", "incidence['N_DOSED'] = incidence['N_WITH_AE']\n"
                  "incidence['PCT_WITH_AE'] = (100.0*incidence['N_WITH_AE']/incidence['N_DOSED']).round(1)",
     "taken from the event log"),
    ("W2D4", "c", "incidence['N_WITH_AE'] = (ae.groupby('GROUP')['SUBJECT'].size()"
                  ".reindex(GROUP_ORDER).fillna(0).astype(int).to_numpy())", "counted EVENTS"),

    ("W2D5", "a", "summary['CONC_SD'] = summary['CONC_SD'] / np.sqrt(summary['N'])", "SEM"),
    ("W2D5", "b", "_SEM = summary['CONC_SD'] / np.sqrt(summary['N'])\n"
                  "fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))\n"
                  "for _n, _p in summary.assign(SE=_SEM.to_numpy()).groupby('DOSE'):\n"
                  "    for _a in axes:\n"
                  "        _a.errorbar(_p['TIME_H'], _p['CONC_MEAN'], yerr=_p['SE'], fmt='o-', label=_n)\n"
                  "axes[1].set_yscale('log')\n"
                  "for _a in axes:\n"
                  "    _a.set_xlabel('t'); _a.set_ylabel('c'); _a.set_title('x')\n"
                  "axes[0].legend()", "SEM error bars"),
    ("W2D5", "b", "axes[0].set_yscale('log')", "Both panels are logarithmic"),
    ("W2D5", "c", "fig2, ax2 = plt.subplots()\n"
                  "for _n, _p in pk.groupby('DOSE'):\n"
                  "    ax2.plot(_p['TIME_H'], _p['CONC_MG_L'])\n"
                  "ax2.set_yscale('log')\n"
                  "ax2.set_xlabel('t'); ax2.set_ylabel('c'); ax2.set_title('x')",
     "one per dose group"),
    ("W2D5", "c", "ax2.set_yscale('linear')", "not logarithmic"),
# ------------------------------------------------------------ week 3
    ("W3D1", "a", "desc['SD'] = [round(float(np.std(vehicle)),2), round(float(np.std(treated)),2)]",
     "n in the denominator"),
    ("W3D1", "b", "t_welch = t_student; p_welch = p_student", "equal_var=False was not passed"),
    ("W3D1", "b", "_r = stats.ttest_ind(treated, vehicle)\n"
                  "t_student = float(_r.statistic); p_student = float(_r.pvalue)",
     "sign flipped"),
    ("W3D1", "c", "_r = stats.mannwhitneyu(vehicle, treated, alternative='greater')\n"
                  "p_mw = float(_r.pvalue)", "one sided"),
    ("W3D1", "c", "_r = stats.ttest_ind(vehicle, treated, equal_var=False)\n"
                  "t_log = float(_r.statistic)", "test on the raw volumes"),

    ("W3D2", "a", "arith_sd = float(np.std(auc))", "n in the denominator"),
    ("W3D2", "a", "arith_cv = arith_sd / arith_mean", "fraction"),
    ("W3D2", "b", "geo_mean = float(np.mean(np.log(auc)))", "still on the log scale"),
    ("W3D2", "b", "geo_cv = 100.0 * sd_log", "needs the exponential"),
    ("W3D2", "c", "use_log = False", "should be True here"),

    ("W3D3", "a", "def _swapped(t, k, c0):\n"
                  "    return c0 * np.exp(-k * t)\n"
                  "popt, pcov = curve_fit(_swapped, obs['TIME_H'].to_numpy(),\n"
                  "    obs['CONC_MG_L'].to_numpy(), p0=[0.1, 50.0])",
     "other way round"),
    ("W3D3", "b", "se_c0 = float(np.diag(pcov)[0])", "variance, not the standard error"),
    ("W3D3", "b", "c0_hat = float(popt[1]); k_hat = float(popt[0])", "rate constant"),
    ("W3D3", "c", "t_half = 1.0 / k_hat", "1/k"),
    ("W3D3", "c", "rmse = float(np.sum((obs['CONC_MG_L'].to_numpy() - pred) ** 2))",
     "sum of squared residuals"),

    ("W3D4", "a", "conc_num = sol.y[0]", "still an AMOUNT"),
    ("W3D4", "a", "sol = solve_ivp(rhs, (0.0, 24.0), [DOSE_MG], t_eval=T_GRID,\n"
                  "    args=(CL_L_H, V_L))\nconc_num = sol.y[0] / V_L", "default tolerances"),
    ("W3D4", "b", "max_abs_err = float(np.max(conc_num - conc_exact))\n"
                  "max_abs_err = -abs(max_abs_err)", "negative"),
    ("W3D4", "c", "conc_inf = sol_inf.y[0]", "amount in mg"),

    ("W3D5", "a", "agg = assay.assign(MEAN_PCT=assay['INHIB_PCT'], SD_PCT=0.0, N=1)\n"
                  "agg = agg[['CONC_UM','MEAN_PCT','SD_PCT','N']]", "one row per well"),
    ("W3D5", "b", "emax1 = float(emax1) * 2.0", "not the fitted ceiling"),
    ("W3D5", "c", "hill = 1.0", "never moved"),
    ("W3D5", "c", "def sigmoid_model(c, emax, ec50, h):\n"
                  "    return emax * c ** h / (ec50 + c ** h)", "h is raised on the ec50 term"),
# ------------------------------------------------------------ week 4
    ("W4D1", "a", "mask = field1 <= otsu_t", "inverted"),
    ("W4D1", "a", "frac_fg = float(mask.mean()) * 100.0", "percent"),
    ("W4D1", "b", "otsu_t2 = otsu_t", "field1's threshold"),
    ("W4D1", "b", "frac_fixed_2 = frac_otsu_2", "FIXED_T was not actually used"),
    ("W4D1", "c", "n_obj_smooth = n_obj_raw", "equals the raw count"),

    ("W4D2", "a", "mask_open = morphology.opening(mask_raw, morphology.disk(3))",
     "radius 3 footprint"),
    ("W4D2", "a", "n_obj_open = n_obj_raw", "counted on mask_raw"),
    ("W4D2", "b", "mask_filled = morphology.closing(mask_open, morphology.disk(4))",
     "result of a closing"),
    ("W4D2", "b", "n_hole_px = int(mask_filled.sum())", "total pixel count"),
    ("W4D2", "c", "n_border_objects = n_obj_clean", "number REMAINING"),

    ("W4D3", "b", "cells = props[props['area'] >= MIN_AREA].reset_index(drop=True)\n"
                  "n_cells = int(len(cells)); n_dropped = int(len(props) - len(cells))",
     "only the lower bound"),
    ("W4D3", "b", "n_dropped = n_cells", "number KEPT"),
    ("W4D3", "c", "mean_area = float(props['area'].mean())", "every object"),
    ("W4D3", "c", "mean_intensity = float((cells['mean_intensity'] * cells['area']).sum()"
                  " / cells['area'].sum())", "pixel weighted"),

    ("W4D4", "a", "t_cv = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)",
     "tuple"),
    ("W4D4", "c", "areas = [float(cv2.arcLength(c, True)) for c in contours]", "perimeters"),
    ("W4D4", "c", "n_cells_cv = int(len(contours))", "counted every contour"),
    ("W4D4", "c", "area_ratio = 1.0 / area_ratio", "other way up"),

    ("W4D5", "b", "per_animal['MEAN_INTENSITY'] = [\n"
                  "    float((g['N_CELLS'] * g['MEAN_INTENSITY']).sum() / g['N_CELLS'].sum())\n"
                  "    for _, g in per_field.sort_values('ANIMAL').groupby('ANIMAL')]",
     "weighted by how many cells"),
    ("W4D5", "c", "group_stats['SD'] = group_stats['SD'] / np.sqrt(group_stats['N_ANIMALS'])",
     "SEM"),
    ("W4D5", "c", "group_stats = (per_field.groupby('GROUP')\n"
                  "    .agg(MEAN=('MEAN_INTENSITY','mean'), SD=('MEAN_INTENSITY','std'),\n"
                  "         N_ANIMALS=('MEAN_INTENSITY','size')).reset_index())\n"
                  "group_stats['_o'] = group_stats['GROUP'].map({g: i for i, g in enumerate(GROUP_ORDER)})\n"
                  "group_stats = group_stats.sort_values('_o').drop(columns=['_o']).reset_index(drop=True)",
     "counted FIELDS"),
# ------------------------------------------------------------ week 5
    ("W5D1", "a", "X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.25, random_state=42)",
     "unpacking order"),
    ("W5D1", "a", "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)",
     "random_state=42"),
    ("W5D1", "b", "r2_test = float(r2_score(y_train, model.predict(X_train)))", "training data"),
    ("W5D1", "b", "rmse_test = float(mean_squared_error(y_test, pred_test))", "not its root"),
    ("W5D1", "c", "optimism = float(r2_test - r2_train)", "sign the wrong way"),

    ("W5D2", "a", "cv_sd = float(cv_scores.std())", "ddof=0"),
    ("W5D2", "b", "X_leaky = X_panel", "no selection happened"),
    ("W5D2", "c", "cv_clean_mean = cv_leaky_mean", "pipeline was run on"),
    ("W5D2", "c", "leak_inflation = cv_clean_mean - cv_leaky_mean", "sign the wrong way"),

    ("W5D3", "a", "gap_rf = float(r2_test_rf - r2_train_rf)", "sign the wrong way"),
    ("W5D3", "b", "imp_impurity = imp_impurity.sort_values(ascending=True)", "not sorted descending"),
    ("W5D3", "c", "imp_perm = pd.Series(permutation_importance(rf, X_train, y_train,\n"
                  "    n_repeats=20, random_state=0).importances_mean,\n"
                  "    index=FEATURES).sort_values(ascending=False)", "TRAINING data"),
    ("W5D3", "c", "noise_rank_perm = noise_rank_perm - 1", "not the position"),

    ("W5D4", "a", "coef_raw = coef_scaled", "matches the standardised fit"),
    ("W5D4", "b", "path['R2_TEST'] = path['R2_TRAIN']", "training scores"),
    ("W5D4", "c", "n_nonzero_cv = len(FEATURES) - n_nonzero_cv", "coefficients that ARE zero"),

    ("W5D5", "a", "pca = PCA().fit(panel); evr = pca.explained_variance_ratio_", "RAW panel"),
    ("W5D5", "a", "evr_unscaled_first = float(evr[0])", "also fitted on scaled"),
    ("W5D5", "b", "cumulative = evr", "running total"),
    ("W5D5", "b", "n_for_90 = n_for_90 - 1", "0 based position"),
    ("W5D5", "c", "loadings = pd.DataFrame(pca.components_.T, columns=['PC%d' % (i+1)\n"
                  "    for i in range(pca.components_.shape[0])], index=FEATURES)", "transposed"),
    ("W5D5", "c", "pc1_top3 = pc1_top3[::-1]", "ascending order"),

    ("W6D1", "a", "t_tensor = torch.tensor(time_np, dtype=torch.float32)", "float64"),
    ("W6D1", "a", "t_tensor = time_np", "torch Tensor"),
    ("W6D1", "a", "dtype_default = torch.tensor([1, 2, 3], dtype=torch.float64).dtype",
     "a dtype was passed"),
    ("W6D1", "b", "cmax = c_tensor.max()", "still a tensor"),
    ("W6D1", "b", "tmax = float(torch.argmax(c_tensor).item())", "index of the peak"),
    ("W6D1", "b", "mask = c_tensor < LLOQ", "inverted"),
    ("W6D1", "b", "c_measurable = c_tensor", "still has every sample"),
    ("W6D1", "c", "has_grad_fn_plain = True", "should be False"),
    ("W6D1", "c", "shares_memory = False", "should be True"),

    ("W6D2", "a", "k_a = torch.tensor(K_START, dtype=torch.float64)", "requires_grad"),
    ("W6D2", "a", "loss_a = loss_a / 9.0", "MEAN squared residual"),
    ("W6D2", "a", "grad_auto = -grad_auto", "wrong sign"),
    ("W6D2", "a", "grad_manual = grad_manual / 2.0", "half the right value"),
    ("W6D2", "b", "grad_k = grad_v", "swapped"),
    ("W6D2", "b", "fd_v = (loss_at(V_START + FD_STEP, K_START)"
                  " - loss_at(V_START - FD_STEP, K_START)) / FD_STEP", "twice the gradient"),
    ("W6D2", "c", "grad_twice = grad_once", "did not accumulate"),
    ("W6D2", "c", "grad_after_zero = 3.0 * grad_once", "never cleared"),
    ("W6D2", "c", "second_backward_error = False", "should be True"),

    ("W6D3", "a", "loss_after_small = loss_before", "no step was taken"),
    ("W6D3", "a", "loss_after_big = loss_after_small", "LR_BIG was not used"),
    ("W6D3", "a", "inplace_error = False", "should be True"),
    ("W6D3", "b", "loss_hist = loss_hist[:100]", "N_ITER"),
    ("W6D3", "b", "v_hat = float(np.log(v_hat))", "still on the log scale"),
    ("W6D3", "b", "v_scipy = v_scipy * 1.001", "close but not converged"),
    ("W6D3", "c", "v_big = v_hat", "sensible fitted volume"),
    ("W6D3", "c", "loss_big = zero_model_loss * 2.0", "zero model loss"),
    ("W6D3", "c", "first_nan_index = 0", "nothing is"),

    ("W6D4", "a", "n_params = 1", "should be 2"),
    ("W6D4", "a", "param_names = ['log_v']", "log_k"),
    ("W6D4", "a", "loss_start = loss_start / 9.0", "mean, not the sum"),
    ("W6D4", "a", "class OneCompartmentIV(torch.nn.Module):\n"
                  "    def __init__(self, v_init, k_init):\n"
                  "        super().__init__()\n"
                  "        self.log_v = torch.tensor(np.log(v_init), dtype=torch.float64)\n"
                  "        self.log_k = torch.tensor(np.log(k_init), dtype=torch.float64)\n"
                  "    def forward(self, t):\n"
                  "        return (DOSE / torch.exp(self.log_v)) * torch.exp("
                  "-torch.exp(self.log_k) * t)", "torch.nn.Parameter"),
    ("W6D4", "b", "loss_hist_sgd = list(loss_hist_adam)", "plain SGD does"),
    ("W6D4", "b", "v_hat = float(np.log(v_hat))", "still on the log scale"),
    ("W6D4", "c", "cl_hat = k_hat / v_hat", "k divided by V"),
    ("W6D4", "c", "t_half_hat = 1.0 / k_hat", "mean residence time"),
    ("W6D4", "c", "resid_sd = float(np.sqrt(((model(t_obs) - c_obs) ** 2)"
                  ".sum().item() / t_obs.numel()))", "divides the SSE by n"),

    ("W6D5", "a", "subject_bias = (c_obs - model_pool(t_obs)).mean(dim=0).detach().numpy()",
     "one entry per TIME"),
    ("W6D5", "a", "subject_bias = -subject_bias", "sign flipped"),
    ("W6D5", "a", "sse_pool = sse_ind", "individual fit loss"),
    ("W6D5", "b", "v_ind = np.full(N_SUBJ, v_pool)", "pooled volume"),
    ("W6D5", "b", "sse_ind = sse_pool", "is the pooled loss"),
    ("W6D5", "c", "v_gm = float(np.mean(v_ind))", "arithmetic mean"),
    ("W6D5", "c", "cv_v_pct = cv_v_pct / 100.0", "fraction"),
    ("W6D5", "c", "sse_ratio = 1.0 / sse_ratio", "upside down"),

    ("W7D1", "a", "tmax = float(np.argmax(conc))", "argmax gives a position"),
    ("W7D1", "a", "auc_linear = auc_linlog", "lin-up/log-down value"),
    ("W7D1", "b", "auc_linlog = auc_linear", "log form was never applied"),
    ("W7D1", "b", "pct_diff = 100.0 * (auc_linear - auc_linlog) / auc_linlog",
     "other way round"),
    ("W7D1", "c", "auc_0_12 = auc_linlog", "window was not applied"),
    ("W7D1", "c", "frac_in_12h = frac_in_12h * 100.0", "is a percent"),

    ("W7D2", "a", "lambda_z = -lambda_z", "is negative"),
    ("W7D2", "a", "t_half = 1.0 / lambda_z", "factor of ln(2)"),
    ("W7D2", "b", "auc_inf = AUC_LAST", "no tail was added"),
    ("W7D2", "b", "pct_extrap = 100.0 * (auc_inf - AUC_LAST) / AUC_LAST",
     "denominator is auc_inf"),
    ("W7D2", "c", "cl_f = DOSE_MG / AUC_LAST", "used AUC_LAST"),
    ("W7D2", "c", "vz_f = cl_f * lambda_z", "multiplied by lambda_z"),

    ("W7D3", "a", "rhs = lambda t, y, ka, v, cl: [-ka * y[0], -ka * y[0] - (cl / v) * y[1]]",
     "absorption sign"),
    ("W7D3", "b", "rmse = rmse ** 2", "mean squared error"),
    ("W7D3", "c", "t_half_model = 1.0 / (cl_hat / v_hat)", "factor of ln(2)"),
    ("W7D3", "c", "cl_ratio = 1.0 / cl_ratio", "other way up"),

    ("W7D4", "a", "popt2 = popt2[::-1]", "does not match the fit"),
    ("W7D4", "b", "rmse1 = rmse1 ** 2", "mean squared error"),
    ("W7D4", "c", "vss = float(popt2[1])", "V1 alone"),
    ("W7D4", "c", "auc_bias = -auc_bias", "sign the wrong way round"),

    ("W7D5", "b", "cmax_first = cmax_ss", "steady state peak"),
    ("W7D5", "b", "r_sim = 1.0 / r_sim", "other way up"),
    ("W7D5", "c", "cavg_ss = DOSE_MG / CL_F", "left out the dosing interval"),
    ("W7D5", "c", "t_to_90pct = float(np.argmax(conc >= 0.9 * cmax_ss))", "is the index"),

    ("W8D1", "a", "e_falling = float(np.interp(C_PROBE, c_obs[peak_idx:], e_obs[peak_idx:]))",
     "decreasing array"),
    ("W8D1", "a", "tmax_c = float(peak_idx)", "index of the peak"),
    ("W8D1", "b", "resid_early = -resid_early", "should be negative"),
    ("W8D1", "b", "sse_direct = sse_link", "link model"),
    ("W8D1", "c", "emax_hat = emax_direct", "direct fit"),
    ("W8D1", "c", "sse_link = sse_direct", "direct fit"),

    ("W8D2", "a", "kin = R_BASE / KOUT", "divided by KOUT"),
    ("W8D2", "a", "pct_suppression = 100.0 * nadir / R_BASE", "level remaining"),
    ("W8D2", "a", "t_nadir = float(np.argmin(r_single))", "an index, not a time"),
    ("W8D2", "b", "t_above_by_dose = t_above_by_dose / DT", "count of grid points"),
    ("W8D2", "b", "t_link_by_dose = t_nadir_by_dose.copy()", "same model"),
    ("W8D2", "c", "auec = -auec", "wrong sign"),
    ("W8D2", "c", "t_nadir_fast = t_nadir", "did not reach the solver"),

    ("W8D3", "a", "cl_gm = cl_mean", "arithmetic mean"),
    ("W8D3", "a", "cl_cv_pct = float(100.0 * np.std(np.log(cl_i), ddof=1))", "approximation"),
    ("W8D3", "a", "mean_over_median_theory = float(np.exp(OMEGA_CL ** 2))", "division by 2"),
    ("W8D3", "b", "auc_p5, auc_p95 = auc_p95, auc_p5", "wrong way round"),
    ("W8D3", "b", "cmax_i = np.log(KA / (cl_i / v_i)) / (KA - cl_i / v_i)", "peak TIMES"),
    ("W8D3", "c", "cv_hat_12 = cv_hat_12[:50]", "N_STUDIES"),
    ("W8D3", "c", "cv12_hi = cv120_hi", "percentile it should be"),

    ("W8D4", "a", "nm = nm.sort_values(['ID', 'TIME', 'EVID']).reset_index(drop=True)\n"
                  "first_evid_per_id = nm.groupby('ID')['EVID'].first().values",
     "ahead of the dose"),
    ("W8D4", "a", "nm = nm[['TIME', 'ID', 'AMT', 'DV', 'EVID', 'MDV', 'CMT']]",
     "wrong order"),
    ("W8D4", "b", "n_blq = n_mdv", "including the dosing records"),
    ("W8D4", "b", "n_mdv = n_blq", "count the dosing rows too"),
    ("W8D4", "c", "auc_dn_low = float(np.mean(nca.loc[nca['AMT'] == 50.0, 'AUC_DN'].values))",
     "arithmetic mean"),
    ("W8D4", "c", "auc_dn_ratio = 1.0 / auc_dn_ratio", "upside down"),

    ("W8D5", "a", "c_typ = (DOSE_MID * KA / (V_POP * (KA - k_typ))"
                  " * (np.exp(-k_typ * t_grid) - np.exp(-KA * t_grid)))", "single dose profile"),
    ("W8D5", "a", "cavg_ss = cavg_ss * TAU", "area, not the average"),
    ("W8D5", "b", "p_tox = 1.0 - p_tox", "fraction within the limit"),
    ("W8D5", "b", "p_success = p_eff * (1.0 - p_tox)", "independent"),
    ("W8D5", "b", "cmax_i = cmax_i[:10]", "patients"),
    ("W8D5", "c", "best_dose = float(np.argmax(p_success_by_dose))", "index of the best"),
    ("W8D5", "c", "p_eff_by_dose = p_eff_by_dose[::-1]", "reverse dose order"),
]






def load_specs():
    out = {}
    for path in sorted(SPEC_DIR.glob("w*.py")):
        spec_obj = importlib.util.spec_from_file_location("spec_" + path.stem, path)
        mod = importlib.util.module_from_spec(spec_obj)
        spec_obj.loader.exec_module(mod)
        out[mod.PID] = mod
    return out


def check_cell_source(spec):
    nb_path = ROOT / "problems" / ("week%02d" % spec.WEEK) / (spec.SLUG + ".ipynb")
    nb = nbformat.read(str(nb_path), as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "code" and "_EXP = _json.loads" in cell.source:
            lines = [ln for ln in cell.source.splitlines() if ln.strip() != "check()"]
            return "\n".join(lines)
    raise RuntimeError("no check cell found in %s" % nb_path)


def run_case(spec, check_src, part, break_code):
    ns = {"__name__": "__neg__"}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(spec.DATA_CODE, "<data>", "exec"), ns)
        for p in spec.PARTS:
            exec(compile(p["solution"], "<sol>", "exec"), ns)
        exec(compile(check_src, "<check>", "exec"), ns)
        exec(compile(break_code, "<break>", "exec"), ns)
        try:
            import matplotlib.pyplot as _plt
            _plt.close("all")
        except Exception:
            pass
        fn = ns["_part_" + part]
        try:
            result = fn()
            kind = "FAIL" if result else "PASS"
        except ns["_Todo"] as exc:
            result, kind = str(exc), "TODO"
    return kind, (result or "")


def main():
    wanted = [a.upper() for a in sys.argv[1:]]
    cases = [c for c in CASES if not wanted or c[0] in wanted]
    specs = load_specs()
    needed = sorted({c[0] for c in cases})
    sources = {pid: check_cell_source(specs[pid]) for pid in needed}

    bad = []
    for pid, part, break_code, expect in cases:
        spec = specs[pid]
        try:
            kind, msg = run_case(spec, sources[pid], part, break_code)
        except Exception as exc:
            bad.append("%s part %s: the test itself blew up: %s: %s" % (pid, part, type(exc).__name__, exc))
            print("ERROR  %s %s  %s" % (pid, part, break_code.splitlines()[0][:50]))
            continue
        if kind == "PASS":
            bad.append("%s part %s: a wrong answer was accepted: %s" % (pid, part, break_code.splitlines()[0][:70]))
            print("MISSED %s %s  %s" % (pid, part, break_code.splitlines()[0][:50]))
        elif expect.lower() not in msg.lower():
            bad.append("%s part %s: caught it but the message does not mention %r.\n"
                       "        got: %s" % (pid, part, expect, msg))
            print("WRONGMSG %s %s  %s" % (pid, part, break_code.splitlines()[0][:48]))
        else:
            print("caught %s %s  %s" % (pid, part, break_code.splitlines()[0][:50]))

    print("")
    if bad:
        print("PROBLEMS (%d of %d cases):" % (len(bad), len(cases)))
        for b in bad:
            print("  - " + b)
        return 1
    print("All %d wrong answers were caught with a useful message." % len(cases))
    return 0


if __name__ == "__main__":
    sys.exit(main())
