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
    specs = load_specs()
    sources = {pid: check_cell_source(spec) for pid, spec in specs.items()}

    bad = []
    for pid, part, break_code, expect in CASES:
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
        print("PROBLEMS (%d of %d cases):" % (len(bad), len(CASES)))
        for b in bad:
            print("  - " + b)
        return 1
    print("All %d wrong answers were caught with a useful message." % len(CASES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
