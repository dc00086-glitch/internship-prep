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
