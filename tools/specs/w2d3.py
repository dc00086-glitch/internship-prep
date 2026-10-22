PID = "W2D3"
WEEK = 2
DAY = 3
SLUG = "w2d3_pandas_nominal_actual"
TITLE = "Nominal versus actual sampling times, and the samples that never arrived"
TOPIC = "pandas: time deviations, building a full grid, finding missing records"
EST = "20-30 min"

SCENARIO_MD = '''
## Scenario

The protocol says blood is drawn at 0.5, 1, 2, 4, 6, 8, 12 and 24 hours after
dosing, in twelve subjects. Reality says a nurse was with another patient, a
cannula blocked, and one subject went home early.

So you have two different times for every sample. The **nominal** time is what
the protocol asked for and what every summary table is grouped by. The
**actual** time is when the needle really went in, recorded to the minute, and
it is what any model must be fitted against.

Two questions have to be answered before this dataset is usable. How far did
the actual times drift from nominal, and did any drift far enough to count as
a protocol deviation. And which scheduled samples are simply not there, which
is harder than it sounds, because a sample that was never collected leaves no
row behind to notice.
'''

CONCEPT_MD = '''
## You cannot find a missing row by looking at the rows you have

This is the idea worth taking away, and it generalises far beyond sampling
times.

Everything you have done so far filters, groups or joins the rows that exist.
None of that can find a row that is **absent**, because absence is not a value
sitting in a column. A subject missing their 8 hour sample does not have a row
saying so. They have no row at all, and `isna()` will never find it.

The way out is to build the set of rows that **should** exist, then compare.
You know the design: twelve subjects, eight nominal timepoints, so ninety-six
samples were scheduled. `pd.MultiIndex.from_product` builds exactly that grid
from the two lists, every combination, in one line. Join your real data onto
that grid with an indicator and the gaps become ordinary rows you can count,
group and report.

The same move solves a whole family of problems: a dose level with no
subjects, a visit nobody attended, a day with no data in a time series. Build
the complete frame, join the observed data onto it, and the holes are visible.

**On time windows.** Protocols allow a tolerance that widens as time goes on,
because being three minutes late matters enormously at half an hour, when the
concentration is moving fast, and barely at all at 24 hours, when it is
crawling. So the allowed window is usually tight early and loose late. In code
that is a condition that depends on another column, which is exactly what
`np.where` is for: pick one tolerance where the nominal time is early, another
where it is late, then compare the absolute deviation against it.

**On which time to use.** Group and tabulate by **nominal** time, because that
is what makes subjects comparable and what the protocol defines. Fit and
integrate against **actual** time, because that is what the drug experienced.
Using nominal time in a model quietly introduces error that looks like
variability, and it is a mistake that survives all the way into a report
because nothing ever complains.
'''

TASK_MD = '''
## Your task

Three parts, worked one at a time.

| Part | What you work out | Variables to fill in |
|---|---|---|
| a | how far each sample drifted, and which broke the window | `df`, `n_out_of_window` |
| b | which scheduled samples never arrived | `grid`, `missing`, `n_missing` |
| c | a per subject completeness and drift report | `per_subject` |

Below this you will find the **Setup** cell, then a **Toolkit** cell that
demonstrates every move on a tiny made up example, then **Part a**, **Part b**
and **Part c**, each with its own explanation and a cell to write in.
'''

DATA_CODE = '''
import numpy as np
import pandas as pd

PROBLEM_ID = "W2D3"
TOPIC = "pandas: time deviations, building a full grid, finding missing records"

rng = np.random.default_rng(2103)

NOMINAL_TIMES = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0]
SUBJECTS = ["T%02d" % i for i in range(1, 13)]

_rows = []
for _subj in SUBJECTS:
    _V = 38.0 * rng.lognormal(0.0, 0.20)
    _kel = 0.105 * rng.lognormal(0.0, 0.22)
    for _nom in NOMINAL_TIMES:
        # early draws are kept tight, later ones drift more
        _spread = 2.0 if _nom < 2.0 else 12.0
        _actual = _nom + rng.normal(0.0, _spread) / 60.0
        _conc = (120.0 / _V) * np.exp(-_kel * max(_actual, 0.0)) * rng.lognormal(0.0, 0.10)
        _rows.append((_subj, _nom, round(_actual, 4), round(_conc, 3)))

samples = pd.DataFrame(_rows, columns=["SUBJECT", "NOMINAL_H", "ACTUAL_H", "CONC_MG_L"])

# Reality: a blocked cannula, a missed draw, and one subject who went home early.
_dropped = [("T03", 8.0), ("T07", 2.0), ("T11", 12.0), ("T11", 24.0), ("T05", 24.0)]
_mask = pd.Series(True, index=samples.index)
for _s, _t in _dropped:
    _mask &= ~((samples["SUBJECT"] == _s) & (samples["NOMINAL_H"] == _t))
samples = samples[_mask].reset_index(drop=True)

# A few draws ran badly late.
for _i, _extra in [(4, 0.45), (29, 0.70), (52, 0.55)]:
    samples.loc[_i, "ACTUAL_H"] = round(samples.loc[_i, "ACTUAL_H"] + _extra, 4)

print("samples:", samples.shape)
print(samples.head(10))
print("")
print("scheduled:", len(SUBJECTS), "subjects x", len(NOMINAL_TIMES), "timepoints =",
      len(SUBJECTS) * len(NOMINAL_TIMES))
'''

TOOLKIT_MD = '''
## Toolkit for this problem

| What you want to do | How |
|---|---|
| every combination of two lists | `pd.MultiIndex.from_product([a, b], names=[...]).to_frame(index=False)` |
| absolute value of a column | `df["D"].abs()` |
| pick a value based on a condition | `np.where(cond, if_true, if_false)` |
| label where each row came from | `pd.merge(..., how="left", indicator=True)` |
| rows present in the grid but not the data | `df[df["_merge"] == "left_only"]` |
| count True values in a column | `int(df["FLAG"].sum())` |
| several statistics at once, named | `.agg(N=("C", "size"), WORST=("D", "max"))` |

A tolerance that changes with time is a one line `np.where`, not an `if`. An
`if` asks a whole column to be a single True or False and raises
`ValueError: The truth value of a Series is ambiguous`.
'''

TOOLKIT_CODE = '''
import numpy as np
import pandas as pd

# A tiny made up example. Nothing here is the answer.
demo = pd.DataFrame({
    "ID":  ["a", "a", "b"],
    "NOM": [1.0, 4.0, 1.0],
    "ACT": [1.05, 4.60, 0.98],
})
print("demo, two subjects and two scheduled times, one sample missing:")
print(demo)

print("")
print("--- 1. deviation in minutes, and its size ---")
demo["DEV_MIN"] = (demo["ACT"] - demo["NOM"]) * 60.0
print(demo)
print("")
print("abs() gives the size regardless of direction:", demo["DEV_MIN"].abs().round(1).to_list())

print("")
print("--- 2. a tolerance that depends on another column ---")
demo_tol = np.where(demo["NOM"] < 2.0, 5.0, 30.0)
print("tolerance per row:", demo_tol)
demo["OUT"] = demo["DEV_MIN"].abs() > demo_tol
print(demo)
print("how many out of window:", int(demo["OUT"].sum()))
print("")
print("an if would not work here:")
try:
    if demo["NOM"] < 2.0:
        pass
except ValueError as err:
    print("  ValueError:", err)

print("")
print("--- 3. building the grid of rows that SHOULD exist ---")
demo_grid = pd.MultiIndex.from_product(
    [["a", "b"], [1.0, 4.0]], names=["ID", "NOM"]).to_frame(index=False)
print(demo_grid)
print("")
print("scheduled:", len(demo_grid), " collected:", len(demo))

print("")
print("--- 4. joining the data onto the grid finds the holes ---")
demo_check = pd.merge(demo_grid, demo, on=["ID", "NOM"], how="left", indicator=True)
print(demo_check[["ID", "NOM", "ACT", "_merge"]])
print("")
demo_missing = demo_check[demo_check["_merge"] == "left_only"]
print("missing rows:")
print(demo_missing[["ID", "NOM"]])
print("")
print("note the direction: the GRID is on the left, so left_only means scheduled")
print("but never collected")
'''

PARTS = [
    dict(
        label="a",
        prompt_md='''
## Part a. How far did each sample drift

Work out the deviation of every collected sample from its nominal time, in
**minutes**, signed, so that a late draw is positive and an early one
negative. Keep the sign: a systematic lateness is a different finding from
random scatter, and taking the absolute value too early hides it.

Then flag the protocol deviations. The window in this study is the usual
widening one:

- nominal time **under 2 hours**: allowed within 5 minutes,
- nominal time **2 hours or later**: allowed within 30 minutes.

The tolerance therefore depends on the nominal time of each row, which is a
`np.where`, not an `if`. Compare the **size** of the deviation against it, so
being 40 minutes early counts just as much as 40 minutes late.

Fill in `df`, a copy of `samples` with two columns added: `DEV_MIN`, the
signed deviation in minutes, and `OUT_OF_WINDOW`, a boolean. And
`n_out_of_window`, a whole number.

Moves: arithmetic between columns, `np.where`, `.abs()`, `.sum()`.
''',
        starter='''
# Part a. Deviation from nominal time, and the protocol window.

df = None               # YOUR CODE: samples plus DEV_MIN and OUT_OF_WINDOW columns
n_out_of_window = None  # YOUR CODE: how many samples fell outside the window

print("n_out_of_window:", n_out_of_window)
if df is not None:
    print(df.head())
    print("")
    print(df[df["OUT_OF_WINDOW"]])
''',
        solution='''
# Part a. Deviation from nominal time, and the protocol window.

df = samples.copy()
df["DEV_MIN"] = (df["ACTUAL_H"] - df["NOMINAL_H"]) * 60.0
_tol = np.where(df["NOMINAL_H"] < 2.0, 5.0, 30.0)
df["OUT_OF_WINDOW"] = df["DEV_MIN"].abs() > _tol
n_out_of_window = int(df["OUT_OF_WINDOW"].sum())

print("n_out_of_window:", n_out_of_window)
if df is not None:
    print(df.head())
    print("")
    print(df[df["OUT_OF_WINDOW"]])
''',
    ),
    dict(
        label="b",
        prompt_md='''
## Part b. Which scheduled samples never arrived

A sample that was never collected has no row, so no amount of filtering the
data you have will find it. Build the rows that **should** exist, then see
which of them your data cannot match.

`pd.MultiIndex.from_product` takes your list of subjects and your list of
nominal times and produces every combination, which is the full scheduled
design. Put the grid on the **left** of the merge and the collected samples on
the right, with `indicator=True`. Rows labelled `left_only` were scheduled and
never collected.

Getting the direction right matters. Grid on the left answers "what is
missing". Data on the left would answer "what did we collect that was not
scheduled", which is a different and also useful question, but not this one.

Fill in `grid`, a DataFrame with one row per scheduled sample and columns
`SUBJECT` and `NOMINAL_H`. `missing`, the `SUBJECT` and `NOMINAL_H` of the
samples that never arrived, sorted by subject then time, index renumbered. And
`n_missing`, a whole number.

Moves: `pd.MultiIndex.from_product([...], names=[...]).to_frame(index=False)`,
`pd.merge(..., how="left", indicator=True)`, filtering on `_merge`,
`.sort_values(...).reset_index(drop=True)`.
''',
        starter='''
# Part b. Find the samples that were scheduled but never collected.

grid = None       # YOUR CODE: every SUBJECT and NOMINAL_H combination that was scheduled
missing = None    # YOUR CODE: SUBJECT and NOMINAL_H of the samples that never arrived,
                  #            sorted by SUBJECT then NOMINAL_H, index renumbered
n_missing = None  # YOUR CODE: how many

print("scheduled:", None if grid is None else len(grid))
print("collected:", len(samples))
print("n_missing:", n_missing)
print(missing)
''',
        solution='''
# Part b. Find the samples that were scheduled but never collected.

grid = pd.MultiIndex.from_product(
    [SUBJECTS, NOMINAL_TIMES], names=["SUBJECT", "NOMINAL_H"]).to_frame(index=False)

_checked = pd.merge(grid, samples, on=["SUBJECT", "NOMINAL_H"],
                    how="left", indicator=True)
missing = (_checked.loc[_checked["_merge"] == "left_only", ["SUBJECT", "NOMINAL_H"]]
           .sort_values(["SUBJECT", "NOMINAL_H"])
           .reset_index(drop=True))
n_missing = int(len(missing))

print("scheduled:", None if grid is None else len(grid))
print("collected:", len(samples))
print("n_missing:", n_missing)
print(missing)
''',
    ),
    dict(
        label="c",
        prompt_md='''
## Part c. A per subject report

Now turn both findings into something you could actually send to a data
manager: one row per subject, so whoever reads it can see at a glance who
needs chasing.

For each subject give how many samples were collected, how many were missed,
how many fell outside the window, and the largest absolute deviation in
minutes that subject experienced. Round that last one to one decimal so it
reads cleanly.

The missed count comes from part b, and a subject who missed nothing will not
appear in `missing` at all. They still need a row with zero rather than being
dropped, which is the same "absence is not a value" problem as before, solved
the same way: start from the full subject list and fill the gaps.

Fill in `per_subject`, sorted by `SUBJECT`, index renumbered, with exactly:

| column | meaning |
|---|---|
| `SUBJECT` | subject identifier |
| `N_COLLECTED` | samples actually collected |
| `N_MISSING` | scheduled samples that never arrived |
| `N_OUT_OF_WINDOW` | collected samples outside the allowed window |
| `MAX_ABS_DEV_MIN` | largest absolute deviation in minutes, rounded to 1 dp |

Moves: `.groupby("SUBJECT").agg(...)`, `pd.merge(..., how="left")`,
`.fillna(0)`, `.astype(int)`, `.round(1)`.
''',
        starter='''
# Part c. One row per subject: collected, missing, out of window, worst drift.

per_subject = None   # YOUR CODE

print(per_subject)
''',
        solution='''
# Part c. One row per subject: collected, missing, out of window, worst drift.

_collected = (df
              .assign(ABS_DEV=df["DEV_MIN"].abs())
              .groupby("SUBJECT")
              .agg(N_COLLECTED=("CONC_MG_L", "size"),
                   N_OUT_OF_WINDOW=("OUT_OF_WINDOW", "sum"),
                   MAX_ABS_DEV_MIN=("ABS_DEV", "max"))
              .reset_index())

_missed = (missing.groupby("SUBJECT").size().reset_index(name="N_MISSING"))

per_subject = pd.merge(_collected, _missed, on="SUBJECT", how="left")
per_subject["N_MISSING"] = per_subject["N_MISSING"].fillna(0).astype(int)
per_subject["N_OUT_OF_WINDOW"] = per_subject["N_OUT_OF_WINDOW"].astype(int)
per_subject["MAX_ABS_DEV_MIN"] = per_subject["MAX_ABS_DEV_MIN"].round(1)
per_subject = (per_subject[["SUBJECT", "N_COLLECTED", "N_MISSING",
                            "N_OUT_OF_WINDOW", "MAX_ABS_DEV_MIN"]]
               .sort_values("SUBJECT")
               .reset_index(drop=True))

print(per_subject)
''',
    ),
]


def build_expected(ns):
    import numpy as np
    import pandas as pd
    samples = ns["samples"]
    df = ns["df"].sort_values(["SUBJECT", "NOMINAL_H"]).reset_index(drop=True)
    ps = ns["per_subject"].sort_values("SUBJECT").reset_index(drop=True)

    flat5 = (df["DEV_MIN"].abs() > 5.0)
    flat30 = (df["DEV_MIN"].abs() > 30.0)
    unsigned = df["DEV_MIN"].abs()
    hours = (ns["df"].sort_values(["SUBJECT", "NOMINAL_H"])["ACTUAL_H"].to_numpy()
             - ns["df"].sort_values(["SUBJECT", "NOMINAL_H"])["NOMINAL_H"].to_numpy())
    return {
        "n_samples": int(len(samples)),
        "n_scheduled": int(len(ns["SUBJECTS"]) * len(ns["NOMINAL_TIMES"])),
        "df_subject": [str(v) for v in df["SUBJECT"]],
        "df_nominal": [float(v) for v in df["NOMINAL_H"]],
        "dev_min": [float(v) for v in df["DEV_MIN"]],
        "dev_abs": [float(v) for v in unsigned],
        "dev_hours": [float(v) for v in hours],
        "out_flags": [bool(v) for v in df["OUT_OF_WINDOW"]],
        "n_out": int(ns["n_out_of_window"]),
        "n_out_flat5": int(flat5.sum()),
        "n_out_flat30": int(flat30.sum()),
        "grid_cols": ["SUBJECT", "NOMINAL_H"],
        "n_missing": int(ns["n_missing"]),
        "missing_subject": [str(v) for v in ns["missing"]["SUBJECT"]],
        "missing_time": [float(v) for v in ns["missing"]["NOMINAL_H"]],
        "ps_cols": ["SUBJECT", "N_COLLECTED", "N_MISSING", "N_OUT_OF_WINDOW",
                    "MAX_ABS_DEV_MIN"],
        "ps_subject": [str(v) for v in ps["SUBJECT"]],
        "ps_collected": [int(v) for v in ps["N_COLLECTED"]],
        "ps_missing": [int(v) for v in ps["N_MISSING"]],
        "ps_out": [int(v) for v in ps["N_OUT_OF_WINDOW"]],
        "ps_maxdev": [float(v) for v in ps["MAX_ABS_DEV_MIN"]],
    }


CHECK_CODE = '''
def _part_a():
    df = _need("df")
    n_out = _need("n_out_of_window")

    if not isinstance(df, _pd.DataFrame):
        return "`df` should be a pandas DataFrame."
    cols = _cols_of(df) or []
    for name in ("DEV_MIN", "OUT_OF_WINDOW"):
        if name not in cols:
            return "`df` has no %s column yet." % name
    if len(df) != _EXP["n_samples"]:
        return ("`df` has %d rows but samples has %d. Part a adds columns, it does not "
                "add or remove rows." % (len(df), _EXP["n_samples"]))

    got = df.sort_values(["SUBJECT", "NOMINAL_H"])
    dev = got["DEV_MIN"].to_numpy()
    if _arr_close(dev, _EXP["dev_hours"]):
        return ("`DEV_MIN` is in hours, not minutes. Multiply the difference by 60.")
    if _arr_close(dev, _EXP["dev_abs"]):
        return ("`DEV_MIN` has had its sign removed. Keep it signed, so a late draw is "
                "positive, and take the absolute value only when you compare it to the "
                "window.")
    if _arr_close(dev, [-v for v in _EXP["dev_min"]]):
        return ("`DEV_MIN` has the sign the wrong way round. It is actual minus nominal, "
                "so running late gives a positive number.")
    if not _arr_close(dev, _EXP["dev_min"]):
        return "`DEV_MIN` is not (ACTUAL_H - NOMINAL_H) expressed in minutes."

    flags = got["OUT_OF_WINDOW"].to_numpy()
    if flags.dtype != bool:
        return ("`OUT_OF_WINDOW` should be a boolean column, not %s. A comparison gives "
                "booleans directly." % flags.dtype)

    val = _as_scalar(n_out)
    if val is None:
        return "`n_out_of_window` should be a single whole number."
    got_n = int(round(val))
    if got_n == _EXP["n_out_flat5"] and _EXP["n_out_flat5"] != _EXP["n_out"]:
        return ("`n_out_of_window` used a flat 5 minute window everywhere. The window "
                "widens to 30 minutes from 2 hours onwards.")
    if got_n == _EXP["n_out_flat30"] and _EXP["n_out_flat30"] != _EXP["n_out"]:
        return ("`n_out_of_window` used a flat 30 minute window everywhere. The early "
                "timepoints, under 2 hours, only allow 5 minutes.")
    if got_n != _EXP["n_out"]:
        return ("`n_out_of_window` does not match. Compare the absolute deviation against "
                "a tolerance of 5 minutes below 2 hours and 30 minutes from 2 hours on.")
    if [bool(v) for v in flags] != _EXP["out_flags"]:
        return "`OUT_OF_WINDOW` flags the wrong rows, even though the count happens to match."
    return None


def _part_b():
    grid = _need("grid")
    missing = _need("missing")
    n_missing = _need("n_missing")

    if not isinstance(grid, _pd.DataFrame):
        return ("`grid` should be a pandas DataFrame. .to_frame(index=False) turns the "
                "MultiIndex into one.")
    gcols = _cols_of(grid) or []
    for name in _EXP["grid_cols"]:
        if name not in gcols:
            return ("`grid` has no %s column. Pass names=['SUBJECT', 'NOMINAL_H'] to "
                    "from_product." % name)
    if len(grid) == _EXP["n_samples"]:
        return ("`grid` has the same number of rows as the collected samples, so it is not "
                "the full scheduled design. It should have one row per subject per nominal "
                "timepoint, whether or not that sample exists.")
    if len(grid) != _EXP["n_scheduled"]:
        return ("`grid` has %d rows but %d samples were scheduled, which is the number of "
                "subjects times the number of nominal times."
                % (len(grid), _EXP["n_scheduled"]))

    if not isinstance(missing, _pd.DataFrame):
        return "`missing` should be a pandas DataFrame."
    mcols = _cols_of(missing) or []
    for name in _EXP["grid_cols"]:
        if name not in mcols:
            return "`missing` has no %s column." % name
    if len(missing) == _EXP["n_scheduled"] - _EXP["n_samples"] and False:
        pass
    if len(missing) == _EXP["n_samples"]:
        return ("`missing` holds the collected samples, not the absent ones. Rows labelled "
                "left_only are the ones in the grid with no matching sample.")
    if len(missing) != _EXP["n_missing"]:
        return ("`missing` has %d rows. It should hold only the scheduled samples that "
                "never arrived." % len(missing))
    if [str(v) for v in missing["SUBJECT"]] != _EXP["missing_subject"]:
        if sorted(str(v) for v in missing["SUBJECT"]) == sorted(_EXP["missing_subject"]):
            return "`missing` holds the right rows but is not sorted by SUBJECT then NOMINAL_H."
        return "`missing` does not name the right subjects."
    if not _arr_close(missing["NOMINAL_H"].to_numpy(), _EXP["missing_time"]):
        return "`missing` does not name the right timepoints."
    if list(missing.index) != list(range(len(missing))):
        return "`missing` has not had its index renumbered. Use .reset_index(drop=True)."

    val = _as_scalar(n_missing)
    if val is None or int(round(val)) != _EXP["n_missing"]:
        return "`n_missing` should be the number of rows in missing."
    return None


def _part_c():
    ps = _need("per_subject")
    if not isinstance(ps, _pd.DataFrame):
        return "`per_subject` should be a pandas DataFrame."
    cols = _cols_of(ps) or []
    missing_cols = [c for c in _EXP["ps_cols"] if c not in cols]
    if missing_cols:
        return "`per_subject` is missing the column(s) %s." % ", ".join(missing_cols)
    if len(ps) != len(_EXP["ps_subject"]):
        return ("`per_subject` has %d rows but there are %d subjects. A subject who missed "
                "nothing still needs a row." % (len(ps), len(_EXP["ps_subject"])))

    got = ps.sort_values("SUBJECT").reset_index(drop=True)
    if [str(v) for v in got["SUBJECT"]] != _EXP["ps_subject"]:
        return "`per_subject` does not list every subject once, sorted by SUBJECT."
    if not _arr_close(got["N_COLLECTED"].to_numpy(), _EXP["ps_collected"]):
        return "`N_COLLECTED` is not the number of samples each subject actually has."
    if _np.any(_pd.isna(got["N_MISSING"].to_numpy())):
        return ("`N_MISSING` has NaN in it, for the subjects who missed nothing and so do "
                "not appear in `missing`. Fill those with 0 after the join.")
    if not _arr_close(got["N_MISSING"].to_numpy(), _EXP["ps_missing"]):
        return ("`N_MISSING` does not match part b. Subjects who missed nothing should "
                "show 0, not be dropped.")
    if not _arr_close(got["N_OUT_OF_WINDOW"].to_numpy(), _EXP["ps_out"]):
        return ("`N_OUT_OF_WINDOW` is not the per subject count of out of window samples. "
                "Summing a boolean column counts the True values.")
    if not _arr_close(got["MAX_ABS_DEV_MIN"].to_numpy(), _EXP["ps_maxdev"], rtol=1e-3):
        if _arr_close(_np.abs(got["MAX_ABS_DEV_MIN"].to_numpy()),
                      _EXP["ps_maxdev"], rtol=1e-3):
            return ("`MAX_ABS_DEV_MIN` came out negative somewhere, so you took the max of "
                    "the signed deviation. Take the absolute value first.")
        return ("`MAX_ABS_DEV_MIN` is not the largest absolute deviation per subject, "
                "rounded to 1 decimal place.")
    return None
'''

# Functions this problem introduces. Keys must exist in tools/function_catalog.py.
FUNCTIONS = [
    "pd.MultiIndex.from_product",
    ".abs",
    ".fillna",
    ".assign",
]
