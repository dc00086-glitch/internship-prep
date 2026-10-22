PID = "W2D1"
WEEK = 2
DAY = 1
SLUG = "w2d1_pandas_transform_baseline"
TITLE = "Baseline correction: transform, and why it is not agg"
TOPIC = "pandas: groupby.transform, baseline correction, alignment"
EST = "20-30 min"

SCENARIO_MD = '''
## Scenario

A phase 1 study measured a target engagement biomarker in 15 subjects, five
each on placebo, a low dose and a high dose. Every subject was sampled at
baseline and at six timepoints after dosing.

The file arrives in assay run order, because that is how the bioanalytical lab
reports, so the rows are in no useful order at all.

Subjects start from very different baselines, which is normal for almost any
biomarker. Comparing raw values between groups would mostly tell you who
happened to start high. So the readout everyone actually reports is **percent
change from that subject's own baseline**, which means every row needs to know
its own subject's baseline value.

That is a different shape of problem from anything in week 1. A `groupby` that
summarises collapses 105 rows down to 15. Here you need to go the other way:
compute one number per subject, then put it back on all seven of that
subject's rows.
'''

CONCEPT_MD = '''
## agg collapses, transform aligns

This is the pandas idea that unlocks most real analysis work, and it is worth
being precise about.

`groupby(...).agg(...)` **collapses**. You hand it 105 rows and 15 subjects,
and you get 15 rows back, one per subject. The result has a different shape
from the input, so you cannot simply attach it as a new column.

`groupby(...).transform(...)` **aligns**. You hand it the same 105 rows, it
computes per group exactly like `agg` does, and then it broadcasts each
group's answer back out over every row of that group. You get 105 values back,
in the original row order, so it drops straight into a new column:

```python
df["SUBJ_MEAN"] = df.groupby("SUBJECT")["VALUE"].transform("mean")
```

Every row now carries its own subject's mean. No merge, no index juggling, no
risk of rows being reordered under you.

**Where this matters.** Any "relative to its own group" quantity is a
transform: percent change from baseline, deviation from a subject's own mean,
normalising each plate to its own control wells, each animal to its own
pre-dose value. Doing these with `agg` plus `merge` works, but it is three
steps instead of one and every one of them is a chance to lose or duplicate
rows.

**The catch with `"first"`.** To get a baseline you want the value at time
zero, and the natural shorthand is `transform("first")`, which takes the first
row of each group **as the data currently sits**. It does not look at your
time column and it has no idea what baseline means. If the rows are not sorted
so that time zero comes first within each subject, you will silently pick up
the wrong value, with no error. So you sort first, deliberately, and you check.

This is a recurring pattern in pandas: a convenient function that depends on
row order, used on data whose order nobody guaranteed. Sort first, every time.

**A third option worth knowing.** `groupby(...).apply(...)` can do anything,
because you hand it a whole chunk of the frame. It is also much slower and its
return shape is hard to predict. Reach for `agg` when you want a summary,
`transform` when you want to stay aligned, and `apply` only when neither fits.
'''

TASK_MD = '''
## Your task

Three parts, worked one at a time.

| Part | What you work out | Variables to fill in |
|---|---|---|
| a | the data sorted, and each subject's baseline on every row | `n_subjects`, `df`, `baseline_ok` |
| b | percent change from baseline, post-dose rows only | `df_post` |
| c | the group level summary of that change | `summary` |

Below this you will find the **Setup** cell, then a **Toolkit** cell that
demonstrates every move on a tiny made up example, then **Part a**, **Part b**
and **Part c**, each with its own explanation and a cell to write in.
'''

DATA_CODE = '''
import numpy as np
import pandas as pd

PROBLEM_ID = "W2D1"
TOPIC = "pandas: groupby.transform, baseline correction, alignment"

rng = np.random.default_rng(2101)

times = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0])
groups = np.repeat(["Placebo", "Low", "High"], 5)
subjects = ["S%02d" % i for i in range(1, 16)]
_effect = {"Placebo": 0.0, "Low": 0.28, "High": 0.55}

_rows = []
for _subj, _grp in zip(subjects, groups):
    _base = 100.0 * rng.lognormal(0.0, 0.30)          # baselines differ a lot
    _shape = np.exp(-0.09 * times) - np.exp(-1.1 * times)
    _shape = _shape / _shape.max()
    _true = _base * (1.0 - _effect[_grp] * _shape)
    _rows.append(np.round(_true * rng.lognormal(0.0, 0.05, size=times.size), 2))

raw = pd.DataFrame({
    "SUBJECT": np.repeat(subjects, times.size),
    "GROUP": np.repeat(groups, times.size),
    "TIME_H": np.tile(times, len(subjects)),
    "BIOMARKER": np.concatenate(_rows),
})

# The bioanalytical lab reports in assay run order, so the rows arrive scrambled.
raw = raw.sample(frac=1.0, random_state=7).reset_index(drop=True)

print("raw:", raw.shape)
print(raw.head(8))
print("")
print("note the row order: it is assay run order, which means nothing useful")
'''

TOOLKIT_MD = '''
## Toolkit for this problem

| What you want to do | How |
|---|---|
| summarise, one row per group | `df.groupby("ID")["V"].agg("mean")` |
| the same number, on every row | `df.groupby("ID")["V"].transform("mean")` |
| the first row of each group | `.transform("first")`, which depends on row order |
| sort, and renumber the index | `df.sort_values([...], ignore_index=True)` |
| add a column from two others | `df["PCT"] = 100 * (df["A"] - df["B"]) / df["B"]` |
| keep some rows | `df[df["TIME_H"] > 0]` |
| check something is true everywhere | `(df["A"] == df["B"]).all()` |

`ignore_index=True` on `sort_values` is the same as following it with
`.reset_index(drop=True)`, just shorter.
'''

TOOLKIT_CODE = '''
import numpy as np
import pandas as pd

# A tiny made up table. Nothing here is the answer.
demo = pd.DataFrame({
    "ID":   ["a", "b", "a", "b", "a", "b"],
    "TIME": [0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
    "V":    [10.0, 50.0, 7.0, 40.0, 5.0, 25.0],
})
print("demo, sorted by TIME like the real data:")
print(demo)

print("")
print("--- 1. agg collapses, transform aligns ---")
print("agg gives one row per group:")
print(demo.groupby("ID")["V"].agg("mean"))
print("")
print("transform gives one value per ORIGINAL row:")
print(demo.groupby("ID")["V"].transform("mean"))
print("")
print("so transform drops straight into a column:")
demo["ID_MEAN"] = demo.groupby("ID")["V"].transform("mean")
print(demo)

print("")
print("--- 2. transform('first') depends on row order ---")
demo_sorted = demo.sort_values(["ID", "TIME"], ignore_index=True)
print("sorted by ID then TIME, so time 0 really is first in each group:")
print(demo_sorted)
demo_sorted["BASE"] = demo_sorted.groupby("ID")["V"].transform("first")
print("")
print(demo_sorted)
print("")
print("check it picked the time 0 value on every row:")
print((demo_sorted.loc[demo_sorted["TIME"] == 0, "V"].to_numpy()
       == demo_sorted.loc[demo_sorted["TIME"] == 0, "BASE"].to_numpy()).all())

print("")
print("--- 3. what happens if you forget to sort ---")
demo_bad = demo.sort_values(["TIME"], ascending=False, ignore_index=True)
demo_bad["BASE"] = demo_bad.groupby("ID")["V"].transform("first")
print("sorted with the LAST timepoint first, so 'first' picks the wrong value:")
print(demo_bad[["ID", "TIME", "V", "BASE"]])
print("no error, no warning, just wrong numbers all the way down")

print("")
print("--- 4. percent change, and keeping post dose rows ---")
demo_sorted["PCT"] = 100.0 * (demo_sorted["V"] - demo_sorted["BASE"]) / demo_sorted["BASE"]
print(demo_sorted[demo_sorted["TIME"] > 0][["ID", "TIME", "V", "BASE", "PCT"]])
'''

PARTS = [
    dict(
        label="a",
        prompt_md='''
## Part a. Sort, then put each subject's baseline on every row

The data arrived sorted by timepoint, which is exactly the order that makes
`transform("first")` give the right answer by luck rather than by design. Sort
it properly first, by subject and then by time, so that time zero really is
the first row of every subject.

Then add a `BASELINE` column holding that subject's time zero value, repeated
on all seven of their rows. This is a `transform`, not an `agg`: you want 105
values back, not 15.

Finally, prove it worked rather than assuming. Pull out the rows where
`TIME_H` is 0 and check that `BIOMARKER` and `BASELINE` agree on every one of
them. If the sort were wrong this check is what would catch it.

Fill in `n_subjects`, how many distinct subjects there are. `df`, the sorted
frame with its index renumbered and a `BASELINE` column added. And
`baseline_ok`, a single True or False.

Moves: `.nunique()`, `.sort_values([...], ignore_index=True)`,
`.groupby(...)["COL"].transform("first")`, and `(a == b).all()`.
''',
        starter='''
# Part a. Sort, then put each subject's baseline on every row.

n_subjects = None    # YOUR CODE: how many distinct subjects
df = None            # YOUR CODE: raw sorted by SUBJECT then TIME_H, index renumbered,
                     #            with a BASELINE column added
baseline_ok = None   # YOUR CODE: True if BIOMARKER equals BASELINE on every TIME_H == 0 row

print("n_subjects :", n_subjects)
print("baseline_ok:", baseline_ok)
if df is not None:
    print(df.head(8))
''',
        solution='''
# Part a. Sort, then put each subject's baseline on every row.

n_subjects = int(raw["SUBJECT"].nunique())

df = raw.sort_values(["SUBJECT", "TIME_H"], ignore_index=True)
df["BASELINE"] = df.groupby("SUBJECT")["BIOMARKER"].transform("first")

_at_zero = df[df["TIME_H"] == 0]
baseline_ok = bool((_at_zero["BIOMARKER"] == _at_zero["BASELINE"]).all())

print("n_subjects :", n_subjects)
print("baseline_ok:", baseline_ok)
if df is not None:
    print(df.head(8))
''',
    ),
    dict(
        label="b",
        prompt_md='''
## Part b. Percent change from baseline

Now that every row knows its own subject's baseline, the change is one line of
arithmetic between two columns.

Percent change from baseline is `100 * (value - baseline) / baseline`. A
subject whose biomarker fell by a fifth gives `-20.0`. Note the sign: a drug
that suppresses the marker produces negative numbers, and reporting the
magnitude without the sign is a good way to confuse a reviewer.

Then drop the baseline rows themselves. At time zero the percent change is
exactly 0 for everybody by construction, so leaving those rows in a summary
adds a timepoint with no variability and drags the group means toward zero.

Fill in `df_post`: the post-dose rows only, so `TIME_H` greater than 0, with a
`PCT_CHANGE` column, index renumbered.

Moves: arithmetic between columns, boolean filtering,
`.reset_index(drop=True)`.
''',
        starter='''
# Part b. Percent change from baseline, post-dose rows only.

df_post = None   # YOUR CODE: TIME_H > 0 rows, with a PCT_CHANGE column, index renumbered

if df_post is not None:
    print(df_post.shape)
    print(df_post.head(6))
''',
        solution='''
# Part b. Percent change from baseline, post-dose rows only.

df_post = df.copy()
df_post["PCT_CHANGE"] = 100.0 * (df_post["BIOMARKER"] - df_post["BASELINE"]) / df_post["BASELINE"]
df_post = df_post[df_post["TIME_H"] > 0].reset_index(drop=True)

if df_post is not None:
    print(df_post.shape)
    print(df_post.head(6))
''',
    ),
    dict(
        label="c",
        prompt_md='''
## Part c. Summarise the change by group and timepoint

Now collapse, because the question has changed. You are no longer asking about
individual rows, you are asking how each dose group behaved over time, so this
is an `agg`.

One row per `GROUP` and `TIME_H`, giving the mean percent change, its sample
SD across subjects, and how many subjects contributed. Keep an eye on that
count: it should be 5 in every cell, and if it is not, something upstream
dropped rows.

SD, never SEM. Pandas `std` already uses `n - 1`.

Fill in `summary`, a flat DataFrame with exactly these columns:

| column | meaning |
|---|---|
| `GROUP` | `High`, `Low` or `Placebo` |
| `TIME_H` | hours after dose |
| `PCT_MEAN` | mean percent change from baseline |
| `PCT_SD` | sample SD of percent change across subjects |
| `N` | how many subjects contributed |

Moves: `.groupby([...]).agg(NAME=("COL", "mean"), ...)`, `.reset_index()`.
''',
        starter='''
# Part c. Mean and SD percent change per group per timepoint.
# Columns wanted: GROUP, TIME_H, PCT_MEAN, PCT_SD, N

summary = None   # YOUR CODE

print(summary)
''',
        solution='''
# Part c. Mean and SD percent change per group per timepoint.
# Columns wanted: GROUP, TIME_H, PCT_MEAN, PCT_SD, N

summary = (df_post
           .groupby(["GROUP", "TIME_H"])
           .agg(PCT_MEAN=("PCT_CHANGE", "mean"),
                PCT_SD=("PCT_CHANGE", "std"),
                N=("PCT_CHANGE", "size"))
           .reset_index())

print(summary)
''',
    ),
]


def build_expected(ns):
    import numpy as np
    raw = ns["raw"]
    df = ns["df"].sort_values(["SUBJECT", "TIME_H"]).reset_index(drop=True)
    post = ns["df_post"].sort_values(["SUBJECT", "TIME_H"]).reset_index(drop=True)
    summary = ns["summary"].sort_values(["GROUP", "TIME_H"]).reset_index(drop=True)

    # the wrong baseline you get from forgetting to sort
    unsorted = raw.copy()
    unsorted["BASELINE"] = unsorted.groupby("SUBJECT")["BIOMARKER"].transform("first")
    unsorted = unsorted.sort_values(["SUBJECT", "TIME_H"]).reset_index(drop=True)

    counts = (ns["df_post"].groupby(["GROUP", "TIME_H"])["PCT_CHANGE"].count()
              .reset_index().sort_values(["GROUP", "TIME_H"])["PCT_CHANGE"])
    return {
        "n_subjects": int(raw["SUBJECT"].nunique()),
        "n_raw_rows": int(len(raw)),
        "n_post_rows": int(len(post)),
        "df_subject": [str(v) for v in df["SUBJECT"]],
        "df_time": [float(v) for v in df["TIME_H"]],
        "df_baseline": [float(v) for v in df["BASELINE"]],
        "df_baseline_unsorted": [float(v) for v in unsorted["BASELINE"]],
        "unsorted_differs": bool(not np.allclose(
            df["BASELINE"].to_numpy(), unsorted["BASELINE"].to_numpy())),
        "df_mean_baseline": [float(v) for v in
                             df.groupby("SUBJECT")["BIOMARKER"].transform("mean")],
        "post_pct": [float(v) for v in post["PCT_CHANGE"]],
        "post_frac": [float(v) / 100.0 for v in post["PCT_CHANGE"]],
        "post_flipped": [-float(v) for v in post["PCT_CHANGE"]],
        "summary_cols": ["GROUP", "TIME_H", "PCT_MEAN", "PCT_SD", "N"],
        "summary_n": int(len(summary)),
        "summary_group": [str(v) for v in summary["GROUP"]],
        "summary_time": [float(v) for v in summary["TIME_H"]],
        "summary_mean": [float(v) for v in summary["PCT_MEAN"]],
        "summary_sd": [float(v) for v in summary["PCT_SD"]],
        "summary_sem": [float(v) for v in summary["PCT_SD"] / np.sqrt(counts.to_numpy())],
        "summary_count": [int(v) for v in summary["N"]],
    }


CHECK_CODE = '''
def _part_a():
    n_subjects = _need("n_subjects")
    df = _need("df")
    baseline_ok = _need("baseline_ok")

    got = _as_scalar(n_subjects)
    if got is None:
        return "`n_subjects` should be a single whole number."
    if int(round(got)) == _EXP["n_raw_rows"]:
        return ("`n_subjects` is the number of rows, not the number of subjects. Each "
                "subject appears once per timepoint.")
    if int(round(got)) != _EXP["n_subjects"]:
        return "`n_subjects` is not the number of distinct subjects in raw."

    if not isinstance(df, _pd.DataFrame):
        return "`df` should be a pandas DataFrame, but it is a %s." % type(df).__name__
    if "BASELINE" not in (_cols_of(df) or []):
        return "`df` has no BASELINE column yet."
    if len(df) != _EXP["n_raw_rows"]:
        return ("`df` has %d rows but it should still have all %d rows of raw. Part a "
                "adds a column, it does not remove rows." % (len(df), _EXP["n_raw_rows"]))
    if list(df.index) != list(range(len(df))):
        return ("`df` has not had its index renumbered. Pass ignore_index=True to "
                "sort_values, or follow with .reset_index(drop=True).")
    if [str(v) for v in df["SUBJECT"]] != _EXP["df_subject"]:
        return ("`df` is not sorted by SUBJECT then TIME_H. Sort before you transform, "
                "because transform('first') reads whatever row happens to be first.")
    if not _arr_close(df["TIME_H"].to_numpy(), _EXP["df_time"]):
        return "`df` is not in SUBJECT then TIME_H order."

    base = df["BASELINE"].to_numpy()
    if _arr_close(base, _EXP["df_mean_baseline"]):
        return ("`BASELINE` holds each subject's MEAN, not their time zero value. The "
                "aggregation you want is 'first', after sorting.")
    if _EXP["unsorted_differs"] and _arr_close(base, _EXP["df_baseline_unsorted"]):
        return ("`BASELINE` was taken with transform('first') on the UNSORTED data, so it "
                "picked whichever row happened to come first. Sort by SUBJECT and TIME_H "
                "before transforming.")
    if len(set(_np.round(base, 9))) == len(base):
        return ("`BASELINE` has a different value on every row, so it was not broadcast "
                "per subject. transform repeats one value across a whole group.")
    if not _arr_close(base, _EXP["df_baseline"]):
        return "`BASELINE` does not hold each subject's own time zero value on every row."

    if not isinstance(baseline_ok, (bool, _np.bool_)):
        val = _as_scalar(baseline_ok)
        if val is None:
            return ("`baseline_ok` should be a single True or False. Comparing two columns "
                    "gives a Series, so finish with .all().")
        baseline_ok = bool(val)
    if not bool(baseline_ok):
        return ("`baseline_ok` came out False. Either the baseline is wrong, or the check "
                "itself is comparing the wrong things.")
    return None


def _part_b():
    df_post = _need("df_post")
    if not isinstance(df_post, _pd.DataFrame):
        return "`df_post` should be a pandas DataFrame, but it is a %s." % type(df_post).__name__
    cols = _cols_of(df_post) or []
    if "PCT_CHANGE" not in cols:
        return "`df_post` has no PCT_CHANGE column yet."
    if "BASELINE" not in cols:
        return "`df_post` lost the BASELINE column. Build it from your df, not from raw."

    if len(df_post) == _EXP["n_raw_rows"]:
        return ("`df_post` still has the baseline rows. Drop TIME_H equal to 0, where the "
                "percent change is 0 for everybody by construction.")
    if len(df_post) != _EXP["n_post_rows"]:
        return ("`df_post` has %d rows but there should be %d post-dose rows."
                % (len(df_post), _EXP["n_post_rows"]))
    if (df_post["TIME_H"] <= 0).any():
        return "`df_post` still contains a TIME_H of 0 or less."
    if list(df_post.index) != list(range(len(df_post))):
        return "`df_post` has not had its index renumbered. Use .reset_index(drop=True)."

    got = df_post.sort_values(["SUBJECT", "TIME_H"])["PCT_CHANGE"].to_numpy()
    if _arr_close(got, _EXP["post_frac"]):
        return "`PCT_CHANGE` is a fraction. The task asks for a percent, so multiply by 100."
    if _arr_close(got, _EXP["post_flipped"]):
        return ("`PCT_CHANGE` has the sign the wrong way round. It is "
                "(value - baseline) / baseline, so a drug that suppresses the marker gives "
                "negative numbers.")
    if not _arr_close(got, _EXP["post_pct"]):
        return ("`PCT_CHANGE` is not 100 * (BIOMARKER - BASELINE) / BASELINE. Check that "
                "you divided by the baseline rather than by the current value.")
    return None


def _part_c():
    summary = _need("summary")
    if not isinstance(summary, _pd.DataFrame):
        return "`summary` should be a pandas DataFrame, but it is a %s." % type(summary).__name__
    cols = _cols_of(summary) or []
    for name in ("GROUP", "TIME_H"):
        if name not in cols and name in [str(n) for n in (summary.index.names or [])]:
            return "`%s` is in the index rather than a column. Call .reset_index()." % name
    missing = [c for c in _EXP["summary_cols"] if c not in cols]
    if missing:
        return "`summary` is missing the column(s) %s." % ", ".join(missing)

    if 0.0 in set(_np.asarray(summary["TIME_H"], dtype=float).tolist()):
        return ("`summary` includes the baseline timepoint, so it was built before the "
                "post-dose filter in part b.")
    if len(summary) != _EXP["summary_n"]:
        return ("`summary` has %d rows but there should be one per group and timepoint "
                "combination." % len(summary))

    got = summary.sort_values(["GROUP", "TIME_H"]).reset_index(drop=True)
    if [str(v) for v in got["GROUP"]] != _EXP["summary_group"]:
        return "`summary` does not cover the expected set of groups."
    if not _arr_close(got["TIME_H"].to_numpy(), _EXP["summary_time"]):
        return "`summary` does not cover the expected set of timepoints."
    if not _arr_close(got["N"].to_numpy(), _EXP["summary_count"]):
        return ("`N` is not the number of subjects in each cell. It should be 5 everywhere; "
                "the aggregation 'size' counts rows within the group.")
    if not _arr_close(got["PCT_MEAN"].to_numpy(), _EXP["summary_mean"]):
        return "`PCT_MEAN` is not the mean of PCT_CHANGE within each group and timepoint."
    if _arr_close(got["PCT_SD"].to_numpy(), _EXP["summary_sem"]):
        return ("`PCT_SD` is the SEM, not the SD. Something divided by the square root of "
                "n. This repo always reports SD.")
    if not _arr_close(got["PCT_SD"].to_numpy(), _EXP["summary_sd"]):
        return "`PCT_SD` is not the sample SD of PCT_CHANGE within each group and timepoint."
    return None
'''

# Functions this problem introduces. Keys must exist in tools/function_catalog.py.
FUNCTIONS = [
    ".transform",
    ".all",
    "ignore_index",
]
