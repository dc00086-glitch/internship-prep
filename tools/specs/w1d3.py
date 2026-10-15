PID = "W1D3"
WEEK = 1
DAY = 3
SLUG = "w1d3_pandas_wrangle_merge"
TITLE = "Rat PK study: filter, summarise by group, and join the dosing record"
TOPIC = "pandas: filtering, derived columns, groupby.agg, left merge"
EST = "20-30 min"

SCENARIO_MD = '''
## Scenario

You have the raw bioanalytical output from a rat single dose PK study. Twelve
animals in three dose groups were sampled at 8 timepoints including a predose
blank, which gives you 96 rows in long format (`pk`).

The study director separately sends the dosing record (`demo`): sex, body
weight on the day of dosing, and the actual milligrams each animal received.
Nominal dosing was 1, 3 and 10 mg/kg, but the actual dose depends on each
animal's weight, so the real mg/kg has to be recomputed from the record.

One thing the study director has not mentioned: the dosing record is not
complete. Finding that out yourself, rather than after you have fitted a
model, is most of the point of this problem. A join that silently drops rows is
one of the most common ways a PK analysis goes quietly wrong.
'''

TASK_MD = '''
## Your task

### Part a. Get your bearings and drop the predose samples

Before anything else, find out how big the dataset really is. Counting rows is
not counting animals: each animal appears once per timepoint, so the row count
is a multiple of the animal count.

Then drop the predose samples. They are all zero by definition and they would
drag every mean down and inflate every SD if you left them in a summary.

Fill in `n_animals`, `df_post`.
Moves: `df["COL"].nunique()`, and boolean filtering `df[df["COL"] > 0]`.
- `n_animals`: a whole number, how many distinct animals appear in `pk`.
- `df_post`: the post-dose rows only, meaning `TIME_H` strictly greater than 0.
  Keep all four columns.

### Part b. Mean and SD profile per dose group

This is split, apply, combine: pandas splits the table into one chunk per
group and timepoint, computes your statistics on each chunk, and stacks the
answers back up.

Two details decide whether the result is usable downstream. The grouping keys
come back sitting in the **index**, not as columns, so `.reset_index()` is
needed to flatten them. And the aggregation names come out as `mean` and
`std`, so rename them to the names the task asks for.

Pandas `std` already divides by `n - 1`, so you do not need `ddof` here. That
is the opposite of numpy.

Build `summary` from `df_post`, with columns `GROUP`, `TIME_H`, `CONC_MEAN`,
`CONC_SD`.
Moves: `.groupby([...])["COL"].agg(["mean", "std"])`, `.reset_index()`,
`.rename(columns={...})`.
From `df_post`, build `summary`: one row per `GROUP` and `TIME_H` combination,
with exactly these four columns, named exactly like this:

| column | meaning |
|---|---|
| `GROUP` | dose group |
| `TIME_H` | nominal time in hours |
| `CONC_MEAN` | mean concentration in that group at that time |
| `CONC_SD` | **sample** SD of concentration in that group at that time |

`summary` must be a flat DataFrame, so `GROUP` and `TIME_H` are ordinary
columns and not an index. SD, never SEM.

### Part c. Join the dosing record and recover the real mg/kg
Starting from the full `pk` table, not `df_post`:

- `pk_merged`: `pk` joined to `demo` on `ANIMAL` in a way that **keeps every
  row of `pk`**, even for an animal that is missing from the dosing record.
  Add one new column `DOSE_MG_PER_KG`, which is `DOSE_MG` divided by body
  weight **in kilograms**. The record stores grams.
- `n_missing_weight`: a whole number, how many rows of `pk_merged` end up with
  no body weight.

### Hints
- `df["COL"].nunique()` counts distinct values.
- Boolean filtering: `df[df["TIME_H"] > 0]`.
- `df.groupby([...])["COL"].agg(["mean", "std"])` gives you both statistics at
  once. Pandas `std` already uses `n - 1`. Then `.reset_index()` flattens it
  and `.rename(columns={...})` fixes the names.
- `pd.merge(left, right, on="ANIMAL", how="left")` keeps every left row.
  Compare `len()` before and after, every time.
- `df["COL"].isna().sum()` counts the gaps.
'''

DATA_CODE = '''
import numpy as np
import pandas as pd

PROBLEM_ID = "W1D3"
TOPIC = "pandas: filtering, derived columns, groupby.agg, left merge"

rng = np.random.default_rng(1103)

N_ANIMALS = 12
times = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0])
animals = np.array(["R%02d" % i for i in range(1, N_ANIMALS + 1)])
groups = np.repeat(["Low", "Mid", "High"], 4)
nominal_mg_kg = np.repeat([1.0, 3.0, 10.0], 4)
sex = np.tile(["M", "M", "F", "F"], 3)

bw_g = np.round(250.0 + rng.normal(0.0, 18.0, size=N_ANIMALS), 0)
# Syringe accuracy means the delivered dose is close to, but not exactly, nominal.
dose_mg = np.round(nominal_mg_kg * bw_g / 1000.0 * rng.lognormal(0.0, 0.02, size=N_ANIMALS), 4)

# Between animal variability, one compartment oral, concentrations in ng/mL.
_V = 0.20 * rng.lognormal(0.0, 0.22, size=N_ANIMALS)
_kel = 0.30 * rng.lognormal(0.0, 0.25, size=N_ANIMALS)
_ka = 1.50 * rng.lognormal(0.0, 0.15, size=N_ANIMALS)
_coef = (0.90 * dose_mg * _ka) / (_V * (_ka - _kel))
_true = 1000.0 * _coef[:, None] * (np.exp(-_kel[:, None] * times[None, :])
                                  - np.exp(-_ka[:, None] * times[None, :]))
_obs = np.round(_true * rng.lognormal(0.0, 0.12, size=_true.shape), 1)

pk = pd.DataFrame({
    "ANIMAL": np.repeat(animals, times.size),
    "GROUP": np.repeat(groups, times.size),
    "TIME_H": np.tile(times, N_ANIMALS),
    "CONC_NG_ML": _obs.ravel(),
})

# The dosing record the study director sent over.
_in_record = animals != "R07"
demo = pd.DataFrame({
    "ANIMAL": animals[_in_record],
    "SEX": sex[_in_record],
    "BODYWEIGHT_G": bw_g[_in_record],
    "DOSE_MG": dose_mg[_in_record],
})

print("pk:", pk.shape)
print(pk.head(10))
print("")
print("demo:", demo.shape)
print(demo.head())
'''

PARTS = [
    dict(
        label="a",
        prompt_md='''
### Part a. Get your bearings and drop the predose samples
''',
        starter='''
# Part a. Count the animals and keep only post-dose rows.

n_animals = None   # YOUR CODE: how many distinct animals are in pk
df_post = None     # YOUR CODE: the rows of pk with TIME_H greater than 0

print("n_animals:", n_animals)
print("df_post shape:", None if df_post is None else df_post.shape)
''',
        solution='''
# Part a. Count the animals and keep only post-dose rows.

n_animals = int(pk["ANIMAL"].nunique())
df_post = pk[pk["TIME_H"] > 0]

print("n_animals:", n_animals)
print("df_post shape:", None if df_post is None else df_post.shape)
''',
    ),
    dict(
        label="b",
        prompt_md='''
### Part b. Mean and SD profile per dose group
''',
        starter='''
# Part b. Mean and sample SD per group per timepoint, from df_post.
# Columns must come out named GROUP, TIME_H, CONC_MEAN, CONC_SD.

summary = None   # YOUR CODE

print(summary)
''',
        solution='''
# Part b. Mean and sample SD per group per timepoint, from df_post.
# Columns must come out named GROUP, TIME_H, CONC_MEAN, CONC_SD.

summary = (df_post
           .groupby(["GROUP", "TIME_H"])["CONC_NG_ML"]
           .agg(["mean", "std"])
           .reset_index()
           .rename(columns={"mean": "CONC_MEAN", "std": "CONC_SD"}))

print(summary)
''',
    ),
    dict(
        label="c",
        prompt_md='''
### Part c. Join the dosing record and recover the real mg/kg

Start from `pk`, the full table, not `df_post`.

The dosing record is incomplete, and how you join decides whether you find
that out. The default join is `inner`, which keeps only matched rows and
discards the rest with no error and no warning. Use a **left** join so every
row of `pk` survives and the gaps show up as `NaN` where you can see them.

Then check: print `len(pk)` and `len(pk_merged)` and make sure they agree. Get
into this habit now. A merge that silently drops an animal is one of the most
common ways a PK analysis produces a confident wrong answer.

Body weight is recorded in **grams** and the dose in milligrams, so convert
before dividing. Sanity check the result against the nominal 1, 3 and 10
mg/kg.

Fill in `pk_merged` (with a `DOSE_MG_PER_KG` column), `n_missing_weight`.
Moves: `pd.merge(left, right, on=..., how="left")`, arithmetic between
columns, `df["COL"].isna().sum()`.
''',
        starter='''
# Part c. Join the dosing record onto pk without losing any rows.

pk_merged = None         # YOUR CODE: pk joined to demo, plus a DOSE_MG_PER_KG column
n_missing_weight = None  # YOUR CODE: rows of pk_merged with no body weight

print("rows in pk       :", len(pk))
print("rows in pk_merged:", None if pk_merged is None else len(pk_merged))
print("n_missing_weight :", n_missing_weight)
if pk_merged is not None:
    print(pk_merged.head())
''',
        solution='''
# Part c. Join the dosing record onto pk without losing any rows.

pk_merged = pd.merge(pk, demo, on="ANIMAL", how="left")
pk_merged["DOSE_MG_PER_KG"] = pk_merged["DOSE_MG"] / (pk_merged["BODYWEIGHT_G"] / 1000.0)
n_missing_weight = int(pk_merged["BODYWEIGHT_G"].isna().sum())

print("rows in pk       :", len(pk))
print("rows in pk_merged:", None if pk_merged is None else len(pk_merged))
print("n_missing_weight :", n_missing_weight)
if pk_merged is not None:
    print(pk_merged.head())
''',
    ),
]


def build_expected(ns):
    import numpy as np
    import pandas as pd
    pk, demo = ns["pk"], ns["demo"]
    post = pk[pk["TIME_H"] > 0]

    summary = ns["summary"].sort_values(["GROUP", "TIME_H"]).reset_index(drop=True)
    sd_pop = (post.groupby(["GROUP", "TIME_H"])["CONC_NG_ML"].std(ddof=0)
              .reset_index().sort_values(["GROUP", "TIME_H"])["CONC_NG_ML"])
    counts = (post.groupby(["GROUP", "TIME_H"])["CONC_NG_ML"].count()
              .reset_index().sort_values(["GROUP", "TIME_H"])["CONC_NG_ML"])
    sem = summary["CONC_SD"].to_numpy() / np.sqrt(counts.to_numpy())

    merged = ns["pk_merged"].copy()
    return {
        "n_animals": int(pk["ANIMAL"].nunique()),
        "n_pk_rows": int(len(pk)),
        "n_post_rows": int(len(post)),
        "post_cols": [str(c) for c in pk.columns],
        "summary_cols": ["GROUP", "TIME_H", "CONC_MEAN", "CONC_SD"],
        "summary_n": int(len(summary)),
        "summary_group": [str(v) for v in summary["GROUP"]],
        "summary_time": [float(v) for v in summary["TIME_H"]],
        "summary_mean": [float(v) for v in summary["CONC_MEAN"]],
        "summary_sd": [float(v) for v in summary["CONC_SD"]],
        "summary_sd_pop": [float(v) for v in sd_pop],
        "summary_sem": [float(v) for v in sem],
        "n_inner_rows": int(len(pd.merge(pk, demo, on="ANIMAL", how="inner"))),
        "n_missing_weight": int(merged["BODYWEIGHT_G"].isna().sum()),
        "merged_cols": [str(c) for c in merged.columns],
        "dose_per_kg": [float(v) for v in merged["DOSE_MG_PER_KG"].fillna(-1.0)],
        "dose_per_g": [float(v) for v in (merged["DOSE_MG"] / merged["BODYWEIGHT_G"]).fillna(-1.0)],
    }


CHECK_CODE = '''
def _part_a():
    n_animals = _need("n_animals")
    df_post = _need("df_post")

    got = _as_scalar(n_animals)
    if got is None:
        return "`n_animals` should be a single whole number."
    if int(round(got)) == _EXP["n_pk_rows"]:
        return ("`n_animals` is the number of rows in pk, not the number of animals. "
                "Each animal appears once per timepoint, so count distinct values.")
    if int(round(got)) != _EXP["n_animals"]:
        return "`n_animals` is not the number of distinct animals in pk. Try nunique on the ANIMAL column."

    if not isinstance(df_post, _pd.DataFrame):
        return "`df_post` should be a pandas DataFrame, but it is a %s." % type(df_post).__name__
    cols = _cols_of(df_post)
    missing = [c for c in _EXP["post_cols"] if c not in cols]
    if missing:
        return "`df_post` is missing the column(s) %s. Keep all four columns of pk." % ", ".join(missing)
    if len(df_post) == _EXP["n_pk_rows"]:
        return ("`df_post` still has every row of pk, so the predose samples were not "
                "removed. Filter on TIME_H greater than 0.")
    if len(df_post) != _EXP["n_post_rows"]:
        return ("`df_post` has %d rows, which is not the number of post-dose rows. Use "
                "strictly greater than 0, which drops exactly one sample per animal."
                % len(df_post))
    if (df_post["TIME_H"] <= 0).any():
        return "`df_post` still contains a TIME_H of 0 or less."
    return None


def _part_b():
    summary = _need("summary")

    if isinstance(summary, _pd.Series):
        return ("`summary` is a Series, not a DataFrame. Aggregating with a list of "
                "statistics, as in .agg(['mean', 'std']), gives you a frame.")
    if not isinstance(summary, _pd.DataFrame):
        return "`summary` should be a pandas DataFrame, but it is a %s." % type(summary).__name__

    cols = _cols_of(summary)
    want = _EXP["summary_cols"]
    for name in ("GROUP", "TIME_H"):
        if name not in cols and name in [str(n) for n in (summary.index.names or [])]:
            return ("`%s` is sitting in the index rather than being a column. Call "
                    ".reset_index() after the groupby to flatten it." % name)
    missing = [c for c in want if c not in cols]
    if missing:
        extra = [c for c in cols if c not in want]
        return ("`summary` is missing the column(s) %s%s. The four columns have to be "
                "named exactly GROUP, TIME_H, CONC_MEAN, CONC_SD."
                % (", ".join(missing), (" and has unexpected column(s) " + ", ".join(extra)) if extra else ""))

    if 0.0 in set(_np.asarray(summary["TIME_H"], dtype=float).tolist()):
        return ("`summary` includes the predose timepoint, so it was built from pk rather "
                "than from your post-dose table from part a.")
    if len(summary) != _EXP["summary_n"]:
        return ("`summary` has %d rows but there should be one row per group and timepoint "
                "combination. Check that you grouped by both keys." % len(summary))

    got = summary.sort_values(["GROUP", "TIME_H"]).reset_index(drop=True)
    if [str(v) for v in got["GROUP"]] != _EXP["summary_group"]:
        return "`summary` does not cover the expected set of dose groups."
    if not _arr_close(got["TIME_H"].to_numpy(), _EXP["summary_time"]):
        return "`summary` does not cover the expected set of timepoints."
    if not _arr_close(got["CONC_MEAN"].to_numpy(), _EXP["summary_mean"]):
        return ("`CONC_MEAN` does not match a plain mean of CONC_NG_ML within each group and "
                "timepoint. Check that you aggregated the concentration column and nothing else.")
    if _arr_close(got["CONC_SD"].to_numpy(), _EXP["summary_sem"]):
        return ("`CONC_SD` is the SEM, not the SD. Something divided by the square root of n. "
                "This repo always reports SD.")
    if _arr_close(got["CONC_SD"].to_numpy(), _EXP["summary_sd_pop"]):
        return ("`CONC_SD` used n in the denominator rather than n - 1. That happens if you "
                "hand numpy's std to .agg; the pandas std already uses n - 1.")
    if not _arr_close(got["CONC_SD"].to_numpy(), _EXP["summary_sd"]):
        return "`CONC_SD` does not match the sample SD within each group and timepoint."
    return None


def _part_c():
    pk_merged = _need("pk_merged")
    n_missing_weight = _need("n_missing_weight")

    if not isinstance(pk_merged, _pd.DataFrame):
        return "`pk_merged` should be a pandas DataFrame, but it is a %s." % type(pk_merged).__name__

    cols = _cols_of(pk_merged)
    for name in ("BODYWEIGHT_G", "DOSE_MG", "SEX"):
        if name not in cols:
            return ("`pk_merged` has no %s column, so the join did not bring the dosing "
                    "record across. Merge on ANIMAL." % name)
    if "DOSE_MG_PER_KG" not in cols:
        return "`pk_merged` has no DOSE_MG_PER_KG column yet. Add it after the merge."

    if len(pk_merged) == _EXP["n_inner_rows"] and _EXP["n_inner_rows"] != _EXP["n_pk_rows"]:
        return ("`pk_merged` has %d rows but pk has %d, so the join quietly dropped the "
                "animal that is missing from the dosing record. That is what an inner join "
                "does. Use how='left' to keep every row of pk."
                % (len(pk_merged), _EXP["n_pk_rows"]))
    if len(pk_merged) > _EXP["n_pk_rows"]:
        return ("`pk_merged` has %d rows but pk has only %d, so the join multiplied rows. "
                "That happens when the right table has more than one row per ANIMAL."
                % (len(pk_merged), _EXP["n_pk_rows"]))
    if len(pk_merged) != _EXP["n_pk_rows"]:
        return "`pk_merged` has %d rows but it should have the same %d rows as pk." % (len(pk_merged), _EXP["n_pk_rows"])

    got = _np.asarray(pk_merged["DOSE_MG_PER_KG"].fillna(-1.0).to_numpy(), dtype=float)
    if _arr_close(got, _EXP["dose_per_g"]):
        return ("`DOSE_MG_PER_KG` is milligrams per gram, because the record stores body "
                "weight in grams. Convert to kilograms before dividing.")
    if not _arr_close(got, _EXP["dose_per_kg"]):
        return ("`DOSE_MG_PER_KG` does not match DOSE_MG divided by body weight in kilograms. "
                "Sanity check it: the values should land near the nominal 1, 3 and 10 mg/kg.")

    val = _as_scalar(n_missing_weight)
    if val is None:
        return "`n_missing_weight` should be a single whole number."
    if int(round(val)) == 1:
        return ("`n_missing_weight` counts 1, which is the number of animals missing from the "
                "record. The question asks how many rows of pk_merged have no body weight, "
                "and each animal contributes one row per timepoint.")
    if int(round(val)) != _EXP["n_missing_weight"]:
        return ("`n_missing_weight` is not the number of rows with no body weight. Count where "
                "BODYWEIGHT_G is na in pk_merged.")
    return None
'''


TOOLKIT_MD = '''
## Toolkit for this problem

A DataFrame is a table. Each column has a name, and you pull one out with
`df["NAME"]`. Everything below runs on a tiny made up table, so none of it is
the answer.

| What you want to do | How |
|---|---|
| look at the first rows | `df.head()` |
| its size | `df.shape`, which reads `(rows, columns)` |
| the column names | `df.columns` |
| one column | `df["CONC"]` |
| how many distinct values | `df["ANIMAL"].nunique()` |
| keep some rows | `df[df["TIME_H"] > 0]` |
| add a column | `df["NEW"] = df["A"] / df["B"]` |
| mean and SD per group | `df.groupby(["G", "T"])["CONC"].agg(["mean", "std"])` |
| turn group keys back into columns | `.reset_index()` |
| rename columns | `.rename(columns={"mean": "CONC_MEAN"})` |
| join two tables | `pd.merge(left, right, on="ANIMAL", how="left")` |
| count the gaps | `df["COL"].isna().sum()` |

Two things worth knowing before you start.

- Pandas `std` already divides by `n - 1`, so it is the sample SD you want.
  This is the opposite of numpy, where you have to ask with `ddof=1`.
- `how="left"` keeps every row of the left table. `how="inner"` keeps only
  rows that matched, and quietly drops the rest. Always compare `len()`
  before and after a merge.
'''

TOOLKIT_CODE = '''
import numpy as np
import pandas as pd

# A tiny made up table. Nothing here is the answer to the problem.
demo_tbl = pd.DataFrame({
    "SUBJ": ["S1", "S1", "S2", "S2", "S3", "S3"],
    "ARM":  ["A", "A", "A", "A", "B", "B"],
    "TIME": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    "VAL":  [0.0, 10.0, 0.0, 14.0, 0.0, 30.0],
})
print("demo_tbl:")
print(demo_tbl)
print("shape:", demo_tbl.shape, "   columns:", list(demo_tbl.columns))

print("")
print("--- 1. one column, and counting distinct values ---")
print("demo_tbl['VAL']      :", list(demo_tbl["VAL"]))
print("distinct subjects:", demo_tbl["SUBJ"].nunique())

print("")
print("--- 2. keeping some rows ---")
demo_post = demo_tbl[demo_tbl["TIME"] > 0]
print(demo_post)
print("rows before:", len(demo_tbl), " rows after:", len(demo_post))

print("")
print("--- 3. mean and SD per group, then flattened ---")
demo_summary = (demo_post
                .groupby(["ARM", "TIME"])["VAL"]
                .agg(["mean", "std"])
                .reset_index()
                .rename(columns={"mean": "VAL_MEAN", "std": "VAL_SD"}))
print(demo_summary)
print("note: ARM and TIME are ordinary columns now, thanks to reset_index()")
print("note: pandas std already uses n-1, so VAL_SD is the sample SD")

print("")
print("--- 4. joining a second table, and what a missing row does ---")
demo_info = pd.DataFrame({"SUBJ": ["S1", "S2"], "WEIGHT_G": [300.0, 280.0]})
print("demo_info has no row for S3:")
print(demo_info)

demo_left = pd.merge(demo_tbl, demo_info, on="SUBJ", how="left")
demo_inner = pd.merge(demo_tbl, demo_info, on="SUBJ", how="inner")
print("")
print("rows in demo_tbl        :", len(demo_tbl))
print("rows after how=left :", len(demo_left), " <- every row kept, S3 gets NaN")
print("rows after how=inner:", len(demo_inner), " <- S3 silently dropped")
print("")
print(demo_left)
print("")
print("rows with no weight :", int(demo_left["WEIGHT_G"].isna().sum()))

print("")
print("--- 5. adding a column built from other columns ---")
demo_left["VAL_PER_KG"] = demo_left["VAL"] / (demo_left["WEIGHT_G"] / 1000.0)
print(demo_left[["SUBJ", "VAL", "WEIGHT_G", "VAL_PER_KG"]])
'''


CONCEPT_MD = '''
## How pandas works, in two minutes

**A DataFrame is a dictionary of columns, not a grid of cells.** That sounds
pedantic but it explains most of the behaviour. Each column is a **Series**: a
numpy array plus an **index**, which is a set of labels for the rows. So
`df["CONC"]` hands you one Series, and arithmetic on it is elementwise, just
like numpy. Each column has its own type, which is why one table can hold
animal names, group labels and concentrations at the same time, something a
numpy array cannot do.

**The index is the part people trip over.** Every DataFrame has row labels,
and by default they are 0, 1, 2 and so on. They look like positions but they
are labels, and they stick to their rows. Filter out some rows and the
survivors keep their original numbers, so the index now has gaps. That is
usually harmless, but it matters after a `groupby`, where the grouping keys
get *moved into the index*. The columns you grouped by then stop being
columns, and anything downstream looking for them fails. `.reset_index()`
pushes them back out into ordinary columns, which is why you will see it after
almost every groupby.

**Split, apply, combine.** That is the whole model behind `groupby`. Pandas
splits the table into one chunk per group, applies your summary to each chunk,
then stacks the answers back into a new table. `df.groupby(["GROUP",
"TIME_H"])["CONC"].agg(["mean", "std"])` reads exactly that way: split by
group and time, look at the concentration column, compute mean and SD of each
chunk. You get one row per combination that actually occurs.

**Joins, and the one that bites.** `pd.merge` matches rows between two tables
on a shared column. The `how` argument decides what happens to rows that find
no partner:

- `how="left"` keeps every row of the left table, filling the missing columns
  with `NaN`. You can see what is missing.
- `how="inner"`, which is the **default**, keeps only rows that matched and
  silently discards the rest. No error, no warning, just a smaller table.

That default is why a PK analysis can lose an entire animal and still run to
completion with a plausible-looking answer. The habit that protects you is
boring and absolute: print `len()` before and after every merge, and if the
number changed, know why. Rows can also go *up*, if the right-hand table has
more than one row per key, which quietly duplicates your data.

**One more pandas quirk.** Pandas `std()` already divides by `n - 1`, unlike
numpy, which needs `ddof=1`. Two libraries, two defaults, and nothing warns
you.
'''


# Functions this problem introduces. Keys must exist in tools/function_catalog.py.
FUNCTIONS = [
    'pd.DataFrame',
    '.head',
    '.columns',
    'selecting a column',
    'boolean filtering',
    'adding a column',
    '.nunique / .unique',
    '.isna',
    '.groupby',
    '.agg',
    '.reset_index',
    '.rename',
    'pd.merge',
    'np.repeat / np.tile',
]
