# Week 2, pandas in depth

The full text of every problem this week, the data it generates, the
toolkit the student is given, what each part asks for, and what every
possible check failure means. No solutions in this file.

---

# W2D1. Baseline correction: transform, and why it is not agg

- topic: pandas: groupby.transform, baseline correction, alignment
- notebook: problems/week02/w2d1_pandas_transform_baseline.ipynb
- solution: solutions/week02/w2d1_pandas_transform_baseline_solution.ipynb
- expected time: 20-30 min

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

## The setup cell the student runs

This is given to them, already working. The data is synthetic and
seeded, so their numbers are always these numbers.

```python
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
```

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

The toolkit cell demonstrates each of those on a tiny made up example.
It is in the notebook and none of it is the answer.

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

**Variables this part is graded on:** `n_subjects`, `df`, `baseline_ok`

The starter cell they see:

```python
# Part a. Sort, then put each subject's baseline on every row.

n_subjects = None    # YOUR CODE: how many distinct subjects
df = None            # YOUR CODE: raw sorted by SUBJECT then TIME_H, index renumbered,
                     #            with a BASELINE column added
baseline_ok = None   # YOUR CODE: True if BIOMARKER equals BASELINE on every TIME_H == 0 row

print("n_subjects :", n_subjects)
print("baseline_ok:", baseline_ok)
if df is not None:
    print(df.head(8))
```

**Every failure message part a can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `n_subjects` should be a single whole number.
- `n_subjects` is the number of rows, not the number of subjects. Each subject appears once per timepoint.
- `n_subjects` is not the number of distinct subjects in raw.
- `df` should be a pandas DataFrame, but it is a %s.
- `df` has no BASELINE column yet.
- `df` has %d rows but it should still have all %d rows of raw. Part a adds a column, it does not remove rows.
- `df` has not had its index renumbered. Pass ignore_index=True to sort_values, or follow with .reset_index(drop=True).
- `df` is not sorted by SUBJECT then TIME_H. Sort before you transform, because transform('first') reads whatever row happens to be first.
- `df` is not in SUBJECT then TIME_H order.
- `BASELINE` holds each subject's MEAN, not their time zero value. The aggregation you want is 'first', after sorting.
- `BASELINE` was taken with transform('first') on the UNSORTED data, so it picked whichever row happened to come first. Sort by SUBJECT and TIME_H before transforming.
- `BASELINE` has a different value on every row, so it was not broadcast per subject. transform repeats one value across a whole group.
- `BASELINE` does not hold each subject's own time zero value on every row.
- `baseline_ok` came out False. Either the baseline is wrong, or the check itself is comparing the wrong things.
- `baseline_ok` should be a single True or False. Comparing two columns gives a Series, so finish with .all().

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

**Variables this part is graded on:** `df_post`

The starter cell they see:

```python
# Part b. Percent change from baseline, post-dose rows only.

df_post = None   # YOUR CODE: TIME_H > 0 rows, with a PCT_CHANGE column, index renumbered

if df_post is not None:
    print(df_post.shape)
    print(df_post.head(6))
```

**Every failure message part b can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `df_post` should be a pandas DataFrame, but it is a %s.
- `df_post` has no PCT_CHANGE column yet.
- `df_post` lost the BASELINE column. Build it from your df, not from raw.
- `df_post` still has the baseline rows. Drop TIME_H equal to 0, where the percent change is 0 for everybody by construction.
- `df_post` has %d rows but there should be %d post-dose rows.
- `df_post` still contains a TIME_H of 0 or less.
- `df_post` has not had its index renumbered. Use .reset_index(drop=True).
- `PCT_CHANGE` is a fraction. The task asks for a percent, so multiply by 100.
- `PCT_CHANGE` has the sign the wrong way round. It is (value - baseline) / baseline, so a drug that suppresses the marker gives negative numbers.
- `PCT_CHANGE` is not 100 * (BIOMARKER - BASELINE) / BASELINE. Check that you divided by the baseline rather than by the current value.

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

**Variables this part is graded on:** `summary`

The starter cell they see:

```python
# Part c. Mean and SD percent change per group per timepoint.
# Columns wanted: GROUP, TIME_H, PCT_MEAN, PCT_SD, N

summary = None   # YOUR CODE

print(summary)
```

**Every failure message part c can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `summary` should be a pandas DataFrame, but it is a %s.
- `summary` is missing the column(s) %s. ,
- `summary` includes the baseline timepoint, so it was built before the post-dose filter in part b.
- `summary` has %d rows but there should be one per group and timepoint combination.
- `summary` does not cover the expected set of groups.
- `summary` does not cover the expected set of timepoints.
- `N` is not the number of subjects in each cell. It should be 5 everywhere; the aggregation 'size' counts rows within the group.
- `PCT_MEAN` is not the mean of PCT_CHANGE within each group and timepoint.
- `PCT_SD` is the SEM, not the SD. Something divided by the square root of n. This repo always reports SD.
- `PCT_SD` is not the sample SD of PCT_CHANGE within each group and timepoint.
- `%s` is in the index rather than a column. Call .reset_index().

**Functions this problem introduces:** .transform, .all / .any, ignore_index

---

# W2D2. Joins that lie: validate the merge before you trust it

- topic: pandas: join types, validate, indicator, duplicate keys
- notebook: problems/week02/w2d2_pandas_merge_validate.ipynb
- solution: solutions/week02/w2d2_pandas_merge_validate_solution.ipynb
- expected time: 20-30 min

## Scenario

You are pulling a dataset together from three sources, which is the normal
state of affairs. The bioanalytical lab sent concentrations, the clinical
database sent the dosing record, and the site sent a subject listing.

Two things are wrong with it, and neither announces itself:

- one subject's dosing record was entered twice, because a query was raised
  and the correction was added rather than replacing the original,
- one subject has concentrations but no dosing record at all.

Merge carelessly and you get a dataset that is both **too big** and **too
small** at the same time: the duplicated subject's rows silently double, and
the undosed subject's rows silently vanish. The row count might even come out
looking reasonable, because the two errors partly cancel.

This problem is about never letting that happen to you again.

## How a merge can go wrong without telling you

A merge matches rows between two tables on a key. What it does with rows that
match zero times, or more than once, is entirely up to the arguments you pass,
and the defaults are not the safe choice.

**Row counts can go down.** `how="inner"`, the default, keeps only rows that
found a partner. A subject missing from the right-hand table loses every one
of their rows. No error, no warning, just a smaller table.

**Row counts can go up.** This one surprises people more. If the right-hand
table has two rows for the same key, every matching left row is duplicated to
pair with each of them. A subject with a duplicated dosing record comes out
with twice as many concentration rows as they actually have. Every later
summary then counts that subject twice, their variance contribution doubles,
and an `agg` that looks perfectly healthy is quietly wrong.

That is the dangerous one, because a dataset that got **bigger** does not feel
like data loss, so nobody checks.

**The three tools that make this safe.**

`validate=` states the relationship you believe holds, and pandas raises if it
does not. `validate="m:1"` means many left rows may match, but the right-hand
key must be unique. That single argument turns a silent duplication into a
loud `MergeError`, which is exactly what you want. The other values are
`"1:1"`, `"1:m"` and `"m:m"`.

`indicator=True` adds a `_merge` column saying whether each row came from
`left_only`, `right_only` or `both`. Counting it tells you immediately how
many rows found no partner, and which ones.

`len()` before and after, every single time. Even with the other two, this is
the check that costs nothing and catches everything.

**On fixing the duplicate.** `drop_duplicates()` removes rows that are
identical across all columns. That is the right fix here, because the record
really was entered twice. If the two rows had **disagreed**, dropping one
would be choosing an answer, and the honest move is to go back and ask which
is correct. A cleaning step that silently picks a winner is how a transcription
error becomes permanent.

## Your task

Three parts, worked one at a time.

| Part | What you work out | Variables to fill in |
|---|---|---|
| a | what is actually wrong with the dosing record | `n_pk_rows`, `n_dup_rows`, `dup_subjects` |
| b | a clean, validated merge that keeps every row | `dosing_clean`, `merged` |
| c | who is missing, found with an indicator column | `checked`, `n_left_only`, `unmatched_subjects` |

Below this you will find the **Setup** cell, then a **Toolkit** cell that
demonstrates every move on a tiny made up example, then **Part a**, **Part b**
and **Part c**, each with its own explanation and a cell to write in.

## The setup cell the student runs

This is given to them, already working. The data is synthetic and
seeded, so their numbers are always these numbers.

```python
import numpy as np
import pandas as pd

PROBLEM_ID = "W2D2"
TOPIC = "pandas: join types, validate, indicator, duplicate keys"

rng = np.random.default_rng(2102)

times = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 24.0])
subjects = ["P%02d" % i for i in range(1, 11)]
_dose_level = np.repeat([25.0, 100.0], 5)

_V = 42.0 * rng.lognormal(0.0, 0.22, size=len(subjects))
_kel = 0.11 * rng.lognormal(0.0, 0.25, size=len(subjects))
_ka = 1.10 * rng.lognormal(0.0, 0.18, size=len(subjects))
_coef = (0.88 * _dose_level * _ka) / (_V * (_ka - _kel))
_true = _coef[:, None] * (np.exp(-_kel[:, None] * times[None, :])
                          - np.exp(-_ka[:, None] * times[None, :]))

pk = pd.DataFrame({
    "SUBJECT": np.repeat(subjects, times.size),
    "TIME_H": np.tile(times, len(subjects)),
    "CONC_MG_L": np.round(_true.ravel() * rng.lognormal(0.0, 0.11, size=_true.size), 3),
})

# Dosing record. P04 was entered twice after a data query, and P09 is absent.
dosing = pd.DataFrame({
    "SUBJECT": subjects,
    "DOSE_MG": _dose_level,
    "COHORT": np.repeat(["A", "B"], 5),
})
dosing = pd.concat([dosing, dosing[dosing["SUBJECT"] == "P04"]], ignore_index=True)
dosing = dosing[dosing["SUBJECT"] != "P09"].reset_index(drop=True)

print("pk:", pk.shape)
print(pk.head())
print("")
print("dosing:", dosing.shape)
print(dosing)
```

## Toolkit for this problem

| What you want to do | How |
|---|---|
| flag duplicate keys | `df.duplicated(subset=["SUBJECT"], keep=False)` |
| count fully repeated rows | `df.duplicated().sum()` |
| the distinct values of a column | `sorted(df["SUBJECT"].unique())` |
| remove exact repeats | `df.drop_duplicates()` |
| a merge that refuses to duplicate | `pd.merge(a, b, on="K", how="left", validate="m:1")` |
| label where each row came from | `pd.merge(..., indicator=True)` adds `_merge` |
| count the labels | `df["_merge"].value_counts()` |
| rows that found no partner | `df[df["_merge"] == "left_only"]` |

`keep=False` on `duplicated` flags **every** copy, which is what you want when
you are trying to see the problem. Plain `duplicated()` flags only the repeats
after the first, which is what you want when you are counting how many to
remove.

The toolkit cell demonstrates each of those on a tiny made up example.
It is in the notebook and none of it is the answer.

## Part a. Find out what is wrong before you merge anything

Never merge a table you have not inspected. Thirty seconds here saves an
afternoon later.

Count the rows of `pk` so you have a number to compare against afterwards.
Then look at `dosing` and find the repeated record: how many rows are exact
repeats of a row already present, and which subject or subjects they belong
to.

For the count use plain `duplicated()`, which flags repeats after the first,
so it answers "how many rows would I remove". For working out **who**, use
`keep=False`, which flags every copy including the first, so you see the whole
problem rather than half of it.

Fill in `n_pk_rows`, a whole number. `n_dup_rows`, how many rows of `dosing`
are exact repeats. And `dup_subjects`, a sorted list of the subject
identifiers involved.

Moves: `len()`, `.duplicated().sum()`, `.duplicated(subset=[...], keep=False)`,
`.unique()`, `sorted()`.

**Variables this part is graded on:** `n_pk_rows`, `n_dup_rows`, `dup_subjects`

The starter cell they see:

```python
# Part a. Inspect before merging.

n_pk_rows = None      # YOUR CODE: how many rows pk has
n_dup_rows = None     # YOUR CODE: how many rows of dosing are exact repeats
dup_subjects = None   # YOUR CODE: sorted list of the subjects involved

print("n_pk_rows   :", n_pk_rows)
print("n_dup_rows  :", n_dup_rows)
print("dup_subjects:", dup_subjects)
```

**Every failure message part a can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `n_pk_rows` is not the number of rows in pk. len(pk) gives it.
- `n_dup_rows` should be a single whole number.
- `n_dup_rows` counted every copy, because you passed keep=False. For how many rows you would REMOVE, use plain duplicated(), which leaves the first occurrence unflagged.
- `n_dup_rows` is not the number of exactly repeated rows in dosing. Use dosing.duplicated().sum() across all columns.
- `dup_subjects` is empty, but one subject does appear twice in dosing.
- `dup_subjects` does not name the repeated subject. Flag duplicate SUBJECT values with keep=False, then take the unique values of that column.
- `dup_subjects` should be a list of subject identifiers.
- `dup_subjects` has %d entries, so the duplicate subject is listed more than once. Take .unique() before sorting.

## Part b. Clean the key, then merge under a contract

Drop the exact repeats from `dosing` first. They really are the same record
entered twice, so removing one changes nothing except the row count.

Then merge onto `pk`, keeping every row of `pk`, and state the relationship
you believe holds: many concentration rows may match, but each subject must
appear at most once in the dosing table. That is `validate="m:1"`.

Two things will happen if you get this wrong, and both are useful. Merge
without cleaning and `validate` raises a `MergeError`, telling you the
relationship you claimed is false. Merge without `validate` and you get a
quietly oversized table. Try it both ways if you want to see it.

Afterwards, `len(merged)` must equal `n_pk_rows`. Not approximately. Exactly.

Fill in `dosing_clean`, the deduplicated dosing table with its index
renumbered. And `merged`, `pk` joined to it on `SUBJECT`, keeping every row of
`pk`, with `validate="m:1"`.

Moves: `.drop_duplicates().reset_index(drop=True)`,
`pd.merge(left, right, on=..., how="left", validate="m:1")`.

**Variables this part is graded on:** `dosing_clean`, `merged`

The starter cell they see:

```python
# Part b. Deduplicate, then merge with a contract.

dosing_clean = None   # YOUR CODE: dosing with exact repeats removed, index renumbered
merged = None         # YOUR CODE: pk joined to dosing_clean, every pk row kept,
                      #            with validate="m:1"

print("rows in pk    :", len(pk))
print("rows in merged:", None if merged is None else len(merged))
if merged is not None:
    print(merged.head())
```

**Every failure message part b can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `dosing_clean` should be a pandas DataFrame.
- `dosing_clean` still has every row of dosing, so nothing was removed. drop_duplicates() takes out the exact repeat.
- `dosing_clean` has %d rows but it should have %d, one per subject that has a dosing record.
- `dosing_clean` does not hold one row per dosed subject. Remove only exact repeats, do not drop on SUBJECT alone.
- `merged` should be a pandas DataFrame.
- `merged` has %d rows but pk has %d, so the duplicated dosing record doubled one subject's rows. Deduplicate before merging. This is the failure that validate='m:1' would have raised on.
- `merged` has %d rows but pk has %d, so an inner join dropped the subject with no dosing record. Use how='left'.
- `merged` has %d rows but it must have exactly the same %d rows as pk.
- `merged` has the right number of rows but the doses do not line up with the subjects. Check you merged on SUBJECT.
- `merged` has no %s column, so the dosing record did not come across. Merge on SUBJECT.

## Part c. Find out who did not match

The merge kept every row of `pk`, which means the subject with no dosing
record is still there, carrying `NaN` where the dose should be. Finding them
by eye does not scale, so ask pandas directly.

Repeat the merge with `indicator=True`. That adds a `_merge` column labelling
every row `both`, `left_only` or `right_only`. Count the labels, then pull out
the subjects labelled `left_only`: those are the ones in `pk` with no dosing
record.

This is the habit worth keeping. A left join plus an indicator turns a silent
data loss into a number you can put in an email to the data manager.

Fill in `checked`, the same merge but with `indicator=True`. `n_left_only`, a
whole number counting rows that found no dosing record. And
`unmatched_subjects`, a sorted list of those subjects.

Moves: `pd.merge(..., indicator=True)`, comparing `_merge` to `"left_only"`,
`.sum()`, `.unique()`, `sorted()`.

**Variables this part is graded on:** `checked`, `n_left_only`, `unmatched_subjects`

The starter cell they see:

```python
# Part c. Use an indicator column to find the unmatched rows.

checked = None             # YOUR CODE: the same merge, plus indicator=True
n_left_only = None         # YOUR CODE: how many rows found no dosing record
unmatched_subjects = None  # YOUR CODE: sorted list of those subjects

print("n_left_only       :", n_left_only)
print("unmatched_subjects:", unmatched_subjects)
if checked is not None:
    print(checked["_merge"].value_counts().to_string())
```

**Every failure message part c can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `checked` should be a pandas DataFrame.
- `checked` has no _merge column. Pass indicator=True to pd.merge and it is added for you.
- `checked` has %d rows but it should be the same merge as part b, so %d.
- `n_left_only` should be a single whole number.
- `n_left_only` counted the rows that DID match. left_only means the row came from the left table only, so it found no partner.
- `n_left_only` counts 1, which is the number of unmatched SUBJECTS. The question asks for rows, and that subject has one row per timepoint.
- `n_left_only` is not the number of rows labelled left_only. Compare the _merge column to the string 'left_only' and sum it.
- `unmatched_subjects` lists %d entries, so the subject is repeated once per row. Take .unique() before sorting.
- `unmatched_subjects` does not name the subject with no dosing record. Filter checked to the left_only rows, then take the unique SUBJECT values.
- `unmatched_subjects` should be a list of subject identifiers.

**Functions this problem introduces:** validate=, indicator=, .value_counts, sorted

---

# W2D3. Nominal versus actual sampling times, and the samples that never arrived

- topic: pandas: time deviations, building a full grid, finding missing records
- notebook: problems/week02/w2d3_pandas_nominal_actual.ipynb
- solution: solutions/week02/w2d3_pandas_nominal_actual_solution.ipynb
- expected time: 20-30 min

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

## The setup cell the student runs

This is given to them, already working. The data is synthetic and
seeded, so their numbers are always these numbers.

```python
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
```

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

The toolkit cell demonstrates each of those on a tiny made up example.
It is in the notebook and none of it is the answer.

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

**Variables this part is graded on:** `df`, `n_out_of_window`

The starter cell they see:

```python
# Part a. Deviation from nominal time, and the protocol window.

df = None               # YOUR CODE: samples plus DEV_MIN and OUT_OF_WINDOW columns
n_out_of_window = None  # YOUR CODE: how many samples fell outside the window

print("n_out_of_window:", n_out_of_window)
if df is not None:
    print(df.head())
    print("")
    print(df[df["OUT_OF_WINDOW"]])
```

**Every failure message part a can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `df` should be a pandas DataFrame.
- `df` has %d rows but samples has %d. Part a adds columns, it does not add or remove rows.
- `DEV_MIN` is in hours, not minutes. Multiply the difference by 60.
- `DEV_MIN` has had its sign removed. Keep it signed, so a late draw is positive, and take the absolute value only when you compare it to the window.
- `DEV_MIN` has the sign the wrong way round. It is actual minus nominal, so running late gives a positive number.
- `DEV_MIN` is not (ACTUAL_H - NOMINAL_H) expressed in minutes.
- `OUT_OF_WINDOW` should be a boolean column, not %s. A comparison gives booleans directly.
- `n_out_of_window` should be a single whole number.
- `n_out_of_window` used a flat 5 minute window everywhere. The window widens to 30 minutes from 2 hours onwards.
- `n_out_of_window` used a flat 30 minute window everywhere. The early timepoints, under 2 hours, only allow 5 minutes.
- `n_out_of_window` does not match. Compare the absolute deviation against a tolerance of 5 minutes below 2 hours and 30 minutes from 2 hours on.
- `OUT_OF_WINDOW` flags the wrong rows, even though the count happens to match.
- `df` has no %s column yet.

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

**Variables this part is graded on:** `grid`, `missing`, `n_missing`

The starter cell they see:

```python
# Part b. Find the samples that were scheduled but never collected.

grid = None       # YOUR CODE: every SUBJECT and NOMINAL_H combination that was scheduled
missing = None    # YOUR CODE: SUBJECT and NOMINAL_H of the samples that never arrived,
                  #            sorted by SUBJECT then NOMINAL_H, index renumbered
n_missing = None  # YOUR CODE: how many

print("scheduled:", None if grid is None else len(grid))
print("collected:", len(samples))
print("n_missing:", n_missing)
print(missing)
```

**Every failure message part b can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `grid` should be a pandas DataFrame. .to_frame(index=False) turns the MultiIndex into one.
- `grid` has the same number of rows as the collected samples, so it is not the full scheduled design. It should have one row per subject per nominal timepoint, whether or not that sample exists.
- `grid` has %d rows but %d samples were scheduled, which is the number of subjects times the number of nominal times.
- `missing` should be a pandas DataFrame.
- `missing` holds the collected samples, not the absent ones. Rows labelled left_only are the ones in the grid with no matching sample.
- `missing` has %d rows. It should hold only the scheduled samples that never arrived.
- `missing` does not name the right subjects.
- `missing` does not name the right timepoints.
- `missing` has not had its index renumbered. Use .reset_index(drop=True).
- `n_missing` should be the number of rows in missing.
- `grid` has no %s column. Pass names=['SUBJECT', 'NOMINAL_H'] to from_product.
- `missing` has no %s column.
- `missing` holds the right rows but is not sorted by SUBJECT then NOMINAL_H.

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

**Variables this part is graded on:** `per_subject`

The starter cell they see:

```python
# Part c. One row per subject: collected, missing, out of window, worst drift.

per_subject = None   # YOUR CODE

print(per_subject)
```

**Every failure message part c can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `per_subject` should be a pandas DataFrame.
- `per_subject` is missing the column(s) %s. ,
- `per_subject` has %d rows but there are %d subjects. A subject who missed nothing still needs a row.
- `per_subject` does not list every subject once, sorted by SUBJECT.
- `N_COLLECTED` is not the number of samples each subject actually has.
- `N_MISSING` has NaN in it, for the subjects who missed nothing and so do not appear in `missing`. Fill those with 0 after the join.
- `N_MISSING` does not match part b. Subjects who missed nothing should show 0, not be dropped.
- `N_OUT_OF_WINDOW` is not the per subject count of out of window samples. Summing a boolean column counts the True values.
- `MAX_ABS_DEV_MIN` is not the largest absolute deviation per subject, rounded to 1 decimal place.
- `MAX_ABS_DEV_MIN` came out negative somewhere, so you took the max of the signed deviation. Take the absolute value first.

**Functions this problem introduces:** pd.MultiIndex.from_product, .abs, .fillna, .assign

---

# W2D4. Adverse events: counting events is not counting patients

- topic: pandas: crosstab, pivot_table, normalize, subject level incidence
- notebook: problems/week02/w2d4_pandas_crosstab_incidence.ipynb
- solution: solutions/week02/w2d4_pandas_crosstab_incidence_solution.ipynb
- expected time: 20-30 min

## Scenario

A 36 subject dose ranging study has finished and you have been asked for the
safety tables: adverse events broken down by dose group and severity.

The data is an event log. One row per adverse event, so a subject who had
three events has three rows, and a subject who sailed through has **no rows at
all**. There is a separate enrolment listing of everyone who was dosed.

This is where safety tables go wrong. The obvious table counts events, and it
is the wrong table. A single unlucky subject with six mild headaches makes a
dose group look worse than one where six different people each had one event,
even though the second group is clearly the more concerning signal. What a
reviewer wants is **incidence**: how many patients experienced at least one
event, out of how many were treated.

Both tables have their uses. Confusing them is how a safety summary misleads.

## Counting rows, counting people, and the denominator problem

**`crosstab` is a frequency table.** Hand it two columns and it counts how
many rows fall in each combination, laying one out across the columns and the
other down the rows. It is the fastest way to a counts table and it is doing
nothing more complicated than `groupby` plus `unstack`, with nicer defaults.

`normalize` turns those counts into proportions. `normalize="index"` divides
each row by its own total, so each row sums to 1, which is what you want when
you are asking "of this group's events, what fraction were severe". Other
options are `"columns"` and `"all"`. Multiply by 100 for percent. Get the
direction wrong and you answer a question nobody asked, so say out loud what
each row should sum to before you pick.

**`pivot_table` is crosstab's general form.** `crosstab` can only count rows.
`pivot_table` takes a `values` column and any `aggfunc`, so it will give you
mean severity scores, maximum durations, or counts of distinct subjects. When
you find yourself wanting something other than a count, you have outgrown
`crosstab`.

**Percentages on small denominators swing wildly.** You will see it in this
dataset: the placebo group's severity split is computed from a handful of
events, so one extra severe event moves its percentage by fifteen points.
That is not a signal, it is arithmetic on a small number. This is exactly why
a safety table always shows the counts next to the percentages, and why nobody
should read a percentage without looking at what it was divided by.

**The denominator is the hard part, and it is not in the event log.**

Counting rows in the event log counts **events**. To count **patients** you
need distinct subjects, which is `nunique` rather than `size`. That difference
alone fixes the "one subject with six headaches" problem.

But there is a second, worse trap. The event log only contains subjects who
had at least one event. Subjects who had none are simply absent, so any
denominator you compute from the event log is the number of people who had
events, which makes the incidence come out as 100 percent in every group.
This is the same "absence is not a value" problem as yesterday, and it has the
same shape of answer: the denominator has to come from the **enrolment**
listing, which is the table that knows about everyone.

Incidence is therefore a join between two different tables counted two
different ways: distinct subjects with an event, over all subjects dosed. Get
either side from the wrong table and the number is wrong in a way that always
flatters the drug.

## Your task

Three parts, worked one at a time.

| Part | What you work out | Variables to fill in |
|---|---|---|
| a | the event counts table, group by severity | `ae_counts` |
| b | the same table as row percentages | `ae_pct` |
| c | subject level incidence, with the right denominator | `incidence` |

Below this you will find the **Setup** cell, then a **Toolkit** cell that
demonstrates every move on a tiny made up example, then **Part a**, **Part b**
and **Part c**, each with its own explanation and a cell to write in.

## The setup cell the student runs

This is given to them, already working. The data is synthetic and
seeded, so their numbers are always these numbers.

```python
import numpy as np
import pandas as pd

PROBLEM_ID = "W2D4"
TOPIC = "pandas: crosstab, pivot_table, normalize, subject level incidence"

rng = np.random.default_rng(2119)

GROUP_ORDER = ["Placebo", "Low", "High"]
enrolled = pd.DataFrame({
    "SUBJECT": ["A%02d" % i for i in range(1, 37)],
    "GROUP": np.repeat(GROUP_ORDER, 12),
})

_terms = ["Headache", "Nausea", "Fatigue", "Dizziness", "Rash"]
_rate = {"Placebo": 0.60, "Low": 1.10, "High": 2.00}
_severe_p = {"Placebo": [0.80, 0.18, 0.02], "Low": [0.68, 0.26, 0.06],
             "High": [0.52, 0.33, 0.15]}

_rows = []
for _subj, _grp in zip(enrolled["SUBJECT"], enrolled["GROUP"]):
    _n = rng.poisson(_rate[_grp])
    for _ in range(_n):
        _rows.append((_subj, _grp,
                      str(rng.choice(_terms)),
                      str(rng.choice(["Mild", "Moderate", "Severe"], p=_severe_p[_grp]))))

ae = pd.DataFrame(_rows, columns=["SUBJECT", "GROUP", "AE_TERM", "SEVERITY"])

print("enrolled:", enrolled.shape, " 12 subjects per group")
print(enrolled.head())
print("")
print("ae, one row per EVENT:", ae.shape)
print(ae.head(10))
print("")
print("distinct subjects appearing in the event log:", ae["SUBJECT"].nunique(),
      "out of", len(enrolled), "dosed")
```

## Toolkit for this problem

| What you want to do | How |
|---|---|
| a counts table from two columns | `pd.crosstab(df["A"], df["B"])` |
| each row as a fraction of itself | `pd.crosstab(..., normalize="index")` |
| row and column totals | `pd.crosstab(..., margins=True, margins_name="Total")` |
| any aggregation, not just counts | `pd.pivot_table(df, index=..., columns=..., values=..., aggfunc=...)` |
| count rows per group | `.agg("size")` |
| count distinct values per group | `.agg("nunique")` |
| fix the row order | `.reindex(["Placebo", "Low", "High"])` |

`size` counts rows, which means events. `nunique` on the subject column counts
people. The whole problem turns on that one word.

The toolkit cell demonstrates each of those on a tiny made up example.
It is in the notebook and none of it is the answer.

## Part a. The event counts table

The straightforward table first: how many adverse **events** occurred in each
dose group at each severity.

`pd.crosstab` does this in one call. Put `GROUP` down the rows and `SEVERITY`
across the columns, and it counts the rows falling into each cell.

Two bits of tidying make it a table someone can read. The rows come out in
alphabetical order, which puts High before Low before Placebo, and that is not
the order anyone wants to read a dose response in; reindex them to the dose
order in `GROUP_ORDER`. Then add row and column totals with `margins=True`,
calling the total row and column `Total`.

Remember what this table is and is not. It counts events, so one subject with
several events contributes several times. Part c is where that gets fixed.

Fill in `ae_counts`, a DataFrame indexed by `GROUP` in the order Placebo, Low,
High then Total, with severity columns plus a `Total` column.

Moves: `pd.crosstab(a, b, margins=True, margins_name="Total")`,
`.reindex([...])`.

**Variables this part is graded on:** `ae_counts`

The starter cell they see:

```python
# Part a. Event counts by group and severity, with totals.

ae_counts = None   # YOUR CODE

print(ae_counts)
```

**Every failure message part a can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `ae_counts` should be a DataFrame, which is what pd.crosstab returns.
- `ae_counts` has no Total row. Pass margins=True and margins_name='Total' to crosstab.
- `ae_counts` does not have one row per dose group plus a Total row.
- `ae_counts` has no Total column. margins=True adds both the row and the column.
- `ae_counts` has the right rows in the wrong order. Reindex to GROUP_ORDER plus 'Total' so it reads Placebo, Low, High.
- `ae_counts` has no %s column. Put SEVERITY across the columns.
- `ae_counts` has the right shape but the %s column does not match the event counts. crosstab counts rows of ae, nothing else.

## Part b. The same table as row percentages

Counts alone do not let you compare groups that had different numbers of
events. Convert to percentages **within each group**, so that each dose
group's row sums to 100 and you can read off how its events were distributed
across severities.

That is `normalize="index"`: divide each row by its own total. Multiply by 100
and round to one decimal.

Before you run it, say out loud what you expect each row to add up to. If you
reach for `normalize="columns"` you would be asking what fraction of all the
severe events happened in each group, which is a real question but a different
one, and the two get mixed up constantly.

No margins this time, and keep the dose order.

One thing to notice when you look at the result. The placebo row rests on only
a handful of events, so its percentages are extremely unstable: a single extra
severe event would move that row by double digits. The dose groups have more
events behind them and their rows are correspondingly steadier. This is why a
real safety table always prints the counts beside the percentages, and why you
should distrust any percentage whose denominator you have not seen.

Fill in `ae_pct`, indexed by `GROUP` in the order Placebo, Low, High, with the
severity columns as percentages rounded to 1 decimal place, each row summing
to 100.

Moves: `pd.crosstab(..., normalize="index")`, `* 100`, `.round(1)`,
`.reindex([...])`.

**Variables this part is graded on:** `ae_pct`

The starter cell they see:

```python
# Part b. Row percentages: of each group's events, what share was each severity.

ae_pct = None   # YOUR CODE

print(ae_pct)
if ae_pct is not None:
    print("")
    print("row sums:", ae_pct.sum(axis=1).to_dict())
```

**Every failure message part b can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `ae_pct` should be a DataFrame.
- `ae_pct` should not have a Total row. Drop margins for this one.
- `ae_pct` does not have one row per dose group.
- `ae_pct` is still a fraction, since the rows sum to 1. Multiply by 100 to get percent.
- `ae_pct` was normalised down the columns, so it answers what share of all the severe events came from each group. The task asks for each group's events split across severities, which is normalize='index'.
- `ae_pct` was normalised over the whole table, so everything sums to 100 once. Each ROW should sum to 100, which is normalize='index'.
- `ae_pct` rows do not sum to 100. Each dose group's events should be split across the severity columns.
- `ae_pct` rows are not in the dose order Placebo, Low, High. Reindex them.
- `ae_pct` has no %s column.
- `ae_pct` rows sum to 100 but the %s column does not match.

## Part c. Subject level incidence, with the denominator from the right table

Now the table a reviewer actually wants: of the people dosed in each group,
what share had at least one adverse event.

Two different counts, from two different tables.

The **numerator** is the number of distinct subjects appearing in the event
log for that group, so `nunique` on the subject column, not `size`. A subject
with six events counts once.

The **denominator** is the number of subjects **dosed** in that group, and it
has to come from `enrolled`. Taking it from `ae` would count only people who
had events, and every group would read 100 percent.

Join them and watch for the group with no events at all, if there is one: it
will be absent from the numerator side and needs a 0 rather than a gap.

Fill in `incidence`, sorted in the order Placebo, Low, High, index renumbered,
with exactly:

| column | meaning |
|---|---|
| `GROUP` | dose group |
| `N_DOSED` | subjects dosed in that group, from `enrolled` |
| `N_WITH_AE` | distinct subjects with at least one event |
| `PCT_WITH_AE` | `100 * N_WITH_AE / N_DOSED`, rounded to 1 decimal |

Moves: `.groupby(...)["SUBJECT"].nunique()`, `.reset_index(name=...)`,
`pd.merge(..., how="left")`, `.fillna(0).astype(int)`, `.round(1)`.

**Variables this part is graded on:** `incidence`

The starter cell they see:

```python
# Part c. Incidence: subjects with at least one event, over subjects dosed.

incidence = None   # YOUR CODE

print(incidence)
```

**Every failure message part c can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `incidence` should be a pandas DataFrame.
- `incidence` is missing the column(s) %s. ,
- `incidence` has %d rows but there are %d dose groups.
- `incidence` is not in the dose order Placebo, Low, High. Sort it into GROUP_ORDER rather than leaving it alphabetical.
- `N_DOSED` was taken from the event log, so it counts only subjects who had an event. The denominator has to come from `enrolled`, which knows about everyone who was dosed.
- `N_DOSED` is not the number of subjects dosed in each group, from enrolled.
- `N_WITH_AE` counted EVENTS, not subjects. A subject with several events should count once, so use nunique on the subject column rather than size.
- `N_WITH_AE` is not the number of distinct subjects with at least one event.
- `PCT_WITH_AE` is 100 in every group, which is the giveaway that the denominator came from the event log.
- `PCT_WITH_AE` used the wrong denominator. It should be N_WITH_AE over N_DOSED.
- `PCT_WITH_AE` is not 100 * N_WITH_AE / N_DOSED, rounded to 1 decimal place.

**Functions this problem introduces:** pd.crosstab, pd.pivot_table, size versus nunique, .reindex, .map

---

# W2D5. Two panels and a spaghetti plot: the figure every PK report opens with

- topic: matplotlib: subplots, semilog axes, per subject overlays
- notebook: problems/week02/w2d5_matplotlib_panels_semilog.ipynb
- solution: solutions/week02/w2d5_matplotlib_panels_semilog_solution.ipynb
- expected time: 20-30 min

## Scenario

Sixteen subjects, eight on each of two dose levels, sampled seven times over
24 hours. Your group leader wants the standard first look at a new dataset,
which is always the same two figures.

The first is the **mean profile**, shown twice: once on a linear y axis, which
is how a clinician reads exposure, and once on a log y axis, which is where a
pharmacometrician can see whether elimination is first order and read the
slope by eye. Same data, same panel size, side by side.

The second is the **spaghetti plot**: every subject's own profile, drawn
individually, no averaging at all. It is the figure that tells you whether the
mean means anything, or whether it is an average of two populations, or is
being dragged around by one outlier.

Between them these two figures answer "what does this drug do" and "do I
believe the summary", which is why they open every report.

## Panels, log axes, and why the spaghetti plot comes first

**Several panels from one call.** `plt.subplots(1, 2)` returns the figure and
an **array** of axes, which you index: `axes[0]` and `axes[1]`. With no rows
and columns argument you get a single axes which you must not index. Getting
this wrong gives either `TypeError: 'Axes' object is not subscriptable` or an
`AttributeError` when you call a plotting method on a numpy array, and both
mean you disagree with matplotlib about how many panels you asked for.

`figsize` is the whole figure in inches, not each panel, so a two panel figure
wants roughly double the width of a one panel figure or the panels come out
squashed. `fig.tight_layout()` at the end stops the labels of one panel
colliding with the next.

**Why the log axis matters here.** First order elimination means a constant
*fraction* leaves per hour, which is exponential decay. Take the log of an
exponential and you get a straight line. So on a semilog axis the terminal
phase of a PK profile becomes straight, and you can see at a glance whether
one compartment is enough, whether there is a second slower phase, and roughly
what the half-life is. On a linear axis all of that is hidden in a curve that
hugs the x axis.

The trap: a log axis cannot show zero or negative values. They do not error,
they silently vanish, taking your predose samples with them. If a profile
looks like it starts late, that is usually why.

**Why the spaghetti plot comes first in practice.** A mean with an SD bar
describes the group as though every subject behaves like the average one with
some scatter around it. Plot the individuals and you find out whether that is
true. Two subpopulations, one subject who did not absorb, a sample swapped
between two people: none of these show up in a mean, and all of them are
obvious in thirty seconds of looking at the lines. Draw it before you believe
any summary statistic you computed, not after.

Keep the individual lines visually quiet: thin, partly transparent, one
colour per group rather than sixteen different colours. The reader is meant to
see the shape of the bundle, not trace any one line.

## Your task

Three parts, worked one at a time.

| Part | What you work out | Variables to fill in |
|---|---|---|
| a | the mean and SD profile per dose group | `summary` |
| b | a two panel figure, linear and semilog | `fig`, `axes` |
| c | a spaghetti plot of every subject | `fig2`, `ax2` |

Below this you will find the **Setup** cell, then a **Toolkit** cell that
demonstrates every move on a tiny made up example, then **Part a**, **Part b**
and **Part c**, each with its own explanation and a cell to write in.

## The setup cell the student runs

This is given to them, already working. The data is synthetic and
seeded, so their numbers are always these numbers.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROBLEM_ID = "W2D5"
TOPIC = "matplotlib: subplots, semilog axes, per subject overlays"

rng = np.random.default_rng(2105)

TIMES = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0])
DOSE_ORDER = ["50 mg", "200 mg"]
SUBJECTS = ["D%02d" % i for i in range(1, 17)]
_group = np.repeat(DOSE_ORDER, 8)
_dose = np.repeat([50.0, 200.0], 8)

_V = 44.0 * rng.lognormal(0.0, 0.24, size=len(SUBJECTS))
_kel = 0.115 * rng.lognormal(0.0, 0.26, size=len(SUBJECTS))
_ka = 1.05 * rng.lognormal(0.0, 0.20, size=len(SUBJECTS))
_coef = (0.86 * _dose * _ka) / (_V * (_ka - _kel))
_true = _coef[:, None] * (np.exp(-_kel[:, None] * TIMES[None, :])
                          - np.exp(-_ka[:, None] * TIMES[None, :]))

pk = pd.DataFrame({
    "SUBJECT": np.repeat(SUBJECTS, TIMES.size),
    "DOSE": np.repeat(_group, TIMES.size),
    "TIME_H": np.tile(TIMES, len(SUBJECTS)),
    "CONC_MG_L": np.round(_true.ravel() * rng.lognormal(0.0, 0.13, size=_true.size), 4),
})

print("pk:", pk.shape)
print(pk.head())
print("")
print("subjects per dose:", pk.groupby("DOSE")["SUBJECT"].nunique().to_dict())
```

## Toolkit for this problem

| What you want to do | How |
|---|---|
| two panels side by side | `fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))` |
| one panel | `fig, ax = plt.subplots(figsize=(6, 4.5))`, and do not index it |
| points with SD bars | `ax.errorbar(x, y, yerr=sd, fmt="o-", capsize=4, label=...)` |
| a log y axis | `ax.set_yscale("log")` |
| one line per subject | loop over groups, `ax.plot(x, y, lw=0.9, alpha=0.55)` |
| split a frame into groups | `for name, part in df.groupby("DOSE"):` |
| stop labels colliding | `fig.tight_layout()` |
| a legend | pass `label=` to each call, then `ax.legend()` |

Looping over `df.groupby("COL")` hands you the group name and that group's
rows on each turn, which is usually tidier than filtering by hand.

The toolkit cell demonstrates each of those on a tiny made up example.
It is in the notebook and none of it is the answer.

## Part a. The mean and SD profile per dose group

Before any figure, the numbers it will be built from.

One row per `DOSE` and `TIME_H`, with the mean concentration, its sample SD
across subjects, and the number of subjects contributing. The count should be
8 in every cell; if it is not, something upstream lost rows and the figure
would quietly misrepresent it.

SD, never SEM. Pandas `std` already uses `n - 1`.

Fill in `summary`, a flat DataFrame with exactly:

| column | meaning |
|---|---|
| `DOSE` | `50 mg` or `200 mg` |
| `TIME_H` | hours after dose |
| `CONC_MEAN` | mean concentration at that time |
| `CONC_SD` | sample SD across subjects |
| `N` | how many subjects contributed |

Moves: `.groupby([...]).agg(NAME=("COL", "mean"), ...)`, `.reset_index()`.

**Variables this part is graded on:** `summary`

The starter cell they see:

```python
# Part a. Mean, SD and n per dose group per timepoint.

summary = None   # YOUR CODE

print(summary)
```

**Every failure message part a can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `summary` should be a pandas DataFrame.
- `summary` is missing the column(s) %s. ,
- `summary` has %d rows but there should be one per dose group and timepoint, so %d.
- `summary` does not cover the expected dose groups.
- `summary` does not cover the expected timepoints.
- `N` is not the number of subjects per cell. It should be 8 everywhere; the aggregation 'size' counts rows within the group.
- `CONC_MEAN` is not the mean concentration within each dose and timepoint.
- `CONC_SD` is the SEM, not the SD. Something divided by the square root of n. This repo always reports SD.
- `CONC_SD` used n in the denominator. Pandas std already uses n-1, so this usually means numpy's std got involved.
- `CONC_SD` is not the sample SD within each dose and timepoint.
- `%s` is in the index rather than a column. Call .reset_index().

## Part b. Two panels, linear and semilog

One figure, two panels side by side, showing the **same** mean profiles twice.

Build it with `plt.subplots(1, 2)`, which hands you an array of two axes.
Loop over the dose groups in `summary` and draw each group on **both** panels
with `ax.errorbar`, using the `CONC_SD` column as `yerr` and a label so the
legend knows what it is looking at.

Then make the right panel logarithmic in y. Everything else about the two
panels is identical: that is the point, since the only thing changing is how
the eye reads the same numbers.

Label both panels on both axes and give each a title, so the figure survives
being pasted into a slide on its own. Put a legend on at least one panel, and
finish with `fig.tight_layout()`.

Fill in `fig` and `axes`, where `axes` holds the two panels, `axes[0]` linear
and `axes[1]` logarithmic in y.

Moves: `plt.subplots(1, 2, figsize=...)`, `for name, part in
summary.groupby("DOSE")`, `ax.errorbar(..., yerr=..., label=...)`,
`ax.set_yscale("log")`, `ax.set_xlabel`, `ax.set_ylabel`, `ax.set_title`,
`ax.legend()`, `fig.tight_layout()`.

**Variables this part is graded on:** see the starter

The starter cell they see:

```python
# Part b. Two panels: the same mean profiles, linear on the left, log y on the right.

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

# YOUR CODE: loop over the dose groups in summary and errorbar each one onto BOTH panels,
#            with yerr taken from CONC_SD and a label for the legend

# YOUR CODE: make axes[1] logarithmic in y

# YOUR CODE: xlabel, ylabel and title on both panels, and a legend on at least one

fig.tight_layout()
plt.show()
```

**Every failure message part b can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `fig` should be a matplotlib Figure from plt.subplots().
- `axes` is a single Axes, so plt.subplots() was called without asking for two panels. Use plt.subplots(1, 2).
- `axes` holds %d panels but the figure should have exactly 2.
- `axes` should hold matplotlib Axes objects.
- Panel %d has nothing drawn on it. Both panels show the SAME mean profiles; only the y scale differs.
- Neither panel has a log y axis. Call axes[1].set_yscale('log').
- Both panels are logarithmic. axes[0] should stay linear for the comparison.
- Neither panel has a legend, so nothing says which line is which dose. Pass label= to each errorbar call, then call ax.legend() on one panel.
- Panel %d has no error bars. Draw the means with ax.errorbar and pass yerr, on both panels.
- Panel %d has SEM error bars, not SD. This repo always uses SD, so pass the CONC_SD column as yerr.
- Panel %d's error bars do not match the CONC_SD column of summary. Both panels should show the same numbers.
- The LEFT panel is logarithmic and the right one is linear. The task wants axes[0] linear and axes[1] logarithmic.
- Panel %d has no x label. Label both panels.
- Panel %d has no y label, so a reader cannot tell what the numbers are.
- Panel %d has no title. Give each panel one, so the figure stands alone.

## Part c. The spaghetti plot

A single panel, one line per subject, no averaging anywhere.

Loop over subjects and draw each one's concentration against time. Colour by
dose group rather than by subject, so the two bundles separate without
sixteen competing colours, and keep the lines thin and partly transparent so
the reader sees the shape of each bundle rather than trying to follow
individual lines.

Use a log y axis here too. On a linear axis the low dose bundle is squashed
into the bottom of the panel and you cannot see its shape at all.

Draw **exactly one line per subject and nothing else on this axes**, so there
should be sixteen lines when you are done. No mean, no error bars. That is
what makes this figure honest: it is the raw data and nothing else.

Fill in `fig2` and `ax2`, a single axes with one line per subject, a log y
axis, and all three labels.

Moves: `plt.subplots(figsize=...)`, `for sid, rows in pk.groupby("SUBJECT")`,
`ax.plot(..., lw=..., alpha=..., color=...)`, `ax.set_yscale("log")`,
`ax.set_xlabel`, `ax.set_ylabel`, `ax.set_title`.

**Variables this part is graded on:** see the starter

The starter cell they see:

```python
# Part c. Spaghetti plot: every subject, no averaging.

fig2, ax2 = plt.subplots(figsize=(6.5, 4.5))

colours = {"50 mg": "tab:blue", "200 mg": "tab:red"}

# YOUR CODE: loop over subjects and plot each one's profile, thin and semi transparent,
#            coloured by that subject's dose group

# YOUR CODE: log y axis

# YOUR CODE: xlabel, ylabel and title

plt.show()
```

**Every failure message part c can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `fig2` should be a matplotlib Figure.
- `ax2` should be a single Axes. plt.subplots() with no rows or columns argument gives exactly one.
- There are error bars on this axes. The spaghetti plot is raw data only, one plain line per subject and nothing else.
- Only %d lines were drawn, which is one per dose group rather than one per subject. Loop over SUBJECT, not DOSE.
- There are %d lines, which is one per subject plus a group mean. Draw the subjects only; the means are the other figure.
- There are %d lines on the axes but there are %d subjects, so there should be exactly that many.
- The y axis is not logarithmic. Call ax2.set_yscale('log'), or the low dose bundle is squashed into the bottom of the panel.
- The x axis has no label.
- The y axis has no label, so a reader cannot tell what the numbers are.
- The axes has no title.
- One of the lines has %d points but each subject has %d timepoints. Plot each subject's own rows.

**Functions this problem introduces:** looping over groupby, ax.set_yscale, plt.subplots(rows, cols), fig.tight_layout
