PID = "W2D2"
WEEK = 2
DAY = 2
SLUG = "w2d2_pandas_merge_validate"
TITLE = "Joins that lie: validate the merge before you trust it"
TOPIC = "pandas: join types, validate, indicator, duplicate keys"
EST = "20-30 min"

SCENARIO_MD = '''
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
'''

CONCEPT_MD = '''
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
'''

TASK_MD = '''
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
'''

DATA_CODE = '''
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
'''

TOOLKIT_MD = '''
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
'''

TOOLKIT_CODE = '''
import numpy as np
import pandas as pd

# Tiny made up tables. Nothing here is the answer.
demo_left = pd.DataFrame({"K": ["a", "a", "b", "b", "c", "c"],
                          "V": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
demo_right = pd.DataFrame({"K": ["a", "b", "b"], "INFO": ["x", "y", "y"]})

print("demo_left:");  print(demo_left)
print("");  print("demo_right, note that b appears twice and c is absent:")
print(demo_right)

print("")
print("--- 1. finding the duplicate key before you merge ---")
print("duplicated(subset=['K'], keep=False):")
print(demo_right.duplicated(subset=["K"], keep=False).to_list())
print("fully repeated rows:", int(demo_right.duplicated().sum()))
print("which keys repeat :", sorted(demo_right.loc[
    demo_right.duplicated(subset=["K"], keep=False), "K"].unique()))

print("")
print("--- 2. what each join does to the row count ---")
print("rows in demo_left              :", len(demo_left))
print("how='inner'                    :", len(pd.merge(demo_left, demo_right, on="K", how="inner")),
      " <- c was dropped, b was doubled")
print("how='left'                     :", len(pd.merge(demo_left, demo_right, on="K", how="left")),
      " <- c kept as NaN, b still doubled")
print("how='left' after drop_duplicates:",
      len(pd.merge(demo_left, demo_right.drop_duplicates(), on="K", how="left")),
      " <- matches the left table, which is what you want")

print("")
print("--- 3. validate turns the silent duplication into an error ---")
try:
    pd.merge(demo_left, demo_right, on="K", how="left", validate="m:1")
except Exception as err:
    print(type(err).__name__ + ":", err)
print("")
print("after dropping the repeat, the same call is happy:")
print(len(pd.merge(demo_left, demo_right.drop_duplicates(), on="K", how="left",
                   validate="m:1")), "rows")

print("")
print("--- 4. indicator shows you exactly who did not match ---")
demo_checked = pd.merge(demo_left, demo_right.drop_duplicates(), on="K",
                        how="left", indicator=True)
print(demo_checked)
print("")
print(demo_checked["_merge"].value_counts().to_string())
print("")
print("keys with no partner:", sorted(demo_checked.loc[
    demo_checked["_merge"] == "left_only", "K"].unique()))
'''

PARTS = [
    dict(
        label="a",
        prompt_md='''
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
''',
        starter='''
# Part a. Inspect before merging.

n_pk_rows = None      # YOUR CODE: how many rows pk has
n_dup_rows = None     # YOUR CODE: how many rows of dosing are exact repeats
dup_subjects = None   # YOUR CODE: sorted list of the subjects involved

print("n_pk_rows   :", n_pk_rows)
print("n_dup_rows  :", n_dup_rows)
print("dup_subjects:", dup_subjects)
''',
        solution='''
# Part a. Inspect before merging.

n_pk_rows = len(pk)
n_dup_rows = int(dosing.duplicated().sum())
dup_subjects = sorted(dosing.loc[dosing.duplicated(subset=["SUBJECT"], keep=False),
                                 "SUBJECT"].unique())

print("n_pk_rows   :", n_pk_rows)
print("n_dup_rows  :", n_dup_rows)
print("dup_subjects:", dup_subjects)
''',
    ),
    dict(
        label="b",
        prompt_md='''
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
''',
        starter='''
# Part b. Deduplicate, then merge with a contract.

dosing_clean = None   # YOUR CODE: dosing with exact repeats removed, index renumbered
merged = None         # YOUR CODE: pk joined to dosing_clean, every pk row kept,
                      #            with validate="m:1"

print("rows in pk    :", len(pk))
print("rows in merged:", None if merged is None else len(merged))
if merged is not None:
    print(merged.head())
''',
        solution='''
# Part b. Deduplicate, then merge with a contract.

dosing_clean = dosing.drop_duplicates().reset_index(drop=True)
merged = pd.merge(pk, dosing_clean, on="SUBJECT", how="left", validate="m:1")

print("rows in pk    :", len(pk))
print("rows in merged:", None if merged is None else len(merged))
if merged is not None:
    print(merged.head())
''',
    ),
    dict(
        label="c",
        prompt_md='''
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
''',
        starter='''
# Part c. Use an indicator column to find the unmatched rows.

checked = None             # YOUR CODE: the same merge, plus indicator=True
n_left_only = None         # YOUR CODE: how many rows found no dosing record
unmatched_subjects = None  # YOUR CODE: sorted list of those subjects

print("n_left_only       :", n_left_only)
print("unmatched_subjects:", unmatched_subjects)
if checked is not None:
    print(checked["_merge"].value_counts().to_string())
''',
        solution='''
# Part c. Use an indicator column to find the unmatched rows.

checked = pd.merge(pk, dosing_clean, on="SUBJECT", how="left",
                   validate="m:1", indicator=True)
n_left_only = int((checked["_merge"] == "left_only").sum())
unmatched_subjects = sorted(checked.loc[checked["_merge"] == "left_only",
                                        "SUBJECT"].unique())

print("n_left_only       :", n_left_only)
print("unmatched_subjects:", unmatched_subjects)
if checked is not None:
    print(checked["_merge"].value_counts().to_string())
''',
    ),
]


def build_expected(ns):
    import pandas as pd
    pk, dosing = ns["pk"], ns["dosing"]
    clean = dosing.drop_duplicates().reset_index(drop=True)
    naive = pd.merge(pk, dosing, on="SUBJECT", how="left")
    inner = pd.merge(pk, clean, on="SUBJECT", how="inner")
    merged = ns["merged"]
    checked = ns["checked"]
    left_only = checked.loc[checked["_merge"] == "left_only", "SUBJECT"]
    return {
        "n_pk_rows": int(len(pk)),
        "n_dosing_rows": int(len(dosing)),
        "n_dup_rows": int(dosing.duplicated().sum()),
        "n_dup_all_copies": int(dosing.duplicated(subset=["SUBJECT"], keep=False).sum()),
        "dup_subjects": sorted(str(v) for v in dosing.loc[
            dosing.duplicated(subset=["SUBJECT"], keep=False), "SUBJECT"].unique()),
        "n_clean_rows": int(len(clean)),
        "clean_subjects": sorted(str(v) for v in clean["SUBJECT"]),
        "n_naive_rows": int(len(naive)),
        "n_inner_rows": int(len(inner)),
        "merged_cols": [str(c) for c in merged.columns],
        "merged_dose": [float(v) for v in merged["DOSE_MG"].fillna(-1.0)],
        "n_left_only": int((checked["_merge"] == "left_only").sum()),
        "n_both": int((checked["_merge"] == "both").sum()),
        "unmatched_subjects": sorted(str(v) for v in left_only.unique()),
    }


CHECK_CODE = '''
def _part_a():
    n_pk_rows = _need("n_pk_rows")
    n_dup_rows = _need("n_dup_rows")
    dup_subjects = _need("dup_subjects")

    got = _as_scalar(n_pk_rows)
    if got is None or int(round(got)) != _EXP["n_pk_rows"]:
        return "`n_pk_rows` is not the number of rows in pk. len(pk) gives it."

    got = _as_scalar(n_dup_rows)
    if got is None:
        return "`n_dup_rows` should be a single whole number."
    if int(round(got)) == _EXP["n_dup_all_copies"] and _EXP["n_dup_all_copies"] != _EXP["n_dup_rows"]:
        return ("`n_dup_rows` counted every copy, because you passed keep=False. For "
                "how many rows you would REMOVE, use plain duplicated(), which leaves the "
                "first occurrence unflagged.")
    if int(round(got)) != _EXP["n_dup_rows"]:
        return ("`n_dup_rows` is not the number of exactly repeated rows in dosing. Use "
                "dosing.duplicated().sum() across all columns.")

    try:
        got_list = sorted(str(v) for v in dup_subjects)
    except TypeError:
        return "`dup_subjects` should be a list of subject identifiers."
    if len(got_list) == 0:
        return "`dup_subjects` is empty, but one subject does appear twice in dosing."
    if got_list != _EXP["dup_subjects"]:
        if len(got_list) > len(_EXP["dup_subjects"]):
            return ("`dup_subjects` has %d entries, so the duplicate subject is listed more "
                    "than once. Take .unique() before sorting." % len(got_list))
        return ("`dup_subjects` does not name the repeated subject. Flag duplicate SUBJECT "
                "values with keep=False, then take the unique values of that column.")
    return None


def _part_b():
    dosing_clean = _need("dosing_clean")
    merged = _need("merged")

    if not isinstance(dosing_clean, _pd.DataFrame):
        return "`dosing_clean` should be a pandas DataFrame."
    if len(dosing_clean) == _EXP["n_dosing_rows"]:
        return ("`dosing_clean` still has every row of dosing, so nothing was removed. "
                "drop_duplicates() takes out the exact repeat.")
    if len(dosing_clean) != _EXP["n_clean_rows"]:
        return ("`dosing_clean` has %d rows but it should have %d, one per subject that "
                "has a dosing record." % (len(dosing_clean), _EXP["n_clean_rows"]))
    if sorted(str(v) for v in dosing_clean["SUBJECT"]) != _EXP["clean_subjects"]:
        return ("`dosing_clean` does not hold one row per dosed subject. Remove only exact "
                "repeats, do not drop on SUBJECT alone.")

    if not isinstance(merged, _pd.DataFrame):
        return "`merged` should be a pandas DataFrame."
    cols = _cols_of(merged) or []
    for name in ("DOSE_MG", "COHORT"):
        if name not in cols:
            return ("`merged` has no %s column, so the dosing record did not come across. "
                    "Merge on SUBJECT." % name)
    if len(merged) == _EXP["n_naive_rows"] and _EXP["n_naive_rows"] != _EXP["n_pk_rows"]:
        return ("`merged` has %d rows but pk has %d, so the duplicated dosing record "
                "doubled one subject's rows. Deduplicate before merging. This is the "
                "failure that validate='m:1' would have raised on."
                % (len(merged), _EXP["n_pk_rows"]))
    if len(merged) == _EXP["n_inner_rows"] and _EXP["n_inner_rows"] != _EXP["n_pk_rows"]:
        return ("`merged` has %d rows but pk has %d, so an inner join dropped the subject "
                "with no dosing record. Use how='left'."
                % (len(merged), _EXP["n_pk_rows"]))
    if len(merged) != _EXP["n_pk_rows"]:
        return ("`merged` has %d rows but it must have exactly the same %d rows as pk."
                % (len(merged), _EXP["n_pk_rows"]))
    if not _arr_close(merged["DOSE_MG"].fillna(-1.0).to_numpy(), _EXP["merged_dose"]):
        return ("`merged` has the right number of rows but the doses do not line up with "
                "the subjects. Check you merged on SUBJECT.")
    return None


def _part_c():
    checked = _need("checked")
    n_left_only = _need("n_left_only")
    unmatched_subjects = _need("unmatched_subjects")

    if not isinstance(checked, _pd.DataFrame):
        return "`checked` should be a pandas DataFrame."
    if "_merge" not in (_cols_of(checked) or []):
        return ("`checked` has no _merge column. Pass indicator=True to pd.merge and it is "
                "added for you.")
    if len(checked) != _EXP["n_pk_rows"]:
        return ("`checked` has %d rows but it should be the same merge as part b, so %d."
                % (len(checked), _EXP["n_pk_rows"]))

    got = _as_scalar(n_left_only)
    if got is None:
        return "`n_left_only` should be a single whole number."
    if int(round(got)) == _EXP["n_both"]:
        return ("`n_left_only` counted the rows that DID match. left_only means the row "
                "came from the left table only, so it found no partner.")
    if int(round(got)) == 1:
        return ("`n_left_only` counts 1, which is the number of unmatched SUBJECTS. The "
                "question asks for rows, and that subject has one row per timepoint.")
    if int(round(got)) != _EXP["n_left_only"]:
        return ("`n_left_only` is not the number of rows labelled left_only. Compare the "
                "_merge column to the string 'left_only' and sum it.")

    try:
        got_list = sorted(str(v) for v in unmatched_subjects)
    except TypeError:
        return "`unmatched_subjects` should be a list of subject identifiers."
    if len(got_list) > len(_EXP["unmatched_subjects"]):
        return ("`unmatched_subjects` lists %d entries, so the subject is repeated once per "
                "row. Take .unique() before sorting." % len(got_list))
    if got_list != _EXP["unmatched_subjects"]:
        return ("`unmatched_subjects` does not name the subject with no dosing record. "
                "Filter checked to the left_only rows, then take the unique SUBJECT values.")
    return None
'''

# Functions this problem introduces. Keys must exist in tools/function_catalog.py.
FUNCTIONS = [
    "validate=",
    "indicator=",
    ".value_counts",
    "sorted",
]
