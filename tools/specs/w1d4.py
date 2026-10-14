PID = "W1D4"
WEEK = 1
DAY = 4
SLUG = "w1d4_pandas_tidy_wide"
TITLE = "A collaborator's spreadsheet: reshape wide to tidy and back again"
TOPIC = "pandas: melt, string cleaning, duplicates, pivot"
EST = "20-30 min"

SCENARIO_MD = '''
## Scenario

A collaborator has emailed you the target engagement readout from a mouse PD
study: percent inhibition of the target enzyme in whole blood, at 6 timepoints
after a single oral dose, for 10 animals in two groups.

It arrives the way these things always arrive: as a spreadsheet laid out for
human eyes, with **one row per animal and one column per timepoint**
(`T_0`, `T_1`, `T_2`, and so on). Every plotting and modelling tool you will
use wants the opposite shape, one row per observation, so the first job is
always the same reshape.

There is also something wrong with the file. The collaborator re-sent one
animal's record after a transcription fix and the two copies both ended up in
the sheet. You will not notice until part c, when `pivot` refuses to run. That
is the normal way you find out.
'''

TASK_MD = '''
## Your task

### Part a. Wide to tidy
Reshape `wide` into `tidy`: one row per animal per timepoint, with these four
columns, named exactly:

| column | meaning |
|---|---|
| `ANIMAL` | animal identifier |
| `GROUP` | `Vehicle` or `Drug` |
| `TIME_LABEL` | the original column name, so still the text `T_0`, `T_1`, and so on |
| `INHIB_PCT` | the percent inhibition value |

### Part b. Make the time column numeric and sort
Build `tidy_clean` from `tidy`:
- add a numeric column `TIME_H`, the hours pulled out of `TIME_LABEL`, so
  `T_24` becomes `24.0`,
- drop the `TIME_LABEL` column, leaving `ANIMAL`, `GROUP`, `TIME_H`,
  `INHIB_PCT`,
- sort by `ANIMAL` then `TIME_H`, and reset the index so it runs 0, 1, 2, ...

### Part c. Find the duplicate, then go back to wide
- `n_dup_rows`: a whole number, how many rows of `tidy_clean` are exact repeats
  of a row that already appeared.
- `wide_again`: the duplicate-free data pivoted back to wide, with `ANIMAL` as
  the index and the values of `TIME_H` as the columns, holding `INHIB_PCT`.

If you go straight to `pivot` you will get `ValueError: Index contains
duplicate entries, cannot reshape`. That error is the clue, not a bug.

### Hints
- `df.melt(id_vars=[...], var_name=..., value_name=...)` goes wide to tidy.
- `df["COL"].str.replace("T_", "", regex=False)` strips the prefix, and
  `.astype(float)` converts it.
- `df.drop(columns=[...])`, `df.sort_values([...])`,
  `df.reset_index(drop=True)`.
- `df.duplicated()` flags every row after the first occurrence, so
  `.sum()` counts the repeats. `df.drop_duplicates()` removes them.
- `df.pivot(index=..., columns=..., values=...)` goes tidy to wide.
'''

DATA_CODE = '''
import numpy as np
import pandas as pd

PROBLEM_ID = "W1D4"
TOPIC = "pandas: melt, string cleaning, duplicates, pivot"

rng = np.random.default_rng(1104)

times = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 24.0])
time_cols = ["T_%g" % t for t in times]
animals = ["M%02d" % i for i in range(1, 11)]
groups = ["Vehicle"] * 5 + ["Drug"] * 5

_rows = []
for _animal, _group in zip(animals, groups):
    if _group == "Drug":
        _imax = 105.0 * rng.lognormal(0.0, 0.12)
        _signal = _imax * (np.exp(-0.08 * times) - np.exp(-1.50 * times))
    else:
        _signal = np.zeros_like(times) + 2.0
    _obs = np.clip(_signal + rng.normal(0.0, 3.0, size=times.size), 0.0, 100.0)
    _obs[0] = np.clip(rng.normal(1.5, 1.5), 0.0, 100.0)   # predose, no inhibition yet
    _rows.append(np.round(_obs, 1))

wide = pd.DataFrame(np.array(_rows), columns=time_cols)
wide.insert(0, "GROUP", groups)
wide.insert(0, "ANIMAL", animals)

# The collaborator re-sent M04 after a transcription fix, and both copies stayed in.
wide = pd.concat([wide, wide[wide["ANIMAL"] == "M04"]], ignore_index=True)

print("wide:", wide.shape)
print(wide)
'''

PARTS = [
    dict(
        label="a",
        prompt_md='''
### Part a. Wide to tidy
''',
        starter='''
# Part a. Reshape wide into tidy.
# Columns wanted: ANIMAL, GROUP, TIME_LABEL, INHIB_PCT

tidy = None   # YOUR CODE

print(None if tidy is None else tidy.shape)
print(None if tidy is None else tidy.head())
''',
        solution='''
# Part a. Reshape wide into tidy.
# Columns wanted: ANIMAL, GROUP, TIME_LABEL, INHIB_PCT

tidy = wide.melt(id_vars=["ANIMAL", "GROUP"],
                 var_name="TIME_LABEL",
                 value_name="INHIB_PCT")

print(None if tidy is None else tidy.shape)
print(None if tidy is None else tidy.head())
''',
    ),
    dict(
        label="b",
        prompt_md='''
### Part b. Make the time column numeric and sort
''',
        starter='''
# Part b. Numeric TIME_H, drop TIME_LABEL, sort by ANIMAL then TIME_H, reset the index.

tidy_clean = None   # YOUR CODE

print(None if tidy_clean is None else tidy_clean.head(8))
''',
        solution='''
# Part b. Numeric TIME_H, drop TIME_LABEL, sort by ANIMAL then TIME_H, reset the index.

tidy_clean = tidy.copy()
tidy_clean["TIME_H"] = tidy_clean["TIME_LABEL"].str.replace("T_", "", regex=False).astype(float)
tidy_clean = (tidy_clean
              .drop(columns=["TIME_LABEL"])
              .sort_values(["ANIMAL", "TIME_H"])
              .reset_index(drop=True))

print(None if tidy_clean is None else tidy_clean.head(8))
''',
    ),
    dict(
        label="c",
        prompt_md='''
### Part c. Find the duplicate, then go back to wide
''',
        starter='''
# Part c. Count the repeated rows, drop them, pivot back to wide.

n_dup_rows = None   # YOUR CODE: how many rows repeat a row already seen
wide_again = None   # YOUR CODE: pivot of the duplicate-free data

print("n_dup_rows:", n_dup_rows)
print(None if wide_again is None else wide_again)
''',
        solution='''
# Part c. Count the repeated rows, drop them, pivot back to wide.

n_dup_rows = int(tidy_clean.duplicated().sum())
wide_again = (tidy_clean
              .drop_duplicates()
              .pivot(index="ANIMAL", columns="TIME_H", values="INHIB_PCT"))

print("n_dup_rows:", n_dup_rows)
print(None if wide_again is None else wide_again)
''',
    ),
]


def build_expected(ns):
    wide = ns["wide"]
    tidy_clean = ns["tidy_clean"]
    wide_again = ns["wide_again"]
    return {
        "n_wide_rows": int(len(wide)),
        "n_time_cols": int(len(ns["time_cols"])),
        "tidy_cols": ["ANIMAL", "GROUP", "TIME_LABEL", "INHIB_PCT"],
        "n_tidy_rows": int(len(ns["tidy"])),
        "clean_cols": ["ANIMAL", "GROUP", "TIME_H", "INHIB_PCT"],
        "clean_animal": [str(v) for v in tidy_clean["ANIMAL"]],
        "clean_time": [float(v) for v in tidy_clean["TIME_H"]],
        "clean_value": [float(v) for v in tidy_clean["INHIB_PCT"]],
        "n_dup_rows": int(tidy_clean.duplicated().sum()),
        "n_unique_animals": int(wide["ANIMAL"].nunique()),
        "wide_index": [str(v) for v in wide_again.index],
        "wide_columns": [float(v) for v in wide_again.columns],
        "wide_values": wide_again.to_numpy().tolist(),
    }


CHECK_CODE = '''
def _part_a():
    tidy = _need("tidy")
    if not isinstance(tidy, _pd.DataFrame):
        return "`tidy` should be a pandas DataFrame, but it is a %s." % type(tidy).__name__

    cols = _cols_of(tidy)
    want = _EXP["tidy_cols"]
    missing = [c for c in want if c not in cols]
    if missing:
        leftover = [c for c in cols if c.startswith("T_")]
        if leftover:
            return ("`tidy` still has the timepoint columns %s, so it has not been reshaped "
                    "yet. melt turns those column names into the values of one new column."
                    % ", ".join(leftover[:3]))
        if "variable" in cols or "value" in cols:
            return ("`tidy` kept melt's default names `variable` and `value`. Pass var_name "
                    "and value_name so they come out as TIME_LABEL and INHIB_PCT.")
        return "`tidy` is missing the column(s) %s." % ", ".join(missing)
    if len(tidy) == _EXP["n_wide_rows"]:
        return ("`tidy` has the same number of rows as `wide`, so nothing was reshaped. One "
                "row per animal per timepoint means many more rows than that.")
    if len(tidy) != _EXP["n_tidy_rows"]:
        return ("`tidy` has %d rows. It should be one row per animal per timepoint, so the "
                "row count of `wide` times the number of timepoint columns." % len(tidy))
    labels = set(str(v) for v in tidy["TIME_LABEL"].unique())
    if len(labels) != _EXP["n_time_cols"]:
        return ("`TIME_LABEL` holds %d distinct values but there are %d timepoint columns. "
                "Only ANIMAL and GROUP should be id_vars." % (len(labels), _EXP["n_time_cols"]))
    if not all(v.startswith("T_") for v in labels):
        return "`TIME_LABEL` should still hold the original text column names, like T_0 and T_24."
    return None


def _part_b():
    tidy_clean = _need("tidy_clean")
    if not isinstance(tidy_clean, _pd.DataFrame):
        return "`tidy_clean` should be a pandas DataFrame, but it is a %s." % type(tidy_clean).__name__

    cols = _cols_of(tidy_clean)
    if "TIME_LABEL" in cols:
        return "`tidy_clean` still has the TIME_LABEL column. Drop it once TIME_H exists."
    missing = [c for c in _EXP["clean_cols"] if c not in cols]
    if missing:
        return "`tidy_clean` is missing the column(s) %s." % ", ".join(missing)
    if len(tidy_clean) != _EXP["n_tidy_rows"]:
        return ("`tidy_clean` has %d rows but part b should not add or remove any rows, only "
                "change and reorder them." % len(tidy_clean))

    if not _pd.api.types.is_numeric_dtype(tidy_clean["TIME_H"]):
        return ("`TIME_H` is not numeric, its dtype is %s. Strip the T_ prefix and then "
                "convert with .astype(float)." % tidy_clean["TIME_H"].dtype)

    if list(tidy_clean.index) != list(range(len(tidy_clean))):
        return ("`tidy_clean` has not had its index reset, so it does not run 0, 1, 2 and up. "
                "Use .reset_index(drop=True) after sorting.")

    got_animal = [str(v) for v in tidy_clean["ANIMAL"]]
    if got_animal != _EXP["clean_animal"]:
        if sorted(got_animal) == sorted(_EXP["clean_animal"]):
            return ("`tidy_clean` holds the right rows but not in the right order. Sort by "
                    "ANIMAL first and TIME_H second, in that order.")
        return "`tidy_clean` does not hold the expected set of animal rows."
    if not _arr_close(tidy_clean["TIME_H"].to_numpy(), _EXP["clean_time"]):
        ok_sorted = _arr_close(_np.sort(tidy_clean["TIME_H"].to_numpy()),
                               _np.sort(_np.asarray(_EXP["clean_time"], dtype=float)))
        if ok_sorted:
            return ("`TIME_H` has the right values but the rows are not in the right order. "
                    "Within each animal, time has to run upwards. Watch out for sorting the "
                    "text label rather than the number, which puts T_24 before T_4.")
        return ("`TIME_H` does not hold the right hours. T_0 should become 0.0 and T_24 "
                "should become 24.0.")
    if not _arr_close(tidy_clean["INHIB_PCT"].to_numpy(), _EXP["clean_value"]):
        return ("`INHIB_PCT` does not line up with the animal and time columns. Sort the whole "
                "frame at once rather than sorting single columns.")
    return None


def _part_c():
    n_dup_rows = _need("n_dup_rows")
    wide_again = _need("wide_again")

    val = _as_scalar(n_dup_rows)
    if val is None:
        return "`n_dup_rows` should be a single whole number."
    if int(round(val)) == 1:
        return ("`n_dup_rows` counts 1, which is the number of animals that got re-sent. The "
                "question asks for rows of tidy_clean, and that animal contributes one row "
                "per timepoint.")
    if int(round(val)) == _EXP["n_dup_rows"] * 2:
        return ("`n_dup_rows` is double what is wanted, so you counted both copies. "
                "df.duplicated() flags only the repeats, leaving the first occurrence alone, "
                "whereas keep=False flags both.")
    if int(round(val)) != _EXP["n_dup_rows"]:
        return ("`n_dup_rows` is not the number of repeated rows. Use "
                "tidy_clean.duplicated().sum() across all columns.")

    if not isinstance(wide_again, _pd.DataFrame):
        return "`wide_again` should be a pandas DataFrame, but it is a %s." % type(wide_again).__name__
    if list(wide_again.index.names) != ["ANIMAL"]:
        return ("`wide_again` should have ANIMAL as its index. Pass index='ANIMAL' to pivot "
                "and do not reset the index afterwards.")
    if len(wide_again) != _EXP["n_unique_animals"]:
        return ("`wide_again` has %d rows but there are %d distinct animals. Drop the repeated "
                "rows before pivoting." % (len(wide_again), _EXP["n_unique_animals"]))
    if [str(v) for v in wide_again.index] != _EXP["wide_index"]:
        return "`wide_again` does not have the expected animals on its index."
    if not _arr_close(_np.asarray([float(c) for c in wide_again.columns]), _EXP["wide_columns"]):
        return ("`wide_again` does not have the timepoints as its columns. Pass "
                "columns='TIME_H' and values='INHIB_PCT' to pivot.")
    if not _arr_close(wide_again.to_numpy(), _np.asarray(_EXP["wide_values"], dtype=float)):
        return ("`wide_again` has the right shape but the wrong cell values. Check that you "
                "pivoted INHIB_PCT, and that you dropped duplicates rather than aggregating "
                "them into a mean.")
    return None
'''
