PID = "W2D4"
WEEK = 2
DAY = 4
SLUG = "w2d4_pandas_crosstab_incidence"
TITLE = "Adverse events: counting events is not counting patients"
TOPIC = "pandas: crosstab, pivot_table, normalize, subject level incidence"
EST = "20-30 min"

SCENARIO_MD = '''
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
'''

CONCEPT_MD = '''
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
'''

TASK_MD = '''
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
'''

DATA_CODE = '''
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
'''

TOOLKIT_MD = '''
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
'''

TOOLKIT_CODE = '''
import numpy as np
import pandas as pd

# A tiny made up event log, plus who was enrolled. Nothing here is the answer.
demo_enrolled = pd.DataFrame({"S": ["s1", "s2", "s3", "s4"],
                              "G": ["ctl", "ctl", "trt", "trt"]})
demo_ae = pd.DataFrame({
    "S":   ["s1", "s1", "s1", "s3"],
    "G":   ["ctl", "ctl", "ctl", "trt"],
    "SEV": ["Mild", "Mild", "Severe", "Mild"],
})
print("demo_enrolled, 2 per group:");  print(demo_enrolled)
print("");  print("demo_ae, note s1 alone has three events and s2 and s4 have none:")
print(demo_ae)

print("")
print("--- 1. crosstab counts ROWS, so it counts events ---")
demo_counts = pd.crosstab(demo_ae["G"], demo_ae["SEV"])
print(demo_counts)
print("")
print("ctl looks worse than trt, but that is one subject with three events")

print("")
print("--- 2. normalize='index' makes each row sum to 1 ---")
print((pd.crosstab(demo_ae["G"], demo_ae["SEV"], normalize="index") * 100).round(1))
print("")
print("normalize='all' would instead make the WHOLE table sum to 1:")
print((pd.crosstab(demo_ae["G"], demo_ae["SEV"], normalize="all") * 100).round(1))

print("")
print("--- 3. size counts events, nunique counts people ---")
print("events per group :")
print(demo_ae.groupby("G")["S"].agg("size").to_string())
print("subjects with at least one event:")
print(demo_ae.groupby("G")["S"].agg("nunique").to_string())

print("")
print("--- 4. the denominator has to come from the enrolment table ---")
demo_denom = demo_enrolled.groupby("G")["S"].nunique()
demo_numer = demo_ae.groupby("G")["S"].nunique()
print("dosed per group       :", demo_denom.to_dict())
print("with an event         :", demo_numer.to_dict())
demo_inc = pd.merge(demo_denom.reset_index(name="N_DOSED"),
                    demo_numer.reset_index(name="N_WITH_AE"), on="G", how="left")
demo_inc["N_WITH_AE"] = demo_inc["N_WITH_AE"].fillna(0).astype(int)
demo_inc["PCT"] = (100.0 * demo_inc["N_WITH_AE"] / demo_inc["N_DOSED"]).round(1)
print("")
print(demo_inc)
print("")
print("if you had taken the denominator from the event log instead, every group")
print("would read 100 percent, because subjects with no events are not in it")
'''

PARTS = [
    dict(
        label="a",
        prompt_md='''
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
''',
        starter='''
# Part a. Event counts by group and severity, with totals.

ae_counts = None   # YOUR CODE

print(ae_counts)
''',
        solution='''
# Part a. Event counts by group and severity, with totals.

ae_counts = pd.crosstab(ae["GROUP"], ae["SEVERITY"],
                        margins=True, margins_name="Total")
ae_counts = ae_counts.reindex(GROUP_ORDER + ["Total"])

print(ae_counts)
''',
    ),
    dict(
        label="b",
        prompt_md='''
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
''',
        starter='''
# Part b. Row percentages: of each group's events, what share was each severity.

ae_pct = None   # YOUR CODE

print(ae_pct)
if ae_pct is not None:
    print("")
    print("row sums:", ae_pct.sum(axis=1).to_dict())
''',
        solution='''
# Part b. Row percentages: of each group's events, what share was each severity.

ae_pct = pd.crosstab(ae["GROUP"], ae["SEVERITY"], normalize="index") * 100.0
ae_pct = ae_pct.reindex(GROUP_ORDER).round(1)

print(ae_pct)
if ae_pct is not None:
    print("")
    print("row sums:", ae_pct.sum(axis=1).to_dict())
''',
    ),
    dict(
        label="c",
        prompt_md='''
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
''',
        starter='''
# Part c. Incidence: subjects with at least one event, over subjects dosed.

incidence = None   # YOUR CODE

print(incidence)
''',
        solution='''
# Part c. Incidence: subjects with at least one event, over subjects dosed.

_denom = enrolled.groupby("GROUP")["SUBJECT"].nunique().reset_index(name="N_DOSED")
_numer = ae.groupby("GROUP")["SUBJECT"].nunique().reset_index(name="N_WITH_AE")

incidence = pd.merge(_denom, _numer, on="GROUP", how="left")
incidence["N_WITH_AE"] = incidence["N_WITH_AE"].fillna(0).astype(int)
incidence["PCT_WITH_AE"] = (100.0 * incidence["N_WITH_AE"] / incidence["N_DOSED"]).round(1)

incidence["_ord"] = incidence["GROUP"].map({g: i for i, g in enumerate(GROUP_ORDER)})
incidence = (incidence.sort_values("_ord")
             .drop(columns=["_ord"])
             .reset_index(drop=True))

print(incidence)
''',
    ),
]


def build_expected(ns):
    import numpy as np
    import pandas as pd
    ae, enrolled = ns["ae"], ns["enrolled"]
    order = ns["GROUP_ORDER"]
    sev = sorted(ae["SEVERITY"].unique())

    counts = pd.crosstab(ae["GROUP"], ae["SEVERITY"], margins=True,
                         margins_name="Total").reindex(order + ["Total"])
    pct = (pd.crosstab(ae["GROUP"], ae["SEVERITY"], normalize="index") * 100.0
           ).reindex(order).round(1)
    pct_cols = (pd.crosstab(ae["GROUP"], ae["SEVERITY"], normalize="columns") * 100.0
                ).reindex(order).round(1)
    pct_all = (pd.crosstab(ae["GROUP"], ae["SEVERITY"], normalize="all") * 100.0
               ).reindex(order).round(1)

    inc = ns["incidence"]
    wrong_denom = (ae.groupby("GROUP")["SUBJECT"].nunique().reindex(order)).to_numpy()
    event_numer = (ae.groupby("GROUP")["SUBJECT"].size().reindex(order).fillna(0)).to_numpy()
    dosed = (enrolled.groupby("GROUP")["SUBJECT"].nunique().reindex(order)).to_numpy()
    return {
        "severities": [str(s) for s in sev],
        "group_order": [str(g) for g in order],
        "counts_index": [str(i) for i in counts.index],
        "counts": {str(c): [int(v) for v in counts[c]] for c in counts.columns},
        "n_events": int(len(ae)),
        "pct": {str(c): [float(v) for v in pct[c]] for c in pct.columns},
        "pct_cols": {str(c): [float(v) for v in pct_cols[c]] for c in pct_cols.columns},
        "pct_all": {str(c): [float(v) for v in pct_all[c]] for c in pct_all.columns},
        "inc_cols": ["GROUP", "N_DOSED", "N_WITH_AE", "PCT_WITH_AE"],
        "inc_group": [str(v) for v in inc["GROUP"]],
        "inc_dosed": [int(v) for v in dosed],
        "inc_with": [int(v) for v in inc["N_WITH_AE"]],
        "inc_pct": [float(v) for v in inc["PCT_WITH_AE"]],
        "inc_pct_wrong_denom": [float(v) for v in
                                np.round(100.0 * inc["N_WITH_AE"].to_numpy() / wrong_denom, 1)],
        "inc_events_numer": [int(v) for v in event_numer],
    }


CHECK_CODE = '''
def _table_cols(obj):
    try:
        return [str(c) for c in obj.columns]
    except Exception:
        return None


def _part_a():
    ae_counts = _need("ae_counts")
    if not isinstance(ae_counts, _pd.DataFrame):
        return "`ae_counts` should be a DataFrame, which is what pd.crosstab returns."

    idx = [str(i) for i in ae_counts.index]
    want_idx = _EXP["counts_index"]
    if "Total" not in idx:
        return ("`ae_counts` has no Total row. Pass margins=True and "
                "margins_name='Total' to crosstab.")
    if idx != want_idx:
        if sorted(idx) == sorted(want_idx):
            return ("`ae_counts` has the right rows in the wrong order. Reindex to "
                    "GROUP_ORDER plus 'Total' so it reads Placebo, Low, High.")
        return "`ae_counts` does not have one row per dose group plus a Total row."

    cols = _table_cols(ae_counts) or []
    for sev in _EXP["severities"]:
        if sev not in cols:
            return "`ae_counts` has no %s column. Put SEVERITY across the columns." % sev
    if "Total" not in cols:
        return "`ae_counts` has no Total column. margins=True adds both the row and the column."

    for col, want in _EXP["counts"].items():
        if col not in cols:
            continue
        if not _arr_close(ae_counts[col].to_numpy(), want):
            return ("`ae_counts` has the right shape but the %s column does not match the "
                    "event counts. crosstab counts rows of ae, nothing else." % col)
    return None


def _part_b():
    ae_pct = _need("ae_pct")
    if not isinstance(ae_pct, _pd.DataFrame):
        return "`ae_pct` should be a DataFrame."

    idx = [str(i) for i in ae_pct.index]
    if "Total" in idx:
        return "`ae_pct` should not have a Total row. Drop margins for this one."
    if idx != _EXP["group_order"]:
        if sorted(idx) == sorted(_EXP["group_order"]):
            return "`ae_pct` rows are not in the dose order Placebo, Low, High. Reindex them."
        return "`ae_pct` does not have one row per dose group."

    cols = _table_cols(ae_pct) or []
    for sev in _EXP["severities"]:
        if sev not in cols:
            return "`ae_pct` has no %s column." % sev

    row_sums = ae_pct[[c for c in _EXP["severities"]]].sum(axis=1).to_numpy()
    if _np.allclose(row_sums, 1.0, atol=0.02):
        return ("`ae_pct` is still a fraction, since the rows sum to 1. Multiply by 100 to "
                "get percent.")
    first = _EXP["severities"][0]
    if _arr_close(ae_pct[first].to_numpy(), _EXP["pct_cols"][first]):
        return ("`ae_pct` was normalised down the columns, so it answers what share of all "
                "the severe events came from each group. The task asks for each group's "
                "events split across severities, which is normalize='index'.")
    if _arr_close(ae_pct[first].to_numpy(), _EXP["pct_all"][first]):
        return ("`ae_pct` was normalised over the whole table, so everything sums to 100 "
                "once. Each ROW should sum to 100, which is normalize='index'.")
    if not _np.allclose(row_sums, 100.0, atol=0.3):
        return ("`ae_pct` rows do not sum to 100. Each dose group's events should be split "
                "across the severity columns.")
    for col, want in _EXP["pct"].items():
        if col in cols and not _arr_close(ae_pct[col].to_numpy(), want, rtol=1e-2, atol=0.15):
            return "`ae_pct` rows sum to 100 but the %s column does not match." % col
    return None


def _part_c():
    inc = _need("incidence")
    if not isinstance(inc, _pd.DataFrame):
        return "`incidence` should be a pandas DataFrame."
    cols = _cols_of(inc) or []
    missing = [c for c in _EXP["inc_cols"] if c not in cols]
    if missing:
        return "`incidence` is missing the column(s) %s." % ", ".join(missing)
    if len(inc) != len(_EXP["group_order"]):
        return ("`incidence` has %d rows but there are %d dose groups."
                % (len(inc), len(_EXP["group_order"])))
    if [str(v) for v in inc["GROUP"]] != _EXP["inc_group"]:
        return ("`incidence` is not in the dose order Placebo, Low, High. Sort it into "
                "GROUP_ORDER rather than leaving it alphabetical.")

    dosed = inc["N_DOSED"].to_numpy()
    if _arr_close(dosed, _EXP["inc_with"]) and not _arr_close(dosed, _EXP["inc_dosed"]):
        return ("`N_DOSED` was taken from the event log, so it counts only subjects who had "
                "an event. The denominator has to come from `enrolled`, which knows about "
                "everyone who was dosed.")
    if not _arr_close(dosed, _EXP["inc_dosed"]):
        return "`N_DOSED` is not the number of subjects dosed in each group, from enrolled."

    with_ae = inc["N_WITH_AE"].to_numpy()
    if _arr_close(with_ae, _EXP["inc_events_numer"]) and not _arr_close(with_ae, _EXP["inc_with"]):
        return ("`N_WITH_AE` counted EVENTS, not subjects. A subject with several events "
                "should count once, so use nunique on the subject column rather than size.")
    if not _arr_close(with_ae, _EXP["inc_with"]):
        return "`N_WITH_AE` is not the number of distinct subjects with at least one event."

    pct = inc["PCT_WITH_AE"].to_numpy()
    if _np.allclose(pct, 100.0, atol=0.01):
        return ("`PCT_WITH_AE` is 100 in every group, which is the giveaway that the "
                "denominator came from the event log.")
    if _arr_close(pct, _EXP["inc_pct_wrong_denom"]) and not _arr_close(pct, _EXP["inc_pct"]):
        return "`PCT_WITH_AE` used the wrong denominator. It should be N_WITH_AE over N_DOSED."
    if not _arr_close(pct, _EXP["inc_pct"], rtol=1e-3, atol=0.06):
        return ("`PCT_WITH_AE` is not 100 * N_WITH_AE / N_DOSED, rounded to 1 decimal "
                "place.")
    return None
'''

# Functions this problem introduces. Keys must exist in tools/function_catalog.py.
FUNCTIONS = [
    "pd.crosstab",
    "pd.pivot_table",
    "size versus nunique",
    ".reindex",
    ".map",
]
