PID = "W8D4"
WEEK = 8
DAY = 4
SLUG = "w8d4_nonmem_dataset"
TITLE = "Building a NONMEM ready dataset, and auditing it"
TOPIC = "Population PK: event records, EVID and MDV, BLQ handling, per subject NCA with groupby"
EST = "25-35 min"

SCENARIO_MD = '''
## Scenario

The modeller needs the phase 1 data as a NONMEM dataset by Friday. What you
have is what you always have: a sample table from the bioanalytical lab, a
dosing record from the pharmacy, and a demographics table from the clinical
database, all keyed on subject number and none of them in the shape the model
needs.

This is the job. Not glamorous, entirely unavoidable, and the place where
real analyses go wrong. A dataset with the dose record sorted after the first
observation runs without complaint and produces estimates that are wrong, and
nothing in the output says so.

Today you build one, audit it, and then do the per subject NCA that the
modeller will compare their fit against.
'''

CONCEPT_MD = '''
## Event records, and the four columns that carry the logic

A NONMEM dataset is not a table of observations. It is a **time ordered list of
events** for each subject, where an event is either something given or
something measured. One row, one event.

The required columns, with the conventional names:

| Column | Meaning |
|---|---|
| `ID` | subject identifier, grouping the rows |
| `TIME` | hours since the subject's first event |
| `AMT` | amount given on a dosing row, 0 on an observation row |
| `DV` | the dependent variable, the measurement, 0 on a dosing row |
| `EVID` | event ID: 1 for a dose, 0 for an observation |
| `MDV` | missing dependent variable: 1 if there is no usable measurement |
| `CMT` | which compartment the event applies to |

**`EVID` is the switch.** `EVID = 1` means give `AMT` into compartment `CMT`
and ignore `DV`. `EVID = 0` means compare `DV` with the model's prediction.
Other values exist, 2 for other type events and 3 for a reset, and you will
meet them when you meet a study design that needs them.

**`MDV = 1` means do not use this `DV`.** Dosing rows always have `MDV = 1`,
because there is nothing to fit there. Observation rows get `MDV = 1` when the
sample is missing, lost, or below the limit of quantification.

**Sort order is part of the data.** Rows are processed in the order they appear,
so they must be sorted by `ID` then `TIME`, and **at identical times the dose
has to come first**. A pre dose sample and the dose are both at time zero, and
if the sample is processed second it will be compared against a prediction that
already includes the dose. The fix is to sort with `EVID` descending within
`TIME`:

```
df.sort_values(["ID", "TIME", "EVID"], ascending=[True, True, False])
```

**BLQ, and why it is handled twice.** A sample below the LLOQ is not a
concentration of zero and not a missing value either, it is the information
that the true value lies somewhere between zero and the LLOQ. There are
several conventions:

- **M1**, drop the record, or equivalently set `MDV = 1`. The default, and what
  you will do today,
- **M3**, keep it and treat it as censored, which is more correct and needs a
  likelihood that NONMEM can do and `curve_fit` cannot,
- substitute `LLOQ / 2`, common in NCA, defensible, biased,
- substitute zero, which is what NCA does for a pre dose sample after an oral
  dose, where zero is not a substitution but the truth.

Note the consequence: the same record can be `MDV = 1` for the model fit and
treated as a hard zero in the NCA of the same dataset. That is not an
inconsistency, it is two analyses answering different questions, but it does
need writing down in the analysis plan.
'''

TASK_MD = '''
## Your task

Three parts, worked one at a time.

| Part | What you work out | Variables to fill in |
|---|---|---|
| a | assemble the event records in the right order | `nm`, `n_rows`, `n_dose_rows`, `first_evid_per_id` |
| b | audit it before anybody fits anything | `n_blq`, `n_mdv`, `records_per_id`, `time_monotone` |
| c | per subject NCA, merged with the dosing and demographics | `nca`, `auc_dn_low`, `auc_dn_high`, `auc_dn_ratio` |

Below this you will find the **Setup** cell, then a **Toolkit** cell that
demonstrates every move on a tiny made up example, then **Part a**, **Part b**
and **Part c**, each with its own explanation and a cell to write in.
'''

DATA_CODE = '''
import numpy as np
import pandas as pd

PROBLEM_ID = "W8D4"
TOPIC = "Population PK: event records, EVID and MDV, BLQ handling, per subject NCA with groupby"

pd.set_option("display.width", 120)

N_SUBJ = 8
LLOQ = 0.05          # mg/L, lower limit of quantification
_KA = 1.0
_CL_POP = 4.2
_V_POP = 35.0

_doses = np.array([50.0] * 4 + [150.0] * 4)
_times = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0])

_rng = np.random.default_rng(8426)
_cl = _CL_POP * _rng.lognormal(0.0, 0.30, N_SUBJ)
_v = _V_POP * _rng.lognormal(0.0, 0.25, N_SUBJ)
_wt = np.round(70.0 * _rng.lognormal(0.0, 0.15, N_SUBJ), 1)
_age = _rng.integers(21, 65, N_SUBJ)
_sex = _rng.integers(0, 2, N_SUBJ)

_rows = []
for _i in range(N_SUBJ):
    _k = _cl[_i] / _v[_i]
    _c = (_doses[_i] * _KA / (_v[_i] * (_KA - _k))
          * (np.exp(-_k * _times) - np.exp(-_KA * _times)))
    _c = np.round(np.maximum(_c * _rng.lognormal(0.0, 0.08, _times.size), 0.0), 4)
    for _t, _cc in zip(_times, _c):
        _rows.append((_i + 1, float(_t), float(_cc)))

# Three source tables, as they would arrive.
samples = pd.DataFrame(_rows, columns=["ID", "TIME", "CONC"])
dosing = pd.DataFrame({"ID": np.arange(1, N_SUBJ + 1), "AMT": _doses})
demog = pd.DataFrame({"ID": np.arange(1, N_SUBJ + 1), "WT_KG": _wt,
                      "AGE_Y": _age, "SEX": _sex})

NM_COLS = ["ID", "TIME", "AMT", "DV", "EVID", "MDV", "CMT"]

print("samples: %d rows" % len(samples))
print(samples.head(10).to_string(index=False))
print("")
print("dosing:")
print(dosing.to_string(index=False))
print("")
print("demog:")
print(demog.to_string(index=False))
print("")
print("LLOQ = %.2f mg/L, and %d of the %d samples are below it"
      % (LLOQ, int((samples["CONC"] < LLOQ).sum()), len(samples)))
print("every dose is given at time 0, and every subject has a pre dose sample")
'''

TOOLKIT_MD = '''
## Toolkit for this problem

| What you want to do | How |
|---|---|
| add constant columns | `df.assign(EVID=1, MDV=1)` |
| a column from a condition | `np.where(df["CONC"] < LLOQ, 1, 0)` |
| stack two tables | `pd.concat([a, b], ignore_index=True)` |
| choose and order columns | `df[NM_COLS]` |
| sort, with one column descending | `df.sort_values([...], ascending=[True, True, False])` |
| renumber the index afterwards | `.reset_index(drop=True)` |
| rows per group | `df.groupby("ID").size()` |
| the first row of each group | `df.groupby("ID").first()` |
| is a series sorted | `s.is_monotonic_increasing` |
| a per group calculation returning several numbers | `df.groupby("ID")[["TIME", "DV"]].apply(fn)` |
| join a lookup table | `left.merge(right, on="ID", validate="many_to_one")` |

The `groupby("ID")[["TIME", "DV"]].apply(fn)` form is the one to use. Selecting
the columns you need before `apply` keeps the group key out of the sub frame,
which is what newer pandas wants and older pandas tolerates, so the same line
works everywhere.
'''

TOOLKIT_CODE = '''
import numpy as np
import pandas as pd

demo_obs = pd.DataFrame({"ID": [1, 1, 1, 2, 2, 2],
                         "TIME": [0.0, 1.0, 4.0, 0.0, 1.0, 4.0],
                         "CONC": [0.0, 2.0, 0.9, 0.0, 3.0, 1.4]})
demo_dose = pd.DataFrame({"ID": [1, 2], "AMT": [10.0, 20.0]})
demo_lloq = 0.1
demo_cols = ["ID", "TIME", "AMT", "DV", "EVID", "MDV", "CMT"]

print("--- 1. assign makes the constant columns ---")
demo_d = demo_dose.assign(TIME=0.0, DV=0.0, EVID=1, MDV=1, CMT=1)[demo_cols]
print(demo_d.to_string(index=False))

print("")
print("--- 2. np.where makes the conditional ones ---")
demo_o = demo_obs.assign(AMT=0.0, DV=demo_obs["CONC"], EVID=0, CMT=2,
                         MDV=np.where(demo_obs["CONC"] < demo_lloq, 1, 0))[demo_cols]
print(demo_o.to_string(index=False))

print("")
print("--- 3. concat then sort, and watch the tie at time 0 ---")
demo_bad = pd.concat([demo_d, demo_o], ignore_index=True).sort_values(["ID", "TIME"])
print("sorted by ID and TIME only, subject 1:")
print(demo_bad[demo_bad["ID"] == 1].to_string(index=False))
demo_nm = (pd.concat([demo_d, demo_o], ignore_index=True)
           .sort_values(["ID", "TIME", "EVID"], ascending=[True, True, False])
           .reset_index(drop=True))
print("with EVID descending inside TIME, subject 1:")
print(demo_nm[demo_nm["ID"] == 1].to_string(index=False))
print("the dose is now first, which is the only order that means what you intend")

print("")
print("--- 4. per group counts and checks ---")
print("rows per subject:")
print(demo_nm.groupby("ID").size().to_string())
print("first EVID per subject:")
print(demo_nm.groupby("ID")["EVID"].first().to_string())
print("TIME non decreasing within every subject:",
      bool(demo_nm.groupby("ID")["TIME"].apply(lambda s: s.is_monotonic_increasing).all()))

print("")
print("--- 5. a per group calculation that returns several numbers ---")


def demo_summary(g):
    g = g.sort_values("TIME")
    t = g["TIME"].values
    c = g["DV"].values
    i = int(np.argmax(c))
    area = float(np.sum(0.5 * (c[1:] + c[:-1]) * np.diff(t)))
    return pd.Series({"CMAX": float(c[i]), "TMAX": float(t[i]), "AUC": area})


demo_only_obs = demo_nm[demo_nm["EVID"] == 0]
demo_res = demo_only_obs.groupby("ID")[["TIME", "DV"]].apply(demo_summary).reset_index()
print(demo_res.to_string(index=False))

print("")
print("--- 6. merge with validate, which catches duplicate keys ---")
print(demo_res.merge(demo_dose, on="ID", validate="one_to_one").to_string(index=False))
demo_dupe = pd.concat([demo_dose, demo_dose.head(1)], ignore_index=True)
try:
    demo_res.merge(demo_dupe, on="ID", validate="one_to_one")
except Exception as err:
    print("a duplicated key is caught:", str(err)[:70])
print("without validate, that merge silently returns extra rows")
'''

PARTS = [
    dict(
        label="a",
        prompt_md='''
## Part a. Assemble the event records

Build `nm`, the NONMEM dataset, from `samples` and `dosing`.

The dosing rows: one per subject, `TIME` 0, `AMT` from `dosing`, `DV` 0,
`EVID` 1, `MDV` 1, `CMT` 1.

The observation rows: one per sample, `AMT` 0, `DV` the concentration, `EVID`
0, `CMT` 2, and `MDV` 1 where the concentration is below `LLOQ` and 0
otherwise.

Then concatenate them, select the columns in the order `NM_COLS`, and sort by
`ID`, `TIME`, and `EVID` **descending**, so that the dose comes before the pre
dose sample at time zero. Reset the index.

Record `n_rows`, the total number of rows, `n_dose_rows`, how many have
`EVID` of 1, and `first_evid_per_id`, the `EVID` of the first row of each
subject as an array of eight numbers. That last one is the check that the sort
worked: every entry must be 1. If any subject starts with a 0, that subject's
first observation is being compared against a model that has not been dosed
yet.

Print the first eleven rows when you have it. You should see the dose for
subject 1, then the pre dose sample with `MDV` of 1, then seven usable
observations, then subject 2's dose.

Fill in `nm`, `n_rows`, `n_dose_rows`, `first_evid_per_id`.

Moves: `assign`, `np.where`, `pd.concat`, column selection, `sort_values`
with mixed `ascending`, `reset_index`, `groupby(...).first()`.
''',
        starter='''
# Part a. Build the dataset.

dose_rows = None  # YOUR CODE: one row per subject, EVID 1, from `dosing`
obs_rows = None   # YOUR CODE: one row per sample, EVID 0, from `samples`

nm = None         # YOUR CODE: concat, select NM_COLS, sort, reset_index

n_rows = None             # YOUR CODE: total rows
n_dose_rows = None        # YOUR CODE: rows with EVID == 1
first_evid_per_id = None  # YOUR CODE: EVID of the first row of each subject

print("rows: %r, of which dosing records: %r" % (n_rows, n_dose_rows))
print("first EVID per subject:", first_evid_per_id)
if nm is not None:
    print("")
    print(nm.head(11).to_string(index=False))
''',
        solution='''
# Part a. Build the dataset.

dose_rows = dosing.assign(TIME=0.0, DV=0.0, EVID=1, MDV=1, CMT=1)[NM_COLS]
obs_rows = samples.assign(AMT=0.0, DV=samples["CONC"], EVID=0, CMT=2,
                          MDV=np.where(samples["CONC"] < LLOQ, 1, 0))[NM_COLS]

nm = (pd.concat([dose_rows, obs_rows], ignore_index=True)
      .sort_values(["ID", "TIME", "EVID"], ascending=[True, True, False])
      .reset_index(drop=True))

n_rows = int(len(nm))
n_dose_rows = int((nm["EVID"] == 1).sum())
first_evid_per_id = nm.groupby("ID")["EVID"].first().values

print("rows: %r, of which dosing records: %r" % (n_rows, n_dose_rows))
print("first EVID per subject:", first_evid_per_id)
if nm is not None:
    print("")
    print(nm.head(11).to_string(index=False))
''',
    ),
    dict(
        label="b",
        prompt_md='''
## Part b. Audit it

Four checks, each one line, each one catching a real class of error.

- `n_blq`, how many observation rows are below the LLOQ. Count rows with
  `EVID` of 0 and `MDV` of 1,
- `n_mdv`, how many rows in total have `MDV` of 1. This should be `n_blq` plus
  the eight dosing rows,
- `records_per_id`, the number of rows for each subject, as an array of eight.
  All eight should be 9: one dose and eight samples,
- `time_monotone`, True if `TIME` is non decreasing within every subject.

You should find ten BLQ records and eighteen `MDV` rows. Eight of the ten are
the pre dose samples, which are below the limit because the drug has not been
given yet. The other two are subjects 3 and 4 at 24 hours, who cleared the
50 mg dose fast enough to fall under the limit by the last sample. Those two
are the ones to think about: a genuine measurement attempt that came back with
no number, which is where the choice of BLQ method starts to matter.

The counts are not busy work. A subject with eight rows instead of nine has
lost a sample somewhere in the merge. A subject whose rows are not time ordered
will produce quiet nonsense. Both have happened in real submissions.

Fill in `n_blq`, `n_mdv`, `records_per_id`, `time_monotone`.

Moves: boolean masks and `.sum()`, `groupby(...).size()`,
`is_monotonic_increasing`.
''',
        starter='''
# Part b. Audit the dataset.

n_blq = None           # YOUR CODE: observation rows below the LLOQ
n_mdv = None           # YOUR CODE: all rows with MDV == 1
records_per_id = None  # YOUR CODE: rows per subject, as an array of 8
time_monotone = None   # YOUR CODE: True if TIME never decreases within a subject

print("observation rows below the LLOQ: %r" % (n_blq,))
print("rows with MDV = 1             : %r" % (n_mdv,))
print("rows per subject              :", records_per_id)
print("TIME non decreasing in every subject:", time_monotone)
if nm is not None:
    print("")
    print("the BLQ observation rows:")
    print(nm[(nm["EVID"] == 0) & (nm["MDV"] == 1)].to_string(index=False))
''',
        solution='''
# Part b. Audit the dataset.

n_blq = int(((nm["EVID"] == 0) & (nm["MDV"] == 1)).sum())
n_mdv = int((nm["MDV"] == 1).sum())
records_per_id = nm.groupby("ID").size().values
time_monotone = bool(nm.groupby("ID")["TIME"]
                     .apply(lambda s: s.is_monotonic_increasing).all())

print("observation rows below the LLOQ: %r" % (n_blq,))
print("rows with MDV = 1             : %r" % (n_mdv,))
print("rows per subject              :", records_per_id)
print("TIME non decreasing in every subject:", time_monotone)
if nm is not None:
    print("")
    print("the BLQ observation rows:")
    print(nm[(nm["EVID"] == 0) & (nm["MDV"] == 1)].to_string(index=False))
''',
    ),
    dict(
        label="c",
        prompt_md='''
## Part c. Per subject NCA, and a dose proportionality trap

Now produce the NCA summary the modeller will compare their fit against.

Work from the observation rows only, `EVID` of 0, and treat the `DV` column as
it stands, so BLQ records count as zero. For a pre dose sample after an oral
dose that is exactly right. For the two 24 hour samples it is the zero
substitution convention, which here costs almost nothing because the values
were under 0.03 mg/L, and which you would record in the analysis plan.

Write a function that takes a sub frame of `TIME` and `DV` for one subject and
returns a Series with `CMAX`, `TMAX` and `AUC`, the last by the linear
trapezoid rule over the samples as they are. Apply it with
`groupby("ID")[["TIME", "DV"]].apply(...)`, reset the index, then merge on the
dosing table and the demographics, using `validate="one_to_one"` so that a
duplicated key would raise rather than silently duplicate rows.

Add a dose normalised column, `AUC` divided by `AMT`, and call the result
`nca`. Then compare the two dose groups:

- `auc_dn_low`, the geometric mean dose normalised AUC in the 50 mg group,
- `auc_dn_high`, the same in the 150 mg group,
- `auc_dn_ratio`, high divided by low.

The ratio comes out at 1.20. Taken at face value that says exposure rises
faster than dose, by twenty percent over a three fold dose range, which is the
kind of finding that gets a slide of its own and a discussion about saturable
first pass metabolism.

It is an artifact, and here you can prove it, because the data were simulated
with exposure exactly proportional to dose in every subject: `AUC` to infinity
is `DOSE / CL` by construction. What differs between the groups is the people.
The four subjects who got 50 mg have a geometric mean clearance of 4.71 L/h and
the four who got 150 mg have 3.66 L/h, 29 percent lower, because with four
subjects per group that is well within the luck of the draw. The dose had
nothing to do with it.

This is why dose proportionality is assessed within subjects, in a crossover,
whenever it can be, and when it cannot, with a power model on the log scale and
a confidence interval wide enough to be honest about eight people.

Fill in `nca`, `auc_dn_low`, `auc_dn_high`, `auc_dn_ratio`.

Moves: `groupby(...)[[cols]].apply`, `reset_index`, `merge` with `validate`,
geometric means.
''',
        starter='''
# Part c. NCA per subject, and the dose groups compared.

def nca_one(g):
    """CMAX, TMAX and linear trapezoid AUC for one subject."""
    g = g.sort_values("TIME")
    # YOUR CODE: pull out TIME and DV, find the peak, integrate
    return None


nca = None  # YOUR CODE: the per subject table, merged with dosing and demog,
            #            with an AUC_DN column

auc_dn_low = None    # YOUR CODE: geometric mean AUC_DN in the 50 mg group
auc_dn_high = None   # YOUR CODE: geometric mean AUC_DN in the 150 mg group
auc_dn_ratio = None  # YOUR CODE: high divided by low

if nca is not None:
    print(nca.to_string(index=False))
print("")
print("dose normalised AUC, geometric mean")
print("  50 mg group : %r" % (auc_dn_low,))
print(" 150 mg group : %r" % (auc_dn_high,))
print("  ratio       : %r" % (auc_dn_ratio,))
''',
        solution='''
# Part c. NCA per subject, and the dose groups compared.

def nca_one(g):
    """CMAX, TMAX and linear trapezoid AUC for one subject."""
    g = g.sort_values("TIME")
    t = g["TIME"].values
    c = g["DV"].values
    i = int(np.argmax(c))
    area = float(np.sum(0.5 * (c[1:] + c[:-1]) * np.diff(t)))
    return pd.Series({"CMAX": float(c[i]), "TMAX": float(t[i]), "AUC": area})


_obs = nm[nm["EVID"] == 0]
nca = _obs.groupby("ID")[["TIME", "DV"]].apply(nca_one).reset_index()
nca = (nca.merge(dosing, on="ID", validate="one_to_one")
          .merge(demog, on="ID", validate="one_to_one"))
nca["AUC_DN"] = nca["AUC"] / nca["AMT"]

_low = nca.loc[nca["AMT"] == 50.0, "AUC_DN"].values
_high = nca.loc[nca["AMT"] == 150.0, "AUC_DN"].values
auc_dn_low = float(np.exp(np.mean(np.log(_low))))
auc_dn_high = float(np.exp(np.mean(np.log(_high))))
auc_dn_ratio = auc_dn_high / auc_dn_low

if nca is not None:
    print(nca.to_string(index=False))
print("")
print("dose normalised AUC, geometric mean")
print("  50 mg group : %r" % (auc_dn_low,))
print(" 150 mg group : %r" % (auc_dn_high,))
print("  ratio       : %r" % (auc_dn_ratio,))
''',
    ),
]


def build_expected(ns):
    import numpy as np
    import pandas as pd

    samples = ns["samples"]
    dosing = ns["dosing"]
    demog = ns["demog"]
    lloq = float(ns["LLOQ"])
    cols = list(ns["NM_COLS"])

    dose_rows = dosing.assign(TIME=0.0, DV=0.0, EVID=1, MDV=1, CMT=1)[cols]
    obs_rows = samples.assign(AMT=0.0, DV=samples["CONC"], EVID=0, CMT=2,
                              MDV=np.where(samples["CONC"] < lloq, 1, 0))[cols]
    nm = (pd.concat([dose_rows, obs_rows], ignore_index=True)
          .sort_values(["ID", "TIME", "EVID"], ascending=[True, True, False])
          .reset_index(drop=True))
    bad_sort = (pd.concat([dose_rows, obs_rows], ignore_index=True)
                .sort_values(["ID", "TIME", "EVID"])
                .reset_index(drop=True))

    obs = nm[nm["EVID"] == 0]

    def one(g):
        g = g.sort_values("TIME")
        t = g["TIME"].values
        c = g["DV"].values
        i = int(np.argmax(c))
        return pd.Series({"CMAX": float(c[i]), "TMAX": float(t[i]),
                          "AUC": float(np.sum(0.5 * (c[1:] + c[:-1]) * np.diff(t)))})

    nca = obs.groupby("ID")[["TIME", "DV"]].apply(one).reset_index()
    nca = nca.merge(dosing, on="ID", validate="one_to_one")
    nca["AUC_DN"] = nca["AUC"] / nca["AMT"]
    low = nca.loc[nca["AMT"] == 50.0, "AUC_DN"].values
    high = nca.loc[nca["AMT"] == 150.0, "AUC_DN"].values
    dn_low = float(np.exp(np.mean(np.log(low))))
    dn_high = float(np.exp(np.mean(np.log(high))))

    return {
        "n_subj": int(ns["N_SUBJ"]),
        "cols": cols,
        "n_rows": int(len(nm)),
        "n_dose_rows": int((nm["EVID"] == 1).sum()),
        "first_evid_per_id": [int(x) for x in nm.groupby("ID")["EVID"].first().values],
        "first_evid_bad_sort": [int(x) for x in
                                bad_sort.groupby("ID")["EVID"].first().values],
        "dv_head": [float(x) for x in nm["DV"].values[:11]],
        "evid_head": [int(x) for x in nm["EVID"].values[:11]],
        "n_blq": int(((nm["EVID"] == 0) & (nm["MDV"] == 1)).sum()),
        "n_mdv": int((nm["MDV"] == 1).sum()),
        "records_per_id": [int(x) for x in nm.groupby("ID").size().values],
        "cmax": [float(x) for x in nca["CMAX"].values],
        "tmax": [float(x) for x in nca["TMAX"].values],
        "auc": [float(x) for x in nca["AUC"].values],
        "auc_dn": [float(x) for x in nca["AUC_DN"].values],
        "auc_dn_low": dn_low,
        "auc_dn_high": dn_high,
        "auc_dn_ratio": dn_high / dn_low,
        "auc_dn_mean_low": float(np.mean(low)),
    }


CHECK_CODE = '''

def _part_a():
    nm_df = _need("nm")
    n_rows = _need("n_rows")
    n_dose = _need("n_dose_rows")
    first_evid = _need("first_evid_per_id")

    cols = _cols_of(nm_df)
    if cols is None:
        return "`nm` should be a pandas DataFrame."
    if cols != _EXP["cols"]:
        missing = [c for c in _EXP["cols"] if c not in cols]
        if missing:
            return ("`nm` is missing the column(s) %s. The dataset needs exactly %s."
                    % (missing, _EXP["cols"]))
        return ("`nm` has the right columns in the wrong order: %s. Select them as "
                "NM_COLS so the file reads the way a modeller expects." % cols)
    if len(nm_df) != _EXP["n_rows"]:
        return ("`nm` has %d rows but it should have %d: one dosing record per subject "
                "plus every sample." % (len(nm_df), _EXP["n_rows"]))

    got = _as_scalar(n_rows)
    if got is None or int(got) != _EXP["n_rows"]:
        return "`n_rows` should be the number of rows in `nm`, which is %d." % _EXP["n_rows"]
    got = _as_scalar(n_dose)
    if got is None or int(got) != _EXP["n_dose_rows"]:
        return ("`n_dose_rows` should be the number of rows with EVID == 1, which is one "
                "per subject.")

    arr = _as_array(first_evid)
    if arr is None:
        return "`first_evid_per_id` should be an array of numbers."
    if arr.shape != (_EXP["n_subj"],):
        return ("`first_evid_per_id` has shape %s but there are %d subjects. Use "
                "groupby('ID')['EVID'].first()." % (arr.shape, _EXP["n_subj"]))
    if _arr_close(arr, _EXP["first_evid_bad_sort"]):
        return ("every subject starts with an observation, so the sort left the pre dose "
                "sample ahead of the dose. Sort with EVID descending inside TIME, which "
                "means ascending=[True, True, False].")
    if not _arr_close(arr, _EXP["first_evid_per_id"]):
        return ("`first_evid_per_id` should be 1 for every subject. The first event in a "
                "subject's record has to be the dose.")

    try:
        evid_head = [int(x) for x in nm_df["EVID"].values[:11]]
        dv_head = [float(x) for x in nm_df["DV"].values[:11]]
    except Exception as exc:
        return "could not read the first rows of `nm`: %s" % exc
    if evid_head != _EXP["evid_head"]:
        return ("the first eleven rows of `nm` do not have the expected EVID pattern. "
                "Subject 1 should be a dose, then eight observations, then subject 2's "
                "dose.")
    if not _arr_close(dv_head, _EXP["dv_head"], rtol=1e-9, atol=1e-9):
        return ("the DV column of the first rows is not right. Dosing rows carry DV 0 and "
                "observation rows carry the measured concentration.")
    try:
        amt_dose = float(nm_df.loc[nm_df["EVID"] == 1, "AMT"].min())
        amt_obs = float(nm_df.loc[nm_df["EVID"] == 0, "AMT"].abs().max())
    except Exception as exc:
        return "could not read AMT from `nm`: %s" % exc
    if amt_dose <= 0:
        return "`nm` has a dosing row with AMT of 0. The dose amount belongs on that row."
    if amt_obs != 0:
        return "`nm` has an observation row with a non zero AMT. Observations carry AMT 0."
    return None


def _part_b():
    n_blq = _need("n_blq")
    n_mdv = _need("n_mdv")
    per_id = _need("records_per_id")
    monotone = _need("time_monotone")

    got = _as_scalar(n_blq)
    if got is None:
        return "`n_blq` should be a single number."
    if int(got) == _EXP["n_mdv"]:
        return ("`n_blq` counted every MDV row, including the dosing records. Restrict it "
                "to rows with EVID == 0.")
    if int(got) != _EXP["n_blq"]:
        return ("`n_blq` should be the number of observation rows with MDV == 1, which is "
                "the samples below the LLOQ.")

    got = _as_scalar(n_mdv)
    if got is None:
        return "`n_mdv` should be a single number."
    if int(got) == _EXP["n_blq"]:
        return ("`n_mdv` should count the dosing rows too: they always carry MDV == 1, "
                "because there is no measurement on them.")
    if int(got) != _EXP["n_mdv"]:
        return "`n_mdv` should be the total number of rows with MDV == 1."

    arr = _as_array(per_id)
    if arr is None:
        return "`records_per_id` should be an array of counts."
    if arr.shape != (_EXP["n_subj"],):
        return ("`records_per_id` has shape %s but there are %d subjects."
                % (arr.shape, _EXP["n_subj"]))
    if not _arr_close(arr, _EXP["records_per_id"]):
        return ("`records_per_id` is not the number of rows per subject. Every subject "
                "should have the same count here, one dose plus every sample.")

    if not isinstance(monotone, bool):
        return "`time_monotone` should be True or False."
    if monotone is not True:
        return ("`time_monotone` should be True for a correctly sorted dataset. If your "
                "check says otherwise, it is probably testing the whole TIME column rather "
                "than each subject's separately.")
    return None


def _part_c():
    nca_df = _need("nca")
    dn_low = _need("auc_dn_low")
    dn_high = _need("auc_dn_high")
    ratio = _need("auc_dn_ratio")

    cols = _cols_of(nca_df)
    if cols is None:
        return "`nca` should be a pandas DataFrame."
    for want in ("ID", "CMAX", "TMAX", "AUC", "AMT", "AUC_DN"):
        if want not in cols:
            return ("`nca` has no %s column. It should carry ID, CMAX, TMAX, AUC, the "
                    "merged dosing and demographic columns, and AUC_DN." % want)
    if len(nca_df) != _EXP["n_subj"]:
        return ("`nca` has %d rows but there are %d subjects. If it has more, a merge "
                "duplicated rows, which validate='one_to_one' would have caught."
                % (len(nca_df), _EXP["n_subj"]))
    if "WT_KG" not in cols:
        return "`nca` is missing the demographic columns. Merge `demog` on as well."

    try:
        nca_sorted = nca_df.sort_values("ID")
        cmax = [float(x) for x in nca_sorted["CMAX"].values]
        tmax = [float(x) for x in nca_sorted["TMAX"].values]
        auc = [float(x) for x in nca_sorted["AUC"].values]
        auc_dn = [float(x) for x in nca_sorted["AUC_DN"].values]
    except Exception as exc:
        return "could not read the NCA columns: %s" % exc

    if not _arr_close(cmax, _EXP["cmax"], rtol=1e-9):
        return ("the CMAX column is not the highest DV for each subject. Work from the "
                "observation rows only.")
    if not _arr_close(tmax, _EXP["tmax"], rtol=1e-9):
        return ("the TMAX column is not the time of each subject's highest DV. Note that "
                "it is the TIME at the peak, not the position of the peak.")
    if not _arr_close(auc, _EXP["auc"], rtol=1e-6):
        return ("the AUC column is not the linear trapezoid area over each subject's "
                "observations. Sort by TIME inside the group first, and integrate DV as "
                "it stands, so that BLQ records count as zero.")
    if not _arr_close(auc_dn, _EXP["auc_dn"], rtol=1e-6):
        return "the AUC_DN column should be AUC divided by AMT."

    for name, val in (("auc_dn_low", dn_low), ("auc_dn_high", dn_high),
                      ("auc_dn_ratio", ratio)):
        if _as_scalar(val) is None:
            return "`%s` should be a single number." % name

    got = _as_scalar(dn_low)
    if _num_close(got, _EXP["auc_dn_mean_low"], rtol=1e-6):
        return ("`auc_dn_low` is the arithmetic mean. Exposure summaries use the geometric "
                "mean, exp(mean(log(x))).")
    if _num_close(got, _EXP["auc_dn_high"], rtol=1e-6):
        return "`auc_dn_low` and `auc_dn_high` are the wrong way round."
    if not _num_close(got, _EXP["auc_dn_low"], rtol=1e-6):
        return ("`auc_dn_low` is not the geometric mean dose normalised AUC in the 50 mg "
                "group.")
    if not _num_close(_as_scalar(dn_high), _EXP["auc_dn_high"], rtol=1e-6):
        return ("`auc_dn_high` is not the geometric mean dose normalised AUC in the 150 mg "
                "group.")
    got = _as_scalar(ratio)
    if _num_close(got, 1.0 / _EXP["auc_dn_ratio"], rtol=1e-6):
        return "`auc_dn_ratio` is upside down. The task asks for high divided by low."
    if not _num_close(got, _EXP["auc_dn_ratio"], rtol=1e-6):
        return "`auc_dn_ratio` is not auc_dn_high divided by auc_dn_low."
    return None
'''

# Functions this problem introduces. Keys must exist in tools/function_catalog.py.
FUNCTIONS = [
    "NONMEM event records",
    "EVID and MDV",
    "BLQ handling",
    "sort_values mixed ascending",
    "groupby apply several outputs",
]
