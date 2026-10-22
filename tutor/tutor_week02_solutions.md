# Week 2 reference solutions

One way to do each part. The checks test the numbers, not the route, so a
different correct answer is still correct.

Do not paste these to the student. Use them to work out where their
version diverges, and say the smallest thing that gets them unstuck.

---

## W2D1. Baseline correction: transform, and why it is not agg

### Part a

```python
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
```

### Part b

```python
# Part b. Percent change from baseline, post-dose rows only.

df_post = df.copy()
df_post["PCT_CHANGE"] = 100.0 * (df_post["BIOMARKER"] - df_post["BASELINE"]) / df_post["BASELINE"]
df_post = df_post[df_post["TIME_H"] > 0].reset_index(drop=True)

if df_post is not None:
    print(df_post.shape)
    print(df_post.head(6))
```

### Part c

```python
# Part c. Mean and SD percent change per group per timepoint.
# Columns wanted: GROUP, TIME_H, PCT_MEAN, PCT_SD, N

summary = (df_post
           .groupby(["GROUP", "TIME_H"])
           .agg(PCT_MEAN=("PCT_CHANGE", "mean"),
                PCT_SD=("PCT_CHANGE", "std"),
                N=("PCT_CHANGE", "size"))
           .reset_index())

print(summary)
```

---

## W2D2. Joins that lie: validate the merge before you trust it

### Part a

```python
# Part a. Inspect before merging.

n_pk_rows = len(pk)
n_dup_rows = int(dosing.duplicated().sum())
dup_subjects = sorted(dosing.loc[dosing.duplicated(subset=["SUBJECT"], keep=False),
                                 "SUBJECT"].unique())

print("n_pk_rows   :", n_pk_rows)
print("n_dup_rows  :", n_dup_rows)
print("dup_subjects:", dup_subjects)
```

### Part b

```python
# Part b. Deduplicate, then merge with a contract.

dosing_clean = dosing.drop_duplicates().reset_index(drop=True)
merged = pd.merge(pk, dosing_clean, on="SUBJECT", how="left", validate="m:1")

print("rows in pk    :", len(pk))
print("rows in merged:", None if merged is None else len(merged))
if merged is not None:
    print(merged.head())
```

### Part c

```python
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
```

---

## W2D3. Nominal versus actual sampling times, and the samples that never arrived

### Part a

```python
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
```

### Part b

```python
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
```

### Part c

```python
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
```

---

## W2D4. Adverse events: counting events is not counting patients

### Part a

```python
# Part a. Event counts by group and severity, with totals.

ae_counts = pd.crosstab(ae["GROUP"], ae["SEVERITY"],
                        margins=True, margins_name="Total")
ae_counts = ae_counts.reindex(GROUP_ORDER + ["Total"])

print(ae_counts)
```

### Part b

```python
# Part b. Row percentages: of each group's events, what share was each severity.

ae_pct = pd.crosstab(ae["GROUP"], ae["SEVERITY"], normalize="index") * 100.0
ae_pct = ae_pct.reindex(GROUP_ORDER).round(1)

print(ae_pct)
if ae_pct is not None:
    print("")
    print("row sums:", ae_pct.sum(axis=1).to_dict())
```

### Part c

```python
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
```

---

## W2D5. Two panels and a spaghetti plot: the figure every PK report opens with

### Part a

```python
# Part a. Mean, SD and n per dose group per timepoint.

summary = (pk
           .groupby(["DOSE", "TIME_H"])
           .agg(CONC_MEAN=("CONC_MG_L", "mean"),
                CONC_SD=("CONC_MG_L", "std"),
                N=("CONC_MG_L", "size"))
           .reset_index())

print(summary)
```

### Part b

```python
# Part b. Two panels: the same mean profiles, linear on the left, log y on the right.

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

for _name, _part in summary.groupby("DOSE"):
    for _ax in axes:
        _ax.errorbar(_part["TIME_H"], _part["CONC_MEAN"], yerr=_part["CONC_SD"],
                     fmt="o-", capsize=4, lw=1.4, label=_name)

axes[1].set_yscale("log")

axes[0].set_title("Mean profile, linear")
axes[1].set_title("Mean profile, semi log")
for _ax in axes:
    _ax.set_xlabel("Time after dose (h)")
    _ax.set_ylabel("Concentration (mg/L)")
axes[0].legend(frameon=False, title="Dose")

fig.tight_layout()
plt.show()
```

### Part c

```python
# Part c. Spaghetti plot: every subject, no averaging.

fig2, ax2 = plt.subplots(figsize=(6.5, 4.5))

colours = {"50 mg": "tab:blue", "200 mg": "tab:red"}

for _sid, _rows in pk.groupby("SUBJECT"):
    _dose = _rows["DOSE"].iloc[0]
    ax2.plot(_rows["TIME_H"], _rows["CONC_MG_L"],
             lw=0.9, alpha=0.6, color=colours[_dose])

ax2.set_yscale("log")

ax2.set_xlabel("Time after dose (h)")
ax2.set_ylabel("Concentration (mg/L)")
ax2.set_title("Every subject, coloured by dose")

plt.show()
```
