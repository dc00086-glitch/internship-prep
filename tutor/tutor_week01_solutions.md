# Week 1 reference solutions

One way to do each part. The checks test the numbers, not the route, so a
different correct answer is still correct.

Do not paste these to the student. Use them to work out where their
version diverges, and say the smallest thing that gets them unstuck.

---

## W1D1. Clean one plasma profile and get the terminal half-life

### Part a

```python
# Part a. Deal with the BLQ samples.

measurable = ~np.isnan(conc)
n_blq = int(np.sum(np.isnan(conc)))
t_meas = time_h[measurable]
c_meas = conc[measurable]

print("measurable:", measurable)
print("n_blq     :", n_blq)
print("t_meas    :", t_meas)
print("c_meas    :", c_meas)
```

### Part b

```python
# Part b. Describe the peak.

cmax = float(np.max(c_meas))
tmax = float(t_meas[np.argmax(c_meas)])

print("cmax:", cmax, "mg/L")
print("tmax:", tmax, "h")
```

### Part c

```python
# Part c. Terminal half-life from the last 4 measurable samples.

t_term = t_meas[-4:]
c_term = c_meas[-4:]

slope, intercept = np.polyfit(t_term, np.log(c_term), deg=1)
lambda_z = float(-slope)
t_half = float(np.log(2.0) / lambda_z)

print("t_term  :", t_term)
print("c_term  :", c_term)
print("lambda_z:", lambda_z, "1/h")
print("t_half  :", t_half, "h")
```

---

## W1D2. A subject by time matrix: axis reductions, SD, and broadcasting

### Part a

```python
# Part a. Reduce along the right axis.

mean_by_time = conc.mean(axis=0)
cmax_by_subject = conc.max(axis=1)

print("mean_by_time   ", np.shape(mean_by_time), mean_by_time)
print("cmax_by_subject", np.shape(cmax_by_subject), cmax_by_subject)
```

### Part b

```python
# Part b. Describe the variability. SD, not SEM. Sample SD, so ddof=1.

sd_by_time = conc.std(axis=0, ddof=1)
cv_by_time = 100.0 * sd_by_time / mean_by_time

print("sd_by_time", np.shape(sd_by_time), np.round(sd_by_time, 3) if sd_by_time is not None else None)
print("cv_by_time", np.shape(cv_by_time), np.round(cv_by_time, 1) if cv_by_time is not None else None)
```

### Part c

```python
# Part c. Normalise each subject's profile to that subject's own Cmax.
# Use broadcasting, not a for loop.

norm_conc = conc / conc.max(axis=1, keepdims=True)

if norm_conc is not None:
    print("norm_conc shape:", np.shape(norm_conc))
    print("max of each row:", np.round(np.max(norm_conc, axis=1), 6))
```

---

## W1D3. Rat PK study: filter, summarise by group, and join the dosing record

### Part a

```python
# Part a. Count the animals and keep only post-dose rows.

n_animals = int(pk["ANIMAL"].nunique())
df_post = pk[pk["TIME_H"] > 0]

print("n_animals:", n_animals)
print("df_post shape:", None if df_post is None else df_post.shape)
```

### Part b

```python
# Part b. Mean and sample SD per group per timepoint, from df_post.
# Columns must come out named GROUP, TIME_H, CONC_MEAN, CONC_SD.

summary = (df_post
           .groupby(["GROUP", "TIME_H"])["CONC_NG_ML"]
           .agg(["mean", "std"])
           .reset_index()
           .rename(columns={"mean": "CONC_MEAN", "std": "CONC_SD"}))

print(summary)
```

### Part c

```python
# Part c. Join the dosing record onto pk without losing any rows.

pk_merged = pd.merge(pk, demo, on="ANIMAL", how="left")
pk_merged["DOSE_MG_PER_KG"] = pk_merged["DOSE_MG"] / (pk_merged["BODYWEIGHT_G"] / 1000.0)
n_missing_weight = int(pk_merged["BODYWEIGHT_G"].isna().sum())

print("rows in pk       :", len(pk))
print("rows in pk_merged:", None if pk_merged is None else len(pk_merged))
print("n_missing_weight :", n_missing_weight)
if pk_merged is not None:
    print(pk_merged.head())
```

---

## W1D4. A collaborator's spreadsheet: reshape wide to tidy and back again

### Part a

```python
# Part a. Reshape wide into tidy.
# Columns wanted: ANIMAL, GROUP, TIME_LABEL, INHIB_PCT

tidy = wide.melt(id_vars=["ANIMAL", "GROUP"],
                 var_name="TIME_LABEL",
                 value_name="INHIB_PCT")

print(None if tidy is None else tidy.shape)
print(None if tidy is None else tidy.head())
```

### Part b

```python
# Part b. Numeric TIME_H, drop TIME_LABEL, sort by ANIMAL then TIME_H, reset the index.

tidy_clean = tidy.copy()
tidy_clean["TIME_H"] = tidy_clean["TIME_LABEL"].str.replace("T_", "", regex=False).astype(float)
tidy_clean = (tidy_clean
              .drop(columns=["TIME_LABEL"])
              .sort_values(["ANIMAL", "TIME_H"])
              .reset_index(drop=True))

print(None if tidy_clean is None else tidy_clean.head(8))
```

### Part c

```python
# Part c. Count the repeated rows, drop them, pivot back to wide.

n_dup_rows = int(tidy_clean.duplicated().sum())
wide_again = (tidy_clean
              .drop_duplicates()
              .pivot(index="ANIMAL", columns="TIME_H", values="INHIB_PCT"))

print("n_dup_rows:", n_dup_rows)
print(None if wide_again is None else wide_again)
```

---

## W1D5. Cells inside animals: aggregate first, then plot mean with SD

### Part a

```python
# Part a. One row per animal: ANIMAL, GROUP, MEAN_INTENSITY, N_CELLS

animal_level = (cells
                .groupby(["ANIMAL", "GROUP"])
                .agg(MEAN_INTENSITY=("INTENSITY", "mean"),
                     N_CELLS=("INTENSITY", "size"))
                .reset_index())

print(animal_level)
```

### Part b

```python
# Part b. One row per group, computed across animals: GROUP, MEAN, SD, N_ANIMALS

group_stats = (animal_level
               .groupby("GROUP")
               .agg(MEAN=("MEAN_INTENSITY", "mean"),
                    SD=("MEAN_INTENSITY", "std"),
                    N_ANIMALS=("MEAN_INTENSITY", "size"))
               .reset_index()
               .sort_values("GROUP")
               .reset_index(drop=True))

print(group_stats)
```

### Part c

```python
# Part c. Group means with SD error bars, every animal overlaid.

fig, ax = plt.subplots(figsize=(6, 4.5))

x_pos = [0, 1]
order = list(group_stats["GROUP"])

ax.errorbar(x_pos, group_stats["MEAN"], yerr=group_stats["SD"],
            fmt="o", color="black", markersize=9, capsize=6,
            linestyle="none", zorder=3, label="group mean, SD")

jitter_rng = np.random.default_rng(7)
animal_x = np.array([order.index(g) for g in animal_level["GROUP"]], dtype=float)
animal_x = animal_x + jitter_rng.normal(0.0, 0.05, size=len(animal_level))
ax.scatter(animal_x, animal_level["MEAN_INTENSITY"],
           s=45, alpha=0.75, color="tab:blue", zorder=2, label="animal mean")

ax.set_xticks(x_pos)
ax.set_xticklabels(order)
ax.set_xlim(-0.5, 1.5)
ax.set_xlabel("Treatment group")
ax.set_ylabel("Target marker intensity (AU)")
ax.set_title("Target engagement by group, mean with SD across animals")
ax.legend(frameon=False)

plt.show()
```
