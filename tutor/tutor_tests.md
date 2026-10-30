# Tests

The questions in each test, the variables they are graded on, and every
failure message their checks can produce. The worked answers are in
tutor_tests_review.md, which is a separate upload on purpose.

---

# Q-W1W2. Test: weeks 1 and 2

- covers: weeks 1 and 2
- notebook: quizzes/week01_02_quiz.ipynb
- revision sheet: quizzes/week01_02_revision.md
- 17 questions, about 60 minutes

The student scores it with `check()`, which marks every question and
names what to revise without revealing anything, and then reads
`review()`, which is the worked review built into the notebook. If they
have already run `review()`, the answers are no longer a secret and you
can talk about them freely.

## The setup cell every question works on

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

QUIZ_ID = "Q-W1W2"
TOPIC = "numpy, pandas and matplotlib, everything from weeks 1 and 2"

rng = np.random.default_rng(1202)

# ---------------------------------------------------------------------------
# A small oral PK study in rats. Two dose groups, four animals each, plasma
# sampled over a day, plus an imaging readout with several cells per animal.
# All synthetic, all from the seed above, so your numbers are my numbers.
# ---------------------------------------------------------------------------

N_ANIMALS = 8
NOMINAL = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0])
_group_of = ["low"] * 4 + ["high"] * 4
_dose_of = {"low": 1.0, "high": 3.0}

_ka, _ke = 1.2, 0.15
_cl = rng.lognormal(0.0, 0.22, N_ANIMALS)

_rows = []
for _i in range(N_ANIMALS):
    _g = _group_of[_i]
    _scale = 420.0 * _dose_of[_g] / _cl[_i]
    _true = _scale * (np.exp(-_ke * NOMINAL) - np.exp(-_ka * NOMINAL))
    _obs = _true * rng.lognormal(0.0, 0.09, NOMINAL.size)
    _actual = NOMINAL + np.concatenate([[0.0], rng.normal(0.0, 0.035, NOMINAL.size - 1)])
    _actual = np.round(np.maximum(_actual, 0.0), 3)
    for _j, _t in enumerate(NOMINAL):
        _c = np.nan if _j == 0 else round(float(_obs[_j]), 1)
        _rows.append((_i + 1, _g, float(_t), float(_actual[_j]), _c))

# The sample table from the bioanalytical lab. Pre dose samples are below the
# limit of quantification and come back as nan. Animal 3's 0.5 h sample was
# haemolysed and never assayed.
pk = pd.DataFrame(_rows, columns=["ANIMAL", "GROUP", "NOMINAL_H", "ACTUAL_H", "CONC_NG_ML"])
pk.loc[(pk["ANIMAL"] == 3) & (pk["NOMINAL_H"] == 0.5), "CONC_NG_ML"] = np.nan

# Demographics from the clinical database. Animal 6 was never entered.
_wt = np.round(rng.normal(285, 22, N_ANIMALS), 0)
demo = pd.DataFrame({"ANIMAL": np.arange(1, N_ANIMALS + 1),
                     "SEX": ["M", "M", "F", "F", "M", "F", "M", "F"],
                     "BODYWEIGHT_G": _wt})
demo = demo[demo["ANIMAL"] != 6].reset_index(drop=True)

# One row per animal, saying whether it had the adverse event of interest.
ae = pd.DataFrame({"ANIMAL": np.arange(1, N_ANIMALS + 1),
                   "GROUP": _group_of,
                   "EVENT": ["no", "no", "no", "no", "yes", "yes", "no", "yes"]})

# The imaging readout: several cells per animal, and not the same number in
# each, because that is how imaging works.
_cell_rows = []
_base = {"low": 18.0, "high": 46.0}
for _i in range(N_ANIMALS):
    _g = _group_of[_i]
    _n = int(rng.integers(9, 23))
    _animal_eff = _base[_g] + rng.normal(0, 7.0)
    for _c in range(_n):
        _cell_rows.append((_i + 1, _g, _c + 1,
                           round(float(_animal_eff + rng.normal(0, 9.0)), 2)))
cells = pd.DataFrame(_cell_rows, columns=["ANIMAL", "GROUP", "CELL_ID", "INHIB_PCT"])

# Two plain numpy views of the same plasma data, for section A.
_one = pk[pk["ANIMAL"] == 3].sort_values("NOMINAL_H")
time_single = _one["NOMINAL_H"].to_numpy()          # shape (7,), includes 0 h
conc_single = _one["CONC_NG_ML"].to_numpy()         # shape (7,), two nan
conc_matrix = (pk[pk["NOMINAL_H"] > 0]
               .pivot_table(values="CONC_NG_ML", index="ANIMAL", columns="NOMINAL_H")
               .to_numpy())                         # shape (8, 6), one nan
time_matrix = NOMINAL[1:]                           # shape (6,)

print("pk           %d rows, %d animals, %d nan concentrations"
      % (len(pk), pk["ANIMAL"].nunique(), int(pk["CONC_NG_ML"].isna().sum())))
print("demo         %d rows" % len(demo))
print("ae           %d rows" % len(ae))
print("cells        %d rows, %d animals" % (len(cells), cells["ANIMAL"].nunique()))
print("conc_single  shape %s, for animal 3" % (conc_single.shape,))
print("conc_matrix  shape %s, animals down the rows, times across" % (conc_matrix.shape,))
print("time_matrix  %s" % time_matrix)
print("")
print(pk.head(9).to_string(index=False))
print("")
print(cells.head(4).to_string(index=False))
```

## Section A, numpy

Five questions on plain arrays. No pandas needed, and none of them wants a loop.

### Question a1. finding and masking the samples you cannot use

`conc_single` holds animal 3's seven concentrations and `time_single` the seven
nominal times. Two of the concentrations are `nan`.

Work out:

- `n_missing`, how many concentrations are `nan`, as an integer,
- `mask_ok`, a boolean array that is True where the concentration is usable,
- `conc_ok` and `time_ok`, the concentrations and their times, keeping only the
  usable samples.

- graded on: `n_missing`, `mask_ok`, `conc_ok`, `time_ok`
- revise: W1D1 part a, and Missing values in the revision sheet

The starter cell they see:

```python
n_missing = None   # YOUR CODE
mask_ok = None     # YOUR CODE
conc_ok = None     # YOUR CODE
time_ok = None     # YOUR CODE

print("missing:", n_missing)
print("kept   :", conc_ok)
print("at     :", time_ok)
```

Every failure message this question can produce:

- `n_missing` should be a single whole number.
- `n_missing` counted the usable samples rather than the missing ones.
- `n_missing` is not the number of nan concentrations.
- `mask_ok` should be a boolean array with one entry per sample, so %d of them.
- `mask_ok` is inverted. True should mean the sample IS usable, so the mask is ~np.isnan(conc_single).
- `mask_ok` does not flag the usable samples.
- `%s` should be an array of numbers.
- `%s` still has all %d samples. Index it with your mask.
- `%s` has %d values but %d samples are usable.
- `%s` does not hold the right values.

### Question a2. peak and time of peak

From the usable samples only, give `cmax`, the highest concentration, and
`tmax`, the time at which it occurred. Both as plain numbers.

- graded on: `cmax`, `tmax`
- revise: W1D1 part b, and Reductions in the revision sheet

The starter cell they see:

```python
cmax = None   # YOUR CODE
tmax = None   # YOUR CODE

print("Cmax %r ng/mL at %r h" % (cmax, tmax))
```

Every failure message this question can produce:

- `cmax` is not the highest usable concentration.
- `tmax` is the position of the peak, not the time. Use it to index the time array.
- `tmax` is not the time at which the peak occurred.
- `%s` should be a single number.

### Question a3. terminal slope and half-life

Take the **last three** usable samples and fit a straight line to the log of
the concentration against time.

Give `lambda_z`, the terminal elimination rate constant in per hour, as a
positive number, and `t_half`, the terminal half-life in hours.

- graded on: `lambda_z`, `t_half`
- revise: W1D1 part c, and Fitting a line to logs in the revision sheet

The starter cell they see:

```python
lambda_z = None   # YOUR CODE
t_half = None     # YOUR CODE

print("lambda_z %r per h, half-life %r h" % (lambda_z, t_half))
```

Every failure message this question can produce:

- `lambda_z` is negative. The fitted slope is negative and lambda_z is defined as its positive counterpart.
- `lambda_z` came from log10. The terminal rate constant is defined on the natural log, and the two differ by a factor of 2.303.
- `lambda_z` used every usable sample, including the absorption phase, which drags the slope shallower. The task asks for the last three.
- `lambda_z` is not the negative slope of log concentration on time.
- `t_half` is 1 / lambda_z. A half-life carries a factor of ln(2).
- `t_half` is not ln(2) divided by lambda_z.
- `%s` should be a single number.

### Question a4. axis reductions, and the sample SD

`conc_matrix` has one row per animal and one column per sampling time, and one
`nan` in it.

Give three arrays of six numbers each, one value per sampling time, across all
eight animals:

- `mean_by_time`, the mean concentration,
- `sd_by_time`, the **sample** standard deviation,
- `cv_by_time`, the coefficient of variation as a percentage.

The `nan` has to be ignored rather than propagated.

- graded on: `mean_by_time`, `sd_by_time`, `cv_by_time`
- revise: W1D2 parts a and b, and Reductions in the revision sheet

The starter cell they see:

```python
mean_by_time = None   # YOUR CODE
sd_by_time = None     # YOUR CODE
cv_by_time = None     # YOUR CODE

print("mean:", np.round(mean_by_time, 2) if mean_by_time is not None else None)
print("sd  :", np.round(sd_by_time, 2) if sd_by_time is not None else None)
print("cv %%:", np.round(cv_by_time, 1) if cv_by_time is not None else None)
```

Every failure message this question can produce:

- `mean_by_time` is not the mean concentration at each sampling time.
- `sd_by_time` divides by n. numpy needs ddof=1 for the sample SD, which is what is reported. Pandas .std() already does this, numpy does not.
- `sd_by_time` is not the sample standard deviation at each time.
- `cv_by_time` is a fraction. The task asks for a percentage.
- `cv_by_time` is not 100 * sd / mean at each time.
- `%s` should be an array of numbers.
- `%s` should hold one value per sampling time, so %d numbers.
- `%s` contains nan. One missing sample turns a plain mean or std into nan for that whole column, which is what np.nanmean and np.nanstd are for.
- `%s` has one value per ANIMAL, so the reduction collapsed the wrong axis. axis=0 goes down the columns and leaves one value per time.

### Question a5. broadcasting with keepdims

Give `norm_matrix`, the same shape as `conc_matrix`, where every animal's
profile has been divided by **that animal's own** highest concentration. Each
row should therefore peak at 1.

One expression, no loop.

- graded on: `norm_matrix`
- revise: W1D2 part c, and Broadcasting in the revision sheet

The starter cell they see:

```python
norm_matrix = None   # YOUR CODE

print(np.round(norm_matrix, 3) if norm_matrix is not None else None)
```

Every failure message this question can produce:

- `norm_matrix` should be an array.
- `norm_matrix` has shape %s and it should have shape %s, the same as conc_matrix.
- `norm_matrix` is not conc_matrix divided by each animal's own peak.
- `norm_matrix` is transposed. Dividing by a shape (8,) array lines the axes up from the right, which is not what you want here.
- each row of `norm_matrix` should peak at exactly 1, and yours does not. That means the divisor was not each animal's own maximum.

## Section B, pandas

Six questions on the sample table and the two lookup tables.

### Question b1. filtering on two conditions

From `pk`, build `post`, the rows that are both **after** dosing and have a
concentration, and give `n_post`, how many rows that is.

- graded on: `post`, `n_post`
- revise: W1D3 part a, and Filtering in the revision sheet

The starter cell they see:

```python
post = None     # YOUR CODE
n_post = None   # YOUR CODE

print("rows kept:", n_post)
```

Every failure message this question can produce:

- `post` should be a pandas DataFrame.
- `post` still has every row of pk. Both conditions have to be applied: after dosing, and with a concentration.
- `post` has %d rows and it should have %d. Pre dose rows and rows with a nan concentration both go.
- `post` should keep all of the original columns.
- `post` still contains nan concentrations.
- `n_post` should be the number of rows in `post`.

### Question b2. groupby.agg with named outputs

From `post`, build `by_group`: the mean and standard deviation of
`CONC_NG_ML` for every combination of `GROUP` and `NOMINAL_H`.

The result should have `GROUP` and `NOMINAL_H` as ordinary columns, not as an
index, and the two summary columns must be named exactly `CONC_MEAN` and
`CONC_SD`.

- graded on: `by_group`
- revise: W1D3 part b, and Grouping in the revision sheet

The starter cell they see:

```python
by_group = None   # YOUR CODE

print(by_group.to_string(index=False) if by_group is not None else None)
```

Every failure message this question can produce:

- `by_group` should be a pandas DataFrame.
- `by_group` has %d rows and it should have %d, one per group and time.
- `by_group` does not hold one row per group and nominal time.
- `CONC_MEAN` is not the mean concentration in each group at each time.
- `CONC_SD` divides by n. Pandas .std() is the sample SD already, so this looks like ddof=0 was passed explicitly.
- `CONC_SD` is not the sample SD within each group and time.
- `by_group` has no %s column.
- `by_group` has no %s column. Named aggregation, CONC_MEAN=("CONC_NG_ML", "mean"), gets the names right in one step.
- `%s` is in the index rather than a column. Call .reset_index() after the groupby.

### Question b3. a left merge, validated

Join the body weights onto the sample table. Build `merged`, every row of `pk`
with its animal's `SEX` and `BODYWEIGHT_G` attached, and give
`n_missing_weight`, the number of **animals** that have no weight.

Pass the `validate` argument that states what you expect of the two key
columns, so that a duplicated demographics row would raise rather than
silently multiply your data.

- graded on: `merged`, `n_missing_weight`
- revise: W1D3 part c and W2D2, and Merges in the revision sheet

The starter cell they see:

```python
merged = None             # YOUR CODE
n_missing_weight = None   # YOUR CODE

print("rows: %s, animals with no weight: %s" % (
    len(merged) if merged is not None else None, n_missing_weight))
```

Every failure message this question can produce:

- `merged` should be a pandas DataFrame.
- `merged` has no BODYWEIGHT_G column, so the join did not bring demo in.
- `merged` has %d rows and pk has %d. An inner join drops the animal with no demographics record, which hides exactly the thing you are looking for. Use how='left'.
- `merged` has %d rows, more than the %d in pk, so a key was duplicated. validate='many_to_one' would have raised instead.
- `n_missing_weight` should be a single whole number.
- `n_missing_weight` counted ROWS with no weight. The task asks for animals, so count distinct ANIMAL values with .nunique().
- `n_missing_weight` is not the number of animals with no body weight.

### Question b4. transform against agg

Add a column to `pk` called `FRAC_OF_PEAK`: each concentration divided by
**that animal's own** highest concentration, so every animal's profile peaks at
1. Put the resulting Series in `frac_of_peak` as well.

This is the pandas version of question a5, and the whole question is which
groupby method broadcasts a per animal number back onto every one of that
animal's rows.

- graded on: `frac_of_peak`
- revise: W2D1, and Transform against agg in the revision sheet

The starter cell they see:

```python
frac_of_peak = None   # YOUR CODE, and also assign it to pk["FRAC_OF_PEAK"]

print(pk[pk["ANIMAL"] == 3].to_string(index=False) if frac_of_peak is not None else None)
```

Every failure message this question can produce:

- `frac_of_peak` should be a Series or array of numbers.
- `frac_of_peak` has one value per animal, so this is .agg or .max rather than .transform. A transform gives the group's value back on every one of its rows, which is what you need to divide by.
- `frac_of_peak` has %d values and pk has %d rows.
- `frac_of_peak` should still be nan wherever the concentration was nan, so %d real values out of %d.
- `frac_of_peak` is not each concentration over that animal's own maximum.
- the column `FRAC_OF_PEAK` has not been added to pk.

### Question b5. nominal against actual times, and pivoting

Two parts, both about the difference between the time a sample was meant to be
taken and the time it was.

- `max_dev_min`, the largest absolute difference between `ACTUAL_H` and
  `NOMINAL_H` anywhere in the study, **in minutes**,
- `wide`, a table of concentrations with one row per animal and one column per
  **nominal** time, built with `pivot_table`.

- graded on: `max_dev_min`, `wide`
- revise: W2D3, and Reshaping in the revision sheet

The starter cell they see:

```python
max_dev_min = None   # YOUR CODE
wide = None          # YOUR CODE

print("worst deviation: %r min" % (max_dev_min,))
print(wide.round(1).to_string() if wide is not None else None)
```

Every failure message this question can produce:

- `max_dev_min` should be a single number.
- `max_dev_min` is in hours. The task asks for minutes.
- `max_dev_min` is not the largest absolute deviation of ACTUAL_H from NOMINAL_H, in minutes.
- `wide` should be a pandas DataFrame from pivot_table.
- `wide` has shape %s and it should have shape %s. The pre dose column is entirely nan, and pivot_table drops it.
- the columns of `wide` should be the nominal times.
- `wide` does not hold the concentrations for each animal at each time.
- `wide` is the wrong way round. index="ANIMAL" puts animals down the rows and columns="NOMINAL_H" puts times across.

### Question b6. an incidence table, and its denominators

From `ae`, build `inc`, a table of counts with one row per `GROUP` and one
column per `EVENT` value, using `pd.crosstab`.

Then give `pct_high_event`, the percentage of **high dose** animals that had
the event.

- graded on: `inc`, `pct_high_event`
- revise: W2D4, and Crosstabs in the revision sheet

The starter cell they see:

```python
inc = None              # YOUR CODE
pct_high_event = None   # YOUR CODE

print(inc.to_string() if inc is not None else None)
print("high dose incidence: %r pct" % (pct_high_event,))
```

Every failure message this question can produce:

- `inc` should be a table, from pd.crosstab.
- `inc` has shape %s and it should have shape %s, one row per group and one column per event value.
- the rows of `inc` should be the dose groups, so GROUP is the first argument to crosstab.
- the columns of `inc` should be the EVENT values.
- `inc` does not hold the counts of animals in each group and event category.
- `pct_high_event` should be a single number.
- `pct_high_event` is a fraction. The task asks for a percentage.
- `pct_high_event` used the whole study as the denominator. Incidence is within group, so divide by the number of high dose animals.
- `pct_high_event` is not the percentage of high dose animals with the event.

## Section C, matplotlib and nested data

Three questions where the figure is the answer, or the aggregation behind it is.

### Question c1. aggregating nested data to the animal first

`cells` holds between 9 and 22 cells for each animal. The unit of replication
is the animal, not the cell.

Build:

- `animal_level`, one row per animal, with columns `ANIMAL`, `GROUP`,
  `MEAN_INHIB`, the mean of that animal's own cells, and `N_CELLS`, how many
  cells it had,
- `group_summary`, one row per group, with columns `GROUP`, `MEAN`, `SD` and
  `N`, computed from the animal means rather than from the cells.

Both with the group keys as ordinary columns.

- graded on: `animal_level`, `group_summary`
- revise: W1D5 parts a and b, and Nested data in the revision sheet

The starter cell they see:

```python
animal_level = None    # YOUR CODE
group_summary = None   # YOUR CODE

print(animal_level.to_string(index=False) if animal_level is not None else None)
print("")
print(group_summary.to_string(index=False) if group_summary is not None else None)
```

Every failure message this question can produce:

- `animal_level` should be a pandas DataFrame.
- `animal_level` still has one row per cell. Group by animal before aggregating.
- `animal_level` has %d rows and there are %d animals.
- `animal_level` does not list the eight animals.
- GROUP does not line up with ANIMAL. Group by both at once so each animal keeps its own label.
- `MEAN_INHIB` is the median of each animal's cells, not the mean.
- `MEAN_INHIB` is not the mean of each animal's own cells.
- `N_CELLS` is not the number of cells in each animal.
- `group_summary` should be a pandas DataFrame.
- `group_summary` has %d rows and there are %d groups.
- `group_summary` does not hold one row per dose group.
- `N` is the number of CELLS. The unit of replication is the animal, so N is the number of animals and `group_summary` has to be built from `animal_level`.
- `N` is not the number of animals in each group.
- `MEAN` pools every cell, which weights each animal by however many cells it happened to have. Average the animal means instead.
- `MEAN` is not the mean of the animal means in each group.
- `SD` is the cell level spread, which is a statement about the assay. The SD you report is between animals, from the animal means.
- `SD` is the standard error. The house rule is SD, which does not shrink when you add animals.
- `SD` is not the sample SD of the animal means in each group.
- `animal_level` has no %s column.
- `group_summary` has no %s column.
- `%s` is in the index rather than a column. Call .reset_index().

### Question c2. a plot with SD error bars

Plot `group_summary` from the question above: the group mean inhibition with
error bars, into `fig_c2` and `ax_c2` from `plt.subplots()`.

The error bars are the standard deviation of the animal means. Label both axes.

The check reads the means and the error bar lengths back off the axes, so what
matters is what you drew, not how you drew it.

- graded on: see the starter
- revise: W1D5 part c, and Error bars in the revision sheet

The starter cell they see:

```python
fig_c2, ax_c2 = plt.subplots(figsize=(5, 4))

# YOUR CODE: draw the group means with SD error bars, and label the axes

plt.show()
```

Every failure message this question can produce:

- `fig_c2` should be a matplotlib Figure, from plt.subplots().
- `ax_c2` should be a single matplotlib Axes.
- no error bars were found on the axes. Draw the means with ax.errorbar and pass yerr, rather than plotting them as plain markers.
- there are %d error bars on the axes and %d groups to plot.
- the error bars are the SEM. They should be the SD of the animal means, which is the house rule and is roughly twice as long here.
- the error bars are the cell level SD rather than the animal level SD.
- the error bar lengths do not match the SD of the animal means.
- the y axis has no label. Say what is plotted and what the error bars are.
- the x axis has no label.
- the plotted means come from pooling every cell rather than averaging the animal means, so c1 needs fixing first.
- the plotted means are not the group means from `group_summary`.

### Question c3. a semilog concentration time plot

Plot the mean concentration against nominal time for both dose groups on the
same axes, into `fig_c3` and `ax_c3`, with a **logarithmic y axis**.

Use `by_group` from question b2. One line per group, markers on the points,
axes labelled, and a legend.

- graded on: see the starter
- revise: W2D5, and Log scales in the revision sheet

The starter cell they see:

```python
fig_c3, ax_c3 = plt.subplots(figsize=(6, 4))

# YOUR CODE: one line per group, log y axis, labels and a legend

plt.show()
```

Every failure message this question can produce:

- `fig_c3` should be a matplotlib Figure, from plt.subplots().
- `ax_c3` should be a single matplotlib Axes.
- the y axis is still linear. ax.set_yscale("log"), or ax.semilogy instead of ax.plot.
- there should be one line per dose group, each with one point per nominal time, and %d were found.
- the plotted lines are not the group mean concentrations from `by_group`. One line per group, sorted by nominal time.
- both axes need labels.
- there is no legend, so nothing says which line is which dose.

## Section D, judgement

Three questions with no code to write. Set each variable to the letter of the right answer, as a string, for example "a".

### Question d1. SD, SEM and which n

You are putting error bars on the group mean inhibition from question c1, where
each group has 4 animals and 61 to 72 cells.

Which is the right quantity?

- **a.** `cells.groupby("GROUP")["INHIB_PCT"].std()`, the SD of all the cells
- **b.** `animal_level.groupby("GROUP")["MEAN_INHIB"].std()`, the SD of the
  animal means
- **c.** the SD of the animal means divided by the square root of 4
- **d.** the SD of all the cells divided by the square root of the number of
  cells

Set `q_sd` to the letter.

- graded on: `q_sd`
- revise: the house rules, and Error bars in the revision sheet

The starter cell they see:

```python
q_sd = None   # YOUR CODE: "a", "b", "c" or "d"
```

Every failure message this question can produce:

- `q_sd` should be one of the letters "a", "b", "c" or "d".
- that is the spread of the individual cells, which describes the assay rather than the animals, and it uses the wrong unit of replication.
- that is a standard error. It answers how precisely you located the mean, and it shrinks as you add animals whether or not the biology changes.
- that is a standard error computed from cells, so it is wrong twice: the wrong level and the wrong quantity.
- that is not the right option.

### Question d2. which axis is which

`conc_matrix` has shape `(8, 6)`: eight animals down the rows, six sampling
times across the columns.

You want **each animal's own peak concentration**, so eight numbers.

- **a.** `np.nanmax(conc_matrix, axis=0)`
- **b.** `np.nanmax(conc_matrix, axis=1)`
- **c.** `np.nanmax(conc_matrix)`
- **d.** `np.nanmax(conc_matrix, keepdims=True)`

Set `q_axis` to the letter.

- graded on: `q_axis`
- revise: W1D2, and Reductions in the revision sheet

The starter cell they see:

```python
q_axis = None   # YOUR CODE: "a", "b", "c" or "d"
```

Every failure message this question can produce:

- `q_axis` should be one of the letters "a", "b", "c" or "d".
- that collapses the animals and leaves one number per sampling time, so six numbers rather than eight. The axis you name is the one that disappears.
- that reduces the whole matrix to a single number.
- that also reduces everything to one number, keeping the shape as (1, 1). keepdims preserves the dimensions, it does not choose an axis.
- that is not the right option.

### Question d3. what validate is for

You are joining `pk`, which has seven rows per animal, onto `demo`, which
should have exactly one row per animal.

Which `validate` argument states that, so that a duplicated demographics row
would raise instead of multiplying your sample rows?

- **a.** `validate="one_to_one"`
- **b.** `validate="one_to_many"`
- **c.** `validate="many_to_one"`
- **d.** no `validate` argument, and check `len()` afterwards instead

Set `q_validate` to the letter.

- graded on: `q_validate`
- revise: W2D2, and Merges in the revision sheet

The starter cell they see:

```python
q_validate = None   # YOUR CODE: "a", "b", "c" or "d"
```

Every failure message this question can produce:

- `q_validate` should be one of the letters "a", "b", "c" or "d".
- that claims both sides are unique, which is false on the left: pk has seven rows per animal.
- that is the right relationship stated backwards. The left frame is the one you call .merge on, and it is the many side here.
- checking len() is a real habit and worth keeping, but it tells you only that something changed, after the fact. The question asks which argument states the expectation in the line that does the work.
- that is not the right option.
