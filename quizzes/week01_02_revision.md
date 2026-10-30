# Revision for the weeks 1 and 2 test

Everything the test can ask about, in the order it is easiest to learn, with
the mistake each thing invites. Read it before you sit the test, and again
after, because the second reading is the one that sticks.

Nothing here is an answer to a test question. It is the material the questions
are drawn from.

---

## 1. Arrays

An array is a block of numbers of one type, with a shape. `np.array([1.0, 2.0])`
has shape `(2,)`, one dimension. A table of animals by times has shape
`(8, 6)`, two dimensions, rows first. `.shape` is the first thing to print when
anything surprises you, and `.dtype` is the second.

Arithmetic is elementwise and there is no loop: `conc * 2`, `conc / peak`,
`np.log(conc)`. A loop over an array is almost always a sign that something
simpler was available.

**Making them:** `np.array([...])` from a list, `np.zeros(5)`, `np.arange(0, 10, 2)`,
`np.linspace(0, 24, 97)` for a grid with both ends included.

## 2. Missing values

`nan` means not a number, and it is how a missing measurement travels through
numpy. Three facts about it, and all three matter.

**It is not equal to itself.** `np.nan == np.nan` is False, so `conc != np.nan`
is True at every position including the `nan` itself, and filtering on it keeps
the whole array. Use `np.isnan(conc)`, or `.isna()` in pandas.

**It poisons reductions.** `conc.mean()` with one `nan` in it is `nan`, for the
whole array, silently. `np.nanmean`, `np.nanstd`, `np.nanmax` ignore it
instead.

**It carries information.** A pre dose sample below the limit of quantification
and a sample the lab dropped are both `nan` and are not the same thing. Count
them and know which is which before you throw them away.

The mask pattern, which is worth memorising:

```python
ok = ~np.isnan(conc)       # True where usable
conc_ok = conc[ok]         # keep those
time_ok = time[ok]         # and the matching times
```

Indexing both arrays with the same mask is what keeps the pairs together.

## 3. Reductions, and the axis argument

`arr.mean()`, `.std()`, `.max()`, `.sum()`, `.min()`, and `np.argmax` for the
position of the largest value.

**`argmax` gives a position, not a value and not a time.** To get the time of
the peak you index the time array with it: `time[np.argmax(conc)]`.

On a 2D array you have to say which direction to collapse:

```python
arr.mean(axis=0)   # down the columns, one number per column
arr.mean(axis=1)   # across the rows, one number per row
```

**The axis you name is the one that disappears.** With animals down the rows
and times across the columns, `axis=0` gives you one number per time and
`axis=1` gives you one per animal. Both are plausible, neither raises, and only
the shape tells you which you got.

**`ddof=1`, every time, in numpy.** `np.std` divides by n, which is the
population SD and is too small. The sample SD divides by n-1 and that is what
every report means by SD. Pandas `.std()` already uses n-1. Two libraries, two
defaults, no warning from either.

A coefficient of variation is `100 * sd / mean`, as a percentage, and it is
only meaningful when the mean is far from zero and the thing is roughly
lognormal. A CV over about 50 percent across a mixed population is often not
variability at all but two groups averaged together.

## 4. Broadcasting

Two arrays can combine if their shapes line up from the **right**, with a 1 or
a missing axis counting as "stretch me".

```
(8, 6) with (6,)     -> fine, the 6 lines up with the 6
(8, 6) with (8,)     -> raises, because 8 does not line up with 6
(8, 6) with (8, 1)   -> fine, the 1 stretches across the row
```

So to divide every animal's row by that animal's own peak:

```python
peaks = np.nanmax(arr, axis=1, keepdims=True)   # shape (8, 1)
norm = arr / peaks
```

`keepdims=True` is what keeps the result 2D so it can broadcast back.
`arr[:, None]` does the same job by inserting the axis by hand.

The dangerous case is a square array, where the wrong orientation broadcasts
happily and gives you a transposed answer with no error at all.

## 5. Fitting a line to logs

For first order elimination, the log of the concentration falls linearly with
time, so:

```python
slope, intercept = np.polyfit(t_term, np.log(c_term), 1)
lambda_z = -slope
t_half = np.log(2) / lambda_z
```

Three things to get right. **`np.log` is the natural log**, and `np.log10`
gives a slope wrong by 2.303. **The slope is negative** and lambda_z is its
positive counterpart. **The half-life carries `ln(2)`**; `1 / lambda_z` is the
mean residence time, 44 percent larger, and a different quantity.

Which points go into the fit is a judgement call, not a formality. Points from
the absorption or distribution phase make the slope shallower and the
half-life longer. Weeks 7 and 8 do this properly, with adjusted R squared
choosing the window.

## 6. The three pandas objects

A **Series** is one column with an index. A **DataFrame** is a dictionary of
Series sharing one index. The **index** is the row labels, and it is not a
column until you say `.reset_index()`.

Most confusion in pandas is one of these three: something is in the index when
you thought it was a column, something is a Series when you thought it was a
DataFrame, or an operation returned a copy when you thought it was a view.

`df.head()`, `df.columns`, `df.dtypes`, `len(df)` and `df.describe()` answer
most of it in one line each.

## 7. Filtering and derived columns

```python
post = df[(df["NOMINAL_H"] > 0) & df["CONC"].notna()].copy()
df["DOSE_MG_PER_KG"] = df["DOSE_MG"] / (df["BODYWEIGHT_G"] / 1000.0)
```

**Each condition in its own brackets**, joined with `&` for and, `|` for or,
`~` for not. The words `and`, `or` and `not` do not work elementwise and raise
a message about an ambiguous truth value.

**`.copy()`** when you intend to add a column to a filtered result, or pandas
may warn that you are setting a value on a slice of something else, and that
warning is sometimes hiding a real bug.

Watch the units when you derive a column. A dose in mg over a weight in grams
is milligrams per gram, which is a thousand times the number anyone expects.

## 8. Grouping

```python
summary = (df.groupby(["GROUP", "NOMINAL_H"])
           .agg(CONC_MEAN=("CONC", "mean"), CONC_SD=("CONC", "std"))
           .reset_index())
```

**Named aggregation** gets you the column names you want in one step. The older
`.agg(["mean", "std"])` gives a two level column index you then have to
flatten.

**`.reset_index()`** turns the group keys back into ordinary columns. Without
it they are an index, which prints almost the same and merges differently.

`.size()` counts rows per group, `.nunique()` counts distinct values,
`.first()` takes the first row of each group.

## 9. Transform against agg

This is the most useful distinction in week 2.

**`agg`** returns one row per group. **`transform`** returns one row per
original row, with the group's value repeated.

```python
df["CHANGE"] = df["VALUE"] - df.groupby("ID")["VALUE"].transform("first")
df["FRAC"] = df["VALUE"] / df.groupby("ID")["VALUE"].transform("max")
```

Anything of the form "each value relative to its own group's something" is a
transform: baseline correction, percent of control, change from screening, a z
score within a batch. Use `agg` there and the shapes do not line up, or worse,
they do and the values are misaligned.

One trap with `transform("first")`: it takes the first row **as the frame is
currently ordered**, not the earliest time. Sort first, or use the value you
actually mean.

## 10. Merges

```python
merged = pk.merge(demo, on="ANIMAL", how="left", validate="many_to_one")
```

**`how`** decides which rows survive. `left` keeps every row of the left frame
whether or not it matched, which is what you want when a missing match is
itself a finding. `inner` keeps only matches, and will quietly delete an animal
rather than tell you about it.

**`validate`** states the relationship and raises when it is false.
`many_to_one` means many left rows per key and one right row per key, which is
the usual shape for attaching a lookup table. It is nine characters and it
catches the duplicated lookup row that would otherwise multiply your data.

**`indicator=True`** adds a `_merge` column saying where each row came from,
which is the fastest way to see what did not match.

**Check `len()` before and after, every time.** A join that drops an animal
produces no error and a perfectly plausible answer. Then check for nulls in the
columns you brought in, because `validate` says nothing about keys that had no
match at all.

## 11. Reshaping

**Wide to long:** `df.melt(id_vars=["ANIMAL"], var_name="TIME", value_name="CONC")`.
A collaborator's spreadsheet with one column per timepoint is wide; almost
everything in pandas wants long.

**Long to wide:** `df.pivot_table(values="CONC", index="ANIMAL", columns="NOMINAL_H")`.
`pivot_table` aggregates duplicates, with `aggfunc="mean"` by default;
`pivot` raises on them. Prefer `pivot_table` and state the `aggfunc` when
duplicates are possible.

Two things `pivot_table` does silently: it **drops entirely empty rows and
columns**, so a column of all `nan` simply is not in the result, and it **sorts
the keys**. Both are reasonable and both surprise people.

Column names coming out of a spreadsheet need cleaning:
`.str.strip()`, `.str.lower()`, `.str.replace()`. Check `.duplicated()` before
you pivot, because duplicate keys are how a reshape silently averages two
animals into one.

## 12. Nominal against actual times

Nominal times are **labels**: the sample was scheduled for 0.5 h. Actual times
are **measurements**: it was taken at 0.513 h.

**Group on nominal, calculate on actual.** A mean concentration at 0.5 h only
means something if every value in that column is labelled 0.5 h. Anything that
uses the spacing between samples, an AUC or a terminal slope, has to use the
actual times or it is systematically wrong.

Deviations of a few minutes look harmless and are not, early in the profile,
where four minutes can be ten percent of the elapsed time and the
concentration is climbing steeply.

## 13. Crosstabs and incidence

```python
inc = pd.crosstab(df["GROUP"], df["EVENT"])
pct = 100 * inc["yes"] / inc.sum(axis=1)
```

`crosstab` counts combinations. `pivot_table` with `aggfunc="size"` does the
same thing with more typing.

**The denominator is the group, not the study.** Incidence is within group, so
divide by the row total.

**Small denominators.** With four animals per group, one animal is 25
percentage points. Zero of four is not zero risk; it is consistent with a true
rate up to about half. Three of four and zero of four is a real difference
worth reporting and is not, on its own, statistically significant. Report the
counts alongside the percentage, always, so the reader can see the denominator.

## 14. Figures

```python
fig, ax = plt.subplots(figsize=(6, 4))
ax.errorbar(x, mean, yerr=sd, fmt="o", capsize=5)
ax.set_xlabel("dose group")
ax.set_ylabel("inhibition (pct), mean and SD of animal means")
ax.set_yscale("log")
ax.legend()
plt.show()
```

The **figure** is the page and the **axes** is one panel on it. Almost
everything you want to set is a method on the axes, and `plt.something` is a
shortcut that acts on whichever axes was most recently touched, which is fine
in a notebook and a liability in a loop.

Panels: `fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)` gives
you an array of axes, and `sharey` makes the two panels comparable, which is
usually the entire point of putting them side by side.

**Log scales** for concentration against time, because first order elimination
is a straight line on a log axis and your eye is very good at straight lines.
Curvature in the tail means a second compartment, or flip flop absorption, or
an assay running out of range. A log axis also turns a threefold dose
difference into a constant vertical offset, so dose proportionality becomes
visible rather than calculated. It cannot show zeros or `nan`, which
matplotlib drops without comment.

## 15. The house rules

These are enforced by the checks, not just mentioned, because they are the
mistakes that cost credibility fastest.

**Error bars are SD, never SEM.** SEM is the SD over the square root of n. It
shrinks as you add subjects even when the biology is identical, because it
describes how precisely you located the mean rather than how much subjects
differ. A reader given SD and n can compute the SEM. The reverse needs n, which
is usually missing.

**SD means the sample SD, with n-1.** `ddof=1` in numpy. Pandas already does
it.

**Aggregate nested data to the animal first.** Many cells inside one animal are
repeat measures of that animal, not independent samples of the drug effect.
Pooling them inflates n, weights each animal by however many cells you happened
to image, and makes every statistic afterwards answer a question nobody asked.
Cells up to one number per animal, then animals up to a group mean and SD.

**Check `len()` before and after every merge.**

**Say what your error bars are**, in the axis label or the caption. An error
bar with no stated meaning is decoration.

## 16. The ten things that actually go wrong

1. Cells run out of order, so a name is stale or undefined. Restart and run
   all, and the confusion goes away.
2. `np.std` without `ddof=1`.
3. `axis=0` where `axis=1` was meant, which does not raise.
4. `nan` turning a whole mean into `nan`.
5. `argmax` reported as if it were a time.
6. Group keys left in the index, so a later merge finds nothing.
7. `agg` where `transform` was meant.
8. An inner join silently dropping a subject.
9. SEM on the error bars.
10. Cells pooled instead of aggregated to the animal.

Nine of those ten produce a plausible number and no error message. That is why
the habit worth building is not carefulness in general but three specific
reflexes: print the shape, print the length, and print what you are about to
plot.
