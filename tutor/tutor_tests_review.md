# Test answers and worked review

The model answer and the written review for every test question. This is
the same text the student gets from `review()` in the notebook, so once
they have run that cell there is nothing here they cannot see.

Before they have run it, do not paste these. Mark the question, name the
cause, and let the notebook do the revealing.

---

# Q-W1W2. Test: weeks 1 and 2

## Question a1. finding and masking the samples you cannot use

```python
n_missing = int(np.isnan(conc_single).sum())
mask_ok = ~np.isnan(conc_single)
conc_ok = conc_single[mask_ok]
time_ok = time_single[mask_ok]

print("missing:", n_missing)
print("kept   :", conc_ok)
print("at     :", time_ok)
```

`np.isnan` gives you the boolean array, `~` flips it, and the same mask indexes
both arrays so the times stay lined up with the concentrations. Two samples are
missing for two different reasons: the pre dose one was never going to have
drug in it, and the 0.5 h one was haemolysed. Both are `nan` and both have to
go, but only the second one is a loss.

The mistake worth naming: `conc_single != np.nan` is always True, because `nan`
is not equal to anything, including itself. That is the whole reason `np.isnan`
exists.

## Question a2. peak and time of peak

```python
cmax = float(conc_ok.max())
tmax = float(time_ok[np.argmax(conc_ok)])

print("Cmax %r ng/mL at %r h" % (cmax, tmax))
```

`argmax` gives a position, not a time, so it has to be used to index the time
array. Handing back the position itself is the single most common slip in this
curriculum, and the check catches it by name.

Note also that `cmax` is the highest **sampled** concentration, not the peak of
the underlying curve. Here the true peak is at 1.98 h and there is a sample at
2 h, so the two nearly coincide, and assay noise means the measured value can
land either side of the real one. Sample at 1 and 4 hours instead and the
measured Cmax would sit well below the truth. NCA reports the measured value
anyway, because it is what was measured, which is worth remembering when
somebody compares a Cmax from two studies with different schedules.

## Question a3. terminal slope and half-life

```python
_slope, _intercept = np.polyfit(time_ok[-3:], np.log(conc_ok[-3:]), 1)
lambda_z = float(-_slope)
t_half = float(np.log(2) / lambda_z)

print("lambda_z %r per h, half-life %r h" % (lambda_z, t_half))
```

The fitted slope is negative, because the concentration falls, and `lambda_z`
is defined as its positive counterpart. So the minus sign is not a detail, it
is the definition.

`np.log` is the natural log. `np.log10` would give you a slope smaller by a
factor of 2.303 and a half-life wrong by the same factor, which is a mistake
that looks plausible on a plot and is not.

The half-life carries the `ln(2)`. `1 / lambda_z` is also a meaningful number,
the mean residence time for a one compartment model, but it is 44 percent
larger than the half-life and they are not interchangeable.

## Question a4. axis reductions, and the sample SD

```python
mean_by_time = np.nanmean(conc_matrix, axis=0)
sd_by_time = np.nanstd(conc_matrix, axis=0, ddof=1)
cv_by_time = 100.0 * sd_by_time / mean_by_time

print("mean:", np.round(mean_by_time, 2) if mean_by_time is not None else None)
print("sd  :", np.round(sd_by_time, 2) if sd_by_time is not None else None)
print("cv %%:", np.round(cv_by_time, 1) if cv_by_time is not None else None)
```

Three things at once here, and all three are the kind of mistake that does not
raise.

**`axis=0` goes down the columns**, giving one number per time, which is what
was asked. `axis=1` would give one number per animal. Both are eight or six
numbers of something plausible, and nothing tells you which you got except
reading the shape.

**`ddof=1`.** `np.nanstd` and `np.std` divide by n unless you say otherwise,
which makes the SD too small by a factor of the square root of n-1 over n: six
percent here, seven percent in the 0.5 h column where one sample is missing.
Pandas `.std()` already divides by n-1, so the same code in two libraries
gives two answers.

**`nanmean` rather than `mean`.** One `nan` anywhere in a column turns the
whole column's plain mean into `nan`, silently, and then every number
downstream is `nan` too.

Worth a look at what came out: the CV is between 52 and 66 percent at every
time. That is not assay noise, it is two dose groups pooled into one summary.
A summary statistic across groups that differ threefold by design is a number
with no owner, and the honest version is one summary per group, which is
question b2.

## Question a5. broadcasting with keepdims

```python
norm_matrix = conc_matrix / np.nanmax(conc_matrix, axis=1, keepdims=True)

print(np.round(norm_matrix, 3) if norm_matrix is not None else None)
```

`np.nanmax(conc_matrix, axis=1)` gives eight numbers of shape `(8,)`. Dividing
an `(8, 6)` array by an `(8,)` array does not do what you want: numpy lines
shapes up from the right, so it tries to match 8 against 6, and raises. That
error message is the lucky case.

`keepdims=True` gives shape `(8, 1)` instead, which lines up against `(8, 6)`
and broadcasts across the row, which is the whole point. The alternative
spelling is `[:, None]`, which does the same thing by inserting the axis
explicitly.

The unlucky case is a square matrix. If this study had eight sampling times as
well as eight animals, the wrong version would broadcast happily and give you a
transposed answer with no error at all. That is why the habit is to print
`.shape` rather than to trust the result.

## Question b1. filtering on two conditions

```python
post = pk[(pk["NOMINAL_H"] > 0) & pk["CONC_NG_ML"].notna()].copy()
n_post = int(len(post))

print("rows kept:", n_post)
```

Two conditions, each in its own brackets, joined with `&`. The brackets are not
optional: `&` binds tighter than `>`, so without them pandas sees something
else entirely and raises.

`and` does not work here. It tries to turn a whole Series into one True or
False and tells you the truth value is ambiguous. `&`, `|` and `~` are the
elementwise versions, and they are what pandas and numpy both want.

`.notna()` rather than `!= np.nan`, for the reason in a1.

The `.copy()` is not needed for the count, but it is a good habit the moment
you intend to add a column to the result, because otherwise pandas may warn you
about setting a value on a slice of something else, and that warning is
occasionally hiding a real bug.

56 rows in, 47 out: 8 pre dose samples and one haemolysed one.

## Question b2. groupby.agg with named outputs

```python
by_group = (post.groupby(["GROUP", "NOMINAL_H"])
            .agg(CONC_MEAN=("CONC_NG_ML", "mean"),
                 CONC_SD=("CONC_NG_ML", "std"))
            .reset_index())

print(by_group.to_string(index=False) if by_group is not None else None)
```

Named aggregation, `NEW=("COLUMN", "function")`, gets you the names you want in
one step. The older `.agg(["mean", "std"])` gives a two level column index that
you then have to flatten, which is three more lines and a different answer
every time you write it.

`.reset_index()` turns the group keys back into columns. Without it they are an
index, which prints the same and merges differently, and that difference shows
up two steps later as a merge that silently produces nothing.

Pandas `.std()` is the sample SD already, with n-1. This is the opposite
default from numpy, in the same analysis, and it is worth saying out loud every
time: numpy needs `ddof=1`, pandas does not.

Now read the two groups. At 2 h the low group is at 252 ng/mL and the high
group at 896, which is 3.6 fold for a 3 fold dose. With four animals a side
that is well inside the noise, and the right conclusion is that exposure looks
dose proportional, not that there is some mild nonlinearity.

## Question b3. a left merge, validated

```python
merged = pk.merge(demo, on="ANIMAL", how="left", validate="many_to_one")
n_missing_weight = int(merged.loc[merged["BODYWEIGHT_G"].isna(), "ANIMAL"].nunique())

print("rows: %s, animals with no weight: %s" % (
    len(merged) if merged is not None else None, n_missing_weight))
```

`how="left"` keeps all 56 sample rows whether or not the animal is in `demo`.
That is what you want: a missing weight is a fact about the data, and an inner
join would hide it by quietly dropping seven rows and leaving you with a
perfectly plausible 49.

`validate="many_to_one"` says many sample rows per animal, one demographics row
per animal. If `demo` ever gained a duplicate, this raises instead of giving
you 63 rows and a mean body weight that is wrong in a way nothing would reveal.
It costs nine characters.

`n_missing_weight` counts animals, so it needs `.nunique()` on the animal
column of the rows with no weight. Counting rows gives 7, which is the number
of samples affected and the answer to a different question.

Animal 6 is the one with no record, and it is in the high dose group, so any
weight normalised analysis would quietly drop a quarter of that group.

## Question b4. transform against agg

```python
frac_of_peak = pk["CONC_NG_ML"] / pk.groupby("ANIMAL")["CONC_NG_ML"].transform("max")
pk["FRAC_OF_PEAK"] = frac_of_peak

print(pk[pk["ANIMAL"] == 3].to_string(index=False) if frac_of_peak is not None else None)
```

`.agg("max")` gives you eight numbers, one per animal. `.transform("max")`
gives you 56, one per row, each animal's maximum repeated down that animal's
rows. Dividing by the first misaligns or raises; dividing by the second is the
answer.

That is the whole distinction, and it is the single most useful pandas idea in
week 2. Anything of the form "each value relative to its own group's
something" is a `transform`: baseline correction, percent of control, z scores
within a batch, change from screening.

The `nan` values stay `nan`, which is correct. Animal 3 has no 0.5 h sample, so
it has no 0.5 h fraction either, and inventing a zero there would be inventing
data.

## Question b5. nominal against actual times, and pivoting

```python
max_dev_min = float((pk["ACTUAL_H"] - pk["NOMINAL_H"]).abs().max() * 60.0)
wide = pk.pivot_table(values="CONC_NG_ML", index="ANIMAL", columns="NOMINAL_H")

print("worst deviation: %r min" % (max_dev_min,))
print(wide.round(1).to_string() if wide is not None else None)
```

Four minutes at the worst. That is a well run study, and it still matters:
four minutes on a 30 minute sample is 13 percent of the elapsed time, and the
concentration is climbing steeply there.

The rule is **group on nominal, calculate on actual**. Nominal times are
labels, and a mean concentration at 0.5 h is only meaningful if everything in
that column is labelled 0.5 h. Actual times are measurements, and anything that
uses the spacing between samples, an AUC or a terminal slope, has to use them
or it is systematically wrong.

`pivot_table` rather than `pivot` because `pivot_table` aggregates duplicates
and `pivot` raises on them. Here there are none, so both work, and the habit
worth having is `pivot_table` with `aggfunc="mean"` stated when there might be.

Notice what happened to the 0 h column: it is not there. Every pre dose value
is `nan`, and `pivot_table` drops an entirely empty column without mentioning
it. Hence seven nominal times in and six columns out.

## Question b6. an incidence table, and its denominators

```python
inc = pd.crosstab(ae["GROUP"], ae["EVENT"])
pct_high_event = 100.0 * float(inc.loc["high", "yes"]) / float(inc.loc["high"].sum())

print(inc.to_string() if inc is not None else None)
print("high dose incidence: %r pct" % (pct_high_event,))
```

`pd.crosstab(rows, columns)` counts the combinations, which is exactly what an
incidence table is. The equivalent `groupby(["GROUP", "EVENT"]).size().unstack()`
is the same thing in three more steps.

The denominator is the row total, not the study total, because incidence is
within group. Dividing by eight gives 37.5 percent and answers nothing anyone
asked.

Now the part that matters. Three of four high dose animals and none of four low
dose animals, so 75 percent against 0 percent. Write that down as "a clear dose
dependent signal" and you have overstated it badly. With four animals per
group, one animal is 25 percentage points. The exact test on a 2 by 2 of 3/4
against 0/4 does not reach conventional significance, and the honest sentence
is that the event occurred only at the high dose, in three of four animals, in
a group of four.

Zero of four is also not zero risk. It is consistent with a true rate anywhere
up to about 50 percent. Small denominators are the most common way a safety
table gets oversold, in both directions.

## Question c1. aggregating nested data to the animal first

```python
animal_level = (cells.groupby(["ANIMAL", "GROUP"])["INHIB_PCT"]
                .agg(MEAN_INHIB="mean", N_CELLS="size")
                .reset_index())
group_summary = (animal_level.groupby("GROUP")
                 .agg(MEAN=("MEAN_INHIB", "mean"),
                      SD=("MEAN_INHIB", "std"),
                      N=("MEAN_INHIB", "size"))
                 .reset_index())

print(animal_level.to_string(index=False) if animal_level is not None else None)
print("")
print(group_summary.to_string(index=False) if group_summary is not None else None)
```

Two steps, and the order is the whole point. Cells up to one number per animal,
then animals up to a group mean and SD.

Group by `["ANIMAL", "GROUP"]` together so each animal keeps its own group
label. Grouping by animal alone loses it, and then you need a merge to get it
back.

`N` comes out as 4 in each group, because there are four animals. Pool the
cells instead and `N` is 72 and 61, so three degrees of freedom become seventy
one, and every p value afterwards is a work of fiction.

The SD moves too, and not in the direction people expect. On the animal means
the low group's SD is 8.4 and the high group's is 3.1. On the pooled cells they
are 11.9 and 7.9, because cell to cell scatter within one animal is larger than
animal to animal scatter between them. So pooling reports a larger SD and a
much larger n at once, and the two do not cancel. The standard error of the low
dose mean falls from 4.2 to 1.4, a factor of three, entirely from the
bookkeeping.

Look at the low group's animal means: 32.5, 15.4, 16.3, 15.9. One animal is not
like the others. That is a real finding in a group of four and it is invisible
in any pooled summary.

## Question c2. a plot with SD error bars

```python
fig_c2, ax_c2 = plt.subplots(figsize=(5, 4))

_g = group_summary.sort_values("GROUP")
ax_c2.errorbar(_g["GROUP"], _g["MEAN"], yerr=_g["SD"], fmt="o", capsize=5)
ax_c2.set_xlabel("dose group")
ax_c2.set_ylabel("inhibition (pct), mean and SD of animal means")
ax_c2.set_title("Q-W1W2 c2")

plt.show()
```

`ax.errorbar(x, y, yerr=sd)` is the whole of it. `fmt="o"` draws markers rather
than joining two unrelated groups with a line, and `capsize` is cosmetic.

**The error bars are SD, not SEM.** SEM is the SD divided by the square root of
n, so with four animals it is half the size, and it shrinks further every time
somebody adds an animal even if the biology never changes. SD answers how much
animals differ. SEM answers how precisely you located the mean. A reader given
the SD and n can work out the SEM; a reader given the SEM cannot work backwards
without n, and n is usually in a different part of the figure legend or nowhere
at all.

The check inspects the axes rather than your code, which is why it can tell the
difference. It also catches error bars drawn from the pooled cell SD, which is
the c1 mistake arriving one step later.

Say what the error bars are in the axis label or the caption, every time. An
error bar with no stated meaning is decoration.

## Question c3. a semilog concentration time plot

```python
fig_c3, ax_c3 = plt.subplots(figsize=(6, 4))

for _name in ["low", "high"]:
    _sub = by_group[by_group["GROUP"] == _name].sort_values("NOMINAL_H")
    ax_c3.plot(_sub["NOMINAL_H"], _sub["CONC_MEAN"], "o-", label=_name)
ax_c3.set_yscale("log")
ax_c3.set_xlabel("nominal time (h)")
ax_c3.set_ylabel("mean concentration (ng/mL)")
ax_c3.set_title("Q-W1W2 c3")
ax_c3.legend(title="dose")

plt.show()
```

`ax.set_yscale("log")` after plotting, or `ax.semilogy(...)` instead of
`ax.plot(...)`. Either is fine.

Why a log y axis for concentration against time: first order elimination is
exponential, so on a log axis the terminal phase is a straight line, and your
eye is very good at straight lines. Curvature in the tail means more than one
compartment, or flip flop absorption, or an assay drifting into its lower
limit. None of that is visible on a linear axis, where the tail is squashed
onto the x axis.

The other half of it is that the two dose groups differ threefold. On a linear
axis the low group is a flat line along the bottom. On a log axis a threefold
difference is a constant vertical offset at every time point, and dose
proportionality becomes something you can see rather than something you have to
calculate.

What a log axis cannot do is show you a zero or a `nan`, because the log of
zero is minus infinity. Matplotlib drops those points without saying so, which
is another reason to know how many samples went into each mean.

## Question d1. SD, SEM and which n

```python
q_sd = "b"
```

**b.** The SD of the animal means.

Two separate decisions are hiding in that sentence, and the four options are
the four ways of getting one or both wrong.

**Which level.** The animal is the unit of replication, so the spread you
report is the spread between animals. Options a and d are computed from cells,
which are repeat measures of the same animal and tell you about your microscope
as much as about the drug.

**SD or SEM.** Options c and d are standard errors. They describe how precisely
the mean was located, which is a statement about your sample size, not about
the animals. The house rule is SD on the error bars, every time, with n stated
nearby.

Pandas `.std()` is already the sample SD with n-1, so there is no `ddof` to
remember in this particular spelling. In numpy there would be.

## Question d2. which axis is which

```python
q_axis = "b"
```

**b.** `axis=1`.

The way to remember it: **the axis you name is the one that disappears.**
`axis=1` is the time axis, so the times collapse and you are left with one
number per animal. `axis=0` collapses the animals and leaves one number per
time, which is option a and is six numbers.

Option c reduces everything to a single number, the largest concentration in
the study. Option d does the same but keeps the shape as `(1, 1)`, which is
what `keepdims` is for: it preserves the number of dimensions so the result can
broadcast back against the original, as in question a5.

The reason this matters more than it looks: with eight animals and six times,
getting it wrong gives you six numbers where you wanted eight, and the next
line raises. With eight animals and eight times, it gives you eight numbers of
the wrong thing and nothing raises at all.

## Question d3. what validate is for

```python
q_validate = "c"
```

**c.** `many_to_one`. Many rows on the left, one row on the right, in that
order, because the left frame is the one you called `.merge` on.

Option b is the same statement backwards and would raise immediately on this
data, which at least tells you something. Option a claims both sides are unique
and would also raise, for the same reason.

Option d is not wrong, and checking `len()` before and after every merge is a
genuine house rule. It is just weaker: it tells you that something changed, not
what, and only if you remember to look. `validate` states the intent in the
line that does the work and raises at the moment the assumption breaks. Do
both.

What `validate` does not check is whether every key on the left has a match on
the right. Animal 6 is missing from `demo` entirely and the merge is perfectly
happy. That is what `how="left"` plus a count of nulls afterwards is for, which
was question b3.
