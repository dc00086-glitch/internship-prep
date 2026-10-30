QUIZ_ID = "Q-W1W2"
SLUG = "week01_02_quiz"
TITLE = "Test: weeks 1 and 2"
TOPIC = "numpy, pandas and matplotlib, everything from weeks 1 and 2"
WEEKS = [1, 2]
MINUTES = 60

INTRO_MD = '''
## What this is

Seventeen questions covering everything in weeks 1 and 2: numpy arrays and
reductions, pandas filtering, grouping, merging and reshaping, matplotlib, and
the judgement calls that go with them. All of it runs on one small study,
generated in the setup cell below.

About an hour. The point is not to get seventeen out of seventeen. The point is
to find out which of these you can do without looking, which you can do with
the reference open, and which you cannot do yet. The last group is the useful
one and the scoring cell names them for you.

## The rules

**Book open.** Use the function reference, the foundations notebook, and your
own earlier solutions. That is what you will do at work. What you should not do
is open the solution to this test, which is in `solutions/` and is far too
easy to reach.

**One hour, roughly.** If a question takes more than five minutes, write down
where you got stuck, put `None` back, and move on. A question you ran out of
time on and a question you could not start are different pieces of
information, and only you can tell them apart.

**Sections run in order.** A few later questions use variables from earlier
ones, and each says so when it does.

**Scoring.** Run `check()` for a score out of seventeen, a subtotal per
section, and a revision pointer for everything you missed. It tells you what
is wrong without telling you the answer, exactly like the daily problems.

**Then run `review()`.** That is the worked review: every question, the model
answer, and why it is that rather than the thing people reach for first. Read
it whether you scored four or seventeen, because several of the questions have
a right answer for a reason that is not obvious from the fact that it passed.
'''

DATA_CODE = '''
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
'''

SECTIONS = [
    dict(key="A", title="numpy", intro="Five questions on plain arrays. "
         "No pandas needed, and none of them wants a loop."),
    dict(key="B", title="pandas", intro="Six questions on the sample table and "
         "the two lookup tables."),
    dict(key="C", title="matplotlib and nested data", intro="Three questions "
         "where the figure is the answer, or the aggregation behind it is."),
    dict(key="D", title="judgement", intro="Three questions with no code to "
         "write. Set each variable to the letter of the right answer, as a "
         "string, for example \"a\"."),
]

ITEMS = [
    # ------------------------------------------------------------------ numpy
    dict(
        id="a1", section="A", points=1,
        topic="finding and masking the samples you cannot use",
        revise="W1D1 part a, and Missing values in the revision sheet",
        prompt_md='''
`conc_single` holds animal 3's seven concentrations and `time_single` the seven
nominal times. Two of the concentrations are `nan`.

Work out:

- `n_missing`, how many concentrations are `nan`, as an integer,
- `mask_ok`, a boolean array that is True where the concentration is usable,
- `conc_ok` and `time_ok`, the concentrations and their times, keeping only the
  usable samples.
''',
        starter='''
n_missing = None   # YOUR CODE
mask_ok = None     # YOUR CODE
conc_ok = None     # YOUR CODE
time_ok = None     # YOUR CODE

print("missing:", n_missing)
print("kept   :", conc_ok)
print("at     :", time_ok)
''',
        solution='''
n_missing = int(np.isnan(conc_single).sum())
mask_ok = ~np.isnan(conc_single)
conc_ok = conc_single[mask_ok]
time_ok = time_single[mask_ok]

print("missing:", n_missing)
print("kept   :", conc_ok)
print("at     :", time_ok)
''',
        review_md='''
`np.isnan` gives you the boolean array, `~` flips it, and the same mask indexes
both arrays so the times stay lined up with the concentrations. Two samples are
missing for two different reasons: the pre dose one was never going to have
drug in it, and the 0.5 h one was haemolysed. Both are `nan` and both have to
go, but only the second one is a loss.

The mistake worth naming: `conc_single != np.nan` is always True, because `nan`
is not equal to anything, including itself. That is the whole reason `np.isnan`
exists.
''',
    ),
    dict(
        id="a2", section="A", points=1,
        topic="peak and time of peak",
        revise="W1D1 part b, and Reductions in the revision sheet",
        prompt_md='''
From the usable samples only, give `cmax`, the highest concentration, and
`tmax`, the time at which it occurred. Both as plain numbers.
''',
        starter='''
cmax = None   # YOUR CODE
tmax = None   # YOUR CODE

print("Cmax %r ng/mL at %r h" % (cmax, tmax))
''',
        solution='''
cmax = float(conc_ok.max())
tmax = float(time_ok[np.argmax(conc_ok)])

print("Cmax %r ng/mL at %r h" % (cmax, tmax))
''',
        review_md='''
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
''',
    ),
    dict(
        id="a3", section="A", points=1,
        topic="terminal slope and half-life",
        revise="W1D1 part c, and Fitting a line to logs in the revision sheet",
        prompt_md='''
Take the **last three** usable samples and fit a straight line to the log of
the concentration against time.

Give `lambda_z`, the terminal elimination rate constant in per hour, as a
positive number, and `t_half`, the terminal half-life in hours.
''',
        starter='''
lambda_z = None   # YOUR CODE
t_half = None     # YOUR CODE

print("lambda_z %r per h, half-life %r h" % (lambda_z, t_half))
''',
        solution='''
_slope, _intercept = np.polyfit(time_ok[-3:], np.log(conc_ok[-3:]), 1)
lambda_z = float(-_slope)
t_half = float(np.log(2) / lambda_z)

print("lambda_z %r per h, half-life %r h" % (lambda_z, t_half))
''',
        review_md='''
The fitted slope is negative, because the concentration falls, and `lambda_z`
is defined as its positive counterpart. So the minus sign is not a detail, it
is the definition.

`np.log` is the natural log. `np.log10` would give you a slope smaller by a
factor of 2.303 and a half-life wrong by the same factor, which is a mistake
that looks plausible on a plot and is not.

The half-life carries the `ln(2)`. `1 / lambda_z` is also a meaningful number,
the mean residence time for a one compartment model, but it is 44 percent
larger than the half-life and they are not interchangeable.
''',
    ),
    dict(
        id="a4", section="A", points=1,
        topic="axis reductions, and the sample SD",
        revise="W1D2 parts a and b, and Reductions in the revision sheet",
        prompt_md='''
`conc_matrix` has one row per animal and one column per sampling time, and one
`nan` in it.

Give three arrays of six numbers each, one value per sampling time, across all
eight animals:

- `mean_by_time`, the mean concentration,
- `sd_by_time`, the **sample** standard deviation,
- `cv_by_time`, the coefficient of variation as a percentage.

The `nan` has to be ignored rather than propagated.
''',
        starter='''
mean_by_time = None   # YOUR CODE
sd_by_time = None     # YOUR CODE
cv_by_time = None     # YOUR CODE

print("mean:", np.round(mean_by_time, 2) if mean_by_time is not None else None)
print("sd  :", np.round(sd_by_time, 2) if sd_by_time is not None else None)
print("cv %%:", np.round(cv_by_time, 1) if cv_by_time is not None else None)
''',
        solution='''
mean_by_time = np.nanmean(conc_matrix, axis=0)
sd_by_time = np.nanstd(conc_matrix, axis=0, ddof=1)
cv_by_time = 100.0 * sd_by_time / mean_by_time

print("mean:", np.round(mean_by_time, 2) if mean_by_time is not None else None)
print("sd  :", np.round(sd_by_time, 2) if sd_by_time is not None else None)
print("cv %%:", np.round(cv_by_time, 1) if cv_by_time is not None else None)
''',
        review_md='''
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
''',
    ),
    dict(
        id="a5", section="A", points=1,
        topic="broadcasting with keepdims",
        revise="W1D2 part c, and Broadcasting in the revision sheet",
        prompt_md='''
Give `norm_matrix`, the same shape as `conc_matrix`, where every animal's
profile has been divided by **that animal's own** highest concentration. Each
row should therefore peak at 1.

One expression, no loop.
''',
        starter='''
norm_matrix = None   # YOUR CODE

print(np.round(norm_matrix, 3) if norm_matrix is not None else None)
''',
        solution='''
norm_matrix = conc_matrix / np.nanmax(conc_matrix, axis=1, keepdims=True)

print(np.round(norm_matrix, 3) if norm_matrix is not None else None)
''',
        review_md='''
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
''',
    ),
    # ----------------------------------------------------------------- pandas
    dict(
        id="b1", section="B", points=1,
        topic="filtering on two conditions",
        revise="W1D3 part a, and Filtering in the revision sheet",
        prompt_md='''
From `pk`, build `post`, the rows that are both **after** dosing and have a
concentration, and give `n_post`, how many rows that is.
''',
        starter='''
post = None     # YOUR CODE
n_post = None   # YOUR CODE

print("rows kept:", n_post)
''',
        solution='''
post = pk[(pk["NOMINAL_H"] > 0) & pk["CONC_NG_ML"].notna()].copy()
n_post = int(len(post))

print("rows kept:", n_post)
''',
        review_md='''
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
''',
    ),
    dict(
        id="b2", section="B", points=1,
        topic="groupby.agg with named outputs",
        revise="W1D3 part b, and Grouping in the revision sheet",
        prompt_md='''
From `post`, build `by_group`: the mean and standard deviation of
`CONC_NG_ML` for every combination of `GROUP` and `NOMINAL_H`.

The result should have `GROUP` and `NOMINAL_H` as ordinary columns, not as an
index, and the two summary columns must be named exactly `CONC_MEAN` and
`CONC_SD`.
''',
        starter='''
by_group = None   # YOUR CODE

print(by_group.to_string(index=False) if by_group is not None else None)
''',
        solution='''
by_group = (post.groupby(["GROUP", "NOMINAL_H"])
            .agg(CONC_MEAN=("CONC_NG_ML", "mean"),
                 CONC_SD=("CONC_NG_ML", "std"))
            .reset_index())

print(by_group.to_string(index=False) if by_group is not None else None)
''',
        review_md='''
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
''',
    ),
    dict(
        id="b3", section="B", points=1,
        topic="a left merge, validated",
        revise="W1D3 part c and W2D2, and Merges in the revision sheet",
        prompt_md='''
Join the body weights onto the sample table. Build `merged`, every row of `pk`
with its animal's `SEX` and `BODYWEIGHT_G` attached, and give
`n_missing_weight`, the number of **animals** that have no weight.

Pass the `validate` argument that states what you expect of the two key
columns, so that a duplicated demographics row would raise rather than
silently multiply your data.
''',
        starter='''
merged = None             # YOUR CODE
n_missing_weight = None   # YOUR CODE

print("rows: %s, animals with no weight: %s" % (
    len(merged) if merged is not None else None, n_missing_weight))
''',
        solution='''
merged = pk.merge(demo, on="ANIMAL", how="left", validate="many_to_one")
n_missing_weight = int(merged.loc[merged["BODYWEIGHT_G"].isna(), "ANIMAL"].nunique())

print("rows: %s, animals with no weight: %s" % (
    len(merged) if merged is not None else None, n_missing_weight))
''',
        review_md='''
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
''',
    ),
    dict(
        id="b4", section="B", points=1,
        topic="transform against agg",
        revise="W2D1, and Transform against agg in the revision sheet",
        prompt_md='''
Add a column to `pk` called `FRAC_OF_PEAK`: each concentration divided by
**that animal's own** highest concentration, so every animal's profile peaks at
1. Put the resulting Series in `frac_of_peak` as well.

This is the pandas version of question a5, and the whole question is which
groupby method broadcasts a per animal number back onto every one of that
animal's rows.
''',
        starter='''
frac_of_peak = None   # YOUR CODE, and also assign it to pk["FRAC_OF_PEAK"]

print(pk[pk["ANIMAL"] == 3].to_string(index=False) if frac_of_peak is not None else None)
''',
        solution='''
frac_of_peak = pk["CONC_NG_ML"] / pk.groupby("ANIMAL")["CONC_NG_ML"].transform("max")
pk["FRAC_OF_PEAK"] = frac_of_peak

print(pk[pk["ANIMAL"] == 3].to_string(index=False) if frac_of_peak is not None else None)
''',
        review_md='''
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
''',
    ),
    dict(
        id="b5", section="B", points=1,
        topic="nominal against actual times, and pivoting",
        revise="W2D3, and Reshaping in the revision sheet",
        prompt_md='''
Two parts, both about the difference between the time a sample was meant to be
taken and the time it was.

- `max_dev_min`, the largest absolute difference between `ACTUAL_H` and
  `NOMINAL_H` anywhere in the study, **in minutes**,
- `wide`, a table of concentrations with one row per animal and one column per
  **nominal** time, built with `pivot_table`.
''',
        starter='''
max_dev_min = None   # YOUR CODE
wide = None          # YOUR CODE

print("worst deviation: %r min" % (max_dev_min,))
print(wide.round(1).to_string() if wide is not None else None)
''',
        solution='''
max_dev_min = float((pk["ACTUAL_H"] - pk["NOMINAL_H"]).abs().max() * 60.0)
wide = pk.pivot_table(values="CONC_NG_ML", index="ANIMAL", columns="NOMINAL_H")

print("worst deviation: %r min" % (max_dev_min,))
print(wide.round(1).to_string() if wide is not None else None)
''',
        review_md='''
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
''',
    ),
    dict(
        id="b6", section="B", points=1,
        topic="an incidence table, and its denominators",
        revise="W2D4, and Crosstabs in the revision sheet",
        prompt_md='''
From `ae`, build `inc`, a table of counts with one row per `GROUP` and one
column per `EVENT` value, using `pd.crosstab`.

Then give `pct_high_event`, the percentage of **high dose** animals that had
the event.
''',
        starter='''
inc = None              # YOUR CODE
pct_high_event = None   # YOUR CODE

print(inc.to_string() if inc is not None else None)
print("high dose incidence: %r pct" % (pct_high_event,))
''',
        solution='''
inc = pd.crosstab(ae["GROUP"], ae["EVENT"])
pct_high_event = 100.0 * float(inc.loc["high", "yes"]) / float(inc.loc["high"].sum())

print(inc.to_string() if inc is not None else None)
print("high dose incidence: %r pct" % (pct_high_event,))
''',
        review_md='''
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
''',
    ),
    # ------------------------------------------------- matplotlib, nested data
    dict(
        id="c1", section="C", points=1,
        topic="aggregating nested data to the animal first",
        revise="W1D5 parts a and b, and Nested data in the revision sheet",
        prompt_md='''
`cells` holds between 9 and 22 cells for each animal. The unit of replication
is the animal, not the cell.

Build:

- `animal_level`, one row per animal, with columns `ANIMAL`, `GROUP`,
  `MEAN_INHIB`, the mean of that animal's own cells, and `N_CELLS`, how many
  cells it had,
- `group_summary`, one row per group, with columns `GROUP`, `MEAN`, `SD` and
  `N`, computed from the animal means rather than from the cells.

Both with the group keys as ordinary columns.
''',
        starter='''
animal_level = None    # YOUR CODE
group_summary = None   # YOUR CODE

print(animal_level.to_string(index=False) if animal_level is not None else None)
print("")
print(group_summary.to_string(index=False) if group_summary is not None else None)
''',
        solution='''
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
''',
        review_md='''
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
''',
    ),
    dict(
        id="c2", section="C", points=1,
        topic="a plot with SD error bars",
        revise="W1D5 part c, and Error bars in the revision sheet",
        prompt_md='''
Plot `group_summary` from the question above: the group mean inhibition with
error bars, into `fig_c2` and `ax_c2` from `plt.subplots()`.

The error bars are the standard deviation of the animal means. Label both axes.

The check reads the means and the error bar lengths back off the axes, so what
matters is what you drew, not how you drew it.
''',
        starter='''
fig_c2, ax_c2 = plt.subplots(figsize=(5, 4))

# YOUR CODE: draw the group means with SD error bars, and label the axes

plt.show()
''',
        solution='''
fig_c2, ax_c2 = plt.subplots(figsize=(5, 4))

_g = group_summary.sort_values("GROUP")
ax_c2.errorbar(_g["GROUP"], _g["MEAN"], yerr=_g["SD"], fmt="o", capsize=5)
ax_c2.set_xlabel("dose group")
ax_c2.set_ylabel("inhibition (pct), mean and SD of animal means")
ax_c2.set_title("Q-W1W2 c2")

plt.show()
''',
        review_md='''
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
''',
    ),
    dict(
        id="c3", section="C", points=1,
        topic="a semilog concentration time plot",
        revise="W2D5, and Log scales in the revision sheet",
        prompt_md='''
Plot the mean concentration against nominal time for both dose groups on the
same axes, into `fig_c3` and `ax_c3`, with a **logarithmic y axis**.

Use `by_group` from question b2. One line per group, markers on the points,
axes labelled, and a legend.
''',
        starter='''
fig_c3, ax_c3 = plt.subplots(figsize=(6, 4))

# YOUR CODE: one line per group, log y axis, labels and a legend

plt.show()
''',
        solution='''
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
''',
        review_md='''
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
''',
    ),
    # -------------------------------------------------------------- judgement
    dict(
        id="d1", section="D", points=1,
        topic="SD, SEM and which n",
        revise="the house rules, and Error bars in the revision sheet",
        prompt_md='''
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
''',
        starter='''
q_sd = None   # YOUR CODE: "a", "b", "c" or "d"
''',
        solution='''
q_sd = "b"
''',
        review_md='''
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
''',
    ),
    dict(
        id="d2", section="D", points=1,
        topic="which axis is which",
        revise="W1D2, and Reductions in the revision sheet",
        prompt_md='''
`conc_matrix` has shape `(8, 6)`: eight animals down the rows, six sampling
times across the columns.

You want **each animal's own peak concentration**, so eight numbers.

- **a.** `np.nanmax(conc_matrix, axis=0)`
- **b.** `np.nanmax(conc_matrix, axis=1)`
- **c.** `np.nanmax(conc_matrix)`
- **d.** `np.nanmax(conc_matrix, keepdims=True)`

Set `q_axis` to the letter.
''',
        starter='''
q_axis = None   # YOUR CODE: "a", "b", "c" or "d"
''',
        solution='''
q_axis = "b"
''',
        review_md='''
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
''',
    ),
    dict(
        id="d3", section="D", points=1,
        topic="what validate is for",
        revise="W2D2, and Merges in the revision sheet",
        prompt_md='''
You are joining `pk`, which has seven rows per animal, onto `demo`, which
should have exactly one row per animal.

Which `validate` argument states that, so that a duplicated demographics row
would raise instead of multiplying your sample rows?

- **a.** `validate="one_to_one"`
- **b.** `validate="one_to_many"`
- **c.** `validate="many_to_one"`
- **d.** no `validate` argument, and check `len()` afterwards instead

Set `q_validate` to the letter.
''',
        starter='''
q_validate = None   # YOUR CODE: "a", "b", "c" or "d"
''',
        solution='''
q_validate = "c"
''',
        review_md='''
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
''',
    ),
]


# Aliases so the shared tooling can treat a test like a problem.
PID = QUIZ_ID
PARTS = ITEMS
NB_PATH = "quizzes/%s.ipynb" % SLUG
WEEK = 0
DAY = 0


def build_expected(ns):
    import numpy as np
    import pandas as pd

    pk = ns["pk"]
    conc_single = np.asarray(ns["conc_single"], dtype=float)
    time_single = np.asarray(ns["time_single"], dtype=float)
    conc_matrix = np.asarray(ns["conc_matrix"], dtype=float)
    cells = ns["cells"]
    ae = ns["ae"]

    ok = ~np.isnan(conc_single)
    conc_ok = conc_single[ok]
    time_ok = time_single[ok]
    slope = float(np.polyfit(time_ok[-3:], np.log(conc_ok[-3:]), 1)[0])
    lam = -slope

    mean_by_time = np.nanmean(conc_matrix, axis=0)
    sd_pop = np.nanstd(conc_matrix, axis=0)
    sd_sample = np.nanstd(conc_matrix, axis=0, ddof=1)
    peaks = np.nanmax(conc_matrix, axis=1, keepdims=True)

    post = pk[(pk["NOMINAL_H"] > 0) & pk["CONC_NG_ML"].notna()]
    by_group = (post.groupby(["GROUP", "NOMINAL_H"])
                .agg(CONC_MEAN=("CONC_NG_ML", "mean"), CONC_SD=("CONC_NG_ML", "std"))
                .reset_index()
                .sort_values(["GROUP", "NOMINAL_H"]))
    merged = pk.merge(ns["demo"], on="ANIMAL", how="left", validate="many_to_one")
    frac = pk["CONC_NG_ML"] / pk.groupby("ANIMAL")["CONC_NG_ML"].transform("max")
    wide = pk.pivot_table(values="CONC_NG_ML", index="ANIMAL", columns="NOMINAL_H")

    inc = pd.crosstab(ae["GROUP"], ae["EVENT"])

    animal_level = (cells.groupby(["ANIMAL", "GROUP"])["INHIB_PCT"]
                    .agg(MEAN_INHIB="mean", N_CELLS="size").reset_index()
                    .sort_values("ANIMAL"))
    group_summary = (animal_level.groupby("GROUP")
                     .agg(MEAN=("MEAN_INHIB", "mean"), SD=("MEAN_INHIB", "std"),
                          N=("MEAN_INHIB", "size")).reset_index().sort_values("GROUP"))
    pooled = (cells.groupby("GROUP")["INHIB_PCT"]
              .agg(MEAN="mean", SD="std", N="size").reset_index().sort_values("GROUP"))

    return {
        "n_missing": int((~ok).sum()),
        "mask_ok": [bool(v) for v in ok],
        "conc_ok": [float(v) for v in conc_ok],
        "time_ok": [float(v) for v in time_ok],
        "n_ok": int(ok.sum()),
        "n_all": int(conc_single.size),
        "cmax": float(conc_ok.max()),
        "tmax": float(time_ok[int(np.argmax(conc_ok))]),
        "argmax_ok": int(np.argmax(conc_ok)),
        "lambda_z": lam,
        "t_half": float(np.log(2) / lam),
        "mrt": float(1.0 / lam),
        "lambda_z_log10": float(-np.polyfit(time_ok[-3:], np.log10(conc_ok[-3:]), 1)[0]),
        "lambda_z_all": float(-np.polyfit(time_ok, np.log(conc_ok), 1)[0]),
        "mean_by_time": [float(v) for v in mean_by_time],
        "sd_by_time": [float(v) for v in sd_sample],
        "sd_population": [float(v) for v in sd_pop],
        "cv_by_time": [float(v) for v in (100.0 * sd_sample / mean_by_time)],
        "mean_by_animal": [float(v) for v in np.nanmean(conc_matrix, axis=1)],
        "peaks": [float(v) for v in peaks.ravel()],
        "norm_matrix": [[None if np.isnan(v) else float(v) for v in row]
                        for row in (conc_matrix / peaks)],
        "matrix_shape": list(conc_matrix.shape),
        "n_post": int(len(post)),
        "n_pk": int(len(pk)),
        "by_group_rows": int(len(by_group)),
        "by_group_mean": [float(v) for v in by_group["CONC_MEAN"]],
        "by_group_sd": [float(v) for v in by_group["CONC_SD"]],
        "by_group_sd_pop": [float(v) for v in
                            post.groupby(["GROUP", "NOMINAL_H"])["CONC_NG_ML"]
                            .std(ddof=0).sort_index().to_numpy()],
        "by_group_keys": ["%s|%g" % (g, t) for g, t in
                          zip(by_group["GROUP"], by_group["NOMINAL_H"])],
        "merged_rows": int(len(merged)),
        "merged_cols": [str(c) for c in merged.columns],
        "n_missing_weight": int(merged.loc[merged["BODYWEIGHT_G"].isna(), "ANIMAL"].nunique()),
        "n_missing_weight_rows": int(merged["BODYWEIGHT_G"].isna().sum()),
        "frac_len": int(frac.size),
        "frac_notna": int(frac.notna().sum()),
        "frac_animal3": [None if pd.isna(v) else float(v)
                         for v in frac[pk["ANIMAL"] == 3].to_numpy()],
        "frac_sum": float(frac.sum()),
        "n_groups_animal": int(pk["ANIMAL"].nunique()),
        "max_dev_min": float((pk["ACTUAL_H"] - pk["NOMINAL_H"]).abs().max() * 60.0),
        "max_dev_h": float((pk["ACTUAL_H"] - pk["NOMINAL_H"]).abs().max()),
        "wide_shape": list(wide.shape),
        "wide_cols": [float(c) for c in wide.columns],
        "wide_first_row": [None if pd.isna(v) else float(v) for v in wide.iloc[0].to_numpy()],
        "inc_shape": list(inc.shape),
        "inc_index": [str(v) for v in inc.index],
        "inc_cols": [str(v) for v in inc.columns],
        "inc_values": [[int(v) for v in row] for row in inc.to_numpy()],
        "pct_high_event": float(100.0 * inc.loc["high", "yes"] / inc.loc["high"].sum()),
        "pct_high_of_study": float(100.0 * inc.loc["high", "yes"] / len(ae)),
        "animal_rows": int(len(animal_level)),
        "n_cells_total": int(len(cells)),
        "animal_ids": [int(v) for v in animal_level["ANIMAL"]],
        "animal_groups": [str(v) for v in animal_level["GROUP"]],
        "animal_mean": [float(v) for v in animal_level["MEAN_INHIB"]],
        "animal_median": [float(v) for v in
                          cells.groupby(["ANIMAL", "GROUP"])["INHIB_PCT"].median()
                          .reset_index().sort_values("ANIMAL")["INHIB_PCT"]],
        "animal_ncells": [int(v) for v in animal_level["N_CELLS"]],
        "group_names": [str(v) for v in group_summary["GROUP"]],
        "group_mean": [float(v) for v in group_summary["MEAN"]],
        "group_sd": [float(v) for v in group_summary["SD"]],
        "group_n": [int(v) for v in group_summary["N"]],
        "group_sem": [float(s) / float(n) ** 0.5
                      for s, n in zip(group_summary["SD"], group_summary["N"])],
        "pooled_mean": [float(v) for v in pooled["MEAN"]],
        "pooled_sd": [float(v) for v in pooled["SD"]],
        "pooled_n": [int(v) for v in pooled["N"]],
        "q_sd": "b",
        "q_axis": "b",
        "q_validate": "c",
    }


CHECK_CODE = '''
def _yerr_values(ax):
    """The half length of every error bar drawn on the axes."""
    import matplotlib.container as _mcont
    out = []
    for cont in ax.containers:
        eb = cont if isinstance(cont, _mcont.ErrorbarContainer) else None
        if eb is None and getattr(cont, "errorbar", None) is not None:
            eb = cont.errorbar
        if eb is None:
            continue
        lines = getattr(eb, "lines", None)
        if not lines or len(lines) < 3 or not lines[2]:
            continue
        for col in lines[2]:
            try:
                segs = col.get_segments()
            except Exception:
                continue
            for seg in segs:
                seg = _np.asarray(seg, dtype=float)
                if seg.shape[0] >= 2 and seg.shape[1] >= 2:
                    out.append(abs(float(seg[-1, 1]) - float(seg[0, 1])) / 2.0)
    return out


def _plotted_means(ax):
    """The y values of the first error bar or bar container on the axes."""
    import matplotlib.container as _mcont
    for cont in ax.containers:
        if isinstance(cont, _mcont.ErrorbarContainer):
            line = cont.lines[0] if cont.lines else None
            if line is not None:
                try:
                    return [float(v) for v in line.get_ydata()]
                except Exception:
                    return []
        if isinstance(cont, _mcont.BarContainer):
            try:
                return [float(v) for v in cont.datavalues]
            except Exception:
                return []
    return []


def _part_a1():
    n_missing = _need("n_missing")
    mask_ok = _need("mask_ok")
    conc_ok = _need("conc_ok")
    time_ok = _need("time_ok")

    got = _as_scalar(n_missing)
    if got is None:
        return "`n_missing` should be a single whole number."
    if int(got) == _EXP["n_ok"]:
        return "`n_missing` counted the usable samples rather than the missing ones."
    if int(got) != _EXP["n_missing"]:
        return "`n_missing` is not the number of nan concentrations."

    arr = _as_array(mask_ok)
    if arr is None or arr.shape != (_EXP["n_all"],):
        return ("`mask_ok` should be a boolean array with one entry per sample, so %d of "
                "them." % _EXP["n_all"])
    flags = [bool(v) for v in arr]
    if flags == [not v for v in _EXP["mask_ok"]]:
        return ("`mask_ok` is inverted. True should mean the sample IS usable, so the mask "
                "is ~np.isnan(conc_single).")
    if flags != _EXP["mask_ok"]:
        return "`mask_ok` does not flag the usable samples."

    for name, val, want in (("conc_ok", conc_ok, _EXP["conc_ok"]),
                            ("time_ok", time_ok, _EXP["time_ok"])):
        arr = _as_array(val)
        if arr is None:
            return "`%s` should be an array of numbers." % name
        if arr.size == _EXP["n_all"]:
            return ("`%s` still has all %d samples. Index it with your mask."
                    % (name, _EXP["n_all"]))
        if arr.size != _EXP["n_ok"]:
            return ("`%s` has %d values but %d samples are usable."
                    % (name, arr.size, _EXP["n_ok"]))
        if not _arr_close(arr, want, rtol=1e-9):
            return "`%s` does not hold the right values." % name
    return None


def _part_a2():
    cmax = _need("cmax")
    tmax = _need("tmax")
    for name, val in (("cmax", cmax), ("tmax", tmax)):
        if _as_scalar(val) is None:
            return "`%s` should be a single number." % name

    if not _num_close(_as_scalar(cmax), _EXP["cmax"], rtol=1e-9):
        return "`cmax` is not the highest usable concentration."
    got = _as_scalar(tmax)
    if _num_close(got, float(_EXP["argmax_ok"])):
        return ("`tmax` is the position of the peak, not the time. Use it to index the "
                "time array.")
    if not _num_close(got, _EXP["tmax"], rtol=1e-9):
        return "`tmax` is not the time at which the peak occurred."
    return None


def _part_a3():
    lam = _need("lambda_z")
    th = _need("t_half")
    for name, val in (("lambda_z", lam), ("t_half", th)):
        if _as_scalar(val) is None:
            return "`%s` should be a single number." % name

    got = _as_scalar(lam)
    if got < 0:
        return ("`lambda_z` is negative. The fitted slope is negative and lambda_z is "
                "defined as its positive counterpart.")
    if _num_close(got, _EXP["lambda_z_log10"], rtol=1e-4):
        return ("`lambda_z` came from log10. The terminal rate constant is defined on the "
                "natural log, and the two differ by a factor of 2.303.")
    if _num_close(got, _EXP["lambda_z_all"], rtol=1e-4):
        return ("`lambda_z` used every usable sample, including the absorption phase, which "
                "drags the slope shallower. The task asks for the last three.")
    if not _num_close(got, _EXP["lambda_z"], rtol=1e-6):
        return "`lambda_z` is not the negative slope of log concentration on time."

    got = _as_scalar(th)
    if _num_close(got, _EXP["mrt"], rtol=1e-6):
        return "`t_half` is 1 / lambda_z. A half-life carries a factor of ln(2)."
    if not _num_close(got, _EXP["t_half"], rtol=1e-6):
        return "`t_half` is not ln(2) divided by lambda_z."
    return None


def _part_a4():
    mean_by_time = _need("mean_by_time")
    sd_by_time = _need("sd_by_time")
    cv_by_time = _need("cv_by_time")

    n_times = _EXP["matrix_shape"][1]
    for name, val in (("mean_by_time", mean_by_time), ("sd_by_time", sd_by_time),
                      ("cv_by_time", cv_by_time)):
        arr = _as_array(val)
        if arr is None:
            return "`%s` should be an array of numbers." % name
        if arr.shape != (n_times,):
            if arr.shape == (_EXP["matrix_shape"][0],):
                return ("`%s` has one value per ANIMAL, so the reduction collapsed the wrong "
                        "axis. axis=0 goes down the columns and leaves one value per time."
                        % name)
            return ("`%s` should hold one value per sampling time, so %d numbers."
                    % (name, n_times))
        if bool(_np.isnan(arr).any()):
            return ("`%s` contains nan. One missing sample turns a plain mean or std into "
                    "nan for that whole column, which is what np.nanmean and np.nanstd are "
                    "for." % name)

    if not _arr_close(_as_array(mean_by_time), _EXP["mean_by_time"], rtol=1e-9):
        return "`mean_by_time` is not the mean concentration at each sampling time."
    got = _as_array(sd_by_time)
    if _arr_close(got, _EXP["sd_population"], rtol=1e-6):
        return ("`sd_by_time` divides by n. numpy needs ddof=1 for the sample SD, which is "
                "what is reported. Pandas .std() already does this, numpy does not.")
    if not _arr_close(got, _EXP["sd_by_time"], rtol=1e-9):
        return "`sd_by_time` is not the sample standard deviation at each time."
    got = _as_array(cv_by_time)
    if _arr_close(got, [v / 100.0 for v in _EXP["cv_by_time"]], rtol=1e-6):
        return "`cv_by_time` is a fraction. The task asks for a percentage."
    if not _arr_close(got, _EXP["cv_by_time"], rtol=1e-9):
        return "`cv_by_time` is not 100 * sd / mean at each time."
    return None


def _part_a5():
    norm = _need("norm_matrix")
    arr = _as_array(norm)
    if arr is None:
        return "`norm_matrix` should be an array."
    want_shape = tuple(_EXP["matrix_shape"])
    if arr.shape != want_shape:
        if arr.shape == want_shape[::-1]:
            return ("`norm_matrix` is transposed. Dividing by a shape (8,) array lines the "
                    "axes up from the right, which is not what you want here.")
        return ("`norm_matrix` has shape %s and it should have shape %s, the same as "
                "conc_matrix." % (arr.shape, want_shape))
    want = [[_np.nan if v is None else v for v in row] for row in _EXP["norm_matrix"]]
    if not _arr_close(arr, want, rtol=1e-9):
        peak_all = _np.nanmax(_np.asarray(_EXP["norm_matrix"], dtype=float))
        row_max = _np.nanmax(arr, axis=1)
        if _np.nanmax(row_max) > 1.0001 or _np.nanmin(row_max) < 0.9999:
            return ("each row of `norm_matrix` should peak at exactly 1, and yours does not. "
                    "That means the divisor was not each animal's own maximum.")
        return "`norm_matrix` is not conc_matrix divided by each animal's own peak."
    return None


def _part_b1():
    post = _need("post")
    n_post = _need("n_post")
    cols = _cols_of(post)
    if cols is None:
        return "`post` should be a pandas DataFrame."
    if len(post) == _EXP["n_pk"]:
        return ("`post` still has every row of pk. Both conditions have to be applied: "
                "after dosing, and with a concentration.")
    if len(post) != _EXP["n_post"]:
        return ("`post` has %d rows and it should have %d. Pre dose rows and rows with a "
                "nan concentration both go." % (len(post), _EXP["n_post"]))
    if "CONC_NG_ML" not in cols:
        return "`post` should keep all of the original columns."
    if bool(post["CONC_NG_ML"].isna().any()):
        return "`post` still contains nan concentrations."
    got = _as_scalar(n_post)
    if got is None or int(got) != _EXP["n_post"]:
        return "`n_post` should be the number of rows in `post`."
    return None


def _part_b2():
    by_group = _need("by_group")
    cols = _cols_of(by_group)
    if cols is None:
        return "`by_group` should be a pandas DataFrame."
    for want in ("GROUP", "NOMINAL_H"):
        if want not in cols:
            idx = [str(n) for n in (by_group.index.names or [])]
            if want in idx:
                return ("`%s` is in the index rather than a column. Call .reset_index() "
                        "after the groupby." % want)
            return "`by_group` has no %s column." % want
    for want in ("CONC_MEAN", "CONC_SD"):
        if want not in cols:
            return ("`by_group` has no %s column. Named aggregation, "
                    "CONC_MEAN=(\\"CONC_NG_ML\\", \\"mean\\"), gets the names right in one "
                    "step." % want)
    if len(by_group) != _EXP["by_group_rows"]:
        return ("`by_group` has %d rows and it should have %d, one per group and time."
                % (len(by_group), _EXP["by_group_rows"]))
    got = by_group.sort_values(["GROUP", "NOMINAL_H"])
    keys = ["%s|%g" % (g, t) for g, t in zip(got["GROUP"], got["NOMINAL_H"])]
    if keys != _EXP["by_group_keys"]:
        return "`by_group` does not hold one row per group and nominal time."
    if not _arr_close(got["CONC_MEAN"].to_numpy(), _EXP["by_group_mean"], rtol=1e-9):
        return "`CONC_MEAN` is not the mean concentration in each group at each time."
    arr = got["CONC_SD"].to_numpy()
    if _arr_close(arr, _EXP["by_group_sd_pop"], rtol=1e-6):
        return ("`CONC_SD` divides by n. Pandas .std() is the sample SD already, so this "
                "looks like ddof=0 was passed explicitly.")
    if not _arr_close(arr, _EXP["by_group_sd"], rtol=1e-9):
        return "`CONC_SD` is not the sample SD within each group and time."
    return None


def _part_b3():
    merged = _need("merged")
    n_missing_weight = _need("n_missing_weight")
    cols = _cols_of(merged)
    if cols is None:
        return "`merged` should be a pandas DataFrame."
    if "BODYWEIGHT_G" not in cols:
        return "`merged` has no BODYWEIGHT_G column, so the join did not bring demo in."
    if len(merged) < _EXP["merged_rows"]:
        return ("`merged` has %d rows and pk has %d. An inner join drops the animal with no "
                "demographics record, which hides exactly the thing you are looking for. "
                "Use how='left'." % (len(merged), _EXP["merged_rows"]))
    if len(merged) > _EXP["merged_rows"]:
        return ("`merged` has %d rows, more than the %d in pk, so a key was duplicated. "
                "validate='many_to_one' would have raised instead."
                % (len(merged), _EXP["merged_rows"]))

    got = _as_scalar(n_missing_weight)
    if got is None:
        return "`n_missing_weight` should be a single whole number."
    if int(got) == _EXP["n_missing_weight_rows"]:
        return ("`n_missing_weight` counted ROWS with no weight. The task asks for animals, "
                "so count distinct ANIMAL values with .nunique().")
    if int(got) != _EXP["n_missing_weight"]:
        return "`n_missing_weight` is not the number of animals with no body weight."
    return None


def _part_b4():
    frac = _need("frac_of_peak")
    arr = _as_array(frac)
    if arr is None:
        return "`frac_of_peak` should be a Series or array of numbers."
    if arr.size == _EXP["n_groups_animal"]:
        return ("`frac_of_peak` has one value per animal, so this is .agg or .max rather "
                "than .transform. A transform gives the group's value back on every one of "
                "its rows, which is what you need to divide by.")
    if arr.size != _EXP["frac_len"]:
        return ("`frac_of_peak` has %d values and pk has %d rows."
                % (arr.size, _EXP["frac_len"]))
    if int(_np.isfinite(arr).sum()) != _EXP["frac_notna"]:
        return ("`frac_of_peak` should still be nan wherever the concentration was nan, so "
                "%d real values out of %d." % (_EXP["frac_notna"], _EXP["frac_len"]))
    if not _num_close(float(_np.nansum(arr)), _EXP["frac_sum"], rtol=1e-9):
        return "`frac_of_peak` is not each concentration over that animal's own maximum."
    cols = _cols_of(globals().get("pk"))
    if cols is not None and "FRAC_OF_PEAK" not in cols:
        return "the column `FRAC_OF_PEAK` has not been added to pk."
    return None


def _part_b5():
    dev = _need("max_dev_min")
    wide = _need("wide")
    got = _as_scalar(dev)
    if got is None:
        return "`max_dev_min` should be a single number."
    if _num_close(got, _EXP["max_dev_h"], rtol=1e-6):
        return "`max_dev_min` is in hours. The task asks for minutes."
    if not _num_close(got, _EXP["max_dev_min"], rtol=1e-6):
        return ("`max_dev_min` is not the largest absolute deviation of ACTUAL_H from "
                "NOMINAL_H, in minutes.")

    cols = _cols_of(wide)
    if cols is None:
        return "`wide` should be a pandas DataFrame from pivot_table."
    if list(wide.shape) != _EXP["wide_shape"]:
        if list(wide.shape) == _EXP["wide_shape"][::-1]:
            return ("`wide` is the wrong way round. index=\\"ANIMAL\\" puts animals down the "
                    "rows and columns=\\"NOMINAL_H\\" puts times across.")
        return ("`wide` has shape %s and it should have shape %s. The pre dose column is "
                "entirely nan, and pivot_table drops it." % (list(wide.shape), _EXP["wide_shape"]))
    if not _arr_close([float(c) for c in wide.columns], _EXP["wide_cols"], rtol=1e-9):
        return "the columns of `wide` should be the nominal times."
    first = [_np.nan if v is None else v for v in _EXP["wide_first_row"]]
    if not _arr_close(wide.iloc[0].to_numpy(), first, rtol=1e-9):
        return "`wide` does not hold the concentrations for each animal at each time."
    return None


def _part_b6():
    inc = _need("inc")
    pct = _need("pct_high_event")
    if _cols_of(inc) is None:
        return "`inc` should be a table, from pd.crosstab."
    if list(inc.shape) != _EXP["inc_shape"]:
        return ("`inc` has shape %s and it should have shape %s, one row per group and one "
                "column per event value." % (list(inc.shape), _EXP["inc_shape"]))
    if [str(v) for v in inc.index] != _EXP["inc_index"]:
        return ("the rows of `inc` should be the dose groups, so GROUP is the first "
                "argument to crosstab.")
    if [str(v) for v in inc.columns] != _EXP["inc_cols"]:
        return "the columns of `inc` should be the EVENT values."
    if [[int(v) for v in row] for row in inc.to_numpy()] != _EXP["inc_values"]:
        return "`inc` does not hold the counts of animals in each group and event category."

    got = _as_scalar(pct)
    if got is None:
        return "`pct_high_event` should be a single number."
    if _num_close(got, _EXP["pct_high_event"] / 100.0, rtol=1e-6):
        return "`pct_high_event` is a fraction. The task asks for a percentage."
    if _num_close(got, _EXP["pct_high_of_study"], rtol=1e-6):
        return ("`pct_high_event` used the whole study as the denominator. Incidence is "
                "within group, so divide by the number of high dose animals.")
    if not _num_close(got, _EXP["pct_high_event"], rtol=1e-6):
        return "`pct_high_event` is not the percentage of high dose animals with the event."
    return None


def _part_c1():
    animal_level = _need("animal_level")
    group_summary = _need("group_summary")

    cols = _cols_of(animal_level)
    if cols is None:
        return "`animal_level` should be a pandas DataFrame."
    for want in ("ANIMAL", "GROUP", "MEAN_INHIB", "N_CELLS"):
        if want not in cols:
            idx = [str(n) for n in (animal_level.index.names or [])]
            if want in idx:
                return "`%s` is in the index rather than a column. Call .reset_index()." % want
            return "`animal_level` has no %s column." % want
    if len(animal_level) == _EXP["n_cells_total"]:
        return ("`animal_level` still has one row per cell. Group by animal before "
                "aggregating.")
    if len(animal_level) != _EXP["animal_rows"]:
        return ("`animal_level` has %d rows and there are %d animals."
                % (len(animal_level), _EXP["animal_rows"]))
    got = animal_level.sort_values("ANIMAL")
    if [int(v) for v in got["ANIMAL"]] != _EXP["animal_ids"]:
        return "`animal_level` does not list the eight animals."
    if [str(v) for v in got["GROUP"]] != _EXP["animal_groups"]:
        return ("GROUP does not line up with ANIMAL. Group by both at once so each animal "
                "keeps its own label.")
    if _arr_close(got["MEAN_INHIB"].to_numpy(), _EXP["animal_median"], rtol=1e-6):
        return "`MEAN_INHIB` is the median of each animal's cells, not the mean."
    if not _arr_close(got["MEAN_INHIB"].to_numpy(), _EXP["animal_mean"], rtol=1e-9):
        return "`MEAN_INHIB` is not the mean of each animal's own cells."
    if not _arr_close(got["N_CELLS"].to_numpy(), _EXP["animal_ncells"], rtol=1e-9):
        return "`N_CELLS` is not the number of cells in each animal."

    cols = _cols_of(group_summary)
    if cols is None:
        return "`group_summary` should be a pandas DataFrame."
    for want in ("GROUP", "MEAN", "SD", "N"):
        if want not in cols:
            idx = [str(n) for n in (group_summary.index.names or [])]
            if want in idx:
                return "`%s` is in the index rather than a column. Call .reset_index()." % want
            return "`group_summary` has no %s column." % want
    if len(group_summary) != len(_EXP["group_names"]):
        return ("`group_summary` has %d rows and there are %d groups."
                % (len(group_summary), len(_EXP["group_names"])))
    got = group_summary.sort_values("GROUP")
    if [str(v) for v in got["GROUP"]] != _EXP["group_names"]:
        return "`group_summary` does not hold one row per dose group."
    if _arr_close(got["N"].to_numpy(), _EXP["pooled_n"], rtol=1e-9):
        return ("`N` is the number of CELLS. The unit of replication is the animal, so N is "
                "the number of animals and `group_summary` has to be built from "
                "`animal_level`.")
    if not _arr_close(got["N"].to_numpy(), _EXP["group_n"], rtol=1e-9):
        return "`N` is not the number of animals in each group."
    if _arr_close(got["MEAN"].to_numpy(), _EXP["pooled_mean"], rtol=1e-6):
        return ("`MEAN` pools every cell, which weights each animal by however many cells it "
                "happened to have. Average the animal means instead.")
    if not _arr_close(got["MEAN"].to_numpy(), _EXP["group_mean"], rtol=1e-9):
        return "`MEAN` is not the mean of the animal means in each group."
    arr = got["SD"].to_numpy()
    if _arr_close(arr, _EXP["pooled_sd"], rtol=1e-6):
        return ("`SD` is the cell level spread, which is a statement about the assay. The "
                "SD you report is between animals, from the animal means.")
    if _arr_close(arr, _EXP["group_sem"], rtol=1e-6):
        return ("`SD` is the standard error. The house rule is SD, which does not shrink "
                "when you add animals.")
    if not _arr_close(arr, _EXP["group_sd"], rtol=1e-9):
        return "`SD` is not the sample SD of the animal means in each group."
    return None


def _part_c2():
    import matplotlib.figure as _mfig
    import matplotlib.axes as _maxes
    fig = _need("fig_c2")
    ax = _need("ax_c2")
    if not isinstance(fig, _mfig.Figure):
        return "`fig_c2` should be a matplotlib Figure, from plt.subplots()."
    if not isinstance(ax, _maxes.Axes):
        return "`ax_c2` should be a single matplotlib Axes."
    if not ax.containers and not ax.collections and not ax.lines:
        raise _Todo("the axes for c2 is still empty.")

    yerr = sorted(_yerr_values(ax))
    if not yerr:
        return ("no error bars were found on the axes. Draw the means with ax.errorbar and "
                "pass yerr, rather than plotting them as plain markers.")
    means = sorted(_plotted_means(ax))
    want_mean = sorted(float(v) for v in _EXP["group_mean"])
    if means and len(means) == len(want_mean):
        if _arr_close(means, sorted(float(v) for v in _EXP["pooled_mean"]), rtol=1e-3):
            return ("the plotted means come from pooling every cell rather than averaging "
                    "the animal means, so c1 needs fixing first.")
        if not _arr_close(means, want_mean, rtol=1e-3):
            return "the plotted means are not the group means from `group_summary`."
    want_sd = sorted(float(v) for v in _EXP["group_sd"])
    if len(yerr) != len(want_sd):
        return ("there are %d error bars on the axes and %d groups to plot."
                % (len(yerr), len(want_sd)))
    if _arr_close(yerr, sorted(float(v) for v in _EXP["group_sem"]), rtol=1e-2):
        return ("the error bars are the SEM. They should be the SD of the animal means, "
                "which is the house rule and is roughly twice as long here.")
    if _arr_close(yerr, sorted(float(v) for v in _EXP["pooled_sd"]), rtol=1e-2):
        return "the error bars are the cell level SD rather than the animal level SD."
    if not _arr_close(yerr, want_sd, rtol=1e-2):
        return "the error bar lengths do not match the SD of the animal means."
    if not ax.get_ylabel().strip():
        return "the y axis has no label. Say what is plotted and what the error bars are."
    if not ax.get_xlabel().strip():
        return "the x axis has no label."
    return None


def _part_c3():
    import matplotlib.figure as _mfig
    import matplotlib.axes as _maxes
    fig = _need("fig_c3")
    ax = _need("ax_c3")
    if not isinstance(fig, _mfig.Figure):
        return "`fig_c3` should be a matplotlib Figure, from plt.subplots()."
    if not isinstance(ax, _maxes.Axes):
        return "`ax_c3` should be a single matplotlib Axes."
    if not ax.lines and not ax.collections:
        raise _Todo("the axes for c3 is still empty.")

    if ax.get_yscale() != "log":
        return ("the y axis is still linear. ax.set_yscale(\\"log\\"), or ax.semilogy "
                "instead of ax.plot.")
    series = []
    for line in ax.lines:
        y = [float(v) for v in line.get_ydata()]
        if len(y) == len(_EXP["wide_cols"]):
            series.append(sorted(y))
    if len(series) < 2:
        return ("there should be one line per dose group, each with one point per nominal "
                "time, and %d were found." % len(series))
    want = [sorted(_EXP["by_group_mean"][:6]), sorted(_EXP["by_group_mean"][6:])]
    matched = 0
    for target in want:
        for got in series:
            if _arr_close(got, target, rtol=1e-3):
                matched += 1
                break
    if matched < 2:
        return ("the plotted lines are not the group mean concentrations from `by_group`. "
                "One line per group, sorted by nominal time.")
    if not ax.get_ylabel().strip() or not ax.get_xlabel().strip():
        return "both axes need labels."
    if ax.get_legend() is None:
        return "there is no legend, so nothing says which line is which dose."
    return None


def _part_d1():
    got = _need("q_sd")
    letter = str(got).strip().lower()[:1]
    if letter not in "abcd":
        return "`q_sd` should be one of the letters \\"a\\", \\"b\\", \\"c\\" or \\"d\\"."
    if letter == "a":
        return ("that is the spread of the individual cells, which describes the assay "
                "rather than the animals, and it uses the wrong unit of replication.")
    if letter == "c":
        return ("that is a standard error. It answers how precisely you located the mean, "
                "and it shrinks as you add animals whether or not the biology changes.")
    if letter == "d":
        return ("that is a standard error computed from cells, so it is wrong twice: the "
                "wrong level and the wrong quantity.")
    if letter != _EXP["q_sd"]:
        return "that is not the right option."
    return None


def _part_d2():
    got = _need("q_axis")
    letter = str(got).strip().lower()[:1]
    if letter not in "abcd":
        return "`q_axis` should be one of the letters \\"a\\", \\"b\\", \\"c\\" or \\"d\\"."
    if letter == "a":
        return ("that collapses the animals and leaves one number per sampling time, so six "
                "numbers rather than eight. The axis you name is the one that disappears.")
    if letter == "c":
        return "that reduces the whole matrix to a single number."
    if letter == "d":
        return ("that also reduces everything to one number, keeping the shape as (1, 1). "
                "keepdims preserves the dimensions, it does not choose an axis.")
    if letter != _EXP["q_axis"]:
        return "that is not the right option."
    return None


def _part_d3():
    got = _need("q_validate")
    letter = str(got).strip().lower()[:1]
    if letter not in "abcd":
        return "`q_validate` should be one of the letters \\"a\\", \\"b\\", \\"c\\" or \\"d\\"."
    if letter == "a":
        return ("that claims both sides are unique, which is false on the left: pk has seven "
                "rows per animal.")
    if letter == "b":
        return ("that is the right relationship stated backwards. The left frame is the one "
                "you call .merge on, and it is the many side here.")
    if letter == "d":
        return ("checking len() is a real habit and worth keeping, but it tells you only "
                "that something changed, after the fact. The question asks which argument "
                "states the expectation in the line that does the work.")
    if letter != _EXP["q_validate"]:
        return "that is not the right option."
    return None
'''

REVISION_MD = '''
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
'''
