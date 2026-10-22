# Week 1, numpy, pandas and matplotlib

The full text of every problem this week, the data it generates, the
toolkit the student is given, what each part asks for, and what every
possible check failure means. No solutions in this file.

---

# W1D1. Clean one plasma profile and get the terminal half-life

- topic: numpy: 1D arrays, nan masking, log-linear terminal slope
- notebook: problems/week01/w1d1_numpy_pk_profile.ipynb
- solution: solutions/week01/w1d1_numpy_pk_profile_solution.ipynb
- expected time: 20-30 min

## Scenario

One healthy volunteer received a single 100 mg intramuscular dose of a new
compound. Plasma was sampled at 11 timepoints out to 36 hours and assayed by
LC-MS/MS. The bioanalytical lab reports any sample below the lower limit of
quantification (LLOQ, here 0.5 mg/L) as BLQ, and in the array you are handed
those arrive as `np.nan`.

This is the first pass a pharmacometrician makes on any new profile: clean the
record, describe the peak, then estimate how fast the drug leaves. Everything
here is plain numpy, no pandas.

## How numpy works, in two minutes

**The problem numpy solves.** A plain Python list holds things in order, but
you cannot do maths on it. `[1, 2, 3] * 2` gives you `[1, 2, 3, 1, 2, 3]`,
which is almost never what a scientist wants. To halve every concentration in
a list you need a loop.

A numpy **array** holds numbers of one kind and does maths on all of them at
once. `np.array([1, 2, 3]) * 2` gives `[2, 4, 6]`. That is the whole idea, and
everything else follows from it:

- **Elementwise by default.** `conc / 1000` converts every value. `t2 - t1`
  subtracts pairwise. No loop, and it runs in compiled code rather than
  Python, so it is fast enough for real datasets.
- **Comparisons give you arrays too.** `conc > 5` does not give one True or
  False, it gives one per entry. That array of True and False is called a
  **mask**, and you can hand it straight back to the array to pull out just
  those entries: `conc[conc > 5]`. This is how nearly all filtering is done.
- **A mask built from one array works on another**, as long as they are the
  same length and in the same order. That is what lets you say "give me the
  times where the concentration was measurable".

**Why `nan` exists.** `nan` means "not a number", a slot where a value should
be but is not. Bioanalytical data is full of them, because a sample below the
assay limit has no number, only a statement that it was too low. numpy
deliberately makes `nan` contagious: `np.max` of anything containing one
returns `nan` rather than quietly ignoring it. That is a feature. It forces
you to decide what to do about missing data instead of letting the decision
get made for you. You either drop them with a mask, or use the `nan`-aware
version of the function, such as `np.nanmax`.

**Positions versus values.** `np.max` answers "what was the biggest". It
cannot tell you *when* that happened, because the array of concentrations
knows nothing about time. `np.argmax` answers "where was the biggest", giving
you a position. You then use that position to look up the matching time in the
other array. Cmax and Tmax are exactly this pair of questions, which is why
they come up together.

## Your task

Three parts, worked one at a time.

| Part | What you work out | Variables to fill in |
|---|---|---|
| a | which samples are usable, and how many came back BLQ | `measurable`, `n_blq`, `t_meas`, `c_meas` |
| b | how high the profile got, and when | `cmax`, `tmax` |
| c | how fast the drug leaves | `t_term`, `c_term`, `lambda_z`, `t_half` |

Below this you will find the **Setup** cell, then a **Toolkit** cell that
demonstrates every move on a tiny made up example, then **Part a**, **Part b**
and **Part c**, each with its own explanation and a cell to write in.

## The setup cell the student runs

This is given to them, already working. The data is synthetic and
seeded, so their numbers are always these numbers.

```python
import numpy as np

PROBLEM_ID = "W1D1"
TOPIC = "numpy: 1D arrays, nan masking, log-linear terminal slope"

rng = np.random.default_rng(1101)

# Study design
DOSE_MG = 100.0
LLOQ = 0.5                     # mg/L, anything below this is reported BLQ
time_h = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 24.0, 36.0])

# One compartment, first order absorption. These are the "truth" the lab cannot see.
_F, _V, _ka, _kel = 0.90, 35.0, 1.20, 0.0866
_true = (_F * DOSE_MG * _ka) / (_V * (_ka - _kel)) * (np.exp(-_kel * time_h) - np.exp(-_ka * time_h))

# Assay noise, then the lab rounds to 3 decimals and censors below the LLOQ.
conc = np.round(_true * rng.lognormal(mean=0.0, sigma=0.15, size=time_h.size), 3)
conc[conc < LLOQ] = np.nan

print("dose:", DOSE_MG, "mg IM        LLOQ:", LLOQ, "mg/L")
print("time_h:", time_h)
print("conc  :", conc)
```

## Toolkit for this problem

Everything below runs on a tiny made up array, so none of it is the answer.
Run the cell, read the output, then use the same moves on the real data.

| What you want to do | How |
|---|---|
| make an array from a list | `np.array([1.0, 2.0, 3.0])` |
| how many entries it has | `a.size`, or `len(a)` |
| flag the missing values | `np.isnan(a)` gives True where the value is nan |
| flip a True/False array | `~mask` turns every True into False and back |
| count the Trues | `np.sum(mask)`, because True counts as 1 |
| keep only some entries | `a[mask]`, where `mask` is True/False and the same length |
| the biggest value | `np.max(a)` |
| where the biggest value is | `np.argmax(a)` gives a position, not a value |
| the value at a position | `a[3]`, counting from 0 |
| the last 4 entries | `a[-4:]` |
| natural log | `np.log(a)`, which is ln, not log base 10 |
| fit a straight line | `np.polyfit(x, y, deg=1)` returns `[slope, intercept]` |

The toolkit cell demonstrates each of those on a tiny made up example.
It is in the notebook and none of it is the answer.

## Part a. Deal with the BLQ samples

The lab handed you `nan` wherever a sample fell below the assay limit. Until
you know which samples are real, no summary of this profile means anything,
so this is always step one.

Build a mask of the usable samples, count the ones that are not, then use that
mask to pull the usable times and concentrations out. The same mask works on
both arrays because they are the same length and in the same order.

Fill in `measurable`, `n_blq`, `t_meas`, `c_meas`.
Moves: `np.isnan`, `~`, `np.sum`, and indexing an array with a mask.

**Variables this part is graded on:** `measurable`, `n_blq`, `t_meas`, `c_meas`

The starter cell they see:

```python
# Part a. Deal with the BLQ samples.

measurable = None   # YOUR CODE: boolean array, True where conc is measurable
n_blq = None        # YOUR CODE: whole number, how many samples are BLQ
t_meas = None       # YOUR CODE: times of the measurable samples
c_meas = None       # YOUR CODE: concentrations of the measurable samples

print("measurable:", measurable)
print("n_blq     :", n_blq)
print("t_meas    :", t_meas)
print("c_meas    :", c_meas)
```

**Every failure message part a can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `measurable` should be a boolean array but its dtype is %s. np.isnan returns booleans already, so no comparison is needed.
- `measurable` has shape %s. It needs one entry per sample, so it should be the same length as time_h.
- `measurable` is flagging the wrong samples. True has to mean the sample IS usable, so you probably need to flip the result of np.isnan with ~.
- `n_blq` should be a single whole number, but it is not a single number.
- `n_blq` is not the number of BLQ samples. Count the entries where conc is nan, which is the opposite of your measurable mask.
- `c_meas` still contains nan. Those are the BLQ samples you were meant to drop.
- `t_meas` has the right length but the wrong times. Index time_h with the same mask you used for conc, and do not sort or reorder it.
- `c_meas` has the right length but the wrong values. Keep the original time order.
- `%s` should be a 1D numeric array.
- `%s` still has every sample in it. Index the full array with your boolean mask so that only the measurable samples survive.
- `%s` holds %d values, which does not match the number of measurable samples.

## Part b. Describe the peak

Cmax and Tmax are two questions about the same point on the curve: how high
did it get, and when. `np.max` answers the first. The second needs
`np.argmax`, which hands you a **position**, and you then look that position
up in the time array to turn it into an hour.

Work from the measurable samples, so `c_meas` and `t_meas`, not `conc` and
`time_h`. Taking a max over an array containing `nan` gives you `nan`.

Fill in `cmax`, `tmax`.
Moves: `np.max`, `np.argmax`, and indexing with the position you get back.

**Variables this part is graded on:** `cmax`, `tmax`

The starter cell they see:

```python
# Part b. Describe the peak.

cmax = None   # YOUR CODE: highest observed concentration, mg/L
tmax = None   # YOUR CODE: the time in hours at which cmax was observed

print("cmax:", cmax, "mg/L")
print("tmax:", tmax, "h")
```

**Every failure message part b can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `cmax` should be a single number. np.max returns one value; np.maximum does not.
- `cmax` is not the highest measurable concentration. It should come straight out of c_meas with no averaging or rounding.
- `tmax` should be a single number.
- `tmax` looks like a concentration. You may have swapped cmax and tmax.
- `tmax` is the index of the peak, not the time of the peak. Feed that index back into your time array.
- `tmax` is not the time of the peak. Use np.argmax on the concentrations to get a position, then read the matching entry of t_meas.
- `cmax` came out as nan, so you took the max over the raw conc array. Use the measurable values from part a, or reach for np.nanmax.
- `cmax` looks like a time, not a concentration. You may have swapped cmax and tmax.

## Part c. Terminal half-life

Once absorption is finished, elimination is first order, meaning a constant
fraction leaves per hour. On a log scale that is a straight line, so fitting
a line to `log(concentration)` against time recovers the rate constant.

Slice off the last 4 measurable samples, fit the line, and convert. The fitted
slope is negative because the drug is going away, and `lambda_z` is defined as
the positive version of it. The half-life is then `ln(2) / lambda_z`.

Fill in `t_term`, `c_term`, `lambda_z`, `t_half`.
Moves: `a[-4:]`, `np.log`, `np.polyfit(x, y, deg=1)` which returns
`[slope, intercept]`.

**Variables this part is graded on:** `t_term`, `c_term`, `lambda_z`, `t_half`

The starter cell they see:

```python
# Part c. Terminal half-life from the last 4 measurable samples.

t_term = None     # YOUR CODE: last 4 measurable times
c_term = None     # YOUR CODE: last 4 measurable concentrations
lambda_z = None   # YOUR CODE: terminal rate constant, 1/h
t_half = None     # YOUR CODE: terminal half-life, h

print("t_term  :", t_term)
print("c_term  :", c_term)
print("lambda_z:", lambda_z, "1/h")
print("t_half  :", t_half, "h")
```

**Every failure message part c can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `c_term` contains nan, so you sliced the raw conc array instead of c_meas.
- `t_term` is not the last 4 measurable times. Slice t_meas from part a, not time_h.
- `c_term` is not the last 4 measurable concentrations. Slice c_meas from part a.
- `lambda_z` should be a single number. np.polyfit returns an array of coefficients, so pull out the one you need.
- `lambda_z` is negative. The fitted slope is negative because the drug is going away, and lambda_z is defined as the positive version of it.
- `lambda_z` is not the terminal rate constant. Fit np.polyfit(t_term, np.log(c_term), deg=1); the first returned value is the slope.
- `t_half` should be a single number.
- `t_half` was computed from the whole profile, not just the terminal phase. The early points are still absorbing, which biases the slope.
- `t_half` does not follow from your lambda_z. It should be np.log(2) divided by lambda_z, in hours.
- `%s` should be a 1D numeric array.
- `%s` holds %d values but the task asks for the last 4 measurable samples. Slicing with [-4:] gives you exactly 4.
- `lambda_z` is close but not right. Check that you regressed the natural log of concentration (np.log, not np.log10) against time, and that you took the slope rather than the intercept.
- `t_half` is the mean residence time style 1/lambda_z, not the half-life. A half-life carries a factor of ln(2).

**Functions this problem introduces:** np.random.default_rng, np.array, .size, np.isnan, ~, boolean indexing, np.sum, np.max / np.min, np.argmax, np.log, np.polyfit, len

---

# W1D2. A subject by time matrix: axis reductions, SD, and broadcasting

- topic: numpy: 2D arrays, axis reductions, sample SD, broadcasting
- notebook: problems/week01/w1d2_numpy_subject_matrix.ipynb
- solution: solutions/week01/w1d2_numpy_subject_matrix_solution.ipynb
- expected time: 20-30 min

## Scenario

A single ascending dose study gave 8 healthy volunteers the same 50 mg oral
dose. Each subject was sampled at the same 9 timepoints, so the data manager
hands you one rectangular block: `conc` with shape `(8, 9)`, one **row per
subject** and one **column per timepoint**.

Before anybody fits a model, the pharmacometrician reports the mean profile,
how variable the subjects are, and then overlays the profiles on a common
scale so shapes can be compared without the amplitude getting in the way.

Two house rules that apply to everything in this repo:

- Variability is reported as **standard deviation (SD)**, never SEM. SEM
  shrinks as you add subjects, which hides exactly the between-subject
  variability a pharmacometrician is paid to see.
- SD means the **sample** SD, with `n - 1` in the denominator. Note that
  `np.std` defaults to `n`, so you have to say so explicitly.

## How numpy works in two dimensions

**A 2D array is a rectangle with a meaning attached to each direction.** Here
rows are subjects and columns are timepoints, so `conc[3, 5]` is subject 4 at
timepoint 6, counting from zero. `conc.shape` reports `(8, 9)`, always as
`(rows, columns)`.

**Reductions and the axis argument.** A reduction is anything that turns many
numbers into fewer: mean, max, sum, std. On a 2D array you have to say which
direction to collapse, and that is what `axis` is for. The rule worth
memorising:

> **The axis you name is the one that disappears.**

So `axis=0` destroys the row direction and leaves one value per column, and
`axis=1` destroys the column direction and leaves one value per row. People
get this backwards constantly, which is why the fix is always the same: print
`.shape` of what came out and ask whether that length is the number of
subjects or the number of timepoints.

**ddof, and why the default is wrong for you.** The SD of a sample divides by
`n - 1` rather than `n`, because you used the data itself to estimate the mean
and that uses up one degree of freedom. Dividing by `n` gives a number that is
systematically too small. numpy defaults to `ddof=0`, dividing by `n`, which
suits physics and image processing where you often have the whole population.
It does not suit you, who almost always has a sample of subjects drawn from a
population. So in numpy you must write `ddof=1` every single time. Pandas made
the opposite choice and already uses `n - 1`, which is a real inconsistency
between the two libraries and a classic source of quietly wrong numbers.

**Broadcasting.** When you combine two arrays of different shapes, numpy tries
to stretch the smaller one to fit, comparing shapes **from the right**. A
`(8, 9)` and a `(9,)` line up, because 9 matches 9, and the `(9,)` is reused
for every row. A `(8, 9)` and an `(8,)` do **not** line up, because it checks
9 against 8 and gives up. This is the error you will hit in part c, and it is
not a bug in your thinking: you want to divide each row by its own number, so
you have to say so by making that array `(8, 1)` instead of `(8,)`. Adding
`keepdims=True` to the reduction does it, and so does `.reshape(-1, 1)`.

## Your task

Three parts, worked one at a time.

| Part | What you work out | Variables to fill in |
|---|---|---|
| a | the mean profile, and each subject's own peak | `mean_by_time`, `cmax_by_subject` |
| b | how much the subjects differ | `sd_by_time`, `cv_by_time` |
| c | every subject rescaled to their own peak | `norm_conc` |

Two house rules apply throughout. Variability is reported as **SD**, never
SEM. And SD means the **sample** SD with `n - 1`, which numpy will not give
you unless you ask with `ddof=1`.

Below this you will find the **Setup** cell, then a **Toolkit** cell that
demonstrates every move on a tiny made up example, then **Part a**, **Part b**
and **Part c**, each with its own explanation and a cell to write in.

## The setup cell the student runs

This is given to them, already working. The data is synthetic and
seeded, so their numbers are always these numbers.

```python
import numpy as np

PROBLEM_ID = "W1D2"
TOPIC = "numpy: 2D arrays, axis reductions, sample SD, broadcasting"

rng = np.random.default_rng(1102)

DOSE_MG = 50.0
N_SUBJECTS = 8
time_h = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 24.0])
subject_id = np.arange(1, N_SUBJECTS + 1)

# Between subject variability in clearance and volume, then assay noise on top.
_V = 40.0 * rng.lognormal(0.0, 0.25, size=N_SUBJECTS)
_kel = 0.10 * rng.lognormal(0.0, 0.30, size=N_SUBJECTS)
_ka = 0.90 * rng.lognormal(0.0, 0.15, size=N_SUBJECTS)

_V, _kel, _ka = _V[:, None], _kel[:, None], _ka[:, None]
_t = time_h[None, :]
_true = (0.85 * DOSE_MG * _ka) / (_V * (_ka - _kel)) * (np.exp(-_kel * _t) - np.exp(-_ka * _t))
conc = np.round(_true * rng.lognormal(0.0, 0.10, size=_true.shape), 3)

print("conc shape:", conc.shape, " (rows = subjects, columns = timepoints)")
print("subject_id:", subject_id)
print("time_h    :", time_h)
print(conc)
```

## Toolkit for this problem

The one idea to get straight here is **axis**. In a 2D array, the axis you
name is the one that **disappears**.

- `axis=0` collapses the rows, so you are left with **one value per column**.
- `axis=1` collapses the columns, so you are left with **one value per row**.

In this problem rows are subjects and columns are timepoints, so "per
timepoint" means `axis=0`, and "per subject" means `axis=1`.

| What you want to do | How |
|---|---|
| make a 2D array | `np.array([[1, 2, 3], [4, 5, 6]])` |
| its size | `a.shape`, which reads `(rows, columns)` |
| mean of each column | `a.mean(axis=0)` |
| mean of each row | `a.mean(axis=1)` |
| biggest in each row | `a.max(axis=1)` |
| sample SD, with `n - 1` | `a.std(axis=0, ddof=1)` |
| population SD, with `n` | `a.std(axis=0)`, the default, which is **not** what we want |
| keep the collapsed axis | `a.max(axis=1, keepdims=True)` gives shape `(rows, 1)` |
| divide every row by its own number | `a / a.max(axis=1, keepdims=True)` |

The toolkit cell demonstrates each of those on a tiny made up example.
It is in the notebook and none of it is the answer.

## Part a. Reduce along the right axis

Both of these are one-line reductions. The only thing to get right is which
direction you collapse, and the rule is that the axis you name is the one that
disappears.

"Mean at each timepoint" means you want one value per column, so the rows have
to go. "Each subject's peak" means one value per row, so the columns have to
go. If you are unsure which you got, print `.shape` of the result and ask
whether that length is 8 or 9.

Fill in `mean_by_time` (length 9), `cmax_by_subject` (length 8).
Moves: `conc.mean(axis=...)`, `conc.max(axis=...)`.

**Variables this part is graded on:** `mean_by_time`, `cmax_by_subject`

The starter cell they see:

```python
# Part a. Reduce along the right axis.

mean_by_time = None       # YOUR CODE: mean at each timepoint, length 9
cmax_by_subject = None    # YOUR CODE: each subject's peak, length 8

print("mean_by_time   ", np.shape(mean_by_time), mean_by_time)
print("cmax_by_subject", np.shape(cmax_by_subject), cmax_by_subject)
```

**Every failure message part a can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `mean_by_time` should be a numeric array.
- `mean_by_time` is a single number, so you averaged the whole matrix. Pass an axis argument so the reduction only collapses one direction.
- `mean_by_time` has one value per subject, not per timepoint. You collapsed the wrong axis: to keep one value per column you have to collapse the rows.
- `mean_by_time` has shape %s but it should be (%d,), one value per timepoint.
- `mean_by_time` has the right shape but the wrong values. It is a plain mean of conc, nothing else.
- `cmax_by_subject` should be a numeric array.
- `cmax_by_subject` is a single number, so you took the max of the whole matrix. Pass an axis argument.
- `cmax_by_subject` has one value per timepoint, not per subject. You collapsed the wrong axis: to keep one value per row you have to collapse the columns.
- `cmax_by_subject` has shape %s but it should be (%d,), one value per subject.
- `cmax_by_subject` is each subject's mean, not each subject's peak.
- `cmax_by_subject` has the right shape but the wrong values.

## Part b. Describe the variability

Same reduction direction as the mean in part a, since you want the spread
across subjects at each timepoint.

Two things to be deliberate about. This is a **sample** of subjects, so the SD
divides by `n - 1`, and numpy will not do that unless you ask with `ddof=1`.
And it is SD, not SEM, so nothing here should involve a square root of the
number of subjects.

CV% is just the SD expressed as a percentage of the mean, which makes
variability comparable across timepoints with very different magnitudes.

Fill in `sd_by_time`, `cv_by_time`, both length 9.
Moves: `conc.std(axis=..., ddof=1)`, then elementwise division and `* 100`.

**Variables this part is graded on:** `sd_by_time`, `cv_by_time`

The starter cell they see:

```python
# Part b. Describe the variability. SD, not SEM. Sample SD, so ddof=1.

sd_by_time = None   # YOUR CODE: sample SD across subjects at each timepoint
cv_by_time = None   # YOUR CODE: 100 * SD / mean at each timepoint

print("sd_by_time", np.shape(sd_by_time), np.round(sd_by_time, 3) if sd_by_time is not None else None)
print("cv_by_time", np.shape(cv_by_time), np.round(cv_by_time, 1) if cv_by_time is not None else None)
```

**Every failure message part b can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `sd_by_time` should be a numeric array.
- `sd_by_time` has one value per subject. The SD you want is across subjects at a fixed timepoint, so collapse the rows and keep the columns.
- `sd_by_time` has shape %s but it should be (%d,), one value per timepoint.
- `sd_by_time` is the SEM, not the SD. You divided by the square root of the number of subjects somewhere. This repo always reports SD.
- `sd_by_time` used the population SD with n in the denominator. np.std defaults to ddof=0, so pass ddof=1 to get the sample SD.
- `sd_by_time` has the right shape but the wrong values.
- `cv_by_time` should be a numeric array.
- `cv_by_time` has shape %s but it should be (%d,), one value per timepoint.
- `cv_by_time` is a fraction. The task asks for a percent, so multiply by 100.
- `cv_by_time` is not 100 * SD / mean at each timepoint. Check that you paired your SD and your mean at the same timepoints, and that both are per timepoint.

## Part c. Put every subject on a common scale

Normalising each profile to its own peak strips out the amplitude differences
so you can compare the **shape** of absorption and elimination between
subjects. It is a standard first look before any modelling.

You already have each subject's peak from part a, but dividing straight by it
will fail, because a `(8, 9)` and an `(8,)` do not line up: numpy matches
shapes from the right and tries 9 against 8. Read the error when it happens,
it is a useful one to recognise. Make the divisor `(8, 1)` and it works.

Fill in `norm_conc`, shape `(8, 9)`, every row peaking at exactly 1.0.
Moves: `conc.max(axis=1, keepdims=True)`, or `.reshape(-1, 1)`, then divide.

**Variables this part is graded on:** `norm_conc`

The starter cell they see:

```python
# Part c. Normalise each subject's profile to that subject's own Cmax.
# Use broadcasting, not a for loop.

norm_conc = None   # YOUR CODE: shape (8, 9), every row peaking at 1.0

if norm_conc is not None:
    print("norm_conc shape:", np.shape(norm_conc))
    print("max of each row:", np.round(np.max(norm_conc, axis=1), 6))
```

**Every failure message part c can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `norm_conc` should be a numeric array.
- `norm_conc` has shape %s but it should be (%d, %d), the same shape as conc. If you got an error about shapes not aligning, that is the (8,) against (8, 9) problem the hint warns about.
- `norm_conc` was divided by the single largest value in the whole matrix, so only one subject peaks at 1.0. Each subject needs their own Cmax.
- `norm_conc` was divided by the peak at each timepoint, which is a reduction down the columns. You want the peak of each subject, which is across the rows.
- `norm_conc` has the right shape but only %d of %d rows peak at exactly 1.0. Divide each row by the maximum of that same row.
- `norm_conc` has the right shape and every row peaks at 1.0, but the other values do not match.
- `norm_conc` has shape %s, which is the transpose of what is wanted. Keep subjects on the rows.

**Functions this problem introduces:** .shape, np.mean, np.std, keepdims, .reshape, broadcasting, np.round

---

# W1D3. Rat PK study: filter, summarise by group, and join the dosing record

- topic: pandas: filtering, derived columns, groupby.agg, left merge
- notebook: problems/week01/w1d3_pandas_wrangle_merge.ipynb
- solution: solutions/week01/w1d3_pandas_wrangle_merge_solution.ipynb
- expected time: 20-30 min

## Scenario

You have the raw bioanalytical output from a rat single dose PK study. Twelve
animals in three dose groups were sampled at 8 timepoints including a predose
blank, which gives you 96 rows in long format (`pk`).

The study director separately sends the dosing record (`demo`): sex, body
weight on the day of dosing, and the actual milligrams each animal received.
Nominal dosing was 1, 3 and 10 mg/kg, but the actual dose depends on each
animal's weight, so the real mg/kg has to be recomputed from the record.

One thing the study director has not mentioned: the dosing record is not
complete. Finding that out yourself, rather than after you have fitted a
model, is most of the point of this problem. A join that silently drops rows is
one of the most common ways a PK analysis goes quietly wrong.

## How pandas works, in two minutes

**A DataFrame is a dictionary of columns, not a grid of cells.** That sounds
pedantic but it explains most of the behaviour. Each column is a **Series**: a
numpy array plus an **index**, which is a set of labels for the rows. So
`df["CONC"]` hands you one Series, and arithmetic on it is elementwise, just
like numpy. Each column has its own type, which is why one table can hold
animal names, group labels and concentrations at the same time, something a
numpy array cannot do.

**The index is the part people trip over.** Every DataFrame has row labels,
and by default they are 0, 1, 2 and so on. They look like positions but they
are labels, and they stick to their rows. Filter out some rows and the
survivors keep their original numbers, so the index now has gaps. That is
usually harmless, but it matters after a `groupby`, where the grouping keys
get *moved into the index*. The columns you grouped by then stop being
columns, and anything downstream looking for them fails. `.reset_index()`
pushes them back out into ordinary columns, which is why you will see it after
almost every groupby.

**Split, apply, combine.** That is the whole model behind `groupby`. Pandas
splits the table into one chunk per group, applies your summary to each chunk,
then stacks the answers back into a new table. `df.groupby(["GROUP",
"TIME_H"])["CONC"].agg(["mean", "std"])` reads exactly that way: split by
group and time, look at the concentration column, compute mean and SD of each
chunk. You get one row per combination that actually occurs.

**Joins, and the one that bites.** `pd.merge` matches rows between two tables
on a shared column. The `how` argument decides what happens to rows that find
no partner:

- `how="left"` keeps every row of the left table, filling the missing columns
  with `NaN`. You can see what is missing.
- `how="inner"`, which is the **default**, keeps only rows that matched and
  silently discards the rest. No error, no warning, just a smaller table.

That default is why a PK analysis can lose an entire animal and still run to
completion with a plausible-looking answer. The habit that protects you is
boring and absolute: print `len()` before and after every merge, and if the
number changed, know why. Rows can also go *up*, if the right-hand table has
more than one row per key, which quietly duplicates your data.

**One more pandas quirk.** Pandas `std()` already divides by `n - 1`, unlike
numpy, which needs `ddof=1`. Two libraries, two defaults, and nothing warns
you.

## Your task

Three parts, worked one at a time.

| Part | What you work out | Variables to fill in |
|---|---|---|
| a | how big the dataset really is, and the post-dose rows | `n_animals`, `df_post` |
| b | the mean and SD profile for each dose group | `summary` |
| c | the dosing record joined on, and what is missing from it | `pk_merged`, `n_missing_weight` |

Below this you will find the **Setup** cell, then a **Toolkit** cell that
demonstrates every move on a tiny made up example, then **Part a**, **Part b**
and **Part c**, each with its own explanation and a cell to write in.

## The setup cell the student runs

This is given to them, already working. The data is synthetic and
seeded, so their numbers are always these numbers.

```python
import numpy as np
import pandas as pd

PROBLEM_ID = "W1D3"
TOPIC = "pandas: filtering, derived columns, groupby.agg, left merge"

rng = np.random.default_rng(1103)

N_ANIMALS = 12
times = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0])
animals = np.array(["R%02d" % i for i in range(1, N_ANIMALS + 1)])
groups = np.repeat(["Low", "Mid", "High"], 4)
nominal_mg_kg = np.repeat([1.0, 3.0, 10.0], 4)
sex = np.tile(["M", "M", "F", "F"], 3)

bw_g = np.round(250.0 + rng.normal(0.0, 18.0, size=N_ANIMALS), 0)
# Syringe accuracy means the delivered dose is close to, but not exactly, nominal.
dose_mg = np.round(nominal_mg_kg * bw_g / 1000.0 * rng.lognormal(0.0, 0.02, size=N_ANIMALS), 4)

# Between animal variability, one compartment oral, concentrations in ng/mL.
_V = 0.20 * rng.lognormal(0.0, 0.22, size=N_ANIMALS)
_kel = 0.30 * rng.lognormal(0.0, 0.25, size=N_ANIMALS)
_ka = 1.50 * rng.lognormal(0.0, 0.15, size=N_ANIMALS)
_coef = (0.90 * dose_mg * _ka) / (_V * (_ka - _kel))
_true = 1000.0 * _coef[:, None] * (np.exp(-_kel[:, None] * times[None, :])
                                  - np.exp(-_ka[:, None] * times[None, :]))
_obs = np.round(_true * rng.lognormal(0.0, 0.12, size=_true.shape), 1)

pk = pd.DataFrame({
    "ANIMAL": np.repeat(animals, times.size),
    "GROUP": np.repeat(groups, times.size),
    "TIME_H": np.tile(times, N_ANIMALS),
    "CONC_NG_ML": _obs.ravel(),
})

# The dosing record the study director sent over.
_in_record = animals != "R07"
demo = pd.DataFrame({
    "ANIMAL": animals[_in_record],
    "SEX": sex[_in_record],
    "BODYWEIGHT_G": bw_g[_in_record],
    "DOSE_MG": dose_mg[_in_record],
})

print("pk:", pk.shape)
print(pk.head(10))
print("")
print("demo:", demo.shape)
print(demo.head())
```

## Toolkit for this problem

A DataFrame is a table. Each column has a name, and you pull one out with
`df["NAME"]`. Everything below runs on a tiny made up table, so none of it is
the answer.

| What you want to do | How |
|---|---|
| look at the first rows | `df.head()` |
| its size | `df.shape`, which reads `(rows, columns)` |
| the column names | `df.columns` |
| one column | `df["CONC"]` |
| how many distinct values | `df["ANIMAL"].nunique()` |
| keep some rows | `df[df["TIME_H"] > 0]` |
| add a column | `df["NEW"] = df["A"] / df["B"]` |
| mean and SD per group | `df.groupby(["G", "T"])["CONC"].agg(["mean", "std"])` |
| turn group keys back into columns | `.reset_index()` |
| rename columns | `.rename(columns={"mean": "CONC_MEAN"})` |
| join two tables | `pd.merge(left, right, on="ANIMAL", how="left")` |
| count the gaps | `df["COL"].isna().sum()` |

Two things worth knowing before you start.

- Pandas `std` already divides by `n - 1`, so it is the sample SD you want.
  This is the opposite of numpy, where you have to ask with `ddof=1`.
- `how="left"` keeps every row of the left table. `how="inner"` keeps only
  rows that matched, and quietly drops the rest. Always compare `len()`
  before and after a merge.

The toolkit cell demonstrates each of those on a tiny made up example.
It is in the notebook and none of it is the answer.

## Part a. Get your bearings and drop the predose samples

Before anything else, find out how big the dataset really is. Counting rows is
not counting animals: each animal appears once per timepoint, so the row count
is a multiple of the animal count.

Then drop the predose samples. They are all zero by definition and they would
drag every mean down and inflate every SD if you left them in a summary.

Fill in `n_animals`, and `df_post` which keeps all four columns.
Moves: `df["COL"].nunique()`, and boolean filtering `df[df["COL"] > 0]`.

**Variables this part is graded on:** `n_animals`, `df_post`

The starter cell they see:

```python
# Part a. Count the animals and keep only post-dose rows.

n_animals = None   # YOUR CODE: how many distinct animals are in pk
df_post = None     # YOUR CODE: the rows of pk with TIME_H greater than 0

print("n_animals:", n_animals)
print("df_post shape:", None if df_post is None else df_post.shape)
```

**Every failure message part a can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `n_animals` should be a single whole number.
- `n_animals` is the number of rows in pk, not the number of animals. Each animal appears once per timepoint, so count distinct values.
- `n_animals` is not the number of distinct animals in pk. Try nunique on the ANIMAL column.
- `df_post` should be a pandas DataFrame, but it is a %s.
- `df_post` is missing the column(s) %s. Keep all four columns of pk. ,
- `df_post` still has every row of pk, so the predose samples were not removed. Filter on TIME_H greater than 0.
- `df_post` has %d rows, which is not the number of post-dose rows. Use strictly greater than 0, which drops exactly one sample per animal.
- `df_post` still contains a TIME_H of 0 or less.

## Part b. Mean and SD profile per dose group

This is split, apply, combine: pandas splits the table into one chunk per
group and timepoint, computes your statistics on each chunk, and stacks the
answers back up.

Two details decide whether the result is usable downstream. The grouping keys
come back sitting in the **index**, not as columns, so `.reset_index()` is
needed to flatten them. And the aggregation names come out as `mean` and
`std`, so rename them.

Pandas `std` already divides by `n - 1`, so you do not need `ddof` here. That
is the opposite of numpy.

Build `summary` from `df_post`, one row per `GROUP` and `TIME_H` combination,
with exactly these four columns, named exactly like this:

| column | meaning |
|---|---|
| `GROUP` | dose group |
| `TIME_H` | nominal time in hours |
| `CONC_MEAN` | mean concentration in that group at that time |
| `CONC_SD` | **sample** SD of concentration in that group at that time |

It has to be a flat DataFrame, so `GROUP` and `TIME_H` are ordinary columns
and not an index. SD, never SEM.

Moves: `.groupby([...])["COL"].agg(["mean", "std"])`, `.reset_index()`,
`.rename(columns={...})`.

**Variables this part is graded on:** `summary`

The starter cell they see:

```python
# Part b. Mean and sample SD per group per timepoint, from df_post.
# Columns must come out named GROUP, TIME_H, CONC_MEAN, CONC_SD.

summary = None   # YOUR CODE

print(summary)
```

**Every failure message part b can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `summary` is a Series, not a DataFrame. Aggregating with a list of statistics, as in .agg(['mean', 'std']), gives you a frame.
- `summary` should be a pandas DataFrame, but it is a %s.
- `summary` is missing the column(s) %s%s. The four columns have to be named exactly GROUP, TIME_H, CONC_MEAN, CONC_SD. , and has unexpected column(s) ,
- `summary` includes the predose timepoint, so it was built from pk rather than from your post-dose table from part a.
- `summary` has %d rows but there should be one row per group and timepoint combination. Check that you grouped by both keys.
- `summary` does not cover the expected set of dose groups.
- `summary` does not cover the expected set of timepoints.
- `CONC_MEAN` does not match a plain mean of CONC_NG_ML within each group and timepoint. Check that you aggregated the concentration column and nothing else.
- `CONC_SD` is the SEM, not the SD. Something divided by the square root of n. This repo always reports SD.
- `CONC_SD` used n in the denominator rather than n - 1. That happens if you hand numpy's std to .agg; the pandas std already uses n - 1.
- `CONC_SD` does not match the sample SD within each group and timepoint.
- `%s` is sitting in the index rather than being a column. Call .reset_index() after the groupby to flatten it.

## Part c. Join the dosing record and recover the real mg/kg

Start from `pk`, the full table, not `df_post`.

The dosing record is incomplete, and how you join decides whether you find
that out. The default join is `inner`, which keeps only matched rows and
discards the rest with no error and no warning. Use a **left** join so every
row of `pk` survives and the gaps show up as `NaN` where you can see them.

Then check: print `len(pk)` and `len(pk_merged)` and make sure they agree. Get
into this habit now. A merge that silently drops an animal is one of the most
common ways a PK analysis produces a confident wrong answer.

Body weight is recorded in **grams** and the dose in milligrams, so convert
before dividing. Sanity check the result against the nominal 1, 3 and 10
mg/kg.

Fill in `pk_merged`, which is `pk` joined to `demo` on `ANIMAL` keeping every
row of `pk`, plus a new `DOSE_MG_PER_KG` column. And `n_missing_weight`, how
many rows end up with no body weight.

Moves: `pd.merge(left, right, on=..., how="left")`, arithmetic between
columns, `df["COL"].isna().sum()`.

**Variables this part is graded on:** `pk_merged`, `n_missing_weight`

The starter cell they see:

```python
# Part c. Join the dosing record onto pk without losing any rows.

pk_merged = None         # YOUR CODE: pk joined to demo, plus a DOSE_MG_PER_KG column
n_missing_weight = None  # YOUR CODE: rows of pk_merged with no body weight

print("rows in pk       :", len(pk))
print("rows in pk_merged:", None if pk_merged is None else len(pk_merged))
print("n_missing_weight :", n_missing_weight)
if pk_merged is not None:
    print(pk_merged.head())
```

**Every failure message part c can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `pk_merged` should be a pandas DataFrame, but it is a %s.
- `pk_merged` has no DOSE_MG_PER_KG column yet. Add it after the merge.
- `pk_merged` has %d rows but pk has %d, so the join quietly dropped the animal that is missing from the dosing record. That is what an inner join does. Use how='left' to keep every row of pk.
- `pk_merged` has %d rows but pk has only %d, so the join multiplied rows. That happens when the right table has more than one row per ANIMAL.
- `pk_merged` has %d rows but it should have the same %d rows as pk.
- `DOSE_MG_PER_KG` is milligrams per gram, because the record stores body weight in grams. Convert to kilograms before dividing.
- `DOSE_MG_PER_KG` does not match DOSE_MG divided by body weight in kilograms. Sanity check it: the values should land near the nominal 1, 3 and 10 mg/kg.
- `n_missing_weight` should be a single whole number.
- `n_missing_weight` counts 1, which is the number of animals missing from the record. The question asks how many rows of pk_merged have no body weight, and each animal contributes one row per timepoint.
- `n_missing_weight` is not the number of rows with no body weight. Count where BODYWEIGHT_G is na in pk_merged.
- `pk_merged` has no %s column, so the join did not bring the dosing record across. Merge on ANIMAL.

**Functions this problem introduces:** pd.DataFrame, .head, .columns, selecting a column, boolean filtering, adding a column, .nunique / .unique, .isna, .groupby, .agg, .reset_index, .rename, pd.merge, np.repeat / np.tile

---

# W1D4. A collaborator's spreadsheet: reshape wide to tidy and back again

- topic: pandas: melt, string cleaning, duplicates, pivot
- notebook: problems/week01/w1d4_pandas_tidy_wide.ipynb
- solution: solutions/week01/w1d4_pandas_tidy_wide_solution.ipynb
- expected time: 20-30 min

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

## Why tidy data, and how reshaping works

**Two layouts for the same numbers.** A spreadsheet laid out for human eyes is
usually **wide**: one row per subject, and one column per timepoint. It is
compact and easy to read down a page. The alternative is **tidy**, sometimes
called long: one row per observation, with columns that say what each
observation is. The same 60 numbers become 60 rows instead of 10 rows by 6
columns.

**Why every tool wants tidy.** In a wide table, the timepoint is stored in the
*column name*, which is metadata, not data. You cannot group by it, filter on
it, do arithmetic with it, or put it on an axis, because as far as the
software is concerned it is just a label. The moment you want "mean by
timepoint" or "plot concentration against time", you need time to be a column
of values. Tidy layout makes every variable a column, so ggplot2, seaborn,
`groupby`, `lme4` and every modelling tool can all find what they need the
same way. This is why reshaping is the first thing you do with a collaborator
file, every time, and why it is worth being quick at.

**melt and pivot are inverses.** `melt` takes a set of columns and folds their
names into one column and their values into another. `id_vars` is the list of
columns to leave alone; everything else gets folded. `pivot` does the reverse,
spreading one column's values out into new columns.

**Why pivot can fail when melt cannot.** Going wide to tidy is always
possible. Going back is not, because wide format has exactly one cell per
(row, column) pair. If your tidy data has two rows claiming the same animal
and the same timepoint, pivot has no way to choose, so it raises `Index
contains duplicate entries`. That error is not a nuisance, it is the reshape
telling you the data has a problem you had not noticed. Here the cause is a
record that got sent twice. Elsewhere it could be two real measurements that
need averaging, which is what `pivot_table` is for. Which one you are looking
at is a scientific judgment, not a coding one, so look before you reach for
either fix.

**Text that looks like numbers.** After melting, your timepoints are strings
like `"T_24"`. Sorting strings compares them character by character, so
`"T_24"` lands before `"T_4"` because `2` comes before `4`. Convert to a
number before you sort, or your profiles come out scrambled in a way that is
surprisingly easy to miss on a plot.

## Your task

Three parts, worked one at a time.

| Part | What you work out | Variables to fill in |
|---|---|---|
| a | the wide table reshaped into tidy | `tidy` |
| b | a numeric time column, sorted, index renumbered | `tidy_clean` |
| c | the repeated rows, and the pivot back to wide | `n_dup_rows`, `wide_again` |

Below this you will find the **Setup** cell, then a **Toolkit** cell that
demonstrates every move on a tiny made up example, then **Part a**, **Part b**
and **Part c**, each with its own explanation and a cell to write in.

## The setup cell the student runs

This is given to them, already working. The data is synthetic and
seeded, so their numbers are always these numbers.

```python
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
```

## Toolkit for this problem

Wide means one row per subject and one column per timepoint, which is how
spreadsheets look. Tidy means one row per observation, which is what every
plotting and modelling tool wants. `melt` goes wide to tidy, `pivot` goes
back.

| What you want to do | How |
|---|---|
| wide to tidy | `df.melt(id_vars=["ID"], var_name="TIME_LABEL", value_name="VAL")` |
| strip text from a column | `df["COL"].str.replace("T_", "", regex=False)` |
| text to number | `.astype(float)` |
| drop a column | `df.drop(columns=["COL"])` |
| sort | `df.sort_values(["ID", "TIME"])` |
| renumber the index 0, 1, 2 | `df.reset_index(drop=True)` |
| flag repeated rows | `df.duplicated()` is True for every repeat after the first |
| count them | `df.duplicated().sum()` |
| remove them | `df.drop_duplicates()` |
| tidy back to wide | `df.pivot(index="ID", columns="TIME", values="VAL")` |

Two traps this problem is built around.

- `id_vars` lists the columns to **keep as they are**. Every other column gets
  folded into the new label and value columns.
- Sorting on the **text** `T_4` and `T_24` puts `T_24` first, because text
  sorts character by character. Sorting on the **number** 4 and 24 does the
  right thing. This is exactly why you convert before you sort.

The toolkit cell demonstrates each of those on a tiny made up example.
It is in the notebook and none of it is the answer.

## Part a. Wide to tidy

`melt` folds a set of columns down into two: one holding what used to be the
column **names**, one holding the **values**. That is what turns the timepoint
from metadata into data you can actually group, filter and plot against.

`id_vars` is the list of columns to leave alone. Everything you do not list
gets folded, so naming `ANIMAL` and `GROUP` is enough; you do not have to list
the timepoint columns.

Use `var_name` and `value_name` to get the output columns named properly
straight away, otherwise they arrive as `variable` and `value`.

Fill in `tidy`, with exactly these four columns, named exactly like this:

| column | meaning |
|---|---|
| `ANIMAL` | animal identifier |
| `GROUP` | `Vehicle` or `Drug` |
| `TIME_LABEL` | the original column name, so still the text `T_0`, `T_1`, and so on |
| `INHIB_PCT` | the percent inhibition value |

Moves: `wide.melt(id_vars=[...], var_name=..., value_name=...)`.

**Variables this part is graded on:** `tidy`

The starter cell they see:

```python
# Part a. Reshape wide into tidy.
# Columns wanted: ANIMAL, GROUP, TIME_LABEL, INHIB_PCT

tidy = None   # YOUR CODE

print(None if tidy is None else tidy.shape)
print(None if tidy is None else tidy.head())
```

**Every failure message part a can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `tidy` should be a pandas DataFrame, but it is a %s.
- `tidy` is missing the column(s) %s. ,
- `tidy` has the same number of rows as `wide`, so nothing was reshaped. One row per animal per timepoint means many more rows than that.
- `tidy` has %d rows. It should be one row per animal per timepoint, so the row count of `wide` times the number of timepoint columns.
- `TIME_LABEL` holds %d distinct values but there are %d timepoint columns. Only ANIMAL and GROUP should be id_vars.
- `TIME_LABEL` should still hold the original text column names, like T_0 and T_24.
- `tidy` still has the timepoint columns %s, so it has not been reshaped yet. melt turns those column names into the values of one new column. ,
- `tidy` kept melt's default names `variable` and `value`. Pass var_name and value_name so they come out as TIME_LABEL and INHIB_PCT.

## Part b. Make the time column numeric and sort

Your timepoints are still text like `"T_24"`. Strip the prefix and convert, or
every later step treats them as labels rather than numbers.

This matters more than it looks. Sorting text compares character by character,
so `"T_24"` sorts **before** `"T_4"`, and a profile ordered that way is
scrambled in a way that is easy to miss once it is on a plot. Convert first,
then sort.

Finish by resetting the index. Sorting moves the rows but the old row labels
travel with them, so without `drop=True` the index comes out shuffled.

Fill in `tidy_clean`: add a numeric `TIME_H` in hours so `T_24` becomes
`24.0`, drop `TIME_LABEL`, leaving `ANIMAL`, `GROUP`, `TIME_H`, `INHIB_PCT`,
sorted by animal then time, index running 0, 1, 2 and up.

Moves: `.str.replace(..., regex=False)`, `.astype(float)`,
`.drop(columns=...)`, `.sort_values([...])`, `.reset_index(drop=True)`.

**Variables this part is graded on:** `tidy_clean`

The starter cell they see:

```python
# Part b. Numeric TIME_H, drop TIME_LABEL, sort by ANIMAL then TIME_H, reset the index.

tidy_clean = None   # YOUR CODE

print(None if tidy_clean is None else tidy_clean.head(8))
```

**Every failure message part b can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `tidy_clean` should be a pandas DataFrame, but it is a %s.
- `tidy_clean` still has the TIME_LABEL column. Drop it once TIME_H exists.
- `tidy_clean` is missing the column(s) %s. ,
- `tidy_clean` has %d rows but part b should not add or remove any rows, only change and reorder them.
- `TIME_H` is not numeric, its dtype is %s. Strip the T_ prefix and then convert with .astype(float).
- `tidy_clean` has not had its index reset, so it does not run 0, 1, 2 and up. Use .reset_index(drop=True) after sorting.
- `tidy_clean` does not hold the expected set of animal rows.
- `TIME_H` does not hold the right hours. T_0 should become 0.0 and T_24 should become 24.0.
- `INHIB_PCT` does not line up with the animal and time columns. Sort the whole frame at once rather than sorting single columns.
- `tidy_clean` holds the right rows but not in the right order. Sort by ANIMAL first and TIME_H second, in that order.
- `TIME_H` has the right values but the rows are not in the right order. Within each animal, time has to run upwards. Watch out for sorting the text label rather than the number, which puts T_24 before T_4.

## Part c. Find the duplicate, then go back to wide

Try the pivot first, and let it fail. `ValueError: Index contains duplicate
entries` is the reshape telling you two rows claim the same animal and the
same timepoint, so there is no single value to put in that cell. That is a
real data problem surfacing, not a coding mistake.

Here the cause is a record that was sent twice, so dropping the exact repeats
is the right fix. If the two rows had held *different* values it would be a
scientific question instead, and `pivot_table` with an aggregation would be
the tool. Knowing which situation you are in means looking first.

Note that `duplicated()` flags only the repeats and leaves the first
occurrence alone, which is what you want to count. `keep=False` flags both
copies and would give you double.

Fill in `n_dup_rows`, how many rows of `tidy_clean` repeat a row already seen.
And `wide_again`, the duplicate-free data pivoted back to wide with `ANIMAL`
as the index and the values of `TIME_H` as the columns, holding `INHIB_PCT`.

Moves: `.duplicated().sum()`, `.drop_duplicates()`,
`.pivot(index=..., columns=..., values=...)`.

**Variables this part is graded on:** `n_dup_rows`, `wide_again`

The starter cell they see:

```python
# Part c. Count the repeated rows, drop them, pivot back to wide.

n_dup_rows = None   # YOUR CODE: how many rows repeat a row already seen
wide_again = None   # YOUR CODE: pivot of the duplicate-free data

print("n_dup_rows:", n_dup_rows)
print(None if wide_again is None else wide_again)
```

**Every failure message part c can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `n_dup_rows` should be a single whole number.
- `n_dup_rows` counts 1, which is the number of animals that got re-sent. The question asks for rows of tidy_clean, and that animal contributes one row per timepoint.
- `n_dup_rows` is double what is wanted, so you counted both copies. df.duplicated() flags only the repeats, leaving the first occurrence alone, whereas keep=False flags both.
- `n_dup_rows` is not the number of repeated rows. Use tidy_clean.duplicated().sum() across all columns.
- `wide_again` should be a pandas DataFrame, but it is a %s.
- `wide_again` should have ANIMAL as its index. Pass index='ANIMAL' to pivot and do not reset the index afterwards.
- `wide_again` has %d rows but there are %d distinct animals. Drop the repeated rows before pivoting.
- `wide_again` does not have the expected animals on its index.
- `wide_again` does not have the timepoints as its columns. Pass columns='TIME_H' and values='INHIB_PCT' to pivot.
- `wide_again` has the right shape but the wrong cell values. Check that you pivoted INHIB_PCT, and that you dropped duplicates rather than aggregating them into a mean.

**Functions this problem introduces:** .melt, .str accessor, .astype, .drop, .sort_values, .duplicated, .drop_duplicates, .pivot, pd.concat

---

# W1D5. Cells inside animals: aggregate first, then plot mean with SD

- topic: matplotlib and pandas: nested data, animal level aggregation, SD error bars
- notebook: problems/week01/w1d5_matplotlib_nested_sd.ipynb
- solution: solutions/week01/w1d5_matplotlib_nested_sd_solution.ipynb
- expected time: 20-30 min

## Scenario

You have quantified a target engagement marker by immunofluorescence. Twelve
mice, six on vehicle and six on drug. For each mouse you imaged a slide and
measured **many individual cells**, somewhere between 20 and 60 of them, so the
table you are holding has hundreds of rows.

This is the single most commonly botched analysis in cell biology, so it is
worth being slow and deliberate about it.

**The unit of replication is the animal, not the cell.** The cells inside one
mouse are not independent samples of the drug effect; they are repeat measures
of one mouse. If you pool all the cells and treat them as n, you inflate n
from 12 to several hundred, you weight each mouse by however many cells you
happened to image, and any statistic you compute afterwards is wrong. So:

1. aggregate the cells up to one number per animal,
2. then do all group level statistics across animals, with n = animals.

The second house rule: error bars are **SD**, never SEM. An SD bar tells the
reader how much the animals differ from each other, which is what they want to
know. A SEM bar mostly tells them how many animals you ran.

## How matplotlib works, and why the unit of replication matters

### The object model

Almost every confusing matplotlib example online comes from mixing two
different styles. There is `plt.plot(...)`, which draws on whatever figure
happens to be current, a hidden piece of state you cannot see. And there is
the explicit style, where you hold the objects yourself:

```python
fig, ax = plt.subplots(figsize=(6, 4.5))
```

- **`fig`** is the whole picture: the canvas, its size, and what gets saved.
- **`ax`** is one set of axes inside it: the data area, its ticks, its labels,
  and the things drawn on it.

You then tell `ax` what to draw, by calling methods on it. `ax.errorbar(...)`,
`ax.scatter(...)`, `ax.set_ylabel(...)`. Everything is a method on the thing
it affects, which means you can always find the right call by asking "which
object does this belong to". It also means a figure with several panels is no
harder: `plt.subplots(1, 2)` hands you an array of axes, and each one is
driven exactly the same way. Use the explicit style always. The hidden-state
style falls apart the moment a notebook has more than one figure.

Drawing order is controlled by `zorder`, higher being nearer the front, which
is how you keep scattered points from hiding your summary markers.

### The statistics this plot is really about

**Replication is about independence, not sample size.** Two measurements count
as replicates only if they are independent draws from the thing you are
studying. You are studying a drug effect in animals, so two animals are two
replicates. Fifty cells from one animal are fifty measurements of **that one
animal**, sharing its genetics, its dose, its exposure, its slide and its
staining batch. They tell you about variation between cells, which is a real
and sometimes interesting quantity, but it is not the quantity your group
comparison is about.

Treating cells as n does three separate kinds of damage:

1. **n is inflated**, here from 12 to several hundred, so confidence intervals
   collapse and almost anything looks significant. This has a name,
   pseudoreplication, and it is one of the most common fatal flaws in cell
   biology papers.
2. **Animals get unequal weight.** Pooling means the animal you imaged 60
   cells from counts twelve times as much as the one you imaged 5 from. The
   group mean drifts toward whichever animal you had most patience with, which
   is a property of your afternoon, not of the drug.
3. **The error bar answers the wrong question.** The SD of pooled cells mostly
   measures cell to cell variation and staining noise. A reader looking at
   your figure wants to know how reproducible the effect is **between
   animals**, because that is what predicts whether it replicates.

The fix is the two-step aggregation you are about to write: cells up to one
number per animal, then animals up to a group mean and SD. After step one,
every animal is one row, which is exactly the shape every standard statistical
test assumes.

**And the SD versus SEM choice.** SEM is the SD divided by the square root of
n, so it shrinks as you add animals even when the biology is identical. It
describes how precisely you have located the mean, not how much the animals
differ. For a figure whose job is to show a biological effect, SD is the
honest choice, and a reader who knows n can always work out the SEM. A reader
shown SEM cannot work backwards without being told n.

## Your task

Three parts, worked one at a time, and the order matters: cells up to animals
first, then animals up to groups, then the figure.

| Part | What you work out | Variables to fill in |
|---|---|---|
| a | one row per animal, out of hundreds of cells | `animal_level` |
| b | group statistics computed across animals | `group_stats` |
| c | the figure, means with SD and every animal shown | `fig`, `ax` |

Below this you will find the **Setup** cell, then a **Toolkit** cell that
demonstrates every move on a tiny made up example, then **Part a**, **Part b**
and **Part c**, each with its own explanation and a cell to write in.

## The setup cell the student runs

This is given to them, already working. The data is synthetic and
seeded, so their numbers are always these numbers.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROBLEM_ID = "W1D5"
TOPIC = "matplotlib and pandas: nested data, animal level aggregation, SD error bars"

rng = np.random.default_rng(1105)

_group_true = {"Vehicle": 1000.0, "Drug": 420.0}

_frames = []
for _i in range(1, 13):
    _group = "Vehicle" if _i <= 6 else "Drug"
    _animal = "MS%02d" % _i
    _animal_mean = _group_true[_group] * rng.lognormal(0.0, 0.18)   # between animal
    _n_cells = int(rng.integers(20, 61))                            # unequal effort per slide
    _vals = rng.normal(_animal_mean, 0.35 * _animal_mean, size=_n_cells)   # within animal
    _frames.append(pd.DataFrame({
        "ANIMAL": _animal,
        "GROUP": _group,
        "CELL_ID": np.arange(1, _n_cells + 1),
        "INTENSITY": np.round(np.clip(_vals, 0.0, None), 1),
    }))

cells = pd.concat(_frames, ignore_index=True)

print("cells:", cells.shape, " rows are individual cells, not animals")
print(cells.head())
print("")
print("cells measured per animal:")
print(cells.groupby("ANIMAL")["CELL_ID"].size().to_string())
```

## Toolkit for this problem

Two steps, in this order, always: **cells up to animals**, then **animals up
to groups**. Everything below runs on a tiny made up table, so none of it is
the answer.

| What you want to do | How |
|---|---|
| aggregate and name the result | `df.groupby("ID").agg(M=("VAL", "mean"), N=("VAL", "size"))` |
| turn group keys back into columns | `.reset_index()` |
| count rows in a group | the aggregation `"size"` |
| sample SD | the aggregation `"std"`, which pandas already does with `n - 1` |
| a figure with one axes | `fig, ax = plt.subplots(figsize=(6, 4.5))` |
| points with error bars | `ax.errorbar([0, 1], means, yerr=sds, fmt="o", capsize=6)` |
| scattered points on top | `ax.scatter(x, y)` |
| name the ticks | `ax.set_xticks([0, 1])` then `ax.set_xticklabels(["Drug", "Vehicle"])` |
| label the axes | `ax.set_xlabel(...)`, `ax.set_ylabel(...)`, `ax.set_title(...)` |

Why `yerr` must be the SD of the animal means, and not either of the other
two numbers that are easy to reach for:

- the SD of all the **cells** describes how much cells vary inside an animal,
  which is a microscope question, not a drug question,
- the **SEM** is that SD divided by the square root of n, so it shrinks when
  you image more cells, which makes a weak effect look convincing.

The toolkit cell demonstrates each of those on a tiny made up example.
It is in the notebook and none of it is the answer.

## Part a. Aggregate the cells up to the animal

This is the step that makes everything after it valid. One row per animal,
holding that animal's mean across its own cells.

Group by animal and group together, so each animal keeps its treatment label
and you do not have to join it back on afterwards. Naming the outputs inside
`.agg()` saves a rename. The aggregation `"size"` counts rows in each group,
which here means cells per animal.

Keep `N_CELLS` even though nothing downstream divides by it. It is the number
that shows how unequal the imaging effort was, and therefore why pooling the
cells would have distorted the group mean.

Fill in `animal_level`, a flat DataFrame with exactly these four columns:

| column | meaning |
|---|---|
| `ANIMAL` | animal identifier |
| `GROUP` | `Drug` or `Vehicle` |
| `MEAN_INTENSITY` | mean of that animal's own cells |
| `N_CELLS` | how many cells were measured in that animal |

Moves: `.groupby([...]).agg(NAME=("COL", "mean"), ...)`, `.reset_index()`.

**Variables this part is graded on:** `animal_level`

The starter cell they see:

```python
# Part a. One row per animal: ANIMAL, GROUP, MEAN_INTENSITY, N_CELLS

animal_level = None   # YOUR CODE

print(animal_level)
```

**Every failure message part a can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `animal_level` should be a pandas DataFrame, but it is a %s.
- `animal_level` is missing the column(s) %s. Naming them as you aggregate, with .agg(MEAN_INTENSITY=("INTENSITY", "mean"), ...), saves a rename. ,
- `animal_level` still has one row per cell. It needs one row per animal, so group by ANIMAL before aggregating.
- `animal_level` has %d rows but there are %d animals.
- `animal_level` does not list the expected animals.
- `GROUP` does not line up with `ANIMAL`. Group by both at once so each animal keeps its own group label.
- `MEAN_INTENSITY` is the median of each animal's cells, not the mean.
- `MEAN_INTENSITY` is not the plain mean of each animal's own cells.
- `N_CELLS` is not the number of cells measured in each animal. The aggregation "size" counts rows within the group.
- `%s` is in the index rather than a column. Call .reset_index() after the groupby.

## Part b. Group statistics across animals

Build this from `animal_level`, **not** from `cells`. That one choice is the
whole point of the problem.

Every number here is computed across animals. `MEAN` is the mean of the animal
means, which is not the same as the mean of all the cells, because the animals
contributed unequal numbers of cells. `SD` is the spread of the animal means,
which is what tells a reader how reproducible the effect is. `N_ANIMALS`
should come out at 6 per group; if you see numbers in the hundreds you are
counting cells.

Fill in `group_stats`, sorted by `GROUP` which puts `Drug` first, with exactly
these four columns:

| column | meaning |
|---|---|
| `GROUP` | `Drug` or `Vehicle` |
| `MEAN` | mean of the animal means in that group |
| `SD` | **sample** SD of the animal means in that group |
| `N_ANIMALS` | number of animals in that group |

Moves: `.groupby("GROUP").agg(MEAN=("MEAN_INTENSITY", "mean"), ...)`,
`.reset_index()`, `.sort_values(...)`.

**Variables this part is graded on:** `group_stats`

The starter cell they see:

```python
# Part b. One row per group, computed across animals: GROUP, MEAN, SD, N_ANIMALS

group_stats = None   # YOUR CODE

print(group_stats)
```

**Every failure message part b can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `group_stats` should be a pandas DataFrame, but it is a %s.
- `GROUP` is in the index rather than a column. Call .reset_index().
- `group_stats` is missing the column(s) %s. ,
- `group_stats` has %d rows but there are %d groups.
- `group_stats` does not list the expected groups.
- `N_ANIMALS` counted cells, not animals. It came out in the hundreds, which is the inflated n the scenario warns about. Build this table from animal_level, not from cells.
- `N_ANIMALS` is not the number of animals per group.
- `MEAN` is the mean of every cell pooled together, which weights each animal by how many cells you happened to image. Take the mean of the animal means instead, so every animal counts once.
- `MEAN` is not the mean of the animal means within each group.
- `SD` is the SEM, not the SD. It was divided by the square root of the number of animals. This repo always reports SD.
- `SD` is the spread of individual cells, not the spread of animals. It is much larger than the animal level SD, and it answers a question nobody asked. Compute it across the animal means.
- `SD` is not the sample SD of the animal means within each group.

## Part c. The figure

A figure that shows only two means and two error bars hides your whole
dataset. Overlaying every animal costs one line and lets a reader see the
spread, the sample size and any outlier at a glance. Make it your default.

Build it the explicit way: get `fig` and `ax` from `plt.subplots()`, then call
methods on `ax`. `yerr` comes from the `SD` column of `group_stats`, which is
the SD **across animals**, not across cells and not the SEM.

What the figure needs:

- `fig, ax = plt.subplots(figsize=(6, 4.5))`
- the two group means drawn with `ax.errorbar`, at x positions `0` and `1`, in
  the row order of `group_stats`, with `yerr` set to the `SD` column and
  `fmt="o"`
- every animal's own mean overlaid with `ax.scatter`, one point per animal, at
  the x position of its group. A little jitter on x is welcome but optional
- x ticks set to `[0, 1]` and labelled with the group names
- an x label, a y label and a title. An unlabelled axis makes a figure
  unreadable to everyone except the person who made it, including you in three
  weeks

Fill in `fig`, `ax`.
Moves: `plt.subplots(figsize=...)`, `ax.errorbar(..., yerr=..., fmt="o")`,
`ax.scatter(...)`, `ax.set_xticks`, `ax.set_xticklabels`, `ax.set_xlabel`,
`ax.set_ylabel`, `ax.set_title`.

**Variables this part is graded on:** see the starter

The starter cell they see:

```python
# Part c. Group means with SD error bars, every animal overlaid.

fig, ax = plt.subplots(figsize=(6, 4.5))

x_pos = [0, 1]

# YOUR CODE: ax.errorbar(...) for the two group means, yerr = the SD column

# YOUR CODE: ax.scatter(...) for the 12 individual animal means

# YOUR CODE: ax.set_xticks(...), ax.set_xticklabels(...)

# YOUR CODE: ax.set_xlabel(...), ax.set_ylabel(...), ax.set_title(...)

plt.show()
```

**Every failure message part c can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `fig` should be a matplotlib Figure, from plt.subplots().
- `ax` should be a single matplotlib Axes. plt.subplots() with no rows or columns argument gives you exactly one.
- No error bars found on the axes. Draw the group means with ax.errorbar and pass yerr, rather than plotting the means as plain markers.
- The plotted group means do not match the mean of the animal means. Plot the MEAN column of group_stats, and check part b first.
- The error bars are SEM, not SD. They are the SD divided by the square root of the number of animals. This repo always uses SD.
- The error bars are the SD of individual cells, not of animal means. Take yerr from the SD column of group_stats.
- The error bar half heights do not match the SD column of group_stats. Pass yerr=group_stats["SD"] and let errorbar handle the rest; a yerr of shape (2, n) would be read as separate lower and upper bars.
- No scatter points found. Overlay the individual animal means with ax.scatter, one point per animal. ax.plot does not register as a scatter for this check.
- The scatter has one point per cell. Overlay the animal means from animal_level, which is %d points, not every cell.
- The scatter has %d points but there are %d animals, so there should be one point each.
- The x axis has no label. Use ax.set_xlabel.
- The y axis has no label, and a reader cannot tell what the numbers are.
- The axes has no title. Use ax.set_title.
- The x ticks are not exactly [0, 1]. Call ax.set_xticks([0, 1]) so the two groups land where you plotted them.
- The plotted group means come from pooling every cell rather than averaging the animal means, so part b needs fixing first.
- The x tick labels do not name the groups. One of them should read %r. Use ax.set_xticklabels with the group names.

**Functions this problem introduces:** plt.subplots, ax.errorbar, ax.scatter, zorder, axis labels, ax.set_xticks, ax.legend, plt.show, np.where, np.sqrt
