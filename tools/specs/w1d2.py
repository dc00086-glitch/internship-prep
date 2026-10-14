PID = "W1D2"
WEEK = 1
DAY = 2
SLUG = "w1d2_numpy_subject_matrix"
TITLE = "A subject by time matrix: axis reductions, SD, and broadcasting"
TOPIC = "numpy: 2D arrays, axis reductions, sample SD, broadcasting"
EST = "20-30 min"

SCENARIO_MD = '''
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
'''

TASK_MD = '''
## Your task

### Part a. Reduce along the right axis

Both of these are one-line reductions. The only thing to get right is which
direction you collapse, and the rule is that the axis you name is the one that
disappears.

"Mean at each timepoint" means you want one value per column, so the rows have
to go. "Each subject's peak" means one value per row, so the columns have to
go. If you are unsure which you got, print `.shape` of the result and ask
whether that length is 8 or 9.

Fill in `mean_by_time` (length 9), `cmax_by_subject` (length 8).
Moves: `conc.mean(axis=...)`, `conc.max(axis=...)`.
- `mean_by_time`: the mean concentration at each timepoint, so one value per
  **column**. Length 9.
- `cmax_by_subject`: each subject's own peak concentration, so one value per
  **row**. Length 8.

### Part b. Describe the variability
- `sd_by_time`: the **sample** SD across subjects at each timepoint. Length 9.
  Use `n - 1` in the denominator.
- `cv_by_time`: the coefficient of variation at each timepoint as a
  **percent**, which is `100 * SD / mean`. Length 9.

### Part c. Put every subject on a common scale

Normalising each profile to its own peak strips out the amplitude differences
so you can compare the **shape** of absorption and elimination between
subjects. It is a standard first look before any modelling.

You already have each subject's peak from part a, but dividing straight by it
will fail, because a `(8, 9)` and an `(8,)` do not line up: numpy matches
shapes from the right and tries 9 against 8. Read the error when it happens,
it is a useful one to recognise. Make the divisor `(8, 1)` and it works.

Fill in `norm_conc`, shape `(8, 9)`, every row peaking at exactly 1.0.
Moves: `conc.max(axis=1, keepdims=True)`, or `.reshape(-1, 1)`, then divide.
- `norm_conc`: shape `(8, 9)`, where every subject's profile has been divided
  by that **same subject's** own Cmax. Each row must therefore peak at exactly
  `1.0`.

Do this with broadcasting rather than a `for` loop. A `(8, 9)` array divides
cleanly by an `(8, 1)` array, but not by an `(8,)` array.

### Hints
- `axis=0` collapses rows and leaves you one value per column.
- `axis=1` collapses columns and leaves you one value per row.
- `np.std(x, axis=..., ddof=1)` is the sample SD.
- `keepdims=True` keeps the reduced axis as a length 1 axis, which is exactly
  what makes broadcasting line up. Otherwise `.reshape(-1, 1)` does the same.
'''

DATA_CODE = '''
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
'''

PARTS = [
    dict(
        label="a",
        prompt_md='''
### Part a. Reduce along the right axis
''',
        starter='''
# Part a. Reduce along the right axis.

mean_by_time = None       # YOUR CODE: mean at each timepoint, length 9
cmax_by_subject = None    # YOUR CODE: each subject's peak, length 8

print("mean_by_time   ", np.shape(mean_by_time), mean_by_time)
print("cmax_by_subject", np.shape(cmax_by_subject), cmax_by_subject)
''',
        solution='''
# Part a. Reduce along the right axis.

mean_by_time = conc.mean(axis=0)
cmax_by_subject = conc.max(axis=1)

print("mean_by_time   ", np.shape(mean_by_time), mean_by_time)
print("cmax_by_subject", np.shape(cmax_by_subject), cmax_by_subject)
''',
    ),
    dict(
        label="b",
        prompt_md='''
### Part b. Describe the variability

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
''',
        starter='''
# Part b. Describe the variability. SD, not SEM. Sample SD, so ddof=1.

sd_by_time = None   # YOUR CODE: sample SD across subjects at each timepoint
cv_by_time = None   # YOUR CODE: 100 * SD / mean at each timepoint

print("sd_by_time", np.shape(sd_by_time), np.round(sd_by_time, 3) if sd_by_time is not None else None)
print("cv_by_time", np.shape(cv_by_time), np.round(cv_by_time, 1) if cv_by_time is not None else None)
''',
        solution='''
# Part b. Describe the variability. SD, not SEM. Sample SD, so ddof=1.

sd_by_time = conc.std(axis=0, ddof=1)
cv_by_time = 100.0 * sd_by_time / mean_by_time

print("sd_by_time", np.shape(sd_by_time), np.round(sd_by_time, 3) if sd_by_time is not None else None)
print("cv_by_time", np.shape(cv_by_time), np.round(cv_by_time, 1) if cv_by_time is not None else None)
''',
    ),
    dict(
        label="c",
        prompt_md='''
### Part c. Put every subject on a common scale
''',
        starter='''
# Part c. Normalise each subject's profile to that subject's own Cmax.
# Use broadcasting, not a for loop.

norm_conc = None   # YOUR CODE: shape (8, 9), every row peaking at 1.0

if norm_conc is not None:
    print("norm_conc shape:", np.shape(norm_conc))
    print("max of each row:", np.round(np.max(norm_conc, axis=1), 6))
''',
        solution='''
# Part c. Normalise each subject's profile to that subject's own Cmax.
# Use broadcasting, not a for loop.

norm_conc = conc / conc.max(axis=1, keepdims=True)

if norm_conc is not None:
    print("norm_conc shape:", np.shape(norm_conc))
    print("max of each row:", np.round(np.max(norm_conc, axis=1), 6))
''',
    ),
]


def build_expected(ns):
    import numpy as np
    conc = ns["conc"]
    mean_by_time = conc.mean(axis=0)
    return {
        "n_subjects": int(conc.shape[0]),
        "n_times": int(conc.shape[1]),
        "mean_by_time": mean_by_time.tolist(),
        "mean_by_subject": conc.mean(axis=1).tolist(),
        "cmax_by_subject": conc.max(axis=1).tolist(),
        "cmax_by_time": conc.max(axis=0).tolist(),
        "sd_by_time": conc.std(axis=0, ddof=1).tolist(),
        "sd_by_time_pop": conc.std(axis=0, ddof=0).tolist(),
        "sem_by_time": (conc.std(axis=0, ddof=1) / np.sqrt(conc.shape[0])).tolist(),
        "sd_by_subject": conc.std(axis=1, ddof=1).tolist(),
        "cv_by_time": (100.0 * conc.std(axis=0, ddof=1) / mean_by_time).tolist(),
        "cv_fraction": (conc.std(axis=0, ddof=1) / mean_by_time).tolist(),
        "norm_conc": conc / conc.max(axis=1, keepdims=True),
        "norm_by_overall": (conc / conc.max()),
        "norm_by_time": (conc / conc.max(axis=0, keepdims=True)),
    }


CHECK_CODE = '''
def _part_a():
    mean_by_time = _need("mean_by_time")
    cmax_by_subject = _need("cmax_by_subject")
    n_sub, n_time = _EXP["n_subjects"], _EXP["n_times"]

    arr = _as_array(mean_by_time)
    if arr is None:
        return "`mean_by_time` should be a numeric array."
    if arr.ndim == 0:
        return ("`mean_by_time` is a single number, so you averaged the whole matrix. "
                "Pass an axis argument so the reduction only collapses one direction.")
    if arr.shape == (n_sub,):
        return ("`mean_by_time` has one value per subject, not per timepoint. You collapsed "
                "the wrong axis: to keep one value per column you have to collapse the rows.")
    if arr.shape != (n_time,):
        return "`mean_by_time` has shape %s but it should be (%d,), one value per timepoint." % (arr.shape, n_time)
    if not _arr_close(arr, _EXP["mean_by_time"]):
        return "`mean_by_time` has the right shape but the wrong values. It is a plain mean of conc, nothing else."

    arr = _as_array(cmax_by_subject)
    if arr is None:
        return "`cmax_by_subject` should be a numeric array."
    if arr.ndim == 0:
        return ("`cmax_by_subject` is a single number, so you took the max of the whole "
                "matrix. Pass an axis argument.")
    if arr.shape == (n_time,):
        return ("`cmax_by_subject` has one value per timepoint, not per subject. You "
                "collapsed the wrong axis: to keep one value per row you have to collapse "
                "the columns.")
    if arr.shape != (n_sub,):
        return "`cmax_by_subject` has shape %s but it should be (%d,), one value per subject." % (arr.shape, n_sub)
    if _arr_close(arr, _EXP["mean_by_subject"]):
        return "`cmax_by_subject` is each subject's mean, not each subject's peak."
    if not _arr_close(arr, _EXP["cmax_by_subject"]):
        return "`cmax_by_subject` has the right shape but the wrong values."
    return None


def _part_b():
    sd_by_time = _need("sd_by_time")
    cv_by_time = _need("cv_by_time")
    n_sub, n_time = _EXP["n_subjects"], _EXP["n_times"]

    arr = _as_array(sd_by_time)
    if arr is None:
        return "`sd_by_time` should be a numeric array."
    if arr.shape == (n_sub,):
        return ("`sd_by_time` has one value per subject. The SD you want is across subjects "
                "at a fixed timepoint, so collapse the rows and keep the columns.")
    if arr.shape != (n_time,):
        return "`sd_by_time` has shape %s but it should be (%d,), one value per timepoint." % (arr.shape, n_time)
    if _arr_close(arr, _EXP["sem_by_time"]):
        return ("`sd_by_time` is the SEM, not the SD. You divided by the square root of the "
                "number of subjects somewhere. This repo always reports SD.")
    if _arr_close(arr, _EXP["sd_by_time_pop"]):
        return ("`sd_by_time` used the population SD with n in the denominator. np.std "
                "defaults to ddof=0, so pass ddof=1 to get the sample SD.")
    if not _arr_close(arr, _EXP["sd_by_time"]):
        return "`sd_by_time` has the right shape but the wrong values."

    arr = _as_array(cv_by_time)
    if arr is None:
        return "`cv_by_time` should be a numeric array."
    if arr.shape != (n_time,):
        return "`cv_by_time` has shape %s but it should be (%d,), one value per timepoint." % (arr.shape, n_time)
    if _arr_close(arr, _EXP["cv_fraction"]):
        return "`cv_by_time` is a fraction. The task asks for a percent, so multiply by 100."
    if not _arr_close(arr, _EXP["cv_by_time"]):
        return ("`cv_by_time` is not 100 * SD / mean at each timepoint. Check that you paired "
                "your SD and your mean at the same timepoints, and that both are per timepoint.")
    return None


def _part_c():
    norm_conc = _need("norm_conc")
    n_sub, n_time = _EXP["n_subjects"], _EXP["n_times"]

    arr = _as_array(norm_conc)
    if arr is None:
        return "`norm_conc` should be a numeric array."
    if arr.shape != (n_sub, n_time):
        if arr.shape == (n_time, n_sub):
            return ("`norm_conc` has shape %s, which is the transpose of what is wanted. "
                    "Keep subjects on the rows." % (arr.shape,))
        return ("`norm_conc` has shape %s but it should be (%d, %d), the same shape as conc. "
                "If you got an error about shapes not aligning, that is the (8,) against "
                "(8, 9) problem the hint warns about." % (arr.shape, n_sub, n_time))

    row_max = _np.max(arr, axis=1)
    n_at_one = int(_np.sum(_np.isclose(row_max, 1.0)))
    if _arr_close(arr, _EXP["norm_by_overall"]):
        return ("`norm_conc` was divided by the single largest value in the whole matrix, so "
                "only one subject peaks at 1.0. Each subject needs their own Cmax.")
    if _arr_close(arr, _EXP["norm_by_time"]):
        return ("`norm_conc` was divided by the peak at each timepoint, which is a reduction "
                "down the columns. You want the peak of each subject, which is across the rows.")
    if n_at_one != n_sub:
        return ("`norm_conc` has the right shape but only %d of %d rows peak at exactly 1.0. "
                "Divide each row by the maximum of that same row." % (n_at_one, n_sub))
    if not _arr_close(arr, _EXP["norm_conc"]):
        return "`norm_conc` has the right shape and every row peaks at 1.0, but the other values do not match."
    return None
'''


TOOLKIT_MD = '''
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
'''

TOOLKIT_CODE = '''
import numpy as np

# A tiny made up example: 2 rows, 3 columns. Nothing here is the answer.
demo = np.array([[1.0, 2.0, 3.0],
                 [4.0, 5.0, 9.0]])

print("demo:")
print(demo)
print("shape:", demo.shape, " <- (rows, columns)")

print("")
print("--- 1. the axis you name is the one that disappears ---")
print("demo.mean(axis=0) :", demo.mean(axis=0), " length", demo.mean(axis=0).size,
      " <- rows gone, one value per COLUMN")
print("demo.mean(axis=1) :", demo.mean(axis=1), " length", demo.mean(axis=1).size,
      " <- columns gone, one value per ROW")
print("demo.max(axis=1)  :", demo.max(axis=1), " <- biggest in each row")

print("")
print("--- 2. sample SD versus population SD ---")
print("std(axis=0)          :", np.round(demo.std(axis=0), 4), " <- divides by n, the default")
print("std(axis=0, ddof=1)  :", np.round(demo.std(axis=0, ddof=1), 4), " <- divides by n-1, what we want")

print("")
print("--- 3. why plain max(axis=1) will not divide cleanly ---")
demo_rowmax = demo.max(axis=1)
print("demo_rowmax       :", demo_rowmax, " shape", demo_rowmax.shape)
try:
    demo / demo_rowmax
except ValueError as err:
    print("demo / demo_rowmax fails with:", err)

print("")
print("--- 4. keepdims makes the shapes line up ---")
demo_rowmax_col = demo.max(axis=1, keepdims=True)
print("with keepdims     :", demo_rowmax_col.tolist(), " shape", demo_rowmax_col.shape)
print("demo / that:")
print(demo / demo_rowmax_col, " <- every row now peaks at 1.0")
print("")
print("reshape(-1, 1) does the same thing:", demo_rowmax.reshape(-1, 1).shape)
'''


CONCEPT_MD = '''
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
'''
