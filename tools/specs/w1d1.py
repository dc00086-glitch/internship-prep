PID = "W1D1"
WEEK = 1
DAY = 1
SLUG = "w1d1_numpy_pk_profile"
TITLE = "Clean one plasma profile and get the terminal half-life"
TOPIC = "numpy: 1D arrays, nan masking, log-linear terminal slope"
EST = "20-30 min"

SCENARIO_MD = '''
## Scenario

One healthy volunteer received a single 100 mg intramuscular dose of a new
compound. Plasma was sampled at 11 timepoints out to 36 hours and assayed by
LC-MS/MS. The bioanalytical lab reports any sample below the lower limit of
quantification (LLOQ, here 0.5 mg/L) as BLQ, and in the array you are handed
those arrive as `np.nan`.

This is the first pass a pharmacometrician makes on any new profile: clean the
record, describe the peak, then estimate how fast the drug leaves. Everything
here is plain numpy, no pandas.
'''

TASK_MD = '''
## Your task

### Part a. Deal with the BLQ samples

The lab handed you `nan` wherever a sample fell below the assay limit. Until
you know which samples are real, no summary of this profile means anything,
so this is always step one.

Build a mask of the usable samples, count the ones that are not, then use that
mask to pull the usable times and concentrations out. The same mask works on
both arrays because they are the same length and in the same order.

Fill in `measurable`, `n_blq`, `t_meas`, `c_meas`.
Moves: `np.isnan`, `~`, `np.sum`, and indexing an array with a mask.
- `measurable`: a boolean array, `True` where `conc` holds a real number and
  `False` where the sample came back BLQ.
- `n_blq`: a whole number, how many samples came back BLQ.
- `t_meas`, `c_meas`: the times and the concentrations of the measurable
  samples only, kept in the original time order.

### Part b. Describe the peak
Using the measurable samples only:
- `cmax`: the highest observed concentration, in mg/L.
- `tmax`: the time in hours at which `cmax` was observed.

### Part c. Terminal half-life
After absorption is over the profile is log-linear, so a straight line through
the log of concentration gives you the elimination rate constant. Use the
**last 4 measurable samples**:
- `t_term`, `c_term`: those last 4 times and concentrations.
- `lambda_z`: the terminal elimination rate constant in 1/h. Fit a degree 1
  polynomial to `np.log(c_term)` against `t_term`, then take `lambda_z` as the
  negative of the fitted slope.
- `t_half`: the terminal half-life in hours, which is `ln(2) / lambda_z`.

### Hints
- `np.isnan(x)` gives you a boolean array, and `~` flips a boolean array.
- A boolean array can be used to index another array of the same length.
- `np.argmax` gives you the position of the largest value, not the value.
- `np.polyfit(x, y, deg=1)` returns `[slope, intercept]` in that order.
- Negative indexing: `a[-4:]` is the last 4 entries of `a`.
'''

DATA_CODE = '''
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
'''

PARTS = [
    dict(
        label="a",
        prompt_md='''
### Part a. Deal with the BLQ samples
''',
        starter='''
# Part a. Deal with the BLQ samples.

measurable = None   # YOUR CODE: boolean array, True where conc is measurable
n_blq = None        # YOUR CODE: whole number, how many samples are BLQ
t_meas = None       # YOUR CODE: times of the measurable samples
c_meas = None       # YOUR CODE: concentrations of the measurable samples

print("measurable:", measurable)
print("n_blq     :", n_blq)
print("t_meas    :", t_meas)
print("c_meas    :", c_meas)
''',
        solution='''
# Part a. Deal with the BLQ samples.

measurable = ~np.isnan(conc)
n_blq = int(np.sum(np.isnan(conc)))
t_meas = time_h[measurable]
c_meas = conc[measurable]

print("measurable:", measurable)
print("n_blq     :", n_blq)
print("t_meas    :", t_meas)
print("c_meas    :", c_meas)
''',
    ),
    dict(
        label="b",
        prompt_md='''
### Part b. Describe the peak

Cmax and Tmax are two questions about the same point on the curve: how high
did it get, and when. `np.max` answers the first. The second needs
`np.argmax`, which hands you a **position**, and you then look that position
up in the time array to turn it into an hour.

Work from the measurable samples, so `c_meas` and `t_meas`, not `conc` and
`time_h`. Taking a max over an array containing `nan` gives you `nan`.

Fill in `cmax`, `tmax`.
Moves: `np.max`, `np.argmax`, and indexing with the position you get back.
''',
        starter='''
# Part b. Describe the peak.

cmax = None   # YOUR CODE: highest observed concentration, mg/L
tmax = None   # YOUR CODE: the time in hours at which cmax was observed

print("cmax:", cmax, "mg/L")
print("tmax:", tmax, "h")
''',
        solution='''
# Part b. Describe the peak.

cmax = float(np.max(c_meas))
tmax = float(t_meas[np.argmax(c_meas)])

print("cmax:", cmax, "mg/L")
print("tmax:", tmax, "h")
''',
    ),
    dict(
        label="c",
        prompt_md='''
### Part c. Terminal half-life

Once absorption is finished, elimination is first order, meaning a constant
fraction leaves per hour. On a log scale that is a straight line, so fitting
a line to `log(concentration)` against time recovers the rate constant.

Slice off the last 4 measurable samples, fit the line, and convert. The fitted
slope is negative because the drug is going away, and `lambda_z` is defined as
the positive version of it. The half-life is then `ln(2) / lambda_z`.

Fill in `t_term`, `c_term`, `lambda_z`, `t_half`.
Moves: `a[-4:]`, `np.log`, `np.polyfit(x, y, deg=1)` which returns
`[slope, intercept]`.
''',
        starter='''
# Part c. Terminal half-life from the last 4 measurable samples.

t_term = None     # YOUR CODE: last 4 measurable times
c_term = None     # YOUR CODE: last 4 measurable concentrations
lambda_z = None   # YOUR CODE: terminal rate constant, 1/h
t_half = None     # YOUR CODE: terminal half-life, h

print("t_term  :", t_term)
print("c_term  :", c_term)
print("lambda_z:", lambda_z, "1/h")
print("t_half  :", t_half, "h")
''',
        solution='''
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
''',
    ),
]


def build_expected(ns):
    import numpy as np
    return {
        "measurable": [bool(v) for v in ns["measurable"]],
        "n_blq": int(ns["n_blq"]),
        "n_meas": int(len(ns["t_meas"])),
        "t_meas": list(map(float, ns["t_meas"])),
        "c_meas": list(map(float, ns["c_meas"])),
        "cmax": float(ns["cmax"]),
        "tmax": float(ns["tmax"]),
        "t_term": list(map(float, ns["t_term"])),
        "c_term": list(map(float, ns["c_term"])),
        "lambda_z": float(ns["lambda_z"]),
        "t_half": float(ns["t_half"]),
        # a wrong answer we want to name specifically
        "t_half_all_points": float(np.log(2.0) / -np.polyfit(ns["t_meas"], np.log(ns["c_meas"]), deg=1)[0]),
    }


CHECK_CODE = '''
def _part_a():
    measurable = _need("measurable")
    n_blq = _need("n_blq")
    t_meas = _need("t_meas")
    c_meas = _need("c_meas")

    mask = _np.asarray(measurable)
    if mask.dtype != bool:
        return ("`measurable` should be a boolean array but its dtype is %s. "
                "np.isnan returns booleans already, so no comparison is needed." % mask.dtype)
    if mask.shape != (len(_EXP["measurable"]),):
        return ("`measurable` has shape %s. It needs one entry per sample, so "
                "it should be the same length as time_h." % (mask.shape,))
    if not _np.array_equal(mask, _np.asarray(_EXP["measurable"], dtype=bool)):
        return ("`measurable` is flagging the wrong samples. True has to mean the "
                "sample IS usable, so you probably need to flip the result of np.isnan with ~.")

    got = _as_scalar(n_blq)
    if got is None:
        return "`n_blq` should be a single whole number, but it is not a single number."
    if int(round(got)) != _EXP["n_blq"]:
        return ("`n_blq` is not the number of BLQ samples. Count the entries where "
                "conc is nan, which is the opposite of your measurable mask.")

    for name, val in (("t_meas", t_meas), ("c_meas", c_meas)):
        arr = _as_array(val)
        if arr is None or arr.ndim != 1:
            return "`%s` should be a 1D numeric array." % name
        if arr.size == len(_EXP["measurable"]):
            return ("`%s` still has every sample in it. Index the full array with your "
                    "boolean mask so that only the measurable samples survive." % name)
        if arr.size != _EXP["n_meas"]:
            return ("`%s` holds %d values, which does not match the number of measurable "
                    "samples." % (name, arr.size))
    if _np.any(_np.isnan(_as_array(c_meas))):
        return "`c_meas` still contains nan. Those are the BLQ samples you were meant to drop."
    if not _arr_close(t_meas, _EXP["t_meas"]):
        return ("`t_meas` has the right length but the wrong times. Index time_h with the "
                "same mask you used for conc, and do not sort or reorder it.")
    if not _arr_close(c_meas, _EXP["c_meas"]):
        return "`c_meas` has the right length but the wrong values. Keep the original time order."
    return None


def _part_b():
    cmax = _need("cmax")
    tmax = _need("tmax")

    got_c = _as_scalar(cmax)
    if got_c is None:
        return "`cmax` should be a single number. np.max returns one value; np.maximum does not."
    if not _num_close(got_c, _EXP["cmax"]):
        if _np.isnan(got_c):
            return ("`cmax` came out as nan, so you took the max over the raw conc array. "
                    "Use the measurable values from part a, or reach for np.nanmax.")
        if _num_close(got_c, _EXP["t_meas"][int(_np.argmax(_EXP["c_meas"]))]):
            return "`cmax` looks like a time, not a concentration. You may have swapped cmax and tmax."
        return ("`cmax` is not the highest measurable concentration. It should come straight "
                "out of c_meas with no averaging or rounding.")

    got_t = _as_scalar(tmax)
    if got_t is None:
        return "`tmax` should be a single number."
    if _num_close(got_t, _EXP["cmax"]):
        return "`tmax` looks like a concentration. You may have swapped cmax and tmax."
    if _num_close(got_t, float(_np.argmax(_EXP["c_meas"]))):
        return ("`tmax` is the index of the peak, not the time of the peak. Feed that index "
                "back into your time array.")
    if not _num_close(got_t, _EXP["tmax"]):
        return ("`tmax` is not the time of the peak. Use np.argmax on the concentrations to "
                "get a position, then read the matching entry of t_meas.")
    return None


def _part_c():
    t_term = _need("t_term")
    c_term = _need("c_term")
    lambda_z = _need("lambda_z")
    t_half = _need("t_half")

    for name, val in (("t_term", t_term), ("c_term", c_term)):
        arr = _as_array(val)
        if arr is None or arr.ndim != 1:
            return "`%s` should be a 1D numeric array." % name
        if arr.size != 4:
            return ("`%s` holds %d values but the task asks for the last 4 measurable "
                    "samples. Slicing with [-4:] gives you exactly 4." % (name, arr.size))
    if _np.any(_np.isnan(_as_array(c_term))):
        return "`c_term` contains nan, so you sliced the raw conc array instead of c_meas."
    if not _arr_close(t_term, _EXP["t_term"]):
        return "`t_term` is not the last 4 measurable times. Slice t_meas from part a, not time_h."
    if not _arr_close(c_term, _EXP["c_term"]):
        return "`c_term` is not the last 4 measurable concentrations. Slice c_meas from part a."

    got_lz = _as_scalar(lambda_z)
    if got_lz is None:
        return ("`lambda_z` should be a single number. np.polyfit returns an array of "
                "coefficients, so pull out the one you need.")
    if _num_close(got_lz, -_EXP["lambda_z"]):
        return ("`lambda_z` is negative. The fitted slope is negative because the drug is "
                "going away, and lambda_z is defined as the positive version of it.")
    if not _num_close(got_lz, _EXP["lambda_z"], rtol=1e-3):
        if _num_close(got_lz, _EXP["lambda_z"], rtol=0.5):
            return ("`lambda_z` is close but not right. Check that you regressed the natural "
                    "log of concentration (np.log, not np.log10) against time, and that you "
                    "took the slope rather than the intercept.")
        return ("`lambda_z` is not the terminal rate constant. Fit np.polyfit(t_term, "
                "np.log(c_term), deg=1); the first returned value is the slope.")

    got_th = _as_scalar(t_half)
    if got_th is None:
        return "`t_half` should be a single number."
    if _num_close(got_th, _EXP["t_half_all_points"], rtol=1e-2):
        return ("`t_half` was computed from the whole profile, not just the terminal phase. "
                "The early points are still absorbing, which biases the slope.")
    if not _num_close(got_th, _EXP["t_half"], rtol=1e-3):
        if _num_close(got_th, 1.0 / got_lz, rtol=1e-3):
            return ("`t_half` is the mean residence time style 1/lambda_z, not the half-life. "
                    "A half-life carries a factor of ln(2).")
        return ("`t_half` does not follow from your lambda_z. It should be np.log(2) divided "
                "by lambda_z, in hours.")
    return None
'''


TOOLKIT_MD = '''
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
'''

TOOLKIT_CODE = '''
import numpy as np

# A tiny made up example. Nothing here is the answer to the problem.
demo_t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])          # make an array from a list
demo_c = np.array([5.0, 9.0, np.nan, 6.0, 2.0])       # nan stands in for a missing value

print("demo_t            :", demo_t)
print("demo_c            :", demo_c)
print("how many entries  :", demo_c.size)

print("")
print("--- 1. finding the missing values ---")
print("np.isnan(demo_c)  :", np.isnan(demo_c))        # True where the value is MISSING
print("~np.isnan(demo_c) :", ~np.isnan(demo_c))       # True where the value is USABLE
print("count of nan      :", int(np.sum(np.isnan(demo_c))))

print("")
print("--- 2. using a True/False array to pick entries out ---")
demo_keep = ~np.isnan(demo_c)
print("times kept        :", demo_t[demo_keep])       # same mask works on both arrays
print("values kept       :", demo_c[demo_keep])

print("")
print("--- 3. the biggest value, and when it happened ---")
demo_good_t = demo_t[demo_keep]
demo_good_c = demo_c[demo_keep]
print("biggest value     :", np.max(demo_good_c))
print("its position      :", np.argmax(demo_good_c), " <- a position, not a value")
print("time at that spot :", demo_good_t[np.argmax(demo_good_c)], " <- feed the position back in")

print("")
print("--- 4. the last few entries ---")
print("last 2 times      :", demo_good_t[-2:])
print("last 2 values     :", demo_good_c[-2:])

print("")
print("--- 5. natural log and a straight line fit ---")
print("np.log(values)    :", np.log(demo_good_c))
demo_slope, demo_intercept = np.polyfit(demo_good_t, np.log(demo_good_c), deg=1)
print("slope             :", round(demo_slope, 4), " <- negative when the line goes down")
print("intercept         :", round(demo_intercept, 4))
'''


CONCEPT_MD = '''
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
'''


# Functions this problem introduces. Keys must exist in tools/function_catalog.py.
FUNCTIONS = [
    'np.random.default_rng',
    'np.array',
    '.size',
    'np.isnan',
    '~',
    'boolean indexing',
    'np.sum',
    'np.max / np.min',
    'np.argmax',
    'np.log',
    'np.polyfit',
    'len',
]
