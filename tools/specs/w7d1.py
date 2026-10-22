PID = "W7D1"
WEEK = 7
DAY = 1
SLUG = "w7d1_nca_auc_trapezoid"
TITLE = "Non-compartmental analysis: Cmax, Tmax and AUC by trapezoid"
TOPIC = "PK: linear and linear-up log-down trapezoidal AUC, partial AUC"
EST = "20-30 min"

SCENARIO_MD = '''
## Scenario

One subject, a single 200 mg oral dose, eleven plasma samples over 48 hours.
You have been asked for the standard non-compartmental summary that goes into
every PK report: how high, how soon, and how much exposure in total.

Non-compartmental analysis earns its name by assuming nothing. No compartments,
no absorption model, no differential equations. It computes exposure directly
from the observed concentrations by joining the dots and measuring the area
underneath. That is its strength, because there is no model to be wrong about,
and its limitation, because it cannot extrapolate to a dose or a schedule you
did not run.

The only real decision is **how** you join the dots, and that decision changes
the answer by a few percent. Regulators have a preferred method, and it is not
the obvious one.
'''

CONCEPT_MD = '''
## Joining the dots, and why straight lines are wrong on the way down

**The trapezoid rule.** Between two samples, treat the profile as a straight
line and the area under that segment is the average of the two heights times
the time between them:

```
area = (c1 + c2) / 2 * (t2 - t1)
```

Sum those over all the intervals and you have the area under the curve from
the first to the last sample, written `AUC_last`. In numpy that is one
vectorised line: take `np.diff` of the times, average neighbouring
concentrations, multiply and sum. No loop required.

**Why that is biased on the descending limb.** After absorption is over, the
profile decays exponentially, which is **convex**: it curves upward away from
the straight line joining its ends. A straight line therefore sits above the
true curve throughout the interval, and linear trapezoids systematically
**overestimate** the area on the way down. The longer the sampling interval
and the faster the decay, the worse it gets, and the terminal intervals are
usually the longest ones you have.

**Linear up, log down.** The standard fix, and what regulatory NCA software
does by default. While concentrations are **rising**, use the linear
trapezoid. Once they are **falling**, assume exponential decay within the
interval, which integrates exactly to

```
area = (c1 - c2) * (t2 - t1) / ln(c1 / c2)
```

That formula is exact if the decay really is exponential across the interval,
which on the terminal limb it very nearly is.

Three practical points. It is undefined when `c1 == c2`, since the log of 1 is
zero and you divide by it, so fall back to the linear form there. It cannot
handle a concentration of zero or a value below the limit of quantification,
so those intervals need the linear form too. And it only ever gives a
**smaller** area than the linear rule on a descending interval, which is why
the log-down total is always the lower and more accurate of the two.

How much lower depends entirely on how long your terminal intervals are. With
dense sampling the two agree to well under a percent, because a straight line
over a short interval barely deviates from the curve. This profile has final
intervals of 12 and 24 hours, which is sparse, and the two methods differ by
about six percent. That number measures your sampling schedule as much as it
measures the drug.

**Partial AUC.** Exposure over a specified window, such as AUC from 0 to 12
hours, which matters when the clinical effect is tied to a dosing interval
rather than to total exposure. If the window ends on a sampling time it is
just a sum over fewer intervals. If it does not, you have to interpolate, and
on a descending limb you should interpolate the same way you integrate, which
is logarithmically.
'''

TASK_MD = '''
## Your task

Three parts, worked one at a time.

| Part | What you work out | Variables to fill in |
|---|---|---|
| a | the peak, and the area by straight lines | `cmax`, `tmax`, `auc_linear` |
| b | the area the regulatory way | `auc_linlog`, `pct_diff` |
| c | exposure over the first 12 hours only | `auc_0_12`, `frac_in_12h` |

Below this you will find the **Setup** cell, then a **Toolkit** cell that
demonstrates every move on a tiny made up example, then **Part a**, **Part b**
and **Part c**, each with its own explanation and a cell to write in.
'''

DATA_CODE = '''
import numpy as np
import pandas as pd

PROBLEM_ID = "W7D1"
TOPIC = "PK: linear and linear-up log-down trapezoidal AUC, partial AUC"

rng = np.random.default_rng(7101)

DOSE_MG = 200.0

time_h = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 24.0, 48.0])

# One compartment with first order absorption, plus a little assay noise.
_F, _V, _ka, _kel = 0.80, 48.0, 1.15, 0.0935
_true = (_F * DOSE_MG * _ka) / (_V * (_ka - _kel)) * (
    np.exp(-_kel * time_h) - np.exp(-_ka * time_h))
conc = np.round(_true * rng.lognormal(0.0, 0.06, size=time_h.size), 4)
conc[0] = 0.0      # predose

profile = pd.DataFrame({"TIME_H": time_h, "CONC_MG_L": conc})

print("dose:", DOSE_MG, "mg oral")
print(profile.to_string(index=False))
print("")
print("sampling intervals (h):", np.diff(time_h))
print("note the last two intervals are 12 and 24 hours long")
'''

TOOLKIT_MD = '''
## Toolkit for this problem

| What you want to do | How |
|---|---|
| gaps between samples | `np.diff(t)` |
| average of neighbouring values | `(c[:-1] + c[1:]) / 2` |
| linear trapezoid area | `np.sum(np.diff(t) * (c[:-1] + c[1:]) / 2)` |
| which intervals are falling | `c[1:] < c[:-1]` |
| log-down area for one interval | `(c1 - c2) * (t2 - t1) / np.log(c1 / c2)` |
| choose per interval | `np.where(condition, log_form, linear_form)` |
| guard a division | compute both, then select, or use `np.errstate` |
| peak and its time | `np.max(c)`, `t[np.argmax(c)]` |

`np.diff` returns one fewer element than it was given, so `c[:-1]` and `c[1:]`
line up with it: each entry is the start and the end of one interval.
'''

TOOLKIT_CODE = '''
import numpy as np

# A tiny made up profile that rises then falls. Nothing here is the answer.
demo_t = np.array([0.0, 1.0, 2.0, 4.0, 8.0])
demo_c = np.array([0.0, 6.0, 8.0, 5.0, 1.5])
print("demo_t:", demo_t)
print("demo_c:", demo_c)

print("")
print("--- 1. intervals line up with the pairs of points ---")
demo_dt = np.diff(demo_t)
demo_c1, demo_c2 = demo_c[:-1], demo_c[1:]
print("np.diff(t) :", demo_dt, " <- %d intervals from %d points" % (demo_dt.size, demo_c.size))
print("start conc :", demo_c1)
print("end conc   :", demo_c2)

print("")
print("--- 2. the linear trapezoid, one vectorised line ---")
demo_lin_each = demo_dt * (demo_c1 + demo_c2) / 2.0
print("area of each interval:", np.round(demo_lin_each, 4))
print("total AUC            : %.4f" % float(demo_lin_each.sum()))

print("")
print("--- 3. which intervals are going down ---")
demo_falling = demo_c2 < demo_c1
print("falling?:", demo_falling)

print("")
print("--- 4. the log-down formula, and why it is smaller ---")
with np.errstate(divide="ignore", invalid="ignore"):
    demo_log_each = (demo_c1 - demo_c2) * demo_dt / np.log(demo_c1 / demo_c2)
print("log form per interval:", np.round(demo_log_each, 4))
print("linear per interval  :", np.round(demo_lin_each, 4))
print("")
print("on the FALLING intervals the log form is smaller, because an exponential")
print("curve sits below the straight line joining its ends")
print("on the rising intervals and the one starting at zero it is unusable:")
print("  dividing by log(0/6) or log(6/6) gives nan or a divide by zero")

print("")
print("--- 5. combining them with np.where ---")
demo_safe = np.where(demo_falling & (demo_c2 > 0) & (demo_c1 != demo_c2),
                     demo_log_each, demo_lin_each)
print("chosen per interval:", np.round(demo_safe, 4))
print("linear AUC  : %.4f" % float(demo_lin_each.sum()))
print("lin-up/log-down: %.4f   <- slightly smaller, as expected" % float(demo_safe.sum()))
print("")
print("np.where evaluates BOTH branches, so compute the log form inside")
print("np.errstate to keep the warnings quiet, then select.")
'''

PARTS = [
    dict(
        label="a",
        prompt_md='''
## Part a. The peak, and the area by straight lines

`cmax` is the highest observed concentration and `tmax` is the time at which
it occurred. Both come straight from the observed data: NCA does not fit
anything, so the peak is simply the largest sample. Note the consequence,
which matters when you report it: Tmax can only ever be one of your sampling
times, so it is as much a property of your sampling schedule as of the drug.

Then the linear trapezoidal AUC from time zero to the last sample. Take
`np.diff` of the times, average each neighbouring pair of concentrations,
multiply, and sum. One vectorised line, no loop.

Watch the shapes line up. From 11 points you get 10 intervals, so `np.diff(t)`
has 10 entries and must be multiplied by something with 10 entries, which is
why it is `(c[:-1] + c[1:]) / 2` rather than anything involving the full array.

The units are worth stating: concentration in mg/L times time in hours gives
AUC in mg*h/L.

Fill in `cmax`, `tmax`, `auc_linear`.

Moves: `np.max`, `np.argmax`, `np.diff`, slicing with `[:-1]` and `[1:]`,
`np.sum`.
''',
        starter='''
# Part a. Peak, time of peak, and the linear trapezoidal AUC.

t = profile["TIME_H"].to_numpy()
c = profile["CONC_MG_L"].to_numpy()

cmax = None         # YOUR CODE: highest observed concentration, mg/L
tmax = None         # YOUR CODE: time at which it occurred, h
auc_linear = None   # YOUR CODE: AUC from 0 to the last sample, linear trapezoid

print("Cmax       : %s mg/L" % cmax)
print("Tmax       : %s h" % tmax)
print("AUC linear : %s mg*h/L" % auc_linear)
''',
        solution='''
# Part a. Peak, time of peak, and the linear trapezoidal AUC.

t = profile["TIME_H"].to_numpy()
c = profile["CONC_MG_L"].to_numpy()

cmax = float(np.max(c))
tmax = float(t[np.argmax(c)])
auc_linear = float(np.sum(np.diff(t) * (c[:-1] + c[1:]) / 2.0))

print("Cmax       : %s mg/L" % cmax)
print("Tmax       : %s h" % tmax)
print("AUC linear : %s mg*h/L" % auc_linear)
''',
    ),
    dict(
        label="b",
        prompt_md='''
## Part b. The area the regulatory way

Linear up, log down. Use the linear trapezoid while concentrations are rising
and the logarithmic one while they are falling.

Work interval by interval, with `c1 = c[:-1]` and `c2 = c[1:]`. An interval is
falling when `c2 < c1`. On those, the exact area under an exponential is

```
(c1 - c2) * dt / log(c1 / c2)
```

Three intervals cannot use that formula and must fall back to the linear one:
any where the concentration is not falling, any ending at zero, and any where
`c1` equals `c2` exactly, since `log(1)` is zero and you would divide by it.
Build a boolean mask of the intervals that are safe, and select with
`np.where`.

One numpy wrinkle. `np.where(cond, a, b)` evaluates **both** `a` and `b` in
full before choosing, so the log form is computed even for the intervals where
it is invalid, and numpy will warn about dividing by zero or taking the log of
zero. Those values are discarded, so wrap the computation in
`with np.errstate(divide="ignore", invalid="ignore"):` to keep the output
clean. Suppressing a warning you have understood is fine; suppressing one you
have not is how bugs hide.

Then compute `pct_diff` as `100 * (auc_linlog - auc_linear) / auc_linear`. It
should come out **negative**, and on this profile around six percent. Negative
because the log-down method is the more accurate one and the linear rule
overestimates on the way down. Six percent rather than a fraction of a percent
because the final two intervals are 12 and 24 hours long, giving the straight
line a long way to deviate from the curve. Shorten those intervals and the gap
shrinks toward nothing.

Fill in `auc_linlog`, `pct_diff`.

Moves: slicing, `np.diff`, `np.log`, `np.where`, `np.errstate`, `np.sum`.
''',
        starter='''
# Part b. Linear up, log down.

c1 = c[:-1]
c2 = c[1:]
dt = np.diff(t)

auc_linlog = None   # YOUR CODE: linear trapezoid while rising, log while falling
pct_diff = None     # YOUR CODE: 100 * (auc_linlog - auc_linear) / auc_linear

print("AUC linear        : %s" % auc_linear)
print("AUC lin-up/log-down: %s" % auc_linlog)
print("difference         : %s percent" % pct_diff)
''',
        solution='''
# Part b. Linear up, log down.

c1 = c[:-1]
c2 = c[1:]
dt = np.diff(t)

_linear_each = dt * (c1 + c2) / 2.0
with np.errstate(divide="ignore", invalid="ignore"):
    _log_each = (c1 - c2) * dt / np.log(c1 / c2)

_use_log = (c2 < c1) & (c2 > 0) & (c1 != c2)
auc_linlog = float(np.sum(np.where(_use_log, _log_each, _linear_each)))
pct_diff = float(100.0 * (auc_linlog - auc_linear) / auc_linear)

print("AUC linear        : %s" % auc_linear)
print("AUC lin-up/log-down: %s" % auc_linlog)
print("difference         : %s percent" % pct_diff)
''',
    ),
    dict(
        label="c",
        prompt_md='''
## Part c. Exposure over the first 12 hours

Partial AUC over a window, using the same linear-up log-down rule but stopping
at 12 hours.

This window is chosen to land exactly on a sampling time, which makes the
arithmetic a sum over the intervals ending at or before 12 hours rather than
an interpolation. Check that for yourself rather than assuming it: find the
position of 12.0 in the time array and sum the interval areas up to it.

Had the window ended between two samples, you would have to interpolate the
concentration at the boundary, and on a descending limb you would interpolate
logarithmically to stay consistent with how you are integrating. Mixing a
linear interpolation into a log-down integration is a small inconsistency that
is easy to introduce and hard to notice.

Then express the window as a fraction of the total: `frac_in_12h` is
`auc_0_12 / auc_linlog`. For a drug with a half-life around 7 hours sampled
out to 48, most but not all of the exposure falls in the first 12 hours, and
the remainder is the long tail that the 24 and 48 hour samples are there to
capture.

Fill in `auc_0_12`, `frac_in_12h`.

Moves: finding the index of 12.0, slicing the per interval areas,
`np.sum`, division.
''',
        starter='''
# Part c. Partial AUC over the first 12 hours.

auc_0_12 = None      # YOUR CODE: lin-up/log-down AUC from 0 to 12 h
frac_in_12h = None   # YOUR CODE: auc_0_12 / auc_linlog

print("AUC 0 to 12 h : %s mg*h/L" % auc_0_12)
print("AUC 0 to last : %s mg*h/L" % auc_linlog)
print("fraction in the first 12 h: %s" % frac_in_12h)
''',
        solution='''
# Part c. Partial AUC over the first 12 hours.

_each = np.where(_use_log, _log_each, _linear_each)
_idx_12 = int(np.where(t == 12.0)[0][0])

auc_0_12 = float(np.sum(_each[:_idx_12]))
frac_in_12h = float(auc_0_12 / auc_linlog)

print("AUC 0 to 12 h : %s mg*h/L" % auc_0_12)
print("AUC 0 to last : %s mg*h/L" % auc_linlog)
print("fraction in the first 12 h: %s" % frac_in_12h)
''',
    ),
]


def build_expected(ns):
    import numpy as np
    t = ns["profile"]["TIME_H"].to_numpy()
    c = ns["profile"]["CONC_MG_L"].to_numpy()
    dt = np.diff(t)
    c1, c2 = c[:-1], c[1:]
    lin_each = dt * (c1 + c2) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        log_each = (c1 - c2) * dt / np.log(c1 / c2)
    use_log = (c2 < c1) & (c2 > 0) & (c1 != c2)
    each = np.where(use_log, log_each, lin_each)
    idx12 = int(np.where(t == 12.0)[0][0])
    auc_lin = float(lin_each.sum())
    auc_ll = float(each.sum())
    return {
        "n_points": int(t.size),
        "cmax": float(np.max(c)),
        "tmax": float(t[np.argmax(c)]),
        "auc_linear": auc_lin,
        "auc_linlog": auc_ll,
        "pct_diff": float(100.0 * (auc_ll - auc_lin) / auc_lin),
        "pct_diff_flipped": float(100.0 * (auc_lin - auc_ll) / auc_ll),
        "auc_0_12": float(each[:idx12].sum()),
        "auc_0_12_linear": float(lin_each[:idx12].sum()),
        "frac_in_12h": float(each[:idx12].sum() / auc_ll),
        "idx12": idx12,
    }


CHECK_CODE = '''
def _part_a():
    cmax = _need("cmax")
    tmax = _need("tmax")
    auc_linear = _need("auc_linear")

    got = _as_scalar(cmax)
    if got is None:
        return "`cmax` should be a single number."
    if not _num_close(got, _EXP["cmax"], rtol=1e-9):
        return "`cmax` is not the highest observed concentration."

    got = _as_scalar(tmax)
    if got is None:
        return "`tmax` should be a single number."
    if _num_close(got, float(_np.argmax(_np.asarray([0.0])) ) ) and False:
        pass
    if not _num_close(got, _EXP["tmax"], rtol=1e-9):
        return ("`tmax` is not the time of the peak. np.argmax gives a position, which you "
                "then look up in the time array.")

    got = _as_scalar(auc_linear)
    if got is None:
        return "`auc_linear` should be a single number."
    if not _num_close(got, _EXP["auc_linear"], rtol=1e-6):
        if _num_close(got, _EXP["auc_linlog"], rtol=1e-6):
            return "`auc_linear` is the lin-up/log-down value. Part a wants plain trapezoids."
        return ("`auc_linear` is not the linear trapezoidal AUC. It is "
                "sum(diff(t) * (c[:-1] + c[1:]) / 2), and the shapes must line up: 11 "
                "points give 10 intervals.")
    return None


def _part_b():
    auc_linlog = _need("auc_linlog")
    pct_diff = _need("pct_diff")

    got = _as_scalar(auc_linlog)
    if got is None:
        return "`auc_linlog` should be a single number."
    if _num_close(got, _EXP["auc_linear"], rtol=1e-9):
        return ("`auc_linlog` equals the linear answer, so the log form was never applied. "
                "Check the falling mask: an interval is falling when c2 < c1.")
    if _np.isnan(got):
        return ("`auc_linlog` is nan. The log form is invalid on the interval starting at a "
                "concentration of zero and on any where c1 equals c2, so those intervals "
                "must fall back to the linear form.")
    if not _num_close(got, _EXP["auc_linlog"], rtol=1e-6):
        return ("`auc_linlog` does not match. Use the linear form while rising and "
                "(c1 - c2) * dt / log(c1 / c2) while falling, with a linear fallback "
                "wherever that formula is undefined.")
    if got > _EXP["auc_linear"]:
        return ("`auc_linlog` came out larger than the linear AUC, which should not happen. "
                "The log form is smaller on every falling interval.")

    got = _as_scalar(pct_diff)
    if got is None:
        return "`pct_diff` should be a single number."
    if _num_close(got, _EXP["pct_diff_flipped"], rtol=1e-4):
        return ("`pct_diff` is the other way round. It is the log-down value minus the "
                "linear one, over the linear one, so it should come out negative.")
    if not _num_close(got, _EXP["pct_diff"], rtol=1e-4):
        return "`pct_diff` is not 100 * (auc_linlog - auc_linear) / auc_linear."
    return None


def _part_c():
    auc_0_12 = _need("auc_0_12")
    frac_in_12h = _need("frac_in_12h")

    got = _as_scalar(auc_0_12)
    if got is None:
        return "`auc_0_12` should be a single number."
    if _num_close(got, _EXP["auc_linlog"], rtol=1e-9):
        return ("`auc_0_12` is the whole AUC, so the window was not applied. Sum only the "
                "intervals that end at or before 12 hours.")
    if _num_close(got, _EXP["auc_0_12_linear"], rtol=1e-6):
        return ("`auc_0_12` used plain linear trapezoids. Part c asks for the same "
                "lin-up/log-down rule as part b, applied over the window.")
    if not _num_close(got, _EXP["auc_0_12"], rtol=1e-6):
        return ("`auc_0_12` does not match. There are %d intervals before 12 hours, so slice "
                "the per interval areas with [:%d]." % (_EXP["idx12"], _EXP["idx12"]))

    got = _as_scalar(frac_in_12h)
    if got is None:
        return "`frac_in_12h` should be a single number."
    if _num_close(got, _EXP["frac_in_12h"] * 100.0, rtol=1e-6):
        return "`frac_in_12h` is a percent. The task asks for a fraction."
    if not _num_close(got, _EXP["frac_in_12h"], rtol=1e-6):
        return "`frac_in_12h` is not auc_0_12 divided by auc_linlog."
    return None
'''

# Functions this problem introduces. Keys must exist in tools/function_catalog.py.
FUNCTIONS = [
    "np.diff",
    "linear trapezoid",
    "linear up log down",
    "np.errstate",
    "partial AUC",
]
