# Week 7, PK and PD applications, part 1

The full text of every problem this week, the data it generates, the
toolkit the student is given, what each part asks for, and what every
possible check failure means. No solutions in this file.

---

# W7D1. Non-compartmental analysis: Cmax, Tmax and AUC by trapezoid

- topic: PK: linear and linear-up log-down trapezoidal AUC, partial AUC
- notebook: problems/week07/w7d1_nca_auc_trapezoid.ipynb
- solution: solutions/week07/w7d1_nca_auc_trapezoid_solution.ipynb
- expected time: 20-30 min

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

## The setup cell the student runs

This is given to them, already working. The data is synthetic and
seeded, so their numbers are always these numbers.

```python
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
```

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

The toolkit cell demonstrates each of those on a tiny made up example.
It is in the notebook and none of it is the answer.

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

**Variables this part is graded on:** `cmax`, `tmax`, `auc_linear`

The starter cell they see:

```python
# Part a. Peak, time of peak, and the linear trapezoidal AUC.

t = profile["TIME_H"].to_numpy()
c = profile["CONC_MG_L"].to_numpy()

cmax = None         # YOUR CODE: highest observed concentration, mg/L
tmax = None         # YOUR CODE: time at which it occurred, h
auc_linear = None   # YOUR CODE: AUC from 0 to the last sample, linear trapezoid

print("Cmax       : %s mg/L" % cmax)
print("Tmax       : %s h" % tmax)
print("AUC linear : %s mg*h/L" % auc_linear)
```

**Every failure message part a can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `cmax` should be a single number.
- `cmax` is not the highest observed concentration.
- `tmax` should be a single number.
- `tmax` is not the time of the peak. np.argmax gives a position, which you then look up in the time array.
- `auc_linear` should be a single number.
- `auc_linear` is not the linear trapezoidal AUC. It is sum(diff(t) * (c[:-1] + c[1:]) / 2), and the shapes must line up: 11 points give 10 intervals.
- `auc_linear` is the lin-up/log-down value. Part a wants plain trapezoids.

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

**Variables this part is graded on:** `auc_linlog`, `pct_diff`

The starter cell they see:

```python
# Part b. Linear up, log down.

c1 = c[:-1]
c2 = c[1:]
dt = np.diff(t)

auc_linlog = None   # YOUR CODE: linear trapezoid while rising, log while falling
pct_diff = None     # YOUR CODE: 100 * (auc_linlog - auc_linear) / auc_linear

print("AUC linear        : %s" % auc_linear)
print("AUC lin-up/log-down: %s" % auc_linlog)
print("difference         : %s percent" % pct_diff)
```

**Every failure message part b can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `auc_linlog` should be a single number.
- `auc_linlog` equals the linear answer, so the log form was never applied. Check the falling mask: an interval is falling when c2 < c1.
- `auc_linlog` is nan. The log form is invalid on the interval starting at a concentration of zero and on any where c1 equals c2, so those intervals must fall back to the linear form.
- `auc_linlog` does not match. Use the linear form while rising and (c1 - c2) * dt / log(c1 / c2) while falling, with a linear fallback wherever that formula is undefined.
- `auc_linlog` came out larger than the linear AUC, which should not happen. The log form is smaller on every falling interval.
- `pct_diff` should be a single number.
- `pct_diff` is the other way round. It is the log-down value minus the linear one, over the linear one, so it should come out negative.
- `pct_diff` is not 100 * (auc_linlog - auc_linear) / auc_linear.

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

**Variables this part is graded on:** `auc_0_12`, `frac_in_12h`

The starter cell they see:

```python
# Part c. Partial AUC over the first 12 hours.

auc_0_12 = None      # YOUR CODE: lin-up/log-down AUC from 0 to 12 h
frac_in_12h = None   # YOUR CODE: auc_0_12 / auc_linlog

print("AUC 0 to 12 h : %s mg*h/L" % auc_0_12)
print("AUC 0 to last : %s mg*h/L" % auc_linlog)
print("fraction in the first 12 h: %s" % frac_in_12h)
```

**Every failure message part c can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `auc_0_12` should be a single number.
- `auc_0_12` is the whole AUC, so the window was not applied. Sum only the intervals that end at or before 12 hours.
- `auc_0_12` used plain linear trapezoids. Part c asks for the same lin-up/log-down rule as part b, applied over the window.
- `auc_0_12` does not match. There are %d intervals before 12 hours, so slice the per interval areas with [:%d].
- `frac_in_12h` should be a single number.
- `frac_in_12h` is a percent. The task asks for a fraction.
- `frac_in_12h` is not auc_0_12 divided by auc_linlog.

**Functions this problem introduces:** np.diff, linear trapezoid, linear up log down, np.errstate, partial AUC

---

# W7D2. Finishing the NCA: lambda_z, AUC to infinity, clearance and MRT

- topic: PK: terminal slope selection, extrapolated AUC, CL/F, Vz/F, mean residence time
- notebook: problems/week07/w7d2_nca_lambdaz_clearance.ipynb
- solution: solutions/week07/w7d2_nca_lambdaz_clearance_solution.ipynb
- expected time: 20-30 min

## Scenario

The same subject and the same 200 mg oral profile as yesterday. You have
`AUC_last`, the exposure from time zero to the final sample. What the report
actually needs is `AUC_inf`, the exposure over all time, and the parameters
that follow from it: apparent clearance, apparent volume, and mean residence
time.

Everything beyond the last sample has to be extrapolated, and that
extrapolation rests entirely on one number: the terminal slope. Get
`lambda_z` wrong and clearance, volume, half-life and the extrapolated
fraction are all wrong together, in a correlated way that is hard to spot
because they stay mutually consistent.

Which is why the first part of this problem is not "compute the slope" but
"choose which points to compute it from", and why NCA software reports that
choice alongside the answer.

## Choosing the terminal points, then everything that follows

**The selection problem.** `lambda_z` is the slope of `ln C` against time over
the terminal phase. The difficulty is that nobody tells you where the terminal
phase starts. Take too few points and the estimate is noisy. Take too many and
you reach back into the absorption or distribution phase, which is not
log-linear, and the fitted slope comes out too shallow, which inflates the
half-life and the extrapolated area along with it.

The standard automated rule, implemented in WinNonlin and in PKNCA, is to try
every window of the last 3 points, last 4, last 5 and so on, and keep the one
with the best **adjusted** R2. Adjusted rather than plain R2 because plain R2
can be nudged upward simply by using fewer points, and the adjustment charges
for that:

```
adj_R2 = 1 - (1 - R2) * (n - 1) / (n - 2)
```

Convention also says start at or after Tmax, since nothing before the peak can
be terminal.

Report `n` alongside `lambda_z`. A half-life estimated from three points
spanning six hours deserves different trust from one estimated from six points
spanning forty, and the number itself does not tell you which you have.

**Extrapolating to infinity.** Past the last sample the profile is assumed to
decay exponentially at `lambda_z`, and the area under that tail is the last
concentration divided by the rate:

```
AUC_inf = AUC_last + C_last / lambda_z
```

The **percent extrapolated** is the share of the total that came from the
tail rather than from data. Under 20 percent is usually considered acceptable;
much above that and `AUC_inf` is mostly an assumption, and a regulator will
say so. It is the single best indicator of whether you sampled long enough.

**What follows from AUC_inf.** For an oral dose, bioavailability is unknown
and inseparable from the rest, so every parameter is apparent, written with an
`/F`:

- `CL/F = Dose / AUC_inf`, the apparent clearance,
- `Vz/F = (CL/F) / lambda_z`, the apparent volume during the terminal phase.

**Mean residence time** is the average time a molecule spends in the body. It
needs the area under the first **moment** curve, which is `t * C` rather than
`C`, integrated the same way, with its own extrapolation that has two terms
because `t * C` decays more slowly than `C`:

```
AUMC_inf = AUMC_last + t_last * C_last / lambda_z + C_last / lambda_z**2
MRT = AUMC_inf / AUC_inf
```

MRT is far more sensitive to the tail than AUC is, because the moment curve
weights late times heavily. A profile with 10 percent extrapolated AUC can
easily have 30 percent extrapolated AUMC, so treat MRT from a short profile
with real suspicion.

## Your task

Three parts, worked one at a time.

| Part | What you work out | Variables to fill in |
|---|---|---|
| a | which points are terminal, and the slope through them | `n_points_used`, `lambda_z`, `t_half`, `adj_r2` |
| b | the extrapolated area, and how much of it is assumption | `auc_inf`, `pct_extrap`, `pct_extrap_12h` |
| c | clearance, volume and mean residence time | `cl_f`, `vz_f`, `mrt` |

Below this you will find the **Setup** cell, then a **Toolkit** cell that
demonstrates every move on a tiny made up example, then **Part a**, **Part b**
and **Part c**, each with its own explanation and a cell to write in.

## The setup cell the student runs

This is given to them, already working. The data is synthetic and
seeded, so their numbers are always these numbers.

```python
import numpy as np
import pandas as pd

PROBLEM_ID = "W7D2"
TOPIC = "PK: terminal slope selection, extrapolated AUC, CL/F, Vz/F, mean residence time"

rng = np.random.default_rng(7101)      # the same subject as yesterday

DOSE_MG = 200.0
time_h = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 24.0, 48.0])

_F, _V, _ka, _kel = 0.80, 48.0, 1.15, 0.0935
_true = (_F * DOSE_MG * _ka) / (_V * (_ka - _kel)) * (
    np.exp(-_kel * time_h) - np.exp(-_ka * time_h))
conc = np.round(_true * rng.lognormal(0.0, 0.06, size=time_h.size), 4)
conc[0] = 0.0

profile = pd.DataFrame({"TIME_H": time_h, "CONC_MG_L": conc})
t = time_h
c = conc

# Yesterday's answers, so you can carry on from them.
_dt = np.diff(t)
_c1, _c2 = c[:-1], c[1:]
_lin = _dt * (_c1 + _c2) / 2.0
with np.errstate(divide="ignore", invalid="ignore"):
    _log = (_c1 - _c2) * _dt / np.log(_c1 / _c2)
_use_log = (_c2 < _c1) & (_c2 > 0) & (_c1 != _c2)
AUC_LAST = float(np.sum(np.where(_use_log, _log, _lin)))

# The first moment curve, t * C, integrated the same way.
_m = t * c
_m1, _m2 = _m[:-1], _m[1:]
_mlin = _dt * (_m1 + _m2) / 2.0
with np.errstate(divide="ignore", invalid="ignore"):
    _mlog = (_m1 - _m2) * _dt / np.log(_m1 / _m2)
_muse = (_m2 < _m1) & (_m2 > 0) & (_m1 != _m2)
AUMC_LAST = float(np.sum(np.where(_muse, _mlog, _mlin)))

TMAX = float(t[np.argmax(c)])
C_LAST = float(c[-1])
T_LAST = float(t[-1])

# The same subject, had the study stopped at 12 hours instead of 48.
_i12 = int(np.where(t == 12.0)[0][0])
AUC_TO_12H = float(np.sum(np.where(_use_log, _log, _lin)[:_i12]))
C_AT_12H = float(c[_i12])

print(profile.to_string(index=False))
print("")
print("carried over from yesterday:")
print("  AUC_last  = %.4f mg*h/L" % AUC_LAST)
print("  AUMC_last = %.4f mg*h*h/L" % AUMC_LAST)
print("  Tmax = %.1f h,  last sample %.0f h at %.4f mg/L" % (TMAX, T_LAST, C_LAST))
print("")
print("and for the comparison in part b, had sampling stopped at 12 h:")
print("  AUC_to_12h = %.4f mg*h/L,  C at 12 h = %.4f mg/L" % (AUC_TO_12H, C_AT_12H))
```

## Toolkit for this problem

| What you want to do | How |
|---|---|
| fit a line to logs | `np.polyfit(t_sub, np.log(c_sub), deg=1)` |
| R2 of that fit | `1 - ss_resid / ss_total` |
| adjusted R2 | `1 - (1 - r2) * (n - 1) / (n - 2)` |
| the last n points | `t[-n:]`, `c[-n:]` |
| try several window sizes | `for n in range(3, max_n + 1):` |
| keep the best so far | compare against a running best |
| sum of squares | `np.sum((y - yhat) ** 2)` |

The residual sum of squares is `sum((y - fitted) ** 2)` and the total is
`sum((y - y.mean()) ** 2)`, both computed on the **log** concentrations,
because that is the scale the line was fitted on.

The toolkit cell demonstrates each of those on a tiny made up example.
It is in the notebook and none of it is the answer.

## Part a. Choose the terminal points, then fit the slope

Implement the standard selection. For each window size `n` from 3 up to the
number of samples after `TMAX`, fit a straight line to `np.log(c)` against `t`
over the last `n` points, compute the adjusted R2, and keep whichever window
scores highest.

Adjusted R2 is

```
adj = 1 - (1 - r2) * (n - 1) / (n - 2)
```

where `r2 = 1 - ss_resid / ss_total`, both sums of squares computed on the
**log** concentrations, because that is the scale you fitted on.

Use plain R2 instead and you will almost always select n = 3, because a line
through three points fits them well by construction. The adjustment is what
stops the rule collapsing to the smallest window.

Restrict the search to points at or after Tmax. Nothing before the peak can
belong to the terminal phase, and including the absorption limb drags the
slope shallower, which lengthens the half-life and inflates everything
downstream.

`lambda_z` is the negative of the fitted slope, and `t_half` is `ln(2)`
divided by it.

Fill in `n_points_used`, `lambda_z`, `t_half`, `adj_r2`.

Moves: a `for` loop over window sizes, `np.polyfit(t_sub, np.log(c_sub),
deg=1)`, sums of squares, the adjustment formula.

**Variables this part is graded on:** `n_points_used`, `lambda_z`, `t_half`, `adj_r2`

The starter cell they see:

```python
# Part a. Select the terminal window by adjusted R2, then fit lambda_z.

n_points_used = None   # YOUR CODE: how many points the best window used
lambda_z = None        # YOUR CODE: terminal rate constant, 1/h
t_half = None          # YOUR CODE: terminal half-life, h
adj_r2 = None          # YOUR CODE: the adjusted R2 of the winning fit

print("points used : %s" % n_points_used)
print("lambda_z    : %s 1/h" % lambda_z)
print("half-life   : %s h" % t_half)
print("adjusted R2 : %s" % adj_r2)
```

**Every failure message part a can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `n_points_used` should be a single whole number.
- `n_points_used` is the window that wins on PLAIN R2. Select on adjusted R2, which charges for using fewer points.
- `n_points_used` is not the window that maximises adjusted R2. Search from 3 points up to the number of samples at or after Tmax.
- `lambda_z` should be a single number.
- `lambda_z` is negative. The fitted slope is negative, and lambda_z is defined as its positive counterpart.
- `lambda_z` used every sample, including the absorption phase. That drags the slope shallower. Use only the selected terminal window.
- `lambda_z` is not the negative slope of the log-linear fit over the chosen window.
- `t_half` should be a single number.
- `t_half` is 1/lambda_z. A half-life carries a factor of ln(2).
- `t_half` is not np.log(2) divided by lambda_z.
- `adj_r2` should be a single number.
- `adj_r2` is the plain R2 of the winning window. Apply the adjustment 1 - (1 - r2) * (n - 1) / (n - 2).
- `adj_r2` is not the adjusted R2 of the winning fit.

## Part b. Extrapolate to infinity, and say how much is assumption

Past the final sample the profile is assumed to keep decaying at `lambda_z`
forever, and the area under that tail is `C_last / lambda_z`.

```
AUC_inf = AUC_last + C_last / lambda_z
```

`AUC_LAST` and `C_LAST` are given in the setup cell, carried over from
yesterday.

Then compute the percent extrapolated, which is the tail as a share of the
total:

```
pct_extrap = 100 * (AUC_inf - AUC_last) / AUC_inf
```

Note the denominator. It is `AUC_inf`, not `AUC_last`, because the question is
what share of the **total** came from extrapolation.

This number decides whether anyone should believe the result. Under 20 percent
is conventionally acceptable. Above that, `AUC_inf` is substantially an
assumption about a decay you did not observe.

You should find this subject is comfortably inside that: sampling ran to 48
hours, which is roughly six and a half half-lives, so there is very little
curve left to extrapolate and the answer comes out near one percent. That is
what adequate sampling looks like, and it is worth seeing so you recognise
the opposite.

So see the opposite. Using the **same** `lambda_z`, compute what the
extrapolated fraction would have been had the study stopped at 12 hours.
`AUC_TO_12H` and `C_AT_12H` are in the setup cell. Same drug, same subject,
same terminal slope, and the only thing that changed is how long anyone
stayed to watch.

```
auc_inf_12h = AUC_TO_12H + C_AT_12H / lambda_z
pct_extrap_12h = 100 * (auc_inf_12h - AUC_TO_12H) / auc_inf_12h
```

Compare the two. The shorter study reports an exposure that is mostly an
assumption, and nothing in its output says so except this one number.

Fill in `auc_inf`, `pct_extrap`, `pct_extrap_12h`.

Moves: division, addition, the percentage formula, applied twice.

**Variables this part is graded on:** `auc_inf`, `pct_extrap`, `pct_extrap_12h`

The starter cell they see:

```python
# Part b. AUC to infinity, and the extrapolated fraction.

auc_inf = None          # YOUR CODE: AUC_LAST plus the extrapolated tail
pct_extrap = None       # YOUR CODE: 100 * (auc_inf - AUC_LAST) / auc_inf
pct_extrap_12h = None   # YOUR CODE: the same figure had sampling stopped at 12 h,
                        #            using AUC_TO_12H, C_AT_12H and the same lambda_z

print("sampled to 48 h: AUC_inf %s, extrapolated %s percent" % (auc_inf, pct_extrap))
print("sampled to 12 h: extrapolated %s percent" % pct_extrap_12h)
```

**Every failure message part b can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `auc_inf` should be a single number.
- `auc_inf` equals AUC_LAST, so no tail was added.
- `auc_inf` came out smaller than AUC_LAST, which cannot happen: the extrapolated tail only adds area.
- `auc_inf` is not AUC_LAST plus C_LAST divided by lambda_z.
- `pct_extrap` should be a single number.
- `pct_extrap` divided by AUC_LAST. The denominator is auc_inf, because the question is what share of the TOTAL came from extrapolation.
- `pct_extrap` is not 100 * (auc_inf - AUC_LAST) / auc_inf.
- `pct_extrap_12h` should be a single number.
- `pct_extrap_12h` equals the 48 hour answer, so the shorter study was not actually used. Build it from AUC_TO_12H and C_AT_12H.
- `pct_extrap_12h` is not the extrapolated share using AUC_TO_12H, C_AT_12H and the same lambda_z.

## Part c. Clearance, volume, and mean residence time

This was an oral dose, so bioavailability is unknown and cannot be separated
from clearance or volume. Every parameter is therefore apparent, and the
convention is to write it with an `/F`:

```
CL/F = Dose / AUC_inf
Vz/F = (CL/F) / lambda_z
```

Then mean residence time, the average time a molecule stays in the body. It
needs the first moment curve `t * C`, whose area to the last sample,
`AUMC_LAST`, is given in the setup. Its extrapolation has **two** terms,
because `t * C` decays more slowly than `C` does:

```
AUMC_inf = AUMC_LAST + T_LAST * C_LAST / lambda_z + C_LAST / lambda_z**2
MRT = AUMC_inf / AUC_inf
```

Forgetting the second term is a common slip and it always makes MRT look
shorter than it is.

Sanity checks worth running. `CL/F` for an oral drug of this exposure should
land in single digits of L/h. `Vz/F` should be tens of litres, a plausible
apparent volume for a body. And `MRT` should be longer than the half-life,
which it is for a one compartment drug given orally, because absorption adds
residence time on top of elimination.

Fill in `cl_f`, `vz_f`, `mrt`.

Moves: division, the AUMC extrapolation formula with both terms.

**Variables this part is graded on:** `cl_f`, `vz_f`, `mrt`

The starter cell they see:

```python
# Part c. Apparent clearance, apparent volume, and mean residence time.

cl_f = None   # YOUR CODE: Dose / AUC_inf, in L/h
vz_f = None   # YOUR CODE: (CL/F) / lambda_z, in L
mrt = None    # YOUR CODE: AUMC_inf / AUC_inf, in h

print("CL/F : %s L/h" % cl_f)
print("Vz/F : %s L" % vz_f)
print("MRT  : %s h   (half-life was %s h)" % (mrt, t_half))
```

**Every failure message part c can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `cl_f` should be a single number.
- `cl_f` used AUC_LAST. Clearance is the dose over the area to INFINITY, otherwise it is biased upward by however much you failed to sample.
- `cl_f` is not DOSE_MG divided by auc_inf.
- `vz_f` should be a single number.
- `vz_f` multiplied by lambda_z instead of dividing. Volume is clearance divided by the rate constant.
- `vz_f` is not cl_f divided by lambda_z.
- `mrt` should be a single number.
- `mrt` used only one extrapolation term for the AUMC. There are two, because t * C decays more slowly than C: add C_LAST / lambda_z ** 2 as well. Leaving it out always makes MRT too short.
- `mrt` is not the extrapolated AUMC divided by auc_inf.
- `mrt` came out shorter than the half-life, which should not happen for an orally dosed one compartment drug, since absorption adds residence time.

**Functions this problem introduces:** lambda_z selection, adjusted R2, AUC extrapolation, mean residence time

---

# W7D3. Fitting a one compartment oral model, and checking it against NCA

- topic: PK modelling: depot and central compartments, fitting an ODE model, model versus NCA
- notebook: problems/week07/w7d3_pk_fit_one_compartment.ipynb
- solution: solutions/week07/w7d3_pk_fit_one_compartment_solution.ipynb
- expected time: 20-30 min

## Scenario

The same 200 mg oral profile you ran NCA on. NCA told you what happened:
exposure, peak, half-life, apparent clearance. It cannot tell you what would
happen at a different dose, on a different schedule, or in a patient with half
the renal function.

For that you need a model with parameters that mean something physically. The
simplest one that fits an oral profile has two compartments in the plumbing
sense: a **depot** holding drug not yet absorbed, and a **central** compartment
representing the body. Drug leaves the depot and enters the central at a rate
`ka`, and leaves the central at `CL/V`.

Three parameters: `ka`, `V/F` and `CL/F`. Fit them to the observed
concentrations, then check the result against the NCA numbers. The two
approaches share no arithmetic at all, so agreement between them is real
evidence that both are right, and disagreement tells you which assumption
broke.

## Fitting a differential equation to data

**The model.** Two states, written as amounts in milligrams:

```
dA_depot/dt   = -ka * A_depot
dA_central/dt = +ka * A_depot - (CL / V) * A_central
```

The whole dose starts in the depot and the central compartment starts empty.
Concentration is `A_central / V`, which is the only place `V` enters the
observation, and it is why `V` and `F` cannot be separated from an oral dose
alone.

**Fitting an ODE is just curve_fit with a slower model function.** The shape is
identical to week 3: write a function taking time first and parameters after,
and hand it to `curve_fit`. The only difference is that evaluating it runs a
numerical integration instead of a formula. Something like:

```python
def predict(t, ka, v, cl):
    sol = solve_ivp(rhs, (0, t.max()), [dose, 0.0], t_eval=t,
                    args=(ka, v, cl), rtol=1e-8, atol=1e-10)
    return sol.y[1] / v
```

Two practical points. The solver is called once per optimiser iteration, which
is hundreds of times, so tight tolerances cost real time; they are still worth
it, because a loose solve makes the objective surface bumpy and the optimiser
chases numerical noise. And `t_eval` must start at or after the start of
`t_span`, so if your first observation is at time zero, integrate from zero.

**Starting values and bounds matter more here than for an algebraic model.**
Read `p0` off the data: `ka` from how fast the rise is, roughly `3 / Tmax`;
`CL/F` from the NCA answer, since dose over AUC is a decent estimate whatever
model you end up with; `V/F` from `CL/F` divided by an eyeballed terminal rate
constant. All three parameters are strictly positive, so bound them below at
zero.

**The flip-flop ambiguity.** For an oral profile, `ka` and `kel` can sometimes
be swapped and give an almost identical curve. The fit may converge on either,
and the usual tell is `ka` coming out smaller than `CL/V`, which means
absorption is rate limiting and the terminal slope you measured is actually
absorption rather than elimination. Check for it rather than assuming it away.

**Why compare with NCA.** NCA assumes almost nothing but cannot extrapolate.
The model can extrapolate but assumes a particular structure. They are
computed by completely different routes, so if `CL/F` agrees between them, the
model structure is probably adequate. A model `CL/F` well away from the NCA
value usually means the structure is wrong, not that the fit failed.

## Your task

Three parts, worked one at a time.

| Part | What you work out | Variables to fill in |
|---|---|---|
| a | the two compartment ODE, solved for given parameters | `rhs`, `predict` |
| b | the three parameters fitted to the data | `popt`, `ka_hat`, `v_hat`, `cl_hat`, `rmse` |
| c | the model's answers against NCA's | `t_half_model`, `cl_ratio`, `flip_flop` |

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
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

PROBLEM_ID = "W7D3"
TOPIC = "PK modelling: depot and central compartments, fitting an ODE model, model versus NCA"

rng = np.random.default_rng(7101)      # the same subject as days 1 and 2

DOSE_MG = 200.0
time_h = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 24.0, 48.0])

_F, _V, _ka, _kel = 0.80, 48.0, 1.15, 0.0935
_true = (_F * DOSE_MG * _ka) / (_V * (_ka - _kel)) * (
    np.exp(-_kel * time_h) - np.exp(-_ka * time_h))
conc = np.round(_true * rng.lognormal(0.0, 0.06, size=time_h.size), 4)
conc[0] = 0.0

t = time_h
c = conc
profile = pd.DataFrame({"TIME_H": t, "CONC_MG_L": c})

# What NCA said, carried over from day 2.
NCA_CL_F = 5.5497
NCA_LAMBDA_Z = 0.09563
NCA_T_HALF = 7.2480
TMAX = float(t[np.argmax(c)])

print(profile.to_string(index=False))
print("")
print("NCA, from day 2:")
print("  CL/F     = %.4f L/h" % NCA_CL_F)
print("  lambda_z = %.5f 1/h   (half-life %.3f h)" % (NCA_LAMBDA_Z, NCA_T_HALF))
print("  Tmax     = %.1f h" % TMAX)
```

## Toolkit for this problem

| What you want to do | How |
|---|---|
| two states in one ODE | `return [d_depot, d_central]`, and `y[0]`, `y[1]` |
| solve for given parameters | `solve_ivp(rhs, (0, t[-1]), [dose, 0.0], t_eval=t, args=(...))` |
| tight tolerances | `rtol=1e-8, atol=1e-10` |
| the central compartment | `sol.y[1]` |
| a model function for fitting | `def predict(t, ka, v, cl): ... return conc` |
| fit it | `curve_fit(predict, t, c, p0=[...], bounds=([0,0,0], [inf]*3))` |
| an upper bound of none | `np.inf` |

The ODE function signature is `rhs(t, y, *params)` and the model function for
`curve_fit` is `predict(t, *params)`. They take their arguments in different
orders, which is worth being careful about.

The toolkit cell demonstrates each of those on a tiny made up example.
It is in the notebook and none of it is the answer.

## Part a. Write the system, and a function that predicts concentrations

Two functions.

`rhs(t, y, ka, v, cl)` returns the two derivatives. `y[0]` is the amount in
the depot and `y[1]` the amount in the central compartment. Drug leaves the
depot at `ka * y[0]` and that same amount arrives in the central, which then
loses `(cl / v) * y[1]`. Return them as a list in that order.

Note the sign structure: the absorption term appears **twice**, negative in
the depot and positive in the central. That is mass balance, and getting one
of the two signs wrong is the most common error in writing a compartment
model. The amounts should never sum to more than the dose.

`predict(t_eval, ka, v, cl)` solves the system from 0 to the last requested
time, starting with the whole dose in the depot and nothing in the central,
and returns the **concentration**, which is the central amount divided by `v`.
Pass `rtol=1e-8, atol=1e-10`.

Note the argument order. `rhs` takes `(t, y, ...)` because that is what
`solve_ivp` requires, and `predict` takes `(t, ...)` with no state, because
that is what `curve_fit` requires.

Fill in `rhs` and `predict`.

Moves: `def rhs(t, y, ka, v, cl)`, `def predict(t, ka, v, cl)`,
`solve_ivp(..., t_eval=t, args=(...), rtol=1e-8, atol=1e-10)`, `sol.y[1] / v`.

**Variables this part is graded on:** see the starter

The starter cell they see:

```python
# Part a. The depot and central system, and a prediction function.

def rhs(t, y, ka, v, cl):
    """y[0] is the depot amount, y[1] the central amount, both in mg."""
    return None   # YOUR CODE: [-ka*y[0], ka*y[0] - (cl/v)*y[1]]


def predict(t_eval, ka, v, cl):
    """Concentration in mg/L at each time in t_eval."""
    return None   # YOUR CODE: solve from 0 with the dose in the depot, return y[1]/v


print("check at the observed times, with roughly sensible parameters:")
print(None if predict(t, 1.0, 50.0, 5.5) is None
      else np.round(predict(t, 1.0, 50.0, 5.5), 4))
print("observed:", c)
```

**Every failure message part a can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `rhs` should be a function.
- `rhs` should return TWO derivatives, one per compartment.
- `rhs` does not match [-ka*y[0], ka*y[0] - (cl/v)*y[1]]. With y = [100, 50], ka 1, v 50, cl 5 it should give [-100, 95].
- `predict` should be a function.
- `predict` should return one concentration per requested time, so shape (%d,) here.
- `predict` returned an AMOUNT in mg. Divide the central compartment by v to get a concentration.
- `predict` does not match. Start with the whole dose in the depot and nothing in the central, return sol.y[1] divided by v, and pass rtol=1e-8, atol=1e-10.
- `rhs` raised %s when called as rhs(t, y, ka, v, cl).
- `rhs` has the absorption sign wrong in the central compartment. Drug LEAVES the depot and ARRIVES in the central, so the same term appears negative in one and positive in the other.
- `rhs` has both absorption signs flipped. The depot can only lose drug.
- `predict` raised %s when called as predict(t, ka, v, cl).

## Part b. Fit the three parameters

Hand `predict` to `curve_fit` along with the observed times and
concentrations.

Read the starting values off what you already know rather than guessing.
`ka` is roughly `3 / Tmax`, because the peak of an oral profile sits a few
absorption half-lives in. `CL/F` is close to the NCA value, since dose over
AUC estimates clearance regardless of structure. `V/F` follows as `CL/F`
divided by the NCA terminal rate constant, because `Vz = CL / lambda_z`.

All three are physically positive, so bound them below at zero and leave the
upper bounds at `np.inf`.

Then pull out the three fitted values and compute the RMSE of the fit against
the observed concentrations.

Compare the fitted parameters against the truth used to generate the data,
which was `ka = 1.15`, `V/F = 60` and `CL/F = 5.61`. Note that `V/F` and
`CL/F` are the apparent values, so `V/F` is the real 48 L divided by the 0.8
bioavailability, and `CL/F` likewise. An oral profile simply cannot tell you
`F`, and no amount of fitting will change that.

Fill in `popt`, `ka_hat`, `v_hat`, `cl_hat`, `rmse`.

Moves: `curve_fit(predict, t, c, p0=[...], bounds=([0,0,0], [np.inf]*3))`,
indexing, `np.sqrt(np.mean(resid ** 2))`.

**Variables this part is graded on:** `popt`, `ka_hat`, `v_hat`, `cl_hat`, `rmse`

The starter cell they see:

```python
# Part b. Fit ka, V/F and CL/F.

popt = None     # YOUR CODE: the three fitted parameters, in order ka, v, cl
ka_hat = None   # YOUR CODE
v_hat = None    # YOUR CODE
cl_hat = None   # YOUR CODE
rmse = None     # YOUR CODE: RMSE of predict(t, *popt) against c

print("ka   = %s 1/h   (true 1.15)" % ka_hat)
print("V/F  = %s L     (true 60.0)" % v_hat)
print("CL/F = %s L/h   (true 5.61)" % cl_hat)
print("rmse = %s mg/L" % rmse)
```

**Every failure message part b can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `popt` should hold the three fitted parameters.
- `popt` does not match the fit. Check the p0 suggested in the prompt and that all three parameters are bounded below at zero.
- `rmse` should be a single number.
- `rmse` is the mean squared error. Take its square root.
- `rmse` is not sqrt(mean((c - predict(t, *popt)) ** 2)).
- `%s` should be a single number.
- `%s` is not popt[%d].

## Part c. Check the model against NCA

Three checks, each answering a different question.

**Half-life.** The model's elimination rate constant is `cl_hat / v_hat`, and
the half-life follows as `ln(2)` over it. Compare with the NCA half-life in
the setup. They are computed by completely different routes, NCA from a
regression through the last few points and the model from a fit to the whole
curve, so agreement is real evidence rather than a tautology.

**Clearance.** `cl_ratio` is `cl_hat / NCA_CL_F`. It should sit very close to
1. A ratio noticeably away from 1 is the signal that the model structure is
inadequate, not that the optimiser struggled. A one compartment model fitted
to genuinely two compartment data typically gets clearance roughly right but
the early points badly wrong, which shows up in the residuals before it shows
up here.

**Flip-flop.** Set `flip_flop` to `True` if `ka_hat` is **less than**
`cl_hat / v_hat`, meaning absorption is slower than elimination. When that
happens the terminal slope you measured in NCA is really the absorption rate,
and the half-life you reported is the absorption half-life wearing
elimination's clothes. For this drug, absorption is fast and the answer should
be `False`, but it is worth writing the test down, because the condition is
easy to check and almost never checked.

Fill in `t_half_model`, `cl_ratio`, `flip_flop`.

Moves: division, `np.log(2)`, a comparison.

**Variables this part is graded on:** `t_half_model`, `cl_ratio`, `flip_flop`

The starter cell they see:

```python
# Part c. Compare the model with NCA.

t_half_model = None   # YOUR CODE: ln(2) / (cl_hat / v_hat), in hours
cl_ratio = None       # YOUR CODE: cl_hat / NCA_CL_F
flip_flop = None      # YOUR CODE: True if ka_hat < cl_hat / v_hat

print("half-life, model : %s h" % t_half_model)
print("half-life, NCA   : %.4f h" % NCA_T_HALF)
print("CL ratio model/NCA: %s" % cl_ratio)
print("flip-flop?        : %s" % flip_flop)
```

**Every failure message part c can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `t_half_model` should be a single number.
- `t_half_model` is 1/kel. A half-life carries a factor of ln(2).
- `t_half_model` used ka rather than the elimination rate constant. The rate constant out of the central compartment is cl_hat / v_hat.
- `t_half_model` is not np.log(2) divided by cl_hat / v_hat.
- `cl_ratio` should be a single number.
- `cl_ratio` is the other way up. It is the model value over the NCA value.
- `cl_ratio` is not cl_hat divided by NCA_CL_F.
- `flip_flop` should be %s here. The test is whether ka_hat is LESS than the elimination rate constant cl_hat / v_hat, which would mean absorption is the slower process.
- `flip_flop` should be a single True or False.

**Functions this problem introduces:** depot and central, fitting an ODE model, flip-flop kinetics

---

# W7D4. Two compartments, and what a one compartment model gets wrong

- topic: PK modelling: distribution phase, alpha and beta, choosing model structure
- notebook: problems/week07/w7d4_pk_two_compartment.ipynb
- solution: solutions/week07/w7d4_pk_two_compartment_solution.ipynb
- expected time: 20-30 min

## Scenario

A different compound, given as a 100 mg intravenous bolus, sampled densely
early: 5 minutes, 15 minutes, half an hour, and on out to 48 hours.

Plot it on a log scale and it does not come out as one straight line. It comes
out as two, a steep one for the first couple of hours and a shallow one after
that. That is the signature of a drug that distributes into tissue before it
is eliminated, and no one compartment model can reproduce it, because a single
exponential is a single straight line on a log axis by definition.

The fix is a second compartment representing tissue, with drug moving back and
forth. Four parameters instead of two. You will fit both structures, see
exactly where the simpler one fails, and quantify the clinical consequence,
which is a badly wrong estimate of the concentration right after dosing.

## Distribution, and the two straight lines

**The model.** Central and peripheral compartments, with drug exchanging
between them and leaving only from the central:

```
dA1/dt = -(CL/V1) * A1 - (Q/V1) * A1 + (Q/V2) * A2
dA2/dt =               + (Q/V1) * A1 - (Q/V2) * A2
```

`CL` is clearance, `V1` the central volume, `Q` the intercompartmental
clearance governing how fast drug moves to tissue, and `V2` the peripheral
volume. Concentration is measured in the central compartment, `A1 / V1`.

**Why the profile bends.** Straight after an IV bolus two things remove drug
from plasma at once: elimination, and distribution into tissue that is still
nearly empty. Together they are fast, which is the steep **alpha** phase. Once
tissue and plasma approach equilibrium, the distribution term nets out to
nearly nothing and only elimination is left, which is the shallow **beta**
phase. On a log axis each phase is a straight line, and the whole profile is
their sum.

**What this means for sampling.** You can only see the alpha phase if you
sample during it. Miss the first hour and a two compartment drug looks exactly
like a one compartment drug with a smaller volume, which is why early samples
are worth disproportionately more than late ones. The profile here has samples
at 5 and 15 minutes for precisely that reason.

**Where the one compartment model fails.** Forced to push a single exponential
through a curve that bends, it has to pick a phase to follow. Which one it
picks depends on where your samples are, and this profile is sampled densely
early, so it follows the steep alpha phase and effectively ignores the tail.

The consequence is severe and not where people expect it. The early points
come out roughly right, within ten percent or so. From about three hours
onward the fit collapses: it predicts near zero while the observed
concentration is still appreciable, and by twelve hours it is wrong by
essentially a hundred percent.

Because clearance is dose over AUC, and this fit has thrown away the entire
tail where most of the AUC lives, **clearance comes out more than twice the
truth and exposure less than half of it**. A model that underestimates total
exposure by more than half is not conservatively wrong, it is dangerously
wrong, and the parameters it reports look perfectly reasonable in isolation.

It also reports a `V` that is neither the central volume nor the volume at
steady state, but an artefact of which phase it happened to chase.

**Volumes in a two compartment model.** `V1` is what the dose initially
distributes into, so it sets the peak. `Vss = V1 + V2` is the volume at steady
state, which is the one to compare against a one compartment `V`. Reporting
`V1` as though it were the volume of distribution is a common and consequential
slip.

**On choosing structure.** More parameters always fit better, as week 3
established, so a lower RMSE alone does not justify the second compartment.
The honest evidence is the residuals: a one compartment fit to two
compartment data leaves residuals with obvious structure, all the same sign
early, which no amount of noise would produce. Structure in the residuals is
the model telling you what it cannot express.

## Your task

Three parts, worked one at a time.

| Part | What you work out | Variables to fill in |
|---|---|---|
| a | the two compartment system, fitted | `rhs2`, `predict2`, `popt2` |
| b | the one compartment fit, for comparison | `popt1`, `rmse1`, `rmse2` |
| c | where the simple model fails, and by how much | `vss`, `late_bias`, `auc_one`, `auc_two`, `auc_bias` |

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
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

PROBLEM_ID = "W7D4"
TOPIC = "PK modelling: distribution phase, alpha and beta, choosing model structure"

rng = np.random.default_rng(7104)

DOSE_MG = 100.0
t = np.array([0.0833, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0,
              6.0, 8.0, 12.0, 24.0, 36.0, 48.0])

# Truth: CL 3.2 L/h, V1 9.0 L, Q 6.5 L/h, V2 26.0 L.
_CL, _V1, _Q, _V2 = 3.2, 9.0, 6.5, 26.0


def _true_rhs(tt, y):
    a1, a2 = y
    return [-(_CL / _V1) * a1 - (_Q / _V1) * a1 + (_Q / _V2) * a2,
            (_Q / _V1) * a1 - (_Q / _V2) * a2]


_sol = solve_ivp(_true_rhs, (0.0, float(t.max())), [DOSE_MG, 0.0],
                 t_eval=t, rtol=1e-10, atol=1e-12)
_true_conc = _sol.y[0] / _V1
c = np.round(_true_conc * rng.lognormal(0.0, 0.05, size=t.size), 5)

profile = pd.DataFrame({"TIME_H": t, "CONC_MG_L": c})

print(profile.to_string(index=False))
print("")
print("true C0 if the dose went instantly into the central compartment:")
print("  dose / V1 = %.4f mg/L" % (DOSE_MG / _V1))

_fig, _axes = plt.subplots(1, 2, figsize=(10, 3.8))
for _ax, _scale in zip(_axes, ["linear", "log"]):
    _ax.plot(t, c, marker="o")
    _ax.set_yscale(_scale)
    _ax.set_xlabel("Time (h)"); _ax.set_ylabel("Concentration (mg/L)")
    _ax.set_title(_scale)
_fig.tight_layout()
plt.show()
print("on the log panel, look for two straight segments rather than one")
```

## Toolkit for this problem

| What you want to do | How |
|---|---|
| two exchanging compartments | see the equations in the concept section |
| solve from an IV bolus | `y0 = [dose, 0.0]`, all of it in the central |
| the measured compartment | `sol.y[0] / v1` |
| fit four parameters | `curve_fit(predict2, t, c, p0=[...], bounds=([0]*4, [np.inf]*4))` |
| residuals | `c - predict(t, *popt)` |
| signs of residuals | `np.sign(resid)` |
| evaluate at one time | `predict(np.array([0.0833]), *popt)[0]` |

With four parameters the fit is more sensitive to `p0` than with two. Reading
starting values off the profile, a steep early slope and a shallow late one,
is what makes it converge.

The toolkit cell demonstrates each of those on a tiny made up example.
It is in the notebook and none of it is the answer.

## Part a. Build and fit the two compartment model

Write `rhs2(t, y, cl, v1, q, v2)` with `y[0]` the central amount and `y[1]`
the peripheral amount. Two transfer terms move drug between them, and only the
central loses drug to elimination:

```
to_tissue   = (q / v1) * y[0]
from_tissue = (q / v2) * y[1]
dA1/dt = -(cl / v1) * y[0] - to_tissue + from_tissue
dA2/dt =                   + to_tissue - from_tissue
```

Check the structure before moving on: the two transfer terms appear in both
equations with opposite signs, because drug leaving one arrives in the other.
Only `-(cl/v1) * y[0]` actually removes drug from the system.

Then `predict2(t_eval, cl, v1, q, v2)` solving from an IV bolus, so the whole
dose starts in the central compartment and the peripheral starts empty, and
returning `sol.y[0] / v1`. Use `rtol=1e-10, atol=1e-12`, tighter than usual
because the alpha phase changes fast and a loose solve will smear it.

Fit all four with `p0=[3.0, 10.0, 6.0, 25.0]` and all four bounded below at
zero. Those starting values are read off the profile: the first concentration
is roughly `dose / V1`, giving `V1` near 10, and the late slope gives clearance
of a few litres per hour.

Fill in `rhs2`, `predict2`, `popt2`.

Moves: `def rhs2(t, y, cl, v1, q, v2)`, `solve_ivp(..., [DOSE_MG, 0.0], ...)`,
`curve_fit(predict2, t, c, p0=[...], bounds=([0]*4, [np.inf]*4))`.

**Variables this part is graded on:** `popt2`

The starter cell they see:

```python
# Part a. The two compartment system, fitted.

def rhs2(t, y, cl, v1, q, v2):
    """y[0] central amount, y[1] peripheral amount, both mg."""
    return None   # YOUR CODE


def predict2(t_eval, cl, v1, q, v2):
    """Central concentration in mg/L at each time."""
    return None   # YOUR CODE


popt2 = None   # YOUR CODE: the four fitted parameters, order cl, v1, q, v2

print("fitted CL %s, V1 %s, Q %s, V2 %s" % tuple(popt2) if popt2 is not None else "not fitted")
print("truth   CL 3.2, V1 9.0, Q 6.5, V2 26.0")
```

**Every failure message part a can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `rhs2` should be a function.
- `rhs2` should return TWO derivatives.
- `rhs2` does not match the equations in the prompt. With y = [100, 20], cl 3, v1 10, q 6, v2 25 it should give [-85.2, 55.2].
- `predict2` should be a function.
- `predict2` should return one concentration per requested time.
- `predict2` does not match. Start the whole dose in the CENTRAL compartment for an IV bolus, return sol.y[0] divided by v1, and use rtol=1e-10, atol=1e-12.
- `popt2` should hold the four fitted parameters.
- `popt2` does not match the fit. Use p0=[3.0, 10.0, 6.0, 25.0] with all four bounded below at zero.
- `rhs2` raised %s when called as rhs2(t, y, cl, v1, q, v2).
- `rhs2` does not conserve mass. The two transfer terms must appear in BOTH equations with opposite signs, so the derivatives sum to minus the elimination term alone.
- `predict2` raised %s.

## Part b. Fit one compartment too, and compare

A one compartment IV bolus is `C(t) = (dose / v) * exp(-(cl / v) * t)`, which
needs no solver at all. Write it as an ordinary function of `t`, `cl` and `v`,
fit it with `p0=[3.0, 25.0]` and both parameters bounded below at zero, and
compute the RMSE of each model against the observed concentrations.

`rmse2` will be lower. That is guaranteed and proves nothing on its own, as
week 3 covered: four parameters can always do at least as well as two.

What is informative is **how much** lower, and where the difference lives.
Compare the two RMSEs and then look at where the one compartment residuals
sit. They will not be scattered around zero. The early ones will all have the
same sign, because a single exponential forced through a bending curve has to
undershoot at one end.

That pattern is the real evidence, and it is a different kind of evidence from
a smaller RMSE. Noise produces residuals with random signs. Structure in the
signs means the model cannot express something the data contains.

Fill in `popt1`, `rmse1`, `rmse2`.

Moves: `def one_comp(t, cl, v)`, `curve_fit`, `np.sqrt(np.mean(resid ** 2))`.

**Variables this part is graded on:** `popt1`, `rmse1`, `rmse2`

The starter cell they see:

```python
# Part b. A one compartment fit, for comparison.

def one_comp(t_eval, cl, v):
    """IV bolus into a single compartment, in closed form."""
    return None   # YOUR CODE: (DOSE_MG / v) * exp(-(cl / v) * t_eval)


popt1 = None   # YOUR CODE: the two fitted parameters, order cl, v
rmse1 = None   # YOUR CODE: RMSE of the one compartment fit
rmse2 = None   # YOUR CODE: RMSE of the two compartment fit

print("one compartment: CL %s, V %s, rmse %s" % (
    popt1[0] if popt1 is not None else None,
    popt1[1] if popt1 is not None else None, rmse1))
print("two compartment: rmse %s" % rmse2)
if popt1 is not None:
    print("")
    print("one compartment residuals (observed minus fitted):")
    print(np.round(c - one_comp(t, *popt1), 4))
```

**Every failure message part b can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `one_comp` should be a function.
- `one_comp` at t = 0 should be dose / v, which is %.4f here, but it gave %.4f.
- `popt1` should hold the two fitted parameters.
- `popt1` does not match the one compartment fit with p0=[3.0, 25.0].
- `rmse1` should be a single number.
- `rmse1` is the mean squared error. Take its square root.
- `rmse1` is not the RMSE of the one compartment fit.
- `rmse2` should be a single number.
- `rmse2` is not the RMSE of the two compartment fit.
- `rmse2` came out worse than `rmse1`, which should not happen. The richer model can always reproduce the simpler one, so a worse fit means it did not converge.

## Part c. The clinical consequence

Put numbers on what the simpler model costs you, and look where the damage
actually is rather than where you would guess.

**Volume.** `vss` is `V1 + V2` from the two compartment fit, the volume at
steady state. Compare it with the one compartment `V` from part b. They are
not describing the same thing and should not match, and the gap is large.

**Where the fit breaks.** `late_bias` is the percentage error of the one
compartment prediction at **12 hours**, against the observed concentration
there:

```
late_bias = 100 * (pred_one_at_12h - observed_at_12h) / observed_at_12h
```

Index 12 hours out of `t` and `c` rather than hard coding a position. Expect
something close to minus one hundred percent: the fitted exponential has
essentially reached zero while there is still real drug in the plasma. Check
the early points too, printed for you below, and notice that they are fine.
The failure is entirely in the tail.

**Exposure.** For an IV bolus, AUC to infinity is simply `dose / CL`, so each
model implies a total exposure:

```
auc_one = DOSE_MG / popt1[0]
auc_two = DOSE_MG / popt2[0]
auc_bias = 100 * (auc_one - auc_two) / auc_two
```

This is the number that matters. The one compartment model discarded the
tail, which is where most of the area lives, so its clearance comes out more
than twice as high and its exposure less than half. Underestimating total
exposure by that much would understate every safety margin you derive from it.

Fill in `vss`, `late_bias`, `auc_one`, `auc_two`, `auc_bias`.

Moves: addition, finding the index of 12.0 in `t`, calling each model,
the percentage formula.

**Variables this part is graded on:** `vss`, `late_bias`, `auc_one`, `auc_two`, `auc_bias`

The starter cell they see:

```python
# Part c. What the one compartment model costs you early.

vss = None        # YOUR CODE: V1 + V2 from the two compartment fit
late_bias = None  # YOUR CODE: percent error of the one compartment fit at 12 h
auc_one = None    # YOUR CODE: DOSE_MG / one compartment clearance
auc_two = None    # YOUR CODE: DOSE_MG / two compartment clearance
auc_bias = None   # YOUR CODE: 100 * (auc_one - auc_two) / auc_two

print("Vss (two compartment) : %s L" % vss)
print("V   (one compartment) : %s L" % (popt1[1] if popt1 is not None else None))
print("")
print("error at 12 h, one compartment: %s percent" % late_bias)
print("")
print("implied AUC: one compartment %s, two compartment %s mg*h/L" % (auc_one, auc_two))
print("exposure understated by       : %s percent" % auc_bias)
if popt1 is not None:
    print("")
    print("for contrast, the early points are fine:")
    for _i in range(4):
        _p = one_comp(t[_i:_i + 1], *popt1)[0]
        print("   t=%6.3f h  observed %7.4f  one cmt %7.4f  (%+.1f percent)"
              % (t[_i], c[_i], _p, 100 * (_p - c[_i]) / c[_i]))
```

**Every failure message part c can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `vss` should be a single number.
- `vss` is V1 alone, the central volume. The volume at steady state is V1 + V2, and reporting V1 as the volume of distribution understates it badly.
- `vss` is not popt2[1] plus popt2[3].
- `late_bias` should be a single number.
- `late_bias` has the sign the wrong way round. It is the prediction minus the observation, over the observation, so a model reading near zero against real drug gives close to minus one hundred.
- `late_bias` is not the percent error of the one compartment fit at 12 hours. Find the index where t equals 12.0 and compare there.
- `auc_one` should be a single number.
- `auc_one` is not DOSE_MG divided by the one compartment clearance, popt1[0].
- `auc_two` should be a single number.
- `auc_two` is not DOSE_MG divided by the two compartment clearance, popt2[0].
- `auc_bias` should be a single number.
- `auc_bias` has the sign the wrong way round. It is the one compartment exposure minus the two compartment one, over the two compartment one, so a model that discards the tail gives a negative number.
- `auc_bias` is not 100 * (auc_one - auc_two) / auc_two.

**Functions this problem introduces:** two compartment model, alpha and beta phases, Vss, residual structure

---

# W7D5. Multiple dosing: accumulation, steady state, and how long it takes

- topic: PK: repeated dosing by simulation, accumulation ratio, time to steady state
- notebook: problems/week07/w7d5_pk_multiple_dosing.ipynb
- solution: solutions/week07/w7d5_pk_multiple_dosing_solution.ipynb
- expected time: 20-30 min

## Scenario

The oral compound from day 3, with its fitted parameters, is going into a
repeat dose study: 200 mg every 12 hours for 10 days.

Three questions have to be answered before the first subject is dosed, and all
three are decided by the parameters you already have.

How much higher will concentrations climb than after a single dose. How long
until they stop climbing. And what the average exposure will be once they do,
because that is what gets compared against the exposure that was safe in the
toxicology species.

All three have exact algebraic answers for a one compartment model, and all
three can be read off a simulation. Doing both and checking they agree is the
point: the algebra is quick but fragile to assumptions, and the simulation
survives into situations where no algebra exists.

## Accumulation, and why steady state takes as long as it does

**Why concentrations accumulate.** If a dose arrives before the previous one
has gone, drug left over adds to drug arriving. Each dose adds the same
amount, and each interval removes a fixed **fraction** of what is present. The
level climbs until the amount removed per interval equals the amount added,
and then stops. That balance point is steady state.

**The accumulation ratio** follows from summing a geometric series:

```
R = 1 / (1 - exp(-kel * tau))
```

where `tau` is the dosing interval. It depends only on the half-life and the
interval, through their ratio. Dose one half-life apart and you accumulate
two-fold. Dose far apart relative to the half-life and `R` approaches 1,
meaning no accumulation at all. Note what is absent: the dose itself does not
appear. Doubling the dose doubles every concentration but changes the
accumulation ratio not at all.

The derivation assumes the dose appears instantly, so it is exact for an
intravenous bolus and approximate for anything absorbed. For an oral drug it
holds well at the trough, where absorption is finished, and runs a few percent
low at the peak, where it is not. You will measure that gap in part b.

**Time to steady state depends only on the half-life.** This surprises people
every time. It takes about 5 half-lives to get within a few percent of steady
state, and that is true regardless of dose and regardless of interval. The
interval sets how high you end up; the half-life sets how long you take to get
there. Dosing more often does not reach steady state sooner, it reaches a
higher steady state in the same time.

The practical consequence is a loading dose: if the half-life is long and you
need the effect now, you cannot get there by waiting, so you give a bigger
first dose to jump straight to the level you want.

**Average concentration at steady state** has a remarkably simple form:

```
Cavg_ss = Dose / (tau * CL)
```

It depends only on dose, interval and clearance, not on volume or absorption
rate at all. Those shape the peaks and troughs within an interval but not the
average. This is why clearance is the parameter that matters most for choosing
a maintenance dose, and why a patient with impaired clearance accumulates to a
higher average on the same regimen.

**Simulating it.** Integrate one interval at a time. At the start of each,
add the dose to the depot amount, then integrate forward to the next dose
using the end state as the next starting state. That carry-forward is the
whole of it: `y0` for each interval is the previous interval's final state
with the dose added.

## Your task

Three parts, worked one at a time.

| Part | What you work out | Variables to fill in |
|---|---|---|
| a | the repeated dose profile, by simulation | `simulate`, `times`, `conc` |
| b | accumulation, simulated and algebraic | `cmax_first`, `cmax_ss`, `r_sim`, `r_theory` |
| c | average exposure and how long it took to get there | `cavg_ss`, `t_to_90pct`, `n_half_lives` |

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
from scipy.integrate import solve_ivp

PROBLEM_ID = "W7D5"
TOPIC = "PK: repeated dosing by simulation, accumulation ratio, time to steady state"

# The parameters fitted on day 3, rounded.
KA = 1.0465          # 1/h
V_F = 57.20          # L
CL_F = 5.6170        # L/h
KEL = CL_F / V_F     # 1/h

DOSE_MG = 200.0
TAU_H = 12.0         # dosing interval
N_DOSES = 20         # 10 days
POINTS_PER_INTERVAL = 121    # every 6 minutes

T_HALF = float(np.log(2.0) / KEL)

print("ka   = %.4f 1/h" % KA)
print("V/F  = %.2f L" % V_F)
print("CL/F = %.4f L/h" % CL_F)
print("kel  = CL/V = %.5f 1/h,  half-life %.3f h" % (KEL, T_HALF))
print("")
print("regimen: %.0f mg every %.0f h for %d doses (%.0f days)"
      % (DOSE_MG, TAU_H, N_DOSES, N_DOSES * TAU_H / 24.0))
print("")
print("the dosing interval is %.2f half-lives" % (TAU_H / T_HALF))
```

## Toolkit for this problem

| What you want to do | How |
|---|---|
| carry a state forward | use `sol.y[:, -1]` as the next `y0` |
| add a dose to the depot | `y0 = [y_end[0] + DOSE_MG, y_end[1]]` |
| stack results | collect arrays in a list, then `np.concatenate` |
| times within one interval | `np.linspace(0, TAU_H, POINTS_PER_INTERVAL)` |
| select one interval | `(times >= a) & (times < b)` |
| peak within a window | `conc[window].max()` |
| mean over a window | integrate and divide by the interval length |
| first time a condition holds | `times[np.argmax(condition)]` |

`sol.y` has one row per state and one column per reported time, so
`sol.y[:, -1]` is the final state of every compartment, which is exactly what
the next interval needs as its starting point.

The toolkit cell demonstrates each of those on a tiny made up example.
It is in the notebook and none of it is the answer.

## Part a. Simulate the repeated dosing

Write `simulate(n_doses)` returning two arrays: the times across the whole
regimen and the concentration at each.

The structure is a loop over intervals, carrying state forward:

- start with `[0.0, 0.0]`, an empty depot and an empty central compartment,
- at the start of each interval, **add** `DOSE_MG` to the depot amount,
- integrate one interval of `TAU_H` using `np.linspace(0, TAU_H,
  POINTS_PER_INTERVAL)` as `t_eval`,
- record the times shifted by how many intervals have passed, and the
  concentration, which is the central amount over `V_F`,
- take `sol.y[:, -1]`, the final state, as the starting state for the next
  interval.

Use `rtol=1e-9, atol=1e-11`.

The carry-forward is the whole trick, and the thing to be careful about is
adding the dose to the **depot** rather than to the central compartment. This
is an oral drug: each tablet arrives in the gut and has to be absorbed, so a
dose given while the previous one is still absorbing stacks up in the depot.
Adding it to the central compartment instead would model an intravenous bolus
and would give a visibly different, spikier profile.

Then call it for `N_DOSES` and keep the result.

Fill in `simulate`, `times`, `conc`.

Moves: a `for` loop, `solve_ivp` per interval, `sol.y[:, -1]`,
`np.concatenate`.

**Variables this part is graded on:** `times`

The starter cell they see:

```python
# Part a. Simulate 20 doses, 12 hours apart.

def rhs(t, y, ka, v, cl):
    """y[0] depot, y[1] central, both mg."""
    absorbed = ka * y[0]
    return [-absorbed, absorbed - (cl / v) * y[1]]


def simulate(n_doses):
    """Return (times, concentrations) across the whole regimen."""
    return None   # YOUR CODE: loop over intervals, carrying the state forward


times, conc = (None, None)   # YOUR CODE: simulate(N_DOSES)

if conc is not None:
    print("points simulated:", len(times))
    print("peak overall    : %.4f mg/L" % conc.max())
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(times, conc, lw=1.0)
    ax.set_xlabel("Time (h)"); ax.set_ylabel("Concentration (mg/L)")
    ax.set_title("200 mg every 12 h")
    plt.show()
```

**Every failure message part a can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `simulate` should be a function taking the number of doses.
- `times` and `conc` should both be numeric arrays.
- `times` and `conc` should be the same length.
- `times` has %d points but %d were expected, which is N_DOSES intervals of POINTS_PER_INTERVAL each.
- `times` does not run continuously across the regimen. Shift each interval's grid by i * TAU_H before stacking.
- `conc` matches dosing into the CENTRAL compartment, which is an intravenous bolus. This is an oral drug, so each dose goes into the depot and has to be absorbed.
- `conc` does not match. Check that the dose is ADDED to the existing depot amount rather than replacing it, and that concentration is the central amount divided by V_F.
- `conc` never accumulates, so the state is not being carried forward. Each interval should start from the previous one's final state, sol.y[:, -1], with the dose added.

## Part b. Accumulation, two ways

Take the peak concentration within the **first** dosing interval, so times
below `TAU_H`, and the peak within the **last** one, so times at or above
`(N_DOSES - 1) * TAU_H`. Their ratio is the observed accumulation.

Then compute the algebraic prediction:

```
r_theory = 1 / (1 - exp(-KEL * TAU_H))
```

They will be close but not identical: expect the simulated Cmax ratio to come
out a few percent **above** the formula.

That gap is worth understanding rather than rounding away. The formula is
derived assuming the dose appears instantly, so concentration starts falling
immediately. This drug absorbs over a couple of hours, so at steady state the
peak is reached while the previous dose is still washing out, which lifts the
observed peak ratio slightly. If you instead compare the **trough**
concentrations, where absorption is long finished, the simulation lands within
about one percent of the formula.

So the formula is a good answer for troughs and a slight underestimate for
peaks, and the size of the discrepancy is set by how fast absorption is
relative to elimination. Here `ka` is about eleven times `kel`, which is fast,
and even so it moves the answer by four percent. For a slowly absorbed
formulation it would move it considerably more, and the formula would stop
being usable for Cmax at all.

This is the general shape of the thing: the algebra is quick and tells you
what to expect, the simulation is what you trust when the assumptions behind
the algebra are not quite true. Running both and understanding the gap beats
running either alone.

Note what `r_theory` does not contain. There is no dose in it, and no volume,
and no absorption rate. Accumulation is set entirely by how the dosing
interval compares with the half-life. A drug dosed at exactly one half-life
accumulates two-fold whatever else is true about it.

Fill in `cmax_first`, `cmax_ss`, `r_sim`, `r_theory`.

Moves: boolean masks on `times`, `.max()`, division, `np.exp`.

**Variables this part is graded on:** `cmax_first`, `cmax_ss`, `r_sim`, `r_theory`

The starter cell they see:

```python
# Part b. Accumulation, simulated and predicted.

cmax_first = None   # YOUR CODE: peak within the FIRST dosing interval
cmax_ss = None      # YOUR CODE: peak within the LAST dosing interval
r_sim = None        # YOUR CODE: cmax_ss / cmax_first
r_theory = None     # YOUR CODE: 1 / (1 - exp(-KEL * TAU_H))

print("Cmax, first interval : %s mg/L" % cmax_first)
print("Cmax, last interval  : %s mg/L" % cmax_ss)
print("accumulation ratio, simulated : %s" % r_sim)
print("accumulation ratio, algebraic : %s" % r_theory)
```

**Every failure message part b can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `cmax_first` should be a single number.
- `cmax_first` is the steady state peak. Restrict to times below TAU_H for the first interval.
- `cmax_first` is not the peak within the first dosing interval.
- `cmax_ss` should be a single number.
- `cmax_ss` is not the peak within the last dosing interval, which is times at or above (N_DOSES - 1) * TAU_H.
- `r_sim` should be a single number.
- `r_sim` is the other way up. It is the steady state peak over the first peak.
- `r_sim` is not cmax_ss divided by cmax_first.
- `r_theory` should be a single number.
- `r_theory` is the denominator rather than the ratio. It is 1 divided by (1 - exp(-KEL * TAU_H)).
- `r_theory` is not 1 / (1 - exp(-KEL * TAU_H)).

## Part c. Average exposure, and how long it took

**Average concentration at steady state** has the simple closed form

```
cavg_ss = DOSE_MG / (TAU_H * CL_F)
```

Only dose, interval and clearance. Volume and absorption rate shape the peaks
and troughs within an interval but cancel out of the average entirely.

**How long it took.** Find the first time the simulated concentration reaches
90 percent of `cmax_ss`. A boolean comparison plus `np.argmax` gives the first
position where it becomes true, and that position indexes into `times`.

Then express that as a number of half-lives, `t_to_90pct / T_HALF`.

The number you get should land near three and a bit, which is what 90 percent
of the way to steady state costs. Getting to 95 percent takes about four
half-lives and 99 percent about seven, which is where the rule of thumb about
five half-lives comes from.

The thing worth taking away is what that number is **not** a function of. It
does not depend on the dose, and it does not depend on the dosing interval.
Halve the interval and you reach a higher steady state in the same elapsed
time. If you need the effect sooner than the half-life allows, no regimen of
ordinary doses will get you there, which is exactly why loading doses exist.

Fill in `cavg_ss`, `t_to_90pct`, `n_half_lives`.

Moves: division, a boolean comparison on `conc`, `np.argmax`, indexing
`times`.

**Variables this part is graded on:** `cavg_ss`, `t_to_90pct`, `n_half_lives`

The starter cell they see:

```python
# Part c. Average exposure, and time to steady state.

cavg_ss = None       # YOUR CODE: DOSE_MG / (TAU_H * CL_F)
t_to_90pct = None    # YOUR CODE: first time conc reaches 90 percent of cmax_ss
n_half_lives = None  # YOUR CODE: t_to_90pct / T_HALF

print("Cavg at steady state : %s mg/L" % cavg_ss)
print("time to 90 percent   : %s h" % t_to_90pct)
print("which is             : %s half-lives" % n_half_lives)
```

**Every failure message part c can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `cavg_ss` should be a single number.
- `cavg_ss` left out the dosing interval. It is the dose divided by TAU_H times CL_F, because the average is over the whole interval.
- `cavg_ss` is not DOSE_MG / (TAU_H * CL_F).
- `t_to_90pct` should be a single number.
- `t_to_90pct` is the index, not the time. Use it to look up the matching entry of times.
- `t_to_90pct` is not the first time conc reaches 90 percent of cmax_ss. np.argmax on a boolean array gives the first True.
- `n_half_lives` should be a single number.
- `n_half_lives` is not t_to_90pct divided by T_HALF.

**Functions this problem introduces:** carrying state forward, accumulation ratio, time to steady state, Cavg at steady state
