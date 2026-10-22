# Week 8, PK and PD applications, part 2

The full text of every problem this week, the data it generates, the
toolkit the student is given, what each part asks for, and what every
possible check failure means. No solutions in this file.

---

# W8D1. PK/PD link: why the effect peaks hours after the concentration

- topic: PK/PD: hysteresis, direct Emax fitting and its residual structure, the effect compartment
- notebook: problems/week08/w8d1_pkpd_hysteresis.ipynb
- solution: solutions/week08/w8d1_pkpd_hysteresis_solution.ipynb
- expected time: 25-35 min

## Scenario

A thorough QT study has produced paired samples: plasma concentration and
change from baseline in QTc, at twelve time points after a single oral dose in
one subject.

The concentration peaks at three hours. The QTc change peaks at six. Somebody
in the meeting is going to say that the assay must be wrong, and somebody else
is going to fit a concentration response curve to the pairs, get an EC50, and
put it on a slide.

Both are mistakes, and today you will produce the evidence for why. The delay
is real, it is a property of where the drug has to get to before it acts, and
ignoring it does not just add noise: it biases the Emax and EC50 estimates
badly enough to change a dose decision.

## Hysteresis, and what to do about it

**The loop.** Plot effect against concentration, with the points joined in
time order. If effect depended only on the concentration in plasma at that
instant, the points would lie on a single curve: the same concentration would
always give the same effect. Instead they trace a loop, because a concentration
on the way up and the same concentration on the way down produce different
effects.

If the effect is larger on the way down, the loop runs **counter clockwise**,
and the cause is almost always distribution: the drug has to reach a site that
plasma is not. If the effect is smaller on the way down, the loop runs
clockwise, and the usual causes are tolerance or an active metabolite being
cleared.

**Why a direct fit fails.** Fitting `E = Emax * C / (EC50 + C)` to the pairs
asks the optimiser to find one curve through a loop. It will succeed, because
least squares always returns something, and the answer will be a compromise
that over predicts the early points and under predicts the late ones. The
signature is unmistakable once you look for it: **the residuals correlate with
time**. Negative before the concentration peak, positive after, in a run rather
than a scatter.

That is the single most useful diagnostic in PK/PD work. Plot residuals against
time, every time.

**The effect compartment.** The standard fix, from Sheiner and Holford around
1979, is to add one hypothetical compartment linked to plasma by a single rate
constant `ke0`:

```
dCe/dt = ke0 * (C(t) - Ce)
```

`Ce` is the concentration at the effect site. It has no volume, carries no
mass, and exists only to delay and smooth the plasma profile. The effect is
then driven by `Ce` instead of `C`, through the same Emax equation.

One extra parameter buys you the whole loop. `ke0` has units of per hour, and
`ln(2) / ke0` is the effect site equilibration half life: how long the site
takes to catch up with plasma. A large `ke0` means no delay and the loop
collapses to a curve. A small `ke0` means a slow, heavily smoothed effect that
peaks long after the concentration.

**On fitting it.** Three parameters, `Emax`, `EC50` and `ke0`, and the model
needs an ODE solved for every candidate `ke0`. `curve_fit` handles this without
complaint as long as you give it bounds and a sensible starting point. An
alternative worth knowing is to fix `ke0` on a grid, fit the two Emax
parameters at each value, and keep the best: slower, but it shows you the whole
profile of the objective function and whether `ke0` is actually identified.

## Your task

Three parts, worked one at a time.

| Part | What you work out | Variables to fill in |
|---|---|---|
| a | the loop, and how wide it is | `tmax_c`, `tmax_e`, `e_rising`, `e_falling` |
| b | the direct fit, and the residual pattern that condemns it | `emax_direct`, `ec50_direct`, `sse_direct`, `resid_early`, `resid_late` |
| c | the effect compartment, and what it recovers | `ke0_hat`, `emax_hat`, `ec50_hat`, `link_pred`, `sse_link` |

Below this you will find the **Setup** cell, then a **Toolkit** cell that
demonstrates every move on a tiny made up example, then **Part a**, **Part b**
and **Part c**, each with its own explanation and a cell to write in.

## The setup cell the student runs

This is given to them, already working. The data is synthetic and
seeded, so their numbers are always these numbers.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

PROBLEM_ID = "W8D1"
TOPIC = "PK/PD: hysteresis, direct Emax fitting and its residual structure, the effect compartment"

rng = np.random.default_rng(8101)

DOSE = 100.0        # mg, single oral dose
_VF = 35.0          # L, apparent volume, not known to you
_KA = 1.0           # per h, absorption, not known to you
_KE = 0.12          # per h, elimination, not known to you
_KE0 = 0.25         # per h, effect site equilibration, not known to you
_EMAX = 30.0        # ms, not known to you
_EC50 = 0.5         # mg/L, not known to you

_coef = DOSE * _KA / (_VF * (_KA - _KE))


def _true_conc(t):
    return _coef * (np.exp(-_KE * t) - np.exp(-_KA * t))


def _true_ce(t_eval, ke0):
    out = solve_ivp(lambda t, y: ke0 * (_true_conc(t) - y[0]),
                    (0.0, float(np.max(t_eval))), [0.0],
                    t_eval=np.asarray(t_eval, dtype=float), rtol=1e-9, atol=1e-12)
    return out.y[0]


t_obs = np.array([0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 18.0, 24.0])
c_obs = np.round(_true_conc(t_obs) * rng.lognormal(0.0, 0.06, t_obs.size), 4)
_ce = _true_ce(t_obs, _KE0)
e_obs = np.round(_EMAX * _ce / (_EC50 + _ce) + rng.normal(0.0, 1.5, t_obs.size), 3)

C_PROBE = 1.2       # mg/L, the concentration at which to compare the two limbs

print("time (h)      :", t_obs)
print("conc (mg/L)   :", c_obs)
print("dQTc (ms)     :", e_obs)
print("")
print("concentration peaks at t = %g h" % t_obs[int(np.argmax(c_obs))])
print("effect peaks at        t = %g h" % t_obs[int(np.argmax(e_obs))])
print("probe concentration for the hysteresis comparison: %g mg/L" % C_PROBE)
```

## Toolkit for this problem

| What you want to do | How |
|---|---|
| position of a maximum | `int(np.argmax(y))` |
| split at the peak | `idx = int(np.argmax(c))`, then `c[:idx + 1]` and `c[idx:]` |
| reverse an array | `a[::-1]` |
| interpolate | `np.interp(x_new, x, y)` with `x` increasing |
| solve the effect compartment | `solve_ivp(lambda t, y: ke0 * (conc(t) - y[0]), (0, tmax), [0.0], t_eval=times)` |
| fit with bounds | `curve_fit(f, x, y, p0=[...], bounds=([lo], [hi]))` |
| standard errors | `np.sqrt(np.diag(pcov))` |
| mean of a slice | `resid[:idx + 1].mean()` |

`np.interp` is the trap in this problem. It requires its `x` argument to be
increasing and **does not check**: hand it a decreasing array and you get
numbers back, silently wrong ones. The falling limb of a concentration curve is
decreasing, so it has to be reversed, both arrays together.

The toolkit cell demonstrates each of those on a tiny made up example.
It is in the notebook and none of it is the answer.

## Part a. Measure the loop

First establish that there is a delay at all, then measure how wide it is.

- `tmax_c`, the time of the highest observed concentration,
- `tmax_e`, the time of the highest observed effect,
- `e_rising`, the effect you would expect at `C_PROBE` on the way up, by
  interpolating the rising limb,
- `e_falling`, the same thing on the way down.

Split the samples at the observed concentration peak, with the peak sample
belonging to both limbs. The rising limb's concentrations are already
increasing so `np.interp` will accept them directly. The falling limb's are
decreasing, so reverse both arrays before interpolating, or you will get a
number that looks plausible and is wrong.

What comes out is a difference of about fifteen milliseconds at the same plasma
concentration, which for a QT study is the difference between a drug you can
develop and one you cannot. That is the whole argument for taking the delay
seriously, and it took four lines of numpy.

The cell also plots the loop. Look at the direction: the points move left along
the bottom as the concentration rises with little effect, climb at the far
left, and come back along the top. Counter clockwise, so the effect site lags
plasma.

Fill in `tmax_c`, `tmax_e`, `e_rising`, `e_falling`.

Moves: `np.argmax`, slicing at the peak, `a[::-1]`, `np.interp`.

**Variables this part is graded on:** `peak_idx`, `tmax_c`, `tmax_e`, `e_rising`, `e_falling`

The starter cell they see:

```python
# Part a. The hysteresis loop.

peak_idx = None   # YOUR CODE: index of the highest observed concentration

tmax_c = None     # YOUR CODE: time of the concentration peak
tmax_e = None     # YOUR CODE: time of the effect peak

e_rising = None   # YOUR CODE: effect at C_PROBE, interpolated on the rising limb
e_falling = None  # YOUR CODE: effect at C_PROBE, interpolated on the falling limb

print("concentration peaks at %r h, effect at %r h" % (tmax_c, tmax_e))
print("at %.2f mg/L: rising limb %r ms, falling limb %r ms" % (C_PROBE, e_rising, e_falling))
if e_rising is not None and e_falling is not None:
    print("difference at the same concentration: %.3f ms" % (e_falling - e_rising))

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(c_obs, e_obs, "o-", color="#333333")
for _i in range(0, len(t_obs), 2):
    ax.annotate("%g h" % t_obs[_i], (c_obs[_i], e_obs[_i]),
                textcoords="offset points", xytext=(6, -3), fontsize=8)
ax.axvline(C_PROBE, ls=":", color="grey")
ax.set_xlabel("plasma concentration (mg/L)")
ax.set_ylabel("change in QTc (ms)")
ax.set_title("W8D1 effect against concentration, joined in time order")
plt.show()
```

**Every failure message part a can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `tmax_c` is the index of the peak, not the time. Use it to index t_obs.
- `tmax_c` is not the time of the highest observed concentration.
- `tmax_e` equals `tmax_c`. It should be the time of the highest observed EFFECT, which is a different sample.
- `tmax_e` is not the time of the highest observed effect.
- `e_rising` is not the effect interpolated at C_PROBE on the rising limb. The rising limb is c_obs up to and including the peak sample.
- `e_falling` came from np.interp on a decreasing array, which returns a number without complaining and is wrong. Reverse the falling limb, both the concentrations and the effects.
- `e_falling` matches `e_rising`, so both came from the same limb.
- `e_falling` is not the effect interpolated at C_PROBE on the falling limb. The falling limb runs from the peak sample to the last sample.
- `%s` should be a single number.

## Part b. Fit the wrong model, properly

Fit the direct Emax model

```
E = Emax * C / (EC50 + C)
```

to the twelve concentration and effect pairs with `curve_fit`. Start from
`p0=[20.0, 1.0]` and bound both parameters below at something small and
positive, because a negative EC50 is meaningless and an unbounded fit will
wander into it.

Record `emax_direct`, `ec50_direct` and `sse_direct`, the summed squared
residuals.

Then compute the diagnostic that matters. With the residuals as **observed
minus predicted**, in time order:

- `resid_early`, the mean residual up to and including the concentration peak,
- `resid_late`, the mean residual after it.

They come out at about minus five and plus five milliseconds. A good fit has
both near zero; these are a run of five negative residuals followed by a run of
seven positive ones, which is not noise, it is the model being wrong in a
systematic way.

Now look at what the fitted parameters say. The Emax came out around 17 ms and
the EC50 around 0.1 mg/L, which describes a drug whose effect is almost
saturated by the lowest concentration sampled. The truth, which you will
recover in part c, is an Emax of 30 ms and an EC50 of 0.5 mg/L. The direct fit
has not merely added noise, it has flattened the curve and moved the potency by
a factor of five, and both errors push in the direction of thinking the drug is
safer at high doses than it is.

Fill in `emax_direct`, `ec50_direct`, `sse_direct`, `resid_early`,
`resid_late`.

Moves: `curve_fit` with `p0` and `bounds`, residuals, slicing at the peak.

**Variables this part is graded on:** `emax_direct`, `ec50_direct`, `sse_direct`, `resid_early`, `resid_late`

The starter cell they see:

```python
# Part b. The direct Emax fit, and its residuals.

def direct_emax(c, emax, ec50):
    """Effect as a function of the concentration at that instant."""
    return None  # YOUR CODE: emax * c / (ec50 + c)


emax_direct = None  # YOUR CODE: fitted Emax in ms
ec50_direct = None  # YOUR CODE: fitted EC50 in mg/L
sse_direct = None   # YOUR CODE: summed squared residuals of this fit

resid_early = None  # YOUR CODE: mean residual up to and including the conc peak
resid_late = None   # YOUR CODE: mean residual after the conc peak

print("direct fit: Emax %r ms, EC50 %r mg/L" % (emax_direct, ec50_direct))
print("SSE       : %r" % (sse_direct,))
print("mean residual before the peak: %r" % (resid_early,))
print("mean residual after the peak : %r" % (resid_late,))
```

**Every failure message part b can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `emax_direct` is not the fitted Emax of the direct model. Fit direct_emax to c_obs and e_obs from p0=[20.0, 1.0] with both parameters bounded positive.
- `ec50_direct` is not the fitted EC50 of the direct model.
- `sse_direct` is the link model's SSE. Part b is the direct fit.
- `sse_direct` is not the summed squared residuals of the direct fit at the observed concentrations.
- `resid_early` holds the late mean. The early window is up to and including the concentration peak.
- `resid_early` should be negative: the direct fit over predicts the early samples, so observed minus predicted is below zero there. Check the residual order, which is observed minus predicted.
- `resid_early` is not the mean residual up to and including the concentration peak.
- `resid_late` should be positive, since the direct fit under predicts the late samples.
- `resid_late` is not the mean residual after the concentration peak.
- `%s` should be a single number.
- `direct_emax` returns None. It should return emax * c / (ec50 + c).
- `direct_emax` does not evaluate the Emax model. With emax = 10 and ec50 = 1, concentrations of 1 and 2 should give 5 and 6.667.
- `direct_emax` raised %s: %s

## Part c. Add the effect compartment

Write a model that takes the sampling times and three parameters, solves

```
dCe/dt = ke0 * (C(t) - Ce)
```

from zero with `Ce(0) = 0`, and returns `Emax * Ce / (EC50 + Ce)` at those
times.

You need a plasma concentration function `C(t)` to drive it. You do not have
the PK parameters, so use the observations themselves: interpolate `c_obs`
against `t_obs` with `np.interp`, which is legitimate here because the
concentrations are measured densely enough over the range that matters. In a
real analysis you would fit the PK first and drive the effect compartment with
the fitted curve, which avoids feeding assay noise into the PD model.

Fit `Emax`, `EC50` and `ke0` with `curve_fit`, starting from
`p0=[20.0, 1.0, 0.3]` and bounding all three positive. Record `ke0_hat`,
`emax_hat`, `ec50_hat`, the predictions in `link_pred`, and the summed squared
residuals in `sse_link`.

What you should see:

- `Emax` about 30.3 ms and `EC50` about 0.50 mg/L, both within a few percent
  of the truth, where the direct fit gave 17 and 0.1,
- `ke0` about 0.21 per hour, an effect site half life of 3.3 hours, against a
  true 0.25. That is sixteen percent low, and the standard error from `pcov`
  is only 0.008, so the interval does not come close to covering the truth.
  Do not trust that standard error. The model treats the interpolated
  concentrations as exact, when they are straight line segments drawn through
  noisy samples, cutting the corner at the peak. None of that uncertainty is
  in `pcov`, and some of it has been absorbed into `ke0` as bias. This is the
  concrete reason a real analysis fits the PK model first and drives the
  effect compartment with the fitted curve rather than with the observations,
- an SSE of about 27 against the direct fit's 567, a twenty one fold reduction
  for one extra parameter,
- and, in the printout, mean residuals near zero both before and after the
  peak. The structure is gone, which is the real evidence that the extra
  parameter belongs there rather than just having absorbed some noise.

Fill in `ke0_hat`, `emax_hat`, `ec50_hat`, `link_pred`, `sse_link`.

Moves: `solve_ivp` inside a model function, `np.interp` as an input function,
`curve_fit` with three parameters and bounds.

**Variables this part is graded on:** `ke0_hat`, `emax_hat`, `ec50_hat`, `link_pred`, `sse_link`

The starter cell they see:

```python
# Part c. The effect compartment model.

def plasma(t):
    """Plasma concentration at arbitrary times, interpolated from the samples."""
    return None  # YOUR CODE: np.interp(t, t_obs, c_obs)


def link_model(t, emax, ec50, ke0):
    """Effect driven by the effect site concentration."""
    # YOUR CODE: solve dCe/dt = ke0 * (plasma(t) - Ce) from 0, with Ce(0) = 0,
    # evaluated at t, then return emax * ce / (ec50 + ce)
    return None


ke0_hat = None    # YOUR CODE: fitted ke0 in per h
emax_hat = None   # YOUR CODE: fitted Emax in ms
ec50_hat = None   # YOUR CODE: fitted EC50 in mg/L
link_pred = None  # YOUR CODE: the fitted effects at t_obs
sse_link = None   # YOUR CODE: summed squared residuals of this fit

print("link fit: Emax %r ms, EC50 %r mg/L, ke0 %r per h" % (emax_hat, ec50_hat, ke0_hat))
print("SSE     : %r   (direct fit was %r)" % (sse_link, sse_direct))
if ke0_hat is not None:
    print("effect site half life: %.3f h" % (np.log(2) / ke0_hat))
if link_pred is not None:
    _rl = e_obs - np.asarray(link_pred, dtype=float)
    print("mean residual before the peak: %.4f" % _rl[:peak_idx + 1].mean())
    print("mean residual after the peak : %.4f" % _rl[peak_idx + 1:].mean())
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t_obs, e_obs, "o", label="observed")
    ax.plot(t_obs, link_pred, "-", label="effect compartment fit")
    ax.plot(t_obs, direct_emax(c_obs, emax_direct, ec50_direct), "--", label="direct fit")
    ax.set_xlabel("time (h)")
    ax.set_ylabel("change in QTc (ms)")
    ax.set_title("W8D1 two models, one dataset")
    ax.legend()
    plt.show()
```

**Every failure message part c can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `link_pred` should be an array of predicted effects.
- `link_pred` has shape %s but there are %d sampling times.
- `emax_hat` is not the fitted Emax of the effect compartment model. Check that Ce starts at zero and that the solver is evaluated at t_obs.
- `ec50_hat` is not the fitted EC50 of the effect compartment model.
- `ke0_hat` is not the fitted effect site rate constant. It should come out near 0.2 per h, a half life of a few hours.
- `link_pred` does not match the fitted effects at t_obs. It should be the link model evaluated at the fitted parameters.
- `sse_link` is the direct fit's SSE.
- `sse_link` is not the summed squared residuals of the effect compartment fit.
- `%s` should be a single number.
- `emax_hat` is the direct fit's Emax. The link model should give a much larger value, close to the truth.
- `ke0_hat` ran to its upper bound, which means the fit decided there is no delay. Check the sign in dCe/dt = ke0 * (C(t) - Ce).

**Functions this problem introduces:** np.interp, hysteresis, effect compartment, residuals against time, curve_fit bounds

---

# W8D2. Indirect response: when the biomarker, not the drug, sets the clock

- topic: PK/PD: turnover models, inhibition of production, dose dependent time to nadir
- notebook: problems/week08/w8d2_indirect_response.ipynb
- solution: solutions/week08/w8d2_indirect_response_solution.ipynb
- expected time: 25-35 min

## Scenario

The drug inhibits an enzyme that produces an inflammatory biomarker. The
biomarker sits at a steady baseline because it is made and removed
continuously, and the drug slows the making.

Yesterday's effect compartment would describe the delay as the drug taking
time to reach its site of action. Today's mechanism is different and more
common: the drug reaches its target immediately, and the delay belongs to the
biomarker, which cannot fall faster than its own turnover allows. Block
production completely and the level still only falls at the rate it was being
removed anyway.

The practical consequences are large. The effect outlasts the drug by days.
The time of maximum effect moves later as the dose goes up, which no effect
compartment model can produce. And the quantity that governs the duration of
action is not a property of the drug at all, it is the biomarker's turnover
half life.

## The turnover model

**The equation.** One state, the biomarker level `R`, made at rate `kin` and
removed by a first order process `kout`:

```
dR/dt = kin - kout * R
```

Before any drug this settles at `R(0) = kin / kout`, the baseline. That is the
most useful fact in the model and the one people forget: **the baseline pins
`kin`**. If you know the baseline and you know `kout`, then `kin` is not a free
parameter, it is `kout * R0`. A fit that treats all three as free is fitting
one parameter twice and will not converge cleanly.

**The drug.** Four places it can act, and the four standard models are named
for them. Inhibition or stimulation of production, inhibition or stimulation of
loss. Today is inhibition of production, which multiplies `kin` by an inhibitory
Emax term:

```
dR/dt = kin * (1 - Imax * C / (IC50 + C)) - kout * R
```

`Imax` is between 0 and 1 and says how completely production can be shut off.
`Imax = 0.9` means that even at infinite concentration, ten percent of
production continues.

**Where the delay comes from.** Set `C` to something large and hold it there.
Production nearly stops, and `R` decays towards its new steady state with the
rate constant `kout`, so the approach has a half life of `ln(2) / kout`. The
drug cannot make the biomarker fall faster than that, however much of it you
give. A biomarker with a twelve hour turnover half life takes about two days to
reach a new steady state no matter what you do to production.

**Why the time of the nadir moves with dose.** The nadir happens when the
reduced production rate and the removal rate `kout * R` come back into balance.
A bigger dose keeps the concentration above `IC50` for longer, so production
stays suppressed for longer, so the balance point arrives later. The time the
concentration spends above `IC50` grows roughly with the logarithm of the dose,
and the time to the nadir follows it.

This is the clean diagnostic between the two mechanisms. **An effect
compartment puts the maximum effect at the same time for every dose.** An
indirect response model puts it later and later. If you have data at three dose
levels, this tells you which model you need, and it is one of the few places in
PK/PD where the answer is not a matter of taste.

## Your task

Three parts, worked one at a time.

| Part | What you work out | Variables to fill in |
|---|---|---|
| a | the model, and one dose through it | `kin`, `nadir`, `t_nadir`, `pct_suppression` |
| b | three doses, and the diagnostic that separates the mechanisms | `t_nadir_by_dose`, `nadir_by_dose`, `t_above_by_dose`, `t_link_by_dose` |
| c | duration of action, which belongs to the biomarker | `auec`, `t_recover`, `t_nadir_fast`, `t_nadir_slow` |

Below this you will find the **Setup** cell, then a **Toolkit** cell that
demonstrates every move on a tiny made up example, then **Part a**, **Part b**
and **Part c**, each with its own explanation and a cell to write in.

## The setup cell the student runs

This is given to them, already working. The data is synthetic and
seeded, so their numbers are always these numbers.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

PROBLEM_ID = "W8D2"
TOPIC = "PK/PD: turnover models, inhibition of production, dose dependent time to nadir"

# Pharmacokinetics, taken as known from an earlier study.
VF = 35.0            # L, apparent volume of distribution
KA = 1.0             # per h, first order absorption
KE = 0.12            # per h, first order elimination

# Pharmacodynamics of the biomarker.
R_BASE = 100.0       # units/L, the pre dose baseline
T_HALF_TURNOVER = 12.0   # h, the biomarker's own turnover half life
IMAX = 0.9           # at most 90 pct of production can be blocked
IC50 = 0.5           # mg/L, concentration for half of that

KOUT = np.log(2.0) / T_HALF_TURNOVER

DOSE = 100.0         # mg, the dose for part a
DOSES = np.array([25.0, 100.0, 400.0])   # mg, the three dose levels for part b
KE0_COMPARE = 0.03   # per h, for the effect compartment comparison in part b

# One shared time grid, so that everyone reads the same times off it.
t_grid = np.linspace(0.0, 168.0, 1681)   # 0 to 7 days in 0.1 h steps
DT = float(t_grid[1] - t_grid[0])


def conc(t, dose):
    """Plasma concentration after a single oral dose, one compartment."""
    return dose * KA / (VF * (KA - KE)) * (np.exp(-KE * t) - np.exp(-KA * t))


print("turnover half life %.1f h, so kout = %.6f per h" % (T_HALF_TURNOVER, KOUT))
print("baseline %.1f units/L" % R_BASE)
print("Imax %.2f, IC50 %.2f mg/L" % (IMAX, IC50))
print("")
print("grid: %d points from %g to %g h, step %.1f h"
      % (t_grid.size, t_grid[0], t_grid[-1], DT))
print("concentration after %g mg: Cmax %.4f mg/L at %.2f h"
      % (DOSE, conc(t_grid, DOSE).max(), t_grid[int(np.argmax(conc(t_grid, DOSE)))]))
print("concentration after %g mg at 48 h: %.5f mg/L" % (DOSE, conc(48.0, DOSE)))
```

## Toolkit for this problem

| What you want to do | How |
|---|---|
| solve one state | `solve_ivp(rhs, (0, tmax), [r0], t_eval=t_grid, rtol=1e-9, atol=1e-10)` |
| the solution values | `sol.y[0]` |
| a right hand side with a time varying input | `lambda t, y: kin * (1 - imax * c(t) / (ic50 + c(t))) - kout * y[0]` |
| position of a minimum | `int(np.argmin(r))` |
| how long a condition held | `(c > threshold).sum() * DT` |
| area under a curve | `np.trapezoid(y, x)`, or `np.trapz(y, x)` on older numpy |
| the first index satisfying a condition | `int(np.where(flags)[0][0])` |

`np.trapezoid` is the current name and `np.trapz` is the old one. Colab may have
either, so the safe pattern is to compute the trapezoids yourself, or to try one
and fall back to the other. The toolkit below shows a version that works on both.

The toolkit cell demonstrates each of those on a tiny made up example.
It is in the notebook and none of it is the answer.

## Part a. One dose through the turnover model

First work out `kin`. You know the baseline `R_BASE` and you know `KOUT`, and
at baseline production equals removal, so `kin = KOUT * R_BASE`. Writing this
down rather than fitting it is the whole trick with indirect response models.

Then solve

```
dR/dt = kin * (1 - IMAX * C(t) / (IC50 + C(t))) - KOUT * R
```

from `R(0) = R_BASE` over `t_grid`, with `C(t)` given by the `conc` function
and `DOSE` of 100 mg. Use `rtol=1e-9, atol=1e-10`, which matters here: the
default tolerances leave visible error in the tail, and you are about to read a
time off this curve to one decimal place.

From the solution record:

- `nadir`, the lowest biomarker level reached,
- `t_nadir`, the time at which it happened, read off `t_grid`,
- `pct_suppression`, the maximum suppression as a percentage of baseline,
  `100 * (1 - nadir / R_BASE)`.

The concentration peaks at 2.4 hours. The biomarker bottoms out at 18.8 hours,
sixteen hours later, and the drug is down to a hundredth of its peak by then.
Nothing is wrong with the simulation. The biomarker is simply still catching up.

Note also that the suppression is only 36 percent although `Imax` is 0.9. The
concentration is above `IC50` for about fifteen hours out of a week, so
production is never shut off for long enough to approach the maximum effect.
Imax is an asymptote, not a prediction.

Fill in `kin`, `nadir`, `t_nadir`, `pct_suppression`.

Moves: `solve_ivp` with a time varying input, `np.argmin`, indexing `t_grid`.

**Variables this part is graded on:** `kin`, `r_single`, `nadir`, `t_nadir`, `pct_suppression`

The starter cell they see:

```python
# Part a. The turnover model at one dose.

kin = None  # YOUR CODE: production rate implied by the baseline and KOUT


def rhs(t, y, dose):
    """dR/dt for inhibition of production."""
    c = conc(t, dose)
    return None  # YOUR CODE: kin * (1 - IMAX * c / (IC50 + c)) - KOUT * y[0]


r_single = None  # YOUR CODE: the solved biomarker profile over t_grid, as an array

nadir = None            # YOUR CODE: lowest level reached
t_nadir = None          # YOUR CODE: time at which it happened
pct_suppression = None  # YOUR CODE: maximum suppression as a percentage of baseline

print("kin = %r units/L/h" % (kin,))
print("nadir %r units/L at t = %r h" % (nadir, t_nadir))
print("maximum suppression %r pct of baseline" % (pct_suppression,))

if r_single is not None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t_grid, r_single, "-", color="#1f4e79", label="biomarker")
    ax.set_xlabel("time (h)")
    ax.set_ylabel("biomarker (units/L)")
    ax.set_xlim(0, 96)
    ax2 = ax.twinx()
    ax2.plot(t_grid, conc(t_grid, DOSE), "--", color="#999999", label="concentration")
    ax2.set_ylabel("concentration (mg/L)")
    ax.set_title("W8D2 the effect outlasts the drug")
    ax.legend(loc="lower right")
    plt.show()
```

**Every failure message part a can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `kin` is R_BASE divided by KOUT. At baseline production equals removal, so kin = KOUT * R_BASE.
- `kin` is not the production rate implied by the baseline. With dR/dt = kin - kout * R at rest, kin = KOUT * R_BASE.
- `r_single` should be the solved biomarker profile, an array of numbers.
- `r_single` has shape %s but t_grid has %d points. Pass t_eval=t_grid to solve_ivp and take sol.y[0].
- `r_single` does not start at the baseline. The initial condition is [R_BASE], the pre dose steady state.
- `r_single` rises above the baseline, which inhibition of production cannot do. Check the sign: the drug term multiplies kin by (1 - IMAX * c / (IC50 + c)), which is between 0.1 and 1.
- `nadir` is not the lowest biomarker level reached. Check the tolerances: the task asks for rtol=1e-9, atol=1e-10.
- `t_nadir` is an index, not a time. Use it to index t_grid.
- `t_nadir` is not the time of the nadir, read off t_grid.
- `pct_suppression` is a fraction. The task asks for a percentage.
- `pct_suppression` is the nadir as a percentage OF baseline, which is the level remaining. The suppression is 100 * (1 - nadir / R_BASE).
- `pct_suppression` is not the maximum suppression as a percentage.
- `%s` should be a single number.

## Part b. Three doses, and the mechanism diagnostic

Run the same model at 25, 100 and 400 mg, and collect, as arrays of three
numbers in dose order:

- `t_nadir_by_dose`, the time of the nadir at each dose,
- `nadir_by_dose`, the nadir itself,
- `t_above_by_dose`, how long the concentration stays above `IC50` at each
  dose. Count grid points where `conc` exceeds `IC50` and multiply by `DT`.

Then build the competing explanation and watch it fail. For each dose, solve
the effect compartment

```
dCe/dt = KE0_COMPARE * (C(t) - Ce)
```

from `Ce(0) = 0`, and compute the same inhibitory effect on `Ce` instead:
`R_BASE * (1 - IMAX * Ce / (IC50 + Ce))`. Record the time of its minimum in
`t_link_by_dose`.

Here is the comparison you have just produced.

| dose | time above IC50 | indirect nadir time | effect compartment nadir time |
|---|---|---|---|
| 25 mg | 2.2 h | 15.3 h | 16.5 h |
| 100 mg | 15.4 h | 18.8 h | 16.5 h |
| 400 mg | 27.1 h | 24.1 h | 16.5 h |

The effect compartment column is constant to the last digit, and it is constant
for a reason rather than by accident: `Ce` is a linear filter of `C`, so
multiplying the dose multiplies `Ce` by the same factor everywhere and cannot
move the position of its peak. The effect is a monotone function of `Ce`, so its
extreme sits at the same time too, at every dose, forever.

The indirect column moves by nine hours across a sixteen fold dose range, and
it tracks the time above `IC50`. That is the fingerprint, and it means three
dose levels of biomarker data are enough to tell you which mechanism you are
looking at.

The nadir depths, 18.6, 36.4 and 53.1 percent suppression, are worth a second
look as well. Sixteen times the dose did not get you three times the effect.
Between the Emax saturation and the turnover limit, the dose response of an
indirect biomarker is much flatter than the dose response of a concentration.

Fill in `t_nadir_by_dose`, `nadir_by_dose`, `t_above_by_dose`,
`t_link_by_dose`.

Moves: a loop over `DOSES`, `solve_ivp` twice per dose, `np.argmin`,
`(c > IC50).sum() * DT`.

**Variables this part is graded on:** `t_nadir_by_dose`, `nadir_by_dose`, `t_above_by_dose`, `t_link_by_dose`

The starter cell they see:

```python
# Part b. Dose dependence, and the effect compartment comparison.

t_nadir_by_dose = None  # YOUR CODE: array of 3 nadir times, indirect model
nadir_by_dose = None    # YOUR CODE: array of 3 nadir levels, indirect model
t_above_by_dose = None  # YOUR CODE: array of 3 durations above IC50, in h

t_link_by_dose = None   # YOUR CODE: array of 3 nadir times, effect compartment model

print("dose (mg)        :", DOSES)
print("time above IC50  :", t_above_by_dose)
print("indirect t_nadir :", t_nadir_by_dose)
print("indirect nadir   :", nadir_by_dose)
print("link model t_min :", t_link_by_dose)

if nadir_by_dose is not None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for _d in DOSES:
        _s = solve_ivp(rhs, (float(t_grid[0]), float(t_grid[-1])), [R_BASE],
                       t_eval=t_grid, args=(_d,), rtol=1e-9, atol=1e-10)
        ax.plot(t_grid, _s.y[0], "-", label="%g mg" % _d)
    ax.axhline(R_BASE, ls=":", color="grey")
    ax.set_xlim(0, 120)
    ax.set_xlabel("time (h)")
    ax.set_ylabel("biomarker (units/L)")
    ax.set_title("W8D2 three doses, three different nadir times")
    ax.legend()
    plt.show()
```

**Every failure message part b can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- every entry of `t_nadir_by_dose` is the same. For inhibition of production the nadir time moves later with dose, so check that the dose is actually being passed through to conc().
- `t_nadir_by_dose` does not hold the nadir times for 25, 100 and 400 mg in that order.
- `nadir_by_dose` does not hold the nadir levels at the three doses.
- `t_above_by_dose` is not the time spent above IC50 at each dose. Count the grid points where conc(t_grid, dose) exceeds IC50 and multiply by DT.
- `t_link_by_dose` is not the time of the minimum of the effect compartment model at each dose.
- `t_link_by_dose` should be identical at all three doses. Scaling the dose scales Ce everywhere and cannot move the position of its peak, so if yours moves, something else is varying with dose.
- `%s` should be an array of three numbers.
- `%s` has shape %s. It should hold one value per dose level, in the order of DOSES.
- `t_above_by_dose` is a count of grid points, not a duration. Multiply by DT.
- `t_link_by_dose` matches the indirect model times, so both columns came from the same model. The effect compartment drives the same inhibitory term with Ce from dCe/dt = KE0_COMPARE * (C(t) - Ce).

## Part c. Duration of action belongs to the biomarker

Back to the 100 mg dose and the profile from part a.

- `auec`, the area between the baseline and the curve, in units/L times hours:
  integrate `R_BASE - r_single` over `t_grid`. This is the cumulative
  suppression, and it is the quantity that usually correlates with clinical
  benefit, far better than the nadir does,
- `t_recover`, the first time at or after the nadir at which the biomarker has
  come back to at least 90 percent of baseline.

Then change the biomarker and leave the drug alone. Re run the simulation with
the turnover half life set to 6 hours and then to 24 hours, recomputing `kout`
and `kin` each time, and record the nadir times in `t_nadir_fast` and
`t_nadir_slow`.

What this shows:

- recovery to 90 percent of baseline takes 54 hours, while the plasma
  concentration at 48 hours is 0.01 mg/L, one fiftieth of the `IC50`. For more
  than a day the biomarker is suppressed by a drug that has effectively gone,
- halving the turnover half life to 6 hours moves the nadir from 18.8 h to
  14.2 h and makes it deeper, 51.4 against 63.6. Doubling it to 24 hours moves
  the nadir out to 23.8 h and makes it shallower, 75.5. The drug, its dose and
  its PK were identical in all three.

That is the message of the day. For an indirect response, the time course of
the effect is a property of the biomarker, and the dose only sets how hard you
push on it. It is also why dosing intervals for these drugs are often set by
the biomarker's turnover and not by the drug's half life, and why a daily dose
can be right for a drug with a four hour plasma half life.

Fill in `auec`, `t_recover`, `t_nadir_fast`, `t_nadir_slow`.

Moves: trapezoidal integration, `np.where` for the first crossing,
re running `solve_ivp` with a different `kout`.

**Variables this part is graded on:** `auec`, `t_recover`, `t_nadir_fast`, `t_nadir_slow`

The starter cell they see:

```python
# Part c. Duration of action, and whose clock it runs on.

auec = None       # YOUR CODE: area between baseline and the 100 mg profile
t_recover = None  # YOUR CODE: first time at or after the nadir with R >= 90 pct of baseline


def simulate(dose, t_half_turnover):
    """Solve the turnover model for a given dose and turnover half life."""
    kout = np.log(2.0) / t_half_turnover
    kin_local = kout * R_BASE
    # YOUR CODE: solve dR/dt = kin_local * (1 - IMAX * c / (IC50 + c)) - kout * R
    # from R_BASE over t_grid, and return the solution array
    return None


t_nadir_fast = None  # YOUR CODE: nadir time with a 6 h turnover half life
t_nadir_slow = None  # YOUR CODE: nadir time with a 24 h turnover half life

print("AUEC of suppression: %r units/L*h" % (auec,))
print("back to 90 pct of baseline at %r h" % (t_recover,))
print("")
print("nadir time with a  6 h turnover half life: %r h" % (t_nadir_fast,))
print("nadir time with a 12 h turnover half life: %r h" % (t_nadir,))
print("nadir time with a 24 h turnover half life: %r h" % (t_nadir_slow,))
```

**Every failure message part c can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `auec` is the area under the biomarker curve itself. The task asks for the area between the baseline and the curve, so integrate R_BASE - r_single.
- `auec` has the right size and the wrong sign. Integrate R_BASE - r_single, so that suppression counts as positive.
- `auec` is not the area between the baseline and the 100 mg profile over the whole grid.
- `t_recover` is not the first grid time at or after the nadir where the biomarker is back to at least 90 percent of baseline.
- `t_nadir_fast` equals the 12 h turnover answer, so the new half life did not reach the solver. Recompute both kout and kin inside the function.
- `t_nadir_fast` is not the nadir time with a 6 hour turnover half life. A faster turnover means an earlier and deeper nadir.
- `t_nadir_slow` is not the nadir time with a 24 hour turnover half life, which should be later than both of the others.
- `%s` should be a single number.
- `t_recover` is earlier than the nadir, so the search started at time zero, where the biomarker is still at baseline. Search from the nadir onward.

**Functions this problem introduces:** indirect response model, turnover half life, baseline pins kin, AUEC, np.where first index

---

# W8D3. A virtual population: simulating variability and reading it back

- topic: Population PK: lognormal parameters, exposure percentiles, how precisely a CV can be estimated
- notebook: problems/week08/w8d3_population_variability.ipynb
- solution: solutions/week08/w8d3_population_variability_solution.ipynb
- expected time: 25-35 min

## Scenario

Before the phase 2 dose is chosen, somebody has to answer this: if we give
100 mg to five hundred people, what range of exposures do we get, and how many
of them end up outside the range we consider acceptable.

You cannot answer it from a mean. A mean exposure of 24 mg*h/L is compatible
with everybody getting 24 and with half the population getting 15 and the
other half 40. The answer has to be a distribution, and simulating one is the
standard tool: give every virtual subject their own clearance and volume,
compute their exposure, and look at the spread.

Today you build that population, read the percentiles off it, and then ask a
question people usually skip: how well do you actually know the variability you
just simulated. The answer, with the sample sizes early clinical pharmacology
actually has, is sobering.

## Lognormal parameters, and three numbers that are not the same

**Why lognormal.** Clearance cannot be negative, and its distribution across
people is right skewed: a few people clear a drug much faster than typical,
nobody clears it at minus two litres an hour. The standard model is therefore

```
CL_i = CL_pop * exp(eta_i),    eta_i ~ Normal(0, omega^2)
```

`CL_pop` is the typical value, `eta_i` is that subject's deviation on the log
scale, and `omega` is the between subject standard deviation of the logs. In
NONMEM output `omega` squared is what you see, and people quote its square root
as a CV.

**Three different centres.** For a lognormal distribution:

- the **median** is `CL_pop`, since half the `eta` values are below zero,
- the **geometric mean** is also `CL_pop`, which is why PK parameters are
  reported as geometric means,
- the **arithmetic mean** is `CL_pop * exp(omega^2 / 2)`, which is larger.

With `omega = 0.3` the arithmetic mean is 4.6 percent above the typical value.
With `omega = 0.6` it is 20 percent above. This is a real source of confusion
in reports: the mean clearance and the typical clearance are different numbers,
and which one somebody meant is often not stated.

**CV as a percentage.** Two conventions exist and both are in use. The exact
one is

```
CV = sqrt(exp(omega^2) - 1)
```

and the approximation, good to a few percent for small variability, is `CV` is
about `omega`. For `omega = 0.3` the exact answer is 30.69 percent and the
approximation says 30, so it rarely matters, but know which one a number came
from before comparing two reports.

**Reading a simulated population.** Percentiles, not standard deviations.
`np.percentile(auc, [5, 95])` gives you the central ninety percent of the
exposures directly, which is the interval a safety discussion actually wants.
A standard deviation of a skewed distribution tells you much less.

**How well do you know omega.** This is the part that gets skipped. The sample
variance of normally distributed logs follows a chi squared distribution with
`n - 1` degrees of freedom, which is a wide distribution for small `n`. With
twelve subjects, a true CV of 31 percent produces estimates between about 18
and 44 percent, ninety five times out of a hundred. With a hundred and twenty
subjects the same range narrows to about 27 to 35 percent.

So a phase 1 study reporting "between subject variability in clearance was
22 percent" has said something considerably vaguer than it sounds.

## Your task

Three parts, worked one at a time.

| Part | What you work out | Variables to fill in |
|---|---|---|
| a | the population, and its three different centres | `cl_gm`, `cl_cv_pct`, `cl_mean`, `mean_over_median`, `mean_over_median_theory` |
| b | the exposure distribution | `auc_i`, `cmax_i`, `auc_gm`, `auc_p5`, `auc_p95`, `fold_range` |
| c | how precisely variability can be estimated | `cv_hat_12`, `cv_hat_120`, `cv12_lo`, `cv12_hi`, `cv120_lo`, `cv120_hi` |

Below this you will find the **Setup** cell, then a **Toolkit** cell that
demonstrates every move on a tiny made up example, then **Part a**, **Part b**
and **Part c**, each with its own explanation and a cell to write in.

## The setup cell the student runs

This is given to them, already working. The data is synthetic and
seeded, so their numbers are always these numbers.

```python
import numpy as np
import matplotlib.pyplot as plt

PROBLEM_ID = "W8D3"
TOPIC = "Population PK: lognormal parameters, exposure percentiles, how precisely a CV can be estimated"

N_SUBJ = 500
DOSE = 100.0          # mg, single oral dose

CL_POP = 4.2          # L/h, typical clearance
V_POP = 35.0          # L, typical volume
KA = 1.0              # per h, absorption, taken as the same in everybody

OMEGA_CL = 0.30       # between subject SD of log clearance
OMEGA_V = 0.25        # between subject SD of log volume

# The virtual population, drawn once here so that everyone works with the
# same five hundred subjects.
_rng = np.random.default_rng(8301)
cl_i = CL_POP * _rng.lognormal(0.0, OMEGA_CL, size=N_SUBJ)
v_i = V_POP * _rng.lognormal(0.0, OMEGA_V, size=N_SUBJ)

# For part c.
SEED_C = 8302
N_STUDIES = 2000

print("%d subjects drawn" % N_SUBJ)
print("clearance: first five %s L/h" % np.round(cl_i[:5], 4))
print("volume   : first five %s L" % np.round(v_i[:5], 4))
print("")
print("typical values: CL %.1f L/h, V %.1f L, ka %.1f per h" % (CL_POP, V_POP, KA))
print("omega: %.2f on log CL, %.2f on log V" % (OMEGA_CL, OMEGA_V))
print("")
print("elimination rate constants run from %.4f to %.4f per h"
      % ((cl_i / v_i).min(), (cl_i / v_i).max()))
print("all well below ka = %.1f, so nobody in this population is flip flop" % KA)
```

## Toolkit for this problem

| What you want to do | How |
|---|---|
| lognormal draws | `rng.lognormal(mean_of_logs, sd_of_logs, size=n)` |
| a matrix of draws | `rng.lognormal(0.0, omega, size=(n_studies, n_per_study))` |
| geometric mean | `np.exp(np.mean(np.log(x)))` |
| CV from logs | `np.sqrt(np.exp(np.var(np.log(x), ddof=1)) - 1)` |
| percentiles | `np.percentile(x, [5, 95])` |
| row wise variance | `np.var(logs, axis=1, ddof=1)` |
| a histogram | `ax.hist(x, bins=40)` |

Note what `rng.lognormal(0.0, 0.3, n)` means: the **logs** are Normal with mean
0 and SD 0.3, so the values themselves have median `exp(0) = 1`. Multiply by the
typical value to centre them where you want. Passing the typical value as the
first argument instead is a common and silent error, because it is interpreted
as a mean of logs.

The toolkit cell demonstrates each of those on a tiny made up example.
It is in the notebook and none of it is the answer.

## Part a. Read the population back

The five hundred clearances are in `cl_i`. Summarise them.

- `cl_gm`, the geometric mean,
- `cl_cv_pct`, the between subject CV as a percentage, from the exact formula
  `100 * sqrt(exp(var(log CL)) - 1)` with `ddof=1`,
- `cl_mean`, the plain arithmetic mean,
- `mean_over_median`, the arithmetic mean divided by the median, as observed,
- `mean_over_median_theory`, what that ratio should be in the limit,
  `exp(OMEGA_CL ** 2 / 2)`.

The geometric mean comes out at 4.198 L/h against a typical value of 4.2, and
the CV at 30.9 percent against the 30.69 the omega implies. Five hundred
subjects is enough to recover both closely, which is worth seeing once so that
part c lands properly.

The two ratios are 1.041 observed and 1.046 in theory. Both say the same thing:
the arithmetic mean clearance is about four percent higher than the typical
clearance, because the distribution has a long right tail and the mean follows
it. If a report gives you a mean clearance and a model gives you a typical
clearance, they are not the same quantity and the difference grows fast with
variability.

Fill in `cl_gm`, `cl_cv_pct`, `cl_mean`, `mean_over_median`,
`mean_over_median_theory`.

Moves: `np.log`, `np.exp`, `np.mean`, `np.median`, `np.var(..., ddof=1)`.

**Variables this part is graded on:** `cl_gm`, `cl_cv_pct`, `cl_mean`, `mean_over_median`, `mean_over_median_theory`

The starter cell they see:

```python
# Part a. Three different centres of one distribution.

cl_gm = None                     # YOUR CODE: geometric mean of cl_i
cl_cv_pct = None                 # YOUR CODE: exact CV as a percentage
cl_mean = None                   # YOUR CODE: arithmetic mean
mean_over_median = None          # YOUR CODE: arithmetic mean / median, observed
mean_over_median_theory = None   # YOUR CODE: exp(OMEGA_CL ** 2 / 2)

print("typical clearance in the simulation: %.1f L/h" % CL_POP)
print("geometric mean  : %r L/h" % (cl_gm,))
print("arithmetic mean : %r L/h" % (cl_mean,))
print("CV              : %r pct   (omega %.2f implies %.2f pct)"
      % (cl_cv_pct, OMEGA_CL, 100 * np.sqrt(np.exp(OMEGA_CL ** 2) - 1)))
print("mean / median   : %r observed, %r in theory"
      % (mean_over_median, mean_over_median_theory))
```

**Every failure message part a can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `cl_gm` is the arithmetic mean. The geometric mean is exp(mean(log(cl_i))), and it is slightly smaller.
- `cl_gm` is not the geometric mean of cl_i.
- `cl_cv_pct` is a fraction. The task asks for a percentage.
- `cl_cv_pct` is 100 times the SD of the logs, the approximation. The task asks for the exact form, 100 * sqrt(exp(var(log CL)) - 1).
- `cl_cv_pct` is not the exact CV as a percentage. Use np.var(np.log(cl_i), ddof=1).
- `cl_mean` is not the arithmetic mean of cl_i.
- `mean_over_median` is the theoretical ratio. This one should be computed from the simulated numbers, mean divided by median, and it comes out a little different.
- `mean_over_median` is not the observed arithmetic mean divided by the median.
- `mean_over_median_theory` is missing the division by 2 in the exponent. The arithmetic mean of a lognormal is the median times exp(omega ** 2 / 2).
- `mean_over_median_theory` should be exp(OMEGA_CL ** 2 / 2).
- `%s` should be a single number.

## Part b. The exposure distribution

For a one compartment model with first order absorption, every subject's
exposure follows from their own parameters without solving anything:

```
AUC(0 to infinity) = DOSE / CL
k = CL / V
Tmax = ln(ka / k) / (ka - k)
Cmax = DOSE * ka / (V * (ka - k)) * (exp(-k * Tmax) - exp(-ka * Tmax))
```

The AUC result is the one worth remembering: for first order elimination the
total exposure depends only on dose and clearance. Not on volume, not on
absorption rate. Volume and `ka` decide the shape of the curve, clearance
decides its area.

Compute `auc_i` and `cmax_i`, both arrays of five hundred, then summarise the
exposures:

- `auc_gm`, the geometric mean AUC,
- `auc_p5` and `auc_p95`, the 5th and 95th percentiles,
- `fold_range`, the 95th divided by the 5th.

What comes out:

- a geometric mean AUC of 23.82 mg*h/L, which is `DOSE / CL_POP` to three
  figures, as it must be since the AUC is an exact function of clearance,
- a central ninety percent interval of 14.9 to 40.5 mg*h/L, a **2.72 fold
  range**. That is the number a safety argument has to survive. The person at
  the 95th percentile gets nearly three times the exposure of the person at the
  5th, from the same tablet,
- a Cmax distribution that is visibly tighter than the AUC one, roughly two
  fold against nearly three. Cmax depends on clearance only through `k`, which
  is a ratio of two correlated quantities, so some of the variability cancels.

The cell plots both distributions. Note that they are skewed, with a tail to
the right, which is what the lognormal parameters guarantee and the reason the
interval is not symmetric about the middle.

Fill in `auc_i`, `cmax_i`, `auc_gm`, `auc_p5`, `auc_p95`, `fold_range`.

Moves: elementwise arithmetic on arrays, `np.log`, `np.exp`,
`np.percentile`.

**Variables this part is graded on:** `k_i`, `auc_i`, `tmax_i`, `cmax_i`, `auc_gm`, `auc_p5`, `auc_p95`, `fold_range`

The starter cell they see:

```python
# Part b. From parameters to exposures.

k_i = None       # YOUR CODE: elimination rate constant for each subject
auc_i = None     # YOUR CODE: AUC from 0 to infinity for each subject
tmax_i = None    # YOUR CODE: time of the peak for each subject
cmax_i = None    # YOUR CODE: peak concentration for each subject

auc_gm = None    # YOUR CODE: geometric mean AUC
auc_p5 = None    # YOUR CODE: 5th percentile of AUC
auc_p95 = None   # YOUR CODE: 95th percentile of AUC
fold_range = None  # YOUR CODE: auc_p95 / auc_p5

print("AUC  geometric mean %r mg*h/L   (DOSE / CL_POP = %.4f)" % (auc_gm, DOSE / CL_POP))
print("AUC  5th to 95th percentile: %r to %r mg*h/L" % (auc_p5, auc_p95))
print("AUC  fold range: %r" % (fold_range,))
if cmax_i is not None:
    _p = np.percentile(cmax_i, [5, 95])
    print("Cmax 5th to 95th percentile: %.4f to %.4f mg/L, fold range %.4f"
          % (_p[0], _p[1], _p[1] / _p[0]))

if auc_i is not None and cmax_i is not None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(auc_i, bins=40, color="#1f4e79")
    axes[0].set_xlabel("AUC (mg*h/L)")
    axes[0].set_ylabel("subjects")
    axes[1].hist(cmax_i, bins=40, color="#7f3f00")
    axes[1].set_xlabel("Cmax (mg/L)")
    for _ax, _x in zip(axes, (auc_i, cmax_i)):
        for _q in np.percentile(_x, [5, 95]):
            _ax.axvline(_q, ls=":", color="black")
    fig.suptitle("W8D3 exposure distribution in 500 virtual subjects")
    plt.show()
```

**Every failure message part b can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `auc_i` is not DOSE / cl_i. For first order elimination the total exposure depends on dose and clearance only.
- `cmax_i` holds the peak TIMES. It should hold the peak concentrations.
- `cmax_i` is not the peak concentration for each subject. Work out Tmax first, then evaluate the concentration equation at it.
- `auc_gm` is the arithmetic mean AUC. The geometric mean is what matches DOSE / CL_POP.
- `auc_gm` is not the geometric mean of auc_i.
- `auc_p5` and `auc_p95` are the wrong way round.
- `auc_p5` is not the 5th percentile of auc_i.
- `auc_p95` is not the 95th percentile of auc_i.
- `fold_range` is upside down. It is the 95th divided by the 5th.
- `fold_range` is not auc_p95 divided by auc_p5.
- `%s` should be an array of numbers.
- `%s` has shape %s but there are %d subjects.
- `%s` should be a single number.

## Part c. How well do you know the variability

You simulated with `OMEGA_CL = 0.30`, so the true CV is 30.69 percent. Now ask
what a study would have told you.

Simulate `N_STUDIES` studies of twelve subjects each, and `N_STUDIES` studies of
a hundred and twenty subjects each, drawing from the same lognormal. Estimate
the CV within each study with the same exact formula you used in part a, and
look at how much those estimates scatter.

Do it like this, in this order, so that your numbers match the ones quoted
below:

```
rng_c = np.random.default_rng(SEED_C)
draws_12 = rng_c.lognormal(0.0, OMEGA_CL, size=(N_STUDIES, 12))
draws_120 = rng_c.lognormal(0.0, OMEGA_CL, size=(N_STUDIES, 120))
```

Two draws, each one call, each a matrix with one row per study. Then take logs,
compute the variance **along axis 1** with `ddof=1`, and convert to a CV
percentage. That gives `cv_hat_12` and `cv_hat_120`, each with `N_STUDIES`
entries.

Record the 2.5th and 97.5th percentiles of each: `cv12_lo`, `cv12_hi`,
`cv120_lo`, `cv120_hi`.

What you get, against a truth of 30.69 percent:

| study size | 2.5th percentile | 97.5th percentile |
|---|---|---|
| 12 subjects | about 17 pct | about 44 pct |
| 120 subjects | about 27 pct | about 35 pct |

With twelve subjects, the estimate is as likely as not to be wrong by a third
in either direction. The interval is 27 percentage points wide, more than three
times the width you get from a hundred and twenty, and the median estimate is
also slightly low, 29.4 against 30.69, because the variance of the logs is
unbiased but its square root is not.

Two things follow, both practical.

First, do not compare a between subject CV from a twelve subject phase 1 study
with one from a large phase 3 population and conclude anything about the
population having changed. The phase 1 number is barely an estimate.

Second, this is the argument for mixed effects models over the two stage
approach you used in week 6. They borrow information across subjects, so the
variability estimate uses all the data at once instead of being computed from
twelve separate numbers each with its own error.

Fill in `cv_hat_12`, `cv_hat_120`, `cv12_lo`, `cv12_hi`, `cv120_lo`,
`cv120_hi`.

Moves: one `rng.lognormal` call per study size with a 2-D `size`,
`np.var(..., axis=1, ddof=1)`, `np.percentile`.

**Variables this part is graded on:** `draws_12`, `draws_120`, `cv_hat_12`, `cv_hat_120`, `cv12_lo`, `cv12_hi`, `cv120_lo`, `cv120_hi`

The starter cell they see:

```python
# Part c. The sampling distribution of a CV estimate.

rng_c = np.random.default_rng(SEED_C)

draws_12 = None   # YOUR CODE: lognormal draws, shape (N_STUDIES, 12)
draws_120 = None  # YOUR CODE: lognormal draws, shape (N_STUDIES, 120)

cv_hat_12 = None  # YOUR CODE: CV percentage estimated in each 12 subject study
cv_hat_120 = None  # YOUR CODE: the same for the 120 subject studies

cv12_lo = None    # YOUR CODE: 2.5th percentile of cv_hat_12
cv12_hi = None    # YOUR CODE: 97.5th percentile of cv_hat_12
cv120_lo = None   # YOUR CODE: 2.5th percentile of cv_hat_120
cv120_hi = None   # YOUR CODE: 97.5th percentile of cv_hat_120

print("true CV: %.2f pct" % (100 * np.sqrt(np.exp(OMEGA_CL ** 2) - 1)))
print("")
print(" 12 subjects: 95 pct of estimates between %r and %r" % (cv12_lo, cv12_hi))
print("120 subjects: 95 pct of estimates between %r and %r" % (cv120_lo, cv120_hi))
if cv_hat_12 is not None and cv_hat_120 is not None:
    print("")
    print("median estimate,  12 subjects: %.3f pct" % np.median(cv_hat_12))
    print("median estimate, 120 subjects: %.3f pct" % np.median(cv_hat_120))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(cv_hat_12, bins=50, alpha=0.6, label="studies of 12")
    ax.hist(cv_hat_120, bins=50, alpha=0.6, label="studies of 120")
    ax.axvline(100 * np.sqrt(np.exp(OMEGA_CL ** 2) - 1), color="black", ls="--",
               label="truth")
    ax.set_xlabel("estimated between subject CV (pct)")
    ax.set_ylabel("studies")
    ax.set_title("W8D3 what a study would have told you")
    ax.legend()
    plt.show()
```

**Every failure message part c can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- the median of `cv_hat_12` is %.2f, which is not where it should be. Each row of the draws is one study of twelve, so the variance is taken along axis=1 with ddof=1.
- the median of `cv_hat_120` is not where it should be.
- `cv_hat_120` has a lower median than `cv_hat_12`, which is backwards: the larger studies should centre closer to the truth, which is above both.
- the interval from the twelve subject studies should be more than twice as wide as the one from the hundred and twenty subject studies. Check that the two draws have the shapes (N_STUDIES, 12) and (N_STUDIES, 120).
- `%s` should be an array of CV estimates.
- `%s` is a single number. It should hold one estimate per simulated study, so compute the variance along axis=1.
- `%s` has shape %s but there are N_STUDIES = %d studies.
- `%s` looks like a fraction rather than a percentage. Multiply by 100.
- `%s` should be a single number.
- `%s` is not the percentile it should be. The four values are the 2.5th and 97.5th percentiles of each set of estimates.

**Functions this problem introduces:** rng.lognormal, np.percentile, geometric mean and CV, arithmetic against geometric mean, Monte Carlo study simulation

---

# W8D4. Building a NONMEM ready dataset, and auditing it

- topic: Population PK: event records, EVID and MDV, BLQ handling, per subject NCA with groupby
- notebook: problems/week08/w8d4_nonmem_dataset.ipynb
- solution: solutions/week08/w8d4_nonmem_dataset_solution.ipynb
- expected time: 25-35 min

## Scenario

The modeller needs the phase 1 data as a NONMEM dataset by Friday. What you
have is what you always have: a sample table from the bioanalytical lab, a
dosing record from the pharmacy, and a demographics table from the clinical
database, all keyed on subject number and none of them in the shape the model
needs.

This is the job. Not glamorous, entirely unavoidable, and the place where
real analyses go wrong. A dataset with the dose record sorted after the first
observation runs without complaint and produces estimates that are wrong, and
nothing in the output says so.

Today you build one, audit it, and then do the per subject NCA that the
modeller will compare their fit against.

## Event records, and the four columns that carry the logic

A NONMEM dataset is not a table of observations. It is a **time ordered list of
events** for each subject, where an event is either something given or
something measured. One row, one event.

The required columns, with the conventional names:

| Column | Meaning |
|---|---|
| `ID` | subject identifier, grouping the rows |
| `TIME` | hours since the subject's first event |
| `AMT` | amount given on a dosing row, 0 on an observation row |
| `DV` | the dependent variable, the measurement, 0 on a dosing row |
| `EVID` | event ID: 1 for a dose, 0 for an observation |
| `MDV` | missing dependent variable: 1 if there is no usable measurement |
| `CMT` | which compartment the event applies to |

**`EVID` is the switch.** `EVID = 1` means give `AMT` into compartment `CMT`
and ignore `DV`. `EVID = 0` means compare `DV` with the model's prediction.
Other values exist, 2 for other type events and 3 for a reset, and you will
meet them when you meet a study design that needs them.

**`MDV = 1` means do not use this `DV`.** Dosing rows always have `MDV = 1`,
because there is nothing to fit there. Observation rows get `MDV = 1` when the
sample is missing, lost, or below the limit of quantification.

**Sort order is part of the data.** Rows are processed in the order they appear,
so they must be sorted by `ID` then `TIME`, and **at identical times the dose
has to come first**. A pre dose sample and the dose are both at time zero, and
if the sample is processed second it will be compared against a prediction that
already includes the dose. The fix is to sort with `EVID` descending within
`TIME`:

```
df.sort_values(["ID", "TIME", "EVID"], ascending=[True, True, False])
```

**BLQ, and why it is handled twice.** A sample below the LLOQ is not a
concentration of zero and not a missing value either, it is the information
that the true value lies somewhere between zero and the LLOQ. There are
several conventions:

- **M1**, drop the record, or equivalently set `MDV = 1`. The default, and what
  you will do today,
- **M3**, keep it and treat it as censored, which is more correct and needs a
  likelihood that NONMEM can do and `curve_fit` cannot,
- substitute `LLOQ / 2`, common in NCA, defensible, biased,
- substitute zero, which is what NCA does for a pre dose sample after an oral
  dose, where zero is not a substitution but the truth.

Note the consequence: the same record can be `MDV = 1` for the model fit and
treated as a hard zero in the NCA of the same dataset. That is not an
inconsistency, it is two analyses answering different questions, but it does
need writing down in the analysis plan.

## Your task

Three parts, worked one at a time.

| Part | What you work out | Variables to fill in |
|---|---|---|
| a | assemble the event records in the right order | `nm`, `n_rows`, `n_dose_rows`, `first_evid_per_id` |
| b | audit it before anybody fits anything | `n_blq`, `n_mdv`, `records_per_id`, `time_monotone` |
| c | per subject NCA, merged with the dosing and demographics | `nca`, `auc_dn_low`, `auc_dn_high`, `auc_dn_ratio` |

Below this you will find the **Setup** cell, then a **Toolkit** cell that
demonstrates every move on a tiny made up example, then **Part a**, **Part b**
and **Part c**, each with its own explanation and a cell to write in.

## The setup cell the student runs

This is given to them, already working. The data is synthetic and
seeded, so their numbers are always these numbers.

```python
import numpy as np
import pandas as pd

PROBLEM_ID = "W8D4"
TOPIC = "Population PK: event records, EVID and MDV, BLQ handling, per subject NCA with groupby"

pd.set_option("display.width", 120)

N_SUBJ = 8
LLOQ = 0.05          # mg/L, lower limit of quantification
_KA = 1.0
_CL_POP = 4.2
_V_POP = 35.0

_doses = np.array([50.0] * 4 + [150.0] * 4)
_times = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0])

_rng = np.random.default_rng(8426)
_cl = _CL_POP * _rng.lognormal(0.0, 0.30, N_SUBJ)
_v = _V_POP * _rng.lognormal(0.0, 0.25, N_SUBJ)
_wt = np.round(70.0 * _rng.lognormal(0.0, 0.15, N_SUBJ), 1)
_age = _rng.integers(21, 65, N_SUBJ)
_sex = _rng.integers(0, 2, N_SUBJ)

_rows = []
for _i in range(N_SUBJ):
    _k = _cl[_i] / _v[_i]
    _c = (_doses[_i] * _KA / (_v[_i] * (_KA - _k))
          * (np.exp(-_k * _times) - np.exp(-_KA * _times)))
    _c = np.round(np.maximum(_c * _rng.lognormal(0.0, 0.08, _times.size), 0.0), 4)
    for _t, _cc in zip(_times, _c):
        _rows.append((_i + 1, float(_t), float(_cc)))

# Three source tables, as they would arrive.
samples = pd.DataFrame(_rows, columns=["ID", "TIME", "CONC"])
dosing = pd.DataFrame({"ID": np.arange(1, N_SUBJ + 1), "AMT": _doses})
demog = pd.DataFrame({"ID": np.arange(1, N_SUBJ + 1), "WT_KG": _wt,
                      "AGE_Y": _age, "SEX": _sex})

NM_COLS = ["ID", "TIME", "AMT", "DV", "EVID", "MDV", "CMT"]

print("samples: %d rows" % len(samples))
print(samples.head(10).to_string(index=False))
print("")
print("dosing:")
print(dosing.to_string(index=False))
print("")
print("demog:")
print(demog.to_string(index=False))
print("")
print("LLOQ = %.2f mg/L, and %d of the %d samples are below it"
      % (LLOQ, int((samples["CONC"] < LLOQ).sum()), len(samples)))
print("every dose is given at time 0, and every subject has a pre dose sample")
```

## Toolkit for this problem

| What you want to do | How |
|---|---|
| add constant columns | `df.assign(EVID=1, MDV=1)` |
| a column from a condition | `np.where(df["CONC"] < LLOQ, 1, 0)` |
| stack two tables | `pd.concat([a, b], ignore_index=True)` |
| choose and order columns | `df[NM_COLS]` |
| sort, with one column descending | `df.sort_values([...], ascending=[True, True, False])` |
| renumber the index afterwards | `.reset_index(drop=True)` |
| rows per group | `df.groupby("ID").size()` |
| the first row of each group | `df.groupby("ID").first()` |
| is a series sorted | `s.is_monotonic_increasing` |
| a per group calculation returning several numbers | `df.groupby("ID")[["TIME", "DV"]].apply(fn)` |
| join a lookup table | `left.merge(right, on="ID", validate="many_to_one")` |

The `groupby("ID")[["TIME", "DV"]].apply(fn)` form is the one to use. Selecting
the columns you need before `apply` keeps the group key out of the sub frame,
which is what newer pandas wants and older pandas tolerates, so the same line
works everywhere.

The toolkit cell demonstrates each of those on a tiny made up example.
It is in the notebook and none of it is the answer.

## Part a. Assemble the event records

Build `nm`, the NONMEM dataset, from `samples` and `dosing`.

The dosing rows: one per subject, `TIME` 0, `AMT` from `dosing`, `DV` 0,
`EVID` 1, `MDV` 1, `CMT` 1.

The observation rows: one per sample, `AMT` 0, `DV` the concentration, `EVID`
0, `CMT` 2, and `MDV` 1 where the concentration is below `LLOQ` and 0
otherwise.

Then concatenate them, select the columns in the order `NM_COLS`, and sort by
`ID`, `TIME`, and `EVID` **descending**, so that the dose comes before the pre
dose sample at time zero. Reset the index.

Record `n_rows`, the total number of rows, `n_dose_rows`, how many have
`EVID` of 1, and `first_evid_per_id`, the `EVID` of the first row of each
subject as an array of eight numbers. That last one is the check that the sort
worked: every entry must be 1. If any subject starts with a 0, that subject's
first observation is being compared against a model that has not been dosed
yet.

Print the first eleven rows when you have it. You should see the dose for
subject 1, then the pre dose sample with `MDV` of 1, then seven usable
observations, then subject 2's dose.

Fill in `nm`, `n_rows`, `n_dose_rows`, `first_evid_per_id`.

Moves: `assign`, `np.where`, `pd.concat`, column selection, `sort_values`
with mixed `ascending`, `reset_index`, `groupby(...).first()`.

**Variables this part is graded on:** `dose_rows`, `obs_rows`, `nm`, `n_rows`, `n_dose_rows`, `first_evid_per_id`

The starter cell they see:

```python
# Part a. Build the dataset.

dose_rows = None  # YOUR CODE: one row per subject, EVID 1, from `dosing`
obs_rows = None   # YOUR CODE: one row per sample, EVID 0, from `samples`

nm = None         # YOUR CODE: concat, select NM_COLS, sort, reset_index

n_rows = None             # YOUR CODE: total rows
n_dose_rows = None        # YOUR CODE: rows with EVID == 1
first_evid_per_id = None  # YOUR CODE: EVID of the first row of each subject

print("rows: %r, of which dosing records: %r" % (n_rows, n_dose_rows))
print("first EVID per subject:", first_evid_per_id)
if nm is not None:
    print("")
    print(nm.head(11).to_string(index=False))
```

**Every failure message part a can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `nm` should be a pandas DataFrame.
- `nm` has the right columns in the wrong order: %s. Select them as NM_COLS so the file reads the way a modeller expects.
- `nm` has %d rows but it should have %d: one dosing record per subject plus every sample.
- `n_rows` should be the number of rows in `nm`, which is %d.
- `n_dose_rows` should be the number of rows with EVID == 1, which is one per subject.
- `first_evid_per_id` should be an array of numbers.
- `first_evid_per_id` has shape %s but there are %d subjects. Use groupby('ID')['EVID'].first().
- every subject starts with an observation, so the sort left the pre dose sample ahead of the dose. Sort with EVID descending inside TIME, which means ascending=[True, True, False].
- `first_evid_per_id` should be 1 for every subject. The first event in a subject's record has to be the dose.
- the first eleven rows of `nm` do not have the expected EVID pattern. Subject 1 should be a dose, then eight observations, then subject 2's dose.
- the DV column of the first rows is not right. Dosing rows carry DV 0 and observation rows carry the measured concentration.
- `nm` has a dosing row with AMT of 0. The dose amount belongs on that row.
- `nm` has an observation row with a non zero AMT. Observations carry AMT 0.
- `nm` is missing the column(s) %s. The dataset needs exactly %s.
- could not read the first rows of `nm`: %s
- could not read AMT from `nm`: %s

## Part b. Audit it

Four checks, each one line, each one catching a real class of error.

- `n_blq`, how many observation rows are below the LLOQ. Count rows with
  `EVID` of 0 and `MDV` of 1,
- `n_mdv`, how many rows in total have `MDV` of 1. This should be `n_blq` plus
  the eight dosing rows,
- `records_per_id`, the number of rows for each subject, as an array of eight.
  All eight should be 9: one dose and eight samples,
- `time_monotone`, True if `TIME` is non decreasing within every subject.

You should find ten BLQ records and eighteen `MDV` rows. Eight of the ten are
the pre dose samples, which are below the limit because the drug has not been
given yet. The other two are subjects 3 and 4 at 24 hours, who cleared the
50 mg dose fast enough to fall under the limit by the last sample. Those two
are the ones to think about: a genuine measurement attempt that came back with
no number, which is where the choice of BLQ method starts to matter.

The counts are not busy work. A subject with eight rows instead of nine has
lost a sample somewhere in the merge. A subject whose rows are not time ordered
will produce quiet nonsense. Both have happened in real submissions.

Fill in `n_blq`, `n_mdv`, `records_per_id`, `time_monotone`.

Moves: boolean masks and `.sum()`, `groupby(...).size()`,
`is_monotonic_increasing`.

**Variables this part is graded on:** `n_blq`, `n_mdv`, `records_per_id`, `time_monotone`

The starter cell they see:

```python
# Part b. Audit the dataset.

n_blq = None           # YOUR CODE: observation rows below the LLOQ
n_mdv = None           # YOUR CODE: all rows with MDV == 1
records_per_id = None  # YOUR CODE: rows per subject, as an array of 8
time_monotone = None   # YOUR CODE: True if TIME never decreases within a subject

print("observation rows below the LLOQ: %r" % (n_blq,))
print("rows with MDV = 1             : %r" % (n_mdv,))
print("rows per subject              :", records_per_id)
print("TIME non decreasing in every subject:", time_monotone)
if nm is not None:
    print("")
    print("the BLQ observation rows:")
    print(nm[(nm["EVID"] == 0) & (nm["MDV"] == 1)].to_string(index=False))
```

**Every failure message part b can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `n_blq` should be a single number.
- `n_blq` counted every MDV row, including the dosing records. Restrict it to rows with EVID == 0.
- `n_blq` should be the number of observation rows with MDV == 1, which is the samples below the LLOQ.
- `n_mdv` should be a single number.
- `n_mdv` should count the dosing rows too: they always carry MDV == 1, because there is no measurement on them.
- `n_mdv` should be the total number of rows with MDV == 1.
- `records_per_id` should be an array of counts.
- `records_per_id` has shape %s but there are %d subjects.
- `records_per_id` is not the number of rows per subject. Every subject should have the same count here, one dose plus every sample.
- `time_monotone` should be True or False.
- `time_monotone` should be True for a correctly sorted dataset. If your check says otherwise, it is probably testing the whole TIME column rather than each subject's separately.

## Part c. Per subject NCA, and a dose proportionality trap

Now produce the NCA summary the modeller will compare their fit against.

Work from the observation rows only, `EVID` of 0, and treat the `DV` column as
it stands, so BLQ records count as zero. For a pre dose sample after an oral
dose that is exactly right. For the two 24 hour samples it is the zero
substitution convention, which here costs almost nothing because the values
were under 0.03 mg/L, and which you would record in the analysis plan.

Write a function that takes a sub frame of `TIME` and `DV` for one subject and
returns a Series with `CMAX`, `TMAX` and `AUC`, the last by the linear
trapezoid rule over the samples as they are. Apply it with
`groupby("ID")[["TIME", "DV"]].apply(...)`, reset the index, then merge on the
dosing table and the demographics, using `validate="one_to_one"` so that a
duplicated key would raise rather than silently duplicate rows.

Add a dose normalised column, `AUC` divided by `AMT`, and call the result
`nca`. Then compare the two dose groups:

- `auc_dn_low`, the geometric mean dose normalised AUC in the 50 mg group,
- `auc_dn_high`, the same in the 150 mg group,
- `auc_dn_ratio`, high divided by low.

The ratio comes out at 1.20. Taken at face value that says exposure rises
faster than dose, by twenty percent over a three fold dose range, which is the
kind of finding that gets a slide of its own and a discussion about saturable
first pass metabolism.

It is an artifact, and here you can prove it, because the data were simulated
with exposure exactly proportional to dose in every subject: `AUC` to infinity
is `DOSE / CL` by construction. What differs between the groups is the people.
The four subjects who got 50 mg have a geometric mean clearance of 4.71 L/h and
the four who got 150 mg have 3.66 L/h, 29 percent lower, because with four
subjects per group that is well within the luck of the draw. The dose had
nothing to do with it.

This is why dose proportionality is assessed within subjects, in a crossover,
whenever it can be, and when it cannot, with a power model on the log scale and
a confidence interval wide enough to be honest about eight people.

Fill in `nca`, `auc_dn_low`, `auc_dn_high`, `auc_dn_ratio`.

Moves: `groupby(...)[[cols]].apply`, `reset_index`, `merge` with `validate`,
geometric means.

**Variables this part is graded on:** `nca`, `auc_dn_low`, `auc_dn_high`, `auc_dn_ratio`

The starter cell they see:

```python
# Part c. NCA per subject, and the dose groups compared.

def nca_one(g):
    """CMAX, TMAX and linear trapezoid AUC for one subject."""
    g = g.sort_values("TIME")
    # YOUR CODE: pull out TIME and DV, find the peak, integrate
    return None


nca = None  # YOUR CODE: the per subject table, merged with dosing and demog,
            #            with an AUC_DN column

auc_dn_low = None    # YOUR CODE: geometric mean AUC_DN in the 50 mg group
auc_dn_high = None   # YOUR CODE: geometric mean AUC_DN in the 150 mg group
auc_dn_ratio = None  # YOUR CODE: high divided by low

if nca is not None:
    print(nca.to_string(index=False))
print("")
print("dose normalised AUC, geometric mean")
print("  50 mg group : %r" % (auc_dn_low,))
print(" 150 mg group : %r" % (auc_dn_high,))
print("  ratio       : %r" % (auc_dn_ratio,))
```

**Every failure message part c can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `nca` should be a pandas DataFrame.
- `nca` has %d rows but there are %d subjects. If it has more, a merge duplicated rows, which validate='one_to_one' would have caught.
- `nca` is missing the demographic columns. Merge `demog` on as well.
- the CMAX column is not the highest DV for each subject. Work from the observation rows only.
- the TMAX column is not the time of each subject's highest DV. Note that it is the TIME at the peak, not the position of the peak.
- the AUC column is not the linear trapezoid area over each subject's observations. Sort by TIME inside the group first, and integrate DV as it stands, so that BLQ records count as zero.
- the AUC_DN column should be AUC divided by AMT.
- `auc_dn_low` is the arithmetic mean. Exposure summaries use the geometric mean, exp(mean(log(x))).
- `auc_dn_low` and `auc_dn_high` are the wrong way round.
- `auc_dn_low` is not the geometric mean dose normalised AUC in the 50 mg group.
- `auc_dn_high` is not the geometric mean dose normalised AUC in the 150 mg group.
- `auc_dn_ratio` is upside down. The task asks for high divided by low.
- `auc_dn_ratio` is not auc_dn_high divided by auc_dn_low.
- `nca` has no %s column. It should carry ID, CMAX, TMAX, AUC, the merged dosing and demographic columns, and AUC_DN.
- could not read the NCA columns: %s
- `%s` should be a single number.

**Functions this problem introduces:** NONMEM event records, EVID and MDV, BLQ handling, sort_values mixed ascending, groupby apply several outputs

---

# W8D5. Capstone: choosing the phase 2 dose

- topic: PK/PD: steady state exposure, probability of target attainment, joint efficacy and safety
- notebook: problems/week08/w8d5_dose_selection.ipynb
- solution: solutions/week08/w8d5_dose_selection_solution.ipynb
- expected time: 30-40 min

## Scenario

The phase 2 dose has to be chosen this month, from five candidates given once
daily: 25, 50, 100, 150 and 200 mg. You have a PK model, a PD model, and
estimates of how much both vary between people. What you do not have is any
phase 2 data, which is the point: the decision has to be made before the study
that would settle it.

The efficacy criterion, agreed with the clinical team, is that average
inhibition of the target over the dosing interval should be at least 55
percent. The safety criterion, from the thorough QT study and the phase 1
tolerability data, is that the steady state peak concentration should stay at
or below 3 mg/L.

A patient counts as a success only if both hold. Your job is to work out, for
each dose, what fraction of patients that is, and to recommend one.

This is the last problem of the Python phase, and it uses most of it: a
population drawn from lognormal distributions, a closed form steady state PK
model evaluated by broadcasting, an Emax model on top, numerical integration
over the dosing interval, and a decision read off the numbers at the end.

## Steady state, and the two things that decide a dose

**The steady state profile, in closed form.** For a one compartment model with
first order absorption, dosing every `tau` hours, the concentration during a
dosing interval at steady state is

```
C_ss(t) = (DOSE * ka) / (V * (ka - k)) * ( exp(-k t) / (1 - exp(-k tau))
                                         - exp(-ka t) / (1 - exp(-ka tau)) )
```

for `t` from 0 to `tau`. It is the single dose equation with each exponential
divided by one minus its own accumulation factor, which is what summing an
infinite series of doses gives you. No solver needed, so you can evaluate it
for four hundred subjects at five doses in one expression.

**Average concentration is simpler than it looks.** Integrate that expression
over one interval and divide by `tau` and almost everything cancels:

```
C_avg,ss = DOSE / (CL * tau)
```

Clearance and the dosing interval, nothing else. Not volume, not absorption
rate. Those two decide how far the concentration swings above and below
`C_avg`, which is a different question, and often the more important one, since
the peak drives tolerability and the trough drives loss of effect.

**Probability of target attainment.** With a population rather than one
typical subject, the output of a simulation is not a number, it is a fraction:
what proportion of subjects meet the criterion. Doing this for both an efficacy
and a safety criterion, and requiring both at once, is standard dose selection
practice.

The thing to watch for is that **the two criteria move in opposite directions
with dose**, so the joint probability has a maximum somewhere in the middle and
the best dose is almost never the one that maximises efficacy. A dose that
treats ninety percent of patients effectively and poisons seventy percent of
them is not a better dose than one that treats seventy two percent and poisons
nine.

**Why the safety curve turns so sharply.** Peak concentration is proportional
to dose, so the whole distribution of `Cmax` slides up as the dose rises. When
the safety limit sits between the median `Cmax` at one dose and the median at
the next, a single dose step flips most of the population from one side of it
to the other. That is why dose steps near the top of the range are small, and
why the interesting question is never the mean exposure, it is where the limit
falls in the distribution.

## Your task

Three parts, worked one at a time.

| Part | What you work out | Variables to fill in |
|---|---|---|
| a | the steady state profile for a typical subject | `c_typ`, `cmax_ss`, `cmin_ss`, `cavg_ss`, `cavg_identity` |
| b | the population at the middle dose | `cmax_i`, `mean_inh_i`, `p_eff`, `p_tox`, `p_success` |
| c | all five doses, and a recommendation | `p_eff_by_dose`, `p_tox_by_dose`, `p_success_by_dose`, `best_dose` |

Below this you will find the **Setup** cell, then a **Toolkit** cell that
demonstrates every move on a tiny made up example, then **Part a**, **Part b**
and **Part c**, each with its own explanation and a cell to write in.

## The setup cell the student runs

This is given to them, already working. The data is synthetic and
seeded, so their numbers are always these numbers.

```python
import numpy as np
import matplotlib.pyplot as plt

PROBLEM_ID = "W8D5"
TOPIC = "PK/PD: steady state exposure, probability of target attainment, joint efficacy and safety"

N_SUBJ = 400
TAU = 24.0            # h, once daily dosing
KA = 1.0              # per h, absorption, the same in everybody

# Typical values and between subject variability.
CL_POP = 4.2          # L/h
V_POP = 35.0          # L
IC50_POP = 0.4        # mg/L
IMAX = 1.0            # full inhibition is achievable at high enough concentration

OMEGA_CL = 0.30
OMEGA_V = 0.25
OMEGA_IC50 = 0.40     # PD variability is usually larger than PK variability

# The decision criteria.
EFF_TARGET = 0.55     # average inhibition over the interval must reach this
CMAX_LIMIT = 3.0      # mg/L, steady state peak must not exceed this

DOSES = np.array([25.0, 50.0, 100.0, 150.0, 200.0])
DOSE_MID = 100.0

# One dosing interval, on a fixed grid.
t_grid = np.linspace(0.0, TAU, 241)

# The virtual population, drawn once so that everyone works with the same
# four hundred patients.
_rng = np.random.default_rng(8501)
cl_i = CL_POP * _rng.lognormal(0.0, OMEGA_CL, size=N_SUBJ)
v_i = V_POP * _rng.lognormal(0.0, OMEGA_V, size=N_SUBJ)
ic50_i = IC50_POP * _rng.lognormal(0.0, OMEGA_IC50, size=N_SUBJ)

print("%d patients, dosed every %g h" % (N_SUBJ, TAU))
print("clearance   : geometric mean %.4f L/h" % np.exp(np.mean(np.log(cl_i))))
print("volume      : geometric mean %.4f L" % np.exp(np.mean(np.log(v_i))))
print("IC50        : geometric mean %.4f mg/L" % np.exp(np.mean(np.log(ic50_i))))
print("")
print("efficacy criterion: average inhibition over the interval at least %.0f pct"
      % (100 * EFF_TARGET))
print("safety criterion  : steady state Cmax at most %.1f mg/L" % CMAX_LIMIT)
print("candidate doses   :", DOSES)
print("grid              : %d points over one %g h interval" % (t_grid.size, TAU))
```

## Toolkit for this problem

| What you want to do | How |
|---|---|
| the steady state factor | `1 - np.exp(-k * TAU)` |
| one subject's profile | plain arithmetic on `t_grid` |
| four hundred profiles at once | `k[:, None]` against `t_grid[None, :]`, giving `(400, 241)` |
| the peak for each subject | `C.max(axis=1)` |
| the average over an interval | trapezoids along `axis=1`, divided by `TAU` |
| a fraction meeting a condition | `(x >= target).mean()` |
| both conditions at once | `((a >= x) & (b <= y)).mean()` |
| which entry is largest | `int(np.argmax(values))` |

The trapezoid along rows is worth writing out once:

```
area = np.sum(0.5 * (C[:, 1:] + C[:, :-1]) * np.diff(t_grid)[None, :], axis=1)
```

`np.diff(t_grid)` has 240 entries, one per interval, and `[None, :]` lines it
up against the 400 by 240 array of trapezoid heights.

The toolkit cell demonstrates each of those on a tiny made up example.
It is in the notebook and none of it is the answer.

## Part a. The typical subject at steady state

Work out the steady state profile over one dosing interval for a subject with
the typical parameters `CL_POP` and `V_POP`, at `DOSE_MID`, using the closed
form from the concept section with `k = CL / V`.

Record:

- `c_typ`, the profile over `t_grid`, an array of 241 concentrations,
- `cmax_ss`, the peak,
- `cmin_ss`, the trough,
- `cavg_ss`, the average over the interval, by trapezoidal integration divided
  by `TAU`,
- `cavg_identity`, the same quantity from `DOSE_MID / (CL_POP * TAU)`.

The last two should agree to four decimal places, which is the arithmetic
confirming the algebra: average exposure at steady state is set by clearance
and the dosing interval alone.

What the numbers say. The peak is 2.285 mg/L and the trough is 0.193, a peak to
trough ratio of 11.8 over a day. That swing is the work of the volume and the
absorption rate, and it is why a once daily dose of this drug spends part of
the day below its IC50 of about 0.4 mg/L. A twice daily regimen at half the
dose would have the same `C_avg`, the same total exposure, and a much smaller
swing, which is the usual reason for splitting a dose.

Fill in `c_typ`, `cmax_ss`, `cmin_ss`, `cavg_ss`, `cavg_identity`.

Moves: `np.exp` on `t_grid`, `.max()`, `.min()`, trapezoids.

**Variables this part is graded on:** `k_typ`, `c_typ`, `cmax_ss`, `cmin_ss`, `cavg_ss`, `cavg_identity`

The starter cell they see:

```python
# Part a. One typical subject, at steady state.

k_typ = None   # YOUR CODE: CL_POP / V_POP
c_typ = None   # YOUR CODE: the steady state profile over t_grid at DOSE_MID

cmax_ss = None         # YOUR CODE: peak concentration
cmin_ss = None         # YOUR CODE: trough concentration
cavg_ss = None         # YOUR CODE: average over the interval, by trapezoids
cavg_identity = None   # YOUR CODE: DOSE_MID / (CL_POP * TAU)

print("Cmax %r mg/L, Cmin %r mg/L" % (cmax_ss, cmin_ss))
if cmax_ss is not None and cmin_ss is not None:
    print("peak to trough ratio %.4f" % (cmax_ss / cmin_ss))
print("Cavg by integration %r mg/L" % (cavg_ss,))
print("Cavg from DOSE / (CL * tau) %r mg/L" % (cavg_identity,))

if c_typ is not None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t_grid, c_typ, "-", color="#1f4e79")
    ax.axhline(CMAX_LIMIT, ls="--", color="#aa0000", label="safety limit")
    ax.axhline(IC50_POP, ls=":", color="grey", label="typical IC50")
    if cavg_ss is not None:
        ax.axhline(cavg_ss, ls="-.", color="#007700", label="Cavg")
    ax.set_xlabel("time within the dosing interval (h)")
    ax.set_ylabel("concentration (mg/L)")
    ax.set_title("W8D5 steady state, typical subject, %g mg daily" % DOSE_MID)
    ax.legend()
    plt.show()
```

**Every failure message part a can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `c_typ` should be an array of concentrations.
- `c_typ` has shape %s but t_grid has %d points.
- `c_typ` is the single dose profile. The steady state version divides each exponential by one minus its own accumulation factor, 1 - exp(-k * TAU) and 1 - exp(-KA * TAU).
- `c_typ` is not the steady state profile at the typical parameters. Use k = CL_POP / V_POP and the closed form from the concept section.
- `cmax_ss` is not the peak of the steady state profile.
- `cmin_ss` is zero, which is the concentration at the start of a SINGLE dose profile. At steady state the trough is what is left from the previous doses.
- `cmin_ss` is not the trough of the steady state profile.
- `cavg_ss` is the area, not the average. Divide by TAU.
- `cavg_ss` is not the interval average of c_typ. Integrate over the interval with trapezoids and divide by TAU.
- `cavg_identity` should be DOSE_MID / (CL_POP * TAU).
- `%s` should be a single number.

## Part b. The population at 100 mg

Write a function that takes a dose and returns two arrays of four hundred
numbers: each patient's steady state `Cmax`, and each patient's average
inhibition over the dosing interval. Build the `(400, 241)` concentration
matrix by broadcasting, exactly as in the toolkit, rather than looping over
patients.

Inhibition for patient `i` at time `t` is

```
IMAX * C[i, t] / (ic50_i[i] + C[i, t])
```

and the average over the interval is the trapezoidal area along `axis=1`
divided by `TAU`.

Call it at `DOSE_MID` and record `cmax_i` and `mean_inh_i`. Then the three
fractions:

- `p_eff`, the fraction of patients whose average inhibition reaches
  `EFF_TARGET`,
- `p_tox`, the fraction whose `Cmax` exceeds `CMAX_LIMIT`,
- `p_success`, the fraction that meet the efficacy criterion **and** stay
  within the safety limit.

At 100 mg you should get 72.0 percent efficacy, 8.7 percent above the safety
limit, and 64.5 percent success.

Note that `p_success` is not `p_eff` times one minus `p_tox`, which would be
65.7 percent. The two criteria are not independent: a patient with a low
clearance has both a high average exposure, which helps efficacy, and a high
peak, which hurts safety. Multiplying marginal probabilities would overstate
your chances, and the only reliable way to get the joint fraction is to count
the patients who satisfy both, which is what the `&` does.

Fill in `cmax_i`, `mean_inh_i`, `p_eff`, `p_tox`, `p_success`.

Moves: broadcasting with `[:, None]` and `[None, :]`, `.max(axis=1)`,
trapezoids along `axis=1`, `(condition).mean()`, `&`.

**Variables this part is graded on:** `k`, `cmax_i`, `mean_inh_i`, `p_eff`, `p_tox`, `p_success`

The starter cell they see:

```python
# Part b. Four hundred patients at the middle dose.

def population_metrics(dose):
    """Steady state Cmax and average inhibition for every patient."""
    k = None        # YOUR CODE: cl_i / v_i, one per patient
    # YOUR CODE: build the (N_SUBJ, 241) concentration matrix by broadcasting,
    # then the peak per patient and the average inhibition per patient
    return None, None


cmax_i = None       # YOUR CODE: steady state Cmax for each patient at DOSE_MID
mean_inh_i = None   # YOUR CODE: average inhibition for each patient at DOSE_MID

p_eff = None        # YOUR CODE: fraction reaching EFF_TARGET
p_tox = None        # YOUR CODE: fraction above CMAX_LIMIT
p_success = None    # YOUR CODE: fraction meeting both criteria

print("at %g mg daily:" % DOSE_MID)
print("  efficacy criterion met : %r" % (p_eff,))
print("  above the safety limit : %r" % (p_tox,))
print("  both at once           : %r" % (p_success,))
if p_eff is not None and p_tox is not None:
    print("  product of the margins : %.4f  <- not the same thing"
          % (p_eff * (1.0 - p_tox)))
if cmax_i is not None:
    print("")
    print("  Cmax   5th, 50th, 95th percentile:", np.round(np.percentile(cmax_i, [5, 50, 95]), 4))
    print("  mean inhibition, same percentiles:", np.round(np.percentile(mean_inh_i, [5, 50, 95]), 4))
```

**Every failure message part b can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `p_eff` is the fraction that FAILS the efficacy criterion. The comparison is mean inhibition at or above EFF_TARGET.
- `p_eff` is not the fraction of patients reaching EFF_TARGET.
- `p_tox` is the fraction within the limit. The task asks for the fraction ABOVE CMAX_LIMIT.
- `p_tox` is not the fraction of patients whose Cmax exceeds CMAX_LIMIT.
- `p_success` is p_eff times one minus p_tox, which assumes the two criteria are independent. They are not: count the patients who satisfy both, with a single & expression.
- `p_success` is not the fraction meeting the efficacy criterion while staying at or below the safety limit.
- `%s` should be an array of numbers.
- `%s` has shape %s but there are %d patients.
- `%s` does not match the expected values at DOSE_MID. Check the broadcasting: k[:, None] against t_grid[None, :].
- `%s` should be a single number.
- `%s` should be a fraction between 0 and 1.
- `%s` has one entry per time point, so the reduction collapsed the wrong axis. With a (patients, times) matrix you want axis=1.

## Part c. All five doses, and the recommendation

Run the same calculation at every dose in `DOSES` and collect three arrays of
five numbers, `p_eff_by_dose`, `p_tox_by_dose` and `p_success_by_dose`. Then
set `best_dose` to the dose with the highest joint success probability, found
with `np.argmax` on the success array and used to index `DOSES`.

Here is what you have produced.

| dose | efficacy met | above safety limit | both |
|---|---|---|---|
| 25 mg | 6.2 pct | 0.0 pct | 6.2 pct |
| 50 mg | 28.0 pct | 0.0 pct | 28.0 pct |
| 100 mg | 72.0 pct | 8.7 pct | 64.5 pct |
| 150 mg | 88.7 pct | 77.7 pct | 18.0 pct |
| 200 mg | 94.0 pct | 98.0 pct | 1.8 pct |

Read the columns separately and then together.

**The efficacy column never stops improving.** 200 mg treats 94 percent of
patients effectively, the best of the five. Anyone shown only this column will
argue for the highest dose, and they will be arguing from a real number.

**The safety column collapses between 100 and 150 mg.** 8.7 percent to 77.7
percent for a 1.5 fold dose step. Nothing special happens to the drug there.
The median patient's `Cmax` is 2.32 mg/L at 100 mg and 3.48 at 150 mg, and the
limit is 3.0, so the step moves the limit from the upper tail of the
distribution to below its middle. The sensitivity of the answer to the dose is
entirely about where the limit falls among the patients, which is why this
calculation cannot be done with a typical subject.

**Together, 100 mg wins by more than a factor of two** over its nearest
rival, and it is neither the most effective dose nor the safest. That is the
normal shape of this problem.

So the recommendation is 100 mg once daily, and the honest version of it comes
with three caveats you should be able to state. The criteria were given to you
rather than derived, and the whole answer moves if the safety limit is 3.5
instead of 3.0. The variability estimates come from a small phase 1 study, and
week 8 day 3 showed how loosely a CV is known from a dozen subjects. And
nothing here allows for the dose being split: the same daily amount given twice
daily would have the identical `C_avg` with half the peak, which on these two
criteria would beat every one of the five regimens you just tested.

That last point is worth sitting with. The best answer to the question as asked
is 100 mg once daily. The best answer to the question the team actually cares
about may well be 75 mg twice daily, and nothing in the five candidate doses
would ever have found it.

Fill in `p_eff_by_dose`, `p_tox_by_dose`, `p_success_by_dose`, `best_dose`.

Moves: a loop over `DOSES` calling your part b function, `np.array`,
`np.argmax`.

**Variables this part is graded on:** `p_eff_by_dose`, `p_tox_by_dose`, `p_success_by_dose`, `best_dose`

The starter cell they see:

```python
# Part c. The dose response of the decision.

p_eff_by_dose = None      # YOUR CODE: array of 5 efficacy fractions
p_tox_by_dose = None      # YOUR CODE: array of 5 safety exceedance fractions
p_success_by_dose = None  # YOUR CODE: array of 5 joint success fractions

best_dose = None          # YOUR CODE: the dose with the highest joint success

if p_success_by_dose is not None:
    print(" dose   efficacy   over limit   both")
    for _i, _d in enumerate(DOSES):
        print("%5.0f %9.3f %12.3f %7.3f"
              % (_d, p_eff_by_dose[_i], p_tox_by_dose[_i], p_success_by_dose[_i]))
print("")
print("recommended dose: %r mg once daily" % (best_dose,))

if p_success_by_dose is not None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(DOSES, p_eff_by_dose, "o-", label="efficacy criterion met")
    ax.plot(DOSES, p_tox_by_dose, "s-", label="above the safety limit")
    ax.plot(DOSES, p_success_by_dose, "^-", lw=2, color="black", label="both")
    ax.set_xscale("log")
    ax.set_xticks(DOSES)
    ax.set_xticklabels(["%g" % _d for _d in DOSES])
    ax.set_xlabel("daily dose (mg)")
    ax.set_ylabel("fraction of patients")
    ax.set_title("W8D5 dose selection in 400 virtual patients")
    ax.legend()
    plt.show()
```

**Every failure message part c can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `best_dose` should be a single number, a dose in mg.
- `best_dose` is the index of the best dose, not the dose. Use it to index DOSES.
- `best_dose` is the dose that maximises efficacy alone. The decision is on the joint criterion, where that dose does badly.
- `best_dose` is not the dose with the highest joint success fraction. Take np.argmax of the success array and index DOSES with it.
- `%s` should be an array of five numbers.
- `%s` has shape %s. It should hold one fraction per candidate dose, in the order of DOSES.
- `%s` does not match the expected fractions. Each dose needs its own call to the part b calculation.
- `%s` is in reverse dose order.

**Functions this problem introduces:** steady state closed form, Cavg identity, probability of target attainment, joint criteria, boolean mean as a fraction
