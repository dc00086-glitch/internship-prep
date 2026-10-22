PID = "W8D1"
WEEK = 8
DAY = 1
SLUG = "w8d1_pkpd_hysteresis"
TITLE = "PK/PD link: why the effect peaks hours after the concentration"
TOPIC = "PK/PD: hysteresis, direct Emax fitting and its residual structure, the effect compartment"
EST = "25-35 min"

SCENARIO_MD = '''
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
'''

CONCEPT_MD = '''
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
'''

TASK_MD = '''
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
'''

DATA_CODE = '''
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
'''

TOOLKIT_MD = '''
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
'''

TOOLKIT_CODE = '''
import numpy as np
from scipy.integrate import solve_ivp

print("--- 1. interpolation needs increasing x, and will not tell you otherwise ---")
demo_x = np.array([1.0, 2.0, 3.0, 4.0])
demo_y = np.array([10.0, 20.0, 30.0, 40.0])
print("increasing x, interp at 2.5:", np.interp(2.5, demo_x, demo_y), " correct")
print("decreasing x, interp at 2.5:", np.interp(2.5, demo_x[::-1], demo_y[::-1]), " nonsense")
print("reverse BOTH arrays together:", np.interp(2.5, demo_x[::-1][::-1], demo_y[::-1][::-1]))

print("")
print("--- 2. splitting a curve at its peak ---")
demo_c = np.array([0.5, 1.4, 2.1, 1.8, 1.1, 0.6])
demo_e = np.array([1.0, 4.0, 9.0, 12.0, 11.0, 7.0])
demo_i = int(np.argmax(demo_c))
print("peak at index", demo_i, "value", demo_c[demo_i])
print("rising limb  :", demo_c[:demo_i + 1], "->", demo_e[:demo_i + 1])
print("falling limb :", demo_c[demo_i:], "->", demo_e[demo_i:])
print("the peak sample belongs to both limbs, which is the usual convention")
demo_probe = 1.5
demo_rise = np.interp(demo_probe, demo_c[:demo_i + 1], demo_e[:demo_i + 1])
demo_fall = np.interp(demo_probe, demo_c[demo_i:][::-1], demo_e[demo_i:][::-1])
print("at c = %.1f the rising limb gives %.3f and the falling limb %.3f"
      % (demo_probe, demo_rise, demo_fall))
print("effect larger on the way down means a counter clockwise loop")

print("")
print("--- 3. a one compartment effect site, driven by an input function ---")


def demo_input(t):
    """A made up plasma profile: up then down."""
    return 2.0 * (np.exp(-0.2 * t) - np.exp(-1.0 * t))


demo_t = np.linspace(0.0, 24.0, 13)
for demo_ke0 in (0.05, 0.5, 5.0):
    demo_sol = solve_ivp(lambda t, y: demo_ke0 * (demo_input(t) - y[0]),
                         (0.0, 24.0), [0.0], t_eval=demo_t, rtol=1e-9, atol=1e-12)
    demo_ce = demo_sol.y[0]
    print("ke0 %-5g  half life %5.2f h  Ce peaks at t = %5.2f  (input peaks at %.2f)"
          % (demo_ke0, np.log(2) / demo_ke0, demo_t[int(np.argmax(demo_ce))],
             demo_t[int(np.argmax(demo_input(demo_t)))]))
print("a big ke0 means no delay, a small one means a slow smoothed effect")

print("")
print("--- 4. residuals in time order are the diagnostic ---")
demo_res_good = np.array([0.4, -0.6, 0.2, -0.3, 0.5, -0.2])
demo_res_bad = np.array([-1.8, -1.2, -0.5, 0.6, 1.3, 1.9])
print("scattered :", demo_res_good, " mean of first half %.2f second half %.2f"
      % (demo_res_good[:3].mean(), demo_res_good[3:].mean()))
print("structured:", demo_res_bad, " mean of first half %.2f second half %.2f"
      % (demo_res_bad[:3].mean(), demo_res_bad[3:].mean()))
print("the second one has the same sum of squares spread very differently")
'''

PARTS = [
    dict(
        label="a",
        prompt_md='''
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
''',
        starter='''
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
''',
        solution='''
# Part a. The hysteresis loop.

peak_idx = int(np.argmax(c_obs))

tmax_c = float(t_obs[peak_idx])
tmax_e = float(t_obs[int(np.argmax(e_obs))])

e_rising = float(np.interp(C_PROBE, c_obs[:peak_idx + 1], e_obs[:peak_idx + 1]))
e_falling = float(np.interp(C_PROBE, c_obs[peak_idx:][::-1], e_obs[peak_idx:][::-1]))

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
''',
    ),
    dict(
        label="b",
        prompt_md='''
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
''',
        starter='''
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
''',
        solution='''
# Part b. The direct Emax fit, and its residuals.

def direct_emax(c, emax, ec50):
    """Effect as a function of the concentration at that instant."""
    return emax * c / (ec50 + c)


_popt, _pcov = curve_fit(direct_emax, c_obs, e_obs, p0=[20.0, 1.0],
                         bounds=([0.0, 1e-3], [200.0, 100.0]))
emax_direct = float(_popt[0])
ec50_direct = float(_popt[1])

_pred = direct_emax(c_obs, emax_direct, ec50_direct)
_res = e_obs - _pred
sse_direct = float((_res ** 2).sum())

resid_early = float(_res[:peak_idx + 1].mean())
resid_late = float(_res[peak_idx + 1:].mean())

print("direct fit: Emax %r ms, EC50 %r mg/L" % (emax_direct, ec50_direct))
print("SSE       : %r" % (sse_direct,))
print("mean residual before the peak: %r" % (resid_early,))
print("mean residual after the peak : %r" % (resid_late,))
''',
    ),
    dict(
        label="c",
        prompt_md='''
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
''',
        starter='''
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
''',
        solution='''
# Part c. The effect compartment model.

def plasma(t):
    """Plasma concentration at arbitrary times, interpolated from the samples."""
    return np.interp(t, t_obs, c_obs)


def link_model(t, emax, ec50, ke0):
    """Effect driven by the effect site concentration."""
    t = np.asarray(t, dtype=float)
    sol = solve_ivp(lambda tt, y: ke0 * (plasma(tt) - y[0]),
                    (0.0, float(t.max())), [0.0], t_eval=t,
                    rtol=1e-9, atol=1e-12)
    ce = sol.y[0]
    return emax * ce / (ec50 + ce)


_popt2, _pcov2 = curve_fit(link_model, t_obs, e_obs, p0=[20.0, 1.0, 0.3],
                           bounds=([0.0, 1e-3, 1e-3], [200.0, 100.0, 10.0]))
emax_hat = float(_popt2[0])
ec50_hat = float(_popt2[1])
ke0_hat = float(_popt2[2])

link_pred = link_model(t_obs, emax_hat, ec50_hat, ke0_hat)
sse_link = float(((e_obs - link_pred) ** 2).sum())

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
''',
    ),
]


def build_expected(ns):
    import numpy as np
    from scipy.integrate import solve_ivp
    from scipy.optimize import curve_fit

    t = np.asarray(ns["t_obs"], dtype=float)
    c = np.asarray(ns["c_obs"], dtype=float)
    e = np.asarray(ns["e_obs"], dtype=float)
    probe = float(ns["C_PROBE"])
    idx = int(np.argmax(c))

    def direct(cc, emax, ec50):
        return emax * cc / (ec50 + cc)

    popt, _ = curve_fit(direct, c, e, p0=[20.0, 1.0],
                        bounds=([0.0, 1e-3], [200.0, 100.0]))
    res_d = e - direct(c, *popt)

    def plasma(tt):
        return np.interp(tt, t, c)

    def link(tt, emax, ec50, ke0):
        tt = np.asarray(tt, dtype=float)
        sol = solve_ivp(lambda s, y: ke0 * (plasma(s) - y[0]),
                        (0.0, float(tt.max())), [0.0], t_eval=tt,
                        rtol=1e-9, atol=1e-12)
        ce = sol.y[0]
        return emax * ce / (ec50 + ce)

    popt2, pcov2 = curve_fit(link, t, e, p0=[20.0, 1.0, 0.3],
                             bounds=([0.0, 1e-3, 1e-3], [200.0, 100.0, 10.0]))
    pred_l = link(t, *popt2)

    return {
        "n": int(t.size),
        "peak_idx": idx,
        "tmax_c": float(t[idx]),
        "tmax_e": float(t[int(np.argmax(e))]),
        "e_rising": float(np.interp(probe, c[:idx + 1], e[:idx + 1])),
        "e_falling": float(np.interp(probe, c[idx:][::-1], e[idx:][::-1])),
        "e_falling_unreversed": float(np.interp(probe, c[idx:], e[idx:])),
        "emax_direct": float(popt[0]),
        "ec50_direct": float(popt[1]),
        "sse_direct": float((res_d ** 2).sum()),
        "resid_early": float(res_d[:idx + 1].mean()),
        "resid_late": float(res_d[idx + 1:].mean()),
        "emax_hat": float(popt2[0]),
        "ec50_hat": float(popt2[1]),
        "ke0_hat": float(popt2[2]),
        "link_pred": [float(x) for x in pred_l],
        "sse_link": float(((e - pred_l) ** 2).sum()),
        "ke0_se": float(np.sqrt(np.diag(pcov2))[2]),
    }


CHECK_CODE = '''

def _part_a():
    tmax_c = _need("tmax_c")
    tmax_e = _need("tmax_e")
    e_rising = _need("e_rising")
    e_falling = _need("e_falling")

    for name, val in (("tmax_c", tmax_c), ("tmax_e", tmax_e),
                      ("e_rising", e_rising), ("e_falling", e_falling)):
        if _as_scalar(val) is None:
            return "`%s` should be a single number." % name

    got = _as_scalar(tmax_c)
    if _num_close(got, float(_EXP["peak_idx"])):
        return ("`tmax_c` is the index of the peak, not the time. Use it to index t_obs.")
    if not _num_close(got, _EXP["tmax_c"]):
        return "`tmax_c` is not the time of the highest observed concentration."
    if _num_close(_as_scalar(tmax_e), _EXP["tmax_c"]):
        return ("`tmax_e` equals `tmax_c`. It should be the time of the highest observed "
                "EFFECT, which is a different sample.")
    if not _num_close(_as_scalar(tmax_e), _EXP["tmax_e"]):
        return "`tmax_e` is not the time of the highest observed effect."

    got = _as_scalar(e_rising)
    if not _num_close(got, _EXP["e_rising"], rtol=1e-3):
        return ("`e_rising` is not the effect interpolated at C_PROBE on the rising limb. "
                "The rising limb is c_obs up to and including the peak sample.")

    got = _as_scalar(e_falling)
    if _num_close(got, _EXP["e_falling_unreversed"], rtol=1e-3):
        return ("`e_falling` came from np.interp on a decreasing array, which returns a "
                "number without complaining and is wrong. Reverse the falling limb, both "
                "the concentrations and the effects.")
    if _num_close(got, _EXP["e_rising"], rtol=1e-3):
        return "`e_falling` matches `e_rising`, so both came from the same limb."
    if not _num_close(got, _EXP["e_falling"], rtol=1e-3):
        return ("`e_falling` is not the effect interpolated at C_PROBE on the falling limb. "
                "The falling limb runs from the peak sample to the last sample.")
    return None


def _part_b():
    emax = _need("emax_direct")
    ec50 = _need("ec50_direct")
    sse = _need("sse_direct")
    early = _need("resid_early")
    late = _need("resid_late")

    for name, val in (("emax_direct", emax), ("ec50_direct", ec50), ("sse_direct", sse),
                      ("resid_early", early), ("resid_late", late)):
        if _as_scalar(val) is None:
            return "`%s` should be a single number." % name

    fn = globals().get("direct_emax")
    if fn is not None:
        try:
            probe = fn(_np.array([1.0, 2.0]), 10.0, 1.0)
        except Exception as exc:
            return "`direct_emax` raised %s: %s" % (type(exc).__name__, exc)
        if probe is None:
            return "`direct_emax` returns None. It should return emax * c / (ec50 + c)."
        if not _arr_close(probe, [5.0, 20.0 / 3.0], rtol=1e-9):
            return ("`direct_emax` does not evaluate the Emax model. With emax = 10 and "
                    "ec50 = 1, concentrations of 1 and 2 should give 5 and 6.667.")

    if not _num_close(_as_scalar(emax), _EXP["emax_direct"], rtol=1e-2):
        return ("`emax_direct` is not the fitted Emax of the direct model. Fit "
                "direct_emax to c_obs and e_obs from p0=[20.0, 1.0] with both parameters "
                "bounded positive.")
    if not _num_close(_as_scalar(ec50), _EXP["ec50_direct"], rtol=1e-2):
        return "`ec50_direct` is not the fitted EC50 of the direct model."
    if _num_close(_as_scalar(sse), _EXP["sse_link"], rtol=1e-2):
        return "`sse_direct` is the link model's SSE. Part b is the direct fit."
    if not _num_close(_as_scalar(sse), _EXP["sse_direct"], rtol=1e-2):
        return ("`sse_direct` is not the summed squared residuals of the direct fit at the "
                "observed concentrations.")

    got = _as_scalar(early)
    if _num_close(got, _EXP["resid_late"], rtol=1e-3):
        return ("`resid_early` holds the late mean. The early window is up to and including "
                "the concentration peak.")
    if got > 0:
        return ("`resid_early` should be negative: the direct fit over predicts the early "
                "samples, so observed minus predicted is below zero there. Check the "
                "residual order, which is observed minus predicted.")
    if not _num_close(got, _EXP["resid_early"], rtol=2e-2):
        return ("`resid_early` is not the mean residual up to and including the "
                "concentration peak.")

    got = _as_scalar(late)
    if got < 0:
        return ("`resid_late` should be positive, since the direct fit under predicts the "
                "late samples.")
    if not _num_close(got, _EXP["resid_late"], rtol=2e-2):
        return "`resid_late` is not the mean residual after the concentration peak."
    return None


def _part_c():
    ke0 = _need("ke0_hat")
    emax = _need("emax_hat")
    ec50 = _need("ec50_hat")
    pred = _need("link_pred")
    sse = _need("sse_link")

    for name, val in (("ke0_hat", ke0), ("emax_hat", emax), ("ec50_hat", ec50),
                      ("sse_link", sse)):
        if _as_scalar(val) is None:
            return "`%s` should be a single number." % name

    arr = _as_array(pred)
    if arr is None:
        return "`link_pred` should be an array of predicted effects."
    if arr.shape != (_EXP["n"],):
        return ("`link_pred` has shape %s but there are %d sampling times."
                % (arr.shape, _EXP["n"]))

    if not _num_close(_as_scalar(emax), _EXP["emax_hat"], rtol=2e-2):
        if _num_close(_as_scalar(emax), _EXP["emax_direct"], rtol=1e-2):
            return ("`emax_hat` is the direct fit's Emax. The link model should give a much "
                    "larger value, close to the truth.")
        return ("`emax_hat` is not the fitted Emax of the effect compartment model. Check "
                "that Ce starts at zero and that the solver is evaluated at t_obs.")
    if not _num_close(_as_scalar(ec50), _EXP["ec50_hat"], rtol=3e-2):
        return "`ec50_hat` is not the fitted EC50 of the effect compartment model."
    if not _num_close(_as_scalar(ke0), _EXP["ke0_hat"], rtol=5e-2):
        if _as_scalar(ke0) > 5.0:
            return ("`ke0_hat` ran to its upper bound, which means the fit decided there is "
                    "no delay. Check the sign in dCe/dt = ke0 * (C(t) - Ce).")
        return ("`ke0_hat` is not the fitted effect site rate constant. It should come out "
                "near 0.2 per h, a half life of a few hours.")

    if not _arr_close(arr, _EXP["link_pred"], rtol=2e-2, atol=1e-2):
        return ("`link_pred` does not match the fitted effects at t_obs. It should be the "
                "link model evaluated at the fitted parameters.")
    if _num_close(_as_scalar(sse), _EXP["sse_direct"], rtol=1e-2):
        return "`sse_link` is the direct fit's SSE."
    if not _num_close(_as_scalar(sse), _EXP["sse_link"], rtol=5e-2):
        return ("`sse_link` is not the summed squared residuals of the effect compartment "
                "fit.")
    return None
'''

# Functions this problem introduces. Keys must exist in tools/function_catalog.py.
FUNCTIONS = [
    "np.interp",
    "hysteresis",
    "effect compartment",
    "residuals against time",
    "curve_fit bounds",
]
