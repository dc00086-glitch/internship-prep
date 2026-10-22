PID = "W7D4"
WEEK = 7
DAY = 4
SLUG = "w7d4_pk_two_compartment"
TITLE = "Two compartments, and what a one compartment model gets wrong"
TOPIC = "PK modelling: distribution phase, alpha and beta, choosing model structure"
EST = "20-30 min"

SCENARIO_MD = '''
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
'''

CONCEPT_MD = '''
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
'''

TASK_MD = '''
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
'''

DATA_CODE = '''
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
'''

TOOLKIT_MD = '''
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
'''

TOOLKIT_CODE = '''
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# A tiny made up two compartment system. Nothing here is the answer.
demo_dose = 100.0


def demo_rhs2(tt, y, cl, v1, q, v2):
    a1, a2 = y
    to_tissue = (q / v1) * a1
    from_tissue = (q / v2) * a2
    return [-(cl / v1) * a1 - to_tissue + from_tissue,
            to_tissue - from_tissue]


demo_grid = np.linspace(0.02, 24, 400)
demo_sol = solve_ivp(demo_rhs2, (0.0, 24.0), [demo_dose, 0.0], t_eval=demo_grid,
                     args=(3.0, 8.0, 6.0, 25.0), rtol=1e-10, atol=1e-12)
demo_c = demo_sol.y[0] / 8.0

print("--- 1. the two phases are visible on a log axis ---")
for demo_t in [0.1, 0.5, 1.0, 2.0, 6.0, 12.0, 24.0]:
    demo_i = int(np.argmin(np.abs(demo_grid - demo_t)))
    print("  t=%5.1f h   C=%7.4f mg/L" % (demo_t, demo_c[demo_i]))

print("")
print("--- 2. the slope changes between early and late ---")
demo_early = np.log(demo_c[np.argmin(np.abs(demo_grid - 0.1))] /
                    demo_c[np.argmin(np.abs(demo_grid - 1.0))]) / (1.0 - 0.1)
demo_late = np.log(demo_c[np.argmin(np.abs(demo_grid - 12.0))] /
                   demo_c[np.argmin(np.abs(demo_grid - 24.0))]) / (24.0 - 12.0)
print("apparent rate constant, 0.1 to 1 h : %.4f 1/h  <- alpha, steep" % demo_early)
print("apparent rate constant, 12 to 24 h : %.4f 1/h  <- beta, shallow" % demo_late)
print("a one compartment model has only ONE rate constant to offer")

print("")
print("--- 3. where the drug is over time ---")
for demo_t in [0.1, 1.0, 6.0, 24.0]:
    demo_i = int(np.argmin(np.abs(demo_grid - demo_t)))
    print("  t=%5.1f h   central %6.2f mg   peripheral %6.2f mg"
          % (demo_t, demo_sol.y[0][demo_i], demo_sol.y[1][demo_i]))
print("early on the tissue is filling, which is what makes plasma fall so fast")

demo_fig, demo_ax = plt.subplots(figsize=(5.2, 3.4))
demo_ax.plot(demo_grid, demo_c)
demo_ax.set_yscale("log")
demo_ax.set_xlabel("Time (h)"); demo_ax.set_ylabel("Made up units")
demo_ax.set_title("Toolkit demo, not the answer")
plt.show()
'''

PARTS = [
    dict(
        label="a",
        prompt_md='''
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
''',
        starter='''
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
''',
        solution='''
# Part a. The two compartment system, fitted.

def rhs2(t, y, cl, v1, q, v2):
    """y[0] central amount, y[1] peripheral amount, both mg."""
    to_tissue = (q / v1) * y[0]
    from_tissue = (q / v2) * y[1]
    return [-(cl / v1) * y[0] - to_tissue + from_tissue,
            to_tissue - from_tissue]


def predict2(t_eval, cl, v1, q, v2):
    """Central concentration in mg/L at each time."""
    t_eval = np.asarray(t_eval, dtype=float)
    sol = solve_ivp(rhs2, (0.0, float(t_eval.max())), [DOSE_MG, 0.0],
                    t_eval=t_eval, args=(cl, v1, q, v2), rtol=1e-10, atol=1e-12)
    return sol.y[0] / v1


popt2, _ = curve_fit(predict2, t, c, p0=[3.0, 10.0, 6.0, 25.0],
                     bounds=([0.0] * 4, [np.inf] * 4))

print("fitted CL %s, V1 %s, Q %s, V2 %s" % tuple(popt2) if popt2 is not None else "not fitted")
print("truth   CL 3.2, V1 9.0, Q 6.5, V2 26.0")
''',
    ),
    dict(
        label="b",
        prompt_md='''
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
''',
        starter='''
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
''',
        solution='''
# Part b. A one compartment fit, for comparison.

def one_comp(t_eval, cl, v):
    """IV bolus into a single compartment, in closed form."""
    return (DOSE_MG / v) * np.exp(-(cl / v) * np.asarray(t_eval, dtype=float))


popt1, _ = curve_fit(one_comp, t, c, p0=[3.0, 25.0],
                     bounds=([0.0, 0.0], [np.inf, np.inf]))
rmse1 = float(np.sqrt(np.mean((c - one_comp(t, *popt1)) ** 2)))
rmse2 = float(np.sqrt(np.mean((c - predict2(t, *popt2)) ** 2)))

print("one compartment: CL %s, V %s, rmse %s" % (
    popt1[0] if popt1 is not None else None,
    popt1[1] if popt1 is not None else None, rmse1))
print("two compartment: rmse %s" % rmse2)
if popt1 is not None:
    print("")
    print("one compartment residuals (observed minus fitted):")
    print(np.round(c - one_comp(t, *popt1), 4))
''',
    ),
    dict(
        label="c",
        prompt_md='''
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
''',
        starter='''
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
''',
        solution='''
# Part c. What the one compartment model costs you early.

vss = float(popt2[1] + popt2[3])

_i12 = int(np.where(t == 12.0)[0][0])
_pred12 = float(one_comp(t[_i12:_i12 + 1], *popt1)[0])
late_bias = float(100.0 * (_pred12 - c[_i12]) / c[_i12])

auc_one = float(DOSE_MG / popt1[0])
auc_two = float(DOSE_MG / popt2[0])
auc_bias = float(100.0 * (auc_one - auc_two) / auc_two)

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
''',
    ),
]


def build_expected(ns):
    import numpy as np
    from scipy.integrate import solve_ivp
    from scipy.optimize import curve_fit
    t, c, dose = ns["t"], ns["c"], ns["DOSE_MG"]

    def rhs2(tt, y, cl, v1, q, v2):
        to_t = (q / v1) * y[0]
        from_t = (q / v2) * y[1]
        return [-(cl / v1) * y[0] - to_t + from_t, to_t - from_t]

    def predict2(te, cl, v1, q, v2):
        te = np.asarray(te, dtype=float)
        sol = solve_ivp(rhs2, (0.0, float(te.max())), [dose, 0.0], t_eval=te,
                        args=(cl, v1, q, v2), rtol=1e-10, atol=1e-12)
        return sol.y[0] / v1

    def one_comp(te, cl, v):
        return (dose / v) * np.exp(-(cl / v) * np.asarray(te, dtype=float))

    popt2, _ = curve_fit(predict2, t, c, p0=[3.0, 10.0, 6.0, 25.0],
                         bounds=([0.0] * 4, [np.inf] * 4))
    popt1, _ = curve_fit(one_comp, t, c, p0=[3.0, 25.0],
                         bounds=([0.0, 0.0], [np.inf, np.inf]))
    r1 = float(np.sqrt(np.mean((c - one_comp(t, *popt1)) ** 2)))
    r2 = float(np.sqrt(np.mean((c - predict2(t, *popt2)) ** 2)))
    c0_one = float(one_comp(t[:1], *popt1)[0])
    return {
        "n_points": int(t.size),
        "probe2": [float(v) for v in predict2(t, 3.0, 10.0, 6.0, 25.0)],
        "popt2": [float(v) for v in popt2],
        "popt1": [float(v) for v in popt1],
        "rmse1": r1, "rmse2": r2,
        "mse1": float(np.mean((c - one_comp(t, *popt1)) ** 2)),
        "vss": float(popt2[1] + popt2[3]),
        "v1": float(popt2[1]),
        "i12": int(np.where(t == 12.0)[0][0]),
        "late_bias": float(100.0 * (float(one_comp(t[t == 12.0], *popt1)[0])
                                    - float(c[t == 12.0][0])) / float(c[t == 12.0][0])),
        "auc_one": float(dose / popt1[0]),
        "auc_two": float(dose / popt2[0]),
        "auc_bias": float(100.0 * (dose / popt1[0] - dose / popt2[0]) / (dose / popt2[0])),
    }


CHECK_CODE = '''
def _part_a():
    rhs2 = _need("rhs2")
    predict2 = _need("predict2")
    popt2 = _need("popt2")

    if not callable(rhs2):
        return "`rhs2` should be a function."
    try:
        got = rhs2(0.0, [100.0, 20.0], 3.0, 10.0, 6.0, 25.0)
    except Exception as exc:
        return "`rhs2` raised %s when called as rhs2(t, y, cl, v1, q, v2)." % type(exc).__name__
    if got is None:
        raise _Todo("`rhs2` returns None. Replace the # YOUR CODE line.")
    arr = _as_array(got)
    if arr is None or arr.size != 2:
        return "`rhs2` should return TWO derivatives."
    to_t, from_t = (6.0 / 10.0) * 100.0, (6.0 / 25.0) * 20.0
    want = [-(3.0 / 10.0) * 100.0 - to_t + from_t, to_t - from_t]
    if not _np.allclose(arr.ravel(), want):
        if abs(float(arr.ravel().sum()) + (3.0 / 10.0) * 100.0) > 1e-8:
            return ("`rhs2` does not conserve mass. The two transfer terms must appear in "
                    "BOTH equations with opposite signs, so the derivatives sum to minus "
                    "the elimination term alone.")
        return ("`rhs2` does not match the equations in the prompt. With y = [100, 20], "
                "cl 3, v1 10, q 6, v2 25 it should give [-85.2, 55.2].")

    if not callable(predict2):
        return "`predict2` should be a function."
    try:
        got = predict2(_need("t"), 3.0, 10.0, 6.0, 25.0)
    except Exception as exc:
        return "`predict2` raised %s." % type(exc).__name__
    if got is None:
        raise _Todo("`predict2` returns None.")
    arr = _as_array(got)
    if arr is None or arr.shape != (_EXP["n_points"],):
        return "`predict2` should return one concentration per requested time."
    if not _arr_close(arr, _EXP["probe2"], rtol=1e-4, atol=1e-8):
        return ("`predict2` does not match. Start the whole dose in the CENTRAL compartment "
                "for an IV bolus, return sol.y[0] divided by v1, and use "
                "rtol=1e-10, atol=1e-12.")

    arr = _as_array(popt2)
    if arr is None or arr.size != 4:
        return "`popt2` should hold the four fitted parameters."
    if not _arr_close(arr, _EXP["popt2"], rtol=1e-2):
        return ("`popt2` does not match the fit. Use p0=[3.0, 10.0, 6.0, 25.0] with all four "
                "bounded below at zero.")
    return None


def _part_b():
    one_comp = _need("one_comp")
    popt1 = _need("popt1")
    rmse1 = _need("rmse1")
    rmse2 = _need("rmse2")

    if not callable(one_comp):
        return "`one_comp` should be a function."
    probe = one_comp(_np.array([0.0, 1.0]), 3.0, 25.0)
    if probe is None:
        raise _Todo("`one_comp` returns None. Replace the # YOUR CODE line.")
    probe = _np.asarray(probe, dtype=float)
    want0 = _need("DOSE_MG") / 25.0
    if not _np.isclose(float(probe[0]), want0):
        return ("`one_comp` at t = 0 should be dose / v, which is %.4f here, but it gave "
                "%.4f." % (want0, float(probe[0])))

    arr = _as_array(popt1)
    if arr is None or arr.size != 2:
        return "`popt1` should hold the two fitted parameters."
    if not _arr_close(arr, _EXP["popt1"], rtol=1e-2):
        return "`popt1` does not match the one compartment fit with p0=[3.0, 25.0]."

    got = _as_scalar(rmse1)
    if got is None:
        return "`rmse1` should be a single number."
    if _num_close(got, _EXP["mse1"], rtol=1e-3):
        return "`rmse1` is the mean squared error. Take its square root."
    if not _num_close(got, _EXP["rmse1"], rtol=1e-2):
        return "`rmse1` is not the RMSE of the one compartment fit."

    got = _as_scalar(rmse2)
    if got is None:
        return "`rmse2` should be a single number."
    if not _num_close(got, _EXP["rmse2"], rtol=5e-2):
        return "`rmse2` is not the RMSE of the two compartment fit."
    if got > _EXP["rmse1"]:
        return ("`rmse2` came out worse than `rmse1`, which should not happen. The richer "
                "model can always reproduce the simpler one, so a worse fit means it did "
                "not converge.")
    return None


def _part_c():
    vss = _need("vss")
    late_bias = _need("late_bias")
    auc_one = _need("auc_one")
    auc_two = _need("auc_two")
    auc_bias = _need("auc_bias")

    got = _as_scalar(vss)
    if got is None:
        return "`vss` should be a single number."
    if _num_close(got, _EXP["v1"], rtol=1e-6):
        return ("`vss` is V1 alone, the central volume. The volume at steady state is "
                "V1 + V2, and reporting V1 as the volume of distribution understates it "
                "badly.")
    if not _num_close(got, _EXP["vss"], rtol=1e-2):
        return "`vss` is not popt2[1] plus popt2[3]."

    got = _as_scalar(late_bias)
    if got is None:
        return "`late_bias` should be a single number."
    if _num_close(got, -_EXP["late_bias"], rtol=1e-2):
        return ("`late_bias` has the sign the wrong way round. It is the prediction minus "
                "the observation, over the observation, so a model reading near zero "
                "against real drug gives close to minus one hundred.")
    if not _num_close(got, _EXP["late_bias"], rtol=1e-2, atol=0.5):
        return ("`late_bias` is not the percent error of the one compartment fit at 12 "
                "hours. Find the index where t equals 12.0 and compare there.")

    got = _as_scalar(auc_one)
    if got is None:
        return "`auc_one` should be a single number."
    if not _num_close(got, _EXP["auc_one"], rtol=1e-2):
        return "`auc_one` is not DOSE_MG divided by the one compartment clearance, popt1[0]."

    got = _as_scalar(auc_two)
    if got is None:
        return "`auc_two` should be a single number."
    if not _num_close(got, _EXP["auc_two"], rtol=1e-2):
        return "`auc_two` is not DOSE_MG divided by the two compartment clearance, popt2[0]."

    got = _as_scalar(auc_bias)
    if got is None:
        return "`auc_bias` should be a single number."
    if _num_close(got, -_EXP["auc_bias"], rtol=1e-2):
        return ("`auc_bias` has the sign the wrong way round. It is the one compartment "
                "exposure minus the two compartment one, over the two compartment one, so "
                "a model that discards the tail gives a negative number.")
    if not _num_close(got, _EXP["auc_bias"], rtol=2e-2):
        return "`auc_bias` is not 100 * (auc_one - auc_two) / auc_two."
    return None
'''

# Functions this problem introduces. Keys must exist in tools/function_catalog.py.
FUNCTIONS = [
    "two compartment model",
    "alpha and beta phases",
    "Vss",
    "residual structure",
]
