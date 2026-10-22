PID = "W3D4"
WEEK = 3
DAY = 4
SLUG = "w3d4_scipy_solve_ivp"
TITLE = "Solving the compartment model as an ODE, and surviving a discontinuity"
TOPIC = "scipy.integrate: solve_ivp, t_eval, checking against an analytic solution"
EST = "20-30 min"

SCENARIO_MD = '''
## Scenario

So far every profile you have worked with was a formula you could evaluate
directly. That stops being possible very quickly: add a second compartment,
saturable elimination, a transit absorption chain or a PD effect feeding back
on clearance, and there is no closed form left to write down.

What there always is, is a differential equation. A one compartment model says
something simple about **rates**: the amount in the body falls at a rate
proportional to how much is there.

```
dA/dt = -(CL / V) * A
```

Numerical integration turns that statement into a curve. This problem works
through it on a case you can check, an IV bolus, where the exact answer is
known, so you can confirm the machinery before trusting it somewhere you
cannot check. Then you do an infusion, where the dosing rate switches off part
way through, which is where naive integration quietly goes wrong.
'''

CONCEPT_MD = '''
## solve_ivp, and why discontinuities need care

**The shape of the call.** `solve_ivp(fun, t_span, y0, ...)` needs three
things: a function giving the derivative, the interval to integrate over as a
pair, and the starting state as a **sequence**, even with one compartment, so
`[500.0]` rather than `500.0`.

The derivative function takes `(t, y)` in that order, with `y` an array of
states, and returns the derivatives in the same order. For one compartment:

```python
def rhs(t, y, cl, v):
    amount = y[0]
    return [-(cl / v) * amount]
```

Extra parameters are passed through with `args=(cl, v)`.

**`t_eval` controls output, not accuracy.** By default the solver returns the
points it happened to step through, which are chosen for efficiency and are
not evenly spaced. Passing `t_eval=np.linspace(...)` makes it report at the
times you want. It does not make the answer more accurate, it only changes
where it is sampled.

**Accuracy is controlled by `rtol` and `atol`, and the defaults are loose.**
`rtol=1e-3` means the solver is content with roughly three correct digits,
which is fine for a quick sketch and not fine for anything you will compare
against data or integrate to get an AUC. For PK work pass something like
`rtol=1e-8, atol=1e-10`. It costs a few milliseconds and removes solver error
as a thing you have to think about. You will see the difference in part b: on
the defaults the numerical and exact curves differ in the third decimal, and
with tight tolerances they agree to about eight.

The result carries `sol.t`, `sol.y` with one **row per state**, so
`sol.y[0]` is the first compartment, and `sol.success`, which is worth
checking rather than assuming.

**Amounts versus concentrations.** The state is an amount in mg, because that
is what mass balance is written in. Concentration is amount divided by volume.
Mixing the two up is the single most common error in writing PK models, and
it does not announce itself: the curve looks perfectly plausible, just scaled
wrong by a factor of V.

**The discontinuity problem.** An infusion that stops at 2 hours makes the
derivative jump. Adaptive solvers choose their step size by looking at how
smooth the solution has been, so a solver happily taking one hour steps can
step straight over the moment the infusion ended, never sampling the jump, and
return a curve that keeps rising past the end of the infusion.

Nothing errors. `sol.success` is still `True`. The answer is just wrong.

Two fixes. Cap the step size with `max_step` so the solver cannot stride over
the event, which is the blunt and reliable option. Or integrate each phase
separately and hand the end state of one in as the start of the next, which is
exact and is what real PK software does. You will use `max_step` here, but
knowing why it is needed matters more than the argument itself.
'''

TASK_MD = '''
## Your task

Three parts, worked one at a time.

| Part | What you work out | Variables to fill in |
|---|---|---|
| a | the ODE, solved numerically after a bolus | `rhs`, `sol`, `conc_num` |
| b | the same thing exactly, and the gap between them | `conc_exact`, `max_abs_err` |
| c | a 2 hour infusion, where the rate switches off | `sol_inf`, `conc_inf`, `cmax_inf`, `t_cmax` |

Below this you will find the **Setup** cell, then a **Toolkit** cell that
demonstrates every move on a tiny made up example, then **Part a**, **Part b**
and **Part c**, each with its own explanation and a cell to write in.
'''

DATA_CODE = '''
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

PROBLEM_ID = "W3D4"
TOPIC = "scipy.integrate: solve_ivp, t_eval, checking against an analytic solution"

# A one compartment model. No noise anywhere: this problem is about the
# machinery being right, so everything is exactly checkable.
CL_L_H = 4.20      # clearance, L/h
V_L = 40.0         # volume of distribution, L
K_EL = CL_L_H / V_L

DOSE_MG = 500.0            # IV bolus
T_GRID = np.linspace(0.0, 24.0, 97)        # every 15 minutes

INF_DOSE_MG = 500.0        # the same dose, given as an infusion
INF_DURATION_H = 2.0
INF_RATE_MG_H = INF_DOSE_MG / INF_DURATION_H
T_GRID_INF = np.linspace(0.0, 12.0, 241)   # every 3 minutes

print("CL = %.2f L/h   V = %.1f L   k = CL/V = %.5f 1/h" % (CL_L_H, V_L, K_EL))
print("bolus    : %.0f mg at t = 0" % DOSE_MG)
print("infusion : %.0f mg over %.0f h, so %.1f mg/h"
      % (INF_DOSE_MG, INF_DURATION_H, INF_RATE_MG_H))
print("")
print("bolus output grid :", T_GRID[:4], "...", T_GRID[-1], " (%d points)" % T_GRID.size)
print("infusion grid     :", T_GRID_INF[:4], "...", T_GRID_INF[-1],
      " (%d points)" % T_GRID_INF.size)
'''

TOOLKIT_MD = '''
## Toolkit for this problem

| What you want to do | How |
|---|---|
| import the solver | `from scipy.integrate import solve_ivp` |
| write a derivative | `def rhs(t, y, a): return [-a * y[0]]`, t first, y an array |
| solve it | `solve_ivp(rhs, (0, 24), [y0], t_eval=grid, args=(a,))` |
| the times it reported at | `sol.t` |
| the first state over time | `sol.y[0]`, one row per state |
| did it work | `sol.success` |
| stop it striding over a jump | `max_step=0.05` |
| largest absolute difference | `np.max(np.abs(a - b))` |
| where the maximum is | `np.argmax(x)` |

`y0` must be a sequence even with one compartment, so `[500.0]`, not `500.0`.
`sol.y` is always two dimensional, so `sol.y[0]` is how you get the first
compartment out.
'''

TOOLKIT_CODE = '''
import numpy as np
from scipy.integrate import solve_ivp

# A tiny made up decay, nothing to do with the real parameters.
demo_k = 0.4


def demo_rhs(t, y, k):
    """dy/dt = -k * y. t first, y is an ARRAY, return a sequence."""
    return [-k * y[0]]


print("--- 1. a basic solve ---")
demo_grid = np.linspace(0.0, 5.0, 6)
demo_sol = solve_ivp(demo_rhs, (0.0, 5.0), [100.0], t_eval=demo_grid, args=(demo_k,))
print("success :", demo_sol.success)
print("sol.t   :", demo_sol.t)
print("sol.y shape:", demo_sol.y.shape, " <- one ROW per state")
print("sol.y[0]:", np.round(demo_sol.y[0], 4))
print("exact   :", np.round(100.0 * np.exp(-demo_k * demo_grid), 4))
print("max abs error: %.3e" % float(np.max(np.abs(demo_sol.y[0]
                                                  - 100.0 * np.exp(-demo_k * demo_grid)))))

print("")
print("--- 2. without t_eval you get the solver's own steps ---")
demo_bare = solve_ivp(demo_rhs, (0.0, 5.0), [100.0], args=(demo_k,))
print("times chosen by the solver:", np.round(demo_bare.t, 3))
print("they are not evenly spaced, which is fine for accuracy but awkward to plot")

print("")
print("--- 3. a discontinuous input, and what goes wrong ---")


def demo_inf_rhs(t, y, k, rate, t_off):
    """Zero order input that stops at t_off."""
    r = rate if t < t_off else 0.0
    return [r - k * y[0]]


demo_t = np.linspace(0.0, 6.0, 61)
demo_loose = solve_ivp(demo_inf_rhs, (0.0, 6.0), [0.0], t_eval=demo_t,
                       args=(demo_k, 50.0, 2.0))
demo_tight = solve_ivp(demo_inf_rhs, (0.0, 6.0), [0.0], t_eval=demo_t,
                       args=(demo_k, 50.0, 2.0), max_step=0.05)
print("both report success:", demo_loose.success, demo_tight.success)
print("peak without max_step: %.4f at t = %.2f"
      % (demo_loose.y[0].max(), demo_t[int(np.argmax(demo_loose.y[0]))]))
print("peak with max_step   : %.4f at t = %.2f"
      % (demo_tight.y[0].max(), demo_t[int(np.argmax(demo_tight.y[0]))]))
print("the true peak is at t = 2.00, when the input stops")
print("")
print("the loose solve can keep climbing past the switch off, because its step")
print("size was chosen while the solution was smooth and nothing told it a jump")
print("was coming. No error, no warning, just a wrong curve.")
'''

PARTS = [
    dict(
        label="a",
        prompt_md='''
## Part a. Write the differential equation and solve it

Define `rhs(t, y, cl, v)`. The amount in the body is `y[0]`, and the
derivative is `-(cl / v) * amount`. Return it as a sequence, so a list of one
element, because the solver expects one derivative per state.

Then integrate from 0 to 24 hours starting from the whole bolus in the
compartment, so `y0` is `[DOSE_MG]`. Report at `T_GRID` by passing it as
`t_eval`, and pass the parameters through with `args=(CL_L_H, V_L)`.

Pass `rtol=1e-8, atol=1e-10` as well. The defaults ask for only about three
correct digits, which is not enough for work you will compare against data.

Finally convert to concentration. The state is an **amount** in mg, so divide
by the volume to get mg/L. This is the step people skip, and the resulting
curve looks entirely reasonable while being wrong by a factor of 40.

Fill in `rhs`, the derivative function. `sol`, the object `solve_ivp` returns.
And `conc_num`, the concentration in mg/L at each time in `T_GRID`.

Moves: `def rhs(t, y, cl, v)`, `solve_ivp(fun, (0, 24), [y0], t_eval=...,
args=(...))`, `sol.y[0]`, division.
''',
        starter='''
# Part a. The one compartment ODE, solved numerically after an IV bolus.

def rhs(t, y, cl, v):
    """dA/dt for one compartment. y[0] is the amount in mg."""
    return None   # YOUR CODE: return a LIST holding -(cl / v) * y[0]


sol = None       # YOUR CODE: solve_ivp from 0 to 24 h, starting from the full bolus,
                 #            reporting at T_GRID, with args=(CL_L_H, V_L)
conc_num = None  # YOUR CODE: concentration in mg/L at each time

print("solved ok:", None if sol is None else sol.success)
if conc_num is not None:
    print("conc at 0, 6, 12, 24 h:", np.round(conc_num[[0, 24, 48, 96]], 4))
''',
        solution='''
# Part a. The one compartment ODE, solved numerically after an IV bolus.

def rhs(t, y, cl, v):
    """dA/dt for one compartment. y[0] is the amount in mg."""
    return [-(cl / v) * y[0]]


sol = solve_ivp(rhs, (0.0, 24.0), [DOSE_MG], t_eval=T_GRID, args=(CL_L_H, V_L),
                rtol=1e-8, atol=1e-10)
conc_num = sol.y[0] / V_L

print("solved ok:", None if sol is None else sol.success)
if conc_num is not None:
    print("conc at 0, 6, 12, 24 h:", np.round(conc_num[[0, 24, 48, 96]], 4))
''',
    ),
    dict(
        label="b",
        prompt_md='''
## Part b. Check it against the answer you already know

An IV bolus into one compartment has an exact solution, so you can measure how
well the integration did rather than hoping.

```
C(t) = (dose / V) * exp(-(CL / V) * t)
```

Compute that on the same `T_GRID`, then take the largest absolute difference
between the two curves across all 97 points.

You should get something tiny, around ten to the minus eight. That number is
the whole point of this part: it is evidence the machinery is assembled
correctly. If instead you get something near five thousandths, your solve ran
on the default tolerances and `rtol` never got passed. When you move to a two compartment model next week, there
is no exact answer to compare against, and the only reason to trust the result
is that you checked the method somewhere you could.

If the error comes out large and roughly proportional to the values
themselves, suspect the volume division rather than the solver.

Fill in `conc_exact`, the analytic concentration at each time in `T_GRID`, and
`max_abs_err`, a single number.

Moves: `np.exp`, `np.abs`, `np.max`.
''',
        starter='''
# Part b. The exact solution, and how far the numerical one strayed.

conc_exact = None    # YOUR CODE: (dose / V) * exp(-(CL/V) * t) on T_GRID
max_abs_err = None   # YOUR CODE: largest absolute difference from conc_num

print("exact at 0, 6, 12, 24 h:",
      None if conc_exact is None else np.round(conc_exact[[0, 24, 48, 96]], 4))
print("max absolute error:", max_abs_err)
''',
        solution='''
# Part b. The exact solution, and how far the numerical one strayed.

conc_exact = (DOSE_MG / V_L) * np.exp(-(CL_L_H / V_L) * T_GRID)
max_abs_err = float(np.max(np.abs(conc_num - conc_exact)))

print("exact at 0, 6, 12, 24 h:",
      None if conc_exact is None else np.round(conc_exact[[0, 24, 48, 96]], 4))
print("max absolute error:", max_abs_err)
''',
    ),
    dict(
        label="c",
        prompt_md='''
## Part c. A 2 hour infusion, where the rate switches off

Same compartment, different input. Instead of arriving instantly, the drug
goes in at a constant `INF_RATE_MG_H` for the first `INF_DURATION_H` hours and
then stops. The compartment starts empty.

```
dA/dt = rate(t) - (CL / V) * A,   where rate(t) = R for t < 2, and 0 after
```

Write that as `rhs_inf(t, y, cl, v, rate, t_off)`, using an ordinary `if` on
`t`, which is a single number here rather than an array, so a plain `if` is
fine.

Integrate from 0 to 12 hours from an empty compartment, reporting at
`T_GRID_INF`, with the same tight tolerances, and **pass `max_step=0.05`**.

A word on why, because it is more interesting than it looks. With tolerances
this tight the solver is already taking small steps, so on this particular
problem it would probably get the right answer without `max_step`. You pass it
anyway, because "probably" depends on a step size you do not control: the
solver picks it by looking at how smooth the solution has been, and a
discontinuity is precisely the thing that history cannot warn it about. The
toolkit cell shows the failure on looser tolerances, where it bites hard.
Relying on tight tolerances to paper over a discontinuity is a bet that
happens to be winning here.

The peak must land exactly at the end of the infusion, because that is the
last moment drug is still arriving faster than it is leaving.

Fill in `sol_inf`, the solver result. `conc_inf`, the concentration in mg/L at
each time. `cmax_inf`, the highest concentration reached. And `t_cmax`, the
time at which it occurs.

Moves: `def rhs_inf(t, y, cl, v, rate, t_off)`, an `if` on `t`,
`solve_ivp(..., max_step=0.05)`, `np.max`, `np.argmax`.
''',
        starter='''
# Part c. A 2 hour infusion into the same compartment.

def rhs_inf(t, y, cl, v, rate, t_off):
    """dA/dt with a zero order input that stops at t_off."""
    return None   # YOUR CODE: input rate while t < t_off, then elimination only


sol_inf = None    # YOUR CODE: solve 0 to 12 h from an empty compartment,
                  #            report at T_GRID_INF, and pass max_step=0.05
conc_inf = None   # YOUR CODE: concentration in mg/L at each time
cmax_inf = None   # YOUR CODE: the peak concentration
t_cmax = None     # YOUR CODE: the time at which the peak occurs

print("solved ok:", None if sol_inf is None else sol_inf.success)
print("Cmax = %s mg/L at t = %s h" % (cmax_inf, t_cmax))
''',
        solution='''
# Part c. A 2 hour infusion into the same compartment.

def rhs_inf(t, y, cl, v, rate, t_off):
    """dA/dt with a zero order input that stops at t_off."""
    r = rate if t < t_off else 0.0
    return [r - (cl / v) * y[0]]


sol_inf = solve_ivp(rhs_inf, (0.0, 12.0), [0.0], t_eval=T_GRID_INF,
                    args=(CL_L_H, V_L, INF_RATE_MG_H, INF_DURATION_H),
                    max_step=0.05, rtol=1e-8, atol=1e-10)
conc_inf = sol_inf.y[0] / V_L
cmax_inf = float(np.max(conc_inf))
t_cmax = float(sol_inf.t[int(np.argmax(conc_inf))])

print("solved ok:", None if sol_inf is None else sol_inf.success)
print("Cmax = %s mg/L at t = %s h" % (cmax_inf, t_cmax))
''',
    ),
]


def build_expected(ns):
    import numpy as np
    cl, v = ns["CL_L_H"], ns["V_L"]
    grid, grid_inf = ns["T_GRID"], ns["T_GRID_INF"]
    exact = (ns["DOSE_MG"] / v) * np.exp(-(cl / v) * grid)
    rate, t_off = ns["INF_RATE_MG_H"], ns["INF_DURATION_H"]
    k = cl / v
    # exact infusion solution, during and after
    during = (rate / cl) * (1.0 - np.exp(-k * np.minimum(grid_inf, t_off)))
    after = during * np.exp(-k * np.maximum(grid_inf - t_off, 0.0))
    return {
        "n_grid": int(grid.size),
        "n_grid_inf": int(grid_inf.size),
        "conc_exact": [float(x) for x in exact],
        "amount": [float(x) for x in exact * v],
        "conc_inf": [float(x) for x in after],
        "amount_inf": [float(x) for x in after * v],
        "cmax_inf": float(np.max(after)),
        "t_cmax": float(t_off),
        "cmax_amount": float(np.max(after) * v),
        "t_end": 12.0,
    }


CHECK_CODE = '''
def _part_a():
    rhs = _need("rhs")
    sol = _need("sol")
    conc_num = _need("conc_num")

    if not callable(rhs):
        return "`rhs` should be a function."
    try:
        probe = rhs(0.0, [100.0], 4.2, 40.0)
    except Exception as exc:
        return "`rhs` raised %s when called as rhs(t, y, cl, v)." % type(exc).__name__
    if probe is None:
        raise _Todo("`rhs` returns None. Replace the # YOUR CODE line.")
    arr = _as_array(probe)
    if arr is None or arr.ndim == 0:
        return ("`rhs` should return a SEQUENCE of derivatives, one per state, so a list "
                "like [-(cl/v) * y[0]] rather than a bare number.")
    if not _np.isclose(float(_np.ravel(arr)[0]), -(4.2 / 40.0) * 100.0):
        if _np.isclose(float(_np.ravel(arr)[0]), (4.2 / 40.0) * 100.0):
            return "`rhs` has the wrong sign. The amount is falling, so the derivative is negative."
        return "`rhs` does not compute -(cl / v) * y[0]."

    if not hasattr(sol, "success"):
        return "`sol` should be the object solve_ivp returns."
    if not sol.success:
        return "`sol` reports success = False, so the integration failed."
    if _np.asarray(sol.t).size != _EXP["n_grid"]:
        return ("`sol` reported at %d times but T_GRID has %d. Pass t_eval=T_GRID so the "
                "output lands on the grid you want." % (_np.asarray(sol.t).size, _EXP["n_grid"]))

    arr = _as_array(conc_num)
    if arr is None:
        return "`conc_num` should be a numeric array."
    if arr.ndim == 2:
        return ("`conc_num` is two dimensional, so it is still sol.y. Take the first state "
                "with sol.y[0].")
    if arr.shape != (_EXP["n_grid"],):
        return "`conc_num` should have one value per time in T_GRID, so shape (%d,)." % _EXP["n_grid"]
    if _arr_close(arr, _EXP["amount"], rtol=1e-4):
        return ("`conc_num` is still an AMOUNT in mg. The state is mass, so divide by the "
                "volume to get a concentration in mg/L.")
    if not _arr_close(arr, _EXP["conc_exact"], rtol=1e-5, atol=1e-8):
        if _arr_close(arr, _EXP["conc_exact"], rtol=2e-2, atol=1e-3):
            return ("`conc_num` is close but only to about three digits, which is the "
                    "signature of solve_ivp running on its default tolerances. Pass "
                    "rtol=1e-8, atol=1e-10.")
        return ("`conc_num` does not match the expected profile. Check the starting amount "
                "is the full dose and that args passes CL then V.")
    return None


def _part_b():
    conc_exact = _need("conc_exact")
    max_abs_err = _need("max_abs_err")

    arr = _as_array(conc_exact)
    if arr is None:
        return "`conc_exact` should be a numeric array."
    if arr.shape != (_EXP["n_grid"],):
        return "`conc_exact` should have one value per time in T_GRID."
    if _arr_close(arr, _EXP["amount"], rtol=1e-6):
        return ("`conc_exact` is an amount in mg. Divide the dose by the volume before the "
                "exponential, so it is a concentration.")
    if not _arr_close(arr, _EXP["conc_exact"], rtol=1e-6):
        return ("`conc_exact` is not (dose / V) * exp(-(CL/V) * t). Check the rate constant "
                "is CL divided by V.")

    got = _as_scalar(max_abs_err)
    if got is None:
        return "`max_abs_err` should be a single number. np.max reduces the whole array."
    if got < 0:
        return "`max_abs_err` is negative, so the absolute value was not taken."
    if 1e-4 < got < 5e-2:
        return ("`max_abs_err` is %.4g. That is the signature of solve_ivp running on its "
                "default tolerances, which only ask for about three correct digits. Pass "
                "rtol=1e-8, atol=1e-10 to the solve in part a." % got)
    if got > 1e-4:
        return ("`max_abs_err` is %.4g, which is far too large. The two curves should agree "
                "to roughly 1e-8. Suspect the volume division or the starting amount." % got)
    return None


def _part_c():
    rhs_inf = _need("rhs_inf")
    sol_inf = _need("sol_inf")
    conc_inf = _need("conc_inf")
    cmax_inf = _need("cmax_inf")
    t_cmax = _need("t_cmax")

    if not callable(rhs_inf):
        return "`rhs_inf` should be a function."
    try:
        on = rhs_inf(0.5, [0.0], 4.2, 40.0, 250.0, 2.0)
        off = rhs_inf(5.0, [100.0], 4.2, 40.0, 250.0, 2.0)
    except Exception as exc:
        return ("`rhs_inf` raised %s when called as rhs_inf(t, y, cl, v, rate, t_off)."
                % type(exc).__name__)
    if on is None or off is None:
        raise _Todo("`rhs_inf` returns None. Replace the # YOUR CODE line.")
    on_v = float(_np.ravel(_as_array(on))[0])
    off_v = float(_np.ravel(_as_array(off))[0])
    if not _np.isclose(on_v, 250.0):
        return ("During the infusion, with an empty compartment, the derivative should be "
                "the input rate alone. Got %.4f rather than 250." % on_v)
    if not _np.isclose(off_v, -(4.2 / 40.0) * 100.0):
        if _np.isclose(off_v, 250.0 - (4.2 / 40.0) * 100.0):
            return ("After t_off the input should be zero, but the rate is still being "
                    "added. Check the condition is t < t_off.")
        return "After the infusion stops the derivative should be elimination only."

    if not hasattr(sol_inf, "success"):
        return "`sol_inf` should be the object solve_ivp returns."
    if not sol_inf.success:
        return "`sol_inf` reports success = False."
    if _np.asarray(sol_inf.t).size != _EXP["n_grid_inf"]:
        return ("`sol_inf` reported at %d times but T_GRID_INF has %d. Pass t_eval."
                % (_np.asarray(sol_inf.t).size, _EXP["n_grid_inf"]))

    arr = _as_array(conc_inf)
    if arr is None or arr.shape != (_EXP["n_grid_inf"],):
        return "`conc_inf` should have one value per time in T_GRID_INF."
    if _arr_close(arr, _EXP["amount_inf"], rtol=1e-3):
        return "`conc_inf` is an amount in mg. Divide by the volume."
    if not _arr_close(arr, _EXP["conc_inf"], rtol=1e-4, atol=1e-7):
        return ("`conc_inf` does not match the expected infusion profile. If it keeps "
                "climbing past 2 hours, the solver stepped over the switch off: pass "
                "max_step=0.05.")

    got = _as_scalar(cmax_inf)
    if got is None:
        return "`cmax_inf` should be a single number."
    if _num_close(got, _EXP["cmax_amount"], rtol=1e-3):
        return "`cmax_inf` is an amount in mg, not a concentration."
    if not _num_close(got, _EXP["cmax_inf"], rtol=1e-4):
        return "`cmax_inf` is not the peak of conc_inf."

    got = _as_scalar(t_cmax)
    if got is None:
        return "`t_cmax` should be a single number."
    if _num_close(got, float(_np.argmax(_np.asarray(_EXP["conc_inf"])))):
        return ("`t_cmax` is the index of the peak, not the time. Use it to look up the "
                "matching entry of sol_inf.t.")
    if not _num_close(got, _EXP["t_cmax"], rtol=1e-6, atol=1e-6):
        if got > _EXP["t_cmax"]:
            return ("`t_cmax` is after the infusion ended, which cannot happen: once the "
                    "input stops the concentration can only fall. Either the condition in "
                    "rhs_inf is not switching the rate off, or the solver stepped over the "
                    "switch off, which max_step=0.05 prevents.")
        return "`t_cmax` should be the moment the infusion stops, which is when the peak occurs."
    return None
'''

# Functions this problem introduces. Keys must exist in tools/function_catalog.py.
FUNCTIONS = [
    "solve_ivp",
    "t_eval",
    "max_step",
    "sol.y",
]
