PID = "W7D3"
WEEK = 7
DAY = 3
SLUG = "w7d3_pk_fit_one_compartment"
TITLE = "Fitting a one compartment oral model, and checking it against NCA"
TOPIC = "PK modelling: depot and central compartments, fitting an ODE model, model versus NCA"
EST = "20-30 min"

SCENARIO_MD = '''
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
'''

CONCEPT_MD = '''
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
'''

TASK_MD = '''
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
'''

DATA_CODE = '''
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
'''

TOOLKIT_MD = '''
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
'''

TOOLKIT_CODE = '''
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

# A tiny made up one compartment oral system. Nothing here is the answer.
demo_dose = 100.0


def demo_rhs(tt, y, ka, v, cl):
    """Two states: y[0] is the depot, y[1] is the central compartment."""
    absorb = ka * y[0]
    return [-absorb, absorb - (cl / v) * y[1]]


def demo_predict(tt, ka, v, cl):
    """Concentration at the times tt, for these parameters."""
    sol = solve_ivp(demo_rhs, (0.0, float(np.max(tt))), [demo_dose, 0.0],
                    t_eval=tt, args=(ka, v, cl), rtol=1e-8, atol=1e-10)
    return sol.y[1] / v


print("--- 1. solving the system for one set of parameters ---")
demo_t = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0])
demo_c_true = demo_predict(demo_t, 1.0, 30.0, 3.0)
print("times:", demo_t)
print("conc :", np.round(demo_c_true, 4))
print("mass balance at the end: depot + central should be below the dose")

print("")
print("--- 2. the two signatures take arguments in different orders ---")
print("rhs(t, y, ka, v, cl)      <- y is the state vector")
print("predict(t, ka, v, cl)     <- t is an array of output times")

print("")
print("--- 3. fitting it back ---")
demo_obs = demo_c_true * np.random.default_rng(1).lognormal(0, 0.05, demo_t.size)
demo_popt, _ = curve_fit(demo_predict, demo_t, demo_obs,
                         p0=[0.8, 25.0, 2.5],
                         bounds=([0.0, 0.0, 0.0], [np.inf, np.inf, np.inf]))
print("true parameters  : ka 1.000, V 30.000, CL 3.000")
print("recovered        : ka %.3f, V %.3f, CL %.3f" % tuple(demo_popt))

print("")
print("--- 4. what flip-flop looks like ---")
print("absorption rate ka  : %.4f 1/h" % demo_popt[0])
print("elimination CL/V    : %.4f 1/h" % (demo_popt[2] / demo_popt[1]))
print("ka is the larger, so absorption is fast and the terminal slope really")
print("is elimination. Were it the other way round, the slope you measured in")
print("NCA would be absorption wearing elimination's clothes.")

demo_grid = np.linspace(0, 12, 200)
demo_fig, demo_ax = plt.subplots(figsize=(5.5, 3.6))
demo_ax.plot(demo_grid, demo_predict(demo_grid, *demo_popt), label="fitted")
demo_ax.scatter(demo_t, demo_obs, color="black", zorder=3, label="observed")
demo_ax.set_xlabel("Time (h)"); demo_ax.set_ylabel("Made up units")
demo_ax.set_title("Toolkit demo, not the answer"); demo_ax.legend(frameon=False)
plt.show()
'''

PARTS = [
    dict(
        label="a",
        prompt_md='''
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
''',
        starter='''
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
''',
        solution='''
# Part a. The depot and central system, and a prediction function.

def rhs(t, y, ka, v, cl):
    """y[0] is the depot amount, y[1] the central amount, both in mg."""
    absorbed = ka * y[0]
    return [-absorbed, absorbed - (cl / v) * y[1]]


def predict(t_eval, ka, v, cl):
    """Concentration in mg/L at each time in t_eval."""
    t_eval = np.asarray(t_eval, dtype=float)
    sol = solve_ivp(rhs, (0.0, float(t_eval.max())), [DOSE_MG, 0.0],
                    t_eval=t_eval, args=(ka, v, cl), rtol=1e-8, atol=1e-10)
    return sol.y[1] / v


print("check at the observed times, with roughly sensible parameters:")
print(None if predict(t, 1.0, 50.0, 5.5) is None
      else np.round(predict(t, 1.0, 50.0, 5.5), 4))
print("observed:", c)
''',
    ),
    dict(
        label="b",
        prompt_md='''
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
''',
        starter='''
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
''',
        solution='''
# Part b. Fit ka, V/F and CL/F.

_p0 = [3.0 / TMAX, NCA_CL_F / NCA_LAMBDA_Z, NCA_CL_F]
popt, _pcov = curve_fit(predict, t, c, p0=_p0,
                        bounds=([0.0, 0.0, 0.0], [np.inf, np.inf, np.inf]))

ka_hat = float(popt[0])
v_hat = float(popt[1])
cl_hat = float(popt[2])
rmse = float(np.sqrt(np.mean((c - predict(t, *popt)) ** 2)))

print("ka   = %s 1/h   (true 1.15)" % ka_hat)
print("V/F  = %s L     (true 60.0)" % v_hat)
print("CL/F = %s L/h   (true 5.61)" % cl_hat)
print("rmse = %s mg/L" % rmse)
''',
    ),
    dict(
        label="c",
        prompt_md='''
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
''',
        starter='''
# Part c. Compare the model with NCA.

t_half_model = None   # YOUR CODE: ln(2) / (cl_hat / v_hat), in hours
cl_ratio = None       # YOUR CODE: cl_hat / NCA_CL_F
flip_flop = None      # YOUR CODE: True if ka_hat < cl_hat / v_hat

print("half-life, model : %s h" % t_half_model)
print("half-life, NCA   : %.4f h" % NCA_T_HALF)
print("CL ratio model/NCA: %s" % cl_ratio)
print("flip-flop?        : %s" % flip_flop)
''',
        solution='''
# Part c. Compare the model with NCA.

_kel_model = cl_hat / v_hat
t_half_model = float(np.log(2.0) / _kel_model)
cl_ratio = float(cl_hat / NCA_CL_F)
flip_flop = bool(ka_hat < _kel_model)

print("half-life, model : %s h" % t_half_model)
print("half-life, NCA   : %.4f h" % NCA_T_HALF)
print("CL ratio model/NCA: %s" % cl_ratio)
print("flip-flop?        : %s" % flip_flop)
''',
    ),
]


def build_expected(ns):
    import numpy as np
    from scipy.integrate import solve_ivp
    from scipy.optimize import curve_fit
    t, c, dose = ns["t"], ns["c"], ns["DOSE_MG"]

    def rhs(tt, y, ka, v, cl):
        a = ka * y[0]
        return [-a, a - (cl / v) * y[1]]

    def predict(te, ka, v, cl):
        te = np.asarray(te, dtype=float)
        sol = solve_ivp(rhs, (0.0, float(te.max())), [dose, 0.0], t_eval=te,
                        args=(ka, v, cl), rtol=1e-8, atol=1e-10)
        return sol.y[1] / v

    p0 = [3.0 / ns["TMAX"], ns["NCA_CL_F"] / ns["NCA_LAMBDA_Z"], ns["NCA_CL_F"]]
    popt, _ = curve_fit(predict, t, c, p0=p0,
                        bounds=([0.0, 0.0, 0.0], [np.inf, np.inf, np.inf]))
    pred = predict(t, *popt)
    kel = popt[2] / popt[1]
    return {
        "n_points": int(t.size),
        "probe_conc": [float(v) for v in predict(t, 1.0, 50.0, 5.5)],
        "popt": [float(v) for v in popt],
        "pred": [float(v) for v in pred],
        "rmse": float(np.sqrt(np.mean((c - pred) ** 2))),
        "mse": float(np.mean((c - pred) ** 2)),
        "kel_model": float(kel),
        "t_half_model": float(np.log(2.0) / kel),
        "t_half_wrong": float(1.0 / kel),
        "cl_ratio": float(popt[2] / ns["NCA_CL_F"]),
        "flip_flop": bool(popt[0] < kel),
    }


CHECK_CODE = '''
def _part_a():
    rhs = _need("rhs")
    predict = _need("predict")

    if not callable(rhs):
        return "`rhs` should be a function."
    try:
        got = rhs(0.0, [100.0, 50.0], 1.0, 50.0, 5.0)
    except Exception as exc:
        return "`rhs` raised %s when called as rhs(t, y, ka, v, cl)." % type(exc).__name__
    if got is None:
        raise _Todo("`rhs` returns None. Replace the # YOUR CODE line.")
    arr = _as_array(got)
    if arr is None or arr.size != 2:
        return "`rhs` should return TWO derivatives, one per compartment."
    want = [-100.0, 100.0 - (5.0 / 50.0) * 50.0]
    if not _np.allclose(arr.ravel(), want):
        if _np.allclose(arr.ravel(), [-100.0, -100.0 - (5.0 / 50.0) * 50.0]):
            return ("`rhs` has the absorption sign wrong in the central compartment. Drug "
                    "LEAVES the depot and ARRIVES in the central, so the same term appears "
                    "negative in one and positive in the other.")
        if _np.allclose(arr.ravel(), [100.0, -100.0 - (5.0 / 50.0) * 50.0]):
            return "`rhs` has both absorption signs flipped. The depot can only lose drug."
        return ("`rhs` does not match [-ka*y[0], ka*y[0] - (cl/v)*y[1]]. With y = [100, 50], "
                "ka 1, v 50, cl 5 it should give [-100, 95].")

    if not callable(predict):
        return "`predict` should be a function."
    try:
        got = predict(_need("t"), 1.0, 50.0, 5.5)
    except Exception as exc:
        return "`predict` raised %s when called as predict(t, ka, v, cl)." % type(exc).__name__
    if got is None:
        raise _Todo("`predict` returns None. Replace the # YOUR CODE line.")
    arr = _as_array(got)
    if arr is None or arr.shape != (_EXP["n_points"],):
        return ("`predict` should return one concentration per requested time, so shape "
                "(%d,) here." % _EXP["n_points"])
    if _arr_close(arr * 50.0, _EXP["probe_conc"], rtol=1e-4):
        return ("`predict` returned an AMOUNT in mg. Divide the central compartment by v to "
                "get a concentration.")
    if not _arr_close(arr, _EXP["probe_conc"], rtol=1e-4, atol=1e-8):
        return ("`predict` does not match. Start with the whole dose in the depot and "
                "nothing in the central, return sol.y[1] divided by v, and pass "
                "rtol=1e-8, atol=1e-10.")
    return None


def _part_b():
    popt = _need("popt")
    ka_hat = _need("ka_hat")
    v_hat = _need("v_hat")
    cl_hat = _need("cl_hat")
    rmse = _need("rmse")

    arr = _as_array(popt)
    if arr is None or arr.size != 3:
        return "`popt` should hold the three fitted parameters."
    if not _arr_close(arr, _EXP["popt"], rtol=5e-3):
        return ("`popt` does not match the fit. Check the p0 suggested in the prompt and "
                "that all three parameters are bounded below at zero.")

    for name, value, idx in (("ka_hat", ka_hat, 0), ("v_hat", v_hat, 1), ("cl_hat", cl_hat, 2)):
        got = _as_scalar(value)
        if got is None:
            return "`%s` should be a single number." % name
        if not _num_close(got, _EXP["popt"][idx], rtol=5e-3):
            return "`%s` is not popt[%d]." % (name, idx)

    got = _as_scalar(rmse)
    if got is None:
        return "`rmse` should be a single number."
    if _num_close(got, _EXP["mse"], rtol=1e-3):
        return "`rmse` is the mean squared error. Take its square root."
    if not _num_close(got, _EXP["rmse"], rtol=1e-2):
        return "`rmse` is not sqrt(mean((c - predict(t, *popt)) ** 2))."
    return None


def _part_c():
    t_half_model = _need("t_half_model")
    cl_ratio = _need("cl_ratio")
    flip_flop = _need("flip_flop")

    got = _as_scalar(t_half_model)
    if got is None:
        return "`t_half_model` should be a single number."
    if _num_close(got, _EXP["t_half_wrong"], rtol=1e-3):
        return "`t_half_model` is 1/kel. A half-life carries a factor of ln(2)."
    if _num_close(got, _np.log(2.0) / _EXP["popt"][0], rtol=1e-3):
        return ("`t_half_model` used ka rather than the elimination rate constant. The rate "
                "constant out of the central compartment is cl_hat / v_hat.")
    if not _num_close(got, _EXP["t_half_model"], rtol=5e-3):
        return "`t_half_model` is not np.log(2) divided by cl_hat / v_hat."

    got = _as_scalar(cl_ratio)
    if got is None:
        return "`cl_ratio` should be a single number."
    if _num_close(got, 1.0 / _EXP["cl_ratio"], rtol=1e-3):
        return "`cl_ratio` is the other way up. It is the model value over the NCA value."
    if not _num_close(got, _EXP["cl_ratio"], rtol=5e-3):
        return "`cl_ratio` is not cl_hat divided by NCA_CL_F."

    if not isinstance(flip_flop, (bool, _np.bool_)):
        val = _as_scalar(flip_flop)
        if val is None:
            return "`flip_flop` should be a single True or False."
        flip_flop = bool(val)
    if bool(flip_flop) != _EXP["flip_flop"]:
        return ("`flip_flop` should be %s here. The test is whether ka_hat is LESS than the "
                "elimination rate constant cl_hat / v_hat, which would mean absorption is "
                "the slower process." % _EXP["flip_flop"])
    return None
'''

# Functions this problem introduces. Keys must exist in tools/function_catalog.py.
FUNCTIONS = [
    "depot and central",
    "fitting an ODE model",
    "flip-flop kinetics",
]
