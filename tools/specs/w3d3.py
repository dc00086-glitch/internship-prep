PID = "W3D3"
WEEK = 3
DAY = 3
SLUG = "w3d3_scipy_curve_fit"
TITLE = "Fitting a decay curve, and how sure you are of the parameters"
TOPIC = "scipy.optimize: curve_fit, starting values, parameter standard errors"
EST = "20-30 min"

SCENARIO_MD = '''
## Scenario

One subject received a 500 mg intravenous bolus and was sampled nine times
over 24 hours. An IV bolus into a single compartment gives the simplest curve
in all of pharmacokinetics:

```
C(t) = C0 * exp(-k * t)
```

Two parameters. `C0` is the concentration the drug would have had at time zero
if mixing were instant, and `k` is the elimination rate constant.

In week 1 you got a rate constant by taking logs and fitting a straight line.
That works, but it quietly assumes the error is constant **on the log scale**,
and it cannot give you an uncertainty on `C0` in the original units. Fitting
the curve directly by least squares is the general tool, and it comes with the
thing a straight line fit on logs does not hand you: a standard error for each
parameter.
'''

CONCEPT_MD = '''
## curve_fit, starting values, and the covariance matrix

**What curve_fit does.** You hand it a model function, your x values and your
y values, and it searches for the parameter values that minimise the sum of
squared differences between model and data. The model function must take the
independent variable **first** and the parameters after it:

```python
def decay(t, c0, k):
    return c0 * np.exp(-k * t)
```

curve_fit reads the parameter names and count from that signature, so the
order you write them in is the order they come back in.

**Why p0 matters more than people expect.** The search is local. It walks
downhill from wherever you start, and if you do not say where to start it
assumes every parameter is 1.0. For this model that means starting at
`C0 = 1 mg/L` when the truth is near 50, and `k = 1 /h` when the truth is near
0.1. Sometimes it recovers. Sometimes it walks into a flat region and returns
something absurd, or raises `RuntimeError: Optimal parameters not found`.

A starting guess does not need to be good, only the right order of magnitude,
and you can almost always read one off the data: `C0` is roughly the first
concentration, and `k` is roughly `ln(2)` over an eyeballed half-life. Thirty
seconds of looking saves a confusing failure.

**The second return value is the useful one.** `curve_fit` returns `popt`, the
fitted parameters, and `pcov`, their covariance matrix. The diagonal of `pcov`
holds the variance of each parameter, so

```python
perr = np.sqrt(np.diag(pcov))
```

gives one standard error per parameter, in the parameter's own units. That is
what turns "k is 0.104" into "k is 0.104 with a standard error of 0.004",
which is the difference between a number and a result.

The off-diagonal entries matter too: they say how much the parameters trade
off against each other. A strong correlation between `C0` and `k` means the
data cannot separate them well, and that many different pairs fit almost as
well as the best one.

**When to distrust `pcov`.** If it comes back full of `inf`, the fit did not
converge properly or the model has more parameters than the data can support.
That is information, not a nuisance to round away.
'''

TASK_MD = '''
## Your task

Three parts, worked one at a time.

| Part | What you work out | Variables to fill in |
|---|---|---|
| a | the model function, and the fit | `decay`, `popt`, `pcov` |
| b | the parameters and their standard errors | `c0_hat`, `k_hat`, `se_c0`, `se_k` |
| c | what the fit implies, and how well it fits | `t_half`, `pred`, `rmse` |

Below this you will find the **Setup** cell, then a **Toolkit** cell that
demonstrates every move on a tiny made up example, then **Part a**, **Part b**
and **Part c**, each with its own explanation and a cell to write in.
'''

DATA_CODE = '''
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

PROBLEM_ID = "W3D3"
TOPIC = "scipy.optimize: curve_fit, starting values, parameter standard errors"

rng = np.random.default_rng(3103)

DOSE_MG = 500.0
time_h = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 18.0, 24.0])

# The truth the fit is trying to recover: C0 = 52 mg/L, k = 0.105 /h.
_true_c0, _true_k = 52.0, 0.105
conc = np.round(_true_c0 * np.exp(-_true_k * time_h)
                * rng.lognormal(0.0, 0.07, size=time_h.size), 3)

obs = pd.DataFrame({"TIME_H": time_h, "CONC_MG_L": conc})

print("dose:", DOSE_MG, "mg IV bolus")
print(obs.to_string(index=False))
print("")
print("first concentration %.3f mg/L, last %.3f mg/L" % (conc[0], conc[-1]))
'''

TOOLKIT_MD = '''
## Toolkit for this problem

| What you want to do | How |
|---|---|
| import the fitter | `from scipy.optimize import curve_fit` |
| define a model | `def f(x, a, b): return a * np.exp(-b * x)`, x first |
| fit it | `popt, pcov = curve_fit(f, xdata, ydata, p0=[guess_a, guess_b])` |
| the fitted parameters | `popt`, in the order of the function signature |
| standard errors | `np.sqrt(np.diag(pcov))` |
| the diagonal of a matrix | `np.diag(m)` |
| predictions at the data points | `f(xdata, *popt)` |
| root mean squared error | `np.sqrt(np.mean(resid ** 2))` |

`f(xdata, *popt)` unpacks the fitted array into the function's arguments, so
it works whatever the number of parameters.
'''

TOOLKIT_CODE = '''
import numpy as np
from scipy.optimize import curve_fit

# A tiny made up decay. Nothing here is the answer.
demo_x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0])
demo_y = np.array([9.8, 7.3, 5.3, 4.1, 3.0, 1.6, 0.9])
print("demo_x:", demo_x)
print("demo_y:", demo_y)


def demo_model(x, a, b):
    """The independent variable comes FIRST, then the parameters."""
    return a * np.exp(-b * x)


print("")
print("--- 1. a fit with sensible starting values ---")
demo_popt, demo_pcov = curve_fit(demo_model, demo_x, demo_y, p0=[10.0, 0.3])
print("popt :", np.round(demo_popt, 5), " <- in the order (a, b) from the signature")
print("pcov shape:", demo_pcov.shape, " <- one row and column per parameter")

print("")
print("--- 2. standard errors come from the diagonal ---")
demo_perr = np.sqrt(np.diag(demo_pcov))
print("np.diag(pcov) :", np.round(np.diag(demo_pcov), 8))
print("standard errors:", np.round(demo_perr, 5))
print("so a = %.4f +/- %.4f and b = %.4f +/- %.4f"
      % (demo_popt[0], demo_perr[0], demo_popt[1], demo_perr[1]))

print("")
print("--- 3. the off diagonal says how the parameters trade off ---")
demo_corr = demo_pcov[0, 1] / np.sqrt(demo_pcov[0, 0] * demo_pcov[1, 1])
print("correlation between a and b: %.3f" % demo_corr)
print("close to zero means the data separates them well")

print("")
print("--- 4. what a bad starting guess can do ---")
try:
    demo_bad, _ = curve_fit(demo_model, demo_x, demo_y, p0=[1.0, 50.0], maxfev=2000)
    print("from p0=[1, 50] it returned:", np.round(demo_bad, 5))
    print("compare with the good fit    :", np.round(demo_popt, 5))
except RuntimeError as err:
    print("from p0=[1, 50] it failed outright:", err)
print("")
print("the default p0 is all ones, which is rarely the right order of magnitude")

print("")
print("--- 5. predictions and residuals ---")
demo_pred = demo_model(demo_x, *demo_popt)
demo_resid = demo_y - demo_pred
print("predicted :", np.round(demo_pred, 3))
print("residuals :", np.round(demo_resid, 3))
print("rmse      : %.4f" % float(np.sqrt(np.mean(demo_resid ** 2))))
'''

PARTS = [
    dict(
        label="a",
        prompt_md='''
## Part a. Write the model and fit it

Define `decay(t, c0, k)` returning `c0 * np.exp(-k * t)`. The independent
variable has to come first and the parameters after it, in the order you want
them returned.

Then fit it to the observed data with `curve_fit`, passing a starting guess.
Read the guess off the data rather than inventing one: the first observed
concentration is a fair stand-in for `C0`, and the profile looks like it
roughly halves every seven hours or so, which gives `k` near `ln(2) / 7`.
Anything in the right order of magnitude will do.

Keep both return values. The second one, `pcov`, is what part b needs, and
discarding it is the most common way people end up reporting a parameter with
no uncertainty attached.

Fill in `decay`, the model function. `popt`, the fitted parameters. And
`pcov`, the covariance matrix.

Moves: `def`, `np.exp`, `curve_fit(f, xdata, ydata, p0=[...])`.
''',
        starter='''
# Part a. Define the model, then fit it.

def decay(t, c0, k):
    """Concentration after an IV bolus into one compartment."""
    return None   # YOUR CODE: c0 * exp(-k * t)


popt = None   # YOUR CODE: fitted parameters from curve_fit, with a sensible p0
pcov = None   # YOUR CODE: the covariance matrix curve_fit also returns

print("popt:", popt)
print("pcov:", pcov)
''',
        solution='''
# Part a. Define the model, then fit it.

def decay(t, c0, k):
    """Concentration after an IV bolus into one compartment."""
    return c0 * np.exp(-k * t)


popt, pcov = curve_fit(decay, obs["TIME_H"].to_numpy(), obs["CONC_MG_L"].to_numpy(),
                       p0=[conc[0], np.log(2.0) / 7.0])

print("popt:", popt)
print("pcov:", pcov)
''',
    ),
    dict(
        label="b",
        prompt_md='''
## Part b. Pull out the parameters and their standard errors

`popt` holds the two fitted values in the order of your function signature, so
`popt[0]` is `C0` and `popt[1]` is `k`.

The standard errors are the square roots of the diagonal of `pcov`. Take the
diagonal with `np.diag`, then `np.sqrt` it, and you get one standard error per
parameter in that parameter's own units: mg/L for `C0`, per hour for `k`.

Look at the ratio of each estimate to its standard error when you have them.
A parameter whose estimate is many times its standard error is well determined
by the data. One whose standard error is a sizeable fraction of the estimate
is not, and quoting it to three decimal places would be false precision.

Fill in `c0_hat`, `k_hat`, `se_c0`, `se_k`, all plain floats.

Moves: indexing `popt`, `np.diag`, `np.sqrt`, `float(...)`.
''',
        starter='''
# Part b. The fitted parameters and their standard errors.

c0_hat = None   # YOUR CODE: fitted C0, mg/L
k_hat = None    # YOUR CODE: fitted k, 1/h
se_c0 = None    # YOUR CODE: standard error of C0
se_k = None     # YOUR CODE: standard error of k

if None not in (c0_hat, k_hat, se_c0, se_k):
    print("C0 = %.4f +/- %.4f mg/L" % (c0_hat, se_c0))
    print("k  = %.5f +/- %.5f 1/h" % (k_hat, se_k))
    print("estimate over standard error: C0 %.1f, k %.1f"
          % (c0_hat / se_c0, k_hat / se_k))
''',
        solution='''
# Part b. The fitted parameters and their standard errors.

_perr = np.sqrt(np.diag(pcov))

c0_hat = float(popt[0])
k_hat = float(popt[1])
se_c0 = float(_perr[0])
se_k = float(_perr[1])

if None not in (c0_hat, k_hat, se_c0, se_k):
    print("C0 = %.4f +/- %.4f mg/L" % (c0_hat, se_c0))
    print("k  = %.5f +/- %.5f 1/h" % (k_hat, se_k))
    print("estimate over standard error: C0 %.1f, k %.1f"
          % (c0_hat / se_c0, k_hat / se_k))
''',
    ),
    dict(
        label="c",
        prompt_md='''
## Part c. What the fit implies, and how well it fits

Two different questions, and it is worth keeping them apart.

**What it implies.** The half-life follows from `k` exactly as it did in week
1: `ln(2) / k`. This is the number that goes in a report.

**How well it fits.** Predict the concentration at each observed time by
calling your model with the fitted parameters, `decay(times, *popt)`, which
unpacks the array into the function's arguments. Subtract to get residuals,
and summarise them with the root mean squared error, `sqrt(mean(resid**2))`.

A small RMSE alone does not mean the model is right, which is worth saying
plainly. Residuals should also look like noise: scattered either side of zero
with no pattern. A model that is systematically high early and low late can
still post a respectable RMSE while being the wrong shape, and the way you
catch that is by looking at the residuals rather than at their summary.

Fill in `t_half` in hours, `pred`, the predicted concentration at each
observed time, and `rmse`, a plain float.

Moves: `np.log(2) / k`, `decay(times, *popt)`, `np.sqrt`, `np.mean`.
''',
        starter='''
# Part c. Half-life, predictions, and the residual summary.

t_half = None   # YOUR CODE: ln(2) / k, in hours
pred = None     # YOUR CODE: predicted concentration at each observed time
rmse = None     # YOUR CODE: root mean squared error of the residuals

print("half-life : %s h" % t_half)
print("predicted :", None if pred is None else np.round(pred, 3))
print("observed  :", obs["CONC_MG_L"].to_numpy())
print("rmse      : %s" % rmse)
''',
        solution='''
# Part c. Half-life, predictions, and the residual summary.

t_half = float(np.log(2.0) / k_hat)
pred = decay(obs["TIME_H"].to_numpy(), *popt)
_resid = obs["CONC_MG_L"].to_numpy() - pred
rmse = float(np.sqrt(np.mean(_resid ** 2)))

print("half-life : %s h" % t_half)
print("predicted :", None if pred is None else np.round(pred, 3))
print("observed  :", obs["CONC_MG_L"].to_numpy())
print("rmse      : %s" % rmse)
''',
    ),
]


def build_expected(ns):
    import numpy as np
    popt = np.asarray(ns["popt"], dtype=float)
    pcov = np.asarray(ns["pcov"], dtype=float)
    perr = np.sqrt(np.diag(pcov))
    t = ns["obs"]["TIME_H"].to_numpy()
    y = ns["obs"]["CONC_MG_L"].to_numpy()
    pred = popt[0] * np.exp(-popt[1] * t)
    resid = y - pred
    return {
        "popt": [float(v) for v in popt],
        "perr": [float(v) for v in perr],
        "pcov_diag": [float(v) for v in np.diag(pcov)],
        "t_half": float(np.log(2.0) / popt[1]),
        "t_half_wrong": float(1.0 / popt[1]),
        "pred": [float(v) for v in pred],
        "obs_conc": [float(v) for v in y],
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "mae": float(np.mean(np.abs(resid))),
        "sse": float(np.sum(resid ** 2)),
        "n_points": int(t.size),
    }


CHECK_CODE = '''
def _part_a():
    decay = _need("decay")
    popt = _need("popt")
    pcov = _need("pcov")

    if not callable(decay):
        return "`decay` should be a function."
    try:
        probe = decay(_np.array([0.0, 1.0]), 10.0, 0.5)
    except Exception as exc:
        return "`decay` raised %s when called as decay(t, c0, k)." % type(exc).__name__
    if probe is None:
        return "`decay` returns None. Replace the # YOUR CODE line with the formula."
    probe = _np.asarray(probe, dtype=float)
    if probe.shape != (2,):
        return ("`decay` should return one value per time, so it must work on a whole "
                "array. Use np.exp rather than math.exp.")
    if not _np.allclose(probe, [10.0, 10.0 * _np.exp(-0.5)]):
        if _np.allclose(probe, [10.0, 10.0 * _np.exp(0.5)]):
            return ("`decay` is growing rather than decaying. The exponent needs the minus "
                    "sign: c0 * np.exp(-k * t).")
        return "`decay` does not compute c0 * exp(-k * t)."

    arr = _as_array(popt)
    if arr is None or arr.shape != (2,):
        return ("`popt` should hold the two fitted parameters. curve_fit returns a pair, so "
                "unpack it as popt, pcov = curve_fit(...).")
    if not _arr_close(arr, _EXP["popt"], rtol=1e-3):
        if _arr_close(arr[::-1], _EXP["popt"], rtol=1e-3):
            return ("`popt` has the parameters the other way round, so your function "
                    "signature is decay(t, k, c0). The task asks for c0 first.")
        return ("`popt` is not the least squares fit. Check the model formula, and pass a "
                "p0 in the right order of magnitude: C0 near the first concentration, k "
                "near ln(2) over an eyeballed half-life.")

    cov = _as_array(pcov)
    if cov is None or cov.shape != (2, 2):
        return ("`pcov` should be the 2 by 2 covariance matrix, which is the second thing "
                "curve_fit returns.")
    if not _np.all(_np.isfinite(cov)):
        return ("`pcov` contains inf, which means the fit did not pin the parameters down. "
                "Usually a starting guess problem.")
    if not _arr_close(_np.diag(cov), _EXP["pcov_diag"], rtol=1e-2):
        return "`pcov` does not match the covariance of this fit."
    return None


def _part_b():
    c0_hat = _need("c0_hat")
    k_hat = _need("k_hat")
    se_c0 = _need("se_c0")
    se_k = _need("se_k")

    got = _as_scalar(c0_hat)
    if got is None:
        return "`c0_hat` should be a single number, taken from popt."
    if _num_close(got, _EXP["popt"][1], rtol=1e-6):
        return "`c0_hat` is the rate constant. popt[0] is C0, popt[1] is k."
    if not _num_close(got, _EXP["popt"][0], rtol=1e-3):
        return "`c0_hat` is not the fitted C0."

    got = _as_scalar(k_hat)
    if got is None:
        return "`k_hat` should be a single number."
    if not _num_close(got, _EXP["popt"][1], rtol=1e-3):
        return "`k_hat` is not the fitted k."

    got = _as_scalar(se_c0)
    if got is None:
        return "`se_c0` should be a single number."
    if _num_close(got, _EXP["pcov_diag"][0], rtol=1e-3):
        return ("`se_c0` is the variance, not the standard error. Take the square root of "
                "the diagonal entry.")
    if _num_close(got, _EXP["perr"][1], rtol=1e-3):
        return "`se_c0` is the standard error of k. The diagonal is in parameter order."
    if not _num_close(got, _EXP["perr"][0], rtol=1e-2):
        return "`se_c0` is not sqrt of the first diagonal entry of pcov."

    got = _as_scalar(se_k)
    if got is None:
        return "`se_k` should be a single number."
    if _num_close(got, _EXP["pcov_diag"][1], rtol=1e-3):
        return "`se_k` is the variance, not the standard error. Take the square root."
    if not _num_close(got, _EXP["perr"][1], rtol=1e-2):
        return "`se_k` is not sqrt of the second diagonal entry of pcov."
    return None


def _part_c():
    t_half = _need("t_half")
    pred = _need("pred")
    rmse = _need("rmse")

    got = _as_scalar(t_half)
    if got is None:
        return "`t_half` should be a single number."
    if _num_close(got, _EXP["t_half_wrong"], rtol=1e-4):
        return ("`t_half` is 1/k, which is the mean residence time style number, not the "
                "half-life. A half-life carries a factor of ln(2).")
    if not _num_close(got, _EXP["t_half"], rtol=1e-3):
        return "`t_half` is not np.log(2) divided by your fitted k."

    arr = _as_array(pred)
    if arr is None:
        return "`pred` should be a numeric array."
    if arr.shape != (_EXP["n_points"],):
        return ("`pred` has shape %s but there are %d observed timepoints, so there should "
                "be one prediction each." % (arr.shape, _EXP["n_points"]))
    if _arr_close(arr, _EXP["obs_conc"]):
        return ("`pred` is the observed data, not the model prediction. Call your decay "
                "function with the fitted parameters.")
    if not _arr_close(arr, _EXP["pred"], rtol=1e-3):
        return ("`pred` does not match decay(times, *popt). Make sure you evaluated at the "
                "observed times and used the fitted parameters.")

    got = _as_scalar(rmse)
    if got is None:
        return "`rmse` should be a single number."
    if _num_close(got, _EXP["sse"], rtol=1e-3):
        return ("`rmse` is the sum of squared residuals. Take the mean of the squares, then "
                "the square root.")
    if _num_close(got, _EXP["mae"], rtol=1e-3):
        return ("`rmse` is the mean absolute error. Square the residuals, average, then "
                "square root.")
    if not _num_close(got, _EXP["rmse"], rtol=1e-3):
        return "`rmse` is not sqrt(mean(residuals ** 2))."
    return None
'''

# Functions this problem introduces. Keys must exist in tools/function_catalog.py.
FUNCTIONS = [
    "curve_fit",
    "p0",
    "np.diag",
    "argument unpacking",
]
