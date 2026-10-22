PID = "W3D5"
WEEK = 3
DAY = 5
SLUG = "w3d5_scipy_emax_hill"
TITLE = "Emax and sigmoid Emax: fitting a concentration response curve"
TOPIC = "scipy.optimize: Emax models, bounds, and comparing nested fits"
EST = "20-30 min"

SCENARIO_MD = '''
## Scenario

An in vitro assay measured percent inhibition of a target enzyme across ten
compound concentrations spanning four orders of magnitude, three replicate
wells at each concentration.

The relationship between concentration and effect is not a straight line and
never will be. There is a floor, because zero drug gives zero effect, and a
ceiling, because once every enzyme molecule is occupied there is nothing left
to inhibit. In between the curve rises. That shape is the Emax model, and it
is the most used equation in pharmacology:

```
E = Emax * C / (EC50 + C)
```

`Emax` is the ceiling and `EC50` is the concentration giving half of it. Many
real curves rise more steeply than this allows, so a third parameter is added,
the Hill coefficient, giving the sigmoid Emax model. Your job is to fit both
and work out whether the extra parameter earns its place.
'''

CONCEPT_MD = '''
## Two curves, three parameters, and the trap in comparing them

**The Emax model.** `E = Emax * C / (EC50 + C)` is the same hyperbola as
Michaelis-Menten kinetics and as simple receptor occupancy, which is not a
coincidence: all three describe a fixed number of binding sites filling up.
At `C = EC50` the effect is exactly half of `Emax`, which is what makes EC50
interpretable. Note it is a **concentration**, in the same units as `C`, so it
belongs on a log axis and is reported as a single number, not a percentage.

**The sigmoid Emax model** adds a Hill coefficient `h`:

```
E = Emax * C**h / (EC50**h + C**h)
```

With `h = 1` it collapses back to the plain Emax model, which makes the two
**nested**: the simpler one is a special case of the richer one. `h` controls
steepness. Above 1 the curve switches on more abruptly, which can come from
cooperative binding, from several binding events being needed, or simply from
the assay compressing its own dynamic range. Below 1 it is shallower than mass
action alone would give.

**Why fitting this needs care.** Concentrations spanning four orders of
magnitude mean `C**h` spans sixteen, and the optimiser can easily wander
somewhere `EC50` goes negative, which is meaningless, or `h` explodes.
`bounds=` keeps the search in the region where the parameters mean something:
all three are positive, `Emax` cannot sensibly exceed 100 percent inhibition
by much. Bounds are not cheating, they are stating what you already know.

Start `EC50` near the concentration where the response passes half its
maximum, read off the data by eye. Starting it at the default 1.0 when the
truth is 0.0008 is how fits fail.

**The trap in comparing them.** A model with more parameters **always** fits
the data it was fitted to at least as well. The sigmoid will never have a
larger residual sum of squares than the plain Emax, so "the RMSE went down"
is not evidence of anything, it is arithmetic. Judging an extra parameter
needs something that charges for it: an F test for nested models, an
information criterion like AIC, or held-out data. You will compute both RMSEs
here so you can see the improvement, and the point of the exercise is knowing
that the improvement alone does not settle the question.
'''

TASK_MD = '''
## Your task

Three parts, worked one at a time.

| Part | What you work out | Variables to fill in |
|---|---|---|
| a | the replicates aggregated to one point per concentration | `agg` |
| b | the plain Emax fit | `emax_model`, `emax1`, `ec50_1`, `rmse1` |
| c | the sigmoid Emax fit, and the comparison | `sigmoid_model`, `emax2`, `ec50_2`, `hill`, `rmse2` |

Below this you will find the **Setup** cell, then a **Toolkit** cell that
demonstrates every move on a tiny made up example, then **Part a**, **Part b**
and **Part c**, each with its own explanation and a cell to write in.
'''

DATA_CODE = '''
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

PROBLEM_ID = "W3D5"
TOPIC = "scipy.optimize: Emax models, bounds, and comparing nested fits"

rng = np.random.default_rng(3105)

# Ten concentrations, evenly spaced on a log scale, three wells each.
CONCS = np.round(np.logspace(-2.0, 2.0, 10), 4)      # micromolar
N_REPLICATES = 3

# The truth: a ceiling near 96 percent, EC50 of 0.8 uM, and a steep Hill slope.
_TRUE_EMAX, _TRUE_EC50, _TRUE_H = 96.0, 0.80, 1.7
_signal = _TRUE_EMAX * CONCS ** _TRUE_H / (_TRUE_EC50 ** _TRUE_H + CONCS ** _TRUE_H)

_rows = []
for _c, _mu in zip(CONCS, _signal):
    for _rep in range(1, N_REPLICATES + 1):
        _rows.append((_c, _rep, round(float(np.clip(_mu + rng.normal(0.0, 3.2), 0.0, 100.0)), 2)))

assay = pd.DataFrame(_rows, columns=["CONC_UM", "REPLICATE", "INHIB_PCT"])

print("assay:", assay.shape, " ten concentrations, three wells each")
print(assay.head(9).to_string(index=False))
print("")
print("concentrations:", CONCS)
'''

TOOLKIT_MD = '''
## Toolkit for this problem

| What you want to do | How |
|---|---|
| mean and SD per concentration | `.groupby("CONC_UM").agg(M=("E", "mean"), S=("E", "std"))` |
| the Emax model | `emax * c / (ec50 + c)` |
| the sigmoid Emax model | `emax * c ** h / (ec50 ** h + c ** h)` |
| fit with a starting guess | `curve_fit(f, x, y, p0=[...])` |
| keep parameters sensible | `curve_fit(..., bounds=([0, 0], [200, 1000]))` |
| predictions | `f(x, *popt)` |
| root mean squared error | `np.sqrt(np.mean((y - pred) ** 2))` |
| log spaced values | `np.logspace(-2, 2, 10)` |

`bounds` takes a pair of sequences, lower bounds then upper bounds, one entry
per parameter in signature order. Use `np.inf` for an upper bound you do not
want to set.
'''

TOOLKIT_CODE = '''
import numpy as np
from scipy.optimize import curve_fit

# A tiny made up concentration response. Nothing here is the answer.
demo_c = np.array([0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0])
demo_e = np.array([0.4, 4.0, 11.0, 29.0, 52.0, 68.0, 74.0])
print("demo_c:", demo_c)
print("demo_e:", demo_e)


def demo_emax(c, emax, ec50):
    """Plain Emax. Half the ceiling is reached at c = ec50."""
    return emax * c / (ec50 + c)


def demo_sigmoid(c, emax, ec50, h):
    """Sigmoid Emax. h = 1 gives the plain model back."""
    return emax * c ** h / (ec50 ** h + c ** h)


print("")
print("--- 1. reading starting values off the data ---")
demo_top = demo_e.max()
demo_half = demo_c[int(np.argmin(np.abs(demo_e - demo_top / 2.0)))]
print("highest response seen : %.1f  <- a guess for Emax" % demo_top)
print("concentration nearest half of that: %.3f  <- a guess for EC50" % demo_half)

print("")
print("--- 2. fitting both models ---")
demo_p1, _ = curve_fit(demo_emax, demo_c, demo_e, p0=[demo_top, demo_half],
                       bounds=([0.0, 1e-6], [200.0, 1e4]))
demo_p2, _ = curve_fit(demo_sigmoid, demo_c, demo_e, p0=[demo_top, demo_half, 1.0],
                       bounds=([0.0, 1e-6, 0.1], [200.0, 1e4, 10.0]))
print("Emax model   : Emax = %.2f  EC50 = %.4f" % (demo_p1[0], demo_p1[1]))
print("sigmoid model: Emax = %.2f  EC50 = %.4f  h = %.3f" % (demo_p2[0], demo_p2[1], demo_p2[2]))

print("")
print("--- 3. the extra parameter ALWAYS fits at least as well ---")
demo_r1 = float(np.sqrt(np.mean((demo_e - demo_emax(demo_c, *demo_p1)) ** 2)))
demo_r2 = float(np.sqrt(np.mean((demo_e - demo_sigmoid(demo_c, *demo_p2)) ** 2)))
print("rmse, Emax model    : %.4f" % demo_r1)
print("rmse, sigmoid model : %.4f" % demo_r2)
print("sigmoid is never worse: %s" % (demo_r2 <= demo_r1 + 1e-9))
print("")
print("that is guaranteed by the models being nested, so it is not evidence")
print("that the third parameter is real")

print("")
print("--- 4. what bounds are for ---")
try:
    demo_bad, _ = curve_fit(demo_sigmoid, demo_c, demo_e, p0=[1.0, 1.0, 1.0], maxfev=4000)
    print("unbounded from a poor p0:", np.round(demo_bad, 4))
    print("an EC50 at or below zero would be meaningless, which bounds prevent")
except RuntimeError as err:
    print("unbounded from a poor p0 failed:", err)
'''

PARTS = [
    dict(
        label="a",
        prompt_md='''
## Part a. Aggregate the replicate wells

Three wells were run at each concentration, so there are three rows per
concentration. Reduce them to one point each before fitting.

This is the same unit-of-replication idea as week 1, in a different costume.
The wells are technical replicates of one condition, not independent samples
of the concentration response, so the curve is fitted through the ten
condition means rather than through thirty wells. Fitting all thirty would
weight each concentration by how many wells happened to survive, and would
make the fit look more certain than it is.

Keep the SD as well. You are not using it in the fit here, but a concentration
whose replicates disagree wildly is one you want to notice.

Fill in `agg`, sorted by concentration, index renumbered, with exactly:

| column | meaning |
|---|---|
| `CONC_UM` | concentration in micromolar |
| `MEAN_PCT` | mean percent inhibition across the wells |
| `SD_PCT` | sample SD across the wells |
| `N` | how many wells |

Moves: `.groupby("CONC_UM").agg(NAME=("COL", "mean"), ...)`,
`.reset_index()`, `.sort_values(...)`.
''',
        starter='''
# Part a. One point per concentration.

agg = None   # YOUR CODE: CONC_UM, MEAN_PCT, SD_PCT, N

print(agg)
''',
        solution='''
# Part a. One point per concentration.

agg = (assay
       .groupby("CONC_UM")
       .agg(MEAN_PCT=("INHIB_PCT", "mean"),
            SD_PCT=("INHIB_PCT", "std"),
            N=("INHIB_PCT", "size"))
       .reset_index()
       .sort_values("CONC_UM")
       .reset_index(drop=True))

print(agg)
''',
    ),
    dict(
        label="b",
        prompt_md='''
## Part b. Fit the plain Emax model

Define `emax_model(c, emax, ec50)` returning `emax * c / (ec50 + c)`.
Concentration first, parameters after, as always.

Fit it to the aggregated means. Read the starting values off the data rather
than guessing blind: the largest mean response is a fair guess for `Emax`, and
the concentration whose response is nearest half of that is a fair guess for
`EC50`. Both are one line each.

Pass `bounds` too. Both parameters are physically positive, and an `Emax`
above about 200 percent inhibition is not a thing, so
`bounds=([0, 1e-6], [200, 1e4])` states what you already know and keeps the
optimiser out of nonsense. A lower bound of exactly zero on `EC50` would let
it sit on the boundary and divide badly, which is why the tiny positive floor
is there.

Then measure the fit: predict at the ten concentrations and compute the RMSE
against the observed means.

Fill in `emax_model`, `emax1`, `ec50_1`, `rmse1`.

Moves: `def`, `curve_fit(f, x, y, p0=[...], bounds=(...))`, `f(x, *popt)`,
`np.sqrt(np.mean(resid ** 2))`.
''',
        starter='''
# Part b. The plain Emax model.

def emax_model(c, emax, ec50):
    """Plain Emax: half of emax is reached at c = ec50."""
    return None   # YOUR CODE


emax1 = None    # YOUR CODE: fitted Emax, percent
ec50_1 = None   # YOUR CODE: fitted EC50, micromolar
rmse1 = None    # YOUR CODE: RMSE of this fit against MEAN_PCT

print("Emax model: Emax = %s  EC50 = %s  rmse = %s" % (emax1, ec50_1, rmse1))
''',
        solution='''
# Part b. The plain Emax model.

def emax_model(c, emax, ec50):
    """Plain Emax: half of emax is reached at c = ec50."""
    return emax * c / (ec50 + c)


_c = agg["CONC_UM"].to_numpy()
_y = agg["MEAN_PCT"].to_numpy()
_top = float(_y.max())
_half = float(_c[int(np.argmin(np.abs(_y - _top / 2.0)))])

_p1, _ = curve_fit(emax_model, _c, _y, p0=[_top, _half],
                   bounds=([0.0, 1e-6], [200.0, 1e4]))

emax1 = float(_p1[0])
ec50_1 = float(_p1[1])
rmse1 = float(np.sqrt(np.mean((_y - emax_model(_c, *_p1)) ** 2)))

print("Emax model: Emax = %s  EC50 = %s  rmse = %s" % (emax1, ec50_1, rmse1))
''',
    ),
    dict(
        label="c",
        prompt_md='''
## Part c. Fit the sigmoid, and compare honestly

Define `sigmoid_model(c, emax, ec50, h)` returning
`emax * c ** h / (ec50 ** h + c ** h)`.

Fit it the same way, with a third starting value of `1.0` for `h`, which is
the plain model, so you are starting from "no extra steepness" and letting the
data move it if it wants to. Extend the bounds with a sensible range for `h`,
say 0.1 to 10: below 0.1 the curve is barely a curve, above 10 it is a step
function and the fit stops being identifiable.

Compute the RMSE the same way as part b.

Then look at the two numbers together, and keep your head. `rmse2` will be
less than or equal to `rmse1`. That is guaranteed, because `h = 1` is
available to the sigmoid, so it can always reproduce the simpler fit and will
only move away if doing so helps. The question worth asking is whether the
fitted `h` is far enough from 1, relative to its own uncertainty, to be worth
carrying, and the RMSE comparison on its own cannot answer that.

Fill in `sigmoid_model`, `emax2`, `ec50_2`, `hill`, `rmse2`.

Moves: `def`, `curve_fit(f, x, y, p0=[...], bounds=(...))`, `f(x, *popt)`,
`np.sqrt(np.mean(resid ** 2))`.
''',
        starter='''
# Part c. The sigmoid Emax model, and the comparison.

def sigmoid_model(c, emax, ec50, h):
    """Sigmoid Emax. h = 1 gives the plain model back."""
    return None   # YOUR CODE


emax2 = None    # YOUR CODE: fitted Emax
ec50_2 = None   # YOUR CODE: fitted EC50
hill = None     # YOUR CODE: fitted Hill coefficient
rmse2 = None    # YOUR CODE: RMSE of this fit

print("sigmoid: Emax = %s  EC50 = %s  h = %s  rmse = %s" % (emax2, ec50_2, hill, rmse2))
if None not in (rmse1, rmse2):
    print("")
    print("rmse went from %.4f to %.4f" % (rmse1, rmse2))
    print("the richer model is never worse, which is why this alone proves nothing")
''',
        solution='''
# Part c. The sigmoid Emax model, and the comparison.

def sigmoid_model(c, emax, ec50, h):
    """Sigmoid Emax. h = 1 gives the plain model back."""
    return emax * c ** h / (ec50 ** h + c ** h)


_p2, _ = curve_fit(sigmoid_model, _c, _y, p0=[_top, _half, 1.0],
                   bounds=([0.0, 1e-6, 0.1], [200.0, 1e4, 10.0]))

emax2 = float(_p2[0])
ec50_2 = float(_p2[1])
hill = float(_p2[2])
rmse2 = float(np.sqrt(np.mean((_y - sigmoid_model(_c, *_p2)) ** 2)))

print("sigmoid: Emax = %s  EC50 = %s  h = %s  rmse = %s" % (emax2, ec50_2, hill, rmse2))
if None not in (rmse1, rmse2):
    print("")
    print("rmse went from %.4f to %.4f" % (rmse1, rmse2))
    print("the richer model is never worse, which is why this alone proves nothing")
''',
    ),
]


def build_expected(ns):
    import numpy as np
    assay = ns["assay"]
    agg = ns["agg"].sort_values("CONC_UM").reset_index(drop=True)
    c = agg["CONC_UM"].to_numpy()
    y = agg["MEAN_PCT"].to_numpy()
    sd_pop = (assay.groupby("CONC_UM")["INHIB_PCT"].std(ddof=0)
              .reset_index().sort_values("CONC_UM")["INHIB_PCT"].to_numpy())
    return {
        "agg_cols": ["CONC_UM", "MEAN_PCT", "SD_PCT", "N"],
        "n_conc": int(len(agg)),
        "n_wells": int(len(assay)),
        "conc": [float(v) for v in c],
        "mean_pct": [float(v) for v in y],
        "sd_pct": [float(v) for v in agg["SD_PCT"]],
        "sd_pop": [float(v) for v in sd_pop],
        "n_per": [int(v) for v in agg["N"]],
        "emax1": float(ns["emax1"]), "ec50_1": float(ns["ec50_1"]),
        "rmse1": float(ns["rmse1"]),
        "emax2": float(ns["emax2"]), "ec50_2": float(ns["ec50_2"]),
        "hill": float(ns["hill"]), "rmse2": float(ns["rmse2"]),
        "true_emax": float(ns["_TRUE_EMAX"]), "true_ec50": float(ns["_TRUE_EC50"]),
    }


CHECK_CODE = '''
def _part_a():
    agg = _need("agg")
    if not isinstance(agg, _pd.DataFrame):
        return "`agg` should be a pandas DataFrame."
    cols = _cols_of(agg) or []
    if "CONC_UM" not in cols and "CONC_UM" in [str(n) for n in (agg.index.names or [])]:
        return "`CONC_UM` is in the index rather than a column. Call .reset_index()."
    missing = [c for c in _EXP["agg_cols"] if c not in cols]
    if missing:
        return "`agg` is missing the column(s) %s." % ", ".join(missing)
    if len(agg) == _EXP["n_wells"]:
        return ("`agg` still has one row per well. Group by CONC_UM so the replicates "
                "collapse to one point each.")
    if len(agg) != _EXP["n_conc"]:
        return "`agg` has %d rows but there are %d concentrations." % (len(agg), _EXP["n_conc"])

    got = agg.sort_values("CONC_UM").reset_index(drop=True)
    if not _arr_close(got["CONC_UM"].to_numpy(), _EXP["conc"]):
        return "`agg` does not cover the expected concentrations."
    if not _arr_close(got["N"].to_numpy(), _EXP["n_per"]):
        return "`N` is not the number of wells at each concentration."
    if not _arr_close(got["MEAN_PCT"].to_numpy(), _EXP["mean_pct"]):
        return "`MEAN_PCT` is not the mean inhibition across the wells at each concentration."
    if _arr_close(got["SD_PCT"].to_numpy(), _EXP["sd_pop"]):
        return ("`SD_PCT` used n in the denominator. Pandas std already uses n-1, so this "
                "usually means numpy's std got involved.")
    if not _arr_close(got["SD_PCT"].to_numpy(), _EXP["sd_pct"]):
        return "`SD_PCT` is not the sample SD across the wells."
    return None


def _part_b():
    emax_model = _need("emax_model")
    emax1 = _need("emax1")
    ec50_1 = _need("ec50_1")
    rmse1 = _need("rmse1")

    if not callable(emax_model):
        return "`emax_model` should be a function."
    try:
        probe = emax_model(_np.array([1.0, 2.0]), 100.0, 1.0)
    except Exception as exc:
        return "`emax_model` raised %s when called as emax_model(c, emax, ec50)." % type(exc).__name__
    if probe is None:
        raise _Todo("`emax_model` returns None. Replace the # YOUR CODE line.")
    probe = _np.asarray(probe, dtype=float)
    if probe.shape != (2,):
        return "`emax_model` should return one value per concentration, so it must work on arrays."
    if not _np.allclose(probe, [50.0, 100.0 * 2.0 / 3.0]):
        return ("`emax_model` does not compute emax * c / (ec50 + c). At c equal to ec50 it "
                "should return exactly half of emax.")

    got = _as_scalar(emax1)
    if got is None:
        return "`emax1` should be a single number."
    if not _num_close(got, _EXP["emax1"], rtol=5e-3):
        return ("`emax1` is not the fitted ceiling. Check the model formula and that you "
                "fitted the aggregated means from part a.")

    got = _as_scalar(ec50_1)
    if got is None:
        return "`ec50_1` should be a single number."
    if not _num_close(got, _EXP["ec50_1"], rtol=5e-3):
        return ("`ec50_1` is not the fitted EC50. It is a concentration in micromolar, and "
                "should land in the same range as the tested concentrations.")

    got = _as_scalar(rmse1)
    if got is None:
        return "`rmse1` should be a single number."
    if not _num_close(got, _EXP["rmse1"], rtol=5e-3):
        return ("`rmse1` is not the RMSE of this fit against MEAN_PCT. It is "
                "sqrt(mean((observed - predicted) ** 2)) at the ten concentrations.")
    return None


def _part_c():
    sigmoid_model = _need("sigmoid_model")
    emax2 = _need("emax2")
    ec50_2 = _need("ec50_2")
    hill = _need("hill")
    rmse2 = _need("rmse2")

    if not callable(sigmoid_model):
        return "`sigmoid_model` should be a function."
    try:
        probe = sigmoid_model(_np.array([1.0, 2.0]), 100.0, 1.0, 1.0)
        steep = sigmoid_model(_np.array([4.0]), 100.0, 4.0, 2.0)
    except Exception as exc:
        return ("`sigmoid_model` raised %s when called as sigmoid_model(c, emax, ec50, h)."
                % type(exc).__name__)
    if probe is None:
        raise _Todo("`sigmoid_model` returns None. Replace the # YOUR CODE line.")
    probe = _np.asarray(probe, dtype=float)
    if probe.shape != (2,):
        return "`sigmoid_model` should return one value per concentration."
    if not _np.allclose(probe, [50.0, 100.0 * 2.0 / 3.0]):
        return ("`sigmoid_model` with h = 1 should give exactly the plain Emax model back. "
                "Check that h is applied to BOTH c and ec50.")
    if not _np.allclose(_np.asarray(steep, dtype=float), [50.0]):
        return ("`sigmoid_model` does not match emax * c**h / (ec50**h + c**h). With "
                "emax 100, ec50 4, h 2 and c 4 it should give exactly 50, because c equals "
                "ec50. Check that h is raised on the ec50 term as well as on c.")

    got = _as_scalar(emax2)
    if got is None:
        return "`emax2` should be a single number."
    if not _num_close(got, _EXP["emax2"], rtol=5e-3):
        return "`emax2` is not the fitted ceiling of the sigmoid model."

    got = _as_scalar(ec50_2)
    if got is None:
        return "`ec50_2` should be a single number."
    if not _num_close(got, _EXP["ec50_2"], rtol=5e-3):
        return "`ec50_2` is not the fitted EC50 of the sigmoid model."

    got = _as_scalar(hill)
    if got is None:
        return "`hill` should be a single number."
    if _num_close(got, 1.0, rtol=1e-6):
        return ("`hill` came back at exactly 1.0, so it probably never moved. Check it is a "
                "free parameter in your model rather than hard coded.")
    if not _num_close(got, _EXP["hill"], rtol=1e-2):
        return ("`hill` is not the fitted Hill coefficient. Make sure h appears on both c "
                "and ec50, and that p0 starts it at 1.0.")

    got = _as_scalar(rmse2)
    if got is None:
        return "`rmse2` should be a single number."
    if not _num_close(got, _EXP["rmse2"], rtol=5e-3):
        return "`rmse2` is not the RMSE of the sigmoid fit against MEAN_PCT."
    if got > _EXP["rmse1"] * (1.0 + 1e-6):
        return ("`rmse2` came out larger than `rmse1`, which should be impossible for "
                "nested models. The sigmoid can always set h to 1 and reproduce the "
                "simpler fit, so a worse result means the fit did not converge.")
    return None
'''

# Functions this problem introduces. Keys must exist in tools/function_catalog.py.
FUNCTIONS = [
    "Emax model",
    "sigmoid Emax",
    "bounds",
    "np.logspace",
]
