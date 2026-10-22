PID = "W7D2"
WEEK = 7
DAY = 2
SLUG = "w7d2_nca_lambdaz_clearance"
TITLE = "Finishing the NCA: lambda_z, AUC to infinity, clearance and MRT"
TOPIC = "PK: terminal slope selection, extrapolated AUC, CL/F, Vz/F, mean residence time"
EST = "20-30 min"

SCENARIO_MD = '''
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
'''

CONCEPT_MD = '''
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
'''

TASK_MD = '''
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
'''

DATA_CODE = '''
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
'''

TOOLKIT_MD = '''
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
'''

TOOLKIT_CODE = '''
import numpy as np

# A tiny made up terminal phase, with one early point that is NOT terminal.
demo_t = np.array([1.0, 2.0, 4.0, 8.0, 12.0, 24.0])
demo_c = np.array([9.5, 9.0, 6.0, 3.6, 2.2, 0.55])
print("demo_t:", demo_t)
print("demo_c:", demo_c)
print("the first point is still distributing, so it is not part of the terminal phase")


def demo_fit(tt, cc):
    """Slope, R2 and adjusted R2 for a log-linear fit."""
    logc = np.log(cc)
    slope, intercept = np.polyfit(tt, logc, deg=1)
    fitted = slope * tt + intercept
    ss_res = float(np.sum((logc - fitted) ** 2))
    ss_tot = float(np.sum((logc - logc.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    n = tt.size
    adj = 1.0 - (1.0 - r2) * (n - 1) / (n - 2)
    return slope, r2, adj


print("")
print("--- 1. try each window of the last n points ---")
for demo_n in range(3, demo_t.size + 1):
    demo_s, demo_r2, demo_adj = demo_fit(demo_t[-demo_n:], demo_c[-demo_n:])
    print("  n=%d  lambda_z=%.5f  R2=%.5f  adjR2=%.5f" % (demo_n, -demo_s, demo_r2, demo_adj))

print("")
print("--- 2. why adjusted rather than plain R2 ---")
print("plain R2 is easy to improve just by using fewer points, since a line")
print("through 3 points fits better than a line through 6 almost automatically.")
print("the adjustment charges for each extra point you drop.")

print("")
print("--- 3. what including a non terminal point costs ---")
demo_s_all, _, _ = demo_fit(demo_t, demo_c)
demo_s_term, _, _ = demo_fit(demo_t[-4:], demo_c[-4:])
print("using all 6 points : lambda_z %.5f, half-life %.2f h"
      % (-demo_s_all, np.log(2) / -demo_s_all))
print("using the last 4   : lambda_z %.5f, half-life %.2f h"
      % (-demo_s_term, np.log(2) / -demo_s_term))
print("the flat early point drags the slope shallower and the half-life longer")

print("")
print("--- 4. extrapolating the tail ---")
demo_lz = -demo_s_term
demo_auc_last = 60.0           # pretend
demo_tail = demo_c[-1] / demo_lz
print("AUC_last         : %.3f" % demo_auc_last)
print("tail, C_last/lz  : %.3f" % demo_tail)
print("AUC_inf          : %.3f" % (demo_auc_last + demo_tail))
print("percent extrapolated: %.2f" % (100.0 * demo_tail / (demo_auc_last + demo_tail)))
'''

PARTS = [
    dict(
        label="a",
        prompt_md='''
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
''',
        starter='''
# Part a. Select the terminal window by adjusted R2, then fit lambda_z.

n_points_used = None   # YOUR CODE: how many points the best window used
lambda_z = None        # YOUR CODE: terminal rate constant, 1/h
t_half = None          # YOUR CODE: terminal half-life, h
adj_r2 = None          # YOUR CODE: the adjusted R2 of the winning fit

print("points used : %s" % n_points_used)
print("lambda_z    : %s 1/h" % lambda_z)
print("half-life   : %s h" % t_half)
print("adjusted R2 : %s" % adj_r2)
''',
        solution='''
# Part a. Select the terminal window by adjusted R2, then fit lambda_z.

_eligible = int(np.sum(t >= TMAX))

_best = None
for _n in range(3, _eligible + 1):
    _tt, _cc = t[-_n:], c[-_n:]
    _logc = np.log(_cc)
    _slope, _intercept = np.polyfit(_tt, _logc, deg=1)
    _fitted = _slope * _tt + _intercept
    _r2 = 1.0 - float(np.sum((_logc - _fitted) ** 2)) / float(np.sum((_logc - _logc.mean()) ** 2))
    _adj = 1.0 - (1.0 - _r2) * (_n - 1) / (_n - 2)
    if _best is None or _adj > _best[0]:
        _best = (_adj, _n, _slope)

adj_r2 = float(_best[0])
n_points_used = int(_best[1])
lambda_z = float(-_best[2])
t_half = float(np.log(2.0) / lambda_z)

print("points used : %s" % n_points_used)
print("lambda_z    : %s 1/h" % lambda_z)
print("half-life   : %s h" % t_half)
print("adjusted R2 : %s" % adj_r2)
''',
    ),
    dict(
        label="b",
        prompt_md='''
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
''',
        starter='''
# Part b. AUC to infinity, and the extrapolated fraction.

auc_inf = None          # YOUR CODE: AUC_LAST plus the extrapolated tail
pct_extrap = None       # YOUR CODE: 100 * (auc_inf - AUC_LAST) / auc_inf
pct_extrap_12h = None   # YOUR CODE: the same figure had sampling stopped at 12 h,
                        #            using AUC_TO_12H, C_AT_12H and the same lambda_z

print("sampled to 48 h: AUC_inf %s, extrapolated %s percent" % (auc_inf, pct_extrap))
print("sampled to 12 h: extrapolated %s percent" % pct_extrap_12h)
''',
        solution='''
# Part b. AUC to infinity, and the extrapolated fraction.

auc_inf = float(AUC_LAST + C_LAST / lambda_z)
pct_extrap = float(100.0 * (auc_inf - AUC_LAST) / auc_inf)

_auc_inf_12h = AUC_TO_12H + C_AT_12H / lambda_z
pct_extrap_12h = float(100.0 * (_auc_inf_12h - AUC_TO_12H) / _auc_inf_12h)

print("sampled to 48 h: AUC_inf %s, extrapolated %s percent" % (auc_inf, pct_extrap))
print("sampled to 12 h: extrapolated %s percent" % pct_extrap_12h)
''',
    ),
    dict(
        label="c",
        prompt_md='''
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
''',
        starter='''
# Part c. Apparent clearance, apparent volume, and mean residence time.

cl_f = None   # YOUR CODE: Dose / AUC_inf, in L/h
vz_f = None   # YOUR CODE: (CL/F) / lambda_z, in L
mrt = None    # YOUR CODE: AUMC_inf / AUC_inf, in h

print("CL/F : %s L/h" % cl_f)
print("Vz/F : %s L" % vz_f)
print("MRT  : %s h   (half-life was %s h)" % (mrt, t_half))
''',
        solution='''
# Part c. Apparent clearance, apparent volume, and mean residence time.

cl_f = float(DOSE_MG / auc_inf)
vz_f = float(cl_f / lambda_z)

_aumc_inf = (AUMC_LAST
             + T_LAST * C_LAST / lambda_z
             + C_LAST / lambda_z ** 2)
mrt = float(_aumc_inf / auc_inf)

print("CL/F : %s L/h" % cl_f)
print("Vz/F : %s L" % vz_f)
print("MRT  : %s h   (half-life was %s h)" % (mrt, t_half))
''',
    ),
]


def build_expected(ns):
    import numpy as np
    t, c = ns["t"], ns["c"]
    tmax, auc_last, aumc_last = ns["TMAX"], ns["AUC_LAST"], ns["AUMC_LAST"]
    c_last, t_last, dose = ns["C_LAST"], ns["T_LAST"], ns["DOSE_MG"]

    def fit(n):
        tt, cc = t[-n:], c[-n:]
        logc = np.log(cc)
        slope, intercept = np.polyfit(tt, logc, deg=1)
        fitted = slope * tt + intercept
        r2 = 1.0 - float(np.sum((logc - fitted) ** 2)) / float(np.sum((logc - logc.mean()) ** 2))
        adj = 1.0 - (1.0 - r2) * (n - 1) / (n - 2)
        return slope, r2, adj

    eligible = int(np.sum(t >= tmax))
    cands = {n: fit(n) for n in range(3, eligible + 1)}
    best_n = max(cands, key=lambda n: cands[n][2])
    best_n_plain = max(cands, key=lambda n: cands[n][1])
    slope, r2, adj = cands[best_n]
    lz = float(-slope)
    auc_inf = float(auc_last + c_last / lz)
    aumc_inf = aumc_last + t_last * c_last / lz + c_last / lz ** 2
    aumc_inf_one_term = aumc_last + t_last * c_last / lz
    return {
        "eligible": eligible,
        "n_points_used": int(best_n),
        "n_points_plain_r2": int(best_n_plain),
        "lambda_z": lz,
        "lambda_z_all": float(-np.polyfit(t[c > 0], np.log(c[c > 0]), deg=1)[0]),
        "t_half": float(np.log(2.0) / lz),
        "t_half_wrong": float(1.0 / lz),
        "adj_r2": float(adj),
        "r2": float(r2),
        "auc_last": float(auc_last),
        "auc_inf": auc_inf,
        "pct_extrap": float(100.0 * (auc_inf - auc_last) / auc_inf),
        "pct_extrap_over_last": float(100.0 * (auc_inf - auc_last) / auc_last),
        "pct_extrap_12h": float(100.0 * ((ns["AUC_TO_12H"] + ns["C_AT_12H"] / lz)
                                         - ns["AUC_TO_12H"])
                                / (ns["AUC_TO_12H"] + ns["C_AT_12H"] / lz)),
        "cl_f": float(dose / auc_inf),
        "cl_f_over_last": float(dose / auc_last),
        "vz_f": float((dose / auc_inf) / lz),
        "mrt": float(aumc_inf / auc_inf),
        "mrt_one_term": float(aumc_inf_one_term / auc_inf),
    }


CHECK_CODE = '''
def _part_a():
    n_points_used = _need("n_points_used")
    lambda_z = _need("lambda_z")
    t_half = _need("t_half")
    adj_r2 = _need("adj_r2")

    got = _as_scalar(n_points_used)
    if got is None:
        return "`n_points_used` should be a single whole number."
    if int(round(got)) == _EXP["n_points_plain_r2"] and \\
            _EXP["n_points_plain_r2"] != _EXP["n_points_used"]:
        return ("`n_points_used` is the window that wins on PLAIN R2. Select on adjusted "
                "R2, which charges for using fewer points.")
    if int(round(got)) != _EXP["n_points_used"]:
        return ("`n_points_used` is not the window that maximises adjusted R2. Search from "
                "3 points up to the number of samples at or after Tmax.")

    got = _as_scalar(lambda_z)
    if got is None:
        return "`lambda_z` should be a single number."
    if _num_close(got, -_EXP["lambda_z"], rtol=1e-6):
        return ("`lambda_z` is negative. The fitted slope is negative, and lambda_z is "
                "defined as its positive counterpart.")
    if _num_close(got, _EXP["lambda_z_all"], rtol=1e-4):
        return ("`lambda_z` used every sample, including the absorption phase. That drags "
                "the slope shallower. Use only the selected terminal window.")
    if not _num_close(got, _EXP["lambda_z"], rtol=1e-5):
        return "`lambda_z` is not the negative slope of the log-linear fit over the chosen window."

    got = _as_scalar(t_half)
    if got is None:
        return "`t_half` should be a single number."
    if _num_close(got, _EXP["t_half_wrong"], rtol=1e-5):
        return "`t_half` is 1/lambda_z. A half-life carries a factor of ln(2)."
    if not _num_close(got, _EXP["t_half"], rtol=1e-5):
        return "`t_half` is not np.log(2) divided by lambda_z."

    got = _as_scalar(adj_r2)
    if got is None:
        return "`adj_r2` should be a single number."
    if _num_close(got, _EXP["r2"], rtol=1e-6):
        return ("`adj_r2` is the plain R2 of the winning window. Apply the adjustment "
                "1 - (1 - r2) * (n - 1) / (n - 2).")
    if not _num_close(got, _EXP["adj_r2"], rtol=1e-5):
        return "`adj_r2` is not the adjusted R2 of the winning fit."
    return None


def _part_b():
    auc_inf = _need("auc_inf")
    pct_extrap = _need("pct_extrap")

    got = _as_scalar(auc_inf)
    if got is None:
        return "`auc_inf` should be a single number."
    if _num_close(got, _EXP["auc_last"], rtol=1e-9):
        return "`auc_inf` equals AUC_LAST, so no tail was added."
    if got < _EXP["auc_last"]:
        return ("`auc_inf` came out smaller than AUC_LAST, which cannot happen: the "
                "extrapolated tail only adds area.")
    if not _num_close(got, _EXP["auc_inf"], rtol=1e-6):
        return "`auc_inf` is not AUC_LAST plus C_LAST divided by lambda_z."

    got = _as_scalar(pct_extrap)
    if got is None:
        return "`pct_extrap` should be a single number."
    if _num_close(got, _EXP["pct_extrap_over_last"], rtol=1e-5):
        return ("`pct_extrap` divided by AUC_LAST. The denominator is auc_inf, because the "
                "question is what share of the TOTAL came from extrapolation.")
    if not _num_close(got, _EXP["pct_extrap"], rtol=1e-5):
        return "`pct_extrap` is not 100 * (auc_inf - AUC_LAST) / auc_inf."

    got = _as_scalar(_need("pct_extrap_12h"))
    if got is None:
        return "`pct_extrap_12h` should be a single number."
    if _num_close(got, _EXP["pct_extrap"], rtol=1e-6):
        return ("`pct_extrap_12h` equals the 48 hour answer, so the shorter study was not "
                "actually used. Build it from AUC_TO_12H and C_AT_12H.")
    if not _num_close(got, _EXP["pct_extrap_12h"], rtol=1e-5):
        return ("`pct_extrap_12h` is not the extrapolated share using AUC_TO_12H, C_AT_12H "
                "and the same lambda_z.")
    return None


def _part_c():
    cl_f = _need("cl_f")
    vz_f = _need("vz_f")
    mrt = _need("mrt")

    got = _as_scalar(cl_f)
    if got is None:
        return "`cl_f` should be a single number."
    if _num_close(got, _EXP["cl_f_over_last"], rtol=1e-6):
        return ("`cl_f` used AUC_LAST. Clearance is the dose over the area to INFINITY, "
                "otherwise it is biased upward by however much you failed to sample.")
    if not _num_close(got, _EXP["cl_f"], rtol=1e-6):
        return "`cl_f` is not DOSE_MG divided by auc_inf."

    got = _as_scalar(vz_f)
    if got is None:
        return "`vz_f` should be a single number."
    if _num_close(got, _EXP["cl_f"] * _EXP["lambda_z"], rtol=1e-6):
        return ("`vz_f` multiplied by lambda_z instead of dividing. Volume is clearance "
                "divided by the rate constant.")
    if not _num_close(got, _EXP["vz_f"], rtol=1e-6):
        return "`vz_f` is not cl_f divided by lambda_z."

    got = _as_scalar(mrt)
    if got is None:
        return "`mrt` should be a single number."
    if _num_close(got, _EXP["mrt_one_term"], rtol=1e-5):
        return ("`mrt` used only one extrapolation term for the AUMC. There are two, "
                "because t * C decays more slowly than C: add C_LAST / lambda_z ** 2 as "
                "well. Leaving it out always makes MRT too short.")
    if not _num_close(got, _EXP["mrt"], rtol=1e-5):
        return "`mrt` is not the extrapolated AUMC divided by auc_inf."
    if got < _EXP["t_half"]:
        return ("`mrt` came out shorter than the half-life, which should not happen for an "
                "orally dosed one compartment drug, since absorption adds residence time.")
    return None
'''

# Functions this problem introduces. Keys must exist in tools/function_catalog.py.
FUNCTIONS = [
    "lambda_z selection",
    "adjusted R2",
    "AUC extrapolation",
    "mean residence time",
]
