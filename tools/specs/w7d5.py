PID = "W7D5"
WEEK = 7
DAY = 5
SLUG = "w7d5_pk_multiple_dosing"
TITLE = "Multiple dosing: accumulation, steady state, and how long it takes"
TOPIC = "PK: repeated dosing by simulation, accumulation ratio, time to steady state"
EST = "20-30 min"

SCENARIO_MD = '''
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
'''

CONCEPT_MD = '''
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
'''

TASK_MD = '''
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
'''

DATA_CODE = '''
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
'''

TOOLKIT_MD = '''
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
'''

TOOLKIT_CODE = '''
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# A tiny made up one compartment oral drug. Nothing here is the answer.
demo_ka, demo_v, demo_cl = 1.2, 40.0, 4.0
demo_dose, demo_tau, demo_n = 100.0, 8.0, 6


def demo_rhs(tt, y, ka, v, cl):
    a = ka * y[0]
    return [-a, a - (cl / v) * y[1]]


print("--- 1. carrying the state forward, interval by interval ---")
demo_state = np.array([0.0, 0.0])
demo_grid = np.linspace(0.0, demo_tau, 25)
demo_all_t, demo_all_c = [], []
for demo_i in range(demo_n):
    demo_state = demo_state.copy()
    demo_state[0] += demo_dose                     # the dose lands in the depot
    demo_sol = solve_ivp(demo_rhs, (0.0, demo_tau), demo_state, t_eval=demo_grid,
                         args=(demo_ka, demo_v, demo_cl), rtol=1e-9, atol=1e-11)
    demo_all_t.append(demo_grid + demo_i * demo_tau)
    demo_all_c.append(demo_sol.y[1] / demo_v)
    demo_state = demo_sol.y[:, -1]                 # final state becomes the next start
    print("  after dose %d: depot %7.3f mg, central %7.3f mg, trough %6.4f"
          % (demo_i + 1, demo_state[0], demo_state[1], demo_sol.y[1][-1] / demo_v))

demo_t = np.concatenate(demo_all_t)
demo_c = np.concatenate(demo_all_c)

print("")
print("--- 2. accumulation, simulated against algebraic ---")
demo_kel = demo_cl / demo_v
demo_first = demo_c[demo_t < demo_tau].max()
demo_last = demo_c[demo_t >= (demo_n - 1) * demo_tau].max()
print("Cmax after dose 1 : %.4f" % demo_first)
print("Cmax in last interval: %.4f" % demo_last)
print("ratio, simulated  : %.4f" % (demo_last / demo_first))
print("ratio, 1/(1-exp(-kel*tau)): %.4f" % (1.0 / (1.0 - np.exp(-demo_kel * demo_tau))))
print("they agree, which is the check worth doing")

print("")
print("--- 3. the dose does not affect the ratio ---")
for demo_d in [50.0, 100.0, 400.0]:
    print("  dose %5.0f -> accumulation ratio still %.4f"
          % (demo_d, 1.0 / (1.0 - np.exp(-demo_kel * demo_tau))))
print("doubling the dose doubles every concentration and changes the ratio not at all")

demo_fig, demo_ax = plt.subplots(figsize=(6.5, 3.4))
demo_ax.plot(demo_t, demo_c, lw=1.2)
demo_ax.set_xlabel("Time (h)"); demo_ax.set_ylabel("Made up units")
demo_ax.set_title("Toolkit demo, not the answer")
plt.show()
'''

PARTS = [
    dict(
        label="a",
        prompt_md='''
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
''',
        starter='''
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
''',
        solution='''
# Part a. Simulate 20 doses, 12 hours apart.

def rhs(t, y, ka, v, cl):
    """y[0] depot, y[1] central, both mg."""
    absorbed = ka * y[0]
    return [-absorbed, absorbed - (cl / v) * y[1]]


def simulate(n_doses):
    """Return (times, concentrations) across the whole regimen."""
    grid = np.linspace(0.0, TAU_H, POINTS_PER_INTERVAL)
    state = np.array([0.0, 0.0])
    all_t, all_c = [], []
    for i in range(n_doses):
        state = state.copy()
        state[0] += DOSE_MG
        sol = solve_ivp(rhs, (0.0, TAU_H), state, t_eval=grid,
                        args=(KA, V_F, CL_F), rtol=1e-9, atol=1e-11)
        all_t.append(grid + i * TAU_H)
        all_c.append(sol.y[1] / V_F)
        state = sol.y[:, -1]
    return np.concatenate(all_t), np.concatenate(all_c)


times, conc = simulate(N_DOSES)

if conc is not None:
    print("points simulated:", len(times))
    print("peak overall    : %.4f mg/L" % conc.max())
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(times, conc, lw=1.0)
    ax.set_xlabel("Time (h)"); ax.set_ylabel("Concentration (mg/L)")
    ax.set_title("200 mg every 12 h")
    plt.show()
''',
    ),
    dict(
        label="b",
        prompt_md='''
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
''',
        starter='''
# Part b. Accumulation, simulated and predicted.

cmax_first = None   # YOUR CODE: peak within the FIRST dosing interval
cmax_ss = None      # YOUR CODE: peak within the LAST dosing interval
r_sim = None        # YOUR CODE: cmax_ss / cmax_first
r_theory = None     # YOUR CODE: 1 / (1 - exp(-KEL * TAU_H))

print("Cmax, first interval : %s mg/L" % cmax_first)
print("Cmax, last interval  : %s mg/L" % cmax_ss)
print("accumulation ratio, simulated : %s" % r_sim)
print("accumulation ratio, algebraic : %s" % r_theory)
''',
        solution='''
# Part b. Accumulation, simulated and predicted.

_first = times < TAU_H
_last = times >= (N_DOSES - 1) * TAU_H

cmax_first = float(conc[_first].max())
cmax_ss = float(conc[_last].max())
r_sim = float(cmax_ss / cmax_first)
r_theory = float(1.0 / (1.0 - np.exp(-KEL * TAU_H)))

print("Cmax, first interval : %s mg/L" % cmax_first)
print("Cmax, last interval  : %s mg/L" % cmax_ss)
print("accumulation ratio, simulated : %s" % r_sim)
print("accumulation ratio, algebraic : %s" % r_theory)
''',
    ),
    dict(
        label="c",
        prompt_md='''
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
''',
        starter='''
# Part c. Average exposure, and time to steady state.

cavg_ss = None       # YOUR CODE: DOSE_MG / (TAU_H * CL_F)
t_to_90pct = None    # YOUR CODE: first time conc reaches 90 percent of cmax_ss
n_half_lives = None  # YOUR CODE: t_to_90pct / T_HALF

print("Cavg at steady state : %s mg/L" % cavg_ss)
print("time to 90 percent   : %s h" % t_to_90pct)
print("which is             : %s half-lives" % n_half_lives)
''',
        solution='''
# Part c. Average exposure, and time to steady state.

cavg_ss = float(DOSE_MG / (TAU_H * CL_F))
t_to_90pct = float(times[int(np.argmax(conc >= 0.90 * cmax_ss))])
n_half_lives = float(t_to_90pct / T_HALF)

print("Cavg at steady state : %s mg/L" % cavg_ss)
print("time to 90 percent   : %s h" % t_to_90pct)
print("which is             : %s half-lives" % n_half_lives)
''',
    ),
]


def build_expected(ns):
    import numpy as np
    from scipy.integrate import solve_ivp
    ka, v, cl = ns["KA"], ns["V_F"], ns["CL_F"]
    dose, tau, n = ns["DOSE_MG"], ns["TAU_H"], ns["N_DOSES"]
    kel, t_half = ns["KEL"], ns["T_HALF"]

    def rhs(tt, y, ka_, v_, cl_):
        a = ka_ * y[0]
        return [-a, a - (cl_ / v_) * y[1]]

    grid = np.linspace(0.0, tau, ns["POINTS_PER_INTERVAL"])
    state = np.array([0.0, 0.0])
    all_t, all_c = [], []
    for i in range(n):
        state = state.copy()
        state[0] += dose
        sol = solve_ivp(rhs, (0.0, tau), state, t_eval=grid, args=(ka, v, cl),
                        rtol=1e-9, atol=1e-11)
        all_t.append(grid + i * tau)
        all_c.append(sol.y[1] / v)
        state = sol.y[:, -1]
    times = np.concatenate(all_t)
    conc = np.concatenate(all_c)

    # the IV mistake: dose into the central compartment instead of the depot
    state = np.array([0.0, 0.0])
    iv_c = []
    for i in range(n):
        state = state.copy()
        state[1] += dose
        sol = solve_ivp(rhs, (0.0, tau), state, t_eval=grid, args=(ka, v, cl),
                        rtol=1e-9, atol=1e-11)
        iv_c.append(sol.y[1] / v)
        state = sol.y[:, -1]
    iv_conc = np.concatenate(iv_c)

    cmax_first = float(conc[times < tau].max())
    cmax_ss = float(conc[times >= (n - 1) * tau].max())
    t90 = float(times[int(np.argmax(conc >= 0.90 * cmax_ss))])
    return {
        "n_points": int(times.size),
        "times": [float(x) for x in times],
        "conc": [float(x) for x in conc],
        "iv_conc": [float(x) for x in iv_conc],
        "cmax_first": cmax_first,
        "cmax_ss": cmax_ss,
        "r_sim": float(cmax_ss / cmax_first),
        "r_theory": float(1.0 / (1.0 - np.exp(-kel * tau))),
        "cavg_ss": float(dose / (tau * cl)),
        "cavg_wrong": float(dose / cl),
        "t_to_90pct": t90,
        "n_half_lives": float(t90 / t_half),
    }


CHECK_CODE = '''
def _part_a():
    simulate = _need("simulate")
    times = _need("times")
    conc = _need("conc")

    if not callable(simulate):
        return "`simulate` should be a function taking the number of doses."

    arr_t = _as_array(times)
    arr_c = _as_array(conc)
    if arr_t is None or arr_c is None:
        return "`times` and `conc` should both be numeric arrays."
    if arr_t.shape != arr_c.shape:
        return "`times` and `conc` should be the same length."
    if arr_t.size != _EXP["n_points"]:
        return ("`times` has %d points but %d were expected, which is N_DOSES intervals of "
                "POINTS_PER_INTERVAL each." % (arr_t.size, _EXP["n_points"]))
    if not _arr_close(arr_t, _EXP["times"], rtol=1e-9, atol=1e-9):
        return ("`times` does not run continuously across the regimen. Shift each "
                "interval's grid by i * TAU_H before stacking.")
    if _arr_close(arr_c, _EXP["iv_conc"], rtol=1e-4):
        return ("`conc` matches dosing into the CENTRAL compartment, which is an "
                "intravenous bolus. This is an oral drug, so each dose goes into the depot "
                "and has to be absorbed.")
    if not _arr_close(arr_c, _EXP["conc"], rtol=1e-5, atol=1e-8):
        if abs(float(arr_c.max()) - _EXP["cmax_first"]) < 1e-6:
            return ("`conc` never accumulates, so the state is not being carried forward. "
                    "Each interval should start from the previous one's final state, "
                    "sol.y[:, -1], with the dose added.")
        return ("`conc` does not match. Check that the dose is ADDED to the existing depot "
                "amount rather than replacing it, and that concentration is the central "
                "amount divided by V_F.")
    return None


def _part_b():
    cmax_first = _need("cmax_first")
    cmax_ss = _need("cmax_ss")
    r_sim = _need("r_sim")
    r_theory = _need("r_theory")

    got = _as_scalar(cmax_first)
    if got is None:
        return "`cmax_first` should be a single number."
    if _num_close(got, _EXP["cmax_ss"], rtol=1e-6):
        return ("`cmax_first` is the steady state peak. Restrict to times below TAU_H for "
                "the first interval.")
    if not _num_close(got, _EXP["cmax_first"], rtol=1e-5):
        return "`cmax_first` is not the peak within the first dosing interval."

    got = _as_scalar(cmax_ss)
    if got is None:
        return "`cmax_ss` should be a single number."
    if not _num_close(got, _EXP["cmax_ss"], rtol=1e-5):
        return ("`cmax_ss` is not the peak within the last dosing interval, which is times "
                "at or above (N_DOSES - 1) * TAU_H.")

    got = _as_scalar(r_sim)
    if got is None:
        return "`r_sim` should be a single number."
    if _num_close(got, 1.0 / _EXP["r_sim"], rtol=1e-4):
        return "`r_sim` is the other way up. It is the steady state peak over the first peak."
    if not _num_close(got, _EXP["r_sim"], rtol=1e-4):
        return "`r_sim` is not cmax_ss divided by cmax_first."

    got = _as_scalar(r_theory)
    if got is None:
        return "`r_theory` should be a single number."
    if _num_close(got, 1.0 - _np.exp(-_need("KEL") * _need("TAU_H")), rtol=1e-6):
        return ("`r_theory` is the denominator rather than the ratio. It is 1 divided by "
                "(1 - exp(-KEL * TAU_H)).")
    if not _num_close(got, _EXP["r_theory"], rtol=1e-6):
        return "`r_theory` is not 1 / (1 - exp(-KEL * TAU_H))."
    return None


def _part_c():
    cavg_ss = _need("cavg_ss")
    t_to_90pct = _need("t_to_90pct")
    n_half_lives = _need("n_half_lives")

    got = _as_scalar(cavg_ss)
    if got is None:
        return "`cavg_ss` should be a single number."
    if _num_close(got, _EXP["cavg_wrong"], rtol=1e-6):
        return ("`cavg_ss` left out the dosing interval. It is the dose divided by TAU_H "
                "times CL_F, because the average is over the whole interval.")
    if not _num_close(got, _EXP["cavg_ss"], rtol=1e-6):
        return "`cavg_ss` is not DOSE_MG / (TAU_H * CL_F)."

    got = _as_scalar(t_to_90pct)
    if got is None:
        return "`t_to_90pct` should be a single number."
    if _num_close(got, float(_np.argmax(_np.asarray(_EXP["conc"])
                                        >= 0.90 * _EXP["cmax_ss"]))):
        return ("`t_to_90pct` is the index, not the time. Use it to look up the matching "
                "entry of times.")
    if not _num_close(got, _EXP["t_to_90pct"], rtol=1e-6):
        return ("`t_to_90pct` is not the first time conc reaches 90 percent of cmax_ss. "
                "np.argmax on a boolean array gives the first True.")

    got = _as_scalar(n_half_lives)
    if got is None:
        return "`n_half_lives` should be a single number."
    if not _num_close(got, _EXP["n_half_lives"], rtol=1e-5):
        return "`n_half_lives` is not t_to_90pct divided by T_HALF."
    return None
'''

# Functions this problem introduces. Keys must exist in tools/function_catalog.py.
FUNCTIONS = [
    "carrying state forward",
    "accumulation ratio",
    "time to steady state",
    "Cavg at steady state",
]
