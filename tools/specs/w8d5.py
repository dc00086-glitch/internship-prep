PID = "W8D5"
WEEK = 8
DAY = 5
SLUG = "w8d5_dose_selection"
TITLE = "Capstone: choosing the phase 2 dose"
TOPIC = "PK/PD: steady state exposure, probability of target attainment, joint efficacy and safety"
EST = "30-40 min"


SCENARIO_MD = '''
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
'''

CONCEPT_MD = '''
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
'''

TASK_MD = '''
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
'''

DATA_CODE = '''
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
'''

TOOLKIT_MD = '''
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
'''

TOOLKIT_CODE = '''
import numpy as np

print("--- 1. the steady state factor, and what it does ---")
demo_tau = 24.0
for demo_k in (0.03, 0.12, 0.5):
    print("k %.2f per h, half life %5.1f h, 1 - exp(-k tau) = %.4f, so the slow term is "
          "divided by that" % (demo_k, np.log(2) / demo_k, 1 - np.exp(-demo_k * demo_tau)))
print("a long half life relative to the interval means a small divisor and real accumulation")

print("")
print("--- 2. one profile, then four hundred, with no loop ---")
demo_t = np.linspace(0.0, 24.0, 5)
demo_k = np.array([0.08, 0.12, 0.20])        # three subjects
demo_coef = np.array([3.0, 3.2, 3.4])
demo_one = demo_coef[1] * np.exp(-demo_k[1] * demo_t)
print("one subject :", np.round(demo_one, 4))
demo_all = demo_coef[:, None] * np.exp(-demo_k[:, None] * demo_t[None, :])
print("three at once, shape", demo_all.shape)
print(np.round(demo_all, 4))

print("")
print("--- 3. peaks and interval averages along rows ---")
print("max along axis=1 :", np.round(demo_all.max(axis=1), 4))
demo_area = np.sum(0.5 * (demo_all[:, 1:] + demo_all[:, :-1])
                   * np.diff(demo_t)[None, :], axis=1)
print("trapezoid area   :", np.round(demo_area, 4))
print("average over 24 h:", np.round(demo_area / 24.0, 4))
print("axis=1 collapses time and leaves one number per subject")

print("")
print("--- 4. a fraction is just the mean of a boolean ---")
demo_vals = np.array([0.4, 0.6, 0.7, 0.52, 0.9])
print("values:", demo_vals)
print("fraction at or above 0.55:", float((demo_vals >= 0.55).mean()))
demo_cmax = np.array([1.0, 2.0, 3.5, 2.9, 4.0])
print("fraction above a Cmax limit of 3.0:", float((demo_cmax > 3.0).mean()))
print("both conditions:", float(((demo_vals >= 0.55) & (demo_cmax <= 3.0)).mean()))
print("note the & and the brackets: `and` does not work elementwise")

print("")
print("--- 5. picking the winner ---")
demo_doses = np.array([10.0, 20.0, 40.0, 80.0])
demo_success = np.array([0.1, 0.5, 0.42, 0.05])
demo_best = int(np.argmax(demo_success))
print("success by dose:", demo_success)
print("best is index %d, which is %g mg" % (demo_best, demo_doses[demo_best]))
print("argmax gives the position, so index the dose array with it")
'''

PARTS = [
    dict(
        label="a",
        prompt_md='''
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
''',
        starter='''
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
''',
        solution='''
# Part a. One typical subject, at steady state.

k_typ = CL_POP / V_POP
c_typ = (DOSE_MID * KA / (V_POP * (KA - k_typ))
         * (np.exp(-k_typ * t_grid) / (1.0 - np.exp(-k_typ * TAU))
            - np.exp(-KA * t_grid) / (1.0 - np.exp(-KA * TAU))))

cmax_ss = float(c_typ.max())
cmin_ss = float(c_typ.min())
cavg_ss = float(np.sum(0.5 * (c_typ[1:] + c_typ[:-1]) * np.diff(t_grid)) / TAU)
cavg_identity = float(DOSE_MID / (CL_POP * TAU))

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
''',
    ),
    dict(
        label="b",
        prompt_md='''
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
''',
        starter='''
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
''',
        solution='''
# Part b. Four hundred patients at the middle dose.

def population_metrics(dose):
    """Steady state Cmax and average inhibition for every patient."""
    k = cl_i / v_i
    coef = dose * KA / (v_i * (KA - k))
    c = coef[:, None] * (np.exp(-k[:, None] * t_grid[None, :])
                         / (1.0 - np.exp(-k[:, None] * TAU))
                         - np.exp(-KA * t_grid[None, :])
                         / (1.0 - np.exp(-KA * TAU)))
    cmax = c.max(axis=1)
    inh = IMAX * c / (ic50_i[:, None] + c)
    area = np.sum(0.5 * (inh[:, 1:] + inh[:, :-1]) * np.diff(t_grid)[None, :], axis=1)
    return cmax, area / TAU


cmax_i, mean_inh_i = population_metrics(DOSE_MID)

p_eff = float((mean_inh_i >= EFF_TARGET).mean())
p_tox = float((cmax_i > CMAX_LIMIT).mean())
p_success = float(((mean_inh_i >= EFF_TARGET) & (cmax_i <= CMAX_LIMIT)).mean())

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
''',
    ),
    dict(
        label="c",
        prompt_md='''
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
''',
        starter='''
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
''',
        solution='''
# Part c. The dose response of the decision.

_eff, _tox, _suc = [], [], []
for _d in DOSES:
    _cmax, _inh = population_metrics(_d)
    _eff.append(float((_inh >= EFF_TARGET).mean()))
    _tox.append(float((_cmax > CMAX_LIMIT).mean()))
    _suc.append(float(((_inh >= EFF_TARGET) & (_cmax <= CMAX_LIMIT)).mean()))

p_eff_by_dose = np.array(_eff)
p_tox_by_dose = np.array(_tox)
p_success_by_dose = np.array(_suc)

best_dose = float(DOSES[int(np.argmax(p_success_by_dose))])

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
''',
    ),
]


def build_expected(ns):
    import numpy as np

    grid = np.asarray(ns["t_grid"], dtype=float)
    tau = float(ns["TAU"])
    ka = float(ns["KA"])
    cl = np.asarray(ns["cl_i"], dtype=float)
    v = np.asarray(ns["v_i"], dtype=float)
    ic50 = np.asarray(ns["ic50_i"], dtype=float)
    imax = float(ns["IMAX"])
    doses = np.asarray(ns["DOSES"], dtype=float)
    dose_mid = float(ns["DOSE_MID"])
    eff_target = float(ns["EFF_TARGET"])
    cmax_limit = float(ns["CMAX_LIMIT"])
    cl_pop = float(ns["CL_POP"])
    v_pop = float(ns["V_POP"])

    def profile(dose, clv, vv):
        k = clv / vv
        coef = dose * ka / (vv * (ka - k))
        return coef * (np.exp(-k * grid) / (1.0 - np.exp(-k * tau))
                       - np.exp(-ka * grid) / (1.0 - np.exp(-ka * tau)))

    c_typ = profile(dose_mid, cl_pop, v_pop)
    single = (dose_mid * ka / (v_pop * (ka - cl_pop / v_pop))
              * (np.exp(-cl_pop / v_pop * grid) - np.exp(-ka * grid)))

    def pop(dose):
        k = cl / v
        coef = dose * ka / (v * (ka - k))
        c = coef[:, None] * (np.exp(-k[:, None] * grid[None, :])
                             / (1.0 - np.exp(-k[:, None] * tau))
                             - np.exp(-ka * grid[None, :]) / (1.0 - np.exp(-ka * tau)))
        cmax = c.max(axis=1)
        inh = imax * c / (ic50[:, None] + c)
        area = np.sum(0.5 * (inh[:, 1:] + inh[:, :-1]) * np.diff(grid)[None, :], axis=1)
        return cmax, area / tau

    cmax_mid, inh_mid = pop(dose_mid)
    eff, tox, suc = [], [], []
    for d in doses:
        cm, ih = pop(d)
        eff.append(float((ih >= eff_target).mean()))
        tox.append(float((cm > cmax_limit).mean()))
        suc.append(float(((ih >= eff_target) & (cm <= cmax_limit)).mean()))

    return {
        "n_grid": int(grid.size),
        "n_subj": int(cl.size),
        "n_doses": int(doses.size),
        "c_typ": [float(x) for x in c_typ],
        "c_single": [float(x) for x in single],
        "cmax_ss": float(c_typ.max()),
        "cmin_ss": float(c_typ.min()),
        "cavg_ss": float(np.sum(0.5 * (c_typ[1:] + c_typ[:-1]) * np.diff(grid)) / tau),
        "cavg_identity": float(dose_mid / (cl_pop * tau)),
        "cmax_i": [float(x) for x in cmax_mid],
        "mean_inh_i": [float(x) for x in inh_mid],
        "p_eff": eff[int(np.argmin(np.abs(doses - dose_mid)))],
        "p_tox": tox[int(np.argmin(np.abs(doses - dose_mid)))],
        "p_success": suc[int(np.argmin(np.abs(doses - dose_mid)))],
        "p_eff_by_dose": eff,
        "p_tox_by_dose": tox,
        "p_success_by_dose": suc,
        "best_dose": float(doses[int(np.argmax(suc))]),
        "best_index": int(np.argmax(suc)),
    }


CHECK_CODE = '''

def _part_a():
    c_typ = _need("c_typ")
    cmax = _need("cmax_ss")
    cmin = _need("cmin_ss")
    cavg = _need("cavg_ss")
    ident = _need("cavg_identity")

    arr = _as_array(c_typ)
    if arr is None:
        return "`c_typ` should be an array of concentrations."
    if arr.shape != (_EXP["n_grid"],):
        return ("`c_typ` has shape %s but t_grid has %d points."
                % (arr.shape, _EXP["n_grid"]))
    if _arr_close(arr, _EXP["c_single"], rtol=1e-6):
        return ("`c_typ` is the single dose profile. The steady state version divides each "
                "exponential by one minus its own accumulation factor, "
                "1 - exp(-k * TAU) and 1 - exp(-KA * TAU).")
    if not _arr_close(arr, _EXP["c_typ"], rtol=1e-6):
        return ("`c_typ` is not the steady state profile at the typical parameters. Use "
                "k = CL_POP / V_POP and the closed form from the concept section.")

    for name, val in (("cmax_ss", cmax), ("cmin_ss", cmin), ("cavg_ss", cavg),
                      ("cavg_identity", ident)):
        if _as_scalar(val) is None:
            return "`%s` should be a single number." % name

    if not _num_close(_as_scalar(cmax), _EXP["cmax_ss"], rtol=1e-6):
        return "`cmax_ss` is not the peak of the steady state profile."
    got = _as_scalar(cmin)
    if _num_close(got, 0.0, atol=1e-9):
        return ("`cmin_ss` is zero, which is the concentration at the start of a SINGLE "
                "dose profile. At steady state the trough is what is left from the "
                "previous doses.")
    if not _num_close(got, _EXP["cmin_ss"], rtol=1e-6):
        return "`cmin_ss` is not the trough of the steady state profile."

    got = _as_scalar(cavg)
    if _num_close(got, _EXP["cavg_ss"] * TAU, rtol=1e-6):
        return "`cavg_ss` is the area, not the average. Divide by TAU."
    if not _num_close(got, _EXP["cavg_ss"], rtol=1e-5):
        return ("`cavg_ss` is not the interval average of c_typ. Integrate over the "
                "interval with trapezoids and divide by TAU.")
    got = _as_scalar(ident)
    if not _num_close(got, _EXP["cavg_identity"], rtol=1e-9):
        return "`cavg_identity` should be DOSE_MID / (CL_POP * TAU)."
    return None


def _part_b():
    cmax_i = _need("cmax_i")
    inh_i = _need("mean_inh_i")
    p_eff = _need("p_eff")
    p_tox = _need("p_tox")
    p_suc = _need("p_success")

    for name, val, want in (("cmax_i", cmax_i, _EXP["cmax_i"]),
                            ("mean_inh_i", inh_i, _EXP["mean_inh_i"])):
        arr = _as_array(val)
        if arr is None:
            return "`%s` should be an array of numbers." % name
        if arr.shape != (_EXP["n_subj"],):
            if arr.shape == (_EXP["n_grid"],):
                return ("`%s` has one entry per time point, so the reduction collapsed the "
                        "wrong axis. With a (patients, times) matrix you want axis=1."
                        % name)
            return ("`%s` has shape %s but there are %d patients."
                    % (name, arr.shape, _EXP["n_subj"]))
        if not _arr_close(arr, want, rtol=1e-6):
            return ("`%s` does not match the expected values at DOSE_MID. Check the "
                    "broadcasting: k[:, None] against t_grid[None, :]." % name)

    for name, val in (("p_eff", p_eff), ("p_tox", p_tox), ("p_success", p_suc)):
        if _as_scalar(val) is None:
            return "`%s` should be a single number." % name
        if _as_scalar(val) < 0 or _as_scalar(val) > 1:
            return "`%s` should be a fraction between 0 and 1." % name

    got = _as_scalar(p_eff)
    if _num_close(got, 1.0 - _EXP["p_eff"], rtol=1e-6):
        return ("`p_eff` is the fraction that FAILS the efficacy criterion. The comparison "
                "is mean inhibition at or above EFF_TARGET.")
    if not _num_close(got, _EXP["p_eff"], rtol=1e-6):
        return "`p_eff` is not the fraction of patients reaching EFF_TARGET."
    got = _as_scalar(p_tox)
    if _num_close(got, 1.0 - _EXP["p_tox"], rtol=1e-6):
        return ("`p_tox` is the fraction within the limit. The task asks for the fraction "
                "ABOVE CMAX_LIMIT.")
    if not _num_close(got, _EXP["p_tox"], rtol=1e-6):
        return "`p_tox` is not the fraction of patients whose Cmax exceeds CMAX_LIMIT."
    got = _as_scalar(p_suc)
    if _num_close(got, _EXP["p_eff"] * (1.0 - _EXP["p_tox"]), rtol=1e-4):
        return ("`p_success` is p_eff times one minus p_tox, which assumes the two criteria "
                "are independent. They are not: count the patients who satisfy both, with "
                "a single & expression.")
    if not _num_close(got, _EXP["p_success"], rtol=1e-6):
        return ("`p_success` is not the fraction meeting the efficacy criterion while "
                "staying at or below the safety limit.")
    return None


def _part_c():
    eff = _need("p_eff_by_dose")
    tox = _need("p_tox_by_dose")
    suc = _need("p_success_by_dose")
    best = _need("best_dose")

    for name, val, want in (("p_eff_by_dose", eff, _EXP["p_eff_by_dose"]),
                            ("p_tox_by_dose", tox, _EXP["p_tox_by_dose"]),
                            ("p_success_by_dose", suc, _EXP["p_success_by_dose"])):
        arr = _as_array(val)
        if arr is None:
            return "`%s` should be an array of five numbers." % name
        if arr.shape != (_EXP["n_doses"],):
            return ("`%s` has shape %s. It should hold one fraction per candidate dose, in "
                    "the order of DOSES." % (name, arr.shape))
        if not _arr_close(arr, want, rtol=1e-6):
            if _arr_close(arr[::-1], want, rtol=1e-6):
                return "`%s` is in reverse dose order." % name
            return ("`%s` does not match the expected fractions. Each dose needs its own "
                    "call to the part b calculation." % name)

    got = _as_scalar(best)
    if got is None:
        return "`best_dose` should be a single number, a dose in mg."
    if _num_close(got, float(_EXP["best_index"])):
        return ("`best_dose` is the index of the best dose, not the dose. Use it to index "
                "DOSES.")
    if _num_close(got, float(DOSES[int(_np.argmax(_EXP["p_eff_by_dose"]))])):
        return ("`best_dose` is the dose that maximises efficacy alone. The decision is on "
                "the joint criterion, where that dose does badly.")
    if not _num_close(got, _EXP["best_dose"], rtol=1e-9):
        return ("`best_dose` is not the dose with the highest joint success fraction. Take "
                "np.argmax of the success array and index DOSES with it.")
    return None
'''

# Functions this problem introduces. Keys must exist in tools/function_catalog.py.
FUNCTIONS = [
    "steady state closed form",
    "Cavg identity",
    "probability of target attainment",
    "joint criteria",
    "boolean mean as a fraction",
]
