PID = "W8D3"
WEEK = 8
DAY = 3
SLUG = "w8d3_population_variability"
TITLE = "A virtual population: simulating variability and reading it back"
TOPIC = "Population PK: lognormal parameters, exposure percentiles, how precisely a CV can be estimated"
EST = "25-35 min"

SCENARIO_MD = '''
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
'''

CONCEPT_MD = '''
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
'''

TASK_MD = '''
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
'''

DATA_CODE = '''
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
'''

TOOLKIT_MD = '''
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
'''

TOOLKIT_CODE = '''
import numpy as np

print("--- 1. what the two arguments of lognormal mean ---")
demo_rng = np.random.default_rng(0)
demo_a = demo_rng.lognormal(0.0, 0.3, size=100000)
print("lognormal(0.0, 0.3): median %.4f  geometric mean %.4f  arithmetic mean %.4f"
      % (np.median(demo_a), np.exp(np.mean(np.log(demo_a))), demo_a.mean()))
print("theory             : median 1.0000  geometric mean 1.0000  arithmetic mean %.4f"
      % np.exp(0.3 ** 2 / 2))
demo_b = demo_rng.lognormal(4.2, 0.3, size=5)
print("lognormal(4.2, 0.3) gives", np.round(demo_b, 2), "<- 4.2 was a mean of LOGS")
print("what you wanted was 4.2 * lognormal(0.0, 0.3):",
      np.round(4.2 * demo_rng.lognormal(0.0, 0.3, size=5), 3))

print("")
print("--- 2. the exact CV and the omega approximation ---")
for demo_om in (0.1, 0.3, 0.6, 1.0):
    print("omega %.1f  exact CV %6.2f pct  approximation %5.1f pct  mean over median %.4f"
          % (demo_om, 100 * np.sqrt(np.exp(demo_om ** 2) - 1), 100 * demo_om,
             np.exp(demo_om ** 2 / 2)))

print("")
print("--- 3. percentiles of a skewed distribution ---")
demo_c = 20.0 * demo_rng.lognormal(0.0, 0.4, size=20000)
print("mean %.3f  SD %.3f" % (demo_c.mean(), demo_c.std(ddof=1)))
print("mean minus 2 SD = %.3f, which is below the 2.5th percentile of %.3f"
      % (demo_c.mean() - 2 * demo_c.std(ddof=1), np.percentile(demo_c, 2.5)))
print("percentiles 5, 50, 95:", np.round(np.percentile(demo_c, [5, 50, 95]), 3))
print("for a skewed distribution the percentiles are the honest summary")

print("")
print("--- 4. many studies at once, with one draw ---")
demo_rng2 = np.random.default_rng(1)
demo_draws = demo_rng2.lognormal(0.0, 0.3, size=(5, 12))   # 5 studies of 12 people
print("shape:", demo_draws.shape)
demo_logs = np.log(demo_draws)
demo_cvs = 100 * np.sqrt(np.exp(np.var(demo_logs, axis=1, ddof=1)) - 1)
print("CV estimated in each of the 5 studies:", np.round(demo_cvs, 2))
print("the truth was %.2f pct in every one of them" % (100 * np.sqrt(np.exp(0.09) - 1)))
print("one draw of shape (n_studies, n_per_study) is reproducible in a way that")
print("a loop of separate draws is not, because the order of the draws is fixed")

print("")
print("--- 5. row wise against whole array reductions ---")
demo_m = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
print("var(axis=1, ddof=1), one per row:", demo_m.var(axis=1, ddof=1))
print("var(ddof=1) over everything     :", demo_m.var(ddof=1))
print("forgetting axis=1 gives one number where you wanted n_studies of them")
'''

PARTS = [
    dict(
        label="a",
        prompt_md='''
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
''',
        starter='''
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
''',
        solution='''
# Part a. Three different centres of one distribution.

cl_gm = float(np.exp(np.mean(np.log(cl_i))))
cl_cv_pct = float(100.0 * np.sqrt(np.exp(np.var(np.log(cl_i), ddof=1)) - 1.0))
cl_mean = float(np.mean(cl_i))
mean_over_median = float(np.mean(cl_i) / np.median(cl_i))
mean_over_median_theory = float(np.exp(OMEGA_CL ** 2 / 2.0))

print("typical clearance in the simulation: %.1f L/h" % CL_POP)
print("geometric mean  : %r L/h" % (cl_gm,))
print("arithmetic mean : %r L/h" % (cl_mean,))
print("CV              : %r pct   (omega %.2f implies %.2f pct)"
      % (cl_cv_pct, OMEGA_CL, 100 * np.sqrt(np.exp(OMEGA_CL ** 2) - 1)))
print("mean / median   : %r observed, %r in theory"
      % (mean_over_median, mean_over_median_theory))
''',
    ),
    dict(
        label="b",
        prompt_md='''
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
''',
        starter='''
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
''',
        solution='''
# Part b. From parameters to exposures.

k_i = cl_i / v_i
auc_i = DOSE / cl_i
tmax_i = np.log(KA / k_i) / (KA - k_i)
cmax_i = DOSE * KA / (v_i * (KA - k_i)) * (np.exp(-k_i * tmax_i) - np.exp(-KA * tmax_i))

auc_gm = float(np.exp(np.mean(np.log(auc_i))))
auc_p5 = float(np.percentile(auc_i, 5))
auc_p95 = float(np.percentile(auc_i, 95))
fold_range = auc_p95 / auc_p5

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
''',
    ),
    dict(
        label="c",
        prompt_md='''
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
''',
        starter='''
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
''',
        solution='''
# Part c. The sampling distribution of a CV estimate.

rng_c = np.random.default_rng(SEED_C)

draws_12 = rng_c.lognormal(0.0, OMEGA_CL, size=(N_STUDIES, 12))
draws_120 = rng_c.lognormal(0.0, OMEGA_CL, size=(N_STUDIES, 120))

cv_hat_12 = 100.0 * np.sqrt(np.exp(np.var(np.log(draws_12), axis=1, ddof=1)) - 1.0)
cv_hat_120 = 100.0 * np.sqrt(np.exp(np.var(np.log(draws_120), axis=1, ddof=1)) - 1.0)

cv12_lo = float(np.percentile(cv_hat_12, 2.5))
cv12_hi = float(np.percentile(cv_hat_12, 97.5))
cv120_lo = float(np.percentile(cv_hat_120, 2.5))
cv120_hi = float(np.percentile(cv_hat_120, 97.5))

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
''',
    ),
]


def build_expected(ns):
    import numpy as np

    cl = np.asarray(ns["cl_i"], dtype=float)
    v = np.asarray(ns["v_i"], dtype=float)
    dose = float(ns["DOSE"])
    ka = float(ns["KA"])
    om = float(ns["OMEGA_CL"])
    n_studies = int(ns["N_STUDIES"])
    seed_c = int(ns["SEED_C"])

    k = cl / v
    auc = dose / cl
    tmax = np.log(ka / k) / (ka - k)
    cmax = dose * ka / (v * (ka - k)) * (np.exp(-k * tmax) - np.exp(-ka * tmax))

    rng_c = np.random.default_rng(seed_c)
    d12 = rng_c.lognormal(0.0, om, size=(n_studies, 12))
    d120 = rng_c.lognormal(0.0, om, size=(n_studies, 120))
    cv12 = 100.0 * np.sqrt(np.exp(np.var(np.log(d12), axis=1, ddof=1)) - 1.0)
    cv120 = 100.0 * np.sqrt(np.exp(np.var(np.log(d120), axis=1, ddof=1)) - 1.0)

    return {
        "n": int(cl.size),
        "n_studies": n_studies,
        "cl_gm": float(np.exp(np.mean(np.log(cl)))),
        "cl_cv_pct": float(100.0 * np.sqrt(np.exp(np.var(np.log(cl), ddof=1)) - 1.0)),
        "cl_sd_log_pct": float(100.0 * np.std(np.log(cl), ddof=1)),
        "cl_mean": float(np.mean(cl)),
        "cl_median": float(np.median(cl)),
        "mean_over_median": float(np.mean(cl) / np.median(cl)),
        "mean_over_median_theory": float(np.exp(om ** 2 / 2.0)),
        "auc": [float(x) for x in auc],
        "cmax": [float(x) for x in cmax],
        "tmax": [float(x) for x in tmax],
        "auc_gm": float(np.exp(np.mean(np.log(auc)))),
        "auc_mean": float(np.mean(auc)),
        "auc_p5": float(np.percentile(auc, 5)),
        "auc_p95": float(np.percentile(auc, 95)),
        "fold_range": float(np.percentile(auc, 95) / np.percentile(auc, 5)),
        "cv12_med": float(np.median(cv12)),
        "cv120_med": float(np.median(cv120)),
        "cv12_lo": float(np.percentile(cv12, 2.5)),
        "cv12_hi": float(np.percentile(cv12, 97.5)),
        "cv120_lo": float(np.percentile(cv120, 2.5)),
        "cv120_hi": float(np.percentile(cv120, 97.5)),
    }


CHECK_CODE = '''

def _part_a():
    gm = _need("cl_gm")
    cv = _need("cl_cv_pct")
    mean = _need("cl_mean")
    ratio = _need("mean_over_median")
    theory = _need("mean_over_median_theory")

    for name, val in (("cl_gm", gm), ("cl_cv_pct", cv), ("cl_mean", mean),
                      ("mean_over_median", ratio),
                      ("mean_over_median_theory", theory)):
        if _as_scalar(val) is None:
            return "`%s` should be a single number." % name

    got = _as_scalar(gm)
    if _num_close(got, _EXP["cl_mean"], rtol=1e-6):
        return ("`cl_gm` is the arithmetic mean. The geometric mean is "
                "exp(mean(log(cl_i))), and it is slightly smaller.")
    if not _num_close(got, _EXP["cl_gm"], rtol=1e-6):
        return "`cl_gm` is not the geometric mean of cl_i."

    got = _as_scalar(cv)
    if _num_close(got, _EXP["cl_cv_pct"] / 100.0, rtol=1e-4):
        return "`cl_cv_pct` is a fraction. The task asks for a percentage."
    if _num_close(got, _EXP["cl_sd_log_pct"], rtol=1e-4):
        return ("`cl_cv_pct` is 100 times the SD of the logs, the approximation. The task "
                "asks for the exact form, 100 * sqrt(exp(var(log CL)) - 1).")
    if not _num_close(got, _EXP["cl_cv_pct"], rtol=1e-5):
        return ("`cl_cv_pct` is not the exact CV as a percentage. Use "
                "np.var(np.log(cl_i), ddof=1).")

    if not _num_close(_as_scalar(mean), _EXP["cl_mean"], rtol=1e-6):
        return "`cl_mean` is not the arithmetic mean of cl_i."

    got = _as_scalar(ratio)
    if _num_close(got, _EXP["mean_over_median_theory"], rtol=1e-6):
        return ("`mean_over_median` is the theoretical ratio. This one should be computed "
                "from the simulated numbers, mean divided by median, and it comes out a "
                "little different.")
    if not _num_close(got, _EXP["mean_over_median"], rtol=1e-6):
        return "`mean_over_median` is not the observed arithmetic mean divided by the median."

    got = _as_scalar(theory)
    if _num_close(got, _np.exp(OMEGA_CL ** 2), rtol=1e-9):
        return ("`mean_over_median_theory` is missing the division by 2 in the exponent. "
                "The arithmetic mean of a lognormal is the median times "
                "exp(omega ** 2 / 2).")
    if not _num_close(got, _EXP["mean_over_median_theory"], rtol=1e-9):
        return "`mean_over_median_theory` should be exp(OMEGA_CL ** 2 / 2)."
    return None


def _part_b():
    auc = _need("auc_i")
    cmax = _need("cmax_i")
    gm = _need("auc_gm")
    p5 = _need("auc_p5")
    p95 = _need("auc_p95")
    fold = _need("fold_range")

    for name, val in (("auc_i", auc), ("cmax_i", cmax)):
        arr = _as_array(val)
        if arr is None:
            return "`%s` should be an array of numbers." % name
        if arr.shape != (_EXP["n"],):
            return ("`%s` has shape %s but there are %d subjects."
                    % (name, arr.shape, _EXP["n"]))

    arr = _as_array(auc)
    if not _arr_close(arr, _EXP["auc"], rtol=1e-9):
        return ("`auc_i` is not DOSE / cl_i. For first order elimination the total "
                "exposure depends on dose and clearance only.")
    arr = _as_array(cmax)
    if _arr_close(arr, _EXP["tmax"], rtol=1e-6):
        return "`cmax_i` holds the peak TIMES. It should hold the peak concentrations."
    if not _arr_close(arr, _EXP["cmax"], rtol=1e-6):
        return ("`cmax_i` is not the peak concentration for each subject. Work out Tmax "
                "first, then evaluate the concentration equation at it.")

    for name, val in (("auc_gm", gm), ("auc_p5", p5), ("auc_p95", p95),
                      ("fold_range", fold)):
        if _as_scalar(val) is None:
            return "`%s` should be a single number." % name

    got = _as_scalar(gm)
    if _num_close(got, _EXP["auc_mean"], rtol=1e-6):
        return ("`auc_gm` is the arithmetic mean AUC. The geometric mean is what matches "
                "DOSE / CL_POP.")
    if not _num_close(got, _EXP["auc_gm"], rtol=1e-6):
        return "`auc_gm` is not the geometric mean of auc_i."

    got_lo = _as_scalar(p5)
    got_hi = _as_scalar(p95)
    if _num_close(got_lo, _EXP["auc_p95"], rtol=1e-6):
        return "`auc_p5` and `auc_p95` are the wrong way round."
    if not _num_close(got_lo, _EXP["auc_p5"], rtol=1e-6):
        return "`auc_p5` is not the 5th percentile of auc_i."
    if not _num_close(got_hi, _EXP["auc_p95"], rtol=1e-6):
        return "`auc_p95` is not the 95th percentile of auc_i."

    got = _as_scalar(fold)
    if _num_close(got, 1.0 / _EXP["fold_range"], rtol=1e-6):
        return "`fold_range` is upside down. It is the 95th divided by the 5th."
    if not _num_close(got, _EXP["fold_range"], rtol=1e-6):
        return "`fold_range` is not auc_p95 divided by auc_p5."
    return None


def _part_c():
    cv12 = _need("cv_hat_12")
    cv120 = _need("cv_hat_120")
    lo12 = _need("cv12_lo")
    hi12 = _need("cv12_hi")
    lo120 = _need("cv120_lo")
    hi120 = _need("cv120_hi")

    for name, val in (("cv_hat_12", cv12), ("cv_hat_120", cv120)):
        arr = _as_array(val)
        if arr is None:
            return "`%s` should be an array of CV estimates." % name
        if arr.size == 1:
            return ("`%s` is a single number. It should hold one estimate per simulated "
                    "study, so compute the variance along axis=1." % name)
        if arr.shape != (_EXP["n_studies"],):
            return ("`%s` has shape %s but there are N_STUDIES = %d studies."
                    % (name, arr.shape, _EXP["n_studies"]))
        if float(_np.nanmax(arr)) < 1.0:
            return ("`%s` looks like a fraction rather than a percentage. Multiply by 100."
                    % name)

    med12 = float(_np.median(_as_array(cv12)))
    med120 = float(_np.median(_as_array(cv120)))
    if not _num_close(med12, _EXP["cv12_med"], rtol=5e-2):
        return ("the median of `cv_hat_12` is %.2f, which is not where it should be. Each "
                "row of the draws is one study of twelve, so the variance is taken along "
                "axis=1 with ddof=1." % med12)
    if not _num_close(med120, _EXP["cv120_med"], rtol=5e-2):
        return "the median of `cv_hat_120` is not where it should be."
    if med120 < med12:
        return ("`cv_hat_120` has a lower median than `cv_hat_12`, which is backwards: the "
                "larger studies should centre closer to the truth, which is above both.")

    for name, val, want in (("cv12_lo", lo12, _EXP["cv12_lo"]),
                            ("cv12_hi", hi12, _EXP["cv12_hi"]),
                            ("cv120_lo", lo120, _EXP["cv120_lo"]),
                            ("cv120_hi", hi120, _EXP["cv120_hi"])):
        if _as_scalar(val) is None:
            return "`%s` should be a single number." % name
        if not _num_close(_as_scalar(val), want, rtol=6e-2):
            return ("`%s` is not the percentile it should be. The four values are the 2.5th "
                    "and 97.5th percentiles of each set of estimates." % name)

    width12 = _as_scalar(hi12) - _as_scalar(lo12)
    width120 = _as_scalar(hi120) - _as_scalar(lo120)
    if not width12 > 2.0 * width120:
        return ("the interval from the twelve subject studies should be more than twice as "
                "wide as the one from the hundred and twenty subject studies. Check that "
                "the two draws have the shapes (N_STUDIES, 12) and (N_STUDIES, 120).")
    return None
'''

# Functions this problem introduces. Keys must exist in tools/function_catalog.py.
FUNCTIONS = [
    "rng.lognormal",
    "np.percentile",
    "geometric mean and CV",
    "arithmetic against geometric mean",
    "Monte Carlo study simulation",
]
