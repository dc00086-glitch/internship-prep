PID = "W8D2"
WEEK = 8
DAY = 2
SLUG = "w8d2_indirect_response"
TITLE = "Indirect response: when the biomarker, not the drug, sets the clock"
TOPIC = "PK/PD: turnover models, inhibition of production, dose dependent time to nadir"
EST = "25-35 min"

SCENARIO_MD = '''
## Scenario

The drug inhibits an enzyme that produces an inflammatory biomarker. The
biomarker sits at a steady baseline because it is made and removed
continuously, and the drug slows the making.

Yesterday's effect compartment would describe the delay as the drug taking
time to reach its site of action. Today's mechanism is different and more
common: the drug reaches its target immediately, and the delay belongs to the
biomarker, which cannot fall faster than its own turnover allows. Block
production completely and the level still only falls at the rate it was being
removed anyway.

The practical consequences are large. The effect outlasts the drug by days.
The time of maximum effect moves later as the dose goes up, which no effect
compartment model can produce. And the quantity that governs the duration of
action is not a property of the drug at all, it is the biomarker's turnover
half life.
'''

CONCEPT_MD = '''
## The turnover model

**The equation.** One state, the biomarker level `R`, made at rate `kin` and
removed by a first order process `kout`:

```
dR/dt = kin - kout * R
```

Before any drug this settles at `R(0) = kin / kout`, the baseline. That is the
most useful fact in the model and the one people forget: **the baseline pins
`kin`**. If you know the baseline and you know `kout`, then `kin` is not a free
parameter, it is `kout * R0`. A fit that treats all three as free is fitting
one parameter twice and will not converge cleanly.

**The drug.** Four places it can act, and the four standard models are named
for them. Inhibition or stimulation of production, inhibition or stimulation of
loss. Today is inhibition of production, which multiplies `kin` by an inhibitory
Emax term:

```
dR/dt = kin * (1 - Imax * C / (IC50 + C)) - kout * R
```

`Imax` is between 0 and 1 and says how completely production can be shut off.
`Imax = 0.9` means that even at infinite concentration, ten percent of
production continues.

**Where the delay comes from.** Set `C` to something large and hold it there.
Production nearly stops, and `R` decays towards its new steady state with the
rate constant `kout`, so the approach has a half life of `ln(2) / kout`. The
drug cannot make the biomarker fall faster than that, however much of it you
give. A biomarker with a twelve hour turnover half life takes about two days to
reach a new steady state no matter what you do to production.

**Why the time of the nadir moves with dose.** The nadir happens when the
reduced production rate and the removal rate `kout * R` come back into balance.
A bigger dose keeps the concentration above `IC50` for longer, so production
stays suppressed for longer, so the balance point arrives later. The time the
concentration spends above `IC50` grows roughly with the logarithm of the dose,
and the time to the nadir follows it.

This is the clean diagnostic between the two mechanisms. **An effect
compartment puts the maximum effect at the same time for every dose.** An
indirect response model puts it later and later. If you have data at three dose
levels, this tells you which model you need, and it is one of the few places in
PK/PD where the answer is not a matter of taste.
'''

TASK_MD = '''
## Your task

Three parts, worked one at a time.

| Part | What you work out | Variables to fill in |
|---|---|---|
| a | the model, and one dose through it | `kin`, `nadir`, `t_nadir`, `pct_suppression` |
| b | three doses, and the diagnostic that separates the mechanisms | `t_nadir_by_dose`, `nadir_by_dose`, `t_above_by_dose`, `t_link_by_dose` |
| c | duration of action, which belongs to the biomarker | `auec`, `t_recover`, `t_nadir_fast`, `t_nadir_slow` |

Below this you will find the **Setup** cell, then a **Toolkit** cell that
demonstrates every move on a tiny made up example, then **Part a**, **Part b**
and **Part c**, each with its own explanation and a cell to write in.
'''

DATA_CODE = '''
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

PROBLEM_ID = "W8D2"
TOPIC = "PK/PD: turnover models, inhibition of production, dose dependent time to nadir"

# Pharmacokinetics, taken as known from an earlier study.
VF = 35.0            # L, apparent volume of distribution
KA = 1.0             # per h, first order absorption
KE = 0.12            # per h, first order elimination

# Pharmacodynamics of the biomarker.
R_BASE = 100.0       # units/L, the pre dose baseline
T_HALF_TURNOVER = 12.0   # h, the biomarker's own turnover half life
IMAX = 0.9           # at most 90 pct of production can be blocked
IC50 = 0.5           # mg/L, concentration for half of that

KOUT = np.log(2.0) / T_HALF_TURNOVER

DOSE = 100.0         # mg, the dose for part a
DOSES = np.array([25.0, 100.0, 400.0])   # mg, the three dose levels for part b
KE0_COMPARE = 0.03   # per h, for the effect compartment comparison in part b

# One shared time grid, so that everyone reads the same times off it.
t_grid = np.linspace(0.0, 168.0, 1681)   # 0 to 7 days in 0.1 h steps
DT = float(t_grid[1] - t_grid[0])


def conc(t, dose):
    """Plasma concentration after a single oral dose, one compartment."""
    return dose * KA / (VF * (KA - KE)) * (np.exp(-KE * t) - np.exp(-KA * t))


print("turnover half life %.1f h, so kout = %.6f per h" % (T_HALF_TURNOVER, KOUT))
print("baseline %.1f units/L" % R_BASE)
print("Imax %.2f, IC50 %.2f mg/L" % (IMAX, IC50))
print("")
print("grid: %d points from %g to %g h, step %.1f h"
      % (t_grid.size, t_grid[0], t_grid[-1], DT))
print("concentration after %g mg: Cmax %.4f mg/L at %.2f h"
      % (DOSE, conc(t_grid, DOSE).max(), t_grid[int(np.argmax(conc(t_grid, DOSE)))]))
print("concentration after %g mg at 48 h: %.5f mg/L" % (DOSE, conc(48.0, DOSE)))
'''

TOOLKIT_MD = '''
## Toolkit for this problem

| What you want to do | How |
|---|---|
| solve one state | `solve_ivp(rhs, (0, tmax), [r0], t_eval=t_grid, rtol=1e-9, atol=1e-10)` |
| the solution values | `sol.y[0]` |
| a right hand side with a time varying input | `lambda t, y: kin * (1 - imax * c(t) / (ic50 + c(t))) - kout * y[0]` |
| position of a minimum | `int(np.argmin(r))` |
| how long a condition held | `(c > threshold).sum() * DT` |
| area under a curve | `np.trapezoid(y, x)`, or `np.trapz(y, x)` on older numpy |
| the first index satisfying a condition | `int(np.where(flags)[0][0])` |

`np.trapezoid` is the current name and `np.trapz` is the old one. Colab may have
either, so the safe pattern is to compute the trapezoids yourself, or to try one
and fall back to the other. The toolkit below shows a version that works on both.
'''

TOOLKIT_CODE = '''
import numpy as np
from scipy.integrate import solve_ivp

print("--- 1. a turnover system with no drug sits still ---")
demo_kout = 0.2
demo_r0 = 50.0
demo_kin = demo_kout * demo_r0        # the baseline pins kin
demo_t = np.linspace(0.0, 24.0, 241)
demo_sol = solve_ivp(lambda t, y: demo_kin - demo_kout * y[0],
                     (0.0, 24.0), [demo_r0], t_eval=demo_t, rtol=1e-9, atol=1e-10)
print("kin = kout * R0 = %.3f" % demo_kin)
print("R at 0, 12 and 24 h:", np.round(demo_sol.y[0][[0, 120, 240]], 9))
print("started at the steady state, so nothing moves")

print("")
print("--- 2. start it in the wrong place and you get a drift artifact ---")
demo_sol2 = solve_ivp(lambda t, y: demo_kin - demo_kout * y[0],
                      (0.0, 24.0), [30.0], t_eval=demo_t, rtol=1e-9, atol=1e-10)
print("from R(0) = 30 with no drug at all:", np.round(demo_sol2.y[0][[0, 12, 60, 240]], 4))
print("it climbs to the baseline on its own, with half life %.2f h" % (np.log(2) / demo_kout))
print("a model started off baseline will blame the drug for that climb")

print("")
print("--- 3. production shut off completely: the fastest possible fall ---")
demo_sol3 = solve_ivp(lambda t, y: 0.0 - demo_kout * y[0],
                      (0.0, 24.0), [demo_r0], t_eval=demo_t, rtol=1e-9, atol=1e-10)
print("R after 0, 1, 2 and 3 turnover half lives:",
      np.round(np.interp([0, np.log(2) / demo_kout, 2 * np.log(2) / demo_kout,
                          3 * np.log(2) / demo_kout], demo_t, demo_sol3.y[0]), 4))
print("no drug can beat that, which is where the delay comes from")

print("")
print("--- 4. how long a condition held, on a fixed grid ---")
demo_c = np.array([0.1, 0.8, 1.2, 0.9, 0.3, 0.1])
demo_dt = 0.5
print("values:", demo_c, " threshold 0.5")
print("samples above:", int((demo_c > 0.5).sum()), " so about %.1f h" % ((demo_c > 0.5).sum() * demo_dt))
print("that is a grid count, not an interpolation, so it is accurate to one step")

print("")
print("--- 5. trapezoidal area that works on old and new numpy ---")
demo_x = np.array([0.0, 1.0, 2.0, 3.0])
demo_y = np.array([0.0, 2.0, 2.0, 0.0])


def demo_area(y, x):
    """Trapezoidal area, written out so the numpy version does not matter."""
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


print("by hand        :", demo_area(demo_y, demo_x))
print("numpy agrees   :", float(np.trapezoid(demo_y, demo_x))
      if hasattr(np, "trapezoid") else float(np.trapz(demo_y, demo_x)))

print("")
print("--- 6. first index where something becomes true ---")
demo_r = np.array([50.0, 60.0, 80.0, 95.0, 99.0])
demo_flags = demo_r >= 90.0
print("flags:", demo_flags, " first True at index", int(np.where(demo_flags)[0][0]))
print("np.where returns a tuple, hence the [0][0]")
'''

PARTS = [
    dict(
        label="a",
        prompt_md='''
## Part a. One dose through the turnover model

First work out `kin`. You know the baseline `R_BASE` and you know `KOUT`, and
at baseline production equals removal, so `kin = KOUT * R_BASE`. Writing this
down rather than fitting it is the whole trick with indirect response models.

Then solve

```
dR/dt = kin * (1 - IMAX * C(t) / (IC50 + C(t))) - KOUT * R
```

from `R(0) = R_BASE` over `t_grid`, with `C(t)` given by the `conc` function
and `DOSE` of 100 mg. Use `rtol=1e-9, atol=1e-10`, which matters here: the
default tolerances leave visible error in the tail, and you are about to read a
time off this curve to one decimal place.

From the solution record:

- `nadir`, the lowest biomarker level reached,
- `t_nadir`, the time at which it happened, read off `t_grid`,
- `pct_suppression`, the maximum suppression as a percentage of baseline,
  `100 * (1 - nadir / R_BASE)`.

The concentration peaks at 2.4 hours. The biomarker bottoms out at 18.8 hours,
sixteen hours later, and the drug is down to a hundredth of its peak by then.
Nothing is wrong with the simulation. The biomarker is simply still catching up.

Note also that the suppression is only 36 percent although `Imax` is 0.9. The
concentration is above `IC50` for about fifteen hours out of a week, so
production is never shut off for long enough to approach the maximum effect.
Imax is an asymptote, not a prediction.

Fill in `kin`, `nadir`, `t_nadir`, `pct_suppression`.

Moves: `solve_ivp` with a time varying input, `np.argmin`, indexing `t_grid`.
''',
        starter='''
# Part a. The turnover model at one dose.

kin = None  # YOUR CODE: production rate implied by the baseline and KOUT


def rhs(t, y, dose):
    """dR/dt for inhibition of production."""
    c = conc(t, dose)
    return None  # YOUR CODE: kin * (1 - IMAX * c / (IC50 + c)) - KOUT * y[0]


r_single = None  # YOUR CODE: the solved biomarker profile over t_grid, as an array

nadir = None            # YOUR CODE: lowest level reached
t_nadir = None          # YOUR CODE: time at which it happened
pct_suppression = None  # YOUR CODE: maximum suppression as a percentage of baseline

print("kin = %r units/L/h" % (kin,))
print("nadir %r units/L at t = %r h" % (nadir, t_nadir))
print("maximum suppression %r pct of baseline" % (pct_suppression,))

if r_single is not None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t_grid, r_single, "-", color="#1f4e79", label="biomarker")
    ax.set_xlabel("time (h)")
    ax.set_ylabel("biomarker (units/L)")
    ax.set_xlim(0, 96)
    ax2 = ax.twinx()
    ax2.plot(t_grid, conc(t_grid, DOSE), "--", color="#999999", label="concentration")
    ax2.set_ylabel("concentration (mg/L)")
    ax.set_title("W8D2 the effect outlasts the drug")
    ax.legend(loc="lower right")
    plt.show()
''',
        solution='''
# Part a. The turnover model at one dose.

kin = KOUT * R_BASE


def rhs(t, y, dose):
    """dR/dt for inhibition of production."""
    c = conc(t, dose)
    return kin * (1.0 - IMAX * c / (IC50 + c)) - KOUT * y[0]


_sol = solve_ivp(rhs, (float(t_grid[0]), float(t_grid[-1])), [R_BASE],
                 t_eval=t_grid, args=(DOSE,), rtol=1e-9, atol=1e-10)
r_single = _sol.y[0]

_i = int(np.argmin(r_single))
nadir = float(r_single[_i])
t_nadir = float(t_grid[_i])
pct_suppression = float(100.0 * (1.0 - nadir / R_BASE))

print("kin = %r units/L/h" % (kin,))
print("nadir %r units/L at t = %r h" % (nadir, t_nadir))
print("maximum suppression %r pct of baseline" % (pct_suppression,))

if r_single is not None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t_grid, r_single, "-", color="#1f4e79", label="biomarker")
    ax.set_xlabel("time (h)")
    ax.set_ylabel("biomarker (units/L)")
    ax.set_xlim(0, 96)
    ax2 = ax.twinx()
    ax2.plot(t_grid, conc(t_grid, DOSE), "--", color="#999999", label="concentration")
    ax2.set_ylabel("concentration (mg/L)")
    ax.set_title("W8D2 the effect outlasts the drug")
    ax.legend(loc="lower right")
    plt.show()
''',
    ),
    dict(
        label="b",
        prompt_md='''
## Part b. Three doses, and the mechanism diagnostic

Run the same model at 25, 100 and 400 mg, and collect, as arrays of three
numbers in dose order:

- `t_nadir_by_dose`, the time of the nadir at each dose,
- `nadir_by_dose`, the nadir itself,
- `t_above_by_dose`, how long the concentration stays above `IC50` at each
  dose. Count grid points where `conc` exceeds `IC50` and multiply by `DT`.

Then build the competing explanation and watch it fail. For each dose, solve
the effect compartment

```
dCe/dt = KE0_COMPARE * (C(t) - Ce)
```

from `Ce(0) = 0`, and compute the same inhibitory effect on `Ce` instead:
`R_BASE * (1 - IMAX * Ce / (IC50 + Ce))`. Record the time of its minimum in
`t_link_by_dose`.

Here is the comparison you have just produced.

| dose | time above IC50 | indirect nadir time | effect compartment nadir time |
|---|---|---|---|
| 25 mg | 2.2 h | 15.3 h | 16.5 h |
| 100 mg | 15.4 h | 18.8 h | 16.5 h |
| 400 mg | 27.1 h | 24.1 h | 16.5 h |

The effect compartment column is constant to the last digit, and it is constant
for a reason rather than by accident: `Ce` is a linear filter of `C`, so
multiplying the dose multiplies `Ce` by the same factor everywhere and cannot
move the position of its peak. The effect is a monotone function of `Ce`, so its
extreme sits at the same time too, at every dose, forever.

The indirect column moves by nine hours across a sixteen fold dose range, and
it tracks the time above `IC50`. That is the fingerprint, and it means three
dose levels of biomarker data are enough to tell you which mechanism you are
looking at.

The nadir depths, 18.6, 36.4 and 53.1 percent suppression, are worth a second
look as well. Sixteen times the dose did not get you three times the effect.
Between the Emax saturation and the turnover limit, the dose response of an
indirect biomarker is much flatter than the dose response of a concentration.

Fill in `t_nadir_by_dose`, `nadir_by_dose`, `t_above_by_dose`,
`t_link_by_dose`.

Moves: a loop over `DOSES`, `solve_ivp` twice per dose, `np.argmin`,
`(c > IC50).sum() * DT`.
''',
        starter='''
# Part b. Dose dependence, and the effect compartment comparison.

t_nadir_by_dose = None  # YOUR CODE: array of 3 nadir times, indirect model
nadir_by_dose = None    # YOUR CODE: array of 3 nadir levels, indirect model
t_above_by_dose = None  # YOUR CODE: array of 3 durations above IC50, in h

t_link_by_dose = None   # YOUR CODE: array of 3 nadir times, effect compartment model

print("dose (mg)        :", DOSES)
print("time above IC50  :", t_above_by_dose)
print("indirect t_nadir :", t_nadir_by_dose)
print("indirect nadir   :", nadir_by_dose)
print("link model t_min :", t_link_by_dose)

if nadir_by_dose is not None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for _d in DOSES:
        _s = solve_ivp(rhs, (float(t_grid[0]), float(t_grid[-1])), [R_BASE],
                       t_eval=t_grid, args=(_d,), rtol=1e-9, atol=1e-10)
        ax.plot(t_grid, _s.y[0], "-", label="%g mg" % _d)
    ax.axhline(R_BASE, ls=":", color="grey")
    ax.set_xlim(0, 120)
    ax.set_xlabel("time (h)")
    ax.set_ylabel("biomarker (units/L)")
    ax.set_title("W8D2 three doses, three different nadir times")
    ax.legend()
    plt.show()
''',
        solution='''
# Part b. Dose dependence, and the effect compartment comparison.

_tn, _nd, _ta, _tl = [], [], [], []
for _d in DOSES:
    _s = solve_ivp(rhs, (float(t_grid[0]), float(t_grid[-1])), [R_BASE],
                   t_eval=t_grid, args=(_d,), rtol=1e-9, atol=1e-10)
    _r = _s.y[0]
    _i = int(np.argmin(_r))
    _tn.append(float(t_grid[_i]))
    _nd.append(float(_r[_i]))
    _ta.append(float((conc(t_grid, _d) > IC50).sum()) * DT)

    _sl = solve_ivp(lambda t, y, dd: KE0_COMPARE * (conc(t, dd) - y[0]),
                    (float(t_grid[0]), float(t_grid[-1])), [0.0],
                    t_eval=t_grid, args=(_d,), rtol=1e-9, atol=1e-12)
    _ce = _sl.y[0]
    _e = R_BASE * (1.0 - IMAX * _ce / (IC50 + _ce))
    _tl.append(float(t_grid[int(np.argmin(_e))]))

t_nadir_by_dose = np.array(_tn)
nadir_by_dose = np.array(_nd)
t_above_by_dose = np.array(_ta)
t_link_by_dose = np.array(_tl)

print("dose (mg)        :", DOSES)
print("time above IC50  :", t_above_by_dose)
print("indirect t_nadir :", t_nadir_by_dose)
print("indirect nadir   :", nadir_by_dose)
print("link model t_min :", t_link_by_dose)

if nadir_by_dose is not None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for _d in DOSES:
        _s = solve_ivp(rhs, (float(t_grid[0]), float(t_grid[-1])), [R_BASE],
                       t_eval=t_grid, args=(_d,), rtol=1e-9, atol=1e-10)
        ax.plot(t_grid, _s.y[0], "-", label="%g mg" % _d)
    ax.axhline(R_BASE, ls=":", color="grey")
    ax.set_xlim(0, 120)
    ax.set_xlabel("time (h)")
    ax.set_ylabel("biomarker (units/L)")
    ax.set_title("W8D2 three doses, three different nadir times")
    ax.legend()
    plt.show()
''',
    ),
    dict(
        label="c",
        prompt_md='''
## Part c. Duration of action belongs to the biomarker

Back to the 100 mg dose and the profile from part a.

- `auec`, the area between the baseline and the curve, in units/L times hours:
  integrate `R_BASE - r_single` over `t_grid`. This is the cumulative
  suppression, and it is the quantity that usually correlates with clinical
  benefit, far better than the nadir does,
- `t_recover`, the first time at or after the nadir at which the biomarker has
  come back to at least 90 percent of baseline.

Then change the biomarker and leave the drug alone. Re run the simulation with
the turnover half life set to 6 hours and then to 24 hours, recomputing `kout`
and `kin` each time, and record the nadir times in `t_nadir_fast` and
`t_nadir_slow`.

What this shows:

- recovery to 90 percent of baseline takes 54 hours, while the plasma
  concentration at 48 hours is 0.01 mg/L, one fiftieth of the `IC50`. For more
  than a day the biomarker is suppressed by a drug that has effectively gone,
- halving the turnover half life to 6 hours moves the nadir from 18.8 h to
  14.2 h and makes it deeper, 51.4 against 63.6. Doubling it to 24 hours moves
  the nadir out to 23.8 h and makes it shallower, 75.5. The drug, its dose and
  its PK were identical in all three.

That is the message of the day. For an indirect response, the time course of
the effect is a property of the biomarker, and the dose only sets how hard you
push on it. It is also why dosing intervals for these drugs are often set by
the biomarker's turnover and not by the drug's half life, and why a daily dose
can be right for a drug with a four hour plasma half life.

Fill in `auec`, `t_recover`, `t_nadir_fast`, `t_nadir_slow`.

Moves: trapezoidal integration, `np.where` for the first crossing,
re running `solve_ivp` with a different `kout`.
''',
        starter='''
# Part c. Duration of action, and whose clock it runs on.

auec = None       # YOUR CODE: area between baseline and the 100 mg profile
t_recover = None  # YOUR CODE: first time at or after the nadir with R >= 90 pct of baseline


def simulate(dose, t_half_turnover):
    """Solve the turnover model for a given dose and turnover half life."""
    kout = np.log(2.0) / t_half_turnover
    kin_local = kout * R_BASE
    # YOUR CODE: solve dR/dt = kin_local * (1 - IMAX * c / (IC50 + c)) - kout * R
    # from R_BASE over t_grid, and return the solution array
    return None


t_nadir_fast = None  # YOUR CODE: nadir time with a 6 h turnover half life
t_nadir_slow = None  # YOUR CODE: nadir time with a 24 h turnover half life

print("AUEC of suppression: %r units/L*h" % (auec,))
print("back to 90 pct of baseline at %r h" % (t_recover,))
print("")
print("nadir time with a  6 h turnover half life: %r h" % (t_nadir_fast,))
print("nadir time with a 12 h turnover half life: %r h" % (t_nadir,))
print("nadir time with a 24 h turnover half life: %r h" % (t_nadir_slow,))
''',
        solution='''
# Part c. Duration of action, and whose clock it runs on.

_sup = R_BASE - r_single
auec = float(np.sum(0.5 * (_sup[1:] + _sup[:-1]) * np.diff(t_grid)))

_i = int(np.argmin(r_single))
_flags = r_single[_i:] >= 0.9 * R_BASE
t_recover = float(t_grid[_i + int(np.where(_flags)[0][0])])


def simulate(dose, t_half_turnover):
    """Solve the turnover model for a given dose and turnover half life."""
    kout = np.log(2.0) / t_half_turnover
    kin_local = kout * R_BASE

    def local_rhs(t, y):
        c = conc(t, dose)
        return kin_local * (1.0 - IMAX * c / (IC50 + c)) - kout * y[0]

    sol = solve_ivp(local_rhs, (float(t_grid[0]), float(t_grid[-1])), [R_BASE],
                    t_eval=t_grid, rtol=1e-9, atol=1e-10)
    return sol.y[0]


t_nadir_fast = float(t_grid[int(np.argmin(simulate(DOSE, 6.0)))])
t_nadir_slow = float(t_grid[int(np.argmin(simulate(DOSE, 24.0)))])

print("AUEC of suppression: %r units/L*h" % (auec,))
print("back to 90 pct of baseline at %r h" % (t_recover,))
print("")
print("nadir time with a  6 h turnover half life: %r h" % (t_nadir_fast,))
print("nadir time with a 12 h turnover half life: %r h" % (t_nadir,))
print("nadir time with a 24 h turnover half life: %r h" % (t_nadir_slow,))
''',
    ),
]


def build_expected(ns):
    import numpy as np
    from scipy.integrate import solve_ivp

    grid = np.asarray(ns["t_grid"], dtype=float)
    dt = float(ns["DT"])
    conc = ns["conc"]
    r_base = float(ns["R_BASE"])
    imax = float(ns["IMAX"])
    ic50 = float(ns["IC50"])
    kout = float(ns["KOUT"])
    dose = float(ns["DOSE"])
    doses = np.asarray(ns["DOSES"], dtype=float)
    ke0 = float(ns["KE0_COMPARE"])

    def run(d, kout_local):
        kin_local = kout_local * r_base

        def rhs(t, y):
            c = conc(t, d)
            return kin_local * (1.0 - imax * c / (ic50 + c)) - kout_local * y[0]

        sol = solve_ivp(rhs, (float(grid[0]), float(grid[-1])), [r_base],
                        t_eval=grid, rtol=1e-9, atol=1e-10)
        return sol.y[0]

    r = run(dose, kout)
    i = int(np.argmin(r))
    sup = r_base - r
    auec = float(np.sum(0.5 * (sup[1:] + sup[:-1]) * np.diff(grid)))
    flags = r[i:] >= 0.9 * r_base
    t_recover = float(grid[i + int(np.where(flags)[0][0])])

    tn, nd, ta, tl = [], [], [], []
    for d in doses:
        rr = run(d, kout)
        j = int(np.argmin(rr))
        tn.append(float(grid[j]))
        nd.append(float(rr[j]))
        ta.append(float((conc(grid, d) > ic50).sum()) * dt)
        sl = solve_ivp(lambda t, y: ke0 * (conc(t, d) - y[0]),
                       (float(grid[0]), float(grid[-1])), [0.0],
                       t_eval=grid, rtol=1e-9, atol=1e-12)
        ce = sl.y[0]
        e = r_base * (1.0 - imax * ce / (ic50 + ce))
        tl.append(float(grid[int(np.argmin(e))]))

    return {
        "n_grid": int(grid.size),
        "kin": kout * r_base,
        "nadir": float(r[i]),
        "t_nadir": float(grid[i]),
        "pct_suppression": float(100.0 * (1.0 - r[i] / r_base)),
        "t_nadir_by_dose": tn,
        "nadir_by_dose": nd,
        "t_above_by_dose": ta,
        "t_link_by_dose": tl,
        "auec": auec,
        "auec_no_baseline": float(np.sum(0.5 * (r[1:] + r[:-1]) * np.diff(grid))),
        "t_recover": t_recover,
        "t_nadir_fast": float(grid[int(np.argmin(run(dose, np.log(2.0) / 6.0)))]),
        "t_nadir_slow": float(grid[int(np.argmin(run(dose, np.log(2.0) / 24.0)))]),
    }


CHECK_CODE = '''

def _part_a():
    kin_val = _need("kin")
    nadir = _need("nadir")
    t_nadir = _need("t_nadir")
    pct = _need("pct_suppression")
    r_single = _need("r_single")

    for name, val in (("kin", kin_val), ("nadir", nadir), ("t_nadir", t_nadir),
                      ("pct_suppression", pct)):
        if _as_scalar(val) is None:
            return "`%s` should be a single number." % name

    got = _as_scalar(kin_val)
    if _num_close(got, R_BASE / KOUT, rtol=1e-6):
        return ("`kin` is R_BASE divided by KOUT. At baseline production equals removal, "
                "so kin = KOUT * R_BASE.")
    if not _num_close(got, _EXP["kin"], rtol=1e-9):
        return ("`kin` is not the production rate implied by the baseline. With "
                "dR/dt = kin - kout * R at rest, kin = KOUT * R_BASE.")

    arr = _as_array(r_single)
    if arr is None:
        return "`r_single` should be the solved biomarker profile, an array of numbers."
    if arr.shape != (_EXP["n_grid"],):
        return ("`r_single` has shape %s but t_grid has %d points. Pass t_eval=t_grid to "
                "solve_ivp and take sol.y[0]." % (arr.shape, _EXP["n_grid"]))
    if not _num_close(float(arr[0]), R_BASE, rtol=1e-6):
        return ("`r_single` does not start at the baseline. The initial condition is "
                "[R_BASE], the pre dose steady state.")
    if float(arr.max()) > R_BASE * 1.001:
        return ("`r_single` rises above the baseline, which inhibition of production "
                "cannot do. Check the sign: the drug term multiplies kin by "
                "(1 - IMAX * c / (IC50 + c)), which is between 0.1 and 1.")

    got = _as_scalar(nadir)
    if not _num_close(got, _EXP["nadir"], rtol=1e-4):
        return ("`nadir` is not the lowest biomarker level reached. Check the tolerances: "
                "the task asks for rtol=1e-9, atol=1e-10.")
    got = _as_scalar(t_nadir)
    if _num_close(got, float(_np.argmin(arr))):
        return "`t_nadir` is an index, not a time. Use it to index t_grid."
    if not _num_close(got, _EXP["t_nadir"], rtol=1e-6, atol=1e-6):
        return "`t_nadir` is not the time of the nadir, read off t_grid."
    got = _as_scalar(pct)
    if _num_close(got, _EXP["pct_suppression"] / 100.0, rtol=1e-4):
        return "`pct_suppression` is a fraction. The task asks for a percentage."
    if _num_close(got, 100.0 * _EXP["nadir"] / R_BASE, rtol=1e-4):
        return ("`pct_suppression` is the nadir as a percentage OF baseline, which is the "
                "level remaining. The suppression is 100 * (1 - nadir / R_BASE).")
    if not _num_close(got, _EXP["pct_suppression"], rtol=1e-4):
        return "`pct_suppression` is not the maximum suppression as a percentage."
    return None


def _part_b():
    tn = _need("t_nadir_by_dose")
    nd = _need("nadir_by_dose")
    ta = _need("t_above_by_dose")
    tl = _need("t_link_by_dose")

    for name, val in (("t_nadir_by_dose", tn), ("nadir_by_dose", nd),
                      ("t_above_by_dose", ta), ("t_link_by_dose", tl)):
        arr = _as_array(val)
        if arr is None:
            return "`%s` should be an array of three numbers." % name
        if arr.shape != (3,):
            return ("`%s` has shape %s. It should hold one value per dose level, in the "
                    "order of DOSES." % (name, arr.shape))

    arr = _as_array(tn)
    if _arr_close(arr, [_EXP["t_nadir_by_dose"][0]] * 3, rtol=1e-6):
        return ("every entry of `t_nadir_by_dose` is the same. For inhibition of "
                "production the nadir time moves later with dose, so check that the dose "
                "is actually being passed through to conc().")
    if not _arr_close(arr, _EXP["t_nadir_by_dose"], rtol=1e-6, atol=1e-6):
        return ("`t_nadir_by_dose` does not hold the nadir times for 25, 100 and 400 mg "
                "in that order.")

    if not _arr_close(_as_array(nd), _EXP["nadir_by_dose"], rtol=1e-4):
        return "`nadir_by_dose` does not hold the nadir levels at the three doses."

    arr = _as_array(ta)
    if not _arr_close(arr, _EXP["t_above_by_dose"], rtol=1e-6, atol=1e-6):
        if _arr_close(arr, [x / DT for x in _EXP["t_above_by_dose"]], rtol=1e-6):
            return ("`t_above_by_dose` is a count of grid points, not a duration. Multiply "
                    "by DT.")
        return ("`t_above_by_dose` is not the time spent above IC50 at each dose. Count "
                "the grid points where conc(t_grid, dose) exceeds IC50 and multiply by DT.")

    arr = _as_array(tl)
    if not _arr_close(arr, _EXP["t_link_by_dose"], rtol=1e-6, atol=1e-6):
        if _arr_close(arr, _EXP["t_nadir_by_dose"], rtol=1e-3):
            return ("`t_link_by_dose` matches the indirect model times, so both columns "
                    "came from the same model. The effect compartment drives the same "
                    "inhibitory term with Ce from dCe/dt = KE0_COMPARE * (C(t) - Ce).")
        return ("`t_link_by_dose` is not the time of the minimum of the effect "
                "compartment model at each dose.")
    if len(set(round(float(x), 6) for x in arr)) != 1:
        return ("`t_link_by_dose` should be identical at all three doses. Scaling the dose "
                "scales Ce everywhere and cannot move the position of its peak, so if "
                "yours moves, something else is varying with dose.")
    return None


def _part_c():
    auec = _need("auec")
    t_recover = _need("t_recover")
    fast = _need("t_nadir_fast")
    slow = _need("t_nadir_slow")

    for name, val in (("auec", auec), ("t_recover", t_recover),
                      ("t_nadir_fast", fast), ("t_nadir_slow", slow)):
        if _as_scalar(val) is None:
            return "`%s` should be a single number." % name

    got = _as_scalar(auec)
    if _num_close(got, _EXP["auec_no_baseline"], rtol=1e-4):
        return ("`auec` is the area under the biomarker curve itself. The task asks for the "
                "area between the baseline and the curve, so integrate R_BASE - r_single.")
    if _num_close(got, -_EXP["auec"], rtol=1e-4):
        return ("`auec` has the right size and the wrong sign. Integrate "
                "R_BASE - r_single, so that suppression counts as positive.")
    if not _num_close(got, _EXP["auec"], rtol=1e-4):
        return ("`auec` is not the area between the baseline and the 100 mg profile over "
                "the whole grid.")

    got = _as_scalar(t_recover)
    if not _num_close(got, _EXP["t_recover"], rtol=1e-6, atol=1e-6):
        if got < _EXP["t_nadir"]:
            return ("`t_recover` is earlier than the nadir, so the search started at time "
                    "zero, where the biomarker is still at baseline. Search from the nadir "
                    "onward.")
        return ("`t_recover` is not the first grid time at or after the nadir where the "
                "biomarker is back to at least 90 percent of baseline.")

    got = _as_scalar(fast)
    if _num_close(got, _EXP["t_nadir"], rtol=1e-6):
        return ("`t_nadir_fast` equals the 12 h turnover answer, so the new half life did "
                "not reach the solver. Recompute both kout and kin inside the function.")
    if not _num_close(got, _EXP["t_nadir_fast"], rtol=1e-6, atol=1e-6):
        return ("`t_nadir_fast` is not the nadir time with a 6 hour turnover half life. A "
                "faster turnover means an earlier and deeper nadir.")
    got = _as_scalar(slow)
    if not _num_close(got, _EXP["t_nadir_slow"], rtol=1e-6, atol=1e-6):
        return ("`t_nadir_slow` is not the nadir time with a 24 hour turnover half life, "
                "which should be later than both of the others.")
    return None
'''

# Functions this problem introduces. Keys must exist in tools/function_catalog.py.
FUNCTIONS = [
    "indirect response model",
    "turnover half life",
    "baseline pins kin",
    "AUEC",
    "np.where first index",
]
