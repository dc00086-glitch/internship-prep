PID = "W3D2"
WEEK = 3
DAY = 2
SLUG = "w3d2_scipy_geometric_stats"
TITLE = "AUC is lognormal, so report the geometric mean"
TOPIC = "scipy.stats: lognormal data, geometric mean and CV, normality testing"
EST = "20-30 min"

SCENARIO_MD = '''
## Scenario

Twenty-four subjects took a single oral dose and you have each one's AUC, the
total exposure in mg*h/L. You have been asked for a one line summary of
central tendency and variability to go at the top of the report.

Exposure parameters are almost always lognormal. Clearance and volume vary
multiplicatively between people, not additively: one subject does not have
"20 units more clearance" than another, they have "30 percent more". Divide
through and the ratios are symmetric even though the raw values are not.

That is why regulatory PK summaries report the **geometric** mean and the
**geometric CV**, and why bioequivalence is assessed on log-transformed AUC
and Cmax. Reporting an arithmetic mean and SD for this data is not quite
wrong, but it describes a bell curve that is not the one you have.
'''

CONCEPT_MD = '''
## Multiplicative variability, and the statistics that match it

**What lognormal means.** A quantity is lognormal when its **logarithm** is
normal. Equivalently, it is what you get when many small effects multiply
rather than add. Clearance depends on liver blood flow times enzyme content
times free fraction, each varying by some percentage, so the product is
lognormal almost by construction. The result is strictly positive, has a long
right tail, and has a spread that grows with the magnitude.

**The geometric mean** is the mean computed in log space and brought back:

```python
gm = np.exp(np.mean(np.log(x)))
```

It equals the median of a perfect lognormal distribution, it is always less
than or equal to the arithmetic mean, and it is the value a ratio based
comparison is actually about. When a label says exposure increased 1.8 fold,
that is a ratio of geometric means.

**The geometric CV** is the one people get wrong, because it is not the
ordinary CV of anything. It is derived from the SD **of the logs**:

```python
sd_log = np.std(np.log(x), ddof=1)
geo_cv = 100 * np.sqrt(np.exp(sd_log ** 2) - 1)
```

Taking `100 * SD / mean` on the raw values gives you a different number and
does not have the interpretation of a percentage spread on the ratio scale. If
you have ever wondered why a PK table shows a CV of 45 percent that does not
match anything you can compute from the mean and SD, this is why.

**Testing normality, and what the test is worth.** `stats.shapiro` tests the
null hypothesis that a sample came from a normal distribution. A small p value
is evidence against normality. Two cautions that matter more than the test
itself. With a small sample it has almost no power, so a large p means "not
enough data to object", not "normal". With a very large sample it rejects on
trivial, harmless departures. So treat it as one input alongside a look at the
data, not as a gate that decides your analysis.

Here it will do its job: the raw AUCs should fail and the logs should pass,
which is the expected signature of lognormal data and a tidy confirmation of
what the biology already told you.
'''

TASK_MD = '''
## Your task

Three parts, worked one at a time.

| Part | What you work out | Variables to fill in |
|---|---|---|
| a | the arithmetic summary, the one people reach for | `arith_mean`, `arith_sd`, `arith_cv` |
| b | the geometric summary, the one that fits | `geo_mean`, `sd_log`, `geo_cv` |
| c | whether the data backs that up | `p_raw`, `p_log`, `use_log` |

Below this you will find the **Setup** cell, then a **Toolkit** cell that
demonstrates every move on a tiny made up example, then **Part a**, **Part b**
and **Part c**, each with its own explanation and a cell to write in.
'''

DATA_CODE = '''
import numpy as np
import pandas as pd
from scipy import stats

PROBLEM_ID = "W3D2"
TOPIC = "scipy.stats: lognormal data, geometric mean and CV, normality testing"

rng = np.random.default_rng(3102)

N_SUBJECTS = 24
# True geometric mean 48 mg*h/L, with 45 percent geometric CV.
_sd_log_true = np.sqrt(np.log(1.0 + 0.45 ** 2))
auc = np.round(rng.lognormal(mean=np.log(48.0), sigma=_sd_log_true, size=N_SUBJECTS), 2)

pk = pd.DataFrame({
    "SUBJECT": ["B%02d" % i for i in range(1, N_SUBJECTS + 1)],
    "AUC": auc,
})

print("pk:", pk.shape)
print(pk.to_string(index=False))
print("")
print("sorted AUC:", np.sort(auc))
print("")
print("notice the long right tail: a few subjects sit far above the rest")
'''

TOOLKIT_MD = '''
## Toolkit for this problem

| What you want to do | How |
|---|---|
| natural log of an array | `np.log(x)`, and back with `np.exp(x)` |
| arithmetic mean and sample SD | `np.mean(x)`, `np.std(x, ddof=1)` |
| ordinary CV as a percent | `100 * np.std(x, ddof=1) / np.mean(x)` |
| geometric mean | `np.exp(np.mean(np.log(x)))` |
| SD of the logs | `np.std(np.log(x), ddof=1)` |
| geometric CV as a percent | `100 * np.sqrt(np.exp(sd_log ** 2) - 1)` |
| test for normality | `stats.shapiro(x)`, then `.pvalue` |

The geometric CV comes from the SD **of the logs**, not from the raw SD. There
is no way to get it from the arithmetic mean and SD alone, which is why the
two summaries have to be computed separately.
'''

TOOLKIT_CODE = '''
import numpy as np
from scipy import stats

# A tiny made up right skewed sample. Nothing here is the answer.
demo = np.array([10.0, 12.0, 14.0, 15.0, 18.0, 22.0, 30.0, 61.0])
print("demo       :", demo)
print("sorted, the tail is on the right")

print("")
print("--- 1. arithmetic summary ---")
demo_m = float(np.mean(demo))
demo_sd = float(np.std(demo, ddof=1))
print("mean       : %.3f" % demo_m)
print("SD         : %.3f" % demo_sd)
print("CV percent : %.2f" % (100.0 * demo_sd / demo_m))
print("median     : %.3f   <- well below the mean, which is what skew does" % float(np.median(demo)))

print("")
print("--- 2. working in log space ---")
demo_log = np.log(demo)
print("np.log(demo):", np.round(demo_log, 3))
print("mean of logs: %.4f" % float(np.mean(demo_log)))
print("exp of that : %.3f   <- the geometric mean" % float(np.exp(np.mean(demo_log))))
print("")
print("geometric mean is always <= arithmetic mean:")
print("  %.3f <= %.3f" % (float(np.exp(np.mean(demo_log))), demo_m))

print("")
print("--- 3. geometric CV comes from the SD of the LOGS ---")
demo_sdlog = float(np.std(demo_log, ddof=1))
demo_gcv = 100.0 * np.sqrt(np.exp(demo_sdlog ** 2) - 1.0)
print("SD of logs     : %.4f" % demo_sdlog)
print("geometric CV   : %.2f percent" % demo_gcv)
print("ordinary CV    : %.2f percent   <- a different number entirely"
      % (100.0 * demo_sd / demo_m))

print("")
print("--- 4. shapiro tests normality, small p means NOT normal ---")
demo_raw = stats.shapiro(demo)
demo_logged = stats.shapiro(demo_log)
print("raw  : W = %.4f  p = %.4f" % (demo_raw.statistic, demo_raw.pvalue))
print("logs : W = %.4f  p = %.4f" % (demo_logged.statistic, demo_logged.pvalue))
print("")
print("with only 8 values this test has very little power, so do not read too")
print("much into either number. It is one input, not a decision procedure.")
'''

PARTS = [
    dict(
        label="a",
        prompt_md='''
## Part a. The arithmetic summary

Start with the summary most people would write without thinking, so you have
something to compare against.

Compute the arithmetic mean of the AUC values, the sample SD with `n - 1`, and
the ordinary coefficient of variation as a **percent**, which is
`100 * SD / mean`.

Keep these as plain Python floats rather than numpy scalars, which is what
`float(...)` around the result does. It matters less than it used to, but it
keeps the printed output clean.

Compare the mean you get against the median of the same data. The gap between
them is the skew, and it is the reason the rest of this problem exists.

Fill in `arith_mean`, `arith_sd`, `arith_cv`, all plain floats.

Moves: `np.mean`, `np.std(..., ddof=1)`, arithmetic, `float(...)`.
''',
        starter='''
# Part a. The arithmetic summary.

arith_mean = None   # YOUR CODE: arithmetic mean of auc
arith_sd = None     # YOUR CODE: sample SD, so ddof=1
arith_cv = None     # YOUR CODE: 100 * SD / mean

print("arithmetic mean : %s" % arith_mean)
print("arithmetic SD   : %s" % arith_sd)
print("CV percent      : %s" % arith_cv)
print("median for comparison: %.3f" % float(np.median(auc)))
''',
        solution='''
# Part a. The arithmetic summary.

arith_mean = float(np.mean(auc))
arith_sd = float(np.std(auc, ddof=1))
arith_cv = 100.0 * arith_sd / arith_mean

print("arithmetic mean : %s" % arith_mean)
print("arithmetic SD   : %s" % arith_sd)
print("CV percent      : %s" % arith_cv)
print("median for comparison: %.3f" % float(np.median(auc)))
''',
    ),
    dict(
        label="b",
        prompt_md='''
## Part b. The geometric summary

Now the summary that matches the shape of the data.

The geometric mean is the arithmetic mean computed on the logs and then
exponentiated back. The SD of the logs is a plain sample SD, computed on
`np.log(auc)` rather than on `auc`. The geometric CV as a percent is
`100 * sqrt(exp(sd_log**2) - 1)`.

Two sanity checks once you have them. The geometric mean must come out at or
below the arithmetic mean from part a, always, for any data with any spread in
it. And it should land close to the median, because for lognormal data those
two coincide.

Note that the geometric CV cannot be obtained from `arith_mean` and
`arith_sd`. It needs the SD of the logs, which is why this part computes
`sd_log` explicitly rather than reusing anything from part a.

Fill in `geo_mean`, `sd_log`, `geo_cv`, all plain floats.

Moves: `np.log`, `np.exp`, `np.mean`, `np.std(..., ddof=1)`, `np.sqrt`.
''',
        starter='''
# Part b. The geometric summary.

geo_mean = None   # YOUR CODE: exp of the mean of the logs
sd_log = None     # YOUR CODE: sample SD of the logs
geo_cv = None     # YOUR CODE: 100 * sqrt(exp(sd_log ** 2) - 1)

print("geometric mean : %s" % geo_mean)
print("SD of logs     : %s" % sd_log)
print("geometric CV   : %s" % geo_cv)
if geo_mean is not None and arith_mean is not None:
    print("")
    print("geometric %.3f <= arithmetic %.3f ?" % (geo_mean, arith_mean),
          geo_mean <= arith_mean)
''',
        solution='''
# Part b. The geometric summary.

_log_auc = np.log(auc)

geo_mean = float(np.exp(np.mean(_log_auc)))
sd_log = float(np.std(_log_auc, ddof=1))
geo_cv = 100.0 * float(np.sqrt(np.exp(sd_log ** 2) - 1.0))

print("geometric mean : %s" % geo_mean)
print("SD of logs     : %s" % sd_log)
print("geometric CV   : %s" % geo_cv)
if geo_mean is not None and arith_mean is not None:
    print("")
    print("geometric %.3f <= arithmetic %.3f ?" % (geo_mean, arith_mean),
          geo_mean <= arith_mean)
''',
    ),
    dict(
        label="c",
        prompt_md='''
## Part c. Does the data actually back this up

You have asserted that AUC is lognormal because the biology says so. Check it
against the data rather than leaving it as an article of faith.

Run `stats.shapiro` twice: once on the raw AUC values and once on their logs.
Keep the p value from each. Remember the direction: a **small** p is evidence
**against** normality, so you expect the raw values to produce a small p and
the logs a large one.

Then make the decision explicit. Set `use_log` to `True` if the logs look more
normal than the raw values do, judged by the p values being on opposite sides
of 0.05, and `False` otherwise. Writing the rule down as code, before you see
whether you like the answer, is the discipline this part is really teaching.

One caveat worth carrying forward. Twenty-four values is not many. Had this
test come back ambiguous, the right response would be to fall back on what is
known about exposure parameters rather than to let an underpowered test
overrule the pharmacology.

Fill in `p_raw`, `p_log`, both plain floats, and `use_log`, a single True or
False.

Moves: `stats.shapiro(x)`, `.pvalue`, `np.log`, a comparison.
''',
        starter='''
# Part c. Test normality on the raw values and on the logs.

p_raw = None    # YOUR CODE: shapiro p value for the raw AUC
p_log = None    # YOUR CODE: shapiro p value for the logged AUC
use_log = None  # YOUR CODE: True if the raw fails and the logs pass at 0.05

print("shapiro p, raw  : %s" % p_raw)
print("shapiro p, logs : %s" % p_log)
print("work on the log scale?", use_log)
''',
        solution='''
# Part c. Test normality on the raw values and on the logs.

p_raw = float(stats.shapiro(auc).pvalue)
p_log = float(stats.shapiro(np.log(auc)).pvalue)
use_log = bool(p_raw < 0.05 and p_log >= 0.05)

print("shapiro p, raw  : %s" % p_raw)
print("shapiro p, logs : %s" % p_log)
print("work on the log scale?", use_log)
''',
    ),
]


def build_expected(ns):
    import numpy as np
    from scipy import stats
    auc = ns["auc"]
    log_auc = np.log(auc)
    sd_log = float(np.std(log_auc, ddof=1))
    arith_mean = float(np.mean(auc))
    arith_sd = float(np.std(auc, ddof=1))
    return {
        "n": int(auc.size),
        "arith_mean": arith_mean,
        "arith_sd": arith_sd,
        "arith_sd_pop": float(np.std(auc)),
        "arith_cv": 100.0 * arith_sd / arith_mean,
        "arith_cv_frac": arith_sd / arith_mean,
        "median": float(np.median(auc)),
        "geo_mean": float(np.exp(np.mean(log_auc))),
        "geo_mean_log10": float(10.0 ** np.mean(np.log10(auc))),
        "mean_of_logs": float(np.mean(log_auc)),
        "sd_log": sd_log,
        "sd_log_pop": float(np.std(log_auc)),
        "geo_cv": 100.0 * float(np.sqrt(np.exp(sd_log ** 2) - 1.0)),
        "geo_cv_naive": 100.0 * sd_log,
        "p_raw": float(stats.shapiro(auc).pvalue),
        "p_log": float(stats.shapiro(log_auc).pvalue),
        "use_log": bool(stats.shapiro(auc).pvalue < 0.05 <= stats.shapiro(log_auc).pvalue),
    }


CHECK_CODE = '''
def _part_a():
    arith_mean = _need("arith_mean")
    arith_sd = _need("arith_sd")
    arith_cv = _need("arith_cv")

    got = _as_scalar(arith_mean)
    if got is None:
        return "`arith_mean` should be a single number."
    if _num_close(got, _EXP["geo_mean"], rtol=1e-6):
        return "`arith_mean` is the geometric mean. Part a wants the plain arithmetic one."
    if _num_close(got, _EXP["median"], rtol=1e-9):
        return "`arith_mean` is the median, not the mean."
    if not _num_close(got, _EXP["arith_mean"], rtol=1e-5):
        return "`arith_mean` is not np.mean of auc."

    got = _as_scalar(arith_sd)
    if got is None:
        return "`arith_sd` should be a single number."
    if _num_close(got, _EXP["arith_sd_pop"], rtol=1e-6):
        return ("`arith_sd` used n in the denominator. numpy defaults to ddof=0, so pass "
                "ddof=1 for the sample SD.")
    if _num_close(got, _EXP["sd_log"], rtol=1e-6):
        return "`arith_sd` is the SD of the logs. Part a works on the raw values."
    if not _num_close(got, _EXP["arith_sd"], rtol=1e-5):
        return "`arith_sd` is not the sample SD of auc."

    got = _as_scalar(arith_cv)
    if got is None:
        return "`arith_cv` should be a single number."
    if _num_close(got, _EXP["arith_cv_frac"], rtol=1e-6):
        return "`arith_cv` is a fraction. The task asks for a percent, so multiply by 100."
    if _num_close(got, _EXP["geo_cv"], rtol=1e-6):
        return "`arith_cv` is the geometric CV. Part a wants 100 * SD / mean on raw values."
    if not _num_close(got, _EXP["arith_cv"], rtol=1e-5):
        return "`arith_cv` is not 100 * arith_sd / arith_mean."
    return None


def _part_b():
    geo_mean = _need("geo_mean")
    sd_log = _need("sd_log")
    geo_cv = _need("geo_cv")

    got = _as_scalar(geo_mean)
    if got is None:
        return "`geo_mean` should be a single number."
    if _num_close(got, _EXP["mean_of_logs"], rtol=1e-6):
        return ("`geo_mean` is the mean of the logs, still on the log scale. Exponentiate "
                "it with np.exp to bring it back to mg*h/L.")
    if _num_close(got, _EXP["arith_mean"], rtol=1e-6):
        return ("`geo_mean` equals the arithmetic mean, so no log transform happened. It is "
                "exp of the mean of np.log(auc).")
    if not _num_close(got, _EXP["geo_mean"], rtol=1e-5):
        if _num_close(got, _EXP["geo_mean_log10"], rtol=1e-4):
            return ("`geo_mean` used log base 10 somewhere without matching it. np.log and "
                    "np.exp are a matched pair; np.log10 needs 10 ** x to undo it.")
        return "`geo_mean` is not np.exp(np.mean(np.log(auc)))."
    if got > _EXP["arith_mean"] * (1 + 1e-9):
        return ("`geo_mean` came out above the arithmetic mean, which is impossible for any "
                "data with spread in it. Something is wrong with the transform.")

    got = _as_scalar(sd_log)
    if got is None:
        return "`sd_log` should be a single number."
    if _num_close(got, _EXP["sd_log_pop"], rtol=1e-6):
        return "`sd_log` used ddof=0. Pass ddof=1 for the sample SD."
    if _num_close(got, _EXP["arith_sd"], rtol=1e-6):
        return "`sd_log` is the SD of the raw values. Take np.log first."
    if not _num_close(got, _EXP["sd_log"], rtol=1e-5):
        return "`sd_log` is not the sample SD of np.log(auc)."

    got = _as_scalar(geo_cv)
    if got is None:
        return "`geo_cv` should be a single number."
    if _num_close(got, _EXP["geo_cv_naive"], rtol=1e-6):
        return ("`geo_cv` is just 100 times the SD of the logs. The formula needs the "
                "exponential: 100 * sqrt(exp(sd_log ** 2) - 1).")
    if _num_close(got, _EXP["arith_cv"], rtol=1e-6):
        return ("`geo_cv` is the ordinary CV of the raw values. The geometric CV is derived "
                "from the SD of the logs instead.")
    if not _num_close(got, _EXP["geo_cv"], rtol=1e-5):
        return "`geo_cv` is not 100 * sqrt(exp(sd_log ** 2) - 1)."
    return None


def _part_c():
    p_raw = _need("p_raw")
    p_log = _need("p_log")
    use_log = _need("use_log")

    got = _as_scalar(p_raw)
    if got is None:
        return ("`p_raw` should be a single number. stats.shapiro returns a result object, "
                "so take .pvalue from it.")
    if _num_close(got, _EXP["p_log"], rtol=1e-6) and not _num_close(got, _EXP["p_raw"], rtol=1e-6):
        return "`p_raw` was computed on the logs. It should test the raw AUC values."
    if not _num_close(got, _EXP["p_raw"], rtol=1e-4):
        return "`p_raw` is not the shapiro p value for the raw AUC values."

    got = _as_scalar(p_log)
    if got is None:
        return "`p_log` should be a single number."
    if _num_close(got, _EXP["p_raw"], rtol=1e-6) and not _num_close(got, _EXP["p_log"], rtol=1e-6):
        return "`p_log` was computed on the raw values. Take np.log first."
    if not _num_close(got, _EXP["p_log"], rtol=1e-4):
        return "`p_log` is not the shapiro p value for np.log(auc)."

    if not isinstance(use_log, (bool, _np.bool_)):
        val = _as_scalar(use_log)
        if val is None:
            return "`use_log` should be a single True or False."
        use_log = bool(val)
    if bool(use_log) != _EXP["use_log"]:
        if _EXP["use_log"]:
            return ("`use_log` should be True here. The raw p is below 0.05, which is "
                    "evidence against normality, and the logged p is above it. Check you "
                    "have the direction the right way round: small p means NOT normal.")
        return "`use_log` does not follow from the two p values and the 0.05 rule."
    return None
'''

# Functions this problem introduces. Keys must exist in tools/function_catalog.py.
FUNCTIONS = [
    "geometric mean",
    "geometric CV",
    "stats.shapiro",
    "np.exp",
]
