PID = "W3D1"
WEEK = 3
DAY = 1
SLUG = "w3d1_scipy_group_comparison"
TITLE = "Comparing two groups when the data is skewed"
TOPIC = "scipy.stats: t-test, Welch, Mann-Whitney, and choosing between them"
EST = "20-30 min"

SCENARIO_MD = '''
## Scenario

A tumour growth inhibition study has read out. Twelve mice on vehicle, eight
on a candidate compound, and the endpoint is final tumour volume in cubic
millimetres on day 21. The groups are unequal because four treated animals
were withdrawn early for reasons unrelated to the compound, which is ordinary.

Tumour volumes are the textbook example of right-skewed data. They cannot go
below zero, most animals cluster at the low end, and a couple of fast growers
sit far out to the right. The mean sits well above the median, and the spread
grows with the size of the tumour rather than staying constant.

That shape matters, because the default test everybody reaches for assumes
roughly the opposite: symmetric data with similar spread in both groups. Your
job is to run the comparison three defensible ways and understand why they
disagree, rather than running one and reporting whichever number looks best.
'''

CONCEPT_MD = '''
## Three tests, and what each one actually assumes

**Student's t-test** compares two means. It assumes the values are roughly
normal and, in its classic form, that both groups have the **same variance**.
That second assumption is the one that quietly fails in drug studies, because
a treatment that shrinks tumours usually shrinks their spread too. You end up
comparing a wide group with a narrow one, and the pooled variance it computes
describes neither.

The damage is worst when the group **sizes** are unequal as well, which is the
situation here. Pooling lets the larger group dominate the shared variance
estimate, so if the larger group is also the wider one, the test becomes
conservative and hides real effects. You will see that happen in this dataset:
the two tests land on opposite sides of 0.05, and the one that says there is
nothing here is the one whose assumptions are violated.

**Welch's t-test** is the same test without the equal variance assumption. It
estimates each group's variance separately and adjusts the degrees of freedom,
which come out fractional, which is why a Welch result often shows df like
13.7. It costs almost nothing when variances happen to be equal, and it saves
you when they are not. This is why Welch is the better default, and why
`scipy.stats.ttest_ind` takes `equal_var=False`. Note that the **default is
`True`**, so the classic version is what you get unless you ask otherwise.

**Mann-Whitney U** does not compare means at all. It ranks every value across
both groups and asks whether one group's ranks run higher. Because it only
uses ordering, no distributional shape can upset it, and one enormous tumour
moves a rank by one position instead of dragging a mean. The trade is that it
answers a slightly different question, about stochastic ordering rather than
about means, and the effect size that comes with it is less familiar to
clinicians.

**The log transform option.** Right-skewed, strictly positive data is often
close to normal once you take logs. A t-test on the logs is comparing
**geometric** means, which for multiplicative biology is usually the more
meaningful comparison anyway. This is the standard approach for AUC and Cmax
in bioequivalence work, for exactly this reason.

**The thing that matters more than any of this.** Decide which test you will
run **before** you see the result, and report that one. Running all three and
picking the smallest p is how a null result becomes a publication, and it is
not a judgment call, it is a fabrication. You are running all three here to
understand them, which is a different activity from analysing a study.
'''

TASK_MD = '''
## Your task

Three parts, worked one at a time.

| Part | What you work out | Variables to fill in |
|---|---|---|
| a | the descriptive picture, including how skewed it is | `desc` |
| b | Student's t and Welch's t, and why they differ | `t_student`, `p_student`, `t_welch`, `p_welch` |
| c | Mann-Whitney, and a t-test on the logs | `u_stat`, `p_mw`, `t_log`, `p_log` |

Below this you will find the **Setup** cell, then a **Toolkit** cell that
demonstrates every move on a tiny made up example, then **Part a**, **Part b**
and **Part c**, each with its own explanation and a cell to write in.
'''

DATA_CODE = '''
import numpy as np
import pandas as pd
from scipy import stats

PROBLEM_ID = "W3D1"
TOPIC = "scipy.stats: t-test, Welch, Mann-Whitney, and choosing between them"

rng = np.random.default_rng(3194)

N_VEHICLE, N_TREATED = 12, 8
# Tumour volumes are lognormal. The treated group is smaller AND far tighter,
# and the groups are unequal in size because four treated animals were
# withdrawn early for reasons unrelated to the compound.
_vehicle = rng.lognormal(mean=np.log(820.0), sigma=0.70, size=N_VEHICLE)
_treated = rng.lognormal(mean=np.log(430.0), sigma=0.30, size=N_TREATED)

tumour = pd.DataFrame({
    "MOUSE": ["M%02d" % i for i in range(1, N_VEHICLE + N_TREATED + 1)],
    "GROUP": ["Vehicle"] * N_VEHICLE + ["Treated"] * N_TREATED,
    "VOLUME_MM3": np.round(np.concatenate([_vehicle, _treated]), 1),
})

vehicle = tumour.loc[tumour["GROUP"] == "Vehicle", "VOLUME_MM3"].to_numpy()
treated = tumour.loc[tumour["GROUP"] == "Treated", "VOLUME_MM3"].to_numpy()

print("tumour:", tumour.shape)
print(tumour.to_string(index=False))
print("")
print("vehicle:", np.round(np.sort(vehicle), 1))
print("treated:", np.round(np.sort(treated), 1))
'''

TOOLKIT_MD = '''
## Toolkit for this problem

| What you want to do | How |
|---|---|
| import the stats tools | `from scipy import stats` |
| Student's t-test | `stats.ttest_ind(a, b)`, which assumes equal variance |
| Welch's t-test | `stats.ttest_ind(a, b, equal_var=False)` |
| Mann-Whitney U | `stats.mannwhitneyu(a, b, alternative="two-sided")` |
| read a test result | `res = stats.ttest_ind(...)` then `res.statistic`, `res.pvalue` |
| median | `np.median(x)` |
| interquartile range | `stats.iqr(x)`, or `np.percentile(x, 75) - np.percentile(x, 25)` |
| natural log of an array | `np.log(x)` |

Every test in `scipy.stats` returns a result object, not a plain pair. Reach
for `.statistic` and `.pvalue` by name rather than unpacking by position,
which is clearer and survives scipy adding fields.
'''

TOOLKIT_CODE = '''
import numpy as np
from scipy import stats

# Two tiny made up samples. Nothing here is the answer.
demo_a = np.array([12.0, 15.0, 11.0, 19.0, 14.0, 60.0])   # one big outlier
demo_b = np.array([9.0, 10.0, 8.0, 12.0, 11.0, 10.0])

print("demo_a:", demo_a, "  mean", round(demo_a.mean(), 2), " median", np.median(demo_a))
print("demo_b:", demo_b, "  mean", round(demo_b.mean(), 2), " median", np.median(demo_b))
print("")
print("the mean of demo_a sits far above its median, which is what skew looks like")
print("SDs are very different too:", round(demo_a.std(ddof=1), 2), "against",
      round(demo_b.std(ddof=1), 2))

print("")
print("--- 1. a test result is an object, read it by name ---")
demo_res = stats.ttest_ind(demo_a, demo_b)
print("type           :", type(demo_res).__name__)
print("res.statistic  :", round(demo_res.statistic, 4))
print("res.pvalue     :", round(demo_res.pvalue, 4))

print("")
print("--- 2. Student versus Welch ---")
demo_student = stats.ttest_ind(demo_a, demo_b)
demo_welch = stats.ttest_ind(demo_a, demo_b, equal_var=False)
print("Student t = %.4f  p = %.4f   <- assumes equal variance, the DEFAULT"
      % (demo_student.statistic, demo_student.pvalue))
print("Welch   t = %.4f  p = %.4f   <- does not"
      % (demo_welch.statistic, demo_welch.pvalue))
print("the two disagree here precisely because the spreads are so different")

print("")
print("--- 3. Mann-Whitney works on ranks ---")
demo_mw = stats.mannwhitneyu(demo_a, demo_b, alternative="two-sided")
print("U = %.1f  p = %.4f" % (demo_mw.statistic, demo_mw.pvalue))
print("")
print("watch what one extreme value does to each approach:")
demo_worse = demo_a.copy()
demo_worse[-1] = 600.0
print("after moving 60 to 600:")
print("  Welch p : %.4f -> %.4f" % (demo_welch.pvalue,
      stats.ttest_ind(demo_worse, demo_b, equal_var=False).pvalue))
print("  Mann-Whitney p : %.4f -> %.4f" % (demo_mw.pvalue,
      stats.mannwhitneyu(demo_worse, demo_b, alternative="two-sided").pvalue))
print("the rank based p does not move at all, because the ordering did not change")

print("")
print("--- 4. a t-test on the logs compares geometric means ---")
demo_log = stats.ttest_ind(np.log(demo_a), np.log(demo_b), equal_var=False)
print("t on logs = %.4f  p = %.4f" % (demo_log.statistic, demo_log.pvalue))
print("geometric mean of demo_a:", round(float(np.exp(np.mean(np.log(demo_a)))), 2))
print("arithmetic mean of demo_a:", round(float(demo_a.mean()), 2))
'''

PARTS = [
    dict(
        label="a",
        prompt_md='''
## Part a. Describe the data before you test it

Never run a test on a dataset you have not looked at. The descriptive table
is what tells you which test is defensible, and you build it first so the
choice is not influenced by the p value.

For each group give the count, the arithmetic mean, the sample SD, the median,
and the interquartile range. Round everything to 2 decimals so the table is
readable.

Look at what comes out. The mean sitting noticeably above the median is the
signature of right skew. The two SDs being far apart is the signature that
Student's equal variance assumption does not hold. Both are visible before any
test has been run.

Fill in `desc`, a DataFrame with one row per group, sorted with `Vehicle`
first then `Treated`, index renumbered, with exactly:

| column | meaning |
|---|---|
| `GROUP` | `Vehicle` or `Treated` |
| `N` | how many mice |
| `MEAN` | arithmetic mean volume |
| `SD` | sample SD |
| `MEDIAN` | median volume |
| `IQR` | interquartile range |

Moves: `np.mean`, `np.std(..., ddof=1)`, `np.median`, `stats.iqr`,
`pd.DataFrame([...])`, `.round(2)`.
''',
        starter='''
# Part a. Descriptive statistics for each group.

desc = None   # YOUR CODE: one row per group, Vehicle first, columns
              #            GROUP, N, MEAN, SD, MEDIAN, IQR, rounded to 2 dp

print(desc)
''',
        solution='''
# Part a. Descriptive statistics for each group.

desc = pd.DataFrame([
    {"GROUP": "Vehicle", "N": vehicle.size,
     "MEAN": np.mean(vehicle), "SD": np.std(vehicle, ddof=1),
     "MEDIAN": np.median(vehicle), "IQR": stats.iqr(vehicle)},
    {"GROUP": "Treated", "N": treated.size,
     "MEAN": np.mean(treated), "SD": np.std(treated, ddof=1),
     "MEDIAN": np.median(treated), "IQR": stats.iqr(treated)},
])
desc[["MEAN", "SD", "MEDIAN", "IQR"]] = desc[["MEAN", "SD", "MEDIAN", "IQR"]].round(2)

print(desc)
''',
    ),
    dict(
        label="b",
        prompt_md='''
## Part b. Student's t and Welch's t

Run both forms of the two sample t-test on the raw volumes, and keep the
statistic and the p value from each.

`stats.ttest_ind(a, b)` gives you Student's version, which pools the two
variances into one estimate. Adding `equal_var=False` gives Welch's version,
which keeps them separate. Note which way round the default goes: you get
Student unless you ask for Welch, which is the opposite of what most people
should want.

Compare the two p values against the SDs you computed in part a. When the
spreads are similar the two tests agree closely. When they are not, they
diverge, and Welch is the one that is still telling the truth.

Pass `vehicle` first and `treated` second, so the sign of the statistic is
positive when vehicle tumours are larger.

Fill in `t_student`, `p_student`, `t_welch`, `p_welch`, all plain floats.

Moves: `stats.ttest_ind(a, b)`, `stats.ttest_ind(a, b, equal_var=False)`,
`res.statistic`, `res.pvalue`, `float(...)`.
''',
        starter='''
# Part b. Student's t-test and Welch's t-test on the raw volumes.

t_student = None   # YOUR CODE: statistic, equal variance assumed
p_student = None   # YOUR CODE: its p value
t_welch = None     # YOUR CODE: statistic, equal variance NOT assumed
p_welch = None     # YOUR CODE: its p value

print("Student t = %s  p = %s" % (t_student, p_student))
print("Welch   t = %s  p = %s" % (t_welch, p_welch))
''',
        solution='''
# Part b. Student's t-test and Welch's t-test on the raw volumes.

_student = stats.ttest_ind(vehicle, treated)
_welch = stats.ttest_ind(vehicle, treated, equal_var=False)

t_student = float(_student.statistic)
p_student = float(_student.pvalue)
t_welch = float(_welch.statistic)
p_welch = float(_welch.pvalue)

print("Student t = %s  p = %s" % (t_student, p_student))
print("Welch   t = %s  p = %s" % (t_welch, p_welch))
''',
    ),
    dict(
        label="c",
        prompt_md='''
## Part c. Mann-Whitney, and a t-test on the logs

Two ways of handling the skew rather than ignoring it.

`stats.mannwhitneyu` ranks all twenty values together and asks whether the
vehicle ranks run higher. Pass `alternative="two-sided"` explicitly: it is the
sensible default for a comparison with no prespecified direction, and writing
it down makes the choice visible to whoever reads your code.

Then take natural logs of both groups and run Welch's t-test on those. Because
the data is lognormal, the logs are close to normal, and comparing their means
is comparing the **geometric** means of the original volumes. This is exactly
what bioequivalence analysis does to AUC and Cmax, for the same reason.

Vehicle first, treated second, as in part b.

When you have all three, look at them together. They will not be identical,
and the useful question is not which is smallest but whether they agree on the
conclusion. If they do, the finding is robust to how you analysed it. If they
disagree sharply, that disagreement is itself the result, and the honest move
is to say so rather than to pick.

Fill in `u_stat`, `p_mw`, `t_log`, `p_log`, all plain floats.

Moves: `stats.mannwhitneyu(a, b, alternative="two-sided")`, `np.log`,
`stats.ttest_ind(..., equal_var=False)`, `.statistic`, `.pvalue`.
''',
        starter='''
# Part c. Mann-Whitney on the raw data, and Welch's t-test on the logs.

u_stat = None   # YOUR CODE: Mann-Whitney U statistic
p_mw = None     # YOUR CODE: its p value, two sided
t_log = None    # YOUR CODE: Welch t statistic on log transformed volumes
p_log = None    # YOUR CODE: its p value

print("Mann-Whitney U = %s  p = %s" % (u_stat, p_mw))
print("Welch on logs  t = %s  p = %s" % (t_log, p_log))
''',
        solution='''
# Part c. Mann-Whitney on the raw data, and Welch's t-test on the logs.

_mw = stats.mannwhitneyu(vehicle, treated, alternative="two-sided")
_logt = stats.ttest_ind(np.log(vehicle), np.log(treated), equal_var=False)

u_stat = float(_mw.statistic)
p_mw = float(_mw.pvalue)
t_log = float(_logt.statistic)
p_log = float(_logt.pvalue)

print("Mann-Whitney U = %s  p = %s" % (u_stat, p_mw))
print("Welch on logs  t = %s  p = %s" % (t_log, p_log))
''',
    ),
]


def build_expected(ns):
    import numpy as np
    from scipy import stats
    v, t = ns["vehicle"], ns["treated"]
    student = stats.ttest_ind(v, t)
    welch = stats.ttest_ind(v, t, equal_var=False)
    mw_two = stats.mannwhitneyu(v, t, alternative="two-sided")
    mw_less = stats.mannwhitneyu(v, t, alternative="greater")
    mw_rev = stats.mannwhitneyu(t, v, alternative="two-sided")
    logt = stats.ttest_ind(np.log(v), np.log(t), equal_var=False)
    logt_student = stats.ttest_ind(np.log(v), np.log(t))
    return {
        "desc_cols": ["GROUP", "N", "MEAN", "SD", "MEDIAN", "IQR"],
        "desc_group": ["Vehicle", "Treated"],
        "desc_n": [int(v.size), int(t.size)],
        "desc_mean": [round(float(np.mean(v)), 2), round(float(np.mean(t)), 2)],
        "desc_sd": [round(float(np.std(v, ddof=1)), 2), round(float(np.std(t, ddof=1)), 2)],
        "desc_sd_pop": [round(float(np.std(v)), 2), round(float(np.std(t)), 2)],
        "desc_median": [round(float(np.median(v)), 2), round(float(np.median(t)), 2)],
        "desc_iqr": [round(float(stats.iqr(v)), 2), round(float(stats.iqr(t)), 2)],
        "t_student": float(student.statistic), "p_student": float(student.pvalue),
        "t_welch": float(welch.statistic), "p_welch": float(welch.pvalue),
        "u_stat": float(mw_two.statistic), "p_mw": float(mw_two.pvalue),
        "u_rev": float(mw_rev.statistic), "p_mw_one": float(mw_less.pvalue),
        "t_log": float(logt.statistic), "p_log": float(logt.pvalue),
        "p_log_student": float(logt_student.pvalue),
    }


CHECK_CODE = '''
def _part_a():
    desc = _need("desc")
    if not isinstance(desc, _pd.DataFrame):
        return "`desc` should be a pandas DataFrame."
    cols = _cols_of(desc) or []
    missing = [c for c in _EXP["desc_cols"] if c not in cols]
    if missing:
        return "`desc` is missing the column(s) %s." % ", ".join(missing)
    if len(desc) != 2:
        return "`desc` should have exactly 2 rows, one per group, but it has %d." % len(desc)
    if [str(v) for v in desc["GROUP"]] != _EXP["desc_group"]:
        return ("`desc` should list Vehicle first and Treated second, which is the order "
                "the task asks for.")
    if not _arr_close(desc["N"].to_numpy(), _EXP["desc_n"]):
        return "`N` is not the number of mice in each group."
    if not _arr_close(desc["MEAN"].to_numpy(), _EXP["desc_mean"], rtol=1e-3, atol=0.02):
        return "`MEAN` is not the arithmetic mean volume of each group, rounded to 2 dp."
    if _arr_close(desc["SD"].to_numpy(), _EXP["desc_sd_pop"], rtol=1e-3, atol=0.02):
        return ("`SD` used n in the denominator. numpy defaults to ddof=0, so pass ddof=1 "
                "for the sample SD.")
    if not _arr_close(desc["SD"].to_numpy(), _EXP["desc_sd"], rtol=1e-3, atol=0.02):
        return "`SD` is not the sample SD of each group, rounded to 2 dp."
    if not _arr_close(desc["MEDIAN"].to_numpy(), _EXP["desc_median"], rtol=1e-3, atol=0.02):
        return "`MEDIAN` is not the median volume of each group."
    if not _arr_close(desc["IQR"].to_numpy(), _EXP["desc_iqr"], rtol=1e-3, atol=0.02):
        return ("`IQR` is not the interquartile range. It is the 75th percentile minus the "
                "25th, which stats.iqr gives directly.")
    return None


def _part_b():
    t_student = _need("t_student")
    p_student = _need("p_student")
    t_welch = _need("t_welch")
    p_welch = _need("p_welch")

    got = _as_scalar(t_student)
    if got is None:
        return ("`t_student` should be a single number. A scipy result is an object, so "
                "take res.statistic from it.")
    if _num_close(got, -_EXP["t_student"], rtol=1e-4):
        return ("`t_student` has the sign flipped, so the arguments went in the other way "
                "round. Pass vehicle first and treated second.")
    if _num_close(got, _EXP["t_welch"], rtol=1e-9) and not _num_close(got, _EXP["t_student"], rtol=1e-9):
        return ("`t_student` is the Welch statistic. ttest_ind without equal_var gives "
                "Student's version, which is the default.")
    if not _num_close(got, _EXP["t_student"], rtol=1e-4):
        return "`t_student` is not the Student t statistic for vehicle against treated."

    got = _as_scalar(p_student)
    if got is None:
        return "`p_student` should be a single number, from res.pvalue."
    if not _num_close(got, _EXP["p_student"], rtol=1e-4):
        if _num_close(got, _EXP["p_student"] / 2.0, rtol=1e-4):
            return "`p_student` looks like a one sided p value. ttest_ind returns two sided."
        return "`p_student` is not the p value from the Student t-test."

    got = _as_scalar(t_welch)
    if got is None:
        return "`t_welch` should be a single number."
    if _num_close(got, _EXP["t_student"], rtol=1e-9) and not _num_close(got, _EXP["t_welch"], rtol=1e-9):
        return ("`t_welch` equals the Student statistic, so equal_var=False was not passed. "
                "The default is True, which is the pooled variance version.")
    if not _num_close(got, _EXP["t_welch"], rtol=1e-4):
        return "`t_welch` is not the Welch t statistic."

    got = _as_scalar(p_welch)
    if got is None:
        return "`p_welch` should be a single number."
    if _num_close(got, _EXP["p_student"], rtol=1e-9) and not _num_close(got, _EXP["p_welch"], rtol=1e-9):
        return "`p_welch` is the Student p value. Pass equal_var=False to get Welch's."
    if not _num_close(got, _EXP["p_welch"], rtol=1e-4):
        return "`p_welch` is not the p value from Welch's t-test."
    return None


def _part_c():
    u_stat = _need("u_stat")
    p_mw = _need("p_mw")
    t_log = _need("t_log")
    p_log = _need("p_log")

    got = _as_scalar(u_stat)
    if got is None:
        return "`u_stat` should be a single number, from res.statistic."
    if _num_close(got, _EXP["u_rev"], rtol=1e-6) and not _num_close(got, _EXP["u_stat"], rtol=1e-6):
        return ("`u_stat` is the U you get passing treated first. Pass vehicle first, as in "
                "part b, so the two parts are comparable.")
    if not _num_close(got, _EXP["u_stat"], rtol=1e-4):
        return "`u_stat` is not the Mann-Whitney U for vehicle against treated."

    got = _as_scalar(p_mw)
    if got is None:
        return "`p_mw` should be a single number."
    if _num_close(got, _EXP["p_mw_one"], rtol=1e-4) and not _num_close(got, _EXP["p_mw"], rtol=1e-4):
        return ("`p_mw` is a one sided p value. Pass alternative='two-sided' explicitly.")
    if not _num_close(got, _EXP["p_mw"], rtol=1e-4):
        return "`p_mw` is not the two sided Mann-Whitney p value."

    got = _as_scalar(t_log)
    if got is None:
        return "`t_log` should be a single number."
    if _num_close(got, _EXP["t_welch"], rtol=1e-6):
        return ("`t_log` is the test on the raw volumes. Take np.log of both groups before "
                "the t-test.")
    if not _num_close(got, _EXP["t_log"], rtol=1e-4):
        if _num_close(got, -_EXP["t_log"], rtol=1e-4):
            return "`t_log` has the sign flipped. Pass the logged vehicle values first."
        return ("`t_log` is not the Welch statistic on the logged volumes. Check you used "
                "np.log, which is natural log, and equal_var=False.")

    got = _as_scalar(p_log)
    if got is None:
        return "`p_log` should be a single number."
    if _num_close(got, _EXP["p_log_student"], rtol=1e-9) and not _num_close(got, _EXP["p_log"], rtol=1e-9):
        return "`p_log` came from Student's version. Pass equal_var=False for Welch's."
    if not _num_close(got, _EXP["p_log"], rtol=1e-4):
        return "`p_log` is not the p value of Welch's t-test on the logged volumes."
    return None
'''

# Functions this problem introduces. Keys must exist in tools/function_catalog.py.
FUNCTIONS = [
    "from scipy import stats",
    "stats.ttest_ind",
    "stats.mannwhitneyu",
    "stats.iqr",
    "result objects",
]
