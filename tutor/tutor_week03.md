# Week 3, scipy

The full text of every problem this week, the data it generates, the
toolkit the student is given, what each part asks for, and what every
possible check failure means. No solutions in this file.

---

# W3D1. Comparing two groups when the data is skewed

- topic: scipy.stats: t-test, Welch, Mann-Whitney, and choosing between them
- notebook: problems/week03/w3d1_scipy_group_comparison.ipynb
- solution: solutions/week03/w3d1_scipy_group_comparison_solution.ipynb
- expected time: 20-30 min

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

## The setup cell the student runs

This is given to them, already working. The data is synthetic and
seeded, so their numbers are always these numbers.

```python
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
```

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

The toolkit cell demonstrates each of those on a tiny made up example.
It is in the notebook and none of it is the answer.

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

**Variables this part is graded on:** `desc`

The starter cell they see:

```python
# Part a. Descriptive statistics for each group.

desc = None   # YOUR CODE: one row per group, Vehicle first, columns
              #            GROUP, N, MEAN, SD, MEDIAN, IQR, rounded to 2 dp

print(desc)
```

**Every failure message part a can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `desc` should be a pandas DataFrame.
- `desc` is missing the column(s) %s. ,
- `desc` should have exactly 2 rows, one per group, but it has %d.
- `desc` should list Vehicle first and Treated second, which is the order the task asks for.
- `N` is not the number of mice in each group.
- `MEAN` is not the arithmetic mean volume of each group, rounded to 2 dp.
- `SD` used n in the denominator. numpy defaults to ddof=0, so pass ddof=1 for the sample SD.
- `SD` is not the sample SD of each group, rounded to 2 dp.
- `MEDIAN` is not the median volume of each group.
- `IQR` is not the interquartile range. It is the 75th percentile minus the 25th, which stats.iqr gives directly.

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

**Variables this part is graded on:** `t_student`, `p_student`, `t_welch`, `p_welch`

The starter cell they see:

```python
# Part b. Student's t-test and Welch's t-test on the raw volumes.

t_student = None   # YOUR CODE: statistic, equal variance assumed
p_student = None   # YOUR CODE: its p value
t_welch = None     # YOUR CODE: statistic, equal variance NOT assumed
p_welch = None     # YOUR CODE: its p value

print("Student t = %s  p = %s" % (t_student, p_student))
print("Welch   t = %s  p = %s" % (t_welch, p_welch))
```

**Every failure message part b can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `t_student` should be a single number. A scipy result is an object, so take res.statistic from it.
- `t_student` has the sign flipped, so the arguments went in the other way round. Pass vehicle first and treated second.
- `t_student` is the Welch statistic. ttest_ind without equal_var gives Student's version, which is the default.
- `t_student` is not the Student t statistic for vehicle against treated.
- `p_student` should be a single number, from res.pvalue.
- `p_student` is not the p value from the Student t-test.
- `t_welch` should be a single number.
- `t_welch` equals the Student statistic, so equal_var=False was not passed. The default is True, which is the pooled variance version.
- `t_welch` is not the Welch t statistic.
- `p_welch` should be a single number.
- `p_welch` is the Student p value. Pass equal_var=False to get Welch's.
- `p_welch` is not the p value from Welch's t-test.
- `p_student` looks like a one sided p value. ttest_ind returns two sided.

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

**Variables this part is graded on:** `u_stat`, `p_mw`, `t_log`, `p_log`

The starter cell they see:

```python
# Part c. Mann-Whitney on the raw data, and Welch's t-test on the logs.

u_stat = None   # YOUR CODE: Mann-Whitney U statistic
p_mw = None     # YOUR CODE: its p value, two sided
t_log = None    # YOUR CODE: Welch t statistic on log transformed volumes
p_log = None    # YOUR CODE: its p value

print("Mann-Whitney U = %s  p = %s" % (u_stat, p_mw))
print("Welch on logs  t = %s  p = %s" % (t_log, p_log))
```

**Every failure message part c can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `u_stat` should be a single number, from res.statistic.
- `u_stat` is the U you get passing treated first. Pass vehicle first, as in part b, so the two parts are comparable.
- `u_stat` is not the Mann-Whitney U for vehicle against treated.
- `p_mw` should be a single number.
- `p_mw` is a one sided p value. Pass alternative='two-sided' explicitly.
- `p_mw` is not the two sided Mann-Whitney p value.
- `t_log` should be a single number.
- `t_log` is the test on the raw volumes. Take np.log of both groups before the t-test.
- `t_log` is not the Welch statistic on the logged volumes. Check you used np.log, which is natural log, and equal_var=False.
- `p_log` should be a single number.
- `p_log` came from Student's version. Pass equal_var=False for Welch's.
- `p_log` is not the p value of Welch's t-test on the logged volumes.
- `t_log` has the sign flipped. Pass the logged vehicle values first.

**Functions this problem introduces:** from scipy import stats, stats.ttest_ind, stats.mannwhitneyu, stats.iqr, result objects

---

# W3D2. AUC is lognormal, so report the geometric mean

- topic: scipy.stats: lognormal data, geometric mean and CV, normality testing
- notebook: problems/week03/w3d2_scipy_geometric_stats.ipynb
- solution: solutions/week03/w3d2_scipy_geometric_stats_solution.ipynb
- expected time: 20-30 min

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

## The setup cell the student runs

This is given to them, already working. The data is synthetic and
seeded, so their numbers are always these numbers.

```python
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
```

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

The toolkit cell demonstrates each of those on a tiny made up example.
It is in the notebook and none of it is the answer.

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

**Variables this part is graded on:** `arith_mean`, `arith_sd`, `arith_cv`

The starter cell they see:

```python
# Part a. The arithmetic summary.

arith_mean = None   # YOUR CODE: arithmetic mean of auc
arith_sd = None     # YOUR CODE: sample SD, so ddof=1
arith_cv = None     # YOUR CODE: 100 * SD / mean

print("arithmetic mean : %s" % arith_mean)
print("arithmetic SD   : %s" % arith_sd)
print("CV percent      : %s" % arith_cv)
print("median for comparison: %.3f" % float(np.median(auc)))
```

**Every failure message part a can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `arith_mean` should be a single number.
- `arith_mean` is the geometric mean. Part a wants the plain arithmetic one.
- `arith_mean` is the median, not the mean.
- `arith_mean` is not np.mean of auc.
- `arith_sd` should be a single number.
- `arith_sd` used n in the denominator. numpy defaults to ddof=0, so pass ddof=1 for the sample SD.
- `arith_sd` is the SD of the logs. Part a works on the raw values.
- `arith_sd` is not the sample SD of auc.
- `arith_cv` should be a single number.
- `arith_cv` is a fraction. The task asks for a percent, so multiply by 100.
- `arith_cv` is the geometric CV. Part a wants 100 * SD / mean on raw values.
- `arith_cv` is not 100 * arith_sd / arith_mean.

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

**Variables this part is graded on:** `geo_mean`, `sd_log`, `geo_cv`

The starter cell they see:

```python
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
```

**Every failure message part b can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `geo_mean` should be a single number.
- `geo_mean` is the mean of the logs, still on the log scale. Exponentiate it with np.exp to bring it back to mg*h/L.
- `geo_mean` equals the arithmetic mean, so no log transform happened. It is exp of the mean of np.log(auc).
- `geo_mean` is not np.exp(np.mean(np.log(auc))).
- `geo_mean` came out above the arithmetic mean, which is impossible for any data with spread in it. Something is wrong with the transform.
- `sd_log` should be a single number.
- `sd_log` used ddof=0. Pass ddof=1 for the sample SD.
- `sd_log` is the SD of the raw values. Take np.log first.
- `sd_log` is not the sample SD of np.log(auc).
- `geo_cv` should be a single number.
- `geo_cv` is just 100 times the SD of the logs. The formula needs the exponential: 100 * sqrt(exp(sd_log ** 2) - 1).
- `geo_cv` is the ordinary CV of the raw values. The geometric CV is derived from the SD of the logs instead.
- `geo_cv` is not 100 * sqrt(exp(sd_log ** 2) - 1).
- `geo_mean` used log base 10 somewhere without matching it. np.log and np.exp are a matched pair; np.log10 needs 10 ** x to undo it.

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

**Variables this part is graded on:** `p_raw`, `p_log`, `use_log`

The starter cell they see:

```python
# Part c. Test normality on the raw values and on the logs.

p_raw = None    # YOUR CODE: shapiro p value for the raw AUC
p_log = None    # YOUR CODE: shapiro p value for the logged AUC
use_log = None  # YOUR CODE: True if the raw fails and the logs pass at 0.05

print("shapiro p, raw  : %s" % p_raw)
print("shapiro p, logs : %s" % p_log)
print("work on the log scale?", use_log)
```

**Every failure message part c can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `p_raw` should be a single number. stats.shapiro returns a result object, so take .pvalue from it.
- `p_raw` was computed on the logs. It should test the raw AUC values.
- `p_raw` is not the shapiro p value for the raw AUC values.
- `p_log` should be a single number.
- `p_log` was computed on the raw values. Take np.log first.
- `p_log` is not the shapiro p value for np.log(auc).
- `use_log` does not follow from the two p values and the 0.05 rule.
- `use_log` should be a single True or False.
- `use_log` should be True here. The raw p is below 0.05, which is evidence against normality, and the logged p is above it. Check you have the direction the right way round: small p means NOT normal.

**Functions this problem introduces:** geometric mean, geometric CV, stats.shapiro, np.exp

---

# W3D3. Fitting a decay curve, and how sure you are of the parameters

- topic: scipy.optimize: curve_fit, starting values, parameter standard errors
- notebook: problems/week03/w3d3_scipy_curve_fit.ipynb
- solution: solutions/week03/w3d3_scipy_curve_fit_solution.ipynb
- expected time: 20-30 min

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

## The setup cell the student runs

This is given to them, already working. The data is synthetic and
seeded, so their numbers are always these numbers.

```python
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
```

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

The toolkit cell demonstrates each of those on a tiny made up example.
It is in the notebook and none of it is the answer.

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

**Variables this part is graded on:** `popt`, `pcov`

The starter cell they see:

```python
# Part a. Define the model, then fit it.

def decay(t, c0, k):
    """Concentration after an IV bolus into one compartment."""
    return None   # YOUR CODE: c0 * exp(-k * t)


popt = None   # YOUR CODE: fitted parameters from curve_fit, with a sensible p0
pcov = None   # YOUR CODE: the covariance matrix curve_fit also returns

print("popt:", popt)
print("pcov:", pcov)
```

**Every failure message part a can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `decay` should be a function.
- `decay` returns None. Replace the # YOUR CODE line with the formula.
- `decay` should return one value per time, so it must work on a whole array. Use np.exp rather than math.exp.
- `decay` does not compute c0 * exp(-k * t).
- `popt` should hold the two fitted parameters. curve_fit returns a pair, so unpack it as popt, pcov = curve_fit(...).
- `popt` is not the least squares fit. Check the model formula, and pass a p0 in the right order of magnitude: C0 near the first concentration, k near ln(2) over an eyeballed half-life.
- `pcov` should be the 2 by 2 covariance matrix, which is the second thing curve_fit returns.
- `pcov` contains inf, which means the fit did not pin the parameters down. Usually a starting guess problem.
- `pcov` does not match the covariance of this fit.
- `decay` raised %s when called as decay(t, c0, k).
- `decay` is growing rather than decaying. The exponent needs the minus sign: c0 * np.exp(-k * t).
- `popt` has the parameters the other way round, so your function signature is decay(t, k, c0). The task asks for c0 first.

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

**Variables this part is graded on:** `c0_hat`, `k_hat`, `se_c0`, `se_k`

The starter cell they see:

```python
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
```

**Every failure message part b can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `c0_hat` should be a single number, taken from popt.
- `c0_hat` is the rate constant. popt[0] is C0, popt[1] is k.
- `c0_hat` is not the fitted C0.
- `k_hat` should be a single number.
- `k_hat` is not the fitted k.
- `se_c0` should be a single number.
- `se_c0` is the variance, not the standard error. Take the square root of the diagonal entry.
- `se_c0` is the standard error of k. The diagonal is in parameter order.
- `se_c0` is not sqrt of the first diagonal entry of pcov.
- `se_k` should be a single number.
- `se_k` is the variance, not the standard error. Take the square root.
- `se_k` is not sqrt of the second diagonal entry of pcov.

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

**Variables this part is graded on:** `t_half`, `pred`, `rmse`

The starter cell they see:

```python
# Part c. Half-life, predictions, and the residual summary.

t_half = None   # YOUR CODE: ln(2) / k, in hours
pred = None     # YOUR CODE: predicted concentration at each observed time
rmse = None     # YOUR CODE: root mean squared error of the residuals

print("half-life : %s h" % t_half)
print("predicted :", None if pred is None else np.round(pred, 3))
print("observed  :", obs["CONC_MG_L"].to_numpy())
print("rmse      : %s" % rmse)
```

**Every failure message part c can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `t_half` should be a single number.
- `t_half` is 1/k, which is the mean residence time style number, not the half-life. A half-life carries a factor of ln(2).
- `t_half` is not np.log(2) divided by your fitted k.
- `pred` should be a numeric array.
- `pred` has shape %s but there are %d observed timepoints, so there should be one prediction each.
- `pred` is the observed data, not the model prediction. Call your decay function with the fitted parameters.
- `pred` does not match decay(times, *popt). Make sure you evaluated at the observed times and used the fitted parameters.
- `rmse` should be a single number.
- `rmse` is the sum of squared residuals. Take the mean of the squares, then the square root.
- `rmse` is the mean absolute error. Square the residuals, average, then square root.
- `rmse` is not sqrt(mean(residuals ** 2)).

**Functions this problem introduces:** curve_fit, p0, np.diag, argument unpacking

---

# W3D4. Solving the compartment model as an ODE, and surviving a discontinuity

- topic: scipy.integrate: solve_ivp, t_eval, checking against an analytic solution
- notebook: problems/week03/w3d4_scipy_solve_ivp.ipynb
- solution: solutions/week03/w3d4_scipy_solve_ivp_solution.ipynb
- expected time: 20-30 min

## Scenario

So far every profile you have worked with was a formula you could evaluate
directly. That stops being possible very quickly: add a second compartment,
saturable elimination, a transit absorption chain or a PD effect feeding back
on clearance, and there is no closed form left to write down.

What there always is, is a differential equation. A one compartment model says
something simple about **rates**: the amount in the body falls at a rate
proportional to how much is there.

```
dA/dt = -(CL / V) * A
```

Numerical integration turns that statement into a curve. This problem works
through it on a case you can check, an IV bolus, where the exact answer is
known, so you can confirm the machinery before trusting it somewhere you
cannot check. Then you do an infusion, where the dosing rate switches off part
way through, which is where naive integration quietly goes wrong.

## solve_ivp, and why discontinuities need care

**The shape of the call.** `solve_ivp(fun, t_span, y0, ...)` needs three
things: a function giving the derivative, the interval to integrate over as a
pair, and the starting state as a **sequence**, even with one compartment, so
`[500.0]` rather than `500.0`.

The derivative function takes `(t, y)` in that order, with `y` an array of
states, and returns the derivatives in the same order. For one compartment:

```python
def rhs(t, y, cl, v):
    amount = y[0]
    return [-(cl / v) * amount]
```

Extra parameters are passed through with `args=(cl, v)`.

**`t_eval` controls output, not accuracy.** By default the solver returns the
points it happened to step through, which are chosen for efficiency and are
not evenly spaced. Passing `t_eval=np.linspace(...)` makes it report at the
times you want. It does not make the answer more accurate, it only changes
where it is sampled.

**Accuracy is controlled by `rtol` and `atol`, and the defaults are loose.**
`rtol=1e-3` means the solver is content with roughly three correct digits,
which is fine for a quick sketch and not fine for anything you will compare
against data or integrate to get an AUC. For PK work pass something like
`rtol=1e-8, atol=1e-10`. It costs a few milliseconds and removes solver error
as a thing you have to think about. You will see the difference in part b: on
the defaults the numerical and exact curves differ in the third decimal, and
with tight tolerances they agree to about eight.

The result carries `sol.t`, `sol.y` with one **row per state**, so
`sol.y[0]` is the first compartment, and `sol.success`, which is worth
checking rather than assuming.

**Amounts versus concentrations.** The state is an amount in mg, because that
is what mass balance is written in. Concentration is amount divided by volume.
Mixing the two up is the single most common error in writing PK models, and
it does not announce itself: the curve looks perfectly plausible, just scaled
wrong by a factor of V.

**The discontinuity problem.** An infusion that stops at 2 hours makes the
derivative jump. Adaptive solvers choose their step size by looking at how
smooth the solution has been, so a solver happily taking one hour steps can
step straight over the moment the infusion ended, never sampling the jump, and
return a curve that keeps rising past the end of the infusion.

Nothing errors. `sol.success` is still `True`. The answer is just wrong.

Two fixes. Cap the step size with `max_step` so the solver cannot stride over
the event, which is the blunt and reliable option. Or integrate each phase
separately and hand the end state of one in as the start of the next, which is
exact and is what real PK software does. You will use `max_step` here, but
knowing why it is needed matters more than the argument itself.

## Your task

Three parts, worked one at a time.

| Part | What you work out | Variables to fill in |
|---|---|---|
| a | the ODE, solved numerically after a bolus | `rhs`, `sol`, `conc_num` |
| b | the same thing exactly, and the gap between them | `conc_exact`, `max_abs_err` |
| c | a 2 hour infusion, where the rate switches off | `sol_inf`, `conc_inf`, `cmax_inf`, `t_cmax` |

Below this you will find the **Setup** cell, then a **Toolkit** cell that
demonstrates every move on a tiny made up example, then **Part a**, **Part b**
and **Part c**, each with its own explanation and a cell to write in.

## The setup cell the student runs

This is given to them, already working. The data is synthetic and
seeded, so their numbers are always these numbers.

```python
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

PROBLEM_ID = "W3D4"
TOPIC = "scipy.integrate: solve_ivp, t_eval, checking against an analytic solution"

# A one compartment model. No noise anywhere: this problem is about the
# machinery being right, so everything is exactly checkable.
CL_L_H = 4.20      # clearance, L/h
V_L = 40.0         # volume of distribution, L
K_EL = CL_L_H / V_L

DOSE_MG = 500.0            # IV bolus
T_GRID = np.linspace(0.0, 24.0, 97)        # every 15 minutes

INF_DOSE_MG = 500.0        # the same dose, given as an infusion
INF_DURATION_H = 2.0
INF_RATE_MG_H = INF_DOSE_MG / INF_DURATION_H
T_GRID_INF = np.linspace(0.0, 12.0, 241)   # every 3 minutes

print("CL = %.2f L/h   V = %.1f L   k = CL/V = %.5f 1/h" % (CL_L_H, V_L, K_EL))
print("bolus    : %.0f mg at t = 0" % DOSE_MG)
print("infusion : %.0f mg over %.0f h, so %.1f mg/h"
      % (INF_DOSE_MG, INF_DURATION_H, INF_RATE_MG_H))
print("")
print("bolus output grid :", T_GRID[:4], "...", T_GRID[-1], " (%d points)" % T_GRID.size)
print("infusion grid     :", T_GRID_INF[:4], "...", T_GRID_INF[-1],
      " (%d points)" % T_GRID_INF.size)
```

## Toolkit for this problem

| What you want to do | How |
|---|---|
| import the solver | `from scipy.integrate import solve_ivp` |
| write a derivative | `def rhs(t, y, a): return [-a * y[0]]`, t first, y an array |
| solve it | `solve_ivp(rhs, (0, 24), [y0], t_eval=grid, args=(a,))` |
| the times it reported at | `sol.t` |
| the first state over time | `sol.y[0]`, one row per state |
| did it work | `sol.success` |
| stop it striding over a jump | `max_step=0.05` |
| largest absolute difference | `np.max(np.abs(a - b))` |
| where the maximum is | `np.argmax(x)` |

`y0` must be a sequence even with one compartment, so `[500.0]`, not `500.0`.
`sol.y` is always two dimensional, so `sol.y[0]` is how you get the first
compartment out.

The toolkit cell demonstrates each of those on a tiny made up example.
It is in the notebook and none of it is the answer.

## Part a. Write the differential equation and solve it

Define `rhs(t, y, cl, v)`. The amount in the body is `y[0]`, and the
derivative is `-(cl / v) * amount`. Return it as a sequence, so a list of one
element, because the solver expects one derivative per state.

Then integrate from 0 to 24 hours starting from the whole bolus in the
compartment, so `y0` is `[DOSE_MG]`. Report at `T_GRID` by passing it as
`t_eval`, and pass the parameters through with `args=(CL_L_H, V_L)`.

Pass `rtol=1e-8, atol=1e-10` as well. The defaults ask for only about three
correct digits, which is not enough for work you will compare against data.

Finally convert to concentration. The state is an **amount** in mg, so divide
by the volume to get mg/L. This is the step people skip, and the resulting
curve looks entirely reasonable while being wrong by a factor of 40.

Fill in `rhs`, the derivative function. `sol`, the object `solve_ivp` returns.
And `conc_num`, the concentration in mg/L at each time in `T_GRID`.

Moves: `def rhs(t, y, cl, v)`, `solve_ivp(fun, (0, 24), [y0], t_eval=...,
args=(...))`, `sol.y[0]`, division.

**Variables this part is graded on:** `sol`, `conc_num`

The starter cell they see:

```python
# Part a. The one compartment ODE, solved numerically after an IV bolus.

def rhs(t, y, cl, v):
    """dA/dt for one compartment. y[0] is the amount in mg."""
    return None   # YOUR CODE: return a LIST holding -(cl / v) * y[0]


sol = None       # YOUR CODE: solve_ivp from 0 to 24 h, starting from the full bolus,
                 #            reporting at T_GRID, with args=(CL_L_H, V_L)
conc_num = None  # YOUR CODE: concentration in mg/L at each time

print("solved ok:", None if sol is None else sol.success)
if conc_num is not None:
    print("conc at 0, 6, 12, 24 h:", np.round(conc_num[[0, 24, 48, 96]], 4))
```

**Every failure message part a can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `rhs` should be a function.
- `rhs` should return a SEQUENCE of derivatives, one per state, so a list like [-(cl/v) * y[0]] rather than a bare number.
- `rhs` does not compute -(cl / v) * y[0].
- `sol` should be the object solve_ivp returns.
- `sol` reports success = False, so the integration failed.
- `sol` reported at %d times but T_GRID has %d. Pass t_eval=T_GRID so the output lands on the grid you want.
- `conc_num` should be a numeric array.
- `conc_num` is two dimensional, so it is still sol.y. Take the first state with sol.y[0].
- `conc_num` should have one value per time in T_GRID, so shape (%d,).
- `conc_num` is still an AMOUNT in mg. The state is mass, so divide by the volume to get a concentration in mg/L.
- `conc_num` does not match the expected profile. Check the starting amount is the full dose and that args passes CL then V.
- `rhs` raised %s when called as rhs(t, y, cl, v).
- `rhs` has the wrong sign. The amount is falling, so the derivative is negative.
- `conc_num` is close but only to about three digits, which is the signature of solve_ivp running on its default tolerances. Pass rtol=1e-8, atol=1e-10.

## Part b. Check it against the answer you already know

An IV bolus into one compartment has an exact solution, so you can measure how
well the integration did rather than hoping.

```
C(t) = (dose / V) * exp(-(CL / V) * t)
```

Compute that on the same `T_GRID`, then take the largest absolute difference
between the two curves across all 97 points.

You should get something tiny, around ten to the minus eight. That number is
the whole point of this part: it is evidence the machinery is assembled
correctly. If instead you get something near five thousandths, your solve ran
on the default tolerances and `rtol` never got passed. When you move to a two compartment model next week, there
is no exact answer to compare against, and the only reason to trust the result
is that you checked the method somewhere you could.

If the error comes out large and roughly proportional to the values
themselves, suspect the volume division rather than the solver.

Fill in `conc_exact`, the analytic concentration at each time in `T_GRID`, and
`max_abs_err`, a single number.

Moves: `np.exp`, `np.abs`, `np.max`.

**Variables this part is graded on:** `conc_exact`, `max_abs_err`

The starter cell they see:

```python
# Part b. The exact solution, and how far the numerical one strayed.

conc_exact = None    # YOUR CODE: (dose / V) * exp(-(CL/V) * t) on T_GRID
max_abs_err = None   # YOUR CODE: largest absolute difference from conc_num

print("exact at 0, 6, 12, 24 h:",
      None if conc_exact is None else np.round(conc_exact[[0, 24, 48, 96]], 4))
print("max absolute error:", max_abs_err)
```

**Every failure message part b can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `conc_exact` should be a numeric array.
- `conc_exact` should have one value per time in T_GRID.
- `conc_exact` is an amount in mg. Divide the dose by the volume before the exponential, so it is a concentration.
- `conc_exact` is not (dose / V) * exp(-(CL/V) * t). Check the rate constant is CL divided by V.
- `max_abs_err` should be a single number. np.max reduces the whole array.
- `max_abs_err` is negative, so the absolute value was not taken.
- `max_abs_err` is %.4g. That is the signature of solve_ivp running on its default tolerances, which only ask for about three correct digits. Pass rtol=1e-8, atol=1e-10 to the solve in part a.
- `max_abs_err` is %.4g, which is far too large. The two curves should agree to roughly 1e-8. Suspect the volume division or the starting amount.

## Part c. A 2 hour infusion, where the rate switches off

Same compartment, different input. Instead of arriving instantly, the drug
goes in at a constant `INF_RATE_MG_H` for the first `INF_DURATION_H` hours and
then stops. The compartment starts empty.

```
dA/dt = rate(t) - (CL / V) * A,   where rate(t) = R for t < 2, and 0 after
```

Write that as `rhs_inf(t, y, cl, v, rate, t_off)`, using an ordinary `if` on
`t`, which is a single number here rather than an array, so a plain `if` is
fine.

Integrate from 0 to 12 hours from an empty compartment, reporting at
`T_GRID_INF`, with the same tight tolerances, and **pass `max_step=0.05`**.

A word on why, because it is more interesting than it looks. With tolerances
this tight the solver is already taking small steps, so on this particular
problem it would probably get the right answer without `max_step`. You pass it
anyway, because "probably" depends on a step size you do not control: the
solver picks it by looking at how smooth the solution has been, and a
discontinuity is precisely the thing that history cannot warn it about. The
toolkit cell shows the failure on looser tolerances, where it bites hard.
Relying on tight tolerances to paper over a discontinuity is a bet that
happens to be winning here.

The peak must land exactly at the end of the infusion, because that is the
last moment drug is still arriving faster than it is leaving.

Fill in `sol_inf`, the solver result. `conc_inf`, the concentration in mg/L at
each time. `cmax_inf`, the highest concentration reached. And `t_cmax`, the
time at which it occurs.

Moves: `def rhs_inf(t, y, cl, v, rate, t_off)`, an `if` on `t`,
`solve_ivp(..., max_step=0.05)`, `np.max`, `np.argmax`.

**Variables this part is graded on:** `sol_inf`, `conc_inf`, `cmax_inf`, `t_cmax`

The starter cell they see:

```python
# Part c. A 2 hour infusion into the same compartment.

def rhs_inf(t, y, cl, v, rate, t_off):
    """dA/dt with a zero order input that stops at t_off."""
    return None   # YOUR CODE: input rate while t < t_off, then elimination only


sol_inf = None    # YOUR CODE: solve 0 to 12 h from an empty compartment,
                  #            report at T_GRID_INF, and pass max_step=0.05
conc_inf = None   # YOUR CODE: concentration in mg/L at each time
cmax_inf = None   # YOUR CODE: the peak concentration
t_cmax = None     # YOUR CODE: the time at which the peak occurs

print("solved ok:", None if sol_inf is None else sol_inf.success)
print("Cmax = %s mg/L at t = %s h" % (cmax_inf, t_cmax))
```

**Every failure message part c can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `rhs_inf` should be a function.
- During the infusion, with an empty compartment, the derivative should be the input rate alone. Got %.4f rather than 250.
- After the infusion stops the derivative should be elimination only.
- `sol_inf` should be the object solve_ivp returns.
- `sol_inf` reports success = False.
- `sol_inf` reported at %d times but T_GRID_INF has %d. Pass t_eval.
- `conc_inf` should have one value per time in T_GRID_INF.
- `conc_inf` is an amount in mg. Divide by the volume.
- `conc_inf` does not match the expected infusion profile. If it keeps climbing past 2 hours, the solver stepped over the switch off: pass max_step=0.05.
- `cmax_inf` should be a single number.
- `cmax_inf` is an amount in mg, not a concentration.
- `cmax_inf` is not the peak of conc_inf.
- `t_cmax` should be a single number.
- `t_cmax` is the index of the peak, not the time. Use it to look up the matching entry of sol_inf.t.
- `t_cmax` should be the moment the infusion stops, which is when the peak occurs.
- `rhs_inf` raised %s when called as rhs_inf(t, y, cl, v, rate, t_off).
- After t_off the input should be zero, but the rate is still being added. Check the condition is t < t_off.
- `t_cmax` is after the infusion ended, which cannot happen: once the input stops the concentration can only fall. Either the condition in rhs_inf is not switching the rate off, or the solver stepped over the switch off, which max_step=0.05 prevents.

**Functions this problem introduces:** solve_ivp, t_eval, max_step, sol.y

---

# W3D5. Emax and sigmoid Emax: fitting a concentration response curve

- topic: scipy.optimize: Emax models, bounds, and comparing nested fits
- notebook: problems/week03/w3d5_scipy_emax_hill.ipynb
- solution: solutions/week03/w3d5_scipy_emax_hill_solution.ipynb
- expected time: 20-30 min

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

## The setup cell the student runs

This is given to them, already working. The data is synthetic and
seeded, so their numbers are always these numbers.

```python
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
```

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

The toolkit cell demonstrates each of those on a tiny made up example.
It is in the notebook and none of it is the answer.

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

**Variables this part is graded on:** `agg`

The starter cell they see:

```python
# Part a. One point per concentration.

agg = None   # YOUR CODE: CONC_UM, MEAN_PCT, SD_PCT, N

print(agg)
```

**Every failure message part a can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `agg` should be a pandas DataFrame.
- `CONC_UM` is in the index rather than a column. Call .reset_index().
- `agg` is missing the column(s) %s. ,
- `agg` still has one row per well. Group by CONC_UM so the replicates collapse to one point each.
- `agg` has %d rows but there are %d concentrations.
- `agg` does not cover the expected concentrations.
- `N` is not the number of wells at each concentration.
- `MEAN_PCT` is not the mean inhibition across the wells at each concentration.
- `SD_PCT` used n in the denominator. Pandas std already uses n-1, so this usually means numpy's std got involved.
- `SD_PCT` is not the sample SD across the wells.

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

**Variables this part is graded on:** `emax1`, `ec50_1`, `rmse1`

The starter cell they see:

```python
# Part b. The plain Emax model.

def emax_model(c, emax, ec50):
    """Plain Emax: half of emax is reached at c = ec50."""
    return None   # YOUR CODE


emax1 = None    # YOUR CODE: fitted Emax, percent
ec50_1 = None   # YOUR CODE: fitted EC50, micromolar
rmse1 = None    # YOUR CODE: RMSE of this fit against MEAN_PCT

print("Emax model: Emax = %s  EC50 = %s  rmse = %s" % (emax1, ec50_1, rmse1))
```

**Every failure message part b can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `emax_model` should be a function.
- `emax_model` should return one value per concentration, so it must work on arrays.
- `emax_model` does not compute emax * c / (ec50 + c). At c equal to ec50 it should return exactly half of emax.
- `emax1` should be a single number.
- `emax1` is not the fitted ceiling. Check the model formula and that you fitted the aggregated means from part a.
- `ec50_1` should be a single number.
- `ec50_1` is not the fitted EC50. It is a concentration in micromolar, and should land in the same range as the tested concentrations.
- `rmse1` should be a single number.
- `rmse1` is not the RMSE of this fit against MEAN_PCT. It is sqrt(mean((observed - predicted) ** 2)) at the ten concentrations.
- `emax_model` raised %s when called as emax_model(c, emax, ec50).

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

**Variables this part is graded on:** `emax2`, `ec50_2`, `hill`, `rmse2`

The starter cell they see:

```python
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
```

**Every failure message part c can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `sigmoid_model` should be a function.
- `sigmoid_model` should return one value per concentration.
- `sigmoid_model` with h = 1 should give exactly the plain Emax model back. Check that h is applied to BOTH c and ec50.
- `sigmoid_model` does not match emax * c**h / (ec50**h + c**h). With emax 100, ec50 4, h 2 and c 4 it should give exactly 50, because c equals ec50. Check that h is raised on the ec50 term as well as on c.
- `emax2` should be a single number.
- `emax2` is not the fitted ceiling of the sigmoid model.
- `ec50_2` should be a single number.
- `ec50_2` is not the fitted EC50 of the sigmoid model.
- `hill` should be a single number.
- `hill` came back at exactly 1.0, so it probably never moved. Check it is a free parameter in your model rather than hard coded.
- `hill` is not the fitted Hill coefficient. Make sure h appears on both c and ec50, and that p0 starts it at 1.0.
- `rmse2` should be a single number.
- `rmse2` is not the RMSE of the sigmoid fit against MEAN_PCT.
- `rmse2` came out larger than `rmse1`, which should be impossible for nested models. The sigmoid can always set h to 1 and reproduce the simpler fit, so a worse result means the fit did not converge.
- `sigmoid_model` raised %s when called as sigmoid_model(c, emax, ec50, h).

**Functions this problem introduces:** Emax model, sigmoid Emax, bounds, np.logspace
