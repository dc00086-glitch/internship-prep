PID = "W5D2"
WEEK = 5
DAY = 2
SLUG = "w5d2_sklearn_cv_leakage"
TITLE = "Cross-validation, and the leak that invents a result from noise"
TOPIC = "scikit-learn: KFold, cross_val_score, Pipeline, and data leakage"
EST = "20-30 min"

SCENARIO_MD = '''
## Scenario

A collaborator has run an exploratory biomarker panel on 80 subjects: 150
analytes, measured because the platform could measure them, with no prior
hypothesis attached to any of them. The question is whether any of them
predict clearance beyond what weight and renal function already explain.

In truth, **none of the 150 do**. They are noise. This problem is built that
way on purpose, because it means you know the right answer and can therefore
recognise a method that manufactures the wrong one.

The usual workflow is to screen the panel for the most promising analytes,
then cross-validate a model built on those. Done in that order it reports a
respectable predictive performance on pure noise, and nothing about the output
looks wrong. This is probably the single most common serious error in applied
machine learning, and it has put a lot of biomarkers into a lot of papers.
'''

CONCEPT_MD = '''
## Cross-validation, and what counts as touching the test data

**Cross-validation reuses every subject.** A single train/test split spends a
quarter of your data on one noisy estimate. K-fold splits the data into k
parts, trains on k-1 and tests on the one left out, and rotates. Every subject
is tested exactly once, every subject is trained on k-1 times, and you get k
estimates whose spread tells you how stable the answer is.

`KFold(n_splits=5, shuffle=True, random_state=1)` is the usual setup. Shuffle
matters: without it the folds are consecutive blocks of the file, and if the
file is ordered by anything, by site, by cohort, by date, your folds are
confounded with it.

`cross_val_score(model, X, y, cv=kf, scoring="r2")` runs the whole thing and
returns the k scores. Report their **mean and SD**, because a mean of 0.4 made
of folds from 0.1 to 0.7 is a very different claim from one made of folds all
near 0.4.

**Leakage is the failure where information from the test fold reaches the
model.** It does not need to be dramatic. Anything fitted on all the data
before cross-validation starts carries information from every fold into every
other, and that includes steps that feel like data preparation rather than
modelling.

**Feature selection is the worst offender, and here is why.** With 150 pure
noise analytes and only 80 subjects, some of them will correlate with
clearance by chance. Screen on all the data, keep the best six, and those six
were chosen
*because* they happened to match the target in the very rows you will later
use for testing. Cross-validation then dutifully holds out a fold, but the
features it is working with already know about that fold. The reported score
comes out clearly positive for data containing no signal at all.

The tell is that it is invisible. There is no warning, the code is short, and
the number that comes out is modest rather than absurd. You should get an R2
around 0.13 here, which is not a suspicious result at all. For a novel
biomarker signature explaining 13 percent of the variance in clearance, that
is a figure, a paragraph in the discussion, and quite possibly a follow-up
study. From columns of random numbers.

**The fix is Pipeline.** A `Pipeline` chains steps into a single estimator,
and when cross-validation refits that estimator on each training fold, every
step inside it is refitted too, on that fold only. Selection happens 5 times,
on 5 different subsets, each blind to its own test fold.

The rule worth memorising: **anything that learns from the data belongs inside
the pipeline.** Scaling learns a mean and an SD. Imputation learns a fill
value. Feature selection learns which columns. All of them belong inside. Only
operations that treat each row independently, such as taking a log, are safe
outside.
'''

TASK_MD = '''
## Your task

Three parts, worked one at a time.

| Part | What you work out | Variables to fill in |
|---|---|---|
| a | honest cross-validation on the real covariates | `cv_scores`, `cv_mean`, `cv_sd` |
| b | the leak: screen first, then cross-validate | `X_leaky`, `cv_leaky_mean` |
| c | the fix: screen inside the pipeline | `pipe`, `cv_clean_mean`, `leak_inflation` |

Below this you will find the **Setup** cell, then a **Toolkit** cell that
demonstrates every move on a tiny made up example, then **Part a**, **Part b**
and **Part c**, each with its own explanation and a cell to write in.
'''

DATA_CODE = '''
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression

PROBLEM_ID = "W5D2"
TOPIC = "scikit-learn: KFold, cross_val_score, Pipeline, and data leakage"

rng = np.random.default_rng(5102)

N = 80
N_NOISE = 150
K_SELECT = 6

_weight = np.round(np.clip(rng.normal(74.0, 13.0, N), 45.0, 125.0), 1)
_crcl = np.round(np.clip(rng.normal(104.0, 24.0, N), 35.0, 180.0), 1)
_cl = 4.1 * (_weight / 70.0) ** 0.75 * (_crcl / 100.0) ** 0.40
cl = np.round(_cl * rng.lognormal(0.0, 0.18, N), 3)

# The biomarker panel. Every column is pure noise, unrelated to clearance.
_panel = rng.normal(0.0, 1.0, size=(N, N_NOISE))
panel = pd.DataFrame(_panel, columns=["BM%02d" % i for i in range(1, N_NOISE + 1)])

X_real = pd.DataFrame({"WEIGHT_KG": _weight, "CRCL_ML_MIN": _crcl})
X_panel = panel
y = pd.Series(cl, name="CL_L_H")

print("subjects:", N)
print("real covariates :", list(X_real.columns))
print("biomarker panel :", X_panel.shape[1], "analytes, all of them pure noise")
print("")
print("strongest single correlation between any analyte and clearance:")
_corrs = np.array([abs(np.corrcoef(X_panel.iloc[:, i], y)[0, 1]) for i in range(N_NOISE)])
print("  %.4f  (%s)" % (_corrs.max(), X_panel.columns[int(_corrs.argmax())]))
print("  with %d noise columns and only %d subjects, something always looks promising"
      % (N_NOISE, N))
'''

TOOLKIT_MD = '''
## Toolkit for this problem

| What you want to do | How |
|---|---|
| set up the folds | `KFold(n_splits=5, shuffle=True, random_state=1)` |
| run cross-validation | `cross_val_score(model, X, y, cv=kf, scoring="r2")` |
| summarise the folds | `scores.mean()`, `scores.std(ddof=1)` |
| pick the k best features | `SelectKBest(f_regression, k=10)` |
| apply a selector | `.fit_transform(X, y)` |
| chain steps into one estimator | `Pipeline([("sel", ...), ("model", ...)])` |

`cross_val_score` refits the estimator you hand it on each training fold. If
that estimator is a `Pipeline`, every step inside gets refitted per fold,
which is exactly what stops a selection step from seeing its own test data.
'''

TOOLKIT_CODE = '''
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression

# A tiny made up example with NO signal at all. Nothing here is the answer.
demo_rng = np.random.default_rng(3)
demo_n, demo_p = 60, 40
demo_X = pd.DataFrame(demo_rng.normal(size=(demo_n, demo_p)),
                      columns=["f%02d" % i for i in range(demo_p)])
demo_y = pd.Series(demo_rng.normal(size=demo_n))      # unrelated to demo_X
demo_kf = KFold(n_splits=5, shuffle=True, random_state=1)

print("40 random features, 60 samples, and a target that is pure noise.")
print("The honest answer for any method here is: no predictive power.")

print("")
print("--- 1. honest cross-validation on everything ---")
demo_all = cross_val_score(LinearRegression(), demo_X, demo_y, cv=demo_kf, scoring="r2")
print("fold scores:", np.round(demo_all, 3))
print("mean R2    : %.3f   <- negative, which is correct: worse than the mean"
      % demo_all.mean())

print("")
print("--- 2. the leak: select on ALL the data, then cross-validate ---")
demo_sel = SelectKBest(f_regression, k=5).fit(demo_X, demo_y)
demo_leaky = demo_X.loc[:, demo_sel.get_support()]
print("features chosen:", list(demo_leaky.columns))
demo_leak_scores = cross_val_score(LinearRegression(), demo_leaky, demo_y,
                                   cv=demo_kf, scoring="r2")
print("fold scores:", np.round(demo_leak_scores, 3))
print("mean R2    : %.3f   <- a positive result, from a target that is pure noise"
      % demo_leak_scores.mean())

print("")
print("--- 3. the fix: selection inside the pipeline ---")
demo_pipe = Pipeline([("sel", SelectKBest(f_regression, k=5)),
                      ("model", LinearRegression())])
demo_clean = cross_val_score(demo_pipe, demo_X, demo_y, cv=demo_kf, scoring="r2")
print("fold scores:", np.round(demo_clean, 3))
print("mean R2    : %.3f   <- back to correctly reporting nothing" % demo_clean.mean())

print("")
print("--- 4. the three numbers side by side ---")
print("  everything, honest        : %+.3f" % demo_all.mean())
print("  selected first, then CV   : %+.3f   <- invented" % demo_leak_scores.mean())
print("  selection inside pipeline : %+.3f" % demo_clean.mean())
print("")
print("Nothing about the middle one looks wrong from the outside. That is the problem.")
'''

PARTS = [
    dict(
        label="a",
        prompt_md='''
## Part a. Honest cross-validation on the real covariates

Start with what you know works. Cross-validate a `LinearRegression` on
`X_real`, the two genuine covariates, using `KFold` with 5 splits, shuffling,
and `random_state=1`.

Keep all five fold scores, then summarise them with a mean and a sample SD.

Report both. The mean is the headline, but the SD tells you how much to trust
it. Five folds of 30 subjects each give noticeably variable estimates, and a
mean of 0.5 built from folds spanning 0.2 to 0.8 is a much weaker claim than
the same mean built from folds all near 0.5.

This number is your reference point. Weight and renal function genuinely drive
clearance in this dataset, so whatever you get here is roughly what real
signal looks like. Parts b and c are about what noise can be made to look
like.

Fill in `cv_scores`, `cv_mean`, `cv_sd`.

Moves: `KFold(n_splits=5, shuffle=True, random_state=1)`,
`cross_val_score(model, X, y, cv=kf, scoring="r2")`, `.mean()`,
`.std(ddof=1)`.
''',
        starter='''
# Part a. Cross-validate on the two real covariates.

cv_scores = None   # YOUR CODE: the five fold R2 scores for LinearRegression on X_real
cv_mean = None     # YOUR CODE: their mean
cv_sd = None       # YOUR CODE: their sample SD

print("fold scores:", None if cv_scores is None else np.round(cv_scores, 4))
print("mean R2    :", cv_mean)
print("SD across folds:", cv_sd)
''',
        solution='''
# Part a. Cross-validate on the two real covariates.

_kf = KFold(n_splits=5, shuffle=True, random_state=1)
cv_scores = cross_val_score(LinearRegression(), X_real, y, cv=_kf, scoring="r2")
cv_mean = float(cv_scores.mean())
cv_sd = float(cv_scores.std(ddof=1))

print("fold scores:", None if cv_scores is None else np.round(cv_scores, 4))
print("mean R2    :", cv_mean)
print("SD across folds:", cv_sd)
''',
    ),
    dict(
        label="b",
        prompt_md='''
## Part b. Do it the way everybody does it, and watch what happens

Now the biomarker panel, screened the natural way round.

Fit a `SelectKBest(f_regression, k=K_SELECT)` on the **whole** panel and the
whole target, to find the ten analytes most associated with clearance. Use
`.get_support()` to get a boolean mask of which columns were chosen, and build
`X_leaky` from those columns of `X_panel`.

Then cross-validate a plain `LinearRegression` on `X_leaky`, with the same
folds, and take the mean.

Read the number you get against what you know: every one of those 150 analytes
is random noise with no relationship to clearance whatsoever. The honest
answer is zero or below. Compare it with your part a result, which came from
genuine biology.

Nothing in this code is sloppy-looking. It is two short, readable steps in the
order that feels natural, screen then validate, and it is wrong because the
screening step has already read the answers in every fold it will later be
tested on.

Fill in `X_leaky`, `cv_leaky_mean`.

Moves: `SelectKBest(f_regression, k=K_SELECT).fit(X_panel, y)`,
`.get_support()`, `X_panel.loc[:, mask]`, `cross_val_score`, `.mean()`.
''',
        starter='''
# Part b. Screen on all the data first, then cross-validate. The common mistake.

X_leaky = None         # YOUR CODE: the K_SELECT columns of X_panel that score best
                       #            against y, chosen using ALL the data
cv_leaky_mean = None   # YOUR CODE: mean CV R2 of LinearRegression on X_leaky

print("columns chosen:", None if X_leaky is None else list(X_leaky.columns))
print("mean R2, selected first:", cv_leaky_mean)
print("mean R2, real covariates:", cv_mean)
''',
        solution='''
# Part b. Screen on all the data first, then cross-validate. The common mistake.

_selector = SelectKBest(f_regression, k=K_SELECT).fit(X_panel, y)
X_leaky = X_panel.loc[:, _selector.get_support()]
cv_leaky_mean = float(cross_val_score(LinearRegression(), X_leaky, y,
                                      cv=_kf, scoring="r2").mean())

print("columns chosen:", None if X_leaky is None else list(X_leaky.columns))
print("mean R2, selected first:", cv_leaky_mean)
print("mean R2, real covariates:", cv_mean)
''',
    ),
    dict(
        label="c",
        prompt_md='''
## Part c. Put the selection inside the pipeline

Build a `Pipeline` with two steps: a `SelectKBest(f_regression, k=K_SELECT)`
named `"sel"`, and a `LinearRegression` named `"model"`. Cross-validate that
pipeline on the **full** panel, `X_panel`, with the same folds, and take the
mean.

The difference is entirely about *when* the selection happens.
`cross_val_score` refits whatever estimator you give it on each training fold.
Hand it a pipeline and the selection step is refitted five times, each on four
fifths of the data, each blind to the fold it will be scored on. Different
analytes get chosen each time, which is itself the signal that they were never
real.

Then quantify the damage: `leak_inflation` is `cv_leaky_mean -
cv_clean_mean`.

The clean number should land at or below zero, which is the correct answer for
a panel of pure noise. Negative R2 is not a bug: it means the model does worse
than predicting the mean every time, which is exactly what you would expect
from fitting six random columns.

The gap between the two is the size of a finding you could have published.

Fill in `pipe`, `cv_clean_mean`, `leak_inflation`.

Moves: `Pipeline([("sel", ...), ("model", ...)])`, `cross_val_score`,
`.mean()`, subtraction.
''',
        starter='''
# Part c. Selection inside the pipeline, so each fold is blind to its own test data.

pipe = None             # YOUR CODE: Pipeline with steps named "sel" and "model"
cv_clean_mean = None    # YOUR CODE: mean CV R2 of the pipeline on the FULL X_panel
leak_inflation = None   # YOUR CODE: cv_leaky_mean - cv_clean_mean

print("mean R2, selection leaked  : %s" % cv_leaky_mean)
print("mean R2, selection in pipe : %s" % cv_clean_mean)
print("inflation caused by the leak: %s" % leak_inflation)
print("")
print("for reference, the real covariates gave: %s" % cv_mean)
''',
        solution='''
# Part c. Selection inside the pipeline, so each fold is blind to its own test data.

pipe = Pipeline([("sel", SelectKBest(f_regression, k=K_SELECT)),
                 ("model", LinearRegression())])
cv_clean_mean = float(cross_val_score(pipe, X_panel, y, cv=_kf, scoring="r2").mean())
leak_inflation = float(cv_leaky_mean - cv_clean_mean)

print("mean R2, selection leaked  : %s" % cv_leaky_mean)
print("mean R2, selection in pipe : %s" % cv_clean_mean)
print("inflation caused by the leak: %s" % leak_inflation)
print("")
print("for reference, the real covariates gave: %s" % cv_mean)
''',
    ),
]


def build_expected(ns):
    import numpy as np
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.feature_selection import SelectKBest, f_regression
    X_real, X_panel, y = ns["X_real"], ns["X_panel"], ns["y"]
    k = ns["K_SELECT"]
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    real = cross_val_score(LinearRegression(), X_real, y, cv=kf, scoring="r2")
    sel = SelectKBest(f_regression, k=k).fit(X_panel, y)
    leaky_cols = [str(c) for c in X_panel.columns[sel.get_support()]]
    leaky = cross_val_score(LinearRegression(), X_panel.loc[:, sel.get_support()],
                            y, cv=kf, scoring="r2")
    pipe = Pipeline([("sel", SelectKBest(f_regression, k=k)),
                     ("model", LinearRegression())])
    clean = cross_val_score(pipe, X_panel, y, cv=kf, scoring="r2")
    allcols = cross_val_score(LinearRegression(), X_panel, y, cv=kf, scoring="r2")
    return {
        "n_folds": 5,
        "cv_scores": [float(v) for v in real],
        "cv_mean": float(real.mean()),
        "cv_sd": float(real.std(ddof=1)),
        "cv_sd_pop": float(real.std()),
        "leaky_cols": leaky_cols,
        "k_select": int(k),
        "cv_leaky_mean": float(leaky.mean()),
        "cv_clean_mean": float(clean.mean()),
        "cv_allcols_mean": float(allcols.mean()),
        "leak_inflation": float(leaky.mean() - clean.mean()),
        "n_panel_cols": int(X_panel.shape[1]),
    }


CHECK_CODE = '''
def _part_a():
    cv_scores = _need("cv_scores")
    cv_mean = _need("cv_mean")
    cv_sd = _need("cv_sd")

    arr = _as_array(cv_scores)
    if arr is None or arr.ndim != 1:
        return "`cv_scores` should be the 1D array cross_val_score returns."
    if arr.size != _EXP["n_folds"]:
        return ("`cv_scores` has %d entries but KFold with 5 splits gives 5."
                % arr.size)
    if not _arr_close(arr, _EXP["cv_scores"], rtol=1e-4):
        return ("`cv_scores` does not match. Use LinearRegression on X_real, with "
                "KFold(n_splits=5, shuffle=True, random_state=1) and scoring='r2'.")

    got = _as_scalar(cv_mean)
    if got is None or not _num_close(got, _EXP["cv_mean"], rtol=1e-4):
        return "`cv_mean` is not the mean of cv_scores."

    got = _as_scalar(cv_sd)
    if got is None:
        return "`cv_sd` should be a single number."
    if _num_close(got, _EXP["cv_sd_pop"], rtol=1e-6):
        return ("`cv_sd` used ddof=0. numpy and the array returned by cross_val_score both "
                "default to the population SD, so pass ddof=1.")
    if not _num_close(got, _EXP["cv_sd"], rtol=1e-4):
        return "`cv_sd` is not the sample SD of cv_scores."
    return None


def _part_b():
    X_leaky = _need("X_leaky")
    cv_leaky_mean = _need("cv_leaky_mean")

    if _np.ndim(X_leaky) != 2:
        return "`X_leaky` should be a 2D table of the selected columns."
    if _np.shape(X_leaky)[1] == _EXP["n_panel_cols"]:
        return ("`X_leaky` still has all %d analytes, so no selection happened. Use "
                "get_support() to pick the K_SELECT best columns."
                % _EXP["n_panel_cols"])
    if _np.shape(X_leaky)[1] != _EXP["k_select"]:
        return ("`X_leaky` has %d columns but K_SELECT is %d."
                % (_np.shape(X_leaky)[1], _EXP["k_select"]))
    try:
        cols = [str(c) for c in X_leaky.columns]
    except Exception:
        cols = None
    if cols is not None and sorted(cols) != sorted(_EXP["leaky_cols"]):
        return ("`X_leaky` does not hold the columns SelectKBest picks. Fit it on the whole "
                "X_panel and y, which is the mistake this part is demonstrating.")

    got = _as_scalar(cv_leaky_mean)
    if got is None:
        return "`cv_leaky_mean` should be a single number."
    if _num_close(got, _EXP["cv_clean_mean"], rtol=1e-4):
        return ("`cv_leaky_mean` matches the honest answer, so the selection was not done "
                "on all the data first. Part b is deliberately demonstrating the leak.")
    if not _num_close(got, _EXP["cv_leaky_mean"], rtol=1e-4):
        return ("`cv_leaky_mean` does not match. Cross-validate a plain LinearRegression on "
                "X_leaky with the same KFold.")
    return None


def _part_c():
    pipe = _need("pipe")
    cv_clean_mean = _need("cv_clean_mean")
    leak_inflation = _need("leak_inflation")

    if not hasattr(pipe, "steps"):
        return "`pipe` should be a sklearn Pipeline."
    names = [str(n) for n, _ in pipe.steps]
    if names != ["sel", "model"]:
        return ("`pipe` should have exactly two steps named 'sel' and 'model', in that "
                "order. Yours are %s." % names)
    first = pipe.steps[0][1]
    if not hasattr(first, "get_support"):
        return "`pipe`'s first step should be the SelectKBest selector."
    if not hasattr(pipe.steps[1][1], "predict"):
        return "`pipe`'s second step should be the LinearRegression."

    got = _as_scalar(cv_clean_mean)
    if got is None:
        return "`cv_clean_mean` should be a single number."
    if _num_close(got, _EXP["cv_leaky_mean"], rtol=1e-4):
        return ("`cv_clean_mean` equals the leaked answer, so the pipeline was run on "
                "X_leaky. Run it on the FULL X_panel, so the selection happens inside each "
                "fold.")
    if _num_close(got, _EXP["cv_allcols_mean"], rtol=1e-4) and \\
            not _num_close(got, _EXP["cv_clean_mean"], rtol=1e-4):
        return ("`cv_clean_mean` looks like a plain regression on all 80 analytes with no "
                "selection step. Check the pipeline really contains the selector.")
    if not _num_close(got, _EXP["cv_clean_mean"], rtol=1e-4):
        return "`cv_clean_mean` does not match cross-validating the pipeline on X_panel."

    got = _as_scalar(leak_inflation)
    if got is None:
        return "`leak_inflation` should be a single number."
    if _num_close(got, -_EXP["leak_inflation"], rtol=1e-4):
        return ("`leak_inflation` has the sign the wrong way round. It is the leaked score "
                "minus the clean one, so the inflation comes out positive.")
    if not _num_close(got, _EXP["leak_inflation"], rtol=1e-3):
        return "`leak_inflation` is not cv_leaky_mean minus cv_clean_mean."
    return None
'''

# Functions this problem introduces. Keys must exist in tools/function_catalog.py.
FUNCTIONS = [
    "KFold",
    "cross_val_score",
    "Pipeline",
    "SelectKBest",
    "data leakage",
]
