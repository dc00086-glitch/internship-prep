PID = "W5D3"
WEEK = 5
DAY = 3
SLUG = "w5d3_sklearn_random_forest"
TITLE = "Random forests, and two different answers to which feature matters"
TOPIC = "scikit-learn: RandomForestRegressor, impurity importance, permutation importance"
EST = "20-30 min"

SCENARIO_MD = '''
## Scenario

The same 200 subjects and the same five covariates as day 1, so the comparison
is exact. On day 1 a linear regression explained clearance with almost no gap
between its training and test performance, because five coefficients cannot
memorise a hundred and fifty subjects.

A random forest has no such limitation. It is hundreds of decision trees, each
grown until its leaves are nearly pure, each seeing a different bootstrap
sample and a different random subset of features at every split. Left
unrestricted it can fit the training data almost perfectly, including its
noise.

That flexibility buys you the ability to capture interactions and curvature
nobody specified in advance, which matters for allometric scaling and
threshold effects. It costs you the comfortable honesty of a model that cannot
overfit. You will see both in the same set of numbers.
'''

CONCEPT_MD = '''
## Forests, and why importance has two meanings

**What a random forest is.** Many decision trees, each trained on a bootstrap
resample of the rows, each choosing from a random subset of features at every
split. Averaging their predictions cancels most of the variance any single
deep tree would have. It needs no feature scaling, handles non-linearity and
interactions without being told they exist, and is close to the strongest
thing you can run without tuning.

Its main cost is interpretability. There is no coefficient to report, no
"clearance rises 0.05 L/h per kilogram". There is only a prediction and some
notion of which inputs mattered.

**Impurity importance**, which is `feature_importances_`, totals up how much
each feature reduced the error at the splits where it was used, across all
trees. It is free, because it falls out of training, and it has two known
biases. It favours features with many distinct values, because a continuous
variable offers far more candidate split points than a binary one. And when
two features are correlated, it splits the credit between them more or less
arbitrarily, so each looks half as important as either really is.

**Permutation importance** asks a different and more honest question: if I
shuffle this one column and leave everything else alone, how much worse do the
predictions get? Shuffling destroys that feature's relationship with the
target while preserving its distribution, so the drop in score is a direct
measure of how much the fitted model was relying on it.

Three things make it the better default. It is computed on **held-out** data,
so it measures reliance on something that generalises rather than on something
memorised. It is in units of the score, so a drop of 0.08 R2 means something
you can state. And it has no cardinality bias, because every feature gets the
same treatment.

Its own weakness is correlated features: shuffling one of a correlated pair
leaves the model able to lean on the other, so both can look unimportant even
when the pair together matters a great deal.

**So read them together.** A feature that is high on both is doing real work.
A feature high on impurity but near zero on permutation is usually being
rewarded for offering lots of split points rather than for carrying
information. In this dataset one covariate is pure noise by construction, and
watching where each method puts it is the exercise.
'''

TASK_MD = '''
## Your task

Three parts, worked one at a time.

| Part | What you work out | Variables to fill in |
|---|---|---|
| a | the forest, and how much it flatters itself | `rf`, `r2_test_rf`, `r2_train_rf`, `gap_rf` |
| b | impurity importance, which comes free with training | `imp_impurity` |
| c | permutation importance, measured on held-out data | `perm`, `imp_perm`, `noise_rank_impurity`, `noise_rank_perm` |

Below this you will find the **Setup** cell, then a **Toolkit** cell that
demonstrates every move on a tiny made up example, then **Part a**, **Part b**
and **Part c**, each with its own explanation and a cell to write in.
'''

DATA_CODE = '''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score

PROBLEM_ID = "W5D3"
TOPIC = "scikit-learn: RandomForestRegressor, impurity importance, permutation importance"

rng = np.random.default_rng(5101)      # the same study as day 1

N = 200
FEATURES = ["WEIGHT_KG", "AGE_Y", "SEX_M", "CRCL_ML_MIN", "ALT_U_L"]
NOISE_FEATURE = "ALT_U_L"              # has no effect on clearance, by construction

_weight = np.round(np.clip(rng.normal(74.0, 13.0, N), 45.0, 125.0), 1)
_age = np.round(np.clip(rng.normal(42.0, 13.0, N), 18.0, 75.0), 0)
_sex = rng.integers(0, 2, N)
_crcl = np.round(np.clip(rng.normal(104.0, 24.0, N) - 0.45 * (_age - 42.0), 35.0, 180.0), 1)
_alt = np.round(np.clip(rng.lognormal(np.log(24.0), 0.42, N), 5.0, 160.0), 1)

_cl = (4.1 * (_weight / 70.0) ** 0.75
       * (_crcl / 100.0) ** 0.40
       * np.exp(-0.0045 * (_age - 42.0))
       * (1.0 + 0.06 * _sex))
cl = np.round(_cl * rng.lognormal(0.0, 0.17, N), 3)

subjects = pd.DataFrame({
    "WEIGHT_KG": _weight, "AGE_Y": _age, "SEX_M": _sex,
    "CRCL_ML_MIN": _crcl, "ALT_U_L": _alt, "CL_L_H": cl,
})
X = subjects[FEATURES]
y = subjects["CL_L_H"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print("same 200 subjects and the same split as day 1")
print("train:", X_train.shape, "  test:", X_test.shape)
print("")
print("for reference, day 1's linear regression on this split gave")
print("  R2 test 0.6095, R2 train 0.5636, so a gap of -0.046")
print("")
print("%s is pure noise: it was generated independently of clearance" % NOISE_FEATURE)
'''

TOOLKIT_MD = '''
## Toolkit for this problem

| What you want to do | How |
|---|---|
| fit a forest | `RandomForestRegressor(n_estimators=300, random_state=0).fit(X_train, y_train)` |
| score it | `r2_score(y_test, rf.predict(X_test))` |
| impurity importance | `rf.feature_importances_`, in column order |
| as a sorted table | `pd.Series(values, index=X.columns).sort_values(ascending=False)` |
| permutation importance | `permutation_importance(rf, X_test, y_test, n_repeats=20, random_state=0)` |
| its mean drop per feature | `result.importances_mean` |
| where a name sits in a ranking | `list(series.index).index("NAME") + 1` |

`feature_importances_` always sums to 1, because it is a share of total
impurity reduction. Permutation importances do not, because each is a drop in
R2 measured on its own.
'''

TOOLKIT_CODE = '''
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# A tiny made up dataset. "useful" drives y, "manyvals" and "binary" do not.
demo_rng = np.random.default_rng(0)
demo_n = 300
demo_X = pd.DataFrame({
    "useful": demo_rng.normal(size=demo_n),
    "manyvals": demo_rng.normal(size=demo_n),            # continuous, but irrelevant
    "binary": demo_rng.integers(0, 2, demo_n),           # irrelevant and low cardinality
})
demo_y = 2.0 * demo_X["useful"] + demo_rng.normal(0, 0.6, demo_n)
demo_Xtr, demo_Xte, demo_ytr, demo_yte = train_test_split(
    demo_X, demo_y, test_size=0.3, random_state=0)

print("--- 1. a forest fits its training data very closely ---")
demo_rf = RandomForestRegressor(n_estimators=200, random_state=0).fit(demo_Xtr, demo_ytr)
demo_r2_tr = r2_score(demo_ytr, demo_rf.predict(demo_Xtr))
demo_r2_te = r2_score(demo_yte, demo_rf.predict(demo_Xte))
print("R2 train: %.4f" % demo_r2_tr)
print("R2 test : %.4f" % demo_r2_te)
print("gap     : %.4f   <- far larger than a linear model would show"
      % (demo_r2_tr - demo_r2_te))

print("")
print("--- 2. impurity importance, free with training ---")
demo_imp = pd.Series(demo_rf.feature_importances_,
                     index=demo_X.columns).sort_values(ascending=False)
print(demo_imp.round(4).to_string())
print("sums to %.4f, because it is a share of total impurity reduction" % demo_imp.sum())
print("")
print("note how much the two IRRELEVANT features still receive, and that the")
print("continuous one receives more than the binary one despite both being noise")

print("")
print("--- 3. permutation importance, measured on held out data ---")
demo_perm = permutation_importance(demo_rf, demo_Xte, demo_yte,
                                   n_repeats=20, random_state=0)
demo_pi = pd.Series(demo_perm.importances_mean,
                    index=demo_X.columns).sort_values(ascending=False)
print(demo_pi.round(4).to_string())
print("these are drops in R2, so they do not sum to anything in particular")
print("the irrelevant features sit at or below zero, which is the honest answer")

print("")
print("--- 4. the two rankings side by side ---")
demo_cmp = pd.DataFrame({"impurity": demo_imp, "permutation": demo_pi}).round(4)
print(demo_cmp.to_string())
'''

PARTS = [
    dict(
        label="a",
        prompt_md='''
## Part a. Fit the forest and measure the gap

Fit a `RandomForestRegressor` with `n_estimators=300` and `random_state=0` on
the training data. Leave the depth unrestricted, which is the default, because
the point here is to see what an unrestricted forest does.

Score it on the test set and on the training set, exactly as on day 1, and
take the gap as `r2_train_rf - r2_test_rf`.

Then compare both numbers with day 1, printed for you in the setup cell. Two
things should stand out, and they pull in opposite directions.

The training R2 will be very high, far above the linear model's. That is not
the forest understanding clearance better. It is the forest having enough
capacity to remember a hundred and fifty subjects, so its description of them
is close to perfect and close to meaningless.

The test R2 is the number that counts, and it may or may not beat the linear
model. With five well-behaved covariates and a smooth underlying relationship,
a linear model is hard to improve on, and a forest often does not. Flexibility
is only an advantage when there is something there that a straight line
cannot express.

`random_state=0` matters as much here as the split did. A forest is built from
bootstrap samples and random feature subsets, so without it every run gives
different importances and you cannot tell a change from a reshuffle.

Fill in `rf`, `r2_test_rf`, `r2_train_rf`, `gap_rf`.

Moves: `RandomForestRegressor(n_estimators=300, random_state=0).fit(...)`,
`.predict(...)`, `r2_score`, subtraction.
''',
        starter='''
# Part a. Fit a random forest and compare train against test.

rf = None            # YOUR CODE: RandomForestRegressor, 300 trees, random_state=0,
                     #            fitted on the training data
r2_test_rf = None    # YOUR CODE: R2 on the test set
r2_train_rf = None   # YOUR CODE: R2 on the training set
gap_rf = None        # YOUR CODE: train minus test

print("R2 test  : %s" % r2_test_rf)
print("R2 train : %s" % r2_train_rf)
print("gap      : %s   (day 1 linear model: -0.046)" % gap_rf)
''',
        solution='''
# Part a. Fit a random forest and compare train against test.

rf = RandomForestRegressor(n_estimators=300, random_state=0).fit(X_train, y_train)
r2_test_rf = float(r2_score(y_test, rf.predict(X_test)))
r2_train_rf = float(r2_score(y_train, rf.predict(X_train)))
gap_rf = float(r2_train_rf - r2_test_rf)

print("R2 test  : %s" % r2_test_rf)
print("R2 train : %s" % r2_train_rf)
print("gap      : %s   (day 1 linear model: -0.046)" % gap_rf)
''',
    ),
    dict(
        label="b",
        prompt_md='''
## Part b. Impurity importance

`rf.feature_importances_` gives one number per feature, in the same order as
the columns of `X`. Put them in a `pd.Series` indexed by the feature names and
sort descending, so the table reads as a ranking.

They sum to 1, because each is a share of the total impurity reduction across
all trees.

Look at where `ALT_U_L` lands. It was generated independently of clearance and
contributes nothing, yet it will receive a visible share, not zero. That is
not a bug in the method. Every split on a noise feature still reduces impurity
a little, purely by chance, and a forest with hundreds of unrestricted trees
makes a great many splits.

Notice also that `ALT_U_L` is continuous with many distinct values while
`SEX_M` is binary with two. That alone gives ALT far more candidate split
points to get lucky with, which is the cardinality bias in action. The
comparison between those two features is the one to look at.

Fill in `imp_impurity`, a Series indexed by feature name, sorted descending.

Moves: `rf.feature_importances_`, `pd.Series(values, index=...)`,
`.sort_values(ascending=False)`.
''',
        starter='''
# Part b. Impurity importance, sorted.

imp_impurity = None   # YOUR CODE: Series indexed by feature name, sorted descending

print(None if imp_impurity is None else imp_impurity.round(4).to_string())
print("")
print("sums to:", None if imp_impurity is None else round(float(imp_impurity.sum()), 6))
''',
        solution='''
# Part b. Impurity importance, sorted.

imp_impurity = pd.Series(rf.feature_importances_,
                         index=FEATURES).sort_values(ascending=False)

print(None if imp_impurity is None else imp_impurity.round(4).to_string())
print("")
print("sums to:", None if imp_impurity is None else round(float(imp_impurity.sum()), 6))
''',
    ),
    dict(
        label="c",
        prompt_md='''
## Part c. Permutation importance, and where the noise feature lands

Run `permutation_importance` on the **test** set with `n_repeats=20` and
`random_state=0`. It shuffles one column at a time and measures how much the
test score falls, repeating to average over the randomness of the shuffle.

Put `importances_mean` into a Series indexed by feature name, sorted
descending. These are drops in R2, so they are in score units and do not sum
to anything meaningful.

Then record where `ALT_U_L` sits in each ranking, counting from 1 for the most
important.

Compare the two. `ALT_U_L` should sit mid-table on impurity importance, with a
non-trivial share, and at or near the bottom on permutation importance, quite
possibly with a **negative** value. A negative permutation importance means
shuffling the column made predictions slightly better, which is the
unambiguous signature of a feature the model should never have leaned on.

Which ranking you would take to a meeting is the point of the exercise. The
one computed on data the model has never seen is measuring something real.

Fill in `perm`, `imp_perm`, `noise_rank_impurity`, `noise_rank_perm`.

Moves: `permutation_importance(rf, X_test, y_test, n_repeats=20,
random_state=0)`, `.importances_mean`, `pd.Series(...)`,
`list(series.index).index(NOISE_FEATURE) + 1`.
''',
        starter='''
# Part c. Permutation importance on held out data.

perm = None                  # YOUR CODE: permutation_importance on the TEST set,
                             #            n_repeats=20, random_state=0
imp_perm = None              # YOUR CODE: Series of importances_mean, sorted descending
noise_rank_impurity = None   # YOUR CODE: position of ALT_U_L in imp_impurity, from 1
noise_rank_perm = None       # YOUR CODE: position of ALT_U_L in imp_perm, from 1

print(None if imp_perm is None else imp_perm.round(4).to_string())
print("")
print("%s ranks %s by impurity, %s by permutation (of %d features)"
      % (NOISE_FEATURE, noise_rank_impurity, noise_rank_perm, len(FEATURES)))
''',
        solution='''
# Part c. Permutation importance on held out data.

perm = permutation_importance(rf, X_test, y_test, n_repeats=20, random_state=0)
imp_perm = pd.Series(perm.importances_mean, index=FEATURES).sort_values(ascending=False)
noise_rank_impurity = int(list(imp_impurity.index).index(NOISE_FEATURE) + 1)
noise_rank_perm = int(list(imp_perm.index).index(NOISE_FEATURE) + 1)

print(None if imp_perm is None else imp_perm.round(4).to_string())
print("")
print("%s ranks %s by impurity, %s by permutation (of %d features)"
      % (NOISE_FEATURE, noise_rank_impurity, noise_rank_perm, len(FEATURES)))
''',
    ),
]


def build_expected(ns):
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import r2_score
    Xtr, Xte, ytr, yte = ns["X_train"], ns["X_test"], ns["y_train"], ns["y_test"]
    feats = ns["FEATURES"]
    rf = RandomForestRegressor(n_estimators=300, random_state=0).fit(Xtr, ytr)
    r2te = float(r2_score(yte, rf.predict(Xte)))
    r2tr = float(r2_score(ytr, rf.predict(Xtr)))
    imp = pd.Series(rf.feature_importances_, index=feats).sort_values(ascending=False)
    perm = permutation_importance(rf, Xte, yte, n_repeats=20, random_state=0)
    pi = pd.Series(perm.importances_mean, index=feats).sort_values(ascending=False)
    perm_train = permutation_importance(rf, Xtr, ytr, n_repeats=20, random_state=0)
    pit = pd.Series(perm_train.importances_mean, index=feats).sort_values(ascending=False)
    noise = ns["NOISE_FEATURE"]
    return {
        "features": [str(f) for f in feats],
        "noise_feature": str(noise),
        "r2_test": r2te, "r2_train": r2tr, "gap": r2tr - r2te,
        "imp_order": [str(i) for i in imp.index],
        "imp_values": [float(v) for v in imp],
        "perm_order": [str(i) for i in pi.index],
        "perm_values": [float(v) for v in pi],
        "perm_train_values": [float(v) for v in pit],
        "perm_train_order": [str(i) for i in pit.index],
        "noise_rank_impurity": int(list(imp.index).index(noise) + 1),
        "noise_rank_perm": int(list(pi.index).index(noise) + 1),
    }


CHECK_CODE = '''
def _part_a():
    rf = _need("rf")
    r2_test_rf = _need("r2_test_rf")
    r2_train_rf = _need("r2_train_rf")
    gap_rf = _need("gap_rf")

    if not hasattr(rf, "feature_importances_"):
        return ("`rf` should be a fitted RandomForestRegressor. Call .fit(X_train, "
                "y_train), which returns the fitted model.")
    if getattr(rf, "n_estimators", None) != 300:
        return "`rf` should have n_estimators=300, but it has %s." % getattr(rf, "n_estimators", None)

    got = _as_scalar(r2_test_rf)
    if got is None:
        return "`r2_test_rf` should be a single number."
    if _num_close(got, _EXP["r2_train"], rtol=1e-6):
        return "`r2_test_rf` was scored on the training data."
    if not _num_close(got, _EXP["r2_test"], rtol=1e-3):
        return ("`r2_test_rf` does not match. Check n_estimators=300 and random_state=0, "
                "and that you fitted on the training data only.")

    got = _as_scalar(r2_train_rf)
    if got is None:
        return "`r2_train_rf` should be a single number."
    if not _num_close(got, _EXP["r2_train"], rtol=1e-3):
        return "`r2_train_rf` is not the R2 of the same forest on the training data."

    got = _as_scalar(gap_rf)
    if got is None:
        return "`gap_rf` should be a single number."
    if _num_close(got, -_EXP["gap"], rtol=1e-3):
        return ("`gap_rf` has the sign the wrong way round. It is train minus test, so a "
                "model that flatters itself gives a positive number.")
    if not _num_close(got, _EXP["gap"], rtol=1e-2):
        return "`gap_rf` is not r2_train_rf minus r2_test_rf."
    return None


def _part_b():
    imp = _need("imp_impurity")
    if not isinstance(imp, _pd.Series):
        return ("`imp_impurity` should be a pandas Series indexed by feature name, so the "
                "ranking is readable.")
    if sorted(str(i) for i in imp.index) != sorted(_EXP["features"]):
        return "`imp_impurity` should be indexed by the five feature names."
    vals = _np.asarray(imp.to_numpy(), dtype=float)
    if not _np.all(_np.diff(vals) <= 1e-12):
        return "`imp_impurity` is not sorted descending. Use .sort_values(ascending=False)."
    if not _num_close(float(vals.sum()), 1.0, rtol=1e-6):
        return ("`imp_impurity` does not sum to 1, so these are not feature_importances_. "
                "Impurity importance is a share of the total.")
    if [str(i) for i in imp.index] != _EXP["imp_order"]:
        return ("`imp_impurity` is in the wrong order. Check it came from "
                "rf.feature_importances_ with the index in FEATURES order before sorting.")
    if not _arr_close(vals, _EXP["imp_values"], rtol=1e-3, atol=1e-4):
        return "`imp_impurity` values do not match rf.feature_importances_."
    return None


def _part_c():
    perm = _need("perm")
    imp_perm = _need("imp_perm")
    noise_rank_impurity = _need("noise_rank_impurity")
    noise_rank_perm = _need("noise_rank_perm")

    if not hasattr(perm, "importances_mean"):
        return ("`perm` should be the object permutation_importance returns, which has an "
                "importances_mean attribute.")

    if not isinstance(imp_perm, _pd.Series):
        return "`imp_perm` should be a pandas Series indexed by feature name."
    if sorted(str(i) for i in imp_perm.index) != sorted(_EXP["features"]):
        return "`imp_perm` should be indexed by the five feature names."
    vals = _np.asarray(imp_perm.to_numpy(), dtype=float)
    if not _np.all(_np.diff(vals) <= 1e-12):
        return "`imp_perm` is not sorted descending."
    if _num_close(float(vals.sum()), 1.0, rtol=1e-4):
        return ("`imp_perm` sums to 1, which means it is the impurity importance again. "
                "Permutation importances are drops in R2 and do not sum to 1.")
    if _arr_close(vals, _EXP["perm_train_values"], rtol=1e-2, atol=1e-3) and \\
            not _arr_close(vals, _EXP["perm_values"], rtol=1e-2, atol=1e-3):
        return ("`imp_perm` was computed on the TRAINING data. Permutation importance has "
                "to be measured on held out data, or it rewards memorisation.")
    if not _arr_close(vals, _EXP["perm_values"], rtol=5e-2, atol=5e-3):
        return ("`imp_perm` does not match permutation_importance on the test set with "
                "n_repeats=20 and random_state=0.")

    got = _as_scalar(noise_rank_impurity)
    if got is None:
        return "`noise_rank_impurity` should be a single whole number."
    if int(round(got)) == _EXP["noise_rank_impurity"] - 1:
        return ("`noise_rank_impurity` looks like a 0 based position. The task asks for a "
                "rank counting from 1, so add 1.")
    if int(round(got)) != _EXP["noise_rank_impurity"]:
        return "`noise_rank_impurity` is not the position of %s in imp_impurity." % _EXP["noise_feature"]

    got = _as_scalar(noise_rank_perm)
    if got is None:
        return "`noise_rank_perm` should be a single whole number."
    if int(round(got)) != _EXP["noise_rank_perm"]:
        return "`noise_rank_perm` is not the position of %s in imp_perm." % _EXP["noise_feature"]
    return None
'''

# Functions this problem introduces. Keys must exist in tools/function_catalog.py.
FUNCTIONS = [
    "RandomForestRegressor",
    "feature_importances_",
    "permutation_importance",
]
