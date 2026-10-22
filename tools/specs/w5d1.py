PID = "W5D1"
WEEK = 5
DAY = 1
SLUG = "w5d1_sklearn_train_test"
TITLE = "Predicting clearance: the train/test split and why it exists"
TOPIC = "scikit-learn: train_test_split, fit and predict, R2 and RMSE, optimism"
EST = "20-30 min"

SCENARIO_MD = '''
## Scenario

Two hundred subjects from a phase 1 programme, each with an individually
estimated clearance and a handful of covariates: body weight, age, sex,
creatinine clearance and ALT.

The question is the one every population PK analysis eventually asks. How much
of the between-subject variability in clearance can these covariates explain,
and how well could you predict a new subject's clearance before dosing them.

Those two questions are not the same, and the distance between them is the
entire subject of this problem. A model can describe the subjects you have
beautifully and predict the next one badly, and the only way to find out is to
hold data back and test on it.
'''

CONCEPT_MD = '''
## Fit, predict, and the difference between describing and predicting

**The scikit-learn shape.** Every model in the library has the same three
calls. `model.fit(X, y)` learns, `model.predict(X)` produces predictions, and
`model.score(X, y)` gives a default quality measure. `X` is always a 2D array
or DataFrame of shape `(n_samples, n_features)` and `y` is 1D with one entry
per sample. Once you know this shape, swapping a linear regression for a
random forest is a one line change, which is a large part of why the library
is used so widely.

**Why you hold data back.** A model fitted on some data will always describe
that data better than it describes new data, because it has been tuned,
partly, to that data's noise. The more flexible the model and the more
features it has, the more noise it can absorb and the larger that gap grows.
Evaluated on the data it learned from, every model looks better than it is.

So you split once, at the start, before looking at anything. The model sees
only the training part. The test part is spent exactly once, at the end, to
get an honest number. Looking at test performance and then adjusting the model
quietly turns the test set into training data, and the honest number stops
being honest.

**train_test_split, and random_state.** It shuffles and splits in one call.
`test_size=0.25` holds back a quarter. `random_state` fixes the shuffle so the
split is reproducible, which matters because without it every run gives a
different answer and you cannot tell a real improvement from a different
shuffle.

**R2 and RMSE say different things.** R2 is the fraction of variance
explained: 1 is perfect, 0 is no better than always predicting the mean, and
it can go **negative** on held-out data, which simply means your model is
worse than the mean. It is unitless, so it compares across problems. RMSE is
the typical size of an error in the units of the thing you are predicting, so
for clearance it is in L/h and you can tell your pharmacologist whether that
is tolerable. Report both: R2 for whether the model is doing anything, RMSE
for whether it is useful.

**What a gap between train and test tells you.** A large gap means the model
has fitted noise. A small one means it generalises.

With a linear model, five covariates and two hundred subjects, expect the gap
to be very small, and do not be surprised if it comes out slightly
**negative**, with the test score higher than the training score. That is not
a bug and it does not mean the model predicts new data better than old. It
means the model is far too rigid to memorise anything, so the gap is near
zero, and a fifty subject test set is small enough that chance decides which
side of zero it lands on.

The gap only becomes reliably positive, and large, when the model is flexible
enough to absorb noise. You will see exactly that on day 3 with a random
forest. Holding both pictures in mind is the point: the split is not a ritual,
it measures something that varies with how much freedom you gave the model.
'''

TASK_MD = '''
## Your task

Three parts, worked one at a time.

| Part | What you work out | Variables to fill in |
|---|---|---|
| a | the split, held back before anything else | `X_train`, `X_test`, `y_train`, `y_test`, `n_train`, `n_test` |
| b | the fitted model and its honest performance | `model`, `pred_test`, `r2_test`, `rmse_test` |
| c | how much better it looks on its own training data | `r2_train`, `rmse_train`, `optimism` |

Below this you will find the **Setup** cell, then a **Toolkit** cell that
demonstrates every move on a tiny made up example, then **Part a**, **Part b**
and **Part c**, each with its own explanation and a cell to write in.
'''

DATA_CODE = '''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

PROBLEM_ID = "W5D1"
TOPIC = "scikit-learn: train_test_split, fit and predict, R2 and RMSE, optimism"

rng = np.random.default_rng(5101)

N = 200
FEATURES = ["WEIGHT_KG", "AGE_Y", "SEX_M", "CRCL_ML_MIN", "ALT_U_L"]

_weight = np.round(np.clip(rng.normal(74.0, 13.0, N), 45.0, 125.0), 1)
_age = np.round(np.clip(rng.normal(42.0, 13.0, N), 18.0, 75.0), 0)
_sex = rng.integers(0, 2, N)
_crcl = np.round(np.clip(rng.normal(104.0, 24.0, N) - 0.45 * (_age - 42.0), 35.0, 180.0), 1)
_alt = np.round(np.clip(rng.lognormal(np.log(24.0), 0.42, N), 5.0, 160.0), 1)

# Truth: allometric on weight, renal on CrCl, a mild age effect, nothing on ALT.
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

print("subjects:", subjects.shape)
print(subjects.head().to_string(index=False))
print("")
print("X:", X.shape, "  y:", y.shape)
print("clearance ranges %.2f to %.2f L/h, mean %.2f" % (y.min(), y.max(), y.mean()))
'''

TOOLKIT_MD = '''
## Toolkit for this problem

| What you want to do | How |
|---|---|
| split off a test set | `train_test_split(X, y, test_size=0.25, random_state=42)` |
| fit a model | `model = LinearRegression().fit(X_train, y_train)` |
| predict | `model.predict(X_test)` |
| fraction of variance explained | `r2_score(y_true, y_pred)` |
| typical error, in real units | `np.sqrt(mean_squared_error(y_true, y_pred))` |
| the fitted coefficients | `model.coef_`, `model.intercept_` |

`train_test_split` returns **four** things in this order: X train, X test, y
train, y test. Getting that order wrong gives shapes that look plausible and
results that are nonsense.

Compute RMSE as `np.sqrt(mean_squared_error(...))` rather than with a
`squared=False` argument, which exists only in some versions.
'''

TOOLKIT_CODE = '''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# A tiny made up dataset. Nothing here is the answer.
demo_rng = np.random.default_rng(0)
demo_X = pd.DataFrame({"a": demo_rng.normal(size=40), "b": demo_rng.normal(size=40)})
demo_y = 3.0 * demo_X["a"] - 1.5 * demo_X["b"] + demo_rng.normal(0, 1.0, 40)

print("--- 1. the split returns FOUR things, in this order ---")
demo_Xtr, demo_Xte, demo_ytr, demo_yte = train_test_split(
    demo_X, demo_y, test_size=0.25, random_state=42)
print("X_train", demo_Xtr.shape, " X_test", demo_Xte.shape)
print("y_train", demo_ytr.shape, " y_test", demo_yte.shape)
print("rows add back up:", len(demo_Xtr) + len(demo_Xte), "of", len(demo_X))

print("")
print("--- 2. random_state makes the split reproducible ---")
demo_a = train_test_split(demo_X, demo_y, test_size=0.25, random_state=42)[1].index.tolist()
demo_b = train_test_split(demo_X, demo_y, test_size=0.25, random_state=42)[1].index.tolist()
demo_c = train_test_split(demo_X, demo_y, test_size=0.25, random_state=7)[1].index.tolist()
print("same seed, same test rows:", demo_a == demo_b)
print("different seed, different:", demo_a != demo_c)

print("")
print("--- 3. fit on train only, then predict ---")
demo_model = LinearRegression().fit(demo_Xtr, demo_ytr)
print("coefficients:", np.round(demo_model.coef_, 3), " intercept %.3f" % demo_model.intercept_)
demo_pred = demo_model.predict(demo_Xte)
print("first three predictions:", np.round(demo_pred[:3], 3))
print("first three actuals    :", np.round(demo_yte.to_numpy()[:3], 3))

print("")
print("--- 4. two scores that say different things ---")
demo_r2 = r2_score(demo_yte, demo_pred)
demo_rmse = float(np.sqrt(mean_squared_error(demo_yte, demo_pred)))
print("R2   : %.4f   <- fraction of variance explained, unitless" % demo_r2)
print("RMSE : %.4f   <- typical error, in the units of y" % demo_rmse)

print("")
print("--- 5. the same model always looks better on its own training data ---")
demo_r2_train = r2_score(demo_ytr, demo_model.predict(demo_Xtr))
print("R2 on train: %.4f" % demo_r2_train)
print("R2 on test : %.4f" % demo_r2)
print("gap        : %.4f  <- optimism, and it grows with model flexibility"
      % (demo_r2_train - demo_r2))
'''

PARTS = [
    dict(
        label="a",
        prompt_md='''
## Part a. Hold out a test set before you look at anything

Split `X` and `y` into a training set and a test set, holding back 25 percent
for testing, with `random_state=42` so the split is reproducible.

`train_test_split` returns **four** things and the order is X train, X test, y
train, y test. Unpack them in that order. Swapping the middle two is a classic
and produces shapes that look fine while the model learns from the wrong
rows.

Then record how many subjects ended up in each part. They should add back up
to 200, and the test set should hold 50.

The `random_state` is not decoration. Without it, every run reshuffles and
your R2 wobbles by a few hundredths, which is more than enough to make you
believe a change helped when it did not.

Fill in `X_train`, `X_test`, `y_train`, `y_test`, `n_train`, `n_test`.

Moves: `train_test_split(X, y, test_size=0.25, random_state=42)`, `len()`.
''',
        starter='''
# Part a. Split off a test set, reproducibly.

X_train = None   # YOUR CODE
X_test = None    # YOUR CODE
y_train = None   # YOUR CODE
y_test = None    # YOUR CODE
n_train = None   # YOUR CODE: how many subjects train on
n_test = None    # YOUR CODE: how many are held back

print("n_train:", n_train, "  n_test:", n_test)
print("X_train shape:", None if X_train is None else X_train.shape)
print("X_test shape :", None if X_test is None else X_test.shape)
''',
        solution='''
# Part a. Split off a test set, reproducibly.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)
n_train = int(len(X_train))
n_test = int(len(X_test))

print("n_train:", n_train, "  n_test:", n_test)
print("X_train shape:", None if X_train is None else X_train.shape)
print("X_test shape :", None if X_test is None else X_test.shape)
''',
    ),
    dict(
        label="b",
        prompt_md='''
## Part b. Fit, predict, and score honestly

Fit a `LinearRegression` on the **training** data only. The test set does not
exist as far as the model is concerned.

Then predict on the test set and score those predictions two ways. `r2_score`
gives the fraction of variance explained, unitless, where 0 means no better
than always guessing the mean. `np.sqrt(mean_squared_error(...))` gives the
typical error in L/h, which is the number to quote to a pharmacologist.

Compute RMSE that way rather than with a `squared=False` argument. That
argument was removed from recent scikit-learn in favour of a separate
function, so the square root of the mean squared error is the form that works
everywhere.

Have a look at `model.coef_` as well. With the features on wildly different
scales, weight in kilograms against sex as 0 or 1, the coefficients are not
comparable to each other and the largest one is not the most important
feature. Week 5 day 4 deals with that properly.

Fill in `model`, `pred_test`, `r2_test`, `rmse_test`.

Moves: `LinearRegression().fit(X_train, y_train)`, `.predict(X_test)`,
`r2_score`, `np.sqrt(mean_squared_error(...))`.
''',
        starter='''
# Part b. Fit on train, score on test.

model = None       # YOUR CODE: a LinearRegression fitted on the TRAINING data
pred_test = None   # YOUR CODE: its predictions for the test set
r2_test = None     # YOUR CODE: R2 on the test set
rmse_test = None   # YOUR CODE: RMSE on the test set, in L/h

print("R2 on test   : %s" % r2_test)
print("RMSE on test : %s L/h" % rmse_test)
if model is not None:
    print("coefficients :", dict(zip(FEATURES, np.round(model.coef_, 4))))
''',
        solution='''
# Part b. Fit on train, score on test.

model = LinearRegression().fit(X_train, y_train)
pred_test = model.predict(X_test)
r2_test = float(r2_score(y_test, pred_test))
rmse_test = float(np.sqrt(mean_squared_error(y_test, pred_test)))

print("R2 on test   : %s" % r2_test)
print("RMSE on test : %s L/h" % rmse_test)
if model is not None:
    print("coefficients :", dict(zip(FEATURES, np.round(model.coef_, 4))))
''',
    ),
    dict(
        label="c",
        prompt_md='''
## Part c. Measure the optimism

Score the **same fitted model** on the training data it learned from, and
compare.

Compute `r2_train` and `rmse_train` the same way as part b but against
`X_train` and `y_train`, then set `optimism` to `r2_train - r2_test`.

What you should find is a gap very close to zero, and on this particular split
it actually comes out **negative**: the model scores slightly better on the
fifty subjects it never saw than on the hundred and fifty it learned from.

That is worth sitting with rather than explaining away. A linear model with
five coefficients has nowhere near enough freedom to memorise a hundred and
fifty subjects, so there is almost no optimism to find, and what remains is
swamped by the luck of which fifty subjects landed in the test set. A small
negative gap is exactly what "this model cannot overfit" looks like.

Keep the number, because it is the baseline for the rest of the week. On day 3
a random forest on this same data will show a clearly positive gap, not
because it is a worse model but because it is flexible enough to absorb noise.
The contrast between the two is the thing to remember.

What would be alarming is a **large** gap in either direction. Large and
positive means overfitting. Large and negative usually means the split leaked
or `pred_test` was lined up against the wrong rows.

Fill in `r2_train`, `rmse_train`, `optimism`.

Moves: `model.predict(X_train)`, `r2_score`, `np.sqrt(mean_squared_error(...))`,
subtraction.
''',
        starter='''
# Part c. How much better does it look on its own training data.

r2_train = None     # YOUR CODE: R2 of the SAME model on the training data
rmse_train = None   # YOUR CODE: RMSE on the training data
optimism = None     # YOUR CODE: r2_train - r2_test

print("R2   train %s  vs test %s" % (r2_train, r2_test))
print("RMSE train %s  vs test %s" % (rmse_train, rmse_test))
print("optimism   %s" % optimism)
''',
        solution='''
# Part c. How much better does it look on its own training data.

_pred_train = model.predict(X_train)
r2_train = float(r2_score(y_train, _pred_train))
rmse_train = float(np.sqrt(mean_squared_error(y_train, _pred_train)))
optimism = float(r2_train - r2_test)

print("R2   train %s  vs test %s" % (r2_train, r2_test))
print("RMSE train %s  vs test %s" % (rmse_train, rmse_test))
print("optimism   %s" % optimism)
''',
    ),
]


def build_expected(ns):
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_squared_error
    X, y = ns["X"], ns["y"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
    m = LinearRegression().fit(Xtr, ytr)
    pte, ptr = m.predict(Xte), m.predict(Xtr)
    # the split you get with the default seed, for diagnosing a missing random_state
    Xtr0, Xte0, ytr0, yte0 = train_test_split(X, y, test_size=0.25, random_state=0)
    m0 = LinearRegression().fit(Xtr0, ytr0)
    return {
        "n_total": int(len(X)),
        "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
        "n_features": int(X.shape[1]),
        "test_index": [int(v) for v in Xte.index],
        "coef": [float(v) for v in m.coef_],
        "pred_test": [float(v) for v in pte],
        "r2_test": float(r2_score(yte, pte)),
        "rmse_test": float(np.sqrt(mean_squared_error(yte, pte))),
        "mse_test": float(mean_squared_error(yte, pte)),
        "r2_train": float(r2_score(ytr, ptr)),
        "rmse_train": float(np.sqrt(mean_squared_error(ytr, ptr))),
        "optimism": float(r2_score(ytr, ptr) - r2_score(yte, pte)),
        "r2_test_seed0": float(r2_score(yte0, m0.predict(Xte0))),
    }


CHECK_CODE = '''
def _part_a():
    X_train = _need("X_train")
    X_test = _need("X_test")
    y_train = _need("y_train")
    y_test = _need("y_test")
    n_train = _need("n_train")
    n_test = _need("n_test")

    if _np.ndim(X_train) != 2:
        return ("`X_train` should be 2D, samples by features. If it is 1D you probably "
                "unpacked train_test_split in the wrong order: it returns X train, X test, "
                "y train, y test.")
    if _np.ndim(y_train) != 1:
        return ("`y_train` should be 1D, one clearance per subject. Check the unpacking "
                "order of train_test_split.")
    if _np.shape(X_train)[1] != _EXP["n_features"]:
        return "`X_train` should have %d feature columns." % _EXP["n_features"]
    if len(X_train) + len(X_test) != _EXP["n_total"]:
        return ("`X_train` and `X_test` hold %d rows between them but there are %d "
                "subjects." % (len(X_train) + len(X_test), _EXP["n_total"]))
    if len(X_test) != _EXP["n_test"]:
        return ("`X_test` holds %d rows but test_size=0.25 of %d subjects is %d."
                % (len(X_test), _EXP["n_total"], _EXP["n_test"]))
    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        return "`X` and `y` parts do not line up in length."

    try:
        idx = [int(v) for v in X_test.index]
    except Exception:
        idx = None
    if idx is not None and idx != _EXP["test_index"]:
        return ("`X_test` holds the right number of rows but not the expected ones, so the "
                "shuffle differed. Pass random_state=42.")

    got = _as_scalar(n_train)
    if got is None or int(round(got)) != _EXP["n_train"]:
        return "`n_train` is not the number of rows in X_train."
    got = _as_scalar(n_test)
    if got is None or int(round(got)) != _EXP["n_test"]:
        return "`n_test` is not the number of rows in X_test."
    return None


def _part_b():
    model = _need("model")
    pred_test = _need("pred_test")
    r2_test = _need("r2_test")
    rmse_test = _need("rmse_test")

    if not hasattr(model, "predict"):
        return "`model` should be a fitted scikit-learn model."
    if not hasattr(model, "coef_"):
        return ("`model` has no coef_, so it was never fitted. Call .fit(X_train, y_train), "
                "which returns the fitted model.")
    if not _arr_close(model.coef_, _EXP["coef"], rtol=1e-4):
        return ("`model` coefficients do not match a LinearRegression fitted on X_train and "
                "y_train. Make sure you fitted on the TRAINING data only.")

    arr = _as_array(pred_test)
    if arr is None or arr.shape != (_EXP["n_test"],):
        return ("`pred_test` should hold one prediction per test subject, so shape (%d,)."
                % _EXP["n_test"])
    if not _arr_close(arr, _EXP["pred_test"], rtol=1e-4):
        return "`pred_test` is not model.predict(X_test)."

    got = _as_scalar(r2_test)
    if got is None:
        return "`r2_test` should be a single number."
    if _num_close(got, _EXP["r2_train"], rtol=1e-6):
        return "`r2_test` was scored on the training data. Score against y_test and pred_test."
    if not _num_close(got, _EXP["r2_test"], rtol=1e-4):
        return "`r2_test` is not r2_score(y_test, pred_test)."

    got = _as_scalar(rmse_test)
    if got is None:
        return "`rmse_test` should be a single number."
    if _num_close(got, _EXP["mse_test"], rtol=1e-4):
        return ("`rmse_test` is the mean squared error, not its root. Take np.sqrt of it so "
                "the number is in L/h.")
    if not _num_close(got, _EXP["rmse_test"], rtol=1e-4):
        return "`rmse_test` is not np.sqrt(mean_squared_error(y_test, pred_test))."
    return None


def _part_c():
    r2_train = _need("r2_train")
    rmse_train = _need("rmse_train")
    optimism = _need("optimism")

    got = _as_scalar(r2_train)
    if got is None:
        return "`r2_train` should be a single number."
    if _num_close(got, _EXP["r2_test"], rtol=1e-6):
        return ("`r2_train` equals the test score, so it was scored on the test set again. "
                "Predict on X_train and score against y_train.")
    if not _num_close(got, _EXP["r2_train"], rtol=1e-4):
        return "`r2_train` is not r2_score(y_train, model.predict(X_train))."

    got = _as_scalar(rmse_train)
    if got is None:
        return "`rmse_train` should be a single number."
    if not _num_close(got, _EXP["rmse_train"], rtol=1e-4):
        return "`rmse_train` is not the RMSE on the training data."

    got = _as_scalar(optimism)
    if got is None:
        return "`optimism` should be a single number."
    if _num_close(got, -_EXP["optimism"], rtol=1e-4):
        return ("`optimism` has the sign the wrong way round. It is the training score "
                "minus the test score, so a model that flatters itself gives a positive "
                "number.")
    if not _num_close(got, _EXP["optimism"], rtol=1e-3, atol=1e-9):
        return "`optimism` is not r2_train minus r2_test."
    return None
'''

# Functions this problem introduces. Keys must exist in tools/function_catalog.py.
FUNCTIONS = [
    "train_test_split",
    "fit and predict",
    "r2_score",
    "mean_squared_error",
    "random_state",
]
