PID = "W5D4"
WEEK = 5
DAY = 4
SLUG = "w5d4_sklearn_elastic_net"
TITLE = "Elastic net: why you must standardise, and what the penalty buys"
TOPIC = "scikit-learn: ElasticNet, standardisation, regularisation paths, ElasticNetCV"
EST = "20-30 min"

SCENARIO_MD = '''
## Scenario

A covariate screen with 24 candidate predictors of clearance across 120
subjects. Four of them genuinely matter. The rest are a mixture of noise and
near-duplicates of the real ones, which is what a real covariate table looks
like: weight, lean body mass, BSA and BMI are four columns carrying roughly
one piece of information.

Ordinary least squares handles this badly. With 24 columns and 120 subjects it
has enough freedom to chase noise, and with several nearly collinear columns
its coefficients become large, unstable and opposite in sign to each other.

Regularisation fixes both by charging for coefficient size. The catch, and the
reason this problem exists, is that the charge is levied on the coefficients
themselves, which means it depends entirely on what units your features happen
to be in.
'''

CONCEPT_MD = '''
## The penalty is in coefficient units, which is why scale matters

**What elastic net minimises.** Ordinary least squares minimises the squared
error alone. Elastic net minimises the squared error **plus** a penalty on the
coefficients:

- an L1 part, the sum of absolute values, which drives coefficients exactly to
  zero and so performs selection,
- an L2 part, the sum of squares, which shrinks everything smoothly and shares
  credit between correlated features instead of picking one arbitrarily.

`alpha` sets the overall strength and `l1_ratio` the mix, with 1.0 being pure
L1, or lasso, and 0.0 pure L2, or ridge. A mix around 0.5 gets the selection
from L1 and the stability under collinearity from L2.

**Why standardisation is not optional here.** The penalty is applied to the
coefficients. A feature measured in grams has a coefficient a thousand times
smaller than the same feature measured in kilograms, for exactly the same
underlying effect. So the gram version is charged a thousandth as much, and
survives the penalty while the kilogram version gets shrunk away.

Put plainly: **without standardisation, which features the model keeps depends
on the units somebody chose to record them in.** That is not a modelling
decision, it is an accident, and it is invisible in the output.

Standardising puts every feature on the same footing, so a unit of penalty
means the same thing everywhere and the coefficients become comparable to each
other as effect sizes. This is the one place where "scale your features" stops
being advice and becomes a correctness requirement.

And because the scaler learns a mean and an SD, it belongs **inside** the
pipeline, for the reasons day 2 covered.

**Reading a regularisation path.** Sweep `alpha` from small to large and watch
two things: the number of surviving non-zero coefficients, which falls, and
the test score, which typically rises then falls. Too little penalty and you
are back to overfitting; too much and you have shrunk real effects away. The
useful alpha is somewhere in between, and the only honest way to find it is
cross-validation.

`ElasticNetCV` does that sweep internally, cross-validating each alpha and
keeping the best. Note what it means for your test set: the alpha was chosen
using the training data only, so the test set stays clean.
'''

TASK_MD = '''
## Your task

Three parts, worked one at a time.

| Part | What you work out | Variables to fill in |
|---|---|---|
| a | the same model with and without standardising | `coef_raw`, `coef_scaled`, `n_zero_raw`, `n_zero_scaled` |
| b | what happens as the penalty grows | `path` |
| c | letting cross-validation choose alpha | `encv`, `best_alpha`, `n_nonzero_cv`, `r2_cv_test` |

Below this you will find the **Setup** cell, then a **Toolkit** cell that
demonstrates every move on a tiny made up example, then **Part a**, **Part b**
and **Part c**, each with its own explanation and a cell to write in.
'''

DATA_CODE = '''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

PROBLEM_ID = "W5D4"
TOPIC = "scikit-learn: ElasticNet, standardisation, regularisation paths, ElasticNetCV"

rng = np.random.default_rng(5104)

N = 120
L1_RATIO = 0.5
ALPHAS = [0.001, 0.01, 0.05, 0.1, 0.3, 1.0, 3.0]

_weight_kg = np.clip(rng.normal(74.0, 13.0, N), 45.0, 125.0)
_height_m = np.clip(rng.normal(1.70, 0.09, N), 1.45, 1.98)
_crcl = np.clip(rng.normal(104.0, 24.0, N), 35.0, 180.0)
_age = np.clip(rng.normal(42.0, 13.0, N), 18.0, 75.0)

cols = {
    "WEIGHT_G": _weight_kg * 1000.0,                 # grams, so a huge scale
    "HEIGHT_M": _height_m,                           # metres, a tiny scale
    "BMI": _weight_kg / _height_m ** 2,
    "BSA_M2": 0.007184 * (_weight_kg ** 0.425) * ((_height_m * 100) ** 0.725),
    "LBM_KG": 0.85 * _weight_kg - 4.0 + rng.normal(0, 1.5, N),
    "CRCL_ML_MIN": _crcl,
    "EGFR": _crcl * 0.95 + rng.normal(0, 6.0, N),
    "AGE_Y": _age,
}
for _i in range(1, 17):
    cols["NOISE%02d" % _i] = rng.normal(0.0, 1.0, N)

X = pd.DataFrame(cols)
FEATURES = list(X.columns)

# Truth: clearance depends on weight, CrCl and age. Nothing else.
_cl = (4.1 * (_weight_kg / 70.0) ** 0.75
       * (_crcl / 100.0) ** 0.40
       * np.exp(-0.0045 * (_age - 42.0)))
y = pd.Series(np.round(_cl * rng.lognormal(0.0, 0.15, N), 3), name="CL_L_H")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)

print("X:", X.shape, " features:", len(FEATURES))
print("")
print("look at the scales these features are on:")
print(X[["WEIGHT_G", "HEIGHT_M", "BMI", "CRCL_ML_MIN", "NOISE01"]].describe()
      .loc[["mean", "std"]].round(3).to_string())
print("")
print("WEIGHT_G and HEIGHT_M carry comparable information about body size,")
print("but their numbers differ by a factor of about forty thousand")
'''

TOOLKIT_MD = '''
## Toolkit for this problem

| What you want to do | How |
|---|---|
| fit an elastic net | `ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=0).fit(X, y)` |
| standardise inside a model | `make_pipeline(StandardScaler(), ElasticNet(...))` |
| reach a step of a pipeline | `pipe[-1]` for the last step, or `pipe.named_steps` |
| the coefficients | `model.coef_`, in column order |
| count the ones driven to zero | `int((model.coef_ == 0).sum())` |
| let CV pick alpha | `ElasticNetCV(l1_ratio=0.5, cv=5, random_state=0)` |
| the alpha it chose | `model.alpha_` |

`make_pipeline` is `Pipeline` without having to name the steps. The names are
generated from the class, so the scaler is `standardscaler` and the model is
`elasticnet`, and `pipe[-1]` gets the last step regardless.
'''

TOOLKIT_CODE = '''
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# A tiny made up example. The SAME feature, recorded in two units.
demo_rng = np.random.default_rng(0)
demo_n = 80
demo_kg = demo_rng.normal(70, 12, demo_n)
demo_X = pd.DataFrame({
    "mass_g": demo_kg * 1000.0,        # same information
    "mass_kg": demo_kg,                # different units
    "other": demo_rng.normal(size=demo_n),
})
demo_y = 0.05 * demo_kg + demo_rng.normal(0, 0.4, demo_n)

print("--- 1. unscaled, the penalty punishes the big numbered feature ---")
demo_raw = ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=0).fit(demo_X, demo_y)
print(pd.Series(demo_raw.coef_, index=demo_X.columns).round(6).to_string())
print("")
print("mass_g and mass_kg carry identical information, yet they get wildly")
print("different coefficients, purely because of the units they were recorded in")

print("")
print("--- 2. standardised, they are treated alike ---")
demo_pipe = make_pipeline(StandardScaler(),
                          ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=0))
demo_pipe.fit(demo_X, demo_y)
print(pd.Series(demo_pipe[-1].coef_, index=demo_X.columns).round(6).to_string())
print("")
print("now the two versions of the same measurement get comparable treatment")

print("")
print("--- 3. what alpha does ---")
for demo_a in [0.001, 0.05, 0.5, 5.0]:
    demo_m = make_pipeline(StandardScaler(),
                           ElasticNet(alpha=demo_a, l1_ratio=0.5, random_state=0))
    demo_m.fit(demo_X, demo_y)
    demo_c = demo_m[-1].coef_
    print("alpha %7.3f  non-zero %d of %d  coefficients %s"
          % (demo_a, int((demo_c != 0).sum()), len(demo_c), np.round(demo_c, 4)))
print("")
print("as alpha rises, coefficients shrink and then hit exactly zero")

print("")
print("--- 4. reaching inside a pipeline ---")
print("steps        :", list(demo_pipe.named_steps.keys()))
print("pipe[-1] is  :", type(demo_pipe[-1]).__name__)
print("its coef_ is what you want, not the pipeline's, which has none")
'''

PARTS = [
    dict(
        label="a",
        prompt_md='''
## Part a. The same model, with and without standardising

Fit two elastic nets on the training data, both with `alpha=0.05`,
`l1_ratio=L1_RATIO` and `random_state=0`.

The first goes straight onto the raw features. The second goes inside
`make_pipeline(StandardScaler(), ElasticNet(...))`, so the features are
standardised first. Pull the coefficients out of each, remembering that a
pipeline has no `coef_` of its own, so you need its last step.

Then count how many coefficients each drove to exactly zero.

Then look at which features each fit keeps, not just how many.

The unscaled fit zeroes `LBM_KG` and `BSA_M2` outright, while ten of the
sixteen pure noise columns survive it. Lean body mass and body surface area
are genuinely informative about clearance; the noise columns are not. The
standardised fit reverses that, giving `LBM_KG` a substantial coefficient.

The reason is purely arithmetic. `LBM_KG` has a standard deviation of about 11
and the noise columns have one of about 1, so for the same real effect
`LBM_KG` needs a coefficient roughly eleven times smaller. A penalty on
coefficient magnitude therefore charges it almost nothing for being useful and
charges the noise columns almost nothing for being useless, and the comparison
between them becomes meaningless. `WEIGHT_G`, in grams, is the extreme case:
its coefficient has to be around 0.00005 to do anything at all, so the penalty
simply does not see it.

Which features survive is being decided by the units somebody recorded them
in. That is the whole lesson, and it is invisible in the output.

Fill in `coef_raw`, `coef_scaled`, `n_zero_raw`, `n_zero_scaled`.

Moves: `ElasticNet(...).fit(X_train, y_train)`,
`make_pipeline(StandardScaler(), ElasticNet(...))`, `pipe[-1].coef_`,
`(coef == 0).sum()`.
''',
        starter='''
# Part a. Elastic net with and without standardising.

coef_raw = None        # YOUR CODE: coefficients from ElasticNet on the RAW features
coef_scaled = None     # YOUR CODE: coefficients from the same model inside a
                       #            StandardScaler pipeline
n_zero_raw = None      # YOUR CODE: how many raw coefficients are exactly zero
n_zero_scaled = None   # YOUR CODE: how many scaled ones are

print("zeros, raw    :", n_zero_raw, "of", len(FEATURES))
print("zeros, scaled :", n_zero_scaled, "of", len(FEATURES))
if coef_raw is not None:
    print("")
    print(pd.DataFrame({"raw": coef_raw, "scaled": coef_scaled},
                       index=FEATURES).head(8).round(6).to_string())
''',
        solution='''
# Part a. Elastic net with and without standardising.

_raw_model = ElasticNet(alpha=0.05, l1_ratio=L1_RATIO, random_state=0).fit(X_train, y_train)
coef_raw = _raw_model.coef_

_scaled_model = make_pipeline(
    StandardScaler(), ElasticNet(alpha=0.05, l1_ratio=L1_RATIO, random_state=0))
_scaled_model.fit(X_train, y_train)
coef_scaled = _scaled_model[-1].coef_

n_zero_raw = int((coef_raw == 0).sum())
n_zero_scaled = int((coef_scaled == 0).sum())

print("zeros, raw    :", n_zero_raw, "of", len(FEATURES))
print("zeros, scaled :", n_zero_scaled, "of", len(FEATURES))
if coef_raw is not None:
    print("")
    print(pd.DataFrame({"raw": coef_raw, "scaled": coef_scaled},
                       index=FEATURES).head(8).round(6).to_string())
''',
    ),
    dict(
        label="b",
        prompt_md='''
## Part b. Sweep alpha and watch the trade

Build a small table showing what the penalty strength does. For each alpha in
`ALPHAS`, fit a standardised elastic net on the training data and record three
things: how many coefficients survive non-zero, the R2 on the training data,
and the R2 on the test data.

Always use the pipeline version from here on. Having seen what unscaled does,
there is no reason to go back.

The shape to look for is the classic one. At the smallest alpha the model
keeps almost everything and the training R2 is at its highest, which is
overfitting. At the largest alpha almost nothing survives and both scores
collapse, which is a model that has been penalised into predicting the mean.
Test R2 peaks somewhere in the middle, and that peak is the useful setting.

Notice that the training R2 falls monotonically as alpha rises, while the test
R2 does not. If you only ever looked at training performance you would always
choose the smallest alpha, and always be wrong.

Fill in `path`, a DataFrame with one row per alpha, in the order of `ALPHAS`,
index renumbered, with exactly:

| column | meaning |
|---|---|
| `ALPHA` | the penalty strength |
| `N_NONZERO` | coefficients not driven to zero |
| `R2_TRAIN` | R2 on the training data |
| `R2_TEST` | R2 on the test data |

Moves: a `for` loop over `ALPHAS`, `make_pipeline`, `.fit`, `pipe[-1].coef_`,
`r2_score`, `pd.DataFrame(rows)`.
''',
        starter='''
# Part b. The regularisation path.

rows = []
# YOUR CODE: for each alpha in ALPHAS, fit a standardised ElasticNet on the training
#            data and append a dict with ALPHA, N_NONZERO, R2_TRAIN, R2_TEST

path = None   # YOUR CODE: pd.DataFrame(rows)

print(None if path is None else path.round(4).to_string(index=False))
''',
        solution='''
# Part b. The regularisation path.

rows = []
for a in ALPHAS:
    m = make_pipeline(StandardScaler(),
                      ElasticNet(alpha=a, l1_ratio=L1_RATIO, random_state=0))
    m.fit(X_train, y_train)
    rows.append({
        "ALPHA": float(a),
        "N_NONZERO": int((m[-1].coef_ != 0).sum()),
        "R2_TRAIN": float(r2_score(y_train, m.predict(X_train))),
        "R2_TEST": float(r2_score(y_test, m.predict(X_test))),
    })

path = pd.DataFrame(rows)

print(None if path is None else path.round(4).to_string(index=False))
''',
    ),
    dict(
        label="c",
        prompt_md='''
## Part c. Let cross-validation choose alpha

`ElasticNetCV` runs the sweep internally, cross-validating each candidate
alpha and keeping the best. Put it inside a `StandardScaler` pipeline as
before, with `l1_ratio=L1_RATIO`, `cv=5` and `random_state=0`, and fit it on
the training data.

Then read off the alpha it chose, how many coefficients survived at that
setting, and the R2 that model gets on the test set.

Note what makes this legitimate. The alpha was chosen by cross-validation
**within the training data**. The test set played no part in the choice, so
scoring on it afterwards is still honest. Had you instead picked the alpha
with the best `R2_TEST` from your part b table, you would have used the test
set to make a modelling decision, and its score would no longer be an
unbiased estimate of anything. That is the same leak as day 2, wearing
different clothes.

Compare the surviving features against the truth: clearance here was built
from weight, creatinine clearance and age, and the body-size columns are
near-duplicates of each other. A sensible fit keeps a couple of the body-size
family, keeps a renal measure, and discards most of the sixteen noise columns.
Do not expect it to pick exactly the right four. With collinear predictors no
method can, and that is a property of the data rather than a failure of the
model.

Fill in `encv`, `best_alpha`, `n_nonzero_cv`, `r2_cv_test`.

Moves: `make_pipeline(StandardScaler(), ElasticNetCV(l1_ratio=..., cv=5,
random_state=0))`, `.fit`, `pipe[-1].alpha_`, `pipe[-1].coef_`, `r2_score`.
''',
        starter='''
# Part c. ElasticNetCV picks alpha using the training data only.

encv = None           # YOUR CODE: StandardScaler + ElasticNetCV pipeline, fitted on train
best_alpha = None     # YOUR CODE: the alpha it chose
n_nonzero_cv = None   # YOUR CODE: coefficients surviving at that alpha
r2_cv_test = None     # YOUR CODE: R2 on the test set

print("alpha chosen   :", best_alpha)
print("features kept  :", n_nonzero_cv, "of", len(FEATURES))
print("R2 on test     :", r2_cv_test)
if encv is not None:
    kept = pd.Series(encv[-1].coef_, index=FEATURES)
    print("")
    print("surviving coefficients:")
    print(kept[kept != 0].sort_values(key=abs, ascending=False).round(4).to_string())
''',
        solution='''
# Part c. ElasticNetCV picks alpha using the training data only.

encv = make_pipeline(StandardScaler(),
                     ElasticNetCV(l1_ratio=L1_RATIO, cv=5, random_state=0))
encv.fit(X_train, y_train)

best_alpha = float(encv[-1].alpha_)
n_nonzero_cv = int((encv[-1].coef_ != 0).sum())
r2_cv_test = float(r2_score(y_test, encv.predict(X_test)))

print("alpha chosen   :", best_alpha)
print("features kept  :", n_nonzero_cv, "of", len(FEATURES))
print("R2 on test     :", r2_cv_test)
if encv is not None:
    kept = pd.Series(encv[-1].coef_, index=FEATURES)
    print("")
    print("surviving coefficients:")
    print(kept[kept != 0].sort_values(key=abs, ascending=False).round(4).to_string())
''',
    ),
]


def build_expected(ns):
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import ElasticNet, ElasticNetCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import r2_score
    Xtr, Xte, ytr, yte = ns["X_train"], ns["X_test"], ns["y_train"], ns["y_test"]
    lr = ns["L1_RATIO"]
    raw = ElasticNet(alpha=0.05, l1_ratio=lr, random_state=0).fit(Xtr, ytr)
    sc = make_pipeline(StandardScaler(), ElasticNet(alpha=0.05, l1_ratio=lr, random_state=0))
    sc.fit(Xtr, ytr)
    rows = []
    for a in ns["ALPHAS"]:
        m = make_pipeline(StandardScaler(), ElasticNet(alpha=a, l1_ratio=lr, random_state=0))
        m.fit(Xtr, ytr)
        rows.append({"ALPHA": float(a), "N_NONZERO": int((m[-1].coef_ != 0).sum()),
                     "R2_TRAIN": float(r2_score(ytr, m.predict(Xtr))),
                     "R2_TEST": float(r2_score(yte, m.predict(Xte)))})
    path = pd.DataFrame(rows)
    cv = make_pipeline(StandardScaler(), ElasticNetCV(l1_ratio=lr, cv=5, random_state=0))
    cv.fit(Xtr, ytr)
    return {
        "n_features": int(Xtr.shape[1]),
        "coef_raw": [float(v) for v in raw.coef_],
        "coef_scaled": [float(v) for v in sc[-1].coef_],
        "n_zero_raw": int((raw.coef_ == 0).sum()),
        "n_zero_scaled": int((sc[-1].coef_ == 0).sum()),
        "alphas": [float(a) for a in ns["ALPHAS"]],
        "path_cols": ["ALPHA", "N_NONZERO", "R2_TRAIN", "R2_TEST"],
        "path_nonzero": [int(v) for v in path["N_NONZERO"]],
        "path_r2_train": [float(v) for v in path["R2_TRAIN"]],
        "path_r2_test": [float(v) for v in path["R2_TEST"]],
        "best_alpha": float(cv[-1].alpha_),
        "n_nonzero_cv": int((cv[-1].coef_ != 0).sum()),
        "r2_cv_test": float(r2_score(yte, cv.predict(Xte))),
    }


CHECK_CODE = '''
def _part_a():
    coef_raw = _need("coef_raw")
    coef_scaled = _need("coef_scaled")
    n_zero_raw = _need("n_zero_raw")
    n_zero_scaled = _need("n_zero_scaled")

    arr = _as_array(coef_raw)
    if arr is None or arr.shape != (_EXP["n_features"],):
        return ("`coef_raw` should hold one coefficient per feature, so shape (%d,). If you "
                "got an AttributeError, remember a pipeline has no coef_ of its own."
                % _EXP["n_features"])
    if _arr_close(arr, _EXP["coef_scaled"], rtol=1e-4, atol=1e-9):
        return ("`coef_raw` matches the standardised fit, so both models were scaled. Part "
                "a wants one fitted straight onto the raw features.")
    if not _arr_close(arr, _EXP["coef_raw"], rtol=1e-3, atol=1e-9):
        return ("`coef_raw` does not match ElasticNet(alpha=0.05, l1_ratio=L1_RATIO, "
                "random_state=0) fitted on the raw X_train.")

    arr = _as_array(coef_scaled)
    if arr is None or arr.shape != (_EXP["n_features"],):
        return "`coef_scaled` should hold one coefficient per feature."
    if _arr_close(arr, _EXP["coef_raw"], rtol=1e-4, atol=1e-9):
        return ("`coef_scaled` matches the raw fit, so the StandardScaler had no effect. "
                "Check you took coefficients from the pipeline's model step, pipe[-1].")
    if not _arr_close(arr, _EXP["coef_scaled"], rtol=1e-3, atol=1e-9):
        return ("`coef_scaled` does not match the StandardScaler plus ElasticNet pipeline.")

    got = _as_scalar(n_zero_raw)
    if got is None or int(round(got)) != _EXP["n_zero_raw"]:
        return "`n_zero_raw` is not the count of exactly zero coefficients in coef_raw."
    got = _as_scalar(n_zero_scaled)
    if got is None or int(round(got)) != _EXP["n_zero_scaled"]:
        return "`n_zero_scaled` is not the count of exactly zero coefficients in coef_scaled."
    return None


def _part_b():
    path = _need("path")
    if not isinstance(path, _pd.DataFrame):
        return "`path` should be a pandas DataFrame."
    cols = _cols_of(path) or []
    missing = [c for c in _EXP["path_cols"] if c not in cols]
    if missing:
        return "`path` is missing the column(s) %s." % ", ".join(missing)
    if len(path) != len(_EXP["alphas"]):
        return "`path` should have one row per alpha in ALPHAS, so %d." % len(_EXP["alphas"])
    if not _arr_close(path["ALPHA"].to_numpy(), _EXP["alphas"]):
        return "`path` does not hold the alphas from ALPHAS, in order."
    if not _arr_close(path["N_NONZERO"].to_numpy(), _EXP["path_nonzero"]):
        return ("`N_NONZERO` does not match. It is the count of coefficients that are NOT "
                "zero, taken from the pipeline's model step, and the model must be the "
                "standardised one.")
    if not _arr_close(path["R2_TRAIN"].to_numpy(), _EXP["path_r2_train"], rtol=1e-3, atol=1e-6):
        return "`R2_TRAIN` does not match scoring each fitted pipeline on the training data."
    if _arr_close(path["R2_TEST"].to_numpy(), _EXP["path_r2_train"], rtol=1e-6):
        return "`R2_TEST` holds the training scores. Score against X_test and y_test."
    if not _arr_close(path["R2_TEST"].to_numpy(), _EXP["path_r2_test"], rtol=1e-3, atol=1e-6):
        return "`R2_TEST` does not match scoring each fitted pipeline on the test data."
    return None


def _part_c():
    encv = _need("encv")
    best_alpha = _need("best_alpha")
    n_nonzero_cv = _need("n_nonzero_cv")
    r2_cv_test = _need("r2_cv_test")

    if not hasattr(encv, "predict"):
        return "`encv` should be a fitted pipeline."
    last = encv[-1] if hasattr(encv, "__getitem__") else encv
    if not hasattr(last, "alpha_"):
        return ("`encv`'s model step has no alpha_, so it is an ElasticNet rather than an "
                "ElasticNetCV. Only the CV version chooses alpha for you.")

    got = _as_scalar(best_alpha)
    if got is None:
        return "`best_alpha` should be a single number."
    if not _num_close(got, _EXP["best_alpha"], rtol=1e-3):
        return ("`best_alpha` is not the alpha ElasticNetCV chose. Check l1_ratio=L1_RATIO, "
                "cv=5 and random_state=0, and that it was fitted on the training data.")

    got = _as_scalar(n_nonzero_cv)
    if got is None:
        return "`n_nonzero_cv` should be a single whole number."
    if int(round(got)) == _EXP["n_features"] - _EXP["n_nonzero_cv"]:
        return ("`n_nonzero_cv` counted the coefficients that ARE zero. The task asks how "
                "many survived.")
    if int(round(got)) != _EXP["n_nonzero_cv"]:
        return "`n_nonzero_cv` is not the count of non-zero coefficients at the chosen alpha."

    got = _as_scalar(r2_cv_test)
    if got is None:
        return "`r2_cv_test` should be a single number."
    if not _num_close(got, _EXP["r2_cv_test"], rtol=1e-3):
        return "`r2_cv_test` is not the R2 of the fitted ElasticNetCV pipeline on the test set."
    return None
'''

# Functions this problem introduces. Keys must exist in tools/function_catalog.py.
FUNCTIONS = [
    "ElasticNet",
    "StandardScaler",
    "make_pipeline",
    "ElasticNetCV",
]
