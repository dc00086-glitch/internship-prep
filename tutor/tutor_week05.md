# Week 5, scikit-learn

The full text of every problem this week, the data it generates, the
toolkit the student is given, what each part asks for, and what every
possible check failure means. No solutions in this file.

---

# W5D1. Predicting clearance: the train/test split and why it exists

- topic: scikit-learn: train_test_split, fit and predict, R2 and RMSE, optimism
- notebook: problems/week05/w5d1_sklearn_train_test.ipynb
- solution: solutions/week05/w5d1_sklearn_train_test_solution.ipynb
- expected time: 20-30 min

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

## The setup cell the student runs

This is given to them, already working. The data is synthetic and
seeded, so their numbers are always these numbers.

```python
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
```

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

The toolkit cell demonstrates each of those on a tiny made up example.
It is in the notebook and none of it is the answer.

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

**Variables this part is graded on:** `X_train`, `X_test`, `y_train`, `y_test`, `n_train`, `n_test`

The starter cell they see:

```python
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
```

**Every failure message part a can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `X_train` should be 2D, samples by features. If it is 1D you probably unpacked train_test_split in the wrong order: it returns X train, X test, y train, y test.
- `y_train` should be 1D, one clearance per subject. Check the unpacking order of train_test_split.
- `X_train` should have %d feature columns.
- `X_train` and `X_test` hold %d rows between them but there are %d subjects.
- `X_test` holds %d rows but test_size=0.25 of %d subjects is %d.
- `X` and `y` parts do not line up in length.
- `X_test` holds the right number of rows but not the expected ones, so the shuffle differed. Pass random_state=42.
- `n_train` is not the number of rows in X_train.
- `n_test` is not the number of rows in X_test.

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

**Variables this part is graded on:** `model`, `pred_test`, `r2_test`, `rmse_test`

The starter cell they see:

```python
# Part b. Fit on train, score on test.

model = None       # YOUR CODE: a LinearRegression fitted on the TRAINING data
pred_test = None   # YOUR CODE: its predictions for the test set
r2_test = None     # YOUR CODE: R2 on the test set
rmse_test = None   # YOUR CODE: RMSE on the test set, in L/h

print("R2 on test   : %s" % r2_test)
print("RMSE on test : %s L/h" % rmse_test)
if model is not None:
    print("coefficients :", dict(zip(FEATURES, np.round(model.coef_, 4))))
```

**Every failure message part b can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `model` should be a fitted scikit-learn model.
- `model` has no coef_, so it was never fitted. Call .fit(X_train, y_train), which returns the fitted model.
- `model` coefficients do not match a LinearRegression fitted on X_train and y_train. Make sure you fitted on the TRAINING data only.
- `pred_test` should hold one prediction per test subject, so shape (%d,).
- `pred_test` is not model.predict(X_test).
- `r2_test` should be a single number.
- `r2_test` was scored on the training data. Score against y_test and pred_test.
- `r2_test` is not r2_score(y_test, pred_test).
- `rmse_test` should be a single number.
- `rmse_test` is the mean squared error, not its root. Take np.sqrt of it so the number is in L/h.
- `rmse_test` is not np.sqrt(mean_squared_error(y_test, pred_test)).

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

**Variables this part is graded on:** `r2_train`, `rmse_train`, `optimism`

The starter cell they see:

```python
# Part c. How much better does it look on its own training data.

r2_train = None     # YOUR CODE: R2 of the SAME model on the training data
rmse_train = None   # YOUR CODE: RMSE on the training data
optimism = None     # YOUR CODE: r2_train - r2_test

print("R2   train %s  vs test %s" % (r2_train, r2_test))
print("RMSE train %s  vs test %s" % (rmse_train, rmse_test))
print("optimism   %s" % optimism)
```

**Every failure message part c can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `r2_train` should be a single number.
- `r2_train` equals the test score, so it was scored on the test set again. Predict on X_train and score against y_train.
- `r2_train` is not r2_score(y_train, model.predict(X_train)).
- `rmse_train` should be a single number.
- `rmse_train` is not the RMSE on the training data.
- `optimism` should be a single number.
- `optimism` has the sign the wrong way round. It is the training score minus the test score, so a model that flatters itself gives a positive number.
- `optimism` is not r2_train minus r2_test.

**Functions this problem introduces:** train_test_split, fit and predict, r2_score, mean_squared_error, random_state

---

# W5D2. Cross-validation, and the leak that invents a result from noise

- topic: scikit-learn: KFold, cross_val_score, Pipeline, and data leakage
- notebook: problems/week05/w5d2_sklearn_cv_leakage.ipynb
- solution: solutions/week05/w5d2_sklearn_cv_leakage_solution.ipynb
- expected time: 20-30 min

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

## The setup cell the student runs

This is given to them, already working. The data is synthetic and
seeded, so their numbers are always these numbers.

```python
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
```

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

The toolkit cell demonstrates each of those on a tiny made up example.
It is in the notebook and none of it is the answer.

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

**Variables this part is graded on:** `cv_scores`, `cv_mean`, `cv_sd`

The starter cell they see:

```python
# Part a. Cross-validate on the two real covariates.

cv_scores = None   # YOUR CODE: the five fold R2 scores for LinearRegression on X_real
cv_mean = None     # YOUR CODE: their mean
cv_sd = None       # YOUR CODE: their sample SD

print("fold scores:", None if cv_scores is None else np.round(cv_scores, 4))
print("mean R2    :", cv_mean)
print("SD across folds:", cv_sd)
```

**Every failure message part a can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `cv_scores` should be the 1D array cross_val_score returns.
- `cv_scores` has %d entries but KFold with 5 splits gives 5.
- `cv_scores` does not match. Use LinearRegression on X_real, with KFold(n_splits=5, shuffle=True, random_state=1) and scoring='r2'.
- `cv_mean` is not the mean of cv_scores.
- `cv_sd` should be a single number.
- `cv_sd` used ddof=0. numpy and the array returned by cross_val_score both default to the population SD, so pass ddof=1.
- `cv_sd` is not the sample SD of cv_scores.

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

**Variables this part is graded on:** `X_leaky`, `cv_leaky_mean`

The starter cell they see:

```python
# Part b. Screen on all the data first, then cross-validate. The common mistake.

X_leaky = None         # YOUR CODE: the K_SELECT columns of X_panel that score best
                       #            against y, chosen using ALL the data
cv_leaky_mean = None   # YOUR CODE: mean CV R2 of LinearRegression on X_leaky

print("columns chosen:", None if X_leaky is None else list(X_leaky.columns))
print("mean R2, selected first:", cv_leaky_mean)
print("mean R2, real covariates:", cv_mean)
```

**Every failure message part b can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `X_leaky` should be a 2D table of the selected columns.
- `X_leaky` still has all %d analytes, so no selection happened. Use get_support() to pick the K_SELECT best columns.
- `X_leaky` has %d columns but K_SELECT is %d.
- `X_leaky` does not hold the columns SelectKBest picks. Fit it on the whole X_panel and y, which is the mistake this part is demonstrating.
- `cv_leaky_mean` should be a single number.
- `cv_leaky_mean` matches the honest answer, so the selection was not done on all the data first. Part b is deliberately demonstrating the leak.
- `cv_leaky_mean` does not match. Cross-validate a plain LinearRegression on X_leaky with the same KFold.

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

**Variables this part is graded on:** `pipe`, `cv_clean_mean`, `leak_inflation`

The starter cell they see:

```python
# Part c. Selection inside the pipeline, so each fold is blind to its own test data.

pipe = None             # YOUR CODE: Pipeline with steps named "sel" and "model"
cv_clean_mean = None    # YOUR CODE: mean CV R2 of the pipeline on the FULL X_panel
leak_inflation = None   # YOUR CODE: cv_leaky_mean - cv_clean_mean

print("mean R2, selection leaked  : %s" % cv_leaky_mean)
print("mean R2, selection in pipe : %s" % cv_clean_mean)
print("inflation caused by the leak: %s" % leak_inflation)
print("")
print("for reference, the real covariates gave: %s" % cv_mean)
```

**Every failure message part c can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `pipe` should be a sklearn Pipeline.
- `pipe` should have exactly two steps named 'sel' and 'model', in that order. Yours are %s.
- `pipe`'s first step should be the SelectKBest selector.
- `pipe`'s second step should be the LinearRegression.
- `cv_clean_mean` should be a single number.
- `cv_clean_mean` equals the leaked answer, so the pipeline was run on X_leaky. Run it on the FULL X_panel, so the selection happens inside each fold.
- `cv_clean_mean` looks like a plain regression on all 80 analytes with no selection step. Check the pipeline really contains the selector.
- `cv_clean_mean` does not match cross-validating the pipeline on X_panel.
- `leak_inflation` should be a single number.
- `leak_inflation` has the sign the wrong way round. It is the leaked score minus the clean one, so the inflation comes out positive.
- `leak_inflation` is not cv_leaky_mean minus cv_clean_mean.

**Functions this problem introduces:** KFold, cross_val_score, Pipeline, SelectKBest, data leakage

---

# W5D3. Random forests, and two different answers to which feature matters

- topic: scikit-learn: RandomForestRegressor, impurity importance, permutation importance
- notebook: problems/week05/w5d3_sklearn_random_forest.ipynb
- solution: solutions/week05/w5d3_sklearn_random_forest_solution.ipynb
- expected time: 20-30 min

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

## The setup cell the student runs

This is given to them, already working. The data is synthetic and
seeded, so their numbers are always these numbers.

```python
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
```

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

The toolkit cell demonstrates each of those on a tiny made up example.
It is in the notebook and none of it is the answer.

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

**Variables this part is graded on:** `rf`, `r2_test_rf`, `r2_train_rf`, `gap_rf`

The starter cell they see:

```python
# Part a. Fit a random forest and compare train against test.

rf = None            # YOUR CODE: RandomForestRegressor, 300 trees, random_state=0,
                     #            fitted on the training data
r2_test_rf = None    # YOUR CODE: R2 on the test set
r2_train_rf = None   # YOUR CODE: R2 on the training set
gap_rf = None        # YOUR CODE: train minus test

print("R2 test  : %s" % r2_test_rf)
print("R2 train : %s" % r2_train_rf)
print("gap      : %s   (day 1 linear model: -0.046)" % gap_rf)
```

**Every failure message part a can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `rf` should be a fitted RandomForestRegressor. Call .fit(X_train, y_train), which returns the fitted model.
- `rf` should have n_estimators=300, but it has %s. n_estimators
- `r2_test_rf` should be a single number.
- `r2_test_rf` was scored on the training data.
- `r2_test_rf` does not match. Check n_estimators=300 and random_state=0, and that you fitted on the training data only.
- `r2_train_rf` should be a single number.
- `r2_train_rf` is not the R2 of the same forest on the training data.
- `gap_rf` should be a single number.
- `gap_rf` has the sign the wrong way round. It is train minus test, so a model that flatters itself gives a positive number.
- `gap_rf` is not r2_train_rf minus r2_test_rf.

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

**Variables this part is graded on:** `imp_impurity`

The starter cell they see:

```python
# Part b. Impurity importance, sorted.

imp_impurity = None   # YOUR CODE: Series indexed by feature name, sorted descending

print(None if imp_impurity is None else imp_impurity.round(4).to_string())
print("")
print("sums to:", None if imp_impurity is None else round(float(imp_impurity.sum()), 6))
```

**Every failure message part b can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `imp_impurity` should be a pandas Series indexed by feature name, so the ranking is readable.
- `imp_impurity` should be indexed by the five feature names.
- `imp_impurity` is not sorted descending. Use .sort_values(ascending=False).
- `imp_impurity` does not sum to 1, so these are not feature_importances_. Impurity importance is a share of the total.
- `imp_impurity` is in the wrong order. Check it came from rf.feature_importances_ with the index in FEATURES order before sorting.
- `imp_impurity` values do not match rf.feature_importances_.

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

**Variables this part is graded on:** `perm`, `imp_perm`, `noise_rank_impurity`, `noise_rank_perm`

The starter cell they see:

```python
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
```

**Every failure message part c can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `perm` should be the object permutation_importance returns, which has an importances_mean attribute.
- `imp_perm` should be a pandas Series indexed by feature name.
- `imp_perm` should be indexed by the five feature names.
- `imp_perm` is not sorted descending.
- `imp_perm` sums to 1, which means it is the impurity importance again. Permutation importances are drops in R2 and do not sum to 1.
- `imp_perm` was computed on the TRAINING data. Permutation importance has to be measured on held out data, or it rewards memorisation.
- `imp_perm` does not match permutation_importance on the test set with n_repeats=20 and random_state=0.
- `noise_rank_impurity` should be a single whole number.
- `noise_rank_impurity` looks like a 0 based position. The task asks for a rank counting from 1, so add 1.
- `noise_rank_impurity` is not the position of %s in imp_impurity.
- `noise_rank_perm` should be a single whole number.
- `noise_rank_perm` is not the position of %s in imp_perm.

**Functions this problem introduces:** RandomForestRegressor, feature_importances_, permutation_importance

---

# W5D4. Elastic net: why you must standardise, and what the penalty buys

- topic: scikit-learn: ElasticNet, standardisation, regularisation paths, ElasticNetCV
- notebook: problems/week05/w5d4_sklearn_elastic_net.ipynb
- solution: solutions/week05/w5d4_sklearn_elastic_net_solution.ipynb
- expected time: 20-30 min

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

## The setup cell the student runs

This is given to them, already working. The data is synthetic and
seeded, so their numbers are always these numbers.

```python
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
```

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

The toolkit cell demonstrates each of those on a tiny made up example.
It is in the notebook and none of it is the answer.

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

**Variables this part is graded on:** `coef_raw`, `coef_scaled`, `n_zero_raw`, `n_zero_scaled`

The starter cell they see:

```python
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
```

**Every failure message part a can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `coef_raw` should hold one coefficient per feature, so shape (%d,). If you got an AttributeError, remember a pipeline has no coef_ of its own.
- `coef_raw` matches the standardised fit, so both models were scaled. Part a wants one fitted straight onto the raw features.
- `coef_raw` does not match ElasticNet(alpha=0.05, l1_ratio=L1_RATIO, random_state=0) fitted on the raw X_train.
- `coef_scaled` should hold one coefficient per feature.
- `coef_scaled` matches the raw fit, so the StandardScaler had no effect. Check you took coefficients from the pipeline's model step, pipe[-1].
- `coef_scaled` does not match the StandardScaler plus ElasticNet pipeline.
- `n_zero_raw` is not the count of exactly zero coefficients in coef_raw.
- `n_zero_scaled` is not the count of exactly zero coefficients in coef_scaled.

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

**Variables this part is graded on:** `path`

The starter cell they see:

```python
# Part b. The regularisation path.

rows = []
# YOUR CODE: for each alpha in ALPHAS, fit a standardised ElasticNet on the training
#            data and append a dict with ALPHA, N_NONZERO, R2_TRAIN, R2_TEST

path = None   # YOUR CODE: pd.DataFrame(rows)

print(None if path is None else path.round(4).to_string(index=False))
```

**Every failure message part b can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `path` should be a pandas DataFrame.
- `path` is missing the column(s) %s. ,
- `path` should have one row per alpha in ALPHAS, so %d.
- `path` does not hold the alphas from ALPHAS, in order.
- `N_NONZERO` does not match. It is the count of coefficients that are NOT zero, taken from the pipeline's model step, and the model must be the standardised one.
- `R2_TRAIN` does not match scoring each fitted pipeline on the training data.
- `R2_TEST` holds the training scores. Score against X_test and y_test.
- `R2_TEST` does not match scoring each fitted pipeline on the test data.

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

**Variables this part is graded on:** `encv`, `best_alpha`, `n_nonzero_cv`, `r2_cv_test`

The starter cell they see:

```python
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
```

**Every failure message part c can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `encv` should be a fitted pipeline.
- `encv`'s model step has no alpha_, so it is an ElasticNet rather than an ElasticNetCV. Only the CV version chooses alpha for you.
- `best_alpha` should be a single number.
- `best_alpha` is not the alpha ElasticNetCV chose. Check l1_ratio=L1_RATIO, cv=5 and random_state=0, and that it was fitted on the training data.
- `n_nonzero_cv` should be a single whole number.
- `n_nonzero_cv` counted the coefficients that ARE zero. The task asks how many survived.
- `n_nonzero_cv` is not the count of non-zero coefficients at the chosen alpha.
- `r2_cv_test` should be a single number.
- `r2_cv_test` is not the R2 of the fitted ElasticNetCV pipeline on the test set.

**Functions this problem introduces:** ElasticNet, StandardScaler, make_pipeline, ElasticNetCV

---

# W5D5. PCA on a correlated covariate panel

- topic: scikit-learn: PCA, explained variance, loadings, projecting onto components
- notebook: problems/week05/w5d5_sklearn_pca.ipynb
- solution: solutions/week05/w5d5_sklearn_pca_solution.ipynb
- expected time: 20-30 min

## Scenario

Ten laboratory measurements on 180 subjects: liver enzymes, renal markers and
body size measures. They are heavily correlated within those three families,
because they are three underlying things measured several ways each. ALT and
AST move together. Creatinine clearance and eGFR are nearly the same number.
Weight, BMI and body surface area are three views of body size.

Ten columns, but nothing like ten independent pieces of information.

Principal component analysis finds the directions along which the data varies
most, and rewrites each subject in terms of those directions instead of the
original columns. The useful outcome here is not compression for its own sake:
it is finding out **how many** things this panel is really measuring, and
reading off which measurements move together.

## What PCA actually does, and the one thing you must not forget

**The idea.** PCA finds a new set of axes for the data. The first, PC1, points
along the direction of greatest variance. PC2 points along the greatest
remaining variance subject to being perpendicular to PC1, and so on. Each
subject then gets a coordinate on each axis, and because the axes are
perpendicular, those coordinates are uncorrelated with each other.

Nothing is lost if you keep all the components. The value comes from the
variance being concentrated in the first few, so a handful of them carry
almost everything.

**Standardise first, and this one is not negotiable.** PCA maximises variance,
and variance is in squared units of the feature. A measurement in grams has a
variance a million times larger than the same measurement in kilograms, so
PC1 will simply point along whichever column happens to be recorded in the
largest units. Without standardisation PCA is a report on your unit choices.

With standardisation every feature has variance 1, they contribute equally,
and PCA is working on the correlation structure, which is what you wanted.

**explained_variance_ratio_** gives the share of total variance each component
carries, summing to 1 across all of them. Its cumulative sum answers "how many
components do I need for 90 percent", which is the usual way of deciding how
many to keep. With ten standardised features and three underlying factors,
expect the first three components to carry most of it.

**Loadings** are in `components_`, one row per component and one column per
original feature, saying how much each feature contributes to that component.
Reading PC1's loadings is how you find out what PC1 *is*. If the body size
measures all load heavily on it with the same sign, PC1 is body size.

**Two cautions.** First, a sign flip is meaningless: PC1 and its negative are
the same axis, and different library versions or platforms can return either.
Never interpret the sign of a loading on its own, only signs relative to each
other within the same component. Second, components are built to explain
variance in `X`, with no reference to any outcome. The direction of greatest
variance need not be the direction that predicts anything, and a component
carrying 2 percent of the variance can be the one that matters clinically.

## Your task

Three parts, worked one at a time.

| Part | What you work out | Variables to fill in |
|---|---|---|
| a | the components, and how much each carries | `pca`, `evr`, `evr_unscaled_first` |
| b | how many you need, and the projection | `cumulative`, `n_for_90`, `scores` |
| c | what PC1 and PC2 actually are | `loadings`, `pc1_top3`, `pc2_top3` |

Below this you will find the **Setup** cell, then a **Toolkit** cell that
demonstrates every move on a tiny made up example, then **Part a**, **Part b**
and **Part c**, each with its own explanation and a cell to write in.

## The setup cell the student runs

This is given to them, already working. The data is synthetic and
seeded, so their numbers are always these numbers.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROBLEM_ID = "W5D5"
TOPIC = "scikit-learn: PCA, explained variance, loadings, projecting onto components"

rng = np.random.default_rng(5105)

N = 180

# Three underlying factors: body size, renal function, hepatic enzymes.
_size = rng.normal(0.0, 1.0, N)
_renal = rng.normal(0.0, 1.0, N)
_hepatic = rng.normal(0.0, 1.0, N)

_weight_kg = 74.0 + 13.0 * _size + rng.normal(0, 1.5, N)
_height_m = 1.70 + 0.06 * _size + rng.normal(0, 0.03, N)

panel = pd.DataFrame({
    "WEIGHT_KG": np.round(_weight_kg, 1),
    "BMI": np.round(_weight_kg / _height_m ** 2, 2),
    "BSA_M2": np.round(0.007184 * _weight_kg ** 0.425 * (_height_m * 100) ** 0.725, 3),
    "LBM_KG": np.round(0.85 * _weight_kg - 4.0 + rng.normal(0, 1.2, N), 1),
    "CRCL_ML_MIN": np.round(104.0 + 22.0 * _renal + rng.normal(0, 4.0, N), 1),
    "EGFR": np.round(98.0 + 20.0 * _renal + rng.normal(0, 5.0, N), 1),
    "CYSTATIN_C": np.round(np.clip(1.0 - 0.18 * _renal + rng.normal(0, 0.05, N), 0.3, 3.0), 3),
    "ALT_U_L": np.round(np.clip(26.0 + 9.0 * _hepatic + rng.normal(0, 3.0, N), 5, 200), 1),
    "AST_U_L": np.round(np.clip(24.0 + 8.0 * _hepatic + rng.normal(0, 3.5, N), 5, 200), 1),
    "GGT_U_L": np.round(np.clip(30.0 + 12.0 * _hepatic + rng.normal(0, 6.0, N), 5, 300), 1),
})
FEATURES = list(panel.columns)

# A group label, used only for the plot in part b.
group = pd.Series(np.where(_renal > 0.4, "Normal renal", "Impaired renal"), name="GROUP")

print("panel:", panel.shape)
print(panel.head().to_string(index=False))
print("")
print("standard deviations, which differ by orders of magnitude:")
print(panel.std().round(3).to_string())
```

## Toolkit for this problem

| What you want to do | How |
|---|---|
| standardise | `Z = StandardScaler().fit_transform(X)` |
| fit PCA | `pca = PCA().fit(Z)` |
| variance share per component | `pca.explained_variance_ratio_` |
| running total | `np.cumsum(evr)` |
| first index reaching a level | `int(np.argmax(cumulative >= 0.90)) + 1` |
| project the data | `scores = pca.transform(Z)` |
| the loadings | `pca.components_`, one ROW per component |
| as a readable table | `pd.DataFrame(pca.components_, columns=FEATURES)` |
| the biggest loadings | `series.abs().sort_values(ascending=False).head(3)` |

`np.argmax` on a boolean array returns the first True, which is the tidy way
to ask how many components you need. Add 1 because positions count from zero.

The toolkit cell demonstrates each of those on a tiny made up example.
It is in the notebook and none of it is the answer.

## Part a. Standardise, then fit

Standardise the panel with `StandardScaler().fit_transform(panel)`, then fit a
`PCA()` with no `n_components` argument, which keeps all ten so you can see
the whole variance spectrum.

Take `explained_variance_ratio_`, the share of total variance each component
carries. It sums to 1.

Then do the thing the concept section warns about, so you can see it rather
than take it on faith. Fit a second PCA on the **raw**, unstandardised panel
and record the share its first component carries. Look at the standard
deviations printed in the setup: `GGT_U_L` varies by about 12 units while
`CYSTATIN_C` varies by about 0.18. Squared, that is a factor of several
thousand, and PC1 of the unscaled fit will be almost entirely the
largest-numbered column.

Fill in `pca`, the PCA fitted on standardised data. `evr`, its explained
variance ratios. And `evr_unscaled_first`, the first component's share when
fitted on the raw panel.

Moves: `StandardScaler().fit_transform(...)`, `PCA().fit(...)`,
`.explained_variance_ratio_`.

**Variables this part is graded on:** `pca`, `evr`, `evr_unscaled_first`

The starter cell they see:

```python
# Part a. PCA on the standardised panel, and what happens without standardising.

pca = None                  # YOUR CODE: PCA() fitted on the STANDARDISED panel
evr = None                  # YOUR CODE: its explained_variance_ratio_
evr_unscaled_first = None   # YOUR CODE: first component's share, fitted on the RAW panel

print("explained variance ratio:", None if evr is None else np.round(evr, 4))
print("sums to                 :", None if evr is None else round(float(np.sum(evr)), 6))
print("")
print("PC1 share, standardised :", None if evr is None else round(float(evr[0]), 4))
print("PC1 share, raw          :", evr_unscaled_first)
```

**Every failure message part a can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `pca` should be a fitted PCA. Call .fit on it, which returns the fitted object.
- `pca` kept %d components but PCA() with no n_components keeps all %d.
- `evr` should hold one share per component, so shape (%d,).
- `evr` should sum to 1. explained_variance_ratio_ gives shares of the total.
- `evr` came from PCA on the RAW panel. Standardise first, or PC1 just points along whichever column has the biggest units.
- `evr` does not match PCA fitted on the standardised panel.
- `evr_unscaled_first` should be a single number.
- `evr_unscaled_first` matches the standardised answer, so the second PCA was also fitted on scaled data. Fit it on the raw panel.
- `evr_unscaled_first` is not the first explained variance ratio on the raw panel.

## Part b. How many components, and where the subjects land

Take the running total of `evr` with `np.cumsum`, then find how many
components you need to reach 90 percent of the variance.

`np.argmax` on a boolean array returns the position of the first `True`, so
`np.argmax(cumulative >= 0.90)` gives a zero based position and adding 1 turns
it into a count. That idiom is worth remembering; the obvious loop is three
lines longer and no clearer.

Then project the subjects with `pca.transform(Z)`. The result has one row per
subject and one column per component, and those columns are uncorrelated with
each other by construction.

Think about the number you get. The panel has ten columns but was built from
three underlying factors plus measurement noise, so three components should
carry the bulk and the count for 90 percent should be small. That number is
the answer to "how many things is this panel really measuring", which is a
more interesting question than how to compress it.

Fill in `cumulative`, `n_for_90`, `scores`.

Moves: `np.cumsum`, `np.argmax(cumulative >= 0.90) + 1`, `pca.transform(Z)`.

**Variables this part is graded on:** `cumulative`, `n_for_90`, `scores`

The starter cell they see:

```python
# Part b. How many components carry 90 percent, and the projection.

cumulative = None   # YOUR CODE: running total of evr
n_for_90 = None     # YOUR CODE: how many components reach 90 percent
scores = None       # YOUR CODE: the subjects projected onto the components

print("cumulative:", None if cumulative is None else np.round(cumulative, 4))
print("components needed for 90 percent:", n_for_90)
print("scores shape:", None if scores is None else scores.shape)

if scores is not None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name in group.unique():
        sel = (group == name).to_numpy()
        ax.scatter(scores[sel, 0], scores[sel, 1], s=28, alpha=0.75, label=name)
    ax.set_xlabel("PC1 (%.1f%% of variance)" % (100 * evr[0]))
    ax.set_ylabel("PC2 (%.1f%% of variance)" % (100 * evr[1]))
    ax.set_title("Subjects in the first two components")
    ax.legend(frameon=False)
    plt.show()
```

**Every failure message part b can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `cumulative` should have one entry per component.
- `cumulative` is not np.cumsum(evr).
- `n_for_90` should be a single whole number.
- `n_for_90` looks like a 0 based position. np.argmax gives a position, so add 1 to turn it into a count.
- `n_for_90` is not the number of components whose cumulative share first reaches 0.90.
- `scores` should be 2D, one row per subject and one column per component.
- `scores` has shape %s but it should be (%d, %d).
- `scores` columns are correlated (PC1 against PC2 gives %.3f), which cannot happen for a real projection. Use pca.transform on the standardised data.
- `cumulative` is just evr again. Take its running total with np.cumsum.

## Part c. Read the loadings to find out what the components are

A component is only useful once you can say what it means, and that comes from
the loadings in `pca.components_`: one **row** per component, one column per
original feature.

Build `loadings` as a DataFrame with the components as rows, indexed `PC1`,
`PC2` and so on, and `FEATURES` as the columns.

Then for PC1 and PC2, take the three features with the largest **absolute**
loading, as a list of names in descending order of absolute size. Absolute,
because a loading of minus 0.5 contributes as much as plus 0.5, just in the
opposite direction.

Read what you get against how the panel was built. There were three underlying
factors: body size, renal function and hepatic enzymes. Each component should
come out dominated by one family, which is PCA recovering the structure
without having been told it exists.

On signs: within a component, features loading with the **same** sign move
together and opposite signs move oppositely. `CYSTATIN_C` rises as renal
function falls, so it should load opposite to `CRCL_ML_MIN` and `EGFR` on
whichever component is renal. But the sign of the component as a whole is
arbitrary, so do not read anything into whether that component's loadings are
mostly positive or mostly negative.

Fill in `loadings`, `pc1_top3`, `pc2_top3`.

Moves: `pd.DataFrame(pca.components_, columns=FEATURES, index=[...])`,
`.abs().sort_values(ascending=False)`, `.head(3)`, `list(series.index)`.

**Variables this part is graded on:** `loadings`, `pc1_top3`, `pc2_top3`

The starter cell they see:

```python
# Part c. What are PC1 and PC2 made of.

loadings = None   # YOUR CODE: DataFrame, rows PC1..PC10, columns FEATURES
pc1_top3 = None   # YOUR CODE: the 3 features with the largest ABSOLUTE PC1 loading,
                  #            as a list of names, largest first
pc2_top3 = None   # YOUR CODE: the same for PC2

print("PC1 is dominated by:", pc1_top3)
print("PC2 is dominated by:", pc2_top3)
if loadings is not None:
    print("")
    print(loadings.head(3).round(3).to_string())
```

**Every failure message part c can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `loadings` should be a pandas DataFrame.
- `loadings` should have the feature names as its columns.
- `loadings` should have one row per component.
- `loadings` should be indexed PC1, PC2 and so on.
- `loadings` row PC1 does not match pca.components_[0].
- `loadings` is transposed. components_ has one ROW per component, so FEATURES are the columns.
- `%s` should hold exactly 3 feature names, but it holds %d.
- `%s` is in ascending order. Sort by absolute loading descending, so the biggest contributor comes first.
- `%s` holds the right features but not in descending order of absolute loading.
- `%s` does not hold the three largest absolute loadings. Take .abs() before sorting, since a large negative loading counts as much as a large positive one.
- `%s` should be a list of feature names.

**Functions this problem introduces:** PCA, explained_variance_ratio_, components_, np.cumsum
