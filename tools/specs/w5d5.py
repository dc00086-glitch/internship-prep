PID = "W5D5"
WEEK = 5
DAY = 5
SLUG = "w5d5_sklearn_pca"
TITLE = "PCA on a correlated covariate panel"
TOPIC = "scikit-learn: PCA, explained variance, loadings, projecting onto components"
EST = "20-30 min"

SCENARIO_MD = '''
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
'''

CONCEPT_MD = '''
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
'''

TASK_MD = '''
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
'''

DATA_CODE = '''
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
'''

TOOLKIT_MD = '''
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
'''

TOOLKIT_CODE = '''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# A tiny made up panel: two correlated pairs plus one independent column.
demo_rng = np.random.default_rng(0)
demo_n = 150
demo_a, demo_b = demo_rng.normal(size=demo_n), demo_rng.normal(size=demo_n)
demo = pd.DataFrame({
    "a1": demo_a * 1000.0 + demo_rng.normal(0, 50, demo_n),   # huge units on purpose
    "a2": demo_a + demo_rng.normal(0, 0.1, demo_n),
    "b1": demo_b + demo_rng.normal(0, 0.1, demo_n),
    "b2": demo_b + demo_rng.normal(0, 0.1, demo_n),
    "c1": demo_rng.normal(size=demo_n),
})
print("standard deviations:", demo.std().round(2).to_dict())

print("")
print("--- 1. without standardising, PCA just finds the biggest units ---")
demo_raw = PCA().fit(demo)
print("explained variance ratio:", np.round(demo_raw.explained_variance_ratio_, 4))
print("PC1 loadings            :",
      dict(zip(demo.columns, np.round(demo_raw.components_[0], 3))))
print("PC1 is entirely a1, because a1 is recorded in thousands")

print("")
print("--- 2. standardised, it finds the correlation structure ---")
demo_Z = StandardScaler().fit_transform(demo)
demo_pca = PCA().fit(demo_Z)
demo_evr = demo_pca.explained_variance_ratio_
print("explained variance ratio:", np.round(demo_evr, 4))
print("cumulative              :", np.round(np.cumsum(demo_evr), 4))
print("components for 90 percent:", int(np.argmax(np.cumsum(demo_evr) >= 0.90)) + 1)

print("")
print("--- 3. loadings say what each component IS ---")
demo_load = pd.DataFrame(demo_pca.components_, columns=demo.columns,
                         index=["PC%d" % (i + 1) for i in range(len(demo.columns))])
print(demo_load.round(3).to_string())
print("")
print("the a pair load together on one component and the b pair on another")
print("signs are arbitrary: PC1 and minus PC1 are the same axis, so only read")
print("signs RELATIVE to each other within one row")

print("")
print("--- 4. projecting the subjects ---")
demo_scores = demo_pca.transform(demo_Z)
print("scores shape:", demo_scores.shape, " <- one row per subject, one column per PC")
print("correlation between PC1 and PC2 scores: %.2e  <- zero by construction"
      % float(np.corrcoef(demo_scores[:, 0], demo_scores[:, 1])[0, 1]))
'''

PARTS = [
    dict(
        label="a",
        prompt_md='''
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
''',
        starter='''
# Part a. PCA on the standardised panel, and what happens without standardising.

pca = None                  # YOUR CODE: PCA() fitted on the STANDARDISED panel
evr = None                  # YOUR CODE: its explained_variance_ratio_
evr_unscaled_first = None   # YOUR CODE: first component's share, fitted on the RAW panel

print("explained variance ratio:", None if evr is None else np.round(evr, 4))
print("sums to                 :", None if evr is None else round(float(np.sum(evr)), 6))
print("")
print("PC1 share, standardised :", None if evr is None else round(float(evr[0]), 4))
print("PC1 share, raw          :", evr_unscaled_first)
''',
        solution='''
# Part a. PCA on the standardised panel, and what happens without standardising.

Z = StandardScaler().fit_transform(panel)
pca = PCA().fit(Z)
evr = pca.explained_variance_ratio_
evr_unscaled_first = float(PCA().fit(panel).explained_variance_ratio_[0])

print("explained variance ratio:", None if evr is None else np.round(evr, 4))
print("sums to                 :", None if evr is None else round(float(np.sum(evr)), 6))
print("")
print("PC1 share, standardised :", None if evr is None else round(float(evr[0]), 4))
print("PC1 share, raw          :", evr_unscaled_first)
''',
    ),
    dict(
        label="b",
        prompt_md='''
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
''',
        starter='''
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
''',
        solution='''
# Part b. How many components carry 90 percent, and the projection.

cumulative = np.cumsum(evr)
n_for_90 = int(np.argmax(cumulative >= 0.90) + 1)
scores = pca.transform(Z)

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
''',
    ),
    dict(
        label="c",
        prompt_md='''
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
''',
        starter='''
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
''',
        solution='''
# Part c. What are PC1 and PC2 made of.

loadings = pd.DataFrame(pca.components_, columns=FEATURES,
                        index=["PC%d" % (i + 1) for i in range(pca.components_.shape[0])])
pc1_top3 = list(loadings.loc["PC1"].abs().sort_values(ascending=False).head(3).index)
pc2_top3 = list(loadings.loc["PC2"].abs().sort_values(ascending=False).head(3).index)

print("PC1 is dominated by:", pc1_top3)
print("PC2 is dominated by:", pc2_top3)
if loadings is not None:
    print("")
    print(loadings.head(3).round(3).to_string())
''',
    ),
]


def build_expected(ns):
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    panel, feats = ns["panel"], ns["FEATURES"]
    Z = StandardScaler().fit_transform(panel)
    pca = PCA().fit(Z)
    evr = pca.explained_variance_ratio_
    load = pd.DataFrame(pca.components_, columns=feats,
                        index=["PC%d" % (i + 1) for i in range(pca.components_.shape[0])])
    return {
        "n_features": len(feats),
        "features": [str(f) for f in feats],
        "n_subjects": int(len(panel)),
        "evr": [float(v) for v in evr],
        "evr_raw": [float(v) for v in PCA().fit(panel).explained_variance_ratio_],
        "evr_unscaled_first": float(PCA().fit(panel).explained_variance_ratio_[0]),
        "cumulative": [float(v) for v in np.cumsum(evr)],
        "n_for_90": int(np.argmax(np.cumsum(evr) >= 0.90) + 1),
        "abs_load_pc1": [float(abs(v)) for v in load.loc["PC1"]],
        "abs_load_pc2": [float(abs(v)) for v in load.loc["PC2"]],
        "pc1_top3": [str(v) for v in load.loc["PC1"].abs().sort_values(ascending=False).head(3).index],
        "pc2_top3": [str(v) for v in load.loc["PC2"].abs().sort_values(ascending=False).head(3).index],
    }


CHECK_CODE = '''
def _part_a():
    pca = _need("pca")
    evr = _need("evr")
    evr_unscaled_first = _need("evr_unscaled_first")

    if not hasattr(pca, "components_"):
        return "`pca` should be a fitted PCA. Call .fit on it, which returns the fitted object."
    if pca.components_.shape[0] != _EXP["n_features"]:
        return ("`pca` kept %d components but PCA() with no n_components keeps all %d."
                % (pca.components_.shape[0], _EXP["n_features"]))

    arr = _as_array(evr)
    if arr is None or arr.shape != (_EXP["n_features"],):
        return "`evr` should hold one share per component, so shape (%d,)." % _EXP["n_features"]
    if not _num_close(float(arr.sum()), 1.0, rtol=1e-6):
        return "`evr` should sum to 1. explained_variance_ratio_ gives shares of the total."
    if _arr_close(arr, _EXP["evr_raw"], rtol=1e-4):
        return ("`evr` came from PCA on the RAW panel. Standardise first, or PC1 just points "
                "along whichever column has the biggest units.")
    if not _arr_close(arr, _EXP["evr"], rtol=1e-4):
        return "`evr` does not match PCA fitted on the standardised panel."

    got = _as_scalar(evr_unscaled_first)
    if got is None:
        return "`evr_unscaled_first` should be a single number."
    if _num_close(got, _EXP["evr"][0], rtol=1e-4):
        return ("`evr_unscaled_first` matches the standardised answer, so the second PCA was "
                "also fitted on scaled data. Fit it on the raw panel.")
    if not _num_close(got, _EXP["evr_unscaled_first"], rtol=1e-4):
        return "`evr_unscaled_first` is not the first explained variance ratio on the raw panel."
    return None


def _part_b():
    cumulative = _need("cumulative")
    n_for_90 = _need("n_for_90")
    scores = _need("scores")

    arr = _as_array(cumulative)
    if arr is None or arr.shape != (_EXP["n_features"],):
        return "`cumulative` should have one entry per component."
    if not _arr_close(arr, _EXP["cumulative"], rtol=1e-5):
        if _arr_close(arr, _EXP["evr"], rtol=1e-5):
            return "`cumulative` is just evr again. Take its running total with np.cumsum."
        return "`cumulative` is not np.cumsum(evr)."

    got = _as_scalar(n_for_90)
    if got is None:
        return "`n_for_90` should be a single whole number."
    if int(round(got)) == _EXP["n_for_90"] - 1:
        return ("`n_for_90` looks like a 0 based position. np.argmax gives a position, so "
                "add 1 to turn it into a count.")
    if int(round(got)) != _EXP["n_for_90"]:
        return "`n_for_90` is not the number of components whose cumulative share first reaches 0.90."

    arr = _as_array(scores)
    if arr is None or arr.ndim != 2:
        return "`scores` should be 2D, one row per subject and one column per component."
    if arr.shape != (_EXP["n_subjects"], _EXP["n_features"]):
        return ("`scores` has shape %s but it should be (%d, %d)."
                % (arr.shape, _EXP["n_subjects"], _EXP["n_features"]))
    corr = float(_np.corrcoef(arr[:, 0], arr[:, 1])[0, 1])
    if abs(corr) > 1e-6:
        return ("`scores` columns are correlated (PC1 against PC2 gives %.3f), which cannot "
                "happen for a real projection. Use pca.transform on the standardised data."
                % corr)
    return None


def _part_c():
    loadings = _need("loadings")
    pc1_top3 = _need("pc1_top3")
    pc2_top3 = _need("pc2_top3")

    if not isinstance(loadings, _pd.DataFrame):
        return "`loadings` should be a pandas DataFrame."
    cols = _cols_of(loadings) or []
    if sorted(cols) != sorted(_EXP["features"]):
        if len(cols) == _EXP["n_features"] and list(loadings.index) == _EXP["features"]:
            return ("`loadings` is transposed. components_ has one ROW per component, so "
                    "FEATURES are the columns.")
        return "`loadings` should have the feature names as its columns."
    if len(loadings) != _EXP["n_features"]:
        return "`loadings` should have one row per component."
    if "PC1" not in [str(i) for i in loadings.index]:
        return "`loadings` should be indexed PC1, PC2 and so on."

    got = _np.abs(_np.asarray(loadings.loc["PC1"].to_numpy(), dtype=float))
    order = [str(c) for c in loadings.columns]
    want = dict(zip(_EXP["features"], _EXP["abs_load_pc1"]))
    if not _arr_close(got, [want[c] for c in order], rtol=1e-3, atol=1e-4):
        return "`loadings` row PC1 does not match pca.components_[0]."

    for name, value, expect in (("pc1_top3", pc1_top3, _EXP["pc1_top3"]),
                                ("pc2_top3", pc2_top3, _EXP["pc2_top3"])):
        try:
            got_list = [str(v) for v in value]
        except TypeError:
            return "`%s` should be a list of feature names." % name
        if len(got_list) != 3:
            return "`%s` should hold exactly 3 feature names, but it holds %d." % (name, len(got_list))
        if got_list == expect[::-1]:
            return ("`%s` is in ascending order. Sort by absolute loading descending, so the "
                    "biggest contributor comes first." % name)
        if sorted(got_list) == sorted(expect) and got_list != expect:
            return "`%s` holds the right features but not in descending order of absolute loading." % name
        if got_list != expect:
            return ("`%s` does not hold the three largest absolute loadings. Take .abs() "
                    "before sorting, since a large negative loading counts as much as a "
                    "large positive one." % name)
    return None
'''

# Functions this problem introduces. Keys must exist in tools/function_catalog.py.
FUNCTIONS = [
    "PCA",
    "explained_variance_ratio_",
    "components_",
    "np.cumsum",
]
