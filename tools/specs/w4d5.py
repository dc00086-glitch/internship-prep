PID = "W4D5"
WEEK = 4
DAY = 5
SLUG = "w4d5_imaging_pipeline_animals"
TITLE = "A whole imaging study: fields inside animals inside groups"
TOPIC = "scikit-image pipeline: per field, per animal, per group, with the right n"
EST = "20-30 min"

SCENARIO_MD = '''
## Scenario

The full study. Two groups, four animals in each, and between two and four
usable fields of view imaged per animal, because some slides had more
intact tissue than others. Every field contains some cells. The readout is
the mean staining intensity of the cells.

So the data is nested **three** deep: cells inside fields, fields inside
animals, animals inside groups.

This is the week 1 lesson again, with an extra layer and a more convincing
disguise. Pooling every cell from every field gives you a number computed
from a few hundred measurements, which feels reassuringly large and is wrong
for exactly the same reasons as before: n is inflated, and animals are
weighted by how much tissue survived sectioning rather than by anything
biological.

The fix is the same, applied twice. Cells up to fields, fields up to animals,
animals up to groups. The group SD at the end is across **eight** animals, and
it is the only n in this study that means anything.
'''

CONCEPT_MD = '''
## Aggregating up a hierarchy, one level at a time

**The rule.** Summarise at each level before moving to the next, and let every
unit at a level count once, regardless of how much data sits beneath it.

Here that is three steps. Each field gives one number, the mean over its
cells. Each animal gives one number, the mean over its fields. Each group
gives a mean and an SD over its animals.

**Why not just average all the cells.** Because the number of cells per field
and the number of fields per animal both vary, and neither variation is
biological. An animal whose sections happened to come out well contributes
four fields and perhaps two hundred cells; one whose sections tore contributes
two fields and eighty. Pool them and the first animal counts two and a half
times as much as the second. Your group mean is then partly a measurement of
sectioning quality.

**Why not weight by cell count.** It is tempting, because more cells feels
like more information. But the thing you are trying to generalise over is
**animals**, and you have eight of them whatever happens on the slides.
Weighting by cells makes your estimate more precise about the animals you
happened to image well, which is not the same as being more precise about the
population. Unweighted means at each level is the standard approach, and the
one a reviewer will expect.

**What the final SD must be across.** Animals. Not fields, not cells. With
four animals per group, that SD rests on four numbers and is correspondingly
uncertain, and reporting it honestly rather than reaching for a tighter number
derived from cells is the whole discipline. A cell level SD would be perhaps
ten times narrower and would describe the spread of cells, which nobody asked
about.

**The shape this takes in pandas.** Each level is a `groupby` on the keys that
survive, aggregating the column produced by the level below. Three small
steps, each obvious on its own. The discipline is not technical, it is
remembering to do it at all when the pooled version is one line shorter and
gives you a smaller p value.
'''

TASK_MD = '''
## Your task

Three parts, worked one at a time, each feeding the next.

| Part | What you work out | Variables to fill in |
|---|---|---|
| a | one row per field, from the images | `per_field` |
| b | one row per animal | `per_animal` |
| c | one row per group, SD across animals | `group_stats` |

Below this you will find the **Setup** cell, then a **Toolkit** cell that
demonstrates every move on a tiny made up example, then **Part a**, **Part b**
and **Part c**, each with its own explanation and a cell to write in.
'''

DATA_CODE = '''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import draw, filters, measure, morphology, segmentation
from scipy import ndimage as ndi

PROBLEM_ID = "W4D5"
TOPIC = "scikit-image pipeline: per field, per animal, per group, with the right n"

IMG_SIZE = 128
MIN_AREA = 60
rng = np.random.default_rng(4501)

GROUP_ORDER = ["Vehicle", "Drug"]
ANIMALS = [("V%d" % i, "Vehicle") for i in range(1, 5)] + \\
          [("D%d" % i, "Drug") for i in range(1, 5)]
_group_level = {"Vehicle": 128.0, "Drug": 74.0}


def _make_field(seed, level):
    """One field of view: a few cells on a dim background."""
    r = np.random.default_rng(seed)
    img = np.full((IMG_SIZE, IMG_SIZE), 35.0)
    n = int(r.integers(6, 12))
    for _ in range(n):
        ra, rb = float(r.uniform(7.0, 10.0)), float(r.uniform(6.0, 9.0))
        cy = int(r.integers(12, IMG_SIZE - 12))
        cx = int(r.integers(12, IMG_SIZE - 12))
        rr, cc = draw.ellipse(cy, cx, ra, rb, shape=img.shape,
                              rotation=float(r.uniform(0, np.pi)))
        img[rr, cc] += level * r.uniform(0.80, 1.20)
    img += r.normal(0.0, 5.0, size=img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


# Build the study. Animals differ from each other, and the number of usable
# fields differs too, because some sections tore.
fields = []
_seed = 100
for _animal, _grp in ANIMALS:
    _animal_level = _group_level[_grp] * rng.lognormal(0.0, 0.16)
    _n_fields = int(rng.integers(2, 5))
    for _f in range(1, _n_fields + 1):
        fields.append({"ANIMAL": _animal, "GROUP": _grp, "FIELD": _f,
                       "IMAGE": _make_field(_seed, _animal_level)})
        _seed += 1


def segment_field(img):
    """The week 4 pipeline in one function. Returns one row per cell."""
    mask = img > filters.threshold_otsu(img)
    mask = morphology.opening(mask, morphology.disk(2))
    mask = ndi.binary_fill_holes(mask)
    mask = segmentation.clear_border(mask)
    labels = measure.label(mask)
    props = pd.DataFrame(measure.regionprops_table(
        labels, intensity_image=img, properties=("label", "area", "mean_intensity")))
    return props[props["area"] >= MIN_AREA].reset_index(drop=True)


print("fields imaged:", len(fields))
print(pd.DataFrame([{k: v for k, v in f.items() if k != "IMAGE"} for f in fields])
      .groupby(["GROUP", "ANIMAL"]).size().to_string())
print("")
print("example of what segment_field gives you for one field:")
print(segment_field(fields[0]["IMAGE"]).round(2).to_string(index=False))

_fig, _axes = plt.subplots(1, 4, figsize=(12, 3.2))
for _ax, _f in zip(_axes, fields[:4]):
    _ax.imshow(_f["IMAGE"], cmap="gray", vmin=0, vmax=255)
    _ax.set_title("%s field %d" % (_f["ANIMAL"], _f["FIELD"]))
    _ax.axis("off")
_fig.tight_layout()
plt.show()
'''

TOOLKIT_MD = '''
## Toolkit for this problem

| What you want to do | How |
|---|---|
| loop with an index | `for f in fields:` then `f["ANIMAL"]`, `f["IMAGE"]` |
| segment one field | `segment_field(f["IMAGE"])` |
| build rows then a table | collect dicts in a list, then `pd.DataFrame(rows)` |
| a column mean | `float(df["mean_intensity"].mean())` |
| one level up | `.groupby([...]).agg(NAME=("COL", "mean"), ...).reset_index()` |
| count rows in a group | the aggregation `"size"` |
| sample SD | the aggregation `"std"`, already n-1 in pandas |

Building a list of dictionaries and handing it to `pd.DataFrame` at the end is
usually clearer than growing a DataFrame row by row, and far faster.
'''

TOOLKIT_CODE = '''
import numpy as np
import pandas as pd

# A tiny made up three level dataset. Nothing here is the answer.
demo_cells = pd.DataFrame({
    "ANIMAL": ["a1"] * 7 + ["a2"] * 3,
    "GROUP":  ["ctl"] * 10,
    "FIELD":  [1, 1, 1, 1, 2, 2, 2, 1, 1, 1],
    "VALUE":  [10.0, 12.0, 14.0, 16.0, 40.0, 42.0, 44.0, 70.0, 72.0, 74.0],
})
print("demo, one row per CELL. a1 gave 2 fields and 7 cells, a2 gave 1 field and 3:")
print(demo_cells)

print("")
print("--- 1. cells up to fields ---")
demo_field = (demo_cells.groupby(["ANIMAL", "GROUP", "FIELD"])
              .agg(N_CELLS=("VALUE", "size"), MEAN_VALUE=("VALUE", "mean"))
              .reset_index())
print(demo_field)

print("")
print("--- 2. fields up to animals ---")
demo_animal = (demo_field.groupby(["ANIMAL", "GROUP"])
               .agg(N_FIELDS=("MEAN_VALUE", "size"),
                    TOTAL_CELLS=("N_CELLS", "sum"),
                    MEAN_VALUE=("MEAN_VALUE", "mean"))
               .reset_index())
print(demo_animal)

print("")
print("--- 3. the three answers you could report, and what each weights ---")
demo_pooled = float(demo_cells["VALUE"].mean())
demo_by_field = float(demo_field["MEAN_VALUE"].mean())
demo_by_animal = float(demo_animal["MEAN_VALUE"].mean())
print("mean of every cell pooled : %.3f   <- weights by cells imaged" % demo_pooled)
print("mean of the field means   : %.3f   <- weights by fields imaged" % demo_by_field)
print("mean of the animal means  : %.3f   <- weights each ANIMAL once" % demo_by_animal)
print("")
print("all three are different, and only the last one answers a question about animals")

print("")
print("--- 4. and what n you would be claiming ---")
print("n if you count cells  :", len(demo_cells))
print("n if you count fields :", len(demo_field))
print("n if you count animals:", len(demo_animal), " <- the only honest one")
'''

PARTS = [
    dict(
        label="a",
        prompt_md='''
## Part a. Run the pipeline on every field

Loop over `fields`. Each entry is a dictionary with `ANIMAL`, `GROUP`,
`FIELD` and `IMAGE`. Hand the image to `segment_field`, which runs the whole
week 4 pipeline and returns one row per surviving cell, with `area` and
`mean_intensity`.

For each field record four things: the animal, the group, the field number,
how many cells were found, and the mean of those cells' `mean_intensity`
values. That last number is the field's single summary, and it is the first
level of aggregation.

Collect the rows as a list of dictionaries and build the DataFrame once at the
end. Growing a DataFrame inside a loop works but is slow and reads worse.

One thing to watch: a field could in principle yield no cells at all, and the
mean of nothing is `nan`. With this data every field does find cells, but
noticing that the case exists is the habit worth having.

Fill in `per_field`, sorted by `ANIMAL` then `FIELD`, index renumbered, with
exactly:

| column | meaning |
|---|---|
| `ANIMAL` | animal identifier |
| `GROUP` | `Vehicle` or `Drug` |
| `FIELD` | field number within that animal |
| `N_CELLS` | cells found in that field |
| `MEAN_INTENSITY` | mean of that field's cell intensities |

Moves: a `for` loop, `segment_field(...)`, `len()`, `.mean()`,
`pd.DataFrame(rows)`, `.sort_values([...]).reset_index(drop=True)`.
''',
        starter='''
# Part a. Segment every field and summarise each one.

rows = []
# YOUR CODE: loop over fields, segment each image, and append one dict per field
#            with ANIMAL, GROUP, FIELD, N_CELLS, MEAN_INTENSITY

per_field = None   # YOUR CODE: pd.DataFrame(rows), sorted by ANIMAL then FIELD

print(per_field)
''',
        solution='''
# Part a. Segment every field and summarise each one.

rows = []
for f in fields:
    cells = segment_field(f["IMAGE"])
    rows.append({
        "ANIMAL": f["ANIMAL"],
        "GROUP": f["GROUP"],
        "FIELD": f["FIELD"],
        "N_CELLS": int(len(cells)),
        "MEAN_INTENSITY": float(cells["mean_intensity"].mean()),
    })

per_field = (pd.DataFrame(rows)
             .sort_values(["ANIMAL", "FIELD"])
             .reset_index(drop=True))

print(per_field)
''',
    ),
    dict(
        label="b",
        prompt_md='''
## Part b. Fields up to animals

One row per animal. Three numbers: how many fields it contributed, how many
cells in total across those fields, and its mean intensity.

That last one is the mean of the **field means**, not of the cells. Each field
counts once regardless of how many cells it happened to contain, for the same
reason each animal will count once in part c regardless of how many fields it
contributed. You are applying the same rule at a second level.

`TOTAL_CELLS` is a sum rather than a mean, because it is a count of things
rather than an average of measurements. It is worth keeping: it is the number
that shows how uneven the imaging was, and therefore why the unweighted mean
matters.

Fill in `per_animal`, sorted by `ANIMAL`, index renumbered, with exactly:

| column | meaning |
|---|---|
| `ANIMAL` | animal identifier |
| `GROUP` | `Vehicle` or `Drug` |
| `N_FIELDS` | fields imaged for that animal |
| `TOTAL_CELLS` | cells found across all its fields |
| `MEAN_INTENSITY` | mean of that animal's field means |

Moves: `.groupby(["ANIMAL", "GROUP"]).agg(NAME=("COL", "mean"), ...)`,
`"size"`, `"sum"`, `.reset_index()`.
''',
        starter='''
# Part b. One row per animal.

per_animal = None   # YOUR CODE: ANIMAL, GROUP, N_FIELDS, TOTAL_CELLS, MEAN_INTENSITY

print(per_animal)
''',
        solution='''
# Part b. One row per animal.

per_animal = (per_field
              .groupby(["ANIMAL", "GROUP"])
              .agg(N_FIELDS=("MEAN_INTENSITY", "size"),
                   TOTAL_CELLS=("N_CELLS", "sum"),
                   MEAN_INTENSITY=("MEAN_INTENSITY", "mean"))
              .reset_index()
              .sort_values("ANIMAL")
              .reset_index(drop=True))

print(per_animal)
''',
    ),
    dict(
        label="c",
        prompt_md='''
## Part c. Animals up to groups

The last level, and the one that goes in the figure.

One row per group: the mean of the animal means, the sample SD **across
animals**, and the number of animals. Sort so `Vehicle` comes before `Drug`,
which is dose order rather than alphabetical order.

Check `N_ANIMALS` when you have it. It should be 4 in each group. If you see
anything in the tens it came from fields, and anything in the hundreds from
cells, and in either case the SD beside it is describing the wrong thing.

Look at the SD too. Built from four animals, it will be wide, and that is the
honest answer. A cell level SD would be far narrower and would be answering a
question about cells that nobody asked. The temptation to reach for the
narrower number is the whole reason this rule needs stating.

Fill in `group_stats`, index renumbered, with exactly:

| column | meaning |
|---|---|
| `GROUP` | `Vehicle` then `Drug` |
| `MEAN` | mean of the animal means |
| `SD` | sample SD of the animal means |
| `N_ANIMALS` | animals in that group |

Moves: `.groupby("GROUP").agg(...)`, `.reset_index()`, `.map` or a sort key
for the group order.
''',
        starter='''
# Part c. One row per group, with the SD across animals.

group_stats = None   # YOUR CODE: GROUP, MEAN, SD, N_ANIMALS, Vehicle first

print(group_stats)
''',
        solution='''
# Part c. One row per group, with the SD across animals.

group_stats = (per_animal
               .groupby("GROUP")
               .agg(MEAN=("MEAN_INTENSITY", "mean"),
                    SD=("MEAN_INTENSITY", "std"),
                    N_ANIMALS=("MEAN_INTENSITY", "size"))
               .reset_index())

group_stats["_ord"] = group_stats["GROUP"].map({g: i for i, g in enumerate(GROUP_ORDER)})
group_stats = (group_stats.sort_values("_ord")
               .drop(columns=["_ord"])
               .reset_index(drop=True))

print(group_stats)
''',
    ),
]


def build_expected(ns):
    import numpy as np
    import pandas as pd
    pf = ns["per_field"].sort_values(["ANIMAL", "FIELD"]).reset_index(drop=True)
    pa = ns["per_animal"].sort_values("ANIMAL").reset_index(drop=True)
    gs = ns["group_stats"]
    order = ns["GROUP_ORDER"]

    # the wrong ways, for diagnostics
    pooled_animal = (pf.assign(W=pf["N_CELLS"] * pf["MEAN_INTENSITY"])
                     .groupby("ANIMAL")
                     .apply(lambda d: d["W"].sum() / d["N_CELLS"].sum(), include_groups=False))
    by_field_group = (pf.groupby("GROUP")["MEAN_INTENSITY"].mean().reindex(order)).to_numpy()
    sd_field_group = (pf.groupby("GROUP")["MEAN_INTENSITY"].std().reindex(order)).to_numpy()
    n_field_group = (pf.groupby("GROUP")["MEAN_INTENSITY"].size().reindex(order)).to_numpy()
    sd_animal = (pa.groupby("GROUP")["MEAN_INTENSITY"].std().reindex(order)).to_numpy()
    n_animal = (pa.groupby("GROUP")["MEAN_INTENSITY"].size().reindex(order)).to_numpy()
    mean_animal = (pa.groupby("GROUP")["MEAN_INTENSITY"].mean().reindex(order)).to_numpy()
    return {
        "pf_cols": ["ANIMAL", "GROUP", "FIELD", "N_CELLS", "MEAN_INTENSITY"],
        "n_fields": int(len(pf)),
        "pf_animal": [str(v) for v in pf["ANIMAL"]],
        "pf_field": [int(v) for v in pf["FIELD"]],
        "pf_ncells": [int(v) for v in pf["N_CELLS"]],
        "pf_mean": [float(v) for v in pf["MEAN_INTENSITY"]],
        "pa_cols": ["ANIMAL", "GROUP", "N_FIELDS", "TOTAL_CELLS", "MEAN_INTENSITY"],
        "n_animals": int(len(pa)),
        "pa_animal": [str(v) for v in pa["ANIMAL"]],
        "pa_nfields": [int(v) for v in pa["N_FIELDS"]],
        "pa_total": [int(v) for v in pa["TOTAL_CELLS"]],
        "pa_mean": [float(v) for v in pa["MEAN_INTENSITY"]],
        "pa_mean_cellweighted": [float(pooled_animal[a]) for a in pa["ANIMAL"]],
        "gs_cols": ["GROUP", "MEAN", "SD", "N_ANIMALS"],
        "gs_group": [str(v) for v in order],
        "gs_mean": [float(v) for v in mean_animal],
        "gs_sd": [float(v) for v in sd_animal],
        "gs_n": [int(v) for v in n_animal],
        "gs_sem": [float(a / np.sqrt(b)) for a, b in zip(sd_animal, n_animal)],
        "gs_mean_byfield": [float(v) for v in by_field_group],
        "gs_sd_byfield": [float(v) for v in sd_field_group],
        "gs_n_byfield": [int(v) for v in n_field_group],
    }


CHECK_CODE = '''
def _part_a():
    pf = _need("per_field")
    if not isinstance(pf, _pd.DataFrame):
        return "`per_field` should be a pandas DataFrame."
    cols = _cols_of(pf) or []
    missing = [c for c in _EXP["pf_cols"] if c not in cols]
    if missing:
        return "`per_field` is missing the column(s) %s." % ", ".join(missing)
    if len(pf) != _EXP["n_fields"]:
        return ("`per_field` has %d rows but %d fields were imaged, so there should be one "
                "row each." % (len(pf), _EXP["n_fields"]))

    got = pf.sort_values(["ANIMAL", "FIELD"]).reset_index(drop=True)
    if [str(v) for v in got["ANIMAL"]] != _EXP["pf_animal"]:
        return "`per_field` does not list the expected animals."
    if not _arr_close(got["FIELD"].to_numpy(), _EXP["pf_field"]):
        return "`per_field` does not list the expected field numbers."
    if not _arr_close(got["N_CELLS"].to_numpy(), _EXP["pf_ncells"]):
        return ("`N_CELLS` does not match what segment_field returns. It is the number of "
                "ROWS in that result, one per cell.")
    if _np.any(_pd.isna(got["MEAN_INTENSITY"].to_numpy())):
        return "`MEAN_INTENSITY` contains nan, so some field produced no cells to average."
    if not _arr_close(got["MEAN_INTENSITY"].to_numpy(), _EXP["pf_mean"], rtol=1e-6):
        return ("`MEAN_INTENSITY` is not the mean of that field's cell mean_intensity "
                "values.")
    if list(pf.index) != list(range(len(pf))):
        return "`per_field` has not had its index renumbered. Use .reset_index(drop=True)."
    return None


def _part_b():
    pa = _need("per_animal")
    if not isinstance(pa, _pd.DataFrame):
        return "`per_animal` should be a pandas DataFrame."
    cols = _cols_of(pa) or []
    missing = [c for c in _EXP["pa_cols"] if c not in cols]
    if missing:
        return "`per_animal` is missing the column(s) %s." % ", ".join(missing)
    if len(pa) == _EXP["n_fields"]:
        return ("`per_animal` still has one row per field. Group by ANIMAL so the fields "
                "collapse.")
    if len(pa) != _EXP["n_animals"]:
        return "`per_animal` has %d rows but there are %d animals." % (len(pa), _EXP["n_animals"])

    got = pa.sort_values("ANIMAL").reset_index(drop=True)
    if [str(v) for v in got["ANIMAL"]] != _EXP["pa_animal"]:
        return "`per_animal` does not list the expected animals."
    if not _arr_close(got["N_FIELDS"].to_numpy(), _EXP["pa_nfields"]):
        return "`N_FIELDS` is not the number of fields each animal contributed."
    if not _arr_close(got["TOTAL_CELLS"].to_numpy(), _EXP["pa_total"]):
        return ("`TOTAL_CELLS` is not the sum of N_CELLS across that animal's fields. It is "
                "a sum, not a mean.")
    if _arr_close(got["MEAN_INTENSITY"].to_numpy(), _EXP["pa_mean_cellweighted"], rtol=1e-6):
        return ("`MEAN_INTENSITY` is weighted by how many cells each field contained, so a "
                "field with more cells counts more. Take the plain mean of the field means, "
                "giving each field equal weight.")
    if not _arr_close(got["MEAN_INTENSITY"].to_numpy(), _EXP["pa_mean"], rtol=1e-6):
        return "`MEAN_INTENSITY` is not the mean of that animal's field means."
    return None


def _part_c():
    gs = _need("group_stats")
    if not isinstance(gs, _pd.DataFrame):
        return "`group_stats` should be a pandas DataFrame."
    cols = _cols_of(gs) or []
    missing = [c for c in _EXP["gs_cols"] if c not in cols]
    if missing:
        return "`group_stats` is missing the column(s) %s." % ", ".join(missing)
    if len(gs) != len(_EXP["gs_group"]):
        return "`group_stats` should have one row per group, so %d." % len(_EXP["gs_group"])
    if [str(v) for v in gs["GROUP"]] != _EXP["gs_group"]:
        return ("`group_stats` is not in the order Vehicle then Drug. Sort it into "
                "GROUP_ORDER rather than leaving it alphabetical.")

    n_got = _np.asarray(gs["N_ANIMALS"].to_numpy(), dtype=float)
    if _arr_close(n_got, _EXP["gs_n_byfield"]):
        return ("`N_ANIMALS` counted FIELDS, not animals. Build this from per_animal, where "
                "each animal is already one row.")
    if not _arr_close(n_got, _EXP["gs_n"]):
        return "`N_ANIMALS` is not the number of animals per group. It should be 4 in each."

    if _arr_close(gs["MEAN"].to_numpy(), _EXP["gs_mean_byfield"], rtol=1e-6):
        return ("`MEAN` is the mean over fields, so animals with more usable sections count "
                "more. Average the animal means instead.")
    if not _arr_close(gs["MEAN"].to_numpy(), _EXP["gs_mean"], rtol=1e-6):
        return "`MEAN` is not the mean of the animal means within each group."

    sd_got = gs["SD"].to_numpy()
    if _arr_close(sd_got, _EXP["gs_sem"], rtol=1e-6):
        return ("`SD` is the SEM, not the SD. It was divided by the square root of the "
                "number of animals. This repo always reports SD.")
    if _arr_close(sd_got, _EXP["gs_sd_byfield"], rtol=1e-6):
        return ("`SD` is the spread across FIELDS, not across animals. It is narrower than "
                "the truth because fields within an animal resemble each other.")
    if not _arr_close(sd_got, _EXP["gs_sd"], rtol=1e-6):
        return "`SD` is not the sample SD of the animal means within each group."
    return None
'''

# Functions this problem introduces. Keys must exist in tools/function_catalog.py.
FUNCTIONS = [
    "building a DataFrame from rows",
    "three level aggregation",
]
