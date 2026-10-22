PID = "W1D5"
WEEK = 1
DAY = 5
SLUG = "w1d5_matplotlib_nested_sd"
TITLE = "Cells inside animals: aggregate first, then plot mean with SD"
TOPIC = "matplotlib and pandas: nested data, animal level aggregation, SD error bars"
EST = "20-30 min"

SCENARIO_MD = '''
## Scenario

You have quantified a target engagement marker by immunofluorescence. Twelve
mice, six on vehicle and six on drug. For each mouse you imaged a slide and
measured **many individual cells**, somewhere between 20 and 60 of them, so the
table you are holding has hundreds of rows.

This is the single most commonly botched analysis in cell biology, so it is
worth being slow and deliberate about it.

**The unit of replication is the animal, not the cell.** The cells inside one
mouse are not independent samples of the drug effect; they are repeat measures
of one mouse. If you pool all the cells and treat them as n, you inflate n
from 12 to several hundred, you weight each mouse by however many cells you
happened to image, and any statistic you compute afterwards is wrong. So:

1. aggregate the cells up to one number per animal,
2. then do all group level statistics across animals, with n = animals.

The second house rule: error bars are **SD**, never SEM. An SD bar tells the
reader how much the animals differ from each other, which is what they want to
know. A SEM bar mostly tells them how many animals you ran.
'''

TASK_MD = '''
## Your task

Three parts, worked one at a time, and the order matters: cells up to animals
first, then animals up to groups, then the figure.

| Part | What you work out | Variables to fill in |
|---|---|---|
| a | one row per animal, out of hundreds of cells | `animal_level` |
| b | group statistics computed across animals | `group_stats` |
| c | the figure, means with SD and every animal shown | `fig`, `ax` |

Below this you will find the **Setup** cell, then a **Toolkit** cell that
demonstrates every move on a tiny made up example, then **Part a**, **Part b**
and **Part c**, each with its own explanation and a cell to write in.
'''

DATA_CODE = '''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROBLEM_ID = "W1D5"
TOPIC = "matplotlib and pandas: nested data, animal level aggregation, SD error bars"

rng = np.random.default_rng(1105)

_group_true = {"Vehicle": 1000.0, "Drug": 420.0}

_frames = []
for _i in range(1, 13):
    _group = "Vehicle" if _i <= 6 else "Drug"
    _animal = "MS%02d" % _i
    _animal_mean = _group_true[_group] * rng.lognormal(0.0, 0.18)   # between animal
    _n_cells = int(rng.integers(20, 61))                            # unequal effort per slide
    _vals = rng.normal(_animal_mean, 0.35 * _animal_mean, size=_n_cells)   # within animal
    _frames.append(pd.DataFrame({
        "ANIMAL": _animal,
        "GROUP": _group,
        "CELL_ID": np.arange(1, _n_cells + 1),
        "INTENSITY": np.round(np.clip(_vals, 0.0, None), 1),
    }))

cells = pd.concat(_frames, ignore_index=True)

print("cells:", cells.shape, " rows are individual cells, not animals")
print(cells.head())
print("")
print("cells measured per animal:")
print(cells.groupby("ANIMAL")["CELL_ID"].size().to_string())
'''

PARTS = [
    dict(
        label="a",
        prompt_md='''
## Part a. Aggregate the cells up to the animal

This is the step that makes everything after it valid. One row per animal,
holding that animal's mean across its own cells.

Group by animal and group together, so each animal keeps its treatment label
and you do not have to join it back on afterwards. Naming the outputs inside
`.agg()` saves a rename. The aggregation `"size"` counts rows in each group,
which here means cells per animal.

Keep `N_CELLS` even though nothing downstream divides by it. It is the number
that shows how unequal the imaging effort was, and therefore why pooling the
cells would have distorted the group mean.

Fill in `animal_level`, a flat DataFrame with exactly these four columns:

| column | meaning |
|---|---|
| `ANIMAL` | animal identifier |
| `GROUP` | `Drug` or `Vehicle` |
| `MEAN_INTENSITY` | mean of that animal's own cells |
| `N_CELLS` | how many cells were measured in that animal |

Moves: `.groupby([...]).agg(NAME=("COL", "mean"), ...)`, `.reset_index()`.
''',
        starter='''
# Part a. One row per animal: ANIMAL, GROUP, MEAN_INTENSITY, N_CELLS

animal_level = None   # YOUR CODE

print(animal_level)
''',
        solution='''
# Part a. One row per animal: ANIMAL, GROUP, MEAN_INTENSITY, N_CELLS

animal_level = (cells
                .groupby(["ANIMAL", "GROUP"])
                .agg(MEAN_INTENSITY=("INTENSITY", "mean"),
                     N_CELLS=("INTENSITY", "size"))
                .reset_index())

print(animal_level)
''',
    ),
    dict(
        label="b",
        prompt_md='''
## Part b. Group statistics across animals

Build this from `animal_level`, **not** from `cells`. That one choice is the
whole point of the problem.

Every number here is computed across animals. `MEAN` is the mean of the animal
means, which is not the same as the mean of all the cells, because the animals
contributed unequal numbers of cells. `SD` is the spread of the animal means,
which is what tells a reader how reproducible the effect is. `N_ANIMALS`
should come out at 6 per group; if you see numbers in the hundreds you are
counting cells.

Fill in `group_stats`, sorted by `GROUP` which puts `Drug` first, with exactly
these four columns:

| column | meaning |
|---|---|
| `GROUP` | `Drug` or `Vehicle` |
| `MEAN` | mean of the animal means in that group |
| `SD` | **sample** SD of the animal means in that group |
| `N_ANIMALS` | number of animals in that group |

Moves: `.groupby("GROUP").agg(MEAN=("MEAN_INTENSITY", "mean"), ...)`,
`.reset_index()`, `.sort_values(...)`.
''',
        starter='''
# Part b. One row per group, computed across animals: GROUP, MEAN, SD, N_ANIMALS

group_stats = None   # YOUR CODE

print(group_stats)
''',
        solution='''
# Part b. One row per group, computed across animals: GROUP, MEAN, SD, N_ANIMALS

group_stats = (animal_level
               .groupby("GROUP")
               .agg(MEAN=("MEAN_INTENSITY", "mean"),
                    SD=("MEAN_INTENSITY", "std"),
                    N_ANIMALS=("MEAN_INTENSITY", "size"))
               .reset_index()
               .sort_values("GROUP")
               .reset_index(drop=True))

print(group_stats)
''',
    ),
    dict(
        label="c",
        prompt_md='''
## Part c. The figure

A figure that shows only two means and two error bars hides your whole
dataset. Overlaying every animal costs one line and lets a reader see the
spread, the sample size and any outlier at a glance. Make it your default.

Build it the explicit way: get `fig` and `ax` from `plt.subplots()`, then call
methods on `ax`. `yerr` comes from the `SD` column of `group_stats`, which is
the SD **across animals**, not across cells and not the SEM.

What the figure needs:

- `fig, ax = plt.subplots(figsize=(6, 4.5))`
- the two group means drawn with `ax.errorbar`, at x positions `0` and `1`, in
  the row order of `group_stats`, with `yerr` set to the `SD` column and
  `fmt="o"`
- every animal's own mean overlaid with `ax.scatter`, one point per animal, at
  the x position of its group. A little jitter on x is welcome but optional
- x ticks set to `[0, 1]` and labelled with the group names
- an x label, a y label and a title. An unlabelled axis makes a figure
  unreadable to everyone except the person who made it, including you in three
  weeks

Fill in `fig`, `ax`.
Moves: `plt.subplots(figsize=...)`, `ax.errorbar(..., yerr=..., fmt="o")`,
`ax.scatter(...)`, `ax.set_xticks`, `ax.set_xticklabels`, `ax.set_xlabel`,
`ax.set_ylabel`, `ax.set_title`.
''',
        starter='''
# Part c. Group means with SD error bars, every animal overlaid.

fig, ax = plt.subplots(figsize=(6, 4.5))

x_pos = [0, 1]

# YOUR CODE: ax.errorbar(...) for the two group means, yerr = the SD column

# YOUR CODE: ax.scatter(...) for the 12 individual animal means

# YOUR CODE: ax.set_xticks(...), ax.set_xticklabels(...)

# YOUR CODE: ax.set_xlabel(...), ax.set_ylabel(...), ax.set_title(...)

plt.show()
''',
        solution='''
# Part c. Group means with SD error bars, every animal overlaid.

fig, ax = plt.subplots(figsize=(6, 4.5))

x_pos = [0, 1]
order = list(group_stats["GROUP"])

ax.errorbar(x_pos, group_stats["MEAN"], yerr=group_stats["SD"],
            fmt="o", color="black", markersize=9, capsize=6,
            linestyle="none", zorder=3, label="group mean, SD")

jitter_rng = np.random.default_rng(7)
animal_x = np.array([order.index(g) for g in animal_level["GROUP"]], dtype=float)
animal_x = animal_x + jitter_rng.normal(0.0, 0.05, size=len(animal_level))
ax.scatter(animal_x, animal_level["MEAN_INTENSITY"],
           s=45, alpha=0.75, color="tab:blue", zorder=2, label="animal mean")

ax.set_xticks(x_pos)
ax.set_xticklabels(order)
ax.set_xlim(-0.5, 1.5)
ax.set_xlabel("Treatment group")
ax.set_ylabel("Target marker intensity (AU)")
ax.set_title("Target engagement by group, mean with SD across animals")
ax.legend(frameon=False)

plt.show()
''',
    ),
]


def build_expected(ns):
    import numpy as np
    cells = ns["cells"]
    animal_level = ns["animal_level"].sort_values("ANIMAL").reset_index(drop=True)
    gs = ns["group_stats"].sort_values("GROUP").reset_index(drop=True)
    pooled = cells.groupby("GROUP")["INTENSITY"].agg(["mean", "std", "size"]).sort_index()
    return {
        "n_cells_total": int(len(cells)),
        "n_animals": int(cells["ANIMAL"].nunique()),
        "animal_cols": ["ANIMAL", "GROUP", "MEAN_INTENSITY", "N_CELLS"],
        "animal_id": [str(v) for v in animal_level["ANIMAL"]],
        "animal_group": [str(v) for v in animal_level["GROUP"]],
        "animal_mean": [float(v) for v in animal_level["MEAN_INTENSITY"]],
        "animal_median": [float(v) for v in
                          cells.groupby("ANIMAL")["INTENSITY"].median().sort_index()],
        "animal_ncells": [int(v) for v in animal_level["N_CELLS"]],
        "group_cols": ["GROUP", "MEAN", "SD", "N_ANIMALS"],
        "group_names": [str(v) for v in gs["GROUP"]],
        "group_mean": [float(v) for v in gs["MEAN"]],
        "group_sd": [float(v) for v in gs["SD"]],
        "group_n": [int(v) for v in gs["N_ANIMALS"]],
        "group_sem": [float(v) for v in gs["SD"] / np.sqrt(gs["N_ANIMALS"])],
        "group_mean_pooled": [float(v) for v in pooled["mean"]],
        "group_sd_pooled": [float(v) for v in pooled["std"]],
        "group_n_pooled": [int(v) for v in pooled["size"]],
    }


CHECK_CODE = '''
def _part_a():
    animal_level = _need("animal_level")
    if not isinstance(animal_level, _pd.DataFrame):
        return "`animal_level` should be a pandas DataFrame, but it is a %s." % type(animal_level).__name__

    cols = _cols_of(animal_level)
    for name in ("ANIMAL", "GROUP"):
        if name not in cols and name in [str(n) for n in (animal_level.index.names or [])]:
            return ("`%s` is in the index rather than a column. Call .reset_index() after "
                    "the groupby." % name)
    missing = [c for c in _EXP["animal_cols"] if c not in cols]
    if missing:
        return ("`animal_level` is missing the column(s) %s. Naming them as you aggregate, "
                "with .agg(MEAN_INTENSITY=(\\"INTENSITY\\", \\"mean\\"), ...), saves a rename."
                % ", ".join(missing))
    if len(animal_level) == _EXP["n_cells_total"]:
        return ("`animal_level` still has one row per cell. It needs one row per animal, so "
                "group by ANIMAL before aggregating.")
    if len(animal_level) != _EXP["n_animals"]:
        return ("`animal_level` has %d rows but there are %d animals." % (len(animal_level), _EXP["n_animals"]))

    got = animal_level.sort_values("ANIMAL").reset_index(drop=True)
    if [str(v) for v in got["ANIMAL"]] != _EXP["animal_id"]:
        return "`animal_level` does not list the expected animals."
    if [str(v) for v in got["GROUP"]] != _EXP["animal_group"]:
        return ("`GROUP` does not line up with `ANIMAL`. Group by both at once so each "
                "animal keeps its own group label.")
    if _arr_close(got["MEAN_INTENSITY"].to_numpy(), _EXP["animal_median"]):
        return "`MEAN_INTENSITY` is the median of each animal's cells, not the mean."
    if not _arr_close(got["MEAN_INTENSITY"].to_numpy(), _EXP["animal_mean"]):
        return "`MEAN_INTENSITY` is not the plain mean of each animal's own cells."
    if not _arr_close(got["N_CELLS"].to_numpy(), _EXP["animal_ncells"]):
        return ("`N_CELLS` is not the number of cells measured in each animal. The aggregation "
                "\\"size\\" counts rows within the group.")
    return None


def _part_b():
    group_stats = _need("group_stats")
    if not isinstance(group_stats, _pd.DataFrame):
        return "`group_stats` should be a pandas DataFrame, but it is a %s." % type(group_stats).__name__

    cols = _cols_of(group_stats)
    if "GROUP" not in cols and "GROUP" in [str(n) for n in (group_stats.index.names or [])]:
        return "`GROUP` is in the index rather than a column. Call .reset_index()."
    missing = [c for c in _EXP["group_cols"] if c not in cols]
    if missing:
        return "`group_stats` is missing the column(s) %s." % ", ".join(missing)
    if len(group_stats) != len(_EXP["group_names"]):
        return ("`group_stats` has %d rows but there are %d groups."
                % (len(group_stats), len(_EXP["group_names"])))

    got = group_stats.sort_values("GROUP").reset_index(drop=True)
    if [str(v) for v in got["GROUP"]] != _EXP["group_names"]:
        return "`group_stats` does not list the expected groups."

    n_got = _np.asarray(got["N_ANIMALS"].to_numpy(), dtype=float)
    if _arr_close(n_got, _EXP["group_n_pooled"]):
        return ("`N_ANIMALS` counted cells, not animals. It came out in the hundreds, which "
                "is the inflated n the scenario warns about. Build this table from "
                "animal_level, not from cells.")
    if not _arr_close(n_got, _EXP["group_n"]):
        return "`N_ANIMALS` is not the number of animals per group."

    if _arr_close(got["MEAN"].to_numpy(), _EXP["group_mean_pooled"]):
        return ("`MEAN` is the mean of every cell pooled together, which weights each animal "
                "by how many cells you happened to image. Take the mean of the animal means "
                "instead, so every animal counts once.")
    if not _arr_close(got["MEAN"].to_numpy(), _EXP["group_mean"]):
        return "`MEAN` is not the mean of the animal means within each group."

    sd_got = got["SD"].to_numpy()
    if _arr_close(sd_got, _EXP["group_sem"]):
        return ("`SD` is the SEM, not the SD. It was divided by the square root of the number "
                "of animals. This repo always reports SD.")
    if _arr_close(sd_got, _EXP["group_sd_pooled"]):
        return ("`SD` is the spread of individual cells, not the spread of animals. It is "
                "much larger than the animal level SD, and it answers a question nobody "
                "asked. Compute it across the animal means.")
    if not _arr_close(sd_got, _EXP["group_sd"]):
        return "`SD` is not the sample SD of the animal means within each group."
    return None


def _yerr_values(ax):
    import matplotlib.container as _mcont
    out = []
    conts = list(ax.containers)
    for cont in conts:
        eb = None
        if isinstance(cont, _mcont.ErrorbarContainer):
            eb = cont
        elif isinstance(cont, _mcont.BarContainer) and getattr(cont, "errorbar", None) is not None:
            eb = cont.errorbar
        if eb is None:
            continue
        lines = getattr(eb, "lines", None)
        if not lines or len(lines) < 3 or not lines[2]:
            continue
        for col in lines[2]:
            try:
                segs = col.get_segments()
            except Exception:
                continue
            for seg in segs:
                seg = _np.asarray(seg, dtype=float)
                if seg.shape[0] >= 2 and seg.shape[1] >= 2:
                    out.append(abs(float(seg[-1, 1]) - float(seg[0, 1])) / 2.0)
    return out


def _plotted_means(ax):
    import matplotlib.container as _mcont
    for cont in ax.containers:
        if isinstance(cont, _mcont.ErrorbarContainer):
            line = cont.lines[0] if cont.lines else None
            if line is not None:
                try:
                    return [float(v) for v in line.get_ydata()]
                except Exception:
                    return []
        if isinstance(cont, _mcont.BarContainer):
            try:
                return [float(v) for v in cont.datavalues]
            except Exception:
                return []
    return []


def _scatter_point_count(ax):
    import matplotlib.collections as _mcoll
    total = 0
    for coll in ax.collections:
        if isinstance(coll, _mcoll.PathCollection):
            try:
                total += len(coll.get_offsets())
            except Exception:
                pass
    return total


def _part_c():
    import matplotlib.figure as _mfig
    import matplotlib.axes as _maxes

    fig = _need("fig")
    ax = _need("ax")
    if not isinstance(fig, _mfig.Figure):
        return "`fig` should be a matplotlib Figure, from plt.subplots()."
    if not isinstance(ax, _maxes.Axes):
        return ("`ax` should be a single matplotlib Axes. plt.subplots() with no rows or "
                "columns argument gives you exactly one.")

    if not ax.containers and not ax.collections and not ax.lines:
        raise _Todo("the axes is still empty. Fill in the # YOUR CODE lines for part c.")

    yerr = sorted(_yerr_values(ax))
    if not yerr:
        return ("No error bars found on the axes. Draw the group means with ax.errorbar and "
                "pass yerr, rather than plotting the means as plain markers.")
    means = sorted(_plotted_means(ax))
    want_mean = sorted(float(v) for v in _EXP["group_mean"])
    if means and len(means) == len(want_mean) and not _arr_close(means, want_mean, rtol=1e-3):
        if _arr_close(means, sorted(float(v) for v in _EXP["group_mean_pooled"]), rtol=1e-3):
            return ("The plotted group means come from pooling every cell rather than "
                    "averaging the animal means, so part b needs fixing first.")
        return ("The plotted group means do not match the mean of the animal means. Plot "
                "the MEAN column of group_stats, and check part b first.")

    want_sd = sorted(float(v) for v in _EXP["group_sd"])
    want_sem = sorted(float(v) for v in _EXP["group_sem"])
    want_cell = sorted(float(v) for v in _EXP["group_sd_pooled"])
    uniq = sorted(set(round(v, 6) for v in yerr))
    if len(uniq) >= 2:
        pick = sorted(uniq)[-len(want_sd):] if len(uniq) > len(want_sd) else sorted(uniq)
    else:
        pick = uniq
    if len(pick) == len(want_sem) and _arr_close(pick, want_sem, rtol=1e-2):
        return ("The error bars are SEM, not SD. They are the SD divided by the square root "
                "of the number of animals. This repo always uses SD.")
    if len(pick) == len(want_cell) and _arr_close(pick, want_cell, rtol=1e-2):
        return ("The error bars are the SD of individual cells, not of animal means. Take "
                "yerr from the SD column of group_stats.")
    if not (len(pick) == len(want_sd) and _arr_close(pick, want_sd, rtol=1e-2)):
        return ("The error bar half heights do not match the SD column of group_stats. Pass "
                "yerr=group_stats[\\"SD\\"] and let errorbar handle the rest; a yerr of shape "
                "(2, n) would be read as separate lower and upper bars.")

    n_pts = _scatter_point_count(ax)
    if n_pts == 0:
        return ("No scatter points found. Overlay the individual animal means with "
                "ax.scatter, one point per animal. ax.plot does not register as a scatter "
                "for this check.")
    if n_pts == _EXP["n_cells_total"]:
        return ("The scatter has one point per cell. Overlay the animal means from "
                "animal_level, which is %d points, not every cell." % _EXP["n_animals"])
    if n_pts != _EXP["n_animals"]:
        return ("The scatter has %d points but there are %d animals, so there should be one "
                "point each." % (n_pts, _EXP["n_animals"]))

    if not ax.get_xlabel().strip():
        return "The x axis has no label. Use ax.set_xlabel."
    if not ax.get_ylabel().strip():
        return "The y axis has no label, and a reader cannot tell what the numbers are."
    if not ax.get_title().strip():
        return "The axes has no title. Use ax.set_title."

    try:
        fig.canvas.draw()
    except Exception:
        pass
    ticks = [float(t) for t in ax.get_xticks()]
    if len(ticks) != 2 or not _arr_close(sorted(ticks), [0.0, 1.0]):
        return ("The x ticks are not exactly [0, 1]. Call ax.set_xticks([0, 1]) so the two "
                "groups land where you plotted them.")
    labels = " ".join(t.get_text() for t in ax.get_xticklabels()).lower()
    for name in _EXP["group_names"]:
        if name.lower() not in labels:
            return ("The x tick labels do not name the groups. One of them should read %r. "
                    "Use ax.set_xticklabels with the group names." % name)
    return None
'''


TOOLKIT_MD = '''
## Toolkit for this problem

Two steps, in this order, always: **cells up to animals**, then **animals up
to groups**. Everything below runs on a tiny made up table, so none of it is
the answer.

| What you want to do | How |
|---|---|
| aggregate and name the result | `df.groupby("ID").agg(M=("VAL", "mean"), N=("VAL", "size"))` |
| turn group keys back into columns | `.reset_index()` |
| count rows in a group | the aggregation `"size"` |
| sample SD | the aggregation `"std"`, which pandas already does with `n - 1` |
| a figure with one axes | `fig, ax = plt.subplots(figsize=(6, 4.5))` |
| points with error bars | `ax.errorbar([0, 1], means, yerr=sds, fmt="o", capsize=6)` |
| scattered points on top | `ax.scatter(x, y)` |
| name the ticks | `ax.set_xticks([0, 1])` then `ax.set_xticklabels(["Drug", "Vehicle"])` |
| label the axes | `ax.set_xlabel(...)`, `ax.set_ylabel(...)`, `ax.set_title(...)` |

Why `yerr` must be the SD of the animal means, and not either of the other
two numbers that are easy to reach for:

- the SD of all the **cells** describes how much cells vary inside an animal,
  which is a microscope question, not a drug question,
- the **SEM** is that SD divided by the square root of n, so it shrinks when
  you image more cells, which makes a weak effect look convincing.
'''

TOOLKIT_CODE = '''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# A tiny made up table: 2 animals per arm, a handful of cells each, on purpose
# an unequal number of cells. Nothing here is the answer to the problem.
demo = pd.DataFrame({
    "ID":  ["a1", "a1", "a1", "a2", "a2", "b1", "b1", "b2", "b2", "b2"],
    "ARM": ["ctl", "ctl", "ctl", "ctl", "ctl", "trt", "trt", "trt", "trt", "trt"],
    "VAL": [10.0, 12.0, 14.0, 30.0, 34.0, 4.0, 6.0, 9.0, 11.0, 13.0],
})
print("demo, one row per cell:")
print(demo)

print("")
print("--- 1. cells up to animals ---")
demo_animal = (demo
               .groupby(["ID", "ARM"])
               .agg(MEAN_VAL=("VAL", "mean"), N_CELLS=("VAL", "size"))
               .reset_index())
print(demo_animal)
print("note: the aggregation names become the column names, so no rename is needed")

print("")
print("--- 2. animals up to groups ---")
demo_group = (demo_animal
              .groupby("ARM")
              .agg(MEAN=("MEAN_VAL", "mean"), SD=("MEAN_VAL", "std"), N_ANIMALS=("MEAN_VAL", "size"))
              .reset_index()
              .sort_values("ARM")
              .reset_index(drop=True))
print(demo_group)

print("")
print("--- 3. why the order matters ---")
demo_pooled = demo.groupby("ARM")["VAL"].agg(["mean", "std", "size"])
print("mean of the animal means :", [round(v, 3) for v in demo_group["MEAN"]])
print("mean of every cell pooled:", [round(v, 3) for v in demo_pooled["mean"]],
      " <- different, because the animals gave unequal numbers of cells")
print("")
print("SD across animals        :", [round(v, 3) for v in demo_group["SD"]], " <- what we plot")
print("SD across pooled cells   :", [round(v, 3) for v in demo_pooled["std"]], " <- a microscope question")
demo_sem = demo_group["SD"] / np.sqrt(demo_group["N_ANIMALS"])
print("SEM across animals       :", [round(v, 3) for v in demo_sem], " <- never plot this")
print("")
print("n if you count animals   :", list(demo_group["N_ANIMALS"]))
print("n if you count cells     :", list(demo_pooled["size"]), " <- the inflated n to avoid")

print("")
print("--- 4. the figure, built the same way you will build yours ---")
demo_fig, demo_ax = plt.subplots(figsize=(5, 3.5))
demo_order = list(demo_group["ARM"])
demo_ax.errorbar([0, 1], demo_group["MEAN"], yerr=demo_group["SD"],
                 fmt="o", color="black", capsize=6, linestyle="none", zorder=3)
demo_x = [demo_order.index(a) for a in demo_animal["ARM"]]
demo_ax.scatter(demo_x, demo_animal["MEAN_VAL"], s=40, alpha=0.75, zorder=2)
demo_ax.set_xticks([0, 1])
demo_ax.set_xticklabels(demo_order)
demo_ax.set_xlim(-0.5, 1.5)
demo_ax.set_xlabel("Made up arm")
demo_ax.set_ylabel("Made up value")
demo_ax.set_title("Toolkit demo, not the answer")
plt.show()
'''


CONCEPT_MD = '''
## How matplotlib works, and why the unit of replication matters

### The object model

Almost every confusing matplotlib example online comes from mixing two
different styles. There is `plt.plot(...)`, which draws on whatever figure
happens to be current, a hidden piece of state you cannot see. And there is
the explicit style, where you hold the objects yourself:

```python
fig, ax = plt.subplots(figsize=(6, 4.5))
```

- **`fig`** is the whole picture: the canvas, its size, and what gets saved.
- **`ax`** is one set of axes inside it: the data area, its ticks, its labels,
  and the things drawn on it.

You then tell `ax` what to draw, by calling methods on it. `ax.errorbar(...)`,
`ax.scatter(...)`, `ax.set_ylabel(...)`. Everything is a method on the thing
it affects, which means you can always find the right call by asking "which
object does this belong to". It also means a figure with several panels is no
harder: `plt.subplots(1, 2)` hands you an array of axes, and each one is
driven exactly the same way. Use the explicit style always. The hidden-state
style falls apart the moment a notebook has more than one figure.

Drawing order is controlled by `zorder`, higher being nearer the front, which
is how you keep scattered points from hiding your summary markers.

### The statistics this plot is really about

**Replication is about independence, not sample size.** Two measurements count
as replicates only if they are independent draws from the thing you are
studying. You are studying a drug effect in animals, so two animals are two
replicates. Fifty cells from one animal are fifty measurements of **that one
animal**, sharing its genetics, its dose, its exposure, its slide and its
staining batch. They tell you about variation between cells, which is a real
and sometimes interesting quantity, but it is not the quantity your group
comparison is about.

Treating cells as n does three separate kinds of damage:

1. **n is inflated**, here from 12 to several hundred, so confidence intervals
   collapse and almost anything looks significant. This has a name,
   pseudoreplication, and it is one of the most common fatal flaws in cell
   biology papers.
2. **Animals get unequal weight.** Pooling means the animal you imaged 60
   cells from counts twelve times as much as the one you imaged 5 from. The
   group mean drifts toward whichever animal you had most patience with, which
   is a property of your afternoon, not of the drug.
3. **The error bar answers the wrong question.** The SD of pooled cells mostly
   measures cell to cell variation and staining noise. A reader looking at
   your figure wants to know how reproducible the effect is **between
   animals**, because that is what predicts whether it replicates.

The fix is the two-step aggregation you are about to write: cells up to one
number per animal, then animals up to a group mean and SD. After step one,
every animal is one row, which is exactly the shape every standard statistical
test assumes.

**And the SD versus SEM choice.** SEM is the SD divided by the square root of
n, so it shrinks as you add animals even when the biology is identical. It
describes how precisely you have located the mean, not how much the animals
differ. For a figure whose job is to show a biological effect, SD is the
honest choice, and a reader who knows n can always work out the SEM. A reader
shown SEM cannot work backwards without being told n.
'''


# Functions this problem introduces. Keys must exist in tools/function_catalog.py.
FUNCTIONS = [
    'plt.subplots',
    'ax.errorbar',
    'ax.scatter',
    'zorder',
    'axis labels',
    'ax.set_xticks',
    'ax.legend',
    'plt.show',
    'np.where',
    'np.sqrt',
]
