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

### Part a. Aggregate the cells up to the animal
Build `animal_level`: one row per animal, with exactly these columns, named
exactly:

| column | meaning |
|---|---|
| `ANIMAL` | animal identifier |
| `GROUP` | `Drug` or `Vehicle` |
| `MEAN_INTENSITY` | mean of that animal's cells |
| `N_CELLS` | how many cells were measured in that animal |

Flat DataFrame, so `ANIMAL` and `GROUP` are columns and not an index.

### Part b. Group statistics across animals
Build `group_stats` **from `animal_level`**, one row per group, with exactly:

| column | meaning |
|---|---|
| `GROUP` | `Drug` or `Vehicle` |
| `MEAN` | mean of the animal means in that group |
| `SD` | **sample** SD of the animal means in that group |
| `N_ANIMALS` | number of animals in that group |

Sort it by `GROUP`, which puts `Drug` first. Note that the mean of the animal
means is **not** the mean of all the cells, because the animals contributed
unequal numbers of cells.

### Part c. The figure
Build a single axes figure, assigned to `fig` and `ax`:

- `fig, ax = plt.subplots(figsize=(6, 4.5))`
- plot the two group means with `ax.errorbar`, at x positions `0` and `1`, in
  the row order of `group_stats`, with `yerr` set to the `SD` column and
  `fmt="o"`,
- overlay **every animal's own mean** with `ax.scatter`, one point per animal,
  at the x position of its group. A little jitter on x is welcome but not
  required,
- set the x ticks to `[0, 1]` and label them with the group names,
- set an x label, a y label and a title.

### Hints
- `df.groupby([...]).agg(MEAN=("COL", "mean"), N=("COL", "size"))` lets you
  name the output columns as you build them, then `.reset_index()` flattens.
- `"size"` counts rows, `"std"` in pandas already uses `n - 1`.
- `ax.errorbar([0, 1], means, yerr=sds, fmt="o", capsize=5)`.
- For the jitter: `x = np.where(animal_level["GROUP"] == "Drug", 0, 1) +
  rng.normal(0, 0.04, len(animal_level))`.
- `ax.set_xticks([0, 1])` then `ax.set_xticklabels([...])`.
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
### Part a. Aggregate the cells up to the animal
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
### Part b. Group statistics across animals

Build this from `animal_level`, so that n is the number of animals.
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
### Part c. The figure
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
