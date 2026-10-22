PID = "W2D5"
WEEK = 2
DAY = 5
SLUG = "w2d5_matplotlib_panels_semilog"
TITLE = "Two panels and a spaghetti plot: the figure every PK report opens with"
TOPIC = "matplotlib: subplots, semilog axes, per subject overlays"
EST = "20-30 min"

SCENARIO_MD = '''
## Scenario

Sixteen subjects, eight on each of two dose levels, sampled seven times over
24 hours. Your group leader wants the standard first look at a new dataset,
which is always the same two figures.

The first is the **mean profile**, shown twice: once on a linear y axis, which
is how a clinician reads exposure, and once on a log y axis, which is where a
pharmacometrician can see whether elimination is first order and read the
slope by eye. Same data, same panel size, side by side.

The second is the **spaghetti plot**: every subject's own profile, drawn
individually, no averaging at all. It is the figure that tells you whether the
mean means anything, or whether it is an average of two populations, or is
being dragged around by one outlier.

Between them these two figures answer "what does this drug do" and "do I
believe the summary", which is why they open every report.
'''

CONCEPT_MD = '''
## Panels, log axes, and why the spaghetti plot comes first

**Several panels from one call.** `plt.subplots(1, 2)` returns the figure and
an **array** of axes, which you index: `axes[0]` and `axes[1]`. With no rows
and columns argument you get a single axes which you must not index. Getting
this wrong gives either `TypeError: 'Axes' object is not subscriptable` or an
`AttributeError` when you call a plotting method on a numpy array, and both
mean you disagree with matplotlib about how many panels you asked for.

`figsize` is the whole figure in inches, not each panel, so a two panel figure
wants roughly double the width of a one panel figure or the panels come out
squashed. `fig.tight_layout()` at the end stops the labels of one panel
colliding with the next.

**Why the log axis matters here.** First order elimination means a constant
*fraction* leaves per hour, which is exponential decay. Take the log of an
exponential and you get a straight line. So on a semilog axis the terminal
phase of a PK profile becomes straight, and you can see at a glance whether
one compartment is enough, whether there is a second slower phase, and roughly
what the half-life is. On a linear axis all of that is hidden in a curve that
hugs the x axis.

The trap: a log axis cannot show zero or negative values. They do not error,
they silently vanish, taking your predose samples with them. If a profile
looks like it starts late, that is usually why.

**Why the spaghetti plot comes first in practice.** A mean with an SD bar
describes the group as though every subject behaves like the average one with
some scatter around it. Plot the individuals and you find out whether that is
true. Two subpopulations, one subject who did not absorb, a sample swapped
between two people: none of these show up in a mean, and all of them are
obvious in thirty seconds of looking at the lines. Draw it before you believe
any summary statistic you computed, not after.

Keep the individual lines visually quiet: thin, partly transparent, one
colour per group rather than sixteen different colours. The reader is meant to
see the shape of the bundle, not trace any one line.
'''

TASK_MD = '''
## Your task

Three parts, worked one at a time.

| Part | What you work out | Variables to fill in |
|---|---|---|
| a | the mean and SD profile per dose group | `summary` |
| b | a two panel figure, linear and semilog | `fig`, `axes` |
| c | a spaghetti plot of every subject | `fig2`, `ax2` |

Below this you will find the **Setup** cell, then a **Toolkit** cell that
demonstrates every move on a tiny made up example, then **Part a**, **Part b**
and **Part c**, each with its own explanation and a cell to write in.
'''

DATA_CODE = '''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROBLEM_ID = "W2D5"
TOPIC = "matplotlib: subplots, semilog axes, per subject overlays"

rng = np.random.default_rng(2105)

TIMES = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0])
DOSE_ORDER = ["50 mg", "200 mg"]
SUBJECTS = ["D%02d" % i for i in range(1, 17)]
_group = np.repeat(DOSE_ORDER, 8)
_dose = np.repeat([50.0, 200.0], 8)

_V = 44.0 * rng.lognormal(0.0, 0.24, size=len(SUBJECTS))
_kel = 0.115 * rng.lognormal(0.0, 0.26, size=len(SUBJECTS))
_ka = 1.05 * rng.lognormal(0.0, 0.20, size=len(SUBJECTS))
_coef = (0.86 * _dose * _ka) / (_V * (_ka - _kel))
_true = _coef[:, None] * (np.exp(-_kel[:, None] * TIMES[None, :])
                          - np.exp(-_ka[:, None] * TIMES[None, :]))

pk = pd.DataFrame({
    "SUBJECT": np.repeat(SUBJECTS, TIMES.size),
    "DOSE": np.repeat(_group, TIMES.size),
    "TIME_H": np.tile(TIMES, len(SUBJECTS)),
    "CONC_MG_L": np.round(_true.ravel() * rng.lognormal(0.0, 0.13, size=_true.size), 4),
})

print("pk:", pk.shape)
print(pk.head())
print("")
print("subjects per dose:", pk.groupby("DOSE")["SUBJECT"].nunique().to_dict())
'''

TOOLKIT_MD = '''
## Toolkit for this problem

| What you want to do | How |
|---|---|
| two panels side by side | `fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))` |
| one panel | `fig, ax = plt.subplots(figsize=(6, 4.5))`, and do not index it |
| points with SD bars | `ax.errorbar(x, y, yerr=sd, fmt="o-", capsize=4, label=...)` |
| a log y axis | `ax.set_yscale("log")` |
| one line per subject | loop over groups, `ax.plot(x, y, lw=0.9, alpha=0.55)` |
| split a frame into groups | `for name, part in df.groupby("DOSE"):` |
| stop labels colliding | `fig.tight_layout()` |
| a legend | pass `label=` to each call, then `ax.legend()` |

Looping over `df.groupby("COL")` hands you the group name and that group's
rows on each turn, which is usually tidier than filtering by hand.
'''

TOOLKIT_CODE = '''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# A tiny made up dataset. Nothing here is the answer.
demo_t = np.array([1.0, 2.0, 4.0, 8.0])
demo = pd.DataFrame({
    "S": np.repeat(["s1", "s2", "s3", "s4"], 4),
    "G": np.repeat(["lo", "lo", "hi", "hi"], 4),
    "T": np.tile(demo_t, 4),
    "C": [2.0, 1.6, 1.0, 0.4, 2.4, 1.8, 1.1, 0.5,
          8.0, 6.2, 3.9, 1.5, 9.1, 7.0, 4.4, 1.8],
})
print("demo:");  print(demo.head())

print("")
print("--- 1. looping over groups ---")
for demo_name, demo_part in demo.groupby("G"):
    print("group", demo_name, "has", len(demo_part), "rows and",
          demo_part["S"].nunique(), "subjects")

print("")
print("--- 2. one panel versus two ---")
demo_fig1, demo_ax1 = plt.subplots(figsize=(4, 2.6))
print("plt.subplots() gives:", type(demo_ax1).__name__, " <- do NOT index this")
plt.close(demo_fig1)
demo_fig2, demo_axes2 = plt.subplots(1, 2, figsize=(8, 2.6))
print("plt.subplots(1, 2) gives:", type(demo_axes2).__name__,
      "of length", len(demo_axes2), " <- index it as axes[0], axes[1]")
plt.close(demo_fig2)

print("")
print("--- 3. the same data on linear and log y ---")
demo_summary = (demo.groupby(["G", "T"])
                .agg(M=("C", "mean"), S=("C", "std"))
                .reset_index())
demo_fig, demo_axes = plt.subplots(1, 2, figsize=(9, 3.4))
for demo_name, demo_part in demo_summary.groupby("G"):
    for demo_ax in demo_axes:
        demo_ax.errorbar(demo_part["T"], demo_part["M"], yerr=demo_part["S"],
                         fmt="o-", capsize=4, label=demo_name)
demo_axes[1].set_yscale("log")
demo_axes[0].set_title("Linear")
demo_axes[1].set_title("Semi log")
for demo_ax in demo_axes:
    demo_ax.set_xlabel("Time (h)")
    demo_ax.set_ylabel("Made up units")
demo_axes[0].legend(frameon=False)
demo_fig.tight_layout()
plt.show()
print("on the log panel the decay is straighter, which is what first order looks like")

print("")
print("--- 4. a spaghetti plot, one line per subject ---")
demo_fig3, demo_ax3 = plt.subplots(figsize=(5, 3.2))
demo_colour = {"lo": "tab:blue", "hi": "tab:red"}
for demo_sid, demo_rows in demo.groupby("S"):
    demo_g = demo_rows["G"].iloc[0]
    demo_ax3.plot(demo_rows["T"], demo_rows["C"], lw=0.9, alpha=0.6,
                  color=demo_colour[demo_g])
demo_ax3.set_yscale("log")
demo_ax3.set_xlabel("Time (h)")
demo_ax3.set_ylabel("Made up units")
demo_ax3.set_title("Every subject, no averaging")
plt.show()
print("lines drawn on that axes:", len(demo_ax3.lines), " <- one per subject")
'''

PARTS = [
    dict(
        label="a",
        prompt_md='''
## Part a. The mean and SD profile per dose group

Before any figure, the numbers it will be built from.

One row per `DOSE` and `TIME_H`, with the mean concentration, its sample SD
across subjects, and the number of subjects contributing. The count should be
8 in every cell; if it is not, something upstream lost rows and the figure
would quietly misrepresent it.

SD, never SEM. Pandas `std` already uses `n - 1`.

Fill in `summary`, a flat DataFrame with exactly:

| column | meaning |
|---|---|
| `DOSE` | `50 mg` or `200 mg` |
| `TIME_H` | hours after dose |
| `CONC_MEAN` | mean concentration at that time |
| `CONC_SD` | sample SD across subjects |
| `N` | how many subjects contributed |

Moves: `.groupby([...]).agg(NAME=("COL", "mean"), ...)`, `.reset_index()`.
''',
        starter='''
# Part a. Mean, SD and n per dose group per timepoint.

summary = None   # YOUR CODE

print(summary)
''',
        solution='''
# Part a. Mean, SD and n per dose group per timepoint.

summary = (pk
           .groupby(["DOSE", "TIME_H"])
           .agg(CONC_MEAN=("CONC_MG_L", "mean"),
                CONC_SD=("CONC_MG_L", "std"),
                N=("CONC_MG_L", "size"))
           .reset_index())

print(summary)
''',
    ),
    dict(
        label="b",
        prompt_md='''
## Part b. Two panels, linear and semilog

One figure, two panels side by side, showing the **same** mean profiles twice.

Build it with `plt.subplots(1, 2)`, which hands you an array of two axes.
Loop over the dose groups in `summary` and draw each group on **both** panels
with `ax.errorbar`, using the `CONC_SD` column as `yerr` and a label so the
legend knows what it is looking at.

Then make the right panel logarithmic in y. Everything else about the two
panels is identical: that is the point, since the only thing changing is how
the eye reads the same numbers.

Label both panels on both axes and give each a title, so the figure survives
being pasted into a slide on its own. Put a legend on at least one panel, and
finish with `fig.tight_layout()`.

Fill in `fig` and `axes`, where `axes` holds the two panels, `axes[0]` linear
and `axes[1]` logarithmic in y.

Moves: `plt.subplots(1, 2, figsize=...)`, `for name, part in
summary.groupby("DOSE")`, `ax.errorbar(..., yerr=..., label=...)`,
`ax.set_yscale("log")`, `ax.set_xlabel`, `ax.set_ylabel`, `ax.set_title`,
`ax.legend()`, `fig.tight_layout()`.
''',
        starter='''
# Part b. Two panels: the same mean profiles, linear on the left, log y on the right.

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

# YOUR CODE: loop over the dose groups in summary and errorbar each one onto BOTH panels,
#            with yerr taken from CONC_SD and a label for the legend

# YOUR CODE: make axes[1] logarithmic in y

# YOUR CODE: xlabel, ylabel and title on both panels, and a legend on at least one

fig.tight_layout()
plt.show()
''',
        solution='''
# Part b. Two panels: the same mean profiles, linear on the left, log y on the right.

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

for _name, _part in summary.groupby("DOSE"):
    for _ax in axes:
        _ax.errorbar(_part["TIME_H"], _part["CONC_MEAN"], yerr=_part["CONC_SD"],
                     fmt="o-", capsize=4, lw=1.4, label=_name)

axes[1].set_yscale("log")

axes[0].set_title("Mean profile, linear")
axes[1].set_title("Mean profile, semi log")
for _ax in axes:
    _ax.set_xlabel("Time after dose (h)")
    _ax.set_ylabel("Concentration (mg/L)")
axes[0].legend(frameon=False, title="Dose")

fig.tight_layout()
plt.show()
''',
    ),
    dict(
        label="c",
        prompt_md='''
## Part c. The spaghetti plot

A single panel, one line per subject, no averaging anywhere.

Loop over subjects and draw each one's concentration against time. Colour by
dose group rather than by subject, so the two bundles separate without
sixteen competing colours, and keep the lines thin and partly transparent so
the reader sees the shape of each bundle rather than trying to follow
individual lines.

Use a log y axis here too. On a linear axis the low dose bundle is squashed
into the bottom of the panel and you cannot see its shape at all.

Draw **exactly one line per subject and nothing else on this axes**, so there
should be sixteen lines when you are done. No mean, no error bars. That is
what makes this figure honest: it is the raw data and nothing else.

Fill in `fig2` and `ax2`, a single axes with one line per subject, a log y
axis, and all three labels.

Moves: `plt.subplots(figsize=...)`, `for sid, rows in pk.groupby("SUBJECT")`,
`ax.plot(..., lw=..., alpha=..., color=...)`, `ax.set_yscale("log")`,
`ax.set_xlabel`, `ax.set_ylabel`, `ax.set_title`.
''',
        starter='''
# Part c. Spaghetti plot: every subject, no averaging.

fig2, ax2 = plt.subplots(figsize=(6.5, 4.5))

colours = {"50 mg": "tab:blue", "200 mg": "tab:red"}

# YOUR CODE: loop over subjects and plot each one's profile, thin and semi transparent,
#            coloured by that subject's dose group

# YOUR CODE: log y axis

# YOUR CODE: xlabel, ylabel and title

plt.show()
''',
        solution='''
# Part c. Spaghetti plot: every subject, no averaging.

fig2, ax2 = plt.subplots(figsize=(6.5, 4.5))

colours = {"50 mg": "tab:blue", "200 mg": "tab:red"}

for _sid, _rows in pk.groupby("SUBJECT"):
    _dose = _rows["DOSE"].iloc[0]
    ax2.plot(_rows["TIME_H"], _rows["CONC_MG_L"],
             lw=0.9, alpha=0.6, color=colours[_dose])

ax2.set_yscale("log")

ax2.set_xlabel("Time after dose (h)")
ax2.set_ylabel("Concentration (mg/L)")
ax2.set_title("Every subject, coloured by dose")

plt.show()
''',
    ),
]


def build_expected(ns):
    import numpy as np
    pk = ns["pk"]
    s = ns["summary"].sort_values(["DOSE", "TIME_H"]).reset_index(drop=True)
    counts = (pk.groupby(["DOSE", "TIME_H"])["CONC_MG_L"].count()
              .reset_index().sort_values(["DOSE", "TIME_H"])["CONC_MG_L"].to_numpy())
    sd_pop = (pk.groupby(["DOSE", "TIME_H"])["CONC_MG_L"].std(ddof=0)
              .reset_index().sort_values(["DOSE", "TIME_H"])["CONC_MG_L"].to_numpy())
    return {
        "summary_cols": ["DOSE", "TIME_H", "CONC_MEAN", "CONC_SD", "N"],
        "summary_n": int(len(s)),
        "summary_dose": [str(v) for v in s["DOSE"]],
        "summary_time": [float(v) for v in s["TIME_H"]],
        "summary_mean": [float(v) for v in s["CONC_MEAN"]],
        "summary_sd": [float(v) for v in s["CONC_SD"]],
        "summary_sem": [float(v) for v in s["CONC_SD"].to_numpy() / np.sqrt(counts)],
        "summary_sd_pop": [float(v) for v in sd_pop],
        "summary_count": [int(v) for v in s["N"]],
        "n_subjects": int(pk["SUBJECT"].nunique()),
        "n_groups": int(pk["DOSE"].nunique()),
        "n_times": int(pk["TIME_H"].nunique()),
    }


CHECK_CODE = '''
def _yerr_values(ax):
    import matplotlib.container as _mcont
    out = []
    for cont in ax.containers:
        eb = cont if isinstance(cont, _mcont.ErrorbarContainer) else None
        if eb is None and isinstance(cont, _mcont.BarContainer):
            eb = getattr(cont, "errorbar", None)
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


def _part_a():
    s = _need("summary")
    if not isinstance(s, _pd.DataFrame):
        return "`summary` should be a pandas DataFrame."
    cols = _cols_of(s) or []
    for name in ("DOSE", "TIME_H"):
        if name not in cols and name in [str(n) for n in (s.index.names or [])]:
            return "`%s` is in the index rather than a column. Call .reset_index()." % name
    missing = [c for c in _EXP["summary_cols"] if c not in cols]
    if missing:
        return "`summary` is missing the column(s) %s." % ", ".join(missing)
    if len(s) != _EXP["summary_n"]:
        return ("`summary` has %d rows but there should be one per dose group and "
                "timepoint, so %d." % (len(s), _EXP["summary_n"]))

    got = s.sort_values(["DOSE", "TIME_H"]).reset_index(drop=True)
    if [str(v) for v in got["DOSE"]] != _EXP["summary_dose"]:
        return "`summary` does not cover the expected dose groups."
    if not _arr_close(got["TIME_H"].to_numpy(), _EXP["summary_time"]):
        return "`summary` does not cover the expected timepoints."
    if not _arr_close(got["N"].to_numpy(), _EXP["summary_count"]):
        return ("`N` is not the number of subjects per cell. It should be 8 everywhere; "
                "the aggregation 'size' counts rows within the group.")
    if not _arr_close(got["CONC_MEAN"].to_numpy(), _EXP["summary_mean"]):
        return "`CONC_MEAN` is not the mean concentration within each dose and timepoint."
    if _arr_close(got["CONC_SD"].to_numpy(), _EXP["summary_sem"]):
        return ("`CONC_SD` is the SEM, not the SD. Something divided by the square root of "
                "n. This repo always reports SD.")
    if _arr_close(got["CONC_SD"].to_numpy(), _EXP["summary_sd_pop"]):
        return ("`CONC_SD` used n in the denominator. Pandas std already uses n-1, so this "
                "usually means numpy's std got involved.")
    if not _arr_close(got["CONC_SD"].to_numpy(), _EXP["summary_sd"]):
        return "`CONC_SD` is not the sample SD within each dose and timepoint."
    return None


def _part_b():
    import matplotlib.figure as _mfig
    import matplotlib.axes as _maxes

    fig = _need("fig")
    axes = _need("axes")
    if not isinstance(fig, _mfig.Figure):
        return "`fig` should be a matplotlib Figure from plt.subplots()."

    try:
        axlist = list(_np.ravel(_np.asarray(axes, dtype=object)))
    except Exception:
        axlist = [axes]
    if isinstance(axes, _maxes.Axes):
        return ("`axes` is a single Axes, so plt.subplots() was called without asking for "
                "two panels. Use plt.subplots(1, 2).")
    if len(axlist) != 2:
        return ("`axes` holds %d panels but the figure should have exactly 2." % len(axlist))
    if not all(isinstance(a, _maxes.Axes) for a in axlist):
        return "`axes` should hold matplotlib Axes objects."

    empty = [i for i, a in enumerate(axlist)
             if not a.containers and not a.lines and not a.collections]
    if len(empty) == 2:
        raise _Todo("both panels are still empty. Fill in the # YOUR CODE lines for part b.")
    if empty:
        return ("Panel %d has nothing drawn on it. Both panels show the SAME mean profiles; "
                "only the y scale differs." % empty[0])

    want_sd = sorted(float(v) for v in _EXP["summary_sd"])
    want_sem = sorted(float(v) for v in _EXP["summary_sem"])
    for i, ax in enumerate(axlist):
        yerr = sorted(set(round(v, 9) for v in _yerr_values(ax)))
        if not yerr:
            return ("Panel %d has no error bars. Draw the means with ax.errorbar and pass "
                    "yerr, on both panels." % i)
        if len(yerr) == len(want_sem) and _arr_close(yerr, want_sem, rtol=1e-2):
            return ("Panel %d has SEM error bars, not SD. This repo always uses SD, so pass "
                    "the CONC_SD column as yerr." % i)
        if not (len(yerr) == len(want_sd) and _arr_close(yerr, want_sd, rtol=1e-2)):
            return ("Panel %d's error bars do not match the CONC_SD column of summary. Both "
                    "panels should show the same numbers." % i)

    scales = [a.get_yscale() for a in axlist]
    if scales[1] != "log":
        if scales[0] == "log":
            return ("The LEFT panel is logarithmic and the right one is linear. The task "
                    "wants axes[0] linear and axes[1] logarithmic.")
        return ("Neither panel has a log y axis. Call axes[1].set_yscale('log').")
    if scales[0] == "log":
        return "Both panels are logarithmic. axes[0] should stay linear for the comparison."

    for i, ax in enumerate(axlist):
        if not ax.get_xlabel().strip():
            return "Panel %d has no x label. Label both panels." % i
        if not ax.get_ylabel().strip():
            return "Panel %d has no y label, so a reader cannot tell what the numbers are." % i
        if not ax.get_title().strip():
            return "Panel %d has no title. Give each panel one, so the figure stands alone." % i

    if not any(a.get_legend() is not None for a in axlist):
        return ("Neither panel has a legend, so nothing says which line is which dose. Pass "
                "label= to each errorbar call, then call ax.legend() on one panel.")
    return None


def _part_c():
    import matplotlib.figure as _mfig
    import matplotlib.axes as _maxes

    fig2 = _need("fig2")
    ax2 = _need("ax2")
    if not isinstance(fig2, _mfig.Figure):
        return "`fig2` should be a matplotlib Figure."
    if not isinstance(ax2, _maxes.Axes):
        return ("`ax2` should be a single Axes. plt.subplots() with no rows or columns "
                "argument gives exactly one.")

    if not ax2.lines and not ax2.collections and not ax2.containers:
        raise _Todo("the axes is still empty. Fill in the # YOUR CODE lines for part c.")

    n_lines = len(ax2.lines)
    n_sub = _EXP["n_subjects"]
    if ax2.containers:
        return ("There are error bars on this axes. The spaghetti plot is raw data only, "
                "one plain line per subject and nothing else.")
    if n_lines == _EXP["n_groups"]:
        return ("Only %d lines were drawn, which is one per dose group rather than one per "
                "subject. Loop over SUBJECT, not DOSE." % n_lines)
    if n_lines == n_sub + _EXP["n_groups"]:
        return ("There are %d lines, which is one per subject plus a group mean. Draw the "
                "subjects only; the means are the other figure." % n_lines)
    if n_lines != n_sub:
        return ("There are %d lines on the axes but there are %d subjects, so there should "
                "be exactly that many." % (n_lines, n_sub))

    for line in ax2.lines:
        if len(line.get_xdata()) != _EXP["n_times"]:
            return ("One of the lines has %d points but each subject has %d timepoints. "
                    "Plot each subject's own rows."
                    % (len(line.get_xdata()), _EXP["n_times"]))

    if ax2.get_yscale() != "log":
        return ("The y axis is not logarithmic. Call ax2.set_yscale('log'), or the low dose "
                "bundle is squashed into the bottom of the panel.")
    if not ax2.get_xlabel().strip():
        return "The x axis has no label."
    if not ax2.get_ylabel().strip():
        return "The y axis has no label, so a reader cannot tell what the numbers are."
    if not ax2.get_title().strip():
        return "The axes has no title."
    return None
'''

# Functions this problem introduces. Keys must exist in tools/function_catalog.py.
FUNCTIONS = [
    "looping over groupby",
    "ax.set_yscale",
    "plt.subplots(rows, cols)",
    "fig.tight_layout",
]
