"""Build reference/foundations.ipynb, the from the ground up reference notebook.

    python3 tools/build_foundations.py

This one is not a problem. It has no blanks and no check(). It is meant to sit
open in a second tab while you work on the problem of the day.
"""

import os
import pathlib
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "reference" / "foundations.ipynb"
COLAB = ("https://colab.research.google.com/github/dc00086-glitch/"
         "internship-prep/blob/main/reference/foundations.ipynb")

CELLS = []


def md(text):
    CELLS.append(new_markdown_cell(text.strip()))


def code(text):
    CELLS.append(new_code_cell(text.strip() + "\n"))


md('''
# Foundations

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](%s)

Everything the problem notebooks assume, from the ground up. Nothing here is
the answer to any problem.

Use it two ways:

- **Before you start**, run it top to bottom once. It takes a few minutes and
  every cell prints what it did.
- **While you work**, keep it open in a second browser tab and jump to the
  section you need.

Sections

1. Python essentials
2. numpy, the array library
3. pandas, the table library
4. matplotlib, the plotting library
5. The two house rules, shown with numbers
6. Error messages, and what they actually mean
''' % COLAB)

# ---------------------------------------------------------------- python
md('''
## 1. Python essentials

Only the parts you need for these problems.
''')

code('''
# Variables hold values. No declaration, no types to write down.
dose_mg = 100.0
drug_name = "example"
n_subjects = 8

print(dose_mg, drug_name, n_subjects)
print("types:", type(dose_mg), type(drug_name), type(n_subjects))

# f-strings put values inside text. Handy for printing.
print(f"{drug_name}: {dose_mg} mg in {n_subjects} subjects")

# round() for display, and the : .2f form inside an f-string
half_life = 7.48219
print("rounded:", round(half_life, 2))
print(f"formatted: {half_life:.2f} h")
''')

code('''
# A list holds things in order. Counting starts at 0.
times = [0.0, 1.0, 2.0, 4.0, 8.0]

print("the list   :", times)
print("first      :", times[0])
print("third      :", times[2])
print("last       :", times[-1], "   <- negative counts from the end")
print("last two   :", times[-2:])
print("first three:", times[:3], "  <- up to but NOT including position 3")
print("how many   :", len(times))
''')

code('''
# A loop does something once per item.
for t in [0.0, 1.0, 2.0]:
    print("time is", t)

print("")

# A function packages up something you do more than once.
def half_life_from_k(k):
    """Turn an elimination rate constant into a half-life."""
    import math
    return math.log(2) / k

print("half life when k = 0.0866:", round(half_life_from_k(0.0866), 2), "h")

# A dictionary maps names to values.
params = {"CL": 5.2, "V": 35.0}
print("params    :", params)
print("params['V']:", params["V"])
''')

# ---------------------------------------------------------------- numpy
md('''
## 2. numpy, the array library

A numpy array is a list of numbers that you can do maths on all at once. That
last part is the whole point: with a list you need a loop, with an array you
do not.

By convention it is always imported as `np`.
''')

code('''
import numpy as np

# Make an array from a list.
conc = np.array([0.0, 8.2, 12.5, 9.1, 4.3])
time_h = np.array([0.0, 1.0, 2.0, 4.0, 8.0])

print("conc      :", conc)
print("type      :", type(conc))
print("shape     :", conc.shape, "  <- a 1D array of 5 values")
print("how many  :", conc.size)
print("dtype     :", conc.dtype, "  <- every entry is the same kind of number")
''')

code('''
# Maths happens to every entry at once. No loop.
print("conc            :", conc)
print("conc * 2        :", conc * 2)
print("conc / 1000     :", conc / 1000)
print("conc + conc     :", conc + conc)
print("np.log(conc + 1):", np.round(np.log(conc + 1), 3))

# Some arrays you can make from scratch
print("")
print("np.zeros(4)     :", np.zeros(4))
print("np.arange(0,5)  :", np.arange(0, 5))
print("np.linspace(0,1,5):", np.linspace(0, 1, 5), " <- 5 evenly spaced values")
''')

code('''
# Indexing and slicing work like lists.
print("conc        :", conc)
print("conc[0]     :", conc[0])
print("conc[-1]    :", conc[-1])
print("conc[1:3]   :", conc[1:3], "  <- positions 1 and 2, not 3")
print("conc[-3:]   :", conc[-3:], "  <- the last three")
''')

code('''
# Comparing an array gives you an array of True and False.
mask = conc > 5
print("conc       :", conc)
print("conc > 5   :", mask)

# That True/False array can then PICK ENTRIES OUT of any array the same length.
print("conc[mask] :", conc[mask], "          <- only the values above 5")
print("time[mask] :", time_h[mask], "   <- the times those came from")

# ~ flips it, np.sum counts it, because True counts as 1.
print("~mask      :", ~mask)
print("how many True:", int(np.sum(mask)))
''')

code('''
# nan means "no number here". It shows up when a sample is below the assay limit.
with_gap = np.array([5.0, 9.0, np.nan, 6.0, 2.0])

print("with_gap          :", with_gap)
print("np.isnan          :", np.isnan(with_gap), " <- True where the value is MISSING")
print("flip it with ~    :", ~np.isnan(with_gap), " <- True where it is USABLE")

# nan poisons ordinary maths, which is usually what you want: it is a warning.
print("")
print("np.max(with_gap)   :", np.max(with_gap), "  <- nan, because one value is missing")
print("np.nanmax(with_gap):", np.nanmax(with_gap), "  <- the nan-aware version")
print("or drop them first :", np.max(with_gap[~np.isnan(with_gap)]))
''')

code('''
# The summaries you will reach for constantly.
vals = np.array([4.0, 8.0, 15.0, 16.0, 23.0, 42.0])

print("vals        :", vals)
print("np.max      :", np.max(vals))
print("np.min      :", np.min(vals))
print("np.mean     :", np.mean(vals))
print("np.median   :", np.median(vals))
print("np.sum      :", np.sum(vals))
print("")
print("np.argmax   :", np.argmax(vals), "  <- the POSITION of the biggest, not the value")
print("vals[np.argmax(vals)]:", vals[np.argmax(vals)], " <- feed the position back in")
print("")
print("np.std(vals)         :", round(float(np.std(vals)), 4), "  <- divides by n")
print("np.std(vals, ddof=1) :", round(float(np.std(vals, ddof=1)), 4), "  <- divides by n-1, the SAMPLE SD")
''')

md('''
### 2D arrays and the axis argument

A 2D array is a rectangle: rows and columns. In these problems rows are
usually subjects or animals, and columns are usually timepoints.

The rule to memorise: **the axis you name is the one that disappears.**

- `axis=0` collapses the rows, leaving one value per **column**
- `axis=1` collapses the columns, leaving one value per **row**
''')

code('''
grid = np.array([[1.0, 2.0, 3.0],
                 [4.0, 5.0, 9.0]])

print("grid:")
print(grid)
print("shape:", grid.shape, "  <- (rows, columns), so 2 rows and 3 columns")

print("")
print("grid.mean()       :", grid.mean(), "        <- no axis, so everything collapses to one number")
print("grid.mean(axis=0) :", grid.mean(axis=0), "  <- rows gone, one value per COLUMN")
print("grid.mean(axis=1) :", grid.mean(axis=1), "       <- columns gone, one value per ROW")
print("")
print("grid.max(axis=1)  :", grid.max(axis=1), "       <- biggest in each row")
print("grid.std(axis=0, ddof=1):", np.round(grid.std(axis=0, ddof=1), 3))
''')

code('''
# Broadcasting: doing maths between arrays of different shapes.
row_max = grid.max(axis=1)
print("row_max        :", row_max, " shape", row_max.shape)

# This fails, because (2,3) and (2,) do not line up.
try:
    grid / row_max
except ValueError as err:
    print("grid / row_max fails:", err)

# keepdims leaves the collapsed axis in place as a length 1 axis, which lines up.
row_max_col = grid.max(axis=1, keepdims=True)
print("")
print("with keepdims  :", row_max_col.tolist(), " shape", row_max_col.shape)
print("grid / that:")
print(grid / row_max_col, "  <- every row now peaks at 1.0")
''')

code('''
# Fitting a straight line, which is how a terminal half-life gets estimated.
x = np.array([4.0, 6.0, 8.0, 12.0])
y = np.array([2.2, 1.4, 1.2, 1.0])

slope, intercept = np.polyfit(x, np.log(y), deg=1)
print("slope      :", round(slope, 5), "  <- negative because the line goes down")
print("intercept  :", round(intercept, 5))
print("lambda_z   :", round(-slope, 5), "   <- the positive version of the slope")
print("half life  :", round(np.log(2) / -slope, 3), "h")
''')

code('''
# Random numbers you can reproduce. Every problem notebook does this.
rng = np.random.default_rng(42)       # the number is the seed
print("first draw :", np.round(rng.normal(0, 1, 4), 4))

rng = np.random.default_rng(42)       # same seed, so the same numbers come back
print("again      :", np.round(rng.normal(0, 1, 4), 4))

rng = np.random.default_rng(7)        # different seed, different numbers
print("other seed :", np.round(rng.normal(0, 1, 4), 4))
''')

# ---------------------------------------------------------------- pandas
md('''
## 3. pandas, the table library

A DataFrame is a table with named columns. Think of a spreadsheet you can
program. By convention it is imported as `pd`.
''')

code('''
import pandas as pd

df = pd.DataFrame({
    "ANIMAL": ["R1", "R1", "R1", "R2", "R2", "R2"],
    "GROUP":  ["Low", "Low", "Low", "High", "High", "High"],
    "TIME_H": [0.0, 1.0, 4.0, 0.0, 1.0, 4.0],
    "CONC":   [0.0, 120.0, 45.0, 0.0, 900.0, 390.0],
})

print(df)
print("")
print("shape  :", df.shape, "  <- (rows, columns)")
print("columns:", list(df.columns))
print("")
print("df.head(2):")
print(df.head(2))
''')

code('''
# One column out of the table is a Series, which behaves a lot like an array.
print("df['CONC']:")
print(df["CONC"])
print("")
print("as an array    :", df["CONC"].to_numpy())
print("mean           :", df["CONC"].mean())
print("max            :", df["CONC"].max())
print("distinct animals:", df["ANIMAL"].nunique())
print("which animals   :", list(df["ANIMAL"].unique()))

# Several columns at once needs a list inside the brackets, so double brackets.
print("")
print(df[["ANIMAL", "CONC"]].head(3))
''')

code('''
# Keeping some rows. The thing inside the brackets is a True/False Series.
print("df['TIME_H'] > 0:")
print(df["TIME_H"] > 0)

post = df[df["TIME_H"] > 0]
print("")
print("rows before:", len(df), "  rows after:", len(post))
print(post)

# Two conditions at once need & for and, | for or, and brackets around each.
both = df[(df["TIME_H"] > 0) & (df["GROUP"] == "High")]
print("")
print("high group, post dose:")
print(both)
''')

code('''
# Adding a column. Work on a copy if the original still matters.
df2 = df.copy()
df2["CONC_UG_ML"] = df2["CONC"] / 1000.0
print(df2)
''')

code('''
# groupby: do the same summary inside every group.
summary = (df[df["TIME_H"] > 0]
           .groupby(["GROUP", "TIME_H"])["CONC"]
           .agg(["mean", "std"])
           .reset_index()
           .rename(columns={"mean": "CONC_MEAN", "std": "CONC_SD"}))
print(summary)
print("")
print("Without reset_index(), GROUP and TIME_H sit in the index instead of being")
print("columns, and anything downstream that expects columns will not find them:")
print(df[df["TIME_H"] > 0].groupby(["GROUP", "TIME_H"])["CONC"].agg(["mean", "std"]))
''')

code('''
# You can also name the output columns as you build them, which skips the rename.
named = (df[df["TIME_H"] > 0]
         .groupby("GROUP")
         .agg(CONC_MEAN=("CONC", "mean"),
              CONC_SD=("CONC", "std"),
              N_ROWS=("CONC", "size"))
         .reset_index())
print(named)
print("")
print("Useful aggregations: 'mean', 'std', 'size', 'count', 'min', 'max', 'median', 'sum'")
print("'size' counts every row. 'count' skips the missing ones.")
print("pandas 'std' already divides by n-1, so it is the sample SD.")
''')

code('''
# merge: glue a second table on by a shared column.
info = pd.DataFrame({"ANIMAL": ["R1"], "WEIGHT_G": [280.0]})
print("info has no row for R2:")
print(info)

left = pd.merge(df, info, on="ANIMAL", how="left")
inner = pd.merge(df, info, on="ANIMAL", how="inner")

print("")
print("rows in df          :", len(df))
print("rows after how=left :", len(left), " <- every row kept, R2 gets NaN")
print("rows after how=inner:", len(inner), " <- R2 silently dropped")
print("")
print(left)
print("")
print("rows missing a weight:", int(left["WEIGHT_G"].isna().sum()))
print("")
print("Check len() before and after EVERY merge. A merge that quietly drops rows")
print("is one of the most common ways an analysis goes wrong without an error.")
''')

code('''
# Reshaping. Wide has one column per timepoint, tidy has one row per observation.
wide = pd.DataFrame({
    "ANIMAL": ["R1", "R2"],
    "T_0": [0.0, 0.0],
    "T_1": [120.0, 900.0],
    "T_4": [45.0, 390.0],
})
print("wide:")
print(wide)

tidy = wide.melt(id_vars=["ANIMAL"], var_name="TIME_LABEL", value_name="CONC")
print("")
print("after melt (tidy):")
print(tidy)
print("")
print("id_vars lists the columns to KEEP AS THEY ARE. Everything else gets folded")
print("into the label column and the value column.")
''')

code('''
# The time label is still text, so turn it into a number before sorting.
tidy["TIME_H"] = tidy["TIME_LABEL"].str.replace("T_", "", regex=False).astype(float)
print(tidy)
print("")
print("sorted as TEXT  :", sorted(["T_0", "T_1", "T_4", "T_24"]), " <- T_24 before T_4")
print("sorted as NUMBER:", sorted([0.0, 1.0, 4.0, 24.0]), "           <- correct")
''')

code('''
# Repeated rows, and going back to wide.
dupey = pd.concat([tidy, tidy.head(1)], ignore_index=True)
print("duplicated() :", list(dupey.duplicated()), " <- True only for repeats after the first")
print("how many     :", int(dupey.duplicated().sum()))
print("keep=False   :", int(dupey.duplicated(keep=False).sum()), " <- flags BOTH copies")

print("")
print("pivot before dropping them:")
try:
    dupey.pivot(index="ANIMAL", columns="TIME_H", values="CONC")
except ValueError as err:
    print("  fails with:", err)

print("")
print("pivot after dropping them:")
print(dupey.drop_duplicates().pivot(index="ANIMAL", columns="TIME_H", values="CONC"))
''')

code('''
# Sorting and renumbering the index.
shuffled = tidy.sample(frac=1.0, random_state=1)
print("shuffled, note the index on the left:")
print(shuffled)

fixed = shuffled.sort_values(["ANIMAL", "TIME_H"]).reset_index(drop=True)
print("")
print("sorted and renumbered:")
print(fixed)
print("")
print("Without reset_index(drop=True) the old numbers come along for the ride.")
''')

# ---------------------------------------------------------------- matplotlib
md('''
## 4. matplotlib, the plotting library

Always build plots the explicit way: ask for a figure and an axes, then tell
the axes what to draw. `fig` is the whole picture, `ax` is one set of axes
inside it.
''')

code('''
import matplotlib.pyplot as plt

t = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0])
c = np.array([0.7, 1.5, 2.1, 1.9, 1.4, 1.0])

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(t, c, marker="o")
ax.set_xlabel("Time (h)")
ax.set_ylabel("Concentration (mg/L)")
ax.set_title("A concentration time profile")
plt.show()

print("Every plot needs all three labels. A reader cannot guess your units.")
''')

code('''
# Error bars. In this repo yerr is ALWAYS an SD.
means = np.array([2.0, 5.0])
sds = np.array([0.4, 1.1])

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.errorbar([0, 1], means, yerr=sds, fmt="o", color="black", capsize=6, linestyle="none")
ax.set_xticks([0, 1])
ax.set_xticklabels(["Vehicle", "Drug"])
ax.set_xlim(-0.5, 1.5)
ax.set_xlabel("Group")
ax.set_ylabel("Response (AU)")
ax.set_title("Mean with SD")
plt.show()
''')

code('''
# Scatter on top of error bars, which is how you show every animal.
animal_x = [0, 0, 0, 1, 1, 1]
animal_y = [1.7, 2.1, 2.2, 4.1, 5.2, 5.7]

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.errorbar([0, 1], means, yerr=sds, fmt="o", color="black", capsize=6,
            linestyle="none", zorder=3, label="group mean, SD")
ax.scatter(animal_x, animal_y, s=45, alpha=0.75, zorder=2, label="animal")
ax.set_xticks([0, 1])
ax.set_xticklabels(["Vehicle", "Drug"])
ax.set_xlim(-0.5, 1.5)
ax.set_xlabel("Group")
ax.set_ylabel("Response (AU)")
ax.set_title("Showing every animal, not just the summary")
ax.legend(frameon=False)
plt.show()

print("zorder decides what is drawn on top. Higher numbers win.")
''')

code('''
# A log y axis, which straightens out the terminal phase of a PK profile.
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(t, c, marker="o")
axes[0].set_title("Linear")
axes[1].plot(t, c, marker="o")
axes[1].set_yscale("log")
axes[1].set_title("Semi log")

for a in axes:
    a.set_xlabel("Time (h)")
    a.set_ylabel("Concentration (mg/L)")

fig.tight_layout()
plt.show()

print("plt.subplots(1, 2) gives you an ARRAY of axes, reached as axes[0], axes[1].")
print("plt.subplots() on its own gives you a single axes, not an array.")
''')

# ---------------------------------------------------------------- rules
md('''
## 5. The two house rules, shown with numbers

These are not style preferences. Both of them change what a reader concludes.
''')

code('''
# Rule 1. Error bars are SD, never SEM.
vals = np.array([12.0, 15.0, 9.0, 18.0, 11.0, 14.0])
n = vals.size
sd = float(np.std(vals, ddof=1))
sem = sd / np.sqrt(n)

print("six animals :", vals)
print("SD          :", round(sd, 3), "  <- how much the animals differ from each other")
print("SEM         :", round(sem, 3), "  <- the SD divided by sqrt(n)")
print("")
print("Now run 4 times as many animals, with exactly the same spread:")
big = np.repeat(vals, 4)
print("SD with n=24 :", round(float(np.std(big, ddof=1)), 3), "  <- barely moves, as it should")
print("SEM with n=24:", round(float(np.std(big, ddof=1)) / np.sqrt(big.size), 3), "  <- halves")
print("")
print("The SEM shrank because n grew, not because the biology got tighter.")
print("Plot SD, and your reader sees the biology.")
''')

code('''
# Rule 2. With cells inside animals, aggregate to the animal first.
nested = pd.DataFrame({
    "ANIMAL": (["a1"] * 50) + (["a2"] * 5) + (["a3"] * 5),
    "GROUP": "Vehicle",
    "VALUE": list(np.full(50, 10.0)) + list(np.full(5, 40.0)) + list(np.full(5, 40.0)),
})

pooled_mean = nested["VALUE"].mean()
animal = nested.groupby("ANIMAL")["VALUE"].mean()
animal_mean = animal.mean()

print("Three animals. One contributed 50 cells, the other two contributed 5 each.")
print("animal means      :", list(animal.round(2)))
print("")
print("mean of all cells :", round(pooled_mean, 2), "  <- the 50 cell animal dominates")
print("mean of the means :", round(animal_mean, 2), "  <- every animal counts once")
print("")
print("n if you count cells  :", len(nested), " <- wrong, and it makes everything look significant")
print("n if you count animals:", nested['ANIMAL'].nunique(), "  <- right")
print("")
print("The cells inside one animal are repeat measures of that animal, not")
print("independent samples of the drug effect. Aggregate first, every time.")
''')

# ---------------------------------------------------------------- errors
md('''
## 6. Error messages, and what they actually mean

Python errors read the right way are the fastest debugging tool you have.
Read the **last line first**: it names the problem.

| What you see | What it usually means |
|---|---|
| `NameError: name 'x' is not defined` | You never made `x`, or you ran the cells out of order. Run from the top. |
| `KeyError: 'CONC'` | That column name is not in the DataFrame. Check `df.columns` for a typo or different capitalisation. |
| `TypeError: 'NoneType' object is not subscriptable` | Something is `None`, usually a `# YOUR CODE` blank you have not filled in. |
| `ValueError: operands could not be broadcast together with shapes (8,9) (8,)` | Two arrays do not line up. Usually wants `keepdims=True` or `.reshape(-1, 1)`. |
| `ValueError: Index contains duplicate entries, cannot reshape` | `pivot` found more than one row for the same index and column. Drop duplicates first. |
| `IndexError: index 9 is out of bounds for axis 0 with size 9` | You asked for position 9 in a 9 long array, which only goes up to 8. Counting starts at 0. |
| `AttributeError: 'DataFrame' object has no attribute 'xyz'` | Misspelled method, or you are calling a Series method on a DataFrame. |
| `SyntaxError: invalid syntax` | Usually a missing bracket or colon on the line **above** the one being pointed at. |
| Everything comes out `nan` | One `nan` got into the maths. Mask it out, or use `np.nanmean` and friends. |

Two habits worth building.

- Print the thing. `print(x)`, `print(type(x))`, `print(np.shape(x))` answers
  most questions faster than staring at the code does.
- Check the shape whenever maths will not line up. Most numpy errors are
  shape errors wearing a disguise.
''')

code('''
# Reading an error on purpose. This one is deliberate.
try:
    broken = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) / np.array([1.0, 2.0])
except ValueError as err:
    print("ValueError:", err)
    print("")
    print("Read it as: I have a (2,3) and a (2,), and numpy lines shapes up from")
    print("the RIGHT, so it tried to match 3 against 2 and gave up.")
    print("Fix: make the second one (2,1) with keepdims=True or .reshape(-1, 1).")
''')

md('''
## That is the whole toolkit

Everything the week 1 problems need is above. When something in a problem does
not make sense, the move is usually to find the matching cell here, run it,
and read what it prints.
''')


def main():
    nb = new_notebook(cells=CELLS, metadata={
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, str(OUT))
    print("wrote %s (%d cells)" % (OUT.relative_to(ROOT), len(CELLS)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
