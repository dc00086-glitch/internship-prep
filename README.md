# internship-prep

A problem of the day for clinical pharmacology and pharmacometrics internship
prep, built around the packages on my CV. One notebook a weekday, 20 to 30
minutes, no installs, no file uploads.

**Site:** https://dc00086-glitch.github.io/internship-prep/

## Start here

[**The foundations notebook**](https://colab.research.google.com/github/dc00086-glitch/internship-prep/blob/main/reference/foundations.ipynb)
covers Python, numpy, pandas and matplotlib from the ground up: making an
array, what `nan` is for, the `axis` argument, what a DataFrame actually is,
`groupby`, merges, reshaping, the figure and axes model, and a table of common
error messages with what each one really means. Every cell runs and prints
what it did.

Run it once before week 1, then keep it open in a second tab.

[**The function reference (Word)**](https://github.com/dc00086-glitch/internship-prep/raw/main/reference/function_reference.docx)
is a lookup for every function the problems use: what it does, one line of
real use, and the mistake it tends to invite. It also carries the house rules
and a table of common error messages. Each entry is tagged with the problem
that introduces it, so you can see what you have already met.

It is generated, and it is rebuilt whenever a week is added, so it always
matches the notebooks. Do not edit it by hand.

## How a day works

1. Open the site and press **Open in Colab** on today's problem.
2. Read the scenario and the **How it works** section, which explains the
   package idea the problem is built on, not just the syntax.
3. Run the setup cell. The data is synthetic and generated in the notebook
   from a fixed seed, so your numbers always match mine.
4. Run the **Toolkit** cell. It demonstrates every move the problem needs on a
   tiny made up example, so none of it is the answer. Read its output before
   you start typing.
5. Fill in the `# YOUR CODE` blanks, one part at a time. Each part restates
   what it wants and names the moves you need.
6. Run the check cell as often as you like. Each part comes back `PASS`,
   `FAIL` with a reason, or `TODO` if you have not started it. A `FAIL` tells
   you what is wrong and nudges you, but never hands you the answer.
7. Run the last cell and copy the **Send to tutor** block, along with your
   code, into chat.
8. Only after that, look in `solutions/` if you want to compare.

Nothing to install. Everything used is already in Colab.

### What is in each problem notebook

| Section | What it is for |
|---|---|
| Scenario | the pharmacology framing, so the task has a reason to exist |
| How it works | how the package itself works, written out properly |
| Your task | parts a, b, c, with the exact variable names wanted |
| Setup | synthetic data, fixed seed, no uploads |
| Toolkit | every move demonstrated on a tiny made up example |
| Parts a, b, c | `# YOUR CODE` blanks, each with its own explanation |
| Check | PASS, FAIL with a reason, or TODO, per part |
| Send to tutor | a block to copy into chat along with your code |

The teaching sections are heaviest in the early weeks, where the point is
learning the packages, and get lighter from about week 5 on, where the point
is applying them.

## House rules baked into the checks

These are enforced by the checker, not just mentioned in the text, because
they are the two mistakes that cost people credibility fastest.

- **Error bars are SD, never SEM.** If you plot SEM, the check says so by
  name. SEM mostly reports how many subjects you ran; SD reports how much they
  actually differ, which is the question.
- **SD means the sample SD, with `n - 1`.** Note that `np.std` defaults to
  `n`, so you have to pass `ddof=1`. Pandas `.std()` already uses `n - 1`.
- **Nested data gets aggregated to the animal first.** When there are many
  cells per animal, the unit of replication is the animal. Pool the cells and
  you inflate n, you weight each animal by however many cells you happened to
  image, and every statistic afterwards is wrong. The check catches both the
  inflated n and the cell level SD.
- **No em dashes or en dashes anywhere.** Hyphens only.
  `python3 tools/check_dashes.py` enforces it.

## Repository layout

```
problems/week01/      the notebooks you work in
solutions/week01/     reference solutions, kept out of the problem notebooks
reference/            the foundations notebook and the function reference docx
tools/function_catalog.py   every function, with what it does and what to watch for
tools/build_glossary.py     builds the function reference docx
tools/make_docx.js          turns the glossary data into a Word file
tools/build_all.py          runs all three builders in order
index.html            the site, picks today's problem by date
problems.json         the problem list, generated
tools/build.py        builds every problem notebook from tools/specs/
tools/build_foundations.py  builds reference/foundations.ipynb
tools/specs/          one small python file per problem, the actual source
tools/verify.py       runs every notebook top to bottom in a clean kernel
tools/negative_test.py feeds wrong answers to every check and insists they are caught
tools/check_dashes.py  fails if a banned dash character appears anywhere
```

Problem notebooks and solution notebooks are generated from the same spec, so
a solution can never drift away from the problem it solves.

## Moving the start date

The schedule runs one problem per weekday. Open `index.html`, find this near
the top of the script, and change the date:

```js
const START_DATE = "2026-10-19";   // first problem lands on this date
```

Weekends are rest days: the site keeps showing Friday's problem until Monday.
To look at any problem out of order, add `?n=3` to the address.

## Curriculum

Five problems a week, easy to hard within each week.

### Phase 1, Python

| Week | Topic | Status |
|---|---|---|
| 1 | numpy, pandas, matplotlib | **built** |
| 2 | pandas in depth: groupby, merges, tidy vs wide, more matplotlib | planned |
| 3 | scipy: stats tests, `optimize.curve_fit`, `integrate.solve_ivp` | planned |
| 4 | scikit-image and OpenCV: thresholding, segmentation, morphology | planned |
| 5 | scikit-learn: train/test split, cross-validation, random forest, elastic net, PCA | planned |
| 6 | PyTorch basics: tensors, autograd, a small regression model | planned |
| 7 | PK and PD: NCA (AUC trapezoid, Cmax, half-life, CL, Vd), one and two compartment ODE models | planned |
| 8 | PK and PD: multiple dosing, Emax and sigmoid Emax fitting, between-subject variability, reading a NONMEM-style dataset | planned |

### Phase 2, R

Colab R runtime: tidyverse with explicit `dplyr::select()`, ggplot2, lme4,
PKNCA, mrgsolve, and a small Shiny app.

## Week 1

| ID | Problem | Topic |
|---|---|---|
| W1D1 | Clean one plasma profile and get the terminal half-life | numpy: 1D arrays, nan masking, log-linear terminal slope |
| W1D2 | A subject by time matrix: axis reductions, SD, and broadcasting | numpy: 2D arrays, axis reductions, sample SD, broadcasting |
| W1D3 | Rat PK study: filter, summarise by group, and join the dosing record | pandas: filtering, derived columns, groupby.agg, left merge |
| W1D4 | A collaborator's spreadsheet: reshape wide to tidy and back again | pandas: melt, string cleaning, duplicates, pivot |
| W1D5 | Cells inside animals: aggregate first, then plot mean with SD | matplotlib and pandas: nested data, animal level aggregation, SD error bars |

## Rebuilding (only needed when problems change)

```bash
python3 tools/build_all.py       # notebooks, foundations, and the Word reference
python3 tools/verify.py          # run all notebooks, solutions must pass
python3 tools/negative_test.py   # wrong answers must be caught
python3 tools/check_dashes.py    # hyphens only, including inside the Word file
```

Needs `numpy pandas matplotlib scipy nbformat nbclient ipykernel`, plus node
and the `docx` npm package for the Word reference.

### Adding a function to the reference

1. Add an entry to `tools/function_catalog.py`.
2. List its key in the `FUNCTIONS` list of whichever spec introduces it.
3. Run `python3 tools/build_all.py`.

The build refuses to run if a spec lists a function the catalog does not
know, so the two cannot drift apart.
