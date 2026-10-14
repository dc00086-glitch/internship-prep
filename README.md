# internship-prep

A problem of the day for clinical pharmacology and pharmacometrics internship
prep, built around the packages on my CV. One notebook a weekday, 20 to 30
minutes, no installs, no file uploads.

**Site:** https://dc00086-glitch.github.io/internship-prep/

## How a day works

1. Open the site and press **Open in Colab** on today's problem.
2. Read the scenario, run the setup cell. The data is synthetic and generated
   in the notebook from a fixed seed, so your numbers always match mine.
3. Fill in the `# YOUR CODE` blanks, one part at a time.
4. Run the check cell as often as you like. Each part comes back `PASS`,
   `FAIL` with a reason, or `TODO` if you have not started it. A `FAIL` tells
   you what is wrong and nudges you, but never hands you the answer.
5. Run the last cell and copy the **Send to tutor** block, along with your
   code, into chat.
6. Only after that, look in `solutions/` if you want to compare.

Nothing to install. Everything used is already in Colab.

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
index.html            the site, picks today's problem by date
problems.json         the problem list, generated
tools/build.py        builds every notebook from tools/specs/
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
python3 tools/build.py           # regenerate notebooks and the site list
python3 tools/verify.py          # run all notebooks, solutions must pass
python3 tools/negative_test.py   # wrong answers must be caught
python3 tools/check_dashes.py    # hyphens only
```

Needs `numpy pandas matplotlib scipy nbformat nbclient ipykernel`.
