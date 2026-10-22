# internship-prep

A problem of the day for clinical pharmacology and pharmacometrics internship
prep, built around the packages on my CV. One notebook a weekday, 20 to 30
minutes, no file uploads, nothing to download inside a problem.

Forty problems, eight weeks, all built and all verified.

**Site:** https://dc00086-glitch.github.io/internship-prep/

**Two ways to run a problem.** In Colab, from the site, with nothing to
install. Or on your own computer with `python start.py`, which needs internet
once and then works offline for good. The notebooks are identical either way,
so you can switch whenever you like. See [OFFLINE.md](OFFLINE.md).

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
matches the notebooks. Do not edit it by hand. It now covers 215 functions
across nine sections, including PyTorch and a PK and PD section with the
things that are not functions at all: what lambda_z selection is, why the
baseline pins `kin`, what BLQ handling does to an analysis.

[**The browse page**](browse.html) is every problem on one page: scenario,
explanation, task, setup, toolkit and parts, with a sidebar to flip between
days and a toolkit drawer that stays open while you read the questions. It is
one file with everything embedded, so it works by double clicking it, offline,
with nothing running. Use it to read ahead or to look something up. The blanks
are not fillable there, which is the point of it.

## Working offline

```
python start.py
```

That installs what it needs the first time, which needs internet once, and
then opens today's problem in JupyterLab. Every run after that is offline.
`python start.py --list` shows the schedule, `python start.py 12` opens a
particular problem, `python start.py --foundations` opens the foundations
notebook, and `python start.py --browse` opens the browse page.

On Windows you can double click `start_windows.bat` instead, and on a Mac
`start_mac_linux.command`.

[**OFFLINE.md**](OFFLINE.md) has the detail: what to install, what the
differences are between JupyterLab and Colab, the Anaconda route, and what to
do when something goes wrong.

## How a day works

1. Open the site and press **Open in Colab** on today's problem, or run
   `python start.py` on your own computer, which opens the same notebook in
   JupyterLab.
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

In Colab there is nothing to install: everything used is already there.
On your own computer, `start.py` installs it once. Either way, no problem
ever asks you to upload or download a data file.

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
tools/build_all.py          runs every builder in order
tools/build_tutor_pack.py   builds tutor/, the files to upload to a Claude project
tutor/                      generated tutor pack: instructions, index, reference,
                            one file per week, solutions kept separate
index.html            the site, picks today's problem by date
browse.html           every problem on one page, works offline from the file
start.py              sets up and opens today's problem on your own computer
requirements.txt      the packages start.py installs
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
| 2 | pandas in depth: transform vs agg, merge validation, tidy vs wide, crosstabs, panels | **built** |
| 3 | scipy: group comparisons, geometric statistics, `curve_fit`, `solve_ivp`, Emax | **built** |
| 4 | scikit-image and OpenCV: thresholding, morphology, regionprops, contours, a nested pipeline | **built** |
| 5 | scikit-learn: train/test split, leakage in cross-validation, random forest, elastic net, PCA | **built** |
| 6 | PyTorch: tensors, autograd, gradient descent by hand, `nn.Module` and optimisers, a population fit | **built** |
| 7 | PK: NCA by trapezoid, lambda_z and clearance, one and two compartment fits, multiple dosing | **built** |
| 8 | PK and PD: hysteresis and the effect compartment, indirect response, variability simulation, a NONMEM dataset, dose selection | **built** |

### Phase 2, R

Colab R runtime: tidyverse with explicit `dplyr::select()`, ggplot2, lme4,
PKNCA, mrgsolve, and a small Shiny app.

## Week 1, numpy, pandas and matplotlib

| ID | Problem | Topic |
|---|---|---|
| W1D1 | Clean one plasma profile and get the terminal half-life | numpy: 1D arrays, nan masking, log-linear terminal slope |
| W1D2 | A subject by time matrix: axis reductions, SD, and broadcasting | numpy: 2D arrays, axis reductions, sample SD, broadcasting |
| W1D3 | Rat PK study: filter, summarise by group, and join the dosing record | pandas: filtering, derived columns, groupby.agg, left merge |
| W1D4 | A collaborator's spreadsheet: reshape wide to tidy and back again | pandas: melt, string cleaning, duplicates, pivot |
| W1D5 | Cells inside animals: aggregate first, then plot mean with SD | matplotlib and pandas: nested data, animal level aggregation, SD error bars |

## Week 2, pandas in depth

| ID | Problem | Topic |
|---|---|---|
| W2D1 | Baseline correction: transform, and why it is not agg | pandas: groupby.transform, baseline correction, alignment |
| W2D2 | Joins that lie: validate the merge before you trust it | pandas: join types, validate, indicator, duplicate keys |
| W2D3 | Nominal versus actual sampling times, and the samples that never arrived | pandas: time deviations, building a full grid, finding missing records |
| W2D4 | Adverse events: counting events is not counting patients | pandas: crosstab, pivot_table, normalize, subject level incidence |
| W2D5 | Two panels and a spaghetti plot: the figure every PK report opens with | matplotlib: subplots, semilog axes, per subject overlays |

## Week 3, scipy

| ID | Problem | Topic |
|---|---|---|
| W3D1 | Comparing two groups when the data is skewed | scipy.stats: t-test, Welch, Mann-Whitney, and choosing between them |
| W3D2 | AUC is lognormal, so report the geometric mean | scipy.stats: lognormal data, geometric mean and CV, normality testing |
| W3D3 | Fitting a decay curve, and how sure you are of the parameters | scipy.optimize: curve_fit, starting values, parameter standard errors |
| W3D4 | Solving the compartment model as an ODE, and surviving a discontinuity | scipy.integrate: solve_ivp, t_eval, checking against an analytic solution |
| W3D5 | Emax and sigmoid Emax: fitting a concentration response curve | scipy.optimize: Emax models, bounds, and comparing nested fits |

## Week 4, scikit-image and OpenCV

| ID | Problem | Topic |
|---|---|---|
| W4D1 | From a microscope image to a binary mask | scikit-image: images as arrays, Otsu thresholding, smoothing before you threshold |
| W4D2 | Cleaning up a mask: opening, filling holes, and dropping edge cells | scikit-image: morphological opening, filling holes, clearing the border |
| W4D3 | Labelling objects and measuring them one by one | scikit-image: connected components, regionprops_table, filtering by measurement |
| W4D4 | The same segmentation in OpenCV, and where the two libraries disagree | OpenCV: blurring, Otsu thresholding, findContours and contour areas |
| W4D5 | A whole imaging study: fields inside animals inside groups | scikit-image pipeline: per field, per animal, per group, with the right n |

## Week 5, scikit-learn

| ID | Problem | Topic |
|---|---|---|
| W5D1 | Predicting clearance: the train/test split and why it exists | scikit-learn: train_test_split, fit and predict, R2 and RMSE, optimism |
| W5D2 | Cross-validation, and the leak that invents a result from noise | scikit-learn: KFold, cross_val_score, Pipeline, and data leakage |
| W5D3 | Random forests, and two different answers to which feature matters | scikit-learn: RandomForestRegressor, impurity importance, permutation importance |
| W5D4 | Elastic net: why you must standardise, and what the penalty buys | scikit-learn: ElasticNet, standardisation, regularisation paths, ElasticNetCV |
| W5D5 | PCA on a correlated covariate panel | scikit-learn: PCA, explained variance, loadings, projecting onto components |

## Week 6, PyTorch

| ID | Problem | Topic |
|---|---|---|
| W6D1 | Tensors: numpy with two things added | PyTorch: tensors, dtypes, shapes, numpy interop, the autograd flag |
| W6D2 | Autograd: derivatives you did not have to derive | PyTorch: backward, .grad, gradient accumulation, no_grad, autograd vs finite differences |
| W6D3 | Gradient descent, written out by hand | PyTorch: the fitting loop, learning rate, log parameters, divergence |
| W6D4 | nn.Module and an optimiser: the loop everyone actually writes | PyTorch: nn.Module, nn.Parameter, torch.optim, Adam against plain SGD |
| W6D5 | Twelve subjects, twenty four parameters, one backward pass | PyTorch: vectorised parameters, pooled against individual fitting, recovering between subject variability |

## Week 7, PK and PD applications, part 1

| ID | Problem | Topic |
|---|---|---|
| W7D1 | Non-compartmental analysis: Cmax, Tmax and AUC by trapezoid | PK: linear and linear-up log-down trapezoidal AUC, partial AUC |
| W7D2 | Finishing the NCA: lambda_z, AUC to infinity, clearance and MRT | PK: terminal slope selection, extrapolated AUC, CL/F, Vz/F, mean residence time |
| W7D3 | Fitting a one compartment oral model, and checking it against NCA | PK modelling: depot and central compartments, fitting an ODE model, model versus NCA |
| W7D4 | Two compartments, and what a one compartment model gets wrong | PK modelling: distribution phase, alpha and beta, choosing model structure |
| W7D5 | Multiple dosing: accumulation, steady state, and how long it takes | PK: repeated dosing by simulation, accumulation ratio, time to steady state |

## Week 8, PK and PD applications, part 2

| ID | Problem | Topic |
|---|---|---|
| W8D1 | PK/PD link: why the effect peaks hours after the concentration | PK/PD: hysteresis, direct Emax fitting and its residual structure, the effect compartment |
| W8D2 | Indirect response: when the biomarker, not the drug, sets the clock | PK/PD: turnover models, inhibition of production, dose dependent time to nadir |
| W8D3 | A virtual population: simulating variability and reading it back | Population PK: lognormal parameters, exposure percentiles, how precisely a CV can be estimated |
| W8D4 | Building a NONMEM ready dataset, and auditing it | Population PK: event records, EVID and MDV, BLQ handling, per subject NCA with groupby |
| W8D5 | Capstone: choosing the phase 2 dose | PK/PD: steady state exposure, probability of target attainment, joint efficacy and safety |

## Rebuilding (only needed when problems change)

```bash
python3 tools/build_all.py       # notebooks, foundations, and the Word reference
python3 tools/verify.py          # run all notebooks, solutions must pass
python3 tools/negative_test.py   # wrong answers must be caught
python3 tools/check_dashes.py    # hyphens only, including inside the Word file
```

Needs everything in `requirements.txt`, plus `nbformat nbclient ipykernel`
from `tools/offline_tools.txt`, plus node and the `docx` npm package for the
Word reference. `tools/verify.py week06` and
`tools/negative_test.py W6D1 W6D2` take a filter, which is much faster than
running the lot while you are working on one week.

## The tutor project

`tutor/` holds everything a Claude project needs to tutor these problems:
`TUTOR_INSTRUCTIONS.md` to paste into the project's custom instructions, then
`tutor_index.md`, `tutor_reference.md`, `tutor_environment.md` and one file per
week as project knowledge. The week files carry the full problem text, the
setup code, the toolkit, and every failure message each check can produce, so a
quoted `FAIL` can be looked up rather than guessed at. Solutions are written to
separate files, so uploading them is a decision rather than an accident.

It is generated from `tools/specs/`, the same source as the notebooks:

```bash
python3 tools/build_tutor_pack.py
```

The whole pack is about 830 KB, which is more than one project should carry, so
upload the four core files and swap the week file as you move through the
curriculum.

### Adding a function to the reference

1. Add an entry to `tools/function_catalog.py`.
2. List its key in the `FUNCTIONS` list of whichever spec introduces it.
3. Run `python3 tools/build_all.py`.

The build refuses to run if a spec lists a function the catalog does not
know, so the two cannot drift apart.
