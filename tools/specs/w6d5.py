PID = "W6D5"
WEEK = 6
DAY = 5
SLUG = "w6d5_torch_population_fit"
TITLE = "Twelve subjects, twenty four parameters, one backward pass"
TOPIC = "PyTorch: vectorised parameters, pooled against individual fitting, recovering between subject variability"
EST = "25-35 min"

SCENARIO_MD = '''
## Scenario

A first in human single ascending dose study has produced twelve subjects at
one dose level, each with a full profile. The question on the table is the one
that is always on the table: what are the typical PK parameters, and how much do
they vary between people.

You have two ways to answer it with what you now know. You can throw all the
data into one fit and get one volume and one rate constant for everybody, which
is called naive pooling and is what a spreadsheet would do. Or you can give
every subject their own parameters and fit all twenty four at once, which is
the two stage approach and the direct ancestor of the mixed effects models in
week 8.

Both are three lines of difference in the same torch code, because a parameter
and a vector of parameters are the same thing to an optimiser. That is what
today demonstrates, and it is the reason this week exists: `curve_fit` would
need a loop and a reshape to do the second fit, and in week 8 when the
parameters become a hierarchy, it could not do it at all.
'''

CONCEPT_MD = '''
## One model, two shapes

**The parameters become vectors.** Instead of a scalar `log_v`, carry a tensor
of shape `(n_subjects,)`. Everything else about the machinery is unchanged:
`nn.Parameter` registers it, `backward()` fills in a gradient for every
element, `Adam` steps every element with its own adapted step size. There is no
loop over subjects anywhere.

**The predictions become a matrix, by broadcasting.** With `n` subjects and `m`
sampling times, you want an `(n, m)` matrix of predictions. The parameters are
`(n,)` and the times are `(m,)`, so insert axes to line them up:

```
v = torch.exp(self.log_v)[:, None]     # (n, 1)
k = torch.exp(self.log_k)[:, None]     # (n, 1)
return (DOSE / v) * torch.exp(-k * t[None, :])   # (n, 1) with (1, m) -> (n, m)
```

This is the numpy broadcasting from week 1, in torch, with `None` doing the
work that `np.newaxis` did. Get an axis wrong and you will not get an error,
you will get a `(12, 12)` or a `(9, 9)` of nonsense, so check `.shape` before
you check anything else.

**And the pooled fit is the same class with `n = 1`.** A `(1, m)` prediction
matrix broadcasts against a `(12, m)` observation matrix, giving every subject
the same predicted curve and summing the residuals over all of them. One class,
one loop, two models.

**What the comparison shows.** The pooled fit gives you a single typical curve.
Its residuals will not be scattered around zero at random: each subject will sit
mostly above or mostly below the common curve, because real people differ. That
pattern in the residuals is between subject variability, and the pooled fit
cannot represent it, so it hides it in the error term.

The individual fits represent it exactly, and now you can measure it. Take the
twelve fitted volumes, work on the log scale, and the standard deviation of
those logs is the between subject variability. The usual way to report it is as
a coefficient of variation:

```
CV = sqrt(exp(var(log V)) - 1)
```

which for small variability is close to the standard deviation of the logs
itself, and is the number you see in every population PK report.

**What it does not show.** Each of these twelve estimates carries its own
uncertainty, and the spread of noisy estimates is wider than the spread of the
truth. Two stage estimates of variability are therefore biased upward, which is
exactly the problem mixed effects models solve, and that is week 8.
'''

TASK_MD = '''
## Your task

Three parts, worked one at a time.

| Part | What you work out | Variables to fill in |
|---|---|---|
| a | the pooled fit, and the structure it leaves in the residuals | `v_pool`, `k_pool`, `sse_pool`, `subject_bias` |
| b | twenty four parameters at once, checked against a curve_fit loop | `v_ind`, `k_ind`, `sse_ind`, `v_scipy` |
| c | the population summary the study was run to produce | `v_gm`, `cv_v_pct`, `cv_k_pct`, `sse_ratio` |

Below this you will find the **Setup** cell, then a **Toolkit** cell that
demonstrates every move on a tiny made up example, then **Part a**, **Part b**
and **Part c**, each with its own explanation and a cell to write in.
'''

DATA_CODE = '''
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

PROBLEM_ID = "W6D5"
TOPIC = "PyTorch: vectorised parameters, pooled against individual fitting, recovering between subject variability"

torch.manual_seed(0)
rng = np.random.default_rng(6505)

DOSE = 100.0                 # mg, IV bolus, the same dose for everyone
N_SUBJ = 12

# The truth, which the fits do not get to see.
V_POP_TRUE = 35.0            # L, population typical value
K_POP_TRUE = 0.12            # per h, population typical value
CV_V_TRUE = 0.25             # between subject variability in V, 25 pct
CV_K_TRUE = 0.20             # between subject variability in k, 20 pct

_times = np.array([0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 18.0, 24.0])
_v_i = V_POP_TRUE * rng.lognormal(0.0, CV_V_TRUE, size=N_SUBJ)
_k_i = K_POP_TRUE * rng.lognormal(0.0, CV_K_TRUE, size=N_SUBJ)
_c = (DOSE / _v_i)[:, None] * np.exp(-_k_i[:, None] * _times[None, :])
_c = np.round(_c * rng.lognormal(0.0, 0.07, size=_c.shape), 4)

t_obs = torch.tensor(_times, dtype=torch.float64)          # shape (9,)
c_obs = torch.tensor(_c, dtype=torch.float64)              # shape (12, 9)

V_START = 30.0               # L, starting guess for every subject
K_START = 0.15               # per h, starting guess for every subject

LR = 0.05
N_ITER = 2000

print("t_obs shape:", tuple(t_obs.shape), " c_obs shape:", tuple(c_obs.shape))
print("")
print("subject 1 profile:", c_obs[0].numpy())
print("subject 2 profile:", c_obs[1].numpy())
print("concentrations run from %.4f to %.4f mg/L" % (c_obs.min().item(), c_obs.max().item()))
print("")
print("every subject starts the fit at V = %.1f L, k = %.2f per h" % (V_START, K_START))
print("Adam, lr = %g, %d iterations" % (LR, N_ITER))
'''

TOOLKIT_MD = '''
## Toolkit for this problem

| What you want to do | How |
|---|---|
| a vector of parameters | `torch.nn.Parameter(torch.full((n,), value, dtype=torch.float64))` |
| add a trailing axis | `x[:, None]`, giving shape `(n, 1)` |
| add a leading axis | `t[None, :]`, giving shape `(1, m)` |
| check what you built | `tuple(x.shape)` |
| sum over times, keep subjects | `.sum(dim=1)`, or `.mean(dim=1)` |
| sum everything | `.sum()` |
| tensor to numpy, after fitting | `x.detach().numpy()` |
| geometric mean | `np.exp(np.mean(np.log(x)))` |
| variance of logs, sample version | `np.var(np.log(x), ddof=1)` |

`dim=1` in torch is `axis=1` in numpy. With a `(12, 9)` matrix of subjects by
times, `dim=1` collapses the times and leaves you one number per subject, and
`dim=0` collapses the subjects and leaves you one number per time. Which one you
want is the difference between a per subject summary and a per timepoint one,
and the shapes will not tell you which you asked for, since 12 and 9 are both
plausible answers.
'''

TOOLKIT_CODE = '''
import numpy as np
import torch

print("--- 1. broadcasting a parameter vector against a time vector ---")
demo_p = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)      # 3 subjects
demo_t = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)  # 4 times
print("p shape", tuple(demo_p.shape), " t shape", tuple(demo_t.shape))
print("p[:, None] shape:", tuple(demo_p[:, None].shape))
print("t[None, :] shape:", tuple(demo_t[None, :].shape))
demo_grid = demo_p[:, None] * demo_t[None, :]
print("product shape   :", tuple(demo_grid.shape), "  one row per subject")
print(demo_grid)

print("")
print("--- 2. the axis mistake does not raise, it just gives nonsense ---")
try:
    demo_bad = demo_p[:, None] * demo_t[:, None]
    print("that worked and gave shape", tuple(demo_bad.shape))
except RuntimeError as err:
    print("shapes (3,1) and (4,1) do not broadcast:", str(err)[:70])
demo_same = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
print("but (3,1) times (3,) gives shape", tuple((demo_p[:, None] * demo_same).shape),
      "which is a 3 by 3 of something you did not ask for")

print("")
print("--- 3. a vector of parameters fits exactly like a scalar one ---")
# Three independent straight lines through three different slopes.
demo_x = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
demo_y = torch.tensor([[0.0, 1.0, 2.0, 3.0],
                       [0.0, 2.0, 4.0, 6.0],
                       [0.0, 5.0, 10.0, 15.0]], dtype=torch.float64)
demo_slope = torch.nn.Parameter(torch.full((3,), 0.0, dtype=torch.float64))
demo_opt = torch.optim.Adam([demo_slope], lr=0.1)
for _ in range(800):
    demo_opt.zero_grad()
    demo_loss = ((demo_slope[:, None] * demo_x[None, :] - demo_y) ** 2).sum()
    demo_loss.backward()
    demo_opt.step()
print("fitted slopes:", np.round(demo_slope.detach().numpy(), 6), " true 1, 2, 5")
print("three parameters, one loss, one backward pass, no loop over the three")

print("")
print("--- 4. a (1, m) prediction broadcasts against (n, m) observations ---")
demo_one = torch.nn.Parameter(torch.full((1,), 0.0, dtype=torch.float64))
demo_opt2 = torch.optim.Adam([demo_one], lr=0.1)
for _ in range(800):
    demo_opt2.zero_grad()
    demo_loss2 = ((demo_one[:, None] * demo_x[None, :] - demo_y) ** 2).sum()
    demo_loss2.backward()
    demo_opt2.step()
print("one slope for all three series:", np.round(demo_one.detach().numpy(), 6))
print("that is the pooled fit, from the same code with n = 1")

print("")
print("--- 5. dim=1 against dim=0 ---")
demo_m = torch.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], dtype=torch.float64)
print(demo_m)
print("mean(dim=1), one per row   :", demo_m.mean(dim=1))
print("mean(dim=0), one per column:", demo_m.mean(dim=0))

print("")
print("--- 6. variability on the log scale ---")
demo_vals = np.array([20.0, 30.0, 45.0, 28.0, 36.0])
print("values         :", demo_vals)
print("geometric mean : %.4f" % np.exp(np.mean(np.log(demo_vals))))
print("sd of the logs : %.4f" % np.std(np.log(demo_vals), ddof=1))
print("CV from logs   : %.2f pct" % (100 * np.sqrt(np.exp(np.var(np.log(demo_vals), ddof=1)) - 1)))
'''

PARTS = [
    dict(
        label="a",
        prompt_md='''
## Part a. The pooled fit, and what it leaves behind

Fill in the class below so that it carries `n_subjects` volumes and
`n_subjects` rate constants, and returns an `(n_subjects, n_times)` matrix of
predictions. The parameters must be named `log_v` and `log_k` so the check can
find them.

Then make one with `n_subjects=1` and fit it with Adam at `LR` for `N_ITER`
iterations, against all twelve subjects at once. The `(1, 9)` predictions
broadcast against the `(12, 9)` observations, so the loss is the total squared
residual of one common curve across the whole study. Record:

- `v_pool` and `k_pool`, the fitted values as plain floats,
- `sse_pool`, the final loss as a plain float,
- `subject_bias`, the mean residual for each subject, as a numpy array of
  twelve numbers, with the residual taken as **observed minus predicted**.

That order matters here. For a squared loss it makes no difference, which is why
the earlier days were relaxed about it, but a mean residual has a sign, and
everywhere in pharmacometrics the residual is observed minus predicted, so a
positive value means the model is under predicting that subject.

Look at `subject_bias` when you have it. Eight of the twelve subjects sit
mostly below the common curve and four sit mostly above it, and the largest of
the twelve is 0.84 mg/L against a mean concentration across the whole study of
1.41 mg/L. Those are not random residuals
around a good fit, they are twelve people with twelve different volumes being
described by one number. Nothing is wrong with the optimiser. The model is
wrong.

Fill in the class, then `v_pool`, `k_pool`, `sse_pool`, `subject_bias`.

Moves: `torch.nn.Parameter`, `torch.full`, `[:, None]`, `[None, :]`,
`torch.optim.Adam`, the five line loop, `.mean(dim=1)`, `.detach().numpy()`.
''',
        starter='''
# Part a. The pooled fit: one curve for everybody.

class PopulationPK(torch.nn.Module):
    """One compartment IV bolus for n_subjects subjects, on log parameters."""

    def __init__(self, n_subjects, v_init, k_init):
        super().__init__()
        self.log_v = None   # YOUR CODE: nn.Parameter, shape (n_subjects,), log(v_init)
        self.log_k = None   # YOUR CODE: nn.Parameter, shape (n_subjects,), log(k_init)

    def forward(self, t):
        v = None            # YOUR CODE: exp(log_v) with a trailing axis, shape (n, 1)
        k = None            # YOUR CODE: exp(log_k) with a trailing axis, shape (n, 1)
        return None         # YOUR CODE: (DOSE / v) * exp(-k * t[None, :])


model_pool = PopulationPK(1, V_START, K_START)

# YOUR CODE: an Adam optimiser over model_pool.parameters() at lr = LR, then
# the five line loop for N_ITER iterations against c_obs.

v_pool = None         # YOUR CODE: the fitted volume, as a plain float
k_pool = None         # YOUR CODE: the fitted rate constant, as a plain float
sse_pool = None       # YOUR CODE: the final loss, as a plain float
subject_bias = None   # YOUR CODE: mean residual per subject, observed minus
                      #            predicted, as an array of 12 numbers

print("pooled fit: V = %r L, k = %r per h" % (v_pool, k_pool))
print("total SSE : %r" % (sse_pool,))
print("")
print("mean residual by subject (observed minus predicted):")
print(np.round(subject_bias, 4) if subject_bias is not None else "not set yet")
''',
        solution='''
# Part a. The pooled fit: one curve for everybody.

class PopulationPK(torch.nn.Module):
    """One compartment IV bolus for n_subjects subjects, on log parameters."""

    def __init__(self, n_subjects, v_init, k_init):
        super().__init__()
        self.log_v = torch.nn.Parameter(
            torch.full((n_subjects,), float(np.log(v_init)), dtype=torch.float64))
        self.log_k = torch.nn.Parameter(
            torch.full((n_subjects,), float(np.log(k_init)), dtype=torch.float64))

    def forward(self, t):
        v = torch.exp(self.log_v)[:, None]
        k = torch.exp(self.log_k)[:, None]
        return (DOSE / v) * torch.exp(-k * t[None, :])


def fit_pk(model, lr=None, n_iter=None):
    """Fit a PopulationPK against c_obs. Returns the final loss as a float."""
    lr = LR if lr is None else lr
    n_iter = N_ITER if n_iter is None else n_iter
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss = None
    for _ in range(n_iter):
        opt.zero_grad()
        loss = ((model(t_obs) - c_obs) ** 2).sum()
        loss.backward()
        opt.step()
    return loss.item()


model_pool = PopulationPK(1, V_START, K_START)
sse_pool = fit_pk(model_pool)

v_pool = torch.exp(model_pool.log_v)[0].item()
k_pool = torch.exp(model_pool.log_k)[0].item()

with torch.no_grad():
    subject_bias = (c_obs - model_pool(t_obs)).mean(dim=1).numpy()

print("pooled fit: V = %r L, k = %r per h" % (v_pool, k_pool))
print("total SSE : %r" % (sse_pool,))
print("")
print("mean residual by subject (observed minus predicted):")
print(np.round(subject_bias, 4) if subject_bias is not None else "not set yet")
''',
    ),
    dict(
        label="b",
        prompt_md='''
## Part b. Twenty four parameters, one backward pass

Now make a `PopulationPK` with `n_subjects=N_SUBJ` and fit it exactly the same
way, with the same learning rate and the same number of iterations. The only
change is the shape of the parameters.

Record `v_ind` and `k_ind`, the twelve fitted volumes and rate constants as
numpy arrays, and `sse_ind`, the final total loss.

Then check the whole thing against the tool you already trust, one subject at a
time. Loop over the twelve rows of `c_obs`, call `curve_fit` on each with the
same model and the same starting guesses, collect the twelve volume estimates
into `v_scipy` as a numpy array.

The two should agree to six significant figures, subject by subject. Twelve
separate Levenberg-Marquardt fits with numerical derivatives, and one torch fit
of twenty four parameters, find the same twelve answers.

This is the point of the whole week. There was no loop over subjects in your
torch code, the shape of the parameter did all the work, and the same code would
handle a hundred and twenty subjects without a line changing. The curve_fit
version needs a Python loop, which is fine for twelve independent fits and
impossible once the subjects stop being independent, which is what happens the
moment you put a population distribution over them in week 8.

One thing to notice in the numbers: subject 5's fitted volume is 29.1 L against
a true 27.4, and subject 9's is 37.0 against 34.0. Individual estimates from
nine samples with seven percent noise are good but not exact, and that
estimation error is what inflates the variability estimate in part c.

Fill in `v_ind`, `k_ind`, `sse_ind`, `v_scipy`.

Moves: the same class with a different `n_subjects`, the same loop,
`.detach().numpy()`, a `curve_fit` loop over rows.
''',
        starter='''
# Part b. One set of parameters per subject.

model_ind = PopulationPK(N_SUBJ, V_START, K_START)

# YOUR CODE: fit it the same way you fitted the pooled model. If you wrote the
# loop as a function in part a, this is one line.

v_ind = None     # YOUR CODE: the 12 fitted volumes, as a numpy array
k_ind = None     # YOUR CODE: the 12 fitted rate constants, as a numpy array
sse_ind = None   # YOUR CODE: the final total loss, as a plain float


def one_subject(t, v, k):
    """The same model for a single subject, in numpy, for curve_fit."""
    return (DOSE / v) * np.exp(-k * t)


v_scipy = None   # YOUR CODE: the 12 volume estimates from a curve_fit loop

print("torch     V:", np.round(v_ind, 4) if v_ind is not None else "not set")
print("curve_fit V:", np.round(v_scipy, 4) if v_scipy is not None else "not set")
print("")
print("torch     k:", np.round(k_ind, 5) if k_ind is not None else "not set")
print("")
print("total SSE, individual fits: %r" % (sse_ind,))
print("total SSE, pooled fit     : %r" % (sse_pool,))
''',
        solution='''
# Part b. One set of parameters per subject.

model_ind = PopulationPK(N_SUBJ, V_START, K_START)

sse_ind = fit_pk(model_ind)

v_ind = torch.exp(model_ind.log_v).detach().numpy()
k_ind = torch.exp(model_ind.log_k).detach().numpy()


def one_subject(t, v, k):
    """The same model for a single subject, in numpy, for curve_fit."""
    return (DOSE / v) * np.exp(-k * t)


_v_list = []
for _i in range(N_SUBJ):
    _popt, _ = curve_fit(one_subject, t_obs.numpy(), c_obs[_i].numpy(),
                         p0=[V_START, K_START])
    _v_list.append(_popt[0])
v_scipy = np.array(_v_list)

print("torch     V:", np.round(v_ind, 4) if v_ind is not None else "not set")
print("curve_fit V:", np.round(v_scipy, 4) if v_scipy is not None else "not set")
print("")
print("torch     k:", np.round(k_ind, 5) if k_ind is not None else "not set")
print("")
print("total SSE, individual fits: %r" % (sse_ind,))
print("total SSE, pooled fit     : %r" % (sse_pool,))
''',
    ),
    dict(
        label="c",
        prompt_md='''
## Part c. The population summary

Turn the twelve individual fits into the three numbers a study report would
carry, plus one that tells you whether the extra parameters earned their place.

- `v_gm`, the geometric mean of the twelve fitted volumes. PK parameters are
  reported as geometric means because they are approximately lognormal, and the
  geometric mean of a lognormal is its median,
- `cv_v_pct`, the between subject variability in volume as a percentage, from
  `100 * sqrt(exp(var(log V)) - 1)` with the sample variance, `ddof=1`,
- `cv_k_pct`, the same for the rate constants,
- `sse_ratio`, `sse_pool` divided by `sse_ind`.

Then read what you have produced.

**The ratio is about 23.** Twenty two extra parameters bought a twenty three
fold reduction in squared error, which in a study of twelve people is not a
close call. Between subject variability is the dominant source of structure in
this dataset, and a model without it is not describing the data.

**The pooled volume is biased low.** `v_pool` came out at 35.3 L against a
geometric mean of 36.7 L from the individual fits, 3.6 percent lower.
Pooling on the additive scale weights the subjects with the highest
concentrations most heavily, and the highest concentrations belong to the
subjects with the smallest volumes. The bias is modest here because the design
is balanced, with every subject sampled identically. It is not modest when it
is not.

**The variability estimate needs a caveat.** You will get about 31 percent for
volume, against the 25 percent the data was simulated with. Before concluding
the method is biased, note that these particular twelve subjects happen to have
a spread of 30.4 percent, so your estimate recovered the sample almost exactly.
A sample of twelve simply does not pin down a population standard deviation,
and this one came out high. The remaining gap, 30.6 against 30.4, is the upward
bias of the two stage approach, which is small here because nine samples per
subject with seven percent noise make for precise individual estimates. With
three samples per subject it would not be small.

That caveat is the handover to week 8. Mixed effects models estimate the
typical value and the variability simultaneously instead of in two stages, which
removes this bias and lets you fit subjects who have only two usable samples
each.

Fill in `v_gm`, `cv_v_pct`, `cv_k_pct`, `sse_ratio`.

Moves: `np.log`, `np.exp`, `np.mean`, `np.var(..., ddof=1)`, division.
''',
        starter='''
# Part c. From twelve fits to a population summary.

v_gm = None       # YOUR CODE: geometric mean of the 12 fitted volumes
cv_v_pct = None   # YOUR CODE: between subject CV of V, as a percentage
cv_k_pct = None   # YOUR CODE: between subject CV of k, as a percentage
sse_ratio = None  # YOUR CODE: sse_pool / sse_ind

print("individual fits, n = %d" % N_SUBJ)
print("  V geometric mean : %r L" % (v_gm,))
print("  V between subject CV: %r pct" % (cv_v_pct,))
print("  k between subject CV: %r pct" % (cv_k_pct,))
print("")
print("pooled V for comparison: %r L" % (v_pool,))
print("SSE pooled / SSE individual: %r" % (sse_ratio,))
print("")
print("the truth, which the fits never saw:")
print("  V %.1f L with %.0f pct CV, k %.2f per h with %.0f pct CV"
      % (V_POP_TRUE, 100 * CV_V_TRUE, K_POP_TRUE, 100 * CV_K_TRUE))

if v_ind is not None:
    with torch.no_grad():
        fitted = model_ind(t_obs).numpy()
    fig, ax = plt.subplots(figsize=(7, 5))
    for _i in range(N_SUBJ):
        ax.plot(t_obs.numpy(), c_obs[_i].numpy(), "o", ms=4, alpha=0.6)
        ax.plot(t_obs.numpy(), fitted[_i], "-", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("time (h)")
    ax.set_ylabel("concentration (mg/L)")
    ax.set_title("W6D5 twelve individual fits")
    plt.show()
''',
        solution='''
# Part c. From twelve fits to a population summary.

v_gm = float(np.exp(np.mean(np.log(v_ind))))
cv_v_pct = float(100.0 * np.sqrt(np.exp(np.var(np.log(v_ind), ddof=1)) - 1.0))
cv_k_pct = float(100.0 * np.sqrt(np.exp(np.var(np.log(k_ind), ddof=1)) - 1.0))
sse_ratio = sse_pool / sse_ind

print("individual fits, n = %d" % N_SUBJ)
print("  V geometric mean : %r L" % (v_gm,))
print("  V between subject CV: %r pct" % (cv_v_pct,))
print("  k between subject CV: %r pct" % (cv_k_pct,))
print("")
print("pooled V for comparison: %r L" % (v_pool,))
print("SSE pooled / SSE individual: %r" % (sse_ratio,))
print("")
print("the truth, which the fits never saw:")
print("  V %.1f L with %.0f pct CV, k %.2f per h with %.0f pct CV"
      % (V_POP_TRUE, 100 * CV_V_TRUE, K_POP_TRUE, 100 * CV_K_TRUE))

if v_ind is not None:
    with torch.no_grad():
        fitted = model_ind(t_obs).numpy()
    fig, ax = plt.subplots(figsize=(7, 5))
    for _i in range(N_SUBJ):
        ax.plot(t_obs.numpy(), c_obs[_i].numpy(), "o", ms=4, alpha=0.6)
        ax.plot(t_obs.numpy(), fitted[_i], "-", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("time (h)")
    ax.set_ylabel("concentration (mg/L)")
    ax.set_title("W6D5 twelve individual fits")
    plt.show()
''',
    ),
]


def build_expected(ns):
    import numpy as np
    from scipy.optimize import curve_fit

    t = ns["t_obs"].detach().numpy()
    c = ns["c_obs"].detach().numpy()
    dose = ns["DOSE"]
    v0, k0 = ns["V_START"], ns["K_START"]
    n_subj = int(ns["N_SUBJ"])

    def one(tt, v, k):
        return (dose / v) * np.exp(-k * tt)

    pooled, _ = curve_fit(one, np.tile(t, n_subj), c.ravel(), p0=[v0, k0])
    v_pool, k_pool = float(pooled[0]), float(pooled[1])
    pred_pool = one(t, v_pool, k_pool)
    sse_pool = float(((pred_pool[None, :] - c) ** 2).sum())
    bias = (c - pred_pool[None, :]).mean(axis=1)

    vs, ks, sse_ind = [], [], 0.0
    for i in range(n_subj):
        popt, _ = curve_fit(one, t, c[i], p0=[v0, k0])
        vs.append(float(popt[0]))
        ks.append(float(popt[1]))
        sse_ind += float(((one(t, *popt) - c[i]) ** 2).sum())
    vs = np.array(vs)
    ks = np.array(ks)

    return {
        "n_subj": n_subj,
        "n_times": int(t.size),
        "v_pool": v_pool,
        "k_pool": k_pool,
        "sse_pool": sse_pool,
        "bias": [float(x) for x in bias],
        "bias_by_time": [float(x) for x in (c - pred_pool[None, :]).mean(axis=0)],
        "v_ind": [float(x) for x in vs],
        "k_ind": [float(x) for x in ks],
        "sse_ind": sse_ind,
        "v_gm": float(np.exp(np.mean(np.log(vs)))),
        "v_mean": float(np.mean(vs)),
        "cv_v_pct": float(100.0 * np.sqrt(np.exp(np.var(np.log(vs), ddof=1)) - 1.0)),
        "cv_k_pct": float(100.0 * np.sqrt(np.exp(np.var(np.log(ks), ddof=1)) - 1.0)),
        "sd_log_v_pct": float(100.0 * np.std(np.log(vs), ddof=1)),
        "sse_ratio": sse_pool / sse_ind,
    }


CHECK_CODE = '''
import torch as _torch


def _part_a():
    v_pool = _need("v_pool")
    k_pool = _need("k_pool")
    sse_pool = _need("sse_pool")
    bias = _need("subject_bias")

    cls = globals().get("PopulationPK")
    if cls is None:
        return "`PopulationPK` is not defined. The class skeleton is in the part a cell."
    try:
        probe = cls(3, 30.0, 0.15)
    except Exception as exc:
        return "`PopulationPK(3, 30.0, 0.15)` raised %s: %s" % (type(exc).__name__, exc)
    names = sorted(n for n, _ in probe.named_parameters())
    if names != ["log_k", "log_v"]:
        return ("the model should register exactly two parameters named log_v and log_k, "
                "but it reports %r. Each has to be a torch.nn.Parameter assigned to an "
                "attribute of self." % (names,))
    for nm, p in probe.named_parameters():
        if tuple(p.shape) != (3,):
            return ("`%s` has shape %s in a 3 subject model. It should be a vector with one "
                    "entry per subject, which torch.full((n_subjects,), ...) gives you."
                    % (nm, tuple(p.shape)))
        if p.dtype != _torch.float64:
            return "`%s` should be float64." % nm
    try:
        with _torch.no_grad():
            pred = probe(t_obs)
    except Exception as exc:
        return "calling a 3 subject model raised %s: %s" % (type(exc).__name__, exc)
    if not isinstance(pred, _torch.Tensor):
        return "`forward` should return a tensor of predictions."
    if tuple(pred.shape) != (3, _EXP["n_times"]):
        return ("a 3 subject model predicted shape %s for %d times. It should be "
                "(3, %d): one row per subject, one column per time. Use v[:, None] and "
                "t[None, :]." % (tuple(pred.shape), _EXP["n_times"], _EXP["n_times"]))

    for name, val in (("v_pool", v_pool), ("k_pool", k_pool), ("sse_pool", sse_pool)):
        if isinstance(val, _torch.Tensor):
            return "`%s` is still a tensor. Call .item() on it." % name
        if _as_scalar(val) is None:
            return "`%s` should be a single number." % name

    if not _num_close(_as_scalar(v_pool), _EXP["v_pool"], rtol=1e-3):
        return ("`v_pool` is not the pooled volume. With n_subjects = 1 the (1, 9) "
                "predictions broadcast against the (12, 9) observations, so the loss "
                "covers the whole study.")
    if not _num_close(_as_scalar(k_pool), _EXP["k_pool"], rtol=1e-3):
        return "`k_pool` is not the pooled rate constant."
    if _num_close(_as_scalar(sse_pool), _EXP["sse_ind"], rtol=1e-2):
        return ("`sse_pool` is the individual fit loss. Part a fits one common curve, "
                "which cannot do nearly as well.")
    if not _num_close(_as_scalar(sse_pool), _EXP["sse_pool"], rtol=1e-3):
        return ("`sse_pool` is not the total squared residual of the pooled fit summed "
                "over all 12 subjects and all 9 times.")

    arr = _as_array(bias)
    if arr is None:
        return "`subject_bias` should be an array of numbers."
    if arr.shape == (_EXP["n_times"],):
        return ("`subject_bias` has one entry per TIME, so it was averaged over subjects. "
                "With a (12, 9) residual matrix you want dim=1, which collapses the times "
                "and leaves one number per subject.")
    if arr.shape != (_EXP["n_subj"],):
        return ("`subject_bias` has shape %s. It should hold one mean residual per "
                "subject, so %d numbers." % (arr.shape, _EXP["n_subj"]))
    if _arr_close(arr, [-x for x in _EXP["bias"]], rtol=1e-2, atol=1e-4):
        return ("`subject_bias` has the right magnitudes with every sign flipped. The task "
                "asks for observed minus predicted, so a positive value means the model "
                "under predicts that subject.")
    if not _arr_close(arr, _EXP["bias"], rtol=1e-2, atol=1e-4):
        return ("`subject_bias` is not the mean residual per subject at the pooled fit. "
                "Compute c_obs minus the pooled predictions, then average over times.")
    return None


def _part_b():
    v_ind = _need("v_ind")
    k_ind = _need("k_ind")
    sse_ind = _need("sse_ind")
    v_scipy = _need("v_scipy")

    if isinstance(sse_ind, _torch.Tensor):
        return "`sse_ind` is still a tensor. Call .item() on it."
    if _as_scalar(sse_ind) is None:
        return "`sse_ind` should be a single number."

    for name, val in (("v_ind", v_ind), ("k_ind", k_ind), ("v_scipy", v_scipy)):
        if isinstance(val, _torch.Tensor):
            return ("`%s` is still a tensor. Use .detach().numpy() to get the fitted "
                    "values out." % name)
        arr = _as_array(val)
        if arr is None:
            return "`%s` should be an array of numbers." % name
        if arr.shape != (_EXP["n_subj"],):
            return ("`%s` has shape %s but there are %d subjects."
                    % (name, arr.shape, _EXP["n_subj"]))

    arr = _as_array(v_ind)
    if _arr_close(arr, _np.log(_np.array(_EXP["v_ind"])), rtol=1e-2):
        return "`v_ind` is still on the log scale. Exponentiate before reading it out."
    if _arr_close(arr, _np.full(_EXP["n_subj"], _EXP["v_pool"]), rtol=1e-2):
        return ("every entry of `v_ind` is the pooled volume, so the model still has "
                "n_subjects = 1. Build it with N_SUBJ.")
    if not _arr_close(arr, _EXP["v_ind"], rtol=1e-3):
        return ("`v_ind` does not match the individual volume estimates. At this learning "
                "rate N_ITER iterations is enough to reach them to six decimal places, so "
                "check the broadcasting in forward and that the fit used c_obs.")
    if not _arr_close(_as_array(k_ind), _EXP["k_ind"], rtol=1e-3):
        return "`k_ind` does not match the individual rate constant estimates."

    if _num_close(_as_scalar(sse_ind), _EXP["sse_pool"], rtol=1e-2):
        return ("`sse_ind` is the pooled loss. It should be the final loss of the 24 "
                "parameter fit, which is far smaller.")
    if not _num_close(_as_scalar(sse_ind), _EXP["sse_ind"], rtol=1e-3):
        return "`sse_ind` is not the total loss at the end of the individual fit."

    if not _arr_close(_as_array(v_scipy), _EXP["v_ind"], rtol=1e-4):
        return ("`v_scipy` does not match the per subject curve_fit estimates. Loop over "
                "the rows of c_obs, fitting one subject at a time from "
                "p0=[V_START, K_START], and collect the volume from each.")
    return None


def _part_c():
    v_gm = _need("v_gm")
    cv_v = _need("cv_v_pct")
    cv_k = _need("cv_k_pct")
    ratio = _need("sse_ratio")

    for name, val in (("v_gm", v_gm), ("cv_v_pct", cv_v), ("cv_k_pct", cv_k),
                      ("sse_ratio", ratio)):
        if isinstance(val, _torch.Tensor):
            return "`%s` is still a tensor. Call .item() on it." % name
        if _as_scalar(val) is None:
            return "`%s` should be a single number." % name

    got = _as_scalar(v_gm)
    if _num_close(got, _EXP["v_mean"], rtol=1e-6):
        return ("`v_gm` is the arithmetic mean. The geometric mean is "
                "exp(mean(log(v_ind))), and it comes out slightly lower.")
    if not _num_close(got, _EXP["v_gm"], rtol=1e-4):
        return "`v_gm` is not the geometric mean of the 12 fitted volumes."

    got = _as_scalar(cv_v)
    if _num_close(got, _EXP["cv_v_pct"] / 100.0, rtol=1e-3):
        return "`cv_v_pct` is a fraction. The task asks for a percentage, so multiply by 100."
    if _num_close(got, _EXP["sd_log_v_pct"], rtol=1e-4):
        return ("`cv_v_pct` is 100 times the standard deviation of the logs, which is the "
                "small variability approximation. The task asks for "
                "100 * sqrt(exp(var(log V)) - 1).")
    if not _num_close(got, _EXP["cv_v_pct"], rtol=1e-3):
        return ("`cv_v_pct` is not the between subject CV of volume as a percentage. Use "
                "the sample variance of the logs, np.var(..., ddof=1).")

    if not _num_close(_as_scalar(cv_k), _EXP["cv_k_pct"], rtol=1e-3):
        return "`cv_k_pct` is not the between subject CV of the rate constant."

    got = _as_scalar(ratio)
    if _num_close(got, 1.0 / _EXP["sse_ratio"], rtol=1e-3):
        return "`sse_ratio` is upside down. The task asks for sse_pool divided by sse_ind."
    if not _num_close(got, _EXP["sse_ratio"], rtol=1e-3):
        return "`sse_ratio` is not sse_pool divided by sse_ind."
    return None
'''

# Functions this problem introduces. Keys must exist in tools/function_catalog.py.
FUNCTIONS = [
    "torch.full",
    "parameter vectors",
    "None axis indexing",
    "pooled against individual fitting",
    "between subject CV",
]
