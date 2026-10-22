PID = "W6D3"
WEEK = 6
DAY = 3
SLUG = "w6d3_torch_gradient_descent"
TITLE = "Gradient descent, written out by hand"
TOPIC = "PyTorch: the fitting loop, learning rate, log parameters, divergence"
EST = "20-30 min"

SCENARIO_MD = '''
## Scenario

You have a loss and you have its gradient. An optimiser is what turns those
two things into a parameter estimate, and the simplest one that works is four
lines long.

Writing it yourself once is worth more than reading about it ten times,
because every failure mode of every fitting tool you will ever use in this
field is visible in those four lines. A fit that will not move, a fit that
explodes, a fit that reports convergence while sitting on nonsense: all three
show up today, on a dataset you can see all of.

The subject is the same one as yesterday, so the loss and its gradient are
numbers you already recognise.
'''

CONCEPT_MD = '''
## The loop, and the three things that go wrong

**The loop.** Gradient descent says: the gradient points uphill, so step the
other way.

```
for i in range(n_iter):
    p.grad = None                 # or p.grad.zero_()
    loss = model_loss(p)          # forward
    loss.backward()               # gradients
    with torch.no_grad():
        p -= lr * p.grad          # step downhill
```

Four lines, and three of them are bookkeeping. Note the `no_grad()` around the
update. Without it torch refuses, with the message *a leaf Variable that
requires grad is being used in an in-place operation*, because you are
modifying a tensor whose history it is tracking. The update is not part of the
model, so it should not be recorded.

**Problem one: the learning rate.** The gradient tells you the direction, not
how far to go. Too small and you crawl. Too large and you step past the
minimum to a point that is worse than where you started, and the next step is
larger still. There is no universal right answer, which is why every deep
learning paper reports its learning rate.

**Problem two: the parameters are not on the same scale.** You saw yesterday
that the volume gradient and the rate gradient differ by a factor of sixteen.
One learning rate has to serve both, so it ends up too big for one and too
small for the other.

The standard fix in PK is to fit on **log parameters**. Instead of carrying `V`
around, carry `log V` and use `exp(log V)` in the model. Three things improve
at once: a fixed step in log space is a fixed *percentage* change, so the
scales stop mattering; the parameter cannot go negative, which a volume must
not do; and the loss surface for exponential models becomes a great deal more
pleasant. The cost is that you have to remember to exponentiate at the end.

**Problem three: divergence that looks like convergence.** A diverging fit does
not always announce itself with a NaN. Push the learning rate up and the
parameters can run off to absurd values where the model predicts essentially
zero at every time point. The loss then settles on the sum of the squared
observations and stops changing, so a convergence criterion based on "the loss
stopped moving" is satisfied. You will have a converged fit, a flat loss trace,
and a volume of distribution of fourteen million litres.

The lesson is the one habit worth taking from today: **always look at the
parameter estimates, not just the loss.**
'''

TASK_MD = '''
## Your task

Three parts, worked one at a time.

| Part | What you work out | Variables to fill in |
|---|---|---|
| a | one step downhill, and one step too far | `loss_before`, `loss_after_small`, `loss_after_big`, `inplace_error` |
| b | the whole loop, on log parameters, checked against scipy | `loss_hist`, `v_hat`, `k_hat`, `v_scipy`, `k_scipy` |
| c | divergence dressed up as convergence | `loss_big`, `v_big`, `zero_model_loss`, `first_nan_index` |

Below this you will find the **Setup** cell, then a **Toolkit** cell that
demonstrates every move on a tiny made up example, then **Part a**, **Part b**
and **Part c**, each with its own explanation and a cell to write in.
'''

DATA_CODE = '''
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

PROBLEM_ID = "W6D3"
TOPIC = "PyTorch: the fitting loop, learning rate, log parameters, divergence"

torch.manual_seed(0)
rng = np.random.default_rng(6202)

# The same subject as yesterday: 100 mg IV bolus, sampled over a day.
DOSE = 100.0                 # mg
V_TRUE = 35.0                # L, not known to the fit
K_TRUE = 0.12                # per h, not known to the fit

_t = np.array([0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 18.0, 24.0])
_c = (DOSE / V_TRUE) * np.exp(-K_TRUE * _t)
_c = np.round(_c * rng.lognormal(0.0, 0.07, size=_t.size), 4)

t_obs = torch.tensor(_t, dtype=torch.float64)
c_obs = torch.tensor(_c, dtype=torch.float64)

# Starting guesses, the same as yesterday.
V_START = 30.0               # L
K_START = 0.15               # per h

# Learning rates and iteration counts, all fixed so that everyone sees the
# same thing. Part a steps once on the raw parameters.
LR_SMALL = 1e-3
LR_BIG = 1e-2

# Part b runs the loop on log parameters.
LR = 0.01
N_ITER = 500

# Part c pushes the learning rate until it breaks.
LR_TOO_BIG = 0.1
N_TOO_BIG = 300
LR_HUGE = 0.5
N_HUGE = 20

print("t_obs:", t_obs)
print("c_obs:", c_obs)
print("")
print("starting guesses: V = %.1f L, k = %.2f per h" % (V_START, K_START))
print("part a learning rates: %g and %g" % (LR_SMALL, LR_BIG))
print("part b: lr %g for %d iterations on log parameters" % (LR, N_ITER))
print("part c: lr %g for %d, then lr %g for %d" % (LR_TOO_BIG, N_TOO_BIG, LR_HUGE, N_HUGE))
'''

TOOLKIT_MD = '''
## Toolkit for this problem

| What you want to do | How |
|---|---|
| clear a gradient | `p.grad = None`, or `p.grad.zero_()` if it exists |
| step downhill | `with torch.no_grad(): p -= lr * p.grad` |
| read a tracked tensor as a float | `p.item()`, or `torch.exp(p).item()` |
| fit on a log parameter | carry `log_v`, use `torch.exp(log_v)` in the model |
| record a trace | append `loss.item()` to a list each iteration |
| find the first True in a boolean array | `np.argmax(flags)` |
| is anything not a number | `np.isnan(arr)`, `np.isfinite(arr)` |

`p.grad = None` and `p.grad.zero_()` both clear the gradient. The difference
only matters on the very first iteration, where `.grad` does not exist yet and
`zero_()` would fail on `None`. Setting it to `None` always works, which is why
the loops below do that.
'''

TOOLKIT_CODE = '''
import numpy as np
import torch

print("--- 1. the four line loop, on a function whose answer you know ---")
# Minimise f(x) = (x - 4) ** 2, whose minimum is obviously at x = 4.
demo_x = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
for demo_i in range(25):
    demo_x.grad = None
    demo_loss = (demo_x - 4.0) ** 2
    demo_loss.backward()
    with torch.no_grad():
        demo_x -= 0.1 * demo_x.grad
    if demo_i in (0, 1, 2, 24):
        print("iter %2d  x = %.6f  loss = %.6f" % (demo_i + 1, demo_x.item(), demo_loss.item()))

print("")
print("--- 2. without no_grad, torch refuses the update ---")
demo_y = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
((demo_y - 4.0) ** 2).backward()
try:
    demo_y -= 0.1 * demo_y.grad
except RuntimeError as err:
    print("RuntimeError:", str(err)[:80])
print("the update is bookkeeping, not model arithmetic, so it should not be recorded")

print("")
print("--- 3. the learning rate decides everything ---")
for demo_lr in (0.01, 0.5, 1.01):
    demo_z = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    demo_trace = []
    for _ in range(30):
        demo_z.grad = None
        demo_l = (demo_z - 4.0) ** 2
        demo_l.backward()
        with torch.no_grad():
            demo_z -= demo_lr * demo_z.grad
        demo_trace.append(demo_l.item())
    print("lr %-5g  x after 30 steps %14.4f  loss %12.4g" % (demo_lr, demo_z.item(), demo_trace[-1]))
print("too small and it crawls, just right and it lands, too large and it leaves")

print("")
print("--- 4. a log parameter, so a step is a percentage ---")
demo_log_p = torch.tensor(np.log(2.0), dtype=torch.float64, requires_grad=True)
print("p starts at %.4f" % torch.exp(demo_log_p).item())
with torch.no_grad():
    demo_log_p -= 0.05
print("after subtracting 0.05 from log p, p = %.4f, a change of %.2f pct"
      % (torch.exp(demo_log_p).item(), 100.0 * (np.exp(-0.05) - 1.0)))
print("the same 0.05 step on a parameter of 2000 would also move it 4.88 pct")

print("")
print("--- 5. finding the first True in a boolean array ---")
demo_vals = np.array([1.0, 2.0, np.nan, 4.0])
print("values      :", demo_vals)
print("isnan       :", np.isnan(demo_vals))
print("first NaN at:", int(np.argmax(np.isnan(demo_vals))), "(a 0 based index)")
print("careful: argmax returns 0 when nothing is True, so check .any() as well")
print("any NaN     :", bool(np.isnan(demo_vals).any()))
'''

PARTS = [
    dict(
        label="a",
        prompt_md='''
## Part a. One step, twice

The model and loss are yesterday's, with both parameters free:

```
c_pred = (DOSE / v) * exp(-k * t_obs)
loss = ((c_pred - c_obs) ** 2).sum()
```

Take exactly one gradient descent step on the raw parameters, from `V_START`
and `K_START`, with learning rate `LR_SMALL`. Record the loss before the step
in `loss_before` and the loss after it in `loss_after_small`. Both as plain
floats.

Then start over from fresh tensors at the same starting values and take one
step with `LR_BIG`, which is ten times larger. Record that loss in
`loss_after_big`.

The small step improves the fit. The big one makes it worse, from the identical
starting point, using the identical gradient. The gradient told you the
direction and nothing at all about the distance, and ten times too far is past
the bottom and up the other side.

Finally, provoke the error that everybody meets on their first loop. With a
tracked tensor whose `.grad` is filled in, try `v -= LR_SMALL * v.grad`
**outside** any `no_grad()` block, inside a `try`. Catch the `RuntimeError` and
set `inplace_error` to True if it was raised, False if it was not.

Fill in `loss_before`, `loss_after_small`, `loss_after_big`, `inplace_error`.

Moves: `.backward()`, `with torch.no_grad():`, an in place update,
`try` and `except RuntimeError`.
''',
        starter='''
# Part a. One step with a sensible learning rate, one step with a reckless one.

v1 = None            # YOUR CODE: float64 tensor at V_START, requires_grad=True
k1 = None            # YOUR CODE: float64 tensor at K_START, requires_grad=True

loss_before = None   # YOUR CODE: the loss at the starting guesses, as a float

# YOUR CODE: backward, then one step of size LR_SMALL inside torch.no_grad()

loss_after_small = None  # YOUR CODE: the loss after that step, as a float

# YOUR CODE: fresh tensors at the same starting values, one step of size LR_BIG

loss_after_big = None    # YOUR CODE: the loss after the big step, as a float

inplace_error = None     # YOUR CODE: True if an in place update outside
                         #            no_grad() raises RuntimeError, else False

print("loss before            : %r" % (loss_before,))
print("after one step of %-6s: %r" % (LR_SMALL, loss_after_small))
print("after one step of %-6s: %r" % (LR_BIG, loss_after_big))
print("in place update outside no_grad raises RuntimeError:", inplace_error)
''',
        solution='''
# Part a. One step with a sensible learning rate, one step with a reckless one.

v1 = torch.tensor(V_START, dtype=torch.float64, requires_grad=True)
k1 = torch.tensor(K_START, dtype=torch.float64, requires_grad=True)

loss = ((DOSE / v1) * torch.exp(-k1 * t_obs) - c_obs).pow(2).sum()
loss_before = loss.item()
loss.backward()
with torch.no_grad():
    v1 -= LR_SMALL * v1.grad
    k1 -= LR_SMALL * k1.grad
with torch.no_grad():
    loss_after_small = ((DOSE / v1) * torch.exp(-k1 * t_obs) - c_obs).pow(2).sum().item()

v2 = torch.tensor(V_START, dtype=torch.float64, requires_grad=True)
k2 = torch.tensor(K_START, dtype=torch.float64, requires_grad=True)
loss2 = ((DOSE / v2) * torch.exp(-k2 * t_obs) - c_obs).pow(2).sum()
loss2.backward()
with torch.no_grad():
    v2 -= LR_BIG * v2.grad
    k2 -= LR_BIG * k2.grad
with torch.no_grad():
    loss_after_big = ((DOSE / v2) * torch.exp(-k2 * t_obs) - c_obs).pow(2).sum().item()

v3 = torch.tensor(V_START, dtype=torch.float64, requires_grad=True)
k3 = torch.tensor(K_START, dtype=torch.float64, requires_grad=True)
((DOSE / v3) * torch.exp(-k3 * t_obs) - c_obs).pow(2).sum().backward()
try:
    v3 -= LR_SMALL * v3.grad
    inplace_error = False
except RuntimeError:
    inplace_error = True

print("loss before            : %r" % (loss_before,))
print("after one step of %-6s: %r" % (LR_SMALL, loss_after_small))
print("after one step of %-6s: %r" % (LR_BIG, loss_after_big))
print("in place update outside no_grad raises RuntimeError:", inplace_error)
''',
    ),
    dict(
        label="b",
        prompt_md='''
## Part b. The loop, on log parameters, checked against scipy

Now run the fit properly, and on log parameters so that one learning rate can
serve both.

Make `log_v` and `log_k`, tracked float64 tensors holding `np.log(V_START)` and
`np.log(K_START)`. Then loop `N_ITER` times. Each iteration:

1. clear both gradients with `log_v.grad = None` and the same for `log_k`,
2. build the predictions using `torch.exp(log_v)` and `torch.exp(log_k)` in
   place of the parameters,
3. compute the summed squared residuals,
4. `backward()`,
5. step both parameters downhill by `LR` inside `torch.no_grad()`,
6. append `loss.item()` to `loss_hist`.

Append the loss **before** the update, which is the loss of the parameters you
just evaluated. That makes `loss_hist[0]` the starting loss, so it should equal
your `loss_before` from part a.

Afterwards put the parameters back on their natural scale into `v_hat` and
`k_hat`, as plain floats.

Then check your work against a tool you already trust. Fit the same model to
the same data with `scipy.optimize.curve_fit`, starting from the same guesses,
and record its estimates in `v_scipy` and `k_scipy`.

The two should agree to seven or eight significant figures. Yours is a loop you wrote
in six lines; `curve_fit` is Levenberg-Marquardt with numerical derivatives and
decades of hardening. On a problem this small they find the same answer, which
is the reassurance you need before trusting the loop on a problem where you
cannot check.

Look at the trace too. The loss falls steeply for the first twenty iterations,
is within one percent of its final value by about iteration sixty-four, and
from roughly iteration three hundred and forty onward it only jitters in the
fifteenth decimal place, which is floating point noise rather than progress.
Four hundred of your five hundred iterations bought you nothing, and that is
normal.

Fill in `loss_hist`, `v_hat`, `k_hat`, `v_scipy`, `k_scipy`.

Moves: the loop, `torch.exp`, `.grad = None`, `torch.no_grad()`, `.item()`,
`curve_fit`.
''',
        starter='''
# Part b. The whole fit, on log parameters.

log_v = None     # YOUR CODE: tracked float64 tensor at np.log(V_START)
log_k = None     # YOUR CODE: tracked float64 tensor at np.log(K_START)

loss_hist = []   # YOUR CODE: fill this with one loss per iteration

# YOUR CODE: the loop, N_ITER iterations

v_hat = None     # YOUR CODE: exp(log_v) as a plain float
k_hat = None     # YOUR CODE: exp(log_k) as a plain float


def model(t, v, k):
    """The same model, for curve_fit, in plain numpy."""
    return (DOSE / v) * np.exp(-k * t)


v_scipy = None   # YOUR CODE: curve_fit estimate of V, from the same p0
k_scipy = None   # YOUR CODE: curve_fit estimate of k

print("hand written descent: V = %r  k = %r" % (v_hat, k_hat))
print("curve_fit           : V = %r  k = %r" % (v_scipy, k_scipy))
if loss_hist:
    print("")
    for _i in (1, 2, 5, 10, 50, 100, N_ITER):
        print("iter %3d  loss %.8f" % (_i, loss_hist[_i - 1]))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(loss_hist) + 1), loss_hist)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("sum of squared residuals")
    ax.set_title("W6D3 gradient descent trace")
    plt.show()
''',
        solution='''
# Part b. The whole fit, on log parameters.

log_v = torch.tensor(np.log(V_START), dtype=torch.float64, requires_grad=True)
log_k = torch.tensor(np.log(K_START), dtype=torch.float64, requires_grad=True)

loss_hist = []

for _ in range(N_ITER):
    log_v.grad = None
    log_k.grad = None
    c_pred = (DOSE / torch.exp(log_v)) * torch.exp(-torch.exp(log_k) * t_obs)
    loss = ((c_pred - c_obs) ** 2).sum()
    loss.backward()
    with torch.no_grad():
        log_v -= LR * log_v.grad
        log_k -= LR * log_k.grad
    loss_hist.append(loss.item())

v_hat = torch.exp(log_v).item()
k_hat = torch.exp(log_k).item()


def model(t, v, k):
    """The same model, for curve_fit, in plain numpy."""
    return (DOSE / v) * np.exp(-k * t)


_popt, _pcov = curve_fit(model, t_obs.numpy(), c_obs.numpy(), p0=[V_START, K_START])
v_scipy = float(_popt[0])
k_scipy = float(_popt[1])

print("hand written descent: V = %r  k = %r" % (v_hat, k_hat))
print("curve_fit           : V = %r  k = %r" % (v_scipy, k_scipy))
if loss_hist:
    print("")
    for _i in (1, 2, 5, 10, 50, 100, N_ITER):
        print("iter %3d  loss %.8f" % (_i, loss_hist[_i - 1]))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(loss_hist) + 1), loss_hist)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("sum of squared residuals")
    ax.set_title("W6D3 gradient descent trace")
    plt.show()
''',
    ),
    dict(
        label="c",
        prompt_md='''
## Part c. Divergence dressed up as convergence

Run the same log parameter loop again, unchanged except for the learning rate,
with `LR_TOO_BIG` for `N_TOO_BIG` iterations. Record the final loss in
`loss_big` and the final `exp(log_v)` in `v_big`, both as plain floats.

Look at what you get. The loss is flat to ten decimal places over the last
fifty iterations, which any sensible stopping rule would call convergence. The
volume is in the millions of litres.

To show what has happened, compute `zero_model_loss`, the loss of a model that
predicts exactly zero at every time point. That is just the sum of the squared
observations, and you do not need a model to work it out. Compare it with
`loss_big`: they agree to about one part in ten million. The optimiser has not
converged on a fit. It has run the volume off to infinity, where the predicted
concentration is zero everywhere, and parked on the flattest part of the
surface it could find.

This is the thing to take away from the week. A loss that stops moving is not
evidence of a fit. The number that would have told you immediately is the
volume of distribution, and the plot that would have told you immediately is
predicted against observed.

Then push harder. Run the loop with `LR_HUGE` for `N_HUGE` iterations,
appending every loss including the ones that are not finite, and record in
`first_nan_index` the position of the first NaN in that list, as the 0 based
index that `np.argmax(np.isnan(...))` gives you. It arrives within the first
handful of iterations. This failure is the kind you want, because it is loud.

Fill in `loss_big`, `v_big`, `zero_model_loss`, `first_nan_index`.

Moves: the same loop with a different learning rate, `np.isnan`, `np.argmax`.
''',
        starter='''
# Part c. A learning rate too large, then one far too large.

loss_big = None         # YOUR CODE: final loss after N_TOO_BIG iterations at LR_TOO_BIG
v_big = None            # YOUR CODE: exp(log_v) at the end of that run, as a float

zero_model_loss = None  # YOUR CODE: the loss of predicting zero everywhere

loss_hist_huge = []     # YOUR CODE: N_HUGE losses at LR_HUGE, NaNs included
first_nan_index = None  # YOUR CODE: 0 based position of the first NaN in it

print("at lr %g after %d iterations:" % (LR_TOO_BIG, N_TOO_BIG))
print("  loss  : %r" % (loss_big,))
print("  V     : %r L" % (v_big,))
print("  loss of predicting zero everywhere: %r" % (zero_model_loss,))
print("")
print("at lr %g the loss trace starts:" % LR_HUGE)
print(" ", ["%.4g" % _x for _x in loss_hist_huge[:8]])
print("  first NaN at index:", first_nan_index)
''',
        solution='''
# Part c. A learning rate too large, then one far too large.

log_v = torch.tensor(np.log(V_START), dtype=torch.float64, requires_grad=True)
log_k = torch.tensor(np.log(K_START), dtype=torch.float64, requires_grad=True)
for _ in range(N_TOO_BIG):
    log_v.grad = None
    log_k.grad = None
    c_pred = (DOSE / torch.exp(log_v)) * torch.exp(-torch.exp(log_k) * t_obs)
    loss = ((c_pred - c_obs) ** 2).sum()
    loss.backward()
    with torch.no_grad():
        log_v -= LR_TOO_BIG * log_v.grad
        log_k -= LR_TOO_BIG * log_k.grad
loss_big = loss.item()
v_big = torch.exp(log_v).item()

zero_model_loss = (c_obs ** 2).sum().item()

log_v = torch.tensor(np.log(V_START), dtype=torch.float64, requires_grad=True)
log_k = torch.tensor(np.log(K_START), dtype=torch.float64, requires_grad=True)
loss_hist_huge = []
for _ in range(N_HUGE):
    log_v.grad = None
    log_k.grad = None
    c_pred = (DOSE / torch.exp(log_v)) * torch.exp(-torch.exp(log_k) * t_obs)
    loss = ((c_pred - c_obs) ** 2).sum()
    loss.backward()
    with torch.no_grad():
        log_v -= LR_HUGE * log_v.grad
        log_k -= LR_HUGE * log_k.grad
    loss_hist_huge.append(loss.item())
first_nan_index = int(np.argmax(np.isnan(np.array(loss_hist_huge))))

print("at lr %g after %d iterations:" % (LR_TOO_BIG, N_TOO_BIG))
print("  loss  : %r" % (loss_big,))
print("  V     : %r L" % (v_big,))
print("  loss of predicting zero everywhere: %r" % (zero_model_loss,))
print("")
print("at lr %g the loss trace starts:" % LR_HUGE)
print(" ", ["%.4g" % _x for _x in loss_hist_huge[:8]])
print("  first NaN at index:", first_nan_index)
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

    def loss_at(v, k):
        pred = (dose / v) * np.exp(-k * t)
        return float(((pred - c) ** 2).sum())

    def grads_at(v, k):
        pred = (dose / v) * np.exp(-k * t)
        resid = pred - c
        gv = float((2.0 * resid * (-pred / v)).sum())
        gk = float((2.0 * resid * (-t * pred)).sum())
        return gv, gk

    gv, gk = grads_at(v0, k0)
    small = ns["LR_SMALL"]
    big = ns["LR_BIG"]

    def model(tt, v, k):
        return (dose / v) * np.exp(-k * tt)

    popt, _ = curve_fit(model, t, c, p0=[v0, k0])

    return {
        "n": int(t.size),
        "loss_before": loss_at(v0, k0),
        "loss_after_small": loss_at(v0 - small * gv, k0 - small * gk),
        "loss_after_big": loss_at(v0 - big * gv, k0 - big * gk),
        "n_iter": int(ns["N_ITER"]),
        "n_huge": int(ns["N_HUGE"]),
        "v_opt": float(popt[0]),
        "k_opt": float(popt[1]),
        "loss_opt": loss_at(float(popt[0]), float(popt[1])),
        "zero_model_loss": float((c ** 2).sum()),
    }


CHECK_CODE = '''
import torch as _torch


def _part_a():
    loss_before = _need("loss_before")
    small = _need("loss_after_small")
    big = _need("loss_after_big")
    inplace_error = _need("inplace_error")

    for name, val in (("loss_before", loss_before), ("loss_after_small", small),
                      ("loss_after_big", big)):
        if isinstance(val, _torch.Tensor):
            return "`%s` is still a tensor. Call .item() on it." % name
        if _as_scalar(val) is None:
            return "`%s` should be a single number." % name

    if not _num_close(_as_scalar(loss_before), _EXP["loss_before"], rtol=1e-6):
        return ("`loss_before` is not the summed squared residuals at V_START and K_START. "
                "The model is (DOSE / v) * exp(-k * t_obs).")

    got = _as_scalar(small)
    if _num_close(got, _EXP["loss_before"], rtol=1e-9):
        return ("`loss_after_small` equals the loss before the step, so no step was taken. "
                "The update has to be inside a torch.no_grad() block to work at all.")
    if got > _EXP["loss_before"]:
        return ("`loss_after_small` is larger than `loss_before`, so the step went uphill. "
                "Gradient descent SUBTRACTS lr * grad.")
    if not _num_close(got, _EXP["loss_after_small"], rtol=1e-6):
        return ("`loss_after_small` is not the loss after one step of size LR_SMALL from "
                "the starting guesses. Check that both parameters were stepped.")

    got = _as_scalar(big)
    if _num_close(got, _EXP["loss_after_small"], rtol=1e-9):
        return ("`loss_after_big` matches the small step, so LR_BIG was not used, or the "
                "second step was taken from the already updated parameters rather than "
                "from fresh tensors at V_START and K_START.")
    if got < _EXP["loss_before"]:
        return ("`loss_after_big` is better than the starting loss, which it should not be. "
                "Start again from fresh tensors at V_START and K_START and take ONE step "
                "of size LR_BIG.")
    if not _num_close(got, _EXP["loss_after_big"], rtol=1e-6):
        return "`loss_after_big` is not the loss after one step of size LR_BIG."

    if not isinstance(inplace_error, bool):
        return "`inplace_error` should be True or False."
    if inplace_error is not True:
        return ("`inplace_error` should be True. An in place update of a tracked leaf "
                "tensor outside no_grad() raises RuntimeError, which is why every loop "
                "wraps the step.")
    return None


def _part_b():
    hist = _need("loss_hist")
    v_hat = _need("v_hat")
    k_hat = _need("k_hat")
    v_scipy = _need("v_scipy")
    k_scipy = _need("k_scipy")

    try:
        vals = [float(x) for x in hist]
    except Exception:
        return "`loss_hist` should be a list of numbers, one loss per iteration."
    if len(vals) == 0:
        return "`loss_hist` is empty. Append loss.item() inside the loop."
    if len(vals) != _EXP["n_iter"]:
        return ("`loss_hist` has %d entries but the loop runs N_ITER = %d times."
                % (len(vals), _EXP["n_iter"]))
    if not _num_close(vals[0], _EXP["loss_before"], rtol=1e-6):
        if _num_close(vals[0], _EXP["loss_opt"], rtol=1e-3):
            return ("`loss_hist` looks reversed: its first entry is the converged loss. "
                    "Append as you go, so entry 0 is the starting loss.")
        return ("`loss_hist[0]` should be the loss at the starting guesses, the same number "
                "as part a's `loss_before`. Append the loss BEFORE the update.")
    rises = [i for i in range(len(vals) - 1) if vals[i + 1] > vals[i] + 1e-12]
    if rises:
        return ("`loss_hist` goes up at iteration %d, so the descent is not descending. "
                "The usual cause is not clearing .grad at the top of the loop, which "
                "makes every step larger than the last." % (rises[0] + 2))
    if not _num_close(vals[-1], _EXP["loss_opt"], rtol=1e-4):
        return ("the last entry of `loss_hist` is not the converged loss. At this learning "
                "rate N_ITER iterations is far more than enough, so check the model and "
                "that exp() is applied to both log parameters.")

    for name, val in (("v_hat", v_hat), ("k_hat", k_hat),
                      ("v_scipy", v_scipy), ("k_scipy", k_scipy)):
        if isinstance(val, _torch.Tensor):
            return "`%s` is still a tensor. Call .item() on it." % name
        if _as_scalar(val) is None:
            return "`%s` should be a single number." % name

    got = _as_scalar(v_hat)
    if _num_close(got, _np.log(_EXP["v_opt"]), rtol=1e-3):
        return "`v_hat` is still on the log scale. Exponentiate it."
    if not _num_close(got, _EXP["v_opt"], rtol=1e-3):
        return ("`v_hat` is not the fitted volume. It should be exp(log_v) after the loop, "
                "on the natural scale in litres.")

    got = _as_scalar(k_hat)
    if _num_close(got, _np.log(_EXP["k_opt"]), rtol=1e-3):
        return "`k_hat` is still on the log scale. Exponentiate it."
    if not _num_close(got, _EXP["k_opt"], rtol=1e-3):
        return "`k_hat` is not the fitted rate constant."

    got = _as_scalar(v_scipy)
    if not _num_close(got, _EXP["v_opt"], rtol=1e-5):
        if _num_close(got, _EXP["v_opt"], rtol=1e-2):
            return ("`v_scipy` is close but not converged. Pass p0=[V_START, K_START] and "
                    "let curve_fit run to its own tolerance.")
        return ("`v_scipy` is not the curve_fit estimate of V. Fit the same model to "
                "t_obs and c_obs as numpy arrays.")
    if not _num_close(_as_scalar(k_scipy), _EXP["k_opt"], rtol=1e-5):
        return "`k_scipy` is not the curve_fit estimate of k."
    return None


def _part_c():
    loss_big = _need("loss_big")
    v_big = _need("v_big")
    zero_loss = _need("zero_model_loss")
    first_nan = _need("first_nan_index")

    for name, val in (("loss_big", loss_big), ("v_big", v_big),
                      ("zero_model_loss", zero_loss)):
        if isinstance(val, _torch.Tensor):
            return "`%s` is still a tensor. Call .item() on it." % name
        if _as_scalar(val) is None:
            return "`%s` should be a single number." % name

    if not _num_close(_as_scalar(zero_loss), _EXP["zero_model_loss"], rtol=1e-9):
        return ("`zero_model_loss` should be the loss of predicting zero at every time "
                "point, which is the sum of the squared observations.")

    got = _as_scalar(v_big)
    if _num_close(got, _EXP["v_opt"], rtol=1e-2):
        return ("`v_big` is the sensible fitted volume, so the run used the part b learning "
                "rate. Part c uses LR_TOO_BIG.")
    if not (got > 1e5):
        return ("`v_big` should be enormous, millions of litres. At LR_TOO_BIG the volume "
                "runs away. If yours is reasonable, check that LR_TOO_BIG was used for all "
                "N_TOO_BIG iterations and that both gradients are cleared each time.")

    got = _as_scalar(loss_big)
    if _num_close(got, _EXP["loss_opt"], rtol=1e-2):
        return ("`loss_big` is the converged loss from part b. Part c uses LR_TOO_BIG, "
                "which does not converge to a fit.")
    if not _num_close(got, _EXP["zero_model_loss"], rtol=1e-4):
        return ("`loss_big` should have settled on essentially the zero model loss, because "
                "the runaway volume makes every prediction about zero. Record the loss of "
                "the last iteration of the LR_TOO_BIG run.")

    hist_huge = globals().get("loss_hist_huge")
    if hist_huge is not None and len(hist_huge) and len(hist_huge) != _EXP["n_huge"]:
        return ("`loss_hist_huge` has %d entries but the run is N_HUGE = %d iterations. "
                "Append every loss, including the ones that are not finite."
                % (len(hist_huge), _EXP["n_huge"]))

    try:
        idx = int(first_nan)
    except Exception:
        return "`first_nan_index` should be an integer index."
    if idx == 0:
        return ("`first_nan_index` is 0, which is what np.argmax gives when nothing is "
                "True. The first loss is finite, so either no NaN appeared or the list is "
                "not the loss trace. At LR_HUGE the loss should blow up within a few "
                "iterations.")
    if idx < 2 or idx > 10:
        return ("`first_nan_index` of %d is not where the NaN turns up. At LR_HUGE the loss "
                "should first come out as not a number within the first handful of "
                "iterations. Check that both gradients are cleared each iteration and that "
                "every loss is appended, non-finite ones included." % idx)
    return None
'''

# Functions this problem introduces. Keys must exist in tools/function_catalog.py.
FUNCTIONS = [
    "gradient descent loop",
    "learning rate",
    "log parameters",
    "np.isnan",
    "np.argmax on booleans",
]
