PID = "W6D4"
WEEK = 6
DAY = 4
SLUG = "w6d4_torch_module_optimizer"
TITLE = "nn.Module and an optimiser: the loop everyone actually writes"
TOPIC = "PyTorch: nn.Module, nn.Parameter, torch.optim, Adam against plain SGD"
EST = "20-30 min"

SCENARIO_MD = '''
## Scenario

Yesterday's loop works, but it has two parameters loose in the global
namespace, and every line of it mentions them by name. Add a third parameter
and you edit the loop in four places. Add between subject variability and the
whole thing becomes unmanageable.

PyTorch's answer is to put the parameters inside the model, where they belong,
and to put the update rule inside an optimiser object. What is left is a
training loop of five lines that never changes again, no matter how big the
model gets. You will see that exact loop in every PyTorch tutorial, every
repository and every paper's code, so it is worth writing once with your own
hands on a model you understand completely.

Same subject, same model, same answer. Different machinery.
'''

CONCEPT_MD = '''
## nn.Module, nn.Parameter, and the five line loop

**`nn.Module` is a container that knows what its parameters are.** You subclass
it, create your parameters in `__init__`, and define `forward`, which is the
arithmetic that turns inputs into predictions.

```
class OneCompartmentIV(torch.nn.Module):
    def __init__(self, v_init, k_init):
        super().__init__()
        self.log_v = torch.nn.Parameter(torch.tensor(np.log(v_init), dtype=torch.float64))
        self.log_k = torch.nn.Parameter(torch.tensor(np.log(k_init), dtype=torch.float64))

    def forward(self, t):
        v = torch.exp(self.log_v)
        k = torch.exp(self.log_k)
        return (DOSE / v) * torch.exp(-k * t)
```

Three things to notice.

`super().__init__()` comes first. Skip it and torch raises an error about
calling `__init__` before assigning parameters, because the bookkeeping
dictionary it needs does not exist yet.

**`nn.Parameter` is a tensor that announces itself.** It is an ordinary tensor
with `requires_grad=True` set for you, plus one extra property: assigning one
to an attribute of a Module registers it, so that `model.parameters()` can find
it later. A plain tensor assigned to `self.something` is invisible to the
optimiser, which is a silent failure worth knowing about: the fit runs, and
that parameter never moves.

**You call the model, not `forward`.** `model(t)` runs `forward` plus the hooks
that torch wraps around it. Calling `model.forward(t)` directly works today and
will quietly break something eventually.

**The loop.**

```
optimiser = torch.optim.Adam(model.parameters(), lr=0.05)
for i in range(n_iter):
    optimiser.zero_grad()            # clear last iteration's gradients
    loss = ((model(t_obs) - c_obs) ** 2).sum()
    loss.backward()                  # fill in every parameter's .grad
    optimiser.step()                 # apply the update rule
```

`zero_grad()` replaces the `p.grad = None` you were writing by hand, for every
parameter at once. `step()` replaces the `with torch.no_grad(): p -= lr *
p.grad` block, also for every parameter at once. The optimiser was handed
`model.parameters()`, so it knows what to clear and what to step.

**The optimisers.** `torch.optim.SGD(params, lr=lr)` with nothing else set is
exactly yesterday's loop, line for line. `torch.optim.Adam` keeps a running
estimate of the size of each parameter's recent gradients and divides by it, so
every parameter gets its own effective step size, adapted as it goes. That is
precisely the problem log parameters were invented to work around, which is why
Adam makes the badly scaled version survivable too, and why it is the default
choice when you have no reason to prefer another.

Today you fit the same model with both at the same learning rate. One lands on
the answer; the other does not.
'''

TASK_MD = '''
## Your task

Three parts, worked one at a time.

| Part | What you work out | Variables to fill in |
|---|---|---|
| a | the model as a Module, and proof it is wired up | `n_params`, `param_names`, `all_require_grad`, `loss_start` |
| b | the five line loop, Adam against plain SGD | `loss_hist_adam`, `loss_hist_sgd`, `v_hat`, `k_hat` |
| c | the numbers a pharmacologist would actually ask for | `cl_hat`, `t_half_hat`, `auc_inf`, `resid_sd` |

Below this you will find the **Setup** cell, then a **Toolkit** cell that
demonstrates every move on a tiny made up example, then **Part a**, **Part b**
and **Part c**, each with its own explanation and a cell to write in.
'''

DATA_CODE = '''
import numpy as np
import torch
import matplotlib.pyplot as plt

PROBLEM_ID = "W6D4"
TOPIC = "PyTorch: nn.Module, nn.Parameter, torch.optim, Adam against plain SGD"

torch.manual_seed(0)
rng = np.random.default_rng(6202)

# The same subject as the last two days: 100 mg IV bolus, sampled over a day.
DOSE = 100.0                 # mg
V_TRUE = 35.0                # L, not known to the fit
K_TRUE = 0.12                # per h, not known to the fit

_t = np.array([0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 18.0, 24.0])
_c = (DOSE / V_TRUE) * np.exp(-K_TRUE * _t)
_c = np.round(_c * rng.lognormal(0.0, 0.07, size=_t.size), 4)

t_obs = torch.tensor(_t, dtype=torch.float64)
c_obs = torch.tensor(_c, dtype=torch.float64)

V_START = 30.0               # L
K_START = 0.15               # per h

# One learning rate, used for both optimisers, so the comparison is fair.
LR = 0.05
N_ITER = 500

# A fine grid for plotting the fitted curve.
t_grid = torch.linspace(0.25, 24.0, 96, dtype=torch.float64)

print("t_obs:", t_obs)
print("c_obs:", c_obs)
print("")
print("starting guesses: V = %.1f L, k = %.2f per h" % (V_START, K_START))
print("both optimisers get lr = %g and %d iterations" % (LR, N_ITER))
'''

TOOLKIT_MD = '''
## Toolkit for this problem

| What you want to do | How |
|---|---|
| start a model | `class M(torch.nn.Module):` then `def __init__` and `def forward` |
| the first line of `__init__` | `super().__init__()` |
| make a fittable parameter | `torch.nn.Parameter(torch.tensor(x, dtype=torch.float64))` |
| list the parameters | `list(model.parameters())` |
| list them with names | `model.named_parameters()` |
| predict | `model(t)`, never `model.forward(t)` |
| build an optimiser | `torch.optim.Adam(model.parameters(), lr=LR)` |
| clear all gradients | `optimiser.zero_grad()` |
| apply the update | `optimiser.step()` |
| save the fitted values | `model.state_dict()` |
| predict without recording | `with torch.no_grad(): model(t_grid)` |

The demo below fits a straight line, so that the only new things on screen are
the Module and the optimiser.
'''

TOOLKIT_CODE = '''
import numpy as np
import torch

print("--- 1. a Module with two parameters ---")


class DemoLine(torch.nn.Module):
    """y = a * x + b. Nothing here is the answer to the problem."""

    def __init__(self, a_init, b_init):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(a_init, dtype=torch.float64))
        self.b = torch.nn.Parameter(torch.tensor(b_init, dtype=torch.float64))

    def forward(self, x):
        return self.a * x + self.b


demo_line = DemoLine(0.0, 0.0)
print("parameters      :", [p.item() for p in demo_line.parameters()])
print("named           :", [(n, p.item()) for n, p in demo_line.named_parameters()])
print("requires_grad   :", [p.requires_grad for p in demo_line.parameters()])
print("state_dict keys :", list(demo_line.state_dict().keys()))
print("nn.Parameter sets requires_grad for you, and registers itself with the Module")

print("")
print("--- 2. a plain tensor attribute is INVISIBLE to the optimiser ---")


class DemoBroken(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.good = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        self.bad = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)

    def forward(self, x):
        return self.good * x + self.bad


demo_broken = DemoBroken()
print("parameters found:", [n for n, _ in demo_broken.named_parameters()])
print("`bad` tracks gradients but is not registered, so no optimiser will ever move it")

print("")
print("--- 3. the five line loop ---")
demo_x = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
demo_y = 2.5 * demo_x + 1.0          # the answer is a = 2.5, b = 1.0
demo_model = DemoLine(0.0, 0.0)
demo_opt = torch.optim.Adam(demo_model.parameters(), lr=0.1)
for demo_i in range(800):
    demo_opt.zero_grad()
    demo_loss = ((demo_model(demo_x) - demo_y) ** 2).sum()
    demo_loss.backward()
    demo_opt.step()
print("fitted a = %.6f (true 2.5), b = %.6f (true 1.0)"
      % (demo_model.a.item(), demo_model.b.item()))

print("")
print("--- 4. optim.SGD with nothing set is the loop you wrote yesterday ---")
# By hand.
demo_p = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
for _ in range(25):
    demo_p.grad = None
    demo_l = (demo_p - 4.0) ** 2
    demo_l.backward()
    with torch.no_grad():
        demo_p -= 0.1 * demo_p.grad
# With an optimiser.
demo_q = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
demo_sgd = torch.optim.SGD([demo_q], lr=0.1)
for _ in range(25):
    demo_sgd.zero_grad()
    demo_l2 = (demo_q - 4.0) ** 2
    demo_l2.backward()
    demo_sgd.step()
print("by hand   : %.15f" % demo_p.item())
print("optim.SGD : %.15f" % demo_q.item())
print("identical :", demo_p.item() == demo_q.item())

print("")
print("--- 5. predicting afterwards, without recording ---")
with torch.no_grad():
    demo_pred = demo_model(torch.tensor([10.0], dtype=torch.float64))
print("prediction at x = 10:", demo_pred.item(), " grad_fn:", demo_pred.grad_fn)
'''

PARTS = [
    dict(
        label="a",
        prompt_md='''
## Part a. The model as a Module

Fill in the class below. Two parameters, named exactly `log_v` and `log_k` so
that the check can find them, each an `nn.Parameter` holding the log of the
starting value as a float64 tensor. `forward` exponentiates them and returns

```
(DOSE / v) * exp(-k * t)
```

Then make one instance at the starting guesses and inspect it, which is how you
confirm a model is wired up before wasting an hour on a fit that cannot move.

- `n_params`, how many entries `list(model0.parameters())` has,
- `param_names`, the parameter names from `model0.named_parameters()`, sorted
  alphabetically, as a list of strings,
- `all_require_grad`, True if every parameter has `requires_grad` set. You will
  not have set it anywhere. `nn.Parameter` does that for you, which is the
  main reason it exists,
- `loss_start`, the summed squared residuals of the fresh model, as a plain
  float.

`loss_start` is the same number you have computed on each of the last two days,
which is the point of recomputing it here: the machinery changed and the
arithmetic did not.

Fill in the class, then `n_params`, `param_names`, `all_require_grad`,
`loss_start`.

Moves: `torch.nn.Module`, `super().__init__()`, `torch.nn.Parameter`,
`model.named_parameters()`, `model(t)`.
''',
        starter='''
# Part a. The model as a Module.

class OneCompartmentIV(torch.nn.Module):
    """One compartment IV bolus, fitted on log parameters."""

    def __init__(self, v_init, k_init):
        super().__init__()
        self.log_v = None   # YOUR CODE: nn.Parameter holding np.log(v_init), float64
        self.log_k = None   # YOUR CODE: nn.Parameter holding np.log(k_init), float64

    def forward(self, t):
        v = None            # YOUR CODE: exp of log_v
        k = None            # YOUR CODE: exp of log_k
        return None         # YOUR CODE: (DOSE / v) * exp(-k * t)


model0 = OneCompartmentIV(V_START, K_START)

n_params = None          # YOUR CODE: how many parameters the Module reports
param_names = None       # YOUR CODE: their names, sorted, as a list of strings
all_require_grad = None  # YOUR CODE: True if every parameter requires grad
loss_start = None        # YOUR CODE: summed squared residuals of the fresh model

print("parameters reported:", n_params)
print("names              :", param_names)
print("all require grad   :", all_require_grad)
print("loss at the start  : %r" % (loss_start,))
''',
        solution='''
# Part a. The model as a Module.

class OneCompartmentIV(torch.nn.Module):
    """One compartment IV bolus, fitted on log parameters."""

    def __init__(self, v_init, k_init):
        super().__init__()
        self.log_v = torch.nn.Parameter(torch.tensor(np.log(v_init), dtype=torch.float64))
        self.log_k = torch.nn.Parameter(torch.tensor(np.log(k_init), dtype=torch.float64))

    def forward(self, t):
        v = torch.exp(self.log_v)
        k = torch.exp(self.log_k)
        return (DOSE / v) * torch.exp(-k * t)


model0 = OneCompartmentIV(V_START, K_START)

n_params = len(list(model0.parameters()))
param_names = sorted(name for name, _ in model0.named_parameters())
all_require_grad = all(p.requires_grad for p in model0.parameters())
with torch.no_grad():
    loss_start = ((model0(t_obs) - c_obs) ** 2).sum().item()

print("parameters reported:", n_params)
print("names              :", param_names)
print("all require grad   :", all_require_grad)
print("loss at the start  : %r" % (loss_start,))
''',
    ),
    dict(
        label="b",
        prompt_md='''
## Part b. The five line loop, twice

Fit the model with `torch.optim.Adam` at learning rate `LR` for `N_ITER`
iterations. Build a fresh `OneCompartmentIV` at the starting guesses first, so
that neither fit inherits the other's progress.

The loop is the five lines from the concept section: `zero_grad`, forward and
loss, `backward`, `step`, and record `loss.item()` in `loss_hist_adam`. Record
the loss before the step, as you did yesterday, so the first entry is the
starting loss.

Afterwards read the fitted parameters back onto their natural scale into
`v_hat` and `k_hat`, as plain floats.

Then do the whole thing again with `torch.optim.SGD` at the **same** learning
rate, into `loss_hist_sgd`.

Adam lands on the optimum: the same volume and rate constant you reached
yesterday, and the same loss to ten decimal places. Plain SGD at this learning
rate does not converge at all. Watch its column in the printout: the loss goes
up before it comes down, sits above the starting loss at iteration ten, is
worse still at a hundred, and the value it happens to show at five hundred, ten
times the Adam loss, is a point on a bouncing trace rather than an estimate.
Its steps in the volume direction are larger than the curvature of the surface
tolerates, so it overshoots, overshoots back, and never settles.

That is not a criticism of SGD. Yesterday the same algorithm at a fifth of this
learning rate converged perfectly. It is an illustration of the one real
difficulty in gradient based fitting: the step size that works depends on the
problem, the parameterisation and the point you are standing on, and Adam's
contribution is to work most of that out for itself.

Fill in `loss_hist_adam`, `loss_hist_sgd`, `v_hat`, `k_hat`.

Moves: `torch.optim.Adam`, `torch.optim.SGD`, `zero_grad`, `backward`, `step`.
''',
        starter='''
# Part b. The loop, with Adam and then with plain SGD.

model = OneCompartmentIV(V_START, K_START)
opt = None             # YOUR CODE: torch.optim.Adam over model.parameters(), lr = LR

loss_hist_adam = []    # YOUR CODE: one loss per iteration, filled in by the loop

# YOUR CODE: the loop. N_ITER iterations of
#   opt.zero_grad() / loss = ... / loss.backward() / opt.step() / record the loss

v_hat = None           # YOUR CODE: exp(model.log_v) as a plain float
k_hat = None           # YOUR CODE: exp(model.log_k) as a plain float

model_sgd = OneCompartmentIV(V_START, K_START)
loss_hist_sgd = []     # YOUR CODE: the same loop with torch.optim.SGD at the same LR

print("Adam: V = %r  k = %r" % (v_hat, k_hat))
if loss_hist_adam and loss_hist_sgd:
    print("")
    print("iter     Adam loss      SGD loss")
    for _i in (1, 2, 10, 50, 100, N_ITER):
        print("%4d  %13.8f %13.8f" % (_i, loss_hist_adam[_i - 1], loss_hist_sgd[_i - 1]))
''',
        solution='''
# Part b. The loop, with Adam and then with plain SGD.

model = OneCompartmentIV(V_START, K_START)
opt = torch.optim.Adam(model.parameters(), lr=LR)

loss_hist_adam = []
for _ in range(N_ITER):
    opt.zero_grad()
    loss = ((model(t_obs) - c_obs) ** 2).sum()
    loss.backward()
    opt.step()
    loss_hist_adam.append(loss.item())

v_hat = torch.exp(model.log_v).item()
k_hat = torch.exp(model.log_k).item()

model_sgd = OneCompartmentIV(V_START, K_START)
opt_sgd = torch.optim.SGD(model_sgd.parameters(), lr=LR)

loss_hist_sgd = []
for _ in range(N_ITER):
    opt_sgd.zero_grad()
    loss_s = ((model_sgd(t_obs) - c_obs) ** 2).sum()
    loss_s.backward()
    opt_sgd.step()
    loss_hist_sgd.append(loss_s.item())

print("Adam: V = %r  k = %r" % (v_hat, k_hat))
if loss_hist_adam and loss_hist_sgd:
    print("")
    print("iter     Adam loss      SGD loss")
    for _i in (1, 2, 10, 50, 100, N_ITER):
        print("%4d  %13.8f %13.8f" % (_i, loss_hist_adam[_i - 1], loss_hist_sgd[_i - 1]))
''',
    ),
    dict(
        label="c",
        prompt_md='''
## Part c. The numbers a pharmacologist would ask for

Nobody wants `log_v`. Turn the Adam fit into the four numbers that would go
into a report, all as plain floats, all computed under `torch.no_grad()`
because none of this is being differentiated.

- `cl_hat`, the clearance in L/h, which for this model is `k * V`,
- `t_half_hat`, the terminal half life in hours, `ln(2) / k`,
- `auc_inf`, the area under the curve from zero to infinity in mg*h/L. For an
  IV bolus with first order elimination the whole curve integrates exactly, so
  this is `DOSE / CL` and needs no trapezoids,
- `resid_sd`, the residual standard deviation on the observed scale, as
  `sqrt(SSE / (n - p))` where SSE is the summed squared residuals at the fitted
  parameters, `n` is the number of samples and `p` is the number of fitted
  parameters, which is 2.

The `n - p` matters. Dividing by `n` gives you a number that is too small by
twelve percent here, and the habit of ignoring degrees of freedom is how people
end up quoting confidence intervals that do not cover.

Sanity check what comes out. The clearance should be a few L/h, the half life
around six hours, the AUC in the low twenties, and the residual standard
deviation under a tenth of a mg/L.

Then be suspicious of that last one. It is a single absolute number describing
a model whose error is plainly proportional: the largest residual is 0.116 mg/L
at 4 h, where the concentration is 1.7, and the smallest is 0.002 mg/L at 24 h,
where the concentration is 0.16. One standard deviation cannot describe both.
Divide each residual by its own prediction instead and the spread is 6.6
percent, which is a close match to the 7 percent the data was built with. That
is the argument for fitting PK data on log concentrations, or with a
proportional error model, rather than on the additive scale used here.

The cell also plots the fitted curve through the observations, which is the
check that no number can replace. Note that the predictions for the plot are
computed inside `no_grad` and then pulled out with `.numpy()`. Without the
`no_grad`, `.numpy()` would refuse, because the tensor would be carrying a
gradient history.

Fill in `cl_hat`, `t_half_hat`, `auc_inf`, `resid_sd`.

Moves: `torch.no_grad()`, `.item()`, `np.log(2)`, `np.sqrt`.
''',
        starter='''
# Part c. From log parameters to a report.

cl_hat = None       # YOUR CODE: clearance in L/h
t_half_hat = None   # YOUR CODE: terminal half life in h
auc_inf = None      # YOUR CODE: AUC from 0 to infinity in mg*h/L
resid_sd = None     # YOUR CODE: sqrt(SSE / (n - 2)) at the fitted parameters

print("V      = %r L" % (v_hat,))
print("k      = %r per h" % (k_hat,))
print("CL     = %r L/h" % (cl_hat,))
print("t half = %r h" % (t_half_hat,))
print("AUC    = %r mg*h/L" % (auc_inf,))
print("residual SD = %r mg/L" % (resid_sd,))

if v_hat is not None:
    with torch.no_grad():
        fitted = model(t_grid).numpy()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t_obs.numpy(), c_obs.numpy(), "o", label="observed")
    ax.plot(t_grid.numpy(), fitted, "-", label="fitted")
    ax.set_xlabel("time (h)")
    ax.set_ylabel("concentration (mg/L)")
    ax.set_title("W6D4 one compartment fit")
    ax.legend()
    plt.show()
''',
        solution='''
# Part c. From log parameters to a report.

with torch.no_grad():
    cl_hat = k_hat * v_hat
    t_half_hat = float(np.log(2.0) / k_hat)
    auc_inf = DOSE / cl_hat
    sse = ((model(t_obs) - c_obs) ** 2).sum().item()
    resid_sd = float(np.sqrt(sse / (t_obs.numel() - 2)))

print("V      = %r L" % (v_hat,))
print("k      = %r per h" % (k_hat,))
print("CL     = %r L/h" % (cl_hat,))
print("t half = %r h" % (t_half_hat,))
print("AUC    = %r mg*h/L" % (auc_inf,))
print("residual SD = %r mg/L" % (resid_sd,))

if v_hat is not None:
    with torch.no_grad():
        fitted = model(t_grid).numpy()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t_obs.numpy(), c_obs.numpy(), "o", label="observed")
    ax.plot(t_grid.numpy(), fitted, "-", label="fitted")
    ax.set_xlabel("time (h)")
    ax.set_ylabel("concentration (mg/L)")
    ax.set_title("W6D4 one compartment fit")
    ax.legend()
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

    def model(tt, v, k):
        return (dose / v) * np.exp(-k * tt)

    popt, _ = curve_fit(model, t, c, p0=[v0, k0])
    v_opt, k_opt = float(popt[0]), float(popt[1])
    sse_opt = float(((model(t, v_opt, k_opt) - c) ** 2).sum())
    cl = k_opt * v_opt
    n = int(t.size)
    return {
        "n": n,
        "n_iter": int(ns["N_ITER"]),
        "loss_start": float(((model(t, v0, k0) - c) ** 2).sum()),
        "v_opt": v_opt,
        "k_opt": k_opt,
        "loss_opt": sse_opt,
        "cl": cl,
        "t_half": float(np.log(2.0) / k_opt),
        "auc_inf": dose / cl,
        "resid_sd": float(np.sqrt(sse_opt / (n - 2))),
        "resid_sd_wrong": float(np.sqrt(sse_opt / n)),
    }


CHECK_CODE = '''
import torch as _torch


def _part_a():
    n_params = _need("n_params")
    param_names = _need("param_names")
    all_rg = _need("all_require_grad")
    loss_start = _need("loss_start")

    cls = globals().get("OneCompartmentIV")
    if cls is None:
        return "`OneCompartmentIV` is not defined. The class skeleton is in the part a cell."
    try:
        probe = cls(30.0, 0.15)
    except Exception as exc:
        return "`OneCompartmentIV(30.0, 0.15)` raised %s: %s" % (type(exc).__name__, exc)
    if not isinstance(probe, _torch.nn.Module):
        return "`OneCompartmentIV` should subclass torch.nn.Module."
    probe_params = list(probe.parameters())
    if len(probe_params) == 0:
        return ("the model reports no parameters. Each one has to be a torch.nn.Parameter "
                "assigned to an attribute of self, which is what registers it.")
    if any(p.dtype != _torch.float64 for p in probe_params):
        return "the parameters should be float64. Pass dtype=torch.float64 to torch.tensor."

    got = _as_scalar(n_params)
    if got is None:
        return "`n_params` should be a single integer."
    if int(got) != 2:
        return ("`n_params` should be 2, one for each fitted parameter, and it should come "
                "from len(list(model0.parameters())).")

    try:
        names = sorted(str(x) for x in param_names)
    except Exception:
        return "`param_names` should be a list of strings."
    if names != ["log_k", "log_v"]:
        return ("`param_names` should be ['log_k', 'log_v'], sorted alphabetically. The task "
                "asks for those two names exactly, and it got %r." % (names,))

    if not isinstance(all_rg, bool):
        return "`all_require_grad` should be True or False."
    if all_rg is not True:
        return ("`all_require_grad` should be True. nn.Parameter sets requires_grad for you, "
                "so there is nothing to set by hand.")

    if isinstance(loss_start, _torch.Tensor):
        return "`loss_start` is still a tensor. Call .item() on it."
    got = _as_scalar(loss_start)
    if got is None:
        return "`loss_start` should be a single number."
    if _num_close(got, _EXP["loss_start"] / _EXP["n"], rtol=1e-6):
        return "`loss_start` is the mean, not the sum, of the squared residuals."
    if _num_close(got, _EXP["loss_opt"], rtol=1e-2):
        return ("`loss_start` is the converged loss, so the model was fitted first. It "
                "should be the loss of a fresh model at the starting guesses.")
    if not _num_close(got, _EXP["loss_start"], rtol=1e-6):
        return ("`loss_start` is not the summed squared residuals of the fresh model. Check "
                "that forward returns (DOSE / v) * exp(-k * t) with v and k exponentiated.")

    try:
        with _torch.no_grad():
            pred = probe(t_obs)
    except Exception as exc:
        return "calling the model raised %s: %s" % (type(exc).__name__, exc)
    if pred is None:
        return "`forward` returns None. It should return the predicted concentrations."
    if not isinstance(pred, _torch.Tensor):
        return "`forward` should return a tensor."
    if tuple(pred.shape) != tuple(t_obs.shape):
        return ("`forward` returned shape %s for %d input times. It should return one "
                "prediction per time." % (tuple(pred.shape), t_obs.numel()))
    return None


def _part_b():
    hist_a = _need("loss_hist_adam")
    hist_s = _need("loss_hist_sgd")
    v_hat = _need("v_hat")
    k_hat = _need("k_hat")

    try:
        vals_a = [float(x) for x in hist_a]
        vals_s = [float(x) for x in hist_s]
    except Exception:
        return "`loss_hist_adam` and `loss_hist_sgd` should be lists of numbers."
    for name, vals in (("loss_hist_adam", vals_a), ("loss_hist_sgd", vals_s)):
        if len(vals) == 0:
            return "`%s` is empty. Append loss.item() inside the loop." % name
        if len(vals) != _EXP["n_iter"]:
            return ("`%s` has %d entries but the loop runs N_ITER = %d times."
                    % (name, len(vals), _EXP["n_iter"]))
        if not _num_close(vals[0], _EXP["loss_start"], rtol=1e-6):
            return ("`%s[0]` should be the loss at the starting guesses. Record the loss "
                    "before calling step(), and build a fresh model for each fit." % name)

    for name, val in (("v_hat", v_hat), ("k_hat", k_hat)):
        if isinstance(val, _torch.Tensor):
            return "`%s` is still a tensor. Call .item() on it." % name
        if _as_scalar(val) is None:
            return "`%s` should be a single number." % name

    got = _as_scalar(v_hat)
    if _num_close(got, _np.log(_EXP["v_opt"]), rtol=1e-3):
        return "`v_hat` is still on the log scale. Exponentiate it."
    if not _num_close(got, _EXP["v_opt"], rtol=1e-3):
        return ("`v_hat` is not the fitted volume. Adam at this learning rate reaches the "
                "optimum well inside N_ITER iterations, so check the model and the loop "
                "order: zero_grad, loss, backward, step.")
    if not _num_close(_as_scalar(k_hat), _EXP["k_opt"], rtol=1e-3):
        return "`k_hat` is not the fitted rate constant."

    if not _num_close(vals_a[-1], _EXP["loss_opt"], rtol=1e-4):
        return ("the last entry of `loss_hist_adam` is not the converged loss, although the "
                "parameters look right. Check that the history is the Adam run.")
    if _num_close(vals_s[-1], _EXP["loss_opt"], rtol=1e-2):
        return ("`loss_hist_sgd` converged to the same loss as Adam, which plain SGD does "
                "not manage at this learning rate. Check that the SGD fit used a fresh "
                "model and torch.optim.SGD(..., lr=LR).")
    if vals_s[-1] < 3.0 * _EXP["loss_opt"]:
        return ("`loss_hist_sgd` ends much closer to the optimum than plain SGD gets at "
                "lr = LR. Check that it is a separate fit with its own model and its own "
                "optimiser.")
    return None


def _part_c():
    cl = _need("cl_hat")
    th = _need("t_half_hat")
    auc = _need("auc_inf")
    sd = _need("resid_sd")

    for name, val in (("cl_hat", cl), ("t_half_hat", th), ("auc_inf", auc),
                      ("resid_sd", sd)):
        if isinstance(val, _torch.Tensor):
            return "`%s` is still a tensor. Call .item() on it." % name
        if _as_scalar(val) is None:
            return "`%s` should be a single number." % name

    got = _as_scalar(cl)
    if _num_close(got, _EXP["k_opt"] / _EXP["v_opt"], rtol=1e-3):
        return "`cl_hat` is k divided by V. Clearance is k times V."
    if not _num_close(got, _EXP["cl"], rtol=1e-3):
        return "`cl_hat` is not the clearance. For this model CL = k * V, in L/h."

    got = _as_scalar(th)
    if _num_close(got, 1.0 / _EXP["k_opt"], rtol=1e-3):
        return ("`t_half_hat` is 1 / k, which is the mean residence time for this model, "
                "not the half life. The half life is ln(2) / k.")
    if not _num_close(got, _EXP["t_half"], rtol=1e-3):
        return "`t_half_hat` is not the terminal half life in hours."

    got = _as_scalar(auc)
    if not _num_close(got, _EXP["auc_inf"], rtol=1e-3):
        return ("`auc_inf` is not the AUC from zero to infinity. For an IV bolus it is "
                "DOSE / CL, in mg*h/L.")

    got = _as_scalar(sd)
    if _num_close(got, _EXP["resid_sd"] ** 2, rtol=1e-3):
        return "`resid_sd` is the variance. Take the square root."
    if _num_close(got, _EXP["resid_sd_wrong"], rtol=1e-4):
        return ("`resid_sd` divides the SSE by n. Two parameters were fitted, so the "
                "divisor is n - 2.")
    if not _num_close(got, _EXP["resid_sd"], rtol=1e-3):
        return ("`resid_sd` is not sqrt(SSE / (n - 2)) at the fitted parameters. The SSE "
                "is the one the Adam fit ended on.")
    return None
'''

# Functions this problem introduces. Keys must exist in tools/function_catalog.py.
FUNCTIONS = [
    "torch.nn.Module",
    "torch.nn.Parameter",
    "torch.optim",
    "zero_grad and step",
    "state_dict",
]
