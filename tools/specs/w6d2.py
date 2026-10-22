PID = "W6D2"
WEEK = 6
DAY = 2
SLUG = "w6d2_torch_autograd"
TITLE = "Autograd: derivatives you did not have to derive"
TOPIC = "PyTorch: backward, .grad, gradient accumulation, no_grad, autograd vs finite differences"
EST = "20-30 min"

SCENARIO_MD = '''
## Scenario

Every fit you have run so far has handed the derivative problem to somebody
else. `curve_fit` worked out its own gradients by finite differences and never
told you. That is fine until the model gets big enough that finite differences
become slow, or badly scaled enough that the step size starts to matter.

Today you take the derivative machinery into your own hands, which turns out to
require almost no work, because torch does it for you exactly.

You will differentiate a one compartment fit by hand, differentiate the same
thing with autograd, and confirm the two agree to the last digit. Then you will
meet the one piece of autograd bookkeeping that catches everybody: gradients
add up until you clear them.
'''

CONCEPT_MD = '''
## What autograd actually does

Yesterday you saw that a tensor computed from a tracked tensor carries a
`grad_fn`. That attribute is a pointer into a graph torch has been quietly
building as you computed. Every operation adds a node recording what it did and
what it was applied to.

`loss.backward()` walks that graph from the loss back to every tensor with
`requires_grad=True`, applies the chain rule at each node, and deposits the
result in that tensor's `.grad`. This is called **reverse mode** automatic
differentiation, and the useful fact about it is that one backward pass costs
about the same as one forward pass **no matter how many parameters you have**.
One hundred parameters, one backward pass.

**It is not numerical differentiation.** Finite differences approximate
`df/dx` with `(f(x + h) - f(x - h)) / (2h)`, which needs two extra evaluations
per parameter and forces you to pick `h`: too large and the approximation is
wrong, too small and floating point cancellation eats the answer. Autograd does
not approximate anything. It applies the exact derivative rule for each
operation and multiplies them together, so the only error is ordinary floating
point rounding.

**It is also not symbolic differentiation.** Torch never produces a formula
for `df/dx`. It evaluates the derivative at the point you are standing on,
which is all an optimiser ever needs.

**The three rules of use.**

1. `backward()` wants a **scalar**. A loss is one number, so this is natural.
   Call it on a vector and torch asks you which direction you meant. The fix is
   to reduce first, usually `.sum()` or `.mean()`.

2. **Gradients accumulate.** `.grad` is added to, not overwritten. Call
   `backward()` twice without clearing and you get twice the gradient. This is
   deliberate, because it lets you split a batch across several passes, but in
   a normal fitting loop it is a bug waiting to happen. Clear with
   `k.grad.zero_()`, or in a real training loop `optimizer.zero_grad()`.

3. **The graph is freed after the backward pass.** Torch throws away the
   recorded history once it has been used, because keeping it would leak memory
   in a long loop. Calling `backward()` a second time on the same loss raises
   a RuntimeError telling you to pass `retain_graph=True`. Usually the right
   response is not to retain the graph but to recompute the loss, which is
   what a fitting loop does anyway.

**`torch.no_grad()`** is a context manager that switches the recording off.
Use it when you are evaluating rather than fitting: predictions for a plot,
a validation loss, a finite difference check, or the actual parameter update
inside a hand written loop. Nothing inside it gets a `grad_fn`, which saves
memory and avoids torch complaining that you modified a tracked tensor in
place.
'''

TASK_MD = '''
## Your task

Three parts, worked one at a time.

| Part | What you work out | Variables to fill in |
|---|---|---|
| a | one derivative, twice: by hand and by autograd | `k_a`, `loss_a`, `grad_auto`, `grad_manual` |
| b | a two parameter gradient, against finite differences | `loss_b`, `grad_v`, `grad_k`, `fd_v`, `fd_k` |
| c | the bookkeeping that catches everybody | `grad_once`, `grad_twice`, `grad_after_zero`, `second_backward_error` |

Below this you will find the **Setup** cell, then a **Toolkit** cell that
demonstrates every move on a tiny made up example, then **Part a**, **Part b**
and **Part c**, each with its own explanation and a cell to write in.
'''

DATA_CODE = '''
import numpy as np
import torch

PROBLEM_ID = "W6D2"
TOPIC = "PyTorch: backward, .grad, gradient accumulation, no_grad, autograd vs finite differences"

torch.manual_seed(0)
rng = np.random.default_rng(6202)

# One subject, 100 mg as an IV bolus, plasma sampled over a day.
DOSE = 100.0                 # mg
V_TRUE = 35.0                # L, not known to the fit
K_TRUE = 0.12                # per h, not known to the fit

_t = np.array([0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 18.0, 24.0])
_c = (DOSE / V_TRUE) * np.exp(-K_TRUE * _t)
_c = np.round(_c * rng.lognormal(0.0, 0.07, size=_t.size), 4)

t_obs = torch.tensor(_t, dtype=torch.float64)
c_obs = torch.tensor(_c, dtype=torch.float64)

# For part a only V is treated as settled: this is the C0 an NCA report would
# give you from log-linear back extrapolation to time zero.
C0 = 2.86                    # mg/L

# Starting guesses, deliberately a little off.
K_START = 0.15               # per h
V_START = 30.0               # L

FD_STEP = 1e-6               # step size for the finite difference comparison

print("t_obs:", t_obs)
print("c_obs:", c_obs)
print("")
print("C0 from back extrapolation: %.2f mg/L" % C0)
print("starting guesses: k = %.2f per h, V = %.1f L" % (K_START, V_START))
print("finite difference step:", FD_STEP)
'''

TOOLKIT_MD = '''
## Toolkit for this problem

| What you want to do | How |
|---|---|
| make a parameter | `p = torch.tensor(0.15, dtype=torch.float64, requires_grad=True)` |
| get the derivative | `loss.backward()`, then read `p.grad` |
| that derivative as a float | `p.grad.item()` |
| clear it before the next pass | `p.grad.zero_()` |
| compute without recording | `with torch.no_grad(): ...` |
| values without the history | `t.detach()` |
| reduce a vector to a scalar first | `.sum()`, `.mean()` |
| a central finite difference | `(f(p + h) - f(p - h)) / (2 * h)` |

The thing to notice in the demo below is that `.grad` is **added to**, never
replaced. Every fitting loop you write for the rest of your life clears it at
the top.
'''

TOOLKIT_CODE = '''
import torch

print("--- 1. one parameter, one derivative ---")
# f(x) = x ** 2, so df/dx = 2x, which at x = 3 is 6.
demo_x = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
demo_f = demo_x ** 2
demo_f.backward()
print("f(3) =", demo_f.item(), " autograd df/dx =", demo_x.grad.item(), " by hand = 6.0")

print("")
print("--- 2. several parameters, one backward pass ---")
# g(a, b) = a * exp(-b), so dg/da = exp(-b) and dg/db = -a * exp(-b).
demo_a = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
demo_b = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
demo_g = demo_a * torch.exp(-demo_b)
demo_g.backward()
print("dg/da autograd %.6f  by hand %.6f" % (demo_a.grad.item(), float(torch.exp(-demo_b).item())))
print("dg/db autograd %.6f  by hand %.6f" % (demo_b.grad.item(), float(-(demo_a * torch.exp(-demo_b)).item())))
print("one backward() filled in both, and would fill in a thousand")

print("")
print("--- 3. backward() wants a scalar ---")
demo_v = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
try:
    (demo_v * 2).backward()
except RuntimeError as err:
    print("on a vector it refuses:", str(err)[:60])
demo_s = (demo_v * 2).sum()
demo_s.backward()
print("after .sum() it works, grad =", demo_v.grad)

print("")
print("--- 4. gradients ACCUMULATE ---")
demo_p = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
(demo_p ** 2).backward()
print("after one backward :", demo_p.grad.item())
(demo_p ** 2).backward()
print("after two backwards:", demo_p.grad.item(), " <- not the derivative, the sum of two")
demo_p.grad.zero_()
(demo_p ** 2).backward()
print("after zero_() then one backward:", demo_p.grad.item())

print("")
print("--- 5. the graph is freed once it has been used ---")
demo_q = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
demo_loss = demo_q ** 2
demo_loss.backward()
try:
    demo_loss.backward()
except RuntimeError as err:
    print("a second backward on the SAME loss refuses:")
    print("  ", str(err)[:95])
print("the cure is to recompute the loss, which a fitting loop does anyway")

print("")
print("--- 6. no_grad(), for when you are evaluating rather than fitting ---")
demo_r = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
demo_tracked = demo_r * 3
with torch.no_grad():
    demo_untracked = demo_r * 3
print("outside no_grad: grad_fn is", demo_tracked.grad_fn is not None)
print("inside  no_grad: grad_fn is", demo_untracked.grad_fn is not None)
print("same number either way, but nothing was recorded")
'''

PARTS = [
    dict(
        label="a",
        prompt_md='''
## Part a. The same derivative twice

Treat `C0` as settled and fit only the elimination rate constant. The model is

```
c_pred = C0 * exp(-k * t_obs)
```

and the loss is the sum of squared residuals, with the residual written in the
order `c_pred - c_obs` so the sign of the gradient comes out the way the
formula below says.

Do it in this order.

1. Make `k_a`, a float64 tensor holding `K_START`, with `requires_grad=True`.
2. Compute the predictions, the residuals, and the loss. Record the loss as a
   plain float in `loss_a`.
3. Call `.backward()` on the loss and read `k_a.grad`. Record it as a plain
   float in `grad_auto`.
4. Now derive the same number yourself. By the chain rule,

   ```
   dL/dk = sum over samples of 2 * resid_i * d(c_pred_i)/dk
   ```

   and since `c_pred_i = C0 * exp(-k * t_i)`, the inner derivative is
   `-t_i * c_pred_i`. Put those together, evaluate it, and record the result
   as a plain float in `grad_manual`.

The two should agree to every digit that prints, since they are the same
arithmetic in the same order. That is the point of the exercise: once you have
seen autograd reproduce a derivative you worked out yourself, you can stop
checking and start trusting it, which is what makes the rest of the week
possible.

A detail worth noticing: the derivative you wrote by hand took real thought for
a two line model. For a two compartment model with a depot, absorption lag and
between subject variability, nobody writes it by hand.

Fill in `k_a`, `loss_a`, `grad_auto`, `grad_manual`.

Moves: `requires_grad=True`, `torch.exp`, `.sum()`, `.backward()`, `.grad`,
`.item()`.
''',
        starter='''
# Part a. One derivative, worked out twice.

k_a = None          # YOUR CODE: float64 tensor holding K_START, requires_grad=True

c_pred = None       # YOUR CODE: C0 * exp(-k_a * t_obs)
resid = None        # YOUR CODE: c_pred - c_obs
loss = None         # YOUR CODE: sum of squared residuals

loss_a = None       # YOUR CODE: that loss as a plain float

# YOUR CODE: call .backward() on the loss

grad_auto = None    # YOUR CODE: k_a.grad as a plain float
grad_manual = None  # YOUR CODE: the same derivative from the chain rule formula

print("loss at k = %.2f : %r" % (K_START, loss_a))
print("dL/dk autograd  : %r" % (grad_auto,))
print("dL/dk by hand   : %r" % (grad_manual,))
''',
        solution='''
# Part a. One derivative, worked out twice.

k_a = torch.tensor(K_START, dtype=torch.float64, requires_grad=True)

c_pred = C0 * torch.exp(-k_a * t_obs)
resid = c_pred - c_obs
loss = (resid ** 2).sum()

loss_a = loss.item()

loss.backward()

grad_auto = k_a.grad.item()
grad_manual = (2.0 * resid * (-t_obs * c_pred)).sum().item()

print("loss at k = %.2f : %r" % (K_START, loss_a))
print("dL/dk autograd  : %r" % (grad_auto,))
print("dL/dk by hand   : %r" % (grad_manual,))
''',
    ),
    dict(
        label="b",
        prompt_md='''
## Part b. Two parameters, and the method you are replacing

Now let the volume float too, so the model is

```
c_pred = (DOSE / v) * exp(-k * t_obs)
```

with two unknowns starting at `V_START` and `K_START`. Keep the same loss and
the same residual direction, `c_pred - c_obs`.

First the autograd route. Make two tracked float64 tensors, compute the loss,
record it as a plain float in `loss_b`, call `.backward()` once, and read both
partial derivatives out into `grad_v` and `grad_k`. Notice that one backward
pass filled in both.

Then the finite difference route, which is what `curve_fit` was doing on your
behalf all along. For each parameter separately, evaluate the loss at the
parameter plus `FD_STEP` and at the parameter minus `FD_STEP`, holding the
other parameter fixed, and take

```
(loss_plus - loss_minus) / (2 * FD_STEP)
```

Record those in `fd_v` and `fd_k`. Do the finite difference evaluations with
plain untracked tensors, or inside `with torch.no_grad():`. There is no reason
to record history for a calculation you are not differentiating.

The two routes should agree to about nine or ten significant digits here, and
every bit of the remaining disagreement belongs to the finite difference rather
than to autograd. Note the cost too: autograd needed one forward pass and one
backward pass. The finite difference needed four forward passes for two
parameters, and would need forty for twenty.

Note also that the two partial derivatives are not the same size. The volume
one is roughly sixteen times smaller, because one litre of volume and one per
hour of rate constant are not comparable amounts of change. A factor of sixteen
is harmless. In a model carrying a clearance in L/h, a volume in L and a lag
time in minutes, the same effect reaches factors of thousands, and an optimiser
stepping on raw parameters has to cope with all of it. That is one reason PK
fitting is usually done on log parameters.

Fill in `loss_b`, `grad_v`, `grad_k`, `fd_v`, `fd_k`.

Moves: two tracked tensors, one `.backward()`, `.grad.item()`,
`torch.no_grad()`, a central difference.
''',
        starter='''
# Part b. A two parameter gradient, checked against finite differences.

v_b = None       # YOUR CODE: float64 tensor holding V_START, requires_grad=True
k_b = None       # YOUR CODE: float64 tensor holding K_START, requires_grad=True

# YOUR CODE: predictions, residuals, loss, then one backward pass

loss_b = None    # YOUR CODE: the loss at the starting guesses, as a plain float
grad_v = None    # YOUR CODE: dL/dV from autograd, as a plain float
grad_k = None    # YOUR CODE: dL/dk from autograd, as a plain float

# YOUR CODE: the same two derivatives by central finite differences.
# A small helper that takes v and k and returns the loss as a float will
# save you writing the model four times.

fd_v = None      # YOUR CODE: dL/dV by finite difference
fd_k = None      # YOUR CODE: dL/dk by finite difference

print("loss at the starting guesses: %r" % (loss_b,))
print("dL/dV  autograd %r  finite difference %r" % (grad_v, fd_v))
print("dL/dk  autograd %r  finite difference %r" % (grad_k, fd_k))
''',
        solution='''
# Part b. A two parameter gradient, checked against finite differences.

v_b = torch.tensor(V_START, dtype=torch.float64, requires_grad=True)
k_b = torch.tensor(K_START, dtype=torch.float64, requires_grad=True)

c_pred_b = (DOSE / v_b) * torch.exp(-k_b * t_obs)
resid_b = c_pred_b - c_obs
loss = (resid_b ** 2).sum()
loss.backward()

loss_b = loss.item()
grad_v = v_b.grad.item()
grad_k = k_b.grad.item()


def loss_at(v, k):
    with torch.no_grad():
        vt = torch.tensor(v, dtype=torch.float64)
        kt = torch.tensor(k, dtype=torch.float64)
        pred = (DOSE / vt) * torch.exp(-kt * t_obs)
        return ((pred - c_obs) ** 2).sum().item()


fd_v = (loss_at(V_START + FD_STEP, K_START) - loss_at(V_START - FD_STEP, K_START)) / (2 * FD_STEP)
fd_k = (loss_at(V_START, K_START + FD_STEP) - loss_at(V_START, K_START - FD_STEP)) / (2 * FD_STEP)

print("loss at the starting guesses: %r" % (loss_b,))
print("dL/dV  autograd %r  finite difference %r" % (grad_v, fd_v))
print("dL/dk  autograd %r  finite difference %r" % (grad_k, fd_k))
''',
    ),
    dict(
        label="c",
        prompt_md='''
## Part c. The bookkeeping

Two facts about `backward()` cause most of the confusion in a first week of
torch, and the fastest way to stop being confused by them is to make each one
happen on purpose.

The helper `sse_of(k)` is written for you in the cell below: it is the part a
loss, as a function of `k`.

**Fact one: gradients accumulate.** Make `k_c`, a tracked float64 tensor at
`K_START`. Call `sse_of(k_c).backward()` and record `k_c.grad` as a float in
`grad_once`. Now do it again, exactly the same way, and record the new value of
`k_c.grad` in `grad_twice`. You should find it has doubled, because `.grad` is
added to rather than replaced. Then clear it with `k_c.grad.zero_()`, run one
more backward pass, and record the result in `grad_after_zero`, which should
match `grad_once`.

In a fitting loop this is the difference between converging and marching off to
infinity on the second step, and the symptom, a loss that explodes after one
good iteration, points nowhere near the cause.

**Fact two: the graph is freed when it is used.** Make a separate tracked
tensor `k_d`, compute `loss_d = sse_of(k_d)`, and call `loss_d.backward()`.
Then call `loss_d.backward()` a second time on that same loss object inside a
`try`, catch the `RuntimeError`, and set `second_backward_error` to True if it
was raised and False if it was not.

Note what distinguishes the two cases. In fact one you recomputed the loss each
time, so there was a fresh graph each time and the gradients simply piled up.
In fact two you reused one loss object whose graph had already been consumed.
Recomputing is cheap and is what a loop does naturally, so `retain_graph=True`
is almost never the right answer to this error.

Fill in `grad_once`, `grad_twice`, `grad_after_zero`, `second_backward_error`.

Moves: `.backward()`, `.grad.zero_()`, `try` and `except RuntimeError`.
''',
        starter='''
# Part c. Gradient accumulation, and the freed graph.

def sse_of(k):
    """The part a loss, as a function of k. Written for you."""
    pred = C0 * torch.exp(-k * t_obs)
    return ((pred - c_obs) ** 2).sum()


k_c = None                   # YOUR CODE: float64 tensor at K_START, requires_grad=True

grad_once = None             # YOUR CODE: k_c.grad as a float after ONE backward pass
grad_twice = None            # YOUR CODE: k_c.grad as a float after a SECOND backward pass
grad_after_zero = None       # YOUR CODE: k_c.grad as a float after zero_() then one more

k_d = None                   # YOUR CODE: a fresh tracked tensor at K_START
second_backward_error = None # YOUR CODE: True if a second backward() on the SAME
                             #            loss object raises RuntimeError, else False

print("after one backward   : %r" % (grad_once,))
print("after two backwards  : %r" % (grad_twice,))
print("after zero_() and one: %r" % (grad_after_zero,))
print("second backward on one loss raised RuntimeError:", second_backward_error)
''',
        solution='''
# Part c. Gradient accumulation, and the freed graph.

def sse_of(k):
    """The part a loss, as a function of k. Written for you."""
    pred = C0 * torch.exp(-k * t_obs)
    return ((pred - c_obs) ** 2).sum()


k_c = torch.tensor(K_START, dtype=torch.float64, requires_grad=True)

sse_of(k_c).backward()
grad_once = k_c.grad.item()

sse_of(k_c).backward()
grad_twice = k_c.grad.item()

k_c.grad.zero_()
sse_of(k_c).backward()
grad_after_zero = k_c.grad.item()

k_d = torch.tensor(K_START, dtype=torch.float64, requires_grad=True)
loss_d = sse_of(k_d)
loss_d.backward()
try:
    loss_d.backward()
    second_backward_error = False
except RuntimeError:
    second_backward_error = True

print("after one backward   : %r" % (grad_once,))
print("after two backwards  : %r" % (grad_twice,))
print("after zero_() and one: %r" % (grad_after_zero,))
print("second backward on one loss raised RuntimeError:", second_backward_error)
''',
    ),
]


def build_expected(ns):
    import numpy as np
    t = ns["t_obs"].detach().numpy()
    c = ns["c_obs"].detach().numpy()
    c0 = ns["C0"]
    k0 = ns["K_START"]
    v0 = ns["V_START"]
    dose = ns["DOSE"]

    pred_a = c0 * np.exp(-k0 * t)
    resid_a = pred_a - c
    loss_a = float((resid_a ** 2).sum())
    grad_a = float((2.0 * resid_a * (-t * pred_a)).sum())

    pred_b = (dose / v0) * np.exp(-k0 * t)
    resid_b = pred_b - c
    loss_b = float((resid_b ** 2).sum())
    grad_v = float((2.0 * resid_b * (-pred_b / v0)).sum())
    grad_k = float((2.0 * resid_b * (-t * pred_b)).sum())

    return {
        "n": int(t.size),
        "loss_a": loss_a,
        "grad_a": grad_a,
        "loss_b": loss_b,
        "grad_v": grad_v,
        "grad_k": grad_k,
    }


CHECK_CODE = '''
import torch as _torch


def _part_a():
    k_a = _need("k_a")
    loss_a = _need("loss_a")
    grad_auto = _need("grad_auto")
    grad_manual = _need("grad_manual")

    if not isinstance(k_a, _torch.Tensor):
        return "`k_a` should be a torch tensor, not a %s." % type(k_a).__name__
    if k_a.dtype != _torch.float64:
        return "`k_a` has dtype %s. Ask for float64 explicitly." % k_a.dtype
    if not k_a.requires_grad:
        return ("`k_a` was made without requires_grad=True, so torch recorded nothing and "
                "there is no derivative to read.")
    if not _num_close(float(k_a.detach().item()), K_START, rtol=1e-12):
        return "`k_a` should hold K_START, the starting guess."

    got = _as_scalar(loss_a)
    if got is None:
        return "`loss_a` should be a single number. Call .item() on the loss tensor."
    if isinstance(loss_a, _torch.Tensor):
        return "`loss_a` is still a tensor. Call .item() on it."
    if _num_close(got, _EXP["loss_a"] / _EXP["n"], rtol=1e-6):
        return ("`loss_a` is the MEAN squared residual. The task asks for the sum, so use "
                ".sum() rather than .mean().")
    if not _num_close(got, _EXP["loss_a"], rtol=1e-6):
        return ("`loss_a` is not the sum of squared residuals at k = K_START. Check that "
                "the model is C0 * exp(-k * t_obs).")

    if isinstance(grad_auto, _torch.Tensor):
        return "`grad_auto` is still a tensor. Call .item() on k_a.grad."
    got = _as_scalar(grad_auto)
    if got is None:
        return ("`grad_auto` should be a single number. If k_a.grad is None, .backward() "
                "was never called on the loss.")
    if _num_close(got, -_EXP["grad_a"], rtol=1e-6):
        return ("`grad_auto` has the right size but the wrong sign, so the residual was "
                "written the other way round. The task asks for c_pred - c_obs.")
    if not _num_close(got, _EXP["grad_a"], rtol=1e-6):
        return ("`grad_auto` is not dL/dk at the starting guess. It should come from "
                "k_a.grad after calling .backward() on the summed squared residuals.")

    if isinstance(grad_manual, _torch.Tensor):
        return "`grad_manual` is still a tensor. Call .item() on it."
    got = _as_scalar(grad_manual)
    if got is None:
        return "`grad_manual` should be a single number."
    if _num_close(got, _EXP["grad_a"] / 2.0, rtol=1e-6):
        return ("`grad_manual` is half the right value, so the factor of 2 from "
                "differentiating the square is missing.")
    if _num_close(got, -_EXP["grad_a"], rtol=1e-6):
        return ("`grad_manual` has the right size but the wrong sign. The inner derivative "
                "of C0 * exp(-k * t) with respect to k is -t * c_pred, negative.")
    if not _num_close(got, _EXP["grad_a"], rtol=1e-6):
        return ("`grad_manual` does not match the chain rule result. It is the sum over "
                "samples of 2 * resid * (-t * c_pred).")
    return None


def _part_b():
    loss_b = _need("loss_b")
    grad_v = _need("grad_v")
    grad_k = _need("grad_k")
    fd_v = _need("fd_v")
    fd_k = _need("fd_k")

    for name, val in (("loss_b", loss_b), ("grad_v", grad_v), ("grad_k", grad_k),
                      ("fd_v", fd_v), ("fd_k", fd_k)):
        if isinstance(val, _torch.Tensor):
            return "`%s` is still a tensor. Call .item() on it." % name
        if _as_scalar(val) is None:
            return "`%s` should be a single number." % name

    got = _as_scalar(loss_b)
    if _num_close(got, _EXP["loss_a"], rtol=1e-6):
        return ("`loss_b` is the part a loss. Part b uses DOSE / v for the intercept, with "
                "v starting at V_START, not the fixed C0.")
    if _num_close(got, _EXP["loss_b"] / _EXP["n"], rtol=1e-6):
        return "`loss_b` is the mean, not the sum, of the squared residuals."
    if not _num_close(got, _EXP["loss_b"], rtol=1e-6):
        return ("`loss_b` is not the sum of squared residuals at V_START and K_START. "
                "The model is (DOSE / v) * exp(-k * t_obs).")

    got = _as_scalar(grad_v)
    if _num_close(got, -_EXP["grad_v"], rtol=1e-5):
        return ("`grad_v` has the right size but the wrong sign, so the residual is the "
                "other way round. Use c_pred - c_obs.")
    if not _num_close(got, _EXP["grad_v"], rtol=1e-5):
        return ("`grad_v` is not dL/dV. It should be v_b.grad after one .backward() call "
                "on the loss.")

    got = _as_scalar(grad_k)
    if not _num_close(got, _EXP["grad_k"], rtol=1e-5) and _num_close(got, _EXP["grad_v"], rtol=1e-5):
        return "`grad_k` holds the volume gradient. The two have been swapped."
    if _num_close(got, -_EXP["grad_k"], rtol=1e-5):
        return "`grad_k` has the right size but the wrong sign. Use c_pred - c_obs."
    if not _num_close(got, _EXP["grad_k"], rtol=1e-5):
        return ("`grad_k` is not dL/dk. Note that one .backward() fills in both parameters, "
                "so you should not need a second pass.")

    got = _as_scalar(fd_v)
    if not _num_close(got, _EXP["grad_v"], rtol=1e-4):
        if _num_close(got, _EXP["grad_v"] * 2.0, rtol=1e-3):
            return ("`fd_v` is twice the gradient, so the division is by FD_STEP rather "
                    "than 2 * FD_STEP.")
        return ("`fd_v` does not match the volume gradient. A central difference is "
                "(loss at v + h minus loss at v - h) / (2 * h), with k held at K_START.")

    got = _as_scalar(fd_k)
    if not _num_close(got, _EXP["grad_k"], rtol=1e-4):
        if _num_close(got, _EXP["grad_k"] * 2.0, rtol=1e-3):
            return ("`fd_k` is twice the gradient, so the division is by FD_STEP rather "
                    "than 2 * FD_STEP.")
        return ("`fd_k` does not match the rate constant gradient. Perturb k only, holding "
                "v at V_START.")
    return None


def _part_c():
    grad_once = _need("grad_once")
    grad_twice = _need("grad_twice")
    grad_after_zero = _need("grad_after_zero")
    second = _need("second_backward_error")

    for name, val in (("grad_once", grad_once), ("grad_twice", grad_twice),
                      ("grad_after_zero", grad_after_zero)):
        if isinstance(val, _torch.Tensor):
            return ("`%s` is still a tensor. Call .item() on k_c.grad, and do it at the "
                    "moment you read it: a tensor you saved earlier is a view that will "
                    "change under you." % name)
        if _as_scalar(val) is None:
            return "`%s` should be a single number." % name

    got = _as_scalar(grad_once)
    if not _num_close(got, _EXP["grad_a"], rtol=1e-6):
        return ("`grad_once` is not dL/dk at K_START. It is the same derivative as part a, "
                "read straight after the first backward pass.")

    got = _as_scalar(grad_twice)
    if _num_close(got, _EXP["grad_a"], rtol=1e-6):
        return ("`grad_twice` equals `grad_once`, so the gradient did not accumulate. "
                "Either .grad was cleared in between, or grad_once was read again instead "
                "of being read after a second backward pass on a freshly computed loss.")
    if not _num_close(got, 2.0 * _EXP["grad_a"], rtol=1e-6):
        return ("`grad_twice` should be exactly twice `grad_once`. Recompute the loss and "
                "call .backward() again without clearing .grad.")

    got = _as_scalar(grad_after_zero)
    if _num_close(got, 0.0, atol=1e-12):
        return ("`grad_after_zero` is zero, so it was read after zero_() but before the "
                "next backward pass. Clear, then run one more pass, then read.")
    if _num_close(got, 3.0 * _EXP["grad_a"], rtol=1e-6):
        return ("`grad_after_zero` is three times the gradient, so .grad was never "
                "cleared. k_c.grad.zero_() does it in place.")
    if not _num_close(got, _EXP["grad_a"], rtol=1e-6):
        return ("`grad_after_zero` should match `grad_once`, which is the whole point of "
                "clearing.")

    if not isinstance(second, bool):
        return "`second_backward_error` should be True or False."
    if second is not True:
        return ("`second_backward_error` should be True. Calling .backward() twice on the "
                "same loss object raises RuntimeError, because the graph is freed once it "
                "has been used. Note that part c fact one recomputed the loss each time, "
                "which is why it did not hit this.")
    return None
'''

# Functions this problem introduces. Keys must exist in tools/function_catalog.py.
FUNCTIONS = [
    "backward",
    "tensor .grad",
    "gradient accumulation",
    "torch.no_grad",
    "finite differences",
]
