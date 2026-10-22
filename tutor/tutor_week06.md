# Week 6, PyTorch

The full text of every problem this week, the data it generates, the
toolkit the student is given, what each part asks for, and what every
possible check failure means. No solutions in this file.

---

# W6D1. Tensors: numpy with two things added

- topic: PyTorch: tensors, dtypes, shapes, numpy interop, the autograd flag
- notebook: problems/week06/w6d1_torch_tensors.ipynb
- solution: solutions/week06/w6d1_torch_tensors_solution.ipynb
- expected time: 20-30 min

## Scenario

Before fitting anything with PyTorch you need to be comfortable with its one
data structure, the tensor. The good news is that you already are: a tensor is
a numpy array with two additions.

It can live on a GPU, which will not matter for anything in this curriculum,
and it can **remember how it was computed**, which matters enormously and is
the subject of tomorrow.

Today is about the parts that are not new, so that tomorrow the only new thing
is the genuinely new thing. You will move a PK dataset between numpy and
torch, hit the two dtype traps that catch everyone, and see exactly what the
autograd flag changes.

## What carries over from numpy, and the two things that do not

**Almost everything carries over.** Indexing, slicing, boolean masks,
broadcasting, `.shape`, reductions with `dim=` instead of `axis=`. If you can
write it in numpy you can usually write it in torch by changing the import and
the word `axis` to `dim`.

**Trap one: the default dtype is float32.** numpy defaults to float64;
PyTorch defaults to float32, because that is what neural networks want and it
is twice as fast on a GPU. For PK work float32 gives you about seven
significant digits, which is usually fine but will bite you when an optimiser
needs to distinguish two nearly identical losses.

Worse, `torch.tensor` **infers** its dtype from what you hand it. Pass Python
integers and you get an integer tensor, on which division behaves differently
and gradients are impossible. The habit that avoids all of it is to say what
you want: `torch.tensor(x, dtype=torch.float64)`.

**Trap two: `.numpy()` shares memory.** `tensor.numpy()` does not copy, it
hands you a numpy view of the same buffer. Change one and the other changes
too. That is efficient and occasionally catastrophic. `torch.from_numpy` has
the same property in the other direction. When you want a snapshot, say
`tensor.numpy().copy()`.

A related gotcha: you cannot call `.numpy()` on a tensor that is tracking
gradients. Torch refuses, because handing out a numpy view of something whose
history matters would let you corrupt it silently. The fix is
`.detach().numpy()`, which says "I want the values, not the history".

**The autograd flag.** `requires_grad=True` tells torch to record every
operation involving this tensor, building a graph it can later differentiate.
Tensors made from your data do not need it. Tensors holding **parameters you
want to fit** do.

You can see the recording happening: a result computed from a tracked tensor
has a `grad_fn`, which is the node in the graph saying what produced it.
A result computed from ordinary tensors has `grad_fn` of `None`. That one
attribute is the whole difference between a tensor and an array, and tomorrow
it is what lets you fit a model without writing a single derivative.

**On reproducibility.** `torch.manual_seed(0)` fixes torch's own random number
generator. It is separate from numpy's, so seeding one does not seed the
other.

## Your task

Three parts, worked one at a time.

| Part | What you work out | Variables to fill in |
|---|---|---|
| a | the data as tensors, with the dtype you chose | `t_tensor`, `c_tensor`, `dtype_default`, `dtype_asked` |
| b | the same numpy moves, in torch | `cmax`, `tmax`, `log_c`, `mask`, `c_measurable` |
| c | what the autograd flag actually changes | `plain`, `tracked`, `has_grad_fn_plain`, `has_grad_fn_tracked`, `shares_memory` |

Below this you will find the **Setup** cell, then a **Toolkit** cell that
demonstrates every move on a tiny made up example, then **Part a**, **Part b**
and **Part c**, each with its own explanation and a cell to write in.

## The setup cell the student runs

This is given to them, already working. The data is synthetic and
seeded, so their numbers are always these numbers.

```python
import numpy as np
import torch

PROBLEM_ID = "W6D1"
TOPIC = "PyTorch: tensors, dtypes, shapes, numpy interop, the autograd flag"

torch.manual_seed(0)
rng = np.random.default_rng(6101)

# A familiar single subject profile, as plain numpy to start with.
time_np = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 24.0, 36.0, 48.0])
_true = 2.8 * (np.exp(-0.095 * time_np) - np.exp(-1.2 * time_np))
conc_np = np.round(_true * rng.lognormal(0.0, 0.08, size=time_np.size), 4)

LLOQ = 0.25

print("torch version:", torch.__version__)
print("")
print("time_np :", time_np)
print("conc_np :", conc_np)
print("dtypes  :", time_np.dtype, conc_np.dtype)
print("LLOQ    :", LLOQ, "mg/L")
```

## Toolkit for this problem

| What you want to do | How |
|---|---|
| numpy array to tensor | `torch.tensor(a, dtype=torch.float64)` |
| tensor back to numpy | `t.numpy()`, or `t.detach().numpy()` if it tracks grads |
| the dtype | `t.dtype` |
| shape | `t.shape` |
| reductions | `t.max()`, `t.mean()`, and `dim=` where numpy says `axis=` |
| position of the max | `torch.argmax(t)` |
| a single python number out | `t.item()` |
| natural log | `torch.log(t)` |
| track gradients | `torch.tensor(..., requires_grad=True)` |
| was this computed from a tracked tensor | `t.grad_fn is not None` |
| do two share memory | `np.shares_memory(a, b)` |

`.item()` turns a one element tensor into an ordinary Python float, which is
what you want whenever a number is leaving torch for printing or comparison.

The toolkit cell demonstrates each of those on a tiny made up example.
It is in the notebook and none of it is the answer.

## Part a. Move the data into torch, deliberately

Make two tensors from the numpy arrays, both with `dtype=torch.float64`.
Saying the dtype explicitly is the habit worth building, because the default
is float32 and the inferred dtype depends on what you passed.

Then demonstrate the trap to yourself. Make a tensor from the Python list
`[1, 2, 3]` with no dtype argument and record what dtype you got, and make
another from the same list **with** `dtype=torch.float64` and record that.

The first will be an integer tensor. Integer tensors cannot carry gradients
at all, so if parameters accidentally arrive as integers, nothing in the rest
of this week works and the error message you get is some distance from the
cause.

Fill in `t_tensor`, `c_tensor`, both float64, and `dtype_default`,
`dtype_asked`, the two dtypes you observed.

Moves: `torch.tensor(a, dtype=torch.float64)`, `.dtype`.

**Variables this part is graded on:** `t_tensor`, `c_tensor`, `dtype_default`, `dtype_asked`

The starter cell they see:

```python
# Part a. Into torch, with the dtype stated.

t_tensor = None       # YOUR CODE: time_np as a float64 tensor
c_tensor = None       # YOUR CODE: conc_np as a float64 tensor
dtype_default = None  # YOUR CODE: the dtype of torch.tensor([1, 2, 3])
dtype_asked = None    # YOUR CODE: the dtype of torch.tensor([1, 2, 3], dtype=torch.float64)

print("t_tensor:", t_tensor)
print("c_tensor:", c_tensor)
print("dtype from python ints, inferred:", dtype_default)
print("dtype when asked for explicitly  :", dtype_asked)
```

**Every failure message part a can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `t_tensor` should be a torch Tensor, not a %s.
- `c_tensor` should be a torch Tensor.
- `t_tensor` has dtype %s. The task asks for float64, stated explicitly, because torch defaults to float32.
- `c_tensor` has dtype %s, but float64 was asked for.
- `t_tensor` should hold the %d times.
- `t_tensor` does not hold the values of time_np.
- `c_tensor` does not hold the values of conc_np.
- `dtype_default` came out float64, so a dtype was passed. The point is to see what torch.tensor([1, 2, 3]) infers on its own.
- `dtype_default` should be the dtype torch infers from a list of Python integers, which is an integer type, but it is %s.
- `dtype_asked` should be torch.float64, from passing dtype explicitly.

## Part b. The week 1 moves, in torch

Everything here you did in numpy on day one of week one. The point is to see
how little changes.

- `cmax`, the highest concentration, as a plain Python float via `.item()`,
- `tmax`, the time at which it occurred, also a float, found with
  `torch.argmax` exactly as you used `np.argmax`,
- `log_c`, the natural log of the concentrations, as a tensor,
- `mask`, a boolean tensor, True where the concentration is at or above
  `LLOQ`,
- `c_measurable`, the concentrations that pass, selected with that mask.

The sampling here runs out to 48 h, by which point the profile has dropped
under the assay limit, so the mask has real work to do: two of the twelve
samples are below the LLOQ and should not be carried into anything downstream.

Use `.item()` for the two scalars. A one element tensor prints as
`tensor(2.0312, dtype=torch.float64)`, which is correct but unreadable in a
report, and comparisons against it behave in ways that surprise people.
`.item()` gets you an ordinary float.

Fill in `cmax`, `tmax`, `log_c`, `mask`, `c_measurable`.

Moves: `.max()`, `torch.argmax`, `.item()`, `torch.log`, a comparison,
boolean indexing.

**Variables this part is graded on:** `cmax`, `tmax`, `log_c`, `mask`, `c_measurable`

The starter cell they see:

```python
# Part b. Familiar numpy moves, in torch.

cmax = None          # YOUR CODE: highest concentration, as a plain float
tmax = None          # YOUR CODE: time of that peak, as a plain float
log_c = None         # YOUR CODE: natural log of c_tensor, as a tensor
mask = None          # YOUR CODE: boolean tensor, True where conc >= LLOQ
c_measurable = None  # YOUR CODE: the concentrations passing that mask

print("cmax: %s mg/L at tmax %s h" % (cmax, tmax))
print("log_c:", log_c)
print("mask :", mask)
print("kept :", c_measurable)
```

**Every failure message part b can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `cmax` is still a tensor. Call .item() on it to get an ordinary Python float.
- `cmax` is not the highest concentration.
- `tmax` is still a tensor. Call .item() on it.
- `tmax` should be a single number.
- `tmax` is the index of the peak, not the time. Use it to index t_tensor.
- `tmax` is not the time at which the peak occurred.
- `log_c` should be a tensor, from torch.log.
- `log_c` is not the natural log of c_tensor.
- `mask` should be a tensor.
- `mask` has dtype %s. A comparison like c_tensor >= LLOQ gives a boolean tensor directly.
- `mask` does not flag the measurable samples.
- `c_measurable` should be a tensor.
- `c_measurable` still has every sample. Index c_tensor with your boolean mask.
- `c_measurable` holds %d values but %d samples are at or above the LLOQ.
- `c_measurable` does not hold the measurable concentrations.
- `mask` is inverted. True should mean the sample IS measurable, so the comparison is c_tensor >= LLOQ.

## Part c. What the autograd flag changes, and the shared memory trap

Make two tensors holding the same single value `2.0`, both float64. One
plain, one with `requires_grad=True`. Compute `x * 3 + 1` from each.

Then record whether each **result** has a `grad_fn`, as a True or False.
A `grad_fn` is the recorded history of how the value was produced, and it
appears only when something in the computation was being tracked. That single
attribute is the entire difference between a tensor and a numpy array, and
tomorrow it is what gets differentiated.

Finally, the memory trap. Take `c_tensor`, call `.numpy()` on it, and check
with `np.shares_memory` whether the result shares storage with another
`.numpy()` view of the same tensor. It does, because `.numpy()` returns a
view rather than a copy.

That is worth knowing before it costs you an afternoon: a numpy array you
pulled out of a tensor is not a snapshot, and modifying either one changes
both. When you want a genuine copy, say `.numpy().copy()`.

Fill in `plain`, `tracked`, `has_grad_fn_plain`, `has_grad_fn_tracked`,
`shares_memory`.

Moves: `torch.tensor(..., requires_grad=True)`, arithmetic, `.grad_fn is not
None`, `np.shares_memory`.

**Variables this part is graded on:** `plain`, `tracked`, `result_plain`, `result_tracked`, `has_grad_fn_plain`, `has_grad_fn_tracked`, `shares_memory`

The starter cell they see:

```python
# Part c. The autograd flag, and shared memory.

plain = None                  # YOUR CODE: float64 tensor holding 2.0, no tracking
tracked = None                # YOUR CODE: the same, but requires_grad=True

result_plain = None           # YOUR CODE: plain * 3 + 1
result_tracked = None         # YOUR CODE: tracked * 3 + 1

has_grad_fn_plain = None      # YOUR CODE: True/False, does result_plain have a grad_fn
has_grad_fn_tracked = None    # YOUR CODE: True/False, does result_tracked have one
shares_memory = None          # YOUR CODE: True/False, do two .numpy() views of
                              #            c_tensor share storage

print("result from plain tensor  : grad_fn present?", has_grad_fn_plain)
print("result from tracked tensor: grad_fn present?", has_grad_fn_tracked)
print("two .numpy() views of c_tensor share memory?", shares_memory)
```

**Every failure message part c can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `plain` and `tracked` should both be torch tensors.
- `plain` should NOT have requires_grad set. That is the point of the contrast.
- `tracked` should be created with requires_grad=True.
- Both tensors should be float64.
- `has_grad_fn_plain` should be False. Nothing in that computation was being tracked, so there is no history to record.
- `has_grad_fn_tracked` should be True. A result computed from a tracked tensor carries a grad_fn, which is the recorded history.
- `shares_memory` should be True. tensor.numpy() returns a VIEW of the same buffer, not a copy, which is why .numpy().copy() exists.

**Functions this problem introduces:** torch.tensor, tensor dtypes, .item(), tensor to numpy, requires_grad

---

# W6D2. Autograd: derivatives you did not have to derive

- topic: PyTorch: backward, .grad, gradient accumulation, no_grad, autograd vs finite differences
- notebook: problems/week06/w6d2_torch_autograd.ipynb
- solution: solutions/week06/w6d2_torch_autograd_solution.ipynb
- expected time: 20-30 min

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

## The setup cell the student runs

This is given to them, already working. The data is synthetic and
seeded, so their numbers are always these numbers.

```python
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
```

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

The toolkit cell demonstrates each of those on a tiny made up example.
It is in the notebook and none of it is the answer.

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

**Variables this part is graded on:** `k_a`, `c_pred`, `resid`, `loss`, `loss_a`, `grad_auto`, `grad_manual`

The starter cell they see:

```python
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
```

**Every failure message part a can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `k_a` should be a torch tensor, not a %s.
- `k_a` has dtype %s. Ask for float64 explicitly.
- `k_a` was made without requires_grad=True, so torch recorded nothing and there is no derivative to read.
- `k_a` should hold K_START, the starting guess.
- `loss_a` should be a single number. Call .item() on the loss tensor.
- `loss_a` is still a tensor. Call .item() on it.
- `loss_a` is the MEAN squared residual. The task asks for the sum, so use .sum() rather than .mean().
- `loss_a` is not the sum of squared residuals at k = K_START. Check that the model is C0 * exp(-k * t_obs).
- `grad_auto` is still a tensor. Call .item() on k_a.grad.
- `grad_auto` should be a single number. If k_a.grad is None, .backward() was never called on the loss.
- `grad_auto` has the right size but the wrong sign, so the residual was written the other way round. The task asks for c_pred - c_obs.
- `grad_auto` is not dL/dk at the starting guess. It should come from k_a.grad after calling .backward() on the summed squared residuals.
- `grad_manual` is still a tensor. Call .item() on it.
- `grad_manual` should be a single number.
- `grad_manual` is half the right value, so the factor of 2 from differentiating the square is missing.
- `grad_manual` has the right size but the wrong sign. The inner derivative of C0 * exp(-k * t) with respect to k is -t * c_pred, negative.
- `grad_manual` does not match the chain rule result. It is the sum over samples of 2 * resid * (-t * c_pred).

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

**Variables this part is graded on:** `v_b`, `k_b`, `loss_b`, `grad_v`, `grad_k`, `fd_v`, `fd_k`

The starter cell they see:

```python
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
```

**Every failure message part b can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `loss_b` is the part a loss. Part b uses DOSE / v for the intercept, with v starting at V_START, not the fixed C0.
- `loss_b` is the mean, not the sum, of the squared residuals.
- `loss_b` is not the sum of squared residuals at V_START and K_START. The model is (DOSE / v) * exp(-k * t_obs).
- `grad_v` has the right size but the wrong sign, so the residual is the other way round. Use c_pred - c_obs.
- `grad_v` is not dL/dV. It should be v_b.grad after one .backward() call on the loss.
- `grad_k` holds the volume gradient. The two have been swapped.
- `grad_k` has the right size but the wrong sign. Use c_pred - c_obs.
- `grad_k` is not dL/dk. Note that one .backward() fills in both parameters, so you should not need a second pass.
- `fd_v` does not match the volume gradient. A central difference is (loss at v + h minus loss at v - h) / (2 * h), with k held at K_START.
- `fd_k` does not match the rate constant gradient. Perturb k only, holding v at V_START.
- `%s` is still a tensor. Call .item() on it.
- `%s` should be a single number.
- `fd_v` is twice the gradient, so the division is by FD_STEP rather than 2 * FD_STEP.
- `fd_k` is twice the gradient, so the division is by FD_STEP rather than 2 * FD_STEP.

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

**Variables this part is graded on:** `k_c`, `grad_once`, `grad_twice`, `grad_after_zero`, `k_d`, `second_backward_error`

The starter cell they see:

```python
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
```

**Every failure message part c can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `grad_once` is not dL/dk at K_START. It is the same derivative as part a, read straight after the first backward pass.
- `grad_twice` equals `grad_once`, so the gradient did not accumulate. Either .grad was cleared in between, or grad_once was read again instead of being read after a second backward pass on a freshly computed loss.
- `grad_twice` should be exactly twice `grad_once`. Recompute the loss and call .backward() again without clearing .grad.
- `grad_after_zero` is zero, so it was read after zero_() but before the next backward pass. Clear, then run one more pass, then read.
- `grad_after_zero` is three times the gradient, so .grad was never cleared. k_c.grad.zero_() does it in place.
- `grad_after_zero` should match `grad_once`, which is the whole point of clearing.
- `second_backward_error` should be True or False.
- `second_backward_error` should be True. Calling .backward() twice on the same loss object raises RuntimeError, because the graph is freed once it has been used. Note that part c fact one recomputed the loss each time, which is why it did not hit this.
- `%s` is still a tensor. Call .item() on k_c.grad, and do it at the moment you read it: a tensor you saved earlier is a view that will change under you.
- `%s` should be a single number.

**Functions this problem introduces:** backward, tensor .grad, gradient accumulation, torch.no_grad, finite differences

---

# W6D3. Gradient descent, written out by hand

- topic: PyTorch: the fitting loop, learning rate, log parameters, divergence
- notebook: problems/week06/w6d3_torch_gradient_descent.ipynb
- solution: solutions/week06/w6d3_torch_gradient_descent_solution.ipynb
- expected time: 20-30 min

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

## The setup cell the student runs

This is given to them, already working. The data is synthetic and
seeded, so their numbers are always these numbers.

```python
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
```

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

The toolkit cell demonstrates each of those on a tiny made up example.
It is in the notebook and none of it is the answer.

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

**Variables this part is graded on:** `v1`, `k1`, `loss_before`, `loss_after_small`, `loss_after_big`, `inplace_error`

The starter cell they see:

```python
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
```

**Every failure message part a can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `loss_before` is not the summed squared residuals at V_START and K_START. The model is (DOSE / v) * exp(-k * t_obs).
- `loss_after_small` equals the loss before the step, so no step was taken. The update has to be inside a torch.no_grad() block to work at all.
- `loss_after_small` is larger than `loss_before`, so the step went uphill. Gradient descent SUBTRACTS lr * grad.
- `loss_after_small` is not the loss after one step of size LR_SMALL from the starting guesses. Check that both parameters were stepped.
- `loss_after_big` matches the small step, so LR_BIG was not used, or the second step was taken from the already updated parameters rather than from fresh tensors at V_START and K_START.
- `loss_after_big` is better than the starting loss, which it should not be. Start again from fresh tensors at V_START and K_START and take ONE step of size LR_BIG.
- `loss_after_big` is not the loss after one step of size LR_BIG.
- `inplace_error` should be True or False.
- `inplace_error` should be True. An in place update of a tracked leaf tensor outside no_grad() raises RuntimeError, which is why every loop wraps the step.
- `%s` is still a tensor. Call .item() on it.
- `%s` should be a single number.

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

**Variables this part is graded on:** `log_v`, `log_k`, `loss_hist`, `v_hat`, `k_hat`, `v_scipy`, `k_scipy`

The starter cell they see:

```python
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
```

**Every failure message part b can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `loss_hist` is empty. Append loss.item() inside the loop.
- `loss_hist` has %d entries but the loop runs N_ITER = %d times.
- `loss_hist[0]` should be the loss at the starting guesses, the same number as part a's `loss_before`. Append the loss BEFORE the update.
- `loss_hist` goes up at iteration %d, so the descent is not descending. The usual cause is not clearing .grad at the top of the loop, which makes every step larger than the last.
- the last entry of `loss_hist` is not the converged loss. At this learning rate N_ITER iterations is far more than enough, so check the model and that exp() is applied to both log parameters.
- `v_hat` is still on the log scale. Exponentiate it.
- `v_hat` is not the fitted volume. It should be exp(log_v) after the loop, on the natural scale in litres.
- `k_hat` is still on the log scale. Exponentiate it.
- `k_hat` is not the fitted rate constant.
- `v_scipy` is not the curve_fit estimate of V. Fit the same model to t_obs and c_obs as numpy arrays.
- `k_scipy` is not the curve_fit estimate of k.
- `loss_hist` should be a list of numbers, one loss per iteration.
- `loss_hist` looks reversed: its first entry is the converged loss. Append as you go, so entry 0 is the starting loss.
- `%s` is still a tensor. Call .item() on it.
- `%s` should be a single number.
- `v_scipy` is close but not converged. Pass p0=[V_START, K_START] and let curve_fit run to its own tolerance.

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

**Variables this part is graded on:** `loss_big`, `v_big`, `zero_model_loss`, `loss_hist_huge`, `first_nan_index`

The starter cell they see:

```python
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
```

**Every failure message part c can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `zero_model_loss` should be the loss of predicting zero at every time point, which is the sum of the squared observations.
- `v_big` is the sensible fitted volume, so the run used the part b learning rate. Part c uses LR_TOO_BIG.
- `v_big` should be enormous, millions of litres. At LR_TOO_BIG the volume runs away. If yours is reasonable, check that LR_TOO_BIG was used for all N_TOO_BIG iterations and that both gradients are cleared each time.
- `loss_big` is the converged loss from part b. Part c uses LR_TOO_BIG, which does not converge to a fit.
- `loss_big` should have settled on essentially the zero model loss, because the runaway volume makes every prediction about zero. Record the loss of the last iteration of the LR_TOO_BIG run.
- `loss_hist_huge` has %d entries but the run is N_HUGE = %d iterations. Append every loss, including the ones that are not finite.
- `first_nan_index` is 0, which is what np.argmax gives when nothing is True. The first loss is finite, so either no NaN appeared or the list is not the loss trace. At LR_HUGE the loss should blow up within a few iterations.
- `first_nan_index` of %d is not where the NaN turns up. At LR_HUGE the loss should first come out as not a number within the first handful of iterations. Check that both gradients are cleared each iteration and that every loss is appended, non-finite ones included.
- `%s` is still a tensor. Call .item() on it.
- `%s` should be a single number.
- `first_nan_index` should be an integer index.

**Functions this problem introduces:** gradient descent loop, learning rate, log parameters, np.isnan, np.argmax on booleans

---

# W6D4. nn.Module and an optimiser: the loop everyone actually writes

- topic: PyTorch: nn.Module, nn.Parameter, torch.optim, Adam against plain SGD
- notebook: problems/week06/w6d4_torch_module_optimizer.ipynb
- solution: solutions/week06/w6d4_torch_module_optimizer_solution.ipynb
- expected time: 20-30 min

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

## The setup cell the student runs

This is given to them, already working. The data is synthetic and
seeded, so their numbers are always these numbers.

```python
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
```

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

The toolkit cell demonstrates each of those on a tiny made up example.
It is in the notebook and none of it is the answer.

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

**Variables this part is graded on:** `v`, `k`, `n_params`, `param_names`, `all_require_grad`, `loss_start`

The starter cell they see:

```python
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
```

**Every failure message part a can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `OneCompartmentIV` is not defined. The class skeleton is in the part a cell.
- `OneCompartmentIV` should subclass torch.nn.Module.
- the model reports no parameters. Each one has to be a torch.nn.Parameter assigned to an attribute of self, which is what registers it.
- the parameters should be float64. Pass dtype=torch.float64 to torch.tensor.
- `n_params` should be a single integer.
- `n_params` should be 2, one for each fitted parameter, and it should come from len(list(model0.parameters())).
- `param_names` should be ['log_k', 'log_v'], sorted alphabetically. The task asks for those two names exactly, and it got %r.
- `all_require_grad` should be True or False.
- `all_require_grad` should be True. nn.Parameter sets requires_grad for you, so there is nothing to set by hand.
- `loss_start` is still a tensor. Call .item() on it.
- `loss_start` should be a single number.
- `loss_start` is the mean, not the sum, of the squared residuals.
- `loss_start` is the converged loss, so the model was fitted first. It should be the loss of a fresh model at the starting guesses.
- `loss_start` is not the summed squared residuals of the fresh model. Check that forward returns (DOSE / v) * exp(-k * t) with v and k exponentiated.
- `forward` returns None. It should return the predicted concentrations.
- `forward` should return a tensor.
- `forward` returned shape %s for %d input times. It should return one prediction per time.
- `OneCompartmentIV(30.0, 0.15)` raised %s: %s
- `param_names` should be a list of strings.
- calling the model raised %s: %s

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

**Variables this part is graded on:** `opt`, `loss_hist_adam`, `v_hat`, `k_hat`, `loss_hist_sgd`

The starter cell they see:

```python
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
```

**Every failure message part b can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `v_hat` is still on the log scale. Exponentiate it.
- `v_hat` is not the fitted volume. Adam at this learning rate reaches the optimum well inside N_ITER iterations, so check the model and the loop order: zero_grad, loss, backward, step.
- `k_hat` is not the fitted rate constant.
- the last entry of `loss_hist_adam` is not the converged loss, although the parameters look right. Check that the history is the Adam run.
- `loss_hist_sgd` converged to the same loss as Adam, which plain SGD does not manage at this learning rate. Check that the SGD fit used a fresh model and torch.optim.SGD(..., lr=LR).
- `loss_hist_sgd` ends much closer to the optimum than plain SGD gets at lr = LR. Check that it is a separate fit with its own model and its own optimiser.
- `loss_hist_adam` and `loss_hist_sgd` should be lists of numbers.
- `%s` is empty. Append loss.item() inside the loop.
- `%s` has %d entries but the loop runs N_ITER = %d times.
- `%s[0]` should be the loss at the starting guesses. Record the loss before calling step(), and build a fresh model for each fit.
- `%s` is still a tensor. Call .item() on it.
- `%s` should be a single number.

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

**Variables this part is graded on:** `cl_hat`, `t_half_hat`, `auc_inf`, `resid_sd`

The starter cell they see:

```python
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
```

**Every failure message part c can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `cl_hat` is k divided by V. Clearance is k times V.
- `cl_hat` is not the clearance. For this model CL = k * V, in L/h.
- `t_half_hat` is 1 / k, which is the mean residence time for this model, not the half life. The half life is ln(2) / k.
- `t_half_hat` is not the terminal half life in hours.
- `auc_inf` is not the AUC from zero to infinity. For an IV bolus it is DOSE / CL, in mg*h/L.
- `resid_sd` is the variance. Take the square root.
- `resid_sd` divides the SSE by n. Two parameters were fitted, so the divisor is n - 2.
- `resid_sd` is not sqrt(SSE / (n - 2)) at the fitted parameters. The SSE is the one the Adam fit ended on.
- `%s` is still a tensor. Call .item() on it.
- `%s` should be a single number.

**Functions this problem introduces:** torch.nn.Module, torch.nn.Parameter, torch.optim, zero_grad and step, state_dict

---

# W6D5. Twelve subjects, twenty four parameters, one backward pass

- topic: PyTorch: vectorised parameters, pooled against individual fitting, recovering between subject variability
- notebook: problems/week06/w6d5_torch_population_fit.ipynb
- solution: solutions/week06/w6d5_torch_population_fit_solution.ipynb
- expected time: 25-35 min

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

## The setup cell the student runs

This is given to them, already working. The data is synthetic and
seeded, so their numbers are always these numbers.

```python
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
```

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

The toolkit cell demonstrates each of those on a tiny made up example.
It is in the notebook and none of it is the answer.

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

**Variables this part is graded on:** `v`, `k`, `v_pool`, `k_pool`, `sse_pool`, `subject_bias`

The starter cell they see:

```python
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
```

**Every failure message part a can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `PopulationPK` is not defined. The class skeleton is in the part a cell.
- the model should register exactly two parameters named log_v and log_k, but it reports %r. Each has to be a torch.nn.Parameter assigned to an attribute of self.
- `forward` should return a tensor of predictions.
- a 3 subject model predicted shape %s for %d times. It should be (3, %d): one row per subject, one column per time. Use v[:, None] and t[None, :].
- `v_pool` is not the pooled volume. With n_subjects = 1 the (1, 9) predictions broadcast against the (12, 9) observations, so the loss covers the whole study.
- `k_pool` is not the pooled rate constant.
- `sse_pool` is the individual fit loss. Part a fits one common curve, which cannot do nearly as well.
- `sse_pool` is not the total squared residual of the pooled fit summed over all 12 subjects and all 9 times.
- `subject_bias` should be an array of numbers.
- `subject_bias` has one entry per TIME, so it was averaged over subjects. With a (12, 9) residual matrix you want dim=1, which collapses the times and leaves one number per subject.
- `subject_bias` has shape %s. It should hold one mean residual per subject, so %d numbers.
- `subject_bias` has the right magnitudes with every sign flipped. The task asks for observed minus predicted, so a positive value means the model under predicts that subject.
- `subject_bias` is not the mean residual per subject at the pooled fit. Compute c_obs minus the pooled predictions, then average over times.
- `PopulationPK(3, 30.0, 0.15)` raised %s: %s
- `%s` has shape %s in a 3 subject model. It should be a vector with one entry per subject, which torch.full((n_subjects,), ...) gives you.
- `%s` should be float64.
- calling a 3 subject model raised %s: %s
- `%s` is still a tensor. Call .item() on it.
- `%s` should be a single number.

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

**Variables this part is graded on:** `v_ind`, `k_ind`, `sse_ind`, `v_scipy`

The starter cell they see:

```python
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
```

**Every failure message part b can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `sse_ind` is still a tensor. Call .item() on it.
- `sse_ind` should be a single number.
- `v_ind` is still on the log scale. Exponentiate before reading it out.
- every entry of `v_ind` is the pooled volume, so the model still has n_subjects = 1. Build it with N_SUBJ.
- `v_ind` does not match the individual volume estimates. At this learning rate N_ITER iterations is enough to reach them to six decimal places, so check the broadcasting in forward and that the fit used c_obs.
- `k_ind` does not match the individual rate constant estimates.
- `sse_ind` is the pooled loss. It should be the final loss of the 24 parameter fit, which is far smaller.
- `sse_ind` is not the total loss at the end of the individual fit.
- `v_scipy` does not match the per subject curve_fit estimates. Loop over the rows of c_obs, fitting one subject at a time from p0=[V_START, K_START], and collect the volume from each.
- `%s` is still a tensor. Use .detach().numpy() to get the fitted values out.
- `%s` should be an array of numbers.
- `%s` has shape %s but there are %d subjects.

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

**Variables this part is graded on:** `v_gm`, `cv_v_pct`, `cv_k_pct`, `sse_ratio`

The starter cell they see:

```python
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
```

**Every failure message part c can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `v_gm` is the arithmetic mean. The geometric mean is exp(mean(log(v_ind))), and it comes out slightly lower.
- `v_gm` is not the geometric mean of the 12 fitted volumes.
- `cv_v_pct` is a fraction. The task asks for a percentage, so multiply by 100.
- `cv_v_pct` is 100 times the standard deviation of the logs, which is the small variability approximation. The task asks for 100 * sqrt(exp(var(log V)) - 1).
- `cv_v_pct` is not the between subject CV of volume as a percentage. Use the sample variance of the logs, np.var(..., ddof=1).
- `cv_k_pct` is not the between subject CV of the rate constant.
- `sse_ratio` is upside down. The task asks for sse_pool divided by sse_ind.
- `sse_ratio` is not sse_pool divided by sse_ind.
- `%s` is still a tensor. Call .item() on it.
- `%s` should be a single number.

**Functions this problem introduces:** torch.full, parameter vectors, None axis indexing, pooled against individual fitting, between subject CV
