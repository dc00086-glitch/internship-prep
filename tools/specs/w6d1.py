PID = "W6D1"
WEEK = 6
DAY = 1
SLUG = "w6d1_torch_tensors"
TITLE = "Tensors: numpy with two things added"
TOPIC = "PyTorch: tensors, dtypes, shapes, numpy interop, the autograd flag"
EST = "20-30 min"

SCENARIO_MD = '''
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
'''

CONCEPT_MD = '''
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
'''

TASK_MD = '''
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
'''

DATA_CODE = '''
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
'''

TOOLKIT_MD = '''
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
'''

TOOLKIT_CODE = '''
import numpy as np
import torch

# A tiny made up array. Nothing here is the answer.
demo_np = np.array([1.0, 4.0, 9.0, 16.0])
print("--- 1. making tensors, and the dtype surprise ---")
demo_a = torch.tensor(demo_np)
demo_b = torch.tensor([1, 2, 3])
demo_c = torch.tensor([1.0, 2.0, 3.0])
demo_d = torch.tensor([1, 2, 3], dtype=torch.float64)
print("from a float64 numpy array :", demo_a.dtype)
print("from python ints           :", demo_b.dtype, " <- integer, no gradients possible")
print("from python floats         :", demo_c.dtype, " <- float32, not 64")
print("asked for explicitly       :", demo_d.dtype, " <- say what you want")

print("")
print("--- 2. nearly everything works as it does in numpy ---")
print("demo_a          :", demo_a)
print("demo_a * 2      :", demo_a * 2)
print("demo_a[1:3]     :", demo_a[1:3])
print("demo_a > 5      :", demo_a > 5)
print("demo_a[demo_a>5]:", demo_a[demo_a > 5])
print("max %.1f at position %d" % (demo_a.max().item(), torch.argmax(demo_a).item()))
print("torch says dim=, numpy says axis=:")
demo_2d = torch.arange(6, dtype=torch.float64).reshape(2, 3)
print(demo_2d)
print("sum(dim=0):", demo_2d.sum(dim=0))

print("")
print("--- 3. .numpy() SHARES memory ---")
demo_t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
demo_view = demo_t.numpy()
print("before :", demo_view)
demo_t[0] = 99.0
print("after changing the TENSOR, the numpy array reads:", demo_view)
print("shares memory:", np.shares_memory(demo_view, demo_t.numpy()))
print("use .numpy().copy() when you want a snapshot")

print("")
print("--- 4. the autograd flag, and what it does to a result ---")
demo_plain = torch.tensor([2.0], dtype=torch.float64)
demo_track = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
demo_r1 = demo_plain * 3 + 1
demo_r2 = demo_track * 3 + 1
print("from a plain tensor  : grad_fn =", demo_r1.grad_fn)
print("from a tracked tensor: grad_fn =", demo_r2.grad_fn)
print("that grad_fn is the recorded history, and tomorrow it is what gets differentiated")
print("")
try:
    demo_track.numpy()
except RuntimeError as err:
    print("calling .numpy() on a tracked tensor refuses:")
    print("  ", str(err)[:90])
print("the fix is .detach().numpy(), meaning: values yes, history no")
'''

PARTS = [
    dict(
        label="a",
        prompt_md='''
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
''',
        starter='''
# Part a. Into torch, with the dtype stated.

t_tensor = None       # YOUR CODE: time_np as a float64 tensor
c_tensor = None       # YOUR CODE: conc_np as a float64 tensor
dtype_default = None  # YOUR CODE: the dtype of torch.tensor([1, 2, 3])
dtype_asked = None    # YOUR CODE: the dtype of torch.tensor([1, 2, 3], dtype=torch.float64)

print("t_tensor:", t_tensor)
print("c_tensor:", c_tensor)
print("dtype from python ints, inferred:", dtype_default)
print("dtype when asked for explicitly  :", dtype_asked)
''',
        solution='''
# Part a. Into torch, with the dtype stated.

t_tensor = torch.tensor(time_np, dtype=torch.float64)
c_tensor = torch.tensor(conc_np, dtype=torch.float64)
dtype_default = torch.tensor([1, 2, 3]).dtype
dtype_asked = torch.tensor([1, 2, 3], dtype=torch.float64).dtype

print("t_tensor:", t_tensor)
print("c_tensor:", c_tensor)
print("dtype from python ints, inferred:", dtype_default)
print("dtype when asked for explicitly  :", dtype_asked)
''',
    ),
    dict(
        label="b",
        prompt_md='''
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
''',
        starter='''
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
''',
        solution='''
# Part b. Familiar numpy moves, in torch.

cmax = c_tensor.max().item()
tmax = t_tensor[torch.argmax(c_tensor)].item()
log_c = torch.log(c_tensor)
mask = c_tensor >= LLOQ
c_measurable = c_tensor[mask]

print("cmax: %s mg/L at tmax %s h" % (cmax, tmax))
print("log_c:", log_c)
print("mask :", mask)
print("kept :", c_measurable)
''',
    ),
    dict(
        label="c",
        prompt_md='''
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
''',
        starter='''
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
''',
        solution='''
# Part c. The autograd flag, and shared memory.

plain = torch.tensor(2.0, dtype=torch.float64)
tracked = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)

result_plain = plain * 3 + 1
result_tracked = tracked * 3 + 1

has_grad_fn_plain = bool(result_plain.grad_fn is not None)
has_grad_fn_tracked = bool(result_tracked.grad_fn is not None)
shares_memory = bool(np.shares_memory(c_tensor.numpy(), c_tensor.numpy()))

print("result from plain tensor  : grad_fn present?", has_grad_fn_plain)
print("result from tracked tensor: grad_fn present?", has_grad_fn_tracked)
print("two .numpy() views of c_tensor share memory?", shares_memory)
''',
    ),
]


def build_expected(ns):
    import numpy as np
    t, c = ns["time_np"], ns["conc_np"]
    lloq = ns["LLOQ"]
    return {
        "n": int(t.size),
        "time": [float(v) for v in t],
        "conc": [float(v) for v in c],
        "cmax": float(c.max()),
        "tmax": float(t[int(np.argmax(c))]),
        "argmax": int(np.argmax(c)),
        "log_c": [float(v) for v in np.log(c)],
        "mask": [bool(v) for v in (c >= lloq)],
        "n_measurable": int((c >= lloq).sum()),
        "measurable": [float(v) for v in c[c >= lloq]],
    }


CHECK_CODE = '''
import torch as _torch


def _part_a():
    t_tensor = _need("t_tensor")
    c_tensor = _need("c_tensor")
    dtype_default = _need("dtype_default")
    dtype_asked = _need("dtype_asked")

    if not isinstance(t_tensor, _torch.Tensor):
        return "`t_tensor` should be a torch Tensor, not a %s." % type(t_tensor).__name__
    if not isinstance(c_tensor, _torch.Tensor):
        return "`c_tensor` should be a torch Tensor."
    if t_tensor.dtype != _torch.float64:
        return ("`t_tensor` has dtype %s. The task asks for float64, stated explicitly, "
                "because torch defaults to float32." % t_tensor.dtype)
    if c_tensor.dtype != _torch.float64:
        return "`c_tensor` has dtype %s, but float64 was asked for." % c_tensor.dtype
    if tuple(t_tensor.shape) != (_EXP["n"],):
        return "`t_tensor` should hold the %d times." % _EXP["n"]
    if not _arr_close(t_tensor.detach().numpy(), _EXP["time"]):
        return "`t_tensor` does not hold the values of time_np."
    if not _arr_close(c_tensor.detach().numpy(), _EXP["conc"]):
        return "`c_tensor` does not hold the values of conc_np."

    if dtype_default == _torch.float64:
        return ("`dtype_default` came out float64, so a dtype was passed. The point is to "
                "see what torch.tensor([1, 2, 3]) infers on its own.")
    if dtype_default != _torch.int64:
        return ("`dtype_default` should be the dtype torch infers from a list of Python "
                "integers, which is an integer type, but it is %s." % dtype_default)
    if dtype_asked != _torch.float64:
        return "`dtype_asked` should be torch.float64, from passing dtype explicitly."
    return None


def _part_b():
    cmax = _need("cmax")
    tmax = _need("tmax")
    log_c = _need("log_c")
    mask = _need("mask")
    c_measurable = _need("c_measurable")

    if isinstance(cmax, _torch.Tensor):
        return ("`cmax` is still a tensor. Call .item() on it to get an ordinary Python "
                "float.")
    got = _as_scalar(cmax)
    if got is None or not _num_close(got, _EXP["cmax"], rtol=1e-9):
        return "`cmax` is not the highest concentration."

    if isinstance(tmax, _torch.Tensor):
        return "`tmax` is still a tensor. Call .item() on it."
    got = _as_scalar(tmax)
    if got is None:
        return "`tmax` should be a single number."
    if _num_close(got, float(_EXP["argmax"])):
        return ("`tmax` is the index of the peak, not the time. Use it to index t_tensor.")
    if not _num_close(got, _EXP["tmax"], rtol=1e-9):
        return "`tmax` is not the time at which the peak occurred."

    if not isinstance(log_c, _torch.Tensor):
        return "`log_c` should be a tensor, from torch.log."
    if not _arr_close(log_c.detach().numpy(), _EXP["log_c"], rtol=1e-9):
        return "`log_c` is not the natural log of c_tensor."

    if not isinstance(mask, _torch.Tensor):
        return "`mask` should be a tensor."
    if mask.dtype != _torch.bool:
        return ("`mask` has dtype %s. A comparison like c_tensor >= LLOQ gives a boolean "
                "tensor directly." % mask.dtype)
    if [bool(v) for v in mask.tolist()] != _EXP["mask"]:
        if [bool(v) for v in (~mask).tolist()] == _EXP["mask"]:
            return ("`mask` is inverted. True should mean the sample IS measurable, so the "
                    "comparison is c_tensor >= LLOQ.")
        return "`mask` does not flag the measurable samples."

    if not isinstance(c_measurable, _torch.Tensor):
        return "`c_measurable` should be a tensor."
    if c_measurable.numel() == _EXP["n"]:
        return ("`c_measurable` still has every sample. Index c_tensor with your boolean "
                "mask.")
    if c_measurable.numel() != _EXP["n_measurable"]:
        return ("`c_measurable` holds %d values but %d samples are at or above the LLOQ."
                % (c_measurable.numel(), _EXP["n_measurable"]))
    if not _arr_close(c_measurable.detach().numpy(), _EXP["measurable"], rtol=1e-9):
        return "`c_measurable` does not hold the measurable concentrations."
    return None


def _part_c():
    plain = _need("plain")
    tracked = _need("tracked")
    has_plain = _need("has_grad_fn_plain")
    has_tracked = _need("has_grad_fn_tracked")
    shares = _need("shares_memory")

    if not isinstance(plain, _torch.Tensor) or not isinstance(tracked, _torch.Tensor):
        return "`plain` and `tracked` should both be torch tensors."
    if plain.requires_grad:
        return "`plain` should NOT have requires_grad set. That is the point of the contrast."
    if not tracked.requires_grad:
        return "`tracked` should be created with requires_grad=True."
    if plain.dtype != _torch.float64 or tracked.dtype != _torch.float64:
        return "Both tensors should be float64."

    if bool(has_plain) is not False:
        return ("`has_grad_fn_plain` should be False. Nothing in that computation was being "
                "tracked, so there is no history to record.")
    if bool(has_tracked) is not True:
        return ("`has_grad_fn_tracked` should be True. A result computed from a tracked "
                "tensor carries a grad_fn, which is the recorded history.")
    if bool(shares) is not True:
        return ("`shares_memory` should be True. tensor.numpy() returns a VIEW of the same "
                "buffer, not a copy, which is why .numpy().copy() exists.")
    return None
'''

# Functions this problem introduces. Keys must exist in tools/function_catalog.py.
FUNCTIONS = [
    "torch.tensor",
    "tensor dtypes",
    ".item()",
    "tensor to numpy",
    "requires_grad",
]
