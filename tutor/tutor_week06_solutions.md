# Week 6 reference solutions

One way to do each part. The checks test the numbers, not the route, so a
different correct answer is still correct.

Do not paste these to the student. Use them to work out where their
version diverges, and say the smallest thing that gets them unstuck.

---

## W6D1. Tensors: numpy with two things added

### Part a

```python
# Part a. Into torch, with the dtype stated.

t_tensor = torch.tensor(time_np, dtype=torch.float64)
c_tensor = torch.tensor(conc_np, dtype=torch.float64)
dtype_default = torch.tensor([1, 2, 3]).dtype
dtype_asked = torch.tensor([1, 2, 3], dtype=torch.float64).dtype

print("t_tensor:", t_tensor)
print("c_tensor:", c_tensor)
print("dtype from python ints, inferred:", dtype_default)
print("dtype when asked for explicitly  :", dtype_asked)
```

### Part b

```python
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
```

### Part c

```python
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
```

---

## W6D2. Autograd: derivatives you did not have to derive

### Part a

```python
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
```

### Part b

```python
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
```

### Part c

```python
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
```

---

## W6D3. Gradient descent, written out by hand

### Part a

```python
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
```

### Part b

```python
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
```

### Part c

```python
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
```

---

## W6D4. nn.Module and an optimiser: the loop everyone actually writes

### Part a

```python
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
```

### Part b

```python
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
```

### Part c

```python
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
```

---

## W6D5. Twelve subjects, twenty four parameters, one backward pass

### Part a

```python
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
```

### Part b

```python
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
```

### Part c

```python
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
```
