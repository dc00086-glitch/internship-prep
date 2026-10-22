# Week 7 reference solutions

One way to do each part. The checks test the numbers, not the route, so a
different correct answer is still correct.

Do not paste these to the student. Use them to work out where their
version diverges, and say the smallest thing that gets them unstuck.

---

## W7D1. Non-compartmental analysis: Cmax, Tmax and AUC by trapezoid

### Part a

```python
# Part a. Peak, time of peak, and the linear trapezoidal AUC.

t = profile["TIME_H"].to_numpy()
c = profile["CONC_MG_L"].to_numpy()

cmax = float(np.max(c))
tmax = float(t[np.argmax(c)])
auc_linear = float(np.sum(np.diff(t) * (c[:-1] + c[1:]) / 2.0))

print("Cmax       : %s mg/L" % cmax)
print("Tmax       : %s h" % tmax)
print("AUC linear : %s mg*h/L" % auc_linear)
```

### Part b

```python
# Part b. Linear up, log down.

c1 = c[:-1]
c2 = c[1:]
dt = np.diff(t)

_linear_each = dt * (c1 + c2) / 2.0
with np.errstate(divide="ignore", invalid="ignore"):
    _log_each = (c1 - c2) * dt / np.log(c1 / c2)

_use_log = (c2 < c1) & (c2 > 0) & (c1 != c2)
auc_linlog = float(np.sum(np.where(_use_log, _log_each, _linear_each)))
pct_diff = float(100.0 * (auc_linlog - auc_linear) / auc_linear)

print("AUC linear        : %s" % auc_linear)
print("AUC lin-up/log-down: %s" % auc_linlog)
print("difference         : %s percent" % pct_diff)
```

### Part c

```python
# Part c. Partial AUC over the first 12 hours.

_each = np.where(_use_log, _log_each, _linear_each)
_idx_12 = int(np.where(t == 12.0)[0][0])

auc_0_12 = float(np.sum(_each[:_idx_12]))
frac_in_12h = float(auc_0_12 / auc_linlog)

print("AUC 0 to 12 h : %s mg*h/L" % auc_0_12)
print("AUC 0 to last : %s mg*h/L" % auc_linlog)
print("fraction in the first 12 h: %s" % frac_in_12h)
```

---

## W7D2. Finishing the NCA: lambda_z, AUC to infinity, clearance and MRT

### Part a

```python
# Part a. Select the terminal window by adjusted R2, then fit lambda_z.

_eligible = int(np.sum(t >= TMAX))

_best = None
for _n in range(3, _eligible + 1):
    _tt, _cc = t[-_n:], c[-_n:]
    _logc = np.log(_cc)
    _slope, _intercept = np.polyfit(_tt, _logc, deg=1)
    _fitted = _slope * _tt + _intercept
    _r2 = 1.0 - float(np.sum((_logc - _fitted) ** 2)) / float(np.sum((_logc - _logc.mean()) ** 2))
    _adj = 1.0 - (1.0 - _r2) * (_n - 1) / (_n - 2)
    if _best is None or _adj > _best[0]:
        _best = (_adj, _n, _slope)

adj_r2 = float(_best[0])
n_points_used = int(_best[1])
lambda_z = float(-_best[2])
t_half = float(np.log(2.0) / lambda_z)

print("points used : %s" % n_points_used)
print("lambda_z    : %s 1/h" % lambda_z)
print("half-life   : %s h" % t_half)
print("adjusted R2 : %s" % adj_r2)
```

### Part b

```python
# Part b. AUC to infinity, and the extrapolated fraction.

auc_inf = float(AUC_LAST + C_LAST / lambda_z)
pct_extrap = float(100.0 * (auc_inf - AUC_LAST) / auc_inf)

_auc_inf_12h = AUC_TO_12H + C_AT_12H / lambda_z
pct_extrap_12h = float(100.0 * (_auc_inf_12h - AUC_TO_12H) / _auc_inf_12h)

print("sampled to 48 h: AUC_inf %s, extrapolated %s percent" % (auc_inf, pct_extrap))
print("sampled to 12 h: extrapolated %s percent" % pct_extrap_12h)
```

### Part c

```python
# Part c. Apparent clearance, apparent volume, and mean residence time.

cl_f = float(DOSE_MG / auc_inf)
vz_f = float(cl_f / lambda_z)

_aumc_inf = (AUMC_LAST
             + T_LAST * C_LAST / lambda_z
             + C_LAST / lambda_z ** 2)
mrt = float(_aumc_inf / auc_inf)

print("CL/F : %s L/h" % cl_f)
print("Vz/F : %s L" % vz_f)
print("MRT  : %s h   (half-life was %s h)" % (mrt, t_half))
```

---

## W7D3. Fitting a one compartment oral model, and checking it against NCA

### Part a

```python
# Part a. The depot and central system, and a prediction function.

def rhs(t, y, ka, v, cl):
    """y[0] is the depot amount, y[1] the central amount, both in mg."""
    absorbed = ka * y[0]
    return [-absorbed, absorbed - (cl / v) * y[1]]


def predict(t_eval, ka, v, cl):
    """Concentration in mg/L at each time in t_eval."""
    t_eval = np.asarray(t_eval, dtype=float)
    sol = solve_ivp(rhs, (0.0, float(t_eval.max())), [DOSE_MG, 0.0],
                    t_eval=t_eval, args=(ka, v, cl), rtol=1e-8, atol=1e-10)
    return sol.y[1] / v


print("check at the observed times, with roughly sensible parameters:")
print(None if predict(t, 1.0, 50.0, 5.5) is None
      else np.round(predict(t, 1.0, 50.0, 5.5), 4))
print("observed:", c)
```

### Part b

```python
# Part b. Fit ka, V/F and CL/F.

_p0 = [3.0 / TMAX, NCA_CL_F / NCA_LAMBDA_Z, NCA_CL_F]
popt, _pcov = curve_fit(predict, t, c, p0=_p0,
                        bounds=([0.0, 0.0, 0.0], [np.inf, np.inf, np.inf]))

ka_hat = float(popt[0])
v_hat = float(popt[1])
cl_hat = float(popt[2])
rmse = float(np.sqrt(np.mean((c - predict(t, *popt)) ** 2)))

print("ka   = %s 1/h   (true 1.15)" % ka_hat)
print("V/F  = %s L     (true 60.0)" % v_hat)
print("CL/F = %s L/h   (true 5.61)" % cl_hat)
print("rmse = %s mg/L" % rmse)
```

### Part c

```python
# Part c. Compare the model with NCA.

_kel_model = cl_hat / v_hat
t_half_model = float(np.log(2.0) / _kel_model)
cl_ratio = float(cl_hat / NCA_CL_F)
flip_flop = bool(ka_hat < _kel_model)

print("half-life, model : %s h" % t_half_model)
print("half-life, NCA   : %.4f h" % NCA_T_HALF)
print("CL ratio model/NCA: %s" % cl_ratio)
print("flip-flop?        : %s" % flip_flop)
```

---

## W7D4. Two compartments, and what a one compartment model gets wrong

### Part a

```python
# Part a. The two compartment system, fitted.

def rhs2(t, y, cl, v1, q, v2):
    """y[0] central amount, y[1] peripheral amount, both mg."""
    to_tissue = (q / v1) * y[0]
    from_tissue = (q / v2) * y[1]
    return [-(cl / v1) * y[0] - to_tissue + from_tissue,
            to_tissue - from_tissue]


def predict2(t_eval, cl, v1, q, v2):
    """Central concentration in mg/L at each time."""
    t_eval = np.asarray(t_eval, dtype=float)
    sol = solve_ivp(rhs2, (0.0, float(t_eval.max())), [DOSE_MG, 0.0],
                    t_eval=t_eval, args=(cl, v1, q, v2), rtol=1e-10, atol=1e-12)
    return sol.y[0] / v1


popt2, _ = curve_fit(predict2, t, c, p0=[3.0, 10.0, 6.0, 25.0],
                     bounds=([0.0] * 4, [np.inf] * 4))

print("fitted CL %s, V1 %s, Q %s, V2 %s" % tuple(popt2) if popt2 is not None else "not fitted")
print("truth   CL 3.2, V1 9.0, Q 6.5, V2 26.0")
```

### Part b

```python
# Part b. A one compartment fit, for comparison.

def one_comp(t_eval, cl, v):
    """IV bolus into a single compartment, in closed form."""
    return (DOSE_MG / v) * np.exp(-(cl / v) * np.asarray(t_eval, dtype=float))


popt1, _ = curve_fit(one_comp, t, c, p0=[3.0, 25.0],
                     bounds=([0.0, 0.0], [np.inf, np.inf]))
rmse1 = float(np.sqrt(np.mean((c - one_comp(t, *popt1)) ** 2)))
rmse2 = float(np.sqrt(np.mean((c - predict2(t, *popt2)) ** 2)))

print("one compartment: CL %s, V %s, rmse %s" % (
    popt1[0] if popt1 is not None else None,
    popt1[1] if popt1 is not None else None, rmse1))
print("two compartment: rmse %s" % rmse2)
if popt1 is not None:
    print("")
    print("one compartment residuals (observed minus fitted):")
    print(np.round(c - one_comp(t, *popt1), 4))
```

### Part c

```python
# Part c. What the one compartment model costs you early.

vss = float(popt2[1] + popt2[3])

_i12 = int(np.where(t == 12.0)[0][0])
_pred12 = float(one_comp(t[_i12:_i12 + 1], *popt1)[0])
late_bias = float(100.0 * (_pred12 - c[_i12]) / c[_i12])

auc_one = float(DOSE_MG / popt1[0])
auc_two = float(DOSE_MG / popt2[0])
auc_bias = float(100.0 * (auc_one - auc_two) / auc_two)

print("Vss (two compartment) : %s L" % vss)
print("V   (one compartment) : %s L" % (popt1[1] if popt1 is not None else None))
print("")
print("error at 12 h, one compartment: %s percent" % late_bias)
print("")
print("implied AUC: one compartment %s, two compartment %s mg*h/L" % (auc_one, auc_two))
print("exposure understated by       : %s percent" % auc_bias)
if popt1 is not None:
    print("")
    print("for contrast, the early points are fine:")
    for _i in range(4):
        _p = one_comp(t[_i:_i + 1], *popt1)[0]
        print("   t=%6.3f h  observed %7.4f  one cmt %7.4f  (%+.1f percent)"
              % (t[_i], c[_i], _p, 100 * (_p - c[_i]) / c[_i]))
```

---

## W7D5. Multiple dosing: accumulation, steady state, and how long it takes

### Part a

```python
# Part a. Simulate 20 doses, 12 hours apart.

def rhs(t, y, ka, v, cl):
    """y[0] depot, y[1] central, both mg."""
    absorbed = ka * y[0]
    return [-absorbed, absorbed - (cl / v) * y[1]]


def simulate(n_doses):
    """Return (times, concentrations) across the whole regimen."""
    grid = np.linspace(0.0, TAU_H, POINTS_PER_INTERVAL)
    state = np.array([0.0, 0.0])
    all_t, all_c = [], []
    for i in range(n_doses):
        state = state.copy()
        state[0] += DOSE_MG
        sol = solve_ivp(rhs, (0.0, TAU_H), state, t_eval=grid,
                        args=(KA, V_F, CL_F), rtol=1e-9, atol=1e-11)
        all_t.append(grid + i * TAU_H)
        all_c.append(sol.y[1] / V_F)
        state = sol.y[:, -1]
    return np.concatenate(all_t), np.concatenate(all_c)


times, conc = simulate(N_DOSES)

if conc is not None:
    print("points simulated:", len(times))
    print("peak overall    : %.4f mg/L" % conc.max())
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(times, conc, lw=1.0)
    ax.set_xlabel("Time (h)"); ax.set_ylabel("Concentration (mg/L)")
    ax.set_title("200 mg every 12 h")
    plt.show()
```

### Part b

```python
# Part b. Accumulation, simulated and predicted.

_first = times < TAU_H
_last = times >= (N_DOSES - 1) * TAU_H

cmax_first = float(conc[_first].max())
cmax_ss = float(conc[_last].max())
r_sim = float(cmax_ss / cmax_first)
r_theory = float(1.0 / (1.0 - np.exp(-KEL * TAU_H)))

print("Cmax, first interval : %s mg/L" % cmax_first)
print("Cmax, last interval  : %s mg/L" % cmax_ss)
print("accumulation ratio, simulated : %s" % r_sim)
print("accumulation ratio, algebraic : %s" % r_theory)
```

### Part c

```python
# Part c. Average exposure, and time to steady state.

cavg_ss = float(DOSE_MG / (TAU_H * CL_F))
t_to_90pct = float(times[int(np.argmax(conc >= 0.90 * cmax_ss))])
n_half_lives = float(t_to_90pct / T_HALF)

print("Cavg at steady state : %s mg/L" % cavg_ss)
print("time to 90 percent   : %s h" % t_to_90pct)
print("which is             : %s half-lives" % n_half_lives)
```
