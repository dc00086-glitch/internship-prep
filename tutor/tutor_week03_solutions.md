# Week 3 reference solutions

One way to do each part. The checks test the numbers, not the route, so a
different correct answer is still correct.

Do not paste these to the student. Use them to work out where their
version diverges, and say the smallest thing that gets them unstuck.

---

## W3D1. Comparing two groups when the data is skewed

### Part a

```python
# Part a. Descriptive statistics for each group.

desc = pd.DataFrame([
    {"GROUP": "Vehicle", "N": vehicle.size,
     "MEAN": np.mean(vehicle), "SD": np.std(vehicle, ddof=1),
     "MEDIAN": np.median(vehicle), "IQR": stats.iqr(vehicle)},
    {"GROUP": "Treated", "N": treated.size,
     "MEAN": np.mean(treated), "SD": np.std(treated, ddof=1),
     "MEDIAN": np.median(treated), "IQR": stats.iqr(treated)},
])
desc[["MEAN", "SD", "MEDIAN", "IQR"]] = desc[["MEAN", "SD", "MEDIAN", "IQR"]].round(2)

print(desc)
```

### Part b

```python
# Part b. Student's t-test and Welch's t-test on the raw volumes.

_student = stats.ttest_ind(vehicle, treated)
_welch = stats.ttest_ind(vehicle, treated, equal_var=False)

t_student = float(_student.statistic)
p_student = float(_student.pvalue)
t_welch = float(_welch.statistic)
p_welch = float(_welch.pvalue)

print("Student t = %s  p = %s" % (t_student, p_student))
print("Welch   t = %s  p = %s" % (t_welch, p_welch))
```

### Part c

```python
# Part c. Mann-Whitney on the raw data, and Welch's t-test on the logs.

_mw = stats.mannwhitneyu(vehicle, treated, alternative="two-sided")
_logt = stats.ttest_ind(np.log(vehicle), np.log(treated), equal_var=False)

u_stat = float(_mw.statistic)
p_mw = float(_mw.pvalue)
t_log = float(_logt.statistic)
p_log = float(_logt.pvalue)

print("Mann-Whitney U = %s  p = %s" % (u_stat, p_mw))
print("Welch on logs  t = %s  p = %s" % (t_log, p_log))
```

---

## W3D2. AUC is lognormal, so report the geometric mean

### Part a

```python
# Part a. The arithmetic summary.

arith_mean = float(np.mean(auc))
arith_sd = float(np.std(auc, ddof=1))
arith_cv = 100.0 * arith_sd / arith_mean

print("arithmetic mean : %s" % arith_mean)
print("arithmetic SD   : %s" % arith_sd)
print("CV percent      : %s" % arith_cv)
print("median for comparison: %.3f" % float(np.median(auc)))
```

### Part b

```python
# Part b. The geometric summary.

_log_auc = np.log(auc)

geo_mean = float(np.exp(np.mean(_log_auc)))
sd_log = float(np.std(_log_auc, ddof=1))
geo_cv = 100.0 * float(np.sqrt(np.exp(sd_log ** 2) - 1.0))

print("geometric mean : %s" % geo_mean)
print("SD of logs     : %s" % sd_log)
print("geometric CV   : %s" % geo_cv)
if geo_mean is not None and arith_mean is not None:
    print("")
    print("geometric %.3f <= arithmetic %.3f ?" % (geo_mean, arith_mean),
          geo_mean <= arith_mean)
```

### Part c

```python
# Part c. Test normality on the raw values and on the logs.

p_raw = float(stats.shapiro(auc).pvalue)
p_log = float(stats.shapiro(np.log(auc)).pvalue)
use_log = bool(p_raw < 0.05 and p_log >= 0.05)

print("shapiro p, raw  : %s" % p_raw)
print("shapiro p, logs : %s" % p_log)
print("work on the log scale?", use_log)
```

---

## W3D3. Fitting a decay curve, and how sure you are of the parameters

### Part a

```python
# Part a. Define the model, then fit it.

def decay(t, c0, k):
    """Concentration after an IV bolus into one compartment."""
    return c0 * np.exp(-k * t)


popt, pcov = curve_fit(decay, obs["TIME_H"].to_numpy(), obs["CONC_MG_L"].to_numpy(),
                       p0=[conc[0], np.log(2.0) / 7.0])

print("popt:", popt)
print("pcov:", pcov)
```

### Part b

```python
# Part b. The fitted parameters and their standard errors.

_perr = np.sqrt(np.diag(pcov))

c0_hat = float(popt[0])
k_hat = float(popt[1])
se_c0 = float(_perr[0])
se_k = float(_perr[1])

if None not in (c0_hat, k_hat, se_c0, se_k):
    print("C0 = %.4f +/- %.4f mg/L" % (c0_hat, se_c0))
    print("k  = %.5f +/- %.5f 1/h" % (k_hat, se_k))
    print("estimate over standard error: C0 %.1f, k %.1f"
          % (c0_hat / se_c0, k_hat / se_k))
```

### Part c

```python
# Part c. Half-life, predictions, and the residual summary.

t_half = float(np.log(2.0) / k_hat)
pred = decay(obs["TIME_H"].to_numpy(), *popt)
_resid = obs["CONC_MG_L"].to_numpy() - pred
rmse = float(np.sqrt(np.mean(_resid ** 2)))

print("half-life : %s h" % t_half)
print("predicted :", None if pred is None else np.round(pred, 3))
print("observed  :", obs["CONC_MG_L"].to_numpy())
print("rmse      : %s" % rmse)
```

---

## W3D4. Solving the compartment model as an ODE, and surviving a discontinuity

### Part a

```python
# Part a. The one compartment ODE, solved numerically after an IV bolus.

def rhs(t, y, cl, v):
    """dA/dt for one compartment. y[0] is the amount in mg."""
    return [-(cl / v) * y[0]]


sol = solve_ivp(rhs, (0.0, 24.0), [DOSE_MG], t_eval=T_GRID, args=(CL_L_H, V_L),
                rtol=1e-8, atol=1e-10)
conc_num = sol.y[0] / V_L

print("solved ok:", None if sol is None else sol.success)
if conc_num is not None:
    print("conc at 0, 6, 12, 24 h:", np.round(conc_num[[0, 24, 48, 96]], 4))
```

### Part b

```python
# Part b. The exact solution, and how far the numerical one strayed.

conc_exact = (DOSE_MG / V_L) * np.exp(-(CL_L_H / V_L) * T_GRID)
max_abs_err = float(np.max(np.abs(conc_num - conc_exact)))

print("exact at 0, 6, 12, 24 h:",
      None if conc_exact is None else np.round(conc_exact[[0, 24, 48, 96]], 4))
print("max absolute error:", max_abs_err)
```

### Part c

```python
# Part c. A 2 hour infusion into the same compartment.

def rhs_inf(t, y, cl, v, rate, t_off):
    """dA/dt with a zero order input that stops at t_off."""
    r = rate if t < t_off else 0.0
    return [r - (cl / v) * y[0]]


sol_inf = solve_ivp(rhs_inf, (0.0, 12.0), [0.0], t_eval=T_GRID_INF,
                    args=(CL_L_H, V_L, INF_RATE_MG_H, INF_DURATION_H),
                    max_step=0.05, rtol=1e-8, atol=1e-10)
conc_inf = sol_inf.y[0] / V_L
cmax_inf = float(np.max(conc_inf))
t_cmax = float(sol_inf.t[int(np.argmax(conc_inf))])

print("solved ok:", None if sol_inf is None else sol_inf.success)
print("Cmax = %s mg/L at t = %s h" % (cmax_inf, t_cmax))
```

---

## W3D5. Emax and sigmoid Emax: fitting a concentration response curve

### Part a

```python
# Part a. One point per concentration.

agg = (assay
       .groupby("CONC_UM")
       .agg(MEAN_PCT=("INHIB_PCT", "mean"),
            SD_PCT=("INHIB_PCT", "std"),
            N=("INHIB_PCT", "size"))
       .reset_index()
       .sort_values("CONC_UM")
       .reset_index(drop=True))

print(agg)
```

### Part b

```python
# Part b. The plain Emax model.

def emax_model(c, emax, ec50):
    """Plain Emax: half of emax is reached at c = ec50."""
    return emax * c / (ec50 + c)


_c = agg["CONC_UM"].to_numpy()
_y = agg["MEAN_PCT"].to_numpy()
_top = float(_y.max())
_half = float(_c[int(np.argmin(np.abs(_y - _top / 2.0)))])

_p1, _ = curve_fit(emax_model, _c, _y, p0=[_top, _half],
                   bounds=([0.0, 1e-6], [200.0, 1e4]))

emax1 = float(_p1[0])
ec50_1 = float(_p1[1])
rmse1 = float(np.sqrt(np.mean((_y - emax_model(_c, *_p1)) ** 2)))

print("Emax model: Emax = %s  EC50 = %s  rmse = %s" % (emax1, ec50_1, rmse1))
```

### Part c

```python
# Part c. The sigmoid Emax model, and the comparison.

def sigmoid_model(c, emax, ec50, h):
    """Sigmoid Emax. h = 1 gives the plain model back."""
    return emax * c ** h / (ec50 ** h + c ** h)


_p2, _ = curve_fit(sigmoid_model, _c, _y, p0=[_top, _half, 1.0],
                   bounds=([0.0, 1e-6, 0.1], [200.0, 1e4, 10.0]))

emax2 = float(_p2[0])
ec50_2 = float(_p2[1])
hill = float(_p2[2])
rmse2 = float(np.sqrt(np.mean((_y - sigmoid_model(_c, *_p2)) ** 2)))

print("sigmoid: Emax = %s  EC50 = %s  h = %s  rmse = %s" % (emax2, ec50_2, hill, rmse2))
if None not in (rmse1, rmse2):
    print("")
    print("rmse went from %.4f to %.4f" % (rmse1, rmse2))
    print("the richer model is never worse, which is why this alone proves nothing")
```
