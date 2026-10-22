# Week 8 reference solutions

One way to do each part. The checks test the numbers, not the route, so a
different correct answer is still correct.

Do not paste these to the student. Use them to work out where their
version diverges, and say the smallest thing that gets them unstuck.

---

## W8D1. PK/PD link: why the effect peaks hours after the concentration

### Part a

```python
# Part a. The hysteresis loop.

peak_idx = int(np.argmax(c_obs))

tmax_c = float(t_obs[peak_idx])
tmax_e = float(t_obs[int(np.argmax(e_obs))])

e_rising = float(np.interp(C_PROBE, c_obs[:peak_idx + 1], e_obs[:peak_idx + 1]))
e_falling = float(np.interp(C_PROBE, c_obs[peak_idx:][::-1], e_obs[peak_idx:][::-1]))

print("concentration peaks at %r h, effect at %r h" % (tmax_c, tmax_e))
print("at %.2f mg/L: rising limb %r ms, falling limb %r ms" % (C_PROBE, e_rising, e_falling))
if e_rising is not None and e_falling is not None:
    print("difference at the same concentration: %.3f ms" % (e_falling - e_rising))

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(c_obs, e_obs, "o-", color="#333333")
for _i in range(0, len(t_obs), 2):
    ax.annotate("%g h" % t_obs[_i], (c_obs[_i], e_obs[_i]),
                textcoords="offset points", xytext=(6, -3), fontsize=8)
ax.axvline(C_PROBE, ls=":", color="grey")
ax.set_xlabel("plasma concentration (mg/L)")
ax.set_ylabel("change in QTc (ms)")
ax.set_title("W8D1 effect against concentration, joined in time order")
plt.show()
```

### Part b

```python
# Part b. The direct Emax fit, and its residuals.

def direct_emax(c, emax, ec50):
    """Effect as a function of the concentration at that instant."""
    return emax * c / (ec50 + c)


_popt, _pcov = curve_fit(direct_emax, c_obs, e_obs, p0=[20.0, 1.0],
                         bounds=([0.0, 1e-3], [200.0, 100.0]))
emax_direct = float(_popt[0])
ec50_direct = float(_popt[1])

_pred = direct_emax(c_obs, emax_direct, ec50_direct)
_res = e_obs - _pred
sse_direct = float((_res ** 2).sum())

resid_early = float(_res[:peak_idx + 1].mean())
resid_late = float(_res[peak_idx + 1:].mean())

print("direct fit: Emax %r ms, EC50 %r mg/L" % (emax_direct, ec50_direct))
print("SSE       : %r" % (sse_direct,))
print("mean residual before the peak: %r" % (resid_early,))
print("mean residual after the peak : %r" % (resid_late,))
```

### Part c

```python
# Part c. The effect compartment model.

def plasma(t):
    """Plasma concentration at arbitrary times, interpolated from the samples."""
    return np.interp(t, t_obs, c_obs)


def link_model(t, emax, ec50, ke0):
    """Effect driven by the effect site concentration."""
    t = np.asarray(t, dtype=float)
    sol = solve_ivp(lambda tt, y: ke0 * (plasma(tt) - y[0]),
                    (0.0, float(t.max())), [0.0], t_eval=t,
                    rtol=1e-9, atol=1e-12)
    ce = sol.y[0]
    return emax * ce / (ec50 + ce)


_popt2, _pcov2 = curve_fit(link_model, t_obs, e_obs, p0=[20.0, 1.0, 0.3],
                           bounds=([0.0, 1e-3, 1e-3], [200.0, 100.0, 10.0]))
emax_hat = float(_popt2[0])
ec50_hat = float(_popt2[1])
ke0_hat = float(_popt2[2])

link_pred = link_model(t_obs, emax_hat, ec50_hat, ke0_hat)
sse_link = float(((e_obs - link_pred) ** 2).sum())

print("link fit: Emax %r ms, EC50 %r mg/L, ke0 %r per h" % (emax_hat, ec50_hat, ke0_hat))
print("SSE     : %r   (direct fit was %r)" % (sse_link, sse_direct))
if ke0_hat is not None:
    print("effect site half life: %.3f h" % (np.log(2) / ke0_hat))
if link_pred is not None:
    _rl = e_obs - np.asarray(link_pred, dtype=float)
    print("mean residual before the peak: %.4f" % _rl[:peak_idx + 1].mean())
    print("mean residual after the peak : %.4f" % _rl[peak_idx + 1:].mean())
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t_obs, e_obs, "o", label="observed")
    ax.plot(t_obs, link_pred, "-", label="effect compartment fit")
    ax.plot(t_obs, direct_emax(c_obs, emax_direct, ec50_direct), "--", label="direct fit")
    ax.set_xlabel("time (h)")
    ax.set_ylabel("change in QTc (ms)")
    ax.set_title("W8D1 two models, one dataset")
    ax.legend()
    plt.show()
```

---

## W8D2. Indirect response: when the biomarker, not the drug, sets the clock

### Part a

```python
# Part a. The turnover model at one dose.

kin = KOUT * R_BASE


def rhs(t, y, dose):
    """dR/dt for inhibition of production."""
    c = conc(t, dose)
    return kin * (1.0 - IMAX * c / (IC50 + c)) - KOUT * y[0]


_sol = solve_ivp(rhs, (float(t_grid[0]), float(t_grid[-1])), [R_BASE],
                 t_eval=t_grid, args=(DOSE,), rtol=1e-9, atol=1e-10)
r_single = _sol.y[0]

_i = int(np.argmin(r_single))
nadir = float(r_single[_i])
t_nadir = float(t_grid[_i])
pct_suppression = float(100.0 * (1.0 - nadir / R_BASE))

print("kin = %r units/L/h" % (kin,))
print("nadir %r units/L at t = %r h" % (nadir, t_nadir))
print("maximum suppression %r pct of baseline" % (pct_suppression,))

if r_single is not None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t_grid, r_single, "-", color="#1f4e79", label="biomarker")
    ax.set_xlabel("time (h)")
    ax.set_ylabel("biomarker (units/L)")
    ax.set_xlim(0, 96)
    ax2 = ax.twinx()
    ax2.plot(t_grid, conc(t_grid, DOSE), "--", color="#999999", label="concentration")
    ax2.set_ylabel("concentration (mg/L)")
    ax.set_title("W8D2 the effect outlasts the drug")
    ax.legend(loc="lower right")
    plt.show()
```

### Part b

```python
# Part b. Dose dependence, and the effect compartment comparison.

_tn, _nd, _ta, _tl = [], [], [], []
for _d in DOSES:
    _s = solve_ivp(rhs, (float(t_grid[0]), float(t_grid[-1])), [R_BASE],
                   t_eval=t_grid, args=(_d,), rtol=1e-9, atol=1e-10)
    _r = _s.y[0]
    _i = int(np.argmin(_r))
    _tn.append(float(t_grid[_i]))
    _nd.append(float(_r[_i]))
    _ta.append(float((conc(t_grid, _d) > IC50).sum()) * DT)

    _sl = solve_ivp(lambda t, y, dd: KE0_COMPARE * (conc(t, dd) - y[0]),
                    (float(t_grid[0]), float(t_grid[-1])), [0.0],
                    t_eval=t_grid, args=(_d,), rtol=1e-9, atol=1e-12)
    _ce = _sl.y[0]
    _e = R_BASE * (1.0 - IMAX * _ce / (IC50 + _ce))
    _tl.append(float(t_grid[int(np.argmin(_e))]))

t_nadir_by_dose = np.array(_tn)
nadir_by_dose = np.array(_nd)
t_above_by_dose = np.array(_ta)
t_link_by_dose = np.array(_tl)

print("dose (mg)        :", DOSES)
print("time above IC50  :", t_above_by_dose)
print("indirect t_nadir :", t_nadir_by_dose)
print("indirect nadir   :", nadir_by_dose)
print("link model t_min :", t_link_by_dose)

if nadir_by_dose is not None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for _d in DOSES:
        _s = solve_ivp(rhs, (float(t_grid[0]), float(t_grid[-1])), [R_BASE],
                       t_eval=t_grid, args=(_d,), rtol=1e-9, atol=1e-10)
        ax.plot(t_grid, _s.y[0], "-", label="%g mg" % _d)
    ax.axhline(R_BASE, ls=":", color="grey")
    ax.set_xlim(0, 120)
    ax.set_xlabel("time (h)")
    ax.set_ylabel("biomarker (units/L)")
    ax.set_title("W8D2 three doses, three different nadir times")
    ax.legend()
    plt.show()
```

### Part c

```python
# Part c. Duration of action, and whose clock it runs on.

_sup = R_BASE - r_single
auec = float(np.sum(0.5 * (_sup[1:] + _sup[:-1]) * np.diff(t_grid)))

_i = int(np.argmin(r_single))
_flags = r_single[_i:] >= 0.9 * R_BASE
t_recover = float(t_grid[_i + int(np.where(_flags)[0][0])])


def simulate(dose, t_half_turnover):
    """Solve the turnover model for a given dose and turnover half life."""
    kout = np.log(2.0) / t_half_turnover
    kin_local = kout * R_BASE

    def local_rhs(t, y):
        c = conc(t, dose)
        return kin_local * (1.0 - IMAX * c / (IC50 + c)) - kout * y[0]

    sol = solve_ivp(local_rhs, (float(t_grid[0]), float(t_grid[-1])), [R_BASE],
                    t_eval=t_grid, rtol=1e-9, atol=1e-10)
    return sol.y[0]


t_nadir_fast = float(t_grid[int(np.argmin(simulate(DOSE, 6.0)))])
t_nadir_slow = float(t_grid[int(np.argmin(simulate(DOSE, 24.0)))])

print("AUEC of suppression: %r units/L*h" % (auec,))
print("back to 90 pct of baseline at %r h" % (t_recover,))
print("")
print("nadir time with a  6 h turnover half life: %r h" % (t_nadir_fast,))
print("nadir time with a 12 h turnover half life: %r h" % (t_nadir,))
print("nadir time with a 24 h turnover half life: %r h" % (t_nadir_slow,))
```

---

## W8D3. A virtual population: simulating variability and reading it back

### Part a

```python
# Part a. Three different centres of one distribution.

cl_gm = float(np.exp(np.mean(np.log(cl_i))))
cl_cv_pct = float(100.0 * np.sqrt(np.exp(np.var(np.log(cl_i), ddof=1)) - 1.0))
cl_mean = float(np.mean(cl_i))
mean_over_median = float(np.mean(cl_i) / np.median(cl_i))
mean_over_median_theory = float(np.exp(OMEGA_CL ** 2 / 2.0))

print("typical clearance in the simulation: %.1f L/h" % CL_POP)
print("geometric mean  : %r L/h" % (cl_gm,))
print("arithmetic mean : %r L/h" % (cl_mean,))
print("CV              : %r pct   (omega %.2f implies %.2f pct)"
      % (cl_cv_pct, OMEGA_CL, 100 * np.sqrt(np.exp(OMEGA_CL ** 2) - 1)))
print("mean / median   : %r observed, %r in theory"
      % (mean_over_median, mean_over_median_theory))
```

### Part b

```python
# Part b. From parameters to exposures.

k_i = cl_i / v_i
auc_i = DOSE / cl_i
tmax_i = np.log(KA / k_i) / (KA - k_i)
cmax_i = DOSE * KA / (v_i * (KA - k_i)) * (np.exp(-k_i * tmax_i) - np.exp(-KA * tmax_i))

auc_gm = float(np.exp(np.mean(np.log(auc_i))))
auc_p5 = float(np.percentile(auc_i, 5))
auc_p95 = float(np.percentile(auc_i, 95))
fold_range = auc_p95 / auc_p5

print("AUC  geometric mean %r mg*h/L   (DOSE / CL_POP = %.4f)" % (auc_gm, DOSE / CL_POP))
print("AUC  5th to 95th percentile: %r to %r mg*h/L" % (auc_p5, auc_p95))
print("AUC  fold range: %r" % (fold_range,))
if cmax_i is not None:
    _p = np.percentile(cmax_i, [5, 95])
    print("Cmax 5th to 95th percentile: %.4f to %.4f mg/L, fold range %.4f"
          % (_p[0], _p[1], _p[1] / _p[0]))

if auc_i is not None and cmax_i is not None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(auc_i, bins=40, color="#1f4e79")
    axes[0].set_xlabel("AUC (mg*h/L)")
    axes[0].set_ylabel("subjects")
    axes[1].hist(cmax_i, bins=40, color="#7f3f00")
    axes[1].set_xlabel("Cmax (mg/L)")
    for _ax, _x in zip(axes, (auc_i, cmax_i)):
        for _q in np.percentile(_x, [5, 95]):
            _ax.axvline(_q, ls=":", color="black")
    fig.suptitle("W8D3 exposure distribution in 500 virtual subjects")
    plt.show()
```

### Part c

```python
# Part c. The sampling distribution of a CV estimate.

rng_c = np.random.default_rng(SEED_C)

draws_12 = rng_c.lognormal(0.0, OMEGA_CL, size=(N_STUDIES, 12))
draws_120 = rng_c.lognormal(0.0, OMEGA_CL, size=(N_STUDIES, 120))

cv_hat_12 = 100.0 * np.sqrt(np.exp(np.var(np.log(draws_12), axis=1, ddof=1)) - 1.0)
cv_hat_120 = 100.0 * np.sqrt(np.exp(np.var(np.log(draws_120), axis=1, ddof=1)) - 1.0)

cv12_lo = float(np.percentile(cv_hat_12, 2.5))
cv12_hi = float(np.percentile(cv_hat_12, 97.5))
cv120_lo = float(np.percentile(cv_hat_120, 2.5))
cv120_hi = float(np.percentile(cv_hat_120, 97.5))

print("true CV: %.2f pct" % (100 * np.sqrt(np.exp(OMEGA_CL ** 2) - 1)))
print("")
print(" 12 subjects: 95 pct of estimates between %r and %r" % (cv12_lo, cv12_hi))
print("120 subjects: 95 pct of estimates between %r and %r" % (cv120_lo, cv120_hi))
if cv_hat_12 is not None and cv_hat_120 is not None:
    print("")
    print("median estimate,  12 subjects: %.3f pct" % np.median(cv_hat_12))
    print("median estimate, 120 subjects: %.3f pct" % np.median(cv_hat_120))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(cv_hat_12, bins=50, alpha=0.6, label="studies of 12")
    ax.hist(cv_hat_120, bins=50, alpha=0.6, label="studies of 120")
    ax.axvline(100 * np.sqrt(np.exp(OMEGA_CL ** 2) - 1), color="black", ls="--",
               label="truth")
    ax.set_xlabel("estimated between subject CV (pct)")
    ax.set_ylabel("studies")
    ax.set_title("W8D3 what a study would have told you")
    ax.legend()
    plt.show()
```

---

## W8D4. Building a NONMEM ready dataset, and auditing it

### Part a

```python
# Part a. Build the dataset.

dose_rows = dosing.assign(TIME=0.0, DV=0.0, EVID=1, MDV=1, CMT=1)[NM_COLS]
obs_rows = samples.assign(AMT=0.0, DV=samples["CONC"], EVID=0, CMT=2,
                          MDV=np.where(samples["CONC"] < LLOQ, 1, 0))[NM_COLS]

nm = (pd.concat([dose_rows, obs_rows], ignore_index=True)
      .sort_values(["ID", "TIME", "EVID"], ascending=[True, True, False])
      .reset_index(drop=True))

n_rows = int(len(nm))
n_dose_rows = int((nm["EVID"] == 1).sum())
first_evid_per_id = nm.groupby("ID")["EVID"].first().values

print("rows: %r, of which dosing records: %r" % (n_rows, n_dose_rows))
print("first EVID per subject:", first_evid_per_id)
if nm is not None:
    print("")
    print(nm.head(11).to_string(index=False))
```

### Part b

```python
# Part b. Audit the dataset.

n_blq = int(((nm["EVID"] == 0) & (nm["MDV"] == 1)).sum())
n_mdv = int((nm["MDV"] == 1).sum())
records_per_id = nm.groupby("ID").size().values
time_monotone = bool(nm.groupby("ID")["TIME"]
                     .apply(lambda s: s.is_monotonic_increasing).all())

print("observation rows below the LLOQ: %r" % (n_blq,))
print("rows with MDV = 1             : %r" % (n_mdv,))
print("rows per subject              :", records_per_id)
print("TIME non decreasing in every subject:", time_monotone)
if nm is not None:
    print("")
    print("the BLQ observation rows:")
    print(nm[(nm["EVID"] == 0) & (nm["MDV"] == 1)].to_string(index=False))
```

### Part c

```python
# Part c. NCA per subject, and the dose groups compared.

def nca_one(g):
    """CMAX, TMAX and linear trapezoid AUC for one subject."""
    g = g.sort_values("TIME")
    t = g["TIME"].values
    c = g["DV"].values
    i = int(np.argmax(c))
    area = float(np.sum(0.5 * (c[1:] + c[:-1]) * np.diff(t)))
    return pd.Series({"CMAX": float(c[i]), "TMAX": float(t[i]), "AUC": area})


_obs = nm[nm["EVID"] == 0]
nca = _obs.groupby("ID")[["TIME", "DV"]].apply(nca_one).reset_index()
nca = (nca.merge(dosing, on="ID", validate="one_to_one")
          .merge(demog, on="ID", validate="one_to_one"))
nca["AUC_DN"] = nca["AUC"] / nca["AMT"]

_low = nca.loc[nca["AMT"] == 50.0, "AUC_DN"].values
_high = nca.loc[nca["AMT"] == 150.0, "AUC_DN"].values
auc_dn_low = float(np.exp(np.mean(np.log(_low))))
auc_dn_high = float(np.exp(np.mean(np.log(_high))))
auc_dn_ratio = auc_dn_high / auc_dn_low

if nca is not None:
    print(nca.to_string(index=False))
print("")
print("dose normalised AUC, geometric mean")
print("  50 mg group : %r" % (auc_dn_low,))
print(" 150 mg group : %r" % (auc_dn_high,))
print("  ratio       : %r" % (auc_dn_ratio,))
```

---

## W8D5. Capstone: choosing the phase 2 dose

### Part a

```python
# Part a. One typical subject, at steady state.

k_typ = CL_POP / V_POP
c_typ = (DOSE_MID * KA / (V_POP * (KA - k_typ))
         * (np.exp(-k_typ * t_grid) / (1.0 - np.exp(-k_typ * TAU))
            - np.exp(-KA * t_grid) / (1.0 - np.exp(-KA * TAU))))

cmax_ss = float(c_typ.max())
cmin_ss = float(c_typ.min())
cavg_ss = float(np.sum(0.5 * (c_typ[1:] + c_typ[:-1]) * np.diff(t_grid)) / TAU)
cavg_identity = float(DOSE_MID / (CL_POP * TAU))

print("Cmax %r mg/L, Cmin %r mg/L" % (cmax_ss, cmin_ss))
if cmax_ss is not None and cmin_ss is not None:
    print("peak to trough ratio %.4f" % (cmax_ss / cmin_ss))
print("Cavg by integration %r mg/L" % (cavg_ss,))
print("Cavg from DOSE / (CL * tau) %r mg/L" % (cavg_identity,))

if c_typ is not None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t_grid, c_typ, "-", color="#1f4e79")
    ax.axhline(CMAX_LIMIT, ls="--", color="#aa0000", label="safety limit")
    ax.axhline(IC50_POP, ls=":", color="grey", label="typical IC50")
    if cavg_ss is not None:
        ax.axhline(cavg_ss, ls="-.", color="#007700", label="Cavg")
    ax.set_xlabel("time within the dosing interval (h)")
    ax.set_ylabel("concentration (mg/L)")
    ax.set_title("W8D5 steady state, typical subject, %g mg daily" % DOSE_MID)
    ax.legend()
    plt.show()
```

### Part b

```python
# Part b. Four hundred patients at the middle dose.

def population_metrics(dose):
    """Steady state Cmax and average inhibition for every patient."""
    k = cl_i / v_i
    coef = dose * KA / (v_i * (KA - k))
    c = coef[:, None] * (np.exp(-k[:, None] * t_grid[None, :])
                         / (1.0 - np.exp(-k[:, None] * TAU))
                         - np.exp(-KA * t_grid[None, :])
                         / (1.0 - np.exp(-KA * TAU)))
    cmax = c.max(axis=1)
    inh = IMAX * c / (ic50_i[:, None] + c)
    area = np.sum(0.5 * (inh[:, 1:] + inh[:, :-1]) * np.diff(t_grid)[None, :], axis=1)
    return cmax, area / TAU


cmax_i, mean_inh_i = population_metrics(DOSE_MID)

p_eff = float((mean_inh_i >= EFF_TARGET).mean())
p_tox = float((cmax_i > CMAX_LIMIT).mean())
p_success = float(((mean_inh_i >= EFF_TARGET) & (cmax_i <= CMAX_LIMIT)).mean())

print("at %g mg daily:" % DOSE_MID)
print("  efficacy criterion met : %r" % (p_eff,))
print("  above the safety limit : %r" % (p_tox,))
print("  both at once           : %r" % (p_success,))
if p_eff is not None and p_tox is not None:
    print("  product of the margins : %.4f  <- not the same thing"
          % (p_eff * (1.0 - p_tox)))
if cmax_i is not None:
    print("")
    print("  Cmax   5th, 50th, 95th percentile:", np.round(np.percentile(cmax_i, [5, 50, 95]), 4))
    print("  mean inhibition, same percentiles:", np.round(np.percentile(mean_inh_i, [5, 50, 95]), 4))
```

### Part c

```python
# Part c. The dose response of the decision.

_eff, _tox, _suc = [], [], []
for _d in DOSES:
    _cmax, _inh = population_metrics(_d)
    _eff.append(float((_inh >= EFF_TARGET).mean()))
    _tox.append(float((_cmax > CMAX_LIMIT).mean()))
    _suc.append(float(((_inh >= EFF_TARGET) & (_cmax <= CMAX_LIMIT)).mean()))

p_eff_by_dose = np.array(_eff)
p_tox_by_dose = np.array(_tox)
p_success_by_dose = np.array(_suc)

best_dose = float(DOSES[int(np.argmax(p_success_by_dose))])

if p_success_by_dose is not None:
    print(" dose   efficacy   over limit   both")
    for _i, _d in enumerate(DOSES):
        print("%5.0f %9.3f %12.3f %7.3f"
              % (_d, p_eff_by_dose[_i], p_tox_by_dose[_i], p_success_by_dose[_i]))
print("")
print("recommended dose: %r mg once daily" % (best_dose,))

if p_success_by_dose is not None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(DOSES, p_eff_by_dose, "o-", label="efficacy criterion met")
    ax.plot(DOSES, p_tox_by_dose, "s-", label="above the safety limit")
    ax.plot(DOSES, p_success_by_dose, "^-", lw=2, color="black", label="both")
    ax.set_xscale("log")
    ax.set_xticks(DOSES)
    ax.set_xticklabels(["%g" % _d for _d in DOSES])
    ax.set_xlabel("daily dose (mg)")
    ax.set_ylabel("fraction of patients")
    ax.set_title("W8D5 dose selection in 400 virtual patients")
    ax.legend()
    plt.show()
```
