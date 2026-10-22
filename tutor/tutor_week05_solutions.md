# Week 5 reference solutions

One way to do each part. The checks test the numbers, not the route, so a
different correct answer is still correct.

Do not paste these to the student. Use them to work out where their
version diverges, and say the smallest thing that gets them unstuck.

---

## W5D1. Predicting clearance: the train/test split and why it exists

### Part a

```python
# Part a. Split off a test set, reproducibly.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)
n_train = int(len(X_train))
n_test = int(len(X_test))

print("n_train:", n_train, "  n_test:", n_test)
print("X_train shape:", None if X_train is None else X_train.shape)
print("X_test shape :", None if X_test is None else X_test.shape)
```

### Part b

```python
# Part b. Fit on train, score on test.

model = LinearRegression().fit(X_train, y_train)
pred_test = model.predict(X_test)
r2_test = float(r2_score(y_test, pred_test))
rmse_test = float(np.sqrt(mean_squared_error(y_test, pred_test)))

print("R2 on test   : %s" % r2_test)
print("RMSE on test : %s L/h" % rmse_test)
if model is not None:
    print("coefficients :", dict(zip(FEATURES, np.round(model.coef_, 4))))
```

### Part c

```python
# Part c. How much better does it look on its own training data.

_pred_train = model.predict(X_train)
r2_train = float(r2_score(y_train, _pred_train))
rmse_train = float(np.sqrt(mean_squared_error(y_train, _pred_train)))
optimism = float(r2_train - r2_test)

print("R2   train %s  vs test %s" % (r2_train, r2_test))
print("RMSE train %s  vs test %s" % (rmse_train, rmse_test))
print("optimism   %s" % optimism)
```

---

## W5D2. Cross-validation, and the leak that invents a result from noise

### Part a

```python
# Part a. Cross-validate on the two real covariates.

_kf = KFold(n_splits=5, shuffle=True, random_state=1)
cv_scores = cross_val_score(LinearRegression(), X_real, y, cv=_kf, scoring="r2")
cv_mean = float(cv_scores.mean())
cv_sd = float(cv_scores.std(ddof=1))

print("fold scores:", None if cv_scores is None else np.round(cv_scores, 4))
print("mean R2    :", cv_mean)
print("SD across folds:", cv_sd)
```

### Part b

```python
# Part b. Screen on all the data first, then cross-validate. The common mistake.

_selector = SelectKBest(f_regression, k=K_SELECT).fit(X_panel, y)
X_leaky = X_panel.loc[:, _selector.get_support()]
cv_leaky_mean = float(cross_val_score(LinearRegression(), X_leaky, y,
                                      cv=_kf, scoring="r2").mean())

print("columns chosen:", None if X_leaky is None else list(X_leaky.columns))
print("mean R2, selected first:", cv_leaky_mean)
print("mean R2, real covariates:", cv_mean)
```

### Part c

```python
# Part c. Selection inside the pipeline, so each fold is blind to its own test data.

pipe = Pipeline([("sel", SelectKBest(f_regression, k=K_SELECT)),
                 ("model", LinearRegression())])
cv_clean_mean = float(cross_val_score(pipe, X_panel, y, cv=_kf, scoring="r2").mean())
leak_inflation = float(cv_leaky_mean - cv_clean_mean)

print("mean R2, selection leaked  : %s" % cv_leaky_mean)
print("mean R2, selection in pipe : %s" % cv_clean_mean)
print("inflation caused by the leak: %s" % leak_inflation)
print("")
print("for reference, the real covariates gave: %s" % cv_mean)
```

---

## W5D3. Random forests, and two different answers to which feature matters

### Part a

```python
# Part a. Fit a random forest and compare train against test.

rf = RandomForestRegressor(n_estimators=300, random_state=0).fit(X_train, y_train)
r2_test_rf = float(r2_score(y_test, rf.predict(X_test)))
r2_train_rf = float(r2_score(y_train, rf.predict(X_train)))
gap_rf = float(r2_train_rf - r2_test_rf)

print("R2 test  : %s" % r2_test_rf)
print("R2 train : %s" % r2_train_rf)
print("gap      : %s   (day 1 linear model: -0.046)" % gap_rf)
```

### Part b

```python
# Part b. Impurity importance, sorted.

imp_impurity = pd.Series(rf.feature_importances_,
                         index=FEATURES).sort_values(ascending=False)

print(None if imp_impurity is None else imp_impurity.round(4).to_string())
print("")
print("sums to:", None if imp_impurity is None else round(float(imp_impurity.sum()), 6))
```

### Part c

```python
# Part c. Permutation importance on held out data.

perm = permutation_importance(rf, X_test, y_test, n_repeats=20, random_state=0)
imp_perm = pd.Series(perm.importances_mean, index=FEATURES).sort_values(ascending=False)
noise_rank_impurity = int(list(imp_impurity.index).index(NOISE_FEATURE) + 1)
noise_rank_perm = int(list(imp_perm.index).index(NOISE_FEATURE) + 1)

print(None if imp_perm is None else imp_perm.round(4).to_string())
print("")
print("%s ranks %s by impurity, %s by permutation (of %d features)"
      % (NOISE_FEATURE, noise_rank_impurity, noise_rank_perm, len(FEATURES)))
```

---

## W5D4. Elastic net: why you must standardise, and what the penalty buys

### Part a

```python
# Part a. Elastic net with and without standardising.

_raw_model = ElasticNet(alpha=0.05, l1_ratio=L1_RATIO, random_state=0).fit(X_train, y_train)
coef_raw = _raw_model.coef_

_scaled_model = make_pipeline(
    StandardScaler(), ElasticNet(alpha=0.05, l1_ratio=L1_RATIO, random_state=0))
_scaled_model.fit(X_train, y_train)
coef_scaled = _scaled_model[-1].coef_

n_zero_raw = int((coef_raw == 0).sum())
n_zero_scaled = int((coef_scaled == 0).sum())

print("zeros, raw    :", n_zero_raw, "of", len(FEATURES))
print("zeros, scaled :", n_zero_scaled, "of", len(FEATURES))
if coef_raw is not None:
    print("")
    print(pd.DataFrame({"raw": coef_raw, "scaled": coef_scaled},
                       index=FEATURES).head(8).round(6).to_string())
```

### Part b

```python
# Part b. The regularisation path.

rows = []
for a in ALPHAS:
    m = make_pipeline(StandardScaler(),
                      ElasticNet(alpha=a, l1_ratio=L1_RATIO, random_state=0))
    m.fit(X_train, y_train)
    rows.append({
        "ALPHA": float(a),
        "N_NONZERO": int((m[-1].coef_ != 0).sum()),
        "R2_TRAIN": float(r2_score(y_train, m.predict(X_train))),
        "R2_TEST": float(r2_score(y_test, m.predict(X_test))),
    })

path = pd.DataFrame(rows)

print(None if path is None else path.round(4).to_string(index=False))
```

### Part c

```python
# Part c. ElasticNetCV picks alpha using the training data only.

encv = make_pipeline(StandardScaler(),
                     ElasticNetCV(l1_ratio=L1_RATIO, cv=5, random_state=0))
encv.fit(X_train, y_train)

best_alpha = float(encv[-1].alpha_)
n_nonzero_cv = int((encv[-1].coef_ != 0).sum())
r2_cv_test = float(r2_score(y_test, encv.predict(X_test)))

print("alpha chosen   :", best_alpha)
print("features kept  :", n_nonzero_cv, "of", len(FEATURES))
print("R2 on test     :", r2_cv_test)
if encv is not None:
    kept = pd.Series(encv[-1].coef_, index=FEATURES)
    print("")
    print("surviving coefficients:")
    print(kept[kept != 0].sort_values(key=abs, ascending=False).round(4).to_string())
```

---

## W5D5. PCA on a correlated covariate panel

### Part a

```python
# Part a. PCA on the standardised panel, and what happens without standardising.

Z = StandardScaler().fit_transform(panel)
pca = PCA().fit(Z)
evr = pca.explained_variance_ratio_
evr_unscaled_first = float(PCA().fit(panel).explained_variance_ratio_[0])

print("explained variance ratio:", None if evr is None else np.round(evr, 4))
print("sums to                 :", None if evr is None else round(float(np.sum(evr)), 6))
print("")
print("PC1 share, standardised :", None if evr is None else round(float(evr[0]), 4))
print("PC1 share, raw          :", evr_unscaled_first)
```

### Part b

```python
# Part b. How many components carry 90 percent, and the projection.

cumulative = np.cumsum(evr)
n_for_90 = int(np.argmax(cumulative >= 0.90) + 1)
scores = pca.transform(Z)

print("cumulative:", None if cumulative is None else np.round(cumulative, 4))
print("components needed for 90 percent:", n_for_90)
print("scores shape:", None if scores is None else scores.shape)

if scores is not None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name in group.unique():
        sel = (group == name).to_numpy()
        ax.scatter(scores[sel, 0], scores[sel, 1], s=28, alpha=0.75, label=name)
    ax.set_xlabel("PC1 (%.1f%% of variance)" % (100 * evr[0]))
    ax.set_ylabel("PC2 (%.1f%% of variance)" % (100 * evr[1]))
    ax.set_title("Subjects in the first two components")
    ax.legend(frameon=False)
    plt.show()
```

### Part c

```python
# Part c. What are PC1 and PC2 made of.

loadings = pd.DataFrame(pca.components_, columns=FEATURES,
                        index=["PC%d" % (i + 1) for i in range(pca.components_.shape[0])])
pc1_top3 = list(loadings.loc["PC1"].abs().sort_values(ascending=False).head(3).index)
pc2_top3 = list(loadings.loc["PC2"].abs().sort_values(ascending=False).head(3).index)

print("PC1 is dominated by:", pc1_top3)
print("PC2 is dominated by:", pc2_top3)
if loadings is not None:
    print("")
    print(loadings.head(3).round(3).to_string())
```
