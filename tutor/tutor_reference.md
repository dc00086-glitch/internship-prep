# Function reference, error messages and house rules

The same content as reference/function_reference.docx, which the student
also has. Quote the watch out line when it is the cause of their problem.

## Python

### len

- signature: `len(x)`
- what it does: How many items are in a list, array, DataFrame or string.
- example: `len(times)  gives  5`
- watch out: On a DataFrame this counts rows, not cells. Print it before and after every merge.

### round

- signature: `round(x, n)`
- what it does: Rounds a number to n decimal places, for printing.
- example: `round(7.48219, 2)  gives  7.48`
- watch out: Round for display only. Never round the numbers you then do more maths on, or the error compounds.

### f-string

- signature: `f"text {value}"`
- what it does: Puts values inside text. The part in braces is evaluated.
- example: `f"half life {t_half:.2f} h"  gives  half life 7.48 h`
- watch out: The :.2f inside the braces formats to 2 decimals without changing the underlying value.

### type

- signature: `type(x)`
- what it does: What kind of thing x is. The fastest way to find out why something is behaving oddly.
- example: `type(conc)  gives  numpy.ndarray`
- watch out: If you expected an array and got a list, or expected a DataFrame and got a Series, this is how you find out.

### def

- signature: `def name(args):`
- what it does: Defines a function, so you can write something once and reuse it.
- example: `def t_half(k):     return np.log(2) / k`
- watch out: Everything inside the function must be indented. The indentation is the syntax, not decoration.

### try/except

- signature: `try: ... except ValueError as err:`
- what it does: Runs code, and catches a specific error instead of stopping.
- example: `try:     risky() except ValueError as err:     print(err)`
- watch out: Catch the specific error you expect. A bare except hides real bugs.

### sorted

- signature: `sorted(x)`
- what it does: Returns a new sorted list. Works on the result of .unique(), which comes back in whatever order pandas found the values.
- example: `sorted(df["SUBJECT"].unique())`
- watch out: Sorting text compares character by character, so '10' lands before '9'. Convert to numbers first when the order has to be numeric.

## numpy

### np.array

- signature: `np.array([1.0, 2.0, 3.0])`
- what it does: Makes an array from a list. An array does maths on every entry at once, which a plain list cannot.
- example: `np.array([1, 2, 3]) * 2  gives  [2 4 6]`
- watch out: A list would give [1, 2, 3, 1, 2, 3] instead. That difference is the whole reason numpy exists.

### .shape

- signature: `a.shape`
- what it does: The size of an array, as (rows, columns) for 2D or (length,) for 1D.
- example: `conc.shape  gives  (8, 9)`
- watch out: The first thing to print when maths will not line up. Most numpy errors are shape errors wearing a disguise.

### .size

- signature: `a.size`
- what it does: The total number of entries in an array.
- example: `time_h.size  gives  11`
- watch out: On a 2D array this is rows times columns, not either one on its own.

### np.zeros

- signature: `np.zeros(4)`
- what it does: An array of zeros of the length you ask for. Useful as a starting point.
- example: `np.zeros(4)  gives  [0. 0. 0. 0.]`
- watch out: 

### np.arange

- signature: `np.arange(0, 5)`
- what it does: Evenly spaced whole numbers, like range but as an array.
- example: `np.arange(1, 5)  gives  [1 2 3 4]`
- watch out: The end value is not included, same as Python slicing.

### np.linspace

- signature: `np.linspace(0, 1, 5)`
- what it does: A set number of evenly spaced values between two endpoints. The usual way to make a smooth time grid for plotting a fitted curve.
- example: `np.linspace(0, 1, 5)  gives  [0. 0.25 0.5 0.75 1.]`
- watch out: Unlike arange, the end value IS included, and you say how many points you want rather than the step size.

### np.isnan

- signature: `np.isnan(a)`
- what it does: Flags the missing values. Gives a True/False array, True where the entry is nan.
- example: `np.isnan(conc)  gives  [False False True ...]`
- watch out: True means MISSING. To keep the usable ones you almost always want ~np.isnan(a).

### ~

- signature: `~mask`
- what it does: Flips a True/False array. Every True becomes False and back.
- example: `~np.isnan(conc)  gives  True where the sample IS usable`
- watch out: This is for arrays. The word not does not work on an array, because numpy cannot reduce a whole array to one True or False.

### boolean indexing

- signature: `a[mask]`
- what it does: Keeps only the entries where the mask is True. The main way filtering is done in numpy.
- example: `c_meas = conc[~np.isnan(conc)]`
- watch out: The mask must be the same length as the array. A mask built from one array works on another, which is how you get the times that go with the measurable concentrations.

### np.sum

- signature: `np.sum(a)`
- what it does: Adds everything up. On a True/False array it counts the Trues, because True counts as 1.
- example: `np.sum(np.isnan(conc))  gives  2`
- watch out: Wrap it in int() when you want a plain whole number rather than a numpy integer.

### np.max / np.min

- signature: `np.max(a)`
- what it does: The largest or smallest value in an array. Cmax is np.max of the measurable concentrations.
- example: `np.max(c_meas)  gives  2.602`
- watch out: If any entry is nan the answer is nan. Either mask them out first, or use np.nanmax.

### np.argmax

- signature: `np.argmax(a)`
- what it does: The POSITION of the largest value, counting from 0. This is how Tmax is found: get the position of Cmax, then look that position up in the time array.
- example: `t_meas[np.argmax(c_meas)]  gives  3.0`
- watch out: It returns a position, not a value. Reporting the position as if it were a time is a very common slip.

### np.nanmax

- signature: `np.nanmax(a)`
- what it does: Like np.max but ignores nan. There is a nan version of most summaries: nanmean, nanstd, nansum.
- example: `np.nanmax(conc)  ignores the BLQ samples`
- watch out: Convenient, but masking first is usually clearer, because then every later step is working on the same clean data.

### np.mean

- signature: `a.mean(axis=0)`
- what it does: The average. On a 2D array, axis says which direction to collapse.
- example: `conc.mean(axis=0)  gives one value per timepoint`
- watch out: The axis you name is the one that DISAPPEARS. axis=0 leaves one value per column, axis=1 leaves one value per row.

### np.median

- signature: `np.median(a)`
- what it does: The middle value. Less sensitive to one extreme point than the mean.
- example: `np.median(vals)  gives  15.5`
- watch out: Report whichever you decided on in advance. Switching to the one that looks better after seeing the data is not an analysis choice.

### np.std

- signature: `a.std(axis=0, ddof=1)`
- what it does: The standard deviation, which is the spread of the data.
- example: `conc.std(axis=0, ddof=1)`
- watch out: numpy divides by n unless you pass ddof=1, which divides by n-1 and is the SAMPLE SD you almost always want. Pandas does the opposite and already uses n-1. This inconsistency causes quietly wrong numbers.

### keepdims

- signature: `a.max(axis=1, keepdims=True)`
- what it does: Keeps the collapsed direction as a length 1 axis, so the result still lines up with the original array for broadcasting.
- example: `conc / conc.max(axis=1, keepdims=True)`
- watch out: Without it you get shape (8,) which will not divide into (8, 9). This is the fix for the broadcast error below.

### .reshape

- signature: `a.reshape(-1, 1)`
- what it does: Changes the shape without changing the data. -1 means work it out.
- example: `row_max.reshape(-1, 1)  turns (8,) into (8, 1)`
- watch out: Does the same job as keepdims here. Use whichever reads more clearly.

### broadcasting

- signature: `a / b`
- what it does: Doing maths between arrays of different shapes. numpy compares shapes from the RIGHT and stretches length 1 axes to fit.
- example: `(8, 9) and (9,) line up. (8, 9) and (8,) do not.`
- watch out: The error operands could not be broadcast together means the shapes failed that check. Usually the fix is keepdims=True.

### np.log

- signature: `np.log(a)`
- what it does: The natural log, ln. Turns an exponential decay into a straight line, which is how terminal slopes are fitted.
- example: `np.log(c_term)`
- watch out: np.log is ln. np.log10 is base 10. Mixing them up changes lambda_z by a factor of 2.303.

### np.sqrt

- signature: `np.sqrt(a)`
- what it does: The square root.
- example: `sem = sd / np.sqrt(n)`
- watch out: If you find yourself writing this when making an error bar, stop. That is the SEM, and this repo plots SD.

### np.polyfit

- signature: `np.polyfit(x, y, deg=1)`
- what it does: Fits a polynomial. With deg=1 it is a straight line, and it returns [slope, intercept] in that order.
- example: `slope, intercept = np.polyfit(t_term, np.log(c_term), deg=1)`
- watch out: lambda_z is the NEGATIVE of the slope, because the slope is negative when the drug is leaving. Fit only the terminal points: including the absorption phase biases the slope badly.

### np.round

- signature: `np.round(a, 3)`
- what it does: Rounds every entry, for printing.
- example: `np.round(cv_by_time, 1)`
- watch out: For display only, same as the built-in round.

### np.where

- signature: `np.where(cond, a, b)`
- what it does: Picks from a where the condition is True and from b where it is False. Also useful for turning labels into x positions.
- example: `np.where(df["GROUP"] == "Drug", 0, 1)`
- watch out: With one argument it does something different, returning the positions where the condition is True.

### np.repeat / np.tile

- signature: `np.repeat(a, 3)`
- what it does: Builds up long columns. repeat stretches each element in place, tile repeats the whole sequence.
- example: `np.repeat([1, 2], 2) gives [1 1 2 2], np.tile([1, 2], 2) gives [1 2 1 2]`
- watch out: The usual way to build a long format table: repeat the subject ids, tile the timepoints.

### np.random.default_rng

- signature: `rng = np.random.default_rng(1101)`
- what it does: A random number generator you can reproduce. The number is the seed, and the same seed always gives the same numbers.
- example: `rng.normal(0, 1, 4), rng.lognormal(0, 0.15, 8)`
- watch out: Every problem notebook seeds this, which is why your numbers match the checker. The older np.random.seed style still works but this is the current way.

### np.exp

- signature: `np.exp(x)`
- what it does: The exponential, which undoes np.log. Together they are how you move between the raw scale and the log scale.
- example: `geo_mean = np.exp(np.mean(np.log(auc)))`
- watch out: np.exp undoes np.log. If you used np.log10 you have to undo it with 10 ** x instead, and mixing the pair up changes the answer.

### np.logspace

- signature: `np.logspace(-2, 2, 10)`
- what it does: Values evenly spaced on a LOG scale, which is how concentration ranges in a dose response are laid out.
- example: `np.logspace(-2, 2, 10) gives 0.01 up to 100`
- watch out: The arguments are the exponents, not the values. np.linspace would bunch almost every point at the top of the range.

### np.diag

- signature: `np.diag(m)`
- what it does: The diagonal of a matrix. On a covariance matrix that is the variance of each parameter, so its square root is the standard error.
- example: `perr = np.sqrt(np.diag(pcov))`
- watch out: Given a 1D array instead it does the opposite and builds a diagonal matrix, which is occasionally a confusing surprise.

### argument unpacking

- signature: `f(x, *popt)`
- what it does: Spreads an array into a function's separate arguments, so the same call works whatever the number of parameters.
- example: `pred = decay(times, *popt)`
- watch out: Without the star you pass the whole array as the first parameter, which gives either a shape error or silent broadcasting.

### np.cumsum

- signature: `np.cumsum(x)`
- what it does: Running total. The usual way to turn per component variance shares into a cumulative curve.
- example: `cumulative = np.cumsum(evr)`
- watch out: Pair it with np.argmax on a comparison to find the first point crossing a level: np.argmax(cumulative >= 0.90) + 1.

### np.diff

- signature: `np.diff(a)`
- what it does: The gaps between consecutive entries. On n values it returns n-1 of them.
- example: `np.diff([0, 0.5, 1, 2])  gives  [0.5, 0.5, 1.0]`
- watch out: The result is one element SHORTER than the input, which is what makes a trapezoid sum line up: n-1 widths for n points. Index errors here are almost always an off by one between a[1:] and a[:-1].

### np.errstate

- signature: `with np.errstate(divide='ignore'):`
- what it does: Silences a numpy warning inside the block, for a case where you know the warning is harmless.
- example: `with np.errstate(divide='ignore'): logc = np.log(conc)`
- watch out: Use it only where you have already handled the bad values, for example a log of zero you are about to mask out. Silencing a warning you have not understood is how a NaN gets into a final number.

### np.interp

- signature: `np.interp(x_new, x, y)`
- what it does: Straight line interpolation: reads y at x_new from the points you give it.
- example: `np.interp(1.2, [1.0, 2.0], [10.0, 20.0])  gives  12.0`
- watch out: It REQUIRES x to be increasing and does not check. Hand it a falling concentration curve and you get numbers back, silently wrong ones. Reverse both arrays together first.

### np.percentile

- signature: `np.percentile(a, [5, 50, 95])`
- what it does: The value below which that percentage of the data falls. The honest summary of a skewed distribution.
- example: `np.percentile(auc, [5, 95])  gives  [14.9, 40.5]`
- watch out: For a skewed distribution the mean plus or minus two SD can fall below the 2.5th percentile, or below zero. Percentiles cannot.

### np.argmax on booleans

- signature: `int(np.argmax(flags))`
- what it does: The position of the FIRST True in a boolean array, because True counts as 1 and argmax stops at the first maximum.
- example: `np.argmax(np.isnan(losses))  gives  5`
- watch out: It returns 0 when nothing is True, which is indistinguishable from a True in position 0. Always pair it with flags.any().

### np.where first index

- signature: `int(np.where(flags)[0][0])`
- what it does: The index of the first True, the explicit way. np.where returns a tuple of arrays, hence the two subscripts.
- example: `np.where(r >= 0.9 * baseline)[0][0]  gives  543`
- watch out: It raises IndexError when nothing matches, which is better than argmax quietly returning 0. Check the size first, or catch it.

### rng.lognormal

- signature: `rng.lognormal(mean_of_logs, sd_of_logs, size=n)`
- what it does: Draws from a lognormal distribution: the LOGS are normal with that mean and SD.
- example: `cl = 4.2 * rng.lognormal(0.0, 0.30, size=500)`
- watch out: Both arguments describe the logs, so pass 0.0 as the mean and multiply by the typical value outside. Passing 4.2 as the first argument gives a median of exp(4.2), which is 67, and no error.

### None axis indexing

- signature: `a[:, None], t[None, :]`
- what it does: Inserts a length one axis so two arrays broadcast into a grid. a[:, None] * t[None, :] on n and m values gives an n by m matrix.
- example: `conc = coef[:, None] * np.exp(-k[:, None] * t[None, :])`
- watch out: Getting the axes the wrong way round does not raise, it gives you a square matrix of nonsense when n and m happen to differ, or silently the wrong orientation when they do not. Print .shape.

### boolean mean as a fraction

- signature: `(x >= target).mean()`
- what it does: The fraction of entries satisfying a condition, because the mean of True and False is the proportion of True.
- example: `(auc >= 20.0).mean()  gives  0.72`
- watch out: Combine conditions with & and brackets, never with the word and, which tries to turn a whole array into one True or False and raises.

## pandas

### pd.DataFrame

- signature: `pd.DataFrame({"A": [1, 2]})`
- what it does: Makes a table from a dictionary of columns. Each column has a name and its own type.
- example: `pd.DataFrame({"ANIMAL": ids, "CONC": vals})`
- watch out: A DataFrame is really a dictionary of columns, not a grid of cells. That explains most of its behaviour.

### .head

- signature: `df.head(5)`
- what it does: The first few rows. The first thing to run on any table you did not make yourself.
- example: `pk.head(10)`
- watch out: 

### .columns

- signature: `df.columns`
- what it does: The column names. Check here first when you get a KeyError.
- example: `list(pk.columns)`
- watch out: Column names are case sensitive, and a trailing space in a name is invisible and will bite you.

### selecting a column

- signature: `df["CONC"]`
- what it does: One column, which comes back as a Series. Maths on it is elementwise, like numpy.
- example: `pk["CONC_NG_ML"].mean()`
- watch out: Double brackets, df[["A", "B"]], gives a DataFrame of several columns. Single brackets give one Series. The difference matters downstream.

### boolean filtering

- signature: `df[df["TIME_H"] > 0]`
- what it does: Keeps the rows where the condition is True.
- example: `df_post = pk[pk["TIME_H"] > 0]`
- watch out: Combine conditions with & for and, | for or, and put brackets around each one. The words and/or do not work here.

### adding a column

- signature: `df["NEW"] = df["A"] / df["B"]`
- what it does: Creates a column from other columns, elementwise.
- example: `m["DOSE_MG_PER_KG"] = m["DOSE_MG"] / (m["BODYWEIGHT_G"] / 1000)`
- watch out: Work on a .copy() if the original table still matters, or you may get a SettingWithCopyWarning.

### .nunique / .unique

- signature: `df["ANIMAL"].nunique()`
- what it does: How many distinct values, and what they are.
- example: `pk["ANIMAL"].nunique()  gives  12`
- watch out: Counting rows is not counting animals. Each animal appears once per timepoint.

### .isna

- signature: `df["COL"].isna().sum()`
- what it does: Flags the missing entries, and summing counts them.
- example: `int(merged["BODYWEIGHT_G"].isna().sum())`
- watch out: Run this after every left merge. It is how you find out what did not match.

### .groupby

- signature: `df.groupby(["GROUP", "TIME_H"])`
- what it does: Split, apply, combine. Splits the table into one chunk per group, so you can summarise each chunk.
- example: `df_post.groupby(["GROUP", "TIME_H"])["CONC"].agg(["mean", "std"])`
- watch out: The grouping keys come back in the INDEX, not as columns. Follow with .reset_index() or downstream code will not find them.

### .agg

- signature: `.agg(["mean", "std"])`
- what it does: The summaries to compute on each group. Useful ones: mean, std, size, count, min, max, median, sum.
- example: `.agg(MEAN=("VAL", "mean"), N=("VAL", "size"))`
- watch out: The named form on the right also names the output columns, which saves a rename. size counts every row, count skips the missing ones.

### .reset_index

- signature: `.reset_index()`
- what it does: Moves index levels back out into ordinary columns.
- example: `.groupby(...).agg(...).reset_index()`
- watch out: Use .reset_index(drop=True) after sorting, to renumber 0, 1, 2 and throw the old numbers away rather than keeping them as a column.

### .rename

- signature: `.rename(columns={"mean": "CONC_MEAN"})`
- what it does: Renames columns.
- example: `.rename(columns={"mean": "CONC_MEAN", "std": "CONC_SD"})`
- watch out: 

### .sort_values

- signature: `.sort_values(["ANIMAL", "TIME_H"])`
- what it does: Sorts by one or more columns, in the order you list them.
- example: `.sort_values(["ANIMAL", "TIME_H"]).reset_index(drop=True)`
- watch out: Sort on the NUMBER, not the text. Text sorting puts T_24 before T_4.

### pd.merge

- signature: `pd.merge(a, b, on="ANIMAL", how="left")`
- what it does: Joins two tables on a shared column.
- example: `pd.merge(pk, demo, on="ANIMAL", how="left")`
- watch out: how='left' keeps every row of the left table and shows gaps as NaN. how='inner' is the DEFAULT and silently drops rows that did not match. Print len() before and after, every time. Rows can also go up, if the right table has more than one row per key.

### .melt

- signature: `df.melt(id_vars=["ANIMAL"], var_name=..., value_name=...)`
- what it does: Wide to tidy. Folds a set of columns into one column of names and one of values, so the timepoint becomes data rather than a column heading.
- example: `wide.melt(id_vars=["ANIMAL", "GROUP"], var_name="TIME_LABEL", value_name="INHIB_PCT")`
- watch out: id_vars lists the columns to LEAVE ALONE. Everything else gets folded. Without var_name and value_name you get columns called variable and value.

### .pivot

- signature: `df.pivot(index=..., columns=..., values=...)`
- what it does: Tidy back to wide. The inverse of melt.
- example: `tidy.pivot(index="ANIMAL", columns="TIME_H", values="INHIB_PCT")`
- watch out: Fails with Index contains duplicate entries if two rows claim the same index and column. That error is the data telling you something, so look before you silence it. pivot_table averages duplicates instead, which is right only if averaging them is scientifically right.

### .duplicated

- signature: `df.duplicated().sum()`
- what it does: Flags rows that repeat a row already seen, and summing counts them.
- example: `int(tidy_clean.duplicated().sum())`
- watch out: By default the FIRST occurrence is not flagged, only the repeats. keep=False flags both copies, which gives double the count.

### .drop_duplicates

- signature: `df.drop_duplicates()`
- what it does: Removes the repeated rows, keeping the first of each.
- example: `tidy_clean.drop_duplicates().pivot(...)`
- watch out: Only safe when the rows really are the same record twice. If the values differ it is a scientific question, not a cleaning step.

### .drop

- signature: `df.drop(columns=["TIME_LABEL"])`
- what it does: Removes columns, or rows with index=.
- example: `.drop(columns=["TIME_LABEL"])`
- watch out: 

### .str accessor

- signature: `df["COL"].str.replace("T_", "", regex=False)`
- what it does: Text operations on a whole column. Also .str.strip, .str.upper, .str.contains, .str.split.
- example: `.str.replace("T_", "", regex=False).astype(float)`
- watch out: regex=False when you mean the characters literally. Without it, characters like . and ( are treated as pattern syntax.

### .astype

- signature: `.astype(float)`
- what it does: Converts a column to another type, usually text to number.
- example: `df["TIME_H"] = df["TIME_LABEL"].str.replace("T_", "").astype(float)`
- watch out: Fails loudly if any value cannot convert, which is useful. pd.to_numeric(..., errors='coerce') turns the bad ones into NaN instead.

### pd.concat

- signature: `pd.concat([a, b], ignore_index=True)`
- what it does: Stacks tables on top of each other.
- example: `pd.concat(frames, ignore_index=True)`
- watch out: ignore_index=True renumbers the result. Without it you get repeated index values, which breaks later lookups.

### .to_numpy

- signature: `df["CONC"].to_numpy()`
- what it does: Hands you the plain numpy array behind a column, when you want numpy functions.
- example: `np.polyfit(df["T"].to_numpy(), df["C"].to_numpy(), 1)`
- watch out: 

### .transform

- signature: `df.groupby("ID")["V"].transform("mean")`
- what it does: Computes per group like agg does, then broadcasts each group's answer back over every row of that group. The result has the same length as the input, so it drops straight into a new column.
- example: `df["BASELINE"] = df.groupby("SUBJECT")["V"].transform("first")`
- watch out: transform('first') takes whatever row is physically first in each group. It does not know what your time column means, so sort before you use it.

### .all / .any

- signature: `(a == b).all()`
- what it does: Reduces a column of True and False to a single True or False. .all() asks whether every value is True, .any() whether at least one is.
- example: `(at_zero["VALUE"] == at_zero["BASELINE"]).all()`
- watch out: Comparing two columns gives a Series, not one answer, so an if on it raises 'The truth value of a Series is ambiguous'. Finish with .all().

### ignore_index

- signature: `df.sort_values([...], ignore_index=True)`
- what it does: Renumbers the index 0, 1, 2 as part of the sort, instead of carrying the old row labels along.
- example: `df.sort_values(["SUBJECT", "TIME_H"], ignore_index=True)`
- watch out: Exactly the same as following the sort with .reset_index(drop=True).

### validate=

- signature: `pd.merge(a, b, on="K", how="left", validate="m:1")`
- what it does: States the relationship you believe the keys have, and raises a MergeError if it is false. 'm:1' means the right hand key must be unique.
- example: `pd.merge(pk, dosing, on="SUBJECT", how="left", validate="m:1")`
- watch out: This is the cheapest protection against a merge that silently duplicates rows. Without it a repeated key doubles your data with no warning.

### indicator=

- signature: `pd.merge(a, b, on=..., how=..., indicator=True)`
- what it does: Adds a _merge column saying whether each row came from left_only, right_only or both, so you can see exactly what failed to match.
- example: `checked[checked["_merge"] == "left_only"]`
- watch out: Put the table you want to account for on the LEFT. Grid on the left answers what is missing; data on the left answers what was unexpected.

### .value_counts

- signature: `df["COL"].value_counts()`
- what it does: Counts how many times each distinct value appears, largest first.
- example: `checked["_merge"].value_counts()`
- watch out: Add dropna=False when you want the NaN count shown rather than skipped.

### pd.MultiIndex.from_product

- signature: `pd.MultiIndex.from_product([a, b], names=[...]).to_frame(index=False)`
- what it does: Every combination of two or more lists, which is how you build the set of rows that SHOULD exist. Join your data onto it and the gaps become rows.
- example: `pd.MultiIndex.from_product([SUBJECTS, TIMES], names=["SUBJECT", "TIME_H"])`
- watch out: Absence is not a value, so no amount of filtering the rows you have will find a row that was never collected. This is the way round that.

### .abs

- signature: `df["DEV"].abs()`
- what it does: The size of each value, ignoring sign.
- example: `out = df["DEV_MIN"].abs() > tolerance`
- watch out: Take it only when you compare, not when you store. A systematic lateness is a different finding from random scatter, and abs() hides which it is.

### .fillna

- signature: `df["N"].fillna(0)`
- what it does: Replaces missing values. The usual finish after a left join that left gaps where a group legitimately had nothing.
- example: `df["N_MISSING"] = df["N_MISSING"].fillna(0).astype(int)`
- watch out: Only fill where zero is genuinely the right answer. Filling a missing measurement with 0 invents data.

### .assign

- signature: `df.assign(ABS_DEV=df["DEV"].abs())`
- what it does: Returns a copy with a new column added, so it chains without touching the original frame.
- example: `.assign(ABS_DEV=df["DEV_MIN"].abs()).groupby("SUBJECT").agg(...)`
- watch out: 

### pd.crosstab

- signature: `pd.crosstab(df["A"], df["B"])`
- what it does: A frequency table: counts how many ROWS fall into each combination of two columns, one down the rows and one across the columns.
- example: `pd.crosstab(ae["GROUP"], ae["SEVERITY"], margins=True, margins_name="Total")`
- watch out: It counts rows. If one subject contributed six rows, they count six times, which is rarely what a safety table should show.

### normalize=

- signature: `pd.crosstab(a, b, normalize="index")`
- what it does: Turns counts into proportions. 'index' makes each row sum to 1, 'columns' each column, 'all' the whole table. Multiply by 100 for percent.
- example: `pd.crosstab(a, b, normalize="index") * 100`
- watch out: Say out loud what you want each row to add up to before you choose, because index and columns answer genuinely different questions.

### pd.pivot_table

- signature: `pd.pivot_table(df, index=..., columns=..., values=..., aggfunc=...)`
- what it does: The general form of crosstab. Takes a values column and any aggregation, so it gives means, maxima or counts of distinct subjects, not just counts.
- example: `pd.pivot_table(ae, index="GROUP", columns="SEVERITY", values="SUBJECT", aggfunc="nunique")`
- watch out: When you want anything other than a row count, you have outgrown crosstab.

### size versus nunique

- signature: `.agg("size") against .agg("nunique")`
- what it does: size counts rows, so it counts events. nunique counts distinct values, so on a subject column it counts people.
- example: `ae.groupby("GROUP")["SUBJECT"].nunique()`
- watch out: The difference between an event count and a patient incidence. Also watch the denominator: subjects with no events are absent from an event log, so it must come from the enrolment table.

### .reindex

- signature: `.reindex(["Placebo", "Low", "High"])`
- what it does: Puts the rows in an order you choose, rather than the alphabetical order groupby and crosstab give you.
- example: `ae_counts.reindex(GROUP_ORDER + ["Total"])`
- watch out: A label not present in the data comes back as a row of NaN rather than an error, which is sometimes what you want and sometimes a silent typo.

### .map

- signature: `df["G"].map({"Low": 0, "High": 1})`
- what it does: Translates each value through a dictionary. Handy for forcing a custom sort order or turning labels into x positions.
- example: `df["_ord"] = df["GROUP"].map({g: i for i, g in enumerate(GROUP_ORDER)})`
- watch out: Values missing from the dictionary become NaN, which then sort to the end.

### looping over groupby

- signature: `for name, part in df.groupby("G"):`
- what it does: Hands you each group's name and that group's rows in turn. The tidy way to draw one line or one series per group.
- example: `for name, part in summary.groupby("DOSE"):     ax.errorbar(part["T"], part["M"], yerr=part["SD"], label=name)`
- watch out: Use it for plotting and for anything genuinely per group. For computing statistics, agg and transform are faster and clearer.

### building a DataFrame from rows

- signature: `pd.DataFrame(list_of_dicts)`
- what it does: Collect one dictionary per row in a list, then build the table once at the end. Clearer and far faster than growing a frame in a loop.
- example: `rows.append({"ANIMAL": a, "N_CELLS": n}) df = pd.DataFrame(rows)`
- watch out: The dictionary keys become the column names, so keep them consistent across rows or you get columns full of NaN.

### three level aggregation

- signature: `.groupby(...).agg(...) once per level`
- what it does: Summarise at each level of nesting before moving up: cells to fields, fields to animals, animals to groups.
- example: `per_field -> per_animal -> group_stats`
- watch out: Each unit counts once at its own level. Pooling instead weights by how much data happened to sit underneath, which is usually a measure of technical luck rather than biology.

### sort_values mixed ascending

- signature: `df.sort_values(cols, ascending=[True, True, False])`
- what it does: Sorts by several columns, each in its own direction. One entry in the list per column.
- example: `nm.sort_values(["ID", "TIME", "EVID"], ascending=[True, True, False])`
- watch out: In a dosing dataset this is the difference between a dose that is given before the time zero sample and one that is given after it. The second runs without complaint and fits the wrong thing.

### groupby apply several outputs

- signature: `df.groupby(key)[cols].apply(fn)`
- what it does: Runs your function on each group and builds a DataFrame from the Series it returns. The way to get several numbers per group.
- example: `obs.groupby("ID")[["TIME", "DV"]].apply(nca_one)`
- watch out: Select the columns before .apply, as shown. Without the selection, newer pandas passes the grouping column into your function and warns about it, and the same line then behaves differently on two versions.

## matplotlib

### plt.subplots

- signature: `fig, ax = plt.subplots(figsize=(6, 4.5))`
- what it does: Makes a figure and its axes. fig is the whole picture, ax is the one set of axes you draw on.
- example: `fig, ax = plt.subplots(figsize=(6, 4.5))`
- watch out: Always build plots this explicit way. plt.plot on its own draws on whatever figure is current, which is hidden state that falls apart as soon as a notebook has more than one figure.

### plt.subplots(rows, cols)

- signature: `fig, axes = plt.subplots(1, 2)`
- what it does: Several panels. You get an ARRAY of axes, reached as axes[0], axes[1].
- example: `fig, axes = plt.subplots(1, 2, figsize=(10, 4))`
- watch out: With rows and columns you must index into axes. Without them you get a single axes that you must not index.

### ax.plot

- signature: `ax.plot(x, y, marker='o')`
- what it does: A line, optionally with markers.
- example: `ax.plot(time_h, conc, marker="o")`
- watch out: 

### ax.errorbar

- signature: `ax.errorbar(x, y, yerr=sd, fmt='o')`
- what it does: Points with error bars.
- example: `ax.errorbar([0, 1], g["MEAN"], yerr=g["SD"], fmt="o", capsize=6)`
- watch out: In this repo yerr is ALWAYS an SD. fmt='o' with linestyle='none' gives markers and no connecting line. A yerr of shape (2, n) is read as separate lower and upper bars.

### ax.scatter

- signature: `ax.scatter(x, y)`
- what it does: Scattered points. The way to overlay every animal on top of a group summary.
- example: `ax.scatter(animal_x, animal_level["MEAN_INTENSITY"], s=45, alpha=0.75)`
- watch out: s is the marker size and alpha is the transparency, which helps when points overlap.

### zorder

- signature: `zorder=3`
- what it does: What gets drawn on top. Higher numbers are nearer the front.
- example: `ax.errorbar(..., zorder=3) over ax.scatter(..., zorder=2)`
- watch out: Without it the scattered points can hide the summary markers.

### axis labels

- signature: `ax.set_xlabel / set_ylabel / set_title`
- what it does: Labels the axes and titles the panel. Include units in the y label.
- example: `ax.set_ylabel("Concentration (mg/L)")`
- watch out: A figure without all three is unreadable to everyone except the person who made it, including you in three weeks.

### ax.set_xticks

- signature: `ax.set_xticks([0, 1])`
- what it does: Where the ticks go. Pair with set_xticklabels to name them.
- example: `ax.set_xticks([0, 1]) ax.set_xticklabels(["Vehicle", "Drug"])`
- watch out: Set the ticks before the labels, or the labels can land in the wrong places.

### ax.set_yscale

- signature: `ax.set_yscale("log")`
- what it does: A log axis. A PK profile that curves on a linear axis becomes a straight terminal line on a semi log axis, which is how you eyeball lambda_z.
- example: `ax.set_yscale("log")`
- watch out: Zero and negative values cannot be shown on a log axis and silently vanish, including your predose samples.

### ax.legend

- signature: `ax.legend(frameon=False)`
- what it does: Shows the legend, built from the label= you passed to each plotting call.
- example: `ax.errorbar(..., label="group mean, SD") ax.legend(frameon=False)`
- watch out: Nothing appears unless you gave the plotting calls a label.

### fig.tight_layout

- signature: `fig.tight_layout()`
- what it does: Tidies up spacing so labels do not overlap or get cut off.
- example: `fig.tight_layout()`
- watch out: Call it after everything is drawn, just before showing or saving.

### plt.show

- signature: `plt.show()`
- what it does: Displays the figure.
- example: `plt.show()`
- watch out: In a notebook the figure usually appears anyway, but calling it is explicit and stops a stray text repr printing above your plot.

### ax.imshow

- signature: `ax.imshow(img, cmap="gray", vmin=0, vmax=255)`
- what it does: Displays an array as an image. Row 0 is drawn at the top.
- example: `ax.imshow(field, cmap="gray", vmin=0, vmax=255) ax.axis("off")`
- watch out: Without vmin and vmax it autoscales each image to its own range, which makes two images with different exposures look identical. Fix the range whenever you are comparing.

## scipy

### from scipy import stats

- signature: `from scipy import stats`
- what it does: The statistics module: tests, distributions and summary helpers.
- example: `from scipy import stats stats.ttest_ind(a, b)`
- watch out: Import the submodule explicitly. A bare import scipy does not give you scipy.stats.

### result objects

- signature: `res = stats.ttest_ind(a, b)`
- what it does: Every scipy.stats test returns an object, not a plain pair. Read it by name with res.statistic and res.pvalue.
- example: `res = stats.shapiro(x) print(res.pvalue)`
- watch out: Unpacking by position still works but breaks when scipy adds fields, and it reads worse.

### stats.ttest_ind

- signature: `stats.ttest_ind(a, b, equal_var=False)`
- what it does: Two sample t-test comparing means. With equal_var=False it is Welch's version, which does not assume equal variances.
- example: `stats.ttest_ind(vehicle, treated, equal_var=False)`
- watch out: The DEFAULT is equal_var=True, the pooled version, which is usually not what you want. Welch costs nothing when variances match and saves you when they do not, especially with unequal group sizes.

### stats.mannwhitneyu

- signature: `stats.mannwhitneyu(a, b, alternative="two-sided")`
- what it does: A rank based comparison, asking whether one group's values tend to be larger, with no assumption about distribution shape.
- example: `stats.mannwhitneyu(vehicle, treated, alternative="two-sided")`
- watch out: Pass alternative explicitly so the choice is visible. Being rank based it barely notices an extreme value, which is its appeal on skewed data.

### stats.shapiro

- signature: `stats.shapiro(x)`
- what it does: Tests the null hypothesis that a sample came from a normal distribution. A SMALL p value is evidence AGAINST normality.
- example: `p_raw = stats.shapiro(auc).pvalue`
- watch out: Nearly powerless on small samples and oversensitive on large ones, so treat it as one input next to a look at the data, never as a gate.

### stats.iqr

- signature: `stats.iqr(x)`
- what it does: The interquartile range, the 75th percentile minus the 25th. A spread measure no single extreme value can move.
- example: `stats.iqr(vehicle)`
- watch out: Report it next to the median, the way SD goes next to the mean.

### geometric mean

- signature: `np.exp(np.mean(np.log(x)))`
- what it does: The mean computed in log space and brought back. The right centre for multiplicative quantities such as AUC and Cmax.
- example: `geo_mean = np.exp(np.mean(np.log(auc)))`
- watch out: Always at or below the arithmetic mean, and equal to the median for perfectly lognormal data. A fold change on a label is a ratio of these.

### geometric CV

- signature: `100 * np.sqrt(np.exp(sd_log ** 2) - 1)`
- what it does: The spread of lognormal data as a percentage, derived from the SD of the LOGS rather than from the raw values.
- example: `sd_log = np.std(np.log(auc), ddof=1) geo_cv = 100 * np.sqrt(np.exp(sd_log ** 2) - 1)`
- watch out: There is no way to get this from the arithmetic mean and SD, which is why a PK table can show a CV matching nothing else on the page.

### curve_fit

- signature: `popt, pcov = curve_fit(f, xdata, ydata, p0=[...])`
- what it does: Least squares fitting of a model you write. The model takes the independent variable FIRST, then the parameters.
- example: `def decay(t, c0, k):     return c0 * np.exp(-k * t) popt, pcov = curve_fit(decay, t, c, p0=[50, 0.1])`
- watch out: Keep pcov. Its diagonal, square rooted, is the standard error of each parameter, which is what turns an estimate into a result.

### p0

- signature: `curve_fit(f, x, y, p0=[50.0, 0.1])`
- what it does: Where the search starts. The default is 1.0 for every parameter, which is rarely the right order of magnitude.
- example: `p0=[conc[0], np.log(2) / 7.0]`
- watch out: It only needs the right order of magnitude, and you can usually read one off the data. A bad p0 gives nonsense or RuntimeError: Optimal parameters not found.

### bounds

- signature: `curve_fit(f, x, y, bounds=([0, 0], [200, 1e4]))`
- what it does: Lower bounds then upper bounds, one entry per parameter in signature order, keeping the search where the parameters mean something.
- example: `bounds=([0.0, 1e-6, 0.1], [200.0, 1e4, 10.0])`
- watch out: Not cheating: it states what you already know, such as a concentration being positive. Use np.inf for an upper bound you do not want to set.

### Emax model

- signature: `emax * c / (ec50 + c)`
- what it does: The standard concentration response curve: zero at zero, a ceiling at Emax, and half the ceiling at c = EC50.
- example: `def emax_model(c, emax, ec50):     return emax * c / (ec50 + c)`
- watch out: EC50 is a CONCENTRATION in the units of c, not a percentage. The same hyperbola as Michaelis-Menten and as simple receptor occupancy.

### sigmoid Emax

- signature: `emax * c ** h / (ec50 ** h + c ** h)`
- what it does: Emax with a Hill coefficient controlling steepness. h = 1 gives the plain model back, so the two models are nested.
- example: `def sigmoid_model(c, emax, ec50, h):     return emax * c ** h / (ec50 ** h + c ** h)`
- watch out: h must be applied to BOTH c and ec50, or EC50 stops meaning half maximal. The richer model always fits at least as well, so a lower RMSE is not evidence that h is real.

### solve_ivp

- signature: `solve_ivp(fun, (t0, t1), [y0], t_eval=grid, args=(...))`
- what it does: Numerical integration of a differential equation. The derivative takes (t, y) with y an array, and returns one derivative per state.
- example: `def rhs(t, y, cl, v):     return [-(cl / v) * y[0]] sol = solve_ivp(rhs, (0, 24), [dose], t_eval=grid, args=(cl, v))`
- watch out: y0 must be a sequence even with one compartment. The state is an AMOUNT, so divide by volume to get a concentration.

### t_eval

- signature: `solve_ivp(..., t_eval=np.linspace(0, 24, 97))`
- what it does: Where the solution is reported. Without it you get the solver's own unevenly spaced steps.
- example: `t_eval=T_GRID`
- watch out: It controls OUTPUT, not accuracy. Accuracy is rtol and atol, whose defaults ask for about three correct digits. For PK work pass rtol=1e-8, atol=1e-10.

### max_step

- signature: `solve_ivp(..., max_step=0.05)`
- what it does: Caps the step size so the solver cannot stride over something that happens suddenly, such as an infusion stopping.
- example: `solve_ivp(rhs_inf, (0, 12), [0.0], max_step=0.05, ...)`
- watch out: An adaptive solver picks its step from how smooth the solution has been, and a discontinuity is exactly what that history cannot warn it about. It fails silently, with success still True.

### sol.y

- signature: `sol.y[0]`
- what it does: The solution, one ROW per state and one column per reported time. sol.t holds the times and sol.success says whether it worked.
- example: `conc = sol.y[0] / V_L`
- watch out: sol.y is always two dimensional, even for one compartment, so sol.y[0] is how you get a 1D array out.

### curve_fit bounds

- signature: `curve_fit(f, x, y, p0=[...], bounds=([lo], [hi]))`
- what it does: Restricts the search to parameter values that mean something. Two sequences, lower bounds then upper.
- example: `curve_fit(emax, c, e, p0=[20.0, 1.0], bounds=([0.0, 1e-3], [200.0, 100.0]))`
- watch out: Bound a concentration or a rate constant away from zero rather than at zero, since the model usually divides by it. If a fitted value comes back exactly ON a bound, the fit did not converge, it stopped.

### finite differences

- signature: `(f(p + h) - f(p - h)) / (2 * h)`
- what it does: The numerical derivative. Two extra evaluations per parameter, and you have to choose h.
- example: `(loss(v + 1e-6) - loss(v - 1e-6)) / 2e-6  gives  -0.2011`
- watch out: Too large an h and the approximation is wrong, too small and floating point cancellation eats the answer. This is what curve_fit does internally, and what autograd replaces with exact derivatives.

## imaging

### images as arrays

- signature: `img[row, column]`
- what it does: A greyscale image is a 2D numpy array, usually uint8 from 0 to 255. img.shape is (height, width) and row 0 is the top.
- example: `field.shape, field.dtype, field[:4, :4]`
- watch out: uint8 arithmetic WRAPS: 250 + 10 gives 4, not 260. Convert to float before any arithmetic you care about.

### filters.threshold_otsu

- signature: `filters.threshold_otsu(img)`
- what it does: Computes a threshold from the image's own histogram, splitting it into background and foreground.
- example: `mask = img > filters.threshold_otsu(img)`
- watch out: Because it is derived per image it follows changes in illumination, which a hard coded number cannot. It assumes two populations, so on a field that is almost all background it cuts somewhere arbitrary.

### filters.gaussian

- signature: `filters.gaussian(img, sigma=1.5)`
- what it does: Blurs the image, averaging each pixel with its neighbours. Removes pixel level noise before thresholding.
- example: `smoothed = filters.gaussian(field, sigma=1.5)`
- watch out: It returns FLOATS rescaled to 0 to 1, not uint8, so a threshold computed on the original is meaningless on the result. Keep sigma well below the size of the objects you are measuring.

### measure.label

- signature: `measure.label(mask)`
- what it does: Connected component labelling. Background becomes 0 and each separate object gets its own integer, so labels.max() counts them.
- example: `labels = measure.label(mask) n = int(labels.max())`
- watch out: connectivity=2, the default in 2D, joins objects touching only at a corner. connectivity=1 uses the four edge neighbours only.

### measure.regionprops_table

- signature: `measure.regionprops_table(labels, intensity_image=img, properties=(...))`
- what it does: Measures every labelled object at once and returns a dict of arrays, which goes straight into pd.DataFrame.
- example: `pd.DataFrame(measure.regionprops_table(labels, intensity_image=field,     properties=("label", "area", "eccentricity", "mean_intensity")))`
- watch out: area is in PIXELS until you multiply by the pixel size. Properties with more than one number, such as centroid, arrive as centroid-0 and centroid-1.

### eccentricity

- signature: `properties=("eccentricity",)`
- what it does: Shape measure from 0 for a perfect circle to 1 for a line. Separates round cells from elongated ones and from merged pairs.
- example: `cells[cells["eccentricity"] < 0.7]`
- watch out: Two touching cells label as one object with roughly double the area and a stretched shape, so area and eccentricity together spot them.

### connectivity

- signature: `measure.label(mask, connectivity=1)`
- what it does: Which neighbours count as connected. 1 means the four edge neighbours, 2 also counts the four diagonals and is the 2D default.
- example: `measure.label(mask, connectivity=1)`
- watch out: Diagonal connectivity merges objects that touch only at a corner, which is usually right for cells and occasionally not.

### morphology.opening

- signature: `morphology.opening(mask, morphology.disk(2))`
- what it does: Erosion then dilation. Deletes anything the footprint cannot fit inside and restores the survivors, so it removes speckle without shrinking real objects.
- example: `mask_open = morphology.opening(mask_raw, morphology.disk(2))`
- watch out: Too small a footprint leaves speckle; too large eats thin parts of real objects. Check against the thinnest structure you care about.

### morphology.closing

- signature: `morphology.closing(mask, morphology.disk(2))`
- what it does: Dilation then erosion. Fills small gaps and joins things that nearly touch.
- example: `morphology.closing(mask, morphology.disk(2))`
- watch out: It will also MERGE two cells sitting within the footprint of each other, silently costing you a count. For holes inside objects prefer binary_fill_holes, which cannot do that.

### morphology.disk

- signature: `morphology.disk(2)`
- what it does: A roughly circular footprint of the given radius, the usual choice for round objects because it treats every direction alike.
- example: `morphology.opening(mask, morphology.disk(2))`
- watch out: The radius is the thing to tune, and it is in pixels.

### binary_fill_holes

- signature: `ndi.binary_fill_holes(mask)`
- what it does: Fills background regions completely enclosed by foreground, whatever their size or shape. The right tool for cells that stain as rings.
- example: `from scipy import ndimage as ndi mask_filled = ndi.binary_fill_holes(mask_open)`
- watch out: Lives in scipy.ndimage, not skimage. Works topologically rather than geometrically, so unlike closing it can never merge two objects.

### clear_border

- signature: `segmentation.clear_border(mask)`
- what it does: Removes every object touching the edge of the frame, which are only partly imaged.
- example: `mask_clean = segmentation.clear_border(mask_filled)`
- watch out: Right for PER CELL measurements, since a cropped cell gives a wrong area, shape and intensity. Wrong for DENSITY, where it undercounts a real quantity. The question decides, not the image.

### cv2.GaussianBlur

- signature: `cv2.GaussianBlur(img, (5, 5), 0)`
- what it does: OpenCV's blur. The kernel is given as (width, height) and both must be ODD. A sigma of 0 means derive it from the kernel size.
- example: `blurred = cv2.GaussianBlur(field, (5, 5), 0)`
- watch out: OpenCV sizes are (width, height), the opposite of a numpy shape. Most OpenCV functions also want uint8 and will not accept floats.

### cv2.threshold

- signature: `t, binimg = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)`
- what it does: Thresholds an image, returning BOTH the threshold used and the result.
- example: `t_cv, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)`
- watch out: It returns a PAIR, so unpack it. Adding THRESH_OTSU makes it compute the threshold and ignore the value you passed, which is why that argument is written as 0. The output is uint8 0 and 255, not booleans.

### cv2.findContours

- signature: `cv2.findContours(binimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)`
- what it does: Traces the outline of every object. RETR_EXTERNAL keeps outermost contours only; CHAIN_APPROX_SIMPLE compresses straight runs of points.
- example: `contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]`
- watch out: It returned three values in OpenCV 3 and two in 4 and 5, so taking [-2] is the version proof way to get the contours.

### cv2.contourArea

- signature: `cv2.contourArea(c)`
- what it does: The area of the polygon through a contour's boundary points.
- example: `areas = [cv2.contourArea(c) for c in contours]`
- watch out: NOT the same as skimage's pixel count. The polygon runs through the centres of the boundary pixels, so it comes out a few percent smaller, and proportionally more so for small objects. Pick one and say which.

## sklearn

### fit and predict

- signature: `model.fit(X, y); model.predict(X)`
- what it does: Every estimator has the same interface, which is why swapping one model for another is a one line change. X is 2D, samples by features; y is 1D.
- example: `model = LinearRegression().fit(X_train, y_train) pred = model.predict(X_test)`
- watch out: fit returns the fitted model, so the one line form above works. A model with no coef_ or feature_importances_ was never fitted.

### train_test_split

- signature: `train_test_split(X, y, test_size=0.25, random_state=42)`
- what it does: Shuffles and splits in one call, returning FOUR things: X train, X test, y train, y test, in that order.
- example: `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)`
- watch out: Swapping the middle two gives shapes that look plausible and results that are nonsense. Spend the test set once, at the end.

### random_state

- signature: `random_state=42`
- what it does: Fixes the randomness of a split, a forest or a CV shuffle so a run is reproducible.
- example: `RandomForestRegressor(n_estimators=300, random_state=0)`
- watch out: Without it a rerun changes the answer by more than most real improvements do, and you cannot tell the two apart.

### r2_score

- signature: `r2_score(y_true, y_pred)`
- what it does: Fraction of variance explained. 1 is perfect, 0 is no better than predicting the mean, and it can go NEGATIVE on held out data.
- example: `r2_test = r2_score(y_test, pred_test)`
- watch out: Unitless, so it compares across problems. A negative value is not an error, it means the model is worse than the mean.

### mean_squared_error

- signature: `np.sqrt(mean_squared_error(y_true, y_pred))`
- what it does: RMSE, the typical size of an error in the units of y, which is the number to quote to a pharmacologist.
- example: `rmse = np.sqrt(mean_squared_error(y_test, pred_test))`
- watch out: Take the square root yourself. The squared=False argument exists only in some versions and was replaced by a separate function in newer ones.

### KFold

- signature: `KFold(n_splits=5, shuffle=True, random_state=1)`
- what it does: Splits the data into k parts so every sample is tested exactly once, giving k estimates rather than one.
- example: `kf = KFold(n_splits=5, shuffle=True, random_state=1)`
- watch out: Shuffle matters. Without it the folds are consecutive blocks of the file, so any ordering by site, cohort or date confounds them.

### cross_val_score

- signature: `cross_val_score(model, X, y, cv=kf, scoring="r2")`
- what it does: Runs the whole cross-validation and returns the k fold scores.
- example: `scores = cross_val_score(pipe, X, y, cv=kf, scoring='r2') scores.mean(), scores.std(ddof=1)`
- watch out: Report the mean AND the spread. A mean of 0.4 from folds spanning 0.1 to 0.7 is a very different claim from one made of folds all near 0.4.

### Pipeline

- signature: `Pipeline([("sel", SelectKBest(...)), ("model", LinearRegression())])`
- what it does: Chains steps into one estimator, so cross-validation refits every step on each training fold rather than once on everything.
- example: `pipe = Pipeline([('sel', SelectKBest(f_regression, k=6)), ('model', LinearRegression())])`
- watch out: A pipeline has no coef_ of its own; reach the model with pipe[-1]. make_pipeline is the same thing without naming the steps.

### data leakage

- signature: `anything fitted before the split`
- what it does: Information from the test data reaching the model. Usually silent, always flattering.
- example: `WRONG: select features on all data, then cross-validate RIGHT: put the selector inside a Pipeline`
- watch out: The rule: anything that LEARNS from the data belongs inside the pipeline. Scaling learns a mean, imputation learns a fill value, selection learns which columns. Only per row operations such as a log are safe outside.

### SelectKBest

- signature: `SelectKBest(f_regression, k=6)`
- what it does: Keeps the k features most associated with the target. .get_support() gives a boolean mask of which were chosen.
- example: `pipe = Pipeline([('sel', SelectKBest(f_regression, k=6)), ('model', LinearRegression())])`
- watch out: The classic leakage source. Fitted on all the data it picks the columns that happen to match the target in the very rows you will test on, which can report a solid result for pure noise.

### RandomForestRegressor

- signature: `RandomForestRegressor(n_estimators=300, random_state=0)`
- what it does: Many decision trees on bootstrap samples, averaged. Needs no scaling and captures interactions and curvature nobody specified.
- example: `rf = RandomForestRegressor(n_estimators=300, random_state=0).fit(X_train, y_train)`
- watch out: Unrestricted it fits training data almost perfectly, so the train to test gap is large. That is capacity, not understanding, and on smooth data a linear model often still wins on the test set.

### feature_importances_

- signature: `rf.feature_importances_`
- what it does: Impurity importance: how much each feature reduced error at the splits using it. Free with training, and sums to 1.
- example: `pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)`
- watch out: Biased toward features with many distinct values, so a continuous noise column can outrank a real binary one. It also splits credit arbitrarily between correlated features.

### permutation_importance

- signature: `permutation_importance(model, X_test, y_test, n_repeats=20, random_state=0)`
- what it does: Shuffles one column at a time and measures how much the score falls. Read .importances_mean.
- example: `perm = permutation_importance(rf, X_test, y_test, n_repeats=20, random_state=0)`
- watch out: Compute it on HELD OUT data, or it rewards memorisation. No cardinality bias and the units are score units. A negative value means the model should never have leaned on that feature.

### StandardScaler

- signature: `StandardScaler().fit_transform(X)`
- what it does: Centres each feature and scales it to unit variance.
- example: `make_pipeline(StandardScaler(), ElasticNet(alpha=0.05))`
- watch out: Required, not advisory, for anything that penalises coefficient size or maximises variance. It learns a mean and an SD, so it belongs INSIDE the pipeline.

### ElasticNet

- signature: `ElasticNet(alpha=0.05, l1_ratio=0.5)`
- what it does: Least squares plus a penalty on coefficients: L1 drives them to exactly zero, L2 shrinks smoothly and shares credit between correlated features.
- example: `make_pipeline(StandardScaler(), ElasticNet(alpha=0.05, l1_ratio=0.5, random_state=0))`
- watch out: The penalty is in COEFFICIENT units, so without standardising, which features survive depends on the units they were recorded in. A feature in grams is charged a thousandth of the same feature in kilograms.

### ElasticNetCV

- signature: `ElasticNetCV(l1_ratio=0.5, cv=5, random_state=0)`
- what it does: Cross-validates over a grid of alphas and keeps the best, exposed as .alpha_.
- example: `encv = make_pipeline(StandardScaler(), ElasticNetCV(l1_ratio=0.5, cv=5))`
- watch out: Choosing alpha this way uses the training data only, so the test set stays clean. Picking the alpha with the best test score instead turns the test set into training data.

### make_pipeline

- signature: `make_pipeline(StandardScaler(), ElasticNet())`
- what it does: Pipeline without having to name the steps; names are generated from the class.
- example: `pipe = make_pipeline(StandardScaler(), ElasticNet(alpha=0.05)) pipe[-1].coef_`
- watch out: pipe[-1] gets the final estimator whatever it is called, which is how you reach coef_ or alpha_.

### PCA

- signature: `PCA().fit(Z)`
- what it does: Finds the perpendicular directions of greatest variance and rewrites each sample in terms of them.
- example: `Z = StandardScaler().fit_transform(panel) pca = PCA().fit(Z) scores = pca.transform(Z)`
- watch out: Standardise first, without exception. PCA maximises variance, and variance is in squared feature units, so otherwise PC1 just points along whichever column has the biggest numbers.

### explained_variance_ratio_

- signature: `pca.explained_variance_ratio_`
- what it does: The share of total variance each component carries, summing to 1.
- example: `n_for_90 = int(np.argmax(np.cumsum(evr) >= 0.90) + 1)`
- watch out: It answers how many things your panel is really measuring. Note that components explain variance in X with no reference to any outcome, so a small component can still be the clinically important one.

### components_

- signature: `pca.components_`
- what it does: The loadings: one ROW per component, one column per original feature, saying what each component is made of.
- example: `pd.DataFrame(pca.components_, columns=FEATURES, index=["PC1", "PC2"])`
- watch out: The sign of a whole component is arbitrary, so only read signs RELATIVE to each other within one row. Sort by absolute loading to find the dominant features.

## torch

### torch.tensor

- signature: `torch.tensor(data, dtype=torch.float64)`
- what it does: Makes a tensor, which is a numpy array that can also record how it was computed.
- example: `t = torch.tensor(time_np, dtype=torch.float64)`
- watch out: Always say the dtype. torch defaults to float32, and INFERS an integer type from a list of Python integers, on which gradients are impossible and division behaves differently.

### tensor dtypes

- signature: `t.dtype, torch.float64, torch.int64`
- what it does: What kind of number the tensor holds. float64 matches numpy, float32 is the torch default, int64 is what a list of integers infers.
- example: `torch.tensor([1, 2, 3]).dtype  gives  torch.int64`
- watch out: An integer parameter tensor cannot carry a gradient, and the error you get says something else entirely. For PK work use float64 everywhere: float32 gives about seven digits, which an optimiser can run out of.

### .item()

- signature: `t.item()`
- what it does: Pulls a single number out of a one element tensor as an ordinary Python float.
- example: `loss.item()  gives  0.22084644423969818`
- watch out: Use it the moment a number leaves torch for printing, comparison or a list. A one element tensor prints as tensor(0.2208, dtype=...) and compares in ways that surprise people.

### tensor to numpy

- signature: `t.numpy(), t.detach().numpy()`
- what it does: Hands you a numpy VIEW of the same memory. No copy is made.
- example: `arr = c_tensor.detach().numpy()`
- watch out: Change either one and both change. Use .numpy().copy() for a snapshot. On a tensor that tracks gradients .numpy() refuses outright, and .detach() first is the fix.

### requires_grad

- signature: `torch.tensor(x, requires_grad=True)`
- what it does: Tells torch to record every operation involving this tensor, so it can be differentiated later. Parameters need it, data does not.
- example: `k = torch.tensor(0.15, dtype=torch.float64, requires_grad=True)`
- watch out: You can see it working: anything computed from a tracked tensor has a grad_fn, and anything else has grad_fn of None. That one attribute is the whole difference between a tensor and an array.

### backward

- signature: `loss.backward()`
- what it does: Walks the recorded graph from the loss back to every tracked tensor, applying the chain rule, and deposits the result in each .grad.
- example: `loss.backward()  then read  k.grad`
- watch out: It wants a SCALAR, so reduce with .sum() or .mean() first. It also frees the graph afterwards, so a second call on the same loss raises. Recompute the loss instead of passing retain_graph=True.

### tensor .grad

- signature: `p.grad, p.grad.item()`
- what it does: Where backward() leaves the derivative of the loss with respect to that tensor. None until the first backward pass.
- example: `k.grad.item()  gives  12.358762098200062`
- watch out: Read it straight after the backward pass that produced it. It is a tensor that later passes write into, so a reference you saved earlier will have changed under you.

### gradient accumulation

- signature: `p.grad = None  or  p.grad.zero_()`
- what it does: Clears the gradient. Necessary because backward() ADDS to .grad rather than replacing it.
- example: `for i in range(n): p.grad = None; ... loss.backward()`
- watch out: Forget it and every step is larger than the last, so the loss explodes after one good iteration and the symptom points nowhere near the cause. p.grad = None always works, .zero_() fails on the first pass.

### torch.no_grad

- signature: `with torch.no_grad(): ...`
- what it does: Switches the recording off inside the block. For anything you are not differentiating: predictions, a validation loss, the parameter update.
- example: `with torch.no_grad(): p -= lr * p.grad`
- watch out: The update has to be inside it, or torch raises a leaf Variable that requires grad is being used in an in place operation. It is also what lets .numpy() work on a result computed from parameters.

### gradient descent loop

- signature: `zero grad, forward, backward, step`
- what it does: The whole of fitting, in four lines: clear the gradients, compute the loss, differentiate it, move the parameters downhill.
- example: `p.grad = None; loss = f(p); loss.backward(); with torch.no_grad(): p -= lr * p.grad`
- watch out: The gradient gives the direction and says nothing about the distance. Everything that goes wrong in fitting is visible in these four lines.

### learning rate

- signature: `lr, the step size`
- what it does: How far to move along the gradient each iteration.
- example: `torch.optim.Adam(model.parameters(), lr=0.05)`
- watch out: Too small and it crawls, too large and it steps past the minimum to somewhere worse and the next step is larger still. A diverging fit does not always produce a NaN: it can settle on a flat part of the surface with absurd parameters and look converged.

### log parameters

- signature: `carry log_v, use torch.exp(log_v)`
- what it does: Fitting on the log scale, so a fixed step is a fixed percentage change and the parameter cannot go negative.
- example: `v = torch.exp(log_v)  inside forward`
- watch out: This is the standard fix when parameters are on wildly different scales, which in PK they always are. Remember to exponentiate before reporting anything.

### torch.nn.Module

- signature: `class M(torch.nn.Module): __init__, forward`
- what it does: A container that knows what its parameters are. You create them in __init__ and define the arithmetic in forward.
- example: `class OneCompartmentIV(torch.nn.Module): ...`
- watch out: super().__init__() comes first in __init__, or assigning a parameter raises. Call the model as model(t), not model.forward(t), so that the hooks torch wraps around forward still run.

### torch.nn.Parameter

- signature: `torch.nn.Parameter(torch.tensor(x, dtype=torch.float64))`
- what it does: A tensor that sets requires_grad for you and registers itself with the Module it is assigned to, so model.parameters() can find it.
- example: `self.log_v = torch.nn.Parameter(torch.tensor(np.log(35.0), dtype=torch.float64))`
- watch out: A plain tensor assigned to self.something is INVISIBLE to the optimiser. The fit runs, and that parameter never moves, and nothing warns you.

### torch.optim

- signature: `torch.optim.Adam(model.parameters(), lr=0.05)`
- what it does: An optimiser: it holds the parameters and the update rule. SGD with nothing else set is plain gradient descent; Adam adapts a step size per parameter.
- example: `opt = torch.optim.Adam(model.parameters(), lr=0.05)`
- watch out: Adam survives badly scaled parameters that defeat SGD at the same learning rate, which is why it is the default choice. It is not magic: it still diverges if the rate is far too large.

### zero_grad and step

- signature: `opt.zero_grad(); ...; opt.step()`
- what it does: The optimiser versions of clearing every gradient and applying every update, for all the parameters it was given.
- example: `opt.zero_grad(); loss = ...; loss.backward(); opt.step()`
- watch out: The order matters and never changes. step() after backward(), zero_grad() before the next forward pass.

### state_dict

- signature: `model.state_dict()`
- what it does: A dictionary of the model's current parameter values, for saving a fit or inspecting it.
- example: `list(model.state_dict().keys())  gives  ["log_v", "log_k"]`
- watch out: It holds the values on whatever scale the model carries them, so log parameters come out as logs. Exponentiate before anybody reads them.

### torch.full

- signature: `torch.full((n,), value, dtype=torch.float64)`
- what it does: An array of n copies of one value. The usual way to start a vector of per subject parameters at a common guess.
- example: `torch.nn.Parameter(torch.full((12,), float(np.log(30.0)), dtype=torch.float64))`
- watch out: The shape argument is a tuple, so (n,) with the comma. torch.full(n, ...) without it raises.

### parameter vectors

- signature: `nn.Parameter of shape (n_subjects,)`
- what it does: One parameter per subject, fitted all at once. backward() fills in every element and the optimiser steps every element.
- example: `self.log_v = nn.Parameter(torch.full((12,), np.log(30.0), dtype=torch.float64))`
- watch out: There is no loop over subjects anywhere, and the same code handles a hundred and twenty subjects. Check .shape on the predictions before anything else: a wrong axis gives a square matrix, not an error.

## PK

### linear trapezoid

- signature: `sum(0.5 * (c[1:] + c[:-1]) * np.diff(t))`
- what it does: Area under the curve by joining the samples with straight lines. The default for AUC up to the last measurable sample.
- example: `auc = float(np.sum(0.5 * (c[1:] + c[:-1]) * np.diff(t)))`
- watch out: On a declining log linear curve straight lines sit ABOVE the true curve, so the linear rule overestimates the tail. The wider the sampling gaps, the worse it gets.

### linear up log down

- signature: `linear while rising, log trapezoid while falling`
- what it does: The regulatory default: straight lines up to Tmax, then the log trapezoid, which follows an exponential decline exactly.
- example: `c1 * dt if c2 >= c1 else (c1 - c2) * dt / np.log(c1 / c2)`
- watch out: The log form divides by log(c1 / c2), so it is undefined when two consecutive values are equal and impossible when either is zero. Fall back to the linear rule for those intervals.

### partial AUC

- signature: `AUC over a stated window, such as 0 to 12 h`
- what it does: The exposure over part of the interval, used when only part of it matters clinically, or to compare two formulations over a window.
- example: `auc_0_12 = trapezoid over the samples up to 12 h`
- watch out: If the window boundary is not a sampling time you have to interpolate to it, and say which way: linearly in concentration or linearly in log concentration. The two differ by several percent.

### lambda_z selection

- signature: `regress log concentration on time over the terminal points`
- what it does: The terminal elimination rate constant, from the slope of the last few points on a log scale. Choosing WHICH points is the whole problem.
- example: `slope of np.polyfit(t[-n:], np.log(c[-n:]), 1), lambda_z = -slope`
- watch out: The standard rule is to take at least three points and keep adding earlier ones while the adjusted R squared improves. Include a point from the distribution phase and lambda_z comes out too steep, which shortens the half life and shrinks the extrapolated AUC.

### adjusted R2

- signature: `1 - (1 - r2) * (n - 1) / (n - p - 1)`
- what it does: R squared with a penalty for the number of points used, so it can be compared between fits with different n.
- example: `used to choose how many terminal points go into lambda_z`
- watch out: Plain R squared almost always improves when you add a point to a regression, so it cannot choose between windows of different sizes. That is the entire reason the adjusted form is the rule here.

### AUC extrapolation

- signature: `AUC_inf = AUC_last + C_last / lambda_z`
- what it does: Adding the area from the last sample to infinity, assuming the terminal slope continues.
- example: `auc_inf = auc_last + c_last / lambda_z`
- watch out: Report the percentage extrapolated. Above about twenty percent the number is mostly assumption and regulators say so. It is also extremely sensitive to lambda_z, which is the least well determined quantity in the whole NCA.

### mean residence time

- signature: `MRT = AUMC / AUC`
- what it does: The average time a molecule spends in the body. AUMC is the area under the concentration times time curve.
- example: `mrt = aumc_inf / auc_inf`
- watch out: AUMC weights the tail by time, so it is even more sensitive to the extrapolation than AUC is. For a one compartment IV bolus MRT is 1 / k, which is the half life divided by ln(2), not the half life.

### depot and central

- signature: `dA/dt = -ka * A ; dC/dt = ka * A / V - k * C`
- what it does: The two compartment picture of an oral dose: an amount waiting to be absorbed, and a concentration being eliminated.
- example: `solve_ivp over [depot, central] with the dose in the depot`
- watch out: The depot carries an AMOUNT and the central compartment a CONCENTRATION, so the ka term is divided by V on the way in. Mixing up amounts and concentrations is the most common error in writing these systems.

### fitting an ODE model

- signature: `solve the system inside the model function curve_fit calls`
- what it does: Every candidate parameter set means solving the system again and reading it at the sampling times.
- example: `def model(t, ka, k, v): sol = solve_ivp(...); return sol.y[1]`
- watch out: Set rtol and atol tight, 1e-8 and 1e-10, before you fit. Default tolerances leave solver error large enough for the optimiser to chase, and the parameters it returns then depend on the solver settings.

### flip-flop kinetics

- signature: `ka < k, so absorption controls the terminal slope`
- what it does: When absorption is slower than elimination, the terminal phase measures absorption and the two rate constants swap roles.
- example: `a slowly absorbed depot formulation with a short elimination half life`
- watch out: The fitted pair (ka, k) and the pair with them exchanged describe the same curve, so the fit cannot tell them apart from oral data alone. It takes IV data, or a known half life, to break the tie.

### two compartment model

- signature: `central and peripheral, with distribution between them`
- what it does: A drug that distributes into tissue shows a fast early fall as it distributes and a slower one as it is eliminated.
- example: `dC1/dt = -(k10 + k12) * C1 + k21 * C2 * V2 / V1`
- watch out: A one compartment model fitted to two compartment data chases the fast early phase, which inflates clearance, understates the exposure and can be wrong about the concentration at the end of a dosing interval by orders of magnitude.

### alpha and beta phases

- signature: `the two exponentials of a two compartment curve`
- what it does: Alpha is the fast distribution phase, beta the slow terminal phase. The half life quoted for a drug is usually the beta one.
- example: `C(t) = A * exp(-alpha * t) + B * exp(-beta * t)`
- watch out: Sampling that stops too early catches only the alpha phase and gives a half life several fold too short. If a terminal slope looks too steep, ask how long the samples ran, not just how many there were.

### Vss

- signature: `Vss = V1 + V2`
- what it does: The steady state volume of distribution: all the space the drug eventually occupies, not just the compartment it is measured in.
- example: `vss = v1 + v2  for a two compartment model`
- watch out: V1 alone is the volume the first sample sees, and is much smaller. Quoting V1 as the volume of distribution understates the tissue burden and the loading dose.

### residual structure

- signature: `plot residuals against time and against prediction`
- what it does: Residuals that run in a block of one sign rather than scattering are telling you the model is wrong, not that the data are noisy.
- example: `mean residual before Tmax -5.07, after Tmax +5.10`
- watch out: The sum of squares cannot see this. Two fits with the same SSE, one scattered and one structured, are not equally good, and only the plot distinguishes them.

### carrying state forward

- signature: `end the interval, then start the next from there`
- what it does: Simulating repeated doses by solving one interval at a time and using its final state as the next interval's initial condition.
- example: `y0 = sol.y[:, -1].copy()  then add the next dose to the depot`
- watch out: Copy the final state, or you keep a view into the solver's array. And add the dose to what is already there rather than replacing it, which is the difference between accumulation and no accumulation.

### accumulation ratio

- signature: `R = 1 / (1 - exp(-k * tau))`
- what it does: How much higher the steady state concentration is than after the first dose, for a given interval.
- example: `R = 1 / (1 - np.exp(-0.12 * 12))  gives  1.31`
- watch out: The formula assumes the input is instantaneous, so it matches the ratio of TROUGHS closely and the ratio of peaks less well, since absorption smears the peak of the first dose differently.

### time to steady state

- signature: `about 4 to 5 terminal half lives`
- what it does: Accumulation is 90 percent complete after about 3.3 half lives and 97 percent after 5, whatever the dose or the interval.
- example: `a 6 h half life reaches steady state in about 30 h`
- watch out: It depends on the half life only. Raising the dose raises every concentration proportionally and does not get you there sooner, which is what loading doses are for.

### Cavg at steady state

- signature: `C_avg = DOSE / (CL * tau)`
- what it does: The average concentration over a dosing interval at steady state. Clearance and the interval, nothing else.
- example: `100 / (4.2 * 24)  gives  0.992 mg/L`
- watch out: Volume and absorption rate do not appear. They decide how far the concentration swings above and below this average, which is a different question and often the one that matters for tolerability.

### steady state closed form

- signature: `each exponential divided by 1 - exp(-rate * tau)`
- what it does: The single dose equation turned into the steady state one, by summing the infinite series of previous doses.
- example: `coef * (np.exp(-k * t) / (1 - np.exp(-k * TAU))         - np.exp(-ka * t) / (1 - np.exp(-ka * TAU)))`
- watch out: Valid only for a linear model with identical doses at identical intervals. One missed dose, or saturable kinetics, and you are back to simulating interval by interval.

### Cavg identity

- signature: `integrate the steady state profile, divide by tau`
- what it does: Doing the integral numerically and checking it against DOSE / (CL * tau) is the cheapest confirmation that a simulation is right.
- example: `trapezoid(c_ss) / TAU  should equal  DOSE / (CL * TAU)`
- watch out: If the two disagree, the error is usually in the accumulation factors or in using a single dose profile where a steady state one was meant.

### hysteresis

- signature: `plot effect against concentration, joined in time order`
- what it does: A loop rather than a single curve means the effect does not depend on the plasma concentration at that instant.
- example: `effect 6.8 ms on the way up and 22.0 on the way down, at 1.2 mg/L`
- watch out: Counter clockwise, with more effect on the way down, usually means distribution to an effect site. Clockwise usually means tolerance or an active metabolite. Either way a direct Emax fit is the wrong model.

### effect compartment

- signature: `dCe/dt = ke0 * (C(t) - Ce)`
- what it does: One hypothetical compartment, one rate constant, which delays and smooths the plasma profile. The effect is then driven by Ce.
- example: `ln(2) / ke0 is the effect site equilibration half life`
- watch out: It collapses the hysteresis loop with one extra parameter. Its signature is that the time of maximum effect is the SAME at every dose, which is how you tell it from an indirect response.

### residuals against time

- signature: `mean residual before and after the peak`
- what it does: The quickest numerical version of the residual plot: two means that should both be near zero.
- example: `direct fit: -5.07 before Tmax, +5.10 after. link fit: +0.10 and -0.18`
- watch out: A run of one sign followed by a run of the other is a model error, not noise, and it is invisible in the SSE alone.

### indirect response model

- signature: `dR/dt = kin * (1 - Imax * C / (IC50 + C)) - kout * R`
- what it does: The drug acts on the production or loss of a biomarker that has its own turnover, so the biomarker cannot respond faster than it turns over.
- example: `inhibition of production, the most common of the four forms`
- watch out: The time of maximum effect moves LATER as the dose rises, because the concentration stays above IC50 for longer. An effect compartment cannot produce that, so three dose levels distinguish the two mechanisms.

### turnover half life

- signature: `ln(2) / kout`
- what it does: How fast the biomarker itself equilibrates. It sets the whole time course of the effect, independently of the drug.
- example: `a 12 h turnover gives a nadir at 18.8 h; 6 h gives 14.2 h`
- watch out: The drug, its dose and its PK can be identical and the effect profile still changes completely with this one number. It is also why dosing intervals for these drugs often follow the biomarker rather than the plasma half life.

### baseline pins kin

- signature: `kin = kout * R0`
- what it does: At baseline production equals removal, so the production rate is not a free parameter once you know the baseline and kout.
- example: `kin = KOUT * R_BASE`
- watch out: Fitting R0, kin and kout together is fitting one parameter twice, and the fit will not converge cleanly. Also start the system AT the baseline, or it drifts there on its own and the model blames the drug.

### AUEC

- signature: `area between the baseline and the effect curve`
- what it does: Cumulative effect over time, which usually tracks clinical benefit better than the peak effect does.
- example: `auec = trapezoid(R_BASE - r, t)  gives  1481 units/L*h`
- watch out: Say which direction you integrated. Baseline minus curve makes suppression positive, the other way round makes it negative, and both appear in the literature.

### geometric mean and CV

- signature: `exp(mean(log x)) and sqrt(exp(var(log x)) - 1)`
- what it does: The standard summary for PK parameters, which are approximately lognormal. The geometric mean is the median of that distribution.
- example: `gm 4.198 L/h, CV 30.9 pct`
- watch out: Use the sample variance, ddof=1. The approximation CV is about omega is fine for small variability and wrong by several points above 50 percent, so know which form a reported number came from.

### arithmetic against geometric mean

- signature: `mean = median * exp(omega ** 2 / 2)`
- what it does: For a lognormal, the arithmetic mean sits above the median, by 4.6 percent at omega 0.3 and 20 percent at omega 0.6.
- example: `np.exp(0.30 ** 2 / 2)  gives  1.046`
- watch out: A mean clearance and a typical clearance are different quantities. Which one a report means is often not stated, and the gap grows fast with variability.

### between subject CV

- signature: `the spread of individual estimates on the log scale`
- what it does: How much the parameter varies between people, the omega of a population model.
- example: `100 * np.sqrt(np.exp(np.var(np.log(v_ind), ddof=1)) - 1)`
- watch out: From twelve subjects a true 31 percent comes back as anything between 18 and 44 percent, so do not compare a phase 1 CV with a phase 3 one and conclude the population changed.

### pooled against individual fitting

- signature: `one parameter set for everybody, or one each`
- what it does: Naive pooling fits a single curve through the whole study. Two stage fitting gives every subject their own parameters.
- example: `SSE 31.5 pooled against 1.37 individual, a 23 fold difference`
- watch out: Pooling hides between subject variability in the error term and biases the typical value, because the subjects with the highest concentrations dominate the sum of squares. Two stage recovers the variability but biases it upward.

### Monte Carlo study simulation

- signature: `simulate many studies, look at the spread of the answers`
- what it does: Instead of asking what the estimate is, ask what the estimate would have been across repeated studies of that size.
- example: `rng.lognormal(0.0, omega, size=(2000, 12)) then a CV per row`
- watch out: Draw once with a two dimensional size rather than looping, so the draw order is fixed and the result reproduces. One row is one study.

### NONMEM event records

- signature: `ID, TIME, AMT, DV, EVID, MDV, CMT`
- what it does: A time ordered list of events per subject, where an event is either something given or something measured. One row, one event.
- example: `a dosing row with AMT 100 and EVID 1, then observation rows with EVID 0`
- watch out: Rows are processed in order, so sort by ID then TIME, and put the dose FIRST at identical times. A pre dose sample processed before the dose runs without complaint and fits the wrong thing.

### EVID and MDV

- signature: `EVID 1 for a dose, 0 for an observation; MDV 1 means ignore DV`
- what it does: The two flags that carry the logic of a population dataset.
- example: `dosing rows always have MDV 1, since there is nothing to fit there`
- watch out: An observation row with MDV 1 is kept in the file and excluded from the fit, which is how missing and below limit samples are handled without losing the record of the time.

### BLQ handling

- signature: `M1 drop it, M3 treat it as censored, or substitute`
- what it does: A sample below the limit of quantification is not zero and not missing, it is the information that the value lies between zero and the LLOQ.
- example: `M1: set MDV to 1. NCA: a pre dose sample after an oral dose IS zero.`
- watch out: The same record can be MDV 1 for the model fit and a hard zero in the NCA of the same dataset. That is two analyses answering different questions, and it belongs in the analysis plan.

### probability of target attainment

- signature: `the fraction of a simulated population meeting a criterion`
- what it does: With a population rather than a typical subject, the answer is a fraction, not a number.
- example: `(mean_inhibition >= 0.55).mean()  gives  0.72`
- watch out: Efficacy and safety criteria move in opposite directions with dose, so the joint probability peaks in the middle and the best dose is almost never the one that maximises efficacy.

### joint criteria

- signature: `((eff >= target) & (cmax <= limit)).mean()`
- what it does: Counting the subjects who satisfy both criteria at once, rather than multiplying two marginal probabilities.
- example: `0.645 jointly, against 0.657 from multiplying the margins`
- watch out: The criteria are correlated: a low clearance raises both the average exposure, which helps efficacy, and the peak, which hurts safety. Multiplying margins overstates your chances.

## Error messages and what they really mean

### `NameError: name 'x' is not defined`

You never made x, or you ran the cells out of order. Run the notebook from the top.

### `KeyError: 'CONC'`

No such column. Check df.columns for a typo, different capitalisation, or a trailing space.

### `TypeError: 'NoneType' object is not subscriptable`

Something is None, usually a # YOUR CODE blank you have not filled in yet.

### `ValueError: operands could not be broadcast together with shapes (8,9) (8,)`

Two arrays do not line up. numpy matches shapes from the right. Usually wants keepdims=True or .reshape(-1, 1).

### `ValueError: Index contains duplicate entries, cannot reshape`

pivot found more than one row for the same index and column. Look at why before dropping or averaging them.

### `IndexError: index 9 is out of bounds for axis 0 with size 9`

A 9 long array has positions 0 to 8. Counting starts at 0.

### `AttributeError: 'DataFrame' object has no attribute 'xyz'`

Misspelled method, or a Series method called on a DataFrame.

### `SyntaxError: invalid syntax`

Usually a missing bracket or colon on the line ABOVE the one being pointed at.

### `Everything comes out nan`

One nan got into the maths and spread. Mask it out, or use the nan aware version such as np.nanmean.

### `SettingWithCopyWarning`

You changed something that may be a view of another DataFrame. Take an explicit .copy() first.

## House rules, enforced by the checks

### Error bars are SD, never SEM

SEM is the SD divided by the square root of n, so it shrinks as you add subjects even when the biology is identical. It describes how precisely you located the mean, not how much the subjects differ. A reader who knows n can work out the SEM from the SD. A reader shown the SEM cannot work backwards.

### SD means the sample SD, with n-1

numpy divides by n unless you pass ddof=1. Pandas already divides by n-1. Two libraries, two defaults, and nothing warns you, so say ddof=1 in numpy every time.

### Aggregate nested data to the animal first

Many cells inside one animal are repeat measures of that animal, not independent samples of the drug effect. Pooling them inflates n, weights each animal by however many cells you imaged, and makes the error bar answer a question nobody asked. Cells up to one number per animal, then animals up to a group mean and SD.

### Check len() before and after every merge

An inner join that drops an animal produces no error and a perfectly plausible answer. This one habit catches it.
