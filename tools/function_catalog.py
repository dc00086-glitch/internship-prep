"""Every function the problems use, with what it does and what to watch for.

This is the source for reference/function_reference.docx. To add a function,
add an entry here, then list its key in the FUNCTIONS list of whichever spec
introduces it. Rebuild with:

    python3 tools/build_glossary.py
"""

# group, key, signature shown, what it does, example, watch out for
ENTRIES = [

    # ---------------------------------------------------------------- Python
    ("Python", "len", "len(x)",
     "How many items are in a list, array, DataFrame or string.",
     "len(times)  gives  5",
     "On a DataFrame this counts rows, not cells. Print it before and after "
     "every merge."),

    ("Python", "round", "round(x, n)",
     "Rounds a number to n decimal places, for printing.",
     "round(7.48219, 2)  gives  7.48",
     "Round for display only. Never round the numbers you then do more maths "
     "on, or the error compounds."),

    ("Python", "f-string", 'f"text {value}"',
     "Puts values inside text. The part in braces is evaluated.",
     'f"half life {t_half:.2f} h"  gives  half life 7.48 h',
     "The :.2f inside the braces formats to 2 decimals without changing the "
     "underlying value."),

    ("Python", "type", "type(x)",
     "What kind of thing x is. The fastest way to find out why something is "
     "behaving oddly.",
     "type(conc)  gives  numpy.ndarray",
     "If you expected an array and got a list, or expected a DataFrame and got "
     "a Series, this is how you find out."),

    ("Python", "def", "def name(args):",
     "Defines a function, so you can write something once and reuse it.",
     "def t_half(k):\n    return np.log(2) / k",
     "Everything inside the function must be indented. The indentation is the "
     "syntax, not decoration."),

    ("Python", "try/except", "try: ... except ValueError as err:",
     "Runs code, and catches a specific error instead of stopping.",
     "try:\n    risky()\nexcept ValueError as err:\n    print(err)",
     "Catch the specific error you expect. A bare except hides real bugs."),

    # ---------------------------------------------------------------- numpy
    ("numpy", "np.array", "np.array([1.0, 2.0, 3.0])",
     "Makes an array from a list. An array does maths on every entry at once, "
     "which a plain list cannot.",
     "np.array([1, 2, 3]) * 2  gives  [2 4 6]",
     "A list would give [1, 2, 3, 1, 2, 3] instead. That difference is the "
     "whole reason numpy exists."),

    ("numpy", ".shape", "a.shape",
     "The size of an array, as (rows, columns) for 2D or (length,) for 1D.",
     "conc.shape  gives  (8, 9)",
     "The first thing to print when maths will not line up. Most numpy errors "
     "are shape errors wearing a disguise."),

    ("numpy", ".size", "a.size",
     "The total number of entries in an array.",
     "time_h.size  gives  11",
     "On a 2D array this is rows times columns, not either one on its own."),

    ("numpy", "np.zeros", "np.zeros(4)",
     "An array of zeros of the length you ask for. Useful as a starting point.",
     "np.zeros(4)  gives  [0. 0. 0. 0.]",
     ""),

    ("numpy", "np.arange", "np.arange(0, 5)",
     "Evenly spaced whole numbers, like range but as an array.",
     "np.arange(1, 5)  gives  [1 2 3 4]",
     "The end value is not included, same as Python slicing."),

    ("numpy", "np.linspace", "np.linspace(0, 1, 5)",
     "A set number of evenly spaced values between two endpoints. The usual "
     "way to make a smooth time grid for plotting a fitted curve.",
     "np.linspace(0, 1, 5)  gives  [0. 0.25 0.5 0.75 1.]",
     "Unlike arange, the end value IS included, and you say how many points "
     "you want rather than the step size."),

    ("numpy", "np.isnan", "np.isnan(a)",
     "Flags the missing values. Gives a True/False array, True where the entry "
     "is nan.",
     "np.isnan(conc)  gives  [False False True ...]",
     "True means MISSING. To keep the usable ones you almost always want "
     "~np.isnan(a)."),

    ("numpy", "~", "~mask",
     "Flips a True/False array. Every True becomes False and back.",
     "~np.isnan(conc)  gives  True where the sample IS usable",
     "This is for arrays. The word not does not work on an array, because "
     "numpy cannot reduce a whole array to one True or False."),

    ("numpy", "boolean indexing", "a[mask]",
     "Keeps only the entries where the mask is True. The main way filtering is "
     "done in numpy.",
     "c_meas = conc[~np.isnan(conc)]",
     "The mask must be the same length as the array. A mask built from one "
     "array works on another, which is how you get the times that go with the "
     "measurable concentrations."),

    ("numpy", "np.sum", "np.sum(a)",
     "Adds everything up. On a True/False array it counts the Trues, because "
     "True counts as 1.",
     "np.sum(np.isnan(conc))  gives  2",
     "Wrap it in int() when you want a plain whole number rather than a numpy "
     "integer."),

    ("numpy", "np.max / np.min", "np.max(a)",
     "The largest or smallest value in an array. Cmax is np.max of the "
     "measurable concentrations.",
     "np.max(c_meas)  gives  2.602",
     "If any entry is nan the answer is nan. Either mask them out first, or "
     "use np.nanmax."),

    ("numpy", "np.argmax", "np.argmax(a)",
     "The POSITION of the largest value, counting from 0. This is how Tmax is "
     "found: get the position of Cmax, then look that position up in the time "
     "array.",
     "t_meas[np.argmax(c_meas)]  gives  3.0",
     "It returns a position, not a value. Reporting the position as if it were "
     "a time is a very common slip."),

    ("numpy", "np.nanmax", "np.nanmax(a)",
     "Like np.max but ignores nan. There is a nan version of most summaries: "
     "nanmean, nanstd, nansum.",
     "np.nanmax(conc)  ignores the BLQ samples",
     "Convenient, but masking first is usually clearer, because then every "
     "later step is working on the same clean data."),

    ("numpy", "np.mean", "a.mean(axis=0)",
     "The average. On a 2D array, axis says which direction to collapse.",
     "conc.mean(axis=0)  gives one value per timepoint",
     "The axis you name is the one that DISAPPEARS. axis=0 leaves one value "
     "per column, axis=1 leaves one value per row."),

    ("numpy", "np.median", "np.median(a)",
     "The middle value. Less sensitive to one extreme point than the mean.",
     "np.median(vals)  gives  15.5",
     "Report whichever you decided on in advance. Switching to the one that "
     "looks better after seeing the data is not an analysis choice."),

    ("numpy", "np.std", "a.std(axis=0, ddof=1)",
     "The standard deviation, which is the spread of the data.",
     "conc.std(axis=0, ddof=1)",
     "numpy divides by n unless you pass ddof=1, which divides by n-1 and is "
     "the SAMPLE SD you almost always want. Pandas does the opposite and "
     "already uses n-1. This inconsistency causes quietly wrong numbers."),

    ("numpy", "keepdims", "a.max(axis=1, keepdims=True)",
     "Keeps the collapsed direction as a length 1 axis, so the result still "
     "lines up with the original array for broadcasting.",
     "conc / conc.max(axis=1, keepdims=True)",
     "Without it you get shape (8,) which will not divide into (8, 9). This is "
     "the fix for the broadcast error below."),

    ("numpy", ".reshape", "a.reshape(-1, 1)",
     "Changes the shape without changing the data. -1 means work it out.",
     "row_max.reshape(-1, 1)  turns (8,) into (8, 1)",
     "Does the same job as keepdims here. Use whichever reads more clearly."),

    ("numpy", "broadcasting", "a / b",
     "Doing maths between arrays of different shapes. numpy compares shapes "
     "from the RIGHT and stretches length 1 axes to fit.",
     "(8, 9) and (9,) line up. (8, 9) and (8,) do not.",
     "The error operands could not be broadcast together means the shapes "
     "failed that check. Usually the fix is keepdims=True."),

    ("numpy", "np.log", "np.log(a)",
     "The natural log, ln. Turns an exponential decay into a straight line, "
     "which is how terminal slopes are fitted.",
     "np.log(c_term)",
     "np.log is ln. np.log10 is base 10. Mixing them up changes lambda_z by a "
     "factor of 2.303."),

    ("numpy", "np.sqrt", "np.sqrt(a)",
     "The square root.",
     "sem = sd / np.sqrt(n)",
     "If you find yourself writing this when making an error bar, stop. That "
     "is the SEM, and this repo plots SD."),

    ("numpy", "np.polyfit", "np.polyfit(x, y, deg=1)",
     "Fits a polynomial. With deg=1 it is a straight line, and it returns "
     "[slope, intercept] in that order.",
     "slope, intercept = np.polyfit(t_term, np.log(c_term), deg=1)",
     "lambda_z is the NEGATIVE of the slope, because the slope is negative "
     "when the drug is leaving. Fit only the terminal points: including the "
     "absorption phase biases the slope badly."),

    ("numpy", "np.round", "np.round(a, 3)",
     "Rounds every entry, for printing.",
     "np.round(cv_by_time, 1)",
     "For display only, same as the built-in round."),

    ("numpy", "np.where", "np.where(cond, a, b)",
     "Picks from a where the condition is True and from b where it is False. "
     "Also useful for turning labels into x positions.",
     'np.where(df["GROUP"] == "Drug", 0, 1)',
     "With one argument it does something different, returning the positions "
     "where the condition is True."),

    ("numpy", "np.repeat / np.tile", "np.repeat(a, 3)",
     "Builds up long columns. repeat stretches each element in place, tile "
     "repeats the whole sequence.",
     "np.repeat([1, 2], 2) gives [1 1 2 2], np.tile([1, 2], 2) gives [1 2 1 2]",
     "The usual way to build a long format table: repeat the subject ids, tile "
     "the timepoints."),

    ("numpy", "np.random.default_rng", "rng = np.random.default_rng(1101)",
     "A random number generator you can reproduce. The number is the seed, and "
     "the same seed always gives the same numbers.",
     "rng.normal(0, 1, 4), rng.lognormal(0, 0.15, 8)",
     "Every problem notebook seeds this, which is why your numbers match the "
     "checker. The older np.random.seed style still works but this is the "
     "current way."),

    # ---------------------------------------------------------------- pandas
    ("pandas", "pd.DataFrame", 'pd.DataFrame({"A": [1, 2]})',
     "Makes a table from a dictionary of columns. Each column has a name and "
     "its own type.",
     'pd.DataFrame({"ANIMAL": ids, "CONC": vals})',
     "A DataFrame is really a dictionary of columns, not a grid of cells. That "
     "explains most of its behaviour."),

    ("pandas", ".head", "df.head(5)",
     "The first few rows. The first thing to run on any table you did not make "
     "yourself.",
     "pk.head(10)",
     ""),

    ("pandas", ".columns", "df.columns",
     "The column names. Check here first when you get a KeyError.",
     "list(pk.columns)",
     "Column names are case sensitive, and a trailing space in a name is "
     "invisible and will bite you."),

    ("pandas", "selecting a column", 'df["CONC"]',
     "One column, which comes back as a Series. Maths on it is elementwise, "
     "like numpy.",
     'pk["CONC_NG_ML"].mean()',
     'Double brackets, df[["A", "B"]], gives a DataFrame of several columns. '
     "Single brackets give one Series. The difference matters downstream."),

    ("pandas", "boolean filtering", 'df[df["TIME_H"] > 0]',
     "Keeps the rows where the condition is True.",
     'df_post = pk[pk["TIME_H"] > 0]',
     "Combine conditions with & for and, | for or, and put brackets around "
     "each one. The words and/or do not work here."),

    ("pandas", "adding a column", 'df["NEW"] = df["A"] / df["B"]',
     "Creates a column from other columns, elementwise.",
     'm["DOSE_MG_PER_KG"] = m["DOSE_MG"] / (m["BODYWEIGHT_G"] / 1000)',
     "Work on a .copy() if the original table still matters, or you may get a "
     "SettingWithCopyWarning."),

    ("pandas", ".nunique / .unique", 'df["ANIMAL"].nunique()',
     "How many distinct values, and what they are.",
     'pk["ANIMAL"].nunique()  gives  12',
     "Counting rows is not counting animals. Each animal appears once per "
     "timepoint."),

    ("pandas", ".isna", 'df["COL"].isna().sum()',
     "Flags the missing entries, and summing counts them.",
     'int(merged["BODYWEIGHT_G"].isna().sum())',
     "Run this after every left merge. It is how you find out what did not "
     "match."),

    ("pandas", ".groupby", 'df.groupby(["GROUP", "TIME_H"])',
     "Split, apply, combine. Splits the table into one chunk per group, so you "
     "can summarise each chunk.",
     'df_post.groupby(["GROUP", "TIME_H"])["CONC"].agg(["mean", "std"])',
     "The grouping keys come back in the INDEX, not as columns. Follow with "
     ".reset_index() or downstream code will not find them."),

    ("pandas", ".agg", '.agg(["mean", "std"])',
     "The summaries to compute on each group. Useful ones: mean, std, size, "
     "count, min, max, median, sum.",
     '.agg(MEAN=("VAL", "mean"), N=("VAL", "size"))',
     "The named form on the right also names the output columns, which saves a "
     "rename. size counts every row, count skips the missing ones."),

    ("pandas", ".reset_index", ".reset_index()",
     "Moves index levels back out into ordinary columns.",
     ".groupby(...).agg(...).reset_index()",
     "Use .reset_index(drop=True) after sorting, to renumber 0, 1, 2 and "
     "throw the old numbers away rather than keeping them as a column."),

    ("pandas", ".rename", '.rename(columns={"mean": "CONC_MEAN"})',
     "Renames columns.",
     '.rename(columns={"mean": "CONC_MEAN", "std": "CONC_SD"})',
     ""),

    ("pandas", ".sort_values", '.sort_values(["ANIMAL", "TIME_H"])',
     "Sorts by one or more columns, in the order you list them.",
     '.sort_values(["ANIMAL", "TIME_H"]).reset_index(drop=True)',
     "Sort on the NUMBER, not the text. Text sorting puts T_24 before T_4."),

    ("pandas", "pd.merge", 'pd.merge(a, b, on="ANIMAL", how="left")',
     "Joins two tables on a shared column.",
     'pd.merge(pk, demo, on="ANIMAL", how="left")',
     "how='left' keeps every row of the left table and shows gaps as NaN. "
     "how='inner' is the DEFAULT and silently drops rows that did not match. "
     "Print len() before and after, every time. Rows can also go up, if the "
     "right table has more than one row per key."),

    ("pandas", ".melt", 'df.melt(id_vars=["ANIMAL"], var_name=..., value_name=...)',
     "Wide to tidy. Folds a set of columns into one column of names and one of "
     "values, so the timepoint becomes data rather than a column heading.",
     'wide.melt(id_vars=["ANIMAL", "GROUP"], var_name="TIME_LABEL", value_name="INHIB_PCT")',
     "id_vars lists the columns to LEAVE ALONE. Everything else gets folded. "
     "Without var_name and value_name you get columns called variable and "
     "value."),

    ("pandas", ".pivot", 'df.pivot(index=..., columns=..., values=...)',
     "Tidy back to wide. The inverse of melt.",
     'tidy.pivot(index="ANIMAL", columns="TIME_H", values="INHIB_PCT")',
     "Fails with Index contains duplicate entries if two rows claim the same "
     "index and column. That error is the data telling you something, so look "
     "before you silence it. pivot_table averages duplicates instead, which is "
     "right only if averaging them is scientifically right."),

    ("pandas", ".duplicated", "df.duplicated().sum()",
     "Flags rows that repeat a row already seen, and summing counts them.",
     "int(tidy_clean.duplicated().sum())",
     "By default the FIRST occurrence is not flagged, only the repeats. "
     "keep=False flags both copies, which gives double the count."),

    ("pandas", ".drop_duplicates", "df.drop_duplicates()",
     "Removes the repeated rows, keeping the first of each.",
     "tidy_clean.drop_duplicates().pivot(...)",
     "Only safe when the rows really are the same record twice. If the values "
     "differ it is a scientific question, not a cleaning step."),

    ("pandas", ".drop", 'df.drop(columns=["TIME_LABEL"])',
     "Removes columns, or rows with index=.",
     '.drop(columns=["TIME_LABEL"])',
     ""),

    ("pandas", ".str accessor", 'df["COL"].str.replace("T_", "", regex=False)',
     "Text operations on a whole column. Also .str.strip, .str.upper, "
     ".str.contains, .str.split.",
     '.str.replace("T_", "", regex=False).astype(float)',
     "regex=False when you mean the characters literally. Without it, "
     "characters like . and ( are treated as pattern syntax."),

    ("pandas", ".astype", '.astype(float)',
     "Converts a column to another type, usually text to number.",
     'df["TIME_H"] = df["TIME_LABEL"].str.replace("T_", "").astype(float)',
     "Fails loudly if any value cannot convert, which is useful. "
     "pd.to_numeric(..., errors='coerce') turns the bad ones into NaN instead."),

    ("pandas", "pd.concat", "pd.concat([a, b], ignore_index=True)",
     "Stacks tables on top of each other.",
     "pd.concat(frames, ignore_index=True)",
     "ignore_index=True renumbers the result. Without it you get repeated "
     "index values, which breaks later lookups."),

    ("pandas", ".to_numpy", 'df["CONC"].to_numpy()',
     "Hands you the plain numpy array behind a column, when you want numpy "
     "functions.",
     'np.polyfit(df["T"].to_numpy(), df["C"].to_numpy(), 1)',
     ""),

    # ------------------------------------------------------------ matplotlib
    ("matplotlib", "plt.subplots", "fig, ax = plt.subplots(figsize=(6, 4.5))",
     "Makes a figure and its axes. fig is the whole picture, ax is the one set "
     "of axes you draw on.",
     "fig, ax = plt.subplots(figsize=(6, 4.5))",
     "Always build plots this explicit way. plt.plot on its own draws on "
     "whatever figure is current, which is hidden state that falls apart as "
     "soon as a notebook has more than one figure."),

    ("matplotlib", "plt.subplots(rows, cols)", "fig, axes = plt.subplots(1, 2)",
     "Several panels. You get an ARRAY of axes, reached as axes[0], axes[1].",
     "fig, axes = plt.subplots(1, 2, figsize=(10, 4))",
     "With rows and columns you must index into axes. Without them you get a "
     "single axes that you must not index."),

    ("matplotlib", "ax.plot", "ax.plot(x, y, marker='o')",
     "A line, optionally with markers.",
     'ax.plot(time_h, conc, marker="o")',
     ""),

    ("matplotlib", "ax.errorbar", "ax.errorbar(x, y, yerr=sd, fmt='o')",
     "Points with error bars.",
     'ax.errorbar([0, 1], g["MEAN"], yerr=g["SD"], fmt="o", capsize=6)',
     "In this repo yerr is ALWAYS an SD. fmt='o' with linestyle='none' gives "
     "markers and no connecting line. A yerr of shape (2, n) is read as "
     "separate lower and upper bars."),

    ("matplotlib", "ax.scatter", "ax.scatter(x, y)",
     "Scattered points. The way to overlay every animal on top of a group "
     "summary.",
     'ax.scatter(animal_x, animal_level["MEAN_INTENSITY"], s=45, alpha=0.75)',
     "s is the marker size and alpha is the transparency, which helps when "
     "points overlap."),

    ("matplotlib", "zorder", "zorder=3",
     "What gets drawn on top. Higher numbers are nearer the front.",
     "ax.errorbar(..., zorder=3) over ax.scatter(..., zorder=2)",
     "Without it the scattered points can hide the summary markers."),

    ("matplotlib", "axis labels", "ax.set_xlabel / set_ylabel / set_title",
     "Labels the axes and titles the panel. Include units in the y label.",
     'ax.set_ylabel("Concentration (mg/L)")',
     "A figure without all three is unreadable to everyone except the person "
     "who made it, including you in three weeks."),

    ("matplotlib", "ax.set_xticks", "ax.set_xticks([0, 1])",
     "Where the ticks go. Pair with set_xticklabels to name them.",
     'ax.set_xticks([0, 1])\nax.set_xticklabels(["Vehicle", "Drug"])',
     "Set the ticks before the labels, or the labels can land in the wrong "
     "places."),

    ("matplotlib", "ax.set_yscale", 'ax.set_yscale("log")',
     "A log axis. A PK profile that curves on a linear axis becomes a straight "
     "terminal line on a semi log axis, which is how you eyeball lambda_z.",
     'ax.set_yscale("log")',
     "Zero and negative values cannot be shown on a log axis and silently "
     "vanish, including your predose samples."),

    ("matplotlib", "ax.legend", "ax.legend(frameon=False)",
     "Shows the legend, built from the label= you passed to each plotting "
     "call.",
     'ax.errorbar(..., label="group mean, SD")\nax.legend(frameon=False)',
     "Nothing appears unless you gave the plotting calls a label."),

    ("matplotlib", "fig.tight_layout", "fig.tight_layout()",
     "Tidies up spacing so labels do not overlap or get cut off.",
     "fig.tight_layout()",
     "Call it after everything is drawn, just before showing or saving."),

    ("matplotlib", "plt.show", "plt.show()",
     "Displays the figure.",
     "plt.show()",
     "In a notebook the figure usually appears anyway, but calling it is "
     "explicit and stops a stray text repr printing above your plot."),
# ---------------------------------------------------- added for week 2
    ("Python", "sorted", "sorted(x)",
     "Returns a new sorted list. Works on the result of .unique(), which comes "
     "back in whatever order pandas found the values.",
     'sorted(df["SUBJECT"].unique())',
     "Sorting text compares character by character, so '10' lands before '9'. "
     "Convert to numbers first when the order has to be numeric."),

    ("pandas", ".transform", 'df.groupby("ID")["V"].transform("mean")',
     "Computes per group like agg does, then broadcasts each group's answer "
     "back over every row of that group. The result has the same length as the "
     "input, so it drops straight into a new column.",
     'df["BASELINE"] = df.groupby("SUBJECT")["V"].transform("first")',
     "transform('first') takes whatever row is physically first in each group. "
     "It does not know what your time column means, so sort before you use it."),

    ("pandas", ".all / .any", "(a == b).all()",
     "Reduces a column of True and False to a single True or False. .all() asks "
     "whether every value is True, .any() whether at least one is.",
     '(at_zero["VALUE"] == at_zero["BASELINE"]).all()',
     "Comparing two columns gives a Series, not one answer, so an if on it "
     "raises 'The truth value of a Series is ambiguous'. Finish with .all()."),

    ("pandas", "ignore_index", "df.sort_values([...], ignore_index=True)",
     "Renumbers the index 0, 1, 2 as part of the sort, instead of carrying the "
     "old row labels along.",
     'df.sort_values(["SUBJECT", "TIME_H"], ignore_index=True)',
     "Exactly the same as following the sort with .reset_index(drop=True)."),

    ("pandas", "validate=", 'pd.merge(a, b, on="K", how="left", validate="m:1")',
     "States the relationship you believe the keys have, and raises a MergeError "
     "if it is false. 'm:1' means the right hand key must be unique.",
     'pd.merge(pk, dosing, on="SUBJECT", how="left", validate="m:1")',
     "This is the cheapest protection against a merge that silently duplicates "
     "rows. Without it a repeated key doubles your data with no warning."),

    ("pandas", "indicator=", "pd.merge(a, b, on=..., how=..., indicator=True)",
     "Adds a _merge column saying whether each row came from left_only, "
     "right_only or both, so you can see exactly what failed to match.",
     'checked[checked["_merge"] == "left_only"]',
     "Put the table you want to account for on the LEFT. Grid on the left "
     "answers what is missing; data on the left answers what was unexpected."),

    ("pandas", ".value_counts", 'df["COL"].value_counts()',
     "Counts how many times each distinct value appears, largest first.",
     'checked["_merge"].value_counts()',
     "Add dropna=False when you want the NaN count shown rather than skipped."),

    ("pandas", "pd.MultiIndex.from_product",
     'pd.MultiIndex.from_product([a, b], names=[...]).to_frame(index=False)',
     "Every combination of two or more lists, which is how you build the set of "
     "rows that SHOULD exist. Join your data onto it and the gaps become rows.",
     'pd.MultiIndex.from_product([SUBJECTS, TIMES], names=["SUBJECT", "TIME_H"])',
     "Absence is not a value, so no amount of filtering the rows you have will "
     "find a row that was never collected. This is the way round that."),

    ("pandas", ".abs", 'df["DEV"].abs()',
     "The size of each value, ignoring sign.",
     'out = df["DEV_MIN"].abs() > tolerance',
     "Take it only when you compare, not when you store. A systematic lateness "
     "is a different finding from random scatter, and abs() hides which it is."),

    ("pandas", ".fillna", 'df["N"].fillna(0)',
     "Replaces missing values. The usual finish after a left join that left "
     "gaps where a group legitimately had nothing.",
     'df["N_MISSING"] = df["N_MISSING"].fillna(0).astype(int)',
     "Only fill where zero is genuinely the right answer. Filling a missing "
     "measurement with 0 invents data."),

    ("pandas", ".assign", 'df.assign(ABS_DEV=df["DEV"].abs())',
     "Returns a copy with a new column added, so it chains without touching the "
     "original frame.",
     '.assign(ABS_DEV=df["DEV_MIN"].abs()).groupby("SUBJECT").agg(...)',
     ""),

    ("pandas", "pd.crosstab", 'pd.crosstab(df["A"], df["B"])',
     "A frequency table: counts how many ROWS fall into each combination of two "
     "columns, one down the rows and one across the columns.",
     'pd.crosstab(ae["GROUP"], ae["SEVERITY"], margins=True, margins_name="Total")',
     "It counts rows. If one subject contributed six rows, they count six "
     "times, which is rarely what a safety table should show."),

    ("pandas", "normalize=", 'pd.crosstab(a, b, normalize="index")',
     "Turns counts into proportions. 'index' makes each row sum to 1, 'columns' "
     "each column, 'all' the whole table. Multiply by 100 for percent.",
     'pd.crosstab(a, b, normalize="index") * 100',
     "Say out loud what you want each row to add up to before you choose, "
     "because index and columns answer genuinely different questions."),

    ("pandas", "pd.pivot_table",
     "pd.pivot_table(df, index=..., columns=..., values=..., aggfunc=...)",
     "The general form of crosstab. Takes a values column and any aggregation, "
     "so it gives means, maxima or counts of distinct subjects, not just counts.",
     'pd.pivot_table(ae, index="GROUP", columns="SEVERITY", values="SUBJECT", aggfunc="nunique")',
     "When you want anything other than a row count, you have outgrown crosstab."),

    ("pandas", "size versus nunique", '.agg("size") against .agg("nunique")',
     "size counts rows, so it counts events. nunique counts distinct values, so "
     "on a subject column it counts people.",
     'ae.groupby("GROUP")["SUBJECT"].nunique()',
     "The difference between an event count and a patient incidence. Also watch "
     "the denominator: subjects with no events are absent from an event log, so "
     "it must come from the enrolment table."),

    ("pandas", ".reindex", '.reindex(["Placebo", "Low", "High"])',
     "Puts the rows in an order you choose, rather than the alphabetical order "
     "groupby and crosstab give you.",
     "ae_counts.reindex(GROUP_ORDER + [\"Total\"])",
     "A label not present in the data comes back as a row of NaN rather than an "
     "error, which is sometimes what you want and sometimes a silent typo."),

    ("pandas", ".map", 'df["G"].map({"Low": 0, "High": 1})',
     "Translates each value through a dictionary. Handy for forcing a custom "
     "sort order or turning labels into x positions.",
     'df["_ord"] = df["GROUP"].map({g: i for i, g in enumerate(GROUP_ORDER)})',
     "Values missing from the dictionary become NaN, which then sort to the end."),

    ("pandas", "looping over groupby", 'for name, part in df.groupby("G"):',
     "Hands you each group's name and that group's rows in turn. The tidy way "
     "to draw one line or one series per group.",
     'for name, part in summary.groupby("DOSE"):\n    ax.errorbar(part["T"], part["M"], yerr=part["SD"], label=name)',
     "Use it for plotting and for anything genuinely per group. For computing "
     "statistics, agg and transform are faster and clearer."),
]



ERRORS = [
    ("NameError: name 'x' is not defined",
     "You never made x, or you ran the cells out of order. Run the notebook "
     "from the top."),
    ("KeyError: 'CONC'",
     "No such column. Check df.columns for a typo, different capitalisation, "
     "or a trailing space."),
    ("TypeError: 'NoneType' object is not subscriptable",
     "Something is None, usually a # YOUR CODE blank you have not filled in "
     "yet."),
    ("ValueError: operands could not be broadcast together with shapes (8,9) (8,)",
     "Two arrays do not line up. numpy matches shapes from the right. Usually "
     "wants keepdims=True or .reshape(-1, 1)."),
    ("ValueError: Index contains duplicate entries, cannot reshape",
     "pivot found more than one row for the same index and column. Look at why "
     "before dropping or averaging them."),
    ("IndexError: index 9 is out of bounds for axis 0 with size 9",
     "A 9 long array has positions 0 to 8. Counting starts at 0."),
    ("AttributeError: 'DataFrame' object has no attribute 'xyz'",
     "Misspelled method, or a Series method called on a DataFrame."),
    ("SyntaxError: invalid syntax",
     "Usually a missing bracket or colon on the line ABOVE the one being "
     "pointed at."),
    ("Everything comes out nan",
     "One nan got into the maths and spread. Mask it out, or use the nan aware "
     "version such as np.nanmean."),
    ("SettingWithCopyWarning",
     "You changed something that may be a view of another DataFrame. Take an "
     "explicit .copy() first."),
]


RULES = [
    ("Error bars are SD, never SEM",
     "SEM is the SD divided by the square root of n, so it shrinks as you add "
     "subjects even when the biology is identical. It describes how precisely "
     "you located the mean, not how much the subjects differ. A reader who "
     "knows n can work out the SEM from the SD. A reader shown the SEM cannot "
     "work backwards."),
    ("SD means the sample SD, with n-1",
     "numpy divides by n unless you pass ddof=1. Pandas already divides by "
     "n-1. Two libraries, two defaults, and nothing warns you, so say ddof=1 "
     "in numpy every time."),
    ("Aggregate nested data to the animal first",
     "Many cells inside one animal are repeat measures of that animal, not "
     "independent samples of the drug effect. Pooling them inflates n, weights "
     "each animal by however many cells you imaged, and makes the error bar "
     "answer a question nobody asked. Cells up to one number per animal, then "
     "animals up to a group mean and SD."),
    ("Check len() before and after every merge",
     "An inner join that drops an animal produces no error and a perfectly "
     "plausible answer. This one habit catches it."),
]
