PID = "W4D3"
WEEK = 4
DAY = 3
SLUG = "w4d3_skimage_regionprops"
TITLE = "Labelling objects and measuring them one by one"
TOPIC = "scikit-image: connected components, regionprops_table, filtering by measurement"
EST = "20-30 min"

SCENARIO_MD = '''
## Scenario

You have a clean binary mask and the original intensity image behind it. Up to
now every number has described the whole field: how many foreground pixels,
what fraction of the frame. That is not what anyone wants.

The question is about **cells**: how many are there, how big are they, how
round are they, and how bright is each one. To answer any of that, the mask
has to stop being a sheet of True and False and become a list of objects, each
with an identity you can refer to.

That step is called **connected component labelling**, and everything per cell
follows from it.

This field also contains two problems that labelling alone will not solve. A
small piece of debris got through the cleanup, and two cells are touching, so
they label as a single object with roughly twice the area. Both are visible in
the measurements, which is how you deal with them.
'''

CONCEPT_MD = '''
## From a mask to a table of objects

**Labelling.** `measure.label(mask)` returns an integer array the same shape
as the mask. Background is 0, the first object is 1, the second is 2, and so
on. Pixels get the same number exactly when they are connected to each other
through the foreground. So `labels == 3` is a boolean mask of the third
object, and `labels.max()` is how many objects there are.

Connectivity is worth knowing about. In 2D, `connectivity=1` means pixels
count as connected only through their four edge neighbours, while
`connectivity=2`, the default, also counts the four diagonals. Diagonal
connectivity joins objects that touch only at a corner, which is usually what
you want for cells and occasionally not.

**Measuring.** `measure.regionprops_table` walks the labelled array and
returns a dictionary of arrays, one row per object, which goes straight into a
DataFrame. Pass `intensity_image=` as well and you also get intensity
measurements taken over each object's own pixels.

The properties worth knowing here:

- `area`, the pixel count of the object,
- `eccentricity`, from 0 for a perfect circle up to 1 for a line, which is how
  you tell round cells from elongated ones or from merged pairs,
- `mean_intensity`, the average of the original image over that object,
- `centroid`, which comes back as two columns, `centroid-0` for the row and
  `centroid-1` for the column.

Note the units. `area` is in **pixels**, not square microns, until you
multiply by the pixel size, and forgetting to is how an area ends up off by a
factor of ten thousand.

**Filtering by measurement rather than by morphology.** Yesterday you removed
debris geometrically, with an opening. Here you remove it statistically: label
everything, measure everything, then keep the rows whose area falls in a
plausible range. This is often better, because the criterion is explicit and
visible in a table rather than buried in a footprint radius, and because you
can filter on any property, not only size.

An upper bound matters as much as a lower one. Two cells that touch label as
one object with about double the area and a stretched shape. Dropping those is
honest, and it undercounts. Splitting them properly needs a watershed, which
is a different and larger topic. Knowing that your count is a slight
underestimate, and why, beats a number you cannot account for.
'''

TASK_MD = '''
## Your task

Three parts, worked one at a time.

| Part | What you work out | Variables to fill in |
|---|---|---|
| a | the mask turned into labelled objects and a table | `labels`, `n_objects`, `props` |
| b | debris and merged pairs filtered out by area | `cells`, `n_cells`, `n_dropped` |
| c | the per cell summary | `mean_area`, `mean_ecc`, `mean_intensity` |

Below this you will find the **Setup** cell, then a **Toolkit** cell that
demonstrates every move on a tiny made up example, then **Part a**, **Part b**
and **Part c**, each with its own explanation and a cell to write in.
'''

DATA_CODE = '''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import draw, measure, morphology, segmentation
from scipy import ndimage as ndi

PROBLEM_ID = "W4D3"
TOPIC = "scikit-image: connected components, regionprops_table, filtering by measurement"

IMG_SIZE = 256
MIN_AREA = 150      # anything smaller is debris
MAX_AREA = 900      # anything larger is two cells that touched

rng = np.random.default_rng(4301)

_img = np.full((IMG_SIZE, IMG_SIZE), 40.0)
_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=bool)

# Twelve well separated cells, with brightness varying between them.
_centres = [(45, 40), (50, 120), (40, 200), (110, 65), (120, 150), (105, 225),
            (180, 45), (175, 130), (195, 215), (240, 90), (235, 180), (150, 20)]
for _cy, _cx in _centres:
    # real cells are not perfect circles, so draw ellipses at random angles
    _ra = float(rng.uniform(11.0, 14.5))
    _rb = float(rng.uniform(8.5, 13.0))
    _rot = float(rng.uniform(0.0, np.pi))
    _rr, _cc = draw.ellipse(_cy, _cx, _ra, _rb, shape=_img.shape, rotation=_rot)
    _mask[_rr, _cc] = True
    _img[_rr, _cc] += 110.0 * rng.uniform(0.7, 1.35)

# Two cells that happen to touch, so they will label as one object.
for _cy, _cx in [(70, 162), (70, 188)]:
    _rr, _cc = draw.ellipse(_cy, _cx, 12.5, 13.5, shape=_img.shape)
    _mask[_rr, _cc] = True
    _img[_rr, _cc] += 120.0

# One piece of debris that survived the cleanup.
_rr, _cc = draw.disk((205, 140), 5, shape=_img.shape)
_mask[_rr, _cc] = True
_img[_rr, _cc] += 95.0

_img += rng.normal(0.0, 5.0, size=_img.shape)
field = np.clip(_img, 0, 255).astype(np.uint8)
mask = _mask

print("field:", field.shape, field.dtype, "   mask foreground px:", int(mask.sum()))
print("area limits: keep objects between %d and %d pixels" % (MIN_AREA, MAX_AREA))
print("")
print("twelve separated cells, one touching pair, and one speck of debris")

_fig, _axes = plt.subplots(1, 2, figsize=(9, 4.3))
_axes[0].imshow(field, cmap="gray", vmin=0, vmax=255); _axes[0].set_title("field")
_axes[1].imshow(mask, cmap="gray"); _axes[1].set_title("mask")
for _a in _axes: _a.axis("off")
_fig.tight_layout()
plt.show()
'''

TOOLKIT_MD = '''
## Toolkit for this problem

| What you want to do | How |
|---|---|
| number the objects | `labels = measure.label(mask)` |
| how many there are | `int(labels.max())` |
| one object's pixels | `labels == 3` |
| measure them all | `measure.regionprops_table(labels, intensity_image=img, properties=(...))` |
| that as a table | `pd.DataFrame(measure.regionprops_table(...))` |
| keep rows in a range | `df[(df["area"] >= lo) & (df["area"] <= hi)]` |
| a column mean | `float(df["area"].mean())` |

`regionprops_table` returns a dictionary of numpy arrays, so `pd.DataFrame`
around it gives you an ordinary table. Properties that have more than one
number, such as `centroid`, arrive as several columns named `centroid-0` and
`centroid-1`.
'''

TOOLKIT_CODE = '''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import draw, measure

# A tiny made up mask: two discs, one ellipse, one speck. Nothing here is the answer.
demo_img = np.full((60, 60), 30.0)
demo_mask = np.zeros((60, 60), dtype=bool)
demo_rr, demo_cc = draw.disk((15, 15), 9, shape=demo_mask.shape)
demo_mask[demo_rr, demo_cc] = True;  demo_img[demo_rr, demo_cc] += 120.0
demo_rr, demo_cc = draw.disk((15, 45), 7, shape=demo_mask.shape)
demo_mask[demo_rr, demo_cc] = True;  demo_img[demo_rr, demo_cc] += 60.0
demo_rr, demo_cc = draw.ellipse(42, 30, 6, 16, shape=demo_mask.shape)
demo_mask[demo_rr, demo_cc] = True;  demo_img[demo_rr, demo_cc] += 90.0
demo_mask[50, 5] = True
demo_img = demo_img.astype(np.uint8)

print("--- 1. labelling numbers the objects ---")
demo_labels = measure.label(demo_mask)
print("labels dtype:", demo_labels.dtype, " max:", int(demo_labels.max()))
print("distinct values:", np.unique(demo_labels).tolist(), " <- 0 is background")
print("pixels in object 1:", int((demo_labels == 1).sum()))

print("")
print("--- 2. regionprops_table measures every object at once ---")
demo_props = pd.DataFrame(measure.regionprops_table(
    demo_labels, intensity_image=demo_img,
    properties=("label", "area", "eccentricity", "mean_intensity", "centroid")))
print(demo_props.round(3))
print("")
print("note centroid became two columns, centroid-0 is the ROW and centroid-1 the column")

print("")
print("--- 3. eccentricity tells round from stretched ---")
for _, demo_row in demo_props.iterrows():
    demo_shape = "round" if demo_row["eccentricity"] < 0.6 else "elongated"
    print("  object %d: area %5.0f  ecc %.3f  -> %s"
          % (demo_row["label"], demo_row["area"], demo_row["eccentricity"], demo_shape))

print("")
print("--- 4. filtering by measurement ---")
demo_keep = demo_props[(demo_props["area"] >= 50) & (demo_props["area"] <= 400)]
print("objects before filtering:", len(demo_props))
print("objects after filtering :", len(demo_keep))
print("the single pixel speck is gone")

demo_fig, demo_ax = plt.subplots(figsize=(3.4, 3.4))
demo_ax.imshow(demo_labels, cmap="nipy_spectral")
demo_ax.set_title("labels, one colour each")
demo_ax.axis("off")
plt.show()
'''

PARTS = [
    dict(
        label="a",
        prompt_md='''
## Part a. Label the objects and measure them

`measure.label(mask)` turns the boolean mask into an integer array: background
0, first object 1, second object 2, and so on. The number of objects is its
maximum.

Then measure them all at once with `measure.regionprops_table`. Pass the
labelled array, pass `intensity_image=field` so intensity measurements are
taken over each object's own pixels, and ask for these properties:

`"label"`, `"area"`, `"eccentricity"`, `"mean_intensity"`

Wrap the result in `pd.DataFrame` to get an ordinary table with one row per
object.

Look at the table before moving on. You should see twelve or so rows with
similar areas, one row with a very small area, and one with about double the
typical area and a noticeably higher eccentricity. Those last two are the
debris and the touching pair, and spotting them in the numbers is the point.

Fill in `labels`, `n_objects`, `props`.

Moves: `measure.label`, `.max()`, `measure.regionprops_table(...,
intensity_image=..., properties=(...))`, `pd.DataFrame`.
''',
        starter='''
# Part a. Label the objects, then measure every one of them.

labels = None      # YOUR CODE: the labelled integer array
n_objects = None   # YOUR CODE: how many objects there are
props = None       # YOUR CODE: DataFrame with label, area, eccentricity, mean_intensity

print("n_objects:", n_objects)
print(None if props is None else props.round(3))
''',
        solution='''
# Part a. Label the objects, then measure every one of them.

labels = measure.label(mask)
n_objects = int(labels.max())
props = pd.DataFrame(measure.regionprops_table(
    labels, intensity_image=field,
    properties=("label", "area", "eccentricity", "mean_intensity")))

print("n_objects:", n_objects)
print(None if props is None else props.round(3))
''',
    ),
    dict(
        label="b",
        prompt_md='''
## Part b. Filter out the debris and the merged pair

Keep only the rows whose `area` is between `MIN_AREA` and `MAX_AREA`
inclusive. That drops the speck of debris at the bottom and the touching pair
at the top.

Two conditions at once needs `&` between them, with brackets around each,
because the words `and` and `or` do not work on columns.

Then count what survived and what did not.

Be clear with yourself about what this does to the count. Dropping the merged
object removes **two** real cells from the tally while removing one row, so
your cell count is now an underestimate by one. That is a defensible choice,
because measuring a merged pair as though it were one cell would corrupt every
per cell statistic. The alternative is to split them with a watershed, which
is real work and a topic of its own. What is not defensible is not knowing
which of the two you did.

Reset the index afterwards so the surviving rows renumber cleanly.

Fill in `cells`, `n_cells`, `n_dropped`.

Moves: boolean filtering with `&`, `.reset_index(drop=True)`, `len()`.
''',
        starter='''
# Part b. Keep only objects with a plausible area.

cells = None       # YOUR CODE: rows of props with MIN_AREA <= area <= MAX_AREA,
                   #            index renumbered
n_cells = None     # YOUR CODE: how many survived
n_dropped = None   # YOUR CODE: how many rows were removed

print("objects measured:", n_objects)
print("cells kept      :", n_cells)
print("rows dropped    :", n_dropped)
print(cells)
''',
        solution='''
# Part b. Keep only objects with a plausible area.

cells = props[(props["area"] >= MIN_AREA) & (props["area"] <= MAX_AREA)].reset_index(drop=True)
n_cells = int(len(cells))
n_dropped = int(len(props) - len(cells))

print("objects measured:", n_objects)
print("cells kept      :", n_cells)
print("rows dropped    :", n_dropped)
print(cells)
''',
    ),
    dict(
        label="c",
        prompt_md='''
## Part c. Summarise the cells

Three numbers describing the surviving cells: the mean area in pixels, the
mean eccentricity, and the mean of their mean intensities.

That last one deserves a sentence. `mean_intensity` is already an average,
taken over each cell's own pixels. Averaging those gives every **cell** equal
weight regardless of how many pixels it has. The alternative, averaging all
foreground pixels together, gives every **pixel** equal weight, so big cells
count more.

Which is right depends on the question, but notice that this is the same
choice you made in week 1 with cells inside animals, one level down. Any time
you average things that are themselves averages of different sized groups,
that question is waiting for you, and the usual answer in biology is to give
each unit equal weight rather than each measurement.

Keep them as plain floats.

Fill in `mean_area`, `mean_ecc`, `mean_intensity`.

Moves: `float(df["COL"].mean())`.
''',
        starter='''
# Part c. Summarise the surviving cells.

mean_area = None        # YOUR CODE: mean area in pixels
mean_ecc = None         # YOUR CODE: mean eccentricity
mean_intensity = None   # YOUR CODE: mean of the per cell mean intensities

print("mean area       : %s px" % mean_area)
print("mean eccentricity: %s" % mean_ecc)
print("mean intensity  : %s" % mean_intensity)
''',
        solution='''
# Part c. Summarise the surviving cells.

mean_area = float(cells["area"].mean())
mean_ecc = float(cells["eccentricity"].mean())
mean_intensity = float(cells["mean_intensity"].mean())

print("mean area       : %s px" % mean_area)
print("mean eccentricity: %s" % mean_ecc)
print("mean intensity  : %s" % mean_intensity)
''',
    ),
]


def build_expected(ns):
    import numpy as np
    import pandas as pd
    from skimage import measure
    labels = measure.label(ns["mask"])
    props = pd.DataFrame(measure.regionprops_table(
        labels, intensity_image=ns["field"],
        properties=("label", "area", "eccentricity", "mean_intensity")))
    lo, hi = ns["MIN_AREA"], ns["MAX_AREA"]
    cells = props[(props["area"] >= lo) & (props["area"] <= hi)].reset_index(drop=True)
    only_lo = props[props["area"] >= lo].reset_index(drop=True)
    px_weighted = float((cells["mean_intensity"] * cells["area"]).sum() / cells["area"].sum())
    return {
        "shape": [int(v) for v in ns["mask"].shape],
        "n_objects": int(labels.max()),
        "prop_cols": ["label", "area", "eccentricity", "mean_intensity"],
        "areas": [float(v) for v in props["area"]],
        "ecc": [float(v) for v in props["eccentricity"]],
        "inten": [float(v) for v in props["mean_intensity"]],
        "n_cells": int(len(cells)),
        "n_cells_only_lo": int(len(only_lo)),
        "n_dropped": int(len(props) - len(cells)),
        "cell_areas": [float(v) for v in cells["area"]],
        "mean_area": float(cells["area"].mean()),
        "mean_area_all": float(props["area"].mean()),
        "mean_ecc": float(cells["eccentricity"].mean()),
        "mean_inten": float(cells["mean_intensity"].mean()),
        "mean_inten_all": float(props["mean_intensity"].mean()),
        "mean_inten_pxweighted": px_weighted,
    }


CHECK_CODE = '''
def _part_a():
    labels = _need("labels")
    n_objects = _need("n_objects")
    props = _need("props")

    arr = _np.asarray(labels)
    if arr.dtype == bool:
        return ("`labels` is still a boolean mask. measure.label returns an INTEGER array "
                "where each object has its own number.")
    if list(arr.shape) != _EXP["shape"]:
        return "`labels` should be the same shape as the mask."
    if int(arr.max()) != _EXP["n_objects"]:
        return "`labels` does not have the expected number of objects."

    got = _as_scalar(n_objects)
    if got is None:
        return "`n_objects` should be a single whole number."
    if int(round(got)) != _EXP["n_objects"]:
        return "`n_objects` is not labels.max()."

    if not isinstance(props, _pd.DataFrame):
        return ("`props` should be a pandas DataFrame. regionprops_table returns a dict of "
                "arrays, so wrap it in pd.DataFrame.")
    cols = _cols_of(props) or []
    missing = [c for c in _EXP["prop_cols"] if c not in cols]
    if missing:
        if any(c.startswith("centroid") for c in cols):
            return ("`props` is missing %s. Ask for exactly label, area, eccentricity and "
                    "mean_intensity." % ", ".join(missing))
        return "`props` is missing the column(s) %s." % ", ".join(missing)
    if len(props) != _EXP["n_objects"]:
        return ("`props` has %d rows but there are %d labelled objects."
                % (len(props), _EXP["n_objects"]))
    if not _arr_close(props["area"].to_numpy(), _EXP["areas"]):
        return "`props` areas do not match the labelled objects."
    if not _arr_close(props["mean_intensity"].to_numpy(), _EXP["inten"], rtol=1e-4):
        return ("`props` mean_intensity does not match. Pass intensity_image=field so the "
                "measurement is taken over the original image.")
    return None


def _part_b():
    cells = _need("cells")
    n_cells = _need("n_cells")
    n_dropped = _need("n_dropped")

    if not isinstance(cells, _pd.DataFrame):
        return "`cells` should be a pandas DataFrame."
    if len(cells) == _EXP["n_objects"]:
        return ("`cells` still has every object, so nothing was filtered. Keep rows with "
                "area between MIN_AREA and MAX_AREA.")
    if len(cells) == _EXP["n_cells_only_lo"] and _EXP["n_cells_only_lo"] != _EXP["n_cells"]:
        return ("`cells` applied only the lower bound, so the merged pair is still in "
                "there. It has about double the usual area, which is what MAX_AREA is for.")
    if len(cells) != _EXP["n_cells"]:
        return "`cells` has %d rows but %d objects fall inside the area limits." % (len(cells), _EXP["n_cells"])
    if not _arr_close(cells["area"].to_numpy(), _EXP["cell_areas"]):
        return "`cells` does not hold the expected rows."
    if list(cells.index) != list(range(len(cells))):
        return "`cells` has not had its index renumbered. Use .reset_index(drop=True)."

    got = _as_scalar(n_cells)
    if got is None or int(round(got)) != _EXP["n_cells"]:
        return "`n_cells` is not the number of rows in cells."

    got = _as_scalar(n_dropped)
    if got is None:
        return "`n_dropped` should be a single whole number."
    if int(round(got)) == _EXP["n_cells"]:
        return "`n_dropped` is the number KEPT. It should be how many rows were removed."
    if int(round(got)) != _EXP["n_dropped"]:
        return "`n_dropped` is not len(props) minus len(cells)."
    return None


def _part_c():
    mean_area = _need("mean_area")
    mean_ecc = _need("mean_ecc")
    mean_intensity = _need("mean_intensity")

    got = _as_scalar(mean_area)
    if got is None:
        return "`mean_area` should be a single number."
    if _num_close(got, _EXP["mean_area_all"], rtol=1e-6):
        return ("`mean_area` was computed over every object, including the debris and the "
                "merged pair. Use the filtered table from part b.")
    if not _num_close(got, _EXP["mean_area"], rtol=1e-6):
        return "`mean_area` is not the mean of the area column of cells."

    got = _as_scalar(mean_ecc)
    if got is None:
        return "`mean_ecc` should be a single number."
    if not _num_close(got, _EXP["mean_ecc"], rtol=1e-6):
        return "`mean_ecc` is not the mean of the eccentricity column of cells."

    got = _as_scalar(mean_intensity)
    if got is None:
        return "`mean_intensity` should be a single number."
    if _num_close(got, _EXP["mean_inten_all"], rtol=1e-6):
        return ("`mean_intensity` was computed over every object. Use the filtered table "
                "from part b.")
    if _num_close(got, _EXP["mean_inten_pxweighted"], rtol=1e-5):
        return ("`mean_intensity` is pixel weighted, so larger cells count more. The task "
                "asks for the mean of the per cell means, giving each cell equal weight.")
    if not _num_close(got, _EXP["mean_inten"], rtol=1e-6):
        return "`mean_intensity` is not the mean of the mean_intensity column of cells."
    return None
'''

# Functions this problem introduces. Keys must exist in tools/function_catalog.py.
FUNCTIONS = [
    "measure.regionprops_table",
    "connectivity",
    "eccentricity",
]
