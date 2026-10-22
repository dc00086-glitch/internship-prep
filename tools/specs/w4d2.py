PID = "W4D2"
WEEK = 4
DAY = 2
SLUG = "w4d2_skimage_morphology"
TITLE = "Cleaning up a mask: opening, filling holes, and dropping edge cells"
TOPIC = "scikit-image: morphological opening, filling holes, clearing the border"
EST = "20-30 min"

SCENARIO_MD = '''
## Scenario

Yesterday's threshold gave you a mask, and the mask has three problems that
every real segmentation has.

**Speckle.** Hot pixels on the sensor crossed the threshold and became
objects, so a field with two dozen cells reports well over a hundred.

**Holes.** The stain is weakest at the centre of each cell, where the nucleus
sits, so many cells come out as rings rather than filled discs. Anything that
measures area will undercount, and anything that counts objects may split one
ring into pieces.

**Edge cells.** Cells at the border of the field are only partly in frame. You
can see a sliver of them, but their area and shape are whatever the crop
happened to leave, and including them biases every per cell measurement
downward.

All three are fixed with morphology, which is a small set of operations that
reshape a binary mask based on local geometry rather than on brightness.
'''

CONCEPT_MD = '''
## Morphology: reshaping a mask by what fits inside it

Every morphological operation slides a small shape, called the **structuring
element** or footprint, over the mask and asks a local question. In
scikit-image `morphology.disk(2)` gives you a roughly circular footprint of
radius 2, which is the usual choice for round objects because it treats every
direction the same.

**Erosion** keeps a pixel only if the whole footprint fits inside the
foreground there. It shrinks objects and deletes anything smaller than the
footprint. **Dilation** is the opposite and grows them.

**Opening is erosion then dilation.** The erosion deletes anything too small
to contain the footprint, and the dilation restores the survivors to roughly
their original size. The net effect is that small specks vanish and real
objects come back nearly unchanged. This is precisely the tool for hot pixel
speckle, and the thing to tune is the footprint radius: big enough to swallow
the noise, small enough to leave the cells.

**Closing is dilation then erosion**, which fills small gaps and joins things
that are nearly touching. It is the obvious tool for holes, but it has a
serious side effect: it also joins cells that are close together, turning two
cells into one object and losing a count you cared about.

**So for holes, prefer `binary_fill_holes`.** It works topologically rather
than geometrically: it finds background regions that are completely surrounded
by foreground and fills them, whatever their size or shape. A hole in the
middle of a cell is enclosed and gets filled. The background around the cells
connects to the image border and does not. Because it is about enclosure
rather than size, it cannot accidentally merge two separate cells, which
closing can. It lives in `scipy.ndimage`, not skimage, which is a small
annoyance to remember.

**`clear_border` removes objects touching the edge.** Whether you should use
it depends on the question. For **per cell** measurements, area, shape,
intensity, you must, because a partly imaged cell gives a wrong value for all
three. For **density**, cells per square millimetre, removing them biases the
count downward and you want a different correction instead. Think about which
question you are answering before you reach for it.

**Order matters.** Clear the speckle first, so you are not filling holes in
noise. Fill holes next, so objects are solid. Clear the border last, when each
object is a single connected piece and it is unambiguous whether it touches
the edge.
'''

TASK_MD = '''
## Your task

Three parts, worked one at a time, and they chain in this order.

| Part | What you work out | Variables to fill in |
|---|---|---|
| a | speckle removed with an opening | `mask_open`, `n_obj_raw`, `n_obj_open` |
| b | holes inside cells filled in | `mask_filled`, `n_hole_px` |
| c | partly imaged cells dropped | `mask_clean`, `n_obj_clean`, `n_border_objects` |

Below this you will find the **Setup** cell, then a **Toolkit** cell that
demonstrates every move on a tiny made up example, then **Part a**, **Part b**
and **Part c**, each with its own explanation and a cell to write in.
'''

DATA_CODE = '''
import numpy as np
import matplotlib.pyplot as plt
from skimage import draw, filters, measure, morphology, segmentation
from scipy import ndimage as ndi

PROBLEM_ID = "W4D2"
TOPIC = "scikit-image: morphological opening, filling holes, clearing the border"

IMG_SIZE = 256
rng = np.random.default_rng(4201)

# Build a field whose cells are RINGS, because the stain is weak at the nucleus,
# with several cells deliberately running off the edge of the frame.
_img = np.full((IMG_SIZE, IMG_SIZE), 45.0)
_yy, _xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
_img += 12.0 * (_yy / IMG_SIZE)

_centres = [(40, 35), (60, 120), (45, 200), (120, 60), (135, 150), (115, 225),
            (200, 40), (190, 130), (210, 210), (90, 90), (160, 95),
            (8, 70), (250, 160), (130, 4), (70, 252)]      # last four touch an edge
for _cy, _cx in _centres:
    _r = int(rng.integers(10, 14))
    _rr, _cc = draw.disk((_cy, _cx), _r, shape=_img.shape)
    _img[_rr, _cc] += 150.0
    # the nucleus stains weakly, leaving a hole in the middle
    _rr, _cc = draw.disk((_cy, _cx), max(3, _r - 6), shape=_img.shape)
    _img[_rr, _cc] -= 140.0

_img += rng.normal(0.0, 8.0, size=_img.shape)
_hot_y = rng.integers(0, IMG_SIZE, size=160)
_hot_x = rng.integers(0, IMG_SIZE, size=160)
_img[_hot_y, _hot_x] = 255.0
field = np.clip(_img, 0, 255).astype(np.uint8)

# The starting mask, exactly as yesterday's approach would produce it.
mask_raw = field > filters.threshold_otsu(field)

print("field:", field.shape, field.dtype)
print("mask_raw foreground pixels:", int(mask_raw.sum()))
print("connected objects in mask_raw:", int(measure.label(mask_raw).max()))
print("")
print("there are %d real cells in this field, four of them running off an edge"
      % len(_centres))

_fig, _axes = plt.subplots(1, 2, figsize=(9, 4.3))
_axes[0].imshow(field, cmap="gray", vmin=0, vmax=255); _axes[0].set_title("field")
_axes[1].imshow(mask_raw, cmap="gray"); _axes[1].set_title("mask_raw")
for _a in _axes: _a.axis("off")
_fig.tight_layout()
plt.show()
'''

TOOLKIT_MD = '''
## Toolkit for this problem

| What you want to do | How |
|---|---|
| a round footprint | `morphology.disk(2)` |
| remove small specks | `morphology.opening(mask, morphology.disk(2))` |
| close small gaps | `morphology.closing(mask, morphology.disk(2))` |
| fill enclosed holes | `ndi.binary_fill_holes(mask)` from `scipy.ndimage` |
| drop objects touching the edge | `segmentation.clear_border(mask)` |
| count objects | `int(measure.label(mask).max())` |
| count pixels | `int(mask.sum())` |

`binary_fill_holes` lives in `scipy.ndimage`, imported here as `ndi`, not in
skimage. `opening` and `closing` work on boolean masks as well as greyscale.
'''

TOOLKIT_CODE = '''
import numpy as np
import matplotlib.pyplot as plt
from skimage import draw, measure, morphology, segmentation
from scipy import ndimage as ndi

# A tiny made up mask with all three problems. Nothing here is the answer.
demo = np.zeros((40, 40), dtype=bool)
demo_rr, demo_cc = draw.disk((15, 15), 8, shape=demo.shape); demo[demo_rr, demo_cc] = True
demo_rr, demo_cc = draw.disk((15, 15), 3, shape=demo.shape); demo[demo_rr, demo_cc] = False
demo_rr, demo_cc = draw.disk((30, 32), 7, shape=demo.shape); demo[demo_rr, demo_cc] = True
demo_rr, demo_cc = draw.disk((2, 30), 6, shape=demo.shape); demo[demo_rr, demo_cc] = True
demo[5, 35] = True; demo[35, 5] = True; demo[22, 8] = True      # three specks

print("one ring shaped cell, one solid cell, one cell on the top edge, three specks")
print("objects to begin with :", int(measure.label(demo).max()))
print("foreground pixels     :", int(demo.sum()))

print("")
print("--- 1. opening deletes anything the footprint cannot fit inside ---")
demo_open = morphology.opening(demo, morphology.disk(2))
print("after opening, objects:", int(measure.label(demo_open).max()),
      " pixels:", int(demo_open.sum()))
print("the three single pixel specks are gone, the cells survive")

print("")
print("--- 2. filling holes versus closing ---")
demo_filled = ndi.binary_fill_holes(demo_open)
demo_closed = morphology.closing(demo_open, morphology.disk(4))
print("pixels after fill_holes:", int(demo_filled.sum()),
      " objects:", int(measure.label(demo_filled).max()))
print("pixels after closing   :", int(demo_closed.sum()),
      " objects:", int(measure.label(demo_closed).max()))
print("")
print("fill_holes targets enclosed background only, so it cannot merge two cells.")
print("closing with a big enough footprint can, which is why it is the wrong tool here.")

print("")
print("--- 3. clear_border drops anything touching an edge ---")
demo_clean = segmentation.clear_border(demo_filled)
print("objects before:", int(measure.label(demo_filled).max()))
print("objects after :", int(measure.label(demo_clean).max()))
print("the cell running off the top edge has gone")

demo_fig, demo_axes = plt.subplots(1, 4, figsize=(12, 3.2))
for demo_ax, demo_im, demo_title in zip(
        demo_axes, [demo, demo_open, demo_filled, demo_clean],
        ["raw", "opened", "holes filled", "border cleared"]):
    demo_ax.imshow(demo_im, cmap="gray"); demo_ax.set_title(demo_title); demo_ax.axis("off")
demo_fig.tight_layout()
plt.show()
'''

PARTS = [
    dict(
        label="a",
        prompt_md='''
## Part a. Open away the speckle

`mask_raw` contains the real cells plus one object per hot pixel. Count what
you are starting with, then remove the specks with a morphological opening.

Use `morphology.opening` with `morphology.disk(2)` as the footprint. The
erosion half deletes anything a radius 2 disc cannot fit inside, which is
every isolated pixel and every tiny cluster, and the dilation half restores
the survivors.

Count connected objects before and after with `measure.label(...).max()`. The
drop should be dramatic, from well over a hundred down to something in the
range of the number of cells actually present.

A note on choosing the radius. It has to be large enough to swallow the noise
and small enough to leave real structure alone. Here the noise is single
pixels, so anything from radius 1 upwards clears it, and in fact radius 1 and
radius 2 give an identical mask on this field. The binding constraint is the
other end: the cells are rings whose walls are only a few pixels thick, and at
radius 3 the opening cuts through those walls and shatters each ring into
fragments. Try it and watch the object count jump.

That asymmetry is typical. The lower limit is usually obvious and forgiving;
the upper limit is set by the thinnest real structure you care about, and
crossing it does visible damage.

Fill in `mask_open`, `n_obj_raw`, `n_obj_open`.

Moves: `morphology.opening(mask, morphology.disk(2))`,
`measure.label(mask).max()`.
''',
        starter='''
# Part a. Remove the hot pixel speckle with an opening.

mask_open = None   # YOUR CODE: mask_raw opened with a disk of radius 2
n_obj_raw = None   # YOUR CODE: connected objects in mask_raw
n_obj_open = None  # YOUR CODE: connected objects in mask_open

print("objects before opening:", n_obj_raw)
print("objects after opening :", n_obj_open)
''',
        solution='''
# Part a. Remove the hot pixel speckle with an opening.

mask_open = morphology.opening(mask_raw, morphology.disk(2))
n_obj_raw = int(measure.label(mask_raw).max())
n_obj_open = int(measure.label(mask_open).max())

print("objects before opening:", n_obj_raw)
print("objects after opening :", n_obj_open)
''',
    ),
    dict(
        label="b",
        prompt_md='''
## Part b. Fill the holes left by the weak nuclear stain

The cells are rings, because the stain is faint where the nucleus sits. Fill
those centres in so each cell becomes one solid object.

Use `ndi.binary_fill_holes` on `mask_open`. It finds background regions that
are entirely enclosed by foreground and turns them on. The background outside
the cells connects to the image border, so it is not enclosed and is left
alone.

Then measure how much was filled: the number of pixels `mask_filled` has that
`mask_open` did not. Since filling only ever adds pixels, that is just the
difference of the two sums.

Resist reaching for `closing` here even though it would also close the rings.
With a footprint large enough to span the nuclear hole, it would also bridge
any two cells sitting within that distance of each other, silently merging
them into one object. You would not see it in the mask, you would see it later
as a cell count that is slightly too low and a mean area that is too high.

Fill in `mask_filled`, `n_hole_px`.

Moves: `ndi.binary_fill_holes(mask)`, `.sum()`.
''',
        starter='''
# Part b. Fill the enclosed holes.

mask_filled = None   # YOUR CODE: mask_open with enclosed holes filled in
n_hole_px = None     # YOUR CODE: how many pixels that added

print("pixels before filling:", None if mask_open is None else int(mask_open.sum()))
print("pixels after filling :", None if mask_filled is None else int(mask_filled.sum()))
print("hole pixels filled   :", n_hole_px)
''',
        solution='''
# Part b. Fill the enclosed holes.

mask_filled = ndi.binary_fill_holes(mask_open)
n_hole_px = int(mask_filled.sum() - mask_open.sum())

print("pixels before filling:", None if mask_open is None else int(mask_open.sum()))
print("pixels after filling :", None if mask_filled is None else int(mask_filled.sum()))
print("hole pixels filled   :", n_hole_px)
''',
    ),
    dict(
        label="c",
        prompt_md='''
## Part c. Drop the cells that run off the edge

Four of the cells in this field are only partly in frame. Their area is
whatever the crop left, their shape is a truncated disc, and their mean
intensity is measured over the wrong region. Any per cell statistic computed
including them is biased.

`segmentation.clear_border` removes every object touching the image boundary.
Apply it to `mask_filled`, then count what is left and how many objects were
removed.

This is the right call here because the next problem measures per cell
properties. It would be the wrong call if you were measuring **density**,
cells per unit area, because then dropping edge cells undercounts a real
quantity and the correction is a different one entirely. Same operation,
opposite verdict, decided by the question rather than by the image.

Fill in `mask_clean`, `n_obj_clean`, `n_border_objects`.

Moves: `segmentation.clear_border(mask)`, `measure.label(mask).max()`,
subtraction.
''',
        starter='''
# Part c. Remove objects touching the frame edge.

mask_clean = None        # YOUR CODE: mask_filled with edge touching objects removed
n_obj_clean = None       # YOUR CODE: objects remaining
n_border_objects = None  # YOUR CODE: how many were removed

print("objects before:", None if mask_filled is None else int(measure.label(mask_filled).max()))
print("objects after :", n_obj_clean)
print("removed       :", n_border_objects)

if mask_clean is not None:
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    for a, im, ttl in zip(axes, [mask_raw, mask_open, mask_filled, mask_clean],
                          ["raw", "opened", "holes filled", "border cleared"]):
        a.imshow(im, cmap="gray"); a.set_title(ttl); a.axis("off")
    fig.tight_layout()
    plt.show()
''',
        solution='''
# Part c. Remove objects touching the frame edge.

mask_clean = segmentation.clear_border(mask_filled)
n_obj_clean = int(measure.label(mask_clean).max())
n_border_objects = int(measure.label(mask_filled).max()) - n_obj_clean

print("objects before:", None if mask_filled is None else int(measure.label(mask_filled).max()))
print("objects after :", n_obj_clean)
print("removed       :", n_border_objects)

if mask_clean is not None:
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    for a, im, ttl in zip(axes, [mask_raw, mask_open, mask_filled, mask_clean],
                          ["raw", "opened", "holes filled", "border cleared"]):
        a.imshow(im, cmap="gray"); a.set_title(ttl); a.axis("off")
    fig.tight_layout()
    plt.show()
''',
    ),
]


def build_expected(ns):
    import numpy as np
    from skimage import measure, morphology, segmentation
    from scipy import ndimage as ndi
    raw = ns["mask_raw"]
    opened = morphology.opening(raw, morphology.disk(2))
    opened1 = morphology.opening(raw, morphology.disk(1))
    closed = morphology.closing(opened, morphology.disk(4))
    filled = ndi.binary_fill_holes(opened)
    clean = segmentation.clear_border(filled)
    return {
        "shape": [int(v) for v in raw.shape],
        "n_obj_raw": int(measure.label(raw).max()),
        "n_obj_open": int(measure.label(opened).max()),
        "n_obj_open_r1": int(measure.label(opened1).max()),
        "px_open_r3": int(morphology.opening(raw, morphology.disk(3)).sum()),
        "n_obj_open_r3": int(measure.label(morphology.opening(raw, morphology.disk(3))).max()),
        "px_open": int(opened.sum()),
        "px_filled": int(filled.sum()),
        "px_closed": int(closed.sum()),
        "n_hole_px": int(filled.sum() - opened.sum()),
        "n_obj_filled": int(measure.label(filled).max()),
        "n_obj_clean": int(measure.label(clean).max()),
        "px_clean": int(clean.sum()),
        "n_border": int(measure.label(filled).max() - measure.label(clean).max()),
    }


CHECK_CODE = '''
def _part_a():
    mask_open = _need("mask_open")
    n_obj_raw = _need("n_obj_raw")
    n_obj_open = _need("n_obj_open")

    arr = _np.asarray(mask_open)
    if arr.dtype != bool:
        return "`mask_open` should be a boolean array, not %s." % arr.dtype
    if list(arr.shape) != _EXP["shape"]:
        return "`mask_open` should be the same shape as mask_raw."
    if int(arr.sum()) == int(_np.asarray(_need("mask_raw")).sum()):
        return "`mask_open` is unchanged from mask_raw, so the opening did not happen."
    if int(arr.sum()) != _EXP["px_open"]:
        if int(arr.sum()) == _EXP["px_open_r3"]:
            return ("`mask_open` was opened with a radius 3 footprint, which is too large "
                    "here: the cells are rings only a few pixels thick, so it cuts their "
                    "walls and breaks each one into %d pieces. Use morphology.disk(2)."
                    % _EXP["n_obj_open_r3"])
        return ("`mask_open` does not match morphology.opening(mask_raw, "
                "morphology.disk(2)). Check the footprint radius.")

    got = _as_scalar(n_obj_raw)
    if got is None or int(round(got)) != _EXP["n_obj_raw"]:
        return "`n_obj_raw` is not the number of connected objects in mask_raw."

    got = _as_scalar(n_obj_open)
    if got is None:
        return "`n_obj_open` should be a single whole number."
    if int(round(got)) == _EXP["n_obj_raw"]:
        return ("`n_obj_open` equals the raw count, so it was counted on mask_raw rather "
                "than on the opened mask.")
    if int(round(got)) != _EXP["n_obj_open"]:
        return "`n_obj_open` is not the number of connected objects in mask_open."
    return None


def _part_b():
    mask_filled = _need("mask_filled")
    n_hole_px = _need("n_hole_px")

    arr = _np.asarray(mask_filled)
    if arr.dtype != bool:
        return "`mask_filled` should be a boolean array."
    if list(arr.shape) != _EXP["shape"]:
        return "`mask_filled` should be the same shape as the mask."
    if int(arr.sum()) == _EXP["px_open"]:
        return ("`mask_filled` has the same pixel count as mask_open, so nothing was "
                "filled. The cells are rings; their centres should end up solid.")
    if int(arr.sum()) == _EXP["px_closed"] and _EXP["px_closed"] != _EXP["px_filled"]:
        return ("`mask_filled` looks like the result of a closing, not a fill. Closing "
                "grows then shrinks, which also bridges nearby cells. Use "
                "ndi.binary_fill_holes, which only fills enclosed background.")
    if int(arr.sum()) != _EXP["px_filled"]:
        return "`mask_filled` does not match ndi.binary_fill_holes(mask_open)."
    if int(measure.label(arr).max()) != _EXP["n_obj_filled"]:
        return ("`mask_filled` has the wrong number of objects, which suggests cells got "
                "merged. Filling holes should not change the object count.")

    got = _as_scalar(n_hole_px)
    if got is None:
        return "`n_hole_px` should be a single whole number."
    if int(round(got)) == _EXP["px_filled"]:
        return ("`n_hole_px` is the total pixel count of mask_filled, not the number ADDED. "
                "Subtract the pixel count of mask_open.")
    if int(round(got)) != _EXP["n_hole_px"]:
        return "`n_hole_px` is not mask_filled.sum() minus mask_open.sum()."
    return None


def _part_c():
    mask_clean = _need("mask_clean")
    n_obj_clean = _need("n_obj_clean")
    n_border_objects = _need("n_border_objects")

    arr = _np.asarray(mask_clean)
    if arr.dtype != bool:
        return "`mask_clean` should be a boolean array."
    if list(arr.shape) != _EXP["shape"]:
        return "`mask_clean` should be the same shape as the mask."
    if int(arr.sum()) == _EXP["px_filled"]:
        return ("`mask_clean` is unchanged from mask_filled, so clear_border removed "
                "nothing. Check you applied it and kept the result.")
    if int(arr.sum()) != _EXP["px_clean"]:
        return "`mask_clean` does not match segmentation.clear_border(mask_filled)."

    got = _as_scalar(n_obj_clean)
    if got is None:
        return "`n_obj_clean` should be a single whole number."
    if int(round(got)) != _EXP["n_obj_clean"]:
        return "`n_obj_clean` is not the number of objects left in mask_clean."

    got = _as_scalar(n_border_objects)
    if got is None:
        return "`n_border_objects` should be a single whole number."
    if int(round(got)) == _EXP["n_obj_clean"]:
        return ("`n_border_objects` is the number REMAINING, not the number removed. "
                "Subtract the remaining count from the count before clearing.")
    if int(round(got)) != _EXP["n_border"]:
        return ("`n_border_objects` is not the drop in object count caused by "
                "clear_border.")
    return None
'''

# Functions this problem introduces. Keys must exist in tools/function_catalog.py.
FUNCTIONS = [
    "morphology.opening",
    "morphology.closing",
    "morphology.disk",
    "binary_fill_holes",
    "clear_border",
]
