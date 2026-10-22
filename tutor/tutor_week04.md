# Week 4, scikit-image and OpenCV

The full text of every problem this week, the data it generates, the
toolkit the student is given, what each part asks for, and what every
possible check failure means. No solutions in this file.

---

# W4D1. From a microscope image to a binary mask

- topic: scikit-image: images as arrays, Otsu thresholding, smoothing before you threshold
- notebook: problems/week04/w4d1_skimage_thresholding.ipynb
- solution: solutions/week04/w4d1_skimage_thresholding_solution.ipynb
- expected time: 20-30 min

## Scenario

You have two fields of view from a fluorescence experiment. Same slide, same
antibody, same cells. The second one was imaged after the lamp had warmed up
and drifted, so it is noticeably brighter overall.

An image is just an array of numbers, so every tool from week 1 applies
unchanged. What is new is the question: which pixels are cell and which are
background. That decision is called **segmentation**, and everything
downstream, every count and every intensity measurement, inherits whatever you
decide here.

The naive approach is to pick a brightness number and call anything above it a
cell. It works beautifully on the image you tuned it on, and falls apart on
the next one, which is exactly the situation these two fields are set up to
demonstrate.

## Images are arrays, and thresholds should be computed rather than chosen

**An image is a numpy array.** A greyscale image is 2D, indexed `[row,
column]`, with row 0 at the **top**. `img.shape` is `(height, width)`.
Microscope images usually arrive as `uint8`, whole numbers from 0 to 255, or
`uint16` for a better camera. Everything you know about slicing, masking and
`axis` works without modification, because there is nothing special about an
image other than what it depicts.

One trap worth knowing: `uint8` arithmetic **wraps around**. `np.uint8(250) +
np.uint8(10)` gives 4, not 260. Convert to float before doing arithmetic you
care about.

**A threshold turns an image into a mask.** `img > t` gives a boolean array of
the same shape, True where the pixel is bright enough. That boolean array is
the mask, and `mask.sum()` counts foreground pixels because True counts as 1.
The fraction of the field that is foreground is `mask.mean()`.

**Why a hard-coded threshold is a trap.** A fixed number encodes the exposure,
the lamp intensity and the gain of the one image you tuned it on. Lamps drift,
gain gets adjusted, a different day gives a different background. The number
that gave a perfect mask on Monday silently includes half the background on
Tuesday, and nothing errors. You just get more cells.

**Otsu computes the threshold from the image itself.** It assumes the
histogram has two humps, background and foreground, and picks the cut that
best separates them, formally the one minimising the variance within each
group. Because it is derived from each image separately, it moves when the
illumination moves, which is the entire point.

Otsu's assumption matters: it expects two populations. On an image that is
almost all background, with only a handful of cells, the histogram has one
hump and Otsu will cut it somewhere arbitrary. Always look at what you got.

**Smooth before you threshold.** Camera noise is pixel-level and independent,
so thresholding raw data scatters isolated bright pixels across the
background, and each one becomes a tiny spurious object. A small Gaussian blur
averages each pixel with its neighbours, which flattens that noise while
leaving real objects, because real objects are many pixels across. The cost is
slightly blurred edges, which is a good trade. The rule of thumb is a sigma
well below the size of the things you are measuring.

## Your task

Three parts, worked one at a time.

| Part | What you work out | Variables to fill in |
|---|---|---|
| a | an Otsu threshold and the mask it gives | `otsu_t`, `mask`, `n_fg`, `frac_fg` |
| b | what a fixed threshold does on a brighter field | `otsu_t2`, `frac_otsu_2`, `frac_fixed_2` |
| c | smoothing first, and the speckle it removes | `smoothed`, `mask_smooth`, `n_obj_raw`, `n_obj_smooth` |

Below this you will find the **Setup** cell, then a **Toolkit** cell that
demonstrates every move on a tiny made up example, then **Part a**, **Part b**
and **Part c**, each with its own explanation and a cell to write in.

## The setup cell the student runs

This is given to them, already working. The data is synthetic and
seeded, so their numbers are always these numbers.

```python
import numpy as np
import matplotlib.pyplot as plt
from skimage import draw, filters, measure

PROBLEM_ID = "W4D1"
TOPIC = "scikit-image: images as arrays, Otsu thresholding, smoothing before you threshold"

IMG_SIZE = 256
FIXED_T = 110          # a threshold someone tuned on field 1 and wrote into a script


def make_field(seed, n_cells=26, background=42.0, cell_peak=150.0, noise=9.0,
               n_hot=130):
    """Build one synthetic fluorescence field of view, as uint8.

    n_hot adds isolated very bright pixels, which is what a real sensor gives
    you from hot pixels and photon shot noise.
    """
    rng = np.random.default_rng(seed)
    img = np.full((IMG_SIZE, IMG_SIZE), background, dtype=float)

    # gentle uneven illumination, brighter towards one corner
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    img += 14.0 * (yy / IMG_SIZE) + 9.0 * (xx / IMG_SIZE)

    for _ in range(n_cells):
        r = int(rng.integers(6, 12))
        cy = int(rng.integers(r + 2, IMG_SIZE - r - 2))
        cx = int(rng.integers(r + 2, IMG_SIZE - r - 2))
        rr, cc = draw.disk((cy, cx), r, shape=img.shape)
        img[rr, cc] += cell_peak * rng.uniform(0.75, 1.25)

    img += rng.normal(0.0, noise, size=img.shape)

    # isolated hot pixels, scattered over the whole frame
    hot_y = rng.integers(0, IMG_SIZE, size=n_hot)
    hot_x = rng.integers(0, IMG_SIZE, size=n_hot)
    img[hot_y, hot_x] = 255.0

    return np.clip(img, 0, 255).astype(np.uint8)


# Field 1, and field 2 after the lamp drifted brighter.
field1 = make_field(seed=4101)
field2 = make_field(seed=4102, background=100.0, cell_peak=150.0)

print("field1:", field1.shape, field1.dtype,
      " min %d  max %d  mean %.1f" % (field1.min(), field1.max(), field1.mean()))
print("field2:", field2.shape, field2.dtype,
      " min %d  max %d  mean %.1f" % (field2.min(), field2.max(), field2.mean()))
print("")
print("someone tuned a fixed threshold of %d on field 1" % FIXED_T)

_fig, _axes = plt.subplots(1, 2, figsize=(9, 4.2))
for _ax, _im, _name in zip(_axes, [field1, field2], ["field1", "field2"]):
    _ax.imshow(_im, cmap="gray", vmin=0, vmax=255)
    _ax.set_title("%s (mean %.0f)" % (_name, _im.mean()))
    _ax.axis("off")
_fig.tight_layout()
plt.show()
```

## Toolkit for this problem

| What you want to do | How |
|---|---|
| look at an image | `ax.imshow(img, cmap="gray", vmin=0, vmax=255)` |
| its size and type | `img.shape`, `img.dtype` |
| compute a threshold from the image | `filters.threshold_otsu(img)` |
| turn it into a mask | `mask = img > t` |
| count foreground pixels | `int(mask.sum())` |
| the foreground fraction | `float(mask.mean())` |
| blur away pixel noise | `filters.gaussian(img, sigma=1.5)` |
| count separate objects | `measure.label(mask).max()` |

`filters.gaussian` returns **floats scaled to 0 to 1**, not `uint8`. That does
not matter here, because Otsu is computed on whatever you hand it and the
comparison stays consistent, but it does mean the smoothed image will not look
like the original if you print its values.

The toolkit cell demonstrates each of those on a tiny made up example.
It is in the notebook and none of it is the answer.

## Part a. An Otsu threshold on field 1

Compute the threshold from the image rather than choosing one.
`filters.threshold_otsu` looks at the histogram of `field1` and returns the
grey level that best separates its two populations.

Turn that into a mask by comparing: `field1 > otsu_t` gives a boolean array of
the same shape, True where the pixel is bright enough to be cell.

Then two summaries of it. The number of foreground pixels is the sum of the
mask, because True counts as 1. The fraction of the field that is foreground
is its mean. Wrap them in `int` and `float` so they print cleanly.

Have a look at the number you get for `frac_fg`. For a field with a couple of
dozen cells on a plain background, a few percent is about right. Something
near a half would mean the threshold landed inside the background rather than
between the two populations.

Fill in `otsu_t`, `mask`, `n_fg`, `frac_fg`.

Moves: `filters.threshold_otsu`, a comparison, `.sum()`, `.mean()`.

**Variables this part is graded on:** `otsu_t`, `mask`, `n_fg`, `frac_fg`

The starter cell they see:

```python
# Part a. Threshold field 1 with Otsu.

otsu_t = None    # YOUR CODE: the Otsu threshold for field1
mask = None      # YOUR CODE: boolean array, True where field1 is above it
n_fg = None      # YOUR CODE: how many foreground pixels
frac_fg = None   # YOUR CODE: what fraction of the field is foreground

print("otsu threshold :", otsu_t)
print("foreground px  :", n_fg)
print("foreground frac:", frac_fg)
if mask is not None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    axes[0].imshow(field1, cmap="gray", vmin=0, vmax=255); axes[0].set_title("field1")
    axes[1].imshow(mask, cmap="gray"); axes[1].set_title("mask")
    for a in axes: a.axis("off")
    plt.show()
```

**Every failure message part a can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `otsu_t` should be a single number.
- `otsu_t` was computed from field2. Part a is about field1.
- `otsu_t` is not filters.threshold_otsu(field1).
- `mask` should be a boolean array, not %s. A comparison like field1 > t gives booleans directly.
- `mask` should be the same shape as field1, which is %s.
- `mask` does not match field1 thresholded at otsu_t.
- `n_fg` should be a single whole number.
- `n_fg` is not the number of True pixels in the mask. mask.sum() gives it.
- `frac_fg` should be a single number.
- `frac_fg` is a percent. The task asks for a fraction, so do not multiply by 100.
- `frac_fg` is the pixel count, not the fraction. Use mask.mean().
- `frac_fg` is not the mean of the mask.
- `mask` is inverted, so True is marking background. The comparison runs the other way: field1 > otsu_t.

## Part b. What the fixed threshold does on the brighter field

`field2` is the same kind of sample imaged after the lamp drifted, so its
background sits much higher. Someone has already written `FIXED_T` into a
processing script, tuned on field 1.

Compute three numbers on `field2`: its own Otsu threshold, the foreground
fraction that threshold gives, and the foreground fraction the fixed threshold
gives.

Then compare them against `frac_fg` from part a. Both fields contain a similar
number of similar cells, so the honest answer is that both should come out at
a similar foreground fraction. One of these two approaches manages that and
the other does not, and the gap between them is the size of the error a fixed
threshold introduces when conditions move.

Nothing about this failure is detectable from inside the script. No warning,
no exception. The counts just change, and if you were not expecting them to,
you would read it as biology.

Fill in `otsu_t2`, `frac_otsu_2`, `frac_fixed_2`.

Moves: `filters.threshold_otsu`, comparisons, `.mean()`.

**Variables this part is graded on:** `otsu_t2`, `frac_otsu_2`, `frac_fixed_2`

The starter cell they see:

```python
# Part b. Field 2, thresholded two ways.

otsu_t2 = None       # YOUR CODE: the Otsu threshold computed from field2
frac_otsu_2 = None   # YOUR CODE: foreground fraction of field2 using its own Otsu
frac_fixed_2 = None  # YOUR CODE: foreground fraction of field2 using FIXED_T

print("field1 foreground frac (part a) :", frac_fg)
print("field2 otsu threshold           :", otsu_t2)
print("field2 foreground frac, otsu    :", frac_otsu_2)
print("field2 foreground frac, FIXED_T :", frac_fixed_2)
```

**Every failure message part b can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `otsu_t2` should be a single number.
- `otsu_t2` is field1's threshold. Compute it from field2, which is the whole point: Otsu adapts per image.
- `otsu_t2` is not filters.threshold_otsu(field2).
- `frac_otsu_2` should be a single number.
- `frac_otsu_2` is not the foreground fraction of field2 using its own Otsu value.
- `frac_fixed_2` should be a single number.
- `frac_fixed_2` was computed on field1. Apply FIXED_T to field2.
- `frac_fixed_2` equals the Otsu answer, so FIXED_T was not actually used. Compare field2 against the constant FIXED_T.
- `frac_fixed_2` is not the fraction of field2 above FIXED_T.

## Part c. Smooth first, and count what the noise was inventing

Camera noise is independent from pixel to pixel, so thresholding raw data
scatters isolated bright pixels across the background. Each of those becomes
its own tiny object, and anything that counts objects counts them as cells.

Blur `field1` with `filters.gaussian` at `sigma=1.5`, which averages each
pixel with its immediate neighbours. Real cells are ten or more pixels across
and survive this easily; single-pixel noise does not.

Then threshold the smoothed image with its own Otsu value, since the blur
changes the histogram and the old threshold no longer applies. Note that
`filters.gaussian` returns floats rescaled to 0 to 1, so `otsu_t` from part a
is meaningless on it.

Finally count connected objects in both masks with `measure.label(...).max()`,
which labels each separate blob with its own integer and so tells you how many
there are. Compare the two counts against the couple of dozen cells actually
in the field.

Fill in `smoothed`, `mask_smooth`, `n_obj_raw`, `n_obj_smooth`.

Moves: `filters.gaussian(img, sigma=1.5)`, `filters.threshold_otsu`,
`measure.label(mask).max()`.

**Variables this part is graded on:** `smoothed`, `mask_smooth`, `n_obj_raw`, `n_obj_smooth`

The starter cell they see:

```python
# Part c. Blur first, then threshold, and count objects both ways.

smoothed = None       # YOUR CODE: field1 blurred with sigma=1.5
mask_smooth = None    # YOUR CODE: smoothed, thresholded with ITS OWN otsu value
n_obj_raw = None      # YOUR CODE: connected objects in the part a mask
n_obj_smooth = None   # YOUR CODE: connected objects in mask_smooth

print("objects from the raw mask     :", n_obj_raw)
print("objects after blurring first  :", n_obj_smooth)
if mask_smooth is not None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    axes[0].imshow(mask, cmap="gray"); axes[0].set_title("mask from raw")
    axes[1].imshow(mask_smooth, cmap="gray"); axes[1].set_title("mask after blur")
    for a in axes: a.axis("off")
    plt.show()
```

**Every failure message part c can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `smoothed` should be the same shape as field1.
- `smoothed` still has values above 2, so it looks like the raw image rather than the output of filters.gaussian, which rescales to 0 to 1.
- `smoothed` does not match filters.gaussian(field1, sigma=1.5). Check the sigma.
- `mask_smooth` should be a boolean array.
- `mask_smooth` should be the same shape as field1.
- `n_obj_raw` should be a single whole number.
- `n_obj_raw` is not the number of connected objects in the part a mask. measure.label(mask).max() gives it.
- `n_obj_smooth` should be a single whole number.
- `n_obj_smooth` equals the raw count, so the smoothed mask was not actually used, or the blur was not applied.
- `n_obj_smooth` does not match. Threshold the SMOOTHED image with its own Otsu value, since gaussian rescales to 0 to 1 and the old threshold no longer means anything.

**Functions this problem introduces:** images as arrays, filters.threshold_otsu, filters.gaussian, measure.label, ax.imshow

---

# W4D2. Cleaning up a mask: opening, filling holes, and dropping edge cells

- topic: scikit-image: morphological opening, filling holes, clearing the border
- notebook: problems/week04/w4d2_skimage_morphology.ipynb
- solution: solutions/week04/w4d2_skimage_morphology_solution.ipynb
- expected time: 20-30 min

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

## The setup cell the student runs

This is given to them, already working. The data is synthetic and
seeded, so their numbers are always these numbers.

```python
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
```

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

The toolkit cell demonstrates each of those on a tiny made up example.
It is in the notebook and none of it is the answer.

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

**Variables this part is graded on:** `mask_open`, `n_obj_raw`, `n_obj_open`

The starter cell they see:

```python
# Part a. Remove the hot pixel speckle with an opening.

mask_open = None   # YOUR CODE: mask_raw opened with a disk of radius 2
n_obj_raw = None   # YOUR CODE: connected objects in mask_raw
n_obj_open = None  # YOUR CODE: connected objects in mask_open

print("objects before opening:", n_obj_raw)
print("objects after opening :", n_obj_open)
```

**Every failure message part a can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `mask_open` should be a boolean array, not %s.
- `mask_open` should be the same shape as mask_raw.
- `mask_open` is unchanged from mask_raw, so the opening did not happen.
- `mask_open` does not match morphology.opening(mask_raw, morphology.disk(2)). Check the footprint radius.
- `n_obj_raw` is not the number of connected objects in mask_raw.
- `n_obj_open` should be a single whole number.
- `n_obj_open` equals the raw count, so it was counted on mask_raw rather than on the opened mask.
- `n_obj_open` is not the number of connected objects in mask_open.
- `mask_open` was opened with a radius 3 footprint, which is too large here: the cells are rings only a few pixels thick, so it cuts their walls and breaks each one into %d pieces. Use morphology.disk(2).

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

**Variables this part is graded on:** `mask_filled`, `n_hole_px`

The starter cell they see:

```python
# Part b. Fill the enclosed holes.

mask_filled = None   # YOUR CODE: mask_open with enclosed holes filled in
n_hole_px = None     # YOUR CODE: how many pixels that added

print("pixels before filling:", None if mask_open is None else int(mask_open.sum()))
print("pixels after filling :", None if mask_filled is None else int(mask_filled.sum()))
print("hole pixels filled   :", n_hole_px)
```

**Every failure message part b can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `mask_filled` should be a boolean array.
- `mask_filled` should be the same shape as the mask.
- `mask_filled` has the same pixel count as mask_open, so nothing was filled. The cells are rings; their centres should end up solid.
- `mask_filled` looks like the result of a closing, not a fill. Closing grows then shrinks, which also bridges nearby cells. Use ndi.binary_fill_holes, which only fills enclosed background.
- `mask_filled` does not match ndi.binary_fill_holes(mask_open).
- `mask_filled` has the wrong number of objects, which suggests cells got merged. Filling holes should not change the object count.
- `n_hole_px` should be a single whole number.
- `n_hole_px` is the total pixel count of mask_filled, not the number ADDED. Subtract the pixel count of mask_open.
- `n_hole_px` is not mask_filled.sum() minus mask_open.sum().

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

**Variables this part is graded on:** `mask_clean`, `n_obj_clean`, `n_border_objects`

The starter cell they see:

```python
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
```

**Every failure message part c can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `mask_clean` should be a boolean array.
- `mask_clean` should be the same shape as the mask.
- `mask_clean` is unchanged from mask_filled, so clear_border removed nothing. Check you applied it and kept the result.
- `mask_clean` does not match segmentation.clear_border(mask_filled).
- `n_obj_clean` should be a single whole number.
- `n_obj_clean` is not the number of objects left in mask_clean.
- `n_border_objects` should be a single whole number.
- `n_border_objects` is the number REMAINING, not the number removed. Subtract the remaining count from the count before clearing.
- `n_border_objects` is not the drop in object count caused by clear_border.

**Functions this problem introduces:** morphology.opening, morphology.closing, morphology.disk, binary_fill_holes, clear_border

---

# W4D3. Labelling objects and measuring them one by one

- topic: scikit-image: connected components, regionprops_table, filtering by measurement
- notebook: problems/week04/w4d3_skimage_regionprops.ipynb
- solution: solutions/week04/w4d3_skimage_regionprops_solution.ipynb
- expected time: 20-30 min

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

## The setup cell the student runs

This is given to them, already working. The data is synthetic and
seeded, so their numbers are always these numbers.

```python
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
```

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

The toolkit cell demonstrates each of those on a tiny made up example.
It is in the notebook and none of it is the answer.

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

**Variables this part is graded on:** `labels`, `n_objects`, `props`

The starter cell they see:

```python
# Part a. Label the objects, then measure every one of them.

labels = None      # YOUR CODE: the labelled integer array
n_objects = None   # YOUR CODE: how many objects there are
props = None       # YOUR CODE: DataFrame with label, area, eccentricity, mean_intensity

print("n_objects:", n_objects)
print(None if props is None else props.round(3))
```

**Every failure message part a can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `labels` is still a boolean mask. measure.label returns an INTEGER array where each object has its own number.
- `labels` should be the same shape as the mask.
- `labels` does not have the expected number of objects.
- `n_objects` should be a single whole number.
- `n_objects` is not labels.max().
- `props` should be a pandas DataFrame. regionprops_table returns a dict of arrays, so wrap it in pd.DataFrame.
- `props` is missing the column(s) %s. ,
- `props` has %d rows but there are %d labelled objects.
- `props` areas do not match the labelled objects.
- `props` mean_intensity does not match. Pass intensity_image=field so the measurement is taken over the original image.
- `props` is missing %s. Ask for exactly label, area, eccentricity and mean_intensity. ,

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

**Variables this part is graded on:** `cells`, `n_cells`, `n_dropped`

The starter cell they see:

```python
# Part b. Keep only objects with a plausible area.

cells = None       # YOUR CODE: rows of props with MIN_AREA <= area <= MAX_AREA,
                   #            index renumbered
n_cells = None     # YOUR CODE: how many survived
n_dropped = None   # YOUR CODE: how many rows were removed

print("objects measured:", n_objects)
print("cells kept      :", n_cells)
print("rows dropped    :", n_dropped)
print(cells)
```

**Every failure message part b can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `cells` should be a pandas DataFrame.
- `cells` still has every object, so nothing was filtered. Keep rows with area between MIN_AREA and MAX_AREA.
- `cells` applied only the lower bound, so the merged pair is still in there. It has about double the usual area, which is what MAX_AREA is for.
- `cells` has %d rows but %d objects fall inside the area limits.
- `cells` does not hold the expected rows.
- `cells` has not had its index renumbered. Use .reset_index(drop=True).
- `n_cells` is not the number of rows in cells.
- `n_dropped` should be a single whole number.
- `n_dropped` is the number KEPT. It should be how many rows were removed.
- `n_dropped` is not len(props) minus len(cells).

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

**Variables this part is graded on:** `mean_area`, `mean_ecc`, `mean_intensity`

The starter cell they see:

```python
# Part c. Summarise the surviving cells.

mean_area = None        # YOUR CODE: mean area in pixels
mean_ecc = None         # YOUR CODE: mean eccentricity
mean_intensity = None   # YOUR CODE: mean of the per cell mean intensities

print("mean area       : %s px" % mean_area)
print("mean eccentricity: %s" % mean_ecc)
print("mean intensity  : %s" % mean_intensity)
```

**Every failure message part c can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `mean_area` should be a single number.
- `mean_area` was computed over every object, including the debris and the merged pair. Use the filtered table from part b.
- `mean_area` is not the mean of the area column of cells.
- `mean_ecc` should be a single number.
- `mean_ecc` is not the mean of the eccentricity column of cells.
- `mean_intensity` should be a single number.
- `mean_intensity` was computed over every object. Use the filtered table from part b.
- `mean_intensity` is pixel weighted, so larger cells count more. The task asks for the mean of the per cell means, giving each cell equal weight.
- `mean_intensity` is not the mean of the mean_intensity column of cells.

**Functions this problem introduces:** measure.regionprops_table, connectivity, eccentricity

---

# W4D4. The same segmentation in OpenCV, and where the two libraries disagree

- topic: OpenCV: blurring, Otsu thresholding, findContours and contour areas
- notebook: problems/week04/w4d4_opencv_contours.ipynb
- solution: solutions/week04/w4d4_opencv_contours_solution.ipynb
- expected time: 20-30 min

## Scenario

The imaging core's existing analysis script is written in OpenCV, because
OpenCV got there first and is very fast. You have been asked to understand it
well enough to check its numbers against your scikit-image pipeline.

The two libraries do the same job with different conventions, and the places
they differ are exactly the places bugs live. OpenCV came out of real time
computer vision, so it is built around 8 bit images, it returns things in
orders that make sense for video and not for arrays, and it describes objects
by their **outlines** rather than by which pixels belong to them.

That last difference is the interesting one. A contour is a polygon tracing
the boundary of an object, and the area of that polygon is not the same number
as the count of pixels inside it. Neither is wrong. They answer slightly
different questions, and a pipeline that mixes them produces numbers that are
consistently a few percent off with no obvious cause.

## OpenCV conventions, and what a contour actually is

**It wants 8 bit arrays.** Most OpenCV functions expect `uint8`. Hand them
floats and you get either an error or silence followed by nonsense. skimage,
by contrast, is happy with floats and often prefers them. Converting back and
forth is normal; forgetting to is a classic.

**Sizes come as (width, height).** Numpy shapes are `(rows, columns)`, which
is height then width. OpenCV kernel and size arguments are the other way
round. For a square kernel it makes no difference, which is why the mistake
survives until the day you use a non-square one.

**Kernels must be odd.** `cv2.GaussianBlur(img, (5, 5), 0)` is fine, `(4, 4)`
raises. The zero is sigma, and zero means "work it out from the kernel size".

**Thresholding returns two things.** `cv2.threshold` gives back the threshold
it used and the thresholded image, in that order. Combining
`cv2.THRESH_BINARY + cv2.THRESH_OTSU` tells it to compute the threshold by
Otsu's method and ignore the value you passed, which is why that argument is
conventionally written as 0. The output is `uint8` with values 0 and 255, not
booleans, which matters if you then try to use it as a mask.

**findContours traces outlines.** It returns a list of contours, each an array
of boundary points. `cv2.RETR_EXTERNAL` keeps only outermost contours and
ignores holes inside objects, which is usually what you want for counting
cells. `cv2.CHAIN_APPROX_SIMPLE` compresses straight runs of boundary points,
which saves memory and changes nothing about the shape.

It returns **two** values in OpenCV 4 and 5, and returned three in OpenCV 3.
Code written against the old version breaks, which is why you still find
`cv2.findContours(...)[-2]` in the wild as a version-proof way to grab the
contours whichever it is.

**Why contour area is not pixel count.** `cv2.contourArea` computes the area
of the polygon through the boundary pixel centres. The pixels on the boundary
straddle that line, so roughly half of each sits outside the polygon. The
result is consistently a little **smaller** than the pixel count, and the gap
is proportional to the perimeter, so it hurts small objects proportionally
more than big ones.

So the honest comparison is not "which is right" but "which question did I
ask". Pixel count is what skimage's `area` gives you. Polygon area is what
`contourArea` gives you. Pick one, say which, and do not mix them within a
study.

## Your task

Three parts, worked one at a time.

| Part | What you work out | Variables to fill in |
|---|---|---|
| a | blur and threshold the OpenCV way | `blurred`, `t_cv`, `binary` |
| b | the contours, and how many there are | `contours`, `n_contours` |
| c | contour areas, filtered, and compared with skimage | `areas`, `n_cells_cv`, `area_ratio` |

Below this you will find the **Setup** cell, then a **Toolkit** cell that
demonstrates every move on a tiny made up example, then **Part a**, **Part b**
and **Part c**, each with its own explanation and a cell to write in.

## The setup cell the student runs

This is given to them, already working. The data is synthetic and
seeded, so their numbers are always these numbers.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from skimage import draw, measure

PROBLEM_ID = "W4D4"
TOPIC = "OpenCV: blurring, Otsu thresholding, findContours and contour areas"

IMG_SIZE = 256
MIN_AREA = 150
rng = np.random.default_rng(4401)

_img = np.full((IMG_SIZE, IMG_SIZE), 44.0)
_centres = [(45, 45), (50, 125), (42, 205), (115, 70), (125, 155), (110, 228),
            (185, 50), (180, 135), (200, 218), (240, 100), (232, 185)]
for _cy, _cx in _centres:
    _ra = float(rng.uniform(11.0, 14.0))
    _rb = float(rng.uniform(9.0, 13.0))
    _rr, _cc = draw.ellipse(_cy, _cx, _ra, _rb, shape=_img.shape,
                            rotation=float(rng.uniform(0.0, np.pi)))
    _img[_rr, _cc] += 120.0 * rng.uniform(0.8, 1.2)

# a little debris, small enough that the area filter should remove it
for _cy, _cx in [(150, 30), (30, 160), (215, 150)]:
    _rr, _cc = draw.disk((_cy, _cx), 4, shape=_img.shape)
    _img[_rr, _cc] += 110.0

_img += rng.normal(0.0, 6.0, size=_img.shape)
field = np.clip(_img, 0, 255).astype(np.uint8)

# The skimage answer, computed for you, so part c has something to compare against.
_sk_labels = measure.label(field > 110)
_sk_props = pd.DataFrame(measure.regionprops_table(_sk_labels, properties=("label", "area")))
SK_CELLS = _sk_props[_sk_props["area"] >= MIN_AREA].reset_index(drop=True)
SK_N_CELLS = int(len(SK_CELLS))
SK_TOTAL_AREA = float(SK_CELLS["area"].sum())

print("field:", field.shape, field.dtype)
print("eleven cells and three specks of debris were drawn")
print("")
print("scikit-image, for comparison later:")
print("  cells found      :", SK_N_CELLS)
print("  total pixel area : %.1f" % SK_TOTAL_AREA)

_fig, _ax = plt.subplots(figsize=(4.2, 4.2))
_ax.imshow(field, cmap="gray", vmin=0, vmax=255)
_ax.set_title("field"); _ax.axis("off")
plt.show()
```

## Toolkit for this problem

| What you want to do | How |
|---|---|
| blur, odd kernel only | `cv2.GaussianBlur(img, (5, 5), 0)` |
| Otsu threshold | `t, binimg = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)` |
| find outlines | `cv2.findContours(binimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)` |
| grab the contours safely | take `[-2]` of what it returns, which works on OpenCV 3, 4 and 5 |
| the area of one contour | `cv2.contourArea(c)` |
| its perimeter | `cv2.arcLength(c, True)` |
| draw them on a copy | `cv2.drawContours(img_copy, contours, -1, 255, 1)` |

`cv2.threshold` returns a **pair**, the threshold and the image, so unpack it.
Passing `0` as the threshold value is the convention when you add
`THRESH_OTSU`, because the value is ignored and computed instead.

The toolkit cell demonstrates each of those on a tiny made up example.
It is in the notebook and none of it is the answer.

## Part a. Blur and threshold, the OpenCV way

Blur `field` with `cv2.GaussianBlur` using a 5 by 5 kernel and sigma 0, which
means "derive sigma from the kernel size". The kernel is passed as a tuple and
both entries must be odd.

Then threshold the blurred image with `cv2.threshold`. Pass `0` as the
threshold value, `255` as the maximum, and
`cv2.THRESH_BINARY + cv2.THRESH_OTSU` as the mode. That combination tells
OpenCV to ignore the value you gave and compute one by Otsu instead, which is
why the convention is to pass zero.

It returns **two** things: the threshold it chose and the thresholded image.
Unpack both. Forgetting that it returns a pair, and assigning the whole tuple
to one name, is the single most common OpenCV mistake.

Note what `binary` is: a `uint8` array of 0 and 255, not a boolean mask. If
you want to use it with numpy or skimage later, compare it to 0 first.

Fill in `blurred`, `t_cv`, `binary`.

Moves: `cv2.GaussianBlur(img, (5, 5), 0)`,
`t, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)`.

**Variables this part is graded on:** `blurred`, `t_cv`, `binary`

The starter cell they see:

```python
# Part a. Blur, then threshold with Otsu, the OpenCV way.

blurred = None   # YOUR CODE: field blurred with a 5x5 Gaussian, sigma 0
t_cv = None      # YOUR CODE: the threshold cv2 chose
binary = None    # YOUR CODE: the thresholded image

print("threshold chosen:", t_cv)
print("binary dtype    :", None if binary is None else binary.dtype)
print("unique values   :", None if binary is None else np.unique(binary).tolist())
```

**Every failure message part a can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `blurred` should be the same shape as field.
- `blurred` came back as %s. cv2.GaussianBlur on a uint8 image returns uint8; a float result means field was converted first.
- `t_cv` should be a single number, the first thing cv2.threshold returns.
- `t_cv` is not the threshold cv2 chose. Pass 0 as the value and cv2.THRESH_BINARY + cv2.THRESH_OTSU as the mode.
- `binary` should be the same shape as field.
- `binary` is boolean, but cv2.threshold returns a uint8 image of 0 and 255. Keep what cv2 gave you.
- `binary` has values %s but cv2.threshold with THRESH_BINARY gives only 0 and 255.
- `binary` does not match thresholding the blurred image.
- `t_cv` is a tuple, so the pair cv2.threshold returns was assigned to one name. Unpack it: t_cv, binary = cv2.threshold(...).

## Part b. Find the contours

`cv2.findContours` traces the outline of every object in the binary image.
Pass `cv2.RETR_EXTERNAL`, which keeps only outermost contours and ignores any
holes inside them, and `cv2.CHAIN_APPROX_SIMPLE`, which compresses straight
runs of boundary points without changing the shape.

Take the contours out of what it returns using `[-2]`. OpenCV 3 returned three
values and OpenCV 4 and 5 return two, so indexing from the end works on all of
them. This is genuinely how people write it in production code, and it is
worth recognising when you meet it.

Then count them. Compare that count against `SK_N_CELLS`, the scikit-image
answer printed in the setup cell. They may not match exactly, and either
library may be the one over-counting: the debris specks are near the area
cutoff, and the two libraries blurred and thresholded slightly differently.
Part c is where you filter by size and see whether they converge.

Fill in `contours`, `n_contours`.

Moves: `cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)`,
indexing with `[-2]`, `len()`.

**Variables this part is graded on:** `contours`, `n_contours`

The starter cell they see:

```python
# Part b. Trace the outlines.

contours = None     # YOUR CODE: the contours, taken with [-2] so it works on any version
n_contours = None   # YOUR CODE: how many

print("contours found   :", n_contours)
print("skimage found    :", SK_N_CELLS, "cells after its own area filter")
```

**Every failure message part b can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `contours` is everything cv2.findContours returned, not the contours themselves. Take [-2] of it.
- `contours` holds %d items but %d were expected. Pass cv2.RETR_EXTERNAL and cv2.CHAIN_APPROX_SIMPLE, and run it on `binary`.
- `n_contours` should be a single whole number.
- `n_contours` is not len(contours).
- `contours` should be a sequence of contours.

## Part c. Areas, filtering, and the comparison

Compute the area of every contour with `cv2.contourArea`, as a plain list of
floats in the order the contours came.

Filter to the ones at least `MIN_AREA`, the same cutoff skimage used, and
count them. That should drop the debris specks and leave you close to the
skimage count.

Then quantify the systematic difference. Sum the areas of the contours that
passed the filter, and divide by `SK_TOTAL_AREA`, the total pixel area
scikit-image measured for its cells. Keep it as a ratio.

Think about which way that ratio should go before you look. The contour
polygon runs through the **centres** of the boundary pixels, so roughly half
of each boundary pixel lies outside it, while skimage counts those pixels in
full. So the ratio should come out a little under 1, and it should be nearer 1
for larger objects, because the boundary is a smaller share of them.

A few percent is not a bug, and it does not go away by being careful. It is
the two libraries answering slightly different questions. What matters is
picking one and not mixing them inside a study.

Fill in `areas`, `n_cells_cv`, `area_ratio`.

Moves: a list comprehension over `contours`, `cv2.contourArea`, filtering,
`sum()`, division.

**Variables this part is graded on:** `areas`, `n_cells_cv`, `area_ratio`

The starter cell they see:

```python
# Part c. Contour areas, filtered, and compared with skimage.

areas = None         # YOUR CODE: list of cv2.contourArea for every contour
n_cells_cv = None    # YOUR CODE: how many contours have area >= MIN_AREA
area_ratio = None    # YOUR CODE: total area of those contours / SK_TOTAL_AREA

print("all contour areas :", None if areas is None else sorted(round(a, 1) for a in areas))
print("cells by OpenCV   :", n_cells_cv)
print("cells by skimage  :", SK_N_CELLS)
print("area ratio cv2/skimage: %s" % area_ratio)
```

**Every failure message part c can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `areas` has %d entries but there are %d contours, so there should be one each.
- `areas` holds perimeters, not areas. cv2.arcLength gives the perimeter and cv2.contourArea gives the area.
- `areas` does not match cv2.contourArea over the contours.
- `n_cells_cv` should be a single whole number.
- `n_cells_cv` counted every contour, so the area filter was not applied. Keep only those with area at least MIN_AREA.
- `n_cells_cv` is not the number of contours with area at least MIN_AREA.
- `area_ratio` should be a single number.
- `area_ratio` is the total contour area, not a ratio. Divide by SK_TOTAL_AREA.
- `area_ratio` is the other way up. The task asks for the OpenCV total divided by the scikit-image total, which should come out a little under 1.
- `area_ratio` used every contour, including the debris. Sum only the ones that passed the area filter, so it matches what skimage counted.
- `area_ratio` is not the filtered contour area total over SK_TOTAL_AREA.
- `areas` should be a sequence of numbers, one per contour.

**Functions this problem introduces:** cv2.GaussianBlur, cv2.threshold, cv2.findContours, cv2.contourArea

---

# W4D5. A whole imaging study: fields inside animals inside groups

- topic: scikit-image pipeline: per field, per animal, per group, with the right n
- notebook: problems/week04/w4d5_imaging_pipeline_animals.ipynb
- solution: solutions/week04/w4d5_imaging_pipeline_animals_solution.ipynb
- expected time: 20-30 min

## Scenario

The full study. Two groups, four animals in each, and between two and four
usable fields of view imaged per animal, because some slides had more
intact tissue than others. Every field contains some cells. The readout is
the mean staining intensity of the cells.

So the data is nested **three** deep: cells inside fields, fields inside
animals, animals inside groups.

This is the week 1 lesson again, with an extra layer and a more convincing
disguise. Pooling every cell from every field gives you a number computed
from a few hundred measurements, which feels reassuringly large and is wrong
for exactly the same reasons as before: n is inflated, and animals are
weighted by how much tissue survived sectioning rather than by anything
biological.

The fix is the same, applied twice. Cells up to fields, fields up to animals,
animals up to groups. The group SD at the end is across **eight** animals, and
it is the only n in this study that means anything.

## Aggregating up a hierarchy, one level at a time

**The rule.** Summarise at each level before moving to the next, and let every
unit at a level count once, regardless of how much data sits beneath it.

Here that is three steps. Each field gives one number, the mean over its
cells. Each animal gives one number, the mean over its fields. Each group
gives a mean and an SD over its animals.

**Why not just average all the cells.** Because the number of cells per field
and the number of fields per animal both vary, and neither variation is
biological. An animal whose sections happened to come out well contributes
four fields and perhaps two hundred cells; one whose sections tore contributes
two fields and eighty. Pool them and the first animal counts two and a half
times as much as the second. Your group mean is then partly a measurement of
sectioning quality.

**Why not weight by cell count.** It is tempting, because more cells feels
like more information. But the thing you are trying to generalise over is
**animals**, and you have eight of them whatever happens on the slides.
Weighting by cells makes your estimate more precise about the animals you
happened to image well, which is not the same as being more precise about the
population. Unweighted means at each level is the standard approach, and the
one a reviewer will expect.

**What the final SD must be across.** Animals. Not fields, not cells. With
four animals per group, that SD rests on four numbers and is correspondingly
uncertain, and reporting it honestly rather than reaching for a tighter number
derived from cells is the whole discipline. A cell level SD would be perhaps
ten times narrower and would describe the spread of cells, which nobody asked
about.

**The shape this takes in pandas.** Each level is a `groupby` on the keys that
survive, aggregating the column produced by the level below. Three small
steps, each obvious on its own. The discipline is not technical, it is
remembering to do it at all when the pooled version is one line shorter and
gives you a smaller p value.

## Your task

Three parts, worked one at a time, each feeding the next.

| Part | What you work out | Variables to fill in |
|---|---|---|
| a | one row per field, from the images | `per_field` |
| b | one row per animal | `per_animal` |
| c | one row per group, SD across animals | `group_stats` |

Below this you will find the **Setup** cell, then a **Toolkit** cell that
demonstrates every move on a tiny made up example, then **Part a**, **Part b**
and **Part c**, each with its own explanation and a cell to write in.

## The setup cell the student runs

This is given to them, already working. The data is synthetic and
seeded, so their numbers are always these numbers.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import draw, filters, measure, morphology, segmentation
from scipy import ndimage as ndi

PROBLEM_ID = "W4D5"
TOPIC = "scikit-image pipeline: per field, per animal, per group, with the right n"

IMG_SIZE = 128
MIN_AREA = 60
rng = np.random.default_rng(4501)

GROUP_ORDER = ["Vehicle", "Drug"]
ANIMALS = [("V%d" % i, "Vehicle") for i in range(1, 5)] + \
          [("D%d" % i, "Drug") for i in range(1, 5)]
_group_level = {"Vehicle": 128.0, "Drug": 74.0}


def _make_field(seed, level):
    """One field of view: a few cells on a dim background."""
    r = np.random.default_rng(seed)
    img = np.full((IMG_SIZE, IMG_SIZE), 35.0)
    n = int(r.integers(6, 12))
    for _ in range(n):
        ra, rb = float(r.uniform(7.0, 10.0)), float(r.uniform(6.0, 9.0))
        cy = int(r.integers(12, IMG_SIZE - 12))
        cx = int(r.integers(12, IMG_SIZE - 12))
        rr, cc = draw.ellipse(cy, cx, ra, rb, shape=img.shape,
                              rotation=float(r.uniform(0, np.pi)))
        img[rr, cc] += level * r.uniform(0.80, 1.20)
    img += r.normal(0.0, 5.0, size=img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


# Build the study. Animals differ from each other, and the number of usable
# fields differs too, because some sections tore.
fields = []
_seed = 100
for _animal, _grp in ANIMALS:
    _animal_level = _group_level[_grp] * rng.lognormal(0.0, 0.16)
    _n_fields = int(rng.integers(2, 5))
    for _f in range(1, _n_fields + 1):
        fields.append({"ANIMAL": _animal, "GROUP": _grp, "FIELD": _f,
                       "IMAGE": _make_field(_seed, _animal_level)})
        _seed += 1


def segment_field(img):
    """The week 4 pipeline in one function. Returns one row per cell."""
    mask = img > filters.threshold_otsu(img)
    mask = morphology.opening(mask, morphology.disk(2))
    mask = ndi.binary_fill_holes(mask)
    mask = segmentation.clear_border(mask)
    labels = measure.label(mask)
    props = pd.DataFrame(measure.regionprops_table(
        labels, intensity_image=img, properties=("label", "area", "mean_intensity")))
    return props[props["area"] >= MIN_AREA].reset_index(drop=True)


print("fields imaged:", len(fields))
print(pd.DataFrame([{k: v for k, v in f.items() if k != "IMAGE"} for f in fields])
      .groupby(["GROUP", "ANIMAL"]).size().to_string())
print("")
print("example of what segment_field gives you for one field:")
print(segment_field(fields[0]["IMAGE"]).round(2).to_string(index=False))

_fig, _axes = plt.subplots(1, 4, figsize=(12, 3.2))
for _ax, _f in zip(_axes, fields[:4]):
    _ax.imshow(_f["IMAGE"], cmap="gray", vmin=0, vmax=255)
    _ax.set_title("%s field %d" % (_f["ANIMAL"], _f["FIELD"]))
    _ax.axis("off")
_fig.tight_layout()
plt.show()
```

## Toolkit for this problem

| What you want to do | How |
|---|---|
| loop with an index | `for f in fields:` then `f["ANIMAL"]`, `f["IMAGE"]` |
| segment one field | `segment_field(f["IMAGE"])` |
| build rows then a table | collect dicts in a list, then `pd.DataFrame(rows)` |
| a column mean | `float(df["mean_intensity"].mean())` |
| one level up | `.groupby([...]).agg(NAME=("COL", "mean"), ...).reset_index()` |
| count rows in a group | the aggregation `"size"` |
| sample SD | the aggregation `"std"`, already n-1 in pandas |

Building a list of dictionaries and handing it to `pd.DataFrame` at the end is
usually clearer than growing a DataFrame row by row, and far faster.

The toolkit cell demonstrates each of those on a tiny made up example.
It is in the notebook and none of it is the answer.

## Part a. Run the pipeline on every field

Loop over `fields`. Each entry is a dictionary with `ANIMAL`, `GROUP`,
`FIELD` and `IMAGE`. Hand the image to `segment_field`, which runs the whole
week 4 pipeline and returns one row per surviving cell, with `area` and
`mean_intensity`.

For each field record four things: the animal, the group, the field number,
how many cells were found, and the mean of those cells' `mean_intensity`
values. That last number is the field's single summary, and it is the first
level of aggregation.

Collect the rows as a list of dictionaries and build the DataFrame once at the
end. Growing a DataFrame inside a loop works but is slow and reads worse.

One thing to watch: a field could in principle yield no cells at all, and the
mean of nothing is `nan`. With this data every field does find cells, but
noticing that the case exists is the habit worth having.

Fill in `per_field`, sorted by `ANIMAL` then `FIELD`, index renumbered, with
exactly:

| column | meaning |
|---|---|
| `ANIMAL` | animal identifier |
| `GROUP` | `Vehicle` or `Drug` |
| `FIELD` | field number within that animal |
| `N_CELLS` | cells found in that field |
| `MEAN_INTENSITY` | mean of that field's cell intensities |

Moves: a `for` loop, `segment_field(...)`, `len()`, `.mean()`,
`pd.DataFrame(rows)`, `.sort_values([...]).reset_index(drop=True)`.

**Variables this part is graded on:** `per_field`

The starter cell they see:

```python
# Part a. Segment every field and summarise each one.

rows = []
# YOUR CODE: loop over fields, segment each image, and append one dict per field
#            with ANIMAL, GROUP, FIELD, N_CELLS, MEAN_INTENSITY

per_field = None   # YOUR CODE: pd.DataFrame(rows), sorted by ANIMAL then FIELD

print(per_field)
```

**Every failure message part a can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `per_field` should be a pandas DataFrame.
- `per_field` is missing the column(s) %s. ,
- `per_field` has %d rows but %d fields were imaged, so there should be one row each.
- `per_field` does not list the expected animals.
- `per_field` does not list the expected field numbers.
- `N_CELLS` does not match what segment_field returns. It is the number of ROWS in that result, one per cell.
- `MEAN_INTENSITY` contains nan, so some field produced no cells to average.
- `MEAN_INTENSITY` is not the mean of that field's cell mean_intensity values.
- `per_field` has not had its index renumbered. Use .reset_index(drop=True).

## Part b. Fields up to animals

One row per animal. Three numbers: how many fields it contributed, how many
cells in total across those fields, and its mean intensity.

That last one is the mean of the **field means**, not of the cells. Each field
counts once regardless of how many cells it happened to contain, for the same
reason each animal will count once in part c regardless of how many fields it
contributed. You are applying the same rule at a second level.

`TOTAL_CELLS` is a sum rather than a mean, because it is a count of things
rather than an average of measurements. It is worth keeping: it is the number
that shows how uneven the imaging was, and therefore why the unweighted mean
matters.

Fill in `per_animal`, sorted by `ANIMAL`, index renumbered, with exactly:

| column | meaning |
|---|---|
| `ANIMAL` | animal identifier |
| `GROUP` | `Vehicle` or `Drug` |
| `N_FIELDS` | fields imaged for that animal |
| `TOTAL_CELLS` | cells found across all its fields |
| `MEAN_INTENSITY` | mean of that animal's field means |

Moves: `.groupby(["ANIMAL", "GROUP"]).agg(NAME=("COL", "mean"), ...)`,
`"size"`, `"sum"`, `.reset_index()`.

**Variables this part is graded on:** `per_animal`

The starter cell they see:

```python
# Part b. One row per animal.

per_animal = None   # YOUR CODE: ANIMAL, GROUP, N_FIELDS, TOTAL_CELLS, MEAN_INTENSITY

print(per_animal)
```

**Every failure message part b can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `per_animal` should be a pandas DataFrame.
- `per_animal` is missing the column(s) %s. ,
- `per_animal` still has one row per field. Group by ANIMAL so the fields collapse.
- `per_animal` has %d rows but there are %d animals.
- `per_animal` does not list the expected animals.
- `N_FIELDS` is not the number of fields each animal contributed.
- `TOTAL_CELLS` is not the sum of N_CELLS across that animal's fields. It is a sum, not a mean.
- `MEAN_INTENSITY` is weighted by how many cells each field contained, so a field with more cells counts more. Take the plain mean of the field means, giving each field equal weight.
- `MEAN_INTENSITY` is not the mean of that animal's field means.

## Part c. Animals up to groups

The last level, and the one that goes in the figure.

One row per group: the mean of the animal means, the sample SD **across
animals**, and the number of animals. Sort so `Vehicle` comes before `Drug`,
which is dose order rather than alphabetical order.

Check `N_ANIMALS` when you have it. It should be 4 in each group. If you see
anything in the tens it came from fields, and anything in the hundreds from
cells, and in either case the SD beside it is describing the wrong thing.

Look at the SD too. Built from four animals, it will be wide, and that is the
honest answer. A cell level SD would be far narrower and would be answering a
question about cells that nobody asked. The temptation to reach for the
narrower number is the whole reason this rule needs stating.

Fill in `group_stats`, index renumbered, with exactly:

| column | meaning |
|---|---|
| `GROUP` | `Vehicle` then `Drug` |
| `MEAN` | mean of the animal means |
| `SD` | sample SD of the animal means |
| `N_ANIMALS` | animals in that group |

Moves: `.groupby("GROUP").agg(...)`, `.reset_index()`, `.map` or a sort key
for the group order.

**Variables this part is graded on:** `group_stats`

The starter cell they see:

```python
# Part c. One row per group, with the SD across animals.

group_stats = None   # YOUR CODE: GROUP, MEAN, SD, N_ANIMALS, Vehicle first

print(group_stats)
```

**Every failure message part c can produce.** Each one is a real
mistake the check was written to catch, so when the student quotes
one, it tells you exactly what they did. A `%s` or `%d` in a message
is filled in at the time with the variable name or the number they
got:

- `group_stats` should be a pandas DataFrame.
- `group_stats` is missing the column(s) %s. ,
- `group_stats` should have one row per group, so %d.
- `group_stats` is not in the order Vehicle then Drug. Sort it into GROUP_ORDER rather than leaving it alphabetical.
- `N_ANIMALS` counted FIELDS, not animals. Build this from per_animal, where each animal is already one row.
- `N_ANIMALS` is not the number of animals per group. It should be 4 in each.
- `MEAN` is the mean over fields, so animals with more usable sections count more. Average the animal means instead.
- `MEAN` is not the mean of the animal means within each group.
- `SD` is the SEM, not the SD. It was divided by the square root of the number of animals. This repo always reports SD.
- `SD` is the spread across FIELDS, not across animals. It is narrower than the truth because fields within an animal resemble each other.
- `SD` is not the sample SD of the animal means within each group.

**Functions this problem introduces:** building a DataFrame from rows, three level aggregation
