PID = "W4D1"
WEEK = 4
DAY = 1
SLUG = "w4d1_skimage_thresholding"
TITLE = "From a microscope image to a binary mask"
TOPIC = "scikit-image: images as arrays, Otsu thresholding, smoothing before you threshold"
EST = "20-30 min"

SCENARIO_MD = '''
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
'''

CONCEPT_MD = '''
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
'''

TASK_MD = '''
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
'''

DATA_CODE = '''
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
'''

TOOLKIT_MD = '''
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
'''

TOOLKIT_CODE = '''
import numpy as np
import matplotlib.pyplot as plt
from skimage import draw, filters, measure

# A tiny made up field. Nothing here is the answer.
demo_rng = np.random.default_rng(1)
demo = np.full((60, 60), 40.0)
for demo_c in [(15, 15), (40, 42), (20, 45)]:
    demo_rr, demo_cc = draw.disk(demo_c, 7, shape=demo.shape)
    demo[demo_rr, demo_cc] += 120.0
demo += demo_rng.normal(0.0, 12.0, size=demo.shape)       # heavy noise on purpose
demo = np.clip(demo, 0, 255).astype(np.uint8)

print("--- 1. an image is just an array ---")
print("shape :", demo.shape, " dtype:", demo.dtype)
print("min %d  max %d  mean %.1f" % (demo.min(), demo.max(), demo.mean()))
print("top left 4x4 corner:")
print(demo[:4, :4])

print("")
print("--- 2. a threshold gives a boolean mask ---")
demo_t = filters.threshold_otsu(demo)
demo_mask = demo > demo_t
print("otsu threshold   :", demo_t)
print("mask dtype       :", demo_mask.dtype)
print("foreground pixels:", int(demo_mask.sum()))
print("foreground frac  : %.4f  <- same as mask.mean()" % float(demo_mask.mean()))

print("")
print("--- 3. a fixed threshold does not travel ---")
demo_bright = np.clip(demo.astype(float) + 35.0, 0, 255).astype(np.uint8)
print("original, fixed t=90 : %.4f foreground" % float((demo > 90).mean()))
print("brighter, fixed t=90 : %.4f foreground  <- the background crossed the line"
      % float((demo_bright > 90).mean()))
print("brighter, otsu       : %.4f foreground  <- otsu moved with the image"
      % float((demo_bright > filters.threshold_otsu(demo_bright)).mean()))

print("")
print("--- 4. smoothing before thresholding removes speckle ---")
demo_smooth = filters.gaussian(demo, sigma=1.5)
print("gaussian output dtype:", demo_smooth.dtype,
      " range %.3f to %.3f  <- rescaled to 0..1" % (demo_smooth.min(), demo_smooth.max()))
demo_mask_s = demo_smooth > filters.threshold_otsu(demo_smooth)
print("objects found on raw     :", int(measure.label(demo_mask).max()))
print("objects found on smoothed:", int(measure.label(demo_mask_s).max()))
print("there are really only 3; the extras are single noisy pixels")

demo_fig, demo_axes = plt.subplots(1, 3, figsize=(10, 3.4))
for demo_ax, demo_im, demo_title in zip(
        demo_axes, [demo, demo_mask, demo_mask_s],
        ["raw", "mask from raw", "mask after blur"]):
    demo_ax.imshow(demo_im, cmap="gray")
    demo_ax.set_title(demo_title)
    demo_ax.axis("off")
demo_fig.tight_layout()
plt.show()
'''

PARTS = [
    dict(
        label="a",
        prompt_md='''
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
''',
        starter='''
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
''',
        solution='''
# Part a. Threshold field 1 with Otsu.

otsu_t = filters.threshold_otsu(field1)
mask = field1 > otsu_t
n_fg = int(mask.sum())
frac_fg = float(mask.mean())

print("otsu threshold :", otsu_t)
print("foreground px  :", n_fg)
print("foreground frac:", frac_fg)
if mask is not None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    axes[0].imshow(field1, cmap="gray", vmin=0, vmax=255); axes[0].set_title("field1")
    axes[1].imshow(mask, cmap="gray"); axes[1].set_title("mask")
    for a in axes: a.axis("off")
    plt.show()
''',
    ),
    dict(
        label="b",
        prompt_md='''
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
''',
        starter='''
# Part b. Field 2, thresholded two ways.

otsu_t2 = None       # YOUR CODE: the Otsu threshold computed from field2
frac_otsu_2 = None   # YOUR CODE: foreground fraction of field2 using its own Otsu
frac_fixed_2 = None  # YOUR CODE: foreground fraction of field2 using FIXED_T

print("field1 foreground frac (part a) :", frac_fg)
print("field2 otsu threshold           :", otsu_t2)
print("field2 foreground frac, otsu    :", frac_otsu_2)
print("field2 foreground frac, FIXED_T :", frac_fixed_2)
''',
        solution='''
# Part b. Field 2, thresholded two ways.

otsu_t2 = filters.threshold_otsu(field2)
frac_otsu_2 = float((field2 > otsu_t2).mean())
frac_fixed_2 = float((field2 > FIXED_T).mean())

print("field1 foreground frac (part a) :", frac_fg)
print("field2 otsu threshold           :", otsu_t2)
print("field2 foreground frac, otsu    :", frac_otsu_2)
print("field2 foreground frac, FIXED_T :", frac_fixed_2)
''',
    ),
    dict(
        label="c",
        prompt_md='''
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
''',
        starter='''
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
''',
        solution='''
# Part c. Blur first, then threshold, and count objects both ways.

smoothed = filters.gaussian(field1, sigma=1.5)
mask_smooth = smoothed > filters.threshold_otsu(smoothed)
n_obj_raw = int(measure.label(mask).max())
n_obj_smooth = int(measure.label(mask_smooth).max())

print("objects from the raw mask     :", n_obj_raw)
print("objects after blurring first  :", n_obj_smooth)
if mask_smooth is not None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    axes[0].imshow(mask, cmap="gray"); axes[0].set_title("mask from raw")
    axes[1].imshow(mask_smooth, cmap="gray"); axes[1].set_title("mask after blur")
    for a in axes: a.axis("off")
    plt.show()
''',
    ),
]


def build_expected(ns):
    import numpy as np
    from skimage import filters, measure
    f1, f2 = ns["field1"], ns["field2"]
    t1 = filters.threshold_otsu(f1)
    m1 = f1 > t1
    sm = filters.gaussian(f1, sigma=1.5)
    ms = sm > filters.threshold_otsu(sm)
    return {
        "otsu_t": float(t1),
        "n_fg": int(m1.sum()),
        "frac_fg": float(m1.mean()),
        "frac_fg_pct": float(m1.mean() * 100.0),
        "otsu_t2": float(filters.threshold_otsu(f2)),
        "frac_otsu_2": float((f2 > filters.threshold_otsu(f2)).mean()),
        "frac_fixed_2": float((f2 > ns["FIXED_T"]).mean()),
        "frac_fixed_1": float((f1 > ns["FIXED_T"]).mean()),
        "n_obj_raw": int(measure.label(m1).max()),
        "n_obj_smooth": int(measure.label(ms).max()),
        "smooth_max": float(sm.max()),
        "img_shape": [int(v) for v in f1.shape],
    }


CHECK_CODE = '''
def _part_a():
    otsu_t = _need("otsu_t")
    mask = _need("mask")
    n_fg = _need("n_fg")
    frac_fg = _need("frac_fg")

    got = _as_scalar(otsu_t)
    if got is None:
        return "`otsu_t` should be a single number."
    if _num_close(got, _EXP["otsu_t2"], rtol=1e-6):
        return "`otsu_t` was computed from field2. Part a is about field1."
    if not _num_close(got, _EXP["otsu_t"], rtol=1e-6):
        return "`otsu_t` is not filters.threshold_otsu(field1)."

    arr = _np.asarray(mask)
    if arr.dtype != bool:
        return ("`mask` should be a boolean array, not %s. A comparison like field1 > t "
                "gives booleans directly." % arr.dtype)
    if list(arr.shape) != _EXP["img_shape"]:
        return "`mask` should be the same shape as field1, which is %s." % (tuple(_EXP["img_shape"]),)
    if int(arr.sum()) != _EXP["n_fg"]:
        if int((~arr).sum()) == _EXP["n_fg"]:
            return ("`mask` is inverted, so True is marking background. The comparison runs "
                    "the other way: field1 > otsu_t.")
        return "`mask` does not match field1 thresholded at otsu_t."

    got = _as_scalar(n_fg)
    if got is None:
        return "`n_fg` should be a single whole number."
    if int(round(got)) != _EXP["n_fg"]:
        return "`n_fg` is not the number of True pixels in the mask. mask.sum() gives it."

    got = _as_scalar(frac_fg)
    if got is None:
        return "`frac_fg` should be a single number."
    if _num_close(got, _EXP["frac_fg_pct"], rtol=1e-6):
        return "`frac_fg` is a percent. The task asks for a fraction, so do not multiply by 100."
    if _num_close(got, float(_EXP["n_fg"]), rtol=1e-6):
        return "`frac_fg` is the pixel count, not the fraction. Use mask.mean()."
    if not _num_close(got, _EXP["frac_fg"], rtol=1e-6):
        return "`frac_fg` is not the mean of the mask."
    return None


def _part_b():
    otsu_t2 = _need("otsu_t2")
    frac_otsu_2 = _need("frac_otsu_2")
    frac_fixed_2 = _need("frac_fixed_2")

    got = _as_scalar(otsu_t2)
    if got is None:
        return "`otsu_t2` should be a single number."
    if _num_close(got, _EXP["otsu_t"], rtol=1e-6):
        return ("`otsu_t2` is field1's threshold. Compute it from field2, which is the "
                "whole point: Otsu adapts per image.")
    if not _num_close(got, _EXP["otsu_t2"], rtol=1e-6):
        return "`otsu_t2` is not filters.threshold_otsu(field2)."

    got = _as_scalar(frac_otsu_2)
    if got is None:
        return "`frac_otsu_2` should be a single number."
    if not _num_close(got, _EXP["frac_otsu_2"], rtol=1e-6):
        return "`frac_otsu_2` is not the foreground fraction of field2 using its own Otsu value."

    got = _as_scalar(frac_fixed_2)
    if got is None:
        return "`frac_fixed_2` should be a single number."
    if _num_close(got, _EXP["frac_fixed_1"], rtol=1e-6):
        return "`frac_fixed_2` was computed on field1. Apply FIXED_T to field2."
    if _num_close(got, _EXP["frac_otsu_2"], rtol=1e-9):
        return ("`frac_fixed_2` equals the Otsu answer, so FIXED_T was not actually used. "
                "Compare field2 against the constant FIXED_T.")
    if not _num_close(got, _EXP["frac_fixed_2"], rtol=1e-6):
        return "`frac_fixed_2` is not the fraction of field2 above FIXED_T."
    return None


def _part_c():
    smoothed = _need("smoothed")
    mask_smooth = _need("mask_smooth")
    n_obj_raw = _need("n_obj_raw")
    n_obj_smooth = _need("n_obj_smooth")

    arr = _as_array(smoothed)
    if arr is None or list(arr.shape) != _EXP["img_shape"]:
        return "`smoothed` should be the same shape as field1."
    if float(arr.max()) > 2.0:
        return ("`smoothed` still has values above 2, so it looks like the raw image rather "
                "than the output of filters.gaussian, which rescales to 0 to 1.")
    if not _num_close(float(arr.max()), _EXP["smooth_max"], rtol=1e-3):
        return ("`smoothed` does not match filters.gaussian(field1, sigma=1.5). Check the "
                "sigma.")

    arr = _np.asarray(mask_smooth)
    if arr.dtype != bool:
        return "`mask_smooth` should be a boolean array."
    if list(arr.shape) != _EXP["img_shape"]:
        return "`mask_smooth` should be the same shape as field1."

    got = _as_scalar(n_obj_raw)
    if got is None:
        return "`n_obj_raw` should be a single whole number."
    if int(round(got)) != _EXP["n_obj_raw"]:
        return ("`n_obj_raw` is not the number of connected objects in the part a mask. "
                "measure.label(mask).max() gives it.")

    got = _as_scalar(n_obj_smooth)
    if got is None:
        return "`n_obj_smooth` should be a single whole number."
    if int(round(got)) == _EXP["n_obj_raw"]:
        return ("`n_obj_smooth` equals the raw count, so the smoothed mask was not actually "
                "used, or the blur was not applied.")
    if int(round(got)) != _EXP["n_obj_smooth"]:
        return ("`n_obj_smooth` does not match. Threshold the SMOOTHED image with its own "
                "Otsu value, since gaussian rescales to 0 to 1 and the old threshold no "
                "longer means anything.")
    return None
'''

# Functions this problem introduces. Keys must exist in tools/function_catalog.py.
FUNCTIONS = [
    "images as arrays",
    "filters.threshold_otsu",
    "filters.gaussian",
    "measure.label",
    "ax.imshow",
]
