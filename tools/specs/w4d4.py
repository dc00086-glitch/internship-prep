PID = "W4D4"
WEEK = 4
DAY = 4
SLUG = "w4d4_opencv_contours"
TITLE = "The same segmentation in OpenCV, and where the two libraries disagree"
TOPIC = "OpenCV: blurring, Otsu thresholding, findContours and contour areas"
EST = "20-30 min"

SCENARIO_MD = '''
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
'''

CONCEPT_MD = '''
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
'''

TASK_MD = '''
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
'''

DATA_CODE = '''
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
'''

TOOLKIT_MD = '''
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
'''

TOOLKIT_CODE = '''
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import draw, measure

# A tiny made up field. Nothing here is the answer.
demo = np.full((60, 60), 40.0)
for demo_c in [(18, 18), (40, 40)]:
    demo_rr, demo_cc = draw.disk(demo_c, 9, shape=demo.shape)
    demo[demo_rr, demo_cc] += 130.0
demo = np.clip(demo + np.random.default_rng(0).normal(0, 5, demo.shape), 0, 255).astype(np.uint8)

print("--- 1. OpenCV wants uint8 ---")
print("demo dtype:", demo.dtype)
try:
    cv2.GaussianBlur(demo.astype(float) / 255.0, (4, 4), 0)
except Exception as err:
    print("a float image with an even kernel:", type(err).__name__)
print("kernels must be odd, and sizes are given as (width, height)")

print("")
print("--- 2. threshold returns TWO things ---")
demo_blur = cv2.GaussianBlur(demo, (5, 5), 0)
demo_t, demo_bin = cv2.threshold(demo_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print("returned threshold:", demo_t)
print("returned image    :", demo_bin.dtype, " unique values:", np.unique(demo_bin).tolist())
print("note it is uint8 0 and 255, not True and False")

print("")
print("--- 3. findContours, and the version trap ---")
demo_found = cv2.findContours(demo_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("it returned", len(demo_found), "values on this OpenCV")
demo_contours = demo_found[-2]      # works whether it returned 2 or 3
print("contours found:", len(demo_contours))
print("first contour is an array of boundary points, shape:", demo_contours[0].shape)

print("")
print("--- 4. contour area is NOT pixel count ---")
demo_labels = measure.label(demo_bin > 0)
demo_px = [int((demo_labels == i).sum()) for i in range(1, demo_labels.max() + 1)]
demo_poly = sorted(round(cv2.contourArea(c), 1) for c in demo_contours)
print("skimage pixel counts :", sorted(demo_px))
print("cv2 polygon areas    :", demo_poly)
print("ratio                : %.4f" % (sum(demo_poly) / sum(demo_px)))
print("")
print("the polygon runs through the centres of the boundary pixels, so about half")
print("of each boundary pixel falls outside it. The gap scales with perimeter,")
print("which means small objects lose proportionally more than big ones.")

demo_vis = cv2.cvtColor(demo, cv2.COLOR_GRAY2BGR)
cv2.drawContours(demo_vis, demo_contours, -1, (255, 0, 0), 1)
demo_fig, demo_ax = plt.subplots(figsize=(3.4, 3.4))
demo_ax.imshow(demo_vis)
demo_ax.set_title("contours drawn on the image"); demo_ax.axis("off")
plt.show()
'''

PARTS = [
    dict(
        label="a",
        prompt_md='''
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
''',
        starter='''
# Part a. Blur, then threshold with Otsu, the OpenCV way.

blurred = None   # YOUR CODE: field blurred with a 5x5 Gaussian, sigma 0
t_cv = None      # YOUR CODE: the threshold cv2 chose
binary = None    # YOUR CODE: the thresholded image

print("threshold chosen:", t_cv)
print("binary dtype    :", None if binary is None else binary.dtype)
print("unique values   :", None if binary is None else np.unique(binary).tolist())
''',
        solution='''
# Part a. Blur, then threshold with Otsu, the OpenCV way.

blurred = cv2.GaussianBlur(field, (5, 5), 0)
t_cv, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

print("threshold chosen:", t_cv)
print("binary dtype    :", None if binary is None else binary.dtype)
print("unique values   :", None if binary is None else np.unique(binary).tolist())
''',
    ),
    dict(
        label="b",
        prompt_md='''
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
''',
        starter='''
# Part b. Trace the outlines.

contours = None     # YOUR CODE: the contours, taken with [-2] so it works on any version
n_contours = None   # YOUR CODE: how many

print("contours found   :", n_contours)
print("skimage found    :", SK_N_CELLS, "cells after its own area filter")
''',
        solution='''
# Part b. Trace the outlines.

contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
n_contours = int(len(contours))

print("contours found   :", n_contours)
print("skimage found    :", SK_N_CELLS, "cells after its own area filter")
''',
    ),
    dict(
        label="c",
        prompt_md='''
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
''',
        starter='''
# Part c. Contour areas, filtered, and compared with skimage.

areas = None         # YOUR CODE: list of cv2.contourArea for every contour
n_cells_cv = None    # YOUR CODE: how many contours have area >= MIN_AREA
area_ratio = None    # YOUR CODE: total area of those contours / SK_TOTAL_AREA

print("all contour areas :", None if areas is None else sorted(round(a, 1) for a in areas))
print("cells by OpenCV   :", n_cells_cv)
print("cells by skimage  :", SK_N_CELLS)
print("area ratio cv2/skimage: %s" % area_ratio)
''',
        solution='''
# Part c. Contour areas, filtered, and compared with skimage.

areas = [float(cv2.contourArea(c)) for c in contours]
_big = [a for a in areas if a >= MIN_AREA]
n_cells_cv = int(len(_big))
area_ratio = float(sum(_big) / SK_TOTAL_AREA)

print("all contour areas :", None if areas is None else sorted(round(a, 1) for a in areas))
print("cells by OpenCV   :", n_cells_cv)
print("cells by skimage  :", SK_N_CELLS)
print("area ratio cv2/skimage: %s" % area_ratio)
''',
    ),
]


def build_expected(ns):
    import numpy as np
    import cv2
    field = ns["field"]
    blurred = cv2.GaussianBlur(field, (5, 5), 0)
    t_cv, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    areas = [float(cv2.contourArea(c)) for c in contours]
    big = [a for a in areas if a >= ns["MIN_AREA"]]
    perims = [float(cv2.arcLength(c, True)) for c in contours]
    return {
        "shape": [int(v) for v in field.shape],
        "t_cv": float(t_cv),
        "binary_values": sorted(int(v) for v in np.unique(binary)),
        "n_binary_fg": int((binary > 0).sum()),
        "n_contours": int(len(contours)),
        "areas_sorted": sorted(round(a, 4) for a in areas),
        "perims_sorted": sorted(round(p, 4) for p in perims),
        "n_cells_cv": int(len(big)),
        "sum_big": float(sum(big)),
        "sum_all": float(sum(areas)),
        "area_ratio": float(sum(big) / ns["SK_TOTAL_AREA"]),
        "sk_total_area": float(ns["SK_TOTAL_AREA"]),
        "sk_n_cells": int(ns["SK_N_CELLS"]),
    }


CHECK_CODE = '''
def _part_a():
    blurred = _need("blurred")
    t_cv = _need("t_cv")
    binary = _need("binary")

    arr = _np.asarray(blurred)
    if list(arr.shape) != _EXP["shape"]:
        return "`blurred` should be the same shape as field."
    if arr.dtype != _np.uint8:
        return ("`blurred` came back as %s. cv2.GaussianBlur on a uint8 image returns "
                "uint8; a float result means field was converted first." % arr.dtype)

    got = _as_scalar(t_cv)
    if got is None:
        if isinstance(t_cv, tuple):
            return ("`t_cv` is a tuple, so the pair cv2.threshold returns was assigned to "
                    "one name. Unpack it: t_cv, binary = cv2.threshold(...).")
        return "`t_cv` should be a single number, the first thing cv2.threshold returns."
    if not _num_close(got, _EXP["t_cv"], rtol=1e-6):
        return ("`t_cv` is not the threshold cv2 chose. Pass 0 as the value and "
                "cv2.THRESH_BINARY + cv2.THRESH_OTSU as the mode.")

    arr = _np.asarray(binary)
    if list(arr.shape) != _EXP["shape"]:
        return "`binary` should be the same shape as field."
    if arr.dtype == bool:
        return ("`binary` is boolean, but cv2.threshold returns a uint8 image of 0 and 255. "
                "Keep what cv2 gave you.")
    vals = sorted(int(v) for v in _np.unique(arr))
    if vals != _EXP["binary_values"]:
        return ("`binary` has values %s but cv2.threshold with THRESH_BINARY gives only "
                "0 and 255." % vals)
    if int((arr > 0).sum()) != _EXP["n_binary_fg"]:
        return "`binary` does not match thresholding the blurred image."
    return None


def _part_b():
    contours = _need("contours")
    n_contours = _need("n_contours")

    if isinstance(contours, tuple) and len(contours) in (2, 3) and \\
            not hasattr(contours[0], "shape"):
        return ("`contours` is everything cv2.findContours returned, not the contours "
                "themselves. Take [-2] of it.")
    try:
        count = len(contours)
    except TypeError:
        return "`contours` should be a sequence of contours."
    if count != _EXP["n_contours"]:
        return ("`contours` holds %d items but %d were expected. Pass cv2.RETR_EXTERNAL "
                "and cv2.CHAIN_APPROX_SIMPLE, and run it on `binary`."
                % (count, _EXP["n_contours"]))

    got = _as_scalar(n_contours)
    if got is None:
        return "`n_contours` should be a single whole number."
    if int(round(got)) != _EXP["n_contours"]:
        return "`n_contours` is not len(contours)."
    return None


def _part_c():
    areas = _need("areas")
    n_cells_cv = _need("n_cells_cv")
    area_ratio = _need("area_ratio")

    try:
        got = sorted(round(float(a), 4) for a in areas)
    except TypeError:
        return "`areas` should be a sequence of numbers, one per contour."
    if len(got) != _EXP["n_contours"]:
        return ("`areas` has %d entries but there are %d contours, so there should be one "
                "each." % (len(got), _EXP["n_contours"]))
    if got == _EXP["perims_sorted"]:
        return ("`areas` holds perimeters, not areas. cv2.arcLength gives the perimeter and "
                "cv2.contourArea gives the area.")
    if not _arr_close(got, _EXP["areas_sorted"], rtol=1e-4, atol=1e-6):
        return "`areas` does not match cv2.contourArea over the contours."

    got = _as_scalar(n_cells_cv)
    if got is None:
        return "`n_cells_cv` should be a single whole number."
    if int(round(got)) == _EXP["n_contours"]:
        return ("`n_cells_cv` counted every contour, so the area filter was not applied. "
                "Keep only those with area at least MIN_AREA.")
    if int(round(got)) != _EXP["n_cells_cv"]:
        return "`n_cells_cv` is not the number of contours with area at least MIN_AREA."

    got = _as_scalar(area_ratio)
    if got is None:
        return "`area_ratio` should be a single number."
    if _num_close(got, _EXP["sum_big"], rtol=1e-6):
        return ("`area_ratio` is the total contour area, not a ratio. Divide by "
                "SK_TOTAL_AREA.")
    if _num_close(got, 1.0 / _EXP["area_ratio"], rtol=1e-4):
        return ("`area_ratio` is the other way up. The task asks for the OpenCV total "
                "divided by the scikit-image total, which should come out a little under 1.")
    if _num_close(got, _EXP["sum_all"] / _EXP["sk_total_area"], rtol=1e-6) and \\
            not _num_close(got, _EXP["area_ratio"], rtol=1e-6):
        return ("`area_ratio` used every contour, including the debris. Sum only the ones "
                "that passed the area filter, so it matches what skimage counted.")
    if not _num_close(got, _EXP["area_ratio"], rtol=1e-4):
        return "`area_ratio` is not the filtered contour area total over SK_TOTAL_AREA."
    return None
'''

# Functions this problem introduces. Keys must exist in tools/function_catalog.py.
FUNCTIONS = [
    "cv2.GaussianBlur",
    "cv2.threshold",
    "cv2.findContours",
    "cv2.contourArea",
]
