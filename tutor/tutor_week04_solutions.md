# Week 4 reference solutions

One way to do each part. The checks test the numbers, not the route, so a
different correct answer is still correct.

Do not paste these to the student. Use them to work out where their
version diverges, and say the smallest thing that gets them unstuck.

---

## W4D1. From a microscope image to a binary mask

### Part a

```python
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
```

### Part b

```python
# Part b. Field 2, thresholded two ways.

otsu_t2 = filters.threshold_otsu(field2)
frac_otsu_2 = float((field2 > otsu_t2).mean())
frac_fixed_2 = float((field2 > FIXED_T).mean())

print("field1 foreground frac (part a) :", frac_fg)
print("field2 otsu threshold           :", otsu_t2)
print("field2 foreground frac, otsu    :", frac_otsu_2)
print("field2 foreground frac, FIXED_T :", frac_fixed_2)
```

### Part c

```python
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
```

---

## W4D2. Cleaning up a mask: opening, filling holes, and dropping edge cells

### Part a

```python
# Part a. Remove the hot pixel speckle with an opening.

mask_open = morphology.opening(mask_raw, morphology.disk(2))
n_obj_raw = int(measure.label(mask_raw).max())
n_obj_open = int(measure.label(mask_open).max())

print("objects before opening:", n_obj_raw)
print("objects after opening :", n_obj_open)
```

### Part b

```python
# Part b. Fill the enclosed holes.

mask_filled = ndi.binary_fill_holes(mask_open)
n_hole_px = int(mask_filled.sum() - mask_open.sum())

print("pixels before filling:", None if mask_open is None else int(mask_open.sum()))
print("pixels after filling :", None if mask_filled is None else int(mask_filled.sum()))
print("hole pixels filled   :", n_hole_px)
```

### Part c

```python
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
```

---

## W4D3. Labelling objects and measuring them one by one

### Part a

```python
# Part a. Label the objects, then measure every one of them.

labels = measure.label(mask)
n_objects = int(labels.max())
props = pd.DataFrame(measure.regionprops_table(
    labels, intensity_image=field,
    properties=("label", "area", "eccentricity", "mean_intensity")))

print("n_objects:", n_objects)
print(None if props is None else props.round(3))
```

### Part b

```python
# Part b. Keep only objects with a plausible area.

cells = props[(props["area"] >= MIN_AREA) & (props["area"] <= MAX_AREA)].reset_index(drop=True)
n_cells = int(len(cells))
n_dropped = int(len(props) - len(cells))

print("objects measured:", n_objects)
print("cells kept      :", n_cells)
print("rows dropped    :", n_dropped)
print(cells)
```

### Part c

```python
# Part c. Summarise the surviving cells.

mean_area = float(cells["area"].mean())
mean_ecc = float(cells["eccentricity"].mean())
mean_intensity = float(cells["mean_intensity"].mean())

print("mean area       : %s px" % mean_area)
print("mean eccentricity: %s" % mean_ecc)
print("mean intensity  : %s" % mean_intensity)
```

---

## W4D4. The same segmentation in OpenCV, and where the two libraries disagree

### Part a

```python
# Part a. Blur, then threshold with Otsu, the OpenCV way.

blurred = cv2.GaussianBlur(field, (5, 5), 0)
t_cv, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

print("threshold chosen:", t_cv)
print("binary dtype    :", None if binary is None else binary.dtype)
print("unique values   :", None if binary is None else np.unique(binary).tolist())
```

### Part b

```python
# Part b. Trace the outlines.

contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
n_contours = int(len(contours))

print("contours found   :", n_contours)
print("skimage found    :", SK_N_CELLS, "cells after its own area filter")
```

### Part c

```python
# Part c. Contour areas, filtered, and compared with skimage.

areas = [float(cv2.contourArea(c)) for c in contours]
_big = [a for a in areas if a >= MIN_AREA]
n_cells_cv = int(len(_big))
area_ratio = float(sum(_big) / SK_TOTAL_AREA)

print("all contour areas :", None if areas is None else sorted(round(a, 1) for a in areas))
print("cells by OpenCV   :", n_cells_cv)
print("cells by skimage  :", SK_N_CELLS)
print("area ratio cv2/skimage: %s" % area_ratio)
```

---

## W4D5. A whole imaging study: fields inside animals inside groups

### Part a

```python
# Part a. Segment every field and summarise each one.

rows = []
for f in fields:
    cells = segment_field(f["IMAGE"])
    rows.append({
        "ANIMAL": f["ANIMAL"],
        "GROUP": f["GROUP"],
        "FIELD": f["FIELD"],
        "N_CELLS": int(len(cells)),
        "MEAN_INTENSITY": float(cells["mean_intensity"].mean()),
    })

per_field = (pd.DataFrame(rows)
             .sort_values(["ANIMAL", "FIELD"])
             .reset_index(drop=True))

print(per_field)
```

### Part b

```python
# Part b. One row per animal.

per_animal = (per_field
              .groupby(["ANIMAL", "GROUP"])
              .agg(N_FIELDS=("MEAN_INTENSITY", "size"),
                   TOTAL_CELLS=("N_CELLS", "sum"),
                   MEAN_INTENSITY=("MEAN_INTENSITY", "mean"))
              .reset_index()
              .sort_values("ANIMAL")
              .reset_index(drop=True))

print(per_animal)
```

### Part c

```python
# Part c. One row per group, with the SD across animals.

group_stats = (per_animal
               .groupby("GROUP")
               .agg(MEAN=("MEAN_INTENSITY", "mean"),
                    SD=("MEAN_INTENSITY", "std"),
                    N_ANIMALS=("MEAN_INTENSITY", "size"))
               .reset_index())

group_stats["_ord"] = group_stats["GROUP"].map({g: i for i, g in enumerate(GROUP_ORDER)})
group_stats = (group_stats.sort_values("_ord")
               .drop(columns=["_ord"])
               .reset_index(drop=True))

print(group_stats)
```
