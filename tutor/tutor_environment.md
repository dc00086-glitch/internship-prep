# How the student runs these, and what breaks

## Two routes, same notebooks

**Colab.** They open https://dc00086-glitch.github.io/internship-prep/, press
the Colab link on today's problem, and work in the browser. Nothing is
installed. Their edits live in their own Colab copy, not in the repository.

**Their own computer.** `python start.py` in the project folder installs
numpy, pandas, matplotlib, scipy, scikit-learn, scikit-image, OpenCV,
JupyterLab and PyTorch into a `.venv` inside the folder, then opens today's
problem in JupyterLab. The first run needs internet; every run after is
offline. `python start.py 12` opens a particular problem, `--list` prints the
schedule, `--browse` opens the one page view, `--foundations` opens the
foundations notebook, `--no-torch` skips the one large download, `--no-venv`
uses an existing Anaconda environment.

Either way the notebook files are identical, so advice never has to differ.

## How a problem notebook is laid out

| Cell | What it is |
|---|---|
| header | problem ID, topic, how to work it |
| Scenario | the pharmacology framing |
| How it works | how the package idea works, written out properly |
| Your task | parts a, b, c, with the exact variable names wanted |
| Setup | synthetic data from a fixed seed, no uploads |
| Toolkit | every move demonstrated on a tiny made up example |
| Part a, b, c | `# YOUR CODE` blanks, each with its own explanation |
| Check | `check()`, printing PASS, FAIL with a reason, or TODO, per part |
| Send to tutor | the block they paste into chat |

## How check() works, and what its answers mean

- **PASS** means the numbers are right. It does not mean the code is tidy.
- **FAIL** means a specific wrong thing was detected, and the message says
  which. The messages are written to diagnose without revealing, so a FAIL
  never contains the answer. Each week file lists every message a part can
  produce, which is the fastest way to know what the student actually did.
- **TODO** means the variable is still `None` or undefined, so they have not
  started that part or did not run the cell.
- An **unexpected exception** inside a part is reported as a crash with the
  exception text, which usually means a shape or type error upstream.

The expected answers are stored in the check cell as a base64 blob so they
cannot be read by accident. Tell the student not to decode it; it will not
teach them anything and it removes the only feedback loop they have.

## The schedule

One problem per weekday from the start date in `index.html`, currently
2026-10-19. Weekends hold on Friday's problem. `?n=3` on the site, or
`python start.py 3`, opens any problem out of order.

## Things that actually go wrong

**`NameError`.** Cells run out of order. The fix is Runtime, then Restart and
run all, in Colab, or Kernel, then Restart kernel and run all cells, in
JupyterLab. This is the single most common cause of confusing behaviour.

**A part says TODO when they think they answered it.** They edited the cell
and did not run it, or they ran the check cell before the part cell.

**`np.std` giving a smaller number than expected.** numpy divides by n unless
you pass `ddof=1`. Pandas `.std()` already divides by `n - 1`. Several checks
test for exactly this.

**SEM where SD was asked for.** The checks name it. SEM shrinks with sample
size and answers a different question.

**Shapes.** `axis=0` goes down the columns and `axis=1` across the rows, and
with a subjects by times matrix both are plausible numbers, so the mistake
does not raise. Tell them to print `.shape` before anything else.

**Matplotlib figures not appearing.** In JupyterLab they need nothing special,
`plt.show()` works. If a plot cell is slow or blank, it is usually that an
earlier cell raised.

**`No module named cv2` or similar, local only.** The install did not finish.
`python start.py --reinstall`.

**PyTorch missing, local only.** Only week 6 needs it. It is the one large
download and the launcher may have skipped it, which it says at the time.

**A solver result that moves when tolerances change.** Several problems ask for
`rtol=1e-8, atol=1e-10` or tighter and the checks are tight enough to notice,
because default tolerances leave error an optimiser will chase.

**Version differences.** The notebooks avoid APIs that moved: no
`squared=False`, no `np.trapz`, no `min_size=` on `remove_small_objects`. They
are verified on two package stacks, one Colab-like and one current. If
something still looks version dependent, it is worth reporting rather than
working around.

## Rebuilding, if they ever need a blank notebook back

`python3 tools/build.py` rewrites every problem notebook from its spec, which
restores the blanks and discards whatever they had typed. Their answers should
be copied out first. Problem and solution notebooks are generated from the same
spec, so a solution cannot drift from its problem.
