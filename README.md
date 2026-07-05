# How an MRI Works — a first-principles explainer

An 8-scene educational video explaining how magnetic resonance imaging works
from first principles, in the visual style of 3Blue1Brown. Animated with
[Manim Community Edition](https://www.manim.community/).

## Scenes

| # | File | Scene class | Teaches |
|---|------|-------------|---------|
| 1 | `scene1_proton_as_magnet.py` | `ProtonAsMagnet` | A proton acts like a tiny bar magnet; random cloud cancels; in a field B₀ a slight statistical excess aligns, giving net magnetization **M** ∥ B₀. |

*(Scenes 2–8 to follow: Precession, RF excitation, relaxation T₁/T₂, signal/FID, spatial encoding, k-space, image reconstruction.)*

## Rendering

Manim CE 0.20.1 is installed for `python`. Video encodes via bundled PyAV — no
system ffmpeg needed.

```bash
# Iterate fast (480p15):
python -m manim -ql scene1_proton_as_magnet.py ProtonAsMagnet

# Final pass (1080p30):
python -m manim -qh --fps 30 scene1_proton_as_magnet.py ProtonAsMagnet
```

Each scene file also defines `Snap*` helper scenes for still previews:

```bash
python -m manim -s -ql scene1_proton_as_magnet.py SnapHero
```

## Conventions

- **One scene per file**, self-contained, so one failure never blocks the others.
- Target **1080p / 30 fps**; iterate at `-ql`, final HQ pass later.
- Dark 3b1b background, high-contrast colours, physically faithful (e.g. field
  alignment is a *statistical excess*, not every proton snapping parallel).
- **No LaTeX** is installed on this machine, so labels use Unicode `Text`
  (`MathTex`/`Tex` would fail). Install MiKTeX/TinyTeX if real typeset equations
  are needed later.
- `media/` and `renders/` are git-ignored (regenerated from the sources).
