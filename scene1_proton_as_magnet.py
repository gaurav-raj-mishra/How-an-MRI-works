"""
Scene 1 — ProtonAsMagnet  (MRI-from-first-principles, 3Blue1Brown style)

Concept: A proton acts like a tiny bar magnet. On their own, protons point in
random directions and cancel out. Immersed in a strong field B0, a *slight
statistical excess* aligns with the field, giving a small net magnetization M.

Render (low quality while iterating):
    python -m manim -ql -p scene1_proton_as_magnet.py ProtonAsMagnet

Static snapshot frames (for review):
    python -m manim -s -ql scene1_proton_as_magnet.py SnapHero
    python -m manim -s -ql scene1_proton_as_magnet.py SnapCloudSum
    python -m manim -s -ql scene1_proton_as_magnet.py SnapMagnetization
"""

import numpy as np
from manim import *

# ---------------------------------------------------------------------------
# Palette — dark 3b1b-ish slate background with high-contrast accents.
# ---------------------------------------------------------------------------
config.background_color = "#0e1117"

N_COL = "#ff5a5f"        # North pole / arrow tip (warm red)
S_COL = "#4d9de0"        # South pole (cool blue)
B0_COL = "#f4d35e"       # external field B0 (gold)
M_COL = "#7be08a"        # net magnetization M (green, the "answer" colour)
SPIN_COL = "#9aa7b8"     # subtle spin indicator (muted grey-blue)
FAINT = "#6b7683"        # captions / de-emphasised text

# Reproducible "randomness" so the composition is stable between renders.
RNG = np.random.default_rng(7)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def make_dipole(length=0.72, stroke=5, tip_len=0.18):
    """A magnetic-dipole arrow: blue (S) tail -> red (N) tip, pointing +Y.

    Returned as a single Mobject (Arrow) so callers can rotate about its
    centre to set orientation and translate freely.
    """
    half = length / 2.0
    arr = Arrow(
        start=[0, -half, 0],
        end=[0, half, 0],
        buff=0,
        stroke_width=stroke,
        max_tip_length_to_length_ratio=0.4,
        tip_length=tip_len,
    )
    # Blue at the S (tail) end grading to red at the N (tip) end.
    arr.set_color_by_gradient(S_COL, N_COL)
    arr.tip.set_color(N_COL)  # guarantee the N tip reads as red
    return arr


def field_label(fs_main=44, fs_sub=26, color=B0_COL):
    """Render 'B₀' as B + a true subscript 0 (the font's ₀ glyph looks like an
    'o', so we typeset the subscript by hand)."""
    b = Text("B", font_size=fs_main, color=color, weight=BOLD)
    z = Text("0", font_size=fs_sub, color=color, weight=BOLD)
    z.next_to(b, RIGHT, buff=0.03).align_to(b, DOWN).shift(DOWN * 0.05)
    return VGroup(b, z)


def make_hero_proton():
    """The recurring 'character': a labelled proton drawn as a dipole with a
    spin indicator and N/S pole labels. Returns a VGroup centred at origin."""
    dot = Dot(ORIGIN, radius=0.09, color=WHITE)

    dipole = make_dipole(length=1.6, stroke=8, tip_len=0.32)

    # Pole labels sitting just beyond each end of the moment arrow.
    n_lbl = Text("N", font_size=26, color=N_COL, weight=BOLD).next_to(dipole, UP, buff=0.08)
    s_lbl = Text("S", font_size=26, color=S_COL, weight=BOLD).next_to(dipole, DOWN, buff=0.08)

    # Subtle spin indicator: a partial circular arrow curling around the dot.
    spin = Arc(radius=0.40, start_angle=PI / 3, angle=1.4 * PI, color=SPIN_COL, stroke_width=3)
    spin.set_fill(opacity=0.0)          # arc outline only — no filled blob
    spin.set_stroke(SPIN_COL, width=3, opacity=0.85)
    spin.add_tip(tip_length=0.13)

    group = VGroup(spin, dipole, dot, n_lbl, s_lbl)
    return group


def random_angles_balanced(n):
    """n orientation angles (radians) that sum to ~zero as unit vectors.

    Built as pairs (theta, theta+pi) so the net vector cancels exactly — this
    lets the 'no net magnetism' sum genuinely collapse to zero rather than the
    sqrt(N) drift a raw random walk would show.
    """
    base = RNG.uniform(0, TAU, size=n // 2)
    angles = np.concatenate([base, base + PI])
    if n % 2:  # odd n: append one more balanced-ish angle
        angles = np.append(angles, RNG.uniform(0, TAU))
    RNG.shuffle(angles)
    return angles[:n]


def bias_angles_toward_up(angles, excess_frac=0.28, tilt=0.9):
    """Given random angles, tip a slight statistical majority toward +Y (up).

    Returns new angles. A fraction (~55% up vs ~45% down after biasing) is
    nudged so the net vector points weakly upward — a statistical excess, not
    a snap-to-alignment.
    """
    new = np.array(angles, dtype=float)
    n = len(new)
    order = RNG.permutation(n)
    n_up = int(n * (0.5 + excess_frac / 2))  # slight majority pulled up
    for k, i in enumerate(order):
        # Nudge each dipole a *small* amount toward vertical (up or down),
        # with more of them pulled up than down.
        target = PI / 2 if k < n_up else -PI / 2
        # Move only partway toward the target (statistical, not absolute).
        pull = tilt * RNG.uniform(0.25, 0.6)
        # shortest-path interpolation of angle toward target
        d = (target - new[i] + PI) % TAU - PI
        new[i] = new[i] + pull * d
    return new


def cloud_positions(n, x_range=(-5.2, 5.2), y_range=(-2.6, 2.6), min_d=0.62):
    """Poisson-ish scattered positions (rejection sampled) to avoid overlap."""
    pts = []
    tries = 0
    while len(pts) < n and tries < n * 400:
        tries += 1
        p = np.array([
            RNG.uniform(*x_range),
            RNG.uniform(*y_range),
            0.0,
        ])
        if all(np.linalg.norm(p - q) > min_d for q in pts):
            pts.append(p)
    while len(pts) < n:  # fallback if packing got tight
        pts.append(np.array([RNG.uniform(*x_range), RNG.uniform(*y_range), 0.0]))
    return pts


def resultant_arrow(angles, origin, unit=0.5, color=M_COL, scale=1.0, stroke=10,
                    vertical=False):
    """Vector sum of unit dipoles (given by angles) drawn from `origin`.

    `vertical=True` keeps only the component along B0 (up). This is the
    physically honest picture for M: the transverse components of the tilted
    dipoles precess out of phase and cancel, leaving M parallel to B0.
    """
    vx = np.sum(np.cos(angles)) * unit * scale
    vy = np.sum(np.sin(angles)) * unit * scale
    if vertical:
        vx = 0.0
    vec = np.array([vx, vy, 0.0])
    end = origin + vec
    arr = Arrow(origin, end, buff=0, color=color, stroke_width=stroke,
                max_tip_length_to_length_ratio=0.3)
    return arr, np.linalg.norm(vec)


# ---------------------------------------------------------------------------
# Shared cloud state (positions + angle sets), so the animated scene and the
# static snapshot scenes render the *same* composition.
# ---------------------------------------------------------------------------
N_CLOUD = 46
_POS = cloud_positions(N_CLOUD)
_ANG0 = random_angles_balanced(N_CLOUD)          # random, cancels to ~0
_ANG1 = bias_angles_toward_up(_ANG0)             # slight excess toward B0


def build_cloud(angles):
    """Return a VGroup of small dipoles at _POS with the given orientations."""
    g = VGroup()
    for p, a in zip(_POS, angles):
        d = make_dipole(length=0.5, stroke=4, tip_len=0.13)
        d.rotate(a - PI / 2)   # dipoles are built pointing +Y (angle=pi/2)
        d.move_to(p)
        g.add(d)
    return g


# ---------------------------------------------------------------------------
# Main animated scene
# ---------------------------------------------------------------------------
class ProtonAsMagnet(Scene):
    def construct(self):
        self.beat1_single_proton()
        self.beat2_random_cloud_cancels()
        self.beat3_introduce_B0()
        self.beat4_net_magnetization()
        self.beat5_honesty_caption()

    # -- Beat 1 -------------------------------------------------------------
    def beat1_single_proton(self):
        hero = make_hero_proton().scale(1.1).move_to(ORIGIN)
        spin, dipole, dot, n_lbl, s_lbl = hero

        title = Text("¹H — a proton", font_size=34, color=WHITE)
        title.to_edge(UP, buff=0.7)

        self.play(FadeIn(dot, scale=0.5), Write(title), run_time=1.4)
        self.play(GrowArrow(dipole), run_time=1.2)
        self.wait(1.5)  # "this arrow is its magnetic moment"
        self.play(FadeIn(n_lbl, shift=UP * 0.15), FadeIn(s_lbl, shift=DOWN * 0.15), run_time=0.8)
        self.play(Create(spin), run_time=1.0)
        self.wait(4.0)  # hold on the hero — narration introduces the proton

        caption = Text("behaves like a tiny bar magnet", font_size=26, color=FAINT)
        caption.next_to(hero, DOWN, buff=0.7)
        self.play(FadeIn(caption, shift=UP * 0.1), run_time=0.9)
        self.wait(4.5)

        # Shrink the hero away to make room for the cloud.
        self.hero = hero
        self.play(
            FadeOut(caption),
            FadeOut(title),
            hero.animate.scale(0.32).to_corner(UL, buff=0.6).set_opacity(0.0),
            run_time=1.2,
        )
        self.remove(hero)

    # -- Beat 2 -------------------------------------------------------------
    def beat2_random_cloud_cancels(self):
        cloud = build_cloud(_ANG0)
        self.cloud = cloud

        header = Text("many protons — random directions", font_size=28, color=WHITE).to_edge(UP, buff=0.5)
        self.play(Write(header), run_time=1.0)
        self.play(LaggedStart(*[GrowArrow(d) for d in cloud], lag_ratio=0.04, run_time=2.6))
        self.wait(5.0)  # let the eye register the randomness

        # Vector sum: copy every dipole to a common origin (a starburst), then
        # draw the resultant — which is ~zero.
        sum_origin = np.array([0.0, -0.4, 0.0])
        copies = VGroup()
        anims = []
        for d, a in zip(cloud, _ANG0):
            c = make_dipole(length=0.55, stroke=3, tip_len=0.12).set_opacity(0.55)
            c.rotate(a - PI / 2)
            # place tail at common origin: shift so its centre sits half-length out
            c.move_to(sum_origin + 0.275 * np.array([np.cos(a), np.sin(a), 0]))
            copies.add(c)
            anims.append(TransformFromCopy(d, c))

        sum_label = Text("add them up", font_size=26, color=FAINT).to_edge(UP, buff=0.5)
        self.play(
            FadeOut(header),
            cloud.animate.set_opacity(0.28),
            run_time=0.8,
        )
        self.play(FadeIn(sum_label, shift=DOWN * 0.1), LaggedStart(*anims, lag_ratio=0.03, run_time=2.4))
        self.wait(1.5)

        res, mag = resultant_arrow(_ANG0, sum_origin, unit=0.5, color=WHITE, stroke=9)
        zero_lbl = Text("Σ μ  ≈  0", font_size=34, color=WHITE)
        zero_lbl.next_to(sum_origin, DOWN, buff=1.2)
        no_net = Text("no net magnetism", font_size=30, color=FAINT)
        no_net.next_to(zero_lbl, DOWN, buff=0.3)

        # Resultant is essentially a point; pulse a small marker instead of a
        # misleading long arrow.
        dot_zero = Dot(sum_origin, radius=0.12, color=WHITE)
        self.play(FadeIn(dot_zero, scale=0.4), Write(zero_lbl), run_time=1.2)
        self.play(FadeIn(no_net, shift=UP * 0.1), run_time=0.8)
        self.wait(5.0)  # sit on "no net magnetism"

        # Clean up the sum apparatus; keep the cloud for the B0 beat.
        self.play(
            FadeOut(copies), FadeOut(dot_zero), FadeOut(zero_lbl),
            FadeOut(no_net), FadeOut(sum_label),
            cloud.animate.set_opacity(1.0),
            run_time=1.0,
        )

    # -- Beat 3 -------------------------------------------------------------
    def beat3_introduce_B0(self):
        # Big semi-transparent vertical field arrow on the right, labelled B0.
        b0 = Arrow(
            start=[5.6, -2.7, 0], end=[5.6, 2.9, 0], buff=0,
            color=B0_COL, stroke_width=14, max_tip_length_to_length_ratio=0.08,
        ).set_opacity(0.55)
        b0_lbl = field_label().next_to(b0, LEFT, buff=0.25).shift(UP * 2.2)
        self.b0 = VGroup(b0, b0_lbl)

        header = Text("switch on a strong field", font_size=28, color=WHITE).to_edge(UP, buff=0.5)
        self.play(Write(header), run_time=0.9)
        self.play(GrowArrow(b0), FadeIn(b0_lbl, shift=LEFT * 0.15), run_time=1.6)
        self.wait(2.0)  # B0 is now present; narration sets up the response

        # The cloud responds: each dipole rotates toward its biased angle. A
        # slight majority ends up tipped toward B0 (up) — a statistical excess.
        rot_anims = []
        for d, a0, a1 in zip(self.cloud, _ANG0, _ANG1):
            rot_anims.append(Rotate(d, angle=(a1 - a0), about_point=d.get_center()))
        self.play(LaggedStart(*rot_anims, lag_ratio=0.02, run_time=3.0))

        note = Text("a slight majority tips toward the field", font_size=26, color=FAINT)
        note.to_edge(DOWN, buff=0.5)
        self.play(FadeOut(header), FadeIn(note, shift=UP * 0.1), run_time=0.9)
        self.wait(5.0)  # emphasise: a *slight* excess, not a snap-to-alignment
        self.play(FadeOut(note), run_time=0.7)

    # -- Beat 4 -------------------------------------------------------------
    def beat4_net_magnetization(self):
        sum_origin = np.array([0.0, -0.3, 0.0])
        copies = VGroup()
        anims = []
        for d, a in zip(self.cloud, _ANG1):
            c = make_dipole(length=0.55, stroke=3, tip_len=0.12).set_opacity(0.5)
            c.rotate(a - PI / 2)
            c.move_to(sum_origin + 0.275 * np.array([np.cos(a), np.sin(a), 0]))
            copies.add(c)
            anims.append(TransformFromCopy(d, c))

        sum_label = Text("add them up again", font_size=26, color=FAINT).to_edge(UP, buff=0.5)
        self.play(
            FadeIn(sum_label, shift=DOWN * 0.1),
            self.cloud.animate.set_opacity(0.3),
            run_time=0.8,
        )
        self.play(LaggedStart(*anims, lag_ratio=0.03, run_time=2.4))
        self.wait(0.3)

        # This time the resultant is small but clearly nonzero, pointing up
        # along B0 (transverse parts cancel — see resultant_arrow docstring).
        res, mag = resultant_arrow(_ANG1, sum_origin, unit=0.55, color=M_COL,
                                   scale=1.0, stroke=11, vertical=True)

        # Glow: a thicker, translucent copy behind the crisp arrow.
        glow = res.copy().set_stroke(width=26, opacity=0.28, color=M_COL)
        m_lbl = Text("M", font_size=44, color=M_COL, weight=BOLD).next_to(res.get_end(), RIGHT, buff=0.25)
        m_sub = Text("net magnetization", font_size=22, color=M_COL).next_to(
            m_lbl, DOWN, buff=0.2, aligned_edge=LEFT)  # sit right of the M shaft

        self.play(FadeOut(copies), FadeOut(sum_label), run_time=0.6)
        self.play(GrowArrow(res), FadeIn(glow), run_time=1.4)
        self.play(Write(m_lbl), FadeIn(m_sub, shift=UP * 0.1), run_time=1.0)
        self.wait(4.5)  # the payoff: a small but real net vector along B0

        # Gentle pulse to emphasise M, held beside B0.
        self.play(
            glow.animate.set_stroke(width=34, opacity=0.42),
            rate_func=there_and_back, run_time=1.4,
        )
        self.wait(2.0)

        self.m_group = VGroup(glow, res, m_lbl, m_sub)
        self.play(self.cloud.animate.set_opacity(0.85), run_time=0.8)
        self.wait(4.0)  # hold on M beside B0 to close the beat

    # -- Beat 5 -------------------------------------------------------------
    def beat5_honesty_caption(self):
        # 3b1b-style aside: acknowledge the quantum truth, keep it unobtrusive.
        box_text = VGroup(
            Text("The honest picture is quantum-mechanical.", font_size=24, color=FAINT),
            Text("But this net-vector M is all the rest of the video needs.",
                 font_size=24, color=FAINT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        box_text.to_edge(DOWN, buff=0.55)
        underline = Line(box_text.get_corner(UL) + LEFT * 0.1 + UP * 0.12,
                         box_text.get_corner(UR) + RIGHT * 0.1 + UP * 0.12,
                         color=FAINT, stroke_width=1).set_opacity(0.4)

        self.play(FadeIn(box_text, shift=UP * 0.1), Create(underline), run_time=1.4)
        self.wait(6.5)  # aside stays up long enough to read comfortably
        self.play(FadeOut(box_text), FadeOut(underline), run_time=1.0)
        self.wait(0.8)


# ---------------------------------------------------------------------------
# Static snapshot scenes (render with -s to get review stills)
# ---------------------------------------------------------------------------
class SnapHero(Scene):
    def construct(self):
        hero = make_hero_proton().scale(1.1)
        title = Text("¹H — a proton", font_size=34, color=WHITE).to_edge(UP, buff=0.7)
        caption = Text("behaves like a tiny bar magnet", font_size=26, color=FAINT).next_to(hero, DOWN, buff=0.7)
        self.add(hero, title, caption)


class SnapCloudSum(Scene):
    def construct(self):
        cloud = build_cloud(_ANG0).set_opacity(0.3)
        sum_origin = np.array([0.0, -0.4, 0.0])
        copies = VGroup()
        for a in _ANG0:
            c = make_dipole(length=0.55, stroke=3, tip_len=0.12).set_opacity(0.55)
            c.rotate(a - PI / 2)
            c.move_to(sum_origin + 0.275 * np.array([np.cos(a), np.sin(a), 0]))
            copies.add(c)
        dot_zero = Dot(sum_origin, radius=0.12, color=WHITE)
        zero_lbl = Text("Σ μ  ≈  0", font_size=34, color=WHITE).next_to(sum_origin, DOWN, buff=1.2)
        no_net = Text("no net magnetism", font_size=30, color=FAINT).next_to(zero_lbl, DOWN, buff=0.3)
        self.add(cloud, copies, dot_zero, zero_lbl, no_net)


class SnapMagnetization(Scene):
    def construct(self):
        cloud = build_cloud(_ANG1).set_opacity(0.85)
        b0 = Arrow(start=[5.6, -2.7, 0], end=[5.6, 2.9, 0], buff=0, color=B0_COL,
                   stroke_width=14, max_tip_length_to_length_ratio=0.08).set_opacity(0.55)
        b0_lbl = field_label().next_to(b0, LEFT, buff=0.25).shift(UP * 2.2)
        sum_origin = np.array([0.0, -0.3, 0.0])
        res, mag = resultant_arrow(_ANG1, sum_origin, unit=0.55, color=M_COL, stroke=11, vertical=True)
        glow = res.copy().set_stroke(width=26, opacity=0.28, color=M_COL)
        m_lbl = Text("M", font_size=44, color=M_COL, weight=BOLD).next_to(res.get_end(), RIGHT, buff=0.25)
        m_sub = Text("net magnetization", font_size=24, color=M_COL).next_to(m_lbl, DOWN, buff=0.2)
        self.add(cloud, b0, b0_lbl, glow, res, m_lbl, m_sub)
