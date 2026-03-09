from __future__ import annotations

import colorsys
import math
import numpy as np

from config import (
    SIZE_MIN, SIZE_MAX, SPEED_MIN, SPEED_MAX,
    VIS_RANGE_MIN, VIS_RANGE_MAX, VIS_ANGLE_MIN, VIS_ANGLE_MAX,
    REPR_THRESH_MIN, REPR_THRESH_MAX, OFFSPRING_MIN, OFFSPRING_MAX,
    GESTATION_MIN, GESTATION_MAX, MUTABILITY_MIN, MUTABILITY_MAX,
    BASE_SPEED_PX, ENTITY_RADIUS_MULT, BASE_ATTACK_DMG, HP_PER_SIZE,
    NN_INPUT, NN_HIDDEN, NN_OUTPUT,
)

# ── Gene indices ──────────────────────────────────────────────────────────────
GENE_COUNT   = 13
G_SIZE       = 0
G_SPEED      = 1
G_VIS_RANGE  = 2
G_VIS_ANGLE  = 3
G_DIET       = 4   # 0 = pure herbivore, 1 = pure carnivore
G_AGGRESS    = 5
G_REPR_THR   = 6
G_OFFSPRING  = 7
G_GESTATION  = 8
G_MUTABILI   = 9
G_HUE        = 10  # family colour identity; drifts with mutation
G_SOCIALITY  = 11  # 0 = loner, 1 = highly social (flocks with kin, pack bonuses)
G_KIN_PROT   = 12  # 0 = no kin recognition, 1 = never attacks genetic kin


class Genome:
    """
    Fixed-length float array in [0, 1].
    All phenotype accessors map to biologically meaningful ranges.

    Phenotypes are computed once at construction and cached in slots so that
    every property read is an O(1) attribute lookup, not a computation.
    Genes only ever change via mutation(), which returns a new Genome.
    """
    __slots__ = (
        "genes",
        # cached phenotypes ──────────────────────────────────────────────────
        "_size", "_speed_raw", "_max_speed",
        "_vision_range", "_vision_half_angle",
        "_diet", "_aggression", "_sociality", "_kin_protection",
        "_reproduce_threshold", "_offspring_count",
        "_gestation", "_mutability", "_hue",
        "_radius", "_max_hp",
        "_plant_efficiency", "_meat_efficiency", "_attack_damage",
        "_color",
        # neural-network brain ───────────────────────────────────────────────
        "w1", "b1",   # input→hidden  shapes (NN_HIDDEN, NN_INPUT) and (NN_HIDDEN,)
        "w2", "b2",   # hidden→output shapes (NN_OUTPUT, NN_HIDDEN) and (NN_OUTPUT,)
    )

    def __init__(
        self,
        genes: np.ndarray | None = None,
        w1: np.ndarray | None = None,
        b1: np.ndarray | None = None,
        w2: np.ndarray | None = None,
        b2: np.ndarray | None = None,
    ) -> None:
        if genes is None:
            self.genes: np.ndarray = np.random.uniform(0.0, 1.0, GENE_COUNT)
        else:
            self.genes = np.asarray(genes, dtype=np.float64)

        # NN weights — warm-start from designed defaults + diversity noise
        if w1 is None:
            dw1, db1, dw2, db2 = Genome._default_weights()
            noise = 0.10   # σ=0.30 drowns the warm-start signal; 0.10 keeps designed behaviour intact
            self.w1 = dw1 + np.random.normal(0.0, noise, dw1.shape)
            self.b1 = db1 + np.random.normal(0.0, noise, db1.shape)
            self.w2 = dw2 + np.random.normal(0.0, noise, dw2.shape)
            self.b2 = db2 + np.random.normal(0.0, noise, db2.shape)
        else:
            self.w1 = w1
            self.b1 = b1
            self.w2 = w2
            self.b2 = b2

        self._compute_phenotypes()

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _default_weights():
        """
        Manually designed weight matrices encoding basic survival instincts.
        Used as warm-start for generation-0 entities (Gaussian noise added on top).

        Observation layout (18 inputs):
          0  hunger      1  hp_frac    2  diet
          3,4 food_cos/sin   5  food_dist
          6,7 corpse_cos/sin 8  corpse_dist
          9,10 threat_cos/sin 11 threat_dist
          12,13 prey_cos/sin  14 prey_dist
          15,16 kin_cos/sin   17 kin_dist

        Hidden units (8):
          h0 food cos · h1 food sin
          h2 prey cos · h3 prey sin
          h4 attack gate (prey dist × diet × hunger, biased off by default)
          h5 −threat cos (flee x) · h6 −threat sin (flee y)
          h7 corpse cos (scavenging)
        """
        w1 = np.zeros((NN_HIDDEN, NN_INPUT))
        b1 = np.zeros(NN_HIDDEN)
        w2 = np.zeros((NN_OUTPUT, NN_HIDDEN))
        b2 = np.zeros(NN_OUTPUT)

        # h0/h1: track nearest food direction
        w1[0, 3] = 2.0;  w1[1, 4] = 2.0
        # h2/h3: track nearest prey direction (stronger so carnivores commit to the chase)
        w1[2, 12] = 3.0; w1[3, 13] = 3.0
        # h4: attack gate — fires strongly when prey is close (prey_dist weight 2.5)
        # so the gate is decisive at close range without needing prey to be literally adjacent.
        w1[4, 14] = 2.5; w1[4, 2] = 2.0; w1[4, 0] = 1.5; b1[4] = -1.8
        # h5/h6: flee direction (inverted threat → negative = move away)
        w1[5, 9]  = -2.5; w1[6, 10] = -2.5
        # h7: track nearest corpse — both cos AND sin so feeding works in all directions
        w1[7, 6]  = 2.0;  w1[7, 7] = 2.0

        # out[0] move_dx: food + prey (stronger) + flee + corpse
        w2[0, 0] = 1.5; w2[0, 2] = 2.0; w2[0, 5] = 1.2; w2[0, 7] = 1.0
        # out[1] move_dy: food + prey (stronger) + flee + corpse (was missing corpse y)
        w2[1, 1] = 1.5; w2[1, 3] = 2.0; w2[1, 6] = 1.2; w2[1, 7] = 1.0
        # out[2] attack: lower bias so a starving carnivore near prey reliably fires
        w2[2, 4] = 3.0; b2[2] = -0.5

        return w1, b1, w2, b2

    @staticmethod
    def _lerp(raw: float, lo: float, hi: float) -> float:
        return lo + raw * (hi - lo)

    def _compute_phenotypes(self) -> None:
        """Pre-compute all derived values once from the current gene array."""
        g = self.genes
        self._size         = SIZE_MIN  + g[G_SIZE]    * (SIZE_MAX  - SIZE_MIN)
        self._speed_raw    = SPEED_MIN + g[G_SPEED]   * (SPEED_MAX - SPEED_MIN)
        self._max_speed    = self._speed_raw * BASE_SPEED_PX / math.sqrt(self._size)
        self._vision_range = VIS_RANGE_MIN + g[G_VIS_RANGE] * (VIS_RANGE_MAX - VIS_RANGE_MIN)
        _deg               = VIS_ANGLE_MIN + g[G_VIS_ANGLE] * (VIS_ANGLE_MAX - VIS_ANGLE_MIN)
        self._vision_half_angle = math.radians(_deg / 2.0)
        self._diet         = float(g[G_DIET])
        self._aggression   = float(g[G_AGGRESS])
        self._sociality    = float(g[G_SOCIALITY])
        self._kin_protection = float(g[G_KIN_PROT])
        self._reproduce_threshold = (REPR_THRESH_MIN
                                     + g[G_REPR_THR] * (REPR_THRESH_MAX - REPR_THRESH_MIN))
        self._offspring_count = max(1, round(
            OFFSPRING_MIN + g[G_OFFSPRING] * (OFFSPRING_MAX - OFFSPRING_MIN)))
        self._gestation    = GESTATION_MIN + g[G_GESTATION] * (GESTATION_MAX - GESTATION_MIN)
        self._mutability   = MUTABILITY_MIN + g[G_MUTABILI] * (MUTABILITY_MAX - MUTABILITY_MIN)
        self._hue          = float(g[G_HUE])
        self._radius       = self._size * ENTITY_RADIUS_MULT
        self._max_hp       = self._size * HP_PER_SIZE
        # Quadratic specialisation bonus: pure specialists are 4× more efficient
        # than omnivores.  Total efficiency at diet=0.5 is 0.5 vs 1.0 for specialists.
        self._plant_efficiency = (1.0 - self._diet) ** 2
        self._meat_efficiency  = self._diet ** 2
        _diet_mult         = 0.1 + 0.9 * self._diet
        self._attack_damage = BASE_ATTACK_DMG * _diet_mult * self._size
        # HSV colour (also used every render frame — worth caching)
        _r, _g, _b = colorsys.hsv_to_rgb(
            self._hue,
            0.50 + 0.50 * self._diet,
            0.65 + 0.25 * float(g[G_SIZE]),
        )
        self._color = (int(_r * 255), int(_g * 255), int(_b * 255))

    # ── Body ──────────────────────────────────────────────────────────────────

    @property
    def size(self) -> float:
        return self._size

    @property
    def speed_raw(self) -> float:
        """Genome speed score in [SPEED_MIN, SPEED_MAX]."""
        return self._speed_raw

    @property
    def max_speed(self) -> float:
        """Pixels/second; larger bodies are slower."""
        return self._max_speed

    @property
    def vision_range(self) -> float:
        return self._vision_range

    @property
    def vision_half_angle(self) -> float:
        """Half of total FOV in radians. >= π means effectively omnidirectional."""
        return self._vision_half_angle

    # ── Diet & behaviour ──────────────────────────────────────────────────────

    @property
    def diet(self) -> float:
        """0.0 = pure herbivore, 1.0 = pure carnivore."""
        return self._diet

    @property
    def aggression(self) -> float:
        return self._aggression

    @property
    def sociality(self) -> float:
        """0 = complete loner, 1 = highly social (flocks with kin, pack bonuses)."""
        return self._sociality

    @property
    def kin_protection(self) -> float:
        """0 = indifferent to kin, 1 = never attacks genetic kin (including own offspring)."""
        return self._kin_protection

    # ── Reproduction ──────────────────────────────────────────────────────────

    @property
    def reproduce_threshold(self) -> float:
        return self._reproduce_threshold

    @property
    def offspring_count(self) -> int:
        return self._offspring_count

    @property
    def gestation(self) -> float:
        """Energy invested per offspring (parent loses this, offspring starts with this)."""
        return self._gestation

    @property
    def mutability(self) -> float:
        return self._mutability

    @property
    def hue(self) -> float:
        """Cosmetic gene in [0, 1]; drifts with mutation."""
        return self._hue

    # ── Derived phenotypes ────────────────────────────────────────────────────

    @property
    def radius(self) -> float:
        return self._radius

    @property
    def max_hp(self) -> float:
        """Maximum hit points; scales with body size."""
        return self._max_hp

    @property
    def plant_efficiency(self) -> float:
        """Energy fraction extracted from plant food. Peaks at diet=0."""
        return self._plant_efficiency

    @property
    def meat_efficiency(self) -> float:
        """Energy fraction extracted from meat/corpses. Peaks at diet=1."""
        return self._meat_efficiency

    @property
    def attack_damage(self) -> float:
        """
        Damage per strike.
        Pure herbivores deal 10 % of base; pure carnivores deal 100 %.
        Larger bodies hit harder.
        """
        return self._attack_damage

    @property
    def color(self) -> tuple[int, int, int]:
        """
        HSV colour so genetic relatives share a visible hue.
          H = hue gene  — family/breed identity; drifts slowly across generations
          S = diet      — herbivores are pastel (S≈0.5), carnivores vivid (S≈1.0)
          V = size gene — bigger bodies are slightly brighter
        """
        return self._color

    # ── Brain ─────────────────────────────────────────────────────────────────

    def forward(self, obs: np.ndarray) -> np.ndarray:
        """
        Single forward pass through the two-layer network.

        obs  : shape (NN_INPUT,)  — normalised sensory + state inputs
        returns shape (NN_OUTPUT,) — move_dx, move_dy, attack_signal in [-1, 1]
        """
        h = np.tanh(self.w1 @ obs + self.b1)
        return np.tanh(self.w2 @ h + self.b2)

    # ── Mutation ──────────────────────────────────────────────────────────────

    def mutate(self, preserve_diet: bool = False) -> "Genome":
        """Return a child genome with Gaussian noise applied to genes and NN weights."""
        m = self._mutability
        child_genes = np.clip(self.genes + np.random.normal(0.0, m, GENE_COUNT), 0.0, 1.0)
        if preserve_diet:
            child_genes[G_DIET] = self.genes[G_DIET]
        return Genome(
            child_genes,
            self.w1 + np.random.normal(0.0, m, self.w1.shape),
            self.b1 + np.random.normal(0.0, m, self.b1.shape),
            self.w2 + np.random.normal(0.0, m, self.w2.shape),
            self.b2 + np.random.normal(0.0, m, self.b2.shape),
        )
