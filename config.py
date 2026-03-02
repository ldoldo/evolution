# ── Display ───────────────────────────────────────────────────────────────────
WIDTH       = 1400
HEIGHT      = 900
PANEL_WIDTH = 280           # right-hand stats panel
WORLD_WIDTH = WIDTH - PANEL_WIDTH   # 1120
WORLD_HEIGHT = HEIGHT
FPS         = 60
BG_COLOR    = (12, 12, 20)

# ── World / Food ──────────────────────────────────────────────────────────────
FOOD_SPAWN_RATE      = 8.0   # items per second (boosted automatically when herbivores scarce)
MAX_FOOD             = 450
HERB_LOW_THRESHOLD   = 12    # herbivore count below which food spawn is boosted
FOOD_SPAWN_HERB_BONUS = 0.4  # max extra spawn fraction (at herb=0 → ×1.4 rate)
FOOD_ENERGY          = 70.0
FOOD_RADIUS          = 4
FOOD_CLUSTER_CHANCE  = 0.72  # probability a new food item spawns near an existing one
FOOD_CLUSTER_SIGMA   = 50.0  # std-dev of Gaussian offset from anchor (pixels)

# ── Initial population ────────────────────────────────────────────────────────
INITIAL_POPULATION = 60
MIN_POPULATION     = 8      # respawn floor to prevent total extinction

# ── Genome parameter ranges ───────────────────────────────────────────────────
SIZE_MIN,      SIZE_MAX      = 0.2,  3.0
SPEED_MIN,     SPEED_MAX     = 0.5,  5.0
VIS_RANGE_MIN, VIS_RANGE_MAX = 20.0, 200.0
VIS_ANGLE_MIN, VIS_ANGLE_MAX = 30.0, 360.0   # degrees (total FOV)
REPR_THRESH_MIN, REPR_THRESH_MAX = 200.0, 600.0
OFFSPRING_MIN, OFFSPRING_MAX = 1, 4
GESTATION_MIN, GESTATION_MAX = 30.0, 90.0    # energy invested per offspring
MUTABILITY_MIN, MUTABILITY_MAX = 0.005, 0.12

# ── Speed / radius ────────────────────────────────────────────────────────────
BASE_SPEED_PX  = 42.0       # pixels/s per unit of speed_raw; scaled by 1/sqrt(size)
ENTITY_RADIUS_MULT = 7.0    # visual/collision radius = size * this

# ── Metabolism (energy per second) ───────────────────────────────────────────
BASE_METABOLISM      = 0.28
SIZE_COST            = 0.15   # × size²
MOVE_COST            = 0.12   # × normalised_speed × size   (only when moving)
DIET_METABOLISM_EXTRA = 0.08  # extra energy/s at diet=1.0; scales linearly with diet

# ── Combat ────────────────────────────────────────────────────────────────────
BASE_ATTACK_DMG         = 14.0
ATTACK_RANGE_EXTRA      = 8.0   # attack range = attacker.radius + target.radius + this
ATTACK_COOLDOWN         = 1.2   # seconds between attacks
MIN_AGGRESSION_TO_ATTACK = 0.25  # aggression gate; below this the entity won't attack
ATTACK_ENERGY_COST      = 2.0   # energy (stamina) spent per strike regardless of outcome
HP_PER_SIZE             = 40.0  # max HP = size * this; combat damage reduces HP, not energy
HP_REGEN_RATE           = 0.03  # fraction of effective_max_hp recovered per second
PACK_BONUS_PER_KIN  = 0.20  # +20% damage per kin supporting the attack
PACK_BONUS_RANGE    = 90.0  # px radius counted as "supporting range"
MAX_PACK_BONUS      = 4     # cap: 4 kin → ×1.80 damage (diminishing beyond that)
PACK_FLEE_PER_KIN    = 0.20  # +20% flee steering force per visible kin
PACK_DEFENSE_PER_KIN = 0.08  # -8% incoming damage per kin near the target
HERD_RETALIATION_PER_KIN = 0.30  # +30% retaliation damage per kin near the defender

# ── Corpse ────────────────────────────────────────────────────────────────────
CORPSE_DECAY       = 28.0   # seconds until full decay
CORPSE_BITE        = 18.0   # max energy extracted per tick per entity

# ── Reproduction ──────────────────────────────────────────────────────────────
SPAWN_SCATTER = 25.0        # max pixel offset from parent
MIN_PARENT_ENERGY_FRAC = 0.28  # parent must keep ≥ this fraction of reproduce_threshold
BIRTH_COOLDOWN = 8.0        # seconds a parent must wait after giving birth before gestating again

# ── Kin recognition / sociality ───────────────────────────────────────────────
KIN_THRESHOLD    = 0.80     # 1-mean(|gene_a-gene_b|) above this → treat as kin
FLOCK_IDEAL_DIST = 55.0     # preferred px separation within a flock

# ── Gestation / growth ────────────────────────────────────────────────────────
GESTATION_DURATION = 15.0   # seconds from conception to birth
BIRTH_GROWTH       = 0.25   # newborns start at this fraction of adult size/HP
GROWTH_RATE        = 0.018  # adult-size fraction per second (~56s to reach 1.0)

# ── Stats graph ───────────────────────────────────────────────────────────────
POP_HISTORY_LEN    = 400
STATS_TICK         = 0.5    # seconds between history samples
