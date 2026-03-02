# Evolution Simulator

A continuous-space 2-D evolution sandbox.  Entities live, eat, fight, reproduce,
and die.  Their behaviour and appearance are entirely determined by an evolvable
genome — no hard-coded species.

---

## Quick start

```bash
pip install -r requirements.txt
python main.py
```

Python 3.10+ required.

---

## Controls

| Key | Action |
|-----|--------|
| `SPACE` | Pause / resume |
| `+` / `=` | Speed up (1× → 2× → 4× → 8× → 16×) |
| `-` | Slow down |
| `D` | Toggle debug overlay (vision-range circles) |
| `R` | Reset simulation |
| `Q` / `Esc` | Quit |

---

## How it works

### The world

The simulation runs on a **continuous toroidal 2-D plane** (entities wrap at
the edges).  Each tick:

1. Food (plants) spawns at a fixed rate up to a world cap.
2. Every entity senses its surroundings, decides where to move, and acts.
3. Eating, fighting, and corpse-decay are resolved.
4. Energy is drained by metabolism; zero-energy entities die and become corpses.
5. High-energy entities reproduce asexually; offspring inherit a mutated genome.

---

### The genome

Every entity carries a fixed-length float array (genes ∈ [0, 1]).
Each gene maps to a biologically meaningful range:

| Gene | Range | Effect |
|------|-------|--------|
| `size` | 0.2 – 3.0 | Larger = harder to kill + hits harder, but needs more food |
| `speed` | 0.5 – 5.0 | Raw speed score; actual px/s scaled down by √size |
| `vision_range` | 20 – 200 px | How far the entity can perceive |
| `vision_angle` | 30° – 360° | Total field of view |
| `diet` | 0.0 – 1.0 | **0 = pure herbivore, 1 = pure carnivore** |
| `aggression` | 0.0 – 1.0 | Willingness to attack vs. flee |
| `reproduce_threshold` | 100 – 300 | Energy level that triggers reproduction |
| `offspring_count` | 1 – 4 | Number of children per birth event |
| `gestation` | 30 – 90 | Energy invested per offspring (parent pays, child starts with this) |
| `mutability` | 0.005 – 0.12 | Std-dev of Gaussian mutation noise per gene |
| `hue` | 0 – 1 | Cosmetic; drifts neutrally, tints the body colour |

---

### Diet as a spectrum

`diet` is a **continuous value**, not a binary flag.

* **Digestion efficiency**
  * Eating plants:  efficiency = `1 − diet`
  * Eating meat/corpses: efficiency = `diet`
  * An omnivore (diet ≈ 0.5) gets 50 % efficiency from both sources —
    flexible but less effective than a specialist.

* **Attack capability**
  * Damage scales with `diet`: pure carnivores deal full damage, pure
    herbivores deal only ~10 %.

* **Steering preference**
  * Herbivores are attracted to food clusters; carnivores are attracted to
    other entities.  These weights come directly from the genome, so
    preference emerges naturally.

---

### Behaviour model

Entities use **weighted steering forces** — no neural network, just a handful
of attraction/repulsion vectors combined each tick:

| Force | Who feels it |
|-------|-------------|
| Seek food (plants) | Weighted by `(1−diet) × hunger` |
| Seek corpses | Weighted by `diet × hunger` |
| Chase prey | Weighted by `diet × aggression × hunger` |
| Flee predators | Weighted by `1 − aggression` |
| Wander noise | Always-on, prevents lock-in |

---

### Energy & metabolism

```
cost/s = BASE + size² × SIZE_COEFF + normalised_speed × size × MOVE_COEFF
```

Large, fast-moving entities are expensive.  The only sustainable path for
high-metabolism phenotypes is to be an effective carnivore (meat is
energy-dense) or to eat constantly as a small herbivore.

Corpse energy = `size × 38`; killing a large entity is therefore a huge
caloric reward for carnivores.

---

### Reproduction

Asexual (single-parent).  When `energy ≥ reproduce_threshold`:

1. Parent splits off 1–4 offspring (from `offspring_count` gene).
2. Each offspring costs `gestation` energy from the parent.
3. Each offspring receives exactly that `gestation` energy to start.
4. The parent keeps a minimum fraction of its threshold to survive.
5. Each offspring's genome = parent's genes + Gaussian noise (σ = `mutability`).

`mutability` itself is a gene — under stable conditions low-mutability wins;
under crash conditions high-mutability can find niches faster.

---

### What to watch for

* **Predator–prey oscillations** — carnivore booms follow herbivore booms;
  herbivore recoveries follow carnivore busts.
* **Arms races** — herbivore speed vs. carnivore size.
* **Specialists vs. generalists** — omnivores survive crashes that wipe out
  specialists, but lose in stable environments.
* **Evolutionary explosions** — a new phenotype occasionally sweeps through
  the population when a niche opens up.

---

## File structure

```
evolution/
├── config.py       — all tunable constants
├── genome.py       — Genome class + gene accessors
├── food.py         — Food and Corpse objects
├── entity.py       — Entity: steering, eating, combat, reproduction
├── simulation.py   — master loop; owns all entities, food, corpses
├── renderer.py     — pygame drawing (world + side-panel HUD + graph)
└── main.py         — pygame event loop and entry point
```

---

## Tuning tips

All important constants live in `config.py`.  Key levers:

* `FOOD_SPAWN_RATE` / `MAX_FOOD` — how food-rich the world is.
  Richer food → herbivore dominance.
* `CORPSE_ENERGY_MULT` — how rewarding a kill is.
  Higher values favour carnivores.
* `INITIAL_POPULATION` — starting headcount.
* `MUTABILITY_MIN/MAX` — evolutionary pace.

---

## Possible extensions

- Neural-net brains (weights as genome genes) for richer emergent intelligence
- Sexual reproduction (genomes merged from two parents)
- Environmental gradients (food clusters, hazard zones)
- Seasonal food variation
- Parasitism / symbiosis mechanics
