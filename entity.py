from __future__ import annotations

import math
import random
import numpy as np

_TWO_PI = math.pi * 2

from genome import Genome
from food import Food, Corpse
from config import (
    BASE_METABOLISM, SIZE_COST, MOVE_COST,
    ATTACK_RANGE_EXTRA, ATTACK_COOLDOWN,
    MIN_AGGRESSION_TO_ATTACK, ATTACK_ENERGY_COST,
    SPAWN_SCATTER, MIN_PARENT_ENERGY_FRAC, BIRTH_COOLDOWN,
    KIN_THRESHOLD, FLOCK_IDEAL_DIST,
    GESTATION_DURATION, BIRTH_GROWTH, GROWTH_RATE,
    PACK_BONUS_PER_KIN, PACK_BONUS_RANGE, MAX_PACK_BONUS,
)


class Entity:
    """
    A single organism in the simulation.

    Position, velocity, energy, HP, and all behavioural traits are driven by
    the genome.  The entity itself only knows about objects passed to it
    each tick; spatial queries live in Simulation.

    Lifecycle
    ---------
    Energy  : fuel for metabolism and reproduction.  Hits 0 → starvation death.
    HP      : structural integrity.  Reduced by combat.  Hits 0 → death.
    Growth  : newborns start at BIRTH_GROWTH fraction of adult size/HP and
              grow at GROWTH_RATE per second.  Can't reproduce until fully grown.
    Gestation: when energy is sufficient, reproduction is committed immediately
              (energy deducted) but offspring emerge after GESTATION_DURATION s.
    """

    _next_id: int = 0

    def __init__(
        self,
        x: float,
        y: float,
        genome: Genome | None = None,
        energy: float | None = None,
        generation: int = 0,
    ) -> None:
        Entity._next_id += 1
        self.id         = Entity._next_id
        self.x          = x
        self.y          = y
        self.genome     = genome if genome is not None else Genome()
        self.energy     = energy if energy is not None else self.genome.reproduce_threshold * 0.55
        self.hp         = self.genome.max_hp   # set properly for adult founders
        self.vx         = 0.0
        self.vy         = 0.0
        self.alive      = True
        self.age        = 0.0
        self.generation = generation
        self.parent_id: int | None = None
        self.birth_time: float     = 0.0
        self._atk_cd    = 0.0   # attack cooldown timer
        self._birth_cd  = 0.0   # post-birth cooldown timer (starts after giving birth)
        self._growth:           float = 1.0   # 1.0 = fully grown adult
        self._gestation_timer:  float = 0.0
        self._pending_offspring: list[tuple[Genome, float]] = []
        # Per-tick FOV cache — set once in update() before _compute_desired
        self._cached_spd:      float = 0.0
        self._cached_heading:  float = 0.0
        self._cached_vis_half: float = math.pi   # default: omnivision
        # Per-tick kin cache — computed once in update(), shared by steering + combat
        self._kin_cache: dict[int, bool] = {}

    # ── Derived runtime phenotype ──────────────────────────────────────────────

    @property
    def effective_radius(self) -> float:
        """Physical radius used for collisions and eating reach; shrinks while juvenile."""
        return max(3.0, self.genome.radius * self._growth)

    @property
    def effective_max_hp(self) -> float:
        """HP ceiling scales with body size during growth."""
        return self.genome.max_hp * self._growth

    # ── Kin recognition ───────────────────────────────────────────────────────

    def _kin_similarity(self, other: "Entity") -> float:
        """
        Gene-space similarity in [0, 1].
        Formula: 1 − mean(|g_self − g_other|)
        Random pairs average ≈ 0.67.  Parent/offspring ≈ 0.96.
        KIN_THRESHOLD = 0.80 cleanly separates family from strangers.
        """
        return float(1.0 - np.mean(np.abs(self.genome.genes - other.genome.genes)))

    # ── Metabolism ────────────────────────────────────────────────────────────

    def _metabolic_cost(self, dt: float) -> float:
        spd   = math.hypot(self.vx, self.vy)
        g     = self.genome
        norm  = spd / max(g.max_speed, 1.0)
        cost  = BASE_METABOLISM + g.size ** 2 * SIZE_COST + norm * g.size * MOVE_COST
        return cost * dt

    # ── Field of view ─────────────────────────────────────────────────────────

    def _in_fov(self, tx: float, ty: float) -> tuple[bool, float]:
        """Return (visible, distance).  Uses cached heading + half-angle.

        Callers are expected to pre-filter by vision_range via _nearby(), so
        that check is omitted here.  Cache fields (_cached_spd, _cached_heading,
        _cached_vis_half) must be refreshed each tick before calling this.
        """
        dx   = tx - self.x
        dy   = ty - self.y
        dist = math.hypot(dx, dy)
        half = self._cached_vis_half
        if half >= math.pi:               # 360-degree vision — common fast path
            return True, dist
        if self._cached_spd < 1.0:        # stationary → omnidirectional
            return True, dist
        to_tgt = math.atan2(dy, dx)
        diff   = abs(math.atan2(math.sin(to_tgt - self._cached_heading),
                                math.cos(to_tgt - self._cached_heading)))
        return diff <= half, dist

    # ── Steering ──────────────────────────────────────────────────────────────

    def _compute_desired(
        self,
        food_items: list[Food],
        entities: list["Entity"],
        corpses: list[Corpse],
    ) -> tuple[float, float]:
        """
        Weighted-force steering.  Returns an un-normalised desired direction.

        Forces:
          • Seek plants      — weighted by plant_efficiency × hunger
          • Seek corpses     — weighted by meat_efficiency × hunger
          • Flock with kin   — cohesion when far, separation when close (if social)
          • Chase prey       — weighted by meat_efficiency × aggression × hunger (skips kin if social)
          • Flee predators   — weighted by (1-aggression)
          • Wander noise     — small constant
        """
        g        = self.genome
        diet     = g.diet
        aggress  = g.aggression
        hunger   = 1.0 - min(self.energy / g.reproduce_threshold, 1.0)
        plant_w  = g.plant_efficiency    # (1-diet)² — zero for pure carnivores
        meat_w   = g.meat_efficiency     # diet²     — zero for pure herbivores

        dx = dy = 0.0

        sx = self.x
        sy = self.y

        # ── Seek plants ───────────────────────────────────────────────────────
        # Food items are already filtered to vision_range by _nearby; no FOV
        # angle check needed (smell/sense is omnidirectional).
        herb_w = plant_w * hunger * 2.8
        if herb_w > 0.02:
            for f in food_items:
                fdx = f.x - sx
                fdy = f.y - sy
                dist = math.hypot(fdx, fdy)
                if dist > 0:
                    w  = herb_w / (dist + 1.0)
                    dx += fdx / dist * w
                    dy += fdy / dist * w

        # ── Seek corpses ──────────────────────────────────────────────────────
        corp_w = meat_w * hunger * 2.2
        if corp_w > 0.02:
            for c in corpses:
                cdx = c.x - sx
                cdy = c.y - sy
                dist = math.hypot(cdx, cdy)
                if dist > 0:
                    w  = corp_w / (dist + 1.0)
                    dx += cdx / dist * w
                    dy += cdy / dist * w

        # ── Interactions with other entities ──────────────────────────────────
        soc       = g.sociality
        flock_w   = soc * 2.0
        # Avoidance strength: peaks at 1.0 for soc=0, zero at soc=0.5
        avoid_str = max(0.0, (0.5 - soc) * 2.0)

        # Kin dict pre-computed once in update() and cached on self._kin_cache
        kin = self._kin_cache

        self_size = g.size   # accessed twice per iteration below
        vis_half  = self._cached_vis_half          # cache outside loop
        limited_fov = vis_half < math.pi and self._cached_spd >= 1.0

        for other in entities:
            if other.id == self.id:
                continue
            # Inline _in_fov: avoids function-call overhead and reuses _dx/_dy below
            _dx  = other.x - sx
            _dy  = other.y - sy
            dist = math.hypot(_dx, _dy)
            if dist <= 0:
                continue
            if limited_fov:
                to_tgt = math.atan2(_dy, _dx)
                diff   = abs(math.atan2(math.sin(to_tgt - self._cached_heading),
                                        math.cos(to_tgt - self._cached_heading)))
                if diff > vis_half:
                    continue

            ox = _dx / dist
            oy = _dy / dist

            is_kin = kin.get(other.id, False)

            # Am I a meaningful predator to this entity?  (needed before avoid)
            other_diet = other.genome.diet
            other_size = other.genome.size
            diet_advantage = diet - other_diet
            size_advantage = self_size / max(other_size, 0.01)
            is_hunter = diet_advantage > 0.15 and size_advantage >= 0.6

            # Is it a meaningful threat to me?
            their_diet_adv = other_diet - diet
            their_size_adv = other_size / max(self_size, 0.01)
            is_threat = their_diet_adv > 0.15 and their_size_adv >= 0.6

            # ── Sociality-driven spatial forces ───────────────────────────────
            if soc > 0.5 and is_kin:
                # Flock: cohesion when far, gentle separation when close
                if dist > FLOCK_IDEAL_DIST:
                    w = flock_w / (dist + 1.0)
                    dx += ox * w
                    dy += oy * w
                else:
                    w = flock_w * 1.5 / (dist + 0.5)
                    dx -= ox * w
                    dy -= oy * w
            elif avoid_str > 0.0 and not is_kin and not is_hunter:
                # Loners repel non-kin — but don't avoid prey they're hunting
                w = avoid_str * 1.6 / (dist + 0.5)
                dx -= ox * w
                dy -= oy * w

            if is_hunter and aggress > 0.25:
                # Social entities don't chase kin
                if not (is_kin and soc > 0.3):
                    chase_w = meat_w * aggress * hunger * 2.0 / (dist + 1.0)
                    dx += ox * chase_w
                    dy += oy * chase_w
            elif is_threat:
                flee_w = (1.0 - aggress) * 3.5 / (dist + 1.0)
                dx -= ox * flee_w
                dy -= oy * flee_w

        # ── Wander noise ──────────────────────────────────────────────────────
        angle = random.random() * _TWO_PI
        dx += math.cos(angle) * 0.14
        dy += math.sin(angle) * 0.14

        return dx, dy

    # ── Per-tick update ───────────────────────────────────────────────────────

    def update(
        self,
        food_items: list[Food],
        entities: list["Entity"],
        corpses: list[Corpse],
        world_w: float,
        world_h: float,
        dt: float,
    ) -> None:
        if not self.alive:
            return

        self.age      += dt
        self._atk_cd   = max(0.0, self._atk_cd  - dt)
        self._birth_cd = max(0.0, self._birth_cd - dt)

        # ── Growth ────────────────────────────────────────────────────────────
        if self._growth < 1.0:
            prev_max     = self.genome.max_hp * self._growth
            self._growth = min(1.0, self._growth + GROWTH_RATE * dt)
            new_max      = self.genome.max_hp * self._growth
            # HP grows proportionally with the body
            self.hp = min(self.hp + (new_max - prev_max), new_max)

        # ── Gestation countdown ───────────────────────────────────────────────
        if self._gestation_timer > 0:
            self._gestation_timer = max(0.0, self._gestation_timer - dt)

        # ── FOV geometry cache ────────────────────────────────────────────────
        # Refreshed here so _in_fov() never recomputes these per-call.
        self._cached_spd     = math.hypot(self.vx, self.vy)
        self._cached_heading = (math.atan2(self.vy, self.vx)
                                if self._cached_spd >= 1.0 else 0.0)
        self._cached_vis_half = self.genome.vision_half_angle

        # ── Kin cache ─────────────────────────────────────────────────────────
        # Computed once per tick; read by _compute_desired() and attack() so
        # the numpy array construction and mean() run only once per entity.
        self._kin_cache = {}
        if entities and self.genome.sociality > 0.05:
            sg  = self.genome.genes
            og  = np.array([e.genome.genes for e in entities])   # (n, 12)
            sim = 1.0 - np.abs(sg - og).mean(axis=1)             # (n,)
            for e, s in zip(entities, sim):
                self._kin_cache[e.id] = bool(s >= KIN_THRESHOLD)

        # ── Steering ─────────────────────────────────────────────────────────
        ddx, ddy = self._compute_desired(food_items, entities, corpses)
        mag = math.hypot(ddx, ddy)
        if mag > 0:
            ddx /= mag
            ddy /= mag

        max_spd = self.genome.max_speed
        accel   = 9.0
        self.vx += (ddx * max_spd - self.vx) * accel * dt
        self.vy += (ddy * max_spd - self.vy) * accel * dt

        spd = math.hypot(self.vx, self.vy)
        if spd > max_spd:
            self.vx = self.vx / spd * max_spd
            self.vy = self.vy / spd * max_spd

        # ── Move & wrap ───────────────────────────────────────────────────────
        self.x = (self.x + self.vx * dt) % world_w
        self.y = (self.y + self.vy * dt) % world_h

        # ── Metabolism ────────────────────────────────────────────────────────
        self.energy -= self._metabolic_cost(dt)
        if self.energy <= 0 or self.hp <= 0:
            self.alive = False

    # ── Eating ────────────────────────────────────────────────────────────────

    def eat_food(self, food_items: list[Food]) -> list[Food]:
        """
        Consume any plant within reach.
        Returns the list of Food objects that were eaten (to be removed).
        """
        eaten: list[Food] = []
        reach = self.effective_radius + 6.0
        eff   = self.genome.plant_efficiency
        if eff < 0.001:
            return eaten
        for f in food_items:
            if math.hypot(f.x - self.x, f.y - self.y) <= reach:
                self.energy += f.energy * eff
                eaten.append(f)
        return eaten

    def eat_corpse(self, corpses: list[Corpse]) -> list[Corpse]:
        """
        Take a need-sized bite from every corpse within reach.

        The entity only extracts as much raw energy as it needs to reach its
        reproduce_threshold — leaving the remainder on the corpse for others.
        Returns corpses that were fully consumed.
        """
        finished: list[Corpse] = []
        eff = self.genome.meat_efficiency
        if eff < 0.001:
            return finished

        # How much gained energy the entity still wants
        still_hungry = max(0.0, self.genome.reproduce_threshold - self.energy)
        if still_hungry < 0.1:
            return finished   # already full — leave the corpse alone

        for c in corpses:
            reach = self.effective_radius + c.radius + 4.0
            if math.hypot(c.x - self.x, c.y - self.y) <= reach:
                # Raw corpse energy needed to satisfy hunger (accounting for eff)
                needed_raw = still_hungry / eff
                gained = c.bite(needed_raw)      # capped internally at min(c.energy, needed_raw)
                self.energy    += gained * eff
                still_hungry   -= gained * eff
                if c.energy <= 0:
                    finished.append(c)
                if still_hungry < 0.1:
                    break   # satisfied, don't strip other corpses this tick
        return finished

    # ── Combat ────────────────────────────────────────────────────────────────

    def attack(
        self,
        candidates: list["Entity"],
        nearby: list["Entity"] | None = None,
    ) -> list["Entity"]:
        """
        Attack the nearest reachable entity, gated on aggression and kin.

        Social entities (sociality > 0.3) skip targets that are genetic kin
        (similarity > KIN_THRESHOLD).  This emergently protects offspring —
        no hard-coded parent/child immunity needed.

        Pack bonus: kin within PACK_BONUS_RANGE who are alive add +PACK_BONUS_PER_KIN
        to the damage multiplier (capped at MAX_PACK_BONUS supporters).

        Damage hits HP (not energy); attacker pays an energy stamina cost.
        Returns list of entities killed this tick.
        """
        if self._atk_cd > 0:
            return []

        g = self.genome
        if g.aggression < MIN_AGGRESSION_TO_ATTACK:
            return []

        need_kin = g.sociality > 0.3

        # Kin dict pre-computed in update() this tick — no numpy rebuild needed
        kin = self._kin_cache if need_kin else {}

        reach = self.effective_radius + ATTACK_RANGE_EXTRA
        dmg   = g.attack_damage
        dead: list[Entity] = []

        best_dist = float("inf")
        best: Entity | None = None
        for other in candidates:
            if not other.alive or other.id == self.id:
                continue
            # Kin protection: social entities won't attack their own kind
            if need_kin and kin.get(other.id, False):
                continue
            dist = math.hypot(other.x - self.x, other.y - self.y)
            if dist < reach + other.effective_radius and dist < best_dist:
                best_dist = dist
                best = other

        if best is not None:
            # Pack bonus: count kin supporters within PACK_BONUS_RANGE
            pack_mult = 1.0
            if nearby is not None and need_kin:
                kin_support = sum(
                    1 for e in nearby
                    if (e.id != self.id and e.alive
                        and math.hypot(e.x - self.x, e.y - self.y) <= PACK_BONUS_RANGE
                        and kin.get(e.id, False))
                )
                pack_mult = 1.0 + PACK_BONUS_PER_KIN * min(kin_support, MAX_PACK_BONUS)

            best.hp      -= dmg * pack_mult                    # combat damages health
            self.hp      -= best.genome.attack_damage * 0.25   # defender retaliates (health)
            self.energy  -= ATTACK_ENERGY_COST                 # stamina cost (energy)
            self._atk_cd  = ATTACK_COOLDOWN
            if best.hp <= 0:
                best.alive = False
                dead.append(best)
            if self.hp <= 0:
                self.alive = False

        return dead

    # ── Reproduction ──────────────────────────────────────────────────────────

    def reproduce(self) -> None:
        """
        Commit to reproducing when energy is sufficient.

        Energy is deducted immediately; offspring emerge after GESTATION_DURATION
        seconds via _give_birth() called by the simulation.
        Only fully-grown adults can reproduce.
        """
        g = self.genome
        if (self._birth_cd > 0
                or self._gestation_timer > 0
                or self._pending_offspring
                or self._growth < 1.0
                or self.energy < g.reproduce_threshold):
            return

        min_keep = g.reproduce_threshold * MIN_PARENT_ENERGY_FRAC
        pending: list[tuple[Genome, float]] = []

        for _ in range(g.offspring_count):
            cost = g.gestation
            if self.energy - cost < min_keep:
                break
            self.energy -= cost
            pending.append((g.mutate(), cost))

        if pending:
            self._pending_offspring = pending
            self._gestation_timer   = GESTATION_DURATION

    def _give_birth(self, world_w: float, world_h: float) -> list["Entity"]:
        """
        Release gestating offspring into the world.
        Called by Simulation when _gestation_timer reaches 0.
        """
        offspring: list[Entity] = []
        for genome, start_energy in self._pending_offspring:
            ox = (self.x + np.random.uniform(-SPAWN_SCATTER, SPAWN_SCATTER)) % world_w
            oy = (self.y + np.random.uniform(-SPAWN_SCATTER, SPAWN_SCATTER)) % world_h
            child = Entity(ox, oy, genome=genome, energy=start_energy,
                           generation=self.generation + 1)
            child._growth = BIRTH_GROWTH
            child.hp      = child.effective_max_hp   # start at juvenile HP
            child.parent_id = self.id
            offspring.append(child)

        self._pending_offspring = []
        self._birth_cd          = BIRTH_COOLDOWN   # cooldown begins after giving birth
        return offspring
