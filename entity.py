from __future__ import annotations

import math
import numpy as np

from genome import Genome
from food import Food, Corpse
from config import (
    BASE_METABOLISM, SIZE_COST, MOVE_COST, DIET_METABOLISM_EXTRA,
    ATTACK_RANGE_EXTRA, ATTACK_COOLDOWN, ATTACK_ENERGY_COST,
    SPAWN_SCATTER, MIN_PARENT_ENERGY_FRAC, BIRTH_COOLDOWN,
    KIN_THRESHOLD,
    GESTATION_DURATION, BIRTH_GROWTH, GROWTH_RATE,
    PACK_BONUS_PER_KIN, PACK_BONUS_RANGE, MAX_PACK_BONUS,
    PACK_DEFENSE_PER_KIN, HERD_RETALIATION_PER_KIN,
    HP_REGEN_RATE,
    PLANT_STOMACH_MAX, PLANT_STOMACH_DECAY,
    NN_INPUT,
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
        # Per-tick NN output — set in update(), consumed by simulation for attack gate
        self._nn_attack: float = 0.0
        # Per-tick kin cache — computed once in update(), shared by brain + combat
        self._kin_cache: dict[int, bool] = {}
        # Plant stomach: tracks recent plant-energy intake; carnivores fill up fast
        self._plant_stomach: float = 0.0

    # ── Derived runtime phenotype ──────────────────────────────────────────────

    @property
    def effective_radius(self) -> float:
        """Physical radius used for collisions and eating reach; shrinks while juvenile."""
        return max(3.0, self.genome.radius * self._growth)

    @property
    def effective_max_hp(self) -> float:
        """HP ceiling scales with body size during growth."""
        return self.genome.max_hp * self._growth

    # ── Metabolism ────────────────────────────────────────────────────────────

    def _metabolic_cost(self, dt: float) -> float:
        spd      = math.hypot(self.vx, self.vy)
        g        = self.genome
        eff_size = g.size * self._growth
        norm     = spd / max(g.max_speed, 1.0)
        cost     = (BASE_METABOLISM + g.diet * DIET_METABOLISM_EXTRA
                    + eff_size ** 2 * SIZE_COST + norm * eff_size * MOVE_COST)
        return cost * dt

    # ── Neural-network brain ──────────────────────────────────────────────────

    def _build_obs(
        self,
        food_items: list[Food],
        entities: list["Entity"],
        corpses: list[Corpse],
    ) -> np.ndarray:
        """
        Build the 18-element observation vector fed to the NN each tick.

        Layout (indices):
          0  hunger          — 1 − energy/threshold  (0=full, 1=starving)
          1  hp_frac         — hp / effective_max_hp
          2  diet            — genome diet gene  (0=herb, 1=carn)
          3,4  food_cos/sin  — unit direction to nearest visible plant
          5    food_dist     — 1 − dist/vision_range  (1=adjacent, 0=edge)
          6,7  corpse_cos/sin
          8    corpse_dist
          9,10 threat_cos/sin — nearest entity that is a predator to this one
          11   threat_dist
          12,13 prey_cos/sin  — nearest entity this one could hunt
          14    prey_dist
          15,16 kin_cos/sin   — nearest genetic kin
          17    kin_dist
        All direction components are 0 if no such object is visible.
        """
        obs = np.zeros(NN_INPUT)
        g   = self.genome
        sx, sy = self.x, self.y
        vis    = g.vision_range

        obs[0] = 1.0 - min(self.energy / g.reproduce_threshold, 1.0)
        obs[1] = self.hp / max(self.effective_max_hp, 1.0)
        obs[2] = g.diet

        # ── Nearest food ──────────────────────────────────────────────────────
        best = vis + 1.0; bdx = bdy = 0.0
        for f in food_items:
            dx = f.x - sx; dy = f.y - sy
            d  = math.hypot(dx, dy)
            if 0 < d < best:
                best = d; bdx = dx / d; bdy = dy / d
        if best <= vis:
            obs[3] = bdx; obs[4] = bdy
            obs[5] = 1.0 - best / vis

        # ── Nearest corpse ────────────────────────────────────────────────────
        best = vis + 1.0; bdx = bdy = 0.0
        for c in corpses:
            dx = c.x - sx; dy = c.y - sy
            d  = math.hypot(dx, dy)
            if 0 < d < best:
                best = d; bdx = dx / d; bdy = dy / d
        if best <= vis:
            obs[6] = bdx; obs[7] = bdy
            obs[8] = 1.0 - best / vis

        # ── Nearest threat / prey / kin from entity list ──────────────────────
        self_diet     = g.diet
        self_eff_size = g.size * self._growth
        kin           = self._kin_cache

        bt = bp = bk = vis + 1.0
        tdx = tdy = pdx = pdy = kdx = kdy = 0.0

        for e in entities:
            if e.id == self.id:
                continue
            dx = e.x - sx; dy = e.y - sy
            d  = math.hypot(dx, dy)
            if d <= 0:
                continue
            ndx = dx / d; ndy = dy / d

            other_diet     = e.genome.diet
            other_eff_size = e.genome.size * e._growth

            # Is e a threat to me?
            if (other_diet - self_diet > 0.15
                    and other_eff_size / max(self_eff_size, 0.01) >= 0.6
                    and d < bt):
                bt = d; tdx = ndx; tdy = ndy

            # Is e prey for me?
            if (self_diet - other_diet > 0.15
                    and self_eff_size / max(other_eff_size, 0.01) >= 0.6
                    and d < bp):
                bp = d; pdx = ndx; pdy = ndy

            # Is e kin?
            if kin.get(e.id, False) and d < bk:
                bk = d; kdx = ndx; kdy = ndy

        if bt <= vis:
            obs[9]  = tdx; obs[10] = tdy
            obs[11] = 1.0 - bt / vis
        if bp <= vis:
            obs[12] = pdx; obs[13] = pdy
            obs[14] = 1.0 - bp / vis
        if bk <= vis:
            obs[15] = kdx; obs[16] = kdy
            obs[17] = 1.0 - bk / vis

        return obs

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
        self._plant_stomach = max(0.0, self._plant_stomach - PLANT_STOMACH_DECAY * dt)

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

        # ── Passive HP regen ──────────────────────────────────────────────────
        max_hp = self.effective_max_hp
        if self.hp < max_hp:
            self.hp = min(self.hp + max_hp * HP_REGEN_RATE * dt, max_hp)

        # ── Kin cache ─────────────────────────────────────────────────────────
        # Always computed — needed by both _build_obs() and attack().
        self._kin_cache = {}
        if entities:
            sg  = self.genome.genes
            og  = np.array([e.genome.genes for e in entities])
            sim = 1.0 - np.abs(sg - og).mean(axis=1)
            for e, s in zip(entities, sim):
                self._kin_cache[e.id] = bool(s >= KIN_THRESHOLD)

        # ── Brain: build observation → NN forward pass ────────────────────────
        obs = self._build_obs(food_items, entities, corpses)
        out = self.genome.forward(obs)          # (move_dx, move_dy, attack_signal)
        self._nn_attack = float(out[2])

        ddx, ddy = float(out[0]), float(out[1])
        # NN magnitude is meaningful: near-zero → rest, large → full speed.
        # Clamp to unit circle so max_spd stays the true ceiling.
        mag = math.hypot(ddx, ddy)
        if mag > 1.0:
            ddx /= mag
            ddy /= mag
        # Wander noise only when the NN is near-idle (nothing worth moving toward),
        # so entities don't freeze permanently but can still choose to rest.
        elif mag < 0.15:
            ddx += np.random.randn() * 0.12
            ddy += np.random.randn() * 0.12

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
        Consume any plant within reach, subject to the plant stomach cap.

        The stomach tracks item count (not energy), so the cap in items is the
        same formula regardless of how much energy each plant provides:
          cap = (1−diet)² × PLANT_STOMACH_MAX
        This gives herbivores a large item budget; high-diet entities fill up
        after 1-2 plants and must wait ~20 s per plant to digest before eating more.
        Returns the list of Food objects that were eaten (to be removed).
        """
        eaten: list[Food] = []
        g   = self.genome
        eff = g.plant_efficiency
        if eff < 0.04:          # diet > ~0.8 → can't digest plants meaningfully
            return eaten
        cap   = (1.0 - g.diet) ** 2 * PLANT_STOMACH_MAX
        reach = self.effective_radius + 6.0
        for f in food_items:
            if self._plant_stomach >= cap:
                break           # stomach full — leave remaining plants for others
            if math.hypot(f.x - self.x, f.y - self.y) <= reach:
                self._plant_stomach += 1.0      # count items, not energy
                self.energy         += f.energy * eff
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
        skip_kin  = g.kin_protection > 0.3   # won't attack genetic kin
        is_social = g.sociality > 0.3        # coordinates pack hunts

        # Kin dict pre-computed in update() this tick — no numpy rebuild needed
        kin = self._kin_cache

        reach = self.effective_radius + ATTACK_RANGE_EXTRA
        dmg   = g.attack_damage
        dead: list[Entity] = []

        best_dist = float("inf")
        best: Entity | None = None
        for other in candidates:
            if not other.alive or other.id == self.id:
                continue
            # Kin protection: entities with high kin_protection won't attack kin
            if skip_kin and kin.get(other.id, False):
                continue
            dist = math.hypot(other.x - self.x, other.y - self.y)
            if dist < reach + other.effective_radius and dist < best_dist:
                best_dist = dist
                best = other

        if best is not None:
            # Pack bonus: social entities coordinate their attacks
            pack_mult = 1.0
            if nearby is not None and is_social:
                kin_support = sum(
                    1 for e in nearby
                    if (e.id != self.id and e.alive
                        and math.hypot(e.x - self.x, e.y - self.y) <= PACK_BONUS_RANGE
                        and kin.get(e.id, False))
                )
                pack_mult = 1.0 + PACK_BONUS_PER_KIN * min(kin_support, MAX_PACK_BONUS)

            # Herd defense + retaliation: kin near the target reduce damage and bite back
            herd_def   = 1.0
            herd_retal = 1.0
            if best._kin_cache:
                kin_near = sum(
                    1 for e in (nearby or [])
                    if e.id != best.id and e.alive
                    and math.hypot(e.x - best.x, e.y - best.y) <= PACK_BONUS_RANGE
                    and best._kin_cache.get(e.id, False)
                )
                capped = min(kin_near, MAX_PACK_BONUS)
                herd_def   = 1.0 - PACK_DEFENSE_PER_KIN      * capped
                herd_retal = 1.0 + HERD_RETALIATION_PER_KIN  * capped

            best.hp  -= dmg * pack_mult * self._growth * herd_def           # incoming hit
            self.hp  -= best.genome.attack_damage * best._growth * 0.25 * herd_retal  # retaliation
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
