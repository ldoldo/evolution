import numpy as np

from entity import Entity
from food import Food, Corpse
from genome import Genome
from config import (
    WORLD_WIDTH, WORLD_HEIGHT,
    FOOD_SPAWN_RATE, MAX_FOOD,
    FOOD_CLUSTER_CHANCE, FOOD_CLUSTER_SIGMA,
    INITIAL_POPULATION, MIN_POPULATION,
    STATS_TICK, POP_HISTORY_LEN,
    MIN_AGGRESSION_TO_ATTACK,
    HERB_LOW_THRESHOLD, FOOD_SPAWN_HERB_BONUS,
)


class Simulation:
    """
    Owns all simulation state and drives one timestep at a time.

    Tick order
    ----------
    1. Spawn food
    2. For every alive entity:
       a. Gather visible neighbours (vectorised distance filter)
       b. entity.update()  — steering, movement, metabolism
       c. eat_food / eat_corpse
       d. attack
       e. reproduce
    3. Commit deaths → corpses
    4. Advance / prune corpses
    5. Respawn if population critically low
    """

    def __init__(self, width: int = WORLD_WIDTH, height: int = WORLD_HEIGHT) -> None:
        self.width    = width
        self.height   = height
        self.entities: list[Entity]  = []
        self.food:     list[Food]    = []
        self.corpses:  list[Corpse]  = []
        self.time     = 0.0
        self.max_gen  = 0

        self._food_acc = 0.0   # fractional food accumulator

        # Population history sampled every STATS_TICK seconds (capped for live graph)
        self.pop_history:  list[int] = []
        self.herb_history: list[int] = []
        self.carn_history: list[int] = []
        self._stats_acc = 0.0

        # Lifetime statistics
        self.total_births    = 0
        self.total_deaths    = 0
        self.peak_population = INITIAL_POPULATION
        self._peak_pop_time  = 0.0

        # Full-game history (no cap) — used by end-screen graphs
        self._all_pop:  list[int]   = []
        self._all_herb: list[int]   = []
        self._all_carn: list[int]   = []
        self._all_diet: list[float] = []

        # Event log — notable things that happened during the run
        self.events: list[tuple[float, str]] = []
        self._prev_herb = -1   # -1 = not yet sampled; skip first extinction check
        self._prev_carn = -1

        self._init_population()
        self._seed_food(MAX_FOOD // 2)
        self._log(f"Simulation started ({INITIAL_POPULATION} founders)")

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_population(self) -> None:
        for _ in range(INITIAL_POPULATION):
            x = np.random.uniform(0, self.width)
            y = np.random.uniform(0, self.height)
            self.entities.append(Entity(x, y))

    def _seed_food(self, n: int) -> None:
        for _ in range(n):
            self._place_food()

    def _place_food(self) -> None:
        if self.food and np.random.random() < FOOD_CLUSTER_CHANCE:
            # Spawn near a randomly chosen existing food item → creates patches
            anchor = self.food[np.random.randint(len(self.food))]
            x = (anchor.x + np.random.normal(0.0, FOOD_CLUSTER_SIGMA)) % self.width
            y = (anchor.y + np.random.normal(0.0, FOOD_CLUSTER_SIGMA)) % self.height
        else:
            x = np.random.uniform(0, self.width)
            y = np.random.uniform(0, self.height)
        self.food.append(Food(x, y))

    # ── Event log ─────────────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        self.events.append((self.time, msg))
        if len(self.events) > 60:
            self.events.pop(0)

    # ── Spatial query (numpy-accelerated) ────────────────────────────────────

    @staticmethod
    def _nearby(cx: float, cy: float, radius: float, items: list,
                positions: np.ndarray) -> list:
        """
        Return items whose (x, y) is within *radius* of (cx, cy).

        *positions* is a pre-built (N, 2) float32 array parallel to *items*,
        computed once per tick so the arithmetic is vectorised.
        """
        if len(items) == 0:
            return []
        dx  = positions[:, 0] - cx
        dy  = positions[:, 1] - cy
        mask = dx * dx + dy * dy <= radius * radius
        return [items[i] for i in np.where(mask)[0]]

    # ── Main tick ─────────────────────────────────────────────────────────────

    def update(self, dt: float) -> None:
        self.time += dt

        # 1. Spawn food (boosted when herbivores are scarce to help recovery)
        herb_count  = self.herbivore_count
        spawn_bonus = FOOD_SPAWN_HERB_BONUS * max(0.0, 1.0 - herb_count / HERB_LOW_THRESHOLD)
        self._food_acc += FOOD_SPAWN_RATE * (1.0 + spawn_bonus) * dt
        while self._food_acc >= 1.0 and len(self.food) < MAX_FOOD:
            self._place_food()
            self._food_acc -= 1.0

        # 2. Snapshot positions for bulk queries
        ent_snap   = list(self.entities)       # stable reference for this tick
        food_snap  = list(self.food)
        corps_snap = list(self.corpses)

        # Build parallel position arrays for vectorised distance queries
        # (corpses are excluded — small N makes Python loop faster than numpy)
        ent_pos   = (np.array([[e.x, e.y] for e in ent_snap],  dtype=np.float32)
                     if ent_snap  else np.empty((0, 2), np.float32))
        food_pos  = (np.array([[f.x, f.y] for f in food_snap], dtype=np.float32)
                     if food_snap else np.empty((0, 2), np.float32))

        new_entities: list[Entity] = []
        new_corpses:  list[Corpse] = []
        eaten_food:   set[int]     = set()
        eaten_corps:  set[int]     = set()

        for entity in ent_snap:
            if not entity.alive:
                continue

            ex  = entity.x
            ey  = entity.y
            vr  = entity.genome.vision_range
            vr2 = vr * vr

            nearby_ents  = self._nearby(ex, ey, vr, ent_snap,  ent_pos)
            nearby_food  = self._nearby(ex, ey, vr, food_snap, food_pos)
            # Corpse counts are tiny (≈1–5); Python loop avoids numpy call overhead
            nearby_corps = ([c for c in corps_snap
                             if (c.x-ex)*(c.x-ex) + (c.y-ey)*(c.y-ey) <= vr2]
                            if corps_snap else [])

            # a. Update (steering / movement / metabolism)
            entity.update(nearby_food, nearby_ents, nearby_corps,
                          self.width, self.height, dt)

            if not entity.alive:
                new_corpses.append(Corpse(entity.x, entity.y, entity.genome.size * entity._growth, entity.energy))
                continue

            # b. Eat plants (skip pure carnivores that can't digest plants)
            if nearby_food and entity.genome.plant_efficiency >= 0.001:
                available_food = [f for f in nearby_food if id(f) not in eaten_food]
                for f in entity.eat_food(available_food):
                    eaten_food.add(id(f))

            # c. Eat corpses (skip pure herbivores that can't digest meat)
            if nearby_corps and entity.genome.meat_efficiency >= 0.001:
                available_corps = [c for c in nearby_corps if id(c) not in eaten_corps]
                for c in entity.eat_corpse(available_corps):
                    eaten_corps.add(id(c))

            # d. Attack (skip non-aggressors and entities still on cooldown)
            if entity.genome.aggression >= MIN_AGGRESSION_TO_ATTACK and entity._atk_cd <= 0:
                attack_targets = [e for e in nearby_ents if e.id != entity.id and e.alive]
                for dead in entity.attack(attack_targets, nearby_ents):
                    new_corpses.append(Corpse(dead.x, dead.y, dead.genome.size * dead._growth, dead.energy))

            # e. Start gestation if energy is sufficient
            entity.reproduce()

            # f. Release offspring when gestation completes
            if entity._pending_offspring and entity._gestation_timer <= 0.0:
                children = entity._give_birth(self.width, self.height)
                self.total_births += len(children)
                for child in children:
                    child.birth_time = self.time
                    new_entities.append(child)
                    if child.generation > self.max_gen:
                        self.max_gen = child.generation
                        self._log(f"Gen {self.max_gen} reached")

        # 3. Commit deaths (entities killed as defenders die mid-tick)
        for entity in ent_snap:
            if not entity.alive and entity not in ent_snap:
                pass   # already handled above

        self.entities = [e for e in ent_snap if e.alive]

        # 4. Remove eaten food
        self.food = [f for f in food_snap if id(f) not in eaten_food]

        # 5. Advance corpses; remove decayed / fully consumed
        surviving_corps = []
        for c in corps_snap:
            if id(c) in eaten_corps:
                continue
            if not c.update(dt):
                surviving_corps.append(c)
        self.corpses = surviving_corps + new_corpses

        # 6. Add offspring
        self.entities.extend(new_entities)

        # Track deaths (every corpse created this tick is one death)
        self.total_deaths += len(new_corpses)

        # Track peak population
        pop = self.population
        if pop > self.peak_population:
            # Log when crossing a new multiple-of-10 milestone
            old_bucket = self.peak_population // 10
            new_bucket = pop // 10
            if new_bucket > old_bucket:
                self._log(f"Pop peak: {pop}")
            self.peak_population = pop
            self._peak_pop_time  = self.time
        elif pop < self.peak_population // 2 and self.peak_population >= 20:
            # Population has crashed to less than half the all-time peak
            pass   # logged in stats tick to avoid spamming

        # 7. Population floor
        if len(self.entities) < MIN_POPULATION:
            self._log(f"Pop critical ({pop}) — respawning")
            for _ in range(INITIAL_POPULATION // 4):
                x = np.random.uniform(0, self.width)
                y = np.random.uniform(0, self.height)
                self.entities.append(Entity(x, y))

        # 8. Stats history
        self._stats_acc += dt
        if self._stats_acc >= STATS_TICK:
            self._stats_acc = 0.0
            self.pop_history.append(self.population)
            self.herb_history.append(self.herbivore_count)
            self.carn_history.append(self.carnivore_count)
            if len(self.pop_history) > POP_HISTORY_LEN:
                self.pop_history.pop(0)
                self.herb_history.pop(0)
                self.carn_history.pop(0)
            # Full-game history (unbounded, used by end-screen)
            herb = self.herbivore_count
            carn = self.carnivore_count
            self._all_pop.append(self.population)
            self._all_herb.append(herb)
            self._all_carn.append(carn)
            self._all_diet.append(self.avg_diet)

            # Crash event: pop drops below 50% of all-time peak (throttled to stats tick)
            cur_pop = self.population
            if (self.peak_population >= 20
                    and cur_pop < self.peak_population // 2
                    and (not self.events or "crash" not in self.events[-1][1])):
                self._log(f"Pop crash: {cur_pop} (peak was {self.peak_population})")

            # Extinction / resurgence events (skip the very first sample)
            if self._prev_herb >= 0:
                if self._prev_herb > 0 and herb == 0:
                    self._log("Herbivores extinct!")
                elif self._prev_herb == 0 and herb > 0:
                    self._log(f"Herbivores returned ({herb})")
            if self._prev_carn >= 0:
                if self._prev_carn > 0 and carn == 0:
                    self._log("Carnivores extinct!")
                elif self._prev_carn == 0 and carn > 0:
                    self._log(f"Carnivores returned ({carn})")
            self._prev_herb = herb
            self._prev_carn = carn

    # ── Live statistics ────────────────────────────────────────────────────────

    @property
    def population(self) -> int:
        return len(self.entities)

    @property
    def herbivore_count(self) -> int:
        return sum(1 for e in self.entities if e.genome.diet < 0.35)

    @property
    def carnivore_count(self) -> int:
        return sum(1 for e in self.entities if e.genome.diet > 0.65)

    @property
    def omnivore_count(self) -> int:
        return self.population - self.herbivore_count - self.carnivore_count

    @property
    def avg_diet(self) -> float:
        if not self.entities:
            return 0.5
        return float(np.mean([e.genome.diet for e in self.entities]))

    @property
    def avg_size(self) -> float:
        if not self.entities:
            return 1.0
        return float(np.mean([e.genome.size for e in self.entities]))

    @property
    def avg_speed(self) -> float:
        if not self.entities:
            return 2.0
        return float(np.mean([e.genome.speed_raw for e in self.entities]))

    @property
    def avg_mutability(self) -> float:
        if not self.entities:
            return 0.05
        return float(np.mean([e.genome.mutability for e in self.entities]))
