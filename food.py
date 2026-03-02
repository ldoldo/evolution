from config import FOOD_ENERGY, FOOD_RADIUS, CORPSE_DECAY, CORPSE_BITE


class Food:
    """A static plant tile that respawns elsewhere after being eaten."""
    __slots__ = ("x", "y")

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    # Keep as properties so renderer can reference them uniformly
    @property
    def energy(self) -> float:
        return FOOD_ENERGY

    @property
    def radius(self) -> int:
        return FOOD_RADIUS


class Corpse:
    """
    Remains of a dead entity.  Any meat-eating entity can extract energy
    from it in bite-sized chunks until the energy pool is drained or the
    corpse decays.
    """
    __slots__ = ("x", "y", "energy", "radius", "age")

    def __init__(self, x: float, y: float, entity_size: float, energy: float) -> None:
        self.x      = x
        self.y      = y
        self.energy = energy
        self.radius = max(3, int(entity_size * 5.5))
        self.age    = 0.0

    @property
    def freshness(self) -> float:
        """1.0 when fresh, 0.0 when fully decayed."""
        return max(0.0, 1.0 - self.age / CORPSE_DECAY)

    def bite(self, amount: float = CORPSE_BITE) -> float:
        """Extract up to *amount* energy, scaled by freshness.  Returns actual energy extracted."""
        taken = min(self.energy, amount * self.freshness)
        self.energy -= taken
        return taken

    def update(self, dt: float) -> bool:
        """Advance time.  Returns True if the corpse should be removed."""
        self.age += dt
        return self.age >= CORPSE_DECAY or self.energy <= 0.0
