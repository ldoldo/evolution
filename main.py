"""
Evolution Simulator — entry point.

Controls
--------
SPACE        Pause / resume  (in replay: restore snapshot and resume)
+ / =        Increase simulation speed (1× → 2× → 4× → 8× → 16×)
-            Decrease simulation speed
D            Toggle debug overlay (vision-range circles)
R            Reset simulation  (on end-screen: restart)
E            End game — show summary screen
Q / Escape   Quit

While paused
------------
Left-click   Select entity under cursor to inspect its genome & lineage
Left-click   Click empty space (or anywhere when inspector is open) to deselect
LEFT arrow   Enter replay at most-recent snapshot; then step back one snapshot
RIGHT arrow  Step forward one snapshot (exits replay at the live edge)
HOME         Jump to oldest snapshot
END          Jump to newest snapshot
SPACE        (in replay) Restore snapshot and resume from that point
"""

import copy
import math
import sys
import pygame

from simulation import Simulation
from renderer import Renderer
from config import WIDTH, HEIGHT, FPS, WORLD_WIDTH


SPEED_LEVELS   = [1, 2, 4, 8, 16]
CLICK_RADIUS   = 20    # pixel tolerance for entity selection
SNAPSHOT_EVERY = 1.0   # wall-clock seconds between snapshots
MAX_SNAPSHOTS  = 180   # 3 minutes of rewind history


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Evolution Simulator")
    clock = pygame.time.Clock()

    sim      = Simulation()
    renderer = Renderer(screen)
    paused   = False
    spd_idx  = 0          # index into SPEED_LEVELS
    selected = None       # currently inspected Entity | None

    # Replay / end-game state
    snapshots:  list  = []
    replay_idx: int | None = None
    next_snap   = SNAPSHOT_EVERY
    game_over   = False

    while True:
        raw_dt = clock.tick(FPS) / 1000.0
        dt     = min(raw_dt, 0.05)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                _quit()

            elif event.type == pygame.KEYDOWN:
                k = event.key

                if k in (pygame.K_q, pygame.K_ESCAPE):
                    _quit()

                elif k == pygame.K_r:
                    sim        = Simulation()
                    renderer   = Renderer(screen)
                    selected   = None
                    snapshots  = []
                    replay_idx = None
                    next_snap  = SNAPSHOT_EVERY
                    game_over  = False
                    paused     = False

                elif k == pygame.K_e and not game_over:
                    game_over = True

                elif k == pygame.K_SPACE:
                    if replay_idx is not None:
                        # Restore to snapshot and resume
                        sim        = copy.deepcopy(snapshots[replay_idx])
                        replay_idx = None
                        paused     = False
                        selected   = None
                    elif not game_over:
                        paused = not paused

                elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    spd_idx = min(spd_idx + 1, len(SPEED_LEVELS) - 1)

                elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    spd_idx = max(spd_idx - 1, 0)

                elif k == pygame.K_d:
                    renderer.toggle_debug()

                elif k == pygame.K_LEFT and paused and not game_over and snapshots:
                    if replay_idx is None:
                        replay_idx = len(snapshots) - 1   # enter at most-recent
                    else:
                        replay_idx = max(0, replay_idx - 1)

                elif k == pygame.K_RIGHT and replay_idx is not None:
                    replay_idx += 1
                    if replay_idx >= len(snapshots):      # stepped past end
                        replay_idx = None

                elif k == pygame.K_HOME and replay_idx is not None:
                    replay_idx = 0

                elif k == pygame.K_END and replay_idx is not None:
                    replay_idx = len(snapshots) - 1

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if game_over or replay_idx is not None:
                    pass   # no entity picking in these modes
                else:
                    mx, my = event.pos
                    if mx < WORLD_WIDTH:
                        if selected is not None:
                            selected = None
                        else:
                            selected = _pick_entity(sim, mx, my)
                            if selected is not None:
                                paused = True

        speedup = SPEED_LEVELS[spd_idx]

        # Snapshot (wall-clock timer, only while running)
        if not paused and not game_over:
            next_snap -= raw_dt
            if next_snap <= 0.0:
                next_snap = SNAPSHOT_EVERY
                snapshots.append(copy.deepcopy(sim))
                if len(snapshots) > MAX_SNAPSHOTS:
                    snapshots.pop(0)

        # Advance simulation
        if not paused and not game_over:
            sub_dt = dt / speedup
            for _ in range(speedup):
                sim.update(sub_dt)
            # Clear selection if the tracked entity died this tick
            if selected is not None and not selected.alive:
                selected = None

        # Render
        if game_over:
            renderer.render_end_screen(sim)
        elif replay_idx is not None:
            renderer.render(
                snapshots[replay_idx],
                speedup=speedup, paused=True, selected=None,
                replay_idx=replay_idx, total_snaps=len(snapshots),
            )
        else:
            renderer.render(
                sim,
                speedup=speedup, paused=paused, selected=selected,
                replay_idx=None, total_snaps=len(snapshots),
            )

        pygame.display.flip()

        # Window title
        if game_over:
            pygame.display.set_caption("Evolution Simulator  |  GAME OVER")
        elif replay_idx is not None:
            pct = int((replay_idx + 1) / len(snapshots) * 100)
            pygame.display.set_caption(
                f"Evolution Simulator  |  REPLAY  {replay_idx + 1}/{len(snapshots)}  ({pct}%)"
            )
        else:
            title = f"Evolution Simulator  |  {speedup}×"
            if paused:
                title += "  |  PAUSED"
            pygame.display.set_caption(title)


def _pick_entity(sim: Simulation, mx: float, my: float):
    """Return the Entity closest to (mx, my) within CLICK_RADIUS, or None."""
    best      = None
    best_dist = float("inf")
    for e in sim.entities:
        if not e.alive:
            continue
        d = math.hypot(e.x - mx, e.y - my)
        if d < max(e.genome.radius + 4, CLICK_RADIUS) and d < best_dist:
            best_dist = d
            best      = e
    return best


def _quit() -> None:
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
