from __future__ import annotations

import math
import pygame

from simulation import Simulation
from config import (
    BG_COLOR, WIDTH, HEIGHT, WORLD_WIDTH, PANEL_WIDTH,
    SIZE_MIN, SIZE_MAX, SPEED_MIN, SPEED_MAX,
    VIS_RANGE_MIN, VIS_RANGE_MAX, VIS_ANGLE_MIN, VIS_ANGLE_MAX,
    REPR_THRESH_MIN, REPR_THRESH_MAX, OFFSPRING_MIN, OFFSPRING_MAX,
    GESTATION_MIN, GESTATION_MAX, MUTABILITY_MIN, MUTABILITY_MAX,
)


# ── Colour helpers ─────────────────────────────────────────────────────────────

_PANEL_BG   = (18, 18, 30)
_PANEL_LINE = (45, 45, 65)
_TXT_MAIN   = (210, 210, 220)
_TXT_DIM    = (120, 120, 140)
_GREEN      = (60,  200, 60)
_RED        = (210, 60,  60)
_WHITE      = (220, 220, 220)
_YELLOW     = (220, 200, 60)
_ORANGE     = (220, 130, 40)


class Renderer:
    """Draws the world and the side-panel HUD each frame."""

    FONT_SIZE  = 14
    SMALL_SIZE = 11

    def __init__(self, screen: pygame.Surface) -> None:
        self.screen = screen
        pygame.font.init()
        self.font  = pygame.font.SysFont("monospace", self.FONT_SIZE)
        self.small = pygame.font.SysFont("monospace", self.SMALL_SIZE)
        self._debug = False   # toggle with D key

    # ── Public entry point ────────────────────────────────────────────────────

    def render(
        self,
        sim: Simulation,
        speedup: int = 1,
        paused: bool = False,
        selected=None,              # Entity | None
        replay_idx=None,            # int | None — current snapshot index
        total_snaps: int = 0,
    ) -> None:
        self.screen.fill(BG_COLOR)
        self._draw_world(sim, selected)
        if replay_idx is not None:
            self._draw_replay_panel(sim, replay_idx, total_snaps)
        elif paused and selected is not None:
            self._draw_entity_inspector(selected, sim.time)
        else:
            self._draw_panel(sim, speedup, paused, hint=paused)
        self._draw_graph(sim)

    def toggle_debug(self) -> None:
        self._debug = not self._debug

    # ── World layer ───────────────────────────────────────────────────────────

    def _draw_world(self, sim: Simulation, selected=None) -> None:
        # Food
        for f in sim.food:
            pygame.draw.circle(
                self.screen, (35, 170, 40),
                (int(f.x), int(f.y)), f.radius,
            )

        # Corpses — fade to dark as they age
        for c in sim.corpses:
            t     = c.freshness           # 1.0 → fresh, 0.0 → gone
            color = (int(110 * t), int(45 * t), int(18 * t))
            r     = max(2, int(c.radius * (0.5 + 0.5 * t)))
            pygame.draw.circle(self.screen, color, (int(c.x), int(c.y)), r)

        # Entities
        for e in sim.entities:
            if not e.alive:
                continue
            g  = e.genome
            px = int(e.x)
            py = int(e.y)
            r  = max(3, int(e.effective_radius))

            # Debug: vision circle
            if self._debug:
                vr  = int(g.vision_range)
                surf = pygame.Surface((vr * 2, vr * 2), pygame.SRCALPHA)
                pygame.draw.circle(surf, (*g.color, 25), (vr, vr), vr)
                self.screen.blit(surf, (px - vr, py - vr))

            # Body
            color = g.color
            pygame.draw.circle(self.screen, color, (px, py), r)
            pygame.draw.circle(self.screen, (0, 0, 0), (px, py), r, 1)

            # ── Diet indicator ────────────────────────────────────────────────
            # Heading: velocity direction when moving, otherwise point up
            spd = math.hypot(e.vx, e.vy)
            if spd > 5.0:
                nx_d, ny_d = e.vx / spd, e.vy / spd
            else:
                nx_d, ny_d = 0.0, -1.0

            diet     = g.diet
            perp_x, perp_y = -ny_d, nx_d

            if diet > 0.65:
                # Carnivore: large white tooth
                tooth_len = max(3, int(diet * r * 0.9))
                tooth_w   = max(2, int(diet * r * 0.4))
                tip    = (int(px + nx_d * (r + tooth_len)),
                          int(py + ny_d * (r + tooth_len)))
                base_l = (int(px + nx_d * r + perp_x * tooth_w),
                          int(py + ny_d * r + perp_y * tooth_w))
                base_r = (int(px + nx_d * r - perp_x * tooth_w),
                          int(py + ny_d * r - perp_y * tooth_w))
                pygame.draw.polygon(self.screen, _WHITE, [tip, base_l, base_r])
            elif diet >= 0.35:
                # Omnivore: small orange tooth — shorter and narrower than carnivore
                tooth_len = max(2, int(diet * r * 0.45))
                tooth_w   = max(1, int(diet * r * 0.20))
                tip    = (int(px + nx_d * (r + tooth_len)),
                          int(py + ny_d * (r + tooth_len)))
                base_l = (int(px + nx_d * r + perp_x * tooth_w),
                          int(py + ny_d * r + perp_y * tooth_w))
                base_r = (int(px + nx_d * r - perp_x * tooth_w),
                          int(py + ny_d * r - perp_y * tooth_w))
                pygame.draw.polygon(self.screen, _ORANGE, [tip, base_l, base_r])
            else:
                # Herbivore: two small leaf-green dots at the front
                dot_r  = max(1, r // 4)
                offset = r + dot_r + 2
                cx_l = int(px + nx_d * offset + perp_x * (dot_r + 1))
                cy_l = int(py + ny_d * offset + perp_y * (dot_r + 1))
                cx_r = int(px + nx_d * offset - perp_x * (dot_r + 1))
                cy_r = int(py + ny_d * offset - perp_y * (dot_r + 1))
                pygame.draw.circle(self.screen, _GREEN, (cx_l, cy_l), dot_r)
                pygame.draw.circle(self.screen, _GREEN, (cx_r, cy_r), dot_r)

            # Low-energy warning: thin yellow ring
            energy_frac = e.energy / g.reproduce_threshold
            if energy_frac < 0.25:
                pygame.draw.circle(self.screen, _YELLOW, (px, py), r + 2, 1)

            # Selected entity: bright white ring
            if selected is not None and e.id == selected.id:
                pygame.draw.circle(self.screen, _WHITE, (px, py), r + 4, 2)

        # World border
        pygame.draw.line(self.screen, _PANEL_LINE, (WORLD_WIDTH, 0), (WORLD_WIDTH, HEIGHT), 1)

    # ── Side panel ────────────────────────────────────────────────────────────

    def _draw_panel(self, sim: Simulation, speedup: int, paused: bool, hint: bool = False) -> None:
        px = WORLD_WIDTH + 1
        panel_rect = pygame.Rect(px, 0, PANEL_WIDTH, HEIGHT)
        pygame.draw.rect(self.screen, _PANEL_BG, panel_rect)

        def txt(text: str, x: int, y: int, color=_TXT_MAIN, small: bool = False) -> int:
            f    = self.small if small else self.font
            surf = f.render(text, True, color)
            self.screen.blit(surf, (x, y))
            return y + surf.get_height() + 2

        def sep(y: int) -> int:
            pygame.draw.line(self.screen, _PANEL_LINE, (px + 6, y), (px + PANEL_WIDTH - 6, y), 1)
            return y + 6

        x = px + 10
        y = 10

        # Title
        y = txt("EVOLUTION SIM", x, y, _WHITE)
        y = sep(y + 2)

        # Time / gen
        y = txt(f"Time    {sim.time:>9.1f}s", x, y)
        y = txt(f"Max gen {sim.max_gen:>9d}",  x, y)
        y = sep(y + 2)

        # Population
        y = txt("POPULATION", x, y, _TXT_DIM)
        y = txt(f"Total   {sim.population:>9d}",        x, y)
        y = txt(f"Herb    {sim.herbivore_count:>9d}",   x, y, _GREEN)
        y = txt(f"Omni    {sim.omnivore_count:>9d}",    x, y, _YELLOW)
        y = txt(f"Carn    {sim.carnivore_count:>9d}",   x, y, _RED)
        y = sep(y + 2)

        # Environment
        y = txt("ENVIRONMENT", x, y, _TXT_DIM)
        y = txt(f"Food    {len(sim.food):>9d}",         x, y)
        y = txt(f"Corpses {len(sim.corpses):>9d}",      x, y)
        y = sep(y + 2)

        # Average genome stats
        y = txt("AVG GENOME", x, y, _TXT_DIM)
        y = txt(f"Diet    {sim.avg_diet:>9.3f}", x, y)
        y = txt(f"Size    {sim.avg_size:>9.2f}", x, y)
        y = txt(f"Speed   {sim.avg_speed:>9.2f}", x, y)
        y = txt(f"Mutab.  {sim.avg_mutability:>9.4f}", x, y)
        y = sep(y + 2)

        # Diet bar
        y = self._diet_bar(x, y, sim.avg_diet)
        y += 8
        y = sep(y)

        # Controls
        y = txt("CONTROLS", x, y, _TXT_DIM)
        y = txt("[SPACE]  pause/resume",  x, y, _TXT_DIM, small=True)
        y = txt("[+] / [-]  speed",       x, y, _TXT_DIM, small=True)
        y = txt("[D]  debug vision",      x, y, _TXT_DIM, small=True)
        y = txt("[R]  reset",             x, y, _TXT_DIM, small=True)
        y = txt("[E]  end / summary",     x, y, _TXT_DIM, small=True)

        if hint:
            y = sep(y + 4)
            y = txt("click entity to", x, y, _YELLOW, small=True)
            y = txt("inspect genome",  x, y, _YELLOW, small=True)
            y = sep(y + 2)
            y = txt("[<] enter replay", x, y, _YELLOW, small=True)

        # Event log
        graph_top = HEIGHT - 145   # graph starts here; don't overlap
        if y + 30 < graph_top:
            y = sep(y + 2)
            y = txt("EVENTS", x, y, _TXT_DIM, small=True)
            log_entries = sim.events
            if not log_entries:
                txt("  —", x, y, _TXT_DIM, small=True)
            else:
                max_lines = max(0, (graph_top - y - 4) // 14)
                for t, msg in list(reversed(log_entries))[:max_lines]:
                    line = f"{t:6.0f}s  {msg}"
                    y = txt(line, x, y, _TXT_DIM, small=True)

        # Speed/pause status
        bar_y = HEIGHT - 24
        status = f"{'PAUSED  ' if paused else ''}{speedup}×"
        txt(status, x, bar_y, _YELLOW if paused else _TXT_MAIN)

    def _diet_bar(self, x: int, y: int, avg_diet: float) -> int:
        """Horizontal bar showing average diet from green→red."""
        bw, bh = PANEL_WIDTH - 20, 12
        # Background gradient (drawn as a series of thin rects)
        seg = max(1, bw // 64)
        for i in range(0, bw, seg):
            frac   = i / bw
            r_c    = int(40  + frac * 210)
            g_c    = int(200 - frac * 160)
            pygame.draw.rect(self.screen, (r_c, g_c, 40), (x + i, y, seg + 1, bh))
        pygame.draw.rect(self.screen, _WHITE, (x, y, bw, bh), 1)

        # Marker
        mx = int(x + avg_diet * bw)
        pygame.draw.line(self.screen, _WHITE, (mx, y - 3), (mx, y + bh + 3), 2)

        self.screen.blit(
            self.small.render("herb←diet→carn", True, _TXT_DIM),
            (x, y + bh + 3),
        )
        return y + bh + 16

    # ── Entity inspector (shown when paused + entity selected) ───────────────

    def _draw_entity_inspector(self, entity, sim_time: float) -> None:
        px  = WORLD_WIDTH + 1
        x   = px + 10
        bw  = PANEL_WIDTH - 22   # bar width

        pygame.draw.rect(self.screen, _PANEL_BG, (px, 0, PANEL_WIDTH, HEIGHT))

        def txt(text: str, y: int, color=_TXT_MAIN, small: bool = False) -> int:
            f    = self.small if small else self.font
            surf = f.render(text, True, color)
            self.screen.blit(surf, (x, y))
            return y + surf.get_height() + 2

        def sep(y: int) -> int:
            pygame.draw.line(self.screen, _PANEL_LINE,
                             (px + 6, y), (px + PANEL_WIDTH - 6, y), 1)
            return y + 6

        g   = entity.genome
        y   = 10

        # ── Header ────────────────────────────────────────────────────────────
        status_color = _TXT_MAIN if entity.alive else _RED
        y = txt(f"ENTITY #{entity.id}", y, _WHITE)
        if not entity.alive:
            y = txt("[ DEAD ]", y, _RED)
        y = sep(y + 2)

        # ── Lineage ───────────────────────────────────────────────────────────
        parent_str = f"#{entity.parent_id}" if entity.parent_id is not None else "none (founder)"
        y = txt(f"Generation  {entity.generation}", y)
        y = txt(f"Parent      {parent_str}", y, _TXT_DIM)
        y = txt(f"Born at     {entity.birth_time:.1f}s", y)
        y = txt(f"Age         {entity.age:.1f}s", y)
        y = sep(y + 2)

        # ── Energy ────────────────────────────────────────────────────────────
        thresh     = g.reproduce_threshold
        energy_frac = max(0.0, min(entity.energy / thresh, 1.0))
        y = txt(f"Energy  {entity.energy:.0f} / {thresh:.0f}", y)
        # Bar
        pygame.draw.rect(self.screen, (40, 40, 60), (x, y, bw, 9))
        bar_color = _GREEN if energy_frac > 0.4 else (_YELLOW if energy_frac > 0.2 else _RED)
        pygame.draw.rect(self.screen, bar_color, (x, y, int(energy_frac * bw), 9))
        pygame.draw.rect(self.screen, _PANEL_LINE, (x, y, bw, 9), 1)
        y += 13

        # ── HP ────────────────────────────────────────────────────────────────
        max_hp   = entity.effective_max_hp
        hp_frac  = max(0.0, min(entity.hp / max_hp, 1.0))
        y = txt(f"HP      {entity.hp:.0f} / {max_hp:.0f}", y)
        # Bar
        pygame.draw.rect(self.screen, (40, 40, 60), (x, y, bw, 9))
        hp_color = _GREEN if hp_frac > 0.5 else (_YELLOW if hp_frac > 0.25 else _RED)
        pygame.draw.rect(self.screen, hp_color, (x, y, int(hp_frac * bw), 9))
        pygame.draw.rect(self.screen, _PANEL_LINE, (x, y, bw, 9), 1)
        y += 13

        # ── Growth (juveniles only) ────────────────────────────────────────────
        if entity._growth < 1.0:
            y = txt(f"Growth  {entity._growth * 100:.0f}%", y, _YELLOW)
            pygame.draw.rect(self.screen, (40, 40, 60), (x, y, bw, 6))
            pygame.draw.rect(self.screen, _YELLOW, (x, y, int(entity._growth * bw), 6))
            pygame.draw.rect(self.screen, _PANEL_LINE, (x, y, bw, 6), 1)
            y += 10

        # ── Gestating ─────────────────────────────────────────────────────────
        if entity._gestation_timer > 0:
            y = txt(f"Gestating  {entity._gestation_timer:.1f}s", y, _YELLOW)

        y = sep(y + 2)

        # ── Diet label ────────────────────────────────────────────────────────
        d = g.diet
        if d < 0.35:
            diet_label, diet_color = "herbivore", _GREEN
        elif d > 0.65:
            diet_label, diet_color = "carnivore", _RED
        else:
            diet_label, diet_color = "omnivore",  _YELLOW
        y = txt(f"Diet class  {diet_label}", y, diet_color)
        y = sep(y + 2)

        # ── Genome bars ───────────────────────────────────────────────────────
        y = txt("GENOME", y, _TXT_DIM)

        genes = [
            ("Size",      g.genes[0],  f"{g.size:.2f}"),
            ("Speed",     g.genes[1],  f"{g.speed_raw:.2f}"),
            ("Vis.range", g.genes[2],  f"{g.vision_range:.0f}px"),
            ("Vis.angle", g.genes[3],  f"{math.degrees(g.vision_half_angle*2):.0f}°"),
            ("Diet",      g.genes[4],  f"{g.diet:.3f}"),
            ("Aggress.",  g.genes[5],  f"{g.aggression:.3f}"),
            ("Repr.thr.", g.genes[6],  f"{g.reproduce_threshold:.0f}"),
            ("Offspring", g.genes[7],  f"{g.offspring_count}"),
            ("Gestation", g.genes[8],  f"{g.gestation:.1f}"),
            ("Mutabil.",  g.genes[9],  f"{g.mutability:.4f}"),
            ("Hue",       g.genes[10], f"{g.hue:.3f}"),
            ("Sociality", g.genes[11], f"{g.sociality:.3f}"),
        ]

        for label, raw, value_str in genes:
            line  = f"{label:<10} {value_str:>7}"
            surf  = self.small.render(line, True, _TXT_MAIN)
            self.screen.blit(surf, (x, y))
            y    += surf.get_height() + 1
            # Gene bar
            pygame.draw.rect(self.screen, (35, 35, 55), (x, y, bw, 6))
            fill = int(raw * bw)
            # Colour bars by meaning
            if label == "Diet":
                bc = (int(40 + raw * 210), int(200 - raw * 160), 40)
            elif label == "Aggress.":
                bc = (int(80 + raw * 160), 80, 80)
            elif label == "Sociality":
                bc = (130, 70, 200)   # purple
            else:
                bc = (70, 130, 200)
            pygame.draw.rect(self.screen, bc, (x, y, fill, 6))
            pygame.draw.rect(self.screen, _PANEL_LINE, (x, y, bw, 6), 1)
            y += 9

        y = sep(y + 4)
        txt("click elsewhere to close", y, _TXT_DIM, small=True)

    # ── Population graph ─────────────────────────────────────────────────────

    def _draw_graph(self, sim: Simulation) -> None:
        if len(sim.pop_history) < 2:
            return

        gx = WORLD_WIDTH + 8
        gy = HEIGHT - 140
        gw = PANEL_WIDTH - 16
        gh = 100

        bg = pygame.Surface((gw, gh), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 160))
        self.screen.blit(bg, (gx, gy))
        pygame.draw.rect(self.screen, _PANEL_LINE, (gx, gy, gw, gh), 1)

        label = self.small.render("Population history", True, _TXT_DIM)
        self.screen.blit(label, (gx, gy - 14))

        max_pop = max(max(sim.pop_history, default=1), 1)
        n       = len(sim.pop_history)

        def draw_series(history: list[int], color: tuple) -> None:
            if len(history) < 2:
                return
            pts = [
                (
                    gx + int(i / (n - 1) * (gw - 1)),
                    gy + gh - 1 - int(v / max_pop * (gh - 2)),
                )
                for i, v in enumerate(history)
            ]
            pygame.draw.lines(self.screen, color, False, pts, 1)

        draw_series(sim.pop_history,  _WHITE)
        draw_series(sim.herb_history, _GREEN)
        draw_series(sim.carn_history, _RED)

    # ── Replay panel ──────────────────────────────────────────────────────────

    def _draw_replay_panel(self, sim: Simulation, replay_idx: int, total_snaps: int) -> None:
        px = WORLD_WIDTH + 1
        pygame.draw.rect(self.screen, _PANEL_BG, (px, 0, PANEL_WIDTH, HEIGHT))

        def txt(text: str, x: int, y: int, color=_TXT_MAIN, small: bool = False) -> int:
            f    = self.small if small else self.font
            surf = f.render(text, True, color)
            self.screen.blit(surf, (x, y))
            return y + surf.get_height() + 2

        def sep(y: int) -> int:
            pygame.draw.line(self.screen, _PANEL_LINE,
                             (px + 6, y), (px + PANEL_WIDTH - 6, y), 1)
            return y + 6

        x = px + 10
        y = 10

        y = txt("< REPLAY >", x, y, _YELLOW)
        y = sep(y + 2)

        y = txt(f"Snapshot  {replay_idx + 1:>4} / {total_snaps}", x, y)
        y = txt(f"Sim time  {sim.time:>8.1f}s",                   x, y)
        y = sep(y + 2)

        y = txt("POPULATION", x, y, _TXT_DIM)
        y = txt(f"Total   {sim.population:>9d}",      x, y)
        y = txt(f"Herb    {sim.herbivore_count:>9d}",  x, y, _GREEN)
        y = txt(f"Omni    {sim.omnivore_count:>9d}",   x, y, _YELLOW)
        y = txt(f"Carn    {sim.carnivore_count:>9d}",  x, y, _RED)
        y = sep(y + 2)

        y = txt("CONTROLS", x, y, _TXT_DIM)
        y = txt("[<]       step back",        x, y, _TXT_DIM, small=True)
        y = txt("[>]       step forward",     x, y, _TXT_DIM, small=True)
        y = txt("[HOME]    oldest snapshot",  x, y, _TXT_DIM, small=True)
        y = txt("[END]     newest snapshot",  x, y, _TXT_DIM, small=True)
        y = txt("[SPACE]   resume from here", x, y, _YELLOW,  small=True)

        # Progress bar
        bar_y = HEIGHT - 36
        bw    = PANEL_WIDTH - 20
        pygame.draw.rect(self.screen, (35, 35, 55), (x, bar_y, bw, 8))
        fill = int((replay_idx + 1) / max(total_snaps, 1) * bw)
        pygame.draw.rect(self.screen, _YELLOW, (x, bar_y, fill, 8))
        pygame.draw.rect(self.screen, _PANEL_LINE, (x, bar_y, bw, 8), 1)

    # ── End-game summary screen ───────────────────────────────────────────────

    def render_end_screen(self, sim: Simulation) -> None:
        """Full-screen stats overlay shown after the user presses E."""
        self.screen.fill((8, 8, 16))

        W, H = WIDTH, HEIGHT
        mid_x = W // 2

        # ── Title ─────────────────────────────────────────────────────────────
        big = pygame.font.SysFont("monospace", 28, bold=True)
        title_surf = big.render("SIMULATION SUMMARY", True, _WHITE)
        self.screen.blit(title_surf, (W // 2 - title_surf.get_width() // 2, 20))

        # ── Separator ─────────────────────────────────────────────────────────
        pygame.draw.line(self.screen, _PANEL_LINE, (20, 60), (W - 20, 60), 1)

        # ── Left column: text stats ────────────────────────────────────────────
        def row(label: str, value: str, y: int, color=_TXT_MAIN) -> int:
            ls = self.font.render(f"{label:<22}", True, _TXT_DIM)
            vs = self.font.render(value,          True, color)
            self.screen.blit(ls, (40, y))
            self.screen.blit(vs, (40 + ls.get_width(), y))
            return y + ls.get_height() + 6

        y = 80
        y = row("Duration",       f"{sim.time:.1f} s",              y)
        y = row("Max generation", str(sim.max_gen),                  y)
        y = row("Peak population",
                f"{sim.peak_population}  (t={sim._peak_pop_time:.0f}s)", y, _WHITE)
        y = row("Total births",   str(sim.total_births),             y, _GREEN)
        y = row("Total deaths",   str(sim.total_deaths),             y, _RED)
        y += 6
        y = row("Final population", str(sim.population),             y, _WHITE)
        y = row("  Herbivores",   str(sim.herbivore_count),          y, _GREEN)
        y = row("  Omnivores",    str(sim.omnivore_count),           y, _YELLOW)
        y = row("  Carnivores",   str(sim.carnivore_count),          y, _RED)
        y += 6
        if sim.population > 0:
            avg_diet = sim.avg_diet
            y = row("Avg diet",   f"{avg_diet:.3f}", y)
            y = self._diet_bar(40, y, avg_diet) + 8

        # ── Right column: graphs ───────────────────────────────────────────────
        graph_x = mid_x + 20
        graph_w = W - graph_x - 20

        # Population over time
        pop_data = sim._all_pop
        if len(pop_data) >= 2:
            graph_h = (H - 80) // 2 - 30
            graph_y = 70
            self._end_graph(
                graph_x, graph_y, graph_w, graph_h,
                "Population over time",
                [
                    (sim._all_pop,  _WHITE,  "total"),
                    (sim._all_herb, _GREEN,  "herb"),
                    (sim._all_carn, _RED,    "carn"),
                ],
            )

            # Diet over time
            diet_data = sim._all_diet
            if len(diet_data) >= 2:
                diet_y = graph_y + graph_h + 40
                self._end_graph(
                    graph_x, diet_y, graph_w, graph_h,
                    "Avg diet over time  (0=herb, 1=carn)",
                    [(diet_data, _YELLOW, "diet")],
                    y_min=0.0, y_max=1.0,
                )

        # ── Footer ────────────────────────────────────────────────────────────
        pygame.draw.line(self.screen, _PANEL_LINE, (20, H - 40), (W - 20, H - 40), 1)
        footer = self.font.render(
            "Press  R  to restart    ·    Q / Esc  to quit", True, _TXT_DIM
        )
        self.screen.blit(footer, (W // 2 - footer.get_width() // 2, H - 28))

    def _end_graph(
        self,
        gx: int, gy: int, gw: int, gh: int,
        label: str,
        series: list,               # list of (data, color, legend_label)
        y_min: float | None = None,
        y_max: float | None = None,
    ) -> None:
        """Draw a line graph inside the given rect with an optional label."""
        bg = pygame.Surface((gw, gh), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 140))
        self.screen.blit(bg, (gx, gy))
        pygame.draw.rect(self.screen, _PANEL_LINE, (gx, gy, gw, gh), 1)

        lbl_surf = self.small.render(label, True, _TXT_DIM)
        self.screen.blit(lbl_surf, (gx, gy - 16))

        # Determine y range
        all_vals = [v for data, _, __ in series for v in data]
        lo = y_min if y_min is not None else min(all_vals, default=0)
        hi = y_max if y_max is not None else max(all_vals, default=1)
        if hi == lo:
            hi = lo + 1

        n = max(len(data) for data, _, __ in series)

        for data, color, legend in series:
            if len(data) < 2:
                continue
            pts = [
                (
                    gx + int(i / (len(data) - 1) * (gw - 1)),
                    gy + gh - 1 - int((v - lo) / (hi - lo) * (gh - 2)),
                )
                for i, v in enumerate(data)
            ]
            pygame.draw.lines(self.screen, color, False, pts, 1)
