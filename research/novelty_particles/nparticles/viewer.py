"""Live gallery: watch the archive of discoveries grow, in real time.

Layout
  * big canvas (left)  : the currently selected archive member, animated
  * 3x3 thumbnails      : the nine most recent discoveries, animated
  * HUD                 : generation, archive size, novelty, thresholds...

Rendering is a numpy float framebuffer with additive gaussian splats and
per-frame decay -- identical in spirit to the billboards+glow of the paper,
and it gives free motion trails.  Evolution is time-sliced into the main
loop (no threads): `evolver.work(budget)` runs between rendered frames.

Keys
  SPACE  pause/resume evolution        ENTER  follow latest discovery
  <- ->  browse the archive            1..9   zoom to a thumbnail
  B      boost-breed the shown effect  S      save its genome
  H      toggle HUD                    ESC    quit
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pygame

from .evolution import Evolver
from .genome import Genome
from .render import Canvas, draw_step
from .sim import SPACE, Sim

W, H = 1280, 860
ZOOM_W = ZOOM_H = 640
THUMB = 180
THUMB_GAP = 15
GRID_X = ZOOM_W + 60
GRID_Y = 130
HUD_X, HUD_Y = 24, 16


def blit_canvas(cv: Canvas, screen, pos):
    surf = pygame.Surface((cv.w, cv.h))
    pygame.surfarray.blit_array(surf, cv.frame().transpose(1, 0, 2))
    screen.blit(surf, pos)


def make_sim(genome: Genome, n: int, ttl: int, seed: int) -> Sim:
    return Sim(genome, n=n, ttl=ttl, seed=seed)





def run_viewer(evolver: Evolver, out_dir: str):
    pygame.display.init()
    pygame.font.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Novelty Particles — open-ended particle evolution")
    font = pygame.font.SysFont("menlo,monaco,consolas,couriernew,monospace", 14)
    small = pygame.font.SysFont("menlo,monaco,consolas,couriernew,monospace", 12)

    zoom_cv = Canvas(ZOOM_W, ZOOM_H, spr_r=24, fade=0.90, exposure=1.5)
    zoom_sim: Sim | None = None
    zoom_meta = {}

    thumb_cvs = [Canvas(THUMB, THUMB, spr_r=7, fade=0.87, exposure=1.7)
                 for _ in range(9)]
    thumb_sims: list[Sim | None] = [None] * 9
    thumb_meta = [None] * 9
    thumb_rects = [(GRID_X + (i % 3) * (THUMB + THUMB_GAP),
                    GRID_Y + (i // 3) * (THUMB + THUMB_GAP),
                    THUMB, THUMB) for i in range(9)]
    thumb_cursor = 0
    selected = -1                 # archive index browsed manually, -1 = follow
    paused = False
    follow = True
    hud = True
    frame = 0
    msg, msg_t = "", 0
    rng = random.Random(99)
    saves_dir = Path(out_dir) / "saves"
    saves_dir.mkdir(parents=True, exist_ok=True)

    def refresh_thumbs(entries):
        nonlocal thumb_cursor
        for e in entries[-9:]:
            i = thumb_cursor % 9
            thumb_cursor += 1
            thumb_sims[i] = make_sim(e["genome"], 90, 140, rng.randrange(1 << 30))
            thumb_meta[i] = e

    def set_zoom(genome, meta):
        nonlocal zoom_sim, zoom_meta
        zoom_sim = make_sim(genome, 300, 170, rng.randrange(1 << 30))
        zoom_meta = meta
        zoom_cv.clear()

    # initial content: latest archive entries (resume) or the seed population
    if evolver.archive.entries:
        refresh_thumbs(evolver.archive.entries)
        e = evolver.archive.entries[-1]
        set_zoom(e["genome"], e)
    else:
        set_zoom(evolver.population[0], {"uid": "seed", "gen": 0, "novelty": 0.0})

    clock = pygame.time.Clock()
    running = True
    while running:
        frame += 1
        # ---------------- events ----------------
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                k = ev.key
                if k == pygame.K_ESCAPE:
                    running = False
                elif k == pygame.K_SPACE:
                    paused = not paused
                elif k == pygame.K_RETURN:
                    follow = True
                    selected = -1
                    if evolver.archive.entries:
                        e = evolver.archive.entries[-1]
                        set_zoom(e["genome"], e)
                        msg, msg_t = "following latest", 90
                elif k in (pygame.K_LEFT, pygame.K_RIGHT):
                    n = len(evolver.archive)
                    if n:
                        d = -1 if k == pygame.K_LEFT else 1
                        selected = (selected if selected >= 0 else n - 1) + d
                        selected = max(0, min(n - 1, selected))
                        follow = False
                        e = evolver.archive.entries[selected]
                        set_zoom(e["genome"], e)
                elif pygame.K_1 <= k <= pygame.K_9:
                    i = k - pygame.K_1
                    if thumb_meta[i]:
                        follow = False
                        set_zoom(thumb_meta[i]["genome"], thumb_meta[i])
                elif k == pygame.K_b:
                    if zoom_sim is not None:
                        evolver.boost(zoom_sim.g, 8)
                        msg, msg_t = "boost-bred 8 mutants of shown effect", 120
                elif k == pygame.K_s:
                    if zoom_sim is not None:
                        p = saves_dir / f"{zoom_meta.get('uid', zoom_sim.g.uid)}.json"
                        p.write_text(json.dumps(zoom_sim.g.to_dict()))
                        msg, msg_t = f"saved {p.name}", 120
                elif k == pygame.K_h:
                    hud = not hud
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                for i, r in enumerate(thumb_rects):
                    if r[0] <= ev.pos[0] < r[0] + r[2] and r[1] <= ev.pos[1] < r[1] + r[3]:
                        if thumb_meta[i]:
                            follow = False
                            set_zoom(thumb_meta[i]["genome"], thumb_meta[i])

        # ---------------- evolution (time-sliced) ----------------
        if not paused:
            new = evolver.work(budget=0.012)
            if new:
                refresh_thumbs(new)
                if follow:
                    e = new[-1]
                    set_zoom(e["genome"], e)

        # ---------------- simulate + render ----------------
        screen.fill((8, 8, 12))
        if zoom_sim is not None:
            zoom_cv.tick()
            draw_step(zoom_cv, zoom_sim.step())
            blit_canvas(zoom_cv, screen, (24, 120))
            pygame.draw.rect(screen, (70, 90, 120), (24, 120, ZOOM_W, ZOOM_H), 1)

        if frame % 2 == 0:                      # thumbnails at half rate
            for i, cv in enumerate(thumb_cvs):
                s = thumb_sims[i]
                if s is None:
                    continue
                cv.tick()
                draw_step(cv, s.step())
                blit_canvas(cv, screen, thumb_rects[i][:2])
                pygame.draw.rect(screen, (60, 70, 95), thumb_rects[i], 1)

        if hud:
            e_lines = [
                "NOVELTY PARTICLES   open-ended evolution of particle effects",
                evolver.status_line() + (f"   [PAUSED]" if paused else ""),
                f"zoom: {zoom_meta.get('uid', '?')}  gen {zoom_meta.get('gen', '?')}  "
                f"novelty {zoom_meta.get('novelty', 0):.2f}  shape={zoom_sim.g.shape if zoom_sim else '?'}"
                f"  trail={'on' if zoom_sim and zoom_sim.trail else 'off'}"
                f"  net {zoom_sim.g.size()[0]}h/{zoom_sim.g.size()[1]}c" if zoom_sim else "",
                "SPACE pause  ENTER latest  <- -> browse  1-9 zoom  B boost-breed"
                "  S save  H hud  ESC quit",
            ]
            for i, t in enumerate(e_lines):
                screen.blit(font.render(t, True, (200, 210, 230)), (HUD_X, HUD_Y + 20 * i))
            for i, r in enumerate(thumb_rects):
                m = thumb_meta[i]
                if m:
                    t = small.render(
                        f"{i + 1} g{m['gen']} n{m['novelty']:.2f}", True, (150, 160, 190))
                    screen.blit(t, (r[0] + 4, r[1] + r[3] + 2))
        if msg_t > 0:
            screen.blit(font.render(msg, True, (240, 200, 120)), (HUD_X, 110))
            msg_t -= 1

        pygame.display.flip()
        clock.tick(60)
    pygame.display.quit()