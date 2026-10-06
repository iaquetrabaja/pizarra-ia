"""Animación de dibujo a mano (sólo CPU, sin segmentación ni GPU).

Idea (inspirada en storyboard-ai/draw_animation.py, reescrita):
  1. La máscara de tinta se divide en celdas pequeñas (p. ej. 8x8 px).
  2. Las celdas ocupadas se agrupan en componentes conexos (= "trazos" o
     grupos de trazos; las letras de una palabra suelen quedar juntas).
  3. Los componentes se ordenan: primero los grandes (el motivo principal),
     luego los pequeños (detalles, etiquetas), siempre buscando el más cercano
     a la posición actual del rotulador para que la mano no salte de un lado a otro.
  4. Dentro de cada componente se recorre un camino en profundidad que prefiere
     seguir recto: así la mano "sigue la línea" como si la trazara.
  5. Todo se traduce a un único array de índices de píxel ordenados; cada
     fotograma revela un tramo con una sola operación numpy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .config import ASSETS_DIR
from .style import SceneLayers

NB = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]


@dataclass
class StrokePlan:
    order_px: np.ndarray   # índices planos de píxel (y*W+x) en orden de dibujo
    cell_end: np.ndarray   # nº acumulado de píxeles revelados tras cada celda
    cell_xy: np.ndarray    # (n, 2) posición del rotulador (x, y) por celda

    @property
    def n_cells(self) -> int:
        return len(self.cell_end)


def _trace_component(lab: list, comp: int, start: int, gw: int, gh: int, revealed: bytearray,
                     cells: np.ndarray) -> tuple[list[int], list[int]]:
    """Recorre un componente con un "pincel" de 3x3 celdas.

    La punta avanza a la celda vecina que destapa más tinta nueva, prefiriendo seguir
    recta; así una línea de 2-3 celdas de grosor se dibuja en UNA pasada. Cuando no
    queda nada nuevo alrededor retrocede (pila) o salta a la celda pendiente más cercana.
    Devuelve (celdas en orden de revelado, celda de la punta que reveló cada una)."""
    order: list[int] = []
    pen_of: list[int] = []
    expanded = set()

    def in_comp(j):
        return lab[j] == comp

    def brush(c):
        cy, cx = divmod(c, gw)
        for oy in (-1, 0, 1):
            ny = cy + oy
            if 0 <= ny < gh:
                for ox in (-1, 0, 1):
                    nx = cx + ox
                    if 0 <= nx < gw:
                        j = ny * gw + nx
                        if not revealed[j] and lab[j] == comp:
                            revealed[j] = 1
                            order.append(j)
                            pen_of.append(c)

    def gain(c):
        cy, cx = divmod(c, gw)
        g = 0
        for oy in (-1, 0, 1):
            ny = cy + oy
            if 0 <= ny < gh:
                for ox in (-1, 0, 1):
                    nx = cx + ox
                    if 0 <= nx < gw:
                        j = ny * gw + nx
                        if not revealed[j] and lab[j] == comp:
                            g += 1
        return g

    cur = start
    dy, dx = 0, 1
    stack = []
    while True:
        expanded.add(cur)
        brush(cur)
        stack.append(cur)
        nxt = -1
        while stack and nxt < 0:
            c = stack[-1]
            cy, cx = divmod(c, gw)
            bs = 0.0
            for oy, ox in NB:
                ny, nx = cy + oy, cx + ox
                if 0 <= ny < gh and 0 <= nx < gw:
                    j = ny * gw + nx
                    if in_comp(j) and j not in expanded:
                        g = gain(j)
                        if g:
                            s = g + 2.0 * (oy * dy + ox * dx) / (1.414 if oy and ox else 1.0)
                            if nxt < 0 or s > bs:
                                bs, nxt, bo = s, j, (oy, ox)
            if nxt < 0:
                stack.pop()
        if nxt >= 0:
            dy, dx = bo
            cur = nxt
            continue
        # salto: celda pendiente más cercana del componente (raro)
        pend = cells[np.frombuffer(revealed, np.uint8)[cells] == 0]
        if len(pend) == 0:
            break
        py, px = divmod(cur, gw)
        cy, cx = np.divmod(pend, gw)
        cur = int(pend[np.argmin((cy - py) ** 2 + (cx - px) ** 2)])
    return order, pen_of


def _bbox_dist(px: float, py: float, b) -> float:
    x0, y0, x1, y1 = b
    dx = max(x0 - px, 0, px - x1)
    dy = max(y0 - py, 0, py - y1)
    return math.hypot(dx, dy)


def plan_strokes(mask: np.ndarray, cell: int = 8) -> StrokePlan:
    """Calcula el orden de dibujo de una máscara booleana de tinta."""
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return StrokePlan(np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, 2), np.int32))
    gh, gw = -(-H // cell), -(-W // cell)
    cid = (ys // cell) * gw + (xs // cell)
    occ = np.zeros(gh * gw, np.uint8)
    occ[cid] = 1
    n, labels, stats, _ = cv2.connectedComponentsWithStats(occ.reshape(gh, gw), connectivity=8)
    lab_flat = labels.ravel()
    lab = lab_flat.tolist()

    comps = []
    for c in range(1, n):
        x, y, w, h, area = stats[c]
        comps.append({"id": c, "area": int(area), "bbox": (x, y, x + w - 1, y + h - 1)})
    max_area = max(c["area"] for c in comps)
    big = [c for c in comps if c["area"] >= max(0.2 * max_area, 6)]
    small = [c for c in comps if c not in big]

    vis = bytearray(gh * gw)
    order: list[int] = []
    pen_of: list[int] = []
    # posición inicial: el componente más grande, empezando por su celda superior-izquierda
    pen = None
    cells_of = {}

    def comp_cells(cid_):
        if cid_ not in cells_of:
            cells_of[cid_] = np.flatnonzero(lab_flat == cid_)
        return cells_of[cid_]

    for tier in (big, small):
        remaining = list(tier)
        if pen is None:
            remaining.sort(key=lambda c: -c["area"])
            first = remaining.pop(0)
            cells = comp_cells(first["id"])
            cy, cx = np.divmod(cells, gw)
            start = int(cells[np.argmin(cy * 2 + cx)])
            seq, pens = _trace_component(lab, first["id"], start, gw, gh, vis, cells)
            order += seq
            pen_of += pens
            pen = divmod(pens[-1], gw)
        while remaining:
            py, px = pen
            # el más cercano, con un pequeño sesgo a favor de los grandes
            k = min(range(len(remaining)),
                    key=lambda i: _bbox_dist(px, py, remaining[i]["bbox"]) - 0.5 * math.sqrt(remaining[i]["area"]))
            comp = remaining.pop(k)
            cells = comp_cells(comp["id"])
            cy, cx = np.divmod(cells, gw)
            start = int(cells[np.argmin((cy - py) ** 2 + (cx - px) ** 2)])
            seq, pens = _trace_component(lab, comp["id"], start, gw, gh, vis, cells)
            order += seq
            pen_of += pens
            pen = divmod(pens[-1], gw)

    order_arr = np.asarray(order, np.int64)
    rank = np.full(gh * gw, -1, np.int64)
    rank[order_arr] = np.arange(len(order_arr))
    prank = rank[cid]
    srt = np.argsort(prank, kind="stable")
    order_px = (ys.astype(np.int64) * W + xs)[srt]
    counts = np.bincount(prank, minlength=len(order_arr))
    cell_end = np.cumsum(counts)
    oy, ox = np.divmod(np.asarray(pen_of, np.int64), gw)
    cell_xy = np.stack([np.minimum(ox * cell + cell // 2, W - 1), np.minimum(oy * cell + cell // 2, H - 1)], 1)
    return StrokePlan(order_px, cell_end, cell_xy.astype(np.int32))


# --------------------------------------------------------------------------
# Mano
# --------------------------------------------------------------------------
class Hand:
    """Sprite de la mano con rotulador. La punta del rotulador es el ancla."""

    def __init__(self, height: int):
        img = cv2.imread(str(ASSETS_DIR / "drawing-hand.png"), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(ASSETS_DIR / "hand-mask.png"), cv2.IMREAD_GRAYSCALE)
        ys, xs = np.nonzero(mask > 127)
        img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        mask = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        s = height / img.shape[0]
        size = (max(1, int(img.shape[1] * s)), height)
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_AREA)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        a = mask.astype(np.float32)[..., None] / 255.0
        self.pre = img.astype(np.float32) * a
        self.inv = 1.0 - a
        self.h, self.w = mask.shape
        # punta: el píxel opaco más arriba-izquierda
        ys, xs = np.nonzero(mask > 128)
        k = np.argmin(xs + ys)
        self.tip = (int(xs[k]), int(ys[k]))

    def blit(self, frame: np.ndarray, x: int, y: int):
        """Dibuja la mano con la punta en (x, y). Devuelve copia de seguridad para restaurar."""
        H, W = frame.shape[:2]
        x0, y0 = x - self.tip[0], y - self.tip[1]
        fx0, fy0 = max(0, x0), max(0, y0)
        fx1, fy1 = min(W, x0 + self.w), min(H, y0 + self.h)
        if fx1 <= fx0 or fy1 <= fy0:
            return None
        sx0, sy0 = fx0 - x0, fy0 - y0
        sx1, sy1 = sx0 + (fx1 - fx0), sy0 + (fy1 - fy0)
        roi = frame[fy0:fy1, fx0:fx1]
        backup = (fy0, fy1, fx0, fx1, roi.copy())
        out = roi * self.inv[sy0:sy1, sx0:sx1] + self.pre[sy0:sy1, sx0:sx1]
        roi[:] = out.astype(np.uint8)
        return backup


def restore(frame: np.ndarray, backup) -> None:
    if backup is not None:
        y0, y1, x0, x1, data = backup
        frame[y0:y1, x0:x1] = data


# --------------------------------------------------------------------------
# Animación de una escena
# --------------------------------------------------------------------------
class SceneAnimator:
    """Genera los fotogramas de una escena con la duración pedida.

    Fases: dibujo (la mano sigue la punta) -> fundido a color -> mano sale -> espera.
    """

    def __init__(self, layers: SceneLayers, plan: StrokePlan, hand: Hand, n_frames: int, fps: int):
        self.L = layers
        self.plan = plan
        self.hand = hand
        self.n = max(1, n_frames)
        self.fps = fps
        fade = int(round(0.6 * fps)) if layers.has_color else 0
        hold = max(int(0.7 * fps), int(0.12 * self.n))
        self.draw_frames = max(1, self.n - fade - hold) if plan.n_cells else 0
        self.fade_frames = min(fade, max(0, self.n - self.draw_frames))
        self.exit_frames = int(0.45 * fps)

    def frames(self):
        """Itera (frame, hand_pos | None). El frame puede reutilizarse entre iteraciones:
        el llamador no debe guardarlo, sólo escribirlo."""
        L, P = self.L, self.plan
        H, W = L.base.shape[:2]
        canvas = L.base.copy()
        flat = canvas.reshape(-1, 3)
        src = L.lines.reshape(-1, 3)
        done_px = 0
        last = (W // 2, H // 2)
        exit_to = (int(W * 1.05), int(H * 1.1))
        for f in range(self.n):
            if f < self.draw_frames:
                k = min(P.n_cells, int(math.ceil(P.n_cells * (f + 1) / self.draw_frames)))
                end = int(P.cell_end[k - 1])
                if end > done_px:
                    idx = P.order_px[done_px:end]
                    flat[idx] = src[idx]
                    done_px = end
                x, y = P.cell_xy[k - 1]
                jx = int(3 * math.sin(f * 0.9))
                jy = int(3 * math.cos(f * 1.3))
                last = (int(x), int(y))
                yield canvas, (last[0] + jx, last[1] + jy)
                continue
            if f == self.draw_frames or (self.draw_frames == 0 and f == 0):
                if done_px < len(P.order_px):
                    idx = P.order_px[done_px:]
                    flat[idx] = src[idx]
                    done_px = len(P.order_px)
                if not L.has_color:
                    canvas = L.lines.copy() if self.draw_frames else L.final.copy()
            g = f - self.draw_frames
            frame = canvas
            if L.has_color and g < self.fade_frames:
                a = (g + 1) / max(1, self.fade_frames)
                a = a * a * (3 - 2 * a)
                frame = cv2.addWeighted(canvas, 1 - a, L.final, a, 0)
            elif L.has_color:
                frame = L.final
            pos = None
            if self.draw_frames and g < self.exit_frames:
                t = (g + 1) / self.exit_frames
                t = t * t
                pos = (int(last[0] + (exit_to[0] - last[0]) * t), int(last[1] + (exit_to[1] - last[1]) * t))
            yield frame, pos
