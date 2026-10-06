"""Dibujo vectorial "a mano alzada".

Convierte una lista de primitivas sencillas (líneas, círculos, rectángulos,
flechas, textos...) en una imagen de rotulador negro sobre blanco. Se usa:
  * en el modo --offline (escenas de demostración incluidas), y
  * como alternativa gratuita cuando el modelo de imagen de Gemini no está
    disponible: el modelo de texto describe el dibujo con estas primitivas.

Coordenadas: el lienzo mide 1000 de ancho y 1000/aspecto de alto.
"""
from __future__ import annotations

import math
import random
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import FONTS_DIR

INK = (24, 24, 24)
SS = 2  # supersampling para antialiasing

NAMED_COLORS = {
    "amarillo": "#F6C744", "yellow": "#F6C744",
    "naranja": "#F28C38", "orange": "#F28C38",
    "rojo": "#E5533D", "red": "#E5533D",
    "azul": "#4A90D9", "blue": "#4A90D9",
    "celeste": "#8FC7F0", "lightblue": "#8FC7F0",
    "verde": "#5DB85C", "green": "#5DB85C",
    "morado": "#9B6FD0", "purple": "#9B6FD0",
    "rosa": "#F29BB8", "pink": "#F29BB8",
    "gris": "#B8B8B8", "gray": "#B8B8B8", "grey": "#B8B8B8",
    "marron": "#A9744F", "brown": "#A9744F",
}


def _color(c) -> tuple[int, int, int] | None:
    if not c or c in ("none", "null", "transparent"):
        return None
    if isinstance(c, (list, tuple)) and len(c) >= 3:
        return tuple(int(v) for v in c[:3])
    c = str(c).strip().lower()
    c = NAMED_COLORS.get(c, c)
    if c.startswith("#") and len(c) in (4, 7):
        if len(c) == 4:
            c = "#" + "".join(ch * 2 for ch in c[1:])
        try:
            return tuple(int(c[i:i + 2], 16) for i in (1, 3, 5))
        except ValueError:
            return None
    return None


def _font(size: int, hand: bool = True) -> ImageFont.FreeTypeFont:
    path = FONTS_DIR / ("PatrickHand-Regular.ttf" if hand else "Roboto.ttf")
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.load_default()


class Sketcher:
    def __init__(self, width: int = 1000, aspect: float = 3 / 4, seed: int = 7, px_width: int = 1200):
        self.w = width
        self.h = int(round(width / aspect))
        self.scale = px_width / width * SS
        self.img = Image.new("RGB", (int(self.w * self.scale), int(self.h * self.scale)), "white")
        self.fill_layer = Image.new("RGB", self.img.size, "white")
        self.fill_mask = Image.new("L", self.img.size, 0)
        self.draw = ImageDraw.Draw(self.img)
        self.fdraw = ImageDraw.Draw(self.fill_layer)
        self.mdraw = ImageDraw.Draw(self.fill_mask)
        self.rng = random.Random(seed)
        self.lw = 7  # grosor en unidades de lienzo

    # -- utilidades -------------------------------------------------------
    def _p(self, x, y):
        return (x * self.scale, y * self.scale)

    def _wobble(self, pts: list[tuple[float, float]], amp: float = 2.2, closed: bool = False):
        """Subdivide la polilínea y añade un temblor suave tipo trazo a mano."""
        if closed and pts[0] != pts[-1]:
            pts = pts + [pts[0]]
        out = []
        phase = self.rng.uniform(0, 6.28)
        for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
            seg = math.hypot(x2 - x1, y2 - y1)
            n = max(2, int(seg / 14))
            nx, ny = (-(y2 - y1) / (seg or 1), (x2 - x1) / (seg or 1))
            for i in range(n):
                t = i / n
                off = amp * math.sin(phase + t * 3.1 + seg * 0.01) * math.sin(math.pi * t + 0.3)
                out.append((x1 + (x2 - x1) * t + nx * off, y1 + (y2 - y1) * t + ny * off))
            phase += 1.7
        out.append(pts[-1])
        return out

    def _stroke(self, pts, width=None, color=INK):
        width = width or self.lw
        sp = [self._p(x, y) for x, y in pts]
        wpx = max(1, int(width * self.scale))
        self.draw.line(sp, fill=color, width=wpx, joint="curve")
        r = wpx / 2
        for x, y in (sp[0], sp[-1]):
            self.draw.ellipse([x - r, y - r, x + r, y + r], fill=color)

    def _fill(self, pts, color):
        sp = [self._p(x + self.rng.uniform(-3, 3), y + self.rng.uniform(-3, 3)) for x, y in pts]
        self.fdraw.polygon(sp, fill=color)
        self.mdraw.polygon(sp, fill=255)

    # -- primitivas -------------------------------------------------------
    def line(self, pts, width=None, color=INK, closed=False, fill=None):
        pts = [tuple(map(float, p)) for p in pts]
        if len(pts) < 2:
            return
        if fill is not None and len(pts) >= 3:
            self._fill(pts, fill)
        self._stroke(self._wobble(pts, closed=closed), width, color)

    def circle(self, cx, cy, r, width=None, fill=None, color=INK, start=0.0, end=360.0, ry=None):
        ry = ry if ry is not None else r
        n = max(16, int(abs(end - start) / 360 * (r + ry) / 2))
        jitter = self.rng.uniform(0.97, 1.03)
        pts = []
        for i in range(n + 1):
            a = math.radians(start + (end - start) * i / n)
            pts.append((cx + r * jitter * math.cos(a), cy + ry * math.sin(a)))
        full = abs(end - start) >= 359
        if fill is not None and full:
            self._fill(pts, fill)
        if full:  # pequeño solape típico de un círculo dibujado a mano
            pts = pts + pts[1:4]
        self._stroke(self._wobble(pts, amp=1.2), width, color)

    def rect(self, x, y, w, h, width=None, fill=None, color=INK):
        pts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        if fill is not None:
            self._fill(pts, fill)
        self.line(pts, width=width, closed=True, color=color)

    def arrow(self, x1, y1, x2, y2, width=None, color=INK, head=None):
        self.line([(x1, y1), (x2, y2)], width=width, color=color)
        ang = math.atan2(y2 - y1, x2 - x1)
        head = head or max(18, min(40, math.hypot(x2 - x1, y2 - y1) * 0.25))
        for s in (-1, 1):
            a = ang + math.pi + s * 0.45
            self.line([(x2 + head * math.cos(a), y2 + head * math.sin(a)), (x2, y2)], width=width, color=color)

    def text(self, x, y, s, size=48, color=INK, anchor="mm"):
        s = str(s)[:40]
        font = _font(int(size * self.scale))
        self.draw.text(self._p(x, y), s, font=font, fill=color, anchor=anchor,
                       stroke_width=max(1, int(size * self.scale * 0.025)), stroke_fill=color)

    # -- salida -----------------------------------------------------------
    def render(self) -> np.ndarray:
        """Devuelve la imagen final en BGR (numpy) con los rellenos debajo del trazo."""
        ink = np.asarray(self.img, dtype=np.float32)
        fill = np.asarray(self.fill_layer, dtype=np.float32)
        # multiplicar: los rellenos quedan "debajo" de la tinta
        out = ink * fill / 255.0
        img = Image.fromarray(out.clip(0, 255).astype(np.uint8))
        img = img.resize((img.width // SS, img.height // SS), Image.LANCZOS)
        arr = np.asarray(img)[:, :, ::-1].copy()
        return arr


def render_shapes(shapes: Iterable[dict], aspect: float = 3 / 4, seed: int = 7, px_width: int = 1200) -> np.ndarray:
    """Renderiza una lista de primitivas (formato JSON) a imagen BGR."""
    sk = Sketcher(aspect=aspect, seed=seed, px_width=px_width)
    for sh in shapes or []:
        try:
            _draw_shape(sk, sh)
        except Exception:
            continue  # una primitiva mal formada no debe romper el dibujo
    return sk.render()


def _num(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _draw_shape(sk: Sketcher, sh: dict) -> None:
    t = str(sh.get("type", sh.get("tipo", ""))).lower()
    fill = _color(sh.get("fill", sh.get("relleno")))
    width = _num(sh.get("width"), 0) or None
    if t in ("line", "linea", "polyline", "path"):
        sk.line(sh.get("points", []), width=width, closed=bool(sh.get("closed")), fill=fill)
    elif t in ("poly", "polygon", "forma"):
        sk.line(sh.get("points", []), width=width, closed=True, fill=fill)
    elif t in ("circle", "circulo", "ellipse"):
        c = sh.get("center", [500, 500])
        r = _num(sh.get("r", sh.get("radius", 50)), 50)
        ry = sh.get("ry")
        sk.circle(_num(c[0]), _num(c[1]), r, width=width, fill=fill, ry=_num(ry) if ry else None)
    elif t in ("arc", "arco"):
        c = sh.get("center", [500, 500])
        sk.circle(_num(c[0]), _num(c[1]), _num(sh.get("r", 50)), width=width,
                  start=_num(sh.get("start", 0)), end=_num(sh.get("end", 180)))
    elif t in ("rect", "rectangle", "rectangulo"):
        sk.rect(_num(sh.get("x")), _num(sh.get("y")), _num(sh.get("w", sh.get("width_box", 100))),
                _num(sh.get("h", 100)), width=width, fill=fill)
    elif t in ("arrow", "flecha"):
        a, b = sh.get("from", [0, 0]), sh.get("to", [100, 100])
        sk.arrow(_num(a[0]), _num(a[1]), _num(b[0]), _num(b[1]), width=width)
    elif t in ("text", "texto", "label"):
        p = sh.get("pos", sh.get("at", [500, 500]))
        sk.text(_num(p[0]), _num(p[1]), sh.get("text", sh.get("texto", "")), size=_num(sh.get("size", 48), 48))
