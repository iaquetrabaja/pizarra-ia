"""Estilos de pizarra: fondo, color de la tinta y capa de color.

Todas las imágenes de origen son "rotulador negro sobre blanco" (con algún
toque de color). A partir de ellas se obtienen tres capas a tamaño de vídeo:

  base   -> la pizarra vacía
  lines  -> la pizarra con todos los trazos dibujados (sin rellenos de color)
  final  -> la pizarra con trazos y rellenos de color
  ink    -> máscara booleana de los píxeles de tinta (lo que dibuja la mano)
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import Layout

# Colores en BGR
STYLE_DEFS = {
    "pizarra": {"bg": (248, 249, 250), "ink": (32, 30, 28)},
    "tiza": {"bg": (48, 62, 34), "ink": (236, 240, 240)},
    "cuaderno": {"bg": (236, 248, 252), "ink": (110, 48, 30)},
}


@dataclass
class SceneLayers:
    base: np.ndarray
    lines: np.ndarray
    final: np.ndarray
    ink: np.ndarray          # bool HxW
    has_color: bool


def make_board(layout: Layout, estilo: str, seed: int = 3) -> np.ndarray:
    """Genera la pizarra vacía para un estilo."""
    W, H = layout.width, layout.height
    d = STYLE_DEFS[estilo]
    board = np.empty((H, W, 3), np.uint8)
    board[:] = d["bg"]
    rng = np.random.default_rng(seed)
    if estilo == "tiza":
        # textura de tiza borrada: ruido suave de baja frecuencia + grano fino
        small = rng.normal(0, 1, (H // 24 + 1, W // 24 + 1)).astype(np.float32)
        low = cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC) * 4.0
        grain = rng.normal(0, 2.2, (H, W)).astype(np.float32)
        tex = (low + grain)[..., None]
        board = np.clip(board.astype(np.float32) + tex, 0, 255).astype(np.uint8)
    elif estilo == "cuaderno":
        step = int(H / 36) if H > W else int(H / 22)
        line_col = (225, 200, 160)  # azul claro BGR
        for y in range(int(step * 2.2), H, step):
            cv2.line(board, (0, y), (W, y), line_col, 2, cv2.LINE_AA)
        mx = int(W * 0.09)
        cv2.line(board, (mx, 0), (mx, H), (140, 140, 230), 2, cv2.LINE_AA)
    else:
        grain = rng.normal(0, 1.0, (H, W, 1)).astype(np.float32)
        board = np.clip(board.astype(np.float32) + grain, 0, 255).astype(np.uint8)
    return board


def fit_into_area(src: np.ndarray, layout: Layout) -> np.ndarray:
    """Encaja la imagen de origen (contain) en la zona de dibujo de un lienzo blanco."""
    W, H = layout.width, layout.height
    ax, ay, aw, ah = layout.area
    sh, sw = src.shape[:2]
    s = min(aw / sw, ah / sh)
    nw, nh = max(1, int(sw * s)), max(1, int(sh * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    img = cv2.resize(src, (nw, nh), interpolation=interp)
    canvas = np.full((H, W, 3), 255, np.uint8)
    x0 = ax + (aw - nw) // 2
    y0 = ay + (ah - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = img
    return canvas


def normalize_background(img: np.ndarray) -> np.ndarray:
    """Blanquea fondos grisáceos/amarillentos (balance de blancos global).

    Se estima el color del papel con los píxeles más claros y se escala cada canal
    para que pase a ser blanco. Es global a propósito: un filtro local borraría
    los rellenos de color grandes y planos."""
    small = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA).reshape(-1, 3)
    v = small.max(1)
    bright = small[v >= np.percentile(v, 60)]
    if len(bright) == 0:
        return img
    paper = np.median(bright, axis=0).astype(np.float32)
    if paper.min() < 120:      # fondo oscuro: no es una imagen de pizarra blanca, no tocar
        return img
    scale = 255.0 / np.maximum(paper, 1)
    out = img.astype(np.float32) * scale
    # lo que ya es casi blanco, a blanco puro (quita textura de papel)
    out[out.min(axis=2) > 236] = 255
    return np.clip(out, 0, 255).astype(np.uint8)


def extract_ink(img: np.ndarray, min_area: int = 14) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (alpha de tinta float32 0..1, máscara booleana limpia)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2].astype(np.float32)
    s = hsv[..., 1].astype(np.float32)
    # tinta = oscuro. Un azul/rojo saturado y luminoso NO es tinta (es relleno).
    alpha = np.clip((175.0 - v) / 105.0, 0, 1)
    alpha *= np.clip(1.25 - s / 400.0, 0, 1)
    mask = alpha > 0.35
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    small = stats[:, cv2.CC_STAT_AREA] < min_area
    small[0] = False
    if small.any():
        mask[small[labels]] = False
    # incluimos 1 px alrededor para que el antialiasing también se "dibuje"
    mask = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    alpha[~mask] = 0
    return alpha, mask


def build_layers(src_bgr: np.ndarray, layout: Layout, estilo: str, board: np.ndarray | None = None,
                 color: bool = True) -> SceneLayers:
    """Calcula las capas de una escena a partir de la imagen de origen."""
    if board is None:
        board = make_board(layout, estilo)
    img = fit_into_area(src_bgr, layout)
    ax, ay, aw, ah = layout.area
    if np.median(cv2.cvtColor(img[ay:ay + ah, ax:ax + aw], cv2.COLOR_BGR2GRAY)) < 100:
        img = 255 - img   # el modelo devolvió fondo oscuro: lo pasamos a tinta oscura sobre claro
    img = normalize_background(img)
    alpha, mask = extract_ink(img)

    d = STYLE_DEFS[estilo]
    ink_col = np.array(d["ink"], np.float32)
    bf = board.astype(np.float32)

    # Capa de color: píxeles saturados que no son tinta
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1].astype(np.float32)
    val = hsv[..., 2].astype(np.float32)
    col_a = np.clip((sat - 35.0) / 60.0, 0, 1) * np.clip((val - 60.0) / 60.0, 0, 1)
    col_a = cv2.GaussianBlur(col_a, (3, 3), 0)
    has_color = bool(color and (col_a > 0.5).sum() > 500)

    a3 = alpha[..., None]
    if estilo == "tiza":
        # tiza: trazo con grano (alpha modulado por ruido)
        rng = np.random.default_rng(11)
        grain = rng.uniform(0.72, 1.0, alpha.shape).astype(np.float32)
        a3 = (alpha * grain)[..., None]
    lines = bf * (1 - a3) + ink_col * a3

    if has_color:
        c = img.astype(np.float32)
        if estilo == "tiza":
            c = c * 0.8 + 255 * 0.15          # pastel de tiza
            ca = (col_a * 0.85)[..., None]
            filled = bf * (1 - ca) + c * ca
        elif estilo == "cuaderno":
            ca = (col_a * 0.8)[..., None]
            filled = bf * (1 - ca) + (bf * c / 255.0) * ca   # multiplicar sobre el papel
        else:
            ca = col_a[..., None]
            filled = bf * (1 - ca) + c * ca
        final = filled * (1 - a3) + ink_col * a3
    else:
        final = lines

    return SceneLayers(
        base=board,
        lines=np.clip(lines, 0, 255).astype(np.uint8),
        final=np.clip(final, 0, 255).astype(np.uint8),
        ink=mask,
        has_color=has_color,
    )
