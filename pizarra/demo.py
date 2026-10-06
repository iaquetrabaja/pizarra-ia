"""Guion y dibujos de demostración para el modo --offline (sin clave)."""
from __future__ import annotations

import copy

Y = "#F6C744"   # amarillo
B = "#8FC7F0"   # celeste
G = "#7CC47A"   # verde
O = "#F28C38"   # naranja

# Dibujos diseñados en un lienzo cuadrado de 1000x1000; luego se encajan.
DEMO_SHAPES = [
    # 1. Bombilla: la idea
    [
        {"type": "circle", "center": [500, 400], "r": 220, "fill": Y},
        {"type": "line", "points": [[400, 610], [420, 700], [580, 700], [600, 610]]},
        {"type": "rect", "x": 420, "y": 700, "w": 160, "h": 45},
        {"type": "rect", "x": 435, "y": 745, "w": 130, "h": 40},
        {"type": "line", "points": [[470, 785], [500, 820], [530, 785]]},
        {"type": "line", "points": [[450, 560], [470, 420], [500, 480], [530, 420], [550, 560]]},
        {"type": "line", "points": [[500, 120], [500, 60]]},
        {"type": "line", "points": [[300, 200], [255, 155]]},
        {"type": "line", "points": [[700, 200], [745, 155]]},
        {"type": "line", "points": [[230, 400], [165, 400]]},
        {"type": "line", "points": [[770, 400], [835, 400]]},
        {"type": "text", "pos": [500, 920], "text": "IDEA", "size": 90},
    ],
    # 2. Portátil con la IA escribiendo el guion
    [
        {"type": "rect", "x": 170, "y": 180, "w": 660, "h": 430, "fill": B},
        {"type": "rect", "x": 205, "y": 215, "w": 590, "h": 360, "fill": "#FFFFFF"},
        {"type": "line", "points": [[110, 690], [890, 690], [830, 610], [170, 610]], "closed": True},
        {"type": "line", "points": [[250, 290], [560, 290]]},
        {"type": "line", "points": [[250, 360], [700, 360]]},
        {"type": "line", "points": [[250, 430], [640, 430]]},
        {"type": "line", "points": [[250, 500], [480, 500]]},
        {"type": "circle", "center": [700, 280], "r": 45, "fill": O},
        {"type": "text", "pos": [700, 282], "text": "IA", "size": 50},
        {"type": "arrow", "from": [500, 900], "to": [500, 730]},
        {"type": "text", "pos": [500, 960], "text": "guion", "size": 80},
    ],
    # 3. Un cuadro con paisaje y un lápiz: dibuja cada escena
    [
        {"type": "rect", "x": 150, "y": 160, "w": 620, "h": 480},
        {"type": "poly", "points": [[170, 620], [360, 360], [470, 500], [560, 400], [750, 620]], "fill": G},
        {"type": "circle", "center": [640, 270], "r": 60, "fill": Y},
        {"type": "poly", "points": [[700, 900], [880, 560], [940, 590], [760, 930]], "fill": O},
        {"type": "line", "points": [[700, 900], [690, 975], [760, 930]]},
        {"type": "text", "pos": [400, 780], "text": "escena 1", "size": 70},
    ],
    # 4. Móvil con el vídeo listo
    [
        {"type": "rect", "x": 320, "y": 80, "w": 360, "h": 700},
        {"type": "rect", "x": 345, "y": 150, "w": 310, "h": 540, "fill": "#FFFFFF"},
        {"type": "circle", "center": [500, 735], "r": 22},
        {"type": "circle", "center": [500, 420], "r": 105, "fill": O},
        {"type": "poly", "points": [[465, 360], [465, 480], [565, 420]], "fill": "#FFFFFF"},
        {"type": "line", "points": [[150, 870], [215, 935], [330, 800]]},
        {"type": "text", "pos": [620, 875], "text": "LISTO", "size": 100},
    ],
]

DEMO_PLAN = {
    "titulo": "Pizarra IA en 25 segundos",
    "idioma": "es",
    "escenas": [
        {
            "narracion": "Todo empieza con una idea. Escribes un tema, y Pizarra IA se pone a trabajar.",
            "visual": "Una bombilla encendida con rayos y la palabra IDEA.",
        },
        {
            "narracion": "La inteligencia artificial escribe el guion y lo divide en escenas. Tú puedes revisarlo y cambiar lo que quieras.",
            "visual": "Un portátil con líneas de texto en la pantalla y un círculo con las letras IA.",
        },
        {
            "narracion": "Después dibuja cada escena a mano, trazo a trazo, sobre la pizarra.",
            "visual": "Un cuadro con montañas y un sol, junto a un lápiz.",
        },
        {
            "narracion": "Y en unos minutos tienes tu vídeo con voz y subtítulos, listo para publicar.",
            "visual": "Un móvil con un botón de reproducir y un tick de listo.",
        },
    ],
}


def demo_plan() -> dict:
    plan = copy.deepcopy(DEMO_PLAN)
    for i, sc in enumerate(plan["escenas"]):
        sc["formas"] = copy.deepcopy(DEMO_SHAPES[i % len(DEMO_SHAPES)])
    return plan


def fit_shapes(shapes: list[dict], aspect: float) -> list[dict]:
    """Encaja formas diseñadas en 1000x1000 dentro de un lienzo 1000 x 1000/aspect."""
    h = 1000 / aspect
    s = min(1.0, h / 1000)
    ox = (1000 - 1000 * s) / 2
    oy = (h - 1000 * s) / 2

    def P(p):
        return [p[0] * s + ox, p[1] * s + oy]

    out = []
    for sh in copy.deepcopy(shapes):
        if "points" in sh:
            sh["points"] = [P(p) for p in sh["points"]]
        for k in ("center", "from", "to", "pos"):
            if k in sh:
                sh[k] = P(sh[k])
        if "x" in sh:
            sh["x"], sh["y"] = P([sh["x"], sh["y"]])
            sh["w"] = sh.get("w", 100) * s
            sh["h"] = sh.get("h", 100) * s
        for k in ("r", "size"):
            if k in sh:
                sh[k] = sh[k] * s
        out.append(sh)
    return out
