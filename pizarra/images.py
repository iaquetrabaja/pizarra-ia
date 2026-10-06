"""Imágenes de escena: prompt de pizarra para el modelo de imagen, o dibujo
vectorial (primitivas) generado por el modelo de texto como alternativa gratuita."""
from __future__ import annotations

import textwrap

import numpy as np

from .demo import fit_shapes
from .gemini import Gemini, GeminiError, parse_json
from .vector_art import render_shapes


def image_prompt(scene: dict, aspect: str, titulo: str = "") -> str:
    label = (scene.get("etiqueta") or "").strip()
    if label:
        text_rule = (f'The ONLY text allowed is the short label "{label}" written once in simple, '
                     "clean handwritten capital letters. No other words, letters or numbers.")
    else:
        text_rule = "Absolutely no text, letters, numbers or words anywhere in the image."
    return textwrap.dedent(f"""\
        Whiteboard explainer illustration drawn with a thick black marker on a pure white background.
        Subject: {scene.get('visual') or scene.get('narracion')}
        Context (do not write it): {scene.get('narracion', '')[:300]}
        Style: minimal clean line art like a whiteboard animation video; bold, continuous, uniform black
        outlines; simple iconic shapes; flat; no shading, no gradients, no hatching, no shadows, no
        texture, no 3D; one or two flat color accents filled inside some shapes (soft yellow, orange,
        light blue or green), everything else black and white; generous empty white space; centered,
        uncluttered composition with a few large elements that fill most of the frame.
        {text_rule} Avoid tiny details and tiny writing inside objects.
        Show only the finished drawing itself, seen straight on, with nobody around it: no human hand,
        arm or writing tool entering the picture (unless the subject itself is a person). No frame, no
        border, no whiteboard edge, no signature, no watermark. Pure white (#FFFFFF) background.
        Aspect ratio {aspect}.""")


def vector_prompt(scene: dict, w: int, h: int) -> str:
    label = (scene.get("etiqueta") or "").strip()
    return textwrap.dedent(f"""\
        You are a whiteboard illustrator. Draw this idea as simple line art using primitives:
        "{scene.get('visual') or scene.get('narracion')}"
        Canvas: {w} wide x {h} tall, origin top-left. Keep 60px margins. Use large, simple, recognizable
        icon-like shapes (objects, people as simple figures, arrows, diagrams) that fill the canvas.
        Use 12–40 primitives. At most 2 shapes with a fill color (soft hex colors like #F6C744, #8FC7F0,
        #7CC47A, #F28C38). {'Include the text label "' + label + '" once (size 60-90).' if label else 'No text.'}
        Allowed primitives (JSON objects):
          {{"type":"line","points":[[x,y],[x,y],...],"closed":false}}
          {{"type":"poly","points":[[x,y],...],"fill":"#hex"}}
          {{"type":"circle","center":[x,y],"r":R,"fill":"#hex"}}
          {{"type":"arc","center":[x,y],"r":R,"start":deg,"end":deg}}
          {{"type":"rect","x":X,"y":Y,"w":W,"h":H,"fill":"#hex"}}
          {{"type":"arrow","from":[x,y],"to":[x,y]}}
          {{"type":"text","pos":[x,y],"text":"...","size":70}}
        Primitives are drawn in order; "fill" is optional. Respond ONLY with JSON: {{"shapes":[...]}}""")


def vector_image(gem: Gemini, model: str, scene: dict, aspect: float, seed: int = 0) -> np.ndarray:
    w = 1000
    h = int(round(w / aspect))
    data = None
    for attempt in range(2):  # a veces el modelo devuelve JSON roto: un reintento basta
        try:
            data = parse_json(gem.text(model, vector_prompt(scene, w, h), json_mode=True, temperature=0.6))
            break
        except GeminiError as e:
            if e.kind != "other" or attempt == 1:
                raise
    shapes = data.get("shapes", []) if isinstance(data, dict) else data
    if not shapes:
        raise ValueError("dibujo vacío")
    return render_shapes(shapes, aspect=aspect, seed=seed)


def placeholder_image(scene: dict, aspect: float, seed: int = 0) -> np.ndarray:
    """Dibujo de reserva sin IA: las formas incluidas en el guion o una etiqueta en un bocadillo."""
    if scene.get("formas"):
        return render_shapes(fit_shapes(scene["formas"], aspect), aspect=aspect, seed=seed)
    label = (scene.get("etiqueta") or " ".join((scene.get("visual") or scene.get("narracion", "")).split()[:3]))
    shapes = [
        {"type": "circle", "center": [500, 430], "r": 300, "fill": "#F6C744"},
        {"type": "line", "points": [[380, 700], [330, 880], [520, 720]]},
        {"type": "text", "pos": [500, 430], "text": label[:22], "size": 80},
    ]
    return render_shapes(fit_shapes(shapes, aspect), aspect=aspect, seed=seed)
