"""Guion: generación con Gemini, normalización y validación del JSON editable.

Formato del guion (lo que el usuario revisa y edita):
{
  "titulo": "...",
  "idioma": "es",
  "escenas": [
    {"narracion": "texto que se lee en voz alta",
     "visual": "qué se dibuja en la pizarra",
     "etiqueta": "texto corto opcional dentro del dibujo (máx. 3 palabras)"}
  ]
}
"""
from __future__ import annotations

import re

from .config import LANGUAGES
from .gemini import Gemini, parse_json

CHARS_PER_SEC = 13.5   # ritmo medio de locución (≈ 135 palabras/min)
MAX_SCENES = 20


def estimate_seconds(plan: dict) -> float:
    txt = " ".join(s.get("narracion", "") for s in plan.get("escenas", []))
    n = len(plan.get("escenas", []))
    return len(txt) / CHARS_PER_SEC + n * 0.9


def normalize_plan(plan: dict, idioma: str | None = None) -> dict:
    """Limpia y valida un guion (venga de Gemini o editado a mano). Lanza ValueError."""
    if not isinstance(plan, dict):
        raise ValueError("El guion debe ser un objeto JSON.")
    scenes_in = plan.get("escenas") or plan.get("scenes") or []
    if not isinstance(scenes_in, list) or not scenes_in:
        raise ValueError("El guion no tiene escenas.")
    scenes = []
    for s in scenes_in[:MAX_SCENES]:
        if not isinstance(s, dict):
            continue
        narr = str(s.get("narracion") or s.get("narration") or "").strip()
        narr = re.sub(r"\s+", " ", narr)
        if not narr:
            continue
        vis = str(s.get("visual") or s.get("description") or narr).strip()
        et = str(s.get("etiqueta") or s.get("label") or "").strip()
        et = " ".join(et.split()[:4])[:32]
        sc = {"narracion": narr[:900], "visual": vis[:600], "etiqueta": et}
        if isinstance(s.get("formas"), list):
            sc["formas"] = s["formas"]
        scenes.append(sc)
    if not scenes:
        raise ValueError("Todas las escenas están vacías: escribe la narración de al menos una.")
    return {
        "titulo": str(plan.get("titulo") or plan.get("title") or "Vídeo de pizarra").strip()[:120],
        "idioma": idioma or plan.get("idioma") or "es",
        "escenas": scenes,
    }


def scene_count(duracion: float) -> int:
    return int(max(3, min(12, round(duracion / 8.5))))


def build_script_prompt(tema: str, idioma: str, duracion: float, research: str | None = None) -> str:
    lang = LANGUAGES.get(idioma, idioma)
    n = scene_count(duracion)
    words = int(duracion * 2.1)  # ≈ 13 caracteres/s de locución + pausas entre escenas
    research_block = ""
    if research:
        research_block = f"\nDATOS VERIFICADOS (úsalos, son de una búsqueda web reciente):\n---\n{research[:6000]}\n---\n"
    return f"""Eres guionista de vídeos cortos explicativos con animación de pizarra (whiteboard).
Tema: "{tema}"
{research_block}
Escribe el guion de un vídeo de unos {int(duracion)} segundos en {lang}.
Reglas:
- {n} escenas (puedes usar {max(3, n - 1)}–{n + 1}). En total, unas {words} palabras de narración.
- La primera escena engancha con una pregunta o dato sorprendente; la última cierra con una idea clara.
- Frases cortas, tono cercano y concreto, sin relleno ni frases hechas. Nada de "en este vídeo".
- Cada escena: "narracion" (lo que se dice, 1–3 frases), "visual" (UNA idea visual muy simple,
  dibujable con pocas líneas: objetos, iconos, diagramas, flechas; describe composición) y
  "etiqueta" (opcional, 1–3 palabras que aparecerán escritas en el dibujo, o cadena vacía).
- Escribe los números como se pronuncian si son ambiguos.
Responde SOLO con JSON:
{{"titulo": "...", "escenas": [{{"narracion": "...", "visual": "...", "etiqueta": "..."}}]}}"""


def build_research_prompt(tema: str, idioma: str) -> str:
    lang = LANGUAGES.get(idioma, idioma)
    return (f"Busca en la web información actual y fiable sobre: \"{tema}\". "
            f"Resume en {lang} los datos clave (cifras, fechas, ejemplos concretos) en 10-15 viñetas breves. "
            "Sin introducción.")


def generate_plan(gem: Gemini, model: str, tema: str, idioma: str = "es", duracion: float = 60,
                  investigar: bool = False, log=lambda m: None) -> dict:
    research = None
    if investigar:
        try:
            log("Investigando en la web…")
            research = gem.text(model, build_research_prompt(tema, idioma), search=True)
        except Exception as e:  # noqa: BLE001
            log(f"No se pudo investigar en la web ({e}); sigo sin investigación.")
    log("Escribiendo el guion…")
    raw = gem.text(model, build_script_prompt(tema, idioma, duracion, research), json_mode=True, temperature=0.8)
    data = parse_json(raw)
    if isinstance(data, list):
        data = {"escenas": data}
    plan = normalize_plan(data, idioma)
    plan["tema"] = tema
    return plan
