"""Línea de comandos.

Ejemplos:
  python -m pizarra "Cómo funciona la fotosíntesis" --clave $GEMINI_API_KEY
  python -m pizarra "Tema" --review                # escribe salida/guion.json y para
  python -m pizarra --guion salida/guion.json      # renderiza un guion editado
  python -m pizarra --offline                      # demo sin clave (dibujos y voz locales)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

from .config import GEMINI_VOICES, LANGUAGES, MAX_SECONDS_DEFAULT, STYLES, RenderOptions


def _slug(s: str) -> str:
    s = re.sub(r"[^\w\s-]", "", s.lower(), flags=re.U).strip()
    return re.sub(r"[\s_-]+", "-", s)[:40] or "video"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m pizarra",
                                description="Pizarra IA: vídeos de animación de pizarra con tu clave de Gemini.")
    p.add_argument("tema", nargs="?", help="Tema del vídeo")
    p.add_argument("--clave", default=os.environ.get("GEMINI_API_KEY"),
                   help="Clave de Gemini (o variable de entorno GEMINI_API_KEY)")
    p.add_argument("--formato", default="9:16", choices=["9:16", "16:9"])
    p.add_argument("--estilo", default="pizarra", choices=list(STYLES))
    p.add_argument("--duracion", type=float, default=60, help="Duración objetivo en segundos (máx. 120)")
    p.add_argument("--idioma", default="es", choices=list(LANGUAGES))
    p.add_argument("--investigar", action="store_true", help="Investigar el tema en la web (Google Search)")
    p.add_argument("--review", action="store_true", help="Sólo escribir el guion (guion.json) para editarlo")
    p.add_argument("--guion", help="Renderizar un guion JSON ya escrito/editado")
    p.add_argument("--offline", action="store_true", help="Demo sin clave: guion, dibujos y voz locales")
    p.add_argument("--salida", default=None, help="Carpeta de salida")
    p.add_argument("--voz", default="Kore", help=f"Voz Gemini ({', '.join(GEMINI_VOICES[:8])}…)")
    p.add_argument("--tts", default="auto", choices=["auto", "gemini", "piper"])
    p.add_argument("--velocidad", type=float, default=1.0, help="Velocidad de la voz Piper")
    p.add_argument("--imagenes", default="auto", choices=["auto", "ia", "vector"],
                   help="ia = modelo de imagen; vector = dibujo por primitivas con el modelo de texto (gratis)")
    p.add_argument("--sin-color", action="store_true", help="No rellenar de color al final de cada escena")
    p.add_argument("--sin-subtitulos", action="store_true")
    p.add_argument("--musica", help="Archivo de música de fondo (opcional)")
    p.add_argument("--volumen-musica", type=float, default=0.12)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--modelo-texto")
    p.add_argument("--modelo-imagen")
    p.add_argument("--modelo-tts")
    p.add_argument("--listar-modelos", action="store_true", help="Mostrar modelos disponibles para tu clave")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")
    for noisy in ("httpx", "google_genai", "google_genai.models", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    from .gemini import Gemini, GeminiError, pick_models
    from .pipeline import Cancelled, make_plan, render_video
    from .script import normalize_plan

    if args.listar_modelos:
        if not args.clave:
            print("Necesitas --clave o GEMINI_API_KEY.", file=sys.stderr)
            return 2
        g = Gemini(args.clave)
        ms = g.list_models()
        for m in ms:
            if "generateContent" in m["actions"]:
                print(m["name"])
        print("\nPor defecto:", json.dumps(pick_models(ms), ensure_ascii=False))
        return 0

    opts = RenderOptions(
        formato=args.formato, estilo=args.estilo, idioma=args.idioma, fps=args.fps, tts=args.tts,
        voz=args.voz, velocidad=args.velocidad, imagenes=args.imagenes, color=not args.sin_color,
        subtitulos=not args.sin_subtitulos, musica=args.musica, volumen_musica=args.volumen_musica,
        modelo_texto=args.modelo_texto, modelo_imagen=args.modelo_imagen, modelo_tts=args.modelo_tts,
        max_segundos=None)
    try:
        opts.validate()
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    duracion = max(15, min(MAX_SECONDS_DEFAULT, args.duracion))

    last = {"t": 0.0, "msg": ""}

    def progress(stage, frac, msg):
        now = time.time()
        if msg != last["msg"] or now - last["t"] > 2 or frac >= 1:
            print(f"[{frac * 100:5.1f}%] {msg}", flush=True)
            last.update(t=now, msg=msg)

    try:
        if args.guion:
            plan = json.loads(Path(args.guion).read_text(encoding="utf-8"))
            plan = normalize_plan(plan, plan.get("idioma") or args.idioma)
            tema = plan.get("tema") or plan["titulo"]
        else:
            if not args.tema and not args.offline:
                print("Indica un tema, --guion archivo.json o --offline.", file=sys.stderr)
                return 2
            if not args.offline and not args.clave:
                print("Falta la clave de Gemini: usa --clave o la variable GEMINI_API_KEY "
                      "(consíguela gratis en https://aistudio.google.com/apikey), o prueba --offline.",
                      file=sys.stderr)
                return 2
            tema = args.tema or "demo"
            print("Escribiendo el guion…" if not args.offline else "Modo offline: guion de demostración.")
            plan = make_plan(tema, opts, args.clave, duracion, args.investigar, args.offline, progress)

        out = Path(args.salida or Path("salida") / f"{time.strftime('%Y%m%d-%H%M%S')}-{_slug(tema)}")
        out.mkdir(parents=True, exist_ok=True)
        if args.review:
            gp = out / "guion.json"
            plan_out = {k: v for k, v in plan.items() if not k.startswith("_")}
            gp.write_text(json.dumps(plan_out, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\nGuion guardado en {gp}\nEdítalo y después ejecuta:\n"
                  f"  python -m pizarra --guion \"{gp}\" --formato {args.formato} --estilo {args.estilo}"
                  + (" --offline" if args.offline else ""))
            return 0

        res = render_video(plan, opts, out, api_key=args.clave, offline=args.offline, progress=progress)
        print(f"\nVídeo: {res.video}\nSubtítulos: {res.srt}\nGuion: {res.plan}")
        print(f"Duración: {res.duration:.1f} s · tiempos: {res.seconds}")
        print(f"Motores: {res.engines}")
        if res.api_calls or plan.get("_llamadas_api"):
            calls = dict(plan.get("_llamadas_api", {}))
            for k, v in res.api_calls.items():
                calls[k] = calls.get(k, 0) + v
            print(f"Llamadas a la API de Gemini: {calls}")
        for w in res.warnings:
            print(f"Aviso: {w}")
        return 0
    except (GeminiError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except (KeyboardInterrupt, Cancelled):
        print("Cancelado.", file=sys.stderr)
        return 130
