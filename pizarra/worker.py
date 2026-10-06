"""Proceso hijo que renderiza UN trabajo de la web.

Recibe por stdin un JSON {plan, opciones, clave, offline, salida} (la clave nunca
toca el disco ni la línea de comandos) y escribe por stdout líneas:
  P {"stage":..,"progress":..,"message":..}
  R {resultado}
  E {"error": "mensaje para el usuario"}
Al terminar el proceso, la memoria (y la clave) desaparecen con él.
"""
from __future__ import annotations

import json
import logging
import sys


def _emit(kind: str, data: dict) -> None:
    sys.stdout.write(f"{kind} {json.dumps(data, ensure_ascii=False)}\n")
    sys.stdout.flush()


def main() -> int:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr, format="%(message)s")
    for noisy in ("httpx", "google_genai", "google_genai.models"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    try:
        payload = json.loads(sys.stdin.readline())
    except Exception:
        _emit("E", {"error": "Datos de trabajo no válidos."})
        return 2
    key = payload.pop("clave", None)

    from .config import RenderOptions
    from .gemini import GeminiError
    from .pipeline import render_video

    opts = RenderOptions.from_dict(payload.get("opciones", {}))

    def progress(stage, frac, msg):
        _emit("P", {"stage": stage, "progress": round(frac, 4), "message": msg})

    try:
        res = render_video(payload["plan"], opts, payload["salida"], api_key=key,
                           offline=bool(payload.get("offline")), progress=progress)
        _emit("R", res.to_dict())
        return 0
    except (GeminiError, ValueError) as e:
        _emit("E", {"error": str(e)})
        return 1
    except Exception as e:  # noqa: BLE001
        logging.exception("Fallo en el render")
        _emit("E", {"error": f"Error inesperado al crear el vídeo: {type(e).__name__}: {str(e)[:300]}"})
        return 1
    finally:
        key = None


if __name__ == "__main__":
    raise SystemExit(main())
