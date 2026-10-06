"""Cliente Gemini (BYOK). La clave sólo vive en memoria mientras dura el trabajo."""
from __future__ import annotations

import base64
import json
import logging
import re
import threading
import time
from collections import Counter

import numpy as np

log = logging.getLogger("pizarra")

EXCLUDE_TEXT = ("image", "tts", "live", "audio", "lite", "transcribe", "omni", "robotics",
                "computer", "embedding", "customtools", "learnlm", "gemma", "thinking", "exp")


class GeminiError(RuntimeError):
    """Error con mensaje en español apto para mostrar al usuario."""

    def __init__(self, msg: str, kind: str = "other"):
        super().__init__(msg)
        self.kind = kind  # quota | auth | permission | unavailable | other


def _version(name: str) -> float:
    m = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
    return float(m.group(1)) if m else 0.0


def _rank(name: str) -> tuple:
    """Mayor versión primero; a igualdad, estable > preview; 'latest' al final."""
    stable = 0 if ("preview" in name or "exp" in name) else 1
    return (_version(name), stable, -len(name))


def pick_models(models: list[dict]) -> dict:
    """Elige modelos por defecto a partir de ListModels."""
    gen = [m["name"] for m in models if "generateContent" in m.get("actions", [])]

    text = [n for n in gen if "flash" in n and not any(x in n for x in EXCLUDE_TEXT)]
    text_ranked = sorted([n for n in text if _version(n) > 0], key=_rank, reverse=True)
    texto = text_ranked[0] if text_ranked else ("gemini-flash-latest" if "gemini-flash-latest" in gen else None)

    imgs = [n for n in gen if "image" in n and "gemini" in n]
    pref = [n for n in imgs if "flash" in n and "lite" not in n] or [n for n in imgs if "lite" not in n] or imgs
    imagen = sorted(pref, key=_rank, reverse=True)[0] if pref else None

    ttss = [n for n in gen if "tts" in n]
    pref = [n for n in ttss if "flash" in n and "lite" not in n] or [n for n in ttss if "pro" not in n] or ttss
    tts = sorted(pref, key=_rank, reverse=True)[0] if pref else None
    return {"texto": texto, "imagen": imagen, "tts": tts}


def _friendly(e: Exception, what: str) -> GeminiError:
    code = getattr(e, "code", None)
    msg = str(e)
    low = msg.lower()
    if code == 429 or "resource_exhausted" in low or "quota" in low:
        return GeminiError(f"Gemini ({what}): has superado la cuota o el límite de peticiones de tu clave. "
                           "Espera un minuto o revisa tu plan en Google AI Studio.", "quota")
    if code in (401,) or "api key not valid" in low or "api_key_invalid" in low:
        return GeminiError("La clave de Gemini no es válida. Cópiala de nuevo desde Google AI Studio.", "auth")
    if code == 403 or "permission" in low:
        return GeminiError(f"Gemini ({what}): tu clave no tiene acceso a este modelo "
                           "(puede requerir facturación activa).", "permission")
    if code == 404 or "not found" in low:
        return GeminiError(f"Gemini ({what}): modelo no disponible para tu clave.", "unavailable")
    if code == 400 and ("billing" in low or "free tier" in low or "not available" in low):
        return GeminiError(f"Gemini ({what}): esta función no está incluida en el plan gratuito.", "permission")
    if code in (500, 502, 503, 504):
        return GeminiError(f"Gemini ({what}): servicio saturado o caído temporalmente. Inténtalo más tarde.",
                           "unavailable")
    return GeminiError(f"Gemini ({what}): {msg[:300]}", "other")


def _retry_delay(e: Exception) -> float | None:
    m = re.search(r"retry(?:Delay)?['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", str(e), re.I)
    if m:
        return float(m.group(1))
    m = re.search(r"retry in (\d+(?:\.\d+)?)s", str(e), re.I)
    return float(m.group(1)) if m else None


def parse_json(text: str):
    """Extrae JSON de una respuesta (tolera ```json ... ``` y texto alrededor)."""
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    for open_, close in (("{", "}"), ("[", "]")):
        a, b = t.find(open_), t.rfind(close)
        if a >= 0 and b > a:
            try:
                return json.loads(t[a:b + 1])
            except json.JSONDecodeError:
                continue
    raise GeminiError("Gemini devolvió una respuesta que no es JSON válido. Vuelve a intentarlo.")


class Gemini:
    def __init__(self, api_key: str, max_wait: float = 45.0):
        if not api_key or not api_key.strip():
            raise GeminiError("Falta la clave de Gemini.", "auth")
        from google import genai
        self._client = genai.Client(api_key=api_key.strip())
        self.calls: Counter = Counter()
        self.max_wait = max_wait
        self._lock = threading.Lock()

    def close(self):
        try:
            self._client.close()
        except Exception:
            pass
        self._client = None

    # ------------------------------------------------------------------
    def _call(self, what: str, fn, retries: int = 3):
        last = None
        for attempt in range(retries + 1):
            try:
                with self._lock:
                    self.calls[what] += 1
                return fn()
            except Exception as e:  # noqa: BLE001
                fe = _friendly(e, what)
                last = fe
                code = getattr(e, "code", None)
                if fe.kind in ("auth", "permission") or code == 404 or attempt == retries:
                    break
                if fe.kind == "quota":
                    d = _retry_delay(e)
                    # cuota diaria agotada -> no tiene sentido esperar
                    if d is None and "day" in str(e).lower():
                        break
                    d = min(self.max_wait, d if d is not None else 8 * (attempt + 1))
                else:
                    d = 3 * (attempt + 1)
                log.warning("%s: %s (reintento en %.0fs)", what, fe, d)
                time.sleep(d)
        raise last

    def list_models(self) -> list[dict]:
        def go():
            out = []
            for m in self._client.models.list():
                out.append({"name": m.name.replace("models/", ""),
                            "actions": list(m.supported_actions or []),
                            "display": m.display_name or ""})
            return out
        return self._call("modelos", go, retries=1)

    def text(self, model: str, prompt: str, json_mode: bool = False, search: bool = False,
             temperature: float | None = None) -> str:
        from google.genai import types
        cfg = {}
        if json_mode and not search:
            cfg["response_mime_type"] = "application/json"
        if search:
            cfg["tools"] = [types.Tool(google_search=types.GoogleSearch())]
        if temperature is not None:
            cfg["temperature"] = temperature

        def go():
            r = self._client.models.generate_content(
                model=model, contents=prompt, config=types.GenerateContentConfig(**cfg))
            if not r.text:
                raise GeminiError("Gemini devolvió una respuesta vacía.")
            return r.text
        return self._call("texto" if not search else "investigación", go)

    def image(self, model: str, prompt: str, aspect: str) -> np.ndarray:
        """Genera una imagen y la devuelve en BGR."""
        import cv2
        from google.genai import types

        def req(with_aspect: bool):
            cfg = {"response_modalities": ["IMAGE", "TEXT"]}
            if with_aspect:
                cfg["image_config"] = types.ImageConfig(aspect_ratio=aspect)
            r = self._client.models.generate_content(
                model=model, contents=prompt, config=types.GenerateContentConfig(**cfg))
            for cand in r.candidates or []:
                for part in (cand.content.parts if cand.content else []) or []:
                    if part.inline_data is not None and part.inline_data.data:
                        data = part.inline_data.data
                        if isinstance(data, str):
                            data = base64.b64decode(data)
                        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                        if img is not None:
                            return img
            raise GeminiError("El modelo de imagen no devolvió ninguna imagen (¿filtro de seguridad?).")

        def go():
            try:
                return req(True)
            except Exception as e:  # algunos modelos no aceptan image_config
                if getattr(e, "code", None) == 400 and "aspect" in str(e).lower():
                    return req(False)
                raise
        return self._call("imagen", go, retries=2)

    def tts(self, model: str, text: str, voice: str) -> tuple[np.ndarray, int]:
        """Sólo se envía el texto a leer: los modelos TTS a veces leen en voz alta las
        instrucciones de estilo (aunque vayan entre corchetes). El tono lo da la voz."""
        from google.genai import types

        def go():
            r = self._client.models.generate_content(
                model=model,
                contents=text,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice))),
                ))
            part = r.candidates[0].content.parts[0] if r.candidates and r.candidates[0].content else None
            if part is None or part.inline_data is None:
                raise GeminiError("Gemini TTS no devolvió audio.")
            data = part.inline_data.data
            if isinstance(data, str):
                data = base64.b64decode(data)
            mime = part.inline_data.mime_type or ""
            m = re.search(r"rate=(\d+)", mime)
            sr = int(m.group(1)) if m else 24000
            if "wav" in mime and data[:4] == b"RIFF":
                import io
                import wave
                with wave.open(io.BytesIO(data)) as wf:
                    sr = wf.getframerate()
                    data = wf.readframes(wf.getnframes())
            return np.frombuffer(data, np.int16).copy(), sr
        return self._call("voz", go, retries=2)
