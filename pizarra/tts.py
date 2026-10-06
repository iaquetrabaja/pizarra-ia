"""Narración: Gemini TTS (con la clave del usuario) o Piper (local, gratis).

Cada escena devuelve audio int16 mono y los tramos temporales de cada frase,
que luego se usan para cronometrar los subtítulos:
  * Piper: se sintetiza frase a frase -> tiempos exactos.
  * Gemini: una llamada por escena -> tiempos estimados por longitud de texto
    y "pegados" a los silencios reales del audio (alineación barata).
"""
from __future__ import annotations

import logging
import subprocess
import threading
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .config import CACHE_DIR, PIPER_VOICES
from .subtitles import split_sentences

if TYPE_CHECKING:  # pragma: no cover
    from .gemini import Gemini

log = logging.getLogger("pizarra")

LEAD_IN = 0.25    # silencio al principio de cada escena (la mano empieza antes)
TAIL = 0.45       # silencio al final de cada escena
SENT_GAP = 0.22   # pausa entre frases (Piper)


@dataclass
class Speech:
    samples: np.ndarray            # int16 mono
    sr: int
    spans: list[tuple[str, float, float]]   # (frase, inicio, fin) en segundos
    engine: str

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sr


def silence(sec: float, sr: int) -> np.ndarray:
    return np.zeros(int(round(sec * sr)), np.int16)


def write_wav(path: Path, samples: np.ndarray, sr: int) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(np.ascontiguousarray(samples, dtype=np.int16).tobytes())


# --------------------------------------------------------------------------
# Piper
# --------------------------------------------------------------------------
_piper_lock = threading.Lock()
_piper_cache: dict[str, object] = {}


def piper_voice_name(idioma: str) -> str:
    return PIPER_VOICES.get(idioma) or PIPER_VOICES.get(idioma.split("-")[0], PIPER_VOICES["es"])


def load_piper(voice: str):
    with _piper_lock:
        if voice in _piper_cache:
            return _piper_cache[voice]
        try:
            from piper import PiperVoice
            from piper.download_voices import download_voice
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("Piper no está instalado: pip install piper-tts") from e
        d = CACHE_DIR / "piper"
        d.mkdir(parents=True, exist_ok=True)
        model = d / f"{voice}.onnx"
        if not model.exists() or not (d / f"{voice}.onnx.json").exists():
            log.info("Descargando voz Piper %s (sólo la primera vez, ~60 MB)...", voice)
            download_voice(voice, d)
        v = PiperVoice.load(str(model))
        _piper_cache[voice] = v
        return v


def unload_piper() -> None:
    """Libera el modelo Piper (~200 MB) cuando ya no hace falta."""
    with _piper_lock:
        _piper_cache.clear()
    import gc
    gc.collect()


def synth_piper(text: str, idioma: str = "es", velocidad: float = 1.0) -> Speech:
    voice = load_piper(piper_voice_name(idioma))
    from piper import SynthesisConfig
    cfg = SynthesisConfig(length_scale=1.0 / max(0.5, min(2.0, velocidad or 1.0)))
    sr = voice.config.sample_rate
    parts = [silence(LEAD_IN, sr)]
    spans = []
    t = LEAD_IN
    sents = split_sentences(text)
    for i, s in enumerate(sents):
        with _piper_lock:
            chunks = list(voice.synthesize(s, syn_config=cfg))
        audio = np.concatenate([c.audio_int16_array for c in chunks]) if chunks else np.zeros(0, np.int16)
        audio = trim_silence(audio, sr, keep=0.03)
        d = len(audio) / sr
        spans.append((s, t, t + d))
        parts.append(audio)
        t += d
        if i < len(sents) - 1:
            gap = SENT_GAP + (0.1 if s.endswith(("?", "!", ".")) else 0)
            parts.append(silence(gap, sr))
            t += gap
    parts.append(silence(TAIL, sr))
    return Speech(np.concatenate(parts), sr, spans, "piper")


# --------------------------------------------------------------------------
# Utilidades de audio
# --------------------------------------------------------------------------
def _rms_frames(samples: np.ndarray, sr: int, win: float = 0.02) -> tuple[np.ndarray, int]:
    n = max(1, int(sr * win))
    m = len(samples) // n
    if m == 0:
        return np.zeros(0), n
    x = samples[: m * n].astype(np.float32).reshape(m, n)
    return np.sqrt((x * x).mean(1)), n


def trim_silence(samples: np.ndarray, sr: int, keep: float = 0.08) -> np.ndarray:
    rms, n = _rms_frames(samples, sr)
    if len(rms) == 0:
        return samples
    thr = max(80.0, 0.04 * float(np.percentile(rms, 95)))
    voiced = np.nonzero(rms > thr)[0]
    if len(voiced) == 0:
        return samples
    k = int(keep * sr)
    a = max(0, voiced[0] * n - k)
    b = min(len(samples), (voiced[-1] + 1) * n + k)
    return samples[a:b]


def pause_intervals(samples: np.ndarray, sr: int, min_len: float = 0.12) -> list[tuple[float, float]]:
    """(inicio, fin) en segundos de los silencios internos de al menos min_len segundos."""
    rms, n = _rms_frames(samples, sr)
    if len(rms) == 0:
        return []
    thr = max(80.0, 0.06 * float(np.percentile(rms, 95)))
    quiet = rms < thr
    out, i, win = [], 0, n / sr
    while i < len(quiet):
        if quiet[i]:
            j = i
            while j < len(quiet) and quiet[j]:
                j += 1
            if (j - i) * win >= min_len and i > 0 and j < len(quiet):
                out.append((i * win, j * win))
            i = j
        else:
            i += 1
    return out


def find_pauses(samples: np.ndarray, sr: int, min_len: float = 0.12) -> list[float]:
    """Centros (s) de los silencios internos de al menos min_len segundos."""
    return [(a + b) / 2 for a, b in pause_intervals(samples, sr, min_len)]


def voiced_extent(samples: np.ndarray, sr: int) -> tuple[float, float]:
    """(inicio, fin) en segundos de la parte con voz (mismo umbral que trim_silence)."""
    rms, n = _rms_frames(samples, sr)
    if len(rms) == 0:
        return 0.0, len(samples) / sr
    thr = max(80.0, 0.04 * float(np.percentile(rms, 95)))
    voiced = np.nonzero(rms > thr)[0]
    if len(voiced) == 0:
        return 0.0, len(samples) / sr
    return voiced[0] * n / sr, min(len(samples), (voiced[-1] + 1) * n) / sr


def estimate_spans(sents: list[str], samples: np.ndarray, sr: int, t0: float) -> list[tuple[str, float, float]]:
    """Tramos por frase: proporcionales al nº de caracteres y ajustados a pausas reales."""
    dur = len(samples) / sr
    if not sents:
        return []
    w = np.array([len(s) + 3 for s in sents], np.float64)
    bounds = list(np.cumsum(w)[:-1] / w.sum() * dur)
    pauses = find_pauses(samples, sr)
    used = set()
    snapped = []
    prev = 0.0
    for b in bounds:
        best, bd = None, 0.9
        for k, p in enumerate(pauses):
            if k in used or p <= prev + 0.2:
                continue
            if abs(p - b) < bd:
                best, bd = k, abs(p - b)
        if best is not None:
            used.add(best)
            b = pauses[best]
        b = max(b, prev + 0.2)
        snapped.append(b)
        prev = b
    edges = [0.0] + snapped + [dur]
    return [(s, t0 + edges[i], t0 + edges[i + 1]) for i, s in enumerate(sents)]


def speech_from_pcm(text: str, pcm: np.ndarray, sr: int, engine: str) -> Speech:
    audio = trim_silence(pcm, sr)
    spans = estimate_spans(split_sentences(text), audio, sr, LEAD_IN)
    full = np.concatenate([silence(LEAD_IN, sr), audio, silence(TAIL, sr)])
    return Speech(full, sr, spans, engine)


# --------------------------------------------------------------------------
# Narración ya grabada (sin TTS)
# --------------------------------------------------------------------------
def narration_speeches(path: str | Path, texts: list[str], idioma: str = "es", fps: int = 24,
                       sr: int = 24000, log=log.info) -> list[Speech]:
    """Trocea una narración ya grabada (WAV, MP3, MP4...) en una Speech por escena, sin recortarla:
    corta en el silencio entre la última palabra de una escena y la primera de la siguiente
    (alineación con el guion; sin ASR, por longitud de texto ajustada a las pausas). Los cortes caen
    en fotogramas exactos y el vídeo dura lo mismo que el audio."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr),
                          "-f", "s16le", "-"], check=True, capture_output=True).stdout
    a = np.frombuffer(raw, dtype="<i2").astype(np.int16)
    dur = len(a) / sr
    log(f"Narración: {Path(path).name} ({dur:.1f} s)")
    cuts = None
    if len(texts) > 1:
        from .align import align_script
        words = align_script(texts, [(0.0, dur)] * len(texts), a, sr, idioma, log)
        if words and all(words):
            pauses = pause_intervals(a, sr)
            cuts = []
            for prev, nxt in zip(words, words[1:]):
                e, s0 = prev[-1].end, nxt[0].start
                cand = [(g0, g1) for g0, g1 in pauses if g0 < s0 + 0.2 and g1 > e - 0.2]
                g = max(cand, key=lambda x: x[1] - x[0]) if cand else (e, s0)
                # corte más cerca del final de la pausa (cola de 0,45 s + entrada de 0,25 s por escena)
                cuts.append(g[0] + 0.64 * (g[1] - g[0]))
            if not all(c2 > c1 for c1, c2 in zip(cuts, cuts[1:])):
                cuts = None
    if cuts is None:
        log("Narración: troceo por pausas (sin alineación).")
        cuts = [sp[2] for sp in estimate_spans(texts, a, sr, 0.0)[:-1]]
    idx = [0] + [int(round(round(c * fps) / fps * sr)) for c in cuts]
    end = int(round(np.ceil(dur * fps) / fps * sr))
    a = np.concatenate([a, np.zeros(max(0, end - len(a)), np.int16)])
    idx.append(len(a))
    out = []
    for text, i, j in zip(texts, idx, idx[1:]):
        piece = a[i:j]
        v0, v1 = voiced_extent(piece, sr)
        spans = estimate_spans(split_sentences(text), piece[int(v0 * sr):int(v1 * sr)], sr, v0)
        out.append(Speech(piece, sr, spans, "audio"))
    return out


# --------------------------------------------------------------------------
# Gemini: comprobación de duración
# --------------------------------------------------------------------------
# Segundos de habla esperables por carácter (voces de Gemini en español: ~14-16 caracteres/s).
CHARS_PER_SEC = 14.0
MAX_RATIO = 1.45   # si el audio dura más que esto x lo esperado, ha leído algo que no es el guion


def too_long(seconds: float, text: str) -> bool:
    expected = max(1.5, len(text) / CHARS_PER_SEC)
    return seconds > expected * MAX_RATIO + 1.0


def gemini_speech(gem: "Gemini", model: str, text: str, voice: str) -> Speech:
    """Narración de una escena con Gemini (sólo el texto, sin instrucciones de estilo).

    Si el audio sale mucho más largo de lo esperable (ha leído algo que no es el guion
    o se ha ido por las ramas) se repite; si vuelve a pasar, se sintetiza frase a frase."""
    pcm = sr = None
    for intento in range(2):
        pcm, sr = gem.tts(model, text, voice)
        audio = trim_silence(pcm, sr)
        if not too_long(len(audio) / sr, text):
            return speech_from_pcm(text, pcm, sr, "gemini")
        log.warning("Voz sospechosamente larga (%.1fs para %d caracteres); %s.", len(audio) / sr, len(text),
                    "repito" if intento == 0 else "voy frase a frase")
    sents = split_sentences(text)
    if len(sents) <= 1:
        return speech_from_pcm(text, pcm, sr, "gemini")
    parts = []
    for s in sents:
        best = None
        for _ in range(2):
            p, sr = gem.tts(model, s, voice)
            best = trim_silence(p, sr, keep=0.03)
            if not too_long(len(best) / sr, s):
                break
        parts += [best, silence(SENT_GAP + 0.1, sr)]
    return speech_from_pcm(text, np.concatenate(parts[:-1]), sr, "gemini")
