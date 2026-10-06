"""Alineación palabra a palabra de la narración (subtítulos sincronizados con la voz).

1. Reconocimiento de voz con marcas de tiempo por palabra (faster-whisper, CPU, int8).
2. El GUION manda en lo que se muestra: las palabras reconocidas se alinean con las del
   guion (difflib.SequenceMatcher sobre tokens normalizados: sin tildes, minúsculas, sin
   puntuación). Las palabras del guion que el ASR no reconoce o reconoce distinto heredan
   tiempos interpolados de sus vecinas.
3. Ajuste fino con la energía del audio: si el ASR pone el inicio de una palabra en un
   silencio, se adelanta al primer instante con voz (Whisper tiende a "comerse" la pausa).

Si faster-whisper no está instalado o falla, quien llama usa la estimación por longitud.
(Mismo módulo que en Stickman IA.)
"""
from __future__ import annotations

import difflib
import os
import re
import threading
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

Log = Callable[[str], None]

MODEL_NAME = os.environ.get("PIZARRA_WHISPER_MODEL", "base")
ASR_SR = 16000


@dataclass
class Word:
    text: str
    start: float
    end: float


# --------------------------------------------------------------------------- #
# Modelo (uno por proceso)
# --------------------------------------------------------------------------- #
_model = None
_model_lock = threading.Lock()


def whisper_dir() -> Path:
    env = os.environ.get("PIZARRA_WHISPER_DIR")
    if env:
        p = Path(env)
    else:
        from .config import CACHE_DIR
        p = CACHE_DIR / "whisper"
    p.mkdir(parents=True, exist_ok=True)
    return p


def available() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def load_model(name: str = MODEL_NAME, log: Log = print):
    """Carga (y descarga la primera vez) el modelo Whisper. int8, 2 hilos."""
    global _model
    with _model_lock:
        if _model is None:
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
            from faster_whisper import WhisperModel

            folder = whisper_dir()
            if not any(folder.glob(f"models--*faster-whisper-{name}")):
                log(f"Descargando modelo de alineación Whisper '{name}' (solo la primera vez)...")
            threads = int(os.environ.get("PIZARRA_WHISPER_THREADS", "2"))
            _model = WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=threads,
                                  num_workers=1, download_root=str(folder))
        return _model


def _resample(a: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out or len(a) == 0:
        return a.astype(np.float32)
    n_out = int(round(len(a) * sr_out / sr_in))
    x_old = np.linspace(0.0, 1.0, num=len(a), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, a).astype(np.float32)


def transcribe(samples: np.ndarray, sr: int, language: str = "es", prompt: Optional[str] = None,
               log: Log = print) -> List[Word]:
    """Palabras reconocidas con su inicio/fin (s). ``samples`` en float32 [-1, 1] o int16."""
    a = samples.astype(np.float32)
    if samples.dtype == np.int16:
        a = a / 32768.0
    a = _resample(a, sr, ASR_SR)
    model = load_model(log=log)
    segments, _info = model.transcribe(
        a, language=(language or "es").split("-")[0], word_timestamps=True, beam_size=1,
        condition_on_previous_text=False, initial_prompt=prompt or None, vad_filter=False)
    out: List[Word] = []
    for seg in segments:
        for w in seg.words or []:
            t = w.word.strip()
            if t:
                out.append(Word(t, float(w.start), float(w.end)))
    return out


# --------------------------------------------------------------------------- #
# Alineación guion <-> ASR
# --------------------------------------------------------------------------- #
def norm_token(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "", s)


def _weight(w: str) -> float:
    return len(w) + 1.5


def align_tokens(script: Sequence[str], asr: Sequence[Word]) -> List[Optional[Tuple[float, float]]]:
    """Tiempos (inicio, fin) del ASR para cada palabra del guion, o None si no hay pareja.

    - bloques iguales: 1 a 1;
    - bloques sustituidos ("2" ~ "dos", "por ciento" ~ "%", palabras mal reconocidas): el
      tramo de tiempo del bloque ASR se reparte entre las palabras del guion por longitud;
    - palabras del guion que faltan en el ASR: None (se interpolan después);
    - palabras de más en el ASR: se ignoran.
    """
    a_idx = [i for i, w in enumerate(script) if norm_token(w)]
    b_idx = [j for j, w in enumerate(asr) if norm_token(w.text)]
    a = [norm_token(script[i]) for i in a_idx]
    b = [norm_token(asr[j].text) for j in b_idx]
    out: List[Optional[Tuple[float, float]]] = [None] * len(script)
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal" or (tag == "replace" and i2 - i1 == j2 - j1):
            for k in range(i2 - i1):
                w = asr[b_idx[j1 + k]]
                out[a_idx[i1 + k]] = (w.start, w.end)
        elif tag == "replace":
            # sustitución desigual: si son pocas palabras, reparto proporcional del tramo
            if (i2 - i1) > 4 * (j2 - j1) + 3 or (j2 - j1) > 4 * (i2 - i1) + 3:
                continue  # demasiado distinto: mejor interpolar
            s0, s1 = asr[b_idx[j1]].start, asr[b_idx[j2 - 1]].end
            ws = [_weight(script[a_idx[i]]) for i in range(i1, i2)]
            tot = sum(ws)
            cur = s0
            for i, wgt in zip(range(i1, i2), ws):
                d = (s1 - s0) * wgt / tot
                out[a_idx[i]] = (cur, cur + d)
                cur += d
    # descarta anclas no monótonas (emparejamientos espurios)
    last = -1e9
    for i, t in enumerate(out):
        if t is None:
            continue
        if t[0] < last - 0.05:
            out[i] = None
        else:
            last = max(last, t[0])
    return out


def fill_missing(script: Sequence[str], times: Sequence[Optional[Tuple[float, float]]],
                 lo: float, hi: float) -> List[Word]:
    """Interpola por longitud las palabras sin tiempo entre sus vecinas y deja los tiempos
    monótonos dentro de [lo, hi]."""
    n = len(script)
    res: List[Optional[Tuple[float, float]]] = list(times)
    i = 0
    while i < n:
        if res[i] is not None:
            i += 1
            continue
        j = i
        while j < n and res[j] is None:
            j += 1
        a = res[i - 1][1] if i > 0 else lo
        b = res[j][0] if j < n else hi
        if b < a:
            b = a
        ws = [_weight(script[k]) for k in range(i, j)]
        tot = sum(ws)
        cur = a
        for k, wgt in zip(range(i, j), ws):
            d = (b - a) * wgt / tot
            res[k] = (cur, cur + d)
            cur += d
        i = j
    out: List[Word] = []
    for w, (s, e) in zip(script, res):  # type: ignore[misc]
        s = min(max(s, lo), hi)
        if out:
            p = out[-1]
            s = max(s, p.start)
            if s < p.end:  # sin solapes: recorta el final de la anterior
                p.end = s
        e = min(max(e, s), hi)
        out.append(Word(w, s, e))
    return out


_SENT_END = re.compile(r"[.!?…:;]+[»\"')]*$")


def _longest_run(mask: np.ndarray) -> Tuple[int, int]:
    """(inicio, longitud) del tramo más largo de True."""
    best, run, run0 = (0, 0), 0, 0
    for i, v in enumerate(mask):
        if v:
            if run == 0:
                run0 = i
            run += 1
            if run > best[1]:
                best = (run0, run)
        else:
            run = 0
    return best


def refine_with_energy(words: List[Word], samples: np.ndarray, sr: int, offset: float = 0.0,
                       max_shift: float = 0.3) -> List[Word]:
    """Corrige los errores típicos de Whisper con la energía del audio:

    - si el inicio de una palabra cae en silencio, lo lleva al primer fotograma con voz;
    - si el final cae en silencio, lo trae al último con voz (como mucho ``max_shift``);
    - si la primera palabra de una frase queda "pegada" delante de una pausa larga (Whisper
      la coloca al final de la frase anterior), la mueve detrás de la pausa;
    - si una palabra "se traga" una pausa (>= 0,15 s) justo al principio, empieza tras la pausa.

    ``offset`` = tiempo del primer sample de ``samples`` en la línea de tiempo de ``words``."""
    if not words or len(samples) == 0:
        return words
    a = samples.astype(np.float32)
    if samples.dtype == np.int16:
        a = a / 32768.0
    hop = max(1, int(sr * 0.01))
    nfr = len(a) // hop
    if nfr == 0:
        return words
    rms = np.sqrt((a[: nfr * hop].reshape(nfr, hop) ** 2).mean(axis=1))
    voiced_rms = rms[rms > 1e-4]
    ref = float(np.percentile(voiced_rms, 95)) if len(voiced_rms) else 0.0
    thr = max(0.004, 0.06 * ref)
    voiced = rms > thr

    def fr(t: float) -> int:
        return int(np.clip(round((t - offset) / 0.01), 0, nfr - 1))

    lim = int(max_shift / 0.01)
    for k, w in enumerate(words):
        f0, f1 = fr(w.start), fr(w.end)
        nxt = words[k + 1] if k + 1 < len(words) else None
        if (nxt is not None and k > 0 and _SENT_END.search(words[k - 1].text)
                and not _SENT_END.search(w.text)):
            fn = fr(nxt.start)
            q = ~voiced[f0:fn]
            if len(q) > 30:
                # pausa más larga dentro de [inicio de la palabra, inicio de la siguiente]
                r0, rl = _longest_run(q)
                if rl >= 30 and r0 <= 35:   # >= 0.3 s de pausa y la palabra iba justo antes
                    w.start = offset + (f0 + r0 + rl) * 0.01
                    w.end = max(w.start, min(nxt.start, w.start + max(0.05, w.end - (offset + f0 * 0.01))))
                    f0, f1 = fr(w.start), fr(w.end)
        if f1 <= f0:
            continue
        r0, rl = _longest_run(~voiced[f0:f1])
        if rl >= 15 and r0 <= 15 and f0 + r0 + rl < f1:
            # la palabra empieza con el final de la anterior + una pausa: empieza tras la pausa
            f0 = f0 + r0 + rl
            w.start = offset + f0 * 0.01
        if not voiced[f0]:
            seg = np.nonzero(voiced[f0:f1 + 1])[0]
            if len(seg):
                w.start = offset + (f0 + seg[0]) * 0.01
        if not voiced[f1]:
            b0 = max(fr(w.start), f1 - lim)
            seg = np.nonzero(voiced[b0:f1 + 1])[0]
            if len(seg):
                w.end = offset + (b0 + seg[-1] + 1) * 0.01
        if k + 1 < len(words):
            w.end = min(w.end, max(w.start, words[k + 1].start))
        w.end = max(w.end, w.start)
    return words


def align_script(texts: Sequence[str], ranges: Sequence[Tuple[float, float]], samples: np.ndarray, sr: int,
                 language: str = "es", log: Log = print) -> Optional[List[List[Word]]]:
    """Palabras del guion (separadas por espacios) con tiempos reales, por escena.

    ``texts``: narración de cada escena; ``ranges``: (inicio, fin) en segundos de la voz de
    cada escena dentro de ``samples``. Devuelve None si el ASR no está disponible o falla."""
    if not available():
        log("Alineación: faster-whisper no está instalado; uso la estimación por longitud.")
        return None
    t = time.time()
    try:
        asr = transcribe(samples, sr, language, log=log)
    except Exception as e:  # noqa: BLE001
        log(f"Alineación: falló el reconocimiento de voz ({e}); uso la estimación por longitud.")
        return None
    script_words: List[str] = []
    owner: List[int] = []
    for i, tx in enumerate(texts):
        ws = tx.split()
        script_words += ws
        owner += [i] * len(ws)
    times = align_tokens(script_words, asr)
    # una palabra solo puede caer dentro del tramo de su escena
    for k, tm in enumerate(times):
        if tm is not None:
            lo, hi = ranges[owner[k]]
            if tm[1] < lo - 0.3 or tm[0] > hi + 0.3:
                times[k] = None
    matched = sum(1 for tm in times if tm is not None)
    out: List[List[Word]] = []
    k = 0
    for i, tx in enumerate(texts):
        n = len(tx.split())
        lo, hi = ranges[i]
        out.append(fill_missing(script_words[k:k + n], times[k:k + n], lo, hi))
        k += n
    flat = [w for ws in out for w in ws]
    refine_with_energy(flat, samples, sr)
    ratio = matched / max(1, len(script_words))
    log(f"Alineación: {len(asr)} palabras reconocidas, {matched}/{len(script_words)} emparejadas "
        f"con el guion ({ratio:.0%}) en {time.time() - t:.1f} s (Whisper {MODEL_NAME}).")
    if ratio < 0.4:
        log("Alineación: el audio no se parece al guion; uso la estimación por longitud.")
        return None
    return out
