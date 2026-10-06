"""Subtítulos: frases cortas cronometradas, exportación .srt y quemado en vídeo."""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import FONTS_DIR, Layout


@dataclass
class Cue:
    start: float
    end: float
    text: str


_SENT_RE = re.compile(r"(?<=[.!?…;:])\s+")


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


def load_font(size: int) -> ImageFont.FreeTypeFont:
    path = FONTS_DIR / "Roboto.ttf"
    try:
        f = ImageFont.truetype(str(path), size)
        try:
            f.set_variation_by_name("Bold")
        except Exception:
            pass
        return f
    except OSError:
        for p in ("C:/Windows/Fonts/arialbd.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                  "/System/Library/Fonts/Supplemental/Arial Bold.ttf"):
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
    return ImageFont.load_default()


def font_size_for(layout: Layout) -> int:
    return int(layout.width * (0.068 if layout.height > layout.width else 0.036))


def wrap_lines(text: str, font, max_width: int) -> list[str]:
    words = text.split()
    lines, cur = [], []
    for w in words:
        test = " ".join(cur + [w])
        if cur and font.getlength(test) > max_width:
            lines.append(" ".join(cur))
            cur = [w]
        else:
            cur.append(w)
    if cur:
        lines.append(" ".join(cur))
    return lines


def balanced_lines(text: str, font, max_width: int) -> list[str]:
    """Como wrap_lines, pero si salen 2 líneas las equilibra (sin huérfanas)."""
    lines = wrap_lines(text, font, max_width)
    if len(lines) != 2:
        return lines
    words = text.split()
    best, bw = lines, max(font.getlength(l) for l in lines)
    for i in range(1, len(words)):
        a, b = " ".join(words[:i]), " ".join(words[i:])
        w = max(font.getlength(a), font.getlength(b))
        if w <= max_width and w < bw:
            best, bw = [a, b], w
    return best


def chunk_sentence(sentence: str, font, max_width: int, max_lines: int = 2, max_words: int = 9) -> list[str]:
    """Divide una frase en trozos legibles (máx. 2 líneas), cortando mejor en comas."""
    words = sentence.split()
    if not words:
        return []

    def fits(ws):
        return len(ws) <= max_words and len(wrap_lines(" ".join(ws), font, max_width)) <= max_lines

    # Partición óptima (programación dinámica): todos los trozos caben y son de
    # longitud parecida; se premia cortar tras coma y se penalizan trozos de 1-2 palabras.
    n = len(words)
    total = sum(len(w) + 1 for w in words)
    INF = float("inf")
    best = [INF] * (n + 1)
    prev = [0] * (n + 1)
    best[0] = 0.0
    # nº mínimo de trozos para fijar la longitud objetivo
    k_min, i = 0, 0
    while i < n:
        j = i + 1
        while j < n and fits(words[i:j + 1]):
            j += 1
        k_min += 1
        i = j
    target = total / max(1, k_min)
    for j in range(1, n + 1):
        for i in range(max(0, j - max_words), j):
            if best[i] == INF or not fits(words[i:j]):
                continue
            seg = words[i:j]
            ln = sum(len(w) + 1 for w in seg)
            cost = ((ln - target) / max(target, 1)) ** 2 + 1.0
            if len(seg) <= 2 and n > 3:
                cost += 1.5
            if j < n and re.search(r"[,—–;:]$", seg[-1]):
                cost -= 0.6
            if best[i] + cost < best[j]:
                best[j], prev[j] = best[i] + cost, i
    if best[n] == INF:  # palabra larguísima: un trozo por palabra
        return words
    out, j = [], n
    while j > 0:
        out.append(" ".join(words[prev[j]:j]))
        j = prev[j]
    return out[::-1]


def _weight(s: str) -> float:
    return len(s) + 4 * len(re.findall(r"[,;:]", s)) + 2


def cues_for_spans(spans: list[tuple[str, float, float]], font, max_width: int, offset: float = 0.0) -> list[Cue]:
    """spans: [(frase, inicio, fin)] relativos al audio de la escena."""
    cues = []
    for sent, t0, t1 in spans:
        parts = chunk_sentence(sent, font, max_width)
        if not parts:
            continue
        ws = [_weight(p) for p in parts]
        tot = sum(ws)
        t = t0
        for p, w in zip(parts, ws):
            d = (t1 - t0) * w / tot
            cues.append(Cue(offset + t, offset + t + d, p))
            t += d
    return cues


CAPTION_LEAD = 0.08    # el subtítulo aparece un poco antes de que empiece la voz
CAPTION_HOLD = 0.15    # y se mantiene un poco tras la última palabra
CAPTION_BRIDGE = 0.35  # huecos más cortos que esto se cierran (sin parpadeo)


def finalize_cues(cues: list[Cue]) -> list[Cue]:
    """Sin solapes; los huecos cortos entre subtítulos se cierran."""
    for a, b in zip(cues, cues[1:]):
        if a.end > b.start:
            a.end = max(a.start + 0.05, b.start)
            b.start = max(b.start, a.end)
        elif b.start - a.end < CAPTION_BRIDGE:
            a.end = b.start
    return cues


def cues_for_words(words, font, max_width: int, lead: float = CAPTION_LEAD,
                   hold: float = CAPTION_HOLD) -> list[Cue]:
    """Subtítulos a partir de palabras cronometradas (objetos con .text, .start, .end en la línea de
    tiempo global). Mismos trozos que ``cues_for_spans`` (frase a frase, máx. 2 líneas), pero cada uno
    empieza con su primera palabra (- ``lead``) y acaba con la última (+ ``hold``)."""
    cues: list[Cue] = []
    k = 0
    text = " ".join(w.text for w in words)
    for sent in split_sentences(text):
        for part in chunk_sentence(sent, font, max_width):
            n = len(part.split())
            ws = words[k:k + n]
            k += n
            if ws:
                cues.append(Cue(max(0.0, ws[0].start - lead), ws[-1].end + hold, part))
    return finalize_cues(cues)


def _ts(t: float) -> str:
    ms = int(round(max(0.0, t) * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(cues: list[Cue]) -> str:
    out = []
    for i, c in enumerate(cues, 1):
        out.append(f"{i}\n{_ts(c.start)} --> {_ts(c.end)}\n{c.text}\n")
    return "\n".join(out)


class SubtitleRenderer:
    """Pre-renderiza cada subtítulo como sprite RGBA recortado y lo mezcla por fotograma."""

    def __init__(self, layout: Layout):
        self.layout = layout
        self.size = font_size_for(layout)
        self.font = load_font(self.size)
        self.max_width = layout.sub_box[2]
        self._cache: dict[int, tuple] = {}

    def sprite(self, idx: int, text: str):
        if idx in self._cache:
            return self._cache[idx]
        bx, by, bw, bh = self.layout.sub_box
        stroke = max(3, int(self.size * 0.11))
        lines = balanced_lines(text, self.font, bw)
        lh = int(self.size * 1.18)
        total_h = lh * len(lines)
        pad = stroke + 4
        img = Image.new("RGBA", (bw + 2 * pad, total_h + 2 * pad + self.size // 3), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        y = pad
        for ln in lines:
            d.text(((bw + 2 * pad) / 2, y), ln, font=self.font, fill=(255, 255, 255, 255), anchor="ma",
                   stroke_width=stroke, stroke_fill=(0, 0, 0, 255))
            y += lh
        arr = np.asarray(img)
        alpha_any = np.nonzero(arr[..., 3].max(0))[0]
        rows = np.nonzero(arr[..., 3].max(1))[0]
        if len(alpha_any) == 0:
            self._cache[idx] = None
            return None
        arr = arr[rows[0]:rows[-1] + 1, alpha_any[0]:alpha_any[-1] + 1]
        a = arr[..., 3:4].astype(np.float32) / 255.0
        pre = arr[..., 2::-1].astype(np.float32) * a  # RGB->BGR premultiplicado
        h, w = arr.shape[:2]
        x0 = bx + (bw - w) // 2
        y0 = by + (bh - h) // 2
        H, W = self.layout.height, self.layout.width
        x0 = max(0, min(W - w, x0))
        y0 = max(0, min(H - h, y0))
        sp = (x0, y0, pre, 1.0 - a)
        self._cache = {k: v for k, v in self._cache.items() if k >= idx - 1}
        self._cache[idx] = sp
        return sp

    def blit(self, frame: np.ndarray, idx: int, text: str):
        sp = self.sprite(idx, text)
        if sp is None:
            return None
        x0, y0, pre, inv = sp
        h, w = pre.shape[:2]
        roi = frame[y0:y0 + h, x0:x0 + w]
        backup = (y0, y0 + h, x0, x0 + w, roi.copy())
        roi[:] = (roi * inv + pre).astype(np.uint8)
        return backup
