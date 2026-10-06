"""Subtítulos alineados con la voz: guion <-> ASR, agrupado, narración grabada e integración Piper + Whisper."""
import numpy as np
import pytest

from pizarra.align import Word, align_tokens, fill_missing, norm_token, refine_with_energy
from pizarra.subtitles import cues_for_words, load_font, to_srt


def asr(*items):
    return [Word(t, s, e) for t, s, e in items]


def test_norm_token():
    assert norm_token("¿Estrés?") == "estres"
    assert norm_token("Año,") == norm_token("año") == "ano"


def test_align_insertions_deletions_accents_numbers():
    script = "Ante una imagen nueva, predice si encaja con el 99% de acierto.".split()
    rec = asr(("Ante", 0.0, 0.2), ("una", 0.2, 0.4), ("imágen", 0.4, 0.7), ("nueva", 0.7, 1.0), ("eh", 1.0, 1.1),
              ("predice", 1.3, 1.8), ("si", 1.8, 1.9), ("encaja", 1.9, 2.3), ("con", 2.3, 2.4), ("el", 2.4, 2.5),
              ("noventa", 2.5, 2.8), ("y", 2.8, 2.85), ("nueve", 2.85, 3.1), ("por", 3.1, 3.2), ("ciento", 3.2, 3.5),
              ("acierto", 3.7, 4.2))
    t = align_tokens(script, rec)
    assert t[2] == (0.4, 0.7)                     # tildes distintas
    assert t[4] == (1.3, 1.8)                     # "eh" insertado no desplaza
    # "99% de" <-> "noventa y nueve por ciento" ("de" no reconocida): el tramo se reparte entre las dos
    assert t[9][0] == pytest.approx(2.5) and t[10][1] == pytest.approx(3.5)
    assert t[9][1] <= t[10][0]
    assert t[11] == (3.7, 4.2)
    words = fill_missing(script, t, 0.0, 4.5)
    assert all(a.end <= b.start + 1e-9 for a, b in zip(words, words[1:]))


def test_refine_word_swallowing_a_pause():
    sr = 16000
    tone = lambda s: 0.3 * np.ones(int(s * sr), np.float32)  # noqa: E731
    a = np.concatenate([tone(1.0), np.zeros(int(0.25 * sr), np.float32), tone(0.6)])
    ws = [Word("nueva,", 0.5, 0.92), Word("predice", 0.92, 1.8)]
    refine_with_energy(ws, a, sr)
    assert ws[1].start == pytest.approx(1.25, abs=0.02)


def test_cues_for_words_timing():
    text = "Al final, calcula probabilidades matemáticas. Ante una imagen nueva, predice si encaja."
    ws, t = [], 1.0
    for w in text.split():
        ws.append(Word(w, t, t + 0.3))
        t += 0.4
    cues = cues_for_words(ws, load_font(60), 900)
    assert " ".join(c.text for c in cues) == text
    assert cues[0].start == pytest.approx(1.0 - 0.08)
    starts = {c.text.split()[0]: c.start for c in cues}
    assert starts["Ante"] == pytest.approx(ws[5].start - 0.08)   # nunca cruza la frase
    assert all(a.end <= b.start + 1e-9 for a, b in zip(cues, cues[1:]))
    assert to_srt(cues).count("-->") == len(cues)


def _have(mod):
    try:
        __import__(mod)
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not (_have("faster_whisper") and _have("piper")), reason="faltan faster-whisper o piper")
def test_piper_captions_follow_asr_words():
    from pizarra import align
    from pizarra.tts import synth_piper

    texts = ["La IA no es magia ni tiene conciencia. Es pura estadística reconociendo patrones.",
             "Al final, calcula probabilidades. Ante una imagen nueva, predice si encaja con lo que ya conoce."]
    try:
        speeches = [synth_piper(t, "es") for t in texts]
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Piper no disponible: {e}")
    sr = speeches[0].sr
    audio = np.concatenate([s.samples for s in speeches])
    ranges, t = [], 0.0
    for s in speeches:
        ranges.append((t + s.spans[0][1], t + s.spans[-1][2]))
        t += s.duration
    aligned = align.align_script(texts, ranges, audio, sr, "es", log=lambda m: None)
    if aligned is None:
        pytest.skip("Whisper no disponible")
    font = load_font(60)
    cues = [c for ws in aligned for c in cues_for_words(ws, font, 900)]
    # referencia: palabras del ASR (con el inicio sacado del silencio previo, error típico de Whisper)
    rec = refine_with_energy(align.transcribe(audio, sr, "es", log=lambda m: None), audio, sr)
    words = [w for c in cues for w in c.text.split()]
    times = align_tokens(words, rec)
    k, checked = 0, 0
    for c in cues:
        if times[k] is not None:
            assert abs(c.start - times[k][0]) <= 0.15, (c.text, c.start, times[k][0])
            checked += 1
        k += len(c.text.split())
    assert checked >= 0.8 * len(cues)
    # Piper sintetiza frase a frase: el inicio real de cada frase se conoce exactamente
    t = 0.0
    for s in speeches:
        for sent, s0, _s1 in s.spans:
            first = [c for c in cues if c.text.split()[0] == sent.split()[0] and abs(c.start - (t + s0)) < 1.0]
            assert first, sent
            assert abs(first[0].start + 0.08 - (t + s0)) <= 0.15, (sent, first[0].start, t + s0)
        t += s.duration
