import json
import subprocess

import numpy as np
import pytest

from pizarra.config import Layout, RenderOptions
from pizarra.demo import DEMO_SHAPES, demo_plan, fit_shapes
from pizarra.draw import Hand, SceneAnimator, plan_strokes
from pizarra.gemini import GeminiError, parse_json, pick_models
from pizarra.script import build_script_prompt, estimate_seconds, normalize_plan
from pizarra.style import build_layers, make_board
from pizarra.subtitles import (Cue, balanced_lines, chunk_sentence, cues_for_spans, load_font,
                               split_sentences, to_srt, wrap_lines)
from pizarra.tts import estimate_spans, find_pauses, trim_silence
from pizarra.vector_art import render_shapes


# ---------------------------------------------------------------- modelos
MODELS = [{"name": n, "actions": ["generateContent"]} for n in [
    "gemini-2.5-flash", "gemini-2.5-pro", "gemini-flash-latest", "gemini-3.8-flash", "gemini-3.8-flash-lite",
    "gemini-3.7-flash", "gemini-3-flash-preview", "gemini-2.5-flash-image", "gemini-3.1-flash-image",
    "gemini-3.1-flash-image-preview", "gemini-3-pro-image", "gemini-3.1-flash-lite-image",
    "gemini-2.5-flash-preview-tts", "gemini-3.8-flash-tts", "gemini-3.8-flash-lite-tts", "gemini-2.5-pro-preview-tts",
]] + [{"name": "gemini-3.8-live", "actions": ["bidiGenerateContent"]}]


def test_pick_models_prefers_latest_stable_flash():
    d = pick_models(MODELS)
    assert d == {"texto": "gemini-3.8-flash", "imagen": "gemini-3.1-flash-image", "tts": "gemini-3.8-flash-tts"}


def test_pick_models_handles_missing():
    d = pick_models([{"name": "gemini-flash-latest", "actions": ["generateContent"]}])
    assert d == {"texto": "gemini-flash-latest", "imagen": None, "tts": None}


@pytest.mark.parametrize("raw", ['{"a": 1}', '```json\n{"a": 1}\n```', 'Aquí tienes: {"a": 1} ¡listo!'])
def test_parse_json_tolerant(raw):
    assert parse_json(raw) == {"a": 1}


def test_parse_json_invalid():
    with pytest.raises(GeminiError):
        parse_json("no es json")


# ---------------------------------------------------------------- guion
def test_normalize_plan_cleans_and_validates():
    p = normalize_plan({"title": "T", "scenes": [{"narration": "  Hola   mundo. "}, {"narracion": ""},
                                                  {"narracion": "Dos", "etiqueta": "uno dos tres cuatro cinco"}]})
    assert p["titulo"] == "T"
    assert [s["narracion"] for s in p["escenas"]] == ["Hola mundo.", "Dos"]
    assert p["escenas"][0]["visual"] == "Hola mundo."
    assert p["escenas"][1]["etiqueta"] == "uno dos tres cuatro"
    with pytest.raises(ValueError):
        normalize_plan({"escenas": [{"narracion": " "}]})
    with pytest.raises(ValueError):
        normalize_plan({"escenas": []})


def test_demo_plan_is_valid_and_short():
    p = normalize_plan(demo_plan())
    assert len(p["escenas"]) == 4 and all("formas" in s for s in p["escenas"])
    assert 15 < estimate_seconds(p) < 35


def test_script_prompt_mentions_language_and_length():
    pr = build_script_prompt("La fotosíntesis", "es", 60, research="dato 1")
    assert "español de España" in pr and "La fotosíntesis" in pr and "dato 1" in pr


def test_render_options_validation():
    RenderOptions(formato="16:9", estilo="tiza").validate()
    with pytest.raises(ValueError):
        RenderOptions(formato="4:3").validate()
    with pytest.raises(ValueError):
        RenderOptions(estilo="acuarela").validate()


# ---------------------------------------------------------------- subtítulos
def test_chunks_fit_and_no_orphans():
    font = load_font(73)
    s = "Escribes un tema, y Pizarra IA se pone a trabajar para ti sin descanso alguno."
    chunks = chunk_sentence(s, font, 950)
    assert " ".join(chunks) == s
    assert all(len(c.split()) >= 2 for c in chunks)
    assert all(len(wrap_lines(c, font, 950)) <= 2 for c in chunks)


def test_balanced_lines():
    font = load_font(73)
    lines = balanced_lines("y Pizarra IA se pone a trabajar.", font, 950)
    assert len(lines) <= 2
    if len(lines) == 2:
        assert len(lines[1].split()) >= 2


def test_split_sentences():
    assert split_sentences("Hola. ¿Qué tal? Bien!  Fin") == ["Hola.", "¿Qué tal?", "Bien!", "Fin"]


def test_cues_and_srt():
    font = load_font(60)
    cues = cues_for_spans([("Primera frase corta.", 0.25, 1.5), ("Segunda frase, algo más larga que la otra.", 1.8, 4.0)],
                          font, 900, offset=10)
    assert cues[0].start == pytest.approx(10.25) and cues[-1].end == pytest.approx(14.0)
    assert all(a.end <= b.start + 1e-6 for a, b in zip(cues, cues[1:]))
    srt = to_srt([Cue(0.0, 1.5, "Hola"), Cue(61.25, 3725.5, "Adiós")])
    assert "1\n00:00:00,000 --> 00:00:01,500\nHola" in srt
    assert "2\n00:01:01,250 --> 01:02:05,500\nAdiós" in srt


# ---------------------------------------------------------------- audio
def _tone(sec, sr=24000, f=220):
    t = np.arange(int(sec * sr)) / sr
    return (np.sin(2 * np.pi * f * t) * 8000).astype(np.int16)


def test_pauses_and_span_snapping():
    sr = 24000
    z = lambda s: np.zeros(int(s * sr), np.int16)  # noqa: E731
    audio = np.concatenate([z(0.3), _tone(1.0), z(0.4), _tone(2.0), z(0.3)])
    trimmed = trim_silence(audio, sr, keep=0.05)
    assert len(trimmed) / sr == pytest.approx(3.5, abs=0.15)
    pauses = find_pauses(trimmed, sr)
    assert len(pauses) == 1 and pauses[0] == pytest.approx(1.25, abs=0.1)
    # texto de igual longitud -> estimación a mitad (1.75 s); debe pegarse a la pausa real (1.25 s)
    spans = estimate_spans(["Frase uno aquí.", "Frase dos aquí."], trimmed, sr, 0.0)
    assert spans[0][2] == pytest.approx(1.25, abs=0.1)
    assert spans[1][2] == pytest.approx(len(trimmed) / sr)


# ---------------------------------------------------------------- dibujo
def test_plan_strokes_covers_every_pixel_once():
    m = np.zeros((300, 400), bool)
    m[50:56, 20:380] = True          # línea horizontal
    m[100:250, 200:206] = True       # línea vertical
    m[280:284, 10:20] = True         # detalle pequeño
    P = plan_strokes(m, cell=8)
    assert len(P.order_px) == m.sum()
    assert len(np.unique(P.order_px)) == m.sum()
    assert P.cell_end[-1] == m.sum()
    assert (P.cell_xy[:, 0] < 400).all() and (P.cell_xy[:, 1] < 300).all()


def test_line_is_drawn_in_one_pass():
    m = np.zeros((100, 600), bool)
    m[40:52, 10:590] = True  # trazo grueso: 2 celdas de alto
    P = plan_strokes(m, cell=8)
    xs = P.cell_xy[:, 0]
    # la punta recorre la línea en un sentido sin volver atrás
    assert (np.diff(xs) >= 0).all() or (np.diff(xs) <= 0).all()


def test_empty_mask():
    P = plan_strokes(np.zeros((50, 50), bool))
    assert P.n_cells == 0


@pytest.mark.parametrize("estilo", ["pizarra", "tiza", "cuaderno"])
def test_layers_keep_large_color_fills(estilo):
    layout = Layout.for_format("9:16")
    img = render_shapes(fit_shapes(DEMO_SHAPES[2], 0.75), aspect=0.75)
    L = build_layers(img, layout, estilo)
    assert L.has_color
    assert L.ink.any()
    # el relleno verde de la montaña debe seguir presente en la imagen final
    diff = np.abs(L.final.astype(int) - L.lines.astype(int)).sum(2)
    assert (diff > 40).sum() > 20000


def test_animator_frames_and_final_state():
    layout = Layout.for_format("16:9")
    img = render_shapes(fit_shapes(DEMO_SHAPES[0], 16 / 9), aspect=16 / 9)
    L = build_layers(img, layout, "pizarra", make_board(layout, "pizarra"))
    P = plan_strokes(L.ink)
    hand = Hand(300)
    anim = SceneAnimator(L, P, hand, n_frames=60, fps=24)
    frames = list(anim.frames())
    assert len(frames) == 60
    last, pos = frames[-1]
    assert pos is None
    assert np.array_equal(last, L.final)
    # en mitad del dibujo hay mano y el lienzo está a medias
    mid, mpos = frames[10]
    assert mpos is not None


def test_hand_blit_restores_exactly():
    from pizarra.draw import restore
    frame = np.full((500, 400, 3), 200, np.uint8)
    ref = frame.copy()
    h = Hand(250)
    b = h.blit(frame, 380, 450)  # cerca del borde: se recorta
    assert not np.array_equal(frame, ref)
    restore(frame, b)
    assert np.array_equal(frame, ref)


# ---------------------------------------------------------------- extremo a extremo (offline)
def test_offline_render_end_to_end(tmp_path):
    from pizarra.pipeline import render_video
    plan = demo_plan()
    plan["escenas"] = plan["escenas"][:1]
    res = render_video(plan, RenderOptions(formato="9:16", fps=12), tmp_path, offline=True)
    assert res.video.exists() and res.srt.exists()
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height",
                          "-of", "json", str(res.video)], capture_output=True, text=True, check=True)
    streams = json.loads(out.stdout)["streams"]
    v = [s for s in streams if s["codec_type"] == "video"][0]
    assert (v["width"], v["height"]) == (1080, 1920)
    assert any(s["codec_type"] == "audio" for s in streams)
    assert "-->" in res.srt.read_text(encoding="utf-8")


# ---------------------------------------------------------------- alternativas sin cuota (Gemini simulado)
class FakeGemini:
    """Clave gratuita sin cuota de imagen ni de voz: debe caer a vectorial + Piper."""

    def __init__(self, key):
        from collections import Counter
        self.calls = Counter()

    def list_models(self):
        return MODELS

    def image(self, model, prompt, aspect):
        self.calls["imagen"] += 1
        raise GeminiError("Gemini (imagen): has superado la cuota.", "quota")

    def text(self, model, prompt, json_mode=False, search=False, temperature=None):
        self.calls["texto"] += 1
        return json.dumps({"shapes": [{"type": "circle", "center": [500, 500], "r": 200, "fill": "#F6C744"},
                                      {"type": "text", "pos": [500, 800], "text": "HOLA", "size": 80}]})

    def tts(self, model, text, voice, style):
        self.calls["voz"] += 1
        raise GeminiError("Gemini (voz): has superado la cuota.", "quota")

    def close(self):
        pass


def test_fallbacks_without_quota(tmp_path, monkeypatch):
    import pizarra.pipeline as pl
    monkeypatch.setattr(pl, "Gemini", FakeGemini)
    plan = {"titulo": "t", "escenas": [{"narracion": "Una frase de prueba."}, {"narracion": "Otra frase más."},
                                       {"narracion": "Y la última."}]}
    res = pl.render_video(plan, RenderOptions(fps=12), tmp_path, api_key="fake-key-123")
    assert res.video.exists()
    assert res.engines["voz"].startswith("piper")
    assert "vector" in res.engines["imagenes"]
    assert res.api_calls["imagen"] == 1          # tras el primer 429 no se insiste
    assert res.api_calls["texto"] == 3           # un dibujo vectorial por escena
    assert any("cuota" in w for w in res.warnings)
