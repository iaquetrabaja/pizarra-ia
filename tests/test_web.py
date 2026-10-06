import importlib
import os

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ROOT_PATH", "/pizarra")
    monkeypatch.setenv("PIZARRA_DATA", str(tmp_path))
    monkeypatch.setenv("RENDERS_POR_DIA", "2")
    import pizarra.web.app as appmod
    appmod = importlib.reload(appmod)
    # no lanzamos renders reales en los tests: sustituimos el envío a la cola
    submitted = []

    class FakeJob:
        id = "abc"

    def fake_submit(payload, ip):
        submitted.append(payload)
        return FakeJob()

    monkeypatch.setattr(appmod.jobs, "submit", fake_submit)
    monkeypatch.setattr(appmod.jobs, "position", lambda job: 1)
    c = TestClient(appmod.app)
    c.submitted = submitted
    return c


def test_prefix_and_static(client):
    assert client.get("/pizarra/").status_code == 200
    assert client.get("/").status_code == 200  # nginx que quita el prefijo
    r = client.get("/pizarra", follow_redirects=False)
    assert r.status_code in (301, 302, 307) and r.headers["location"].endswith("/pizarra/")
    assert client.get("/pizarra/static/app.js").status_code == 200
    assert client.get("/pizarra/fonts/Roboto.ttf").status_code == 200
    html = client.get("/pizarra/").text
    assert 'src="static/app.js' in html  # rutas relativas


def test_config(client):
    d = client.get("/pizarra/api/config").json()
    assert d["max_segundos"] == 120 and "Kore" in d["voces"] and d["renders_restantes"] == 2
    assert d["voz_defecto"] == "Charon" and d["voces_info"]["Charon"] == "informativa"
    assert len(d["voces"]) == 30


def test_render_requires_key_and_limits(client):
    plan = {"titulo": "x", "escenas": [{"narracion": "Hola, esto es una prueba."}]}
    r = client.post("/pizarra/api/render", json={"plan": plan})
    assert r.status_code == 400 and "clave" in r.json()["detail"].lower()
    long_plan = {"escenas": [{"narracion": "palabra " * 120}] * 4}
    r = client.post("/pizarra/api/render", json={"clave": "x" * 20, "plan": long_plan})
    assert r.status_code == 400 and "máximo" in r.json()["detail"]
    # la clave va al trabajo pero nunca se devuelve
    r = client.post("/pizarra/api/render", json={"clave": "AIzaSECRET-123456", "plan": plan})
    assert r.status_code == 200 and "SECRET" not in r.text
    assert client.submitted[-1]["clave"] == "AIzaSECRET-123456"
    assert client.submitted[-1]["opciones"]["max_segundos"] == 120
    assert client.post("/pizarra/api/render", json={"demo": True}).status_code == 200
    r = client.post("/pizarra/api/render", json={"demo": True})
    assert r.status_code == 429


def test_unknown_job(client):
    assert client.get("/pizarra/api/trabajos/nada").status_code == 404


def test_rate_limiter_unit():
    from pizarra.web.jobs import RateLimiter
    rl = RateLimiter(2)
    assert rl.hit("a") and rl.hit("a") and not rl.hit("a")
    assert rl.hit("b")
    rl.refund("a")
    assert rl.remaining("a") == 1


# ---------------------------------------------------------------------------
# Probar voz
# ---------------------------------------------------------------------------
def _fake_pcm(seconds=1.0, sr=24000):
    import numpy as np
    t = np.arange(int(seconds * sr)) / sr
    return (np.sin(2 * np.pi * 220 * t) * 8000).astype(np.int16), sr


def test_probar_voz_validation(client):
    r = client.post("/pizarra/api/probar-voz", json={"voz": "Charon", "motor": "gemini"})
    assert r.status_code == 400 and "clave" in r.json()["detail"].lower()
    r = client.post("/pizarra/api/probar-voz", json={"clave": "x" * 20, "voz": "Nadie", "motor": "gemini"})
    assert r.status_code == 400 and "voz" in r.json()["detail"].lower()
    r = client.post("/pizarra/api/probar-voz", json={"motor": "otro"})
    assert r.status_code == 400


def test_probar_voz_gemini_sends_only_sample(client, monkeypatch):
    import pizarra.web.app as appmod
    seen = {}

    class FakeGemini:
        def __init__(self, key, max_wait=45.0):
            seen["key"] = key

        def list_models(self):
            return [{"name": "gemini-9-flash-tts", "actions": ["generateContent"]}]

        def tts(self, model, text, voice):
            seen.update(model=model, text=text, voice=voice)
            return _fake_pcm()

        def close(self):
            pass

    monkeypatch.setattr(appmod, "Gemini", FakeGemini)
    r = client.post("/pizarra/api/probar-voz", json={"clave": "AIzaSECRET-123456", "voz": "Sulafat"})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav"
    assert r.content[:4] == b"RIFF" and b"SECRET" not in r.content
    assert seen["voice"] == "Sulafat" and seen["model"] == "gemini-9-flash-tts"
    assert seen["text"] == "Hola, soy la voz Sulafat. Así sonará la narración de tu vídeo."


def test_probar_voz_piper_and_rate_limit(client, monkeypatch):
    import pizarra.tts as tts
    import pizarra.web.app as appmod
    from pizarra.tts import Speech

    def fake_piper(text, idioma="es", velocidad=1.0):
        pcm, sr = _fake_pcm(0.5)
        return Speech(pcm, sr, [(text, 0, 0.5)], "piper")

    monkeypatch.setattr(tts, "synth_piper", fake_piper)
    monkeypatch.setattr(appmod.preview_limit, "per_day", 2)
    for _ in range(2):
        r = client.post("/pizarra/api/probar-voz", json={"motor": "piper"})
        assert r.status_code == 200 and r.content[:4] == b"RIFF"
    r = client.post("/pizarra/api/probar-voz", json={"motor": "piper"})
    assert r.status_code == 429


def test_voice_catalog():
    from pizarra.config import (DEFAULT_VOICE, GEMINI_VOICE_INFO, GEMINI_VOICES, RenderOptions, voice_label,
                                voice_sample)
    assert len(GEMINI_VOICES) == 30 and DEFAULT_VOICE == "Charon" == RenderOptions().voz
    assert all(GEMINI_VOICE_INFO[v] for v in GEMINI_VOICES)
    assert voice_label("Charon") == "Charon — informativa"
    assert "Puck" in voice_sample("Puck") and voice_sample("x", "zz").startswith("Hola")
