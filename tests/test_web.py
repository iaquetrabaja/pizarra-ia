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
    assert 'src="static/app.js"' in html  # rutas relativas


def test_config(client):
    d = client.get("/pizarra/api/config").json()
    assert d["max_segundos"] == 120 and "Kore" in d["voces"] and d["renders_restantes"] == 2


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
