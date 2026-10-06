"""Aplicación web (FastAPI).

Variables de entorno:
  ROOT_PATH          prefijo público, p. ej. /pizarra (nginx puede quitarlo o no: funciona igual)
  PIZARRA_DATA       carpeta de trabajos (por defecto ./datos)
  RENDERS_POR_DIA    vídeos por IP y día (3)
  GUIONES_POR_DIA    guiones por IP y día (20)
  MAX_SEGUNDOS       duración máxima del vídeo (120)
  MAX_COLA           trabajos en cola como máximo (20)
  RETENCION_HORAS    horas que se guardan los vídeos (24)
  TRUST_PROXY        1 si está detrás de nginx (usa X-Forwarded-For / X-Real-IP)
  RENDER_THREADS     hilos de ffmpeg (0 = auto)

Arranque:  uvicorn pizarra.web.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..config import FONTS_DIR, GEMINI_VOICES, LANGUAGES, STYLES, RenderOptions
from ..demo import demo_plan
from ..gemini import Gemini, GeminiError, pick_models
from ..script import estimate_seconds, generate_plan, normalize_plan
from .jobs import JobManager, RateLimiter

STATIC = Path(__file__).parent / "static"
ROOT_PATH = os.environ.get("ROOT_PATH", "").rstrip("/")
DATA = Path(os.environ.get("PIZARRA_DATA", "datos")).resolve()
RENDERS_POR_DIA = int(os.environ.get("RENDERS_POR_DIA", "3"))
GUIONES_POR_DIA = int(os.environ.get("GUIONES_POR_DIA", "20"))
MAX_SEGUNDOS = float(os.environ.get("MAX_SEGUNDOS", "120"))
TRUST_PROXY = os.environ.get("TRUST_PROXY", "0") == "1"
THREADS = int(os.environ.get("RENDER_THREADS", "0"))

jobs = JobManager(DATA, max_queue=int(os.environ.get("MAX_COLA", "20")),
                  retention_hours=float(os.environ.get("RETENCION_HORAS", "24")))
render_limit = RateLimiter(RENDERS_POR_DIA)
script_limit = RateLimiter(GUIONES_POR_DIA)

router = APIRouter()


def client_ip(request: Request) -> str:
    if TRUST_PROXY:
        xff = request.headers.get("x-forwarded-for")
        if xff:
            return xff.split(",")[0].strip()
        xr = request.headers.get("x-real-ip")
        if xr:
            return xr.strip()
    return request.client.host if request.client else "?"


def _err(e: Exception, status: int = 400):
    raise HTTPException(status_code=status, detail=str(e))


# ---------------------------------------------------------------------------
class KeyIn(BaseModel):
    clave: str = Field(min_length=10, max_length=200)


class ScriptIn(KeyIn):
    tema: str = Field(min_length=3, max_length=500)
    idioma: str = "es"
    duracion: float = 60
    investigar: bool = False
    modelo_texto: str | None = None


class RenderIn(BaseModel):
    clave: str | None = Field(default=None, max_length=200)
    demo: bool = False
    plan: dict | None = None
    opciones: dict = {}


# ---------------------------------------------------------------------------
@router.get("/", response_class=HTMLResponse)
def index():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html, headers={"Cache-Control": "no-cache"})


@router.get("/api/config")
def config(request: Request):
    ip = client_ip(request)
    return {"voces": GEMINI_VOICES, "estilos": STYLES, "idiomas": LANGUAGES, "max_segundos": MAX_SEGUNDOS,
            "renders_restantes": render_limit.remaining(ip), "renders_por_dia": RENDERS_POR_DIA,
            **jobs.stats()}


@router.post("/api/modelos")
def modelos(body: KeyIn):
    try:
        g = Gemini(body.clave)
        ms = g.list_models()
        g.close()
    except GeminiError as e:
        _err(e, 400)
    gen = [m["name"] for m in ms if "generateContent" in m["actions"]]
    return {
        "defecto": pick_models(ms),
        "texto": [n for n in gen if "gemini" in n and not any(x in n for x in ("image", "tts", "embedding"))],
        "imagen": [n for n in gen if "image" in n],
        "tts": [n for n in gen if "tts" in n],
    }


@router.post("/api/guion")
def guion(body: ScriptIn, request: Request):
    ip = client_ip(request)
    if not script_limit.hit(ip):
        _err(Exception(f"Has llegado al límite de {GUIONES_POR_DIA} guiones por día."), 429)
    dur = max(15.0, min(MAX_SEGUNDOS, body.duracion))
    try:
        g = Gemini(body.clave)
        try:
            model = body.modelo_texto or pick_models(g.list_models())["texto"]
            if not model:
                raise GeminiError("Tu clave no tiene ningún modelo de texto disponible.")
            plan = generate_plan(g, model, body.tema, body.idioma if body.idioma in LANGUAGES else "es",
                                 dur, body.investigar)
        finally:
            g.close()
    except GeminiError as e:
        script_limit.refund(ip)
        _err(e, 400)
    except ValueError as e:
        script_limit.refund(ip)
        _err(e, 400)
    plan["segundos_estimados"] = round(estimate_seconds(plan))
    return plan


@router.post("/api/render")
def render(body: RenderIn, request: Request):
    ip = client_ip(request)
    opts = RenderOptions.from_dict(body.opciones)
    # opciones que no decide el usuario en la web
    opts.musica = None
    opts.max_segundos = MAX_SEGUNDOS
    opts.threads = THREADS
    opts.fps = 24
    try:
        opts.validate()
        if body.demo:
            plan, offline = demo_plan(), True
            opts.idioma, opts.tts = "es", "piper"
        else:
            if not body.clave:
                raise ValueError("Falta la clave de Gemini.")
            plan, offline = normalize_plan(body.plan or {}, opts.idioma), False
            est = estimate_seconds(plan)
            if est > MAX_SEGUNDOS * 1.15:
                raise ValueError(f"El guion dura unos {est:.0f} s y el máximo es {MAX_SEGUNDOS:.0f} s. "
                                 "Acorta la narración.")
    except ValueError as e:
        _err(e, 400)
    if not render_limit.hit(ip):
        _err(Exception(f"Has llegado al límite de {RENDERS_POR_DIA} vídeos por día. Vuelve mañana o "
                       "instala Pizarra IA en tu ordenador (es gratis y de código abierto)."), 429)
    payload = {"plan": plan, "opciones": opts.to_dict(), "clave": body.clave, "offline": offline}
    try:
        job = jobs.submit(payload, ip)
    except OverflowError as e:
        render_limit.refund(ip)
        _err(e, 503)
    return {"id": job.id, "posicion": jobs.position(job)}


def _job_or_404(jid: str):
    job = jobs.get(jid)
    if not job:
        raise HTTPException(404, "Trabajo no encontrado (los vídeos se borran a las 24 h).")
    return job


@router.get("/api/trabajos/{jid}")
def estado(jid: str):
    job = _job_or_404(jid)
    return job.public(jobs.position(job))


@router.post("/api/trabajos/{jid}/cancelar")
def cancelar(jid: str):
    job = _job_or_404(jid)
    jobs.cancel(job)
    return job.public(None)


@router.get("/api/trabajos/{jid}/{name}")
def descargar(jid: str, name: str):
    job = _job_or_404(jid)
    files = {"video.mp4": ("video.mp4", "video/mp4"), "subtitulos.srt": ("subtitulos.srt", "application/x-subrip"),
             "guion.json": ("guion.json", "application/json")}
    if name not in files or job.status != "listo":
        raise HTTPException(404, "Archivo no disponible.")
    fn, mime = files[name]
    path = job.out_dir / fn
    if not path.exists():
        raise HTTPException(404, "Archivo caducado.")
    return FileResponse(path, media_type=mime, filename=f"pizarra-{jid[:6]}-{fn}")


@router.get("/api/salud")
def salud():
    return {"ok": True, **jobs.stats()}


def build_app() -> FastAPI:
    inner = FastAPI(title="Pizarra IA", docs_url=None, redoc_url=None, openapi_url=None)
    inner.include_router(router)
    inner.mount("/static", StaticFiles(directory=STATIC), name="static")
    inner.mount("/fonts", StaticFiles(directory=FONTS_DIR), name="fonts")

    @inner.exception_handler(HTTPException)
    async def http_exc(_, exc: HTTPException):
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)

    if not ROOT_PATH:
        return inner
    outer = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @outer.get(ROOT_PATH, include_in_schema=False)
    def _slash():
        return RedirectResponse(ROOT_PATH + "/")

    outer.mount(ROOT_PATH, inner)   # nginx sin quitar el prefijo
    outer.mount("/", inner)         # nginx quitando el prefijo (proxy_pass .../;)
    return outer


app = build_app()
