"""Cola de trabajos: UN render a la vez, el resto espera con su posición."""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("pizarra.web")


@dataclass
class Job:
    id: str
    ip: str
    created: float
    out_dir: Path
    status: str = "en_cola"          # en_cola | procesando | listo | error | cancelado
    stage: str = ""
    progress: float = 0.0
    message: str = "En cola"
    error: str | None = None
    result: dict | None = None
    started: float | None = None
    finished: float | None = None
    payload: dict | None = field(default=None, repr=False)   # contiene la clave: se borra al empezar
    proc: subprocess.Popen | None = field(default=None, repr=False)
    cancel_requested: bool = False

    def public(self, position: int | None) -> dict:
        d = {"id": self.id, "estado": self.status, "fase": self.stage, "progreso": round(self.progress, 3),
             "mensaje": self.message, "error": self.error, "posicion": position}
        if self.status == "listo" and self.result:
            d["duracion"] = self.result.get("duration")
            d["avisos"] = self.result.get("warnings", [])
            d["motores"] = self.result.get("engines", {})
            d["tiempo_render"] = self.result.get("seconds", {}).get("total")
        return d


class RateLimiter:
    def __init__(self, per_day: int):
        self.per_day = per_day
        self.hits: dict[str, deque] = {}
        self.lock = threading.Lock()

    def _clean(self, ip: str) -> deque:
        q = self.hits.setdefault(ip, deque())
        cutoff = time.time() - 86400
        while q and q[0] < cutoff:
            q.popleft()
        return q

    def remaining(self, ip: str) -> int:
        with self.lock:
            return max(0, self.per_day - len(self._clean(ip)))

    def hit(self, ip: str) -> bool:
        with self.lock:
            q = self._clean(ip)
            if len(q) >= self.per_day:
                return False
            q.append(time.time())
            return True

    def refund(self, ip: str) -> None:
        with self.lock:
            q = self.hits.get(ip)
            if q:
                q.pop()


class JobManager:
    def __init__(self, data_dir: Path, max_queue: int = 20, retention_hours: float = 24,
                 max_render_seconds: int = 1200):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self.queue: deque[str] = deque()
        self.max_queue = max_queue
        self.retention = retention_hours * 3600
        self.max_render_seconds = max_render_seconds
        self.cv = threading.Condition()
        self.current: str | None = None
        threading.Thread(target=self._worker, daemon=True, name="render-worker").start()
        threading.Thread(target=self._janitor, daemon=True, name="janitor").start()

    # -- API ---------------------------------------------------------------
    def submit(self, payload: dict, ip: str) -> Job:
        with self.cv:
            if len(self.queue) >= self.max_queue:
                raise OverflowError("La cola está llena ahora mismo. Inténtalo en unos minutos.")
            jid = uuid.uuid4().hex[:16]
            job = Job(jid, ip, time.time(), self.data_dir / jid, payload=payload)
            self.jobs[jid] = job
            self.queue.append(jid)
            self.cv.notify_all()
            return job

    def get(self, jid: str) -> Job | None:
        return self.jobs.get(jid)

    def position(self, job: Job) -> int | None:
        with self.cv:
            if job.status != "en_cola":
                return None
            try:
                return self.queue.index(job.id) + 1 + (1 if self.current else 0)
            except ValueError:
                return None

    def cancel(self, job: Job) -> None:
        with self.cv:
            job.cancel_requested = True
            if job.status == "en_cola":
                try:
                    self.queue.remove(job.id)
                except ValueError:
                    pass
                job.status, job.message, job.payload = "cancelado", "Cancelado", None
        if job.proc and job.proc.poll() is None:
            job.proc.kill()

    def stats(self) -> dict:
        with self.cv:
            return {"en_cola": len(self.queue), "procesando": bool(self.current)}

    # -- worker ------------------------------------------------------------
    def _worker(self) -> None:
        while True:
            with self.cv:
                while not self.queue:
                    self.cv.wait()
                jid = self.queue.popleft()
                job = self.jobs[jid]
                self.current = jid
            try:
                self._run(job)
            except Exception as e:  # noqa: BLE001
                log.exception("fallo del worker")
                job.status, job.error = "error", f"Error interno: {e}"
            finally:
                job.payload = None
                job.proc = None
                job.finished = time.time()
                with self.cv:
                    self.current = None

    def _run(self, job: Job) -> None:
        payload = job.payload or {}
        job.payload = None  # la clave ya sólo vive en la variable local hasta pasarla al hijo
        job.status, job.started, job.message = "procesando", time.time(), "Empezando…"
        job.out_dir.mkdir(parents=True, exist_ok=True)
        payload["salida"] = str(job.out_dir)
        env = dict(os.environ)
        env.pop("GEMINI_API_KEY", None)
        env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.Popen([sys.executable, "-m", "pizarra.worker"], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                                env=env)
        job.proc = proc
        proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        proc.stdin.close()
        payload.clear()
        del payload

        timer = threading.Timer(self.max_render_seconds, proc.kill)
        timer.start()
        stderr_tail: deque[str] = deque(maxlen=20)
        threading.Thread(target=lambda: [stderr_tail.append(l) for l in proc.stderr], daemon=True).start()
        try:
            for line in proc.stdout:
                kind, _, data = line.strip().partition(" ")
                try:
                    obj = json.loads(data) if data else {}
                except json.JSONDecodeError:
                    continue
                if kind == "P":
                    job.stage = obj.get("stage", "")
                    job.progress = float(obj.get("progress", 0))
                    job.message = obj.get("message", "")
                elif kind == "R":
                    job.result = obj
                elif kind == "E":
                    job.error = obj.get("error")
            proc.wait()
        finally:
            timer.cancel()
        if job.cancel_requested:
            job.status, job.message = "cancelado", "Cancelado"
        elif job.result and proc.returncode == 0:
            job.status, job.progress, job.message = "listo", 1.0, "¡Vídeo listo!"
            # no hace falta guardar el wav ni las imágenes en el servidor
            for p in job.out_dir.glob("*.wav"):
                p.unlink(missing_ok=True)
        else:
            job.status = "error"
            if not job.error:
                job.error = ("El render tardó demasiado y se ha detenido." if proc.returncode in (-9, 9, 1) and
                             time.time() - job.started >= self.max_render_seconds - 1
                             else "No se pudo crear el vídeo.")
                log.warning("worker stderr: %s", "".join(stderr_tail)[-1500:])

    # -- limpieza ----------------------------------------------------------
    def _janitor(self) -> None:
        while True:
            try:
                self.cleanup()
            except Exception:  # noqa: BLE001
                log.exception("limpieza")
            time.sleep(600)

    def cleanup(self) -> None:
        cutoff = time.time() - self.retention
        with self.cv:
            for jid, job in list(self.jobs.items()):
                if job.status not in ("en_cola", "procesando") and (job.finished or job.created) < cutoff:
                    self.jobs.pop(jid, None)
        for d in self.data_dir.iterdir():
            if d.is_dir() and d.stat().st_mtime < cutoff and d.name not in self.jobs:
                shutil.rmtree(d, ignore_errors=True)
