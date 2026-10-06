"""Orquestación: guion -> imágenes -> voz -> subtítulos -> vídeo."""
from __future__ import annotations

import json
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .config import LANGUAGES, Layout, RenderOptions
from .demo import demo_plan
from .draw import Hand, SceneAnimator, plan_strokes, restore
from .gemini import Gemini, GeminiError, pick_models
from .images import image_prompt, placeholder_image, vector_image
from .script import estimate_seconds, generate_plan, normalize_plan
from .style import build_layers, make_board
from .subtitles import SubtitleRenderer, cues_for_spans, cues_for_words, to_srt
from .tts import Speech, gemini_speech, narration_speeches, synth_piper, write_wav
from .video import FFmpegWriter

log = logging.getLogger("pizarra")

Progress = Callable[[str, float, str], None]


class Cancelled(Exception):
    pass


def _noop(stage: str, frac: float, msg: str) -> None:
    pass


@dataclass
class RenderResult:
    video: Path
    srt: Path
    plan: Path
    duration: float
    seconds: dict = field(default_factory=dict)
    api_calls: dict = field(default_factory=dict)
    engines: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"video": str(self.video), "srt": str(self.srt), "plan": str(self.plan),
                "duration": round(self.duration, 2), "seconds": self.seconds,
                "api_calls": self.api_calls, "engines": self.engines, "warnings": self.warnings}


def load_scene_image(folder: str | Path, i: int) -> np.ndarray:
    path = Path(folder) / f"escena_{i + 1:02d}.png"
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"No encuentro la imagen {path}")
    return img


def resolve_models(gem: Gemini, opts: RenderOptions) -> dict:
    models = gem.list_models()
    d = pick_models(models)
    return {"texto": opts.modelo_texto or d["texto"],
            "imagen": opts.modelo_imagen or d["imagen"],
            "tts": opts.modelo_tts or d["tts"]}


# ----------------------------------------------------------------------------
# Paso 1: guion
# ----------------------------------------------------------------------------
def make_plan(tema: str, opts: RenderOptions, api_key: str | None = None, duracion: float = 60,
              investigar: bool = False, offline: bool = False, progress: Progress = _noop) -> dict:
    if offline or not api_key:
        if not offline:
            raise GeminiError("Necesitas una clave de Gemini (o usa el modo --offline).", "auth")
        plan = demo_plan()
        plan["idioma"] = "es"
        return plan
    gem = Gemini(api_key)
    try:
        models = resolve_models(gem, opts)
        if not models["texto"]:
            raise GeminiError("Tu clave no tiene ningún modelo de texto Gemini disponible.")
        plan = generate_plan(gem, models["texto"], tema, opts.idioma, duracion, investigar,
                             log=lambda m: progress("guion", 0.5, m))
        plan["_llamadas_api"] = dict(gem.calls)
        return plan
    finally:
        gem.close()


# ----------------------------------------------------------------------------
# Paso 2: render
# ----------------------------------------------------------------------------
def render_video(plan: dict, opts: RenderOptions, out_dir: str | Path, api_key: str | None = None,
                 offline: bool = False, progress: Progress = _noop,
                 cancel: Callable[[], bool] = lambda: False, narracion: str | Path | None = None,
                 imagenes_dir: str | Path | None = None, alinear: bool = True) -> RenderResult:
    """Render completo. Opcionales (CLI): ``narracion`` = audio ya grabado en lugar de TTS;
    ``imagenes_dir`` = carpeta con escena_01.png, escena_02.png... en lugar de generarlas;
    ``alinear`` = subtítulos con tiempos reales por palabra (faster-whisper)."""
    t_start = time.time()
    opts.validate()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plan = normalize_plan(plan, opts.idioma or plan.get("idioma"))
    opts.idioma = plan["idioma"]
    layout = Layout.for_format(opts.formato)
    aspect = layout.aspect_float
    warnings: list[str] = []
    timings: dict[str, float] = {}
    engines: dict[str, str] = {}

    def check():
        if cancel():
            raise Cancelled("Cancelado")

    if opts.max_segundos and estimate_seconds(plan) > opts.max_segundos * 1.15:
        raise ValueError(f"El guion es demasiado largo (≈{estimate_seconds(plan):.0f} s). "
                         f"Máximo: {opts.max_segundos:.0f} s. Acorta la narración.")

    gem = None
    models = {"texto": None, "imagen": None, "tts": None}
    if api_key and not offline:
        gem = Gemini(api_key)
        progress("modelos", 0.01, "Comprobando modelos disponibles…")
        models = resolve_models(gem, opts)
    scenes = plan["escenas"]
    n = len(scenes)

    try:
        # ---------------- imágenes -------------------------------------------
        t0 = time.time()
        mode = opts.imagenes
        if not gem:
            mode = "demo"
        elif mode == "auto":
            mode = "ia" if models["imagen"] else "vector"
        elif mode == "ia" and not models["imagen"]:
            warnings.append("Tu clave no tiene modelo de imagen; uso dibujo vectorial.")
            mode = "vector"
        state = {"mode": mode}
        images: list[np.ndarray | None] = [None] * n
        done = [0]

        def make_image(i: int) -> np.ndarray:
            sc = scenes[i]
            if state["mode"] == "ia":
                try:
                    return gem.image(models["imagen"], image_prompt(sc, layout.image_aspect, plan["titulo"]),
                                     layout.image_aspect)
                except GeminiError as e:
                    if opts.imagenes != "auto":
                        raise
                    if e.kind in ("quota", "permission", "unavailable"):
                        if state["mode"] == "ia":
                            warnings.append(f"{e} Paso a dibujo vectorial generado por el modelo de texto.")
                        state["mode"] = "vector"
                    else:
                        warnings.append(f"Escena {i + 1}: {e} Uso dibujo vectorial.")
            if gem and models["texto"]:
                try:
                    return vector_image(gem, models["texto"], sc, aspect, seed=i)
                except Exception as e:  # noqa: BLE001
                    warnings.append(f"Escena {i + 1}: no se pudo generar el dibujo vectorial ({e}).")
            return placeholder_image(sc, aspect, seed=i)

        def job(i):
            check()
            img = load_scene_image(imagenes_dir, i) if imagenes_dir else make_image(i)
            cv2.imwrite(str(out / f"escena_{i + 1:02d}.png"), img)
            images[i] = img
            done[0] += 1
            progress("imagenes", 0.02 + 0.33 * done[0] / n, f"Dibujos {done[0]}/{n}")

        progress("imagenes", 0.02, "Creando los dibujos de cada escena…")
        job(0)  # la primera sola: así detectamos pronto si falta cuota de imagen
        workers = 3 if state["mode"] in ("ia", "vector") else 1
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(job, range(1, n)))
        engines["imagenes"] = {"ia": f"gemini:{models['imagen']}", "vector": f"vector:{models['texto']}",
                               "demo": "local"}.get(state["mode"], state["mode"])
        if mode == "ia" and state["mode"] != "ia":
            engines["imagenes"] = f"mixto (ia→vector:{models['texto']})"
        if imagenes_dir:
            engines["imagenes"] = f"archivos:{Path(imagenes_dir).name}"
        timings["imagenes"] = round(time.time() - t0, 1)

        # ---------------- voz ------------------------------------------------
        check()
        t0 = time.time()
        speeches: list[Speech] = []
        if narracion:
            progress("voz", 0.36, "Usando la narración grabada…")
            speeches = narration_speeches(narracion, [sc["narracion"] for sc in scenes], opts.idioma, opts.fps,
                                          log=lambda m: progress("voz", 0.4, m))
            engines["voz"] = f"audio:{Path(narracion).name}"
        else:
            tts = opts.tts
            if tts == "auto":
                tts = "gemini" if (gem and models["tts"]) else "piper"
            if tts == "gemini" and not (gem and models["tts"]):
                warnings.append("No hay modelo TTS de Gemini disponible; uso Piper (local).")
                tts = "piper"
            if tts == "gemini":
                try:
                    for i, sc in enumerate(scenes):
                        check()
                        progress("voz", 0.36 + 0.14 * i / n, f"Voz {i + 1}/{n} (Gemini)")
                        speeches.append(gemini_speech(gem, models["tts"], sc["narracion"], opts.voz))
                    engines["voz"] = f"gemini:{models['tts']} ({opts.voz})"
                except GeminiError as e:
                    warnings.append(f"{e} Uso la voz local Piper para todo el vídeo.")
                    speeches = []
                    tts = "piper"
            if tts == "piper":
                for i, sc in enumerate(scenes):
                    check()
                    progress("voz", 0.36 + 0.14 * i / n, f"Voz {i + 1}/{n} (Piper)")
                    speeches.append(synth_piper(sc["narracion"], opts.idioma, opts.velocidad))
                from .tts import piper_voice_name, unload_piper
                engines["voz"] = f"piper:{piper_voice_name(opts.idioma)}"
                unload_piper()
        timings["voz"] = round(time.time() - t0, 1)

        sr = speeches[0].sr
        audio = np.concatenate([s.samples for s in speeches])
        total = len(audio) / sr
        if opts.max_segundos and total > opts.max_segundos + 5:
            raise ValueError(f"La narración dura {total:.0f} s y el máximo es {opts.max_segundos:.0f} s. "
                             "Acorta el guion.")
        wav = out / "narracion.wav"
        write_wav(wav, audio, sr)

        # ---------------- subtítulos ----------------------------------------
        # Tiempos reales por palabra: ASR local (faster-whisper) alineado con el guion. Si no se
        # puede, los de siempre (por frase, estimados por longitud y pausas).
        t0 = time.time()
        subs = SubtitleRenderer(layout)
        cues = []
        bounds, ranges = [], []
        t = 0.0
        for sp in speeches:
            bounds.append((t, t + sp.duration))
            ranges.append((t + sp.spans[0][1], t + sp.spans[-1][2]) if sp.spans else (t, t + sp.duration))
            t += sp.duration
        aligned = None
        if alinear:
            from .align import align_script
            progress("subtitulos", 0.5, "Sincronizando los subtítulos con la voz…")
            aligned = align_script([sc["narracion"] for sc in scenes], ranges, audio, sr, opts.idioma,
                                   log=lambda m: (log.info(m), progress("subtitulos", 0.5, m)))
        if aligned:
            for ws in aligned:
                cues += cues_for_words(ws, subs.font, subs.max_width)
            engines["subtitulos"] = "asr"
        else:
            for sp, (b0, _b1) in zip(speeches, bounds):
                cues += cues_for_spans(sp.spans, subs.font, subs.max_width, offset=b0)
            engines["subtitulos"] = "estimado"
        timings["subtitulos"] = round(time.time() - t0, 1)
        srt_path = out / "subtitulos.srt"
        srt_path.write_text(to_srt(cues), encoding="utf-8")

        # ---------------- vídeo ---------------------------------------------
        check()
        t0 = time.time()
        fps = opts.fps
        W, H = layout.width, layout.height
        hand = Hand(int(0.48 * min(W, H)))
        board = make_board(layout, opts.estilo)
        video_path = out / "video.mp4"
        writer = FFmpegWriter(video_path, W, H, fps, wav, music=opts.musica, music_volume=opts.volumen_musica,
                              duration=total, crf=opts.crf, preset=opts.preset, threads=opts.threads)
        total_frames = int(round(total * fps))
        cue_i = 0
        frame_no = 0
        prev_final = None
        xfade = max(1, int(0.25 * fps))
        try:
            with ThreadPoolExecutor(max_workers=1) as prep:
                def prepare(i):
                    L = build_layers(images[i], layout, opts.estilo, board, opts.color)
                    return L, plan_strokes(L.ink)
                fut = prep.submit(prepare, 0)
                for i in range(n):
                    layers, strokes = fut.result()
                    if i + 1 < n:  # preparar la siguiente escena mientras se codifica ésta
                        fut = prep.submit(prepare, i + 1)
                    images[i] = None
                    end = total_frames if i == n - 1 else int(round(bounds[i][1] * fps))
                    nf = end - frame_no
                    anim = SceneAnimator(layers, strokes, hand, nf, fps)
                    for f, (frame, pos) in enumerate(anim.frames()):
                        if prev_final is not None and f < xfade:
                            a = (f + 1) / (xfade + 1)
                            frame = cv2.addWeighted(prev_final, 1 - a, frame, a, 0)
                        b1 = hand.blit(frame, *pos) if pos else None
                        tsec = (frame_no + 0.5) / fps
                        while cue_i < len(cues) and cues[cue_i].end <= tsec:
                            cue_i += 1
                        b2 = None
                        if opts.subtitulos and cue_i < len(cues) and cues[cue_i].start <= tsec:
                            b2 = subs.blit(frame, cue_i, cues[cue_i].text)
                        writer.write(frame)
                        restore(frame, b2)
                        restore(frame, b1)
                        frame_no += 1
                        if frame_no % fps == 0:
                            check()
                            progress("video", 0.5 + 0.5 * frame_no / max(1, total_frames),
                                     f"Animando escena {i + 1}/{n}")
                    prev_final = layers.final
            writer.close()
        except BaseException:
            writer.kill()
            raise
        timings["video"] = round(time.time() - t0, 1)
        timings["total"] = round(time.time() - t_start, 1)

        plan_path = out / "guion.json"
        plan_out = {k: v for k, v in plan.items() if not k.startswith("_")}
        plan_path.write_text(json.dumps(plan_out, ensure_ascii=False, indent=2), encoding="utf-8")
        progress("listo", 1.0, "¡Vídeo listo!")
        return RenderResult(video_path, srt_path, plan_path, total, timings,
                            dict(gem.calls) if gem else {}, engines, warnings)
    finally:
        if gem:
            gem.close()
