"""Escritura de vídeo: fotogramas en crudo por tubería a ffmpeg (sin guardarlos en RAM/disco)."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def ffmpeg_bin() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("No se encuentra ffmpeg. Instálalo y asegúrate de que está en el PATH.")
    return exe


class FFmpegWriter:
    def __init__(self, out: Path, width: int, height: int, fps: int, audio: Path,
                 music: str | None = None, music_volume: float = 0.12, duration: float | None = None,
                 crf: int = 23, preset: str = "veryfast", threads: int = 0):
        cmd = [ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
               "-i", str(audio)]
        if music:
            cmd += ["-stream_loop", "-1", "-i", str(music)]
            fade_st = max(0.0, (duration or 0) - 2.0)
            cmd += ["-filter_complex",
                    f"[2:a]volume={music_volume:.3f},afade=t=out:st={fade_st:.2f}:d=2[m];"
                    f"[1:a][m]amix=inputs=2:duration=first:normalize=0[a]",
                    "-map", "0:v", "-map", "[a]"]
        else:
            cmd += ["-map", "0:v", "-map", "1:a"]
        cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
                "-tune", "animation", "-g", str(fps * 4)]
        # Limitar hilos: x264 crea ~1,5 hilos por núcleo DEL HOST (también dentro de Docker)
        # y cada hilo retiene fotogramas 1080x1920 -> sin límite puede pasar de 1 GB de RAM.
        threads = threads or min(2, os.cpu_count() or 2)
        cmd += ["-threads", str(threads), "-x264-params", f"threads={threads}:lookahead-threads=1:rc-lookahead=10"]
        # Ojo: NO usar -shortest: su cola de sincronización acumula fotogramas en crudo (cientos de MB).
        # El nº de fotogramas ya coincide con la duración del audio; -t sólo recorta por seguridad.
        if duration:
            cmd += ["-t", f"{duration:.3f}"]
        cmd += ["-c:a", "aac", "-b:a", "160k", "-ar", "44100", "-movflags", "+faststart", str(out)]
        self.cmd = cmd
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self.frames = 0

    def write(self, frame) -> None:
        try:
            self.proc.stdin.write(memoryview(frame).cast("B"))
        except (BrokenPipeError, OSError) as e:
            err = self.proc.stderr.read().decode("utf-8", "replace") if self.proc.stderr else ""
            raise RuntimeError(f"ffmpeg se ha detenido: {err[-800:]}") from e
        self.frames += 1

    def close(self) -> None:
        if self.proc.stdin and not self.proc.stdin.closed:
            self.proc.stdin.close()
        err = self.proc.stderr.read().decode("utf-8", "replace") if self.proc.stderr else ""
        rc = self.proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg terminó con error {rc}: {err[-800:]}")

    def kill(self) -> None:
        try:
            self.proc.kill()
        except Exception:
            pass
