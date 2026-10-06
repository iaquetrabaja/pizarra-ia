# Pizarra IA — imagen para servidor (CPU, sin GPU)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIZARRA_CACHE=/cache \
    PIZARRA_DATA=/data \
    ROOT_PATH=/pizarra \
    TRUST_PROXY=1 \
    RENDER_THREADS=2 \
    RENDERS_POR_DIA=3 \
    MAX_SEGUNDOS=120 \
    RETENCION_HORAS=24

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Modelo Whisper para alinear los subtítulos con la voz, dentro de la imagen
# (no depende de la red ni retrasa la primera petición)
ENV PIZARRA_WHISPER_DIR=/opt/whisper \
    PIZARRA_WHISPER_MODEL=base \
    HF_HUB_DISABLE_TELEMETRY=1
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8', download_root='/opt/whisper')"
ENV HF_HUB_OFFLINE=1

COPY pizarra ./pizarra
COPY LICENSE README.md ./

# Usuario sin privilegios + voz española de Piper descargada en la imagen (~60 MB)
RUN useradd -m -u 1000 app \
 && mkdir -p /data /cache \
 && python -c "from pizarra.tts import load_piper; load_piper('es_ES-davefx-medium')" \
 && chown -R app:app /data /cache
USER app

VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=60s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/api/salud')" || exit 1

# UN solo proceso: la cola de trabajos vive en memoria (un render a la vez).
CMD ["uvicorn", "pizarra.web.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers", "--forwarded-allow-ips", "*"]
