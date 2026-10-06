"""Configuración general: formatos, estilos, rutas y opciones de render."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
ASSETS_DIR = PKG_DIR / "assets"
FONTS_DIR = ASSETS_DIR / "fonts"

# Directorio de caché (voces Piper, etc.). Configurable con PIZARRA_CACHE.
CACHE_DIR = Path(os.environ.get("PIZARRA_CACHE", Path.home() / ".cache" / "pizarra-ia"))

FORMATS = {
    # nombre: (ancho, alto, relación de aspecto pedida al modelo de imagen,
    #          fracción inferior reservada para subtítulos)
    # (la franja de subtítulos de 9:16 queda por encima de la zona que tapan
    #  los botones/descripción de TikTok, Reels y Shorts)
    "9:16": (1080, 1920, "3:4", 0.30),
    "16:9": (1920, 1080, "16:9", 0.17),
}

STYLES = {
    "pizarra": "Pizarra blanca",
    "tiza": "Pizarra de tiza",
    "cuaderno": "Cuaderno",
}

LANGUAGES = {
    "es": "español de España",
    "es-419": "español latinoamericano",
    "en": "English",
    "pt": "português",
    "fr": "français",
    "it": "italiano",
    "de": "Deutsch",
}

# Voces Piper (fallback local) por idioma. Se descargan la primera vez.
PIPER_VOICES = {
    "es": "es_ES-davefx-medium",
    "es-419": "es_MX-claude-high",
    "en": "en_US-lessac-medium",
    "pt": "pt_BR-faber-medium",
    "fr": "fr_FR-siwis-medium",
    "it": "it_IT-paola-medium",
    "de": "de_DE-thorsten-medium",
}

# Voces precompiladas de Gemini TTS (multilingües) con su carácter según Google
# (https://ai.google.dev/gemini-api/docs/speech-generation#voices), traducido.
GEMINI_VOICE_INFO = {
    "Zephyr": "brillante",
    "Puck": "animada",
    "Charon": "informativa",
    "Kore": "firme",
    "Fenrir": "enérgica",
    "Leda": "juvenil",
    "Orus": "firme",
    "Aoede": "desenfadada",
    "Callirrhoe": "tranquila",
    "Autonoe": "brillante",
    "Enceladus": "suave, aspirada",
    "Iapetus": "clara",
    "Umbriel": "tranquila",
    "Algieba": "suave",
    "Despina": "suave",
    "Erinome": "clara",
    "Algenib": "grave, rasgada",
    "Rasalgethi": "informativa",
    "Laomedeia": "animada",
    "Achernar": "suave",
    "Alnilam": "firme",
    "Schedar": "equilibrada",
    "Gacrux": "madura",
    "Pulcherrima": "directa",
    "Achird": "cercana",
    "Zubenelgenubi": "informal",
    "Vindemiatrix": "amable",
    "Sadachbia": "viva",
    "Sadaltager": "experta",
    "Sulafat": "cálida",
}
GEMINI_VOICES = list(GEMINI_VOICE_INFO)
DEFAULT_VOICE = "Charon"

# Frase corta para probar una voz antes de generar el vídeo.
VOICE_SAMPLE = {
    "es": "Hola, soy la voz {nombre}. Así sonará la narración de tu vídeo.",
    "es-419": "Hola, soy la voz {nombre}. Así sonará la narración de tu video.",
    "en": "Hi, I'm the voice {nombre}. This is how your video's narration will sound.",
    "pt": "Olá, eu sou a voz {nombre}. É assim que vai soar a narração do seu vídeo.",
    "fr": "Bonjour, je suis la voix {nombre}. Voici comment sonnera la narration de ta vidéo.",
    "it": "Ciao, sono la voce {nombre}. Così suonerà la narrazione del tuo video.",
    "de": "Hallo, ich bin die Stimme {nombre}. So wird die Erzählung deines Videos klingen.",
}


def voice_sample(nombre: str, idioma: str = "es") -> str:
    return VOICE_SAMPLE.get(idioma, VOICE_SAMPLE["es"]).format(nombre=nombre)


def voice_label(voz: str) -> str:
    """'Charon — informativa' (para los desplegables)."""
    info = GEMINI_VOICE_INFO.get(voz)
    return f"{voz} — {info}" if info else voz


MAX_SECONDS_DEFAULT = 120


@dataclass
class RenderOptions:
    formato: str = "9:16"
    estilo: str = "pizarra"
    idioma: str = "es"
    fps: int = 24
    tts: str = "auto"            # auto | gemini | piper
    voz: str = DEFAULT_VOICE      # voz Gemini
    velocidad: float = 1.0       # >1 más rápido (sólo Piper)
    imagenes: str = "auto"       # auto | ia | vector
    color: bool = True           # relleno de color al final de cada escena
    subtitulos: bool = True
    musica: str | None = None
    volumen_musica: float = 0.12
    modelo_texto: str | None = None
    modelo_imagen: str | None = None
    modelo_tts: str | None = None
    max_segundos: float | None = None
    crf: int = 23
    preset: str = "veryfast"
    threads: int = 0             # 0 = automático

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RenderOptions":
        names = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in (d or {}).items() if k in names})

    def validate(self) -> None:
        if self.formato not in FORMATS:
            raise ValueError(f"Formato no válido: {self.formato} (usa 9:16 o 16:9)")
        if self.estilo not in STYLES:
            raise ValueError(f"Estilo no válido: {self.estilo} ({', '.join(STYLES)})")
        if self.tts not in ("auto", "gemini", "piper"):
            raise ValueError("tts debe ser auto, gemini o piper")
        if self.imagenes not in ("auto", "ia", "vector"):
            raise ValueError("imagenes debe ser auto, ia o vector")
        self.fps = int(max(12, min(30, self.fps)))


@dataclass
class Layout:
    width: int
    height: int
    image_aspect: str
    area: tuple[int, int, int, int]      # x, y, w, h de la zona de dibujo
    sub_box: tuple[int, int, int, int]   # x, y, w, h de la zona de subtítulos

    @classmethod
    def for_format(cls, formato: str) -> "Layout":
        w, h, aspect, band = FORMATS[formato]
        band_h = int(h * band)
        margin = int(min(w, h) * 0.04)
        area = (margin, margin, w - 2 * margin, h - band_h - margin)
        sub_h = int(band_h * (0.55 if h > w else 0.9))
        sub_box = (int(w * 0.06), h - band_h, int(w * 0.88), sub_h)
        return cls(w, h, aspect, area, sub_box)

    @property
    def aspect_float(self) -> float:
        a, b = self.image_aspect.split(":")
        return float(a) / float(b)
