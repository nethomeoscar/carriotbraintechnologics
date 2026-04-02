"""
Pixora — Servidor Unificado
============================
Combina: FSRCNN (mejora de imagen), TTS (texto a audio),
STT (audio a texto), eliminación de fondo y GIF desde vídeo.

Instalación de dependencias:
    pip install flask flask-cors opencv-contrib-python numpy pillow \
                edge-tts whisper PyPDF2 python-docx rembg

Para GIF desde vídeo también necesitas ffmpeg instalado en el sistema.

Uso:
    python server.py

El servidor corre en http://localhost:5000
"""

import asyncio
import base64
import io
import os
import subprocess
import tempfile
import urllib.request
import uuid
from datetime import datetime
import cv2
import numpy as np
import edge_tts
import whisper
from flask import Flask, jsonify, request, send_from_directory, Response
from flask_cors import CORS
from PIL import Image
import requests
from collections import Counter
from sklearn.cluster import KMeans

DEEPSEEK_API_KEY = 'sk-8cd9ef8da0244f79a3759d6d4ad49984'

try:
    from rembg import remove as rembg_remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    print("[Pixora] rembg no disponible — /remove-bg no funcionará.")

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "pixora_secret_key_2025"
CORS(app)

# ─── Directorios ──────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR  = os.path.join(BASE_DIR, "static", "audios")
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
MODELS_DIR = os.path.join(BASE_DIR, "fsrcnn_models")
for d in [AUDIO_DIR, UPLOAD_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

ALLOWED_AUDIO = {"wav", "mp3", "m4a", "ogg", "flac", "webm"}
ALLOWED_DOCS  = {"txt", "pdf", "docx"}

# ─── Whisper ──────────────────────────────────────────────────────────────────
print("[Pixora] Cargando modelo Whisper (tiny)…")
try:
    whisper_model = whisper.load_model("tiny")
    print("[Pixora] Whisper listo.")
except Exception as e:
    whisper_model = None
    print(f"[Pixora] Whisper no disponible: {e}")

# ─── Voces TTS ────────────────────────────────────────────────────────────────
VOCES_DEFAULT = {
    "Español (MX) - Marina":  "es-MX-MarinaNeural",
    "Español (MX) - Gerardo": "es-MX-GerardoNeural",
    "Español (ES) - Álvaro":  "es-ES-AlvaroNeural",
    "Inglés (US) - Guy":      "en-US-GuyNeural",
    "Inglés (US) - Jenny":    "en-US-JennyNeural",
}

async def _fetch_voices_async():
    return await edge_tts.list_voices()

try:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    raw_voices = loop.run_until_complete(_fetch_voices_async())
    loop.close()
    VOCES = {}
    for v in raw_voices:
        locale = v["Locale"]
        short  = v["ShortName"].split("-")
        name   = short[-1] if len(short) >= 3 else v["ShortName"]
        VOCES[f"{locale} - {name}"] = v["ShortName"]
    print(f"[Pixora] {len(VOCES)} voces TTS cargadas.")
except Exception as e:
    VOCES = VOCES_DEFAULT
    print(f"[Pixora] Usando voces TTS de respaldo ({e}).")

# ─── FSRCNN ───────────────────────────────────────────────────────────────────
FSRCNN_MODELS = {
    2: {"filename": "FSRCNN_x2.pb", "url": "https://raw.githubusercontent.com/opencv/opencv_contrib/master/modules/dnn_superres/models/FSRCNN_x2.pb"},
    3: {"filename": "FSRCNN_x3.pb", "url": "https://raw.githubusercontent.com/opencv/opencv_contrib/master/modules/dnn_superres/models/FSRCNN_x3.pb"},
    4: {"filename": "FSRCNN_x4.pb", "url": "https://raw.githubusercontent.com/opencv/opencv_contrib/master/modules/dnn_superres/models/FSRCNN_x4.pb"},
}
_fsrcnn_cache = {}

def download_model(scale):
    info = FSRCNN_MODELS[scale]
    path = os.path.join(MODELS_DIR, info["filename"])
    if not os.path.exists(path):
        print(f"[Pixora] Descargando FSRCNN x{scale}…")
        urllib.request.urlretrieve(info["url"], path)
    return path

def get_fsrcnn(scale):
    if scale not in _fsrcnn_cache:
        from cv2 import dnn_superres
        sr = dnn_superres.DnnSuperResImpl_create()
        sr.readModel(download_model(scale))
        sr.setModel("fsrcnn", scale)
        _fsrcnn_cache[scale] = sr
    return _fsrcnn_cache[scale]

# ─── Utilidades imagen ────────────────────────────────────────────────────────
def b64_to_cv2(b64):
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    arr = np.frombuffer(base64.b64decode(b64), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("No se pudo decodificar la imagen.")
    return img

def cv2_to_b64_png(img):
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("No se pudo codificar la imagen.")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()

def apply_post_process(img, enhance_type, intensity):
    lv = {"subtle": 0.28, "balanced": 0.62, "strong": 1.0}.get(intensity, 0.62)
    if enhance_type in ("auto", "color", "landscape"):
        img = _color_boost(img, lv)
    if enhance_type == "landscape":
        f = img.astype(np.float32)
        f[...,1] = np.clip(f[...,1] + 14*lv, 0, 255)
        f[...,0] = np.clip(f[...,0] + 9*lv,  0, 255)
        img = f.astype(np.uint8)
    if enhance_type in ("lowlight", "portrait"):
        f = img.astype(np.float32)
        f[...,0] = np.clip(f[...,0] + 9*lv,  0, 255)
        f[...,1] = np.clip(f[...,1] + 16*lv, 0, 255)
        f[...,2] = np.clip(f[...,2] + 22*lv, 0, 255)
        img = f.astype(np.uint8)
    if enhance_type == "noise":
        h = max(1, int(3 + 7*lv))
        img = cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)
    if enhance_type in ("auto", "sharpness"):
        blur  = cv2.GaussianBlur(img, (0,0), 2)
        img   = cv2.addWeighted(img, 1 + 0.55*lv, blur, -0.55*lv, 0)
    return img

def _color_boost(img, lv):
    f = img.astype(np.float32)
    f[...,0] = np.clip(f[...,0] + 2*lv, 0, 255)
    f[...,1] = np.clip(f[...,1] + 4*lv, 0, 255)
    f[...,2] = np.clip(f[...,2] + 6*lv, 0, 255)
    hsv = cv2.cvtColor(f.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[...,1] = np.clip(hsv[...,1] * (1 + 0.18*lv), 0, 255)
    f = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    return np.clip((f - 128) * (1 + 0.06*lv) + 128 + 2*lv, 0, 255).astype(np.uint8)

def analyze_image_heuristic(img, enhance_type, intensity):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    mb = float(np.mean(gray))
    sb = float(np.std(gray))
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    noise = float(np.std(gray.astype(np.float32) - blur.astype(np.float32)))
    lap   = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    score = 50
    if 80 < mb < 200: score += 15
    if sb > 40: score += 10
    if lap > 100: score += 10
    if noise < 5: score += 10
    score = min(score, 95)
    after = min(score + {"subtle":5,"balanced":12,"strong":20}.get(intensity,12), 99)
    tags = []
    if mb < 80:   tags.append("Subexpuesta")
    if mb > 200:  tags.append("Sobreexpuesta")
    if lap < 50:  tags.append("Baja nitidez")
    if noise > 8: tags.append("Ruido visible")
    if sb > 60:   tags.append("Alto contraste")
    if not tags:  tags = ["Imagen balanceada"]
    return {"analysis": f"Imagen {w}×{h}px · brillo {mb:.0f} · σ={sb:.0f}",
            "improvements": f"Mejoras de tipo '{enhance_type}' con intensidad {intensity} vía FSRCNN.",
            "tags": tags[:5], "quality_score_before": int(score), "quality_score_after": int(after)}

# ─── Utilidades texto ─────────────────────────────────────────────────────────
def extract_text(filepath, ext):
    if ext == "txt":
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    if ext == "pdf" and PDF_AVAILABLE:
        text = ""
        with open(filepath, "rb") as f:
            for page in PyPDF2.PdfReader(f).pages:
                t = page.extract_text()
                if t: text += t + "\n"
        return text
    if ext == "docx" and DOCX_AVAILABLE:
        doc = DocxDocument(filepath)
        return "\n".join(p.text for p in doc.paragraphs)
    return ""

async def _tts_async(text, voice, rate, path):
    await edge_tts.Communicate(text, voice, rate=rate).save(path)

# ═══════════════════════════════════════════════════════════════════════════════
#  ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "whisper": whisper_model is not None,
        "rembg":   REMBG_AVAILABLE,
        "fsrcnn_scales": [2, 3, 4],
        "tts_voices": len(VOCES),
        "palette": [4, 5, 6, 8, 10],
    })

# ── FSRCNN ────────────────────────────────────────────────────────────────────
@app.route("/enhance", methods=["POST"])
def enhance():
    try:
        body         = request.get_json(force=True)
        enhance_type = body.get("enhance_type", "auto")
        intensity    = body.get("intensity", "balanced")
        scale        = int(body.get("scale", 2))
        if scale not in FSRCNN_MODELS:
            return jsonify({"error": f"Escala {scale} no soportada."}), 400
        img      = b64_to_cv2(body.get("image",""))
        analysis = analyze_image_heuristic(img, enhance_type, intensity)
        sr       = get_fsrcnn(scale)
        img_up   = sr.upsample(img)
        img_out  = apply_post_process(img_up, enhance_type, intensity)
        return jsonify({"enhanced_image": cv2_to_b64_png(img_out), "analysis": analysis, "scale": scale})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Eliminar fondo ────────────────────────────────────────────────────────────
@app.route("/remove-bg", methods=["POST"])
def remove_bg():
    if not REMBG_AVAILABLE:
        return jsonify({"error": "rembg no instalado. Ejecuta: pip install rembg"}), 503
    try:
        body = request.get_json(force=True)
        b64  = body.get("image", "")
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        img_bytes = base64.b64decode(b64)
        result    = rembg_remove(img_bytes)
        out_b64   = "data:image/png;base64," + base64.b64encode(result).decode()
        # Análisis básico
        pil = Image.open(io.BytesIO(result))
        w, h = pil.size
        analysis = {
            "subject":       "Detectado automáticamente",
            "mask_quality":  "Alta (U2Net)",
            "removed_area":  "Fondo eliminado correctamente",
            "tags":          ["Sin fondo", "PNG transparente", f"{w}×{h}px"],
        }
        return jsonify({"result_image": out_b64, "analysis": analysis})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── TTS ───────────────────────────────────────────────────────────────────────
@app.route("/tts/voices")
def tts_voices():
    return jsonify({"voices": VOCES})

@app.route("/tts/convert", methods=["POST"])
def tts_convert():
    try:
        texto    = request.form.get("texto", "").strip()
        voz      = request.form.get("voz", "es-MX-MarinaNeural")
        vel_raw  = request.form.get("velocidad", "+0%")

        if not texto:
            return jsonify({"error": "El texto no puede estar vacío."}), 400
        if voz not in VOCES.values():
            return jsonify({"error": f"Voz '{voz}' no válida."}), 400

        filename = f"{uuid.uuid4()}.mp3"
        path     = os.path.join(AUDIO_DIR, filename)
        loop     = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_tts_async(texto, voz, vel_raw, path))
        finally:
            loop.close()

        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return jsonify({"error": "El audio generado está vacío."}), 500

        audio_url = f"/static/audios/{filename}"
        return jsonify({"url": audio_url, "filename": filename})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/tts/upload", methods=["POST"])
def tts_upload():
    if "file" not in request.files:
        return jsonify({"error": "No se envió ningún archivo."}), 400
    f   = request.files["file"]
    ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
    if ext not in ALLOWED_DOCS:
        return jsonify({"error": "Solo se aceptan txt, pdf, docx."}), 400
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        f.save(tmp.name)
        path = tmp.name
    try:
        text = extract_text(path, ext)
        if not text.strip():
            return jsonify({"error": "No se pudo extraer texto del archivo."}), 400
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(path):
            os.remove(path)

# ── STT ───────────────────────────────────────────────────────────────────────
@app.route("/stt/transcribe", methods=["POST"])
def stt_transcribe():
    if whisper_model is None:
        return jsonify({"error": "Whisper no está disponible."}), 503
    if "audio" not in request.files:
        return jsonify({"error": "No se envió archivo de audio."}), 400
    audio_file = request.files["audio"]
    idioma = request.form.get("idioma", "es")
    ext    = audio_file.filename.rsplit(".", 1)[-1].lower() if "." in audio_file.filename else "webm"
    fname  = f"{uuid.uuid4()}.{ext}"
    fpath  = os.path.join(UPLOAD_DIR, fname)
    audio_file.save(fpath)
    try:
        result = whisper_model.transcribe(
            fpath,
            language=idioma if idioma != "auto" else None,
            task="transcribe",
            fp16=False,
        )
        return jsonify({
            "success": True,
            "texto":   result["text"].strip(),
            "idioma":  result.get("language", idioma),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(fpath):
            os.remove(fpath)

@app.route("/stt/download", methods=["POST"])
def stt_download():
    texto  = request.json.get("texto", "")
    if not texto:
        return jsonify({"error": "Sin texto."}), 400
    nombre = f"transcripcion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    return Response(texto, mimetype="text/plain",
                    headers={"Content-Disposition": f'attachment; filename="{nombre}"'})

# ── GIF desde vídeo ───────────────────────────────────────────────────────────
@app.route("/video-to-gif", methods=["POST"])
def video_to_gif():
    try:
        body  = request.get_json(force=True)
        b64v  = body.get("video", "")
        start = float(body.get("start_time", 0))
        end   = float(body.get("end_time", 5))
        fps   = int(body.get("fps", 12))
        width = int(body.get("width", 480))

        if "," in b64v:
            b64v = b64v.split(",", 1)[1]
        video_bytes = base64.b64decode(b64v)

        with tempfile.TemporaryDirectory() as tmpdir:
            in_path  = os.path.join(tmpdir, "input.mp4")
            pal_path = os.path.join(tmpdir, "palette.png")
            out_path = os.path.join(tmpdir, "output.gif")

            with open(in_path, "wb") as f:
                f.write(video_bytes)

            duration = end - start
            scale_filter = f"fps={fps},scale={width if width>0 else -1}:-1:flags=lanczos"

            # Generar paleta
            subprocess.run([
                "ffmpeg", "-y", "-ss", str(start), "-t", str(duration),
                "-i", in_path, "-vf", f"{scale_filter},palettegen",
                pal_path
            ], check=True, capture_output=True)

            # Generar GIF
            subprocess.run([
                "ffmpeg", "-y", "-ss", str(start), "-t", str(duration),
                "-i", in_path, "-i", pal_path,
                "-lavfi", f"{scale_filter} [x]; [x][1:v] paletteuse",
                out_path
            ], check=True, capture_output=True)

            with open(out_path, "rb") as f:
                gif_b64 = "data:image/gif;base64," + base64.b64encode(f.read()).decode()

            size = os.path.getsize(out_path)

        # Obtener dimensiones aproximadas
        return jsonify({
            "gif": gif_b64,
            "meta": {
                "file_size":    size,
                "frame_count":  int(duration * fps),
                "colors":       256,
                "compression_ratio": "~auto",
            }
        })
    except subprocess.CalledProcessError as e:
        return jsonify({"error": "ffmpeg falló. ¿Está instalado? " + (e.stderr.decode() if e.stderr else "")}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Generar paleta de colores ─────────────────────────────────────────────────
def extract_dominant_colors(img_b64, count=5):
    """Extrae los colores más representativos de la imagen usando K-Means."""
    try:
        # Limpiar prefijo data URI si existe (ej: "data:image/png;base64,...")
        if "," in img_b64:
            img_b64 = img_b64.split(",", 1)[1]

        img_data = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("No se pudo decodificar la imagen.")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Redimensionar para procesar rápido
        img = cv2.resize(img, (150, 150), interpolation=cv2.INTER_AREA)
        pixels = img.reshape(-1, 3).astype(np.float32)

        # Filtrar píxeles muy claros (blancos) y muy oscuros (negros)
        mask = (
            (pixels[:, 0] > 20) & (pixels[:, 1] > 20) & (pixels[:, 2] > 20) &
            (pixels[:, 0] < 235) & (pixels[:, 1] < 235) & (pixels[:, 2] < 235)
        )
        filtered = pixels[mask]
        if len(filtered) < count * 10:
            filtered = pixels  # usar todos si no hay suficientes

        # K-Means para agrupar colores similares
        kmeans = KMeans(n_clusters=count, n_init=10, random_state=42)
        kmeans.fit(filtered)
        colors = kmeans.cluster_centers_.astype(int)

        # Ordenar por frecuencia (tamaño de cada cluster)
        labels = kmeans.labels_
        counts = Counter(labels)
        sorted_colors = [colors[i] for i, _ in sorted(counts.items(), key=lambda x: -x[1])]

        hex_colors = [f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}" for c in sorted_colors]
        return hex_colors
    except Exception as e:
        print(f"Error extrayendo colores: {e}")
        return []


def generate_fallback_palette(mode, prompt_text="", count=5, style="any"):
    """Genera una paleta de fallback cuando la IA no está disponible."""
    palettes = {
        "pastel":  ["#FADADD","#C9E4DE","#F9F3C9","#C5E0B4","#FBC8B5","#D4C5E2","#FAF0CA","#B5EAD7"],
        "vivid":   ["#FF6B6B","#4ECDC4","#FFE66D","#1A535C","#F7B05E","#6A0572","#3D405B","#81B29A"],
        "dark":    ["#1A1A2E","#16213E","#0F3460","#E94560","#533483","#2B2D42","#8D99AE","#EF233C"],
        "earth":   ["#8D6B54","#C2A575","#E6D5B8","#5C4033","#A77B55","#D4A373","#CCD5AE","#E9EDC9"],
        "ocean":   ["#006D77","#83C5BE","#FFDDD2","#EDF6F9","#E29578","#0077B6","#00B4D8","#ADE8F4"],
        "mono":    ["#2D2D2D","#5A5A5A","#8A8A8A","#B8B8B8","#E5E5E5","#4A4A4A","#6E6E6E","#C0C0C0"],
        "neon":    ["#FF00CC","#00FF99","#FF9900","#00CCFF","#FF0066","#9B5DE5","#F15BB5","#FEE440"],
        "nordic":  ["#3B4252","#4C566A","#D8DEE9","#E5E9F0","#A3BE8C","#ECEFF4","#88C0D0","#81A1C1"],
        "any":     ["#E63946","#F1FA8C","#8AC6D1","#C7B9FF","#A7C5BD","#F4A261","#2A9D8F","#E76F51"],
    }
    roles = ["primario","secundario","acento","neutro","complementario","apoyo","énfasis","fondo"]
    base = palettes.get(style, palettes["any"])
    colors = []
    for i in range(min(count, len(base))):
        hex_val = base[i].upper()
        r = int(hex_val[1:3], 16)
        g = int(hex_val[3:5], 16)
        b = int(hex_val[5:7], 16)
        colors.append({
            "hex": hex_val,
            "name": f"Color {i+1}",
            "role": roles[i] if i < len(roles) else "neutro",
            "rgb": f"rgb({r},{g},{b})"
        })
    return {
        "colors": colors,
        "palette_name": f"Paleta {style.capitalize()}",
        "description": f"Paleta generada localmente con estilo '{style}'."
    }


@app.route("/palette", methods=["POST"])
def generate_palette():
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "JSON inválido"}), 400

    mode        = data.get("mode", "text")        # "image", "text" o "random"
    prompt_text = data.get("prompt", "").strip()
    img_b64     = data.get("image", "")
    count       = max(1, min(int(data.get("count", 5)), 12))
    harmony     = data.get("harmony", "auto")
    application = data.get("application", "web")
    style       = data.get("style", "any")        # para modo random

    # ── Construir prompt para la IA ──────────────────────────────────────────
    if mode == "image" and img_b64:
        dominant = extract_dominant_colors(img_b64, count)
        if not dominant:
            return jsonify({"error": "No se pudieron extraer colores de la imagen."}), 400

        dominant_str = ", ".join(dominant)
        prompt_ia = (
            f"Analicé una imagen y extraje estos {len(dominant)} colores dominantes reales (en orden de importancia): {dominant_str}.\n"
            f"Crea una paleta profesional de EXACTAMENTE {count} colores para uso en '{application}'.\n"
            f"DEBES incluir estos colores dominantes como base principal, puedes ajustar ligeramente el tono si mejora la armonía '{harmony}'.\n"
            f"Para cada color indica: hex exacto, nombre descriptivo en español y su rol en la paleta (primario, secundario, acento, neutro, etc).\n"
            f"El nombre de la paleta debe reflejar los colores y el mood de la imagen."
        )
    elif mode == "random":
        prompt_ia = (
            f"Genera una paleta de colores aleatoria de EXACTAMENTE {count} colores con estilo '{style}' "
            f"para uso en '{application}' con armonía '{harmony}'.\n"
            f"Para cada color indica: hex exacto, nombre descriptivo en español y su rol en la paleta."
        )
    else:  # text
        if not prompt_text:
            return jsonify({"error": "Escribe una descripción para la paleta."}), 400
        prompt_ia = (
            f"Crea una paleta de colores profesional de EXACTAMENTE {count} colores basada en: '{prompt_text}'.\n"
            f"La paleta es para '{application}' y debe tener armonía '{harmony}'.\n"
            f"Los colores deben capturar visualmente la esencia de la descripción.\n"
            f"Para cada color indica: hex exacto, nombre descriptivo en español y su rol (primario, secundario, acento, neutro, etc)."
        )

    # ── Llamada a DeepSeek ───────────────────────────────────────────────────
    try:
        resp = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek-chat",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Eres un experto en diseño, teoría del color y branding. "
                            "Responde ÚNICAMENTE con un objeto JSON válido, sin texto adicional ni bloques de código markdown. "
                            "Estructura exacta requerida:\n"
                            '{"colors": [{"hex": "#RRGGBB", "name": "nombre en español", "role": "rol"}], '
                            '"palette_name": "nombre creativo", "description": "descripción técnica breve"}'
                        )
                    },
                    {"role": "user", "content": prompt_ia}
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.7,
                "max_tokens": 1000
            },
            timeout=20
        )

        if resp.status_code != 200:
            print(f"[DeepSeek] Error {resp.status_code}: {resp.text}")
            raise Exception(f"API respondió con código {resp.status_code}")

        raw_content = resp.json()["choices"][0]["message"]["content"]

        # Parsear el contenido: puede llegar como string JSON o ya como dict
        import json as _json
        if isinstance(raw_content, str):
            # Limpiar posibles bloques markdown
            clean = raw_content.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            palette_data = _json.loads(clean.strip())
        else:
            palette_data = raw_content

        # Validar estructura mínima
        if "colors" not in palette_data or not palette_data["colors"]:
            raise ValueError("Respuesta de IA no contiene colores válidos.")

        # Asegurar que hex tenga formato correcto
        for color in palette_data["colors"]:
            h = color.get("hex", "#000000").strip()
            if not h.startswith("#"):
                h = "#" + h
            color["hex"] = h.upper()
            # Añadir rgb calculado
            r = int(h[1:3], 16) if len(h) >= 7 else 0
            g = int(h[3:5], 16) if len(h) >= 7 else 0
            b = int(h[5:7], 16) if len(h) >= 7 else 0
            color["rgb"] = f"rgb({r},{g},{b})"

        # Si modo imagen, agregar los dominantes extraídos como referencia
        if mode == "image":
            palette_data["dominant_extracted"] = dominant

        return jsonify(palette_data)

    except Exception as e:
        print(f"[Palette] Fallo IA ({e}), usando fallback local.")
        fallback = generate_fallback_palette(mode, prompt_text, count, style)
        if mode == "image" and dominant:
            fallback["dominant_extracted"] = dominant
            # Usar los colores dominantes reales en el fallback
            for i, hex_val in enumerate(dominant[:count]):
                if i < len(fallback["colors"]):
                    fallback["colors"][i]["hex"] = hex_val.upper()
        fallback["fallback"] = True
        return jsonify(fallback)

# ── Archivos estáticos ────────────────────────────────────────────────────────
@app.route("/static/audios/<path:filename>")
def serve_audio(filename):
    return send_from_directory(AUDIO_DIR, filename)

# ─── Arranque ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Pixora — Servidor Unificado")
    print("  http://localhost:5000")
    print("  Endpoints disponibles:")
    print("    GET  /health")
    print("    POST /enhance          (FSRCNN)")
    print("    POST /remove-bg        (rembg)")
    print("    POST /tts/convert      (Edge TTS)")
    print("    POST /tts/upload       (extrae texto de doc)")
    print("    GET  /tts/voices       (lista de voces)")
    print("    POST /stt/transcribe   (Whisper)")
    print("    POST /stt/download     (descarga txt)")
    print("    POST /video-to-gif     (ffmpeg)")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False)
