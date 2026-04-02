# Pixora — Suite de Herramientas IA

## Estructura del proyecto

```
pixora_project/
├── server.py              ← Servidor Flask unificado (único backend)
├── requirements.txt       ← Dependencias Python
├── index.html             ← Portal principal
├── pixora.html            ← Mejorar calidad de imágenes (FSRCNN)
├── tts.html               ← Texto a Audio (Edge TTS)
├── stt.html               ← Audio a Texto (Whisper)
├── paleta-colores.html    ← Generador de paleta de colores
├── eliminar-fondo.html    ← Eliminar fondo (rembg / U2Net)
├── generar-qr.html        ← Generador de código QR (sin servidor)
├── gif-desde-video.html   ← Crear GIF desde vídeo (ffmpeg)
├── static/
│   ├── audios/            ← Audios TTS generados
│   └── uploads/           ← Archivos temporales STT
└── fsrcnn_models/         ← Modelos FSRCNN (se descargan automáticamente)
```

## Instalación

### 1. Dependencias Python

```bash
pip install -r requirements.txt
```

> ⚠️ Usar `opencv-contrib-python` (NO `opencv-python`) para tener el módulo `dnn_superres`.

### 2. ffmpeg (para GIF desde vídeo)

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg

# Windows
# Descarga desde https://ffmpeg.org/download.html y agrégalo al PATH
```

### 3. Iniciar el servidor

```bash
python server.py
```

El servidor corre en **http://localhost:5000**

### 4. Abrir el portal

Abre `index.html` en tu navegador. Todos los HTMLs se comunican con el servidor en `localhost:5000`.

---

## Endpoints del servidor

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/health` | Estado del servidor |
| POST | `/enhance` | Mejora de imagen con FSRCNN |
| POST | `/remove-bg` | Eliminación de fondo con rembg |
| GET | `/tts/voices` | Lista de voces disponibles |
| POST | `/tts/convert` | Convertir texto a audio |
| POST | `/tts/upload` | Extraer texto de TXT/PDF/DOCX |
| POST | `/stt/transcribe` | Transcribir audio con Whisper |
| POST | `/stt/download` | Descargar transcripción en TXT |
| POST | `/video-to-gif` | Convertir vídeo a GIF con ffmpeg |

---

## Herramientas

| Herramienta | Archivo | Requiere servidor |
|-------------|---------|-------------------|
| Mejorar imagen | `pixora.html` | ✅ FSRCNN |
| Texto a Audio | `tts.html` | ✅ Edge TTS |
| Audio a Texto | `stt.html` | ✅ Whisper |
| Paleta de colores | `paleta-colores.html` | ✅ Claude AI (API) |
| Eliminar fondo | `eliminar-fondo.html` | ✅ rembg |
| Código QR | `generar-qr.html` | ❌ 100% navegador |
| GIF desde vídeo | `gif-desde-video.html` | ✅ ffmpeg |

---

## Notas

- La paleta de colores usa la API de Anthropic Claude directamente desde el navegador.
- El generador de QR funciona completamente en el navegador sin servidor.
- Los modelos FSRCNN (~1 MB c/u) se descargan automáticamente la primera vez.
- El modelo Whisper `tiny` se descarga automáticamente la primera vez (~150 MB).
- El modelo U2Net de rembg se descarga automáticamente la primera vez (~170 MB).
