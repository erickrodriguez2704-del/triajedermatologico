"""
Servidor Flask embebido para recibir fotos de alta calidad desde el celular vía QR.
Se ejecuta en un hilo en segundo plano mientras corre la app Streamlit.
"""

import io
import json
import os
import socket
import threading
import time
from datetime import datetime
from pathlib import Path

import qrcode
from flask import Flask, jsonify, render_template_string, request
from werkzeug.serving import make_server

RUTA_BASE = Path(__file__).resolve().parent
CARPETA_FOTOS = RUTA_BASE / "fotos_recibidas"
ARCHIVO_ULTIMA = CARPETA_FOTOS / "_ultima_foto.json"
PUERTO_QR = 5000
EXTENSIONES_IMAGEN = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".bmp"}

HTML_MOBILE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Subir Foto de Lesión</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; padding: 20px; background-color: #f4f4f9; }
        .btn { background-color: #28a745; color: white; padding: 15px 25px; border: none;
               font-size: 18px; border-radius: 8px; cursor: pointer; margin-top: 20px; display: inline-block; }
        input[type="file"] { display: none; }
        #status { margin-top: 20px; font-weight: bold; color: #333; }
    </style>
</head>
<body>
    <h2>Escáner Dermatológico</h2>
    <p>Selecciona una fotografía previamente tomada con la cámara nativa de tu celular.</p>
    <p><small>Asegúrate de haber posicionado correctamente el <b>cono</b> al momento de la captura original.</small></p>

    <label for="cameraInput" class="btn">📷 Elegir desde la Galería o tomar foto</label>
    <input type="file" id="cameraInput" accept="image/*">

    <div id="status"></div>

    <script>
        const input = document.getElementById('cameraInput');
        const status = document.getElementById('status');

        input.addEventListener('change', async () => {
            const file = input.files[0];
            if (!file) return;

            status.innerText = "Subiendo imagen en alta resolución...";
            const formData = new FormData();
            formData.append('photo', file);

            try {
                const response = await fetch('/upload', { method: 'POST', body: formData });
                const result = await response.json();
                if (result.success) {
                    status.innerText = "✅ ¡Foto recibida en la computadora!";
                    status.style.color = "green";
                } else {
                    status.innerText = "❌ Error al subir la foto.";
                    status.style.color = "red";
                }
            } catch (error) {
                status.innerText = "❌ Error de conexión. Verifica que estés en la misma red Wi-Fi.";
                status.style.color = "red";
            }
        });
    </script>
</body>
</html>
"""

flask_app = Flask(__name__)
CARPETA_FOTOS.mkdir(parents=True, exist_ok=True)
flask_app.config["UPLOAD_FOLDER"] = str(CARPETA_FOTOS)


def _registrar_ultima_foto(filepath):
    info = {
        "path": str(filepath),
        "nombre": Path(filepath).name,
        "mtime": Path(filepath).stat().st_mtime,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    ARCHIVO_ULTIMA.write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
    return info


def _listar_fotos():
    if not CARPETA_FOTOS.exists():
        return []

    fotos = []
    for p in CARPETA_FOTOS.iterdir():
        if not p.is_file() or p.name.startswith("_"):
            continue
        if p.suffix.lower() in EXTENSIONES_IMAGEN or p.name.startswith("foto_"):
            fotos.append(p)
    return fotos


def obtener_ultima_foto():
    if ARCHIVO_ULTIMA.exists():
        try:
            info = json.loads(ARCHIVO_ULTIMA.read_text(encoding="utf-8"))
            ruta = Path(info["path"])
            if ruta.exists():
                return str(ruta)
        except (json.JSONDecodeError, KeyError, OSError):
            pass

    fotos = _listar_fotos()
    if not fotos:
        return None
    return str(max(fotos, key=lambda p: p.stat().st_mtime))


def info_ultima_foto():
    ruta = obtener_ultima_foto()
    if not ruta:
        return None
    p = Path(ruta)
    return {
        "path": str(p),
        "nombre": p.name,
        "mtime": p.stat().st_mtime,
    }


@flask_app.route("/")
def index():
    return render_template_string(HTML_MOBILE)


@flask_app.route("/upload", methods=["POST"])
def upload():
    if "photo" not in request.files:
        return jsonify({"success": False, "message": "No se encontró la foto"})

    file = request.files["photo"]
    if file.filename == "":
        return jsonify({"success": False, "message": "Nombre de archivo vacío"})

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    extension = os.path.splitext(file.filename)[1].lower() or ".jpg"
    nombre_unico = f"foto_{timestamp}{extension}"
    filepath = Path(flask_app.config["UPLOAD_FOLDER"]) / nombre_unico

    file.save(filepath)
    info = _registrar_ultima_foto(filepath)
    print(f"\n[QR] Nueva foto guardada en: {filepath}")

    return jsonify({"success": True, "path": str(filepath), "nombre": info["nombre"]})


@flask_app.route("/status")
def status():
    info = info_ultima_foto()
    fotos = _listar_fotos()
    return jsonify({
        "ok": True,
        "carpeta": str(CARPETA_FOTOS),
        "total_fotos": len(fotos),
        "ultima_foto": info,
    })


def obtener_ip_local():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def generar_imagen_qr(url):
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def _puerto_en_uso(puerto):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", puerto)) == 0


class _ServidorQR(threading.Thread):
    def __init__(self, puerto):
        super().__init__(daemon=True)
        self.puerto = puerto
        self._httpd = None
        self.error = None

    def run(self):
        try:
            self._httpd = make_server("0.0.0.0", self.puerto, flask_app, threaded=True)
            self._httpd.serve_forever()
        except OSError as exc:
            self.error = str(exc)


_servidor_thread = None


def iniciar_servidor_qr(puerto=PUERTO_QR):
    """Arranca el servidor Flask en segundo plano (una sola vez por proceso)."""
    global _servidor_thread

    ip = obtener_ip_local()
    url = f"http://{ip}:{puerto}"
    puerto_ocupado = _puerto_en_uso(puerto)

    if _servidor_thread is None:
        if puerto_ocupado:
            return {
                "ip": ip,
                "puerto": puerto,
                "url": url,
                "activo": True,
                "advertencia": (
                    f"El puerto {puerto} ya está en uso. "
                    "Cierra app.py u otro servidor Flask antes de continuar."
                ),
                "carpeta_fotos": str(CARPETA_FOTOS),
            }

        _servidor_thread = _ServidorQR(puerto)
        _servidor_thread.start()
        time.sleep(0.3)

        if _servidor_thread.error:
            return {
                "ip": ip,
                "puerto": puerto,
                "url": url,
                "activo": False,
                "advertencia": f"No se pudo iniciar el servidor QR: {_servidor_thread.error}",
                "carpeta_fotos": str(CARPETA_FOTOS),
            }

    return {
        "ip": ip,
        "puerto": puerto,
        "url": url,
        "activo": True,
        "advertencia": None,
        "carpeta_fotos": str(CARPETA_FOTOS),
    }