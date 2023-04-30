from http.server import BaseHTTPRequestHandler, HTTPServer
import os
import json
from pathlib import Path
import base64
from urllib.parse import urlparse

WEBUI_CONTENT = open(os.path.join(os.path.dirname(__file__), "index.html"), "r").read()
WEBUI_CSS = open(os.path.join(os.path.dirname(__file__), "style.css"), "r").read()

class openWakeWordWebUI(BaseHTTPRequestHandler):
    def __init__(self, custom_object, *args, **kwargs):
        self.oww_instance = custom_object
        super().__init__(*args, **kwargs)

    def _set_headers(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

    def do_GET(self):
        if self.path == "/style.css":
            self.wfile.write(bytes(WEBUI_CSS, "utf8"))

        if self.path == "/":
            self._set_headers()

            # Load WebUI
            self.wfile.write(bytes(WEBUI_CONTENT, "utf8"))

        if "/delete_clip" in self.path:
            self._set_headers()
            query = urlparse(self.path).query
            query_params = dict(qc.split("=") for qc in query.split("&"))
            os.remove(os.path.join(os.getcwd(), "activation_clips", query_params["filename"]))

        if self.path == "/list_cache_files":
            self._set_headers()

            # Load files and prepare data
            data = []
            for i in Path(os.path.join(os.getcwd(), "activation_clips")).glob("**/*.ogg"):
                audio_bytes = open(i, "rb").read()
                base64_data = base64.b64encode(audio_bytes).decode('utf-8')
                data.append(
                    {
                        "filename": str(i).split(os.path.sep)[-1],
                        "duration": 0,
                        "data": "data:audio/ogg;base64," + base64_data
                    }
                )
            
            self.wfile.write(bytes(json.dumps(data), "utf8"))
