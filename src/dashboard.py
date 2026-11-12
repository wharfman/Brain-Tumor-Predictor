"""Local web dashboard: upload a brain MRI scan and see the classifier's prediction.

Usage:
    python -m src.dashboard [--port 8000] [--weights models/classify_best.pt]

Then open http://127.0.0.1:8000 in a browser. The server only listens on this machine, and scans
are run through the same pipeline as src.predict (see src/transforms.py) before classifying.
"""

import argparse
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from src.model import load_best_model
from src.predict import summarize
from src.transforms import preprocess_image

PAGE = Path(__file__).with_name("dashboard.html")
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
NO_TUMOR = "notumor"


def make_classifier(weights: str | None = None):
    """Load the trained classifier once and return a function mapping image bytes to a result dict."""
    model = load_best_model("classify", weights)
    lock = threading.Lock()  # the server is threaded; run one prediction at a time

    def classify(data: bytes) -> dict:
        with Image.open(io.BytesIO(data)) as image:
            image = preprocess_image(image).convert("RGB")  # YOLO expects 3 channels
        with lock:
            result = model.predict(image, verbose=False)[0]
        summary = summarize(result)
        return {**summary, "tumor": summary["label"] != NO_TUMOR}

    return classify


def make_handler(classify):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/":
                return self._send(404, "text/plain; charset=utf-8", b"Not found")
            self._send(200, "text/html; charset=utf-8", PAGE.read_bytes())

        def do_POST(self):
            if self.path != "/predict":
                return self._json(404, {"error": "Not found"})
            length = int(self.headers.get("Content-Length") or 0)
            if length == 0:
                return self._json(400, {"error": "No image was uploaded."})
            if length > MAX_UPLOAD_BYTES:
                return self._json(413, {"error": f"Image is larger than {MAX_UPLOAD_BYTES // 2**20} MB."})
            try:
                self._json(200, classify(self.rfile.read(length)))
            except (UnidentifiedImageError, OSError):
                self._json(400, {"error": "That file isn't a readable image. Use a JPG, PNG or BMP scan."})

        def _json(self, status: int, body: dict):
            self._send(status, "application/json", json.dumps(body).encode())

        def _send(self, status: int, content_type: str, body: bytes):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--weights", help="checkpoint to use (default: models/classify_best.pt)")
    args = parser.parse_args()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(make_classifier(args.weights)))
    print(f"Dashboard running at http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
