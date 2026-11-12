import io
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest
import torch
from PIL import Image
from ultralytics.nn.tasks import ClassificationModel

from src.dashboard import make_classifier, make_handler


def png_bytes(size=(48, 40)) -> bytes:
    buffer = io.BytesIO()
    Image.new("L", size, 120).save(buffer, "PNG")
    return buffer.getvalue()


@pytest.fixture
def server():
    """Dashboard server on a free port, with a stand-in classifier that checks it got an image."""

    def classify(data):
        Image.open(io.BytesIO(data)).verify()  # raises like the real classifier on non-images
        return {"label": "glioma", "confidence": 0.9, "probabilities": {"glioma": 0.9, "notumor": 0.1}, "tumor": True}

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(classify))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def request(url, data=None):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=data, method="POST" if data is not None else "GET")) as r:
            return r.status, r.headers["Content-Type"], r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers["Content-Type"], e.read()


def test_serves_page(server):
    status, content_type, body = request(server + "/")
    assert status == 200 and content_type.startswith("text/html") and b"/predict" in body
    assert request(server + "/missing")[0] == 404


def test_predict_returns_json(server):
    status, content_type, body = request(server + "/predict", png_bytes())
    assert status == 200 and content_type == "application/json"
    assert json.loads(body)["label"] == "glioma"


def test_predict_rejects_bad_uploads(server):
    status, _, body = request(server + "/predict", b"not an image")
    assert status == 400 and "readable image" in json.loads(body)["error"]
    assert request(server + "/predict", b"")[0] == 400


def test_classifier_runs_pipeline_and_model(tmp_path):
    model = ClassificationModel("yolo11n-cls.yaml", nc=4, verbose=False).eval()
    model.names = {0: "glioma", 1: "meningioma", 2: "notumor", 3: "pituitary"}
    path = tmp_path / "classify.pt"
    torch.save({"model": model, "train_args": {"task": "classify"}}, path)

    result = make_classifier(str(path))(png_bytes())
    assert result["tumor"] == (result["label"] != "notumor")
    assert 0 <= result["confidence"] <= 1
    assert abs(sum(result["probabilities"].values()) - 1) < 1e-3
