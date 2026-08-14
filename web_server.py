# -*- coding: utf-8 -*-
"""web_server.py - Local web demo for camouflaged object detection (zero-dependency).

Usage:
  python web_server.py --port 8090 --pth ./model_pth/OUR/Net_epoch_40.pth

Then open http://127.0.0.1:8090 in a browser.
"""
import argparse
import base64
import io
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from Network_PVTv2 import Network

MAX_UPLOAD = 15 * 1024 * 1024  # 15MB
MAX_OUTPUT = 1024              # longest edge of returned result

# ---------------- global model ----------------
model = None
device = None

img_transform = transforms.Compose([
    transforms.Resize((352, 352)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model(pth_path):
    global model, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Network(channel=64)
    if os.path.exists(pth_path):
        model.load_state_dict(torch.load(pth_path, map_location=device), strict=False)
        print(f">>> Model loaded: {pth_path}")
    else:
        print(f">>> WARNING: {pth_path} not found, using random init")
    model.to(device).eval()
    print(f">>> Device: {device}")


def predict(image_bytes):
    """Run model on uploaded image, return compressed result PNG bytes."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size

    x = img_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        _, _, _, c4 = model(x)

    res = F.interpolate(c4, size=(h, w), mode="bilinear", align_corners=False)
    res = res.sigmoid().squeeze().cpu().numpy()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    mask = Image.fromarray((res * 255).astype(np.uint8)).convert("L")

    # limit output size to keep response small
    ratio = MAX_OUTPUT / max(w, h)
    if ratio < 1.0:
        mask = mask.resize((int(w * ratio), int(h * ratio)), Image.BILINEAR)

    buf = io.BytesIO()
    mask.save(buf, format="PNG")
    return buf.getvalue()


# ---------------- HTTP handler ----------------
INDEX_HTML = None


def load_index():
    global INDEX_HTML
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates", "index.html")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            INDEX_HTML = f.read()
    else:
        INDEX_HTML = "<h1>templates/index.html not found</h1>"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass  # quiet

    def _send(self, body, ctype="text/html; charset=utf-8"):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(INDEX_HTML.encode("utf-8"))
        elif self.path == "/health":
            self._send(b"{\"status\":\"ok\"}", "application/json")
        else:
            self._send(b"404 Not Found", "text/plain")

    def do_POST(self):
        if self.path != "/predict":
            self._send(b"404 Not Found", "text/plain")
            return
        length = int(self.headers.get("Content-Length", 0))
        if length > MAX_UPLOAD:
            payload = json.dumps({"status": "error", "message": "图片过大(>15MB)"}).encode("utf-8")
            self._send(payload, "application/json")
            return
        body = self.rfile.read(length)
        try:
            t0 = time.time()
            result_bytes = predict(body)
            elapsed = (time.time() - t0) * 1000
            payload = json.dumps({
                "status": "ok",
                "result": base64.b64encode(result_bytes).decode(),
                "time_ms": round(elapsed, 1),
            }).encode("utf-8")
            self._send(payload, "application/json")
        except Exception as e:
            import traceback
            traceback.print_exc()
            payload = json.dumps({"status": "error", "message": str(e)}).encode("utf-8")
            self._send(payload, "application/json")


class WebServer(ThreadingHTTPServer):
    daemon_threads = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--pth", type=str, default="./model_pth/OUR/Net_epoch_40.pth")
    opt = parser.parse_args()

    load_model(opt.pth)
    load_index()
    server = WebServer(("127.0.0.1", opt.port), Handler)
    print(f">>> Web demo running at http://127.0.0.1:{opt.port}")
    print(">>> Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n>>> Stopped")