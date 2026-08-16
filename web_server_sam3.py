"""web_server_sam3.py - Web demo using trained SAM3-prior network (Network_SAM3).

Usage:
  python web_server_sam3.py --port 8091 --pth ./model_pth/SAM3/KKK_best_41.pth
Then open http://127.0.0.1:8091

Upload an image -> if a prior exists in prior_sam3/, use target+aux;
otherwise fall back to zero prior (network segments autonomously).
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

from Network_SAM3 import Network

MAX_UPLOAD = 15 * 1024 * 1024
MAX_OUTPUT = 1024

model = None
device = None
PRIOR_ROOT = "./prior_sam3"

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
        print(f">>> WARNING: {pth_path} not found")
    model.to(device).eval()


def get_prior(img_name):
    """Try to load target+aux prior for an image by basename."""
    base = os.path.basename(img_name)
    stem = os.path.splitext(base)[0]
    for tag in ["test_", "train_"]:
        npz = os.path.join(PRIOR_ROOT, f"{tag}{stem.replace('.', '_')}.npz")
        if os.path.exists(npz):
            d = np.load(npz, allow_pickle=True)
            return torch.from_numpy(d["target_mask"]).float(), torch.from_numpy(d["aux_masks"]).float()
    # no prior
    return torch.zeros(1, 1, 1), torch.zeros(3, 1, 1)


def predict(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    x = img_transform(img).unsqueeze(0).to(device)

    target, aux = get_prior(img_name="upload.jpg")
    # fallback: no prior -> zeros sized 352
    if target.numel() == 1:
        target = torch.zeros(1, 352, 352)
        aux = torch.zeros(3, 352, 352)
    target = F.interpolate(target.unsqueeze(0), size=(352, 352), mode="bilinear",
                           align_corners=False).to(device)
    aux = F.interpolate(aux.unsqueeze(0), size=(352, 352), mode="bilinear",
                        align_corners=False).to(device)

    with torch.no_grad():
        _, _, _, c4 = model(x, target, aux)

    res = F.interpolate(c4, size=(h, w), mode="bilinear", align_corners=False)
    res = res.sigmoid().squeeze().cpu().numpy()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    mask = Image.fromarray((res * 255).astype(np.uint8)).convert("L")
    ratio = MAX_OUTPUT / max(w, h)
    if ratio < 1.0:
        mask = mask.resize((int(w * ratio), int(h * ratio)), Image.BILINEAR)
    buf = io.BytesIO()
    mask.save(buf, format="PNG")
    return buf.getvalue()


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
        pass

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
            self._send(b"404", "text/plain")

    def do_POST(self):
        if self.path != "/predict":
            self._send(b"404", "text/plain")
            return
        length = int(self.headers.get("Content-Length", 0))
        if length > MAX_UPLOAD:
            self._send(json.dumps({"status": "error", "message": "图片过大"}).encode(), "application/json")
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
            self._send(json.dumps({"status": "error", "message": str(e)}).encode(), "application/json")


class WebServer(ThreadingHTTPServer):
    daemon_threads = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--pth", type=str, default="./model_pth/SAM3/KKK_best_41.pth")
    parser.add_argument("--prior_root", type=str, default="./prior_sam3")
    opt = parser.parse_args()

    PRIOR_ROOT = opt.prior_root
    load_model(opt.pth)
    load_index()
    server = WebServer(("127.0.0.1", opt.port), Handler)
    print(f">>> SAM3 web demo running at http://127.0.0.1:{opt.port}")
    server.serve_forever()