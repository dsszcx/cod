"""web_server_sam3.py - Web demo using trained SAM3-prior network (Network_SAM3).

Usage:
  python web_server_sam3.py --port 8091 --pth ./model_pth/SAM3/KKK_best_41.pth
Then open http://127.0.0.1:8091

Upload an image -> if a prior exists in prior_sam3/, use target+aux;
otherwise fall back to zero prior (network segments autonomously).
POST /predict?name=<original file name> makes the prior lookup deterministic
(without it the demo always shows the autonomous / no-prior behaviour).

Layer-0 notes (2026-10):
  * the prior is resolved by dataloader_sam3.load_test_prior and goes through the
    shared dataloader_sam3.prepare_prior_at, so the demo sees exactly the same
    input distribution as training/inference (one resolver for all entry points).
  * a missing prior npz is reported on stdout before falling back to zeros
    (autonomous mode); the demo used to show nothing at all about coverage.
  * the checkpoint is channel-checked (a 7ch file used to load silently and the
    demo would look "trained but bad" instead of reporting the mismatch).
  * --pth MUST point at a checkpoint produced by MyTrain_SAM3.py (8ch+fusion).
"""
import argparse
import base64
import io
import json
import os
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from dataloader_sam3 import is_noncam, load_test_prior, prepare_prior_at
from check_ckpt_channels import load_with_report
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
    # The checkpoint supplies every weight -> skip the ImageNet load (a missing
    # ./pretrained/pvt_v2_b2.pth must not take the demo server down).
    model = Network(channel=64, imagenet_pretrained=False)
    if os.path.exists(pth_path):
        # Layer-0 guard: only an 8ch + prior_fusion checkpoint may be loaded
        load_with_report(model, pth_path, expected=8)
        print(f">>> Model loaded: {pth_path}")
    else:
        print(f">>> WARNING: {pth_path} not found")
    model.to(device).eval()


def get_prior(img_name):
    """Load target+aux prior for an image by basename -> (target, aux, found).

    Goes through ``dataloader_sam3.load_test_prior`` (the single resolver used by
    MyTrain_SAM3 / MyTest_SAM3), so the ``.jpg``/``.png`` npz naming and the
    full-path case cannot diverge between entry points again. The returned masks
    are already resized to 352 and binarized at 0.5.
    """
    base = os.path.basename(img_name or "")
    if base:
        for split in ("test", "train"):
            target, aux, found = load_test_prior(split, base, PRIOR_ROOT, (352, 352))
            if found:
                return target, aux, True
    # no prior for this image (demo falls back to autonomous / zero-prior mode)
    return torch.zeros(1, 352, 352), torch.zeros(3, 352, 352), False


def predict(image_bytes, img_name=None):
    """img_name: original file name (e.g. camourflage_00001.jpg) so the matching
    prior_sam3/*.npz can be found; None -> zero prior (autonomous mode)."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    x = img_transform(img).unsqueeze(0).to(device)

    target, aux, prior_found = get_prior(img_name or "")
    if not prior_found:
        # Layer-0: never fall back to zeros silently -- say so in the log
        print(f">>> [prior] no npz for {img_name!r} under {PRIOR_ROOT} "
              f"-> zero prior (autonomous mode)")
    # Layer-0: same resize+binarize path as training and MyTest_SAM3
    target, aux = prepare_prior_at(target, aux, (352, 352))
    target = target.unsqueeze(0).to(device)
    aux = aux.unsqueeze(0).to(device)

    with torch.no_grad():
        # same 'NonCAM' rule as the training loader (0 when the name is unknown)
        neg_flag = torch.full((1, 1, x.shape[-2], x.shape[-1]),
                              float(is_noncam(img_name or "")), device=x.device)
        _, _, _, c4 = model(x, target, aux, neg_flag)

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
        if not self.path.startswith("/predict"):
            self._send(b"404", "text/plain")
            return
        length = int(self.headers.get("Content-Length", 0))
        if length > MAX_UPLOAD:
            self._send(json.dumps({"status": "error", "message": "图片过大"}).encode(), "application/json")
            return
        body = self.rfile.read(length)
        try:
            t0 = time.time()
            qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            result_bytes = predict(body, img_name=qs.get("name", [None])[0])
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
    parser.add_argument("--pth", type=str, default="./model_pth/SAM3/KKK_best_41.pth",
                        help="MUST be an 8ch + prior_fusion checkpoint written by "
                             "MyTrain_SAM3.py (validated by check_ckpt_channels.py)")
    parser.add_argument("--prior_root", type=str, default="./prior_sam3")
    opt = parser.parse_args()

    PRIOR_ROOT = opt.prior_root
    load_model(opt.pth)
    load_index()
    server = WebServer(("127.0.0.1", opt.port), Handler)
    print(f">>> SAM3 web demo running at http://127.0.0.1:{opt.port}")
    server.serve_forever()
