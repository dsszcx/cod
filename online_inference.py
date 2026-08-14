"""online_inference.py - Full online pipeline: image -> Qwen2.5-VL prior -> prior encoder -> network -> saliency map.

Usage:
  python online_inference.py --image ./test.jpg --pth_path ./model_pth/PRIOR/KKK_best_2.pth

If --llm is enabled (server with Qwen2.5-VL), priors are generated live.
Otherwise offline priors from --prior_root are used (fallback for local demo).
"""
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image


def load_network(pth_path, vocab_size):
    from Network_Prior import Network
    net = Network(channel=64, vocab_size=vocab_size).cuda()
    if pth_path and os.path.exists(pth_path):
        net.load_state_dict(torch.load(pth_path, map_location="cuda"), strict=False)
        print(f">>> Loaded model: {pth_path}")
    else:
        print(">>> WARNING: no pretrained weights, using random init")
    net.eval()
    return net


def load_vocab(vocab_path):
    import json
    with open(vocab_path, "r", encoding="utf-8") as f:
        return json.load(f)  # {tag: idx}


def get_offline_prior(image_path, prior_root, vocab_size, size=352):
    """Fallback: read precomputed prior npz by image name."""
    name = os.path.basename(image_path)
    npz_name = f"test_{name.replace('.', '_')}.npz"
    npz_path = os.path.join(prior_root, npz_name)
    if os.path.exists(npz_path):
        d = np.load(npz_path)
        pos = torch.from_numpy(d["pos_map"]).float()
        bow = torch.from_numpy(d["tag_bow"]).float()
        return pos, bow
    return torch.zeros(1, size, size), torch.zeros(vocab_size)


def get_live_prior(image_path, annotator, vocab):
    """Live: call Qwen2.5-VL to get tags+bbox, then encode priors."""
    from prior_encoder import bbox_to_heatmap, tags_to_bow
    ann = annotator.generate(image_path)
    bbox = ann.get("bbox", [])
    tags = ann.get("tags", [])
    pos = torch.from_numpy(bbox_to_heatmap(bbox, 352)).float()
    bow = torch.from_numpy(tags_to_bow(tags, vocab)).float()
    print(f">>> LLM prior: tags={tags}, bbox={bbox}, conf={ann.get('confidence')}")
    return pos, bow, ann


def run_single(net, image_path, pos, bow, size=352):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).cuda()
    pos = pos.unsqueeze(0).cuda()
    bow = bow.unsqueeze(0).cuda()
    with torch.no_grad():
        _, _, _, c4 = net(x, pos, bow)
    res = c4.sigmoid().squeeze().cpu().numpy()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    return (res * 255).astype(np.uint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="input image path")
    parser.add_argument("--pth_path", type=str, default=None, help="trained network weights")
    parser.add_argument("--vocab_path", type=str, default="./prior/vocab.json")
    parser.add_argument("--vocab_size", type=int, default=256)
    parser.add_argument("--prior_root", type=str, default="./prior/")
    parser.add_argument("--llm", action="store_true", help="use live Qwen2.5-VL (requires server)")
    parser.add_argument("--llm_model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--out", type=str, default="online_result.png")
    opt = parser.parse_args()

    net = load_network(opt.pth_path, opt.vocab_size)
    vocab = load_vocab(opt.vocab_path) if os.path.exists(opt.vocab_path) else {}

    if opt.llm:
        from llm_annotator import QwenVLAnnotator
        annotator = QwenVLAnnotator(model_path=opt.llm_model)
        pos, bow, _ = get_live_prior(opt.image, annotator, vocab)
    else:
        pos, bow = get_offline_prior(opt.image, opt.prior_root, opt.vocab_size)
        print(">>> Using offline prior (no --llm). Enable --llm on server for live inference.")

    mask = run_single(net, opt.image, pos, bow)
    Image.fromarray(mask).save(opt.out)
    print(f">>> Saliency map saved to {opt.out}")