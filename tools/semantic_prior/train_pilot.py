"""Three-arm pilot trainer. Explicit fixed manifests; never reads test dataset."""
import argparse
from datetime import datetime, timezone
import gc
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.e0e4 import common as C
from tools.semantic_prior.common import resolve
from tools.semantic_prior.training_core import build_model, deep_loss, forward_arm, seed_all, state_digest, optimizer_parameters, core_state_dict
from tools.semantic_prior.prepare_cam_only import category
from dataloader_relation import RelationDataset, collate_relation_batch
from tools.semantic_prior.candidate_supervision import candidate_reliability_loss


def evaluate(model, dataset, arm, device):
    model.eval()
    results = []
    with torch.inference_mode():
        for index, row in enumerate(dataset.rows):
            item = dataset[index]
            batch = collate_relation_batch([item])
            inp = {k: t.to(device) for k, t in batch["inputs"].items()}
            output = forward_arm(model, inp, arm)[-1]
            logits = F.interpolate(output.float(), size=item["native_hw"], mode="bilinear", align_corners=False)
            prob = logits.sigmoid()[0, 0].cpu().numpy()
            with Image.open(row["gt_path"]) as im:
                target = np.asarray(im.convert("L"), np.float32)
            if target.max() > 1:
                target /= 255.
            if not np.isfinite(prob).all():
                raise ValueError("Nonfinite validation output")
            results.append({"image_id": row["image_id"], "mae": float(np.abs(prob-target).mean()),
                            "gt_empty": bool(not target.any()), "prob": prob})
    return float(np.mean([r["mae"] for r in results])), results


def fusion_snapshot(module, inputs, output):
    with torch.no_grad():
        _, debug = module.forward(*inputs,return_debug=True)
        denom = inputs[0].float().norm().clamp_min(1e-12)
        module._pilot_snapshot = {
            "direct_relative_l2":float((module.alpha_direct*debug["direct_map"]).float().norm()/denom),
            "relation_relative_l2":float((module.alpha_relation*debug["relation_map"]).float().norm()/denom),
            "valid_nodes":int(debug["effective_valid"].sum()),
            "rgb_feature_rms":float(inputs[0].float().square().mean().sqrt())}


def gradient_snapshot(model):
    groups={"backbone":[],"node":[],"direct":[],"relation":[],"alpha":[]}
    for name,param in model.named_parameters():
        if param.grad is None: continue
        if name.startswith("backbone."): group="backbone"
        elif name.startswith("region_fusion.alpha"): group="alpha"
        elif name.startswith("region_fusion.node_encoder"): group="node"
        elif name.startswith(("region_fusion.direct_project","region_fusion.direct_gate")): group="direct"
        elif name.startswith("region_fusion."): group="relation"
        else: continue
        groups[group].append(float(param.grad.detach().float().square().sum()))
    return {k:math.sqrt(sum(v)) for k,v in groups.items()}


def manifest_paths_for(cfg, root):
    """Keep pilot defaults; full-data jobs explicitly request train.csv/val.csv."""
    names = cfg.get('manifest_files', {'train': 'pilot_train.csv', 'val': 'pilot_val.csv'})
    if set(names) != {'train', 'val'} or any(Path(v).name != v for v in names.values()):
        raise ValueError('manifest_files requires plain train/val filenames')
    return [Path(root)/names[split] for split in ('train', 'val')]


def require_prior_files(manifests, arms):
    # RGB never reads masks/roles. Missing priors must still fail for either
    # region arm, rather than being silently replaced with empty candidates.
    if set(arms) == {'rgb'}:
        return
    for rows in manifests:
        for row in rows:
            if not row.get('region_package') or not Path(row['region_package']).is_file():
                raise ValueError('Prior missing for a region arm: '+row['image_id'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/relation_smoke_20261026.yaml")
    args = ap.parse_args()
    cfg = C.load_config(str(resolve(args.config)))
    reliability_weight = float(cfg.get("candidate_reliability_weight", 0.))
    quality_weight = float(cfg.get('quality_supervision_weight', 0.))
    if not math.isfinite(quality_weight) or quality_weight < 0:
        raise ValueError('quality_supervision_weight must be finite and nonnegative')
    if quality_weight and (cfg.get('quality_options') is None or reliability_weight):
        raise ValueError('Independent quality loss requires its head and no direct-gate supervision')
    if cfg.get('quality_options') is not None and ('rgb' in cfg['arms'] or cfg.get('freeze_all_priors', False)):
        raise ValueError('Independent quality experiment requires trainable prior arms')
    if not math.isfinite(reliability_weight) or reliability_weight < 0:
        raise ValueError("candidate_reliability_weight must be finite and nonnegative")
    if reliability_weight and ("rgb" in cfg["arms"] or cfg.get("freeze_all_priors", False)):
        raise ValueError("Candidate supervision requires trainable prior arms")
    torch.set_num_threads(4)
    if cfg.get("deterministic_algorithms", False):
        # Decoder bilinear backward may lack a deterministic CUDA kernel in
        # some torch builds. Emit warnings rather than silently claiming exact
        # determinism or changing the architecture for the experiment.
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("Pilot requires the assigned CUDA device; no silent CPU fallback")
    device = torch.device("cuda:0")
    if not cfg.get("deadline_utc"):
        raise ValueError("Set an explicit deadline_utc before starting a training run")
    deadline = datetime.fromisoformat(cfg["deadline_utc"])
    def check_time():
        if datetime.now(timezone.utc) >= deadline:
            raise TimeoutError("Configured training deadline reached")
    check_time()
    root = resolve(cfg["paths"]["out_dir"])
    root.mkdir(exist_ok=False)
    shutil.copy2(cfg["_path"], root/"CONFIG.yaml")
    manifest_root = resolve(cfg["paths"]["manifests"])
    manifest_paths = manifest_paths_for(cfg, manifest_root)
    manifests = [C.read_csv(str(p)) for p in manifest_paths]
    expected = [int(cfg.get("expected_train_images",64)), int(cfg.get("expected_val_images",16))]
    if [len(x) for x in manifests] != expected or {r["image_sha256"] for r in manifests[0]} & {r["image_sha256"] for r in manifests[1]}:
        raise ValueError("Fixed split changed or exact-content leakage")
    if cfg.get("data_scope") == "cam_only" and any(category(r) == "NonCAM" for rows in manifests for r in rows):
        raise ValueError("NonCAM present in a CAM-only experiment")
    require_prior_files(manifests, cfg['arms'])
    pretrained = resolve(cfg["paths"]["imagenet"])
    pretrained_hash = C.sha256_file(str(pretrained))
    if not pretrained_hash:
        raise FileNotFoundError(str(pretrained))
    code_files = ["Network_Relation.py", "Network_PVTv2.py", "lib/region_fusion.py", "dataloader_relation.py",
                  "tools/semantic_prior/training_core.py", "tools/semantic_prior/train_pilot.py",
                  "tools/semantic_prior/candidate_supervision.py"]
    provenance = {"imagenet_sha256": pretrained_hash,
                  "manifest_sha256": {p.name:C.sha256_file(str(p)) for p in manifest_paths},
                  "code_sha256": {p:C.sha256_file(str(resolve(p))) for p in code_files},
                  "test_set_used": False, "historical_cod_checkpoint_used": False,
                  "data_scope":cfg.get("data_scope","historical_mixed"),
                  "candidate_reliability_weight": reliability_weight,
                  'quality_supervision_weight': quality_weight,
                  'quality_options': cfg.get('quality_options'),
                  'quality_optimizer': 'separate AdamW and separate clipping; head excluded from main clip norm',
                  "candidate_supervision_scope": "train GT after forward only; no relation labels",
                  "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
    provenance.update(deterministic_algorithms=cfg.get("deterministic_algorithms",False),
                      deterministic_warn_only=cfg.get("deterministic_algorithms",False),
                      matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                      cudnn_tf32=torch.backends.cudnn.allow_tf32)
    C.write_json(str(root/"PROVENANCE.json"), provenance)
    summaries, common_rgb, common_region, common_orders = [], None, None, {}
    for arm in cfg["arms"]:
        check_time()
        arm_root = root/arm
        arm_root.mkdir()
        model, initialization = build_model(arm, cfg["seed"], str(pretrained), cfg.get("fusion_options"), cfg.get('quality_options'))
        if cfg.get("freeze_all_priors",False):
            if arm == "rgb":
                raise ValueError("Residual control requires a relation model")
            with torch.no_grad():
                model.region_fusion.alpha_direct.zero_()
                model.region_fusion.alpha_relation.zero_()
            initialization['frozen_parameters'] = []
            for name,param in model.region_fusion.named_parameters():
                param.requires_grad_(False)
                initialization['frozen_parameters'].append('region_fusion.'+name)
            initialization['control'] = 'all prior parameters frozen, both residual coefficients zero'
        if common_rgb is None:
            common_rgb = initialization["rgb_initial_sha256"]
        if common_rgb != initialization["rgb_initial_sha256"]:
            raise ValueError("Different RGB initialization across arms")
        if arm != "rgb":
            if common_region is None:
                common_region = initialization["region_initial_sha256"]
            if common_region != initialization["region_initial_sha256"]:
                raise ValueError("Different region initialization")
        C.write_json(str(arm_root/"INITIALIZATION.json"), initialization)
        frozen_before = {n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
        model.to(device)
        train = RelationDataset(manifest_paths[0], "train", arm, cfg["input_size"], cfg["seed"])
        val = RelationDataset(manifest_paths[1], "val", arm, cfg["input_size"], cfg["seed"])
        main_parameters, quality_parameters = optimizer_parameters(model)
        optimizer = torch.optim.AdamW(main_parameters,
                                     lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
        quality_optimizer = (torch.optim.AdamW(quality_parameters, lr=cfg['learning_rate'],
                                              weight_decay=cfg['weight_decay']) if quality_weight else None)
        best, best_epoch, history, steps = float("inf"), None, [], 0
        started = time.time()
        for epoch in range(cfg["epochs"]):
            check_time()
            train.set_epoch(epoch)
            generator = torch.Generator().manual_seed(cfg["seed"]+epoch)
            loader = DataLoader(train, batch_size=cfg["batch_size"], shuffle=True, generator=generator,
                                num_workers=0, drop_last=False, collate_fn=collate_relation_batch)
            seed_all(cfg["seed"]+epoch)  # independent of extra module initialization/drop-path calls
            model.train()
            losses, order, alpha_log, fusion_stats = [], [], [], {}
            segmentation_losses, reliability_losses, quality_losses = [], [], []
            for index, batch in enumerate(loader, 1):
                check_time()
                order.extend(batch["image_ids"])
                inp = {k:t.to(device) for k,t in batch["inputs"].items()}
                target = batch["gt"].to(device)
                optimizer.zero_grad(set_to_none=True)
                if quality_optimizer is not None:
                    quality_optimizer.zero_grad(set_to_none=True)
                handle = None
                if cfg.get("collect_fusion_stats",False) and index == 1 and arm != "rgb":
                    handle = model.region_fusion.register_forward_hook(fusion_snapshot)
                if reliability_weight or quality_weight:
                    outputs, auxiliary = model(**inp, return_aux=True)
                else:
                    outputs = forward_arm(model, inp, arm)
                if handle is not None:
                    handle.remove()
                    fusion_stats.update(model.region_fusion._pilot_snapshot)
                segmentation_loss = deep_loss(outputs, target)
                reliability_loss = (candidate_reliability_loss(
                    auxiliary["direct_logits"], inp["region_masks"],
                    auxiliary["effective_valid"], target) if reliability_weight
                    else segmentation_loss.new_zeros(()))
                quality_loss = (candidate_reliability_loss(
                    auxiliary['quality_logits'], inp['region_masks'], auxiliary['effective_valid'], target)
                    if quality_weight else segmentation_loss.new_zeros(()))
                loss = segmentation_loss + reliability_weight * reliability_loss + quality_weight * quality_loss
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                loss.backward()
                if cfg.get("collect_fusion_stats",False) and index == 1:
                    fusion_stats["unclipped_gradient_l2"] = gradient_snapshot(model)
                norm = torch.nn.utils.clip_grad_norm_(main_parameters, 1.)
                if not torch.isfinite(norm):
                    raise ValueError("Nonfinite training gradients")
                optimizer.step()
                if quality_optimizer is not None:
                    quality_norm = torch.nn.utils.clip_grad_norm_(quality_parameters, 1.)
                    if not torch.isfinite(quality_norm):
                        raise ValueError('Nonfinite independent quality gradients')
                    quality_optimizer.step()
                steps += 1
                losses.append(float(loss.detach()))
                segmentation_losses.append(float(segmentation_loss.detach()))
                reliability_losses.append(float(reliability_loss.detach()))
                quality_losses.append(float(quality_loss.detach()))
                if reliability_weight or quality_weight:
                    del auxiliary
                if index % 8 == 0:
                    print(arm, "epoch", epoch+1, "step", index, "/", len(loader), "loss", losses[-1], flush=True)
            if epoch in common_orders and common_orders[epoch] != order:
                raise ValueError("Different sample ordering across arms")
            common_orders[epoch] = order
            mae, details = evaluate(model, val, arm, device)
            if arm != "rgb":
                alpha_log = [float(model.region_fusion.alpha_direct.detach()), float(model.region_fusion.alpha_relation.detach())]
            record = {"epoch":epoch+1, "mean_training_loss":float(np.mean(losses)), "val_native_mae":mae,
                      "mean_segmentation_loss":float(np.mean(segmentation_losses)),
                      "mean_candidate_reliability_loss":float(np.mean(reliability_losses)),
                      "candidate_reliability_weight":reliability_weight,
                      'quality_supervision_weight': quality_weight,
                      'mean_independent_quality_loss': float(np.mean(quality_losses)),
                      "optimizer_steps":steps, "alphas":alpha_log,"first_batch_fusion_stats":fusion_stats}
            history.append(record)
            C.write_json(str(arm_root/("epoch_%02d.json"%(epoch+1))),
                         {**record, "sample_order":order, "per_image":[{k:v for k,v in d.items() if k!="prob"} for d in details]})
            if mae < best:
                best, best_epoch = mae, epoch+1
                torch.save({"model":model.state_dict(), "epoch":best_epoch, "arm":arm,
                            "val_native_mae":mae, "config":cfg, "initialization":initialization}, arm_root/"best.pth")
                preds = arm_root/("val_epoch_%02d"%(epoch+1))
                preds.mkdir()
                for d in details:
                    name = d["image_id"].split("/")[-1]
                    np.save(preds/(name+".npy"), d["prob"])
            print("EPOCH_DONE", arm, record, flush=True)
            C.write_json(str(arm_root/"HISTORY.json"), history)
        frozen_equal = all(torch.equal(dict(model.named_parameters())[n].detach().cpu(), p.cpu()) for n,p in frozen_before.items())
        if not frozen_equal:
            raise AssertionError("Frozen relation parameters changed")
        summary = {"arm":arm,"best_val_mae":best,"best_epoch":best_epoch,"optimizer_steps":steps,
                   'core_final_sha256': state_digest(core_state_dict(model)),
                   "frozen_parameters_unchanged":frozen_equal,"elapsed_s":round(time.time()-started,1),"history":history}
        summaries.append(summary)
        C.write_json(str(root/"SUMMARY.json"), {"status":"RUNNING","arms":summaries})
        print("ARM_DONE", summary, flush=True)
        del model, optimizer, quality_optimizer, main_parameters, quality_parameters
        gc.collect()
        torch.cuda.empty_cache()
    if C.sha256_file(str(pretrained)) != pretrained_hash:
        raise ValueError("Source ImageNet file changed")
    C.write_json(str(root/"SUMMARY.json"), {"status":"DONE", "arms":summaries,
        "same_rgb_initialization":True,"same_region_initialization":True,"same_sample_orders":True,
        "n_train":len(manifests[0]),"n_val":len(manifests[1]),"epochs":cfg["epochs"],"test_set_used":False,
        "data_scope":cfg.get("data_scope","historical_mixed"),
        "claim":"pipeline smoke test only, not evidence of generalization"})
    print("RELATION_PILOT_TRAINING_DONE", flush=True)


if __name__ == "__main__":
    main()
