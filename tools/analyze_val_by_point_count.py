"""Compare two checkpoints on val/test buckets by point_count / organ."""

import argparse
import json
from collections import defaultdict

import gorilla
import numpy as np
import torch
from tqdm import tqdm

from gres_model.dataset import build_dataloader, build_dataset
from gres_model.model import MODEL
from gres_model.utils import get_root_logger


def bucket_label(p):
    if p < 2000:
        return "<2000 (ROI trigger)"
    if p < 10000:
        return "2000-9999"
    return ">=10000"


@torch.no_grad()
def eval_split(cfg, checkpoint, gpu_ids, split="val"):
    gorilla.set_cuda_visible_devices(gpu_ids=gpu_ids, num_gpu=1)
    logger = get_root_logger()
    logger.info(f"Evaluating {checkpoint} on {split}")

    model = MODEL(**cfg.model).cuda()
    gorilla.load_checkpoint(model, checkpoint, strict=False)
    model.eval()

    data_cfg = getattr(cfg.data, split)
    loader_cfg = getattr(cfg.dataloader, split)
    dataset = build_dataset(data_cfg, logger)
    dataloader = build_dataloader(dataset, training=False, **loader_cfg)

    records = []
    for batch in tqdm(dataloader, desc=checkpoint):
        result = model(batch, mode="predict")
        pious = result["piou"]
        if torch.is_tensor(pious):
            pious = pious.cpu().numpy()
        else:
            pious = np.asarray(pious)
        for idx, meta in enumerate(result["meta_datas"]):
            pc = int(meta.get("point_count", 0) or 0)
            records.append(
                {
                    "iou": float(pious[idx]),
                    "organ": meta.get("object_name", "unknown"),
                    "point_count": pc,
                    "bucket": bucket_label(pc),
                    "scene_id": meta.get("scene_id", ""),
                    "ann_id": meta.get("ann_id"),
                }
            )
    return records


def summarize(records):
    out = {}
    groups = defaultdict(list)
    organ_bucket = defaultdict(list)
    organ_all = defaultdict(list)
    for r in records:
        groups[r["bucket"]].append(r["iou"])
        organ_bucket[(r["organ"], r["bucket"])].append(r["iou"])
        organ_all[r["organ"]].append(r["iou"])

    out["overall_miou"] = float(np.mean([r["iou"] for r in records]))
    out["by_bucket"] = {
        k: {"n": len(v), "miou": float(np.mean(v))} for k, v in sorted(groups.items())
    }
    out["by_organ"] = {
        k: {"n": len(v), "miou": float(np.mean(v))} for k, v in sorted(organ_all.items())
    }
    out["by_organ_bucket"] = {
        f"{organ}|{bucket}": {"n": len(v), "miou": float(np.mean(v))}
        for (organ, bucket), v in sorted(organ_bucket.items())
    }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--baseline", type=str, required=True)
    parser.add_argument("--sparse", type=str, required=True)
    parser.add_argument("--gpu_ids", type=int, default=[2], nargs="+")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    cfg = gorilla.Config.fromfile(args.config)
    base_records = eval_split(cfg, args.baseline, args.gpu_ids, args.split)
    sparse_records = eval_split(cfg, args.sparse, args.gpu_ids, args.split)

    base_sum = summarize(base_records)
    sparse_sum = summarize(sparse_records)

    print(f"\n=== Overall {args.split} mIoU ===")
    print(f"baseline (no sparse ROI): {base_sum['overall_miou']:.4f}")
    print(f"sparse ROI run:           {sparse_sum['overall_miou']:.4f}")
    print(f"delta:                    {sparse_sum['overall_miou'] - base_sum['overall_miou']:+.4f}")

    print("\n=== buckets by point_count ===")
    all_buckets = sorted(set(base_sum["by_bucket"]) | set(sparse_sum["by_bucket"]))
    for b in all_buckets:
        b0 = base_sum["by_bucket"].get(b, {"n": 0, "miou": float("nan")})
        b1 = sparse_sum["by_bucket"].get(b, {"n": 0, "miou": float("nan")})
        d = b1["miou"] - b0["miou"] if b0["n"] and b1["n"] else float("nan")
        print(
            f"  {b:18s}  baseline={b0['miou']:.4f} (n={b0['n']:3d})  "
            f"sparse={b1['miou']:.4f} (n={b1['n']:3d})  delta={d:+.4f}"
        )

    print("\n=== key organs x buckets (stem/soil/tassel) ===")
    focus_organs = ["stem", "soil", "tassel"]
    focus_buckets = ["<2000 (ROI trigger)", "2000-9999", ">=10000"]
    for organ in focus_organs:
        print(f"  [{organ}]")
        for b in focus_buckets:
            key = f"{organ}|{b}"
            b0 = base_sum["by_organ_bucket"].get(key, {"n": 0, "miou": float("nan")})
            b1 = sparse_sum["by_organ_bucket"].get(key, {"n": 0, "miou": float("nan")})
            if b0["n"] == 0 and b1["n"] == 0:
                continue
            d = b1["miou"] - b0["miou"] if b0["n"] and b1["n"] else float("nan")
            print(
                f"    {b:18s}  baseline={b0['miou']:.4f} (n={b0['n']:3d})  "
                f"sparse={b1['miou']:.4f} (n={b1['n']:3d})  delta={d:+.4f}"
            )

    print("\n=== overall by organ ===")
    all_organs = sorted(set(base_sum["by_organ"]) | set(sparse_sum["by_organ"]))
    for organ in all_organs:
        b0 = base_sum["by_organ"].get(organ, {"n": 0, "miou": float("nan")})
        b1 = sparse_sum["by_organ"].get(organ, {"n": 0, "miou": float("nan")})
        d = b1["miou"] - b0["miou"] if b0["n"] and b1["n"] else float("nan")
        print(
            f"  {organ:16s}  baseline={b0['miou']:.4f} (n={b0['n']:3d})  "
            f"sparse={b1['miou']:.4f} (n={b1['n']:3d})  delta={d:+.4f}"
        )

    if args.out:
        payload = {
            "baseline_ckpt": args.baseline,
            "sparse_ckpt": args.sparse,
            "baseline": base_sum,
            "sparse": sparse_sum,
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
