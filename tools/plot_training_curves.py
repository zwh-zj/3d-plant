"""Rebuild training curves from an experiment dir (also called by the trainer)."""

import argparse
import glob
import os.path as osp


def main():
    parser = argparse.ArgumentParser("Generate Crops3D training curves")
    parser.add_argument(
        "exp_dir",
        type=str,
        help="experiment dir, e.g. exps/default_crops3d/20260621_230618",
    )
    parser.add_argument("--log_file", type=str, default=None, help="training log path; default exp_dir/*.log")
    parser.add_argument("--output_dir", type=str, default=None, help="output dir; default same as exp_dir")
    args = parser.parse_args()

    log_file = args.log_file
    if log_file is None:
        logs = sorted(glob.glob(osp.join(args.exp_dir, "*.log")))
        log_file = logs[0] if logs else None

    from gres_model.utils.training_curves import save_training_curves

    result = save_training_curves(args.exp_dir, log_file=log_file, output_dir=args.output_dir)
    print("Wrote:")
    print(" ", result["training_curves"])
    print(" ", result["training_curves_detail"])
    print(" ", result["training_summary"])
    if result["summary"]:
        print("Summary:", result["summary"])


if __name__ == "__main__":
    main()
