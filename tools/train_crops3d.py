"""Crops3D text-query organ segmentation trainer (step 5)."""

print("Loading Python dependencies...", flush=True)

import argparse
import datetime
import os
import os.path as osp
import shutil
import sys
import time
from collections import defaultdict

# Prefer this repo's gres_model over editable MDIN-main (egg-link).
# Must force-front: easy-install.pth already lists both repos, so
# `not in sys.path` would skip insert and leave MDIN-main ahead.
_REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

import gorilla
import numpy as np
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

from gres_model.dataset import build_dataloader, build_dataset
from gres_model.model import MODEL
from gres_model.utils import AverageMeter, get_root_logger


def get_args():
    parser = argparse.ArgumentParser("Crops3D Organ Segmentation")
    parser.add_argument("config", type=str, help="config file path")
    parser.add_argument("--resume", type=str, help="checkpoint path to resume")
    parser.add_argument("--work_dir", type=str, help="experiment output directory")
    parser.add_argument("--skip_validate", action="store_true", help="skip validation")
    parser.add_argument("--skip_training", action="store_true", help="skip training")
    parser.add_argument("--dist", action="store_true", help="distributed training")
    parser.add_argument("--num_gpus", type=int, default=1, help="number of GPUs")
    parser.add_argument("--num_machines", type=int, default=1)
    parser.add_argument("--machine_rank", type=int, default=0)
    parser.add_argument("--dist_url", type=str, default="auto")
    parser.add_argument("--gpu_ids", type=int, default=[0], nargs="+", help="GPU ids to use")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="log data/forward/backward/step time each iter to debug stalls",
    )
    return parser.parse_args()


def _format_ann_id(ann_id):
    if isinstance(ann_id, (list, tuple)) and len(ann_id) == 1:
        return ann_id[0]
    return ann_id


def _print_worst_cases(sample_records, logger, topk=10):
    """Print lowest-IoU samples to help locate failures."""
    if not sample_records:
        return
    logger.info("-" * 80)
    logger.info(f"Worst {min(topk, len(sample_records))} samples by IoU:")
    for item in sorted(sample_records, key=lambda x: x["iou"])[:topk]:
        logger.info(
            f"  IoU={item['iou']:.4f}  spIoU={item['spiou']:.4f}  "
            f"crop={item['crop']}  organ={item['organ']}  "
            f"scene={item['scene_id']}  ann_id={item['ann_id']}"
        )


def _print_crop_organ_results(
    crop_metrics,
    organ_metrics,
    sample_records,
    summary,
    acc_50,
    acc_25,
    logger,
    miou=None,
):
    """Print mIoU / P / R / F1 grouped by crop and organ."""
    from gres_model.utils.segmentation_metrics import format_group_line, format_overall_line, macro_mean

    macro_crop_iou = macro_mean(crop_metrics, "iou")
    macro_organ_iou = macro_mean(organ_metrics, "iou")
    logger.info("=" * 80)
    logger.info(format_overall_line(summary, acc_50, acc_25, prefix="Overall", miou=miou))
    logger.info(
        f"Macro crop mIoU: {macro_crop_iou:.4f}  Macro organ mIoU: {macro_organ_iou:.4f}  "
        f"Macro crop F1: {macro_mean(crop_metrics, 'f1'):.4f}  "
        f"Macro organ F1: {macro_mean(organ_metrics, 'f1'):.4f}"
    )
    logger.info("-" * 80)
    logger.info("Per-crop metrics:")
    for crop in sorted(crop_metrics):
        logger.info(format_group_line(crop, crop_metrics[crop], width=12))
    logger.info("-" * 80)
    logger.info("Per-organ metrics:")
    for organ in sorted(organ_metrics, key=lambda x: -len(organ_metrics[x]["iou"])):
        logger.info(format_group_line(organ, organ_metrics[organ], width=16))
    _print_worst_cases(sample_records, logger)
    logger.info("=" * 80)


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _scalarize_log_vars(log_vars):
    """Unify loss tensor shapes before DDP reduce so [] and [1] do not break stack."""
    scalar_log_vars = {}
    for key, value in log_vars.items():
        if torch.is_tensor(value):
            scalar_log_vars[key] = value.reshape(-1).mean()
        else:
            scalar_log_vars[key] = torch.tensor(float(value), device="cuda" if torch.cuda.is_available() else "cpu")
    return scalar_log_vars


EARLY_STOP_METRIC_KEYS = {
    "mIOU": "miou",
    "macro_organ_mIOU": "macro_organ_miou",
    "macro_crop_mIOU": "macro_crop_miou",
    "Acc_50": "acc_50",
}


def _get_early_stop_cfg(cfg):
    """Read early-stop config; return None if disabled."""
    early_stop = getattr(cfg.train, "early_stop", None)
    if early_stop is None or not getattr(early_stop, "enabled", False):
        return None
    metric = getattr(early_stop, "metric", "mIOU")
    if metric not in EARLY_STOP_METRIC_KEYS:
        raise ValueError(
            f"Unsupported early_stop.metric={metric!r}, "
            f"choose from {list(EARLY_STOP_METRIC_KEYS)}"
        )
    return {
        "patience": int(getattr(early_stop, "patience", 10)),
        "min_delta": float(getattr(early_stop, "min_delta", 0.0)),
        "metric": metric,
    }


def _select_eval_metric(metrics, metric_name):
    """Pick the early-stop metric from the eval metrics dict."""
    return float(metrics[EARLY_STOP_METRIC_KEYS[metric_name]])


def _broadcast_stop_flag(should_stop, dist_enabled):
    """Broadcast the early-stop flag from rank0 so all ranks exit together."""
    if not dist_enabled or not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return should_stop
    stop_tensor = torch.tensor(int(should_stop), device="cuda", dtype=torch.int32)
    torch.distributed.broadcast(stop_tensor, src=0)
    return bool(stop_tensor.item())


def train(epoch, model, dataloader, optimizer, lr_scheduler, cfg, logger, writer, debug=False, start_iter=1):
    """Train one epoch."""
    model.train()
    iter_time = AverageMeter()
    data_time = AverageMeter()
    meter_dict = {}
    end = time.time()
    save_iter = int(getattr(cfg.train, "save_iter", 0) or 0)

    if dataloader.sampler is not None and cfg.dist:
        dataloader.sampler.set_epoch(epoch)

    for i, batch in enumerate(dataloader, start=1):
        if i < start_iter:
            continue
        t_iter_start = time.time()
        data_time.update(t_iter_start - end)

        scan_ids = batch.get("scan_ids", ["?"])
        meta = batch.get("meta_datas", [{}])
        crop = meta[0].get("crop", "") if meta else ""
        organ = meta[0].get("object_name", "") if meta else ""
        scan_tag = f"{scan_ids[0] if scan_ids else '?'}|{crop}|{organ}"

        if gorilla.is_main_process() and (debug or i % 10 == 1):
            logger.info(
                f"Epoch [{epoch}/{cfg.train.epochs}][{i}/{len(dataloader)}] start  "
                f"scan={scan_tag}"
            )

        t_fwd_start = time.time()
        try:
            loss, log_vars = model(batch, mode="loss")
            _cuda_sync()
        except Exception as exc:
            if gorilla.is_main_process():
                logger.warning(
                    f"Skip iter {i} scan={scan_tag}: {type(exc).__name__}: {exc}"
                )
            end = time.time()
            continue
        t_fwd_end = time.time()

        log_vars = gorilla.reduce_dict(_scalarize_log_vars(log_vars), average=True)
        if gorilla.is_main_process():
            for k, v in log_vars.items():
                if k not in meter_dict:
                    meter_dict[k] = AverageMeter()
                meter_dict[k].update(v.item())

        optimizer.zero_grad()
        if not torch.isfinite(loss):
            if gorilla.is_main_process():
                logger.warning(f"Skip non-finite loss at epoch {epoch} iter {i}")
            continue

        t_bwd_start = time.time()
        loss.backward()
        _cuda_sync()
        t_bwd_end = time.time()

        grad_total_norm = 0.0
        if cfg.train.grad_clip > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.train.grad_clip
            )
        optimizer.step()
        _cuda_sync()
        t_step_end = time.time()

        # WSL/CUDA long runs can hang; periodically free unused VRAM fragments
        if i % 500 == 0 and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        if save_iter > 0 and i % save_iter == 0 and gorilla.is_main_process():
            save_file = osp.join(cfg.work_dir, "last_iter.pth")
            meta = dict(epoch=epoch, iter=i)
            gorilla.save_checkpoint(model, save_file, optimizer, lr_scheduler, meta)
            logger.info(f"Saved iter checkpoint: {save_file} (epoch={epoch}, iter={i})")

        if debug and gorilla.is_main_process():
            logger.info(
                "[debug] iter=%d/%d scan=%s crop=%s organ=%s "
                "data=%.3fs forward=%.3fs backward=%.3fs step=%.3fs total=%.3fs loss=%.4f",
                i,
                len(dataloader),
                scan_ids[0] if scan_ids else "?",
                crop,
                organ,
                t_fwd_start - t_iter_start,
                t_fwd_end - t_fwd_start,
                t_bwd_end - t_bwd_start,
                t_step_end - t_bwd_end,
                t_step_end - t_iter_start,
                loss.item(),
            )

        remain_iter = len(dataloader) * (cfg.train.epochs - epoch + 1) - i
        iter_time.update(time.time() - end)
        end = time.time()
        remain_time = str(datetime.timedelta(seconds=int(remain_iter * iter_time.avg)))
        lr = optimizer.param_groups[0]["lr"]

        if i % 10 == 0 and gorilla.is_main_process():
            log_str = (
                f"Epoch [{epoch}/{cfg.train.epochs}][{i}/{len(dataloader)}]  "
                f"scan={scan_tag}, lr: {lr:.2g}, eta: {remain_time}, "
                f"data_time: {data_time.avg:.2f}, iter_time: {iter_time.avg:.2f}"
            )
            for k, v in meter_dict.items():
                log_str += f", {k}: {v.val:.4f}"
                writer.add_scalar(f"train/{k}", v.val, i + (epoch - 1) * len(dataloader))
            if cfg.train.grad_clip > 0:
                writer.add_scalar("train/grad_total_norm", grad_total_norm, i + (epoch - 1) * len(dataloader))
            logger.info(log_str)
            writer.add_scalar("train/learning_rate", lr, i + (epoch - 1) * len(dataloader))

    gorilla.synchronize()
    lr_scheduler.step()

    if gorilla.is_main_process():
        save_file = osp.join(cfg.work_dir, "last.pth")
        meta = dict(epoch=epoch, iter=len(dataloader))
        gorilla.save_checkpoint(model, save_file, optimizer, lr_scheduler, meta)


@torch.no_grad()
def eval(epoch, best_miou, model, dataloader, cfg, logger, writer, save_ckp=False):
    """Validate: mIoU by crop and organ."""
    if gorilla.is_main_process():
        logger.info("Validation")
        pious, spious, meta_datas, sample_metrics = [], [], [], []
        progress_bar = tqdm(total=len(dataloader))

    from gres_model.utils.segmentation_metrics import (
        METRIC_NAMES,
        acc_at_threshold,
        aggregate_group_metrics,
        format_overall_line,
        macro_mean,
        metrics_from_predict_result,
        overall_summary,
    )

    model.eval()
    for batch in dataloader:
        result = model(batch, mode="predict")
        if gorilla.is_main_process():
            pious.extend(result["piou"])
            spious.extend(result["spiou"])
            sample_metrics.extend(metrics_from_predict_result(result))
            if "meta_datas" in result:
                meta_datas.extend(result["meta_datas"])
            progress_bar.update()

    metrics = None
    if gorilla.is_main_process():
        progress_bar.close()
        pious = torch.stack(pious, dim=0).cpu().numpy()
        spious = torch.stack(spious, dim=0).cpu().numpy()

        summary = overall_summary(sample_metrics)
        miou = float(pious.mean())
        spmiou = float(spious.mean())
        acc_50 = acc_at_threshold(pious, 0.5)
        acc_25 = acc_at_threshold(pious, 0.25)
        sp_acc_50 = acc_at_threshold(spious, 0.5)
        sp_acc_25 = acc_at_threshold(spious, 0.25)

        writer.add_scalar("val/mIOU", miou, epoch)
        writer.add_scalar("val/IoU", miou, epoch)
        writer.add_scalar("val/Precision", summary["precision"], epoch)
        writer.add_scalar("val/Recall", summary["recall"], epoch)
        writer.add_scalar("val/F1", summary["f1"], epoch)
        writer.add_scalar("val/Acc_50", acc_50, epoch)
        writer.add_scalar("val/Acc_25", acc_25, epoch)
        writer.add_scalar("val/spmIOU", spmiou, epoch)
        writer.add_scalar("val/spAcc_50", sp_acc_50, epoch)
        writer.add_scalar("val/spAcc_25", sp_acc_25, epoch)

        crop_metrics = aggregate_group_metrics(sample_metrics, meta_datas, "crop")
        organ_metrics = aggregate_group_metrics(sample_metrics, meta_datas, "object_name")
        macro_crop_miou = macro_mean(crop_metrics, "iou")
        macro_organ_miou = macro_mean(organ_metrics, "iou")
        sample_records = []
        for idx, meta in enumerate(meta_datas):
            sample_records.append(
                {
                    "iou": float(sample_metrics[idx]["iou"]),
                    "precision": float(sample_metrics[idx]["precision"]),
                    "recall": float(sample_metrics[idx]["recall"]),
                    "f1": float(sample_metrics[idx]["f1"]),
                    "spiou": float(spious[idx]),
                    "crop": meta.get("crop", "unknown"),
                    "organ": meta.get("object_name", "unknown"),
                    "scene_id": meta.get("scene_id", "?"),
                    "ann_id": _format_ann_id(meta.get("ann_id", "?")),
                }
            )

        writer.add_scalar("val/macro_crop_mIOU", macro_crop_miou, epoch)
        writer.add_scalar("val/macro_organ_mIOU", macro_organ_miou, epoch)
        for crop, group in crop_metrics.items():
            writer.add_scalar(f"val/crop_{crop}_IoU", group["iou"].mean(), epoch)
            writer.add_scalar(f"val/crop_{crop}_Precision", group["precision"].mean(), epoch)
            writer.add_scalar(f"val/crop_{crop}_Recall", group["recall"].mean(), epoch)
            writer.add_scalar(f"val/crop_{crop}_F1", group["f1"].mean(), epoch)
            writer.add_scalar(f"val/crop_{crop}_Acc_50", acc_at_threshold(group["iou"], 0.5), epoch)
            writer.add_scalar(f"val/crop_{crop}_Acc_25", acc_at_threshold(group["iou"], 0.25), epoch)
        for organ, group in organ_metrics.items():
            writer.add_scalar(f"val/organ_{organ}_IoU", group["iou"].mean(), epoch)
            writer.add_scalar(f"val/organ_{organ}_Precision", group["precision"].mean(), epoch)
            writer.add_scalar(f"val/organ_{organ}_Recall", group["recall"].mean(), epoch)
            writer.add_scalar(f"val/organ_{organ}_F1", group["f1"].mean(), epoch)
            writer.add_scalar(f"val/organ_{organ}_Acc_50", acc_at_threshold(group["iou"], 0.5), epoch)
            writer.add_scalar(f"val/organ_{organ}_Acc_25", acc_at_threshold(group["iou"], 0.25), epoch)

        logger.info(format_overall_line(summary, acc_50, acc_25, prefix="Val", miou=miou))
        logger.info(
            f"mIoU: {miou:.4f}  Acc@0.5: {acc_50:.4f}  Acc@0.25: {acc_25:.4f}  "
            f"P: {summary['precision']:.4f}  R: {summary['recall']:.4f}  F1: {summary['f1']:.4f}"
        )
        logger.info(f"spmIoU: {spmiou:.4f}  spAcc@0.5: {sp_acc_50:.4f}")
        _print_crop_organ_results(
            crop_metrics, organ_metrics, sample_records, summary, acc_50, acc_25, logger, miou=miou
        )

        if save_ckp:
            if miou > best_miou:
                save_file = osp.join(cfg.work_dir, "best.pth")
                gorilla.save_checkpoint(model, save_file)
                logger.info(f"Saved best checkpoint: {save_file}")
            elif epoch > cfg.train.save_epoch:
                save_file = osp.join(cfg.work_dir, f"epoch_{epoch:04d}.pth")
                gorilla.save_checkpoint(model, save_file)

        metrics = {
            "miou": miou,
            "iou": miou,
            "precision": summary["precision"],
            "recall": summary["recall"],
            "f1": summary["f1"],
            "macro_organ_miou": macro_organ_miou,
            "macro_crop_miou": macro_crop_miou,
            "acc_50": acc_50,
            "acc_25": acc_25,
        }

    return metrics


def main(args):
    cfg = gorilla.Config.fromfile(args.config)
    cfg.dist = args.dist

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    cfg.work_dir = args.work_dir or osp.join(
        "./exps", osp.splitext(osp.basename(args.config))[0], timestamp
    )
    os.makedirs(osp.abspath(cfg.work_dir), exist_ok=True)

    log_file = osp.join(cfg.work_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file) if gorilla.is_main_process() else None
    if logger:
        logger.info(f"config: {args.config}")
    dst_cfg = osp.join(cfg.work_dir, osp.basename(args.config))
    if osp.abspath(args.config) != osp.abspath(dst_cfg):
        shutil.copy(args.config, dst_cfg)
    writer = SummaryWriter(cfg.work_dir)

    gorilla.set_random_seed(cfg.train.seed)
    model = MODEL(**cfg.model).cuda()

    if (not args.resume) and cfg.train.pretrain:
        if logger:
            logger.info(f"Load pretrain from {cfg.train.pretrain}")
        gorilla.load_checkpoint(model, cfg.train.pretrain, strict=False, map_location="cpu")

    if args.num_gpus > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[gorilla.get_local_rank()], find_unused_parameters=True
        )

    if gorilla.is_main_process():
        count = gorilla.parameter_count(model)[""]
        logger.info(f"Parameters: {count / 1e6:.2f}M")

    optimizer_model = model.module if hasattr(model, "module") else model
    optimizer = gorilla.build_optimizer(optimizer_model, cfg.optimizer)
    lr_scheduler = gorilla.build_lr_scheduler(optimizer, cfg.lr_scheduler)

    start_epoch = 1
    start_iter = 1
    if args.resume:
        if logger:
            logger.info(f"Resume from {args.resume}")
        meta = gorilla.resume(
            model=model,
            filename=args.resume,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            strict=False,
            map_location="cpu",
        )
        start_epoch = meta.get("epoch", 1)
        start_iter = meta.get("iter", 0) + 1
        gorilla.set_random_seed(cfg.train.seed)
        if logger:
            logger.info(f"Resume at epoch={start_epoch}, iter={start_iter}")
    cfg.dataloader.train.batch_size = max(1, int(cfg.dataloader.train.batch_size / args.num_gpus))
    if logger:
        logger.info(f"Train batch size per gpu: {cfg.dataloader.train.batch_size}")

    train_loader = val_loader = None
    if not args.skip_training:
        train_dataset = build_dataset(cfg.data.train, logger)
        if args.debug:
            train_dataset.debug_data = True
            if logger:
                logger.info("Debug on: log data/forward/backward/step time each iter")
        train_loader = build_dataloader(train_dataset, dist=args.dist, **cfg.dataloader.train)
        # last.pth stores epoch=N, iter=len after an epoch; if resume iter>len, advance to next epoch
        if args.resume and start_iter > len(train_loader):
            if logger:
                logger.info(
                    f"Resume iter={start_iter} exceeds epoch length={len(train_loader)}; "
                    f"advance to epoch={start_epoch + 1}, iter=1"
                )
            start_epoch += 1
            start_iter = 1
    if not args.skip_validate and gorilla.is_main_process():
        val_dataset = build_dataset(cfg.data.val, logger)
        val_loader = build_dataloader(val_dataset, **cfg.dataloader.val)

    if logger:
        logger.info("Training Crops3D organ segmentation")
    early_stop_cfg = _get_early_stop_cfg(cfg)
    if logger and early_stop_cfg:
        logger.info(
            "Early stopping enabled: metric=%s, patience=%d, min_delta=%.4f "
            "(counted on consecutive validations, interval=%d epoch(s))",
            early_stop_cfg["metric"],
            early_stop_cfg["patience"],
            early_stop_cfg["min_delta"],
            cfg.train.interval,
        )
    best_miou = 0.0
    best_monitor = 0.0
    no_improve = 0
    for epoch in range(start_epoch, cfg.train.epochs + 1):
        if not args.skip_training:
            epoch_start_iter = start_iter if epoch == start_epoch else 1
            train(
                epoch,
                model,
                train_loader,
                optimizer,
                lr_scheduler,
                cfg,
                logger,
                writer,
                args.debug,
                start_iter=epoch_start_iter,
            )
            start_iter = 1
        should_stop = False
        if not args.skip_validate and epoch % cfg.train.interval == 0:
            if gorilla.is_main_process():
                metrics = eval(epoch, best_miou, model, val_loader, cfg, logger, writer, not args.skip_training)
                if metrics is not None:
                    if metrics["miou"] > best_miou:
                        best_miou = metrics["miou"]
                    if early_stop_cfg is not None:
                        current = _select_eval_metric(metrics, early_stop_cfg["metric"])
                        if current > best_monitor + early_stop_cfg["min_delta"]:
                            best_monitor = current
                            no_improve = 0
                        else:
                            no_improve += 1
                            logger.info(
                                "Early stop: no improvement for %d/%d validation(s), "
                                "%s=%.4f, best=%.4f",
                                no_improve,
                                early_stop_cfg["patience"],
                                early_stop_cfg["metric"],
                                current,
                                best_monitor,
                            )
                        if no_improve >= early_stop_cfg["patience"]:
                            logger.info(
                                "Early stopping triggered at epoch %d "
                                "(best %s=%.4f)",
                                epoch,
                                early_stop_cfg["metric"],
                                best_monitor,
                            )
                            should_stop = True
            should_stop = _broadcast_stop_flag(should_stop, cfg.dist)
        writer.flush()
        if args.skip_training:
            break
        if should_stop:
            break

    # Auto-plot curves after training (main process, full train run only)
    if gorilla.is_main_process() and not args.skip_training:
        writer.close()
        try:
            from gres_model.utils.training_curves import save_training_curves

            curve_paths = save_training_curves(cfg.work_dir, log_file=log_file)
            if logger:
                logger.info("Training curves saved:")
                logger.info(f"  {curve_paths['training_curves']}")
                logger.info(f"  {curve_paths['training_curves_detail']}")
                logger.info(f"  {curve_paths['training_summary']}")
                if curve_paths.get("loss_total_curve"):
                    logger.info(f"  total loss curve: {curve_paths['loss_total_curve']}")
                if curve_paths.get("loss_all_curves"):
                    logger.info(f"  all-loss overview: {curve_paths['loss_all_curves']}")
                if curve_paths.get("loss_curves_dir"):
                    logger.info(f"  per-loss curve dir: {curve_paths['loss_curves_dir']}")
                if curve_paths.get("summary"):
                    logger.info(f"  best val mIoU: {curve_paths['summary'].get('best_val_miou', 'N/A')}")
        except Exception as exc:
            if logger:
                logger.warning(f"Failed to plot training curves (run tools/plot_training_curves.py): {exc}")


if __name__ == "__main__":
    args = get_args()
    if args.num_gpus > 1:
        args.dist = True
    gorilla.set_cuda_visible_devices(gpu_ids=args.gpu_ids, num_gpu=args.num_gpus)
    gorilla.launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
