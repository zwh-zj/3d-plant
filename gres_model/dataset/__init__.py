from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

from .crops3d_organ import Crops3DOrganDataset
from .scannetv2_3dgres import ScanNetDataset_sample_graph_edge


__all__ = [
    'Crops3DOrganDataset',
    'ScanNetDataset_sample_graph_edge',
    'build_dataset',
    'build_dataloader',
]


def build_dataset(data_cfg, logger):
    assert 'type' in data_cfg
    _data_cfg = data_cfg.copy()
    _data_cfg['logger'] = logger
    data_type = _data_cfg.pop('type')
    if data_type == 'scannetv2_sample_graph_edge':
        return ScanNetDataset_sample_graph_edge(**_data_cfg)
    if data_type == 'crops3d_organ':
        return Crops3DOrganDataset(**_data_cfg)
    raise ValueError(f'Unknown {data_type}')


def build_dataloader(dataset, batch_size=1, num_workers=1, training=True, dist=False, persistent_workers=True):
    shuffle = training
    sampler = DistributedSampler(dataset, shuffle=shuffle) if dist else None
    if sampler is not None:
        shuffle = False

    # Sparse-organ weights: train and non-DDP only; keep DistributedSampler under DDP.
    use_sparse_weight = False
    if (
        training
        and not dist
        and hasattr(dataset, "sparse_organ")
        and hasattr(dataset, "get_sample_weights")
    ):
        cfg = getattr(dataset, "sparse_organ", None) or {}
        use_sparse_weight = bool(cfg.get("enabled")) and bool(cfg.get("sampler_weighting"))

    if use_sparse_weight:
        weights = dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(dataset),
            replacement=True,
        )
        shuffle = False
    elif dist and training and hasattr(dataset, "sparse_organ"):
        cfg = dataset.sparse_organ or {}
        if cfg.get("enabled") and cfg.get("sampler_weighting"):
            logger = getattr(dataset, "logger", None)
            if logger is not None:
                logger.warning(
                    "sparse_organ.sampler_weighting skipped under DistributedSampler (DDP)"
                )

    if training:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=dataset.collate_fn,
            shuffle=shuffle,
            sampler=sampler,
            drop_last=True,
            pin_memory=True,
            persistent_workers=persistent_workers)
    else:
        assert batch_size == 1
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=dataset.collate_fn,
            shuffle=False,
            sampler=sampler,
            drop_last=False,
            pin_memory=True,
            persistent_workers=persistent_workers)
