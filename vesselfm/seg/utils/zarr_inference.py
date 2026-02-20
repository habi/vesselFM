"""OME-Zarr inference using ZarrNii and chunk-wise processing.

This module adapts the vesselFM inference pipeline to process large 3D OME-Zarr
volumes in a memory-efficient, blockwise manner with real-time progress tracking.
"""

import itertools
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


class VesselFMPlugin:
    """ZarrNii segmentation plugin that runs vesselFM inference on each Dask block.

    Each Dask block (3-D numpy array) is independently pre-processed, run
    through the sliding-window inferer, and thresholded to produce a binary
    ``uint8`` result of the same spatial shape.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: str,
        transforms,
        inferer,
        threshold: float = 0.5,
    ):
        """
        Args:
            model: The vesselFM PyTorch model (already in eval mode on *device*).
            device: Torch device string, e.g. ``"cuda:0"`` or ``"cpu"``.
            transforms: MONAI/vesselFM pre-processing transform pipeline.
            inferer: MONAI ``SlidingWindowInfererAdapt`` instance.
            threshold: Sigmoid threshold for binarising logits (default 0.5).
        """
        self.model = model
        self.device = device
        self.transforms = transforms
        self.inferer = inferer
        self.threshold = threshold

    # ------------------------------------------------------------------
    # ZarrNii SegmentationPlugin interface
    # ------------------------------------------------------------------

    def segment(
        self,
        image: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Run vesselFM inference on a single block from a Dask map_blocks call.

        ZarrNii data has shape ``(c, z, y, x)``.  Each block may therefore be
        either 3-D ``(z, y, x)`` or 4-D ``(c, z, y, x)`` depending on the chunk
        layout.  The method squeezes / restores the channel dimension as needed.

        Args:
            image: numpy array block (3-D or 4-D float32).
            metadata: Optional metadata dict passed by ZarrNii (unused).

        Returns:
            Binary segmentation as a ``uint8`` numpy array of the same shape.
        """
        # Squeeze channel dim for the inference pipeline, track original ndim
        squeeze_channel = image.ndim == 4
        if squeeze_channel:
            img_3d = image.squeeze(0)
        else:
            img_3d = image

        with torch.no_grad():
            img_tensor = self.transforms(img_3d.astype(np.float32))[None].to(
                self.device
            )
            logits = self.inferer(img_tensor, self.model)
            pred = (logits.cpu().squeeze().sigmoid() > self.threshold).numpy()

        result = pred.astype(np.uint8)

        # Restore channel dimension so the output shape matches the input
        if squeeze_channel:
            result = result[np.newaxis, ...]

        return result

    def segmentation_plugin_name(self) -> str:
        return "VesselFM"

    def segmentation_plugin_description(self) -> str:
        return "VesselFM vessel segmentation via sliding-window inference"

    # Make the plugin usable directly without ZarrNii's pluggy machinery
    @property
    def name(self) -> str:
        return self.segmentation_plugin_name()

    @property
    def description(self) -> str:
        return self.segmentation_plugin_description()


def run_zarr_inference(
    znimg: "ZarrNii",
    model: torch.nn.Module,
    device: str,
    transforms,
    inferer,
    threshold: float = 0.5,
    chunk_size: Optional[Tuple[int, int, int]] = None,
) -> "ZarrNii":
    """Process an OME-Zarr volume with vesselFM chunk by chunk.

    The image is re-chunked to *chunk_size* (if given) and each chunk is
    independently segmented by :class:`VesselFMPlugin` with a real-time
    progress bar.

    Args:
        znimg: Input :class:`zarrnii.ZarrNii` object backed by a Dask array.
        model: vesselFM model (eval mode, on *device*).
        device: Torch device string.
        transforms: Pre-processing transform pipeline.
        inferer: MONAI ``SlidingWindowInfererAdapt``.
        threshold: Sigmoid threshold for binarisation (default 0.5).
        chunk_size: Optional 3-D tuple ``(D, H, W)`` controlling how the volume
            is partitioned.  ``None`` keeps the existing zarr chunk layout.

    Returns:
        New :class:`zarrnii.ZarrNii` instance whose ``.data`` is a Dask array
        of ``uint8`` segmentation values.
    """
    import dask.array as da
    from zarrnii import ZarrNii

    plugin = VesselFMPlugin(
        model=model,
        device=device,
        transforms=transforms,
        inferer=inferer,
        threshold=threshold,
    )

    # Rechunk if requested
    data = znimg.data
    if chunk_size is not None:
        # ZarrNii data is (c, z, y, x); prepend the channel dim to the spatial
        # chunk_size so rechunk targets the spatial dimensions only.
        if data.ndim == 4 and len(chunk_size) == 3:
            effective_chunk = (data.shape[0],) + tuple(chunk_size)
        else:
            effective_chunk = tuple(chunk_size)
        logger.info(f"Rechunking OME-Zarr data to effective_chunk={effective_chunk}")
        data = data.rechunk(effective_chunk)
    else:
        logger.info("Using existing OME-Zarr chunk layout for inference")

    n_chunks = int(np.prod(data.numblocks))
    logger.info(f"Applying vesselFM inference ({n_chunks} chunks)")

    # Pre-allocate the output array and iterate over blocks explicitly so the
    # progress bar updates in real time without putting tqdm inside a dask task.
    seg_np = np.empty(data.shape, dtype=np.uint8)

    with tqdm(total=n_chunks, desc="Processing chunks", unit="chunk") as pbar:
        for block_idx in itertools.product(*[range(n) for n in data.numblocks]):
            slices = tuple(
                slice(
                    sum(data.chunks[dim][:i]),
                    sum(data.chunks[dim][: i + 1]),
                )
                for dim, i in enumerate(block_idx)
            )
            block = data[slices].compute()
            seg_np[slices] = plugin.segment(block)
            pbar.update(1)

    segmented = da.from_array(seg_np, chunks=data.chunks)

    result_znimg = znimg.copy(name=f"{znimg.name}_vesselFM_seg")
    result_znimg.data = segmented
    return result_znimg
