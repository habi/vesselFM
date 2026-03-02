"""OME-Zarr inference using ZarrNii and chunk-wise processing.

This module adapts the vesselFM inference pipeline to process large 3D OME-Zarr
volumes in a memory-efficient, blockwise manner with real-time progress tracking.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


def estimate_ram_gb(data) -> dict:
    """Estimate RAM requirements for processing a volume with map_blocks inference.

    Computes three figures:

    * **per_chunk_mb** – peak RAM to process a single chunk (input float32 +
      torch intermediate tensors + uint8 output, conservatively estimated at
      3 × input chunk size).
    * **full_output_gb** – RAM that would be needed if the *entire* segmentation
      were materialised in memory at once (old behaviour that caused 150 GB+
      usage).
    * **streaming_peak_gb** – maximum RAM actually needed when results are
      streamed to disk chunk-by-chunk (≈ per_chunk_mb converted to GB).

    Args:
        data: Dask array representing the input volume (any shape/dtype).

    Returns:
        dict with keys ``n_chunks``, ``per_chunk_mb``, ``full_output_gb``,
        ``streaming_peak_gb``.
    """
    n_chunks = int(np.prod(data.numblocks))

    # Bytes for one input chunk (float32)
    chunk_elements = int(np.prod([c[0] for c in data.chunks]))
    input_bytes = chunk_elements * np.dtype(np.float32).itemsize

    # During inference: input float32 + PyTorch intermediates + uint8 output
    # ~3× the raw input is a conservative estimate
    processing_bytes_per_chunk = input_bytes * 3

    # Full uint8 output if materialised (1 byte / voxel)
    output_bytes = int(np.prod(data.shape)) * np.dtype(np.uint8).itemsize

    return {
        "n_chunks": n_chunks,
        "per_chunk_mb": processing_bytes_per_chunk / 1e6,
        "full_output_gb": output_bytes / 1e9,
        "streaming_peak_gb": processing_bytes_per_chunk / 1e9,
    }


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
    out_path: Optional[str] = None,
) -> "ZarrNii":
    """Process an OME-Zarr volume with vesselFM chunk by chunk.

    The image is re-chunked to *chunk_size* (if given) and each chunk is
    independently segmented by :class:`VesselFMPlugin`.  Results are streamed
    directly to *out_path* on disk so that the full segmentation array is
    **never materialised in RAM** – peak memory usage stays proportional to a
    single chunk, not to the entire volume.

    Args:
        znimg: Input :class:`zarrnii.ZarrNii` object backed by a Dask array.
        model: vesselFM model (eval mode, on *device*).
        device: Torch device string.
        transforms: Pre-processing transform pipeline.
        inferer: MONAI ``SlidingWindowInfererAdapt``.
        threshold: Sigmoid threshold for binarisation (default 0.5).
        chunk_size: Optional 3-D tuple ``(D, H, W)`` controlling how the volume
            is partitioned.  ``None`` keeps the existing zarr chunk layout.
        out_path: Destination OME-Zarr path.  When provided the segmentation is
            written there directly (chunk-by-chunk) and the returned
            :class:`zarrnii.ZarrNii` is backed by that store.  When ``None``
            the lazy Dask graph is returned without writing to disk.

    Returns:
        :class:`zarrnii.ZarrNii` instance whose ``.data`` is a Dask array of
        ``uint8`` segmentation values.  If *out_path* was supplied the data is
        already persisted to that OME-Zarr store.
    """
    import dask.array as da
    from dask.diagnostics import ProgressBar
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

    # Log RAM estimate so the user knows what to expect
    ram = estimate_ram_gb(data)
    logger.info(
        f"Volume: {ram['n_chunks']} chunks. "
        f"RAM estimate – per chunk: ~{ram['per_chunk_mb']:.0f} MB, "
        f"streaming peak: ~{ram['per_chunk_mb']:.0f} MB; "
        f"full materialisation would need ~{ram['full_output_gb']:.2f} GB."
    )

    # Build a lazy computation graph with map_blocks – no full-array pre-allocation
    segmented = data.map_blocks(plugin.segment, dtype=np.uint8)

    result_znimg = znimg.copy(name=f"{znimg.name}_vesselFM_seg")
    result_znimg.data = segmented  # lazy – nothing computed yet

    if out_path is not None:
        # Stream results chunk-by-chunk directly to OME-Zarr on disk.
        # The ome-zarr-py backend uses da.to_zarr() internally so only
        # O(1 chunk) of RAM is live at any moment.
        logger.info(f"Streaming segmentation to {out_path} ...")
        with ProgressBar(dt=0.5):
            result_znimg.to_ome_zarr(str(out_path), backend="ome-zarr-py")
        logger.info("Segmentation written successfully.")
        return ZarrNii.from_ome_zarr(str(out_path))

    return result_znimg
