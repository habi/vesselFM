import torch
import torch.nn.functional as F
import hydra
import numpy as np
from monai.inferers import SlidingWindowInfererAdapt

from zarrnii.plugins import SegmentationPlugin
from zarrnii.plugins.segmentation.base import hookimpl

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)



class VesselFMPlugin(SegmentationPlugin):
    """ZarrNii segmentation plugin that runs vesselFM inference on each Dask block.

    Each Dask block (3-D numpy array) is independently pre-processed, run
    through the sliding-window inferer, and thresholded to produce a binary
    ``uint8`` result of the same spatial shape.
    """

    def __init__(
        self,
        device: str,
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
        # seed libraries
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

        # set device
        logger.info(f"Using device {cfg.device}.")
        device = cfg.device

        # load model and ckpt
        model = load_model(cfg, device)
        model.to(device)
        model.eval()
       

        # init pre-processing transforms
        transforms = generate_transforms(cfg.transforms_config)

     
        self.model = model
        self.device = device
        self.transforms = transforms
        self.inferer = inferer
        self.threshold = threshold


        # init sliding window inferer
        logger.debug(f"Sliding window patch size: {cfg.patch_size}")
        logger.debug(f"Sliding window batch size: {cfg.batch_size}.")
        logger.debug(f"Sliding window overlap: {cfg.overlap}.")
        inferer = SlidingWindowInfererAdapt(
            roi_size=cfg.patch_size, sw_batch_size=cfg.batch_size, overlap=cfg.overlap, 
            mode=cfg.mode, sigma_scale=cfg.sigma_scale, padding_mode=cfg.padding_mode
        )

    @hookimpl
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

    @hookimpl
    def segmentation_plugin_name(self) -> str:
        return "VesselFM"

    @hookimpl
    def segmentation_plugin_description(self) -> str:
        return "VesselFM vessel segmentation via sliding-window inference"

    # Make the plugin usable directly without ZarrNii's pluggy machinery
#    @property
#    def name(self) -> str:
#        return self.segmentation_plugin_name()

#    @property
#    def description(self) -> str:
#        return self.segmentation_plugin_description()


