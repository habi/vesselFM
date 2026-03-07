import torch
import torch.nn.functional as F
import hydra
import numpy as np
from monai.inferers import SlidingWindowInfererAdapt
from pathlib import Path
from vesselfm.seg.utils.data import generate_transforms

from zarrnii_plugin_api import hookimpl


import logging
from typing import Any, Dict, Optional, Tuple


logger = logging.getLogger(__name__)


def create_config():
    """
    Create a Hydra-composed configuration.

    Returns:
        DictConfig: A configuration object compatible with ``hydra.utils.instantiate``.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    config_dir = Path(__file__).parent.parent / "seg" / "configs"
    if not config_dir.exists():
        raise FileNotFoundError(f"Config dir not found: {config_dir}")


    # If create_config could be called multiple times in the same process (tests, notebooks),
    # Hydra needs to be cleared before re-initializing.
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(config_name="inference_zarrnii")

    return cfg

def load_model(cfg, device):
    try:
        logger.info(f"Loading model from {cfg.ckpt_path}.")
        ckpt = torch.load(Path(cfg.ckpt_path), map_location=device, weights_only=True)
    except:
        logger.info(f"Loading model from Hugging Face.")
        hf_hub_download(repo_id='bwittmann/vesselFM', filename='meta.yaml') # required to track downloads
        ckpt = torch.load(
            hf_hub_download(repo_id='bwittmann/vesselFM', filename='vesselFM_base.pt'),
            map_location=device, weights_only=True
        )

    model = hydra.utils.instantiate(cfg.model)
    model.load_state_dict(ckpt)
    return model



class VesselFMPlugin:
    """ZarrNii segmentation plugin that runs vesselFM inference on each Dask block.

    Each Dask block (3-D numpy array) is independently pre-processed, run
    through the sliding-window inferer, and thresholded to produce a binary
    ``uint8`` result of the same spatial shape.
    """

    @hookimpl
    def __init__(
        self,
        device: str = "cuda:0",
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

        cfg = create_config()

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

    
        # init sliding window inferer
        logger.debug(f"Sliding window patch size: {cfg.patch_size}")
        logger.debug(f"Sliding window batch size: {cfg.batch_size}.")
        logger.debug(f"Sliding window overlap: {cfg.overlap}.")
        inferer = SlidingWindowInfererAdapt(
            roi_size=cfg.patch_size, sw_batch_size=cfg.batch_size, overlap=cfg.overlap, 
            mode=cfg.mode, sigma_scale=cfg.sigma_scale, padding_mode=cfg.padding_mode
        )

        self.model = model
        self.device = device
        self.transforms = transforms
        self.inferer = inferer
        self.threshold = threshold



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


