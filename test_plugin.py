from vesselfm.zarrnii_plugin import  VesselFMPlugin
from zarrnii import ZarrNii

znimg=ZarrNii.from_ome_zarr('/nfs/trident3/lightsheet/prado/mouse_app_lecanemab_batch3/bids/sub-AS161F3/micr/sub-AS161F3_sample-brain_acq-imaris4x_SPIM.ome.zarr', level=5, channel_labels=['CD31'])
znimg.segment(VesselFMPlugin)
