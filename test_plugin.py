from vesselfm.zarrnii_plugin import  VesselFMPlugin
from zarrnii import ZarrNii
from dask.diagnostics import ProgressBar

znimg=ZarrNii.from_ome_zarr('/nfs/trident3/lightsheet/prado/mouse_app_lecanemab_batch3/bids/sub-AS161F3/micr/sub-AS161F3_sample-brain_acq-imaris4x_SPIM.ome.zarr', level=5, channel_labels=['CD31'])
znseg = znimg.segment(VesselFMPlugin, chunk_size=(1, 128,128,128))
with ProgressBar():
    znseg.to_ome_zarr('test_out.ome.zarr',zarr_format=2)


