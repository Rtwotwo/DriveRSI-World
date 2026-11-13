"""Geometry checks prevent invalid overlap metrics from appearing to pass."""
import numpy as np
from driversi_world.world_models.hyperflow.diagnostics import overlap_map,overlap_metrics,image_metrics


def camera(tx=0):
    transform=np.eye(4); transform[0,3]=tx
    return dict(K=[[40,0,32],[0,40,24],[0,0,1]],distortion=[0,0,0,0,0],camera_to_ego=transform.tolist())


def test_identity_and_occlusion():
    depth=np.full((48,64),10,np.float32)
    uv,mask=overlap_map(depth,depth,camera(),camera())
    assert mask.sum()>100
    image=np.random.default_rng(0).integers(0,256,(48,64,3),dtype=np.uint8)
    assert overlap_metrics(image,image,uv,mask)['rgb_mae']==0
    _,hidden=overlap_map(depth,depth*2,camera(),camera())
    assert hidden.sum()==0


def test_translated_pinhole_geometry():
    depth=np.full((48,64),10,np.float32)
    uv,mask=overlap_map(depth,depth,camera(),camera(1))
    yy,xx=np.mgrid[:48,:64]
    assert np.allclose(uv[...,0][mask],xx[mask]-4,atol=1e-5)
    assert np.allclose(uv[...,1][mask],yy[mask],atol=1e-5)


def test_identical_image_ssim():
    image=np.random.default_rng(2).integers(0,256,(48,64,3),dtype=np.uint8)
    metrics=image_metrics(image,image)
    assert metrics['rgb_mae']==0 and abs(metrics['luma_ssim']-1)<1e-5
