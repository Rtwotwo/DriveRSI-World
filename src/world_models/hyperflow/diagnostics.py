"""Paired image diagnostics and fixed-geometry overlap tests, not geometry certification."""
import cv2
import numpy as np


def gray(image):
    return cv2.cvtColor(image.astype(np.float32)/255,cv2.COLOR_RGB2GRAY)


def image_metrics(image, reference):
    """Report detail energy separately from fidelity; artifacts can inflate sharpness."""
    a,b=gray(image),gray(reference)
    def blur(x): return cv2.GaussianBlur(x,(11,11),1.5)
    ma,mb=blur(a),blur(b); va=blur(a*a)-ma*ma; vb=blur(b*b)-mb*mb
    covariance=blur(a*b)-ma*mb
    ssim=((2*ma*mb+.01**2)*(2*covariance+.03**2))/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2))
    gx=cv2.Sobel(a,cv2.CV_32F,1,0); gy=cv2.Sobel(a,cv2.CV_32F,0,1)
    return dict(rgb_mae=float(np.abs(image.astype(np.float32)-reference).mean()/255),
        luma_ssim=float(ssim[5:-5,5:-5].mean()),
        laplacian_energy=float(cv2.Laplacian(a,cv2.CV_32F).var()),
        gradient_energy=float((gx*gx+gy*gy).mean()))


def overlap_map(source_depth,target_depth,source,target):
    """Project distorted source rays into the target; reject depth-inconsistent/edge pixels.

    Gaussian expected depth is approximate, so these are fixed-mask photometric
    diagnostics. Passing them is not proof of generated 3D consistency.
    """
    h,w=source_depth.shape
    yy,xx=np.mgrid[:h,:w].astype(np.float32)
    rays=cv2.undistortPoints(np.stack((xx,yy),-1).reshape(-1,1,2),
        np.array(source['K']),np.array(source['distortion'])).reshape(h,w,2)
    xyz=np.dstack((rays,np.ones((h,w))))*source_depth[...,None]
    transform=np.linalg.inv(np.array(target['camera_to_ego']))@np.array(source['camera_to_ego'])
    mapped=xyz@transform[:3,:3].T+transform[:3,3]
    uv,_=cv2.projectPoints(mapped.reshape(-1,1,3),np.zeros(3),np.zeros(3),
        np.array(target['K']),np.array(target['distortion']))
    # Near-zero/negative target z can project far outside the sensor. Bound these
    # rejected coordinates before float32 conversion and OpenCV remapping.
    uv=np.nan_to_num(uv,nan=-1e6,posinf=1e6,neginf=-1e6)
    uv=np.clip(uv,-1e6,1e6).reshape(h,w,2).astype(np.float32)
    target_z=cv2.remap(target_depth,uv[...,0],uv[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    z=mapped[...,2]
    valid=np.isfinite(source_depth)&(source_depth>1)&(source_depth<80)&(z>1)&(z<80)
    valid&=(uv[...,0]>3)&(uv[...,0]<w-4)&(uv[...,1]>3)&(uv[...,1]<h-4)
    valid&=(target_z>1)&(np.abs(target_z-z)<(.5+.03*z))
    # Avoid silhouette mixing, distortion borders and Gaussian-depth discontinuities.
    valid=cv2.erode(valid.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
    return uv,valid


def overlap_metrics(source_image,target_image,uv,mask):
    warped=cv2.remap(target_image,uv[...,0],uv[...,1],cv2.INTER_LINEAR)
    if mask.sum()<100: return dict(valid_pixels=int(mask.sum()),valid_fraction=float(mask.mean()),rgb_mae=None)
    error=np.abs(source_image.astype(np.float32)-warped)/255
    return dict(valid_pixels=int(mask.sum()),valid_fraction=float(mask.mean()),rgb_mae=float(error[mask].mean()))
