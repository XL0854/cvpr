"""Diagnostic evaluation of the ORIGINAL inverse TPS (not a swapped-control inverse).

Coordinates: source images use pixels in a 512-square normalization. Canvas uses
the fixed baseline bounding box. Coefficients/radial kernel match official TPS.
Newton inversion is used to evaluate correspondence transfer, with residuals
explicitly checked. Swapped controls are ONLY an initial guess for Newton.
"""
import bootstrap
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from skimage.metrics import structural_similarity


def rigid(n=13, device='cpu', dtype=torch.float32):
    y, x = torch.meshgrid(torch.linspace(-1, 1, n, device=device, dtype=dtype),
                          torch.linspace(-1, 1, n, device=device, dtype=dtype), indexing='ij')
    return torch.stack((x, y), -1).reshape(-1, 2)


def solve(controls, targets):
    p = torch.cat((torch.ones_like(controls[:, :1]), controls), -1)
    d2 = (controls[:, None]-controls[None]).square().sum(-1)
    r = d2*torch.log(d2+1e-6)
    w = torch.cat((torch.cat((p, r), 1),
                   torch.cat((p.new_zeros(3, 3), p.T), 1)), 0)
    rhs = torch.cat((targets, targets.new_zeros(3, 2)), 0)
    return torch.linalg.solve(w.double(), rhs.double()).to(controls.dtype)


def evaluate(controls, coeff, query, jacobian=False):
    diff = query[:, None]-controls[None]
    d2 = diff.square().sum(-1)
    r = d2*torch.log(d2+1e-6)
    basis = torch.cat((torch.ones_like(query[:, :1]), query, r), -1)
    out = basis@coeff
    if not jacobian:
        return out
    dr = 2*diff*(torch.log(d2+1e-6)+d2/(d2+1e-6))[..., None]
    jac = coeff[1:3].T[None] + torch.einsum('nki,ko->noi', dr, coeff[3:])
    return out, jac


class Warp:
    def __init__(self, mesh, origin, extent):
        self.mesh = mesh.reshape(-1, 2)
        self.controls = 2*(self.mesh-origin)/extent-1
        self.target = rigid(int(len(self.mesh)**.5), mesh.device, mesh.dtype)
        self.coeff = solve(self.controls, self.target)

    def at(self, z, jacobian=False):
        return evaluate(self.controls, self.coeff, z, jacobian)

    @torch.no_grad()
    def invert(self, p, steps=15):
        # This approximation is never treated as the actual forward warp.
        z = evaluate(self.target, solve(self.target, self.controls), p)
        for _ in range(steps):
            pred, jac = self.at(z, True)
            safe = torch.linalg.det(jac).abs() > 1e-6
            eye = torch.eye(2, device=z.device)[None]
            jac = torch.where(safe[:, None, None], jac, eye)
            delta = torch.linalg.solve(jac, (pred-p)[..., None])[..., 0]
            z = z-delta.clamp(-.15, .15)
        residual = (self.at(z)-p).norm(dim=-1)*256
        valid = torch.isfinite(z).all(-1) & torch.isfinite(residual) & (residual < .25)
        return z, residual, valid


@torch.no_grad()
def transfer_error(w1, w2, matches):
    p, q = matches[:, :2]/256-1, matches[:, 2:]/256-1
    z1, r1, v1 = w1.invert(p)
    z2, r2, v2 = w2.invert(q)
    error = ((w2.at(z1)-q).norm(dim=-1)+(w1.at(z2)-p).norm(dim=-1))*128
    valid = v1 & v2 & torch.isfinite(error)
    # A failed inversion is counted as a failure, never silently dropped.
    error = torch.where(valid, error, error.new_full(error.shape, 512.))
    return error, valid, z1, z2


def triangle_areas(mesh):
    m = mesh.reshape(13, 13, 2)
    a, b, c, d = m[:-1, :-1], m[:-1, 1:], m[1:, :-1], m[1:, 1:]
    def cross(x,y):
        return x[...,0]*y[...,1]-x[...,1]*y[...,0]
    return torch.stack((cross(b-a,c-a),cross(d-b,c-b)))


def summaries(errors, valid):
    x=errors.detach().cpu().numpy()
    if len(x)==0:
        return dict(n=0)
    return dict(n=len(x), mean_px=float(x.mean()), median_px=float(np.median(x)),
                p90_px=float(np.quantile(x,.9)), within3=float((x<3).mean()),
                inversion_failure=float((~valid).float().mean()))


@torch.no_grad()
def render(warp, image, size):
    h,w=size
    y,x=torch.meshgrid(torch.linspace(-1,1,h,device=image.device),
                       torch.linspace(-1,1,w,device=image.device),indexing='ij')
    z=torch.stack((x,y),-1).reshape(-1,2)
    mapped=torch.cat([warp.at(part) for part in z.split(8192)],0)
    # Official TPS converts normalized coordinates with width/2, NOT (width-1)/2.
    pixel=(mapped+1)*256
    grid=2*pixel/511-1
    output=F.grid_sample(image,grid.reshape(1,h,w,2),align_corners=True,padding_mode='zeros')
    mask=((pixel>=1)&(pixel<=510)).all(-1).reshape(h,w)
    return output[0].permute(1,2,0).cpu().numpy(), mask.cpu().numpy()


def image_scores(a,b,mask):
    mask=cv2.erode(mask.astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)
    if mask.sum()<64:
        return dict(valid=False, pixels=int(mask.sum()))
    _,s=structural_similarity(a,b,channel_axis=2,data_range=255,full=True)
    mse=np.square((a-b)[mask]/255).mean()
    return dict(valid=True,pixels=int(mask.sum()),mSSIM=float(s[mask].mean()),
                mPSNR=float(-10*np.log10(max(float(mse),1e-12))))


@torch.no_grad()
def structure(warp):
    # Evaluate derivative within each mesh cell, not just at the control points.
    m=warp.controls.reshape(13,13,2)
    z=[]
    for u in (.2,.5,.8):
        for v in (.2,.5,.8):
            z.append(((1-u)*(1-v)*m[:-1,:-1]+u*(1-v)*m[:-1,1:]
                      +(1-u)*v*m[1:,:-1]+u*v*m[1:,1:]).reshape(-1,2))
    _,j=warp.at(torch.cat(z),True)
    det=torch.linalg.det(j)
    sv=torch.linalg.svdvals(j)
    return dict(triangle_fold_fraction=float((triangle_areas(warp.mesh)<=0).float().mean()),
                sampled_tps_fold_fraction=float((det<=0).float().mean()),
                local_anisotropy_p95=float(torch.quantile(sv[:,0]/sv[:,1].clamp_min(1e-8),.95)))
