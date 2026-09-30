import bootstrap
from bootstrap import DIAG
import json
import numpy as np
import torch
from geometry import rigid, solve, evaluate, Warp
from matching import split_matches
from utils.torch_tps_transform import transformer


def main():
    torch.set_num_threads(2)
    torch.manual_seed(20260915)
    r=rigid()
    c=r+torch.stack((.03*torch.sin(2*r[:,1]),.02*torch.sin(3*r[:,0])),1)
    coeff=solve(c,r)
    y,x=torch.meshgrid(torch.arange(512.),torch.arange(512.),indexing='ij')
    ramp=torch.stack((x,y),0)[None]
    original=transformer(ramp,c[None],r[None],(65,65))[0].permute(1,2,0)
    y,x=torch.meshgrid(torch.linspace(-1,1,65),torch.linspace(-1,1,65),indexing='ij')
    z=torch.stack((x,y),-1).reshape(-1,2)
    mapped=evaluate(c,coeff,z)
    predicted=(mapped+1)*256
    mask=((predicted>1)&(predicted<510)).all(-1)
    parity=float((original.reshape(-1,2)[mask]-predicted[mask]).abs().max())
    assert parity<.003,parity
    warp=Warp((c+1)*256,torch.zeros(2),torch.ones(2)*512)
    query=torch.rand(100,2)*1.5-.75
    p=warp.at(query)
    inv,residual,valid=warp.invert(p)
    inverse_error=float((inv-query).abs().max())
    assert valid.all() and inverse_error<1e-4
    _,jac=warp.at(query,True)
    eps=1e-3
    for i in range(2):
        delta=torch.zeros_like(query);delta[:,i]=eps
        numeric=(warp.at(query+delta)-warp.at(query-delta))/(2*eps)
        torch.testing.assert_close(jac[:,:,i],numeric,atol=3e-3,rtol=1e-3)
    controls=c.clone().requires_grad_()
    evaluate(controls,solve(controls,r),query).square().mean().backward()
    assert torch.isfinite(controls.grad).all() and controls.grad.abs().sum()>0
    matches=np.array([[x,y,x+5,y+3] for x in range(12,501,8) for y in range(12,501,8)],np.float32)
    audit=matches[::100]
    train,held=split_matches(matches,audit,3)
    from scipy.spatial import cKDTree
    guard=np.concatenate((matches[held],audit))
    assert cKDTree(guard[:,:2]).query(matches[train,:2])[0].min()>=8
    assert cKDTree(guard[:,2:]).query(matches[train,2:])[0].min()>=8
    report=dict(status='passed',original_tps_coordinate_parity_max_px=parity,
                nonlinear_inverse_error_normalized=inverse_error,gradient='finite nonzero',
                split_guard='passed in both source images')
    p=DIAG/'runs/geometry_checks.json';p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(report,indent=2));print(report)


if __name__=='__main__':main()
