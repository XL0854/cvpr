"""Bounded solver/constraint/capacity diagnosis on three frozen truth-selected pairs."""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import bootstrap
from eth3d_geometry import correspondences_at_pixels, read_cameras, read_images, read_raw_depth, scale_pixels

sys.path.insert(0, str(bootstrap.ROOT/"diagnostics_geometry_20260915/scripts"))
from geometry import Warp, render


SAMPLES = [
    ("courtyard", "DSC_0317.JPG", "DSC_0318.JPG"),
    ("electro", "DSC_9302.JPG", "DSC_9303.JPG"),
    ("courtyard", "DSC_0315.JPG", "DSC_0317.JPG"),
]


def triangles(mesh):
    n=int(round(math.sqrt(len(mesh)))); m=mesh.reshape(n,n,2)
    a,b,c,d=m[:-1,:-1],m[:-1,1:],m[1:,:-1],m[1:,1:]
    cross=lambda x,y:x[...,0]*y[...,1]-x[...,1]*y[...,0]
    return torch.stack((cross(b-a,c-a),cross(d-b,c-b)))


def sampled(warp):
    n=int(round(math.sqrt(len(warp.mesh)))); m=warp.controls.reshape(n,n,2); points=[]
    for u in (.2,.5,.8):
        for v in (.2,.5,.8):
            points.append(((1-u)*(1-v)*m[:-1,:-1]+u*(1-v)*m[:-1,1:]
                           +(1-u)*v*m[1:,:-1]+u*v*m[1:,1:]).reshape(-1,2))
    _,jac=warp.at(torch.cat(points),True); det=torch.linalg.det(jac); sv=torch.linalg.svdvals(jac)
    return det,sv


def eval_points(ref_mesh,tgt_mesh,origin,extent,p512,q512):
    ref,tgt=Warp(ref_mesh,origin,extent),Warp(tgt_mesh,origin,extent)
    with torch.no_grad():
        zref,_,vr=ref.invert(p512/256-1); ztgt,_,vt=tgt.invert(q512/256-1)
        forward=(tgt.at(zref)-(q512/256-1)).norm(dim=-1)*256
        canvas=((zref-ztgt)*extent/2).norm(dim=-1)
        valid=vr&vt&torch.isfinite(forward)&torch.isfinite(canvas)
    return forward,canvas,valid


def distortion(ref_mesh,tgt_mesh,origin,extent,baseline_area,output_size):
    warp=Warp(tgt_mesh,origin,extent); area=triangles(tgt_mesh); det,sv=sampled(warp)
    ones=torch.ones(1,1,512,512,device=tgt_mesh.device)
    _,rm=render(Warp(ref_mesh,origin,extent),ones,output_size); _,tm=render(warp,ones,output_size)
    overlap=int((rm&tm).sum())
    return {"triangle_fold_fraction":float((area<=0).float().mean()),
            "sampled_fold_fraction":float((det<=0).float().mean()),
            "det_p05":float(torch.quantile(det,.05)),"det_min":float(det.min()),
            "anisotropy_p95":float(torch.quantile(sv[:,0]/sv[:,1].clamp_min(1e-8),.95)),
            "triangle_area_ratio":float(area.abs().sum()/baseline_area),"overlap_pixels":overlap}


def objective(base,delta,origin,extent,z,target,weight,initial_area,baseline_det,baseline_sign,
              reg_scale,cap):
    current=base+delta; warp=Warp(current,origin,extent); error=(warp.at(z)-target)*256
    raw=error.norm(dim=-1); point=F.smooth_l1_loss(error,torch.zeros_like(error),reduction="none",beta=2).mean(-1)
    geometry=(point*weight).sum()/weight.sum().clamp_min(1e-6)
    anchor=delta.square().mean(); n=int(round(math.sqrt(len(delta)))); grid=delta.reshape(n,n,2)
    smooth=(grid[1:]-grid[:-1]).square().mean()+(grid[:,1:]-grid[:,:-1]).square().mean()
    ratio=triangles(current)/initial_area.abs().clamp_min(1.); fold=F.relu(.2-ratio).square().mean()
    det,_=sampled(warp); tps=F.relu(.2*baseline_det.abs()-det*baseline_sign).square().mean()
    terms={"geometry":geometry,"raw_euclidean_mean":raw.mean(),"anchor":anchor,"smooth":smooth,
           "triangle_barrier":fold,"tps_barrier":tps}
    loss=geometry+reg_scale*(.002*anchor+.02*smooth+10*fold+100*tps)
    return loss,terms


def optimize(base,ref,origin,extent,p,q,selected,config):
    refwarp=Warp(ref,origin,extent)
    with torch.no_grad(): zall,_,valid=refwarp.invert(p/256-1)
    ids=torch.nonzero(selected&valid).flatten(); z=zall[ids]; target=q[ids]/256-1; weight=torch.ones(len(ids),device=base.device)
    initial_area=triangles(base).detach(); baseline_det,_=sampled(Warp(base,origin,extent)); baseline_det=baseline_det.detach(); sign=torch.sign(baseline_det)
    delta=torch.nn.Parameter(torch.zeros_like(base)); history=[]; cap=config["cap"]
    feasible_delta=delta.detach().clone()
    def feasible(value):
        with torch.no_grad():
            det,_=sampled(Warp(base+value,origin,extent))
            return bool((triangles(base+value)>0).all() and (det*sign>0).all())
    def record(step,loss,terms,grad):
        history.append({"step":step,"loss":float(loss.detach()),"grad_norm":float(grad),
                        **{k:float(v.detach()) for k,v in terms.items()}})
    started=time.perf_counter()
    if config["optimizer"]=="adam":
        opt=torch.optim.Adam([delta],lr=config.get("lr",.2))
        for step in range(config["steps"]):
            opt.zero_grad(); loss,terms=objective(base,delta,origin,extent,z,target,weight,initial_area,baseline_det,sign,config["reg_scale"],cap)
            loss.backward(); grad=delta.grad.norm(); opt.step()
            with torch.no_grad(): delta.clamp_(-cap,cap)
            if feasible(delta): feasible_delta.copy_(delta.detach())
            if step==0 or (step+1)%25==0 or step==config["steps"]-1: record(step,loss,terms,grad)
    else:
        opt=torch.optim.LBFGS([delta],lr=.8,max_iter=20,history_size=20,line_search_fn="strong_wolfe")
        calls=0
        for outer in range(config["steps"]//20):
            cache={}
            def closure():
                nonlocal calls
                opt.zero_grad(); loss,terms=objective(base,delta,origin,extent,z,target,weight,initial_area,baseline_det,sign,config["reg_scale"],cap)
                loss.backward(); calls+=1; cache.update(loss=loss,terms=terms,grad=delta.grad.norm()); return loss
            opt.step(closure)
            with torch.no_grad(): delta.clamp_(-cap,cap)
            if feasible(delta): feasible_delta.copy_(delta.detach())
            loss,terms=objective(base,delta,origin,extent,z,target,weight,initial_area,baseline_det,sign,config["reg_scale"],cap)
            grad=torch.autograd.grad(loss,delta)[0].norm(); record(calls,loss,terms,grad)
    raw_valid=feasible(delta); reverted=bool(config.get("safe",True) and not raw_valid)
    final_delta=feasible_delta if reverted else delta.detach()
    return base+final_delta,history,time.perf_counter()-started,int(len(ids)),raw_valid,reverted


def resize_mesh(mesh,n):
    old=int(round(math.sqrt(len(mesh)))); value=mesh.reshape(old,old,2).permute(2,0,1)[None]
    return F.interpolate(value,size=(n,n),mode="bilinear",align_corners=True)[0].permute(1,2,0).reshape(-1,2)


def numerical_checks():
    dtype=torch.float64; origin=torch.tensor([0.,0.],dtype=dtype); extent=torch.tensor([512.,512.],dtype=dtype)
    axis=torch.linspace(0,512,13,dtype=dtype); yy,xx=torch.meshgrid(axis,axis,indexing="ij"); mesh=torch.stack((xx,yy),-1).reshape(-1,2)
    warp=Warp(mesh,origin,extent); z=torch.tensor([[-.7,-.2],[0.,0.],[.6,.8]],dtype=dtype); p=warp.at(z); back,res,val=warp.invert(p)
    translated=mesh+torch.tensor([17.,-9.],dtype=dtype); tw=Warp(translated,origin,extent); tz,_,tv=tw.invert(p)
    expected=z+torch.tensor([34/512,-18/512],dtype=dtype)
    delta=torch.zeros_like(mesh,requires_grad=True); target=p+torch.tensor([[.01,-.015]],dtype=dtype)
    loss=(Warp(mesh+delta,origin,extent).at(z)-target).square().mean(); grad=torch.autograd.grad(loss,delta)[0]
    direction=torch.randn_like(delta); direction/=direction.norm(); eps=1e-4
    plus=(Warp(mesh+eps*direction,origin,extent).at(z)-target).square().mean(); minus=(Warp(mesh-eps*direction,origin,extent).at(z)-target).square().mean()
    finite=(plus-minus)/(2*eps); auto=(grad*direction).sum()
    # An identity inverse warp rendered from a horizontal pixel ramp must
    # sample the source coordinate, not its forward-warped opposite.
    ramp=torch.arange(512,dtype=dtype)[None,None,None,:].expand(1,1,512,512)
    rendered,mask=render(warp,ramp,(512,512)); expected_sampling=np.linspace(0,512,512)[None,:].repeat(512,0)
    sampling_error=float(np.max(np.abs(rendered[...,0][mask]-expected_sampling[mask])))
    return {"identity_forward_max":float((p-z).abs().max()),"roundtrip_max":float((back-z).abs().max()),
            "roundtrip_residual_px_max":float(res.max()),"roundtrip_all_valid":bool(val.all()),
            "translation_inverse_max":float((tz-expected).abs().max()),"translation_all_valid":bool(tv.all()),
            "gradient_finite":bool(torch.isfinite(grad).all()),"gradient_norm":float(grad.norm()),
            "directional_autograd":float(auto),"directional_finite_difference":float(finite),
            "directional_relative_error":float(abs(auto-finite)/max(abs(finite),1e-12)),
            "identity_inverse_sampling_max_px":sampling_error,
            "sampling_convention":"render uses inverse canvas->image TPS and torch.grid_sample align_corners=True; pixel=(normalized+1)*256, grid=2*pixel/511-1"}


def paths(scene,pair):
    if scene=="courtyard":
        geometry=bootstrap.ROOT/"diagnostics_parallax_20260929/runs"/f"rop_eth3d_courtyard_{pair[4:8]}_{pair[-4:]}_cpu"/"initial_geometry.npz"
        candidates=bootstrap.ROOT/"diagnostics_parallax_20260929/runs/roma_courtyard_small5"/pair/"candidates.npz"
    else:
        geometry=bootstrap.ROOT/"diagnostics_parallax_20260929/runs/rop_electro_final5_cpu"/pair/"initial_geometry.npz"
        candidates=bootstrap.ROOT/"diagnostics_parallax_20260929/runs/roma_electro_final5"/pair/"candidates.npz"
    return geometry,candidates


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,required=True); parser.add_argument("--device",default="cpu"); args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True); device=torch.device(args.device); torch.manual_seed(20260930)
    checks=numerical_checks(); (args.output/"numerical_checks.json").write_text(json.dumps(checks,indent=2)+"\n")
    configs=[{"name":"adam150_baseline","optimizer":"adam","steps":150,"lr":.2,"reg_scale":1.,"cap":128.,"safe":True},
             {"name":"adam600","optimizer":"adam","steps":600,"lr":.2,"reg_scale":1.,"cap":128.,"safe":True},
             {"name":"lbfgs300","optimizer":"lbfgs","steps":300,"reg_scale":1.,"cap":128.,"safe":True}]
    configs += [{"name":f"reg_{v:g}","optimizer":"adam","steps":300,"lr":.2,"reg_scale":v,"cap":128.,"safe":False} for v in (0.,.1,1.,10.)]
    configs += [{"name":f"cap_{v:g}","optimizer":"adam","steps":300,"lr":.2,"reg_scale":1.,"cap":v,"safe":True} for v in (16.,32.,64.,128.)]
    # Remove exact duplicate scan configurations while preserving diagnostic labels.
    rows=[]
    for scene,a,b in SAMPLES:
        pair=f"{Path(a).stem}_{Path(b).stem}"; scene_root=bootstrap.ROOT/"data/ETH3D"/scene
        cameras=read_cameras(scene_root/"dslr_calibration_jpg/cameras.txt"); poses=read_images(scene_root/"dslr_calibration_jpg/images.txt")
        cand=np.load(paths(scene,pair)[1]); geom=np.load(paths(scene,pair)[0]); pose1,pose2=poses[f"dslr_images/{a}"],poses[f"dslr_images/{b}"]
        c1,c2=cameras[pose1.camera_id],cameras[pose2.camera_id]; droot=scene_root/"ground_truth_depth/dslr_images"
        truth=correspondences_at_pixels(c1,pose1,read_raw_depth(droot/a,c1),c2,pose2,read_raw_depth(droot/b,c2),cand["source_native"])
        e=np.linalg.norm(cand["target_native"]-truth["q_star"],axis=1); cell=(cand["source_512"][:,0]//64).astype(int)+8*(cand["source_512"][:,1]//64).astype(int)
        held=cell%4==0; selected=(~held)&cand["valid"]&truth["evaluable"]&(e<=3); evaluation=held&cand["valid"]&truth["evaluable"]
        p=torch.from_numpy(cand["source_512"]).float().to(device); q=torch.from_numpy(cand["target_512"]).float().to(device)
        qstar=torch.from_numpy(scale_pixels(truth["q_star"],(c2.width,c2.height),(512,512))).float().to(device)
        ref=torch.from_numpy(geom["mesh_ref"]).float().reshape(-1,2).to(device); base=torch.from_numpy(geom["mesh_tgt"]).float().reshape(-1,2).to(device)
        both=torch.cat((ref,base)); origin,extent=both.min(0).values,both.max(0).values-both.min(0).values; outsize=(max(16,int(extent[1])),max(16,int(extent[0])))
        initial_area=triangles(base).abs().sum(); baseline_dist=distortion(ref,base,origin,extent,initial_area,outsize)
        _,initial_train,_=eval_points(ref,base,origin,extent,p[selected],q[selected]); _,initial_held,_=eval_points(ref,base,origin,extent,p[evaluation],qstar[evaluation])
        pairout=args.output/pair; pairout.mkdir(exist_ok=True)
        seen=set()
        for config in configs:
            key=(config["optimizer"],config["steps"],config.get("lr"),config["reg_scale"],config["cap"],13)
            if key in seen: continue
            seen.add(key); mesh,history,seconds,n,raw_valid,reverted=optimize(base,ref,origin,extent,p,q,torch.from_numpy(selected).to(device),config)
            train_forward,train_canvas,train_valid=eval_points(ref,mesh,origin,extent,p[selected],q[selected]); _,held_canvas,held_valid=eval_points(ref,mesh,origin,extent,p[evaluation],qstar[evaluation])
            aligned=torch.from_numpy(evaluation).to(device)&(eval_points(ref,base,origin,extent,p,qstar)[1]<=3)
            _,aligned_canvas,aligned_valid=eval_points(ref,mesh,origin,extent,p[aligned],qstar[aligned]) if aligned.any() else (None,torch.empty(0),torch.empty(0,dtype=torch.bool))
            dist=distortion(ref,mesh,origin,extent,initial_area,outsize); dist["overlap_retention"]=dist["overlap_pixels"]/baseline_dist["overlap_pixels"]
            row={"scene":scene,"pair":pair,"config":config["name"],"grid":13,"optimizer":config["optimizer"],"steps":config["steps"],"reg_scale":config["reg_scale"],"cap":config["cap"],
                 "train_n":n,"held_n":int(evaluation.sum()),"initial_train_canvas_mean":float(initial_train.mean()),"train_forward_mean":float(train_forward.mean()),"train_canvas_mean":float(train_canvas.mean()),
                 "initial_held_canvas_mean":float(initial_held.mean()),"held_canvas_mean":float(held_canvas.mean()),"aligned_n":int(aligned.sum()),"aligned_canvas_mean":float(aligned_canvas.mean()) if len(aligned_canvas) else None,
                 "raw_final_structurally_valid":raw_valid,"reverted_to_last_feasible":reverted,
                 "train_invalid":float((~train_valid).float().mean()),"held_invalid":float((~held_valid).float().mean()),"seconds":seconds,"max_displacement":float((mesh-base).norm(dim=1).max()),**dist}
            rows.append(row); (pairout/f"{config['name']}_history.json").write_text(json.dumps(history,indent=2)+"\n"); np.save(pairout/f"{config['name']}_mesh.npy",mesh.cpu().numpy())
        # One finer-grid check, after all 13x13 checks, with identical data and coefficients.
        fine=resize_mesh(base,25); config={"name":"fine25_adam600","optimizer":"adam","steps":600,"lr":.2,"reg_scale":1.,"cap":128.,"safe":True}
        mesh,history,seconds,n,raw_valid,reverted=optimize(fine,ref,origin,extent,p,q,torch.from_numpy(selected).to(device),config)
        tf,tc,tv=eval_points(ref,mesh,origin,extent,p[selected],q[selected]); _,hc,hv=eval_points(ref,mesh,origin,extent,p[evaluation],qstar[evaluation])
        aligned=torch.from_numpy(evaluation).to(device)&(eval_points(ref,base,origin,extent,p,qstar)[1]<=3); _,ac,av=eval_points(ref,mesh,origin,extent,p[aligned],qstar[aligned]) if aligned.any() else (None,torch.empty(0),torch.empty(0,dtype=torch.bool))
        fine_area=triangles(fine).abs().sum(); dist=distortion(ref,mesh,origin,extent,fine_area,outsize); dist["overlap_retention"]=dist["overlap_pixels"]/baseline_dist["overlap_pixels"]
        rows.append({"scene":scene,"pair":pair,"config":config["name"],"grid":25,"optimizer":"adam","steps":600,"reg_scale":1.,"cap":128.,"train_n":n,"held_n":int(evaluation.sum()),"initial_train_canvas_mean":float(initial_train.mean()),"train_forward_mean":float(tf.mean()),"train_canvas_mean":float(tc.mean()),"initial_held_canvas_mean":float(initial_held.mean()),"held_canvas_mean":float(hc.mean()),"aligned_n":int(aligned.sum()),"aligned_canvas_mean":float(ac.mean()) if len(ac) else None,"raw_final_structurally_valid":raw_valid,"reverted_to_last_feasible":reverted,"train_invalid":float((~tv).float().mean()),"held_invalid":float((~hv).float().mean()),"seconds":seconds,"max_displacement":float((mesh-fine).norm(dim=1).max()),**dist})
        (pairout/"fine25_adam600_history.json").write_text(json.dumps(history,indent=2)+"\n"); np.save(pairout/"fine25_adam600_mesh.npy",mesh.cpu().numpy())
        print("DONE",pair,flush=True)
    with (args.output/"results.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.output/"run.json").write_text(json.dumps({"status":"completed","samples":len(SAMPLES),"numerical_checks":checks,"rows":len(rows)},indent=2)+"\n")


if __name__=="__main__": main()
