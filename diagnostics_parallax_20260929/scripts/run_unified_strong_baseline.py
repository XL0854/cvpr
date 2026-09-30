"""Unified 15-pair C/D L-BFGS review using cached A and C-Adam results."""
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import bootstrap
from diagnose_residual_bottleneck import (distortion, eval_points, objective,
                                            paths as diagnostic_paths,
                                            sampled, triangles)
from eth3d_geometry import correspondences_at_pixels, read_cameras, read_images, read_raw_depth, scale_pixels
from geometry import Warp


ROOT=bootstrap.ROOT/"diagnostics_parallax_20260929"
SCENES={
 "courtyard":("configs/courtyard_small5.json","runs/roma_courtyard_small5","runs/abcd_courtyard_final_audit"),
 "delivery_area":("configs/delivery_area_final5.json","runs/roma_delivery_area_final5","runs/abcd_delivery_area_final_audit"),
 "electro":("configs/electro_final5.json","runs/roma_electro_final5","runs/abcd_electro_final_audit"),
}


def asset_paths(scene,pair):
    if scene=="courtyard":
        short=pair.replace("DSC_","")
        geom=ROOT/"runs"/f"rop_eth3d_courtyard_{short}_cpu"/"initial_geometry.npz"
    else:
        geom=ROOT/"runs"/f"rop_{scene}_final5_cpu"/pair/"initial_geometry.npz"
    roma=ROOT/"runs"/SCENES[scene][1].split("runs/")[-1]/pair
    return geom,roma


def feasible(base,delta,origin,extent,sign):
    with torch.no_grad():
        det,_=sampled(Warp(base+delta,origin,extent))
        return bool((triangles(base+delta)>0).all() and (det*sign>0).all())


def lbfgs(base,ref,origin,extent,p,q,selected,weights,time_budget):
    with torch.no_grad(): zall,_,valid=Warp(ref,origin,extent).invert(p/256-1)
    ids=torch.nonzero(selected&valid).flatten(); z=zall[ids]; target=q[ids]/256-1; weight=weights[ids]
    initial_area=triangles(base).detach(); baseline_det,_=sampled(Warp(base,origin,extent)); baseline_det=baseline_det.detach(); sign=torch.sign(baseline_det)
    delta=torch.nn.Parameter(torch.zeros_like(base)); optimizer=torch.optim.LBFGS([delta],lr=.8,max_iter=20,history_size=20,line_search_fn="strong_wolfe")
    last_feasible=delta.detach().clone(); budget_feasible=delta.detach().clone(); budget_outer=0; budget_evals=0; budget_seconds=0.
    cumulative=0.; closure_evals=0; accepted_feasible=0; checkpoints=[]; stagnant=0; termination="max_outer_15"
    for outer in range(1,16):
        before=delta.detach().clone(); local_evals=0; cache={}
        def closure():
            nonlocal closure_evals,local_evals
            optimizer.zero_grad(); loss,terms=objective(base,delta,origin,extent,z,target,weight,initial_area,baseline_det,sign,1.,128.)
            loss.backward(); closure_evals+=1; local_evals+=1
            cache.update(loss=float(loss.detach()),grad_norm=float(delta.grad.norm()),terms={k:float(v.detach()) for k,v in terms.items()})
            return loss
        started=time.perf_counter(); optimizer.step(closure); cumulative+=time.perf_counter()-started
        with torch.no_grad(): delta.clamp_(-128,128)
        valid_state=feasible(base,delta,origin,extent,sign)
        if valid_state:
            last_feasible.copy_(delta.detach()); accepted_feasible+=1
            if cumulative<=time_budget:
                budget_feasible.copy_(delta.detach()); budget_outer=outer; budget_evals=closure_evals; budget_seconds=cumulative
        change=float((delta.detach()-before).abs().max())
        checkpoints.append({"outer":outer,"cumulative_seconds":cumulative,"closure_evals":closure_evals,
                            "local_evals":local_evals,"accepted_feasible":valid_state,"parameter_change_max":change,**cache})
        stagnant=stagnant+1 if change<1e-6 else 0
        if stagnant>=2:
            termination="parameter_stagnation_two_outer_calls"; break
    raw_valid=feasible(base,delta,origin,extent,sign); final_delta=delta.detach() if raw_valid else last_feasible
    return {"mesh":base+final_delta,"budget_mesh":base+budget_feasible,"selected_after_inversion":int(len(ids)),
            "optimizer_seconds":cumulative,"closure_evals":closure_evals,"gradient_evals":closure_evals,
            "outer_calls":len(checkpoints),"accepted_feasible_outer":accepted_feasible,
            "raw_final_valid":raw_valid,"reverted_to_last_feasible":not raw_valid,
            "termination_reason":termination,"time_budget_seconds":time_budget,
            "budget_outer":budget_outer,"budget_closure_evals":budget_evals,
            "budget_optimizer_seconds":budget_seconds,"checkpoints":checkpoints}


def means(error,mask):
    value=error[mask]
    return {"n":int(mask.sum()),"mean":float(value.mean()) if len(value) else None,
            "median":float(value.median()) if len(value) else None}


def evaluate_group(name,mesh,ref,base,origin,extent,p,q,qstar,selected,evaluation,aligned,difficult,
                   baseline_area,baseline_overlap,outsize,source,fit):
    train_forward,train_canvas,train_valid=eval_points(ref,mesh,origin,extent,p[selected],q[selected])
    _,held,held_valid=eval_points(ref,mesh,origin,extent,p,qstar)
    dist=distortion(ref,mesh,origin,extent,baseline_area,outsize); dist["overlap_retention"]=dist["overlap_pixels"]/baseline_overlap
    result={"name":name,"train":means(train_canvas,torch.ones(len(train_canvas),dtype=torch.bool)),
            "train_forward":means(train_forward,torch.ones(len(train_forward),dtype=torch.bool)),
            "held":means(held,evaluation),"aligned":means(held,aligned),"difficult":means(held,difficult),
            "train_inversion_failure":float((~train_valid).float().mean()) if len(train_valid) else None,
            "held_inversion_failure":float((~held_valid[evaluation]).float().mean()) if evaluation.any() else None,
            "max_displacement_px":float((mesh-base).norm(dim=1).max()),"max_component_displacement_px":float((mesh-base).abs().max()),
            "distortion":dist,"fit":fit}
    np.save(source/f"{name}_mesh.npy",mesh.detach().cpu().numpy()); return result,held.detach().cpu().numpy()


def main():
    output=ROOT/"runs/unified_strong_baseline_review"; output.mkdir(parents=True,exist_ok=True); rows=[]
    for scene,(manifest_rel,candidate_rel,abcd_rel) in SCENES.items():
        config=json.loads((ROOT/manifest_rel).read_text()); scene_root=bootstrap.ROOT/config["scene"]
        cameras=read_cameras(scene_root/"dslr_calibration_jpg/cameras.txt"); poses=read_images(scene_root/"dslr_calibration_jpg/images.txt"); droot=scene_root/"ground_truth_depth/dslr_images"
        for a,b in config["pairs"]:
            pair=f"{Path(a).stem}_{Path(b).stem}"; pairout=output/pair; pairout.mkdir(exist_ok=True)
            candidate_dir=ROOT/candidate_rel/pair; cand=np.load(candidate_dir/"candidates.npz"); geom=np.load(asset_paths(scene,pair)[0]); cached=json.loads((ROOT/abcd_rel/pair/"result.json").read_text())
            p1,p2=poses[f"dslr_images/{a}"],poses[f"dslr_images/{b}"]; c1,c2=cameras[p1.camera_id],cameras[p2.camera_id]
            truth=correspondences_at_pixels(c1,p1,read_raw_depth(droot/a,c1),c2,p2,read_raw_depth(droot/b,c2),cand["source_native"])
            e=np.linalg.norm(cand["target_native"]-truth["q_star"],axis=1); cell=(cand["source_512"][:,0]//64).astype(int)+8*(cand["source_512"][:,1]//64).astype(int); held=cell%4==0; train=~held
            csel=train&cand["valid"]&(cand["confidence"]>=.5)&(cand["reverse_confidence"]>=.5)&(cand["cycle_512_px"]<=2)
            dsel=train&cand["valid"]&truth["evaluable"]&(e<=3); confidence=np.minimum(cand["confidence"],cand["reverse_confidence"]); cw=np.clip(confidence,.05,1).astype(np.float32); dw=np.ones(len(e),np.float32)
            evaluation=torch.from_numpy(held&cand["valid"]&truth["evaluable"]); p=torch.from_numpy(cand["source_512"]).float(); q=torch.from_numpy(cand["target_512"]).float(); qstar=torch.from_numpy(scale_pixels(truth["q_star"],(c2.width,c2.height),(512,512))).float()
            ref=torch.from_numpy(geom["mesh_ref"]).float().reshape(-1,2); base=torch.from_numpy(geom["mesh_tgt"]).float().reshape(-1,2); both=torch.cat((ref,base)); origin,extent=both.min(0).values,both.max(0).values-both.min(0).values
            _,initial_all,_=eval_points(ref,base,origin,extent,p,qstar); aligned=evaluation&(initial_all<=3); difficult=evaluation&(initial_all>=10)
            baseline_area=triangles(base).abs().sum(); outsize=(max(16,int(extent[1])),max(16,int(extent[0]))); baseline_overlap=cached["groups"]["A_baseline"]["overlap_pixels"]
            adam_seconds=float(cached["optimization"]["C_reliability"]["elapsed_seconds"])
            fits={"C":lbfgs(base,ref,origin,extent,p,q,torch.from_numpy(csel),torch.from_numpy(cw),adam_seconds),
                  "D":lbfgs(base,ref,origin,extent,p,q,torch.from_numpy(dsel),torch.from_numpy(dw),adam_seconds)}
            groups={}; point_errors={"source_512":cand["source_512"],"evaluation":evaluation.numpy(),"cell":cell}
            # Cached A and C-Adam meshes/results are reused, not re-optimized.
            for name,mesh_path,selection in (("A",ROOT/abcd_rel/pair/"A_baseline_mesh_tgt.npy",np.zeros(len(e),bool)),
                                             ("C_Adam",ROOT/abcd_rel/pair/"C_reliability_mesh_tgt.npy",csel)):
                mesh=torch.from_numpy(np.load(mesh_path)).float().reshape(-1,2); fit={"reused":True,"optimizer_seconds":0. if name=="A" else adam_seconds,
                    "objective_evals":0 if name=="A" else 150,"gradient_evals":0 if name=="A" else 150,
                    "termination_reason":"not_optimized" if name=="A" else "fixed_150_steps"}
                groups[name],point_errors[name]=evaluate_group(name,mesh,ref,base,origin,extent,p,q,qstar,torch.from_numpy(selection),evaluation,aligned,difficult,baseline_area,baseline_overlap,outsize,pairout,fit)
            for short,selection in (("C",csel),("D",dsel)):
                fit=fits[short]; public={k:v for k,v in fit.items() if k not in ("mesh","budget_mesh","checkpoints")}; public["checkpoints_file"]=f"{short}_LBFGS_checkpoints.json"
                (pairout/f"{short}_LBFGS_checkpoints.json").write_text(json.dumps(fit["checkpoints"],indent=2)+"\n")
                for suffix,mesh in (("LBFGS",fit["mesh"]),("LBFGS_time_matched",fit["budget_mesh"])):
                    name=f"{short}_{suffix}"; local=dict(public); local["time_matched"]=suffix.endswith("time_matched")
                    if local["time_matched"]:
                        local["optimizer_seconds"]=fit["budget_optimizer_seconds"]
                        local["closure_evals"]=fit["budget_closure_evals"]
                        local["gradient_evals"]=fit["budget_closure_evals"]
                    groups[name],point_errors[name]=evaluate_group(name,mesh,ref,base,origin,extent,p,q,qstar,torch.from_numpy(selection),evaluation,aligned,difficult,baseline_area,baseline_overlap,outsize,pairout,local)
            # Spatial support and held failure localization.
            eval_ids=np.flatnonzero(evaluation.numpy()); src=cand["source_512"]
            for short,selection in (("C",csel),("D",dsel)):
                selected_points=src[selection]; nearest=np.sqrt(((src[eval_ids,None]-selected_points[None])**2).sum(2)).min(1) if len(selected_points) else np.full(len(eval_ids),np.inf)
                point_errors[f"{short}_nearest_support_px"]=nearest
            np.savez_compressed(pairout/"held_point_errors.npz",**point_errors)
            roma=json.loads((candidate_dir/"report.json").read_text()); rop=json.loads((asset_paths(scene,pair)[0].parent/"result.json").read_text())
            result={"scene":scene,"pair":pair,"development_evidence":True,"groups":groups,
                    "selection":{"C_points":int(csel.sum()),"D_points":int(dsel.sum()),"C_cells":int(len(np.unique(cell[csel]))),"D_cells":int(len(np.unique(cell[dsel])))},
                    "timing_components":{"rop_cpu_seconds":rop["elapsed_seconds_cold_single_pair"],"roma_gpu_seconds":roma["seconds"],"mixed_device_warning":True}}
            (pairout/"result.json").write_text(json.dumps(result,indent=2)+"\n")
            for name,g in groups.items():
                rows.append({"scene":scene,"pair":pair,"group":name,"held_n":g["held"]["n"],"held_mean":g["held"]["mean"],"aligned_n":g["aligned"]["n"],"aligned_mean":g["aligned"]["mean"],"difficult_n":g["difficult"]["n"],"difficult_mean":g["difficult"]["mean"],"fold":g["distortion"]["sampled_fold_fraction"],"anisotropy_p95":g["distortion"]["anisotropy_p95"],"area_ratio":g["distortion"]["triangle_area_ratio"],"overlap_retention":g["distortion"]["overlap_retention"],"fit_seconds":g["fit"]["optimizer_seconds"],"objective_evals":g["fit"].get("closure_evals",g["fit"].get("objective_evals")),"termination":g["fit"]["termination_reason"]})
            print("DONE",scene,pair,flush=True)
    with (output/"all_groups.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output/"run.json").write_text(json.dumps({"status":"completed","pairs":15,"groups":["A","C_Adam","C_LBFGS","D_LBFGS"],"time_matched_supplement":True},indent=2)+"\n")


if __name__=="__main__": main()
