"""Post-process the frozen residual bottleneck scan without further fitting."""
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

import bootstrap
from eth3d_geometry import correspondences_at_pixels, read_cameras, read_images, read_raw_depth, scale_pixels
from diagnose_residual_bottleneck import SAMPLES, eval_points, paths


def put(image,text,xy,scale=.42,color=(20,20,20),thickness=1):
    cv2.putText(image,text,xy,cv2.FONT_HERSHEY_SIMPLEX,scale,color,thickness,cv2.LINE_AA)


def tradeoff(rows,out):
    pairs=sorted(set(r["pair"] for r in rows)); W,H=720,500
    image=np.full((H*len(pairs),W*2,3),255,np.uint8)
    for pi,pair in enumerate(pairs):
        subset=[r for r in rows if r["pair"]==pair and (r["config"].startswith("reg_") or r["config"].startswith("cap_"))]
        for panel,(xkey,title,xmax) in enumerate((("anisotropy_p95","anisotropy p95",max(float(r["anisotropy_p95"]) for r in subset)*1.05),("overlap_retention","overlap retention",1.35))):
            x0,y0=panel*W,pi*H; left,top,right,bottom=x0+70,y0+45,x0+W-25,y0+H-55
            ys=[float(r["held_canvas_mean"]) for r in subset]; ymax=max(ys)*1.12
            cv2.rectangle(image,(left,top),(right,bottom),(0,0,0),1); put(image,f"{pair}: held error vs {title}",(left,25+y0),.55)
            for r in subset:
                xv=float(r[xkey]); yv=float(r["held_canvas_mean"]); x=int(left+min(xv/xmax,1)*(right-left)); y=int(bottom-yv/ymax*(bottom-top))
                valid=r["raw_final_structurally_valid"]=="True"; color=(50,150,60) if valid else (40,40,220)
                cv2.circle(image,(x,y),5,color,-1); put(image,r["config"].replace("reg_","r").replace("cap_","c"),(x+6,y-4),.32,color)
            put(image,"green=valid red=folded/invalid",(left,bottom+35),.36)
    cv2.imwrite(str(out),image)


def convergence(root,out):
    pairs=["DSC_0317_DSC_0318","DSC_9302_DSC_9303","DSC_0315_DSC_0317"]; names=["adam150_baseline","adam600","lbfgs300"]
    W,H=720,410; image=np.full((H*3,W,3),255,np.uint8); colors=[(220,120,40),(50,150,60),(60,60,210)]
    for pi,pair in enumerate(pairs):
        x0,y0=65,pi*H+40; right,bottom=W-25,(pi+1)*H-50
        cv2.rectangle(image,(x0,y0),(right,bottom),(0,0,0),1); put(image,f"{pair}: optimization geometry loss (log)",(x0,pi*H+25),.55)
        allh=[json.loads((root/pair/f"{n}_history.json").read_text()) for n in names]
        xmax=max(max(float(x["step"]) for x in h) for h in allh); vals=[float(x["geometry"]) for h in allh for x in h]; lo,hi=np.log10(max(min(vals),1e-3)),np.log10(max(vals))
        for name,h,color in zip(names,allh,colors):
            pts=[]
            for x in h:
                xx=int(x0+float(x["step"])/xmax*(right-x0)); yy=int(bottom-(np.log10(max(float(x["geometry"]),1e-3))-lo)/max(hi-lo,1e-6)*(bottom-y0)); pts.append((xx,yy))
            cv2.polylines(image,[np.array(pts)],False,color,2); put(image,name,(right-180,pi*H+65+names.index(name)*20),.36,color)
    cv2.imwrite(str(out),image)


def spatial_support(root):
    rows=[]
    for scene,a,b in SAMPLES:
        pair=f"{Path(a).stem}_{Path(b).stem}"; scene_root=bootstrap.ROOT/"data/ETH3D"/scene
        cams=read_cameras(scene_root/"dslr_calibration_jpg/cameras.txt"); poses=read_images(scene_root/"dslr_calibration_jpg/images.txt")
        cand=np.load(paths(scene,pair)[1]); geom=np.load(paths(scene,pair)[0]); p1,p2=poses[f"dslr_images/{a}"],poses[f"dslr_images/{b}"]; c1,c2=cams[p1.camera_id],cams[p2.camera_id]
        droot=scene_root/"ground_truth_depth/dslr_images"; truth=correspondences_at_pixels(c1,p1,read_raw_depth(droot/a,c1),c2,p2,read_raw_depth(droot/b,c2),cand["source_native"])
        e=np.linalg.norm(cand["target_native"]-truth["q_star"],axis=1); cell=(cand["source_512"][:,0]//64).astype(int)+8*(cand["source_512"][:,1]//64).astype(int); held=cell%4==0
        selected=(~held)&cand["valid"]&truth["evaluable"]&(e<=3); evaluation=held&cand["valid"]&truth["evaluable"]
        support=np.isin(cell,np.unique(cell[selected])); points=cand["source_512"]; nearest=np.sqrt(((points[evaluation,None]-points[selected][None])**2).sum(2)).min(1)
        p=torch.from_numpy(points).float(); qstar=torch.from_numpy(scale_pixels(truth["q_star"],(c2.width,c2.height),(512,512))).float(); ref=torch.from_numpy(geom["mesh_ref"]).float().reshape(-1,2); base=torch.from_numpy(geom["mesh_tgt"]).float().reshape(-1,2); both=torch.cat((ref,base)); origin,extent=both.min(0).values,both.max(0).values-both.min(0).values
        for config in ("adam150_baseline","lbfgs300","fine25_adam600"):
            mesh=torch.from_numpy(np.load(root/pair/f"{config}_mesh.npy")).float(); _,err,valid=eval_points(ref,mesh,origin,extent,p[evaluation],qstar[evaluation])
            eval_support=support[evaluation]
            for region,mask in (("training_cell_present",eval_support),("training_cell_absent",~eval_support),("nearest_le_64",nearest<=64),("nearest_gt_64",nearest>64)):
                value=err[torch.from_numpy(mask)]
                rows.append({"scene":scene,"pair":pair,"config":config,"region":region,"n":len(value),"held_canvas_mean":float(value.mean()) if len(value) else None,"inversion_failure":float((~valid[torch.from_numpy(mask)]).float().mean()) if len(value) else None})
    with (root/"spatial_support.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    root=Path(sys.argv[1]); rows=list(csv.DictReader((root/"results.csv").open()))
    tradeoff(rows,root/"constraint_tradeoff.png"); convergence(root,root/"solver_convergence.png"); spatial_support(root)


if __name__=="__main__": main()
