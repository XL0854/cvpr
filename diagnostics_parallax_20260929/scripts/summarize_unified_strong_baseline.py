"""Aggregate, time rendering, and localize failures for the unified review."""
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

import bootstrap
from diagnose_residual_bottleneck import paths as diagnostic_paths
from geometry import Warp, render

ROOT=bootstrap.ROOT/"diagnostics_parallax_20260929"; RUN=ROOT/"runs/unified_strong_baseline_review"
GROUPS=["A","C_Adam","C_LBFGS","D_LBFGS","C_LBFGS_time_matched","D_LBFGS_time_matched"]


def put(im,text,xy,scale=.45,color=(20,20,20),thickness=1):
    cv2.putText(im,text,xy,cv2.FONT_HERSHEY_SIMPLEX,scale,color,thickness,cv2.LINE_AA)


def weighted(items,group,key):
    vals=[(x["groups"][group][key]["n"],x["groups"][group][key]["mean"]) for x in items if x["groups"][group][key]["n"]]
    n=sum(a for a,_ in vals); return {"n":n,"mean":sum(a*b for a,b in vals)/n if n else None}


def geometry_path(scene,pair):
    if scene=="courtyard": return ROOT/"runs"/f"rop_eth3d_courtyard_{pair.replace('DSC_','')}_cpu"/"initial_geometry.npz"
    return ROOT/"runs"/f"rop_{scene}_final5_cpu"/pair/"initial_geometry.npz"


def image_path(scene,name): return bootstrap.ROOT/"data/ETH3D"/scene/"images/dslr_images"/name


def time_postprocess(result,pairdir):
    scene,pair=result["scene"],result["pair"]; target_name=pair.split("_")[2]+"_"+pair.split("_")[3]+".JPG" if False else None
    # Pair ids always have DSC_xxxx_DSC_xxxx form.
    parts=pair.split("_"); target_name=f"{parts[2]}_{parts[3]}.JPG"
    image=cv2.resize(cv2.imread(str(image_path(scene,target_name))),(512,512),interpolation=cv2.INTER_AREA)
    tensor=torch.from_numpy(image.astype(np.float32).transpose(2,0,1))[None]
    geom=np.load(geometry_path(scene,pair)); ref=torch.from_numpy(geom["mesh_ref"]).float().reshape(-1,2); base=torch.from_numpy(geom["mesh_tgt"]).float().reshape(-1,2)
    both=torch.cat((ref,base)); origin,extent=both.min(0).values,both.max(0).values-both.min(0).values; size=(max(16,int(extent[1])),max(16,int(extent[0])))
    timings={}
    for group in ("C_Adam","C_LBFGS","D_LBFGS"):
        mesh=torch.from_numpy(np.load(pairdir/f"{group}_mesh.npy")).float(); started=time.perf_counter(); rendered,mask=render(Warp(mesh,origin,extent),tensor,size); _=rendered*mask[...,None]; timings[group]=time.perf_counter()-started
    return timings


def bar_chart(entries,path):
    W,H=1900,650; image=np.full((H,W,3),255,np.uint8); left,bottom,top=75,H-100,45; vmax=max(x["groups"][g]["held"]["mean"] for x in entries for g in ("A","C_Adam","C_LBFGS","D_LBFGS"))*1.05
    colors=[(150,150,150),(220,150,60),(60,150,70),(70,70,220)]; step=(W-left-20)/len(entries)
    cv2.line(image,(left,bottom),(W-20,bottom),(0,0,0),1)
    for i,x in enumerate(entries):
        cx=int(left+(i+.5)*step)
        for j,(g,color) in enumerate(zip(("A","C_Adam","C_LBFGS","D_LBFGS"),colors)):
            value=x["groups"][g]["held"]["mean"]; h=int(value/vmax*(bottom-top)); xx=cx+(j-2)*8; cv2.rectangle(image,(xx,bottom-h),(xx+7,bottom),color,-1)
        put(image,x["scene"][:3]+" "+x["pair"].replace("DSC_",""),(cx-42,H-65),.31)
    put(image,"Held canvas error per pair (lower is better)",(left,28),.65)
    for j,(name,color) in enumerate(zip(("A","C-Adam","C-LBFGS","D-LBFGS"),colors)): put(image,name,(W-170,25+j*20),.35,color,2)
    cv2.imwrite(str(path),image)


def region_maps(entries,path,cell_rows):
    chosen=["DSC_0309_DSC_0311","DSC_0315_DSC_0317","DSC_0695_DSC_0696"]; panels=[]
    for pair in chosen:
        x=next(v for v in entries if v["pair"]==pair); scene=x["scene"]; z=np.load(RUN/pair/"held_point_errors.npz"); ids=np.flatnonzero(z["evaluation"]); points=z["source_512"][ids]; gain=z["C_LBFGS"][ids]-z["D_LBFGS"][ids]
        parts=pair.split("_"); source_name=f"{parts[0]}_{parts[1]}.JPG"; im=cv2.resize(cv2.imread(str(image_path(scene,source_name))),(512,512),interpolation=cv2.INTER_AREA)
        for point,value in zip(points,gain):
            color=(30,190,30) if value>2 else ((30,30,220) if value<-2 else (180,180,180)); cv2.circle(im,tuple(np.rint(point).astype(int)),4,color,-1)
        put(im,f"{scene} {pair}: green D better, red C better",(8,22),.43,(255,255,255),2)
        panels.append(im)
        cells=z["cell"][ids]
        for cell in sorted(set(cells)):
            m=cells==cell
            cell_rows.append({"scene":scene,"pair":pair,"cell":int(cell),"n":int(m.sum()),"mean_x":float(points[m,0].mean()),"mean_y":float(points[m,1].mean()),"C_error":float(z["C_LBFGS"][ids][m].mean()),"D_error":float(z["D_LBFGS"][ids][m].mean()),"D_gain":float(gain[m].mean()),"C_support_distance":float(z["C_nearest_support_px"][m].mean()),"D_support_distance":float(z["D_nearest_support_px"][m].mean())})
    cv2.imwrite(str(path),np.hstack(panels))


def main():
    entries=[json.loads(p.read_text()) for p in sorted(RUN.glob("DSC_*/result.json"))]
    render_rows=[]
    for x in entries:
        timing=time_postprocess(x,RUN/x["pair"])
        for g,v in timing.items(): x["groups"][g]["postprocess_cpu_seconds"]=v
        (RUN/x["pair"]/"result.json").write_text(json.dumps(x,indent=2)+"\n")
        render_rows.append({"scene":x["scene"],"pair":x["pair"],**timing})
    with (RUN/"render_timing.csv").open("w",newline="") as s:
        w=csv.DictWriter(s,fieldnames=list(render_rows[0])); w.writeheader(); w.writerows(render_rows)
    summary={}
    for scene in ("courtyard","delivery_area","electro","all"):
        items=entries if scene=="all" else [x for x in entries if x["scene"]==scene]; groups={}
        for g in GROUPS:
            groups[g]={"held":weighted(items,g,"held"),"aligned":weighted(items,g,"aligned"),"difficult":weighted(items,g,"difficult"),
                       "max_fold":max(x["groups"][g]["distortion"]["sampled_fold_fraction"] for x in items),
                       "min_overlap_retention":min(x["groups"][g]["distortion"]["overlap_retention"] for x in items),
                       "max_anisotropy_p95":max(x["groups"][g]["distortion"]["anisotropy_p95"] for x in items),
                       "fit_seconds_mean":statistics.mean(x["groups"][g]["fit"]["optimizer_seconds"] for x in items),
                       "objective_evals_mean":statistics.mean(x["groups"][g]["fit"].get("closure_evals",x["groups"][g]["fit"].get("objective_evals",0)) for x in items)}
        summary[scene]={"pairs":len(items),"groups":groups}
    (RUN/"aggregate.json").write_text(json.dumps(summary,indent=2)+"\n")
    bar_chart(entries,RUN/"all_pair_geometry.png"); cells=[]; region_maps(entries,RUN/"failure_region_maps.jpg",cells)
    with (RUN/"selected_region_cells.csv").open("w",newline="") as s:
        w=csv.DictWriter(s,fieldnames=list(cells[0])); w.writeheader(); w.writerows(cells)


if __name__=="__main__": main()
