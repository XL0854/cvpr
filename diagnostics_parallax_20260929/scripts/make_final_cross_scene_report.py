"""Create auditable tables and figures for the frozen three-scene diagnosis."""
import csv
import json
from pathlib import Path

import cv2
import numpy as np

import bootstrap


ROOT = bootstrap.ROOT / "diagnostics_parallax_20260929"
OUT = ROOT / "final_cross_scene_audit"
SCENES = {
    "courtyard": ("abcd_courtyard_final_audit", "roma_er_courtyard_small5", "roma_courtyard_small5"),
    "delivery_area": ("abcd_delivery_area_final_audit", "roma_er_delivery_area_final5", "roma_delivery_area_final5"),
    "electro": ("abcd_electro_final_audit", "roma_er_electro_final5", "roma_electro_final5"),
}


def put(image, text, xy, scale=.5, color=(20, 20, 20), thickness=1):
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def save_delta_chart(rows, path):
    width, height, margin = 1800, 650, 90
    image = np.full((height, width, 3), 255, np.uint8)
    values = [value for row in rows for value in (row["D_minus_C"], row["Dc_minus_Cc"])]
    limit = max(12., max(abs(v) for v in values)*1.1)
    zero = height//2
    cv2.line(image, (margin, zero), (width-20, zero), (0,0,0), 1)
    step = (width-margin-20)/len(rows)
    for i,row in enumerate(rows):
        cx=int(margin+(i+.5)*step)
        for dx,key,color in ((-7,"D_minus_C",(210,120,45)),(7,"Dc_minus_Cc",(55,155,80))):
            value=row[key]; end=int(zero-value*(zero-55)/limit)
            cv2.rectangle(image,(cx+dx-6,min(zero,end)),(cx+dx+6,max(zero,end)),color,-1)
        label=f"{row['scene'][:3]} {row['pair'].replace('DSC_','')}"
        put(image,label,(cx-38,height-25),.32)
    put(image,"D-C canvas error; below zero favors D",(margin,30),.7)
    put(image,"primary",(width-260,30),.5,(210,120,45),2); put(image,"controlled",(width-260,55),.5,(55,155,80),2)
    cv2.imwrite(str(path),image)


def save_er_scatter(scene_data, path):
    panel_w, height, margin = 600, 520, 65
    image=np.full((height,panel_w*3,3),255,np.uint8)
    colors={"correct":(90,160,70),"uncertain":(60,190,230),"incorrect":(70,80,220)}
    for si,(scene,e,r,kind) in enumerate(scene_data):
        x0=si*panel_w
        cv2.rectangle(image,(x0+margin,40),(x0+panel_w-20,height-margin),(0,0,0),1)
        for label,color in colors.items():
            ids=np.flatnonzero((kind==label)&np.isfinite(e)&np.isfinite(r))
            if len(ids)>2500: ids=ids[np.linspace(0,len(ids)-1,2500).astype(int)]
            for idx in ids:
                x=int(x0+margin+np.clip(e[idx],0,80)/80*(panel_w-margin-20))
                y=int(height-margin-np.clip(r[idx],0,150)/150*(height-margin-40))
                cv2.circle(image,(x,y),1,color,-1)
        y10=int(height-margin-10/150*(height-margin-40)); x3=int(x0+margin+3/80*(panel_w-margin-20)); x6=int(x0+margin+6/80*(panel_w-margin-20))
        cv2.line(image,(x0+margin,y10),(x0+panel_w-20,y10),(0,0,0),1); cv2.line(image,(x3,40),(x3,height-margin),(0,0,0),1); cv2.line(image,(x6,40),(x6,height-margin),(0,0,0),1)
        put(image,scene,(x0+250,25),.65); put(image,"e native px",(x0+250,height-15),.45)
    put(image,"r (512-scale canvas px)",(5,height-15),.36)
    cv2.imwrite(str(path),image)


def metric(group, name):
    return group[name].get("mean")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, entries = [], {}
    for scene, (run_name, diagnosis_name, candidate_name) in SCENES.items():
        run = ROOT / "runs" / run_name
        filter_rows = {r["pair"]: r for r in csv.DictReader(
            (ROOT/"runs"/diagnosis_name/"filter_audit_per_pair.csv").open())}
        for path in sorted(run.glob("DSC_*/result.json")):
            result = json.loads(path.read_text()); pair = result["id"]; groups = result["groups"]
            row = {"scene": scene, "pair": pair, "held_points": result["held_truth_points"],
                   "held_aligned": result["held_initial_aligned"], "held_difficult": result["held_initial_difficult"]}
            for short, group in (("A","A_baseline"),("B","B_residual"),("C","C_reliability"),("D","D_oracle_select"),
                                 ("Cc","C_spatial_count_control"),("Dc","D_spatial_count_control")):
                value = groups[group]
                row[f"{short}_canvas_mean"] = metric(value,"held_canvas_error")
                row[f"{short}_aligned_mean"] = metric(value,"aligned_canvas_error")
                row[f"{short}_difficult_mean"] = metric(value,"difficult_canvas_error")
                row[f"{short}_fold_fraction"] = value["structure"]["sampled_tps_fold_fraction"]
                row[f"{short}_triangle_area_ratio"] = value["triangle_area_ratio"]
                row[f"{short}_overlap_retention"] = value["overlap_pixels"]/groups["A_baseline"]["overlap_pixels"]
                row[f"{short}_inversion_failure"] = value["held_inversion_failure"]
            for group, short in (("C_reliability","C"),("D_oracle_select","D")):
                audit=result["selection_audit"][group]
                row[f"{short}_selected"] = audit["points"]
                row[f"{short}_source_cells"] = audit["source_cells_8x8"]
                row[f"{short}_weight_mean"] = audit["weight_mean"]
                row[f"{short}_fit_seconds_cpu"] = result["optimization"][group]["elapsed_seconds"]
                history=result["optimization"][group].get("history",[])
                row[f"{short}_final_train_geometry"] = history[-1]["geometry"] if history else None
            row["D_minus_C"] = row["D_canvas_mean"]-row["C_canvas_mean"]
            row["Dc_minus_Cc"] = row["Dc_canvas_mean"]-row["Cc_canvas_mean"]
            row.update({f"filter_{key}": value for key,value in filter_rows[pair].items() if key != "pair"})
            roma=json.loads((ROOT/"runs"/candidate_name/pair/"report.json").read_text())
            row["roma_seconds_gpu"] = roma["seconds"]; row["roma_peak_memory_bytes"] = roma["peak_memory_bytes"]
            rows.append(row); entries[(scene,pair)] = result
    with (OUT/"all_pairs.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    save_delta_chart(rows, OUT/"all_pair_D_minus_C.png")
    scene_data=[]
    for scene,(_, diagnosis_name,_) in SCENES.items():
        e,r,kind=[],[],[]
        for path in sorted((ROOT/"runs"/diagnosis_name).glob("DSC_*/diagnosis.npz")):
            data=np.load(path,allow_pickle=True); e.append(data["e_native"]); r.append(data["r_canvas"]); kind.append(data["correctness"])
        scene_data.append((scene,np.concatenate(e),np.concatenate(r),np.concatenate(kind)))
    save_er_scatter(scene_data, OUT/"e_r_scatter_three_scenes.png")

    examples=[("improved","courtyard","DSC_0317_DSC_0318"),
              ("unchanged","courtyard","DSC_0286_DSC_0287"),
              ("degraded","electro","DSC_9302_DSC_9303")]
    panels=[]
    for category,scene,pair in examples:
        run=ROOT/"runs"/SCENES[scene][0]/pair
        ims=[]
        for group in ("A_baseline","C_reliability","D_oracle_select"):
            image=cv2.imread(str(run/f"{group}_fusion.jpg"))
            image=cv2.resize(image,(420,300),interpolation=cv2.INTER_AREA)
            cv2.putText(image,group.split('_')[0],(12,28),cv2.FONT_HERSHEY_SIMPLEX,.8,(20,20,240),2,cv2.LINE_AA)
            ims.append(image)
        panel=np.hstack(ims); row=next(r for r in rows if r["scene"]==scene and r["pair"]==pair)
        title=f"{category}: {scene} {pair}; C={row['C_canvas_mean']:.2f}, D={row['D_canvas_mean']:.2f} px"
        cv2.putText(panel,title,(10,292),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(panel,title,(10,292),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
        panels.append(panel)
    cv2.imwrite(str(OUT/"improved_unchanged_degraded_examples.jpg"),np.vstack(panels))

    summary={}
    for scene in SCENES:
        aggregate=json.loads((ROOT/"runs"/SCENES[scene][0]/"aggregate.json").read_text())["aggregate"]
        pool=json.loads((ROOT/"runs"/SCENES[scene][1]/"pooled_summary.json").read_text())
        summary[scene]={"pairs":5,"geometry":aggregate,"filter_counts":pool["counts"],"frozen_C_filter":pool["frozen_C_rule"],
                        "roma_gpu_seconds_mean":float(np.mean([r["roma_seconds_gpu"] for r in rows if r["scene"]==scene])),
                        "roma_gpu_peak_memory_max":int(max(r["roma_peak_memory_bytes"] for r in rows if r["scene"]==scene)),
                        "C_fit_cpu_seconds_mean":float(np.mean([r["C_fit_seconds_cpu"] for r in rows if r["scene"]==scene])),
                        "D_fit_cpu_seconds_mean":float(np.mean([r["D_fit_seconds_cpu"] for r in rows if r["scene"]==scene]))}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")


if __name__ == "__main__":
    main()
