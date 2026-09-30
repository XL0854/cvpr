"""Evaluate confidence thresholds for the already trained fusion probe."""
import bootstrap
from bootstrap import DIAG
import json, numpy as np, torch
from geometry import Warp, transfer_error, summaries

def up(q): return __import__('cv2').resize(q.reshape(5,5,2),(13,13),interpolation=__import__('cv2').INTER_CUBIC).reshape(-1,2)
def main():
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'); torch.set_num_threads(2)
    root=DIAG/'runs'; z=np.load(root/'fusion_probe_v2/predictions.npz'); pred=z['pred'];scores=z['confidence']; test=z['test']; rows=json.loads((root/'corr_distill/rows.json').read_text())
    thresholds=[0.90,0.95,0.97,0.98,0.99,0.995,0.999]
    out=[]
    for threshold in thresholds:
        vals={k:[] for k in ['held','independent']}; used=0
        for j,i in enumerate(test):
            r=rows[int(i)]; d=root/'pilot100/pairs'/r['id']; b=np.load(d/'baseline_geometry.npz'); origin=torch.as_tensor(b['origin'],device=device);extent=torch.as_tensor(b['extent'],device=device); bm=[torch.as_tensor(b[k],device=device).reshape(-1,2) for k in ['ref','tgt']]
            if scores[j] < threshold:mm=bm
            else:
                q=pred[j];ds=[up(q[:50]),up(q[50:])];mx=max(float(np.linalg.norm(a,axis=1).max()) for a in ds);fac=min(1.,20./max(mx,1e-6));mm=[m+torch.as_tensor(a*fac,device=device) for m,a in zip(bm,ds)];used+=1
            data=np.load(d/'teacher.npz');split=np.load(d/'split.npz')
            for k,pts in [('held',data['matches'][split['held']]),('independent',data['independent'])]:
                if len(pts):
                    e,v,_,_=transfer_error(*[Warp(m,origin,extent) for m in mm],torch.as_tensor(pts,device=device));vals[k].append(summaries(e,v))
        out.append(dict(threshold=threshold,used=used,fallback=30-used,metrics={k:dict(n=len(v),mean=float(np.mean([x['mean_px'] for x in v]))) for k,v in vals.items()}))
    p=root/'gate_sweep';p.mkdir(exist_ok=True);(p/'result.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
