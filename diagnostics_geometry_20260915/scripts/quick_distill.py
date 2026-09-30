"""Fast learnability probe for low-dimensional TPS residuals.

Uses only saved pilot100 meshes/matches. No RoMa, original network training, or
image metrics are run here. The mesh itself is a deliberately optimistic proxy
feature; this is a gate for whether a learned residual target is worth trying.
"""
import bootstrap
from bootstrap import DIAG
import json, hashlib
from pathlib import Path
import cv2
import numpy as np
import torch
from geometry import Warp, transfer_error, summaries

def coarse(x):
    return cv2.resize(x.reshape(13,13,2), (5,5), interpolation=cv2.INTER_AREA)
def expand(x):
    return cv2.resize(x.reshape(5,5,2), (13,13), interpolation=cv2.INTER_CUBIC).reshape(-1,2)

def main():
    torch.set_num_threads(2)
    root=DIAG/'runs/pilot100';out=DIAG/'runs/quick_distill';out.mkdir(parents=True,exist_ok=True)
    samples=json.loads((root/'manifest.json').read_text())['samples']
    rows=[]
    for i,s in enumerate(samples):
        d=root/'pairs'/s['id']; r=json.loads((d/'result.json').read_text())
        b=np.load(d/'baseline_geometry.npz'); o=np.load(d/'optimized_geometry.npz')
        base=np.concatenate([b['ref'].reshape(-1,2),b['tgt'].reshape(-1,2)])
        delta=np.concatenate([o['ref'].reshape(-1,2)-b['ref'].reshape(-1,2),o['tgt'].reshape(-1,2)-b['tgt'].reshape(-1,2)])
        # Features are only baseline geometry and correspondence counts: an
        # optimistic proxy, intentionally separate from correlation features.
        both=base; geom_feat=((both-both.mean(0))/(both.std(0)+1e-6)).reshape(-1)
        feat=np.concatenate([geom_feat,
            [np.log1p(r.get('train_count',0)),np.log1p(r.get('held_count',0)),np.log1p(r.get('independent_count',0)),
             r.get('teacher',{}).get('filtered_fraction',0),r.get('teacher',{}).get('sift_all',{}).get('mean_px',0)]])
        rows.append(dict(id=s['id'],domain=s['domain'],i=i,feature=feat,target=np.concatenate([coarse(delta[:169]),coarse(delta[169:])]).reshape(-1),
                         base=b,optimized=o,result=r))
    # Fixed, stratified split: first 70% of each domain train, last 30% test.
    train=[r for r in rows if (r['domain']=='udis_train' and r['i']<35) or (r['domain']=='classic_development' and r['i']>=50 and r['i']<85)]
    test=[r for r in rows if r not in train]
    X=np.stack([r['feature'] for r in train]);Y=np.stack([r['target'] for r in train]);Xt=np.stack([r['feature'] for r in test])
    # Standardized ridge multi-output; lambda selected a priori, no test tuning.
    mu=X.mean(0);sd=X.std(0)+1e-6;Xn=(X-mu)/sd;Xtn=(Xt-mu)/sd
    lam=10.; A=Xn.T@Xn+lam*np.eye(Xn.shape[1]); W=np.linalg.solve(A,Xn.T@Y)
    pred=Xtn@W
    pred=np.stack([np.concatenate([expand(p[:50]),expand(p[50:])]).astype(np.float32) for p in pred])
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    def evaluate(rr,which):
        vals=[]
        for r,p in zip(rr,pred if which=='pred' else [None]*len(rr)):
            b=r['base']; origin=torch.as_tensor(b['origin'],device=device);extent=torch.as_tensor(b['extent'],device=device)
            bm=[torch.as_tensor(b[k],device=device).reshape(-1,2) for k in ['ref','tgt']]
            if which=='baseline': mm=bm
            elif which=='oracle':
                o=r['optimized'];mm=[torch.as_tensor(o[k],device=device).reshape(-1,2) for k in ['ref','tgt']]
            else:mm=[m+torch.as_tensor(q,device=device) for m,q in zip(bm,[p[:169],p[169:]])]
            points=[]
            for key in ['held','independent']:
                f=root/'pairs'/r['id']/('teacher.npz'); data=np.load(f);split=np.load(root/'pairs'/r['id']/'split.npz')
                pts=data['matches'][split['held']] if key=='held' else data['independent']
                if len(pts):
                    e,v,_,_=transfer_error(*[Warp(m,origin,extent) for m in mm],torch.as_tensor(pts,device=device));points.append((key,summaries(e,v)))
            vals.append(dict(id=r['id'],domain=r['domain'],metrics=dict(points)))
        return vals
    result=dict(protocol='Fixed stratified 70/30 split; ridge lambda=10; target 5x5 TPS residual; baseline mesh is optimistic proxy feature.',
        train=[r['id'] for r in train],test=[r['id'] for r in test],feature='baseline TPS meshes + correspondence count/teacher proxy statistics',
        target='coarse 5x5 residual for both TPS meshes, bicubic expansion to 13x13',
        metrics={k:evaluate(test,k) for k in ['baseline','pred','oracle']})
    summary={}
    for key in ['held','independent']:
        summary[key]={}
        for method in ['baseline','pred','oracle']:
            a=result['metrics'][method]
            vals=[dict(x['metrics'])[key]['mean_px'] for x in a if key in dict(x['metrics'])]
            summary[key][method]=dict(n=len(vals),mean=float(np.mean(vals)) if vals else None)
    result['summary']=summary
    result['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
