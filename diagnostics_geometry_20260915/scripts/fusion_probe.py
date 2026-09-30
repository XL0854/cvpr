"""Correlation-guided safe residual + confidence fallback probe."""
import bootstrap
from bootstrap import DIAG
import json, hashlib
from pathlib import Path
import cv2, numpy as np, torch
import torch.nn as nn

class Head(nn.Module):
    def __init__(self, c=8):
        super().__init__(); self.body=nn.Sequential(nn.Conv2d(c,32,3,padding=1),nn.ReLU(),nn.Conv2d(32,32,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d((5,5))); self.out=nn.Conv2d(32,5,1)
    def forward(self,x):
        z=self.out(self.body(x)).flatten(1); return z[:,:100],torch.sigmoid(z[:,100:].mean(1,keepdim=True))
def coarse(d):
    d=d.reshape(13,13,2);return np.concatenate([cv2.resize(d[:,:],(5,5),interpolation=cv2.INTER_AREA)],axis=0)
def up(q): return cv2.resize(q.reshape(5,5,2),(13,13),interpolation=cv2.INTER_CUBIC).reshape(-1,2)
def main():
    torch.set_num_threads(2); device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    (DIAG/'runs/fusion_probe_v2').mkdir(parents=True,exist_ok=True)
    root=DIAG/'runs'; feat=np.load(root/'corr_distill/features.npz')['features']; rows=json.loads((root/'corr_distill/rows.json').read_text())
    target=[]; conf=[]
    for r in rows:
        d=root/'safety100'/r['id']; b=np.load(root/'pilot100/pairs'/r['id']/'baseline_geometry.npz'); s=np.load(d/'safe_geometry.npz')
        delta=np.concatenate([s['ref'].reshape(13,13,2)-b['ref'].reshape(13,13,2),s['tgt'].reshape(13,13,2)-b['tgt'].reshape(13,13,2)]).reshape(338,2)
        target.append(np.concatenate([cv2.resize(delta[:169].reshape(13,13,2),(5,5),interpolation=cv2.INTER_AREA),cv2.resize(delta[169:].reshape(13,13,2),(5,5),interpolation=cv2.INTER_AREA)]).reshape(-1).astype(np.float32))
        rr=json.loads((d/'result.json').read_text());
        # Confidence label: whether the safe teacher actually improves the
        # independent proxy (held fallback when no independent points exist).
        metric=rr.get('independent',rr.get('held'))
        conf.append(float(metric is not None and metric['safe']['mean_px'] < metric['baseline']['mean_px']))
    target=np.stack(target);conf=np.array(conf,np.float32)
    tr=np.array([i for i,r in enumerate(rows) if (r['domain']=='udis_train' and i<35) or (r['domain']=='classic_development' and 50<=i<85)]);te=np.array([i for i in range(100) if i not in set(tr)])
    x=torch.as_tensor(feat,dtype=torch.float32,device=device); y=torch.as_tensor(target,device=device); c=torch.as_tensor(conf[:,None],device=device)
    model=Head(feat.shape[1]).to(device);opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4)
    for ep in range(160):
        p,q=model(x[tr]); loss=nn.functional.smooth_l1_loss(p,y[tr],beta=.5)+.5*nn.functional.binary_cross_entropy(q,c[tr]); opt.zero_grad();loss.backward();opt.step()
    model.eval()
    with torch.no_grad(): pred,score=model(x[te])
    pred=pred.cpu().numpy();score=score[:,0].cpu().numpy()
    np.savez_compressed(DIAG/'runs/fusion_probe_v2/predictions.npz',test=te,pred=pred,confidence=score)
    # A geometry-only evaluation reuses the exact metric implementation.
    from geometry import Warp,transfer_error,summaries
    result={'protocol':'35/15 per domain; safe teacher; confidence fallback to baseline; 160 epochs; threshold=0.5; low-rank 5x5 TPS residual.','test_ids':[rows[i]['id'] for i in te],'confidence_threshold':.5,'metrics':{}}
    for method in ['baseline','fused','oracle']:
        allv={k:[] for k in ['held','independent']}; used=0
        for j,i in enumerate(te):
            r=rows[i];d=root/'pilot100/pairs'/r['id']; b=np.load(d/'baseline_geometry.npz');o=np.load(root/'safety100'/r['id']/'safe_geometry.npz');origin=torch.as_tensor(b['origin'],device=device);extent=torch.as_tensor(b['extent'],device=device);bm=[torch.as_tensor(b[k],device=device).reshape(-1,2) for k in ['ref','tgt']]
            if method=='baseline':mm=bm
            elif method=='oracle':mm=[torch.as_tensor(o[k],device=device).reshape(-1,2) for k in ['ref','tgt']]
            elif score[j] < .5:mm=bm
            else:
                q=pred[j];ds=[up(q[:50]),up(q[50:])];mx=max(float(np.linalg.norm(z,axis=1).max()) for z in ds);fac=min(1.,20./max(mx,1e-6));mm=[m+torch.as_tensor(z*fac,device=device) for m,z in zip(bm,ds)];used+=1
            data=np.load(d/'teacher.npz');split=np.load(d/'split.npz')
            for k,pts in [('held',data['matches'][split['held']]),('independent',data['independent'])]:
                if len(pts):
                    e,v,_,_=transfer_error(*[Warp(m,origin,extent) for m in mm],torch.as_tensor(pts,device=device));allv[k].append(summaries(e,v))
        result['metrics'][method]={'used_predictions':used if method=='fused' else None,**{k:{'n':len(v),'mean':float(np.mean([z['mean_px'] for z in v]))} for k,v in allv.items()}}
    result['source_note']='Original RopStitch frozen; RoMa teacher offline; geometry safety represented by confidence fallback and displacement cap in this probe.'
    result['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();out=DIAG/'runs/fusion_probe_v2';out.mkdir(exist_ok=True);(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result['metrics'],indent=2))
if __name__=='__main__':main()
