"""Small correlation-to-TPS-residual learnability probe.

All writes are under diagnostics_geometry_20260915. The original Network and
CoefNetwork are eval/frozen; only the new probe head is trained.
"""
import bootstrap
from bootstrap import DIAG, ROOT
import json, hashlib
from pathlib import Path
from unittest.mock import patch
import cv2, numpy as np, torch
import torch.nn as nn
import torchvision.models as models
import network

class ResidualHead(nn.Module):
    def __init__(self, in_ch=8):
        super().__init__()
        # Preserve spatial layout; global pooling loses where a TPS residual
        # should be applied.
        self.body=nn.Sequential(nn.Conv2d(in_ch,32,3,padding=1),nn.ReLU(),nn.Conv2d(32,32,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d((5,5)))
        self.out=nn.Conv2d(32,5,1)
    def forward(self,x):
        z=self.out(self.body(x)).flatten(1)
        return z[:,:100],torch.sigmoid(z[:,100:].mean(1,keepdim=True))

def image(path):
    x=cv2.imread(path)
    x=torch.from_numpy((x.astype(np.float32)/127.5-1).transpose(2,0,1))[None]
    return network.resize_512(x)

def main():
    torch.set_num_threads(2); device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda':torch.cuda.set_device(device)
    run=DIAG/'runs/pilot100';out=DIAG/'runs/corr_distill';out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((run/'manifest.json').read_text())['samples']
    ctor=models.resnet.resnet18
    with patch.object(models.resnet,'resnet18',side_effect=lambda *a,**k:ctor(weights=None)),patch.object(torch.cuda,'is_available',return_value=False):
        net=network.Network()
    net.load_state_dict(torch.load(ROOT/'woCoefNet/model_homo/epoch100_model.pth',map_location='cpu',weights_only=True)['model'],strict=True)
    net.to(device).eval().requires_grad_(False)
    feats=[]; targets=[]; rows=[]
    cache=out/'features.npz'
    if cache.exists():
        z=np.load(cache); feats=z['features'];targets=z['targets']; rows=json.loads((out/'rows.json').read_text())
    else:
        for i,s in enumerate(manifest):
            d=run/'pairs'/s['id']; b=np.load(d/'baseline_geometry.npz');o=np.load(d/'optimized_geometry.npz')
            with torch.inference_mode():
                f,a=net.extract_correlations(image(s['input1']).to(device),image(s['input2']).to(device))
            # frozen, active, difference, product; actual spatial correlation input.
            feats.append(torch.cat([f,a,f-a,f*a],1)[0].cpu().numpy())
            delta=np.concatenate([o['ref'].reshape(13,13,2)-b['ref'].reshape(13,13,2),o['tgt'].reshape(13,13,2)-b['tgt'].reshape(13,13,2)]).reshape(338,2)
            coarse=np.concatenate([cv2.resize(delta[:169].reshape(13,13,2),(5,5),interpolation=cv2.INTER_AREA),cv2.resize(delta[169:].reshape(13,13,2),(5,5),interpolation=cv2.INTER_AREA)]).reshape(-1)
            targets.append(coarse.astype(np.float32));rows.append(dict(id=s['id'],domain=s['domain']))
            print('FEATURE',i+1,len(manifest),s['id'],flush=True)
        feats=np.stack(feats);targets=np.stack(targets);np.savez_compressed(cache,features=feats,targets=targets);(out/'rows.json').write_text(json.dumps(rows))
    train=np.array([i for i,r in enumerate(rows) if (r['domain']=='udis_train' and i<35) or (r['domain']=='classic_development' and 50<=i<85)])
    test=np.array([i for i in range(len(rows)) if i not in set(train)])
    x=torch.as_tensor(feats,dtype=torch.float32,device=device);y=torch.as_tensor(targets,dtype=torch.float32,device=device)
    model=ResidualHead(feats.shape[1]).to(device);opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4)
    torch.manual_seed(20260917)
    for ep in range(120):
        model.train();pred,conf=model(x[train]);loss=nn.functional.smooth_l1_loss(pred,y[train],beta=.5)+.02*nn.functional.mse_loss(conf,torch.ones_like(conf)*.7)
        opt.zero_grad();loss.backward();opt.step()
        if ep in [0,39,79,119]:print('EPOCH',ep+1,float(loss),flush=True)
    model.eval();withtorch=torch.no_grad();pred,_=model(x[test]);pred=pred.detach().cpu().numpy()
    np.savez_compressed(out/'predictions.npz',test=test,pred=pred)
    # Evaluate correspondence transfer with the predicted residual meshes.
    from geometry import Warp, transfer_error, summaries
    result={'protocol':'35 train + 15 test per domain; correlation tensor input; 5x5 residual target; 120 epochs; no test tuning.','test_ids':[rows[i]['id'] for i in test], 'metrics':{}}
    for method in ['baseline','pred','oracle']:
        vals={k:[] for k in ['held','independent']}
        for j,i in enumerate(test):
            r=rows[i];d=run/'pairs'/r['id'];b=np.load(d/'baseline_geometry.npz');o=np.load(d/'optimized_geometry.npz');origin=torch.as_tensor(b['origin'],device=device);extent=torch.as_tensor(b['extent'],device=device)
            bm=[torch.as_tensor(b[k],device=device).reshape(-1,2) for k in ['ref','tgt']]
            if method=='baseline':mm=bm
            elif method=='oracle':mm=[torch.as_tensor(o[k],device=device).reshape(-1,2) for k in ['ref','tgt']]
            else:
                q=pred[j];ups=[cv2.resize(q[k:k+50].reshape(5,5,2),(13,13),interpolation=cv2.INTER_CUBIC).reshape(-1,2) for k in [0,50]]
                mm=[m+torch.as_tensor(u,dtype=torch.float32,device=device) for m,u in zip(bm,ups)]
            data=np.load(d/'teacher.npz');split=np.load(d/'split.npz')
            for k,pts in [('held',data['matches'][split['held']]),('independent',data['independent'])]:
                if len(pts):
                    e,v,_,_=transfer_error(*[Warp(m,origin,extent) for m in mm],torch.as_tensor(pts,device=device));vals[k].append(summaries(e,v))
        result['metrics'][method]={k:{'n':len(v),'mean':float(np.mean([z['mean_px'] for z in v]))} for k,v in vals.items()}
    result['source_note']='Features are original Network.extract_correlations; Network frozen. This is a small learnability gate, not final test.'
    result['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result['metrics'],indent=2))
if __name__=='__main__':main()
