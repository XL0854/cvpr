"""Post-hoc development diagnostic: geometry-only residual attenuation, no training."""
import faulthandler
faulthandler.dump_traceback_later(45, repeat=True)
print('SAFETY_IMPORT_START', flush=True)
import bootstrap
from bootstrap import DIAG
import json, hashlib, time
from collections import Counter
import cv2
import numpy as np
import torch
from geometry import Warp, render, structure, transfer_error, summaries, image_scores
from run_diagnostics import load_images, save

@torch.no_grad()
def main():
    faulthandler.cancel_dump_traceback_later()
    print('SAFETY_IMPORT_COMPLETE', flush=True)
    torch.set_num_threads(2)
    device=torch.device('cuda:0')
    if not torch.cuda.is_available():raise RuntimeError('CUDA access required')
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32=False
    source=DIAG/'runs/pilot100';out=DIAG/'runs/safety100'
    out.mkdir(parents=True,exist_ok=True)
    protocol=dict(source='pilot100',scales=[1.,.75,.5,.25,0.],min_support=.99,max_anisotropy_ratio=1.1,
        selection='Largest scale satisfying overlap/union/each-image original support retention, per-side anisotropy and sampled no-fold constraints; no quality or correspondence scores used.',
        scope='Post-hoc development diagnostic; thresholds chosen after pilot100, not independent confirmatory evidence.',
        script_sha256=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest())
    if (out/'protocol.json').exists():
        old=json.loads((out/'protocol.json').read_text())
        if old!=protocol:raise RuntimeError('Protocol differs: use a new output directory')
    else:save(out/'protocol.json',protocol)
    samples=json.loads((source/'manifest.json').read_text())['samples']
    save(out/'manifest.json',dict(samples=samples))
    for index,sample in enumerate(samples):
        dest=out/sample['id'];dest.mkdir(exist_ok=True)
        if (dest/'result.json').exists():continue
        t=time.perf_counter();src=source/'pairs'/sample['id']
        raw=json.loads((src/'result.json').read_text())
        base=np.load(src/'baseline_geometry.npz');fit=np.load(src/'optimized_geometry.npz')
        origin=torch.as_tensor(base['origin'],device=device);extent=torch.as_tensor(base['extent'],device=device)
        meshes=[torch.as_tensor(base[k],device=device).reshape(-1,2) for k in ['ref','tgt']]
        deltas=[torch.as_tensor(fit[k],device=device).reshape(-1,2)-m for k,m in zip(['ref','tgt'],meshes)]
        tensors,_=load_images(sample,device)
        factor=min(1.,768/float(extent.max()));size=(max(16,int(float(extent[1])*factor)),max(16,int(float(extent[0])*factor)))
        bw=[Warp(m,origin,extent) for m in meshes]
        br=[render(w,(im+1)*127.5,size) for w,im in zip(bw,tensors)]
        bs=[structure(w) for w in bw]
        def supports(rr):
            a,b=[r[1] for r in rr];return [a&b,a|b,a,b]
        original_support=supports(br)
        attempts=[];raw_render=None
        for scale in protocol['scales']:
            current=[m+scale*d for m,d in zip(meshes,deltas)]
            ww=[Warp(m,origin,extent) for m in current]
            ss=[structure(w) for w in ww]
            rr=br if scale==0 else [render(w,(im+1)*127.5,size) for w,im in zip(ww,tensors)]
            if scale==1:raw_render=rr
            ret=[float((a&b).sum()/max(a.sum(),1)) for a,b in zip(original_support,supports(rr))]
            ratios=[s['local_anisotropy_p95']/b['local_anisotropy_p95'] for s,b in zip(ss,bs)]
            folds=any(s['triangle_fold_fraction']>0 or s['sampled_tps_fold_fraction']>0 for s in ss)
            accepted=min(ret)>=.99 and max(ratios)<=1.1 and not folds
            attempts.append(dict(scale=scale,retention=dict(zip(['overlap','union','ref','tgt'],ret)),anisotropy_ratios=ratios,folds=folds,accepted=accepted))
            if accepted or scale==0:break
        result=dict(id=sample['id'],domain=sample['domain'],scale=scale,attempts=attempts,passes_constraints=accepted,baseline_structure=bs,final_structure=ss)
        # All three methods use the identical mask, so raw/safe deltas are comparable.
        common=np.logical_and.reduce([mask for rr0 in [br,raw_render,rr] for _,mask in rr0])
        result['image_metrics']={name:image_scores(r[0][0],r[1][0],common) for name,r in [('baseline',br),('raw',raw_render),('safe',rr)]}
        data=np.load(src/'teacher.npz');split=np.load(src/'split.npz')
        for kind,points in [('held',data['matches'][split['held']]),('independent',data['independent'])]:
            if len(points):
                err,valid,_,_=transfer_error(*ww,torch.as_tensor(points,device=device))
                result[kind]=dict(baseline=raw[kind]['before'],raw=raw[kind]['after'],safe=summaries(err,valid))
        a,ma=rr[0];b,mb=rr[1]
        fusion=(a*ma[...,None]+b*mb[...,None])/np.maximum(ma.astype(float)+mb,1)[...,None]
        cv2.imwrite(str(dest/'safe_fusion.jpg'),np.clip(fusion,0,255).astype(np.uint8))
        np.savez_compressed(dest/'safe_geometry.npz',ref=current[0].cpu().numpy(),tgt=current[1].cpu().numpy())
        result['seconds']=time.perf_counter()-t
        save(dest/'result.json',result)
        print('SAFETY',index+1,len(samples),sample['id'],'scale',scale,'seconds',round(result['seconds'],1),flush=True)
    rows=[json.loads((out/s['id']/'result.json').read_text()) for s in samples]
    groups={}
    for domain in ['udis_train','classic_development']:
        rs=[r for r in rows if r['domain']==domain]
        g=dict(n=len(rs),scales=dict(Counter(str(r['scale']) for r in rs)),constraint_failures=sum(not r['passes_constraints'] for r in rs),
            min_retention={k:min(r['attempts'][-1]['retention'][k] for r in rs) for k in ['overlap','union','ref','tgt']},
            max_anisotropy_ratio=max(max(r['attempts'][-1]['anisotropy_ratios']) for r in rs))
        for metric in ['mSSIM','mPSNR']:
            g[metric]={name:float(np.mean([r['image_metrics'][name][metric] for r in rs])) for name in ['baseline','raw','safe']}
            g[metric]['safe_improved']=sum(r['image_metrics']['safe'][metric]>r['image_metrics']['baseline'][metric] for r in rs)
            g[metric]['safe_worse']=sum(r['image_metrics']['safe'][metric]<r['image_metrics']['baseline'][metric] for r in rs)
        for kind in ['held','independent']:
            avail=[r for r in rs if kind in r]
            g[kind]=dict(n=len(avail),**{name:float(np.mean([r[kind][name]['mean_px'] for r in avail])) for name in ['baseline','raw','safe']})
        groups[domain]=g
    save(out/'summary.json',groups)
    save(out/'status.json',dict(status='completed',count=len(rows)))
    print('SAFETY_COMPLETED',json.dumps(groups),flush=True)

if __name__=='__main__':main()
