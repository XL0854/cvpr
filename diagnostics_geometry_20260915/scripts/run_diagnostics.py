"""A: teacher audit; B: per-pair TPS residual capacity probe, no network training.

Reads original source/checkpoints. All writes live in the diagnostic directory.
The held-out set and independent SIFT points NEVER influence optimization or
checkpoint selection. Teacher agreement is explicitly not ground-truth accuracy.
"""
import bootstrap
from bootstrap import DIAG, ROOT
import argparse
import hashlib
import json
import time
import traceback
from unittest.mock import patch
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
import network as official
from geometry import Warp, transfer_error, summaries, triangle_areas, render, image_scores, structure
from matching import teacher_matches, query_teacher, sift_audit, split_matches


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def load_models(device):
    ctor=models.resnet.resnet18
    with patch.object(models.resnet,'resnet18',side_effect=lambda *a,**k:ctor(weights=None)), \
         patch.object(torch.cuda,'is_available',return_value=False):
        net,coef=official.Network(),official.CoefNetwork()
    for module,path in [(net,ROOT/'woCoefNet/model_homo/epoch100_model.pth'),
                        (coef,ROOT/'wCoefNet/model_coef/epoch050_coefmodel.pth')]:
        module.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['model'],strict=True)
        module.to(device).eval().requires_grad_(False)
    from romatch import roma_outdoor
    weights=DIAG/'cache/torch/hub/checkpoints'
    teacher=roma_outdoor(device=device,
        weights=torch.load(weights/'roma_outdoor.pth',map_location='cpu',weights_only=True),
        dinov2_weights=torch.load(weights/'dinov2_vitl14_pretrain.pth',map_location='cpu',weights_only=True),
        coarse_res=560,upsample_res=864,symmetric=True,use_custom_corr=False)
    teacher.eval().requires_grad_(False)
    return net,coef,teacher


def load_images(sample, device):
    a,b=[cv2.imread(sample[k]) for k in ['input1','input2']]
    if a is None or b is None:raise ValueError('Image unreadable')
    if a.shape!=b.shape:
        b=cv2.resize(b,(a.shape[1],a.shape[0]),interpolation=cv2.INTER_AREA)
    tensors=[torch.from_numpy((im.astype(np.float32)/127.5-1).transpose(2,0,1))[None].to(device) for im in [a,b]]
    tensors=[official.resize_512(im) for im in tensors]
    floats=[((im[0].permute(1,2,0)+1)*127.5).cpu().numpy() for im in tensors]
    byte_images=[np.clip(x,0,255).round().astype(np.uint8) for x in floats]
    return tensors,byte_images


def preview_matches(out, images, matches, held, seed):
    rng=np.random.default_rng(seed)
    ids=rng.choice(len(matches),min(12,len(matches)),replace=False) if len(matches) else []
    records=[];tiles=[]
    for i in ids:
        x,y,u,v=matches[i]
        panels=[]
        for image,point in zip(images,[(x,y),(u,v)]):
            patchim=cv2.getRectSubPix(image,(64,64),tuple(map(float,point)))
            patchim=cv2.resize(patchim,(160,160),interpolation=cv2.INTER_NEAREST)
            cv2.drawMarker(patchim,(80,80),(0,180,255),cv2.MARKER_CROSS,12,1)
            panels.append(patchim)
        tile=np.vstack((np.full((28,320,3),245,np.uint8),np.hstack(panels)))
        cv2.putText(tile,f'#{i} '+('HELD' if i in held else 'OTHER'),(8,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(30,30,30),1)
        tiles.append(tile)
        records.append(dict(match_index=int(i),p=[float(x),float(y)],q=[float(u),float(v)],human_verdict=None))
    if tiles:
        while len(tiles)%3:tiles.append(np.full_like(tiles[0],245))
        cv2.imwrite(str(out/'match_review.jpg'),np.vstack([np.hstack(tiles[i:i+3]) for i in range(0,len(tiles),3)]))
    save(out/'human_review.json',dict(status='not_manually_verified',sampled_matches=records))


def fit_mesh(meshes, origin, extent, matches, weights, steps):
    base=[m.detach().clone().reshape(-1,2) for m in meshes]
    baseline=[Warp(m,origin,extent) for m in base]
    _,valid,z1,z2=transfer_error(*baseline,matches)
    # Reject unresolved inversion points from fit only; track the number.
    fit=matches[valid];w=weights[valid]
    if len(fit)<32:return base,dict(status='insufficient_invertible_fit_points',fit_points=len(fit))
    delta=[torch.nn.Parameter(torch.zeros_like(m)) for m in base]
    z=torch.nn.Parameter(((z1[valid]+z2[valid])/2).detach())
    optim=torch.optim.Adam([{'params':delta,'lr':.15},{'params':[z],'lr':.002}])
    targets=[fit[:,:2]/256-1,fit[:,2:]/256-1]
    initial_area=[triangle_areas(m).detach() for m in base]
    history=[]
    for step in range(steps):
        optim.zero_grad()
        current=[m+d for m,d in zip(base,delta)]
        warps=[Warp(m,origin,extent) for m in current]
        errors=[(ww.at(z)-target)*256 for ww,target in zip(warps,targets)]
        perpoint=sum(F.smooth_l1_loss(e,torch.zeros_like(e),reduction='none',beta=2).mean(-1) for e in errors)/2
        geom=(perpoint*w).sum()/w.sum().clamp_min(1e-6)
        anchor=sum(d.square().mean() for d in delta)/2
        smooth=0
        for d in delta:
            dm=d.reshape(13,13,2)
            smooth=smooth+(dm[1:]-dm[:-1]).square().mean()+(dm[:,1:]-dm[:,:-1]).square().mean()
        fold=0
        for m,area in zip(current,initial_area):
            ratio=triangle_areas(m)/area.clamp_min(1.)
            fold=fold+F.relu(.2-ratio).square().mean()
        loss=geom+.005*anchor+.02*smooth+10*fold
        if not torch.isfinite(loss):
            return base,dict(status='nonfinite_optimization_reverted',step=step)
        loss.backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in [*delta,z]):
            return base,dict(status='nonfinite_gradient_reverted',step=step)
        optim.step()
        with torch.no_grad():
            for d in delta:d.clamp_(-20,20)
            z.clamp_(-1.5,1.5)
        if step%25==0 or step==steps-1:
            history.append(dict(step=step,loss=float(loss.detach()),geometry=float(geom.detach()),
                                anchor=float(anchor.detach()),smooth=float(smooth.detach()),fold=float(fold.detach())))
    return [(m+d.detach()) for m,d in zip(base,delta)],dict(status='completed',steps=steps,
        fit_points=len(fit),history=history,max_displacement_px=max(float(d.detach().norm(dim=-1).max()) for d in delta),
        selection='last fixed-budget iterate; no selection by held-out data',
        optimized='TPS control-point residuals and nuisance common-canvas match locations only')


def run_pair(sample, net, coef, teacher, out, steps, seed, device):
    started=time.perf_counter()
    out.mkdir(parents=True,exist_ok=True)
    tensors,images=load_images(sample,device)
    with torch.no_grad():
        outputs=net(*tensors,coef,.5)
        meshes=[m[0].detach() for m in outputs[-2:]]
        both=torch.cat([m.reshape(-1,2) for m in meshes])
        origin=both.min(0).values;extent=both.max(0).values-origin
        if not torch.isfinite(extent).all() or extent.min()<32 or extent.max()>4000:
            return dict(id=sample['id'],domain=sample['domain'],status='invalid_baseline_canvas')
        np.savez_compressed(out/'baseline_geometry.npz',ref=meshes[0].cpu().numpy(),tgt=meshes[1].cpu().numpy(),
            origin=origin.cpu().numpy(),extent=extent.cpu().numpy(),
            dst_ref=outputs[0][-1].cpu().numpy(),dst_tgt=outputs[1][-1].cpu().numpy(),coef=outputs[2][-1].cpu().numpy())
    base=[Warp(m,origin,extent) for m in meshes]
    cache=out/'teacher.npz'
    if cache.exists():
        data=np.load(cache)
        matches,weights,cycles,independent=[data[k] for k in ['matches','weights','cycles','independent']]
        teacher_audit=json.loads((out/'teacher_audit.json').read_text())
    else:
        torch.cuda.synchronize(device);t=time.perf_counter()
        with torch.inference_mode():
            warp,certainty=teacher.match(*[Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)) for im in images],device=device)
            warp,certainty=warp[0],certainty[0]
            matches,weights,cycles,stats=teacher_matches(warp,certainty,seed)
            independent=sift_audit(*images)
            if len(independent):
                ip=torch.as_tensor(independent[:,:2],device=device)
                iq=torch.as_tensor(independent[:,2:],device=device)
                prediction,_,_,safe=query_teacher(warp,certainty,ip)
                width=warp.shape[1]//2
                reverse_warp=torch.cat((warp[:,width:][:,:,[2,3,0,1]],
                                        warp[:,:width][:,:,[2,3,0,1]]),1)
                reverse_certainty=torch.cat((certainty[:,width:],certainty[:,:width]),1)
                prediction_reverse,_,_,safe_reverse=query_teacher(reverse_warp,reverse_certainty,iq)
                err=((prediction-iq).norm(dim=-1)+(prediction_reverse-ip).norm(dim=-1))/2
                safe=safe & safe_reverse
                stats['sift_all']=summaries(err,torch.isfinite(err))
                stats['sift_teacher_filtered']=summaries(err[safe],torch.isfinite(err[safe]))
                stats['sift_filter_coverage']=float(safe.float().mean())
            torch.cuda.synchronize(device)
            stats['seconds']=time.perf_counter()-t
            stats['independent_proxy']='mutual SIFT ratio<0.7 plus MAGSAC fundamental-matrix inliers at 1px; not truth'
            stats['metric']='symmetric transfer error in 512-square source pixels, same convention as student'
            teacher_audit=stats
        np.savez_compressed(cache,matches=matches,weights=weights,cycles=cycles,independent=independent)
        save(out/'teacher_audit.json',teacher_audit)
        del warp,certainty
    train,held=split_matches(matches,independent,seed)
    np.savez_compressed(out/'split.npz',train=train,held=held)
    preview_matches(out,images,matches,held,seed)
    mt=torch.as_tensor(matches,device=device);wt=torch.as_tensor(weights,device=device)
    it=torch.as_tensor(independent,device=device)
    row=dict(id=sample['id'],domain=sample['domain'],status='completed',teacher=teacher_audit,
             train_count=len(train),held_count=len(held),independent_count=len(independent),
             baseline_structure=[structure(w) for w in base])
    if len(train)<32 or len(held)<16:
        row['status']='insufficient_correspondences'
        fitted=[m.reshape(-1,2) for m in meshes]
        row['optimization']=dict(status='not_run')
    else:
        fitted,row['optimization']=fit_mesh(meshes,origin,extent,mt[train],wt[train],steps)
    after=[Warp(m,origin,extent) for m in fitted]
    for name,pts in [('train',mt[train]),('held',mt[held]),('independent',it)]:
        if len(pts):
            eb,vb,_,_=transfer_error(*base,pts);ea,va,_,_=transfer_error(*after,pts)
            row[name]=dict(before=summaries(eb,vb),after=summaries(ea,va),
                           paired_mean_change_px=float((ea-eb).mean()))
    row['final_structure']=[structure(w) for w in after]
    np.savez_compressed(out/'optimized_geometry.npz',ref=fitted[0].cpu().numpy(),tgt=fitted[1].cpu().numpy())
    scale=min(1.,768/float(extent.max()))
    size=(max(16,int(float(extent[1])*scale)),max(16,int(float(extent[0])*scale)))
    # Same fixed canvas, identical samples and a common mask across both methods.
    renders=[]
    for warps in [base,after]:
        renders.append([render(w,(im+1)*127.5,size) for w,im in zip(warps,tensors)])
    common=np.logical_and.reduce([m for result in renders for _,m in result])
    row['image_metrics']={}
    overlap=[]
    for name,result in zip(['before','after'],renders):
        (a,ma),(b,mb)=result;mask=ma&mb;overlap.append(int(mask.sum()))
        row['image_metrics'][name]=image_scores(a,b,common)
        row['image_metrics'][name]['own_overlap_pixels']=int(mask.sum())
        row['image_metrics'][name]['union_pixels']=int((ma|mb).sum())
        fusion=(a*ma[...,None]+b*mb[...,None])/np.maximum(ma.astype(float)+mb,1)[...,None]
        cv2.imwrite(str(out/f'{name}_fusion.jpg'),np.clip(fusion,0,255).astype(np.uint8))
    row['image_metrics']['overlap_retention']=overlap[1]/max(overlap[0],1)
    row['image_metrics']['note']='512-square inputs; fixed baseline canvas, common eroded support; diagnostic, not official native score'
    row['seconds_total']=time.perf_counter()-started
    return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default='pilot100')
    p.add_argument('--device',default='cuda:0');p.add_argument('--limit',type=int,default=0)
    p.add_argument('--steps',type=int,default=150)
    p.add_argument('--audit_quality',action='store_true',help='Post-hoc measurement only; never optimize')
    args=p.parse_args()
    device=torch.device(args.device)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; run with authorized GPU access')
    torch.cuda.set_device(device);torch.set_num_threads(2);torch.manual_seed(20260915)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    run=DIAG/'runs'/args.run
    if args.audit_quality:
        from audit_quality import run_audit
        run_audit(run,device)
        return
    manifest=json.loads((run/'manifest.json').read_text())
    selected=manifest['samples'];selected=selected[:args.limit] if args.limit else selected
    save(run/'execution.json',dict(device=str(device),steps=args.steps,selected=len(selected),
        torch=torch.__version__,roma_commit='77f8d68803526dcddfd9b7a46bc76125bdc25f15',
        diagnostic_script_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (DIAG/'scripts').glob('*.py')}))
    net,coef,teacher=load_models(device)
    versions=[p._version for m in [net,coef] for p in m.parameters()]
    print('ALL_MODELS_READY',flush=True)
    for index,sample in enumerate(selected):
        out=run/'pairs'/sample['id']
        if (out/'result.json').exists():
            print('RESUME_SKIP',sample['id'],flush=True);continue
        print('START_PAIR',index+1,len(selected),sample['id'],flush=True)
        try:
            row=run_pair(sample,net,coef,teacher,out,args.steps,20260915+index,device)
            assert versions==[p._version for m in [net,coef] for p in m.parameters()]
            save(out/'result.json',row)
            print('DONE',sample['id'],row['status'],
                  'held_delta',row.get('held',{}).get('paired_mean_change_px'),
                  'sift_delta',row.get('independent',{}).get('paired_mean_change_px'),flush=True)
        except Exception as exc:
            out.mkdir(parents=True,exist_ok=True)
            save(out/'error.json',dict(error=str(exc),traceback=traceback.format_exc()))
            raise
        torch.cuda.empty_cache()
    save(run/'status.json',dict(status='completed',selected=len(selected),
        original_parameter_versions_unchanged=True,manual_verification='pending'))
    print('DIAGNOSTICS_COMPLETED',run,flush=True)


if __name__=='__main__':main()
