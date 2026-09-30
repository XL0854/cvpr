"""Post-hoc measurement only: penalize lost baseline support; never refit meshes."""
import bootstrap
from bootstrap import DIAG
import json
import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity
from geometry import Warp, render


def run_audit(run, device):
    from run_diagnostics import load_images
    manifest=json.loads((run/'manifest.json').read_text())
    for index,sample in enumerate(manifest['samples']):
        folder=run/'pairs'/sample['id']
        if not (folder/'result.json').exists():continue
        if (folder/'support_audit.json').exists():continue
        data=np.load(folder/'baseline_geometry.npz')
        after=np.load(folder/'optimized_geometry.npz')
        origin=torch.tensor(data['origin'],device=device);extent=torch.tensor(data['extent'],device=device)
        tensors,_=load_images(sample,device)
        scale=min(1.,768/float(extent.max()))
        size=(max(16,int(float(extent[1])*scale)),max(16,int(float(extent[0])*scale)))
        rendered=[]
        for meshes in [data,after]:
            warps=[Warp(torch.tensor(meshes[k],device=device),origin,extent) for k in ['ref','tgt']]
            rendered.append([render(w,(im+1)*127.5,size) for w,im in zip(warps,tensors)])
        (a,ma),(b,mb)=rendered[0];(c,mc),(d,md)=rendered[1]
        mask=cv2.erode((ma&mb).astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)
        valid_after=cv2.erode((mc&md).astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)
        if mask.sum()<64:
            report=dict(valid=False)
        else:
            _,s0=structural_similarity(a,b,channel_axis=2,data_range=255,full=True)
            _,s1=structural_similarity(c,d,channel_axis=2,data_range=255,full=True)
            mse0=np.square((a-b)/255).mean(-1)
            mse1=np.square((c-d)/255).mean(-1)
            # Missing support gets maximum normalized squared error, not accidental
            # agreement between two black pixels. This is NOT standard PSNR/SSIM.
            penalized_ssim=np.where(valid_after,s1.mean(-1),0.)
            penalized_mse=np.where(valid_after,mse1,1.)
            report=dict(valid=True,baseline_support_pixels=int(mask.sum()),
                retained_baseline_fraction=float(valid_after[mask].mean()),
                before=dict(ssim=float(s0.mean(-1)[mask].mean()),
                            psnr=float(-10*np.log10(max(float(mse0[mask].mean()),1e-12)))),
                after=dict(ssim=float(penalized_ssim[mask].mean()),
                           psnr=float(-10*np.log10(max(float(penalized_mse[mask].mean()),1e-12)))),
                note='Post-hoc conservative sensitivity audit: fixed original overlap; missing pixels SSIM=0, MSE=1. Not paper metrics.')
        (folder/'support_audit.json').write_text(json.dumps(report,indent=2))
        print('SUPPORT_AUDIT',index+1,sample['id'],flush=True)
    print('SUPPORT_AUDIT_COMPLETED',flush=True)

