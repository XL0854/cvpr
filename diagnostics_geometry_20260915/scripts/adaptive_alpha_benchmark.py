"""Benchmark a conservative, objective-evaluated low-call alpha search."""
import bootstrap
from bootstrap import DIAG, ROOT
import json,time,argparse
from pathlib import Path
from unittest.mock import patch
import cv2,numpy as np,torch,torchvision.models as models
from network import Network,CoefNetwork
from test import test_once,ternary_search

def load(path,device):
 x=cv2.imread(path); x=cv2.resize(x,(512,512),interpolation=cv2.INTER_AREA).astype(np.float32)
 return torch.from_numpy((x/127.5-1).transpose(2,0,1))[None].to(device)
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',default=str(DIAG/'runs/pilot100/manifest.json'));p.add_argument('--out',default='adaptive_alpha_100');args=p.parse_args()
 device=torch.device('cuda:0');torch.set_num_threads(2);torch.cuda.set_device(device)
 ctor=models.resnet.resnet18
 with patch.object(models.resnet,'resnet18',side_effect=lambda *a,**k:ctor(weights=None)),patch.object(torch.cuda,'is_available',return_value=False):net,coef=Network(),CoefNetwork()
 net.load_state_dict(torch.load(ROOT/'woCoefNet/model_homo/epoch100_model.pth',map_location='cpu',weights_only=True)['model']);coef.load_state_dict(torch.load(ROOT/'wCoefNet/model_coef/epoch050_coefmodel.pth',map_location='cpu',weights_only=True)['model']);net.to(device).eval();coef.to(device).eval()
 samples=json.loads(Path(args.manifest).read_text())['samples']
 rows=[]
 for i,s in enumerate(samples):
  a,b=load(s['input1'],device),load(s['input2'],device); torch.cuda.synchronize();st=time.perf_counter()
  # Five fixed probes, then three local ternary rounds (six evaluations).
  grid=np.array([-1.,-.25,.5,1.25,2.]); vals=[]
  for alpha in grid: vals.append((float(alpha),)+tuple(float(v) for v in test_once(net,coef,a,b,alpha=alpha)[:2]))
  order=np.argsort([-v[1] for v in vals]);best=vals[int(order[0])]; boundary=best[0] in (-1.,2.)
  calls=5; fallback=False
  if boundary:
   alpha,ss,ps,_=ternary_search(net,coef,a,b,low=-1,high=2,max_iter=20);calls+=40;fallback=True
  else:
   k=int(np.where(grid==best[0])[0][0]);lo,hi=grid[k-1],grid[k+1];alpha,ss,ps,img=None,None,None,None
   best_ss=-1e9
   for _ in range(3):
    m1=lo+(hi-lo)/3;m2=hi-(hi-lo)/3
    q1=test_once(net,coef,a,b,alpha=float(m1));q2=test_once(net,coef,a,b,alpha=float(m2));calls+=2
    if q1[0]>best_ss:alpha,ss,ps,img=float(m1),*q1
    if q2[0]>best_ss:alpha,ss,ps,img=float(m2),*q2
    best_ss=ss; lo,hi=(m1,hi) if q1[0]<q2[0] else (lo,m2)
  torch.cuda.synchronize();rows.append(dict(id=s['id'],domain=s['domain'],adaptive_ssim=float(ss),adaptive_psnr=float(ps),adaptive_alpha=float(alpha),calls=int(calls),fallback=bool(fallback),seconds=float(time.perf_counter()-st)))
  print('PAIR',i+1,s['id'],'calls',calls,'fallback',fallback,'ssim',ss,flush=True)
 # Full search only for comparison and separately timed.
 for r,s in zip(rows,samples):
  a=load(s['input1'],device);b=load(s['input2'],device);torch.cuda.synchronize();st=time.perf_counter();al,ss,ps,_=ternary_search(net,coef,a,b,low=-1,high=2,max_iter=20);torch.cuda.synchronize();r.update(full_ssim=float(ss),full_psnr=float(ps),full_alpha=float(al),full_calls=40,full_seconds=float(time.perf_counter()-st))
 out=DIAG/'runs'/args.out;out.mkdir(exist_ok=True);summary=dict(n=len(rows),manifest=str(args.manifest),adaptive_calls=float(np.mean([r['calls'] for r in rows])),full_calls=40,fallback_rate=float(np.mean([r['fallback'] for r in rows])),adaptive_ssim=float(np.mean([r['adaptive_ssim'] for r in rows])),full_ssim=float(np.mean([r['full_ssim'] for r in rows])),adaptive_psnr=float(np.mean([r['adaptive_psnr'] for r in rows])),full_psnr=float(np.mean([r['full_psnr'] for r in rows])),adaptive_seconds=float(np.mean([r['seconds'] for r in rows])),full_seconds=float(np.mean([r['full_seconds'] for r in rows])),rows=rows)
 (out/'result.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:summary[k] for k in summary if k!='rows'},indent=2))
if __name__=='__main__':main()
