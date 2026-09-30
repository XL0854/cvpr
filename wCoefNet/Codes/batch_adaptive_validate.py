"""Resumable batch validation for the conservative alpha search."""
import argparse,json,time
from pathlib import Path
import cv2,numpy as np,torch
from adaptive_alpha_search import adaptive_alpha_search
from network import Network,CoefNetwork
from test import ternary_search

def load(p,d):
 x=cv2.imread(str(p));
 if x is None: raise ValueError(str(p))
 x=cv2.resize(x,(512,512),interpolation=cv2.INTER_AREA).astype(np.float32)/127.5-1
 return torch.from_numpy(x.transpose(2,0,1))[None].to(d)
def main():
 p=argparse.ArgumentParser();p.add_argument('--input1',required=True);p.add_argument('--input2',required=True);p.add_argument('--woCoefNet_path',required=True);p.add_argument('--coef_path',required=True);p.add_argument('--output_dir',required=True);p.add_argument('--mode',choices=['adaptive','full','both'],default='both');p.add_argument('--limit',type=int,default=0);p.add_argument('--start',type=int,default=0);p.add_argument('--stop',type=int,default=0);p.add_argument('--device',default='cuda');a=p.parse_args();d=torch.device(a.device);torch.set_num_threads(2)
 out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);files1={x.name:x for x in Path(a.input1).glob('*') if x.is_file()};files2={x.name:x for x in Path(a.input2).glob('*') if x.is_file()};names=sorted(files1.keys()&files2.keys());names=names[:a.limit] if a.limit else names;names=names[a.start:a.stop or None]
 net,coef=Network().to(d),CoefNetwork().to(d);net.load_state_dict(torch.load(a.woCoefNet_path,map_location=d)['model'],strict=True);coef.load_state_dict(torch.load(a.coef_path,map_location=d)['model'],strict=True);net.eval();coef.eval()
 for i,name in enumerate(names,1):
  q=out/(name+'.json');r=json.loads(q.read_text()) if q.exists() else {'name':name}
  x,y=load(files1[name],d),load(files2[name],d)
  if a.mode in ('adaptive','both') and 'adaptive' not in r:
   al,ss,ps,_,st=adaptive_alpha_search(net,coef,x,y);r['adaptive']={'alpha':float(al),'ssim':float(ss),'psnr':float(ps),**st}
  if a.mode in ('full','both') and 'full' not in r:
   t=time.perf_counter();al,ss,ps,_=ternary_search(net,coef,x,y,low=-1,high=2,max_iter=20);r['full']={'alpha':float(al),'ssim':float(ss),'psnr':float(ps),'calls':40,'elapsed_seconds':float(time.perf_counter()-t)}
  q.write_text(json.dumps(r,indent=2));print(f'[{i}/{len(names)}] {name} adaptive={"adaptive" in r} full={"full" in r}',flush=True)
 rows=[json.loads((out/(n+'.json')).read_text()) for n in names if (out/(n+'.json')).exists()]
 if rows:
  summary={'n':len(rows),'mode':a.mode,'mean_calls':float(np.mean([r['adaptive']['calls'] for r in rows if 'adaptive' in r])) if any('adaptive' in r for r in rows) else None,'fallback_rate':float(np.mean([r['adaptive']['fallback'] for r in rows if 'adaptive' in r])) if any('adaptive' in r for r in rows) else None}
  if all('full' in r and 'adaptive' in r for r in rows):
   summary.update(adaptive_ssim=float(np.mean([r['adaptive']['ssim'] for r in rows])),full_ssim=float(np.mean([r['full']['ssim'] for r in rows])),adaptive_psnr=float(np.mean([r['adaptive']['psnr'] for r in rows])),full_psnr=float(np.mean([r['full']['psnr'] for r in rows])),adaptive_seconds=float(np.mean([r['adaptive']['elapsed_seconds'] for r in rows])),full_seconds=float(np.mean([r['full']['elapsed_seconds'] for r in rows])))
  (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
