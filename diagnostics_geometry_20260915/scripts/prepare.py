"""Fix a diagnostic manifest before observing teacher/optimization outcomes."""
import bootstrap
from bootstrap import ROOT, DIAG
import argparse
import hashlib
import json
import random
from pathlib import Path
import cv2
import numpy as np


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def dhash(im):
    a=cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),(9,8))
    return a[:,1:]>a[:,:-1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--per_domain',type=int,default=50)
    p.add_argument('--run',default='pilot100');args=p.parse_args()
    out=DIAG/'runs'/args.run;out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():
        raise RuntimeError('Manifest exists; refusing to change sample selection')
    oldroot=Path('/workspace/python_pro/RopStitch-main/fastrop_stitch/runs')
    old_hashes=set();visual=[]
    for path in oldroot.glob('*/manifest.json'):
        data=json.loads(path.read_text())
        for s in data.get('samples',[]):
            old_hashes.update(s.get('content_hashes',[]))
            for k in ['input1','input2']:
                if k in s and Path(s[k]).exists():
                    im=cv2.imread(s[k])
                    if im is not None:visual.append(dhash(im))
    selected=[];rejections={}
    for domain,root in [('udis_train',ROOT/'data/UDIS-D/training'),
                         ('classic_development',Path('/workspace/python_pro/roptest/Data/stitch_real'))]:
        paths=sorted((root/'input1').glob('*'));random.Random(20260915).shuffle(paths)
        count=0
        for a in paths:
            b=root/'input2'/a.name
            if not a.is_file() or not b.is_file():continue
            h=[sha(a),sha(b)]
            if any(x in old_hashes for x in h):
                rejections['prior_or_exact']=rejections.get('prior_or_exact',0)+1;continue
            ims=[cv2.imread(str(x)) for x in [a,b]]
            if any(x is None for x in ims):continue
            ds=[dhash(im) for im in ims]
            if any(np.count_nonzero(x!=y)<=4 for x in ds for y in visual):
                rejections['near_duplicate']=rejections.get('near_duplicate',0)+1;continue
            selected.append(dict(id=domain+'_'+a.stem,domain=domain,input1=str(a.resolve()),
                                 input2=str(b.resolve()),hashes=h,native_shapes=[list(x.shape) for x in ims]))
            old_hashes.update(h);visual.extend(ds);count+=1
            if count>=args.per_domain:break
        if count<args.per_domain:raise RuntimeError(f'Only {count} usable pairs in {domain}')
    protocol=dict(seed=20260915,alpha=.5,images='512-square diagnostic coordinates; NOT native-resolution paper benchmark',
        teacher='RoMa outdoor, 560 coarse / 864 fine, bidirectional, local weights',
        correspondence_filter='certainty>=0.5 in both directions; cycle<=2px at 512; 8x8 spatial balancing',
        split='25% spatial cells held out, 8px train exclusion around holdout/SIFT audit coordinates in BOTH views',
        optimization=dict(steps=150,lr=.15,max_delta_per_axis_px=20,anchor=.005,smoothness=.02,fold=10),
        selection='random before outcomes; excludes hashes from accessible previous manifests and near duplicate dHash<=4',
        limitations=['No scene identity annotations: exact/reversed and near-duplicate filtering does not prove scene independence',
                     'Classic is an already explored development domain, NOT an untouched final test set',
                     'Teacher-held-out points are not independent ground truth; SIFT is an independent algorithmic proxy, not human labels',
                     'Per-pair TPS fitting diagnoses capacity only, not student training/generalization or deployable latency'])
    (out/'manifest.json').write_text(json.dumps(dict(samples=selected,rejections=rejections),indent=2))
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    print('MANIFEST_FIXED',len(selected),out)


if __name__=='__main__':main()
