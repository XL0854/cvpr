"""Freeze the official UDIS-D testing pairs as a separate evaluation manifest."""
import bootstrap
from bootstrap import DIAG
import hashlib,json,random
from pathlib import Path

def sha(p):
 h=hashlib.sha256();h.update(p.read_bytes());return h.hexdigest()
def main():
 root=Path(bootstrap.ROOT)/'data/UDIS-D/testing'; out=DIAG/'runs/adaptive_alpha_test';out.mkdir(parents=True,exist_ok=True)
 one={p.stem:p for p in (root/'input1').glob('*.jpg')};two={p.stem:p for p in (root/'input2').glob('*.jpg')};samples=[]
 for stem in sorted(one.keys()&two.keys()):
  samples.append(dict(id='udis_test_'+stem,domain='udis_test',input1=str(one[stem]),input2=str(two[stem]),hashes=[sha(one[stem]),sha(two[stem])],native_shapes=[]))
 if not samples:raise RuntimeError('No UDIS testing pairs')
 random.Random(20260917).shuffle(samples); selected=samples[:100]
 (out/'manifest_100.json').write_text(json.dumps(dict(samples=selected,source='official UDIS-D testing split; deterministic 100-pair evaluation sample',scene_independent_from_training=True,seed=20260917),indent=2))
 print('TEST_MANIFEST',len(samples),'SELECTED',len(selected),out/'manifest_100.json')
if __name__=='__main__':main()
