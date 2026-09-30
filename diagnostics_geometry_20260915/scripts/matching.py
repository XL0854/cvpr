import bootstrap
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from scipy.spatial import cKDTree


def sample_field(field, pixels):
    """RoMa uses align_corners=False pixel-centre normalized coordinates."""
    grid=2*(pixels+.5)/512-1
    if field.ndim==2:field=field[...,None]
    return F.grid_sample(field.permute(2,0,1)[None],grid[None,None],align_corners=False)[0,:,0].T


def query_teacher(warp, certainty, pixels):
    width=warp.shape[1]//2
    ab,ba=warp[:,:width,2:],warp[:,width:,:2]
    ca,cb=certainty[:,:width],certainty[:,width:]
    q=(sample_field(ab,pixels)+1)*256-.5
    back=(sample_field(ba,q)+1)*256-.5
    score=sample_field(ca,pixels)[:,0]
    reverse_score=sample_field(cb,q)[:,0]
    cycle=(back-pixels).norm(dim=-1)
    valid=((pixels>=2)&(pixels<=509)&(q>=2)&(q<=509)).all(-1)
    valid &= torch.isfinite(q).all(-1)&torch.isfinite(cycle)
    keep=valid&(score>=.5)&(reverse_score>=.5)&(cycle<=2)
    return q,score,cycle,keep


def teacher_matches(warp, certainty, seed):
    a=torch.arange(12,501,8,device=warp.device,dtype=torch.float32)
    y,x=torch.meshgrid(a,a,indexing='ij');p=torch.stack((x,y),-1).reshape(-1,2)
    q,c,cyc,keep=query_teacher(warp,certainty,p)
    stats=dict(candidates=len(p),filtered=int(keep.sum()),filtered_fraction=float(keep.float().mean()))
    p,q,c,cyc=[x[keep].cpu().numpy() for x in [p,q,c,cyc]]
    rng=np.random.default_rng(seed);selected=[]
    cells=(p//64).astype(int);cellid=cells[:,0]+8*cells[:,1]
    for cell in range(64):
        ids=np.flatnonzero(cellid==cell);rng.shuffle(ids);selected.extend(ids[:16])
    selected=np.asarray(selected,dtype=int)
    stats['occupied_cells']=int(len(np.unique(cellid)))
    return np.concatenate((p[selected],q[selected]),1),c[selected],cyc[selected],stats


def sift_audit(im1, im2):
    sift=cv2.SIFT_create(nfeatures=6000)
    k1,d1=sift.detectAndCompute(cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY),None)
    k2,d2=sift.detectAndCompute(cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY),None)
    empty=np.empty((0,4),np.float32)
    if d1 is None or d2 is None or len(d1)<2 or len(d2)<2:return empty
    bf=cv2.BFMatcher()
    def good(a,b):
        return {m[0].queryIdx:m[0].trainIdx for m in bf.knnMatch(a,b,k=2)
                if len(m)==2 and m[0].distance<.7*m[1].distance}
    f,r=good(d1,d2),good(d2,d1)
    pairs=[(i,j) for i,j in f.items() if r.get(j)==i]
    if len(pairs)<16:return empty
    pts=np.array([[*k1[i].pt,*k2[j].pt] for i,j in pairs],np.float32)
    cv2.setRNGSeed(20260915)
    fundamental,mask=cv2.findFundamentalMat(pts[:,:2],pts[:,2:],cv2.USAC_MAGSAC,1.,.999,10000)
    if fundamental is None or mask is None:return empty
    return pts[mask.ravel().astype(bool)]


def split_matches(matches, independent, seed):
    if not len(matches):return np.array([],int),np.array([],int)
    rng=np.random.default_rng(seed)
    cells=(matches[:,:2]//64).astype(int);cellid=cells[:,0]+8*cells[:,1]
    occupied=np.unique(cellid);rng.shuffle(occupied)
    heldcells=occupied[:max(1,int(np.ceil(.25*len(occupied))))]
    held=np.flatnonzero(np.isin(cellid,heldcells));train=np.flatnonzero(~np.isin(cellid,heldcells))
    # Independent SIFT positions never enter the fit, even if teacher agrees.
    guard=np.concatenate((matches[held],independent),axis=0)
    if len(train) and len(guard):
        distance1=cKDTree(guard[:,:2]).query(matches[train,:2])[0]
        distance2=cKDTree(guard[:,2:]).query(matches[train,2:])[0]
        train=train[(distance1>=8)&(distance2>=8)]
    return train,held
