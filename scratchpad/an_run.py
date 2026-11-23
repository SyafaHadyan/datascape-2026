"""Usage, an_run.py <run number> <probs path> <submission csv>. Checks alignment, scores the run and tests blends with it."""
import sys, numpy as np, pandas as pd, zipfile, io, itertools
from sklearn.metrics import f1_score, recall_score
def load(p):
    if p.endswith('.zip'):
        z=zipfile.ZipFile(p); return {n[:-4]:np.load(io.BytesIO(z.read(n)),allow_pickle=True) for n in z.namelist() if n.endswith('.npy')}
    d=np.load(p,allow_pickle=True); return {k:d[k] for k in d.files}
N=int(sys.argv[1]); newp=sys.argv[2]; newcsv=sys.argv[3]
P={23:'archive/probs-run23-convnext-small-lr3e-5.npz',26:'archive/probs-run26-clip-b16.zip',28:'archive/probs-run28-clip-b16-laion2b.zip',29:'archive/probs-run29-clip-b16-shiftaug.zip',30:'archive/probs-run30-siglip-b16.zip',31:'archive/probs-run31-siglip-b16-256.npz',32:'archive/probs-run32-siglip2-b16.zip',34:'archive/probs-run34-siglip2-b16-seed1-l40.npz',35:'archive/probs-run35-siglip2-shiftaug.npz',37:'archive/probs-run37-siglip-v1-seed1.npz',38:'archive/probs-run38-clip-dfn2b.npz',39:'archive/probs-run39-convnext-base-clip256.npz',41:'archive/probs-run41-clip-laion2b-ftin12k.npz',44:'archive/probs-run44-dinov3-b16.npz',43:'archive/probs-run43-dinov2-b14.npz',42:'archive/probs-run42-clip-datacompxl.npz',40:'archive/probs-run40-clip-metaclip.npz',45:'archive/probs-run45-clip-openai-ftin12k.npz',59:'archive/probs-run59-dinov3-b16-seed2.npz',60:'archive/probs-run60-convnext-base-dinov3.npz'}
P[N]=newp
R={k:load(v) for k,v in P.items()}
tail=lambda r:['/'.join(str(x).split('/')[-2:]) for x in r['oof_filepaths']]
t0=tail(R[26]); print('rows ok',all(tail(R[k])==t0 for k in R),'test ok',all((R[k]['test_filenames']==R[26]['test_filenames']).all() for k in R),'targets ok',all((R[k]['oof_targets']==R[26]['oof_targets']).all() for k in R))
cls=[str(c) for c in R[26]['classes']]; names=R[26]['test_filenames']
s=pd.read_csv(newcsv); print('csv rows',len(s),'order ok',(s.filename.to_numpy()==names).all(),'labels equal this run argmax',(s.label.to_numpy()==np.array(cls)[R[N]['test_probs'].argmax(1)]).mean())
y=R[26]['oof_targets']; O={k:R[k]['oof_probs_flip'] for k in R}; T={k:R[k]['test_probs'] for k in R}
f=lambda p:f1_score(y,p.argmax(1),average='macro'); mix=lambda c,D:sum(D[k] for k in c)/len(c)
print('single flip',{k:round(f(O[k]),4) for k in O}); print(f'run{N} no-flip',round(f1_score(y,R[N]['oof_probs_tta'].argmax(1),average='macro'),4))
print(f'run{N} per-class F1',np.round(f1_score(y,O[N].argmax(1),average=None),4),'Rusak recall',round(recall_score(y,O[N].argmax(1),labels=[2],average=None)[0],4))
print(f'pairs with run{N}',{k:round(f(mix((k,N),O)),4) for k in (23,26,30,32,34,37,38,39,40,42,43,44) if k!=N and k in P})
B7=(23,26,29,30,31,32,34); NEW=(30,31,32,34,37,38,39,44,43); ALL=tuple(k for k in P if k not in (28,))
cands=[B7,B7+(N,),NEW,NEW+(N,),ALL]
for c in cands: print(c,round(f(mix(c,O)),4))
rng=np.random.default_rng(0); idx=[rng.integers(0,len(y),len(y)) for _ in range(1000)]
def boot(a,b):
    pa=a.argmax(1);pb=b.argmax(1); d=[f1_score(y[i],pa[i],average='macro')-f1_score(y[i],pb[i],average='macro') for i in idx]; return round(float(np.mean(d)),4),np.round(np.percentile(d,[2.5,97.5]),4).tolist()
print(f'B7+{N} vs B7',boot(mix(B7+(N,),O),mix(B7,O)),'| NEW+run vs NEW',boot(mix(NEW+(N,),O),mix(NEW,O)))
tb=mix(B7,T).argmax(1); tn=T[N].argmax(1)
print(f'test counts [Aman,Lainnya,Rusak] run{N}',np.bincount(tn,minlength=3),'agreement with',{k:round(float((tn==T[k].argmax(1)).mean()),4) for k in (26,30,32,37,38,39) if k!=N},'submitted blend',round(float((tn==tb).mean()),4))
e=lambda k:O[k].argmax(1)!=y
print('share of this run OOF errors also made by run32',round(float((e(N)&e(32)).sum()/e(N).sum()),3))
for c in [B7+(N,),NEW+(N,),ALL]:
    t=mix(c,T).argmax(1); print(c,'test counts',np.bincount(t,minlength=3),'differs from submitted blend',int((t!=tb).sum()))
tags={B7+(N,):'seven-plus-run%d'%N,NEW+(N,):'new-models-plus-run%d'%N,ALL:'all-runs'}
for c,tag in tags.items():
    pd.DataFrame({'filename':names,'label':np.array(cls)[mix(c,T).argmax(1)]}).to_csv(f'submission-blend-{tag}-equal.csv',index=False)
