import numpy as np, pandas as pd, zipfile, io, itertools
from sklearn.metrics import f1_score, recall_score
def load(p):
    if p.endswith('.zip'):
        z=zipfile.ZipFile(p); return {n[:-4]:np.load(io.BytesIO(z.read(n)),allow_pickle=True) for n in z.namelist() if n.endswith('.npy')}
    d=np.load(p,allow_pickle=True); return {k:d[k] for k in d.files}
P={23:'archive/probs-run23-convnext-small-lr3e-5.npz',26:'archive/probs-run26-clip-b16.zip',28:'archive/probs-run28-clip-b16-laion2b.zip',29:'archive/probs-run29-clip-b16-shiftaug.zip',30:'archive/probs-run30-siglip-b16.zip',31:'archive/probs-run31-siglip-b16-256.npz',32:'archive/probs-run32-siglip2-b16.zip'}
R={k:load(v) for k,v in P.items()}
tail=lambda r:['/'.join(str(x).split('/')[-2:]) for x in r['oof_filepaths']]
t0=tail(R[26]); print('rows ok',all(tail(R[k])==t0 for k in R),'test ok',all((R[k]['test_filenames']==R[26]['test_filenames']).all() for k in R),'targets ok',all((R[k]['oof_targets']==R[26]['oof_targets']).all() for k in R))
y=R[26]['oof_targets']; cls=list(R[26]['classes']); O={k:R[k]['oof_probs_flip'] for k in R}; T={k:R[k]['test_probs'] for k in R}
f=lambda p:f1_score(y,p.argmax(1),average='macro'); mix=lambda c,D:sum(D[k] for k in c)/len(c)
print('single flip',{k:round(f(O[k]),4) for k in O})
print('run32 per-class F1',np.round(f1_score(y,O[32].argmax(1),average=None),4),'Rusak recall',round(recall_score(y,O[32].argmax(1),labels=[2],average=None)[0],4))
rows=[(c,f(mix(c,O))) for n in (2,3,4,5) for c in itertools.combinations(O,n) if 32 in c]
print('top with 32'); [print(' ',c,round(s,4)) for c,s in sorted(rows,key=lambda r:-r[1])[:6]]
for c in [(23,26,29),(23,26,29,30,31),(23,26,29,30,31,32),(23,26,29,30,32),(26,29,30,31,32),(30,31,32),(23,26,28,29,30,31,32)]: print(c,round(f(mix(c,O)),4))
rng=np.random.default_rng(0); idx=[rng.integers(0,len(y),len(y)) for _ in range(1000)]
def boot(a,b):
    pa=a.argmax(1);pb=b.argmax(1); d=[f1_score(y[i],pa[i],average='macro')-f1_score(y[i],pb[i],average='macro') for i in idx]; return round(float(np.mean(d)),4),np.round(np.percentile(d,[2.5,97.5]),4).tolist()
print('32 vs 30',boot(O[32],O[30]),'| 32 vs 26',boot(O[32],O[26]))
b5=mix((23,26,29,30,31),O); b6=mix((23,26,29,30,31,32),O); b3=mix((23,26,29),O)
print('6-way vs 5-way',boot(b6,b5),'| 6-way vs submitted 3-way',boot(b6,b3))
print('test counts [Aman,Lain,Rusak] run32',np.bincount(T[32].argmax(1),minlength=3),'agreement 32 vs 30',round(float((T[32].argmax(1)==T[30].argmax(1)).mean()),4),'vs 26',round(float((T[32].argmax(1)==T[26].argmax(1)).mean()),4))
t3=mix((23,26,29),T).argmax(1); t5=mix((23,26,29,30,31),T).argmax(1)
for c in [(23,26,29,30,31,32),(23,26,29,30,32),(26,29,30,31,32)]:
    t=mix(c,T).argmax(1); print(c,'test counts',np.bincount(t,minlength=3),'changed vs 3-blend',int((t!=t3).sum()),'vs 5-blend',int((t!=t5).sum()))
names=R[26]['test_filenames']
for c,tag in [((23,26,29,30,31,32),'run23-run26-run29-run30-run31-run32'),((23,26,29,30,32),'run23-run26-run29-run30-run32')]:
    pd.DataFrame({'filename':names,'label':np.array(cls)[mix(c,T).argmax(1)]}).to_csv(f'submission-blend-{tag}-equal.csv',index=False)
