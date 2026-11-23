import numpy as np, pandas as pd, zipfile, io, itertools
from sklearn.metrics import f1_score, recall_score
def load(p):
    if p.endswith('.zip'):
        z=zipfile.ZipFile(p); return {n[:-4]:np.load(io.BytesIO(z.read(n)),allow_pickle=True) for n in z.namelist() if n.endswith('.npy')}
    d=np.load(p,allow_pickle=True); return {k:d[k] for k in d.files}
P={23:'archive/probs-run23-convnext-small-lr3e-5.npz',26:'archive/probs-run26-clip-b16.zip',28:'archive/probs-run28-clip-b16-laion2b.zip',29:'archive/probs-run29-clip-b16-shiftaug.zip',30:'archive/probs-run30-siglip-b16.zip'}
R={k:load(v) for k,v in P.items()}
tail=lambda r:['/'.join(str(x).split('/')[-2:]) for x in r['oof_filepaths']]
t0=tail(R[26]); print('row order identical', {k:tail(R[k])==t0 for k in R}, 'test order', {k:bool((R[k]['test_filenames']==R[26]['test_filenames']).all()) for k in R})
y=R[26]['oof_targets']; cls=list(R[26]['classes']); O={k:R[k]['oof_probs_flip'] for k in R}; T={k:R[k]['test_probs'] for k in R}
f=lambda p:f1_score(y,p.argmax(1),average='macro')
print('single (flip)',{k:round(f(O[k]),4) for k in O})
print('run30 per-class F1',np.round(f1_score(y,O[30].argmax(1),average=None),4),'Rusak recall',round(recall_score(y,O[30].argmax(1),labels=[2],average=None)[0],4))
rows=[]
for n in (2,3,4):
    for c in itertools.combinations(O,n):
        if 30 in c: rows.append((c,f(sum(O[k] for k in c)/n)))
for c,s in sorted(rows,key=lambda r:-r[1])[:8]: print(c,round(s,4))
for c in [(23,26,29),(23,26,29,30),(23,26,30),(23,30)]:
    print(c,round(f(sum(O[k] for k in c)/len(c)),4))
rng=np.random.default_rng(0); idx=[rng.integers(0,len(y),len(y)) for _ in range(1000)]
def boot(a,b):
    pa=a.argmax(1);pb=b.argmax(1); d=[f1_score(y[i],pa[i],average='macro')-f1_score(y[i],pb[i],average='macro') for i in idx]; return round(np.mean(d),4),np.round(np.percentile(d,[2.5,97.5]),4)
b3=sum(O[k] for k in (23,26,29))/3; b4=sum(O[k] for k in (23,26,29,30))/4
print('30 vs 26',boot(O[30],O[26])); print('23+26+29+30 vs 23+26+29',boot(b4,b3))
t3=sum(T[k] for k in (23,26,29))/3; t4=sum(T[k] for k in (23,26,29,30))/4
print('test agreement 30 vs 26',round((T[30].argmax(1)==T[26].argmax(1)).mean(),4),'4-blend vs submitted 3-blend',round((t4.argmax(1)==t3.argmax(1)).mean(),4),'changed',int((t4.argmax(1)!=t3.argmax(1)).sum()))
print('test counts run30',np.bincount(T[30].argmax(1),minlength=3),'4-blend',np.bincount(t4.argmax(1),minlength=3))
names=R[26]['test_filenames']
for c,tag in [((23,26,29,30),'run23-run26-run29-run30'),((23,26,30),'run23-run26-run30')]:
    t=sum(T[k] for k in c)/len(c); pd.DataFrame({'filename':names,'label':np.array(cls)[t.argmax(1)]}).to_csv(f'submission-blend-{tag}-equal.csv',index=False)
