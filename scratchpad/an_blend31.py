import numpy as np, pandas as pd, zipfile, io, itertools
from sklearn.metrics import f1_score, recall_score
def load(p):
    if p.endswith('.zip'):
        z=zipfile.ZipFile(p); return {n[:-4]:np.load(io.BytesIO(z.read(n)),allow_pickle=True) for n in z.namelist() if n.endswith('.npy')}
    d=np.load(p,allow_pickle=True); return {k:d[k] for k in d.files}
P={23:'archive/probs-run23-convnext-small-lr3e-5.npz',26:'archive/probs-run26-clip-b16.zip',28:'archive/probs-run28-clip-b16-laion2b.zip',29:'archive/probs-run29-clip-b16-shiftaug.zip',30:'archive/probs-run30-siglip-b16.zip',31:'archive/probs-run31-siglip-b16-256.npz'}
R={k:load(v) for k,v in P.items()}
tail=lambda r:['/'.join(str(x).split('/')[-2:]) for x in r['oof_filepaths']]
t0=tail(R[26]); print('row order identical',{k:tail(R[k])==t0 for k in R},'test order',{k:bool((R[k]['test_filenames']==R[26]['test_filenames']).all()) for k in R}, 'targets', {k:bool((R[k]['oof_targets']==R[26]['oof_targets']).all()) for k in R})
y=R[26]['oof_targets']; cls=list(R[26]['classes']); O={k:R[k]['oof_probs_flip'] for k in R}; T={k:R[k]['test_probs'] for k in R}
f=lambda p:f1_score(y,p.argmax(1),average='macro')
print('single flip',{k:round(f(O[k]),4) for k in O})
print('run31 per-class F1',np.round(f1_score(y,O[31].argmax(1),average=None),4),'Rusak recall',round(recall_score(y,O[31].argmax(1),labels=[2],average=None)[0],4))
rows=[]
for n in (2,3,4):
    for c in itertools.combinations(O,n): rows.append((c,f(sum(O[k] for k in c)/n)))
print('top blends'); [print(' ',c,round(s,4)) for c,s in sorted(rows,key=lambda r:-r[1])[:8]]
print('with 31 only'); [print(' ',c,round(s,4)) for c,s in sorted([r for r in rows if 31 in r[0]],key=lambda r:-r[1])[:5]]
for c in [(23,26,29),(23,26,29,30),(23,26,29,30,31),(26,30,31),(30,31)]:
    print(c,round(f(sum(O[k] for k in c)/len(c)),4))
rng=np.random.default_rng(0); idx=[rng.integers(0,len(y),len(y)) for _ in range(1000)]
def boot(a,b):
    pa=a.argmax(1);pb=b.argmax(1); d=[f1_score(y[i],pa[i],average='macro')-f1_score(y[i],pb[i],average='macro') for i in idx]; return round(float(np.mean(d)),4),np.round(np.percentile(d,[2.5,97.5]),4).tolist()
mix=lambda c,D:sum(D[k] for k in c)/len(c)
print('31 vs 30',boot(O[31],O[30]),'| 31 vs 26',boot(O[31],O[26]))
print('23+26+29+30+31 vs 23+26+29',boot(mix((23,26,29,30,31),O),mix((23,26,29),O)))
print('23+26+29+30+31 vs 23+26+29+30',boot(mix((23,26,29,30,31),O),mix((23,26,29,30),O)))
print('test counts [Aman,Lain,Rusak]: run31',np.bincount(T[31].argmax(1),minlength=3))
for a,b in [(31,30),(31,26)]: print(f'test agreement {a} vs {b}',round(float((T[a].argmax(1)==T[b].argmax(1)).mean()),4))
t3=mix((23,26,29),T).argmax(1)
for c in [(23,26,29,30,31),(23,26,30,31),(26,30,31)]:
    t=mix(c,T).argmax(1); print(c,'test counts',np.bincount(t,minlength=3),'changed vs submitted 3-blend',int((t!=t3).sum()))
names=R[26]['test_filenames']
for c,tag in [((23,26,29,30,31),'run23-run26-run29-run30-run31'),((23,26,30,31),'run23-run26-run30-run31')]:
    pd.DataFrame({'filename':names,'label':np.array(cls)[mix(c,T).argmax(1)]}).to_csv(f'submission-blend-{tag}-equal.csv',index=False)
