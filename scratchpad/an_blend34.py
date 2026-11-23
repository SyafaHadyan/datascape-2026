import numpy as np, pandas as pd, zipfile, io, itertools
from sklearn.metrics import f1_score, recall_score
def load(p):
    if p.endswith('.zip'):
        z=zipfile.ZipFile(p); return {n[:-4]:np.load(io.BytesIO(z.read(n)),allow_pickle=True) for n in z.namelist() if n.endswith('.npy')}
    d=np.load(p,allow_pickle=True); return {k:d[k] for k in d.files}
P={23:'archive/probs-run23-convnext-small-lr3e-5.npz',26:'archive/probs-run26-clip-b16.zip',28:'archive/probs-run28-clip-b16-laion2b.zip',29:'archive/probs-run29-clip-b16-shiftaug.zip',30:'archive/probs-run30-siglip-b16.zip',31:'archive/probs-run31-siglip-b16-256.npz',32:'archive/probs-run32-siglip2-b16.zip',34:'archive/probs-run34-siglip2-b16-seed1-l40.npz'}
R={k:load(v) for k,v in P.items()}
tail=lambda r:['/'.join(str(x).split('/')[-2:]) for x in r['oof_filepaths']]
t0=tail(R[26]); print('rows ok',all(tail(R[k])==t0 for k in R),'test ok',all((R[k]['test_filenames']==R[26]['test_filenames']).all() for k in R),'targets ok',all((R[k]['oof_targets']==R[26]['oof_targets']).all() for k in R))
cls=list(R[26]['classes']); names=R[26]['test_filenames']
s=pd.read_csv('submission-5fold-focal-siglip2-b16-seed1-run34.csv'); sample_ok=(s['filename'].to_numpy()==names).all()
print('csv rows',len(s),'order matches probs test_filenames',sample_ok)
t34=np.array(cls)[R[34]['test_probs'].argmax(1)]; print('csv labels equal argmax of run34 test_probs (flip, no TTA scales beyond 224)',(s['label'].to_numpy()==t34).mean())
y=R[26]['oof_targets']; O={k:R[k]['oof_probs_flip'] for k in R}; T={k:R[k]['test_probs'] for k in R}
f=lambda p:f1_score(y,p.argmax(1),average='macro'); mix=lambda c,D:sum(D[k] for k in c)/len(c)
print('classes',cls)
print('single flip',{k:round(f(O[k]),4) for k in O}); print('run34 no-flip',round(f1_score(y,R[34]['oof_probs_tta'].argmax(1),average='macro'),4))
print('run34 per-class F1',np.round(f1_score(y,O[34].argmax(1),average=None),4),'Rusak recall',round(recall_score(y,O[34].argmax(1),labels=[2],average=None)[0],4))
print('run32+run34 pair',round(f(mix((32,34),O)),4))
for c in [(23,26,29),(23,26,29,30,31),(23,26,29,30,31,32),(23,26,29,30,31,34),(23,26,29,30,31,32,34),(26,29,30,32,34),(23,26,28,29,30,31,32,34)]: print(c,round(f(mix(c,O)),4))
rows=[(c,f(mix(c,O))) for n in (3,4,5) for c in itertools.combinations(O,n) if 34 in c]
print('top with 34'); [print(' ',c,round(x,4)) for c,x in sorted(rows,key=lambda r:-r[1])[:5]]
rng=np.random.default_rng(0); idx=[rng.integers(0,len(y),len(y)) for _ in range(1000)]
def boot(a,b):
    pa=a.argmax(1);pb=b.argmax(1); d=[f1_score(y[i],pa[i],average='macro')-f1_score(y[i],pb[i],average='macro') for i in idx]; return round(float(np.mean(d)),4),np.round(np.percentile(d,[2.5,97.5]),4).tolist()
b5=mix((23,26,29,30,31),O); b7=mix((23,26,29,30,31,32,34),O); b3=mix((23,26,29),O)
print('7-way vs 5-way',boot(b7,b5),'| 7-way vs submitted 3-way',boot(b7,b3))
t3=mix((23,26,29),T).argmax(1); t5=mix((23,26,29,30,31),T).argmax(1)
print('test counts run34',np.bincount(T[34].argmax(1),minlength=3),'agree 34 vs 32',round(float((T[34].argmax(1)==T[32].argmax(1)).mean()),4))
for c in [(23,26,29,30,31,34),(23,26,29,30,31,32,34)]:
    t=mix(c,T).argmax(1); print(c,'test counts',np.bincount(t,minlength=3),'changed vs 3-blend',int((t!=t3).sum()),'vs 5-blend',int((t!=t5).sum()))
for c,tag in [((23,26,29,30,31,32,34),'run23-run26-run29-run30-run31-run32-run34'),((23,26,29,30,31,34),'run23-run26-run29-run30-run31-run34')]:
    pd.DataFrame({'filename':names,'label':np.array(cls)[mix(c,T).argmax(1)]}).to_csv(f'submission-blend-{tag}-equal.csv',index=False)
