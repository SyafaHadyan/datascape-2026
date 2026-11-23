import numpy as np, pandas as pd, zipfile, io, itertools
from sklearn.metrics import f1_score, recall_score
def load(p):
    if p.endswith('.zip'):
        z=zipfile.ZipFile(p); return {n[:-4]:np.load(io.BytesIO(z.read(n)),allow_pickle=True) for n in z.namelist() if n.endswith('.npy')}
    d=np.load(p,allow_pickle=True); return {k:d[k] for k in d.files}
P={23:'archive/probs-run23-convnext-small-lr3e-5.npz',26:'archive/probs-run26-clip-b16.zip',28:'archive/probs-run28-clip-b16-laion2b.zip',29:'archive/probs-run29-clip-b16-shiftaug.zip',30:'archive/probs-run30-siglip-b16.zip',31:'archive/probs-run31-siglip-b16-256.npz',32:'archive/probs-run32-siglip2-b16.zip',34:'archive/probs-run34-siglip2-b16-seed1-l40.npz',35:'archive/probs-run35-siglip2-shiftaug.npz',37:'archive/probs-run37-siglip-v1-seed1.npz',38:'archive/probs-run38-clip-dfn2b.npz',39:'archive/probs-run39-convnext-base-clip256.npz'}
R={k:load(v) for k,v in P.items()}
tail=lambda r:['/'.join(str(x).split('/')[-2:]) for x in r['oof_filepaths']]
t0=tail(R[26]); print('rows ok',all(tail(R[k])==t0 for k in R),'test ok',all((R[k]['test_filenames']==R[26]['test_filenames']).all() for k in R),'targets ok',all((R[k]['oof_targets']==R[26]['oof_targets']).all() for k in R))
cls=[str(c) for c in R[26]['classes']]; names=R[26]['test_filenames']
s=pd.read_csv('submission-run39-convnext-base-clip256.csv'); print('csv rows',len(s),'order ok',(s.filename.to_numpy()==names).all(),'labels equal run39 argmax',(s.label.to_numpy()==np.array(cls)[R[39]['test_probs'].argmax(1)]).mean())
y=R[26]['oof_targets']; O={k:R[k]['oof_probs_flip'] for k in R}; T={k:R[k]['test_probs'] for k in R}
f=lambda p:f1_score(y,p.argmax(1),average='macro'); mix=lambda c,D:sum(D[k] for k in c)/len(c)
print('single flip',{k:round(f(O[k]),4) for k in O}); print('run39 no-flip',round(f1_score(y,R[39]['oof_probs_tta'].argmax(1),average='macro'),4))
print('run39 per-class F1',np.round(f1_score(y,O[39].argmax(1),average=None),4),'Rusak recall',round(recall_score(y,O[39].argmax(1),labels=[2],average=None)[0],4))
B7=(23,26,29,30,31,32,34)
print('pairs with 38 (OOF)',{k:round(f(mix((k,39),O)),4) for k in (23,26,30,32,34,37,38)})
for c in [B7,B7+(39,),B7+(38,39),B7+(37,38,39),(30,31,32,34,37,38,39),(30,31,32,34,37,38),(23,30,32,34,39),(23,26,29,30,31,32,34,37,38,39)]: print(c,round(f(mix(c,O)),4))
rng=np.random.default_rng(0); idx=[rng.integers(0,len(y),len(y)) for _ in range(1000)]
def boot(a,b):
    pa=a.argmax(1);pb=b.argmax(1); d=[f1_score(y[i],pa[i],average='macro')-f1_score(y[i],pb[i],average='macro') for i in idx]; return round(float(np.mean(d)),4),np.round(np.percentile(d,[2.5,97.5]),4).tolist()
print('B7+39 vs B7',boot(mix(B7+(39,),O),mix(B7,O)))
tb=mix(B7,T).argmax(1); t39=T[39].argmax(1)
print('test counts [Aman,Lainnya,Rusak] run39',np.bincount(t39,minlength=3),'agree with run23',round(float((t39==T[23].argmax(1)).mean()),4),'run28',round(float((t39==T[28].argmax(1)).mean()),4),'run32',round(float((t39==T[32].argmax(1)).mean()),4),'run37',round(float((t39==T[37].argmax(1)).mean()),4),'submitted blend',round(float((t39==tb).mean()),4))
print('OOF error overlap, share of run39 errors also made by run32',round(float(((O[39].argmax(1)!=y)&(O[32].argmax(1)!=y)).sum()/(O[39].argmax(1)!=y).sum()),3),'| run34 vs run32',round(float(((O[34].argmax(1)!=y)&(O[32].argmax(1)!=y)).sum()/(O[34].argmax(1)!=y).sum()),3))
for c in [B7+(39,),(30,31,32,34,37,38,39),(30,31,32,34,37,38)]:
    t=mix(c,T).argmax(1); print(c,'test counts',np.bincount(t,minlength=3),'differs from submitted blend',int((t!=tb).sum()))
for c,tag in [(B7+(39,),'run23-run26-run29-run30-run31-run32-run34-run39'),((30,31,32,34,37,38,39),'run30-run31-run32-run34-run37-run38-run39')]:
    pd.DataFrame({'filename':names,'label':np.array(cls)[mix(c,T).argmax(1)]}).to_csv(f'submission-blend-{tag}-equal.csv',index=False)
