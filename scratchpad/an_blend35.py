import numpy as np, pandas as pd, zipfile, io, itertools
from sklearn.metrics import f1_score, recall_score
def load(p):
    if p.endswith('.zip'):
        z=zipfile.ZipFile(p); return {n[:-4]:np.load(io.BytesIO(z.read(n)),allow_pickle=True) for n in z.namelist() if n.endswith('.npy')}
    d=np.load(p,allow_pickle=True); return {k:d[k] for k in d.files}
P={23:'archive/probs-run23-convnext-small-lr3e-5.npz',26:'archive/probs-run26-clip-b16.zip',28:'archive/probs-run28-clip-b16-laion2b.zip',29:'archive/probs-run29-clip-b16-shiftaug.zip',30:'archive/probs-run30-siglip-b16.zip',31:'archive/probs-run31-siglip-b16-256.npz',32:'archive/probs-run32-siglip2-b16.zip',34:'archive/probs-run34-siglip2-b16-seed1-l40.npz',35:'archive/probs-run35-siglip2-shiftaug.npz'}
R={k:load(v) for k,v in P.items()}
tail=lambda r:['/'.join(str(x).split('/')[-2:]) for x in r['oof_filepaths']]
t0=tail(R[26]); print('rows ok',all(tail(R[k])==t0 for k in R),'test ok',all((R[k]['test_filenames']==R[26]['test_filenames']).all() for k in R),'targets ok',all((R[k]['oof_targets']==R[26]['oof_targets']).all() for k in R))
cls=list(R[26]['classes']); names=R[26]['test_filenames']
s=pd.read_csv('submission-run35-siglip2-shiftaug.csv'); sample_ok=(s['filename'].to_numpy()==names).all()
print('csv rows',len(s),'order matches probs test_filenames',sample_ok)
t34=np.array(cls)[R[35]['test_probs'].argmax(1)]; print('csv labels equal argmax of run34 test_probs (flip, no TTA scales beyond 224)',(s['label'].to_numpy()==t34).mean())
y=R[26]['oof_targets']; O={k:R[k]['oof_probs_flip'] for k in R}; T={k:R[k]['test_probs'] for k in R}
f=lambda p:f1_score(y,p.argmax(1),average='macro'); mix=lambda c,D:sum(D[k] for k in c)/len(c)

print('classes',cls)
print('single flip',{k:round(f(O[k]),4) for k in O})
print('run35 no-flip',round(f1_score(y,R[35]['oof_probs_tta'].argmax(1),average='macro'),4))
print('run35 per-class F1',np.round(f1_score(y,O[35].argmax(1),average=None),4),'Rusak recall',round(recall_score(y,O[35].argmax(1),labels=[2],average=None)[0],4))
B7=(23,26,29,30,31,32,34)
print('seven-way (submitted, LB 0.81321)',round(f(mix(B7,O)),4))
for c in [B7+(35,),(30,31,32,34,35),(30,31,32,34),(32,34,35),(29,35),(26,29,30,31,32,34,35)]: print(c,round(f(mix(c,O)),4))
tb=mix(B7,T).argmax(1); t35=T[35].argmax(1); t34=T[34].argmax(1); t32=T[32].argmax(1)
print('test counts [Aman,Lainnya,Rusak] run35',np.bincount(t35,minlength=3),'run34',np.bincount(t34,minlength=3),'submitted blend',np.bincount(tb,minlength=3))
print('agreement run35 vs run34',round(float((t35==t34).mean()),4),'vs run32',round(float((t35==t32).mean()),4),'vs submitted blend',round(float((t35==tb).mean()),4),'| run34 vs run32',round(float((t34==t32).mean()),4))
for c in [B7+(35,),(30,31,32,34,35),(30,31,32,34)]:
    t=mix(c,T).argmax(1); print(c,'test counts',np.bincount(t,minlength=3),'differs from submitted blend',int((t!=tb).sum()))
names=R[26]['test_filenames']
for c,tag in [(B7+(35,),'run23-run26-run29-run30-run31-run32-run34-run35'),((30,31,32,34,35),'run30-run31-run32-run34-run35'),((30,31,32,34),'run30-run31-run32-run34')]:
    pd.DataFrame({'filename':names,'label':np.array(cls)[mix(c,T).argmax(1)]}).to_csv(f'submission-blend-{tag}-equal.csv',index=False)
