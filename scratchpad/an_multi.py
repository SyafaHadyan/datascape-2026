import sys, numpy as np, pandas as pd, zipfile, io
from sklearn.metrics import f1_score, recall_score
def load(p):
    if p.endswith('.zip'):
        z=zipfile.ZipFile(p); return {n[:-4]:np.load(io.BytesIO(z.read(n)),allow_pickle=True) for n in z.namelist() if n.endswith('.npy')}
    d=np.load(p,allow_pickle=True); return {k:d[k] for k in d.files}
base={26:'archive/probs-run26-clip-b16.zip',30:'archive/probs-run30-siglip-b16.zip',31:'archive/probs-run31-siglip-b16-256.npz',32:'archive/probs-run32-siglip2-b16.zip',34:'archive/probs-run34-siglip2-b16-seed1-l40.npz',37:'archive/probs-run37-siglip-v1-seed1.npz',43:'archive/probs-run43-dinov2-b14.npz',44:'archive/probs-run44-dinov3-b16.npz',59:'archive/probs-run59-dinov3-b16-seed2.npz',61:'archive/probs-run61-dinov2-b14-seed2.npz'}
new={55:'run55-dinov3-large',36:'run36-siglip2-seed2',46:'run46-siglip2-lr5e-5',47:'run47-siglip2-lr2e-5'}
extra=dict(a.split('=') for a in sys.argv[1:])  # extra N=tag
for k,v in extra.items(): new[int(k)]=v
P=dict(base); P.update({k:f'archive/probs-{t}.npz' for k,t in new.items()})
R={k:load(v) for k,v in P.items()}
tail=lambda r:['/'.join(str(x).split('/')[-2:]) for x in r['oof_filepaths']]
t0=tail(R[26]); print('rows ok',all(tail(R[k])==t0 for k in R),'test ok',all((R[k]['test_filenames']==R[26]['test_filenames']).all() for k in R),'targets ok',all((R[k]['oof_targets']==R[26]['oof_targets']).all() for k in R))
cls=[str(c) for c in R[26]['classes']]; names=R[26]['test_filenames']; y=R[26]['oof_targets']
O={k:R[k]['oof_probs_flip'] for k in R}; T={k:R[k]['test_probs'] for k in R}
f=lambda S:f1_score(y,sum(O[k] for k in S).argmax(1),average='macro')
best=pd.read_csv('submission-blend-siglip5-plus-dino4-equal.csv'); B=(30,31,32,34,37,43,44,59,61)
tb=np.array(cls)[sum(T[k] for k in B).argmax(1)]; assert (tb==best.label.to_numpy()).all()
print('reference run34 0.9164, run32 0.9183 | best file members OOF',round(f(B),4))
for k,t in new.items():
    csv=pd.read_csv(f'submission-{t}.csv'); ok=(csv.filename.to_numpy()==names).all() and (csv.label.to_numpy()==np.array(cls)[T[k].argmax(1)]).all()
    p=O[k].argmax(1); tn=T[k].argmax(1)
    print(f"\nrun{k} {t} csv ok {ok}")
    print(f"  OOF {f((k,)):.4f} no-flip {f1_score(y,R[k]['oof_probs_tta'].argmax(1),average='macro'):.4f} per-class F1 {np.round(f1_score(y,p,average=None),4)} Rusak recall {recall_score(y,p,labels=[2],average=None)[0]:.4f}")
    print(f"  test [Aman,Lainnya,Rusak] {np.bincount(tn,minlength=3)} agreement with best file {float((np.array(cls)[tn]==best.label.to_numpy()).mean()):.4f}  OOF errors shared with run32 {float(((p!=y)&(O[32].argmax(1)!=y)).sum()/(p!=y).sum()):.3f}")
    S=B+(k,); t2=np.array(cls)[sum(T[j] for j in S).argmax(1)]
    print(f"  best-file members plus this run OOF {f(S):.4f} ({f(S)-f(B):+.4f}), differs from best file on {int((t2!=best.label.to_numpy()).sum())} labels")
