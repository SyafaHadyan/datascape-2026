import numpy as np, pandas as pd, zipfile, io
from sklearn.metrics import f1_score, accuracy_score, recall_score
def load(p):
    if p.endswith('.zip'):
        z=zipfile.ZipFile(p); return {n[:-4]:np.load(io.BytesIO(z.read(n)),allow_pickle=True) for n in z.namelist() if n.endswith('.npy')}
    d=np.load(p,allow_pickle=True); return {k:d[k] for k in d.files}
P={21:'archive/probs-run21-convnext-lr3e-5.npz',23:'archive/probs-run23-convnext-small-lr3e-5.npz',24:'archive/probs-run24-dinov2-small.zip',26:'archive/probs-run26-clip-b16.zip',28:'archive/probs-run28-clip-b16-laion2b.zip',29:'archive/probs-run29-clip-b16-shiftaug.zip'}
R={k:load(v) for k,v in P.items()}
base=R[26]; y=base['oof_targets']; cls=list(base['classes'])
tails=[ '/'.join(str(x).split('/')[-2:]) for x in base['oof_filepaths']]
st=pd.read_csv('scratchpad/img_stats.csv')
tr=st[st.split=='train'].copy(); tr['key']=tr.label+'/'+tr.name
tr=tr.set_index('key')
m=tr.loc[tails].reset_index()
assert (m.label.map({c:i for i,c in enumerate(cls)}).values==y).all()
def feats(d):
    f=d.fmt.where(d.fmt.isin(['JPEG','WEBP']),'PNG/other')
    q=pd.cut(d.quality,[0,78,92,101],labels=['q<78','q78-92','q>92']).astype(object)
    q=np.where(f=='JPEG',q,'-')
    L=d[['w','h']].max(axis=1)
    sz=pd.cut(L,[0,300,800,1500,10**6],labels=['<300','300-800','800-1500','>1500']).astype(str)
    return f.astype(str)+'|'+pd.Series(q,index=d.index).astype(str)+'|'+sz
m['cell']=feats(m)
te=st[st.split=='test'].copy(); te['cell']=feats(te)
ptr=m.cell.value_counts(normalize=True); pte=te.cell.value_counts(normalize=True)
tab=pd.DataFrame({'train':ptr,'test':pte}).fillna(0)
tab['n_train']=m.cell.value_counts()
tab['w']=tab.test/tab.train.replace(0,np.nan)
print(tab.sort_values('test',ascending=False).round(3).head(14).to_string())
print('test mass in cells with no train rows', round(float(tab.loc[tab.train==0,'test'].sum()),4))
# weights, clip
w=m.cell.map(tab.w).fillna(0).clip(upper=20).values
print('effective sample size', round(w.sum()**2/(w**2).sum()), 'of', len(w))
O={k:v['oof_probs_flip'] for k,v in R.items()}
O['23+26+29']=(O[23]+O[26]+O[29])/3; O['23+26']=(O[23]+O[26])/2
rows=[]
for k,pr in O.items():
    p=pr.argmax(1)
    rows.append(dict(run=k,oof=f1_score(y,p,average='macro'),weighted=f1_score(y,p,average='macro',sample_weight=w),
                     acc=accuracy_score(y,p),w_acc=accuracy_score(y,p,sample_weight=w)))
print(pd.DataFrame(rows).round(4).to_string(index=False))
# per quality-bucket performance
m['qb']=m.cell.str.split('|').str[1]; m['fb']=m.cell.str.split('|').str[0]; m['sb']=m.cell.str.split('|').str[2]
for col in ['fb','qb','sb']:
    print('\nby',col)
    out=[]
    for g,idx in m.groupby(col).groups.items():
        idx=np.array(list(idx)); r=dict(group=g,n=len(idx),Aman=(y[idx]==0).sum(),Lain=(y[idx]==1).sum(),Rusak=(y[idx]==2).sum())
        for k in (26,29,'23+26+29'):
            p=O[k][idx].argmax(1); r[f'acc{k}']=accuracy_score(y[idx],p)
            rr=recall_score(y[idx],p,labels=[0,2],average=None,zero_division=0); r[f'recR{k}']=rr[1]
        out.append(r)
    print(pd.DataFrame(out).round(3).to_string(index=False))

print("\n=== predicted class share on TEST vs true share in TRAIN, by bucket (blend 23+26+29) ===")
T={k:v['test_probs'] for k,v in R.items()}
tb=(T[23]+T[26]+T[29])/3; tp=tb.argmax(1)
assert (te.name.values==R[26]['test_filenames']).all() if hasattr(R[26]['test_filenames'][0],'__len__') else True
te=te.reset_index(drop=True)
te['pred']=np.array(cls)[tp]
te['qb']=te.cell.str.split('|').str[1]; te['fb']=te.cell.str.split('|').str[0]; te['sb']=te.cell.str.split('|').str[2]
for col in ['qb','fb','sb']:
    rows=[]
    for g in sorted(set(m[col])|set(te[col])):
        a=m[m[col]==g]; b=te[te[col]==g]
        r=dict(group=g,n_train=len(a),n_test=len(b))
        for c in cls:
            r['tr_'+c[:4]]=(a.label==c).mean() if len(a) else np.nan
            r['te_'+c[:4]]=(b.pred==c).mean() if len(b) else np.nan
        rows.append(r)
    print('\nby',col); print(pd.DataFrame(rows).round(3).to_string(index=False))
print('\noverall test predicted share', te.pred.value_counts(normalize=True).round(3).to_dict(), 'train true share', m.label.value_counts(normalize=True).round(3).to_dict())

print("\n=== Lainnya prior-matching candidate (assumes test Lainnya share equals train share, no test labels used) ===")
target=int(round(float(np.mean(y==1))*len(tp)))
def count(k):
    p=tb.copy(); p[:,1]*=k; return int((p.argmax(1)==1).sum())
lo,hi=1.0,50.0
for _ in range(40):
    mid=(lo+hi)/2
    if count(mid)<target: lo=mid
    else: hi=mid
k=hi; p=tb.copy(); p[:,1]*=k; newp=p.argmax(1)
print('train Lainnya share',round(float(np.mean(y==1)),4),'target count',target,'factor',round(k,2),'resulting count',int((newp==1).sum()))
import collections
print('label changes vs blend', collections.Counter(zip(np.array(cls)[tp][tp!=newp],np.array(cls)[newp][tp!=newp])))
# effect of same factor on OOF
ob=(O[23]+O[26]+O[29])/3; po=ob.copy(); po[:,1]*=k
print('OOF macro F1 blend',round(f1_score(y,ob.argmax(1),average='macro'),4),'with same factor',round(f1_score(y,po.argmax(1),average='macro'),4),
      'Lainnya F1 before/after',np.round(f1_score(y,ob.argmax(1),average=None)[1],4),np.round(f1_score(y,po.argmax(1),average=None)[1],4))
for kk in (1.5,2,3,4):
    q=ob.copy(); q[:,1]*=kk; t=tb.copy(); t[:,1]*=kk
    print('factor',kk,'OOF F1',round(f1_score(y,q.argmax(1),average='macro'),4),'test Lainnya count',int((t.argmax(1)==1).sum()))
out=pd.DataFrame({'filename':R[26]['test_filenames'],'label':np.array(cls)[newp]})
out.to_csv('submission-blend-run23-run26-run29-lainnya-prior-candidate.csv',index=False)
print(out.label.value_counts().to_dict())
