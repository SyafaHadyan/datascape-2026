import numpy as np, pandas as pd, zipfile, io
def load(p):
    if p.endswith('.zip'):
        z=zipfile.ZipFile(p); return {n[:-4]:np.load(io.BytesIO(z.read(n)),allow_pickle=True) for n in z.namelist() if n.endswith('.npy')}
    d=np.load(p,allow_pickle=True); return {k:d[k] for k in d.files}
R={k:load(v) for k,v in {23:'archive/probs-run23-convnext-small-lr3e-5.npz',26:'archive/probs-run26-clip-b16.zip',29:'archive/probs-run29-clip-b16-shiftaug.zip'}.items()}
cls=list(R[26]['classes']); y=R[26]['oof_targets']
st=pd.read_csv('scratchpad/img_stats.csv'); blob=np.load('scratchpad/shift_blob.npy')
st['cl']=blob[:,0]; st['nn']=blob[:,1]/255; st['adv']=blob[:,2]/255
te=st[st.split=='test'].reset_index(drop=True)
assert (te.name.values==R[26]['test_filenames']).all()
tb=(R[23]['test_probs']+R[26]['test_probs']+R[29]['test_probs'])/3
te['pred']=np.array(cls)[tb.argmax(1)]; te['conf']=tb.max(1)
tr=st[st.split=='train'].copy(); tr['key']=tr.label+'/'+tr.name
tails=['/'.join(str(x).split('/')[-2:]) for x in R[26]['oof_filepaths']]
tr=tr.set_index('key').loc[tails].reset_index()
ob=(R[23]['oof_probs_flip']+R[26]['oof_probs_flip']+R[29]['oof_probs_flip'])/3
tr['oofpred']=np.array(cls)[ob.argmax(1)]; tr['oofconf']=ob.max(1)
rows=[]
for c in range(40):
    a=tr[tr.cl==c]; b=te[te.cl==c]
    if len(b)==0 and len(a)==0: continue
    rows.append(dict(cl=c,n_tr=len(a),n_te=len(b),tr_Aman=(a.label=='Aman').mean(),tr_Rusak=(a.label=='Rusak').mean(),tr_Lain=(a.label=='Lainnya').mean(),
                     oof_acc=(a.oofpred==a.label).mean() if len(a) else np.nan,
                     te_Aman=(b.pred=='Aman').mean() if len(b) else np.nan,te_Rusak=(b.pred=='Rusak').mean() if len(b) else np.nan,te_Lain=(b.pred=='Lainnya').mean() if len(b) else np.nan,
                     conf_te=b.conf.mean() if len(b) else np.nan,conf_oof=a.oofconf.mean() if len(a) else np.nan,nn_te=b.nn.mean() if len(b) else np.nan))
cs=pd.DataFrame(rows); cs['ratio']=(cs.n_te/len(te))/(cs.n_tr/len(tr))
print(cs.sort_values('n_te',ascending=False).head(16).round(3).to_string(index=False))
print('\nclusters where test share is >2x train share, pooled')
hv=cs[cs.ratio>2]; print('n_test',int(hv.n_te.sum()),'of',len(te),'| train analogs',int(hv.n_tr.sum()))
idx=te.cl.isin(hv.cl); idt=tr.cl.isin(hv.cl)
print(' train label mix in those clusters',tr[idt].label.value_counts(normalize=True).round(3).to_dict(),'OOF acc',round((tr[idt].oofpred==tr[idt].label).mean(),3))
print(' test predicted mix in those clusters',te[idx].pred.value_counts(normalize=True).round(3).to_dict(),'mean conf',round(te[idx].conf.mean(),3),'| OOF conf there',round(tr[idt].oofconf.mean(),3))
print(' rest of test predicted mix',te[~idx].pred.value_counts(normalize=True).round(3).to_dict())
# adversarial score quantiles vs predictions on test
te['advq']=pd.qcut(te.adv,5,labels=False)
print('\ntest predicted share by adversarial-score quintile (0 = most train-like, 4 = most test-like)')
print(pd.crosstab(te.advq,te.pred,normalize='index').round(3).to_string()); print('mean conf by quintile',te.groupby('advq').conf.mean().round(3).to_dict())
tr['advq']=pd.qcut(tr.adv,5,labels=False)
print('\ntrain: OOF accuracy and macro-ish by adversarial quintile'); print(tr.groupby('advq').apply(lambda d: pd.Series({'n':len(d),'acc':(d.oofpred==d.label).mean(),'Rusak_rate':(d.label=='Rusak').mean(),'Lain_rate':(d.label=='Lainnya').mean()}),include_groups=False).round(3).to_string())
