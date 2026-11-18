"""SVD-only classification on the supplied canonical SVD split."""
import argparse,json,time,warnings
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix

def main():
 p=argparse.ArgumentParser();p.add_argument('--svd',required=True);p.add_argument('--out',default='results');p.add_argument('--manifest',required=True);a=p.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);df=pd.read_csv(a.svd)
 assert not df.duplicated(['dataset','id']).any()
 m=pd.read_csv(a.manifest);z=m.merge(df[['dataset','id','split','label']],on=['dataset','id'],suffixes=('_old','_svd'),how='outer',validate='one_to_one',indicator=True)
 assert (z._merge=='both').all() and (z.split_old==z.split_svd).all() and (z.label_old==z.label_svd).all()
 assert ((df.split=='test')==(df.dataset=='CRC-VAL-HE-7K')).all()
 tr=df[df.split=='train'];va=df[df.split=='val'];assert len(tr)==80000 and len(va)==20000
 labels=sorted(tr.label.unique());rows=[]
 for k in [32,64,128,256]:
  cols=[f'svd_{i:03d}' for i in range(k)]
  assert np.isfinite(df[cols].to_numpy()).all()
  for name,model in [('LogisticRegression',make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000,solver='lbfgs',random_state=4266457447))),('RandomForest',RandomForestClassifier(n_estimators=100,n_jobs=4,random_state=4266457447))]:
   t=time.perf_counter()
   with warnings.catch_warnings(record=True) as caught,threadpool_limits(limits=4):
    warnings.simplefilter('always',ConvergenceWarning);model.fit(tr[cols],tr.label);seconds=time.perf_counter()-t;pred=model.predict(va[cols])
   tag=f'{name}__SVD_{k}';row={'model':name,'features':f'SVD_{k}','n_features':k,'accuracy':accuracy_score(va.label,pred),'macro_f1':f1_score(va.label,pred,average='macro'),'weighted_f1':f1_score(va.label,pred,average='weighted'),'fit_seconds':seconds,'convergence_warning':any(issubclass(w.category,ConvergenceWarning) for w in caught)}
   rows.append(row);pd.DataFrame(rows).to_csv(out/'svd_comparison.csv',index=False)
   pd.DataFrame(classification_report(va.label,pred,labels=labels,output_dict=True,zero_division=0)).T.to_csv(out/(tag+'_per_class.csv'))
   pd.DataFrame(confusion_matrix(va.label,pred,labels=labels),index=labels,columns=labels).to_csv(out/(tag+'_confusion.csv'))
   print(json.dumps(row),flush=True)
 print('DONE',flush=True)
if __name__=='__main__':main()
