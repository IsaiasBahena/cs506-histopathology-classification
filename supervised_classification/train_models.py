"""Person 4: reproducible preliminary validation experiments (no test evaluation)."""
import argparse, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

SEED = 4266457447
META = ['id','path','label','dataset']

def markdown_table(df, index=False):
 if index: df=df.reset_index()
 header='| '+' | '.join(map(str,df.columns))+' |'
 sep='| '+' | '.join(['---']*len(df.columns))+' |'
 rows=['| '+' | '.join(str(v) for v in row)+' |' for row in df.itertuples(index=False,name=None)]
 return '\n'.join([header,sep]+rows)

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--features', default='combined_features.csv')
 p.add_argument('--out', default='results')
 p.add_argument('--train', help='Optional canonical train.csv; requires --val')
 p.add_argument('--val', help='Optional canonical val.csv; requires --train')
 a=p.parse_args(); out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 df=pd.read_csv(a.features)
 assert set(META)<=set(df.columns)
 assert not df[META].isna().any().any()
 assert not df.duplicated(['dataset','id']).any()
 assert set(df.dataset)=={'NCT-CRC-HE-100K','CRC-VAL-HE-7K'}
 cols=[c for c in df if c not in META]
 assert len(cols)==162, 'Expected the documented 162 handcrafted features'
 assert np.isfinite(df[cols].to_numpy()).all()
 dev=df[df.dataset=='NCT-CRC-HE-100K'].copy()
 assert len(dev.label.unique())==9
 if bool(a.train)!=bool(a.val):raise ValueError('Provide both --train and --val')
 if a.train:
  def align(path):
   split=pd.read_csv(path); assert not split.duplicated(['dataset','id']).any()
   assert (split.dataset=='NCT-CRC-HE-100K').all()
   joined=split[['dataset','id']].merge(dev,on=['dataset','id'],validate='one_to_one')
   assert len(joined)==len(split)
   return joined
  tr,va=align(a.train),align(a.val); status='canonical files supplied'
 else:
  tr,va=train_test_split(dev,test_size=.2,random_state=SEED,stratify=dev.label)
  status='PROVISIONAL: split generated from feature-table row order; verify IDs with Person 1/3 before SVD comparisons'
 keys=lambda d:set(map(tuple,d[['dataset','id']].to_numpy()))
 assert not keys(tr)&keys(va)
 assert keys(tr)|keys(va)==keys(dev)
 tr[META].to_csv(out/'train.csv',index=False);va[META].to_csv(out/'val.csv',index=False)
 manifest=df[META].copy();tk=keys(tr);vk=keys(va)
 manifest['split']=['train' if k in tk else 'val' if k in vk else 'test' for k in map(tuple,manifest[['dataset','id']].to_numpy())]
 manifest.to_csv(out/'split_manifest.csv',index=False)
 groups={
  'RGB':[c for c in cols if c.startswith(('r_hist_','g_hist_','b_hist_'))],
  'GLCM':[c for c in cols if c.startswith('glcm_')],
  'LBP':[c for c in cols if c.startswith('lbp_')],
  'Gabor':[c for c in cols if c.startswith('gabor_')]}
 groups['RGB_GLCM']=groups['RGB']+groups['GLCM']
 groups['RGB_LBP']=groups['RGB']+groups['LBP']
 groups['Texture']=groups['GLCM']+groups['LBP']+groups['Gabor']
 groups['All_handcrafted']=cols
 labels=sorted(dev.label.unique()); results=[];best=-1;best_info=None
 def evaluate(name,group,model):
  nonlocal best,best_info
  cc=groups[group]; start=time.perf_counter()
  with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=4):
   warnings.simplefilter('always',ConvergenceWarning)
   model.fit(tr[cc],tr.label)
   fit_seconds=time.perf_counter()-start;pred=model.predict(va[cc])
  tag=name+'__'+group
  row={'model':name,'features':group,'n_features':len(cc),'accuracy':accuracy_score(va.label,pred),'macro_f1':f1_score(va.label,pred,average='macro'),'weighted_f1':f1_score(va.label,pred,average='weighted'),'fit_seconds':fit_seconds,'convergence_warning':any(issubclass(w.category,ConvergenceWarning) for w in caught)}
  results.append(row);pd.DataFrame(results).to_csv(out/'model_comparison.csv',index=False)
  report=classification_report(va.label,pred,labels=labels,output_dict=True,zero_division=0)
  pd.DataFrame(report).T.to_csv(out/(tag+'_per_class.csv'))
  cm=confusion_matrix(va.label,pred,labels=labels)
  pd.DataFrame(cm,index=labels,columns=labels).to_csv(out/(tag+'_confusion.csv'))
  if row['macro_f1']>best:
   best=row['macro_f1'];best_info=(tag,cm,pred.copy(),report)
  print(json.dumps(row),flush=True)
 evaluate('Dummy','RGB',DummyClassifier(strategy='most_frequent'))
 for group in groups:
  evaluate('LogisticRegression',group,make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,C=1.0,solver='lbfgs',random_state=SEED)))
 evaluate('DecisionTree','All_handcrafted',DecisionTreeClassifier(random_state=SEED))
 for group in ['RGB','GLCM','LBP','Gabor','All_handcrafted']:
  evaluate('RandomForest',group,RandomForestClassifier(n_estimators=100,n_jobs=4,random_state=SEED))
 table=pd.DataFrame(results).sort_values('macro_f1',ascending=False)
 tag,cm,pred,report=best_info
 fig,ax=plt.subplots(figsize=(9,8));ConfusionMatrixDisplay(cm,display_labels=labels).plot(ax=ax,cmap='Blues',colorbar=False,values_format='d');ax.set_title('Validation: '+tag);fig.tight_layout();fig.savefig(out/'best_confusion_matrix.png',dpi=180);plt.close(fig)
 va[META].assign(prediction=pred).to_csv(out/'best_validation_predictions.csv',index=False)
 ax=table.sort_values('macro_f1').plot.barh(x='features',y='macro_f1',legend=False,figsize=(9,7))
 ax.set_yticklabels(table.sort_values('macro_f1').apply(lambda r:r['model']+' / '+r['features'],axis=1));ax.set_xlim(0,1);ax.set_xlabel('Validation macro F1');ax.figure.tight_layout();ax.figure.savefig(out/'model_comparison.png',dpi=180);plt.close(ax.figure)
 baseline=next(r for r in results if r['model']=='LogisticRegression' and r['features']=='RGB')
 config={'split_status':status,'split_seed':SEED,'train_rows':len(tr),'val_rows':len(va),'held_out_test_rows':len(df)-len(dev),'test_evaluated':False,'feature_groups':groups,'sklearn_version':sklearn.__version__,'pandas_version':pd.__version__,'numpy_version':np.__version__,'tuning':'No hyperparameter search; fixed baseline settings in source','brightness':'No explicit mean_brightness or brightness_std columns supplied','svd':'Not supplied; not evaluated'}
 (out/'run_config.json').write_text(json.dumps(config,indent=2))
 lines=['# Person 4 preliminary supervised results','',status,'',f'Training: {len(tr):,}; validation: {len(va):,}; independent test: 7,180 rows excluded from modeling.','',f'RGB + Logistic Regression baseline macro F1: {baseline["macro_f1"]:.4f}.',f'Best: {tag}; macro F1 {best:.4f}; improvement {(best-baseline["macro_f1"])*100:.2f} percentage points.','',markdown_table(table),'','## Interpretation and limitations','- Results use an image-level provisional split. Canonical split alignment with SVD is not verified.','- Scalers were fitted only on training data. Labels, paths, IDs and dataset names were excluded from model inputs.','- The independent test set was not evaluated; these are internal validation scores, not final performance.','- No SVD, explicit brightness features, RBF SVM, feature selection or hyperparameter search was performed.','- STR is cancer-associated stroma; this is nine-class patch classification, not patient-level diagnosis.','- Macro F1 weights all nine classes equally.','', '## Best-model per-class results',markdown_table(pd.DataFrame({k:v for k,v in report.items() if k in labels}).T, index=True)]
 (out/'RESULTS.md').write_text('\n'.join(lines))
 print('DONE',tag,best,flush=True)
if __name__=='__main__':main()
