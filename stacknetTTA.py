from pystacknet.pystacknet import StackNetClassifier
import pandas as pd
import numpy as np
from collections import OrderedDict
import xgboost
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.feature_selection import SelectFromModel,SelectPercentile
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from catboost import CatBoostClassifier
import lightgbm
from sklearn.linear_model import LogisticRegressionCV

TARGET_COL = "hospital_death"

train = pd.read_csv('train_fs.csv') 
test = pd.read_csv('test_fs.csv') 

train.drop(["encounter_id"],axis=1,inplace=True)
test.drop(["encounter_id"],axis=1,inplace=True)

print(train.drop(TARGET_COL,axis=1).columns.get_loc('apache_4a_hospital_death_prob'))
print(train.drop(TARGET_COL,axis=1).columns.get_loc('apache_4a_icu_death_prob'))
print(train.drop(TARGET_COL,axis=1).columns.get_loc('apache_hospital_minus_apache_icu'))
print(train.drop(TARGET_COL,axis=1).columns.get_loc('apache_icu_div_apache_hospital'))
print(train.drop(TARGET_COL,axis=1).columns.get_loc('age'))
print(train.drop(TARGET_COL,axis=1).columns.get_loc('ventilated_apache'))

models=[ 
        [
           xgboost.XGBRFClassifier(n_estimators=300,max_depth=50,tree_method="gpu_hist",verbose=10),
           xgboost.XGBRFClassifier(n_estimators=2000,max_depth=12,tree_method="gpu_hist",n_jobs=1), 
           SelectFromModel(CatBoostClassifier(iterations=2200, depth=10, objective="Logloss",nan_mode="Max",verbose=1000, task_type="GPU")),
           SelectFromModel(xgboost.XGBClassifier(n_estimators=3000,eta=0.02,booster='gbtree',max_depth=7,min_child_weight=3,colsample_bytree=1,objective='binary:logistic',tree_method="gpu_hist",n_jobs=1)),
           xgboost.XGBClassifier(n_estimators=2000,eta=0.01,booster='gbtree',max_depth=9,min_child_weight=1,colsample_bytree=0.5,objective='binary:logistic',tree_method="gpu_hist",n_jobs=1),
           xgboost.XGBClassifier(n_estimators=2500,eta=0.02,booster='gbtree',max_depth=12,min_child_weight=5,colsample_bytree=0.8,objective='binary:logistic',tree_method="gpu_hist",n_jobs=1),
           xgboost.XGBClassifier(n_estimators=1000,eta=0.02,booster='gbtree',max_depth=14,min_child_weight=500,colsample_bytree=0.9,objective='binary:logistic',tree_method="gpu_hist",n_jobs=1),            
           CatBoostClassifier(iterations=1000, depth=10, objective="Logloss",verbose=1000, task_type="GPU"),
           CatBoostClassifier(iterations=900, depth=10, objective="Logloss",nan_mode="Max",verbose=1000, task_type="GPU"),
           CatBoostClassifier(iterations=900, depth=11, objective="Logloss",min_child_samples=200,verbose=1000, task_type="GPU"),     
           CatBoostClassifier(iterations=600, depth=12, objective="Logloss",verbose=1000, task_type="GPU"),     
           CatBoostClassifier(iterations=800, depth=11, objective="Logloss",min_data_in_leaf=5,verbose=1000, task_type="GPU"),
           CatBoostClassifier(iterations=700, depth=12, objective="Logloss",min_data_in_leaf=250,verbose=1000, task_type="GPU"), 
           CatBoostClassifier(iterations=3000, depth=5, objective="Logloss",min_data_in_leaf=3,verbose=1000, task_type="GPU"),  
           CatBoostClassifier(iterations=2000, depth=6, objective="Logloss",grow_policy='Depthwise',verbose=1000, task_type="GPU"),
           CatBoostClassifier(iterations=3000, depth=9,ignored_features=[171], objective="Logloss",grow_policy='Lossguide',max_leaves=50,verbose=1000, task_type="GPU"),
           CatBoostClassifier(iterations=3000, depth=9,ignored_features=[172], objective="Logloss",grow_policy='Lossguide',max_leaves=80,verbose=1000, task_type="GPU"),
           CatBoostClassifier(iterations=3000, depth=9,ignored_features=[372], objective="Logloss",grow_policy='Lossguide',max_leaves=64,verbose=1000, task_type="GPU"),
           CatBoostClassifier(iterations=3000, depth=9,ignored_features=[370], objective="Logloss",grow_policy='Lossguide',max_leaves=64,verbose=1000, task_type="GPU"),
           CatBoostClassifier(iterations=4000, depth=9,ignored_features=[171,172,372], objective="Logloss",grow_policy='Lossguide',max_leaves=64,verbose=1000, task_type="GPU"),
           CatBoostClassifier(iterations=4000, depth=9,ignored_features=[171,172,370], objective="Logloss",grow_policy='Lossguide',max_leaves=64,verbose=1000, task_type="GPU"), 
           CatBoostClassifier(iterations=3000, depth=9,ignored_features=[2], objective="Logloss",grow_policy='Lossguide',max_leaves=90,verbose=1000, task_type="GPU"), 
           CatBoostClassifier(iterations=3000, depth=9,ignored_features=[41], objective="Logloss",grow_policy='Lossguide',max_leaves=75,verbose=1000, task_type="GPU"),  
           CatBoostClassifier(iterations=4500, depth=9,ignored_features=[171,172,370,372], objective="Logloss",grow_policy='Lossguide',max_leaves=99,verbose=1000, task_type="GPU"),
           CatBoostClassifier(iterations=2000, depth=9, objective="Logloss",grow_policy='Lossguide',max_leaves=64,verbose=1000, task_type="GPU"),     
           CatBoostClassifier(iterations=1000, depth=5, objective="Logloss",grow_policy='Depthwise',verbose=1000, task_type="GPU"),
           CatBoostClassifier(iterations=2000, depth=7, objective="Logloss",grow_policy='Lossguide',max_leaves=52,verbose=1000, task_type="GPU"),
        ],
        [
         RFE(LogisticRegressionCV(cv=10,scoring='roc_auc', n_jobs=-1,max_iter=15000),2,step=1,verbose=2),
         LogisticRegressionCV(Cs=300,cv=10,scoring='roc_auc', n_jobs=-1,max_iter=15000,random_state=123),
         SelectFromModel(LogisticRegressionCV(cv=10,scoring='roc_auc', n_jobs=-1,max_iter=15000)),
         KNeighborsClassifier(n_neighbors=100),
         KNeighborsClassifier(n_neighbors=500),
         xgboost.XGBRFClassifier(n_estimators=500,max_depth=10,n_jobs=1),
         ExtraTreesClassifier(n_estimators=1000,n_jobs=5),
         lightgbm.LGBMClassifier(n_estimators=1000, num_leaves=51),
         lightgbm.LGBMClassifier(n_estimators=2000, num_leaves=25),
         CatBoostClassifier(iterations=2000, depth=4, objective="Logloss",verbose=1000),
         CatBoostClassifier(iterations=2000, depth=3, objective="Logloss",verbose=1000),
         CatBoostClassifier(iterations=1250, depth=5,verbose=1000, task_type="GPU"),
        ],
        [
        EFS(LogisticRegressionCV(Cs=100,cv=10,scoring='roc_auc', n_jobs=-1,max_iter=15000,random_state=1234,tol=0.00001), 
           min_features=6,
           max_features=7,
           scoring='roc_auc',
           print_progress=True,
       cv=5)
        ]
    ]


model=StackNetClassifier(models, metric="auc", folds=5,
    restacking=False,use_retraining=False, use_proba=True, 
    random_state=555,n_jobs=1, verbose=2)


model.fit(train.drop(TARGET_COL,axis=1),train[TARGET_COL])

test.shape

y_pred = model.predict_proba(test[list(train.drop(TARGET_COL,axis=1).columns)].values)



sample_submission = pd.read_csv('sb_test.csv')[['encounter_id','hospital_death']]

sample_submission[TARGET_COL] = y_pred[:,1]

import pandas as pd
test = pd.read_csv('unlabeled.csv')

test_2 = test[list(train.drop(TARGET_COL,axis=1).columns)].copy()
test_2['age'] =(test['age']/10).round()*10
y_pred_2 = model.predict_proba(test_2)[:,1]

test_3 = test[list(train.drop(TARGET_COL,axis=1).columns)].copy()
test_3['gender'] = test['gender'].apply(lambda x: 'F' if x=='M' else 'M')
y_pred_3 = model.predict_proba(test_3)[:,1]

test_4 = test[list(train.drop(TARGET_COL,axis=1).columns)].copy()
test_4['ethnicity'] = np.random.choice(test['ethnicity'].unique())
y_pred_4 = model.predict_proba(test_4)[:,1]

test_4 = test[list(train.drop(TARGET_COL,axis=1).columns)].copy()
test_4['ethnicity'] = np.random.choice(test['ethnicity'].unique())
y_pred_4 = model.predict_proba(test_4)[:,1]

mean_tta = (y_pred[:,1] + y_pred_2 + y_pred_3 + y_pred_4)/4.
sample_submission[TARGET_COL] = mean_tta
sample_submission.to_csv('StacknetTTA.csv', index=False)