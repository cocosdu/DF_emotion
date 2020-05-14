from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
import xgboost as xgb
import pandas as pd
import numpy as np

import utils
import preprocess

#train_data,train_label,test_data,ids = preprocess.get_data(IsStr=True,IsSeg=True)
train_data,train_label,test_data,ids,_ = utils.getData()
train_data =np.array(train_data)
train_label = np.array(train_label)
test_data = np.array(test_data)
ids = np.array(ids)

lbl_enc = preprocessing.LabelEncoder()
y = lbl_enc.fit_transform(train_label)
print(lbl_enc.classes_) #['-1' '0' '1']

skf = StratifiedKFold(n_splits = 5).split(train_data,y)

tfv =TfidfVectorizer()

test_res =[]

for i,(train_index,valid_index) in enumerate(skf):
	print(i+1)
	xtrain,ytrain = train_data[train_index],y[train_index]
	xvalid,yvalid = train_data[valid_index],y[valid_index]

	tfv.fit(xtrain)
	xtrain_tfv = tfv.transform(xtrain)
	xvalid_tfv = tfv.transform(xvalid)
	xtest_tfv = tfv.transform(test_data)

	clf = LogisticRegression(penalty='l2',C=10.0)
	clf.fit(xtrain_tfv, ytrain)
	print(clf.score(xtrain_tfv,ytrain),clf.score(xvalid_tfv,yvalid))
	valid_pred = clf.predict(xvalid_tfv)

	test_res.append(clf.predict_proba(xtest_tfv))
	P,F1 = utils.getF1(valid_pred,yvalid)
	print(P,F1)

aver = np.average(test_res,axis=0)
aver = np.argmax(aver,axis=1)

utils.saveResult(ids,aver,'LR_Result')



# xtrain,xvalid,ytrain,yvalid = train_test_split(train_data,y,stratify=y, random_state=42,test_size=0.1,shuffle=True)
#print(len(xtrain)) #89921
#print(len(xvalid)) #9992

'''
tfv = TfidfVectorizer()
tfv.fit(xtrain) #+xvalid
xtrain_tfv = tfv.transform(xtrain)
xvalid_tfv = tfv.transform(xvalid)
#tfv.get_feature_names()
print(xtrain_tfv.shape)
xtest_tfv = tfv.transform(test_data)
'''

'''
ctv = CountVectorizer()
ctv.fit(xtrain)
xtrain_ctv = ctv.transform(xtrain)
xvalid_ctv = ctv.transform(xvalid)
print(xtrain_ctv.shape)
xtest_ctv = ctv.transform(test_data)
'''
'''

"LR "
clf = LogisticRegression(penalty='l2',C=10.0)
clf.fit(xtrain_tfv, ytrain)
print(clf.score(xtrain_tfv,ytrain),clf.score(xvalid_tfv,yvalid))
valid_pred = clf.predict(xvalid_tfv)
test_pred = clf.predict(xtest_tfv)
F1 = utils.getF1(valid_pred,yvalid)
print(F1)
utils.saveResult(ids,test_pred,'LR_Result_%0.4f'%(F1))
'''

''' # LR ctv
clf = LogisticRegression(C=1.0)
clf.fit(xtrain_ctv, ytrain)
print(score(xtrain_ctv,ytrain),score(xvalid_ctv,yvalid))
valid_pred = clf.predict(xvalid_ctv)
test_pred = clf.predict(xtest_ctv)
F1 = utils.getF1(valid_pred,yvalid)
print(F1)
utils.saveResult(ids,test_pred,'LR_Result_ctv_%0.4f'%(F1))
'''

'''
clf = MultinomialNB()
clf.fit(xtrain_tfv,ytrain)
valid_pred = clf.predict(xvalid_tfv)
test_pred = clf.predict(xtest_tfv)
F1 = utils.getF1(valid_pred,yvalid)
print(F1)
utils.saveResult(ids,test_pred,'NB_Result_%0.4f'%(F1))
''' #0.5012

''' #SVM
svd = TruncatedSVD(n_components=120)
svd.fit(xtrain_tfv)
xtrain_svd = svd.transform(xtrain_tfv)
xvalid_svd = svd.transform(xvalid_tfv)
xtest_svd = svd.transform(xtest_tfv)

scl = preprocessing.StandardScaler()
scl.fit(xtrain_svd)
xtrain_svd_scl = scl.transform(xtrain_svd)
xvalid_svd_scl = scl.transform(xvalid_svd)
xtest_svd_scl = scl.transform(xtest_svd)

clf = SVC(C=1.0) # since we need probabilities
clf.fit(xtrain_svd_scl, ytrain)
valid_pred = clf.predict(xvalid_svd_scl)
test_pred = clf.predict(xtest_svd_scl)
F1 = utils.getF1(valid_pred,yvalid)
print(F1)
utils.saveResult(ids,test_pred,'SVM_Result_%0.4f'%(F1))
'''#0.5992

'''
clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 
                        subsample=0.8, nthread=10, learning_rate=0.1)
clf.fit(xtrain_tfv.tocsc(), ytrain)
valid_pred = clf.predict(xvalid_tfv.tocsc())
test_pred = clf.predict(xtest_tfv.tocsc())

F1 = utils.getF1(valid_pred,yvalid)
print(F1)
utils.saveResult(ids,test_pred,'xgb_Result_%0.4f'%(F1))
''' #0.5981