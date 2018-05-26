import numpy as np
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier,GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
#from sklearn.neural_network import MLPClassifier

#import warnings
#warnings.filterwarnings('ignore')

model=LogisticRegression(penalty='l2',C=5,n_jobs=-1);

def datainput():
	train = pd.read_csv('train.csv')
	test = pd.read_csv('test.csv')
	
	#deal with missing data
	df = pd.concat([train['comment_text'], test['comment_text']], keys=['train','test'])
	df = df.fillna("unknown")
	train_feature=df.loc['train'];
	test_feature=df.loc['test'];

	#feature extraction
	vectorizer = TfidfVectorizer(stop_words='english', max_features=20000)
	vectorizer.fit(train_feature)
	Train_f = vectorizer.fit_transform(train_feature)
	Test_f= vectorizer.fit_transform(test_feature)

	#target label
	#col = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
	col=train.columns[2:]	
	train_labels=train[col]

	return Train_f,train_labels,Test_f

def modeltrain(Train_f,train_labels,Test_f):
	col=train_labels.columns
	#preprocessing
	X_train,X_val,y_train,y_val=train_test_split(Train_f,train_labels,test_size=0.25)

	#testing
	preds_test = np.zeros((Test_f.shape[0], len(col)))

	loss = []

	for index, name in enumerate(col):
	    print(name+'  fitting result:')
	    model.fit(X_train, y_train[name])
	    preds_test[:,index] = model.predict_proba(Test_f)[:,1]
	    pred_train = model.predict_proba(X_val)[:,1]
	    print('log loss:', log_loss(y_val[name], pred_train))
	    loss.append(log_loss(y_val[name], pred_train))
	    
	print('mean column-wise log loss:', np.mean(loss))
	    
	subm = pd.read_csv('sample_submission.csv')
	submid = pd.DataFrame({'id': subm["id"]})
	submission = pd.concat([submid, pd.DataFrame(preds_test, columns = col)], axis=1)
	submission.to_csv('submission.csv', index=False)

if __name__ == '__main__':
	train,train_label,test = datainput()
	modeltrain(train,train_label,test)
