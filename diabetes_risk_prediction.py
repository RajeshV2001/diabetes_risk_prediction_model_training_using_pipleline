import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

data=pd.read_csv("diabetes_risk_prediction_dataset.csv")
df=pd.DataFrame(data)
col=df.columns
encd=OrdinalEncoder()
ct=ColumnTransformer([("label",encd,col[1:])])
clmn=ct.fit_transform(df)

df.iloc[:,1:]=clmn
X=df.drop('class',axis=1)
y=df['class'].astype('int')

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42)

from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier,RandomForestClassifier
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


mnb=Pipeline([('MultiNominal',MultinomialNB())])
bg=Pipeline([('BaggingClf',BaggingClassifier(n_estimators=14))])
rf=Pipeline([('RandomForest',RandomForestClassifier())])
dt=Pipeline([('DecisionTree',DecisionTreeClassifier())])
lr=Pipeline([('Logistic',LogisticRegression(max_iter=200))])
svc=Pipeline([('SVC',SVC(max_iter=3))])
knn=Pipeline([('Kneighbor',KNeighborsClassifier())])

mypipes=[mnb,bg,rf,dt,lr,svc,knn]
for model in mypipes:
    model.fit(X_train,y_train)
    
names={0:'MultiNominal',1:'BaggingClf',2:'RandomForest',3:'DecionsionTree',4:'LogisticReg',5:'SVC',6:'KneighborClf'}

print('*** Train Score ***\n')

train_score=[]

for i,model in enumerate(mypipes):
    train_score.append(model.score(X_train,y_train))
    print(f'{names[i]} : {model.score(X_train,y_train)}')

print('\n\n***Test Score ***\n')
pred=0
test_score=[]
for i,model in enumerate(mypipes):
    pred=model.predict(X_test)
    test_score.append(metrics.accuracy_score(y_test,pred))
    print(f'{names[i]} : {metrics.accuracy_score(y_test,pred)}')



plt.plot(names.values(),train_score,'o:r',label="Train_Score")
plt.plot(test_score,'*-b',label="Test_Score")
plt.xticks(rotation=45)
plt.legend()














