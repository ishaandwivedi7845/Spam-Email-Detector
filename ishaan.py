
import pandas as pd
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from  sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score

a=pd.read_csv("spam.csv",encoding='latin1')
print(a.dropna().head())
b=a[["v1","v2"]]
b.columns=["Label","message"]
c=b["message"]
d=b["Label"]
tfid=TfidfVectorizer()
X=tfid.fit_transform(c)
le=LabelEncoder()
y=le.fit_transform(d)
x1,x2,y1,y2=train_test_split(X,y,test_size=0.2)
e=LogisticRegression(max_iter=2000)
f=LinearSVC()
g=RandomForestClassifier()

vote=VotingClassifier(estimators=[
     ("n",e),
     ("m",f),
     ("o",g)
],
voting="hard"


   )
vote.fit(x1,y1)
rr=vote.predict(x2)
print("Accuracy:",accuracy_score(y2,rr))

print("confusion matrix=",confusion_matrix(y2,rr))
