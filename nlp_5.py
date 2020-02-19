# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 12:53:50 2019

@author: ali hussain
"""


from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
def tfidf(x):
    v=TfidfVectorizer()
    ftr=v.fit_transform(x)
    X=ftr.toarray()
    ones=np.ones(X.shape[0])
    return np.c_[ones,X]


o=np.ones(5)
a=np.array([[1,2],
            [3,5],
            [5,6],
            [9,6],
            [3,6]])
np.c_[o,a]


f=open("D:/mystuff/commentsss.txt")
lines=f.readlines()
text=[]
labels=[]
for line in lines:
    w=line.lower().strip().split(",")
    text.append(w[0])
    labels.append(w[1])
    
text    

labels


# Feature extraction

X=tfidf(text)
print(X)


lbl={}
i=0
for k in set(labels):
    lbl[k]=i
    i+=1
print(lbl)    

nlabels=[lbl[w] for w in labels]
print(nlabels)


binary=[]
for v in nlabels:
    n=len(set(nlabels))
    cells=np.zeros(n)
    cells[v]=1
    binary.append(cells)
    
Y=np.array(binary)
print(Y)    


# 3 hidden layers
np.random.seed(101)
rc=X.shape[1]
cc=int(rc+rc/2)
W1=2*np.random.random((rc,cc))-1
ccc=int(cc+cc/2)
W2=2*np.random.random((cc,ccc))-1
cccc=int(ccc+ccc/2)
W3=2*np.random.random((ccc,cccc))-1
W4=2*np.random.random((cccc,Y.shape[1]))-1

print(W1.shape)
print(W2.shape)
print(W3.shape)
print(W4.shape)


def sigmoid(x):
    return 1/(1+np.exp(-x))
def loss(y,ycap):
    return ((y-ycap)**2).mean()
def derivative(x):
    return x*(1-x)
ploss=0
flag=0
conv=0.000000001
for i in range(100000):
    l1=sigmoid(X.dot(W1))
    l2=sigmoid(l1.dot(W2))
    l3=sigmoid(l2.dot(W3))
    l4=sigmoid(l3.dot(W4))
    e4=Y-l4
    closs=loss(Y,l4)
    diff=abs(ploss-closs)
    if diff<=conv:
        print("Training completed ",i+1)
        flag=1
        break
    if i%1000==0:
        print(" Current loss ",closs)
        
    delta4=e4*derivative(l4)
    e3=delta4.dot(W4.T)
    delta3=e3*derivative(l3)
    e2=delta3.dot(W3.T)
    delta2=e2*derivative(l2)
    e1=delta2.dot(W2.T)
    delta1=e1*derivative(l1)
    W1 +=X.T.dot(delta1)
    W2 +=l1.T.dot(delta2)
    W3 +=l2.T.dot(delta3)
    W4 +=l3.T.dot(delta4)
    ploss=closs
if flag==0:
    print(" Retrain the model with more interation")
    
    
    
def predict(x,w):
    r=x
    for v in w:
        r=sigmoid(r.dot(v))
    return r
    
Ycap=predict(X,[W1,W2,W3,W4])
Ycap[Ycap>0.5]=1
Ycap[Ycap<0.5]=0
YC=[int(np.where(v==1)[0]) for v in Ycap]
def accuracy(y,ycap):
    r=y==ycap
    return r[r==True].size/r.size*100

accuracy(np.array(nlabels),YC)



############################################################################### 31/10/19

f=open("D:/mystuff/commentss.txt")
lines=f.readlines()
print(lines)

x=[]
y=[]

for line in lines:
    w=line.strip().lower().split(",")
    x.append(w[0])
    l=w[1]
    if l=="pos":
        y.append(1)
    else:
        y.append(0)
        
print(x)

print(y)
"""
from sklearn.feature_extraction.text import TfidfVectorizer
def tfidf(x):
    v=TfidfVectorizer()
    return v.fit_transform(x).toarray()
"""
X=tfidf(x)
print(X)

X.shape

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
model=GaussianNB()
model.fit(X,y)
ycap=model.predict(X)
print(ycap)
accuracy_score(y,ycap)




from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
model=DecisionTreeClassifier()
model.fit(X,y)
ycap=model.predict(X)
print(ycap)
accuracy_score(y,ycap)



from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
model=RandomForestClassifier()
model.fit(X,y)
ycap=model.predict(X)
print(ycap)
accuracy_score(y,ycap)




from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
model=SVC()
model.fit(X,y)
ycap=model.predict(X)
print(ycap)
accuracy_score(y,ycap)


################################################################################# 05/11/19

from sklearn.cluster import KMeans
model=KMeans(n_clusters=2,random_state=0)
import numpy as np
X=np.array([[1,2],[1,4],[1,0],[10,2],[10,4],[10,0]])
print(X)
X.shape
model.fit(X)

model.cluster_centers_


def dist(a,b):
    return ((a-b)**2).sum()**0.5


centr=model.cluster_centers_
grps=[]
for c in X:
    if dist(c,centr[0]<dist(c,centr[1])):
        grps.append((c,"Grp1"))
    else:
        grps.append((c,"Grp2"))

for g in grps:
    print(g)


################################################################################### 09/11/19

lines=""" hello all how are you ? 
                 how is prepration?
                 I hope you all doing well . practice well without compromize !"""
lines.split("?")             


from nltk.tokenize import sent_tokenize
sents=sent_tokenize(lines)
print(sents)


line="hello how are you?hope fine . "
line.split()

from nltk.tokenize import word_tokenize
words=word_tokenize(line)
print(words)


from nltk import stem
s=stem.PorterStemmer()
s.stem("cooking")






    
    




























