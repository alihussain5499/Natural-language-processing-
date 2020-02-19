# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 18:41:54 2019

@author: ali hussain
"""

f=open("D:/mystuff/commentss.txt")
lines=f.readlines()
print(lines)
text=[]
y=[]
for line in lines:
    w=line.strip().lower().split(",")
    text.append(w[0])
    l=0
    if w[1]=="pos":
        l=1
    y.append(l)
    
print(text)
print(y)

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
def tfidf(x):
    v=TfidfVectorizer()
    ftr=v.fit_transform(x)
    print(v.get_feature_names())
    return ftr.toarray()

x=tfidf(text)
print(x)

print(x.shape)
    
ones=np.ones(10)
ones

ones=np.ones((7,1))
ones

ones=np.ones((x.shape[0],1))
X=np.concatenate((ones,x),axis=1)
X.shape

y

Y=np.c_[y]
print(Y)

def logistic(x):
    # x is dot product of inputs and weight
    return 1/(1+np.exp(-x))

def weight(x,y):
    from numpy.linalg import inv
    left=inv(x.T.dot(x))
    rgt=x.T.dot(y)
    return left.dot(rgt)

w=weight(X,Y)

print(w)

w.shape

yc=logistic(X.dot(w))
yc

yc[yc>0.5]=1
yc[yc<0.5]=0
print(yc)

def accuracy(y,ycap):
    r=y==ycap
    pcnt=r[r==True].size
    n=y.size
    return pcnt/n*100

accuracy(Y,yc)

def loss(y,ycap):
    return((y-ycap**2).mean())

loss(Y,yc)

def derivative(ycap):
    return ycap*(1-ycap)

derivative(yc)

#train
np.random.seed(100)
W=2*np.random.random((8,1))-1
W

ploss=0
flag=0
for i in range(100000):
    ycap=logistic(X.dot(W))
    closs=loss(Y,ycap)
    e=Y-ycap
    if abs(ploss-closs)<=0.00000001:
        print("Training Completed ",i+1,"Iteration")
        flag=1
        break
    if i%200==0:
        print("Current loss ",closs)
    delta=e*derivative(ycap)
    W+=X.T.dot(delta)
    ploss=closs
    
derivative(W)    
        
print(w.ravel())

print(W.ravel())

ycap=logistic(X.dot(W))
np.c_[Y,ycap]

ycap=logistic(X.dot(W))
ycap[ycap<0.5]=0
ycap[ycap>0.5]=1
accuracy(Y,ycap)

############################################################################### 02/10/19
"""
Build network
X---->i/p
Y---->label
W1--->8x12
W2--->12x1

"""
np.random.seed(100)
W1=2*np.random.random((8,12))-1
W2=2*np.random.random((12,1))-1
print(W1)
print(W2)

def activate(x,w):
    d=x.dot(w)
    return 1/(1+np.exp(-d))

l1=activate(X,W1)
l1.shape

l2=activate(l1,W2)
l2.shape

l1

l2

def loss(y,ycap):
    return ((y-ycap)**2).mean()

loss(Y,l2)

def derivative(ycap):
    return ycap*(1-ycap)

# Testing weather loss is dicreasing or not
    
l1=activate(X,W1)
l2=activate(l1,W2)
loss(Y,l2)

# Backwarding
e2=Y-l2
d2=e2*derivative(l2)
e1=d2.dot(W2.T)
d1=e1*derivative(l1)
W1+=X.T.dot(d1)
W2+=l1.T.dot(d2)


# Train the Network
ploss=0
flag=0
for i in range(100000):
    l1=activate(X,W1)
    l2=activate(l1,W2)
    closs=loss(Y,l2)
    if i%200==0:
        print("Current loss",closs)
    if abs(ploss-closs)<=0.000000001:
        print("Traning is complete After ",i+1,"iteration")
        flag=1
        break
    e2=Y-l2
    d2=e2*derivative(l2)
    e1=d2.dot(W2.T)
    d1=e1*derivative(l1)
    W1+=X.T.dot(d1)
    W2+=l1.T.dot(d2)
    ploss=closs
    
print(W1)    

def pred(X,W):
    r=X
    for v in W:
        r=activate(r,v)
    return r

ycap=pred(X,[W1,W2])
print(ycap)

ycap[ycap>0.5]=1
ycap[ycap<0.5]=0
print(ycap)

accuracy(Y,ycap)

############################################################################### 03/10/19

ones=np.ones((X.shape[0],1))
ones

np.concatenate((ones,X),axis=1)
print(X.shape)
print(Y)

# Random Weight
# W1,W2,W3,W4
# Network Building

W1=2*np.random.random((8,12))-1
W2=2*np.random.random((12,18))-1
W3=2*np.random.random((18,12))-1
W4=2*np.random.random((12,1))-1

# Traning

def sigmoid(x):
    return 1/(1+np.exp(-x))
def loss(y,ycap):
    return ((y-ycap)**2).mean()
def derivative(x):
    return x*(1-x)
  
conv=0.000000001
ploss=0
flag=0

for i in range(100000):
    l1=sigmoid(X.dot(W1))
    l2=sigmoid(l1.dot(W2))
    l3=sigmoid(l2.dot(W3))
    l4=sigmoid(l3.dot(W4))
    closs=loss(Y,l4)
    if i%2000==0:
        print("Current loss After ",i+1,"iteration",closs)
    diff=abs(ploss-closs)
    if diff<=conv:
        print("Traning completed after",i+1,"iteration",closs)
        flag=1
        break
    
    e4=Y-l4
    d4=e4*derivative(l4)
    e3=d4.dot(W4.T)
    d3=e3*derivative(l3)
    e2=d3.dot(W3.T)
    d2=e2*derivative(l2)
    e1=d2.dot(W2.T)
    d1=e1*derivative(l1)
    W1+=X.T.dot(d1)
    W2+=l1.T.dot(d2)
    W3+=l2.T.dot(d3)
    W4+=l3.T.dot(d4)
    ploss=closs

def predict(X,W):
    r=X
    for w in W:
        r=sigmoid(r.dot(w))
    return r
    
yc=predict(X,[W1,W2,W3,W4])
print(yc)

yc[yc<0.5]=0
yc[yc>0.5]=1
yc

np.c_[Y,yc]


def accuracy(y,ycap):
    r=y==ycap
    return r[r==True].size/y.size*100

accuracy(Y,yc)


























