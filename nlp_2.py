# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 11:28:24 2019

@author: ali hussain
"""

f=open("D:/mystuff/comments.txt")
lines=f.readlines()
print(lines)
print(len(lines))

lines=[line.strip().lower() for line in lines]
print(lines)

def tfidf(x):
    from sklearn.feature_extraction.text import TfidfVectorizer
    v=TfidfVectorizer()
    ftr=v.fit_transform(x)
    arr=ftr.toarray()
    return arr

X=tfidf(lines)

print(X)

print(lines[0].split(","))

text=[]
y=[]
for line in lines:
    w=line.split(",")
    inp=w[0]
    l=w[1]
    if l=="pos":
        y.append(1)
    else:
        y.append(0)
    text.append(inp)
print(text)

print(y)

X=tfidf(text)

import numpy as np
Y=np.c_[y]
print(Y)























