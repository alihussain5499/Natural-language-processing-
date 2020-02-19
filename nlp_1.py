# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 14:47:42 2019

@author: ali hussain
"""

from sklearn.feature_extraction.text import TfidfVectorizer
corpus=['This is the first document.',
        'This document is the second document .',
        'And this is the third one .',
        'Is this the first document ?']
v=TfidfVectorizer()
X=v.fit_transform(corpus)

print(X)

print(X.shape)

v.get_feature_names()

X.toarray()
