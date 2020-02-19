# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 18:18:38 2019

@author: ali hussain
"""

f=open("D:/mystuff/textcomment.txt")
lines=f.readlines()

ftr=[]
y=[]
for line in lines:
    w=line.strip().lower().split(",")
    l=w[-1]
    inp=" ".join(w[:-1])
    ftr.append(inp)
    if l=="pos":
        y.append(1)
    else:    
        y.append(0)
        
print(ftr)
print(y)        