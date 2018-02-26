#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Naive Bayes algorithm for Sentimental Analysis
Created on Sun Feb 18 18:44:17 2018
@author: sahilbehl
references: https://web.stanford.edu/~jurafsky/slp3/6.pdf
"""

import math
import re
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
kf = KFold(n_splits=10)

#cleaning file
file = open("amazon_cells_labelled.txt").read()
lines = file.split("\n")
lines.pop()
classes = []
words = []
documents = []


# Ndoc = number of documents in D
i=0
while i < len(lines):
    documents.append(lines[i].split("\t"))
    i+=1

for line in documents:
    line[0] = re.sub('[-.,!:()?]'," ",line[0])
    line[0] = line[0].split(" ")
    

def naive_bayes_trainer(documents):
    N_doc = len(documents)
    classes = []
    for line in documents:
        classes.append(line[1])
    #Nc = number of documents from D in class c    
    N_c = [0,0]
    for c in classes:   
        if(c == '0'):
            N_c[0] +=1 
        else:
            N_c[1] +=1
    #logprior[c]← log Nc/Ndoc        
    logprior = []
    for c in N_c:
        logprior.append(math.log10(float(c)/N_doc))
    All_words = []
    for line in documents:
        for word in line[0]:
            if word == " ":
                continue
            elif word == "":
                continue
            All_words.append(word)
    #V←vocabulary of D
    V = []              
    for word in All_words:
        if V.count(word.lower()) == 0:
            V.append(word.lower())
        else:
            continue
    #bigdoc[c]←append(d) for d ∈ D with class c
    bigdoc=[[],[]]
    for line in documents:
        if line[1] == '0':    
            for word in line[0]:
                if word == " ":
                    continue
                elif word == "":
                    continue
                bigdoc[0].append(word)
        else:
            for word in line[0]:
                if word == " ":
                    continue
                elif word == "":
                    continue
                bigdoc[1].append(word)
    loglikelihood = [[],[]]
     #for each word w in V 
    for word in V:    
         loglikelihood[0].append(math.log10(float( bigdoc[0].count(word)+1 )/(len(bigdoc[0])+len(V)) ) )
         loglikelihood[1].append(math.log10(float( bigdoc[1].count(word)+1 )/(len(bigdoc[1])+len(V)) ) )
    return logprior,loglikelihood,V


def naive_bayes_tester(testing_data,logprior, loglikelihood,V):
    result = []
    words = []
    for line in testing_data:    
        for word in line[0]:
            if V.count(word.lower()) > 0:
                words.append(word)
            else:
                continue
        #sum[c]← logprior[c]
        sum = [logprior[0],logprior[1]]
        for word in words:
            index = V.index(word.lower())
            sum[0] = sum[0] + loglikelihood[0][index]
            sum[1] = sum[1] + loglikelihood[1][index]
        if sum[0] >= sum[1]:
            result.append('0')
        else:
            result.append('1')
    return result


avg = 0;
for train, test in kf.split(documents):
    result = []
    true = []
    train = train.tolist()
    test = test.tolist()
    training_data = [documents[i] for i in train]
    test_data = [documents[i] for i in test]
    for row in test_data:
        true.append(row[1])
    result.append(true)
    logprior, loglikelihood,V =  naive_bayes_trainer(training_data)
    pred = naive_bayes_tester(test_data,logprior, loglikelihood,V)
    result.append(pred)
    avg += accuracy_score(true,pred)
avg = avg/10
print "Accuracy of classical naive bayes algorithm: ", avg
          


  




     