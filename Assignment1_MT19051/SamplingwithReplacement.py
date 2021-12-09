# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 22:34:27 2019

@author: Prashant Pathak
"""
import random as r
import csv
import matplotlib.pyplot as plot

fileName = 'SamleDate.csv'
data = []
k = 4

def allClassesIncluded(sample,k):
    classes = [0] * 10  
    for i in sample:
        #print (i)
        classes[int(i[2])] = k if classes[int(i[2])] == k else classes[int(i[2])] + 1 
        if sum(classes) == k*10:
            return True;
    
    return False;
        

with open(fileName, 'r') as csvFile:
    reader = csv.reader(csvFile,quoting = csv.QUOTE_NONNUMERIC)
    data = list(reader)

sampleSizeForK = []
for m in range(1,k+1):
    samplesizes = []
    for i in range (1001):
        samplepoint = 10
        while 1:
            #Use random.choices function when you want to choose multiple items 
            #out of a list including repeated.
            sample = r.choices(data,k=samplepoint)
            #print (sample)
            if (allClassesIncluded(sample,m)):
                break
            samplepoint += 1    
        samplesizes.append(samplepoint);
    sampleSizeForK.append(samplesizes)

#print(sampleSizeForK) 

probForK = []
xforK = []
for k_sample in sampleSizeForK:     
    prob = []
    x = []
    xi = 0;
    while 1:
        count = len(list(y for y in k_sample if y < xi))
        probi = count / 1000
        #print (prob)
        x.append(xi)
        prob.append(probi)
        if probi >= 1:
            break   
        xi += 1;
    #print(x)    
    #print (prob) 
    probForK.append(prob);
    xforK.append(x);


for i in range(k):
    #print(xforK[i],probForK[i])
    label = "k ="+ str(i+1)
    plot.plot(xforK[i],probForK[i],label =label)   
    
plot.legend(loc = 'upper left')
plot.xlabel("No of samples")
plot.ylabel("Probability")
plot.show()

# NOW for frequecy
for k_sample in sampleSizeForK:
     barrange = 0
     count = []
     tick_label = []
     barx =[]
     while sum(count) < 1000:
         count.append(len(list( x for x in k_sample if barrange <= x < barrange+10 )))
         barx.append(barrange)
         tick_label.append(str(barrange)+ "-" + str(barrange+9))
         barrange +=10
     var = len(count)
     print('var',var)
     print(barx)
     print(tick_label)
     print(count)
     plot.bar(barx,count,tick_label = tick_label ,width = 8)
     plot.xlabel("Class group")
     plot.ylabel("Frequency")
     plot.show()     
         