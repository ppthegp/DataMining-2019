# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 13:110:05 2019

@author: Prashant Pathak
"""

import random as r
import csv

fileName = 'SamleDate.csv'    

a= [chr(x) for x in range(65,91)]

with open (fileName,'w', newline='') as sampleFile:
    writer = csv.writer(sampleFile,quoting = csv.QUOTE_NONNUMERIC)
   # writer.writerow(['attribute1','attribute2','classtype'])
    for i in range(1,1001):
        s = ""        
        csv_row = [s.join(r.sample(a,10)),r.randrange(0,100000,2),i%10] 
        writer.writerow(csv_row)

#shuffle files
datafile = open(fileName,'r')
lines =  datafile.readlines()
datafile.close();

r.shuffle(lines)

datafileshuffle = open(fileName,'w')
datafileshuffle.writelines(lines)
datafileshuffle.close()


