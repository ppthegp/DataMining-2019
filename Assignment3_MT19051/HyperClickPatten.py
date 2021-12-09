import numpy as np
import pandas as pd
import copy as cp
from tqdm import tqdm_notebook
import time 
import matplotlib.pyplot as pt


#Dummy Data to try running code
# Data = [[1],[2],[3,4],[1,2],[1,2],[1,2],[1,2,3,4,5],[1],[2],[3,5]]
# setData = list(map(set, Data))
# dataSize = len(setData)
# ItemSet1 = list(map(frozenset,[[1],[2],[3],[4],[5]]))

def readFile(fileName):
    data = pd.read_csv(fileName,sep = ' ',header=None).iloc[:,:-1]
    colNo = np.max(data.count(axis = 1))
    dataSize = len(data)
    return data,colNo,dataSize

def find1ItemSet(data):
    actualItemList = np.unique(data)
    listofItem = [[i] for i in actualItemList]
    return list(map(frozenset,listofItem))

def calSupportAndPrune(setData,ItemSet,minSupportCount,dataSize):
    ItemSetDic = {}
    for transaction in tqdm_notebook(setData):
        for Item in ItemSet:
            if(Item.issubset(transaction)):
                if Item not in ItemSetDic : 
                    ItemSetDic[Item] = 1
                else:
                    ItemSetDic[Item] += 1 
    ItemSetAndSupport = {}
    for ItemKey in ItemSetDic:
        support = ItemSetDic[ItemKey]/dataSize
        if(support >= minSupportCount):
            ItemSetAndSupport[ItemKey] = support    
    ItemSetAndSupport = dict(sorted(ItemSetAndSupport.items(),key = lambda x: x[1],reverse = True))
    FItemSet = list(ItemSetAndSupport.keys())
    return FItemSet,ItemSetAndSupport

def calHconfAndprune(ItemSet,supportDataOf1,supportData_k,hc):
    itemSetWithLessHconf = []
    for i in ItemSet:
        #cal the element with maximumHconf
        itemList = [frozenset([j]) for j in list(i)]
        maxSingleHconf = max([supportDataOf1[k] for k in itemList])
        ItemSetHconf = supportData_k[i]/maxSingleHconf
        if(ItemSetHconf < hc):
            itemSetWithLessHconf.append(i)
    fItemSet = [i for i in ItemSet if i not in itemSetWithLessHconf]
    return fItemSet 

def candidateGeneration(ItemSetk_1,SupportItemSet1,hc):
    ItemSetK = []
    for i in range(len(ItemSetk_1)):
        y = ItemSetk_1[i]
        for j in range(i+1,len(ItemSetk_1)):
            x = ItemSetk_1[j]
            #if(SupportItemSet1[x] < SupportItemSet1[y] * hc):
            #    break;
            L1 = list(y);
            L2 = list(x);
            if L1[:-1] == L2[:-1]:
                ItemSetK.append(x | y)
    return ItemSetK;

def candidateGenerationCrossSupport(ItemSetk_1,SupportItemSet1,hc):
    ItemSetK = []
    for i in range(len(ItemSetk_1)):
        y = ItemSetk_1[i]
        for j in range(i+1,len(ItemSetk_1)):
            x = ItemSetk_1[j]
            if(SupportItemSet1[x] < SupportItemSet1[y] * hc):
                break;
            L1 = list(y);
            L2 = list(x);
            if L1[:-1] == L2[:-1]:
                ItemSetK.append(x | y)
    return ItemSetK;

def plotLossFunction(x , y, z,hc):
    pt.ion()
    fig, (ax1, ax2) = pt.subplots(nrows=1, ncols=2,figsize=(10, 4))
    #axes = pt.gca()
    #axes.set_xlim([xmin,xmax])
    #axes.set_ylim([0,10000])
    ax1.plot(x,y[hc[0]],label = 'hconf'+ str(hc[0]))
    ax1.plot(x,y[hc[1]],label = 'hconf'+ str(hc[1]))
    ax1.set_title('Confidence-Pruning Effect')
    ax1.legend()
    ax1.grid()
    ax1.set_xlabel('Minimum Support Thresholds')
    ax1.set_ylabel('Number of Hyperclique Patterns')
    
    #axes.set_ylim([0,max(z[hc[1]])])
    ax2.plot(x,z[hc[0]], label = 'hconf = '+ str(hc[0]))
    ax2.plot(x,z[hc[1]], label = 'hconf = '+ str(hc[1]))
    ax2.legend()
    ax2.grid()
    ax2.set_title('Time Taken')
    ax2.set_xlabel('Minimum Support Thresholds')
    ax2.set_ylabel('Execution Time (sec)')
    ax2.yaxis.tick_right()
    pt.show()



t1= time.time()
fileName = 'pumsb.dat'
dataMain,colNo,dataSize = readFile(fileName)
#setData = list(map(set, data.to_numpy()))
#ItemSet1 = find1ItemSet(data)
#print("Length of 1 itemSet",len(ItemSet1))
print(time.time() - t1)

samplesize = [.2,.4,.6,.8,1]
dictTime = {}

for sample in samplesize:
    time1 = time.time();
    data = dataMain.sample(frac=sample, replace=True, random_state=1)
    dataSize = len(data)
    setData = list(map(set, data.to_numpy()))
    ItemSet1 = find1ItemSet(data)
    print("Length of 1 itemSet",len(ItemSet1))
    minSupportList = [0.5]
    hcList = [.95]
    timeExecuted ={}
    patternfound ={}
    for hc in hcList:
        tmexe = []
        patternfnx = []
        for minSupport in minSupportList: 
            t1= time.time()
            FItemSet1 , SupportItemSet1 = calSupportAndPrune(setData,ItemSet1,minSupport,dataSize)
            print (len(FItemSet1))
            #print(time.time() - t1)
            #t1= time.time()
            ItemSetk = cp.deepcopy(FItemSet1)
            totalItemCount = len(ItemSetk)
            SupportItemSetk = cp.deepcopy(SupportItemSet1)
            #print("1 ItemSet are:\n",ItemSetk)
            for i in tqdm_notebook(range(2,len(ItemSetk))):
                CItemSetk = candidateGenerationCrossSupport(ItemSetk , SupportItemSetk, minSupport)
                ItemSetk , SupportItemSetk = calSupportAndPrune(setData,CItemSetk,minSupport,dataSize)
                ItemSetk = calHconfAndprune(ItemSetk,SupportItemSet1,SupportItemSetk,hc)
                if(len(ItemSetk) ==0 ):
                    break;
                print("Length of ",i ," ItemSet is:\n",len(ItemSetk))
                totalItemCount += len(ItemSetk)
            #print(time.time() - t1)
            #print ("Total number of item generated are",totalItemCount)
            tmexe.append(time.time() - t1);
            patternfnx.append(totalItemCount) 
        timeExecuted[hc] = tmexe
        patternfound[hc] = patternfnx
    dictTime[dataSize] = time.time() - time1
        


# # Output

for hc in hcList:
    print("For Hconficence value  of ", hc)
    print ("No of Pattern \t Execution time")
    for i in range(len(minSupportList)):
        print (patternfound[hc][i] ,"\t\t", timeExecuted[hc][i])
    print ("\n")



plotLossFunction(minSupportList,patternfound,timeExecuted,hcList)

# To Plot graph the bwtween the time taken and the size of the attribute 
axes = pt.gca()
pt.plot(samplesize,list(dictTime.values()),label = 'hconf: .95 and minSupp: .5')
pt.legend()
pt.grid(True)
pt.xlabel('Transaction Fraction')
pt.ylabel('Execution Time')



