#!/usr/bin/env python
# coding: utf-8

import tweepy
import tqdm
import csv
import json
import time
from tqdm import tqdm_notebook as tqdm

def makeAuthConnection():
    consumerApiKey = 'XXXXXXX'
    consumerApiSecret = 'XXXXXXX'
    acessToken = 'XXXXXX'
    acessTokenSecret  = 'XXXXXX'
    
    auth = tweepy.OAuthHandler(consumerApiKey, consumerApiSecret)
    #auth = tweepy.AppAuthHandler(consumerApiKey, consumerApiSecret)
    auth.set_access_token(acessToken, acessTokenSecret)
    
    return tweepy.API(auth , wait_on_rate_limit = True,wait_on_rate_limit_notify = True)


# In[3]:


api = makeAuthConnection()
# for status in tweepy.Cursor(api.search, q='tweepy').items(10):
#     print(status.text)


# In[4]:


def checkRemainingSearchCount():
    jsonString = api.rate_limit_status()['resources']['search']['/search/tweets']
    upperLimit = jsonString['limit']
    remiaingFetch = jsonString['remaining']
    #resetTime = jsonString['reset']/60000 
    print (jsonString)
    return upperLimit,remiaingFetch


# In[5]:


checkRemainingSearchCount()

# This method will generate a file containng the tweets of the data 
# This uses the tweepy API to fetch the data
# TODO This method generate the maxind tweets twice. Will have to check on it.
def searchTweetsByHashtag(searchlist):
    # use this filter to filter the tweets based on the key words -filter:retweets AND -filter:replies
    searchFilter = ' AND -filter:links and -filter:videos and -filter:retweets'
    fileName = 'tweetDataset.csv'
    with open (fileName,'a', newline='',encoding='utf-8') as sampleFile:
        writer = csv.writer(sampleFile,quoting = csv.QUOTE_NONNUMERIC)
        try:
            for searchString in searchlist: 
                search_result = api.search(q=searchString + searchFilter,count=1,lang="en",tweet_mode='extended'
                                           , result_type  = 'recent')
                if(len(search_result) == 0):
                    print("*************No data on "+ searchString +" hashtag.***************")
                else : 
                    max_id = search_result[0].id
                    #print("max_id",max_id)
                    old_id = -1
                    i = 1
                    while(max_id != old_id):
                        old_id = max_id
                        tweetDic = tweepy.Cursor(api.search,q = searchString + searchFilter  ,lang  = 'en'
                                                 ,include_entities=False,tweet_mode='extended',count = 100
                                                 ,max_id = max_id).items(300)
                        print("loop count",i)
                        for tweets in tweetDic:
                            jsonString = tweets._json
                            #print(jsonString['id'],jsonString['full_text'].replace('\n', ' '))
                            csv_row = [jsonString['id'],jsonString['user']['screen_name'],jsonString['retweet_count']
                                       ,jsonString['full_text'].replace('\n', ' ')]  
                            # we can also encode the text here to remove emojies from the text.
                            max_id = jsonString['id'] + 1
                            writer.writerow(csv_row)
                        print("Going to sleep to keep limit to check")    
                        time.sleep(3)
                        print("Waking Up")
                print("*************No more data to exact.*************")
        except tweepy.TweepError as e:
            print("Some error!!:"+str(e))


# In[8]:


search_criteria = ['#MotichoorChaknachoorReview','#jhalkireview','#FordVsFerrari','#MotherlessBrooklyn'
                   ,'#Charlie\'sAngels','#DoctorSleepReview','#MidwayMovie','#Actionreview','#SangathamizhanReview'
                  ,'#JhalleReview']
searchTweetsByHashtag(search_criteria)


# secound File

#!/usr/bin/env python
# coding: utf-8

# In[46]:


import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sea
import copy
import emoji
import time as time
from nltk.tokenize import TweetTokenizer
from nltk.corpus import sentiwordnet as swm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob
from afinn import Afinn
from statistics import mode


# In[47]:


data = pd.read_csv('InitialData.csv',header = None)
data = data.iloc[:,3]


# In[3]:


print(data.shape)


# # Data Preprocessing

# ### Removing handle name and hashtags

# In[4]:


def dataCleaning(data):
    # reger for handle, for RT and for URls
    regexes = ['@[A-Z0-9a-z_:]+','^[RT]+','https?://[A-Za-z0-9./]+','(#\w+)','[!,)(.:*“”""+_’\'?\-]+']
    for regex in regexes:
        data = data.replace(to_replace =regex, value = '', regex = True)
        data = data.str.strip()
        data = data.str.lower()
    return data


# In[5]:


data = dataCleaning(data)


# In[6]:


data.tail(10)


# ### Encode tweets so as to simplify the Emojis

# In[7]:


def encodeString(tweets):
    return tweets.encode('ascii', 'ignore').decode('ascii')


# In[8]:


data = data.apply(emoji.demojize)


# In[9]:


data[25]


# In[10]:


data = data.replace(to_replace ='[_:]+', value = ' ', regex = True)


# In[11]:


data.iloc[25]


# ### Removing dublicate rows

# In[12]:


def removeDublicate(data):
    print(data.shape[0])
    dublicateRows=data.duplicated().tolist()
    if len(dublicateRows) > 0:
        print("Completly Dublicate rows",dublicateRows.count(True))
    dublicateRows=data.iloc[:].duplicated().tolist()
    if len(dublicateRows) > 0:
        print("Dublicate Tweets",dublicateRows.count(True))
    data=data.iloc[:].drop_duplicates()
    return data;


# In[13]:


data = removeDublicate(data)
print(data.shape)


# In[14]:


# Remove word which has length less than 3
data = data.apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))


# In[15]:


data.tail(20)


# ### Tokennization and POS tagging

# In[16]:


def convertToPosTag(tokens):
    tagged_sent = nltk.pos_tag(tokens)
    store_it = [(word, nltk.map_tag('en-ptb', 'universal', tag)) for word, tag in tagged_sent] 
    return store_it


# In[17]:


tt = TweetTokenizer()
tokenizedTweets = data.apply(tt.tokenize)
POStaggedLabel = tokenizedTweets.apply(convertToPosTag)
POStaggedLabel[0]


# In[18]:


POStaggedLabel[25]


# ### Removing STOP word and lemmatizing the tweets

# In[36]:


def ConvertToSimplerPosTag(tag):
    if(tag=='NOUN'):
        tag='n'
    elif(tag=='VERB'):
        tag='v'
    elif(tag=='ADJ'):
        tag='a'
    elif(tag=='ADV'):
        tag = 'r'
    else:
        tag='nothing'
    return tag 


# In[37]:


stop_words = stopwords.words('english')
pstem = PorterStemmer()
lem = WordNetLemmatizer()


# In[38]:


def removeStopWord(row):
    filteredList = [(i,j) for i,j in row if i not in stop_words ]
    return filteredList


# In[39]:


noStopWordList =  POStaggedLabel.apply(removeStopWord)


# In[42]:


def lemmatize(row):
    lemmatizeWord = [lem.lemmatize(w) for w,tag in row] #,pos= ConvertToSimplerPosTag(tag)
    return [pstem.stem(i) for i in lemmatizeWord]


# In[43]:


lemmatizedDF = noStopWordList.apply(lemmatize)


# In[44]:


lemmatizedDF.head()


# # Ground Truth Labling

# In[48]:


modelType = ["Text Blob","SentiWordNet","Afinn",'Combined']
negative = []
neutral = []
positive =[]


# ### Labeling the tweets with TextBlob

# In[49]:


def getLabels(row):
    polarity =  TextBlob(" ".join(row)).sentiment.polarity
    return 1 if polarity > 0 else 0 if polarity == 0 else -1 


# In[50]:


SetimentLabel = tokenizedTweets.apply(getLabels)


# In[51]:


valueCountSentiment = SetimentLabel.value_counts()


# In[52]:


print(valueCountSentiment.sort_index())
count = list(valueCountSentiment.sort_index())


# In[53]:


print(count)
negative.append(count[0])
neutral.append(count[1])
positive.append(count[2])


# ### Labeling the tweets with sentiwordnet

# In[54]:


def ConvertToSimplerPosTag(tag):
    if(tag=='NOUN'):
        tag='n'
    elif(tag=='VERB'):
        tag='v'
    elif(tag=='ADJ'):
        tag='a'
    elif(tag=='ADV'):
        tag = 'r'
    else:
        tag='nothing'
    return tag 


# In[55]:


def getSentimentOfWorld(row):
    positiveScore = []
    negativeScore = []
    for word ,tag in row:
        try:
            tag = ConvertToSimplerPosTag(tag)      
            if(tag!='nothing'):
                concat =  word+'.'+ tag+ '.01'
                positiveScore.append(swm.senti_synset(concat).pos_score())
                negativeScore.append(swm.senti_synset(concat).neg_score())
        except Exception as e:
            #print (e)
            #print("An exception occurred")
            pstem = PorterStemmer()
            lem = WordNetLemmatizer()
            word = lem.lemmatize(word)
            word = pstem.stem(word)
            concat =  word+'.'+ tag+ '.01'
            try:
                positiveScore.append(swm.senti_synset(concat).pos_score())
                negativeScore.append(swm.senti_synset(concat).neg_score())
            except Exception as ex:
                pass
                #print("Nested error.")
            #continue
    postiveScoreTotal = np.sum(positiveScore)
    negativeScoreTotal = np.sum(negativeScore)
    if(postiveScoreTotal > negativeScoreTotal) : 
        return 1
    elif (postiveScoreTotal < negativeScoreTotal) : 
        return -1
    else:
        return 0


# In[56]:


sentiDF  = POStaggedLabel.apply(getSentimentOfWorld)


# In[57]:


count = list(sentiDF.value_counts().sort_index())


# In[58]:


print(count)
negative.append(count[0])
neutral.append(count[1])
positive.append(count[2])


# ### Labeling Tweets with AFINN 

# In[59]:


def getSentimentAfinn(row):
    af = Afinn()
    polarity = af.score(" ".join(row))
    return 1 if polarity > 0 else 0 if polarity == 0 else -1


# In[60]:


AfinnLabel = tokenizedTweets.apply(getSentimentAfinn)


# In[61]:


count=list(AfinnLabel.value_counts().sort_values())
print(count)
negative.append(count[0])
neutral.append(count[1])
positive.append(count[2])


# # Combing the result of All the sentiment analysor above

# In[62]:


def assignLabel(row):
    notAssigned = []
    try:
        return mode(row)
    except Exception as ex: 
        return row[1]


# In[63]:


combineLabel =  pd.concat([SetimentLabel ,sentiDF, AfinnLabel ] , axis = 1,sort=False)
combineLabel.columns = [1,2,3]


# In[64]:


yLabel= combineLabel.apply(assignLabel,axis =1)


# In[65]:


count = list(yLabel.value_counts().sort_values())
negative.append(count[0])
neutral.append(count[1])
positive.append(count[2])


# In[66]:


print(len(yLabel))
print(len(lemmatizedDF))


# In[67]:


def autolabel(ax,rects, xpos='center'):
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0, 'right': 1, 'left': -1}

    for rect in rects:
        height = float("{0:.2f}".format(rect.get_height()))
        height = int(height)
        ax.annotate('{}'.format(height),xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(offset[xpos]*3, 3),  # use 3 points offset
                    textcoords="offset points",  # in both directions
                    ha=ha[xpos], va='bottom')


# In[96]:


def plotComparisionGraph(modelType,negative,neutral,positive,endValue):
    
    print(len(negative))
    ind = np.array([i for i in range(3,endValue,3)])  # the x locations for the groups
    print(ind)
    width = 0.65  # the width of the bars
    
    fig, ax = plt.subplots(figsize = (6,5) )
    rects1 = ax.bar(ind- width , negative, width,label='Accuracy')  #yerr=men_std
    rects2 = ax.bar(ind, neutral, width, label='Precision') #yerr=women_std,
    rects3 = ax.bar(ind+ width, positive, width, label='Recall') #yerr=women_std,
    #rects4 = ax.bar(ind+ (1.5*width), f1ScoreList, width, label='F1-Score') #yerr=women_std,
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Count')
    #ax.set_title('Count comparision between differnet Lexicon Model')
    ax.set_xticks(ind)
    ax.set_xticklabels(modelType)
    ax.legend(loc='upper center', bbox_to_anchor=(0.90, 0.8), ncol=1) #shadow=True
        
    autolabel(ax,rects1, "center")
    autolabel(ax,rects2, "center")
    autolabel(ax,rects3, "center")
    #autolabel(ax,rects4, "center")    
    
    
    #fig.tight_layout()
    plt.show()


# In[97]:


plotComparisionGraph(modelType,negative,neutral,positive,13)


# ### Visualize with the help of WorldCloud

# In[60]:


def plotWorldCould(Flattenlist,label):
    plt.rcParams['figure.figsize']=(10.0,8.0)    
    plt.rcParams['font.size']=10 
    stopwords = set(STOPWORDS)
    text = " ".join(tweet for tweet in [" ".join(i) for i in Flattenlist])
    #print(text)
    print ("There are {} words in the combination of all tweets.".format(len(text)))

    wordcloud = WordCloud(
                              background_color='black',
                              stopwords=stopwords,
                              max_words=250,
                              max_font_size=50,
                              width=500, 
                              height=300,
                              random_state=42
                             ).generate(str(text))

    fig = plt.figure(1)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.title(label)
    plt.show()
#fig.savefig("word1.png", dpi=1400)


# In[61]:


# seperate the positive and negative data
#yLabel = SetimentLabel.to_numpy()


# In[62]:


def visualizedWordCloud(lemmatizedDF,yLabel):
    # Ploting Tweets
    pos = np.where(yLabel == 0)[0]
    print(len(pos))
    neutralTweets = lemmatizedDF.iloc[pos]
    plotWorldCould(neutralTweets,"Neutral")
    
    #Ploting Positive tweets
    pos = np.where(yLabel == 1)[0]
    print(len(pos))
    print(len(lemmatizedDF))
    positiveTweets = lemmatizedDF.iloc[pos]
    plotWorldCould(positiveTweets,"Positive Word")
    
    #Ploting negative 
    pos = np.where(yLabel == -1)[0]
    print(len(pos))
    negativeTweets = lemmatizedDF.iloc[pos]
    plotWorldCould(negativeTweets,"Negative Word")


# In[63]:


visualizedWordCloud(lemmatizedDF,yLabel)


# # Removing Common words from the tweets

# In[64]:


def removeWords(row):
    unwantedWord =['watch','film','movi','review']
    row = [i for i in row if i not in unwantedWord]
    return row


# In[65]:


lemmatizedDF = lemmatizedDF.apply(removeWords)


# In[66]:


#Re-visualized
visualizedWordCloud(lemmatizedDF,yLabel)


# # Saving PrepossedDF to CSV 
#lemmatizedDF
joinedTweet = lemmatizedDF.apply(lambda x: str(" ".join(x)))
data = pd.concat([joinedTweet,yLabel],axis = 1 )
data.columns = ['tweets','label']
data.to_csv('PrepeocessedFile.csv', index=False)


#3rd File

#!/usr/bin/env python
# coding: utf-8

# In[53]:


import math
import pandas as pd
import numpy as np
import time as time
import matplotlib.pyplot as plt 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import operator
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB 
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
from tqdm import tqdm_notebook as tqdm


# In[54]:


def readData(fileName):
    data = pd.read_csv(fileName)
    return data


# In[55]:


data = readData('PrepeocessedFile.csv')


# In[56]:


#data['tweets']= data['tweets'].apply(list)
data['label'].value_counts()


# In[57]:


xData = data.iloc[:,0]
yLabel = data.iloc[:,1]


# # Vectorized data 

# In[6]:


vectorizedType = ['CV_1G','CV_2G','CV_3G','TV_1G','TV_2G','TV_3G']
accuracyList =[]
precisionList =[]
recallList =[]
f1ScoreList = []


# In[7]:


def plotCount(words,wordCount):
    plt.figure(figsize=(8,6))
    plt.bar(words[:10],wordCount[:10])
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Top words - Count Vectorizer')
    plt.show()


# In[8]:


def testVectorizationNaiveBias(vectorisedData,yLabel):
    xTrain, xTest, yTrain, yTest = train_test_split(vectorisedData, yLabel, test_size=0.25, random_state=27)
    #initialize Model
    NaiveModel = GaussianNB()
    NaiveModel.fit(xTrain,yTrain)
    predictedTrain = NaiveModel.predict(xTrain)
    predictedTest = NaiveModel.predict(xTest)
    accuracyTest = accuracy_score(predictedTest,list(yTest))
    precisionTest  = precision_score(predictedTest,list(yTest),average = 'macro')
    recallTest = recall_score(predictedTest,list(yTest),average = 'macro')
    f1Score = f1_score(predictedTest,list(yTest),average = 'macro')
    print("Accuracy on Training",accuracy_score(predictedTrain,list(yTrain)))
    print("Accuracy on Testing Set",accuracyTest)
    print("Precision on Testing Set",precisionTest)
    print("Recall on Testing Set",recallTest)
    print("F1 score on Testing Set",f1Score)
    return accuracyTest,precisionTest,recallTest,f1Score


# ### Vectorized with CountVector

# In[9]:


def countVectorize(xData,ngramRange):
    cv=CountVectorizer(decode_error='ignore',lowercase=True,analyzer = 'word',ngram_range = ngramRange,max_features = 600 )
    x_traincv=cv.fit_transform(xData)
    x_trainCountVector = x_traincv.toarray()
    columnsName = cv.get_feature_names()
    ColwiseSum=x_trainCountVector.sum(axis=0)
    wordCountPair = sorted(zip(columnsName,ColwiseSum),key=lambda pair: pair[1],reverse=True)
    word = [x for x,y in wordCountPair]
    counts = [y for x,y in wordCountPair]
    plotCount(word,counts)
    return x_trainCountVector


# In[10]:


ngramList = [(1,1),(1,2),(1,3)]
for ngramrange in ngramList:
    vectorisedData = countVectorize(xData,ngramrange)
    accuracyTest,precisionTest,recallTest,f1Score = testVectorizationNaiveBias(vectorisedData,yLabel)
    accuracyList.append(accuracyTest)
    precisionList.append(precisionTest)
    recallList.append(recallTest)
    f1ScoreList.append(f1Score)


# ### Vectorized with tfidfVectorized

# In[11]:


def tfidfVectorize(xData,ngramRange):
    cv=TfidfVectorizer(decode_error='ignore',lowercase=True,analyzer = 'word',ngram_range = ngramRange,max_features = 600 )
    x_traincv=cv.fit_transform(xData)
    x_trainCountVector = x_traincv.toarray()
    columnsName = cv.get_feature_names()
    ColwiseSum=x_trainCountVector.sum(axis=0)
    wordCountPair = sorted(zip(columnsName,ColwiseSum),key=lambda pair: pair[1],reverse=True)
    word = [x for x,y in wordCountPair]
    counts = [y for x,y in wordCountPair]
    plotCount(word,counts)
    return x_trainCountVector


# In[12]:


ngramList = [(1,1),(1,2),(1,3)]
for ngramrange in ngramList:
    vectorisedData = tfidfVectorize(xData,ngramrange)
    accuracyTest,precisionTest,recallTest,f1Score = testVectorizationNaiveBias(vectorisedData,yLabel)
    accuracyList.append(accuracyTest)
    precisionList.append(precisionTest)
    recallList.append(recallTest)
    f1ScoreList.append(f1Score)


# In[13]:


def autolabel(ax,rects, xpos='center'):
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0, 'right': 1, 'left': -1}

    for rect in rects:
        height = float("{0:.2f}".format(rect.get_height()))
        
        ax.annotate('{}'.format(height),xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(offset[xpos]*3, 3),  # use 3 points offset
                    textcoords="offset points",  # in both directions
                    ha=ha[xpos], va='bottom')


# In[14]:


def plotComparisionGraph(vectorizedType,accuracyList,precisionList,recallList,f1ScoreList,endValue):
    
    print(accuracyList)
    ind = np.array([i for i in range(3,endValue,3)])  # the x locations for the groups
    print(ind)
    width = 0.55  # the width of the bars
    
    fig, ax = plt.subplots(figsize = (8,6) )
    rects1 = ax.bar(ind- (1.5*width) , accuracyList, width,label='Accuracy')  #yerr=men_std
    rects2 = ax.bar(ind- width/2, precisionList, width, label='Precision') #yerr=women_std,
    rects3 = ax.bar(ind+ width/2, recallList, width, label='Recall') #yerr=women_std,
    rects4 = ax.bar(ind+ (1.5*width), f1ScoreList, width, label='F1-Score') #yerr=women_std,
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Comparision between different metrics')
    ax.set_xticks(ind)
    ax.set_xticklabels(vectorizedType)
    ax.legend(loc='upper center', bbox_to_anchor=(0.9, 0.5), ncol=1) #shadow=True
        
    autolabel(ax,rects1, "center")
    autolabel(ax,rects2, "center")
    autolabel(ax,rects3, "center")
    autolabel(ax,rects4, "center")    
    
    
    fig.tight_layout()
    plt.show()


# In[15]:


plotComparisionGraph(vectorizedType,accuracyList,precisionList,recallList,f1ScoreList,19)


# ### DocToVec vectorization

# In[16]:


tt = TweetTokenizer()
tokenizedData = xData.apply(tt.tokenize)


# In[17]:


def extractVector(model,rows,col):
    vector = np.zeros((rows,col))
    for i in range(rows):
        vector[i] = model.docvecs[i]
    return vector


# In[18]:


def docToVec(vec_type,tokenizedData):
    max_epochs = 10
    vec_size = 200
    alpha = 0.0025
    #tagging the words to give tags 
    taggedData = [TaggedDocument(data, tags=[str(i)]) for i,data in enumerate(tokenizedData)]
    #Using DoctoVec model
    modle = None
    if vec_type == 'DBOW':
        model = Doc2Vec(dm =0,vector_size=vec_size,alpha=alpha,negative  = 5,min_alpha=0.00025,min_count=1,workers = 3)
    elif vec_type == 'DMC':
        model = Doc2Vec(dm =0,dm_concat=1,vector_size=vec_size,alpha=alpha,negative  = 5
                        ,min_alpha=0.00025,min_count=1,workers = 3)
    else:
        model = Doc2Vec(dm=1,dm_mean=1,vector_size=vec_size,alpha=alpha,negative  = 5
                        ,min_alpha=0.00025,min_count=1,workers = 3)
        
    model.build_vocab(taggedData)

    for epoch in tqdm(range(max_epochs)):
        model.train(taggedData,total_examples=model.corpus_count,epochs=model.iter)
        model.alpha -= 0.0002
        model.min_alpha = model.alpha

    #retreve Vectors
    return extractVector(model,len(taggedData),vec_size)


# In[19]:


doc2VecType = ['DBOW','DMC','DMM']
daccuracyList =[]
dprecisionList =[]
drecallList =[]
df1ScoreList = []
for i in range(3):
    vectorizedData = docToVec(doc2VecType[2],tokenizedData)
    accuracy,Precison,Recall,f1 = testVectorizationNaiveBias(vectorisedData,yLabel)
    daccuracyList.append(accuracyTest)
    dprecisionList.append(precisionTest)
    drecallList.append(recallTest)
    df1ScoreList.append(f1Score)


# In[20]:


plotComparisionGraph(doc2VecType,daccuracyList,dprecisionList,drecallList,df1ScoreList,10)


# ### Finally taking TFIDF with 1-Gram 

# In[58]:


vectorisedData = tfidfVectorize(xData,(1,2))
vectorisedData = pd.DataFrame(vectorisedData)


# # Dealing with unbalances dataset

# ### Note Plot the graph to show that there is a umbalances dataset

# In[59]:


X_train, X_test, y_train, y_test = train_test_split(vectorisedData, yLabel,
                                                    test_size=0.25,stratify=yLabel ,random_state=27)


# In[62]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[24]:


def HandleUnbalancedDataSet(X_train, y_train,samplesize):
    X = pd.concat([X_train, y_train], axis=1)
    xPos = X[X.label == 1]
    xNeg = X[X.label == -1]
    xNeu = X[X.label == 0]
    xPos_sampled = resample(xPos,replace=False, n_samples=samplesize,random_state=25)
    xNeg_sampled = resample(xNeg,replace=True, n_samples=samplesize,random_state=25)
    xNeu_sampled = resample(xNeu,replace=True, n_samples=samplesize,random_state=25)
    resampledData = pd.concat([xPos_sampled,xNeg_sampled,xNeu_sampled])
    print(resampledData['label'].value_counts())
    print(xPos_sampled.shape)
    xData = resampledData.iloc[:,:-1]
    yLabel = resampledData.iloc[:,-1]
    return xData,yLabel


# In[25]:


samplesize = 600
#y_train.to_numpy().reshape(length,1)

xData,yLabel = HandleUnbalancedDataSet(pd.DataFrame(X_train), pd.DataFrame(y_train),samplesize)


# In[63]:


xData =X_train 
yLabel =y_train 


# In[64]:


print(xData.shape)
print(yLabel.shape)


# # Classification

# ### Naive Bayes

# In[65]:


def evaluationMetric(model, xData,yData):
    predictedTest = model.predict(xData)
    accuracyTest = accuracy_score(predictedTest,list(yData))
    precisionTest  = precision_score(predictedTest,list(yData),average = 'macro')
    recallTest = recall_score(predictedTest,list(yData),average = 'macro')
    f1Score = f1_score(predictedTest,list(yData),average = 'macro')
    print("Accuracy on Testing Set",accuracyTest)
    print("Precision on Testing Set",precisionTest)
    print("Recall on Testing Set",recallTest)
    print("F1 score on Testing Set",f1Score)
    return accuracyTest,precisionTest,recallTest,f1Score


# In[66]:


NaiveModel = GaussianNB()
NaiveModel.fit(xData,yLabel)


# In[67]:


evaluationMetric(NaiveModel,X_test,y_test)


# ### MultinomialNB 

# In[68]:


MultiModel = GaussianNB()
MultiModel.fit(xData,yLabel)


# In[69]:


evaluationMetric(MultiModel,X_test,y_test)


# ### SVM

# In[70]:


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
t1  = time.time()
svmModel = SVC(kernel='linear',probability = True)
param_grid = {'C': [0.1, 1, 10,100],
              'kernel' : ['linear']
              }
gridModel = GridSearchCV(svmModel, param_grid,refit = True, verbose = 5, cv=5, n_jobs=4 ,iid = True)
cv_results = gridModel.fit(xData,yLabel.values.ravel())
print(cv_results)
t2 = time.time() -t1
print("Time taken to execute the fitting task :", t2)


# In[71]:


print(cv_results.best_score_ )
print(cv_results.best_params_ )
print(cv_results.cv_results_ )
bestEstimatorSVC = cv_results.best_estimator_ 


# In[72]:


evaluationMetric(bestEstimatorSVC,X_test,y_test)


# ### Logistic Regression

from sklearn.linear_model import LogisticRegression
logisticModel = LogisticRegression(solver='lbfgs',multi_class='multinomial',penalty = 'l2')
grid_values = {'C': [0.001,0.01,0.1,1,10,100,1000]}
gridModel = GridSearchCV(logisticModel, grid_values,refit = True, verbose = 5, cv=5, n_jobs=4 ,iid = True)
cv_results = gridModel.fit(xData,yLabel.values.ravel())
print(cv_results)
t2 = time.time() -t1
print("Time taken to execute the fitting task :", t2)


# In[36]:


print(cv_results.best_score_ )
print(cv_results.best_params_ )
print(cv_results.cv_results_ )
bestEstimatorLogistic = cv_results.best_estimator_ 

evaluationMetric(bestEstimatorLogistic,X_test,y_test)
from sklearn.cluster import KMeans
import collections
kmeans = KMeans(n_clusters=3, random_state=0).fit(xData)
clusterPredictedLabel = kmeans.labels_

collections.Counter(clusterPredictedLabel)
np.bincount(clusterPredictedLabel)