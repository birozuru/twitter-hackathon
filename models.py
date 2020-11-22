#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pickle
import re
import numpy as np
from tqdm import tqdm
import nltk


# In[2]:


nltk.download('stopwords')


# In[3]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor


# In[4]:


def preprocess_tweet(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower())
    text = text+' '.join(emoticons).replace('-', '') 
    return text


# In[6]:


suicidal_tweets=pd.read_csv("suicidal_data.csv")


# In[7]:


tqdm.pandas()
suicidal_tweets['tweet'] = suicidal_tweets['tweet'].progress_apply(preprocess_tweet)


# In[8]:


from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


# In[9]:


from nltk.corpus import stopwords
stop = stopwords.words('english')


# In[11]:


[w for w in tokenizer_porter('a swimmer likes swimming and swims a lot') if w not in stop]


# In[12]:


def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\(|D|P)',text.lower())
    text = re.sub('[\W]+', ' ', text.lower())
    text += ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in tokenizer_porter(text) if w not in stop]
    return tokenized


# # Using the Hashing Vectorizer

# In[13]:


from sklearn.feature_extraction.text import HashingVectorizer
vect = HashingVectorizer(decode_error='ignore', n_features=2**21, 
                         preprocessor=None,tokenizer=tokenizer)


# # Building the ml model

# In[14]:


from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(loss='log', random_state=1)


# In[15]:


X = suicidal_tweets["tweet"].to_list()
y = suicidal_tweets['label']


# # Training the model

# In[16]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# In[17]:


X_train = vect.transform(X_train)
X_test = vect.transform(X_test)


# In[18]:


classes = np.array([0, 1])
clf.partial_fit(X_train, y_train,classes=classes)


# In[19]:


print('Accuracy: %.3f' % clf.score(X_test, y_test))


# # Testing

# In[20]:


label = {0:'negative', 1:'positive'}


# In[21]:


phrases = ["suicide","suicidal", "kill myself", "my suicide note", "my suicide letter", "end my life", "never wake up", "can't go on"," cannot go on", "not worth living", "ready to jump", "swleep forever", "want to die", "be dead", "better off without me", "better off dead", "don't want to being here", "tired of living", "die alone", "go to sleep forever", "wanna die", "wanna suicide", "commit suicide", "slit my wrist","cut my wrist", "slash my wrist", "do not want to be here", "want it to be over", "want to be dead", "nothing to live for", "not worth living","ready to die", "thoughts of suicide", "thoughts of killing myself", "why should i live", "take my own life", "depressed"]


# In[37]:


from config import create_api


# In[59]:


streamed_tweets = []
for query in phrases:
    api = create_api()
    tweets = api.search(q=query + " --filter:retweets", lang="en", show_user=True, result_type="recent")
    for i in tweets:
        print(i.user.screen_name, ":" ,i.user.location)
        streamed_tweets.append({'tweet_id': i.id, 'tweet_text': i.text, 'user_name': i.user.screen_name, 'tweet_loc': i.user.location})
        


# In[60]:


streamed_tweet_df=pd.DataFrame.from_dict(streamed_tweets)
streamed_tweet_df


# In[61]:


analysis= pd.DataFrame(columns={"Prediction","Probability"})
analysis["Prediction"].astype(object)
analysis["Probability"].astype(float)


# In[62]:


for row in range(0,len(streamed_tweet_df)):
    example=[streamed_tweet_df.iloc[row,1]]
    X = vect.transform(example)
    print('Prediction: %s   Probability: %.2f%%'
    %(label[clf.predict(X)[0]],np.max(clf.predict_proba(X))*100))
    analysis=analysis.append({'Prediction': label[clf.predict(X)[0]],
                                             'Probability': np.max(clf.predict_proba(X))*100},
                                            ignore_index=True)


# In[63]:


merge=streamed_tweet_df.merge(analysis, how="inner", left_index=True, right_index=True)
merge


# In[64]:


mask = (merge['Probability'].ge(80)) & (merge['Prediction']=='positive')
merge['mask'] = np.where(mask,'Flagged', 'no problem')
merge


# In[65]:


for row in range(len(merge)):
        if merge['mask'][row]=='Flagged':
            print(row, merge['user_name'][row], merge['tweet_text'][row], merge['tweet_loc'][row])


# In[74]:


def send():
    user_list = {}
    for row in range(len(merge)):
        if merge['mask'][row]=='Flagged':
            user_list[merge['user_name'][row]] = merge['tweet_text'][row]
    return(user_list)


# In[75]:


send()