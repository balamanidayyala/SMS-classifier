#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


# In[2]:


df = pd.read_csv('spam.csv', encoding='latin1')


# In[3]:


df.sample(10)


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)


# In[7]:


df.sample(5)


# In[8]:


df.columns = ['category','Message']


# In[9]:


df.head(10)


# In[10]:


df['category'].value_counts()


# In[21]:


df['category'].value_counts().plot(kind='bar')


# In[23]:


df['spam'] = df['category'].apply(lambda x: 'SPAM' if x == 'spam' else'NOT SPAM')


# In[24]:


df.head(10)


# In[25]:


x = np.array(df["Message"]) 
y = np.array(df["spam"])
cv = CountVectorizer()
X = cv.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

clf = MultinomialNB()
clf.fit(X_train,y_train)


# In[26]:


sample = input('Enter a message: ')
df = cv.transform([sample]).toarray()
print(clf.predict(df))


# In[27]:


sample = input('Enter a message: ')
df = cv.transform([sample]).toarray()
print(clf.predict(df))


# In[28]:


clf.score(X_test,y_test)


# In[ ]:




