
# coding: utf-8

# ### Fake News Classifier
# Dataset:  https://www.kaggle.com/c/fake-news/data#

# In[1]:


import pandas as pd


# In[4]:


df=pd.read_csv('C:/Users/nealn/Desktop/NLP/FakeNewsClassifier/train.csv/train.csv')


# In[5]:


df.head()


# In[6]:


## Get the Independent Features

X=df.drop('label',axis=1)


# In[7]:


X.head()


# In[8]:


## Get the Dependent features
y=df['label']


# In[9]:


y.head()


# In[10]:


df.shape


# In[12]:


get_ipython().system('pip install sklearn')


# In[13]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer


# In[14]:


df=df.dropna()


# In[15]:


df.head(10)


# In[16]:


messages=df.copy()


# In[17]:


messages.reset_index(inplace=True)


# In[18]:


messages.head(10)


# In[19]:


messages['title'][6]


# In[21]:


import re


# In[22]:


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['title'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


# In[24]:


corpus[3]


# In[25]:


## Applying Countvectorizer
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,ngram_range=(1,3))
X = cv.fit_transform(corpus).toarray()


# In[26]:


X.shape


# In[27]:


y=messages['label']


# In[28]:


## Divide the dataset into Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)


# In[29]:


cv.get_feature_names()[:20]


# In[30]:


cv.get_params()


# In[31]:


count_df = pd.DataFrame(X_train, columns=cv.get_feature_names())


# In[32]:


count_df.head()


# In[33]:


import matplotlib.pyplot as plt


# In[34]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    See full source and example: 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# ### MultinomialNB Algorithm

# In[35]:



from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB()


# In[36]:


from sklearn import metrics
import numpy as np
import itertools


# In[37]:


classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred)
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])


# In[38]:


classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
score


# In[39]:


y_train.shape


# ### Passive Aggressive Classifier Algorithm

# In[42]:


from sklearn.linear_model import PassiveAggressiveClassifier
linear_clf = PassiveAggressiveClassifier()


# In[43]:


linear_clf.fit(X_train, y_train)
pred = linear_clf.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred)
plot_confusion_matrix(cm, classes=['FAKE Data', 'REAL Data'])


# ### Multinomial Classifier with Hyperparameter

# In[44]:


classifier=MultinomialNB(alpha=0.1)


# In[45]:


previous_score=0
for alpha in np.arange(0,1,0.1):
    sub_classifier=MultinomialNB(alpha=alpha)
    sub_classifier.fit(X_train,y_train)
    y_pred=sub_classifier.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred)
    if score>previous_score:
        classifier=sub_classifier
    print("Alpha: {}, Score : {}".format(alpha,score))


# In[46]:


## Get Features names
feature_names = cv.get_feature_names()


# In[47]:


classifier.coef_[0]


# In[48]:


### Most real
sorted(zip(classifier.coef_[0], feature_names), reverse=True)[:20]


# In[49]:


### Most fake
sorted(zip(classifier.coef_[0], feature_names))[:5000]

