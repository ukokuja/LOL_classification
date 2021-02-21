#!/usr/bin/env python
# coding: utf-8

# <img src="https://camo.githubusercontent.com/9f7c69771104a2df48a040b897a44ad5387f07f77e1e7e5159e68e874fdf8c7f/68747470733a2f2f7777772e6869742e61632e696c2f2e75706c6f61642f61636164656d69632d656e7472657072656e657572736869702f697269732f706172746e6572732f7368656e6b61724c6f676f2e6a7067">
# 
# # Final project 
# ## Lecturer: Eyal Nussbaum
# ## Student: Lucas Kujawski

# ## 1. Intro
# In the following notebook, we will analyse a dataset describing League of Legends matches.
# 
# The dataset contains around 65K rows, containing on each:
# 
# * ##### gameId: number
# Unique Riot game ID.
# 
# * ##### gameDuraton: number
# Game Duration(seconds)
# 
# * ##### <span style="color:blue">blueWardPlaced</span>/<span style="color:red">redWardPlaced</span>: number
# <span style="color:blue">blue</span>/<span style="color:red">red</span> team ward placed counts(Number of warding totems)
# 
# * ##### <span style="color:blue">blueWardkills</span>/<span style="color:red">redWardkills</span>: number
# <span style="color:blue">blue</span>/<span style="color:red">red</span> team ward killed counts(Number of warding killed)
# 
# * ##### <span style="color:blue">blueTotalMinionKills</span>/<span style="color:red">redTotalMinionKills</span>: number
# <span style="color:blue">blue</span>/<span style="color:red">red</span> team kill minion counts (includign jungle)
# 
# * ##### <span style="color:blue">blueJungleMinionKills</span>/<span style="color:red">redJungleMinionKills</span>: number
# <span style="color:blue">blue</span>/<span style="color:red">red</span> team kill jungle minion counts
# 
# * ##### <span style="color:blue">blueTotalHeal</span>/<span style="color:red">redTotalHeal</span>: number
# <span style="color:blue">blue</span>/<span style="color:red">red</span> team heal amounts
# 
# * ##### FirstBlood: categorical - <span style="color:blue">Blue</span>/<span style="color:red">Red</span>
# Which team got the first kill of an enemy champion
# 
# * ##### FirstTower: categorical - <span style="color:blue">Blue</span>/<span style="color:red">Red</span>
# Which team first destroyed an enemy turret
# 
# * ##### FirstBaron: categorical - <span style="color:blue">Blue</span>/<span style="color:red">Red</span>
# Which team first killed Baron Nashor
# 
# * ##### FirstDragon: categorical - <span style="color:blue">Blue</span>/<span style="color:red">Red</span>
# Which team first killed a Dragon
# 
# * ##### win: Target Class - <span style="color:blue">Blue</span>/<span style="color:red">Red</span>
# Who won the game
# 

# In[1]:
#
#
# get_ipython().system('pip install numpy')
# get_ipython().system('pip install matplotlib')
# get_ipython().system('pip install seaborn')
# get_ipython().system('pip install pandas')
# get_ipython().system('pip install sklearn')
# get_ipython().system('pip install scipy')
# get_ipython().system('pip install pydotplus')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import datetime
from pandas.api.types import is_numeric_dtype
import sklearn as skl
from scipy.stats import skewnorm
import scipy.stats as stats
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
import pydotplus


# In[2]:


df = pd.read_csv('lol3.csv')


# ## 2. Initial data analysis

# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


#definitions
int_columns = ['blueWardPlaced', 'blueWardkills', 'blueTotalMinionKills', 'blueJungleMinionKills',
               'blueTotalHeal', 'redWardPlaced', 'redWardkills', 'redTotalMinionKills',
               'redJungleMinionKills', 'redTotalHeal']
stats_columns = ['win', 'FirstBlood', 'FirstTower', 'FirstBaron', 'FirstDragon']
teams_columns = ['WardPlaced', 'Wardkills', 'TotalMinionKills', 'JungleMinionKills','TotalHeal']
teams = ['blue', 'red']


# ### 2.1 Feature statistical analysis

# In[6]:


df.describe()


# In[7]:


df.info()


# We see that there are many null values in the following columns:
# 
#     1. blueWardkills
#     2. redWardkills
#     3. redTotalMinionKills
#     4. redJungleMinionKills
#     5. FirstBlood
#     6. FirstBaron

# ### 2.2 Data fixes: 

# * We might have repeated values in gameId, let's remove them. Then lets set gameId as index

# In[8]:


df.drop_duplicates(subset='gameId', keep="first", inplace=True)
df.set_index('gameId')


# * In order to replace NA values, let's analyse their relation with other columns.

# In[9]:


def replace_na(_df, x, y):
    int_df = _df[int_columns]
    non_NA_int_df = int_df[int_df > 0].dropna(inplace=False)
    relation = (non_NA_int_df[x]/non_NA_int_df[y]).mean()
    def relation_replace(row):
        if pd.isna(row[x]) and not pd.isna(row[y]):
            row[x] = row[y] * relation
        return row
    return _df.apply(relation_replace, axis=1)
df = replace_na(df,'blueWardkills', 'redWardPlaced')
df = replace_na(df,'redWardkills', 'blueWardPlaced')
df = replace_na(df,'redTotalMinionKills', 'redJungleMinionKills')
df = replace_na(df,'redJungleMinionKills', 'redTotalMinionKills')


# In[10]:


df.info()


# * 0 are also values that we would like to fix, let's try to see how many do we have

# In[11]:


df[int_columns] = df[int_columns].replace([0], np.nan)
df.info()


# * We remove unconsistent rows:
# 1. number of ward killed of a team > number of ward placed of the adversary
# 2. minion jungle kills of a team > total jungle kills of a team

# In[12]:


print("Removing {} rows".format(len(df[df['blueWardkills'] > df['redWardPlaced']])))
print("Removing {} rows".format(len(df[df['redWardkills'] > df['blueWardPlaced']])))
print("Removing {} rows".format(len(df[df['redJungleMinionKills'] > df['redTotalMinionKills']])))
print("Removing {} rows".format(len(df[df['blueJungleMinionKills'] > df['blueTotalMinionKills']])))

df = df[df['blueWardkills'] <= df['redWardPlaced']]
df = df[df['redWardkills'] <= df['blueWardPlaced']]
df = df[df['redJungleMinionKills'] <= df['redTotalMinionKills']]
df = df[df['blueJungleMinionKills'] <= df['blueTotalMinionKills']]
df.head()


# * NA Values on win or FirstX invalidates the row as we can't analyse what happen on that game. We remove those lines
# * We have only a few NA values on integer column. So we remove them

# In[13]:


print("Removing {} rows".format(df.shape[0] - df.dropna().shape[0]))
df.dropna(inplace=True)


# * We remove games shorter than a minute

# In[14]:


df = df[df['gameDuraton'] > 60]
df.head()


# * Numeric columns are integer.

# In[15]:


df[int_columns] = df[int_columns].astype('int32')
df[stats_columns] = df[stats_columns].astype(str).apply(lambda x: x.str.lower())


# In[16]:


df['minutesDuration'] = (df['gameDuraton'] / 60).astype('int32')
df.drop(['gameDuraton'], axis=1, inplace=True)


# ##### Data manipulation
# * As it does not matter if the team is blue or red, we will switch it to winner/looser

# In[17]:


# for column in teams_columns:
#     df['winner{column}'.format(column=column)] = \
#     df.apply(lambda row: row['red{column}'.format(column=column)] 
#              if row['win'] == 'red' else row['blue{column}'.format(column=column)], axis=1)
#     df['looser{column}'.format(column=column)] = \
#     df.apply(lambda row: row['red{column}'.format(column=column)] 
#              if row['win'] != 'red' else row['blue{column}'.format(column=column)], axis=1)
# for column in stats_columns[1:]:
#     df['is{column}Winner'.format(column=column)] = (df['win'] ==  df['{column}'.format(column=column)])


# In[18]:


def winStats(row):
    if row['win'] == 'red':
        row['winnerKillsRate'] = row['redWardkills']/df['minutesDuration']
        row['winnerTotalKills'] = row['redWardkills']
        row['winnerWardkillsPercentage'] = row['redWardkills']/row['blueWardPlaced']
        row['winnerMinionkillsPercentage'] = row['redJungleMinionKills']/row['redTotalMinionKills']
    else:
        row['winnerKillsRate'] = row['blueWardkills']/df['minutesDuration']
        row['winnerTotalKills'] = row['blueWardkills']
        row['winnerWardkillsPercentage'] = row['blueWardkills']/row['redWardPlaced']
        row['winnerMinionkillsPercentage'] = row['blueJungleMinionKills']/row['blueTotalMinionKills']
    return row        
df = df.apply(winStats, axis=1)

for column in stats_columns[1:]:
    df['is{column}Winner'.format(column=column)] = (df['win'] ==  df['{column}'.format(column=column)])


# In[19]:


#df['isRedWinner'.format(column=column)] = df['win'] ==  "red"


# In[20]:


#for team in teams:
#    for column in teams_columns:
#        df.drop(['{team}{column}'.format(team=team, column=column)], axis=1, inplace=True)
        
#for column in stats_columns[1:]:
#    df.drop([column], axis=1, inplace=True)


# * Counts the times that winner and looser were the first team to achieve the milestones

# In[21]:


first_columns = ['isFirstBloodWinner', 'isFirstTowerWinner', 'isFirstBaronWinner','isFirstDragonWinner']
first_df = df[first_columns].astype('int32')
df['winnerFirstTotal'] = first_df.sum(axis=1)


# * Counts the times that winner and looser were the first team to achieve the milestones

# In[22]:


# first_columns = ['isFirstBloodWinner', 'isFirstTowerWinner', 'isFirstBaronWinner','isFirstDragonWinner']
# first_df = df[first_columns].astype('int32')
# looser_df = (df[first_columns] == 0).astype('int32')
# df['total'] = first_df.sum(axis=1)
# looser_df['total'] = looser_df.sum(axis=1)


# #### Summary: 

# * We replace numeric integer NA values for 0, as we assume that this could be a reasonable value.
# * NA Values on win or FirstX invalidates the row as we can't analyse what happen on that game. We remove those lines
# * We remove unconsistent rows:
# 1. number of ward killed of a team > number of ward placed of the adversary
# 2. minion jungle kills of a team > total jungle kills of a team
# * We remove games shorter than a minute
# * NA Values on win or FirstX invalidates the row as we can't analyse what happen on that game. We remove those lines
# * Numeric columns are integer.
# * As it does not matter if the team is blue or red, we will switch it to winner/looser

# In[23]:


df.head()


# ## Exploratory Data Analysis

# In[24]:


plot = sns.violinplot(x='win', y='winnerFirstTotal',
                        order=['red', 'blue'],
                      palette=['r','b'],
                       data=df)


# * When a red team won, they often were the first to do three or more of the following achievements: team got the first kill of an enemy champion, team first destroyed an enemy turret, team first killed Baron Nashor, team first killed a Dragon. Instead, when a blue won, they often did two or three of them.

# In[25]:


plt.scatter(x=df['minutesDuration'], y=df['redWardkillsPercentage'], s=(df['winnerFirstTotal']) ** 4,
                    c=df['win'], alpha=1,edgecolors='k')
plt.title("winnerTotalKills vs minutesDuration")
plt.xlabel("minutesDuration")
plt.ylabel("winnerTotalKills")


# In[ ]:


plt.scatter(x=df['minutesDuration'], y=df['winnerMinionkillsPercentage'], s=df['winnerTotalKills'],
                    c=df['win'], alpha=1,edgecolors='k')
plt.title("winnerTotalKills vs minutesDuration")
plt.xlabel("minutesDuration")
plt.ylabel("winnerTotalKills")


# In[ ]:


corr = df.corr().abs()
mask = np.triu(corr)
cmap = sns.diverging_palette(200, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, center=0.5, square=True, linewidths=.5).set_title("Correlation")


# ## Classification Model

# In[ ]:


df.columns


# In[ ]:


# sns.pairplot(df[['redJungleMinionKills', 'blueJungleMinionKills', 'redTotalHeal', 'total', 'win']], hue='win', height=1.5)


# In[ ]:


#sns.pairplot(df[['redJungleMinionKills', 'blueJungleMinionKills', 'redWardPlaced', 'looserWardPlaced', 'win']], hue='win', height=1.5)


# isFirstBaronWinner and total show a good split

# In[ ]:


# import random
# lst = list(df.sample(n=4,axis='columns',replace=True).columns)
# print(lst)
# sns.pairplot(df[lst + ['win']], hue='win', height=1.5)


# ### 3.1 Gaussian Naïve Bayes

# In[ ]:


import random
for i in range(100):
    lst = ['blueWardPlaced', 'blueWardkills', 'blueTotalMinionKills',
       'blueJungleMinionKills', 'blueTotalHeal', 'redWardPlaced',
       'redWardkills', 'redTotalMinionKills', 'redJungleMinionKills',
       'redTotalHeal', 'win', 'FirstBlood', 'FirstTower', 'FirstBaron',
       'FirstDragon', 'minutesDuration']
    try:
        items = random.sample(lst, 2)
        X = df[[items[0], items[1]]]
        Y = df['win']
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2,
                                                        random_state=1)

        model = GaussianNB()                       # 2. instantiate model
        model.fit(Xtrain, ytrain)                  # 3. fit model to data
        y_model = model.predict(Xtest)             # 4. predict on new data (output is numpy array)

        ypred = pd.Series(y_model,name="prediction")
        predicted = pd.concat([Xtest.reset_index(),ytest.reset_index(),ypred],axis=1)
        accuracy = metrics.accuracy_score(ytest, y_model)
        if accuracy > 0.646:
            print("{} vs {} accuracy: ".format(items[0], items[1]), accuracy)
    except Exception as e:
        print(e)
    


# In[ ]:


X = df[['redJungleMinionKills', 'blueJungleMinionKills']]
Y = df['win']
def bayes_plot(df,model="gnb",spread=30):
    df.dropna()
    colors = 'seismic'
    col1 = df.columns[0]
    col2 = df.columns[1]
    target = df.columns[2]
    sns.scatterplot(data=df, x=col1, y=col2,hue=target)
    plt.show()
    y = df[target]  # Target variable
    X = df.drop(target, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=1)  # 70% training and 30% test

    clf = GaussianNB()
    if (model != "gnb"):
        clf = DecisionTreeClassifier(max_depth=model)
    clf = clf.fit(X_train, y_train)
    
    # Train Classifer
    

    prob = len(clf.classes_) == 2

    # Predict the response for test dataset

    y_pred = clf.predict(X_test)
    print(metrics.classification_report(y_test, y_pred))

    hueorder = clf.classes_
    def numify(val):
        return np.where(clf.classes_ == val)[0]

    Y = y.apply(numify)
    x_min, x_max = X.loc[:, col1].min() - 1, X.loc[:, col1].max() + 1
    y_min, y_max = X.loc[:, col2].min() - 1, X.loc[:, col2].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2),
                         np.arange(y_min, y_max, 0.2))

    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    if prob:

        Z = Z[:,1]-Z[:,0]
    else:
        colors = "Set1"
        Z = np.argmax(Z, axis=1)


    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=colors, alpha=0.5)
    plt.colorbar()
    if not prob:
        plt.clim(0,len(clf.classes_)+3)
    sns.scatterplot(data=df[::spread], x=col1, y=col2, hue=target, hue_order=hueorder,palette=colors)
    fig = plt.gcf()
    fig.set_size_inches(12, 8)
    plt.show()


bayes_plot(pd.concat([X,Y],axis=1),spread=1)


# In[ ]:


X = df[['redJungleMinionKills', 'blueJungleMinionKills']]
Y = df['win']
def bayes_plot(df,model="gnb",spread=30):
    df.dropna()
    colors = 'seismic'
    col1 = df.columns[0]
    col2 = df.columns[1]
    target = df.columns[2]
    sns.scatterplot(data=df, x=col1, y=col2,hue=target)
    plt.show()
    y = df[target]  # Target variable
    X = df.drop(target, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=1)  # 70% training and 30% test

    clf = GaussianNB()
    if (model != "gnb"):
        clf = DecisionTreeClassifier(max_depth=model)
    clf = clf.fit(X_train, y_train)
    
    # Train Classifer
    

    prob = len(clf.classes_) == 2

    # Predict the response for test dataset

    y_pred = clf.predict(X_test)
    print(metrics.classification_report(y_test, y_pred))

    hueorder = clf.classes_
    def numify(val):
        return np.where(clf.classes_ == val)[0]

    Y = y.apply(numify)
    x_min, x_max = X.loc[:, col1].min() - 1, X.loc[:, col1].max() + 1
    y_min, y_max = X.loc[:, col2].min() - 1, X.loc[:, col2].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2),
                         np.arange(y_min, y_max, 0.2))

    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    if prob:

        Z = Z[:,1]-Z[:,0]
    else:
        colors = "Set1"
        Z = np.argmax(Z, axis=1)


    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=colors, alpha=0.5)
    plt.colorbar()
    if not prob:
        plt.clim(0,len(clf.classes_)+3)
    sns.scatterplot(data=df[::spread], x=col1, y=col2, hue=target, hue_order=hueorder,palette=colors)
    fig = plt.gcf()
    fig.set_size_inches(12, 8)
    plt.show()


bayes_plot(pd.concat([X,Y],axis=1),spread=1)


# In[ ]:


# sns.pairplot(df[['redJungleMinionKills', 'blueJungleMinionKills', 'redTotalHeal', 'total', 'win']], hue='win', height=1.5)


# In[ ]:


#sns.pairplot(df[['redJungleMinionKills', 'blueJungleMinionKills', 'redWardPlaced', 'looserWardPlaced', 'win']], hue='win', height=1.5)


# isFirstBaronWinner and total show a good split

# In[ ]:


# import random
# lst = list(df.sample(n=4,axis='columns',replace=True).columns)
# print(lst)
# sns.pairplot(df[lst + ['win']], hue='win', height=1.5)


# ### 3.1 Gaussian Naïve Bayes

# In[ ]:


for i in range(100):
    lst = ['blueWardPlaced', 'blueWardkills', 'blueTotalMinionKills',
       'blueJungleMinionKills', 'blueTotalHeal', 'redWardPlaced',
       'redWardkills', 'redTotalMinionKills', 'redJungleMinionKills',
       'redTotalHeal', 'win', 'FirstBlood', 'FirstTower', 'FirstBaron',
       'FirstDragon', 'minutesDuration']
    try:
        items = random.sample(lst, 2)
        X = df[[items[0], items[1]]]
        Y = df['win']
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2,
                                                        random_state=1)

        model = GaussianNB()                       # 2. instantiate model
        model.fit(Xtrain, ytrain)                  # 3. fit model to data
        y_model = model.predict(Xtest)             # 4. predict on new data (output is numpy array)

        ypred = pd.Series(y_model,name="prediction")
        predicted = pd.concat([Xtest.reset_index(),ytest.reset_index(),ypred],axis=1)
        accuracy = metrics.accuracy_score(ytest, y_model)
        if accuracy > 0.646:
            print("{} vs {} accuracy: ".format(items[0], items[1]), accuracy)
    except Exception as e:
        print("error")
    


# In[ ]:


X = df[['redJungleMinionKills', 'blueJungleMinionKills']]
Y = df['win']
def bayes_plot(df,model="gnb",spread=30):
    df.dropna()
    colors = 'seismic'
    col1 = df.columns[0]
    col2 = df.columns[1]
    target = df.columns[2]
    sns.scatterplot(data=df, x=col1, y=col2,hue=target)
    plt.show()
    y = df[target]  # Target variable
    X = df.drop(target, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=1)  # 70% training and 30% test

    clf = GaussianNB()
    if (model != "gnb"):
        clf = DecisionTreeClassifier(max_depth=model)
    clf = clf.fit(X_train, y_train)
    
    # Train Classifer
    

    prob = len(clf.classes_) == 2

    # Predict the response for test dataset

    y_pred = clf.predict(X_test)
    print(metrics.classification_report(y_test, y_pred))

    hueorder = clf.classes_
    def numify(val):
        return np.where(clf.classes_ == val)[0]

    Y = y.apply(numify)
    x_min, x_max = X.loc[:, col1].min() - 1, X.loc[:, col1].max() + 1
    y_min, y_max = X.loc[:, col2].min() - 1, X.loc[:, col2].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2),
                         np.arange(y_min, y_max, 0.2))

    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    if prob:

        Z = Z[:,1]-Z[:,0]
    else:
        colors = "Set1"
        Z = np.argmax(Z, axis=1)


    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=colors, alpha=0.5)
    plt.colorbar()
    if not prob:
        plt.clim(0,len(clf.classes_)+3)
    sns.scatterplot(data=df[::spread], x=col1, y=col2, hue=target, hue_order=hueorder,palette=colors)
    fig = plt.gcf()
    fig.set_size_inches(12, 8)
    plt.show()


bayes_plot(pd.concat([X,Y],axis=1),spread=1)

