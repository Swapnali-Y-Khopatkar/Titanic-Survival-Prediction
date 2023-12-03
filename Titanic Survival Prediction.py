#!/usr/bin/env python
# coding: utf-8

# # Import the required models and libraries

# In[1]:


# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

import warnings
warnings.filterwarnings("ignore")

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# # 1) Acquire data
# The Python Pandas packages helps us work with our datasets. We start by acquiring the  dataset into Pandas DataFrames.

# In[2]:


df=pd.read_csv('Titanic-Dataset.csv')
df


# ## 2)Analyze by describing data
# Pandas also helps describe the datasets answering following questions early in our project.
# 
# Which features are available in the dataset?
# 
# Noting the feature names for directly manipulating or analyzing these.

# In[3]:


print(df.columns.values)


# ### Which features are categorical?
# 
# These values classify the samples into sets of similar samples. Within categorical features are the values nominal, ordinal, ratio, or interval based? Among other things this helps us select the appropriate plots for visualization.
# 
# Categorical: Survived, Sex, and Embarked. Ordinal: Pclass.
# ### Which features are numerical?
# 
# Which features are numerical? These values change from sample to sample. Within numerical features are the values discrete, continuous, or timeseries based? Among other things this helps us select the appropriate plots for visualization.
# 
# Continous: Age, Fare. Discrete: SibSp, Parch.

# In[4]:


#What are the data types for various features?
df.info()


# In[5]:


df.isnull().sum() #To check Which features contain blank, null or empty values?


#  Cabin , Age , Embarked features contain a number of null values in that order for the training dataset.
# 

# ## 3)Some Predictions:
# Sex: Females are more likely to survive.
# 
# SibSp/Parch: People traveling alone are more likely to survive.
# 
# Age: Young children are more likely to survive.
# 
# Pclass: People of higher socioeconomic class are more likely to survive.

# ## 4) Data Visualization
# It's time to visualize our data so we can see whether our predictions were accurate!

# In[6]:


#draw a bar plot of survival by sex
sns.barplot(x="Sex", y="Survived", data=df)

#print percentages of females vs. males that survive
print("Percentage of females who survived:", df["Survived"][df["Sex"] == 'female'].value_counts(normalize = True)[1]*100)

print("Percentage of males who survived:", df["Survived"][df["Sex"] == 'male'].value_counts(normalize = True)[1]*100)


# As predicted, females have a much higher chance of survival than males. The Sex feature is essential in our predictions.
# 
# 

# In[7]:


#draw a bar plot of survival by Pclass
sns.barplot(x="Pclass", y="Survived", data=df)

#print percentage of people by Pclass that survived
print("Percentage of Pclass = 1 who survived:", df["Survived"][df["Pclass"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of Pclass = 2 who survived:", df["Survived"][df["Pclass"] == 2].value_counts(normalize = True)[1]*100)

print("Percentage of Pclass = 3 who survived:", df["Survived"][df["Pclass"] == 3].value_counts(normalize = True)[1]*100)


# As predicted, people with higher socioeconomic class had a higher rate of survival. (62.9% vs. 47.3% vs. 24.2%)
# 
# 

# In[8]:


#draw a bar plot for SibSp vs. survival
sns.barplot(x="SibSp", y="Survived", data=df)

#I won't be printing individual percent values for all of these.
print("Percentage of SibSp = 0 who survived:", df["Survived"][df["SibSp"] == 0].value_counts(normalize = True)[1]*100)

print("Percentage of SibSp = 1 who survived:", df["Survived"][df["SibSp"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of SibSp = 2 who survived:", df["Survived"][df["SibSp"] == 2].value_counts(normalize = True)[1]*100)


# In general, it's clear that people with more siblings or spouses aboard were less likely to survive. However, contrary to expectations, people with no siblings or spouses were less to likely to survive than those with one or two. (34.5% vs 53.4% vs. 46.4%)
# 
# 

# In[9]:


#sort the ages into logical categories
df["Age"] = df["Age"].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
df['AgeGroup'] = pd.cut(df["Age"], bins, labels = labels)


#draw a bar plot of Age vs. survival
sns.barplot(x="AgeGroup", y="Survived", data=df)
plt.show()


# Babies are more likely to survive than any other age group.
# 
# 

# In[10]:


df["CabinBool"] = (df["Cabin"].notnull().astype('int'))


#calculate percentages of CabinBool vs. survived
print("Percentage of CabinBool = 1 who survived:", df["Survived"][df["CabinBool"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of CabinBool = 0 who survived:", df["Survived"][df["CabinBool"] == 0].value_counts(normalize = True)[1]*100)
#draw a bar plot of CabinBool vs. survival
sns.barplot(x="CabinBool", y="Survived", data=df)
plt.show()


# People with a recorded Cabin number are, in fact, more likely to survive. (66.6% vs 29.9%)
# 
# 

# ### 5) Cleaning Data
# Time to clean our data to account for missing values and unnecessary information!

# In[11]:


df.describe()


# In[12]:


df.info()


# In[13]:


df.isnull().sum()


# In[14]:


#we'll start off by dropping the Cabin feature since not a lot more useful information can be extracted from it.
df = df.drop(['Cabin'], axis = 1)


# In[15]:


#we can also drop the Ticket feature since it's unlikely to yield any useful information
df = df.drop(['Ticket'], axis = 1)


# In[16]:


#now we need to fill in the missing values in the Embarked feature
print("Number of people embarking in Southampton (S):")
southampton = df[df["Embarked"] == "S"].shape[0]
print(southampton)

print("Number of people embarking in Cherbourg (C):")
cherbourg = df[df["Embarked"] == "C"].shape[0]
print(cherbourg)

print("Number of people embarking in Queenstown (Q):")
queenstown = df[df["Embarked"] == "Q"].shape[0]
print(queenstown)


# It's clear that the majority of people embarked in Southampton (S). Let's go ahead and fill in the missing values with S.
# 
# 

# In[17]:


#replacing the missing values in the Embarked feature with S
df = df.fillna({"Embarked": "S"})


# In[18]:


df.select_dtypes(["int64", "float64"])


# In[19]:


df.select_dtypes(["object"])


# In[20]:


#map each Sex value to a numerical value
sex_mapping = {"male": 0, "female": 1}
df['Sex'] = df['Sex'].map(sex_mapping)

df.head()


# In[21]:


#map each Embarked value to a numerical value
embarked_mapping = {"S": 1, "C": 2, "Q": 3}
df['Embarked'] = df['Embarked'].map(embarked_mapping)


df.head()


# In[22]:


df=df.drop(['Name'],axis=1)


# In[23]:


df.columns


# In[24]:


df.info()


# In[25]:


df=df.drop(['AgeGroup'] , axis=1)


# ## 6) Choosing the Best Model
# Splitting the Training Data
# We will use part of our training data (22% in this case) to test the accuracy of our different models.

# In[26]:


from sklearn.model_selection import train_test_split

x = df.drop(['Survived', 'PassengerId'], axis=1)
y = df["Survived"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.22, random_state = 0)


# In[27]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
acc_logreg = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_logreg)


# In[28]:


# Support Vector Machines
from sklearn.svm import SVC

svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
acc_svc = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_svc)


# In[29]:


# Linear SVC
from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_test)
acc_linear_svc = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_linear_svc)


# In[30]:


#Decision Tree
from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_test)
acc_decisiontree = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_decisiontree)


# In[31]:


# Random Forest
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_test)
acc_randomforest = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_randomforest)


# In[32]:


# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_test)
acc_gbk = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_gbk)


# In[33]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'Logistic Regression', 
              'Random Forest', 'Linear SVC', 
              'Decision Tree', 'Gradient Boosting Classifier'],
    'Score': [acc_svc, acc_logreg, 
              acc_randomforest,acc_linear_svc, acc_decisiontree,
               acc_gbk]})
models.sort_values(by='Score', ascending=False)


# ## I decided to use the Gradient Boosting Classifier model for the testing data.
# 
# 

# ## Thank you!!
