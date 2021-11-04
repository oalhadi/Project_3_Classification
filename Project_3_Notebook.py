#!/usr/bin/env python
# coding: utf-8

# # Diabetes Prediction

# ### Done By
# Omar Alhadi

# ## Importing the libraries

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ## Importing the data and reading the data

# In[3]:


df = pd.read_csv('diabetes.csv')
df.head()


# ## EDA of Data

# In[4]:


df.keys()


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df.isna().sum()


# In[9]:


df.corr()


# ### Correlation Matrix

# In[10]:


plt.figure(figsize=[12,8])
heatmap = sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True, cmap="Blues")
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=11);
plt.savefig('images_01_Corr_1.png', dpi=200)
plt.show()


# ### Pairplot of data

# In[11]:


sns.pairplot(df)


# ### Count plot specifying the number of people suffering by diabetes

# In[12]:


#number of people suffering by diabetes
sns.countplot(df['Outcome'])
plt.savefig('images_02_countplot_1.png', dpi=200)
plt.show()


# ## Machine Learning Algorithms part

# ### Separating the data into features and target data

# ### K Nearest Neighbors Classifier Model

# In[13]:


X = df.iloc[:,0 :-1]
y = df.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=63)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
knn.predict(X_test)
score = knn.score(X_train,y_train)
score1 = knn.score(X_test,y_test)
print("Accuracy score of Training data is:",score)
print("Accuracy score of Test data is:",score1)


# ### Logistic Regression Model

# In[14]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=10)
logreg.fit(X_train,y_train)
logreg.predict(X_test)
score = logreg.score(X_train,y_train)
score1 = logreg.score(X_test,y_test)
print("Accuracy score of Training data is:",score)
print("Accuracy score of Test data is",score1)


# ### Decision Tree Classifier

# In[15]:


from sklearn.tree import DecisionTreeClassifier
tre = DecisionTreeClassifier()
tre.fit(X_train,y_train)
tre.predict(X_test)
score= tre.score(X_train,y_train)
score1 =tre.score(X_test,y_test)
print("Accuracy score of Training data is:",score)
print("Accuracy score of Test data is:",score1)


# #### Features Importance Bar Plot

# In[16]:


featur_names = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','Diabetics','Age' ]
features = tre.feature_importances_
features


# In[17]:


plt.barh(featur_names,features)
plt.savefig('images_03', dpi=200)
plt.show()


# ### Random Forest Classifier

# In[18]:


from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier()
forest.fit(X_train,y_train)
forest.predict(X_test)
score = forest.score(X_train,y_train)
score1 = forest.score(X_test,y_test)
print("Accuracy score of Training data is:",score)
print("Accuracy score of Test data is:",score1)


# #### Tuning the parameters of the Model to get some improved results

# In[19]:


from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(max_depth=5, n_estimators=100, random_state=0)
forest.fit(X_train,y_train)
forest.predict(X_test)
score = forest.score(X_train,y_train)
score1 = forest.score(X_test,y_test)
print("Accuracy score of Training data is:",score)
print("Accuracy score of Test data is:",score1)


# #### Features Importance Bar Plot

# In[20]:


forest1 = forest.feature_importances_
forest1


# In[21]:


plt.barh(featur_names, forest1)
plt.show()


# ### Gradient Boosting Classifier

# In[22]:


from sklearn.ensemble import GradientBoostingClassifier
grad = GradientBoostingClassifier()
grad.fit(X_train,y_train)
grad.predict(X_test)
score = grad.score(X_train,y_train)
score1 = grad.score(X_test,y_test)
print("Accuracy score of Training data is:",score)
print("Accuracy score of Test data is:",score1)


# #### Tuning some parameters of the model to reduce overfitting of the model

# In[23]:


from sklearn.ensemble import GradientBoostingClassifier
grad = GradientBoostingClassifier(max_depth=1)
grad.fit(X_train,y_train)
grad.predict(X_test)
score = grad.score(X_train,y_train)
score1 = grad.score(X_test,y_test)
print("Accuracy score of Training data is:",score)
print("Accuracy score of Test data is:",score1)


# #### Tuning the parameters of the model to get improved results

# In[24]:


from sklearn.ensemble import GradientBoostingClassifier
grad = GradientBoostingClassifier(learning_rate=0.1)
grad.fit(X_train,y_train)
grad.predict(X_test)
score = grad.score(X_train,y_train)
score1 = grad.score(X_test,y_test)
print("Accuracy score of Training data is:",score)
print("Accuracy score of Test data is:",score1)


# #### Features Importance Bar Plot

# In[25]:


gradf = grad.feature_importances_
gradf


# In[26]:


plt.barh(featur_names, gradf)
plt.show()


# ### Support Vector Machines (SVM) Model

# In[27]:


from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train,y_train)
svm.predict(X_test)
score = svm.score(X_train, y_train)
score1 = svm.score(X_test, y_test)
print("Accuracy score of Training data is:",score)
print("Accuracy score of Test data is:",score1)


# ### Voting Classifier Model

# In[35]:


from sklearn.ensemble import VotingClassifier
from sklearn import model_selection
Clf1= knn
Clf2= logreg
Clf3= tre
Clf4= forest
Clf5= grad
Clf6= svm
print('Cross validation:')

labels = ['K Nearest Neighbors', 'Logistic Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting', 'Support Vector Machines']

for Clf, label in zip([Clf1, Clf2, Clf3, Clf4,Clf5 ,Clf6], labels):

    scores = model_selection.cross_val_score(Clf, X, y, 
                                              cv=5, 
                                              scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]"
          % (scores.mean(), scores.std(), label))


# In[ ]:




