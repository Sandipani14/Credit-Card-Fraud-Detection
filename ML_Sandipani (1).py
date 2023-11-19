#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from matplotlib import gridspec
from sklearn.model_selection import train_test_split, GridSearchCV
import warnings
warnings.filterwarnings('ignore')


data = pd.read_csv("C:/Users/sandi/Downloads/training_data.csv")
data1 = pd.read_csv("C:/Users/sandi/Downloads/train_data_classlabels.csv")
test = pd.read_csv('C:/Users/sandi/Downloads/testing_data.csv')


# In[2]:


data1.value_counts()


# In[3]:


data 


# In[4]:


data.describe()


# In[5]:


# Print the shape of the data 
# data = data.sample(frac = 0.1, random_state = 48) 
print(data.shape) 
print(data.describe().T)


# In[6]:


colors = {0.0: 'blue', 1.0: 'orange'}
plt.scatter(data['Amount'], data1['Class'],
            c=data1['Class'].map(colors))
plt.show()


# In[7]:


# Determine number of fraud cases in dataset 
fraud = data[data1['Class'] == 1] 
valid = data[data1['Class'] == 0] 
names = {'Valid', 'Fraud'}
outlierFraction = len(fraud)/float(len(valid)) 
print('Outlier Fraction:', outlierFraction) 
print(f'Fraud Cases: {len(fraud)}') 
print(f'Valid Transactions: {len(valid)}')


# In[8]:


print("Details of valid transaction") 
data['Amount'].describe()


# In[9]:


# # Correlation matrix 
# corrmat = data.corr() 
# fig = plt.figure(figsize = (12, 9)) 
# sns.heatmap(corrmat, vmax = .8, square = True) 
# plt.show() 


# In[10]:


X = data
y = data1

X.describe


# In[11]:


y.describe


# In[12]:


# Using Scikit-learn to split data into training and testing sets 

# Split the data into training and testing sets 
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size= 0.2, stratify = y)


# In[13]:


X.shape, y.shape, XTrain.shape, yTrain.shape, XTest.shape, yTest.shape


# In[14]:


#Using Pearson Correlation
plt.figure(figsize = (12,10))
cor = XTrain.corr()
sns.heatmap(cor, annot = True, vmax= .8)
plt.show()


# In[15]:


def correlation(data, threshold):
    col_corr = set() #Set of all the names of correlated columns
    corr_matrix = data.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coefficient value
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr            


# In[16]:


corr_features = correlation(XTrain, 0.50)
len(set(corr_features))


# In[17]:


from sklearn.feature_selection import mutual_info_classif
# determine the mutual information
mutual_info = mutual_info_classif(XTrain, yTrain)
mutual_info


# In[18]:


mutual_info = pd.Series(mutual_info)
mutual_info.index =XTrain.columns
mutual_info.sort_values(ascending = False)


# In[19]:


#Lets plot the mutual_info values per feature
mutual_info.sort_values(ascending = False).plot.bar(figsize= (20,8))


# In[20]:


from sklearn.feature_selection import SelectKBest

select_cols = SelectKBest(mutual_info_classif, k=15)

select_cols.fit(XTrain, yTrain)
XTrain.columns[select_cols.get_support()]


# In[21]:


# Fit on the training data
select_cols.fit(XTrain, yTrain)

# Get the selected feature indices
selected_feature_indices = select_cols.get_support()

# Get the names of the selected features
selected_features = XTrain.columns[selected_feature_indices]


# In[22]:


# Use the selected features for training and testing data
XTrain_selected = XTrain[selected_features]
XTest_selected = XTest[selected_features]


# # RANDOM FOREST

# In[23]:


from sklearn.ensemble import RandomForestClassifier

# random forest model creation 
rfc = RandomForestClassifier()

test_dict = {'n_estimators' : [90, 100, 110],
             'random_state' : [30, 40, 50],
             'max_features' : ['log2', 'sqrt'],
             'max_depth' : [5, 7]}

rfc1 = GridSearchCV(estimator = rfc, param_grid = test_dict, scoring = 'f1_macro', n_jobs = -1)

rfc1.fit(XTrain_selected, yTrain) 
# predictions 
yPred = rfc1.predict(XTest_selected)


# In[24]:


rfc1.best_params_


# In[25]:


from sklearn.metrics import classification_report, confusion_matrix

print("\nConfusion Matrix :\n\n", confusion_matrix(yTest,yPred))
print("\n")
print("Classification Report :\n\n", classification_report(yTest, yPred))


# In[26]:


from sklearn.model_selection import KFold
model = RandomForestClassifier()
kfold_validation = KFold(10)


# In[27]:


from sklearn.model_selection import cross_val_score

results = cross_val_score(model, X , y, cv = kfold_validation)
print(results)
print(np.mean(results))


# # SUPPORT VECTOR MACHINE

# In[ ]:


from sklearn.svm import SVC

svm = SVC()

search_space = {'kernel':['linear','poly','rbf'],'C':[5, 2.5, 1.0],'gamma':['scale','auto'],'degree':[2,3,5]}

svm1 = GridSearchCV(estimator=svm,param_grid=search_space,scoring='f1_macro',n_jobs=-1)

svm1.fit(XTrain_selected,yTrain)

svm_pred = svm1.predict(XTest_selected)


# In[ ]:


svm1.best_params_


# In[ ]:


print("Confusion_Matrix\n", confusion_matrix(yTest,svm_pred))
print("\n")
print("report\n", classification_report(yTest,svm_pred,target_names = names ))


# # DECISION TREE CLASSIFIER

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

decisionTree = DecisionTreeClassifier()

test_dict_dt = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2']
}

decisionTree = GridSearchCV(
    estimator=decisionTree,
    param_grid=test_dict_dt,
    scoring='f1_macro', 
    n_jobs=-1
)

decisionTree.fit(XTrain_selected,yTrain)

decisionTree_pred = decisionTree.predict(XTest_selected)


# In[ ]:


decisionTree.best_params_


# In[ ]:


print("Confusion_Matrix\n", confusion_matrix(yTest, decisionTree_pred))
print("\n")
print("Report\n", classification_report(yTest, decisionTree_pred))


# # GAUSSIAN NAIVE BAYES

# In[ ]:


from sklearn.naive_bayes import GaussianNB

naive_bayes = GaussianNB()

test_dict_nb = {
    'var_smoothing': [1e-09, 1e-08, 1e-07, 1e-06, 1e-05]
}

naive_bayes = GridSearchCV(
    estimator=naive_bayes,
    param_grid=test_dict_nb,
    scoring='f1_macro',  # Use an appropriate scoring metric
    n_jobs=-1
)

naive_bayes.fit(XTrain_selected ,yTrain)

naive_bayes_pred = naive_bayes.predict(XTest_selected)


# In[ ]:


naive_bayes.best_params_


# In[ ]:


print("Confusion_Matrix\n", confusion_matrix(yTest, naive_bayes_pred))
print("\n")
print("Report\n", classification_report(yTest,naive_bayes_pred))


# # LOGISTIC REGRESSION

# In[ ]:


from sklearn.linear_model import LogisticRegression


LR1 = LogisticRegression()

search_dict = {'penalty' : ['l2', 'elasticnet', 'none'],
               'solver' : ['lbfgs','liblinear' , 'sag'],
              'C' : np.logspace(-4, 4, num =20),
              'fit_intercept': [True, False],
              'max_iter': [50, 100, 200],
              'class_weight': [None, 'balanced'],
              'random_state': [42]}

LR1= GridSearchCV(
    estimator=LR1,
    param_grid=search_dict,
    scoring='f1_macro', 
    n_jobs=-1
)



LR1.fit(XTrain_selected,yTrain)

LR_pred = LR1.predict(XTest_selected)


# In[ ]:


LR1.best_params_


# In[ ]:


print("Confusion_Matrix\n", confusion_matrix(yTest,LR_pred))
print("\n")
print("Report\n", classification_report(yTest,LR_pred))


# # KNN

# In[30]:


from sklearn.neighbors import KNeighborsClassifier

KNN = KNeighborsClassifier()

search_dict = {
    'n_neighbors': [3, 7],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size': [10, 20, 30],
    'p': [1, 2]
}

KNN = GridSearchCV(
    estimator=KNN,
    param_grid=search_dict,
    scoring='f1_macro',
    n_jobs=-1
)

KNN.fit(XTrain_selected, yTrain)

KNN_pred = KNN.predict(XTest_selected)


# In[ ]:


KNN.best_params_


# In[ ]:


print("Confusion_Matrix\n", confusion_matrix(yTest,KNN_pred))
print("\n")
print("Report\n", classification_report(yTest,KNN_pred))


# In[ ]:




