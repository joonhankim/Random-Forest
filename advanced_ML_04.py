# -*- coding: utf-8 -*-
 
# DO NOT CHANGE
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt

def create_bootstrap(X,y,ratio):
    # X: input data matrix
    # ratio: sampling ratio
    # return one bootstraped dataset and indices of sub-sampled samples (newX,newy,ind)
    n = len(X)
    ind = np.random.choice(n,int(n*ratio),replace=True)
    newX= X[ind]
    newy=y[ind]
    return newX,newy,ind

def cal_oob_error(X,y,models,ind):
    # X: input data matrix
    # y: y: output target
    # models: list of trained models by different bootstraped sets
    # ind: list of indices of samples in different bootstraped sets
    
    arthur=np.zeros((len(X),len(models)))
    for i in range(len(X)):
        for j in range(len(models)):
            joker=np.in1d(ind[j],i)
            if np.sum(joker) != 0:
                arthur[i,j] = np.nan
            else:
                y_pred=models[j].predict(X[i].reshape(1,-1))
                arthur[i, j]=1-(y[i] == y_pred)
                
      
    return np.nanmean(arthur,axis=1)
    
    
def cal_var_importance(X,y,models,ind,oob_errors):
    # X: input data matrix
    # y: output target
    # models: list of trained models by different bootstraped sets
    # ind: list of indices of samples in different bootstraped sets
    # oob_errors: list of oob error of each sample
    # return variable importance
    v_i=[]
    for i in range(30):
        X_copy= X.copy()
        X_copy[:, i]= shuffle(X[:,i])
        new_err=cal_oob_error(X_copy,y,models,ind)
        v_i.append(np.mean(oob_errors-new_err))
    
    return v_i


def random_forest(X,y,n_estimators,ratio,params):
    # X: input data matrix
    # y: output target
    # n_estimators: the number of classifiers
    # ratio: sampling ratio for bootstraping
    # params: parameter setting for decision tree
    # return list of tree models trained by different bootstraped sets and list of indices of samples in different bootstraped sets
    # (models,ind_set)
    models=[]
    ind_set=[]
    for i in range(n_estimators):
        newX,newy,ind=create_bootstrap(X,y,ratio)
        ind_set.append(ind)
        clf=DecisionTreeClassifier(max_depth=params['max_depth'],min_samples_split=params['min_samples_split'],min_samples_leaf=params['min_samples_leaf'])
        clf=clf.fit(newX,newy)
        models.append(clf)
    return models,ind_set
    
data=datasets.load_breast_cancer()
X, y = shuffle(data.data, data.target, random_state=13)

params = {'max_depth': 4, 'min_samples_split': 0.1, 'min_samples_leaf':0.05}
n_estimators=500
ratio=1.0

models, ind_set = random_forest(X,y,n_estimators,ratio,params)
oob_errors=cal_oob_error(X,y,models,ind_set)
var_imp=cal_var_importance(X,y,models,ind_set,oob_errors)


nfeature=len(X[0])
plt.barh(np.arange(nfeature),var_imp/sum(var_imp))
plt.yticks(np.arange(nfeature) + 0.35 / 2, data.feature_names)






