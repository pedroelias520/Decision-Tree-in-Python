# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 08:19:53 2020

@author: Pedro Elias
"""

import pandas as pd 
dataset = pd.read_csv('adult.csv',encoding='latin-1',engine='python')
feature_cols = dataset.iloc[:, 0:14].columns

#Cria um array com todos os valores do dataset (sem as colunas)
X = dataset[feature_cols].values
#Criar um array com todos os atributos do tipo numérico
y =  dataset['income'].values

from sklearn.preprocessing import LabelEncoder 
enconder_x = LabelEncoder()

X[:,1] = enconder_x.fit_transform(X[:,1])

#Conversão de atributos categoricos para atributos numéricos
columns= [1,3,5,6,7,8,9,13] 
for i in columns: 
    X[:,i] = enconder_x.fit_transform(X[:,i])
    
enconder_y = LabelEncoder()
y = enconder_y.fit_transform(y)

#Escalonar os dados 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 0)

#A base de dados não está balanceada, tornando os dados tendenciosos, então deve balancea-la
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler()
X_train,y_train = rus.fit_sample(X_train,y_train)

#Criando árvore de decisão
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=3,criterion='entropy')
#Treino de modelo
clf = clf.fit(X_train, y_train)

from sklearn.tree import export_graphviz

export_graphviz(
        clf,
        out_file="decision_tree_new.dot",
        feature_names=feature_cols,
        class_names=['<=50','>50'],
        #class_names=base['class'],
        rounded=True,
        filled=True
    )


y_pred = clf.predict(X_test)
from sklearn import metrics

accuracy = metrics.accuracy_score(y_test,y_pred)
print(metrics.confusion_matrix(y_test, y_pred))
print(accuracy)



