# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 05:13:52 2021

@author: wgnr2
"""

# biblioteca da arvore de regressao
import sklearn
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

# criar um senoide
XR = np.sort(2*np.pi * np.random.rand(50, 1), axis=0)
yR = np.sin(XR).ravel()

# plota a senoide destacando as instancias
plt.plot(XR, yR, 'ro-')

# treinamento com niveis de profundidade 2 e 4
modelR2 = DecisionTreeRegressor(max_depth=2)
modelR4 = DecisionTreeRegressor(max_depth=4)

# treina o modelo de arvore de decisao
modelR2.fit(X=XR, y=yR)
modelR4.fit(X=XR, y=yR)

# previsao dos rotulos a partir dos atributos previsores
yR2_prevs = modelR2.predict(XR)
yR4_prevs = modelR4.predict(XR)

# plotagem dos resultados
fig, ax = plt.subplots(figsize=(12,8))
plt.plot(XR, yR, 'ro-', label='Original')
plt.plot(XR, yR2_prevs, 'bx-', label='Regressao max_depth: 2')
plt.plot(XR, yR4_prevs, 'gx-', label='Regressao max_depth: 4')
plt.legend()

# arvores de regressao
fig, ax = plt.subplots(2, 1, figsize=(16, 10))
arvoreR2 = sklearn.tree.plot_tree(modelR2, fontsize=8, filled=True, ax=ax[0])
arvoreR4 = sklearn.tree.plot_tree(modelR4, fontsize=8, filled=True, ax=ax[1])

ax[0].set_title('a) Profundidade: 2')
ax[0].set_title('b) Profundidade: 4')

# Amostragem e avaliacao

# Hold-out
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

Xh, yh = load_iris(return_X_y=True)

acc = []
tt = np.arange(0.1, 1, 0.1)
for t in tt:
    X_treino, X_teste, y_treino, y_teste = train_test_split(Xh, yh,
                                                            test_size=t,
                                                            random_state=4)
    modelh= DecisionTreeClassifier(random_state=4)
    modelh.fit(X=X_treino, y=y_treino)
    acc.append(accuracy_score(y_teste, modelh.predict(X_teste))*100)
    
acc_Df = pd.DataFrame(acc, index=tt*100, columns=['Acuracia'])

# CROSS-VALIDATION
from sklearn.model_selection import cross_val_score

Xc, yc = load_iris(return_X_y=True)

acc2 = []
acc2_df = pd.DataFrame()
for r in range(1, 5, 1):
    modelc = DecisionTreeClassifier(random_state=r)
    s = 'Acuracia R' + str(r)
    cross = cross_val_score(modelc,Xc,yc, cv=10, scoring='accuracy')*100
    cross = np.append(cross, sum(cross)/10)
    acc2_df[s] = cross
                                    
