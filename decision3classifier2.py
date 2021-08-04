# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 23:37:39 2021

@author: wgnr2
"""

from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import sklearn.tree as tr
import matplotlib.pyplot as plt

X,y, targets, _, __, features = load_wine().values()

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.5,
                                                        random_state=1)

model = DecisionTreeClassifier(max_depth=2, criterion='entropy', 
                               min_samples_leaf=10, random_state=1)

model.fit(X_treino, y_treino)

fig, axs = plt.subplots(figsize=(12,6))
avr = tr.plot_tree(model)

    