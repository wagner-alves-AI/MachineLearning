# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 05:30:22 2021

@author: wgnr2
"""

# Este exemplo carrega a base Wine da UCI, treina um classificador MLP
# usando holdout e outro usando validaÃ§Ã£o cruzada com 10 pastas. 

# Importa bibliotecas necessÃ¡rias 
import numpy as np
import urllib
from sklearn.neural_network import MLPClassifier
from sklearn import  model_selection
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Carregar base Cancer
from sklearn.datasets import load_iris
data = load_iris()
# Carrega arquivo como uma matriz
X, y = load_iris(return_X_y=True)


# EXEMPLO USANDO HOLDOUT
# Holdout -> dividindo a base em treinamento (70%) e teste (30%), estratificada
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.3, random_state=42, stratify=y)

# Treina o classificador
clfa = MLPClassifier(hidden_layer_sizes=(8), activation='logistic', max_iter=20000, solver='lbfgs', tol= 1e-10, verbose=True, early_stopping=True, validation_fraction=0.2)
clfa = clfa.fit(X_train, y_train)


# testa usando a base de testes
predicted=clfa.predict(X_test)

# calcula a acurÃ¡cia na base de teste
score=clfa.score(X_test, y_test)
print(score)


#f1-score
fscore_holdout = f1_score(y_test,predicted, average='micro')
print("F1-score: %.2f" % fscore_holdout)

# calcula a matriz de confusÃ£o
matrix = confusion_matrix(y_test, predicted)

# apresenta os resultados
print("Accuracy = %.2f " % score)
print("Confusion Matrix:")
print(matrix)

folds=10
f1 = model.selection.cross_val_score(clfa, X, y, cv=folds, scoring='f1_micro')

# EXEMPLO USANDO VALIDAÃ‡ÃƒO CRUZADA
clfb = MLPClassifier(hidden_layer_sizes=(100), activation='relu', max_iter=20000,solver='lbfgs', tol= 1e-10, verbose=True, early_stopping=True, validation_fraction=0.2)

clfb = clfb.fit(X_train, y_train)

predicted=clfb.predict(X_test)

score=clfb.score(X_test, y_test)
print(score)

# matriz de confusÃ£o da validaÃ§Ã£o cruzada modelo 2
Z = model_selection.cross_val_predict(clfb, X, y, cv=folds)
cm=confusion_matrix(y, Z)
print("Confusion Matrix:")
print(cm)

folds=10
result = model_selection.cross_val_score(clfb, X, y, cv=folds)
print("\nCross Validation Results %d folds:" % folds)
print("Mean Accuracy: %.2f" % result.mean())
print("Mean Std: %.2f" % result.std())

#F1-score
fscore = model_selection.cross_val_score(clfb, X, y, cv=folds,scoring='f1_micro')
print("\nCross Validation Results %d folds:" % folds)
print("Mean F1-score: %.2f" % fscore.mean())
print("Mean Std: %.2f" % result.std())


# matriz de confusÃ£o da validaÃ§Ã£o cruzada modelo 1
Z2 = model_selection.cross_val_predict(clfa, X, y, cv=folds)
cm2=confusion_matrix(y, Z)
print("Confusion Matrix:")
print(cm2)

folds=10
result2 = model_selection.cross_val_score(clfa, X, y, cv=folds)
print("\nCross Validation Results %d folds:" % folds)
print("Mean Accuracy: %.2f" % result.mean())
print("Mean Std: %.2f" % result.std())








