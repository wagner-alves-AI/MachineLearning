
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 19:04:45 2021

@author: wgnr2
"""

# Este exemplo carrega a base Wine da UCI, e avalia dois KNNs
# Um usando holdout e outro usando validaÃ§Ã£o cruzada com 10 pastas. 

# Importa bibliotecas necessÃ¡rias 
import numpy as np
import urllib.request
from sklearn.neighbors import KNeighborsClassifier
from sklearn import  model_selection
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import StratifiedShuffleSplit
# Carrega uma base de dados do UCI
# Exemplo carrega a base Wine
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data"
raw_data = urllib.request.urlopen(url)

# Carrega arquivo como uma matriz
dataset = np.loadtxt(raw_data, delimiter=",")

# Imprime quantide de instÃ¢ncias e atributos da base
print(dataset.shape)

# Coloca em X os 3 atributos de entrada e em y as classes
 
X = dataset[:,0:3]
y = dataset[:,3]

# EXEMPLO USANDO HOLDOUT
# Holdout -> dividindo a base em treinamento (70%) e teste (30%), estratificada
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.3, random_state=42, stratify=y)

# declara o classificador
clfa = KNeighborsClassifier(n_neighbors=3)

# treina o classificador
clfa = clfa.fit(X_train, y_train)

# testa usando a base de testes
predicted=clfa.predict(X_test)

# calcula a acuracia na base de teste
score=clfa.score(X_test, y_test)
score*100

# calcula a matriz de confusÃ£o
matrix = confusion_matrix(y_test, predicted)

# apresenta os resultados
print("Accuracy = %.2f " % score)
print("Confusion Matrix:")
print(matrix)

# Cross validation
clfb = KNeighborsClassifier(n_neighbors=3)
folds=10
result = model_selection.cross_val_score(clfb, X, y, cv=folds)
print("\nCross Validation Results %d folds:" % folds)
print("Mean accuracy: %.2f" % result.mean())
print("Mean Std: %.2f" % result.std())

pres = model_selection.cross_val_score(clfb, X, y, cv=folds, 
                                         scoring='precision')
print("\nCross Validation Results %d folds:" % folds)
print("Mean precision: %.2f" % pres.mean())
print("Mean Std: %.2f" % result.std())

fscore = model_selection.cross_val_score(clfb, X, y, cv=folds, 
                                         scoring='f1')
print("\nCross Validation Results %d folds:" % folds)
print("Mean F1-score: %.2f" % fscore.mean())
print("Mean Std: %.2f" % result.std())

# matriz de confusão da matriz de confusão
Z = model_selection.cross_val_predict(clfb, X, y, cv=folds)
cm=confusion_matrix(y, Z)
print("Confusion Matrix:")
print(cm)
